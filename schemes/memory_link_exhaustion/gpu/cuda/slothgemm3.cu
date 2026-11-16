#pragma once
// Fused sloth64 decode-on-load GEMM, v3: decoded W tiles shared across a CTA cluster (sm_90a).
//
// y = x @ decode(W)^T, W [N, K] sloth64-encoded (see slothgemm.cu). Output tile per CTA: 256 x 128 (BM x BN).
// A cluster of CM CTAs (CM = 1, 2, 4) covers CM * 256 rows of one 128-column N tile. Each CTA TMA-loads only
// its 128 / CM rows of the encoded B tile, decodes them once in shared memory, and pushes the decoded slice to
// its peers' B buffers with a bulk shared-to-shared copy (cp.async.bulk over distributed shared memory). So
// every W word is squared once per CM * 256 rows, and plaintext W exists only in shared memory.
//
// Threads (384, 168 registers): WG0, WG1 consumers (each 2 m64 slabs = 128 rows, 128 accumulators);
// warps 8..10 decoders (96 threads); warp 11 lane 0 TMA producer.
// Per-stage barriers in every CTA:
//   full[s]   own TMA bytes (A tile + own B slice)
//   ready[s]  decoded B tile complete: 96 local decoder arrivals (+ 1 arrive.expect_tx for the peers' slices)
//   empty[s]  stage free: one arrival per consumer warp of EVERY CTA in the cluster (8 * CM), because peers
//             write into this CTA's B buffer
// All CTAs of a cluster run the same (unit, k) sequence, so stage indices and phases line up across CTAs.
// Per-element arithmetic is the same k16 wgmma chain as encgemm.cu / stock's persistent matmul.

#include "slothgemm.cu"

namespace slg3 {

using namespace encgemm;
using slgemm::dec1;

constexpr int BN = 128, BM = 256, NDT = 96;
constexpr int A_BYTES = BM * BK * 2;   // 32768
constexpr int B_BYTES = BN * BK * 2;   // 16384
constexpr int STAGE = A_BYTES + B_BYTES;
template <int S> struct Lay {
  static constexpr int TOTAL = S * STAGE + 3 * S * 8 + 1024;
  static_assert(TOTAL <= 232448, "shared memory over the H100 per-block limit");
};

__device__ __forceinline__ uint32_t mapa(uint32_t local, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local), "r"(rank));
  return r;
}
__device__ __forceinline__ void mbar_arrive_remote(uint32_t cluster_addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(cluster_addr) : "memory");
}
__device__ __forceinline__ void bulk_copy_to_peer(uint32_t dst_cluster, uint32_t src_local, uint32_t bytes,
                                                  uint32_t bar_cluster) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst_cluster), "r"(src_local), "r"(bytes), "r"(bar_cluster) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
}

template <int CM, int S, int DEC>
__global__ void __launch_bounds__(384, 1)
sl3_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
           __nv_bfloat16* __restrict__ C, int M, int N, int K, int ldc, uint64_t seed, uint64_t i0) {
  constexpr int SL_ROWS = BN / CM;                 // B rows decoded by one CTA
  constexpr int SL_BYTES = SL_ROWS * 128;
  constexpr int CHUNKS = SL_ROWS * 8;              // 16-B chunks in the slice
  constexpr int PERMAX = (CHUNKS + NDT - 1) / NDT;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + S * STAGE);
  uint64_t* ready = full + S;
  uint64_t* empty = ready + S;

  const int rank = blockIdx.x % CM, cl = blockIdx.x / CM, ncl = gridDim.x / CM;
  const int num_g = (M + BM * CM - 1) / (BM * CM), num_n = (N + BN - 1) / BN;
  const int units = num_g * num_n;
  const int KT = K / BK;
  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;

  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&ready[s], NDT + (CM > 1 ? 1 : 0));
      mbar_init(&empty[s], 8 * CM);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if constexpr (CM > 1) cluster_sync();            // every CTA's barriers exist before any remote arrive / copy

  if (warp == 11) {
    // ---------------- producer
    if (lane == 0) {
      int it = 0;
      for (int u = cl; u < units; u += ncl) {
        const int m0 = ((u % num_g) * CM + rank) * BM, n0 = (u / num_g) * BN;
        for (int kt = 0; kt < KT; ++kt, ++it) {
          const int s = it % S;
          if (it >= S) mbar_wait(&empty[s], ((it / S) & 1) ^ 1);
          uint8_t* st = smem + s * STAGE;
          if constexpr (CM > 1) mbar_expect_tx(&ready[s], (CM - 1) * SL_BYTES);
          mbar_expect_tx(&full[s], A_BYTES + SL_BYTES);
          tma_load_2d(st, &tmA, &full[s], kt * BK, m0);
          tma_load_2d(st + A_BYTES + rank * SL_BYTES, &tmB, &full[s], kt * BK, n0 + rank * SL_ROWS);
        }
      }
    }
  } else if (warp >= 8) {
    // ---------------- decoders
    const int td = tid - 256;
    int it = 0;
    for (int u = cl; u < units; u += ncl) {
      const int n0 = (u / num_g) * BN;
      for (int kt = 0; kt < KT; ++kt, ++it) {
        const int s = it % S;
        mbar_wait(&full[s], (it / S) & 1);
        uint8_t* bt = smem + s * STAGE + A_BYTES;
        if (DEC == 1) {
#pragma unroll
          for (int j0 = 0; j0 < PERMAX; j0 += 4) {
            ulonglong2 v[4];
            uint32_t off[4];
            uint64_t wi[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const int q = td + NDT * (j0 + j);
              if (j0 + j < PERMAX && q < CHUNKS) {
                const int row = rank * SL_ROWS + (q >> 3), c = q & 7;
                off[j] = uint32_t(row * 128 + ((c ^ (row & 7)) << 4));
                wi[j] = (uint64_t)(((int64_t)(n0 + row) * K + kt * BK + c * 8) >> 2) + i0;
                v[j] = *reinterpret_cast<const ulonglong2*>(bt + off[j]);
              }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const int q = td + NDT * (j0 + j);
              if (j0 + j < PERMAX && q < CHUNKS) {
                ulonglong2 o;
                o.x = dec1(v[j].x, seed, wi[j]);
                o.y = dec1(v[j].y, seed, wi[j] + 1);
                *reinterpret_cast<ulonglong2*>(bt + off[j]) = o;
              }
            }
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        if constexpr (CM > 1) {
          asm volatile("bar.sync 1, 96;" ::: "memory");   // whole slice decoded before it is copied out
          if (td == 0) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            const uint32_t src = smem_u32(bt + rank * SL_BYTES), bar = smem_u32(&ready[s]);
#pragma unroll
            for (int p = 0; p < CM; ++p)
              if (p != rank) bulk_copy_to_peer(mapa(src, p), src, SL_BYTES, mapa(bar, p));
          }
        }
        mbar_arrive(&ready[s]);
      }
    }
  } else {
    // ---------------- consumers
    const int wg = warp / 4, w = warp % 4, l = lane;
    auto release = [&](int s) {
      if (l != 0) return;
      if constexpr (CM > 1) {
        const uint32_t e = smem_u32(&empty[s]);
#pragma unroll
        for (int p = 0; p < CM; ++p) mbar_arrive_remote(mapa(e, p));
      } else {
        mbar_arrive(&empty[s]);
      }
    };
    int it = 0;
    for (int u = cl; u < units; u += ncl) {
      const int m0 = ((u % num_g) * CM + rank) * BM, n0 = (u / num_g) * BN;
      float acc[2][BN / 2];
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int i = 0; i < BN / 2; ++i) acc[h][i] = 0.f;
      for (int kt = 0; kt < KT; ++kt, ++it) {
        const int s = it % S;
        mbar_wait(&full[s], (it / S) & 1);
        mbar_wait(&ready[s], (it / S) & 1);
        uint8_t* st = smem + s * STAGE;
        const uint8_t* btile = st + A_BYTES;
        fence_regs(acc[0]); fence_regs(acc[1]);
        wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk)
#pragma unroll
          for (int h = 0; h < 2; ++h)
            mma_k16<BN>(acc[h], make_desc(st + (wg * 2 + h) * 64 * 128 + kk * 32), make_desc(btile + kk * 32));
        wgmma_commit();
        fence_regs(acc[0]); fence_regs(acc[1]);
        wgmma_wait<1>();
        fence_regs(acc[0]); fence_regs(acc[1]);
        if (kt > 0) release((it - 1) % S);
      }
      wgmma_wait<0>();
      fence_regs(acc[0]); fence_regs(acc[1]);
      release((it - 1) % S);

#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int rbase = m0 + (wg * 2 + h) * 64 + w * 16 + l / 4;
#pragma unroll
        for (int i = 0; i < BN / 8; ++i) {
          const int col = n0 + 8 * i + 2 * (l % 4);
          if (col >= N) continue;
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int row = rbase + 8 * hh;
            if (row >= M) continue;
            __nv_bfloat162 o = __floats2bfloat162_rn(acc[h][4 * i + 2 * hh], acc[h][4 * i + 2 * hh + 1]);
            *reinterpret_cast<__nv_bfloat162*>(C + size_t(row) * ldc + col) = o;
          }
        }
      }
    }
  }
  // no CTA may exit while a peer can still arrive on its barriers or copy into its shared memory
  if constexpr (CM > 1) cluster_sync();
}

}  // namespace slg3
