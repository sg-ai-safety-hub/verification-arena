#pragma once
// Fused sloth64 decode-on-load GEMM, v4: three rings, decoded W tiles written straight into every cluster
// CTA's shared memory (sm_90a).
//
// y = x @ decode(W)^T, W [N, K] sloth64-encoded (see slothgemm.cu). Output tile per CTA: 256 x 128. A cluster of
// CM CTAs (1, 2, 4) covers CM * 256 rows of one N tile; each CTA decodes 128 / CM rows of the B tile once and
// stores the decoded words into the B buffer of every CTA in the cluster (distributed shared memory), so every
// W word is squared once per CM * 256 rows and plaintext W exists only in shared memory.
//
// v3 pushed decoded slices with cp.async.bulk and kept A and B in one 4-stage ring; its cluster variants were
// 1.5-3x slower even without decode (copy latency in a short pipeline). v4 decouples the latencies:
//   A ring   SA stages x 32 KiB   x tile (L2-resident, short latency)
//   E ring   ST stages x 16/CM KiB  encoded B slice staging (HBM latency)
//   B ring   SB stages x 16 KiB   decoded B tile, filled by generic stores from all CTAs of the cluster
// so the encoded loads run ST ahead of the decoders and the decoders up to SB ahead of the consumers.
//
// Threads (384, 168 regs): WG0, WG1 consumers (2 m64 slabs each, 128 accumulators); warps 8..10 decoders (96);
// warp 11: lane 0 A producer, lane 1 E producer.
// Barriers: afull/aempty (A ring, local), efull/eempty (E ring, local), ready[sb] (CM arrivals: one per CTA when
// its slice of the decoded tile is stored and fenced), bempty[sb] (8 * CM: every consumer warp of the cluster).

#include <cooperative_groups.h>
#include "slothgemm.cu"

namespace slg4 {

using namespace encgemm;
using slgemm::dec1;

constexpr int BN = 128, BM = 256, NDT = 96;
constexpr int A_BYTES = BM * BK * 2;   // 32768
constexpr int B_BYTES = BN * BK * 2;   // 16384
template <int CM, int SA, int SB> struct Lay {
  static constexpr int ST = 2 * CM;                      // E ring: 32 KiB in total
  static constexpr int SL_ROWS = BN / CM;
  static constexpr int SL_BYTES = SL_ROWS * 128;
  static constexpr int A_OFF = 0;
  static constexpr int B_OFF = SA * A_BYTES;
  static constexpr int E_OFF = B_OFF + SB * B_BYTES;
  static constexpr int BAR_OFF = E_OFF + ST * SL_BYTES;
  static constexpr int NBAR = 2 * SA + 2 * ST + 2 * SB;
  static constexpr int TOTAL = BAR_OFF + NBAR * 8 + 1024;
  static_assert(TOTAL <= 232448, "shared memory over the H100 per-block limit");
};

__device__ __forceinline__ uint32_t mapa32(uint32_t local, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local), "r"(rank));
  return r;
}
__device__ __forceinline__ void mbar_arrive_remote(uint32_t cluster_addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(cluster_addr) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
}

template <int CM, int SA, int SB, int DEC>
__global__ void __launch_bounds__(384, 1)
sl4_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
           __nv_bfloat16* __restrict__ C, int M, int N, int K, int ldc, uint64_t seed, uint64_t i0) {
  using L = Lay<CM, SA, SB>;
  constexpr int ST = L::ST, SL_ROWS = L::SL_ROWS, SL_BYTES = L::SL_BYTES;
  constexpr int CHUNKS = SL_ROWS * 8;              // 16-B chunks in one CTA's slice
  constexpr int PERMAX = (CHUNKS + NDT - 1) / NDT;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* aring = smem + L::A_OFF;
  uint8_t* bring = smem + L::B_OFF;
  uint8_t* ering = smem + L::E_OFF;
  uint64_t* afull = reinterpret_cast<uint64_t*>(smem + L::BAR_OFF);
  uint64_t* aempty = afull + SA;
  uint64_t* efull = aempty + SA;
  uint64_t* eempty = efull + ST;
  uint64_t* ready = eempty + ST;
  uint64_t* bempty = ready + SB;

  const int rank = blockIdx.x % CM, cl = blockIdx.x / CM, ncl = gridDim.x / CM;
  const int num_g = (M + BM * CM - 1) / (BM * CM), num_n = (N + BN - 1) / BN;
  const int units = num_g * num_n;
  const int KT = K / BK;
  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;

  if (tid == 0) {
    for (int s = 0; s < SA; ++s) { mbar_init(&afull[s], 1); mbar_init(&aempty[s], 8); }
    for (int s = 0; s < ST; ++s) { mbar_init(&efull[s], 1); mbar_init(&eempty[s], NDT); }
    for (int s = 0; s < SB; ++s) { mbar_init(&ready[s], CM); mbar_init(&bempty[s], 8 * CM); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if constexpr (CM > 1) cluster_sync();            // every CTA's barriers exist before any remote arrive / store

  if (warp == 11) {
    if (lane == 0) {
      // ---------------- A producer
      int it = 0;
      for (int u = cl; u < units; u += ncl) {
        const int m0 = ((u % num_g) * CM + rank) * BM;
        for (int kt = 0; kt < KT; ++kt, ++it) {
          const int s = it % SA;
          if (it >= SA) mbar_wait(&aempty[s], ((it / SA) & 1) ^ 1);
          mbar_expect_tx(&afull[s], A_BYTES);
          tma_load_2d(aring + s * A_BYTES, &tmA, &afull[s], kt * BK, m0);
        }
      }
    } else if (lane == 1) {
      // ---------------- E producer (encoded B slice of this CTA)
      int it = 0;
      for (int u = cl; u < units; u += ncl) {
        const int n0 = (u / num_g) * BN;
        for (int kt = 0; kt < KT; ++kt, ++it) {
          const int s = it % ST;
          if (it >= ST) mbar_wait(&eempty[s], ((it / ST) & 1) ^ 1);
          mbar_expect_tx(&efull[s], SL_BYTES);
          tma_load_2d(ering + s * SL_BYTES, &tmB, &efull[s], kt * BK, n0 + rank * SL_ROWS);
        }
      }
    }
  } else if (warp >= 8) {
    // ---------------- decoders
    const int td = tid - 256;
    uint8_t* bdst[CM];
    if constexpr (CM > 1) {
      auto cluster = cooperative_groups::this_cluster();
#pragma unroll
      for (int p = 0; p < CM; ++p) bdst[p] = cluster.map_shared_rank(bring, p);
    } else {
      bdst[0] = bring;
    }
    int it = 0;
    for (int u = cl; u < units; u += ncl) {
      const int n0 = (u / num_g) * BN;
      for (int kt = 0; kt < KT; ++kt, ++it) {
        const int se = it % ST, sb = it % SB;
        mbar_wait(&efull[se], (it / ST) & 1);
        if (it >= SB) mbar_wait(&bempty[sb], ((it / SB) & 1) ^ 1);
        const uint8_t* et = ering + se * SL_BYTES;
#pragma unroll
        for (int j0 = 0; j0 < PERMAX; j0 += 4) {
          ulonglong2 v[4];
          uint32_t off[4];
          uint64_t wi[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int q = td + NDT * (j0 + j);
            if (j0 + j < PERMAX && q < CHUNKS) {
              const int lr = q >> 3, c = q & 7;
              const int row = rank * SL_ROWS + lr;
              off[j] = uint32_t(row * 128 + ((c ^ (row & 7)) << 4));
              wi[j] = (uint64_t)(((int64_t)(n0 + row) * K + kt * BK + c * 8) >> 2) + i0;
              v[j] = *reinterpret_cast<const ulonglong2*>(et + lr * 128 + ((c ^ (lr & 7)) << 4));
            }
          }
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int q = td + NDT * (j0 + j);
            if (j0 + j < PERMAX && q < CHUNKS) {
              ulonglong2 o = v[j];
              if (DEC == 1) {
                o.x = dec1(v[j].x, seed, wi[j]);
                o.y = dec1(v[j].y, seed, wi[j] + 1);
              }
#pragma unroll
              for (int p = 0; p < CM; ++p)
                *reinterpret_cast<ulonglong2*>(bdst[p] + sb * B_BYTES + off[j]) = o;
            }
          }
        }
        if constexpr (CM > 1) asm volatile("fence.proxy.async.shared::cluster;" ::: "memory");
        else asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbar_arrive(&eempty[se]);
        asm volatile("bar.sync 1, 96;" ::: "memory");     // this CTA's whole slice is stored and fenced
        if (td == 0) {
          const uint32_t r = smem_u32(&ready[sb]);
          if constexpr (CM > 1) {
#pragma unroll
            for (int p = 0; p < CM; ++p) mbar_arrive_remote(mapa32(r, p));
          } else {
            mbar_arrive(&ready[sb]);
          }
        }
      }
    }
  } else {
    // ---------------- consumers
    const int wg = warp / 4, w = warp % 4, l = lane;
    auto release = [&](int itp) {
      if (l != 0) return;
      mbar_arrive(&aempty[itp % SA]);
      if constexpr (CM > 1) {
        const uint32_t e = smem_u32(&bempty[itp % SB]);
#pragma unroll
        for (int p = 0; p < CM; ++p) mbar_arrive_remote(mapa32(e, p));
      } else {
        mbar_arrive(&bempty[itp % SB]);
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
        const int sa = it % SA, sb = it % SB;
        mbar_wait(&afull[sa], (it / SA) & 1);
        mbar_wait(&ready[sb], (it / SB) & 1);
        const uint8_t* at = aring + sa * A_BYTES;
        const uint8_t* btile = bring + sb * B_BYTES;
        fence_regs(acc[0]); fence_regs(acc[1]);
        wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk)
#pragma unroll
          for (int h = 0; h < 2; ++h)
            mma_k16<BN>(acc[h], make_desc(at + (wg * 2 + h) * 64 * 128 + kk * 32), make_desc(btile + kk * 32));
        wgmma_commit();
        fence_regs(acc[0]); fence_regs(acc[1]);
        wgmma_wait<1>();
        fence_regs(acc[0]); fence_regs(acc[1]);
        if (kt > 0) release(it - 1);
      }
      wgmma_wait<0>();
      fence_regs(acc[0]); fence_regs(acc[1]);
      release(it - 1);

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
  // no CTA may exit while a peer can still arrive on its barriers or store into its shared memory
  if constexpr (CM > 1) cluster_sync();
}

}  // namespace slg4
