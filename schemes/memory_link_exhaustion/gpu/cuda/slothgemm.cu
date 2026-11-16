#pragma once
// Fused sloth64 decode-on-load GEMM for Hopper (sm_90a), phase-1 prototype (timing; bit-exactness not targeted
// yet, though the per-element k16 wgmma chain is the same as encgemm.cu's): y = x @ decode(W)^T.
//
// W [N, K] bf16 row-major, sloth64-encoded: word i (elements 4i..4i+3 of the matrix, i counted from i0) is
//   x = S(y) ^ splitmix64(seed ^ (i * 0x9E3779B97F4A7C15)),  S(r) = r^2 (r even) | p - r^2 (r odd) mod p,
// p = 2^64 - 189, words y >= p unchanged (encodings/sloth64.py).
//
// The decoded W tile lives only in shared memory. To decode each W element once per CTA tile, the M tile is
// large: BM = 64 * NC * SL rows (NC consumer warpgroups, each SL m64 slabs), so at M <= BM every W element is squared
// exactly once per GEMM; at larger M once per BM rows.
//
// Warp roles (warpgroup-aligned):
//   WG 0 .. NC-1          consumers: async wgmma from shared memory (A: TMA'd x tile, B: decoded W tile)
//   WG NC .. NC+NDEC-1    decoders: square + unmask the B tile in place, round-robin over stages (decoder d
//                         takes stages d, d + NDEC, ...), so NDEC tiles are decoded concurrently
//   last warp             TMA producer (S-stage ring)
// Barriers per stage: full (TMA tx), decd (the 128 threads of that stage's decoder), empty (all consumers).
//
// DEC: 0 = no decode (plain W; decoders only forward the stage), 1 = sloth64 decode.

#include "encgemm.cu"

namespace slgemm {

using namespace encgemm;

__device__ __forceinline__ uint64_t sq189(uint64_t a) {
  uint64_t lo = a * a, hi = __umul64hi(a, a);
  uint64_t t1 = (uint64_t)(uint32_t)hi * 189ull;
  uint64_t t2 = (hi >> 32) * 189ull + (t1 >> 32);
  uint64_t hc_hi = t2 >> 32, hc_lo = (t2 << 32) | (t1 & 0xffffffffull);
  uint64_t s = lo + hc_lo;
  uint64_t h = hc_hi + (s < lo ? 1ull : 0ull);
  uint64_t s2 = s + h * 189ull;
  if (s2 < s) s2 += 189ull;
  const uint64_t pm = 0xFFFFFFFFFFFFFF43ull;
  return s2 >= pm ? s2 - pm : s2;
}
__device__ __forceinline__ uint64_t smix(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__device__ __forceinline__ uint64_t dec1(uint64_t y, uint64_t seed, uint64_t i) {
  const uint64_t pm = 0xFFFFFFFFFFFFFF43ull;
  uint64_t s = sq189(y);
  uint64_t neg = s == 0 ? s : pm - s;
  uint64_t x = (y & 1) ? neg : s;
  if (y >= pm) x = y;
  return x ^ smix(seed ^ (i * 0x9E3779B97F4A7C15ull));
}

template <int NC, int NDEC> struct Cfg {
  static constexpr int THREADS = 128 * (NC + NDEC) + 32;   // + one producer warp
};

// SL m64 slabs per consumer warpgroup: BM = 64 * NC * SL
template <int BN, int NC, int SL, int S>
struct Lay {
  static constexpr int BM = 64 * NC * SL;
  static constexpr int A_BYTES = BM * BK * 2;
  static constexpr int B_BYTES = BN * BK * 2;
  static constexpr int STAGE = A_BYTES + B_BYTES;
  static constexpr int TOTAL = S * STAGE + 3 * S * 8 + S * 16 + 1024;
  static_assert(TOTAL <= 232448, "shared memory over the H100 per-block limit (227 KiB)");
};

template <int BN, int NC, int SL, int S, int NDEC, int DEC>
__global__ void __launch_bounds__(Cfg<NC, NDEC>::THREADS, 1)
sl_gemm_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
               __nv_bfloat16* __restrict__ C, int M, int N, int K, int ldc, uint64_t seed, uint64_t i0,
               int group_m, const int* gate = nullptr, int* ctr = nullptr, int yield = 0) {
  using L = Lay<BN, NC, SL, S>;
  constexpr int BM = L::BM;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + S * L::STAGE);
  uint64_t* decd = full + S;
  uint64_t* empty = decd + S;
  int* tinfo = reinterpret_cast<int*>(empty + S);   // per stage: m0, n0, kt (kt < 0: no more work, exit)

  const int num_pid_m = (M + BM - 1) / BM, num_pid_n = (N + BN - 1) / BN;
  const int num_tiles = num_pid_m * num_pid_n;
  const int in_group = group_m * num_pid_n;
  const int KT = K / BK;
  auto tile_mn = [&](int tile, int& m0, int& n0) {
    const int first_m = (tile / in_group) * group_m;
    const int gsz = min(num_pid_m - first_m, group_m);
    m0 = (first_m + (tile % gsz)) * BM;
    n0 = ((tile % in_group) / gsz) * BN;
  };

  const int tid = threadIdx.x;
  const int wg = tid / 128;
  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&decd[s], 128);
      mbar_init(&empty[s], NC * 128);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (tid >= 128 * (NC + NDEC)) {
    // ---------------- producer warp: TMA loads, S-stage ring
    if (tid == 128 * (NC + NDEC)) {
      // Tile schedule. ctr == nullptr: static (tile = blockIdx + j * grid); with a gate, no new output tile starts
      // while the device gate is raised (the CTA waits in place, as VARENA_MM_GATE on stock's matmul).
      // ctr != nullptr: tiles are claimed from a device counter (as actenc2._launch_mm's pre-emptible matmul).
      // With yield, a CTA that sees the gate raised claims no further tile and exits, so the SM is free for the
      // challenge kernels; the host relaunches on the same counter after a gate-wait kernel. Without yield the
      // gate is ignored (final resume). The stage's (m0, n0, kt) is published in shared memory before its TMA
      // loads; kt = -1 on the NDEC stages after the last tile tells every decoder and the consumers to exit.
      int it = 0;
      const volatile int* vg = reinterpret_cast<const volatile int*>(gate);
      for (int j = 0;; ++j) {
        int tile;
        if (ctr != nullptr) {
          if (yield && vg != nullptr && *vg != 0) break;
          tile = atomicAdd(ctr, 1);
        } else {
          if (vg != nullptr) while (*vg != 0) {}
          tile = blockIdx.x + j * gridDim.x;
        }
        if (tile >= num_tiles) break;
        int m0, n0;
        tile_mn(tile, m0, n0);
        for (int kt = 0; kt < KT; ++kt, ++it) {
          const int s = it % S;
          if (it >= S) mbar_wait(&empty[s], ((it / S) & 1) ^ 1);
          uint8_t* st = smem + s * L::STAGE;
          tinfo[4 * s] = m0; tinfo[4 * s + 1] = n0; tinfo[4 * s + 2] = kt;
          mbar_expect_tx(&full[s], L::STAGE);
          tma_load_2d(st, &tmA, &full[s], kt * BK, m0);
          tma_load_2d(st + L::A_BYTES, &tmB, &full[s], kt * BK, n0);
        }
      }
      for (int e = 0; e < NDEC; ++e, ++it) {
        const int s = it % S;
        if (it >= S) mbar_wait(&empty[s], ((it / S) & 1) ^ 1);
        tinfo[4 * s + 2] = -1;
        mbar_arrive(&full[s]);
      }
    }
    return;
  }

  if (wg >= NC) {
    // ---------------- decoders: decoder warpgroup d decodes the whole B tile of stages it = d, d + NDEC, ...
    // (NDEC stages in flight, no cross-warpgroup sync)
    const int d = wg - NC;
    const int t = tid - 128 * wg;
    constexpr int CHUNKS = BN * 8;                 // 16-B chunks (2 words) per B tile
    constexpr int PER = CHUNKS / 128;
    static_assert(PER * 128 == CHUNKS, "BN * 8 must be a multiple of 128");
    // each stage must always go to the same decoder: otherwise a decoder can wait two phases ahead on full[s],
    // and a parity wait cannot tell phase k from k + 2 (it returns early on stale data)
    static_assert(S % NDEC == 0, "stages must be a multiple of the decoder count");
    for (int it = d;; it += NDEC) {
      {
        const int s = it % S;
        mbar_wait(&full[s], (it / S) & 1);
        const int n0 = tinfo[4 * s + 1], kt = tinfo[4 * s + 2];
        if (kt < 0) {
          mbar_arrive(&decd[s]);
          break;
        }
        if (DEC == 1) {
          uint8_t* bt = smem + s * L::STAGE + L::A_BYTES;
          ulonglong2 v[PER];
          uint32_t off[PER];
#pragma unroll
          for (int j = 0; j < PER; ++j) {
            const int q = t + 128 * j;
            const int row = q >> 3, c = q & 7;
            off[j] = uint32_t(row * 128 + ((c ^ (row & 7)) << 4));
            v[j] = *reinterpret_cast<const ulonglong2*>(bt + off[j]);
          }
          // word index of chunk j: ((n0 + row) * K + kt * 64 + c * 8) / 4, row = t / 8 + 16 j
          const uint64_t wbase = (uint64_t)((((int64_t)(n0 + (t >> 3)) * K + kt * BK + (t & 7) * 8) >> 2)) + i0;
          const uint64_t wstep = (uint64_t)(4 * K);  // 16 rows * K / 4
#pragma unroll
          for (int j = 0; j < PER; ++j) {
            const uint64_t wi = wbase + wstep * j;
            ulonglong2 o;
            o.x = dec1(v[j].x, seed, wi);
            o.y = dec1(v[j].y, seed, wi + 1);
            *reinterpret_cast<ulonglong2*>(bt + off[j]) = o;
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        mbar_arrive(&decd[s]);
      }
    }
    return;
  }

  // ---------------- consumers
  const int w = (tid % 128) / 32, l = tid % 32;
  for (int it = 0;;) {
    int m0, n0;
    {
      const int s = it % S;
      mbar_wait(&decd[s], (it / S) & 1);
      if (tinfo[4 * s + 2] < 0) break;
      m0 = tinfo[4 * s]; n0 = tinfo[4 * s + 1];
    }
    float acc[SL][BN / 2];
#pragma unroll
    for (int u = 0; u < SL; ++u)
#pragma unroll
      for (int i = 0; i < BN / 2; ++i) acc[u][i] = 0.f;
    for (int kt = 0; kt < KT; ++kt, ++it) {
      const int s = it % S;
      mbar_wait(&decd[s], (it / S) & 1);
      uint8_t* st = smem + s * L::STAGE;
      const uint8_t* btile = st + L::A_BYTES;
#pragma unroll
      for (int u = 0; u < SL; ++u) fence_regs(acc[u]);
      wgmma_fence();
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk)
#pragma unroll
        for (int u = 0; u < SL; ++u)
          mma_k16<BN>(acc[u], make_desc(st + (wg * SL + u) * 64 * 128 + kk * 32), make_desc(btile + kk * 32));
      wgmma_commit();
#pragma unroll
      for (int u = 0; u < SL; ++u) fence_regs(acc[u]);
      wgmma_wait<1>();
#pragma unroll
      for (int u = 0; u < SL; ++u) fence_regs(acc[u]);
      if (kt > 0) mbar_arrive(&empty[(it - 1) % S]);
    }
    wgmma_wait<0>();
#pragma unroll
    for (int u = 0; u < SL; ++u) fence_regs(acc[u]);
    mbar_arrive(&empty[(it - 1) % S]);

#pragma unroll
    for (int u = 0; u < SL; ++u) {
      const int rbase = m0 + (wg * SL + u) * 64 + w * 16 + l / 4;
#pragma unroll
      for (int i = 0; i < BN / 8; ++i) {
        const int col = n0 + 8 * i + 2 * (l % 4);
        if (col >= N) continue;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = rbase + 8 * h;
          if (row >= M) continue;
          __nv_bfloat162 o = __floats2bfloat162_rn(acc[u][4 * i + 2 * h], acc[u][4 * i + 2 * h + 1]);
          *reinterpret_cast<__nv_bfloat162*>(C + size_t(row) * ldc + col) = o;
        }
      }
    }
  }
}

// Two-pass reference decoder (the best standalone CUDA decode from tests/bench_wdec_kernels.py): dst = decode(src).
template <int V>
__global__ void sl_decode_kernel(const ulonglong2* __restrict__ src, ulonglong2* __restrict__ dst, int64_t n2,
                                 uint64_t seed, uint64_t i0) {
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; j < n2; j += stride * V) {
    ulonglong2 v[V];
#pragma unroll
    for (int u = 0; u < V; u++) if (j + u * stride < n2) v[u] = src[j + u * stride];
#pragma unroll
    for (int u = 0; u < V; u++) if (j + u * stride < n2) {
      uint64_t wd = (uint64_t)(j + u * stride) * 2 + i0;
      ulonglong2 o; o.x = dec1(v[u].x, seed, wd); o.y = dec1(v[u].y, seed, wd + 1);
      dst[j + u * stride] = o;
    }
  }
}

}  // namespace slgemm

namespace slgemm {
// Blocks its stream while the challenge gate is raised (between a yielding launch and its resume).
__global__ void sl_gate_wait_kernel(const int* gate) {
  const volatile int* g = reinterpret_cast<const volatile int*>(gate);
  while (*g != 0) {}
}
}  // namespace slgemm
