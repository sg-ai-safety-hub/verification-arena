#pragma once
// Decode-on-load GEMM for Hopper (sm_90a): y = x @ decode(W)^T (+ bias).
//
// x [M, K] bf16 (row-major, any row stride), W [N, K] bf16 row-major, encoded with the
// separable weight keystream ks(n, k) = low16(fmix(n ^ key) ^ fmix(fmix(k ^ key) ^ WCOL)).
// fp32 accumulate, bf16 output (RNE), bias added in fp32 before rounding.
//
// Bit-exactness with vLLM's batch-invariant Triton matmul_kernel_persistent on Hopper:
// Triton lowers every tl.dot(a[BMx64], b[64xBN], acc) into four wgmma m64nXk16 in increasing
// k order, accumulating into one fp32 register chain started from zero, so each output element
// is the same chain of k16 tensor-core steps over k = 0, 16, 32, ... K-16. This kernel issues
// exactly that chain per element (same instruction family, same k order, acc starts at +0.0,
// scale-d = 1), so the result depends only on what the hardware does per element. Tile sizes,
// the N of the instruction and the grid are free (verified by the GEMM-level test, not assumed).
//
// Warp specialisation (one CTA per BM x BN output tile):
//   WG0, warp 0, lane 0 : TMA producer (A and B tiles, 128B swizzle) into an S-stage ring
//   WG1 (DEC only)      : decoder, XORs the B tile in shared memory, fence.proxy.async, arrive
//   WG2.. (NC of them)  : consumers, async wgmma from shared memory, release stages
// Barriers per stage: full (TMA tx), dec (128 decoder arrivals), empty (consumer arrivals).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "wgmma.h"
#include "bfly.cuh"

#define WCOL 0x7F4A7C15u

namespace encgemm {

__device__ __forceinline__ uint32_t fmix(uint32_t x) {
  x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16; return x;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}"
               :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "@!p bra WAIT_%=;\n}" :: "r"(smem_u32(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* tm, uint64_t* bar, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%3, %4}], [%2];"
               :: "r"(smem_u32(dst)), "l"(tm), "r"(smem_u32(bar)), "r"(x), "r"(y) : "memory");
}
__device__ __forceinline__ void tma_prefetch_desc(const CUtensorMap* tm) {
  asm volatile("prefetch.tensormap [%0];" :: "l"(tm) : "memory");
}
// K-major operand, 128B swizzle, rows of 128 bytes, 8-row atoms (SBO = 1024 B).
__device__ __forceinline__ uint64_t make_desc(const void* p) {
  uint64_t a = smem_u32(p);
  return ((a & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
}
template <int R> __device__ __forceinline__ void fence_regs(float (&d)[R]) {
#pragma unroll
  for (int i = 0; i < R; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}

template <int BN> __device__ __forceinline__ void mma_k16(float (&d)[BN / 2], uint64_t da, uint64_t db);
template <> __device__ __forceinline__ void mma_k16<16>(float (&d)[8], uint64_t da, uint64_t db) { wgmma_n16(d, da, db, 1); }
template <> __device__ __forceinline__ void mma_k16<32>(float (&d)[16], uint64_t da, uint64_t db) { wgmma_n32(d, da, db, 1); }
template <> __device__ __forceinline__ void mma_k16<64>(float (&d)[32], uint64_t da, uint64_t db) { wgmma_n64(d, da, db, 1); }
template <> __device__ __forceinline__ void mma_k16<128>(float (&d)[64], uint64_t da, uint64_t db) { wgmma_n128(d, da, db, 1); }
template <> __device__ __forceinline__ void mma_k16<256>(float (&d)[128], uint64_t da, uint64_t db) { wgmma_n256(d, da, db, 1); }

constexpr int BK = 64;  // 64 bf16 = one 128-byte swizzle row

// MT: 64-row sub-tiles per consumer warpgroup (MT = 2: the CTA covers 128 * NC rows, so a butterfly tile's keystream
// is generated once for M <= 256 instead of once per 128-row m-tile; every element's k chain is unchanged)
template <int BN, int NC, int S, int XS = 0, int MT = 1>
struct Smem {
  static constexpr int BM = 64 * NC * MT;
  static constexpr int A_BYTES = BM * BK * 2;
  static constexpr int B_BYTES = BN * BK * 2;
  static constexpr int STAGE = A_BYTES + B_BYTES;
  static constexpr int XS_BYTES = XS;                        // butterfly exchange buffer
  static constexpr int TOTAL = S * STAGE + XS + 3 * S * 8 + 128 + 1024;  // + barriers + tile queue + alignment slack
};

// Butterfly W keystream (ALT1+FF, L=11, M=2048 units of 16 B, XB=2): super-chunk = a 128 x 128 bf16
// block of one weight matrix (rows n0..n0+127, cols 128*kp..+127), unit u = 16*r + cg covers row r,
// cols 128*kp + 8*cg .. +7; s = ordinal << 32 | (n0/128 * K/128 + kp); ks(s, u, seed) as bfly::superchunk_ks.
constexpr int BF_L = 11, BF_RB = 4, BF_MIX = 7, BF_XB = 2;

// DEC: 0 plaintext, 1 separable fmix keystream, 2 butterfly keystream (BN must be 128)
// Threads: DEC 0/1: WG0 producer, (WG1 decoder), then NC consumer WGs. DEC 2: consumer WGs first (warps
// 0..4NC-1, warpgroup-aligned for wgmma), then the decoder WG, then one producer warp (416 threads for
// NC = 2, so ptxas may give the butterfly decoder up to 152 registers).
template <int NC, int DEC> struct Threads {
  static constexpr int value = DEC == 2 ? 128 * (NC + 1) : 128 * (NC + (DEC ? 2 : 1));
};

template <int BN, int NC, int S, int DEC, int MT = 1>
__global__ void __maxnreg__((Threads<NC, DEC>::value >= 512 ? 128 : (Threads<NC, DEC>::value > 384 ? 152 : 168)))
gemm_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
            __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ bias,
            int M, int N, int K, int ldc, uint32_t key, int group_m, uint32_t ordinal = 0, uint32_t sc_off = 0,
            int* tile_ctr = nullptr, const int* gate = nullptr, int no_yield = 1) {
  using L = Smem<BN, NC, S, DEC == 2 ? 32768 : 0, MT>;
  constexpr int BM = L::BM;
  constexpr int CWG0 = DEC ? 2 : 1;  // first consumer warpgroup
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + S * L::STAGE + L::XS_BYTES);
  uint64_t* decd = full + S;
  uint64_t* empty = decd + S;
  // pre-emptible schedule (tile_ctr != nullptr, VARENA_FUSE_YIELD): tiles come from an atomic counter; the producer
  // (decoder thread 0) takes the next tile id, or the end marker once the challenge gate is raised (unless no_yield),
  // and publishes it through a 4-entry queue; every role reads tile ids from the queue, so the CTA stops after the
  // tile in hand and a resume launch on the same counter finishes the rest. Each tile is computed whole: identical.
  uint64_t* tqb = empty + S;
  int* tq = reinterpret_cast<int*>(tqb + 4);
  const bool dyn = tile_ctr != nullptr;

  // persistent tile loop with Triton's tile order (_compute_pid, GROUP_SIZE_M): CTA b handles
  // tiles b, b + gridDim.x, ... (gridDim.x = NUM_SMS reproduces matmul_kernel_persistent's
  // schedule; gridDim.x = #tiles gives one tile per CTA)
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
  auto tile_at = [&](int k) -> int {
    if (!dyn) return blockIdx.x + k * gridDim.x;
    mbar_wait(&tqb[k & 3], (k >> 2) & 1);
    return *(volatile int*)&tq[k & 3];
  };
  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&decd[s], 128);
      mbar_init(&empty[s], NC * 128);
    }
    for (int q = 0; q < 4; ++q) mbar_init(&tqb[q], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    tma_prefetch_desc(&tmA);
    tma_prefetch_desc(&tmB);
  }
  __syncthreads();

  // roles
  // DEC 2: no producer warpgroup (12 warps = 3 per SM sub-partition, so the decoder gets 168 regs);
  // the decoder's thread 0 issues the TMA loads between super-chunks
  const bool is_prod = DEC == 2 ? false : (wg == 0);
  const bool prod_leader = tid == 0;
  const bool is_dec = DEC == 2 ? (wg == NC) : (DEC == 1 && wg == 1);
  const int cons_idx = DEC == 2 ? wg : wg - CWG0;
  if (is_prod) {
    if (prod_leader) {
      int it = 0;
      for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        int m0, n0;
        tile_mn(tile, m0, n0);
        for (int kt = 0; kt < KT; ++kt, ++it) {
          const int s = it % S;
          if (it >= S) mbar_wait(&empty[s], ((it / S) & 1) ^ 1);
          uint8_t* st = smem + s * L::STAGE;
          mbar_expect_tx(&full[s], L::STAGE);
          tma_load_2d(st, &tmA, &full[s], kt * BK, m0);
          tma_load_2d(st + L::A_BYTES, &tmB, &full[s], kt * BK, n0);
        }
      }
    }
    return;
  }

  if (DEC == 2 && is_dec) {
    static_assert(DEC != 2 || BN == 128, "butterfly super-chunks are 128 rows");
    const int t = tid - 128 * NC;
    uint4* xs = reinterpret_cast<uint4*>(smem + S * L::STAGE);
    const int cg = t & 15, rb = t >> 4;              // unit u = t + 128 i -> row rb + 8 i, col group cg
    const uint32_t c_off = uint32_t(((cg & 7) ^ (rb & 7)) << 4);
    const int kp_per_row = K / 128;
    int pit = 0, ptile = blockIdx.x, pkt = 0, pk = 0;  // producer cursor (thread 0 only)
    auto fetch = [&]() -> int {
      const int g = (no_yield || gate == nullptr) ? 0 : *(volatile const int*)gate;
      return g ? num_tiles : atomicAdd(tile_ctr, 1);
    };
    auto publish = [&](int k, int v) {
      *(volatile int*)&tq[k & 3] = v;
      __threadfence_block();
      mbar_arrive(&tqb[k & 3]);
    };
    if (dyn && t == 0) {
      ptile = fetch();
      if (ptile > num_tiles) ptile = num_tiles;
      publish(0, ptile);
    }
    auto produce_upto = [&](int lim) {
      if (t != 0) return;
      while (pit < lim && ptile < num_tiles) {
        const int s = pit % S;
        if (pit >= S) mbar_wait(&empty[s], ((pit / S) & 1) ^ 1);
        int pm0, pn0;
        tile_mn(ptile, pm0, pn0);
        uint8_t* st = smem + s * L::STAGE;
        mbar_expect_tx(&full[s], L::STAGE);
        tma_load_2d(st, &tmA, &full[s], pkt * BK, pm0);
        tma_load_2d(st + L::A_BYTES, &tmB, &full[s], pkt * BK, pn0);
        ++pit;
        if (++pkt == KT) {
          pkt = 0;
          if (dyn) {
            ptile = fetch();
            if (ptile > num_tiles) ptile = num_tiles;
            publish(++pk, ptile);
          } else {
            ptile += gridDim.x;
          }
        }
      }
    };
    int it = 0;
    for (int tk = 0;; ++tk) {
      const int tile = tile_at(tk);
      if (tile >= num_tiles) break;
      int m0, n0;
      tile_mn(tile, m0, n0);
      for (int kt = 0; kt < KT; kt += 2, it += 2) {
        // k-tile j can be loaded once j-S is consumed, i.e. once j-S+1 is decoded: j <= it+S-2 here
        produce_upto(it + S - 1);
        // sc_off: super-chunk index of the first row block when W is a row slice of its matrix (sliced launches)
        const unsigned long long sidx = ((unsigned long long)ordinal << 32) |
                                        (unsigned long long)(sc_off + (n0 >> 7) * kp_per_row + (kt >> 1));
        uint4 v[16];
        bfly::superchunk_ks<BF_L, BF_RB, BF_MIX, BF_XB, true>(v, xs, sidx, t, key);
        const int sa = it % S, sb = (it + 1) % S;
        mbar_wait(&full[sa], (it / S) & 1);
        mbar_wait(&full[sb], ((it + 1) / S) & 1);
        uint8_t* bt = smem + (cg < 8 ? sa : sb) * L::STAGE + L::A_BYTES;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          uint4* p = reinterpret_cast<uint4*>(bt + (rb + 8 * i) * 128 + c_off);
          uint4 x = *p;
          x.x ^= v[i].x; x.y ^= v[i].y; x.z ^= v[i].z; x.w ^= v[i].w;
          *p = x;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbar_arrive(&decd[sa]);
        mbar_arrive(&decd[sb]);
      }
    }
    return;
  }

  if (DEC == 1 && is_dec) {
    // thread t owns logical 16B chunk c (k = 8c..8c+7 of the tile) of rows r0 + 16 i
    const int t = tid - 128, c = t & 7, r0 = t >> 3;
    constexpr int RI = BN / 16;
    const uint32_t chunk_off = uint32_t((c ^ (r0 & 7)) << 4);
    int it = 0;
    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
      int m0, n0;
      tile_mn(tile, m0, n0);
      uint32_t rp[RI];
#pragma unroll
      for (int i = 0; i < RI; ++i) {
        const uint32_t r = fmix(uint32_t(n0 + r0 + 16 * i) ^ key) & 0xFFFFu;
        rp[i] = r | (r << 16);
      }
      for (int kt = 0; kt < KT; ++kt, ++it) {
        const int s = it % S;
        const uint32_t k = uint32_t(kt * BK + c * 8);
        uint32_t cw[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t lo = fmix(fmix((k + 2 * j) ^ key) ^ WCOL) & 0xFFFFu;
          const uint32_t hi = fmix(fmix((k + 2 * j + 1) ^ key) ^ WCOL) & 0xFFFFu;
          cw[j] = lo | (hi << 16);
        }
        mbar_wait(&full[s], (it / S) & 1);
        uint8_t* bt = smem + s * L::STAGE + L::A_BYTES;
#pragma unroll
        for (int i = 0; i < RI; ++i) {
          uint4* p = reinterpret_cast<uint4*>(bt + (r0 + 16 * i) * 128 + chunk_off);
          uint4 v = *p;
          v.x ^= rp[i] ^ cw[0]; v.y ^= rp[i] ^ cw[1]; v.z ^= rp[i] ^ cw[2]; v.w ^= rp[i] ^ cw[3];
          *p = v;
        }
        // make the generic-proxy writes visible to the async proxy (wgmma) before signalling
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbar_arrive(&decd[s]);
      }
    }
    return;
  }

  // ---------------- consumers
  const int cw_idx = cons_idx;
  uint64_t* ready = DEC ? decd : full;
  const int w = (tid % 128) / 32, l = tid % 32;
  int it = 0;
  for (int tk = 0;; ++tk) {
    const int tile = tile_at(tk);
    if (tile >= num_tiles) break;
    int m0, n0;
    tile_mn(tile, m0, n0);
    float acc[MT][BN / 2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int i = 0; i < BN / 2; ++i) acc[mt][i] = 0.f;
    for (int kt = 0; kt < KT; ++kt, ++it) {
      const int s = it % S;
      mbar_wait(&ready[s], (it / S) & 1);
      uint8_t* st = smem + s * L::STAGE;
      const uint8_t* btile = st + L::A_BYTES;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) fence_regs(acc[mt]);
      wgmma_fence();
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          mma_k16<BN>(acc[mt], make_desc(st + (mt * NC * 64 + cw_idx * 64) * 128 + kk * 32), make_desc(btile + kk * 32));
      wgmma_commit();
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) fence_regs(acc[mt]);
      wgmma_wait<1>();
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) fence_regs(acc[mt]);
      if (kt > 0) mbar_arrive(&empty[(it - 1) % S]);
    }
    wgmma_wait<0>();
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) fence_regs(acc[mt]);
    mbar_arrive(&empty[(it - 1) % S]);

    // epilogue: wgmma accumulator layout (m64nNk16, f32): reg 4i+j of lane l in warp w holds
    // row 16w + l/4 + 8*(j>>1), col 8i + 2*(l%4) + (j&1)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
    const int rbase = m0 + mt * NC * 64 + cw_idx * 64 + w * 16 + l / 4;
#pragma unroll
    for (int i = 0; i < BN / 8; ++i) {
      const int col = n0 + 8 * i + 2 * (l % 4);
      if (col >= N) continue;
      float b0 = 0.f, b1 = 0.f;
      if (bias != nullptr) { b0 = __bfloat162float(bias[col]); b1 = __bfloat162float(bias[col + 1]); }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = rbase + 8 * h;
        if (row >= M) continue;
        float v0 = acc[mt][4 * i + 2 * h], v1 = acc[mt][4 * i + 2 * h + 1];
        if (bias != nullptr) { v0 += b0; v1 += b1; }
        __nv_bfloat162 o = __floats2bfloat162_rn(v0, v1);
        *reinterpret_cast<__nv_bfloat162*>(C + size_t(row) * ldc + col) = o;
      }
    }
    }
  }
}

}  // namespace encgemm
