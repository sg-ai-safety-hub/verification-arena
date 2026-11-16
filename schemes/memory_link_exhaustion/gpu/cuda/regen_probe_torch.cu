// Benchmark-only extension (not used by the prover, verifier or cheater): the regenerate-on-probe cheater's chase
// kernel, kept to measure what dropping pages costs under each W encoding (tests/bench_regen_probe_gpu.py).
// Same chain as chase.cu's honest kernel; each probed chunk is read if its page is resident
// (kind 0) or regenerated from the zero-weight keystream (kind 3): butterfly W through the chunk's dependency cone, or
// sloth W through one modular square root per word. Moved here unchanged from chase.cu / bfly.cuh / encgemm_torch.cu
// at 4d0f79c.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cstdlib>
#include "chase.cu"
#ifndef VARENA_W_L
#define VARENA_W_L 11
#endif
#ifndef VARENA_KV_LH
#define VARENA_KV_LH 3
#endif

namespace bfly {
// ---------------------------------------------------------------------------------------------
// Regenerating prover: work-efficient dependency-cone recompute of one probed chunk by one warp
// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ uint4 shfl_xor4(uint4 v, int m) {
    return make_uint4(__shfl_xor_sync(~0u, v.x, m), __shfl_xor_sync(~0u, v.y, m), __shfl_xor_sync(~0u, v.z, m),
                      __shfl_xor_sync(~0u, v.w, m));
}
// Work-efficient cone recompute (the attacker we use for slowdowns). Lane l owns U = M/32 consecutive units,
// processed as 16-unit register blocks: 16 leaves, stages 0..3 in registers (32 pair mixes), keep the 8 nodes on
// the chunk's side of bit 3, then fold blocks through stages 4..L-6 with a shared-memory stack. The last 5 stages
// pair lanes and run as a reduce-scatter (lane pairs split the 8 nodes 4/2/1, then 1 redundant mix for the
// last two), so per probe the warp evaluates M leaves + (2.5M - 8) + 40 pair mixes, all lanes busy.
template <int L, int MIX, int k, int j> struct Cone2Fold {
    static __device__ __forceinline__ void run(uint4 (&V)[8], uint4 * stk, int lane, unsigned c) {
        constexpr int g = L - 9;
        if constexpr (j < g) {
            if constexpr (((k >> j) & 1) == 0) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    stk[(j * 8 + i) * 32 + lane] = V[i];
                }
            } else {
                const bool keep_b = (c >> (1 + j)) & 1;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint4 A = stk[(j * 8 + i) * 32 + lane];
                    mix<MIX>(A, V[i], 4 + j);
                    V[i] = keep_b ? V[i] : A;
                }
                Cone2Fold<L, MIX, k, j + 1>::run(V, stk, lane, c);
            }
        }
    }
};
template <int L, int MIX, int XB, int k> struct Cone2Blocks {
    static __device__ __forceinline__ void run(uint4 (&V)[8], uint4 * stk, ull sup, int lane, unsigned c,
                                               uint32_t seed) {
        constexpr int nb = 1 << (L - 9);
        if constexpr (k < nb) {
            const uint32_t u0 = ((uint32_t) lane * nb + k) * 16;
            uint4          W[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                W[i] = leaf<XB>(sup, u0 + i, seed);
            }
            Stages<MIX, 0, 4, 0, 16>::run(W);
            const bool hi = c & 1;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                V[i] = hi ? W[8 + i] : W[i];
            }
            Cone2Fold<L, MIX, k, 0>::run(V, stk, lane, c);
            Cone2Blocks<L, MIX, XB, k + 1>::run(V, stk, sup, lane, c, seed);
        }
    }
};
// Reduce-scatter step over lane bit j (stage l = L-5+j). n = nodes held per lane before the step (8, 4, 2, 1, 1).
template <int MIX, int l, int n>
static __device__ __forceinline__ void rs_step(uint4 (&V)[8], int lane, int j, bool keep_b) {
    const bool up = (lane >> j) & 1;
    if constexpr (n > 1) {
        constexpr int h = n / 2;
        // keep indices [h*up, h*up+h), send the other half
        uint4 mine[h], other[h];
#pragma unroll
        for (int q = 0; q < h; ++q) {
            mine[q]  = up ? V[h + q] : V[q];
            other[q] = up ? V[q] : V[h + q];
        }
#pragma unroll
        for (int q = 0; q < h; ++q) {
            const uint4 P = shfl_xor4(other[q], 1 << j);
            uint4       a = up ? P : mine[q], b = up ? mine[q] : P;
            mix<MIX>(a, b, l);
            V[q] = keep_b ? b : a;
        }
    } else {
        const uint4 P = shfl_xor4(V[0], 1 << j);
        uint4       a = up ? P : V[0], b = up ? V[0] : P;
        mix<MIX>(a, b, l);
        V[0] = keep_b ? b : a;
    }
}
// Chunk c of super-chunk sup: this lane's 32-bit word (word lane&3 of unit 8c + lane>>2). All 32 lanes call it
// converged; stk = per-warp shared memory of CONE_SMEM_UNITS(L) uint4.
static __host__ __device__ constexpr int cone_smem_units(int L) { return (L - 9) * 8 * 32; }
template <int L, int MIX, int XB, bool FF>
static __device__ __forceinline__ unsigned cone_word(ull sup, unsigned c, uint32_t seed, uint4 * stk, int lane) {
    uint4          V[8];
    Cone2Blocks<L, MIX, XB, 0>::run(V, stk, sup, lane, c, seed);
    rs_step<MIX, L - 5, 8>(V, lane, 0, (c >> (L - 8)) & 1);
    // after stage L-5 a lane holds layer-(L-4) units 4*b0 + q (q < 4) of the group named by lane bits 1..4;
    // the chunk's own group holds the feed-forward values
    uint4 FFv[4];
    if constexpr (FF) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            FFv[q] = V[q];
        }
    }
    rs_step<MIX, L - 4, 4>(V, lane, 1, (c >> (L - 7)) & 1);
    rs_step<MIX, L - 3, 2>(V, lane, 2, (c >> (L - 6)) & 1);
    rs_step<MIX, L - 2, 1>(V, lane, 3, (c >> (L - 5)) & 1);
    rs_step<MIX, L - 1, 1>(V, lane, 4, (c >> (L - 4)) & 1);
    // lane holds output unit 4*b0 + 2*b1 + b2 (b = lane bits); fetch word (lane&3) of unit (lane>>2)
    const int n   = lane >> 2;
    const int src = ((n >> 2) & 1) | (((n >> 1) & 1) << 1) | ((n & 1) << 2);
    const unsigned x = __shfl_sync(~0u, V[0].x, src), y = __shfl_sync(~0u, V[0].y, src),
                   z = __shfl_sync(~0u, V[0].z, src), w = __shfl_sync(~0u, V[0].w, src);
    const int wd = lane & 3;
    unsigned  r  = wd == 0 ? x : wd == 1 ? y : wd == 2 ? z : w;
    if constexpr (FF) {
        const int sl = (int) (((c >> (L - 7)) & 15) << 1) | ((n >> 2) & 1);
        unsigned  f  = 0;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const unsigned fx = __shfl_sync(~0u, FFv[q].x, sl), fy = __shfl_sync(~0u, FFv[q].y, sl),
                           fz = __shfl_sync(~0u, FFv[q].z, sl), fw_ = __shfl_sync(~0u, FFv[q].w, sl);
            if (q == (n & 3)) {
                f = wd == 0 ? fx : wd == 1 ? fy : wd == 2 ? fz : fw_;
            }
        }
        r += f;
    }
    return r;
}

}  // namespace bfly

namespace chase {
// --- cheater helpers -------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t fmix32(uint32_t x) {
  x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16; return x;
}
// INDEP unit (16 B, 4 words) j of global chunk q (see ref.indep_words)
__device__ __forceinline__ uint4 indep_unit(uint64_t q, uint32_t j, uint32_t fkey, int hf8) {
  uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  for (int i = 0; i < hf8; ++i) {
    uint32_t c0 = (uint32_t)q, c1 = (uint32_t)(q >> 32), c2 = j + 8u * i, c3 = 0, k0 = fkey, k1 = 0x7F4A7C15u;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const uint32_t h0 = __umulhi(0xD2511F53u, c0), l0 = 0xD2511F53u * c0;
      const uint32_t h1 = __umulhi(0xCD9E8D57u, c2), l1 = 0xCD9E8D57u * c2;
      c0 = h1 ^ c1 ^ k0; c1 = l1; c2 = h0 ^ c3 ^ k1; c3 = l0;
      k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    s0 += c0; s1 += c1; s2 += c2; s3 += c3;
  }
  return make_uint4(s0, s1, s2, s3);
}
__device__ __forceinline__ uint32_t wrow(uint64_t r, uint32_t key) { return fmix32((uint32_t)r ^ key); }
__device__ __forceinline__ uint32_t wcol(uint64_t c, uint32_t key) {
  return fmix32(fmix32((uint32_t)c ^ key) ^ 0x7F4A7C15u);
}


// Butterfly KV chunk (schemes.memory_link_exhaustion.encodings.kvbfly, L = 7 or 8, MIX 7, Philox-2 leaves, feed-forward at stage L-4): word
// (lane & 3) of unit 8c + (lane >> 2) of super-chunk s, all 32 lanes converged. Lane l holds units
// l*U .. l*U+U-1 (U = 2^L / 32): stages below log2(U) mix inside a lane, the rest pair lanes by shuffle
// (both lanes of a pair evaluate the mix, keeping their own half), so the warp evaluates the whole
// 2^L-unit network once per probed chunk.
template <int L, int XB = 2>
__device__ __forceinline__ uint32_t kv_cone_word(uint64_t s, unsigned c, uint32_t seed, int lane) {
  constexpr int U = (1 << L) / 32;
  constexpr int LU = U == 8 ? 3 : U == 4 ? 2 : U == 2 ? 1 : 0;
  uint4 V[U], F[U];
#pragma unroll
  for (int i = 0; i < U; ++i) V[i] = bfly::leaf<XB>((bfly::ull)s, (uint32_t)(lane * U + i), seed);
#pragma unroll
  for (int l = 0; l < L; ++l) {
    if (l == L - 4) {
#pragma unroll
      for (int i = 0; i < U; ++i) F[i] = V[i];
    }
    if (l < LU) {
#pragma unroll
      for (int i = 0; i < U; ++i)
        if (!((i >> l) & 1)) bfly::mix<7>(V[i], V[i | (1 << l)], l);
    } else {
      const int m = 1 << (l - LU);
      const bool up = (lane & m) != 0;
#pragma unroll
      for (int i = 0; i < U; ++i) {
        const uint4 P = bfly::shfl_xor4(V[i], m);
        uint4 a = up ? P : V[i], b = up ? V[i] : P;
        bfly::mix<7>(a, b, l);
        V[i] = up ? b : a;
      }
    }
  }
  const int unit = 8 * (int)c + (lane >> 2), src = unit / U, idx = unit % U, wd = lane & 3;
  uint32_t r = 0;
#pragma unroll
  for (int i = 0; i < U; ++i) {
    const uint4 v = bfly::add4(V[i], F[i]);
    const uint32_t x = __shfl_sync(~0u, v.x, src), y = __shfl_sync(~0u, v.y, src),
                   z = __shfl_sync(~0u, v.z, src), w = __shfl_sync(~0u, v.w, src);
    if (i == idx) r = wd == 0 ? x : wd == 1 ? y : wd == 2 ? z : w;
  }
  return r;
}
#ifndef VARENA_KV_L
#define VARENA_KV_L 8
#endif


// sloth64 (p = 2^64 - 189) on the device, for the vacate cheater: regenerate a zero-plaintext sloth word
// (W / KV / filler) on demand, i.e. pay the modular square root per probe. Mirrors encodings/sloth64.py.
__device__ __forceinline__ uint64_t sl_mix(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__device__ __forceinline__ uint64_t sl_mulmod(uint64_t a, uint64_t b) {
  const uint64_t lo = a * b, hi = __umul64hi(a, b);
  const uint64_t hl = hi * 189ull, hh = __umul64hi(hi, 189ull);
  const uint64_t s = lo + hl;
  const uint64_t h = hh + (s < lo ? 1ull : 0ull);
  uint64_t s2 = s + h * 189ull;
  if (s2 < s) s2 += 189ull;
  const uint64_t P = 0xFFFFFFFFFFFFFF43ull;
  return s2 >= P ? s2 - P : s2;
}
__device__ __forceinline__ uint64_t sl_sigma_inv(uint64_t x) {
  const uint64_t P = 0xFFFFFFFFFFFFFF43ull;
  uint64_t r = x;                                   // x^e, e = 2^62 - 47 = 0x3FFFFFFFFFFFFFD1 (bit 61 implied)
#pragma unroll 1
  for (int i = 60; i >= 0; --i) {
    r = sl_mulmod(r, r);
    if ((0x3FFFFFFFFFFFFFD1ull >> i) & 1ull) r = sl_mulmod(r, x);
  }
  const bool qr = sl_mulmod(r, r) == x;
  const uint64_t r2 = r == 0 ? r : P - r;
  const uint64_t re = (r & 1ull) == 0 ? r : P - r;
  const uint64_t ro = (r2 & 1ull) == 1 ? r2 : P - r2;
  return x >= P ? x : (qr ? re : ro);
}
// 32-bit half `half` of encode(0) at 64-bit word index i under seed
__device__ __forceinline__ uint32_t sl_zero_word(uint64_t seed, uint64_t i, int half) {
  const uint64_t y = sl_sigma_inv(sl_mix(seed ^ (i * 0x9E3779B97F4A7C15ull)));
  return (uint32_t)(half ? (y >> 32) : y);
}

// Cheater: gran_page[j] = first global page id of virtual granule j; kind/a0/a1 per global page.
// USE (compile-time, so a cheater only pays for the resolve paths it uses): bit 1 host reads,
// bit 2 INDEP regen, bit 4 fmix-W regen, bit 8 butterfly-W regen, bit 16 butterfly-KV regen (kind 4,
// a0 = first element of the page = block * 32768, a1 low 32 bits = the layer's KV key). Resident reads are
// always on.
template <int USE>
__global__ void chase_cheat(const int64_t* __restrict__ gran_page, int total_blocks, int nb,
                            const uint32_t* __restrict__ nonce_p, int K, int P, uint64_t* __restrict__ out,
                            const int32_t* __restrict__ base, const int8_t* __restrict__ kind,
                            const int64_t* __restrict__ a0, const int64_t* __restrict__ a1,
                            uint32_t fkey, int hf8, const int32_t* __restrict__ aord, int bfly_w) {
  __shared__ int64_t tab[MAX_BLOCKS * GRAN_PER_BLOCK];
  extern __shared__ uint4 cone_stk[];       // bfly_w: per-warp cone stack (bfly::cone_smem_units(11))
  uint4* stk = cone_stk + (threadIdx.x >> 5) * bfly::cone_smem_units(11);
  (void)stk;
  __shared__ uint32_t nz;
  if (threadIdx.x == 0) nz = *(volatile const uint32_t*)nonce_p;
  __syncthreads();
  const uint64_t n64 = mix64((uint64_t)nz);
  const Geo G = geometry(n64, total_blocks, nb);
  for (int i = threadIdx.x; i < nb * GRAN_PER_BLOCK; i += blockDim.x) tab[i] = gran_page[G.b0 + i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int chain = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  if (chain >= K) return;
  const uint64_t W = lane_weight(lane);
  uint64_t s = mix64(n64 ^ mix64((uint64_t)(chain + 1) * GOLD));
  for (int r = 0; r < P; ++r) {
    const uint32_t g = scale(mix64(s ^ (((uint64_t)r * CPR + (uint64_t)(lane & (CPR - 1)) + 1) * GOLD)), G.nchunks);
    const int64_t tb = tab[g >> GRAN_SHIFT];
    // USE & 64 (mixed chase): a negative table entry is -(device address) of a contiguous granule (activation pool)
    const int64_t gp = ((USE & 64) && tb < 0) ? -1 : tb + ((g & ((1u << GRAN_SHIFT) - 1)) >> 9);
    const uint64_t ga = ((USE & 64) && tb < 0) ? (uint64_t)(-tb) + (uint64_t)(g & ((1u << GRAN_SHIFT) - 1)) * 128ull : 0ull;
    const uint32_t cw = (g & 511u) * 32u;
    uint32_t v[CPR];
#pragma unroll
    for (int c = 0; c < CPR; ++c) {
      const int64_t p = __shfl_sync(0xffffffffu, gp, c);
      const uint32_t w0 = __shfl_sync(0xffffffffu, cw, c);
      if constexpr ((USE & 64) != 0) {
        const uint64_t aa = __shfl_sync(0xffffffffu, ga, c);
        if (p < 0) {
          v[c] = __ldcg((const uint32_t*)aa + lane);
          continue;
        }
      }
      if constexpr ((USE & 32) != 0) {
        // sloth64 vacate: dropped W (kind 3), KV (kind 4) or filler (kind 1) pages regenerate encode(0) per probe
        const int kk = kind[p];
        if constexpr ((USE & 128) != 0) {
          // keyed-XOR KV (VARENA_KVENC=fmix, zero plaintext): one fmix32 per element (kernels._xor16_kernel)
          if (kk == 4) {
            const uint32_t key = (uint32_t)(a1[p] & 0xFFFFFFFF);
            const int64_t e = a0[p] + 2 * (int64_t)(w0 + lane);
            const uint32_t ka = fmix32((uint32_t)e ^ key) & 0xFFFFu, kb = fmix32((uint32_t)(e + 1) ^ key) & 0xFFFFu;
            v[c] = ka | (kb << 16);
            continue;
          }
        }
        if ((kk == 3 && (USE & 8) == 0) || (kk == 4 && (USE & (16 | 128)) == 0)) {
          const int64_t x = a0[p];
          const uint64_t key = (uint64_t)(uint32_t)(a1[p] & 0xFFFFFFFF);
          const uint64_t seed = kk == 3 ? ((key << 32) | (uint64_t)(uint32_t)aord[p]) : ((key << 32) | 0x4B56ull);
          const int64_t e = x + 2 * (int64_t)(w0 + lane);            // first element of this lane's 32-bit word
          v[c] = sl_zero_word(seed, (uint64_t)(e >> 2), (int)((e >> 1) & 1));
          continue;
        }
        if (kk == 1) {
          const uint64_t seed = ((uint64_t)fkey << 32) | 0xF11Eull;
          const uint64_t t = (uint64_t)(w0 + lane);
          v[c] = sl_zero_word(seed, (uint64_t)p * 8192ull + (t >> 1), (int)(t & 1));
          continue;
        }
      }
      if constexpr ((USE & ~1) == 0) {
        // resident / host only: a0 holds absolute (UVA) addresses, one table load, no kind
        v[c] = __ldcg((const uint32_t*)a0[p] + w0 + lane);
        continue;
      }
      const int k = kind[p];
      const int64_t x = a0[p];
      if (k == 0) {
        v[c] = __ldcg((const uint32_t*)(base + x) + w0 + lane);
      } else if ((USE & 1) && k == 2) {
        v[c] = *((const volatile uint32_t*)x + w0 + lane);
      } else if ((USE & 2) && k == 1) {
        const uint64_t q = (uint64_t)p * 512 + (w0 >> 5);
        const uint4 u = indep_unit(q, (uint32_t)(lane >> 2), fkey, hf8);
        const int t = lane & 3;
        v[c] = t == 0 ? u.x : t == 1 ? u.y : t == 2 ? u.z : u.w;
      } else if ((USE & 8) && (bfly_w & 1) && k == 3) {   // butterfly W keystream (zero-weight model): the chunk's dependency cone
        const int64_t y = a1[p];
        const int64_t ncols = y >> 32;
        const uint32_t key = (uint32_t)(y & 0xFFFFFFFF);
        const int64_t e = x + 2 * (int64_t)w0;            // first element of the 128 B chunk
        const int64_t row = e / ncols, col = e % ncols;
#if VARENA_W_L == 8
        // L=8 W chunks (schemes.memory_link_exhaustion.encodings.wbfly8): 16 rows x 128 cols; the 128 B probe = units 8c .. 8c+7 of the chunk
        const unsigned long long s8 = ((unsigned long long)(uint32_t)aord[p] << 32) |
                                      (unsigned long long)((row >> 4) * (ncols >> 7) + (col >> 7));
        const unsigned c8 = (unsigned)((row & 15) * 2 + ((col & 127) >> 6));
        v[c] = kv_cone_word<8, 2>(s8, c8, key, lane);
#else
        const unsigned long long sup = ((unsigned long long)(uint32_t)aord[p] << 32) |
                                       (unsigned long long)((row >> 7) * (ncols >> 7) + (col >> 7));
        const unsigned cc = (unsigned)((row & 127) * 2 + ((col & 127) >> 6));
        v[c] = bfly::cone_word<11, 7, 2, true>(sup, cc, key, stk, lane);
#endif
      } else if ((USE & 16) && k == 4) {   // butterfly KV keystream (zero-plaintext KV page)
        const uint32_t key = (uint32_t)(a1[p] & 0xFFFFFFFF);
        const uint32_t blk = (uint32_t)(x >> 15);
        // page word w0 = tok*512 + head*128 + kv*64 + dg*4: the 128 B chunk is units dg0..dg0+7 of token tok
        const uint32_t tok = w0 >> (7 + VARENA_KV_LH), head = (w0 >> 7) & ((1u << VARENA_KV_LH) - 1), kvh = (w0 >> 6) & 1, dg0 = (w0 >> 2) & 15;
        constexpr int T = 1 << (VARENA_KV_L - 4);
        const uint64_t sc = ((uint64_t)blk << 16) | (head << 8) | (kvh << 7) | (tok / T);
        v[c] = kv_cone_word<VARENA_KV_L>(sc, (tok % T) * 2 + (dg0 >> 3), key, lane);
      } else if ((USE & 4) && k == 3) {   // W keystream (zero-weight model): element pair of word w0 + lane
        const int64_t y = a1[p];
        const int64_t ncols = y >> 32;
        const uint32_t key = (uint32_t)(y & 0xFFFFFFFF);
        const int64_t ea = x + 2 * (int64_t)(w0 + lane), eb = ea + 1;
        const uint32_t ka = (wrow(ea / ncols, key) ^ wcol(ea % ncols, key)) & 0xFFFFu;
        const uint32_t kb = (wrow(eb / ncols, key) ^ wcol(eb % ncols, key)) & 0xFFFFu;
        v[c] = ka | (kb << 16);
      } else {
        v[c] = 0;
      }
    }
    // timing experiment (16 Nov, "stacked sloth"): bfly_w bits 8..15 = d, bits 16..31 = fraction f * 65535. A probe
    // whose page falls in the fraction does d extra sloth64 squarings (mod 2^64 - 189) plus a keystream XOR on the word
    // it read before using it, the per-word cost of d stacked sloth decodes. Content is wrong by design; only the
    // answer time is meant to be measured.
    if (const int sdep = (bfly_w >> 8) & 255) {
      const uint32_t sfr = (uint32_t)bfly_w >> 16;
#pragma unroll
      for (int c = 0; c < CPR; ++c) {
        const int64_t p = __shfl_sync(0xffffffffu, gp, c);
        if ((((uint32_t)p * 2654435761u) >> 16) >= sfr) continue;
        uint64_t x = (((uint64_t)v[c] << 32) | (uint64_t)(lane + 1)) % 0xFFFFFFFFFFFFFF43ull;
        for (int d = 0; d < sdep; ++d) x = sl_mulmod(x, x) ^ (0x9E3779B97F4A7C15ull >> (d & 7));
        v[c] = (uint32_t)(x ^ (x >> 32));
      }
    }
    s = combine(W, s, lane, v) ^ ((uint64_t)r * GOLD);
  }
  if (lane == 0) out[chain] = s;
}


}  // namespace chase

namespace {
static int chase_wpb() {
  static int w = [] { const char* e = getenv("VARENA_CHASE_WPB"); return e ? atoi(e) : 8; }();
  return w;
}

template <int USE>
static void chase_cheat_run(int grid, int wpb, int smem, const at::Tensor& gran_page, int64_t total_blocks,
                            int64_t nb, int64_t nonce_ptr, int64_t K, int64_t P, at::Tensor& out,
                            const at::Tensor& base, const at::Tensor& kind, const at::Tensor& a0,
                            const at::Tensor& a1, int64_t fkey, int64_t hf8, const at::Tensor& aord, int64_t bfly_w) {
  static bool attr = false;
  if (!attr && (USE & 8)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(chase::chase_cheat<USE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        wpb * bfly::cone_smem_units(11) * 16));
    attr = true;
  }
  chase::chase_cheat<USE><<<grid, 32 * wpb, (USE & 8) ? smem : 0, at::cuda::getCurrentCUDAStream()>>>(
      (const int64_t*)gran_page.data_ptr(), (int)total_blocks, (int)nb, (const uint32_t*)nonce_ptr,
      (int)K, (int)P, (uint64_t*)out.data_ptr(), (const int32_t*)base.data_ptr(),
      (const int8_t*)kind.data_ptr(), (const int64_t*)a0.data_ptr(), (const int64_t*)a1.data_ptr(),
      (uint32_t)(fkey & 0xFFFFFFFF), (int)hf8, (const int32_t*)aord.data_ptr(), (int)bfly_w);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// use: bit 1 host, 2 INDEP regen, 4 fmix-W regen, 8 butterfly-W regen, 16 butterfly-KV regen (chase.cu). use 0 or 1:
// a0 must hold absolute UVA addresses (device or mapped host) of each page.
void chase_cheat_launch(at::Tensor gran_page, int64_t total_blocks, int64_t nb, int64_t nonce_ptr,
                        int64_t K, int64_t P, at::Tensor out, at::Tensor base, at::Tensor kind,
                        at::Tensor a0, at::Tensor a1, int64_t fkey, int64_t hf8, at::Tensor aord, int64_t bfly_w,
                        int64_t use) {
  TORCH_CHECK(nb <= chase::MAX_BLOCKS && nb <= total_blocks);
  const int wpb = chase_wpb();
  const int grid = (int)((K + wpb - 1) / wpb);
  const int smem = wpb * bfly::cone_smem_units(11) * 16;
#define CC(U) chase_cheat_run<U>(grid, wpb, smem, gran_page, total_blocks, nb, nonce_ptr, K, P, out, base, kind, \
                                 a0, a1, fkey, hf8, aord, bfly_w)
  switch ((int)use) {
    case 2: CC(2); break;     // all resident (INDEP filler regen path compiled, never taken)
    case 8: CC(8); break;     // butterfly W regenerated through its dependency cone
    case 32: CC(32); break;   // sloth W regenerated: one modular square root per word
    default: TORCH_CHECK(false, "regen_probe: use 2, 8 or 32");
  }
#undef CC
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("chase_cheat", &chase_cheat_launch, "probe chase with regenerate-on-probe (benchmark only)");
}
