#pragma once
// From an earlier butterfly-network prototype by the same authors; only change: BFLY_SYNC() for the round barriers.
// Butterfly (FFT-graph) super-chunk keystream: a non-associative, full-width mixing network inside a
// super-chunk.
//
// Super-chunk = M = 2^L units of 16 B (G = M/8 chunks of 128 B). For super-chunk key s and unit u:
//     x^0_u     = philox_XB(c = {s.lo, s.hi, u, 0x1234}, k = {seed, seed ^ 0x9E37})   (same leaf as the prototype)
//     stage l = 0..L-1: for every pair (i, j = i | 2^l) with bit l of i clear:
//         (x^{l+1}_i, x^{l+1}_j) = MIX_l(x^l_i, x^l_j)
//     ks_u      = x^L_u,   stored = plain XOR ks
// Stage l pairs units that differ in bit l, so the stage order is "chunk bits first": stages 0..2 mix the
// 8 units of each 128 B chunk, stages 3..L-1 mix across chunks. Every output depends on all M leaves
// through a binary tree of M-1 pair mixes; a 128 B chunk's dependency cone is M leaves + (2.5M - 8) mixes.
//
// MIX_l(a, b) on two 128-bit units (4 x 32-bit words each), stage constants are compile-time immediates:
//   MIX 0 "AXR" : a_w <- a_w * KA(l,w) + b_{w+1}        then  b_w <- b_w ^ rotl(a_w, RHO(l,w)) ^ CX(l,w)
//   MIX 1 "MHX" : b_w <- b_w ^ mulhi(a_w, KH(l,w)) ^ CX  then  a_w <- a_w * KA(l,w) + b_{w+1}
//   MIX 2 "BAL" : a_w <- a_w * KA(l,w) + b_{w+1}        then  b_w <- b_w ^ F_w(a_w) ^ CX(l,w),
//                 F_w = rotl(., RHO) for even w, mulhi(., KH) for odd w  (balances the ALU and FMA pipes)
//   MIX 3 "AXR1": as AXR but only word w = l mod 4 is rotated, the other three use F = identity
//   MIX 4 "AXR2": as AXR but words with w = l mod 2 (mod 2) are rotated (two per pair per stage)
//   MIX 5 "AX"  : no rotation at all (T-function; reference only, bit-0 planes are GF(2)-linear)
//   MIX 6/7/8 "ALT"/"ALT1"/"ALT2": Feistel order alternates per stage (even l: a first as above; odd l: b first,
//                 b_w <- b_w ^ F_w(a_w) ^ CX then a_w <- a_w * KA + b_{w+1}); 4 / 1 / 2 rotated words per stage.
//                 With a fixed order, unit 0's cone uses only a-first steps (a' = a*K + b) and is exactly
//                 Z/2^32-linear in the leaves; alternating puts both + and ^ on every path (see butterfly.md).
// All three are bijections on (a, b) (Feistel structure: each half-step is invertible given the other half),
// mix + (Z/2^32) with ^ (GF(2)) and move high bits down (rotate or mulhi), so no ring makes them linear.
// Word index w+1 is taken mod 4: the word permutation is free (register renaming) and spreads every input
// word across all output words over successive stages.

#include <cuda_runtime.h>

#include <cstdint>

// varena: the decode runs in one 128-thread warpgroup of a larger block (the GEMM's decoder), so the
// round exchanges use a named barrier over those 128 threads instead of __syncthreads().
#ifndef BFLY_SYNC
#define BFLY_SYNC() asm volatile("bar.sync 1, 128;" ::: "memory")
#endif

#ifndef BF_HD
#define BF_HD __host__ __device__
#endif
#define BF_FI __forceinline__

namespace bfly {

typedef unsigned long long ull;

constexpr uint32_t PHILOX_M0 = 0xD2511F53u, PHILOX_M1 = 0xCD9E8D57u;
constexpr uint32_t PHILOX_W0 = 0x9E3779B9u, PHILOX_W1 = 0xBB67AE85u;
constexpr uint32_t UNIT_C3 = 0x1234u, KEY1_XOR = 0x9E37u;

static BF_HD BF_FI uint32_t mulhi32(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
    return __umulhi(a, b);
#else
    return (uint32_t) (((uint64_t) a * b) >> 32);
#endif
}
static BF_HD BF_FI uint32_t rotl32(uint32_t x, int r) {
#ifdef __CUDA_ARCH__
    return __funnelshift_l(x, x, r);
#else
    return (x << r) | (x >> ((32 - r) & 31));
#endif
}

template <int X>
static BF_HD BF_FI uint4 philox16(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0, uint32_t k1) {
#pragma unroll
    for (int r = 0; r < X; ++r) {
        const uint32_t h0 = mulhi32(PHILOX_M0, c0), l0 = PHILOX_M0 * c0;
        const uint32_t h1 = mulhi32(PHILOX_M1, c2), l1 = PHILOX_M1 * c2;
        c0 = h1 ^ c1 ^ k0;
        c1 = l1;
        c2 = h0 ^ c3 ^ k1;
        c3 = l0;
        k0 += PHILOX_W0;
        k1 += PHILOX_W1;
    }
    return make_uint4(c0, c1, c2, c3);
}
// Leaf x^0_u of super-chunk s (identical to the prototype's leaf with buffer ordinal 0).
template <int XB> static BF_HD BF_FI uint4 leaf(ull s, uint32_t u, uint32_t seed) {
    return philox16<XB>((uint32_t) s, (uint32_t) (s >> 32), u, UNIT_C3, seed, seed ^ KEY1_XOR);
}

// ---------------------------------------------------------------------------------------------
// Stage constants (compile-time when l is a template constant)
// ---------------------------------------------------------------------------------------------
static BF_HD constexpr uint32_t h32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}
static BF_HD constexpr uint32_t KA(int l, int w) { return h32(0x1000u + 4u * (uint32_t) l + (uint32_t) w) | 1u; }
static BF_HD constexpr uint32_t KH(int l, int w) {
    return h32(0x2000u + 4u * (uint32_t) l + (uint32_t) w) | 0x80000001u;
}
static BF_HD constexpr uint32_t CX(int l, int w) { return h32(0x3000u + 4u * (uint32_t) l + (uint32_t) w); }
static BF_HD constexpr int RHO(int l, int w) {
    return 1 + (int) (h32(0x4000u + 4u * (uint32_t) l + (uint32_t) w) % 31u);
}

// Feedback F_w(a') of the b half-step. MIX 0 rotates every word, MIX 2 alternates rotate / mulhi, MIX 3 rotates
// one word per pair per stage (w = l mod 4), MIX 4 two words (w = l mod 2, l mod 2 + 2), MIX 5 none (a pure
// T-function: high bits never reach low bits; reference only, its bit-0 planes are GF(2)-linear).
// MIX 6/7/8 ("ALT", "ALT1", "ALT2") alternate the Feistel order per stage (even l: a first, odd l: b first) so that
// every path through the network, including the all-a path of unit 0 and the all-b path of unit M-1, crosses
// both + and ^ steps; they rotate 4 / 1 / 2 words per pair per stage.
static BF_HD constexpr int rword(int l) { return ((l >> 1) + 2 * (l & 1)) & 3; }
static BF_HD constexpr int fkind(int MIX, int l, int w) {  // 0 identity, 1 rotate, 2 mulhi
    return (MIX == 0 || MIX == 6) ? 1
           : MIX == 1             ? 2
           : MIX == 2             ? ((w & 1) == 0 ? 1 : 2)
           : MIX == 3             ? (w == (l & 3) ? 1 : 0)
           : MIX == 4             ? ((w & 1) == (l & 1) ? 1 : 0)
           : MIX == 7             ? (w == rword(l) ? 1 : 0)
           : MIX == 8             ? ((w & 1) == (rword(l) & 1) ? 1 : 0)
                                  : 0;
}
// Feistel order of stage l: 0 = a first (a' = a*K + b_{w+1}; b' = b ^ F(a') ^ C), 1 = b first
// (b' = b ^ F(a) ^ C; a' = a*K + b'_{w+1}).
static BF_HD constexpr int border(int MIX, int l) { return MIX == 1 ? 1 : MIX >= 6 ? (l & 1) : 0; }
static BF_HD BF_FI uint32_t fw(int MIX, int l, int w, uint32_t a) {
    const int k = fkind(MIX, l, w);
    return k == 1 ? rotl32(a, RHO(l, w)) : k == 2 ? mulhi32(a, KH(l, w)) : a;
}

// One pair mix at stage l. a = lower index (bit l clear), b = upper index.
template <int MIX> static BF_HD BF_FI void mix(uint4 & a, uint4 & b, int l) {
    if (border(MIX, l) == 1) {
        b.x ^= fw(MIX, l, 0, a.x) ^ CX(l, 0);
        b.y ^= fw(MIX, l, 1, a.y) ^ CX(l, 1);
        b.z ^= fw(MIX, l, 2, a.z) ^ CX(l, 2);
        b.w ^= fw(MIX, l, 3, a.w) ^ CX(l, 3);
        a.x = a.x * KA(l, 0) + b.y;
        a.y = a.y * KA(l, 1) + b.z;
        a.z = a.z * KA(l, 2) + b.w;
        a.w = a.w * KA(l, 3) + b.x;
    } else {
        a.x = a.x * KA(l, 0) + b.y;
        a.y = a.y * KA(l, 1) + b.z;
        a.z = a.z * KA(l, 2) + b.w;
        a.w = a.w * KA(l, 3) + b.x;
        b.x ^= fw(MIX, l, 0, a.x) ^ CX(l, 0);
        b.y ^= fw(MIX, l, 1, a.y) ^ CX(l, 1);
        b.z ^= fw(MIX, l, 2, a.z) ^ CX(l, 2);
        b.w ^= fw(MIX, l, 3, a.w) ^ CX(l, 3);
    }
}

// Inverse of mix (used by tests to confirm bijectivity).
static inline uint32_t inv32(uint32_t k) {  // k odd
    uint32_t x = k;
    for (int i = 0; i < 5; ++i) {
        x *= 2u - k * x;
    }
    return x;
}
template <int MIX> static inline void unmix(uint4 & a, uint4 & b, int l) {
    if (border(MIX, l) == 1) {
        a.x = (a.x - b.y) * inv32(KA(l, 0));
        a.y = (a.y - b.z) * inv32(KA(l, 1));
        a.z = (a.z - b.w) * inv32(KA(l, 2));
        a.w = (a.w - b.x) * inv32(KA(l, 3));
        b.x ^= fw(MIX, l, 0, a.x) ^ CX(l, 0);
        b.y ^= fw(MIX, l, 1, a.y) ^ CX(l, 1);
        b.z ^= fw(MIX, l, 2, a.z) ^ CX(l, 2);
        b.w ^= fw(MIX, l, 3, a.w) ^ CX(l, 3);
    } else {
        b.x ^= fw(MIX, l, 0, a.x) ^ CX(l, 0);
        b.y ^= fw(MIX, l, 1, a.y) ^ CX(l, 1);
        b.z ^= fw(MIX, l, 2, a.z) ^ CX(l, 2);
        b.w ^= fw(MIX, l, 3, a.w) ^ CX(l, 3);
        a.x = (a.x - b.y) * inv32(KA(l, 0));
        a.y = (a.y - b.z) * inv32(KA(l, 1));
        a.z = (a.z - b.w) * inv32(KA(l, 2));
        a.w = (a.w - b.x) * inv32(KA(l, 3));
    }
}

// Host reference: whole super-chunk keystream (M = 2^L units) into x. lstop < L stops early (layer lstop).
template <int MIX, int XB> static inline void ref_superchunk(ull s, int L, uint32_t seed, uint4 * x, int lstop = -1) {
    const int M = 1 << L;
    if (lstop < 0) {
        lstop = L;
    }
    for (int u = 0; u < M; ++u) {
        x[u] = leaf<XB>(s, (uint32_t) u, seed);
    }
    for (int l = 0; l < lstop; ++l) {
        for (int i = 0; i < M; ++i) {
            if (!((i >> l) & 1)) {
                mix<MIX>(x[i], x[i | (1 << l)], l);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Device decode building blocks
// ---------------------------------------------------------------------------------------------
// Register layout for a round whose 2^RB-unit register window covers unit bits [lo, lo+RB): register i holds
// bits [lo, lo+RB) of u, the thread index fills the other bits in increasing order.
static BF_HD BF_FI int umap(int t, int i, int lo, int RB) {
    return (t & ((1 << lo) - 1)) | (i << lo) | ((t >> lo) << (lo + RB));
}
// Shared-memory slot (bank-conflict-free for every round layout, see butterfly.md section 4).
static BF_HD BF_FI int swz(int u, int RB) { return u ^ ((u >> RB) & 7); }

// Stages [l, l1) of a round with window base lo on R in-register units.
template <int MIX, int l, int l1, int lo, int R> struct Stages {
    static __device__ BF_FI void run(uint4 (&v)[R]) {
        if constexpr (l < l1) {
            constexpr int jb = l - lo;
#pragma unroll
            for (int i = 0; i < R; ++i) {
                if (!((i >> jb) & 1)) {
                    mix<MIX>(v[i], v[i | (1 << jb)], l);
                }
            }
            Stages<MIX, l + 1, l1, lo, R>::run(v);
        }
    }
};

// Round partition: rounds 0..NR-2 take the stages [0, L-RB) in groups of up to RB, the last round takes exactly the
// last RB stages [L-RB, L). So the last exchange always carries layer L-RB (the feed-forward layer, FF below).
template <int L, int RB> struct Geo {
    static constexpr int M = 1 << L, R = 1 << RB, T = M / R, NR = 1 + (L - RB + RB - 1) / RB;
    static BF_HD constexpr int lo(int k) { return k < NR - 1 ? k * RB : L - RB; }
    static BF_HD constexpr int s0(int k) { return k < NR - 1 ? k * RB : L - RB; }
    static BF_HD constexpr int s1(int k) { return k < NR - 1 ? ((k + 1) * RB < L - RB ? (k + 1) * RB : L - RB) : L; }
};

// Round k: exchange from round k-1's layout through shared memory (if k > 0), then its stages.
template <int L, int RB, int MIX, int k> struct Rounds {
    using Gm = Geo<L, RB>;
    static __device__ BF_FI void run(uint4 (&v)[Gm::R], uint4 * xs, int t) {
        constexpr int lo = Gm::lo(k);
        if constexpr (k > 0) {
            constexpr int plo = Gm::lo(k - 1);
            BFLY_SYNC();
#pragma unroll
            for (int i = 0; i < Gm::R; ++i) {
                xs[swz(umap(t, i, plo, RB), RB)] = v[i];
            }
            BFLY_SYNC();
#pragma unroll
            for (int i = 0; i < Gm::R; ++i) {
                v[i] = xs[swz(umap(t, i, lo, RB), RB)];
            }
        }
        Stages<MIX, Gm::s0(k), Gm::s1(k), lo, Gm::R>::run(v);
        if constexpr (k + 1 < Gm::NR) {
            Rounds<L, RB, MIX, k + 1>::run(v, xs, t);
        }
    }
};

// Keystream of super-chunk s for thread t (block of Geo::T threads): on return v[i] holds ks of unit
// t + i * T (the final round's window is the top RB bits), i.e. the coalesced layout.
// FF: add (word-wise, mod 2^32) the layer-(L-4) value of the same unit to the network output (Davies-Meyer style
// feed-forward; see butterfly.md section 6). Layer L-4 is still in shared memory from the last exchange, at the
// same slots this thread read at the start of the last round, so this costs one LDS.128 + 4 adds per unit.
static BF_HD BF_FI uint4 add4(uint4 a, uint4 b) { return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
template <int L, int RB, int MIX, int XB, bool FF = false>
static __device__ BF_FI void superchunk_ks(uint4 (&v)[Geo<L, RB>::R], uint4 * xs, ull s, int t, uint32_t seed) {
    using Gm = Geo<L, RB>;
    static_assert(!FF || (RB == 4 && Gm::NR >= 2), "feed-forward layer L-4 must be the last exchange (RB = 4)");
#pragma unroll
    for (int i = 0; i < Gm::R; ++i) {
        v[i] = leaf<XB>(s, (uint32_t) umap(t, i, 0, RB), seed);
    }
    Rounds<L, RB, MIX, 0>::run(v, xs, t);
    if constexpr (FF) {
#pragma unroll
        for (int i = 0; i < Gm::R; ++i) {
            v[i] = add4(v[i], xs[swz(umap(t, i, Gm::lo(Gm::NR - 1), RB), RB)]);
        }
    }
}

// Host reference with feed-forward: ks_u = x^L_u + x^{L-4}_u.
template <int MIX, int XB> static inline void ref_superchunk_ff(ull s, int L, uint32_t seed, uint4 * x) {
    const int M = 1 << L;
    ref_superchunk<MIX, XB>(s, L, seed, x, L - 4);
    uint4 * mid = new uint4[M];
    for (int u = 0; u < M; ++u) {
        mid[u] = x[u];
    }
    for (int l = L - 4; l < L; ++l) {
        for (int i = 0; i < M; ++i) {
            if (!((i >> l) & 1)) {
                mix<MIX>(x[i], x[i | (1 << l)], l);
            }
        }
    }
    for (int u = 0; u < M; ++u) {
        x[u] = add4(x[u], mid[u]);
    }
    delete[] mid;
}

}  // namespace bfly
