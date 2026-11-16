#pragma once
// Probe-chase residency challenge over the arena (W buffer + filler), after an earlier prototype by the same authors.
//
// Exact definition (the CPU reference in schemes/memory_link_exhaustion/encodings/ref.py computes the same):
//   virtual chunk space: the arena's 64 MiB granules (W buffer first, then filler chunks), 2^19 chunks
//   of 128 B each; a challenge probes NB consecutive 4 GiB blocks (64 granules each) starting at block
//   b0 = mix64(n64) mod (total_blocks - NB + 1), nchunks = NB * 2^25, with n64 = mix64(nonce32).
//   chain c: s = mix(n64 ^ mix((c + 1) * GOLD))
//   round r: g_k = scale(mix(s ^ ((r * 8 + k + 1) * GOLD)), nchunks), k = 0..7
//            t[l] = s + l; t[l] = t[l] * MULC + w_k[l] (k = 0..7), w_k[l] = word l of chunk g_k
//            s = (sum_l 31^popcount(l) * t[l]) ^ (r * GOLD)
//   digest[c] = s
// Honest kernel: one warp per chain, lane-parallel addressing, 8 independent loads per round, warp sum by
// redux.sync (as in the prototype's honest kernel); the block's granule table lives in shared memory.

#include <cuda_runtime.h>
#include <cstdint>
#include "bfly.cuh"

namespace chase {

constexpr int CPR = 8;
constexpr uint64_t GOLD = 0x9E3779B97F4A7C15ull;
constexpr uint64_t MULC = 0xD1B54A32D192ED03ull;
constexpr int GRAN_SHIFT = 19;          // 2^19 chunks of 128 B = 64 MiB
constexpr int GRAN_PER_BLOCK = 64;      // 4 GiB blocks
constexpr int MAX_BLOCKS = 16;

__host__ __device__ __forceinline__ uint64_t mix64(uint64_t z) {
  z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ull; z ^= z >> 27; z *= 0x94D049BB133111EBull; z ^= z >> 31;
  return z;
}
__host__ __device__ __forceinline__ uint32_t scale(uint64_t h, uint32_t n) {
  const uint64_t lo = ((h & 0xFFFFFFFFull) * n) >> 32;
  return (uint32_t)(((h >> 32) * n + lo) >> 32);
}
__device__ __forceinline__ uint64_t lane_weight(int lane) {
  uint64_t w = 1;
  for (int b = __popc(lane); b > 0; --b) w *= 31;
  return w;
}
__device__ __forceinline__ uint64_t warp_sum64(uint64_t u) {
  const uint32_t lo = (uint32_t)u, hi = (uint32_t)(u >> 32);
  const uint64_t a = __reduce_add_sync(0xffffffffu, lo & 0xFFFFu);
  const uint64_t b = __reduce_add_sync(0xffffffffu, lo >> 16);
  const uint64_t c = __reduce_add_sync(0xffffffffu, hi & 0xFFFFu);
  const uint64_t d = __reduce_add_sync(0xffffffffu, hi >> 16);
  return a + (b << 16) + (c << 32) + (d << 48);
}
// VARENA_CHASE_MIX = 1 (default): t = rotl(t ^ v, 23) * MULC. The older t = t * MULC + v (0) is affine in the data: a chunk
// then enters a chain only through one weighted 64-bit sum of its 32 words, the same for every nonce, chain and
// round, so 8 bytes per 128-byte chunk reproduce every chain (see tests/test_chase_mix_cpu.py).
#ifndef VARENA_CHASE_MIX
#define VARENA_CHASE_MIX 1
#endif
#ifndef VARENA_CHASE_WIDE
#define VARENA_CHASE_WIDE 1        // probe width in 128-byte units (honest kernel only: 1, 2 or 4)
#endif
#if VARENA_CHASE_WIDE != 1 && (VARENA_CHASE_MIX != 1 || (VARENA_CHASE_WIDE != 2 && VARENA_CHASE_WIDE != 4))
#error "VARENA_CHASE_WIDE is 1, 2 or 4, and wide probes need VARENA_CHASE_MIX=1"
#endif
__device__ __forceinline__ uint64_t combine(uint64_t w, uint64_t s, int lane, const uint32_t (&v)[CPR]) {
  uint64_t t = s + (uint64_t)lane;
#pragma unroll
  for (int c = 0; c < CPR; ++c) {
#if VARENA_CHASE_MIX
    t ^= (uint64_t)v[c];
    t = ((t << 23) | (t >> 41)) * MULC;
#else
    t = t * MULC + v[c];
#endif
  }
  return warp_sum64(t * w);
}

struct Geo {   // per challenge, computed by every block
  uint32_t nchunks;
  int64_t b0;  // first granule of the probed range
};
__device__ __forceinline__ Geo geometry(uint64_t n64, int total_blocks, int nb) {
  Geo g;
  g.nchunks = (uint32_t)nb << 25;
  g.b0 = (int64_t)(mix64(n64) % (uint64_t)(total_blocks - nb + 1)) * GRAN_PER_BLOCK;
  return g;
}

// gran_addr[j]: device address of virtual granule j (honest).
// Noise-ring granules (n_ring > 1): an entry with bit 63 set is logical ring granule (entry & 0xffff), two 32 MiB
// slots; logical slot q is the window q behind the head, physical slot (head - q) mod n_ring. The head is the
// word next to the nonce (nonce_p[1]), written by the challenge server from the same read of the last completed
// window that chains the nonce, so the probed slots are whole windows at or behind it.
constexpr int RING_SLOT_SHIFT = 18;     // 2^18 chunks of 128 B = one 32 MiB noise window
// Settled-KV rule (gran_kv != nullptr): gran_kv[j] is the index inside the KV region of granule j's first page (-1:
// not a KV granule); a KV page is one block of one layer (block = page index % nb_kv), 16 tokens x 4 KiB, so a
// 128-byte chunk c of the page belongs to token c >> 5. kv_tab[block] = lo | hi << 8: token positions lo <= t < hi may
// still be written by a step in flight. A probe there does not read memory: lane l takes the low 32 bits of
// mix64(n64 ^ mix64(MASK_SALT ^ (page index * 512 + c)) ^ (l + 1) * GOLD)  (encodings/ref.py ch_mask_words).
constexpr uint64_t MASK_SALT = 0x6B766D61736B3031ull;
// Tagged in-flight reads (info_off bit 8): a probe in a band READS memory like any other, and the kernel also returns
// a 32-bit tag of the 128 bytes it read, so the verifier's replay can tell whether the position held its old or its
// new content. After the K states, the gate slots and (version, digest): per chain TAG_SLOTS uint32 = [number of
// in-band reads, then their tags in probe order (the first TAG_SLOTS - 1)]. tag = high 32 bits of
// mix64(sum_l word_l * 31^popcount(l) ^ n64 ^ mix64(TAG_SALT ^ chunk id))  (encodings/ref.py ch_tag).
constexpr uint64_t TAG_SALT = 0x6B76746167303031ull;
constexpr int TAG_SLOTS = 256;
// Tagged ring reads (info_off bits 16..23 = zone, with bit 8): the chase reads every ring slot. A read in one of the
// `zone` oldest logical slots (the ones incoming windows overwrite while the chase runs) is tagged like an in-flight KV
// read, with chunk id RING_TAG_ID | q << 18 | chunk in window: the slot holds window head - q (old) or head - q +
// n_ring (new), and the verifier's replay picks by the tag.
constexpr uint64_t RING_TAG_ID = 1ull << 48;
__global__ void chase_honest(const uint64_t* __restrict__ gran_addr, int total_blocks, int nb,
                             const uint32_t* __restrict__ nonce_p, int K, int P, uint64_t* __restrict__ out,
                             uint64_t ring_base, int n_ring,
                             const int32_t* __restrict__ gran_kv, const uint16_t* __restrict__ kv_tab, int nb_kv,
                             const uint64_t* __restrict__ kv_info, int info_off) {
  __shared__ uint64_t tab[MAX_BLOCKS * GRAN_PER_BLOCK];
  // the block table's version and digest sit next to the table on the device and are replaced with it
  // (kv_tab_apply); the chase returns the pair it ran under after the K chain states (and after the gated graph's
  // two time slots, info_off = 2, when the challenge graph is gated)
  const bool tagm = (info_off & 0x100) != 0;
  const int ioff = info_off & 0xff;
  const int rzone = tagm ? (info_off >> 16) & 0xff : 0;
  if (kv_info && blockIdx.x == 0 && threadIdx.x == 0) { out[K + ioff] = kv_info[0]; out[K + ioff + 1] = kv_info[1]; }
  uint32_t* const tag_out = (uint32_t*)(out + K + ioff + 2);      // (only written when tagm)
  int ntag = 0;
  __shared__ int32_t tkv[MAX_BLOCKS * GRAN_PER_BLOCK];
  __shared__ int32_t tkb[MAX_BLOCKS * GRAN_PER_BLOCK];      // block of the granule's first page (tkv % nb_kv)
  __shared__ uint32_t nz;
  __shared__ int head;
  // one PCIe read of the (pinned host) nonce per block, not one per thread: 135k uncached host
  // reads per challenge cost ~3 ms (4x the whole chase)
  if (threadIdx.x == 0) {
    nz = *(volatile const uint32_t*)nonce_p;
    head = n_ring > 1 ? (int)((volatile const uint32_t*)nonce_p)[1] : 0;
  }
  __syncthreads();
  const uint64_t n64 = mix64((uint64_t)nz);
  const Geo G = geometry(n64, total_blocks, nb);
  for (int i = threadIdx.x; i < nb * GRAN_PER_BLOCK; i += blockDim.x) {
    tab[i] = gran_addr[G.b0 + i];
    tkv[i] = gran_kv ? gran_kv[G.b0 + i] : -1;
    tkb[i] = tkv[i] >= 0 ? tkv[i] % nb_kv : 0;
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int chain = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  if (chain >= K) return;
  const uint64_t W = lane_weight(lane);
  uint64_t s = mix64(n64 ^ mix64((uint64_t)(chain + 1) * GOLD));
  for (int r = 0; r < P; ++r) {
    const uint32_t g = scale(mix64(s ^ (((uint64_t)r * CPR + (uint64_t)(lane & (CPR - 1)) + 1) * GOLD)), G.nchunks);
    const uint64_t e = tab[g >> GRAN_SHIFT];
    // VARENA_CHASE_WIDE = w: a probe is a 128 * w byte piece (w = 1, 2, 4), aligned to its own size, and each lane
    // reads 4 * w bytes of it: more bytes per probe for about the same instructions (the idle kernel moves ~2.5 TB/s
    // at w = 1, about half of what HBM delivers; a host answering from DDR has to move the same bytes)
    const uint32_t off = (g & ((1u << GRAN_SHIFT) - 1)) & ~(uint32_t)(VARENA_CHASE_WIDE - 1);
    unsigned long long a;
    // mid = chunk id + 1 if this probe is tagged (in its KV block's in-flight band, or in the ring's zone), else 0
    unsigned long long mid = 0;
    if (e >> 63) {
      // n_ring is a power of two (checked at launch): the mask is the modulo, also for a negative difference
      const int q = (int)(e & 0xffffull) * 2 + (int)(off >> RING_SLOT_SHIFT);
      const int ps = (head - q) & (n_ring - 1);
      const uint32_t cw = off & ((1u << RING_SLOT_SHIFT) - 1);
      a = ring_base + ((uint64_t)ps << (RING_SLOT_SHIFT + 7)) + (uint64_t)cw * 128;
      if (rzone && q >= n_ring - rzone) mid = RING_TAG_ID + ((uint64_t)q << RING_SLOT_SHIFT) + cw + 1ull;
    } else {
      a = e + (uint64_t)off * 128;
    }
    const int kvp = tkv[g >> GRAN_SHIFT];
    if (kvp >= 0) {
      // a granule's 1,024 pages are consecutive blocks of one layer (or run into the next layer: one wrap)
      const uint32_t pi = off >> 9, pg = (uint32_t)kvp + pi, ch = off & 511u;
      uint32_t blk = (uint32_t)tkb[g >> GRAN_SHIFT] + pi;
      if (blk >= (uint32_t)nb_kv) blk -= (uint32_t)nb_kv;
      const uint32_t band = kv_tab[blk], tok = ch >> 5;
      if (tok >= (band & 0xffu) && tok < (band >> 8)) mid = (unsigned long long)pg * 512ull + ch + 1ull;
    }
#if VARENA_CHASE_WIDE == 1
    uint32_t v[CPR];
    // the in-flight band is a small share of probes: only a warp with a masked probe this round pays for it
    if (gran_kv && __ballot_sync(0xffffffffu, mid != 0ull)) {
#pragma unroll
      for (int c = 0; c < CPR; ++c) {
        const unsigned long long ac = __shfl_sync(0xffffffffu, a, c);
        const unsigned long long mc = __shfl_sync(0xffffffffu, mid, c);
        if (tagm) {
          v[c] = __ldcg((const uint32_t*)ac + lane);
          if (mc) {                                   // (the same for every lane of the warp: mc comes from lane c)
            const uint64_t h = warp_sum64((uint64_t)v[c] * W);
            if (lane == 0 && ntag < TAG_SLOTS - 1)
              tag_out[(size_t)chain * TAG_SLOTS + 1 + ntag] = (uint32_t)(mix64(h ^ n64 ^ mix64(TAG_SALT ^ (mc - 1ull))) >> 32);
            ++ntag;
          }
        } else {
          v[c] = mc ? (uint32_t)mix64(n64 ^ mix64(MASK_SALT ^ (mc - 1ull)) ^ ((uint64_t)(lane + 1) * GOLD))
                    : __ldcg((const uint32_t*)ac + lane);
        }
      }
    } else {
#pragma unroll
      for (int c = 0; c < CPR; ++c) v[c] = __ldcg((const uint32_t*)__shfl_sync(0xffffffffu, a, c) + lane);
    }
    s = combine(W, s, lane, v) ^ ((uint64_t)r * GOLD);
#else
    // wide probes: lane l takes the 64-bit words j = l * NW .. l * NW + NW - 1 of the piece (NW = w / 2), each through
    // t = rotl(t ^ x, 23) * MULC. A masked piece gives word j = mix64(n64 ^ mix64(MASK_SALT ^ id) ^ (j + 1) * GOLD).
    constexpr int NW = VARENA_CHASE_WIDE / 2;
    const bool anym = gran_kv && __ballot_sync(0xffffffffu, mid != 0ull);
    uint64_t t = s + (uint64_t)lane;
#pragma unroll
    for (int c = 0; c < CPR; ++c) {
      const unsigned long long ac = __shfl_sync(0xffffffffu, a, c);
      const unsigned long long mc = anym ? __shfl_sync(0xffffffffu, mid, c) : 0ull;
#pragma unroll
      for (int i = 0; i < NW; ++i) {
        const uint64_t x = mc ? mix64(n64 ^ mix64(MASK_SALT ^ (mc - 1ull)) ^ ((uint64_t)(lane * NW + i + 1) * GOLD))
                              : __ldcg((const uint64_t*)ac + lane * NW + i);
        t ^= x;
        t = ((t << 23) | (t >> 41)) * MULC;
      }
    }
    s = warp_sum64(t * W) ^ ((uint64_t)r * GOLD);
#endif
  }
  if (lane == 0) {
    out[chain] = s;
    if (tagm) tag_out[(size_t)chain * TAG_SLOTS] = (uint32_t)ntag;
  }
}

// Replace the device block table and its (version, digest) from pinned host memory in one kernel: no cudaMemcpy, so
// it does not queue behind a noise window. The caller holds the challenge server off while it runs.
__global__ void kv_tab_apply(const uint64_t* __restrict__ host_tab, uint64_t* __restrict__ dev_tab, int n_words,
                             const uint64_t* __restrict__ host_info, uint64_t* __restrict__ dev_info) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_words; i += gridDim.x * blockDim.x)
    dev_tab[i] = ((volatile const uint64_t*)host_tab)[i];
  if (blockIdx.x == 0 && threadIdx.x < 2) dev_info[threadIdx.x] = ((volatile const uint64_t*)host_info)[threadIdx.x];
}

}  // namespace chase
