"""NumPy reference for every keyed function the GPU kernels compute.

The verifier uses these to recompute expected memory contents and digests on the CPU, and the tests use them to check
the GPU kernels bit for bit.

Conventions (shared with kernels.py):
  * fmix(x): murmur3 32-bit finaliser.
  * Weight / KV / filler encoding in the final configuration: sloth64 (VARENA_WENC / VARENA_KVENC / VARENA_FILLER =
    sloth, encodings/sloth64.py): slow to encode, one modular squaring to decode. The XOR keystream
    (low16(fmix(j ^ k)) per bf16 element) and the butterfly network (encodings/kvbfly.py) remain as the other
    encodings the kernels and tests still support.
  * Page digest: d0 = sum_j fmix(w_j ^ j*GOLD ^ nonce), d1 = sum_j v_j*(2j+1), both mod 2^32, over the page's
    uint32 words.
  * Chase granules and the keyed-permutation noise schedule (PERM_TAB_N windows per table).
"""

from __future__ import annotations

import os

import numpy as np

PAGE = 65536                  # bytes
PAGE_WORDS = PAGE // 4
GOLD = 0x9E3779B9
C1 = 0x85EBCA6B
C2 = 0xC2B2AE35
FILLER_ROUNDS = 2
U32 = np.uint32


def fmix(x):
    x = np.asarray(x).astype(np.uint32)
    with np.errstate(over="ignore"):
        x = x * U32(C1)
        x ^= x >> U32(13)
        x = x * U32(C2)
        x ^= x >> U32(16)
    return x


def tensor_key(master: int, offset: int) -> int:
    """Per-tensor key derived from the master key and the arena byte offset."""
    lo = offset & 0xFFFFFFFF
    hi = (offset >> 32) & 0xFFFFFFFF
    k = fmix(U32(master & 0xFFFFFFFF) ^ U32(lo))
    k = fmix(k ^ U32(hi) ^ U32((master >> 32) & 0xFFFFFFFF))
    return int(k)


def keystream16(key: int, j0: int, n: int) -> np.ndarray:
    j = np.arange(j0, j0 + n, dtype=np.uint64).astype(np.uint32)
    return (fmix(j ^ U32(key)) & U32(0xFFFF)).astype(np.uint16)


def encode16(raw_u16: np.ndarray, key: int, j0: int = 0) -> np.ndarray:
    """Encode (or decode; XOR is an involution) bf16 data viewed as uint16."""
    return raw_u16 ^ keystream16(key, j0, raw_u16.size)


def page_seed(fkey: int, page: int) -> np.ndarray:
    p = np.asarray(page, dtype=np.uint64)
    lo = (p & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    hi = (p >> np.uint64(32)).astype(np.uint32)
    return fmix(fmix(U32(fkey) ^ lo) ^ hi ^ U32(GOLD))


def filler_pages(fkey: int, pages, rounds: int = FILLER_ROUNDS) -> np.ndarray:
    """Filler for the given page indices. Returns uint32 [len(pages), PAGE_WORDS]."""
    pages = np.atleast_1d(np.asarray(pages, dtype=np.uint64))
    if FILLER == "indep":
        return np.stack([indep_words(fkey, int(p), 0, PAGE_WORDS) for p in pages]) if pages.size \
            else np.empty((0, PAGE_WORDS), np.uint32)
    s = page_seed(fkey, pages)
    out = np.empty((pages.size, PAGE_WORDS), dtype=np.uint32)
    with np.errstate(over="ignore"):
        for j in range(PAGE_WORDS):
            for _ in range(rounds):
                s = fmix(s + U32(j))
            out[:, j] = s
    return out


def page_digest(words: np.ndarray, nonce: int) -> tuple[int, int]:
    """Digest of one page (or any uint32 array). Returns (d0, d1)."""
    w = np.asarray(words, dtype=np.uint32).ravel()
    j = np.arange(w.size, dtype=np.uint64)
    with np.errstate(over="ignore"):
        jm = (j * np.uint64(GOLD)).astype(np.uint32)
        v = fmix(w ^ jm ^ U32(nonce))
        d0 = int(v.astype(np.uint64).sum() & 0xFFFFFFFF)
        d1 = int((v.astype(np.uint64) * (2 * j + 1)).astype(np.uint32).astype(np.uint64).sum()
                 & 0xFFFFFFFF)
    return d0, d1


def chain_start(nonce: int, chain: int, npages: int) -> int:
    return int(fmix(fmix(U32(nonce) ^ U32(chain)) ^ U32(GOLD))) % npages


def chain_next(d0: int, d1: int, step: int, npages: int) -> int:
    return int(fmix(U32(d0) ^ fmix(U32(d1) ^ U32(step)))) % npages


# Weight keystream (separable, so a decoding matmul pays one XOR per element):
# element (row n, col k) of a weight matrix with key t is XORed with
# low16(wrow(t, n) ^ wcol(t, k)).
WCOL = 0x7F4A7C15


def wrow(key: int, rows) -> np.ndarray:
    return fmix(np.asarray(rows).astype(np.uint32) ^ U32(key))


def wcol(key: int, cols) -> np.ndarray:
    return fmix(fmix(np.asarray(cols).astype(np.uint32) ^ U32(key)) ^ U32(WCOL))


def encode_w(raw_u16: np.ndarray, key: int, j0: int, ncols: int) -> np.ndarray:
    """Encode (or decode) elements [j0, j0+n) of a row-major weight with ncols columns."""
    j = np.arange(j0, j0 + raw_u16.size, dtype=np.int64)
    ks = (wrow(key, j // ncols) ^ wcol(key, j % ncols)) & U32(0xFFFF)
    return raw_u16 ^ ks.astype(np.uint16)


# A memory challenge served after noise window w uses chain_nonce(nonce, s(w)): its nonce is keyed by the newest window.


def chain_nonce(nonce: int, d0_win: int) -> int:
    return int(fmix(U32(nonce) ^ fmix(U32(d0_win) ^ U32(0x68E31DA4))))



# Scheduled noise (verifier-clocked): window i's secret s_i is revealed by the verifier at
# about its due time; s_i picks the pool slice.
def fmix_int(x: int) -> int:
    """fmix on a Python int (same value as fmix, ~20x cheaper per call than numpy scalars)."""
    x = (x * C1) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * C2) & 0xFFFFFFFF
    return x ^ (x >> 16)


def sched_key(vkey: int, i: int) -> int:
    return fmix_int((vkey & 0xFFFFFFFF) ^ fmix_int((i & 0xFFFFFFFF) ^ 0x2545F491))


def sched_offset(s: int, n_slots: int) -> int:
    return fmix_int((s & 0xFFFFFFFF) ^ 0x7FEB352D) % n_slots


# INDEP filler (write-once, chunk-independent; after an earlier prototype by the same authors): global 128 B chunk q
# = 8 units of 16 B; unit j (4 words) = sum over i < HF/8 of Philox4x32-8(ctr=(q_lo, q_hi, 8i+j, 0),
# key=(fkey, INDEP_K1)), wordwise mod 2^32, HF = 256 Philox calls per chunk. Every chunk is
# independent, so a stored subset or checkpoint doesn't help regenerate the rest.
# (Philox counter/key layout is ours; not KAT-verified against that prototype.)
FILLER = os.environ.get("VARENA_FILLER", "chain")
# W encoding: "fmix" (separable per-element keystream) or "bfly" (ALT1+FF butterfly, schemes.memory_link_exhaustion.encodings.bfly_ref)
WENC = os.environ.get("VARENA_WENC", "fmix")
# butterfly W network size: 11 (32 KiB 128 x 128 super-chunks, schemes.memory_link_exhaustion.encodings.bfly_ref) or 8 (4 KiB 16 x 128 chunks, the
# same network as KV and activations: schemes.memory_link_exhaustion.encodings.wbfly8)
W_L = int(os.environ.get("VARENA_W_BFLY_L", "11"))


def sloth_seed(key: int, ordinal: int) -> int:
    return ((int(key) & 0xFFFFFFFF) << 32) | (int(ordinal) & 0xFFFFFFFF)


def w_encode(src, dst, key: int, ordinal: int):
    """GPU: dst = encode(src) for one [N, K] matrix (butterfly: XOR; sloth: square-root permutation)."""
    if WENC == "sloth":
        from schemes.memory_link_exhaustion.encodings import sloth64
        sloth64.encode_to(src, dst, sloth_seed(key, ordinal))
    else:
        w_xor(src, dst, key, ordinal)


def w_xor(src, dst, key: int, ordinal: int, row0: int = 0):
    """GPU: dst = decode(src) of one [N, K] matrix (or rows row0.. of it). Butterfly: XOR with the W keystream
    (encode == decode); WENC=sloth (experiment): one modular squaring per 64-bit word."""
    if WENC == "sloth":
        from schemes.memory_link_exhaustion.encodings import sloth64
        sloth64.decode_to(src, dst, sloth_seed(key, ordinal), i0=row0 * src.shape[-1] // 4)
        return
    # butterfly W (L=11 super-chunks of 128 x 128, encodings/bfly_ref.py; CUDA decode in gpu/cuda/encgemm_torch.cu):
    # restored 16 Nov for the butterfly-vs-sloth comparison
    from schemes.memory_link_exhaustion.gpu.cuda import build as CB
    if row0:
        CB.load().bfly_xor_rows(src, dst, key, ordinal, row0)
    else:
        CB.load().bfly_xor_to(src, dst, key, ordinal)


def w_ordinal(offset: int) -> int:
    """Butterfly super-chunk ordinal of a weight matrix: its 64 KiB page index in the W buffer."""
    return (offset // 65536) & 0xFFFFFFFF
INDEP_HF = int(os.environ.get("VARENA_INDEP_HF", "256"))
INDEP_K1 = 0x7F4A7C15
PH_M0, PH_M1, PH_W0, PH_W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85


def philox8(c0, c1, c2, c3, k0, k1):
    """Philox4x32 with 8 rounds, vectorised over uint32 arrays."""
    c0, c1, c2, c3 = (np.asarray(x, dtype=np.uint64) for x in (c0, c1, c2, c3))
    k0, k1 = np.uint64(k0), np.uint64(k1)
    m = np.uint64(0xFFFFFFFF)
    for _ in range(8):
        p0 = c0 * np.uint64(PH_M0)
        p1 = c2 * np.uint64(PH_M1)
        c0, c1, c2, c3 = ((p1 >> np.uint64(32)) ^ c1 ^ k0) & m, p1 & m, ((p0 >> np.uint64(32)) ^ c3 ^ k1) & m, p0 & m
        k0 = (k0 + np.uint64(PH_W0)) & m
        k1 = (k1 + np.uint64(PH_W1)) & m
    return c0, c1, c2, c3


def indep_words(fkey: int, page: int, w0: int, nwords: int) -> np.ndarray:
    """INDEP filler words [w0, w0 + nwords) of global page `page` (w0, nwords multiples of 32)."""
    q = np.uint64(page) * np.uint64(512) + np.arange(w0 // 32, (w0 + nwords) // 32, dtype=np.uint64)
    qlo, qhi = (q & np.uint64(0xFFFFFFFF)), (q >> np.uint64(32))
    out = np.zeros((q.size, 8, 4), dtype=np.uint64)
    for j in range(8):
        for i in range(INDEP_HF // 8):
            r = philox8(qlo, qhi, np.full(q.size, 8 * i + j, np.uint64), np.zeros(q.size, np.uint64),
                        fkey, INDEP_K1)
            for t in range(4):
                out[:, j, t] += r[t]
    return (out & np.uint64(0xFFFFFFFF)).astype(np.uint32).reshape(-1)


# Probe chase (after an earlier high-concurrency chase prototype by the same authors): C warp chains, each round 8 random
# 128 B chunks (one 4 B word per lane), addresses from a splitmix64 hash of the chain state scaled
# to the chunk count; the state absorbs every word (lane-weighted warp sum, Horner) so the next
# round's addresses depend on all words of this one. The chunk space is a nonce-chosen set of
# blocks (<= 4 GiB each) of the W and F pages.
M64 = (1 << 64) - 1
PROBE_BLOCK_PAGES = 65536          # 4 GiB
PROBE_MUL = 0x9E3779B97F4A7C15


def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)




# CUDA probe chase (schemes/memory_link_exhaustion/gpu/cuda/chase.cu)
CH_GOLD, CH_MULC = 0x9E3779B97F4A7C15, 0xD1B54A32D192ED03
# chase round function (the kernel's combine): 1 (default) = t = rotl(t ^ w, 23) * MULC; 0 = the older t = t * MULC + w, which is
# affine in the data, so a chunk enters only through one weighted sum of its words (tests/test_chase_mix_cpu.py)
CH_MIX = int(__import__("os").environ.get("VARENA_CHASE_MIX", "1") or 0)
# probe width in 128-byte units (the honest kernel's VARENA_CHASE_WIDE): 1 = each lane reads one uint32; 2 / 4 = the
# probe is a 256 / 512 byte piece aligned to its size and each lane reads one / two 64-bit words of it
CH_WIDE = int(__import__("os").environ.get("VARENA_CHASE_WIDE", "1") or 1)
CH_GRAN_PAGES, CH_GRAN_PER_BLOCK = 1024, 64


def mix64(z: int) -> int:
    z ^= z >> 30
    z = (z * 0xBF58476D1CE4E5B9) & M64
    z ^= z >> 27
    z = (z * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def ch_scale(h: int, n: int) -> int:
    lo = ((h & 0xFFFFFFFF) * n) >> 32
    return (((h >> 32) * n + lo) >> 32) & 0xFFFFFFFF


def ch_granules(nW: int, nK: int, nF: int):
    """First global page of each virtual 64 MiB granule (W buffer, then filler)."""
    gw, gf = nW // CH_GRAN_PAGES, nF // CH_GRAN_PAGES
    return [j * CH_GRAN_PAGES for j in range(gw)] + [nW + nK + j * CH_GRAN_PAGES for j in range(gf)]




CH_PA0 = 1 << 40          # pseudo page numbers for activation-pool granules (outside the arena)
CH_MIX_SEED = 0x6D69786368617365


CH_RA0 = 1 << 42          # pseudo page numbers for noise-ring granules (--ring-chase-guard)
CH_RING_TAG = 1 << 63     # table entry of a ring granule on the device: CH_RING_TAG | logical granule index


def ring_chase_granules(n_ring: int, guard: int, tagged: bool = False) -> int:
    """Ring granules in the mixed chase: logical slots 0 .. n_ring - guard - 1 (slot q = the window q behind the
    head), two 32 MiB slots per granule. The oldest `guard` logical slots are the ones incoming windows overwrite
    while the chase runs: they are not probed, or with tagged ring reads (--ring-tag) all n_ring slots are probed and
    a read in those `guard` slots returns a tag (old or new window)."""
    if guard <= 0:
        return 0
    return int(n_ring) // 2 if tagged else max(0, (int(n_ring) - int(guard)) // 2)


def ring_chase_slot(page: int, word_off: int):
    """(logical slot q, uint32 word offset inside the window) of word word_off of ring pseudo page `page`."""
    j, pg = divmod(page - CH_RA0, CH_GRAN_PAGES)
    half = CH_GRAN_PAGES // 2
    return 2 * j + pg // half, (pg % half) * (PAGE // 4) + word_off


def ch_granules_mix(nW: int, nK: int, nF: int, n_pool: int = 0, n_ringg: int = 0):
    """Mixed chase: one granule table over every content-checked region (W, KV, filler, activation pool and, with
    n_ringg > 0, the noise ring), in a fixed interleaved order, wrapped to whole 4 GiB blocks. The nonce picks a
    window of blocks and the content read picks each next granule, so every challenge can touch every region and the
    prover cannot tell in advance which ones a chain will visit. Pool granules are CH_PA0 + j * CH_GRAN_PAGES, ring
    granules CH_RA0 + j * CH_GRAN_PAGES (head-relative: ring_chase_slot)."""
    import numpy as _np
    g = [j * CH_GRAN_PAGES for j in range(nW // CH_GRAN_PAGES)] + \
        [nW + j * CH_GRAN_PAGES for j in range(nK // CH_GRAN_PAGES)] + \
        [nW + nK + j * CH_GRAN_PAGES for j in range(nF // CH_GRAN_PAGES)] + \
        [CH_PA0 + j * CH_GRAN_PAGES for j in range(n_pool)] + \
        [CH_RA0 + j * CH_GRAN_PAGES for j in range(n_ringg)]
    perm = _np.random.default_rng(CH_MIX_SEED).permutation(len(g))
    g = [g[i] for i in perm]
    n = -(-len(g) // CH_GRAN_PER_BLOCK) * CH_GRAN_PER_BLOCK
    return [g[j % len(g)] for j in range(n)]


def ch_geometry(nonce32: int, total_blocks: int, nb: int):
    n64 = mix64(nonce32 & 0xFFFFFFFF)
    b0 = (mix64(n64) % (total_blocks - nb + 1)) * CH_GRAN_PER_BLOCK
    return n64, b0, nb << 25


def ch_range_pages(nonce32: int, gran_page, total_blocks: int, nb: int, lo: int, hi: int):
    """Arena pages in [lo, hi) inside the probed range of a chase with this (effective) nonce: the nb blocks from b0,
    every 1024-page granule whose first page is in [lo, hi). Sorted, unique (wrapped tables repeat granules).
    With K chains x P rounds x 8 probes spread uniformly over the range's nb * 2^16 pages, every page in it is read
    lam = 8 K P / (nb * 2^16) times on average (reference: 4224 x 1200 x 8 / 720,896 = 56), so the chance that a
    page in the range was not touched is exp(-lam) (~5e-25): this set is the chase's touched-page set."""
    out = []
    for x in ch_range_granules(nonce32, gran_page, total_blocks, nb, lo, hi):
        out.extend(range(x, x + CH_GRAN_PAGES))
    return out


def ch_range_granules(nonce32: int, gran_page, total_blocks: int, nb: int, lo: int, hi: int):
    """First pages of the granules behind ch_range_pages (sorted, unique)."""
    _, b0, _ = ch_geometry(nonce32, total_blocks, nb)
    return sorted({int(x) for x in gran_page[b0: b0 + nb * CH_GRAN_PER_BLOCK] if lo <= x < hi})


def ch_coverage(K: int, P: int, nb: int) -> float:
    """Mean probes per page inside a chase's probed range (8 probes per round per chain)."""
    return 8.0 * K * P / (nb * CH_GRAN_PER_BLOCK * CH_GRAN_PAGES)


_LW = None


# --- settled-KV rule: the chase reads only KV that cannot change under it ------------------------------------------
# A KV page is one block of one layer: 16 tokens x 4 KiB, token-major (token = word offset // 1024). The prover keeps
# one table entry per KV block, (lo, hi): token positions lo <= t < hi are the band a step already in flight may still
# write. A probe whose 128-byte chunk lies in that band does not read memory; it takes ch_mask_words, a fixed function
# of the challenge nonce and the chunk's address. Positions below lo hold the current owner's settled KV, positions
# from hi up hold what was there before (the previous owner's KV, or filler). With the table returned next to the
# timed answer, every word a chain used has a source the verifier can recompute, and nothing the chain read can
# change between the timed pass and a later replay.
CH_KV_TOK_WORDS = 1024            # uint32 words per token in a KV page (4 KiB)
CH_MASK_SALT = 0x6B766D61736B3031


def kv_tab_test(seed: int, n_blocks: int):
    """Synthetic block table for kernel validation (--kv-tab-test): lo uniform in 0..16 per block (seed > 0; about half
    of KV masked) or in about 2% of blocks (seed < 0; close to an honest in-flight share), hi = 16, as the
    uint16 lo | hi << 8 the kernel reads. No scheduler involved; the zero-weight cheater and the verifier both build it."""
    rng = np.random.default_rng(abs(int(seed)))
    lo = rng.integers(0, 17, int(n_blocks)).astype(np.uint16)
    if int(seed) < 0:                 # sparse: about 2% of blocks carry a band, the rest are fully settled (lo = 16)
        lo[rng.random(int(n_blocks)) >= 0.02] = 16
    return (lo | np.uint16(16 << 8)).astype(np.uint16)


def ch_kv_token(word_off: int) -> int:
    """Token position (0..15) inside its KV page of the 128-byte chunk at uint32 word offset word_off."""
    return word_off // CH_KV_TOK_WORDS


def ch_masked(band, word_off: int) -> bool:
    """band = (lo, hi) of the page's block, or None. True if the chunk is in the in-flight band."""
    return band is not None and band[0] <= ch_kv_token(word_off) < band[1]


def ch_mask_words(nonce32: int, kv_page: int, word_off: int):
    """The 32 uint32 words a masked chunk contributes: word l = low 32 bits of
    mix64(n64 ^ mix64(CH_MASK_SALT ^ chunk id) ^ (l + 1) * CH_GOLD), chunk id = kv_page * 512 + word_off // 32 and
    kv_page = the page's index inside the KV region (page - nW); n64 = mix64(nonce32)."""
    n64 = mix64(nonce32 & 0xFFFFFFFF)
    base = n64 ^ mix64(CH_MASK_SALT ^ ((kv_page * 512 + word_off // 32) & M64))
    return np.array([mix64(base ^ (((l + 1) * CH_GOLD) & M64)) & 0xFFFFFFFF for l in range(32)], dtype=np.uint32)


def ch_mask_words64(nonce32: int, kv_page: int, word_off: int, n: int):
    """Wide probes: the n 64-bit words of a masked piece starting at word_off (its first 128-byte chunk's id)."""
    n64 = mix64(nonce32 & 0xFFFFFFFF)
    base = n64 ^ mix64(CH_MASK_SALT ^ ((kv_page * 512 + word_off // 32) & M64))
    return [mix64(base ^ (((j + 1) * CH_GOLD) & M64)) for j in range(n)]


# Tagged in-flight reads (--kv-table 3): the chain READS a chunk in an in-flight band (no stand-in) and the kernel
# also returns a 32-bit tag of the 128 bytes it read: per chain CH_TAG_SLOTS uint32 = [number of in-band reads, then
# the tags in probe order (the first CH_TAG_SLOTS - 1 of them)]. The verifier's replay computes the tag of the old
# content (what the position held before the step in flight) and of the new content (the owner's KV), sees which one
# the kernel read, and continues the chain with it. The tag is only a hint: the chain value itself checks the bytes.
CH_TAG_SALT = 0x6B76746167303031
CH_TAG_SLOTS = 256
CH_RING_TAG_ID = 1 << 48      # chunk id of a tagged ring read: CH_RING_TAG_ID | q << 18 | chunk index in the window


def ch_tag_id(nonce32: int, chunk_id: int, words) -> int:
    """Tag of one tagged chunk: high 32 bits of mix64(sum_l words[l] * 31^popcount(l) ^ n64 ^ mix64(CH_TAG_SALT ^
    chunk id))."""
    global _LW
    if _LW is None:
        _LW = [31 ** bin(l).count("1") for l in range(32)]
    n64 = mix64(nonce32 & 0xFFFFFFFF)
    h = sum(int(words[l]) * _LW[l] for l in range(32)) & M64
    return (mix64(h ^ n64 ^ mix64(CH_TAG_SALT ^ (chunk_id & M64))) >> 32) & 0xFFFFFFFF


def ch_tag(nonce32: int, kv_page: int, word_off: int, words) -> int:
    """Tag of one in-band KV chunk: chunk id = kv_page * 512 + word_off // 32 (kv_page = the page's index inside the
    KV region)."""
    return ch_tag_id(nonce32, kv_page * 512 + word_off // 32, words)


def ring_tag_id(q: int, w_in: int) -> int:
    """Chunk id of a tagged ring read: logical slot q, uint32 word offset w_in inside the window."""
    return CH_RING_TAG_ID | (int(q) << 18) | (int(w_in) // 32)


def ch_chain(nonce32, chain, P, total_blocks, nb, gran_page, chunk_words, mask64=None, on_probe=None):
    """Reference chain state; chunk_words(page, word_off) -> uint32 [32]. CH_WIDE > 1: the probe is CH_WIDE
    consecutive 128-byte chunks from an aligned start; mask64(page, word_off) -> None or the piece's 16 * CH_WIDE
    64-bit words when the piece is in an in-flight band (settled-KV rule)."""
    global _LW
    if _LW is None:
        _LW = [31 ** bin(l).count("1") for l in range(32)]
    n64, b0, nch = ch_geometry(nonce32, total_blocks, nb)
    s = mix64(n64 ^ mix64(((chain + 1) * CH_GOLD) & M64))
    for r in range(P):
        t = [(s + l) & M64 for l in range(32)]
        for c in range(8):
            g = ch_scale(mix64(s ^ (((r * 8 + c + 1) * CH_GOLD) & M64)), nch)
            if CH_WIDE > 1:
                gi = (g & 0x7FFFF) & ~(CH_WIDE - 1)
                page, wo = gran_page[b0 + (g >> 19)] + (gi >> 9), (gi & 511) * 32
                x64 = mask64(page, wo) if mask64 is not None else None
                if x64 is None:
                    ws = [int(v) for k in range(CH_WIDE) for v in chunk_words(page, wo + 32 * k)]
                    x64 = [ws[2 * j] | (ws[2 * j + 1] << 32) for j in range(16 * CH_WIDE)]
                nw = CH_WIDE // 2
                for i in range(nw):
                    t = [((((x ^ x64[l * nw + i]) << 23) | ((x ^ x64[l * nw + i]) >> 41)) & M64) * CH_MULC & M64
                         for l, x in enumerate(t)]
                continue
            page = gran_page[b0 + (g >> 19)] + ((g & 0x7FFFF) >> 9)
            w = chunk_words(page, (g & 511) * 32)
            if on_probe is not None:
                on_probe(page, (g & 511) * 32, w)          # (tagged reads: the caller tags the in-band ones)
            if CH_MIX:
                t = [((((x ^ int(w[l])) << 23) | ((x ^ int(w[l])) >> 41)) & M64) * CH_MULC & M64 for l, x in enumerate(t)]
            else:
                t = [(t[l] * CH_MULC + int(w[l])) & M64 for l in range(32)]
        s = (sum(_LW[l] * t[l] for l in range(32)) & M64) ^ ((r * CH_GOLD) & M64)
    return s


# --- keyed permutation noise schedule (--noise-perm) ------------------------------------------------------------
# The default schedule picks each window's pool offset independently (sched_offset of an fmix-mixed index), so a new
# window overlaps pool bytes already in the GPU ring (~ring/pool of its bytes: ~3% for a 2 GiB ring over a 64 GiB pool,
# ~1.7 GB/s at 55 GB/s) that a prover can copy on the GPU instead of over the link; and fmix is invertible, so one
# revealed key gives the 32-bit vkey and every later offset. perm_keys() instead walks a keyed permutation of whole
# window-sized pool slots per cycle (SHAKE-256 under a 128-bit verifier secret), never repeats a slot within `horizon`
# windows across cycle boundaries, and encodes each window's slot as a revealed 32-bit value s_i with
# sched_offset(s_i, ns) == slot * pages_per_slot, padded with secret randomness: s_i determines only its own slot.
# The prover side (sched_offset of the revealed value) is unchanged.
_C1_INV = pow(0x85EBCA6B, -1, 1 << 32)
_C2_INV = pow(0xC2B2AE35, -1, 1 << 32)
PERM_TAB_N = 1 << 23          # windows per table (~85 min at 55 GB/s with 32 MiB windows); index wraps after that


def fmix_inv_int(y: int) -> int:
    """Inverse of fmix_int on 32 bits."""
    x = y & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * _C2_INV) & 0xFFFFFFFF
    x ^= (x >> 13) ^ (x >> 26)
    return (x * _C1_INV) & 0xFFFFFFFF


def _fmix_inv_np(y):
    x = y.astype(np.uint64) & 0xFFFFFFFF
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(_C2_INV)) & np.uint64(0xFFFFFFFF)
    x ^= (x >> np.uint64(13)) ^ (x >> np.uint64(26))
    return ((x * np.uint64(_C1_INV)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def _shake(key: bytes, tag: bytes, c: int, n: int) -> bytes:
    import hashlib
    return hashlib.shake_256(bytes(key) + tag + int(c).to_bytes(8, "little")).digest(n)


def perm_cycles(key: bytes, n_cycles: int, n_slots: int, horizon: int) -> np.ndarray:
    """[n_cycles, n_slots] slot order: each row a keyed permutation of range(n_slots); the first `horizon` entries of a
    row avoid the previous row's last `horizon` slots (each conflict swapped with the next allowed later entry)."""
    assert len(key) >= 16 and 2 * horizon <= n_slots, (len(key), horizon, n_slots)
    out = np.empty((n_cycles, n_slots), dtype=np.int64)
    prev = None
    for c in range(n_cycles):
        order = _raw_cycle(key, c, n_slots).copy()
        if prev is not None and horizon:
            bad = np.zeros(n_slots, dtype=bool)
            bad[prev[-horizon:]] = True
            q = horizon
            for p in np.nonzero(bad[order[:horizon]])[0]:
                while bad[order[q]]:
                    q += 1
                order[p], order[q] = order[q], order[p]
                q += 1
        out[c] = order
        prev = order
    return out


SEQ_KEY = b"SEQUENTIAL-WALK!"      # --noise-perm 2: public schedule, slots 0, 1, 2, ... in order every cycle


def _raw_cycle(key: bytes, c: int, n_slots: int) -> np.ndarray:
    if bytes(key) == SEQ_KEY:        # a slot recurs only after a whole pool cycle, so the boundary guard never fires
        return np.arange(n_slots, dtype=np.int64)
    return np.argsort(np.frombuffer(_shake(key, b"P", c, 8 * n_slots), dtype="<u8"), kind="stable")


def perm_cycle(key: bytes, c: int, n_slots: int, horizon: int) -> np.ndarray:
    """Cycle c's slot order on its own (indefinite schedule): with 4 * horizon <= n_slots the boundary guard only
    ever touches a cycle's first 3 * horizon entries, so the previous cycle's last `horizon` slots are those of its raw
    permutation and every cycle can be computed independently; equals perm_cycles(...)[c]."""
    assert len(key) >= 16 and 4 * horizon <= n_slots, (len(key), horizon, n_slots)
    order = _raw_cycle(key, c, n_slots)
    if c > 0 and horizon:
        bad = np.zeros(n_slots, dtype=bool)
        bad[_raw_cycle(key, c - 1, n_slots)[-horizon:]] = True
        q = horizon
        for p in np.nonzero(bad[order[:horizon]])[0]:
            while bad[order[q]]:
                q += 1
            order[p], order[q] = order[q], order[p]
            q += 1
    return order


def perm_keys_range(key: bytes, c0: int, n_cycles: int, n_slots: int, pages_per_slot: int, ns: int,
                    horizon: int) -> np.ndarray:
    """Revealed values for cycles c0 .. c0 + n_cycles - 1 (windows c0 * n_slots ...), computed independently of the
    cycles before c0: the schedule never wraps; equals perm_keys for the same windows."""
    assert (n_slots - 1) * pages_per_slot < ns
    cyc = np.stack([perm_cycle(key, c, n_slots, horizon) for c in range(c0, c0 + n_cycles)])
    off = (cyc * pages_per_slot).astype(np.uint64)
    rmax = ((np.uint64(1 << 32) - np.uint64(1) - off) // np.uint64(ns)) + np.uint64(1)
    rr = np.stack([np.frombuffer(_shake(key, b"R", c, 8 * n_slots), dtype="<u8") for c in range(c0, c0 + n_cycles)])
    y = off + np.uint64(ns) * (rr % rmax)
    return (_fmix_inv_np(y.reshape(-1)) ^ np.uint32(0x7FEB352D)).astype(np.uint32)


def perm_keys(key: bytes, n_win: int, n_slots: int, pages_per_slot: int, ns: int, horizon: int) -> np.ndarray:
    """Revealed values s_i (uint32) for windows 0..n_win-1, n_win a multiple of n_slots."""
    assert n_win % n_slots == 0 and (n_slots - 1) * pages_per_slot < ns
    cyc = perm_cycles(key, n_win // n_slots, n_slots, horizon)
    off = (cyc * pages_per_slot).astype(np.uint64)                        # [C, S] page offsets
    rmax = ((np.uint64(1 << 32) - np.uint64(1) - off) // np.uint64(ns)) + np.uint64(1)
    rr = np.stack([np.frombuffer(_shake(key, b"R", c, 8 * n_slots), dtype="<u8") for c in range(len(cyc))])
    y = off + np.uint64(ns) * (rr % rmax)                                 # fmix(s ^ K) = y; y % ns = off
    return (_fmix_inv_np(y.reshape(-1)) ^ np.uint32(0x7FEB352D)).astype(np.uint32)
