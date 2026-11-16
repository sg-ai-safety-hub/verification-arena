"""Asymmetric (Sloth-style) word encoding: slow encode, fast decode. Weights, KV and filler in the final configuration
(VARENA_WENC / VARENA_KVENC / VARENA_FILLER = sloth).

Field: p = 2^64 - 189 (prime, p = 3 mod 4), one field element per 64-bit word (4 bf16). Words >= p (only
NaN bit patterns: top bf16 0xFFFF) are left unchanged by both directions.

  encode  y = S^-1(x ^ k(i))       S^-1 = square root with a parity convention (a bijection on [0, p))
  decode  x = S(y) ^ k(i)          S(r) = r^2 if r is even, p - r^2 if r is odd
  k(i)    = splitmix64(seed ^ (i * 0x9E3779B97F4A7C15)), i = word index in the region

S^-1(x): r = x^((p+1)/4); if r^2 == x (x a quadratic residue) take the even one of {r, p - r}; else
(-x is a residue since p = 3 mod 4) take the odd root of -x. Decode is one modular squaring; encode is an
exponentiation (~62 squarings + multiplies), so re-encoding dirtied memory (zeros included: k(i) makes every
word's encoding address-dependent) costs a per-word square root.
"""

from __future__ import annotations


P = (1 << 64) - 189
C = 189
E = (P + 1) // 4
M64 = (1 << 64) - 1


def splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def ks(seed: int, i: int) -> int:
    return splitmix64((seed ^ ((i * 0x9E3779B97F4A7C15) & M64)) & M64)


def sigma(r: int) -> int:
    if r >= P:
        return r
    s = r * r % P
    return s if r % 2 == 0 else (P - s) % P


def sigma_inv(x: int) -> int:
    if x >= P:
        return x
    r = pow(x, E, P)
    if r * r % P == x:
        return r if r % 2 == 0 else P - r
    r = pow((P - x) % P, E, P)
    return r if r % 2 == 1 else P - r


def encode_word(x: int, seed: int, i: int) -> int:
    return sigma_inv(x ^ ks(seed, i))


def decode_word(y: int, seed: int, i: int) -> int:
    return sigma(y) ^ ks(seed, i)


def encode_range(raw_u16, seed: int, j0: int):
    """CPU reference (verifier): encode bf16 elements [j0, j0 + n) of one matrix given as uint16; word i =
    elements 4i .. 4i+3 (little-endian). j0 and n must be multiples of 4."""
    import numpy as np
    n = raw_u16.size
    assert j0 % 4 == 0 and n % 4 == 0, (j0, n)
    words = np.ascontiguousarray(raw_u16).view(np.uint64)
    i0 = j0 // 4
    nb = _numba_encode()
    if nb is not None:
        out = np.empty_like(words)
        nb(words, np.uint64(seed & M64), np.uint64(i0), out)
        return out.view(np.uint16)
    out = np.empty_like(words)
    for t, w in enumerate(words.tolist()):
        out[t] = encode_word(w, seed, i0 + t)
    return out.view(np.uint16)


_NB = []


def _numba_encode():
    """numba version of the encode (same 32-bit-limb arithmetic as the Triton kernel), parallel over words."""
    if _NB:
        return _NB[0]
    try:
        import numba as nb
        import numpy as np
    except Exception:
        _NB.append(None)
        return None
    u = np.uint64

    @nb.njit(inline="always")
    def mulmod(a, b):
        m32 = u(0xFFFFFFFF)
        a0 = a & m32; a1 = a >> u(32); b0 = b & m32; b1 = b >> u(32)
        p00 = a0 * b0; p01 = a0 * b1; p10 = a1 * b0; p11 = a1 * b1
        mid = (p00 >> u(32)) + (p01 & m32) + (p10 & m32)
        lo = (p00 & m32) | (mid << u(32))
        hi = p11 + (p01 >> u(32)) + (p10 >> u(32)) + (mid >> u(32))
        c = u(189)
        t1 = (hi & m32) * c
        t2 = (hi >> u(32)) * c + (t1 >> u(32))
        hc_hi = t2 >> u(32)
        hc_lo = (t2 << u(32)) | (t1 & m32)
        s = lo + hc_lo
        h = hc_hi + (u(1) if s < lo else u(0))
        s2 = s + h * c
        if s2 < s:
            s2 = s2 + c
        pm = u(0xFFFFFFFFFFFFFF43)
        return s2 - pm if s2 >= pm else s2

    @nb.njit(inline="always")
    def powe(x):
        r = x
        e = u(0x3FFFFFFFFFFFFFD1)
        for i in range(60, -1, -1):
            r = mulmod(r, r)
            if (e >> u(i)) & u(1):
                r = mulmod(r, x)
        return r

    @nb.njit(inline="always")
    def splitmix(z):
        z = z + u(0x9E3779B97F4A7C15)
        z = (z ^ (z >> u(30))) * u(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> u(27))) * u(0x94D049BB133111EB)
        return z ^ (z >> u(31))

    @nb.njit(parallel=True, cache=True)
    def enc(words, seed, i0, out):
        pm = u(0xFFFFFFFFFFFFFF43)
        for t in nb.prange(words.size):
            x = words[t] ^ splitmix(seed ^ ((i0 + u(t)) * u(0x9E3779B97F4A7C15)))
            if x >= pm:
                out[t] = x
                continue
            r = powe(x)
            if mulmod(r, r) == x:
                out[t] = r if (r & u(1)) == u(0) else pm - r
            else:
                r2 = pm - r if r != u(0) else r
                out[t] = r2 if (r2 & u(1)) == u(1) else pm - r2
    @nb.njit(parallel=True, cache=True)
    def encg(words, seeds, idx, out):
        pm = u(0xFFFFFFFFFFFFFF43)
        for t in nb.prange(words.size):
            x = words[t] ^ splitmix(seeds[t] ^ (idx[t] * u(0x9E3779B97F4A7C15)))
            if x >= pm:
                out[t] = x
                continue
            r = powe(x)
            if mulmod(r, r) == x:
                out[t] = r if (r & u(1)) == u(0) else pm - r
            else:
                r2 = pm - r if r != u(0) else r
                out[t] = r2 if (r2 & u(1)) == u(1) else pm - r2
    _NBG.append(encg)
    _NB.append(enc)
    return enc


_NBG = []


def encode_gather(words, seeds, idx):
    """CPU reference, scattered words: out[t] = encode_word(words[t], seeds[t], idx[t]) for uint64 arrays (each word
    with its own matrix seed and word index). Used by the batched chain replay, where every probed 128-byte chunk
    comes from a different place."""
    import numpy as np
    words = np.ascontiguousarray(words, dtype=np.uint64)
    seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
    idx = np.ascontiguousarray(idx, dtype=np.uint64)
    out = np.empty_like(words)
    if _numba_encode() is not None:
        _NBG[0](words, seeds, idx, out)
        return out
    for t in range(words.size):
        out[t] = encode_word(int(words[t]), int(seeds[t]), int(idx[t]))
    return out


TRITON_SRC = r'''
import triton
import triton.language as tl

SL_C = tl.constexpr(189)


@triton.jit
def _sl_splitmix(z):
    z = z + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@triton.jit
def _sl_mulmod(a, b):
    """a * b mod p, p = 2^64 - 189 (uint64): native 64-bit low / high products (mul.lo / mul.hi.u64) instead of four
    32x32 partial products; same canonical residue as _sl_mulmod32 below."""
    lo = a * b
    hi = tl.umulhi(a, b)
    # hi * 189 as two 32-bit-wide products (189 < 2^8) instead of a 64-bit low and high multiply: same value,
    # +3% serving throughput (14 Nov)
    t1 = (hi & 0xFFFFFFFF) * SL_C
    t2 = (hi >> 32) * SL_C + (t1 >> 32)
    hc_hi = t2 >> 32
    hc_lo = (t2 << 32) | (t1 & 0xFFFFFFFF)
    s = lo + hc_lo
    h = hc_hi + (s < lo).to(tl.uint64)
    s2 = s + h * SL_C
    s2 = tl.where(s2 < s, s2 + SL_C, s2)
    pm = 0xFFFFFFFFFFFFFF43
    return tl.where(s2 >= pm, s2 - pm, s2)


@triton.jit
def _sl_mulmod32(a, b):
    """a * b mod p, p = 2^64 - 189, a, b < 2^64 (uint64)."""
    m32 = 0xFFFFFFFF
    a0 = a & m32
    a1 = a >> 32
    b0 = b & m32
    b1 = b >> 32
    p00 = a0 * b0
    p01 = a0 * b1
    p10 = a1 * b0
    p11 = a1 * b1
    mid = (p00 >> 32) + (p01 & m32) + (p10 & m32)
    lo = (p00 & m32) | (mid << 32)
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32)
    # x = hi * 2^64 + lo = hi * C + lo (mod p)
    t1 = (hi & m32) * SL_C
    t2 = (hi >> 32) * SL_C + (t1 >> 32)
    hc_hi = t2 >> 32
    hc_lo = (t2 << 32) | (t1 & m32)
    s = lo + hc_lo
    h = hc_hi + (s < lo).to(tl.uint64)
    s2 = s + h * SL_C
    s2 = tl.where(s2 < s, s2 + SL_C, s2)
    pm = 0xFFFFFFFFFFFFFF43
    return tl.where(s2 >= pm, s2 - pm, s2)


@triton.jit
def _sl_sqn(x, N: tl.constexpr):
    for _ in tl.static_range(N):
        x = _sl_mulmod(x, x)
    return x


@triton.jit
def _sl_pow_e(x):
    """x^((p+1)/4), e = 2^62 - 47 = (2^56 - 1) * 64 + 17: an addition chain of 61 squarings and 9 multiplies (the
    plain square-and-multiply over e's bits took 61 + 57). Same value, so encodings do not change."""
    r2 = _sl_mulmod(_sl_sqn(x, 1), x)             # x^(2^2 - 1)
    r4 = _sl_mulmod(_sl_sqn(r2, 2), r2)           # x^(2^4 - 1)
    r8 = _sl_mulmod(_sl_sqn(r4, 4), r4)
    r16 = _sl_mulmod(_sl_sqn(r8, 8), r8)
    r32 = _sl_mulmod(_sl_sqn(r16, 16), r16)
    r48 = _sl_mulmod(_sl_sqn(r32, 16), r16)
    r56 = _sl_mulmod(_sl_sqn(r48, 8), r8)         # x^(2^56 - 1)
    r = _sl_mulmod(_sl_sqn(r56, 2), x)            # ... * 4 + 1
    return _sl_mulmod(_sl_sqn(r, 4), x)           # ... * 16 + 1


@triton.jit
def _sl_sigma(y):
    pm = 0xFFFFFFFFFFFFFF43
    s = _sl_mulmod(y, y)
    neg = tl.where(s == 0, s, pm - s)
    x = tl.where((y & 1) == 0, s, neg)
    return tl.where(y >= pm, y, x)


@triton.jit
def _sl_sigma_inv(x):
    pm = 0xFFFFFFFFFFFFFF43
    r = _sl_pow_e(x)
    qr = _sl_mulmod(r, r) == x
    # e = (p+1)/4 is odd, so (-x)^e = -(x^e): one exponentiation serves both cases
    r2 = tl.where(r == 0, r, pm - r)
    re = tl.where((r & 1) == 0, r, pm - r)          # even root of x
    ro = tl.where((r2 & 1) == 1, r2, pm - r2)       # odd root of -x
    y = tl.where(qr, re, ro)
    return tl.where(x >= pm, x, y)


@triton.jit(do_not_specialize=['n', 'seed', 'i0'])
def _sl_decode_kernel(src, dst, n, seed, i0, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    y = tl.load(src + i, mask=m, other=0).to(tl.uint64, bitcast=True)
    k = _sl_splitmix(seed.to(tl.uint64) ^ ((i + i0).to(tl.uint64) * 0x9E3779B97F4A7C15))
    x = _sl_sigma(y) ^ k
    tl.store(dst + i, x.to(tl.int64, bitcast=True), mask=m)


@triton.jit(do_not_specialize=['seed'])
def _sl_gather_decode_kernel(src, ids, dst, WPR, seed, BLOCK: tl.constexpr):
    """dst[b, :] = decode(src row ids[b]) for a row-major [N, WPR] matrix of 64-bit words (embedding rows)."""
    b = tl.program_id(0)
    j = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = j < WPR
    r = tl.load(ids + b).to(tl.int64)
    i = r * WPR + j
    y = tl.load(src + i, mask=m, other=0).to(tl.uint64, bitcast=True)
    k = _sl_splitmix(seed.to(tl.uint64) ^ (i.to(tl.uint64) * 0x9E3779B97F4A7C15))
    x = _sl_sigma(y) ^ k
    tl.store(dst + b.to(tl.int64) * WPR + j, x.to(tl.int64, bitcast=True), mask=m)


@triton.jit(do_not_specialize=['seq_lo', 'maxb', 'wpr', 'bs', 'seed'])
def _sl_kv_ahead_kernel(src, table, seqused, dst, seq_lo, maxb, wpr, bs, seed, BLOCK: tl.constexpr):
    """KV decode-ahead: entry e of a chunk of sequences = block j = e % maxb of sequence seq_lo + e // maxb. If that
    block holds tokens (j * bs < seqused), decode the layer cache's block table[s, j] (wpr 64-bit words, word index
    block * wpr + jj) into scratch block e. Entries past a sequence's last block do nothing."""
    e = tl.program_id(0).to(tl.int64)
    sq = seq_lo + e // maxb
    j = e % maxb
    n = tl.load(seqused + sq).to(tl.int64)
    if j * bs >= n:
        return
    r = tl.load(table + sq * maxb + j).to(tl.int64)
    jj = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = jj < wpr
    i = r * wpr + jj
    y = tl.load(src + i, mask=m, other=0).to(tl.uint64, bitcast=True)
    k = _sl_splitmix(seed.to(tl.uint64) ^ (i.to(tl.uint64) * 0x9E3779B97F4A7C15))
    x = _sl_sigma(y) ^ k
    tl.store(dst + e * wpr + jj, x.to(tl.int64, bitcast=True), mask=m)


@triton.jit(do_not_specialize=['n', 'seed', 'i0'])
def _sl_encode_kernel(src, dst, n, seed, i0, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(src + i, mask=m, other=0).to(tl.uint64, bitcast=True)
    k = _sl_splitmix(seed.to(tl.uint64) ^ ((i + i0).to(tl.uint64) * 0x9E3779B97F4A7C15))
    y = _sl_sigma_inv(x ^ k)
    tl.store(dst + i, y.to(tl.int64, bitcast=True), mask=m)


@triton.jit(do_not_specialize=['first_page', 'seed'])
def _sl_fill_kernel(dst, page_map, pages, first_page, seed, LIST: tl.constexpr, BLOCK: tl.constexpr):
    """Sloth filler (VARENA_FILLER=sloth): page p's 64-bit word j = encode(0) at index p * 8192 + j, i.e. the
    modular square root of the keystream (writing it costs the square root; reading it back is a squaring)."""
    pi = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if LIST:
        page = tl.load(pages + pi).to(tl.int64)
        off = tl.load(page_map + page).to(tl.int64) // 2          # page_map: int32-word offsets
    else:
        page = first_page.to(tl.int64) + pi
        off = pi * 8192
    k = _sl_splitmix(seed.to(tl.uint64) ^ ((page * 8192 + j).to(tl.uint64) * 0x9E3779B97F4A7C15))
    tl.store(dst + off + j, _sl_sigma_inv(k).to(tl.int64, bitcast=True))
'''

# helpers for injection into other generated modules (no imports): everything from the first jit on, minus the
# standalone kernels
TRITON_HELPERS = TRITON_SRC[TRITON_SRC.index("SL_C = tl.constexpr(189)"):TRITON_SRC.index("@triton.jit(do_not_specialize=['n', 'seed', 'i0'])")] + r'''

@triton.jit
def _sl_pack4(x16):
    """uint16 [R, 4W] -> uint64 [R, W] (element 4w + e at bits 16e: little-endian words)."""
    R: tl.constexpr = x16.shape[0]
    W: tl.constexpr = x16.shape[1] // 4
    a = tl.reshape(x16.to(tl.uint64), (R, W, 2, 2))
    lo, hi = tl.split(a)
    d0, d2 = tl.split(lo)
    d1, d3 = tl.split(hi)
    return d0 | (d1 << 16) | (d2 << 32) | (d3 << 48)


@triton.jit
def _sl_unpack4(w):
    """uint64 [R, W] -> uint16 [R, 4W] (inverse of _sl_pack4)."""
    R: tl.constexpr = w.shape[0]
    W: tl.constexpr = w.shape[1]
    lo = tl.join(w & 0xFFFF, (w >> 32) & 0xFFFF)
    hi = tl.join((w >> 16) & 0xFFFF, (w >> 48) & 0xFFFF)
    return tl.reshape(tl.join(lo, hi), (R, 4 * W)).to(tl.uint16)


@triton.jit
def _sl_ks(seed, widx):
    return _sl_splitmix(seed ^ (widx.to(tl.uint64) * 0x9E3779B97F4A7C15))


@triton.jit
def _sl_enc_tile(x16, eidx, seed):
    """Encode a uint16 [R, 4W] tile; eidx: element index (int) of each element (groups of 4 share a word)."""
    R: tl.constexpr = x16.shape[0]
    W: tl.constexpr = x16.shape[1] // 4
    wi = tl.min(tl.reshape(eidx, (R, W, 4)), 2) // 4
    y = _sl_sigma_inv(_sl_pack4(x16) ^ _sl_ks(seed, wi))
    return _sl_unpack4(y)


@triton.jit
def _sl_dec_tile_fast(y16, rbase, seed):
    """Same result as _sl_dec_tile for a tile whose rows hold contiguous elements: rbase [R] int64 = element
    index of each row's first element (a multiple of 4). Word index = rbase / 4 + column word, and the keystream's
    index product is formed per row plus a per-column constant (one 64-bit multiply per row, not per word)."""
    R: tl.constexpr = y16.shape[0]
    W: tl.constexpr = y16.shape[1] // 4
    rb = (rbase // 4).to(tl.uint64) * 0x9E3779B97F4A7C15
    cw = tl.arange(0, W).to(tl.uint64) * 0x9E3779B97F4A7C15
    k = _sl_splitmix(seed ^ (rb[:, None] + cw[None, :]))
    return _sl_unpack4(_sl_sigma(_sl_pack4(y16)) ^ k)


@triton.jit
def _sl_dec_tile(y16, eidx, seed):
    R: tl.constexpr = y16.shape[0]
    W: tl.constexpr = y16.shape[1] // 4
    wi = tl.min(tl.reshape(eidx, (R, W, 4)), 2) // 4
    x = _sl_sigma(_sl_pack4(y16)) ^ _sl_ks(seed, wi)
    return _sl_unpack4(x)
'''


def kv_seed(key: int) -> int:
    """64-bit sloth seed of a layer's KV (VARENA_KVENC=sloth) from its 32-bit KV key."""
    return ((int(key) & 0xFFFFFFFF) << 32) | 0x4B56


_MOD = None


def load():
    global _MOD
    if _MOD is None:
        from schemes.memory_link_exhaustion.inference.patches import _write_and_import
        _MOD = _write_and_import("sloth64", TRITON_SRC)
    return _MOD


def decode_to(src, dst, seed: int, i0: int = 0, block: int = 1024):
    """dst = decode(src); src, dst: tensors viewed as int64 words (same numel), word index i0 + j."""
    import torch
    s, d = src.view(torch.int64).view(-1), dst.view(torch.int64).view(-1)
    n = s.numel()
    load()._sl_decode_kernel[((n + block - 1) // block,)](s, d, n, seed & M64, i0, BLOCK=block, num_warps=4)


def encode_to(src, dst, seed: int, i0: int = 0, block: int = 256):
    import torch
    s, d = src.view(torch.int64).view(-1), dst.view(torch.int64).view(-1)
    n = s.numel()
    load()._sl_encode_kernel[((n + block - 1) // block,)](s, d, n, seed & M64, i0, BLOCK=block, num_warps=4)


def gather_decode(src, ids, seed: int, block: int = 512):
    """Rows ids of an encoded [N, K] bf16 matrix, decoded: [len(ids), K] bf16 (embedding lookup)."""
    import torch
    N, K = src.shape
    wpr = K * src.element_size() // 8
    ids = ids.reshape(-1).contiguous()
    out = torch.empty(ids.numel(), K, dtype=src.dtype, device=src.device)
    load()._sl_gather_decode_kernel[(ids.numel(), (wpr + block - 1) // block)](
        src.view(torch.int64).view(-1), ids, out.view(torch.int64).view(-1), wpr, seed & M64, BLOCK=block,
        num_warps=4)
    return out


def kv_ahead(src_i64, table, seqused, dst, seq_lo: int, n_seq: int, maxb: int, wpr: int, bs: int, seed: int,
             block: int = 1024):
    """Decode the KV blocks that sequences seq_lo .. seq_lo + n_seq of a batch read into dst (scratch block
    (s - seq_lo) * maxb + j for block j of sequence s). src_i64: the layer cache as flat int64 words; table: the
    batch's int32 block table [num_seqs, maxb], contiguous; seqused: tokens in the cache per sequence. No host sync."""
    import torch
    load()._sl_kv_ahead_kernel[(n_seq * maxb, (wpr + block - 1) // block)](
        src_i64, table, seqused, dst.view(torch.int64).view(-1), seq_lo, maxb, wpr, bs, seed & M64, BLOCK=block,
        num_warps=4)


def gather_decode_to(src, ids, out, seed: int, block: int = 1024):
    """out[b] = decode(row ids[b] of src): src an encoded [N, K] matrix (any 2-byte dtype, K * 2 bytes a multiple of
    8), ids int32 / int64 [n] on the device, out a preallocated [>= n, K] buffer. No allocation, no host sync."""
    import torch
    N, K = src.shape
    wpr = K * src.element_size() // 8
    load()._sl_gather_decode_kernel[(ids.numel(), (wpr + block - 1) // block)](
        src.view(torch.int64).view(-1), ids, out.view(torch.int64).view(-1), wpr, seed & M64, BLOCK=block,
        num_warps=4)


def filler_seed(fkey: int) -> int:
    return ((int(fkey) & 0xFFFFFFFF) << 32) | 0xF11E


def fill_range(dst_bytes, first_page: int, fkey: int, block: int = 1024):
    """dst_bytes: page-aligned uint8 view of global pages first_page.. (sloth filler)."""
    import torch
    npages = dst_bytes.numel() // 65536
    if npages == 0:
        return
    d = dst_bytes.view(torch.int64)
    load()._sl_fill_kernel[(npages, 8192 // block)](d, d, d, int(first_page), filler_seed(fkey) & M64, LIST=False,
                                                     BLOCK=block, num_warps=4)


def fill_pages(base_i32, page_map, pages, fkey: int, block: int = 1024):
    """Refill the listed global pages (int64 tensor) of a page-mapped buffer with sloth filler."""
    import torch
    n = pages.numel()
    if n == 0:
        return
    load()._sl_fill_kernel[(n, 8192 // block)](base_i32.view(torch.int64), page_map, pages, 0,
                                               filler_seed(fkey) & M64, LIST=True, BLOCK=block, num_warps=4)


def filler_words(fkey: int, page: int, w0: int, n: int):
    """CPU reference: uint32 words w0 .. w0+n of filler page `page` (w0, n even)."""
    import numpy as np
    return encode_range(np.zeros(2 * n, np.uint16), filler_seed(fkey), page * 32768 + 2 * w0).view(np.uint32)
