"""CPU reference of the butterfly W keystream (ALT1 + ADD-FF@L-4, L=11, Philox-2 leaves; from an earlier
butterfly-network prototype by the same authors, bit-exact with its reference and KAT); plus the varena W layout:
super-chunk = 128 x 128 bf16 block of one weight matrix, unit u = 16 r + cg (row r, cols 8 cg..),
s = ordinal << 32 | (row_block * K/128 + col_pair)."""
from __future__ import annotations

import numpy as np

try:
    import numba as nb
    _jit = nb.njit(cache=True)
except Exception:          # pragma: no cover
    nb = None
    _jit = lambda f: f

L, M, XB = 11, 2048, 2
MASK = 0xFFFFFFFF


def _h32(x):
    x &= MASK
    x ^= x >> 16
    x = (x * 0x7FEB352D) & MASK
    x ^= x >> 15
    x = (x * 0x846CA68B) & MASK
    x ^= x >> 16
    return x


KA = np.array([[_h32(0x1000 + 4 * l + w) | 1 for w in range(4)] for l in range(L)], dtype=np.uint64)
CX = np.array([[_h32(0x3000 + 4 * l + w) for w in range(4)] for l in range(L)], dtype=np.uint64)
RHO = np.array([[1 + _h32(0x4000 + 4 * l + w) % 31 for w in range(4)] for l in range(L)], dtype=np.int64)
RWORD = np.array([((l >> 1) + 2 * (l & 1)) & 3 for l in range(L)], dtype=np.int64)


@_jit
def _ks(s, seed, KA, CX, RHO, RWORD, out):
    """out[M, 4] uint32 keystream of super-chunk s."""
    msk = np.uint64(0xFFFFFFFF)
    x = np.empty((2048, 4), dtype=np.uint64)
    slo, shi = np.uint64(s) & msk, np.uint64(s) >> np.uint64(32)
    for u in range(2048):
        c0, c1, c2, c3 = slo, shi, np.uint64(u), np.uint64(0x1234)
        k0, k1 = np.uint64(seed), np.uint64(seed) ^ np.uint64(0x9E37)
        for _ in range(2):
            p0 = c0 * np.uint64(0xD2511F53)
            p1 = c2 * np.uint64(0xCD9E8D57)
            c0, c1, c2, c3 = ((p1 >> np.uint64(32)) ^ c1 ^ k0) & msk, p1 & msk, ((p0 >> np.uint64(32)) ^ c3 ^ k1) & msk, p0 & msk
            k0 = (k0 + np.uint64(0x9E3779B9)) & msk
            k1 = (k1 + np.uint64(0xBB67AE85)) & msk
        x[u, 0], x[u, 1], x[u, 2], x[u, 3] = c0, c1, c2, c3
    mid = np.empty((2048, 4), dtype=np.uint64)
    for l in range(11):
        if l == 7:
            mid[:, :] = x
        for i in range(2048):
            if (i >> l) & 1:
                continue
            j = i | (1 << l)
            a0, a1, a2, a3 = x[i, 0], x[i, 1], x[i, 2], x[i, 3]
            b0, b1, b2, b3 = x[j, 0], x[j, 1], x[j, 2], x[j, 3]
            rw = RWORD[l]
            if l & 1:      # b first
                f0 = ((a0 << np.uint64(RHO[l, 0])) | (a0 >> np.uint64(32 - RHO[l, 0]))) & msk if rw == 0 else a0
                f1 = ((a1 << np.uint64(RHO[l, 1])) | (a1 >> np.uint64(32 - RHO[l, 1]))) & msk if rw == 1 else a1
                f2 = ((a2 << np.uint64(RHO[l, 2])) | (a2 >> np.uint64(32 - RHO[l, 2]))) & msk if rw == 2 else a2
                f3 = ((a3 << np.uint64(RHO[l, 3])) | (a3 >> np.uint64(32 - RHO[l, 3]))) & msk if rw == 3 else a3
                b0 ^= f0 ^ CX[l, 0]
                b1 ^= f1 ^ CX[l, 1]
                b2 ^= f2 ^ CX[l, 2]
                b3 ^= f3 ^ CX[l, 3]
                a0 = (a0 * KA[l, 0] + b1) & msk
                a1 = (a1 * KA[l, 1] + b2) & msk
                a2 = (a2 * KA[l, 2] + b3) & msk
                a3 = (a3 * KA[l, 3] + b0) & msk
            else:
                a0 = (a0 * KA[l, 0] + b1) & msk
                a1 = (a1 * KA[l, 1] + b2) & msk
                a2 = (a2 * KA[l, 2] + b3) & msk
                a3 = (a3 * KA[l, 3] + b0) & msk
                f0 = ((a0 << np.uint64(RHO[l, 0])) | (a0 >> np.uint64(32 - RHO[l, 0]))) & msk if rw == 0 else a0
                f1 = ((a1 << np.uint64(RHO[l, 1])) | (a1 >> np.uint64(32 - RHO[l, 1]))) & msk if rw == 1 else a1
                f2 = ((a2 << np.uint64(RHO[l, 2])) | (a2 >> np.uint64(32 - RHO[l, 2]))) & msk if rw == 2 else a2
                f3 = ((a3 << np.uint64(RHO[l, 3])) | (a3 >> np.uint64(32 - RHO[l, 3]))) & msk if rw == 3 else a3
                b0 ^= f0 ^ CX[l, 0]
                b1 ^= f1 ^ CX[l, 1]
                b2 ^= f2 ^ CX[l, 2]
                b3 ^= f3 ^ CX[l, 3]
            x[i, 0], x[i, 1], x[i, 2], x[i, 3] = a0, a1, a2, a3
            x[j, 0], x[j, 1], x[j, 2], x[j, 3] = b0, b1, b2, b3
    for u in range(2048):
        for w in range(4):
            out[u, w] = np.uint32((x[u, w] + mid[u, w]) & msk)


_CACHE: dict = {}


def superchunk_ks(s: int, seed: int) -> np.ndarray:
    """uint32 [2048, 4]: ks(s, u, seed) for every unit u."""
    k = (s, seed)
    v = _CACHE.get(k)
    if v is None:
        if len(_CACHE) > 4096:
            _CACHE.clear()
        v = np.empty((M, 4), dtype=np.uint32)
        _ks(np.uint64(s), np.uint64(seed & MASK), KA, CX, RHO, RWORD, v)
        _CACHE[k] = v
    return v


def encode_range(raw_u16: np.ndarray, seed: int, ordinal: int, j0: int, ncols: int) -> np.ndarray:
    """Encode elements [j0, j0 + n) of a row-major [N, ncols] bf16 matrix (as uint16) with the butterfly
    keystream of the varena W layout."""
    n = raw_u16.size
    out = raw_u16.copy()
    kpr = ncols // 128
    j = j0
    while j < j0 + n:
        r, c = divmod(j, ncols)
        rb, kp = r // 128, c // 128
        seg_end = min(j0 + n, r * ncols + (kp + 1) * 128)      # same row, same 128-col group
        ks = superchunk_ks((ordinal << 32) | (rb * kpr + kp), seed).view(np.uint16).reshape(M, 8)
        u0 = (r % 128) * 16
        cols = np.arange(c, c + (seg_end - j))
        cg, e = (cols - kp * 128) // 8, (cols - kp * 128) % 8
        out[j - j0: seg_end - j0] ^= ks[u0 + cg, e]
        j = seg_end
    return out
