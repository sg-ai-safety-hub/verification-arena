"""Medium-rate asymmetric encoding over a 32-bit prime field (the sloth64 construction, one field element per 32-bit
word, 2 bf16), used for the activation pool's content (VARENA_AE_S32=1). In the final configuration only the
zero-weight cheater allocates an activation pool; the honest prover has none.

  p = 2^32 - 5 (prime, p = 3 mod 4); E = (p + 1) / 4 = 2^30 - 1 (odd)
  encode  y = S^-1(x ^ k(i)),  decode  x = S(y) ^ k(i),  k(i) = fmix32(seed ^ (i * 0x9E3779B9))
  S(r) = r^2 if r even else p - r^2;  S^-1 = square root with the parity convention (bijection on [0, p))
  words >= p (upper bf16 0xFFFF, i.e. NaN bit patterns) map to themselves: the map is a bijection on all 2^32
  words, so NaN / inf activations round-trip exactly.

Security note: the field is small enough for a full 2^32-entry square-root table (16 GiB). A cheater holding such a
table restores by lookup, but the table is itself 16 GiB of memory, far more than the pool it would let him borrow.
"""

from __future__ import annotations

P = (1 << 32) - 5
E = (P + 1) // 4
M32 = (1 << 32) - 1


def fmix32(x: int) -> int:
    x &= M32
    x = (x * 0x85EBCA6B) & M32
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & M32
    return x ^ (x >> 16)


def ks(seed: int, i: int) -> int:
    return fmix32((seed ^ ((i * 0x9E3779B9) & M32)) & M32)


def pow_e_chain(x: int) -> int:
    """Python mirror of the Triton addition chain (must equal pow(x, E, P))."""
    def sq(v, n):
        for _ in range(n):
            v = v * v % P
        return v
    x2 = x * x % P * x % P
    x3 = x2 * x2 % P * x % P
    x6 = sq(x3, 3) * x3 % P
    x12 = sq(x6, 6) * x6 % P
    x15 = sq(x12, 3) * x3 % P
    return sq(x15, 15) * x15 % P


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
    r2 = (P - r) % P                         # (-x)^E = -(x^E), E odd
    return r2 if r2 % 2 == 1 else P - r2


def encode_word(x: int, seed: int, i: int) -> int:
    return sigma_inv(x ^ ks(seed, i))


def decode_word(y: int, seed: int, i: int) -> int:
    return sigma(y) ^ ks(seed, i)


def pool_ks(seed: int, cid: int, wl: int) -> int:
    """Activation-pool keystream word (mirror of actenc2._ae_s32_ks): chunk id cid (address >> 12, full width),
    word wl (0..1023) within the 4 KiB chunk."""
    lo = (((cid & 0xFFFFF) * 1024 + wl) * 0x9E3779B9) & M32
    hi = fmix32(((cid >> 20) & M32) ^ 0x85EBCA6B)
    return fmix32((seed & M32) ^ lo ^ hi)


def pool_unit_zero(seed: int, addr: int):
    """uint32 [32]: the encoded-zero content of the 128 B pool unit at byte address addr (verifier reference)."""
    import numpy as np
    cid, wl0 = (addr >> 12) & 0xFFFFFFFF, (addr & 4095) // 4          # as actenc2.cid()
    return np.array([sigma_inv(pool_ks(seed, cid, wl0 + j)) for j in range(32)], dtype=np.uint32)


TRITON_HELPERS = r'''

@triton.jit
def _s32_fmix(x):
    x = x * 0x85EBCA6B
    x = x ^ (x >> 13)
    x = x * 0xC2B2AE35
    return x ^ (x >> 16)


@triton.jit
def _s32_mulmod(a, b):
    """a * b mod p, p = 2^32 - 5; a, b uint32 < 2^32. 32-bit form: no emulated 64-bit arithmetic (umulhi + adds);
    same canonical residue as the 64-bit form below."""
    lo = a * b
    hi = tl.umulhi(a, b)
    s = lo + hi * 5                                   # hi * 2^32 == 5 * hi (mod p)
    c = tl.umulhi(hi, tl.full(hi.shape, 5, tl.uint32)) + (s < lo).to(tl.uint32)   # carry-out, <= 5
    s2 = s + c * 5
    s2 = tl.where(s2 < s, s2 + 5, s2)                 # wrapped again: s2 < 30 now, add the 5 back
    return tl.where(s2 >= 0xFFFFFFFB, s2 - 0xFFFFFFFB, s2)


@triton.jit
def _s32_mulmod64(a, b):
    """a * b mod p, p = 2^32 - 5; a, b uint32 < 2^32."""
    t = a.to(tl.uint64) * b.to(tl.uint64)
    t = (t & 0xFFFFFFFF) + (t >> 32) * 5          # < 2^32 + 5 * 2^32
    t = (t & 0xFFFFFFFF) + (t >> 32) * 5          # < 2^32 + 25
    t = tl.where(t >= 0xFFFFFFFB, t - 0xFFFFFFFB, t)
    return t.to(tl.uint32)


@triton.jit
def _s32_sqn(x, N: tl.constexpr):
    for _i in tl.static_range(N):
        x = _s32_mulmod(x, x)
    return x


@triton.jit
def _s32_pow_e(x):
    """x^(2^30 - 1): addition chain 1, 2, 3, 6, 12, 15, 30 (35 modular multiplies)."""
    x2 = _s32_mulmod(_s32_mulmod(x, x), x)            # 2^2 - 1
    x3 = _s32_mulmod(_s32_mulmod(x2, x2), x)          # 2^3 - 1
    x6 = _s32_mulmod(_s32_sqn(x3, 3), x3)             # 2^6 - 1
    x12 = _s32_mulmod(_s32_sqn(x6, 6), x6)            # 2^12 - 1
    x15 = _s32_mulmod(_s32_sqn(x12, 3), x3)           # 2^15 - 1
    return _s32_mulmod(_s32_sqn(x15, 15), x15)        # 2^30 - 1


@triton.jit
def _s32_sigma(y):
    s = _s32_mulmod(y, y)
    neg = tl.where(s == 0, s, 0xFFFFFFFB - s)
    x = tl.where((y & 1) == 0, s, neg)
    return tl.where(y >= 0xFFFFFFFB, y, x)


@triton.jit
def _s32_sigma_inv(x):
    r = _s32_pow_e(x)
    qr = _s32_mulmod(r, r) == x
    r2 = tl.where(r == 0, r, 0xFFFFFFFB - r)
    re = tl.where((r & 1) == 0, r, 0xFFFFFFFB - r)
    ro = tl.where((r2 & 1) == 1, r2, 0xFFFFFFFB - r2)
    return tl.where(x >= 0xFFFFFFFB, x, tl.where(qr, re, ro))


@triton.jit
def _s32_ks(seed, widx):
    return _s32_fmix(seed ^ (widx.to(tl.uint32) * 0x9E3779B9))
'''

TRITON_SRC = "import triton\nimport triton.language as tl\n" + TRITON_HELPERS + r'''

@triton.jit(do_not_specialize=['n', 'seed', 'i0'])
def _s32_decode_kernel(src, dst, n, seed, i0, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    y = tl.load(src + i, mask=m, other=0).to(tl.uint32, bitcast=True)
    x = _s32_sigma(y) ^ _s32_ks(seed.to(tl.uint32), i + i0)
    tl.store(dst + i, x.to(tl.int32, bitcast=True), mask=m)


@triton.jit(do_not_specialize=['n', 'seed', 'i0'])
def _s32_encode_kernel(src, dst, n, seed, i0, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(src + i, mask=m, other=0).to(tl.uint32, bitcast=True)
    y = _s32_sigma_inv(x ^ _s32_ks(seed.to(tl.uint32), i + i0))
    tl.store(dst + i, y.to(tl.int32, bitcast=True), mask=m)
'''

_MOD = None


def load():
    global _MOD
    if _MOD is None:
        from schemes.memory_link_exhaustion.inference.patches import _write_and_import
        _MOD = _write_and_import("sloth32", TRITON_SRC)
    return _MOD


def decode_to(src, dst, seed: int, i0: int = 0, block: int = 1024):
    import torch
    s, d = src.view(torch.int32).view(-1), dst.view(torch.int32).view(-1)
    n = s.numel()
    load()._s32_decode_kernel[((n + block - 1) // block,)](s, d, n, seed & M32, i0, BLOCK=block, num_warps=4)


def encode_to(src, dst, seed: int, i0: int = 0, block: int = 512):
    import torch
    s, d = src.view(torch.int32).view(-1), dst.view(torch.int32).view(-1)
    n = s.numel()
    load()._s32_encode_kernel[((n + block - 1) // block,)](s, d, n, seed & M32, i0, BLOCK=block, num_warps=4)
