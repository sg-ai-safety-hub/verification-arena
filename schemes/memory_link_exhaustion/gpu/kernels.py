"""Triton kernels for the sealed workload, the verifier and the PCIe noise stream.

Every keyed function here has a NumPy twin in ref.py, and schemes/memory_link_exhaustion/tests/test_kernels.py
checks them bit for bit (on CPU through TRITON_INTERPRET=1, or on a GPU).
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

from schemes.memory_link_exhaustion.encodings import ref

PAGE = ref.PAGE
PAGE_WORDS = ref.PAGE_WORDS
# Triton's CPU interpreter gets bf16 tl.dot wrong; tests then upcast to fp32 first.
F32DOT = bool(os.environ.get("TRITON_INTERPRET"))


# --------------------------------------------------------------------------- helpers
@triton.jit
def _fmix(x):
    x = x * 0x85EBCA6B
    x = x ^ (x >> 13)
    x = x * 0xC2B2AE35
    x = x ^ (x >> 16)
    return x


@triton.jit
def _dec16(raw, off, key):
    """XOR raw 16-bit values with the data keystream at element offsets `off`."""
    ks = (_fmix(off.to(tl.uint32) ^ key) & 0xFFFF).to(tl.uint16)
    return (raw.to(tl.uint16, bitcast=True) ^ ks).to(tl.bfloat16, bitcast=True)


@triton.jit
def _wrow(rows, key):
    return _fmix(rows.to(tl.uint32) ^ key)


@triton.jit
def _wcol(cols, key):
    return _fmix(_fmix(cols.to(tl.uint32) ^ key) ^ 0x7F4A7C15)


@triton.jit
def _xks(raw, ks):
    """raw (16-bit) XOR low16(ks) -> bf16."""
    return (raw.to(tl.uint16, bitcast=True) ^ (ks & 0xFFFF).to(tl.uint16)).to(tl.bfloat16, bitcast=True)


def _u32(x: int) -> int:
    return int(x) & 0xFFFFFFFF


# --------------------------------------------------------------------------- xor16
@triton.jit(do_not_specialize=['n', 'j0', 'key'])
def _xor16_kernel(src, dst, n, j0, key, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(src + offs, mask=m)
    ks = (_fmix((offs + j0).to(tl.uint32) ^ key.to(tl.uint32)) & 0xFFFF).to(tl.uint16)
    y = x.to(tl.uint16, bitcast=True) ^ ks
    tl.store(dst + offs, y.to(tl.int16, bitcast=True), mask=m)


def xor16(src: torch.Tensor, dst: torch.Tensor, key: int, j0: int = 0):
    """dst = src XOR keystream(key, j0 + i). Both viewed as int16; works in place."""
    s = src.reshape(-1).view(torch.int16)
    d = dst.reshape(-1).view(torch.int16)
    n = s.numel()
    BLOCK = 4096
    _xor16_kernel[(triton.cdiv(n, BLOCK),)](s, d, n, j0, _u32(key), BLOCK=BLOCK)
    return dst


@triton.jit(do_not_specialize=['n', 'j0', 'ncols', 'key'])
def _xor16w_kernel(src, dst, n, j0, ncols, key, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(src + offs, mask=m)
    j = offs + j0
    keyu = key.to(tl.uint32)
    ks = _wrow(j // ncols, keyu) ^ _wcol(j % ncols, keyu)
    y = x.to(tl.uint16, bitcast=True) ^ (ks & 0xFFFF).to(tl.uint16)
    tl.store(dst + offs, y.to(tl.int16, bitcast=True), mask=m)


def xor16_w(src: torch.Tensor, dst: torch.Tensor, key: int, ncols: int, j0: int = 0):
    """Weight encode/decode: dst = src XOR weight keystream (row-major, ncols columns)."""
    s = src.reshape(-1).view(torch.int16)
    d = dst.reshape(-1).view(torch.int16)
    n = s.numel()
    BLOCK = 4096
    _xor16w_kernel[(triton.cdiv(n, BLOCK),)](s, d, n, j0, ncols, _u32(key), BLOCK=BLOCK)
    return dst


# --------------------------------------------------------------------------- matmul
@triton.jit(do_not_specialize=['M', 'N', 'key'])
def _mm_kernel(a, b, c, M, N, R, sam, sar, sbr, sbc, key, r_per,
               ENC: tl.constexpr, SPLIT: tl.constexpr, WT: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr, F32: tl.constexpr):
    """C[M,N] = A[M,R] @ B[R,N]; B(r, n) lives at element offset r*sbr + n*sbc.

    With ENC the B values are stored encoded under `key` and decoded on load.
    SPLIT > 1 writes fp32 partials [SPLIT, M, N] (reduced by the caller, which
    keeps the result deterministic)."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    ps = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    r0 = ps * r_per
    r1 = tl.minimum(r0 + r_per, R)
    keyu = key.to(tl.uint32)
    for r in range(r0, r1, BR):
        rk = r + tl.arange(0, BR)
        am = (rm[:, None] < M) & (rk[None, :] < r1)
        av = tl.load(a + rm[:, None] * sam + rk[None, :] * sar, mask=am, other=0.0)
        boff = rk[:, None] * sbr + rn[None, :] * sbc
        bm = (rk[:, None] < r1) & (rn[None, :] < N)
        bv = tl.load(b + boff, mask=bm, other=0)
        if ENC:
            if WT:      # B(r, n) = W[r, n]
                ks = _wrow(rk, keyu)[:, None] ^ _wcol(rn, keyu)[None, :]
            else:       # B(r, n) = W[n, r]
                ks = _wcol(rk, keyu)[:, None] ^ _wrow(rn, keyu)[None, :]
            bv = tl.where(bm, _xks(bv, ks), 0.0)
        if F32:
            acc = tl.dot(av.to(tl.float32), bv.to(tl.float32), acc)
        else:
            acc = tl.dot(av, bv, acc)
    cm = (rm[:, None] < M) & (rn[None, :] < N)
    if SPLIT == 1:
        tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=cm)
    else:
        tl.store(c + ps * M * N + rm[:, None] * N + rn[None, :], acc, mask=cm)


_SMS = None


def _sms():
    global _SMS
    if _SMS is None:
        _SMS = torch.cuda.get_device_properties(0).multi_processor_count \
            if torch.cuda.is_available() else 8
    return _SMS


def mm_config(M, N, R):
    """Fixed (not autotuned) tile choice, so prover and replay pick the same."""
    BM = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
    BN = 64 if M <= 64 else 128
    BR = 128 if M <= 64 else 64
    tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    split = 1
    while tiles * split < 2 * _sms() and R // (BR * split * 2) >= 4 and split < 16:
        split *= 2
    warps = 4 if BM * BN <= 64 * 64 else 8
    return BM, BN, BR, split, warps


def matmul_bt(x: torch.Tensor, w: torch.Tensor, key: int | None, transpose: bool = False):
    """y = x @ W^T (transpose=False) or y = x @ W (transpose=True).

    W is [N, K] row-major; encoded (int16 storage) when key is not None, plain
    bf16 otherwise. x is [M, K] (or [M, N] when transposed), bf16."""
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    Nw, Kw = w.shape
    M = x.shape[0]
    if not transpose:
        R, N, sbr, sbc = Kw, Nw, 1, Kw
    else:
        R, N, sbr, sbc = Nw, Kw, Kw, 1
    assert x.shape[1] == R, (x.shape, w.shape, transpose)
    BM, BN, BR, split, warps = mm_config(M, N, R)
    r_per = triton.cdiv(triton.cdiv(R, split), BR) * BR
    if split == 1:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    else:
        out = torch.empty((split, M, N), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), split)
    wv = w.view(torch.int16) if key is not None else w
    _mm_kernel[grid](x, wv, out, M, N, R, x.stride(0), x.stride(1), sbr, sbc,
                     _u32(key or 0), r_per, ENC=key is not None, SPLIT=split, WT=transpose,
                     BM=BM, BN=BN, BR=BR, F32=F32DOT, num_warps=warps, num_stages=3)
    if split > 1:
        out = out.sum(0).to(torch.bfloat16)
    return out


# --------------------------------------------------------------------------- embedding
@triton.jit(do_not_specialize=['H', 'key'])
def _gather_kernel(ids, w, out, H, key, ENC: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    cb = tl.program_id(1)
    row = tl.load(ids + i).to(tl.int64)
    c = cb * BLOCK + tl.arange(0, BLOCK)
    m = c < H
    off = row * H + c
    v = tl.load(w + off, mask=m, other=0)
    if ENC:
        keyu = key.to(tl.uint32)
        v = tl.where(m, _xks(v, _wrow(row, keyu) ^ _wcol(c, keyu)), 0.0)
    tl.store(out + i.to(tl.int64) * H + c, v, mask=m)


def embed(ids: torch.Tensor, w: torch.Tensor, key: int | None):
    ids = ids.reshape(-1)
    n, H = ids.numel(), w.shape[1]
    out = torch.empty((n, H), dtype=torch.bfloat16, device=w.device)
    BLOCK = 1024
    wv = w.view(torch.int16) if key is not None else w
    _gather_kernel[(n, triton.cdiv(H, BLOCK))](ids, wv, out, H, _u32(key or 0),
                                               ENC=key is not None, BLOCK=BLOCK)
    return out


# --------------------------------------------------------------------------- KV cache
@triton.jit(do_not_specialize=['b_off', 'n', 'key'])
def _kv_store_kernel(k, v, s_b, s_t, s_h, kv, pos, b_off, n, B, Hkv, T, key,
                     ENC: tl.constexpr, D: tl.constexpr):
    """Write k/v [b, i, h, :] (strided) to slab [2, B, Hkv, T, D] at position pos[b]+i."""
    pid = tl.program_id(0)
    h = pid % Hkv
    i = (pid // Hkv) % n
    bl = pid // (Hkv * n)
    b = bl + b_off
    d = tl.arange(0, D)
    t = tl.load(pos + b) + i
    src = bl * s_b + i * s_t + h * s_h + d
    kk = tl.load(k + src)
    vv = tl.load(v + src)
    ko = ((b * Hkv + h) * T + t) * D + d
    vo = (((B + b) * Hkv + h) * T + t) * D + d
    if ENC:
        keyu = key.to(tl.uint32)
        kk = _dec16(kk, ko, keyu)     # XOR is its own inverse: encode == decode
        vv = _dec16(vv, vo, keyu)
        tl.store(kv + ko, kk.to(tl.int16, bitcast=True))
        tl.store(kv + vo, vv.to(tl.int16, bitcast=True))
    else:
        tl.store(kv + ko, kk)
        tl.store(kv + vo, vv)


def kv_store(k, v, slab, key, pos, b_off=0):
    """k, v: [b, n, Hkv, D] views (last dim contiguous). slab: [2, B, Hkv, T, D]."""
    bsz, n, Hkv, D = k.shape
    _, B, _, T, _ = slab.shape
    assert k.stride(3) == 1 and v.stride() == k.stride()
    sv = slab.view(torch.int16) if key is not None else slab
    _kv_store_kernel[(bsz * n * Hkv,)](k, v, k.stride(0), k.stride(1), k.stride(2), sv, pos,
                                       b_off, n, B, Hkv, T, _u32(key or 0),
                                       ENC=key is not None, D=D)


@triton.jit(do_not_specialize=['key'])
def _attn_decode_kernel(q, kv, o, lens, B, Hkv, T, scale, key, G: tl.constexpr,
                        GP: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
                        ENC: tl.constexpr, F32: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    g = tl.arange(0, GP)
    d = tl.arange(0, D)
    gm = g < G
    qrow = (b * Hkv + h) * G + g
    qv = tl.load(q + qrow[:, None] * D + d[None, :], mask=gm[:, None], other=0.0)
    L = tl.load(lens + b)
    kb = ((b * Hkv + h) * T) * D
    vb = (((B + b) * Hkv + h) * T) * D
    m_i = tl.full((GP,), float("-inf"), tl.float32)
    l_i = tl.zeros((GP,), tl.float32)
    acc = tl.zeros((GP, D), tl.float32)
    keyu = key.to(tl.uint32)
    for t0 in range(0, L, BT):
        t = t0 + tl.arange(0, BT)
        tm = t < L
        ko = kb + t[:, None] * D + d[None, :]
        kt = tl.load(kv + ko, mask=tm[:, None], other=0)
        if ENC:
            kt = tl.where(tm[:, None], _dec16(kt, ko, keyu), 0.0)
        if F32:
            s = tl.dot(qv.to(tl.float32), tl.trans(kt).to(tl.float32)) * scale
        else:
            s = tl.dot(qv, tl.trans(kt)) * scale
        s = tl.where(tm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        vo = vb + t[:, None] * D + d[None, :]
        vt = tl.load(kv + vo, mask=tm[:, None], other=0)
        if ENC:
            vt = tl.where(tm[:, None], _dec16(vt, vo, keyu), 0.0)
        if F32:
            pv = tl.dot(p.to(tl.bfloat16).to(tl.float32), vt.to(tl.float32))
        else:
            pv = tl.dot(p.to(tl.bfloat16), vt)
        acc = acc * alpha[:, None] + pv
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(o + qrow[:, None] * D + d[None, :], out.to(tl.bfloat16), mask=gm[:, None])


def attn_decode(q, slab, key, lens):
    """q: [B, Hq, D] contiguous. slab [2, Bslab, Hkv, T, D]. Returns [B, Hq, D]."""
    Bq, Hq, D = q.shape
    _, B, Hkv, T, _ = slab.shape
    G = Hq // Hkv
    GP = max(16, triton.next_power_of_2(G))
    o = torch.empty_like(q)
    sv = slab.view(torch.int16) if key is not None else slab
    _attn_decode_kernel[(Bq, Hkv)](q, sv, o, lens, B, Hkv, T, 1.0 / math.sqrt(D),
                                   _u32(key or 0), G=G, GP=GP, D=D, BT=64,
                                   ENC=key is not None, F32=F32DOT, num_warps=4)
    return o


# --------------------------------------------------------------------------- filler
@triton.jit(do_not_specialize=['first_page', 'npages', 'fkey'])
def _filler_kernel(dst, first_page, npages, fkey, ROUNDS: tl.constexpr, BLOCK: tl.constexpr):
    """One lane per page: a sequential chain of fmix over the page's words."""
    lp = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = lp < npages
    gp = (lp + first_page).to(tl.int64)
    lo = (gp & 0xFFFFFFFF).to(tl.uint32)
    hi = (gp >> 32).to(tl.uint32)
    s = _fmix(_fmix(fkey.to(tl.uint32) ^ lo) ^ hi ^ 0x9E3779B9)
    base = dst + lp.to(tl.int64) * 16384
    for j in range(16384):
        for _r in tl.static_range(ROUNDS):
            s = _fmix(s + j)
        tl.store(base + j, s.to(tl.int32, bitcast=True), mask=m)


def fill(dst_bytes: torch.Tensor, first_page: int, fkey: int, rounds: int = ref.FILLER_ROUNDS):
    """Fill a page-aligned uint8 view with filler for global pages first_page..."""
    assert dst_bytes.numel() % PAGE == 0
    npages = dst_bytes.numel() // PAGE
    if npages == 0:
        return
    if ref.FILLER == "sloth":
        from schemes.memory_link_exhaustion.encodings import sloth64
        sloth64.fill_range(dst_bytes, first_page, fkey)
        return
    if ref.FILLER == "bfly":
        _kvbfly.fill_range(dst_bytes, first_page, fkey)
        return
    if INDEP:
        d = dst_bytes.view(torch.int32)
        _indep_fill_kernel[(npages,)](d, d, d, first_page, _u32(fkey), LIST=False, HF8=INDEP_HF8,
                                      num_warps=8)
        return
    BLOCK = 128
    _filler_kernel[(triton.cdiv(npages, BLOCK),)](dst_bytes.view(torch.int32), first_page,
                                                  npages, _u32(fkey), ROUNDS=rounds,
                                                  BLOCK=BLOCK, num_warps=4)


# --------------------------------------------------------------------------- page hashing
@triton.jit
def _hash_page(ptr, nonce):
    """Digest of 16384 uint32 words at ptr (any int pointer type)."""
    d0 = tl.zeros((4096,), tl.uint32)
    d1 = tl.zeros((4096,), tl.uint32)
    for c in tl.static_range(4):
        j = c * 4096 + tl.arange(0, 4096)
        w = tl.load(ptr + j).to(tl.uint32, bitcast=True)
        v = _fmix(w ^ (j.to(tl.uint32) * 0x9E3779B9) ^ nonce)
        d0 += v
        d1 += v * (2 * j.to(tl.uint32) + 1)
    return tl.sum(d0, 0), tl.sum(d1, 0)


@triton.jit
def _philox8(c0, c1, c2, c3, k0, k1):
    """Philox4x32, 8 rounds (uint32 tensors of one shape)."""
    m0 = tl.full(c0.shape, 0xD2511F53, tl.uint32)
    m1 = tl.full(c0.shape, 0xCD9E8D57, tl.uint32)
    for _r in tl.static_range(8):
        h0 = tl.umulhi(c0, m0)
        l0 = c0 * m0
        h1 = tl.umulhi(c2, m1)
        l1 = c2 * m1
        n0 = h1 ^ c1 ^ k0
        n2 = h0 ^ c3 ^ k1
        c0, c1, c2, c3 = n0, l1, n2, l0
        k0 = k0 + 0x9E3779B9
        k1 = k1 + 0xBB67AE85
    return c0, c1, c2, c3


@triton.jit
def _indep_unit(q, j, fkey, HF8: tl.constexpr):
    """The 4 words of INDEP unit j (16 B) of global 128 B chunk q (uint64 / uint32 tensors)."""
    qlo = (q & 0xFFFFFFFF).to(tl.uint32)
    qhi = (q >> 32).to(tl.uint32)
    z = tl.zeros(qlo.shape, tl.uint32)
    s0, s1, s2, s3 = z, z, z, z
    k0 = z + fkey
    k1 = z + 0x7F4A7C15
    for i in range(HF8):
        a, b, c, d = _philox8(qlo, qhi, (j + 8 * i).to(tl.uint32), z, k0, k1)
        s0 += a
        s1 += b
        s2 += c
        s3 += d
    return s0, s1, s2, s3


@triton.jit(do_not_specialize=["first_page", "fkey"])
def _indep_fill_kernel(dst, page_map, pages, first_page, fkey, LIST: tl.constexpr, HF8: tl.constexpr):
    """One program per page, one lane per 16 B unit (4096 per page). LIST: pages[] global ids and
    page_map word offsets; else contiguous pages first_page + pid at dst + pid * 16384."""
    pid = tl.program_id(0)
    if LIST:
        gp = tl.load(pages + pid).to(tl.int64)
        ptr = dst + tl.load(page_map + gp)
    else:
        gp = (first_page + pid).to(tl.int64)
        ptr = dst + pid.to(tl.int64) * 16384
    u = tl.arange(0, 4096)
    q = gp * 512 + (u // 8).to(tl.int64)
    j = (u % 8).to(tl.uint32)
    s0, s1, s2, s3 = _indep_unit(q, j, fkey.to(tl.uint32), HF8)
    o = (u // 8) * 32 + (u % 8) * 4
    tl.store(ptr + o, s0.to(tl.int32, bitcast=True))
    tl.store(ptr + o + 1, s1.to(tl.int32, bitcast=True))
    tl.store(ptr + o + 2, s2.to(tl.int32, bitcast=True))
    tl.store(ptr + o + 3, s3.to(tl.int32, bitcast=True))


INDEP = ref.FILLER == "indep"
# butterfly KV (VARENA_KVENC=bfly): a zero-plaintext KV page is its butterfly keystream
from schemes.memory_link_exhaustion.encodings import kvbfly as _kvbfly
KVBF = tl.constexpr(_kvbfly.KVENC == "bfly")
_kvbf_page_hash = _kvbfly.load()._kvbf_page_hash
_kvbf_unit_digest = _kvbfly.load()._kvbf_unit_digest
INDEP_HF8 = ref.INDEP_HF // 8


@triton.jit(do_not_specialize=["nonce"])
def _digest_pages_kernel(base, page_map, pages, nonce, out_d0, out_d1):
    i = tl.program_id(0)
    p = tl.load(pages + i).to(tl.int64)
    woff = tl.load(page_map + p)
    d0, d1 = _hash_page(base + woff, nonce.to(tl.uint32))
    tl.store(out_d0 + i, d0.to(tl.int32, bitcast=True))
    tl.store(out_d1 + i, d1.to(tl.int32, bitcast=True))










def digest_pages(base_i32, page_map, pages: torch.Tensor, nonce: int):
    n = pages.numel()
    d0 = torch.empty(n, dtype=torch.int32, device=base_i32.device)
    d1 = torch.empty_like(d0)
    if n:
        _digest_pages_kernel[(n,)](base_i32, page_map, pages, _u32(nonce), d0, d1, num_warps=8)
    return d0, d1


# --------------------------------------------------------------------------- PCIe noise
@triton.jit(do_not_specialize=["fkey", "n"])
def _fill_pages_kernel(base, page_map, pages, n, fkey, ROUNDS: tl.constexpr, BLOCK: tl.constexpr):
    """Filler for a list of global pages (one lane per page), at page_map addresses."""
    lp = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = lp < n
    gp = tl.load(pages + lp, mask=m, other=0).to(tl.int64)
    woff = tl.load(page_map + gp, mask=m, other=0)
    lo = (gp & 0xFFFFFFFF).to(tl.uint32)
    hi = (gp >> 32).to(tl.uint32)
    s = _fmix(_fmix(fkey.to(tl.uint32) ^ lo) ^ hi ^ 0x9E3779B9)
    ptr = base + woff
    for j in range(16384):
        for _r in tl.static_range(ROUNDS):
            s = _fmix(s + j)
        tl.store(ptr + j, s.to(tl.int32, bitcast=True), mask=m)


def fill_pages(base_i32, page_map, pages: torch.Tensor, fkey: int, rounds: int = ref.FILLER_ROUNDS):
    """Refill the given global pages (int64 tensor on device) with filler."""
    n = pages.numel()
    if n == 0:
        return
    if ref.FILLER == "sloth":
        from schemes.memory_link_exhaustion.encodings import sloth64
        sloth64.fill_pages(base_i32, page_map, pages, fkey)
        return
    if ref.FILLER == "bfly":
        _kvbfly.fill_pages(base_i32, page_map, pages, fkey)
        return
    if INDEP:
        from schemes.memory_link_exhaustion.gpu.cuda import build as _CB
        _CB.load().indep_fill_pages(base_i32, page_map, pages, _u32(fkey), INDEP_HF8)
        return
    BLOCK = 64
    _fill_pages_kernel[(triton.cdiv(n, BLOCK),)](base_i32, page_map, pages, n, _u32(fkey),
                                                 ROUNDS=rounds, BLOCK=BLOCK, num_warps=2)


@triton.jit
def _xor_word_kernel(base, woff, val):
    p = base + woff
    tl.store(p, tl.load(p) ^ val)


def xor_word(base_i32, woff: int, val: int):
    """Selftest corruption: XOR one int32 word at base + woff (any address)."""
    _xor_word_kernel[(1,)](base_i32, woff, val)


# --------------------------------------------------------------------------- cheater
# The cheater (attacks/zero_weight_cheater/cheater.py) keeps every page resident: page kind 0.
CK_RES = 0
