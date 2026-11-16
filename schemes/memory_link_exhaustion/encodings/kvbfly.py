"""Butterfly KV encoding (VARENA_KVENC=bfly).

The cheap KV keystream (one fmix per 16-bit element) lets a zero-plaintext cheater regenerate any KV
byte as fast as it could read it, so it can drop the whole KV region. Here KV uses the W butterfly
network (ALT1 MIX=7, Philox-2 leaves, ADD-FF at stage L-4; constants from `bfly_ref`) on small
super-chunks that line up with vLLM's attention tiles:

  chunk = (layer, block, kv head, K|V, tch): TSC = 2^(L-4) consecutive tokens x 128 dims of one head
          = 2^L units of 16 B (L=8: 16 tokens, 256 units, 4 KiB)
  unit  u = (token % TSC) * 16 + dim // 8;   word w = (dim % 8) // 2;   half = dim % 2
  s     = block << 16 | head << 8 | kv << 7 | tch          seed = the layer's KV key (32-bit)
  stored element = plain XOR low/high half of word w of ks(s, u)

Every unit depends on all 2^L leaves, so producing any 16 B of a chunk costs the whole chunk: a
16-B chase probe into a dropped KV page costs the cheater a full 4 KiB network. The honest prover
decodes each attention tile (16 or 32 tokens = 1 or 2 chunks per kv head) inside the attention
kernel, and encodes one token row (all heads, K and V: 2 * heads chunks) in the cache write.

Physical page layout (TRITON_ATTN, one layer's block, 64 KiB): [token 32][head 4][K 128 | V 128] bf16,
so page uint32 word j = tok * {NHEADS * 128} + head * 128 + kv * 64 + dg * 4 + w  (dg = dim // 8).
"""
from __future__ import annotations

import os

import numpy as np

from schemes.memory_link_exhaustion.encodings.bfly_ref import CX, KA, RHO, RWORD

KVENC = os.environ.get("VARENA_KVENC", "fmix")
KV_L = int(os.environ.get("VARENA_KV_BFLY_L", "8"))
KV_XB = int(os.environ.get("VARENA_KV_XB", "2"))     # Philox rounds per leaf
# KV page geometry: one layer's block of BLOCK_TOKENS tokens x NHEADS kv heads x (K, V) x HEAD dims = 64 KiB
# (Qwen2.5-7B: 32 tokens x 4 heads; Qwen2.5-32B: 16 tokens x 8 heads). Set from the model by the launchers.
HEAD = 128
NHEADS = int(os.environ.get("VARENA_KV_HEADS", "4"))
BLOCK_TOKENS = int(os.environ.get("VARENA_KV_BLOCK", str(64 * 1024 // (NHEADS * 2 * HEAD * 2))))
PAGE_WORDS = BLOCK_TOKENS * NHEADS * 2 * HEAD // 2
MASK = 0xFFFFFFFF


def tsc(L: int = KV_L) -> int:
    return 1 << (L - 4)


def chunk_s(block: int, head: int, kv: int, tch: int) -> int:
    return (block << 16) | (head << 8) | (kv << 7) | tch


def _rotl(x, r):
    return ((x << np.uint64(r)) | (x >> np.uint64(32 - r))) & np.uint64(MASK)


def chunk_ks(seed: int, s: int, L: int = KV_L, XB: int = KV_XB) -> np.ndarray:
    """uint32 [2^L, 4]: keystream of super-chunk s (vectorised numpy; L=11 equals bfly_ref)."""
    M = 1 << L
    m = np.uint64(MASK)
    c0 = np.full(M, s & MASK, np.uint64)
    c1 = np.full(M, (s >> 32) & MASK, np.uint64)
    c2 = np.arange(M, dtype=np.uint64)
    c3 = np.full(M, 0x1234, np.uint64)
    k0, k1 = np.uint64(seed & MASK), np.uint64((seed & MASK) ^ 0x9E37)
    for _ in range(XB):
        p0 = c0 * np.uint64(0xD2511F53)
        p1 = c2 * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = ((p1 >> np.uint64(32)) ^ c1 ^ k0) & m, p1 & m, ((p0 >> np.uint64(32)) ^ c3 ^ k1) & m, p0 & m
        k0 = (k0 + np.uint64(0x9E3779B9)) & m
        k1 = (k1 + np.uint64(0xBB67AE85)) & m
    x = np.stack([c0, c1, c2, c3], 1)
    idx = np.arange(M)
    mid = None
    for l in range(L):
        if l == L - 4:
            mid = x.copy()
        i = idx[((idx >> l) & 1) == 0]
        j = i | (1 << l)
        a, b = x[i].copy(), x[j].copy()
        rw = int(RWORD[l])
        ka, cx = KA[l].astype(np.uint64), CX[l].astype(np.uint64)

        def F(v):
            f = v.copy()
            f[:, rw] = _rotl(v[:, rw], int(RHO[l, rw]))
            return f

        if l & 1:            # b first
            b = b ^ F(a) ^ cx
            a = (a * ka + b[:, [1, 2, 3, 0]]) & m
        else:
            a = (a * ka + b[:, [1, 2, 3, 0]]) & m
            b = b ^ F(a) ^ cx
        x[i], x[j] = a, b
    return ((x + mid) & m).astype(np.uint32)

# --------------------------------------------------------------------------- butterfly arena filler
# (VARENA_FILLER=bfly) 4 KiB chunk c of global page p = the L=8 network's keystream of chunk id
# (FILL_TAG << 56) | (p << 4) | c under the filler seed: the encoding of zeros, like a never-written KV page
FILL_L = 8
FILL_TAG = 0xF1


def filler_seed(fkey: int) -> int:
    return ((int(fkey) * 0x9E3779B1) ^ 0x46494C4C) & MASK


def filler_s(page, sub):
    return (np.uint64(FILL_TAG) << np.uint64(56)) | (np.asarray(page, dtype=np.uint64) << np.uint64(4)) | \
        np.asarray(sub, dtype=np.uint64)


def filler_units(fkey: int, pages, subs) -> np.ndarray:
    """uint32 [n, 1024]: 4 KiB chunks subs[i] of filler pages pages[i]."""
    ss = filler_s(np.atleast_1d(pages), np.atleast_1d(subs))
    return chunk_ks_many(filler_seed(fkey), ss, FILL_L, KV_XB).reshape(-1, 1024)


def filler_page_words(fkey: int, pages) -> np.ndarray:
    """uint32 [n, 16384]: whole filler pages."""
    pages = np.atleast_1d(np.asarray(pages, dtype=np.uint64))
    return filler_units(fkey, np.repeat(pages, 16), np.tile(np.arange(16, dtype=np.uint64), pages.size)).reshape(
        pages.size, 16384)


def chunk_ks_many(seed: int, ss, L: int = KV_L, XB: int = KV_XB) -> np.ndarray:
    """uint32 [S, 2^L, 4]: chunk_ks for many super-chunks at once (same network, vectorised over chunks)."""
    ss = np.asarray(ss, dtype=np.uint64).reshape(-1)
    S, M = ss.size, 1 << L
    m = np.uint64(MASK)
    c0 = np.repeat((ss & m)[:, None], M, 1)
    c1 = np.repeat(((ss >> np.uint64(32)) & m)[:, None], M, 1)
    c2 = np.broadcast_to(np.arange(M, dtype=np.uint64), (S, M)).copy()
    c3 = np.full((S, M), 0x1234, np.uint64)
    k0, k1 = np.uint64(seed & MASK), np.uint64((seed & MASK) ^ 0x9E37)
    for _ in range(XB):
        p0 = c0 * np.uint64(0xD2511F53)
        p1 = c2 * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = ((p1 >> np.uint64(32)) ^ c1 ^ k0) & m, p1 & m, ((p0 >> np.uint64(32)) ^ c3 ^ k1) & m, p0 & m
        k0 = (k0 + np.uint64(0x9E3779B9)) & m
        k1 = (k1 + np.uint64(0xBB67AE85)) & m
    x = np.stack([c0, c1, c2, c3], 2)                    # [S, M, 4]
    idx = np.arange(M)
    mid = None
    for l in range(L):
        if l == L - 4:
            mid = x.copy()
        i = idx[((idx >> l) & 1) == 0]
        j = i | (1 << l)
        a, b = x[:, i].copy(), x[:, j].copy()
        rw = int(RWORD[l])
        ka, cx = KA[l].astype(np.uint64), CX[l].astype(np.uint64)

        def F(v):
            f = v.copy()
            f[..., rw] = _rotl(v[..., rw], int(RHO[l, rw]))
            return f

        if l & 1:
            b = b ^ F(a) ^ cx
            a = (a * ka + b[..., [1, 2, 3, 0]]) & m
        else:
            a = (a * ka + b[..., [1, 2, 3, 0]]) & m
            b = b ^ F(a) ^ cx
        x[:, i], x[:, j] = a, b
    return ((x + mid) & m).astype(np.uint32)



def page_ks(seed: int, block: int, L: int = KV_L, XB: int = KV_XB) -> np.ndarray:
    """uint32 [PAGE_WORDS]: keystream words of one KV page in physical word order."""
    T = tsc(L)
    out = np.empty((BLOCK_TOKENS, NHEADS, 2, 16, 4), np.uint32)
    for h in range(NHEADS):
        for kv in range(2):
            for t in range(BLOCK_TOKENS // T):
                ks = chunk_ks(seed, chunk_s(block, h, kv, t), L, XB).reshape(T, 16, 4)
                out[t * T:(t + 1) * T, h, kv] = ks
    return out.reshape(-1)


def encode_page_u16(words_u16: np.ndarray, seed: int, block: int, L: int = KV_L) -> np.ndarray:
    """XOR one page (uint16 [2 * PAGE_WORDS], physical order) with its butterfly keystream."""
    w = np.ascontiguousarray(words_u16, dtype=np.uint16).view(np.uint32) ^ page_ks(seed, block, L)
    return w.view(np.uint16)


# --------------------------------------------------------------------------- Triton source
def _mix_src(l: int, ind: str) -> list[str]:
    """Source lines of stage l on word tensors a0..a3, b0..b3 (constants inlined)."""
    rw = int(RWORD[l])
    ka = [int(KA[l, w]) & MASK for w in range(4)]
    cx = [int(CX[l, w]) & MASK for w in range(4)]
    r = int(RHO[l, rw])
    f = [f"a{w}" for w in range(4)]
    f[rw] = f"((a{rw} << {r}) | (a{rw} >> {32 - r}))"
    xor_b = [f"{ind}b{w} = b{w} ^ {f[w]} ^ {cx[w]}" for w in range(4)]
    add_a = [f"{ind}a{w} = a{w} * {ka[w]} + b{(w + 1) % 4}" for w in range(4)]
    if l & 1:
        # b first: F uses the old a, then a uses the new b
        return xor_b + add_a
    # a first: all four a updates use the old b; rotate/xor uses the new a
    return add_a + xor_b


def triton_src(L: int = KV_L, XB: int = KV_XB) -> str:
    M, T = 1 << L, tsc(L)
    I = "    "
    s = [f'''
# ---- generated by schemes.memory_link_exhaustion.encodings.kvbfly.triton_src(L={L}): butterfly KV keystream ----
KVBF_L = tl.constexpr({L})
KVBF_M = tl.constexpr({M})
KVBF_T = tl.constexpr({T})


@triton.jit
def _kvbf_split(x, NB: tl.constexpr, A: tl.constexpr, B: tl.constexpr):
    y = tl.permute(tl.reshape(x, (NB, A, 2, B)), (0, 1, 3, 2))
    return tl.split(y)


@triton.jit
def _kvbf_merge(a, b, NB: tl.constexpr):
    y = tl.permute(tl.join(a, b), (0, 1, 3, 2))
    return tl.reshape(y, (NB, {M}))


@triton.jit
def _kvbf_net(sv, seed, NB: tl.constexpr):
    """Keystream words (4 x uint32 [NB, M]) of NB super-chunks, sv: uint32 [NB] chunk ids."""
    z = tl.zeros((NB, {M}), tl.uint32)
    c0 = sv[:, None] + z
    c1 = z
    c2 = tl.arange(0, {M}).to(tl.uint32)[None, :] + z
    c3 = z + 0x1234
    k0 = seed.to(tl.uint32)
    k1 = k0 ^ 0x9E37
    m0 = z + 0xD2511F53
    m1 = z + 0xCD9E8D57
    for _r in tl.static_range({XB}):
        h0 = tl.umulhi(c0, m0)
        l0 = c0 * m0
        h1 = tl.umulhi(c2, m1)
        l1 = c2 * m1
        c0 = h1 ^ c1 ^ k0
        c1 = l1
        c2 = h0 ^ c3 ^ k1
        c3 = l0
        k0 = k0 + 0x9E3779B9
        k1 = k1 + 0xBB67AE85
    x0, x1, x2, x3 = c0, c1, c2, c3
''']
    for l in range(L):
        B, A = 1 << l, M >> (l + 1)
        if l == L - 4:
            s.append(f"{I}y0, y1, y2, y3 = x0, x1, x2, x3\n")
        s.append(f"{I}# stage {l}\n")
        for w in range(4):
            s.append(f"{I}a{w}, b{w} = _kvbf_split(x{w}, NB, {A}, {B})\n")
        s += [ln + "\n" for ln in _mix_src(l, I)]
        for w in range(4):
            s.append(f"{I}x{w} = _kvbf_merge(a{w}, b{w}, NB)\n")
    s.append(f"{I}return x0 + y0, x1 + y1, x2 + y2, x3 + y3\n")

    s.append(f'''

@triton.jit
def _kvbf_u16(w0, w1, w2, w3, R: tl.constexpr):
    """[R, 16] uint32 unit words -> [R, 128] uint16 element keystream (dim = dg*8 + w*2 + half)."""
    q0 = tl.join(w0 & 0xFFFF, w0 >> 16)
    q1 = tl.join(w1 & 0xFFFF, w1 >> 16)
    q2 = tl.join(w2 & 0xFFFF, w2 >> 16)
    q3 = tl.join(w3 & 0xFFFF, w3 >> 16)
    r0 = tl.permute(tl.join(q0, q1), (0, 1, 3, 2))       # [R, 16, l, half]
    r1 = tl.permute(tl.join(q2, q3), (0, 1, 3, 2))
    zz = tl.permute(tl.join(r0, r1), (0, 1, 4, 2, 3))     # [R, 16, h, l, half], w = 2h + l
    return tl.reshape(zz, (R, 128)).to(tl.uint16)


@triton.jit
def _kvbf_tile_bt(seed, bt, j, head, kv, lim, TILE: tl.constexpr, BS: tl.constexpr):
    """uint16 [TILE, 128]: keystream of tile j (tokens j*TILE .. j*TILE+TILE) of one (head, K|V) of a sequence whose
    block-table row starts at bt; the tile may span several blocks (vLLM's own tile sizes, VARENA_ATT_CLAMP=0) or
    part of one. Chunks at or past lim (the sequence's prefix length) read block 0 and are masked by the caller."""
    tl.static_assert(TILE % {T} == 0)
    tl.static_assert(BS % {T} == 0)
    NB: tl.constexpr = TILE // {T}
    tok = j * TILE + tl.arange(0, NB) * {T}
    blk = tl.load(bt + tok // BS, mask=tok < lim, other=0)
    tch = ((tok % BS) // {T}).to(tl.uint32)
    sv = (blk.to(tl.uint32) << 16) | (head.to(tl.uint32) << 8) | (kv << 7) | tch
    w0, w1, w2, w3 = _kvbf_net(sv, seed, NB)
    return _kvbf_u16(tl.reshape(w0, (TILE, 16)), tl.reshape(w1, (TILE, 16)),
                     tl.reshape(w2, (TILE, 16)), tl.reshape(w3, (TILE, 16)), TILE)


@triton.jit
def _kvbf_rows(seed, blk, tok, NH: tl.constexpr):
    """Keystream of one token row of a block for every kv head: (K [NH*128], V [NH*128]) uint16."""
    NB: tl.constexpr = 2 * NH
    i = tl.arange(0, NB).to(tl.uint32)
    tch = (tok // {T}).to(tl.uint32)
    sv = (blk.to(tl.uint32) << 16) | ((i // 2) << 8) | ((i % 2) << 7) | tch
    w0, w1, w2, w3 = _kvbf_net(sv, seed, NB)
    sel = (tl.arange(0, {T}) == (tok % {T}))[None, :, None]
    w0 = tl.sum(tl.where(sel, tl.reshape(w0, (NB, {T}, 16)), 0), 1)
    w1 = tl.sum(tl.where(sel, tl.reshape(w1, (NB, {T}, 16)), 0), 1)
    w2 = tl.sum(tl.where(sel, tl.reshape(w2, (NB, {T}, 16)), 0), 1)
    w3 = tl.sum(tl.where(sel, tl.reshape(w3, (NB, {T}, 16)), 0), 1)
    e = tl.reshape(_kvbf_u16(w0, w1, w2, w3, NB), (NH, 2, 128))
    k, v = tl.split(tl.permute(e, (0, 2, 1)))
    return tl.reshape(k, (NH * 128,)), tl.reshape(v, (NH * 128,))


@triton.jit
def _kvbf_fmix(x):
    x = x * 0x85EBCA6B
    x = x ^ (x >> 13)
    x = x * 0xC2B2AE35
    x = x ^ (x >> 16)
    return x


@triton.jit
def _kvbf_acc(word, j, nu, d0, d1):
    v = _kvbf_fmix(word ^ (j * 0x9E3779B9) ^ nu)
    return d0 + v, d1 + v * (2 * j + 1)


@triton.jit
def _kvbf_page_words(blk, hk, seed):
    """Keystream of the (head, K|V) = (hk // 2, hk % 2) part of a page: 4 x uint32 [32 tok, 16 dg]
    and the page word index of word 0 of each unit."""
    NB: tl.constexpr = {BLOCK_TOKENS // T}
    sv = (blk.to(tl.uint32) << 16) | ((hk // 2).to(tl.uint32) << 8) | ((hk % 2).to(tl.uint32) << 7) \
        | tl.arange(0, NB).to(tl.uint32)
    w0, w1, w2, w3 = _kvbf_net(sv, seed, NB)
    tok = tl.arange(0, {BLOCK_TOKENS}).to(tl.uint32)[:, None]
    dg = tl.arange(0, 16).to(tl.uint32)[None, :]
    jb = tok * {NHEADS * 128} + (hk // 2).to(tl.uint32) * 128 + (hk % 2).to(tl.uint32) * 64 + dg * 4
    return (tl.reshape(w0, ({BLOCK_TOKENS}, 16)), tl.reshape(w1, ({BLOCK_TOKENS}, 16)), tl.reshape(w2, ({BLOCK_TOKENS}, 16)),
            tl.reshape(w3, ({BLOCK_TOKENS}, 16)), jb)


@triton.jit
def _kvbf_page_hash(blk, seed, nonce):
    """Digest (as kernels._hash_page) of an all-zero-plaintext KV page = its keystream words."""
    nu = nonce.to(tl.uint32)
    d0 = tl.zeros(({BLOCK_TOKENS}, 16), tl.uint32)
    d1 = tl.zeros(({BLOCK_TOKENS}, 16), tl.uint32)
    for hk in tl.static_range({2 * NHEADS}):
        w0, w1, w2, w3, jb = _kvbf_page_words(blk, tl.full((), hk, tl.int32), seed)
        d0, d1 = _kvbf_acc(w0, jb, nu, d0, d1)
        d0, d1 = _kvbf_acc(w1, jb + 1, nu, d0, d1)
        d0, d1 = _kvbf_acc(w2, jb + 2, nu, d0, d1)
        d0, d1 = _kvbf_acc(w3, jb + 3, nu, d0, d1)
    return tl.sum(tl.sum(d0, 1), 0), tl.sum(tl.sum(d1, 1), 0)


@triton.jit
def _kvbf_unit_digest(blk, seed, sub, nu):
    """Fan-out digest (word index j within the unit) of 4 KiB unit `sub` (page words sub*1024 ..: tokens
    2sub, 2sub+1, every head, K and V) of an all-zero-plaintext KV page: 8 chunk networks."""
    i = tl.arange(0, {2 * NHEADS}).to(tl.uint32)
    t0 = ({8 // NHEADS} * sub).to(tl.uint32)               # tokens per 4 KiB unit = 1024 words / (NHEADS * 128)
    sv = (blk.to(tl.uint32) << 16) | ((i // 2) << 8) | ((i % 2) << 7) | (t0 // {T})
    w0, w1, w2, w3 = _kvbf_net(sv, seed, {2 * NHEADS})
    trel = tl.arange(0, {T}).to(tl.uint32)[None, :, None] - (t0 % {T})
    inu = trel < {8 // NHEADS}
    jb = trel * {NHEADS * 128} + (i // 2)[:, None, None] * 128 + (i % 2)[:, None, None] * 64 \\
        + tl.arange(0, 16).to(tl.uint32)[None, None, :] * 4
    z = tl.zeros(({2 * NHEADS}, {T}, 16), tl.uint32)
    e0, e1 = _kvbf_acc(tl.reshape(w0, ({2 * NHEADS}, {T}, 16)), jb, nu, z, z)
    e0, e1 = _kvbf_acc(tl.reshape(w1, ({2 * NHEADS}, {T}, 16)), jb + 1, nu, e0, e1)
    e0, e1 = _kvbf_acc(tl.reshape(w2, ({2 * NHEADS}, {T}, 16)), jb + 2, nu, e0, e1)
    e0, e1 = _kvbf_acc(tl.reshape(w3, ({2 * NHEADS}, {T}, 16)), jb + 3, nu, e0, e1)
    e0 = tl.where(inu, e0, 0)
    e1 = tl.where(inu, e1, 0)
    return tl.sum(tl.sum(tl.sum(e0, 2), 1), 0), tl.sum(tl.sum(tl.sum(e1, 2), 1), 0)


@triton.jit(do_not_specialize=['seed', 'blk0'])
def _kvbf_xor_kernel(ptr, seed, blk0, PAGE_W: tl.constexpr):
    """ptr: int32 view of pages; grid (n_pages, heads * 2): XOR page (blk0 + pid) keystream in place."""
    blk = blk0 + tl.program_id(0)
    hk = tl.program_id(1)
    w0, w1, w2, w3, jb = _kvbf_page_words(blk, hk, seed)
    base = ptr + tl.program_id(0).to(tl.int64) * PAGE_W
    jb = jb.to(tl.int64)
    tl.store(base + jb, tl.load(base + jb) ^ w0.to(tl.int32, bitcast=True))
    tl.store(base + jb + 1, tl.load(base + jb + 1) ^ w1.to(tl.int32, bitcast=True))
    tl.store(base + jb + 2, tl.load(base + jb + 2) ^ w2.to(tl.int32, bitcast=True))
    tl.store(base + jb + 3, tl.load(base + jb + 3) ^ w3.to(tl.int32, bitcast=True))
''')
    return "".join(s)


_MOD = {}

_FILL_SRC = r'''

@triton.jit(do_not_specialize=['first_page', 'seed'])
def _kvbf_fill_kernel(dst, page_map, pages, first_page, seed, LIST: tl.constexpr):
    """Butterfly filler (kvbfly.filler_units): grid (n pages, 16 chunks of 4 KiB); dst int32 words, page_map
    int32-word offsets (LIST), else consecutive pages from first_page."""
    pi = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    if LIST:
        page = tl.load(pages + pi).to(tl.int64)
        off = tl.load(page_map + page).to(tl.int64)
    else:
        page = first_page.to(tl.int64) + pi
        off = pi * 16384
    s = (page << 4) | c
    z = tl.zeros((1,), tl.uint32)
    lo = z + (s & 0xFFFFFFFF).to(tl.uint32)
    hi = z + ((s >> 32) | 0xF1000000).to(tl.uint32)
    w0, w1, w2, w3 = _kvbf_net_hi(lo, hi, seed, 1)
    u = tl.arange(0, {M}).to(tl.int64)
    b = dst + off + c * 1024 + u * 4
    tl.store(b, tl.reshape(w0, ({M},)).to(tl.int32, bitcast=True))
    tl.store(b + 1, tl.reshape(w1, ({M},)).to(tl.int32, bitcast=True))
    tl.store(b + 2, tl.reshape(w2, ({M},)).to(tl.int32, bitcast=True))
    tl.store(b + 3, tl.reshape(w3, ({M},)).to(tl.int32, bitcast=True))
'''


def _with_hi(src: str) -> str:
    """Append _kvbf_net_hi(sv, hi, seed, NB): _kvbf_net with the chunk id's high word (c1 = hi)."""
    i = src.index("@triton.jit\ndef _kvbf_net(sv, seed, NB: tl.constexpr):")
    j = src.index("\n\n\n@triton.jit", i + 10)
    body = src[i:j].replace("def _kvbf_net(sv, seed, NB: tl.constexpr):", "def _kvbf_net_hi(sv, hi, seed, NB: tl.constexpr):")
    body = body.replace("    c1 = z\n", "    c1 = hi.to(tl.uint32) + z\n", 1)
    assert "c1 = hi" in body
    return src + "\n\n" + body + "\n"


def load(L: int = KV_L, XB: int = KV_XB):
    """Write the generated Triton helpers for L to a real file and import it (Triton needs source)."""
    if (L, XB) in _MOD:
        return _MOD[(L, XB)]
    import hashlib
    import importlib.util
    import sys
    src = "import triton\nimport triton.language as tl\n" + _with_hi(triton_src(L, XB)) + _FILL_SRC.replace("{M}", str(1 << L))
    d = os.environ.get("VARENA_GEN", "/tmp/varena_gen")
    os.makedirs(d, exist_ok=True)
    name = f"kvbf_{L}_{XB}_{hashlib.sha1(src.encode()).hexdigest()[:10]}"
    path = os.path.join(d, name + ".py")
    if not os.path.exists(path):
        tmp = path + f".{os.getpid()}"
        open(tmp, "w").write(src)
        os.replace(tmp, path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MOD[(L, XB)] = mod
    return mod


def xor_pages(pages_i32, seed: int, blk0: int = 0, L: int = KV_L, XB: int = KV_XB):
    """XOR the butterfly KV keystream into consecutive pages (int32 tensor [n * PAGE_WORDS]) whose
    first page is block blk0 of the layer keyed `seed`."""
    mod = load(L, XB)
    n = pages_i32.numel() // PAGE_WORDS
    if n:
        mod._kvbf_xor_kernel[(n, 2 * NHEADS)](pages_i32, seed & MASK, blk0, PAGE_W=PAGE_WORDS, num_warps=4)


def fill_range(dst_bytes, first_page: int, fkey: int):
    """dst_bytes: page-aligned uint8 view of global pages first_page.. (butterfly filler)."""
    import torch
    n = dst_bytes.numel() // 65536
    if n:
        d = dst_bytes.view(torch.int32)
        load(FILL_L, KV_XB)._kvbf_fill_kernel[(n, 16)](d, d, d, int(first_page), filler_seed(fkey), LIST=False, num_warps=4)


def fill_pages(base_i32, page_map, pages, fkey: int):
    """Refill the listed global pages (int64 tensor) of a page-mapped buffer with butterfly filler."""
    n = pages.numel()
    if n:
        load(FILL_L, KV_XB)._kvbf_fill_kernel[(n, 16)](base_i32, page_map, pages, 0, filler_seed(fkey), LIST=True,
                                                     num_warps=4)
