"""Activation-pool layout, the challenge gate and the pre-emptible matmul.

What the final configuration uses from this module:
- `ae_matmul` / `_launch_mm`: vLLM's batch-invariant persistent matmul (same main loop and tl.dot operands, so
  bit-identical) with a gate check at tile boundaries; with VARENA_AE_PREEMPT=1 CTAs exit when a challenge raises the
  gate and the launch resumes after it. The sliced decode-ahead (inference/patches.py, VARENA_WDEC_SLICE_MB) runs on
  it, and `scratch` holds the shared decode scratch.
- `gate_dev` / `gate_set` / position markers: the device gate every challenge graph raises first and drops last
  (also read by the fused gate_up GEMM and the gated vLLM matmul, VARENA_MM_GATE).
- The activation pool (`slot`, `fill_pool`, `pool_granules`, VARENA_ACT_POOL_TOKENS): fixed, 4 KiB-aligned slots whose
  content is the address-keyed keystream (sloth32 with VARENA_AE_S32=1); the mixed chase covers it when it exists.
  Only the zero-weight cheater allocates it in the final configuration; the honest prover runs with plain activations
  and has none.

The layer-level activation encoding this module was written for (VARENA_ACTENC=2) was removed; `ae_add_norm` and
`ae_silu` are its last unused wrappers. Pool layout: an [M, N] bf16 tensor (N % 128 == 0) is stored tiled, chunk
(rb, cb) = rows 16 rb .. 16 rb + 15, columns 128 cb .. 128 cb + 127, one contiguous 4 KiB chunk at element offset
(rb * N/128 + cb) * 2048; chunk at byte address A is keyed by (A >> 12) and fmix(epoch ^ master), so for the
zero-weight model every byte is predictable whichever step wrote it.
"""

from __future__ import annotations

import inspect
import os
import sys

import torch

MASTER = int(os.environ.get("VARENA_ACT_KEY", "0x5A17C0DE"), 0) & 0x7FFFFFFF
_EPOCH = {}

AE_SRC = r'''

@triton.jit
def _ae_seed(epoch_ptr, master):
    e = tl.load(epoch_ptr).to(tl.uint32)
    return _kvbf_fmix(e ^ master.to(tl.uint32))


@triton.jit
def _ae_ks(seed, cid0, ncb, RB: tl.constexpr, CB: tl.constexpr):
    """uint16 [RB*16, CB*128]: keystream of the RB x CB block of chunks whose top-left chunk id is
    cid0 (chunk ids advance by 1 per column block and by ncb per row block)."""
    NB: tl.constexpr = RB * CB
    i = tl.arange(0, NB)
    sv = (cid0 + (i // CB) * ncb + (i % CB)).to(tl.uint32)
    w0, w1, w2, w3 = _kvbf_net(sv, seed, NB)
    e = _kvbf_u16(tl.reshape(w0, (NB * 16, 16)), tl.reshape(w1, (NB * 16, 16)),
                  tl.reshape(w2, (NB * 16, 16)), tl.reshape(w3, (NB * 16, 16)), NB * 16)
    if CB == 1:
        return e
    e = tl.permute(tl.reshape(e, (RB, CB, 16, 128)), (0, 2, 1, 3))
    return tl.reshape(e, (RB * 16, CB * 128))


@triton.jit
def _ae_s32_ks(seed, cid, wl):
    """sloth32 keystream word of pool word (chunk cid, word wl in the chunk): mixes all 64 bits of the address."""
    c = cid.to(tl.int64)
    lo = (c & 0xFFFFF).to(tl.uint32) * 1024 + wl.to(tl.uint32)
    hi = (c >> 20).to(tl.uint32)
    return _s32_fmix(seed.to(tl.uint32) ^ (lo * 0x9E3779B9) ^ _s32_fmix(hi ^ 0x85EBCA6B))


@triton.jit
def _ae_s32_widx(cid0, ncb, RB: tl.constexpr, CB: tl.constexpr):
    """(chunk id, word-in-chunk) of the 2-element words of an RB x CB chunk block, [RB*16, CB*64] each."""
    r = tl.arange(0, RB * 16)[:, None]
    w = tl.arange(0, CB * 64)[None, :]
    cid = cid0 + (r // 16) * ncb + w // 64
    wl = (r % 16) * 64 + w % 64
    return cid, wl


@triton.jit
def _ae_pack2(v16):
    R: tl.constexpr = v16.shape[0]
    C: tl.constexpr = v16.shape[1]
    lo, hi = tl.split(tl.reshape(v16.to(tl.uint32), (R, C // 2, 2)))
    return lo | (hi << 16)


@triton.jit
def _ae_unpack2(w):
    R: tl.constexpr = w.shape[0]
    C: tl.constexpr = w.shape[1]
    return tl.reshape(tl.join(w & 0xFFFF, w >> 16), (R, 2 * C)).to(tl.uint16)


@triton.jit
def _ae_enc(v16, seed, cid0, ncb, RB: tl.constexpr, CB: tl.constexpr):
    """Encode a uint16 [RB*16, CB*128] block of pool chunks (store side)."""
    if AE_S32:
        cid, wl = _ae_s32_widx(cid0, ncb, RB, CB)
        return _ae_unpack2(_s32_sigma_inv(_ae_pack2(v16) ^ _ae_s32_ks(seed, cid, wl)))
    else:
        return v16 ^ _ae_ks(seed, cid0, ncb, RB, CB)


@triton.jit
def _ae_dec(v16, seed, cid0, ncb, RB: tl.constexpr, CB: tl.constexpr):
    """Decode a uint16 [RB*16, CB*128] block of pool chunks (load side)."""
    if AE_S32:
        cid, wl = _ae_s32_widx(cid0, ncb, RB, CB)
        return _ae_unpack2(_s32_sigma(_ae_pack2(v16)) ^ _ae_s32_ks(seed, cid, wl))
    else:
        return v16 ^ _ae_ks(seed, cid0, ncb, RB, CB)


@triton.jit
def _ae_enc_half(y16, seed, cid0, ncb, RB: tl.constexpr, half):
    """sloth32-encode a uint16 [RB*16, 64] half-chunk block (columns half*64 .. +64 of chunks cid0 + r*ncb)."""
    r = tl.arange(0, RB * 16)[:, None]
    w = tl.arange(0, 32)[None, :]
    cid = cid0 + (r // 16) * ncb + 0 * w
    wl = (r % 16) * 64 + half * 32 + w
    return _ae_unpack2(_s32_sigma_inv(_ae_pack2(y16) ^ _ae_s32_ks(seed, cid, wl)))


@triton.jit
def _ae_toff(rb, cb, ncb):
    """Element offsets [16, 128] of tiled chunk (rb, cb)."""
    return ((rb.to(tl.int64) * ncb + cb) * 2048 + tl.arange(0, 16)[:, None] * 128
            + tl.arange(0, 128)[None, :])


@triton.jit(do_not_specialize=['M', 'xcid', 'rcid', 'ncid'])
def _ae_add_kernel(x, r, rnew, z, part, M, N, ncb, xcid, rcid, ncid, epoch, master,
                   X_TILED: tl.constexpr, HAS_R: tl.constexpr, ENC: tl.constexpr, STORE_Z: tl.constexpr = True):
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    rows = rb * 16 + tl.arange(0, 16)
    cols = cb * 128 + tl.arange(0, 128)
    toff = _ae_toff(rb, cb, ncb)
    roff = rows.to(tl.int64)[:, None] * N + cols[None, :]
    rmask = (rows < M)[:, None]
    if ENC:
        seed = _ae_seed(epoch, master)
    if X_TILED:
        xv = tl.load(x + toff)
        if ENC:
            xv = _ae_dec(xv.to(tl.uint16, bitcast=True), seed, xcid + rb * ncb + cb, ncb, 1, 1).to(tl.bfloat16, bitcast=True)
    else:
        xv = tl.load(x + roff, mask=rmask, other=0.0)
    if HAS_R:
        rv = tl.load(r + toff)
        if ENC:
            rv = _ae_dec(rv.to(tl.uint16, bitcast=True), seed, rcid + rb * ncb + cb, ncb, 1, 1).to(tl.bfloat16, bitcast=True)
        zv = (xv.to(tl.float32) + rv.to(tl.float32)).to(tl.bfloat16)
    else:
        zv = xv
    if ENC:
        tl.store(rnew + toff, _ae_enc(zv.to(tl.uint16, bitcast=True), seed, ncid + rb * ncb + cb, ncb, 1, 1).to(tl.bfloat16, bitcast=True))
    else:
        tl.store(rnew + toff, zv)
    if STORE_Z:
        tl.store(z + roff, zv, mask=rmask)
    zf = zv.to(tl.float32)
    tl.store(part + rows.to(tl.int64) * ncb + cb, tl.sum(zf * zf, 1))


@triton.jit(do_not_specialize=['M', 'rcid', 'ocid'])
def _ae_norm_t_kernel(rnew, part, w, out, M, N, ncb, eps, rcid, ocid, epoch, master, NCB: tl.constexpr,
                      ENC: tl.constexpr, OUT_TILED: tl.constexpr):
    # everything-encoded norm: z = the new residual (tiled, decoded here), output normed hidden encoded into
    # a tiled pool slot (OUT_TILED) or plaintext row-major (final norm -> logits). Same math as _ae_norm_kernel.
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    rows = rb * 16 + tl.arange(0, 16)
    cols = cb * 128 + tl.arange(0, 128)
    j = tl.arange(0, NCB)
    p = tl.load(part + rows.to(tl.int64)[:, None] * ncb + j[None, :], mask=(j < ncb)[None, :], other=0.0)
    inv = 1.0 / tl.sqrt(tl.sum(p, 1) / N + eps)
    toff = _ae_toff(rb, cb, ncb)
    zv = tl.load(rnew + toff)
    if ENC:
        seed = _ae_seed(epoch, master)
        zv = _ae_dec(zv.to(tl.uint16, bitcast=True), seed, rcid + rb * ncb + cb, ncb, 1, 1).to(tl.bfloat16, bitcast=True)
    y = (zv.to(tl.float32) * inv[:, None] * tl.load(w + cols).to(tl.float32)[None, :]).to(tl.bfloat16)
    if OUT_TILED:
        if ENC:
            y = _ae_enc(y.to(tl.uint16, bitcast=True), seed, ocid + rb * ncb + cb, ncb, 1, 1).to(tl.bfloat16, bitcast=True)
        tl.store(out + toff, y)
    else:
        tl.store(out + rows.to(tl.int64)[:, None] * N + cols[None, :], y, mask=(rows < M)[:, None])


@triton.jit(do_not_specialize=['M', 'cid0', 'row0'])
def _ae_untile_kernel(src, dst, M, N, ncb, cid0, row0, epoch, master, ENC: tl.constexpr):
    # decode-ahead of a matmul input slice: tiled (encoded) rows row0 .. row0 + M -> plaintext row-major scratch
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    grb = row0 // 16 + rb
    v = tl.load(src + _ae_toff(grb, cb, ncb))
    if ENC:
        v = _ae_dec(v.to(tl.uint16, bitcast=True), _ae_seed(epoch, master), cid0 + grb * ncb + cb, ncb, 1, 1).to(tl.bfloat16, bitcast=True)
    rows = rb * 16 + tl.arange(0, 16)
    cols = cb * 128 + tl.arange(0, 128)
    tl.store(dst + rows.to(tl.int64)[:, None] * N + cols[None, :], v, mask=(rows < M)[:, None])


@triton.jit(do_not_specialize=['M'])
def _ae_norm_kernel(z, part, w, M, N, ncb, eps, NCB: tl.constexpr):
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    rows = rb * 16 + tl.arange(0, 16)
    cols = cb * 128 + tl.arange(0, 128)
    j = tl.arange(0, NCB)
    p = tl.load(part + rows.to(tl.int64)[:, None] * ncb + j[None, :], mask=(j < ncb)[None, :], other=0.0)
    inv = 1.0 / tl.sqrt(tl.sum(p, 1) / N + eps)
    roff = rows.to(tl.int64)[:, None] * N + cols[None, :]
    rmask = (rows < M)[:, None]
    zf = tl.load(z + roff, mask=rmask, other=0.0).to(tl.float32)
    y = zf * inv[:, None] * tl.load(w + cols).to(tl.float32)[None, :]
    tl.store(z + roff, y.to(tl.bfloat16), mask=rmask)


@triton.jit(do_not_specialize=['cid0'])
def _ae_fill_kernel(ptr, cid0, epoch, master, NB: tl.constexpr):
    # NB whole 4 KiB chunks per program: encoded zeros = the keystream itself
    seed = _ae_seed(epoch, master)
    c0 = tl.program_id(0) * NB
    ks = _ae_enc(tl.zeros((NB * 16, 128), tl.uint16), seed, cid0 + c0, 1, NB, 1)   # encoded zeros, chunk i = rows 16i..16i+15
    off = c0.to(tl.int64) * 2048 + tl.arange(0, NB * 16).to(tl.int64)[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(ptr + off, ks.to(tl.bfloat16, bitcast=True))


@triton.jit
def _ae_drain_wait_kernel(park, tout, LIMIT_NS: tl.constexpr):
    # diagnostic (VARENA_AE_DRAIN=1): after the gate is raised, wait until the compute stream's next marker is
    # parked at the gate (the op in flight has completed), at most LIMIT_NS; then write the GPU time (the chase
    # start) to the pinned host slot
    t0 = tl.extra.cuda.globaltimer()
    p = tl.load(park, volatile=True)
    t = t0
    while (p == 0) & (t - t0 < LIMIT_NS):
        p = tl.load(park, volatile=True)
        t = tl.extra.cuda.globaltimer()
    tl.store(tout, tl.extra.cuda.globaltimer().to(tl.int64))


@triton.jit
def _ae_mark_kernel(pos, gate, START: tl.constexpr, GATED: tl.constexpr):
    # position marker (soft freeze): launched before every pool-writing op, so stream order makes
    # "pos[0] = k" mean ops < k are complete. pos = [op count, first op of this forward, forward count].
    # GATED: yield on challenge: while the prover's challenge-in-flight flag (pinned host memory, set by the
    # native challenge server) is set, the compute stream waits here, so a challenge runs on an idle GPU
    # after at most the op in flight
    if GATED:
        g = tl.load(gate, volatile=True)
        if g != 0:
            tl.store(gate + 1, 1)                      # parked (drain diagnostic)
        while g != 0:
            g = tl.load(gate, volatile=True)
        tl.store(gate + 1, 0)
    c = tl.load(pos) + 1
    tl.store(pos, c)
    if START:
        tl.store(pos + 1, c)
        tl.store(pos + 2, tl.load(pos + 2) + 1)


@triton.jit(do_not_specialize=['v'])
def _ae_gate_set_kernel(gate, v, tout, TIMED: tl.constexpr):
    tl.store(gate, v)
    if TIMED:
        # tail study: GPU time at which the challenge graph started / finished (pinned host slot)
        tl.store(tout, tl.extra.cuda.globaltimer().to(tl.int64))


@triton.jit(do_not_specialize=['cid0', 'mp'])
def _ae_scrub_kernel(ptr, cid0, ncb, mp, hw, epoch, master, NB: tl.constexpr, NPROG: tl.constexpr):
    # rows [mp, hw) of a tiled pool slot back to encoded zeros (stale rows of an earlier, larger step)
    seed = _ae_seed(epoch, master)
    c_lo = (mp // 16) * ncb
    c_hi = (tl.load(hw) // 16) * ncb
    for c0 in range(c_lo + tl.program_id(0) * NB, c_hi, NPROG * NB):
        ks = _ae_enc(tl.zeros((NB * 16, 128), tl.uint16), seed, cid0 + c0, 1, NB, 1)
        off = c0.to(tl.int64) * 2048 + tl.arange(0, NB * 16).to(tl.int64)[:, None] * 128 + tl.arange(0, 128)[None, :]
        cm = (c0 + tl.arange(0, NB * 16) // 16) < c_hi
        tl.store(ptr + off, ks.to(tl.bfloat16, bitcast=True), mask=cm[:, None])


@triton.jit
def _ae_sethw_kernel(hw, mp):
    tl.store(hw, mp)


@triton.jit
def _ae_pos_read_kernel(pos, out, AT: tl.constexpr):
    # soft freeze: op window of a digest challenge (out[AT .. AT+2] = op count, forward's first op, forwards)
    tl.store(out + AT + tl.arange(0, 4), tl.load(pos + tl.arange(0, 4)))


@triton.jit(do_not_specialize=['N', 'ngran'])
def _ae_sample_kernel(gran, ngran, nonce_ptr, out, N, UPP: tl.constexpr):
    # activation digest challenge: raw 128 B units at nonce-derived pool addresses -> out[8 + 16 * i ..]
    nu = tl.load(nonce_ptr).to(tl.uint32)
    i = tl.program_id(0) * UPP + tl.arange(0, UPP)
    h1 = _kvbf_fmix(nu ^ (i.to(tl.uint32) * 0x9E37 + 1))
    h2 = _kvbf_fmix(h1 ^ 0x85EB)
    g = (h1 % ngran.to(tl.uint32)).to(tl.int64)
    addr = tl.load(gran + g) + (h2 % 524288).to(tl.int64) * 128
    w = tl.load((addr[:, None] + tl.arange(0, 16)[None, :].to(tl.int64) * 8).to(tl.pointer_type(tl.int64)),
                mask=(i < N)[:, None])
    tl.store(out + 8 + i[:, None].to(tl.int64) * 16 + tl.arange(0, 16)[None, :], w, mask=(i < N)[:, None])


@triton.jit(do_not_specialize=['M', 'cid0'])
def _ae_embed_kernel(ids, w, out, M, H, ncb, cid0, epoch, master, ENC: tl.constexpr):
    # embedding output straight into its tiled pool slot (encoded): one 16-token x 128-column chunk per program
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    rows = rb * 16 + tl.arange(0, 16)
    cols = cb * 128 + tl.arange(0, 128)
    rm = rows < M
    tok = tl.load(ids + rows, mask=rm, other=0).to(tl.int64)
    v = tl.load(w + tok[:, None] * H + cols[None, :], mask=rm[:, None], other=0.0)
    if ENC:
        v = _ae_enc(v.to(tl.uint16, bitcast=True), _ae_seed(epoch, master), cid0 + rb * ncb + cb, ncb, 1, 1).to(tl.bfloat16, bitcast=True)
    tl.store(out + _ae_toff(rb, cb, ncb), v)


@triton.jit(do_not_specialize=['M', 'gcid'])
def _ae_silu_kernel(gu, out, M, I, ncb, gcid, epoch, master, ENC: tl.constexpr):
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    hb = I // 128
    gv = tl.load(gu + _ae_toff(rb, cb, ncb))
    uv = tl.load(gu + _ae_toff(rb, cb + hb, ncb))
    if ENC:
        seed = _ae_seed(epoch, master)
        gv = _ae_dec(gv.to(tl.uint16, bitcast=True), seed, gcid + rb * ncb + cb, ncb, 1, 1).to(tl.bfloat16, bitcast=True)
        uv = _ae_dec(uv.to(tl.uint16, bitcast=True), seed, gcid + rb * ncb + cb + hb, ncb, 1, 1).to(tl.bfloat16, bitcast=True)
    g = gv.to(tl.float32)
    y = g / (1.0 + tl.exp(-g)) * uv.to(tl.float32)
    rows = rb * 16 + tl.arange(0, 16)
    cols = cb * 128 + tl.arange(0, 128)
    tl.store(out + rows.to(tl.int64)[:, None] * I + cols[None, :], y.to(tl.bfloat16),
             mask=(rows < M)[:, None])
'''

# store epilogue of the copied matmul_kernel_persistent
_MM_STORE_OLD = """        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)"""
_MM_STORE_NEW = """        c = accumulator.to(c_ptr.dtype.element_ty)
        if TILED:
            tl.static_assert(BLOCK_SIZE_M % 16 == 0)
            tl.static_assert(BLOCK_SIZE_N % 128 == 0)
            ncb = N // 128
            t_r = offs_cm.to(tl.int64)
            t_c = offs_cn.to(tl.int64)
            t_off = (((t_r[:, None] // 16) * ncb + t_c[None, :] // 128) * 2048
                     + (t_r[:, None] % 16) * 128 + t_c[None, :] % 128)
            t_mask = (offs_cm[:, None] < MP) & (offs_cn[None, :] < N)
            if ENC:
                e_cid = CCID + pid_m * (BLOCK_SIZE_M // 16) * ncb + pid_n * (BLOCK_SIZE_N // 128)
                e_seed = _ae_seed(epoch_ptr, master)
                if EPI_SPLIT == 1:
                    c = _ae_enc(c.to(tl.uint16, bitcast=True), e_seed, e_cid, ncb, BLOCK_SIZE_M // 16,
                                BLOCK_SIZE_N // 128).to(tl.bfloat16, bitcast=True)
                else:
                    # keystream for the top / bottom half of the tile in turn (fewer live registers)
                    tl.static_assert(BLOCK_SIZE_M % 32 == 0)
                    HM: tl.constexpr = BLOCK_SIZE_M // 2
                    cu = tl.permute(tl.reshape(c.to(tl.uint16, bitcast=True), (2, HM, BLOCK_SIZE_N)), (1, 2, 0))
                    c_lo, c_hi = tl.split(cu)
                    c_lo = _ae_enc(c_lo, e_seed, e_cid, ncb, HM // 16, BLOCK_SIZE_N // 128)
                    c_hi = _ae_enc(c_hi, e_seed, e_cid + (HM // 16) * ncb, ncb, HM // 16, BLOCK_SIZE_N // 128)
                    c = tl.reshape(tl.permute(tl.join(c_lo, c_hi), (2, 0, 1)),
                                   (BLOCK_SIZE_M, BLOCK_SIZE_N)).to(tl.bfloat16, bitcast=True)
            tl.store(c_ptr + t_off, c, mask=t_mask)
        else:
            tl.store(c_ptrs, c, mask=c_mask)"""


def _sub(s, a, b):
    assert s.count(a) == 1, a
    return s.replace(a, b)


def _matmul_src():
    import vllm.model_executor.determinism.batch_invariant as bi
    k = inspect.getsource(getattr(bi.matmul_kernel_persistent, "fn", bi.matmul_kernel_persistent))
    k = _sub(k, "def matmul_kernel_persistent(", "def ae_matmul_kernel_persistent(")
    # address-derived / per-call ints must not be specialised (each new divisibility combination recompiles the
    # kernel mid-run: 93 variants, engine stalls of 15-40 s)
    k = _sub(k, "@triton.jit(launch_metadata=_matmul_launch_metadata)",
             "@triton.jit(launch_metadata=_matmul_launch_metadata, do_not_specialize=['MP', 'CCID', 'w_seed', 'w_n0'])")
    k = _sub(k, "    HAS_BIAS: tl.constexpr,\n):",
             "    HAS_BIAS: tl.constexpr,\n    MP=0,\n    CCID=0,\n    epoch_ptr=None,\n    master=0,\n"
             "    TILED: tl.constexpr = False,\n    ENC: tl.constexpr = False,\n    EPI_SPLIT: tl.constexpr = 1,\n):")
    k = _sub(k, _MM_STORE_OLD, _MM_STORE_NEW)
    # freeze: with STOPPABLE, every tile reads the stop flag (pinned host memory, UVA) first; once it is
    # set, remaining tiles skip their K loop and store, so the kernel stops writing within one tile
    k = _sub(k, "        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)\n",
             "        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)\n"
             "        if PARK:\n"
             "            pg = tl.atomic_add(gate_ptr, 0)\n"
             "            while pg != 0:\n"
             "                pg = tl.atomic_add(gate_ptr, 0)\n"
             "        if STOPPABLE:\n"
             "            go = tl.load(stop_ptr, volatile=True) == 0\n"
             "            k_tiles_go = tl.where(go, k_tiles, 0)\n"
             "        else:\n"
             "            go = True\n"
             "            k_tiles_go = k_tiles\n")
    k = _sub(k, "        for ki in range(k_tiles):", "        for ki in range(k_tiles_go):")
    # fused sloth64 weight decode (VARENA_WDEC_FUSE): b is the ENCODED W^T (W row-major [N, K]); each loaded B tile
    # is decoded in registers (one modular squaring per 64-bit word = 4 consecutive k of one row), so W crosses HBM
    # once. The decode delta is computed in the [BN, BK] word layout and re-laid into the load layout with an
    # identity gather, so tl.dot sees the native operand layout (bit-identical to decode-ahead + this kernel)
    k = _sub(k, """            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
""", """            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            if WDEC:
                wb16 = b.to(tl.uint16, bitcast=True)
                wb16t = tl.trans(wb16)
                w_e = (offs_bn[:, None].to(tl.int64) + w_n0) * K + offs_k[None, :].to(tl.int64)
                w_d = _sl_dec_tile(wb16t, w_e, w_seed.to(tl.int64).to(tl.uint64, bitcast=True)) ^ wb16t
                w_d = tl.gather(tl.trans(w_d), tl.arange(0, BLOCK_SIZE_K)[:, None] + 0 * tl.arange(0, BLOCK_SIZE_N)[None, :], 0)
                b = (wb16 ^ w_d).to(tl.bfloat16, bitcast=True)
""")
    k = _sub(k, "            tl.store(c_ptr + t_off, c, mask=t_mask)", "            tl.store(c_ptr + t_off, c, mask=t_mask & go)")
    k = _sub(k, "    EPI_SPLIT: tl.constexpr = 1,\n):", "    EPI_SPLIT: tl.constexpr = 1,\n    stop_ptr=None,\n    STOPPABLE: tl.constexpr = False,\n"
             "    gate_ptr=None,\n    PARK: tl.constexpr = False,\n"
             "    FUSE_SILU: tl.constexpr = False,\n    HALF_N=0,\n    s_ptr=None,\n    S_TILED: tl.constexpr = False,\n):")
    # fused gate_up + silu_and_mul (FUSE_SILU): output tile n loads gate columns 64n..64n+63 and the matching
    # up columns HALF_N + 64n.. (no change to W's layout), and the epilogue stores silu(g) * u (64 plaintext
    # columns, row-major [M, HALF_N]): the gate_up activation never exists in memory
    k = _sub(k, "        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)\n",
             "        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)\n"
             "        if FUSE_SILU:\n"
             "            f_j = tl.arange(0, BLOCK_SIZE_N)\n"
             "            offs_bn = tl.where(f_j < BLOCK_SIZE_N // 2, pid_n * (BLOCK_SIZE_N // 2) + f_j,\n"
             "                               HALF_N + pid_n * (BLOCK_SIZE_N // 2) + f_j - BLOCK_SIZE_N // 2)\n")
    k = _sub(k, "        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)\n",
             "        if FUSE_SILU:\n"
             "            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N // 2), BLOCK_SIZE_N // 2)\n"
             "        else:\n"
             "            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)\n")
    k = _sub(k, "        c = accumulator.to(c_ptr.dtype.element_ty)\n        if TILED:",
             "        c = accumulator.to(c_ptr.dtype.element_ty)\n"
             "        if FUSE_SILU:\n"
             "            f_g, f_u = tl.split(tl.permute(tl.reshape(c, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2)), (0, 2, 1)))\n"
             "            f_gf = f_g.to(tl.float32)\n"
             "            f_y = (f_gf / (1.0 + tl.exp(-f_gf)) * f_u.to(tl.float32)).to(tl.bfloat16)\n"
             "            f_c = pid_n * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)\n"
             "            if S_TILED:\n"
             "                # silu output encoded into its tiled pool slot: a tile covers half a 128-column chunk\n"
             "                s_ncb = HALF_N // 128\n"
             "                if ENC:\n"
             "                    HM: tl.constexpr = BLOCK_SIZE_M // 2\n"
             "                    e_seed = _ae_seed(epoch_ptr, master)\n"
             "                    e_cid = CCID + pid_m * (BLOCK_SIZE_M // 16) * s_ncb + pid_n // 2\n"
             "                    yu = tl.permute(tl.reshape(f_y.to(tl.uint16, bitcast=True), (2, HM, BLOCK_SIZE_N // 2)), (1, 2, 0))\n"
             "                    y_lo, y_hi = tl.split(yu)\n"
             "                    if AE_S32:\n"
             "                        y_lo = _ae_enc_half(y_lo, e_seed, e_cid, s_ncb, HM // 16, pid_n % 2)\n"
             "                        y_hi = _ae_enc_half(y_hi, e_seed, e_cid + (HM // 16) * s_ncb, s_ncb, HM // 16, pid_n % 2)\n"
             "                    else:\n"
             "                        k_a, k_b = tl.split(tl.permute(tl.reshape(_ae_ks(e_seed, e_cid, s_ncb, HM // 16, 1), (HM, 2, BLOCK_SIZE_N // 2)), (0, 2, 1)))\n"
             "                        y_lo = y_lo ^ tl.where(pid_n % 2 == 0, k_a, k_b)\n"
             "                        k_a, k_b = tl.split(tl.permute(tl.reshape(_ae_ks(e_seed, e_cid + (HM // 16) * s_ncb, s_ncb, HM // 16, 1), (HM, 2, BLOCK_SIZE_N // 2)), (0, 2, 1)))\n"
             "                        y_hi = y_hi ^ tl.where(pid_n % 2 == 0, k_a, k_b)\n"
             "                    f_y = tl.reshape(tl.permute(tl.join(y_lo, y_hi), (2, 0, 1)), (BLOCK_SIZE_M, BLOCK_SIZE_N // 2)).to(tl.bfloat16, bitcast=True)\n"
             "                s_r = offs_cm.to(tl.int64)\n"
             "                s_c = f_c.to(tl.int64)\n"
             "                s_off = (((s_r[:, None] // 16) * s_ncb + s_c[None, :] // 128) * 2048 + (s_r[:, None] % 16) * 128 + s_c[None, :] % 128)\n"
             "                tl.store(s_ptr + s_off, f_y, mask=(offs_cm[:, None] < MP) & (f_c[None, :] < HALF_N) & go)\n"
             "            else:\n"
             "                tl.store(s_ptr + offs_cm.to(tl.int64)[:, None] * HALF_N + f_c[None, :], f_y,\n"
             "                         mask=(offs_cm[:, None] < M) & (f_c[None, :] < HALF_N) & go)\n"
             "        elif TILED:")
    k = _sub(k, "    FUSE_SILU: tl.constexpr = False,\n", "    FUSE_SILU: tl.constexpr = False,\n    WDEC: tl.constexpr = False,\n    w_seed=0,\n    w_n0=0,\n    S_C0=0,\n    S_LD=0,\n")
    # column-sliced gate_up (sliced decode-ahead of a big gate_up W): this launch covers output columns
    # S_C0 .. S_C0 + HALF_N of an S_LD-wide output; tiles, K order and W rows are those of the full product
    k = _sub(k, "            f_c = pid_n * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)\n",
             "            f_c = S_C0 + pid_n * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)\n"
             "            s_ldv = tl.where(S_LD > 0, S_LD, HALF_N)\n"
             "            f_q = S_C0 // (BLOCK_SIZE_N // 2) + pid_n\n")
    k = _sub(k, "                s_ncb = HALF_N // 128\n", "                s_ncb = s_ldv // 128\n")
    k = _sub(k, "e_cid = CCID + pid_m * (BLOCK_SIZE_M // 16) * s_ncb + pid_n // 2\n",
             "e_cid = CCID + pid_m * (BLOCK_SIZE_M // 16) * s_ncb + f_q // 2\n")
    k = k.replace("pid_n % 2", "f_q % 2")
    k = _sub(k, "(f_c[None, :] < HALF_N) & go)\n            else:", "(f_c[None, :] < S_C0 + HALF_N) & go)\n            else:")
    k = _sub(k, "tl.store(s_ptr + offs_cm.to(tl.int64)[:, None] * HALF_N + f_c[None, :], f_y,\n"
                "                         mask=(offs_cm[:, None] < M) & (f_c[None, :] < HALF_N) & go)",
             "tl.store(s_ptr + offs_cm.to(tl.int64)[:, None] * s_ldv + f_c[None, :], f_y,\n"
             "                         mask=(offs_cm[:, None] < M) & (f_c[None, :] < S_C0 + HALF_N) & go)")
    return k


# activation pool encoding: butterfly XOR keystream (default) or sloth32 (VARENA_AE_S32=1, experiment): slow
# square-root encode on every pool store, one modular squaring on every pool load
AE_S32 = os.environ.get("VARENA_AE_S32", "0") == "1"
_MOD = None


def load():
    global _MOD
    if _MOD is not None:
        return _MOD
    from schemes.memory_link_exhaustion.encodings import kvbfly
    from schemes.memory_link_exhaustion.inference.patches import _write_and_import
    head = ("import torch\nimport triton\nimport triton.language as tl\n"
            "from vllm.model_executor.determinism.batch_invariant import _compute_pid, _matmul_launch_metadata\n")
    mm = _matmul_src()
    from schemes.memory_link_exhaustion.encodings import sloth32, sloth64
    _MOD = _write_and_import("actenc2", head + kvbfly.triton_src(8, 2) + sloth32.TRITON_HELPERS + sloth64.TRITON_HELPERS
                             + f"\nAE_S32 = tl.constexpr({AE_S32})\n" + AE_SRC + "\n\n" + mm + "\n\n" + _preempt_src(mm)
                             + _GATE_WAIT_SRC)
    return _MOD


_GATE_WAIT_SRC = """


@triton.jit
def _ae_gate_wait_kernel(gate):
    g = tl.load(gate, volatile=True)
    while g != 0:
        g = tl.load(gate, volatile=True)
"""


def _preempt_src(k: str) -> str:
    """Pre-emptible matmul (VARENA_AE_PREEMPT=1, quiet-chase design): tiles come from an atomic counter and a CTA
    takes a new tile only while the device gate is down, so when a challenge raises the gate every CTA finishes its
    current tile and exits (its SM is free). A resume launch of the same kernel on the same counter, queued behind a
    one-warp gate wait, computes the remaining tiles. Per-tile arithmetic is unchanged (bit-identical)."""
    k = _sub(k, "def ae_matmul_kernel_persistent(", "def ae_matmul_kernel_preempt(")
    k = _sub(k, "    gate_ptr=None,\n", "    gate_ptr=None,\n    ctr_ptr=None,\n    NOYIELD: tl.constexpr = False,\n")
    k = _sub(k, "    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):\n",
             "    pgq = 0 if NOYIELD else tl.atomic_add(gate_ptr, 0)\n"
             "    tile_id = tl.where(pgq == 0, tl.atomic_add(ctr_ptr, 1, mask=pgq == 0), num_tiles)\n"
             "    while tile_id < num_tiles:\n")
    k = _sub(k, "        tile_id_c += NUM_SMS\n", "        tile_id_c = tile_id\n")
    # the gate is read with a scalar atomic (one read per CTA, broadcast), not a per-thread volatile load: with
    # per-thread loads, threads of one CTA could see the gate flip at different times, disagree on taking the next
    # tile and diverge around tl.dot's warp-group barriers (the intermittent illegal-instruction race)
    k = k.rstrip("\n") + ("\n        pgq = 0 if NOYIELD else tl.atomic_add(gate_ptr, 0)\n"
                          "        tile_id = tl.where(pgq == 0, tl.atomic_add(ctr_ptr, 1, mask=pgq == 0), num_tiles)\n")
    return k


PREEMPT_ON = os.environ.get("VARENA_AE_PREEMPT", "0") == "1"
RESUME_N = int(os.environ.get("VARENA_AE_RESUME_N", "0"))
WFUSE = os.environ.get("VARENA_WDEC_FUSE", "0") == "1"


def _wdec_kw(b, wseed=None, wn0=0):
    """Kernel kwargs for the fused sloth W decode: b = an encoded W^T registered in W_KEYS (or an explicit seed for a
    row slice of one)."""
    if not WFUSE:
        return {}
    if wseed is None:
        from schemes.memory_link_exhaustion.inference import patches as P
        key = P.W_KEYS.get(b.data_ptr())
        if not isinstance(key, tuple):
            return {}
        from schemes.memory_link_exhaustion.encodings import ref as _ref
        wseed = _ref.sloth_seed(key[0], key[1])
    return {"WDEC": True, "w_seed": int(wseed) - (1 << 64) if int(wseed) >= 1 << 63 else int(wseed), "w_n0": int(wn0)}


def _launch_mm(mod, grid, args, kw):
    """Launch our matmul copy: persistent, or pre-emptible (main + gate wait + resume on one counter)."""
    if not PREEMPT_ON:
        mod.ae_matmul_kernel_persistent[grid](*args, **kw)
        return
    ctr = torch.zeros(1, dtype=torch.int32, device=args[0].device)
    g = gate_dev(args[0].device)
    kw = dict(kw, gate_ptr=g, ctr_ptr=ctr, PARK=False)
    mod.ae_matmul_kernel_preempt[grid](*args, **kw)
    # VARENA_AE_RESUME_N yielding resumes (each behind a gate wait) before the final one: a challenge that lands
    # during a resume then parks it too (the trace showed half the honest chase overlaps were non-yielding
    # resumes). The last resume never yields, so no tile is left uncomputed whatever the challenge pattern.
    for _ in range(RESUME_N):
        mod._ae_gate_wait_kernel[(1,)](g, num_warps=1)
        mod.ae_matmul_kernel_preempt[grid](*args, **kw)
    mod._ae_gate_wait_kernel[(1,)](g, num_warps=1)
    # the final resume launch does not yield: a challenge during it would otherwise leave tiles uncomputed
    mod.ae_matmul_kernel_preempt[grid](*args, **dict(kw, NOYIELD=True))


def epoch(device) -> torch.Tensor:
    t = _EPOCH.get(device)
    if t is None:
        t = _EPOCH[device] = torch.ones(1, dtype=torch.int32, device=device)
    return t


def mpad(M: int) -> int:
    return (M + 15) // 16 * 16


def tiled_empty(M: int, N: int, like: torch.Tensor) -> torch.Tensor:
    """[M, N] view of a fresh 4 KiB-aligned tiled buffer (Mp x N elements behind it)."""
    assert N % 128 == 0
    n = mpad(M) * N
    buf = torch.empty(n + 2048, dtype=torch.bfloat16, device=like.device)
    off = (-buf.data_ptr() % 4096) // 2
    return buf[off:off + n][:M * N].view(M, N)


# Fixed activation pool (VARENA_ACT_POOL_TOKENS = token cap; 0 = fresh allocations per call). Encoded
# tensors live in a few reused slots at fixed, 4 KiB-aligned addresses, so the prover can declare one
# set of ranges and the verifier can chase them: "gu" (gate_up output), "x" (o_proj and down outputs:
# each is consumed by the next add+norm before the slot is written again), "r0" / "r1" (residual,
# alternating so a call never writes the slot it reads). Reuse is ordered by data dependencies.
# encoded matmul epilogue: keystream for the whole tile at once (1) or per half tile (2); register cap
# (0 = none). The encoded epilogue raised registers 118 -> 219 / thread, which leaves little room for
# the challenge chase's warps on the same SM (more chase waves, longer probe latency).
EPI_SPLIT = int(os.environ.get("VARENA_AE_EPI_SPLIT", "2"))
MAXNREG = int(os.environ.get("VARENA_AE_MAXNREG", "0"))
POOL_TOKENS = int(os.environ.get("VARENA_ACT_POOL_TOKENS", "0"))
# soft freeze: position markers before every pool-writing op and the per-forward scrub of stale rows
MARKERS = os.environ.get("VARENA_ACT_MARKERS", "1") == "1"
POOL: dict = {}             # slot name -> backing tensor (aligned)


GRAN = 64 << 20             # chase granule (schemes.memory_link_exhaustion.encodings.ref.CH_GRAN_PAGES pages of 64 KiB)
POOL_BUFS: dict = {}        # slot name -> uint8 backing buffer (whole granules, 64 KiB aligned)


def pool_ranges():
    """[(address, bytes)] of the activation pool slots (whole 64 MiB granules)."""
    return [(b.data_ptr(), b.numel()) for b in POOL_BUFS.values()]


def pool_granules():
    """Addresses of the pool's 64 MiB granules (the activation chase's address space)."""
    return [a + j * GRAN for a, n in pool_ranges() for j in range(n // GRAN)]


def slot(name: str, M: int, N: int, like: torch.Tensor) -> torch.Tensor:
    """[M, N] tiled view of pool slot `name` (fresh buffer if the pool is off or M > cap). A slot is a
    whole number of 64 MiB granules; bytes past the cap's rows hold encoded zeros (fill_pool)."""
    if not POOL_TOKENS or M > POOL_TOKENS:
        return tiled_empty(M, N, like)
    t = POOL.get(name)
    if t is None:
        nb = -(-mpad(POOL_TOKENS) * N * 2 // GRAN) * GRAN
        raw = torch.empty(nb + (64 << 10), dtype=torch.uint8, device=like.device)
        off = -raw.data_ptr() % (64 << 10)
        POOL_BUFS[name] = raw[off:off + nb]
        SLOT_N[name] = N
        t = POOL[name] = POOL_BUFS[name].view(torch.bfloat16)
        fill_pool(name)
    return t[:M * N].view(M, N)


def fill_pool(name=None):
    """Write encoded zeros (= ks(address, epoch)) over whole pool slots: at allocation and after an
    epoch change, so every byte of the pool is keystream for the zero-weights model."""
    mod = load()
    for k, b in POOL_BUFS.items():
        if name is not None and k != name:
            continue
        n = b.numel() // 4096
        mod._ae_fill_kernel[(n // 8,)](b.view(torch.bfloat16), (b.data_ptr() >> 12) & 0xFFFFFFFF,
                                        epoch(b.device), MASTER, NB=8, num_warps=4)


_POS = {}
_HW = {}                    # slot name -> device int32 high-water row (rows written since the last scrub)


def _dkey(device):
    d = torch.device(device)
    return d.index if d.index is not None else torch.cuda.current_device()


def pos(device) -> torch.Tensor:
    device = _dkey(device)
    t = _POS.get(device)
    if t is None:
        t = _POS[device] = torch.zeros(4, dtype=torch.int64, device=device)
    return t


GATE_ON = os.environ.get("VARENA_AE_GATE", "1") == "1"
_GATE = []


def gate_dev(device="cuda") -> torch.Tensor:
    """Device int32 challenge-in-flight gate (allocated before any CUDA graph capture). Every challenge graph
    sets it first and clears it last (gate_set, prover._start_native), so the compute stream's markers poll
    device memory: a pinned host flag read over PCIe queued behind the noise (2.8x slower prover)."""
    if not _GATE:
        _GATE.append(torch.zeros(2, dtype=torch.int32, device=device))
    return _GATE[0]


DRAIN = os.environ.get("VARENA_AE_DRAIN", "0") == "1"


def drain_wait(tout, limit_ms: float = 8.0):
    """Diagnostic: wait (on the device) until the compute stream is parked at the raised gate; GPU time -> tout."""
    load()._ae_drain_wait_kernel[(1,)](gate_dev()[1:], HostPtr(tout), LIMIT_NS=int(limit_ms * 1e6), num_warps=1)


def gate_set(v: int, tout=None):
    """Raise / drop the yield gate; tout: pinned int64 host slot (one element) for the GPU timestamp."""
    load()._ae_gate_set_kernel[(1,)](gate_dev(), v, HostPtr(tout) if tout is not None else gate_dev(),
                                     TIMED=tout is not None, num_warps=1)


MARK_SITES: list = []      # op index within a forward -> call site (recorded at trace / capture time)


_TAIL = [False]
VLLM_EVENTS: list = []
OPS_SEEN = [0]
VLLM_STREAMS: dict = {}
TAG_IDS: dict = {}


def mark(device, start=False, tag=None):
    if start:
        MARK_SITES.clear()
        _TAIL[0] = False
        _pop_opmode()
    f = sys._getframe(1)
    MARK_SITES.append(tag or f"{f.f_code.co_name}:{f.f_lineno}")
    if tag:
        # tagged marker: its id goes to pos[3] in stream order (index-free: CUDA-graph replays don't rerun
        # Python, so MARK_SITES indices drift; the tag id names the last tagged point the stream passed)
        tid = TAG_IDS.setdefault(tag, len(TAG_IDS) + 1)
        with torch.inference_mode():
            pos(device)[3].fill_(tid)
    load()._ae_mark_kernel[(1,)](pos(device), gate_dev(device), START=start, GATED=GATE_ON, num_warps=1)


def scrub(M: int, device):
    """Forward start: scrub rows >= Mp of every pool slot written by an earlier, larger step (device-side
    high-water mark, so this is CUDA-graph safe), then set the high-water mark to this step's Mp."""
    mod = load()
    mp = mpad(M) if M <= POOL_TOKENS else 0
    for name, b in POOL_BUFS.items():
        hw = _HW.get(name)
        if hw is None:
            hw = _HW[name] = torch.full((1,), mpad(POOL_TOKENS), dtype=torch.int32, device=device)
        N = SLOT_N[name]
        mod._ae_scrub_kernel[(132,)](b.view(torch.bfloat16), (b.data_ptr() >> 12) & 0xFFFFFFFF, N // 128, mp, hw,
                                     epoch(device), MASTER, NB=4, NPROG=132, num_warps=4)
        mod._ae_sethw_kernel[(1,)](hw, mp, num_warps=1)


SLOT_N: dict = {}           # slot name -> columns


def cid(t: torch.Tensor) -> int:
    return (t.data_ptr() >> 12) & 0xFFFFFFFF


# pre-emptible matmuls (VARENA_AE_PARK=1, quiet-chase design v1): at every tile boundary a CTA spins while the device
# gate is raised (a challenge in flight), so the op in flight stops its memory traffic within one tile; per-tile
# arithmetic is unchanged (bit-identical)
PARK_ON = os.environ.get("VARENA_AE_PARK", "0") == "1"


def _park_gate():
    return gate_dev() if PARK_ON else None


STOP = None                 # freeze: HostPtr to the pinned stop flag (None = matmuls are not stoppable)


class HostPtr:
    """Pinned host memory as a Triton pointer argument (UVA: host pointer == device pointer)."""

    def __init__(self, t):
        self.t, self.dtype = t, t.dtype

    def data_ptr(self):
        return self.t.data_ptr()


def ae_matmul(a: torch.Tensor, b: torch.Tensor, enc: bool, bias=None, out: str = "", c=None,
              tiled: bool = True, cfg_n: int = 0, wseed=None, wn0: int = 0) -> torch.Tensor:
    """y = a @ b (+ bias), tiled output (encoded if enc); bit-identical arithmetic to vLLM's
    batch-invariant matmul_persistent (same kernel, same configs; only the store differs)."""
    import triton
    import vllm.model_executor.determinism.batch_invariant as bi
    from vllm.model_executor.determinism.batch_invariant import num_compute_units
    from vllm.model_executor.determinism.batch_invariant_configs import _get_matmul_config
    mod = load()
    assert a.dtype == torch.bfloat16 and a.shape[1] == b.shape[0]
    M, K = a.shape
    N = b.shape[1]
    NUM_SMS = num_compute_units(a.device.index)
    if c is None:
        c = slot(out, M, N, a) if out else (tiled_empty(M, N, a) if tiled else
                                            torch.empty(M, N, dtype=torch.bfloat16, device=a.device))
    if MARKERS and (out or POOL_BUFS):
        mark(a.device)
    if not bi._BLOCK_SIZES_RESOLVED:
        bi._resolve_block_sizes()
    # cfg_n: a column slice of a larger matmul uses the full matrix's config (same tiles, same K order: the
    # slice's outputs are bit-identical to the corresponding columns of the full product)
    cfg = _get_matmul_config(M, cfg_n or N, K, a.dtype, {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
                                                 "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8})
    tiles = triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"])
    grid = (min(NUM_SMS, tiles),)
    _launch_mm(mod, grid, (
        a, b, c, bias, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0) if not tiled else N, 1,
        bias.stride(0) if bias is not None else 1), dict(NUM_SMS=NUM_SMS,
        A_LARGE=a.numel() > 2**31, B_LARGE=b.numel() > 2**31, C_LARGE=mpad(M) * N > 2**31,
        HAS_BIAS=bias is not None, MP=mpad(M), CCID=cid(c), epoch_ptr=epoch(a.device), master=MASTER,
        TILED=tiled, ENC=enc and tiled, EPI_SPLIT=EPI_SPLIT if enc else 1, stop_ptr=STOP, STOPPABLE=STOP is not None, gate_ptr=_park_gate(), PARK=PARK_ON, **cfg, **({"maxnreg": MAXNREG} if MAXNREG else {}), **_wdec_kw(b, wseed, wn0)))
    return c


FUSE_GU = os.environ.get("VARENA_AE_FUSE_GU", "1") == "1"
# pool slot layout: normed hidden and silu output ("h", "s") and the attention output ("a") have slots too
ENCALL = os.environ.get("VARENA_AE_ENCALL", "1") == "1"
AE_ATTN = ENCALL and os.environ.get("VARENA_AE_ATTN", "1") == "1"


def pool_slots(H: int, I: int):
    """(name, columns) of the activation pool slots the current mode uses."""
    sl = [] if FUSE_GU else [("gu", 2 * I)]
    if ENCALL:
        sl += [("h", H), ("s", I)] + ([("a", H)] if AE_ATTN else [])
    return sl + [("x", H), ("r0", H), ("r1", H)]


# Persistent scratch (stall fix): the W decode-ahead buffer and the decode-ahead slice buffer are one fixed
# allocation each, grown only when a larger request arrives (warm-up), so the steady state does no cudaMalloc:
# fresh per-call allocations (e.g. 566 MB for 32B gate_up W) next to an arena that fills the GPU made the caching
# allocator free-and-retry. Safe: the ops run in order on one stream and use the buffer only inside the op.
PERSIST = os.environ.get("VARENA_AE_PERSIST", "1") == "1"
_SCR: dict = {}


def scratch(name: str, shape, like: torch.Tensor) -> torch.Tensor:
    if not PERSIST:
        return torch.empty(shape, dtype=like.dtype, device=like.device)
    n = 1
    for d in shape:
        n *= int(d)
    b = _SCR.get(name)
    if b is None or b.numel() < n:
        _SCR.pop(name, None)
        b = _SCR[name] = torch.empty(n, dtype=like.dtype, device=like.device)
    return b[:n].view(*shape)


def scratch_bytes() -> int:
    return sum(b.numel() * b.element_size() for b in _SCR.values())


def ae_add_norm(x: torch.Tensor, r, w: torch.Tensor, eps: float, x_tiled: bool, enc: bool, plain_out: bool = False):
    """(normed plaintext [M, N] row-major, new residual [M, N] tiled / encoded)."""
    mod = load()
    M, N = x.shape
    ncb = N // 128
    Mp = mpad(M)
    r0 = POOL.get("r0")
    rnew = slot("r1" if r is not None and r0 is not None and r.data_ptr() == r0.data_ptr() else "r0", M, N, x)
    z = torch.empty(M, N, dtype=torch.bfloat16, device=x.device) if (not ENCALL or plain_out) else rnew
    part = torch.empty(Mp, ncb, dtype=torch.float32, device=x.device)
    ep = epoch(x.device)
    if MARKERS and POOL_BUFS:
        if r is None:                     # first op of a forward: scrub stale rows, start-of-forward marker
            scrub(M, x.device)
        mark(x.device, start=r is None)
    if M and ENCALL:
        import triton
        h = slot("h", M, N, x) if not plain_out else z
        mod._ae_add_kernel[(Mp // 16, ncb)](
            x, r if r is not None else x, rnew, z, part, M, N, ncb, cid(x), cid(r) if r is not None else 0,
            cid(rnew), ep, MASTER, X_TILED=x_tiled, HAS_R=r is not None, ENC=enc, STORE_Z=False, num_warps=4)
        mod._ae_norm_t_kernel[(Mp // 16, ncb)](rnew, part, w, h, M, N, ncb, eps, cid(rnew), cid(h) if not plain_out else 0,
                                               ep, MASTER, NCB=triton.next_power_of_2(ncb), ENC=enc,
                                               OUT_TILED=not plain_out, num_warps=4)
        return h, rnew
    if M:
        mod._ae_add_kernel[(Mp // 16, ncb)](
            x, r if r is not None else x, rnew, z, part, M, N, ncb, cid(x), cid(r) if r is not None else 0,
            cid(rnew), ep, MASTER, X_TILED=x_tiled, HAS_R=r is not None, ENC=enc, num_warps=4)
        import triton
        mod._ae_norm_kernel[(Mp // 16, ncb)](z, part, w, M, N, ncb, eps, NCB=triton.next_power_of_2(ncb),
                                             num_warps=4)
    return z, rnew


def ae_silu(gu: torch.Tensor, enc: bool) -> torch.Tensor:
    mod = load()
    M, N2 = gu.shape
    I = N2 // 2
    out = torch.empty(M, I, dtype=torch.bfloat16, device=gu.device)
    if M:
        mod._ae_silu_kernel[(mpad(M) // 16, I // 128)](gu, out, M, I, N2 // 128, cid(gu), epoch(gu.device),
                                                       MASTER, ENC=enc, num_warps=4)
    return out


# --------------------------------------------------------------------------- vLLM integration
_OPMODE = [None]


def _pop_opmode():
    if _OPMODE[0] is not None:
        m, _OPMODE[0] = _OPMODE[0], None
        m.__exit__(None, None, None)


