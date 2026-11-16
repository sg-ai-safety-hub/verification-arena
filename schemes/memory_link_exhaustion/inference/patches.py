"""Encoded twins of the vLLM kernels the deterministic baseline uses, plus the custom decode-ahead GEMM path.

Bit-exactness rule: inference must feed the *same* kernel the *same* decoded values as baseline inference. So instead
of writing new kernels we copy vLLM's own source (this exact installed version), insert the decode right after the
loads (or the encode right before the KV stores), and import the copy from a generated file (Triton needs real source
files). Every substitution must match exactly once, so a vLLM upgrade fails loudly instead of silently.

Final configuration (scripts/reproduce/reference.env):
  weights   sloth64-encoded W (encodings/sloth64.py), decoded ahead of every matmul (VARENA_CUDA_GEMM=1,
            VARENA_BFLY_AHEAD_M=1): gate_up at M <= VARENA_WFUSE_GU_M on the fused decode-in-GEMM kernel
            (gpu/cuda/slothgemm*.cu); W above VARENA_WDEC_SLICE_MB in column slices; the rest whole into one scratch,
            then vLLM's batch-invariant persistent matmul behind the challenge gate (VARENA_MM_GATE)
  KV        kernel_unified_attention decodes sloth64 K / V tiles after the loads (row-base keystream);
            reshape_and_cache_kernel_flash encodes on the KV cache write
  embedding stays plain (see inference/prover.py w_layout)
Design: schemes/memory_link_exhaustion/docs/DESIGN.md.
"""

from __future__ import annotations

import hashlib
from typing import Optional
import importlib.util
import inspect
import os
import sys

import torch

GEN_DIR = os.environ.get("VARENA_GEN", "/tmp/varena_gen")
# prefill-sized matmuls decode the weight into scratch first (0 = always fused)

JIT_HELPERS = '''

@triton.jit
def _varena_fmix(x):
    x = x * 0x85EBCA6B
    x = x ^ (x >> 13)
    x = x * 0xC2B2AE35
    x = x ^ (x >> 16)
    return x


@triton.jit
def _varena_dec16(x, off, key, mask):
    """XOR 16-bit values with the keystream at element offsets `off` (masked lanes kept)."""
    ks = (_varena_fmix(off.to(tl.uint32) ^ key.to(tl.uint32)) & 0xFFFF).to(tl.uint16)
    y = (x.to(tl.uint16, bitcast=True) ^ ks).to(tl.bfloat16, bitcast=True)
    return tl.where(mask, y, x)


@triton.jit
def _varena_decw(b, ks_k, ks_n, key, mask):
    """Decode a B tile of W^T: B[k, n] = W[n, k]; weight keystream
    low16(fmix(n ^ key) ^ fmix(fmix(k ^ key) ^ 0x7F4A7C15)) (see schemes.memory_link_exhaustion.encodings.ref.encode_w)."""
    ku = key.to(tl.uint32)
    kc = _varena_fmix(_varena_fmix(ks_k.to(tl.uint32) ^ ku) ^ 0x7F4A7C15)
    kr = _varena_fmix(ks_n.to(tl.uint32) ^ ku)
    ks = ((kc[:, None] ^ kr[None, :]) & 0xFFFF).to(tl.uint16)
    y = (b.to(tl.uint16, bitcast=True) ^ ks).to(tl.bfloat16, bitcast=True)
    return tl.where(mask, y, b)


@triton.jit
def _varena_enc16(x, off, key):
    ks = (_varena_fmix(off.to(tl.uint32) ^ key.to(tl.uint32)) & 0xFFFF).to(tl.uint16)
    return (x.to(tl.bfloat16).to(tl.uint16, bitcast=True) ^ ks).to(tl.bfloat16, bitcast=True)
'''


def _sub(src: str, old: str, new: str, count: int = 1) -> str:
    n = src.count(old)
    if n != count:
        raise RuntimeError(f"varena patch: expected {count} match(es), found {n}:\n{old[:200]}")
    return src.replace(old, new)


def _sub_after(src: str, marker: str, old: str, new: str) -> str:
    """Replace the first `old` after the (unique) `marker`."""
    if src.count(marker) != 1:
        raise RuntimeError(f"varena patch: marker matched {src.count(marker)} times: {marker}")
    i = src.index(marker)
    j = src.find(old, i)
    if j < 0:
        raise RuntimeError(f"varena patch: not found after {marker}:\n{old[:200]}")
    return src[:j] + new + src[j + len(old):]


def _write_and_import(name: str, src: str):
    os.makedirs(GEN_DIR, exist_ok=True)
    h = hashlib.sha1(src.encode()).hexdigest()[:10]
    path = os.path.join(GEN_DIR, f"{name}_{h}.py")
    if not os.path.exists(path):
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(src)
        os.replace(tmp, path)
    modname = f"varena_gen_{name}"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- matmul
def build_matmul():
    import vllm.model_executor.determinism.batch_invariant as bi
    k = inspect.getsource(getattr(bi.matmul_kernel_persistent, "fn", bi.matmul_kernel_persistent))
    k = _sub(k, "def matmul_kernel_persistent(", "def varena_matmul_kernel_persistent(")
    k = _sub(k, "    HAS_BIAS: tl.constexpr,\n):", "    HAS_BIAS: tl.constexpr,\n    ENC_KEY,\n):")
    k = _sub(k, "            accumulator = tl.dot(a, b, accumulator)",
             "            b = _varena_decw(b, offs_k, offs_bn, ENC_KEY,\n"
             "                             offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K)\n"
             "            accumulator = tl.dot(a, b, accumulator)")
    w = inspect.getsource(bi.matmul_persistent)
    w = _sub(w, "def matmul_persistent(", "def varena_matmul_persistent(enc_key, ")
    w = _sub(w, "matmul_kernel_persistent[grid](", "varena_matmul_kernel_persistent[grid](")
    w = _sub(w, "        HAS_BIAS=bias is not None,\n", "        HAS_BIAS=bias is not None,\n        ENC_KEY=enc_key,\n")
    w = _sub(w, "    if not _BLOCK_SIZES_RESOLVED:\n        _resolve_block_sizes()\n",
             "    if not _bi._BLOCK_SIZES_RESOLVED:\n        _bi._resolve_block_sizes()\n")
    w = w.replace("_fp16_block_size_n", "_bi._fp16_block_size_n").replace(
        "_fp32_block_size_n", "_bi._fp32_block_size_n").replace("_fp32_num_stages", "_bi._fp32_num_stages")
    head = ("import torch\n"
            "import vllm.model_executor.determinism.batch_invariant as _bi\n"
            "from vllm.model_executor.determinism.batch_invariant import (\n"
            "    _compute_pid, _matmul_launch_metadata, _get_matmul_config, num_compute_units)\n"
            "from vllm.triton_utils import tl, triton\n")
    return _write_and_import("encmm", head + JIT_HELPERS + "\n\n" + k + "\n\n" + w)


def _kvbf_src():
    """Butterfly KV helpers for the copied kernels (VARENA_KVENC=bfly); the flag is a module constant."""
    from schemes.memory_link_exhaustion.encodings import kvbfly
    on = kvbfly.KVENC == "bfly"
    sl = kvbfly.KVENC == "sloth"
    from schemes.memory_link_exhaustion.encodings import sloth64
    return ("\n" + (kvbfly.triton_src() if on else "") + f"\n_VARENA_KVBF = {on}\n_VARENA_KVSL = {sl}\n"
            + (sloth64.TRITON_HELPERS if sl else ""))


# --------------------------------------------------------------------------- attention
def build_attention():
    import vllm.v1.attention.ops.triton_unified_attention as ua
    src = inspect.getsource(ua)
    i = src.index("def kernel_unified_attention(")
    j = src.index("\n):\n", i)
    src = src[:j] + ("\n    ENC_KEY=0,\n    ENC: tl.constexpr = False,\n    KVBF: tl.constexpr = False,\n    KVSL: tl.constexpr = False,"
                     "\n    seq_lo=0,\n    seq_hi=1073741824,") + src[j:]
    # KV decode-ahead runs a batch in chunks of sequences (one scratch): programs of other sequences do nothing
    src = _sub(src, "    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:\n        return\n",
               "    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:\n        return\n"
               "    if (seq_idx < seq_lo) | (seq_idx >= seq_hi):\n        return\n")
    src = _sub(src, """            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            )
""", """            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            )
            if ENC:
                if KVBF:
                    # butterfly KV (schemes.memory_link_exhaustion.encodings.kvbfly): whole chunks of one head; a tile
                    # may span several blocks (vLLM's own tile sizes), so each chunk looks up its own block
                    tl.static_assert(HEAD_SIZE_PADDED == 128)
                    ks_k = _kvbf_tile_bt(ENC_KEY, block_tables_ptr + block_table_offset, j, kv_head_idx, 0,
                                         max_seq_prefix_len, TILE_SIZE, BLOCK_SIZE)
                    ks_v = _kvbf_tile_bt(ENC_KEY, block_tables_ptr + block_table_offset, j, kv_head_idx, 1,
                                         max_seq_prefix_len, TILE_SIZE, BLOCK_SIZE)
                    # identity gathers re-lay the keystream into the K/V load layouts, so the XOR is
                    # elementwise in the native layout (as with fmix). Combining K/V directly with the
                    # network's reshape/join output made Triton pick kWidth=4 for the dot operands, which
                    # permutes k inside tl.dot (different summation order: not bit-identical to stock)
                    ks_k = tl.gather(tl.trans(ks_k), offs_d[:, None] + 0 * offs_t[None, :], 0)
                    ks_v = tl.gather(ks_v, offs_d[None, :] + 0 * offs_t[:, None], 1)
                    K_load = tl.where(dim_mask[:, None] & tile_mask[None, :],
                                      (K_load.to(tl.uint16, bitcast=True) ^ ks_k).to(
                                          tl.bfloat16, bitcast=True), K_load)
                    V_load = tl.where(dim_mask[None, :] & tile_mask[:, None],
                                      (V_load.to(tl.uint16, bitcast=True) ^ ks_v).to(
                                          tl.bfloat16, bitcast=True), V_load)
                elif KVSL:
                    # sloth KV (experiment): one modular squaring per 64-bit word (4 consecutive dims); the decode
                    # delta is re-laid into the load layouts with identity gathers (bit-exact tl.dot, as above)
                    tl.static_assert(HEAD_SIZE_PADDED == 128)
                    sl_seed = (ENC_KEY.to(tl.uint64) << 32) | 0x4B56
                    v16 = V_load.to(tl.uint16, bitcast=True)
                    ks_v = _sl_dec_tile(v16, v_offset + HEAD_SIZE, sl_seed) ^ v16
                    k16t = tl.trans(K_load.to(tl.uint16, bitcast=True))
                    ks_k = _sl_dec_tile(k16t, tl.trans(k_offset), sl_seed) ^ k16t
                    ks_k = tl.gather(tl.trans(ks_k), offs_d[:, None] + 0 * offs_t[None, :], 0)
                    ks_v = tl.gather(ks_v, offs_d[None, :] + 0 * offs_t[:, None], 1)  #KVSL_GATHER
                    K_load = tl.where(dim_mask[:, None] & tile_mask[None, :],
                                      (K_load.to(tl.uint16, bitcast=True) ^ ks_k).to(
                                          tl.bfloat16, bitcast=True), K_load)
                    V_load = tl.where(dim_mask[None, :] & tile_mask[:, None],
                                      (V_load.to(tl.uint16, bitcast=True) ^ ks_v).to(
                                          tl.bfloat16, bitcast=True), V_load)
                else:
                    K_load = _varena_dec16(K_load, k_offset, ENC_KEY,
                                           dim_mask[:, None] & tile_mask[None, :])
                    V_load = _varena_dec16(V_load, v_offset + HEAD_SIZE, ENC_KEY,
                                           dim_mask[None, :] & tile_mask[:, None])
""")
    # VARENA_ATT_CLAMP=0 (final configuration): vLLM's own tile sizes (the sloth64 and butterfly KV decodes both handle
    # a tile that spans blocks). VARENA_ATT_CLAMP=1 clamps the tiles to the block size; the baseline must then use the
    # same setting.
    if os.environ.get("VARENA_ATT_CLAMP", "1") == "1":
        src = _sub(src, "    if use_td:\n        TILE_SIZE_PREFILL = min(TILE_SIZE_PREFILL, block_size)\n",
                   "    TILE_SIZE_PREFILL = min(TILE_SIZE_PREFILL, block_size)\n"
                   "    TILE_SIZE_DECODE = min(TILE_SIZE_DECODE, block_size)\n"
                   "    if use_td:\n        TILE_SIZE_PREFILL = min(TILE_SIZE_PREFILL, block_size)\n")
    # VARENA_ATT_BLOCK_M_PREFILL (reference configuration: 32; 0 = vLLM's own 16): query rows per program in batches
    # that carry prefill. vLLM takes 16 rows = 3 query tokens x 5 heads, and every program decodes its whole prefix
    # again, so a 512-token prompt decodes its cache about 170 times over. 32 rows halve that and stay bit-identical
    # to stock (+2.3% throughput, 14 Nov); 64 rows are faster still but change the outputs.
    _bm = int(os.environ.get("VARENA_ATT_BLOCK_M_PREFILL", "0") or 0)
    if _bm:
        src = _sub(src, "    BLOCK_Q = BLOCK_M // num_queries_per_kv\n\n    # Tuned launch parameters",
                   "    if max_seqlen_q > 1:\n        BLOCK_M = %d\n"
                   "    BLOCK_Q = BLOCK_M // num_queries_per_kv\n\n    # Tuned launch parameters" % _bm)
    i = src.index("    kernel_unified_attention[grid](")
    j = src.index("        **launch_kwargs,\n    )", i)
    src = src[:j] + ("        ENC_KEY=_VARENA_KEY[0] or 0,\n"
                     "        ENC=_VARENA_KEY[0] is not None,\n"
                     "        KVBF=_VARENA_KVBF,\n"
                     "        KVSL=_VARENA_KVSL,\n"
                     "        seq_lo=_VARENA_SEQ[0],\n"
                     "        seq_hi=_VARENA_SEQ[1],\n") + src[j:]
    # activation encoding (schemes.memory_link_exhaustion.encodings.actenc2, VARENA_AE_ATTN): the attention output is stored tiled into its pool
    # slot, encoded with the address-keyed keystream (per row: the network of the row's chunk, row selected)
    src = _sub(src, """            tl.store(
                output_ptr + output_offset,
                acc,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            )
""", """            if AE_OUT:
                _ae_attn_store(output_ptr, acc, query_offset_0, query_offset_1, offs_d,
                               dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
                               ae_epoch, ae_master, ae_cid, ae_ncb, AE_ENC, BLOCK_M)
            else:
                tl.store(
                    output_ptr + output_offset,
                    acc,
                    mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
                )
""")
    src = _sub(src, "    tl.store(output_ptr + output_offset, acc, mask=dim_mask)\n",
               "    if AE_OUT:\n"
               "        _ae_attn_store(output_ptr, tl.reshape(acc, (1, HEAD_SIZE_PADDED)),\n"
               "                       tl.full((1,), 0, tl.int64) + query_token_idx,\n"
               "                       tl.full((1,), 0, tl.int64) + query_head_idx, offs_d, dim_mask[None, :],\n"
               "                       ae_epoch, ae_master, ae_cid, ae_ncb, AE_ENC, 1)\n"
               "    else:\n"
               "        tl.store(output_ptr + output_offset, acc, mask=dim_mask)\n")
    i = src.index("def kernel_unified_attention(")
    j = src.index("\n):\n", i)
    src = src[:j] + _AE_PARAMS + src[j:]
    # the activation-pool chunk id is address-derived: never specialise it (recompiles mid-run = stalls)
    k0 = src.rindex("@triton.jit", 0, src.index("def kernel_unified_attention("))
    k1 = src.index("\n", k0)
    deco = src[k0:k1]
    # ENC_KEY: per-layer KV key, derived from the verifier's per-run master key: specialising on it (divisibility /
    # ==1) compiled ~15 new attention variants in every run (mid-run JIT = 15-40 s engine stalls)
    dns = "do_not_specialize=['ae_cid', 'ENC_KEY', 'ae_master']"
    src = src[:k0] + (deco[:-1] + ", " + dns + ")" if deco.endswith(")") else deco + "(" + dns + ")") + src[k1:]
    i = src.index("def reduce_segments(")
    j = src.index("\n):\n", i)
    src = src[:j] + _AE_PARAMS + src[j:]
    i = src.index("    kernel_unified_attention[grid](")
    j = src.index("        **launch_kwargs,\n    )", i)
    src = src[:j] + _AE_KW + src[j:]
    i = src.index("        reduce_segments[(q.shape[0], num_query_heads)](")
    j = src.index("            USE_FP8=output_scale is not None,\n        )", i) + len("            USE_FP8=output_scale is not None,\n")
    src = src[:j] + _AE_KW.replace("        ", "            ") + src[j:]
    # freeze-in-place (VARENA_ATT_GATE): every attention program reads the challenge gate first; while it is raised a
    # program exits without computing (its SM frees within one tile of the programs already running) and leaves its
    # done flag clear. Resume launches behind a gate wait recompute only the programs whose flag is clear (the
    # last resume never yields). Per-program arithmetic is unchanged (bit-identical).
    i = src.index("def kernel_unified_attention(")
    j = src.index("\n):\n", i)
    src = src[:j] + ("\n    att_gate=None,\n    att_done=None,\n    ATT_GATE: tl.constexpr = False,\n"
                     "    ATT_RESUME: tl.constexpr = False,\n    ATT_NOYIELD: tl.constexpr = False,") + src[j:j + 4] + (
        "    if ATT_GATE:\n"
        "        att_lin = tl.program_id(0) + tl.num_programs(0) * (tl.program_id(1) + tl.num_programs(1) * tl.program_id(2))\n"
        "        if ATT_RESUME:\n"
        "            if tl.load(att_done + att_lin) != 0:\n"
        "                return\n"
        "        if not ATT_NOYIELD:\n"
        "            if tl.atomic_add(att_gate, 0) != 0:\n"
        "                return\n") + src[j + 4:]
    # in-loop yield: a program that sees the gate raised mid-sequence exits without storing (its done flag stays
    # clear) and the resume recomputes it from the start: a long decode program no longer runs through a challenge
    k = src.index("def kernel_unified_attention(")
    # (Triton rejects a return inside a loop: the loop becomes a while loop that stops once the gate is seen, and
    # the program returns after it without storing)
    li = src.index("    for j in range(loop_lo, loop_hi):\n", k)
    le = li + len("    for j in range(loop_lo, loop_hi):\n")
    be = src.index("\n    # ---- Epilogue", le)                     # end of the loop body
    body = src[le:be + 1]
    if os.environ.get("VARENA_ATT_GATE", "0") == "1":    # only then: a while loop is not software-pipelined
      src = (src[:li] + "    att_stop = tl.full((), 0, tl.int32)\n    j = loop_lo\n"
           "    while (j < loop_hi) & (att_stop == 0):\n" + body +
           "        j += 1\n"
           "        if ATT_GATE and not ATT_NOYIELD:\n"
           "            if (j - loop_lo) % 4 == 0:\n"
           "                att_stop = (tl.atomic_add(att_gate, 0) != 0).to(tl.int32)\n"
           "    if ATT_GATE:\n"
           "        if att_stop != 0:\n"
           "            return\n" + src[be + 1:])
    k = src.index("def kernel_unified_attention(")
    e = src.index("\n@triton.jit", k)
    src = src[:e].rstrip("\n") + ("\n    if ATT_GATE:\n        tl.store(att_done + tl.program_id(0) + tl.num_programs(0) * "
                                  "(tl.program_id(1) + tl.num_programs(1) * tl.program_id(2)), 1)\n\n") + src[e:]
    i = src.index("    kernel_unified_attention[grid](")
    j = src.index("\n    )\n", i) + len("\n    )\n")
    call = src[i:j]
    call = call.replace("        **launch_kwargs,\n    )", "        **launch_kwargs, **_att_kw,\n"
                        "        ATT_RESUME=_att_pass > 0, ATT_NOYIELD=_att_pass == _att_n - 1,\n    )")
    call = "".join("    " + ln if ln.strip() else ln for ln in call.splitlines(True))
    src = src[:i] + ("    _att_kw = _varena_att_gate(grid, q.device)\n"
                     "    _att_n = 2 + _ATT_RESUME_N if _att_kw else 1\n"
                     "    for _att_pass in range(_att_n):\n"
                     "        if _att_pass:\n"
                     "            _varena_gate_wait(q.device)\n") + call + src[j:]
    src += """

import os as _att_os
_ATT_ON = _att_os.environ.get("VARENA_ATT_GATE", "0") == "1"
_ATT_RESUME_N = int(_att_os.environ.get("VARENA_AE_RESUME_N", "0"))


def _varena_att_gate(grid, dev):
    if not _ATT_ON:
        return {}
    from schemes.memory_link_exhaustion.encodings import actenc2 as _AEg
    n = 1
    for g_ in grid:
        n *= int(g_)
    # one preallocated done-flag buffer per device (a fresh allocation per launch grew the graph pools by ~0.2 GB)
    buf = _ATT_DONE.get(dev)
    if buf is None or buf.numel() < n:
        buf = _ATT_DONE[dev] = torch.zeros(max(n, 1 << 20), dtype=torch.int32, device=dev)
    d = buf[:n]
    d.zero_()
    return {"att_gate": _AEg.gate_dev(dev), "att_done": d, "ATT_GATE": True}


_ATT_DONE = {}


def _varena_gate_wait(dev):
    from schemes.memory_link_exhaustion.encodings import actenc2 as _AEg
    _AEg.load()._ae_gate_wait_kernel[(1,)](_AEg.gate_dev(dev), num_warps=1)
"""
    from schemes.memory_link_exhaustion.encodings import kvbfly
    from schemes.memory_link_exhaustion.encodings import actenc2 as _AE2, sloth32 as _S32
    src += JIT_HELPERS + _kvbf_src() + ("" if kvbfly.KVENC == "bfly" else "\n" + kvbfly.triton_src()) \
        + _S32.TRITON_HELPERS + f"\nAE_S32 = tl.constexpr({_AE2.AE_S32})\n" + _AE_ATTN_SRC \
        + "\n_VARENA_KEY = [None]\n_VARENA_AE = [None]\n"
    # sloth KV decode: row-base word index + per-row keystream product
    a_ = "ks_v = _sl_dec_tile(v16, v_offset + HEAD_SIZE, sl_seed) ^ v16"
    b_ = "ks_k = _sl_dec_tile(k16t, tl.trans(k_offset), sl_seed) ^ k16t"
    assert src.count(a_) == 1 and src.count(b_) == 1
    src = src.replace(a_, "ks_v = _sl_dec_tile_fast(v16, physical_block_idx * stride_v_cache_0 + kv_head_idx * "
                          "stride_v_cache_2 + (seq_offset % BLOCK_SIZE) * stride_v_cache_1 + HEAD_SIZE, sl_seed) ^ v16")
    src = src.replace(b_, "ks_k = _sl_dec_tile_fast(k16t, physical_block_idx * stride_k_cache_0 + kv_head_idx * "
                          "stride_k_cache_2 + (seq_offset % BLOCK_SIZE) * stride_k_cache_1, sl_seed) ^ k16t")
    # Sloth KV decode-ahead (VARENA_KV_AHEAD=0 turns it off: the in-kernel decode, kept as the fallback). The blocks a batch reads are decoded once per layer into the weight
    # decode's scratch (encodings.sloth64.kv_ahead) and the attention kernel runs with encoding off on the scratch
    # through a remapped block table: the stock arithmetic on plaintext, so bit-identical, and each block is decoded
    # once instead of once per query block. The batch goes in chunks of sequences that fit the scratch
    # (VARENA_WDEC_SLICE_MB), so no memory is added. Falls back to the in-kernel decode when the layout is not the
    # prover's interleaved [NB, BS, HKV, 2 * D] cache, a sequence alone exceeds the scratch, or a 3D-path batch
    # would need more than one chunk.
    src = _sub(src, "\ndef unified_attention(\n", "\ndef _unified_attention_inner(\n")
    src += _KV_AHEAD_SRC.replace("AHEAD_YIELD_FLAG", str(os.environ.get("VARENA_MM_GATE", "0") == "1" and
                                                         os.environ.get("VARENA_AE_GATE", "1") == "1")).replace(
        "AHEAD_ON_FLAG", str(os.environ.get("VARENA_KV_AHEAD", "1") == "1")).replace(
        "AHEAD_CAP_BYTES", str(int(os.environ.get("VARENA_WDEC_SLICE_MB", "0")) << 20))
    return _write_and_import("encattn", src)


_KV_AHEAD_SRC = '''

_VARENA_SEQ = [0, 1073741824]
_VARENA_AHEAD = AHEAD_ON_FLAG
_VARENA_AHEAD_CAP = AHEAD_CAP_BYTES
_VARENA_AHEAD_STAT = {"ahead": 0, "chunks": 0, "inner": 0, "why": {}}
# yield to challenges: before each chunk's decode and before its attention launch a one-program kernel waits while the
# challenge gate is raised (encodings.actenc2 gate, the one the matmul gate reads); on when the prover runs gated
_VARENA_AHEAD_YIELD = AHEAD_YIELD_FLAG


def unified_attention(*args, **kw):
    key = _VARENA_KEY[0]
    if not (_VARENA_AHEAD and _VARENA_KVSL and key is not None) or args:
        return _unified_attention_inner(*args, **kw)
    st = _VARENA_AHEAD_STAT
    k, v, bt, sk = kw["k"], kw["v"], kw["block_table"], kw["seqused_k"]
    why = None
    if k.dim() != 4 or bt.dim() != 2:
        why = "shape"
    else:
        NB, BS, HKV, D = k.shape
        kel = BS * HKV * 2 * D
        num_seqs, maxb = bt.shape
        if not (k.stride() == (kel, HKV * 2 * D, 2 * D, 1) and v.stride() == k.stride() and k.element_size() == 2
                and v.data_ptr() - k.data_ptr() == 2 * D and bt.is_contiguous() and bt.dtype == torch.int32
                and sk.dtype == torch.int32 and sk.is_contiguous() and kel % 4 == 0):
            why = "layout"
        else:
            # the scratch is the weight decode's, at the size it already has: it is never grown or replaced here
            # (captured CUDA graphs hold its address)
            from schemes.memory_link_exhaustion.encodings import actenc2 as _AEs
            sb = _AEs._SCR.get("wdec_s")
            cap_el = min(sb.numel(), _VARENA_AHEAD_CAP // 2) if (sb is not None and sb.element_size() == 2) else 0
            g = (cap_el // kel) // max(1, maxb)
            thr = kw.get("seq_threshold_3D")
            if g < 1:
                why = "no scratch yet" if cap_el == 0 else "context longer than the scratch"
            elif g < num_seqs and thr is not None and num_seqs <= thr and kw.get("max_seqlen_q", 2) <= 1:
                why = "3D path over more than one chunk"
    _n = st["ahead"] + st["inner"]
    if _n in (1, 2000, 60000):
        import sys as _sys
        print(f"[varena] KV decode-ahead after {_n} attention calls: {st}", file=_sys.stderr, flush=True)
    if why is not None:
        st["inner"] += 1
        st["why"][why] = st["why"].get(why, 0) + 1
        return _unified_attention_inner(**kw)
    from schemes.memory_link_exhaustion.encodings import sloth64 as _S64
    src = k.as_strided((NB * kel,), (1,)).view(torch.int64)
    seed = ((int(key) & 0xFFFFFFFF) << 32) | 0x4B56
    sv = sb.view(k.dtype)
    st["ahead"] += 1
    try:
        _VARENA_KEY[0] = None                        # the kernel reads plaintext from the scratch
        for a in range(0, num_seqs, g):
            b = min(num_seqs, a + g)
            scr = sv[:(b - a) * maxb * kel].view((b - a) * maxb, BS, HKV, 2 * D)
            if _VARENA_AHEAD_YIELD:
                _varena_gate_wait(k.device)          # a challenge in flight goes first (as the matmul's gate)
            _S64.kv_ahead(src, bt, sk, scr, a, b - a, maxb, kel // 4, BS, seed)
            if _VARENA_AHEAD_YIELD:
                _varena_gate_wait(k.device)
            kw2 = dict(kw)
            kw2["k"], kw2["v"] = scr[..., :D], scr[..., D:]
            # block j of sequence s is scratch block (s - a) * maxb + j. Built here on every call, not cached: under
            # CUDA graph capture a cached tensor would be filled only when the graph that created it replays
            kw2["block_table"] = torch.arange(-a * maxb, (num_seqs - a) * maxb, dtype=torch.int32,
                                              device=bt.device).view(num_seqs, maxb)
            _VARENA_SEQ[0], _VARENA_SEQ[1] = a, b
            _unified_attention_inner(**kw2)
            st["chunks"] += 1
    finally:
        _VARENA_KEY[0] = key
        _VARENA_SEQ[0], _VARENA_SEQ[1] = 0, 1073741824
'''


_AE_PARAMS = ("\n    AE_OUT: tl.constexpr = False,\n    AE_ENC: tl.constexpr = False,\n    ae_epoch=None,\n"
              "    ae_master=0,\n    ae_cid=0,\n    ae_ncb=0,")
_AE_KW = ("        AE_OUT=_VARENA_AE[0] is not None,\n"
          "        AE_ENC=_VARENA_AE[0] is not None and _VARENA_AE[0][0],\n"
          "        ae_epoch=_VARENA_AE[0][1] if _VARENA_AE[0] is not None else None,\n"
          "        ae_master=_VARENA_AE[0][2] if _VARENA_AE[0] is not None else 0,\n"
          "        ae_cid=_VARENA_AE[0][3] if _VARENA_AE[0] is not None else 0,\n"
          "        ae_ncb=_VARENA_AE[0][4] if _VARENA_AE[0] is not None else 0,\n")
_AE_ATTN_SRC = """


@triton.jit
def _ae_attn_store(out, acc, tok, head, offs_d, mask, epoch, master, cid0, ncb, ENC: tl.constexpr, R: tl.constexpr):
    # rows = (token, head) pairs; element (token, head, d) of the [tokens, heads * 128] output goes to tiled chunk
    # (token // 16, head), row token % 16, column d (head_size 128 = one chunk width)
    tok = tok.to(tl.int64)
    head = head.to(tl.int64)
    v = acc.to(tl.bfloat16)
    if ENC and AE_S32:
        # sloth32 pool (experiment): word index = chunk id * 1024 + row-in-chunk * 64 + d / 2
        seed = _kvbf_fmix(tl.load(epoch).to(tl.uint32) ^ master.to(tl.uint32))
        cid = cid0 + (tok // 16) * ncb + head
        R2: tl.constexpr = R
        w = tl.arange(0, 64)[None, :]
        cid2 = cid[:, None] + 0 * w
        wl = ((tok % 16) * 64)[:, None] + w
        c = cid2.to(tl.int64)
        lo = (c & 0xFFFFF).to(tl.uint32) * 1024 + wl.to(tl.uint32)
        hi = (c >> 20).to(tl.uint32)
        ks = _s32_fmix(seed.to(tl.uint32) ^ (lo * 0x9E3779B9) ^ _s32_fmix(hi ^ 0x85EBCA6B))
        lo16, hi16 = tl.split(tl.reshape(v.to(tl.uint16, bitcast=True).to(tl.uint32), (R2, 64, 2)))
        y = _s32_sigma_inv((lo16 | (hi16 << 16)) ^ ks)
        v = tl.reshape(tl.join(y & 0xFFFF, y >> 16), (R2, 128)).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    elif ENC:
        seed = _kvbf_fmix(tl.load(epoch).to(tl.uint32) ^ master.to(tl.uint32))
        sv = (cid0 + (tok // 16) * ncb + head).to(tl.uint32)
        w0, w1, w2, w3 = _kvbf_net(sv, seed, R)
        u = (((tok % 16) * 16)[:, None] + tl.arange(0, 16)[None, :]).to(tl.int32)
        ks = _kvbf_u16(tl.gather(w0, u, 1), tl.gather(w1, u, 1), tl.gather(w2, u, 1), tl.gather(w3, u, 1), R)
        v = (v.to(tl.uint16, bitcast=True) ^ ks).to(tl.bfloat16, bitcast=True)
    off = ((tok // 16) * ncb + head)[:, None] * 2048 + (tok % 16)[:, None] * 128 + offs_d[None, :]
    tl.store(out + off, v, mask=mask)
"""


def build_cache_write():
    import vllm.v1.attention.ops.triton_reshape_and_cache_flash as rc
    src = inspect.getsource(rc)
    src = _sub_after(src, "def reshape_and_cache_kernel_flash(", "    # tune parameters\n    TILE_SIZE: tl.constexpr,\n):",
               "    # tune parameters\n    TILE_SIZE: tl.constexpr,\n    ENC_KEY=0,\n    ENC: tl.constexpr = False,\n"
               "    KVBF: tl.constexpr = False,\n    KVSL: tl.constexpr = False,\n):")
    src = _sub_after(src, "def reshape_and_cache_kernel_flash(", """    tl.store(
        key_cache_ptr + tgt_idx_k,
        key_tile,
        mask=tile_pos < (num_heads * head_size),
    )
""", """    if ENC:
        if KVBF:
            # butterfly KV: one program writes one token row of every kv head (TILE_SIZE = heads * 128)
            tl.static_assert(TILE_SIZE == num_heads * head_size)
            tl.static_assert(head_size == 128)
            ks_k, ks_v = _kvbf_rows(ENC_KEY, block_idx, block_offset, num_heads)
            key_tile = (key_tile.to(tl.uint16, bitcast=True) ^ ks_k).to(tl.bfloat16, bitcast=True)
            value_tile = (value_tile.to(tl.uint16, bitcast=True) ^ ks_v).to(tl.bfloat16, bitcast=True)
        elif KVSL:
            # sloth KV (experiment): square-root permutation per 64-bit word; word index = element index / 4 in
            # the layer's cache (V at +head_size in the same row), seed from the layer key
            tl.static_assert(TILE_SIZE == num_heads * head_size)
            sl_seed = (ENC_KEY.to(tl.uint64) << 32) | 0x4B56
            key_tile = tl.reshape(_sl_enc_tile(tl.reshape(key_tile.to(tl.uint16, bitcast=True), (1, TILE_SIZE)),
                                               tl.reshape(tgt_idx_k, (1, TILE_SIZE)), sl_seed),
                                  (TILE_SIZE,)).to(tl.bfloat16, bitcast=True)
            value_tile = tl.reshape(_sl_enc_tile(tl.reshape(value_tile.to(tl.uint16, bitcast=True), (1, TILE_SIZE)),
                                                 tl.reshape(tgt_idx_v + head_size, (1, TILE_SIZE)), sl_seed),
                                    (TILE_SIZE,)).to(tl.bfloat16, bitcast=True)
        else:
            key_tile = _varena_enc16(key_tile, tgt_idx_k, ENC_KEY)
            value_tile = _varena_enc16(value_tile, tgt_idx_v + head_size, ENC_KEY)
    tl.store(
        key_cache_ptr + tgt_idx_k,
        key_tile,
        mask=tile_pos < (num_heads * head_size),
    )
""")
    i = src.index("    reshape_and_cache_kernel_flash[grid](")
    j = src.index("        TILE_SIZE=TILE_SIZE,\n", i)
    src = src[:j] + ("        ENC_KEY=_VARENA_KEY[0] or 0,\n"
                     "        ENC=_VARENA_KEY[0] is not None,\n"
                     "        KVBF=_VARENA_KVBF,\n"
                     "        KVSL=_VARENA_KVSL,\n") + src[j:]
    src += JIT_HELPERS + _kvbf_src() + "\n_VARENA_KEY = [None]\n"
    k0 = src.rindex("@triton.jit", 0, src.index("def reshape_and_cache_kernel_flash("))
    k1 = src.index("\n", k0)
    deco = src[k0:k1]
    src = src[:k0] + (deco[:-1] + ", do_not_specialize=['ENC_KEY'])" if deco.endswith(")") else deco + "(do_not_specialize=['ENC_KEY'])") + src[k1:]
    return _write_and_import("enccache", src)


# --------------------------------------------------------------------------- custom CUDA GEMM
# VARENA_CUDA_GEMM: "0" Triton copies (above); "1" custom Hopper kernel (schemes/memory_link_exhaustion/gpu/cuda/encgemm.cu)
# for encoded layers
CUDA_GEMM = os.environ.get("VARENA_CUDA_GEMM", "0")
# stock tiling and schedule for these shapes (BLOCK 128x128x64, 2 x m64n128k16 warpgroups, persistent grid of NUM_SMS
# CTAs, Triton's GROUP_SIZE_M=8 tile order, num_stages 3)
CUDA_STAGES = 3
CUDA_STAGES_BFLY = 6
BFLY_AHEAD_M = int(os.environ.get("VARENA_BFLY_AHEAD_M", "512"))
# VARENA_BFLY_FUSE_KINDS (16 Nov, link experiment): matrix kinds that take the fused butterfly decode GEMM below
# BFLY_AHEAD_M (comma list of _kind names; empty = all). The others always decode ahead into scratch.
BFLY_MT2 = os.environ.get("VARENA_BFLY_MT2", "1") == "1"
FUSE_YIELD = os.environ.get("VARENA_FUSE_YIELD", "0") == "1"
BFLY_FUSE_KINDS = set(k for k in os.environ.get("VARENA_BFLY_FUSE_KINDS", "").split(",") if k)
# sliced decode-ahead threshold / slice size in bytes (0: off): W matrices larger than this decode in slices
WDEC_SLICE = int(os.environ.get("VARENA_WDEC_SLICE_MB", "0")) << 20
# fused sloth decode-on-load GEMM for gate_up at M <= this (0: off); it waits on the challenge gate per output tile
# when the stock matmul does (VARENA_MM_GATE) or the activation kernels are pre-emptible (VARENA_AE_PREEMPT)
WFUSE_GU_M = int(os.environ.get("VARENA_WFUSE_GU_M", "0"))
_WFUSE_SEEN: set = set()
WFUSE_GATE = os.environ.get("VARENA_WFUSE_GATE", "1" if (os.environ.get("VARENA_MM_GATE") == "1" or
                                                         os.environ.get("VARENA_AE_PREEMPT") == "1") else "0") == "1"
# with the gate: -1 = wait in place per output tile (holds every SM: challenge kernels are squeezed onto the reserved
# SMs, 72 ms mixed chases); n >= 0 = pre-emptible like actenc2._launch_mm (CTAs exit when the gate is raised, n extra
# yielding resumes, then a final resume that ignores the gate)
WFUSE_PREEMPT = int(os.environ.get("VARENA_WFUSE_PREEMPT", os.environ.get("VARENA_AE_RESUME_N", "0")))


def cuda_cfg(M: int, N: int, K: int, dec: bool):
    """(bn, nc, stages, ctas): vLLM's stock schedule. Any config is bit-exact (tests/test_encgemm.py)."""
    return (128, 2, CUDA_STAGES, _num_sms())


_SMS = None


def _num_sms():
    global _SMS
    if _SMS is None:
        import vllm.model_executor.determinism.batch_invariant as bi
        _SMS = int(bi.num_compute_units(0))     # what matmul_persistent uses (honours reserve-SMs)
    return _SMS


_CUDA_LINEAR = None
# debugging: layer kinds that keep the Triton path (qkv, o, gate_up, down, lm_head)
CUDA_SKIP = set(filter(None, os.environ.get("VARENA_CUDA_SKIP", "").split(",")))


def _kind(w):
    N, K = w.shape
    if N > 100000:
        return "lm_head"
    if K > 8192:
        return "down"
    if N > 16384:
        return "gate_up"
    return "o" if N == K else "qkv"
_CHECK = [int(os.environ.get("VARENA_CUDA_CHECK", "0")), 0, int(os.environ.get("VARENA_CUDA_CHECK_SKIP", "0")),
          int(os.environ.get("VARENA_CUDA_CAPLOG", "0"))]
# weight data_ptr -> key, filled by the prover when it encodes W. The custom op looks the key
# up at run time instead of taking it as an int argument (torch.compile / vLLM's piecewise
# compilation would otherwise see per-layer int constants in structurally identical graphs).
W_KEYS: dict = {}


def _sealed():
    from schemes.memory_link_exhaustion.inference import prover as _pv
    return _pv.STATE is not None and getattr(_pv.STATE, "sealed", False)


# VARENA_WDEC_GATE=1 (quiet-chase experiment, 16 Nov): a challenge in flight goes first before every W decode-ahead
# launch too (a one-CTA wait on the device gate, as the KV decode-ahead does), not only inside vLLM's matmul
WDEC_GATE = os.environ.get("VARENA_WDEC_GATE", "0") == "1"
AHEAD_AE = os.environ.get("VARENA_AHEAD_AE", "0") == "1"


def _wdec_gate(dev):
    if WDEC_GATE:
        from schemes.memory_link_exhaustion.encodings import actenc2 as _AEw
        _AEw.load()._ae_gate_wait_kernel[(1,)](_AEw.gate_dev(dev), num_warps=1)


def build_cuda_linear():
    global _CUDA_LINEAR
    if _CUDA_LINEAR is not None:
        return _CUDA_LINEAR
    from schemes.memory_link_exhaustion.gpu.cuda import build as CB
    from schemes.memory_link_exhaustion.gpu import kernels as VK  # noqa: F401  (kept: import order / registration)
    ext = CB.load()

    @torch.library.custom_op("varena::cuda_linear", mutates_args=())
    def cuda_linear(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor], key: int,
                    dec: bool) -> torch.Tensor:
        M, K = x.shape
        N = w.shape[0]
        if x.stride(1) != 1 or (x.stride(0) * 2) % 16:
            x = x.contiguous()
        if dec:
            key = W_KEYS[w.data_ptr()]
            if isinstance(key, tuple):              # butterfly-encoded W
                key, ordinal = key
                if WFUSE_GU_M and bias is None and M <= WFUSE_GU_M and _kind(w) == "gate_up" and N % 64 == 0 \
                        and K % 64 == 0 and w.is_contiguous() and \
                        __import__("schemes.memory_link_exhaustion.encodings.ref", fromlist=["x"]).WENC == "sloth":
                    # fused sloth decode in the GEMM (gpu/cuda/slothgemm.cu): W is squared in shared memory, no
                    # decode scratch; same per-element wgmma chain as stock (bit-identical). Used only where it
                    # beats decode-ahead + stock: gate_up at small M (docs/DESIGN.md, Encoding).
                    from schemes.memory_link_exhaustion.encodings import actenc2 as _AEs
                    from schemes.memory_link_exhaustion.encodings import ref as _wref
                    seed = _wref.sloth_seed(key, ordinal)
                    seed = seed - (1 << 64) if seed >= (1 << 63) else seed
                    bn, nc, sl, st, nd = (64, 2, 1, 6, 3) if M <= 128 else (64, 2, 2, 4, 2)
                    gate = _AEs.gate_dev(x.device).data_ptr() if WFUSE_GATE else 0
                    if M not in _WFUSE_SEEN:
                        _WFUSE_SEEN.add(M)
                        import sys as _s
                        print(f"[varena-slgemm] fused gate_up M={M} cfg={(bn, nc, sl, st, nd)} ctas={_num_sms()} "
                              f"gate={'on' if gate else 'off'} preempt={WFUSE_PREEMPT} capturing={torch.cuda.is_current_stream_capturing()}",
                              file=_s.stderr, flush=True)
                    y = CB.load_sl().slgemm(x, w, seed, 0, 1, bn, nc, sl, st, nd, 8, _num_sms(), gate,
                                            WFUSE_PREEMPT if gate else -1)
                    return y
                ahead = M >= BFLY_AHEAD_M or bool(BFLY_FUSE_KINDS and _kind(w) not in BFLY_FUSE_KINDS)
                if ahead and WDEC_SLICE and N * K * 2 > WDEC_SLICE and N % 256 == 0 and \
                        __import__("schemes.memory_link_exhaustion.encodings.ref", fromlist=["x"]).WENC in ("sloth", "bfly"):
                    # sliced decode-ahead (large W, e.g. lm_head): decode a column slice of W^T into a fixed small
                    # scratch and multiply it with the full matrix's config, slice by slice (bit-identical columns)
                    from schemes.memory_link_exhaustion.encodings import actenc2 as _AEs
                    from schemes.memory_link_exhaustion.encodings import ref as _wref
                    S = max(256, (WDEC_SLICE // (K * 2)) // 256 * 256)
                    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
                    for n0 in range(0, N, S):
                        n1 = min(N, n0 + S)
                        if _AEs.WFUSE:
                            # fused: the GEMM decodes the encoded row slice in registers (no scratch)
                            _AEs.ae_matmul(x, w[n0:n1].t(), False, bias=None if bias is None else bias[n0:n1],
                                           c=y[:, n0:n1], tiled=False, cfg_n=N,
                                           wseed=_wref.sloth_seed(key, ordinal), wn0=n0)
                            continue
                        wd = _AEs.scratch("wdec_s", (n1 - n0, K), w)
                        _wdec_gate(w.device)
                        _wref.w_xor(w[n0:n1], wd, key, ordinal, row0=n0)
                        _AEs.ae_matmul(x, wd.t(), False, bias=None if bias is None else bias[n0:n1],
                                       c=y[:, n0:n1], tiled=False, cfg_n=N)
                    return y
                if ahead:
                    # decode-ahead: decode the layer's W once into scratch, then the stock kernel
                    from schemes.memory_link_exhaustion.encodings import actenc2 as _AEs
                    # one decode scratch shared with the sliced path: the leftover holds max(slice, largest
                    # unsliced W) once instead of both
                    wd = _AEs.scratch("wdec_s", tuple(w.shape), w)
                    from schemes.memory_link_exhaustion.encodings import ref as _wref
                    _wdec_gate(w.device)
                    _wref.w_xor(w, wd, key, ordinal)
                    if AHEAD_AE:
                        # VARENA_AHEAD_AE (16 Nov, honest answer gap): the same stock arithmetic through our copy of the
                        # kernel, pre-emptible under VARENA_AE_PREEMPT: during a challenge its CTAs exit and resume
                        # behind a gate wait instead of parking resident at the MM gate
                        y = torch.empty(M, N, dtype=x.dtype, device=x.device)
                        _AEs.ae_matmul(x, wd.t(), False, bias=bias, c=y, tiled=False, cfg_n=N)
                    else:
                        import vllm.model_executor.determinism.batch_invariant as _bi
                        y = _bi.matmul_persistent(x, wd.t(), bias=bias)
                else:
                    # 128 < M <= 256: one CTA per 256 rows (nc 4 = 2 consumer warpgroups x 2 sub-tiles), so the
                    # butterfly keystream of a W tile is generated once, not once per 128-row m-tile
                    nc_, st_ = (4, 4) if (128 < M <= 256 and BFLY_MT2) else (2, CUDA_STAGES_BFLY)
                    if FUSE_YIELD:
                        # VARENA_FUSE_YIELD (16 Nov, honest answer gap): the fused GEMM takes tiles from an atomic
                        # counter and stops at the raised challenge gate (after the tile in hand); a resume launch
                        # behind a gate wait finishes the remaining tiles. Whole tiles: bit-identical.
                        from schemes.memory_link_exhaustion.encodings import actenc2 as _AEy
                        g_ = _AEy.gate_dev(x.device)
                        ctr_ = torch.zeros(1, dtype=torch.int32, device=x.device)
                        y = torch.empty(M, N, dtype=x.dtype, device=x.device)
                        ext.encgemm_pre(x, w, bias, key, 2, 128, nc_, st_, 8, _num_sms(), ordinal, y, ctr_, g_, 0)
                        _AEy.load()._ae_gate_wait_kernel[(1,)](g_, num_warps=1)
                        ext.encgemm_pre(x, w, bias, key, 2, 128, nc_, st_, 8, _num_sms(), ordinal, y, ctr_, g_, 1)
                    else:
                        y = ext.encgemm(x, w, bias, key, 2, 128, nc_, st_, 8, _num_sms(), ordinal)
                return y
        bn, nc, st, ctas = cuda_cfg(M, N, K, dec)
        y = ext.encgemm(x, w, bias, key, 1 if dec else 0, bn, nc, st, 8, ctas, 0)
        _CHECK[1] += 1
        if _CHECK[3] > 0 and torch.cuda.is_current_stream_capturing():
            _CHECK[3] -= 1
            import sys as _s
            print(f"[varena-cuda-cap] M={M} N={N} K={K} dec={dec} key={key:#x} wptr={w.data_ptr():#x} "
                  f"xptr={x.data_ptr():#x} xstride={tuple(x.stride())} cfg={cuda_cfg(M, N, K, dec)}", file=_s.stderr, flush=True)
        if _CHECK[0] > 0 and _CHECK[1] > _CHECK[2] and _sealed() and not torch.cuda.is_current_stream_capturing():
            _CHECK[0] -= 1
            ref = build_matmul().varena_matmul_persistent(key, x, w.t(), bias=bias) if dec else \
                __import__("vllm.model_executor.determinism.batch_invariant", fromlist=["x"]).matmul_persistent(x, w.t(), bias=bias)
            ok = torch.equal(y, ref)
            import sys as _s
            print(f"[varena-cuda-check] M={M} N={N} K={K} dec={dec} key={key:#x} xptr%1024={x.data_ptr() % 1024} "
                  f"wptr%1024={w.data_ptr() % 1024} xstride={tuple(x.stride())} wstride={tuple(w.stride())} "
                  f"bias={bias is not None and tuple(bias.stride())} exact={ok} "
                  f"maxdiff={(y.float() - ref.float()).abs().max().item():.3g}", file=_s.stderr, flush=True)
        return y

    @cuda_linear.register_fake
    def _(x, w, bias, key, dec):
        return x.new_empty((x.shape[0], w.shape[0]))

    _CUDA_LINEAR = cuda_linear
    return cuda_linear


# --------------------------------------------------------------------------- install
_INSTALLED = False


def install_triton_gemm_overrides():
    """Baseline and inference: on Hopper/Blackwell also route aten::mm & co. through
    vLLM's Triton persistent matmul (what batch-invariant mode does on SM80)."""
    import vllm.model_executor.determinism.batch_invariant as bi
    orig = bi.enable_batch_invariant_mode
    if getattr(orig, "_varena", False):
        return

    def enable():
        already = bi._batch_invariant_MODE
        orig()
        if already:
            return
        from vllm.platforms import current_platform
        if current_platform.is_cuda() and not current_platform.is_device_capability_family(80):
            lib, key = bi._batch_invariant_LIB, current_platform.dispatch_key
            lib.impl("aten::mm", bi.mm_batch_invariant, key)
            lib.impl("aten::addmm", bi.addmm_batch_invariant, key)
            lib.impl("aten::mm.out", bi.mm_out_batch_invariant, key)
            lib.impl("aten::addmm.out", bi.addmm_out_batch_invariant, key)
            lib.impl("aten::matmul", bi.matmul_batch_invariant, key)
            lib.impl("aten::linear", bi.linear_batch_invariant, key)

    enable._varena = True
    bi.enable_batch_invariant_mode = enable
    if os.environ.get("VARENA_MM_GATE") == "1" and os.environ.get("VARENA") == "1":
        _install_mm_gate(bi)


def _install_mm_gate(bi):
    """VARENA_MM_GATE=1 (prover only): vLLM's batch-invariant persistent matmul waits at every output tile while a
    challenge is in flight (the device gate actenc2.gate_dev, raised and dropped by every challenge graph), so the
    challenge kernels run on a quiet GPU. The tile arithmetic is unchanged: outputs stay identical to stock. Needs
    VARENA_RESERVE_SMS >= 1 so the challenge kernels find free SMs while matmul CTAs wait."""
    import inspect
    if getattr(bi.matmul_kernel_persistent, "_varena_gated", False):
        return
    src = inspect.getsource(bi.matmul_kernel_persistent.fn)
    a = "    HAS_BIAS: tl.constexpr,\n):"
    assert src.count(a) == 1, "matmul_kernel_persistent signature changed"
    src = src.replace(a, "    HAS_BIAS: tl.constexpr,\n    mm_gate=None,\n    MM_GATE: tl.constexpr = False,\n):")
    b = "        pid_m, pid_n = _compute_pid(tile_id, "
    assert src.count(b) == 1, "matmul_kernel_persistent body changed"
    src = src.replace(b, "        if MM_GATE:\n            while tl.load(mm_gate, volatile=True) != 0:\n                pass\n" + b)
    head = ("import triton\nimport triton.language as tl\n"
            "from vllm.model_executor.determinism.batch_invariant import _compute_pid, _matmul_launch_metadata\n\n")
    mod = _write_and_import("mmgate", head + src)
    gated = mod.matmul_kernel_persistent
    from schemes.memory_link_exhaustion.encodings import actenc2 as _AEg

    orig_fn = bi.matmul_kernel_persistent.fn

    class _GatedKernel:
        _varena_gated = True
        fn = orig_fn            # other source rewrites (build_matmul) read the stock kernel's source through .fn

        def __getitem__(self, grid):
            f = gated[grid]

            def call(*args, **kw):
                g = _AEg.gate_dev(args[0].device)
                return f(*args, mm_gate=g, MM_GATE=True, **kw)
            return call

    bi.matmul_kernel_persistent = _GatedKernel()


def install_encoded():
    """Route layers tagged with `_varena_key` (weights) and attention layers with
    `_varena_kv_key` through the encoded kernel copies."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    mm = build_matmul()
    attn = build_attention()
    cache = build_cache_write()
    from vllm.model_executor.layers import linear as L
    from vllm.model_executor.layers import vocab_parallel_embedding as V
    from vllm.v1.attention.backends import triton_attn as TA
    from schemes.memory_link_exhaustion.gpu import kernels as VK

    import vllm.model_executor.determinism.batch_invariant as bi  # noqa: F401  (kept: import side effects)
    cuda_gemm = CUDA_GEMM == "1"
    if cuda_gemm:
        cuda_linear = build_cuda_linear()

    def enc_linear(layer, x, bias):
        w = layer.weight
        if bias is not None:
            bias = bias.expand(w.shape[0])
        if cuda_gemm and _kind(w) not in CUDA_SKIP:
            out = cuda_linear(x.reshape(-1, x.shape[-1]), w, bias, 0, True)
            return out.view(*x.shape[:-1], out.shape[-1])
        # traced by torch.compile exactly like the stock linear_batch_invariant (a user Triton kernel), so the
        # compiled graph around it is the same; an opaque custom op changed Inductor's fusions on Hopper (not bit-exact)
        out = mm.varena_matmul_persistent(layer._varena_key, x.reshape(-1, x.shape[-1]), w.t(), bias=bias)
        return out.view(*x.shape[:-1], out.shape[-1])

    lin_apply = L.UnquantizedLinearMethod.apply

    def lin_apply_p(self, layer, x, bias=None):
        if getattr(layer, "_varena_key", None) is None:
            return lin_apply(self, layer, x, bias)
        return enc_linear(layer, x, bias)

    L.UnquantizedLinearMethod.apply = lin_apply_p

    emb_apply = V.UnquantizedEmbeddingMethod.apply
    emb_embed = V.UnquantizedEmbeddingMethod.embedding

    def emb_apply_p(self, layer, x, bias=None):
        if getattr(layer, "_varena_key", None) is None:
            return emb_apply(self, layer, x, bias)
        return enc_linear(layer, x, bias)

    def emb_embed_p(self, layer, ids):
        k = getattr(layer, "_varena_key", None)
        if k is None:
            return emb_embed(self, layer, ids)
        out = VK.embed(ids, layer.weight, k)
        return out.view(*ids.shape, out.shape[-1])

    V.UnquantizedEmbeddingMethod.apply = emb_apply_p
    V.UnquantizedEmbeddingMethod.embedding = emb_embed_p

    fwd = TA.TritonAttentionImpl.forward
    upd = TA.TritonAttentionImpl.do_kv_cache_update
    stock_attn = TA.unified_attention
    stock_cache = TA.triton_reshape_and_cache_flash

    def fwd_p(self, layer, *a, **kw):
        k = getattr(layer, "_varena_kv_key", None)
        if k is None:
            return fwd(self, layer, *a, **kw)
        attn._VARENA_KEY[0] = k
        TA.unified_attention = attn.unified_attention
        try:
            return fwd(self, layer, *a, **kw)
        finally:
            TA.unified_attention = stock_attn
            attn._VARENA_KEY[0] = None

    def upd_p(self, layer, *a, **kw):
        k = getattr(layer, "_varena_kv_key", None)
        if k is None:
            return upd(self, layer, *a, **kw)
        cache._VARENA_KEY[0] = k
        TA.triton_reshape_and_cache_flash = cache.triton_reshape_and_cache_flash
        try:
            return upd(self, layer, *a, **kw)
        finally:
            TA.triton_reshape_and_cache_flash = stock_cache
            cache._VARENA_KEY[0] = None

    TA.TritonAttentionImpl.forward = fwd_p
    TA.TritonAttentionImpl.do_kv_cache_update = upd_p
