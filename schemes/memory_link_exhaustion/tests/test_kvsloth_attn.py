"""Sloth KV (experiment) end to end at the vLLM kernel level (GPU, VARENA_KVENC=sloth), same checks as the butterfly
test: cache write (patched encoding write == reference encode of the stock write) and attention (patched copy on
the encoded cache == stock on the plaintext cache, bit for bit). Adapted from schemes/memory_link_exhaustion/tests/test_kvbf_attn.py:
  1. cache write: stock write into a plaintext cache vs the patched (encoding) write into its encoded copy;
     XOR-decoding the encoded cache must give the plaintext cache everywhere
  2. attention: stock unified_attention on the plaintext cache vs the patched copy on the encoded cache,
     bit for bit, decode shape (3D segments, TILE 16) and prefill shape (2D, TILE 32)
"""
import os
import sys

os.environ["VARENA_KVENC"] = "sloth"
os.environ.setdefault("VARENA_ATT_BLOCK_M_PREFILL", "32")     # the reference configuration's prefill block
os.environ.setdefault("VARENA_WDEC_SLICE_MB", "270")
os.environ.setdefault("VARENA_ATT_CLAMP", "0")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import torch

from schemes.memory_link_exhaustion.encodings import sloth64 as S
from schemes.memory_link_exhaustion.inference import patches as Pt

import vllm.v1.attention.ops.triton_reshape_and_cache_flash as rc
import vllm.v1.attention.ops.triton_unified_attention as ua

enc_attn = Pt.build_attention()
from schemes.memory_link_exhaustion.encodings import actenc2 as _AE
_AE.scratch("wdec_s", ((270 << 20) // 2,), torch.empty(1, dtype=torch.bfloat16, device="cuda"))   # the weight decode's scratch
enc_cache = Pt.build_cache_write()
assert enc_attn._VARENA_KVSL and enc_cache._VARENA_KVSL, "generated kernels without KVSL"
torch.manual_seed(0)
dev = "cuda"
NB, BS, H, HKV, D = 512, 32, 28, 4, 128
KEY = 0x2468ACE1
ctx = [1, 17, 31, 32, 33, 100, 517, 1500]          # sequence lengths (tokens in the cache)
B = len(ctx)
# blocks: random distinct physical blocks per sequence
perm = torch.randperm(NB).tolist()
tables, used = [], 0
maxb = max((n + BS - 1) // BS for n in ctx)
for n in ctx:
    nb = (n + BS - 1) // BS
    tables.append(perm[used:used + nb] + [0] * (maxb - nb))
    used += nb
block_table = torch.tensor(tables, dtype=torch.int32, device=dev)


def views(phys):
    """phys [NB, BS, HKV, 2D] (the prover's physical layout) -> key/value cache views [NB, BS, HKV, D]."""
    return phys[..., :D], phys[..., D:]


# random initial content (incl. unwritten slots); encoded copy = plain XOR page keystream
plain = (torch.randn(NB, BS, HKV, 2 * D, device=dev) * 0.5).to(torch.bfloat16)
enc = torch.empty_like(plain)
S.encode_to(plain.view(torch.int64), enc.view(torch.int64), S.kv_seed(KEY))
# 1. write every token of every sequence (as one prefill would)
slots = torch.tensor([tables[s][p // BS] * BS + p % BS for s, n in enumerate(ctx) for p in range(n)],
                     dtype=torch.int64, device=dev)
T = slots.numel()
k_new = (torch.randn(T, HKV, D, device=dev) * 0.5).to(torch.bfloat16)
v_new = (torch.randn(T, HKV, D, device=dev) * 0.5).to(torch.bfloat16)
one = torch.ones(1, dtype=torch.float32, device=dev)
kc, vc = views(plain)
rc.triton_reshape_and_cache_flash(k_new, v_new, kc, vc, slots, "auto", one, one)
kc_e, vc_e = views(enc)
enc_cache._VARENA_KEY[0] = KEY
enc_cache.triton_reshape_and_cache_flash(k_new, v_new, kc_e, vc_e, slots, "auto", one, one)
enc_cache._VARENA_KEY[0] = None
torch.cuda.synchronize()
dec = torch.empty_like(enc)
S.decode_to(enc.view(torch.int64), dec.view(torch.int64), S.kv_seed(KEY))
ok_w = torch.equal(dec.view(torch.int16), plain.view(torch.int16))
nbad = int((dec.view(torch.int16) != plain.view(torch.int16)).sum())
print(f"cache write round trip: {'exact' if ok_w else f'{nbad} elements differ'}")


def run(mod, kc, vc, qlen, key):
    cu = [0]
    for n in ctx:
        cu.append(cu[-1] + min(qlen, n))
    Tq = cu[-1]
    q = (torch.randn(Tq, H, D, generator=torch.Generator(device=dev).manual_seed(1), device=dev) * 0.5).to(torch.bfloat16)
    out = torch.empty_like(q)
    thr, seg = 128, 16
    so = torch.empty(thr, H, seg, D, dtype=torch.float32, device=dev)
    sm = torch.empty(thr, H, seg, dtype=torch.float32, device=dev)
    se = torch.empty(thr, H, seg, dtype=torch.float32, device=dev)
    desc = one.expand(B, HKV)
    if hasattr(mod, "_VARENA_KEY"):
        mod._VARENA_KEY[0] = key
    mod.unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device=dev),
                          max_seqlen_q=max(min(qlen, n) for n in ctx),
                          seqused_k=torch.tensor(ctx, dtype=torch.int32, device=dev), max_seqlen_k=max(ctx),
                          softmax_scale=D ** -0.5, causal=True, window_size=(-1, -1), block_table=block_table,
                          softcap=0, q_descale=None, k_descale=desc, v_descale=desc, seq_threshold_3D=thr,
                          num_par_softmax_segments=seg, softmax_segm_output=so, softmax_segm_max=sm,
                          softmax_segm_expsum=se)
    if hasattr(mod, "_VARENA_KEY"):
        mod._VARENA_KEY[0] = None
    torch.cuda.synchronize()
    return out


ok_a = True
for name, qlen in (("decode (3D)", 1), ("prefill (2D)", 64)):
    ref = run(ua, kc, vc, qlen, None)
    got = run(enc_attn, kc_e, vc_e, qlen, KEY)
    fm = run(enc_attn, kc, vc, qlen, None)          # patched copy, encoding off, plaintext cache
    same = torch.equal(ref.view(torch.int16), got.view(torch.int16))
    same_off = torch.equal(ref.view(torch.int16), fm.view(torch.int16))
    diff = (ref.float() - got.float()).abs().max().item()
    ok_a &= same
    print(f"attention {name}: patched+encoded == stock: {same} (max |diff| {diff:.3g}); "
          f"patched with encoding off == stock: {same_off}")
print("ALL OK" if ok_w and ok_a else "MISMATCH")
sys.exit(0 if ok_w and ok_a else 1)
