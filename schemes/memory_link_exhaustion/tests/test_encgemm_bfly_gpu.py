"""Fused butterfly decode GEMM (encgemm mode 2, the VARENA_BFLY_AHEAD_M path) at Qwen2.5-32B shapes and decode-step M,
against the decode-ahead path (bfly_xor_to into scratch + vLLM's batch-invariant matmul_persistent): bit for bit,
plus kernel times (16 Nov, link-noise investigation).

  VARENA_WENC=bfly python -m schemes.memory_link_exhaustion.tests.test_encgemm_bfly_gpu
"""
import os
import sys

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VARENA_WENC", "bfly")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import torch

import vllm.model_executor.determinism.batch_invariant as bi
from schemes.memory_link_exhaustion.gpu.cuda import build as CB
from schemes.memory_link_exhaustion.inference import patches as Pt

bi.enable_batch_invariant_mode()
ext = CB.load()
SHAPES = {"qkv": (7168, 5120), "o": (5120, 5120), "gate_up": (55296, 5120), "down": (5120, 27648)}
KEY, ORD = 0x2468ACE1, 77
ok = True
e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)


def t(fn, n=20):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0.record(); fn(); e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    return sorted(ts)[n // 2]


for name, (N, K) in SHAPES.items():
    w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    we = torch.empty_like(w)
    ext.bfly_xor_to(w, we, KEY, ORD)
    wd = torch.empty_like(w)
    for M in (1, 8, 32, 64, 128, 160, 256):
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        def ahead():
            ext.bfly_xor_to(we, wd, KEY, ORD)
            return bi.matmul_persistent(x, wd.t())
        def fused():
            return ext.encgemm(x, we, None, KEY, 2, 128, 2, Pt.CUDA_STAGES_BFLY, 8, Pt._num_sms(), ORD)
        def fused2():
            return ext.encgemm(x, we, None, KEY, 2, 128, 4, 4, 8, Pt._num_sms(), ORD)
        ya, yf = ahead(), fused()
        same = torch.equal(ya, yf)
        ok &= same
        extra = ""
        if M > 64:
            s2 = torch.equal(ya, fused2())
            ok &= s2
            extra = f"; 256-row CTA (nc 4): == {s2}, {t(fused2):.3f} ms"
        # pre-emptible schedule (VARENA_FUSE_YIELD): gate down (all tiles in the first launch) and gate raised (the first
        # launch stops at once, the no-yield resume does every tile)
        nc_, st_ = (4, 4) if 128 < M <= 256 else (2, Pt.CUDA_STAGES_BFLY)
        yref = ext.encgemm(x, we, None, KEY, 2, 128, nc_, st_, 8, Pt._num_sms(), ORD)
        for raised in (0, 1):
            gt = torch.tensor([raised, 0], dtype=torch.int32, device="cuda")
            ctr = torch.zeros(1, dtype=torch.int32, device="cuda")
            yp = torch.empty_like(ya)
            ext.encgemm_pre(x, we, None, KEY, 2, 128, nc_, st_, 8, Pt._num_sms(), ORD, yp, ctr, gt, 0)
            torch.cuda.synchronize()
            first = int(ctr.item())
            gt[0] = 0
            ext.encgemm_pre(x, we, None, KEY, 2, 128, nc_, st_, 8, Pt._num_sms(), ORD, yp, ctr, gt, 1)
            torch.cuda.synchronize()
            sy = torch.equal(yp.view(torch.int16), yref.view(torch.int16))
            ok &= sy
            extra += f"; yield(gate {'up' if raised else 'down'}, first {first}) == {sy}"
        print(f"{name:8s} M={M:4d}: fused == decode-ahead+stock: {same}; ahead {t(ahead):.3f} ms, fused {t(fused):.3f} ms{extra}",
              flush=True)
print("ALL OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
