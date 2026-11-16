"""Sliced decode-ahead (VARENA_WDEC_SLICE_MB) is bit-identical to full decode-ahead + vLLM's batch-invariant
matmul: lm_head-sized W (152064 x 5120), M in {1, 7, 256, 3000}, sloth64 W (the only W encoding in the final configuration).

  VLLM_BATCH_INVARIANT=1 python -m schemes.memory_link_exhaustion.tests.test_wdec_slice_gpu
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ.setdefault("VARENA_W_BFLY_L", "8")
import torch
import vllm.model_executor.determinism.batch_invariant as bi

from schemes.memory_link_exhaustion.encodings import actenc2 as A
from schemes.memory_link_exhaustion.encodings import ref

N, K = 152064, 5120
key, ordn = 0x2468ACE, 11
W = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
ok = True
for wenc in ("sloth", "bfly"):
    ref.WENC = wenc
    We = torch.empty_like(W)
    ref.w_encode(W, We, key, ordn)
    for M in (1, 7, 256, 3000):
        x = torch.randn(M, K, device="cuda").bfloat16()
        Wd = torch.empty_like(W)
        ref.w_xor(We, Wd, key, ordn)
        yfull = bi.matmul_persistent(x, Wd.t())
        S = 6400
        y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        torch.cuda.synchronize()
        t = time.perf_counter()
        for n0 in range(0, N, S):
            n1 = min(N, n0 + S)
            wd = A.scratch("wdec_s", (n1 - n0, K), We)
            ref.w_xor(We[n0:n1], wd, key, ordn, row0=n0)
            A.ae_matmul(x, wd.t(), False, c=y[:, n0:n1], tiled=False, cfg_n=N)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) * 1e3
        eq = torch.equal(y.view(torch.int16), yfull.view(torch.int16))
        print(f"{wenc} M={M}: sliced == full: {eq}  ({dt:.1f} ms sliced)", flush=True)
        ok &= eq
        del Wd
print("ALL OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
