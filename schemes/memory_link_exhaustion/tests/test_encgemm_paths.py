"""Exactness of the integrated op paths (fused decode, scratch decode, plain), large keys, lm_head shape."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
import torch
import vllm.model_executor.determinism.batch_invariant as bi
from schemes.memory_link_exhaustion.gpu import kernels as K
from schemes.memory_link_exhaustion.inference import patches as P
bi.enable_batch_invariant_mode()
cl = P.build_cuda_linear()
bad = 0
for (N, Kd, hb) in [(4608, 3584, True), (152064, 3584, False), (3584, 18944, False)]:
    for key in (0x1234ABCD, 0xDEADBEEF, 0xFFFFFFFF, 7):
        w = (torch.randn(N, Kd, device="cuda") * 0.02).to(torch.bfloat16)
        we = torch.empty_like(w); K.xor16_w(w, we, key, Kd)
        b = (torch.randn(N, device="cuda") * 0.1).to(torch.bfloat16) if hb else None
        for M in (1, 37, 256, 300, 511, 512, 777, 2048):
            x = torch.randn(M, Kd, device="cuda").to(torch.bfloat16)
            ref = bi.matmul_persistent(x, w.t(), bias=b)
            P.W_KEYS[we.data_ptr()] = key
            yd = cl(x, we, b, 0, True)
            yp = cl(x, w, b, 0, False)
            ok = torch.equal(yd, ref) and torch.equal(yp, ref)
            if not ok:
                bad += 1
                print("MISMATCH", N, Kd, hex(key), M, torch.equal(yd, ref), torch.equal(yp, ref), flush=True)
        del w, we
print("done, bad =", bad)
sys.exit(1 if bad else 0)
