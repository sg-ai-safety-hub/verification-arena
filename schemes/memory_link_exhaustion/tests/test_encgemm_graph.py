"""Custom op under CUDA graph capture + replay (decode and plain), vs the Triton reference."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
import torch
import vllm.model_executor.determinism.batch_invariant as bi
from schemes.memory_link_exhaustion.gpu import kernels as K
from schemes.memory_link_exhaustion.inference import patches as P
bi.enable_batch_invariant_mode()
cl = P.build_cuda_linear()
mm = P.build_matmul()
bad = 0
for (N, Kd, hb) in [(4608, 3584, True), (3584, 18944, False)]:
    key = 0x9D59C1EA
    w = (torch.randn(N, Kd, device="cuda") * 0.02).to(torch.bfloat16)
    we = torch.empty_like(w); K.xor16_w(w, we, key, Kd)
    P.W_KEYS[we.data_ptr()] = key
    b = (torch.randn(N, device="cuda") * 0.1).to(torch.bfloat16) if hb else None
    for M in (1, 8, 64, 256):
        x = torch.randn(M, Kd, device="cuda").to(torch.bfloat16)
        for dec in (True, False):
            ww = we if dec else w
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                cl(x, ww, b, 0, dec)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y = cl(x, ww, b, 0, dec)
            for it in range(3):
                x.copy_(torch.randn(M, Kd, device="cuda").to(torch.bfloat16))
                g.replay(); torch.cuda.synchronize()
                ref = bi.matmul_persistent(x, w.t(), bias=b)
                if not torch.equal(y, ref):
                    bad += 1
                    print("MISMATCH", N, Kd, M, dec, it, (y.float() - ref.float()).abs().max().item(), flush=True)
print("graph test done, bad =", bad)
sys.exit(1 if bad else 0)
