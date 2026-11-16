"""Decode kernel under concurrent load (other streams busy), eager and graph replay, many repeats."""
import os, sys, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
import torch
import vllm.model_executor.determinism.batch_invariant as bi
from schemes.memory_link_exhaustion.gpu import kernels as K
from schemes.memory_link_exhaustion.inference import patches as P
bi.enable_batch_invariant_mode()
cl = P.build_cuda_linear()
stop = threading.Event()
def burn():
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    h = torch.empty(256 << 20, dtype=torch.uint8).pin_memory(); d = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    s = torch.cuda.Stream(priority=-1)
    with torch.cuda.stream(s):
        while not stop.is_set():
            for _ in range(4): a @ a
            d.copy_(h, non_blocking=True)
            s.synchronize()
th = threading.Thread(target=burn, daemon=True)
th.start()
bad = tot = 0
for (N, Kd, hb) in [(4608, 3584, True), (37888, 3584, False), (3584, 18944, False)]:
    key = 0x9D59C1EA
    w = (torch.randn(N, Kd, device="cuda") * 0.02).to(torch.bfloat16)
    we = torch.empty_like(w); K.xor16_w(w, we, key, Kd)
    P.W_KEYS[we.data_ptr()] = key
    b = (torch.randn(N, device="cuda") * 0.1).to(torch.bfloat16) if hb else None
    for M in (1, 3, 17, 64, 200, 256, 512):
        x = torch.randn(M, Kd, device="cuda").to(torch.bfloat16)
        ref = bi.matmul_persistent(x, w.t(), bias=b)
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            cl(x, we, b, 0, True)
        torch.cuda.synchronize()
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            y = cl(x, we, b, 0, True)
        for it in range(30):
            g.replay()
            ye = cl(x, we, b, 0, True)
            torch.cuda.synchronize()
            tot += 2
            for nm, t in (("graph", y), ("eager", ye)):
                if not torch.equal(t, ref):
                    bad += 1
                    nbad = int((t != ref).sum())
                    if bad < 15:
                        print("MISMATCH", nm, N, Kd, M, it, nbad, flush=True)
stop.set()
th.join(timeout=60)          # the load thread must leave CUDA before the interpreter tears down
print("race test done, bad =", bad, "of", tot, flush=True)
sys.exit(1 if bad or th.is_alive() else 0)
