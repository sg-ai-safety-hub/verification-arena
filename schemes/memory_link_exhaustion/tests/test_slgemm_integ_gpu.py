"""Fused sloth64 decode-on-load GEMM in the inference path (VARENA_WFUSE_GU_M): gate_up through varena::cuda_linear.
1. bit-exact: cuda_linear on a really sloth-encoded gate_up W == decode-ahead (ref.w_xor) + stock matmul_persistent,
   for M below, at and above the tile sizes, and falls back to the old path above VARENA_WFUSE_GU_M;
2. challenge pause: with the device gate raised the kernel must not finish; after it drops, same output;
3. speed: fused vs the current sliced decode-ahead path, per M.

  VLLM_BATCH_INVARIANT=1 python schemes/memory_link_exhaustion/tests/test_slgemm_integ_gpu.py
"""
import os
import sys
import time

os.environ.setdefault("VARENA_WENC", "sloth")
os.environ.setdefault("VARENA_BFLY_AHEAD_M", "1")
os.environ.setdefault("VARENA_WDEC_SLICE_MB", "270")
os.environ["VARENA_WFUSE_GU_M"] = "256"
os.environ.setdefault("VARENA_WFUSE_GATE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import torch

import vllm.model_executor.determinism.batch_invariant as bi
from schemes.memory_link_exhaustion.encodings import actenc2 as A
from schemes.memory_link_exhaustion.encodings import ref
from schemes.memory_link_exhaustion.gpu.cuda import build as CB
from schemes.memory_link_exhaustion.inference import patches as P

assert ref.WENC == "sloth" and P.WFUSE_GU_M == 256
torch.manual_seed(0)
dev = "cuda"
ok = True
N, K, key, od = 2 * 27648, 5120, 0x5A17, 9
w = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
we = w.clone()
ref.w_encode(we, we, key, od)
P.W_KEYS[int(we.data_ptr())] = (key, od)
wd = torch.empty_like(we)
ref.w_xor(we, wd, key, od)
torch.cuda.synchronize()
print("encode/decode round trip:", torch.equal(wd.view(torch.int16), w.view(torch.int16)), flush=True)
lin = P.build_cuda_linear()
gate = A.gate_dev(dev)


def same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


for M in (1, 7, 64, 128, 129, 200, 256, 257, 600):
    x = torch.randn(M, K, device=dev).bfloat16()
    y_ref = bi.matmul_persistent(x, wd.t())
    try:
        y = lin(x, we, None, 0, True)
        torch.cuda.synchronize()
    except Exception as ex:  # the old path needs more of the arena set up than this test does
        print(f"M={M}: cuda_linear raised {type(ex).__name__}: {str(ex)[:200]}", flush=True)
        ok &= M > P.WFUSE_GU_M
        continue
    e = same(y, y_ref)
    ok &= e
    print(f"M={M}: cuda_linear ({'fused' if M <= P.WFUSE_GU_M else 'decode-ahead'}) == decode-ahead + stock: {e}",
          flush=True)

# ---- challenge pause
ext = CB.load_sl()
seed = ref.sloth_seed(key, od)
seed = seed - (1 << 64) if seed >= (1 << 63) else seed
sms = torch.cuda.get_device_properties(0).multi_processor_count
side = torch.cuda.Stream()
CFGS = ((128, (64, 2, 1, 6, 3)), (256, (64, 2, 2, 4, 2)))


def ms(e0, e1):
    return e0.elapsed_time(e1)


# a challenge-like load for the side stream: a kernel that wants every SM (stock matmul on the whole GPU)
xs = torch.randn(4096, K, device=dev).bfloat16()
ws = (torch.randn(K, K, device=dev) * 0.02).bfloat16()


def side_time():
    with torch.cuda.stream(side):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(4):
            torch.mm(xs, ws)
        e1.record()
        e1.synchronize()
    return ms(e0, e1) / 4


torch.mm(xs, ws)
torch.cuda.synchronize()
t_alone = min(side_time() for _ in range(5))
for (M, cfg) in CFGS:
    x = torch.randn(M, K, device=dev).bfloat16()
    y_ref = bi.matmul_persistent(x, wd.t())
    for pre in (-1, 0, 1):
        torch.cuda.synchronize()
        gate.fill_(1)
        torch.cuda.synchronize()
        y = ext.slgemm(x, we, seed, 0, 1, *cfg, 8, sms - 2, gate.data_ptr(), pre)
        ev = torch.cuda.Event()
        ev.record()
        time.sleep(0.2)
        held = not ev.query()
        # the other stream's speed while the fused GEMM is paused. Not measured for the in-place wait (-1): its CTAs
        # keep every SM, and on an H200 the side load then never got scheduled at all (the test hung here)
        t_side = min(side_time() for _ in range(3)) if pre >= 0 else float('nan')
        with torch.cuda.stream(side):
            gate.zero_()
        torch.cuda.synchronize()
        e = same(y, y_ref)
        ok &= held and e
        if pre >= 0:
            ok &= t_side < 1.3 * t_alone
        print(f"pause M={M} preempt={pre}: held while gate raised: {held}, output after release identical: {e}, "
              f"side-stream load {t_side:.3f} ms while paused (alone {t_alone:.3f} ms)", flush=True)
    # gate raised at a random point inside the kernel, several times
    bad = 0
    for i in range(40):
        y = ext.slgemm(x, we, seed, 0, 1, *cfg, 8, sms - 2, gate.data_ptr(), i % 2)
        time.sleep(0.0001 * (i % 12))
        with torch.cuda.stream(side):
            gate.fill_(1)
        time.sleep(0.002)
        with torch.cuda.stream(side):
            gate.zero_()
        torch.cuda.synchronize()
        bad += not same(y, y_ref)
    ok &= bad == 0
    print(f"mid-kernel gate M={M}: {40 - bad}/40 identical", flush=True)


def t(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(3):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(reps):
            fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / reps)
    return best


for M in (16, 64, 128, 192, 256):
    x = torch.randn(M, K, device=dev).bfloat16()
    P.WFUSE_GU_M = 256
    t_f = t(lambda: lin(x, we, None, 0, True))
    P.WFUSE_GU_M = 0
    try:
        t_a = t(lambda: lin(x, we, None, 0, True))
    except Exception as ex:
        print(f"current path raised {type(ex).__name__}: {str(ex)[:200]}; using decode + stock", flush=True)
        t_a = t(lambda: (ref.w_xor(we, wd, key, od), bi.matmul_persistent(x, wd.t())))
    t_s = t(lambda: bi.matmul_persistent(x, wd.t()))
    print(f"gate_up M={M}: stock {t_s:.3f} ms, current path {t_a:.3f} ms, fused {t_f:.3f} ms "
          f"({(1 - t_f / t_a) * 100:+.1f}%)", flush=True)
print("ALL OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
