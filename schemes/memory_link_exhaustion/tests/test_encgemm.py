"""GEMM-level test of the custom Hopper decode-on-load kernel (schemes/memory_link_exhaustion/gpu/cuda/encgemm.cu).

For each Qwen2.5-7B linear shape and each M, compares against vLLM's batch-invariant Triton
matmul_persistent on the decoded weight (the stock deterministic baseline):
  plain   custom kernel, plaintext W (no decode)
  dec     custom kernel, encoded W, XOR decode in shared memory
for every kernel config in --cfgs (bn:nc:stages), and times stock / plain / dec.
Weights are rotated over enough copies to defeat L2 (cold-ish weight reads like inference).

  python schemes/memory_link_exhaustion/tests/test_encgemm.py --out encgemm.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
import numpy as np
import torch

SHAPES = {"qkv": (4608, 3584, True), "o": (3584, 3584, False), "gate_up": (37888, 3584, False),
          "down": (3584, 18944, False)}


def load_ext():
    from schemes.memory_link_exhaustion.gpu.cuda import build
    return build.load()


def bench(fn, iters):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="encgemm.json")
    ap.add_argument("--ms", default="1,2,4,8,16,32,64,96,128,192,256,512,1024,2048,4096,8192")
    ap.add_argument("--cfgs", default="128:2:3:sms,128:2:4:sms",
                    help="bn:nc:stages:ctas (ctas: sms = persistent NUM_SMS grid like stock, 0 = one CTA per tile)")
    ap.add_argument("--shapes", default="qkv,o,gate_up,down")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--group-m", type=int, default=8)
    a = ap.parse_args()
    import vllm.model_executor.determinism.batch_invariant as bi
    from schemes.memory_link_exhaustion.gpu import kernels as K
    bi.enable_batch_invariant_mode()
    ext = load_ext()
    key = 0x1234ABCD
    import vllm.model_executor.determinism.batch_invariant as _b
    sms = int(_b.num_compute_units(0))
    cfgs = []
    for c in a.cfgs.split(","):
        v = c.split(":")
        cfgs.append((int(v[0]), int(v[1]), int(v[2]), sms if (len(v) > 3 and v[3] == "sms") else int(v[3]) if len(v) > 3 else 0))
    res = {"gpu": torch.cuda.get_device_name(0), "rows": []}
    torch.manual_seed(0)
    for name in a.shapes.split(","):
        N, Kd, has_bias = SHAPES[name]
        ncopy = max(1, int(np.ceil(160e6 / (N * Kd * 2))))
        ws = [(torch.randn(N, Kd, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(ncopy)]
        wes = []
        for w in ws:
            we = torch.empty_like(w)
            K.xor16_w(w, we, key, Kd)
            wes.append(we)
        bias = (torch.randn(N, device="cuda") * 0.1).to(torch.bfloat16) if has_bias else None
        for M in [int(m) for m in a.ms.split(",")]:
            x = torch.randn(M, Kd, device="cuda").to(torch.bfloat16)
            ref = bi.matmul_persistent(x, ws[0].t(), bias=bias)
            row = {"shape": name, "M": M, "N": N, "K": Kd}
            it = [0]

            def stock():
                it[0] = (it[0] + 1) % ncopy
                bi.matmul_persistent(x, ws[it[0]].t(), bias=bias)
            row["stock_us"] = bench(stock, a.iters)
            for (bn, nc, st, ctas) in cfgs:
                tag = f"{bn}x{nc}x{st}" + ("p" if ctas else "")
                yp = ext.encgemm(x, ws[0], bias, 0, 0, bn, nc, st, a.group_m, ctas, 0)
                yd = ext.encgemm(x, wes[0], bias, key, 1, bn, nc, st, a.group_m, ctas, 0)
                torch.cuda.synchronize()
                row[f"{tag}_plain_exact"] = bool(torch.equal(yp, ref))
                row[f"{tag}_dec_exact"] = bool(torch.equal(yd, ref))
                if not row[f"{tag}_plain_exact"]:
                    d = (yp.float() - ref.float()).abs()
                    row[f"{tag}_plain_ndiff"] = int((yp != ref).sum())
                    row[f"{tag}_plain_maxdiff"] = float(d.max())

                def plain():
                    it[0] = (it[0] + 1) % ncopy
                    ext.encgemm(x, ws[it[0]], bias, 0, 0, bn, nc, st, a.group_m, ctas, 0)

                def dec():
                    it[0] = (it[0] + 1) % ncopy
                    ext.encgemm(x, wes[it[0]], bias, key, 1, bn, nc, st, a.group_m, ctas, 0)
                row[f"{tag}_plain_us"] = bench(plain, a.iters)
                row[f"{tag}_dec_us"] = bench(dec, a.iters)
            res["rows"].append(row)
            print(json.dumps(row), flush=True)
        del ws, wes
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)
    inexact = [(r["shape"], r["M"], k) for r in res["rows"] for k, v in r.items() if k.endswith("_exact") and not v]
    print("ALL EXACT" if not inexact else f"FAIL: {len(inexact)} inexact, first {inexact[:5]}", flush=True)
    sys.exit(1 if inexact else 0)


if __name__ == "__main__":
    main()
