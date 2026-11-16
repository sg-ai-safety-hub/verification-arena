"""Same-box bare references for the 32B-slice shard cheaters (unpaced, no challenges)."""
import json, sys, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
import torch
from baselines.pretrain.pretrain_qwen import PretrainQwen
A = {"d": 5120, "heads": 40, "kv": 8, "hd": 128}
out = {}
for name, ffn, buds in (("shard32_fulllayer_ffn27648", 27648, (int(80e9),)),
                        ("shard32_slice_ffn864", 864, (int(2.6e9), int(8e9))),
                        ("shard32_quarter_ffn3840", 3840, (int(2.6e9), int(8e9)))):
    for bud in buds:
        torch.cuda.empty_cache()
        spec = {"layers": 1, "mb": 1, "seq": 2048, "hop_mib": 40, "global_tokens": 262144, "headroom_gib": 0, "arch": dict(A, ffn=ffn)}
        try:
            wl = PretrainQwen(dict(spec), budget=bud)
            for _ in range(2):
                wl.step()
            torch.cuda.synchronize()
            t0, u = time.perf_counter(), 0
            while time.perf_counter() - t0 < 30:
                u += wl.step()
            k = f"{name}_budget{bud / 1e9:.1f}G"
            ts = u / (time.perf_counter() - t0)
            out[k] = {"tok_s": ts, "params": wl.P, "param_tok_s": ts * wl.P, "mode": wl.mode}
            print(k, json.dumps(out[k]), flush=True)
            del wl
        except Exception as e:
            print(name, bud, "failed", e, flush=True)
# tuned whole-GPU baseline: stacks of full 32B layers, everything resident, no PCIe hop (budget None = normal mode)
for L, mb in ((1, 1), (1, 2), (4, 1), (8, 1), (8, 2), (12, 1), (14, 1)):
    torch.cuda.empty_cache()
    spec = {"layers": L, "mb": mb, "seq": 2048, "global_tokens": 65536, "arch": dict(A, ffn=27648)}
    try:
        wl = PretrainQwen(dict(spec), budget=None)
        for _ in range(2):
            wl.step()
        torch.cuda.synchronize()
        t0, u = time.perf_counter(), 0
        while time.perf_counter() - t0 < 30:
            u += wl.step()
        ts = u / (time.perf_counter() - t0)
        k = f"stack32_L{L}_mb{mb}"
        out[k] = {"tok_s": ts, "params": wl.P, "param_tok_s": ts * wl.P, "mode": wl.mode,
                  "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        print(k, json.dumps(out[k]), flush=True)
        del wl
    except Exception as e:
        print("stack", L, mb, "failed", str(e)[:200], flush=True)
        try:
            del wl
        except Exception:
            pass
    torch.cuda.reset_peak_memory_stats()
json.dump(out, open(sys.argv[1], "w"), indent=1)
