"""Bare GPU check for max_shard: how fast is a stage unthrottled (and a rough standalone fit)?

NOTE: --fit under a fractional cap over-estimates what fits in the cheater (the 91253c6 GPU run: [24576] passed
here and OOMed in the cheater). The authoritative sizing is `ffns: "auto"` in the cheater spec: build_fitted() runs
inside the cheater after sealing, before ready, against its real free memory minus margin_mib, with a probe
micro-batch per candidate.

    GSHARD_NO_GATE=1 python -m schemes.memory_link_exhaustion.attacks.max_shard.bench --cap-gib 1.89 --fit
    GSHARD_NO_GATE=1 python -m schemes.memory_link_exhaustion.attacks.max_shard.bench --ffns 27648,9216 --link 100

--fit bisects the second layer's FFN (multiples of 1,024) for the largest stage whose peak allocation stays under
--cap-gib (allocator capped with set_per_process_memory_fraction, as the rt1 fit test). Each probe runs two
micro-batches in a child process, so an OOM never poisons the next probe. Prints one JSON line per probe and a
final FIT / BENCH line with tok_s and param_tok_s (= params x tokens/s; FLOPs = 6x).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time


def _probe(ffns, cap_gib, link, n, q):
    import torch
    from schemes.memory_link_exhaustion.attacks.max_shard.workload import MaxShard
    try:
        tot = torch.cuda.get_device_properties(0).total_memory
        if cap_gib:
            torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gib * 2**30 / tot))
        s = MaxShard({"ffns": ffns, "link_gbs": link, "burst_s": 0, "global_tokens": 10 ** 12})
        s.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tok = sum(s.step() for _ in range(n))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        d = s.describe()
        s.close()
        q.put({"ok": True, "ffns": ffns, "params": d["params"], "tok_s": round(tok / dt, 1),
               "param_tok_s": float(f"{d['params'] * tok / dt:.4g}"), "peak_alloc_gib": d["peak_alloc_gib"],
               "grad_out_gbs": round(d["grad_out_gb"] / dt, 2)})
    except torch.OutOfMemoryError as e:
        q.put({"ok": False, "ffns": ffns, "err": "OOM", "msg": str(e)[:200]})
    except Exception as e:
        q.put({"ok": False, "ffns": ffns, "err": repr(e)[:300]})


def run(ffns, cap, link, n):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_probe, args=(ffns, cap, link, n, q))
    p.start()
    p.join()
    r = q.get() if not q.empty() else {"ok": False, "ffns": ffns, "err": f"exit {p.exitcode}"}
    print(json.dumps(r), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffns", default="27648,9216")
    ap.add_argument("--cap-gib", type=float, default=0.0)
    ap.add_argument("--link", type=float, default=100.0, help="GB/s cap for the emulated stage traffic")
    ap.add_argument("--n", type=int, default=4, help="timed micro-batches")
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    ffns = [int(x) for x in a.ffns.split(",")]
    if not a.fit:
        r = run(ffns, a.cap_gib, a.link, a.n)
        print("BENCH", json.dumps(r))
        return
    lo, hi, best = 0, 27648 // 1024, None
    while lo <= hi:
        mid = (lo + hi) // 2
        f2 = mid * 1024
        cand = [27648] + ([f2] if f2 else [])
        r = run(cand, a.cap_gib, a.link, 2)
        if r["ok"]:
            best, lo = r, mid + 1
        else:
            hi = mid - 1
    if best is None:
        # one full layer does not fit: bisect the single layer's FFN instead
        lo, hi = 1, 27648 // 1024 - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            r = run([mid * 1024], a.cap_gib, a.link, 2)
            if r["ok"]:
                best, lo = r, mid + 1
            else:
                hi = mid - 1
    print("FIT", json.dumps(best))


if __name__ == "__main__":
    main()
