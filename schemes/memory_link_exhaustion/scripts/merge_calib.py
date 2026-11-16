"""Merge several calibration files (honest runs) into one: latency p50 / p99 / max and deadlines take
the max over runs, tok_s the min, outside / allowance / measured pad the max. A threshold from one run is exceeded
by about half of later honest runs; the max over N runs is the honest envelope.

  python schemes/memory_link_exhaustion/scripts/merge_calib.py out.json calib_1.json calib_2.json ...
"""
import json
import sys

out, ins = sys.argv[1], [json.load(open(p)) for p in sys.argv[2:]]
m = dict(ins[0])
for k in m:
    vals = [c[k] for c in ins if isinstance(c.get(k), (int, float))]
    if not vals:
        continue
    if k.startswith(("lat_", "deadline_", "outside_", "allowance_gib", "pad_used", "pad_mib_cal", "prereserve_mib_cal",
                     "pad_nontorch_mib_cal")):
        m[k] = max(vals)
    elif k == "tok_s":
        m[k] = min(vals)
m["merged_from"] = len(ins)
json.dump(m, open(out, "w"), indent=1)
print("merged", len(ins), "->", out, {k: round(m[k], 3) for k in m if k.startswith("lat_") and k.endswith("p99")})
