#!/usr/bin/env python3
"""CPU replay of the noise-floor rules from saved runs (no GPU).

Every verified run saves <run>.json.replay.npz: the verifier's arrival stamp of each noise window, the ring challenge
times and the Revealer's freeze log. This tool reruns the floor rules on that evidence, so a margin, the freeze
credit or a new rule can be judged on runs already made.

  replay.py need RUN...                         highest cumulative floor each run passes, with / without freeze credit
  replay.py floors [--margin 0.3 --w10-margin 0.5 --credit 1] HONEST...     floors from honest runs
  replay.py judge --floor F [--thr10 T --credit 1] RUN...                    verdicts under given floors
  replay.py end [--floor F | --margin M] RUN...   end-of-session average (the one-number link rule); without
                                                --floor: the floor the given honest runs would set
  replay.py head [--floor F --rmax 60] RUN...    head rule: jumps / steps back between consecutive chases and the
                                                rate counted from the chase heads
  replay.py sweep RUN                           theft sweep: noise rate in each theft segment, and the rate that
                                                crosses --floor (needs the run json next to the npz)
RUN is the run json (the .replay.npz is found next to it) or the npz itself.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from schemes.memory_link_exhaustion.verifier import floor as F  # noqa: E402


def load(path):
    npz = path if path.endswith(".npz") else path + ".replay.npz"
    d = np.load(npz)
    r = {"arr": np.asarray(d["arr_us"], dtype=np.float64) / 1e6, "ring_t": np.asarray(d["ring_t"], dtype=np.float64),
         "freeze": np.asarray(d["freeze"], dtype=np.float64).reshape(-1, 3), "W": int(d["W"]), "t0": float(d["t0_abs"]),
         "heads": np.asarray(d["heads"], dtype=np.float64).reshape(-1, 3) if "heads" in d else np.zeros((0, 3)),
         "slack": float(d["slack"]), "floor": float(d["floor_gbs"]), "name": os.path.basename(npz).split(".json")[0]}
    j = npz[:-len(".replay.npz")]
    r["json"] = json.load(open(j)) if os.path.exists(j) else None
    return r


def need(r, credit=True):
    """Highest cumulative floor (GB/s) every ring challenge of the run passes, and the per-challenge values."""
    out = []
    for t in r["ring_t"]:
        el = t - r["slack"]
        if el <= 1.0:
            continue
        gb = np.searchsorted(r["arr"], el, side="right") * r["W"] / 1e9
        c = F.credit_before(r["freeze"], t) if credit else 0.0
        out.append(gb / max(el - c, 1e-6))
    return (min(out) if out else None), out


def series(r, skip=5.0, tail=2.0):
    """GB per 1 s bin from the arrival stamps, and the freeze credit (s) in each bin; bins start `skip` s after the
    first window and stop `tail` s before the last (as Checker.rate_var does from report times)."""
    a = r["arr"]
    if len(a) < 4:
        return [], [], 0.0
    t_start = a[0] + skip
    n = int((a[-1] - tail - t_start) // 1.0)
    edges = t_start + np.arange(n + 1)
    cw = np.searchsorted(a, edges, side="right")
    return list(np.diff(cw) * r["W"] / 1e9), F.credit_bins(r["freeze"], t_start, n), t_start


def end_avg(r, credit=True):
    """End-of-session rule: GB arrived by the last ring challenge over the elapsed time from t0 (credited clock)."""
    if not len(r["ring_t"]):
        return None, 0.0
    t = float(r["ring_t"][-1]); el = t - r["slack"]
    gb = np.searchsorted(r["arr"], el, side="right") * r["W"] / 1e9
    c = F.credit_before(r["freeze"], t) if credit else 0.0
    return gb / max(el - c, 1e-6), el


def run_stats(r, credit=True, lead=55, trail=10):
    """mean and windowed minima on the body of the run (after `lead` bins, before the last `trail`), as lib.sh WINX."""
    s, c, _ = series(r)
    b = s[lead:]; cb = c[lead:]
    if trail > 0 and len(b) > 2 * trail:
        b, cb = b[:-trail], cb[:-trail]
    if not credit:
        cb = [0.0] * len(b)
    if not b:
        return {"n": 0}
    return {"n": len(b), "mean": F.credited_mean(b, cb), "w10": F.window_min_credit(b, cb, 10),
            "w1": F.window_min_credit(b, cb, 1), "credit_s": sum(cb), "dips": sum(1 for x in b if x < 50)}


def judge(r, floor, thr10=None, credit=True):
    """Cumulative rule: ring challenges whose need is under the floor. Windowed rule: 10 s minimum vs thr10."""
    _, per = need(r, credit)
    bad = sum(1 for x in per if x < floor)
    st = run_stats(r, credit, lead=0, trail=0)
    win_ok = None if thr10 is None or st.get("w10") is None else bool(st["w10"] >= thr10)
    return {"challenges": len(per), "behind_floor": bad, "need": float(min(per)) if per else None, "w10": st.get("w10"),
            "caught": bad > 0 or win_ok is False}


def sweep(r, floor=None, credit=True, settle=3.0):
    wl = (((r["json"] or {}).get("prover") or {}).get("cheat") or {}).get("workload") or {}
    segs = wl.get("sweep") or []
    rows = []
    for a, b, rate, nbytes in segs:
        lo, hi = a - r["t0"] + settle, b - r["t0"]
        if hi - lo < 5:
            continue
        gb = (np.searchsorted(r["arr"], hi, side="right") - np.searchsorted(r["arr"], lo, side="right")) * r["W"] / 1e9
        c = (F.credit_before(r["freeze"], hi) - F.credit_before(r["freeze"], lo)) if credit else 0.0
        rows.append({"declared": rate, "moved": nbytes / 1e9 / max(b - a, 1e-9), "secs": round(hi - lo, 1),
                     "noise": gb / max(hi - lo - c, 1e-6), "credit_s": round(c, 3)})
    fit = None
    if len(rows) >= 3:
        x = np.array([q["moved"] for q in rows]); y = np.array([q["noise"] for q in rows])
        k, b0 = np.polyfit(x, y, 1)
        fit = {"noise_at_0": float(b0), "slope": float(k)}
        if floor and k < 0:
            fit["theft_at_floor"] = float((floor - b0) / k)
    return rows, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["need", "floors", "judge", "sweep", "end", "head"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--credit", type=int, default=1)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--w10-margin", type=float, default=0.5)
    ap.add_argument("--floor", type=float, default=0.0)
    ap.add_argument("--thr10", type=float, default=None)
    ap.add_argument("--rmax", type=float, default=60.0, help="head rule: nominal link maximum, GB/s")
    a = ap.parse_args()
    rs = [load(p) for p in a.runs]
    cr = bool(a.credit)
    if a.cmd == "need":
        for r in rs:
            n1, _ = need(r, True); n0, _ = need(r, False); st = run_stats(r, cr)
            print(f"{r['name']}: need {n1:.2f} GB/s with credit, {n0:.2f} without | body mean {st.get('mean', 0):.2f} "
                  f"w10 {st.get('w10')} dips {st.get('dips')} credit {st.get('credit_s', 0):.2f} s | freezes {len(r['freeze'])}")
    elif a.cmd == "end":
        vals = []
        for r in rs:
            e1, el = end_avg(r, True); e0, _ = end_avg(r, False)
            vals.append(e1)
            print(f"{r['name']}: end-of-session {e1:.2f} GB/s over {el:.0f} s ({e0:.2f} without credit)"
                  + (f" -> {'passes' if e1 >= a.floor else 'CAUGHT'} at floor {a.floor}" if a.floor else ""))
        if not a.floor and len(vals) > 1:
            print(f"lowest {min(vals):.2f}, spread {max(vals) - min(vals):.2f}; END_FLOOR at margin {a.margin}: "
                  f"{min(vals) - a.margin:.2f}")
    elif a.cmd == "head":
        for r in rs:
            h = F.head_rule(r["heads"], r["W"], a.rmax, r["freeze"] if cr else None)
            if h["gbs"] is None:
                print(f"{r['name']}: no chase heads saved"); continue
            bad = h["jumps"] or h["backwards"] or (a.floor and h["gbs"] < a.floor)
            print(f"{r['name']}: {h['jumps']} jumps, {h['backwards']} back over {h['n']} chases; {h['gbs']:.2f} GB/s from heads "
                  f"over {h['elapsed_s']:.0f} s" + (f" -> {'CAUGHT' if bad else 'passes'} at floor {a.floor}" if a.floor else ""))
    elif a.cmd == "floors":
        sts = [run_stats(r, cr) for r in rs]
        needs = [need(r, cr)[0] for r in rs]
        for r, st, n in zip(rs, sts, needs):
            print(f"{r['name']}: mean {st['mean']:.2f} w10 {st['w10']} need {n:.2f}")
        print(f"FLOOR={min(s['mean'] for s in sts) - a.margin:.2f} THR10={min(s['w10'] for s in sts) - a.w10_margin:.2f} "
              f"(margins {a.margin} / {a.w10_margin}, credit {'on' if cr else 'off'}); lowest honest need {min(needs):.2f}")
    elif a.cmd == "judge":
        for r in rs:
            j = judge(r, a.floor, a.thr10, cr)
            print(f"{r['name']}: {'CAUGHT' if j['caught'] else 'passes'} | {j['behind_floor']}/{j['challenges']} ring challenges "
                  f"behind floor {a.floor}, need {j['need']:.2f}, w10 {j['w10']}" + (f" vs {a.thr10}" if a.thr10 else ""))
    else:
        for r in rs:
            rows, fit = sweep(r, a.floor or r["floor"], cr)
            for q in rows:
                print(f"{r['name']}: declared {q['declared']:.2f} moved {q['moved']:.3f} GB/s -> noise {q['noise']:.2f} GB/s "
                      f"over {q['secs']} s (credit {q['credit_s']} s)")
            if fit:
                print(f"{r['name']}: noise = {fit['noise_at_0']:.2f} {fit['slope']:+.2f} x theft"
                      + (f"; crosses the floor at {fit['theft_at_floor']:.2f} GB/s" if "theft_at_floor" in fit else ""))


if __name__ == "__main__":
    main()
