"""CPU replay of the floor rules (scripts/reproduce/replay.py) on synthetic runs.
Run: python -m schemes.memory_link_exhaustion.tests.test_replay_cpu"""
import json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "reproduce"))
import replay as R

W = 32 << 20


def synth(d, name, segs, freezes=(), t_end=200.0, t0=1000.0, sweep=None):
    """Arrival stamps for a run whose link rate (GB/s) is piecewise constant: segs = [(t_from, gbs), ...]; nothing
    arrives during the credited tail of each freeze (start, end, credit)."""
    t, arr, k = -0.01, [], 0
    while t < t_end:
        while k + 1 < len(segs) and t >= segs[k + 1][0]:
            k += 1
        t += W / (segs[k][1] * 1e9)
        for st, en, cr in freezes:
            if en - cr <= t < en:
                t = en
        arr.append(t)
    f = os.path.join(d, name + ".json")
    np.savez_compressed(f + ".replay.npz", arr_us=np.array(arr) * 1e6, ring_t=np.arange(62.0, t_end - 5, 2.7),
                        ring_cid=np.arange(len(np.arange(62.0, t_end - 5, 2.7))), freeze=np.array(freezes).reshape(-1, 3),
                        W=W, t0_abs=t0, slack=0.03, floor_gbs=0.0)
    json.dump({"prover": {"cheat": {"workload": {"sweep": sweep or []}}}}, open(f, "w"))
    return f


def main():
    d = tempfile.mkdtemp()
    fz = [(70.0, 70.6, 0.5)]
    h1 = R.load(synth(d, "h1", [(0, 56.0)]))
    h2 = R.load(synth(d, "h2", [(0, 56.0)], freezes=fz))
    th = R.load(synth(d, "th", [(0, 55.4)]))
    # need: a steady run needs its own rate; a verifier freeze costs it without the credit and not with it
    assert abs(R.need(h1)[0] - 56.0) < 0.05, R.need(h1)[0]
    n_c, n_nc = R.need(h2, True)[0], R.need(h2, False)[0]
    assert abs(n_c - 56.0) < 0.05 and n_nc < 55.7, (n_c, n_nc)
    assert abs(R.need(th)[0] - 55.4) < 0.05
    # floors from the honest pair: with the credit the frozen run does not drag the floor down
    s_c, s_nc = R.run_stats(h2, True), R.run_stats(h2, False)
    assert abs(s_c["mean"] - 56.0) < 0.03 and s_nc["mean"] < 55.85 and s_c["w10"] > 55.9 and s_nc["w10"] < 53.5, (s_c, s_nc)
    floor = min(R.run_stats(h, True)["mean"] for h in (h1, h2)) - 0.3
    # judge: the honest runs pass that floor with the credit; the frozen one fails without it; the theft fails both
    assert not R.judge(h1, floor)["caught"] and not R.judge(h2, floor, credit=True)["caught"]
    assert R.judge(h2, floor, credit=False)["behind_floor"] > 0
    assert R.judge(th, floor, credit=True)["caught"] and R.judge(th, floor, credit=False)["caught"]
    assert R.judge(h1, floor, thr10=57.0)["caught"]                      # the windowed rule alone can catch
    # sweep: three theft segments; the fitted line gives back the rate at which the noise crosses the floor
    segs = [(0, 56.0), (60, 55.8), (100, 55.6), (140, 55.2)]
    sw = [[1000.0 + a, 1000.0 + b, r, int(r * 1e9 * (b - a))] for a, b, r in ((20, 60, 0.0), (60, 100, 0.2), (100, 140, 0.4), (140, 180, 0.8))]
    sr = R.load(synth(d, "sw", segs, sweep=sw))
    rows, fit = R.sweep(sr, floor=55.5)
    assert len(rows) == 4 and abs(rows[0]["noise"] - 56.0) < 0.1 and abs(rows[3]["noise"] - 55.2) < 0.1, rows
    assert abs(fit["slope"] + 1.0) < 0.1 and abs(fit["theft_at_floor"] - 0.5) < 0.08, fit
    # adaptive sweep chooser: noise = 56.4 - 2.3 x theft with +-0.15 GB/s of dip noise, floor 55.98 -> 0.183 GB/s
    from baselines.pretrain.pretrain_qwen import next_theft_rate
    import random
    for seed in range(20):
        rnd = random.Random(seed)
        f = lambda r: 56.4 - 2.3 * r + rnd.uniform(-0.15, 0.15)
        pts, rate = [(0.0, f(0.0))], None
        for k in range(1, 9):
            est, rate = next_theft_rate(pts, 55.98, n_done=k)
            pts.append((rate, f(rate)))
        est, _ = next_theft_rate(pts, 55.98, n_done=9)
        assert 0.10 < est < 0.27, (seed, est, pts)
        assert all(0.02 <= x <= 2.0 for x, _ in pts[1:])
    print("ALL OK")


if __name__ == "__main__":
    main()
