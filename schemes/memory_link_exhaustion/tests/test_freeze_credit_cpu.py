"""Freeze credit for the noise floor (verifier/floor.py), CPU only.
Run: python -m schemes.memory_link_exhaustion.tests.test_freeze_credit_cpu"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from schemes.memory_link_exhaustion.verifier import floor as F

W = 32 << 20
FLOOR = 55.11        # GB/s
RATE = 56.0          # what the link gives when nothing is frozen


def w_due(t, events, credit=True):
    el = t - (F.credit_before(events, t) if credit else 0.0)
    return max(0, int(el * FLOOR * 1e9 / W))


def arrived(t, events, rate=RATE):
    """Windows delivered by t at `rate` GB/s, nothing during the credited part of each freeze."""
    return int((t - F.credit_before(events, t)) * rate * 1e9 / W)


def main():
    # credit_before: none before the freeze, partial inside its credited tail, all of it after
    ev = [(100.0, 100.6, 0.5)]                       # a 0.6 s gap, 0.1 s covered by windows in flight
    assert F.credit_before(ev, 99.0) == 0.0
    assert F.credit_before(ev, 100.05) == 0.0        # still inside the part the in-flight windows cover
    assert abs(F.credit_before(ev, 100.35) - 0.25) < 1e-9
    assert abs(F.credit_before(ev, 100.6) - 0.5) < 1e-9 and abs(F.credit_before(ev, 500.0) - 0.5) < 1e-9
    assert F.credit_before([], 10.0) == 0.0

    # ring rule (cumulative from t0): an honest prover at 56.0 GB/s builds a lead of 0.89 GB/s over the floor. A 0.5 s
    # verifier stall 20 s in costs 28 GB, more than the 18 GB lead it has by then, so without the credit it is behind
    # the floor until the lead grows back (about 31 s); with the credit it is never behind
    e2 = [(20.0, 20.6, 0.5)]
    grid = [x / 10 for x in range(50, 3000)]
    behind = [t for t in grid if arrived(t, e2) < w_due(t, e2, credit=False)]
    assert behind and min(behind) > 20.0 and max(behind) < 33.0, (behind[:3], behind[-3:])
    assert all(arrived(t, e2) >= w_due(t, e2) for t in grid)

    # the credit does not hide a theft: a prover at FLOOR - 0.5 GB/s is behind with or without it
    slow = FLOOR - 0.5
    assert all(arrived(t, e2, slow) < w_due(t, e2) for t in grid if t > 1)
    # and a prover cannot earn credit: with no verifier freeze the rule is the old one
    assert all(w_due(t, []) == w_due(t, [], credit=False) for t in grid)

    # credit_bins: the credited tail [100.1, 100.6] lands in the bin that holds it, split across a boundary
    b = F.credit_bins(ev, 95.0, 10)
    assert abs(sum(b) - 0.5) < 1e-9 and abs(b[5] - 0.5) < 1e-9, b
    b = F.credit_bins([(100.8, 101.3, 0.5)], 95.0, 10)
    assert abs(b[5] - 0.2) < 1e-9 and abs(b[6] - 0.3) < 1e-9, b
    assert F.credit_bins(ev, 200.0, 5) == [0.0] * 5 and F.credit_bins(ev, 0.0, 0) == []

    # windowed floor: the stalled second carries half the bytes; over (10 - 0.5) s the rate is the honest one
    body = [RATE] * 10 + [RATE * 0.5] + [RATE] * 10
    cr = [0.0] * 10 + [0.5] + [0.0] * 10
    assert F.window_min(body, 10) < 53.3
    assert abs(F.window_min_credit(body, cr, 10) - RATE) < 0.01
    assert F.window_min_credit(body, [], 10) == F.window_min(body, 10)            # no credit: unchanged
    assert F.window_min_credit(body, [0.0] * len(body), 10) == F.window_min(body, 10)
    # a dip the verifier did not cause gets nothing
    assert F.window_min_credit(body, [0.5] + [0.0] * 20, 10) == F.window_min(body, 10)
    # the credit is capped: a window is never scored over less than half its length
    assert F.window_min_credit([1.0] * 10, [1.0] * 10, 10) == round(10.0 / 5.0, 3)
    assert F.window_min_credit([1.0] * 5, [], 10) is None
    # credited mean for the cumulative-floor calibration
    assert abs(F.credited_mean(body, cr) - sum(body) / (len(body) - 0.5)) < 1e-9
    assert F.credited_mean(body, []) == sum(body) / len(body) and F.credited_mean([], []) is None
    print("ALL OK")


if __name__ == "__main__":
    main()
