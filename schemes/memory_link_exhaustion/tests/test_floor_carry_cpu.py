"""Windowed-floor carry-back (verifier/floor.py), CPU only. Run: python schemes/memory_link_exhaustion/tests/test_floor_carry_cpu.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from schemes.memory_link_exhaustion.verifier import floor as F

CAP = 56.5
# late-reported completions (an honest run, seconds 133-149): a low bin and a catch-up bin above capacity
tperm = [55.7] * 10 + [55.7, 55.7, 55.8, 55.7, 55.8, 29.1, 82.3, 55.7, 55.8, 55.7, 55.7, 55.6, 55.7, 55.6, 55.6, 55.6, 55.5]
hpad = [56.4] * 10 + [26.8, 86.0] + [56.4] * 10        # an honest pad probe, seconds 120-141
outage = [56.4] * 10 + [33.0] + [56.4] * 10            # an honest run: a real pause, no catch-up bin
theft = [55.06] * 30                                   # a smooth 0.75 GB/s theft


def main():
    c = F.carry_back(tperm, CAP)
    assert abs(sum(c) - sum(tperm)) < 1e-9 and max(c) <= CAP + 1e-12          # bytes conserved, nothing above cap
    assert F.window_min(tperm, 10) < 53.1 and F.window_min(c, 10) > 55.4, (F.window_min(tperm, 10), F.window_min(c, 10))
    c = F.carry_back(hpad, CAP)
    assert F.window_min(c, 10) > 56.0, F.window_min(c, 10)
    assert F.window_min(F.carry_back(outage, CAP), 10) == F.window_min(outage, 10) < 54.1   # a real outage stays a dip
    assert F.carry_back(theft, CAP) == theft                                   # nothing to carry: theft unaffected
    # a cheater cannot manufacture credit: bins at or below cap are never raised by later bins that are also <= cap
    stall = [56.4, 56.4, 20.0, 56.5, 56.5, 56.4]
    assert F.carry_back(stall, CAP) == stall
    assert F.carry_back(tperm, 0) == [float(x) for x in tperm]                # flag off
    print("ALL OK")


if __name__ == "__main__":
    main()
