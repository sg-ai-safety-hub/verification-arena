"""Head rule (verifier/floor.py head_rule) on synthetic chase logs.

An honest prover's head follows the link. A static-ring prover on the sequential schedule keeps its ring and
re-labels it: its head must stay congruent to one value modulo the pool cycle (2,048 windows), so it either stands
still or jumps by whole cycles. The rule must pass the first and fail the second, whether the cheater jumps on every
cycle (jumps counted) or only on gaps long enough to allow it (rate from the heads far under the floor)."""
import math
import random

from schemes.memory_link_exhaustion.verifier import floor as F

W, RMAX, RATE, CYCLE = 32 << 20, 60.0, 56.4, 2048


def chases(T=200.0, rate=4.0, seed=1):
    r, t, out = random.Random(seed), 2.0, []
    while t < T:
        t += r.expovariate(rate)
        out.append((t, t + 0.006))
    return out


def main():
    ch = chases()
    wps = RATE * 1e9 / W                                      # windows per second on the link
    honest = [(a, b, int((a + 0.003) * wps)) for a, b in ch]
    h = F.head_rule(honest, W, RMAX)
    assert h["jumps"] == 0 and h["backwards"] == 0 and abs(h["gbs"] - RATE) < 0.05, h
    # a 0.5 s whole-host freeze, credited: the rate from the heads is unchanged
    fz = [(100.0, 100.5, 0.5)]
    frozen = [(a, b, int((a + 0.003 - (0.5 if a > 100.5 else max(0.0, a - 100.0))) * wps)) for a, b in ch]
    h2 = F.head_rule(frozen, W, RMAX, fz)
    assert h2["jumps"] == 0 and abs(h2["gbs"] - RATE) < 0.08 and h2["gbs_no_credit"] < RATE - 0.1, h2
    # static ring, greedy: head = the newest whole cycle the honest head has passed
    greedy = [(a, b, (int((a + 0.003) * wps) // CYCLE) * CYCLE) for a, b in ch]
    g = F.head_rule(greedy, W, RMAX)
    assert g["jumps"] > 100 and g["backwards"] == 0, g
    # static ring, careful: jump a cycle only when the gap since the last chase allows it under the cap
    head, prev, careful = (int(ch[0][0] * wps) // CYCLE) * CYCLE, None, []
    for a, b in ch:
        if prev is not None and math.ceil(RMAX * 1e9 * (b - prev) / W) + 2 >= CYCLE:
            head += CYCLE
        careful.append((a, b, head))
        prev = a
    c = F.head_rule(careful, W, RMAX)
    assert c["jumps"] == 0 and c["backwards"] == 0 and c["gbs"] < 0.1 * RATE, c
    # a head that steps back is counted
    back = list(honest)
    back[50] = (back[50][0], back[50][1], back[49][2] - 5)
    assert F.head_rule(back, W, RMAX)["backwards"] >= 1
    assert F.head_rule([], W, RMAX)["gbs"] is None and F.head_rule(honest[:1], W, RMAX)["n"] == 1
    print("test_head_rule_cpu: PASS")


if __name__ == "__main__":
    main()
