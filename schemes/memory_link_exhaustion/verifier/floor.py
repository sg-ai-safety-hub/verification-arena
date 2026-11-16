"""Windowed noise floor scoring (pure Python, CPU-testable).

carry_back: a 1 s bin above the link's physical maximum `cap` cannot be real arrivals in that
second; it is completions that were *reported* late (a short stall in the reporting path, e.g. some honest runs
show 29.1 then 82.3 GB/s around a ~0.2-0.5 s prover-side pause). The excess is credited to the preceding bins, each
filled at most to `cap`. The total (the cumulative byte count) is unchanged, so only bytes the cumulative count shows
were delivered are ever credited, and a real outage with no catch-up above capacity still scores as a dip. A cheater
cannot manufacture a catch-up bin: the following second cannot carry more than the link maximum.

Freeze credit: the verifier's Revealer loop logs its own freezes as (start s, end s, credit s), where the credit is
the link time the freeze cost (the gap minus what the windows already in flight covered). While the Revealer is
frozen it reveals nothing, so no prover can deliver noise; that time is taken off the clock the floor is measured
against. credit_before gives the credit accrued before a challenge time; credit_bins spreads it over 1 s bins;
window_min_credit is window_min at the rate over (window length - credited time). Only the verifier's own process
writes the log, so a prover on its own hardware cannot earn credit."""
from __future__ import annotations


def carry_back(series, cap):
    s = [float(x) for x in series]
    if not cap or cap <= 0:
        return s
    for i in range(len(s)):
        ex = s[i] - cap
        if ex <= 0:
            continue
        s[i] = cap
        j = i - 1
        while ex > 1e-12 and j >= 0:
            room = cap - s[j]
            if room > 0:
                d = min(room, ex)
                s[j] += d
                ex -= d
            j -= 1
        if ex > 1e-12:          # nothing earlier to absorb it (start of series): leave it where it was
            s[i] += ex
    return s


def window_min(body, L):
    if len(body) < L:
        return None
    return round(min(sum(body[i:i + L]) / L for i in range(len(body) - L + 1)), 3)


MAX_CREDIT_FRAC = 0.5      # a window is never scored over less than half its length


def credit_before(events, t):
    """Credit (s) accrued before time t. events: rows of (start, end, credit) in the same clock as t; the credited
    time is the tail of each freeze, [end - credit, end]. A freeze still in progress at t counts up to t."""
    c = 0.0
    for st, en, cr in events:
        lo = en - cr
        if t >= en:
            c += cr
        elif t > lo:
            c += t - lo
    return c


def credit_bins(events, t_start, n, width=1.0):
    """Credited seconds falling in each of n consecutive bins of `width` s starting at t_start."""
    out = [0.0] * max(0, int(n))
    for st, en, cr in events:
        lo = en - cr
        i0 = max(0, int((lo - t_start) // width))
        i1 = min(len(out) - 1, int((en - t_start) // width))
        for i in range(i0, i1 + 1):
            a, b = t_start + i * width, t_start + (i + 1) * width
            ov = min(b, en) - max(a, lo)
            if ov > 0:
                out[i] += ov
    return out


def window_min_credit(body, credit, L):
    """window_min with the credited time taken off each window's length (credit: seconds per 1 s bin, same
    alignment as body). With no credit this is window_min."""
    if len(body) < L:
        return None
    cr = list(credit or []) + [0.0] * len(body)
    best = None
    for i in range(len(body) - L + 1):
        c = min(sum(cr[i:i + L]), MAX_CREDIT_FRAC * L)
        r = sum(body[i:i + L]) / (L - c)
        best = r if best is None or r < best else best
    return round(best, 3)


def credited_mean(body, credit):
    """Mean rate of a 1 s series over its length minus the credited time (the cumulative-floor calibration)."""
    if not body:
        return None
    c = min(sum(list(credit or [])[:len(body)]), MAX_CREDIT_FRAC * len(body))
    return sum(body) / (len(body) - c)


def head_rule(heads, W, r_max_gbs, events=None, slop=2):
    """The head cannot advance faster than the link, and the link rate is counted from the heads.

    heads: (t_sent, t_recv, win) of every mixed chase, verifier clock (s), win = the newest completed noise window the
    prover's challenge server read for that chase (-1: none yet; those are skipped). Between consecutive chases a, b:
    0 <= win_b - win_a <= ceil(r_max * (t_recv_b - t_sent_a) / W) + slop. A ring that is kept and re-labelled (its
    content is right again one pool cycle later on a public cyclic schedule) has to jump the head by a whole cycle,
    which only a gap of over a cycle time allows; on every other gap it stands still, so its average rate is far
    under any floor. The average: (last win - first win) * W over the time between the first and last head reads,
    minus the freeze credit in between (events: the Revealer's (start, end, credit) log on the same clock).
    Returns dict(n, jumps, backwards, worst_excess, gbs, gbs_no_credit, elapsed_s, credit_s)."""
    import math
    h = sorted((float(a), float(b), int(w)) for a, b, w in heads if int(w) >= 0)
    out = {"n": len(h), "jumps": 0, "backwards": 0, "worst_excess": 0, "gbs": None, "gbs_no_credit": None,
           "elapsed_s": 0.0, "credit_s": 0.0}
    if len(h) < 2:
        return out
    for (ta, _ra, wa), (_tb, rb, wb) in zip(h, h[1:]):
        d = wb - wa
        if d < 0:
            out["backwards"] += 1
            continue
        cap = math.ceil(r_max_gbs * 1e9 * max(rb - ta, 0.0) / W) + slop
        if d > cap:
            out["jumps"] += 1
            out["worst_excess"] = max(out["worst_excess"], d - cap)
    t0, t1 = h[0][1], h[-1][0]                  # first head read is no later than its t_recv; last no earlier than its t_sent
    el = max(t1 - t0, 1e-6)
    cr = (credit_before(events, t1) - credit_before(events, t0)) if events is not None and len(events) else 0.0
    gb = (h[-1][2] - h[0][2]) * W / 1e9
    out.update({"gbs_no_credit": gb / el, "gbs": gb / max(el - cr, 1e-6), "elapsed_s": el, "credit_s": cr})
    return out
