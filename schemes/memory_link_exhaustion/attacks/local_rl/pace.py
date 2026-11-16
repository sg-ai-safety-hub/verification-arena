"""Closed-loop pacing for the hidden job: work while the session's noise average is above a target, rest when it is
not. The cheater sees its own noise (window completions) and estimates the freeze credit the verifier will grant with
its own freeze meter (the verifier credits the time its own process was frozen; a whole-host freeze stops this
process's meter the same way). Worst case for the scheme: the target is the verifier's floor, which a real cheater
would have to estimate from honest runs on the same host."""
from __future__ import annotations

import multiprocessing as mp
import time


def _meter(credit, lead_s):
    last = time.perf_counter()
    while True:
        time.sleep(0.0002)
        t = time.perf_counter()
        gap = t - last
        if gap > 0.005 and gap - lead_s > 0:          # as the verifier's Revealer: a gap over 5 ms, minus the queued lead
            credit.value += gap - lead_s
        last = t


def start_freeze_meter(lead_s: float = 0.0024):
    """A small spawned process that adds up its own scheduling gaps. Returns f() -> seconds of credit so far."""
    ctx = mp.get_context("spawn")
    credit = ctx.Value("d", 0.0, lock=False)
    p = ctx.Process(target=_meter, args=(credit, lead_s), daemon=True)
    p.start()
    return lambda: float(credit.value)


class Pacer:
    """wait() blocks while the credited noise average since start is under target_gbs."""

    def __init__(self, target_gbs: float, noise_fn, credit_fn=None, grace_s: float = 5.0, poll_s: float = 0.02,
                 flow_windows: int = 4000):
        self.target, self.noise_fn, self.credit_fn = float(target_gbs), noise_fn, credit_fn or (lambda: 0.0)
        self.grace, self.poll = grace_s, poll_s
        self.flow_windows = flow_windows
        self._seen = None
        self.t0 = None
        self.rest_s = 0.0
        self.waits = 0

    def rate(self):
        n, W = self.noise_fn()
        el = time.perf_counter() - self.t0 - (self.credit_fn() - self.c0)
        return (n - self.n0) * W / max(el, 1e-6) / 1e9, el

    def wait(self):
        if self.t0 is None:
            # the average starts once the stream is flowing (flow_windows completed and still arriving): a baseline
            # taken during the cheater's warm-up, before the noise is up to speed, reads low for the whole session
            n, _ = self.noise_fn()
            now = time.perf_counter()
            if n < self.flow_windows or (self._seen is not None and n == self._seen[0]):
                self._seen = (n, now)
                return
            if self._seen is None or now - self._seen[1] < 0.5:
                if self._seen is None:
                    self._seen = (n, now)
                return
            self.t0, self.n0, self.c0 = now, n, self.credit_fn()
            return
        t = time.perf_counter()
        waited = False
        while True:
            r, el = self.rate()
            if el < self.grace or r >= self.target:
                break
            waited = True
            time.sleep(self.poll)
        if waited:
            self.waits += 1
            self.rest_s += time.perf_counter() - t

    def stats(self):
        r, el = self.rate() if self.t0 is not None else (None, 0.0)
        return {"target_gbs": self.target, "own_rate_gbs": None if r is None else round(r, 3), "elapsed_s": round(el, 1),
                "rest_s": round(self.rest_s, 1), "waits": self.waits,
                "credit_s": round(self.credit_fn() - self.c0, 2) if self.t0 is not None else 0.0}
