"""Device-side challenge gate for a hidden job.

The cheater owns the native challenge server, so it can do what the honest prover does with VARENA_MM_GATE:
every challenge graph raises a device word first and drops it last (actenc2.gate_set), and the hidden job's
stream runs a one-CTA wait kernel before each of its ops. While a challenge runs, the hidden stream parks at its
next wait, so the challenge meets at most the one kernel already in flight (bounded by chunking the big ops).

It needs no host round trip per op and gates the backward pass too (GateFn).
"""
from __future__ import annotations

import os

import torch

_ON_GPU = torch.cuda.is_available() and os.environ.get("GSHARD_NO_GATE", "0") != "1"
STATS = {"waits": 0}

if _ON_GPU:
    import triton
    import triton.language as tl

    @triton.jit
    def _gs_wait_kernel(gate):
        g = tl.load(gate, volatile=True)
        while g != 0:
            g = tl.load(gate, volatile=True)


def _gate():
    from schemes.memory_link_exhaustion.encodings import actenc2 as AE
    return AE.gate_dev()


_TIME_EVERY = int(os.environ.get("VARENA_GATE_TIME", "0") or 0)     # time every Nth gate wait
_PEND: list = []
WAIT_MS: list = []
_LK = __import__("threading").Lock()


def _reap():
    with _LK:
        while _PEND and _PEND[0][1].query():
            a, b = _PEND.pop(0)
            WAIT_MS.append(a.elapsed_time(b))
        if len(WAIT_MS) > 200000:
            del WAIT_MS[:100000]


def wait_stats(reset=True):
    """Mean / p99 / max device time (ms) of the sampled gate-wait kernels since the last call."""
    _reap()
    with _LK:
        w = sorted(WAIT_MS)
        if reset:
            WAIT_MS.clear()
    if not w:
        return {"n": 0}
    return {"n": len(w), "mean_ms": round(sum(w) / len(w), 4), "p50_ms": round(w[len(w) // 2], 4),
            "p99_ms": round(w[min(len(w) - 1, int(0.99 * len(w)))], 3), "max_ms": round(w[-1], 2)}


def wait():
    """Queue a gate wait on the current stream (no-op off GPU)."""
    STATS["waits"] += 1
    if _ON_GPU:
        if _TIME_EVERY and STATS["waits"] % _TIME_EVERY == 0:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            _gs_wait_kernel[(1,)](_gate(), num_warps=1)
            b.record()
            with _LK:
                _PEND.append((a, b))
            if len(_PEND) > 64:
                _reap()
        else:
            _gs_wait_kernel[(1,)](_gate(), num_warps=1)


class GateFn(torch.autograd.Function):
    """Identity that queues a gate wait in forward and in backward."""

    @staticmethod
    def forward(ctx, x):
        wait()
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        wait()
        return g


def g(x):
    return GateFn.apply(x) if x.requires_grad else (wait(), x)[1]


def install_gated_graphs(native):
    """Wrap every native challenge graph class: gate up before the challenge kernel, down after it (captured in
    the graph, on the challenge stream)."""
    from schemes.memory_link_exhaustion.encodings import actenc2 as AE
    if getattr(native, "_gshard_gated", False):
        return
    AE.gate_set(0)                           # allocate the gate + compile before any capture

    def wrap(cls):
        class Gated(cls):
            def __init__(self_, launch, *a, **kw):
                def l2(o, n):
                    AE.gate_set(1)
                    launch(o, n)
                    AE.gate_set(0)
                super().__init__(l2, *a, **kw)
        Gated.__name__ = "Gated" + cls.__name__
        return Gated
    for name in ("ChaseGraph",):
        setattr(native, name, wrap(getattr(native, name)))
    native._gshard_gated = True
