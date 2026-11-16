"""Shared pieces for the cheaters that run on the served (encoded, resident) weights: BflyLin (a resident encoded
weight decoded into scratch per call, in row slices) and build_model (the declared model over those weights).
Used by local_rl.
"""
from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

from schemes.memory_link_exhaustion.encodings import ref
from schemes.memory_link_exhaustion.core.model import Cfg, Lin, Model, SafeTensors, small_tensors


def log(*a):
    print(time.strftime("%H:%M:%S"), "[hidden-rl]", *a, file=sys.stderr, flush=True)


# v2: called before every linear (and every rollout decode step) to yield while vLLM runs a heavy step
GATE = None
# SLICE_GATE False (local_rl spec slice_gate false, ): gate once per linear call, not before every decode
# slice; at decode batch 2 the per-slice gate-wait launches were ~a fifth of the cheater's ~17k launches per step
SLICE_GATE = True


class BflyLin(Lin):
    """A resident butterfly-encoded weight [N, K]: decoded into scratch per call.

    v1 decoded the whole matrix into a fresh buffer (lm_head: 1.09 GB per call). With SCRATCH_ROWS set
    (v2) it decodes row slices of at most SCRATCH_ROWS rows into one shared scratch buffer, so the hidden
    job's decode scratch is fixed and small."""

    SCRATCH_ROWS = 0
    SCRATCH_BYTES = 0
    _scr = None

    def __init__(self, t: torch.Tensor, key: int, ordinal: int):
        super().__init__(t, key, "torch")
        self.ordinal = ordinal
        self.sloth = ref.WENC == "sloth"
        if not self.sloth:
            from schemes.memory_link_exhaustion.gpu.cuda import build as CB
            self.ext = CB.load()

    def decoded(self) -> torch.Tensor:
        out = torch.empty(self.t.shape, dtype=torch.bfloat16, device=self.t.device)
        if self.sloth:
            ref.w_xor(self.t, out, self.key, self.ordinal)
        else:
            self.ext.bfly_xor_to(self.t, out, self.key, self.ordinal)
        return out

    def embed_rows(self, ids):
        """Embedding lookup on the encoded table: decode only the rows asked for (sloth64 W)."""
        if not self.sloth:
            raise RuntimeError("embed_rows: sloth W only")
        from schemes.memory_link_exhaustion.encodings import sloth64
        return sloth64.gather_decode(self.t, ids, ref.sloth_seed(self.key, self.ordinal))

    def _slices(self):
        N, K = self.t.shape
        R = self.SCRATCH_ROWS
        # slice granularity: sloth decodes any 64-row slice; the butterfly W keystream comes in 128-row super-chunks
        g = 64 if self.sloth else 128
        if self.SCRATCH_BYTES:
            # fixed scratch in bytes: wide slices for narrow matrices (fewer, larger kernels)
            R = max(g, min(N, self.SCRATCH_BYTES // (2 * K)) // g * g)
        else:
            R = max(g, R // g * g)
        need = R * K
        if BflyLin._scr is None or BflyLin._scr.numel() < need:
            BflyLin._scr = torch.empty(need, dtype=torch.bfloat16, device=self.t.device)
        for r0 in range(0, N, R):
            if GATE and r0 and SLICE_GATE:
                GATE()                       # ring-aware throttle: one decode slice + GEMM per chunk
            n = min(R, N - r0)
            w = BflyLin._scr[:n * K].view(n, K)
            if self.sloth:
                ref.w_xor(self.t[r0:r0 + n], w, self.key, self.ordinal, row0=r0)
            else:
                self.ext.bfly_xor_rows(self.t[r0:r0 + n], w, self.key, self.ordinal, r0)
            yield r0, n, w

    def fwd(self, x):
        if GATE:
            GATE()
        if not self.SCRATCH_ROWS:
            return F.linear(x, self.decoded())
        y = torch.empty(*x.shape[:-1], self.t.shape[0], dtype=x.dtype, device=x.device)
        for r0, n, w in self._slices():
            y[..., r0:r0 + n] = F.linear(x, w)
        return y

    def bwd(self, g):
        if GATE:
            GATE()
        if not self.SCRATCH_ROWS:
            return g @ self.decoded()
        out = None
        for r0, n, w in self._slices():
            part = g[..., r0:r0 + n] @ w
            out = part if out is None else out.add_(part)
        return out


def build_model(prover, model_path, Tmax=4096) -> Model:
    cfg = Cfg.from_hf(model_path)
    lins = {}
    for name, off, shape, src, key in prover.w_entries:
        n = shape[0] * shape[1]
        t = prover.wbuf[off: off + 2 * n].view(torch.bfloat16).view(shape)
        if key is None or key < 0 or ref.WENC not in ("bfly", "sloth"):
            lins[name] = Lin(t, None, "torch")
        else:
            lins[name] = BflyLin(t, int(key), ref.w_ordinal(off))
    st = SafeTensors(model_path)          # norms and biases: a few MB, in the allowance
    small = {n: st.get(n).to(device="cuda", dtype=torch.bfloat16) for n in small_tensors(cfg)}
    return Model(cfg, lins, small, Tmax, "cuda")
