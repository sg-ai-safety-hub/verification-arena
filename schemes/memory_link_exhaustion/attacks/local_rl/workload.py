"""Link-free local LoRA RL: the whole GRPO loop on one cheating GPU, on the served weights.

Every LoRA RL cheater measured so far is bound by memory (lora_rl_1280: 3 rollouts fit, 0.031 samples/s, ~42x) or by
the stolen link (the removed sharded_rl: activations cross the link at bf16). This one uses no link at all, so the inbound floor
does not touch it; it is bound only by the cheater's leftover memory and by the sloth decode of the served weights,
which the honest prover pays too. Real mode (the declared real weights are resident, sloth-encoded; BflyLin decodes
them per call in small scratch slices, as hidden_rl does).

What makes more rollouts fit in ~1.9 GiB:

1. **Shared-prompt KV.** GRPO samples `group` rollouts per prompt; their prompt KV is identical, so it is stored once
   per prompt (level 1) and each rollout stores only its own generated tokens (level 2). Optionally a prefix shared
   by every prompt (`shared_prefix` tokens, e.g. a fixed system prompt or the 1,024-token pad of the benchmark task)
   is stored once per optimiser step (level 0). Attention reads [level 0 | level 1 | own level 2]. Qwen2.5-32B KV is
   256 KiB per token over 64 layers: 1,024 prompt + 4 x 256 generated = 537 MB per prompt group instead of
   4 x 335 MB, so ~2x the rollouts per decode step (~4x with a shared 1,000-token prefix).
2. **Two-level checkpointing in the update.** Segment inputs every `seg` layers are kept; a segment's layer inputs are
   recomputed in its backward. `train_mb` samples per micro-batch share each pass of weight decode.
3. **Streamed LM head.** Log-probs over the 152k vocabulary are computed slice by slice from the decode scratch with
   a running log-sum-exp (forward) and recomputed softmax slices (backward); no logits tensor is ever materialised.
4. **Optional host AdamW** (`host_adam`): LoRA master weights and optimiser state on the host. Per optimiser step
   33 MB of gradients go out and 33 MB of bf16 LoRA weights come in (a few kB/s on average).

Everything is at declared precision (bf16 weights, activations and KV; LoRA rank 16 on q and v of all 64 layers;
AdamW), on-policy within each optimiser step, nothing compressed. Rewards are random (rate measurement only); the
tokens are sampled from the policy at temperature 1.

step() runs one generation batch (`conc_prompts` prompts x `group` rollouts) and its update micro-batches, with an
optimiser step every `n_prompts / conc_prompts` calls, and returns the number of samples.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from schemes.memory_link_exhaustion.attacks.gated_shard import gate as G
from schemes.memory_link_exhaustion.core.model import Model, rms, rope

_core_rms = rms
if __import__("os").environ.get("VARENA_FUSED_NORM", "1") == "1":
    def rms(x, w, eps):  # noqa: F811
        """Fused RMSNorm on the bf16 input: no fp32 [tokens, d] copies kept for autograd."""
        # weight applied after the bf16 cast, exactly as core.model.rms: bitwise equal to it (CPU, d 5,120). Only the
        # 16-bit path is fused; fp32/fp64 (CPU tests) keep core.model.rms, which normalises in fp32
        if x.dtype not in (torch.bfloat16, torch.float16):
            return _core_rms(x, w, eps)
        return F.rms_norm(x, (x.shape[-1],), None, eps) * w


def _w_slices(lin, rows=4096):
    """Row slices (r0, n, W[r0:r0+n]) of a (possibly sloth-encoded) weight, decoded into scratch when encoded."""
    if hasattr(lin, "_slices") and (getattr(lin, "SCRATCH_ROWS", 0) or getattr(lin, "SCRATCH_BYTES", 0)):
        yield from lin._slices()
        return
    W = lin.t if lin.key is None else lin.decoded()
    for r0 in range(0, W.shape[0], rows):
        G.wait()
        yield r0, min(rows, W.shape[0] - r0), W[r0:r0 + rows]


class StreamedLogProb(torch.autograd.Function):
    """log softmax(x W^T)[tgt] over a vocabulary-sized W streamed in row slices; W is frozen."""

    @staticmethod
    def forward(ctx, x, tgt, lin):
        N = x.shape[0]
        acc = torch.promote_types(x.dtype, torch.float32)
        m = torch.full((N,), float("-inf"), device=x.device, dtype=acc)
        s = torch.zeros(N, device=x.device, dtype=acc)
        tl = torch.zeros(N, device=x.device, dtype=acc)
        for r0, n, w in _w_slices(lin):
            z = (x @ w.t()).to(acc)
            sel = (tgt >= r0) & (tgt < r0 + n)
            idx = (tgt - r0).clamp(0, n - 1)
            tl = torch.where(sel, z.gather(1, idx[:, None])[:, 0], tl)
            mn = torch.maximum(m, z.max(1).values)
            s = s * torch.exp(m - mn) + torch.exp(z - mn[:, None]).sum(1)
            m = mn
        lse = m + torch.log(s)
        ctx.save_for_backward(x, tgt, lse)
        ctx.lin = lin
        return tl - lse

    @staticmethod
    def backward(ctx, g):
        x, tgt, lse = ctx.saved_tensors
        acc = torch.promote_types(x.dtype, torch.float32)
        gx = torch.zeros(x.shape, dtype=acc, device=x.device)
        g = g.to(acc)
        for r0, n, w in _w_slices(ctx.lin):
            z = (x @ w.t()).to(acc)
            dz = -torch.exp(z - lse[:, None]) * g[:, None]
            sel = ((tgt >= r0) & (tgt < r0 + n)).to(acc)
            idx = (tgt - r0).clamp(0, n - 1)
            dz.scatter_add_(1, idx[:, None], (g * sel)[:, None])
            gx.add_((dz.to(w.dtype) @ w).to(acc))
        return gx.to(x.dtype), None, None


_SCR = {}


def _rows(lin, r0, n, slot):
    """Rows r0..r0+n of a (possibly encoded) weight: a view of the plain weight, or decoded once into scratch[slot]."""
    if lin.key is None:
        return lin.t[r0:r0 + n]
    K = lin.t.shape[1]
    buf = _SCR.get(slot)
    if buf is None or buf.numel() < n * K or buf.device != lin.t.device:
        buf = _SCR[slot] = torch.empty(n * K, dtype=torch.bfloat16, device=lin.t.device)
    w = buf[:n * K].view(n, K)
    if getattr(lin, "sloth", False):
        from schemes.memory_link_exhaustion.encodings import ref as _ref
        _ref.w_xor(lin.t[r0:r0 + n], w, lin.key, lin.ordinal, row0=r0)
    elif hasattr(lin, "ext"):
        lin.ext.bfly_xor_rows(lin.t[r0:r0 + n], w, lin.key, lin.ordinal, r0)
    else:
        w.copy_(lin.decoded()[r0:r0 + n])
    return w


class SliceMLP(torch.autograd.Function):
    """Slice-outer SwiGLU MLP with LoRA. The token-chunked MLP (mlp_chunk) re-decodes the full gate_up
    and down weights for every chunk of every pass (forward, segment recompute, layer recompute, chunk recompute,
    backward: ~5 x T/chunk decodes per layer per micro-batch). Here each gate_up row slice is decoded once per pass and
    applied to all tokens, building the [T, I] activation slice by slice ([T, 2I] gate/up output never exists); down is
    one decode per pass. Saved for backward: the normed input only. Backward = one down decode + one gate_up decode.
    Forward is bit-identical to _mlp_core's (same per-element dot products); gradients agree to rounding."""

    @staticmethod
    def forward(ctx, x, Ag, Bg, Au, Bu, Ad, Bd, gu, dn, I, scale, rows):
        a = SliceMLP._act(x, Ag, Bg, Au, Bu, gu, I, scale, rows)
        y = dn(a)
        if Ad is not None:
            y = y + torch.mm(torch.mm(a, Ad.t()), Bd.t()) * scale
        ctx.save_for_backward(x, Ag, Bg, Au, Bu, Ad, Bd)
        ctx.cfg = (gu, dn, I, scale, rows)
        return y

    @staticmethod
    def _gu(x, Ag, Bg, Au, Bu, gu, I, scale, i0, n, xag, xau):
        G.wait()
        wg, wu = _rows(gu, i0, n, "g"), _rows(gu, I + i0, n, "u")
        g, u = F.linear(x, wg), F.linear(x, wu)
        if Ag is not None:
            g = g + torch.mm(xag, Bg[i0:i0 + n].t()) * scale
        if Au is not None:
            u = u + torch.mm(xau, Bu[i0:i0 + n].t()) * scale
        return g, u, wg, wu

    @staticmethod
    def _act(x, Ag, Bg, Au, Bu, gu, I, scale, rows):
        xag = torch.mm(x, Ag.t()) if Ag is not None else None
        xau = torch.mm(x, Au.t()) if Au is not None else None
        a = torch.empty(x.shape[0], I, dtype=x.dtype, device=x.device)
        for i0 in range(0, I, rows):
            n = min(rows, I - i0)
            g, u, _, _ = SliceMLP._gu(x, Ag, Bg, Au, Bu, gu, I, scale, i0, n, xag, xau)
            a[:, i0:i0 + n] = F.silu(g) * u
        return a

    @staticmethod
    def backward(ctx, gy):
        x, Ag, Bg, Au, Bu, Ad, Bd = ctx.saved_tensors
        gu, dn, I, scale, rows = ctx.cfg
        acc = torch.promote_types(x.dtype, torch.float32)
        ga = dn.bwd(gy)                                          # one down decode
        ud = None
        if Ad is not None:
            ud = torch.mm(gy, Bd) * scale                         # [T, r]
            ga = ga + torch.mm(ud, Ad)
        xag = torch.mm(x, Ag.t()) if Ag is not None else None
        xau = torch.mm(x, Au.t()) if Au is not None else None
        gx = torch.zeros(x.shape, dtype=acc, device=x.device)
        dAg = dBg = dAu = dBu = dAd = dBd = None
        gxag = torch.zeros_like(xag) if Ag is not None else None
        gxau = torch.zeros_like(xau) if Au is not None else None
        if Bg is not None:
            dBg = torch.zeros_like(Bg)
        if Bu is not None:
            dBu = torch.zeros_like(Bu)
        if Ad is not None:
            dAd = torch.zeros_like(Ad)
            aA = torch.zeros(x.shape[0], Ad.shape[0], dtype=x.dtype, device=x.device)
        for i0 in range(0, I, rows):                             # one gate_up decode
            n = min(rows, I - i0)
            g, u, wg, wu = SliceMLP._gu(x, Ag, Bg, Au, Bu, gu, I, scale, i0, n, xag, xau)
            sg = torch.sigmoid(g)
            sl = g * sg
            gas = ga[:, i0:i0 + n]
            if Ad is not None:
                a_s = sl * u
                aA = aA + torch.mm(a_s, Ad[:, i0:i0 + n].t())
                dAd[:, i0:i0 + n] = torch.mm(ud.t(), a_s)
            dg = gas * u * (sg * (1 + g * (1 - sg)))
            du = gas * sl
            gx.add_(torch.mm(dg, wg).to(acc)).add_(torch.mm(du, wu).to(acc))
            if Ag is not None:
                dBg[i0:i0 + n] = torch.mm((dg * scale).t(), xag)
                gxag = gxag + torch.mm(dg * scale, Bg[i0:i0 + n])
            if Au is not None:
                dBu[i0:i0 + n] = torch.mm((du * scale).t(), xau)
                gxau = gxau + torch.mm(du * scale, Bu[i0:i0 + n])
        if Ad is not None:
            dBd = torch.mm((gy * scale).t(), aA)
        if Ag is not None:
            dAg = torch.mm(gxag.t(), x)
            gx.add_(torch.mm(gxag, Ag).to(acc))
        if Au is not None:
            dAu = torch.mm(gxau.t(), x)
            gx.add_(torch.mm(gxau, Au).to(acc))
        return (gx.to(x.dtype), dAg, dBg, dAu, dBu, dAd, dBd) + (None,) * 5


def _q8(x):
    """int8 with a per-row (last dim) scale: x [..., D] -> (q int8 [..., D], scale [..., 1] in x's dtype)."""
    sc = (x.abs().amax(-1, keepdim=True).float() / 127.0).clamp_min(1e-12)
    q = torch.round(x.float() / sc).clamp_(-127, 127).to(torch.int8)
    return q, sc.to(x.dtype)


def _dec_attn(q, levels, ko, vo, Pc, g, ko_add=None):
    """Decode-step attention without per-rollout copies of the shared levels.
    q [b, Hq, 1, D] (b = Pc * g, prompt-major rows); levels: (K, V) [n, Hkv, L, D] with n = 1 (shared by all rows) or
    Pc (one per prompt); ko, vo [b, Hkv, T, D] (each row's own KV). GQA: query head h reads KV head h // (Hq / Hkv).
    ko_add: None, or an additive mask broadcast over the own-KV logits ([.., T], -inf at not-yet-written positions):
    used by the fixed-shape (graph-capturable) decode, where ko/vo are the full G-length buffers."""
    b, Hq, _, D = q.shape
    Hkv = ko.shape[1]
    rep = Hq // Hkv
    qv = q.reshape(Pc, g, Hkv, rep, D)
    sc = D ** -0.5
    acc = torch.promote_types(q.dtype, torch.float32)
    logits, vals = [], []
    for K, V in levels:
        n, _, L, _ = K.shape
        Kx, Vx = K.view(n, 1, Hkv, L, D), V.view(n, 1, Hkv, L, D)       # broadcast over rollouts (and prompts if n=1)
        logits.append(torch.matmul(qv, Kx.transpose(-1, -2)).to(acc) * sc)   # [Pc, g, Hkv, rep, L]
        vals.append(Vx)
    T = ko.shape[2]
    kx, vx = ko.reshape(Pc, g, Hkv, T, D), vo.reshape(Pc, g, Hkv, T, D)
    own = torch.matmul(qv, kx.transpose(-1, -2)).to(acc) * sc
    if ko_add is not None:
        own = own + ko_add                                             # -inf past the current write index
    logits.append(own)
    vals.append(vx)
    w = torch.softmax(torch.cat(logits, -1), -1)
    out, i0 = None, 0
    for lg, V in zip(logits, vals):
        n = lg.shape[-1]
        part = torch.matmul(w[..., i0:i0 + n].to(V.dtype), V)          # [Pc, g, Hkv, rep, D]
        out = part if out is None else out + part
        i0 += n
    return out.reshape(b, Hq, 1, D)


class LocalRL:
    unit = "samples"

    def __init__(self, model: Model, spec: dict | None = None, device="cuda"):
        spec = dict(spec or {})
        self.m = m = model
        c = m.cfg
        self.dev = device
        self.idle = bool(spec.get("idle", False))
        self.n_prompts = int(spec.get("n_prompts", 16))
        self.group = int(spec.get("group", 4))
        self.conc = int(spec.get("conc_prompts", 2))
        assert self.n_prompts % self.conc == 0, (self.n_prompts, self.conc)
        self.P = int(spec.get("prompt", 1024))
        self.G = int(spec.get("gen", 256))
        self.T = self.P + self.G
        self.L0 = int(spec.get("shared_prefix", 0))
        assert 0 <= self.L0 < self.P
        self.seg = max(1, int(spec.get("seg", 8)))
        self.train_mb = max(1, int(spec.get("train_mb", 2)))
        self.mlp_chunk = int(spec.get("mlp_chunk", 0))     # 0 = one-shot MLP; >0 = token-chunked (fits the cut)
        # mlp_slice > 0: slice-outer MLP (SliceMLP), each weight decoded once per pass; overrides
        # mlp_chunk under grad and in no_grad. Value = gate/up rows per slice.
        self.mlp_slice = int(spec.get("mlp_slice", 0))
        # rows at or below which the no-grad MLP runs one-shot even with mlp_slice/mlp_chunk set (decode steps: the
        # [rows, 2I] gate/up output is tiny, and the contiguous gate_up row slices need ~2/3 of SliceMLP's launches)
        self.mlp_small_t = int(spec.get("mlp_small_t", 64))
        # decode-step attention. "sdpa": the prompt KV is repeat_interleave'd per rollout and cat'ed
        # with the rollout's own KV every decode step (as coded before). "shared": attention over the prompt levels by
        # broadcasting (no per-rollout copy) and over the rollout's own KV, merged in one softmax.
        # kv8: the rollouts' own KV stored int8 with a per (token, head) bf16 scale (~half the bytes; the hidden job is
        # not bit-exact), dequantised per layer per step; implies "shared".
        self.kv8 = bool(spec.get("kv8", False))        # default OFF: lossy, outside the declared-precision rule
        self.dec_attn = "shared" if self.kv8 else str(spec.get("dec_attn", "sdpa"))
        # dec_graph: the per-token decode forward is run at a fixed shape (full G-length cache with a position mask) and
        # captured into a CUDA graph (spec cuda_graph), so a decode step is one driver call instead of ~L*k. Needs the
        # shared decode and resident adapters. Sampling and the write index stay eager.
        self.dec_graph = bool(spec.get("dec_graph", False))
        if self.dec_graph:
            self.dec_attn = "shared"
        self.cg = bool(spec.get("cuda_graph", False)) and self.dec_graph and str(device).startswith("cuda")
        self._dec_graph = None
        self._dec_graph_shape = None
        self.g_replays = 0
        self.g_err = None
        self.slice_gate = bool(spec.get("slice_gate", True))     # read by the cheater (hidden_rl.SLICE_GATE)
        # reentrant: reentrant checkpoints in the update. With non-reentrant checkpoints the forward
        # runs under grad, so every streamed adapter leaf stays referenced by the graph (its AccumulateGrad node) until
        # the backward: all 64 layers' adapters live at once (rank 16, all seven targets: ~268 MB; an allocation snapshot
        # named it). Reentrant forwards run under no_grad, so each layer's adapters are freed after the layer, and the
        # backward recompute streams them in again with their grad hooks.
        self.reentrant = bool(spec.get("reentrant", False))
        self.lm_chunk = int(spec.get("lm_chunk", 0))       # 0 = stream LM head over all positions; >0 = token-chunked too
        self.gen_in_flight = int(spec.get("gen_in_flight", 0))   # 0 = all conc*group rollouts at once in the plain path
        self.update_only = bool(spec.get("update_only", False))  # rehearsal: skip sampling, random tokens
        self.host_adam = bool(spec.get("host_adam", True))
        self.vocab_lo, self.vocab_hi = 100, min(c.V, 30000)
        self.dt = m.lins["embed"].t.dtype if m.lins["embed"].key is None else torch.bfloat16
        # adapters: rank r on every target projection of every layer (LoRA Without Regret: all layers, MLP
        # included; the declared RL workload is all seven projections). Default q, v for the round-1 comparison.
        r = int(spec.get("rank", 16))
        self.rank = r
        self.targets = list(spec.get("targets", ["q", "v"]))
        self.tset = set(self.targets)
        self.scale = float(spec.get("lora_scale", 2.0))
        # per-layer adapter streaming: hold no adapters resident; stream each layer's adapters in (<= chunk H2D) just
        # before that layer is used and free them after, so the device adapter residency is ONE layer (~rank*dims/L)
        # rather than all 64 layers (~268 MB at rank 16, which is half the 0.70 GiB budget -- no cheater would hold it).
        # Grads leave per layer the same way (D2H into a host accumulator). Requires a host master (forced on).
        self.stream_adapters = bool(spec.get("stream_adapters", False))
        if self.stream_adapters:
            self.host_adam = True
        self._chunk_bytes = max(1, int(float(spec.get("stream_chunk_kib", 1024)) * 1024))   # <= 1 MiB inbound chunks
        # phase split: with stream_adapters, keep ALL adapters resident (read-only cache,
        # no grad) during generation and stream them per segment only in the update; the cache is rebuilt from the
        # host's current bf16 adapters before each generation (on-policy) and freed after it. Inbound adapter copies
        # are paced to adapter_gbs (chunk by chunk) so they stay inside the theft the floor passes.
        self.phase_split = bool(spec.get("phase_split", False)) and self.stream_adapters
        self.adapter_gbs = float(spec.get("adapter_gbs", 0.4))
        self._gen_cache = None
        self._last_chunk_t = 0.0
        self.cache_in_bytes = 0
        self.t_adapter_in = 0.0
        self.stream_in_bytes = 0
        self.grad_out_bytes = 0
        self._dev_ad = {}
        dims = {"q": (c.H, c.Hq * c.D), "k": (c.H, c.Hkv * c.D), "v": (c.H, c.Hkv * c.D), "o": (c.Hq * c.D, c.H),
                "gate": (c.H, c.I), "up": (c.H, c.I), "down": (c.I, c.H)}
        gen = torch.Generator(device="cpu").manual_seed(0)
        self.params, self.ad = [], []
        for _ in range(c.L):
            d = {}
            for t in self.targets:
                i, o = dims[t]
                A = (torch.randn(r, i, generator=gen) / i ** 0.5).to(device, self.dt).requires_grad_(True)
                B = torch.zeros(o, r, dtype=self.dt, device=device).requires_grad_(True)
                d[t] = (A, B)
                self.params += [A, B]
            self.ad.append(d)
        m.lora = None                          # adapters are applied here (all projections), not by Model._qkv
        self.lora_params = sum(p.numel() for p in self.params)
        lr = float(spec.get("lr", 1e-4))
        self.flush = None
        if self.host_adam:
            # gradients leave per micro-batch (autograd hook, free outbound), fp32 master + AdamW on the host, new
            # bf16 adapters come back once per optimiser step at the stolen inbound rate
            from schemes.memory_link_exhaustion.attacks.gated_shard.workload import Pacer
            from schemes.memory_link_exhaustion.attacks.max_shard.workload import GradFlusher
            pin = torch.cuda.is_available() and str(device).startswith("cuda")
            self.master = [p.detach().float().cpu().clone().requires_grad_(True) for p in self.params]
            self.hbf = [torch.empty(p.shape, dtype=self.dt, pin_memory=pin) for p in self.params]
            self.opt = torch.optim.AdamW(self.master, lr=lr)
            self.flush = GradFlusher(self.params, 0.0, max_pending=int(float(spec.get("max_pending_mib", 32)) * (1 << 20)))
            self.pacer = Pacer(float(spec.get("link_gbs", 0.01)), 2 << 20, device if pin else "cpu")
        else:
            self.opt = torch.optim.AdamW(self.params, lr=lr)
        if self.stream_adapters:
            # master (fp32) and the bf16 stream-in source are already built above; add a host fp32 grad accumulator per
            # adapter and an (l, t, which) -> flat-index map (same order as self.params was built). Then DROP the
            # resident device adapters -- they are the floor we are removing -- and the per-micro-batch GradFlusher,
            # whose job the per-layer grad hooks now do. hbf holds the current adapter bytes that each layer streams in.
            self.sacc = [torch.zeros(p.shape, dtype=torch.float32) for p in self.master]
            self._idx, k = {}, 0
            for l in range(c.L):
                for t in self.targets:
                    self._idx[(l, t, 0)], self._idx[(l, t, 1)] = k, k + 1
                    k += 2
            if self.flush is not None:
                self.flush.stop()
            self.flush = None
            if getattr(self, "pacer", None) is not None:
                self.pacer.stop()
            self.pacer = None
            _a0 = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            for p_ in self.params:              # release the storage even if a stopped flusher thread or a hook still
                p_.data = torch.empty(0, dtype=p_.dtype, device=p_.device)   # holds the Parameter
            self.ad = None                      # free the resident device adapters (self.params held the same tensors)
            self.params = []
            self._dev_ad = {}
            _sync(device)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.empty_cache()
                import sys as _s
                print(f"{time.strftime('%H:%M:%S')} LRL_PROGRESS stream_adapters: device adapters released, allocated "
                      f"{_a0 >> 20} -> {torch.cuda.memory_allocated() >> 20} MiB", file=_s.stderr, flush=True)
                print(torch.cuda.memory_summary(abbreviated=True), file=_s.stderr, flush=True)
        gp = torch.Generator(device="cpu").manual_seed(1)
        self.prefix_ids = torch.randint(self.vocab_lo, self.vocab_hi, (self.L0,), generator=gp).to(device)
        self._kv0 = None
        self._kv0_step = -1
        self.no_release = bool(spec.get("no_release", False))
        # rest: after each decode step and each update micro-batch, leave the device idle for rest x the time that
        # work took (0 = off). A duty cycle: rest 1.0 halves the job's share of device time.
        self.rest = float(spec.get("rest", 0.0))   # never hand cached blocks back to the device
        # pace_loop: closed-loop pacing instead of a fixed duty (attacks/local_rl/pace.py). The cheater prover sets
        # self.loop_pacer once its noise stream exists; the target comes from VARENA_PACE_TARGET_GBS.
        self.pace_loop = bool(spec.get("pace_loop", False))
        self.loop_pacer = None
        self.calls = self.opt_steps = 0
        self.link_bytes = 0
        self.t_phase = {"prefill": 0.0, "gen": 0.0, "train": 0.0, "opt": 0.0}
        self.t_steps = []
        # optimiser step every n_prompts / conc_prompts calls. opt_every_calls overrides the cadence for measurement
        # only (1 = after every call): the step's own time is then seen in a one-call session, and a whole step is
        # modelled as calls_per_step x call time + opt time
        self.calls_per_step = self.n_prompts // self.conc
        self.opt_every = int(spec.get("opt_every_calls", 0)) or self.calls_per_step
        self.t_opt = []
        # staggered continuous batching (spec stagger): prompt groups are admitted one at a time as the KV budget
        # allows, so rollouts of different ages share every decode step (and its weight decode) and the live KV is
        # about half of the plain layout's for the same rows; the update runs after the optimiser step's 16 groups
        self.stagger = bool(spec.get("stagger", False))
        self.kv_budget = float(spec.get("kv_budget_mib", 1500)) * 2**20
        self.max_groups = int(spec.get("max_groups", 8))
        self.chunk = int(spec.get("kv_chunk", 64))
        self.debug_logits = False
        assert not (self.stagger and self.L0), "stagger: shared_prefix not supported"
        self._cyc = None
        self.cycles = []
        self.steps = 0

    # ------------------------------------------------------------------ helpers
    def describe(self):
        c = self.m.cfg
        kv_tok = 2 * c.L * c.Hkv * c.D * 2
        kv = kv_tok * (self.L0 + self.conc * (self.P - self.L0) + self.conc * self.group * self.G)
        cuda = torch.cuda.is_available()
        return {"n_prompts": self.n_prompts, "group": self.group, "conc_prompts": self.conc,
                "rollouts_in_flight": self.conc * self.group, "prompt": self.P, "gen": self.G,
                "shared_prefix": self.L0, "seg": self.seg, "train_mb": self.train_mb, "host_adam": self.host_adam,
                "kv_mib": round(kv / 2**20, 1), "opt_steps": self.opt_steps, "calls": self.calls,
                "link_mb": round(self.link_bytes / 1e6, 1), "idle": self.idle, "rank": self.rank, "targets": self.targets,
                "lora_params": self.lora_params,
                "link_gbs": self.pacer.gbs if self.flush else None,
                "stagger": self.stagger, "cycles": self.cycles[-3:],
                "cycle_samples_s": round(sum(x["samples"] for x in self.cycles) / sum(x["s"] for x in self.cycles), 4)
                if self.cycles else None,
                "t_phase": {k: round(v, 1) for k, v in self.t_phase.items()},
                "pace": self.loop_pacer.stats() if self.loop_pacer is not None else None,
                "step_s": [round(x, 2) for x in self.t_steps[-5:]], "opt_s": [round(x, 2) for x in self.t_opt[-5:]],
                "calls_per_step": self.calls_per_step, "opt_every_calls": self.opt_every, "gate_waits": G.STATS["waits"],
                "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if cuda else None,
                "mlp_chunk": self.mlp_chunk, "gen_in_flight": self.gen_in_flight, "fit": getattr(self, "fit", None),
                # storage width of the rollouts' own KV (16 = declared bf16; 8 = kv8, lossy, OUTSIDE the declared-precision
                # rule) and the decode attention path; mlp_slice / reentrant / slice_gate for the table
                "kv_bits": 8 if self.kv8 else 16, "dec_attn": self.dec_attn, "mlp_slice": self.mlp_slice,
                "reentrant": self.reentrant, "mlp_small_t": self.mlp_small_t,
                "dec_graph": {"on": self.dec_graph, "captured": self._dec_graph is not None, "replays": self.g_replays,
                              "error": self.g_err} if self.dec_graph else None,
                "stream_adapters": self.stream_adapters, "phase_split": self.phase_split,
                "adapter_gbs": self.adapter_gbs if self.phase_split else None,
                "cache_in_gb": round(self.cache_in_bytes / 1e9, 3) if self.phase_split else None,
                "t_adapter_in_s": round(self.t_adapter_in, 1) if self.stream_adapters else None,
                "stream_in_gb": round(self.stream_in_bytes / 1e9, 3) if self.stream_adapters else None,
                "grad_out_gb": round(self.grad_out_bytes / 1e9, 3) if self.stream_adapters else
                (round(self.flush.moved / 1e9, 3) if self.flush else None)}

    def _cs(self, pos):
        """cos/sin for positions: pos int tensor [n] -> [1, n, 1, D]."""
        return self.m.cos[pos].view(1, -1, 1, self.m.cfg.D), self.m.sin[pos].view(1, -1, 1, self.m.cfg.D)

    def _lo(self, l, t, x):
        A, B = self._adapter(l, t)
        return torch.mm(torch.mm(x, A.t()), B.t()) * self.scale

    # ------------------------------------------------------------------ per-layer adapter streaming
    def _adapter(self, l, t):
        """The (A, B) for layer l target t. Resident when not streaming; otherwise the device copy streamed in for
        this layer (shared across its q/k/v/o/gate/up/down uses, built once per layer visit)."""
        if not self.stream_adapters:
            return self.ad[l][t]
        if self._gen_cache is not None:
            return self._gen_cache[l][t]
        cur = self._dev_ad.get(l)
        if cur is None:
            cur = self._stream_layer_in(l)
        return cur[t]

    def _h2d(self, dst, src):
        """Copy a host adapter into device scratch in <= _chunk_bytes slices (inbound over the stolen link)."""
        per = max(1, self._chunk_bytes // dst.element_size())
        fd, fs = dst.view(-1), src.view(-1)
        t0 = time.perf_counter()
        for i in range(0, fd.numel(), per):
            if self.phase_split and self.adapter_gbs > 0 and str(self.dev).startswith("cuda"):
                gap = per * dst.element_size() / (self.adapter_gbs * 1e9)
                w = self._last_chunk_t + gap - time.perf_counter()
                if w > 0:
                    time.sleep(w)
                self._last_chunk_t = time.perf_counter()
            fd[i:i + per].copy_(fs[i:i + per])
        self.t_adapter_in += time.perf_counter() - t0
        b = dst.numel() * dst.element_size()
        self.link_bytes += b
        self.stream_in_bytes += b

    def _grad_out(self, k, g, l):
        """autograd hook: send a streamed adapter's grad out (D2H, chunked) and add it into the host accumulator, then
        drop layer l's device reference. The grad is accumulated in fp32 (exact); the bytes are counted at bf16, which
        is what a cheater actually ships. Dropping the ref here (not after _mlp) is what frees the streamed adapters in
        the UPDATE backward: non-reentrant checkpoint early-stops the recompute by exception before _finish's release
        runs, but hooks always fire, and by the time one does this layer's graph no longer needs the leaf."""
        gc = g.detach().to(torch.float32).cpu()
        self.sacc[k].add_(gc)
        b = g.numel() * 2
        self.link_bytes += b
        self.grad_out_bytes += b
        self._dev_ad.pop(l, None)
        return None

    def _stream_layer_in(self, l):
        """Stream layer l's adapters onto the device. Under grad they are leaves with a grad hook (freed by the graph
        after this layer's local backward); in no_grad they are plain tensors (freed when _dev_ad[l] is released)."""
        grad = torch.is_grad_enabled()
        d = {}
        for t in self.targets:
            kA, kB = self._idx[(l, t, 0)], self._idx[(l, t, 1)]
            A = torch.empty(self.hbf[kA].shape, dtype=self.dt, device=self.dev)
            B = torch.empty(self.hbf[kB].shape, dtype=self.dt, device=self.dev)
            self._h2d(A, self.hbf[kA])
            self._h2d(B, self.hbf[kB])
            if grad:
                A.requires_grad_(True)
                B.requires_grad_(True)
                A.register_hook(lambda g, k=kA, l=l: self._grad_out(k, g, l))
                B.register_hook(lambda g, k=kB, l=l: self._grad_out(k, g, l))
            d[t] = (A, B)
        self._dev_ad[l] = d
        return d

    @torch.no_grad()
    def _build_gen_cache(self):
        """Phase split: all layers' current adapters resident (plain tensors) for generation."""
        c = self.m.cfg
        cache = []
        b0 = self.stream_in_bytes
        for l in range(c.L):
            d = {}
            for t in self.targets:
                kA, kB = self._idx[(l, t, 0)], self._idx[(l, t, 1)]
                A = torch.empty(self.hbf[kA].shape, dtype=self.dt, device=self.dev)
                B = torch.empty(self.hbf[kB].shape, dtype=self.dt, device=self.dev)
                self._h2d(A, self.hbf[kA])
                self._h2d(B, self.hbf[kB])
                d[t] = (A, B)
            cache.append(d)
        self.cache_in_bytes += self.stream_in_bytes - b0
        self._gen_cache = cache

    def _drop_gen_cache(self):
        self._gen_cache = None
        if str(self.dev).startswith("cuda"):
            _sync(self.dev)
            # no_release (the careful cheater): keep the freed blocks in torch's cache. Handing them back to the
            # device and asking for them again stalls every stream of the process, the noise copies included.
            if not getattr(self, "no_release", False):
                torch.cuda.empty_cache()

    def _release_layer(self, l):
        """Drop the reference to layer l's streamed adapters. In no_grad this frees them; under grad the local graph
        still holds them until this layer's backward (and its hooks) run, after which they are freed."""
        if self.stream_adapters:
            self._dev_ad.pop(l, None)

    def _qkv(self, l, x):
        m, c = self.m, self.m.cfg
        y = m.lins[f"l{l}.qkv"](x)
        if m.qkv_b[l] is not None:
            y = y + m.qkv_b[l]
        ad = self.tset
        if "q" in ad or "k" in ad or "v" in ad:
            parts = []
            for t, w in (("q", c.Hq * c.D), ("k", c.Hkv * c.D), ("v", c.Hkv * c.D)):
                parts.append(self._lo(l, t, x) if t in ad else torch.zeros(x.shape[0], w, dtype=y.dtype, device=y.device))
            y = y + torch.cat(parts, -1)
        return y

    def _mlp_core(self, l, h):
        m, c = self.m, self.m.cfg
        ad = self.tset
        x = rms(h, m.small[f"model.layers.{l}.post_attention_layernorm.weight"], c.eps)
        gu = m.lins[f"l{l}.gate_up"](x)
        g, u = gu[:, : c.I], gu[:, c.I:]
        if "gate" in ad:
            g = g + self._lo(l, "gate", x)
        if "up" in ad:
            u = u + self._lo(l, "up", x)
        a = F.silu(g) * u
        y = m.lins[f"l{l}.down"](a)
        if "down" in ad:
            y = y + self._lo(l, "down", a)        # the down adapter's input is the I-wide activation
        return h + y

    def _mlp(self, l, h):
        """MLP over rows [b*n, H]. With mlp_chunk set, process the rows in token chunks so the I-wide intermediates
        (gate/up output, silu, the down input) are at most [mlp_chunk, I] instead of [b*n, I], which is what OOMs a
        512-token update under the memory cut. Under grad each chunk is checkpointed so its intermediates do not
        persist into the segment's backward; in no_grad (generation, the outer checkpoint's forward) the chunks are
        just computed and freed in turn. The result is bit-identical to the one-shot MLP."""
        if not torch.is_grad_enabled() and h.shape[0] <= self.mlp_small_t:
            return self._mlp_core(l, h)
        if self.mlp_slice:
            m, c = self.m, self.m.cfg
            x = rms(h, m.small[f"model.layers.{l}.post_attention_layernorm.weight"], c.eps)
            ad = self.tset
            Ag, Bg = self._adapter(l, "gate") if "gate" in ad else (None, None)
            Au, Bu = self._adapter(l, "up") if "up" in ad else (None, None)
            Ad, Bd = self._adapter(l, "down") if "down" in ad else (None, None)
            y = SliceMLP.apply(x, Ag, Bg, Au, Bu, Ad, Bd, m.lins[f"l{l}.gate_up"], m.lins[f"l{l}.down"], c.I,
                               self.scale, self.mlp_slice)
            return h + y
        ck = self.mlp_chunk
        if not ck or h.shape[0] <= ck:
            return self._mlp_core(l, h)
        if torch.is_grad_enabled():
            outs = [ckpt.checkpoint(self._mlp_core, l, h[i:i + ck], use_reentrant=False)
                    for i in range(0, h.shape[0], ck)]
        else:
            outs = [self._mlp_core(l, h[i:i + ck]) for i in range(0, h.shape[0], ck)]
        return torch.cat(outs, 0)

    def _qkv_rope(self, l, h, b, n, cos, sin):
        m, c = self.m, self.m.cfg
        x = rms(h, m.small[f"model.layers.{l}.input_layernorm.weight"], c.eps)
        q, k, v = m._split(self._qkv(l, x), b, n)
        return rope(q, cos, sin), rope(k, cos, sin), v

    def _finish(self, l, h, o, b, n):
        m, c = self.m, self.m.cfg
        a = o.transpose(1, 2).reshape(b * n, c.Hq * c.D)
        h = h + m.lins[f"l{l}.o"](a)
        if "o" in self.tset:
            h = h + self._lo(l, "o", a)
        out = self._mlp(l, h)
        self._release_layer(l)      # o/gate/up/down were this layer's last adapter uses -> free the streamed copies
        return out

    # ------------------------------------------------------------------ generation (no grad)
    @torch.no_grad()
    def _prefill_level(self, ids, pos0, prev):
        """ids [b, n]; prev: list per layer of (K, V) [1 or b, Hkv, Lp, D] visible to every row (or None).
        Returns the last hidden state [b, H] and the new level's per-layer (K, V) [b, Hkv, n, D]."""
        m, c = self.m, self.m.cfg
        b, n = ids.shape
        cos, sin = self._cs(torch.arange(pos0, pos0 + n, device=ids.device))
        h = m.embed(ids).reshape(b * n, c.H)
        kvs = []
        for l in range(c.L):
            G.wait()
            q, k, v = self._qkv_rope(l, h, b, n, cos, sin)
            k, v = k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
            kvs.append((k, v))
            qt = q.transpose(1, 2)
            if prev is None:
                o = F.scaled_dot_product_attention(qt, k, v, is_causal=True, enable_gqa=True)
            else:
                Kp, Vp = prev[l]
                Lp = Kp.shape[2]
                K = torch.cat([Kp.expand(b, -1, -1, -1), k], 2)
                V = torch.cat([Vp.expand(b, -1, -1, -1), v], 2)
                mask = torch.ones(n, Lp + n, dtype=torch.bool, device=ids.device)
                mask[:, Lp:] = torch.ones(n, n, dtype=torch.bool, device=ids.device).tril()
                o = F.scaled_dot_product_attention(qt, K, V, attn_mask=mask, enable_gqa=True)
            h = self._finish(l, h, o, b, n)
        return h.view(b, n, c.H)[:, -1], kvs

    def _sample(self, logits):
        return torch.multinomial(torch.softmax(logits.float(), -1), 1)[:, 0]

    def _ensure_kv0(self):
        if self.L0 and (self._kv0 is None or self._kv0_step != self.opt_steps):
            self._kv0 = None
            _, self._kv0 = self._prefill_level(self.prefix_ids.view(1, -1), 0, None)
            self._kv0_step = self.opt_steps

    @torch.no_grad()
    def _generate_sub(self, ids1, s, force=None, want_logits=False):
        """Sub-group generation: each prompt's `group` rollouts are generated s at a time
        (s < group). The prompt KV is computed once per prompt and kept for its group; each sub-batch's generated KV is
        released before the next. Rollouts are independent given the prompt, so the tokens (and, teacher-forced, the
        logits) equal the grouped path's; rewards and advantages are still formed per group after all its rollouts,
        so the update is unchanged and on-policy with the current adapters. Same outputs/order as _generate."""
        Pc, g = ids1.shape[0], self.group
        self._ensure_kv0()
        outs, lgs = [], []
        for i in range(Pc):
            pre = self._prefill_level(ids1[i:i + 1], self.L0, self._kv0 if self.L0 else None)
            for j in range(0, g, s):
                b = min(s, g - j)
                f = force[i * g + j:i * g + j + b] if force is not None else None
                _ts = time.perf_counter()
                r = self._generate(ids1[i:i + 1], force=f, want_logits=want_logits, g=b, pre=pre)
                if force is None and str(self.dev).startswith("cuda"):
                    _sync(self.dev)
                    _dt = time.perf_counter() - _ts
                    self._progress(f"sub-batch prompt {i} rollouts {j}-{j + b - 1}: {self.G} tokens in {_dt:.1f} s "
                                   f"({_dt / max(1, self.G - 1) * 1e3:.0f} ms per decode step)")
                if want_logits:
                    outs.append(r[0]); lgs.append(r[1])
                else:
                    outs.append(r)
                del r
            del pre
        out = torch.cat(outs, 0)
        if not want_logits:
            return out
        return out, [torch.cat([lg[t] for lg in lgs], 0) for t in range(len(lgs[0]))]

    @torch.no_grad()
    def _decode_fwd(self, tok_buf, pos_buf, idx_buf, kv0, kv1, K2, V2, lg_buf, b, g_rollouts, Pc):
        """One decode step at a FIXED shape (graph-capturable): read tok_buf [b,1] at position pos_buf [1], write its
        K/V into slot idx_buf [1] of the full G-length cache, attend over the whole cache with a mask that zeroes the
        not-yet-written slots, and store this step's logits into lg_buf [b,V] in place. No Python-int shapes, no
        growing tensors: every tensor is the same size each call, so the graph replays. Sampling and the index
        increment stay outside (a handful of tiny launches a step, against ~L*k inside)."""
        m, c = self.m, self.m.cfg
        cos, sin = self._cs(pos_buf)
        ar = self._dec_ar                                              # arange(G), built once
        ko_add = torch.where(ar <= idx_buf, 0.0, float("-inf")).view(1, 1, 1, 1, self.G).to(torch.float32)
        h = m.embed(tok_buf.view(b, 1)).reshape(b, c.H)
        for l in range(c.L):
            G.wait()                                                   # yield to a challenge, as the eager path does
            q, k, v = self._qkv_rope(l, h, b, 1, cos, sin)
            K2[l].index_copy_(2, idx_buf, k[:, 0].unsqueeze(2))
            V2[l].index_copy_(2, idx_buf, v[:, 0].unsqueeze(2))
            levels = ([kv0[l]] if self.L0 else []) + [kv1[l]]
            o = _dec_attn(q.transpose(1, 2), levels, K2[l], V2[l], Pc, g_rollouts, ko_add=ko_add)
            h = self._finish(l, h, o, b, 1)
        lg_buf.copy_(m.logits(h))

    @torch.no_grad()
    def _generate_graph(self, ids1, force=None, want_logits=False, g=None, pre=None):
        """Fixed-shape generation whose per-token forward is captured into a CUDA graph and replayed (spec dec_graph).
        Same tokens/logits as _generate (the decode is deterministic given the sampled token, which is drawn outside
        the graph from the same logits). Requires the shared decode and resident adapters."""
        assert self.dec_attn == "shared" and not self.kv8 and not self.stream_adapters, \
            "graph decode: shared dec_attn, 16-bit KV, resident adapters"
        m, c = self.m, self.m.cfg
        Pc, dev = ids1.shape[0], ids1.device
        g = self.group if g is None else g
        b = Pc * g
        self._ensure_kv0()
        h_last, kv1 = pre if pre is not None else self._prefill_level(ids1, self.L0, self._kv0 if self.L0 else None)
        dt_kv = kv1[0][0].dtype
        if not hasattr(self, "_dec_ar"):
            self._dec_ar = torch.arange(self.G, device=dev).view(1, 1, 1, 1, self.G)
        K2 = [torch.zeros(b, c.Hkv, self.G, c.D, dtype=dt_kv, device=dev) for _ in range(c.L)]
        V2 = [torch.zeros_like(K2[0]) for _ in range(c.L)]
        tok_buf = torch.zeros(b, 1, dtype=torch.long, device=dev)
        pos_buf = torch.zeros(1, dtype=torch.long, device=dev)
        idx_buf = torch.zeros(1, dtype=torch.long, device=dev)
        lg0 = m.logits(h_last).repeat_interleave(g, 0)                 # [b, V], the logits' dtype/device
        lg_buf = torch.zeros_like(lg0)
        out = torch.empty(b, self.G, dtype=torch.long, device=dev)
        logits = [lg0] if want_logits else None
        tok = force[:, 0] if force is not None else self._sample(lg0)
        out[:, 0] = tok
        kv0r = self._kv0 if self.L0 else None
        cg = self._dec_graph if getattr(self, "_dec_graph", None) is not None else None
        warm = 0
        for t in range(self.G - 1):
            _tr = time.perf_counter()
            tok_buf[:, 0].copy_(tok)
            pos_buf.fill_(self.P + t)
            idx_buf.fill_(t)
            if self.cg and cg is not None and self._dec_graph_shape == (b, Pc):
                cg.replay()                                            # captured step
                self.g_replays += 1
            elif self.cg and warm >= 2 and self._dec_graph is None:
                try:
                    self._decode_fwd(tok_buf, pos_buf, idx_buf, kv0r, kv1, K2, V2, lg_buf, b, g, Pc)  # warm the pools
                    gph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gph):
                        self._decode_fwd(tok_buf, pos_buf, idx_buf, kv0r, kv1, K2, V2, lg_buf, b, g, Pc)
                    self._dec_graph, cg, self._dec_graph_shape = gph, gph, (b, Pc)
                    cg.replay()
                except Exception as e:
                    self.cg, self.g_err = False, repr(e)[:300]
                    self._decode_fwd(tok_buf, pos_buf, idx_buf, kv0r, kv1, K2, V2, lg_buf, b, g, Pc)
            else:
                self._decode_fwd(tok_buf, pos_buf, idx_buf, kv0r, kv1, K2, V2, lg_buf, b, g, Pc)
                warm += 1
            if want_logits:
                logits.append(lg_buf.clone())
            tok = force[:, t + 1] if force is not None else self._sample(lg_buf)
            out[:, t + 1] = tok
            self._rest(_tr)
        del K2, V2, kv1
        return (out, logits) if want_logits else out

    @torch.no_grad()
    def _generate(self, ids1, force=None, want_logits=False, g=None, pre=None):
        if self.dec_graph:
            return self._generate_graph(ids1, force=force, want_logits=want_logits, g=g, pre=pre)
        return self._generate_eager(ids1, force=force, want_logits=want_logits, g=g, pre=pre)

    @torch.no_grad()
    def _generate_eager(self, ids1, force=None, want_logits=False, g=None, pre=None):
        """ids1 [Pc, P - L0] prompt tokens after the shared prefix; group rollouts each. Returns tokens [Pc*g, G]
        (and the per-step logits when want_logits). force [Pc*g, G] teacher-forces the tokens (tests).
        g: rollouts per prompt (default self.group); pre: (h_last, kv1) of the prompt level, precomputed."""
        m, c = self.m, self.m.cfg
        Pc, dev = ids1.shape[0], ids1.device
        g = self.group if g is None else g
        b = Pc * g
        self._ensure_kv0()
        h_last, kv1 = pre if pre is not None else self._prefill_level(ids1, self.L0, self._kv0 if self.L0 else None)
        lg = m.logits(h_last)                                   # [Pc, V]
        lg = lg.repeat_interleave(g, 0)
        logits = [lg] if want_logits else None
        tok = force[:, 0] if force is not None else self._sample(lg)
        out = torch.empty(b, self.G, dtype=torch.long, device=dev)
        out[:, 0] = tok
        dt_kv = kv1[0][0].dtype
        if self.kv8:
            K2 = [torch.empty(b, c.Hkv, self.G, c.D, dtype=torch.int8, device=dev) for _ in range(c.L)]
            V2 = [torch.empty_like(K2[0]) for _ in range(c.L)]
            K2s = [torch.empty(b, c.Hkv, self.G, 1, dtype=dt_kv, device=dev) for _ in range(c.L)]
            V2s = [torch.empty_like(K2s[0]) for _ in range(c.L)]
        else:
            K2 = [torch.empty(b, c.Hkv, self.G, c.D, dtype=dt_kv, device=dev) for _ in range(c.L)]
            V2 = [torch.empty_like(K2[0]) for _ in range(c.L)]
        for t in range(self.G - 1):
            _tr = time.perf_counter()
            p = self.P + t
            cos, sin = self._cs(torch.tensor([p], device=dev))
            h = m.embed(tok.view(b, 1)).reshape(b, c.H)
            for l in range(c.L):
                G.wait()
                q, k, v = self._qkv_rope(l, h, b, 1, cos, sin)
                if self.dec_attn == "shared":
                    if self.kv8:
                        K2[l][:, :, t], K2s[l][:, :, t] = _q8(k[:, 0])
                        V2[l][:, :, t], V2s[l][:, :, t] = _q8(v[:, 0])
                        ko = K2[l][:, :, :t + 1].to(dt_kv) * K2s[l][:, :, :t + 1]
                        vo = V2[l][:, :, :t + 1].to(dt_kv) * V2s[l][:, :, :t + 1]
                    else:
                        K2[l][:, :, t] = k[:, 0]
                        V2[l][:, :, t] = v[:, 0]
                        ko, vo = K2[l][:, :, :t + 1], V2[l][:, :, :t + 1]
                    levels = ([self._kv0[l]] if self.L0 else []) + [kv1[l]]
                    o = _dec_attn(q.transpose(1, 2), levels, ko, vo, Pc, g)
                    h = self._finish(l, h, o, b, 1)
                    continue
                K2[l][:, :, t] = k[:, 0]
                V2[l][:, :, t] = v[:, 0]
                parts_k = [kv1[l][0].repeat_interleave(g, 0), K2[l][:, :, :t + 1]]
                parts_v = [kv1[l][1].repeat_interleave(g, 0), V2[l][:, :, :t + 1]]
                if self.L0:
                    parts_k.insert(0, self._kv0[l][0].expand(b, -1, -1, -1))
                    parts_v.insert(0, self._kv0[l][1].expand(b, -1, -1, -1))
                o = F.scaled_dot_product_attention(q.transpose(1, 2), torch.cat(parts_k, 2), torch.cat(parts_v, 2),
                                                   enable_gqa=True)
                h = self._finish(l, h, o, b, 1)
            lg = m.logits(h)
            if want_logits:
                logits.append(lg)
            tok = force[:, t + 1] if force is not None else self._sample(lg)
            out[:, t + 1] = tok
            self._rest(_tr)
        del K2, V2, kv1
        return (out, logits) if want_logits else out

    # ------------------------------------------------------------------ update
    def _layer(self, h, l, b, n):
        c = self.m.cfg
        G.wait()
        cos, sin = self.m.cos[:n].view(1, n, 1, c.D), self.m.sin[:n].view(1, n, 1, c.D)
        q, k, v = self._qkv_rope(l, h, b, n, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True, enable_gqa=True)
        return self._finish(l, h, o, b, n)

    def _segment(self, h, l0, l1, b, n):
        for l in range(l0, l1):
            h = ckpt.checkpoint(self._layer, h, l, b, n, use_reentrant=self.reentrant)
        return h

    def forward_train(self, seq, checkpoint=True):
        """Hidden states [b, n, H] of seq [b, n] with LoRA autograd; two-level checkpointing (seg layers per segment)."""
        m, c = self.m, self.m.cfg
        b, n = seq.shape
        with torch.no_grad():
            h = m.embed(seq).reshape(b * n, c.H)
        re = bool(self.reentrant and checkpoint)
        if re and torch.is_grad_enabled():
            h.requires_grad_(True)              # a reentrant checkpoint needs a grad-requiring input to backprop at all
        for l0 in range(0, c.L, self.seg):
            l1 = min(c.L, l0 + self.seg)
            if not checkpoint:
                for l in range(l0, l1):
                    h = self._layer(h, l, b, n)
            elif self.seg == 1:
                h = ckpt.checkpoint(self._layer, h, l0, b, n, use_reentrant=re)
            else:
                h = ckpt.checkpoint(self._segment, h, l0, l1, b, n, use_reentrant=re)
        return h.view(b, n, c.H)

    def logprobs(self, seq, checkpoint=True):
        """Per-sample mean log-prob of the generated tokens seq[:, P:] under the policy. The streamed LM head already
        avoids a 152k-wide logits tensor; lm_chunk additionally splits the positions so the [N, vocab-slice] scratch
        is [lm_chunk, slice] (each position's log-prob is independent, so this is bit-identical)."""
        m, c = self.m, self.m.cfg
        h = self.forward_train(seq[:, :-1], checkpoint)
        hs = h[:, self.P - 1:].reshape(-1, c.H)
        x = rms(hs, m.small["model.norm.weight"], c.eps)
        tgt = seq[:, self.P:].reshape(-1)
        lc = self.lm_chunk
        if lc and x.shape[0] > lc:
            lp = torch.cat([StreamedLogProb.apply(x[i:i + lc], tgt[i:i + lc], m.lm) for i in range(0, x.shape[0], lc)])
        else:
            lp = StreamedLogProb.apply(x, tgt, m.lm)
        return lp.view(seq.shape[0], -1).mean(1)

    def _opt_step(self):
        t0 = time.perf_counter()
        if self.stream_adapters:
            # grads already arrived per layer during the backward passes (host accumulator); AdamW on the host, then
            # the new bf16 adapters land in hbf for the next step's per-layer stream-in (no resident weights to refresh)
            for ms, a in zip(self.master, self.sacc):
                ms.grad = a
            self.opt.step()
            for hb, ms, a in zip(self.hbf, self.master, self.sacc):
                hb.copy_(ms.detach())
                a.zero_()
            self.opt_steps += 1
            self.t_phase["opt"] += time.perf_counter() - t0
            return
        if self.host_adam:
            self.flush.drain()
            for ms, a in zip(self.master, self.flush.acc):
                ms.grad = a
            self.opt.step()
            for a in self.flush.acc:
                a.zero_()
            _sync(self.dev)
            for p, hb, ms in zip(self.params, self.hbf, self.master):
                hb.copy_(ms.detach())
                self.pacer.copy_tensor("h2d", p.data, hb)
                self.link_bytes += p.numel() * p.element_size()
        else:
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        self.opt_steps += 1
        self.t_phase["opt"] += time.perf_counter() - t0

    def step(self):
        if self.idle:
            time.sleep(0.5)
            return 0
        if self.stagger:
            return self._stagger_step()
        t0 = time.perf_counter()
        dev, Pc, g = self.dev, self.conc, self.group
        ids1 = torch.randint(self.vocab_lo, self.vocab_hi, (Pc, self.P - self.L0), device=dev)
        pstep = max(1, self.gen_in_flight // g) if self.gen_in_flight else Pc   # prompts whose rollouts generate together
        if self.update_only:
            # rehearsal of the update phase alone: tokens of the right shape instead of sampling
            out = torch.randint(self.vocab_lo, self.vocab_hi, (Pc * g, self.G), device=dev)
        elif self.phase_split:
            self._build_gen_cache()
        if self.update_only:
            pass
        elif self.gen_in_flight and self.gen_in_flight < g:       # below one group: sub-group generation
            out = self._generate_sub(ids1, self.gen_in_flight)
        elif pstep >= Pc:
            out = self._generate(ids1)
        else:                                               # bound in-flight KV: generate prompt sub-batches in turn
            out = torch.cat([self._generate(ids1[i:i + pstep]) for i in range(0, Pc, pstep)], 0)
        if self.phase_split and not self.update_only:
            self._drop_gen_cache()
        _sync(dev)
        t1 = time.perf_counter()
        self._progress(f"generation {Pc * g} rollouts x {self.G} tokens in {t1 - t0:.1f} s"
                       f"{' (update-only: no generation)' if self.update_only else ''}")
        pre = torch.cat([self.prefix_ids.view(1, -1).expand(Pc, -1), ids1], 1) if self.L0 else ids1
        seq = torch.cat([pre.repeat_interleave(g, 0), out], 1)
        rew = torch.rand(Pc, g)                                  # rate measurement: rewards are random
        adv = ((rew - rew.mean(1, keepdim=True)) / (rew.std(1, keepdim=True) + 1e-4)).view(-1).to(dev)
        R = self.n_prompts * g
        for j in range(0, Pc * g, self.train_mb):
            _tr = time.perf_counter()
            lp = self.logprobs(seq[j:j + self.train_mb])
            (-(adv[j:j + self.train_mb] * lp).sum() / R).backward()
            self._rest(_tr)
        self.calls += 1
        _sync(dev)
        t2 = time.perf_counter()
        if self.calls % self.opt_every == 0:
            self._opt_step()                            # timed apart from the call (opt_s): its cost does not depend on
            _sync(dev)                                  # how many calls fed the accumulated gradient
            self.t_opt.append(time.perf_counter() - t2)
        self.t_phase["gen"] += t1 - t0
        self.t_phase["train"] += t2 - t1
        self.t_steps.append(t2 - t0)
        self.steps += 1
        self._progress(f"update {Pc * g} samples in {t2 - t1:.1f} s; step {self.steps} done in {t2 - t0:.1f} s; "
                       f"opt_steps {self.opt_steps} peak {torch.cuda.max_memory_allocated() / 2**30:.3f} GiB"
                       if torch.cuda.is_available() else f"update in {t2 - t1:.1f} s")
        return Pc * g

    def _rest(self, t_start):
        """Duty cycle: wait for the device to finish the work begun at t_start, then idle rest x its duration."""
        if self.loop_pacer is not None:
            _sync(self.dev)
            self.loop_pacer.wait()
        elif self.rest > 0:
            _sync(self.dev)
            time.sleep(self.rest * (time.perf_counter() - t_start))

    def _progress(self, msg):
        """LRL_PROGRESS lines: phase times inside a step, since t_phase only lands at step end."""
        import sys as _s
        try:      # allocator calls into the device so far (they must stay flat after ready for the careful cheater)
            ms = torch.cuda.memory_stats()
            msg = (f"{msg} | device alloc {ms.get('num_device_alloc', 0)} free {ms.get('num_device_free', 0)} "
                   f"retries {ms.get('num_alloc_retries', 0)}")
        except Exception:
            pass
        print(f"{time.strftime('%H:%M:%S')} LRL_PROGRESS {msg}", file=_s.stderr, flush=True)

    # ------------------------------------------------------------------ staggered continuous batching
    def _kv_tok(self):
        c = self.m.cfg
        return 2 * c.L * c.Hkv * c.D * 2

    def _fits(self, groups, extra=True):
        """Projected peak KV bytes over the remaining life of the active groups (+ one new group) <= budget."""
        P, G, g, ch = self.P, self.G, self.group, self.chunk
        ages = [gr["age"] for gr in groups] + ([0] if extra else [])
        peak = 0
        for tau in range(0, G + 1, max(1, G // 64)):
            live = 0
            for a in ages:
                t = a + tau
                if t < G:
                    live += P + g * (-(-max(t, 1) // ch) * ch)
            peak = max(peak, live)
        return peak * self._kv_tok() <= self.kv_budget

    @torch.no_grad()
    def _admit(self, ids1):
        m, c, g = self.m, self.m.cfg, self.group
        h_last, kv1 = self._prefill_level(ids1.view(1, -1), 0, None)
        lg = m.logits(h_last).repeat_interleave(g, 0)
        tok = self._sample(lg)
        gr = {"ids": ids1, "age": 1, "tok": tok, "out": [tok], "kv1": kv1,
              "K": [[] for _ in range(c.L)], "V": [[] for _ in range(c.L)], "lg": [lg] if self.debug_logits else None}
        return gr

    @torch.no_grad()
    def _decode_all(self, groups):
        """One decode step for every row of every active group (ages differ)."""
        m, c, g, ch = self.m, self.m.cfg, self.group, self.chunk
        dev = groups[0]["tok"].device
        b = g * len(groups)
        tok = torch.cat([gr["tok"] for gr in groups])
        pos = torch.cat([torch.full((g,), self.P + gr["age"] - 1, device=dev, dtype=torch.long) for gr in groups])
        cos, sin = m.cos[pos].view(b, 1, 1, c.D), m.sin[pos].view(b, 1, 1, c.D)
        h = m.embed(tok.view(b, 1)).reshape(b, c.H)
        for l in range(c.L):
            G.wait()
            q, k, v = self._qkv_rope(l, h, b, 1, cos, sin)
            outs = []
            for i, gr in enumerate(groups):
                r = slice(i * g, (i + 1) * g)
                t = gr["age"] - 1                         # suffix index written now
                Kc, Vc = gr["K"][l], gr["V"][l]
                if t // ch == len(Kc):
                    Kc.append(torch.empty(g, c.Hkv, ch, c.D, dtype=k.dtype, device=dev))
                    Vc.append(torch.empty_like(Kc[-1]))
                Kc[t // ch][:, :, t % ch] = k[r, 0]
                Vc[t // ch][:, :, t % ch] = v[r, 0]
                K1, V1 = gr["kv1"][l]
                Ks = torch.cat([K1.expand(g, -1, -1, -1)] + Kc, 2)[:, :, :self.P + t + 1]
                Vs = torch.cat([V1.expand(g, -1, -1, -1)] + Vc, 2)[:, :, :self.P + t + 1]
                outs.append(F.scaled_dot_product_attention(q[r].transpose(1, 2), Ks, Vs, enable_gqa=True))
            h = self._finish(l, h, torch.cat(outs, 0), b, 1)
        lg = m.logits(h)
        for i, gr in enumerate(groups):
            r = slice(i * g, (i + 1) * g)
            gr["tok"] = self._sample(lg[r])
            gr["out"].append(gr["tok"])
            if gr["lg"] is not None:
                gr["lg"].append(lg[r])
            gr["age"] += 1

    def _stagger_step(self, decode_steps=16):
        """Advance the current optimiser step: up to decode_steps generation steps (admitting groups as the KV budget
        allows), then one update micro-batch per call; the optimiser step closes the cycle. Returns samples trained."""
        dev, g = self.dev, self.group
        if self._cyc is None:
            self._cyc = {"t0": time.perf_counter(), "queue": [torch.randint(self.vocab_lo, self.vocab_hi, (self.P,), device=dev)
                                                         for _ in range(self.n_prompts)],
                         "active": [], "done": [], "train": None, "peak_rows": 0, "steps": 0}
        cy = self._cyc
        t0 = time.perf_counter()
        if cy["train"] is None:
            for _ in range(decode_steps):
                while cy["queue"] and len(cy["active"]) < self.max_groups and \
                        (not cy["active"] or self._fits(cy["active"])):
                    cy["active"].append(self._admit(cy["queue"].pop(0)))
                if not cy["active"]:
                    break
                live = [gr for gr in cy["active"] if gr["age"] < self.G]
                if live:
                    cy["peak_rows"] = max(cy["peak_rows"], g * len(live))
                    self._decode_all(live)
                    cy["steps"] += 1
                for gr in [gr for gr in cy["active"] if gr["age"] >= self.G]:
                    cy["active"].remove(gr)
                    seq = torch.cat([gr["ids"].view(1, -1).expand(g, -1), torch.stack(gr["out"], 1)], 1)
                    cy["done"].append((seq, gr["lg"]))
                    del gr["K"], gr["V"], gr["kv1"]
                if not cy["queue"] and not cy["active"]:
                    break
            _sync(dev)
            self.t_phase["gen"] += time.perf_counter() - t0
            if cy["queue"] or cy["active"]:
                return 0
            seqs = torch.cat([s for s, _ in cy["done"]], 0)
            rew = torch.rand(self.n_prompts, g)
            adv = ((rew - rew.mean(1, keepdim=True)) / (rew.std(1, keepdim=True) + 1e-4)).view(-1).to(dev)
            cy["train"] = (seqs, adv, 0)
            return 0
        seqs, adv, j = cy["train"]
        R = self.n_prompts * g
        lp = self.logprobs(seqs[j:j + self.train_mb])
        (-(adv[j:j + self.train_mb] * lp).sum() / R).backward()
        n = min(self.train_mb, seqs.shape[0] - j)
        j += n
        cy["train"] = (seqs, adv, j)
        if j >= seqs.shape[0]:
            self._opt_step()
            _sync(dev)
            self.cycles.append({"s": round(time.perf_counter() - cy["t0"], 1), "samples": seqs.shape[0],
                                "decode_steps": cy["steps"], "peak_rows": cy["peak_rows"]})
            self.last_done = cy["done"]
            self._cyc = None
        _sync(dev)
        self.t_phase["train"] += time.perf_counter() - t0
        self.steps += 1
        return n

    def close(self):
        if self.flush:
            self.flush.stop()
            self.pacer.stop()


def _sync(dev):
    if torch.cuda.is_available() and str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def fit_rl(model: Model, spec: dict, margin_mib: float = 160.0, log=print, device="cuda"):
    """Build a LocalRL sized to the memory actually free in this process now (call in the cheater after sealing,
    before ready). margin_mib is an EXPLICIT reserve for the prover's challenge kernels, whose peak runs concurrently
    at verification time but NOT during these probes. With seg 1 and the token-chunked MLP the 512-token update fits
    the cut; this picks the largest mlp_chunk whose training micro-batch fits and the largest gen_in_flight whose
    generation fits, both under free - margin. Generation and the update are sequential phases (generation frees its
    KV before the update), so each knob is sized against free independently. An OOM (incl. cuBLAS/cuDNN alloc
    failures) steps the knob down; the probes never run the optimiser, so the master weights are untouched."""
    import gc, json
    from schemes.memory_link_exhaustion.attacks.max_shard.workload import _is_oom
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    try:                                   # force cuBLAS/cuDNN handle creation before measuring (one-time, non-caching)
        _w = torch.empty(96, 96, device=device, dtype=torch.bfloat16); (_w @ _w).relu_().sum().item(); del _w
    except Exception as _e:
        log(f"local_rl fit: cublas warmup skipped ({type(_e).__name__})")
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    rl = LocalRL(model, dict(spec, idle=False, seg=1), device=device)   # seg 1 = layer-by-layer, the smallest update
    g = rl.group
    drv, _ = torch.cuda.mem_get_info()
    free = drv + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    target = free - margin_mib * 2**20
    base = torch.cuda.memory_allocated()

    def reset():
        for p in rl.params:
            p.grad = None
        if rl.flush:
            rl.flush.drain()
            for a in rl.flush.acc:
                a.zero_()
        if rl.stream_adapters:
            rl._dev_ad.clear()
            for a in rl.sacc:
                a.zero_()
        rl._kv0, rl._kv0_step = None, -1
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()

    def peak_of(fn):
        torch.cuda.reset_peak_memory_stats()
        try:
            fn(); _sync(device); torch.cuda.synchronize()
        except Exception as e:
            reset()
            if _is_oom(e):
                return None
            raise
        pk = torch.cuda.max_memory_allocated() - base
        reset()
        return pk

    rl.lm_chunk = int(spec.get("lm_chunk", 256))         # token-chunk the LM head too (bit-identical, cheap)
    log(f"local_rl fit: resident {base / 2**30:.3f} GiB "
        f"({'streamed adapters (per-layer)' if rl.stream_adapters else 'adapters+flusher'}), "
        f"free {free / 2**30:.3f}, target {target / 2**30:.3f} GiB")

    def train_peak(seg, ck):
        rl.seg, rl.mlp_chunk = seg, ck
        seq = torch.randint(rl.vocab_lo, rl.vocab_hi, (rl.train_mb, rl.T), device=device)
        rl.logprobs(seq, checkpoint=True).sum().backward()

    # The update's constant floor at seg 1 is the L full-sequence checkpoint inputs (L x [b, T-1, H]); shrinking the
    # MLP chunk cannot touch it. Sweep seg (fewer, coarser checkpoints trade for more per-segment recompute) and pick
    # the minimum-peak seg that fits; then take the largest mlp_chunk that still fits (fewer recompute passes).
    segL = min(64, rl.m.cfg.L)
    seg_cands = [s for s in [16, 8, 4, 2, 1] if s <= segL] or [1]
    best = None                                          # (peak, seg)
    for seg in seg_cands:
        for ck0 in (128, 64, 32):           # smaller MLP chunks lower the update peak
            pk = peak_of(lambda seg=seg, ck0=ck0: train_peak(seg, ck0))
            log(f"local_rl fit: seg {seg} mlp_chunk {ck0} update "
                f"{'OOM' if pk is None else f'{pk / 2**30:.3f} GiB (target {target / 2**30:.3f})'}")
            if pk is not None and pk <= target:
                if best is None or pk < best[0]:
                    best = (pk, seg, ck0)
                break
    if best is None:
        hint = ("checkpoint inputs (adapters are already streamed per layer)" if rl.stream_adapters else
                f"checkpoint inputs + {base / 2**30:.3f} GiB of resident adapters; try stream_adapters=true (point 3)")
        raise RuntimeError(f"local_rl fit: the update does not fit in {target / 2**30:.3f} GiB at any seg -- the floor "
                           f"is the {hint}")
    seg, chunk = best[1], best[2]
    for ck in ([512, 384, 256, 192] if chunk == 128 else []):
        pk = peak_of(lambda ck=ck: train_peak(seg, ck))
        log(f"local_rl fit: seg {seg} mlp_chunk {ck} update "
            f"{'OOM' if pk is None else f'{pk / 2**30:.3f} GiB (target {target / 2**30:.3f})'}")
        if pk is not None and pk <= target:
            chunk = ck
            break
    rl.seg, rl.mlp_chunk = seg, chunk

    gif = None
    for k in range(rl.max_groups, 0, -1):
        cand = k * g
        def gen(cand=cand):
            rl.gen_in_flight = cand
            ids1 = torch.randint(rl.vocab_lo, rl.vocab_hi, (max(1, cand // g), rl.P - rl.L0), device=device)
            rl._generate(ids1)
        pk = peak_of(gen)
        log(f"local_rl fit: gen_in_flight {cand} ({k} groups) "
            f"{'OOM' if pk is None else f'{pk / 2**30:.3f} GiB (target {target / 2**30:.3f})'}")
        if pk is not None and pk <= target:
            gif = cand
            break
    if gif is None:                                      # below one group: sub-group generation (s rollouts at a time)
        for cand in sorted({x for x in (g // 2, 2, 1) if 0 < x < g}, reverse=True):
            def gen_sub(cand=cand):
                rl.gen_in_flight = cand
                ids1 = torch.randint(rl.vocab_lo, rl.vocab_hi, (1, rl.P - rl.L0), device=device)
                rl._generate_sub(ids1, cand)
            pk = peak_of(gen_sub)
            log(f"local_rl fit: gen_in_flight {cand} (sub-group) "
                f"{'OOM' if pk is None else f'{pk / 2**30:.3f} GiB (target {target / 2**30:.3f})'}")
            if pk is not None and pk <= target:
                gif = cand
                break
    if gif is None:
        raise RuntimeError(f"local_rl fit: even one rollout's generation does not fit in {target / 2**30:.3f} GiB")
    rl.gen_in_flight = gif
    rl.idle = bool(spec.get("idle", False))     # idle control: same build + probes before ready, no work after
    rl.link_bytes, rl.calls = 0, 0
    rl.fit = {"free_gib": round(free / 2**30, 3), "target_gib": round(target / 2**30, 3),
              "resident_gib": round(base / 2**30, 3), "seg": seg, "mlp_chunk": chunk,
              "lm_chunk": rl.lm_chunk, "gen_in_flight": gif, "stream_adapters": rl.stream_adapters}
    log(f"local_rl fit: {json.dumps(rl.fit)}")
    return rl
