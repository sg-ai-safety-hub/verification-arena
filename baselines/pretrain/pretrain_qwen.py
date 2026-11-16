"""Unauthorised workload 1b: pretraining the Qwen2.5-7B architecture, one pipeline-stage shard.

The model is treated as sharded: one GPU holds one pipeline stage, `layers` consecutive Qwen2
decoder layers with the real Qwen2.5-7B dimensions (hidden 3584, 28 query heads / 4 KV heads
(GQA), head dim 128, SwiGLU MLP 18944, RMSNorm, RoPE theta 1e6, qkv bias). It is a MIDDLE stage:
no embedding and no lm_head (the first stage would add the 152064 x 3584 embedding, 0.55B params,
the last stage the lm_head of the same size; neither is included). Input activations are random
(what the previous stage would send), the loss is a stand-in (mean square of the stage output,
what a real stage would get back as an output gradient), so throughput doesn't depend on data.

Training state: fp32 master weights + AdamW (16 B/param: param 4, grad 4, two moments 8), bf16
autocast, activation checkpointing per layer. One optimiser step = `global_tokens` tokens through
the shard in micro-batches of `mb` sequences of `seq` tokens.

ZeRO-3 equivalence: under ZeRO-3 (or FSDP full sharding) each GPU owns 1/G of every layer's
parameters, gradients and optimiser state instead of all of `layers` layers; its persistent state per
GPU is the same 16 B x (params / G), it all-gathers each layer's weights just in time and reduce-
scatters gradients. The per-GPU optimiser-state memory and the optimiser step are therefore the same
as for a pipeline stage holding the same number of parameters; what differs is the interconnect
traffic (weights all-gathered every forward/backward vs activations passed between stages), which a
single GPU does not measure.

Modes (as workloads.pretrain): normal (idle GPU, everything resident); resident (budget fits);
offload (GPU keeps bf16 weights and gradients only, fp32 master + AdamW on the host, gradients
D2H and weights H2D over PCIe every optimiser step).

    python -m baselines.pretrain.pretrain_qwen --layers auto --out pq_normal.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from baselines.pretrain.pretrain import lin

QWEN7B = dict(d=3584, heads=28, kv=4, hd=128, ffn=18944, theta=1e6, eps=1e-6)


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def rope(x, cos, sin):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return x * cos + torch.cat((-x2, x1), -1) * sin


SDPA_MATH = False      # set by PretrainQwen(spec sdpa='math')


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, h, kv, hd, f = c["d"], c["heads"], c["kv"], c["hd"], c["ffn"]
        self.h, self.kv, self.hd = h, kv, hd
        self.ln1 = RMSNorm(d, c["eps"])
        self.q = nn.Linear(d, h * hd, bias=True)
        self.k = nn.Linear(d, kv * hd, bias=True)
        self.v = nn.Linear(d, kv * hd, bias=True)
        self.o = nn.Linear(h * hd, d, bias=False)
        self.ln2 = RMSNorm(d, c["eps"])
        self.gate = nn.Linear(d, f, bias=False)
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        y = self.ln1(x)
        q = lin(y, self.q).view(B, T, self.h, self.hd).transpose(1, 2)
        k = lin(y, self.k).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = lin(y, self.v).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        if SDPA_MATH:
            # deterministic training (bit-exact runs): the math backend's backward has no atomics
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(SDPBackend.MATH):
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + lin(a.transpose(1, 2).reshape(B, T, C), self.o)
        y = self.ln2(x)
        return x + lin(F.silu(lin(y, self.gate)) * lin(y, self.up), self.down)


class Stage(nn.Module):
    def __init__(self, layers, c=QWEN7B):
        super().__init__()
        self.c = c
        self.layers = nn.ModuleList([Layer(c) for _ in range(layers)])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.checkpoint = True

    def forward(self, x):
        T = x.shape[1]
        hd = self.c["hd"]
        inv = 1.0 / (self.c["theta"] ** (torch.arange(0, hd, 2, device=x.device).float() / hd))
        ang = torch.outer(torch.arange(T, device=x.device).float(), inv)
        ang = torch.cat((ang, ang), -1)
        cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        for L in self.layers:
            x = ckpt.checkpoint(L, x, cos, sin, use_reentrant=False) if self.checkpoint else L(x, cos, sin)
        return x


def nparams(m):
    return sum(p.numel() for p in m.parameters())


class PretrainQwen:
    unit = "tokens"

    def __init__(self, spec: dict | None = None, budget: int | None = None, pace=None, device="cuda"):
        spec = spec or {}
        self.T = spec.get("seq", 4096)
        self.global_tokens = spec.get("global_tokens", 64 * 1024)
        self.mb = spec.get("mb", 4)
        self.layers = int(spec.get("layers", 32))
        self.pace = pace
        self.dev = device
        self.budget = budget
        self.duty = float(spec.get("duty", 1.0))       # paced variant: fraction of time busy
        # pipeline-sharded fleet: activations in from the previous stage and out to the next one per
        # micro-batch (pinned host <-> GPU, hop_mib each way)
        hop = int(float(spec.get("hop_mib", 0)) * (1 << 20))
        # sub-layer shard (tensor-parallel slice): narrower MLP so weights + Adam stay resident (no weight traffic)
        self.arch = dict(QWEN7B, **spec.get("arch", {}))
        # paced inbound: activations H2D in chunks at a capped rate, yielding between chunks (strongest shard cheater)
        self.pace_gbs = float(spec.get("hop_pace_gbs", 0))
        self.chunk = int(float(spec.get("hop_chunk_mib", 1)) * (1 << 20))
        self.yield_fn = None
        # cuda_graph: replay the micro-batch and the optimiser step as captured CUDA graphs (see _fwd_bwd)
        self.cg, self._ga, self._gb, self._cg_warm = bool(spec.get("cuda_graph", False)), None, None, 0
        # theft sweep: {"rates": [GB/s, ...], "seg_s": seconds each, "delay_s": seconds at the first rate before the
        # clock starts}. The inbound pace steps through the rates in one run; describe() lists each segment with
        # its wall-clock bounds (CLOCK_MONOTONIC, the verifier's clock) and the bytes moved, so the noise the
        # verifier saw in that segment can be read off its arrival stamps (scripts/reproduce/replay.py sweep).
        sw = spec.get("hop_pace_sweep") or None
        self.sweep = None
        if sw:
            self.sweep = {"rates": [float(x) for x in sw["rates"]], "seg_s": float(sw.get("seg_s", 30)),
                          "delay_s": float(sw.get("delay_s", 0)), "t_start": None, "log": []}
            self.pace_gbs = max(self.sweep["rates"]) or 1.0
            # adaptive: {"adapt": {"floor": GB/s (default: VARENA_SAT_FLOOR_GBS), "gain": 0.45, "r_max": 2.0,
            # "segments": 9, "settle_s": 3}}. The first segment runs at rate 0 and measures the noise this process is
            # delivering (noise_fn: windows done, window bytes). The next rate is the estimate (noise - floor) x gain;
            # after that each segment is placed just under or over the current estimate of the theft at which the
            # noise meets the floor (a line fitted through the segments so far, kept inside the passing / failing
            # bracket). The cheater only chooses where to probe; the result is read from the verifier's stamps.
            ad = sw.get("adapt")
            self.noise_fn = None
            if ad is not None:
                import os as _os
                self.sweep["adapt"] = {"floor": float(ad.get("floor") or _os.environ.get("VARENA_SAT_FLOOR_GBS", "0") or 0),
                                       "gain": float(ad.get("gain", 0.45)), "r_max": float(ad.get("r_max", 2.0)),
                                       "segments": int(ad.get("segments", 9)), "settle_s": float(ad.get("settle_s", 3)),
                                       "pts": [], "est": None, "mark": None}
                self.sweep["rates"] = [0.0]
                self.pace_gbs = self.sweep["adapt"]["r_max"]
        # paced outbound (link-slack probe): D2H in chunks at a capped rate, yielding between chunks;
        # hop_in false: no inbound copy at all (outbound-only probe)
        self.out_pace_gbs = float(spec.get("hop_out_pace_gbs", 0))
        self.hop_in = bool(spec.get("hop_in", True))
        # hop_out false (or an outbound pace below 0.05 GB/s, as the inbound probe specs write it): no outbound copy
        self.hop_out = bool(spec.get("hop_out", True)) and not (0 < self.out_pace_gbs < 0.05)
        self.h2d_stream = torch.cuda.Stream(priority=0) if (self.pace_gbs > 0 or self.out_pace_gbs > 0) else None
        self.hop = None
        if hop:
            self.hop = (torch.empty(hop, dtype=torch.uint8).pin_memory(),
                        torch.empty(hop, dtype=torch.uint8, device=device))
        global SDPA_MATH
        SDPA_MATH = spec.get("sdpa") == "math"
        torch.manual_seed(0)
        with torch.device("meta"):
            P = nparams(Stage(self.layers, self.arch))
        self.P = P
        if budget is None:
            self.mode = "normal"
        elif P * 16 + (int(spec.get("headroom_gib", 2)) << 30) < budget:
            self.mode = "resident"
        else:
            self.mode = "offload"
        if self.mode in ("normal", "resident"):
            with torch.device(device):              # init on the GPU (CPU init of ~8B params is slow)
                self.model = Stage(self.layers, self.arch)
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.1,
                                         fused=bool(spec.get("fused_adam", True)),
                                         foreach=False if not spec.get("fused_adam", True) else None, capturable=bool(getattr(self, "cg", False)))
        else:
            # built layer by layer on the GPU (fp32 init), fp32 master copied to pinned host
            # memory, the GPU keeps bf16 only (peak extra: one fp32 layer)
            with torch.device("meta"):
                model = Stage(self.layers, self.arch)
            self.master = []
            for L in model.layers:
                L.to_empty(device=device)
                for m in L.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, RMSNorm):
                        nn.init.ones_(m.weight)
                for p in L.parameters():
                    self.master.append(p.detach().cpu().pin_memory())
                L.to(torch.bfloat16)
                torch.cuda.empty_cache()
            for p in self.master:
                p.requires_grad_(True)
            self.opt = torch.optim.AdamW(self.master, lr=3e-4, weight_decay=0.1, foreach=True)
            self.model = model
            for p in self.model.parameters():
                p.grad = torch.zeros_like(p)
            self.hgrad = [torch.empty(p.shape, dtype=torch.bfloat16).pin_memory() for p in self.master]
            self.hbf = [torch.empty(p.shape, dtype=torch.bfloat16).pin_memory() for p in self.master]
        self.accum = max(1, self.global_tokens // (self.mb * self.T))
        self.k = 0
        self.opt_steps = 0
        self.t_opt = 0.0
        self.t_steps = []

    def _adapt_next(self, now):
        """Adaptive theft sweep: score the segment that just ended (noise this process delivered after settling),
        then choose the next rate."""
        sw, ad = self.sweep, self.sweep["adapt"]
        r = sw["rates"][-1]
        if ad["mark"] is not None:
            n1, w = self.noise_fn()
            dt = now - ad["mark"][0]
            if dt > 2.0:
                ad["pts"].append((r, (n1 - ad["mark"][1]) * w / dt / 1e9))
        ad["mark"] = None
        if len(sw["rates"]) >= ad["segments"] or not ad["pts"]:
            sw["rates"].append(0.0)                                          # done (or nothing measured): stop stealing
            return
        est, nxt = next_theft_rate(ad["pts"], ad["floor"], ad["gain"], ad["r_max"], len(sw["rates"]))
        ad["est"] = est
        sw["rates"].append(nxt)

    def describe(self):
        return {"arch": "Qwen2.5-7B middle pipeline stage", "layers": self.layers, "params": self.P,
                "mode": self.mode, "mb": self.mb, "seq": self.T, "accum": self.accum,
                "tokens_per_step": self.mb * self.T * self.accum, "opt_steps": self.opt_steps,
                "opt_s": round(self.t_opt, 3), "step_s": [round(x, 3) for x in self.t_steps],
                "duty": self.duty, "hop_mib": (self.hop[0].numel() >> 20) if self.hop is not None else 0,
                **({"sweep": [[round(a, 4), round(b, 4), r, n] for a, b, r, n in self.sweep["log"]]} if self.sweep else {}),
                **({"sweep_adapt": {"floor": self.sweep["adapt"]["floor"], "est_gbs": self.sweep["adapt"]["est"],
                                    "points": [[x, round(n, 3)] for x, n in self.sweep["adapt"]["pts"]]}}
                   if self.sweep and self.sweep.get("adapt") else {})}

    def _fwd_bwd(self):
        """One micro-batch. With spec cuda_graph (resident mode) the micro-batch is captured into a CUDA graph after a
        few eager ones and replayed from then on: the same GPU work with one driver call instead of one per kernel."""
        if getattr(self, "cg", False):
            if self._cg_warm < 3 * self.accum:
                self._cg_warm += 1
            elif self._ga is None:
                self._ga = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._ga):
                    self._fwd_bwd_eager()
                return
            else:
                self._ga.replay()
                return
        self._fwd_bwd_eager()

    def _fwd_bwd_eager(self):
        d = self.model.c["d"]
        x = torch.randn(self.mb, self.T, d, device=self.dev, dtype=torch.bfloat16, requires_grad=True)
        if self.mode == "offload":
            y = self.model(x)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = self.model(x)
        loss = y.float().pow(2).mean()
        (loss / self.accum).backward()

    def _opt_step(self):
        t0 = time.perf_counter()
        if self.mode != "offload" and getattr(self, "cg", False) and self._ga is not None:
            if self._gb is None:
                self._gb = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._gb):
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=False)
            else:
                self._gb.replay()
        elif self.mode != "offload":
            self.opt.step()
            self.opt.zero_grad(set_to_none=False)
        else:
            gp = list(self.model.parameters())
            for p, hg, m in zip(gp, self.hgrad, self.master):
                self.pace.copy(hg, p.grad) if self.pace else hg.copy_(p.grad)
            torch.cuda.current_stream().synchronize()
            for hg, m in zip(self.hgrad, self.master):
                m.grad = hg.float()
            self.opt.step()
            for p, hb, m in zip(gp, self.hbf, self.master):
                hb.copy_(m.detach())
                self.pace.copy(p.data, hb) if self.pace else p.data.copy_(hb, non_blocking=True)
                p.grad.zero_()
        self.opt_steps += 1
        self.t_opt += time.perf_counter() - t0

    def _paced(self, src, dst, gbs):
        n = src.numel()
        torch.cuda.current_stream().synchronize()
        for o in range(0, n, self.chunk):
            if self.yield_fn is not None:
                self.yield_fn()
            tc = time.perf_counter()
            m = min(self.chunk, n - o)
            with torch.cuda.stream(self.h2d_stream):
                dst[o:o + m].copy_(src[o:o + m], non_blocking=True)
            self.h2d_stream.synchronize()
            dt = m / (gbs * 1e9) - (time.perf_counter() - tc)
            if dt > 0:
                time.sleep(dt)

    def step(self):
        """One micro-batch (and an optimiser step every `accum`). Returns tokens."""
        t0 = time.perf_counter()
        if self.k % self.accum == 0:
            self._t_step0 = t0
        pace = self.pace_gbs
        if self.sweep is not None:
            sw, now = self.sweep, time.perf_counter()
            if sw["t_start"] is None:
                if getattr(self, "noise_fn", None) is not None:      # the clock starts once the noise is flowing
                    n_ = self.noise_fn()[0]
                    sw.setdefault("n0", n_)
                    if n_ > sw["n0"] + 1000:
                        sw["t_start"] = now + sw["delay_s"]
                else:
                    sw["t_start"] = now + sw["delay_s"]
            if sw["t_start"] is None:
                sw["t_start_wait"] = True
            k = max(0, int((now - (sw["t_start"] or now)) // sw["seg_s"]))
            ad = sw.get("adapt")
            if ad is not None and self.noise_fn is not None:
                seg_t0 = sw["t_start"] + k * sw["seg_s"] if (sw["t_start"] is not None and now >= sw["t_start"]) else None
                if ad["mark"] is None and seg_t0 is not None and now >= seg_t0 + ad["settle_s"] and sw.get("k") == k:
                    ad["mark"] = (now, self.noise_fn()[0])                  # settled: start counting this segment
                if sw.get("k") is not None and k > sw["k"] and len(sw["rates"]) <= k:
                    self._adapt_next(now)                                    # segment over: score it, pick the next
            k = min(len(sw["rates"]) - 1, k)
            pace = sw["rates"][k]
            if sw.get("k") != k:
                sw["k"] = k
                if ad is not None:
                    ad["mark"] = None
                sw["log"].append([now, now, pace, 0])
            if pace <= 0:                       # a zero-rate segment: no hop at all, do not spin
                time.sleep(0.05)
        if self.hop is not None and not self.hop_in:
            pass
        elif self.sweep is not None and pace <= 0:
            pass
        elif self.hop is not None and self.pace_gbs > 0:
            # paced inbound: chunked H2D on its own stream at <= pace_gbs, yielding (cheater hook) between chunks
            n = self.hop[0].numel()
            for o in range(0, n, self.chunk):
                if self.yield_fn is not None:
                    self.yield_fn()
                tc = time.perf_counter()
                m = min(self.chunk, n - o)
                with torch.cuda.stream(self.h2d_stream):
                    self.hop[1][o:o + m].copy_(self.hop[0][o:o + m], non_blocking=True)
                self.h2d_stream.synchronize()
                dt = m / (pace * 1e9) - (time.perf_counter() - tc)
                if dt > 0:
                    time.sleep(dt)
                if self.sweep is not None:
                    self.sweep["log"][-1][3] += m
        elif self.hop is not None:
            self.hop[1].copy_(self.hop[0], non_blocking=True)        # activations in (H2D)
        self._fwd_bwd()
        if self.hop is not None and not self.hop_out:
            pass
        elif self.hop is not None and self.out_pace_gbs > 0:
            self._paced(self.hop[1], self.hop[0], self.out_pace_gbs)  # activations out (D2H), paced and gated
        elif self.hop is not None:
            self.hop[0].copy_(self.hop[1], non_blocking=True)        # activations out (D2H)
        if self.sweep is not None and self.sweep["log"]:
            self.sweep["log"][-1][1] = time.perf_counter()
        self.k += 1
        if self.k % self.accum == 0:
            self._opt_step()
        torch.cuda.current_stream().synchronize()
        if self.k % self.accum == 0:
            self.t_steps.append(time.perf_counter() - self._t_step0)
        if self.duty < 1.0:
            time.sleep((time.perf_counter() - t0) * (1 - self.duty) / self.duty)
        return self.mb * self.T


def next_theft_rate(pts, floor, gain=0.45, r_max=2.0, n_done=1):
    """Adaptive theft sweep: (estimate of the theft at which the noise meets the floor, next rate to probe).
    pts: [(theft GB/s, noise GB/s seen), ...], the first at theft 0. First guess: (noise - floor) x gain. Then a
    line through the points, kept inside the bracket [highest rate that stayed over the floor, lowest that fell
    under it]; probes alternate just under and just over the estimate."""
    passing = [x for x, n in pts if n >= floor]
    failing = [x for x, n in pts if n < floor]
    lo, hi = (max(passing) if passing else 0.0), (min(failing) if failing else None)
    est = None
    xs = sorted(set(x for x, _ in pts))
    if len(xs) >= 2 and xs[-1] - xs[0] > 0.05:
        mx = sum(x for x, _ in pts) / len(pts); my = sum(n for _, n in pts) / len(pts)
        sxx = sum((x - mx) ** 2 for x, _ in pts); sxy = sum((x - mx) * (n - my) for x, n in pts)
        if sxx > 0 and sxy < 0:
            est = mx + (floor - my) / (sxy / sxx)
    if est is None:
        est = (pts[0][1] - floor) * gain if len(pts) == 1 else (hi / 2 if hi is not None else lo * 1.6 + 0.1)
    if hi is not None and lo < hi:
        est = min(max(est, lo + 0.1 * (hi - lo)), hi - 0.1 * (hi - lo))
    est = max(0.0, est)
    side = 0.9 if n_done % 2 else 1.1
    return est, round(min(r_max, max(0.02, est * side)), 3)


def fits(layers, mb, seq):
    """Try one full optimiser step resident; free everything after."""
    try:
        wl = PretrainQwen({"layers": layers, "mb": mb, "seq": seq, "global_tokens": mb * seq})
        wl.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        ok = True
    except torch.OutOfMemoryError:
        ok, peak = False, None
    wl = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return ok, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="auto")
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--global-tokens", type=int, default=64 * 1024)
    ap.add_argument("--steps", type=int, default=2, help="timed optimiser steps (after 1 warm-up)")
    ap.add_argument("--budget-gib", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    probe = []
    if a.layers == "auto":
        n = 40
        while n > 0:
            ok, peak = fits(n, a.mb, a.seq)
            probe.append({"layers": n, "fits": ok, "peak_gib": peak / 2**30 if peak else None})
            print(json.dumps(probe[-1]), flush=True)
            if ok:
                break
            n -= 1
        # the probe step fits at n; the steady-state loop (accumulated gradients plus
        # allocator fragmentation) OOMed at n on the H200, so keep one layer of headroom
        layers = max(1, n - 1)
    else:
        layers = int(a.layers)
    budget = int(a.budget_gib * (1 << 30)) if a.budget_gib else None
    while True:
        torch.cuda.reset_peak_memory_stats()
        try:
            wl = PretrainQwen({"layers": layers, "mb": a.mb, "seq": a.seq, "global_tokens": a.global_tokens},
                              budget=budget)
            for _ in range(wl.accum):       # warm-up optimiser step
                wl.step()
            break
        except torch.OutOfMemoryError:
            # the probe's single micro-batch step fit, the full step (or fragmentation) didn't
            wl = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            probe.append({"layers": layers, "full_step_fits": False})
            print(json.dumps(probe[-1]), flush=True)
            layers -= 1
    wl.t_steps.clear()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for _ in range(a.steps * wl.accum):
        n += wl.step()
    dt = time.perf_counter() - t0
    res = {"workload": "pretrain_qwen", "tok_s": n / dt, "s_per_step": dt / a.steps, **wl.describe(),
           "max_alloc_gib": torch.cuda.max_memory_allocated() / 2**30, "probe": probe,
           "gpu": torch.cuda.get_device_name(0)}
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
