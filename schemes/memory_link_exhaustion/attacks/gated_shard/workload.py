"""Gated sharded pretraining: one pipeline stage of a 32B-shaped model inside the cheater's leftover HBM.

What is different from baselines.pretrain.pretrain_qwen in the cheater runs so far:

* gate before every op, forward and backward (gate.GateFn), with the big ops cut into short kernels (MLP in token
  chunks, attention in head groups), so a challenge meets at most one short kernel; no duty cycle;
* fp32 master weights + AdamW on the host. The GPU holds bf16 weights + a bf16 gradient accumulator only
  (4 B/param), so a whole 32B layer (~488M params) fits in ~2.5 GB;
* pipeline traffic as a stage would need it (bytes per token in / out, default 10,240 each way: one 5,120-wide
  tensor per direction per token, e.g. fp8 activations + fp8 gradients for a middle stage, or bf16 at a first or
  last stage), moved by a pacer thread in small chunks on its own streams, overlapped with compute and capped at
  link_gbs; the optimiser step's gradients go out D2H and new weights come in H2D the same way.

Throughput is model tokens/s through the stage; model FLOPs = 6 * params * tokens (3 MAC).
"""
from __future__ import annotations

import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from schemes.memory_link_exhaustion.attacks.gated_shard import gate as G

QWEN32B = dict(d=5120, heads=40, kv=8, hd=128, ffn=27648, theta=1e6, eps=1e-6)


def _mm(x, w, b=None):
    x2 = x.reshape(-1, x.shape[-1])
    y = torch.addmm(b, x2, w.t()) if b is not None else torch.mm(x2, w.t())
    return y.view(*x.shape[:-1], w.shape[0])


def _rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def _rope(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., h:], x[..., :h]), -1) * sin


class Layer(nn.Module):
    def __init__(self, c, chunk=512, head_groups=8):
        super().__init__()
        d, h, kv, hd, f = c["d"], c["heads"], c["kv"], c["hd"], c["ffn"]
        self.c, self.chunk, self.hg = c, chunk, head_groups
        self.ln1 = nn.Parameter(torch.ones(d))
        self.ln2 = nn.Parameter(torch.ones(d))
        self.wq, self.bq = nn.Parameter(torch.empty(h * hd, d)), nn.Parameter(torch.zeros(h * hd))
        self.wk, self.bk = nn.Parameter(torch.empty(kv * hd, d)), nn.Parameter(torch.zeros(kv * hd))
        self.wv, self.bv = nn.Parameter(torch.empty(kv * hd, d)), nn.Parameter(torch.zeros(kv * hd))
        self.wo = nn.Parameter(torch.empty(d, h * hd))
        self.wg = nn.Parameter(torch.empty(f, d))
        self.wu = nn.Parameter(torch.empty(f, d))
        self.wd = nn.Parameter(torch.empty(d, f))

    def init_(self):
        for p in (self.wq, self.wk, self.wv, self.wo, self.wg, self.wu, self.wd):
            nn.init.normal_(p, 0, 0.02)

    def _mlp(self, y):
        g = G.g
        a = _mm(g(y), self.wg)
        b = _mm(g(y), self.wu)
        return _mm(g(F.silu(a) * b), self.wd)

    def forward(self, x, cos, sin):
        c, g = self.c, G.g
        B, T, C = x.shape
        h, kv, hd = c["heads"], c["kv"], c["hd"]
        y = _rms(g(x), self.ln1, c["eps"])
        q = _mm(g(y), self.wq, self.bq).view(B, T, h, hd).transpose(1, 2)
        k = _mm(g(y), self.wk, self.bk).view(B, T, kv, hd).transpose(1, 2)
        v = _mm(g(y), self.wv, self.bv).view(B, T, kv, hd).transpose(1, 2)
        q, k = _rope(g(q), cos, sin), _rope(k, cos, sin)
        # attention in groups of query heads (each group shares whole KV heads): short kernels
        rep = h // kv
        ng = max(1, min(self.hg, kv))
        per = kv // ng
        outs = []
        for i in range(ng):
            qi = g(q[:, i * per * rep:(i + 1) * per * rep])
            ki, vi = k[:, i * per:(i + 1) * per], v[:, i * per:(i + 1) * per]
            outs.append(F.scaled_dot_product_attention(qi, ki, vi, is_causal=True, enable_gqa=True))
        a = torch.cat(outs, 1).transpose(1, 2).reshape(B, T, h * hd)
        x = x + _mm(g(a), self.wo)
        y = _rms(g(x), self.ln2, c["eps"])
        # MLP in token chunks, each checkpointed (peak: one chunk's intermediates)
        yc = y.reshape(B * T, C)
        parts = [ckpt.checkpoint(self._mlp, yc[i:i + self.chunk], use_reentrant=False)
                 for i in range(0, B * T, self.chunk)]
        return x + torch.cat(parts, 0).view(B, T, C)


class Pacer:
    """Pipeline / optimiser link traffic in chunks on side streams, capped at gbs per direction, with a gate wait
    before every chunk (no DMA runs into a challenge)."""

    def __init__(self, gbs: float, chunk: int = 2 << 20, device="cuda", gbs_out: float | None = None,
                 burst_gbs: float = 0.0, burst_until: float = 0.0):
        # gbs_out: outbound cap (default = gbs). burst_gbs / burst_until (time.time() epoch): a higher inbound cap
        # before burst_until (the verifier's windowed floor skips the first 60 s after the noise starts)
        self.gbs, self.chunk = gbs, chunk
        # routed single H2D queue (VARENA_SINGLE_QUEUE=1, tools/h2d_route.c): only H2D copies <= H2D_ROUTE_MAX (1 MiB)
        # enter the noise queue; larger ones go to the copy engines on their own and starve behind the saturating
        # noise (a 2 MiB chunk waited minutes, the 173M shard's 17.5 s per micro-batch). Split them.
        import os as _os
        self.h2d_max = int(_os.environ.get("H2D_ROUTE_MAX", 1 << 20)) \
            if _os.environ.get("VARENA_SINGLE_QUEUE") == "1" else 0
        self.gbs_out = gbs if gbs_out is None else gbs_out
        self.burst_gbs, self.burst_until = burst_gbs, burst_until
        cuda = torch.cuda.is_available()
        self.dev = device if cuda else "cpu"
        self.s = {d: (torch.cuda.Stream(priority=0) if cuda else None) for d in ("h2d", "d2h")}
        self.host = torch.empty(chunk, dtype=torch.uint8, pin_memory=cuda)
        self.devbuf = torch.empty(chunk, dtype=torch.uint8, device=self.dev)
        self.owed = {"h2d": 0, "d2h": 0}
        self.moved = {"h2d": 0, "d2h": 0}
        self.lock = threading.Condition()
        self.run = True
        self.th = [threading.Thread(target=self._loop, args=(d,), daemon=True) for d in ("h2d", "d2h")]
        for t in self.th:
            t.start()

    def owe(self, n_in, n_out):
        with self.lock:
            self.owed["h2d"] += n_in
            self.owed["d2h"] += n_out
            self.lock.notify_all()

    def debt(self):
        with self.lock:
            return max(self.owed.values())

    def rate(self, d):
        if d == "d2h":
            return self.gbs_out
        if self.burst_gbs and time.time() < self.burst_until:
            return self.burst_gbs
        return self.gbs

    def _loop(self, d):
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        while self.run:
            with self.lock:
                while self.run and self.owed[d] <= 0:
                    self.lock.wait(0.05)
                n = min(self.chunk, self.owed[d])
            if not self.run:
                return
            t0 = time.perf_counter()
            self.copy(d, n)
            with self.lock:
                self.owed[d] -= n
                self.moved[d] += n
                self.lock.notify_all()
            r = self.rate(d)
            dt = (n / (r * 1e9) - (time.perf_counter() - t0)) if r > 0 else 0.0
            if dt > 0:
                time.sleep(dt)

    def copy(self, d, n, dst=None, src=None):
        """One chunk (or a given dst/src pair) in direction d, gated, synchronous."""
        s = self.s[d]
        if dst is None:
            dst, src = (self.devbuf[:n], self.host[:n]) if d == "h2d" else (self.host[:n], self.devbuf[:n])
        if s is None:
            dst.copy_(src)
            return
        with torch.cuda.stream(s):
            G.wait()
            m = self.h2d_max if (d == "h2d" and self.h2d_max) else 0
            if m and dst.numel() * dst.element_size() > m:
                dv, sv = dst.reshape(-1).view(torch.uint8), src.reshape(-1).view(torch.uint8)
                for j in range(0, dv.numel(), m):
                    dv[j:j + m].copy_(sv[j:j + m], non_blocking=True)
            else:
                dst.copy_(src, non_blocking=True)
        s.synchronize()

    def copy_tensor(self, d, dst: torch.Tensor, src: torch.Tensor):
        """A whole tensor in paced chunks (optimiser traffic), counted in moved."""
        dv, sv = dst.reshape(-1).view(torch.uint8), src.reshape(-1).view(torch.uint8)
        n = sv.numel()
        for j in range(0, n, self.chunk):
            m = min(self.chunk, n - j)
            t0 = time.perf_counter()
            self.copy(d, m, dv[j:j + m], sv[j:j + m])
            with self.lock:
                self.moved[d] += m
            r = self.rate(d)
            dt = (m / (r * 1e9) - (time.perf_counter() - t0)) if r > 0 else 0.0
            if dt > 0:
                time.sleep(dt)

    def stop(self):
        self.run = False
        with self.lock:
            self.lock.notify_all()


class GatedShard:
    unit = "tokens"

    def __init__(self, spec: dict | None = None, budget: int | None = None, device="cuda"):
        spec = dict(spec or {})
        cuda = torch.cuda.is_available()
        self.dev = device if cuda else "cpu"
        self.arch = dict(QWEN32B, **spec.get("arch", {}))
        self.T = int(spec.get("seq", 2048))
        self.mb = int(spec.get("mb", 1))
        self.global_tokens = int(spec.get("global_tokens", 1 << 20))
        self.in_b = int(spec.get("in_bytes_tok", 10240))
        self.out_b = int(spec.get("out_bytes_tok", 10240))
        self.max_debt_mb = float(spec.get("max_debt_mb", 4))       # micro-batches of link debt before compute waits
        self.pacer = Pacer(float(spec.get("link_gbs", 2.0)), int(float(spec.get("chunk_mib", 2)) * (1 << 20)),
                           device)
        self.layer = Layer(self.arch, int(spec.get("chunk", 512)), int(spec.get("head_groups", 8)))
        dt = torch.bfloat16 if cuda else torch.float32
        self.layer.init_()
        self.master = [p.detach().float().clone().pin_memory() if cuda else p.detach().float().clone()
                       for p in self.layer.parameters()]
        for m in self.master:
            m.requires_grad_(True)
        self.layer.to(device=self.dev, dtype=dt)
        for p in self.layer.parameters():
            p.grad = torch.zeros_like(p)
        self.P = sum(p.numel() for p in self.layer.parameters())
        self.opt = torch.optim.AdamW(self.master, lr=3e-4, weight_decay=0.1, foreach=True)
        self.hgrad = [torch.empty(p.shape, dtype=dt, pin_memory=cuda) for p in self.master]
        self.hbf = [torch.empty(p.shape, dtype=dt, pin_memory=cuda) for p in self.master]
        self.accum = max(1, self.global_tokens // (self.mb * self.T))
        hd = self.arch["hd"]
        inv = 1.0 / (self.arch["theta"] ** (torch.arange(0, hd, 2, device=self.dev).float() / hd))
        ang = torch.outer(torch.arange(self.T, device=self.dev).float(), inv)
        ang = torch.cat((ang, ang), -1)
        self.cos, self.sin = ang.cos().to(dt), ang.sin().to(dt)
        self.dt = dt
        self.k = self.opt_steps = 0
        self.t_opt = self.t_link_wait = 0.0
        self.t_steps = []
        self.budget = budget
        # stitching: a real 1F1B pipeline stage holds activations for up to P micro-batches in
        # flight (P/2 on average across stages); reserve that memory so the slice that fits is the honest one
        res = int(float(spec.get("inflight_reserve_mib", 0)) * (1 << 20))
        self._inflight = torch.empty(res, dtype=torch.uint8, device=self.dev) if res and cuda else None

    def describe(self):
        return {"arch": "Qwen2.5-32B-shaped pipeline stage, 1 layer", "params": self.P, "ffn": self.arch["ffn"],
                "mb": self.mb, "seq": self.T, "accum": self.accum, "tokens_per_step": self.mb * self.T * self.accum,
                "opt_steps": self.opt_steps, "opt_s": round(self.t_opt, 2), "link_wait_s": round(self.t_link_wait, 2),
                "step_s": [round(x, 3) for x in self.t_steps], "gate_waits": G.STATS["waits"],
                "link_gbs": self.pacer.gbs, "in_bytes_tok": self.in_b, "out_bytes_tok": self.out_b,
                "moved_gb": {k: round(v / 1e9, 3) for k, v in self.pacer.moved.items()},
                "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_available() else None}

    def _opt_step(self):
        t0 = time.perf_counter()
        ps = list(self.layer.parameters())
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        for p, hg in zip(ps, self.hgrad):
            self.pacer.copy_tensor("d2h", hg, p.grad)
        for hg, m in zip(self.hgrad, self.master):
            m.grad = hg.float()
        self.opt.step()
        for p, hb, m in zip(ps, self.hbf, self.master):
            hb.copy_(m.detach())
            self.pacer.copy_tensor("h2d", p.data, hb)
            p.grad.zero_()
        self.opt_steps += 1
        self.t_opt += time.perf_counter() - t0

    def step(self):
        """One micro-batch (+ an optimiser step every accum). Returns tokens."""
        t0 = time.perf_counter()
        if self.k % self.accum == 0:
            self._t_step0 = t0
        tok = self.mb * self.T
        # link: this micro-batch's inbound activations / gradients and outbound ones; compute waits only if the
        # link falls more than max_debt_mb micro-batches behind
        tw = time.perf_counter()
        while self.pacer.debt() > self.max_debt_mb * tok * max(self.in_b, self.out_b):
            time.sleep(1e-4)
        self.t_link_wait += time.perf_counter() - tw
        self.pacer.owe(tok * self.in_b, tok * self.out_b)
        x = torch.randn(self.mb, self.T, self.arch["d"], device=self.dev, dtype=self.dt, requires_grad=True)
        y = ckpt.checkpoint(self.layer, x, self.cos, self.sin, use_reentrant=False)
        (y.float().pow(2).mean() / self.accum).backward()
        self.k += 1
        if self.k % self.accum == 0:
            self._opt_step()
            self.t_steps.append(time.perf_counter() - self._t_step0)
        return tok

    def close(self):
        self.pacer.stop()
