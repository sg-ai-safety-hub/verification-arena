"""Max-param pipeline shard: the largest pretraining stage that fits under the memory cut.

Same attack as attacks/gated_shard (one pipeline stage of a Qwen2.5-32B-shaped model, neighbours emulated over the
gated, paced link), rebuilt for the final verifier configuration (cheater memory ~1.9 GiB, inbound-only saturating
noise with a windowed rate floor, outbound free). Three changes, each worth a factor on FLOPs/s:

1. **Weights only on the GPU (2 B/param, not 4).** No gradient accumulator on the device: every parameter's bf16
   gradient for one micro-batch leaves over D2H as soon as autograd has produced it (post-accumulate hook), and the
   host adds it into an fp32 accumulator. The host holds fp32 master weights and AdamW and sends the new bf16
   weights back once per optimiser step (paced inbound, counted). The MLP weights are row blocks with one
   checkpoint each, so every block's gradient is produced exactly once per micro-batch and only one block's
   gradient is on the device at a time. Outbound per micro-batch: 2 B x params (free today: only the inbound
   direction is exhausted; `out_gbs` caps it if echo noise returns).

2. **Stage-input stash on the host (`stash: "host"`).** A 1F1B stage keeps one checkpointed input per micro-batch
   in flight (P/2 on average for a P-stage chain). On the GPU that reserve grows as params shrink
   (2 x params + K / params, K ~ 3.3e17 B x param at seq 2048), and it is what kept the gated shard from starting
   under the cut. Sending the stash out (free) and back in costs 10,240 B/token more inbound (30,720 instead of
   20,480) but frees the reserve, which roughly doubles the params that fit: net ~1.3x more FLOPs per inbound
   byte. `stash: "gpu"` reserves it on the device instead (the honest comparison point).

3. **Start-up burst.** The windowed floor is enforced only from 60 s after the noise starts, and the whole-run
   pcie_rate check has 5 % tolerance. Until `t0 + burst_s` the inbound cap is `burst_gbs` (default 2 GB/s, which
   already makes this stage compute-bound); afterwards it is `link_gbs` (the smooth theft the floor allows).

Declared precision throughout: bf16 activations, gradients and weights on the link, nothing compressed.

Rate: tokens/s through the stage; FLOPs = 6 x params x tokens (the per-GPU share of the cluster, as gated_shard).
Expected (memory 1.9 GiB, seq 2048, stash host): params ~0.75e9, link-bound tokens/s = link / 30,720 + the
weight upload; at 0.1 GB/s ~3,200 tok/s -> ~1.5e13 FLOP/s (~23x below a bare H200); at 0.3 GB/s ~8x; compute-bound
above ~1 GB/s (~2x, thin-stage GEMM efficiency).
"""
from __future__ import annotations

import queue
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from schemes.memory_link_exhaustion.attacks.gated_shard import gate as G
from schemes.memory_link_exhaustion.attacks.gated_shard.workload import QWEN32B, Pacer, _mm, _rms, _rope


def _rms_fused(x, w, eps):
    """Fused RMSNorm on the bf16 input (fp32 accumulation inside the kernel; autograd keeps the input and the per-row
    rstd only). the plain _rms kept fp32 copies of [T, d] (40 MB each at 2,048 tokens), ~140 MiB of the
    0.47 GiB step peak in the warm-step allocation snapshot, plus their fp32 backward temporaries."""
    return F.rms_norm(x, (x.shape[-1],), w.to(x.dtype), eps)


if __import__("os").environ.get("VARENA_FUSED_NORM", "1") == "1":
    _rms = _rms_fused  # noqa: F811

_MB_TIMER = __import__("os").environ.get("VARENA_MB_TIMER", "0") == "1"
MODEL_PARAMS = 64 * 487_605_248          # Qwen2.5-32B decoder layers (embeddings excluded)


class BlockLayer(nn.Module):
    """Qwen2.5-shaped decoder layer with the MLP split into nb row blocks (one Parameter each, one checkpoint each)."""

    def __init__(self, c, nb=8, head_groups=8):
        super().__init__()
        d, h, kv, hd, f = c["d"], c["heads"], c["kv"], c["hd"], c["ffn"]
        assert f % nb == 0, (f, nb)
        self.c, self.nb, self.hg = c, nb, head_groups
        # (keeps the declared 2,048 context): token-chunked layer. 0 = off. With tchunk > 0
        # the layer computes K/V for all T positions once, then runs query positions in chunks of tchunk (attention
        # of each chunk against the full causal prefix, o-proj, residual, norm, MLP), each chunk checkpointed, so the
        # live activations are O(tchunk) instead of O(T). Same computation and context; gradients accumulate exactly.
        self.tchunk = 0
        fb = f // nb
        self.ln1 = nn.Parameter(torch.ones(d))
        self.ln2 = nn.Parameter(torch.ones(d))
        self.wq, self.bq = nn.Parameter(torch.empty(h * hd, d)), nn.Parameter(torch.zeros(h * hd))
        self.wk, self.bk = nn.Parameter(torch.empty(kv * hd, d)), nn.Parameter(torch.zeros(kv * hd))
        self.wv, self.bv = nn.Parameter(torch.empty(kv * hd, d)), nn.Parameter(torch.zeros(kv * hd))
        self.wo = nn.Parameter(torch.empty(d, h * hd))
        self.wg = nn.ParameterList([nn.Parameter(torch.empty(fb, d)) for _ in range(nb)])
        self.wu = nn.ParameterList([nn.Parameter(torch.empty(fb, d)) for _ in range(nb)])
        self.wd = nn.ParameterList([nn.Parameter(torch.empty(d, fb)) for _ in range(nb)])

    def init_(self):
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, 0, 0.02)

    def _mlp_block(self, y, i):
        g = G.g
        a = _mm(g(y), self.wg[i])
        b = _mm(g(y), self.wu[i])
        return _mm(g(F.silu(a) * b), self.wd[i])

    def _kv(self, x, cos, sin):
        c, g = self.c, G.g
        B, T, _ = x.shape
        y = _rms(g(x), self.ln1, c["eps"])
        k = _mm(g(y), self.wk, self.bk).view(B, T, c["kv"], c["hd"]).transpose(1, 2)
        v = _mm(g(y), self.wv, self.bv).view(B, T, c["kv"], c["hd"]).transpose(1, 2)
        return _rope(k, cos, sin), v

    def _chunk(self, xc, k, v, cosc, sinc, t0):
        c, g = self.c, G.g
        B, Tc, _ = xc.shape
        h, kv, hd = c["heads"], c["kv"], c["hd"]
        y = _rms(g(xc), self.ln1, c["eps"])
        q = _rope(g(_mm(g(y), self.wq, self.bq).view(B, Tc, h, hd).transpose(1, 2)), cosc, sinc)
        Lk = t0 + Tc
        mask = torch.arange(Lk, device=xc.device)[None, :] <= (t0 + torch.arange(Tc, device=xc.device))[:, None]
        rep = h // kv
        ng = max(1, min(self.hg, kv))
        per = kv // ng
        outs = []
        for i in range(ng):
            qi = g(q[:, i * per * rep:(i + 1) * per * rep])
            ki, vi = k[:, i * per:(i + 1) * per, :Lk], v[:, i * per:(i + 1) * per, :Lk]
            outs.append(F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, enable_gqa=True))
        a = torch.cat(outs, 1).transpose(1, 2).reshape(B, Tc, h * hd)
        xc = xc + _mm(g(a), self.wo)
        y = _rms(g(xc), self.ln2, c["eps"])
        for i in range(self.nb):
            xc = xc + self._mlp_block(y, i)
        return xc

    def _forward_chunked(self, x, cos, sin):
        T, tc = x.shape[1], self.tchunk
        if not getattr(BlockLayer, "_said", False):
            BlockLayer._said = True
            import sys as _s
            print(f"[max_shard] chunked layer path: T {T} tchunk {tc} x {tuple(x.shape)} {x.dtype}", file=_s.stderr,
                  flush=True)
        k, v = self._kv(x, cos, sin)
        outs = [ckpt.checkpoint(self._chunk, x[:, t0:t0 + tc], k, v, cos[t0:t0 + tc], sin[t0:t0 + tc], t0,
                                use_reentrant=False) for t0 in range(0, T, tc)]
        return torch.cat(outs, 1)

    def forward(self, x, cos, sin):
        c, g = self.c, G.g
        B, T, C = x.shape
        if self.tchunk and T > self.tchunk:
            return self._forward_chunked(x, cos, sin)
        h, kv, hd = c["heads"], c["kv"], c["hd"]
        y = _rms(g(x), self.ln1, c["eps"])
        q = _mm(g(y), self.wq, self.bq).view(B, T, h, hd).transpose(1, 2)
        k = _mm(g(y), self.wk, self.bk).view(B, T, kv, hd).transpose(1, 2)
        v = _mm(g(y), self.wv, self.bv).view(B, T, kv, hd).transpose(1, 2)
        q, k = _rope(g(q), cos, sin), _rope(k, cos, sin)
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
        for i in range(self.nb):
            x = x + ckpt.checkpoint(self._mlp_block, y, i, use_reentrant=False)
        return x


class GradFlusher:
    """Per-parameter gradient egress: autograd hook -> D2H on a side stream (gated, optionally paced) -> fp32 host
    accumulator. The device never holds more than max_pending bytes of gradients."""

    def __init__(self, params, out_gbs=0.0, chunk=4 << 20, max_pending=256 << 20, hooks=True):
        self.params = list(params)
        self.cuda = torch.cuda.is_available() and self.params[0].is_cuda
        pin = self.cuda
        self.stg = [torch.empty(p.shape, dtype=p.dtype, pin_memory=pin) for p in self.params]
        self.acc = [torch.zeros(p.shape, dtype=torch.float32) for p in self.params]
        self.out_gbs, self.chunk, self.max_pending = out_gbs, chunk, max_pending
        self.stream = torch.cuda.Stream(priority=0) if self.cuda else None
        self.q: queue.Queue = queue.Queue()
        self.pending = 0
        self.cv = threading.Condition()
        self.moved = 0
        self.flushes = 0
        self.err = None
        self.hooks = [p.register_post_accumulate_grad_hook(self._hook_for(i)) for i, p in enumerate(self.params)] \
            if hooks else []
        self.aq: queue.Queue = queue.Queue()
        self.t_issue = self.t_copy_wait = self.t_add = self.t_hook_wait = 0.0
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        self.th2 = threading.Thread(target=self._acc_loop, daemon=True)
        self.th2.start()

    def _hook_for(self, i):
        def hook(p):
            gr = p.grad
            p.grad = None
            n = gr.numel() * gr.element_size()
            _tw = time.perf_counter()
            with self.cv:
                while self.pending > 0 and self.pending + n > self.max_pending:
                    self.cv.wait(0.01)
                self.pending += n
            self.t_hook_wait += time.perf_counter() - _tw
            ev = None
            if self.cuda:
                ev = torch.cuda.Event()
                ev.record()                       # on the autograd (compute) stream
            self.q.put((i, gr, ev, n))
        return hook

    def _loop(self):
        """Copy issuer: issue each gradient's D2H on the side stream as soon as its event is
        recorded and hand (i, copy-done event) to the accumulator thread. The old loop synchronized the stream and did
        the fp32 host accumulate inline per parameter, which serialized egress at ~22 MB/s (ctx-128 shard)."""
        if self.cuda:
            torch.cuda.set_device(self.params[0].device)
        while True:
            item = self.q.get()
            if item is None:
                self.aq.put(None)
                return
            i, gr, ev, n = item
            try:
                t0 = time.perf_counter()
                dst = self.stg[i].reshape(-1).view(torch.uint8)
                src = gr.reshape(-1).view(torch.uint8)
                done = None
                if self.cuda:
                    with torch.cuda.stream(self.stream):
                        self.stream.wait_event(ev)
                        for j in range(0, n, self.chunk):
                            m = min(self.chunk, n - j)
                            tc = time.perf_counter()
                            G.wait()
                            dst[j:j + m].copy_(src[j:j + m], non_blocking=True)
                            if self.out_gbs > 0:
                                self.stream.synchronize()
                                dt = m / (self.out_gbs * 1e9) - (time.perf_counter() - tc)
                                if dt > 0:
                                    time.sleep(dt)
                        done = torch.cuda.Event()
                        done.record(self.stream)
                else:
                    dst.copy_(src)
                self.t_issue += time.perf_counter() - t0
                self.aq.put((i, gr, done, n))
            except Exception as e:                # surface in describe(); never wedge the producer
                self.err = repr(e)
                with self.cv:
                    self.pending -= n
                    self.cv.notify_all()
                self.q.task_done()

    def _acc_loop(self):
        """Accumulator: wait for a gradient's D2H, add it into the fp32 host accumulator, release its pending bytes."""
        if self.cuda:
            torch.cuda.set_device(self.params[0].device)
        while True:
            item = self.aq.get()
            if item is None:
                return
            i, gr, done, n = item
            try:
                t0 = time.perf_counter()
                if done is not None:
                    done.synchronize()
                t1 = time.perf_counter()
                del gr
                self.acc[i].add_(self.stg[i])
                t2 = time.perf_counter()
                self.t_copy_wait += t1 - t0
                self.t_add += t2 - t1
                self.moved += n
                self.flushes += 1
            except Exception as e:
                self.err = repr(e)
            finally:
                with self.cv:
                    self.pending -= n
                    self.cv.notify_all()
                self.q.task_done()

    def drain(self):
        self.q.join()

    def pull(self):
        """grad_on_gpu mode: move the device-accumulated gradients into the host accumulators once (per optimiser
        step) and zero them on the device."""
        for i, p in enumerate(self.params):
            self.stg[i].copy_(p.grad)
            self.acc[i].add_(self.stg[i])
            p.grad.zero_()
            self.moved += p.grad.numel() * p.grad.element_size()
            self.flushes += 1

    def stop(self):
        self.q.put(None)
        for h in self.hooks:
            h.remove()


class MaxShard:
    unit = "tokens"

    def __init__(self, spec: dict | None = None, budget: int | None = None, device="cuda"):
        spec = dict(spec or {})
        cuda = torch.cuda.is_available()
        self.dev = device if cuda else "cpu"
        self.idle = bool(spec.get("idle", False))
        self.arch = dict(QWEN32B, **spec.get("arch", {}))
        self.ffns = [int(f) for f in spec.get("ffns", [27648, 9216])]
        self.T = int(spec.get("seq", 2048))
        self.global_tokens = int(spec.get("global_tokens", 1 << 20))
        self.b_act = int(spec.get("bytes_act", 10240))           # one bf16 5,120-wide vector per token
        self.stash = spec.get("stash", "host")
        # stage boundary per token: activation in + gradient in (and the same out); host stash adds one more
        # stage-input vector each way (out at forward, back in before the recompute)
        extra = self.b_act if self.stash == "host" else 0
        self.in_b = 2 * self.b_act + extra
        self.out_b = 2 * self.b_act + extra
        self.max_debt_mb = float(spec.get("max_debt_mb", 4))
        t0_wall = float(spec.get("t0_wall", time.time()))
        self.pace_loop = bool(spec.get("pace_loop", False))     # closed-loop pacing (attacks/local_rl/pace.py)
        self.loop_pacer = None
        self.pacer = Pacer(float(spec.get("link_gbs", 0.1)), int(float(spec.get("chunk_mib", 2)) * (1 << 20)), device,
                           gbs_out=float(spec.get("out_gbs", 0.0)) or None,
                           burst_gbs=float(spec.get("burst_gbs", 2.0)),
                           burst_until=t0_wall + float(spec.get("burst_s", 0.0)))
        nb = int(spec.get("mlp_blocks", 8))
        hg = int(spec.get("head_groups", 8))
        dt = torch.bfloat16 if cuda else torch.float32
        # built on the meta device and materialised directly on the GPU in bf16: no fp32 copy of the stage on the GPU
        # and no large host-to-device transfer (the cheater builds this before it reports ready, see cheater.py)
        with torch.device("meta"):
            self.layers = nn.ModuleList([BlockLayer(dict(self.arch, ffn=f), nb, hg) for f in self.ffns])
        for l in self.layers:
            l.tchunk = int(spec.get("tchunk", 0))
        gen = torch.Generator(device=self.dev).manual_seed(int(spec.get("seed", 0)))
        for mod in self.layers.modules():
            for name, p in list(mod._parameters.items()):
                t = torch.empty(p.shape, dtype=dt, device=self.dev)
                if p.dim() == 2:
                    t.normal_(0, 0.02, generator=gen)
                elif name.startswith("ln"):
                    t.fill_(1)
                else:
                    t.zero_()
                mod._parameters[name] = nn.Parameter(t)
        self.params = list(self.layers.parameters())
        assert all(p.device.type == torch.device(self.dev).type for p in self.params)
        self.master = [p.detach().float().cpu() for p in self.params]
        for m in self.master:
            m.requires_grad_(True)
        self.P = sum(p.numel() for p in self.params)
        self.opt = torch.optim.AdamW(self.master, lr=3e-4, weight_decay=0.1, foreach=True)
        self.hbf = [torch.empty(p.shape, dtype=dt, pin_memory=cuda) for p in self.params]
        # grad_on_gpu: a bf16 gradient accumulator on the device (4 B/param, the earlier layout) and one gradient
        # transfer per optimiser step; for a high stolen rate, where per-micro-batch egress (2 B/param per sequence)
        # would need tens of GB/s outbound
        self.grad_on_gpu = bool(spec.get("grad_on_gpu", False))
        self.flush = GradFlusher(self.params, float(spec.get("out_gbs", 0.0)),
                                 max_pending=int(float(spec.get("max_pending_mib", 96)) * (1 << 20)),
                                 hooks=not self.grad_on_gpu)
        if self.grad_on_gpu:
            for p in self.params:
                p.grad = torch.zeros_like(p)
        # cuda_graph (needs grad_on_gpu): after a few eager micro-batches the micro-batch (input, forward, backward
        # into the device gradient accumulators, with its challenge-gate waits) is captured as one CUDA graph and
        # replayed: one driver call per micro-batch instead of one per kernel. The optimiser step stays eager.
        self.cg = bool(spec.get("cuda_graph", False)) and self.grad_on_gpu and str(device).startswith("cuda")
        self._g, self._g_T, self._g_warm, self.g_replays, self.g_err = None, None, 0, 0, None
        self.accum = max(1, self.global_tokens // self.T)
        # 1F1B in-flight stage inputs (GPU stash): P/2 micro-batches on average over a P-stage chain
        self.stages = MODEL_PARAMS / self.P
        inflight = spec.get("inflight", "avg")
        if self.stash == "host" or inflight in ("none", 0):
            res = 0
        elif isinstance(inflight, (int, float)):
            res = int(float(inflight) * (1 << 20))
        else:
            n_mb = self.stages / 2 if inflight == "avg" else self.stages
            res = int(n_mb * self.T * self.b_act)
        self.inflight_bytes = res
        self._inflight = torch.empty(res, dtype=torch.uint8, device=self.dev) if res and cuda else None
        hd = self.arch["hd"]
        inv = 1.0 / (self.arch["theta"] ** (torch.arange(0, hd, 2, device=self.dev).float() / hd))
        ang = torch.outer(torch.arange(self.T, device=self.dev).float(), inv)
        ang = torch.cat((ang, ang), -1)
        self.cos, self.sin = ang.cos().to(dt), ang.sin().to(dt)
        self.dt = dt
        self.k = self.opt_steps = 0
        self.t_opt = self.t_link_wait = 0.0
        self.t_steps = []
        # runtime token fallback: the probe sizes the micro-batch before ready, but the challenge Triton kernels and the
        # noise path are live only once the verifier starts. If the first real micro-batch OOMs anyway, halve the
        # sequence (tokens are independent in the MLP stage and the optimiser step already spans global_tokens, so the
        # FLOPs/byte and the weight-upload amortisation are unchanged) and retry, down to seq_floor, instead of dying.
        self.seq_floor = int(spec.get("seq_floor", 128))
        self.last_oom = None
        self.oom_log = []
        self.shrinks = 0

    def describe(self):
        cuda = torch.cuda.is_available()
        return {"arch": "Qwen2.5-32B-shaped pipeline stage", "ffns": self.ffns, "params": self.P,
                "stages_equiv": round(self.stages, 1), "stash": self.stash,
                "tchunk": self.layers[0].tchunk if len(self.layers) else 0,
                "inflight_mib": round(self.inflight_bytes / 2**20, 1), "seq": self.T, "accum": self.accum,
                "in_bytes_tok": self.in_b, "out_bytes_tok": self.out_b, "link_gbs": self.pacer.gbs,
                "burst_gbs": self.pacer.burst_gbs, "burst_left_s": round(max(0.0, self.pacer.burst_until - time.time()), 1),
                "opt_steps": self.opt_steps, "opt_s": round(self.t_opt, 2), "link_wait_s": round(self.t_link_wait, 2),
                "step_s": [round(x, 3) for x in self.t_steps], "gate_waits": G.STATS["waits"],
                "moved_gb": {k: round(v / 1e9, 3) for k, v in self.pacer.moved.items()},
                "grad_out_gb": round(self.flush.moved / 1e9, 3), "grad_flushes": self.flush.flushes,
                "flush_t": {"issue_s": round(self.flush.t_issue, 2), "copy_wait_s": round(self.flush.t_copy_wait, 2),
                            "add_s": round(self.flush.t_add, 2), "hook_wait_s": round(self.flush.t_hook_wait, 2)},
                "mb_t": {k: round(v, 2) for k, v in getattr(self, "mb_t", {}).items()},
                "flush_err": self.flush.err, "grad_on_gpu": self.grad_on_gpu, "idle": self.idle,
                "cuda_graph": {"on": self.cg, "replays": self.g_replays, "error": self.g_err},
                "fit": getattr(self, "fit", None), "last_oom": self.last_oom, "shrinks": self.shrinks, "oom_log": self.oom_log[-4:],
                "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if cuda else None,
                # margin calibration (16 Nov): the job's whole-run peak over its fit-time peak, and the reserved peak
                "run_peak_gib": round((torch.cuda.max_memory_allocated() - (getattr(self, "fit", None) or {}).get("base_gib", 0) * 2**30) / 2**30, 3)
                if cuda and getattr(self, "fit", None) else None,
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3) if cuda else None}

    def _opt_step(self):
        t0 = time.perf_counter()
        self.flush.drain()
        if self.grad_on_gpu:
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
            self.flush.pull()
        for m, a in zip(self.master, self.flush.acc):
            m.grad = a
        self.opt.step()
        for a in self.flush.acc:
            a.zero_()
        # new bf16 weights in over the paced inbound link, after the boundary traffic owed so far
        while self.pacer.owed["h2d"] > 0:
            time.sleep(1e-3)
        for p, hb, m in zip(self.params, self.hbf, self.master):
            hb.copy_(m.detach())
            self.pacer.copy_tensor("h2d", p.data, hb)
        self.opt_steps += 1
        self.t_opt += time.perf_counter() - t0

    def step(self):
        """One micro-batch (one sequence) through the stage, forward + backward; an optimiser step every accum."""
        if self.idle:
            time.sleep(0.5)
            return 0
        t0 = time.perf_counter()
        if self.k % self.accum == 0:
            self._t_step0 = t0
        tok = self.T
        if getattr(self, "loop_pacer", None) is not None:
            self.loop_pacer.wait()              # closed-loop pacing (spec pace_loop): rest while the noise is under target
        tw = time.perf_counter()
        while self.pacer.debt() > self.max_debt_mb * tok * max(self.in_b, self.out_b):
            time.sleep(1e-4)
        self.t_link_wait += time.perf_counter() - tw
        self.pacer.owe(tok * self.in_b, tok * self.out_b)
        try:
            # every device allocation of the step is inside this guard: the micro-batch AND the optimiser step
            self._mb()
            self.k += 1
            if self.k % self.accum == 0:
                self._opt_step()
                self.t_steps.append(time.perf_counter() - self._t_step0)
            return tok
        except Exception as e:
            if not _is_oom(e):
                raise
            msg = f"seq {self.T}: {e}"                      # the failing request ("Tried to allocate ... free")
            e.__traceback__ = None                         # the traceback pins the failed micro-batch's frame (x, h and
            del e                                          # every activation autograd saved): drop it or nothing frees
        self._recover_oom(msg)
        if self._shrink():
            return 0                                       # retried at a smaller seq on the next call, do not die
        try:                                               # what holds the memory at the failure
            import sys as _s
            print(torch.cuda.memory_summary(abbreviated=True), file=_s.stderr, flush=True)
        except Exception:
            pass
        raise RuntimeError(f"max_shard: out of memory at the seq floor {self.seq_floor} ({msg})")

    def _mb(self):
        """One micro-batch, eager or as a replay of its captured CUDA graph (spec cuda_graph)."""
        if not self.cg:
            return self._micro_batch()
        if self._g is not None and self._g_T == self.T:
            self._g.replay()
            self.g_replays += 1
            return
        if self._g_T != self.T:                  # first use, or the sequence length shrank: warm up again
            self._g, self._g_T, self._g_warm = None, self.T, 0
        if self._g_warm < 2 * self.accum:
            self._g_warm += 1
            return self._micro_batch()
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._micro_batch(timer=False)
            self._g = g
        except Exception as e:                   # capture not possible here: stay eager and say so
            self.cg, self.g_err = False, repr(e)[:300]
            return self._micro_batch()

    def _micro_batch(self, timer=True):
        """Forward + backward of one sequence. Its own frame, so a failure releases x, h and the graph on unwind.
        VARENA_MB_TIMER=1: wall time per phase (synchronised), in describe()['mb_t']."""
        tm = timer and _MB_TIMER and str(self.dev).startswith("cuda")
        mt = self.__dict__.setdefault("mb_t", {"input": 0.0, "fwd": 0.0, "bwd": 0.0, "n": 0})
        t0 = time.perf_counter()
        x = torch.randn(1, self.T, self.arch["d"], device=self.dev, dtype=self.dt, requires_grad=True)
        if tm:
            torch.cuda.synchronize(); t1 = time.perf_counter(); mt["input"] += t1 - t0; t0 = t1
        h = x
        for l in self.layers:
            h = ckpt.checkpoint(l, h, self.cos, self.sin, use_reentrant=False)
        gout = torch.randn_like(h) * (1.0 / self.accum)      # gradient from the (emulated) next stage
        if tm:
            torch.cuda.synchronize(); t1 = time.perf_counter(); mt["fwd"] += t1 - t0; t0 = t1
        h.backward(gout)
        if tm:
            torch.cuda.synchronize(); t1 = time.perf_counter(); mt["bwd"] += t1 - t0
            mt["n"] += 1

    def _recover_oom(self, msg):
        """Release everything a failed micro-batch left behind before retrying smaller: pending gradient egress,
        partial device gradients, the dead graph, the allocator cache."""
        import gc
        self.last_oom = msg
        self.oom_log.append(msg[:300])
        self.flush.drain()
        for p in self.params:
            if self.grad_on_gpu:
                if p.grad is not None:
                    p.grad.zero_()
            else:
                p.grad = None
        gc.collect()
        if torch.cuda.is_available() and str(self.dev).startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _shrink(self):
        """Step the micro-batch sequence down the ladder and restart the accumulation window cleanly. Returns False at
        the floor (nothing smaller to try)."""
        if self.T <= self.seq_floor:
            return False
        self.T = max(self.seq_floor, self.T // 2)
        self.cos, self.sin = self.cos[:self.T], self.sin[:self.T]
        self.accum = max(1, self.global_tokens // self.T)
        self.flush.drain()
        for a in self.flush.acc:
            a.zero_()
        if self.grad_on_gpu:
            for p in self.params:
                if p.grad is not None:
                    p.grad.zero_()
        self.k = 0
        self.shrinks += 1
        return True

    def close(self):
        self.pacer.stop()
        self.flush.stop()


def _candidates(f_max=27648, step=1024):
    """Stage shapes, largest first: one full layer plus a partial second layer, then a single partial layer."""
    out = [[f_max, f] for f in range(f_max, 0, -step)] + [[f_max]]
    out += [[f] for f in range(f_max - step, 0, -step)]
    return out


def _is_oom(e: BaseException) -> bool:
    """True for a CUDA out-of-memory raised as OOM or as a cuBLAS/cuDNN alloc failure (cublasCreate and cuDNN
    handle/workspace allocations surface as a plain RuntimeError, not torch.OutOfMemoryError, under a tight cut)."""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    s = str(e)
    return any(k in s for k in ("CUBLAS_STATUS_ALLOC_FAILED", "CUBLAS_STATUS_NOT_INITIALIZED",
                                "CUDNN_STATUS_ALLOC_FAILED", "CUDNN_STATUS_NOT_INITIALIZED",
                                "CUDA out of memory", "cudaErrorMemoryAllocation", "out of memory"))


def build_fitted(spec: dict, margin_mib: float = 160.0, log=print, device="cuda"):
    """The largest MaxShard that fits the memory actually free in this process now (call it in the cheater after
    sealing, before ready): free = driver-free + torch-cached-but-unallocated, target = free - margin. The margin is
    an EXPLICIT reserve for the prover's challenge Triton kernels, whose peak runs concurrently with the workload at
    verification time but NOT during this probe, plus the challenge path's own transients.

    The binding term under the cut is NOT the parameter count but the per-micro-batch working set: 2,048 tokens through
    the MLP blocks with the autograd graph retained (observed to OOM in the first real backward even when the parameters
    fit with 0.45 GiB to spare). So this fits JOINTLY over (ffns, seq): for each shape, largest first, it steps the
    micro-batch sequence down the ladder 2,048 / 1,024 / 512 / 256 / 128 and probes a full forward + backward + gradient
    egress at that length, taking the LARGEST seq whose peak fits. Reducing seq is free throughput-wise: tokens are
    independent in the MLP stage, the optimiser step spans global_tokens either way (so the per-token weight-upload
    amortisation is unchanged), and the stage is link-bound here (bytes/token fixed), so FLOPs/s depends on params, not
    seq -- hence returning the largest params with its largest fitting seq. An OOM (incl. cuBLAS/cuDNN alloc failures)
    at a shape+seq steps to the next smaller seq; if the parameters themselves do not fit, the whole shape is skipped.
    The probe's gradients are discarded, so the optimiser state is untouched; step() still halves seq once more at run
    time if the live challenge path pushes the first real micro-batch over."""
    import gc
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    try:                                                   # force cuBLAS/cuDNN handle creation (lazy, non-caching) now,
        _w = torch.empty(96, 96, device=device, dtype=torch.bfloat16)   # so free already excludes the handle and a
        (_w @ _w).relu_().sum().item()                    # razor-thin probe will not die inside cublasCreate
        del _w
    except Exception as _e:                               # a tiny warmup should never OOM, but never let it crash sizing
        log(f"max_shard fit: cublas warmup skipped ({type(_e).__name__})")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    drv, _ = torch.cuda.mem_get_info()
    free = drv + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    target = free - margin_mib * 2**20
    base = torch.cuda.memory_allocated()
    gpugrad = bool(spec.get("grad_on_gpu", False))
    nb = int(spec.get("mlp_blocks", 8))
    seq_floor = int(spec.get("seq_floor", 128))
    seq0 = int(spec.get("seq", 2048))
    seq_ladder = [t for t in (2048, 1024, 512, 256, 128) if seq_floor <= t <= seq0] or [seq_floor]
    gbs = float(spec.get("link_gbs", 0.1))
    log(f"max_shard fit: free {free / 2**30:.3f} GiB, target {target / 2**30:.3f} (margin {margin_mib:.0f} MiB), "
        f"seq ladder {seq_ladder}")

    def cleanup(wl):
        if wl is not None:
            wl.close()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    for ffns in _candidates():
        if any(f % nb for f in ffns):
            continue
        P = sum(62_914_560 + 3 * 5120 * f + 2 * 5120 for f in ffns)
        if P * (4 if gpugrad else 2) > target:                       # the weights alone do not fit -> no seq helps
            continue
        params_oom = False
        for seq in seq_ladder:
            wl = None
            try:
                torch.cuda.reset_peak_memory_stats()
                wl = MaxShard(dict(spec, ffns=ffns, seq=seq, burst_s=0, link_gbs=1000.0), device=device)
            except Exception as e:                                   # OOM building the weights -> skip the whole shape
                if not _is_oom(e):
                    cleanup(wl)
                    raise
                log(f"max_shard fit: ffns {ffns} ({P / 1e6:.0f}M) weights OOM ({type(e).__name__}) -> next shape")
                params_oom = True
                cleanup(wl)
                break
            try:
                idle, wl.idle = wl.idle, False     # the idle control is probed too (before ready), then does nothing
                wl.step()                          # one full real micro-batch: forward + backward + gradient egress
                wl.idle = idle
                wl.flush.drain()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - base
            except Exception as e:
                if not _is_oom(e):
                    cleanup(wl)
                    raise
                log(f"max_shard fit: ffns {ffns} ({P / 1e6:.0f}M) seq {seq} step OOM ({type(e).__name__}): {e}")
                cleanup(wl)
                continue                           # try a shorter sequence for this same (largest-params) shape
            ok = peak <= target
            log(f"max_shard fit: ffns {ffns} params {wl.P / 1e6:.0f}M seq {seq} peak {peak / 2**30:.3f} GiB "
                f"target {target / 2**30:.3f} GiB: {'OK' if ok else 'too big'}")
            if ok:
                for a in wl.flush.acc:
                    a.zero_()
                if wl.grad_on_gpu:
                    for p in wl.params:
                        p.grad.zero_()
                wl.k = 0
                wl.pacer.gbs = gbs
                wl.fit = {"free_gib": round(free / 2**30, 3), "target_gib": round(target / 2**30, 3),
                          "peak_gib": round(peak / 2**30, 3), "seq": seq, "params_m": round(wl.P / 1e6, 1),
                          # allocator baseline before the job: the run's own peak is peak_alloc_gib - base_gib
                          "base_gib": round(base / 2**30, 3), "base_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3)}
                return wl
            cleanup(wl)                            # fits the weights but not the working set -> shorter sequence
        if params_oom:
            continue
    raise RuntimeError(f"max_shard fit: nothing fits in {target / 2**30:.3f} GiB at any shape x seq "
                       f"{seq_ladder} (largest working set over params was the binding term)")
