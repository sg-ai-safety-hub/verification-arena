"""Unauthorised workload 1: pretraining a new model (GPT-2 small, 124M, from scratch).

Normal:      everything on the GPU, fp32 master weights + AdamW, bf16 autocast,
             micro-batch 16 x 1024 tokens, 4 micro-batches per optimiser step.
Constrained: what fits in the HBM the verifier leaves out. If the full fp32
             training state fits, it stays resident; otherwise (the usual case)
             the GPU keeps only bf16 weights and gradients, and the fp32 master
             weights and AdamW state live on the host (ZeRO-offload style), so
             every optimiser step moves gradients D2H and weights H2D over the
             (noise-saturated, paced) PCIe link. Micro-batch size is the largest
             that fits, with per-block activation checkpointing. Same global
             batch (64K tokens per step) and same model as normal.

Data is random tokens: throughput doesn't depend on content.

    python -m baselines.pretrain.pretrain --seconds 60      # normal run
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


def lin(x, m: nn.Linear):
    """nn.Linear via addmm/mm: inside the vLLM prover, batch-invariant mode overrides
    aten::linear/matmul at the CUDA key (breaking their autograd); mm keeps its own."""
    g = getattr(m, "_varena_gate", None)       # cheater experiments: hook_linears sets a gate (module hooks never
    if g is not None:                          # fire here, since this bypasses nn.Linear.forward)
        g()
    x2 = x.reshape(-1, x.shape[-1])
    w = m.weight.to(x2.dtype)
    if m.bias is not None:
        y = torch.addmm(m.bias.to(x2.dtype), x2, w.t())
    else:
        y = torch.mm(x2, w.t())
    if g is not None and torch.is_grad_enabled() and y.requires_grad:
        y.register_hook(lambda gr: (g(), None)[1])    # gate before this layer's backward too
    return y.view(*x.shape[:-1], y.shape[-1])


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.h = h

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = lin(self.ln1(x), self.attn).split(C, 2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + lin(y.transpose(1, 2).reshape(B, T, C), self.proj)
        return x + lin(F.gelu(lin(self.ln2(x), self.fc), approximate="tanh"), self.fc2)


class GPT(nn.Module):
    def __init__(self, n_layer=12, n_head=12, d=768, vocab=50304, block=1024):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.vocab = vocab
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        self.checkpoint = False

    def forward(self, idx, targets):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = ckpt.checkpoint(b, x, use_reentrant=False) if self.checkpoint else b(x)
        x = self.ln(x)
        # chunked loss so the [B*T, vocab] logits never exist at once
        loss = 0.0
        xs, ts = x.reshape(-1, x.shape[-1]), targets.reshape(-1)
        n = xs.shape[0]
        step = 2048
        for j in range(0, n, step):
            xj = xs[j:j + step]
            lg = torch.mm(xj, self.wte.weight.to(xj.dtype).t()).float()
            loss = loss + F.cross_entropy(lg, ts[j:j + step], reduction="sum")
        return loss / n


def nparams(m):
    return sum(p.numel() for p in m.parameters())


class Pretrain:
    unit = "tokens"

    def __init__(self, spec: dict | None = None, budget: int | None = None, pace=None,
                 device="cuda"):
        spec = spec or {}
        self.T = spec.get("seq", 1024)
        self.global_tokens = spec.get("global_tokens", 64 * 1024)
        self.pace = pace
        self.dev = device
        self.budget = budget
        self._mem0 = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        torch.manual_seed(0)
        cfg = spec.get("model", {})
        model = GPT(**cfg)
        self.P = nparams(model)
        full_state = self.P * 16
        if budget is None:
            self.mode = "normal"
        elif full_state + (512 << 20) < budget:
            self.mode = "resident"
        else:
            self.mode = "offload"
        if self.mode in ("normal", "resident"):
            self.model = model.to(device)
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.1,
                                         fused=True)
        else:
            # fp32 master + AdamW on the host; bf16 copy on the GPU
            self.master = [p.detach().clone().pin_memory() for p in model.parameters()]
            for p in self.master:
                p.requires_grad_(True)
            self.opt = torch.optim.AdamW(self.master, lr=3e-4, weight_decay=0.1, foreach=True)
            self.model = model.to(device=device, dtype=torch.bfloat16)
            self.hgrad = [torch.empty(p.shape, dtype=torch.bfloat16).pin_memory()
                          for p in self.master]
            self.hbf = [torch.empty(p.shape, dtype=torch.bfloat16).pin_memory()
                        for p in self.master]
        self.model.checkpoint = self.mode != "normal"
        self.accum = 1
        if "mb" in spec and self.mode != "normal":
            self.mb = min(int(spec["mb"]), self._fit_mb())          # explicit (small) micro-batch, if it fits
        else:
            self.mb = spec.get("mb", 16) if self.mode == "normal" else self._fit_mb()
        self.accum = max(1, self.global_tokens // (self.mb * self.T))
        self.k = 0
        self.opt_steps = 0
        self.t_opt = 0.0

    def describe(self):
        return {"params": self.P, "mode": self.mode, "mb": self.mb, "accum": self.accum,
                "seq": self.T, "opt_steps": self.opt_steps, "opt_s": round(self.t_opt, 2)}

    def _batch(self, mb):
        x = torch.randint(0, 50257, (mb, self.T + 1), device=self.dev)
        return x[:, :-1], x[:, 1:]

    def _fwd_bwd(self, mb):
        x, y = self._batch(mb)
        if self.mode == "offload":
            loss = self.model(x, y)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = self.model(x, y)
        (loss / self.accum).backward()
        return loss

    def _fit_mb(self):
        """Largest power-of-2 micro-batch whose memory fits the budget. Never runs out of
        memory on purpose (an OOM would make the allocator drop the host engine's
        cached blocks): measure micro-batch 1, scale linearly, keep 20% slack."""
        m0 = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        self._fwd_bwd(1)
        torch.cuda.synchronize()
        d1 = torch.cuda.max_memory_allocated() - m0
        resident = torch.cuda.memory_allocated() - self._mem0
        self.model.zero_grad(set_to_none=False)
        room = (self.budget or (1 << 60)) * 0.8 - resident
        if d1 > room:
            raise RuntimeError(f"micro-batch 1 needs {d1 / 2**20:.0f} MiB, room {room / 2**20:.0f} MiB")
        mb = 1
        while mb < 16 and 2 * mb * d1 <= room:
            mb *= 2
        return mb

    def _opt_step(self):
        t0 = time.perf_counter()
        if self.mode != "offload":
            self.opt.step()
            self.opt.zero_grad(set_to_none=False)
        else:
            gp = list(self.model.parameters())
            for p, hg, m in zip(gp, self.hgrad, self.master):
                self.pace.copy(hg, p.grad) if self.pace else hg.copy_(p.grad)
                m.grad = hg.float()
            self.opt.step()
            for p, hb, m in zip(gp, self.hbf, self.master):
                hb.copy_(m.detach())
                self.pace.copy(p.data, hb) if self.pace else p.data.copy_(hb)
                p.grad.zero_()
        self.opt_steps += 1
        self.t_opt += time.perf_counter() - t0

    def step(self):
        """One micro-batch (and an optimiser step every `accum`). Returns tokens."""
        self._fwd_bwd(self.mb)
        self.k += 1
        if self.k % self.accum == 0:
            self._opt_step()
        torch.cuda.current_stream().synchronize()
        return self.mb * self.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--budget-gib", type=float, default=None,
                    help="simulate a memory budget (constrained mode without a prover)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    budget = int(a.budget_gib * (1 << 30)) if a.budget_gib else None
    wl = Pretrain({}, budget=budget)
    for _ in range(wl.accum):          # warm-up one optimiser step
        wl.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < a.seconds:
        n += wl.step()
    dt = time.perf_counter() - t0
    res = {"workload": "pretrain", "tok_s": n / dt, "seconds": dt, **wl.describe(),
           "max_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
           "gpu": torch.cuda.get_device_name(0)}
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
