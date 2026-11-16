"""Unauthorised workload 2: RL on the current (served) model.

GRPO-style policy gradient with LoRA (rank 16 on q and v) on the same model the
authorised workload serves. Task: two-digit multiplication, reward 1 for the
right answer. One RL step = sample n_prompts x group completions (temperature
1), group-normalised advantages, one gradient step on the LoRA weights.

Normal:      plain bf16 weights, cuBLAS, rollouts in a CUDA-graph engine.
Constrained: inside the prover. The base weights can't be changed or copied
             (the arena is verified and full), so it reads the *encoded* arena
             weights through the decode-on-load kernels for rollouts, forward
             and backward; LoRA weights, optimiser state, rollout KV cache and
             activations must fit in the allowance; compute is paced.
             (Full fine-tuning is impossible without changing W pages, which
             the challenges catch.)

    python -m baselines.rl.rl --model DIR --seconds 120   # normal run
"""

from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from schemes.memory_link_exhaustion.core.engine import Engine
from schemes.memory_link_exhaustion.core.model import KV, Lin, Model


class Task:
    def __init__(self, tok, seed=0):
        self.tok = tok
        self.rng = np.random.default_rng(seed)
        self.plen = None

    def sample(self, n):
        out, ans = [], []
        while len(out) < n:
            a, b = self.rng.integers(10, 100, size=2)
            ids = self.tok(f"Question: what is {a} times {b}? Reply with just the number.\n"
                           f"Answer:", add_special_tokens=False)["input_ids"]
            if self.plen is None:
                self.plen = len(ids)
            if len(ids) != self.plen:
                continue
            out.append(ids)
            ans.append(int(a * b))
        return out, ans

    def reward(self, text, ans):
        m = re.search(r"-?\d[\d,]*", text)
        return 1.0 if m and int(m.group().replace(",", "")) == ans else 0.0


class RL:
    unit = "samples"

    def __init__(self, model: Model, tok, spec: dict, budget=None, pace=None, stream=None):
        self.m, self.tok, self.pace = model, tok, pace
        self.budget = budget
        self._mem0 = torch.cuda.memory_allocated()
        c = model.cfg
        self.n_prompts = spec.get("n_prompts", 16)
        self.group = spec.get("group", 4)
        self.G = spec.get("gen", 64)
        self.R = self.n_prompts * self.group
        self.task = Task(tok)
        self.task.sample(1)
        self.P = self.task.plen
        # long-context variant (spec prompt_pad): a fixed block of context tokens before every prompt
        pad = int(spec.get("prompt_pad", 0))
        gp = torch.Generator(device="cpu").manual_seed(1)
        self.pad_ids = torch.randint(1000, 30000, (pad,), generator=gp).tolist() if pad else []
        self.P += pad
        self.T = self.P + self.G
        r = spec.get("rank", 16)
        dev = model.device
        g = torch.Generator(device="cpu").manual_seed(0)
        self.params = []
        lora = []
        for l in range(c.L):
            Aq = (torch.randn(r, c.H, generator=g) / c.H ** 0.5).to(dev, torch.bfloat16)
            Av = (torch.randn(r, c.H, generator=g) / c.H ** 0.5).to(dev, torch.bfloat16)
            Bq = torch.zeros(c.Hq * c.D, r, device=dev, dtype=torch.bfloat16)
            Bv = torch.zeros(c.Hkv * c.D, r, device=dev, dtype=torch.bfloat16)
            for t in (Aq, Av, Bq, Bv):
                t.requires_grad_(True)
            self.params += [Aq, Bq, Av, Bv]
            lora.append((Aq, Bq, Av, Bv, 2.0))
        model.lora = lora
        if spec.get("lora_modules", "qv") == "all":
            # all-projection LoRA (q, k, v, o, gate, up, down of every layer), as the realistic RL adapter
            model.lora, self.params, mods = None, [], []
            dims = {"q": (c.H, c.Hq * c.D), "k": (c.H, c.Hkv * c.D), "v": (c.H, c.Hkv * c.D), "o": (c.Hq * c.D, c.H),
                    "gate": (c.H, c.I), "up": (c.H, c.I), "down": (c.I, c.H)}
            for l in range(c.L):
                d = {}
                for k_, (fi, fo) in dims.items():
                    A = (torch.randn(r, fi, generator=g) / fi ** 0.5).to(dev, torch.bfloat16).requires_grad_(True)
                    B = torch.zeros(fo, r, device=dev, dtype=torch.bfloat16).requires_grad_(True)
                    d[k_] = (A, B)
                    self.params += [A, B]
                mods.append(d)
            model.lora_mods, model.lora_scale = mods, 2.0
        self.opt = torch.optim.AdamW(self.params, lr=spec.get("lr", 1e-4))
        self.kv = KV.plain(c, self.R, self.T, dev)
        self.engine = Engine(model, self.kv, self.R, self.P, self.G, chunk_seqs=self.R,
                             graph=bool(spec.get("graph", True)),
                             stream=stream, temperature=1.0,
                             capture_mode=spec.get("capture_mode") or
                             ("thread_local" if pace else "global"),
                             phase_hook=(lambda b, k, w: pace.wait() if w == "end" else None)
                             if pace else None)
        self.mb = spec.get("mb", 16) if budget is None else None
        # memory saving (spec free_kv): the rollout KV slabs hold storage only during generation; the update
        # (checkpointed forward + backward) reuses that memory. The engine runs eagerly (graph false) when
        # this is on, since the slab addresses change between steps.
        self.free_kv = bool(spec.get("free_kv", False))
        self.loss_chunk = int(spec.get("loss_chunk", 0))
        self._kv_bytes = [t.untyped_storage().nbytes() for t in self.kv.slabs] if self.free_kv else None
        if self.free_kv:
            self._kv_free()
        self.steps = 0
        self.reward_mean = []
        self.t_roll = self.t_train = 0.0
        if self.mb is None:
            self.mb = self._fit_mb()

    def _kv_free(self):
        for t in self.kv.slabs:
            t.untyped_storage().resize_(0)

    def _kv_alloc(self):
        for t, n in zip(self.kv.slabs, self._kv_bytes):
            t.untyped_storage().resize_(n)

    def describe(self):
        return {"rollouts_per_step": self.R, "gen": self.G, "prompt_len": self.P, "mb": self.mb,
                "steps": self.steps, "rollout_s": round(self.t_roll, 2),
                "train_s": round(self.t_train, 2),
                "mean_reward": float(np.mean(self.reward_mean)) if self.reward_mean else None,
                "lora_params": sum(p.numel() for p in self.params)}

    def _loss(self, seq, adv):
        h = self.m.forward_train(seq[:, :-1])
        hs = h[:, self.P - 1:].reshape(-1, h.shape[-1])
        tgt = seq[:, self.P:].reshape(-1)
        if self.loss_chunk:
            # memory saving (spec loss_chunk): fp32 logits over the 152k vocabulary only for this many tokens at a
            # time, recomputed in the backward
            def _lp(hh, tt):
                return -F.cross_entropy(self.m.logits(hh).float(), tt, reduction="none")
            lp = torch.cat([torch.utils.checkpoint.checkpoint(_lp, hs[i:i + self.loss_chunk], tgt[i:i + self.loss_chunk],
                                                              use_reentrant=False)
                            for i in range(0, hs.shape[0], self.loss_chunk)])
        else:
            lp = -F.cross_entropy(self.m.logits(hs).float(), tgt, reduction="none")
        lp = lp.view(seq.shape[0], -1).mean(1)
        return -(adv * lp).sum() / self.R

    def _fit_mb(self):
        """Largest power-of-2 micro-batch that fits the budget, without ever hitting OOM
        (see Pretrain._fit_mb)."""
        seq = torch.randint(100, 1000, (1, self.T), device=self.m.device)
        adv = torch.zeros(1, device=self.m.device)
        m0 = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        self._loss(seq, adv).backward()
        torch.cuda.current_stream().synchronize()
        d1 = torch.cuda.max_memory_allocated() - m0
        resident = torch.cuda.memory_allocated() - self._mem0
        self.opt.zero_grad(set_to_none=False)
        room = (self.budget or (1 << 60)) * 0.8 - resident
        if d1 > room:
            raise RuntimeError(f"micro-batch 1 needs {d1 / 2**20:.0f} MiB, room {room / 2**20:.0f} MiB")
        mb = 1
        while mb < 16 and 2 * mb * d1 <= room:
            mb *= 2
        return mb

    def step(self):
        t0 = time.perf_counter()
        prompts, ans = self.task.sample(self.n_prompts)
        rep = [self.pad_ids + p for p in prompts for _ in range(self.group)]
        pr = torch.tensor(rep, device=self.m.device)
        if self.free_kv:
            self._kv_alloc()
        out = self.engine.generate(pr, batch_id=self.steps)
        if self.free_kv:
            torch.cuda.current_stream().synchronize()
            self._kv_free()
        texts = self.tok.batch_decode(out.cpu().tolist(), skip_special_tokens=True)
        rew = torch.tensor([self.task.reward(t, ans[i // self.group]) for i, t in enumerate(texts)])
        rg = rew.view(self.n_prompts, self.group)
        adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-4)).view(-1)
        adv = adv.to(self.m.device)
        self.reward_mean.append(float(rew.mean()))
        t1 = time.perf_counter()
        seq = torch.cat([pr, out], 1)
        for j in range(0, self.R, self.mb):
            self._loss(seq[j:j + self.mb], adv[j:j + self.mb]).backward()
            if self.pace:
                torch.cuda.current_stream().synchronize()
                self.pace.wait()
        self.opt.step()
        self.opt.zero_grad(set_to_none=False)
        torch.cuda.current_stream().synchronize()
        t2 = time.perf_counter()
        self.t_roll += t1 - t0
        self.t_train += t2 - t1
        self.steps += 1
        return self.R

    @staticmethod
    def inside_prover(spec, prover, budget, pace):
        """Build the cheat RL on top of the prover's *encoded* arena weights."""
        from transformers import AutoTokenizer
        hm = prover.model
        fused = spec.get("fused_max_m", 1 << 30)
        lins = {k: Lin(v.t, v.key, fused_max=fused) for k, v in hm.lins.items()}
        m = Model(hm.cfg, lins, hm.small, 4096, hm.device)
        tok = AutoTokenizer.from_pretrained(prover.cfg["model"])
        return RL(m, tok, spec, budget=budget, pace=pace,
                  stream=torch.cuda.current_stream())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--spec", default="{}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from schemes.memory_link_exhaustion.core.model import load_plain
    spec = json.loads(a.spec)
    m = load_plain(a.model, 4096, impl="torch")
    tok = AutoTokenizer.from_pretrained(a.model)
    wl = RL(m, tok, spec)
    wl.step()                     # warm-up
    torch.cuda.synchronize()
    wl.t_roll = wl.t_train = 0.0
    wl.steps = 0
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < a.seconds:
        n += wl.step()
    dt = time.perf_counter() - t0
    res = {"workload": "rl", "samples_s": n / dt, "seconds": dt, **wl.describe(),
           "max_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
           "gpu": torch.cuda.get_device_name(0)}
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
