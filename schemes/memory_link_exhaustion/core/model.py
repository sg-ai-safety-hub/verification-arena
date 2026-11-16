"""A small Qwen2/Llama-style inference engine with plain and sealed weight / KV back ends.

Used outside vLLM: the RL baseline (baselines/rl), the cheater workloads (attacks/), the verifier's weight-file reader
(SafeTensors) and the prover's weight layout (Cfg, big_tensors). The vLLM prover itself does not run this engine.

impl="torch":  plain bf16 weights, cuBLAS matmuls.
impl="triton": plain bf16 weights, the Triton batch-invariant matmul.
sealed:        weights and KV cache in the arena (core/arena.py), XOR-encoded, decoded inside the kernels.

Everything runs greedy and in fixed shapes so a replay with the same prompts reproduces the KV cache bit for bit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from schemes.memory_link_exhaustion.gpu import kernels as K

FUSED_MAX_M = int(os.environ.get("VARENA_FUSED_MAX_M", "256"))


@dataclass
class Cfg:
    H: int
    Hq: int
    Hkv: int
    D: int
    I: int
    L: int
    V: int
    eps: float
    theta: float
    tied: bool
    bias: bool

    @staticmethod
    def from_hf(path: str) -> "Cfg":
        c = json.load(open(os.path.join(path, "config.json")))
        H = c["hidden_size"]
        Hq = c["num_attention_heads"]
        return Cfg(H=H, Hq=Hq, Hkv=c.get("num_key_value_heads", Hq),
                   D=c.get("head_dim") or H // Hq, I=c["intermediate_size"],
                   L=c["num_hidden_layers"], V=c["vocab_size"], eps=c["rms_norm_eps"],
                   theta=c.get("rope_theta", 10000.0),
                   tied=bool(c.get("tie_word_embeddings", False)),
                   bias=c.get("model_type", "") == "qwen2" or c.get("attention_bias", False))


def big_tensors(cfg: Cfg):
    """(name, shape, hf source names concatenated along dim 0) for every big matrix.
    These go to the arena; norms and biases (tiny) stay plain outside it."""
    out = [("embed", (cfg.V, cfg.H), ["model.embed_tokens.weight"])]
    qkv_rows = (cfg.Hq + 2 * cfg.Hkv) * cfg.D
    for l in range(cfg.L):
        p = f"model.layers.{l}."
        out.append((f"l{l}.qkv", (qkv_rows, cfg.H),
                    [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                     p + "self_attn.v_proj.weight"]))
        out.append((f"l{l}.o", (cfg.H, cfg.Hq * cfg.D), [p + "self_attn.o_proj.weight"]))
        out.append((f"l{l}.gate_up", (2 * cfg.I, cfg.H),
                    [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]))
        out.append((f"l{l}.down", (cfg.H, cfg.I), [p + "mlp.down_proj.weight"]))
    if not cfg.tied:
        out.append(("lm_head", (cfg.V, cfg.H), ["lm_head.weight"]))
    return out


def small_tensors(cfg: Cfg):
    names = ["model.norm.weight"]
    for l in range(cfg.L):
        p = f"model.layers.{l}."
        names += [p + "input_layernorm.weight", p + "post_attention_layernorm.weight"]
        if cfg.bias:
            names += [p + "self_attn.q_proj.bias", p + "self_attn.k_proj.bias",
                      p + "self_attn.v_proj.bias"]
    return names


class SafeTensors:
    """Lazy reader over a HF model directory (one or more .safetensors files)."""

    def __init__(self, path: str):
        from safetensors import safe_open
        self.files = {}
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            wm = json.load(open(idx))["weight_map"]
            for name, f in wm.items():
                self.files[name] = os.path.join(path, f)
        else:
            f = os.path.join(path, "model.safetensors")
            with safe_open(f, "pt") as h:
                for name in h.keys():
                    self.files[name] = f
        self._open = {}
        self._safe_open = safe_open

    def get(self, name: str) -> torch.Tensor:
        f = self.files[name]
        if f not in self._open:
            self._open[f] = self._safe_open(f, "pt")
        return self._open[f].get_tensor(name)

    def concat(self, names) -> torch.Tensor:
        ts = [self.get(n) for n in names]
        return ts[0] if len(ts) == 1 else torch.cat(ts, 0)


# --------------------------------------------------------------------------- linear
class Lin:
    """A weight matrix [N, K]. key None => plain bf16; else int16 encoded storage."""

    def __init__(self, t: torch.Tensor, key: int | None, impl: str = "torch",
                 fused_max: int = FUSED_MAX_M):
        self.t, self.key, self.impl, self.fused_max = t, key, impl, fused_max
        self.N, self.K = t.shape

    def decoded(self) -> torch.Tensor:
        out = torch.empty(self.t.shape, dtype=torch.bfloat16, device=self.t.device)
        K.xor16_w(self.t, out, self.key, self.K)
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and x.requires_grad and self.key is not None:
            return _EncLinFn.apply(x, self)
        return self.fwd(x)

    def fwd(self, x):
        if self.key is None:
            if self.impl == "torch":
                return F.linear(x, self.t)
            return K.matmul_bt(x, self.t, None)
        if x.shape[0] <= self.fused_max:
            return K.matmul_bt(x, self.t, self.key)
        return F.linear(x, self.decoded())

    def bwd(self, g):
        if self.key is None:
            return g @ self.t
        if g.shape[0] <= self.fused_max:
            return K.matmul_bt(g, self.t, self.key, transpose=True)
        return g @ self.decoded()


class _EncLinFn(torch.autograd.Function):
    """Frozen encoded weight: forward and input-gradient both decode on the fly."""

    @staticmethod
    def forward(ctx, x, lin):
        ctx.lin = lin
        return lin.fwd(x)

    @staticmethod
    def backward(ctx, g):
        return ctx.lin.bwd(g.contiguous()), None


def rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def rope_tables(cfg: Cfg, T: int, device):
    inv = 1.0 / (cfg.theta ** (torch.arange(0, cfg.D, 2, device=device).float() / cfg.D))
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    emb = torch.cat([f, f], -1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def rope(x, cos, sin):
    """x [..., n, h, D]; cos/sin broadcastable [..., n, 1, D]."""
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], -1)
    return x * cos + rot * sin


# --------------------------------------------------------------------------- model
class Model:
    def __init__(self, cfg: Cfg, lins: dict, small: dict, Tmax: int, device="cuda"):
        self.cfg = cfg
        self.lins = lins
        self.small = small
        self.device = device
        self.cos, self.sin = rope_tables(cfg, Tmax, device)
        c = cfg
        self.qkv_b = []
        for l in range(c.L):
            p = f"model.layers.{l}."
            if c.bias:
                self.qkv_b.append(torch.cat([small[p + "self_attn.q_proj.bias"],
                                             small[p + "self_attn.k_proj.bias"],
                                             small[p + "self_attn.v_proj.bias"]]))
            else:
                self.qkv_b.append(None)
        self.lm = lins.get("lm_head") or lins["embed"]
        self.lora = None      # optional {layer: (Aq, Bq, Av, Bv, scale)} for RL
        # optional all-projection LoRA for RL: per layer {"q"|"k"|"v"|"o"|"gate"|"up"|"down": (A [r, in], B [out, r])},
        # scale lora_scale; unset (None) leaves every forward unchanged
        self.lora_mods = None
        self.lora_scale = 2.0

    # -- helpers
    def _qkv(self, l, x):
        y = self.lins[f"l{l}.qkv"](x)
        if self.qkv_b[l] is not None:
            y = y + self.qkv_b[l]
        if self.lora is not None:
            Aq, Bq, Av, Bv, s = self.lora[l]
            c = self.cfg
            # torch.mm, not @: vLLM's batch-invariant mode overrides aten::matmul at the
            # CUDA key, which breaks its autograd; mm keeps its derivative formula
            dq = torch.mm(torch.mm(x, Aq.t()), Bq.t()) * s
            dv = torch.mm(torch.mm(x, Av.t()), Bv.t()) * s
            z = torch.zeros(x.shape[0], c.Hkv * c.D, dtype=y.dtype, device=y.device)
            y = y + torch.cat([dq, z, dv], -1)
        if self.lora_mods is not None:
            y = y + torch.cat([self._ld(l, "q", x), self._ld(l, "k", x), self._ld(l, "v", x)], -1)
        return y

    def _ld(self, l, name, x):
        """LoRA delta x A^T B^T * scale for one projection (all-projection mode), or None."""
        if self.lora_mods is None:
            return None
        A, B = self.lora_mods[l][name]
        return torch.mm(torch.mm(x, A.t()), B.t()) * self.lora_scale

    def _o(self, l, a):
        y = self.lins[f"l{l}.o"](a)
        if self.lora_mods is not None:
            y = y + self._ld(l, "o", a)
        return y

    def _split(self, qkv, b, n):
        c = self.cfg
        q = qkv[:, : c.Hq * c.D].view(b, n, c.Hq, c.D)
        k = qkv[:, c.Hq * c.D: (c.Hq + c.Hkv) * c.D].view(b, n, c.Hkv, c.D)
        v = qkv[:, (c.Hq + c.Hkv) * c.D:].view(b, n, c.Hkv, c.D)
        return q, k, v

    def _mlp(self, l, h):
        c = self.cfg
        x = rms(h, self.small[f"model.layers.{l}.post_attention_layernorm.weight"], c.eps)
        gu = self.lins[f"l{l}.gate_up"](x)
        g_, u_ = gu[:, : c.I], gu[:, c.I:]
        if self.lora_mods is not None:
            # add the deltas to each half separately (no full-width cat next to gu)
            g_, u_ = g_ + self._ld(l, "gate", x), u_ + self._ld(l, "up", x)
        a = F.silu(g_) * u_
        y = self.lins[f"l{l}.down"](a)
        if self.lora_mods is not None:
            y = y + self._ld(l, "down", a)
        return h + y

    def embed(self, ids):
        e = self.lins["embed"]
        if getattr(e, "sloth", False):
            return e.embed_rows(ids)
        if e.key is None and e.impl == "torch":
            return F.embedding(ids.reshape(-1), e.t)
        return K.embed(ids, e.t, e.key)

    def logits(self, h):
        return self.lm(rms(h, self.small["model.norm.weight"], self.cfg.eps))

    # -- prefill: whole prompts, writes KV, returns last-position hidden
    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, kv, b_off: int, zero_pos: torch.Tensor):
        c = self.cfg
        b, n = ids.shape
        h = self.embed(ids)
        cos = self.cos[:n].view(1, n, 1, c.D)
        sin = self.sin[:n].view(1, n, 1, c.D)
        for l in range(c.L):
            x = rms(h, self.small[f"model.layers.{l}.input_layernorm.weight"], c.eps)
            q, k, v = self._split(self._qkv(l, x), b, n)
            q, k = rope(q, cos, sin), rope(k, cos, sin).contiguous()
            v = v.contiguous()
            K.kv_store(k, v, kv.slabs[l], kv.keys[l], zero_pos, b_off)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                               v.transpose(1, 2), is_causal=True,
                                               enable_gqa=True)
            o = o.transpose(1, 2).reshape(b * n, c.Hq * c.D)
            h = h + self._o(l, o)
            h = self._mlp(l, h)
        return h.view(b, n, c.H)[:, -1]

    # -- one decode step for the whole batch (CUDA-graph safe)
    @torch.no_grad()
    def decode(self, ids, pos, lens, kv):
        c = self.cfg
        B = ids.shape[0]
        h = self.embed(ids)
        cos = self.cos.index_select(0, pos).view(B, 1, 1, c.D)
        sin = self.sin.index_select(0, pos).view(B, 1, 1, c.D)
        for l in range(c.L):
            x = rms(h, self.small[f"model.layers.{l}.input_layernorm.weight"], c.eps)
            q, k, v = self._split(self._qkv(l, x), B, 1)
            q, k = rope(q, cos, sin), rope(k, cos, sin).contiguous()
            v = v.contiguous()
            K.kv_store(k, v, kv.slabs[l], kv.keys[l], pos, 0)
            o = K.attn_decode(q.reshape(B, c.Hq, c.D).contiguous(), kv.slabs[l], kv.keys[l], lens)
            h = h + self._o(l, o.view(B, c.Hq * c.D))
            h = self._mlp(l, h)
        return self.logits(h)

    # -- full-sequence forward with autograd (training / log-probs), no KV cache
    def forward_train(self, ids: torch.Tensor, checkpoint: bool = True):
        c = self.cfg
        b, n = ids.shape
        h = self.embed(ids)
        cos = self.cos[:n].view(1, n, 1, c.D)
        sin = self.sin[:n].view(1, n, 1, c.D)

        def layer(h, l):
            x = rms(h, self.small[f"model.layers.{l}.input_layernorm.weight"], c.eps)
            q, k, v = self._split(self._qkv(l, x), b, n)
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                               v.transpose(1, 2), is_causal=True,
                                               enable_gqa=True)
            h = h + self._o(l, o.transpose(1, 2).reshape(b * n, c.Hq * c.D))
            return self._mlp(l, h)

        for l in range(c.L):
            if checkpoint and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(layer, h, l, use_reentrant=False)
            else:
                h = layer(h, l)
        return h.view(b, n, c.H)


class KV:
    """Per-layer KV slabs [2, B, Hkv, T, D]; keys None for a plain cache."""

    def __init__(self, slabs, keys, arena=None, pages=None):
        self.slabs, self.keys, self.arena, self.pages = slabs, keys, arena, pages

    @staticmethod
    def plain(cfg: Cfg, B: int, T: int, device="cuda"):
        slabs = [torch.zeros(2, B, cfg.Hkv, T, cfg.D, dtype=torch.bfloat16, device=device)
                 for _ in range(cfg.L)]
        return KV(slabs, [None] * cfg.L)

    def reset(self):
        """Sealed cache: every slot back to filler (encode(empty)). Plain: no-op."""
        if self.arena is not None:
            self.arena.refill(*self.pages)


def load_plain(path: str, Tmax: int, impl="torch", device="cuda"):
    cfg = Cfg.from_hf(path)
    st = SafeTensors(path)
    lins = {}
    for name, shape, src in big_tensors(cfg):
        t = st.concat(src).to(device=device, dtype=torch.bfloat16).contiguous()
        assert tuple(t.shape) == tuple(shape), (name, t.shape, shape)
        lins[name] = Lin(t, None, impl)
    small = {n: st.get(n).to(device=device, dtype=torch.bfloat16) for n in small_tensors(cfg)}
    return Model(cfg, lins, small, Tmax, device)
