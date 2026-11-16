"""CPU checks for the gated shard (no GPU): chunking does not change the maths, gates fire in forward and backward,
the pacer moves what is owed, an optimiser step updates the device weights from the host master.

    GSHARD_NO_GATE=1 python -m pytest -q schemes/memory_link_exhaustion/attacks/gated_shard/test_cpu.py
"""
import copy
import time

import torch

from schemes.memory_link_exhaustion.attacks.gated_shard import gate as G
from schemes.memory_link_exhaustion.attacks.gated_shard import workload as W

SMALL = dict(d=64, heads=8, kv=4, hd=8, ffn=96, theta=1e6, eps=1e-6)


def _rope(T, hd):
    inv = 1.0 / (1e6 ** (torch.arange(0, hd, 2).float() / hd))
    ang = torch.outer(torch.arange(T).float(), inv)
    ang = torch.cat((ang, ang), -1)
    return ang.cos(), ang.sin()


def test_chunking_is_exact():
    torch.manual_seed(0)
    a = W.Layer(SMALL, chunk=10_000, head_groups=1)
    a.init_()
    b = copy.deepcopy(a)
    b.chunk, b.hg = 7, 4
    x = torch.randn(2, 33, SMALL["d"], dtype=torch.float64, requires_grad=True)
    a, b = a.double(), b.double()
    cos, sin = (t.double() for t in _rope(33, SMALL["hd"]))
    ya, yb = a(x, cos, sin), b(x, cos, sin)
    assert torch.allclose(ya, yb, atol=1e-10)
    ga = torch.autograd.grad(ya.pow(2).mean(), list(a.parameters()) + [x])
    gb = torch.autograd.grad(yb.pow(2).mean(), list(b.parameters()) + [x])
    for u, v in zip(ga, gb):
        assert torch.allclose(u, v, atol=1e-10)


def test_gates_fire_forward_and_backward():
    torch.manual_seed(0)
    L = W.Layer(SMALL, chunk=8, head_groups=2)
    L.init_()
    x = torch.randn(1, 16, SMALL["d"], requires_grad=True)
    cos, sin = _rope(16, SMALL["hd"])
    G.STATS["waits"] = 0
    y = L(x, cos, sin)
    fwd = G.STATS["waits"]
    y.sum().backward()
    assert fwd >= 10 and G.STATS["waits"] >= 2 * fwd - 2


def test_shard_step_and_pacer():
    spec = {"arch": SMALL, "seq": 16, "mb": 2, "global_tokens": 64, "link_gbs": 1.0, "chunk_mib": 0.001,
            "in_bytes_tok": 100, "out_bytes_tok": 50}
    wl = W.GatedShard(spec)
    w0 = [p.detach().clone() for p in wl.layer.parameters()]
    n = sum(wl.step() for _ in range(wl.accum))
    assert n == 64 and wl.opt_steps == 1
    assert any(not torch.equal(a, p.detach()) for a, p in zip(w0, wl.layer.parameters()))
    for p, m in zip(wl.layer.parameters(), wl.master):
        assert torch.allclose(p.detach(), m.detach().to(p.dtype))
    t0 = time.time()
    while wl.pacer.debt() > 0 and time.time() - t0 < 5:
        time.sleep(0.01)
    pbytes = sum(p.numel() * p.element_size() for p in wl.layer.parameters())
    assert wl.pacer.moved["h2d"] >= 64 * 100 + pbytes
    assert wl.pacer.moved["d2h"] >= 64 * 50 + pbytes
    wl.close()
