"""CPU checks for the max-param shard: blocked MLP == one-piece MLP, flushed host gradients == autograd gradients
(nothing kept on the device), the optimiser step lands the host master in the device weights, the stash mode sets the
link bytes, idle does no work.

    GSHARD_NO_GATE=1 python -m pytest -q schemes/memory_link_exhaustion/attacks/max_shard/test_cpu.py
"""
import copy

import torch
import torch.utils.checkpoint as ckpt

from schemes.memory_link_exhaustion.attacks.gated_shard import workload as GW
from schemes.memory_link_exhaustion.attacks.max_shard import workload as W

SMALL = dict(d=64, heads=8, kv=4, hd=8, ffn=96, theta=1e6, eps=1e-6)


def _rope(T, hd):
    inv = 1.0 / (1e6 ** (torch.arange(0, hd, 2).float() / hd))
    ang = torch.outer(torch.arange(T).float(), inv)
    ang = torch.cat((ang, ang), -1)
    return ang.cos(), ang.sin()


def _copy_into_plain(b: W.BlockLayer) -> GW.Layer:
    a = GW.Layer(b.c, chunk=10_000, head_groups=1).double()
    with torch.no_grad():
        for n in ("ln1", "ln2", "wq", "bq", "wk", "bk", "wv", "bv", "wo"):
            getattr(a, n).copy_(getattr(b, n))
        a.wg.copy_(torch.cat(list(b.wg), 0))
        a.wu.copy_(torch.cat(list(b.wu), 0))
        a.wd.copy_(torch.cat(list(b.wd), 1))
    return a


def test_blocked_layer_matches_plain():
    torch.manual_seed(0)
    b = W.BlockLayer(SMALL, nb=4, head_groups=2).double()
    b.init_()
    a = _copy_into_plain(b)
    cos, sin = (t.double() for t in _rope(17, SMALL["hd"]))
    x = torch.randn(1, 17, SMALL["d"], dtype=torch.float64)
    ya = a(x, cos, sin)
    yb = b(x, cos, sin)
    assert torch.allclose(ya, yb, atol=1e-10)


def test_flushed_grads_match_autograd():
    torch.manual_seed(1)
    spec = {"arch": {k: v for k, v in SMALL.items()}, "ffns": [96, 48], "seq": 16, "global_tokens": 10 ** 9,
            "link_gbs": 100.0, "mlp_blocks": 4, "head_groups": 2, "stash": "host"}
    s = W.MaxShard(spec, device="cpu")
    ref = copy.deepcopy(s.layers)
    for p in ref.parameters():
        p.grad = None
    torch.manual_seed(5)
    s.step()
    s.flush.drain()
    # every device grad was taken by the flusher
    assert all(p.grad is None for p in s.params)
    assert s.flush.flushes == len(s.params)
    # replay the same micro-batch on a plain copy (hooks absent) with the same random draws
    torch.manual_seed(5)
    x = torch.randn(1, s.T, s.arch["d"], dtype=s.dt, requires_grad=True)
    h = x
    for l in ref:
        h = ckpt.checkpoint(l, h, s.cos, s.sin, use_reentrant=False)
    h.backward(torch.randn_like(h) * (1.0 / s.accum))
    for p, a in zip(ref.parameters(), s.flush.acc):
        assert torch.allclose(p.grad.float(), a, atol=1e-6, rtol=1e-5)
    s.close()


def test_opt_step_updates_device_weights_and_link_bytes():
    torch.manual_seed(2)
    spec = {"arch": dict(SMALL), "ffns": [96], "seq": 8, "global_tokens": 16, "link_gbs": 100.0,
            "mlp_blocks": 2, "head_groups": 1, "stash": "host"}
    s = W.MaxShard(spec, device="cpu")
    assert s.accum == 2 and s.in_b == 3 * 10240 and s.inflight_bytes == 0
    before = [p.detach().clone() for p in s.params]
    s.step()
    s.step()
    assert s.opt_steps == 1
    assert any(not torch.equal(b, p) for b, p in zip(before, s.params))
    for p, m in zip(s.params, s.master):
        assert torch.equal(p.detach(), m.detach().to(p.dtype))
    d = s.describe()
    assert s.flush.moved > 0 and d["flush_err"] is None
    s.close()
    g = W.MaxShard(dict(spec, stash="gpu", inflight="avg"), device="cpu")
    assert g.in_b == 2 * 10240
    assert g.inflight_bytes == int(W.MODEL_PARAMS / g.P / 2 * g.T * 10240)
    g.close()


def test_idle_does_nothing():
    s = W.MaxShard({"arch": dict(SMALL), "ffns": [96], "seq": 8, "idle": True, "mlp_blocks": 2}, device="cpu")
    assert s.step() == 0 and s.k == 0
    s.close()


def test_grad_on_gpu_matches_flush():
    spec = {"arch": dict(SMALL), "ffns": [96], "seq": 8, "global_tokens": 16, "link_gbs": 100.0,
            "mlp_blocks": 2, "head_groups": 1, "stash": "host"}
    res = []
    for gpu in (False, True):
        torch.manual_seed(7)
        s = W.MaxShard(dict(spec, grad_on_gpu=gpu), device="cpu")
        torch.manual_seed(8)
        s.step()
        s.step()
        assert s.opt_steps == 1
        res.append([p.detach().clone() for p in s.params])
        if gpu:
            assert all(torch.count_nonzero(p.grad) == 0 for p in s.params)
        s.close()
    for a, b in zip(*res):
        assert torch.allclose(a, b, atol=1e-6)


def test_shrink_steps_seq_down_and_still_runs():
    spec = {"arch": dict(SMALL), "ffns": [96, 48], "seq": 32, "global_tokens": 10 ** 6,
            "link_gbs": 100.0, "mlp_blocks": 4, "head_groups": 2, "seq_floor": 8}
    s = W.MaxShard(spec, device="cpu")
    assert s.T == 32 and s.accum == 10 ** 6 // 32
    assert s._shrink() and s.T == 16 and s.cos.shape[0] == 16 and s.accum == 10 ** 6 // 16 and s.k == 0
    assert s._shrink() and s.T == 8                      # halving lands on the floor
    assert not s._shrink() and s.T == 8                  # at the floor there is nothing smaller to try
    assert s.step() == 8 and s.k == 1                    # a real micro-batch still runs at the shrunk length
    s.close()


def test_oom_in_step_shrinks_instead_of_dying(monkeypatch):
    spec = {"arch": dict(SMALL), "ffns": [96], "seq": 32, "global_tokens": 10 ** 6,
            "link_gbs": 100.0, "mlp_blocks": 2, "head_groups": 1, "seq_floor": 8}
    s = W.MaxShard(spec, device="cpu")
    calls = {"n": 0}
    real = W.ckpt.checkpoint

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:                              # first layer of the first micro-batch "OOMs"
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
        return real(*a, **k)
    monkeypatch.setattr(W.ckpt, "checkpoint", flaky)
    assert s.step() == 0 and s.shrinks == 1 and s.T == 16   # caught, stepped down, did not raise
    assert s.last_oom and "seq 32" in s.last_oom and "Tried to allocate" in s.last_oom
    assert s.step() == 16 and s.k == 1                      # the retry at the shorter sequence runs
    s.close()


def test_oom_in_opt_step_is_guarded_and_floor_raises_cleanly(monkeypatch):
    spec = {"arch": dict(SMALL), "ffns": [96], "seq": 16, "global_tokens": 16, "link_gbs": 100.0,
            "mlp_blocks": 2, "head_groups": 1, "seq_floor": 8}
    s = W.MaxShard(spec, device="cpu")
    real = s._opt_step
    n = {"k": 0}

    def flaky():
        n["k"] += 1
        if n["k"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 32.00 MiB")
        return real()
    monkeypatch.setattr(s, "_opt_step", flaky)
    assert s.step() == 0 and s.T == 8 and s.shrinks == 1          # OOM in the optimiser step is caught too
    assert all(p.grad is None for p in s.params) and s.k == 0     # nothing left behind by the failed step
    assert s.step() == 8 and s.step() == 8 and s.opt_steps == 1   # runs on at the shorter sequence

    def always(*a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 MiB")
    monkeypatch.setattr(s, "_micro_batch", always)
    try:
        s.step()
        assert False
    except RuntimeError as e:
        assert "seq floor" in str(e) and len(s.oom_log) == 2
    s.close()


def test_token_chunked_layer_matches_unchunked():
    """tchunk runs query positions in checkpointed chunks against the full causal K/V prefix; same
    outputs and same gradients (inputs and every parameter) as the unchunked layer, for chunks that do and do not
    divide T, through an outer checkpoint as in MaxShard.step()."""
    torch.manual_seed(3)
    T = 17
    cos, sin = (t.double() for t in _rope(T, SMALL["hd"]))
    base = W.BlockLayer(SMALL, nb=4, head_groups=2).double()
    base.init_()
    x0 = torch.randn(1, T, SMALL["d"], dtype=torch.float64)
    gout = torch.randn(1, T, SMALL["d"], dtype=torch.float64)

    def run(tc):
        b = copy.deepcopy(base)
        b.tchunk = tc
        x = x0.clone().requires_grad_(True)
        y = ckpt.checkpoint(b, x, cos, sin, use_reentrant=False)
        y.backward(gout)
        return y.detach(), x.grad.detach(), [p.grad.detach().clone() for p in b.parameters()]

    y0, gx0, gp0 = run(0)
    for tc in (1, 4, 5, 16, 17):
        y, gx, gp = run(tc)
        assert torch.allclose(y, y0, atol=1e-10), tc
        assert torch.allclose(gx, gx0, atol=1e-10), tc
        assert all(torch.allclose(a, b, atol=1e-10) for a, b in zip(gp, gp0)), tc


def test_maxshard_step_with_tchunk_runs():
    spec = {"arch": {k: v for k, v in SMALL.items()}, "ffns": [96], "seq": 16, "global_tokens": 32,
            "link_gbs": 100.0, "mlp_blocks": 4, "head_groups": 2, "stash": "host", "tchunk": 4}
    s = W.MaxShard(spec, device="cpu")
    assert s.layers[0].tchunk == 4 and s.describe()["tchunk"] == 4
    assert s.step() == 16
    s.step()
    assert s.opt_steps == 1


def test_fused_norm_matches_declared_fp32_path_bitwise():
    """(declared precision): F.rms_norm on the bf16 input equals the declared path (cast to fp32,
    normalise, multiply by the weight in fp32, cast to bf16) bit for bit at the real width."""
    torch.manual_seed(0)
    for scale in (1.0, 3.0):
        x = (torch.randn(256, 5120) * scale).to(torch.bfloat16)
        w = (1 + 0.1 * torch.randn(5120)).to(torch.bfloat16)
        xf = x.float()
        decl = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(torch.bfloat16)
        assert torch.equal(W._rms_fused(x, w, 1e-6), decl), scale
