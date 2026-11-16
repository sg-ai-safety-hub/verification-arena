"""CPU checks for the link-free local RL on a tiny random Qwen2-shaped model: shared-prefix / shared-prompt KV
generation gives the same logits as a full forward, the streamed log-prob and two-level checkpointing give the same
LoRA gradients as plain autograd over full logits, host AdamW updates the device LoRA, idle does nothing.

    GSHARD_NO_GATE=1 python -m pytest -q schemes/memory_link_exhaustion/attacks/local_rl/test_cpu.py
"""
import torch

from schemes.memory_link_exhaustion.attacks.local_rl import workload as W
from schemes.memory_link_exhaustion.core.model import Cfg, Lin, Model


def tiny(seed=0, L=4):
    torch.manual_seed(seed)
    c = Cfg(H=64, Hq=8, Hkv=4, D=8, I=96, L=L, V=300, eps=1e-6, theta=1e4, tied=False, bias=True)
    dt = torch.float64
    r = lambda *s: torch.randn(*s, dtype=dt) * 0.1
    lins = {"embed": Lin(r(c.V, c.H), None), "lm_head": Lin(r(c.V, c.H), None)}
    small = {"model.norm.weight": 1 + r(c.H)}
    for l in range(L):
        lins[f"l{l}.qkv"] = Lin(r((c.Hq + 2 * c.Hkv) * c.D, c.H), None)
        lins[f"l{l}.o"] = Lin(r(c.H, c.Hq * c.D), None)
        lins[f"l{l}.gate_up"] = Lin(r(2 * c.I, c.H), None)
        lins[f"l{l}.down"] = Lin(r(c.H, c.I), None)
        p = f"model.layers.{l}."
        small[p + "input_layernorm.weight"] = 1 + r(c.H)
        small[p + "post_attention_layernorm.weight"] = 1 + r(c.H)
        small[p + "self_attn.q_proj.bias"] = r(c.Hq * c.D)
        small[p + "self_attn.k_proj.bias"] = r(c.Hkv * c.D)
        small[p + "self_attn.v_proj.bias"] = r(c.Hkv * c.D)
    m = Model(c, lins, small, 64, "cpu")
    m.cos, m.sin = m.cos.double(), m.sin.double()
    return m


def _rl(m, **kw):
    spec = dict(n_prompts=4, group=3, conc_prompts=2, prompt=12, gen=6, shared_prefix=0, seg=2, train_mb=2)
    spec.update(kw)
    rl = W.LocalRL(m, spec, device="cpu")
    with torch.no_grad():                     # non-zero LoRA B so the adapter matters
        for p in rl.params:
            p.add_(torch.randn_like(p) * 0.05)
    return rl


def _full_logits(rl, seq):
    h = rl.forward_train(seq, checkpoint=False)
    return rl.m.logits(h.reshape(-1, rl.m.cfg.H)).view(seq.shape[0], seq.shape[1], -1)


def _check_generation(L0):
    m = tiny(1)
    rl = _rl(m, shared_prefix=L0)
    Pc, g = 2, rl.group
    ids1 = torch.randint(5, 290, (Pc, rl.P - L0))
    force = torch.randint(5, 290, (Pc * g, rl.G))
    out, logits = rl._generate(ids1, force=force, want_logits=True)
    assert torch.equal(out, force)
    pre = torch.cat([rl.prefix_ids.view(1, -1).expand(Pc, -1), ids1], 1) if L0 else ids1
    seq = torch.cat([pre.repeat_interleave(g, 0), force], 1)
    with torch.no_grad():
        full = _full_logits(rl, seq)
    for t, lg in enumerate(logits):
        assert torch.allclose(lg, full[:, rl.P - 1 + t], atol=1e-9), t


def test_dec_graph_matches_shared_generation():
    """The fixed-shape decode (dec_graph: full G-length cache with a position mask), without CUDA-graph capture on
    CPU, produces the same per-token logits as the growing-slice shared decode."""
    for L0 in (0, 4):
        m = tiny(1)
        rl = _rl(m, shared_prefix=L0, dec_attn="shared")
        gr = W.LocalRL(m, dict(n_prompts=4, group=3, conc_prompts=2, prompt=12, gen=6, shared_prefix=L0, seg=2,
                               train_mb=2, dec_graph=True), device="cpu")
        gr.params, gr.ad = rl.params, rl.ad           # same adapters
        Pc, g = 2, rl.group
        ids1 = torch.randint(5, 290, (Pc, rl.P - L0))
        force = torch.randint(5, 290, (Pc * g, rl.G))
        _, lg_s = rl._generate(ids1, force=force, want_logits=True)
        _, lg_g = gr._generate(ids1, force=force, want_logits=True)
        for t, (a, b) in enumerate(zip(lg_s, lg_g)):
            assert torch.allclose(a, b, atol=1e-9), (L0, t, (a - b).abs().max().item())


def test_generation_shared_prompt_matches_full_forward():
    _check_generation(0)


def test_generation_shared_prefix_matches_full_forward():
    _check_generation(5)


def test_streamed_logprob_and_checkpointing_match_plain_autograd():
    m = tiny(2)
    rl = _rl(m, host_adam=False)
    seq = torch.randint(5, 290, (2, rl.T))
    lp = rl.logprobs(seq, checkpoint=True)
    lp.sum().backward()
    g1 = [p.grad.clone() for p in rl.params]
    for p in rl.params:
        p.grad = None
    full = _full_logits(rl, seq[:, :-1])
    ref = torch.log_softmax(full[:, rl.P - 1:], -1).gather(-1, seq[:, rl.P:, None])[..., 0].mean(1)
    assert torch.allclose(lp, ref, atol=1e-9)
    ref.sum().backward()
    for a, b in zip(g1, (p.grad for p in rl.params)):
        assert torch.allclose(a, b, atol=1e-9)


def test_steps_and_host_adam():
    m = tiny(3, L=2)
    rl = _rl(m, host_adam=True)
    before = [p.detach().clone() for p in rl.params]
    assert rl.step() == 6 and rl.opt_steps == 0
    assert rl.step() == 6 and rl.opt_steps == 1
    assert any(not torch.equal(b, p) for b, p in zip(before, rl.params))
    for p, ms in zip(rl.params, rl.master):
        assert torch.equal(p.detach(), ms.detach().to(p.dtype))
    d = rl.describe()
    assert d["rollouts_in_flight"] == 6 and d["link_mb"] > 0


def test_opt_every_calls_times_the_step_apart():
    """opt_every_calls=1 (measurement cadence): an optimiser step after every call, timed apart from the call, and
    the real cadence still reported so a whole step can be modelled as calls_per_step x call + opt."""
    m = tiny(3, L=2)
    rl = _rl(m, host_adam=True, opt_every_calls=1)
    assert rl.step() == 6 and rl.opt_steps == 1
    assert rl.step() == 6 and rl.opt_steps == 2
    d = rl.describe()
    assert d["calls_per_step"] == 2 and d["opt_every_calls"] == 1
    assert len(d["opt_s"]) == 2 and len(d["step_s"]) == 2 and all(x >= 0 for x in d["opt_s"])
    assert _rl(tiny(3, L=2), host_adam=True).describe()["opt_every_calls"] == 2      # default = the real cadence


def test_idle():
    rl = _rl(tiny(4, L=1), idle=True)
    assert rl.step() == 0 and rl.calls == 0


ALL7 = ["q", "k", "v", "o", "gate", "up", "down"]


def test_all_projection_adapters_equal_merged_weights():
    m = tiny(5, L=2)
    rl = _rl(m, targets=ALL7, rank=2, host_adam=False)
    c = m.cfg
    s = rl.scale
    # merged model: W' = W + s B A on every projection, no adapters
    mm = tiny(5, L=2)
    with torch.no_grad():
        for l in range(c.L):
            ad = {t: (A.double(), B.double()) for t, (A, B) in rl.ad[l].items()}
            d = lambda t: s * ad[t][1] @ ad[t][0]
            q0, k0 = c.Hq * c.D, c.Hkv * c.D
            W = mm.lins[f"l{l}.qkv"].t
            W[:q0] += d("q"); W[q0:q0 + k0] += d("k"); W[q0 + k0:] += d("v")
            mm.lins[f"l{l}.o"].t += d("o")
            Wgu = mm.lins[f"l{l}.gate_up"].t
            Wgu[:c.I] += d("gate"); Wgu[c.I:] += d("up")
            mm.lins[f"l{l}.down"].t += d("down")
    seq = torch.randint(5, 290, (2, 15))
    with torch.no_grad():
        a = _full_logits(rl, seq)
        h = mm.forward_train(seq, checkpoint=False)
        b = mm.logits(h.reshape(-1, c.H)).view(2, 15, -1)
    assert torch.allclose(a, b, atol=1e-9)
    assert rl.lora_params == c.L * 2 * sum(i + o for i, o in [(64, 64), (64, 32), (64, 32), (64, 64), (64, 96), (64, 96), (96, 64)])


def test_all_projection_generation_and_grads():
    m = tiny(6, L=2)
    rl = _rl(m, targets=ALL7, rank=2, host_adam=False)
    Pc, g = 2, rl.group
    ids1 = torch.randint(5, 290, (Pc, rl.P))
    force = torch.randint(5, 290, (Pc * g, rl.G))
    out, logits = rl._generate(ids1, force=force, want_logits=True)
    seq = torch.cat([ids1.repeat_interleave(g, 0), force], 1)
    with torch.no_grad():
        full = _full_logits(rl, seq)
    for t, lg in enumerate(logits):
        assert torch.allclose(lg, full[:, rl.P - 1 + t], atol=1e-9), t
    s2 = seq[:2]
    rl.logprobs(s2, checkpoint=True).sum().backward()
    g1 = [p.grad.clone() for p in rl.params]
    for p in rl.params:
        p.grad = None
    f = _full_logits(rl, s2[:, :-1])
    torch.log_softmax(f[:, rl.P - 1:], -1).gather(-1, s2[:, rl.P:, None]).mean((1, 2)).sum().backward()
    assert all(torch.allclose(a, p.grad, atol=1e-9) for a, p in zip(g1, rl.params))


def test_all_projection_host_adam_flush():
    m = tiny(7, L=2)
    rl = _rl(m, targets=ALL7, rank=2, host_adam=True, link_gbs=100.0)
    rl.step()
    rl.flush.drain()
    assert all(p.grad is None for p in rl.params) and rl.flush.flushes == 3 * len(rl.params)
    rl.step()
    assert rl.opt_steps == 1
    for p, ms in zip(rl.params, rl.master):
        assert torch.equal(p.detach(), ms.detach().to(p.dtype))
    rl.close()


def test_stagger_rows_match_full_forward_and_overlap():
    m = tiny(8, L=2)
    rl = _rl(m, targets=ALL7, rank=2, host_adam=False, stagger=True, n_prompts=5, conc_prompts=1, group=2, prompt=6, gen=7,
             kv_chunk=2, train_mb=3, lr=0.0)
    tok_b = rl._kv_tok()
    # budget for ~2.5 groups at peak (P + g * G tokens each)
    rl.kv_budget = 2.2 * (6 + 2 * 8) * tok_b
    rl.debug_logits = True
    ages_seen = []
    orig = rl._decode_all

    def spy(groups):
        ages_seen.append(sorted(gr["age"] for gr in groups))
        return orig(groups)
    rl._decode_all = spy
    n = 0
    while not rl.cycles:
        n += rl.step()
    assert n == 10 and rl.opt_steps == 1
    assert max(len(a) for a in ages_seen) >= 2                       # several groups share decode steps
    assert any(len(set(a)) > 1 for a in ages_seen)                   # at different ages (staggered)
    assert len(rl.last_done) == 5
    for seq, lgs in rl.last_done:
        with torch.no_grad():
            full = _full_logits(rl, seq)
        for t, lg in enumerate(lgs):
            assert torch.allclose(lg, full[:, rl.P - 1 + t], atol=1e-9), t


def test_chunked_mlp_matches_one_shot_logits_and_grads():
    m = tiny(9, L=2)
    rl = _rl(m, targets=ALL7, rank=2, host_adam=False, seg=1)
    seq = torch.randint(5, 290, (2, rl.T))

    def run(chunk):
        rl.mlp_chunk = chunk
        for p in rl.params:
            p.grad = None
        lp = rl.logprobs(seq, checkpoint=True)
        lp.sum().backward()
        return lp.detach().clone(), [p.grad.clone() for p in rl.params]

    lp_c, g_c = run(5)                      # token-chunked MLP (34 update rows in chunks of 5)
    lp_0, g_0 = run(0)                       # one-shot MLP
    assert torch.allclose(lp_c, lp_0, atol=1e-9)
    assert all(torch.allclose(a, b, atol=1e-9) for a, b in zip(g_c, g_0))
    rl.mlp_chunk = 5                          # forward logits match a one-shot full forward too
    with torch.no_grad():
        a = _full_logits(rl, seq[:, :-1])
    rl.mlp_chunk = 0
    with torch.no_grad():
        b = _full_logits(rl, seq[:, :-1])
    assert torch.allclose(a, b, atol=1e-9)


def test_gen_in_flight_subbatch_matches_full_generation():
    m = tiny(10, L=2)
    rl = _rl(m, targets=ALL7, rank=2, host_adam=False, group=2, prompt=6, gen=5)
    Pc, g = 4, rl.group
    ids1 = torch.randint(5, 290, (Pc, rl.P))
    force = torch.randint(5, 290, (Pc * g, rl.G))
    _, lg_full = rl._generate(ids1, force=force, want_logits=True)
    pstep = 2                                 # two prompts per generation sub-batch
    chunks = [rl._generate(ids1[i:i + pstep], force=force[i * g:(i + pstep) * g], want_logits=True)[1]
              for i in range(0, Pc, pstep)]
    for t in range(rl.G):
        sub = torch.cat([ch[t] for ch in chunks], 0)
        assert torch.allclose(lg_full[t], sub, atol=1e-9), t


def test_step_with_chunk_and_gen_in_flight_runs():
    m = tiny(11, L=2)
    rl = _rl(m, targets=ALL7, rank=2, host_adam=True, seg=1, mlp_chunk=5, gen_in_flight=2,
             n_prompts=2, conc_prompts=2, group=2, prompt=6, gen=4, train_mb=1)
    assert rl.step() == 4 and rl.opt_steps == 1
    for p, ms in zip(rl.params, rl.master):
        assert torch.equal(p.detach(), ms.detach().to(p.dtype))
    rl.close()


def test_adapter_streaming_matches_resident_logits_and_grads():
    """Per-layer adapter streaming (hold no adapters resident; stream each layer in and free it) gives bit-identical
    logprobs and bit-identical per-adapter gradients to the resident path, across seg / lm_chunk / mlp_chunk."""
    m = tiny(13, L=4)
    res = _rl(m, targets=ALL7, rank=2, host_adam=False, seg=1)                # resident, grads kept on the params
    st = _rl(tiny(13, L=4), targets=ALL7, rank=2, host_adam=True, seg=1, stream_adapters=True)
    assert st.params == [] and st.ad is None and st.stream_adapters            # the 268 MB floor is gone
    vals = [res.ad[l][t][w].detach().clone() for l in range(m.cfg.L) for t in res.targets for w in (0, 1)]
    for k, v in enumerate(vals):                                              # same adapters in the host buffers
        st.master[k].data.copy_(v.double())
        st.hbf[k].copy_(v)
    seq = torch.randint(5, 290, (2, res.T))

    for seg, lc, mc in [(1, 0, 0), (1, 3, 5), (2, 0, 4), (4, 5, 0)]:
        res.seg, res.lm_chunk, res.mlp_chunk = seg, lc, mc
        st.seg, st.lm_chunk, st.mlp_chunk = seg, lc, mc
        for p in res.params:
            p.grad = None
        lp_r = res.logprobs(seq, checkpoint=True)
        lp_r.sum().backward()
        g_r = [p.grad.clone() for p in res.params]
        for a in st.sacc:
            a.zero_()
        st._dev_ad.clear()
        lp_s = st.logprobs(seq, checkpoint=True)
        lp_s.sum().backward()
        assert torch.allclose(lp_r, lp_s, atol=1e-9), (seg, lc, mc)
        # streamed grads land in the fp32 host accumulator; compare to the resident grads cast to fp32 (same value)
        assert all(torch.allclose(a.float(), b, atol=1e-9) for a, b in zip(g_r, st.sacc)), (seg, lc, mc)
        assert not st._dev_ad                                                 # every streamed layer was freed
    assert st.stream_in_bytes > 0 and st.grad_out_bytes > 0


def test_adapter_streaming_generation_and_opt_step():
    """Streaming generation matches the resident forward, and a streaming optimiser step moves the host adapters."""
    m = tiny(14, L=2)
    cfg = dict(targets=ALL7, rank=2, host_adam=True, seg=1, mlp_chunk=5,
               n_prompts=2, conc_prompts=2, group=2, prompt=6, gen=4, train_mb=1)
    res = _rl(m, **cfg)
    st = _rl(tiny(14, L=2), stream_adapters=True, **cfg)
    vals = [res.ad[l][t][w].detach().clone() for l in range(m.cfg.L) for t in res.targets for w in (0, 1)]
    for k, v in enumerate(vals):
        st.master[k].data.copy_(v.double())
        st.hbf[k].copy_(v)
    Pc, g = 2, res.group
    ids1 = torch.randint(5, 290, (Pc, res.P))
    force = torch.randint(5, 290, (Pc * g, res.G))
    _, lg_r = res._generate(ids1, force=force, want_logits=True)
    _, lg_s = st._generate(ids1, force=force, want_logits=True)
    for t in range(res.G):
        assert torch.allclose(lg_r[t], lg_s[t], atol=1e-9), t
    assert not st._dev_ad                                                     # generation freed every streamed layer
    before = [h.clone() for h in st.hbf]
    assert st.step() == 4 and st.opt_steps == 1                              # an optimiser step updates the host bf16
    assert any(not torch.equal(b, h) for b, h in zip(before, st.hbf))
    st.close()


def test_lm_chunk_and_seg_are_bit_identical():
    m = tiny(12, L=4)
    rl = _rl(m, targets=ALL7, rank=2, host_adam=False, seg=1)
    seq = torch.randint(5, 290, (2, rl.T))

    def run(seg, lc, mc):
        rl.seg, rl.lm_chunk, rl.mlp_chunk = seg, lc, mc
        for p in rl.params:
            p.grad = None
        lp = rl.logprobs(seq, checkpoint=True)
        lp.sum().backward()
        return lp.detach().clone(), [p.grad.clone() for p in rl.params]

    a_lp, a_g = run(1, 0, 0)                     # reference: seg 1, one-shot LM head and MLP
    for seg, lc, mc in [(1, 3, 0), (2, 0, 0), (4, 5, 5), (2, 3, 4)]:
        lp, g = run(seg, lc, mc)                 # any seg / lm_chunk / mlp_chunk gives the same logprobs and grads
        assert torch.allclose(a_lp, lp, atol=1e-9), (seg, lc, mc)
        assert all(torch.allclose(x, y, atol=1e-9) for x, y in zip(a_g, g)), (seg, lc, mc)


def test_subgroup_generation_matches_grouped():
    """generating a group's rollouts s < group at a time gives the grouped path's tokens and
    (teacher-forced) logits, in the same order; step() runs with gen_in_flight below one group."""
    m = tiny(12, L=2)
    for L0 in (0, 3):
        rl = _rl(m, targets=ALL7, rank=2, host_adam=False, group=4, prompt=6, gen=5, shared_prefix=L0)
        Pc, g = 2, rl.group
        ids1 = torch.randint(5, 290, (Pc, rl.P - rl.L0))
        force = torch.randint(5, 290, (Pc * g, rl.G))
        out_full, lg_full = rl._generate(ids1, force=force, want_logits=True)
        for s in (1, 2, 3):
            out_s, lg_s = rl._generate_sub(ids1, s, force=force, want_logits=True)
            assert torch.equal(out_full, out_s), (L0, s)
            for t in range(rl.G):
                assert torch.allclose(lg_full[t], lg_s[t], atol=1e-9), (L0, s, t)
        torch.manual_seed(5)
        a = rl._generate(ids1[:1])
        torch.manual_seed(5)
        b = rl._generate_sub(ids1[:1], 4)          # s = group: one sub-batch, same sampling stream as grouped
        assert torch.equal(a, b), L0
    rl = _rl(m, targets=ALL7, rank=2, host_adam=True, seg=1, mlp_chunk=5, gen_in_flight=1,
             n_prompts=2, conc_prompts=2, group=4, prompt=6, gen=4, train_mb=1)
    assert rl.step() == 8 and rl.opt_steps == 1
    rl.close()


def test_phase_split_matches_streaming_and_resident():
    """Phase split: all adapters cached for generation, streamed per segment in the update. Against
    the per-layer streaming path (red f68862d) it is bit-identical: logits, sampled tokens, and the updated host
    adapters after an optimiser step; against the all-resident path the logits agree to 1e-6."""
    m = tiny(15, L=2)
    cfg = dict(targets=ALL7, rank=2, host_adam=True, seg=1, mlp_chunk=5, gen_in_flight=1,
               n_prompts=2, conc_prompts=2, group=2, prompt=6, gen=4, train_mb=1)
    res = _rl(m, **cfg)
    sp = _rl(tiny(15, L=2), stream_adapters=True, **cfg)
    ps = _rl(tiny(15, L=2), stream_adapters=True, phase_split=True, **cfg)
    assert ps.phase_split and not sp.phase_split
    vals = [res.ad[l][t][w].detach().float().clone() for l in range(m.cfg.L) for t in res.targets for w in (0, 1)]
    for r in (sp, ps):
        for k, v in enumerate(vals):
            r.master[k].data.copy_(v)
            r.hbf[k].copy_(v.to(r.hbf[k].dtype))
    ids1 = torch.randint(5, 290, (2, res.P))
    force = torch.randint(5, 290, (2 * res.group, res.G))
    _, lg_r = res._generate(ids1, force=force, want_logits=True)
    _, lg_s = sp._generate(ids1, force=force, want_logits=True)
    ps._build_gen_cache()
    _, lg_p = ps._generate(ids1, force=force, want_logits=True)
    ps._drop_gen_cache()
    for t in range(res.G):
        assert torch.equal(lg_s[t], lg_p[t]), t
        assert torch.allclose(lg_r[t], lg_p[t], atol=1e-6), t
    torch.manual_seed(7)
    assert sp.step() == 4
    torch.manual_seed(7)
    assert ps.step() == 4
    assert sp.opt_steps == ps.opt_steps == 1 and ps._gen_cache is None and not ps._dev_ad
    moved = 0
    for k in range(len(vals)):
        assert torch.equal(sp.master[k].detach(), ps.master[k].detach()), k
        assert torch.equal(sp.hbf[k], ps.hbf[k]), k
        moved += int(not torch.equal(ps.master[k].detach(), vals[k]))
    assert moved > 0
    assert ps.cache_in_bytes > 0 and ps.stream_in_bytes > ps.cache_in_bytes and ps.grad_out_bytes > 0
    for r in (res, sp, ps):
        r.close()


def test_fused_norm_matches_core_rms_bitwise():
    """(declared precision): local_rl's fused norm equals core.model.rms (fp32 normalise, bf16 cast,
    then the bf16 weight) bit for bit at the real width."""
    from schemes.memory_link_exhaustion.core import model as CM
    from schemes.memory_link_exhaustion.attacks.local_rl import workload as LW
    torch.manual_seed(0)
    for scale in (1.0, 3.0):
        x = (torch.randn(256, 5120) * scale).to(torch.bfloat16)
        w = (1 + 0.1 * torch.randn(5120)).to(torch.bfloat16)
        assert torch.equal(LW.rms(x, w, 1e-6), CM.rms(x, w, 1e-6)), scale


def test_slice_mlp_matches_chunked_and_one_shot():
    """SliceMLP (mlp_slice): forward bitwise equal to the one-shot MLP, logprobs and per-adapter gradients equal to the
    token-chunked path to rounding (fp64), for all seven targets and for q/v only; streaming adapters too."""
    for targets, stream in ((ALL7, False), (["q", "v"], False), (ALL7, True)):
        m = tiny(11, L=2)
        rl = _rl(m, targets=targets, rank=2, host_adam=stream, seg=1, stream_adapters=stream)
        if stream:                              # non-zero host adapters (B starts at zero)
            for k in range(len(rl.hbf)):
                v = torch.randn_like(rl.hbf[k]) * 0.05
                rl.hbf[k].copy_(v)
                rl.master[k].data.copy_(v.to(rl.master[k].dtype))
        seq = torch.randint(5, 290, (2, rl.T))
        tol = 1e-6 if stream else 1e-10        # streamed grads land in the fp32 host accumulator

        def run(chunk, sl):
            rl.mlp_chunk, rl.mlp_slice = chunk, sl
            for p in rl.params:
                p.grad = None
            if stream:
                for a in rl.sacc:
                    a.zero_()
                rl._dev_ad.clear()
            lp = rl.logprobs(seq, checkpoint=True)
            lp.sum().backward()
            gs = [a.clone() for a in rl.sacc] if stream else [p.grad.clone() for p in rl.params]
            return lp.detach().clone(), gs

        lp_c, g_c = run(5, 0)
        for sl in (7, 32, 96):                  # I = 96: several slices, uneven last slice, one slice
            lp_s, g_s = run(0, sl)
            assert torch.allclose(lp_s, lp_c, atol=1e-10), (targets, sl)
            assert all(torch.allclose(a, b, atol=tol) for a, b in zip(g_s, g_c)), (targets, sl)
            assert max(float(a.abs().max()) for a in g_s) > 1e-4          # the adapters actually get gradient
        rl.mlp_small_t = 0                      # force SliceMLP in no_grad too (34 rows would take the one-shot path)
        with torch.no_grad():                   # forward: bitwise equal to the one-shot MLP
            rl.mlp_chunk, rl.mlp_slice = 0, 0
            a = _full_logits(rl, seq[:, :-1])
            rl.mlp_slice = 7
            b = _full_logits(rl, seq[:, :-1])
        assert torch.equal(a, b), targets


def test_slice_mlp_generation_and_step_run():
    m = tiny(12, L=2)
    rl = _rl(m, targets=ALL7, rank=2, host_adam=True, seg=1, mlp_slice=32, gen_in_flight=2, stream_adapters=True,
             phase_split=True)
    for k in range(len(rl.hbf)):                # initialised host adapters, as test_phase_split does
        v = torch.randn_like(rl.hbf[k]) * 0.05
        rl.hbf[k].copy_(v)
        rl.master[k].data.copy_(v.to(rl.master[k].dtype))
    n = rl.step()
    assert n > 0


def test_reentrant_frees_streamed_adapters_and_matches():
    """reentrant=True: the same logprobs and host gradients as the non-reentrant update (streamed adapters, with and
    without the slice MLP), and after the forward no streamed adapter is alive (non-reentrant keeps every layer's)."""
    import weakref
    m = tiny(16, L=4)
    rl = _rl(m, targets=ALL7, rank=2, host_adam=True, seg=2, stream_adapters=True)
    for k in range(len(rl.hbf)):
        v = torch.randn_like(rl.hbf[k]) * 0.05
        rl.hbf[k].copy_(v)
        rl.master[k].data.copy_(v.to(rl.master[k].dtype))
    seq = torch.randint(5, 290, (2, rl.T))
    refs = []
    orig = rl._stream_layer_in

    def rec(l):
        d = orig(l)
        refs.extend(weakref.ref(a) for ab in d.values() for a in ab)
        return d
    rl._stream_layer_in = rec

    def run(re, sl, mc):
        rl.reentrant, rl.mlp_slice, rl.mlp_chunk = re, sl, mc
        for a in rl.sacc:
            a.zero_()
        rl._dev_ad.clear()
        refs.clear()
        lp = rl.logprobs(seq, checkpoint=True)
        alive = sum(r() is not None for r in refs)
        lp.sum().backward()
        return lp.detach().clone(), [a.clone() for a in rl.sacc], alive

    lp0, g0, alive0 = run(False, 0, 5)
    assert alive0 >= 2 * len(rl.targets) * (m.cfg.L - 1)           # the holder: every layer's adapters alive
    for sl, mc in ((0, 5), (32, 0)):
        lp1, g1, alive1 = run(True, sl, mc)
        assert alive1 == 0, alive1
        assert torch.allclose(lp1, lp0, atol=1e-10)
        assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g1, g0))
    assert not rl._dev_ad


def test_small_t_generation_matches():
    """mlp_small_t: decode-sized no-grad MLP calls run one-shot; generation logits equal the SliceMLP path's."""
    m = tiny(17, L=2)
    rl = _rl(m, targets=ALL7, rank=2, host_adam=False, seg=1, mlp_slice=32)
    Pc, g = 2, rl.group
    ids1 = torch.randint(5, 290, (Pc, rl.P))
    force = torch.randint(5, 290, (Pc * g, rl.G))
    rl.mlp_small_t = 0
    _, a = rl._generate(ids1, force=force, want_logits=True)
    rl.mlp_small_t = 64
    _, b = rl._generate(ids1, force=force, want_logits=True)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_shared_decode_attention_matches_sdpa():
    """dec_attn shared (no per-rollout prompt-KV copy) gives the same generation logits as the SDPA path (fp64),
    with and without a shared prefix level, and through sub-group generation."""
    for L0 in (0, 5):
        m = tiny(18, L=2)
        rl = _rl(m, targets=ALL7, rank=2, host_adam=False, seg=1, shared_prefix=L0)
        Pc, g = 2, rl.group
        ids1 = torch.randint(5, 290, (Pc, rl.P - L0))
        force = torch.randint(5, 290, (Pc * g, rl.G))
        rl.dec_attn = "sdpa"
        _, a = rl._generate(ids1, force=force, want_logits=True)
        rl.dec_attn = "shared"
        _, b = rl._generate(ids1, force=force, want_logits=True)
        assert all(torch.allclose(x, y, atol=1e-9) for x, y in zip(a, b)), L0
        _, c = rl._generate_sub(ids1, 2, force=force, want_logits=True)
        assert all(torch.allclose(x, y, atol=1e-9) for x, y in zip(a, c)), L0


def test_kv8_generation_close_and_step_runs():
    """kv8: own-rollout KV in int8 + scale; logits stay close to the exact path (lossy by design), and a phase-split
    streaming step with sub-group generation runs."""
    m = tiny(19, L=2)
    rl = _rl(m, targets=ALL7, rank=2, host_adam=False, seg=1)
    Pc, g = 2, rl.group
    ids1 = torch.randint(5, 290, (Pc, rl.P))
    force = torch.randint(5, 290, (Pc * g, rl.G))
    _, a = rl._generate(ids1, force=force, want_logits=True)
    rl.kv8, rl.dec_attn = True, "shared"
    _, b = rl._generate(ids1, force=force, want_logits=True)
    for x, y in zip(a, b):
        assert (x - y).abs().max() <= 0.02 * x.abs().max() + 1e-6
    q, sc = W._q8(torch.randn(3, 4, 8, dtype=torch.float64))
    assert q.dtype == torch.int8 and sc.shape == (3, 4, 1)
    st = _rl(tiny(20, L=2), targets=ALL7, rank=2, host_adam=True, seg=1, mlp_slice=32, gen_in_flight=2,
             stream_adapters=True, phase_split=True, kv8=True)
    for k in range(len(st.hbf)):
        v = torch.randn_like(st.hbf[k]) * 0.05
        st.hbf[k].copy_(v)
        st.master[k].data.copy_(v.to(st.master[k].dtype))
    assert st.step() > 0


def test_closed_loop_pacer():
    """The pacer lets work through while the credited noise average is at or above the target and blocks while it is
    under; the freeze credit comes off the clock."""
    import time
    from schemes.memory_link_exhaustion.attacks.local_rl.pace import Pacer
    W = 32 << 20
    t0 = time.perf_counter()
    rate = [56.0]                                             # GB/s the fake stream delivers
    credit = [0.0]
    noise = lambda: (int((time.perf_counter() - t0) * rate[0] * 1e9 / W), W)
    p = Pacer(55.5, noise, lambda: credit[0], grace_s=0.05, poll_s=0.005, flow_windows=10)
    p.wait()                                                  # the stream is seen flowing ...
    assert p.t0 is None
    time.sleep(0.6)
    p.wait()                                                  # ... for half a second: the baseline is taken
    assert p.t0 is not None
    time.sleep(0.1)
    a = time.perf_counter(); p.wait(); assert time.perf_counter() - a < 0.02 and p.waits == 0      # above target
    t0 -= 0.0                                                 # the stream stalls: 0.1 s with no windows
    stall = time.perf_counter()
    base = noise()[0]
    slow = lambda: (base + int((time.perf_counter() - stall - 0.1) * rate[0] * 1e9 / W)
                    if time.perf_counter() - stall > 0.1 else base, W)
    p.noise_fn = slow
    time.sleep(0.12)
    a = time.perf_counter()
    credit[0] = 0.0
    r_before, _ = p.rate()
    assert r_before < 55.5                                    # the stall pulled the average under the target
    credit[0] = 0.1                                           # ... unless the verifier credits it as a freeze
    r_after, _ = p.rate()
    assert r_after > r_before
    st = p.stats()
    assert st["target_gbs"] == 55.5 and st["credit_s"] == 0.1
