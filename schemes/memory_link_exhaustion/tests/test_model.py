"""Tiny random Qwen2: our plain, triton and sealed paths vs HF transformers greedy."""
import os
import sys
import tempfile

if not os.environ.get("VARENA_GPU_TESTS"):
    os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import torch

DEV = "cuda" if os.environ.get("VARENA_GPU_TESTS") else "cpu"


def make_tiny(path, tied=False):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=96, num_hidden_layers=2, vocab_size=97, hidden_act="silu",
                      max_position_embeddings=256, tie_word_embeddings=tied,
                      rope_theta=10000.0)
    torch.manual_seed(1)
    m = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1)
    m.save_pretrained(path, safe_serialization=True)
    return m


def gen(model, kv, prompts, n_new):
    B, P = prompts.shape
    zero = torch.zeros(B, dtype=torch.int32, device=DEV)
    kv.reset()
    h = model.prefill(prompts, kv, 0, zero)
    lg = [model.logits(h)]
    toks = [lg[-1].argmax(-1)]
    for t in range(1, n_new):
        pos = torch.full((B,), P + t - 1, dtype=torch.int32, device=DEV)
        lens = pos + 1
        lg.append(model.decode(toks[-1], pos, lens, kv))
        toks.append(lg[-1].argmax(-1))
    gen.logits = torch.stack(lg, 1).float()
    return torch.stack(toks, 1)


def test_against_hf():
    from schemes.memory_link_exhaustion.core.model import load_plain, KV
    from schemes.memory_link_exhaustion.core.arena import load_sealed
    with tempfile.TemporaryDirectory() as d:
        hf = make_tiny(d)
        B, P, G = 2, 9, 6
        T = 64
        prompts = torch.randint(0, 97, (B, P))
        pr = prompts.to(DEV)
        for impl in ("torch", "triton"):
            m = load_plain(d, T, impl=impl, device=DEV)
            out = gen(m, KV.plain(m.cfg, B, T, DEV), pr, G)
            full = torch.cat([prompts, out.cpu()], 1)
            with torch.no_grad():
                hl = hf(full).logits[:, P - 1:P - 1 + G].float()
            err = (gen.logits.cpu() - hl).abs().max().item()
            print(impl, out.tolist(), "max logit err", err, "scale", hl.abs().max().item())
            assert err < 0.05 * hl.abs().max().item() + 0.05
        ms, kv, arena = load_sealed(d, B, T, master=0x1234567890, fkey=0x55AA, device=DEV)
        arena.grow(2 * 65536)
        out_s = gen(ms, kv, pr, G)
        print("sealed", out_s.tolist())
        m = load_plain(d, T, impl="triton", device=DEV)
        out_t = gen(m, KV.plain(m.cfg, B, T, DEV), pr, G)
        assert torch.equal(out_s, out_t)
        # every byte of the W region decodes back to the weights
        from schemes.memory_link_exhaustion.core.model import SafeTensors
        st = SafeTensors(d)
        e = arena.layout.by_name("l1.qkv")
        enc = arena.view(e).cpu().numpy().view(np.uint16).ravel()
        from schemes.memory_link_exhaustion.encodings import ref as R
        raw = st.concat(e.sources).view(torch.int16).numpy().view(np.uint16).ravel()
        assert (R.encode_w(raw, e.key, 0, e.shape[1]) == enc).all()


if __name__ == "__main__":
    test_against_hf(); print("ok")
