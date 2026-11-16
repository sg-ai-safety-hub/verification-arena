"""Kernel vs NumPy reference. Runs on CPU with TRITON_INTERPRET=1, or on a GPU."""
import os
import sys

if not os.environ.get("VARENA_GPU_TESTS"):
    os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import torch

from schemes.memory_link_exhaustion.gpu import kernels as K
from schemes.memory_link_exhaustion.encodings import ref

DEV = "cuda" if os.environ.get("VARENA_GPU_TESTS") else "cpu"
torch.manual_seed(0)


def u16(t):
    return t.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16)


def test_xor16():
    x = torch.randn(5000, dtype=torch.bfloat16, device=DEV)
    y = torch.empty_like(x)
    K.xor16(x, y, 0xDEADBEEF, j0=7)
    assert (u16(y) == ref.encode16(u16(x), 0xDEADBEEF, 7)).all()
    z = torch.empty_like(x)
    K.xor16(y, z, 0xDEADBEEF, j0=7)
    assert torch.equal(z, x)


def test_xor16_w():
    x = torch.randn(37, 129, dtype=torch.bfloat16, device=DEV)
    y = torch.empty_like(x)
    K.xor16_w(x, y, 0xC0FFEE, 129)
    assert (u16(y).ravel() == ref.encode_w(u16(x).ravel(), 0xC0FFEE, 0, 129)).all()
    part = torch.empty(1000, dtype=torch.bfloat16, device=DEV)
    K.xor16_w(x.reshape(-1)[500:1500], part, 0xC0FFEE, 129, j0=500)
    assert (u16(part) == ref.encode_w(u16(x).ravel()[500:1500], 0xC0FFEE, 500, 129)).all()


def _enc(w, key):
    e = torch.empty_like(w)
    K.xor16_w(w, e, key, w.shape[1])
    return e


def test_matmul_plain_enc_transpose():
    for (M, N, Kd) in [(5, 48, 256), (40, 130, 320), (130, 64, 192)]:
        x = torch.randn(M, Kd, dtype=torch.bfloat16, device=DEV)
        w = torch.randn(N, Kd, dtype=torch.bfloat16, device=DEV) * 0.1
        refy = (x.float() @ w.float().T)
        y0 = K.matmul_bt(x, w, None)
        y1 = K.matmul_bt(x, _enc(w, 1234), 1234)
        assert torch.equal(y0, y1)
        assert (y0.float() - refy).abs().max() < 0.05 * refy.abs().max() + 0.05
        g = torch.randn(M, N, dtype=torch.bfloat16, device=DEV)
        dx0 = K.matmul_bt(g, w, None, transpose=True)
        dx1 = K.matmul_bt(g, _enc(w, 99), 99, transpose=True)
        assert torch.equal(dx0, dx1)
        r2 = g.float() @ w.float()
        assert (dx0.float() - r2).abs().max() < 0.05 * r2.abs().max() + 0.05


def test_embed():
    w = torch.randn(50, 70, dtype=torch.bfloat16, device=DEV)
    ids = torch.tensor([3, 0, 49, 3], device=DEV)
    assert torch.equal(K.embed(ids, w, None), w[ids])
    assert torch.equal(K.embed(ids, _enc(w, 5), 5), w[ids])


def test_kv_and_attention():
    B, Hkv, G, T, D = 3, 2, 3, 128, 32
    for key in (None, 777):
        slab = torch.zeros(2, B, Hkv, T, D, dtype=torch.bfloat16, device=DEV)
        L = 70
        kk = torch.randn(B, L, Hkv, D, dtype=torch.bfloat16, device=DEV)
        vv = torch.randn(B, L, Hkv, D, dtype=torch.bfloat16, device=DEV)
        pos = torch.zeros(B, dtype=torch.int32, device=DEV)
        K.kv_store(kk, vv, slab, key, pos)
        if key is None:
            assert torch.equal(slab[0, :, :, :L].permute(0, 2, 1, 3), kk)
        else:
            dec = torch.empty_like(slab)
            K.xor16(slab, dec, key)
            assert torch.equal(dec[1, :, :, :L].permute(0, 2, 1, 3), vv)
        q = torch.randn(B, Hkv * G, D, dtype=torch.bfloat16, device=DEV)
        lens = torch.full((B,), L, dtype=torch.int32, device=DEV)
        o = K.attn_decode(q, slab, key, lens)
        kf = kk.float().permute(0, 2, 1, 3).repeat_interleave(G, 1)
        vf = vv.float().permute(0, 2, 1, 3).repeat_interleave(G, 1)
        s = torch.einsum("bhd,bhtd->bht", q.float(), kf) / D ** 0.5
        r = torch.einsum("bht,bhtd->bhd", s.softmax(-1), vf)
        assert (o.float() - r).abs().max() < 0.03, (o.float() - r).abs().max()


def test_filler_digest():
    npg = 6
    arena = torch.zeros(npg * ref.PAGE, dtype=torch.uint8, device=DEV)
    fkey = 0xABCDEF
    K.fill(arena[2 * ref.PAGE:], 2, fkey, rounds=1)
    words = arena.view(torch.int32).cpu().numpy().view(np.uint32).reshape(npg, -1)
    exp = ref.filler_pages(fkey, [2, 3], rounds=1)
    assert (words[2:4] == exp[:2]).all()
    pm = torch.arange(npg, dtype=torch.int64, device=DEV) * ref.PAGE_WORDS
    pg = torch.tensor([3, 0], dtype=torch.int32, device=DEV)
    a, b = K.digest_pages(arena.view(torch.int32), pm, pg, 9)
    assert int(a[0].cpu().numpy().view(np.uint32)) == ref.page_digest(words[3], 9)[0]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
