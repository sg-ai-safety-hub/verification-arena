"""The memory arena: one declared page space covering (almost) all of HBM.

Layout, in 64 KiB pages, shared by prover, verifier and replay:

  [ W: encoded weight matrices | K: encoded KV slabs | F: filler ............ ]

Every page starts as filler (the encoding of "empty"); data ranges are then
overwritten with data XOR keystream(tensor key). So at any instant the arena is
encode(state), and the verifier can compute any page's expected value from the
declared weights, the replayed KV state and the seed.

The arena is backed by a few big allocations ("chunks"); page_map gives the
int32-word offset of each page from chunk 0's base, which is what the challenge
kernel follows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from schemes.memory_link_exhaustion.gpu import kernels as K
from schemes.memory_link_exhaustion.encodings import ref
from schemes.memory_link_exhaustion.core.model import Cfg, KV, Lin, Model, SafeTensors, big_tensors, small_tensors

PAGE = ref.PAGE


def align(n, a=PAGE):
    return (n + a - 1) // a * a


@dataclass
class Entry:
    name: str
    offset: int                 # bytes from arena start (page aligned)
    shape: tuple
    sources: list | None        # HF tensor names (weights) or None (KV slab)
    key: int

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * 2


@dataclass
class Layout:
    entries: list = field(default_factory=list)
    w_end: int = 0
    kv_start: int = 0
    kv_end: int = 0

    def to_json(self):
        return {"entries": [[e.name, e.offset, list(e.shape), e.sources, e.key]
                            for e in self.entries],
                "w_end": self.w_end, "kv_start": self.kv_start, "kv_end": self.kv_end}

    @staticmethod
    def from_json(d):
        lay = Layout([Entry(n, o, tuple(s), src, k) for n, o, s, src, k in d["entries"]],
                     d["w_end"], d["kv_start"], d["kv_end"])
        return lay

    def by_name(self, name):
        for e in self.entries:
            if e.name == name:
                return e
        raise KeyError(name)


def build_layout(cfg: Cfg, B: int, T: int, master: int) -> Layout:
    lay = Layout()
    off = 0
    for name, shape, src in big_tensors(cfg):
        lay.entries.append(Entry(name, off, tuple(shape), src, ref.tensor_key(master, off)))
        off = align(off + lay.entries[-1].nbytes)
    lay.w_end = off
    lay.kv_start = off
    for l in range(cfg.L):
        shape = (2, B, cfg.Hkv, T, cfg.D)
        lay.entries.append(Entry(f"kv{l}", off, shape, None, ref.tensor_key(master, off)))
        off = align(off + lay.entries[-1].nbytes)
    lay.kv_end = off
    return lay


class Arena:
    """Prover-side arena. Call grow() once the rest of the process is set up."""

    def __init__(self, layout: Layout, fkey: int, device="cuda", rounds=ref.FILLER_ROUNDS):
        self.layout, self.fkey, self.device, self.rounds = layout, fkey, device, rounds
        self.chunks = []          # (first_page, uint8 tensor)
        self._add_chunk(layout.kv_end)
        self.page_map = None

    @property
    def base(self):
        return self.chunks[0][1]

    @property
    def npages(self):
        return sum(t.numel() // PAGE for _, t in self.chunks)

    def _add_chunk(self, nbytes):
        nbytes = align(nbytes)
        first = self.npages
        t = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
        K.fill(t, first, self.fkey, self.rounds)
        self.chunks.append((first, t))

    def grow(self, nbytes: int):
        """Add a filler chunk of nbytes."""
        if nbytes >= PAGE:
            self._add_chunk(nbytes // PAGE * PAGE)
        self._build_map()

    def _build_map(self):
        pm = np.empty(self.npages, dtype=np.int64)
        b0 = self.base.data_ptr()
        for first, t in self.chunks:
            n = t.numel() // PAGE
            pm[first:first + n] = (t.data_ptr() - b0) // 4 + np.arange(n, dtype=np.int64) * ref.PAGE_WORDS
        self.page_map = torch.from_numpy(pm).to(self.device)

    @property
    def nbytes_resident(self):
        return sum(t.numel() for _, t in self.chunks)

    def view(self, e: Entry, dtype=torch.int16):
        t = self.base[e.offset: e.offset + e.nbytes]
        return t.view(dtype).view(e.shape)

    def refill(self, start: int, end: int):
        K.fill(self.base[start:end], start // PAGE, self.fkey, self.rounds)

    def base_i32(self):
        return self.base.view(torch.int32)

    def info(self):
        return {"npages": self.npages, "resident_bytes": self.nbytes_resident,
                "declared_bytes": self.npages * PAGE,
                "w_bytes": self.layout.w_end, "kv_bytes": self.layout.kv_end - self.layout.kv_start}


def load_sealed(path: str, B: int, T: int, master: int, fkey: int, impl="sealed",
                device="cuda", lay: Layout | None = None):
    """Build the arena (W and K regions), encode the weights into it, and return
    (model, kv, arena). The filler region F is added later with arena.grow()."""
    cfg = Cfg.from_hf(path)
    lay = lay or build_layout(cfg, B, T, master)
    arena = Arena(lay, fkey, device)
    st = SafeTensors(path)
    lins = {}
    for e in lay.entries:
        if e.sources is None:
            continue
        src = st.concat(e.sources).to(dtype=torch.bfloat16)
        assert tuple(src.shape) == tuple(e.shape), (e.name, src.shape, e.shape)
        dst = arena.view(e)
        # stage in slices so the plaintext never needs a full-size GPU copy
        flat = src.reshape(-1)
        dflat = dst.reshape(-1)
        step = 64 << 20
        for j in range(0, flat.numel(), step):
            part = flat[j:j + step].to(device)
            K.xor16_w(part, dflat[j:j + step], e.key, e.shape[1], j0=j)
        lins[e.name] = Lin(dst, e.key)
        del src
    small = {n: st.get(n).to(device=device, dtype=torch.bfloat16) for n in small_tensors(cfg)}
    slabs, keys = [], []
    for l in range(cfg.L):
        e = lay.by_name(f"kv{l}")
        slabs.append(arena.view(e))
        keys.append(e.key)
    kv = KV(slabs, keys, arena, (lay.kv_start, lay.kv_end))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return Model(cfg, lins, small, T, device), kv, arena
