"""Batched chain replay (verifier/chainreplay.py) against the scalar reference (ref.ch_chain).

The scalar side builds every probed chunk the way the verifier's per-chain check does (the declared weights encoded
per matrix, filler, the block table's source for a KV position, the stand-in inside a band); the batched side must
give the same 32 words for every chunk, and the same chain values, for chains with different nonces, heads and block
tables. Integer arithmetic only. Covers: a chunk that straddles the end of a matrix (filler after it), a matrix stored
as is (key < 0), a matrix made of two sources, cautious and tight tables, a request missing from the store."""
import os

os.environ["VARENA_WENC"] = "sloth"
os.environ["VARENA_FILLER"] = "sloth"

import numpy as np

from schemes.memory_link_exhaustion.core.arena import Entry, Layout
from schemes.memory_link_exhaustion.encodings import ref, sloth64
from schemes.memory_link_exhaustion.inference import kvtable as KT
from schemes.memory_link_exhaustion.verifier import chainreplay as CR

G, PAGE = ref.CH_GRAN_PAGES, ref.PAGE
MIB = 1 << 20


class Weights:
    def __init__(self, sizes):
        r = np.random.default_rng(11)
        self.cache = {k: r.integers(0, 65536, n, dtype=np.uint16) for k, n in sizes.items()}

    def flat(self, name):
        return self.cache[name]

    def range(self, sources, j0, n):
        out, base = [], 0
        for s in sources:
            a = self.flat(s)
            lo, hi = max(j0, base), min(j0 + n, base + a.size)
            if lo < hi:
                out.append(a[lo - base: hi - base])
            base += a.size
        return np.concatenate(out)


def main():
    assert CR.supported()
    L, nb = 2, G
    nW, nK, nF = G, L * nb, G
    fkey, kkeys = 0x51F1E, [0xA11CE, 0xB0B]
    na = (20 * MIB + 24) // 2
    lay = Layout([Entry("a", 0, (na,), ["a"], 0x1234),
                  Entry("b", 22 * MIB, (5 * MIB,), ["b"], -1),
                  Entry("c", 34 * MIB, (8 * MIB,), ["c0", "c1"], 0x77)], nW * PAGE, nW * PAGE, (nW + nK) * PAGE)
    w = Weights({"a": na, "b": 5 * MIB, "c0": 3 * MIB, "c1": 5 * MIB})
    # the KV store: request i -> uint16 [blocks][layer] x 64 KiB
    r = np.random.default_rng(3)
    blocks = {0: 300, 1: 200, 2: 40}
    store = {i: r.integers(0, 65536, n * L * (PAGE // 2), dtype=np.uint16) for i, n in blocks.items()}
    fps = {0: 0xAAAA0001, 1: 0xBBBB0002, 2: 0xCCCC0003}
    fpi = {fps[i]: i for i in (0, 1, 2)}
    UNK = 0xDEAD                                              # a fingerprint no request of the workload has

    def tables(tight):
        if tight:
            t1 = KT.KVTable(nb, 16, tight=True)
            t1.update_tight({b: (fps[2], b - 600, 40 * 16, 40 * 16) for b in range(600, 640)})
            t1.update_tight({}, freed={b: (fps[2], b - 600, 40 * 16) for b in range(600, 640)})
            # request 0 on blocks 0..299 (4,700 settled, 4,730 computed), request 1 on 500..699 over the freed blocks
            t1.update_tight({**{b: (fps[0], b, 4700, 4730) for b in range(300)},
                             **{b: (fps[1], b - 500, 1000, 1900) for b in range(500, 700)}})
            t2 = KT.parse(t1.body())
            t2.update_tight({**{b: (fps[0], b, 4760, 4790) for b in range(300)},
                             **{b: (fps[1], b - 500, 3100, 3199) for b in range(500, 700)}})
            return t1, t2
        t1 = KT.KVTable(nb, 16)
        t1.update({b: (fps[2], b - 600, 640) for b in range(600, 640)})
        t1.update({}, freed={b: (fps[2], b - 600, 640) for b in range(600, 640)})
        t1.update({**{b: (fps[0], b, 4700) for b in range(300)}, **{b: (fps[1], b - 500, 1000) for b in range(500, 600)}},
                  fresh=[700, 701])
        t2 = KT.parse(t1.body())
        t2.update({**{b: (fps[0], b, 4760) for b in range(300)}, **{b: (fps[1], b - 500, 1500) for b in range(500, 600)}},
                  freed={700: None})
        return t1, t2

    nrg = ref.ring_chase_granules(8, 4)
    gp = ref.ch_granules_mix(nW, nK, nF, 0, nrg)
    tb = len(gp) // ref.CH_GRAN_PER_BLOCK
    assert nrg == 2 and tb == 1

    def ring(head, page, wo):
        q, w_in = ref.ring_chase_slot(page, wo)
        x = (np.arange(w_in, w_in + 32, dtype=np.uint64) + np.uint64((head - q) << 24)) * np.uint64(0x9E3779B97F4A7C15)
        return ((x >> np.uint64(29)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    def scalar_words(page, wo, nonce, head, t, miss):
        """One chunk, the long way (the per-chain check in verifier.py, with only the probed range encoded)."""
        if page >= ref.CH_RA0:
            return ring(head, page, wo)
        fill = sloth64.filler_words(fkey, page, wo, 32)
        if page >= nW + nK:
            return fill
        if page >= nW:
            layer, block = divmod(page - nW, nb)
            src = t.source(block, ref.ch_kv_token(wo))
            if src is None:
                return ref.ch_mask_words(nonce, page - nW, wo)
            if src == "filler":
                return fill
            i = fpi.get(int(src[0]))
            o16 = ((int(src[1]) * L + layer) * PAGE + wo * 4) // 2
            if i is None or o16 + 64 > store[i].size:
                miss.append((page, wo))
                return np.zeros(32, np.uint32)
            return sloth64.encode_range(np.ascontiguousarray(store[i][o16:o16 + 64]),
                                        sloth64.kv_seed(kkeys[layer]), block * (PAGE // 2) + 2 * wo).view(np.uint32)
        a, b = page * PAGE + wo * 4, page * PAGE + wo * 4 + 128
        u16 = fill.copy().view(np.uint16)
        for e in lay.entries:
            lo, hi = max(a, e.offset), min(b, e.offset + e.nbytes)
            if lo >= hi:
                continue
            j0, n = (lo - e.offset) // 2, (hi - lo) // 2
            raw = w.range(e.sources, j0, n)
            u16[(lo - a) // 2:(hi - a) // 2] = raw if e.key < 0 else \
                sloth64.encode_range(raw, ref.sloth_seed(e.key, ref.w_ordinal(e.offset)), j0)
        return u16.view(np.uint32)

    for tight in (True, False):
        t1, t2 = tables(tight)
        assert KT.check(t1) == [] and KT.check(t2) == []
        jobs = [(0x1111, 0, 900, t1), (0x2222, 5, 901, t2), (0x3333, 4223, 950, t1), (0x4444, 17, 951, t2),
                (0xFFFFFFFF, 1, 999, t2)]
        src = CR.Sources(nW=nW, nK=nK, nb=nb, L=L, layout=lay, weights=w, fkey=fkey,
                         kv_seeds=[sloth64.kv_seed(k) for k in kkeys], store=lambda i: store.get(i), fpi=fpi,
                         scalar=lambda j, page, wo: ring(jobs[j][2], page, wo))
        src.bind([jb[0] for jb in jobs], [jb[3] for jb in jobs])
        # 1. chunk by chunk, hand-picked: the end of matrix a (20 MiB + 24 bytes: the chunk straddles into filler),
        #    matrix b (stored as is) and its edges, the seam between c's two sources, filler, every kind of KV position
        end_a = 20 * MIB + 24
        picks = [(0, 0), (0, 32), (end_a // PAGE, (end_a % PAGE) // 128 * 32), (end_a // PAGE, (end_a % PAGE) // 128 * 32 + 32),
                 (21 * MIB // PAGE, 64), (22 * MIB // PAGE, 0), (22 * MIB // PAGE + 3, 16352), (32 * MIB // PAGE - 1, 16352),
                 (32 * MIB // PAGE, 0), (34 * MIB // PAGE, 0), ((34 + 6) * MIB // PAGE - 1, 16352), ((34 + 6) * MIB // PAGE, 0),
                 (50 * MIB // PAGE - 1, 16352), (50 * MIB // PAGE, 0), (G - 1, 16352),
                 (nW + nK, 0), (nW + nK + G - 1, 16352)]
        for layer in range(L):
            for block in (0, 5, 293, 294, 296, 299, 300, 499, 500, 562, 563, 600, 620, 639, 640, 693, 699, 700, 701, G - 1):
                picks += [(nW + layer * nb + block, tok * 1024 + 32 * k) for tok in (0, 5, 6, 7, 12, 15) for k in (0, 31)]
        picks += [(ref.CH_RA0 + pg, wo) for pg, wo in ((0, 0), (511, 16352), (512, 32), (2 * G - 1, 16352))]
        for jx, (nonce, _, head, t) in enumerate(jobs[:2]):
            pg = np.array([p for p, _ in picks], dtype=np.int64)
            wo = np.array([o for _, o in picks], dtype=np.int64)
            got = src(np.full(pg.size, jx, dtype=np.int64), pg, wo)
            for k, (p, o) in enumerate(picks):
                want = scalar_words(p, o, nonce, head, t, [])
                assert (got[k] == want).all(), (tight, jx, p, o)
        st = src.stats
        assert st["kv_masked"] > 0 and st["kv_live"] > 0 and st["kv_filler"] > 0 and st["other"] > 0, st
        # 2. whole chains, different nonce / chain / head / table per chain
        P = 12
        src.bind([jb[0] for jb in jobs], [jb[3] for jb in jobs])
        got = CR.walk([jb[0] for jb in jobs], [jb[1] for jb in jobs], P, tb, 1, gp, src)
        for jx, (nonce, c, head, t) in enumerate(jobs):
            want = ref.ch_chain(nonce, c, P, tb, 1, gp, lambda p, o: scalar_words(p, o, nonce, head, t, []))
            assert int(got[jx]) == want, (tight, jx, hex(int(got[jx])), hex(want))
        assert int(src.miss.sum()) == 0
        if tight:      # blocks 600..639 were request 2's and are request 1's now: reads above its edge are the old bytes
            assert src.stats["kv_prev_owner"] > 0, src.stats
        # a chain replayed under the other table differs (the table is part of what the chain was computed under)
        src.bind([jobs[0][0]], [t2])
        other = CR.walk([jobs[0][0]], [jobs[0][1]], P, tb, 1, gp, src)
        assert int(other[0]) != int(got[0])
    # 3. a writer the store does not have: counted as a miss on that chain (and the chunk reads as zeros, as scalar)
    t = KT.KVTable(nb, 16, tight=True)
    t.update_tight({b: (UNK, b, 16 * nb, 16 * nb) for b in range(nb)})
    src = CR.Sources(nW=nW, nK=nK, nb=nb, L=L, layout=lay, weights=w, fkey=fkey,
                     kv_seeds=[sloth64.kv_seed(k) for k in kkeys], store=lambda i: store.get(i), fpi=fpi,
                     scalar=lambda j, page, wo: ring(7, page, wo))
    src.bind([5, 6], [t, t])
    got = CR.walk([5, 6], [0, 1], 6, tb, 1, gp, src)
    miss = []
    want = ref.ch_chain(6, 1, 6, tb, 1, gp, lambda p, o: scalar_words(p, o, 6, 7, t, miss))
    assert int(got[1]) == want and len(miss) > 0 and int(src.miss[1]) == len(miss) and src.why.get("fp_unknown", 0) > 0
    # 4. tagged in-flight reads (--kv-table 3): the chain reads band positions; each holds its old content (the last
    #    writer's bytes or filler) or its new content (the live owner's KV); the kernel's tag tells the replay which
    tw = KT.KVTable(nb, 16, tight=True)
    tw.update_tight({b: (fps[2], b - 600, 40 * 16, 40 * 16) for b in range(600, 640)})
    tw.update_tight({}, freed={b: (fps[2], b - 600, 40 * 16) for b in range(600, 640)})
    tw.update_tight({**{b: (fps[0], b, 1000, 4700) for b in range(300)},
                     **{b: (fps[1], b - 500, 200, 3100) for b in range(500, 700)}})
    assert KT.check(tw) == [] and tw.masked_tokens() == 3700 + 2900

    def enc(i, bidx, layer, block, wo):
        o16 = ((bidx * L + layer) * PAGE + wo * 4) // 2
        return sloth64.encode_range(np.ascontiguousarray(store[i][o16:o16 + 64]), sloth64.kv_seed(kkeys[layer]),
                                    block * (PAGE // 2) + 2 * wo).view(np.uint32)

    def device(nonce, c, head, choose, gpx=None, zone=0, n_ring=8):
        """What the kernel computes: the chain over the real bytes, and [count, tags] for its in-band reads (and, with
        zone > 0, for its reads in the ring's zone: the slot holds its old window or the incoming one)."""
        tags = []

        def mem(page, wo):
            if zone and page >= ref.CH_RA0:
                q, w_in = ref.ring_chase_slot(page, wo)
                if q >= n_ring - zone:
                    w = ring(head, page, wo) if choose(page, wo) == 0 else ring(head + n_ring, page, wo)
                    tags.append(ref.ch_tag_id(nonce, ref.ring_tag_id(q, w_in), w))
                    return w
            if nW <= page < nW + nK:
                layer, block = divmod(page - nW, nb)
                tok = ref.ch_kv_token(wo)
                if int(tw.lo[block]) <= tok < int(tw.hi[block]):
                    wid = int(tw.writer[block, tok])
                    old = sloth64.filler_words(fkey, page, wo, 32) if wid == 0 else \
                        enc(fpi[tw.writers[wid][0]], tw.writers[wid][1], layer, block, wo)
                    new = enc(fpi[int(tw.fp[block])], int(tw.idx[block]), layer, block, wo)
                    ch = choose(page, wo)
                    w = old if ch == 0 else new if ch == 1 else (old ^ np.uint32(0x5A5A5A5A))
                    tags.append(ref.ch_tag(nonce, page - nW, wo, w))
                    return w
            return scalar_words(page, wo, nonce, head, tw, [])
        s_ = ref.ch_chain(nonce, c, P, tb, 1, gp if gpx is None else gpx, mem)
        t_ = np.zeros(ref.CH_TAG_SLOTS, np.uint32)
        t_[0] = len(tags)
        t_[1:1 + min(len(tags), ref.CH_TAG_SLOTS - 1)] = tags[:ref.CH_TAG_SLOTS - 1]
        return s_, t_
    P = 24
    jobs = [(0x1111, 0, 900), (0x2222, 5, 901), (0x3333, 4223, 950), (0x4444, 17, 951)]
    mixed = lambda page, wo: (page * 7 + wo // 32) % 2                       # old or new, position by position
    dev = [device(n, c, h, mixed) for n, c, h in jobs]
    src = CR.Sources(nW=nW, nK=nK, nb=nb, L=L, layout=lay, weights=w, fkey=fkey,
                     kv_seeds=[sloth64.kv_seed(k) for k in kkeys], store=lambda i: store.get(i), fpi=fpi,
                     scalar=lambda j, page, wo: ring(jobs[j][2], page, wo))
    src.bind([jb[0] for jb in jobs], [tw] * len(jobs), tags=[d[1] for d in dev])
    got = CR.walk([jb[0] for jb in jobs], [jb[1] for jb in jobs], P, tb, 1, gp, src)
    assert [int(x) for x in got] == [d[0] for d in dev], "tagged replay differs from the device chain"
    assert [int(x) for x in src.nband] == [int(d[1][0]) for d in dev] and int(src.nband.sum()) > 20
    assert src.stats["band_new"] > 5 and src.stats["band_old"] > 5 and src.stats["band_neither"] == 0, src.stats
    assert int(src.neither.sum()) == 0 and not src.tag_over.any() and int(src.miss.sum()) == 0
    # a band position that holds something else (caught mid-write, or a cheater's own bytes): matches neither, and the
    # chain is reported as not replayable instead of being followed with a guess
    odd = lambda page, wo: 2 if (page + wo // 32) % 5 == 0 else (page * 7 + wo // 32) % 2
    dev2 = [device(n, c, h, odd) for n, c, h in jobs]
    src.bind([jb[0] for jb in jobs], [tw] * len(jobs), tags=[d[1] for d in dev2])
    CR.walk([jb[0] for jb in jobs], [jb[1] for jb in jobs], P, tb, 1, gp, src)
    assert (src.neither > 0).all() and src.stats["band_neither"] > 0
    # without tags the same table is replayed under the stand-in rule (unchanged behaviour)
    src.bind([jobs[0][0]], [tw])
    plain = CR.walk([jobs[0][0]], [jobs[0][1]], P, tb, 1, gp, src)
    assert int(plain[0]) == ref.ch_chain(jobs[0][0], jobs[0][1], P, tb, 1, gp,
                                         lambda p_, o_: scalar_words(p_, o_, jobs[0][0], jobs[0][2], tw, []))
    # tagged ring reads: every ring slot is in the chase; a read in the zone's slots (which incoming windows overwrite)
    # is tagged and the replay takes the old or the incoming window by it. KV band and ring tags share one sequence.
    nrg2 = ref.ring_chase_granules(8, 4, tagged=True)
    gp2 = ref.ch_granules_mix(nW, nK, nF, 0, nrg2)
    assert nrg2 == 4 and len(gp2) // ref.CH_GRAN_PER_BLOCK == tb
    dev3 = [device(n, c, h, mixed, gpx=gp2, zone=4) for n, c, h in jobs]
    src3 = CR.Sources(nW=nW, nK=nK, nb=nb, L=L, layout=lay, weights=w, fkey=fkey,
                      kv_seeds=[sloth64.kv_seed(k) for k in kkeys], store=lambda i: store.get(i), fpi=fpi,
                      scalar=lambda j, page, wo: ring(jobs[j][2], page, wo),
                      scalar_new=lambda j, page, wo: ring(jobs[j][2] + 8, page, wo), ring_zone=(8, 4))
    src3.bind([jb[0] for jb in jobs], [tw] * len(jobs), tags=[d[1] for d in dev3])
    got3 = CR.walk([jb[0] for jb in jobs], [jb[1] for jb in jobs], P, tb, 1, gp2, src3)
    assert [int(x) for x in got3] == [d[0] for d in dev3], "tagged ring replay differs from the device chain"
    assert [int(x) for x in src3.nband] == [int(d[1][0]) for d in dev3]
    assert src3.stats["ring_new"] > 0 and src3.stats["ring_old"] > 0 and src3.stats["ring_neither"] == 0, src3.stats
    assert int(src3.neither.sum()) == 0 and not src3.tag_over.any()
    # a zone slot holding neither window: the chain is reported, not followed with a guess
    dev4 = [device(n, c, h, lambda page, wo: 0, gpx=gp2, zone=4) for n, c, h in jobs]
    src3.scalar = lambda j, page, wo: ring(jobs[j][2] + 3, page, wo)
    src3.bind([jb[0] for jb in jobs], [tw] * len(jobs), tags=[d[1] for d in dev4])
    CR.walk([jb[0] for jb in jobs], [jb[1] for jb in jobs], P, tb, 1, gp2, src3)
    assert src3.stats["ring_neither"] > 0 and (src3.neither > 0).any()
    print("OK")


if __name__ == "__main__":
    main()
