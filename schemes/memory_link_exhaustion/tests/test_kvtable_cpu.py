"""KV block table (inference/kvtable.py): band values, life cycle of a block, round trip, digest, structural checks."""
import numpy as np

from schemes.memory_link_exhaustion.encodings import ref
from schemes.memory_link_exhaustion.inference import kvtable as KT


def main():
    t = KT.KVTable(64, 16)
    assert (t.band() == (16 | (16 << 8))).all() and t.masked_tokens() == 0          # sealed: all filler, all readable
    d0 = t.digest()
    # a request with fingerprint 0xABC owns blocks 5, 9, 2 (in that order); 37 tokens settled
    live = {5: (0xABC, 0, 37), 9: (0xABC, 1, 37), 2: (0xABC, 2, 37)}
    v = t.update(live, fresh=[2])
    assert v == 1 and [int(t.lo[b]) for b in (5, 9, 2)] == [16, 16, 5] and t.digest() != d0
    band = t.band()
    for b, lo in ((5, 16), (9, 16), (2, 5)):
        for tok in range(16):
            assert ref.ch_masked((int(band[b]) & 0xff, int(band[b]) >> 8), tok * ref.CH_KV_TOK_WORDS) == (tok >= lo)
    assert t.masked_tokens() == 11
    # settled count grows; a block allocated in this call is not readable until a snapshot covers it
    t.update({5: (0xABC, 0, 50), 9: (0xABC, 1, 50), 2: (0xABC, 2, 50), 7: (0xABC, 3, 50)}, fresh=[7])
    assert int(t.lo[2]) == 16 and int(t.lo[7]) == 2 and int(t.kind[7]) == KT.LIVE
    t.update({}, fresh=[11])
    assert int(t.kind[11]) == KT.FRESH and int(t.lo[11]) == 0
    # the request finishes at 55 tokens: its blocks go stale with the settled count at the free
    t.update({}, freed={5: (0xABC, 0, 55), 9: (0xABC, 1, 55), 2: (0xABC, 2, 55), 7: (0xABC, 3, 55), 11: None})
    assert [int(t.kind[b]) for b in (5, 9, 2, 7, 11)] == [KT.STALE] * 4 + [KT.UNKNOWN]
    assert int(t.lo[7]) == 7 and int(t.lo[11]) == 0
    # reallocation: fresh first, then live under the new owner
    t.update({7: (0xDEF, 0, 0)}, fresh=[7])
    assert int(t.kind[7]) == KT.LIVE and int(t.lo[7]) == 0 and int(t.fp[7]) == 0xDEF
    assert KT.check(t) == []
    # round trip and digest
    u = KT.parse(t.body())
    assert u.version == t.version and (u.band() == t.band()).all() and u.digest() == t.digest() and KT.check(u) == []
    # structural checks catch a lo that does not follow from the settled count, and a duplicated owner
    u.lo[5] = 3
    assert any("lo does not follow" in m for _, m in KT.check(u))
    w = KT.parse(t.body())
    w.set_live(20, 0xDEF, 0, 0)
    assert any("also live" in m for _, m in KT.check(w))
    # ---- tight mode: the band is only what steps in flight may write; every other position has a known writer
    T = KT.KVTable(64, 16, tight=True)
    assert T.masked_tokens() == 0 and T.source(5, 3) == "filler"
    # request A (fp 0xA) on blocks 5, 9: 20 tokens settled, 23 computed (3 in flight)
    T.update_tight({5: (0xA, 0, 20, 23), 9: (0xA, 1, 20, 23)})
    assert (int(T.lo[5]), int(T.hi[5])) == (16, 16) and (int(T.lo[9]), int(T.hi[9])) == (4, 7) and T.masked_tokens() == 3
    assert T.source(5, 15) == (0xA, 0) and T.source(9, 3) == (0xA, 1) and T.source(9, 4) is None and T.source(9, 7) == "filler"
    band = T.band()
    assert ref.ch_masked((int(band[9]) & 0xff, int(band[9]) >> 8), 5 * ref.CH_KV_TOK_WORDS)
    assert not ref.ch_masked((int(band[9]) & 0xff, int(band[9]) >> 8), 7 * ref.CH_KV_TOK_WORDS)
    # A finishes at 25 tokens; its blocks are freed once no step uses them: positions 0..8 of block 9 are A's
    T.update_tight({}, freed={5: (0xA, 0, 25), 9: (0xA, 1, 25)})
    assert T.masked_tokens() == 0 and T.source(9, 8) == (0xA, 1) and T.source(9, 9) == "filler"
    # request B takes block 9 as its block 0, prefill of 6 tokens in flight: the band is positions 0..5, the rest
    # still holds A's tokens (6, 7, 8) and then filler
    T.update_tight({9: (0xB, 0, 0, 6)})
    assert [T.source(9, t) for t in (0, 5, 6, 8, 9)] == [None, None, (0xA, 1), (0xA, 1), "filler"]
    # two calls later B's prefill is settled and it is decoding position 6
    T.update_tight({9: (0xB, 0, 6, 7)})
    assert [T.source(9, t) for t in (0, 5, 6, 7, 8, 9)] == [(0xB, 0), (0xB, 0), None, (0xA, 1), (0xA, 1), "filler"]
    assert KT.check(T) == []
    U2 = KT.parse(T.body())
    assert U2.tight and U2.digest() == T.digest() and (U2.band() == T.band()).all() and U2.source(9, 8) == (0xA, 1)
    assert KT.check(U2) == []
    # ---- history of committed tables (tight): lingering bands and decode pace
    H = KT.KVTable(64, 16, tight=True)
    fpA = (40 << 32) | 0x111                                   # prompt of 40 tokens
    seq = []
    comp = 0
    for v in range(1, 30):                                     # prefill 40 tokens in one step, then one token a call
        comp = 40 if v == 1 else comp + 1
        settled = 0 if v <= 3 else (40 if v == 4 else 40 + (v - 4) - 1)   # lagged: three calls behind
        H.update_tight({b: (fpA, b, settled, comp) for b in range(-(-comp // 16))})
        seq.append((H.version, KT.parse(H.body())))
    h = KT.history(seq)
    assert h["linger_gap_max"] == 3 and h["pace_excess_max"] <= 0 and h["pace_pairs"] > 10, h
    # every second table only (answers do not land on every version): same verdict
    h2 = KT.history(seq[::2])
    assert h2["linger_gap_max"] == 2 and h2["pace_excess_max"] <= 0, h2
    # a request that is preempted and recomputed in two chunks: no excess (an earlier sighting covers the recompute)
    R = KT.KVTable(64, 16, tight=True)
    rs = []
    for v, (st, cp) in enumerate([(0, 40), (0, 41), (40, 42), (41, 43), (42, 44), (43, 45)], 1):
        R.update_tight({b: (fpA, b, st, cp) for b in range(-(-cp // 16))})
        rs.append((R.version, KT.parse(R.body())))
    R.update_tight({}, freed={b: (fpA, b, 45) for b in range(3)})                     # preempted at 45 tokens
    rs.append((R.version, KT.parse(R.body())))
    for st, cp in [(0, 30), (0, 45), (30, 46), (45, 47), (46, 48)]:                    # back in: 30, then 15, then decode
        R.update_tight({b: (fpA, b, st, cp) for b in range(-(-cp // 16))})
        rs.append((R.version, KT.parse(R.body())))
    assert KT.history(rs)["pace_excess_max"] <= 0 and KT.history(rs[::3])["pace_excess_max"] <= 0
    # a band that stays put: position 5 of block 3 claimed in flight for 12 versions
    bad = [(v, KT.parse(t.body())) for v, t in seq]
    for v, t in bad[10:22]:
        t.kind[3], t.fp[3], t.idx[3], t.lo[3], t.hi[3] = KT.LIVE, fpA, 3, 5, 6
    assert KT.history(bad)["linger_gap_max"] == 11
    # version numbers that advance slower than the work: the same tables renumbered 1, 1.25, ... (4 calls per version)
    slow = [(1 + k // 4, t) for k, (v, t) in enumerate(seq)][::4]
    assert KT.history(slow)["pace_excess_max"] == 3, KT.history(slow)     # 4 tokens per version: 3 over the pace
    # writers: a position named as written by a request that is not that long (or not known) is counted
    T2 = KT.KVTable(64, 16, tight=True)
    T2.update_tight({b: (fpA, b, 40, 40) for b in range(3)})
    assert KT.writers_out_of_range(T2, {fpA: 40}) == 0 and KT.writers_out_of_range(T2, {fpA: 33}) == 7
    assert KT.writers_out_of_range(T2, {}) == 40
    # ---- session verdict: what a whole session's committed tables must satisfy (review cases from the main thread:
    # the same 256 blocks = 4,096 tokens = the band bound masked in every table of a session)
    NB, LAG = 10240, 2
    real_fp = (300 << 32) | 0xabc
    tok = {real_fp: 700}

    def static(kind, fp_of, idx, dv):
        tabs = []
        for k in range(60):
            t = KT.KVTable(NB, 16, tight=True)
            t.version = 100 + dv * k
            t.kind[:256], t.lo[:256], t.hi[:256] = kind, 0, 16
            if kind == KT.LIVE:
                t.fp[:256], t.idx[:256] = fp_of(k), idx
            tabs.append((t.version, KT.parse(t.body())))
        return KT.session_verdict(tabs, tok, lag=LAG)
    made_up = (1 << 63) | 7000
    v = static(KT.UNKNOWN, None, 0, 5)
    assert not v["ok"] and v["unknown_owner_blocks_max"] == 256, v                    # U: blocks of an unnamed owner
    v = static(KT.LIVE, lambda k: made_up + k, np.arange(256), 5)
    assert not v["ok"] and v["owners_invalid_max"] == 256, v                          # F: made-up owner, new each table
    v = static(KT.LIVE, lambda k: made_up, np.arange(256), 17)
    assert not v["ok"] and v["owners_invalid_max"] == 256, v                          # F2: one made-up owner, versions +17
    v = static(KT.LIVE, lambda k: real_fp, 0, 17)
    assert not v["ok"] and v["structure_bad"] == 60, v                                # G: a real request on 256 blocks at once
    # a real request, valid block indices, a band that never moves, versions jumping ahead: every promise is broken
    v = static(KT.LIVE, lambda k: real_fp, np.arange(256) % 40, 17)
    assert not v["ok"], v
    t1 = []
    for k in range(60):                                                              # (one block per index: structure is fine)
        t = KT.KVTable(NB, 16, tight=True)
        t.version = 100 + 17 * k
        t.kind[:40], t.lo[:40], t.hi[:40], t.fp[:40], t.idx[:40] = KT.LIVE, 0, 16, real_fp, np.arange(40)
        t1.append((t.version, KT.parse(t.body())))
    v = KT.session_verdict(t1, tok, lag=LAG)
    assert not v["ok"] and v["structure_bad"] == 0 and v["owners_invalid_max"] == 0 and v["linger_gap_max"] == 17 \
        and v["promises"]["broken"] == v["promises"]["positions"] > 0, v
    # rotating real owners over the same blocks (each one "preempted" before it settles): promises broken as well
    fps = [(300 << 32) | (0x100 + j) for j in range(8)]
    tok8 = {f: 700 for f in fps}
    t2 = []
    for k in range(60):
        t = KT.KVTable(NB, 16, tight=True)
        t.version = 100 + 4 * k
        t.kind[:16], t.lo[:16], t.hi[:16], t.fp[:16], t.idx[:16] = KT.LIVE, 0, 16, fps[k % 8], np.arange(16)
        t2.append((t.version, KT.parse(t.body())))
    v = KT.session_verdict(t2, tok8, lag=LAG)
    assert not v["ok"] and v["promises"]["still_in_band"] > 0, v
    # the honest sequence from above passes, also when only every second table is committed
    fpA_tok = {fpA: 80}
    for sub in (seq, seq[::2]):
        v = KT.session_verdict(sub, fpA_tok, lag=2)
        assert v["ok"], v
    assert KT.session_verdict(seq, fpA_tok, lag=2)["promises"]["positions"] > 0
    print("test_kvtable_cpu: PASS")


if __name__ == "__main__":
    main()
