"""Batched chain replay (verifier, CPU): many mixed-chase chains advanced together, round by round.

ref.ch_chain walks one chain in pure Python and asks for each probed 128-byte chunk through a callback that encodes
a whole 64 KiB page to return 32 words (~4 s of CPU per chain). Here N chains (one per challenge) advance in
lockstep: the 8 probe addresses of a round depend only on the chain state at the start of the round, so each round
is 8 N addresses, one gather of 8 N chunks from sources, and the mixing over [N, 32] lanes. Only the 16 words of a
probed chunk are encoded. Everything is integer arithmetic on uint64 (no floating point): the result is the exact
chain value or the replay is wrong. tests/test_chain_replay_cpu.py compares it with ref.ch_chain.

Chunk sources (sloth64 everywhere; anything else is left to the caller's scalar callback):
  W       the declared weights: word = sloth(raw word, the matrix's seed, word index in the matrix); bytes of the W
          buffer outside every matrix hold the arena filler; a matrix with key < 0 is stored as is
  filler  sloth(0, filler seed, byte address / 8)
  KV      from the block table that answer ran under: in the band -> the fixed stand-in (ref.ch_mask_words); never
          written since sealing -> arena filler; else the writer's KV from the reference store, sloth-encoded for the
          page it sits in
  ring / activation pool -> `scalar(job, page, word_off)`; with tagged ring reads a slot in the zone holds that
          window or the incoming one (`scalar_new`), chosen by the tag the kernel returned
"""
from __future__ import annotations

import numpy as np

from schemes.memory_link_exhaustion.encodings import ref, sloth64

U = np.uint64
M32 = U(0xFFFFFFFF)
PAGE = ref.PAGE


def mix64(z):
    z = z ^ (z >> U(30))
    z = z * U(0xBF58476D1CE4E5B9)
    z = z ^ (z >> U(27))
    z = z * U(0x94D049BB133111EB)
    return z ^ (z >> U(31))


def scale(h, n: int):
    assert 0 < n < (1 << 31)
    lo = ((h & M32) * U(n)) >> U(32)
    return (((h >> U(32)) * U(n) + lo) >> U(32)) & M32


def supported() -> bool:
    """The vector walk covers the frozen kernel only (mixing round, 128-byte probes)."""
    return bool(ref.CH_MIX) and ref.CH_WIDE == 1


def walk(nonces, chains, P: int, total_blocks: int, nbk: int, gran_page, chunks, progress=None):
    """Chain values of N chains: chain chains[i] of the chase with (effective) nonce nonces[i]. chunks(j, page, wo)
    -> uint32 [M, 32] for M probes: j = index of the chain in this call, page / wo int64 [M]. Returns uint64 [N]."""
    assert supported()
    with np.errstate(over="ignore"):
        nonces = np.asarray(nonces, dtype=U) & M32
        N = nonces.size
        n64 = mix64(nonces)
        b0 = ((mix64(n64) % U(total_blocks - nbk + 1)) * U(ref.CH_GRAN_PER_BLOCK)).astype(np.int64)
        nch = nbk << 25
        gold = U(ref.CH_GOLD)
        s = mix64(n64 ^ mix64((np.asarray(chains, dtype=U) + U(1)) * gold))
        gp = np.asarray([int(x) for x in gran_page], dtype=np.int64)
        lw = np.array([pow(31, bin(l).count("1"), 1 << 64) for l in range(32)], dtype=U)
        lane = np.arange(32, dtype=U)
        jj = np.repeat(np.arange(N, dtype=np.int64), 8)
        c8 = np.arange(1, 9, dtype=U)
        for r in range(P):
            g = scale(mix64(s[:, None] ^ ((U(r * 8) + c8) * gold)[None, :]), nch).astype(np.int64)      # [N, 8]
            page = gp[b0[:, None] + (g >> 19)] + ((g & 0x7FFFF) >> 9)
            wo = (g & 511) * 32
            w = chunks(jj, page.reshape(-1), wo.reshape(-1)).reshape(N, 8, 32).astype(U)
            t = s[:, None] + lane[None, :]
            for c in range(8):
                x = t ^ w[:, c, :]
                t = ((x << U(23)) | (x >> U(41))) * U(ref.CH_MULC)
            s = (t * lw[None, :]).sum(axis=1, dtype=U) ^ U((r * ref.CH_GOLD) & ref.M64)
            if progress is not None:
                progress(r)
        return s


class Sources:
    """chunks(j, page, wo) for walk(). bind() sets the per-chain context (nonce, block table) of the next walk."""

    def __init__(self, *, nW: int, nK: int, nb: int, L: int, layout=None, weights=None, fkey: int = 0, kv_seeds=None,
                 store=None, fpi=None, scalar=None, scalar_new=None, ring_zone=None, kv_ks=None):
        assert ref.WENC in ("sloth", "bfly") and ref.FILLER in ("sloth", "bfly"), "batched replay: sloth64 or butterfly W and filler"
        self.fkey = int(fkey)
        self._fcache = {}
        # kv_ks(layer, block, wo) -> uint32 [32]: butterfly KV keystream of a written unit (None: sloth64 KV)
        self.kv_ks = kv_ks
        self.nW, self.nK, self.nb, self.L = int(nW), int(nK), int(nb), int(L)
        self.fseed = U(sloth64.filler_seed(fkey))
        self.w = weights
        ents = sorted([e for e in (layout.entries if layout is not None else []) if e.sources is not None],
                      key=lambda e: e.offset)
        self.ents = ents
        self.e_off = np.array([e.offset for e in ents], dtype=np.int64)
        self.e_end = np.array([e.offset + e.nbytes for e in ents], dtype=np.int64)
        self.e_seed = [U(ref.sloth_seed(e.key, ref.w_ordinal(e.offset))) if e.key >= 0 else None for e in ents]
        assert all(o % 8 == 0 for o in self.e_off) and all(o % 8 == 0 for o in self.e_end), "W matrices: 8-byte aligned"
        self._flat = {}
        self.kv_seeds = np.asarray(kv_seeds if kv_seeds is not None else [], dtype=U)
        self.store = store                         # request index -> uint16 array ([block idx][layer] x 64 KiB) or None
        fpi = fpi or {}
        ks = sorted(fpi)
        self.fp_keys = np.array(ks, dtype=U)
        self.fp_req = np.array([fpi[k] for k in ks], dtype=np.int64)
        self.scalar = scalar
        # tagged ring reads: ring_zone = (ring slots, zone); scalar_new(j, page, wo) = the incoming window's words
        self.scalar_new, self.ring_zone = scalar_new, (ring_zone if scalar_new is not None else None)
        self.stats = {"probes": 0, "w": 0, "filler": 0, "kv": 0, "kv_masked": 0, "kv_filler": 0, "kv_live": 0, "kv_prev_owner": 0, "other": 0,
                      "band_new": 0, "band_old": 0, "band_neither": 0, "band_both": 0,
                      "ring_new": 0, "ring_old": 0, "ring_neither": 0}
        self._lw = np.array([pow(31, bin(l).count("1"), 1 << 64) for l in range(32)], dtype=U)
        self.why = {}
        self.miss = None

    def _flat64(self, ei: int):
        f = self._flat.get(ei)
        if f is None:
            e = self.ents[ei]
            raw = self.w.flat(e.sources[0]) if len(e.sources) == 1 else self.w.range(e.sources, 0, e.nbytes // 2)
            assert raw.size * 2 == e.nbytes, (e.name, raw.size, e.nbytes)
            f = self._flat[ei] = np.ascontiguousarray(raw).view(U)
        return f

    def _fill32(self, page, wo):
        """Butterfly filler (kvbfly.filler_units): uint32 [m, 32] of the 128-byte chunks at (page, word offset wo)."""
        from schemes.memory_link_exhaustion.encodings import kvbfly
        page, wo = np.asarray(page, dtype=np.int64), np.asarray(wo, dtype=np.int64)
        key = page * 16 + (wo >> 10)
        out = np.empty((page.size, 32), dtype=np.uint32)
        if len(self._fcache) > 50000:
            self._fcache.clear()
        need = [k for k in np.unique(key).tolist() if k not in self._fcache]
        for i in range(0, len(need), 4096):
            ks = np.asarray(need[i:i + 4096], dtype=np.int64)
            for k, u in zip(ks.tolist(), kvbfly.filler_units(self.fkey, ks >> 4, ks & 15)):
                self._fcache[k] = u
        for x, (k, w) in enumerate(zip(key.tolist(), (wo & 1023).tolist())):
            out[x] = self._fcache[k][w:w + 32]
        return out

    def _wks64(self, ei: int, wi):
        """Butterfly W keystream (uint64) of words wi of matrix ei (encodings/bfly_ref.py layout: 128 x 128
        super-chunks, unit = 8 columns of one row; a 64-bit word is half a unit)."""
        from schemes.memory_link_exhaustion.encodings import bfly_ref
        e = self.ents[ei]
        ncols = int(e.shape[1])
        kpr, o = ncols // 128, int(ref.w_ordinal(e.offset))
        j = wi.astype(np.int64) * 4
        r, c = np.divmod(j, ncols)
        sc = (r // 128) * kpr + c // 128
        u = (r % 128) * 16 + (c % 128) // 8
        half = (c % 8) // 4
        out = np.empty(wi.size, dtype=U)
        for v in np.unique(sc).tolist():
            m = sc == v
            ks = bfly_ref.superchunk_ks((o << 32) | v, e.key).view(U).reshape(-1, 2)
            out[m] = ks[u[m], half[m]]
        return out

    def _req_of(self, fps):
        """Request index of each fingerprint (uint64 array); -1 if no request of the workload has it."""
        fps = np.asarray(fps, dtype=U)
        if self.fp_keys.size == 0:
            return np.full(fps.shape, -1, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self.fp_keys, fps), self.fp_keys.size - 1)
        return np.where(self.fp_keys[pos] == fps, self.fp_req[pos], -1)

    def bind(self, nonces, tables=None, tags=None):
        """nonces: effective nonce of each chain; tables: the kvtable.KVTable each chain's answer ran under (the same
        object may appear many times), or None when the walk has no KV probes to resolve."""
        from schemes.memory_link_exhaustion.inference import kvtable as KT
        with np.errstate(over="ignore"):
            self.n64 = mix64(np.asarray(nonces, dtype=U) & M32)
        self.miss = np.zeros(self.n64.size, dtype=np.int64)
        # tagged in-flight reads (--kv-table 3): per chain [number of in-band reads, their tags] as the kernel returned
        self.tags = None if tags is None else np.stack([np.asarray(t, dtype=np.uint32) for t in tags])
        self.nband = np.zeros(self.n64.size, dtype=np.int64)       # in-band reads met so far, per chain
        self.neither = np.zeros(self.n64.size, dtype=np.int64)     # ... that matched neither old nor new content
        self.tag_over = np.zeros(self.n64.size, dtype=bool)        # more in-band reads than tag slots
        if tables is None:
            self.tix = None
            return
        uniq, tix = {}, []
        for t in tables:
            tix.append(uniq.setdefault(id(t), len(uniq)))
        ts = [None] * len(uniq)
        for t in tables:
            ts[uniq[id(t)]] = t
        self.tix = np.array(tix, dtype=np.int64)
        self.tight = bool(ts[0].tight)
        assert all(bool(t.tight) == self.tight and t.nb == self.nb for t in ts)
        self.bs = ts[0].bs
        self.LO = np.stack([t.lo for t in ts]).astype(np.int64)
        self.HI = np.stack([t.hi for t in ts]).astype(np.int64)
        if self.tight:
            self.WR = np.stack([t.writer for t in ts])
            wl = [np.array(t.writers, dtype=U).reshape(-1, 2) for t in ts]
            self.wbase = np.cumsum([0] + [len(w) for w in wl[:-1]]).astype(np.int64)
            allw = np.concatenate(wl)
            self.g_req = self._req_of(allw[:, 0])
            self.g_bi = allw[:, 1].astype(np.int64)
            self.g_fp = allw[:, 0]
            self.KINDT = np.stack([t.kind for t in ts])
            self.OWN_FP = np.stack([t.fp for t in ts])
            self.OWN_IDX = np.stack([t.idx for t in ts]).astype(np.int64)
            self._LIVE = KT.LIVE
        else:
            self.KIND = np.stack([t.kind for t in ts])
            self.REQ = self._req_of(np.stack([t.fp for t in ts]))
            self.IDX = np.stack([t.idx for t in ts]).astype(np.int64)
            self._FILLER = KT.FILLER

    def _miss(self, j, what, ex=None):
        self.miss[j] += 1
        self.why[what] = self.why.get(what, 0) + 1
        if ex is not None and len(self.why.setdefault("examples", [])) < 4:
            self.why["examples"].append(ex)

    def _kv_words(self, page, wo, layer, block, req, bi, has):
        """Encoded 128-byte chunks of KV rows from their source: has False -> never written (the arena filler of that
        page), else request req's block bi from the store. Returns (uint32 [m, 32], ok [m]): ok False where the
        store does not have the position (those rows are zeros)."""
        m = page.size
        raw = np.zeros((m, 16), dtype=U)
        lane = np.arange(16, dtype=U)
        wa = (page * PAGE + wo * 4)[:, None] + 8 * np.arange(16, dtype=np.int64)[None, :]
        seed = np.full((m, 16), self.fseed, dtype=U)
        idx = (wa >> 3).astype(U)
        ok = np.ones(m, dtype=bool)
        o16 = ((bi * self.L + layer) * PAGE + wo * 4) // 2
        for x in np.nonzero(has)[0].tolist():
            i = int(req[x])
            f = self.store(i) if i >= 0 else None
            o = int(o16[x])
            if f is None or o < 0 or o + 64 > f.size:
                ok[x] = False
                continue
            raw[x] = np.ascontiguousarray(f[o:o + 64]).view(U)
            seed[x] = self.kv_seeds[int(layer[x])]
            idx[x] = U(int(block[x]) * (PAGE // 8) + int(wo[x]) // 2) + lane
        y = sloth64.encode_gather(raw.reshape(-1), seed.reshape(-1), idx.reshape(-1))
        w = np.ascontiguousarray(y).view(np.uint32).reshape(m, 32).copy()
        if ref.FILLER == "bfly" and (~has).any():
            nh = np.nonzero(~has)[0]
            w[nh] = self._fill32(page[nh], wo[nh])
        if self.kv_ks is not None:              # butterfly KV: a written unit is its plaintext XOR the keystream
            for x in np.nonzero(has & ok)[0].tolist():
                w[x] = raw[x].view(np.uint32) ^ self.kv_ks(int(layer[x]), int(block[x]), int(wo[x]))
        w[~ok] = 0
        return w, ok

    def _tag_order(self, j, rows, M):
        """Index of each tagged read inside its chain, for the tagged rows of this call (in row order: chain, then
        probe). Returns int64 [M] (-1 for untagged rows) and advances the per-chain counters."""
        kmap = np.full(M, -1, dtype=np.int64)
        if rows.size:
            jb = j[rows]
            order = np.argsort(jb, kind="stable")
            jb_s = jb[order]
            first = np.r_[0, np.nonzero(np.diff(jb_s))[0] + 1]
            rank = np.arange(jb_s.size) - np.repeat(first, np.diff(np.r_[first, jb_s.size]))
            kmap[rows[order]] = self.nband[jb_s] + rank
            np.add.at(self.nband, jb_s, 1)
        return kmap

    def _tags(self, jrow, cid, words):
        """ref.ch_tag for rows: chain jrow's nonce, chunk id cid, uint32 words [m, 32]."""
        with np.errstate(over="ignore"):
            h = (words.astype(U) * self._lw[None, :]).sum(axis=1, dtype=U)
            return (mix64(h ^ self.n64[jrow] ^ mix64(U(ref.CH_TAG_SALT) ^ cid)) >> U(32)).astype(np.uint32)

    def __call__(self, j, page, wo):
        nW, nK, nb, st = self.nW, self.nK, self.nb, self.stats
        M = page.size
        out = np.zeros((M, 32), dtype=np.uint32)
        direct = np.zeros(M, dtype=bool)                       # rows written straight into out (not encoded)
        isW = page < nW
        isK = (page >= nW) & (page < nW + nK)
        other = page >= ref.CH_PA0
        st["probes"] += M; st["w"] += int(isW.sum()); st["kv"] += int(isK.sum()); st["other"] += int(other.sum())
        st["filler"] += int(M - isW.sum() - isK.sum() - other.sum())
        # default for every arena row: the filler word at that byte address
        wa = (page * PAGE + wo * 4)[:, None] + 8 * np.arange(16, dtype=np.int64)[None, :]         # byte address per word
        raw = np.zeros((M, 16), dtype=U)
        seed = np.full((M, 16), self.fseed, dtype=U)
        idx = (wa >> 3).astype(U)
        enc = np.ones((M, 16), dtype=bool)
        fil = np.ones((M, 16), dtype=bool)                    # words that hold the arena filler
        # --- W: words inside a declared matrix
        wr = np.nonzero(isW)[0]
        if wr.size and self.e_off.size:
            a = wa[wr].reshape(-1)
            e = np.searchsorted(self.e_off, a, "right") - 1
            cov = np.nonzero((e >= 0) & (a < self.e_end[np.maximum(e, 0)]))[0]
            if cov.size:
                order = cov[np.argsort(e[cov], kind="stable")]
                es = e[order]
                rw, sw, iw, ew = raw[wr].reshape(-1), seed[wr].reshape(-1), idx[wr].reshape(-1), enc[wr].reshape(-1)
                fw = fil[wr].reshape(-1)
                fw[cov] = False
                fil[wr] = fw.reshape(-1, 16)
                for seg in np.split(order, np.nonzero(np.diff(es))[0] + 1):
                    ei = int(e[seg[0]])
                    wi = (a[seg] - self.e_off[ei]) >> 3
                    rw[seg] = self._flat64(ei)[wi]
                    if self.e_seed[ei] is None:
                        ew[seg] = False
                    elif ref.WENC == "bfly":
                        rw[seg] ^= self._wks64(ei, wi)
                        ew[seg] = False
                    else:
                        sw[seg] = self.e_seed[ei]
                        iw[seg] = wi.astype(U)
                raw[wr], seed[wr], idx[wr], enc[wr] = rw.reshape(-1, 16), sw.reshape(-1, 16), iw.reshape(-1, 16), ew.reshape(-1, 16)
        # --- tagged ring reads (ring_zone = (n_ring, zone)): rows in the zone's logical slots
        rz, kmap = np.zeros(0, dtype=np.int64), None
        if self.tags is not None and self.ring_zone is not None:
            isR = page >= ref.CH_RA0
            jg, pg = np.divmod(np.where(isR, page - ref.CH_RA0, 0), ref.CH_GRAN_PAGES)
            half = ref.CH_GRAN_PAGES // 2
            rq = 2 * jg + pg // half
            rwin = (pg % half) * (PAGE // 4) + wo
            rz = np.nonzero(isR & (rq >= self.ring_zone[0] - self.ring_zone[1]))[0]
        # --- KV: by the block table of each chain's answer
        kr = np.nonzero(isK)[0]
        if kr.size:
            assert self.tix is not None, "KV probes need the block tables (bind(..., tables))"
            kp = page[kr] - nW
            layer, block = kp // nb, kp % nb
            wok = wo[kr]
            tok = wok // ref.CH_KV_TOK_WORDS
            jk = j[kr]
            ti = self.tix[jk]
            masked = (self.LO[ti, block] <= tok) & (tok < self.HI[ti, block])
            mi = np.nonzero(masked)[0]
            st["kv_masked"] += int(mi.size)
            if self.tags is not None:
                kmap = self._tag_order(j, np.sort(np.r_[kr[mi], rz]), M)
            # what a position holds outside a band (and held before the step in flight, inside one): its last writer
            if self.tight:
                wid = self.WR[ti, block, np.minimum(tok, self.bs - 1)].astype(np.int64)
                has = wid != 0
                g = np.where(has, self.wbase[ti] + wid, 0)
                req, bi = self.g_req[g], self.g_bi[g]
                cur = (self.KINDT[ti, block] == self._LIVE) & (self.OWN_FP[ti, block] == self.g_fp[g]) & \
                    (self.OWN_IDX[ti, block] == bi)
                st["kv_prev_owner"] += int((has & ~masked & ~cur).sum())
            else:
                has = self.KIND[ti, block] != self._FILLER
                req, bi = self.REQ[ti, block], self.IDX[ti, block]
            tagged = self.tags is not None
            assert not tagged or self.tight, "tagged in-flight reads: tight tables only"
            sel = np.nonzero(~masked)[0] if not tagged else np.arange(kr.size)
            w_old, ok_old = self._kv_words(page[kr[sel]], wok[sel], layer[sel], block[sel], req[sel], bi[sel], has[sel])
            st["kv_live"] += int((has & ~masked).sum())
            st["kv_filler"] += int((~has & ~masked).sum())
            out[kr[sel]] = w_old
            direct[kr[sel]] = True
            nm = sel[~masked[sel] & ~ok_old]                    # a read position whose writer the store does not have
            for x in nm.tolist():
                self._miss(int(jk[x]), "fp_unknown" if req[x] < 0 else "past_end",
                           None if req[x] < 0 else {"req": int(req[x]), "idx": int(bi[x])})
            if mi.size and not tagged:
                # settled-KV rule without tags: a band position is not read, it contributes the fixed stand-in
                with np.errstate(over="ignore"):
                    cid = (kp[mi] * 512 + wok[mi] // 32).astype(U)
                    base = self.n64[jk[mi]] ^ mix64(U(ref.CH_MASK_SALT) ^ cid)
                    lg = (np.arange(1, 33, dtype=U) * U(ref.CH_GOLD))[None, :]
                    out[kr[mi]] = (mix64(base[:, None] ^ lg) & M32).astype(np.uint32)
                direct[kr[mi]] = True
            elif mi.size:
                # tagged in-flight reads: the kernel read the position and returned a tag of what it read. Old = the
                # last writer's bytes (above), new = the block's live owner's KV; the tag says which one it was.
                req_n = self._req_of(self.OWN_FP[ti[mi], block[mi]])
                live_n = self.KINDT[ti[mi], block[mi]] == self._LIVE
                w_new, ok_new = self._kv_words(page[kr[mi]], wok[mi], layer[mi], block[mi], np.where(live_n, req_n, -1),
                                               self.OWN_IDX[ti[mi], block[mi]], np.ones(mi.size, dtype=bool))
                w_o, ok_o = out[kr[mi]], ok_old[np.searchsorted(sel, mi)]
                cid = (kp[mi] * 512 + wok[mi] // 32).astype(U)
                t_old, t_new = self._tags(jk[mi], cid, w_o), self._tags(jk[mi], cid, w_new)
                # the index of each tagged read inside its chain (KV band and ring zone reads share one sequence)
                jb = jk[mi]
                k = kmap[kr[mi]]
                over = k >= ref.CH_TAG_SLOTS - 1
                self.tag_over[jb[over]] = True
                want = self.tags[jb, np.minimum(k + 1, ref.CH_TAG_SLOTS - 1)]
                is_new = ok_new & (t_new == want) & ~over
                is_old = ok_o & (t_old == want) & ~over
                st["band_new"] += int((is_new & ~is_old).sum())
                st["band_old"] += int((is_old & ~is_new).sum())
                st["band_both"] += int((is_new & is_old).sum())
                neither = ~is_new & ~is_old & ~over
                st["band_neither"] += int(neither.sum())
                np.add.at(self.neither, jb[neither], 1)
                pick = is_new & ~is_old
                rows = kr[mi]
                out[rows[pick]] = w_new[pick]
        # --- ring / activation pool: the caller's scalar source
        for x in np.nonzero(other)[0].tolist():
            out[x] = self.scalar(int(j[x]), int(page[x]), int(wo[x]))
            direct[x] = True
        if rz.size:
            # tagged ring reads: a slot in the zone holds its old window (above) or the incoming one; the tag says which
            if kmap is None:
                kmap = self._tag_order(j, rz, M)
            w_o = out[rz]
            w_n = np.stack([np.asarray(self.scalar_new(int(j[x]), int(page[x]), int(wo[x])), dtype=np.uint32) for x in rz.tolist()])
            cid = (U(ref.CH_RING_TAG_ID) | (rq[rz].astype(U) << U(18)) | (rwin[rz] // 32).astype(U))
            jr = j[rz]
            t_old, t_new = self._tags(jr, cid, w_o), self._tags(jr, cid, w_n)
            k = kmap[rz]
            over = k >= ref.CH_TAG_SLOTS - 1
            self.tag_over[jr[over]] = True
            want = self.tags[jr, np.minimum(k + 1, ref.CH_TAG_SLOTS - 1)]
            is_new = (t_new == want) & ~over
            is_old = (t_old == want) & ~over
            st["ring_new"] += int((is_new & ~is_old).sum())
            st["ring_old"] += int(is_old.sum())
            neither = ~is_new & ~is_old & ~over
            st["ring_neither"] += int(neither.sum())
            np.add.at(self.neither, jr[neither], 1)
            pick = is_new & ~is_old
            out[rz[pick]] = w_n[pick]
        rows = np.nonzero(~direct)[0]
        if rows.size and ref.FILLER == "bfly":
            # butterfly filler words are the keystream itself (no encode step)
            fr = rows[fil[rows].any(1)]
            if fr.size:
                f64 = np.ascontiguousarray(self._fill32(page[fr], wo[fr])).view(U).reshape(-1, 16)
                m_ = fil[fr]
                rf, ef = raw[fr], enc[fr]
                rf[m_] = f64[m_]
                ef[m_] = False
                raw[fr], enc[fr] = rf, ef
        if rows.size:
            r_, e_ = raw[rows].reshape(-1), enc[rows].reshape(-1)
            y = sloth64.encode_gather(r_, seed[rows].reshape(-1), idx[rows].reshape(-1))
            if not e_.all():
                y = np.where(e_, y, r_)
            out[rows] = np.ascontiguousarray(y).view(np.uint32).reshape(-1, 32)
        return out
