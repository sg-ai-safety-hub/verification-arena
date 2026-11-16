"""KV block table for the settled-KV rule (pure Python / numpy: no torch, CPU-testable).

One entry per KV block (a block is 16 token positions in every layer). The chase kernel reads `band`, uint16
lo | hi << 8: positions lo <= t < hi may still be written by a step in flight and are not read from memory
(encodings/ref.py ch_masked). This first version is conservative: hi is always the block size, so everything from the
first position that is not certainly settled up to the end of the block is masked.

  owner kinds   FILLER  never allocated since sealing: the initial K filler, fully readable (lo = block size)
                LIVE    block i of a running request: lo = positions of it settled `lag` schedule() calls ago
                STALE   freed, last owner known (prompt fingerprint, block index, settled count at the free)
                UNKNOWN freed, owner not in the snapshot when it was freed: nothing readable (lo = 0)
                FRESH   allocated in the current schedule() call, not yet in a snapshot: nothing readable (lo = 0)

The table and its owners are what the prover returns next to a timed chase answer; digest() is the value that goes
inside the timed answer so the table cannot be chosen after the nonce is seen.

Tight mode (KVTable(..., tight=True)): the band of a live block is only [settled `lag` calls ago, computed now), the
positions steps in flight may write, and every position outside a band is read. That needs to know what each
position last held: `writer[block, position]` is the id of the request block that last wrote it (0 = the filler the
arena was sealed with), `writers[id] = (fp, block index in the request)`. A position keeps its writer across frees
and reallocations until a new owner's settled count passes it.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np

FILLER, LIVE, STALE, UNKNOWN, FRESH = 0, 1, 2, 3, 4
KIND_NAMES = {FILLER: "filler", LIVE: "live", STALE: "stale", UNKNOWN: "unknown", FRESH: "fresh"}


class KVTable:
    def __init__(self, n_blocks: int, block_tokens: int = 16, tight: bool = False):
        self.nb, self.bs = int(n_blocks), int(block_tokens)
        self.tight = bool(tight)
        self.hi = np.full(self.nb, self.bs, dtype=np.uint8)      # upper edge of the band (tight mode; else block size)
        self.writer = np.zeros((self.nb, self.bs), dtype=np.uint32) if tight else None
        self.writers = [(0, 0)]                                  # id -> (fp, block index); id 0 = filler
        self._wid = {}
        self.kind = np.full(self.nb, FILLER, dtype=np.uint8)
        self.lo = np.full(self.nb, self.bs, dtype=np.uint8)
        self.idx = np.zeros(self.nb, dtype=np.uint32)           # block index inside the owning request
        self.settled = np.zeros(self.nb, dtype=np.uint32)       # the owner's settled token count used for lo
        self.fp = np.zeros(self.nb, dtype=np.uint64)            # owner prompt fingerprint (low 64 bits)
        self.version = 0

    def _lo(self, settled: int, i: int) -> int:
        return max(0, min(self.bs, int(settled) - int(i) * self.bs))

    def set_live(self, block: int, fp: int, i: int, settled: int):
        self.kind[block], self.idx[block], self.settled[block] = LIVE, i, settled
        self.fp[block] = int(fp) & 0xFFFFFFFFFFFFFFFF
        self.lo[block] = self._lo(settled, i)

    def set_stale(self, block: int, owner):
        """owner: (fp, block index, settled count at the free) or None when the owner was not known."""
        if owner is None:
            self.kind[block], self.lo[block], self.idx[block], self.settled[block], self.fp[block] = UNKNOWN, 0, 0, 0, 0
            return
        fp, i, settled = owner
        self.kind[block], self.idx[block], self.settled[block] = STALE, i, settled
        self.fp[block] = int(fp) & 0xFFFFFFFFFFFFFFFF
        self.lo[block] = self._lo(settled, i)

    def set_fresh(self, block: int):
        self.kind[block], self.lo[block], self.idx[block], self.settled[block], self.fp[block] = FRESH, 0, 0, 0, 0

    def update(self, live: dict, fresh=(), freed: dict | None = None):
        """One schedule() call. live: {block: (fp, index in request, settled count, lagged)} for every block of a
        running request; fresh: blocks allocated in this call; freed: {block: owner | None} for blocks freed since the
        last call. Blocks in none of them keep their entry. Returns the new version."""
        assert not self.tight, "tight tables are driven by update_tight"
        for b, o in (freed or {}).items():
            self.set_stale(b, o)
        for b in fresh:
            self.set_fresh(b)
        for b, (fp, i, settled) in live.items():
            self.set_live(b, fp, i, settled)
        self.version += 1
        return self.version

    def _writer_id(self, fp: int, i: int) -> int:
        k = (int(fp) & 0xFFFFFFFFFFFFFFFF, int(i))
        w = self._wid.get(k)
        if w is None:
            w = self._wid[k] = len(self.writers)
            self.writers.append(k)
        return w

    def update_tight(self, live: dict, freed: dict | None = None):
        """Tight mode, one schedule() call. live: {block: (fp, index in request, settled, computed)} with settled
        the count `lag` calls ago and computed the count including the steps now in flight; freed: {block: (fp,
        index, computed at the free)} for blocks freed since the last call (frees are deferred until no step uses the
        block, so everything computed is written by then). Positions below a block's settled edge become its
        owner's; the band is [settled edge, computed edge); positions above keep their previous writer."""
        assert self.tight
        bs = self.bs
        for b, o in (freed or {}).items():
            self.idx[b], self.settled[b], self.fp[b] = 0, 0, 0
            if o is None:
                # freed by an owner the prover could not name: nothing in the block is readable until a new owner
                # settles over it (kept masked; the verifier counts these blocks)
                self.kind[b], self.lo[b], self.hi[b] = UNKNOWN, 0, bs
                continue
            fp, i, computed = o
            e = self._lo(computed, i)
            if e > 0:
                self.writer[b, :e] = self._writer_id(fp, i)
            self.kind[b], self.lo[b], self.hi[b] = STALE, bs, bs          # no band: nothing writes a free block
        for b, (fp, i, settled, computed) in live.items():
            lo, hi = self._lo(settled, i), self._lo(computed, i)
            if lo > 0:
                self.writer[b, :lo] = self._writer_id(fp, i)
            self.kind[b], self.idx[b], self.settled[b] = LIVE, i, settled
            self.fp[b] = int(fp) & 0xFFFFFFFFFFFFFFFF
            self.lo[b], self.hi[b] = lo, max(lo, hi)
        self.version += 1
        return self.version

    def band(self) -> np.ndarray:
        """uint16 [n_blocks] for the kernel: lo | hi << 8 (hi = block size outside tight mode)."""
        return (self.lo.astype(np.uint16) | (self.hi.astype(np.uint16) << 8)).astype(np.uint16)

    def body(self) -> bytes:
        """The table as sent to the verifier: version, sizes, then kind / lo / idx / settled / fp arrays; in tight mode
        also hi, the per-position writer ids and the writer list."""
        b = (struct.pack("<QII", self.version, self.nb, self.bs | (0x10000 if self.tight else 0)) + self.kind.tobytes()
             + self.lo.tobytes() + self.idx.astype("<u4").tobytes() + self.settled.astype("<u4").tobytes()
             + self.fp.astype("<u8").tobytes())
        if self.tight:
            w = np.array(self.writers, dtype=np.uint64).reshape(-1, 2)
            b += self.hi.tobytes() + self.writer.astype("<u4").tobytes() + struct.pack("<I", len(w)) + w.astype("<u8").tobytes()
        return b

    def digest(self) -> int:
        """64-bit digest of body() (BLAKE2b): returned inside the timed answer."""
        return int.from_bytes(hashlib.blake2b(self.body(), digest_size=8).digest(), "little")

    def masked_tokens(self) -> int:
        """Token positions (per layer) currently not read by the chase."""
        return int((self.hi.astype(np.int64) - self.lo.astype(np.int64)).clip(min=0).sum())

    def source(self, block: int, tok: int):
        """What position `tok` of `block` holds, for the verifier: None (in the band: not read), "filler", or
        (fp, block index in the request)."""
        if int(self.lo[block]) <= tok < int(self.hi[block]):
            return None
        if self.tight:
            w = int(self.writer[block, tok])
            return "filler" if w == 0 else self.writers[w]
        k = int(self.kind[block])
        if k == FILLER:
            return "filler"
        return (int(self.fp[block]), int(self.idx[block]))


def parse(body: bytes) -> KVTable:
    version, nb, bsf = struct.unpack_from("<QII", body, 0)
    bs, tight = bsf & 0xFFFF, bool(bsf & 0x10000)
    t = KVTable(nb, bs, tight=tight)
    o = 16
    t.kind = np.frombuffer(body, np.uint8, nb, o).copy(); o += nb
    t.lo = np.frombuffer(body, np.uint8, nb, o).copy(); o += nb
    t.idx = np.frombuffer(body, "<u4", nb, o).astype(np.uint32); o += 4 * nb
    t.settled = np.frombuffer(body, "<u4", nb, o).astype(np.uint32); o += 4 * nb
    t.fp = np.frombuffer(body, "<u8", nb, o).astype(np.uint64); o += 8 * nb
    if tight:
        t.hi = np.frombuffer(body, np.uint8, nb, o).copy(); o += nb
        t.writer = np.frombuffer(body, "<u4", nb * bs, o).astype(np.uint32).reshape(nb, bs); o += 4 * nb * bs
        (nw,) = struct.unpack_from("<I", body, o); o += 4
        w = np.frombuffer(body, "<u8", 2 * nw, o).reshape(-1, 2); o += 16 * nw
        t.writers = [(int(a), int(b)) for a, b in w]
    assert o == len(body), (o, len(body))
    t.version = version
    return t


def read_log(path: str) -> dict:
    """{version: body} from the prover's table log (uint32 length + zlib(body) records)."""
    import zlib
    out = {}
    with open(path, "rb") as f:
        while True:
            h = f.read(4)
            if len(h) < 4:
                break
            z = f.read(int.from_bytes(h, "little"))
            try:
                body = zlib.decompress(z)
            except zlib.error:
                break                      # a record cut off by the end of the run
            out[struct.unpack_from("<Q", body, 0)[0]] = body
    return out


def body_digest(body: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(body, digest_size=8).digest(), "little")


def check(t: KVTable, mult=None) -> list:
    """Structural checks the verifier runs on every table (the history and client-side bounds need more context):
    lo consistent with (kind, idx, settled); no (fp, idx) owns two live blocks."""
    bad = []
    bs = t.bs
    if t.tight:
        if (t.lo > t.hi).any() or (t.hi > bs).any():
            bad.append((-1, "band edges out of order"))
        if int(t.writer.max(initial=0)) >= len(t.writers):
            bad.append((-1, "writer id without an entry"))
        free = np.nonzero((t.kind != LIVE) & (t.kind != UNKNOWN) & (t.lo != t.hi))[0]
        if free.size:
            bad.append((int(free[0]), "a block that is not live has a band"))
        # one request block lives in one KV block (a prompt the workload holds m times may be live m times: mult)
        live = np.nonzero(t.kind == LIVE)[0]
        if live.size:
            keys = np.stack([t.fp[live], t.idx[live].astype(np.uint64)], axis=1)
            u, cnt = np.unique(keys, axis=0, return_counts=True)
            for (f, i), c in zip(u[cnt > 1], cnt[cnt > 1]):
                if int(c) > int((mult or {}).get(int(f), 1)):
                    bad.append((-1, f"request block ({int(f):#x}, {int(i)}) live in {int(c)} blocks"))
                    break
        return bad
    for b in range(t.nb):
        k, lo = int(t.kind[b]), int(t.lo[b])
        if k == FILLER and lo != bs:
            bad.append((b, "filler block not fully readable"))
        elif k in (UNKNOWN, FRESH) and lo != 0:
            bad.append((b, "unknown / fresh block with readable positions"))
        elif k in (LIVE, STALE) and lo != max(0, min(bs, int(t.settled[b]) - int(t.idx[b]) * bs)):
            bad.append((b, "lo does not follow from the settled count and block index"))
        elif k not in KIND_NAMES:
            bad.append((b, "unknown kind"))
    live = np.nonzero(t.kind == LIVE)[0]
    seen = {}
    for b in live:
        key = (int(t.fp[b]), int(t.idx[b]))
        if key in seen:
            bad.append((int(b), f"request block also live at block {seen[key]}"))
        seen[key] = int(b)
    return bad


def request_counts(t: KVTable):
    """{fp: (settled, computed)} of the live requests of a table. settled: the largest settled value over a
    request's live blocks (in tight mode only the blocks at or after the settled edge are refreshed); computed: the
    upper edge of its band (block index x block size + hi of its last block with a band), or settled when no block
    of it has a band. A fingerprint with two live blocks of the same index (the same prompt running twice) is left
    out."""
    live = np.nonzero(t.kind == LIVE)[0]
    if live.size == 0:
        return {}
    fps, inv = np.unique(t.fp[live], return_inverse=True)
    s = np.zeros(fps.size, dtype=np.int64)
    np.maximum.at(s, inv, t.settled[live].astype(np.int64))
    c = np.zeros(fps.size, dtype=np.int64)
    lo, hi = t.lo[live].astype(np.int64), t.hi[live].astype(np.int64)
    np.maximum.at(c, inv, np.where(hi > lo, t.idx[live].astype(np.int64) * t.bs + hi, 0))
    first = np.zeros(fps.size, dtype=np.int64)
    np.add.at(first, inv, (t.idx[live] == 0).astype(np.int64))
    return {int(f): (int(x), int(max(x, y))) for f, x, y, n in zip(fps, s, c, first) if n <= 1}


def history(tabs, max_gap: int = 16):
    """What a sequence of committed tables says taken together. tabs: [(version, KVTable)], distinct versions in
    increasing order (the tables whose digests came inside timed answers). A version is one schedule() call.

      linger   a position inside the band of the same live owner (fp, block index) in two tables: the largest version
               gap at which that happens, and how many positions at each gap. A token is in flight for a few calls
               (the settle lag plus the steps running), so a large gap is a band that is not following real work.
      pace     how far a request's computed edge is beyond what one token per call allows. Past its prompt (fp carries
               the prompt length) a request gains one token per schedule() call, so in table B
                   computed_B <= max over earlier tables E that show it of (max(prompt, computed_E) + vB - vE)
               (prompt + vB - v of the previous table if none shows it). A request that was preempted recomputes up
               to where it was, which the earlier sighting covers. Version numbers that advance slower than the work
               (which would let a band linger in version terms) break this.
    Tight tables only (the cautious rule has no upper band edge)."""
    out = {"tables": len(tabs), "linger_gap_max": 0, "linger_positions_by_gap": {}, "pace_excess_max": None,
           "pace_pairs": 0}
    if not tabs or not tabs[0][1].tight:
        return out
    bs = tabs[0][1].bs
    by_gap, any_gap = {}, {}
    pace_max, pairs = None, 0
    last, restarts = {}, 0
    best = {}                                    # fp -> max over earlier tables of (max(prompt, computed) - version)
    for j in range(len(tabs)):
        vB, B = tabs[j]
        cB = request_counts(B)
        if j > 0:
            liveB = B.kind == LIVE
            loB, hiB = B.lo.astype(np.int64), B.hi.astype(np.int64)
            i = j - 1
            # the previous committed table is always compared, whatever the version gap: version numbers that jump
            # ahead (so that no pair falls inside max_gap) do not take a band that stayed put out of sight
            while i >= 0 and (i == j - 1 or vB - tabs[i][0] <= max_gap):
                vA, A = tabs[i]
                same = liveB & (A.kind == LIVE) & (A.fp == B.fp) & (A.idx == B.idx)
                ov = np.minimum(hiB, A.hi.astype(np.int64)) - np.maximum(loB, A.lo.astype(np.int64))
                g = int(vB - vA)
                n = int(ov[same & (ov > 0)].sum())
                if n:
                    by_gap[g] = by_gap.get(g, 0) + n
                n = int(ov[ov > 0].sum())             # the same physical positions, whoever is named as owner
                if n:
                    any_gap[g] = any_gap.get(g, 0) + n
                i -= 1
            v_prev = tabs[j - 1][0]
            for fp, (_, comp) in cB.items():
                if comp < last.get(fp, 0):
                    restarts += 1                 # computed edge went down: the request was preempted and restarted
                plen = fp >> 32
                bound = max(plen - v_prev, best.get(fp, -(1 << 62))) + vB
                pairs += 1
                x = comp - bound
                pace_max = x if pace_max is None else max(pace_max, x)
        for fp, (_, comp) in cB.items():
            last[fp] = comp
            k = max(fp >> 32, comp) - vB
            if k > best.get(fp, -(1 << 62)):
                best[fp] = k
    out.update({"linger_gap_max": max(by_gap) if by_gap else 0,
                "linger_positions_by_gap": {str(k): by_gap[k] for k in sorted(by_gap)},
                "linger_any_owner_gap_max": max(any_gap) if any_gap else 0,
                "linger_any_owner_by_gap": {str(k): any_gap[k] for k in sorted(any_gap)},
                "pace_excess_max": pace_max, "pace_pairs": pairs, "restarts": restarts, "block_tokens": bs})
    return out


def owners_invalid(t: KVTable, tokens_of) -> int:
    """Live blocks whose owner is not a request of the workload that could own them. tokens_of: {fp: token count of
    the request (prompt + returned output)}. A live block must name a known fingerprint, a block index inside that
    request, and a band that ends inside it. The pace and lingering rules read the work from live blocks, so a
    live block that is not tied to a real request would make them vacuous (and its band is never read)."""
    live = np.nonzero(t.kind == LIVE)[0]
    if live.size == 0:
        return 0
    lim = np.array([int(tokens_of.get(int(f), -1)) for f in t.fp[live]], dtype=np.int64)
    idx = t.idx[live].astype(np.int64)
    hi = (t.hi[live] if t.tight else np.full(live.size, t.bs)).astype(np.int64)
    lo = t.lo[live].astype(np.int64)
    edge = np.where(hi > lo, idx * t.bs + hi, idx * t.bs)       # the last position the band claims, if any
    return int(((lim < 0) | (idx * t.bs >= np.maximum(lim, 0)) | (edge > lim)).sum())


def writers_out_of_range(t: KVTable, tokens_of) -> int:
    """Tight tables: positions whose named writer cannot have written them. tokens_of: {fp: token count of that
    request (prompt + returned output)}; a writer with an unknown fingerprint counts for all its positions."""
    if not t.tight:
        return 0
    w = np.array(t.writers, dtype=np.uint64).reshape(-1, 2)
    lim = np.array([int(tokens_of.get(int(f), -1)) for f in w[:, 0]], dtype=np.int64)
    lim[0] = 1 << 62                                     # id 0: the filler
    W = t.writer.astype(np.int64)
    pos = w[:, 1].astype(np.int64)[W] * t.bs + np.arange(t.bs, dtype=np.int64)[None, :]
    inband = (np.arange(t.bs)[None, :] >= t.lo[:, None]) & (np.arange(t.bs)[None, :] < t.hi[:, None])
    return int(((W > 0) & ~inband & (pos >= lim[W])).sum())


def promises(tabs, k: int):
    """A band is a promise. For every committed table E and every position in the band of a live block (owner fp,
    block index), in the first committed table B with vB - vE >= k that position must be outside the band and its
    named writer must be that same (fp, block index): the token that was in flight has been written and is read.
    k = settle lag + 2 (an honest position is in the band for exactly that many versions). There is no window:
    version numbers that jump ahead only bring B closer. Version numbers that stand still never reach B; that is
    what the pace rule is for. Returns positions checked, broken (and why), and the tables near the end of the
    session that have no B. Tight tables only."""
    out = {"k": int(k), "positions": 0, "broken": 0, "still_in_band": 0, "other_writer": 0, "tables_without_later": 0}
    if not tabs or not tabs[0][1].tight:
        return out
    vs = np.array([v for v, _ in tabs], dtype=np.int64)
    bs = tabs[0][1].bs
    pos = np.arange(bs, dtype=np.int64)[None, :]
    wcache = {}
    for e, (vE, E) in enumerate(tabs):
        blk = np.nonzero((E.kind == LIVE) & (E.hi > E.lo))[0]
        if blk.size == 0:
            continue
        j = int(np.searchsorted(vs, vE + k))
        if j >= len(tabs):
            out["tables_without_later"] += 1
            continue
        B = tabs[j][1]
        if j not in wcache:
            w = np.array(B.writers, dtype=np.uint64).reshape(-1, 2)
            wcache = {j: (w[:, 0], w[:, 1].astype(np.int64))}
        wfp, widx = wcache[j]
        band = (pos >= E.lo[blk].astype(np.int64)[:, None]) & (pos < E.hi[blk].astype(np.int64)[:, None])
        inB = (pos >= B.lo[blk].astype(np.int64)[:, None]) & (pos < B.hi[blk].astype(np.int64)[:, None])
        W = B.writer[blk].astype(np.int64)
        mine = (wfp[W] == E.fp[blk][:, None]) & (widx[W] == E.idx[blk].astype(np.int64)[:, None]) & (W > 0)
        out["positions"] += int(band.sum())
        out["still_in_band"] += int((band & inB).sum())
        out["other_writer"] += int((band & ~inB & ~mine).sum())
    out["broken"] = out["still_in_band"] + out["other_writer"]
    return out


def session_verdict(tabs, tokens_of, mult=None, lag: int = 2, step_budget: int = 1024, linger_max: int = -1,
                    pace_max: int = 1, promise_frac_max: float = 0.002, mask_mean_max_tokens: float = 0.0):
    """Everything the verifier concludes from the committed tables of a session (tight mode); "ok" is the pass
    condition. tabs: [(version, KVTable)] in increasing version order; tokens_of: {fp: token count} and mult:
    {fp: times the workload holds that prompt} from the verifier's own requests; lag and step_budget from the
    verifier's own configuration.

      structure      band edges, writer ids, bands only on live blocks, one live block per request block
      unknown        no block of an unnamed owner (such a block is fully masked and nothing ties it to work)
      owners         every live block names a request of the workload, a block index inside it, a band inside it
      writers        every read position's named writer is a request long enough to have written it
      bound          no table masks more than (lag + 2) x step_budget tokens
      promises       every band position is, lag + 2 versions later, read and written by the owner named; at most
                     promise_frac_max of them may be broken (honest runs: 0 on ShareGPT, 0.013-0.028% on the
                     preemption-heavy random workload, where a freed block is reused by a new prefill at once)
      pace           no request is ahead of one token per version (slow version numbers)
      linger         reported only (promises replace it: a request that is preempted and resumes on the same blocks
                     puts the same positions in the band again, which is honest)
      mask mean      optional: session mean of masked tokens at most mask_mean_max_tokens (0 = not counted)"""
    k = lag + 2
    if linger_max < 0:
        linger_max = k
    bound = k * int(step_budget)
    masked = [t.masked_tokens() for _, t in tabs]
    out = {"tables": len(tabs), "lag": int(lag), "step_budget": int(step_budget), "band_bound_tokens": bound,
           "structure_bad": sum(1 for _, t in tabs if check(t, mult)),
           "unknown_owner_blocks_max": max((int((t.kind == UNKNOWN).sum()) for _, t in tabs), default=0),
           "owners_invalid_max": max((owners_invalid(t, tokens_of) for _, t in tabs), default=0),
           "writers_out_of_range_max": max((writers_out_of_range(t, tokens_of) for _, t in tabs), default=0),
           "tables_over_bound": sum(1 for m in masked if m > bound),
           "masked_tokens_mean": round(sum(masked) / max(1, len(masked)), 1), "masked_tokens_max": max(masked, default=0),
           "linger_gap_limit": int(linger_max), "pace_excess_limit": int(pace_max), "promise_broken_frac_limit": float(promise_frac_max),
           "mask_mean_max_tokens": float(mask_mean_max_tokens)}
    h = history(tabs)
    out.update({x: h[x] for x in ("linger_gap_max", "linger_positions_by_gap", "pace_excess_max", "pace_pairs", "restarts")
                if x in h})
    pr = promises(tabs, k)
    out["promises"] = pr
    out["ok"] = bool(out["structure_bad"] == 0 and out["unknown_owner_blocks_max"] == 0 and out["owners_invalid_max"] == 0
                     and out["writers_out_of_range_max"] == 0 and out["tables_over_bound"] == 0
                     and pr["broken"] <= promise_frac_max * pr["positions"] and (h.get("pace_excess_max") or 0) <= pace_max
                     and (mask_mean_max_tokens <= 0 or out["masked_tokens_mean"] <= mask_mean_max_tokens))
    return out
