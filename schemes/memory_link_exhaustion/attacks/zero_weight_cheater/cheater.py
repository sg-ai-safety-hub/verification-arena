"""The strongest cheater we can build against the verifier (no inference at all).

Threat model: the cheater only has to pass the verifier. The declared
model is all zeros (--zero-weights on the verifier), so W pages are encode(0)
(keystream + filler in alignment gaps), live KV pages are encode(0), and every
output token is 0. The cheater

  * answers 'generate' at a configurable token rate (default: just above the
    calibrated honest rate), returning zeros;
  * keeps every W, KV and filler page resident (the honest arena layout, so its chase is the honest
    kernel; dropping pages and regenerating them per probe was removed: under butterfly one probe into a
    dropped page costs a whole chunk network, ~28 ms per answer at 1% of pages);
  * runs the real PCIe noise stream;
  * runs an unauthorised workload in the freed HBM.

    python -m schemes.memory_link_exhaustion.attacks.zero_weight_cheater.cheater   (started by the verifier with --prover cheater)
Spec (JSON in VARENA_CHEAT; the named entries are in scripts/reproduce/cheaters.json): {"keep_w": true, "keep_k": true,
    "k_filler": bool, "real_w": bool, "rate_mult": 1.02, "workload": null | "max_shard" | "local_rl",
    "wl_spec": {...}, "dev_gate": bool, "native_head": bool}
"""

from __future__ import annotations

import json
import os
import queue
import socket
import sys
import threading
import time
import traceback

import numpy as np
import torch
# SIGUSR1 -> all-thread Python stack dump to stderr (shard stall diagnosis)
import faulthandler as _fh, signal as _sig
_fh.register(_sig.SIGUSR1, all_threads=True)

from schemes.memory_link_exhaustion.gpu import kernels as K
from schemes.memory_link_exhaustion.encodings import kvbfly
from schemes.memory_link_exhaustion.encodings import ref
from schemes.memory_link_exhaustion.core.proto import Conn
from schemes.memory_link_exhaustion.inference.prover import PAGE, kv_key, w_layout


def log(*a):
    print(time.strftime("%H:%M:%S"), "[cheater]", *a, file=sys.stderr, flush=True)


_REH_N = int(os.environ.get("VARENA_REHEARSAL_STEPS", "0"))   # memory rehearsal (see Cheater._rehearsal_tick)

class Cheater:
    def __init__(self, spec):
        self.spec = spec
        self.busy = torch.zeros(1, dtype=torch.int32).pin_memory()   # set while a challenge is answered
        self.busy_np = self.busy.numpy()
        self.chal_q = queue.Queue()
        self.gen_q = queue.Queue()
        self.lock = threading.Lock()
        self.win_buf = []
        self.stats = {"challenges": 0, "chal_ms": []}
        self.wl = None
        self.wl_stats = None

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(os.environ["VARENA_SOCK"])
        self.conn = Conn(s)
        self.conn.send({"op": "hello", "pid": os.getpid()})
        m = self.conn.recv()
        self.cfg = m["cfg"]
        self.master, self.fkey = int(self.cfg["master"]), int(self.cfg["fkey"])

    @torch.no_grad()
    def setup(self, model, num_blocks):
        from schemes.memory_link_exhaustion.core.model import Cfg
        sp = self.spec
        torch.cuda.set_device(0)
        self.mcfg = c = Cfg.from_hf(model)
        ents, w_bytes = w_layout(c, self.master, plain_all=bool(self.cfg.get("plain_w")))
        nW = w_bytes // PAGE
        per_tok = 2 * c.Hkv * c.D * 2
        bs = PAGE // per_tok
        nb = num_blocks
        nK = c.L * nb
        self.act_bytes = 0
        free, cap = torch.cuda.mem_get_info()
        allowance = int(self.cfg["allowance_bytes"])
        # the verifier judges against the declared GPU model's capacity, not the driver's report (H200 NVL: ~10 MB
        # larger); size against the larger so the capacity check passes exactly (idle controls once failed by 10 MB)
        cap = max(cap, int(self.cfg.get("capacity_bytes", 0)))
        from schemes.memory_link_exhaustion.verifier.act_pool import cheater_declared_min
        declared_min = cheater_declared_min(cap, allowance, self.act_bytes)
        nR = (int(self.cfg.get("ring_mib", 0)) << 20) // PAGE if self.cfg.get("pcie") else 0
        npages = max(nW + nK + nR, (declared_min + PAGE - 1) // PAGE)
        nF = npages - nW - nK - nR
        R0 = nW + nK + nF
        if not (sp.get("keep_w") and sp.get("keep_k")):
            raise RuntimeError("cheater spec: keep_w and keep_k are required (every page resident)")
        kind = np.zeros(npages, dtype=np.int8)
        a0 = np.zeros(npages, dtype=np.int64)
        keep_w = list(range(nW))
        f_res = np.arange(nW + nK, R0)
        # ---- resident memory: kept W pages + resident F pages (one pool, page by page)
        keep = np.concatenate([np.asarray(keep_w, dtype=np.int64), f_res]).astype(np.int64)
        self.res = torch.empty(max(1, len(keep)) * PAGE, dtype=torch.uint8, device="cuda")
        base = self.res.view(torch.int32)
        kind[keep] = K.CK_RES
        a0[keep] = np.arange(len(keep), dtype=np.int64) * (PAGE // 4)
        # the (zero-model) KV pages resident: content = KV keystream
        self.kres = torch.zeros(nK * PAGE, dtype=torch.uint8, device="cuda")
        for l in range(c.L):
            v = self.kres[l * nb * PAGE:(l + 1) * nb * PAGE]
            if kvbfly.KVENC == "bfly":
                kvbfly.xor_pages(v.view(torch.int32), kv_key(self.master, l))
            elif kvbfly.KVENC == "sloth":
                from schemes.memory_link_exhaustion.encodings import sloth64
                sloth64.encode_to(v.view(torch.int64), v.view(torch.int64), sloth64.kv_seed(kv_key(self.master, l)))
            else:
                K.xor16(v.view(torch.int16), v.view(torch.int16), kv_key(self.master, l))
        if sp.get("k_filler"):
            # non-serving cheater, truthful claims: the KV region is left as it is after seal (every page free,
            # holding filler) and every challenged KV page is claimed 'filler'
            K.fill(self.kres, nW, self.fkey)
        sl = np.arange(nW, nW + nK)
        kind[sl] = K.CK_RES
        a0[sl] = (self.kres.data_ptr() - self.res.data_ptr()) // 4 + \
            np.arange(nK, dtype=np.int64) * (PAGE // 4)
        # fill resident pages with their content: W gap pages (filler + zero-weight ks), F filler
        self.kind = torch.from_numpy(kind).cuda()
        self.a0 = torch.from_numpy(a0).cuda()
        pm = torch.from_numpy(a0).cuda()          # word offsets for resident pages
        if len(keep):
            kt = torch.from_numpy(keep).cuda()
            K.fill_pages(base, pm, kt, self.fkey)
        self._overlay_w_gaps(keep_w, ents, base, a0)
        self.w_ents, self.w_bytes = ents, w_bytes
        if sp.get("real_w"):
            # real-weights threat model (verifier without --zero-weights): every W page kept, holding the real
            # declared matrices encoded exactly as the honest prover stores them (no vLLM is run)
            assert len(keep_w) == nW and all(int(a0[p]) == p * (PAGE // 4) for p in (0, nW - 1))
            self._load_real_w(model, ents)
        torch.cuda.synchronize()
        self.npages, self.nW, self.nK, self.nb, self.bs = npages, nW, nK, nb, bs
        self.R0, self.nR = R0, nR
        self.base = base
        cfg = self.cfg
        self.n_ring = max(1, nR * PAGE // cfg["window_bytes"]) if nR else 0
        # slot -> offset of the window now in that ring slot (pinned, read by the GPU)
        self.slot_off = torch.zeros(max(1, self.n_ring), dtype=torch.int64).pin_memory()
        # PCIe noise stream into a ring that is part of the arena
        self.streamer = None
        if cfg.get("pcie"):
            from schemes.memory_link_exhaustion.link.pcie import Streamer
            self.streamer = Streamer(cfg["pool_path"], cfg["pcie_key"],
                                     window_bytes=cfg["window_bytes"], lag=cfg["lag"],
                                     on_window=self._on_window,
                                     ring_bytes=nR * PAGE,
                                     sched=cfg.get("noise_sched"))
        if nR:
            rp = np.arange(R0, npages)
            ring = self.streamer.ring
            kind[rp] = K.CK_RES
            a0[rp] = (ring.data_ptr() - self.res.data_ptr()) // 4 + \
                np.arange(nR, dtype=np.int64) * (PAGE // 4)
            self.kind = torch.from_numpy(kind).cuda()
            self.a0[torch.from_numpy(rp).cuda()] = torch.from_numpy(a0[rp]).cuda()
        self.chal_stream = torch.cuda.Stream(priority=-1)
        self.last_arr = torch.full((1,), -1, dtype=torch.int64).pin_memory()
        self.last_np = self.last_arr.numpy()
        if cfg.get("native_sock"):
            self._start_native()
        torch.cuda.synchronize()
        free_after, cap = torch.cuda.mem_get_info()
        from schemes.memory_link_exhaustion.encodings import actenc2 as _AE
        self.info = {"act_pool": {"granules": _AE.pool_granules(), "bytes": self.act_bytes, "master": _AE.MASTER,
                                  "epoch": 1} if self.act_bytes else None,
                     "npages": npages, "nW": nW, "nK": nK, "L": c.L, "nb": nb,
                     "R0": R0, "nR": nR, "n_ring": self.n_ring,
                     "nF": nF, "n_evicted": 0, "kv_shape": [nb, c.Hkv, bs, 2 * c.D],
                     "declared_bytes": npages * PAGE + self.act_bytes, "capacity": cap, "free_after": free_after,
                     "outside_bytes": cap - npages * PAGE - self.act_bytes, "w_bytes": w_bytes,
                     "gpu": torch.cuda.get_device_name(0), "w_entries": ents,
                     "cheater": {"w_kept_pages": len(keep_w), "f_resident": int(len(f_res)),
                                 "resident_bytes": int(self.res.numel()),
                                 "hbm_free_bytes": int(free_after), "spec": sp}}
        log("ready", json.dumps(self.info["cheater"]))

    def _load_real_w(self, model, ents):
        from schemes.memory_link_exhaustion.core.model import SafeTensors
        st = SafeTensors(model)
        t0 = time.perf_counter()
        for name, off, shape, src, key in ents:
            t = st.concat(src).to(device="cuda", dtype=torch.bfloat16).contiguous()
            assert list(t.shape) == list(shape), (name, t.shape, shape)
            if key >= 0 and ref.WENC in ("bfly", "sloth"):
                ref.w_encode(t, t, key, ref.w_ordinal(off))
            elif key >= 0:
                K.xor16_w(t.view(torch.int16), t.view(torch.int16), key, shape[1])
            n = t.numel() * 2
            self.res[off: off + n].copy_(t.view(torch.uint8).view(-1))
            del t
        torch.cuda.synchronize()
        log(f"real W loaded and encoded ({ref.WENC}) in {time.perf_counter() - t0:.0f} s")

    def _overlay_w_gaps(self, keep_w, ents, base, a0):
        """Kept W pages: filler with the zero-weight keystream over the tensor parts."""
        if not keep_w:
            return
        mem = base.view(torch.int16)
        keep_set = set(keep_w)
        for name, off, shape, src, key in ents:
            n = int(np.prod(shape)) * 2
            if ref.WENC in ("bfly", "sloth") and key >= 0:
                # zero-weight model: the stored matrix is encode(0) (made on the GPU)
                t = torch.zeros(tuple(shape), dtype=torch.bfloat16, device="cuda")
                ref.w_encode(t, t, key, ref.w_ordinal(off))
                tf = t.view(torch.int16).reshape(-1)
                for p in range(off // PAGE, (off + n + PAGE - 1) // PAGE):
                    if p not in keep_set:
                        continue
                    lo, hi = max(off, p * PAGE), min(off + n, (p + 1) * PAGE)
                    j0, cnt = (lo - off) // 2, (hi - lo) // 2
                    dst = int(a0[p]) * 2 + (lo - p * PAGE) // 2
                    mem[dst: dst + cnt].copy_(tf[j0: j0 + cnt])
                del t, tf
                continue
            for p in keep_w:
                lo, hi = max(off, p * PAGE), min(off + n, (p + 1) * PAGE)
                if lo >= hi:
                    continue
                j0 = (lo - off) // 2
                cnt = (hi - lo) // 2
                ks = np.zeros(cnt, dtype=np.uint16) if key < 0 else \
                    ref.encode_w(np.zeros(cnt, dtype=np.uint16), key, j0, int(shape[1]))
                dst = int(a0[p]) * 2 + (lo - p * PAGE) // 2
                mem[dst: dst + cnt].copy_(torch.from_numpy(ks.view(np.int16)))

    # ------------------------------------------------------------------ threads
    def _on_window(self, i, off, d0, d1, t):
        if self.n_ring:
            self.slot_off[i % self.n_ring] = off
        self.last = (i, d0)
        self.last_np[0] = (i << 32) | d0
        with self.lock:
            self.win_buf.append([i, off, d0, d1])

    last = (-1, 0)

    def _flusher(self):
        while True:
            time.sleep(0.005)
            with self.lock:
                buf, self.win_buf = self.win_buf, []
            if buf:
                self.conn.send({"op": "pcie", "w": buf})

    def _reader(self):
        while True:
            m = self.conn.recv()
            if m is None:
                os._exit(0)
            op = m["op"]
            if op == "stop":
                self.chal_q.put(m)
            elif op == "generate":
                self.gen_q.put(m)

    def _generator(self):
        """Fake inference: zeros at rate_mult x the calibrated rate."""
        rate = float(self.spec.get("rate_tok_s") or 1e9)
        t_next = time.perf_counter()
        while True:
            m = self.gen_q.get()
            n = len(m["prompts"]) * m["n_new"]
            t_next = max(t_next, time.perf_counter()) + n / rate
            time.sleep(max(0.0, t_next - time.perf_counter()))
            self.conn.send({"op": "outputs", "id": m["id"],
                            "tokens": [[0] * m["n_new"] for _ in m["prompts"]]})

    def _challenger(self):
        torch.cuda.set_device(0)
        while True:
            m = self.chal_q.get()
            if m["op"] == "stop":       # (challenges arrive on the native socket; this queue only carries the stop)
                self._bye()

    def _start_native(self):
        from schemes.memory_link_exhaustion.gpu import native
        cfg = self.cfg
        if self.spec.get("dev_gate"):
            # attacks/gated_shard: every challenge graph raises the device gate first and drops it last,
            # so the hidden job's per-op gate waits park its stream while a challenge runs
            from schemes.memory_link_exhaustion.attacks.gated_shard.gate import install_gated_graphs
            install_gated_graphs(native)
            log("challenge graphs gated (dev_gate)")

        gs = []            # the challenge graphs, by index: the mixed chase is the only one
        if cfg.get("mchase_P"):
            # mixed chase over W, KV, filler and the pool: needs every region resident (keep_w, keep_k) and each
            # granule contiguous, so the cheater answers with the honest kernel
            from schemes.memory_link_exhaustion.gpu.cuda import build as CB
            from schemes.memory_link_exhaustion.encodings import actenc2 as AE
            ext = CB.load()
            glm = AE.pool_granules() if self.act_bytes else []
            # --ring-chase-guard: noise-ring granules in the chase (head-relative); this cheater answers them only
            # from a resident, contiguous ring in the arena (the honest layout)
            rtag = bool(cfg.get("ring_tag")) and int(cfg.get("kv_table", 0) or 0) == 3
            nrg = ref.ring_chase_granules(self.n_ring, int(cfg.get("ring_chase_guard", 0)), tagged=rtag) if self.n_ring else 0
            gm = ref.ch_granules_mix(self.nW, self.nK, self.R0 - self.nW - self.nK, len(glm), nrg)
            kind_h, a0h = self.kind.cpu().numpy(), self.a0.cpu().numpy()
            if nrg:
                rp = np.arange(self.R0, self.R0 + self.nR)
                if cfg["window_bytes"] != 1 << 25 or not (np.all(kind_h[rp] == K.CK_RES) and np.all(
                        a0h[rp] == a0h[self.R0] + np.arange(self.nR) * (PAGE // 4))):
                    raise RuntimeError("ring chase: this cheater has no resident contiguous 32 MiB-window ring to probe")
            pg = np.asarray([x for x in gm if x < ref.CH_PA0], dtype=np.int64)
            pages = pg[:, None] + np.arange(ref.CH_GRAN_PAGES, dtype=np.int64)[None, :]
            m_ok = bool(np.all(kind_h[pages] == K.CK_RES)) and \
                bool(np.all(a0h[pages] == a0h[pg][:, None] + np.arange(ref.CH_GRAN_PAGES)[None, :] * (PAGE // 4)))
            if not m_ok:
                raise RuntimeError("mixed chase: every challenged granule must be resident and contiguous (keep_w, keep_k)")
            self._mchase_ga = torch.tensor([(x - ref.CH_RA0) // ref.CH_GRAN_PAGES - ref.CH_RING_TAG if x >= ref.CH_RA0
                                            else glm[(x - ref.CH_PA0) // ref.CH_GRAN_PAGES] if x >= ref.CH_PA0 else
                                            self.base.data_ptr() + int(a0h[x]) * 4 for x in gm],
                                           dtype=torch.int64, device="cuda")
            rb_ = self.base.data_ptr() + int(a0h[self.R0]) * 4 if nrg else 0
            # --kv-tab-test: the settled-KV kernel path with a synthetic block table (kernel validation only)
            ktest = int(cfg.get("kv_tab_test", 0) or 0)
            if ktest:
                self._mchase_gkv = torch.tensor([x - self.nW if self.nW <= x < self.nW + self.nK else -1 for x in gm],
                                                dtype=torch.int32, device="cuda")
                self._mchase_kvtab = torch.from_numpy(ref.kv_tab_test(ktest, self.nb).view(np.int16).copy()).cuda()
                log(f"cheater mixed chase: settled-KV kernel, synthetic table seed {ktest} over {self.nb} blocks")
            # --kv-table with a non-serving cheater (k_filler: the KV region is never written after sealing): the
            # truthful block table is the sealed one (every block filler, nothing in flight). The chase runs through
            # the table kernel and returns that table's (version, digest), so the verifier replays the cheater's
            # chains from sources exactly as it does an honest prover's.
            # (--kv-table 3 with a zero-weight cheater: the same kernel path, so its answers carry the ring tags; its
            # KV region holds the zero model's encoded KV, which the verifier checks as such)
            ktab_real = int(cfg.get("kv_table", 0) or 0) if ((self.spec.get("k_filler") or int(cfg.get("kv_table", 0) or 0) == 3)
                                                              and not ktest) else 0
            if ktab_real:
                from schemes.memory_link_exhaustion.inference import kvtable as KT
                # the sealed table: every block filler, nothing in flight, version 0. The verifier builds the same one
                # itself (verifier.prover_tables) and the chase returns this (version, digest); it is not sent to the verifier
                kt = KT.KVTable(self.nb, 16, tight=(ktab_real >= 2))
                dg = kt.digest()
                self._mchase_gkv = torch.tensor([x - self.nW if self.nW <= x < self.nW + self.nK else -1 for x in gm],
                                                dtype=torch.int32, device="cuda")
                n4 = -(-self.nb // 4) * 4
                self._mchase_kvtab = torch.from_numpy(np.full(n4, 16 | (16 << 8), dtype=np.uint16).view(np.int16).copy()).cuda()
                self._mchase_kvinfo = torch.tensor([int(kt.version), dg - (1 << 64) if dg >= (1 << 63) else dg],
                                                   dtype=torch.int64, device="cuda")
                log(f"cheater mixed chase: sealed block table (all filler) over {self.nb} blocks, digest {dg:#x}")
            tbm = len(gm) // ref.CH_GRAN_PER_BLOCK
            nbm = min(int(cfg.get("chase_blocks", 11)), tbm)
            Km, Pm = int(cfg.get("chase_K", 132 * 32)), int(cfg["mchase_P"])
            log(f"cheater mixed chase: honest kernel, {tbm} blocks")

            def mchase(o, nn):
                if ktest:
                    ext.chase_honest_full(self._mchase_ga, tbm, nbm, nn.data_ptr(), Km, Pm, o, rb_,
                                          self.n_ring if nrg else 1, self._mchase_gkv, self._mchase_kvtab, self.nb, 0, 0)
                elif ktab_real:
                    ext.chase_honest_full(self._mchase_ga, tbm, nbm, nn.data_ptr(), Km, Pm, o, rb_,
                                          self.n_ring if nrg else 1, self._mchase_gkv, self._mchase_kvtab, self.nb,
                                          self._mchase_kvinfo.data_ptr(),
                                          (0x100 | ((int(cfg.get("ring_chase_guard", 0)) << 16) if (rtag and nrg) else 0))
                                          if ktab_real == 3 else 0)
                else:
                    assert not nrg, "ring chase needs --kv-table 3 (tagged ring reads); the untagged path was removed"
                    ext.chase_honest(self._mchase_ga, tbm, nbm, nn.data_ptr(), Km, Pm, o)
            # (--kv-table 3: the tag slots follow; the sealed table has no band, so every chain reports 0 tagged reads)
            gs.append(native.ChaseGraph(mchase, Km, self.chal_stream,
                                        extra=(2 + (Km * ref.CH_TAG_SLOTS // 2 if ktab_real == 3 else 0)) if ktab_real else 0))
        self.native_graphs = gs
        # spec native_head: the challenge server reads the head the native noise streamer writes (as the honest
        # prover does), not the copy this cheater's Python window callback keeps. The callback runs under the
        # interpreter lock, so the hidden job's Python work left that copy stale by ~100 ms at a time: a low head in
        # one answer and a catch-up in the next (head "jumps", ring samples refused as newer than the head). Those
        # were artefacts of this implementation, not costs of the link traffic.
        head_arr = self.streamer.last_arr if (self.spec.get("native_head") and self.streamer is not None
                                              and getattr(self.streamer, "last_arr", None) is not None) else self.last_arr
        native.start_server(cfg["native_sock"], gs, head_arr, busy=self.busy)
        log("native challenge server on", cfg["native_sock"])

    def _bye(self):
        st = dict(self.stats)
        ms = sorted(st.pop("chal_ms"))
        st["chal_ms_p50"] = ms[len(ms) // 2] if ms else None
        st["chal_ms_max"] = ms[-1] if ms else None
        st["pcie_rate_self"] = self.streamer.rate(5.0) if self.streamer else None
        st["cheat"] = self.wl_stats or (self._wl_finish() if self.wl else None)
        st["max_reserved"] = torch.cuda.max_memory_reserved()
        self.conn.send({"op": "bye", "stats": st})

    # ------------------------------------------------------------------ workload
    def _build_red(self, kind, model):
        """Hidden workloads (attacks/max_shard, attacks/local_rl), constructed in full: GPU tensors made
        in place in bf16, pinned host buffers, helper threads. Called from run() before ready, so the idle control
        does nothing at all once the verifier starts timing."""
        t0 = time.perf_counter()
        if kind == "max_shard":
            from schemes.memory_link_exhaustion.attacks.max_shard.workload import MaxShard
            ws = dict(self.spec.get("wl_spec", {}))
            ws.setdefault("t0_wall", time.time())
            if ws.get("ffns", "auto") == "auto":
                # size the stage against this process's real free memory after sealing
                from schemes.memory_link_exhaustion.attacks.max_shard.workload import build_fitted
                wl = build_fitted(ws, float(ws.pop("margin_mib", 160)), log=log)
            else:
                wl = MaxShard(ws)
        else:
            from types import SimpleNamespace
            from schemes.memory_link_exhaustion.attacks.hidden_lora_rl import hidden_rl as H
            from schemes.memory_link_exhaustion.attacks.gated_shard import gate as G
            from schemes.memory_link_exhaustion.attacks.local_rl.workload import LocalRL
            ws = dict(self.spec.get("wl_spec", {}))
            H.BflyLin.SCRATCH_ROWS = int(ws.pop("scratch_rows", 4096))
            H.BflyLin.SCRATCH_BYTES = int(float(ws.pop("scratch_mib", 16)) * 2**20)
            if self.spec.get("dev_gate"):
                H.GATE = G.wait
            H.SLICE_GATE = bool(ws.pop("slice_gate", False))     # local_rl: gate per linear call (default)
            m = H.build_model(SimpleNamespace(w_entries=self.w_ents, wbuf=self.res[:self.w_bytes]), model)
            if ws.pop("fit", False) or ws.get("gen_in_flight") == "auto" or ws.get("mlp_chunk") == "auto":
                # size gen_in_flight and the token-chunked MLP against real free memory, reserving
                # for the challenge kernels; seg 1 + chunked MLP is what makes the 512-token update fit the cut
                from schemes.memory_link_exhaustion.attacks.local_rl.workload import fit_rl
                ws.pop("gen_in_flight", None); ws.pop("mlp_chunk", None)
                wl = fit_rl(m, ws, float(ws.pop("margin_mib", 160)), log=log)
            else:
                wl = LocalRL(m, ws)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        log(f"{kind} built in {time.perf_counter() - t0:.1f} s, {free / 2**30:.2f} GiB free after: {json.dumps(wl.describe())}")
        return wl

    def _workload(self, kind, model):
        torch.cuda.set_device(0)
        d = float(self.spec.get("wl_delay", 0))
        if d > 0:
            # start the hidden job only after the warm-up (warm-up challenges see an idle cheater)
            time.sleep(d)
        s = getattr(self, "_wl_stream", None) or torch.cuda.Stream()
        try:
            with torch.cuda.stream(s):
                free, _ = torch.cuda.mem_get_info()
                budget = free - (512 << 20)          # what the strategy freed
                log(f"workload {kind}: budget {budget / 2**30:.2f} GiB")
                # workloads: max_shard, local_rl, pretrain_qwen (the launch / graph storm and link-theft probes)
                if kind in ("max_shard", "local_rl"):
                    # built before ready (_build_red) so no setup runs once the verifier is timing
                    self.wl = getattr(self, "_prewl", None) or self._build_red(kind, model)
                    if getattr(self.wl, "pace_loop", False):
                        # closed-loop pacing: the hidden job works while this process's own noise average (with its
                        # own estimate of the freeze credit) is above the target, and rests when it is not
                        self.wl.loop_pacer = self._loop_pacer()
                    if kind == "max_shard" and not self.wl.idle and not getattr(self, "_prewl_warm", False):
                        self.wl.step()                    # warm-up micro-batch (no optimiser step)
                    self.wl.t_steps.clear()
                elif kind == "pretrain_qwen":
                    from baselines.pretrain.pretrain_qwen import PretrainQwen, Stage, nparams
                    ws = dict(self.spec.get("wl_spec", {}))
                    if "layers" not in ws:
                        # as many Qwen-7B layers as fit fully resident (bf16 params + fp32 Adam ~16 B/param)
                        # next to activations: the strongest hidden pretraining in the freed memory
                        with torch.device("meta"):
                            p1 = nparams(Stage(1))
                        ws["layers"] = max(1, int((budget - (3 << 30)) // (p1 * 16)))
                        ws.setdefault("mb", 1)
                    log(f"pretrain_qwen hidden: {ws['layers']} layers, mb {ws.get('mb')}")
                    self.wl = PretrainQwen(ws, budget=budget)
                    if getattr(self.wl, "sweep", None) is not None and self.streamer is not None:
                        # adaptive theft sweep: the workload reads the noise this process is delivering
                        self.wl.noise_fn = lambda: (self.streamer.n_done, self.streamer.W)
                    if (self.spec.get("wl_spec") or {}).get("pace_loop"):
                        # closed-loop theft: a chunk is copied only while the noise average is above the target
                        _lp = self._loop_pacer()
                        if _lp is not None:
                            self.wl.yield_fn = _lp.wait
                    for _ in range(self.wl.accum):        # warm-up optimiser step
                        self.wl.step()
                    self.wl.t_steps.clear()
                else:
                    raise ValueError(f"unknown workload {kind!r} (max_shard, local_rl, pretrain_qwen)")
                torch.cuda.synchronize()
                self.wl_t0 = time.perf_counter()
                self.wl_units = 0
                self.wl_run = True
                t_on = 0.0
                while self.wl_run:
                    ts = time.perf_counter()
                    self.wl_units += self.wl.step()
                    _nc = getattr(self.wl, "calls", None)
                    if _nc is not None and _nc != getattr(self, "_wl_calls_sent", 0):
                        self._wl_calls_sent = _nc            # the verifier harness may end the session on this
                        try:
                            self.conn.send({"op": "wl_progress", "calls": int(_nc), "t": time.perf_counter()})
                        except Exception:
                            pass
                    if _REH_N:
                        self._rehearsal_tick()
                    t_on += time.perf_counter() - ts
                self.wl_on_s = t_on
                self.wl_t1 = time.perf_counter()
        except Exception:
            self.wl_err = traceback.format_exc()
            log("workload error", self.wl_err)
            try:        # the workload's own state (fit, oom_log, shrinks) next to the traceback
                log("workload describe at error", json.dumps(self.wl.describe(), default=str)[:4000])
            except Exception as e:
                log(f"workload describe at error failed: {type(e).__name__}: {e}")
            if _REH_N:
                log(f"REHEARSAL FAIL steps {getattr(self, '_reh_k', 0)} {self._rehearsal_mem()}")

    def _rehearsal_mem(self):
        """Device free now, the torch peak and reserve: the spare a real run would have at the same point."""
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            resv, peak = torch.cuda.memory_reserved(), torch.cuda.max_memory_allocated()
            spare = free_b + max(0, resv - peak)
            self._reh_spare = min(getattr(self, "_reh_spare", 1 << 62), spare)
            return (f"free_mib {free_b >> 20} reserved_mib {resv >> 20} peak_alloc_mib {peak >> 20} "
                    f"spare_mib {spare >> 20} min_spare_mib {self._reh_spare >> 20}")
        except Exception as e:
            return f"mem n/a ({type(e).__name__})"

    def _rehearsal_tick(self):
        """VARENA_REHEARSAL_STEPS=N: memory rehearsal on a real seal. Log memory after every
        workload step while challenges and noise run, and PASS after N steps (the driver then stops the run)."""
        self._reh_k = getattr(self, "_reh_k", 0) + 1
        m = self._rehearsal_mem()
        log(f"REHEARSAL step {self._reh_k} units {self.wl_units} {m}")
        if self._reh_k >= _REH_N:
            try:
                d = json.dumps(self.wl.describe(), default=str)[:3000]
            except Exception:
                d = "{}"
            log(f"REHEARSAL PASS steps {self._reh_k} {m} describe {d}")

    def _start_gate_log(self, period):
        """VARENA_GATE_LOG_S: sample the device challenge gate on its own copy stream at ~2 kHz
        (a stuck or long-raised gate shows as raised fraction / longest raised run), plus the hidden job's gate-wait
        count, challenges answered and workload units; one GATE_LOG line per period."""
        from schemes.memory_link_exhaustion.encodings import actenc2 as AE
        from schemes.memory_link_exhaustion.attacks.gated_shard import gate as G

        def _run():
            torch.cuda.set_device(0)
            s = torch.cuda.Stream(priority=-1)
            host = torch.zeros(2, dtype=torch.int32).pin_memory()
            gd = AE.gate_dev()
            n = up = run = longest = stuck_reads = 0
            w0, u0, t_last = G.STATS["waits"], getattr(self, "wl_units", 0), time.perf_counter()
            ch0 = getattr(self, "_n_answered", 0)
            while True:
                with torch.cuda.stream(s):
                    host.copy_(gd, non_blocking=True)
                ev = torch.cuda.Event(); ev.record(s)
                t0 = time.perf_counter()
                while not ev.query():
                    if time.perf_counter() - t0 > 0.05:
                        break
                    time.sleep(0.0001)
                if not ev.query():
                    stuck_reads += 1                 # the copy itself is not completing (copy engine / queue)
                    ev.synchronize()
                g = int(host[0])
                n += 1
                if g:
                    up += 1; run += 1; longest = max(longest, run)
                else:
                    run = 0
                time.sleep(0.0005)
                now = time.perf_counter()
                if now - t_last >= period:
                    w, u = G.STATS["waits"], getattr(self, "wl_units", 0)
                    ch = getattr(self, "_n_answered", 0)
                    log(f"GATE_LOG reads {n} raised_frac {up / max(1, n):.4f} longest_raised_reads {longest} "
                        f"(~{longest * (now - t_last) / max(1, n) * 1e3:.0f} ms) now {g} slow_reads {stuck_reads} "
                        f"wl_gate_waits +{w - w0} wl_units +{u - u0} challenges +{ch - ch0} busy {int(self.busy_np[0])} "
                        f"wait_kernel {json.dumps(G.wait_stats())}")
                    n = up = longest = stuck_reads = 0
                    w0, u0, ch0, t_last = w, u, ch, now
        threading.Thread(target=_run, daemon=True, name="gatelog").start()

    def _loop_pacer(self):
        """Closed-loop pacer (attacks/local_rl/pace.py) against VARENA_PACE_TARGET_GBS, or None."""
        tgt = float(os.environ.get("VARENA_PACE_TARGET_GBS", "0") or 0)
        if tgt <= 0 or self.streamer is None:
            return None
        from schemes.memory_link_exhaustion.attacks.local_rl.pace import Pacer, start_freeze_meter
        self.loop_pacer = Pacer(tgt, lambda: (self.streamer.n_done, self.streamer.W), start_freeze_meter())
        log(f"closed-loop pacing: target {tgt} GB/s")
        return self.loop_pacer

    def _wl_finish(self):
        self.wl_run = False
        if hasattr(self.wl_thread, "join"):
            self.wl_thread.join(timeout=300)
        else:                                       # a Future from the warm-up executor
            import concurrent.futures as _cf
            _cf.wait([self.wl_thread], timeout=300)
        dt = getattr(self, "wl_t1", time.perf_counter()) - getattr(self, "wl_t0", time.perf_counter())
        out = {"spec": self.spec, "error": getattr(self, "wl_err", None), "seconds": dt,
               "units": getattr(self, "wl_units", 0), "unit": getattr(self.wl, "unit", None),
               "rate": getattr(self, "wl_units", 0) / dt if dt > 0 else 0,
               "busy_s": getattr(self, "wl_on_s", None), "duty": self.spec.get("duty", 1.0),
               "peak_alloc": torch.cuda.max_memory_allocated()}
        if hasattr(self.wl, "describe"):
            out["workload"] = self.wl.describe()
        if getattr(self, "loop_pacer", None) is not None:
            out["pace"] = self.loop_pacer.stats()
        self.wl_stats = out
        return out

    def run(self, model):
        wk0 = self.spec.get("workload")
        if wk0 in ("max_shard", "local_rl") and os.environ.get("VARENA_WL_WARM", "0") == "1" \
                and not self.spec.get("wl_spec", {}).get("idle"):
            # build AND run one warm step before ready on the thread and stream the workload will use
            # (its cuBLAS handle, workspaces and kernel modules are created now, so free memory at ready is honest),
            # logging non-torch memory after each first use (the ~470 MiB that killed the 2,048 shard in serving)
            import concurrent.futures as _cf
            self._wl_exec = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="wl")

            def _nt(tag):
                f_, t_ = torch.cuda.mem_get_info()
                r_ = torch.cuda.memory_reserved()
                log(f"WL_WARM {tag}: free={f_ >> 20} nontorch={(t_ - f_ - r_) >> 20} reserved={r_ >> 20} "
                    f"allocated={torch.cuda.memory_allocated() >> 20} MiB")

            def _warm():
                torch.cuda.set_device(0)
                _nt("thread start")
                self._wl_stream = torch.cuda.Stream()
                _nt("stream")
                with torch.cuda.stream(self._wl_stream):
                    a_ = torch.randn(256, 5120, device="cuda", dtype=torch.bfloat16)
                    w_ = torch.randn(1024, 5120, device="cuda", dtype=torch.bfloat16)
                    torch.addmm(torch.zeros(1024, device="cuda", dtype=torch.bfloat16), a_, w_.t())
                    torch.cuda.synchronize()
                    _nt("first bf16 addmm")
                    q_ = torch.randn(1, 5, 128, 128, device="cuda", dtype=torch.bfloat16)
                    k_ = torch.randn(1, 1, 256, 128, device="cuda", dtype=torch.bfloat16)
                    m_ = torch.ones(128, 256, dtype=torch.bool, device="cuda")
                    torch.nn.functional.scaled_dot_product_attention(q_, k_, k_, attn_mask=m_, enable_gqa=True)
                    torch.cuda.synchronize()
                    _nt("first masked SDPA")
                    del a_, w_, q_, k_, m_
                    wl = self._build_red(wk0, model)
                    torch.cuda.synchronize()
                    _nt("built")
                    _snap = os.environ.get("VARENA_MEM_SNAPSHOT")
                    if _snap:                         # allocation history of the warm step
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.memory._record_memory_history(max_entries=200000)
                    def _dump_snap():
                        try:
                            torch.cuda.memory._dump_snapshot(_snap)
                            log(f"WL_WARM snapshot written {_snap}; peak allocated "
                                f"{torch.cuda.max_memory_allocated() >> 20} MiB")
                        except Exception as e:
                            log(f"WL_WARM snapshot failed: {type(e).__name__}: {e}")
                        torch.cuda.memory._record_memory_history(enabled=None)
                    if wk0 != "local_rl":
                        wl.step()                     # one warm micro-batch: forward, backward, flush, refresh
                        torch.cuda.synchronize()
                        if getattr(wl, "flush", None) is not None:
                            wl.flush.drain()
                        _nt("warm step")
                        if _snap:
                            _dump_snap()
                    if wk0 == "local_rl":
                        # RL: warm the update's kernels (update-only step on random tokens), then reset the counters
                        uo = wl.update_only
                        wl.update_only = True
                        try:
                            wl.step()
                            torch.cuda.synchronize()
                        finally:
                            if _snap:                 # the RL warm update's history, also when it fails
                                _dump_snap()
                        wl.update_only = uo
                        if getattr(wl, "flush", None) is not None:      # drop the warm step's gradients
                            wl.flush.drain()
                            for a_ in wl.flush.acc:
                                a_.zero_()
                        for a_ in getattr(wl, "sacc", None) or []:
                            a_.zero_()
                        wl.calls = wl.steps = wl.opt_steps = 0
                        wl.t_steps.clear()
                        for k_ in wl.t_phase:
                            wl.t_phase[k_] = 0.0
                        _nt("warm update step")
                    torch.cuda.empty_cache()
                    _nt("after empty_cache")
                    _ws = self.spec.get("wl_spec", {}) or {}
                    if _ws.get("arena"):
                        # the careful cheater: take everything that is free once, hand it to torch's cache, and
                        # never go back to the device (no cudaMalloc / cudaFree after ready). A small remainder is
                        # left for the allocator's small-block pool and the challenge kernels.
                        f_, _t = torch.cuda.mem_get_info()
                        keep = int(float(_ws.get("arena_margin_mib", 64)) * 2**20)
                        if f_ > keep + (64 << 20):
                            blk = torch.empty(f_ - keep, dtype=torch.uint8, device="cuda")
                            del blk
                        if hasattr(wl, "no_release"):
                            wl.no_release = True
                        ms_ = torch.cuda.memory_stats()
                        log(f"WL_WARM arena: device alloc {ms_.get('num_device_alloc', 0)} free "
                            f"{ms_.get('num_device_free', 0)} retries {ms_.get('num_alloc_retries', 0)}")
                        _nt("arena reserved")
                return wl
            self._prewl = self._wl_exec.submit(_warm).result()
            self._prewl_warm = True
        elif wk0 in ("max_shard", "local_rl"):
            self._prewl = self._build_red(wk0, model)     # all setup before ready (idle control = no work after)
        self.t_ready = time.time()     # the verifier's noise starts at ready (max_shard's start-up burst)
        try:
            # one read-only line for the next run's memory breakdown; no allocation, never fails a run.
            # non-torch = everything resident that the torch allocator does not own: the CUDA context, loaded
            # modules (Triton/native challenge + chase kernels), cuBLAS workspace, allocator overhead -- the
            # ~1.16 GiB that shrinks the cheater's free HBM. All MiB.
            free_b, total_b = torch.cuda.mem_get_info()
            resv, alloc = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
            nontorch = (total_b - free_b) - resv
            st = torch.cuda.memory_stats()
            large = st.get("allocated_bytes.large_pool.current", 0)
            log(f"CHEAT_MEM total={total_b >> 20} free={free_b >> 20} nontorch={nontorch >> 20} "
                f"reserved={resv >> 20} allocated={alloc >> 20} alloc_large={large >> 20} "
                f"workload={self.spec.get('workload')}")
        except Exception as e:
            log(f"CHEAT_MEM warn: {type(e).__name__}: {e}")
        _ml = float(os.environ.get("VARENA_MEMLOG_S", "0"))
        if _ml > 0:
            # periodic memory line (non-torch growth once the workload thread runs)
            def _memlog():
                while True:
                    time.sleep(_ml)
                    try:
                        f_, t_ = torch.cuda.mem_get_info()
                        r_, a_ = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
                        log(f"CHEAT_MEMLOG free={f_ >> 20} nontorch={(t_ - f_ - r_) >> 20} reserved={r_ >> 20} "
                            f"allocated={a_ >> 20} peak={torch.cuda.max_memory_allocated() >> 20} "
                            f"threads={threading.active_count()}")
                    except Exception as e:
                        log(f"CHEAT_MEMLOG warn: {type(e).__name__}: {e}")
            threading.Thread(target=_memlog, daemon=True, name="memlog").start()
        # declare the reference scheduler config (self-reported, as any prover's would be)
        self.info.setdefault("sched", {"max_num_batched_tokens": int(os.environ.get("VA_TOKCAP", "1024"))})
        _gl = float(os.environ.get("VARENA_GATE_LOG_S", "0") or 0)
        if _gl > 0:
            self._start_gate_log(_gl)
        self.conn.send({"op": "ready", "info": self.info})
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._challenger, daemon=True).start()
        threading.Thread(target=self._generator, daemon=True).start()
        if self.streamer:
            threading.Thread(target=self._flusher, daemon=True).start()
            self.streamer.start()
        wk = self.spec.get("workload")
        if wk:
            if getattr(self, "_wl_exec", None) is not None:      # the same OS thread that ran the warm step
                self.wl_thread = self._wl_exec.submit(self._workload, wk, model)
            else:
                self.wl_thread = threading.Thread(target=self._workload, args=(wk, model), daemon=True)
                self.wl_thread.start()
        while True:
            time.sleep(1)


def main():
    spec = json.loads(os.environ.get("VARENA_CHEAT", "{}"))
    model = os.environ["VARENA_MODEL"]
    c = Cheater(spec)
    c.connect()
    c.setup(model, int(os.environ.get("VARENA_NUM_BLOCKS", "4096")))
    c.run(model)


if __name__ == "__main__":
    main()
