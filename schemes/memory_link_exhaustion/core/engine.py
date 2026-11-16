"""Static-batch generation: refill KV, chunked prefill, CUDA-graph decode.

Each batch runs as numbered phases (0 = KV refill, 1..C = prefill chunks,
C+1.. = decode steps). The host synchronises at the end of every phase and
publishes the phase number, so a memory challenge that reads (batch, phase)
before and after itself knows which states the KV region could have been in.
"""

from __future__ import annotations

import threading
import time

import torch


class Engine:
    def __init__(self, model, kv, B: int, P: int, n_new: int, chunk_seqs: int = 8,
                 graph: bool = True, stream: torch.cuda.Stream | None = None,
                 temperature: float = 0.0, phase_hook=None, capture_mode: str = "global"):
        self.m, self.kv, self.B, self.P, self.n_new = model, kv, B, P, n_new
        self.chunk = chunk_seqs
        self.stream = stream or torch.cuda.Stream(priority=-1)
        self.temperature = temperature
        self.capture_mode = capture_mode
        self.phase_hook = phase_hook          # f(batch_id, phase, "begin"|"end")
        self.batch_id = -1
        self.phase = -1
        self.lock = threading.Lock()
        dev = model.device
        self.ids = torch.zeros(B, dtype=torch.long, device=dev)
        self.pos = torch.zeros(B, dtype=torch.int32, device=dev)
        self.lens = torch.ones(B, dtype=torch.int32, device=dev)
        self.zero_pos = torch.zeros(B, dtype=torch.int32, device=dev)
        self.graph = None
        self.logits = None
        self.step_times = []      # (t_end, seconds) per decode step, for pacing
        if graph:
            self._capture()
        # capture wrote KV slots; start every engine from an empty (filler) cache
        with torch.cuda.stream(self.stream):
            self.kv.reset()
        self.stream.synchronize()

    def _capture(self):
        with torch.cuda.stream(self.stream):
            for _ in range(2):    # compile Triton kernels and warm cuBLAS
                self.logits = self.m.decode(self.ids, self.pos, self.lens, self.kv)
            self.stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=self.stream, capture_error_mode=self.capture_mode):
                self.logits = self.m.decode(self.ids, self.pos, self.lens, self.kv)
            self.stream.synchronize()
        self.graph = g

    def _set_phase(self, p, what):
        if what == "begin":
            self.phase = p
        if self.phase_hook:
            self.phase_hook(self.batch_id, p, what)

    def _sample(self, logits):
        if self.temperature <= 0:
            return logits.argmax(-1)
        probs = torch.softmax(logits.float() / self.temperature, -1)
        return torch.multinomial(probs, 1).squeeze(-1)

    @torch.no_grad()
    def generate(self, prompts: torch.Tensor, batch_id: int = 0) -> torch.Tensor:
        """prompts [B, P] long on device. Returns tokens [B, n_new] (device)."""
        B, P = prompts.shape
        assert B == self.B and P == self.P
        out = torch.empty(B, self.n_new, dtype=torch.long, device=prompts.device)
        s = self.stream
        self.batch_id = batch_id
        with torch.cuda.stream(s):
            ph = 0
            self._set_phase(ph, "begin")
            self.kv.reset()
            s.synchronize()
            self._set_phase(ph, "end")
            for b0 in range(0, B, self.chunk):
                ph += 1
                self._set_phase(ph, "begin")
                h = self.m.prefill(prompts[b0:b0 + self.chunk], self.kv, b0, self.zero_pos)
                out[b0:b0 + self.chunk, 0] = self._sample(self.m.logits(h))
                s.synchronize()
                self._set_phase(ph, "end")
            self.ids.copy_(out[:, 0])
            for t in range(1, self.n_new):
                ph += 1
                self._set_phase(ph, "begin")
                t0 = time.perf_counter()
                self.pos.fill_(P + t - 1)
                self.lens.fill_(P + t)
                if self.graph is not None:
                    self.graph.replay()
                    logits = self.logits
                else:
                    logits = self.m.decode(self.ids, self.pos, self.lens, self.kv)
                nxt = self._sample(logits)
                out[:, t] = nxt
                self.ids.copy_(nxt)
                s.synchronize()
                t1 = time.perf_counter()
                self.step_times.append((t1, t1 - t0))
                if len(self.step_times) > 4096:
                    del self.step_times[:2048]
                self._set_phase(ph, "end")
        return out

    def n_phases(self):
        return 1 + (self.B + self.chunk - 1) // self.chunk + (self.n_new - 1)
