"""Native (C++, GIL-free) paths: challenge server, noise streamer, single H2D queue, Revealer.

Prover side: `start(...)` runs a C++ thread that serves challenges over a Unix socket. Each challenge kind is a
pre-captured CUDA graph (the mixed chase) whose kernel reads the
nonce from pinned host memory; the thread launches the graph on its own stream, waits, and sends the pinned output
buffer. `sstream_start` runs the noise streamer; `sq_*` is the single host-to-device queue that tools/h2d_route.c
routes vLLM's small copies into (event mode: one queue stream per caller stream, the noise yields to ready copies).

Verifier side: `rtt(fd, id, kind, nonce)` sends a request and reads the answer with the GIL released, timing the
round trip in C++; `revealer` is the window-reveal loop.
"""

from __future__ import annotations

import os

_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <dlfcn.h>
#include <deque>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {
struct Graph { cudaGraphExec_t exec; uint32_t* nonce; const uint8_t* out; uint64_t bytes;
               volatile int32_t* ctl; int32_t seq; cudaEvent_t ev; bool launched; };
std::vector<Graph> G;
cudaStream_t S = nullptr, S2 = nullptr;
volatile int64_t* LAST = nullptr;
volatile int32_t* BUSY = nullptr;   // 1 while a challenge is in flight (prover pauses background GPU work)
// 1 while the prover replaces the device KV block table (settled-KV rule). Table switch and challenge exclude each
// other: the server raises BUSY, then looks at TABUPD and backs off if it is set; the prover raises TABUPD, then waits
// for BUSY to drop before it writes. So no chase ever reads a half-replaced table.
volatile int32_t* TABUPD = nullptr;
int DEV = 0;
uint64_t rt_ns() { timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return uint64_t(ts.tv_sec) * 1000000000ull + ts.tv_nsec; }

// resident kernels: relaunch whenever the previous instance has exited (idle timeout)
void keeper() {
  cudaSetDevice(DEV);
  for (;;) {
    for (auto& g : G) {
      if (!g.ctl) continue;
      if (g.ctl[2] != 0) continue;
      if (g.launched && cudaEventQuery(g.ev) != cudaSuccess) continue;  // still starting/exiting
      g.ctl[2] = 2;                       // starting (the kernel sets 1, and 0 on exit)
      cudaGraphLaunch(g.exec, S2);
      cudaEventRecord(g.ev, S2);
      g.launched = true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

uint32_t fmix(uint32_t x) { x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16; return x; }
struct Req { uint32_t id, kind, nonce, pad; };
struct Resp { uint32_t id, kind; int32_t win; uint32_t nonce_eff; uint64_t serve_ns; uint64_t bytes;
              uint64_t t_recv, t_launch, t_done; };

bool rd(int fd, void* b, size_t n) {
  auto* p = (uint8_t*)b;
  while (n) { ssize_t k = ::read(fd, p, n); if (k <= 0) return false; p += k; n -= k; }
  return true;
}
bool wr(int fd, const void* b, size_t n) {
  auto* p = (const uint8_t*)b;
  while (n) { ssize_t k = ::write(fd, p, n); if (k <= 0) return false; p += k; n -= k; }
  return true;
}

void serve(int lfd) {
  cudaSetDevice(DEV);
  for (;;) {
    int fd = ::accept(lfd, nullptr, nullptr);
    if (fd < 0) return;
    Req r;
    while (rd(fd, &r, sizeof r)) {
      auto t0 = std::chrono::steady_clock::now();
      if (TABUPD && BUSY) {
        for (;;) {
          *BUSY = (int32_t)r.kind + 1;
          __sync_synchronize();
          if (!*TABUPD) break;
          *BUSY = 0;
          __sync_synchronize();
          while (*TABUPD) { }
        }
      }
      uint64_t tr = rt_ns();
      int64_t last = LAST ? *LAST : -1;
      int32_t win = last < 0 ? -1 : (int32_t)(last >> 32);
      uint32_t d0 = (uint32_t)(last & 0xffffffffLL);
      uint32_t n = r.nonce;
      if (win >= 0) n = fmix(r.nonce ^ fmix(d0 ^ 0x68E31DA4u));
      Resp h{r.id, r.kind, win, n, 0, 0, tr, 0, 0};
      if (BUSY) *BUSY = (int32_t)r.kind + 1;
      if (r.kind < G.size()) {
        Graph& g = G[r.kind];
        *(volatile uint32_t*)g.nonce = n;
        ((volatile uint32_t*)g.nonce)[1] = (uint32_t)win;   // head for ring probes: same read of LAST as the nonce chain
        if (g.ctl) {
          __sync_synchronize();
          h.t_launch = rt_ns();
          g.ctl[0] = ++g.seq;               // ring
          auto tw = std::chrono::steady_clock::now();
          while (g.ctl[1] != g.seq) {
            if (std::chrono::steady_clock::now() - tw > std::chrono::seconds(20)) break;
          }
          __sync_synchronize();
        } else {
          h.t_launch = rt_ns();
          cudaGraphLaunch(g.exec, S);
          cudaStreamSynchronize(S);
        }
        h.t_done = rt_ns();
        if (BUSY) *BUSY = 0;
        h.bytes = g.bytes;
        h.serve_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (!wr(fd, &h, sizeof h) || !wr(fd, g.out, g.bytes)) break;
      } else {
        if (!wr(fd, &h, sizeof h)) break;
      }
    }
    ::close(fd);
  }
}
}  // namespace

void start(std::string path, std::vector<int64_t> execs, std::vector<int64_t> nonces,
           std::vector<int64_t> outs, std::vector<int64_t> bytes, int64_t last, int64_t device,
           std::vector<int64_t> ctls, int64_t busy, int64_t tabupd) {
  BUSY = busy ? (volatile int32_t*)busy : nullptr;
  TABUPD = tabupd ? (volatile int32_t*)tabupd : nullptr;
  DEV = (int)device;
  cudaSetDevice(DEV);
  int lo, hi;
  cudaDeviceGetStreamPriorityRange(&lo, &hi);
  cudaStreamCreateWithPriority(&S, cudaStreamNonBlocking, hi);
  cudaStreamCreateWithPriority(&S2, cudaStreamNonBlocking, hi);
  bool any_res = false;
  for (size_t i = 0; i < execs.size(); i++) {
    Graph g{(cudaGraphExec_t)execs[i], (uint32_t*)nonces[i], (const uint8_t*)outs[i],
            (uint64_t)bytes[i], (volatile int32_t*)(i < ctls.size() ? ctls[i] : 0), 0, nullptr, false};
    if (g.ctl) { cudaEventCreateWithFlags(&g.ev, cudaEventDisableTiming); any_res = true; }
    G.push_back(g);
  }
  if (any_res) std::thread(keeper).detach();
  LAST = (volatile int64_t*)last;
  int lfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  sockaddr_un a{};
  a.sun_family = AF_UNIX;
  std::strncpy(a.sun_path, path.c_str(), sizeof(a.sun_path) - 1);
  ::unlink(path.c_str());
  if (::bind(lfd, (sockaddr*)&a, sizeof a) || ::listen(lfd, 1)) throw std::runtime_error("bind/listen");
  std::thread(serve, lfd).detach();
}

// ---------------------------------------------------------------------------------------------
// Scheduled PCIe noise stream (prover), GIL-free. Window i: wait until the verifier has revealed
// s_i (rev[i % N] == i << 32 | s_i), copy pool slice sched_offset(s_i) to ring slot i % n_ring,
// and on completion publish rep[i % N] = i << 32 | s_i (shared with the verifier, whose Revealer process
// timestamps it) plus a local report for Python bookkeeping.
// ---------------------------------------------------------------------------------------------
// Single H2D queue (VARENA_SINGLE_QUEUE=1): the prover's own host-to-device copies (vLLM's input buffers through
// sq_submit, and with tools/h2d_route.so preloaded every cudaMemcpy[Async] H2D in the process through varena_sq_route)
// are enqueued at submit on a highest-priority queue stream of their own for each caller stream: wait(ready event on
// the caller's stream) -> copy -> record done -> caller waits(done). Stream order is exactly what a direct copy gives
// (a kernel still reading the destination is never overwritten under it). Only event waits on work already enqueued,
// so hardware-queue sharing (CUDA_DEVICE_MAX_CONNECTIONS) cannot form a cycle, and one queue stream per caller stream
// means a stream whose earlier work does not finish yet (a matmul held by VARENA_MM_GATE until the host ends a
// challenge) delays only its own copies. Two earlier designs deadlocked on the H200: a caller stream waiting on a flag
// written by the streamer (cuStreamWaitValue64 holds a shared hardware queue), and one shared queue stream for all
// callers (a gated stream blocked every later copy). A guard thread turns routing off (copies then go direct) if a copy
// is stuck. A pageable source is copied into pinned staging at submit, as the runtime does.
// VARENA_SQ_YIELD=1 (default): the noise stream yields the link to queued copies: before each noise window the streamer
// makes the noise stream wait for the done event of every queued copy whose ready event has completed (event waits on
// already-enqueued copies only: no compute dependency, no cycle).
namespace sq {
std::mutex mu;                               // submitters (the streamer never takes it)
std::atomic<int64_t> head{0}, tail{0};       // last submitted / last issued seq
bool started = false;                        // sq_on ran (streamer thread)
std::atomic<bool> active{false};
int dev = 0;
uint8_t* stage = nullptr;                    // pinned staging ring for pageable sources
int64_t stage_cap = 0, stage_head = 0, stage_tail = 0;
std::atomic<int64_t> n_q{0}, n_staged{0}, n_full{0}, n_stage_full{0}, n_err{0};
thread_local bool own = false;               // the streamer's thread: its own copies are noise
thread_local bool in_route = false;          // a submitter issuing a queued copy itself
bool yield = false;
std::atomic<int64_t> n_yield_waits{0}, n_yield_calls{0};
}
namespace sqe {
struct Pending { cudaEvent_t ready, done; cudaStream_t caller; uint64_t t0, t_ready; bool ready_seen, yielded;
                 int64_t n; bool pageable; };
std::deque<Pending> pend;                                   // under sq::mu
std::vector<cudaEvent_t> pool;                              // under sq::mu
std::unordered_map<cudaStream_t, cudaStream_t> qmap;        // caller stream -> its queue stream, under sq::mu
std::deque<std::pair<cudaEvent_t, int64_t>> stage_fifo;     // staged copies in order: done event, staging head after
std::atomic<bool> tripped{false}, guard_run{false};
std::thread guard;
std::atomic<int64_t> n_streams{0};
uint64_t now_ns() { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return uint64_t(ts.tv_sec) * 1000000000ull + ts.tv_nsec; }
cudaEvent_t get_ev() {
  if (!pool.empty()) { cudaEvent_t e = pool.back(); pool.pop_back(); return e; }
  cudaEvent_t e = nullptr;
  if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) { cudaGetLastError(); return nullptr; }
  return e;
}
}

extern "C" int varena_sq_route(void* dst, const void* src, size_t n, int kind, void* stream, int sync);

// queue one H2D copy for `stream`; returns its seq, or -1 (nothing enqueued on the stream: copy it directly)
int64_t sq_enqueue_event(int64_t dst, int64_t src, int64_t n, int64_t stream, bool pageable);
int64_t sq_enqueue_event_retry(int64_t, int64_t, int64_t, int64_t, bool, std::unique_lock<std::mutex>&);
int64_t sq_enqueue(int64_t dst, int64_t src, int64_t n, int64_t stream, bool pageable) {
  return sq_enqueue_event(dst, src, n, stream, pageable);
}

// Python entry (inference/plugin.py NonUvaBuffer.uva): pinned source, caller already checked for graph capture
int64_t sq_submit(int64_t dst, int64_t src, int64_t n, int64_t stream) {
  return sq_enqueue(dst, src, n, stream, false);
}
bool sq_active() { return sq::active.load(); }

void sqe_stage_reclaim() {                                   // under sq::mu
  while (!sqe::stage_fifo.empty()) {
    auto& f = sqe::stage_fifo.front();
    if (cudaEventQuery(f.first) == cudaErrorNotReady) { cudaGetLastError(); break; }
    sq::stage_tail = f.second;
    sqe::pool.push_back(f.first);
    sqe::stage_fifo.pop_front();
  }
}

int64_t sq_enqueue_event_retry(int64_t dst, int64_t src, int64_t n, int64_t stream, bool pageable,
                               std::unique_lock<std::mutex>& g) {
  g.unlock();                                                // staging moved under us: start over
  return sq_enqueue_event(dst, src, n, stream, pageable);
}
int64_t sq_enqueue_event(int64_t dst, int64_t src, int64_t n, int64_t stream, bool pageable) {
  std::unique_lock<std::mutex> g(sq::mu);
  if (!sq::active.load() || sqe::tripped.load()) return -1;
  if (sqe::pend.size() >= 8192) { sq::n_full++; return -1; }
  cudaStream_t cs = (cudaStream_t)stream, qs;
  auto it = sqe::qmap.find(cs);
  if (it != sqe::qmap.end()) qs = it->second;
  else {
    if (sqe::qmap.size() >= 256) { sq::n_full++; return -1; }
    int lo, hi;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    if (cudaStreamCreateWithPriority(&qs, cudaStreamNonBlocking, hi) != cudaSuccess) {
      cudaGetLastError(); sq::n_err++; return -1;
    }
    sqe::qmap[cs] = qs;
    sqe::n_streams = (int64_t)sqe::qmap.size();
  }
  int64_t srcp = src, h = sq::stage_head;
  if (pageable) {
    const int64_t cap = sq::stage_cap;
    if (n > cap) { sq::n_stage_full++; return -1; }
    const int64_t off = h % cap, pad = (off + n > cap) ? cap - off : 0;
    sqe_stage_reclaim();
    if (h + pad + n - sq::stage_tail > cap) {               // wait a little for earlier staged copies to land
      const uint64_t t0 = sqe::now_ns();
      while (h + pad + n - sq::stage_tail > cap && sqe::now_ns() - t0 < 200000000ull) {
        g.unlock(); std::this_thread::sleep_for(std::chrono::microseconds(20)); g.lock();
        if (sq::stage_head != h) return sq_enqueue_event_retry(dst, src, n, stream, pageable, g);
        sqe_stage_reclaim();
      }
      if (h + pad + n - sq::stage_tail > cap) { sq::n_stage_full++; return -1; }
    }
    h += pad;
    std::memcpy(sq::stage + h % cap, (const void*)src, (size_t)n);
    srcp = (int64_t)(sq::stage + h % cap);
    h += n;
  }
  cudaEvent_t ready = sqe::get_ev(), done = sqe::get_ev(), se = pageable ? sqe::get_ev() : nullptr;
  auto give_back = [&] { for (auto e : {ready, done, se}) if (e) sqe::pool.push_back(e); };
  if (!ready || !done || (pageable && !se) || cudaEventRecord(ready, cs) != cudaSuccess) {
    cudaGetLastError(); give_back(); sq::n_err++; return -1;
  }
  sq::in_route = true;
  const bool ok = cudaStreamWaitEvent(qs, ready, 0) == cudaSuccess &&
                  cudaMemcpyAsync((void*)dst, (const void*)srcp, n, cudaMemcpyHostToDevice, qs) == cudaSuccess &&
                  cudaEventRecord(done, qs) == cudaSuccess && (!se || cudaEventRecord(se, qs) == cudaSuccess);
  sq::in_route = false;
  if (!ok) {   // nothing waits on it; the caller copies directly (its stream order is unaffected)
    cudaGetLastError(); sq::n_err++;
    cudaStreamSynchronize(qs);
    give_back();
    return -1;
  }
  if (cudaStreamWaitEvent(cs, done, 0) != cudaSuccess) { cudaGetLastError(); sq::n_err++; cudaEventSynchronize(done); }
  if (se) { sq::stage_head = h; sqe::stage_fifo.emplace_back(se, h); }
  sqe::pend.push_back({ready, done, cs, sqe::now_ns(), 0, false, false, n, pageable});
  const int64_t seq = sq::head.load() + 1;
  sq::head.store(seq); sq::tail.store(seq);
  sq::n_q++;
  if (pageable) sq::n_staged++;
  return seq;
}

// Guard (event mode): every 20 ms retire finished copies; if a copy has not landed VARENA_SQ_GUARD_S (default 5) s
// after its stream's earlier work finished, or its stream's earlier work has not finished after
// VARENA_SQ_GUARD_READY_S (default 30) s, turn routing off for good (every later copy goes direct) and say so loudly.
void sqe_trip(const char* why, const sqe::Pending& p, double s);
void sqe_guard() {
  static const double gs = getenv("VARENA_SQ_GUARD_S") ? atof(getenv("VARENA_SQ_GUARD_S")) : 5.0;
  static const double gr = getenv("VARENA_SQ_GUARD_READY_S") ? atof(getenv("VARENA_SQ_GUARD_READY_S")) : 30.0;
  cudaSetDevice(sq::dev);
  typedef int (*capturing_t)();
  const auto capturing = (capturing_t)dlsym(RTLD_DEFAULT, "varena_h2d_capturing");
  while (sqe::guard_run.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (capturing && capturing() > 0) continue;   // no event queries while a graph is captured (and no deadline counts)
    std::lock_guard<std::mutex> g(sq::mu);
    const uint64_t t = sqe::now_ns();
    for (auto it = sqe::pend.begin(); it != sqe::pend.end();) {
      if (!it->ready_seen) {
        if (cudaEventQuery(it->ready) == cudaErrorNotReady) cudaGetLastError();
        else { it->ready_seen = true; it->t_ready = t; }
      }
      if (it->ready_seen) {
        if (cudaEventQuery(it->done) != cudaErrorNotReady) {
          // VARENA_SQ_SLOW_MS (honest-dip diagnosis): log routed copies that landed more than this
          // long after their caller stream was ready (the caller's stream waits on them on the GPU), +-20 ms (poll)
          static const double slow_ms = getenv("VARENA_SQ_SLOW_MS") ? atof(getenv("VARENA_SQ_SLOW_MS")) : 0.0;
          if (slow_ms > 0) {
            const double lat = (t - it->t_ready) * 1e-6, q = (t - it->t0) * 1e-6;
            if (lat > slow_ms) {
              timespec w; clock_gettime(CLOCK_REALTIME, &w); struct tm tmv; localtime_r(&w.tv_sec, &tmv);
              fprintf(stderr, "%02d:%02d:%02d.%03ld [sstream] SLOW_ROUTED copy landed %.0f ms after its stream was ready "
                      "(%.0f ms after submit), %ld bytes, %s, %zu pending\n", tmv.tm_hour, tmv.tm_min, tmv.tm_sec,
                      w.tv_nsec / 1000000, lat, q, (long)it->n, it->pageable ? "pageable" : "pinned", sqe::pend.size());
              fflush(stderr);
            }
          }
          sqe::pool.push_back(it->ready); sqe::pool.push_back(it->done);
          it = sqe::pend.erase(it);
          continue;
        }
        cudaGetLastError();
        if (!sqe::tripped && (t - it->t_ready) * 1e-9 > gs) sqe_trip("copy not landed after its stream was ready", *it, (t - it->t_ready) * 1e-9);
      } else if (!sqe::tripped && (t - it->t0) * 1e-9 > gr) {
        sqe_trip("caller stream's earlier work not finished", *it, (t - it->t0) * 1e-9);
      }
      ++it;
    }
    sqe_stage_reclaim();
  }
}
void sqe_trip(const char* why, const sqe::Pending& p, double s) {
  sqe::tripped = true;
  typedef void (*set_route_t)(void*);
  if (auto f = (set_route_t)dlsym(RTLD_DEFAULT, "varena_h2d_set_route")) f(nullptr);
  fprintf(stderr, "[sstream] !!!!! H2D QUEUE GUARD TRIPPED: %s (%.1f s, caller stream %p, %ld bytes, %s, ready %s; "
          "%zu pending, %zu caller streams). Routing is OFF from now on: every host-to-device copy goes direct. !!!!!\n",
          why, s, (void*)p.caller, (long)p.n, p.pageable ? "pageable (staged)" : "pinned",
          p.ready_seen ? "seen" : "not seen", sqe::pend.size(), sqe::qmap.size());
  fflush(stderr);
}

// Yield (event mode, VARENA_SQ_YIELD=1, streamer thread): the noise stream waits for every queued copy whose stream
// was ready; never blocks the streamer (try_lock)
void sqe_yield(cudaStream_t noise) {
  std::unique_lock<std::mutex> g(sq::mu, std::try_to_lock);
  if (!g.owns_lock()) return;
  bool any = false;
  for (auto& p : sqe::pend) {
    if (p.yielded) continue;
    if (!p.ready_seen) {
      if (cudaEventQuery(p.ready) == cudaErrorNotReady) { cudaGetLastError(); continue; }
      p.ready_seen = true; p.t_ready = sqe::now_ns();
    }
    cudaStreamWaitEvent(noise, p.done, 0);
    p.yielded = true;
    sq::n_yield_waits++; any = true;
  }
  if (any) sq::n_yield_calls++;
}

// Noise-stream timing (VARENA_NOISE_TIMING=1, default on): timing events right before and right after each window's
// copies on the noise stream. busy = after_i - before_i; gap = before_i - after_{i-1}, the time the link had no noise
// (it may carry a routed copy, inline copies count here too). A gap is "yield" if the streamer placed yield waits on
// the noise stream between the two windows, else "other" (the host enqueued the next window late, or nothing queued).
// Printed every VARENA_NOISE_TIMING_S (10) s and at the end; windows 2L apart share an event slot.
inline void noise_nap(int us) { std::this_thread::sleep_for(std::chrono::microseconds(us)); }

namespace nt {
// VARENA_NOISE_TIMING: 1 (default) both events, timed; 0 none; "b" / "a" only the before / after event (timed, no
// accounting); "notime" both events created with cudaEventDisableTiming (no accounting). Bisect knob: serving
// went from 54.2 to 55.6 GB/s when these events were added, with nothing else changed on the noise path.
bool on = false, rec_b = false, rec_a = false;
int64_t L2 = 0;
std::vector<cudaEvent_t> b, a;
std::vector<char> yf;
int64_t y_mark = 0;
std::atomic<int64_t> busy_ns{0}, idle_y_ns{0}, idle_o_ns{0}, n{0}, n_gap_y{0}, reveal_wait_ns{0}, n_reveal_wait{0};
// "idle other" gaps by size: < 10 us (per-copy turnaround on the GPU), 10-100 us, 0.1-1 ms, >= 1 ms (count, ns)
constexpr int NH = 4;
std::atomic<int64_t> gh_n[NH], gh_ns[NH];
int64_t last_gh_n[NH], last_gh_ns[NH];
// late issues: the previous window's copies had already finished when the next window was enqueued (the link was idle
// because of the host), by what the streamer had just waited on: 0 nothing (reveal, pacing, host), 1 the H2D copy,
// 2 the echo kernel, 3 the D2H echo
std::atomic<int64_t> late[4];
int64_t last_late[4];
int last_wait = 0;
uint64_t t_print = 0, t_start = 0;
int64_t last_n = 0, last_busy = 0, last_iy = 0, last_io = 0, last_y = 0, last_rw = 0, last_nrw = 0;
void init(int64_t lag) {
  const std::string m = getenv("VARENA_NOISE_TIMING") ? getenv("VARENA_NOISE_TIMING") : "1";
  on = m == "1";
  rec_b = on || m == "b" || m == "notime";
  rec_a = on || m == "a" || m == "notime";
  if (!rec_b && !rec_a) return;
  L2 = 2 * std::max<int64_t>(lag, 1);
  b.assign(L2, nullptr); a.assign(L2, nullptr); yf.assign(L2, 0);
  const unsigned fl = m == "notime" ? cudaEventDisableTiming : 0u;
  for (int64_t k = 0; k < L2; k++) { cudaEventCreateWithFlags(&b[k], fl); cudaEventCreateWithFlags(&a[k], fl); }
  if (m != "1") fprintf(stderr, "[sstream] noise timing events: mode %s (no accounting)\n", m.c_str());
}
void before(int64_t i, cudaStream_t st) {          // right before the window's copies (after any yield waits)
  if (!rec_b) return;
  const int64_t y = sq::n_yield_waits.load();
  yf[i % L2] = y != y_mark; y_mark = y;
  cudaEventRecord(b[i % L2], st);
}
void after(int64_t i, cudaStream_t st) { if (rec_a) { cudaEventRecord(a[i % L2], st); y_mark = sq::n_yield_waits.load(); } }
void report(const char* tag, int64_t W, bool final);
void done(int64_t i, int64_t W) {                  // window i has completed (its after-event too)
  if (!on) return;
  float ms = 0;
  int64_t bu = -1, g = 0;
  if (cudaEventElapsedTime(&ms, b[i % L2], a[i % L2]) == cudaSuccess) bu = (int64_t)(ms * 1e6);
  else cudaGetLastError();
  if (i > 0 && cudaEventElapsedTime(&ms, a[(i - 1) % L2], b[i % L2]) == cudaSuccess) {
    g = (int64_t)(ms * 1e6);
    // overlapping windows (defensive): count only the part after window i-1's end as busy, so busy + gaps stays
    // wall time
    if (g < 0) { if (bu >= 0) bu = std::max<int64_t>(0, bu + g); g = 0; }
    if (yf[i % L2]) { idle_y_ns += g; n_gap_y++; }
    else {
      idle_o_ns += g;
      const int h = g < 10000 ? 0 : g < 100000 ? 1 : g < 1000000 ? 2 : 3;
      gh_n[h]++; gh_ns[h] += g;
    }
  } else cudaGetLastError();
  if (bu >= 0) busy_ns += bu;
  n++;
  timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  const uint64_t t = uint64_t(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
  if (!t_start) { t_start = t; t_print = t; }
  static const double per = getenv("VARENA_NOISE_TIMING_S") ? atof(getenv("VARENA_NOISE_TIMING_S")) : 10.0;
  if ((t - t_print) * 1e-9 >= per) { report("interval", W, false); t_print = t; }
}
void report(const char* tag, int64_t W, bool final) {
  const int64_t nn = n.load(), bu = busy_ns.load(), iy = idle_y_ns.load(), io = idle_o_ns.load(),
                yy = sq::n_yield_waits.load(), rw = reveal_wait_ns.load(), nrw = n_reveal_wait.load();
  const int64_t drw = final ? rw : rw - last_rw, dnrw = final ? nrw : nrw - last_nrw;
  const int64_t dn = final ? nn : nn - last_n, db = final ? bu : bu - last_busy, diy = final ? iy : iy - last_iy,
                dio = final ? io : io - last_io, dy = final ? yy : yy - last_y;
  const double tot = (double)(db + diy + dio);
  if (tot > 0)
    fprintf(stderr, "[sstream] noise timing (%s): windows %ld, %.2f GB/s while busy, busy %.2f%%, idle at yields %.3f%%, "
            "idle other %.3f%%, yield waits %ld; streamer waited for the verifier's reveal %ld times, "
            "%.1f ms\n", tag, (long)dn,
            db > 0 ? dn * (double)W / db : 0.0, 100.0 * db / tot, 100.0 * diy / tot, 100.0 * dio / tot, (long)dy,
            (long)dnrw, drw * 1e-6);
  if (tot > 0) {
    int64_t hn[NH], hs[NH], lt[4];
    for (int k = 0; k < NH; k++) { hn[k] = gh_n[k].load() - (final ? 0 : last_gh_n[k]); hs[k] = gh_ns[k].load() - (final ? 0 : last_gh_ns[k]); }
    for (int k = 0; k < 4; k++) lt[k] = late[k].load() - (final ? 0 : last_late[k]);
    fprintf(stderr, "[sstream] noise idle other by gap (%s): <10us %ld (%.3f%%), 10-100us %ld (%.3f%%), 0.1-1ms %ld (%.3f%%), "
            ">=1ms %ld (%.3f%%); late issues (link idle at enqueue) after waiting on: nothing %ld, h2d %ld, kernel %ld, "
            "d2h %ld\n", tag, (long)hn[0], 100.0 * hs[0] / tot, (long)hn[1], 100.0 * hs[1] / tot, (long)hn[2],
            100.0 * hs[2] / tot, (long)hn[3], 100.0 * hs[3] / tot, (long)lt[0], (long)lt[1], (long)lt[2], (long)lt[3]);
    if (!final) {
      for (int k = 0; k < NH; k++) { last_gh_n[k] += hn[k]; last_gh_ns[k] += hs[k]; }
      for (int k = 0; k < 4; k++) last_late[k] += lt[k];
    }
  }
  if (!final) { last_n = nn; last_busy = bu; last_iy = iy; last_io = io; last_y = yy; last_rw = rw; last_nrw = nrw; }
}
}

namespace ss {
struct St {
  volatile int64_t* rev; int64_t N; volatile int64_t* rep;
  const uint8_t* pool; int64_t n_slots, align, W;
  uint8_t* ring; int64_t n_ring; int64_t lag;
  cudaStream_t sh2d;
  volatile int64_t* loc; int64_t cap; volatile int64_t* head; volatile int64_t* last;
  volatile int32_t* run; volatile int64_t* pause_ns;
  std::vector<cudaEvent_t> evh, evk, evd; std::vector<int64_t> offs; std::vector<uint32_t> sv;
  int dev; std::thread th; bool on = false;
  volatile int64_t* hb = nullptr;      // heartbeat: mono ns at the start of an enqueue, 0 after
};
St T;
void sq_drain() {   // copies were issued at submit; make the noise stream yield to the ready ones
  if (sq::yield) sqe_yield(T.sh2d);
}
uint32_t fm(uint32_t x) { x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16; return x; }
uint64_t mono_ns() { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return uint64_t(ts.tv_sec) * 1000000000ull + ts.tv_nsec; }

double ms_since(uint64_t t0) { return (mono_ns() - t0) * 1e-6; }
void slow(const char* what, int64_t i, double ms) {
  if (ms > 10.0) fprintf(stderr, "[sstream] t=%.3f i=%ld %s %.1f ms\n", mono_ns() * 1e-9, (long)i, what, ms);
}

bool issue(int64_t i, bool force) {
  const int64_t slot = i % T.lag, k = i % T.N;
  int64_t v;
  const uint64_t t_rv = mono_ns();
  bool waited = false;
  for (;;) {
    v = T.rev[k];
    if ((v >> 32) == i) break;
    waited = true;
    if (!force && !*T.run) return false;
    sq_drain();
    noise_nap(20);
  }
  if (waited) { nt::reveal_wait_ns += (int64_t)(mono_ns() - t_rv); nt::n_reveal_wait++; }   // verifier-paced
  const uint32_t s = (uint32_t)(v & 0xffffffffLL);
  const int64_t off = (int64_t)(fm(s ^ 0x7FEB352Du) % (uint64_t)T.n_slots);    // the pool offset the revealed value names
  sq_drain();                 // pending prover copies go ahead of the next noise window
  cudaStream_t st = T.sh2d;
  if (!force && nt::on) {     // was the link already idle (previous window's copies done) when this one was enqueued?
    const cudaError_t pq = cudaEventQuery(T.evh[(i - 1) % T.lag]);
    if (pq == cudaSuccess) nt::late[nt::last_wait]++; else cudaGetLastError();
  }
  nt::before(i, st);
  T.offs[slot] = off; T.sv[slot] = s;
  __sync_synchronize();
  uint64_t t0 = mono_ns();
  if (T.hb) *T.hb = (int64_t)t0;
  cudaMemcpyAsync(T.ring + (i % T.n_ring) * T.W, T.pool + off * T.align, T.W, cudaMemcpyHostToDevice, st);
  if (T.hb) *T.hb = 0;
  nt::after(i, st);
  cudaEventRecord(T.evh[slot], st);
  slow("h2d-enqueue", i, ms_since(t0)); t0 = mono_ns();
  cudaEventRecord(T.evk[slot], st);        // (no per-window kernel and no copy back: both events follow the copy)
  cudaEventRecord(T.evd[slot], st);
  return true;
}

void complete(int64_t i) {
  const int64_t slot = i % T.lag;
  const uint64_t tw = mono_ns();
  nt::last_wait = 0;
  if (cudaEventQuery(T.evd[slot]) != cudaSuccess) {
    // which stage is late: H2D copy, kernel (graph), or D2H copy
    const bool h = cudaEventQuery(T.evh[slot]) == cudaSuccess, k = cudaEventQuery(T.evk[slot]) == cudaSuccess;
    nt::last_wait = !h ? 1 : !k ? 2 : 3;
    while (cudaEventQuery(T.evd[slot]) == cudaErrorNotReady) {     // poll: serve prover copies meanwhile
      sq_drain();
      noise_nap(5);
    }
    const double ms = ms_since(tw);
    if (ms > 10.0) slow(!h ? "wait(h2d-pending)" : !k ? "wait(kernel-pending)" : "wait(d2h-pending)", i, ms);
  }
  const uint64_t t = mono_ns();
  nt::done(i, T.W);
  const uint32_t s = T.sv[slot];
  T.rep[i % T.N] = (i << 32) | (int64_t)s;
  const int64_t h = *T.head, e = (h % T.cap) * 4;
  T.loc[e] = i; T.loc[e + 1] = T.offs[slot]; T.loc[e + 2] = s; T.loc[e + 3] = (int64_t)t;
  __sync_synchronize();
  *T.head = h + 1;
  if (T.last) *T.last = (i << 32) | (int64_t)s;
}

void sq_on() {
  cuInit(0);
  sq::dev = T.dev;
  sq::started = true;
  // yield on by default (with 4 windows always queued, a 70 KB copy on another stream sat > 5 s and a direct
  // pageable copy on the default stream hung with routing off: the copy engines do not share fairly with a stream that
  // never runs dry). VARENA_SQ_YIELD=0 turns it off.
  sq::yield = !(getenv("VARENA_SQ_YIELD") && getenv("VARENA_SQ_YIELD")[0] == '0');
  const char* sm = getenv("VARENA_SQ_STAGE_MIB");
  sq::stage_cap = (sm ? atoll(sm) : 64) << 20;
  if (sq::stage_cap > 0 && cudaHostAlloc((void**)&sq::stage, sq::stage_cap, cudaHostAllocDefault) != cudaSuccess) {
    cudaGetLastError(); sq::stage = nullptr; sq::stage_cap = 0;     // pageable sources then fall back
  }
  {
    std::lock_guard<std::mutex> g(sq::mu);
    for (int k = 0; k < 512; k++) if (auto e = sqe::get_ev()) sqe::pool.push_back(e);
  }
  sq::active.store(true);
  sqe::guard_run = true; sqe::guard = std::thread(sqe_guard);
  fprintf(stderr, "[sstream] single H2D queue on (%s mode, staging %ld MiB)\n", sq::yield ? "event+yield" : "event",
          (long)(sq::stage_cap >> 20));
  typedef void (*set_route_t)(void*);
  if (auto f = (set_route_t)dlsym(RTLD_DEFAULT, "varena_h2d_set_route")) {
    f((void*)&varena_sq_route);
    fprintf(stderr, "[sstream] h2d_route interposer: all host-to-device copies go through the queue\n");
  } else {
    fprintf(stderr, "[sstream] h2d_route interposer not preloaded: only sq_submit callers use the queue\n");
  }
}

void sq_off() {
  typedef void (*set_route_t)(void*);
  if (auto f = (set_route_t)dlsym(RTLD_DEFAULT, "varena_h2d_set_route")) f(nullptr);
  {
    std::lock_guard<std::mutex> g(sq::mu);            // no submit is half-way through
    sq::active.store(false);
  }
  {
    sqe::guard_run = false;
    if (sqe::guard.joinable()) sqe::guard.join();
    std::lock_guard<std::mutex> g(sq::mu);
    // never block shutdown on a stuck queue stream: poll each for up to 10 s, then report and go on
    const uint64_t t0 = sqe::now_ns();
    for (auto& kv : sqe::qmap) {
      while (cudaStreamQuery(kv.second) == cudaErrorNotReady) {
        cudaGetLastError();
        if (sqe::now_ns() - t0 > 10000000000ull) {
          fprintf(stderr, "[sstream] event queue: queue stream of caller %p still busy at stop; not waiting\n",
                  (void*)kv.first);
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    fprintf(stderr, "[sstream] event queue: %zu caller streams, guard %s\n", sqe::qmap.size(),
            sqe::tripped ? "TRIPPED" : "ok");
  }
  nt::report("run", T.W, true);
  fprintf(stderr, "[sstream] single H2D queue off: queued %ld (staged %ld), fallback queue-full %ld staging-full %ld "
          "error %ld\n", (long)sq::n_q.load(), (long)sq::n_staged.load(), (long)sq::n_full.load(),
          (long)sq::n_stage_full.load(), (long)sq::n_err.load());
}

void loop() {
  cudaSetDevice(T.dev);
  int64_t i = 0, n_issued = 0;
  sq::own = true;
  if (getenv("VARENA_SINGLE_QUEUE") && getenv("VARENA_SINGLE_QUEUE")[0] == '1' && !sq::started) sq_on();
  for (int64_t j = 0; j < T.lag; j++) issue(j, true);
  n_issued = T.lag;
  while (*T.run) {
    complete(i);
    const int64_t p = T.pause_ns ? *T.pause_ns : 0;
    if (p > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(p));
    if (!issue(i + T.lag, false)) { i++; break; }
    i++;
    n_issued = i + T.lag;
  }
  for (int64_t j = i; j < n_issued; j++) complete(j);
  if (sq::active.load()) sq_off();
}
}  // namespace ss

// Router for tools/h2d_route.so (registered by sq_on). 0: queued; 1: not for the queue; -1: eligible, not queued.
extern "C" int varena_sq_route(void* dst, const void* src, size_t n, int kind, void* stream, int sync) {
  if (!sq::active.load(std::memory_order_relaxed) || sq::own || sq::in_route) return 1;
  cudaStream_t s = (cudaStream_t)stream;
  // the legacy / per-thread default stream is not routed (H200: the first routed copy on it never landed and the
  // caller hung); vLLM runs on its own stream (VARENA_MAIN_STREAM=1). VARENA_SQ_ROUTE_DEFAULT=1 routes it anyway.
  static const bool route_default = getenv("VARENA_SQ_ROUTE_DEFAULT") && getenv("VARENA_SQ_ROUTE_DEFAULT")[0] == '1';
  if (!route_default && (s == nullptr || s == cudaStreamLegacy || s == cudaStreamPerThread)) return 1;
  // VARENA_SQ_ROUTE_PAGEABLE=0: pageable sources go direct (bisect knob)
  static const bool route_pageable = !(getenv("VARENA_SQ_ROUTE_PAGEABLE") && getenv("VARENA_SQ_ROUTE_PAGEABLE")[0] == '0');
  if (s && s == ss::T.sh2d) return 1;   // null = legacy stream, never ours
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(s, &cs) != cudaSuccess) { cudaGetLastError(); return 1; }  // e.g. legacy stream, global capture
  if (cs != cudaStreamCaptureStatusNone) return 1;
  cudaPointerAttributes a{};
  if (cudaPointerGetAttributes(&a, dst) != cudaSuccess) { cudaGetLastError(); return 1; }
  if (a.type != cudaMemoryTypeDevice) return 1;                      // e.g. cudaMemcpyDefault host->host
  if (a.device != sq::dev) return -1;
  int cur = -1;
  if (cudaGetDevice(&cur) != cudaSuccess || cur != sq::dev) { cudaGetLastError(); return -1; }
  if (cudaPointerGetAttributes(&a, src) != cudaSuccess) { cudaGetLastError(); return 1; }
  bool pageable;
  if (a.type == cudaMemoryTypeHost) pageable = false;                // pinned / registered
  else if (a.type == cudaMemoryTypeUnregistered) { if (!route_pageable) return 1; pageable = true; }
  else return 1;                                                     // device or managed source: not H2D
  if (sq_enqueue((int64_t)dst, (int64_t)src, (int64_t)n, (int64_t)stream, pageable) < 0) return -1;
  if (sync) cudaStreamSynchronize(s);
  return 0;
}

void sstream_start(int64_t rev, int64_t N, int64_t pool, int64_t n_slots, int64_t align, int64_t W,
                   int64_t ring, int64_t n_ring, int64_t lag, int64_t s_h2d, int64_t loc, int64_t cap, int64_t head, int64_t last, int64_t run,
                   int64_t pause_ns, int64_t device, int64_t hb) {
  using namespace ss;
  T.rev = (volatile int64_t*)rev; T.N = N; T.rep = (volatile int64_t*)rev + N;
  T.pool = (const uint8_t*)pool; T.n_slots = n_slots; T.align = align; T.W = W;
  T.ring = (uint8_t*)ring; T.n_ring = n_ring; T.lag = lag;
  T.sh2d = (cudaStream_t)s_h2d;
  T.loc = (volatile int64_t*)loc; T.cap = cap; T.head = (volatile int64_t*)head;
  T.last = last ? (volatile int64_t*)last : nullptr; T.run = (volatile int32_t*)run;
  T.pause_ns = pause_ns ? (volatile int64_t*)pause_ns : nullptr; T.dev = (int)device;
  T.hb = hb ? (volatile int64_t*)hb : nullptr;
  cudaSetDevice(T.dev);
  T.evh.assign(lag, nullptr); T.evk.assign(lag, nullptr); T.evd.assign(lag, nullptr);
  for (int64_t j = 0; j < lag; j++) {
    cudaEventCreateWithFlags(&T.evh[j], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&T.evk[j], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&T.evd[j], cudaEventDisableTiming | cudaEventBlockingSync);
  }
  T.offs.assign(lag, 0); T.sv.assign(lag, 0);
  nt::init(lag);
  T.th = std::thread(loop);
  T.on = true;
}

// Self-test driver (tests/test_h2d_route_gpu.py): saturating inbound noise without the verifier (W-byte pinned->device
// copies back to back, `inflight` in flight, prover copies drained between them exactly as the streamer does). One-shot per
// process. use_queue=false runs the same noise with the queue off (the starvation reference).
namespace sqt { std::thread th; std::atomic<bool> run{false}; std::atomic<int64_t> windows{0}; }
void sq_selftest_start(int64_t s_h2d, int64_t pool, int64_t ring, int64_t W, int64_t device, bool use_queue,
                       int64_t inflight) {
  ss::T.sh2d = (cudaStream_t)s_h2d; ss::T.dev = (int)device; ss::T.W = W;
  sqt::run = true; sqt::windows = 0;
  sqt::th = std::thread([=] {
    cudaSetDevice((int)device);
    sq::own = true;
    if (use_queue && !sq::started) ss::sq_on();
    const int64_t K = std::max<int64_t>(1, std::min<int64_t>(inflight, 16));   // windows in flight (SAT_LEAD-like)
    cudaEvent_t ev[16];
    for (int64_t k = 0; k < K; k++) cudaEventCreateWithFlags(&ev[k], cudaEventDisableTiming);
    nt::init(K);
    int64_t i = 0;
    while (sqt::run.load()) {
      const int slot = (int)(i % K);
      if (i >= K)
        while (cudaEventQuery(ev[slot]) == cudaErrorNotReady) {
          cudaGetLastError();
          if (use_queue) ss::sq_drain();
          noise_nap(5);
        }
      if (i >= K) nt::done(i - K, W);                        // window i - K has completed
      if (use_queue) ss::sq_drain();
      cudaStream_t st = ss::T.sh2d;
      nt::before(i, st);
      cudaMemcpyAsync((void*)ring, (const void*)pool, W, cudaMemcpyHostToDevice, st);
      nt::after(i, st);
      cudaEventRecord(ev[slot], st);
      sqt::windows = ++i;
    }
    cudaStreamSynchronize(ss::T.sh2d);
    if (use_queue && sq::active.load()) ss::sq_off();
  });
}
int64_t sq_selftest_stop() {
  pybind11::gil_scoped_release nogil;
  sqt::run = false;
  if (sqt::th.joinable()) sqt::th.join();
  return sqt::windows.load();
}
std::vector<int64_t> sq_stats() {
  return {sq::n_q.load(), sq::n_staged.load(), sq::n_full.load(), sq::n_stage_full.load(), sq::n_err.load(),
          sq::head.load(), sq::tail.load(), sqe::n_streams.load(), (int64_t)sqe::tripped.load(), sqt::windows.load(),
          nt::busy_ns.load(), nt::idle_y_ns.load(), nt::idle_o_ns.load(), nt::n.load(), sq::n_yield_waits.load(),
          sq::n_yield_calls.load(), 0};   // last slot: inline copies (mode removed), kept for the layout
}

void sstream_join() {
  pybind11::gil_scoped_release nogil;
  if (ss::T.on) { ss::T.th.join(); ss::T.on = false; }
}

// ---------------------------------------------------------------------------------------------
// Verifier-side Revealer loop (runs in its own spawned process, pinned): publish rev[i % N] =
// i << 32 | sched_key(vkey, i) at t0 + i*delta - lead, and timestamp the prover's completion reports
// rep[j % N] == j << 32 | s, counting lateness against t0 + j*delta (same layout as pcie.rev_words).
namespace rv {
inline uint32_t fm(uint32_t x) { x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16; return x; }
inline double now() { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
}
void revealer(int64_t base, int64_t N, int64_t vkey, double delta, double lead, double t0, double slack,
              double grace, int64_t n_stats, int64_t n_hist, double delta_min, int64_t dl_base, int64_t dl_n,
              int64_t due_base, int64_t ktab_base, int64_t ktab_n, int64_t fz_base, int64_t fz_n) {
  pybind11::gil_scoped_release nogil;
  const uint32_t* ktab = (const uint32_t*)ktab_base;   // keyed permutation schedule (--noise-perm), else fmix key
  volatile int64_t* a = (volatile int64_t*)base;
  volatile int64_t* rev = a;
  volatile int64_t* rep = a + N;
  volatile int64_t* st = a + 2 * N;
  volatile int64_t* hist = a + 2 * N + n_stats;
  const pid_t ppid = getppid();
  int64_t i = 0, j = 0, n = 0, late = 0, mx = 0, sm = 0, rl = 0, rmax = 0, late_adj = 0, mx_adj = 0;
  uint64_t it = 0;
  // actual reveal time per window: a window revealed late (verifier-side delay) gets its deadline
  // pushed to (actual reveal + lead), capped at CAP_ADJ s of extra time
  std::vector<double> rt(N, 0.0);
  const double CAP_ADJ = 0.5;
  // self-freeze detection: if this loop didn't run for > 5 ms, completions first seen afterwards may
  // have arrived any time since the previous poll; their adjusted lateness counts from that poll
  double last_poll = rv::now();
  int64_t frz = 0, frz_us = 0;
  // freeze log for the floor rule: one (start us, end us, credit us) triple per self-freeze, times from t0. The
  // credit is the link time the freeze cost: the gap minus what the windows already in flight covered
  // (saturating mode keeps `sat` windows revealed ahead, each delta_min long at the link maximum).
  volatile int64_t* fz = fz_base ? (volatile int64_t*)fz_base : nullptr;
  int64_t fzk = 0, fz_credit_us = 0;
  // catch-up-aware deadline: max(due, actual reveal + lead, previous deadline + delta_min), where
  // delta_min = window bytes at the physical link maximum (no prover catches up faster); published
  // per window (us from t0) for the verifier's ring check
  volatile int64_t* dl = dl_base ? (volatile int64_t*)dl_base : nullptr;
  // optional per-window due table: due[j] = t0 + D[j] us (unused: the verifier passes none); else t0 + j*delta
  const int64_t* D = due_base ? (const int64_t*)due_base : nullptr;
  auto due_of = [&](int64_t w) -> double { return (D && w < dl_n) ? t0 + D[w] * 1e-6 : t0 + w * delta; };
  double prev_dl = -1e30;
  // VARENA_SAT_LEAD=k (saturating noise): window i is revealed as soon as fewer than k windows are outstanding
  // (completion-driven, back to back at whatever the link gives); deadlines are the arrivals themselves, and the
  // verifier checks the achieved window rate against a per-host calibration instead of a schedule
  static const int64_t sat = getenv("VARENA_SAT_LEAD") ? atoll(getenv("VARENA_SAT_LEAD")) : 0;
  for (;;) {
    const double tp = rv::now();
    const double gap = tp - last_poll;
    const bool frozen = gap > 0.005;
    if (frozen) {
      frz++; frz_us += (int64_t)(gap * 1e6); st[10] = frz; st[11] = frz_us;
      const double cr = gap - (sat > 0 ? sat * delta_min : 0.0);
      if (cr > 0) {
        fz_credit_us += (int64_t)(cr * 1e6);
        if (fz && 3 * (fzk + 1) <= fz_n) {
          fz[3 * fzk] = (int64_t)((last_poll - t0) * 1e6);
          fz[3 * fzk + 1] = (int64_t)((tp - t0) * 1e6);
          fz[3 * fzk + 2] = (int64_t)(cr * 1e6);
          fzk++;
          st[12] = fzk;
        }
        st[13] = fz_credit_us;
      }
    }
    const double earliest = frozen ? last_poll : tp;
    last_poll = tp;
    while ((rep[j % N] >> 32) == j) {          // stamp completions in order
      const double due = sat ? rv::now() : due_of(j);
      if (sat || due - t0 >= grace) {
        const double lat = rv::now() - due;
        const int64_t us = (int64_t)(lat * 1e6);
        n++; late += lat > slack; mx = us > mx ? us : mx; sm += us;
        double dlj = due;
        if (rt[j % N] > 0 && rt[j % N] + lead > dlj) dlj = rt[j % N] + lead;
        if (prev_dl + delta_min > dlj) dlj = prev_dl + delta_min;
        if (dlj > due + CAP_ADJ) dlj = due + CAP_ADJ;
        prev_dl = dlj;
        const double arr = (earliest - due) < lat ? earliest : due + lat;      // freeze-adjusted arrival
        const double la = arr - dlj;
        late_adj += la > slack;
        const int64_t ua = (int64_t)(la * 1e6);
        mx_adj = ua > mx_adj ? ua : mx_adj;
        st[8] = late_adj; st[9] = mx_adj;
        if (dl && j < dl_n) dl[j] = (int64_t)((dlj - t0) * 1e6);
        int64_t b = us <= 0 ? 0 : 1 + (us - 1) / 1000;
        if (b > n_hist - 1) b = n_hist - 1;
        hist[b] = hist[b] + 1;
        st[0] = n; st[1] = late; st[2] = mx; st[3] = sm;
      }
      j++;
      st[4] = j;
    }
    const double t = sat ? 0.0 : due_of(i) - lead;
    const double tn = rv::now();
    if (sat ? (i < j + sat) : (tn >= t)) {
      const uint32_t s = ktab_n > 0 ? ktab[i % ktab_n] : rv::fm((uint32_t)vkey ^ rv::fm((uint32_t)i ^ 0x2545F491u));
      rev[i % N] = (i << 32) | (int64_t)s;
      rt[i % N] = rv::now();
      const int64_t rd = (int64_t)((rt[i % N] - t) * 1e6);
      if (rd > 5000) { rl++; st[5] = rl; }
      if (rd > rmax) { rmax = rd; st[6] = rmax; }
      i++;
      continue;
    }
    if (sat) std::this_thread::sleep_for(std::chrono::microseconds(20));
    else if (t - tn > 0.0003) std::this_thread::sleep_for(std::chrono::microseconds(100));
    if ((++it & 0xFFFF) == 0 && getppid() != ppid) return;
  }
}

pybind11::tuple rtt(int64_t fd, int64_t id, int64_t kind, int64_t nonce) {
  Req r{(uint32_t)id, (uint32_t)kind, (uint32_t)nonce, 0};
  Resp h{};
  std::string payload;
  int64_t ns = -1;
  {
    pybind11::gil_scoped_release nogil;
    auto t0 = std::chrono::steady_clock::now();
    if (wr((int)fd, &r, sizeof r) && rd((int)fd, &h, sizeof h)) {
      payload.resize(h.bytes);
      if (h.bytes == 0 || rd((int)fd, &payload[0], h.bytes))
        ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count();
    }
  }
  return pybind11::make_tuple(ns, h.id, h.kind, h.win, h.nonce_eff, (int64_t)h.serve_ns,
                              pybind11::bytes(payload),
                              pybind11::make_tuple((int64_t)h.t_recv, (int64_t)h.t_launch, (int64_t)h.t_done));
}
"""

_mod = None


def mod():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        _mod = load_inline(name="varena_native", cpp_sources=_SRC, functions=["start", "rtt", "sstream_start", "sstream_join", "revealer",
                                                                 "sq_submit", "sq_active", "sq_selftest_start", "sq_selftest_stop", "sq_stats"],
                           with_cuda=True, extra_ldflags=["-lcudart", "-lcuda", "-ldl"],
                           verbose=bool(os.environ.get("VARENA_NATIVE_VERBOSE")))
    return _mod


class ChaseGraph:
    """Probe-chase challenge as a CUDA graph; final chain states (uint64 [K]) go to pinned host."""

    def __init__(self, launch, K, stream, extra=0):
        import torch
        self.bytes = 8 * (K + extra)             # extra: uint64 words the kernel appends after the K chain states
        self.host = torch.empty(K + extra, dtype=torch.int64).pin_memory()
        self.nonce = torch.zeros(2, dtype=torch.int32).pin_memory()   # {nonce, head window}
        with torch.cuda.stream(stream):
            launch(self.host, self.nonce)
        stream.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g, stream=stream):
            launch(self.host, self.nonce)
        stream.synchronize()
        self.ctl = None

    def exec_ptr(self):
        return int(self.g.raw_cuda_graph_exec())


def start_server(path, graphs, last_arr, device=0, busy=None, tabupd=None):
    m = mod()
    m.start(path, [g.exec_ptr() for g in graphs], [g.nonce.data_ptr() for g in graphs],
            [g.host.data_ptr() for g in graphs], [g.bytes for g in graphs],
            last_arr.data_ptr(), device,
            [g.ctl.data_ptr() if getattr(g, "ctl", None) is not None else 0 for g in graphs],
            busy.data_ptr() if busy is not None else 0, tabupd.data_ptr() if tabupd is not None else 0)
