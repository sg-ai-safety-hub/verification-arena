// CPU test of the single H2D queue + h2d_route.so interposer against the mock runtime.
//   bash schemes/memory_link_exhaustion/tools/sq_mock/run.sh   (no GPU, no torch: gcc only)
#include "native_src.cpp"   // extracted from gpu/native.py by run.sh
#include <random>
static std::atomic<int> errs{0};
static std::atomic<bool> stop{false};
#define CHECK(c, ...) do { if (!(c)) { errs++; fprintf(stderr, "FAIL %s:%d ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static void noise_sleep(void*) { std::this_thread::sleep_for(std::chrono::microseconds(300)); }
static void streamer() {
  sq::own = true;
  long inflight = 0; std::atomic<long> done{0};
  while (!stop) {
    // keep 2 "noise windows" queued on the streamer's stream, drain prover copies between, as complete()/issue() do
    ss::sq_drain();
    mockKernel(ss::T.sh2d, noise_sleep, nullptr);
    cudaStreamSynchronize(ss::T.sh2d);   // crude pacing; the drain still runs at every window boundary
  }
  ss::sq_off();
}

struct Ver { uint8_t* d; size_t n; uint32_t seed; int sleep_us; };
static uint8_t pat(uint32_t seed, size_t i) { return (uint8_t)(fmix(seed * 2654435761u + (uint32_t)i)); }
static void verify(void* a) {
  Ver* v = (Ver*)a;
  if (v->sleep_us) std::this_thread::sleep_for(std::chrono::microseconds(v->sleep_us));
  for (size_t i = 0; i < v->n; i++) if (v->d[i] != pat(v->seed, i)) { errs++; fprintf(stderr, "FAIL data seed=%u i=%zu\n", v->seed, i); break; }
  delete v;
}

static void worker(int id, int iters) {
  cudaStream_t c; cudaStreamCreateWithPriority(&c, cudaStreamNonBlocking, 0);
  uint8_t* d = (uint8_t*)mockDeviceAlloc(1 << 20);
  std::mt19937 rng(id);
  uint8_t* pinned; cudaHostAlloc((void**)&pinned, (size_t)iters * 8192, 0);
  std::vector<uint8_t> pageable(256 << 10);
  for (int it = 0; it < iters; it++) {
    const uint32_t seed = (uint32_t)(id * 1000003 + it);
    const bool pg = rng() % 2;
    const size_t n = pg ? 1 + rng() % (256 << 10) : 1 + rng() % 8192;
    uint8_t* src = pg ? pageable.data() : pinned + (size_t)it * 8192;
    for (size_t i = 0; i < n; i++) src[i] = pat(seed, i);
    // cudaMemcpy runs on the legacy stream, unordered with c (as in CUDA): the test orders it by syncing c first
    const bool sync = rng() % 8 == 0;
    if (sync) cudaStreamSynchronize(c);
    const cudaError_t e = sync ? cudaMemcpy(d, src, n, cudaMemcpyHostToDevice)
                               : cudaMemcpyAsync(d, src, n, cudaMemcpyHostToDevice, c);
    CHECK(e == cudaSuccess, "copy rc %d", e);
    CHECK(cudaGetLastError() == cudaSuccess, "sticky last error after routed copy");
    if (pg) memset(src, 0xEE, n);                        // caller reuses its pageable buffer at once
    // verify on the caller's stream after a random delay: the next copy must not land before this has read
    mockKernel(c, verify, new Ver{d, n, seed, (int)(rng() % 200)});
  }
  cudaStreamSynchronize(c);
  CHECK(mockCopiesOn(c) == 0 || sq::n_stage_full.load() > 0, "worker %d: %ld copies bypassed the queue on the caller stream", id, mockCopiesOn(c));
}

int main() {
  setenv("VARENA_SQ_STAGE_MIB", "1", 1);
  setenv("VARENA_SQ_GUARD_READY_S", "1", 1);             // short so the guard test is quick (normal worker waits are ms)                 // small staging: wrap-around and staging-full fallbacks
  typedef void (*stats_t)(uint64_t*);
  auto stats = (stats_t)dlsym(RTLD_DEFAULT, "varena_h2d_stats");
  CHECK(stats != nullptr, "h2d_route.so not preloaded");
  cudaStreamCreateWithPriority(&ss::T.sh2d, cudaStreamNonBlocking, 0);
  cudaStreamCreateWithPriority(&ss::T.sk, cudaStreamNonBlocking, 0);
  cudaStreamCreateWithPriority(&ss::T.sd2h, cudaStreamNonBlocking, 0);
  ss::T.dev = 0;
  // before the queue is on: straight through
  { uint8_t* d = (uint8_t*)mockDeviceAlloc(64); uint8_t h[64] = {1};
    cudaStream_t c; cudaStreamCreateWithPriority(&c, cudaStreamNonBlocking, 0);
    cudaMemcpyAsync(d, h, 64, cudaMemcpyHostToDevice, c); cudaStreamSynchronize(c);
    CHECK(mockCopiesOn(c) == 1 && d[0] == 1, "pass-through before start"); }
  ss::sq_on();
  CHECK(sq::active.load(), "queue not active");
  std::thread st(streamer);

  // WAR: a kernel still reading the destination must see the old bytes
  {
    cudaStream_t c; cudaStreamCreateWithPriority(&c, cudaStreamNonBlocking, 0);
    uint8_t* d = (uint8_t*)mockDeviceAlloc(4096);
    uint8_t* a; cudaHostAlloc((void**)&a, 4096, 0);
    for (int i = 0; i < 4096; i++) a[i] = pat(1, i);
    cudaMemcpyAsync(d, a, 4096, cudaMemcpyHostToDevice, c);
    mockKernel(c, verify, new Ver{d, 4096, 1, 50000});    // 50 ms reader of the first contents
    uint8_t* b; cudaHostAlloc((void**)&b, 4096, 0);
    for (int i = 0; i < 4096; i++) b[i] = pat(2, i);
    cudaMemcpyAsync(d, b, 4096, cudaMemcpyHostToDevice, c);
    mockKernel(c, verify, new Ver{d, 4096, 2, 0});
    cudaStreamSynchronize(c);
    CHECK(mockCopiesOn(c) == 0, "WAR test copies not queued");
  }
  // the legacy default stream is not routed
  { uint8_t* d = (uint8_t*)mockDeviceAlloc(64); uint8_t* h; cudaHostAlloc((void**)&h, 64, 0); h[0] = 9;
    long before = mockCopiesOn(nullptr);
    cudaMemcpyAsync(d, h, 64, cudaMemcpyHostToDevice, nullptr); cudaStreamSynchronize(nullptr);
    CHECK(mockCopiesOn(nullptr) == before + 1 && d[0] == 9, "legacy-stream copy was routed"); }
  // the streamer's own streams are never queued (noise sub-copies)
  { uint8_t* d = (uint8_t*)mockDeviceAlloc(64); uint8_t* h; cudaHostAlloc((void**)&h, 64, 0);
    long before = mockCopiesOn(ss::T.sh2d);
    cudaMemcpyAsync(d, h, 64, cudaMemcpyHostToDevice, ss::T.sk);
    CHECK(mockCopiesOn(ss::T.sk) == 1, "own stream copy was queued"); (void)before; }

  // a kernel spinning in the mock blocks its whole worker, so with shared workers (MOCK_CONNECTIONS > 0) a host-gated
  // kernel blocks unrelated streams for any design: the gated tests run with private workers only
  const bool private_hw = !getenv("MOCK_CONNECTIONS") || atoi(getenv("MOCK_CONNECTIONS")) <= 0;
  // head-of-line: stream A's earlier work waits for the host (like a matmul spinning on VARENA_MM_GATE until a challenge
  // ends); a copy queued on A must not hold a copy on B, which the host needs before it releases A
  if (private_hw) {
    static std::atomic<bool> gate{false};
    cudaStream_t A, B; cudaStreamCreateWithPriority(&A, cudaStreamNonBlocking, 0); cudaStreamCreateWithPriority(&B, cudaStreamNonBlocking, 0);
    uint8_t* da = (uint8_t*)mockDeviceAlloc(4096); uint8_t* db = (uint8_t*)mockDeviceAlloc(4096);
    uint8_t* ha; cudaHostAlloc((void**)&ha, 4096, 0); uint8_t* hb; cudaHostAlloc((void**)&hb, 4096, 0);
    for (int i = 0; i < 4096; i++) { ha[i] = pat(11, i); hb[i] = pat(12, i); }
    mockKernel(A, [](void*) { while (!gate.load()) std::this_thread::sleep_for(std::chrono::microseconds(50)); }, nullptr);
    cudaMemcpyAsync(da, ha, 4096, cudaMemcpyHostToDevice, A);
    cudaMemcpyAsync(db, hb, 4096, cudaMemcpyHostToDevice, B);
    std::atomic<bool> bdone{false};
    std::thread w([&] { cudaStreamSynchronize(B); bdone = true; });
    const auto t0 = std::chrono::steady_clock::now();
    while (!bdone && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(3)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(bdone.load(), "head-of-line: copy on B held behind a blocked stream A");
    gate = true; w.join(); cudaStreamSynchronize(A);
    mockKernel(A, verify, new Ver{da, 4096, 11, 0}); mockKernel(B, verify, new Ver{db, 4096, 12, 0});
    cudaStreamSynchronize(A); cudaStreamSynchronize(B);
  }
  // graph capture in progress anywhere: copies go direct (interposer counts them)
  { cudaStream_t c; cudaStreamCreateWithPriority(&c, cudaStreamNonBlocking, 0);
    uint8_t* d = (uint8_t*)mockDeviceAlloc(64); uint8_t* h; cudaHostAlloc((void**)&h, 64, 0);
    cudaStream_t cap; cudaStreamCreateWithPriority(&cap, cudaStreamNonBlocking, 0);
    cudaStreamBeginCapture(cap, 0);
    cudaMemcpyAsync(d, h, 64, cudaMemcpyHostToDevice, c);
    void* gr; cudaStreamEndCapture(cap, &gr);
    cudaStreamSynchronize(c);
    CHECK(mockCopiesOn(c) == 1, "copy during a capture was routed");
    cudaMemcpyAsync(d, h, 64, cudaMemcpyHostToDevice, c); cudaStreamSynchronize(c);
    CHECK(mockCopiesOn(c) == 1, "copy after the capture not routed"); }

  // many caller streams from many threads (more streams than hardware queues)
  std::vector<std::thread> ws;
  for (int i = 0; i < 8; i++) ws.emplace_back(worker, i, 1200);
  for (auto& w : ws) w.join();
  // guard (event mode): a stream whose earlier work never finishes trips it; routing then turns off, copies go direct
  if (private_hw) {
    static std::atomic<bool> gate2{false};
    cudaStream_t A; cudaStreamCreateWithPriority(&A, cudaStreamNonBlocking, 0);
    uint8_t* d = (uint8_t*)mockDeviceAlloc(64); uint8_t* h; cudaHostAlloc((void**)&h, 64, 0);
    mockKernel(A, [](void*) { while (!gate2.load()) std::this_thread::sleep_for(std::chrono::microseconds(50)); }, nullptr);
    cudaMemcpyAsync(d, h, 64, cudaMemcpyHostToDevice, A);
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));      // VARENA_SQ_GUARD_READY_S=1 in main()
    CHECK(sqe::tripped.load(), "guard did not trip");
    cudaStream_t c; cudaStreamCreateWithPriority(&c, cudaStreamNonBlocking, 0);
    cudaMemcpyAsync(d, h, 64, cudaMemcpyHostToDevice, c); cudaStreamSynchronize(c);
    CHECK(mockCopiesOn(c) == 1, "copy after the guard tripped was still routed");
    gate2 = true; cudaStreamSynchronize(A);
  }
  stop = true; st.join();
  // after stop: straight through again
  { uint8_t* d = (uint8_t*)mockDeviceAlloc(64); uint8_t h[64] = {7};
    cudaStream_t c; cudaStreamCreateWithPriority(&c, cudaStreamNonBlocking, 0);
    cudaMemcpyAsync(d, h, 64, cudaMemcpyHostToDevice, c); cudaStreamSynchronize(c);
    CHECK(mockCopiesOn(c) == 1 && d[0] == 7, "pass-through after stop"); }
  // noise timing accounting on the self-test driver (queue off, small windows)
  { cudaStream_t ns; cudaStreamCreateWithPriority(&ns, cudaStreamNonBlocking, 0);
    const int64_t Wt = 1 << 20;
    uint8_t* pool; cudaHostAlloc((void**)&pool, Wt, 0); uint8_t* ring = (uint8_t*)mockDeviceAlloc(Wt);
    sq_selftest_start((int64_t)ns, (int64_t)pool, (int64_t)ring, Wt, 0, false, 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const int64_t w = sq_selftest_stop();
    printf("noise timing: windows %ld timed %ld busy %.1f ms idle(yield) %.1f ms idle(other) %.1f ms\n", (long)w,
           (long)nt::n.load(), nt::busy_ns.load() * 1e-6, nt::idle_y_ns.load() * 1e-6, nt::idle_o_ns.load() * 1e-6);
    if (nt::on) CHECK(nt::n.load() > 10 && nt::busy_ns.load() > 0, "noise timing not accounted");
    else CHECK(w > 10 && nt::n.load() == 0, "timing bisect mode should run the noise without accounting"); }
  uint64_t s4[4]; stats(s4);
  printf("interposer: queued %lu passed %lu fallback %lu over-max %lu; native: queued %ld staged %ld full %ld stage-full %ld err %ld\n",
         s4[0], s4[1], s4[2], s4[3], (long)sq::n_q.load(), (long)sq::n_staged.load(), (long)sq::n_full.load(),
         (long)sq::n_stage_full.load(), (long)sq::n_err.load());
  CHECK(s4[0] > 7000, "too few queued (cudaMemcpy on the legacy stream goes direct)");
  printf("yield waits %ld\n", (long)sq::n_yield_waits.load());
  CHECK(sq::n_err.load() == 0, "queue errors");
  printf(errs ? "FAILED (%d)\n" : "PASS\n", errs.load());
  return errs ? 1 : 0;
}
