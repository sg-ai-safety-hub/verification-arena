#include "cuda.h"
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
// A worker is one hardware work queue (FIFO). Without MOCK_CONNECTIONS every stream has its own; with
// MOCK_CONNECTIONS=k streams share k workers round-robin, like CUDA_DEVICE_MAX_CONNECTIONS: an op blocked at the head
// of a worker (a stream wait) also holds every later op of the other streams mapped to it (false dependency).
struct Worker {
  std::mutex mu; std::condition_variable cv; std::deque<std::function<void()>> q; std::thread th;
  void push(std::function<void()> f) { { std::lock_guard<std::mutex> g(mu); q.push_back(std::move(f)); } cv.notify_one(); }
  Worker() { th = std::thread([this] { for (;;) { std::function<void()> f; { std::unique_lock<std::mutex> g(mu);
      cv.wait(g, [this] { return !q.empty(); }); f = std::move(q.front()); q.pop_front(); } f(); } }); th.detach(); }
};
static Worker* worker_for_new_stream() {
  static std::mutex m; static std::vector<Worker*> ws; static int next = 0;
  static const int k = getenv("MOCK_CONNECTIONS") ? atoi(getenv("MOCK_CONNECTIONS")) : 0;
  std::lock_guard<std::mutex> g(m);
  if (k <= 0) return new Worker;
  if ((int)ws.size() < k) { ws.push_back(new Worker); return ws.back(); }
  return ws[next++ % k];
}
struct MStream {
  Worker* w = worker_for_new_stream();
  std::atomic<long> copies{0};
  void push(std::function<void()> f) { w->push(std::move(f)); }
};
struct MEvent { std::atomic<long> rec{0}, done{0}; std::atomic<int64_t> t_ns{0}; };
static int64_t mono() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static MStream* legacy() { static MStream* s = new MStream; return s; }
static MStream* S(cudaStream_t s) { return s ? s : legacy(); }
static thread_local cudaError_t last = cudaSuccess;
static std::mutex rmu; static std::map<uintptr_t, std::pair<size_t, int>> regions;   // base -> (size, type)
static void reg(void* p, size_t n, int t) { std::lock_guard<std::mutex> g(rmu); regions[(uintptr_t)p] = {n, t}; }
static int type_of(const void* p) {
  std::lock_guard<std::mutex> g(rmu);
  auto it = regions.upper_bound((uintptr_t)p);
  if (it == regions.begin()) return 0;
  --it;
  return (uintptr_t)p < it->first + it->second.first ? it->second.second : 0;
}
extern "C" {
cudaError_t cudaGetLastError(void) { cudaError_t e = last; last = cudaSuccess; return e; }
cudaError_t cudaSetDevice(int) { return 0; }
cudaError_t cudaGetDevice(int* d) { *d = 0; return 0; }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { *e = new MEvent; return 0; }
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s) {
  long g = ++e->rec; S(s)->push([e, g] { e->t_ns = mono(); long d = e->done.load(); while (d < g && !e->done.compare_exchange_weak(d, g)) {} });
  return 0;
}
cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = new MEvent; return 0; }
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
  if (a->done.load() < a->rec.load() || b->done.load() < b->rec.load() || !a->rec || !b->rec) { last = cudaErrorNotReady; return cudaErrorNotReady; }
  *ms = (float)((b->t_ns.load() - a->t_ns.load()) * 1e-6); return 0;
}
cudaError_t cudaEventQuery(cudaEvent_t e) { if (e->done.load() >= e->rec.load()) return 0; last = cudaErrorNotReady; return cudaErrorNotReady; }
cudaError_t cudaEventSynchronize(cudaEvent_t e) { long g = e->rec.load(); while (e->done.load() < g) std::this_thread::yield(); return 0; }
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* s, unsigned, int) { *s = new MStream; return 0; }
cudaError_t cudaStreamGetPriority(cudaStream_t, int* p) { *p = 0; return 0; }
cudaError_t cudaDeviceGetStreamPriorityRange(int* a, int* b) { *a = 0; *b = -1; return 0; }
cudaError_t cudaStreamSynchronize(cudaStream_t s) {
  std::atomic<bool> d{false}; S(s)->push([&d] { d = true; }); while (!d) std::this_thread::yield(); return 0;
}
cudaError_t cudaStreamQuery(cudaStream_t s) {
  auto* e = new MEvent; cudaEventRecord(e, s);
  std::this_thread::sleep_for(std::chrono::microseconds(200));
  return cudaEventQuery(e);
}
cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned) {
  long g = e->rec.load(); S(s)->push([e, g] { while (e->done.load() < g) std::this_thread::yield(); }); return 0;
}
cudaError_t cudaStreamIsCapturing(cudaStream_t, enum cudaStreamCaptureStatus* c) { *c = cudaStreamCaptureStatusNone; return 0; }
cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, enum cudaMemcpyKind, cudaStream_t st) {
  MStream* m = S(st); m->copies++;
  if (type_of(s) == 0) {     // pageable: staged at call time, as the real runtime does
    auto* b = (uint8_t*)malloc(n); memcpy(b, s, n);
    m->push([d, b, n] { memcpy(d, b, n); free(b); });
  } else m->push([d, s, n] { memcpy(d, s, n); });
  return 0;
}
cudaError_t cudaMemcpy(void* d, const void* s, size_t n, enum cudaMemcpyKind k) {
  cudaMemcpyAsync(d, s, n, k, nullptr); return cudaStreamSynchronize(nullptr);
}
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes* a, const void* p) {
  a->type = (enum cudaMemoryType)type_of(p); a->device = 0; return 0;
}
cudaError_t cudaHostAlloc(void** p, size_t n, unsigned) { *p = malloc(n); reg(*p, n, 1); return 0; }
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { return 0; }
cudaError_t cudaStreamBeginCapture(cudaStream_t, int) { return 0; }
cudaError_t cudaStreamEndCapture(cudaStream_t, void** g) { if (g) *g = nullptr; return 0; }
void* mockDeviceAlloc(size_t n) { void* p = calloc(1, n); reg(p, n, 2); return p; }
void mockKernel(cudaStream_t s, void (*fn)(void*), void* arg) { S(s)->push([fn, arg] { fn(arg); }); }
long mockCopiesOn(cudaStream_t s) { return S(s)->copies.load(); }
CUresult cuInit(unsigned) { return 0; }
CUresult cuMemAlloc(CUdeviceptr* p, size_t n) { *p = (CUdeviceptr)mockDeviceAlloc(n); return 0; }
CUresult cuMemsetD8(CUdeviceptr p, unsigned char v, size_t n) { memset((void*)p, v, n); return 0; }
CUresult cuStreamWaitValue64(CUstream s, CUdeviceptr f, cuuint64_t v, unsigned) {
  S(s)->push([f, v] { while (__atomic_load_n((uint64_t*)f, __ATOMIC_ACQUIRE) < v) std::this_thread::yield(); }); return 0;
}
CUresult cuStreamWriteValue64(CUstream s, CUdeviceptr f, cuuint64_t v, unsigned) {
  S(s)->push([f, v] { __atomic_store_n((uint64_t*)f, v, __ATOMIC_RELEASE); }); return 0;
}
}
