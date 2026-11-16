// LD_PRELOAD interposer: route every host-to-device cudaMemcpyAsync / cudaMemcpy issued in the prover process
// (torch's copy_ / .to() and anything else on the CUDA runtime) through the single H2D queue of the noise streamer
// (gpu/native.py, VARENA_SINGLE_QUEUE=1). Without it ~85% of vLLM's small H2D copies (plain tensor copies) go to the
// copy engines on their own and starve behind saturating inbound noise (seen: 37 engine stalls of 8 to 45 s).
//
//   gcc -O2 -shared -fPIC -o h2d_route.so h2d_route.c -ldl
//   VARENA_SINGLE_QUEUE=1 LD_PRELOAD=$PWD/h2d_route.so <prover command>
//
// This file holds no queue logic. While the streamer runs it registers its router (varena_sq_route, C linkage in the
// native module) here through varena_h2d_set_route; each H2D call of at most H2D_ROUTE_MAX bytes (default 1 MiB) is
// offered to it. The router returns 0 (queued: the copy is issued by the streamer and the caller's stream waits for
// it on the GPU), 1 (not for the queue: streamer inactive, the streamer's own thread or streams, not host->device,
// stream being captured) or -1 (eligible but not queued: queue or staging full, foreign device; counted, and with
// H2D_ROUTE_LOG=path logged with the caller's stack). Anything not queued goes to the real runtime unchanged.
// Without a registered router (verifier, baseline, before the streamer starts) every call passes straight through.
//
// At exit a one-line summary goes to stderr: queued, passed (not eligible), fallback.
// The CUDA symbols are unversioned on purpose: an unversioned definition binds references to
// cudaMemcpyAsync@libcudart.so.12 and .so.13 alike; the real ones are looked up by version, then unversioned.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef int cudaError_t;
typedef cudaError_t (*memcpy_async_t)(void*, const void*, size_t, int, void*);
typedef cudaError_t (*memcpy_t)(void*, const void*, size_t, int);
typedef int (*route_t)(void* dst, const void* src, size_t n, int kind, void* stream, int sync);

static route_t route = 0;
static int capturing = 0;                 // CUDA graph captures in progress in this process (any thread)
static uint64_t n_capture_skip = 0;
static size_t maxb = 0;
static uint64_t n_routed = 0, n_passed = 0, n_fallback = 0, n_large = 0;
static FILE* logf = 0;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

#define REAL(name, T)                                                        \
  static T r_##name = 0;                                                     \
  if (!r_##name) r_##name = (T)dlvsym(RTLD_NEXT, #name, "libcudart.so.13"); \
  if (!r_##name) r_##name = (T)dlvsym(RTLD_NEXT, #name, "libcudart.so.12"); \
  if (!r_##name) r_##name = (T)dlsym(RTLD_NEXT, #name);

static size_t max_bytes(void) {
  if (!maxb) {
    const char* m = getenv("H2D_ROUTE_MAX");
    maxb = m ? strtoull(m, 0, 10) : (size_t)1 << 20;
  }
  return maxb;
}

// called by the native streamer when the queue goes on (router) and off (null)
void varena_h2d_set_route(void* fn) {
  __atomic_store_n(&route, (route_t)fn, __ATOMIC_RELEASE);
  if (fn && !logf) {
    const char* p = getenv("H2D_ROUTE_LOG");
    if (p) logf = fopen(p, "a");
  }
}

// counters for tests and the watchdog: routed, passed, fallback, large (over H2D_ROUTE_MAX, not offered)
void varena_h2d_stats(uint64_t* out4) {
  out4[0] = __atomic_load_n(&n_routed, __ATOMIC_RELAXED);
  out4[1] = __atomic_load_n(&n_passed, __ATOMIC_RELAXED);
  out4[2] = __atomic_load_n(&n_fallback, __ATOMIC_RELAXED);
  out4[3] = __atomic_load_n(&n_large, __ATOMIC_RELAXED);
}
int varena_h2d_capturing(void) { return __atomic_load_n(&capturing, __ATOMIC_ACQUIRE); }
uint64_t varena_h2d_capture_skips(void) { return __atomic_load_n(&n_capture_skip, __ATOMIC_RELAXED); }

static void log_fallback(const char* what, size_t n, int kind, void* stream) {
  if (!logf) return;
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  void* bt[10]; int k = backtrace(bt, 10);
  pthread_mutex_lock(&mu);
  fprintf(logf, "%ld.%06ld FALLBACK %s n=%zu kind=%d stream=%p", (long)ts.tv_sec, ts.tv_nsec / 1000, what, n, kind,
          stream);
  for (int i = 2; i < k; i++) {
    Dl_info di;
    if (dladdr(bt[i], &di) && di.dli_fname) fprintf(logf, " | %s:%s", di.dli_fname, di.dli_sname ? di.dli_sname : "?");
  }
  fputc('\n', logf);
  fflush(logf);
  pthread_mutex_unlock(&mu);
}

// 1: handled (queued), 0: call the real runtime
static int offer(const char* what, void* dst, const void* src, size_t n, int kind, void* stream, int sync) {
  route_t r = __atomic_load_n(&route, __ATOMIC_ACQUIRE);
  if (!r || !(kind == 1 || kind == 4)) return 0;     // cudaMemcpyHostToDevice, cudaMemcpyDefault
  // while any graph is being captured (vLLM warm-up), nothing is routed: the router's event calls on other streams
  // must not meet a global-mode capture, and the capture must not record a wait on our events
  if (__atomic_load_n(&capturing, __ATOMIC_ACQUIRE) > 0) { __atomic_add_fetch(&n_capture_skip, 1, __ATOMIC_RELAXED); return 0; }
  if (n > max_bytes()) { __atomic_add_fetch(&n_large, 1, __ATOMIC_RELAXED); return 0; }
  const int rc = r(dst, src, n, kind, stream, sync);
  if (rc == 0) { __atomic_add_fetch(&n_routed, 1, __ATOMIC_RELAXED); return 1; }
  if (rc > 0) { __atomic_add_fetch(&n_passed, 1, __ATOMIC_RELAXED); return 0; }
  __atomic_add_fetch(&n_fallback, 1, __ATOMIC_RELAXED);
  log_fallback(what, n, kind, stream);
  return 0;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, int kind, void* stream) {
  REAL(cudaMemcpyAsync, memcpy_async_t);
  if (offer("cudaMemcpyAsync", dst, src, n, kind, stream, 0)) return 0;
  return r_cudaMemcpyAsync(dst, src, n, kind, stream);
}

typedef cudaError_t (*begin_cap_t)(void*, int);
typedef cudaError_t (*end_cap_t)(void*, void**);
cudaError_t cudaStreamBeginCapture(void* stream, int mode) {
  REAL(cudaStreamBeginCapture, begin_cap_t);
  __atomic_add_fetch(&capturing, 1, __ATOMIC_ACQ_REL);
  const cudaError_t e = r_cudaStreamBeginCapture(stream, mode);
  if (e != 0) __atomic_sub_fetch(&capturing, 1, __ATOMIC_ACQ_REL);
  return e;
}
cudaError_t cudaStreamEndCapture(void* stream, void** graph) {
  REAL(cudaStreamEndCapture, end_cap_t);
  const cudaError_t e = r_cudaStreamEndCapture(stream, graph);
  if (__atomic_load_n(&capturing, __ATOMIC_ACQUIRE) > 0) __atomic_sub_fetch(&capturing, 1, __ATOMIC_ACQ_REL);
  return e;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, int kind) {
  REAL(cudaMemcpy, memcpy_t);
  if (offer("cudaMemcpy", dst, src, n, kind, 0, 1)) return 0;
  return r_cudaMemcpy(dst, src, n, kind);
}

__attribute__((destructor)) static void summary(void) {
  if (!n_routed && !n_fallback) return;
  fprintf(stderr, "[h2d_route] queued %llu passed %llu fallback %llu over-max %llu during-capture %llu\n",
          (unsigned long long)n_routed, (unsigned long long)n_passed, (unsigned long long)n_fallback,
          (unsigned long long)n_large, (unsigned long long)n_capture_skip);
}
