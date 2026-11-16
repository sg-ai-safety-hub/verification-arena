// Stall diagnosis: LD_PRELOAD interposer on libcudart.so.13 recording the last CUDA runtime calls that enqueue
// work (memcpy, memset, kernel launch, graph launch, event record, stream wait) with their stream, in a ring
// buffer. vshim_dump(stream, n, buf, len) prints the last n entries for one stream (stream = -1: all).
//   gcc -O2 -shared -fPIC -o vshim.so vshim.c -ldl -Wl,--version-script=vshim.map
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef struct { unsigned x, y, z; } dim3;
typedef struct { dim3 gridDim, blockDim; size_t dynamicSmemBytes; cudaStream_t stream; void* attrs; unsigned numAttrs; } cfgEx;
struct ptrAttr { int type; int device; void* dp; void* hp; char pad[512]; };

enum { F_MEMCPY = 1, F_MEMSET, F_LAUNCH, F_LAUNCHEX, F_GRAPH, F_EVREC, F_WAIT, F_SSYNC, F_ESYNC };
static const char* FN[] = {"?", "memcpyAsync", "memsetAsync", "launchKernel", "launchKernelEx", "graphLaunch",
                           "eventRecord", "streamWaitEvent", "streamSynchronize", "eventSynchronize"};
typedef struct { uint64_t t; uint32_t fn, tid; void* s; void* a; void* b; uint64_t n; int kind, mt; void* ev; } Ent;
#define NR (1 << 17)
static Ent R[NR];
static volatile uint64_t H = 0;
#define NE 4096
static void* EV_K[NE]; static void* EV_S[NE];        // event -> stream of its last record

#define REAL(name, T) static T r_##name; if (!r_##name) r_##name = (T)dlvsym(RTLD_NEXT, #name, "libcudart.so.13"); \
                      if (!r_##name) r_##name = (T)dlsym(RTLD_NEXT, #name);
static uint64_t now() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1000000000ull + ts.tv_nsec; }
static Ent* rec(uint32_t fn, void* s, void* a, void* b, uint64_t n, int kind, int mt) {
  uint64_t i = __sync_fetch_and_add(&H, 1) % NR;
  Ent* e = &R[i];
  e->t = now(); e->fn = fn; e->tid = (uint32_t)syscall(SYS_gettid); e->s = s; e->a = a; e->b = b; e->n = n;
  e->kind = kind; e->mt = mt; e->ev = 0;
  return e;
}
// completion events for work on the legacy default stream (stream 0): recorded right after the op, queried at dump
#define NEV 8192
static void* EVP[NEV]; static volatile uint64_t EVH = 0;
static void track(Ent* e, void* st) {
  if (st != 0) return;
  typedef cudaError_t (*FC)(void**, unsigned);
  typedef cudaError_t (*FR)(void*, void*);
  typedef cudaError_t (*FI)(void*, int*);
  REAL(cudaEventCreateWithFlags, FC); REAL(cudaEventRecord, FR); REAL(cudaStreamIsCapturing, FI);
  int cap = 0;
  if (r_cudaStreamIsCapturing(st, &cap) != 0 || cap) return;
  uint64_t k = __sync_fetch_and_add(&EVH, 1) % NEV;
  if (!EVP[k] && r_cudaEventCreateWithFlags(&EVP[k], 2) != 0) { EVP[k] = 0; return; }
  if (r_cudaEventRecord(EVP[k], st) == 0) e->ev = EVP[k];
}
static int memtype(const void* p) {
  typedef cudaError_t (*F)(struct ptrAttr*, const void*);
  REAL(cudaPointerGetAttributes, F);
  struct ptrAttr a; memset(&a, 0, sizeof a);
  if (r_cudaPointerGetAttributes(&a, p) != 0) return -1;
  return a.type;   // 0 unregistered (pageable), 1 host (pinned), 2 device, 3 managed
}

cudaError_t cudaMemcpyAsync(void* d, const void* s_, size_t n, int kind, cudaStream_t st) {
  typedef cudaError_t (*F)(void*, const void*, size_t, int, cudaStream_t);
  REAL(cudaMemcpyAsync, F);
  int mt = kind == 1 ? memtype(s_) : kind == 2 ? memtype(d) : -2;   // 1 H2D: host side = src; 2 D2H: dst
  Ent* e = rec(F_MEMCPY, st, d, (void*)s_, n, kind, mt);
  cudaError_t r = r_cudaMemcpyAsync(d, s_, n, kind, st);
  track(e, st);
  return r;
}
cudaError_t cudaMemsetAsync(void* d, int v, size_t n, cudaStream_t st) {
  typedef cudaError_t (*F)(void*, int, size_t, cudaStream_t);
  REAL(cudaMemsetAsync, F);
  Ent* e = rec(F_MEMSET, st, d, 0, n, 0, 0);
  cudaError_t r = r_cudaMemsetAsync(d, v, n, st);
  track(e, st);
  return r;
}
cudaError_t cudaLaunchKernel(const void* f, dim3 g, dim3 b, void** args, size_t sh, cudaStream_t st) {
  typedef cudaError_t (*F)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
  REAL(cudaLaunchKernel, F);
  Ent* e = rec(F_LAUNCH, st, (void*)f, 0, (uint64_t)g.x * g.y * g.z, 0, 0);
  cudaError_t r = r_cudaLaunchKernel(f, g, b, args, sh, st);
  track(e, st);
  return r;
}
cudaError_t __cudaLaunchKernel(const void* f, dim3 g, dim3 b, void** args, size_t sh, cudaStream_t st) {
  typedef cudaError_t (*F)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
  REAL(__cudaLaunchKernel, F);
  Ent* e = rec(F_LAUNCH, st, (void*)f, 0, (uint64_t)g.x * g.y * g.z, 0, 0);
  cudaError_t r = r___cudaLaunchKernel(f, g, b, args, sh, st);
  track(e, st);
  return r;
}
cudaError_t cudaLaunchKernelExC(const cfgEx* c, const void* f, void** args) {
  typedef cudaError_t (*F)(const cfgEx*, const void*, void**);
  REAL(cudaLaunchKernelExC, F);
  rec(F_LAUNCHEX, c ? c->stream : 0, (void*)f, 0, c ? (uint64_t)c->gridDim.x * c->gridDim.y * c->gridDim.z : 0, 0, 0);
  return r_cudaLaunchKernelExC(c, f, args);
}
cudaError_t cudaGraphLaunch(void* ex, cudaStream_t st) {
  typedef cudaError_t (*F)(void*, cudaStream_t);
  REAL(cudaGraphLaunch, F);
  Ent* e = rec(F_GRAPH, st, ex, 0, 0, 0, 0);
  cudaError_t r = r_cudaGraphLaunch(ex, st);
  track(e, st);
  return r;
}
static void ev_set(void* e, void* s) {
  unsigned h = (unsigned)(((uintptr_t)e >> 4) % NE);
  for (int k = 0; k < NE; k++) { unsigned j = (h + k) % NE; if (!EV_K[j] || EV_K[j] == e) { EV_K[j] = e; EV_S[j] = s; return; } }
}
static void* ev_get(void* e) {
  unsigned h = (unsigned)(((uintptr_t)e >> 4) % NE);
  for (int k = 0; k < NE; k++) { unsigned j = (h + k) % NE; if (EV_K[j] == e) return EV_S[j]; if (!EV_K[j]) return (void*)-1; }
  return (void*)-1;
}
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t st) {
  typedef cudaError_t (*F)(cudaEvent_t, cudaStream_t);
  REAL(cudaEventRecord, F);
  ev_set(e, st); rec(F_EVREC, st, e, 0, 0, 0, 0);
  return r_cudaEventRecord(e, st);
}
cudaError_t cudaEventRecordWithFlags(cudaEvent_t e, cudaStream_t st, unsigned fl) {
  typedef cudaError_t (*F)(cudaEvent_t, cudaStream_t, unsigned);
  REAL(cudaEventRecordWithFlags, F);
  ev_set(e, st); rec(F_EVREC, st, e, 0, fl, 0, 0);
  return r_cudaEventRecordWithFlags(e, st, fl);
}
cudaError_t cudaStreamWaitEvent(cudaStream_t st, cudaEvent_t e, unsigned fl) {
  typedef cudaError_t (*F)(cudaStream_t, cudaEvent_t, unsigned);
  REAL(cudaStreamWaitEvent, F);
  Ent* en = rec(F_WAIT, st, e, ev_get(e), fl, 0, 0);
  cudaError_t r = r_cudaStreamWaitEvent(st, e, fl);
  track(en, st);
  return r;
}

int vshim_dump(long stream, int n, char* buf, int len) {
  uint64_t h = H, t1 = now();
  int out = 0, got = 0;
  for (uint64_t k = 0; k < NR && k < h && got < n; k++) {
    Ent* e = &R[(h - 1 - k) % NR];
    if (stream != -1 && (long)e->s != stream) continue;
    got++;
    char nm[160] = "";
    if (e->fn == F_LAUNCH || e->fn == F_LAUNCHEX) {
      typedef cudaError_t (*FN_)(const char**, const void*);
      REAL(cudaFuncGetName, FN_);
      const char* kn = 0;
      Dl_info di;
      if (r_cudaFuncGetName && r_cudaFuncGetName(&kn, e->a) == 0 && kn) snprintf(nm, sizeof nm, "%.150s grid %lu", kn, (unsigned long)e->n);
      else if (dladdr(e->a, &di) && di.dli_sname) snprintf(nm, sizeof nm, "%.150s", di.dli_sname);
      else snprintf(nm, sizeof nm, "%p", e->a);
    } else if (e->fn == F_MEMCPY) {
      snprintf(nm, sizeof nm, "kind %d bytes %lu host %s", e->kind, (unsigned long)e->n,
               e->mt == 0 ? "PAGEABLE" : e->mt == 1 ? "pinned" : e->mt == 2 ? "device" : e->mt == -2 ? "-" : "?");
    } else if (e->fn == F_WAIT) {
      snprintf(nm, sizeof nm, "event %p recorded on stream %p", e->a, e->b);
    } else if (e->fn == F_EVREC) {
      snprintf(nm, sizeof nm, "event %p", e->a);
    }
    const char* st = "";
    if (e->ev) {
      typedef cudaError_t (*FQ)(void*);
      REAL(cudaEventQuery, FQ);
      st = r_cudaEventQuery(e->ev) == 0 ? " [done]" : " [PENDING]";
    }
    out += snprintf(buf + out, len - out, "  -%.3fs tid %u stream %p %s %s%s\n", (t1 - e->t) * 1e-9, e->tid, e->s,
                    FN[e->fn], nm, st);
    if (out >= len - 256) break;
  }
  return out;
}
