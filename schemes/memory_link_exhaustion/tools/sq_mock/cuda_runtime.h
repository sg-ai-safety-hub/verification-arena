// Mock CUDA runtime/driver API for CPU tests of the single H2D queue (streams are worker threads, device memory is
// host memory registered as "device"). Only what gpu/native.py and the tests use.
#pragma once
#include <cstddef>
#include <cstdint>
extern "C" {
typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorNotReady = 600 };
typedef struct MStream* cudaStream_t;
typedef struct MEvent* cudaEvent_t;
typedef struct MGraph* cudaGraphExec_t;
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2,
                      cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 };
enum cudaMemoryType { cudaMemoryTypeUnregistered = 0, cudaMemoryTypeHost = 1, cudaMemoryTypeDevice = 2,
                      cudaMemoryTypeManaged = 3 };
struct cudaPointerAttributes { enum cudaMemoryType type; int device; void* devicePointer; void* hostPointer; };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone = 0, cudaStreamCaptureStatusActive = 1 };
#define cudaEventDisableTiming 2u
#define cudaEventBlockingSync 1u
#define cudaStreamNonBlocking 1u
#define cudaHostAllocDefault 0u
#define cudaStreamLegacy ((cudaStream_t)0x1)
#define cudaStreamPerThread ((cudaStream_t)0x2)
cudaError_t cudaGetLastError(void);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDevice(int*);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
cudaError_t cudaEventCreate(cudaEvent_t*);
cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventQuery(cudaEvent_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t*, unsigned, int);
cudaError_t cudaStreamGetPriority(cudaStream_t, int*);
cudaError_t cudaDeviceGetStreamPriorityRange(int*, int*);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamQuery(cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
cudaError_t cudaStreamIsCapturing(cudaStream_t, enum cudaStreamCaptureStatus*);
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, enum cudaMemcpyKind, cudaStream_t);
cudaError_t cudaMemcpy(void*, const void*, size_t, enum cudaMemcpyKind);
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes*, const void*);
cudaError_t cudaHostAlloc(void**, size_t, unsigned);
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
cudaError_t cudaStreamBeginCapture(cudaStream_t, int);
cudaError_t cudaStreamEndCapture(cudaStream_t, void**);
// test helpers
void* mockDeviceAlloc(size_t);
void mockKernel(cudaStream_t, void (*fn)(void*), void* arg);
long mockCopiesOn(cudaStream_t);
}
