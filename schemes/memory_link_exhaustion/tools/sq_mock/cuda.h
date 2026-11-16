#pragma once
#include "cuda_runtime.h"
extern "C" {
typedef int CUresult;
enum { CUDA_SUCCESS = 0 };
typedef uint64_t CUdeviceptr;
typedef uint64_t cuuint64_t;
typedef struct MStream* CUstream;
enum { CU_STREAM_WAIT_VALUE_GEQ = 0 };
CUresult cuInit(unsigned);
CUresult cuMemAlloc(CUdeviceptr*, size_t);
CUresult cuMemsetD8(CUdeviceptr, unsigned char, size_t);
CUresult cuStreamWaitValue64(CUstream, CUdeviceptr, cuuint64_t, unsigned);
CUresult cuStreamWriteValue64(CUstream, CUdeviceptr, cuuint64_t, unsigned);
}
