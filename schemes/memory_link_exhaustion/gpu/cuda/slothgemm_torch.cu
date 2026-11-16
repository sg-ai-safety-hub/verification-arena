// Torch binding for slothgemm.cu (phase-1 timing prototype).
//   slgemm(x, w, seed, i0, dec, bn, nc, sl, stages, ndec, group_m, ctas, gate) -> y = x @ decode(w)^T
//   sldec(src, dst, seed, i0, blocks, threads)                        two-pass decode into a scratch buffer
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <algorithm>
#include "slothgemm3.cu"
#include "slothgemm4.cu"

namespace {

CUtensorMap make_map(const void* ptr, int64_t rows, int64_t cols, int64_t ld_elems, int box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cuuint64_t(cols), cuuint64_t(rows)};
  cuuint64_t strides[1] = {cuuint64_t(ld_elems * 2)};
  cuuint32_t box[2] = {64, cuuint32_t(box_rows)};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims,
                                      strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", int(r));
  return m;
}

template <int BN, int NC, int SL, int S, int NDEC, int DEC>
void launch(const at::Tensor& x, const at::Tensor& w, at::Tensor& y, uint64_t seed, uint64_t i0, int group_m,
            int ctas, const int* gate, int preempt = -1) {
  using L = slgemm::Lay<BN, NC, SL, S>;
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto tmA = make_map(x.data_ptr(), M, K, x.stride(0), L::BM);
  auto tmB = make_map(w.data_ptr(), N, K, w.stride(0), BN);
  auto kern = slgemm::sl_gemm_kernel<BN, NC, SL, S, NDEC, DEC>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, L::TOTAL));
    attr = true;
  }
  const int tiles = ((M + L::BM - 1) / L::BM) * ((N + BN - 1) / BN);
  const int grid = ctas > 0 ? std::min(ctas, tiles) : tiles;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto run = [&](int* ctr, int yield) {
    kern<<<grid, slgemm::Cfg<NC, NDEC>::THREADS, L::TOTAL, stream>>>(
        tmA, tmB, reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, N, K, y.stride(0), seed, i0, group_m, gate, ctr,
        yield);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  if (preempt < 0 || gate == nullptr) {
    run(nullptr, 0);
    return;
  }
  // pre-emptible (as actenc2._launch_mm): CTAs claim tiles from one counter; a launch that sees the gate raised
  // exits at its tile boundaries, a 1-thread kernel holds the stream until the gate drops, and the next launch
  // takes the remaining tiles. `preempt` extra yielding rounds, then a final launch that ignores the gate.
  auto ctr = at::zeros({1}, at::TensorOptions().dtype(at::kInt).device(x.device()));
  int* cp = ctr.data_ptr<int>();
  for (int r = 0; r <= preempt; ++r) {
    run(cp, 1);
    slgemm::sl_gate_wait_kernel<<<1, 1, 0, stream>>>(gate);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  run(cp, 0);
}


template <int CM, int S, int DEC>
void launch3(const at::Tensor& x, const at::Tensor& w, at::Tensor& y, uint64_t seed, uint64_t i0, int clusters) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto tmA = make_map(x.data_ptr(), M, K, x.stride(0), slg3::BM);
  auto tmB = make_map(w.data_ptr(), N, K, w.stride(0), slg3::BN / CM);
  auto kern = slg3::sl3_kernel<CM, S, DEC>;
  const int smem = slg3::Lay<S>::TOTAL;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    attr = true;
  }
  const int units = ((M + slg3::BM * CM - 1) / (slg3::BM * CM)) * ((N + slg3::BN - 1) / slg3::BN);
  const int ncl = std::min(clusters, units);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(ncl * CM);
  cfg.blockDim = dim3(384);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at_[1];
  at_[0].id = cudaLaunchAttributeClusterDimension;
  at_[0].val.clusterDim.x = CM;
  at_[0].val.clusterDim.y = 1;
  at_[0].val.clusterDim.z = 1;
  cfg.attrs = at_;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, tmA, tmB, reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, N, K,
                                    (int)y.stride(0), seed, i0));
}


template <int CM, int SA, int SB, int DEC>
void launch4(const at::Tensor& x, const at::Tensor& w, at::Tensor& y, uint64_t seed, uint64_t i0, int clusters) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto tmA = make_map(x.data_ptr(), M, K, x.stride(0), slg4::BM);
  auto tmB = make_map(w.data_ptr(), N, K, w.stride(0), slg4::BN / CM);
  auto kern = slg4::sl4_kernel<CM, SA, SB, DEC>;
  const int smem = slg4::Lay<CM, SA, SB>::TOTAL;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    attr = true;
  }
  const int units = ((M + slg4::BM * CM - 1) / (slg4::BM * CM)) * ((N + slg4::BN - 1) / slg4::BN);
  const int ncl = std::min(clusters, units);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(ncl * CM);
  cfg.blockDim = dim3(384);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at_[1];
  at_[0].id = cudaLaunchAttributeClusterDimension;
  at_[0].val.clusterDim.x = CM;
  at_[0].val.clusterDim.y = 1;
  at_[0].val.clusterDim.z = 1;
  cfg.attrs = at_;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, tmA, tmB, reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, N, K,
                                    (int)y.stride(0), seed, i0));
}

}  // namespace

// v4: three rings, decoded tiles stored into every cluster CTA. cm = CTAs per cluster, sa/sb = A/B ring stages.
at::Tensor slgemm4_fwd(at::Tensor x, at::Tensor w, int64_t seed, int64_t i0, int64_t dec, int64_t cm, int64_t sa,
                       int64_t sb, int64_t clusters) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.dtype() == at::kBFloat16 && w.dtype() == at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1));
  TORCH_CHECK(x.stride(1) == 1 && w.is_contiguous(), "x inner dim and w must be contiguous");
  TORCH_CHECK(x.size(1) % 64 == 0 && w.size(0) % 128 == 0, "K % 64, N % 128");
  c10::cuda::CUDAGuard g(x.device());
  auto y = at::empty({x.size(0), w.size(0)}, x.options());
  if (x.size(0) == 0) return y;
  const uint64_t sd = uint64_t(seed), o = uint64_t(i0);
  const int ncl = int(clusters);
#define CASE4(CM_, SA_, SB_)                                              \
  if (cm == CM_ && sa == SA_ && sb == SB_) {                              \
    if (dec) launch4<CM_, SA_, SB_, 1>(x, w, y, sd, o, ncl);              \
    else launch4<CM_, SA_, SB_, 0>(x, w, y, sd, o, ncl);                  \
    return y;                                                             \
  }
  CASE4(1, 3, 6) CASE4(2, 3, 6) CASE4(4, 3, 6) CASE4(1, 2, 8) CASE4(2, 2, 8) CASE4(4, 2, 8)
#undef CASE4
  TORCH_CHECK(false, "unsupported v4 config cm=", cm, " sa=", sa, " sb=", sb);
}

// v3: cluster-shared decoded tiles. cm = CTAs per cluster (1, 2, 4), clusters = persistent cluster count.
at::Tensor slgemm3_fwd(at::Tensor x, at::Tensor w, int64_t seed, int64_t i0, int64_t dec, int64_t cm, int64_t stages,
                       int64_t clusters) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.dtype() == at::kBFloat16 && w.dtype() == at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1));
  TORCH_CHECK(x.stride(1) == 1 && w.is_contiguous(), "x inner dim and w must be contiguous");
  TORCH_CHECK(x.size(1) % 64 == 0 && w.size(0) % 128 == 0, "K % 64, N % 128");
  c10::cuda::CUDAGuard g(x.device());
  auto y = at::empty({x.size(0), w.size(0)}, x.options());
  if (x.size(0) == 0) return y;
  const uint64_t sd = uint64_t(seed), o = uint64_t(i0);
  const int ncl = int(clusters);
#define CASE3(CM_, S_)                                                    \
  if (cm == CM_ && stages == S_) {                                        \
    if (dec) launch3<CM_, S_, 1>(x, w, y, sd, o, ncl);                    \
    else launch3<CM_, S_, 0>(x, w, y, sd, o, ncl);                        \
    return y;                                                             \
  }
  CASE3(1, 4) CASE3(2, 4) CASE3(4, 4) CASE3(1, 3) CASE3(2, 3) CASE3(4, 3)
#undef CASE3
  TORCH_CHECK(false, "unsupported v3 config cm=", cm, " stages=", stages);
}

at::Tensor slgemm_fwd(at::Tensor x, at::Tensor w, int64_t seed, int64_t i0, int64_t dec, int64_t bn, int64_t nc, int64_t sl,
                      int64_t stages, int64_t ndec, int64_t group_m, int64_t ctas, int64_t gate,
                      int64_t preempt = -1) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.dtype() == at::kBFloat16 && w.dtype() == at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1));
  TORCH_CHECK(x.stride(1) == 1 && w.is_contiguous(), "x inner dim and w must be contiguous");
  TORCH_CHECK(x.size(1) % 64 == 0, "K must be a multiple of 64");
  TORCH_CHECK(w.size(0) % bn == 0, "N must be a multiple of bn");
  c10::cuda::CUDAGuard g(x.device());
  auto y = at::empty({x.size(0), w.size(0)}, x.options());
  if (x.size(0) == 0) return y;
  const uint64_t sd = uint64_t(seed), o = uint64_t(i0);
  const int gm = int(group_m), ct = int(ctas);
  const int* gp = reinterpret_cast<const int*>(gate);   // device int32 challenge gate, 0 = none
#define CASE(BN_, NC_, SL_, S_, ND_)                                                         \
  if (bn == BN_ && nc == NC_ && sl == SL_ && stages == S_ && ndec == ND_) {                    \
    if (dec) launch<BN_, NC_, SL_, S_, ND_, 1>(x, w, y, sd, o, gm, ct, gp, int(preempt));                        \
    else launch<BN_, NC_, SL_, S_, ND_, 0>(x, w, y, sd, o, gm, ct, gp, int(preempt));                            \
    return y;                                                                                 \
  }
  CASE(64, 2, 2, 4, 2) CASE(64, 4, 1, 4, 2) CASE(64, 2, 1, 6, 3) CASE(128, 2, 1, 6, 2)
#undef CASE
  TORCH_CHECK(false, "unsupported config bn=", bn, " nc=", nc, " stages=", stages, " ndec=", ndec);
}

void sldec(at::Tensor src, at::Tensor dst, int64_t seed, int64_t i0, int64_t blocks, int64_t threads) {
  TORCH_CHECK(src.is_contiguous() && dst.is_contiguous() && src.nbytes() == dst.nbytes() && src.nbytes() % 16 == 0);
  const int64_t n2 = src.nbytes() / 16;
  slgemm::sl_decode_kernel<1><<<(int)blocks, (int)threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const ulonglong2*)src.data_ptr(), (ulonglong2*)dst.data_ptr(), n2, uint64_t(seed), uint64_t(i0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("slgemm", &slgemm_fwd, "fused sloth64 decode-on-load Hopper GEMM", py::arg("x"), py::arg("w"), py::arg("seed"),
        py::arg("i0"), py::arg("dec"), py::arg("bn"), py::arg("nc"), py::arg("sl"), py::arg("stages"), py::arg("ndec"),
        py::arg("group_m"), py::arg("ctas"), py::arg("gate"), py::arg("preempt") = -1);
  m.def("slgemm3", &slgemm3_fwd, "fused sloth64 GEMM v3 (cluster-shared decoded tiles)");
  m.def("slgemm4", &slgemm4_fwd, "fused sloth64 GEMM v4 (three rings, cluster-shared decoded tiles)");
  m.def("sldec", &sldec, "two-pass sloth64 decode into a scratch buffer");
}
