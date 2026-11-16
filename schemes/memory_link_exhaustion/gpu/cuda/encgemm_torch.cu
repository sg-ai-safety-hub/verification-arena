// Torch binding for encgemm.cu: encgemm(x, w, bias, key, dec, bn, nc, stages) -> y
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <algorithm>
#include <cstdlib>
#include "encgemm.cu"
#include "chase.cu"

namespace {

// pre-emptible launch context (encgemm_pre): tile counter, challenge gate, no-yield flag, caller-owned output
struct PreCtx { int* ctr = nullptr; const int* gate = nullptr; int no_yield = 1; at::Tensor* y = nullptr; };
PreCtx g_pre;

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

template <int BN, int NC, int S, int DEC, int MT = 1>
void launch(const at::Tensor& x, const at::Tensor& w, const at::Tensor* bias, at::Tensor& y, uint32_t key,
            int group_m, int ctas, uint32_t ordinal, uint32_t sc_off = 0) {
  using L = encgemm::Smem<BN, NC, S, DEC == 2 ? 32768 : 0, MT>;
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto tmA = make_map(x.data_ptr(), M, K, x.stride(0), L::BM);
  auto tmB = make_map(w.data_ptr(), N, K, w.stride(0), BN);
  auto kern = encgemm::gemm_kernel<BN, NC, S, DEC, MT>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, L::TOTAL));
    attr = true;
  }
  const int tiles = ((M + L::BM - 1) / L::BM) * ((N + BN - 1) / BN);
  const int threads = encgemm::Threads<NC, DEC>::value;
  const int grid = ctas > 0 ? std::min(ctas, tiles) : tiles;
  kern<<<grid, threads, L::TOTAL, at::cuda::getCurrentCUDAStream()>>>(
      tmA, tmB, reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
      bias ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr()) : nullptr, M, N, K, y.stride(0), key,
      group_m, ordinal, sc_off, g_pre.ctr, g_pre.gate, g_pre.no_yield);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int BN, int NC, int S>
void launch_dec(int dec, const at::Tensor& x, const at::Tensor& w, const at::Tensor* b, at::Tensor& y,
                uint32_t key, int g, int ctas, uint32_t ordinal, uint32_t sc_off = 0) {
  if (dec == 1) launch<BN, NC, S, 1>(x, w, b, y, key, g, ctas, ordinal);
  else if (dec == 2) {
    if constexpr (BN == 128) launch<BN, NC, S, 2>(x, w, b, y, key, g, ctas, ordinal, sc_off);
    else TORCH_CHECK(false, "butterfly decode needs bn=128");
  } else launch<BN, NC, S, 0>(x, w, b, y, key, g, ctas, ordinal);
}

}  // namespace

at::Tensor encgemm_fwd(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, int64_t key, int64_t dec,
                       int64_t bn, int64_t nc, int64_t stages, int64_t group_m, int64_t ctas, int64_t ordinal);

// encgemm with the pre-emptible schedule into a caller-owned y: launch once with no_yield 0 (stops at the raised gate),
// wait for the gate, launch again with no_yield 1 on the same counter (finishes the remaining tiles)
at::Tensor encgemm_pre(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, int64_t key, int64_t dec,
                       int64_t bn, int64_t nc, int64_t stages, int64_t group_m, int64_t ctas, int64_t ordinal,
                       at::Tensor y, at::Tensor ctr, at::Tensor gate, int64_t no_yield) {
  TORCH_CHECK(dec == 2, "encgemm_pre: butterfly decode only");
  TORCH_CHECK(y.is_contiguous() && y.size(0) == x.size(0) && y.size(1) == w.size(0));
  g_pre.ctr = (int*)ctr.data_ptr(); g_pre.gate = (const int*)gate.data_ptr(); g_pre.no_yield = (int)no_yield; g_pre.y = &y;
  at::Tensor out;
  try {
    out = encgemm_fwd(x, w, bias, key, dec, bn, nc, stages, group_m, ctas, ordinal);
  } catch (...) {
    g_pre = PreCtx();
    throw;
  }
  g_pre = PreCtx();
  return out;
}

at::Tensor encgemm_fwd(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, int64_t key, int64_t dec,
                       int64_t bn, int64_t nc, int64_t stages, int64_t group_m, int64_t ctas, int64_t ordinal) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.dtype() == at::kBFloat16 && w.dtype() == at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1));
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1, "inner dims must be contiguous");
  TORCH_CHECK(x.size(1) % 64 == 0, "K must be a multiple of 64");
  TORCH_CHECK((x.stride(0) * 2) % 16 == 0 && (w.stride(0) * 2) % 16 == 0);
  TORCH_CHECK(w.size(0) % 8 == 0, "N must be a multiple of 8");
  c10::cuda::CUDAGuard g(x.device());
  auto y = g_pre.y ? *g_pre.y : at::empty({x.size(0), w.size(0)}, x.options());
  if (x.size(0) == 0) return y;
  at::Tensor bb;
  const at::Tensor* bp = nullptr;
  if (bias.has_value()) {
    bb = bias->contiguous();
    TORCH_CHECK(bb.dtype() == at::kBFloat16 && bb.numel() == w.size(0));
    bp = &bb;
  }
  const uint32_t k = uint32_t(key & 0xFFFFFFFF);
  const int gm = int(group_m);
  // ordinal: low 32 bits = the matrix's butterfly ordinal; high 32 bits = super-chunk offset of a row slice of W
  const uint32_t ord32 = uint32_t(uint64_t(ordinal) & 0xFFFFFFFFull), soff = uint32_t(uint64_t(ordinal) >> 32);
  // nc 4 with the butterfly decode: two consumer warpgroups of two 64-row sub-tiles each (256 rows per CTA)
  if (dec == 2 && bn == 128 && nc == 4 && stages == 4) { launch<128, 2, 4, 2, 2>(x, w, bp, y, k, gm, int(ctas), ord32, soff); return y; }
  if (dec == 2 && bn == 128 && nc == 4 && stages == 3) { launch<128, 2, 3, 2, 2>(x, w, bp, y, k, gm, int(ctas), ord32, soff); return y; }
#define CASE(BN_, NC_, S_) \
  if (bn == BN_ && nc == NC_ && stages == S_) { launch_dec<BN_, NC_, S_>(int(dec), x, w, bp, y, k, gm, int(ctas), ord32, soff); return y; }
  CASE(32, 1, 8) CASE(32, 1, 12) CASE(64, 1, 8) CASE(64, 1, 6) CASE(128, 1, 6) CASE(128, 1, 4)
  CASE(128, 2, 3) CASE(128, 2, 4) CASE(128, 2, 5) CASE(128, 2, 6) CASE(256, 1, 4) CASE(256, 1, 3) CASE(16, 1, 12) CASE(64, 2, 6)
#undef CASE
  TORCH_CHECK(false, "unsupported config bn=", bn, " nc=", nc, " stages=", stages);
}

// In-place XOR of a row-major [N, K] bf16 matrix with its butterfly keystream (encode == decode).
__global__ void __launch_bounds__(128) bfly_xor_kernel(const uint4* __restrict__ src, uint4* __restrict__ w,
                                                       int N, int K, uint32_t seed, uint32_t ordinal,
                                                       int sc0 = 0) {
  __shared__ uint4 xs[2048];
  const int kpr = K / 128;
  const int sc = sc0 + blockIdx.x;                // super-chunk = row block * kpr + kp (of the whole matrix)
  const int nb = blockIdx.x / kpr, kp = sc % kpr; // local row block (src/w may be a row slice)
  const int t = threadIdx.x;
  uint4 v[16];
  bfly::superchunk_ks<encgemm::BF_L, encgemm::BF_RB, encgemm::BF_MIX, encgemm::BF_XB, true>(
      v, xs, ((unsigned long long)ordinal << 32) | (unsigned long long)sc, t, seed);
  const int cg = t & 15, rb = t >> 4;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int64_t row = (int64_t)nb * 128 + rb + 8 * i;
    const int64_t o = (row * K + kp * 128 + cg * 8) / 8;
    uint4 x = src[o];
    x.x ^= v[i].x; x.y ^= v[i].y; x.z ^= v[i].z; x.w ^= v[i].w;
    w[o] = x;
  }
}

// VARENA_BFLY_XOR_SMEM_KB (16 Nov, link experiment): extra dynamic shared memory per decode block, which caps the
// blocks per SM and so the decode-ahead's peak HBM rate (smooths its bursts; 0 = off)
static int bfly_xor_extra_smem() {
  static int v = -1;
  if (v < 0) {
    const char* e = std::getenv("VARENA_BFLY_XOR_SMEM_KB");
    v = e ? std::max(0, std::atoi(e)) * 1024 : 0;
    if (v) C10_CUDA_CHECK(cudaFuncSetAttribute(bfly_xor_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, v));
  }
  return v;
}

void bfly_xor_to(at::Tensor src, at::Tensor w, int64_t key, int64_t ordinal) {
  TORCH_CHECK(w.dim() == 2 && w.is_contiguous() && w.element_size() == 2 && src.is_contiguous());
  TORCH_CHECK(src.numel() == w.numel());
  const int N = w.size(0), K = w.size(1);
  TORCH_CHECK(N % 128 == 0 && K % 128 == 0, "butterfly super-chunks need N, K multiples of 128");
  bfly_xor_kernel<<<(N / 128) * (K / 128), 128, bfly_xor_extra_smem(), at::cuda::getCurrentCUDAStream()>>>(
      (const uint4*)src.data_ptr(), (uint4*)w.data_ptr(), N, K, uint32_t(key & 0xFFFFFFFF), uint32_t(ordinal));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bfly_xor(at::Tensor w, int64_t key, int64_t ordinal) { bfly_xor_to(w, w, key, ordinal); }

// Row slice [row0, row0 + n) of an encoded matrix (row0 % 128 == 0): src/w are the slice's rows only.
void bfly_xor_rows(at::Tensor src, at::Tensor w, int64_t key, int64_t ordinal, int64_t row0) {
  TORCH_CHECK(w.dim() == 2 && w.is_contiguous() && w.element_size() == 2 && src.is_contiguous());
  TORCH_CHECK(src.numel() == w.numel());
  const int N = w.size(0), K = w.size(1);
  TORCH_CHECK(N % 128 == 0 && K % 128 == 0 && row0 % 128 == 0, "butterfly rows: multiples of 128");
  bfly_xor_kernel<<<(N / 128) * (K / 128), 128, bfly_xor_extra_smem(), at::cuda::getCurrentCUDAStream()>>>(
      (const uint4*)src.data_ptr(), (uint4*)w.data_ptr(), N, K, uint32_t(key & 0xFFFFFFFF), uint32_t(ordinal),
      (int)(row0 / 128) * (K / 128));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static int chase_wpb() {
  static int w = [] { const char* e = getenv("VARENA_CHASE_WPB"); return e ? atoi(e) : 8; }();
  return w;
}

void chase_honest_launch(at::Tensor gran_addr, int64_t total_blocks, int64_t nb, int64_t nonce_ptr,
                         int64_t K, int64_t P, at::Tensor out) {
  TORCH_CHECK(nb <= chase::MAX_BLOCKS && nb <= total_blocks);
  const int wpb = chase_wpb();
  const int grid = (int)((K + wpb - 1) / wpb);
  chase::chase_honest<<<grid, 32 * wpb, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint64_t*)gran_addr.data_ptr(), (int)total_blocks, (int)nb, (const uint32_t*)nonce_ptr,
      (int)K, (int)P, (uint64_t*)out.data_ptr(), 0ull, 1, nullptr, nullptr, 1, nullptr, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// mixed chase with noise-ring granules: nonce_ptr points at two pinned words, {nonce, head window}
// the whole-memory chase: ring granules (n_ring > 1) and the settled-KV rule (gran_kv int32 per table entry, kv_tab
// uint16 per KV block on the device, nb_kv blocks)
void chase_honest_full_launch(at::Tensor gran_addr, int64_t total_blocks, int64_t nb, int64_t nonce_ptr,
                              int64_t K, int64_t P, at::Tensor out, int64_t ring_base, int64_t n_ring,
                              at::Tensor gran_kv, at::Tensor kv_tab, int64_t nb_kv, int64_t kv_info_ptr, int64_t info_off) {
  TORCH_CHECK(nb <= chase::MAX_BLOCKS && nb <= total_blocks && nb_kv >= 1024 && (n_ring <= 1 || (n_ring & (n_ring - 1)) == 0));
  TORCH_CHECK(gran_kv.scalar_type() == at::kInt && kv_tab.scalar_type() == at::kShort && kv_tab.numel() >= nb_kv);
  const int wpb = chase_wpb();
  const int grid = (int)((K + wpb - 1) / wpb);
  chase::chase_honest<<<grid, 32 * wpb, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint64_t*)gran_addr.data_ptr(), (int)total_blocks, (int)nb, (const uint32_t*)nonce_ptr,
      (int)K, (int)P, (uint64_t*)out.data_ptr(), (uint64_t)ring_base, (int)(n_ring > 1 ? n_ring : 1),
      (const int32_t*)gran_kv.data_ptr(), (const uint16_t*)kv_tab.data_ptr(), (int)nb_kv,
      (const uint64_t*)kv_info_ptr, (int)info_off);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// debug: copy out.numel() uint32 words from a raw device address (the chase's own addressing: granule base + offset)
__global__ void peek_k(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}
void peek_launch(int64_t addr, at::Tensor out) {
  TORCH_CHECK(out.scalar_type() == at::kInt && out.is_cuda());
  const int n = (int)out.numel();
  peek_k<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint32_t*)addr, (uint32_t*)out.data_ptr(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// block table switch: host_tab_ptr / host_info_ptr are pinned host buffers (uint16 [n] padded to 8 bytes; 2 x uint64)
void kv_tab_apply_launch(int64_t host_tab_ptr, at::Tensor dev_tab, int64_t host_info_ptr, at::Tensor dev_info) {
  TORCH_CHECK(dev_tab.scalar_type() == at::kShort && dev_tab.numel() % 4 == 0 && dev_info.numel() == 2);
  const int nw = (int)(dev_tab.numel() / 4);
  chase::kv_tab_apply<<<(nw + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint64_t*)host_tab_ptr, (uint64_t*)dev_tab.data_ptr(), nw, (const uint64_t*)host_info_ptr,
      (uint64_t*)dev_info.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// INDEP filler for a list of global pages (same values as kernels._indep_fill_kernel / ref.indep_words):
// unit u (16 B) of page gp: chunk q = gp*512 + u/8, j = u%8, value = sum_{i<hf8} Philox4x32-8(ctr =
// (q_lo, q_hi, 8i + j, 0), key = (fkey, 0x7F4A7C15)). One thread per unit, grid-stride over all pages,
// coalesced 128-bit stores (consecutive units are consecutive 16 B).
__global__ void indep_fill_pages_kernel(int32_t* __restrict__ base, const int64_t* __restrict__ page_map,
                                        const int64_t* __restrict__ pages, int64_t n, uint32_t fkey, int hf8) {
  const int64_t total = n * 4096;
  for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (int64_t)gridDim.x * blockDim.x) {
    const int64_t pi = t >> 12;
    const uint32_t u = (uint32_t)(t & 4095);
    const int64_t gp = pages[pi];
    const uint64_t q = (uint64_t)gp * 512 + (u >> 3);
    const uint32_t j = u & 7;
    uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < hf8; ++i) {
      uint32_t c0 = (uint32_t)q, c1 = (uint32_t)(q >> 32), c2 = j + 8u * i, c3 = 0, k0 = fkey, k1 = 0x7F4A7C15u;
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const uint32_t h0 = __umulhi(0xD2511F53u, c0), l0 = 0xD2511F53u * c0;
        const uint32_t h1 = __umulhi(0xCD9E8D57u, c2), l1 = 0xCD9E8D57u * c2;
        c0 = h1 ^ c1 ^ k0; c1 = l1; c2 = h0 ^ c3 ^ k1; c3 = l0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
      }
      s0 += c0; s1 += c1; s2 += c2; s3 += c3;
    }
    int4* dst = reinterpret_cast<int4*>(base + page_map[gp]) + u;
    *dst = make_int4((int)s0, (int)s1, (int)s2, (int)s3);
  }
}

void indep_fill_pages(at::Tensor base, at::Tensor page_map, at::Tensor pages, int64_t fkey, int64_t hf8) {
  const int64_t n = pages.numel();
  if (n == 0) return;
  int dev = 0, sms = 132;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int64_t total = n * 4096;
  const int threads = 256;
  const int64_t want = (total + threads - 1) / threads;
  const int grid = (int)std::min<int64_t>(want, (int64_t)sms * 16);
  indep_fill_pages_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      (int32_t*)base.data_ptr(), (const int64_t*)page_map.data_ptr(), (const int64_t*)pages.data_ptr(), n,
      (uint32_t)(fkey & 0xFFFFFFFF), (int)hf8);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("encgemm", &encgemm_fwd, "decode-on-load Hopper GEMM");
  m.def("encgemm_pre", &encgemm_pre, "pre-emptible butterfly decode-on-load GEMM (tile counter, exits at the gate)");
  m.def("chase_honest", &chase_honest_launch, "probe chase over the arena granules");
  m.def("chase_honest_full", &chase_honest_full_launch, "probe chase: arena, noise ring, settled-KV rule");
  m.def("kv_tab_apply", &kv_tab_apply_launch, "replace the device KV block table from pinned host memory");
  m.def("peek", &peek_launch, "debug: read uint32 words at a raw device address");
  m.def("bfly_xor", &bfly_xor, "XOR a [N, K] bf16 matrix with its butterfly keystream");
  m.def("bfly_xor_to", &bfly_xor_to, "dst = src XOR butterfly keystream (decode-ahead into scratch)");
  m.def("bfly_xor_rows", &bfly_xor_rows, "row slice of bfly_xor_to (rows row0 .. row0 + n)");
  m.def("indep_fill_pages", &indep_fill_pages, "INDEP filler for a page list (CUDA, vector stores)");
}
