"""Paged decode attention in CUDA C++ (JIT-built with torch.utils.cpp_extension.load_inline).
Port of the Metal kernel: one thread block per (sequence, kv head), WARPS warps stride over keys, each lane
owns 4 of the D=128 dims, warp-shuffle dot products, fp32 online softmax per warp, then a shared-memory
merge across warps. All G = H/HKV query heads that share the kv head are handled together."""
import torch
from torch.utils.cpp_extension import load_inline

_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

template <int G, int WARPS>
__global__ void paged_decode_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                    const __nv_bfloat16* __restrict__ vc, const int* __restrict__ bt,
                                    const int* __restrict__ ctx_lens, __nv_bfloat16* __restrict__ out,
                                    int H, int HKV, int BS, int maxb) {
  constexpr int D = 128;
  const int s = blockIdx.y, kvh = blockIdx.x, lane = threadIdx.x % 32, w = threadIdx.x / 32;
  const int ctx = ctx_lens[s];
  const float scale = rsqrtf((float)D);
  float qv[G][4], acc[G][4], m[G], l[G];
  for (int g = 0; g < G; g++) {
    const __nv_bfloat16* qp = q + ((size_t)s * H + kvh * G + g) * D + lane * 4;
    for (int j = 0; j < 4; j++) { qv[g][j] = __bfloat162float(qp[j]) * scale; acc[g][j] = 0.f; }
    m[g] = -INFINITY; l[g] = 0.f;
  }
  for (int k = w; k < ctx; k += WARPS) {
    const size_t slot = (size_t)bt[s * maxb + k / BS] * BS + k % BS;
    const size_t base = (slot * HKV + kvh) * D + lane * 4;
    float kv[4], vv[4];
    for (int j = 0; j < 4; j++) { kv[j] = __bfloat162float(kc[base + j]); vv[j] = __bfloat162float(vc[base + j]); }
    for (int g = 0; g < G; g++) {
      float sc = qv[g][0] * kv[0] + qv[g][1] * kv[1] + qv[g][2] * kv[2] + qv[g][3] * kv[3];
      for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(0xffffffff, sc, o);
      const float mn = fmaxf(m[g], sc), corr = __expf(m[g] - mn), e = __expf(sc - mn);
      l[g] = l[g] * corr + e;
      for (int j = 0; j < 4; j++) acc[g][j] = acc[g][j] * corr + e * vv[j];
      m[g] = mn;
    }
  }
  __shared__ float tm[WARPS][G], tl[WARPS][G], tacc[WARPS][G][D];
  for (int g = 0; g < G; g++) {
    if (lane == 0) { tm[w][g] = m[g]; tl[w][g] = l[g]; }
    for (int j = 0; j < 4; j++) tacc[w][g][lane * 4 + j] = acc[g][j];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < G * D; i += WARPS * 32) {
    const int g = i / D, d = i % D;
    float M = -INFINITY;
    for (int x = 0; x < WARPS; x++) M = fmaxf(M, tm[x][g]);
    float L = 0.f, A = 0.f;
    for (int x = 0; x < WARPS; x++) {
      const float f = tm[x][g] == -INFINITY ? 0.f : __expf(tm[x][g] - M);
      L += tl[x][g] * f; A += tacc[x][g][d] * f;
    }
    out[((size_t)s * H + kvh * G + g) * D + d] = __float2bfloat16(L > 0.f ? A / L : 0.f);
  }
}

torch::Tensor paged_decode(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor bt,
                           torch::Tensor ctx_lens, int64_t bs) {
  TORCH_CHECK(q.size(2) == 128, "head_dim must be 128");
  const int S = q.size(0), H = q.size(1), HKV = kc.size(1), G = H / HKV;
  auto out = torch::empty_like(q);
  auto qc = q.contiguous(); auto btc = bt.contiguous(); auto cl = ctx_lens.contiguous();
  constexpr int WARPS = 8;
  dim3 grid(HKV, S), block(WARPS * 32);
  auto stream = c10::cuda::getCurrentCUDAStream();
  #define LAUNCH(GG) paged_decode_kernel<GG, WARPS><<<grid, block, 0, stream>>>( \
      (const __nv_bfloat16*)qc.data_ptr(), (const __nv_bfloat16*)kc.data_ptr(), (const __nv_bfloat16*)vc.data_ptr(), \
      btc.data_ptr<int>(), cl.data_ptr<int>(), (__nv_bfloat16*)out.data_ptr(), H, HKV, (int)bs, (int)btc.size(1))
  if (G == 2) LAUNCH(2); else if (G == 4) LAUNCH(4); else if (G == 8) LAUNCH(8); else if (G == 1) LAUNCH(1);
  else TORCH_CHECK(false, "unsupported GQA group size");
  return out;
}
"""
_mod = None


def paged_decode(q, kc, vc, block_tables, ctx_lens, bs):
    global _mod
    if _mod is None:
        _mod = load_inline("paged_decode_cuda", cpp_sources="torch::Tensor paged_decode(torch::Tensor q, "
                           "torch::Tensor kc, torch::Tensor vc, torch::Tensor bt, torch::Tensor ctx_lens, int64_t bs);",
                           cuda_sources=_SRC, functions=["paged_decode"], extra_cuda_cflags=["-O3", "--use_fast_math"])
    return _mod.paged_decode(q, kc, vc, block_tables, ctx_lens, bs)
