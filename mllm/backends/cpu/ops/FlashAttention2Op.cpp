// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "mllm/backends/cpu/ops/FlashAttention2Op.hpp"
#include "mllm/backends/cpu/kernels/common/fa2_1/fwd_bshd.hpp"
#if defined(__ARM_NEON)
#include "mllm/backends/cpu/kernels/arm/mllm_blas/mllm_blas_sgemm.hpp"
#include "mllm/backends/cpu/kernels/arm/softmax.hpp"
#include "mllm/backends/cpu/kernels/common/llamafile/llamafile_sgemm.hpp"
#include "mllm/core/Parallel.hpp"
#endif
#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu {

#if defined(__ARM_NEON)
namespace {

bool hasDenseHeadRows(const Tensor& tensor) {
  const auto strides = tensor.stride();
  const auto shape = tensor.shape();
  return strides[3] == 1 && strides[2] == shape[3] && strides[1] >= static_cast<int64_t>(shape[2]) * shape[3];
}

void streamedDenseAttention(const Tensor& query, const Tensor& key, const Tensor& value, Tensor& output,
                            const aops::FlashAttention2OpOptions& options) {
  MLLM_RT_ASSERT(query.rank() == 4 && key.rank() == 4 && value.rank() == 4 && output.rank() == 4);
  MLLM_RT_ASSERT(options.causal_mask && !options.hp_exp && options.B == 1);
  MLLM_RT_ASSERT(query.size(0) == 1 && key.size(0) == 1 && value.size(0) == 1);
  MLLM_RT_ASSERT(query.device() == kCPU && key.device() == kCPU && value.device() == kCPU);
  MLLM_RT_ASSERT(query.dtype() == kFloat32 && key.dtype() == kFloat32 && value.dtype() == kFloat32);
  const int heads = query.size(1);
  const int query_rows = query.size(2);
  const int key_rows = key.size(2);
  const int width = query.size(3);
  MLLM_RT_ASSERT(heads > 0 && heads == options.q_head && heads == options.kv_head);
  MLLM_RT_ASSERT(key.size(1) == heads && value.size(1) == heads && value.size(2) == key_rows);
  MLLM_RT_ASSERT(width > 0 && width % 4 == 0 && width == options.D && key.size(3) == width && value.size(3) == width);
  MLLM_RT_ASSERT(query_rows > 0 && key_rows >= query_rows);
  MLLM_RT_ASSERT(hasDenseHeadRows(query) && hasDenseHeadRows(key) && hasDenseHeadRows(value) && hasDenseHeadRows(output));
  const int threads = options.getThreads();
  MLLM_RT_ASSERT(threads > 0);
  const auto query_stride = query.stride()[1];
  const auto key_stride = key.stride()[1];
  const auto value_stride = value.stride()[1];
  const auto output_stride = output.stride()[1];
  const float scale = 1.f / sqrtf(width);
  // Keep separate score and probability regions for the softmax kernel's no-alias contract.
  auto workspace = Tensor::empty({2, query_rows, key_rows}, kFloat32, kCPU).alloc();
  auto* scores = workspace.ptr<float>();
  auto* weights = scores + static_cast<int64_t>(query_rows) * key_rows;
  for (int head = 0; head < heads; ++head) {
    const auto* q = query.ptr<float>() + static_cast<int64_t>(head) * query_stride;
    const auto* k = key.ptr<float>() + static_cast<int64_t>(head) * key_stride;
    const auto* v = value.ptr<float>() + static_cast<int64_t>(head) * value_stride;
    auto* out = output.ptr<float>() + static_cast<int64_t>(head) * output_stride;
    MLLM_CONDITIONAL_PARALLEL_FOR(threads > 1, threads, id, 0, threads, 1, {
      llamafile_causal_scaled_qk_f32(key_rows, query_rows, width, k, q, scores, scale, id, threads);
    });
    MLLM_CONDITIONAL_PARALLEL_FOR(threads > 1, threads, row, 0, query_rows, 1, {
      const auto offset = static_cast<int64_t>(row) * key_rows;
      arm::softmax_v1_fp32(scores + offset, weights + offset, key_rows, 1, threads);
    });
    const bool completed = arm::__mllm_blas_sgemm_nt_nt(query_rows, width, key_rows, weights, key_rows, v, width, out, width,
                                                       0, threads);
    MLLM_RT_ASSERT(completed);
  }
  static const bool diagnostic = [] {
    const char* value = std::getenv("MLLM_QWEN_NPU_DENSE_BHSD_DIAGNOSTIC");
    return value && std::strcmp(value, "1") == 0;
  }();
  if (diagnostic) {
    std::fprintf(stderr, "SHADOWNPU_DENSE_BHSD prefix=%d queries=%d heads=%d scratch_bytes=%lld\n", key_rows, query_rows,
                 heads, 2LL * query_rows * key_rows * static_cast<long long>(sizeof(float)));
  }
}

}  // namespace
#endif

CPUFlashAttention2Op::CPUFlashAttention2Op(const aops::FlashAttention2OpOptions& options) : aops::FlashAttention2Op(options) {}

void CPUFlashAttention2Op::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& Q = inputs[0];
  auto& K = inputs[1];
  auto& V = inputs[2];
  auto& O = outputs[0];

  if (options_.streamed_bhsd) {
#if defined(__ARM_NEON)
    streamedDenseAttention(Q, K, V, O, options_);
#else
    NYI("Streamed FP32 BHSD attention requires ARM NEON");
#endif
    return;
  }

  // Only Support Contiguous Tensor
  MLLM_RT_ASSERT(Q.isContiguous());

  // NOTE:
  //
  // K, V is in flexible layout, no contiguous is OK.
  //
  // MLLM_RT_ASSERT(K.isContiguous());
  // MLLM_RT_ASSERT(V.isContiguous());

  auto B = Q.shape()[0];
  auto S_Q = Q.shape()[1];
  auto H_Q = Q.shape()[2];
  auto D = Q.shape()[3];
  MLLM_RT_ASSERT_EQ(H_Q, options_.q_head);
  auto S_KV = K.shape()[1];
  MLLM_RT_ASSERT_EQ(S_KV, V.shape()[1]);

  switch (Q.dtype()) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      cpu::flash_attn2::fwd_bshd<cpu::flash_attn2::details::__ArmArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t,
                                 mllm_fp32_t>(B, options_.q_head, options_.kv_head, S_Q, S_KV, D, Q.ptr<mllm_fp32_t>(),
                                              K.ptr<mllm_fp32_t>(), V.ptr<mllm_fp32_t>(), O.ptr<mllm_fp32_t>(),
                                              options_.getThreads());
#elif defined(MLLM_HOST_ARCH_X86) || defined(MLLM_HOST_ARCH_X86_64)
      cpu::flash_attn2::fwd_bshd<cpu::flash_attn2::details::__X86ArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t,
                                 mllm_fp32_t>(B, options_.q_head, options_.kv_head, S_Q, S_KV, D, Q.ptr<mllm_fp32_t>(),
                                              K.ptr<mllm_fp32_t>(), V.ptr<mllm_fp32_t>(), O.ptr<mllm_fp32_t>(),
                                              options_.getThreads());
#endif
      break;
    }
    default: NYI("CPUQuickFlashAttention2Op::forward not support dtype {}", nameOfType(Q.dtype())); break;
  }
}

}  // namespace mllm::cpu
