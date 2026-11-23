// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <limits>
#include <mutex>

#include "mllm/backends/cpu/kernels/arm/cast_types.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::models::qwen_npu {

inline Tensor prepareDenseKV(Tensor states, DataTypes dtype) {
  if (states.dtype() == dtype) { return states; }
#if defined(MLLM_HOST_ARCH_ARM64)
  if (states.device() == kCPU && states.dtype() == kFloat16 && dtype == kFloat32) {
    const auto shape = states.shape();
    const auto stride = states.stride();
    const int64_t head_elements = static_cast<int64_t>(shape[2]) * shape[3];
    if (shape[0] == 1 && stride[2] == shape[3] && stride[3] == 1
        && head_elements <= std::numeric_limits<int>::max()) {
      // Cast the live rows of each head directly from the cache's capacity stride.
      auto expanded = Tensor::empty(shape, dtype, kCPU).alloc();
      for (int32_t head = 0; head < shape[1]; ++head) {
        const auto* source = states.cptrAt<mllm_fp16_t>({0, head, 0, 0});
        auto* destination = expanded.ptrAt<mllm_fp32_t>({0, head, 0, 0});
        cpu::arm::fp16_2_fp32(source, destination, static_cast<int>(head_elements), 1);
      }
      static std::once_flag log_first_expansion;
      std::call_once(log_first_expansion, [&] {
        MLLM_INFO("SHADOWNPU_NATIVE_FP16_DENSE_EXPAND heads={} head_stride={}", shape[1], stride[1]);
      });
      return expanded;
    }
  }
#endif
  return states.contiguous().to(dtype);
}

template <typename Mask, typename Softmax>
inline Tensor nativeKVDenseAttention(Tensor query, Tensor key, Tensor value, Mask& mask, Softmax& softmax) {
  const auto query_shape = query.shape();
  const auto key_shape = key.shape();
  const int batch = query_shape[0];
  const int query_heads = query_shape[1];
  const int query_tokens = query_shape[2];
  const int head_dim = query_shape[3];
  const int kv_heads = key_shape[1];
  const int context_tokens = key_shape[2];
  const int group_rows = query_heads / kv_heads * query_tokens;

  // Fold query heads into GEMM rows within each KV group. The causal mask
  // operates on the restored query-head dimension and the original token rows.
  auto grouped_query = query.to(kFloat32).contiguous().view({batch, kv_heads, group_rows, head_dim});
  auto scores = nn::functional::matmul(grouped_query, prepareDenseKV(key, kFloat32), false, true);
  auto weights = scores.view({batch, query_heads, query_tokens, context_tokens}) * (1.f / std::sqrt(static_cast<float>(head_dim)));
  weights = softmax(mask(weights));
  auto grouped_weights = weights.view({batch, kv_heads, group_rows, context_tokens});
  auto output = nn::functional::matmul(grouped_weights, prepareDenseKV(value, weights.dtype()));
  return output.view({batch, query_heads, query_tokens, head_dim});
}

}  // namespace mllm::models::qwen_npu
