// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cmath>

#include "mllm/nn/Functional.hpp"

namespace mllm::models::qwen_npu {

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
  if (key.dtype() != kFloat32) { key = key.contiguous().to(kFloat32); }
  auto scores = nn::functional::matmul(grouped_query, key, false, true);
  auto weights = scores.view({batch, query_heads, query_tokens, context_tokens}) * (1.f / std::sqrt(static_cast<float>(head_dim)));
  weights = softmax(mask(weights));
  auto grouped_weights = weights.view({batch, kv_heads, group_rows, context_tokens});
  if (value.dtype() != weights.dtype()) { value = value.contiguous().to(weights.dtype()); }
  auto output = nn::functional::matmul(grouped_weights, value);
  return output.view({batch, query_heads, query_tokens, head_dim});
}

}  // namespace mllm::models::qwen_npu
