// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "mllm/core/Tensor.hpp"
#include "mllm/nn/Functional.hpp"

namespace mllm::nn::llm_components {

inline Tensor groupedQueryAttentionEager(const Tensor& query, const Tensor& key, const Tensor& value) {
  const auto q_shape = query.shape();
  const auto k_shape = key.shape();
  const auto v_shape = value.shape();
  const int32_t groups = q_shape[1] / k_shape[1];
  const float scale = 1.0F / std::sqrt(static_cast<float>(q_shape[3]));
  const int32_t context_offset = k_shape[2] - q_shape[2];
  auto causal_mask = Tensor::zeros({q_shape[0], 1, q_shape[2], k_shape[2]}, kFloat32, kCPU);
  auto* mask_data = causal_mask.ptr<float>();
  for (int32_t batch = 0; batch < q_shape[0]; ++batch) {
    for (int32_t query_index = 0; query_index < q_shape[2]; ++query_index) {
      const int32_t first_masked_key = context_offset + query_index + 1;
      for (int32_t key_index = first_masked_key; key_index < k_shape[2]; ++key_index) {
        const auto offset = ((static_cast<size_t>(batch) * q_shape[2] + query_index) * k_shape[2]) + key_index;
        mask_data[offset] = -1.0e10F;
      }
    }
  }

  auto output = Tensor::zeros({q_shape[0], q_shape[1], q_shape[2], v_shape[3]}, value.dtype(), kCPU);
  const size_t output_head_bytes = static_cast<size_t>(q_shape[2]) * static_cast<size_t>(v_shape[3])
                                   * static_cast<size_t>(bytesOfType(value.dtype()))
                                   / static_cast<size_t>(lanesOfType(value.dtype()));

  for (int32_t query_head = 0; query_head < q_shape[1]; ++query_head) {
    const int32_t kv_head = query_head / groups;
    auto q = query[{kAll, {query_head}, kAll, kAll}].contiguous();
    auto k = key[{kAll, {kv_head}, kAll, kAll}].contiguous();
    auto v = value[{kAll, {kv_head}, kAll, kAll}].contiguous();

    Tensor attention;
    if (query.dtype() == kFloat32) {
      attention = functional::matmul(q, k, false, true) * scale;
      attention = functional::softmax(attention + causal_mask, -1);
    } else {
      attention = functional::matmul(q.to(kFloat32), k.to(kFloat32), false, true) * scale;
      attention = functional::softmax(attention + causal_mask, -1).to(kFloat16);
    }
    auto head_output = functional::matmul(attention, v).contiguous();
    for (int32_t batch = 0; batch < q_shape[0]; ++batch) {
      auto* destination = output.offsettedPtr<mllm_byte_t>({batch, query_head, 0, 0});
      const auto* source = head_output.coffsettedPtr<mllm_byte_t>({batch, 0, 0, 0});
      std::memcpy(destination, source, output_head_bytes);
    }
  }
  return output;
}

// Correctness-first eager GQA for [B, H, S, D] tensors. Each query head reads
// its shared KV head directly, so the persistent or temporary full KV history
// is never expanded to query-head count.
inline Tensor groupedQueryAttention(const Tensor& query, const Tensor& key, const Tensor& value) {
  if (query.isNil() || key.isNil() || value.isNil()) {
    throw std::invalid_argument("groupedQueryAttention inputs must not be nil");
  }
  const auto q_shape = query.shape();
  const auto k_shape = key.shape();
  const auto v_shape = value.shape();
  if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4 || q_shape[0] <= 0 || q_shape[0] != k_shape[0]
      || q_shape[0] != v_shape[0] || q_shape[1] <= 0 || k_shape[1] <= 0 || k_shape[1] != v_shape[1]
      || q_shape[1] % k_shape[1] != 0 || q_shape[2] <= 0 || k_shape[2] <= 0 || k_shape[2] != v_shape[2] || q_shape[3] <= 0
      || q_shape[3] != k_shape[3] || v_shape[3] <= 0) {
    throw std::invalid_argument("groupedQueryAttention expects compatible [B, query_heads/KV_heads, sequence, dim] tensors");
  }
  if (query.dtype() != key.dtype() || query.dtype() != value.dtype()) {
    throw std::invalid_argument("groupedQueryAttention inputs must have the same dtype");
  }
  if (query.device() != key.device() || query.device() != value.device()) {
    throw std::invalid_argument("groupedQueryAttention inputs must be on the same device");
  }
  if (query.device() != kCPU) { throw std::invalid_argument("groupedQueryAttention currently supports CPU only"); }
  if (query.dtype() != kFloat32 && query.dtype() != kFloat16) {
    throw std::invalid_argument("groupedQueryAttention supports float32 and float16 only");
  }

  const int32_t context_offset = k_shape[2] - q_shape[2];
  if (context_offset < 0) { throw std::invalid_argument("groupedQueryAttention key sequence cannot be shorter than query"); }

  if (query.dtype() == kFloat32 && q_shape[2] == 1) { return functional::groupedQueryAttentionDecode(query, key, value); }

  return groupedQueryAttentionEager(query, key, value);
}

}  // namespace mllm::nn::llm_components
