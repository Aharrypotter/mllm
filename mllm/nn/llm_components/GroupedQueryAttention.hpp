// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mllm/core/Parallel.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/nn/Functional.hpp"

namespace mllm::nn::llm_components {

inline void validateGroupedQueryAttention(const Tensor& query, const Tensor& key, const Tensor& value) {
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
  if (k_shape[2] < q_shape[2]) {
    throw std::invalid_argument("groupedQueryAttention key sequence cannot be shorter than query");
  }
}

// Direct strided implementation used when a model requires the established
// eager accumulation order. It shares KV heads without materializing an
// expanded cache and parallelizes independent batch/query-head jobs.
inline Tensor groupedQueryAttentionDirectEager(const Tensor& query, const Tensor& key, const Tensor& value) {
  validateGroupedQueryAttention(query, key, value);
  const auto q_shape = query.shape();
  const auto k_shape = key.shape();
  const auto v_shape = value.shape();
  const int32_t groups = q_shape[1] / k_shape[1];
  const float scale = 1.0F / std::sqrt(static_cast<float>(q_shape[3]));
  const int32_t context_offset = k_shape[2] - q_shape[2];

  auto output = Tensor::zeros({q_shape[0], q_shape[1], q_shape[2], v_shape[3]}, value.dtype(), kCPU);
  auto compute = [&]<typename Scalar>() {
    const int32_t jobs = q_shape[0] * q_shape[1];
    std::vector<const Scalar*> query_rows(static_cast<size_t>(jobs) * q_shape[2]);
    std::vector<const Scalar*> key_rows(static_cast<size_t>(q_shape[0]) * k_shape[1] * k_shape[2]);
    std::vector<const Scalar*> value_rows(static_cast<size_t>(q_shape[0]) * v_shape[1] * v_shape[2]);
    std::vector<Scalar*> output_rows(static_cast<size_t>(jobs) * q_shape[2]);
    for (int32_t batch = 0; batch < q_shape[0]; ++batch) {
      for (int32_t head = 0; head < q_shape[1]; ++head) {
        for (int32_t sequence = 0; sequence < q_shape[2]; ++sequence) {
          const size_t row = (static_cast<size_t>(batch) * q_shape[1] + head) * q_shape[2] + sequence;
          query_rows[row] = query.coffsettedPtr<Scalar>({batch, head, sequence, 0});
          output_rows[row] = output.offsettedPtr<Scalar>({batch, head, sequence, 0});
        }
      }
      for (int32_t head = 0; head < k_shape[1]; ++head) {
        for (int32_t sequence = 0; sequence < k_shape[2]; ++sequence) {
          const size_t row = (static_cast<size_t>(batch) * k_shape[1] + head) * k_shape[2] + sequence;
          key_rows[row] = key.coffsettedPtr<Scalar>({batch, head, sequence, 0});
          value_rows[row] = value.coffsettedPtr<Scalar>({batch, head, sequence, 0});
        }
      }
    }
    MLLM_AUTO_PARALLEL_FOR_BEGIN(job, 0, jobs, 1) {
      const int32_t batch = job / q_shape[1];
      const int32_t query_head = job % q_shape[1];
      const int32_t kv_head = query_head / groups;
      std::vector<float> scores(static_cast<size_t>(k_shape[2]));
      for (int32_t query_index = 0; query_index < q_shape[2]; ++query_index) {
        const int32_t visible_keys = context_offset + query_index + 1;
        float maximum = std::numeric_limits<float>::lowest();
        for (int32_t key_index = 0; key_index < visible_keys; ++key_index) {
          float dot = 0.0F;
          const size_t query_row = (static_cast<size_t>(batch) * q_shape[1] + query_head) * q_shape[2] + query_index;
          const size_t key_row = (static_cast<size_t>(batch) * k_shape[1] + kv_head) * k_shape[2] + key_index;
          for (int32_t dim = 0; dim < q_shape[3]; ++dim) {
            dot += static_cast<float>(query_rows[query_row][dim]) * static_cast<float>(key_rows[key_row][dim]);
          }
          scores[key_index] = dot * scale;
          maximum = std::max(maximum, scores[key_index]);
        }

        float denominator = 0.0F;
        for (int32_t key_index = 0; key_index < visible_keys; ++key_index) {
          scores[key_index] = std::exp(scores[key_index] - maximum);
          denominator += scores[key_index];
        }
        for (int32_t value_dim = 0; value_dim < v_shape[3]; ++value_dim) {
          float accumulated = 0.0F;
          for (int32_t key_index = 0; key_index < visible_keys; ++key_index) {
            const size_t value_row = (static_cast<size_t>(batch) * v_shape[1] + kv_head) * v_shape[2] + key_index;
            accumulated += (scores[key_index] / denominator) * static_cast<float>(value_rows[value_row][value_dim]);
          }
          const size_t output_row = (static_cast<size_t>(batch) * q_shape[1] + query_head) * q_shape[2] + query_index;
          output_rows[output_row][value_dim] = static_cast<Scalar>(accumulated);
        }
      }
    }
    MLLM_AUTO_PARALLEL_FOR_END()
  };
  if (query.dtype() == kFloat32) {
    compute.template operator()<float>();
  } else {
    compute.template operator()<half_float::half>();
  }
  return output;
}

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
  validateGroupedQueryAttention(query, key, value);
  const auto q_shape = query.shape();
  const auto k_shape = key.shape();

  if (query.dtype() == kFloat32 && q_shape[2] == 1) { return functional::groupedQueryAttentionDecode(query, key, value); }

  return groupedQueryAttentionEager(query, key, value);
}

}  // namespace mllm::nn::llm_components
