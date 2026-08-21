// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/GroupedQueryAttentionOp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mllm/core/Parallel.hpp"

namespace mllm::cpu {
namespace {

template<typename Scalar>
void groupedQueryAttentionDirectStrided(const Tensor& query, const Tensor& key, const Tensor& value, Tensor& output) {
  const auto q_shape = query.shape();
  const auto k_shape = key.shape();
  const auto v_shape = value.shape();
  const auto q_stride = query.stride();
  const auto k_stride = key.stride();
  const auto v_stride = value.stride();
  const auto o_stride = output.stride();
  const int32_t groups = q_shape[1] / k_shape[1];
  const float scale = 1.0F / std::sqrt(static_cast<float>(q_shape[3]));
  const int32_t context_offset = k_shape[2] - q_shape[2];
  const int32_t jobs = q_shape[0] * q_shape[1];
  // These tables describe arbitrary sequence strides without rebuilding the
  // address expression in the value reduction. Keep their storage per caller
  // thread so steady-state forwards only resize within the retained capacity.
  static thread_local std::vector<const Scalar*> query_rows;
  static thread_local std::vector<const Scalar*> key_rows;
  static thread_local std::vector<const Scalar*> value_rows;
  static thread_local std::vector<Scalar*> output_rows;
  query_rows.resize(static_cast<size_t>(jobs) * q_shape[2]);
  key_rows.resize(static_cast<size_t>(q_shape[0]) * k_shape[1] * k_shape[2]);
  value_rows.resize(static_cast<size_t>(q_shape[0]) * v_shape[1] * v_shape[2]);
  output_rows.resize(static_cast<size_t>(jobs) * q_shape[2]);
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
  const Scalar* const* query_row_data = query_rows.data();
  const Scalar* const* key_row_data = key_rows.data();
  const Scalar* const* value_row_data = value_rows.data();
  Scalar* const* output_row_data = output_rows.data();

  MLLM_AUTO_PARALLEL_FOR_BEGIN(job, 0, jobs, 1) {
    const int32_t batch = job / q_shape[1];
    const int32_t query_head = job % q_shape[1];
    const int32_t kv_head = query_head / groups;
    const size_t query_row_base = (static_cast<size_t>(batch) * q_shape[1] + query_head) * q_shape[2];
    const size_t key_row_base = (static_cast<size_t>(batch) * k_shape[1] + kv_head) * k_shape[2];
    const size_t value_row_base = (static_cast<size_t>(batch) * v_shape[1] + kv_head) * v_shape[2];
    std::vector<float> scores(static_cast<size_t>(k_shape[2]));
    for (int32_t query_index = 0; query_index < q_shape[2]; ++query_index) {
      const int32_t visible_keys = context_offset + query_index + 1;
      const size_t query_row_index = query_row_base + query_index;
      const Scalar* query_row = query_row_data[query_row_index];
      Scalar* output_row = output_row_data[query_row_index];
      float maximum = std::numeric_limits<float>::lowest();
      for (int32_t key_index = 0; key_index < visible_keys; ++key_index) {
        float dot = 0.0F;
        const Scalar* key_row = key_row_data[key_row_base + key_index];
        // Keep the contiguous dot expression separate: folding it into the
        // runtime-stride induction loop changes contraction/codegen, which
        // breaks callers bound to an exact generation-token oracle.
        if (q_stride[3] == 1 && k_stride[3] == 1) {
          for (int32_t dim = 0; dim < q_shape[3]; ++dim) {
            dot += static_cast<float>(query_row[dim]) * static_cast<float>(key_row[dim]);
          }
        } else {
          const Scalar* query_element = query_row;
          const Scalar* key_element = key_row;
          for (int32_t dim = 0; dim < q_shape[3]; ++dim) {
            dot += static_cast<float>(*query_element) * static_cast<float>(*key_element);
            query_element += q_stride[3];
            key_element += k_stride[3];
          }
        }
        scores[key_index] = dot * scale;
        maximum = std::max(maximum, scores[key_index]);
      }

      float denominator = 0.0F;
      for (int32_t key_index = 0; key_index < visible_keys; ++key_index) {
        scores[key_index] = std::exp(scores[key_index] - maximum);
        denominator += scores[key_index];
      }
      const float inverse_denominator = 1.0F / denominator;
      for (int32_t value_dim = 0; value_dim < v_shape[3]; ++value_dim) {
        float accumulated = 0.0F;
        for (int32_t key_index = 0; key_index < visible_keys; ++key_index) {
          accumulated +=
              (scores[key_index] * static_cast<float>(value_row_data[value_row_base + key_index][value_dim * v_stride[3]]))
              * inverse_denominator;
        }
        output_row[value_dim * o_stride[3]] = static_cast<Scalar>(accumulated);
      }
    }
  }
  MLLM_AUTO_PARALLEL_FOR_END()
}

}  // namespace

CPUGroupedQueryAttentionOp::CPUGroupedQueryAttentionOp(const aops::GroupedQueryAttentionOpOptions& options)
    : aops::GroupedQueryAttentionOp(options) {}

void CPUGroupedQueryAttentionOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.implementation != aops::GroupedQueryAttentionImplementation::kDirectStrided) {
    throw std::invalid_argument("Unsupported CPU GroupedQueryAttention implementation");
  }
  if (inputs[0].dtype() == kFloat32) {
    groupedQueryAttentionDirectStrided<float>(inputs[0], inputs[1], inputs[2], outputs[0]);
  } else {
    groupedQueryAttentionDirectStrided<half_float::half>(inputs[0], inputs[1], inputs[2], outputs[0]);
  }
}

}  // namespace mllm::cpu
