// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/GroupedQueryAttentionDecodeOp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "mllm/backends/cpu/kernels/common/gqa_decode/fwd_bhsd.hpp"

namespace mllm::cpu {
namespace {

void groupedQueryAttentionDecodeFloat32Reference(const Tensor& query, const Tensor& key, const Tensor& value, Tensor& output) {
  const auto q_shape = query.shape();
  const auto k_shape = key.shape();
  const auto v_shape = value.shape();
  const auto q_stride = query.stride();
  const auto k_stride = key.stride();
  const auto v_stride = value.stride();
  const int32_t groups = q_shape[1] / k_shape[1];
  const float scale = 1.0F / std::sqrt(static_cast<float>(q_shape[3]));

  static thread_local std::vector<float> probabilities;
  probabilities.resize(static_cast<size_t>(k_shape[2]));

  for (int32_t batch = 0; batch < q_shape[0]; ++batch) {
    for (int32_t query_head = 0; query_head < q_shape[1]; ++query_head) {
      const int32_t kv_head = query_head / groups;
      const auto* q_head = query.coffsettedPtr<float>({batch, query_head, 0, 0});
      const auto* k_head = key.coffsettedPtr<float>({batch, kv_head, 0, 0});
      const auto* v_head = value.coffsettedPtr<float>({batch, kv_head, 0, 0});
      auto* output_head = output.offsettedPtr<float>({batch, query_head, 0, 0});

      // Android release builds use -ffast-math; a finite sentinel keeps stable
      // softmax valid under finite-math assumptions.
      float maximum = std::numeric_limits<float>::lowest();
      for (int32_t key_index = 0; key_index < k_shape[2]; ++key_index) {
        const auto* key_token = k_head + static_cast<size_t>(key_index) * k_stride[2];
        float score = 0.0F;
        for (int32_t dim = 0; dim < q_shape[3]; ++dim) {
          score += q_head[static_cast<size_t>(dim) * q_stride[3]] * key_token[static_cast<size_t>(dim) * k_stride[3]];
        }
        probabilities[static_cast<size_t>(key_index)] = score * scale;
        maximum = std::max(maximum, probabilities[static_cast<size_t>(key_index)]);
      }

      float denominator = 0.0F;
      for (int32_t key_index = 0; key_index < k_shape[2]; ++key_index) {
        auto& probability = probabilities[static_cast<size_t>(key_index)];
        probability = std::exp(probability - maximum);
        denominator += probability;
      }
      const float inverse_denominator = 1.0F / denominator;
      for (int32_t key_index = 0; key_index < k_shape[2]; ++key_index) {
        probabilities[static_cast<size_t>(key_index)] *= inverse_denominator;
      }

      for (int32_t value_dim = 0; value_dim < v_shape[3]; ++value_dim) {
        float result = 0.0F;
        for (int32_t key_index = 0; key_index < k_shape[2]; ++key_index) {
          const auto* value_token = v_head + static_cast<size_t>(key_index) * v_stride[2];
          result += probabilities[static_cast<size_t>(key_index)] * value_token[static_cast<size_t>(value_dim) * v_stride[3]];
        }
        output_head[value_dim] = result;
      }
    }
  }
}

}  // namespace

CPUGroupedQueryAttentionDecodeOp::CPUGroupedQueryAttentionDecodeOp(const aops::GroupedQueryAttentionDecodeOpOptions& options)
    : aops::GroupedQueryAttentionDecodeOp(options) {}

void CPUGroupedQueryAttentionDecodeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& query = inputs[0];
  const auto& key = inputs[1];
  const auto& value = inputs[2];
  auto& output = outputs[0];
  const auto q_shape = query.shape();
  const auto k_shape = key.shape();
  const auto v_shape = value.shape();
  const auto q_stride = query.stride();
  const auto k_stride = key.stride();
  const auto v_stride = value.stride();
  const auto output_stride = output.stride();

  static thread_local std::vector<float> probabilities;
  const int32_t group_size = q_shape[1] / k_shape[1];
  probabilities.resize(static_cast<size_t>(group_size) * k_shape[2]);

  const bool completed = cpu::gqa_decode::fwdBhsdFp32(
      q_shape[0], q_shape[1], k_shape[1], k_shape[2], q_shape[3], v_shape[3], query.ptr<float>(),
      {q_stride[0], q_stride[1], q_stride[2], q_stride[3]}, key.ptr<float>(),
      {k_stride[0], k_stride[1], k_stride[2], k_stride[3]}, value.ptr<float>(),
      {v_stride[0], v_stride[1], v_stride[2], v_stride[3]}, output.ptr<float>(),
      {output_stride[0], output_stride[1], output_stride[2], output_stride[3]}, probabilities.data());
  if (!completed) { groupedQueryAttentionDecodeFloat32Reference(query, key, value, output); }
}

}  // namespace mllm::cpu
