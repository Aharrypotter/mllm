// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/llm_components/GroupedQueryAttention.hpp"

namespace {

using mllm::Tensor;

class GroupedQueryAttentionTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

Tensor sequential(const Tensor::shape_t& shape, float scale) {
  auto tensor = Tensor::empty(shape, mllm::kFloat32, mllm::kCPU).alloc();
  for (int index = 0; index < tensor.numel(); ++index) {
    tensor.ptr<float>()[index] = std::sin(static_cast<float>(index + 1) * scale);
  }
  return tensor;
}

Tensor gqaReference(const Tensor& query, const Tensor& key, const Tensor& value) {
  const auto q_shape = query.shape();
  const auto k_shape = key.shape();
  const auto v_shape = value.shape();
  const int32_t groups = q_shape[1] / k_shape[1];
  const int32_t context_offset = k_shape[2] - q_shape[2];
  const float scale = 1.0F / std::sqrt(static_cast<float>(q_shape[3]));
  auto output = Tensor::zeros({q_shape[0], q_shape[1], q_shape[2], v_shape[3]}, mllm::kFloat32, mllm::kCPU);

  for (int32_t batch = 0; batch < q_shape[0]; ++batch) {
    for (int32_t query_head = 0; query_head < q_shape[1]; ++query_head) {
      const int32_t kv_head = query_head / groups;
      for (int32_t query_index = 0; query_index < q_shape[2]; ++query_index) {
        const int32_t allowed_keys = context_offset + query_index + 1;
        std::vector<float> probabilities(static_cast<size_t>(allowed_keys));
        float maximum = -std::numeric_limits<float>::infinity();
        for (int32_t key_index = 0; key_index < allowed_keys; ++key_index) {
          float score = 0.0F;
          for (int32_t dim = 0; dim < q_shape[3]; ++dim) {
            const auto q_offset =
                (((static_cast<size_t>(batch) * q_shape[1] + query_head) * q_shape[2] + query_index) * q_shape[3]) + dim;
            const auto k_offset =
                (((static_cast<size_t>(batch) * k_shape[1] + kv_head) * k_shape[2] + key_index) * k_shape[3]) + dim;
            score += query.ptr<float>()[q_offset] * key.ptr<float>()[k_offset];
          }
          probabilities[static_cast<size_t>(key_index)] = score * scale;
          maximum = std::max(maximum, probabilities[static_cast<size_t>(key_index)]);
        }
        float denominator = 0.0F;
        for (auto& probability : probabilities) {
          probability = std::exp(probability - maximum);
          denominator += probability;
        }
        for (int32_t value_dim = 0; value_dim < v_shape[3]; ++value_dim) {
          float result = 0.0F;
          for (int32_t key_index = 0; key_index < allowed_keys; ++key_index) {
            const auto v_offset =
                (((static_cast<size_t>(batch) * v_shape[1] + kv_head) * v_shape[2] + key_index) * v_shape[3]) + value_dim;
            result += probabilities[static_cast<size_t>(key_index)] / denominator * value.ptr<float>()[v_offset];
          }
          const auto output_offset =
              (((static_cast<size_t>(batch) * q_shape[1] + query_head) * q_shape[2] + query_index) * v_shape[3]) + value_dim;
          output.ptr<float>()[output_offset] = result;
        }
      }
    }
  }
  return output;
}

void expectNear(Tensor actual, Tensor expected, float tolerance = 1e-5F) {
  ASSERT_EQ(actual.shape(), expected.shape());
  const auto actual_cpu = actual.to(mllm::kCPU).contiguous();
  const auto expected_cpu = expected.to(mllm::kCPU).contiguous();
  for (int index = 0; index < actual_cpu.numel(); ++index) {
    EXPECT_NEAR(actual_cpu.ptr<float>()[index], expected_cpu.ptr<float>()[index], tolerance) << "index " << index;
  }
}

TEST_F(GroupedQueryAttentionTest, MatchesRepeatedKVReference) {
  auto query = sequential({1, 4, 3, 5}, 0.07F);
  auto key = sequential({1, 2, 3, 5}, 0.11F);
  auto value = sequential({1, 2, 3, 5}, 0.13F);
  const auto actual = mllm::nn::llm_components::groupedQueryAttention(query, key, value);
  const auto expected = gqaReference(query, key, value);

  expectNear(actual, expected);
}

TEST_F(GroupedQueryAttentionTest, SupportsOneKVHeadAndRejectsIllegalGeometry) {
  auto query = sequential({1, 4, 1, 4}, 0.09F);
  auto key = sequential({1, 1, 2, 4}, 0.12F);
  auto value = sequential({1, 1, 2, 3}, 0.15F);
  const auto output = mllm::nn::llm_components::groupedQueryAttention(query, key, value);
  EXPECT_EQ(output.shape(), (Tensor::shape_t{1, 4, 1, 3}));
  expectNear(output, gqaReference(query, key, value));

  auto illegal_query = sequential({1, 3, 1, 4}, 0.09F);
  auto illegal_key = sequential({1, 2, 2, 4}, 0.12F);
  auto illegal_value = sequential({1, 2, 2, 3}, 0.15F);
  EXPECT_THROW(mllm::nn::llm_components::groupedQueryAttention(illegal_query, illegal_key, illegal_value),
               std::invalid_argument);
}

}  // namespace
