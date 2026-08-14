// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/compile/ir/Trace.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/jit/binary/LinalgIRSerialization.hpp"
#include "mllm/compile/jit/interpreter/AopsFromJson.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/llm_components/GroupedQueryAttention.hpp"

namespace {

using mllm::Tensor;

class GroupedQueryAttentionTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

class GroupedQueryAttentionDecodeTraceModule final : public mllm::nn::Module {
 public:
  GroupedQueryAttentionDecodeTraceModule() : Module("gqa_decode_trace") {}

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<mllm::AnyValue>& args) override {
    return {mllm::nn::llm_components::groupedQueryAttention(inputs[0], inputs[1], inputs[2])};
  }
};

mllm::ir::linalg::GroupedQueryAttentionDecodeOp::ptr_t findGroupedQueryAttentionDecodeOp(const mllm::ir::node_ptr_t& node) {
  if (node->isa_<mllm::ir::linalg::GroupedQueryAttentionDecodeOp>()) {
    return node->cast_<mllm::ir::linalg::GroupedQueryAttentionDecodeOp>();
  }
  if (!node->isa_<mllm::ir::Op>()) { return nullptr; }
  for (const auto& region : node->cast_<mllm::ir::Op>()->regions()) {
    for (const auto& op : region->ops()) {
      if (auto found = findGroupedQueryAttentionDecodeOp(op)) { return found; }
    }
  }
  return nullptr;
}

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
            score += *query.cptrAt<float>({batch, query_head, query_index, dim})
                     * *key.cptrAt<float>({batch, kv_head, key_index, dim});
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
            result += probabilities[static_cast<size_t>(key_index)] / denominator
                      * *value.cptrAt<float>({batch, kv_head, key_index, value_dim});
          }
          *output.ptrAt<float>({batch, query_head, query_index, value_dim}) = result;
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

TEST_F(GroupedQueryAttentionTest, DirectEagerMatchesReference) {
  auto query = sequential({1, 4, 3, 5}, 0.07F);
  auto key = sequential({1, 2, 3, 5}, 0.11F);
  auto value = sequential({1, 2, 3, 5}, 0.13F);
  const auto actual = mllm::nn::llm_components::groupedQueryAttentionDirectEager(query, key, value);
  const auto expected = gqaReference(query, key, value);

  expectNear(actual, expected, 1e-6F);
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

TEST_F(GroupedQueryAttentionTest, DecodeReadsNativeCacheStrideWithoutKVExpansion) {
  auto query = sequential({1, 4, 1, 5}, 0.07F);
  auto key_buffer = sequential({1, 2, 7, 5}, 0.11F);
  auto value_buffer = sequential({1, 2, 7, 3}, 0.13F);
  auto key = key_buffer[{mllm::kAll, mllm::kAll, {mllm::kAll, 4}, mllm::kAll}];
  auto value = value_buffer[{mllm::kAll, mllm::kAll, {mllm::kAll, 4}, mllm::kAll}];

  const auto actual = mllm::nn::llm_components::groupedQueryAttention(query, key, value);
  const auto expected = gqaReference(query, key, value);

  EXPECT_EQ(actual.shape(), (Tensor::shape_t{1, 4, 1, 3}));
  expectNear(actual, expected);
}

TEST_F(GroupedQueryAttentionTest, DecodeFallsBackForNonContiguousHeadDimension) {
  auto query_buffer = sequential({1, 4, 1, 10}, 0.07F);
  auto query = query_buffer[{mllm::kAll, mllm::kAll, mllm::kAll, {0, 10, 2}}];
  auto key = sequential({1, 2, 4, 5}, 0.11F);
  auto value = sequential({1, 2, 4, 3}, 0.13F);

  ASSERT_NE(query.stride()[3], 1);
  const auto actual = mllm::nn::llm_components::groupedQueryAttention(query, key, value);
  const auto expected = gqaReference(query, key, value);

  expectNear(actual, expected);
}

TEST_F(GroupedQueryAttentionTest, DecodeStaysFiniteAtMiniCPM5ProductGeometry) {
  auto query = sequential({1, 16, 1, 128}, 0.007F);
  auto key_buffer = sequential({1, 2, 2048, 128}, 0.011F);
  auto value_buffer = sequential({1, 2, 2048, 128}, 0.013F);
  auto key = key_buffer[{mllm::kAll, mllm::kAll, {mllm::kAll, 201}, mllm::kAll}];
  auto value = value_buffer[{mllm::kAll, mllm::kAll, {mllm::kAll, 201}, mllm::kAll}];

  const auto actual = mllm::nn::llm_components::groupedQueryAttention(query, key, value);
  const auto expected = gqaReference(query, key, value);

  EXPECT_EQ(actual.shape(), (Tensor::shape_t{1, 16, 1, 128}));
  for (int index = 0; index < actual.numel(); ++index) { EXPECT_TRUE(std::isfinite(actual.ptr<float>()[index])); }
  expectNear(actual, expected, 2e-5F);
}

TEST_F(GroupedQueryAttentionTest, DecodeStaysFiniteAtLfm25ProductGeometry) {
  auto query = sequential({1, 32, 1, 64}, 0.007F);
  auto key_buffer = sequential({1, 8, 2048, 64}, 0.011F);
  auto value_buffer = sequential({1, 8, 2048, 64}, 0.013F);
  auto key = key_buffer[{mllm::kAll, mllm::kAll, {mllm::kAll, 201}, mllm::kAll}];
  auto value = value_buffer[{mllm::kAll, mllm::kAll, {mllm::kAll, 201}, mllm::kAll}];

  const auto actual = mllm::nn::llm_components::groupedQueryAttention(query, key, value);
  const auto expected = gqaReference(query, key, value);

  EXPECT_EQ(actual.shape(), (Tensor::shape_t{1, 32, 1, 64}));
  for (int index = 0; index < actual.numel(); ++index) { EXPECT_TRUE(std::isfinite(actual.ptr<float>()[index])); }
  expectNear(actual, expected, 2e-5F);
}

TEST_F(GroupedQueryAttentionTest, DecodeOpTraceAndSerializationRoundTrip) {
  GroupedQueryAttentionDecodeTraceModule module;
  auto ir_ctx = mllm::ir::trace(module, Tensor::empty({1, 4, 1, 5}, mllm::kFloat32, mllm::kCPU),
                                Tensor::empty({1, 2, 4, 5}, mllm::kFloat32, mllm::kCPU),
                                Tensor::empty({1, 2, 4, 3}, mllm::kFloat32, mllm::kCPU));
  auto ir_op = findGroupedQueryAttentionDecodeOp(ir_ctx->topLevelOp());
  ASSERT_NE(ir_op, nullptr);
  ASSERT_NE(ir_op->getAOp(), nullptr);
  EXPECT_EQ(ir_op->getAOp()->getOpType(), mllm::OpTypes::kGroupedQueryAttentionDecode);

  const auto options = mllm::jit::binary::dumpLinalgIROptions(ir_op);
  EXPECT_TRUE(options.empty());
  const nlohmann::json encoded = {{"op_type", "GroupedQueryAttentionDecode"}, {"backend", "CPU"}, {"op_options", options}};
  const auto restored = mllm::jit::interpreter::aopsFromJson(encoded);
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->getOpType(), mllm::OpTypes::kGroupedQueryAttentionDecode);
}

TEST_F(GroupedQueryAttentionTest, SupportsTransposedNonContiguousHeadViews) {
  auto query_bshd = sequential({1, 3, 4, 5}, 0.07F);
  auto key_bshd = sequential({1, 3, 2, 5}, 0.11F);
  auto value_bshd = sequential({1, 3, 2, 5}, 0.13F);
  auto query = query_bshd.transpose(1, 2);
  auto key = key_bshd.transpose(1, 2);
  auto value = value_bshd.transpose(1, 2);

  const auto actual = mllm::nn::llm_components::groupedQueryAttention(query, key, value);
  const auto expected = gqaReference(query.contiguous(), key.contiguous(), value.contiguous());
  expectNear(actual, expected);
}
}  // namespace
