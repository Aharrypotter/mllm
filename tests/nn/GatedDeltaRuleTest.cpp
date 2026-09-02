// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mllm/compile/ir/Trace.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/jit/binary/LinalgIRSerialization.hpp"
#include "mllm/compile/jit/interpreter/AopsFromJson.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"

namespace {

using mllm::Tensor;

class GatedDeltaRuleTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

class GatedDeltaRuleModule final : public mllm::nn::Module {
 public:
  GatedDeltaRuleModule(std::string name, bool state_inplace) : Module(std::move(name)) {
    gated_delta_rule_ = reg<mllm::nn::GatedDeltaRule>("gated_delta_rule", state_inplace);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<mllm::AnyValue>&) override {
    auto [output, state] =
        gated_delta_rule_(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7]);
    return {output, state};
  }

 private:
  mllm::nn::GatedDeltaRule gated_delta_rule_;
};

template<typename OpType>
auto findOp(const mllm::ir::node_ptr_t& node) -> typename OpType::ptr_t {
  if (node->isa_<OpType>()) { return node->cast_<OpType>(); }
  if (!node->isa_<mllm::ir::Op>()) { return nullptr; }
  for (const auto& region : node->cast_<mllm::ir::Op>()->regions()) {
    for (const auto& op : region->ops()) {
      if (auto found = findOp<OpType>(op)) { return found; }
    }
  }
  return nullptr;
}

Tensor patterned(const Tensor::shape_t& shape, float scale, float offset = 0.0F) {
  auto tensor = Tensor::empty(shape, mllm::kFloat32, mllm::kCPU).alloc();
  for (int index = 0; index < tensor.numel(); ++index) {
    tensor.ptr<float>()[index] = std::sin(static_cast<float>(index + 1) * scale) + offset;
  }
  return tensor;
}

float stableSoftplus(float value) {
  if (value > 20.0F) { return value; }
  if (value < -20.0F) { return std::exp(value); }
  return std::log1p(std::exp(value));
}

float stableSigmoid(float value) {
  if (value >= 0.0F) {
    const float exp_value = std::exp(-value);
    return 1.0F / (1.0F + exp_value);
  }
  const float exp_value = std::exp(value);
  return exp_value / (1.0F + exp_value);
}

std::pair<std::vector<float>, std::vector<float>> referenceGatedDeltaRule(
    const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, const std::vector<float>& a,
    const std::vector<float>& b, const std::vector<float>& a_log, const std::vector<float>& dt_bias, std::vector<float> state,
    int batch, int sequence, int key_heads, int value_heads, int key_dim, int value_dim) {
  const int key_head_repeats = value_heads / key_heads;
  const float query_dim_scale = 1.0F / std::sqrt(static_cast<float>(key_dim));
  std::vector<float> output(static_cast<std::size_t>(batch) * sequence * value_heads * value_dim);
  std::vector<float> normalized_query(key_dim);
  std::vector<float> normalized_key(key_dim);

  for (int batch_index = 0; batch_index < batch; ++batch_index) {
    for (int value_head = 0; value_head < value_heads; ++value_head) {
      const int key_head = value_head / key_head_repeats;
      const std::size_t state_base = (static_cast<std::size_t>(batch_index) * value_heads + value_head) * value_dim * key_dim;
      for (int token = 0; token < sequence; ++token) {
        const std::size_t qk_base =
            ((static_cast<std::size_t>(batch_index) * sequence + token) * key_heads + key_head) * key_dim;
        float query_norm_sq = 0.0F;
        float key_norm_sq = 0.0F;
        for (int dim = 0; dim < key_dim; ++dim) {
          query_norm_sq += q[qk_base + dim] * q[qk_base + dim];
          key_norm_sq += k[qk_base + dim] * k[qk_base + dim];
        }
        const float query_scale = query_dim_scale / std::sqrt(query_norm_sq + 1.0e-6F);
        const float key_scale = 1.0F / std::sqrt(key_norm_sq + 1.0e-6F);
        for (int dim = 0; dim < key_dim; ++dim) {
          normalized_query[dim] = q[qk_base + dim] * query_scale;
          normalized_key[dim] = k[qk_base + dim] * key_scale;
        }

        const std::size_t gate_index = (static_cast<std::size_t>(batch_index) * sequence + token) * value_heads + value_head;
        const float gate = -std::exp(a_log[value_head]) * stableSoftplus(a[gate_index] + dt_bias[value_head]);
        const float decay = std::exp(gate);
        const float beta = stableSigmoid(b[gate_index]);
        const std::size_t value_base = gate_index * value_dim;

        for (int value_index = 0; value_index < value_dim; ++value_index) {
          const std::size_t state_row = state_base + static_cast<std::size_t>(value_index) * key_dim;
          float state_dot_key = 0.0F;
          for (int dim = 0; dim < key_dim; ++dim) {
            state[state_row + dim] *= decay;
            state_dot_key += state[state_row + dim] * normalized_key[dim];
          }
          const float delta = (v[value_base + value_index] - state_dot_key) * beta;
          float result = 0.0F;
          for (int dim = 0; dim < key_dim; ++dim) {
            state[state_row + dim] += delta * normalized_key[dim];
            result += state[state_row + dim] * normalized_query[dim];
          }
          output[value_base + value_index] = result;
        }
      }
    }
  }
  return {output, state};
}

void expectNear(const Tensor& actual, const std::vector<float>& expected, float tolerance = 1.0e-5F) {
  ASSERT_EQ(actual.numel(), expected.size());
  for (int index = 0; index < actual.numel(); ++index) {
    EXPECT_NEAR(actual.ptr<float>()[index], expected[index], tolerance) << "index " << index;
  }
}

TEST_F(GatedDeltaRuleTest, EagerMatchesIndependentGroupedHeadReferenceAndPreservesInputState) {
  constexpr int kBatch = 1;
  constexpr int kSequence = 3;
  constexpr int kKeyHeads = 2;
  constexpr int kValueHeads = 4;
  constexpr int kKeyDim = 4;
  constexpr int kValueDim = 3;
  auto q = patterned({kBatch, kSequence, kKeyHeads, kKeyDim}, 0.03F);
  auto k = patterned(q.shape(), 0.05F);
  auto v = patterned({kBatch, kSequence, kValueHeads, kValueDim}, 0.07F);
  auto a = patterned({kBatch, kSequence, kValueHeads}, 0.09F, -0.2F);
  auto b = patterned({kBatch, kSequence, kValueHeads}, 0.11F);
  auto a_log = patterned({kValueHeads}, 0.13F, -0.4F);
  auto dt_bias = patterned({kValueHeads}, 0.15F, -0.1F);
  auto state = patterned({kBatch, kValueHeads, kValueDim, kKeyDim}, 0.017F);
  const auto state_before = state.toVector<float>();
  const auto [expected_output, expected_state] =
      referenceGatedDeltaRule(q.toVector<float>(), k.toVector<float>(), v.toVector<float>(), a.toVector<float>(),
                              b.toVector<float>(), a_log.toVector<float>(), dt_bias.toVector<float>(), state_before, kBatch,
                              kSequence, kKeyHeads, kValueHeads, kKeyDim, kValueDim);

  GatedDeltaRuleModule module("gated_delta_rule_eager", false);
  const auto outputs = module(q, k, v, a, b, a_log, dt_bias, state);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].shape(), v.shape());
  EXPECT_EQ(outputs[1].shape(), state.shape());
  EXPECT_NE(outputs[1].ptr<float>(), state.ptr<float>());
  EXPECT_EQ(state.toVector<float>(), state_before);
  expectNear(outputs[0], expected_output);
  expectNear(outputs[1], expected_state);
}

TEST_F(GatedDeltaRuleTest, InplaceStateOutputAliasesInput) {
  auto q = patterned({1, 2, 2, 4}, 0.03F);
  auto k = patterned(q.shape(), 0.05F);
  auto v = patterned({1, 2, 4, 3}, 0.07F);
  auto a = patterned({1, 2, 4}, 0.09F, -0.2F);
  auto b = patterned({1, 2, 4}, 0.11F);
  auto a_log = patterned({4}, 0.13F, -0.4F);
  auto dt_bias = patterned({4}, 0.15F, -0.1F);
  auto state = patterned({1, 4, 3, 4}, 0.017F);
  const auto* state_storage = state.ptr<float>();

  GatedDeltaRuleModule module("gated_delta_rule_inplace", true);
  const auto outputs = module(q, k, v, a, b, a_log, dt_bias, state);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[1].ptr<float>(), state_storage);
}

TEST_F(GatedDeltaRuleTest, TraceAndSerializationPreserveStateSemantics) {
  GatedDeltaRuleModule module("gated_delta_rule_trace", true);
  auto ir_context = mllm::ir::trace(
      module, Tensor::empty({1, 2, 2, 4}, mllm::kFloat32, mllm::kCPU), Tensor::empty({1, 2, 2, 4}, mllm::kFloat32, mllm::kCPU),
      Tensor::empty({1, 2, 4, 3}, mllm::kFloat32, mllm::kCPU), Tensor::empty({1, 2, 4}, mllm::kFloat32, mllm::kCPU),
      Tensor::empty({1, 2, 4}, mllm::kFloat32, mllm::kCPU), Tensor::empty({4}, mllm::kFloat32, mllm::kCPU),
      Tensor::empty({4}, mllm::kFloat32, mllm::kCPU), Tensor::empty({1, 4, 3, 4}, mllm::kFloat32, mllm::kCPU));
  auto op = findOp<mllm::ir::linalg::GatedDeltaRuleOp>(ir_context->topLevelOp());
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->getAOp()->getOpType(), mllm::OpTypes::kGatedDeltaRule);
  EXPECT_EQ(op->inputs().size(), 8);
  EXPECT_EQ(op->outputs().size(), 2);

  const auto options = mllm::jit::binary::dumpLinalgIROptions(op);
  EXPECT_TRUE(options.at("state_inplace").get<bool>());
  const auto restored = mllm::jit::interpreter::aopsFromJson(
      nlohmann::json{{"op_type", "GatedDeltaRule"}, {"backend", "CPU"}, {"op_options", options}});
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->getOpType(), mllm::OpTypes::kGatedDeltaRule);
  EXPECT_TRUE(std::static_pointer_cast<mllm::aops::GatedDeltaRuleOp>(restored)->options().state_inplace);
}

}  // namespace
