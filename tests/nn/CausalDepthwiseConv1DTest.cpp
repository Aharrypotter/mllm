// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
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

class CausalDepthwiseConv1DTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

class CausalDepthwiseConv1DModule final : public mllm::nn::Module {
 public:
  CausalDepthwiseConv1DModule(std::string name, bool bias, bool state_inplace) : Module(std::move(name)), bias_(bias) {
    conv_ = reg<mllm::nn::CausalDepthwiseConv1D>("conv", 2, 3, bias, state_inplace,
                                                 mllm::aops::CausalDepthwiseConv1DAccumulationOrder::kHistoryFirst);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<mllm::AnyValue>&) override {
    auto [output, state] = conv_(inputs[0], inputs[1]);
    return {output, state};
  }

  [[nodiscard]] bool hasBias() const { return bias_; }

 private:
  mllm::nn::CausalDepthwiseConv1D conv_;
  bool bias_;
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

Tensor tensor(const Tensor::shape_t& shape, const std::vector<float>& values) {
  return Tensor::fromVector(values, shape, mllm::kFloat32, mllm::kCPU);
}

Tensor parameter(const std::string& name, const Tensor::shape_t& shape, const std::vector<float>& values) {
  auto result = Tensor::empty(shape, mllm::kFloat32, mllm::kCPU).setMemType(mllm::kParamsNormal).setName(name).alloc();
  std::copy(values.begin(), values.end(), result.ptr<float>());
  return result;
}

void loadParameters(CausalDepthwiseConv1DModule& module, const std::string& module_name, const std::vector<float>& weights,
                    const std::vector<float>& bias = {}) {
  auto parameters = mllm::ParameterFile::create();
  parameters->push(module_name + ".conv.weight", parameter(module_name + ".conv.weight", {2, 1, 3}, weights));
  if (module.hasBias()) { parameters->push(module_name + ".conv.bias", parameter(module_name + ".conv.bias", {2}, bias)); }
  module.load(parameters);
}

std::pair<std::vector<float>, std::vector<float>> referenceHistoryFirstK3(const std::vector<float>& input,
                                                                          const std::vector<float>& weights,
                                                                          std::vector<float> state,
                                                                          const std::vector<float>& bias, int batch,
                                                                          int sequence, int channels) {
  constexpr int kKernelSize = 3;
  constexpr int kHistorySize = kKernelSize - 1;
  std::vector<float> output(static_cast<size_t>(batch) * sequence * channels);
  for (int batch_index = 0; batch_index < batch; ++batch_index) {
    for (int token = 0; token < sequence; ++token) {
      for (int channel = 0; channel < channels; ++channel) {
        const auto state_base = (static_cast<size_t>(batch_index) * channels + channel) * kHistorySize;
        const auto element = (static_cast<size_t>(batch_index) * sequence + token) * channels + channel;
        const auto weight_base = static_cast<size_t>(channel) * kKernelSize;
        float value = 0.0F;
        for (int tap = 0; tap < kHistorySize; ++tap) { value += state[state_base + tap] * weights[weight_base + tap]; }
        value += input[element] * weights[weight_base + kHistorySize];
        output[element] = value + (bias.empty() ? 0.0F : bias[channel]);
        state[state_base] = state[state_base + 1];
        state[state_base + 1] = input[element];
      }
    }
  }
  return {output, state};
}

void expectExact(const Tensor& actual, const std::vector<float>& expected) {
  ASSERT_EQ(actual.numel(), expected.size());
  EXPECT_EQ(actual.toVector<float>(), expected);
}

TEST_F(CausalDepthwiseConv1DTest, EagerMatchesReferenceAndMutatesStateInPlace) {
  constexpr char kModuleName[] = "causal_depthwise_conv_eager";
  const std::vector<float> weights = {0.5F, 0.25F, 2.0F, -1.0F, 0.5F, -0.25F};
  const std::vector<float> bias = {0.125F, -0.25F};
  const std::vector<float> input_values = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const std::vector<float> initial_state = {0.5F, -0.5F, 1.0F, -1.0F};
  auto module = CausalDepthwiseConv1DModule(kModuleName, true, true);
  loadParameters(module, kModuleName, weights, bias);
  auto input = tensor({1, 3, 2}, input_values);
  auto state = tensor({1, 2, 2}, initial_state);
  const auto* state_storage = state.ptr<float>();

  const auto outputs = module(input, state);
  const auto [expected_output, expected_state] = referenceHistoryFirstK3(input_values, weights, initial_state, bias, 1, 3, 2);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].shape(), input.shape());
  EXPECT_EQ(outputs[1].shape(), state.shape());
  EXPECT_EQ(outputs[1].ptr<float>(), state_storage);
  expectExact(outputs[0], expected_output);
  expectExact(outputs[1], expected_state);
}

TEST_F(CausalDepthwiseConv1DTest, NonInplaceOutputPreservesInputState) {
  constexpr char kModuleName[] = "causal_depthwise_conv_copy_state";
  const std::vector<float> weights = {0.5F, 0.25F, 2.0F, -1.0F, 0.5F, -0.25F};
  const std::vector<float> input_values = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> initial_state = {0.5F, -0.5F, 1.0F, -1.0F};
  auto module = CausalDepthwiseConv1DModule(kModuleName, false, false);
  loadParameters(module, kModuleName, weights);
  auto input = tensor({1, 2, 2}, input_values);
  auto state = tensor({1, 2, 2}, initial_state);
  const auto* state_storage = state.ptr<float>();

  const auto outputs = module(input, state);
  const auto [expected_output, expected_state] = referenceHistoryFirstK3(input_values, weights, initial_state, {}, 1, 2, 2);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_NE(outputs[1].ptr<float>(), state_storage);
  expectExact(state, initial_state);
  expectExact(outputs[0], expected_output);
  expectExact(outputs[1], expected_state);
}

TEST_F(CausalDepthwiseConv1DTest, ChunkedExecutionMatchesOneShot) {
  const std::vector<float> weights = {0.5F, 0.25F, 2.0F, -1.0F, 0.5F, -0.25F};
  const std::vector<float> full_input = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
  const std::vector<float> initial_state = {0.5F, -0.5F, 1.0F, -1.0F};

  auto one_shot = CausalDepthwiseConv1DModule("causal_depthwise_conv_one_shot", false, true);
  loadParameters(one_shot, "causal_depthwise_conv_one_shot", weights);
  const auto one_shot_outputs = one_shot(tensor({1, 4, 2}, full_input), tensor({1, 2, 2}, initial_state));

  auto chunked = CausalDepthwiseConv1DModule("causal_depthwise_conv_chunked", false, true);
  loadParameters(chunked, "causal_depthwise_conv_chunked", weights);
  auto chunk_state = tensor({1, 2, 2}, initial_state);
  const auto first = chunked(tensor({1, 2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}), chunk_state);
  const auto second = chunked(tensor({1, 2, 2}, {5.0F, 6.0F, 7.0F, 8.0F}), first[1]);

  const auto expected_output = one_shot_outputs[0].toVector<float>();
  const auto first_output = first[0].toVector<float>();
  const auto second_output = second[0].toVector<float>();
  EXPECT_TRUE(std::equal(first_output.begin(), first_output.end(), expected_output.begin()));
  EXPECT_TRUE(std::equal(second_output.begin(), second_output.end(), expected_output.begin() + first_output.size()));
  EXPECT_EQ(second[1].toVector<float>(), one_shot_outputs[1].toVector<float>());
}

TEST_F(CausalDepthwiseConv1DTest, RejectsInvalidStateGeometry) {
  auto module = CausalDepthwiseConv1DModule("causal_depthwise_conv_invalid", false, true);
  EXPECT_THROW((void)module(tensor({1, 2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}), tensor({1, 2, 1}, {0.0F, 0.0F})),
               std::invalid_argument);
}

TEST_F(CausalDepthwiseConv1DTest, TraceAndSerializationPreserveStateSemantics) {
  CausalDepthwiseConv1DModule module("causal_depthwise_conv_trace", true, true);
  auto ir_context = mllm::ir::trace(module, Tensor::empty({1, 2, 2}, mllm::kFloat32, mllm::kCPU),
                                    Tensor::empty({1, 2, 2}, mllm::kFloat32, mllm::kCPU));
  auto op = findOp<mllm::ir::linalg::CausalDepthwiseConv1DOp>(ir_context->topLevelOp());
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->getAOp()->getOpType(), mllm::OpTypes::kCausalDepthwiseConv1D);

  const auto options = mllm::jit::binary::dumpLinalgIROptions(op);
  EXPECT_EQ(options.at("channels"), 2);
  EXPECT_EQ(options.at("kernel_size"), 3);
  EXPECT_EQ(options.at("bias"), true);
  EXPECT_EQ(options.at("state_inplace"), true);
  EXPECT_EQ(options.at("accumulation_order"), "HistoryFirst");

  const auto restored = mllm::jit::interpreter::aopsFromJson(
      nlohmann::json{{"op_type", "CausalDepthwiseConv1D"}, {"backend", "CPU"}, {"op_options", options}});
  ASSERT_NE(restored, nullptr);
  ASSERT_EQ(restored->getOpType(), mllm::OpTypes::kCausalDepthwiseConv1D);
  const auto restored_options = std::static_pointer_cast<mllm::aops::CausalDepthwiseConv1DOp>(restored)->options();
  EXPECT_EQ(restored_options.channels, 2);
  EXPECT_EQ(restored_options.kernel_size, 3);
  EXPECT_TRUE(restored_options.bias);
  EXPECT_TRUE(restored_options.state_inplace);
  EXPECT_EQ(restored_options.accumulation_order, mllm::aops::CausalDepthwiseConv1DAccumulationOrder::kHistoryFirst);
}

}  // namespace
