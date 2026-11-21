// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <vector>

#include "mllm/compile/ir/Trace.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/jit/binary/LinalgIRSerialization.hpp"
#include "mllm/compile/jit/interpreter/AopsFromJson.hpp"
#include "mllm/mllm.hpp"

namespace {

using mllm::Tensor;

class LinearShadowCorrectionTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

class ShadowCorrectionModule final : public mllm::nn::Module {
 public:
  ShadowCorrectionModule() : Module("shadow_test") {
    correction_ = reg<mllm::nn::LinearShadowCorrection>("linear.shadow", 2, 2, 16);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<mllm::AnyValue>& args) override {
    return {correction_(inputs[0], inputs[1], inputs[2])};
  }

 private:
  mllm::nn::LinearShadowCorrection correction_;
};

Tensor floatScalar(const std::string& name, float value) {
  auto tensor = Tensor::empty({1}, mllm::kFloat32, mllm::kCPU).setMemType(mllm::kParamsNormal).setName(name).alloc();
  tensor.ptr<float>()[0] = value;
  return tensor;
}

Tensor int8Scalar(const std::string& name, int8_t value) {
  auto tensor = Tensor::empty({1}, mllm::kInt8, mllm::kCPU).setMemType(mllm::kParamsNormal).setName(name).alloc();
  tensor.ptr<int8_t>()[0] = value;
  return tensor;
}

mllm::ParameterFile::ptr_t parameters(bool clip_input, bool clip_output) {
  auto params = mllm::ParameterFile::create();
  auto weight = Tensor::empty({2, 2}, mllm::kInt8, mllm::kCPU)
                    .setMemType(mllm::kParamsNormal)
                    .setName("shadow_test.linear.weight")
                    .alloc();
  const std::array<int8_t, 4> values = {2, -1, 1, 1};
  std::copy(values.begin(), values.end(), weight.ptr<int8_t>());
  params->push(weight.name(), weight);
  params->push("shadow_test.linear.weight.scale", floatScalar("shadow_test.linear.weight.scale", 1.0F));
  params->push("shadow_test.linear.input_scale", floatScalar("shadow_test.linear.input_scale", 32767.0F));
  params->push("shadow_test.linear.output_scale", floatScalar("shadow_test.linear.output_scale", 32767.0F));
  params->push("shadow_test.linear.clip_input", int8Scalar("shadow_test.linear.clip_input", static_cast<int8_t>(clip_input)));
  params->push("shadow_test.linear.clip_output",
               int8Scalar("shadow_test.linear.clip_output", static_cast<int8_t>(clip_output)));
  return params;
}

Tensor floatTensor(std::array<float, 2> values) {
  auto tensor = Tensor::empty({1, 1, 2}, mllm::kFloat32, mllm::kCPU).alloc();
  std::copy(values.begin(), values.end(), tensor.ptr<float>());
  return tensor;
}

Tensor int16Tensor(std::array<int16_t, 2> values) {
  auto tensor = Tensor::empty({1, 1, 2}, mllm::kInt16, mllm::kCPU).alloc();
  std::copy(values.begin(), values.end(), tensor.ptr<int16_t>());
  return tensor;
}

mllm::ir::linalg::LinearShadowCorrectionOp::ptr_t findShadowCorrection(const mllm::ir::node_ptr_t& node) {
  if (node->isa_<mllm::ir::linalg::LinearShadowCorrectionOp>()) {
    return node->cast_<mllm::ir::linalg::LinearShadowCorrectionOp>();
  }
  if (!node->isa_<mllm::ir::Op>()) { return nullptr; }
  for (const auto& region : node->cast_<mllm::ir::Op>()->regions()) {
    for (const auto& op : region->ops()) {
      if (auto found = findShadowCorrection(op)) { return found; }
    }
  }
  return nullptr;
}

TEST_F(LinearShadowCorrectionTest, RestoresInputTailWithoutMutatingBaseline) {
  ShadowCorrectionModule module;
  module.load(parameters(false, true));
  const auto original = floatTensor({40000.0F, 2.0F});
  const auto quantized = int16Tensor({32767, -32768});
  const auto baseline = floatTensor({100.0F, 200.0F});

  const auto actual = module(original, quantized, baseline)[0];
  EXPECT_FLOAT_EQ(actual.ptr<float>()[0], 14566.0F);
  EXPECT_FLOAT_EQ(actual.ptr<float>()[1], -7033.0F);
  EXPECT_FLOAT_EQ(baseline.ptr<float>()[0], 100.0F);
  EXPECT_FLOAT_EQ(baseline.ptr<float>()[1], 200.0F);
}

TEST_F(LinearShadowCorrectionTest, ReplacesSaturatedOutputWithUnboundedQuantizedDot) {
  ShadowCorrectionModule module;
  module.load(parameters(true, false));
  const auto original = floatTensor({20000.0F, 10000.0F});
  const auto quantized = int16Tensor({32767, 30000});
  const auto baseline = floatTensor({32777.0F, 30020.0F});

  const auto actual = module(original, quantized, baseline)[0];
  EXPECT_FLOAT_EQ(actual.ptr<float>()[0], 50010.0F);
  EXPECT_FLOAT_EQ(actual.ptr<float>()[1], 30020.0F);
}

TEST_F(LinearShadowCorrectionTest, OutputReplacementDoesNotDoubleCountInputTail) {
  ShadowCorrectionModule module;
  module.load(parameters(false, false));
  const auto original = floatTensor({40000.0F, 2.0F});
  const auto quantized = int16Tensor({32767, -32768});
  const auto baseline = floatTensor({32777.0F, -32748.0F});

  const auto actual = module(original, quantized, baseline)[0];
  EXPECT_FLOAT_EQ(actual.ptr<float>()[0], 80012.0F);
  EXPECT_FLOAT_EQ(actual.ptr<float>()[1], -39978.0F);
}

TEST_F(LinearShadowCorrectionTest, BothClipFlagsMakeAnExactCopy) {
  ShadowCorrectionModule module;
  module.load(parameters(true, true));
  const auto original = floatTensor({40000.0F, 2.0F});
  const auto quantized = int16Tensor({32767, -32768});
  const auto baseline = floatTensor({3.25F, -7.5F});

  const auto actual = module(original, quantized, baseline)[0];
  EXPECT_FLOAT_EQ(actual.ptr<float>()[0], baseline.ptr<float>()[0]);
  EXPECT_FLOAT_EQ(actual.ptr<float>()[1], baseline.ptr<float>()[1]);
  EXPECT_NE(actual.ptr<void>(), baseline.ptr<void>());
}

TEST_F(LinearShadowCorrectionTest, HalfRowsMatchPrematerializedFloatForEachClipMode) {
  auto half_rows = Tensor::empty({1, 2, 2}, mllm::kFloat16, mllm::kCPU).alloc();
  auto float_rows = Tensor::empty({1, 2, 2}, mllm::kFloat32, mllm::kCPU).alloc();
  auto quantized = Tensor::empty({1, 2, 2}, mllm::kInt16, mllm::kCPU).alloc();
  auto baseline = Tensor::empty({1, 2, 2}, mllm::kFloat32, mllm::kCPU).alloc();
  const std::array<float, 4> values = {40000.0F, 2.0F, -40000.0F, -2.0F};
  const std::array<int16_t, 4> saturated_first_row = {32767, -32768, 1000, -2000};
  const std::array<float, 4> residual = {32777.0F, -32748.0F, 1010.0F, -1980.0F};
  std::copy(saturated_first_row.begin(), saturated_first_row.end(), quantized.ptr<int16_t>());
  std::copy(residual.begin(), residual.end(), baseline.ptr<float>());
  for (size_t i = 0; i < values.size(); ++i) {
    half_rows.ptr<mllm::mllm_fp16_t>()[i] = static_cast<mllm::mllm_fp16_t>(values[i]);
    float_rows.ptr<float>()[i] = static_cast<float>(half_rows.ptr<mllm::mllm_fp16_t>()[i]);
  }

  for (bool clip_input : {false, true}) {
    for (bool clip_output : {false, true}) {
      ShadowCorrectionModule module;
      module.load(parameters(clip_input, clip_output));
      const auto expected = module(float_rows, quantized, baseline).front();
      const auto actual = module(half_rows, quantized, baseline).front();
      for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(std::bit_cast<uint32_t>(actual.ptr<float>()[i]), std::bit_cast<uint32_t>(expected.ptr<float>()[i]));
        EXPECT_EQ(baseline.ptr<float>()[i], residual[i]);
      }
    }
  }
}

TEST_F(LinearShadowCorrectionTest, TraceAndSerializationRoundTripPreserveContract) {
  ShadowCorrectionModule module;
  module.load(parameters(false, false));
  auto ir_ctx =
      mllm::ir::trace(module, Tensor::empty({1, 1, 2}, mllm::kFloat32, mllm::kCPU),
                      Tensor::empty({1, 1, 2}, mllm::kInt16, mllm::kCPU), Tensor::empty({1, 1, 2}, mllm::kFloat32, mllm::kCPU));
  auto ir_op = findShadowCorrection(ir_ctx->topLevelOp());
  ASSERT_NE(ir_op, nullptr);
  ASSERT_NE(ir_op->getAOp(), nullptr);
  EXPECT_EQ(ir_op->getAOp()->getOpType(), mllm::OpTypes::kLinearShadowCorrection);

  const auto options = mllm::jit::binary::dumpLinalgIROptions(ir_op);
  EXPECT_EQ(options.at("in_channels"), 2);
  EXPECT_EQ(options.at("out_channels"), 2);
  EXPECT_EQ(options.at("activation_bits"), 16);
  const nlohmann::json encoded = {{"op_type", "LinearShadowCorrection"}, {"backend", "CPU"}, {"op_options", options}};
  const auto restored = mllm::jit::interpreter::aopsFromJson(encoded);
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->getOpType(), mllm::OpTypes::kLinearShadowCorrection);
  const auto restored_shadow = std::dynamic_pointer_cast<mllm::aops::LinearShadowCorrectionOp>(restored);
  ASSERT_NE(restored_shadow, nullptr);
  EXPECT_EQ(restored_shadow->options().in_channels, 2);
  EXPECT_EQ(restored_shadow->options().out_channels, 2);
  EXPECT_EQ(restored_shadow->options().activation_bits, 16);
}

}  // namespace
