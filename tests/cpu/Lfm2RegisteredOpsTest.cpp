// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include "mllm/compile/ir/Trace.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/jit/binary/LinalgIRSerialization.hpp"
#include "mllm/compile/jit/interpreter/AopsFromJson.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"

namespace {

using mllm::Tensor;

class Lfm2RegisteredOpsTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

class CausalConvTraceModule final : public mllm::nn::Module {
 public:
  CausalConvTraceModule() : Module("causal_conv_trace") {
    conv_ = reg<mllm::nn::CausalDepthwiseConv1D>("conv", 4, 3, false, true,
                                                 mllm::aops::CausalDepthwiseConv1DAccumulationOrder::kHistoryFirst);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<mllm::AnyValue>&) override {
    auto [output, state] = conv_(inputs[0], inputs[1]);
    return {output, state};
  }

 private:
  mllm::nn::CausalDepthwiseConv1D conv_;
};

class ParallelLinearModule final : public mllm::nn::Module {
 public:
  explicit ParallelLinearModule(std::string name) : Module(std::move(name)) {
    projections_ = reg<mllm::nn::ParallelLinear>(
        "pair", mllm::aops::ParallelLinearOpOptions{.in_channels = 2,
                                                    .out_channels = {2, 1},
                                                    .projection_names = {"left", "right"},
                                                    .bias = false,
                                                    .impl_type = mllm::aops::LinearImplTypes::kGGUF,
                                                    .kai_w4a32_decode_thread_cap = 4,
                                                    .kai_w4a32_prefill_thread_cap = 6});
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<mllm::AnyValue>&) override {
    return projections_(inputs[0]);
  }

 private:
  mllm::nn::ParallelLinear projections_;
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

Tensor parameter(const std::string& name, const Tensor::shape_t& shape, const std::vector<float>& values) {
  auto tensor = Tensor::empty(shape, mllm::kFloat32, mllm::kCPU).setMemType(mllm::kParamsNormal).setName(name).alloc();
  std::copy(values.begin(), values.end(), tensor.ptr<float>());
  return tensor;
}

TEST_F(Lfm2RegisteredOpsTest, CausalConvTracesAndSerializesStateSemantics) {
  CausalConvTraceModule module;
  auto ir_context = mllm::ir::trace(module, Tensor::empty({1, 2, 4}, mllm::kFloat32, mllm::kCPU),
                                    Tensor::empty({1, 4, 2}, mllm::kFloat32, mllm::kCPU));
  auto op = findOp<mllm::ir::linalg::CausalDepthwiseConv1DOp>(ir_context->topLevelOp());
  ASSERT_NE(op, nullptr);
  const auto options = mllm::jit::binary::dumpLinalgIROptions(op);
  EXPECT_EQ(options.at("channels"), 4);
  EXPECT_EQ(options.at("kernel_size"), 3);
  EXPECT_EQ(options.at("state_inplace"), true);
  EXPECT_EQ(options.at("accumulation_order"), "HistoryFirst");

  const auto restored = mllm::jit::interpreter::aopsFromJson(
      nlohmann::json{{"op_type", "CausalDepthwiseConv1D"}, {"backend", "CPU"}, {"op_options", options}});
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->getOpType(), mllm::OpTypes::kCausalDepthwiseConv1D);
}

TEST_F(Lfm2RegisteredOpsTest, ParallelLinearOwnsSiblingParametersAndFallsBackCorrectly) {
  ParallelLinearModule module("parallel_eager");
  auto parameters = mllm::ParameterFile::create();
  parameters->push("parallel_eager.left.weight", parameter("parallel_eager.left.weight", {2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}));
  parameters->push("parallel_eager.right.weight", parameter("parallel_eager.right.weight", {1, 2}, {5.0F, 6.0F}));
  module.load(parameters);

  auto input = Tensor::empty({1, 1, 2}, mllm::kFloat32, mllm::kCPU).alloc();
  input.ptr<float>()[0] = 2.0F;
  input.ptr<float>()[1] = 3.0F;
  const auto outputs = module(input);
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].shape(), (Tensor::shape_t{1, 1, 2}));
  EXPECT_FLOAT_EQ(outputs[0].ptr<float>()[0], 8.0F);
  EXPECT_FLOAT_EQ(outputs[0].ptr<float>()[1], 18.0F);
  EXPECT_FLOAT_EQ(outputs[1].ptr<float>()[0], 28.0F);
}

TEST_F(Lfm2RegisteredOpsTest, ParallelLinearTracesAndSerializesProjectionContract) {
  ParallelLinearModule module("parallel_trace");
  auto ir_context = mllm::ir::trace(module, Tensor::empty({1, 1, 2}, mllm::kFloat32, mllm::kCPU));
  auto op = findOp<mllm::ir::linalg::ParallelLinearOp>(ir_context->topLevelOp());
  ASSERT_NE(op, nullptr);
  const auto options = mllm::jit::binary::dumpLinalgIROptions(op);
  EXPECT_EQ(options.at("out_channels"), (std::vector<int32_t>{2, 1}));
  EXPECT_EQ(options.at("projection_names"), (std::vector<std::string>{"left", "right"}));
  EXPECT_EQ(options.at("kai_w4a32_decode_thread_cap"), 4);
  EXPECT_EQ(options.at("kai_w4a32_prefill_thread_cap"), 6);

  const auto restored = mllm::jit::interpreter::aopsFromJson(
      nlohmann::json{{"op_type", "ParallelLinear"}, {"backend", "CPU"}, {"op_options", options}});
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->getOpType(), mllm::OpTypes::kParallelLinear);
}

}  // namespace
