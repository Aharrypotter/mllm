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
#include "mllm/core/aops/ParallelLinearOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"

namespace {

using mllm::Tensor;

class ParallelLinearTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

class ParallelLinearModule final : public mllm::nn::Module {
 public:
  ParallelLinearModule(std::string name, bool bias) : Module(std::move(name)) {
    projections_ =
        reg<mllm::nn::ParallelLinear>("pair", 2, std::vector<int32_t>{2, 1}, std::vector<std::string>{"left", "right"}, bias,
                                      mllm::aops::LinearImplTypes::kGGUF, 4, 6);
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
  auto result = Tensor::empty(shape, mllm::kFloat32, mllm::kCPU).setMemType(mllm::kParamsNormal).setName(name).alloc();
  std::copy(values.begin(), values.end(), result.ptr<float>());
  return result;
}

TEST_F(ParallelLinearTest, EagerOwnsSiblingParametersAndMatchesIndependentProjections) {
  ParallelLinearModule module("parallel_linear_eager", true);
  auto parameters = mllm::ParameterFile::create();
  parameters->push("parallel_linear_eager.left.weight",
                   parameter("parallel_linear_eager.left.weight", {2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}));
  parameters->push("parallel_linear_eager.left.bias", parameter("parallel_linear_eager.left.bias", {2}, {0.5F, -0.5F}));
  parameters->push("parallel_linear_eager.right.weight", parameter("parallel_linear_eager.right.weight", {1, 2}, {5.0F, 6.0F}));
  parameters->push("parallel_linear_eager.right.bias", parameter("parallel_linear_eager.right.bias", {1}, {1.0F}));
  module.load(parameters);

  auto input = Tensor::fromVector<float>({2.0F, 3.0F}, {1, 1, 2}, mllm::kFloat32, mllm::kCPU);
  const auto outputs = module(input);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].shape(), (Tensor::shape_t{1, 1, 2}));
  EXPECT_EQ(outputs[1].shape(), (Tensor::shape_t{1, 1, 1}));
  EXPECT_EQ(outputs[0].toVector<float>(), (std::vector<float>{8.5F, 17.5F}));
  EXPECT_EQ(outputs[1].toVector<float>(), (std::vector<float>{29.0F}));
}

TEST_F(ParallelLinearTest, RejectsInvalidProjectionAndInputContracts) {
  auto reshapeWith = [](std::vector<int32_t> out_channels, std::vector<std::string> projection_names,
                        const Tensor::shape_t& input_shape = {1, 1, 2}) {
    auto op = std::make_shared<mllm::aops::ParallelLinearOp>(
        mllm::aops::ParallelLinearOpOptions{.in_channels = 2,
                                            .out_channels = std::move(out_channels),
                                            .projection_names = std::move(projection_names),
                                            .bias = false,
                                            .impl_type = mllm::aops::LinearImplTypes::kGGUF});
    std::vector<Tensor> inputs = {Tensor::empty(input_shape, mllm::kFloat32, mllm::kCPU)};
    std::vector<Tensor> outputs;
    op->reshape(inputs, outputs);
  };

  EXPECT_THROW(reshapeWith({2, 1}, {"same", "same"}), std::invalid_argument);
  EXPECT_THROW(reshapeWith({2, 1}, {"left", "nested.right"}), std::invalid_argument);
  EXPECT_THROW(reshapeWith({2, 1}, {"left", ""}), std::invalid_argument);
  EXPECT_THROW(reshapeWith({2}, {"left"}), std::invalid_argument);
  EXPECT_THROW(reshapeWith({2, 0}, {"left", "right"}), std::invalid_argument);
  EXPECT_THROW(reshapeWith({2, 1}, {"left", "right"}, {1, 1, 3}), std::invalid_argument);
  EXPECT_NO_THROW(reshapeWith({2, 1}, {"left", "right"}));
}

TEST_F(ParallelLinearTest, TraceAndSerializationPreserveProjectionContract) {
  ParallelLinearModule module("parallel_linear_trace", false);
  auto ir_context = mllm::ir::trace(module, Tensor::empty({1, 1, 2}, mllm::kFloat32, mllm::kCPU));
  auto op = findOp<mllm::ir::linalg::ParallelLinearOp>(ir_context->topLevelOp());
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->getAOp()->getOpType(), mllm::OpTypes::kParallelLinear);

  const auto serialized = mllm::jit::binary::dumpLinalgIROptions(op);
  EXPECT_EQ(serialized.at("in_channels"), 2);
  EXPECT_EQ(serialized.at("out_channels"), (std::vector<int32_t>{2, 1}));
  EXPECT_EQ(serialized.at("projection_names"), (std::vector<std::string>{"left", "right"}));
  EXPECT_EQ(serialized.at("bias"), false);
  EXPECT_EQ(serialized.at("impl_type"), "GGUF");
  EXPECT_EQ(serialized.at("kai_w4a32_decode_thread_cap"), 4);
  EXPECT_EQ(serialized.at("kai_w4a32_prefill_thread_cap"), 6);

  const auto restored = mllm::jit::interpreter::aopsFromJson(
      nlohmann::json{{"op_type", "ParallelLinear"}, {"backend", "CPU"}, {"op_options", serialized}});
  ASSERT_NE(restored, nullptr);
  ASSERT_EQ(restored->getOpType(), mllm::OpTypes::kParallelLinear);
  const auto restored_options = std::static_pointer_cast<mllm::aops::ParallelLinearOp>(restored)->options();
  EXPECT_EQ(restored_options.in_channels, 2);
  EXPECT_EQ(restored_options.out_channels, (std::vector<int32_t>{2, 1}));
  EXPECT_EQ(restored_options.projection_names, (std::vector<std::string>{"left", "right"}));
  EXPECT_FALSE(restored_options.bias);
  EXPECT_EQ(restored_options.impl_type, mllm::aops::LinearImplTypes::kGGUF);
  EXPECT_EQ(restored_options.kai_w4a32_decode_thread_cap, 4);
  EXPECT_EQ(restored_options.kai_w4a32_prefill_thread_cap, 6);
}

}  // namespace
