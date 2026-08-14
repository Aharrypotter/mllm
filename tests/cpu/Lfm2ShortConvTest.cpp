// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/models/lfm2/modeling_lfm2.hpp"

namespace {

class Lfm2ShortConvTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

auto tensor(const std::string& name, const mllm::Tensor::shape_t& shape, const std::vector<float>& values) -> mllm::Tensor {
  auto result = mllm::Tensor::empty(shape, mllm::kFloat32, mllm::kCPU).setMemType(mllm::kParamsNormal).setName(name).alloc();
  EXPECT_EQ(result.numel(), values.size());
  std::copy(values.begin(), values.end(), result.ptr<float>());
  return result;
}

auto input(const std::vector<float>& values) -> mllm::Tensor {
  auto result = mllm::Tensor::empty({1, static_cast<int32_t>(values.size()), 1}, mllm::kFloat32, mllm::kCPU).alloc();
  std::copy(values.begin(), values.end(), result.ptr<float>());
  return result;
}

auto shortConv() -> mllm::models::lfm2::Lfm2ShortConv {
  mllm::models::lfm2::Lfm2Config config;
  config.hidden_size = 1;
  config.conv_L_cache = 3;
  config.conv_bias = false;
  // Keep this semantics-only test portable. kDefault selects the ARM-only
  // MllmBlas fallback for the deliberately tiny K=1 geometry on non-BLAS x86.
  config.linear_impl_type = mllm::aops::LinearImplTypes::kGGUF;
  auto module = mllm::models::lfm2::Lfm2ShortConv("unit", config);
  auto parameters = mllm::ParameterFile::create();
  parameters->push("unit.in_proj.weight", tensor("unit.in_proj.weight", {3, 1}, {1.0F, 1.0F, 1.0F}));
  parameters->push("unit.conv.weight", tensor("unit.conv.weight", {1, 1, 3}, {1.0F, 2.0F, 3.0F}));
  parameters->push("unit.out_proj.weight", tensor("unit.out_proj.weight", {1, 1}, {1.0F}));
  module.load(parameters);
  return module;
}

auto values(mllm::Tensor output) -> std::vector<float> {
  output = output.contiguous();
  return {output.ptr<float>(), output.ptr<float>() + output.numel()};
}

TEST_F(Lfm2ShortConvTest, ChunkedPrefillAndDecodeMatchOneShotCausalConvolution) {
  auto chunked = shortConv();
  auto prefill = values(chunked(input({1.0F, 2.0F}))[0]);
  auto decode = values(chunked(input({3.0F}))[0]);
  EXPECT_EQ(prefill, (std::vector<float>{3.0F, 28.0F}));
  EXPECT_EQ(decode, (std::vector<float>{108.0F}));
  EXPECT_EQ(values(chunked.state()), (std::vector<float>{4.0F, 9.0F}));

  auto one_shot = shortConv();
  EXPECT_EQ(values(one_shot(input({1.0F, 2.0F, 3.0F}))[0]), (std::vector<float>{3.0F, 28.0F, 108.0F}));
}

TEST_F(Lfm2ShortConvTest, ResetClearsTheTwoRequiredHistoricalSamples) {
  auto module = shortConv();
  (void)module(input({2.0F}));
  module.resetState(1);
  EXPECT_EQ(values(module.state()), (std::vector<float>{0.0F, 0.0F}));
}

}  // namespace
