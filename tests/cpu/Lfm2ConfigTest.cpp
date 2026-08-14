// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <gtest/gtest.h>

#include <string>

#include "mllm/backends/cpu/ops/LinearOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/lfm2/configuration_lfm2.hpp"
#include "mllm/models/lfm2/modeling_lfm2.hpp"

namespace {

auto loadConfig() -> mllm::models::lfm2::Lfm2Config {
  return mllm::models::lfm2::Lfm2Config(std::string(LFM2_EXAMPLE_DIR) + "/config_2.6B_w4a32_kai.json");
}

}  // namespace

TEST(Lfm2ConfigTest, Official26BContractUsesCompactAttentionSlots) {
  const auto config = loadConfig();
  EXPECT_TRUE(mllm::models::lfm2::matchesOfficialRuntimeContract(config));
  EXPECT_EQ(config.numAttentionLayers(), 8);
  EXPECT_EQ(config.numConvLayers(), 22);
  EXPECT_EQ(config.attentionSlotForPhysicalLayer(2), 0);
  EXPECT_EQ(config.attentionSlotForPhysicalLayer(27), 7);
  EXPECT_THROW((void)config.attentionSlotForPhysicalLayer(0), std::invalid_argument);
}

TEST(Lfm2ConfigTest, NativeKVCacheUsesEightHeadsPerLogicalSlot) {
  mllm::initializeContext();
  const auto config = loadConfig();
  auto model = mllm::models::lfm2::Lfm2ForCausalLM(config);
  EXPECT_EQ(model.kvCache().getLayerNums(), 8);
  EXPECT_EQ(model.kvCache().kvHeads(), 8);
  EXPECT_EQ(model.kvCache().headDim(), 64);
  EXPECT_EQ(model.kvCache().maxCacheLength(), 2048);
  EXPECT_NO_THROW(model.resetState());
  EXPECT_EQ(model.kvCache().getCurrentSeqCnt(0), 0);
  model.kvCache().setCurrentSeqCnt(1);
  auto sequence = mllm::Tensor::zeros({1, 1}, mllm::kInt64, mllm::kCPU);
  EXPECT_THROW((void)model.forward({{"sequence", sequence}}, {}), std::invalid_argument);
  mllm::shutdownContext();
}

TEST(Lfm2ConfigTest, RejectsPhysicalLayerScheduleDrift) {
  auto config = loadConfig();
  config.layer_types[0] = "full_attention";
  EXPECT_FALSE(mllm::models::lfm2::matchesOfficialRuntimeContract(config));
}

TEST(Lfm2ConfigTest, KaiW4A32ThreadCapsSeparateDecodeAndPrefill) {
  using mllm::cpu::detail::kaiW4A32ThreadCount;
  EXPECT_EQ(kaiW4A32ThreadCount(1, 8, 4, 6), 4);
  EXPECT_EQ(kaiW4A32ThreadCount(28, 8, 4, 6), 6);
  EXPECT_EQ(kaiW4A32ThreadCount(1, 2, 4, 6), 2);
  EXPECT_EQ(kaiW4A32ThreadCount(28, 4, 4, 6), 4);
  EXPECT_EQ(kaiW4A32ThreadCount(1, 8, 0, 0), 8);
}
