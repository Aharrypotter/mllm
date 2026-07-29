// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "mllm/models/qwen3_5/configuration_qwen3_5.hpp"

TEST(Qwen35ConfigTest, DefaultConfigBuildsHybridLayerSchedule) {
  const mllm::models::qwen3_5::Qwen3_5Config config;

  ASSERT_EQ(config.layer_types.size(), 24);
  EXPECT_EQ(config.numFullAttentionLayers(), 6);
  EXPECT_EQ(config.numGDNLayers(), 18);
  for (int32_t layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
    EXPECT_EQ(config.isFullAttentionLayer(layer_idx), (layer_idx + 1) % config.full_attention_interval == 0);
  }
}
