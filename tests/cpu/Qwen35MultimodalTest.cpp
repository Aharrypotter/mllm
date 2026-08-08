// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/models/qwen3_5/image_preprocessor_qwen3_5.hpp"
#include "mllm/models/qwen3_5/modeling_qwen3_5.hpp"
#include "mllm/models/qwen3_5/multimodal_qwen3_5.hpp"

namespace {

using mllm::models::qwen3_5::advanceQwen3_5PositionIds;
using mllm::models::qwen3_5::expandQwen3_5SingleImagePlaceholders;
using mllm::models::qwen3_5::makeQwen3_5InterleavedRotaryEmbedding;
using mllm::models::qwen3_5::makeQwen3_5SingleImagePositionIds;
using mllm::models::qwen3_5::makeQwen3_5VisionBilinearPositionEmbedding;
using mllm::models::qwen3_5::makeQwen3_5VisionPositionIds;
using mllm::models::qwen3_5::makeQwen3_5VisionRotaryEmbedding;
using mllm::models::qwen3_5::Qwen3_5Config;
using mllm::models::qwen3_5::qwen3_5ExactGelu;
using mllm::models::qwen3_5::Qwen3_5ForCausalLM;
using mllm::models::qwen3_5::Qwen3_5ImagePreprocessor;

Qwen3_5Config makeTinyQwen3_5Config() {
  Qwen3_5Config config;
  config.hidden_size = 8;
  config.head_dim = 4;
  config.intermediate_size = 16;
  config.num_attention_heads = 2;
  config.num_key_value_heads = 1;
  config.num_hidden_layers = 2;
  config.max_position_embeddings = 32;
  config.vocab_size = 32;
  config.layer_types = {"linear_attention", "full_attention"};
  config.linear_num_key_heads = 1;
  config.linear_num_value_heads = 1;
  config.linear_key_head_dim = 4;
  config.linear_value_head_dim = 4;
  config.linear_conv_kernel_dim = 3;
  config.partial_rotary_factor = 0.5F;
  config.max_cache_length = 8;
  return config;
}

class Qwen35MultimodalTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

TEST_F(Qwen35MultimodalTest, SmartResizeUsesFactor32AndDeploymentEnvelope) {
  const Qwen3_5ImagePreprocessor preprocessor;
  const Qwen3_5ImagePreprocessor unconstrained_preprocessor(/*min_pixels=*/1, /*max_pixels=*/1 << 20);

  EXPECT_EQ(preprocessor.smartResize(256, 256), (std::pair<int32_t, int32_t>{256, 256}));
  EXPECT_EQ(preprocessor.smartResize(1200, 800), (std::pair<int32_t, int32_t>{608, 416}));
  EXPECT_EQ(preprocessor.smartResize(100, 200), (std::pair<int32_t, int32_t>{192, 384}));
  EXPECT_EQ(unconstrained_preprocessor.smartResize(80, 80), (std::pair<int32_t, int32_t>{64, 64}));
  EXPECT_THROW((void)preprocessor.smartResize(1, 201), std::invalid_argument);
}

TEST_F(Qwen35MultimodalTest, FlattenPatchesMatchesOfficialBlockMajorLayout) {
  const Qwen3_5ImagePreprocessor preprocessor(/*min_pixels=*/1, /*max_pixels=*/1 << 20);
  auto image = mllm::Tensor::empty({64, 32, 3}, mllm::kFloat32, mllm::kCPU).alloc();
  auto* pixels = image.ptr<float>();
  for (int32_t h = 0; h < 64; ++h) {
    for (int32_t w = 0; w < 32; ++w) {
      for (int32_t c = 0; c < 3; ++c) { pixels[(h * 32 + w) * 3 + c] = static_cast<float>(h + w + c); }
    }
  }

  auto [patches, grid] = preprocessor.flattenNormalizedPatches(image);
  EXPECT_EQ(patches.shape(), (mllm::Tensor::shape_t{8, 1536}));
  EXPECT_EQ(grid.shape(), (mllm::Tensor::shape_t{1, 3}));
  EXPECT_EQ(grid.ptr<int32_t>()[0], 1);
  EXPECT_EQ(grid.ptr<int32_t>()[1], 4);
  EXPECT_EQ(grid.ptr<int32_t>()[2], 2);

  const auto normalized = [](float value) { return value * (2.0F / 255.0F) - 1.0F; };
  EXPECT_FLOAT_EQ(patches.ptr<float>()[0], normalized(0.0F));
  EXPECT_FLOAT_EQ(patches.ptr<float>()[256], normalized(0.0F));
  EXPECT_FLOAT_EQ(patches.ptr<float>()[512], normalized(1.0F));
  EXPECT_FLOAT_EQ(patches.ptr<float>()[2 * 1536], normalized(16.0F));
  EXPECT_FLOAT_EQ(patches.ptr<float>()[4 * 1536], normalized(32.0F));
}

TEST_F(Qwen35MultimodalTest, ExpandsExactlyOneImagePlaceholderAndMarksItsSpan) {
  const std::vector<int64_t> input = {10, 248053, 248056, 248054, 11};
  auto [expanded, token_types] = expandQwen3_5SingleImagePlaceholders(input, 248056, 4);

  EXPECT_EQ(expanded, (std::vector<int64_t>{10, 248053, 248056, 248056, 248056, 248056, 248054, 11}));
  EXPECT_EQ(token_types, (std::vector<int32_t>{0, 0, 1, 1, 1, 1, 0, 0}));
  EXPECT_THROW(expandQwen3_5SingleImagePlaceholders({1, 2, 3}, 248056, 4), std::invalid_argument);
  EXPECT_THROW(expandQwen3_5SingleImagePlaceholders({248056, 248056}, 248056, 4), std::invalid_argument);
}

TEST_F(Qwen35MultimodalTest, InterleavesTemporalHeightAndWidthFrequencies) {
  auto positions = mllm::Tensor::empty({3, 1, 1}, mllm::kInt64, mllm::kCPU).alloc();
  positions.ptr<int64_t>()[0] = 1;
  positions.ptr<int64_t>()[1] = 2;
  positions.ptr<int64_t>()[2] = 3;
  auto inv_freq = mllm::Tensor::empty({32}, mllm::kFloat32, mllm::kCPU).alloc();
  std::fill(inv_freq.ptr<float>(), inv_freq.ptr<float>() + 32, 1.0F);

  auto [sin, cos] = makeQwen3_5InterleavedRotaryEmbedding(positions, inv_freq, {11, 11, 10});
  ASSERT_EQ(sin.shape(), (mllm::Tensor::shape_t{1, 1, 64}));
  EXPECT_FLOAT_EQ(sin.ptr<float>()[0], std::sin(1.0F));
  EXPECT_FLOAT_EQ(sin.ptr<float>()[1], std::sin(2.0F));
  EXPECT_FLOAT_EQ(sin.ptr<float>()[2], std::sin(3.0F));
  EXPECT_FLOAT_EQ(sin.ptr<float>()[30], std::sin(1.0F));
  EXPECT_FLOAT_EQ(sin.ptr<float>()[31], std::sin(2.0F));
  for (int32_t d = 0; d < 32; ++d) {
    EXPECT_FLOAT_EQ(sin.ptr<float>()[d], sin.ptr<float>()[d + 32]);
    EXPECT_FLOAT_EQ(cos.ptr<float>()[d], cos.ptr<float>()[d + 32]);
  }
}

TEST_F(Qwen35MultimodalTest, BuildsBlockMajorVisionPositionsAndBilinearEmbeddings) {
  auto grid = mllm::Tensor::empty({1, 3}, mllm::kInt32, mllm::kCPU).alloc();
  const std::array<int32_t, 3> dimensions = {1, 4, 4};
  std::copy(dimensions.begin(), dimensions.end(), grid.ptr<int32_t>());
  auto positions = makeQwen3_5VisionPositionIds(grid, 2);

  const std::array<int32_t, 8> first_four_pairs = {0, 0, 0, 1, 1, 0, 1, 1};
  EXPECT_TRUE(std::equal(first_four_pairs.begin(), first_four_pairs.end(), positions.ptr<int32_t>()));

  auto learned = mllm::Tensor::empty({4, 1}, mllm::kFloat32, mllm::kCPU).alloc();
  learned.ptr<float>()[0] = 0.0F;
  learned.ptr<float>()[1] = 1.0F;
  learned.ptr<float>()[2] = 2.0F;
  learned.ptr<float>()[3] = 3.0F;
  auto interpolated = makeQwen3_5VisionBilinearPositionEmbedding(learned, grid, 2);
  EXPECT_NEAR(interpolated.ptr<float>()[0], 0.0F, 1.0e-6F);
  EXPECT_NEAR(interpolated.ptr<float>()[1], 1.0F / 3.0F, 1.0e-6F);
  EXPECT_NEAR(interpolated.ptr<float>()[2], 2.0F / 3.0F, 1.0e-6F);
  EXPECT_NEAR(interpolated.ptr<float>()[3], 1.0F, 1.0e-6F);

  auto [sin, cos] = makeQwen3_5VisionRotaryEmbedding(positions, 64);
  EXPECT_EQ(sin.shape(), (mllm::Tensor::shape_t{16, 32}));
  EXPECT_EQ(cos.shape(), (mllm::Tensor::shape_t{16, 32}));
  EXPECT_FLOAT_EQ(sin.ptr<float>()[0], 0.0F);
  EXPECT_FLOAT_EQ(cos.ptr<float>()[0], 1.0F);
}

TEST_F(Qwen35MultimodalTest, UsesExactGeluForVisionMerger) {
  auto input = mllm::Tensor::empty({3}, mllm::kFloat32, mllm::kCPU).alloc();
  input.ptr<float>()[0] = -1.0F;
  input.ptr<float>()[1] = 0.0F;
  input.ptr<float>()[2] = 1.0F;

  const auto output = qwen3_5ExactGelu(input);
  EXPECT_NEAR(output.ptr<float>()[0], -0.15865526F, 1.0e-7F);
  EXPECT_FLOAT_EQ(output.ptr<float>()[1], 0.0F);
  EXPECT_NEAR(output.ptr<float>()[2], 0.84134477F, 1.0e-7F);
}

TEST_F(Qwen35MultimodalTest, BuildsOfficialSingleImagePositionsAndDecodeStep) {
  auto token_types = mllm::Tensor::empty({1, 8}, mllm::kInt32, mllm::kCPU).alloc();
  const std::array<int32_t, 8> types = {0, 0, 1, 1, 1, 1, 0, 0};
  std::copy(types.begin(), types.end(), token_types.ptr<int32_t>());
  auto grid = mllm::Tensor::empty({1, 3}, mllm::kInt32, mllm::kCPU).alloc();
  const std::array<int32_t, 3> dimensions = {1, 4, 4};
  std::copy(dimensions.begin(), dimensions.end(), grid.ptr<int32_t>());

  auto positions = makeQwen3_5SingleImagePositionIds(token_types, grid, 2);
  const std::array<int64_t, 8> expected_t = {0, 1, 2, 2, 2, 2, 4, 5};
  const std::array<int64_t, 8> expected_h = {0, 1, 2, 2, 3, 3, 4, 5};
  const std::array<int64_t, 8> expected_w = {0, 1, 2, 3, 2, 3, 4, 5};
  EXPECT_TRUE(std::equal(expected_t.begin(), expected_t.end(), positions.ptr<int64_t>()));
  EXPECT_TRUE(std::equal(expected_h.begin(), expected_h.end(), positions.ptr<int64_t>() + 8));
  EXPECT_TRUE(std::equal(expected_w.begin(), expected_w.end(), positions.ptr<int64_t>() + 16));

  auto next = advanceQwen3_5PositionIds(positions);
  ASSERT_EQ(next.shape(), (mllm::Tensor::shape_t{3, 1, 1}));
  EXPECT_EQ(next.ptr<int64_t>()[0], 6);
  EXPECT_EQ(next.ptr<int64_t>()[1], 6);
  EXPECT_EQ(next.ptr<int64_t>()[2], 6);
}

TEST_F(Qwen35MultimodalTest, RejectsPlaceholderCountMismatchAndVideoTypes) {
  auto token_types = mllm::Tensor::empty({1, 4}, mllm::kInt32, mllm::kCPU).alloc();
  std::fill(token_types.ptr<int32_t>(), token_types.ptr<int32_t>() + 4, 1);
  auto grid = mllm::Tensor::empty({1, 3}, mllm::kInt32, mllm::kCPU).alloc();
  const std::array<int32_t, 3> dimensions = {1, 2, 2};
  std::copy(dimensions.begin(), dimensions.end(), grid.ptr<int32_t>());

  EXPECT_THROW(makeQwen3_5SingleImagePositionIds(token_types, grid, 2), std::invalid_argument);
  token_types.ptr<int32_t>()[0] = 2;
  EXPECT_THROW(makeQwen3_5SingleImagePositionIds(token_types, grid, 2), std::invalid_argument);
}

TEST_F(Qwen35MultimodalTest, ResetClearsKvProgressAndRejectsInvalidBatchSize) {
  Qwen3_5ForCausalLM model(makeTinyQwen3_5Config());
  ASSERT_EQ(model.kvCache().getLayerNums(), 1);
  model.kvCache().setCurrentSeqCnt(3);
  ASSERT_EQ(model.kvCache().getCurrentSeqCnt(0), 3);

  model.resetState();
  EXPECT_EQ(model.kvCache().getCurrentSeqCnt(0), 0);
  EXPECT_THROW(model.resetState(0), std::invalid_argument);
}

TEST_F(Qwen35MultimodalTest, RejectsIncompleteAndTextOnlyImageInputsBeforeExecution) {
  Qwen3_5ForCausalLM model(makeTinyQwen3_5Config());
  auto sequence = mllm::Tensor::empty({1, 2}, mllm::kInt64, mllm::kCPU).alloc();
  auto pixel_values = mllm::Tensor::empty({4, 24}, mllm::kFloat32, mllm::kCPU).alloc();
  auto image_grid = mllm::Tensor::empty({1, 3}, mllm::kInt32, mllm::kCPU).alloc();
  auto token_types = mllm::Tensor::empty({1, 2}, mllm::kInt32, mllm::kCPU).alloc();

  mllm::models::ARGenerationOutputPast incomplete = {
      {"sequence", sequence},
      {"pixel_values", pixel_values},
  };
  EXPECT_THROW((void)model.forward(incomplete, {}), std::invalid_argument);

  mllm::models::ARGenerationOutputPast text_only_image = {
      {"sequence", sequence},
      {"pixel_values", pixel_values},
      {"image_grid_thw", image_grid},
      {"mm_token_type_ids", token_types},
  };
  EXPECT_THROW((void)model.forward(text_only_image, {}), std::invalid_argument);
}

}  // namespace
