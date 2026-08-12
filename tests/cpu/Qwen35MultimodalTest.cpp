// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/models/qwen3_5/image_preprocessor_qwen3_5.hpp"
#include "mllm/models/qwen3_5/modeling_qwen3_5.hpp"
#include "mllm/models/qwen3_5/multimodal_qwen3_5.hpp"
#include "mllm/models/qwen3_5/tokenization_qwen3_5.hpp"
#include "mllm/models/qwen3_5/video_preprocessor_qwen3_5.hpp"

namespace {

using mllm::models::qwen3_5::advanceQwen3_5PositionIds;
using mllm::models::qwen3_5::calculateQwen3_5VideoTimestamps;
using mllm::models::qwen3_5::convertQwen3_5I420ToRgb;
using mllm::models::qwen3_5::expandQwen3_5ImagePlaceholders;
using mllm::models::qwen3_5::expandQwen3_5SingleImagePlaceholders;
using mllm::models::qwen3_5::expandQwen3_5VideoPlaceholders;
using mllm::models::qwen3_5::injectQwen3_5ImageEmbeddings;
using mllm::models::qwen3_5::injectQwen3_5VideoEmbeddings;
using mllm::models::qwen3_5::makeQwen3_5ImagePositionIds;
using mllm::models::qwen3_5::makeQwen3_5InterleavedRotaryEmbedding;
using mllm::models::qwen3_5::makeQwen3_5MultimodalPositionIds;
using mllm::models::qwen3_5::makeQwen3_5SingleImagePositionIds;
using mllm::models::qwen3_5::makeQwen3_5VideoMarkers;
using mllm::models::qwen3_5::makeQwen3_5VisionBilinearPositionEmbedding;
using mllm::models::qwen3_5::makeQwen3_5VisionPositionIds;
using mllm::models::qwen3_5::makeQwen3_5VisionRotaryEmbedding;
using mllm::models::qwen3_5::Qwen3_5Config;
using mllm::models::qwen3_5::qwen3_5ExactGelu;
using mllm::models::qwen3_5::Qwen3_5ForCausalLM;
using mllm::models::qwen3_5::Qwen3_5ImagePreprocessor;
using mllm::models::qwen3_5::Qwen3_5VideoPreprocessor;
using mllm::models::qwen3_5::resizeQwen3_5RgbLikeTorchvision;
using mllm::models::qwen3_5::sampleQwen3_5VideoFrames;

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

TEST_F(Qwen35MultimodalTest, PreprocessesAndConcatenatesDifferentImagesInOrder) {
  const auto temp_dir = std::filesystem::temp_directory_path();
  const auto first_path = temp_dir / "mllm_qwen35_multi_image_first.ppm";
  const auto second_path = temp_dir / "mllm_qwen35_multi_image_second.ppm";
  const auto write_ppm = [](const std::filesystem::path& path, int32_t width, int32_t height, char value) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream << "P6\n" << width << ' ' << height << "\n255\n";
    const std::string row(static_cast<size_t>(width) * 3, value);
    for (int32_t h = 0; h < height; ++h) stream.write(row.data(), static_cast<std::streamsize>(row.size()));
    if (!stream) throw std::runtime_error("failed to write Qwen3.5 image fixture");
  };
  write_ppm(first_path, 32, 32, static_cast<char>(0));
  write_ppm(second_path, 32, 64, static_cast<char>(255));

  const Qwen3_5ImagePreprocessor preprocessor(/*min_pixels=*/1, /*max_pixels=*/1 << 20);
  auto [patches, grids] = preprocessor(std::vector<std::string>{first_path.string(), second_path.string()});
  EXPECT_EQ(patches.shape(), (mllm::Tensor::shape_t{12, 1536}));
  EXPECT_EQ(grids.shape(), (mllm::Tensor::shape_t{2, 3}));
  const std::array<int32_t, 6> expected_grids = {1, 2, 2, 1, 4, 2};
  EXPECT_TRUE(std::equal(expected_grids.begin(), expected_grids.end(), grids.ptr<int32_t>()));
  EXPECT_FLOAT_EQ(patches.ptr<float>()[0], -1.0F);
  EXPECT_FLOAT_EQ(patches.ptr<float>()[4 * 1536], 1.0F);

  std::filesystem::remove(first_path);
  std::filesystem::remove(second_path);
}

TEST_F(Qwen35MultimodalTest, VideoSmartResizeMatchesOfficialTotalPixelEnvelope) {
  const Qwen3_5VideoPreprocessor preprocessor;
  EXPECT_EQ(preprocessor.smartResize(4, 48, 64), (std::pair<int32_t, int32_t>{64, 64}));
  EXPECT_EQ(preprocessor.smartResize(8, 720, 1280), (std::pair<int32_t, int32_t>{704, 1280}));
  EXPECT_EQ(preprocessor.smartResize(4, 1, 200), (std::pair<int32_t, int32_t>{32, 6400}));
  EXPECT_THROW((void)preprocessor.smartResize(1, 64, 64), std::invalid_argument);
  EXPECT_THROW((void)preprocessor.smartResize(4, 1, 201), std::invalid_argument);
}

TEST_F(Qwen35MultimodalTest, ConvertsBoundedI420FramesToRgb) {
  const std::array<uint8_t, 8> y = {16, 235, 81, 145, 16, 235, 81, 145};
  const std::array<uint8_t, 2> u = {128, 90};
  const std::array<uint8_t, 2> v = {128, 240};
  std::array<float, 24> rgb{};
  convertQwen3_5I420ToRgb(y.data(), 4, u.data(), 2, v.data(), 2, 4, 2, rgb.data());

  EXPECT_EQ((std::array<float, 3>{rgb[0], rgb[1], rgb[2]}), (std::array<float, 3>{0.0F, 0.0F, 0.0F}));
  EXPECT_EQ((std::array<float, 3>{rgb[3], rgb[4], rgb[5]}), (std::array<float, 3>{255.0F, 255.0F, 255.0F}));
  EXPECT_EQ((std::array<float, 3>{rgb[6], rgb[7], rgb[8]}), (std::array<float, 3>{255.0F, 0.0F, 0.0F}));
  EXPECT_EQ((std::array<float, 3>{rgb[9], rgb[10], rgb[11]}), (std::array<float, 3>{255.0F, 74.0F, 74.0F}));
  EXPECT_THROW(convertQwen3_5I420ToRgb(y.data(), 4, u.data(), 2, v.data(), 2, 3, 2, rgb.data()), std::invalid_argument);
  EXPECT_THROW(convertQwen3_5I420ToRgb(nullptr, 4, u.data(), 2, v.data(), 2, 4, 2, rgb.data()), std::invalid_argument);
}

TEST_F(Qwen35MultimodalTest, BicubicResizeMatchesTorchvisionUint8Oracle) {
  const std::array<float, 12> input = {0, 10, 255, 64, 20, 128, 128, 30, 64, 255, 40, 0};
  const std::array<uint8_t, 48> expected = {0,  7,   255, 1,  10,  244, 35, 16,  167, 52, 19,  128, 20, 13,  225, 42,
                                            16, 192, 88,  22, 125, 110, 25, 91,  91,  25, 110, 125, 28, 88,  192, 34,
                                            42, 225, 37,  20, 128, 31,  52, 167, 34,  35, 244, 40,  1,  255, 43,  0};
  EXPECT_EQ(resizeQwen3_5RgbLikeTorchvision(input.data(), 2, 2, 4, 4),
            (std::vector<uint8_t>{expected.begin(), expected.end()}));
}

TEST_F(Qwen35MultimodalTest, SamplesThePinnedVideoLikeTransformers) {
  EXPECT_EQ(sampleQwen3_5VideoFrames(8, 4.0), (std::vector<int32_t>{0, 2, 5, 7}));
  EXPECT_EQ(sampleQwen3_5VideoFrames(3, 30.0), (std::vector<int32_t>{0, 1, 2}));
  const auto capped = sampleQwen3_5VideoFrames(2000, 10.0, 10.0, 4, 768);
  ASSERT_EQ(capped.size(), 768);
  EXPECT_EQ(capped.front(), 0);
  EXPECT_EQ(capped.back(), 1999);
  EXPECT_THROW((void)sampleQwen3_5VideoFrames(0, 4.0), std::invalid_argument);
  EXPECT_THROW((void)sampleQwen3_5VideoFrames(8, -1.0), std::invalid_argument);
}

TEST_F(Qwen35MultimodalTest, BuildsOfficialTimestampSeparatedVideoMarkers) {
  const auto timestamps = calculateQwen3_5VideoTimestamps({0, 2, 5, 7}, 4.0);
  ASSERT_EQ(timestamps.size(), 2);
  EXPECT_DOUBLE_EQ(timestamps[0], 0.25);
  EXPECT_DOUBLE_EQ(timestamps[1], 1.5);
  EXPECT_EQ(makeQwen3_5VideoMarkers(timestamps), "<|vision_start|><0.2 seconds><|vision_start|><|video_pad|><|vision_end|>"
                                                 "<1.5 seconds><|vision_start|><|video_pad|><|vision_end|><|vision_end|>");

  const auto padded = calculateQwen3_5VideoTimestamps({3, 5, 8}, 2.0);
  ASSERT_EQ(padded.size(), 2);
  EXPECT_DOUBLE_EQ(padded[0], 2.0);
  EXPECT_DOUBLE_EQ(padded[1], 4.0);
  EXPECT_THROW((void)calculateQwen3_5VideoTimestamps({}, 4.0), std::invalid_argument);
  EXPECT_THROW((void)calculateQwen3_5VideoTimestamps({1, 0}, 4.0), std::invalid_argument);
  EXPECT_THROW((void)calculateQwen3_5VideoTimestamps({0, 1}, 0.0), std::invalid_argument);
}

TEST_F(Qwen35MultimodalTest, VideoPatchifyInterleavesFramesAndPadsTheLastFrame) {
  const Qwen3_5VideoPreprocessor preprocessor;
  auto video = mllm::Tensor::empty({3, 32, 32, 3}, mllm::kFloat32, mllm::kCPU).alloc();
  const std::array<float, 3> frame_values = {0.0F, 64.0F, 255.0F};
  for (int32_t t = 0; t < 3; ++t) {
    std::fill(video.ptr<float>() + static_cast<int64_t>(t) * 32 * 32 * 3,
              video.ptr<float>() + static_cast<int64_t>(t + 1) * 32 * 32 * 3, frame_values[t]);
  }

  auto [patches, grid] = preprocessor.flattenNormalizedPatches(video);
  EXPECT_EQ(patches.shape(), (mllm::Tensor::shape_t{8, 1536}));
  const std::array<int32_t, 3> expected_grid = {2, 2, 2};
  EXPECT_TRUE(std::equal(expected_grid.begin(), expected_grid.end(), grid.ptr<int32_t>()));
  const auto normalized = [](float value) { return value * (2.0F / 255.0F) - 1.0F; };
  EXPECT_FLOAT_EQ(patches.ptr<float>()[0], normalized(0.0F));
  EXPECT_FLOAT_EQ(patches.ptr<float>()[256], normalized(64.0F));
  EXPECT_FLOAT_EQ(patches.ptr<float>()[4 * 1536], normalized(255.0F));
  EXPECT_FLOAT_EQ(patches.ptr<float>()[4 * 1536 + 256], normalized(255.0F));
}

TEST_F(Qwen35MultimodalTest, ResizesEveryVideoFrameToTheSharedOfficialGeometry) {
  const Qwen3_5VideoPreprocessor preprocessor;
  auto video = mllm::Tensor::empty({4, 48, 64, 3}, mllm::kFloat32, mllm::kCPU).alloc();
  std::fill(video.ptr<float>(), video.ptr<float>() + video.numel(), 64.0F);
  auto resized = preprocessor.resizeFrames(video);
  EXPECT_EQ(resized.shape(), (mllm::Tensor::shape_t{4, 64, 64, 3}));
  EXPECT_NEAR(resized.ptr<float>()[0], 64.0F, 1.0e-4F);
  EXPECT_NEAR(resized.ptr<float>()[resized.numel() - 1], 64.0F, 1.0e-4F);
}

TEST_F(Qwen35MultimodalTest, ExpandsExactlyOneImagePlaceholderAndMarksItsSpan) {
  const std::vector<int64_t> input = {10, 248053, 248056, 248054, 11};
  auto [expanded, token_types] = expandQwen3_5SingleImagePlaceholders(input, 248056, 4);

  EXPECT_EQ(expanded, (std::vector<int64_t>{10, 248053, 248056, 248056, 248056, 248056, 248054, 11}));
  EXPECT_EQ(token_types, (std::vector<int32_t>{0, 0, 1, 1, 1, 1, 0, 0}));
  EXPECT_THROW(expandQwen3_5SingleImagePlaceholders({1, 2, 3}, 248056, 4), std::invalid_argument);
  EXPECT_THROW(expandQwen3_5SingleImagePlaceholders({248056, 248056}, 248056, 4), std::invalid_argument);
}

TEST_F(Qwen35MultimodalTest, ExpandsMultipleImagePlaceholdersInOrder) {
  const std::vector<int64_t> input = {10, 248053, 248056, 248054, 11, 248053, 248056, 248054, 12};
  auto [expanded, token_types] = expandQwen3_5ImagePlaceholders(input, 248056, {2, 3});

  EXPECT_EQ(expanded,
            (std::vector<int64_t>{10, 248053, 248056, 248056, 248054, 11, 248053, 248056, 248056, 248056, 248054, 12}));
  EXPECT_EQ(token_types, (std::vector<int32_t>{0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0}));
  EXPECT_THROW(expandQwen3_5ImagePlaceholders(input, 248056, {2}), std::invalid_argument);
  EXPECT_THROW(expandQwen3_5ImagePlaceholders(input, 248056, {2, 3, 4}), std::invalid_argument);
  EXPECT_THROW(expandQwen3_5ImagePlaceholders(input, 248056, {2, 0}), std::invalid_argument);
}

TEST_F(Qwen35MultimodalTest, ExpandsTimestampSeparatedVideoPlaceholders) {
  const std::vector<int64_t> input = {10, 248053, 20, 248053, 248057, 248054, 21, 248053, 248057, 248054, 248054, 11};
  auto [expanded, token_types] = expandQwen3_5VideoPlaceholders(input, 248057, {4, 4});

  EXPECT_EQ(expanded, (std::vector<int64_t>{10, 248053, 20, 248053, 248057, 248057, 248057, 248057, 248054, 21, 248053, 248057,
                                            248057, 248057, 248057, 248054, 248054, 11}));
  EXPECT_EQ(token_types, (std::vector<int32_t>{0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0}));
  EXPECT_THROW(expandQwen3_5VideoPlaceholders(input, 248057, {4}), std::invalid_argument);
  EXPECT_THROW(expandQwen3_5VideoPlaceholders(input, 248057, {4, 0}), std::invalid_argument);
}

TEST_F(Qwen35MultimodalTest, InjectsDistinctImageEmbeddingsIntoEverySpanInOrder) {
  auto input = mllm::Tensor::empty({1, 8, 3}, mllm::kFloat32, mllm::kCPU).alloc();
  std::fill(input.ptr<float>(), input.ptr<float>() + input.numel(), -1.0F);
  auto images = mllm::Tensor::empty({4, 3}, mllm::kFloat32, mllm::kCPU).alloc();
  for (int32_t row = 0; row < 4; ++row) {
    for (int32_t column = 0; column < 3; ++column) { images.ptr<float>()[row * 3 + column] = row * 10 + column; }
  }
  auto types = mllm::Tensor::empty({1, 8}, mllm::kInt32, mllm::kCPU).alloc();
  const std::array<int32_t, 8> type_values = {0, 1, 1, 0, 0, 1, 1, 0};
  std::copy(type_values.begin(), type_values.end(), types.ptr<int32_t>());

  injectQwen3_5ImageEmbeddings(input, images, types);
  for (int32_t position = 0; position < 8; ++position) {
    for (int32_t column = 0; column < 3; ++column) {
      const int32_t image_row = position == 1 ? 0 : position == 2 ? 1 : position == 5 ? 2 : position == 6 ? 3 : -1;
      const float expected = image_row < 0 ? -1.0F : image_row * 10 + column;
      EXPECT_FLOAT_EQ(input.ptr<float>()[position * 3 + column], expected);
    }
  }
}

TEST_F(Qwen35MultimodalTest, InjectsVideoEmbeddingsWithoutOverwritingTextOrImages) {
  auto input = mllm::Tensor::empty({1, 6, 2}, mllm::kFloat32, mllm::kCPU).alloc();
  std::fill(input.ptr<float>(), input.ptr<float>() + input.numel(), -1.0F);
  auto video = mllm::Tensor::empty({3, 2}, mllm::kFloat32, mllm::kCPU).alloc();
  const std::array<float, 6> values = {10.0F, 11.0F, 20.0F, 21.0F, 30.0F, 31.0F};
  std::copy(values.begin(), values.end(), video.ptr<float>());
  auto types = mllm::Tensor::empty({1, 6}, mllm::kInt32, mllm::kCPU).alloc();
  const std::array<int32_t, 6> type_values = {0, 2, 2, 1, 2, 0};
  std::copy(type_values.begin(), type_values.end(), types.ptr<int32_t>());

  injectQwen3_5VideoEmbeddings(input, video, types);
  const std::array<float, 12> expected = {-1.0F, -1.0F, 10.0F, 11.0F, 20.0F, 21.0F, -1.0F, -1.0F, 30.0F, 31.0F, -1.0F, -1.0F};
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), input.ptr<float>()));
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

TEST_F(Qwen35MultimodalTest, BuildsOfficialMultipleImagePositionsWithDifferentGeometries) {
  auto token_types = mllm::Tensor::empty({1, 12}, mllm::kInt32, mllm::kCPU).alloc();
  const std::array<int32_t, 12> types = {0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0};
  std::copy(types.begin(), types.end(), token_types.ptr<int32_t>());
  auto grids = mllm::Tensor::empty({2, 3}, mllm::kInt32, mllm::kCPU).alloc();
  const std::array<int32_t, 6> dimensions = {1, 4, 4, 1, 2, 4};
  std::copy(dimensions.begin(), dimensions.end(), grids.ptr<int32_t>());

  auto positions = makeQwen3_5ImagePositionIds(token_types, grids, 2);
  const std::array<int64_t, 12> expected_t = {0, 1, 2, 2, 2, 2, 4, 5, 6, 6, 8, 9};
  const std::array<int64_t, 12> expected_h = {0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 8, 9};
  const std::array<int64_t, 12> expected_w = {0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_TRUE(std::equal(expected_t.begin(), expected_t.end(), positions.ptr<int64_t>()));
  EXPECT_TRUE(std::equal(expected_h.begin(), expected_h.end(), positions.ptr<int64_t>() + 12));
  EXPECT_TRUE(std::equal(expected_w.begin(), expected_w.end(), positions.ptr<int64_t>() + 24));

  auto missing_span_types = token_types.clone();
  missing_span_types.ptr<int32_t>()[8] = 0;
  missing_span_types.ptr<int32_t>()[9] = 0;
  EXPECT_THROW(makeQwen3_5ImagePositionIds(missing_span_types, grids, 2), std::invalid_argument);
}

TEST_F(Qwen35MultimodalTest, BuildsTimestampSeparatedVideoPositions) {
  auto token_types = mllm::Tensor::empty({1, 12}, mllm::kInt32, mllm::kCPU).alloc();
  const std::array<int32_t, 12> types = {0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0};
  std::copy(types.begin(), types.end(), token_types.ptr<int32_t>());
  auto video_grid = mllm::Tensor::empty({1, 3}, mllm::kInt32, mllm::kCPU).alloc();
  const std::array<int32_t, 3> dimensions = {2, 4, 4};
  std::copy(dimensions.begin(), dimensions.end(), video_grid.ptr<int32_t>());

  auto positions = makeQwen3_5MultimodalPositionIds(token_types, mllm::Tensor::nil(), video_grid, 2);
  const std::array<int64_t, 12> expected_t = {0, 1, 2, 2, 2, 2, 4, 5, 5, 5, 5, 7};
  const std::array<int64_t, 12> expected_h = {0, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7};
  const std::array<int64_t, 12> expected_w = {0, 1, 2, 3, 2, 3, 4, 5, 6, 5, 6, 7};
  EXPECT_TRUE(std::equal(expected_t.begin(), expected_t.end(), positions.ptr<int64_t>()));
  EXPECT_TRUE(std::equal(expected_h.begin(), expected_h.end(), positions.ptr<int64_t>() + 12));
  EXPECT_TRUE(std::equal(expected_w.begin(), expected_w.end(), positions.ptr<int64_t>() + 24));

  auto missing_span = token_types.clone();
  missing_span.ptr<int32_t>()[7] = 0;
  missing_span.ptr<int32_t>()[8] = 0;
  missing_span.ptr<int32_t>()[9] = 0;
  missing_span.ptr<int32_t>()[10] = 0;
  EXPECT_THROW(makeQwen3_5MultimodalPositionIds(missing_span, mllm::Tensor::nil(), video_grid, 2), std::invalid_argument);
}

TEST_F(Qwen35MultimodalTest, MatchesPinnedTransformersVideoMropeOracle) {
  auto token_types = mllm::Tensor::empty({1, 42}, mllm::kInt32, mllm::kCPU).alloc();
  const std::array<int32_t, 42> types = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,
                                         0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::copy(types.begin(), types.end(), token_types.ptr<int32_t>());
  auto video_grid = mllm::Tensor::empty({1, 3}, mllm::kInt32, mllm::kCPU).alloc();
  const std::array<int32_t, 3> dimensions = {2, 4, 4};
  std::copy(dimensions.begin(), dimensions.end(), video_grid.ptr<int32_t>());

  auto positions = makeQwen3_5MultimodalPositionIds(token_types, mllm::Tensor::nil(), video_grid, 2);
  const std::array<int64_t, 42> expected_t = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 11, 11,
                                              11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 21, 21, 21, 23,
                                              24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37};
  const std::array<int64_t, 42> expected_h = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 11, 12,
                                              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 21, 22, 22, 23,
                                              24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37};
  const std::array<int64_t, 42> expected_w = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 11,
                                              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 21, 22, 23,
                                              24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37};
  EXPECT_TRUE(std::equal(expected_t.begin(), expected_t.end(), positions.ptr<int64_t>()));
  EXPECT_TRUE(std::equal(expected_h.begin(), expected_h.end(), positions.ptr<int64_t>() + 42));
  EXPECT_TRUE(std::equal(expected_w.begin(), expected_w.end(), positions.ptr<int64_t>() + 84));
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

  mllm::models::ARGenerationOutputPast incomplete_video = {
      {"sequence", sequence},
      {"pixel_values_videos", pixel_values},
      {"mm_token_type_ids", token_types},
  };
  EXPECT_THROW((void)model.forward(incomplete_video, {}), std::invalid_argument);
}

}  // namespace
