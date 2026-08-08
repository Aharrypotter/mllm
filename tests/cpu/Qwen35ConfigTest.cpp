// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "mllm/models/qwen3_5/configuration_qwen3_5.hpp"

namespace {

auto exampleDir() -> std::string {
  const char* example_dir_override = std::getenv("MLLM_QWEN35_EXAMPLE_DIR");
  return example_dir_override == nullptr ? std::string(QWEN35_EXAMPLE_DIR) : std::string(example_dir_override);
}

auto loadConfig(const std::string& model_size) -> mllm::models::qwen3_5::Qwen3_5Config {
  return mllm::models::qwen3_5::Qwen3_5Config(exampleDir() + "/config_" + model_size + "_w4a32_kai.json");
}

auto loadMultimodalConfig() -> mllm::models::qwen3_5::Qwen3_5Config {
  return mllm::models::qwen3_5::Qwen3_5Config(exampleDir() + "/config_0.8B_multimodal_w4a32_kai.json");
}

constexpr auto kKaiLinearImpl =
    mllm::aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32;

void pushDescriptor(const mllm::ParameterFile::ptr_t& parameter_file, const std::string& name,
                    const std::vector<int32_t>& shape, mllm::DataTypes dtype = mllm::kFloat32) {
  auto tensor = mllm::Tensor::empty(shape, dtype, mllm::kCPU);
  tensor.impl()->storage()->mem_type_ = mllm::kManual;
  parameter_file->push(name, tensor);
}

auto parameterFileWithEmbedding(mllm::ModelFileVersion version, const std::vector<int32_t>& shape,
                                mllm::DataTypes dtype = mllm::kFloat32) -> mllm::ParameterFile::ptr_t {
  auto parameter_file = mllm::ParameterFile::create(version);
  // These tests validate descriptors without allocating official-size weights.
  pushDescriptor(parameter_file, "model.language_model.embed_tokens.weight", shape, dtype);
  return parameter_file;
}

}  // namespace

TEST(Qwen35ConfigTest, Official08BConfigBuildsHybridLayerSchedule) {
  const auto config = loadConfig("0.8B");

  EXPECT_EQ(mllm::models::qwen3_5::modelNameForConfig(config), "Qwen3.5-0.8B");
  EXPECT_EQ(config.linear_impl_type, kKaiLinearImpl);
  ASSERT_EQ(config.layer_types.size(), 24);
  EXPECT_EQ(config.numFullAttentionLayers(), 6);
  EXPECT_EQ(config.numGDNLayers(), 18);
  for (int32_t layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
    EXPECT_EQ(config.isFullAttentionLayer(layer_idx), (layer_idx + 1) % config.full_attention_interval == 0);
  }
}

TEST(Qwen35ConfigTest, Official4BConfigBuildsGroupedHeadHybridSchedule) {
  const auto config = loadConfig("4B");

  EXPECT_EQ(mllm::models::qwen3_5::modelNameForConfig(config), "Qwen3.5-4B");
  EXPECT_EQ(config.linear_impl_type, kKaiLinearImpl);
  EXPECT_EQ(config.hidden_size, 2560);
  EXPECT_EQ(config.intermediate_size, 9216);
  EXPECT_EQ(config.head_dim, 256);
  EXPECT_EQ(config.num_attention_heads, 16);
  EXPECT_EQ(config.num_key_value_heads, 4);
  EXPECT_EQ(config.num_hidden_layers, 32);
  EXPECT_EQ(config.linear_num_key_heads, 16);
  EXPECT_EQ(config.linear_num_value_heads, 32);
  EXPECT_EQ(config.linear_key_head_dim, 128);
  EXPECT_EQ(config.linear_value_head_dim, 128);
  EXPECT_EQ(config.linear_conv_kernel_dim, 4);
  EXPECT_EQ(config.max_cache_length, 2048);
  EXPECT_TRUE(config.tie_word_embeddings);
  EXPECT_EQ(config.hidden_act, "silu");
  EXPECT_EQ(config.mamba_ssm_dtype, "float32");
  EXPECT_EQ(config.numFullAttentionLayers(), 8);
  EXPECT_EQ(config.numGDNLayers(), 24);

  ASSERT_EQ(config.layer_types.size(), 32);
  for (int32_t layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
    EXPECT_EQ(config.isFullAttentionLayer(layer_idx), (layer_idx + 1) % config.full_attention_interval == 0);
  }
}

TEST(Qwen35ConfigTest, Official08BMultimodalConfigBuildsVisionContract) {
  const auto config = loadMultimodalConfig();

  EXPECT_EQ(mllm::models::qwen3_5::modelNameForConfig(config), "Qwen3.5-0.8B Multimodal");
  EXPECT_TRUE(mllm::models::qwen3_5::isOfficialQwen35_08BMultimodalRuntimeConfig(config));
  EXPECT_TRUE(config.vision_enabled);
  EXPECT_EQ(config.vision_depth, 12);
  EXPECT_EQ(config.vision_hidden_size, 768);
  EXPECT_EQ(config.vision_intermediate_size, 3072);
  EXPECT_EQ(config.vision_num_heads, 12);
  EXPECT_EQ(config.vision_num_position_embeddings, 2304);
  EXPECT_EQ(config.vision_patch_size, 16);
  EXPECT_EQ(config.vision_temporal_patch_size, 2);
  EXPECT_EQ(config.vision_spatial_merge_size, 2);
  EXPECT_EQ(config.vision_out_hidden_size, config.hidden_size);
  EXPECT_EQ(config.mrope_section, (std::vector<int32_t>{11, 11, 10}));
  EXPECT_EQ(config.image_min_pixels, 256 * 256);
  EXPECT_EQ(config.image_max_pixels, 512 * 512);
}

TEST(Qwen35ConfigTest, RejectsModelConfigEmbeddingMismatchWithoutAllocatingWeights) {
  const auto config = loadConfig("0.8B");
  const auto expected_numel =
      static_cast<int32_t>(static_cast<size_t>(config.vocab_size) * static_cast<size_t>(config.hidden_size));

  const auto matching_v2 = parameterFileWithEmbedding(mllm::ModelFileVersion::kV2, {config.vocab_size, config.hidden_size});
  EXPECT_NO_THROW(mllm::models::qwen3_5::validateModelConfigMatch(config, matching_v2));

  const auto mismatched_v2 = parameterFileWithEmbedding(mllm::ModelFileVersion::kV2, {config.vocab_size, 2560});
  try {
    mllm::models::qwen3_5::validateModelConfigMatch(config, mismatched_v2);
    FAIL() << "Expected a model/config mismatch";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("model/config mismatch"), std::string::npos);
  }

  const auto matching_v1 = parameterFileWithEmbedding(mllm::ModelFileVersion::kV1, {expected_numel});
  EXPECT_NO_THROW(mllm::models::qwen3_5::validateModelConfigMatch(config, matching_v1));

  const auto mismatched_v1 = parameterFileWithEmbedding(mllm::ModelFileVersion::kV1, {expected_numel + 1});
  EXPECT_THROW(mllm::models::qwen3_5::validateModelConfigMatch(config, mismatched_v1), std::invalid_argument);

  const auto mismatched_dtype =
      parameterFileWithEmbedding(mllm::ModelFileVersion::kV2, {config.vocab_size, config.hidden_size}, mllm::kFloat16);
  EXPECT_THROW(mllm::models::qwen3_5::validateModelConfigMatch(config, mismatched_dtype), std::invalid_argument);
}

TEST(Qwen35ConfigTest, MultimodalConfigRequiresMatchingV2VisionDescriptors) {
  const auto config = loadMultimodalConfig();
  auto parameter_file = parameterFileWithEmbedding(mllm::ModelFileVersion::kV2, {config.vocab_size, config.hidden_size});

  EXPECT_THROW(mllm::models::qwen3_5::validateModelConfigMatch(config, parameter_file), std::invalid_argument);
  pushDescriptor(parameter_file, "model.visual.patch_embed.proj.weight",
                 {config.vision_hidden_size, config.vision_in_channels, config.vision_temporal_patch_size,
                  config.vision_patch_size, config.vision_patch_size});
  pushDescriptor(parameter_file, "model.visual.pos_embed.weight",
                 {config.vision_num_position_embeddings, config.vision_hidden_size});
  EXPECT_NO_THROW(mllm::models::qwen3_5::validateModelConfigMatch(config, parameter_file));

  const auto v1 = parameterFileWithEmbedding(mllm::ModelFileVersion::kV1, {config.vocab_size * config.hidden_size});
  EXPECT_THROW(mllm::models::qwen3_5::validateModelConfigMatch(config, v1), std::invalid_argument);
}

TEST(Qwen35ConfigTest, RejectsNonOfficialRuntimeContractWithMatchingEmbedding) {
  const auto official_config = loadConfig("0.8B");
  const auto matching_v2 =
      parameterFileWithEmbedding(mllm::ModelFileVersion::kV2, {official_config.vocab_size, official_config.hidden_size});

  auto expect_mismatch = [&](const mllm::models::qwen3_5::Qwen3_5Config& config) {
    EXPECT_EQ(mllm::models::qwen3_5::modelNameForConfig(config), "Qwen3.5 text model");
    try {
      mllm::models::qwen3_5::validateModelConfigMatch(config, matching_v2);
      FAIL() << "Expected a model/config mismatch";
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string(error.what()).find("model/config mismatch"), std::string::npos);
    }
  };

  auto wrong_head_dim = official_config;
  wrong_head_dim.head_dim = 320;
  expect_mismatch(wrong_head_dim);

  auto wrong_layer_schedule = official_config;
  wrong_layer_schedule.layer_types[0] = "full_attention";
  expect_mismatch(wrong_layer_schedule);

  auto wrong_gdn_dim = official_config;
  wrong_gdn_dim.linear_value_head_dim = 64;
  expect_mismatch(wrong_gdn_dim);

  auto wrong_linear_impl = official_config;
  wrong_linear_impl.linear_impl_type = mllm::aops::LinearImplTypes::kDefault;
  expect_mismatch(wrong_linear_impl);
}
