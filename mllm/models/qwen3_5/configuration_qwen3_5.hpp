// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cmath>
#include <string>
#include <stdexcept>
#include <vector>

#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::qwen3_5 {

struct Qwen3_5Config : protected ConfigFile {
  Qwen3_5Config() { populateLayerTypes(); }

  explicit Qwen3_5Config(const std::string& file_path) : ConfigFile(file_path) {
    // The Qwen3.5 config nests text params under "text_config"
    auto& tc = data().contains("text_config") ? data()["text_config"] : data();

    attention_bias = tc["attention_bias"];
    hidden_size = tc["hidden_size"];
    intermediate_size = tc["intermediate_size"];
    num_attention_heads = tc["num_attention_heads"];
    num_key_value_heads = tc["num_key_value_heads"];
    num_hidden_layers = tc["num_hidden_layers"];
    max_position_embeddings = tc["max_position_embeddings"];
    rms_norm_eps = tc["rms_norm_eps"];
    vocab_size = tc["vocab_size"];
    head_dim = tc["head_dim"];
    tie_word_embeddings = tc.value("tie_word_embeddings", data().value("tie_word_embeddings", tie_word_embeddings));
    hidden_act = tc.value("hidden_act", hidden_act);
    mamba_ssm_dtype = tc.value("mamba_ssm_dtype", mamba_ssm_dtype);

    // Qwen3.5 hybrid attention
    attn_output_gate = tc.value("attn_output_gate", true);
    full_attention_interval = tc.value("full_attention_interval", 4);

    // GDN (Gated Delta Network) parameters
    linear_num_key_heads = tc.value("linear_num_key_heads", 16);
    linear_num_value_heads = tc.value("linear_num_value_heads", 16);
    linear_key_head_dim = tc.value("linear_key_head_dim", 128);
    linear_value_head_dim = tc.value("linear_value_head_dim", 128);
    linear_conv_kernel_dim = tc.value("linear_conv_kernel_dim", 4);

    // RoPE parameters (nested under rope_parameters)
    if (tc.contains("rope_parameters")) {
      auto& rp = tc["rope_parameters"];
      rope_theta = rp.value("rope_theta", 10000000.0f);
      partial_rotary_factor = rp.value("partial_rotary_factor", 0.25f);
    }

    // Layer types: explicit list or computed from full_attention_interval
    if (tc.contains("layer_types")) {
      for (auto& lt : tc["layer_types"]) { layer_types.push_back(lt.get<std::string>()); }
    } else {
      populateLayerTypes();
    }

    // Token IDs — Qwen3.5 uses different IDs than Qwen3
    if (tc.contains("eos_token_id")) { eos_token_id = tc["eos_token_id"]; }

    if (data().contains("max_cache_length")) {
      max_cache_length = data()["max_cache_length"];
    } else if (tc.contains("max_cache_length")) {
      max_cache_length = tc["max_cache_length"];
    }
    std::string linear_impl_name;
    bool has_linear_impl = false;
    if (tc.contains("linear_impl_type")) {
      linear_impl_name = tc["linear_impl_type"].get<std::string>();
      has_linear_impl = true;
    } else if (data().contains("linear_impl_type")) {
      linear_impl_name = data()["linear_impl_type"].get<std::string>();
      has_linear_impl = true;
    }
    if (has_linear_impl) {
      linear_impl_type = aops::str2LinearImplTypes(linear_impl_name);
      if (linear_impl_type == aops::LinearImplTypes::kDefault && linear_impl_name != "Default") {
        throw std::invalid_argument("Qwen3.5 contains an unsupported linear_impl_type: " + linear_impl_name);
      }
    }

    if (num_hidden_layers <= 0 || hidden_size <= 0 || head_dim <= 0 || intermediate_size <= 0 || vocab_size <= 0
        || max_position_embeddings <= 0 || num_attention_heads <= 0 || num_key_value_heads <= 0
        || num_attention_heads % num_key_value_heads != 0) {
      throw std::invalid_argument("Qwen3.5 contains invalid transformer dimensions");
    }
    if (linear_num_key_heads <= 0 || linear_num_value_heads <= 0 || linear_num_value_heads % linear_num_key_heads != 0
        || linear_key_head_dim <= 0 || linear_value_head_dim <= 0 || linear_conv_kernel_dim <= 1) {
      throw std::invalid_argument("Qwen3.5 contains invalid GDN dimensions");
    }
    if (layer_types.size() != static_cast<size_t>(num_hidden_layers)) {
      throw std::invalid_argument("Qwen3.5 layer_types size must equal num_hidden_layers");
    }
    for (const auto& layer_type : layer_types) {
      if (layer_type != "full_attention" && layer_type != "linear_attention") {
        throw std::invalid_argument("Qwen3.5 layer_types contains an unsupported value: " + layer_type);
      }
    }
    if (rotary_dim() <= 0 || rotary_dim() > head_dim || rotary_dim() % 2 != 0) {
      throw std::invalid_argument("Qwen3.5 partial rotary dimension must be positive, even, and no larger than head_dim");
    }
    if (!std::isfinite(rope_theta) || rope_theta <= 0.0F || !std::isfinite(rms_norm_eps) || rms_norm_eps <= 0.0F) {
      throw std::invalid_argument("Qwen3.5 rope_theta and rms_norm_eps must be finite and positive");
    }
    if (max_cache_length <= 0) { throw std::invalid_argument("Qwen3.5 max_cache_length must be positive"); }
    if (!tie_word_embeddings) { throw std::invalid_argument("Qwen3.5 CPU currently requires tie_word_embeddings=true"); }
  }

  // Standard transformer params
  bool attention_bias = false;
  int32_t hidden_size = 1024;
  int32_t head_dim = 256;
  int32_t intermediate_size = 3584;
  int32_t num_attention_heads = 8;
  int32_t num_key_value_heads = 2;
  int32_t num_hidden_layers = 24;
  int32_t max_position_embeddings = 262144;
  float rms_norm_eps = 1e-06;
  int32_t vocab_size = 248320;
  std::string hidden_act = "silu";
  std::string mamba_ssm_dtype = "float32";

  // Qwen3.5-specific: hybrid attention
  bool attn_output_gate = true;
  int32_t full_attention_interval = 4;
  std::vector<std::string> layer_types;  // "full_attention" or "linear_attention"

  // Qwen3.5-specific: partial RoPE
  float partial_rotary_factor = 0.25;
  float rope_theta = 10000000.0;
  [[nodiscard]] int32_t rotary_dim() const { return static_cast<int32_t>(head_dim * partial_rotary_factor); }

  // Qwen3.5-specific: GDN (Gated Delta Network) params
  int32_t linear_num_key_heads = 16;
  int32_t linear_num_value_heads = 16;
  int32_t linear_key_head_dim = 128;
  int32_t linear_value_head_dim = 128;
  int32_t linear_conv_kernel_dim = 4;

  // Token IDs
  int64_t eos_token_id = 248044;
  int64_t end_of_text_token_id = 248044;
  int64_t im_start_token_id = 248045;
  int64_t im_end_token_id = 248046;
  int64_t thinking_start_token_id = 248068;
  int64_t thinking_end_token_id = 248069;

  bool tie_word_embeddings = true;
  int32_t max_cache_length = 2048;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;

  // Helpers
  [[nodiscard]] bool isFullAttentionLayer(int layer_idx) const {
    return layer_types.at(static_cast<size_t>(layer_idx)) == "full_attention";
  }
  [[nodiscard]] int32_t numFullAttentionLayers() const {
    int32_t count = 0;
    for (auto& lt : layer_types) {
      if (lt == "full_attention") ++count;
    }
    return count;
  }
  [[nodiscard]] int32_t numGDNLayers() const { return num_hidden_layers - numFullAttentionLayers(); }

 private:
  void populateLayerTypes() {
    if (full_attention_interval <= 0) { throw std::invalid_argument("Qwen3.5 full_attention_interval must be positive"); }
    if (num_hidden_layers <= 0) { throw std::invalid_argument("Qwen3.5 num_hidden_layers must be positive"); }
    layer_types.clear();
    layer_types.reserve(static_cast<size_t>(num_hidden_layers));
    for (int32_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
      layer_types.push_back((layer_idx + 1) % full_attention_interval == 0 ? "full_attention" : "linear_attention");
    }
  }
};

/// Checks that \p config uses the official Qwen3.5 hybrid layer schedule, i.e. a
/// full-attention interval of 4 with every 4th layer marked "full_attention" and all
/// other layers marked "linear_attention".
/// \param config Qwen3.5 text-tower configuration to inspect.
/// \return true when the layer-type list exactly matches the official schedule.
inline auto hasOfficialLayerSchedule(const Qwen3_5Config& config) -> bool {
  if (config.full_attention_interval != 4 || config.layer_types.size() != static_cast<size_t>(config.num_hidden_layers)) {
    return false;
  }
  for (int32_t layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
    const auto expected_type = (layer_idx + 1) % 4 == 0 ? "full_attention" : "linear_attention";
    if (config.layer_types[static_cast<size_t>(layer_idx)] != expected_type) { return false; }
  }
  return true;
}

/// Checks the size-independent half of the supported mobile CPU runtime contract:
/// attention/GDN flags, head and vocabulary geometry, activation and norm settings,
/// RoPE parameters, special-token ids, tied embeddings, cache length, the official
/// layer schedule, and the KleidiAI W4A32 Linear implementation.
/// \param config Qwen3.5 text-tower configuration to inspect.
/// \return true when every shared runtime-contract field matches the official value.
inline auto hasOfficialCommonRuntimeContract(const Qwen3_5Config& config) -> bool {
  constexpr auto kKaiLinearImpl = aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32;
  return !config.attention_bias && config.attn_output_gate && config.head_dim == 256 && config.max_position_embeddings == 262144
         && config.rms_norm_eps == 1e-6F && config.vocab_size == 248320 && config.hidden_act == "silu"
         && config.mamba_ssm_dtype == "float32" && hasOfficialLayerSchedule(config) && config.linear_num_key_heads == 16
         && config.linear_key_head_dim == 128 && config.linear_value_head_dim == 128 && config.linear_conv_kernel_dim == 4
         && config.rope_theta == 10000000.0F && config.partial_rotary_factor == 0.25F && config.eos_token_id == 248044
         && config.im_end_token_id == 248046 && config.tie_word_embeddings && config.max_cache_length == 2048
         && config.linear_impl_type == kKaiLinearImpl;
}

/// Checks whether \p config is the official Qwen3.5-0.8B runtime configuration: the
/// shared contract plus the 0.8B hidden/intermediate sizes, layer count, and head counts.
/// \param config Qwen3.5 text-tower configuration to inspect.
/// \return true only for the supported 0.8B variant.
inline auto isOfficialQwen35_08BRuntimeConfig(const Qwen3_5Config& config) -> bool {
  return hasOfficialCommonRuntimeContract(config) && config.hidden_size == 1024 && config.intermediate_size == 3584
         && config.num_hidden_layers == 24 && config.num_attention_heads == 8 && config.num_key_value_heads == 2
         && config.linear_num_value_heads == 16;
}

/// Checks whether \p config is the official Qwen3.5-4B runtime configuration: the shared
/// contract plus the 4B hidden/intermediate sizes, layer count, and head counts.
/// \param config Qwen3.5 text-tower configuration to inspect.
/// \return true only for the supported 4B variant.
inline auto isOfficialQwen35_4BRuntimeConfig(const Qwen3_5Config& config) -> bool {
  return hasOfficialCommonRuntimeContract(config) && config.hidden_size == 2560 && config.intermediate_size == 9216
         && config.num_hidden_layers == 32 && config.num_attention_heads == 16 && config.num_key_value_heads == 4
         && config.linear_num_value_heads == 32;
}

/// Checks whether \p config is one of the runtime contracts this CPU runner supports.
/// \param config Qwen3.5 text-tower configuration to inspect.
/// \return true when the configuration is the official 0.8B or 4B variant.
inline auto matchesOfficialRuntimeContract(const Qwen3_5Config& config) -> bool {
  return isOfficialQwen35_08BRuntimeConfig(config) || isOfficialQwen35_4BRuntimeConfig(config);
}

/// Resolves a human-readable model name for diagnostics and error messages.
/// \param config Qwen3.5 text-tower configuration to inspect.
/// \return "Qwen3.5-0.8B" or "Qwen3.5-4B" for a supported variant, otherwise the generic
/// fallback "Qwen3.5 text model".
inline auto modelNameForConfig(const Qwen3_5Config& config) -> std::string {
  if (isOfficialQwen35_08BRuntimeConfig(config)) { return "Qwen3.5-0.8B"; }
  if (isOfficialQwen35_4BRuntimeConfig(config)) { return "Qwen3.5-4B"; }
  return "Qwen3.5 text model";
}

/// Verifies that a loaded parameter file matches the requested runtime configuration
/// before the model is built, so a mismatched checkpoint fails early with a descriptive
/// message instead of later during inference.
/// \param config Qwen3.5 text-tower configuration to validate against.
/// \param parameter_file Loaded V1 or V2 parameter file providing the embedding tensor.
/// \throws std::invalid_argument when the runtime configuration is not a supported
/// official contract; when \p parameter_file is null or lacks
/// model.language_model.embed_tokens.weight; when the embedding dtype is not float32;
/// when a V1 embedding has an element count other than vocab_size * hidden_size; or when
/// a V2 embedding shape is not exactly [vocab_size, hidden_size].
inline void validateModelConfigMatch(const Qwen3_5Config& config, const ParameterFile::ptr_t& parameter_file) {
  constexpr auto kEmbeddingWeight = "model.language_model.embed_tokens.weight";
  const auto model_name = modelNameForConfig(config);
  if (!matchesOfficialRuntimeContract(config)) {
    throw std::invalid_argument("Qwen3.5 model/config mismatch: CPU runner supports only the official Qwen3.5-0.8B and "
                                "Qwen3.5-4B runtime contracts");
  }
  if (parameter_file == nullptr || !parameter_file->has(kEmbeddingWeight)) {
    throw std::invalid_argument("Qwen3.5 model/config mismatch: " + model_name + " requires " + kEmbeddingWeight);
  }

  const auto embedding = parameter_file->pull(kEmbeddingWeight);
  if (embedding.dtype() != kFloat32) {
    throw std::invalid_argument("Qwen3.5 model/config mismatch: " + model_name + " requires a float32 embedding tensor");
  }
  const auto expected_numel = static_cast<size_t>(config.vocab_size) * static_cast<size_t>(config.hidden_size);
  if (parameter_file->version() == ModelFileVersion::kV1) {
    if (embedding.numel() != expected_numel) {
      throw std::invalid_argument("Qwen3.5 model/config mismatch: " + model_name + " expects " + std::to_string(expected_numel)
                                  + " embedding elements, got " + std::to_string(embedding.numel()));
    }
    return;
  }

  const auto shape = embedding.shape();
  if (shape.size() != 2 || shape[0] != config.vocab_size || shape[1] != config.hidden_size) {
    throw std::invalid_argument("Qwen3.5 model/config mismatch: " + model_name + " expects embedding shape ["
                                + std::to_string(config.vocab_size) + ", " + std::to_string(config.hidden_size) + "]");
  }
}

}  // namespace mllm::models::qwen3_5
