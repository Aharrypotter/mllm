// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cmath>
#include <string>
#include <stdexcept>
#include <vector>

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::qwen3_5 {

struct Qwen3_5Config : protected ConfigFile {
  Qwen3_5Config() = default;

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
      if (full_attention_interval <= 0) { throw std::invalid_argument("Qwen3.5 full_attention_interval must be positive"); }
      for (int i = 0; i < num_hidden_layers; ++i) {
        if ((i + 1) % full_attention_interval == 0) {
          layer_types.push_back("full_attention");
        } else {
          layer_types.push_back("linear_attention");
        }
      }
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
};

}  // namespace mllm::models::qwen3_5
