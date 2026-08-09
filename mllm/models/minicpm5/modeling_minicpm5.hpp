// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

#include "mllm/mllm.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/minicpm5/configuration_minicpm5.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/llm_components/GroupedQueryAttention.hpp"
#include "mllm/nn/lmcache/KVHeadStaticCache.hpp"
#include "mllm/utils/Enumerate.hpp"

namespace mllm::models::minicpm5 {

inline auto makeMiniCPM5RoPEInvFreq(int32_t output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  for (int32_t dim = 0; dim < output_dim / 2; ++dim) {
    inv_freq.ptr<float>()[dim] = 1.0F / std::pow(rope_theta, 2.0F * static_cast<float>(dim) / output_dim);
  }
  return inv_freq;
}

inline auto makeMiniCPM5RotaryPosEmbedding(const Tensor& position_ids, const Tensor& inv_freq) -> std::pair<Tensor, Tensor> {
  const int32_t batch = position_ids.shape()[0];
  const int32_t sequence = position_ids.shape()[1];
  const int32_t half_dim = inv_freq.shape()[0];
  const int32_t dim = half_dim * 2;
  auto sin_embedding = Tensor::empty({batch, sequence, dim}, kFloat32, kCPU).alloc();
  auto cos_embedding = Tensor::empty({batch, sequence, dim}, kFloat32, kCPU).alloc();

  for (int32_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (int32_t sequence_index = 0; sequence_index < sequence; ++sequence_index) {
      const auto position = position_ids.ptr<int64_t>()[batch_index * sequence + sequence_index];
      for (int32_t index = 0; index < half_dim; ++index) {
        const float frequency = static_cast<float>(position) * inv_freq.ptr<float>()[index];
        const float sine = std::sin(frequency);
        const float cosine = std::cos(frequency);
        const auto offset = (batch_index * sequence + sequence_index) * dim + index;
        sin_embedding.ptr<float>()[offset] = sine;
        sin_embedding.ptr<float>()[offset + half_dim] = sine;
        cos_embedding.ptr<float>()[offset] = cosine;
        cos_embedding.ptr<float>()[offset + half_dim] = cosine;
      }
    }
  }
  return {sin_embedding, cos_embedding};
}

class MiniCPM5MLP final : public nn::Module {
 public:
  MiniCPM5MLP() = default;
  MiniCPM5MLP(const std::string& name, const MiniCPM5Config& config) : nn::Module(name) {
    gate_proj_ = reg<nn::Linear>("gate_proj", config.hidden_size, config.intermediate_size, false, config.linear_impl_type);
    up_proj_ = reg<nn::Linear>("up_proj", config.hidden_size, config.intermediate_size, false, config.linear_impl_type);
    down_proj_ = reg<nn::Linear>("down_proj", config.intermediate_size, config.hidden_size, false, config.linear_impl_type);
    activation_ = reg<nn::SiLU>("act");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto gate = activation_(gate_proj_(inputs[0]));
    auto up = up_proj_(inputs[0]);
    return {down_proj_(gate * up)};
  }

 private:
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU activation_;
};

class MiniCPM5Attention final : public nn::Module {
 public:
  MiniCPM5Attention() = default;
  MiniCPM5Attention(const std::string& name, const MiniCPM5Config& config) : nn::Module(name) {
    hidden_size_ = config.hidden_size;
    head_dim_ = config.head_dim;
    query_heads_ = config.num_attention_heads;
    kv_heads_ = config.num_key_value_heads;
    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, query_heads_ * head_dim_, config.attention_bias, config.linear_impl_type);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, kv_heads_ * head_dim_, config.attention_bias, config.linear_impl_type);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, kv_heads_ * head_dim_, config.attention_bias, config.linear_impl_type);
    o_proj_ = reg<nn::Linear>("o_proj", query_heads_ * head_dim_, hidden_size_, config.attention_bias, config.linear_impl_type);
    q_rope_ = reg<nn::RoPE>("q_rope", config.rope_theta, config.max_position_embeddings, config.head_dim);
    k_rope_ = reg<nn::RoPE>("k_rope", config.rope_theta, config.max_position_embeddings, config.head_dim);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    const auto& input = inputs[0];
    const int32_t batch = input.shape()[0];
    const int32_t sequence = input.shape()[1];
    auto* cache = args[0].get<nn::KVHeadStaticCache*>();

    auto query = q_proj_(input).view({batch, sequence, query_heads_, head_dim_}).transpose(1, 2);
    auto key = k_proj_(input).view({batch, sequence, kv_heads_, head_dim_}).transpose(1, 2);
    auto value = v_proj_(input).view({batch, sequence, kv_heads_, head_dim_}).transpose(1, 2);
    query = q_rope_(query, inputs[1], inputs[2]);
    key = k_rope_(key, inputs[1], inputs[2]);

    auto [cached_key, cached_value] = cache->updateKVCache(logical_cache_slot_, key, value);
    auto output = nn::llm_components::groupedQueryAttention(query, cached_key, cached_value);
    output = output.transpose(1, 2).view({batch, sequence, query_heads_ * head_dim_});
    return {o_proj_(output)};
  }

  int32_t logical_cache_slot_ = 0;

 private:
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::RoPE q_rope_;
  nn::RoPE k_rope_;
  int32_t hidden_size_ = 0;
  int32_t head_dim_ = 0;
  int32_t query_heads_ = 0;
  int32_t kv_heads_ = 0;
};

class MiniCPM5Decoder final : public nn::Module {
 public:
  MiniCPM5Decoder() = default;
  MiniCPM5Decoder(const std::string& name, const MiniCPM5Config& config) : nn::Module(name) {
    self_attn_ = reg<MiniCPM5Attention>("self_attn", config);
    mlp_ = reg<MiniCPM5MLP>("mlp", config);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", config.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", config.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden = input_layer_norm_(inputs[0]);
    hidden = self_attn_(hidden, inputs[1], inputs[2], args[0])[0] + inputs[0];
    return {mlp_(post_attention_layer_norm_(hidden))[0] + hidden};
  }

  MiniCPM5Attention self_attn_;

 private:
  MiniCPM5MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;
};

class MiniCPM5Text final : public nn::Module {
 public:
  MiniCPM5Text() = default;
  MiniCPM5Text(const std::string& name, const MiniCPM5Config& config) : nn::Module(name) {
    embedding_ = reg<nn::Embedding>("embed_tokens", config.vocab_size, config.hidden_size);
    layers_ = reg<nn::ModuleList<MiniCPM5Decoder>>("layers", config.num_hidden_layers, config);
    for (auto [slot, layer] : enumerate(layers_.list())) { layer.self_attn_.logical_cache_slot_ = slot; }
    norm_ = reg<nn::RMSNorm>("norm", config.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden = embedding_(inputs[0]);
    for (auto& layer : layers_.list()) { hidden = layer(hidden, inputs[1], inputs[2], args[0])[0]; }
    return {norm_(hidden)};
  }

 private:
  nn::Embedding embedding_;
  nn::ModuleList<MiniCPM5Decoder> layers_;
  nn::RMSNorm norm_;
};

class MiniCPM5ForCausalLM final : public ARGeneration, public nn::Module {
 public:
  explicit MiniCPM5ForCausalLM(const MiniCPM5Config& config)
      : config_(config),
        kv_cache_(config.max_cache_length, config.num_hidden_layers, config.num_key_value_heads, config.head_dim) {
    eos_token_id_ = config.eos_token_ids.front();
    for (size_t index = 1; index < config.eos_token_ids.size(); ++index) {
      additional_eos_token_ids_.insert(config.eos_token_ids[index]);
    }
    max_length_ = config.max_cache_length;
    model_ = reg<MiniCPM5Text>("model", config);
    lm_head_ = reg<nn::Linear>("lm_head", config.hidden_size, config.vocab_size, false, config.linear_impl_type);
    registerBuffer("inv_freq", makeMiniCPM5RoPEInvFreq(config.head_dim, config.rope_theta));
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    const auto sequence = input.at("sequence");
    const auto shape = sequence.shape();
    if (shape.size() != 2 || shape[0] != 1 || shape[1] <= 0 || sequence.dtype() != kInt64 || sequence.device() != kCPU) {
      throw std::invalid_argument("MiniCPM5 CPU expects a non-empty rank-2 int64 CPU sequence with batch size 1");
    }

    const int32_t cached_tokens = kv_cache_.getCurrentSeqCnt(0);
    if (shape[1] > config_.max_cache_length || cached_tokens > config_.max_cache_length - shape[1]) {
      throw std::invalid_argument("MiniCPM5 sequence exceeds the configured KV-cache capacity");
    }
    for (int32_t index = 0; index < shape[1]; ++index) {
      const auto token_id = sequence.ptr<int64_t>()[index];
      if (token_id < 0 || token_id >= config_.vocab_size) {
        throw std::invalid_argument("MiniCPM5 sequence contains a token id outside the vocabulary");
      }
    }

    auto position_ids = Tensor::empty({1, shape[1]}, kInt64, kCPU).alloc();
    for (int32_t index = 0; index < shape[1]; ++index) { position_ids.ptr<int64_t>()[index] = cached_tokens + index; }
    auto [sine, cosine] = makeMiniCPM5RotaryPosEmbedding(position_ids, getBuffer("inv_freq"));
    auto hidden = model_(sequence, sine, cosine, AnyValue(&kv_cache_))[0];
    hidden = hidden[{kAll, {hidden.shape()[1] - 1}, kAll}];

    return {
        {"sequence", lm_head_(hidden)},
        {"position_ids", position_ids},
    };
  }

  void resetState(int32_t batch_size = 1) {
    if (batch_size != 1) { throw std::invalid_argument("MiniCPM5 CPU currently supports batch size 1 only"); }
    kv_cache_.clearCache();
  }

  [[nodiscard]] nn::KVHeadStaticCache& kvCache() { return kv_cache_; }

 private:
  MiniCPM5Config config_;
  MiniCPM5Text model_;
  nn::Linear lm_head_;
  nn::KVHeadStaticCache kv_cache_;
};

}  // namespace mllm::models::minicpm5
