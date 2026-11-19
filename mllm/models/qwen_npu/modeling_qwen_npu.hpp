// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>

#include "mllm/backends/cpu/kernels/arm/cast_types.hpp"
#include "mllm/backends/qnn/CustomLayers.hpp"
#include "mllm/compile/ir/Trace.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/layers/Param.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "configuration_qwen_npu.hpp"

namespace mllm::models::qwen_npu {

inline auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0f / std::pow(rope_theta, 2.0f * i / output_dim); }
  return inv_freq;
}

inline auto stageHmxPrefillKV(Tensor states) -> Tensor {
#if defined(MLLM_HOST_ARCH_ARM64)
  static const bool direct_cast = [] {
    const char* value = std::getenv("MLLM_HMX_DIRECT_KV_CAST");
    return value && value[0] == '1' && value[1] == '\0';
  }();
  if (direct_cast && !Context::instance().thisThread()->trace_mode && states.device() == kCPU
      && states.dtype() == kFloat32 && !states.isContiguous()) {
    const auto shape = states.shape();
    const auto stride = states.stride();
    if (shape.size() == 4 && shape[0] == 1 && shape[1] > 0 && shape[2] > 0 && shape[3] > 0
        && stride[3] == 1 && stride[2] == shape[3]) {
      const int64_t head_elements = static_cast<int64_t>(shape[2]) * shape[3];
      if (head_elements <= std::numeric_limits<int>::max() && stride[1] >= head_elements) {
        auto staged = Tensor::empty(shape, kFloat16, kCPU).alloc();
        const auto* input = states.ptr<mllm_fp32_t>();
        auto* output = staged.ptr<mllm_fp16_t>();
        for (int64_t head = 0; head < shape[1]; ++head) {
          cpu::arm::fp32_2_fp16(input + head * stride[1], output + head * head_elements,
                                static_cast<int>(head_elements), 1);
        }
        static std::once_flag log_first_cast;
        std::call_once(log_first_cast, [&] {
          MLLM_INFO("SHADOWNPU_HMX_DIRECT_KV_CAST head_stride={} live_head_elements={} heads={}", stride[1],
                    head_elements, shape[1]);
        });
        return staged;
      }
    }
  }
#endif
  // The live BHSD view can retain the full-capacity head stride. The normal
  // cast expects contiguous data, including each physically repeated GQA head.
  return states.contiguous().to(kFloat16);
}

inline auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq, float attention_scaling = 1.0f)
    -> std::pair<Tensor, Tensor> {
  auto batch_size = position_ids.shape()[0];
  auto seq_len = position_ids.shape()[1];
  auto inv_freq_len = inv_freq.shape()[0];
  auto dim = inv_freq_len * 2;

  // Create freqs tensor: position_ids @ inv_freq
  auto freqs = Tensor::empty({batch_size, seq_len, inv_freq_len}, kFloat32, kCPU).alloc();
  auto freqs_ptr = freqs.ptr<float>();
  auto position_ids_ptr = position_ids.ptr<int64_t>();
  auto inv_freq_ptr = inv_freq.ptr<float>();

  // Compute freqs = position_ids[:, :, None] @ inv_freq[None, :]
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      auto pos = position_ids_ptr[b * seq_len + s];
      for (int d = 0; d < inv_freq_len; ++d) {
        freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d] = static_cast<float>(pos) * inv_freq_ptr[d];
      }
    }
  }

  // Create sin and cos tensors with shape [batch_size, seq_len, dim]
  auto sin_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto cos_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto sin_ptr = sin_emb.ptr<float>();
  auto cos_ptr = cos_emb.ptr<float>();

  // Compute sin and cos embeddings: emb = [freqs, freqs]
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      for (int d = 0; d < inv_freq_len; ++d) {
        auto freq = freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d];
        auto sin_val = std::sin(freq) * attention_scaling;
        auto cos_val = std::cos(freq) * attention_scaling;

        // Store the same values in both halves: [freqs, freqs]
        sin_ptr[b * seq_len * dim + s * dim + d] = sin_val;
        sin_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = sin_val;
        cos_ptr[b * seq_len * dim + s * dim + d] = cos_val;
        cos_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = cos_val;
      }
    }
  }

  return {sin_emb, cos_emb};
}

class QwenMLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  QwenMLP() = default;
  QwenMLP(const std::string& name, const QwenNPUConfig& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = gate_proj_(inputs[0]);
    x = silu_(x);
    auto y = up_proj_(inputs[0]);
    x = x * y;
    x = down_proj_(x);
    return {x};
  }
};

class QwenAttentionProjNPU : public nn::Module {
  nn::RMSNorm input_layer_norm_;

  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  std::optional<nn::qnn::DequantizeAdd> q_dequantize_add_;
  std::optional<nn::qnn::DequantizeAdd> k_dequantize_add_;
  std::optional<nn::qnn::DequantizeAdd> v_dequantize_add_;

  // quantize scale should be attached to input tensor and use the prefix name q_proj
  // in QNN Backend, the quant scale key is always "qnn_quant_scale"
  nn::Param quantize_scale_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;
  float rope_theta_;
  int max_position_embeddings_;

 public:
  QwenAttentionProjNPU() = default;

  QwenAttentionProjNPU(const std::string& name, const QwenNPUConfig& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = hidden_size_ / num_attention_heads_;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    rope_theta_ = cfg.rope_theta;
    max_position_embeddings_ = cfg.max_position_embeddings;

    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);

    const bool fuse_bias = !cfg.qnn_qkv_bias_after_dequant;
    q_proj_ = reg<nn::Linear>("self_attn.q_proj",
                              aops::LinearOpOptions{.in_channels = hidden_size_,
                                                    .out_channels = head_dim_ * num_attention_heads_,
                                                    .bias = fuse_bias,
                                                    .impl_type = cfg.linear_impl_type,
                                                    .qnn_impl_type = aops::LinearImplTypes::kQNN_tensor_symm_w8a16});
    k_proj_ = reg<nn::Linear>("self_attn.k_proj",
                              aops::LinearOpOptions{.in_channels = hidden_size_,
                                                    .out_channels = head_dim_ * num_key_value_heads_,
                                                    .bias = fuse_bias,
                                                    .impl_type = cfg.linear_impl_type,
                                                    .qnn_impl_type = aops::LinearImplTypes::kQNN_tensor_symm_w8a16});
    v_proj_ = reg<nn::Linear>("self_attn.v_proj",
                              aops::LinearOpOptions{.in_channels = hidden_size_,
                                                    .out_channels = head_dim_ * num_key_value_heads_,
                                                    .bias = fuse_bias,
                                                    .impl_type = cfg.linear_impl_type,
                                                    .qnn_impl_type = aops::LinearImplTypes::kQNN_tensor_symm_w8a16});

    // quantize scale should be attached to input tensor to let QNN know the scale for quantization output
    quantize_scale_ = reg<nn::Param>("quantize_scale", getModuleName() + ".self_attn.q_proj.input_scale", Tensor::shape_t{1});
    if (!fuse_bias) {
      q_dequantize_add_ = reg<nn::qnn::DequantizeAdd>("self_attn.q_proj.dequantize", kFloat32,
                                                    head_dim_ * num_attention_heads_);
      k_dequantize_add_ = reg<nn::qnn::DequantizeAdd>("self_attn.k_proj.dequantize", kFloat32,
                                                    head_dim_ * num_key_value_heads_);
      v_dequantize_add_ = reg<nn::qnn::DequantizeAdd>("self_attn.v_proj.dequantize", kFloat32,
                                                    head_dim_ * num_key_value_heads_);
      // Register parameter paths above before giving the alternative QNN graph its own symbol.
      impl()->setAbsoluteName(getModuleName() + ".qkv_post_dequant_bias");
    }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[1];

    x = input_layer_norm_(x);

    x = x.view({B, S, 1, inputs[0].shape()[2]});

    // attach input scale for quantization
    x.attach("qnn_quant_scale", quantize_scale_.weight().impl());
    x = x.to(kInt16);

    // for FullyConnected Op in HTP, the output is 2D with [B * S, H * D]
    auto query_states = q_proj_(x);
    auto key_states = k_proj_(x);
    auto value_states = v_proj_(x);

    if (q_dequantize_add_) {
      query_states = (*q_dequantize_add_)(query_states);
      key_states = (*k_dequantize_add_)(key_states);
      value_states = (*v_dequantize_add_)(value_states);
    }

    // [B, S, H, D]
    query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
    value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

    // Convert the fused path from its calibrated post-bias INT16 range. The
    // separate QNN operations above already returned Float32 activations.
    query_states = query_states.to(kFloat32);
    key_states = key_states.to(kFloat32);
    value_states = value_states.to(kFloat32);

    // [B, H, S, D]
    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    return {query_states, key_states, value_states};
  }
};

class QwenAttentionMatmul final : public nn::Module {
  nn::RoPE q_rope_;
  nn::RoPE k_rope_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;

  nn::KVCache kv_cache_;
  nn::PatternSparseAttention pattern_sparse_attention_;
  nn::SparseSoftmaxValue sparse_softmax_value_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;
  float rope_theta_;
  int max_position_embeddings_;
  bool use_sparse_prefill_attention_ = false;
  bool use_pattern_sparse_prefill_attention_ = false;
  bool use_hmx_key_staging_ = false;

 public:
  QwenAttentionMatmul() = default;

  QwenAttentionMatmul(const std::string& name, const QwenNPUConfig& cfg, int32_t layer_idx) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = hidden_size_ / num_attention_heads_;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    rope_theta_ = cfg.rope_theta;
    max_position_embeddings_ = cfg.max_position_embeddings;

    q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings);
    k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);

    kv_cache_ =
        reg<nn::KVCache>("kv_cache", 0 /*only hold 1 layer*/, num_attention_heads_, num_key_value_heads_, head_dim_, false);

    if (cfg.prefill_attention_hmx_selector && !cfg.prefill_attention_pattern_sparse) {
      throw std::invalid_argument("Qwen ShadowNPU HMX selection requires pattern-sparse attention");
    }
    use_sparse_prefill_attention_ = cfg.useSparsePrefillAttention(layer_idx);
    use_pattern_sparse_prefill_attention_ = cfg.prefill_attention_pattern_sparse;
    use_hmx_key_staging_ = cfg.prefill_attention_hmx_selector;
    aops::PatternSparseAttentionOpOptions sparse_options;
    sparse_options.sparsity = cfg.prefillAttentionSparsityForLayer(layer_idx);
    sparse_options.causal_mask = true;
    sparse_options.local_ratio = cfg.prefill_attention_local_ratio;
    sparse_options.prefix_tokens = cfg.prefill_attention_prefix_tokens;
    sparse_options.dense_tokens = cfg.prefill_attention_dense_tokens;
    sparse_options.random_pattern = cfg.prefill_attention_random_pattern;
    sparse_options.random_seed = cfg.prefill_attention_random_seed + static_cast<uint32_t>(layer_idx) * 4099U;
    sparse_options.pack_reserve_tokens =
        cfg.prefill_attention_pack_reserve_tokens > 0 ? cfg.prefill_attention_pack_reserve_tokens : cfg.max_cache_length;
    sparse_options.hmx_selector = cfg.prefill_attention_hmx_selector;
    sparse_options.head_retentions = cfg.prefillAttentionHeadRetentionsForLayer(layer_idx);
    sparse_options.layer_id = layer_idx;
    pattern_sparse_attention_ = reg<nn::PatternSparseAttention>("pattern_sparse_attention", sparse_options);

    aops::SparseSoftmaxValueOpOptions sparse_softmax_value_options;
    sparse_softmax_value_options.sparsity = sparse_options.sparsity;
    sparse_softmax_value_options.causal_mask = true;
    sparse_softmax_value_options.dense_tokens = sparse_options.dense_tokens;
    sparse_softmax_value_options.head_retentions = sparse_options.head_retentions;
    sparse_softmax_value_ = reg<nn::SparseSoftmaxValue>("sparse_softmax_value", sparse_softmax_value_options);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto query_states = inputs[0];
    auto key_states = inputs[1];
    auto value_states = inputs[2];
    auto llm_embedding_sin = inputs[3];
    auto llm_embedding_cos = inputs[4];

    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[2];

    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    // [B, H, S, D]
    auto kv_outputs = kv_cache_(key_states, value_states);
    key_states = kv_outputs[0];
    value_states = kv_outputs[1];

    Tensor output;
    if (use_sparse_prefill_attention_ && S > 1) {
      if (use_pattern_sparse_prefill_attention_) {
        if (use_hmx_key_staging_) {
          const auto sparse_key_states = stageHmxPrefillKV(key_states);
          const auto sparse_value_states = stageHmxPrefillKV(value_states);
          output =
              pattern_sparse_attention_(query_states.to(kFloat32), sparse_key_states, sparse_value_states, sparse_key_states);
        } else {
          output = pattern_sparse_attention_(query_states.to(kFloat32), key_states, value_states);
        }
      } else {
        const auto logits =
            nn::functional::matmul(query_states.to(kFloat32), key_states.to(kFloat32), false, true) * (1.f / sqrtf(head_dim_));
        output = sparse_softmax_value_(logits, value_states);
      }
    } else {
      // attention weight
      // [B, H, S, S]
      Tensor attn;
      if (key_states.dtype() == kFloat32) {
        attn = nn::functional::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
        attn = mask_(attn);
        attn = softmax_(attn);
      } else if (key_states.dtype() == kFloat16) {
        attn =
            nn::functional::matmul(query_states.to(kFloat32), key_states.to(kFloat32), false, true) * (1.f / sqrtf(head_dim_));
        attn = mask_(attn);
        attn = softmax_(attn);
      }

      // [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
      output = nn::functional::matmul(attn, value_states.to(attn.dtype()));
    }
    // [B, H, S, D] -> [B, S, H, D] -> [B, S, H * D]
    output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});

    return {output};
  }

  nn::KVCache& getKVCache() { return kv_cache_; }
};

class QwenOutProjAndMLP final : public nn::Module {
  nn::Linear o_proj_;

  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

  nn::RMSNorm post_attention_layer_norm_;

  nn::Param o_proj_quantize_scale_;
  nn::Param pre_mlp_proj_quantize_scale_;
  nn::Param down_proj_quantize_scale_;

  int hidden_size_, intermediate_size_;
  bool shadow_down_proj_ = false;

 public:
  QwenOutProjAndMLP() = default;
  QwenOutProjAndMLP(const std::string& name, const QwenNPUConfig& cfg, bool shadow_down_proj) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    intermediate_size_ = cfg.intermediate_size;
    shadow_down_proj_ = shadow_down_proj;

    o_proj_ = reg<nn::Linear>("self_attn.o_proj",
                              aops::LinearOpOptions{.in_channels = cfg.hidden_size,
                                                    .out_channels = cfg.hidden_size,
                                                    .bias = false,
                                                    .impl_type = cfg.linear_impl_type,
                                                    .qnn_impl_type = aops::LinearImplTypes::kQNN_tensor_symm_w8a8});

    gate_proj_ =
        reg<nn::Linear>("mlp.gate_proj", aops::LinearOpOptions{.in_channels = cfg.hidden_size,
                                                               .out_channels = cfg.intermediate_size,
                                                               .bias = false,
                                                               .impl_type = cfg.linear_impl_type,
                                                               .qnn_impl_type = aops::LinearImplTypes::kQNN_tensor_symm_w8a8});
    silu_ = reg<nn::SiLU>("mlp.act", aops::SiLUOpOptions{.high_precision = true});
    up_proj_ =
        reg<nn::Linear>("mlp.up_proj", aops::LinearOpOptions{.in_channels = cfg.hidden_size,
                                                             .out_channels = cfg.intermediate_size,
                                                             .bias = false,
                                                             .impl_type = cfg.linear_impl_type,
                                                             .qnn_impl_type = aops::LinearImplTypes::kQNN_tensor_symm_w8a8});
    down_proj_ = reg<nn::Linear>("mlp.down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);

    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);

    o_proj_quantize_scale_ =
        reg<nn::Param>("o_proj_quantize_scale_", getModuleName() + ".self_attn.o_proj.input_scale", Tensor::shape_t{1});
    pre_mlp_proj_quantize_scale_ =
        reg<nn::Param>("pre_mlp_proj_quantize_scale_", getModuleName() + ".mlp.gate_proj.input_scale", Tensor::shape_t{1});
    down_proj_quantize_scale_ =
        reg<nn::Param>("down_proj_quantize_scale_", getModuleName() + ".mlp.down_proj.input_scale", Tensor::shape_t{1});
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[1];

    const auto& res = inputs[1];
    MLLM_RT_ASSERT(res.device() == kQNN);

    // [B, S, H * D] ->  [B * S, H * D]
    auto input = inputs[0];
    input = input.view({B, S, 1, hidden_size_});
    input.attach("qnn_quant_scale", o_proj_quantize_scale_.weight().impl());
    input = input.to(kInt8);
    auto x = o_proj_(input);
    x = x.view({B, S, 1, hidden_size_});
    x = x.to(kFloat32);
    x = x.view({B, S, hidden_size_});

    auto tmp = x + res;

    x = post_attention_layer_norm_(tmp);

    x = x.view({B, S, 1, hidden_size_});
    x.attach("qnn_quant_scale", pre_mlp_proj_quantize_scale_.weight().impl());
    x = x.to(kInt8);
    auto gate = gate_proj_(x);
    gate = gate.view({B, S, 1, intermediate_size_});
    gate = gate.to(kFloat32);

    gate = silu_(gate);
    auto up = up_proj_(x);
    up = up.view({B, S, 1, intermediate_size_});
    up = up.to(kFloat32);

    x = gate * up;
    auto shadow_input = x;

    x = x.view({B, S, 1, intermediate_size_});
    x.attach("qnn_quant_scale", down_proj_quantize_scale_.weight().impl());
    x = x.to(kInt16);
    x = down_proj_(x);
    x = x.view({B, S, 1, hidden_size_});
    auto quantized_down_proj = x;
    x = x.to(kFloat32);
    x = x.view({B, S, hidden_size_});

    x = x + tmp;
    if (shadow_down_proj_) { return {shadow_input, quantized_down_proj, x}; }
    return {x};
  }
};

class QwenDecoder final : public nn::Module {
  bool capture_debug_boundaries_ = false;
  std::vector<Tensor> captured_debug_boundaries_;

 public:
  QwenAttentionProjNPU self_attn_proj_;
  QwenAttentionMatmul self_attn_matmul_;
  QwenOutProjAndMLP self_attn_out_mlp_;
  nn::LinearShadowCorrection down_proj_shadow_correction_;
  bool shadow_down_proj_ = false;

  QwenDecoder() = default;

  QwenDecoder(const std::string& name, const QwenNPUConfig& cfg, int32_t layer_idx) : nn::Module(name) {
    shadow_down_proj_ = cfg.useShadowDownProjection(layer_idx);
    self_attn_proj_ = reg<QwenAttentionProjNPU>("", cfg);
    self_attn_proj_.to(kQNN);
    self_attn_matmul_ = reg<QwenAttentionMatmul>("self_attn", cfg, layer_idx);
    self_attn_out_mlp_ = reg<QwenOutProjAndMLP>("", cfg, shadow_down_proj_);
    self_attn_out_mlp_.to(kQNN);
    if (shadow_down_proj_) {
      down_proj_shadow_correction_ =
          reg<nn::LinearShadowCorrection>("mlp.down_proj.shadow", cfg.intermediate_size, cfg.hidden_size, 16);
    }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    captured_debug_boundaries_.clear();
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];

    x = x.to(kQNN);

    auto states = self_attn_proj_(x);

    auto query_states = states[0].to(kCPU);
    auto key_states = states[1].to(kCPU);
    auto value_states = states[2].to(kCPU);

    if (capture_debug_boundaries_) {
      captured_debug_boundaries_.push_back(query_states.transpose(1, 2).contiguous());
      captured_debug_boundaries_.push_back(key_states.transpose(1, 2).contiguous());
      captured_debug_boundaries_.push_back(value_states.transpose(1, 2).contiguous());
    }

    x = self_attn_matmul_(query_states, key_states, value_states, llm_embedding_sin, llm_embedding_cos)[0];
    if (capture_debug_boundaries_) { captured_debug_boundaries_.push_back(x.contiguous()); }

    x = x.to(kQNN);
    auto res = inputs[0];
    res = res.to(kQNN);

    auto out_mlp_states = self_attn_out_mlp_(x, res);
    if (shadow_down_proj_) {
      MLLM_RT_ASSERT_EQ(out_mlp_states.size(), 3);
      x = down_proj_shadow_correction_(out_mlp_states[0].to(kCPU).contiguous(), out_mlp_states[1].to(kCPU).contiguous(),
                                       out_mlp_states[2].to(kCPU).contiguous());
    } else {
      MLLM_RT_ASSERT_EQ(out_mlp_states.size(), 1);
      x = out_mlp_states[0];
    }

    return {x};
  }

  nn::KVCache& getKVCache() { return self_attn_matmul_.getKVCache(); }
  void setCaptureDebugBoundaries(bool capture) { capture_debug_boundaries_ = capture; }
  const std::vector<Tensor>& capturedDebugBoundaries() const { return captured_debug_boundaries_; }
};

class QwenText final : public nn::Module {
  nn::ModuleListWithIdx<QwenDecoder> decode_blocks_;
  nn::RMSNorm norm_;
  bool capture_layer_outputs_ = false;
  std::vector<Tensor> captured_layer_outputs_;

 public:
  QwenText() = default;

  QwenText(const std::string& name, const QwenNPUConfig& cfg) : nn::Module(name) {
    decode_blocks_ = reg<nn::ModuleListWithIdx<QwenDecoder>>("layers", cfg.num_hidden_layers, cfg);

    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    embedding_.to(kQNN);  // use QNN version of embedding (handle padding token, execute on CPU)

    // Initialize inv_freq for RoPE
    auto inv_freq = makeRoPEInvFreq(cfg.hidden_size / cfg.num_attention_heads, cfg.rope_theta);
    registerBuffer("inv_freq", inv_freq);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();

    // X is already embedded
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];

    captured_layer_outputs_.clear();
    if (capture_layer_outputs_) { captured_layer_outputs_.push_back(x.to(kCPU).contiguous()); }
    for (auto& block : blocks) {
      x = block(x, llm_embedding_sin, llm_embedding_cos)[0];
      if (capture_layer_outputs_) { captured_layer_outputs_.push_back(x.to(kCPU).contiguous()); }
    }

    x = x.to(kCPU);
    x = norm_(x);
    if (capture_layer_outputs_) { captured_layer_outputs_.push_back(x.contiguous()); }

    return {x};
  }

  nn::Embedding embedding_;

  nn::ModuleListWithIdx<QwenDecoder>& decode_blocks() { return decode_blocks_; }
  void setCaptureLayerOutputs(bool capture) {
    capture_layer_outputs_ = capture;
    auto& blocks = decode_blocks_.list();
    for (auto& block : blocks) { block.setCaptureDebugBoundaries(capture); }
  }
  const std::vector<Tensor>& capturedLayerOutputs() const { return captured_layer_outputs_; }
  size_t capturedBoundaryLayerCount() { return decode_blocks_.list().size(); }
  const std::vector<Tensor>& capturedLayerBoundaries(size_t layer_index) {
    auto& blocks = decode_blocks_.list();
    MLLM_RT_ASSERT(layer_index < blocks.size());
    return blocks[layer_index].capturedDebugBoundaries();
  }
  const std::vector<Tensor>& capturedLayerZeroBoundaries() { return capturedLayerBoundaries(0); }

  void clearKVCache() {
    for (auto& block : decode_blocks_.list()) { block.getKVCache().clearCache(); }
  }

  // Manually synchronize KV cache logical sequence length for all layers.
  // This is mainly used in decoding loop where we reuse a fixed-size
  // KV buffer (e.g. 1K) but only part of it is valid.
  void setKVCacheSeqLen(int32_t seq_len) {
    for (auto& block : decode_blocks_.list()) { block.getKVCache().setCurrentSeqCnt(seq_len); }
  }
};

class QwenForCausalLM : public nn::Module, public ARGeneration {
 public:
  QwenForCausalLM() = default;
  explicit QwenForCausalLM(const std::string& name, const QwenNPUConfig& cfg) : cfg(cfg), nn::Module(name) {
    model = reg<QwenText>("model", cfg);
    if (!cfg.tie_word_embeddings) {
      lm_head_ = reg<nn::Linear>("lm_head", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
    }
    tie_word_embeddings_ = cfg.tie_word_embeddings;
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");

    // Generate position_ids for the current sequence
    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];

    auto real_seq = args.count("seq_len") ? args.at("seq_len").get<int>() : seq_len;

    Tensor position_ids;
    if (input.count("position_ids")) {
      // Use existing position_ids for decode phase
      position_ids = input.at("position_ids");

      // For decode phase, increment the last position
      if (seq_len == 1) {
        auto last_pos = *position_ids.offsettedPtr<int64_t>({0, position_ids.shape()[1] - 1});
        position_ids = Tensor::empty({batch_size, 1}, kInt64, kCPU).alloc();
        *position_ids.offsettedPtr<int64_t>({0, 0}) = last_pos + 1;
      }
    } else {
      // Generate position_ids for prefill phase
      position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
      auto position_ids_ptr = position_ids.ptr<int64_t>();
      for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) { position_ids_ptr[b * seq_len + s] = s; }
      }
    }

    // Generate RoPE embeddings using the inv_freq buffer
    auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, model.getBuffer("inv_freq"), 1.0f);

    // Calculate the text embeddings
    auto input_embeddings = model.embedding_(sequence);

    const bool capture_layer_outputs = args.count("debug_layer_outputs") && args.at("debug_layer_outputs").get<bool>();
    model.setCaptureLayerOutputs(capture_layer_outputs);
    auto hidden_states = model(input_embeddings, llm_embedding_sin, llm_embedding_cos)[0];

    // clip hidden_states to one seq length
    {
      auto S = hidden_states.shape()[1];
      hidden_states = hidden_states[{kAll, {real_seq - 1}, kAll}];
    }

    Tensor logits;
    if (!tie_word_embeddings_) {
      logits = lm_head_(hidden_states);
    } else {
      // tie with embedding weights: [B,1,H] @ [V,H]^T -> [B,1,V]
      auto emb_w = model.embedding_.weight();  // shape [V, H]
      logits = nn::functional::matmul(hidden_states, emb_w, /*trans_a=*/false, /*trans_b=*/true);
    }

    ARGenerationOutputPast output{
        {"sequence", logits},
        {"position_ids", position_ids},
    };
    if (args.count("return_hidden_states") && args.at("return_hidden_states").get<bool>()) {
      output["hidden_states"] = hidden_states;
    }
    return output;
  }

  const std::vector<Tensor>& capturedLayerOutputs() const { return model.capturedLayerOutputs(); }
  size_t capturedBoundaryLayerCount() { return model.capturedBoundaryLayerCount(); }
  const std::vector<Tensor>& capturedLayerBoundaries(size_t layer_index) { return model.capturedLayerBoundaries(layer_index); }
  const std::vector<Tensor>& capturedLayerZeroBoundaries() { return model.capturedLayerZeroBoundaries(); }

  IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    // Things we need to return
    ir::IRContext::ptr_t llm_ir = nullptr;

    ir::lowlevel::traceStart();

    auto sequence = input.at("sequence");
    // Calculate the text embeddings
    auto input_embeddings = model.embedding_(sequence);

    ir::lowlevel::traceYield();
    // Trace. We don't want to trace things below and they really need to work.
    // -------------------------------------------------------------------

    // during trace, only need to give an empty position_ids
    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];
    auto position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU);
    auto llm_embedding_sin = Tensor::empty({batch_size, seq_len, model.getBuffer("inv_freq").shape()[0] * 2}, kFloat32, kCPU);
    auto llm_embedding_cos = Tensor::empty({batch_size, seq_len, model.getBuffer("inv_freq").shape()[0] * 2}, kFloat32, kCPU);

    // -------------------------------------------------------------------
    // Trace. We don't want to trace things above and they really need to work.
    ir::lowlevel::traceContinue();

    ir::lowlevel::traceComment("inputs to llm is: [input_embeddings, llm_embedding_sin, llm_embedding_cos]");

    auto hidden_states = ir::lowlevel::traceModule(model, input_embeddings, llm_embedding_sin, llm_embedding_cos)[0];

    // clip hidden_states to one seq length
    {
      auto S = hidden_states.shape()[1];
      hidden_states = hidden_states[{kAll, {S - 1}, kAll}];
    }

    Tensor logits;
    if (!tie_word_embeddings_) { logits = lm_head_(hidden_states); }

    llm_ir = ir::lowlevel::traceStop();
    return {{"model", llm_ir}};
  }

  QwenNPUConfig cfg;
  QwenText model;
  nn::Linear lm_head_;
  bool tie_word_embeddings_;
};

}  // namespace mllm::models::qwen_npu
