// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "mllm/mllm.hpp"
#include "mllm/models/qwen3_5/configuration_qwen3_5.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"

namespace mllm::models::qwen3_5 {

inline auto makeQwen3_5VisionPositionIds(const Tensor& grid_thw, int32_t spatial_merge_size) -> Tensor {
  if (grid_thw.dtype() != kInt32 || grid_thw.device() != kCPU || grid_thw.shape() != Tensor::shape_t({1, 3})
      || spatial_merge_size <= 0) {
    throw std::invalid_argument("Qwen3.5 vision position IDs require one int32 CPU image grid");
  }
  const auto* grid = grid_thw.ptr<int32_t>();
  const int32_t grid_t = grid[0];
  const int32_t grid_h = grid[1];
  const int32_t grid_w = grid[2];
  if (grid_t <= 0 || grid_h <= 0 || grid_w <= 0 || grid_h % spatial_merge_size != 0 || grid_w % spatial_merge_size != 0) {
    throw std::invalid_argument("Qwen3.5 vision grid is incompatible with spatial merge");
  }

  auto position_ids = Tensor::empty({grid_t * grid_h * grid_w, 2}, kInt32, kCPU).alloc();
  auto* output = position_ids.ptr<int32_t>();
  int64_t offset = 0;
  for (int32_t t = 0; t < grid_t; ++t) {
    (void)t;
    for (int32_t block_h = 0; block_h < grid_h / spatial_merge_size; ++block_h) {
      for (int32_t block_w = 0; block_w < grid_w / spatial_merge_size; ++block_w) {
        for (int32_t merge_h = 0; merge_h < spatial_merge_size; ++merge_h) {
          for (int32_t merge_w = 0; merge_w < spatial_merge_size; ++merge_w) {
            output[offset++] = block_h * spatial_merge_size + merge_h;
            output[offset++] = block_w * spatial_merge_size + merge_w;
          }
        }
      }
    }
  }
  return position_ids;
}

inline auto makeQwen3_5VisionRotaryEmbedding(const Tensor& position_ids, int32_t head_dim, float theta = 10000.0F)
    -> std::pair<Tensor, Tensor> {
  if (position_ids.dtype() != kInt32 || position_ids.device() != kCPU || position_ids.shape().size() != 2
      || position_ids.shape()[1] != 2 || head_dim <= 0 || head_dim % 4 != 0 || !std::isfinite(theta) || theta <= 0.0F) {
    throw std::invalid_argument("Qwen3.5 vision rotary embedding received invalid inputs");
  }
  const int32_t sequence = position_ids.shape()[0];
  const int32_t axis_dim = head_dim / 2;
  const int32_t inv_freq_dim = axis_dim / 2;
  auto sin = Tensor::empty({sequence, head_dim / 2}, kFloat32, kCPU).alloc();
  auto cos = Tensor::empty({sequence, head_dim / 2}, kFloat32, kCPU).alloc();
  const auto* positions = position_ids.ptr<int32_t>();
  auto* sin_ptr = sin.ptr<float>();
  auto* cos_ptr = cos.ptr<float>();

  for (int32_t s = 0; s < sequence; ++s) {
    for (int32_t axis = 0; axis < 2; ++axis) {
      for (int32_t d = 0; d < inv_freq_dim; ++d) {
        const float inv_freq = 1.0F / std::pow(theta, static_cast<float>(2 * d) / axis_dim);
        const float value = static_cast<float>(positions[s * 2 + axis]) * inv_freq;
        const int64_t offset = static_cast<int64_t>(s) * (head_dim / 2) + axis * inv_freq_dim + d;
        sin_ptr[offset] = std::sin(value);
        cos_ptr[offset] = std::cos(value);
      }
    }
  }
  return {sin, cos};
}

inline auto makeQwen3_5VisionBilinearPositionEmbedding(const Tensor& position_embedding_weight, const Tensor& grid_thw,
                                                       int32_t spatial_merge_size) -> Tensor {
  if (position_embedding_weight.dtype() != kFloat32 || position_embedding_weight.device() != kCPU
      || position_embedding_weight.shape().size() != 2 || grid_thw.dtype() != kInt32 || grid_thw.device() != kCPU
      || grid_thw.shape() != Tensor::shape_t({1, 3}) || spatial_merge_size <= 0) {
    throw std::invalid_argument("Qwen3.5 bilinear position embedding received invalid inputs");
  }
  const int32_t positions = position_embedding_weight.shape()[0];
  const int32_t hidden = position_embedding_weight.shape()[1];
  const int32_t side = static_cast<int32_t>(std::sqrt(positions));
  if (side * side != positions) { throw std::invalid_argument("Qwen3.5 learned vision position grid must be square"); }

  const auto* grid = grid_thw.ptr<int32_t>();
  const int32_t grid_t = grid[0];
  const int32_t grid_h = grid[1];
  const int32_t grid_w = grid[2];
  if (grid_t <= 0 || grid_h <= 0 || grid_w <= 0 || grid_h % spatial_merge_size != 0 || grid_w % spatial_merge_size != 0) {
    throw std::invalid_argument("Qwen3.5 bilinear position grid is incompatible with spatial merge");
  }

  auto output = Tensor::empty({grid_t * grid_h * grid_w, hidden}, kFloat32, kCPU).alloc();
  const auto* weight = position_embedding_weight.ptr<float>();
  auto* output_ptr = output.ptr<float>();
  int64_t output_row = 0;

  for (int32_t t = 0; t < grid_t; ++t) {
    (void)t;
    for (int32_t block_h = 0; block_h < grid_h / spatial_merge_size; ++block_h) {
      for (int32_t block_w = 0; block_w < grid_w / spatial_merge_size; ++block_w) {
        for (int32_t merge_h = 0; merge_h < spatial_merge_size; ++merge_h) {
          const int32_t h = block_h * spatial_merge_size + merge_h;
          const float h_coord = grid_h == 1 ? 0.0F : static_cast<float>(h) * (side - 1) / (grid_h - 1);
          const int32_t h_floor = static_cast<int32_t>(h_coord);
          const int32_t h_ceil = std::min(h_floor + 1, side - 1);
          const float h_frac = h_coord - h_floor;
          for (int32_t merge_w = 0; merge_w < spatial_merge_size; ++merge_w) {
            const int32_t w = block_w * spatial_merge_size + merge_w;
            const float w_coord = grid_w == 1 ? 0.0F : static_cast<float>(w) * (side - 1) / (grid_w - 1);
            const int32_t w_floor = static_cast<int32_t>(w_coord);
            const int32_t w_ceil = std::min(w_floor + 1, side - 1);
            const float w_frac = w_coord - w_floor;
            const int32_t indices[4] = {
                h_floor * side + w_floor,
                h_floor * side + w_ceil,
                h_ceil * side + w_floor,
                h_ceil * side + w_ceil,
            };
            const float weights[4] = {
                (1.0F - h_frac) * (1.0F - w_frac),
                (1.0F - h_frac) * w_frac,
                h_frac * (1.0F - w_frac),
                h_frac * w_frac,
            };
            for (int32_t d = 0; d < hidden; ++d) {
              float value = 0.0F;
              for (int32_t corner = 0; corner < 4; ++corner) {
                value += weights[corner] * weight[static_cast<int64_t>(indices[corner]) * hidden + d];
              }
              output_ptr[output_row * hidden + d] = value;
            }
            ++output_row;
          }
        }
      }
    }
  }
  return output;
}

inline auto qwen3_5ExactGelu(const Tensor& input) -> Tensor {
  if (input.dtype() != kFloat32 || input.device() != kCPU || !input.isContiguous()) {
    throw std::invalid_argument("Qwen3.5 exact vision merger GELU expects contiguous float32 CPU activations");
  }
  auto output = Tensor::empty(input.shape(), kFloat32, kCPU).alloc();
  const auto* input_ptr = input.ptr<float>();
  auto* output_ptr = output.ptr<float>();
  constexpr float kInvSqrtTwo = 0.70710678118654752440F;
  for (int64_t i = 0; i < input.numel(); ++i) {
    output_ptr[i] = 0.5F * input_ptr[i] * (1.0F + std::erf(input_ptr[i] * kInvSqrtTwo));
  }
  return output;
}

class Qwen3_5VisionPatchEmbed final : public nn::Module {
 public:
  Qwen3_5VisionPatchEmbed() = default;
  Qwen3_5VisionPatchEmbed(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    in_channels_ = cfg.vision_in_channels;
    hidden_size_ = cfg.vision_hidden_size;
    patch_size_ = cfg.vision_patch_size;
    temporal_patch_size_ = cfg.vision_temporal_patch_size;
    proj_ = reg<nn::Conv3D>("proj", in_channels_, hidden_size_,
                            std::vector<int32_t>{temporal_patch_size_, patch_size_, patch_size_},
                            std::vector<int32_t>{temporal_patch_size_, patch_size_, patch_size_}, true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    hidden_states = hidden_states.view({-1, in_channels_, temporal_patch_size_, patch_size_, patch_size_});
    return {proj_(hidden_states).view({-1, hidden_size_})};
  }

 private:
  int32_t in_channels_ = 3;
  int32_t hidden_size_ = 768;
  int32_t patch_size_ = 16;
  int32_t temporal_patch_size_ = 2;
  nn::Conv3D proj_;
};

class Qwen3_5VisionMLP final : public nn::Module {
 public:
  Qwen3_5VisionMLP() = default;
  Qwen3_5VisionMLP(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    linear_fc1_ =
        reg<nn::Linear>("linear_fc1", cfg.vision_hidden_size, cfg.vision_intermediate_size, true, cfg.linear_impl_type);
    act_ = reg<nn::GELU>("act");
    linear_fc2_ =
        reg<nn::Linear>("linear_fc2", cfg.vision_intermediate_size, cfg.vision_hidden_size, true, cfg.linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return {linear_fc2_(act_(linear_fc1_(inputs[0])))};
  }

 private:
  nn::Linear linear_fc1_;
  nn::GELU act_;
  nn::Linear linear_fc2_;
};

class Qwen3_5VisionAttention final : public nn::Module {
 public:
  Qwen3_5VisionAttention() = default;
  Qwen3_5VisionAttention(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.vision_hidden_size;
    num_heads_ = cfg.vision_num_heads;
    head_dim_ = hidden_size_ / num_heads_;
    qkv_ = reg<nn::Linear>("qkv", hidden_size_, hidden_size_ * 3, true, cfg.linear_impl_type);
    proj_ = reg<nn::Linear>("proj", hidden_size_, hidden_size_, true, cfg.linear_impl_type);
    softmax_ = reg<nn::Softmax>("softmax", -1);
    rope_q_ = reg<nn::VisionRoPE>("rope_q", aops::VisionRoPEOpOptionsType::kQwen2VL,
                                  aops::Qwen2VLRoPEOpOptions{
                                      .dims = head_dim_,
                                      .spatial_merge_size = cfg.vision_spatial_merge_size,
                                      .theta = 10000.0F,
                                  });
    rope_k_ = reg<nn::VisionRoPE>("rope_k", aops::VisionRoPEOpOptionsType::kQwen2VL,
                                  aops::Qwen2VLRoPEOpOptions{
                                      .dims = head_dim_,
                                      .spatial_merge_size = cfg.vision_spatial_merge_size,
                                      .theta = 10000.0F,
                                  });
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto sin = inputs[1];
    auto cos = inputs[2];
    const int32_t sequence = hidden_states.shape()[0];
    auto [query, key, value] =
        nn::functional::split<3>(qkv_(hidden_states).view({sequence, 3, num_heads_, head_dim_}).permute({1, 0, 2, 3}), 1, 0);
    query = rope_q_(query, sin, cos);
    key = rope_k_(key, sin, cos);
    query = query.transpose(1, 2);
    key = key.transpose(1, 2);
    value = value.transpose(1, 2);
    auto attention = nn::functional::matmul(query, key, false, true) * (1.0F / std::sqrt(head_dim_));
    attention = softmax_(attention);
    auto output = nn::functional::matmul(attention, value).transpose(1, 2).view({sequence, hidden_size_});
    return {proj_(output)};
  }

 private:
  int32_t hidden_size_ = 768;
  int32_t num_heads_ = 12;
  int32_t head_dim_ = 64;
  nn::Linear qkv_;
  nn::Linear proj_;
  nn::Softmax softmax_;
  nn::VisionRoPE rope_q_;
  nn::VisionRoPE rope_k_;
};

class Qwen3_5VisionBlock final : public nn::Module {
 public:
  Qwen3_5VisionBlock() = default;
  Qwen3_5VisionBlock(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    norm1_ = reg<nn::LayerNorm>("norm1", std::vector<int32_t>{cfg.vision_hidden_size}, true, true, 1.0e-6F);
    norm2_ = reg<nn::LayerNorm>("norm2", std::vector<int32_t>{cfg.vision_hidden_size}, true, true, 1.0e-6F);
    attention_ = reg<Qwen3_5VisionAttention>("attn", cfg);
    mlp_ = reg<Qwen3_5VisionMLP>("mlp", cfg);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    hidden_states = hidden_states + attention_(norm1_(hidden_states), inputs[1], inputs[2])[0];
    hidden_states = hidden_states + mlp_(norm2_(hidden_states))[0];
    return {hidden_states};
  }

 private:
  nn::LayerNorm norm1_;
  nn::LayerNorm norm2_;
  Qwen3_5VisionAttention attention_;
  Qwen3_5VisionMLP mlp_;
};

class Qwen3_5VisionPatchMerger final : public nn::Module {
 public:
  Qwen3_5VisionPatchMerger() = default;
  Qwen3_5VisionPatchMerger(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    context_size_ = cfg.vision_hidden_size;
    hidden_size_ = context_size_ * cfg.vision_spatial_merge_size * cfg.vision_spatial_merge_size;
    norm_ = reg<nn::LayerNorm>("norm", std::vector<int32_t>{context_size_}, true, true, 1.0e-6F);
    linear_fc1_ = reg<nn::Linear>("linear_fc1", hidden_size_, hidden_size_, true, cfg.linear_impl_type);
    linear_fc2_ = reg<nn::Linear>("linear_fc2", hidden_size_, cfg.vision_out_hidden_size, true, cfg.linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = norm_(inputs[0]).view({-1, hidden_size_});
    hidden_states = qwen3_5ExactGelu(linear_fc1_(hidden_states));
    return {linear_fc2_(hidden_states)};
  }

 private:
  int32_t context_size_ = 768;
  int32_t hidden_size_ = 3072;
  nn::LayerNorm norm_;
  nn::Linear linear_fc1_;
  nn::Linear linear_fc2_;
};

class Qwen3_5VisionModel final : public nn::Module {
 public:
  Qwen3_5VisionModel() = default;
  Qwen3_5VisionModel(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.vision_hidden_size;
    num_heads_ = cfg.vision_num_heads;
    spatial_merge_size_ = cfg.vision_spatial_merge_size;
    patch_embed_ = reg<Qwen3_5VisionPatchEmbed>("patch_embed", cfg);
    pos_embed_ = reg<nn::Embedding>("pos_embed", cfg.vision_num_position_embeddings, hidden_size_);
    blocks_ = reg<nn::ModuleList<Qwen3_5VisionBlock>>("blocks", cfg.vision_depth, cfg);
    merger_ = reg<Qwen3_5VisionPatchMerger>("merger", cfg);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    const auto& pixel_values = inputs[0];
    const auto& grid_thw = inputs[1];
    auto hidden_states = patch_embed_(pixel_values)[0];
    auto position_embedding = makeQwen3_5VisionBilinearPositionEmbedding(pos_embed_.weight(), grid_thw, spatial_merge_size_);
    if (hidden_states.shape() != position_embedding.shape()) {
      throw std::invalid_argument("Qwen3.5 patch embeddings do not match the image grid");
    }
    hidden_states = hidden_states + position_embedding;

    auto position_ids = makeQwen3_5VisionPositionIds(grid_thw, spatial_merge_size_);
    auto [sin, cos] = makeQwen3_5VisionRotaryEmbedding(position_ids, hidden_size_ / num_heads_);
    for (auto& block : blocks_.list()) { hidden_states = block(hidden_states, sin, cos)[0]; }
    return {merger_(hidden_states)[0]};
  }

 private:
  int32_t hidden_size_ = 768;
  int32_t num_heads_ = 12;
  int32_t spatial_merge_size_ = 2;
  Qwen3_5VisionPatchEmbed patch_embed_;
  nn::Embedding pos_embed_;
  nn::ModuleList<Qwen3_5VisionBlock> blocks_;
  Qwen3_5VisionPatchMerger merger_;
};

}  // namespace mllm::models::qwen3_5
