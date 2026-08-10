// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mllm/core/Tensor.hpp"

namespace mllm::models::qwen3_5 {

inline auto expandQwen3_5ImagePlaceholders(const std::vector<int64_t>& token_ids, int64_t image_token_id,
                                           const std::vector<int32_t>& image_token_counts)
    -> std::pair<std::vector<int64_t>, std::vector<int32_t>> {
  if (image_token_counts.empty()
      || std::any_of(image_token_counts.begin(), image_token_counts.end(), [](int32_t count) { return count <= 0; })) {
    throw std::invalid_argument("Qwen3.5 image token counts must be non-empty and positive");
  }
  std::vector<int64_t> expanded;
  std::vector<int32_t> token_types;
  expanded.reserve(token_ids.size() + std::accumulate(image_token_counts.begin(), image_token_counts.end(), 0)
                   - image_token_counts.size());
  token_types.reserve(expanded.capacity());
  size_t image_index = 0;
  for (const auto token_id : token_ids) {
    if (token_id != image_token_id) {
      expanded.push_back(token_id);
      token_types.push_back(0);
      continue;
    }
    if (image_index >= image_token_counts.size()) {
      throw std::invalid_argument("Qwen3.5 template contains more image placeholders than images");
    }
    expanded.insert(expanded.end(), image_token_counts[image_index], image_token_id);
    token_types.insert(token_types.end(), image_token_counts[image_index], 1);
    ++image_index;
  }
  if (image_index != image_token_counts.size()) {
    throw std::invalid_argument("Qwen3.5 template contains fewer image placeholders than images");
  }
  return {expanded, token_types};
}

inline auto expandQwen3_5SingleImagePlaceholders(const std::vector<int64_t>& token_ids, int64_t image_token_id,
                                                 int32_t image_token_count)
    -> std::pair<std::vector<int64_t>, std::vector<int32_t>> {
  return expandQwen3_5ImagePlaceholders(token_ids, image_token_id, {image_token_count});
}

inline void injectQwen3_5ImageEmbeddings(Tensor& input_embeddings, const Tensor& image_embeddings,
                                         const Tensor& token_type_ids) {
  const auto& input_shape = input_embeddings.shape();
  const auto& image_shape = image_embeddings.shape();
  if (input_embeddings.dtype() != kFloat32 || input_embeddings.device() != kCPU || !input_embeddings.isContiguous()
      || input_shape.size() != 3 || input_shape[0] != 1 || input_shape[1] <= 0 || input_shape[2] <= 0
      || image_embeddings.dtype() != kFloat32 || image_embeddings.device() != kCPU || !image_embeddings.isContiguous()
      || image_shape.size() != 2 || image_shape[0] <= 0 || image_shape[1] != input_shape[2] || token_type_ids.dtype() != kInt32
      || token_type_ids.device() != kCPU || token_type_ids.shape() != Tensor::shape_t({1, input_shape[1]})) {
    throw std::invalid_argument("Qwen3.5 image embedding injection received invalid inputs");
  }

  const auto* types = token_type_ids.ptr<int32_t>();
  const auto* image_values = image_embeddings.ptr<float>();
  auto* input_values = input_embeddings.ptr<float>();
  const int32_t sequence = input_shape[1];
  const int32_t hidden = input_shape[2];
  int32_t feature_offset = 0;
  for (int32_t position = 0; position < sequence; ++position) {
    if (types[position] != 1) continue;
    if (feature_offset >= image_shape[0]) {
      throw std::invalid_argument("Qwen3.5 image placeholders outnumber image embeddings");
    }
    std::copy_n(image_values + static_cast<int64_t>(feature_offset) * hidden, hidden,
                input_values + static_cast<int64_t>(position) * hidden);
    ++feature_offset;
  }
  if (feature_offset != image_shape[0]) {
    throw std::invalid_argument("Qwen3.5 image embeddings outnumber image placeholders");
  }
}

inline auto makeQwen3_5InterleavedRotaryEmbedding(const Tensor& position_ids, const Tensor& inv_freq,
                                                  const std::vector<int32_t>& mrope_section, float attention_scaling = 1.0F)
    -> std::pair<Tensor, Tensor> {
  const auto& position_shape = position_ids.shape();
  if (position_ids.dtype() != kInt64 || position_ids.device() != kCPU
      || (position_shape.size() != 2 && position_shape.size() != 3)) {
    throw std::invalid_argument("Qwen3.5 position_ids must be rank-2 or rank-3 int64 CPU");
  }
  if (inv_freq.dtype() != kFloat32 || inv_freq.device() != kCPU || inv_freq.shape().size() != 1) {
    throw std::invalid_argument("Qwen3.5 inv_freq must be rank-1 float32 CPU");
  }
  if (mrope_section.size() != 3 || mrope_section[0] + mrope_section[1] + mrope_section[2] != inv_freq.shape()[0]) {
    throw std::invalid_argument("Qwen3.5 MRoPE sections must cover every inverse frequency");
  }

  const int32_t batch = position_shape.size() == 2 ? position_shape[0] : position_shape[1];
  const int32_t sequence = position_shape.size() == 2 ? position_shape[1] : position_shape[2];
  const int32_t half_dim = inv_freq.shape()[0];
  const int32_t dim = half_dim * 2;
  if (batch <= 0 || sequence <= 0) { throw std::invalid_argument("Qwen3.5 position_ids must not be empty"); }

  auto sin = Tensor::empty({batch, sequence, dim}, kFloat32, kCPU).alloc();
  auto cos = Tensor::empty({batch, sequence, dim}, kFloat32, kCPU).alloc();
  const auto* positions = position_ids.ptr<int64_t>();
  const auto* frequencies = inv_freq.ptr<float>();
  auto* sin_ptr = sin.ptr<float>();
  auto* cos_ptr = cos.ptr<float>();

  for (int32_t b = 0; b < batch; ++b) {
    for (int32_t s = 0; s < sequence; ++s) {
      for (int32_t d = 0; d < half_dim; ++d) {
        int32_t axis = 0;
        if (d % 3 == 1 && d < mrope_section[1] * 3) {
          axis = 1;
        } else if (d % 3 == 2 && d < mrope_section[2] * 3) {
          axis = 2;
        }
        const int64_t position =
            position_shape.size() == 2 ? positions[b * sequence + s] : positions[(axis * batch + b) * sequence + s];
        const float value = static_cast<float>(position) * frequencies[d];
        const float sin_value = std::sin(value) * attention_scaling;
        const float cos_value = std::cos(value) * attention_scaling;
        const int64_t output_offset = (static_cast<int64_t>(b) * sequence + s) * dim + d;
        sin_ptr[output_offset] = sin_value;
        sin_ptr[output_offset + half_dim] = sin_value;
        cos_ptr[output_offset] = cos_value;
        cos_ptr[output_offset + half_dim] = cos_value;
      }
    }
  }
  return {sin, cos};
}

inline auto makeQwen3_5ImagePositionIds(const Tensor& token_type_ids, const Tensor& image_grid_thw, int32_t spatial_merge_size)
    -> Tensor {
  const auto& type_shape = token_type_ids.shape();
  const auto& grid_shape = image_grid_thw.shape();
  if (token_type_ids.dtype() != kInt32 || token_type_ids.device() != kCPU || type_shape.size() != 2 || type_shape[0] != 1
      || type_shape[1] <= 0) {
    throw std::invalid_argument("Qwen3.5 token_type_ids must be non-empty rank-2 int32 CPU with batch size 1");
  }
  if (image_grid_thw.dtype() != kInt32 || image_grid_thw.device() != kCPU || grid_shape.size() != 2 || grid_shape[0] <= 0
      || grid_shape[1] != 3 || spatial_merge_size <= 0) {
    throw std::invalid_argument("Qwen3.5 image_grid_thw must contain one valid row per image");
  }
  const int32_t sequence = type_shape[1];
  const auto* token_types = token_type_ids.ptr<int32_t>();
  const auto* grids = image_grid_thw.ptr<int32_t>();
  auto position_ids = Tensor::empty({3, 1, sequence}, kInt64, kCPU).alloc();
  auto* positions = position_ids.ptr<int64_t>();
  int64_t current_position = 0;
  int32_t image_index = 0;
  int32_t s = 0;
  while (s < sequence) {
    const int32_t token_type = token_types[s];
    if (token_type != 0 && token_type != 1) {
      throw std::invalid_argument("Qwen3.5 image support only accepts text and image token types");
    }
    int32_t end = s + 1;
    while (end < sequence && token_types[end] == token_type) ++end;
    if (token_type == 0) {
      for (; s < end; ++s) {
        for (int32_t axis = 0; axis < 3; ++axis) { positions[axis * sequence + s] = current_position; }
        ++current_position;
      }
      continue;
    }
    if (image_index >= grid_shape[0]) { throw std::invalid_argument("Qwen3.5 image token spans outnumber image grids"); }
    const auto* grid = grids + image_index * 3;
    const int32_t grid_t = grid[0];
    const int32_t grid_h = grid[1];
    const int32_t grid_w = grid[2];
    if (grid_t <= 0 || grid_h <= 0 || grid_w <= 0 || grid_h % spatial_merge_size != 0 || grid_w % spatial_merge_size != 0) {
      throw std::invalid_argument("Qwen3.5 image grid is incompatible with the spatial merge size");
    }
    const int32_t llm_h = grid_h / spatial_merge_size;
    const int32_t llm_w = grid_w / spatial_merge_size;
    if (end - s != grid_t * llm_h * llm_w) {
      throw std::invalid_argument("Qwen3.5 image features and image placeholders do not match");
    }
    int32_t image_offset = 0;
    for (int32_t t = 0; t < grid_t; ++t) {
      for (int32_t h = 0; h < llm_h; ++h) {
        for (int32_t w = 0; w < llm_w; ++w) {
          const int32_t sequence_offset = s + image_offset++;
          positions[sequence_offset] = current_position + t;
          positions[sequence + sequence_offset] = current_position + h;
          positions[2 * sequence + sequence_offset] = current_position + w;
        }
      }
    }
    current_position += std::max(grid_h, grid_w) / spatial_merge_size;
    s = end;
    ++image_index;
  }
  if (image_index != grid_shape[0]) { throw std::invalid_argument("Qwen3.5 image grids outnumber image token spans"); }
  return position_ids;
}

inline auto makeQwen3_5SingleImagePositionIds(const Tensor& token_type_ids, const Tensor& image_grid_thw,
                                              int32_t spatial_merge_size) -> Tensor {
  if (image_grid_thw.shape().size() != 2 || image_grid_thw.shape()[0] != 1) {
    throw std::invalid_argument("Qwen3.5 single-image positions require exactly one image grid");
  }
  return makeQwen3_5ImagePositionIds(token_type_ids, image_grid_thw, spatial_merge_size);
}

inline auto advanceQwen3_5PositionIds(const Tensor& previous_position_ids) -> Tensor {
  const auto& shape = previous_position_ids.shape();
  if (previous_position_ids.dtype() != kInt64 || previous_position_ids.device() != kCPU
      || (shape.size() != 2 && shape.size() != 3)) {
    throw std::invalid_argument("Qwen3.5 cached position_ids have an unsupported layout");
  }

  if (shape.size() == 2) {
    if (shape[0] != 1 || shape[1] <= 0) { throw std::invalid_argument("Qwen3.5 cached text positions are invalid"); }
    auto next = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
    next.ptr<int64_t>()[0] = previous_position_ids.ptr<int64_t>()[shape[1] - 1] + 1;
    return next;
  }

  if (shape[0] != 3 || shape[1] != 1 || shape[2] <= 0) {
    throw std::invalid_argument("Qwen3.5 cached multimodal positions must have shape [3,1,S]");
  }
  auto next = Tensor::empty({3, 1, 1}, kInt64, kCPU).alloc();
  const auto* previous = previous_position_ids.ptr<int64_t>();
  auto* output = next.ptr<int64_t>();
  for (int32_t axis = 0; axis < 3; ++axis) { output[axis] = previous[axis * shape[2] + shape[2] - 1] + 1; }
  return next;
}

}  // namespace mllm::models::qwen3_5
