// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mllm/core/Tensor.hpp"

namespace mllm::models::qwen3_5 {

inline auto expandQwen3_5SingleImagePlaceholders(const std::vector<int64_t>& token_ids, int64_t image_token_id,
                                                 int32_t image_token_count)
    -> std::pair<std::vector<int64_t>, std::vector<int32_t>> {
  if (image_token_count <= 0) { throw std::invalid_argument("Qwen3.5 image token count must be positive"); }
  const auto first = std::find(token_ids.begin(), token_ids.end(), image_token_id);
  if (first == token_ids.end() || std::find(first + 1, token_ids.end(), image_token_id) != token_ids.end()) {
    throw std::invalid_argument("Qwen3.5 single-image template must contain exactly one image placeholder");
  }

  const auto image_offset = static_cast<size_t>(std::distance(token_ids.begin(), first));
  std::vector<int64_t> expanded;
  expanded.reserve(token_ids.size() + image_token_count - 1);
  expanded.insert(expanded.end(), token_ids.begin(), first);
  expanded.insert(expanded.end(), image_token_count, image_token_id);
  expanded.insert(expanded.end(), first + 1, token_ids.end());

  std::vector<int32_t> token_types(expanded.size(), 0);
  std::fill(token_types.begin() + static_cast<std::ptrdiff_t>(image_offset),
            token_types.begin() + static_cast<std::ptrdiff_t>(image_offset + image_token_count), 1);
  return {expanded, token_types};
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

inline auto makeQwen3_5SingleImagePositionIds(const Tensor& token_type_ids, const Tensor& image_grid_thw,
                                              int32_t spatial_merge_size) -> Tensor {
  const auto& type_shape = token_type_ids.shape();
  const auto& grid_shape = image_grid_thw.shape();
  if (token_type_ids.dtype() != kInt32 || token_type_ids.device() != kCPU || type_shape.size() != 2 || type_shape[0] != 1
      || type_shape[1] <= 0) {
    throw std::invalid_argument("Qwen3.5 token_type_ids must be non-empty rank-2 int32 CPU with batch size 1");
  }
  if (image_grid_thw.dtype() != kInt32 || image_grid_thw.device() != kCPU || grid_shape.size() != 2 || grid_shape[0] != 1
      || grid_shape[1] != 3 || spatial_merge_size <= 0) {
    throw std::invalid_argument("Qwen3.5 image_grid_thw must contain exactly one valid image grid");
  }

  const auto* grid = image_grid_thw.ptr<int32_t>();
  const int32_t grid_t = grid[0];
  const int32_t grid_h = grid[1];
  const int32_t grid_w = grid[2];
  if (grid_t <= 0 || grid_h <= 0 || grid_w <= 0 || grid_h % spatial_merge_size != 0 || grid_w % spatial_merge_size != 0) {
    throw std::invalid_argument("Qwen3.5 image grid is incompatible with the spatial merge size");
  }
  const int32_t llm_t = grid_t;
  const int32_t llm_h = grid_h / spatial_merge_size;
  const int32_t llm_w = grid_w / spatial_merge_size;
  const int32_t expected_image_tokens = llm_t * llm_h * llm_w;

  const int32_t sequence = type_shape[1];
  const auto* token_types = token_type_ids.ptr<int32_t>();
  int32_t image_begin = -1;
  int32_t image_end = -1;
  bool left_image_span = false;
  for (int32_t s = 0; s < sequence; ++s) {
    if (token_types[s] != 0 && token_types[s] != 1) {
      throw std::invalid_argument("Qwen3.5 single-image support only accepts text and image token types");
    }
    if (token_types[s] == 1) {
      if (left_image_span) { throw std::invalid_argument("Qwen3.5 supports exactly one contiguous image token span"); }
      if (image_begin < 0) image_begin = s;
      image_end = s + 1;
    } else if (image_begin >= 0) {
      left_image_span = true;
    }
  }
  if (image_begin < 0 || image_end - image_begin != expected_image_tokens) {
    throw std::invalid_argument("Qwen3.5 image features and image placeholders do not match");
  }

  auto position_ids = Tensor::empty({3, 1, sequence}, kInt64, kCPU).alloc();
  auto* positions = position_ids.ptr<int64_t>();
  int64_t current_position = 0;

  for (int32_t s = 0; s < image_begin; ++s) {
    for (int32_t axis = 0; axis < 3; ++axis) { positions[axis * sequence + s] = current_position; }
    ++current_position;
  }

  int32_t image_offset = 0;
  for (int32_t t = 0; t < llm_t; ++t) {
    for (int32_t h = 0; h < llm_h; ++h) {
      for (int32_t w = 0; w < llm_w; ++w) {
        const int32_t sequence_offset = image_begin + image_offset++;
        positions[sequence_offset] = current_position + t;
        positions[sequence + sequence_offset] = current_position + h;
        positions[2 * sequence + sequence_offset] = current_position + w;
      }
    }
  }
  current_position += std::max(grid_h, grid_w) / spatial_merge_size;

  for (int32_t s = image_end; s < sequence; ++s) {
    for (int32_t axis = 0; axis < 3; ++axis) { positions[axis * sequence + s] = current_position; }
    ++current_position;
  }
  return position_ids;
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
