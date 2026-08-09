// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "mllm/core/Tensor.hpp"
#include "mllm/preprocessor/visual/Image.hpp"

namespace mllm::models::qwen3_5 {

class Qwen3_5ImagePreprocessor {
 public:
  explicit Qwen3_5ImagePreprocessor(int32_t min_pixels = 256 * 256, int32_t max_pixels = 512 * 512, int32_t patch_size = 16,
                                    int32_t temporal_patch_size = 2, int32_t merge_size = 2)
      : min_pixels_(min_pixels),
        max_pixels_(max_pixels),
        patch_size_(patch_size),
        temporal_patch_size_(temporal_patch_size),
        merge_size_(merge_size) {
    if (min_pixels_ <= 0 || max_pixels_ < min_pixels_ || patch_size_ <= 0 || temporal_patch_size_ <= 0 || merge_size_ <= 0) {
      throw std::invalid_argument("Qwen3.5 image preprocessor received invalid geometry");
    }
  }

  [[nodiscard]] std::pair<int32_t, int32_t> smartResize(int32_t height, int32_t width) const {
    if (height <= 0 || width <= 0) { throw std::invalid_argument("Qwen3.5 image dimensions must be positive"); }
    if (static_cast<double>(std::max(height, width)) / std::min(height, width) > 200.0) {
      throw std::invalid_argument("Qwen3.5 image aspect ratio must not exceed 200");
    }

    const int32_t factor = patch_size_ * merge_size_;
    const auto python_round_to_factor = [factor](int32_t value) {
      const int32_t quotient = value / factor;
      const int32_t remainder = value % factor;
      if (remainder * 2 < factor) return quotient * factor;
      if (remainder * 2 > factor) return (quotient + 1) * factor;
      return (quotient % 2 == 0 ? quotient : quotient + 1) * factor;
    };
    int32_t resized_height = python_round_to_factor(height);
    int32_t resized_width = python_round_to_factor(width);
    const int64_t rounded_pixels = static_cast<int64_t>(resized_height) * resized_width;

    if (rounded_pixels > max_pixels_) {
      const double beta = std::sqrt(static_cast<double>(height) * width / max_pixels_);
      resized_height = std::max(factor, static_cast<int32_t>(std::floor(height / beta / factor)) * factor);
      resized_width = std::max(factor, static_cast<int32_t>(std::floor(width / beta / factor)) * factor);
    } else if (rounded_pixels < min_pixels_) {
      const double beta = std::sqrt(static_cast<double>(min_pixels_) / (static_cast<double>(height) * width));
      resized_height = static_cast<int32_t>(std::ceil(height * beta / factor)) * factor;
      resized_width = static_cast<int32_t>(std::ceil(width * beta / factor)) * factor;
    }

    return {resized_height, resized_width};
  }

  std::pair<Tensor, Tensor> operator()(const std::string& image_path) const {
    if (image_path.empty()) { throw std::invalid_argument("Qwen3.5 image path must not be empty"); }
    auto image = Image::open(image_path);
    auto [height, width] = smartResize(image.h(), image.w());
    image = image.resize(width, height, "bicubic");
    return flattenNormalizedPatches(image.tensor());
  }

  // Exposed for focused tests and reference-oracle comparisons. Input is an
  // already-resized float32 RGB tensor in [H,W,3] with values in [0,255].
  [[nodiscard]] std::pair<Tensor, Tensor> flattenNormalizedPatches(const Tensor& image_hwc) const {
    const auto& shape = image_hwc.shape();
    if (image_hwc.dtype() != kFloat32 || image_hwc.device() != kCPU || shape.size() != 3 || shape[2] != 3) {
      throw std::invalid_argument("Qwen3.5 preprocessor expects a float32 CPU RGB tensor in HWC layout");
    }
    const int32_t height = shape[0];
    const int32_t width = shape[1];
    const int32_t factor = patch_size_ * merge_size_;
    if (height <= 0 || width <= 0 || height % factor != 0 || width % factor != 0) {
      throw std::invalid_argument("Qwen3.5 resized image dimensions must be divisible by patch_size * merge_size");
    }

    const int32_t grid_h = height / patch_size_;
    const int32_t grid_w = width / patch_size_;
    const int32_t patch_features = 3 * temporal_patch_size_ * patch_size_ * patch_size_;
    auto patches = Tensor::empty({grid_h * grid_w, patch_features}, kFloat32, kCPU).alloc();
    const auto* input = image_hwc.ptr<float>();
    auto* output = patches.ptr<float>();

    int64_t output_offset = 0;
    for (int32_t block_h = 0; block_h < grid_h / merge_size_; ++block_h) {
      for (int32_t block_w = 0; block_w < grid_w / merge_size_; ++block_w) {
        for (int32_t merge_h = 0; merge_h < merge_size_; ++merge_h) {
          for (int32_t merge_w = 0; merge_w < merge_size_; ++merge_w) {
            for (int32_t channel = 0; channel < 3; ++channel) {
              for (int32_t temporal = 0; temporal < temporal_patch_size_; ++temporal) {
                (void)temporal;
                for (int32_t patch_h = 0; patch_h < patch_size_; ++patch_h) {
                  const int32_t source_h = (block_h * merge_size_ + merge_h) * patch_size_ + patch_h;
                  for (int32_t patch_w = 0; patch_w < patch_size_; ++patch_w) {
                    const int32_t source_w = (block_w * merge_size_ + merge_w) * patch_size_ + patch_w;
                    const int64_t input_offset = (static_cast<int64_t>(source_h) * width + source_w) * 3 + channel;
                    // mean=std=0.5 after rescaling by 1/255.
                    output[output_offset++] = input[input_offset] * (2.0F / 255.0F) - 1.0F;
                  }
                }
              }
            }
          }
        }
      }
    }

    auto grid_thw = Tensor::empty({1, 3}, kInt32, kCPU).alloc();
    auto* grid = grid_thw.ptr<int32_t>();
    grid[0] = 1;
    grid[1] = grid_h;
    grid[2] = grid_w;
    return {patches, grid_thw};
  }

 private:
  int32_t min_pixels_;
  int32_t max_pixels_;
  int32_t patch_size_;
  int32_t temporal_patch_size_;
  int32_t merge_size_;
};

}  // namespace mllm::models::qwen3_5
