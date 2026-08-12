// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mllm/core/Tensor.hpp"
#include "mllm/preprocessor/visual/Image.hpp"

namespace mllm::models::qwen3_5 {

inline void convertQwen3_5I420ToRgb(const uint8_t* y_plane, int32_t y_stride, const uint8_t* u_plane, int32_t u_stride,
                                    const uint8_t* v_plane, int32_t v_stride, int32_t width, int32_t height, float* rgb) {
  if (y_plane == nullptr || u_plane == nullptr || v_plane == nullptr || rgb == nullptr || width <= 0 || height <= 0
      || width % 2 != 0 || height % 2 != 0 || y_stride < width || u_stride < width / 2 || v_stride < width / 2) {
    throw std::invalid_argument("Qwen3.5 I420 conversion requires even dimensions and valid plane strides");
  }

  const auto clip = [](int32_t value) { return std::clamp(value, 0, 255); };
  for (int32_t row = 0; row < height; ++row) {
    for (int32_t column = 0; column < width; ++column) {
      const int32_t y = static_cast<int32_t>(y_plane[static_cast<int64_t>(row) * y_stride + column]) - 16;
      const int32_t u = static_cast<int32_t>(u_plane[static_cast<int64_t>(row / 2) * u_stride + column / 2]) - 128;
      const int32_t v = static_cast<int32_t>(v_plane[static_cast<int64_t>(row / 2) * v_stride + column / 2]) - 128;
      const int64_t output = (static_cast<int64_t>(row) * width + column) * 3;
      rgb[output] = static_cast<float>(clip((298 * y + 409 * v + 128) >> 8));
      rgb[output + 1] = static_cast<float>(clip((298 * y - 100 * u - 208 * v + 128) >> 8));
      rgb[output + 2] = static_cast<float>(clip((298 * y + 516 * u + 128) >> 8));
    }
  }
}

struct Qwen3_5BicubicSpan {
  int32_t first = 0;
  std::vector<double> weights;
};

inline auto makeQwen3_5BicubicSpans(int32_t input_size, int32_t output_size) -> std::vector<Qwen3_5BicubicSpan> {
  if (input_size <= 0 || output_size <= 0) throw std::invalid_argument("Qwen3.5 resize dimensions must be positive");
  const double scale = static_cast<double>(input_size) / output_size;
  const double support = 2.0 * std::max(scale, 1.0);
  const double inverse_filter_scale = scale >= 1.0 ? 1.0 / scale : 1.0;
  const auto cubic = [](double value) {
    constexpr double coefficient = -0.5;
    value = std::abs(value);
    if (value < 1.0) return ((coefficient + 2.0) * value - (coefficient + 3.0)) * value * value + 1.0;
    if (value < 2.0) return (((value - 5.0) * value + 8.0) * value - 4.0) * coefficient;
    return 0.0;
  };

  std::vector<Qwen3_5BicubicSpan> spans(output_size);
  for (int32_t output = 0; output < output_size; ++output) {
    const double center = scale * (output + 0.5);
    const int32_t first = std::max(static_cast<int32_t>(center - support + 0.5), 0);
    const int32_t end = std::min(static_cast<int32_t>(center + support + 0.5), input_size);
    if (end <= first) throw std::runtime_error("Qwen3.5 bicubic resize produced an empty support span");
    spans[output].first = first;
    spans[output].weights.resize(end - first);
    double total = 0.0;
    for (int32_t index = first; index < end; ++index) {
      const double weight = cubic((index + 0.5 - center) * inverse_filter_scale);
      spans[output].weights[index - first] = weight;
      total += weight;
    }
    if (total == 0.0) throw std::runtime_error("Qwen3.5 bicubic resize produced zero total weight");
    for (double& weight : spans[output].weights) weight /= total;
  }
  return spans;
}

inline auto resizeQwen3_5RgbLikeTorchvision(const float* input, int32_t input_height, int32_t input_width,
                                            int32_t output_height, int32_t output_width) -> std::vector<uint8_t> {
  if (input == nullptr || input_height <= 0 || input_width <= 0 || output_height <= 0 || output_width <= 0) {
    throw std::invalid_argument("Qwen3.5 RGB resize received invalid input");
  }
  const auto quantize = [](double value) {
    if (!std::isfinite(value)) throw std::invalid_argument("Qwen3.5 RGB frames must contain finite values");
    return static_cast<uint8_t>(std::clamp(std::nearbyint(value), 0.0, 255.0));
  };

  std::vector<uint8_t> source(static_cast<size_t>(input_height) * input_width * 3);
  std::transform(input, input + source.size(), source.begin(), quantize);
  std::vector<double> horizontal(static_cast<size_t>(input_height) * output_width * 3);
  if (input_width == output_width) {
    std::transform(source.begin(), source.end(), horizontal.begin(), [](uint8_t value) { return value; });
  } else {
    const auto spans = makeQwen3_5BicubicSpans(input_width, output_width);
    for (int32_t row = 0; row < input_height; ++row) {
      for (int32_t column = 0; column < output_width; ++column) {
        const auto& span = spans[column];
        for (int32_t channel = 0; channel < 3; ++channel) {
          double value = 0.0;
          for (size_t index = 0; index < span.weights.size(); ++index) {
            value += source[(static_cast<int64_t>(row) * input_width + span.first + index) * 3 + channel] * span.weights[index];
          }
          horizontal[(static_cast<int64_t>(row) * output_width + column) * 3 + channel] = value;
        }
      }
    }
  }

  if (input_height == output_height) {
    std::vector<uint8_t> output(horizontal.size());
    std::transform(horizontal.begin(), horizontal.end(), output.begin(), quantize);
    return output;
  }
  const auto spans = makeQwen3_5BicubicSpans(input_height, output_height);
  std::vector<uint8_t> output(static_cast<size_t>(output_height) * output_width * 3);
  for (int32_t row = 0; row < output_height; ++row) {
    const auto& span = spans[row];
    for (int32_t column = 0; column < output_width; ++column) {
      for (int32_t channel = 0; channel < 3; ++channel) {
        double value = 0.0;
        for (size_t index = 0; index < span.weights.size(); ++index) {
          value += horizontal[(static_cast<int64_t>(span.first + index) * output_width + column) * 3 + channel]
                   * span.weights[index];
        }
        output[(static_cast<int64_t>(row) * output_width + column) * 3 + channel] = quantize(value);
      }
    }
  }
  return output;
}

inline auto sampleQwen3_5VideoFrames(int32_t total_frames, double source_frames_per_second,
                                     double target_frames_per_second = 2.0, int32_t min_frames = 4, int32_t max_frames = 768)
    -> std::vector<int32_t> {
  if (total_frames <= 0 || !std::isfinite(source_frames_per_second) || source_frames_per_second <= 0.0
      || !std::isfinite(target_frames_per_second) || target_frames_per_second <= 0.0 || min_frames <= 0
      || max_frames < min_frames) {
    throw std::invalid_argument("Qwen3.5 video sampling received invalid metadata or limits");
  }
  int32_t sampled_frames = static_cast<int32_t>(total_frames / source_frames_per_second * target_frames_per_second);
  sampled_frames = std::min({std::max(sampled_frames, min_frames), max_frames, total_frames});

  std::vector<int32_t> indices(sampled_frames);
  if (sampled_frames == 1) {
    indices[0] = 0;
    return indices;
  }
  const auto python_round = [](double value) {
    const double lower_double = std::floor(value);
    const auto lower = static_cast<int32_t>(lower_double);
    const double fraction = value - lower_double;
    if (fraction < 0.5) return lower;
    if (fraction > 0.5) return lower + 1;
    return lower % 2 == 0 ? lower : lower + 1;
  };
  for (int32_t i = 0; i < sampled_frames; ++i) {
    const double position = static_cast<double>(i) * (total_frames - 1) / (sampled_frames - 1);
    indices[i] = python_round(position);
  }
  return indices;
}

inline auto calculateQwen3_5VideoTimestamps(std::vector<int32_t> frame_indices, double frames_per_second,
                                            int32_t temporal_patch_size = 2) -> std::vector<double> {
  if (frame_indices.empty() || !std::isfinite(frames_per_second) || frames_per_second <= 0.0 || temporal_patch_size <= 0) {
    throw std::invalid_argument("Qwen3.5 video timestamps require frames, positive fps, and temporal patch size");
  }
  for (size_t i = 0; i < frame_indices.size(); ++i) {
    if (frame_indices[i] < 0 || (i > 0 && frame_indices[i] < frame_indices[i - 1])) {
      throw std::invalid_argument("Qwen3.5 video frame indices must be non-negative and ordered");
    }
  }
  while (frame_indices.size() % static_cast<size_t>(temporal_patch_size) != 0) {
    frame_indices.push_back(frame_indices.back());
  }

  std::vector<double> timestamps;
  timestamps.reserve(frame_indices.size() / temporal_patch_size);
  for (size_t i = 0; i < frame_indices.size(); i += temporal_patch_size) {
    const double first = frame_indices[i] / frames_per_second;
    const double last = frame_indices[i + temporal_patch_size - 1] / frames_per_second;
    timestamps.push_back((first + last) / 2.0);
  }
  return timestamps;
}

inline auto makeQwen3_5VideoMarkers(const std::vector<double>& timestamps) -> std::string {
  if (timestamps.empty()) { throw std::invalid_argument("Qwen3.5 video markers require timestamps"); }
  std::ostringstream markers;
  markers << "<|vision_start|>";
  markers << std::fixed << std::setprecision(1);
  for (const double timestamp : timestamps) {
    if (!std::isfinite(timestamp) || timestamp < 0.0) {
      throw std::invalid_argument("Qwen3.5 video timestamps must be finite and non-negative");
    }
    markers << '<' << timestamp << " seconds><|vision_start|><|video_pad|><|vision_end|>";
  }
  markers << "<|vision_end|>";
  return markers.str();
}

class Qwen3_5VideoPreprocessor {
 public:
  explicit Qwen3_5VideoPreprocessor(int32_t min_pixels = 4 * 32 * 32, int32_t max_pixels = 24 * 32 * 32 * 1024,
                                    int32_t patch_size = 16, int32_t temporal_patch_size = 2, int32_t merge_size = 2)
      : min_pixels_(min_pixels),
        max_pixels_(max_pixels),
        patch_size_(patch_size),
        temporal_patch_size_(temporal_patch_size),
        merge_size_(merge_size) {
    if (min_pixels_ <= 0 || max_pixels_ < min_pixels_ || patch_size_ <= 0 || temporal_patch_size_ <= 0 || merge_size_ <= 0) {
      throw std::invalid_argument("Qwen3.5 video preprocessor received invalid geometry");
    }
  }

  [[nodiscard]] std::pair<int32_t, int32_t> smartResize(int32_t num_frames, int32_t height, int32_t width) const {
    if (num_frames < temporal_patch_size_) {
      throw std::invalid_argument("Qwen3.5 video requires at least one complete temporal patch");
    }
    if (height <= 0 || width <= 0) { throw std::invalid_argument("Qwen3.5 video dimensions must be positive"); }
    const int32_t factor = patch_size_ * merge_size_;
    if (height < factor || width < factor) {
      const double scale = std::max(static_cast<double>(factor) / height, static_cast<double>(factor) / width);
      height = static_cast<int32_t>(height * scale);
      width = static_cast<int32_t>(width * scale);
    }
    if (static_cast<double>(std::max(height, width)) / std::min(height, width) > 200.0) {
      throw std::invalid_argument("Qwen3.5 video aspect ratio must not exceed 200");
    }
    const auto python_round_to_factor = [](int32_t value, int32_t divisor) {
      const int32_t quotient = value / divisor;
      const int32_t remainder = value % divisor;
      if (remainder * 2 < divisor) return quotient * divisor;
      if (remainder * 2 > divisor) return (quotient + 1) * divisor;
      return (quotient % 2 == 0 ? quotient : quotient + 1) * divisor;
    };
    int32_t resized_height = python_round_to_factor(height, factor);
    int32_t resized_width = python_round_to_factor(width, factor);
    const int32_t padded_frames = python_round_to_factor(num_frames, temporal_patch_size_);
    const int64_t rounded_pixels = static_cast<int64_t>(padded_frames) * resized_height * resized_width;

    if (rounded_pixels > max_pixels_) {
      const double beta = std::sqrt(static_cast<double>(num_frames) * height * width / max_pixels_);
      resized_height = std::max(factor, static_cast<int32_t>(std::floor(height / beta / factor)) * factor);
      resized_width = std::max(factor, static_cast<int32_t>(std::floor(width / beta / factor)) * factor);
    } else if (rounded_pixels < min_pixels_) {
      const double beta = std::sqrt(static_cast<double>(min_pixels_) / (static_cast<double>(num_frames) * height * width));
      resized_height = std::max(factor, static_cast<int32_t>(std::ceil(height * beta / factor)) * factor);
      resized_width = std::max(factor, static_cast<int32_t>(std::ceil(width * beta / factor)) * factor);
    }
    return {resized_height, resized_width};
  }

  [[nodiscard]] Tensor resizeFrames(const Tensor& video_thwc) const {
    const auto& shape = video_thwc.shape();
    if (video_thwc.dtype() != kFloat32 || video_thwc.device() != kCPU || !video_thwc.isContiguous() || shape.size() != 4
        || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 || shape[3] != 3) {
      throw std::invalid_argument("Qwen3.5 video resize expects a contiguous float32 CPU RGB tensor in THWC layout");
    }
    const auto [resized_height, resized_width] = smartResize(shape[0], shape[1], shape[2]);
    auto resized = Tensor::empty({shape[0], resized_height, resized_width, 3}, kFloat32, kCPU).alloc();
    const int64_t input_frame_values = static_cast<int64_t>(shape[1]) * shape[2] * 3;
    const int64_t output_frame_values = static_cast<int64_t>(resized_height) * resized_width * 3;
    for (int32_t frame = 0; frame < shape[0]; ++frame) {
      const auto output = resizeQwen3_5RgbLikeTorchvision(video_thwc.ptr<float>() + frame * input_frame_values, shape[1],
                                                          shape[2], resized_height, resized_width);
      std::transform(output.begin(), output.end(), resized.ptr<float>() + frame * output_frame_values,
                     [](uint8_t value) { return static_cast<float>(value); });
    }
    return resized;
  }

  // Patchification consumes ordered, already-resized float32 RGB frames in
  // [T,H,W,3], [0,255]. Use resizeFrames() for decoded source-resolution frames.
  [[nodiscard]] std::pair<Tensor, Tensor> flattenNormalizedPatches(const Tensor& video_thwc) const {
    const auto& shape = video_thwc.shape();
    if (video_thwc.dtype() != kFloat32 || video_thwc.device() != kCPU || shape.size() != 4 || shape[3] != 3) {
      throw std::invalid_argument("Qwen3.5 video preprocessor expects a float32 CPU RGB tensor in THWC layout");
    }
    const int32_t frames = shape[0];
    const int32_t height = shape[1];
    const int32_t width = shape[2];
    const int32_t factor = patch_size_ * merge_size_;
    if (frames <= 0 || height <= 0 || width <= 0 || height % factor != 0 || width % factor != 0) {
      throw std::invalid_argument("Qwen3.5 resized video dimensions must be positive and divisible by patch_size * merge_size");
    }

    const int32_t pad = (temporal_patch_size_ - frames % temporal_patch_size_) % temporal_patch_size_;
    const int32_t padded_frames = frames + pad;
    const int32_t grid_t = padded_frames / temporal_patch_size_;
    const int32_t grid_h = height / patch_size_;
    const int32_t grid_w = width / patch_size_;
    const int32_t patch_features = 3 * temporal_patch_size_ * patch_size_ * patch_size_;
    auto patches = Tensor::empty({grid_t * grid_h * grid_w, patch_features}, kFloat32, kCPU).alloc();
    const auto* input = video_thwc.ptr<float>();
    auto* output = patches.ptr<float>();

    int64_t output_offset = 0;
    for (int32_t block_t = 0; block_t < grid_t; ++block_t) {
      for (int32_t block_h = 0; block_h < grid_h / merge_size_; ++block_h) {
        for (int32_t block_w = 0; block_w < grid_w / merge_size_; ++block_w) {
          for (int32_t merge_h = 0; merge_h < merge_size_; ++merge_h) {
            for (int32_t merge_w = 0; merge_w < merge_size_; ++merge_w) {
              for (int32_t channel = 0; channel < 3; ++channel) {
                for (int32_t temporal = 0; temporal < temporal_patch_size_; ++temporal) {
                  const int32_t source_t = std::min(block_t * temporal_patch_size_ + temporal, frames - 1);
                  for (int32_t patch_h = 0; patch_h < patch_size_; ++patch_h) {
                    const int32_t source_h = (block_h * merge_size_ + merge_h) * patch_size_ + patch_h;
                    for (int32_t patch_w = 0; patch_w < patch_size_; ++patch_w) {
                      const int32_t source_w = (block_w * merge_size_ + merge_w) * patch_size_ + patch_w;
                      const int64_t input_offset =
                          (((static_cast<int64_t>(source_t) * height + source_h) * width + source_w) * 3) + channel;
                      output[output_offset++] = input[input_offset] * (2.0F / 255.0F) - 1.0F;
                    }
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
    grid[0] = grid_t;
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
