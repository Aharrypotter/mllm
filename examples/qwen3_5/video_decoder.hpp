// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <mllm/core/Tensor.hpp>

namespace mllm::examples::qwen3_5 {

struct DecodedVideo {
  Tensor frames_thwc = Tensor::nil();
  std::vector<int32_t> source_frame_indices;
  double source_frames_per_second = 0.0;
  int32_t source_frame_count = 0;
  int32_t source_width = 0;
  int32_t source_height = 0;
};

// Portable/reference H.264-in-MP4 backend for the example runner. Qwen3.5
// model code consumes DecodedVideo and remains independent of this backend.
DecodedVideo decodeH264Mp4Portable(const std::string& path, double target_frames_per_second = 2.0,
                                   int32_t min_sampled_frames = 4, int32_t max_sampled_frames = 768,
                                   size_t max_input_bytes = 256ULL * 1024 * 1024,
                                   int64_t max_decoded_pixels = 512LL * 1024 * 1024,
                                   int64_t max_selected_pixels = 16LL * 1024 * 1024);

}  // namespace mllm::examples::qwen3_5
