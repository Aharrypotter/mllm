// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mllm/mllm.hpp>
#include <mllm/models/qwen3_5/video_preprocessor_qwen3_5.hpp>
#include <xxHash/xxhash.h>

#include "video_decoder.hpp"

namespace {

auto toRgbBytes(const mllm::Tensor& frames) -> std::vector<uint8_t> {
  std::vector<uint8_t> bytes(frames.numel());
  std::transform(frames.ptr<float>(), frames.ptr<float>() + frames.numel(), bytes.begin(),
                 [](float value) { return static_cast<uint8_t>(value); });
  return bytes;
}

void verifyMovingSquareFixture(const mllm::examples::qwen3_5::DecodedVideo& video, const std::vector<uint8_t>& rgb) {
  static const std::vector<int32_t> expected_indices = {0, 2, 4, 6, 9, 11, 13, 15};
  constexpr uint64_t expected_rgb_xxh3 = 0xf71b64dcc06dbd9aULL;
  if (video.source_frame_count != 16 || std::abs(video.source_frames_per_second - 4.0) > 1e-12 || video.source_width != 192
      || video.source_height != 128 || video.source_frame_indices != expected_indices || rgb.size() != 589824
      || XXH3_64bits(rgb.data(), rgb.size()) != expected_rgb_xxh3) {
    throw std::runtime_error("moving-square decoder output does not match the committed fixture contract");
  }
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 3 || argc > 6) {
    std::cerr << "usage: mllm-qwen3-5-portable-video-decoder-smoke input.mp4 selected.rgb24 [patches.f32] "
                 "[oracle.rgb24] "
                 "[--expect-moving-square]\n";
    return 2;
  }
  std::vector<std::string> optional_paths;
  bool expect_moving_square = false;
  for (int index = 3; index < argc; ++index) {
    if (std::string(argv[index]) == "--expect-moving-square") {
      if (expect_moving_square) throw std::invalid_argument("--expect-moving-square may be specified only once");
      expect_moving_square = true;
    } else {
      optional_paths.emplace_back(argv[index]);
    }
  }
  if (optional_paths.size() > 2) throw std::invalid_argument("too many output or oracle paths");

  mllm::initializeContext();
  auto video = mllm::examples::qwen3_5::decodeH264Mp4Portable(argv[1], 2.0, 4, 64);
  const auto rgb = toRgbBytes(video.frames_thwc);
  if (expect_moving_square) verifyMovingSquareFixture(video, rgb);
  if (optional_paths.size() == 2) {
    std::ifstream oracle(optional_paths[1], std::ios::binary);
    if (!oracle) throw std::runtime_error("unable to open RGB oracle input");
    std::vector<uint8_t> oracle_bytes(rgb.size());
    oracle.read(reinterpret_cast<char*>(oracle_bytes.data()), static_cast<std::streamsize>(oracle_bytes.size()));
    if (!oracle || oracle.peek() != std::ifstream::traits_type::eof()) {
      throw std::runtime_error("RGB oracle input has the wrong byte size");
    }
    if (oracle_bytes != rgb) throw std::runtime_error("decoded RGB output does not match the oracle input");
  }
  std::ofstream output(argv[2], std::ios::binary);
  if (!output) throw std::runtime_error("unable to open RGB output");
  output.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
  if (!output) throw std::runtime_error("unable to write complete RGB output");
  if (!optional_paths.empty()) {
    const mllm::models::qwen3_5::Qwen3_5VideoPreprocessor preprocessor;
    auto resized = preprocessor.resizeFrames(video.frames_thwc);
    auto [patches, grid] = preprocessor.flattenNormalizedPatches(resized);
    if (expect_moving_square
        && (grid.shape() != std::vector<int32_t>({1, 3}) || grid.ptr<int32_t>()[0] != 4 || grid.ptr<int32_t>()[1] != 8
            || grid.ptr<int32_t>()[2] != 12 || patches.shape() != std::vector<int32_t>({384, 1536}))) {
      throw std::runtime_error("moving-square preprocessing output does not match the committed fixture contract");
    }
    std::ofstream patch_output(optional_paths[0], std::ios::binary);
    if (!patch_output) throw std::runtime_error("unable to open patch output");
    patch_output.write(reinterpret_cast<const char*>(patches.ptr<float>()),
                       static_cast<std::streamsize>(patches.numel() * sizeof(float)));
    if (!patch_output) throw std::runtime_error("unable to write complete patch output");
    std::cout << "video_grid_thw=" << grid.ptr<int32_t>()[0] << ',' << grid.ptr<int32_t>()[1] << ',' << grid.ptr<int32_t>()[2]
              << '\n'
              << "patch_shape=" << patches.shape()[0] << ',' << patches.shape()[1] << '\n';
  }
  std::cout << "QWEN35_PORTABLE_VIDEO_DECODER_SMOKE_OK\n"
            << "source_frames=" << video.source_frame_count << '\n'
            << "source_fps=" << video.source_frames_per_second << '\n'
            << "width=" << video.source_width << '\n'
            << "height=" << video.source_height << '\n'
            << "selected_indices=";
  for (size_t index = 0; index < video.source_frame_indices.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << video.source_frame_indices[index];
  }
  std::cout << "\nselected_rgb_bytes=" << video.frames_thwc.numel() << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "QWEN35_PORTABLE_VIDEO_DECODER_SMOKE_ERROR: " << error.what() << '\n';
  return 1;
}
