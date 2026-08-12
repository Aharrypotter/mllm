// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "video_decoder.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#define MINIMP4_IMPLEMENTATION
#define MP4D_INFO_SUPPORTED 1
#include <minimp4.h>
#include <wels/codec_api.h>

#include <mllm/models/qwen3_5/video_preprocessor_qwen3_5.hpp>

namespace mllm::examples::qwen3_5 {
namespace {

struct InputBuffer {
  const std::vector<uint8_t>* bytes;
};

int readMp4(int64_t offset, void* buffer, size_t size, void* token) {
  const auto* input = static_cast<const InputBuffer*>(token);
  if (offset < 0 || static_cast<uint64_t>(offset) > input->bytes->size()
      || size > input->bytes->size() - static_cast<size_t>(offset)) {
    return 1;
  }
  std::memcpy(buffer, input->bytes->data() + offset, size);
  return 0;
}

auto readFile(const std::string& path, size_t max_input_bytes) -> std::vector<uint8_t> {
  std::error_code size_error;
  const auto size = std::filesystem::file_size(path, size_error);
  if (size_error || size == 0 || size > max_input_bytes || size > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument("video_path is empty, unavailable, or exceeds the configured byte limit");
  }
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::invalid_argument("unable to read video_path");
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw std::invalid_argument("unable to read the complete video_path");
  return bytes;
}

void appendNal(std::vector<uint8_t>& output, const void* bytes, int32_t size) {
  if (bytes == nullptr || size <= 0) throw std::invalid_argument("H.264 parameter set is invalid");
  static constexpr uint8_t start_code[] = {0, 0, 0, 1};
  output.insert(output.end(), std::begin(start_code), std::end(start_code));
  const auto* begin = static_cast<const uint8_t*>(bytes);
  output.insert(output.end(), begin, begin + size);
}

auto readNalSize(const uint8_t* bytes, int32_t length_bytes) -> uint32_t {
  uint32_t size = 0;
  for (int32_t index = 0; index < length_bytes; ++index) { size = (size << 8) | bytes[index]; }
  return size;
}

bool isValidAvcSampleFraming(const std::vector<uint8_t>& bytes, size_t offset, size_t sample_bytes, int32_t length_bytes) {
  size_t position = offset;
  size_t remaining = sample_bytes;
  while (remaining != 0) {
    if (remaining < static_cast<size_t>(length_bytes)) return false;
    const uint32_t nal_size = readNalSize(bytes.data() + position, length_bytes);
    position += length_bytes;
    remaining -= length_bytes;
    if (nal_size == 0 || nal_size > remaining) return false;
    const uint8_t nal_header = bytes[position];
    const uint8_t nal_type = nal_header & 0x1f;
    if ((nal_header & 0x80) != 0 || nal_type == 0 || nal_type > 23) return false;
    position += nal_size;
    remaining -= nal_size;
  }
  return true;
}

auto detectNalLengthBytes(const std::vector<uint8_t>& bytes, const MP4D_demux_t& mp4, int32_t track_index,
                          uint32_t source_frame_count) -> int32_t {
  std::vector<int32_t> candidates;
  for (const int32_t length_bytes : {1, 2, 4}) {
    bool valid = true;
    for (uint32_t sample = 0; sample < source_frame_count && valid; ++sample) {
      uint32_t frame_bytes = 0;
      const auto offset = MP4D_frame_offset(&mp4, track_index, sample, &frame_bytes, nullptr, nullptr);
      valid = offset >= 0 && frame_bytes != 0 && static_cast<uint64_t>(offset) + frame_bytes <= bytes.size()
              && isValidAvcSampleFraming(bytes, static_cast<size_t>(offset), frame_bytes, length_bytes);
    }
    if (valid) candidates.push_back(length_bytes);
  }
  if (candidates.size() != 1) { throw std::invalid_argument("H.264 sample NAL-length framing is invalid or ambiguous"); }
  return candidates.front();
}

struct DecoderDeleter {
  void operator()(ISVCDecoder* decoder) const {
    if (decoder != nullptr) {
      decoder->Uninitialize();
      WelsDestroyDecoder(decoder);
    }
  }
};

void collectFrame(unsigned char* const planes[3], const SBufferInfo& info, const std::vector<int32_t>& selected_indices,
                  int32_t& decoded_frames, int32_t& width, int32_t& height, std::vector<float>& selected_rgb,
                  int64_t max_decoded_pixels, int64_t max_selected_pixels) {
  if (info.iBufferStatus != 1) return;
  const int32_t frame_width = info.UsrData.sSystemBuffer.iWidth;
  const int32_t frame_height = info.UsrData.sSystemBuffer.iHeight;
  if (frame_width <= 0 || frame_height <= 0 || frame_width % 2 != 0 || frame_height % 2 != 0 || planes[0] == nullptr
      || planes[1] == nullptr || planes[2] == nullptr) {
    throw std::runtime_error("OpenH264 produced an invalid I420 frame");
  }
  if (width == 0) {
    width = frame_width;
    height = frame_height;
  } else if (width != frame_width || height != frame_height) {
    throw std::invalid_argument("mid-stream video dimension changes are outside the bounded contract");
  }
  const int64_t frame_pixels = static_cast<int64_t>(width) * height;
  if (frame_pixels > max_decoded_pixels / (decoded_frames + 1)) {
    throw std::invalid_argument("decoded video exceeds the configured pixel limit");
  }

  if (std::binary_search(selected_indices.begin(), selected_indices.end(), decoded_frames)) {
    const int64_t selected_pixels = static_cast<int64_t>(selected_rgb.size() / 3);
    if (selected_pixels > max_selected_pixels || frame_pixels > max_selected_pixels - selected_pixels) {
      throw std::invalid_argument("selected source frames exceed the configured pixel limit");
    }
    const size_t old_size = selected_rgb.size();
    selected_rgb.resize(old_size + static_cast<size_t>(width) * height * 3);
    models::qwen3_5::convertQwen3_5I420ToRgb(
        planes[0], info.UsrData.sSystemBuffer.iStride[0], planes[1], info.UsrData.sSystemBuffer.iStride[1], planes[2],
        info.UsrData.sSystemBuffer.iStride[1], width, height, selected_rgb.data() + old_size);
  }
  ++decoded_frames;
}

}  // namespace

DecodedVideo decodeH264Mp4Portable(const std::string& path, double target_frames_per_second, int32_t min_sampled_frames,
                                   int32_t max_sampled_frames, size_t max_input_bytes, int64_t max_decoded_pixels,
                                   int64_t max_selected_pixels) {
  if (max_input_bytes == 0 || max_decoded_pixels <= 0 || max_selected_pixels <= 0) {
    throw std::invalid_argument("video resource limits must be positive");
  }
  const auto bytes = readFile(path, max_input_bytes);
  InputBuffer input{&bytes};
  MP4D_demux_t mp4{};
  if (!MP4D_open(&mp4, readMp4, &input, bytes.size())) throw std::invalid_argument("minimp4 rejected video_path");
  const auto close_mp4 = [&mp4](void*) { MP4D_close(&mp4); };
  std::unique_ptr<void, decltype(close_mp4)> mp4_guard(reinterpret_cast<void*>(1), close_mp4);

  int32_t track_index = -1;
  for (int32_t index = 0; index < mp4.track_count; ++index) {
    if (mp4.track[index].handler_type == MP4D_HANDLER_TYPE_VIDE) {
      if (track_index >= 0) throw std::invalid_argument("multiple video tracks are outside the bounded contract");
      track_index = index;
    }
  }
  if (track_index < 0 || mp4.track[track_index].object_type_indication != MP4_OBJECT_TYPE_AVC) {
    throw std::invalid_argument("video_path must contain exactly one H.264 video track");
  }
  const auto& track = mp4.track[track_index];
  const uint32_t source_frame_count = track.sample_count;
  const uint32_t source_timescale = track.timescale;
  if (source_frame_count == 0 || source_timescale == 0 || source_frame_count > 100000) {
    throw std::invalid_argument("video track has invalid or excessive frame metadata");
  }

  uint64_t duration_units = 0;
  for (uint32_t sample = 0; sample < source_frame_count; ++sample) {
    uint32_t frame_bytes = 0;
    uint32_t timestamp = 0;
    uint32_t duration = 0;
    const auto offset = MP4D_frame_offset(&mp4, track_index, sample, &frame_bytes, &timestamp, &duration);
    if (offset < 0 || frame_bytes == 0 || duration == 0 || static_cast<uint64_t>(offset) + frame_bytes > bytes.size()) {
      throw std::invalid_argument("MP4 sample metadata is invalid or points outside video_path");
    }
    duration_units += duration;
  }
  const double source_fps = source_frame_count / (static_cast<double>(duration_units) / source_timescale);
  const auto selected_indices = models::qwen3_5::sampleQwen3_5VideoFrames(
      source_frame_count, source_fps, target_frames_per_second, min_sampled_frames, max_sampled_frames);
  const int32_t nal_length_bytes = detectNalLengthBytes(bytes, mp4, track_index, source_frame_count);

  std::vector<uint8_t> parameter_sets;
  for (int32_t index = 0;; ++index) {
    int32_t size = 0;
    const void* sps = MP4D_read_sps(&mp4, track_index, index, &size);
    if (sps == nullptr) break;
    appendNal(parameter_sets, sps, size);
  }
  for (int32_t index = 0;; ++index) {
    int32_t size = 0;
    const void* pps = MP4D_read_pps(&mp4, track_index, index, &size);
    if (pps == nullptr) break;
    appendNal(parameter_sets, pps, size);
  }
  if (parameter_sets.empty()) throw std::invalid_argument("H.264 parameter sets are missing");

  ISVCDecoder* raw_decoder = nullptr;
  if (WelsCreateDecoder(&raw_decoder) != 0 || raw_decoder == nullptr) {
    throw std::runtime_error("cannot create OpenH264 decoder");
  }
  std::unique_ptr<ISVCDecoder, DecoderDeleter> decoder(raw_decoder);
  SDecodingParam decoder_parameters{};
  decoder_parameters.sVideoProperty.size = sizeof(decoder_parameters.sVideoProperty);
  decoder_parameters.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_AVC;
  decoder_parameters.eEcActiveIdc = ERROR_CON_DISABLE;
  if (decoder->Initialize(&decoder_parameters) != 0) throw std::runtime_error("cannot initialize OpenH264 decoder");

  int32_t decoded_frames = 0;
  int32_t width = 0;
  int32_t height = 0;
  std::vector<float> selected_rgb;
  for (uint32_t sample = 0; sample < source_frame_count; ++sample) {
    uint32_t frame_bytes = 0;
    uint32_t timestamp = 0;
    uint32_t duration = 0;
    const auto offset = MP4D_frame_offset(&mp4, track_index, sample, &frame_bytes, &timestamp, &duration);
    std::vector<uint8_t> access_unit;
    if (sample == 0) access_unit = parameter_sets;
    size_t position = static_cast<size_t>(offset);
    size_t remaining = frame_bytes;
    while (remaining != 0) {
      if (remaining < static_cast<size_t>(nal_length_bytes)) { throw std::invalid_argument("truncated MP4 NAL length"); }
      const uint32_t nal_size = readNalSize(bytes.data() + position, nal_length_bytes);
      position += nal_length_bytes;
      remaining -= nal_length_bytes;
      if (nal_size == 0 || nal_size > remaining || nal_size > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("invalid MP4 NAL size");
      }
      appendNal(access_unit, bytes.data() + position, static_cast<int32_t>(nal_size));
      position += nal_size;
      remaining -= nal_size;
    }
    if (access_unit.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      throw std::invalid_argument("H.264 access unit exceeds decoder limits");
    }
    unsigned char* output_planes[3] = {nullptr, nullptr, nullptr};
    SBufferInfo output_info{};
    output_info.uiInBsTimeStamp = timestamp;
    if (decoder->DecodeFrameNoDelay(access_unit.data(), static_cast<int32_t>(access_unit.size()), output_planes, &output_info)
        != dsErrorFree) {
      throw std::invalid_argument("OpenH264 rejected an access unit");
    }
    collectFrame(output_planes, output_info, selected_indices, decoded_frames, width, height, selected_rgb, max_decoded_pixels,
                 max_selected_pixels);
  }

  int32_t buffered_frames = 0;
  decoder->GetOption(DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER, &buffered_frames);
  while (buffered_frames-- > 0) {
    unsigned char* output_planes[3] = {nullptr, nullptr, nullptr};
    SBufferInfo output_info{};
    if (decoder->FlushFrame(output_planes, &output_info) != dsErrorFree) { throw std::runtime_error("OpenH264 flush failed"); }
    collectFrame(output_planes, output_info, selected_indices, decoded_frames, width, height, selected_rgb, max_decoded_pixels,
                 max_selected_pixels);
  }
  if (decoded_frames != static_cast<int32_t>(source_frame_count)
      || selected_rgb.size() != static_cast<size_t>(selected_indices.size()) * width * height * 3) {
    throw std::invalid_argument("decoded frame count does not match MP4 sample metadata");
  }

  auto frames = Tensor::empty({static_cast<int32_t>(selected_indices.size()), height, width, 3}, kFloat32, kCPU).alloc();
  std::copy(selected_rgb.begin(), selected_rgb.end(), frames.ptr<float>());
  return {.frames_thwc = std::move(frames),
          .source_frame_indices = selected_indices,
          .source_frames_per_second = source_fps,
          .source_frame_count = static_cast<int32_t>(source_frame_count),
          .source_width = width,
          .source_height = height};
}

}  // namespace mllm::examples::qwen3_5
