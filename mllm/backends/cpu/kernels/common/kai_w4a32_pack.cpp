// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/common/kai_w4a32_pack.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "kai/kai_common.h"
#include "kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"

namespace mllm::cpu::common {
namespace {

constexpr size_t kBlockLength = 32;

size_t roundup(size_t value, size_t multiple) { return ((value + multiple - 1) / multiple) * multiple; }

size_t nativeStride(size_t k) { return roundup(k, 2) / 2; }

size_t scaleStride(size_t k) { return (roundup(k, kBlockLength) / kBlockLength) * sizeof(uint16_t); }

uint16_t truncateToBFloat16(float value) {
  // Match KleidiAI's portable kai_cast_bf16_f32 conversion so model files are
  // deterministic across converter hosts without native BF16 instructions.
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

void quantizeNxk(const float* weight, size_t n, size_t k, uint8_t* quantized, uint16_t* scales) {
  constexpr int32_t kInt4Min = -8;
  constexpr int32_t kInt4Max = 7;

  const size_t row_stride = nativeStride(k);
  const size_t blocks_per_row = roundup(k, kBlockLength) / kBlockLength;
  std::memset(quantized, 0, n * row_stride);

  for (size_t row = 0; row < n; ++row) {
    const float* src = weight + row * k;
    for (size_t block = 0; block < blocks_per_row; ++block) {
      float absmax = 0.0F;
      float signed_max = 0.0F;
      for (size_t offset = 0; offset < kBlockLength; ++offset) {
        const size_t column = block * kBlockLength + offset;
        if (column >= k) { break; }
        const float absolute = std::fabs(src[column]);
        if (absmax < absolute) {
          absmax = absolute;
          signed_max = src[column];
        }
      }

      const float scale = signed_max / -8.0F;
      const float reciprocal = scale == 0.0F ? 0.0F : 1.0F / scale;
      scales[row * blocks_per_row + block] = truncateToBFloat16(scale);

      for (size_t offset = 0; offset < kBlockLength; ++offset) {
        const size_t column = block * kBlockLength + offset;
        if (column >= k) { break; }
        int32_t quantized_value = static_cast<int32_t>(std::round(src[column] * reciprocal));
        quantized_value = std::clamp(quantized_value, kInt4Min, kInt4Max);
        const uint8_t nibble = static_cast<uint8_t>(quantized_value + 8);
        uint8_t& byte = quantized[row * row_stride + column / 2];
        if ((column % 2) == 0) {
          byte = nibble;
        } else {
          byte |= static_cast<uint8_t>(nibble << 4);
        }
      }
    }
  }
}

}  // namespace

KaiW4A32Tile kaiW4A32TileFromName(std::string_view tile_name) {
  if (tile_name == "qai8dxp1x8_qsi4c32p4x8_1x4x32" || tile_name == "qai8dxp4x8_qsi4c32p4x8_8x4x32"
      || tile_name == "qai8dxp4x8_qsi4c32p4x8_16x4x32") {
    return {.nr = 4, .kr = 16, .sr = 2};
  }
  if (tile_name == "qai8dxp1x8_qsi4c32p8x8_1x8x32" || tile_name == "qai8dxp4x8_qsi4c32p8x8_4x8x32") {
    return {.nr = 8, .kr = 16, .sr = 2};
  }
  if (tile_name == "qai8dxp1x4_qsi4c32p4x4_1x4") { return {.nr = 4, .kr = 8, .sr = 2}; }
  throw std::invalid_argument("Unsupported KAI W4A32 tile configuration: " + std::string(tile_name));
}

size_t kaiW4A32PackedSize(size_t out_channels, size_t in_channels, std::string_view tile_name) {
  if (out_channels == 0 || in_channels == 0 || (in_channels % kBlockLength) != 0) {
    throw std::invalid_argument("KAI W4A32 requires non-zero dimensions and input channels divisible by 32");
  }
  const auto tile = kaiW4A32TileFromName(tile_name);
  return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(out_channels, in_channels, tile.nr, tile.kr, tile.sr,
                                                                   kBlockLength, kai_dt_bf16);
}

void kaiW4A32QuantizeAndPack(uint8_t* packed_weight, const float* weight, const float* bias, size_t out_channels,
                             size_t in_channels, std::string_view tile_name) {
  if (packed_weight == nullptr || weight == nullptr) {
    throw std::invalid_argument("KAI W4A32 packing requires non-null output and weight buffers");
  }
  (void)kaiW4A32PackedSize(out_channels, in_channels, tile_name);
  const auto tile = kaiW4A32TileFromName(tile_name);

  std::vector<uint8_t> quantized(out_channels * nativeStride(in_channels));
  std::vector<uint16_t> scales(out_channels * scaleStride(in_channels) / sizeof(uint16_t));
  quantizeNxk(weight, out_channels, in_channels, quantized.data(), scales.data());

  kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params{
      .lhs_zero_point = 1,
      .rhs_zero_point = 8,
      .scale_dt = kai_dt_bf16,
  };
  kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(1, out_channels, in_channels, tile.nr, tile.kr, tile.sr, kBlockLength,
                                            quantized.data(), nativeStride(in_channels), bias, scales.data(),
                                            scaleStride(in_channels), packed_weight, 0, &params);
}

}  // namespace mllm::cpu::common
