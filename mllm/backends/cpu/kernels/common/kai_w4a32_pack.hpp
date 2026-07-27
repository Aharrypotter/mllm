// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace mllm::cpu::common {

// Host-side packing parameters shared by conversion and ARM runtime paths.
struct KaiW4A32Tile {
  size_t nr;
  size_t kr;
  size_t sr;
};

KaiW4A32Tile kaiW4A32TileFromName(std::string_view tile_name);

size_t kaiW4A32PackedSize(size_t out_channels, size_t in_channels, std::string_view tile_name);

void kaiW4A32QuantizeAndPack(uint8_t* packed_weight, const float* weight, const float* bias, size_t out_channels,
                             size_t in_channels, std::string_view tile_name);

}  // namespace mllm::cpu::common
