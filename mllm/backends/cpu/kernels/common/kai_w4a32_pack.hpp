// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace mllm::cpu::common {

// KAI W4A32 microkernel packing dimensions shared by conversion and ARM
// runtime paths.
struct KaiW4A32Tile {
  size_t nr;
  size_t kr;
  size_t sr;
};

// Resolves the packing dimensions for a supported KAI W4A32 microkernel name.
// Throws std::invalid_argument when tile_name is unsupported.
KaiW4A32Tile kaiW4A32TileFromName(std::string_view tile_name);

// Returns the packed byte size for an [out_channels, in_channels] float32
// weight matrix. Throws std::invalid_argument for zero/unsupported dimensions
// or an unsupported tile_name.
size_t kaiW4A32PackedSize(size_t out_channels, size_t in_channels, std::string_view tile_name);

// Quantizes and packs an [out_channels, in_channels] float32 weight matrix,
// with an optional out_channels-element float32 bias, into packed_weight.
// The caller must allocate kaiW4A32PackedSize() bytes. Throws
// std::invalid_argument for null required buffers, invalid dimensions, or an
// unsupported tile_name.
void kaiW4A32QuantizeAndPack(uint8_t* packed_weight, const float* weight, const float* bias, size_t out_channels,
                             size_t in_channels, std::string_view tile_name);

}  // namespace mllm::cpu::common
