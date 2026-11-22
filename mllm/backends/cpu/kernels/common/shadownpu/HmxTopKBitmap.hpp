// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include "tools/hmx_int8_matmul/include/hmx_i8_packed_topk_layout.h"

namespace mllm::cpu::shadownpu {

// Borrowed from a completed head in the FastRPC arena. Only the callback may use it.
struct HmxTopKBitmapView {
  const void* data = nullptr;
  size_t bytes = 0;
};

class HmxTopKBitmapRows {
 public:
  bool reset(HmxTopKBitmapView view, int32_t rows, int32_t valid_columns) noexcept {
    data_ = nullptr;
    if (view.data == nullptr || rows <= 0 || rows > kMaxRows || rows % HMX_I8_TOPK_BITMAP_ROWS != 0
        || valid_columns < rows || valid_columns > UINT16_MAX) {
      return false;
    }
    rows_ = rows;
    prefix_ = valid_columns - rows;
    group_offset_[0] = 0;
    for (int32_t group = 0; group < rows / HMX_I8_TOPK_BITMAP_ROWS; ++group) {
      const int32_t first_visible = prefix_ + group * HMX_I8_TOPK_BITMAP_ROWS + 1;
      group_offset_[group + 1] = group_offset_[group] + hmx_i8_topk_bitmap_group_words(first_visible);
    }
    if (group_offset_[rows / HMX_I8_TOPK_BITMAP_ROWS] > view.bytes / sizeof(uint32_t)) return false;
    data_ = static_cast<const unsigned char*>(view.data);
    return true;
  }

  bool decode(int32_t row, std::span<int32_t> indices) const noexcept {
    if (data_ == nullptr || row < 0 || row >= rows_ || indices.empty()) return false;
    const int32_t visible = prefix_ + row + 1;
    if (indices.size() > static_cast<size_t>(visible)) return false;
    const int32_t group = row / HMX_I8_TOPK_BITMAP_ROWS;
    const int32_t lane = row % HMX_I8_TOPK_BITMAP_ROWS;
    size_t emitted = 0;
    int32_t column = 0;
    const size_t end = group_offset_[group + 1];
    for (size_t word = group_offset_[group] + lane; word < end;
         word += HMX_I8_TOPK_BITMAP_ROWS, column += HMX_I8_TOPK_BITMAP_COLS) {
      uint32_t mask;
      std::memcpy(&mask, data_ + word * sizeof(mask), sizeof(mask));
      if (mask == 0) continue;
      if (column + static_cast<int32_t>(std::bit_width(mask)) > visible
          || static_cast<size_t>(std::popcount(mask)) > indices.size() - emitted) {
        return false;
      }
      while (mask != 0) {
        indices[emitted++] = column + std::countr_zero(mask);
        mask &= mask - 1;
      }
    }
    return emitted == indices.size();
  }

 private:
  static constexpr int kMaxRows = 512;
  const unsigned char* data_ = nullptr;
  int32_t rows_ = 0;
  int32_t prefix_ = 0;
  // reset initializes the complete prefix consumed by decode.
  std::array<size_t, kMaxRows / HMX_I8_TOPK_BITMAP_ROWS + 1> group_offset_;
};

}  // namespace mllm::cpu::shadownpu
