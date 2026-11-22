#include "mllm/backends/cpu/kernels/common/shadownpu/HmxTopKBitmap.hpp"
#include "tools/hmx_int8_matmul/include/hmx_i8_topk_bitmap_decode.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using mllm::cpu::shadownpu::HmxTopKBitmapRows;

namespace {
constexpr int rows = 512;

template <class T>
std::vector<T> load(const std::filesystem::path& name) {
  std::ifstream file(name, std::ios::binary | std::ios::ate);
  assert(file);
  const auto bytes = file.tellg();
  assert(bytes > 0 && bytes % sizeof(T) == 0);
  std::vector<T> contents(static_cast<size_t>(bytes) / sizeof(T));
  file.seekg(0);
  assert(file.read(reinterpret_cast<char*>(contents.data()), bytes));
  return contents;
}

size_t check(const std::filesystem::path& prefix, int columns) {
  const auto indices = load<uint16_t>(prefix.string() + ".indices");
  const auto offsets = load<int32_t>(prefix.string() + ".offsets");
  assert(offsets.size() == rows + 1 && offsets.front() == 0 && static_cast<size_t>(offsets.back()) == indices.size());
  std::vector<uint32_t> bitmap(hmx_i8_topk_bitmap_words(rows, columns));
  size_t position = 0;
  for (int row = 0; row < rows; row += HMX_I8_TOPK_BITMAP_ROWS) {
    const int visible = columns - rows + row + 1;
    for (int lane = 0; lane < HMX_I8_TOPK_BITMAP_ROWS; ++lane) {
      for (int idx = offsets[row + lane]; idx < offsets[row + lane + 1]; ++idx) {
        const uint16_t key = indices[idx];
        assert(key < visible + lane);
        auto& mask = bitmap.at(position + static_cast<size_t>(key / 32) * 4 + lane);
        const uint32_t bit = uint32_t(1) << (key % 32);
        assert((mask & bit) == 0);
        mask |= bit;
      }
    }
    position += hmx_i8_topk_bitmap_group_words(visible);
  }
  assert(position == bitmap.size());
  std::vector<int32_t> baseline(indices.size());
  const size_t bytes = bitmap.size() * sizeof(uint32_t);
  assert(hmx_i8_decode_topk_bitmap_rows(bitmap.data(), bytes, rows, columns, 0, rows, offsets.data(), baseline.data(),
                                      baseline.size()) == 0);
  assert(std::equal(baseline.begin(), baseline.end(), indices.begin()));

  HmxTopKBitmapRows reader;
  assert(!reader.reset({bitmap.data(), bytes - 1}, rows, columns));
  assert(reader.reset({bitmap.data(), bytes}, rows, columns));
  std::vector<int32_t> scratch(columns + 1);
  // Exercise split starts which are not aligned to the producer's four rows.
  for (int start : {3, 0, 509}) {
    const int end = start == 0 ? 3 : start == 3 ? 509 : rows;
    for (int row = start; row < end; ++row) {
      const int32_t keep = offsets[row + 1] - offsets[row];
      assert(reader.decode(row, {scratch.data(), static_cast<size_t>(keep)}));
      assert(std::equal(scratch.begin(), scratch.begin() + keep, baseline.begin() + offsets[row]));
    }
  }
  const auto first = static_cast<size_t>(offsets[1]);
  assert(!reader.decode(0, {scratch.data(), first + 1}));
  if (first > 1) assert(!reader.decode(0, {scratch.data(), first - 1}));
  assert(!reader.decode(-1, {scratch.data(), first}));
  assert(!reader.decode(rows, {scratch.data(), first}));
  return indices.size();
}

void checkCausalPadding() {
  // At a fresh prefill, row zero sees key zero. A future bit must fail even
  // if it is inside the storage allocated for its four-row group.
  std::vector<uint32_t> bitmap(hmx_i8_topk_bitmap_words(rows, rows));
  bitmap[0] = 1;
  HmxTopKBitmapRows reader;
  assert(reader.reset({bitmap.data(), bitmap.size() * sizeof(uint32_t)}, rows, rows));
  int32_t index = -1;
  assert(reader.decode(0, {&index, 1}) && index == 0);
  bitmap[0] = 2;
  assert(!reader.decode(0, {&index, 1}));
  bitmap[0] = 0;
  assert(!reader.decode(0, {&index, 1}));
}
}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path root = argv[1];
  checkCausalPadding();
  size_t compared = 0;
  int matrices = 0;
  for (int layer : {2, 15, 27}) {
    for (int keys : {2560, 3072, 3584, 4096}) {
      for (int head = 0; head < 12; ++head) {
        const auto stem = "layer" + std::to_string(layer) + "-k" + std::to_string(keys) + "-h" + std::to_string(head);
        compared += check(root / stem, keys);
        ++matrices;
      }
    }
  }
  assert(matrices == 144 && compared == 33857061);
  std::cout << "BITMAP_DIRECT_ROWS_OK matrices=" << matrices << " indices=" << compared << '\n';
}
