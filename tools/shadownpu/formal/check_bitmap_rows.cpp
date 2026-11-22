#include "mllm/backends/cpu/kernels/common/shadownpu/HmxTopKBitmap.hpp"
#include "tools/hmx_int8_matmul/include/hmx_i8_topk_bitmap_decode.h"

#include <algorithm>
#include <array>
#include <bit>
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
constexpr int tile_rows = 16;
constexpr int page_keys = 64;
constexpr uint16_t unused_lane = 0xffff;

struct PanelStats {
  size_t tiles = 0;
  size_t offset_checks = 0;
  size_t selected = 0;
  size_t active_keys = 0;
  size_t row_pages = 0;
  size_t tile_pages = 0;
  size_t potential_keys = 0;
};

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

void checkPanels(const HmxTopKBitmapRows& reader, const std::vector<uint16_t>& indices,
                 const std::vector<int32_t>& offsets, int columns, PanelStats& stats) {
  const size_t pages = (columns + page_keys - 1) / page_keys;
  std::vector<uint64_t> active(pages);
  std::vector<uint64_t> expected_active(pages);
  std::vector<uint16_t> expected(static_cast<size_t>(columns) * tile_rows);
  std::vector<uint16_t> actual(expected.size());
  std::vector<int32_t> scratch(columns);
  std::vector<uint16_t> weights;

  // The measured reuse covers a disjoint partition. Three extra tiles exercise
  // a worker starting inside each of the producer's four-row groups.
  for (int tile = 0; tile < rows / tile_rows + 3; ++tile) {
    const bool disjoint = tile < rows / tile_rows;
    const int first = disjoint ? tile * tile_rows : rows / 2 + 1 + tile - rows / tile_rows;
    const int base = offsets[first];
    std::array<size_t, tile_rows + 1> weight_offsets{};
    weights.resize(offsets[first + tile_rows] - base);
    std::fill(active.begin(), active.end(), 0);
    std::fill(expected_active.begin(), expected_active.end(), 0);
    size_t row_pages = 0;
    for (int lane = 0; lane < tile_rows; ++lane) {
      const int start = offsets[first + lane];
      const int end = offsets[first + lane + 1];
      const auto keep = static_cast<size_t>(end - start);
      weight_offsets[lane + 1] = end - base;
      assert(reader.decode(first + lane, {scratch.data(), keep}, active));
      assert(std::equal(scratch.begin(), scratch.begin() + keep, indices.begin() + start));
      int prior_page = -1;
      for (int i = start; i < end; ++i) {
        const auto key = indices[i];
        const int page = key / page_keys;
        expected_active[page] |= uint64_t(1) << (key % page_keys);
        row_pages += page != prior_page;
        prior_page = page;
        weights[i - base] = static_cast<uint16_t>((static_cast<unsigned>(i) * 101 % 65534) + 1);
      }
    }
    assert(active == expected_active);

    size_t active_count = 0;
    size_t tile_pages = 0;
    std::fill(actual.begin(), actual.end(), unused_lane);
    for (size_t page = 0; page < active.size(); ++page) {
      uint64_t pending = active[page];
      active_count += std::popcount(pending);
      tile_pages += pending != 0;
      while (pending != 0) {
        const size_t key = page * page_keys + std::countr_zero(pending);
        std::fill_n(actual.data() + key * tile_rows, tile_rows, uint16_t(0));
        pending &= pending - 1;
      }
    }
    std::copy(actual.begin(), actual.end(), expected.begin());
    for (int lane = 0; lane < tile_rows; ++lane) {
      for (int i = offsets[first + lane]; i < offsets[first + lane + 1]; ++i) {
        expected[static_cast<size_t>(indices[i]) * tile_rows + lane] = weights[i - base];
      }
    }
    assert((reader.scatterWeights<uint16_t, tile_rows>(first, weight_offsets, weights, actual)));
    assert(actual == expected);
    if (disjoint) {
      ++stats.tiles;
      stats.selected += weights.size();
      stats.active_keys += active_count;
      stats.row_pages += row_pages;
      stats.tile_pages += tile_pages;
      stats.potential_keys += columns;
    } else {
      ++stats.offset_checks;
    }
  }
}

size_t check(const std::filesystem::path& prefix, int columns, PanelStats& panel_stats) {
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
  checkPanels(reader, indices, offsets, columns, panel_stats);
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
  std::array<size_t, 2> offsets{0, 1};
  const uint16_t weight = 17;
  std::array<uint16_t, rows> panel{};
  bitmap[0] = 1;
  assert((!reader.scatterWeights<uint16_t, 1>(0, offsets, {&weight, 1}, {panel.data(), panel.size() - 1})));
  bitmap[0] = 2;
  assert((!reader.scatterWeights<uint16_t, 1>(0, offsets, {&weight, 1}, panel)));
}
}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path root = argv[1];
  checkCausalPadding();
  size_t compared = 0;
  int matrices = 0;
  PanelStats panel_stats;
  for (int layer : {2, 15, 27}) {
    for (int keys : {2560, 3072, 3584, 4096}) {
      for (int head = 0; head < 12; ++head) {
        const auto stem = "layer" + std::to_string(layer) + "-k" + std::to_string(keys) + "-h" + std::to_string(head);
        compared += check(root / stem, keys, panel_stats);
        ++matrices;
      }
    }
  }
  assert(matrices == 144 && compared == 33857061);
  assert(panel_stats.selected == compared && panel_stats.tiles == 4608 && panel_stats.offset_checks == 432);
  std::cout << "BITMAP_DIRECT_ROWS_OK matrices=" << matrices << " indices=" << compared << '\n';
  std::cout << "BITMAP_SME_PANEL_OK tiles=" << panel_stats.tiles << " offset_checks=" << panel_stats.offset_checks
            << " selected=" << panel_stats.selected << " active_keys=" << panel_stats.active_keys
            << " row_pages=" << panel_stats.row_pages << " tile_pages=" << panel_stats.tile_pages
            << " potential_keys=" << panel_stats.potential_keys << '\n';
}
