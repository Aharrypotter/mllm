// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "mllm/backends/cpu/kernels/arm/linear/kai.hpp"

namespace {

using KaiHelper = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk;
using KaiTile = KaiHelper::Tiles;

constexpr int kInputChannels = 2048;
constexpr KaiTile kDecodeTile = KaiTile::qai8dxp1x8_qsi4c32p8x8_1x8x32;
constexpr KaiTile kPrefillTile = KaiTile::qai8dxp4x8_qsi4c32p8x8_4x8x32;

struct ShapeCase {
  std::string_view name;
  std::vector<int> output_channels;
};

struct Buffers {
  std::vector<float> input;
  std::vector<float> weights;
  std::vector<std::vector<uint8_t>> separate_packed_weights;
  std::vector<uint8_t> merged_packed_weight;
  std::vector<uint8_t> workspace;
  std::vector<std::vector<float>> separate_outputs;
  std::vector<std::vector<float>> shared_outputs;
  std::vector<float> merged_output;
};

struct Comparison {
  size_t bitwise_mismatches = 0;
  float max_absolute_error = 0.0F;
};

uint32_t nextRandom(uint32_t& state) {
  state = state * 1664525U + 1013904223U;
  return state;
}

float deterministicValue(uint32_t& state) {
  const int32_t centered = static_cast<int32_t>((nextRandom(state) >> 8U) % 2001U) - 1000;
  return static_cast<float>(centered) / 4096.0F;
}

int parsePositiveInt(const char* value, const char* name) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(name) + " must be a positive integer");
  }
  return static_cast<int>(parsed);
}

ShapeCase parseShape(std::string_view name) {
  if (name == "gate_up") { return {.name = "gate_up", .output_channels = {10752, 10752}}; }
  if (name == "qkv") { return {.name = "qkv", .output_channels = {2048, 512, 512}}; }
  throw std::invalid_argument("shape must be gate_up or qkv");
}

size_t totalOutputChannels(const ShapeCase& shape) {
  return std::accumulate(shape.output_channels.begin(), shape.output_channels.end(), size_t{0});
}

Buffers makeBuffers(const ShapeCase& shape, int m, KaiTile tile) {
  KaiHelper kai;
  Buffers buffers;
  const size_t total_n = totalOutputChannels(shape);

  buffers.input.resize(static_cast<size_t>(m) * kInputChannels);
  buffers.weights.resize(total_n * kInputChannels);
  uint32_t random_state = 0x4C464D32U;
  std::generate(buffers.input.begin(), buffers.input.end(), [&] { return deterministicValue(random_state); });
  std::generate(buffers.weights.begin(), buffers.weights.end(), [&] { return deterministicValue(random_state); });

  buffers.separate_packed_weights.reserve(shape.output_channels.size());
  buffers.separate_outputs.reserve(shape.output_channels.size());
  buffers.shared_outputs.reserve(shape.output_channels.size());
  size_t row_offset = 0;
  for (const int n : shape.output_channels) {
    const size_t packed_size = kai.quant_pack_rhs_size(n, kInputChannels, tile);
    auto& packed = buffers.separate_packed_weights.emplace_back(packed_size);
    kai.quant_pack_rhs_offline(packed.data(), buffers.weights.data() + row_offset * kInputChannels, nullptr, n, kInputChannels,
                               tile);
    buffers.separate_outputs.emplace_back(static_cast<size_t>(m) * n);
    buffers.shared_outputs.emplace_back(static_cast<size_t>(m) * n);
    row_offset += static_cast<size_t>(n);
  }

  buffers.merged_packed_weight.resize(kai.quant_pack_rhs_size(static_cast<int>(total_n), kInputChannels, tile));
  kai.quant_pack_rhs_offline(buffers.merged_packed_weight.data(), buffers.weights.data(), nullptr, static_cast<int>(total_n),
                             kInputChannels, tile);
  buffers.workspace.resize(kai.workspace_size(m, kInputChannels, tile));
  buffers.merged_output.resize(static_cast<size_t>(m) * total_n);
  return buffers;
}

void runIndependent(const ShapeCase& shape, int m, int threads, KaiTile tile, Buffers& buffers) {
  KaiHelper kai;
  for (size_t index = 0; index < shape.output_channels.size(); ++index) {
    kai.matmul(buffers.separate_outputs[index].data(), buffers.input.data(), buffers.separate_packed_weights[index].data(),
               buffers.workspace.data(), m, kInputChannels, shape.output_channels[index], tile, threads);
  }
}

void runShared(const ShapeCase& shape, int m, int threads, KaiTile tile, Buffers& buffers) {
  if (m != 1) { throw std::invalid_argument("shared-input path requires M=1"); }
  std::vector<KaiHelper::SharedInputProjection> projections;
  projections.reserve(shape.output_channels.size());
  for (size_t index = 0; index < shape.output_channels.size(); ++index) {
    projections.push_back({.dst = buffers.shared_outputs[index].data(),
                           .packed_weight_bias = buffers.separate_packed_weights[index].data(),
                           .n = shape.output_channels[index]});
  }
  KaiHelper kai;
  if (!kai.matmul_shared_input_m1(buffers.input.data(), projections.data(), projections.size(), buffers.workspace.data(),
                                  kInputChannels, tile, threads)) {
    throw std::runtime_error("shared-input path rejected a valid LFM2 shape");
  }
}

void runMerged(const ShapeCase& shape, int m, int threads, KaiTile tile, Buffers& buffers) {
  KaiHelper kai;
  kai.matmul(buffers.merged_output.data(), buffers.input.data(), buffers.merged_packed_weight.data(), buffers.workspace.data(),
             m, kInputChannels, static_cast<int>(totalOutputChannels(shape)), tile, threads);
}

Comparison compareSeparate(const ShapeCase& shape, int m, const std::vector<std::vector<float>>& actual,
                           const std::vector<std::vector<float>>& expected) {
  Comparison result;
  for (size_t group = 0; group < shape.output_channels.size(); ++group) {
    const size_t elements = static_cast<size_t>(m) * shape.output_channels[group];
    for (size_t index = 0; index < elements; ++index) {
      const float lhs = actual[group][index];
      const float rhs = expected[group][index];
      if (std::bit_cast<uint32_t>(lhs) != std::bit_cast<uint32_t>(rhs)) { ++result.bitwise_mismatches; }
      result.max_absolute_error = std::max(result.max_absolute_error, std::fabs(lhs - rhs));
    }
  }
  return result;
}

Comparison compareMerged(const ShapeCase& shape, int m, const Buffers& buffers) {
  Comparison result;
  const size_t total_n = totalOutputChannels(shape);
  size_t group_offset = 0;
  for (size_t group = 0; group < shape.output_channels.size(); ++group) {
    const size_t group_n = static_cast<size_t>(shape.output_channels[group]);
    for (int row = 0; row < m; ++row) {
      for (size_t column = 0; column < group_n; ++column) {
        const float lhs = buffers.merged_output[static_cast<size_t>(row) * total_n + group_offset + column];
        const float rhs = buffers.separate_outputs[group][static_cast<size_t>(row) * group_n + column];
        if (std::bit_cast<uint32_t>(lhs) != std::bit_cast<uint32_t>(rhs)) { ++result.bitwise_mismatches; }
        result.max_absolute_error = std::max(result.max_absolute_error, std::fabs(lhs - rhs));
      }
    }
    group_offset += group_n;
  }
  return result;
}

uint64_t outputHash(const ShapeCase& shape, int m, const Buffers& buffers, std::string_view variant) {
  constexpr uint64_t kOffset = 1469598103934665603ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t hash = kOffset;
  auto mix = [&](float value) {
    hash ^= std::bit_cast<uint32_t>(value);
    hash *= kPrime;
  };
  const size_t total_n = totalOutputChannels(shape);
  const int rows[] = {0, m / 2, m - 1};
  size_t group_offset = 0;
  for (size_t group = 0; group < shape.output_channels.size(); ++group) {
    const size_t group_n = static_cast<size_t>(shape.output_channels[group]);
    const size_t columns[] = {0, group_n / 2, group_n - 1};
    for (const int row : rows) {
      for (const size_t column : columns) {
        if (variant == "merged") {
          mix(buffers.merged_output[static_cast<size_t>(row) * total_n + group_offset + column]);
        } else {
          const auto& groups = variant == "shared" ? buffers.shared_outputs : buffers.separate_outputs;
          mix(groups[group][static_cast<size_t>(row) * group_n + column]);
        }
      }
    }
    group_offset += group_n;
  }
  return hash;
}

template<typename Function>
double timeMicros(Function&& function) {
  const auto start = std::chrono::steady_clock::now();
  function();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - start).count();
}

void runVariant(std::string_view variant, const ShapeCase& shape, int m, int threads, KaiTile tile, Buffers& buffers) {
  if (variant == "independent") {
    runIndependent(shape, m, threads, tile, buffers);
  } else if (variant == "shared") {
    runShared(shape, m, threads, tile, buffers);
  } else if (variant == "merged") {
    runMerged(shape, m, threads, tile, buffers);
  } else {
    throw std::invalid_argument("unknown benchmark variant");
  }
}

void runPair(std::string_view pair_name, std::string_view baseline, std::string_view candidate, const ShapeCase& shape, int m,
             int threads, int repeats, KaiTile tile, Buffers& buffers) {
  constexpr std::string_view kSchedule = "ABBA-BAAB";
  for (int warmup = 0; warmup < 2; ++warmup) {
    runVariant(baseline, shape, m, threads, tile, buffers);
    runVariant(candidate, shape, m, threads, tile, buffers);
  }
  int sample = 0;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    int position = 0;
    for (const char selector : kSchedule) {
      if (selector == '-') { continue; }
      const std::string_view variant = selector == 'A' ? baseline : candidate;
      const double latency_us = timeMicros([&] { runVariant(variant, shape, m, threads, tile, buffers); });
      const uint64_t hash = outputHash(shape, m, buffers, variant);
      std::printf("SAMPLE pair=%.*s repeat=%d position=%d sample=%d variant=%.*s latency_us=%.3f sentinel_hash=%016llx\n",
                  static_cast<int>(pair_name.size()), pair_name.data(), repeat, position, sample,
                  static_cast<int>(variant.size()), variant.data(), latency_us, static_cast<unsigned long long>(hash));
      ++position;
      ++sample;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 5) {
      std::fprintf(stderr, "usage: %s <gate_up|qkv> <M> <threads> [schedule_repeats]\n", argv[0]);
      return 2;
    }
    const ShapeCase shape = parseShape(argv[1]);
    const int m = parsePositiveInt(argv[2], "M");
    const int threads = parsePositiveInt(argv[3], "threads");
    const int repeats = argc == 5 ? parsePositiveInt(argv[4], "schedule_repeats") : 2;
    const KaiTile tile = m == 1 ? kDecodeTile : kPrefillTile;

    std::printf("LFM2_PARALLEL_LINEAR_SCREEN_CONFIG shape=%.*s m=%d k=%d groups=%zu total_n=%zu threads=%d "
                "schedule=ABBA-BAAB repeats=%d tile=%s\n",
                static_cast<int>(shape.name.size()), shape.name.data(), m, kInputChannels, shape.output_channels.size(),
                totalOutputChannels(shape), threads, repeats, m == 1 ? "dotprod_1x8" : "i8mm_4x8");
    std::printf("LFM2_PARALLEL_LINEAR_SCREEN_PROVENANCE=replica\n");

    Buffers buffers = makeBuffers(shape, m, tile);
    runIndependent(shape, m, threads, tile, buffers);
    runMerged(shape, m, threads, tile, buffers);
    const Comparison merged_comparison = compareMerged(shape, m, buffers);
    std::printf("CORRECTNESS variant=merged bitwise_mismatches=%zu max_abs_error=%.9g\n", merged_comparison.bitwise_mismatches,
                merged_comparison.max_absolute_error);
    if (merged_comparison.bitwise_mismatches != 0) { return 3; }

    if (m == 1) {
      runShared(shape, m, threads, tile, buffers);
      const Comparison shared_comparison = compareSeparate(shape, m, buffers.shared_outputs, buffers.separate_outputs);
      std::printf("CORRECTNESS variant=shared bitwise_mismatches=%zu max_abs_error=%.9g\n",
                  shared_comparison.bitwise_mismatches, shared_comparison.max_absolute_error);
      if (shared_comparison.bitwise_mismatches != 0) { return 4; }
      runPair("shared_vs_merged", "shared", "merged", shape, m, threads, repeats, tile, buffers);
      runPair("independent_vs_shared", "independent", "shared", shape, m, threads, repeats, tile, buffers);
    } else {
      runPair("independent_vs_merged", "independent", "merged", shape, m, threads, repeats, tile, buffers);
    }

    std::printf("LFM2_PARALLEL_LINEAR_SCREEN_OK\n");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "LFM2_PARALLEL_LINEAR_SCREEN_ERROR %s\n", error.what());
    return 1;
  }
}
