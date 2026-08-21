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
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "mllm/backends/cpu/kernels/arm/linear/kai.hpp"
#include "mllm/backends/cpu/ops/ParallelLinearOp.hpp"
#include "mllm/mllm.hpp"

namespace {

using KaiHelper = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk;
using KaiTile = KaiHelper::Tiles;

constexpr int kInputChannels = 2048;
constexpr KaiTile kPrefillTile = KaiTile::qai8dxp4x8_qsi4c32p8x8_4x8x32;

struct ShapeCase {
  std::string_view name;
  std::vector<int> output_channels;
  std::vector<std::string> projection_names;
};

struct Buffers {
  mllm::Tensor input;
  std::vector<float> weights;
  std::vector<mllm::Tensor> packed_weights;
  std::vector<uint8_t> workspace;
  std::vector<std::vector<float>> independent_outputs;
  std::unique_ptr<mllm::cpu::CPUParallelLinearOp> parallel_op;
  std::vector<mllm::Tensor> shared_outputs;
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
  if (name == "gate_up") { return {.name = "gate_up", .output_channels = {10752, 10752}, .projection_names = {"w1", "w3"}}; }
  if (name == "qkv") {
    return {.name = "qkv", .output_channels = {2048, 512, 512}, .projection_names = {"q_proj", "k_proj", "v_proj"}};
  }
  throw std::invalid_argument("shape must be gate_up or qkv");
}

size_t totalOutputChannels(const ShapeCase& shape) {
  size_t total = 0;
  for (const int n : shape.output_channels) { total += static_cast<size_t>(n); }
  return total;
}

Buffers makeBuffers(const ShapeCase& shape, int m, int threads) {
  KaiHelper kai;
  Buffers buffers;
  const size_t total_n = totalOutputChannels(shape);

  buffers.input = mllm::Tensor::empty({1, m, kInputChannels}, mllm::kFloat32, mllm::kCPU).alloc();
  buffers.weights.resize(total_n * kInputChannels);
  uint32_t random_state = 0x4C464D32U;
  std::generate(buffers.input.ptr<float>(), buffers.input.ptr<float>() + buffers.input.numel(),
                [&] { return deterministicValue(random_state); });
  std::generate(buffers.weights.begin(), buffers.weights.end(), [&] { return deterministicValue(random_state); });

  buffers.packed_weights.reserve(shape.output_channels.size());
  buffers.independent_outputs.reserve(shape.output_channels.size());
  auto parameters = mllm::ParameterFile::create();
  size_t row_offset = 0;
  for (size_t index = 0; index < shape.output_channels.size(); ++index) {
    const int n = shape.output_channels[index];
    const size_t packed_size_value = kai.quant_pack_rhs_size(n, kInputChannels, kPrefillTile);
    if (packed_size_value > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("packed weight exceeds Tensor dimension range");
    }
    const int packed_size = static_cast<int>(packed_size_value);
    auto packed = mllm::Tensor::empty({packed_size}, mllm::kInt8, mllm::kCPU)
                      .setMemType(mllm::kParamsNormal)
                      .setName("screen." + shape.projection_names[index] + ".weight")
                      .alloc();
    kai.quant_pack_rhs_offline(packed.ptr<uint8_t>(), buffers.weights.data() + row_offset * kInputChannels, nullptr, n,
                               kInputChannels, kPrefillTile);
    parameters->push(packed.name(), packed);
    buffers.packed_weights.push_back(std::move(packed));
    buffers.independent_outputs.emplace_back(static_cast<size_t>(m) * n);
    row_offset += static_cast<size_t>(n);
  }
  buffers.workspace.resize(kai.workspace_size(m, kInputChannels, kPrefillTile));
  std::vector<float>().swap(buffers.weights);

  mllm::aops::ParallelLinearOpOptions options{
      .in_channels = kInputChannels,
      .out_channels = shape.output_channels,
      .projection_names = shape.projection_names,
      .bias = false,
      .impl_type = mllm::aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32,
      .kai_w4a32_decode_thread_cap = 4,
      .kai_w4a32_prefill_thread_cap = 6};
  options.setThreads(threads);
  buffers.parallel_op = std::make_unique<mllm::cpu::CPUParallelLinearOp>(options);
  buffers.parallel_op->setName("screen.parallel");
  buffers.parallel_op->load(parameters);
  buffers.parallel_op->reshape({buffers.input}, buffers.shared_outputs);
  buffers.parallel_op->setup({buffers.input}, buffers.shared_outputs);
  return buffers;
}

void runIndependent(const ShapeCase& shape, int m, int threads, Buffers& buffers) {
  KaiHelper kai;
  for (size_t index = 0; index < shape.output_channels.size(); ++index) {
    kai.matmul(buffers.independent_outputs[index].data(), buffers.input.ptr<float>(),
               buffers.packed_weights[index].ptr<uint8_t>(), buffers.workspace.data(), m, kInputChannels,
               shape.output_channels[index], kPrefillTile, threads);
  }
}

void runSharedMx(const ShapeCase&, int, int, Buffers& buffers) {
  buffers.parallel_op->forward({buffers.input}, buffers.shared_outputs);
}

Comparison compareOutputs(const ShapeCase& shape, int m, const Buffers& buffers) {
  Comparison result;
  for (size_t group = 0; group < shape.output_channels.size(); ++group) {
    const size_t elements = static_cast<size_t>(m) * shape.output_channels[group];
    for (size_t index = 0; index < elements; ++index) {
      const float actual = buffers.shared_outputs[group].ptr<float>()[index];
      const float expected = buffers.independent_outputs[group][index];
      if (std::bit_cast<uint32_t>(actual) != std::bit_cast<uint32_t>(expected)) { ++result.bitwise_mismatches; }
      result.max_absolute_error = std::max(result.max_absolute_error, std::fabs(actual - expected));
    }
  }
  return result;
}

uint64_t outputHash(const ShapeCase& shape, int m, const Buffers& buffers, std::string_view variant) {
  constexpr uint64_t kOffset = 1469598103934665603ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t hash = kOffset;
  const int rows[] = {0, m / 2, m - 1};
  for (size_t group = 0; group < shape.output_channels.size(); ++group) {
    const size_t group_n = static_cast<size_t>(shape.output_channels[group]);
    const size_t columns[] = {0, group_n / 2, group_n - 1};
    for (const int row : rows) {
      for (const size_t column : columns) {
        const size_t index = static_cast<size_t>(row) * group_n + column;
        const float value = variant == "shared_mx" ? buffers.shared_outputs[group].ptr<float>()[index]
                                                   : buffers.independent_outputs[group][index];
        hash ^= std::bit_cast<uint32_t>(value);
        hash *= kPrime;
      }
    }
  }
  return hash;
}

void runVariant(std::string_view variant, const ShapeCase& shape, int m, int threads, Buffers& buffers) {
  if (variant == "independent") {
    runIndependent(shape, m, threads, buffers);
  } else if (variant == "shared_mx") {
    runSharedMx(shape, m, threads, buffers);
  } else {
    throw std::invalid_argument("unknown benchmark variant");
  }
}

template<typename Function>
double timeMicros(Function&& function) {
  const auto start = std::chrono::steady_clock::now();
  function();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - start).count();
}

void runPair(const ShapeCase& shape, int m, int threads, int repeats, Buffers& buffers) {
  constexpr std::string_view kSchedule = "ABBA-BAAB";
  for (int warmup = 0; warmup < 2; ++warmup) {
    runIndependent(shape, m, threads, buffers);
    runSharedMx(shape, m, threads, buffers);
  }

  int sample = 0;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    int position = 0;
    for (const char selector : kSchedule) {
      if (selector == '-') { continue; }
      const std::string_view variant = selector == 'A' ? "independent" : "shared_mx";
      const double latency_us = timeMicros([&] { runVariant(variant, shape, m, threads, buffers); });
      const uint64_t hash = outputHash(shape, m, buffers, variant);
      std::printf("SAMPLE pair=independent_vs_shared_mx repeat=%d position=%d sample=%d variant=%.*s "
                  "latency_us=%.3f sentinel_hash=%016llx\n",
                  repeat, position, sample, static_cast<int>(variant.size()), variant.data(), latency_us,
                  static_cast<unsigned long long>(hash));
      ++position;
      ++sample;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    mllm::initializeContext();
    if (argc < 4 || argc > 5) {
      std::fprintf(stderr, "usage: %s <gate_up|qkv> <M> <threads> [schedule_repeats]\n", argv[0]);
      return 2;
    }
    const ShapeCase shape = parseShape(argv[1]);
    const int m = parsePositiveInt(argv[2], "M");
    const int threads = parsePositiveInt(argv[3], "threads");
    const int repeats = argc == 5 ? parsePositiveInt(argv[4], "schedule_repeats") : 2;
    if (m < 4) { throw std::invalid_argument("shared Mx first-class CPU op screen requires M >= 4"); }

    std::printf("LFM2_PARALLEL_LINEAR_SHARED_MX_SCREEN_CONFIG shape=%.*s m=%d k=%d groups=%zu total_n=%zu "
                "threads=%d schedule=ABBA-BAAB repeats=%d tile=i8mm_4x8\n",
                static_cast<int>(shape.name.size()), shape.name.data(), m, kInputChannels, shape.output_channels.size(),
                totalOutputChannels(shape), threads, repeats);
    std::printf("LFM2_PARALLEL_LINEAR_SHARED_MX_SCREEN_PROVENANCE=first_class_cpu_op\n");

    Buffers buffers = makeBuffers(shape, m, threads);
    runIndependent(shape, m, threads, buffers);
    runSharedMx(shape, m, threads, buffers);
    const Comparison comparison = compareOutputs(shape, m, buffers);
    std::printf("CORRECTNESS variant=shared_mx bitwise_mismatches=%zu max_abs_error=%.9g\n", comparison.bitwise_mismatches,
                comparison.max_absolute_error);
    if (comparison.bitwise_mismatches != 0) { return 3; }

    runPair(shape, m, threads, repeats, buffers);
    std::printf("LFM2_PARALLEL_LINEAR_SHARED_MX_SCREEN_OK\n");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "LFM2_PARALLEL_LINEAR_SHARED_MX_SCREEN_ERROR %s\n", error.what());
    return 1;
  }
}
