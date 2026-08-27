// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "mllm/backends/cpu/kernels/common/causal_conv/depthwise_causal_conv.hpp"

#include "KernelTestHelper.hpp"

class CausalDepthwiseConvKernelTest : public KernelTest {
 public:
  CausalDepthwiseConvKernelTest() = default;
  ~CausalDepthwiseConvKernelTest() override = default;

  bool testHistoryFirstK3MatchesScalarReferenceBitwise(const std::vector<std::unordered_map<std::string, int32_t>>& cfgs) {
    for (const auto& cfg : cfgs) {
      if (!testOneCase(cfg)) { return false; }
    }
    return true;
  }

 private:
  static bool testOneCase(const std::unordered_map<std::string, int32_t>& cfg) {
    constexpr int kKernel = 3;
    const int batch = cfg.at("B");
    const int sequence = cfg.at("S");
    const int channels = cfg.at("C");
    const bool non_zero_history = cfg.at("non_zero_history") != 0;

    const auto element_count = static_cast<std::size_t>(batch) * sequence * channels;
    const auto state_count = static_cast<std::size_t>(batch) * channels * (kKernel - 1);
    const std::vector<float> input = makeBuffer(element_count, channels + sequence);
    const std::vector<float> weight = makeBuffer(static_cast<std::size_t>(channels) * kKernel, kKernel);
    const std::vector<float> initial_state =
        non_zero_history ? makeBuffer(state_count, 19) : std::vector<float>(state_count, 0.0F);

    auto kernel_state = initial_state;
    std::vector<float> kernel_output(element_count, 0.0F);
    mllm::cpu::causal_conv::depthwiseCausalConvHistoryFirstF32(input.data(), weight.data(), kernel_state.data(),
                                                               kernel_output.data(), batch, sequence, channels, kKernel);

    auto reference_state = initial_state;
    std::vector<float> reference_output(element_count, 0.0F);
    referenceDepthwiseCausalConvHistoryFirst(input, weight, reference_state, reference_output, batch, sequence, channels,
                                             kKernel);

    if (kernel_output != reference_output || kernel_state != reference_state) {
      std::cerr << "history-first mismatch for B=" << batch << " S=" << sequence << " C=" << channels
                << " history=" << (non_zero_history ? "non-zero" : "zero") << '\n';
      return false;
    }
    return true;
  }

  static float patternValue(std::size_t index, int salt) {
    const auto scaled = static_cast<float>((index * 37U + static_cast<unsigned>(salt) * 11U) % 251U);
    return (scaled - 125.0F) / 64.0F;
  }

  static std::vector<float> makeBuffer(std::size_t count, int salt) {
    std::vector<float> buffer(count);
    for (std::size_t index = 0; index < count; ++index) { buffer[index] = patternValue(index, salt); }
    return buffer;
  }

  static void referenceDepthwiseCausalConvHistoryFirst(const std::vector<float>& input, const std::vector<float>& weight,
                                                       std::vector<float>& state, std::vector<float>& output, int batch_size,
                                                       int sequence_length, int channels, int kernel_size) {
    const int state_width = kernel_size - 1;
    for (int batch = 0; batch < batch_size; ++batch) {
      for (int token = 0; token < sequence_length; ++token) {
        for (int channel = 0; channel < channels; ++channel) {
          const auto state_base = (static_cast<std::size_t>(batch) * channels + channel) * state_width;
          const auto element = (static_cast<std::size_t>(batch) * sequence_length + token) * channels + channel;
          const auto weight_base = static_cast<std::size_t>(channel) * kernel_size;

          float value = 0.0F;
          for (int tap = 0; tap < state_width; ++tap) { value += state[state_base + tap] * weight[weight_base + tap]; }
          value += input[element] * weight[weight_base + state_width];
          output[element] = value;

          for (int tap = 0; tap + 1 < state_width; ++tap) { state[state_base + tap] = state[state_base + tap + 1]; }
          state[state_base + state_width - 1] = input[element];
        }
      }
    }
  }
};
