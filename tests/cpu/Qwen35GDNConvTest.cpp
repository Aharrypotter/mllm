// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Focused oracle for the GDN depthwise causal convolution.
//
// The reference below is an independent scalar implementation of the frozen
// contract. It is deliberately not routed through the production kernel, so a
// vectorized fast path inside depthwiseCausalConvF32 cannot validate itself.
// Both the output and the final history are compared bitwise: an output-only
// comparison would miss a corrupted history that only shows up in the next
// chunk.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "mllm/backends/cpu/kernels/common/gdn/gated_delta_net.hpp"

namespace {

using mllm::cpu::gdn::depthwiseCausalConvF32;
using mllm::cpu::gdn::depthwiseCausalConvHistoryFirstF32;

// Deterministic index-derived fill. No RNG, so every host reproduces the same
// bytes without carrying a seed through the evidence record.
float patternValue(std::size_t index, int salt) {
  const auto scaled = static_cast<float>((index * 37U + static_cast<unsigned>(salt) * 11U) % 251U);
  return (scaled - 125.0F) / 64.0F;
}

std::vector<float> makeBuffer(std::size_t count, int salt) {
  std::vector<float> buffer(count);
  for (std::size_t index = 0; index < count; ++index) { buffer[index] = patternValue(index, salt); }
  return buffer;
}

// Independent scalar reference for the frozen contract:
// input/output [B, S, C], weight [C, K], history [B, C, K - 1] updated in place.
void referenceDepthwiseCausalConv(const std::vector<float>& input, const std::vector<float>& weight,
                                  std::vector<float>& state, std::vector<float>& output, int batch_size,
                                  int sequence_length, int channels, int kernel_size) {
  const int state_width = kernel_size - 1;
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int token = 0; token < sequence_length; ++token) {
      for (int channel = 0; channel < channels; ++channel) {
        const auto state_base = (static_cast<std::size_t>(batch) * channels + channel) * state_width;
        const auto element = (static_cast<std::size_t>(batch) * sequence_length + token) * channels + channel;
        const auto weight_base = static_cast<std::size_t>(channel) * kernel_size;

        float value = input[element] * weight[weight_base + state_width];
        for (int tap = 0; tap < state_width; ++tap) { value += state[state_base + tap] * weight[weight_base + tap]; }
        output[element] = value;

        for (int tap = 0; tap + 1 < state_width; ++tap) { state[state_base + tap] = state[state_base + tap + 1]; }
        state[state_base + state_width - 1] = input[element];
      }
    }
  }
}

void referenceDepthwiseCausalConvHistoryFirst(const std::vector<float>& input, const std::vector<float>& weight,
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

struct ConvCase {
  int batch;
  int sequence;
  int channels;
  int kernel;
  bool non_zero_history;

  std::string describe() const {
    return "B=" + std::to_string(batch) + " S=" + std::to_string(sequence) + " C=" + std::to_string(channels)
           + " K=" + std::to_string(kernel) + " history=" + (non_zero_history ? "non-zero" : "zero");
  }
};

// Runs one case through the production kernel and the reference, and requires
// bitwise agreement on both the output and the final history.
void expectBitwiseAgreement(const ConvCase& test_case) {
  const auto element_count =
      static_cast<std::size_t>(test_case.batch) * test_case.sequence * test_case.channels;
  const auto state_count =
      static_cast<std::size_t>(test_case.batch) * test_case.channels * (test_case.kernel - 1);

  const std::vector<float> input = makeBuffer(element_count, test_case.channels + test_case.sequence);
  const std::vector<float> weight =
      makeBuffer(static_cast<std::size_t>(test_case.channels) * test_case.kernel, test_case.kernel);
  const std::vector<float> initial_state =
      test_case.non_zero_history ? makeBuffer(state_count, 7) : std::vector<float>(state_count, 0.0F);

  std::vector<float> kernel_state = initial_state;
  std::vector<float> kernel_output(element_count, 0.0F);
  depthwiseCausalConvF32(input.data(), weight.data(), kernel_state.data(), kernel_output.data(), test_case.batch,
                         test_case.sequence, test_case.channels, test_case.kernel);

  std::vector<float> reference_state = initial_state;
  std::vector<float> reference_output(element_count, 0.0F);
  referenceDepthwiseCausalConv(input, weight, reference_state, reference_output, test_case.batch, test_case.sequence,
                               test_case.channels, test_case.kernel);

  ASSERT_EQ(kernel_output, reference_output) << "output mismatch for " << test_case.describe();
  ASSERT_EQ(kernel_state, reference_state) << "final history mismatch for " << test_case.describe();
}

TEST(Qwen35GDNConvTest, MatchesScalarReferenceAcrossFocusedMatrix) {
  // Channel counts below, at, and above the natural four-channel vector width,
  // including several that leave a tail.
  const int channel_values[] = {1, 2, 3, 4, 5, 7, 130};
  const int sequence_values[] = {1, 2, 16, 69, 128, 517};
  const int kernel_values[] = {2, 3, 4, 5};

  for (int batch : {1, 2}) {
    for (int sequence : sequence_values) {
      for (int channels : channel_values) {
        for (int kernel : kernel_values) {
          for (bool non_zero_history : {false, true}) {
            ASSERT_NO_FATAL_FAILURE(expectBitwiseAgreement({batch, sequence, channels, kernel, non_zero_history}));
          }
        }
      }
    }
  }
}

TEST(Qwen35GDNConvTest, MatchesScalarReferenceAtProductionChannelWidths) {
  // 6144 is the Qwen3.5-0.8B convolution width, 8192 the 4B width; both are
  // multiples of four, so they never exercise a tail on their own.
  for (int channels : {6144, 8192}) {
    for (int sequence : {1, 16, 69, 128, 517}) {
      ASSERT_NO_FATAL_FAILURE(expectBitwiseAgreement({1, sequence, channels, 4, true}));
    }
  }
}

TEST(Qwen35GDNConvTest, MatchesScalarReferenceWithChannelTailAtProductionScale) {
  // Production width minus one, two, and three channels: a full-width run plus
  // a tail of three, two, and one channel respectively.
  for (int channels : {8189, 8190, 8191, 6141}) {
    ASSERT_NO_FATAL_FAILURE(expectBitwiseAgreement({1, 69, channels, 4, true}));
  }
}

TEST(Qwen35GDNConvTest, HistoryFirstK3MatchesScalarReferenceBitwiseForLfmWidths) {
  constexpr int kKernel = 3;
  for (int batch : {1, 2}) {
    for (int sequence : {1, 2, 28, 225}) {
      for (int channels : {1, 3, 4, 5, 2045, 2048}) {
        const auto element_count = static_cast<std::size_t>(batch) * sequence * channels;
        const auto state_count = static_cast<std::size_t>(batch) * channels * (kKernel - 1);
        const std::vector<float> input = makeBuffer(element_count, channels + sequence);
        const std::vector<float> weight = makeBuffer(static_cast<std::size_t>(channels) * kKernel, kKernel);
        const std::vector<float> initial_state = makeBuffer(state_count, 19);

        auto kernel_state = initial_state;
        std::vector<float> kernel_output(element_count, 0.0F);
        depthwiseCausalConvHistoryFirstF32(input.data(), weight.data(), kernel_state.data(), kernel_output.data(), batch,
                                           sequence, channels, kKernel);

        auto reference_state = initial_state;
        std::vector<float> reference_output(element_count, 0.0F);
        referenceDepthwiseCausalConvHistoryFirst(input, weight, reference_state, reference_output, batch, sequence,
                                                  channels, kKernel);

        ASSERT_EQ(kernel_output, reference_output)
            << "history-first output mismatch for B=" << batch << " S=" << sequence << " C=" << channels;
        ASSERT_EQ(kernel_state, reference_state)
            << "history-first state mismatch for B=" << batch << " S=" << sequence << " C=" << channels;
      }
    }
  }
}

TEST(Qwen35GDNConvTest, ChunkedPartitionsMatchOneShot) {
  struct Partition {
    int channels;
    int kernel;
    std::vector<int> chunks;
  };

  const std::vector<Partition> partitions = {
      {8192, 4, {517}},                    // one-shot reference
      {8192, 4, {1, 516}},                 // prefill then continuation
      {8192, 4, {128, 128, 128, 128, 5}},  // multi-chunk
      {130, 4, {1, 15, 53}},               // tail channels across chunks
      {7, 5, {1, 1, 14}},                  // generic kernel size, odd channels
      {6144, 4, {69}},
      {6144, 4, {16, 16, 16, 21}},
  };

  for (const auto& partition : partitions) {
    int total_sequence = 0;
    for (int chunk : partition.chunks) { total_sequence += chunk; }

    const auto element_count = static_cast<std::size_t>(total_sequence) * partition.channels;
    const auto state_count = static_cast<std::size_t>(partition.channels) * (partition.kernel - 1);
    const std::vector<float> input = makeBuffer(element_count, partition.channels);
    const std::vector<float> weight =
        makeBuffer(static_cast<std::size_t>(partition.channels) * partition.kernel, partition.kernel);
    const std::vector<float> initial_state = makeBuffer(state_count, 7);

    std::vector<float> one_shot_state = initial_state;
    std::vector<float> one_shot_output(element_count, 0.0F);
    referenceDepthwiseCausalConv(input, weight, one_shot_state, one_shot_output, 1, total_sequence, partition.channels,
                                 partition.kernel);

    std::vector<float> chunked_state = initial_state;
    std::vector<float> chunked_output(element_count, 0.0F);
    int consumed = 0;
    for (int chunk : partition.chunks) {
      const auto offset = static_cast<std::size_t>(consumed) * partition.channels;
      depthwiseCausalConvF32(input.data() + offset, weight.data(), chunked_state.data(), chunked_output.data() + offset,
                             1, chunk, partition.channels, partition.kernel);
      consumed += chunk;
    }

    ASSERT_EQ(chunked_output, one_shot_output)
        << "chunked output diverged for C=" << partition.channels << " K=" << partition.kernel;
    ASSERT_EQ(chunked_state, one_shot_state)
        << "chunked history diverged for C=" << partition.channels << " K=" << partition.kernel;
  }
}

TEST(Qwen35GDNConvTest, ResetBetweenRequestsReproducesFirstRequest) {
  constexpr int kChannels = 8192;
  constexpr int kKernel = 4;
  constexpr int kSequence = 69;
  constexpr auto kElements = static_cast<std::size_t>(kSequence) * kChannels;
  constexpr auto kStateCount = static_cast<std::size_t>(kChannels) * (kKernel - 1);

  const std::vector<float> input = makeBuffer(kElements, 5);
  const std::vector<float> weight = makeBuffer(static_cast<std::size_t>(kChannels) * kKernel, kKernel);

  std::vector<float> state(kStateCount, 0.0F);
  std::vector<float> first_output(kElements, 0.0F);
  depthwiseCausalConvF32(input.data(), weight.data(), state.data(), first_output.data(), 1, kSequence, kChannels,
                         kKernel);
  const std::vector<float> first_state = state;

  // A second request that continues the history must differ, proving the
  // history is really being carried.
  std::vector<float> continued_output(kElements, 0.0F);
  depthwiseCausalConvF32(input.data(), weight.data(), state.data(), continued_output.data(), 1, kSequence, kChannels,
                         kKernel);
  ASSERT_NE(continued_output, first_output);

  // Resetting the history reproduces the first request bit for bit.
  std::fill(state.begin(), state.end(), 0.0F);
  std::vector<float> reset_output(kElements, 0.0F);
  depthwiseCausalConvF32(input.data(), weight.data(), state.data(), reset_output.data(), 1, kSequence, kChannels,
                         kKernel);

  ASSERT_EQ(reset_output, first_output);
  ASSERT_EQ(state, first_state);
}

TEST(Qwen35GDNConvTest, RejectsNullBuffersAndInvalidGeometry) {
  constexpr int kBatch = 1;
  constexpr int kSequence = 2;
  constexpr int kChannels = 4;
  constexpr int kKernel = 4;

  std::vector<float> input(static_cast<std::size_t>(kSequence) * kChannels, 0.0F);
  std::vector<float> weight(static_cast<std::size_t>(kChannels) * kKernel, 0.0F);
  std::vector<float> state(static_cast<std::size_t>(kChannels) * (kKernel - 1), 0.0F);
  std::vector<float> output(input.size(), 0.0F);

  EXPECT_THROW(depthwiseCausalConvF32(nullptr, weight.data(), state.data(), output.data(), kBatch, kSequence, kChannels,
                                      kKernel),
               std::invalid_argument);
  EXPECT_THROW(
      depthwiseCausalConvF32(input.data(), nullptr, state.data(), output.data(), kBatch, kSequence, kChannels, kKernel),
      std::invalid_argument);
  EXPECT_THROW(
      depthwiseCausalConvF32(input.data(), weight.data(), nullptr, output.data(), kBatch, kSequence, kChannels, kKernel),
      std::invalid_argument);
  EXPECT_THROW(
      depthwiseCausalConvF32(input.data(), weight.data(), state.data(), nullptr, kBatch, kSequence, kChannels, kKernel),
      std::invalid_argument);

  EXPECT_THROW(depthwiseCausalConvF32(input.data(), weight.data(), state.data(), output.data(), 0, kSequence, kChannels,
                                      kKernel),
               std::invalid_argument);
  EXPECT_THROW(
      depthwiseCausalConvF32(input.data(), weight.data(), state.data(), output.data(), kBatch, 0, kChannels, kKernel),
      std::invalid_argument);
  EXPECT_THROW(
      depthwiseCausalConvF32(input.data(), weight.data(), state.data(), output.data(), kBatch, kSequence, 0, kKernel),
      std::invalid_argument);
  // kernel_size <= 1 leaves no history and is rejected by the frozen contract.
  EXPECT_THROW(
      depthwiseCausalConvF32(input.data(), weight.data(), state.data(), output.data(), kBatch, kSequence, kChannels, 1),
      std::invalid_argument);
}

}  // namespace
