// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>

#include "mllm/backends/cpu/kernels/common/gdn/gated_delta_net.hpp"
#include "mllm/mllm.hpp"

namespace {

using mllm::cpu::gdn::depthwiseCausalConvF32;
using mllm::cpu::gdn::gatedDeltaRuleF32;

TEST(Qwen35GDNTest, CausalConvChunkingMatchesSinglePrefill) {
  constexpr int kBatch = 1;
  constexpr int kSequence = 4;
  constexpr int kChannels = 2;
  constexpr int kKernel = 3;

  const std::array<float, kBatch * kSequence * kChannels> input = {
      1.0F, 10.0F, 2.0F, 20.0F, 3.0F, 30.0F, 4.0F, 40.0F,
  };
  const std::array<float, kChannels * kKernel> weight = {
      0.25F, 0.5F, 1.0F, -0.5F, 0.25F, 2.0F,
  };
  std::array<float, kBatch * kChannels*(kKernel - 1)> full_state = {};
  std::array<float, kBatch * kChannels*(kKernel - 1)> chunked_state = {};
  std::array<float, input.size()> full_output = {};
  std::array<float, input.size()> chunked_output = {};

  depthwiseCausalConvF32(input.data(), weight.data(), full_state.data(), full_output.data(), kBatch, kSequence, kChannels,
                         kKernel);
  depthwiseCausalConvF32(input.data(), weight.data(), chunked_state.data(), chunked_output.data(), kBatch,
                         /*sequence_length=*/1, kChannels, kKernel);
  depthwiseCausalConvF32(input.data() + kChannels, weight.data(), chunked_state.data(), chunked_output.data() + kChannels,
                         kBatch, kSequence - 1, kChannels, kKernel);

  EXPECT_EQ(full_output, chunked_output);
  EXPECT_EQ(full_state, chunked_state);
  EXPECT_FLOAT_EQ(full_output[0], 1.0F);
  EXPECT_FLOAT_EQ(full_output[2], 2.5F);
  EXPECT_FLOAT_EQ(full_output[6], 6.0F);
}

TEST(Qwen35GDNTest, DeltaRuleChunkingMatchesSinglePrefill) {
  constexpr int kBatch = 1;
  constexpr int kSequence = 3;
  constexpr int kKeyHeads = 1;
  constexpr int kValueHeads = 2;
  constexpr int kKeyDim = 2;
  constexpr int kValueDim = 2;

  const std::array<float, kBatch * kSequence * kKeyHeads * kKeyDim> q = {
      1.0F, 0.0F, 0.5F, 0.5F, 0.0F, 1.0F,
  };
  const std::array<float, q.size()> k = {
      0.0F, 1.0F, 1.0F, 1.0F, 1.0F, 0.0F,
  };
  const std::array<float, kBatch * kSequence * kValueHeads * kValueDim> v = {
      1.0F, 2.0F, 3.0F, 4.0F, 2.0F, 1.0F, 4.0F, 3.0F, 3.0F, 2.0F, 1.0F, 0.5F,
  };
  const std::array<float, kBatch * kSequence * kValueHeads> a = {
      -0.5F, 0.25F, 0.1F, -0.2F, 0.3F, 0.4F,
  };
  const std::array<float, a.size()> b = {
      0.0F, 0.5F, -0.25F, 0.25F, 0.75F, -0.5F,
  };
  const std::array<float, kValueHeads> a_log = {-0.2F, 0.1F};
  const std::array<float, kValueHeads> dt_bias = {0.2F, -0.1F};

  constexpr int kStateSize = kBatch * kValueHeads * kValueDim * kKeyDim;
  std::array<float, kStateSize> full_state = {};
  std::array<float, kStateSize> chunked_state = {};
  std::array<float, v.size()> full_output = {};
  std::array<float, v.size()> chunked_output = {};

  gatedDeltaRuleF32(q.data(), k.data(), v.data(), a.data(), b.data(), a_log.data(), dt_bias.data(), full_state.data(),
                    full_output.data(), kBatch, kSequence, kKeyHeads, kValueHeads, kKeyDim, kValueDim);

  gatedDeltaRuleF32(q.data(), k.data(), v.data(), a.data(), b.data(), a_log.data(), dt_bias.data(), chunked_state.data(),
                    chunked_output.data(), kBatch, /*sequence_length=*/1, kKeyHeads, kValueHeads, kKeyDim, kValueDim);
  constexpr int kQKTokenStride = kKeyHeads * kKeyDim;
  constexpr int kValueTokenStride = kValueHeads * kValueDim;
  constexpr int kGateTokenStride = kValueHeads;
  gatedDeltaRuleF32(q.data() + kQKTokenStride, k.data() + kQKTokenStride, v.data() + kValueTokenStride,
                    a.data() + kGateTokenStride, b.data() + kGateTokenStride, a_log.data(), dt_bias.data(),
                    chunked_state.data(), chunked_output.data() + kValueTokenStride, kBatch, kSequence - 1, kKeyHeads,
                    kValueHeads, kKeyDim, kValueDim);

  for (std::size_t i = 0; i < full_output.size(); ++i) {
    EXPECT_NEAR(full_output[i], chunked_output[i], 1.0e-6F) << "output index " << i;
  }
  for (std::size_t i = 0; i < full_state.size(); ++i) {
    EXPECT_NEAR(full_state[i], chunked_state[i], 1.0e-6F) << "state index " << i;
  }
}

TEST(Qwen35GDNTest, RepeatedKeyHeadsMatchExplicitExpansion) {
  constexpr int kBatch = 1;
  constexpr int kSequence = 3;
  constexpr int kKeyHeads = 2;
  constexpr int kValueHeads = 4;
  constexpr int kKeyHeadRepeats = kValueHeads / kKeyHeads;
  constexpr int kKeyDim = 4;
  constexpr int kValueDim = 3;

  std::vector<float> q(kBatch * kSequence * kKeyHeads * kKeyDim);
  std::vector<float> k(q.size());
  std::vector<float> expanded_q(kBatch * kSequence * kValueHeads * kKeyDim);
  std::vector<float> expanded_k(expanded_q.size());
  std::vector<float> v(kBatch * kSequence * kValueHeads * kValueDim);
  std::vector<float> a(kBatch * kSequence * kValueHeads);
  std::vector<float> b(a.size());
  std::vector<float> a_log(kValueHeads);
  std::vector<float> dt_bias(kValueHeads);

  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = 0.05F * static_cast<float>(static_cast<int>(i % 13) - 6);
    k[i] = 0.04F * static_cast<float>(static_cast<int>((i * 3) % 11) - 5);
  }
  for (int token = 0; token < kSequence; ++token) {
    for (int value_head = 0; value_head < kValueHeads; ++value_head) {
      const int key_head = value_head / kKeyHeadRepeats;
      for (int dim = 0; dim < kKeyDim; ++dim) {
        const std::size_t source_index = (static_cast<std::size_t>(token) * kKeyHeads + key_head) * kKeyDim + dim;
        const std::size_t target_index = (static_cast<std::size_t>(token) * kValueHeads + value_head) * kKeyDim + dim;
        expanded_q[target_index] = q[source_index];
        expanded_k[target_index] = k[source_index];
      }
    }
  }
  for (std::size_t i = 0; i < v.size(); ++i) { v[i] = 0.03F * static_cast<float>(static_cast<int>((i * 5) % 17) - 8); }
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = 0.02F * static_cast<float>(static_cast<int>(i % 9) - 4);
    b[i] = 0.025F * static_cast<float>(static_cast<int>((i * 7) % 13) - 6);
  }
  for (int value_head = 0; value_head < kValueHeads; ++value_head) {
    a_log[value_head] = -0.2F + 0.05F * value_head;
    dt_bias[value_head] = 0.1F - 0.03F * value_head;
  }

  constexpr int kStateSize = kBatch * kValueHeads * kValueDim * kKeyDim;
  std::vector<float> grouped_state(kStateSize);
  for (std::size_t i = 0; i < grouped_state.size(); ++i) {
    grouped_state[i] = 0.01F * static_cast<float>(static_cast<int>(i % 7) - 3);
  }
  auto expanded_state = grouped_state;
  std::vector<float> grouped_output(v.size());
  std::vector<float> expanded_output(v.size());

  gatedDeltaRuleF32(q.data(), k.data(), v.data(), a.data(), b.data(), a_log.data(), dt_bias.data(), grouped_state.data(),
                    grouped_output.data(), kBatch, kSequence, kKeyHeads, kValueHeads, kKeyDim, kValueDim);
  gatedDeltaRuleF32(expanded_q.data(), expanded_k.data(), v.data(), a.data(), b.data(), a_log.data(), dt_bias.data(),
                    expanded_state.data(), expanded_output.data(), kBatch, kSequence, kValueHeads, kValueHeads, kKeyDim,
                    kValueDim);

  for (std::size_t i = 0; i < grouped_output.size(); ++i) {
    EXPECT_NEAR(grouped_output[i], expanded_output[i], 1.0e-6F) << "output index " << i;
  }
  for (std::size_t i = 0; i < grouped_state.size(); ++i) {
    EXPECT_NEAR(grouped_state[i], expanded_state[i], 1.0e-6F) << "state index " << i;
  }
}

TEST(Qwen35GDNTest, RejectsIncompatibleHeadCounts) {
  float scalar = 0.0F;
  EXPECT_THROW(gatedDeltaRuleF32(&scalar, &scalar, &scalar, &scalar, &scalar, &scalar, &scalar, &scalar, &scalar,
                                 /*batch_size=*/1, /*sequence_length=*/1,
                                 /*num_key_heads=*/2, /*num_value_heads=*/3, /*key_head_dim=*/1,
                                 /*value_head_dim=*/1),
               std::invalid_argument);
}

TEST(Qwen35GDNTest, MatchesOfficialL2NormalizationEpsilonPlacement) {
  const std::array<float, 2> q = {1.0e-4F, 0.0F};
  const std::array<float, 2> k = {1.0e-4F, 0.0F};
  const std::array<float, 1> v = {1.0F};
  const std::array<float, 1> a = {0.0F};
  const std::array<float, 1> b = {0.0F};
  const std::array<float, 1> a_log = {0.0F};
  const std::array<float, 1> dt_bias = {0.0F};
  std::array<float, 2> state = {};
  std::array<float, 1> output = {};

  gatedDeltaRuleF32(q.data(), k.data(), v.data(), a.data(), b.data(), a_log.data(), dt_bias.data(), state.data(), output.data(),
                    /*batch_size=*/1, /*sequence_length=*/1, /*num_key_heads=*/1,
                    /*num_value_heads=*/1, /*key_head_dim=*/2, /*value_head_dim=*/1);

  // Qwen3.5/FLA normalizes with rsqrt(sum(x^2) + 1e-6), then applies
  // 1/sqrt(key_head_dim) to q. Placing epsilon outside sqrt changes this
  // tiny-vector result by two orders of magnitude.
  EXPECT_NEAR(output[0], 0.0035005286F, 1.0e-8F);
}

TEST(Qwen35GDNTest, ParallelBatchValueHeadsMatchSerialBitwise) {
  constexpr int kBatch = 2;
  constexpr int kSequence = 4;
  // Qwen3.5 4B/9B GDN geometry: each normalized key head is shared by
  // two independently scheduled value-head recurrence tasks.
  constexpr int kKeyHeads = 16;
  constexpr int kValueHeads = 32;
  constexpr int kKeyDim = 128;
  constexpr int kValueDim = 128;
  constexpr int kThreadCount = 4;

  std::vector<float> q(kBatch * kSequence * kKeyHeads * kKeyDim);
  std::vector<float> k(q.size());
  std::vector<float> v(kBatch * kSequence * kValueHeads * kValueDim);
  std::vector<float> a(kBatch * kSequence * kValueHeads);
  std::vector<float> b(a.size());
  std::vector<float> a_log(kValueHeads);
  std::vector<float> dt_bias(kValueHeads);

  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = 0.007F * static_cast<float>(static_cast<int>((i * 5) % 29) - 14);
    k[i] = 0.006F * static_cast<float>(static_cast<int>((i * 11) % 31) - 15);
  }
  for (std::size_t i = 0; i < v.size(); ++i) { v[i] = 0.008F * static_cast<float>(static_cast<int>((i * 7) % 37) - 18); }
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = 0.01F * static_cast<float>(static_cast<int>((i * 13) % 19) - 9);
    b[i] = 0.009F * static_cast<float>(static_cast<int>((i * 17) % 23) - 11);
  }
  for (int value_head = 0; value_head < kValueHeads; ++value_head) {
    a_log[value_head] = -0.15F + 0.04F * static_cast<float>(value_head);
    dt_bias[value_head] = 0.08F - 0.02F * static_cast<float>(value_head);
  }

  std::vector<float> serial_state(kBatch * kValueHeads * kValueDim * kKeyDim);
  for (std::size_t i = 0; i < serial_state.size(); ++i) {
    serial_state[i] = 0.002F * static_cast<float>(static_cast<int>((i * 3) % 17) - 8);
  }
  auto parallel_state = serial_state;
  std::vector<float> serial_output(v.size());
  std::vector<float> parallel_output(v.size());

  gatedDeltaRuleF32(q.data(), k.data(), v.data(), a.data(), b.data(), a_log.data(), dt_bias.data(), serial_state.data(),
                    serial_output.data(), kBatch, kSequence, kKeyHeads, kValueHeads, kKeyDim, kValueDim,
                    /*thread_count=*/1);

  mllm::Context::instance().setCpuOpThreads(kThreadCount);
  mllm::initializeContext();
  gatedDeltaRuleF32(q.data(), k.data(), v.data(), a.data(), b.data(), a_log.data(), dt_bias.data(), parallel_state.data(),
                    parallel_output.data(), kBatch, kSequence, kKeyHeads, kValueHeads, kKeyDim, kValueDim, kThreadCount);

  for (std::size_t i = 0; i < serial_output.size(); ++i) {
    ASSERT_FLOAT_EQ(serial_output[i], parallel_output[i]) << "output index " << i;
  }
  for (std::size_t i = 0; i < serial_state.size(); ++i) {
    ASSERT_FLOAT_EQ(serial_state[i], parallel_state[i]) << "state index " << i;
  }
}

}  // namespace
