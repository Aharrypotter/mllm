// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>

#include "mllm/backends/cpu/kernels/common/gdn/gated_delta_net.hpp"

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

}  // namespace
