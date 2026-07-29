// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/models/ARGeneration.hpp"

namespace {

using mllm::AnyValue;
using mllm::kCPU;
using mllm::kFloat32;
using mllm::kInt64;
using mllm::Tensor;
using mllm::models::ARGeneration;
using mllm::models::ARGenerationArgs;
using mllm::models::ARGenerationOutputPast;

class FakeGeneration final : public ARGeneration {
 public:
  explicit FakeGeneration(std::vector<int64_t> tokens) : tokens_(std::move(tokens)) {}

  ARGenerationOutputPast forward(const ARGenerationOutputPast&, const ARGenerationArgs&) override {
    auto logits = Tensor::empty({1, 1, 4}, kFloat32, kCPU).alloc();
    const auto token = tokens_[std::min(forward_count_, tokens_.size() - 1)];
    std::fill(logits.ptr<float>(), logits.ptr<float>() + 4, 0.0F);
    logits.at<float>({0, 0, static_cast<int>(token)}) = 1.0F;
    ++forward_count_;
    return {{"sequence", logits}};
  }

  void resetTokens(std::vector<int64_t> tokens) {
    tokens_ = std::move(tokens);
    forward_count_ = 0;
  }

  void addEosToken(int64_t token) { additional_eos_token_ids_.insert(token); }

 private:
  std::vector<int64_t> tokens_;
  std::size_t forward_count_ = 0;
};

Tensor makeInput(int sequence_length) {
  auto input = Tensor::empty({1, sequence_length}, kInt64, kCPU).alloc();
  std::fill(input.ptr<mllm::mllm_int64_t>(), input.ptr<mllm::mllm_int64_t>() + sequence_length, 1);
  return input;
}

std::vector<int64_t> runChat(FakeGeneration& model, int max_length, int eos_token_id = 3) {
  std::vector<int64_t> tokens;
  ARGenerationArgs args = {
      {"max_length", AnyValue(max_length)},
      {"eos_token_id", AnyValue(eos_token_id)},
  };
  for (const auto& step : model.chat({{"sequence", makeInput(3)}}, args)) { tokens.push_back(step.cur_token_id); }
  return tokens;
}

TEST(ARGenerationPerformanceTest, MaxLengthProducesCompleteStats) {
  FakeGeneration model({0, 1, 2, 0});

  EXPECT_EQ(runChat(model, 4), (std::vector<int64_t>{0, 1, 2, 0}));

  const auto stats = model.perfStats();
  EXPECT_TRUE(stats.valid);
  EXPECT_TRUE(stats.completed);
  EXPECT_EQ(stats.prefill_tokens, 3);
  EXPECT_EQ(stats.generated_tokens, 4);
  EXPECT_EQ(stats.decode_steps, 3);
  EXPECT_GE(stats.prefill_duration_us, 0);
  EXPECT_GE(stats.decode_duration_us, 0);
  EXPECT_GE(stats.ttft_duration_us, stats.prefill_duration_us);
  EXPECT_EQ(stats.total_duration_us, stats.ttft_duration_us + stats.decode_duration_us);
}

TEST(ARGenerationPerformanceTest, SingleTokenHasValidZeroDecodeStats) {
  FakeGeneration model({0});

  EXPECT_EQ(runChat(model, 1), (std::vector<int64_t>{0}));

  const auto stats = model.perfStats();
  EXPECT_TRUE(stats.valid);
  EXPECT_TRUE(stats.completed);
  EXPECT_EQ(stats.generated_tokens, 1);
  EXPECT_EQ(stats.decode_steps, 0);
  EXPECT_EQ(stats.decode_duration_us, 0);
  EXPECT_GE(stats.ttft_duration_us, stats.prefill_duration_us);
}

TEST(ARGenerationPerformanceTest, EosProducesCompleteStats) {
  FakeGeneration model({0, 3, 2});

  // The chat iterator counts the sampled EOS token in its generation stats,
  // but terminates before exposing EOS to the caller.
  EXPECT_EQ(runChat(model, 8), (std::vector<int64_t>{0}));

  const auto stats = model.perfStats();
  EXPECT_TRUE(stats.valid);
  EXPECT_TRUE(stats.completed);
  EXPECT_EQ(stats.generated_tokens, 2);
  EXPECT_EQ(stats.decode_steps, 1);
}

TEST(ARGenerationPerformanceTest, ConsecutiveChatsResetRequestStats) {
  FakeGeneration model({0, 1, 2, 0});
  EXPECT_EQ(runChat(model, 4), (std::vector<int64_t>{0, 1, 2, 0}));

  model.resetTokens({2, 1});
  EXPECT_EQ(runChat(model, 2), (std::vector<int64_t>{2, 1}));

  const auto stats = model.perfStats();
  EXPECT_TRUE(stats.valid);
  EXPECT_TRUE(stats.completed);
  EXPECT_EQ(stats.prefill_tokens, 3);
  EXPECT_EQ(stats.generated_tokens, 2);
  EXPECT_EQ(stats.decode_steps, 1);
}

TEST(ARGenerationPerformanceTest, BatchGenerateRetainsPrefillTokenCount) {
  FakeGeneration model({0, 1});
  ARGenerationArgs args = {
      {"max_length", AnyValue(2)},
      {"eos_token_id", AnyValue(3)},
  };

  const auto output = model.generate({{"sequence", makeInput(5)}}, args);

  const auto stats = model.perfStats();
  EXPECT_TRUE(stats.valid);
  EXPECT_TRUE(stats.completed);
  EXPECT_EQ(stats.prefill_tokens, 5);
  EXPECT_EQ(stats.generated_tokens, 2);
  EXPECT_EQ(stats.decode_steps, 1);

  const auto& generated = output.at("generated_sequence");
  ASSERT_EQ(generated.shape(), (Tensor::shape_t{2}));
  EXPECT_EQ(std::vector<int64_t>(generated.ptr<int64_t>(), generated.ptr<int64_t>() + generated.numel()),
            (std::vector<int64_t>{0, 1}));
}

TEST(ARGenerationPerformanceTest, AdditionalEosTokenStopsChat) {
  FakeGeneration model({0, 2, 1});
  model.addEosToken(2);

  EXPECT_EQ(runChat(model, 8, 3), (std::vector<int64_t>{0}));

  const auto stats = model.perfStats();
  EXPECT_TRUE(stats.completed);
  EXPECT_EQ(stats.generated_tokens, 2);
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  mllm::initializeContext();
  const auto result = RUN_ALL_TESTS();
  mllm::shutdownContext();
  return result;
}
