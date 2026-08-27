// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/models/lfm2/tokenization_lfm2.hpp"

TEST(Lfm2TokenizerTest, GroupsDigitsInRunsOfAtMostThree) {
  std::vector<std::wstring> pieces;
  ASSERT_TRUE(mllm::models::lfm2::tokenizerRegex("1234567", pieces));
  EXPECT_EQ(pieces, (std::vector<std::wstring>{L"123", L"456", L"7"}));
}

TEST(Lfm2TokenizerTest, GenerationPromptEndsAtThinkingTokenWithoutNewline) {
  const auto text = mllm::models::lfm2::Lfm2Message::render({.prompt = "Hello"});
  EXPECT_EQ(text.substr(text.size() - 7), "<think>");
  EXPECT_EQ(text.find("<|startoftext|>"), 0);
  EXPECT_EQ(text.find("<|im_start|>assistant\n<think>"), text.size() - 29);
}

TEST(Lfm2TokenizerTest, RendersPinnedSystemAndRawToolSchemaContract) {
  const auto text = mllm::models::lfm2::Lfm2Message::render(
      {.prompt = "Weather?", .system_prompt = "Be concise.", .tools = {R"({"type": "function"})"}});
  EXPECT_EQ(text, "<|startoftext|><|im_start|>system\nBe concise.\nList of tools: [{\"type\": \"function\"}]<|im_end|>\n"
                  "<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n<think>");
}

TEST(Lfm2TokenizerTest, StreamsUtf8AcrossTokenBoundaries) {
  mllm::models::lfm2::StreamingUtf8Decoder decoder;
  EXPECT_EQ(decoder.append("\xF0\x9F"), "");
  EXPECT_EQ(decoder.append("\x98\x80"), "\xF0\x9F\x98\x80");
  EXPECT_EQ(decoder.finish(), "");
}

TEST(Lfm2TokenizerTest, MatchesPinnedCheckpointOracleWhenProvided) {
  const char* tokenizer_path = std::getenv("MLLM_LFM2_TOKENIZER_JSON");
  if (tokenizer_path == nullptr) GTEST_SKIP() << "set MLLM_LFM2_TOKENIZER_JSON to run checkpoint oracle";
  mllm::initializeContext();
  auto tokenizer = mllm::models::lfm2::Lfm2Tokenizer(tokenizer_path);
  auto input = tokenizer.convertMessage({.prompt = "Hello"}).at("sequence");
  const std::vector<int64_t> expected = {124894, 124899, 5922, 207, 35808, 124900, 207, 124899, 63514, 207, 124901};
  ASSERT_EQ(input.shape()[1], expected.size());
  for (size_t index = 0; index < expected.size(); ++index) EXPECT_EQ(input.ptr<int64_t>()[index], expected[index]);

  auto tool_input =
      tokenizer.convertMessage({.prompt = "Weather?", .system_prompt = "Be concise.", .tools = {R"({"type": "function"})"}})
          .at("sequence");
  const std::vector<int64_t> tool_expected = {124894, 124899, 23630, 207,  4184,   55911, 318,    3120,   302, 5985,
                                              34,     66155,  5882,  6380, 496,    5545,  66212,  124900, 207, 124899,
                                              5922,   207,    97056, 39,   124900, 207,   124899, 63514,  207, 124901};
  ASSERT_EQ(tool_input.shape()[1], tool_expected.size());
  for (size_t index = 0; index < tool_expected.size(); ++index) {
    EXPECT_EQ(tool_input.ptr<int64_t>()[index], tool_expected[index]);
  }

  // "Croatia" is a vocabulary entry the merge table cannot rebuild, so it only
  // survives as one id when the checkpoint's ignore_merges flag is honoured.
  // Both other oracle strings above happen to avoid such words, which is why a
  // merge-only BPE passed them while producing wrong ids for ordinary prose.
  auto merge_unreachable = tokenizer.convertMessage({.prompt = "Croatia joined the European Union in 2013."}).at("sequence");
  const std::vector<int64_t> merge_unreachable_expected = {124894, 124899, 5922,  207,   116168, 8904,   278,
                                                           4964,   6188,   296,   229,   523,    27,     22,
                                                           124900, 207,    124899, 63514, 207,    124901};
  ASSERT_EQ(merge_unreachable.shape()[1], merge_unreachable_expected.size());
  for (size_t index = 0; index < merge_unreachable_expected.size(); ++index) {
    EXPECT_EQ(merge_unreachable.ptr<int64_t>()[index], merge_unreachable_expected[index]);
  }

  const std::string multilingual = "你好 LFM2.5!";
  const auto ordinary_tokens = tokenizer.tokenize(multilingual);
  const auto ordinary_ids = tokenizer.convert2Ids(ordinary_tokens);
  std::string reconstructed;
  for (int32_t index = 0; index < ordinary_ids.shape()[1]; ++index) {
    reconstructed += tokenizer.detokenizeBytes(ordinary_ids.ptr<int64_t>()[index]);
  }
  EXPECT_EQ(reconstructed, multilingual);
  mllm::shutdownContext();
}
