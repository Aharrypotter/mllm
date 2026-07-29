// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "mllm/models/qwen3_5/tokenization_qwen3_5.hpp"

TEST(Qwen35TokenizerTest, SplitsContractionsCaseInsensitively) {
  std::vector<std::wstring> pieces;
  ASSERT_TRUE(mllm::models::qwen3_5::qwen3_5Regex("I'M sure", pieces));

  const std::vector<std::wstring> expected = {L"I", L"'M", L" sure"};
  EXPECT_EQ(pieces, expected);
}

TEST(Qwen35TokenizerTest, PreservesRegexWhitespaceBacktracking) {
  std::vector<std::wstring> pieces;
  ASSERT_TRUE(mllm::models::qwen3_5::qwen3_5Regex("a  b\n  c", pieces));

  const std::vector<std::wstring> expected = {
      L"a", L" ", L" b", L"\n", L" ", L" c",
  };
  EXPECT_EQ(pieces, expected);
}

TEST(Qwen35TokenizerTest, StreamsUtf8SplitAcrossTokens) {
  mllm::models::qwen3_5::Qwen3_5StreamingUtf8Decoder decoder;

  EXPECT_EQ(decoder.append("\xC2"), "");
  EXPECT_EQ(decoder.append("\xA1"), "\xC2\xA1");
  EXPECT_EQ(decoder.finish(), "");
}

TEST(Qwen35TokenizerTest, ReplacesMalformedAndIncompleteUtf8) {
  mllm::models::qwen3_5::Qwen3_5StreamingUtf8Decoder decoder;

  EXPECT_EQ(decoder.append("\xE2("), "\xEF\xBF\xBD(");
  EXPECT_EQ(decoder.append("\xF0\x9F"), "");
  EXPECT_EQ(decoder.finish(), "\xEF\xBF\xBD");
}
