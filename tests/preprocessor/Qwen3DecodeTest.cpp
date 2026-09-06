// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Byte-level BPE can split one character across tokens. Per-token wide-string
// decoding loses those bytes; detokenizeBytes() plus StreamingUtf8Decoder
// reproduces the input exactly.
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "mllm/mllm.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"
#include "mllm/preprocessor/StreamingUtf8Decoder.hpp"

TEST(Qwen3DecodeTest, StreamingByteDecodeRoundTripsByteSplitCharacters) {
  const char* tokenizer_path = std::getenv("MLLM_QWEN3_TOKENIZER_JSON");
  if (tokenizer_path == nullptr) GTEST_SKIP() << "Set MLLM_QWEN3_TOKENIZER_JSON for the official-tokenizer oracle";
  mllm::initializeContext();
  mllm::models::qwen3::Qwen3Tokenizer tokenizer(tokenizer_path);
  const std::string text = "naïve café — 日本語テキスト 🇨🇳";
  const auto ids = tokenizer.convert2Ids(tokenizer.tokenize(text));

  mllm::preprocessor::StreamingUtf8Decoder decoder;
  std::string streamed;
  std::string per_token_wide;
  int byte_split_tokens = 0;
  for (int i = 0; i < ids.numel(); ++i) {
    const auto id = ids.ptr<int64_t>()[i];
    const auto bytes = tokenizer.detokenizeBytes(id);
    streamed += decoder.append(bytes);
    // The old path: convert each token on its own.
    const auto wide = mllm::preprocessor::wideString2Utf8String(tokenizer.detokenize(id));
    per_token_wide += wide;
    if (wide != bytes) ++byte_split_tokens;
  }
  streamed += decoder.finish();
  EXPECT_EQ(streamed, text);
  EXPECT_GT(byte_split_tokens, 0) << "the fixture must contain at least one character split across tokens";
  EXPECT_NE(per_token_wide, text) << "per-token wide decoding is expected to lose the split bytes";
}
