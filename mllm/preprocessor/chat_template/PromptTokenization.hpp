// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>

#include "mllm/preprocessor/chat_template/ChatTemplate.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"

namespace mllm::preprocessor {

// Tokenizes origin-tagged prompt spans: template-origin spans parse the
// checkpoint's control tokens, input-origin spans do not, so a control token
// that arrived inside message content becomes ordinary text. Token merges do
// not cross span boundaries, which matches the official behavior whenever the
// template ends its literals at a control token or a newline, as the Qwen and
// MiniCPM families do. Works for both the wide-string and the UTF-8 tokenizers.
template<typename Tokenizer>
auto tokenizePromptSpans(Tokenizer& tokenizer, const std::vector<PromptSpan>& spans) {
  using Pieces = decltype(tokenizer.tokenize(std::string{}, TokenizeOptions{}));
  Pieces pieces;
  for (const auto& span : spans) {
    if (span.text.empty()) { continue; }
    auto span_pieces = tokenizer.tokenize(span.text, TokenizeOptions{.parse_special = !span.is_input});
    pieces.insert(pieces.end(), span_pieces.begin(), span_pieces.end());
  }
  return pieces;
}

inline std::string flattenPromptSpans(const std::vector<PromptSpan>& spans) {
  std::string text;
  for (const auto& span : spans) { text += span.text; }
  return text;
}

}  // namespace mllm::preprocessor
