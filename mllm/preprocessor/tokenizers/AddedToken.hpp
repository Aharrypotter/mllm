// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <string>

namespace mllm::preprocessor {

// One entry of a Hugging Face tokenizer.json `added_tokens` array. Added tokens
// are matched atomically before pre-tokenization regardless of `special`;
// `special` marks the checkpoint's control tokens (turn boundaries, vision
// markers) and is what a request-boundary guard or a parse_special gate acts on.
struct AddedToken {
  int64_t id = -1;
  std::string content;
  bool special = false;
  // Absorb white space on the left / right of the token into the match; the
  // white space is dropped and only the token id is emitted.
  bool lstrip = false;
  bool rstrip = false;
  // Match only when the token is not embedded inside a word.
  bool single_word = false;
  // Whether the match runs on normalized text. mllm applies no normalizer, so
  // the flag is recorded but has no effect.
  bool normalized = false;
};

// Coarse per-token attribute, the subset of llama.cpp's token attributes that
// mllm needs today.
enum class TokenAttr : uint8_t { kNormal, kUserDefined, kControl };

// Match-time attributes kept by the special-token tries.
struct AddedTokenAttr {
  bool control = false;
  bool lstrip = false;
  bool rstrip = false;
  bool single_word = false;
};

}  // namespace mllm::preprocessor
