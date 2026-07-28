// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cwctype>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"

namespace mllm::models::qwen3_5 {

// Reuse the Qwen3 regex pattern — same BPE tokenization scheme.
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
inline bool qwen3_5TokenizerMatchPattern(const std::wstring& str, size_t& pos, std::wstring& matched) {
  if (pos >= str.size()) return false;

  // 1. Match contractions: "'s|'t|'re|'ve|'m|'ll|'d"
  static const std::wstring contractions[] = {L"'s", L"'t", L"'re", L"'ve", L"'m", L"'ll", L"'d"};
  for (const auto& contraction : contractions) {
    bool matches = pos + contraction.size() <= str.size();
    for (size_t index = 0; matches && index < contraction.size(); ++index) {
      matches = std::towlower(str[pos + index]) == contraction[index];
    }
    if (matches) {
      matched = str.substr(pos, contraction.size());
      pos += contraction.size();
      return true;
    }
  }

  // 2. Match [^\r\n\p{L}\p{N}]?\p{L}+
  {
    size_t original_pos = pos;
    bool has_prefix = false;
    matched.clear();

    if (!preprocessor::isLetter(str[pos]) && !preprocessor::isDigit(str[pos]) && str[pos] != L'\r' && str[pos] != L'\n') {
      matched += str[pos];
      ++pos;
      has_prefix = true;
    }

    if (pos < str.size() && preprocessor::isLetter(str[pos])) {
      do {
        matched += str[pos];
        ++pos;
      } while (pos < str.size() && preprocessor::isLetter(str[pos]));
      return true;
    } else {
      if (has_prefix) {
        pos = original_pos;
        matched.clear();
      }
    }
  }

  // 3. Match \p{N}
  if (preprocessor::isDigit(str[pos])) {
    matched = str.substr(pos, 1);
    ++pos;
    return true;
  }

  // 4. Match ?[^\s\p{L}\p{N}]+[\r\n]*
  {
    size_t original_pos = pos;
    matched.clear();
    size_t start = pos;

    if (str[pos] == L' ') { ++pos; }

    if (pos < str.size() && !std::iswspace(str[pos]) && !preprocessor::isLetter(str[pos]) && !preprocessor::isDigit(str[pos])) {
      do {
        ++pos;
      } while (pos < str.size() && !std::iswspace(str[pos]) && !preprocessor::isLetter(str[pos])
               && !preprocessor::isDigit(str[pos]));

      matched = str.substr(start, pos - start);

      while (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) {
        matched += str[pos];
        ++pos;
      }
      return true;
    } else {
      pos = original_pos;
    }
  }

  // 5. Match \s*[\r\n]+
  {
    size_t start = pos;
    size_t scan = pos;
    size_t last_line_break = std::wstring::npos;
    while (scan < str.size() && std::iswspace(str[scan])) {
      if (str[scan] == L'\r' || str[scan] == L'\n') { last_line_break = scan + 1; }
      ++scan;
    }
    if (last_line_break != std::wstring::npos) {
      pos = last_line_break;
      matched = str.substr(start, last_line_break - start);
      return true;
    }
  }

  // 6. Match \s+(?!\S)
  if (std::iswspace(str[pos])) {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    if (pos >= str.size()) {
      matched = str.substr(start, pos - start);
      return true;
    }
    // When followed by a non-space, regex backtracking leaves the final
    // whitespace character for the next alternative.
    if (pos - start > 1) {
      --pos;
      matched = str.substr(start, pos - start);
      return true;
    }
    pos = start;
  }

  // 7. Match remaining whitespace
  if (std::iswspace(str[pos])) {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    matched = str.substr(start, pos - start);
    return true;
  }

  return false;
}

inline bool qwen3_5Regex(const std::string& str, std::vector<std::wstring>& splitted) {
  auto w_string = preprocessor::utf8string2WideString(str);
  size_t pos = 0;
  while (pos < w_string.size()) {
    std::wstring matched;
    if (qwen3_5TokenizerMatchPattern(w_string, pos, matched)) {
      splitted.push_back(matched);
    } else {
      splitted.push_back(w_string.substr(pos, 1));
      ++pos;
    }
  }
  return true;
}

struct Qwen3_5Message {
  std::string prompt;
  static inline std::string message_template =
      "<|im_start|>user\n{{{prompt}}}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
};

class Qwen3_5Tokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  explicit Qwen3_5Tokenizer(const std::string& file_path) {
    preprocessor::initLocal();
    preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
    for (auto& kv : bytes_2_unicode_dict_) { bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first}); }
    bpe_.initFromSentencePieceJson(file_path);
    // Qwen3.5 special tokens
    special_tokens_trie_.add(L"<|endoftext|>");
    special_tokens_trie_.add(L"<|im_start|>");
    special_tokens_trie_.add(L"<|im_end|>");
    special_tokens_trie_.add(L"<|object_ref_start|>");
    special_tokens_trie_.add(L"<|object_ref_end|>");
    special_tokens_trie_.add(L"<|box_start|>");
    special_tokens_trie_.add(L"<|box_end|>");
    special_tokens_trie_.add(L"<|quad_start|>");
    special_tokens_trie_.add(L"<|quad_end|>");
    special_tokens_trie_.add(L"<|vision_start|>");
    special_tokens_trie_.add(L"<|vision_end|>");
    special_tokens_trie_.add(L"<|vision_pad|>");
    special_tokens_trie_.add(L"<|image_pad|>");
    special_tokens_trie_.add(L"<|video_pad|>");
    special_tokens_trie_.add(L"<|fim_prefix|>");
    special_tokens_trie_.add(L"<|fim_middle|>");
    special_tokens_trie_.add(L"<|fim_suffix|>");
    special_tokens_trie_.add(L"<|fim_pad|>");
    special_tokens_trie_.add(L"<|repo_name|>");
    special_tokens_trie_.add(L"<|file_sep|>");
    special_tokens_trie_.add(L"<think>");
    special_tokens_trie_.add(L"</think>");
    special_tokens_trie_.add(L"<tool_call>");
    special_tokens_trie_.add(L"</tool_call>");
    special_tokens_trie_.add(L"<tool_response>");
    special_tokens_trie_.add(L"</tool_response>");
  }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    std::vector<std::wstring> ret;
    std::vector<std::wstring> splitted;
    qwen3_5Regex(str, splitted);
    for (const auto& s : splitted) {
      auto utf_8_str = preprocessor::wideString2Utf8String(s);
      std::wstring mapped_str;
      for (unsigned char c : utf_8_str) { mapped_str.push_back(bytes_2_unicode_dict_[c]); }
      auto bpe_ts = bpe_._bpe(mapped_str);
      for (const auto& bpe_t : bpe_ts) { ret.push_back(bpe_t); }
    }
    return ret;
  }

  std::vector<std::wstring> tokenize(const std::string& str) override {
    auto tokens = special_tokens_trie_.split(preprocessor::utf8string2WideString(str));
    std::vector<std::wstring> all_tokens;
    for (const auto& token : tokens) {
      if (special_tokens_trie_.isSpecialToken(token)) {
        all_tokens.emplace_back(token);
        continue;
      }
      auto tmp_tokens = _tokenize(preprocessor::wideString2Utf8String(token));
      all_tokens.insert(all_tokens.end(), tmp_tokens.begin(), tmp_tokens.end());
    }
    return all_tokens;
  }

  std::wstring _detokenize(int64_t pos_idx) override { return bpe_._lookup_inverse_vocab(pos_idx); }

  std::wstring detokenize(int64_t pos_idx) override {
    auto str = _detokenize(pos_idx);
    std::string utf_8_str;
    for (wchar_t c : str) {
      const auto it = bytes_2_unicode_dict_inverse_.find(c);
      if (it == bytes_2_unicode_dict_inverse_.end()) {
        throw std::runtime_error("Qwen3.5 tokenizer encountered an unknown byte-unicode symbol");
      }
      utf_8_str.push_back(static_cast<char>(it->second));
    }
    return {mllm::preprocessor::utf8string2WideString(utf_8_str)};
  }

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override {
    std::vector<int64_t> ids;
    ids.reserve(strs.size());
    for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }
    Tensor ret = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                     .setMemType(kExtraInput)
                     .setName("qwen3_5-tokenizer-i0")
                     .alloc();
    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }
    return ret;
  }

  ARGenerationOutputPast convertMessage(const Qwen3_5Message& message) {
    auto applied_string = Qwen3_5Message::message_template;
    static constexpr char kPromptPlaceholder[] = "{{{prompt}}}";
    size_t pos = applied_string.find(kPromptPlaceholder);
    if (pos == std::string::npos) { throw std::runtime_error("Qwen3.5 message template is missing the prompt placeholder"); }
    applied_string.replace(pos, sizeof(kPromptPlaceholder) - 1, message.prompt);

    auto sequence_str = tokenize(applied_string);
    std::vector<int64_t> ids;
    ids.reserve(sequence_str.size());
    for (const auto& str : sequence_str) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    Tensor sequence = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                          .setMemType(kNormal)
                          .setName("qwen3_5-tokenizer-i0")
                          .alloc();
    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return {
        {"sequence", sequence},
    };
  }

 private:
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
  std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
};

}  // namespace mllm::models::qwen3_5
