// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cwctype>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/StreamingUtf8Decoder.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"

namespace mllm::models::lfm2 {

using StreamingUtf8Decoder = preprocessor::StreamingUtf8Decoder;

// Implements the checkpoint regex. The material difference from the older
// Qwen pattern is the numeric branch: LFM2 groups one to three digits.
inline bool tokenizerMatch(const std::wstring& input, size_t& pos, std::wstring& matched) {
  if (pos >= input.size()) return false;
  static const std::wstring contractions[] = {L"'s", L"'t", L"'re", L"'ve", L"'m", L"'ll", L"'d"};
  for (const auto& contraction : contractions) {
    bool match = pos + contraction.size() <= input.size();
    for (size_t index = 0; match && index < contraction.size(); ++index) {
      match = std::towlower(input[pos + index]) == contraction[index];
    }
    if (match) {
      matched = input.substr(pos, contraction.size());
      pos += contraction.size();
      return true;
    }
  }

  {
    const auto original = pos;
    const auto start = pos;
    if (!preprocessor::isLetter(input[pos]) && !preprocessor::isDigit(input[pos]) && input[pos] != L'\r'
        && input[pos] != L'\n') {
      ++pos;
    }
    if (pos < input.size() && preprocessor::isLetter(input[pos])) {
      while (pos < input.size() && preprocessor::isLetter(input[pos])) ++pos;
      matched = input.substr(start, pos - start);
      return true;
    }
    pos = original;
  }

  if (preprocessor::isDigit(input[pos])) {
    const auto start = pos;
    while (pos < input.size() && pos - start < 3 && preprocessor::isDigit(input[pos])) ++pos;
    matched = input.substr(start, pos - start);
    return true;
  }

  {
    const auto original = pos;
    const auto start = pos;
    if (input[pos] == L' ') ++pos;
    if (pos < input.size() && !std::iswspace(input[pos]) && !preprocessor::isLetter(input[pos])
        && !preprocessor::isDigit(input[pos])) {
      while (pos < input.size() && !std::iswspace(input[pos]) && !preprocessor::isLetter(input[pos])
             && !preprocessor::isDigit(input[pos])) {
        ++pos;
      }
      while (pos < input.size() && (input[pos] == L'\r' || input[pos] == L'\n')) ++pos;
      matched = input.substr(start, pos - start);
      return true;
    }
    pos = original;
  }

  {
    const auto start = pos;
    auto scan = pos;
    size_t last_break = std::wstring::npos;
    while (scan < input.size() && std::iswspace(input[scan])) {
      if (input[scan] == L'\r' || input[scan] == L'\n') last_break = scan + 1;
      ++scan;
    }
    if (last_break != std::wstring::npos) {
      pos = last_break;
      matched = input.substr(start, pos - start);
      return true;
    }
  }

  if (std::iswspace(input[pos])) {
    const auto start = pos;
    while (pos < input.size() && std::iswspace(input[pos])) ++pos;
    if (pos >= input.size()) {
      matched = input.substr(start, pos - start);
      return true;
    }
    if (pos - start > 1) {
      --pos;
      matched = input.substr(start, pos - start);
      return true;
    }
    pos = start;
  }
  if (std::iswspace(input[pos])) {
    const auto start = pos;
    while (pos < input.size() && std::iswspace(input[pos])) ++pos;
    matched = input.substr(start, pos - start);
    return true;
  }
  return false;
}

inline bool tokenizerRegex(const std::string& input, std::vector<std::wstring>& pieces) {
  const auto wide = preprocessor::utf8string2WideString(input);
  size_t pos = 0;
  while (pos < wide.size()) {
    std::wstring matched;
    if (tokenizerMatch(wide, pos, matched)) {
      pieces.push_back(matched);
    } else {
      pieces.push_back(wide.substr(pos++, 1));
    }
  }
  return true;
}

struct Lfm2Message {
  std::string prompt;
  std::string system_prompt;
  // Each item is a raw JSON tool schema. The pinned template accepts string
  // tools verbatim, which keeps key order and JSON bytes caller-controlled.
  std::vector<std::string> tools;

  static auto render(const Lfm2Message& message) -> std::string {
    std::string rendered = "<|startoftext|>";
    if (!message.system_prompt.empty() || !message.tools.empty()) {
      rendered += "<|im_start|>system\n";
      rendered += message.system_prompt;
      if (!message.tools.empty()) {
        if (!message.system_prompt.empty()) rendered += '\n';
        rendered += "List of tools: [";
        for (size_t index = 0; index < message.tools.size(); ++index) {
          if (index != 0) rendered += ", ";
          rendered += message.tools[index];
        }
        rendered += ']';
      }
      rendered += "<|im_end|>\n";
    }
    rendered += "<|im_start|>user\n" + message.prompt + "<|im_end|>\n<|im_start|>assistant\n<think>";
    return rendered;
  }
};

class Lfm2Tokenizer final : public preprocessor::AutoTokenizer {
 public:
  explicit Lfm2Tokenizer(const std::string& tokenizer_json) {
    preprocessor::initLocal();
    preprocessor::makeBytes2UnicodeMap(bytes_to_unicode_);
    for (const auto& [byte, codepoint] : bytes_to_unicode_) unicode_to_bytes_.insert({codepoint, byte});
    bpe_.initFromSentencePieceJson(tokenizer_json);
    for (const auto* token : {L"<|pad|>", L"<|startoftext|>", L"<|endoftext|>", L"<|fim_pre|>", L"<|fim_mid|>", L"<|fim_suf|>",
                              L"<|im_start|>", L"<|im_end|>", L"<think>", L"</think>", L"<|tool_list_start|>",
                              L"<|tool_list_end|>", L"<|tool_call_start|>", L"<|tool_call_end|>"}) {
      special_tokens_trie_.add(token);
    }
  }

  std::vector<std::wstring> _tokenize(const std::string& input) override {
    std::vector<std::wstring> pieces;
    tokenizerRegex(input, pieces);
    std::vector<std::wstring> tokens;
    for (const auto& piece : pieces) {
      std::wstring mapped;
      for (const auto byte : preprocessor::wideString2Utf8String(piece)) {
        mapped.push_back(bytes_to_unicode_.at(static_cast<unsigned char>(byte)));
      }
      const auto bpe_tokens = bpe_._bpe(mapped);
      tokens.insert(tokens.end(), bpe_tokens.begin(), bpe_tokens.end());
    }
    return tokens;
  }

  std::vector<std::wstring> tokenize(const std::string& input) override {
    const auto pieces = special_tokens_trie_.split(preprocessor::utf8string2WideString(input));
    std::vector<std::wstring> tokens;
    for (const auto& piece : pieces) {
      if (special_tokens_trie_.isSpecialToken(piece)) {
        tokens.push_back(piece);
      } else {
        const auto normal = _tokenize(preprocessor::wideString2Utf8String(piece));
        tokens.insert(tokens.end(), normal.begin(), normal.end());
      }
    }
    return tokens;
  }

  std::wstring _detokenize(int64_t id) override { return bpe_._lookup_inverse_vocab(id); }
  std::string detokenizeBytes(int64_t id) {
    const auto token = _detokenize(id);
    std::string bytes;
    for (const auto codepoint : token) {
      const auto found = unicode_to_bytes_.find(codepoint);
      if (found == unicode_to_bytes_.end()) throw std::runtime_error("LFM2 tokenizer encountered unknown byte symbol");
      bytes.push_back(static_cast<char>(found->second));
    }
    return bytes;
  }
  std::wstring detokenize(int64_t id) override { return preprocessor::utf8string2WideString(detokenizeBytes(id)); }

  Tensor convert2Ids(const std::vector<std::wstring>& tokens) override { return idsTensor(tokens, kExtraInput); }
  ARGenerationOutputPast convertMessage(const Lfm2Message& message) {
    const auto rendered = Lfm2Message::render(message);
    return {{"sequence", idsTensor(tokenize(rendered), kNormal)}};
  }

 private:
  Tensor idsTensor(const std::vector<std::wstring>& tokens, TensorMemTypes mem_type) {
    auto result = Tensor::empty({1, static_cast<int32_t>(tokens.size())}, kInt64, kCPU)
                      .setMemType(mem_type)
                      .setName("lfm2-tokenizer-i0")
                      .alloc();
    auto* ids = result.ptr<int64_t>();
    for (size_t index = 0; index < tokens.size(); ++index) ids[index] = bpe_._lookup_vocab(tokens[index]);
    return result;
  }
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_to_unicode_;
  std::unordered_map<wchar_t, std::wint_t> unicode_to_bytes_;
};

}  // namespace mllm::models::lfm2
