// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <unordered_map>
#include <filesystem>
#include <utility>
#include <stdexcept>
#include <vector>

#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/StreamingUtf8Decoder.hpp"
#include "mllm/preprocessor/chat_template/PromptTokenization.hpp"
#include "mllm/models/qwen_ascend/configuration_qwen_ascend.hpp"
#include "mllm/preprocessor/chat_template/LegacyChatMl.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"

namespace mllm::models::qwen_ascend {

struct QwenAscendMessage {
  std::string prompt;
};

class QwenAscendTokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  explicit QwenAscendTokenizer(const std::string& file_path, preprocessor::ChatPreprocessorConfig chat_template = {})
      : chat_preprocessor_(std::move(chat_template), preprocessor::renderLegacyChatMlSingleTurn) {
    preprocessor::initLocal("C.UTF-8");
    preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
    for (auto& kv : bytes_2_unicode_dict_) { bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first}); }
    bpe_.initFromSentencePieceJson(file_path);
    chat_preprocessor_.setControlTokens(bpe_.controlTokens());
    // Added tokens (control tokens and markers such as <think> or <tool_call>)
    // come from tokenizer.json, so every checkpoint of the family tokenizes
    // them exactly as the official tokenizer does.
    registerAddedTokens(bpe_.addedTokens());
  }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    std::vector<std::wstring> ret;
    std::vector<std::wstring> splitted;
    ::mllm::models::qwen3::qwen3Regex(str, splitted);
    for (const auto& s : splitted) {
      auto utf_8_str = preprocessor::wideString2Utf8String(s);
      std::wstring mapped_str;
      for (unsigned char c : utf_8_str) { mapped_str.push_back(bytes_2_unicode_dict_[c]); }

      auto bpe_ts = bpe_._bpe(mapped_str);
      for (const auto& bpe_t : bpe_ts) { ret.push_back(bpe_t); }
    }
    return ret;
  }

    std::vector<std::wstring> tokenize(const std::string& str) override { return tokenize(str, {}); }

  std::vector<std::wstring> tokenize(const std::string& str, const preprocessor::TokenizeOptions& options) override {
    std::vector<std::wstring> all_tokens;
    for (const auto& segment :
         special_tokens_trie_.splitSegments(preprocessor::utf8string2WideString(str), {.parse_special = options.parse_special})) {
      if (segment.is_special) {
        all_tokens.emplace_back(segment.text);
        continue;
      }
      auto tmp_tokens = _tokenize(preprocessor::wideString2Utf8String(segment.text));
      all_tokens.insert(all_tokens.end(), tmp_tokens.begin(), tmp_tokens.end());
    }
    return all_tokens;
  }

  std::wstring _detokenize(int64_t pos_idx) override { return bpe_._lookup_inverse_vocab(pos_idx); }

  // Raw bytes of one token. Byte-level BPE splits a multi-byte character across
  // tokens, so a single token is not necessarily valid UTF-8: stream these bytes
  // through preprocessor::StreamingUtf8Decoder and print only what it returns.
  std::string detokenizeBytes(int64_t pos_idx) {
    auto str = _detokenize(pos_idx);
    std::string bytes;
    bytes.reserve(str.size());
    for (wchar_t c : str) {
      const auto it = bytes_2_unicode_dict_inverse_.find(c);
      if (it == bytes_2_unicode_dict_inverse_.end()) {
        throw std::runtime_error("Qwen Ascend tokenizer encountered an unknown byte-unicode symbol");
      }
      bytes.push_back(static_cast<char>(it->second));
    }
    return bytes;
  }

  // Wide-string view of one token. Incomplete UTF-8 tails are dropped by the
  // conversion; prefer detokenizeBytes() with a streaming decoder.
  std::wstring detokenize(int64_t pos_idx) override {
    return {mllm::preprocessor::utf8string2WideString(detokenizeBytes(pos_idx))};
  }

  // Decode full id sequence as one UTF-8 string to avoid per-token mojibake.
  std::string decode(const std::vector<int64_t>& ids) {
    std::string utf_8_str;
    for (auto id : ids) { utf_8_str += detokenizeBytes(id); }
    return utf_8_str;
  }

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override {
    std::vector<int64_t> ids;
    ids.reserve(strs.size());
    for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    Tensor ret = Tensor::empty({1, (int32_t)ids.size()}, kInt64, kCPU).setMemType(kExtraInput).setName("qwen-ascend-tokenizer-i0").alloc();
    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }
    return ret;
  }

  QwenAscendTokenizer(const std::string& file_path, const QwenAscendConfig& config, std::filesystem::path model_directory = {})
      : QwenAscendTokenizer(file_path,
                            preprocessor::ChatPreprocessorConfig{
                                .backend = config.chat_template_backend,
                                .template_options = {.model_directory = model_directory.empty()
                                                                            ? std::filesystem::path(file_path).parent_path()
                                                                            : std::move(model_directory)},
                           .control_token_policy = config.control_token_policy}) {}

  preprocessor::ChatTemplateBackend chatTemplateBackend() const noexcept { return chat_preprocessor_.backend(); }

  // Renders the single-turn runner prompt through the configured backend.
  preprocessor::ChatTemplateRequest requestFor(const QwenAscendMessage& message) const {
    return preprocessor::makeSingleTurnChatTemplateRequest(message.prompt, "", false);
  }

  std::string renderChatTemplate(const QwenAscendMessage& message) const { return chat_preprocessor_.render(requestFor(message)); }

  // Origin-tagged prompt: under control_token_policy=neutralize the official
  // template's provenance is kept so message content is tokenized without
  // special-token parsing; otherwise the flat prompt is one template span.
  std::vector<preprocessor::PromptSpan> renderPromptSpans(const QwenAscendMessage& message) const {
    if (chat_preprocessor_.controlTokenPolicy() == preprocessor::ControlTokenPolicy::Neutralize) {
      return chat_preprocessor_.renderSpans(requestFor(message));
    }
    return {{renderChatTemplate(message), false}};
  }

  std::vector<std::wstring> tokenizePrompt(const QwenAscendMessage& message) {
    return preprocessor::tokenizePromptSpans(*this, renderPromptSpans(message));
  }

  ARGenerationOutputPast convertMessage(const QwenAscendMessage& message) {
    auto sequence_str = tokenizePrompt(message);
    std::vector<int64_t> ids;
    ids.reserve(sequence_str.size());
    for (const auto& str : sequence_str) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    Tensor sequence = Tensor::empty({1, (int32_t)ids.size()}, kInt64, kCPU).setMemType(kNormal).setName("qwen-ascend-tokenizer-i0").alloc();
    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return {{"sequence", sequence}};
  }

 private:
  preprocessor::BPE bpe_;
  preprocessor::ChatPreprocessor chat_preprocessor_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
  std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
};

}  // namespace mllm::models::qwen_ascend
