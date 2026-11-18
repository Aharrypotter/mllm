// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/llama_cpp_unicode/unicode.h"

namespace mllm::preprocessor {
namespace {

inline bool cptIsWhitespace(uint32_t cpt) { return unicode_cpt_flags(cpt).is_whitespace; }
inline bool cptIsWord(uint32_t cpt) {
  const auto flags = unicode_cpt_flags(cpt);
  return flags.is_letter || flags.is_number || cpt == static_cast<uint32_t>('_');
}

// Leftmost-longest matching of the trie over `text`, the match kind Hugging
// Face's AddedVocabulary uses. At each position the longest registered token
// starting there wins; an overlapping candidate that starts later never
// displaces it, so "</think>" beats a "/think" token nested inside it.
template<typename Node, typename Seq>
std::vector<std::pair<size_t, size_t>> trieMatchSpans(Node* root, const Seq& text) {
  std::vector<std::pair<size_t, size_t>> spans;
  size_t position = 0;
  while (position < text.size()) {
    Node* node = root;
    size_t longest_end = 0;
    for (size_t lookahead = position; lookahead < text.size(); ++lookahead) {
      const auto child = node->children.find(text[lookahead]);
      if (child == node->children.end()) break;
      node = child->second.get();
      if (node->is_end) longest_end = lookahead + 1;
    }
    if (longest_end > position) {
      spans.emplace_back(position, longest_end);
      position = longest_end;
    } else {
      ++position;
    }
  }
  return spans;
}

// Applies added-token attributes to the raw spans and assembles the output.
// `attr_of(span)` resolves the attributes of the matched token, `is_ws(i)` and
// `is_word(i)` classify text[i], `emit(start, end, is_special)` appends output.
template<typename AttrOf, typename IsWs, typename IsWord, typename Emit>
void assembleSegments(size_t text_size, std::vector<std::pair<size_t, size_t>> spans, const SplitOptions& options,
                      AttrOf attr_of, IsWs is_ws, IsWord is_word, Emit emit) {
  struct Match {
    size_t token_start, token_end;  // the token itself
    size_t start, end;              // including absorbed white space
  };
  std::vector<Match> matches;
  for (const auto& [start, end] : spans) {
    const AddedTokenAttr attr = attr_of(start, end);
    if (attr.control && !options.parse_special) continue;
    if (attr.single_word) {
      const bool bounded_left = start == 0 || !is_word(start - 1);
      const bool bounded_right = end >= text_size || !is_word(end);
      if (!bounded_left || !bounded_right) continue;
    }
    matches.push_back({start, end, start, end});
  }
  for (size_t i = 0; i < matches.size(); ++i) {
    const AddedTokenAttr attr = attr_of(matches[i].token_start, matches[i].token_end);
    if (attr.lstrip) {
      const size_t floor = i == 0 ? 0 : matches[i - 1].end;
      while (matches[i].start > floor && is_ws(matches[i].start - 1)) { --matches[i].start; }
    }
    if (attr.rstrip) {
      const size_t ceiling = i + 1 < matches.size() ? matches[i + 1].token_start : text_size;
      while (matches[i].end < ceiling && is_ws(matches[i].end)) { ++matches[i].end; }
    }
  }
  size_t cursor = 0;
  for (const auto& match : matches) {
    if (match.start > cursor) { emit(cursor, match.start, false); }
    emit(match.token_start, match.token_end, true);
    cursor = match.end;
  }
  if (cursor < text_size) { emit(cursor, text_size, false); }
}

}  // namespace

// ---------------------------------------------------------------- wide trie

void Trie::add(const std::wstring& word, AddedTokenAttr attr) {
  if (word.empty()) return;
  special_tokens_.insert(word);
  attrs_[word] = attr;
  TrieNode* current = root_.get();
  for (const auto& c : word) {
    if (!current->children.count(c)) { current->children[c] = std::make_unique<TrieNode>(); }
    current = current->children[c].get();
  }
  current->is_end = true;
}

void Trie::update(const std::vector<std::wstring>& words) {
  for (const auto& word : words) { add(word); }
}

std::vector<std::pair<size_t, size_t>> Trie::matchSpans(const std::wstring& text) {
  return trieMatchSpans(root_.get(), text);
}

std::vector<SplitSegment<std::wstring>> Trie::splitSegments(const std::wstring& text, const SplitOptions& options) {
  std::vector<SplitSegment<std::wstring>> result;
  assembleSegments(
      text.size(), matchSpans(text), options,
      [&](size_t start, size_t end) {
        const auto it = attrs_.find(text.substr(start, end - start));
        return it == attrs_.end() ? AddedTokenAttr{} : it->second;
      },
      [&](size_t i) { return isWhitespace(text[i]); },
      [&](size_t i) { return cptIsWord(static_cast<uint32_t>(text[i])); },
      [&](size_t start, size_t end, bool is_special) {
        result.push_back({text.substr(start, end - start), is_special});
      });
  return result;
}

std::vector<std::wstring> Trie::split(const std::wstring& text) {
  std::vector<std::wstring> result;
  for (auto& segment : splitSegments(text)) { result.push_back(std::move(segment.text)); }
  return result;
}

bool Trie::isSpecialToken(const std::wstring& token) { return special_tokens_.count(token); }

// ---------------------------------------------------------------- utf8 trie

void TrieUTF8::add(const std::string& word_utf8, AddedTokenAttr attr) {
  auto word = utf8String2Cpts(word_utf8);
  if (word.empty()) return;
  special_tokens_.insert(word);
  attrs_[word] = attr;
  TrieNode* current = root_.get();
  for (const auto& c : word) {
    if (!current->children.count(c)) { current->children[c] = std::make_unique<TrieNode>(); }
    current = current->children[c].get();
  }
  current->is_end = true;
}

void TrieUTF8::update(const std::vector<std::string>& words) {
  for (const auto& word : words) { add(word); }
}

std::vector<std::pair<size_t, size_t>> TrieUTF8::matchSpans(const cpts_string_t& text) {
  return trieMatchSpans(root_.get(), text);
}

std::vector<SplitSegment<std::string>> TrieUTF8::splitSegments(const std::string& text_utf8, const SplitOptions& options) {
  const auto text = utf8String2Cpts(text_utf8);
  std::vector<SplitSegment<std::string>> result;
  assembleSegments(
      text.size(), matchSpans(text), options,
      [&](size_t start, size_t end) {
        const auto it = attrs_.find(cpts_string_t(text.begin() + start, text.begin() + end));
        return it == attrs_.end() ? AddedTokenAttr{} : it->second;
      },
      [&](size_t i) { return cptIsWhitespace(text[i]); }, [&](size_t i) { return cptIsWord(text[i]); },
      [&](size_t start, size_t end, bool is_special) {
        result.push_back({cpts2Utf8String(cpts_string_t(text.begin() + start, text.begin() + end)), is_special});
      });
  return result;
}

std::vector<std::string> TrieUTF8::split(const std::string& text) {
  std::vector<std::string> result;
  for (auto& segment : splitSegments(text)) { result.push_back(std::move(segment.text)); }
  return result;
}

bool TrieUTF8::isSpecialToken(const std::string& token) { return special_tokens_.count(utf8String2Cpts(token)); }

// ---------------------------------------------------------------- tokenizers

void AutoTokenizer::addSpecialToken(const std::wstring& special_token) { special_tokens_trie_.add(special_token); }

void AutoTokenizer::registerAddedTokens(const std::vector<AddedToken>& tokens) {
  for (const auto& token : tokens) {
    special_tokens_trie_.add(utf8string2WideString(token.content),
                             {.control = token.special, .lstrip = token.lstrip, .rstrip = token.rstrip, .single_word = token.single_word});
  }
}

void AutoTokenizerUTF8::addSpecialToken(const std::string& special_token) { special_tokens_trie_.add(special_token); }

void AutoTokenizerUTF8::registerAddedTokens(const std::vector<AddedToken>& tokens) {
  for (const auto& token : tokens) {
    special_tokens_trie_.add(token.content,
                             {.control = token.special, .lstrip = token.lstrip, .rstrip = token.rstrip, .single_word = token.single_word});
  }
}

}  // namespace mllm::preprocessor
