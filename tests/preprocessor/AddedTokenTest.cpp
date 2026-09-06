// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Added-token matching semantics, pinned against Hugging Face `tokenizers`
// behavior observed on 2026-09-06 (see the commit message for the probe).
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/preprocessor/tokenizers/BPE.hpp"

namespace mllm::preprocessor {
namespace {

std::vector<std::wstring> texts(const std::vector<SplitSegment<std::wstring>>& segments) {
  std::vector<std::wstring> out;
  for (const auto& segment : segments) out.push_back(segment.text);
  return out;
}

}  // namespace

TEST(AddedTokenTest, LstripAndRstripAbsorbAdjacentWhitespaceIntoTheMatch) {
  Trie trie;
  trie.add(L"<L>", {.lstrip = true});
  trie.add(L"<R>", {.rstrip = true});
  // HF: 'a  <L>b' -> ['a', '  <L>', 'b']  (the id emitted is <L>'s; the spaces vanish)
  EXPECT_EQ(texts(trie.splitSegments(L"a  <L>b")), (std::vector<std::wstring>{L"a", L"<L>", L"b"}));
  // HF: 'a<R>  b' -> ['a', '<R>  ', 'b']
  EXPECT_EQ(texts(trie.splitSegments(L"a<R>  b")), (std::vector<std::wstring>{L"a", L"<R>", L"b"}));
  // HF: 'a <L> b' -> ['a', ' <L>', 'Ġ', 'b']  (the space after stays as text)
  EXPECT_EQ(texts(trie.splitSegments(L"a <L> b")), (std::vector<std::wstring>{L"a", L"<L>", L" b"}));
  EXPECT_EQ(texts(trie.splitSegments(L"a\n<L>b")), (std::vector<std::wstring>{L"a", L"<L>", L"b"}));
  const auto segments = trie.splitSegments(L"a  <L>b");
  ASSERT_EQ(segments.size(), 3u);
  EXPECT_TRUE(segments[1].is_special);
  EXPECT_FALSE(segments[0].is_special);
}

TEST(AddedTokenTest, SingleWordRequiresWordBoundaries) {
  Trie trie;
  trie.add(L"<W>", {.single_word = true});
  // HF: 'x<W>y' -> not matched; 'x <W> y' -> matched
  EXPECT_EQ(texts(trie.splitSegments(L"x<W>y")), (std::vector<std::wstring>{L"x<W>y"}));
  EXPECT_EQ(texts(trie.splitSegments(L"x <W> y")), (std::vector<std::wstring>{L"x ", L"<W>", L" y"}));
}

TEST(AddedTokenTest, ParseSpecialFalseLeavesControlTokensInTheText) {
  Trie trie;
  trie.add(L"<|im_start|>", {.control = true});
  trie.add(L"<think>", {.control = false});
  const std::wstring text = L"a<|im_start|>b<think>c";
  auto parsed = trie.splitSegments(text);
  EXPECT_EQ(texts(parsed), (std::vector<std::wstring>{L"a", L"<|im_start|>", L"b", L"<think>", L"c"}));
  auto unparsed = trie.splitSegments(text, {.parse_special = false});
  EXPECT_EQ(texts(unparsed), (std::vector<std::wstring>{L"a<|im_start|>b", L"<think>", L"c"}));
  EXPECT_FALSE(unparsed[0].is_special);
  EXPECT_TRUE(unparsed[1].is_special);
  // The old string-only API is unchanged.
  EXPECT_EQ(trie.split(text), texts(parsed));
}

TEST(AddedTokenTest, LeftmostLongestWinsOverANestedLaterCandidate) {
  // MiniCPM5 registers both "</think>" and "/think"; the earlier, longer
  // match must win. The previous automaton emitted "<" + "/think" + ">" here.
  Trie trie;
  trie.add(L"</think>");
  trie.add(L"/think");
  trie.add(L"<s>");
  EXPECT_EQ(texts(trie.splitSegments(L"a</think>\n")), (std::vector<std::wstring>{L"a", L"</think>", L"\n"}));
  EXPECT_EQ(texts(trie.splitSegments(L"x /think y")), (std::vector<std::wstring>{L"x ", L"/think", L" y"}));
  EXPECT_EQ(texts(trie.splitSegments(L"<s><s>a")), (std::vector<std::wstring>{L"<s>", L"<s>", L"a"}));
}

TEST(AddedTokenTest, Utf8TrieHasTheSameSemantics) {
  TrieUTF8 trie;
  trie.add("<L>", {.lstrip = true});
  trie.add("<|c|>", {.control = true});
  auto segments = trie.splitSegments("中  <L>文<|c|>x", {.parse_special = false});
  std::vector<std::string> got;
  for (const auto& s : segments) got.push_back(s.text);
  EXPECT_EQ(got, (std::vector<std::string>{"中", "<L>", "文<|c|>x"}));
}

TEST(AddedTokenTest, BpeLoaderRecordsEveryAddedTokenAttribute) {
  const auto dir = std::filesystem::temp_directory_path()
                   / ("mllm-added-token-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(dir);
  const auto path = dir / "tokenizer.json";
  {
    std::ofstream out(path);
    out << R"({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
               "added_tokens": [
                 {"id": 2, "content": "<s>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false},
                 {"id": 3, "content": "<L>", "special": false, "lstrip": true, "rstrip": true, "single_word": true, "normalized": true}]})";
  }
  BPE bpe;
  ASSERT_TRUE(bpe.initFromSentencePieceJson(path.string()));
  ASSERT_EQ(bpe.addedTokens().size(), 2u);
  EXPECT_EQ(bpe.addedTokens()[0].content, "<s>");
  EXPECT_TRUE(bpe.addedTokens()[0].special);
  EXPECT_EQ(bpe.addedTokens()[1].id, 3);
  EXPECT_TRUE(bpe.addedTokens()[1].lstrip);
  EXPECT_TRUE(bpe.addedTokens()[1].rstrip);
  EXPECT_TRUE(bpe.addedTokens()[1].single_word);
  EXPECT_TRUE(bpe.addedTokens()[1].normalized);
  EXPECT_EQ(bpe.controlTokens(), (std::vector<std::string>{"<s>"}));
  EXPECT_EQ(bpe._lookup_vocab(L"<L>"), 3);
  std::filesystem::remove_all(dir);
}

}  // namespace mllm::preprocessor
