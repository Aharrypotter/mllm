// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// The pre-tokenizer character classes must come from Unicode tables, not from
// the process locale: the same input has to split identically on macOS, a
// glibc container without en_US.UTF-8, and Termux.
#include <gtest/gtest.h>

#include <clocale>
#include <locale>

#include "mllm/preprocessor/tokenizers/Unicode.hpp"

namespace mllm::preprocessor {

TEST(UnicodeClassTest, LettersFollowTheGeneralCategoryNotTheLocale) {
  // Force the classic "C" locale: std::iswalpha would reject every non-ASCII letter here.
  std::locale::global(std::locale::classic());
  std::setlocale(LC_ALL, "C");
  EXPECT_TRUE(isLetter(L'a'));
  EXPECT_TRUE(isLetter(L'中'));
  EXPECT_TRUE(isLetter(L'é'));
  EXPECT_TRUE(isLetter(L'ß'));
  EXPECT_FALSE(isLetter(L'1'));
  EXPECT_FALSE(isLetter(L' '));
  EXPECT_FALSE(isLetter(L'，'));
}

TEST(UnicodeClassTest, DigitsCoverEveryNumberCategory) {
  // \p{N} is Nd, Nl, and No; std::iswdigit only accepts ASCII 0-9.
  EXPECT_TRUE(isDigit(L'7'));
  EXPECT_TRUE(isDigit(L'٣'));  // ARABIC-INDIC DIGIT THREE, Nd
  EXPECT_TRUE(isDigit(L'Ⅳ'));  // ROMAN NUMERAL FOUR, Nl
  EXPECT_TRUE(isDigit(L'①'));  // CIRCLED DIGIT ONE, No
  EXPECT_TRUE(isDigit(L'²'));  // SUPERSCRIPT TWO, No
  EXPECT_FALSE(isDigit(L'a'));
  EXPECT_FALSE(isDigit(L'一'));  // CJK numeral is a letter (Lo), as in Hugging Face
}

TEST(UnicodeClassTest, WhitespaceFollowsTheUnicodeProperty) {
  EXPECT_TRUE(isWhitespace(L' '));
  EXPECT_TRUE(isWhitespace(L'\n'));
  EXPECT_TRUE(isWhitespace(L'\t'));
  EXPECT_TRUE(isWhitespace(L' '));  // NO-BREAK SPACE
  EXPECT_TRUE(isWhitespace(L'　'));  // IDEOGRAPHIC SPACE
  EXPECT_FALSE(isWhitespace(L'x'));
  EXPECT_FALSE(isWhitespace(L'​'));  // ZERO WIDTH SPACE is Cf, not white space
}

TEST(UnicodeClassTest, MarksAndCaseFolding) {
  EXPECT_TRUE(isMark(L'́'));  // COMBINING ACUTE ACCENT
  EXPECT_FALSE(isMark(L'a'));
  EXPECT_EQ(toLower(L'A'), L'a');
  EXPECT_EQ(toLower(L'É'), L'é');
  EXPECT_EQ(toLower(L'中'), L'中');
}

TEST(UnicodeClassTest, InitLocalDoesNotChangeTheGlobalLocale) {
  std::locale::global(std::locale::classic());
  initLocal();
  EXPECT_EQ(std::locale().name(), std::locale::classic().name());
  EXPECT_TRUE(isLetter(L'中'));
}

}  // namespace mllm::preprocessor
