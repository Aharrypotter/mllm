// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cwchar>
#include <unordered_map>
#include <string>
#include <cstdint>

#include "mllm/preprocessor/tokenizers/llama_cpp_unicode/unicode.h"
#include <locale>
#include <cwctype>
#include <iostream>

#include "mllm/utils/Common.hpp"

namespace mllm::preprocessor {

inline std::wint_t ord(const wchar_t* str) { return static_cast<std::wint_t>(*str); }

inline wchar_t chr(std::wint_t value) { return static_cast<wchar_t>(value); }

// some OS has no en_US.UTF-8 but has C.UTF-8.
// Historically this set the process-global locale so that std::iswalpha and
// friends classified non-ASCII characters. Classification now comes from the
// vendored Unicode tables below and no longer depends on any locale. The
// function only prepares std::wcout for callers that print wide strings, and it
// never touches the global locale.
inline void initLocal(const std::string& local_name = "en_US.UTF-8") {
  for (const char* candidate : {local_name.c_str(), "C.UTF-8", "en_US.UTF-8"}) {
    try {
      std::wcout.imbue(std::locale(candidate));
      return;
    } catch (const std::exception&) {}
  }
}

// Unicode general-category predicates backed by the vendored llama.cpp tables.
// They match the character classes used by Hugging Face pre-tokenizer regexes:
// \p{L}, \p{N} (Nd, Nl, and No, not only ASCII digits), \p{M}, and \s.
inline bool isLetter(wchar_t c) { return unicode_cpt_flags(static_cast<uint32_t>(c)).is_letter; }

inline bool isDigit(wchar_t c) { return unicode_cpt_flags(static_cast<uint32_t>(c)).is_number; }

inline bool isMark(wchar_t c) { return unicode_cpt_flags(static_cast<uint32_t>(c)).is_accent_mark; }

inline bool isWhitespace(wchar_t c) { return unicode_cpt_flags(static_cast<uint32_t>(c)).is_whitespace; }

inline wchar_t toLower(wchar_t c) { return static_cast<wchar_t>(unicode_tolower(static_cast<uint32_t>(c))); }

std::string wideString2Utf8String(const std::wstring& wstr);

std::wstring utf8string2WideString(const std::string& str);

// This function is used for GPT2 Like Tokenizers
//
// same with gpt2.bytes_to_unicode
//
// same with qwen2.bytes_to_unicode
//
/*
Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to
whitespace/control characters the bpe code barfs on.

The reversible bpe codes work on unicode strings. This means you need a large # of unicode
characters in your vocab if you want to avoid UNKs. When you're at something like a 10B token
dataset you end up needing around 5K for decent coverage. This is a significant percentage of
your normal, say, 32K bpe vocab. To avoid that, we want lookup tables between utf-8 bytes and
unicode strings.
*/
void makeBytes2UnicodeMap(std::unordered_map<std::wint_t, wchar_t>& dict);

}  // namespace mllm::preprocessor
