// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <string>
#include <string_view>

namespace mllm::preprocessor {

// Incrementally validates byte-level tokenizer output and emits only complete
// UTF-8 sequences. Invalid sequences and incomplete final sequences are
// replaced with U+FFFD.
class StreamingUtf8Decoder {
 public:
  std::string append(std::string_view bytes) {
    pending_.append(bytes.data(), bytes.size());
    return drain(false);
  }

  std::string finish() { return drain(true); }

  void reset() { pending_.clear(); }

 private:
  static constexpr std::string_view kReplacementCharacter = "\xEF\xBF\xBD";

  static bool isContinuationByte(unsigned char byte) { return byte >= 0x80 && byte <= 0xBF; }

  static size_t sequenceLength(unsigned char lead) {
    if (lead <= 0x7F) return 1;
    if (lead >= 0xC2 && lead <= 0xDF) return 2;
    if (lead >= 0xE0 && lead <= 0xEF) return 3;
    if (lead >= 0xF0 && lead <= 0xF4) return 4;
    return 0;
  }

  static bool isValidSecondByte(unsigned char lead, unsigned char second) {
    if (!isContinuationByte(second)) return false;
    if (lead == 0xE0) return second >= 0xA0;
    if (lead == 0xED) return second <= 0x9F;
    if (lead == 0xF0) return second >= 0x90;
    if (lead == 0xF4) return second <= 0x8F;
    return true;
  }

  std::string drain(bool flush) {
    std::string output;
    size_t offset = 0;
    while (offset < pending_.size()) {
      const auto lead = static_cast<unsigned char>(pending_[offset]);
      const size_t sequence_length = sequenceLength(lead);
      if (sequence_length == 1) {
        output.push_back(pending_[offset++]);
        continue;
      }
      if (sequence_length == 0) {
        output.append(kReplacementCharacter);
        ++offset;
        continue;
      }

      const size_t available = pending_.size() - offset;
      const size_t prefix_length = std::min(available, sequence_length);
      bool valid_prefix = true;
      for (size_t index = 1; index < prefix_length; ++index) {
        const auto byte = static_cast<unsigned char>(pending_[offset + index]);
        if ((index == 1 && !isValidSecondByte(lead, byte)) || (index > 1 && !isContinuationByte(byte))) {
          valid_prefix = false;
          break;
        }
      }
      if (!valid_prefix) {
        output.append(kReplacementCharacter);
        ++offset;
        continue;
      }
      if (available < sequence_length) {
        if (flush) {
          output.append(kReplacementCharacter);
          offset = pending_.size();
        }
        break;
      }

      output.append(pending_, offset, sequence_length);
      offset += sequence_length;
    }
    pending_.erase(0, offset);
    return output;
  }

  std::string pending_;
};

}  // namespace mllm::preprocessor
