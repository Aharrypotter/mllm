// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

namespace mllm::cpu::kai_w4a32 {

constexpr bool shouldUseI8mmPrefill(int m, bool disabled, bool cpu_supports_i8mm) {
  return m >= 4 && !disabled && cpu_supports_i8mm;
}

[[nodiscard]] bool i8mmPrefillDisabled();

[[nodiscard]] bool cpuSupportsI8mm();

[[nodiscard]] bool shouldUseI8mmPrefill(int m);

constexpr int threadCount(int m, int requested_threads, int decode_thread_cap, int prefill_thread_cap) {
  const int cap = m == 1 ? decode_thread_cap : prefill_thread_cap;
  return cap > 0 && cap < requested_threads ? cap : requested_threads;
}

}  // namespace mllm::cpu::kai_w4a32
