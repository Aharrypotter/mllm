// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/common/linear/kai_w4a32_dispatch.hpp"

#include <cstdlib>

#if defined(__linux__)
#include <sys/auxv.h>
#endif

namespace mllm::cpu::kai_w4a32 {

namespace {

bool environmentFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

}  // namespace

bool i8mmPrefillDisabled() {
  static const bool disabled = environmentFlagEnabled("MLLM_KAI_PREFILL_I8MM_DISABLE");
  return disabled;
}

bool cpuSupportsI8mm() {
#if defined(__linux__) && defined(__aarch64__)
  constexpr unsigned long kHwcap2I8mm = 1UL << 13;
  static const bool supported = (getauxval(AT_HWCAP2) & kHwcap2I8mm) != 0;
  return supported;
#else
  return false;
#endif
}

bool shouldUseI8mmPrefill(int m) { return shouldUseI8mmPrefill(m, i8mmPrefillDisabled(), cpuSupportsI8mm()); }

}  // namespace mllm::cpu::kai_w4a32
