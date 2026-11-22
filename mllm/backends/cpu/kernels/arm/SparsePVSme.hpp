// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#if defined(__ANDROID__) && defined(__aarch64__) && defined(__clang__) && __clang_major__ >= 19 \
    && __has_include(<arm_sme.h>)
#define MLLM_ANDROID_SME_SPARSE_PV 1

#include <cstdint>

namespace mllm::cpu::shadownpu {

inline constexpr int kSparsePvRows = 16;
inline constexpr int kSparsePvChannels = 128;

bool useSparsePvSme() noexcept;
bool traceSparsePvSme() noexcept;

// The panel is key-major, with sixteen FP16 row weights per key.
void sparsePvRowsSme(const __fp16* values, const int32_t* keys, const __fp16* panel, int32_t key_count,
                     float* output) noexcept;

}  // namespace mllm::cpu::shadownpu
#endif
