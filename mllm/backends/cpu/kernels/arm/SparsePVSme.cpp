// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/arm/SparsePVSme.hpp"

#if defined(MLLM_ANDROID_SME_SPARSE_PV)
#include <arm_sme.h>
#include <arm_sve.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <sys/prctl.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace mllm::cpu::shadownpu {
namespace {

bool enabled(const char* name) noexcept {
  const auto* value = std::getenv(name);
  return value != nullptr && std::strcmp(value, "1") == 0;
}

__attribute__((target("+sme"), always_inline)) inline svfloat16_t interleave(
    const __fp16* first, const __fp16* second, svbool_t panel) __arm_streaming {
  return svzip1_f16(svld1_f16(panel, first), svld1_f16(panel, second));
}

__attribute__((target("+sme"))) __arm_new("za") void accumulateRows(
    const __fp16* values, const int32_t* keys, const __fp16* panel, int32_t key_count, float* output) __arm_streaming {
  static constexpr __fp16 zero_value[kSparsePvChannels] = {};
  static constexpr __fp16 zero_panel[kSparsePvRows] = {};
  const svbool_t half_panel = svptrue_pat_b16(SV_VL16);
  const svbool_t lanes = svptrue_b16();
  const svbool_t words = svptrue_b32();
  const svfloat32_t empty = svdup_f32(0);

  for (int dim = 0; dim < kSparsePvChannels; dim += 64) {
    svzero_za();
    for (int index = 0; index < key_count; index += 2) {
      const int key0 = keys[index];
      const int key1 = index + 1 < key_count ? keys[index + 1] : -1;
      const __fp16* value0 = values + static_cast<size_t>(key0) * kSparsePvChannels + dim;
      const __fp16* value1 = key1 >= 0 ? values + static_cast<size_t>(key1) * kSparsePvChannels + dim : zero_value + dim;
      const __fp16* panel0 = panel + static_cast<size_t>(key0) * kSparsePvRows;
      const __fp16* panel1 = key1 >= 0 ? panel + static_cast<size_t>(key1) * kSparsePvRows : zero_panel;
      const auto coeff = interleave(panel0, panel1, half_panel);
      svmopa_za32_f16_m(0, lanes, lanes, coeff, interleave(value0, value1, half_panel));
      svmopa_za32_f16_m(1, lanes, lanes, coeff, interleave(value0 + 16, value1 + 16, half_panel));
      svmopa_za32_f16_m(2, lanes, lanes, coeff, interleave(value0 + 32, value1 + 32, half_panel));
      svmopa_za32_f16_m(3, lanes, lanes, coeff, interleave(value0 + 48, value1 + 48, half_panel));
    }
    for (unsigned row = 0; row < kSparsePvRows; ++row) {
      float* out = output + row * kSparsePvChannels + dim;
      svst1_f32(words, out, svread_hor_za32_f32_m(empty, words, 0, row));
      svst1_f32(words, out + 16, svread_hor_za32_f32_m(empty, words, 1, row));
      svst1_f32(words, out + 32, svread_hor_za32_f32_m(empty, words, 2, row));
      svst1_f32(words, out + 48, svread_hor_za32_f32_m(empty, words, 3, row));
    }
  }
}

}  // namespace

bool useSparsePvSme() noexcept {
  constexpr unsigned long required = HWCAP2_SME | HWCAP2_SME_F16F32;
  static const bool available = enabled("MLLM_HMX_SME_PV") && (getauxval(AT_HWCAP2) & required) == required;
  if (!available) return false;
  const int length = prctl(PR_SME_GET_VL, 0L, 0L, 0L, 0L);
  return length >= 0 && (length & PR_SME_VL_LEN_MASK) == 64;
}

bool traceSparsePvSme() noexcept {
  static const bool trace = enabled("MLLM_HMX_SME_PV_TRACE");
  return trace;
}

__attribute__((target("+sme"))) __arm_locally_streaming void sparsePvRowsSme(
    const __fp16* values, const int32_t* keys, const __fp16* panel, int32_t key_count, float* output) noexcept {
  accumulateRows(values, keys, panel, key_count, output);
}

}  // namespace mllm::cpu::shadownpu
#endif
