// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "ShadowCorrectionPacketKernel.hpp"

namespace shadow_correction_packet {

#if defined(__hexagon__) || defined(__hexagon)
static inline bool group_can_clip(HVX_Vector first, HVX_Vector second, HVX_Vector signless, HVX_Vector limit) {
  const auto first_exceeds = Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(first, signless), limit);
  const auto second_exceeds = Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(second, signless), limit);
  const auto either = Q6_Q_or_QQ(first_exceeds, second_exceeds);
#if defined(__HEXAGON_ARCH__) && __HEXAGON_ARCH__ >= 81
  return any_predicate_byte(either);
#else
  const auto bytes = Q6_V_vand_QR(either, -1);
  auto words = Q6_V_vor_VV(bytes, Q6_V_vror_VR(bytes, 4));
  words = Q6_V_vor_VV(words, Q6_V_vror_VR(words, 8));
  words = Q6_V_vor_VV(words, Q6_V_vror_VR(words, 16));
  words = Q6_V_vor_VV(words, Q6_V_vror_VR(words, 32));
  words = Q6_V_vor_VV(words, Q6_V_vror_VR(words, 64));
  return Q6_R_vextract_VR(words, 0) != 0;
#endif
}
#endif

// A group is marked conservatively with the smaller of the two sign-specific
// clipping thresholds. Packet assembly still applies the exact signed test.
inline void multiply_with_activation_mask(const float* __restrict gate, const float* __restrict up,
                                          float* __restrict product, uint32_t* __restrict mask,
                                          uint32_t threshold) {
#if defined(__hexagon__) || defined(__hexagon)
  const HVX_Vector signless = Q6_V_vsplat_R(INT32_MAX);
  const HVX_Vector limit = Q6_V_vsplat_R(threshold - 1);
  const auto* left = reinterpret_cast<const HVX_UVector*>(gate);
  const auto* right = reinterpret_cast<const HVX_UVector*>(up);
  auto* dst = reinterpret_cast<HVX_UVector*>(product);
#endif
  for (unsigned row = 0; row < tile_rows; ++row) {
    for (unsigned word = 0; word < activation_mask_words_per_row; ++word) {
      const unsigned first_group = word * activation_groups_per_word;
      const unsigned count = activation_groups_per_row - first_group < activation_groups_per_word
                                 ? activation_groups_per_row - first_group : activation_groups_per_word;
      uint32_t marked = 0;
      for (unsigned bit = 0; bit < count; ++bit) {
        const unsigned channel = row * in_width + (first_group + bit) * activation_group_width;
#if defined(__hexagon__) || defined(__hexagon)
        const unsigned vector = channel / vector_width;
        const auto first = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(left[vector], right[vector]));
        const auto second = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(left[vector + 1], right[vector + 1]));
        dst[vector] = first;
        dst[vector + 1] = second;
        const bool found = group_can_clip(first, second, signless, limit);
#else
        bool found = false;
        for (unsigned lane = channel; lane < channel + activation_group_width; ++lane) {
          product[lane] = gate[lane] * up[lane];
          found |= (word_bits(product + lane) & UINT32_C(0x7FFFFFFF)) >= threshold;
        }
#endif
        marked |= uint32_t{found} << bit;
      }
      mask[row * activation_mask_words_per_row + word] = marked;
    }
  }
}

}  // namespace shadow_correction_packet
