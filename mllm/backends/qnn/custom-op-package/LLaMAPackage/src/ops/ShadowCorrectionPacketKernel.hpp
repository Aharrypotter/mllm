#pragma once

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include "ShadowCorrectionPacketLayout.hpp"
#include <string.h>

#if defined(__hexagon__) || defined(__hexagon)
#include <hexagon_protos.h>
#include <hexagon_types.h>
#endif

namespace shadow_correction_packet {

constexpr int vector_width = 32;

enum class code_encoding : uint16_t { signed_fixed = 0, offset_binary = 0x8000 };

static inline uint32_t word_bits(const float *value) {
  uint32_t bits;
  memcpy(&bits, value, sizeof(bits));
  return bits;
}

#if defined(__hexagon__) || defined(__hexagon)
static inline uint32_t clipping_mask(const float *values, uint32_t positive, uint32_t negative) {
  // One uniquely weighted bit per word reduces the predicate to a scalar lane mask.
  static const uint32_t lane_bits[vector_width] __attribute__((aligned(128))) = {
      1u << 0, 1u << 1, 1u << 2, 1u << 3, 1u << 4, 1u << 5, 1u << 6, 1u << 7,
      1u << 8, 1u << 9, 1u << 10, 1u << 11, 1u << 12, 1u << 13, 1u << 14, 1u << 15,
      1u << 16, 1u << 17, 1u << 18, 1u << 19, 1u << 20, 1u << 21, 1u << 22, 1u << 23,
      1u << 24, 1u << 25, 1u << 26, 1u << 27, 1u << 28, 1u << 29, 1u << 30, 1u << 31,
  };
  HVX_Vector bits, lane_values;
  memcpy(&bits, values, sizeof(bits));
  memcpy(&lane_values, lane_bits, sizeof(lane_values));
  const HVX_Vector zero = Q6_V_vsplat_R(0);
  const HVX_Vector magnitude = Q6_V_vand_VV(bits, Q6_V_vsplat_R(INT32_MAX));
  const HVX_VectorPred signs = Q6_Q_vcmp_gt_VwVw(zero, bits);
  const HVX_Vector cuts = Q6_V_vmux_QVV(signs, Q6_V_vsplat_R(negative - 1), Q6_V_vsplat_R(positive - 1));
  const HVX_VectorPred exceed = Q6_Q_vcmp_gt_VwVw(magnitude, cuts);
  HVX_Vector selected = Q6_V_vmux_QVV(exceed, lane_values, zero);
  selected = Q6_V_vor_VV(selected, Q6_V_vror_VR(selected, 4));
  selected = Q6_V_vor_VV(selected, Q6_V_vror_VR(selected, 8));
  selected = Q6_V_vor_VV(selected, Q6_V_vror_VR(selected, 16));
  selected = Q6_V_vor_VV(selected, Q6_V_vror_VR(selected, 32));
  selected = Q6_V_vor_VV(selected, Q6_V_vror_VR(selected, 64));
  return (uint32_t)Q6_R_vextract_VR(selected, 0);
}
#else
static inline uint32_t clipping_mask(const float *values, uint32_t positive, uint32_t negative) {
  uint32_t selected = 0;
  for (unsigned lane = 0; lane < vector_width; ++lane) {
    const uint32_t bits = word_bits(values + lane);
    const uint32_t cut = bits >> 31 ? negative : positive;
    if ((bits & UINT32_C(0x7FFFFFFF)) >= cut) {
      selected |= UINT32_C(1) << lane;
    }
  }
  return selected;
}
#endif

static int row_needs_output_repair(const int16_t *codes, uint16_t minimum, uint16_t maximum) {
  const auto *words = reinterpret_cast<const uint16_t *>(codes);
  for (unsigned channel = 0; channel < out_width; ++channel) {
    if (words[channel] == minimum || words[channel] == maximum) {
      return 1;
    }
  }
  return 0;
}

inline int pack_shadow_correction_tile(const float *activation, const int16_t *codes,
                                uint32_t positive, uint32_t negative, uint32_t flags,
                                struct correction_packet *packet,
                                code_encoding encoding = code_encoding::signed_fixed) {
  memset(packet, 0, sizeof(*packet));
  packet->status = needs_original_tile;
  packet->mask_flags = flags;
  if ((flags & ~(clip_input | clip_output)) != 0 || positive == 0 || negative == 0 ||
      positive >= UINT32_C(0x7F800000) || negative >= UINT32_C(0x7F800000)) {
    return needs_original_tile;
  }

  // Adding the offset 32768 to a signed code flips only its sign bit.
  const uint16_t bias = static_cast<uint16_t>(encoding);
  const uint16_t minimum = static_cast<uint16_t>(INT16_MIN) ^ bias;
  const uint16_t maximum = static_cast<uint16_t>(INT16_MAX) ^ bias;
  unsigned sparse_count = 0;
  unsigned dense_count = 0;
  for (unsigned row = 0; row < tile_rows; ++row) {
    const float *values = activation + row * in_width;
    if (!(flags & clip_output) && row_needs_output_repair(codes + row * out_width, minimum, maximum)) {
      if (dense_count == dense_capacity) return needs_original_tile;
      packet->dense_row_ids[dense_count] = row;
      memcpy(packet->dense_rows[dense_count], values, in_width * sizeof(float));
      ++dense_count;
    } else if (!(flags & clip_input)) {
      for (unsigned channel = 0; channel < in_width; channel += vector_width) {
        uint32_t lanes = clipping_mask(values + channel, positive, negative);
        if ((unsigned)__builtin_popcount(lanes) > sparse_capacity - sparse_count) return needs_original_tile;
        while (lanes) {
          const unsigned column = channel + (unsigned)__builtin_ctz(lanes);
          const uint32_t bits = word_bits(values + column);
          if ((bits & UINT32_C(0x7FFFFFFF)) >= UINT32_C(0x7F800000)) return needs_original_tile;
          packet->sparse[sparse_count++] = sparse_entry{column, bits};
          lanes &= lanes - 1;
        }
      }
    }
    packet->row_offsets[row + 1] = sparse_count;
  }

  packet->sparse_count = sparse_count;
  packet->dense_count = dense_count;
  packet->status = complete;
  return complete;
}

}  // namespace shadow_correction_packet
