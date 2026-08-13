// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "mllm/backends/cpu/kernels/common/fa2_1/arch.hpp"
#include "mllm/backends/cpu/kernels/common/fa2_1/impl-any.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include "mllm/backends/cpu/kernels/common/fa2_1/impl-arm.hpp"
#elif defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
#include "mllm/backends/cpu/kernels/common/fa2_1/impl-any-simd.hpp"
#endif

namespace mllm::cpu::gqa_decode {

struct BhsdStrides {
  int32_t batch;
  int32_t head;
  int32_t sequence;
  int32_t dimension;
};

namespace detail {

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
using attention_arch_tag = flash_attn2::details::arm_arch_tag;
#elif defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
using attention_arch_tag = flash_attn2::details::x86_arch_tag;
#else
using attention_arch_tag = flash_attn2::details::any_arch_tag;
#endif

inline bool validStrides(const BhsdStrides& strides) {
  return strides.batch > 0 && strides.head > 0 && strides.sequence > 0 && strides.dimension == 1;
}

}  // namespace detail

// Single-token float32 GQA for native KV-head [B, H, S, D] cache views.
// QK and softmax preserve C10's complete per-query-head traversal. P@V groups
// the query heads sharing one KV head so each V token is reused while hot, but
// every output head still accumulates in increasing key-index order.
// Scratch: group_size * kv_sequence floats.
inline bool fwdBhsdFp32(int32_t batch_size, int32_t query_heads, int32_t kv_heads, int32_t kv_sequence, int32_t qk_dim,
                        int32_t value_dim, const mllm_fp32_t* query, BhsdStrides query_strides, const mllm_fp32_t* key,
                        BhsdStrides key_strides, const mllm_fp32_t* value, BhsdStrides value_strides, mllm_fp32_t* output,
                        BhsdStrides output_strides, mllm_fp32_t* grouped_scratch) {
  if (batch_size <= 0 || query_heads <= 0 || kv_heads <= 0 || kv_sequence <= 0 || qk_dim <= 0 || value_dim <= 0
      || query_heads % kv_heads != 0 || query == nullptr || key == nullptr || value == nullptr || output == nullptr
      || grouped_scratch == nullptr || !detail::validStrides(query_strides) || !detail::validStrides(key_strides)
      || !detail::validStrides(value_strides) || !detail::validStrides(output_strides)) {
    return false;
  }

  using ArchTag = detail::attention_arch_tag;
  const int32_t group_size = query_heads / kv_heads;
  const float scale = 1.0F / std::sqrt(static_cast<float>(qk_dim));

  for (int32_t batch = 0; batch < batch_size; ++batch) {
    for (int32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
      const int32_t first_query_head = kv_head * group_size;
      const auto* query_group = query + static_cast<size_t>(batch) * query_strides.batch
                                + static_cast<size_t>(first_query_head) * query_strides.head;
      const auto* key_head = key + static_cast<size_t>(batch) * key_strides.batch
                             + static_cast<size_t>(kv_head) * key_strides.head;
      const auto* value_head = value + static_cast<size_t>(batch) * value_strides.batch
                               + static_cast<size_t>(kv_head) * value_strides.head;
      auto* output_group = output + static_cast<size_t>(batch) * output_strides.batch
                           + static_cast<size_t>(first_query_head) * output_strides.head;

      for (int32_t group_index = 0; group_index < group_size; ++group_index) {
        const auto* query_token = query_group + static_cast<size_t>(group_index) * query_strides.head;
        auto* probabilities = grouped_scratch + static_cast<size_t>(group_index) * kv_sequence;
        float maximum = std::numeric_limits<float>::lowest();
        for (int32_t key_index = 0; key_index < kv_sequence; ++key_index) {
          const auto* key_token = key_head + static_cast<size_t>(key_index) * key_strides.sequence;
          float score = 0.0F;
          flash_attn2::details::VectorDotProduct<ArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t>::run(
              query_token, key_token, &score, static_cast<size_t>(qk_dim));
          probabilities[key_index] = score * scale;
          maximum = std::max(maximum, probabilities[key_index]);
        }

        float denominator = 0.0F;
        for (int32_t key_index = 0; key_index < kv_sequence; ++key_index) {
          probabilities[key_index] = std::exp(probabilities[key_index] - maximum);
          denominator += probabilities[key_index];
        }
        const float inverse_denominator = 1.0F / denominator;
        for (int32_t key_index = 0; key_index < kv_sequence; ++key_index) {
          probabilities[key_index] *= inverse_denominator;
        }
      }

      for (int32_t group_index = 0; group_index < group_size; ++group_index) {
        auto* output_token = output_group + static_cast<size_t>(group_index) * output_strides.head;
        flash_attn2::details::FilledWithConst<ArchTag, mllm_fp32_t>::run(output_token, 0.0F,
                                                                       static_cast<size_t>(value_dim));
      }
      for (int32_t key_index = 0; key_index < kv_sequence; ++key_index) {
        const auto* value_token = value_head + static_cast<size_t>(key_index) * value_strides.sequence;
        auto* probability = grouped_scratch + key_index;
        auto* output_token = output_group;
        for (int32_t group_index = 0; group_index < group_size; ++group_index) {
          flash_attn2::details::FMAConstArray<ArchTag, mllm_fp32_t, mllm_fp32_t, mllm_fp32_t>::run(
              output_token, *probability, value_token, static_cast<size_t>(value_dim));
          probability += kv_sequence;
          output_token += output_strides.head;
        }
      }
    }
  }
  return true;
}

}  // namespace mllm::cpu::gqa_decode
