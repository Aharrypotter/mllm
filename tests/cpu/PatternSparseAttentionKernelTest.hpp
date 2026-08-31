// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mllm/backends/cpu/kernels/common/shadownpu/PatternSparseAttention.hpp"

class PatternSparseAttentionKernelTest : public ::testing::Test {
 protected:
  void structuredPatternUsesEndpointUniformHistoryAndLocalSuffix() {
    std::array<int32_t, 6> selected{};
    mllm::cpu::shadownpu::buildStructuredPatternIndices(12, 6, 0.5F, 0, selected.data());
    EXPECT_EQ(selected, (std::array<int32_t, 6>{0, 4, 8, 9, 10, 11}));
  }

  void structuredPatternHonorsPrefixAndKeepOneRule() {
    std::array<int32_t, 5> selected{};
    mllm::cpu::shadownpu::buildStructuredPatternIndices(12, 5, 0.4F, 2, selected.data());
    EXPECT_EQ(selected, (std::array<int32_t, 5>{0, 1, 2, 10, 11}));

    int32_t keep_one = -1;
    mllm::cpu::shadownpu::buildStructuredPatternIndices(5, 1, 0.0F, 0, &keep_one);
    EXPECT_EQ(keep_one, 0);
  }

  void randomPatternIsExactSortedAndRepeatable() {
    constexpr int32_t valid = 100;
    constexpr int32_t keep = 20;
    std::vector<int32_t> first(keep);
    std::vector<int32_t> repeated(keep);
    std::vector<int32_t> different(keep);
    const auto seed = mllm::cpu::shadownpu::makeRandomRowSeed(1234, 17, 0, 2);
    mllm::cpu::shadownpu::buildRandomPatternIndices(valid, keep, seed, first.data());
    mllm::cpu::shadownpu::buildRandomPatternIndices(valid, keep, seed, repeated.data());
    mllm::cpu::shadownpu::buildRandomPatternIndices(valid, keep, mllm::cpu::shadownpu::makeRandomRowSeed(5678, 17, 0, 2),
                                                    different.data());

    EXPECT_EQ(first, repeated);
    EXPECT_NE(first, different);
    EXPECT_TRUE(std::is_sorted(first.begin(), first.end()));
    EXPECT_EQ(std::adjacent_find(first.begin(), first.end()), first.end());
    for (const int32_t index : first) {
      EXPECT_GE(index, 0);
      EXPECT_LT(index, valid);
    }
  }

  void randomPatternKeepsAllKeysInOrder() {
    std::array<int32_t, 7> selected{};
    mllm::cpu::shadownpu::buildRandomPatternIndices(7, 7, mllm::cpu::shadownpu::makeRandomRowSeed(1234, 6, 0, 0),
                                                    selected.data());
    EXPECT_EQ(selected, (std::array<int32_t, 7>{0, 1, 2, 3, 4, 5, 6}));
  }

  void disjointMatrixAndRowRangesMatchWholeExternalSelection() {
    constexpr int32_t batch = 2;
    constexpr int32_t query_heads = 2;
    constexpr int32_t kv_heads = 1;
    constexpr int32_t query_len = 3;
    constexpr int32_t key_len = 4;
    constexpr int32_t dim = 2;
    constexpr size_t matrix_count = static_cast<size_t>(batch) * query_heads;
    const std::array<float, query_heads> retentions{0.5F, 0.75F};

    mllm::cpu::shadownpu::PatternSparseAttentionF32Params params;
    params.batch = batch;
    params.query_heads = query_heads;
    params.kv_heads = kv_heads;
    params.query_len = query_len;
    params.key_len = key_len;
    params.query_key_dim = dim;
    params.value_dim = dim;
    params.query_stride = {query_heads * query_len * dim, query_len * dim, dim, 1};
    params.key_stride = {kv_heads * key_len * dim, key_len * dim, dim, 1};
    params.value_stride = params.key_stride;
    params.output_stride = params.query_stride;
    params.sparsity = 0.5F;
    params.causal_mask = true;
    params.head_retentions = retentions.data();

    std::vector<float> query(static_cast<size_t>(batch) * query_heads * query_len * dim);
    std::vector<float> key(static_cast<size_t>(batch) * kv_heads * key_len * dim);
    std::vector<float> value(key.size());
    for (size_t index = 0; index < query.size(); ++index) {
      query[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.125F;
    }
    for (size_t index = 0; index < key.size(); ++index) {
      key[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.25F;
      value[index] = static_cast<float>(index + 1) * 0.0625F;
    }

    std::array<std::array<int32_t, query_len + 1>, query_heads> offsets{};
    size_t external_stride = 0;
    for (int32_t head = 0; head < query_heads; ++head) {
      for (int32_t row = 0; row < query_len; ++row) {
        const int32_t valid = key_len - query_len + row + 1;
        const int32_t keep = std::max(1, static_cast<int32_t>(std::ceil(retentions[head] * valid)));
        offsets[head][row + 1] = offsets[head][row] + keep;
      }
      external_stride = std::max(external_stride, static_cast<size_t>(offsets[head][query_len]));
    }
    std::vector<int32_t> indices(matrix_count * external_stride, -1);
    for (size_t matrix = 0; matrix < matrix_count; ++matrix) {
      const int32_t head = static_cast<int32_t>(matrix % query_heads);
      for (int32_t row = 0; row < query_len; ++row) {
        const int32_t valid = key_len - query_len + row + 1;
        const int32_t begin = offsets[head][row];
        const int32_t keep = offsets[head][row + 1] - begin;
        mllm::cpu::shadownpu::buildStructuredPatternIndices(
            valid, keep, params.local_ratio, params.prefix_tokens,
            indices.data() + matrix * external_stride + static_cast<size_t>(begin));
      }
    }

    std::vector<float> whole(query.size(), 0.0F);
    std::vector<float> ranged(query.size(), -999.0F);
    ASSERT_TRUE(mllm::cpu::shadownpu::patternSparseAttentionTyped(params, query.data(), key.data(), value.data(), whole.data(),
                                                                  indices.data(), external_stride, offsets[0].data(),
                                                                  query_len + 1));
    for (size_t matrix = 0; matrix < matrix_count; ++matrix) {
      ASSERT_TRUE(mllm::cpu::shadownpu::patternSparseAttentionRangeTyped(params, query.data(), key.data(), value.data(),
                                                                         ranged.data(), matrix, 1, 0, 1, indices.data(),
                                                                         external_stride, offsets[0].data(), query_len + 1));
      ASSERT_TRUE(mllm::cpu::shadownpu::patternSparseAttentionRangeTyped(
          params, query.data(), key.data(), value.data(), ranged.data(), matrix, 1, 1, query_len - 1, indices.data(),
          external_stride, offsets[0].data(), query_len + 1));
    }
    EXPECT_EQ(ranged, whole);
    EXPECT_FALSE(mllm::cpu::shadownpu::patternSparseAttentionRangeTyped(
        params, query.data(), key.data(), value.data(), ranged.data(), matrix_count, 1, 0, query_len, indices.data(),
        external_stride, offsets[0].data(), query_len + 1));
  }

  void hmxFp16RangeMatchesGenericExternalSelectionWithDimensionTail() {
    constexpr int32_t batch = 1;
    constexpr int32_t query_heads = 2;
    constexpr int32_t kv_heads = 1;
    constexpr int32_t query_len = 2;
    constexpr int32_t key_len = 5;
    constexpr int32_t dim = 10;
    constexpr size_t matrix_count = static_cast<size_t>(batch) * query_heads;
    const std::array<float, query_heads> retentions{0.4F, 0.8F};

    mllm::cpu::shadownpu::PatternSparseAttentionF32Params params;
    params.batch = batch;
    params.query_heads = query_heads;
    params.kv_heads = kv_heads;
    params.query_len = query_len;
    params.key_len = key_len;
    params.query_key_dim = dim;
    params.value_dim = dim;
    params.query_stride = {query_heads * query_len * dim, query_len * dim, dim, 1};
    params.key_stride = {kv_heads * key_len * dim, key_len * dim, dim, 1};
    params.value_stride = params.key_stride;
    params.output_stride = params.query_stride;
    params.sparsity = 0.5F;
    params.causal_mask = true;
    params.head_retentions = retentions.data();

    std::vector<float> query(static_cast<size_t>(batch) * query_heads * query_len * dim);
    std::vector<float> key_f32(static_cast<size_t>(batch) * kv_heads * key_len * dim);
    std::vector<float> value_f32(key_f32.size());
    for (size_t index = 0; index < query.size(); ++index) {
      query[index] = static_cast<float>(static_cast<int>(index % 9) - 4) * 0.125F;
    }
    for (size_t index = 0; index < key_f32.size(); ++index) {
      key_f32[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.0625F;
      value_f32[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.03125F;
    }
    std::vector<mllm::mllm_fp16_t> key(key_f32.size());
    std::vector<mllm::mllm_fp16_t> value(value_f32.size());
    mllm::cpu::mllm_fp32_to_fp16_row(key_f32.data(), key.data(), static_cast<int>(key.size()));
    mllm::cpu::mllm_fp32_to_fp16_row(value_f32.data(), value.data(), static_cast<int>(value.size()));

    std::array<std::array<int32_t, query_len + 1>, query_heads> offsets{};
    size_t external_stride = 0;
    for (int32_t head = 0; head < query_heads; ++head) {
      for (int32_t row = 0; row < query_len; ++row) {
        const int32_t valid = key_len - query_len + row + 1;
        const int32_t keep = std::max(1, static_cast<int32_t>(std::ceil(retentions[head] * valid)));
        offsets[head][row + 1] = offsets[head][row] + keep;
      }
      external_stride = std::max(external_stride, static_cast<size_t>(offsets[head][query_len]));
    }
    std::vector<int32_t> indices(matrix_count * external_stride, -1);
    for (size_t matrix = 0; matrix < matrix_count; ++matrix) {
      const int32_t head = static_cast<int32_t>(matrix % query_heads);
      for (int32_t row = 0; row < query_len; ++row) {
        const int32_t valid = key_len - query_len + row + 1;
        const int32_t begin = offsets[head][row];
        const int32_t keep = offsets[head][row + 1] - begin;
        mllm::cpu::shadownpu::buildStructuredPatternIndices(
            valid, keep, params.local_ratio, params.prefix_tokens,
            indices.data() + matrix * external_stride + static_cast<size_t>(begin));
      }
    }

    std::vector<float> reference(query.size(), 0.0F);
    std::vector<float> candidate(query.size(), 0.0F);
    ASSERT_TRUE(mllm::cpu::shadownpu::patternSparseAttentionTyped(params, query.data(), key.data(), value.data(),
                                                                  reference.data(), indices.data(), external_stride,
                                                                  offsets[0].data(), query_len + 1));
    ASSERT_TRUE(mllm::cpu::shadownpu::patternSparseAttentionHmxF16Range(
        params, query.data(), key.data(), value.data(), candidate.data(), 0, matrix_count, 0, query_len, indices.data(),
        external_stride, offsets[0].data(), query_len + 1));
    for (size_t index = 0; index < reference.size(); ++index) {
      EXPECT_NEAR(candidate[index], reference[index], 0.003F) << "index=" << index;
    }
  }
};
