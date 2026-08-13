// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/nn/lmcache/KVHeadStaticCache.hpp"

namespace {

using mllm::Tensor;

class KVHeadStaticCacheTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

Tensor values(int sequence, float base) {
  auto tensor = Tensor::empty({1, 2, sequence, 3}, mllm::kFloat32, mllm::kCPU).alloc();
  for (int index = 0; index < tensor.numel(); ++index) { tensor.ptr<float>()[index] = base + static_cast<float>(index); }
  return tensor;
}

TEST_F(KVHeadStaticCacheTest, StoresNativeKVHeadsAcrossUpdates) {
  mllm::nn::KVHeadStaticCache cache(/*capacity=*/4, /*logical_slots=*/2, /*kv_heads=*/2, /*head_dim=*/3);

  auto [first_k, first_v] = cache.updateKVCache(0, values(2, 0.0F), values(2, 100.0F));
  EXPECT_EQ(first_k.shape(), (Tensor::shape_t{1, 2, 2, 3}));
  EXPECT_EQ(first_v.shape(), (Tensor::shape_t{1, 2, 2, 3}));
  EXPECT_EQ(cache.getKCacheBuffer(0).shape(), (Tensor::shape_t{1, 2, 4, 3}));

  auto [all_k, all_v] = cache.updateKVCache(0, values(1, 20.0F), values(1, 120.0F));
  EXPECT_EQ(all_k.shape(), (Tensor::shape_t{1, 2, 3, 3}));
  EXPECT_EQ(all_v.shape(), (Tensor::shape_t{1, 2, 3, 3}));
  EXPECT_EQ(cache.getCurrentSeqCnt(0), 3);
  EXPECT_EQ(cache.getCurrentSeqCnt(1), 0);
  all_k = all_k.contiguous();
  EXPECT_EQ(std::vector<float>(all_k.ptr<float>(), all_k.ptr<float>() + all_k.numel()),
            (std::vector<float>{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 20.0F, 21.0F, 22.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F,
                                23.0F, 24.0F, 25.0F}));
}

TEST_F(KVHeadStaticCacheTest, KeepsLogicalSlotsIndependentAndClearsCounters) {
  mllm::nn::KVHeadStaticCache cache(/*capacity=*/4, /*logical_slots=*/2, /*kv_heads=*/2, /*head_dim=*/3);
  cache.updateKVCache(0, values(2, 0.0F), values(2, 10.0F));
  cache.updateKVCache(1, values(1, 50.0F), values(1, 60.0F));

  EXPECT_EQ(cache.getCurrentSeqCnt(0), 2);
  EXPECT_EQ(cache.getCurrentSeqCnt(1), 1);
  cache.clearCache();
  EXPECT_EQ(cache.getCurrentSeqCnt(0), 0);
  EXPECT_EQ(cache.getCurrentSeqCnt(1), 0);
  cache.updateKVCache(1, values(1, 70.0F), values(1, 80.0F));
  EXPECT_EQ(cache.getCurrentSeqCnt(1), 1);
}

TEST_F(KVHeadStaticCacheTest, RejectsOverflowBeforeAdvancingTheSlot) {
  mllm::nn::KVHeadStaticCache cache(/*capacity=*/2, /*logical_slots=*/1, /*kv_heads=*/2, /*head_dim=*/3);
  cache.updateKVCache(0, values(1, 0.0F), values(1, 10.0F));

  EXPECT_THROW(cache.updateKVCache(0, values(2, 20.0F), values(2, 30.0F)), std::out_of_range);
  EXPECT_EQ(cache.getCurrentSeqCnt(0), 1);
}

TEST_F(KVHeadStaticCacheTest, RejectsInvalidSlotAndShape) {
  mllm::nn::KVHeadStaticCache cache(/*capacity=*/2, /*logical_slots=*/1, /*kv_heads=*/2, /*head_dim=*/3);
  EXPECT_THROW((void)cache.getCurrentSeqCnt(1), std::out_of_range);

  auto wrong_heads = Tensor::zeros({1, 1, 1, 3}, mllm::kFloat32, mllm::kCPU);
  EXPECT_THROW(cache.updateKVCache(0, wrong_heads, wrong_heads), std::invalid_argument);
}

}  // namespace
