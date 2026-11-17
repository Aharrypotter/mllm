// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <stdexcept>

#include "mllm/mllm.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/utils/Common.hpp"

using namespace mllm;  // NOLINT

int main() {
  mllm::initializeContext();

  constexpr int max_cache_length = 32;
  constexpr int layer_nums = 12;
  constexpr int q_heads = 4;
  constexpr int kv_heads = 2;
  constexpr int kv_dim = 6;

  nn::StaticCache cache(max_cache_length, layer_nums, q_heads, kv_heads, kv_dim, kFloat32, kFloat32, kCPU, false);

  for (int i = 0; i < layer_nums; ++i) { MLLM_RT_ASSERT_EQ(cache.getCurrentSeqCnt(i), 0); }

  // 1. Insert KV, [B, H, S=1, D]
  {
    print("Insert KV, [B, H, S=1, D]");
    auto k = Tensor::random({1, kv_heads, 1, kv_dim});
    auto v = Tensor::random({1, kv_heads, 1, kv_dim});

    auto [key, value] = cache.updateKVCache(0, k, v);

    print(key.shape(), value.shape());
    print(key);
    print(value);
  }

  // 1. Insert KV, [B, H, S=2, D]
  {
    print("Insert KV, [B, H, S=2, D]");
    auto k = Tensor::random({1, kv_heads, 2, kv_dim});
    auto v = Tensor::random({1, kv_heads, 2, kv_dim});

    auto [key, value] = cache.updateKVCache(0, k, v);

    print(key.shape(), value.shape());
    print(key);
    print(value);
  }

  // Filling exactly to capacity succeeds; the next token must be rejected
  // without mutating the stored sequence count.
  nn::StaticCache boundary(2, 1, q_heads, kv_heads, kv_dim, kFloat32, kFloat32, kCPU, false);
  auto two = Tensor::zeros({1, kv_heads, 2, kv_dim}, kFloat32, kCPU);
  auto one = Tensor::zeros({1, kv_heads, 1, kv_dim}, kFloat32, kCPU);
  boundary.updateKVCache(0, two, two);
  bool rejected = false;
  try { boundary.updateKVCache(0, one, one); }
  catch (const std::out_of_range&) { rejected = true; }
  if (!rejected || boundary.getCurrentSeqCnt(0) != 2) { return 1; }
  print("STATIC_CACHE_BOUNDARY_OK");

  mllm::shutdownContext();
}
