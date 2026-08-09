// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"

namespace mllm::nn {

// A batch-1 eager cache that keeps one persistent history per KV head. The
// caller owns the logical slot id, so cache identity does not depend on which
// physical module produced the key/value tensors.
class KVHeadStaticCache final : public AbstractStaticCache {
 public:
  KVHeadStaticCache() = default;
  KVHeadStaticCache(int32_t max_cache_length, int32_t logical_slots, int32_t kv_heads, int32_t head_dim,
                    DataTypes k_dtype = kFloat32, DataTypes v_dtype = kFloat32, DeviceTypes device_type = kCPU);

  void setCurrentSeqCnt(int32_t seq) override;
  void clearCache() override;

  [[nodiscard]] int32_t getCurrentSeqCnt(int32_t logical_slot) const override;
  [[nodiscard]] int32_t getLayerNums() const override { return logical_slots_; }
  [[nodiscard]] int32_t maxCacheLength() const { return max_cache_length_; }
  [[nodiscard]] int32_t kvHeads() const { return kv_heads_; }
  [[nodiscard]] int32_t headDim() const { return head_dim_; }

  std::array<Tensor, 2> updateKVCache(int32_t logical_slot, Tensor k, Tensor v) override;
  std::array<Tensor, 2> getKVCache(int32_t logical_slot) const;

  [[nodiscard]] Tensor getKCacheBuffer(int32_t logical_slot) const;
  [[nodiscard]] Tensor getVCacheBuffer(int32_t logical_slot) const;

 private:
  void validateLogicalSlot(int32_t logical_slot) const;
  void validateInput(const Tensor& k, const Tensor& v) const;

  DeviceTypes device_type_ = kCPU;
  DataTypes k_dtype_ = kFloat32;
  DataTypes v_dtype_ = kFloat32;
  int32_t max_cache_length_ = 0;
  int32_t logical_slots_ = 0;
  int32_t kv_heads_ = 0;
  int32_t head_dim_ = 0;

  std::vector<Tensor> k_cache_;
  std::vector<Tensor> v_cache_;
  std::vector<int32_t> current_seq_cnt_;
};

}  // namespace mllm::nn
