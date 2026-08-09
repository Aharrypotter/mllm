// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/lmcache/KVHeadStaticCache.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mllm::nn {

KVHeadStaticCache::KVHeadStaticCache(int32_t max_cache_length, int32_t logical_slots, int32_t kv_heads, int32_t head_dim,
                                     DataTypes k_dtype, DataTypes v_dtype, DeviceTypes device_type)
    : device_type_(device_type),
      k_dtype_(k_dtype),
      v_dtype_(v_dtype),
      max_cache_length_(max_cache_length),
      logical_slots_(logical_slots),
      kv_heads_(kv_heads),
      head_dim_(head_dim) {
  if (max_cache_length_ <= 0 || logical_slots_ <= 0 || kv_heads_ <= 0 || head_dim_ <= 0) {
    throw std::invalid_argument("KVHeadStaticCache dimensions must be positive");
  }
  if (device_type_ != kCPU) { throw std::invalid_argument("KVHeadStaticCache currently supports CPU only"); }

  k_cache_.reserve(logical_slots_);
  v_cache_.reserve(logical_slots_);
  current_seq_cnt_.assign(logical_slots_, 0);
  for (int32_t slot = 0; slot < logical_slots_; ++slot) {
    k_cache_.push_back(Tensor::zeros({1, kv_heads_, max_cache_length_, head_dim_}, k_dtype_, device_type_));
    v_cache_.push_back(Tensor::zeros({1, kv_heads_, max_cache_length_, head_dim_}, v_dtype_, device_type_));
  }
}

void KVHeadStaticCache::validateLogicalSlot(int32_t logical_slot) const {
  if (logical_slot < 0 || logical_slot >= logical_slots_) {
    throw std::out_of_range("KVHeadStaticCache logical slot is out of range: " + std::to_string(logical_slot));
  }
}

void KVHeadStaticCache::validateInput(const Tensor& k, const Tensor& v) const {
  if (k.isNil() || v.isNil()) { throw std::invalid_argument("KVHeadStaticCache inputs must not be nil"); }
  const auto k_shape = k.shape();
  const auto v_shape = v.shape();
  if (k_shape.size() != 4 || v_shape.size() != 4 || k_shape[0] != 1 || v_shape[0] != 1 || k_shape[1] != kv_heads_
      || v_shape[1] != kv_heads_ || k_shape[2] <= 0 || k_shape[2] != v_shape[2] || k_shape[3] != head_dim_
      || v_shape[3] != head_dim_) {
    throw std::invalid_argument("KVHeadStaticCache expects matching [1, kv_heads, sequence, head_dim] inputs");
  }
  if (k.dtype() != k_dtype_ || v.dtype() != v_dtype_) {
    throw std::invalid_argument("KVHeadStaticCache input dtype does not match the cache contract");
  }
  if (k.device() != device_type_ || v.device() != device_type_) {
    throw std::invalid_argument("KVHeadStaticCache input device does not match the cache contract");
  }
}

void KVHeadStaticCache::setCurrentSeqCnt(int32_t seq) {
  if (seq < 0 || seq > max_cache_length_) { throw std::out_of_range("KVHeadStaticCache sequence count is out of range"); }
  std::fill(current_seq_cnt_.begin(), current_seq_cnt_.end(), seq);
}

void KVHeadStaticCache::clearCache() { std::fill(current_seq_cnt_.begin(), current_seq_cnt_.end(), 0); }

int32_t KVHeadStaticCache::getCurrentSeqCnt(int32_t logical_slot) const {
  validateLogicalSlot(logical_slot);
  return current_seq_cnt_[logical_slot];
}

std::array<Tensor, 2> KVHeadStaticCache::updateKVCache(int32_t logical_slot, Tensor k, Tensor v) {
  validateLogicalSlot(logical_slot);
  validateInput(k, v);

  const int32_t sequence = k.shape()[2];
  const int32_t current = current_seq_cnt_[logical_slot];
  if (sequence > max_cache_length_ - current) {
    throw std::out_of_range("KVHeadStaticCache sequence exceeds the configured capacity");
  }

  k = k.contiguous();
  v = v.contiguous();
  const size_t k_head_bytes = static_cast<size_t>(sequence) * static_cast<size_t>(head_dim_)
                              * static_cast<size_t>(bytesOfType(k_dtype_)) / static_cast<size_t>(lanesOfType(k_dtype_));
  const size_t v_head_bytes = static_cast<size_t>(sequence) * static_cast<size_t>(head_dim_)
                              * static_cast<size_t>(bytesOfType(v_dtype_)) / static_cast<size_t>(lanesOfType(v_dtype_));

  for (int32_t head = 0; head < kv_heads_; ++head) {
    auto* k_dst = k_cache_[logical_slot].offsettedPtr<mllm_byte_t>({0, head, current, 0});
    auto* v_dst = v_cache_[logical_slot].offsettedPtr<mllm_byte_t>({0, head, current, 0});
    const auto* k_src = k.coffsettedPtr<mllm_byte_t>({0, head, 0, 0});
    const auto* v_src = v.coffsettedPtr<mllm_byte_t>({0, head, 0, 0});
    std::memcpy(k_dst, k_src, k_head_bytes);
    std::memcpy(v_dst, v_src, v_head_bytes);
  }

  current_seq_cnt_[logical_slot] = current + sequence;
  return getKVCache(logical_slot);
}

std::array<Tensor, 2> KVHeadStaticCache::getKVCache(int32_t logical_slot) const {
  validateLogicalSlot(logical_slot);
  const int32_t sequence = current_seq_cnt_[logical_slot];
  return {
      k_cache_[logical_slot][{kAll, kAll, {kAll, sequence}, kAll}],
      v_cache_[logical_slot][{kAll, kAll, {kAll, sequence}, kAll}],
  };
}

Tensor KVHeadStaticCache::getKCacheBuffer(int32_t logical_slot) const {
  validateLogicalSlot(logical_slot);
  return k_cache_[logical_slot];
}

Tensor KVHeadStaticCache::getVCacheBuffer(int32_t logical_slot) const {
  validateLogicalSlot(logical_slot);
  return v_cache_[logical_slot];
}

}  // namespace mllm::nn
