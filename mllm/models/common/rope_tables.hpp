// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>

#include "mllm/core/Tensor.hpp"

namespace mllm::models::common {

// Analytical RoPE tables for models whose rotation is performed by the
// registered nn::RoPE operation and whose only model-side responsibility is
// materializing the immutable frequency table and the per-request sin/cos
// buffers.
//
// This is request orchestration that produces constant operation inputs, not a
// tensor operation, so it stays on the model side rather than under
// nn/llm_components.
//
// This is the plain default-RoPE contract with no attention scaling. Models
// that need a scaled or otherwise reparameterized table should not reuse these
// helpers, because the scaling factor changes the frozen numerical result.

inline auto makeRoPEInvFreq(int32_t head_dim, float rope_theta) -> Tensor {
  if (head_dim <= 1 || head_dim % 2 != 0) {
    throw std::invalid_argument("makeRoPEInvFreq requires an even head_dim greater than one");
  }
  if (!std::isfinite(rope_theta) || rope_theta <= 0.0F) {
    throw std::invalid_argument("makeRoPEInvFreq requires a finite positive rope_theta");
  }
  auto inv_freq = Tensor::empty({head_dim / 2}, kFloat32, kCPU).alloc();
  auto* data = inv_freq.ptr<float>();
  for (int32_t index = 0; index < head_dim / 2; ++index) {
    data[index] = 1.0F / std::pow(rope_theta, 2.0F * static_cast<float>(index) / static_cast<float>(head_dim));
  }
  return inv_freq;
}

// position_ids is [B, S] int64. The returned sin/cos are [B, S, head_dim] with
// each half-dimension angle duplicated into both halves, matching the
// rotate-half layout consumed by nn::RoPE.
inline auto makeRotaryPosEmbedding(const Tensor& position_ids, const Tensor& inv_freq) -> std::pair<Tensor, Tensor> {
  if (position_ids.isNil() || inv_freq.isNil()) {
    throw std::invalid_argument("makeRotaryPosEmbedding inputs must not be nil");
  }
  if (position_ids.shape().size() != 2 || position_ids.dtype() != kInt64 || position_ids.device() != kCPU) {
    throw std::invalid_argument("makeRotaryPosEmbedding expects a rank-2 int64 CPU position_ids tensor");
  }
  if (inv_freq.shape().size() != 1 || inv_freq.dtype() != kFloat32 || inv_freq.device() != kCPU) {
    throw std::invalid_argument("makeRotaryPosEmbedding expects a rank-1 float32 CPU inv_freq tensor");
  }

  const auto batch = position_ids.shape()[0];
  const auto sequence = position_ids.shape()[1];
  const auto half_dim = inv_freq.shape()[0];
  const auto head_dim = half_dim * 2;
  auto sin = Tensor::empty({batch, sequence, head_dim}, kFloat32, kCPU).alloc();
  auto cos = Tensor::empty({batch, sequence, head_dim}, kFloat32, kCPU).alloc();
  const auto* positions = position_ids.ptr<int64_t>();
  const auto* frequencies = inv_freq.ptr<float>();
  auto* sin_data = sin.ptr<float>();
  auto* cos_data = cos.ptr<float>();
  for (int32_t b = 0; b < batch; ++b) {
    for (int32_t s = 0; s < sequence; ++s) {
      for (int32_t d = 0; d < half_dim; ++d) {
        const auto angle = static_cast<float>(positions[b * sequence + s]) * frequencies[d];
        const auto offset = (b * sequence + s) * head_dim;
        sin_data[offset + d] = sin_data[offset + d + half_dim] = std::sin(angle);
        cos_data[offset + d] = cos_data[offset + d + half_dim] = std::cos(angle);
      }
    }
  }
  return {sin, cos};
}

}  // namespace mllm::models::common
