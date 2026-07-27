// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/common/gdn/gated_delta_net.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace mllm::cpu::gdn {
namespace {

float stableSoftplus(float value) {
  if (value > 20.0F) { return value; }
  if (value < -20.0F) { return std::exp(value); }
  return std::log1p(std::exp(value));
}

float stableSigmoid(float value) {
  if (value >= 0.0F) {
    const float exp_value = std::exp(-value);
    return 1.0F / (1.0F + exp_value);
  }
  const float exp_value = std::exp(value);
  return exp_value / (1.0F + exp_value);
}

void squaredNorms(const float* query, const float* key, int length, float& query_norm_sq, float& key_norm_sq) {
  query_norm_sq = 0.0F;
  key_norm_sq = 0.0F;
  int index = 0;
#if defined(__aarch64__)
  float32x4_t query_sum = vdupq_n_f32(0.0F);
  float32x4_t key_sum = vdupq_n_f32(0.0F);
  for (; index + 4 <= length; index += 4) {
    const float32x4_t query_value = vld1q_f32(query + index);
    const float32x4_t key_value = vld1q_f32(key + index);
    query_sum = vmlaq_f32(query_sum, query_value, query_value);
    key_sum = vmlaq_f32(key_sum, key_value, key_value);
  }
  query_norm_sq = vaddvq_f32(query_sum);
  key_norm_sq = vaddvq_f32(key_sum);
#endif
  for (; index < length; ++index) {
    query_norm_sq += query[index] * query[index];
    key_norm_sq += key[index] * key[index];
  }
}

float decayAndDot(float* state, const float* key, float decay, float key_scale, int length) {
  float dot = 0.0F;
  int index = 0;
#if defined(__aarch64__)
  float32x4_t sum = vdupq_n_f32(0.0F);
  for (; index + 4 <= length; index += 4) {
    float32x4_t state_value = vld1q_f32(state + index);
    state_value = vmulq_n_f32(state_value, decay);
    vst1q_f32(state + index, state_value);
    const float32x4_t key_value = vmulq_n_f32(vld1q_f32(key + index), key_scale);
    sum = vmlaq_f32(sum, state_value, key_value);
  }
  dot = vaddvq_f32(sum);
#endif
  for (; index < length; ++index) {
    state[index] *= decay;
    dot += state[index] * key[index] * key_scale;
  }
  return dot;
}

float updateAndDot(float* state, const float* key, const float* query, float delta, float key_scale, float query_scale,
                   int length) {
  float dot = 0.0F;
  int index = 0;
#if defined(__aarch64__)
  float32x4_t sum = vdupq_n_f32(0.0F);
  for (; index + 4 <= length; index += 4) {
    float32x4_t state_value = vld1q_f32(state + index);
    const float32x4_t key_value = vmulq_n_f32(vld1q_f32(key + index), key_scale);
    state_value = vmlaq_n_f32(state_value, key_value, delta);
    vst1q_f32(state + index, state_value);
    const float32x4_t query_value = vmulq_n_f32(vld1q_f32(query + index), query_scale);
    sum = vmlaq_f32(sum, state_value, query_value);
  }
  dot = vaddvq_f32(sum);
#endif
  for (; index < length; ++index) {
    state[index] += delta * key[index] * key_scale;
    dot += state[index] * query[index] * query_scale;
  }
  return dot;
}

}  // namespace

void depthwiseCausalConvF32(const float* input, const float* weight, float* state, float* output, int batch_size,
                            int sequence_length, int channels, int kernel_size) {
  if (input == nullptr || weight == nullptr || state == nullptr || output == nullptr) {
    throw std::invalid_argument("GDN depthwise causal convolution received a null pointer");
  }
  if (batch_size <= 0 || sequence_length <= 0 || channels <= 0 || kernel_size <= 1) {
    throw std::invalid_argument("GDN depthwise causal convolution received an invalid shape");
  }

  const int state_width = kernel_size - 1;
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int token = 0; token < sequence_length; ++token) {
      for (int channel = 0; channel < channels; ++channel) {
        const std::size_t state_base = (static_cast<std::size_t>(batch) * channels + channel) * state_width;
        const std::size_t input_index = (static_cast<std::size_t>(batch) * sequence_length + token) * channels + channel;
        const std::size_t weight_base = static_cast<std::size_t>(channel) * kernel_size;

        float value = input[input_index] * weight[weight_base + state_width];
        for (int tap = 0; tap < state_width; ++tap) { value += state[state_base + tap] * weight[weight_base + tap]; }
        output[input_index] = value;

        for (int tap = 0; tap + 1 < state_width; ++tap) { state[state_base + tap] = state[state_base + tap + 1]; }
        state[state_base + state_width - 1] = input[input_index];
      }
    }
  }
}

void gatedDeltaRuleF32(const float* q, const float* k, const float* v, const float* a, const float* b, const float* a_log,
                       const float* dt_bias, float* state, float* output, int batch_size, int sequence_length,
                       int num_key_heads, int num_value_heads, int key_head_dim, int value_head_dim) {
  if (q == nullptr || k == nullptr || v == nullptr || a == nullptr || b == nullptr || a_log == nullptr || dt_bias == nullptr
      || state == nullptr || output == nullptr) {
    throw std::invalid_argument("Gated delta rule received a null pointer");
  }
  if (batch_size <= 0 || sequence_length <= 0 || num_key_heads <= 0 || num_value_heads <= 0 || key_head_dim <= 0
      || value_head_dim <= 0 || num_value_heads % num_key_heads != 0) {
    throw std::invalid_argument("Gated delta rule received an invalid shape");
  }

  const int key_head_repeats = num_value_heads / num_key_heads;
  const float query_dim_scale = 1.0F / std::sqrt(static_cast<float>(key_head_dim));

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int token = 0; token < sequence_length; ++token) {
      for (int value_head = 0; value_head < num_value_heads; ++value_head) {
        const int key_head = value_head / key_head_repeats;
        const std::size_t gate_index =
            (static_cast<std::size_t>(batch) * sequence_length + token) * num_value_heads + value_head;
        const std::size_t qk_base =
            ((static_cast<std::size_t>(batch) * sequence_length + token) * num_key_heads + key_head) * key_head_dim;
        const std::size_t value_base =
            ((static_cast<std::size_t>(batch) * sequence_length + token) * num_value_heads + value_head) * value_head_dim;
        const std::size_t state_base =
            (static_cast<std::size_t>(batch) * num_value_heads + value_head) * value_head_dim * key_head_dim;

        float query_norm_sq = 0.0F;
        float key_norm_sq = 0.0F;
        squaredNorms(q + qk_base, k + qk_base, key_head_dim, query_norm_sq, key_norm_sq);
        const float query_scale = query_dim_scale / std::sqrt(query_norm_sq + 1.0e-6F);
        const float key_scale = 1.0F / std::sqrt(key_norm_sq + 1.0e-6F);

        const float gate = -std::exp(a_log[value_head]) * stableSoftplus(a[gate_index] + dt_bias[value_head]);
        const float decay = std::exp(gate);
        const float beta = stableSigmoid(b[gate_index]);

        for (int value_dim = 0; value_dim < value_head_dim; ++value_dim) {
          float* state_row = state + state_base + static_cast<std::size_t>(value_dim) * key_head_dim;
          const float state_dot_key = decayAndDot(state_row, k + qk_base, decay, key_scale, key_head_dim);
          const float delta = (v[value_base + value_dim] - state_dot_key) * beta;
          output[value_base + value_dim] =
              updateAndDot(state_row, k + qk_base, q + qk_base, delta, key_scale, query_scale, key_head_dim);
        }
      }
    }
  }
}

}  // namespace mllm::cpu::gdn
