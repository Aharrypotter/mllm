// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/common/gdn/gated_delta_net.hpp"
#include "mllm/core/Parallel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#include <sched.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace mllm::cpu::gdn {
namespace {

constexpr int kMaxStackNormalizedHeadDim = 256;
// GDN state updates are bandwidth-heavy on heterogeneous mobile CPUs. The lane
// cap bounds how many tasks share the per-layer completion barrier. 8 lanes
// matches the 8-core phones this kernel targets (4B has 32 recurrence tasks,
// so all 8 cores participate); bitwise-safe because tasks are disjoint.
constexpr int kMaxParallelGDNLanes = 8;
// Scalar state elements updated across all [batch, value_head] tasks.
constexpr std::size_t kMinParallelGDNWork = 65536;

int availableCpuCount(int fallback) {
#if defined(__linux__) || defined(__ANDROID__)
  cpu_set_t affinity = {};
  if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) { return CPU_COUNT(&affinity); }
#endif
  return fallback;
}

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

void normalizeQueryAndKey(const float* query, const float* key, float query_scale, float key_scale, float* normalized_query,
                          float* normalized_key, int length) {
  int index = 0;
#if defined(__aarch64__)
  for (; index + 4 <= length; index += 4) {
    vst1q_f32(normalized_query + index, vmulq_n_f32(vld1q_f32(query + index), query_scale));
    vst1q_f32(normalized_key + index, vmulq_n_f32(vld1q_f32(key + index), key_scale));
  }
#endif
  for (; index < length; ++index) {
    normalized_query[index] = query[index] * query_scale;
    normalized_key[index] = key[index] * key_scale;
  }
}

float decayAndDotNormalized(float* state, const float* normalized_key, float decay, int length) {
  float dot = 0.0F;
  int index = 0;
#if defined(__aarch64__)
  float32x4_t sum = vdupq_n_f32(0.0F);
  for (; index + 4 <= length; index += 4) {
    float32x4_t state_value = vmulq_n_f32(vld1q_f32(state + index), decay);
    vst1q_f32(state + index, state_value);
    sum = vmlaq_f32(sum, state_value, vld1q_f32(normalized_key + index));
  }
  dot = vaddvq_f32(sum);
#endif
  for (; index < length; ++index) {
    state[index] *= decay;
    dot += state[index] * normalized_key[index];
  }
  return dot;
}

float updateAndDotNormalized(float* state, const float* normalized_key, const float* normalized_query, float delta,
                             int length) {
  float dot = 0.0F;
  int index = 0;
#if defined(__aarch64__)
  float32x4_t sum = vdupq_n_f32(0.0F);
  for (; index + 4 <= length; index += 4) {
    float32x4_t state_value = vld1q_f32(state + index);
    state_value = vmlaq_n_f32(state_value, vld1q_f32(normalized_key + index), delta);
    vst1q_f32(state + index, state_value);
    sum = vmlaq_f32(sum, state_value, vld1q_f32(normalized_query + index));
  }
  dot = vaddvq_f32(sum);
#endif
  for (; index < length; ++index) {
    state[index] += delta * normalized_key[index];
    dot += state[index] * normalized_query[index];
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
      int channel = 0;
#if defined(__aarch64__)
      // Production Qwen3.5 uses kernel_size 4, so the history is three taps
      // wide and both the [C, K] weights and the [B, C, K - 1] history are
      // contiguous across channels. vld3/vld4 deinterleave four adjacent
      // channels into per-tap lanes, which lets the history shift happen in
      // registers instead of two scalar loads and two scalar stores per
      // element.
      //
      // The accumulation order matches the scalar body below exactly: a
      // rounded multiply by the newest tap, then taps 0, 1, 2 fused in
      // ascending order. Compilers contract the scalar `value += state * weight`
      // into an FMA, so vfmaq_f32 reproduces it bitwise. The focused
      // convolution oracle asserts that equality per toolchain rather than
      // assuming it.
      if (kernel_size == 4) {
        const std::size_t token_base = (static_cast<std::size_t>(batch) * sequence_length + token) * channels;
        const std::size_t batch_state_base = static_cast<std::size_t>(batch) * channels * state_width;
        for (; channel + 4 <= channels; channel += 4) {
          float* state_block = state + batch_state_base + static_cast<std::size_t>(channel) * state_width;
          const float32x4x3_t history = vld3q_f32(state_block);
          const float32x4x4_t taps = vld4q_f32(weight + static_cast<std::size_t>(channel) * kernel_size);
          const float32x4_t current = vld1q_f32(input + token_base + channel);

          float32x4_t value = vmulq_f32(current, taps.val[3]);
          value = vfmaq_f32(value, history.val[0], taps.val[0]);
          value = vfmaq_f32(value, history.val[1], taps.val[1]);
          value = vfmaq_f32(value, history.val[2], taps.val[2]);
          vst1q_f32(output + token_base + channel, value);

          float32x4x3_t shifted;
          shifted.val[0] = history.val[1];
          shifted.val[1] = history.val[2];
          shifted.val[2] = current;
          vst3q_f32(state_block, shifted);
        }
      }
#endif
      for (; channel < channels; ++channel) {
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

void depthwiseCausalConvHistoryFirstF32(const float* input, const float* weight, float* state, float* output,
                                        int batch_size, int sequence_length, int channels, int kernel_size) {
  if (input == nullptr || weight == nullptr || state == nullptr || output == nullptr) {
    throw std::invalid_argument("History-first depthwise causal convolution received a null pointer");
  }
  if (batch_size <= 0 || sequence_length <= 0 || channels <= 0 || kernel_size <= 1) {
    throw std::invalid_argument("History-first depthwise causal convolution received an invalid shape");
  }

  const int state_width = kernel_size - 1;
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int token = 0; token < sequence_length; ++token) {
      int channel = 0;
#if defined(__aarch64__)
      // LFM2.5 uses K=3. Four adjacent channels are deinterleaved into the
      // two history taps and three weights while input/output stay contiguous.
      // The three FMA steps intentionally match CPUConv1D's k=0,1,2 order.
      if (kernel_size == 3) {
        const std::size_t token_base = (static_cast<std::size_t>(batch) * sequence_length + token) * channels;
        const std::size_t batch_state_base = static_cast<std::size_t>(batch) * channels * state_width;
        for (; channel + 4 <= channels; channel += 4) {
          float* state_block = state + batch_state_base + static_cast<std::size_t>(channel) * state_width;
          const float32x4x2_t history = vld2q_f32(state_block);
          const float32x4x3_t taps = vld3q_f32(weight + static_cast<std::size_t>(channel) * kernel_size);
          const float32x4_t current = vld1q_f32(input + token_base + channel);

          float32x4_t value = vdupq_n_f32(0.0F);
          value = vfmaq_f32(value, history.val[0], taps.val[0]);
          value = vfmaq_f32(value, history.val[1], taps.val[1]);
          value = vfmaq_f32(value, current, taps.val[2]);
          vst1q_f32(output + token_base + channel, value);

          float32x4x2_t shifted;
          shifted.val[0] = history.val[1];
          shifted.val[1] = current;
          vst2q_f32(state_block, shifted);
        }
      }
#endif
      for (; channel < channels; ++channel) {
        const std::size_t state_base = (static_cast<std::size_t>(batch) * channels + channel) * state_width;
        const std::size_t input_index = (static_cast<std::size_t>(batch) * sequence_length + token) * channels + channel;
        const std::size_t weight_base = static_cast<std::size_t>(channel) * kernel_size;

        float value = 0.0F;
        for (int tap = 0; tap < state_width; ++tap) { value += state[state_base + tap] * weight[weight_base + tap]; }
        value += input[input_index] * weight[weight_base + state_width];
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
  gatedDeltaRuleF32(q, k, v, a, b, a_log, dt_bias, state, output, batch_size, sequence_length, num_key_heads, num_value_heads,
                    key_head_dim, value_head_dim, ::mllm::Context::instance().getCpuOpThreads());
}

void gatedDeltaRuleF32(const float* q, const float* k, const float* v, const float* a, const float* b, const float* a_log,
                       const float* dt_bias, float* state, float* output, int batch_size, int sequence_length,
                       int num_key_heads, int num_value_heads, int key_head_dim, int value_head_dim, int thread_count) {
  if (q == nullptr || k == nullptr || v == nullptr || a == nullptr || b == nullptr || a_log == nullptr || dt_bias == nullptr
      || state == nullptr || output == nullptr) {
    throw std::invalid_argument("Gated delta rule received a null pointer");
  }
  if (batch_size <= 0 || sequence_length <= 0 || num_key_heads <= 0 || num_value_heads <= 0 || key_head_dim <= 0
      || value_head_dim <= 0 || num_value_heads % num_key_heads != 0 || thread_count <= 0) {
    throw std::invalid_argument("Gated delta rule received an invalid shape");
  }

  const int key_head_repeats = num_value_heads / num_key_heads;
  const float query_dim_scale = 1.0F / std::sqrt(static_cast<float>(key_head_dim));
  const bool share_pre_normalized_heads = key_head_repeats > 1;
  const bool use_pre_normalized_heads =
      share_pre_normalized_heads || (sequence_length > 1 && key_head_dim <= kMaxStackNormalizedHeadDim);
  std::vector<float> shared_normalized_query;
  std::vector<float> shared_normalized_key;

  if (share_pre_normalized_heads) {
    const std::size_t qk_head_count =
        static_cast<std::size_t>(batch_size) * sequence_length * static_cast<std::size_t>(num_key_heads);
    const std::size_t qk_element_count = qk_head_count * key_head_dim;
    shared_normalized_query.resize(qk_element_count);
    shared_normalized_key.resize(qk_element_count);

    for (std::size_t qk_head_index = 0; qk_head_index < qk_head_count; ++qk_head_index) {
      const std::size_t qk_base = qk_head_index * key_head_dim;
      float query_norm_sq = 0.0F;
      float key_norm_sq = 0.0F;
      squaredNorms(q + qk_base, k + qk_base, key_head_dim, query_norm_sq, key_norm_sq);
      const float query_scale = query_dim_scale / std::sqrt(query_norm_sq + 1.0e-6F);
      const float key_scale = 1.0F / std::sqrt(key_norm_sq + 1.0e-6F);
      normalizeQueryAndKey(q + qk_base, k + qk_base, query_scale, key_scale, shared_normalized_query.data() + qk_base,
                           shared_normalized_key.data() + qk_base, key_head_dim);
    }
  }

  const int task_count = batch_size * num_value_heads;
  const std::size_t recurrence_work = static_cast<std::size_t>(task_count) * sequence_length * value_head_dim * key_head_dim;
  const int parallel_lanes = std::min({thread_count, task_count, availableCpuCount(thread_count), kMaxParallelGDNLanes});
  const bool use_parallel = parallel_lanes > 1 && sequence_length > 1 && recurrence_work >= kMinParallelGDNWork;

  const auto run_task = [&](int combined_task, std::array<float, kMaxStackNormalizedHeadDim>& normalized_query,
                            std::array<float, kMaxStackNormalizedHeadDim>& normalized_key) {
    const int batch = combined_task / num_value_heads;
    const int value_head = combined_task % num_value_heads;
    const int key_head = value_head / key_head_repeats;
    const std::size_t state_base =
        (static_cast<std::size_t>(batch) * num_value_heads + value_head) * value_head_dim * key_head_dim;

    for (int token = 0; token < sequence_length; ++token) {
      const std::size_t gate_index = (static_cast<std::size_t>(batch) * sequence_length + token) * num_value_heads + value_head;
      const std::size_t qk_base =
          ((static_cast<std::size_t>(batch) * sequence_length + token) * num_key_heads + key_head) * key_head_dim;
      const std::size_t value_base =
          ((static_cast<std::size_t>(batch) * sequence_length + token) * num_value_heads + value_head) * value_head_dim;

      float query_scale = 0.0F;
      float key_scale = 0.0F;
      const float* normalized_query_ptr = nullptr;
      const float* normalized_key_ptr = nullptr;
      if (share_pre_normalized_heads) {
        normalized_query_ptr = shared_normalized_query.data() + qk_base;
        normalized_key_ptr = shared_normalized_key.data() + qk_base;
      } else {
        float query_norm_sq = 0.0F;
        float key_norm_sq = 0.0F;
        squaredNorms(q + qk_base, k + qk_base, key_head_dim, query_norm_sq, key_norm_sq);
        query_scale = query_dim_scale / std::sqrt(query_norm_sq + 1.0e-6F);
        key_scale = 1.0F / std::sqrt(key_norm_sq + 1.0e-6F);
        if (use_pre_normalized_heads) {
          normalizeQueryAndKey(q + qk_base, k + qk_base, query_scale, key_scale, normalized_query.data(), normalized_key.data(),
                               key_head_dim);
          normalized_query_ptr = normalized_query.data();
          normalized_key_ptr = normalized_key.data();
        }
      }

      const float gate = -std::exp(a_log[value_head]) * stableSoftplus(a[gate_index] + dt_bias[value_head]);
      const float decay = std::exp(gate);
      const float beta = stableSigmoid(b[gate_index]);

      for (int value_dim = 0; value_dim < value_head_dim; ++value_dim) {
        float* state_row = state + state_base + static_cast<std::size_t>(value_dim) * key_head_dim;
        if (use_pre_normalized_heads) {
          const float state_dot_key = decayAndDotNormalized(state_row, normalized_key_ptr, decay, key_head_dim);
          const float delta = (v[value_base + value_dim] - state_dot_key) * beta;
          output[value_base + value_dim] =
              updateAndDotNormalized(state_row, normalized_key_ptr, normalized_query_ptr, delta, key_head_dim);
        } else {
          const float state_dot_key = decayAndDot(state_row, k + qk_base, decay, key_scale, key_head_dim);
          const float delta = (v[value_base + value_dim] - state_dot_key) * beta;
          output[value_base + value_dim] =
              updateAndDot(state_row, k + qk_base, q + qk_base, delta, key_scale, query_scale, key_head_dim);
        }
      }
    }
  };

  const int scheduled_lanes = use_parallel ? parallel_lanes : 1;
  const auto run_lane = [&](int lane) {
    std::array<float, kMaxStackNormalizedHeadDim> normalized_query = {};
    std::array<float, kMaxStackNormalizedHeadDim> normalized_key = {};
    for (int combined_task = lane; combined_task < task_count; combined_task += scheduled_lanes) {
      run_task(combined_task, normalized_query, normalized_key);
    }
  };

  MLLM_CONDITIONAL_PARALLEL_FOR(use_parallel, scheduled_lanes, lane, 0, scheduled_lanes, 1,
                                { run_lane(static_cast<int>(lane)); });
}

}  // namespace mllm::cpu::gdn
