// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/common/causal_conv/depthwise_causal_conv.hpp"

#include <cstddef>
#include <stdexcept>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace mllm::cpu::causal_conv {

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
      // The K=3 fast path deinterleaves four adjacent channels into the two
      // history taps and three weights while input/output stay contiguous.
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

}  // namespace mllm::cpu::causal_conv
