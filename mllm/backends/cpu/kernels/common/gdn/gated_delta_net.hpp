// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

namespace mllm::cpu::gdn {

// Stateful depthwise causal convolution for [B, S, C] input. State uses
// [B, C, K - 1] layout and is updated in place.
void depthwiseCausalConvF32(const float* input, const float* weight, float* state, float* output, int batch_size,
                            int sequence_length, int channels, int kernel_size);

// Same [B, S, C] / [B, C, K - 1] state contract, with the accumulation order
// used by CPUConv1D: zero, historical taps in ascending order, then current.
// Keeping this explicit lets callers retain bitwise-sensitive model semantics.
void depthwiseCausalConvHistoryFirstF32(const float* input, const float* weight, float* state, float* output,
                                        int batch_size, int sequence_length, int channels, int kernel_size);

// Gated-delta recurrence. Each [batch, value_head] state is independent and
// may run in parallel, while tokens within one state remain strictly ordered.
//
// q/k:   [B, S, Hk, Dk]
// v:     [B, S, Hv, Dv]
// a/b:   [B, S, Hv]
// state: [B, Hv, Dv, Dk], updated in place
// output:[B, S, Hv, Dv]
void gatedDeltaRuleF32(const float* q, const float* k, const float* v, const float* a, const float* b, const float* a_log,
                       const float* dt_bias, float* state, float* output, int batch_size, int sequence_length,
                       int num_key_heads, int num_value_heads, int key_head_dim, int value_head_dim);

void gatedDeltaRuleF32(const float* q, const float* k, const float* v, const float* a, const float* b, const float* a_log,
                       const float* dt_bias, float* state, float* output, int batch_size, int sequence_length,
                       int num_key_heads, int num_value_heads, int key_head_dim, int value_head_dim, int thread_count);

}  // namespace mllm::cpu::gdn
