// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

namespace mllm::cpu::causal_conv {

// Stateful depthwise causal convolution for [B, S, C] input with a
// [B, C, K - 1] history state that is updated in place.
//
// Accumulation order is zero, historical taps in ascending order, then the
// current sample, matching CPUConv1D. Callers whose generation contract is
// bitwise sensitive depend on this order, so it is part of the contract
// rather than an implementation detail.
void depthwiseCausalConvHistoryFirstF32(const float* input, const float* weight, float* state, float* output,
                                        int batch_size, int sequence_length, int channels, int kernel_size);

}  // namespace mllm::cpu::causal_conv
