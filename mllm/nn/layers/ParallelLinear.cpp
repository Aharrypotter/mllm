// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/ParallelLinear.hpp"

#include <utility>

#include "mllm/core/aops/ParallelLinearOp.hpp"

namespace mllm::nn {

ParallelLinear::ParallelLinear() : Layer(OpTypes::kParallelLinear, aops::ParallelLinearOpOptions{}) {}

ParallelLinear::ParallelLinear(int32_t in_channels, std::vector<int32_t> out_channels,
                               std::vector<std::string> projection_names, bool bias, aops::LinearImplTypes impl_type,
                               int32_t decode_thread_cap, int32_t prefill_thread_cap)
    : Layer(OpTypes::kParallelLinear, aops::ParallelLinearOpOptions{.in_channels = in_channels,
                                                                    .out_channels = std::move(out_channels),
                                                                    .projection_names = std::move(projection_names),
                                                                    .bias = bias,
                                                                    .impl_type = impl_type,
                                                                    .kai_w4a32_decode_thread_cap = decode_thread_cap,
                                                                    .kai_w4a32_prefill_thread_cap = prefill_thread_cap}) {}

}  // namespace mllm::nn
