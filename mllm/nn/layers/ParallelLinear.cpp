// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/ParallelLinear.hpp"

namespace mllm::nn {

ParallelLinear::ParallelLinear() : Layer(OpTypes::kParallelLinear, aops::ParallelLinearOpOptions{}) {}

ParallelLinear::ParallelLinear(const aops::ParallelLinearOpOptions& options) : Layer(OpTypes::kParallelLinear, options) {}

}  // namespace mllm::nn
