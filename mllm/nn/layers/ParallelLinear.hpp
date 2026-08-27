// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/nn/Layer.hpp"

namespace mllm::nn {

class ParallelLinear : public Layer {
 public:
  ParallelLinear();

  ParallelLinear(int32_t in_channels, std::vector<int32_t> out_channels, std::vector<std::string> projection_names,
                 bool bias = true, aops::LinearImplTypes impl_type = aops::LinearImplTypes::kDefault,
                 int32_t decode_thread_cap = 0, int32_t prefill_thread_cap = 0);

  MLLM_LAYER_ANY_INPUTS_ANY_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
