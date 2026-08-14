// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "mllm/core/aops/ParallelLinearOp.hpp"
#include "mllm/nn/Layer.hpp"

namespace mllm::nn {

class ParallelLinear : public Layer {
 public:
  ParallelLinear();
  explicit ParallelLinear(const aops::ParallelLinearOpOptions& options);

  std::vector<Tensor> operator()(const Tensor& input) { return __main({input}); }
};

}  // namespace mllm::nn
