// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/GroupedQueryAttentionOp.hpp"
#include "mllm/nn/Layer.hpp"

namespace mllm::nn {

class GroupedQueryAttention : public Layer {
 public:
  GroupedQueryAttention();
  explicit GroupedQueryAttention(const aops::GroupedQueryAttentionOpOptions& options);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
