// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/GroupedQueryAttentionDecodeOp.hpp"
#include "mllm/nn/Layer.hpp"

namespace mllm::nn {

class GroupedQueryAttentionDecode : public Layer {
 public:
  GroupedQueryAttentionDecode();

  explicit GroupedQueryAttentionDecode(const aops::GroupedQueryAttentionDecodeOpOptions& options);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
