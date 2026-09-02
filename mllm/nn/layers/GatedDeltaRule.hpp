// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/GatedDeltaRuleOp.hpp"
#include "mllm/nn/Layer.hpp"

namespace mllm::nn {

class GatedDeltaRule : public Layer {
 public:
  GatedDeltaRule();
  explicit GatedDeltaRule(const aops::GatedDeltaRuleOpOptions& options);
  explicit GatedDeltaRule(bool state_inplace);

  MLLM_LAYER_ANY_INPUTS_2_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
