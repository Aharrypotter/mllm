// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/GatedDeltaRule.hpp"

namespace mllm::nn {

GatedDeltaRule::GatedDeltaRule() : Layer(OpTypes::kGatedDeltaRule, aops::GatedDeltaRuleOpOptions{}) {}

GatedDeltaRule::GatedDeltaRule(const aops::GatedDeltaRuleOpOptions& options) : Layer(OpTypes::kGatedDeltaRule, options) {}

GatedDeltaRule::GatedDeltaRule(bool state_inplace)
    : Layer(OpTypes::kGatedDeltaRule, aops::GatedDeltaRuleOpOptions{.state_inplace = state_inplace}) {}

}  // namespace mllm::nn
