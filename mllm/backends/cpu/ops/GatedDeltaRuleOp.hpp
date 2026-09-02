// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/GatedDeltaRuleOp.hpp"

namespace mllm::cpu {

class CPUGatedDeltaRuleOp final : public aops::GatedDeltaRuleOp {
 public:
  explicit CPUGatedDeltaRuleOp(const aops::GatedDeltaRuleOpOptions& options);
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUGatedDeltaRuleOpFactory : public TypedOpFactory<OpTypes::kGatedDeltaRule, aops::GatedDeltaRuleOpOptions> {
 protected:
  std::shared_ptr<BaseOp> createOpImpl(const aops::GatedDeltaRuleOpOptions& options) override {
    return std::make_shared<CPUGatedDeltaRuleOp>(options);
  }
};

}  // namespace mllm::cpu
