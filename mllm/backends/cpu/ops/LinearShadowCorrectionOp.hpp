// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/LinearShadowCorrectionOp.hpp"

namespace mllm::cpu {

class CPULinearShadowCorrectionOp final : public aops::LinearShadowCorrectionOp {
 public:
  explicit CPULinearShadowCorrectionOp(const aops::LinearShadowCorrectionOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPULinearShadowCorrectionOpFactory
    : public TypedOpFactory<OpTypes::kLinearShadowCorrection, aops::LinearShadowCorrectionOpOptions> {
 protected:
  std::shared_ptr<BaseOp> createOpImpl(const aops::LinearShadowCorrectionOpOptions& options) override {
    return std::make_shared<CPULinearShadowCorrectionOp>(options);
  }
};

}  // namespace mllm::cpu
