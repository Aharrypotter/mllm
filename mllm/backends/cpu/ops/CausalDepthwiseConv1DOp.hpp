// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/CausalDepthwiseConv1DOp.hpp"

namespace mllm::cpu {

class CPUCausalDepthwiseConv1DOp final : public aops::CausalDepthwiseConv1DOp {
 public:
  explicit CPUCausalDepthwiseConv1DOp(const aops::CausalDepthwiseConv1DOpOptions& options);
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUCausalDepthwiseConv1DOpFactory
    : public TypedOpFactory<OpTypes::kCausalDepthwiseConv1D, aops::CausalDepthwiseConv1DOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::CausalDepthwiseConv1DOpOptions& options) override {
    return std::make_shared<CPUCausalDepthwiseConv1DOp>(options);
  }
};

}  // namespace mllm::cpu
