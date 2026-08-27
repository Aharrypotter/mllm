// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/CausalDepthwiseConv1D.hpp"

namespace mllm::nn {

CausalDepthwiseConv1D::CausalDepthwiseConv1D()
    : Layer(OpTypes::kCausalDepthwiseConv1D, aops::CausalDepthwiseConv1DOpOptions{}) {}

CausalDepthwiseConv1D::CausalDepthwiseConv1D(int32_t channels, int32_t kernel_size, bool bias, bool state_inplace,
                                             aops::CausalDepthwiseConv1DAccumulationOrder accumulation_order)
    : Layer(OpTypes::kCausalDepthwiseConv1D, aops::CausalDepthwiseConv1DOpOptions{.channels = channels,
                                                                                  .kernel_size = kernel_size,
                                                                                  .bias = bias,
                                                                                  .state_inplace = state_inplace,
                                                                                  .accumulation_order = accumulation_order}) {}

Tensor CausalDepthwiseConv1D::weight() const {
  return std::static_pointer_cast<aops::CausalDepthwiseConv1DOp>(impl()->getInstancedOp())->weight();
}

Tensor CausalDepthwiseConv1D::bias() const {
  return std::static_pointer_cast<aops::CausalDepthwiseConv1DOp>(impl()->getInstancedOp())->bias();
}

}  // namespace mllm::nn
