// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/CausalDepthwiseConv1DOp.hpp"
#include "mllm/nn/Layer.hpp"

namespace mllm::nn {

class CausalDepthwiseConv1D : public Layer {
 public:
  CausalDepthwiseConv1D();
  explicit CausalDepthwiseConv1D(const aops::CausalDepthwiseConv1DOpOptions& options);
  CausalDepthwiseConv1D(int32_t channels, int32_t kernel_size, bool bias, bool state_inplace,
                        aops::CausalDepthwiseConv1DAccumulationOrder accumulation_order);

  [[nodiscard]] Tensor weight() const;
  [[nodiscard]] Tensor bias() const;

  MLLM_LAYER_ANY_INPUTS_2_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
