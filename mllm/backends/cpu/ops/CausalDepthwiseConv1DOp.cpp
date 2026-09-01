// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/CausalDepthwiseConv1DOp.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "mllm/backends/cpu/kernels/common/causal_conv/depthwise_causal_conv.hpp"
#include "mllm/backends/cpu/kernels/common/gdn/gated_delta_net.hpp"

namespace mllm::cpu {

CPUCausalDepthwiseConv1DOp::CPUCausalDepthwiseConv1DOp(const aops::CausalDepthwiseConv1DOpOptions& options)
    : aops::CausalDepthwiseConv1DOp(options) {}

void CPUCausalDepthwiseConv1DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  for (const auto& input : inputs) {
    if (!input.isContiguous()) { throw std::invalid_argument("CausalDepthwiseConv1D CPU inputs must be contiguous"); }
  }
  if (weight_.isNil() || !weight_.isContiguous() || weight_.dtype() != kFloat32 || weight_.device() != kCPU
      || weight_.shape() != Tensor::shape_t{options_.channels, 1, options_.kernel_size}) {
    throw std::invalid_argument("CausalDepthwiseConv1D requires contiguous float32 [C, 1, K] weights");
  }

  const auto& input = inputs[0];
  const auto& state = inputs[1];
  auto& output = outputs[0];
  auto& updated_state = outputs[1];
  if (!options_.state_inplace) { std::memcpy(updated_state.ptr<float>(), state.ptr<float>(), state.bytes()); }

  const bool history_first = options_.accumulation_order == aops::CausalDepthwiseConv1DAccumulationOrder::kHistoryFirst;
  if (history_first) {
    causal_conv::depthwiseCausalConvHistoryFirstF32(input.ptr<float>(), weight_.ptr<float>(), updated_state.ptr<float>(),
                                                    output.ptr<float>(), input.shape()[0], input.shape()[1],
                                                    input.shape()[2], options_.kernel_size);
  } else {
    gdn::depthwiseCausalConvF32(input.ptr<float>(), weight_.ptr<float>(), updated_state.ptr<float>(), output.ptr<float>(),
                                input.shape()[0], input.shape()[1], input.shape()[2], options_.kernel_size);
  }

  static const bool trace_activation = [] {
    const char* value = std::getenv("MLLM_CAUSAL_CONV1D_TRACE");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  if (trace_activation) {
    // One marker per accumulation order, so a device receipt shows which
    // kernel the operation actually reached.
    static std::atomic<uint32_t> activated_orders{0};
    const uint32_t order_bit = 1U << static_cast<uint32_t>(history_first);
    if ((activated_orders.fetch_or(order_bit, std::memory_order_relaxed) & order_bit) == 0) {
      std::fprintf(stderr, "MLLM_CAUSAL_CONV1D_ACTIVATED order=%s k=%d channels=%d\n",
                   aops::causalDepthwiseConv1DAccumulationOrder2Str(options_.accumulation_order), options_.kernel_size,
                   options_.channels);
    }
  }

  if (options_.bias) {
    if (bias_.isNil() || !bias_.isContiguous() || bias_.dtype() != kFloat32
        || bias_.shape() != Tensor::shape_t{options_.channels}) {
      throw std::invalid_argument("CausalDepthwiseConv1D bias must be contiguous float32 [C]");
    }
    for (int32_t batch = 0; batch < input.shape()[0]; ++batch) {
      for (int32_t token = 0; token < input.shape()[1]; ++token) {
        auto* row = output.offsettedPtr<float>({batch, token, 0});
        for (int32_t channel = 0; channel < options_.channels; ++channel) { row[channel] += bias_.ptr<float>()[channel]; }
      }
    }
  }
}

}  // namespace mllm::cpu
