// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/ParallelLinearOp.hpp"

#include <cstdint>
#include <array>
#include <limits>

#include "mllm/backends/cpu/kernels/common/parallel_linear/shared_input.hpp"

namespace mllm::cpu {

CPUParallelLinearOp::CPUParallelLinearOp(const aops::ParallelLinearOpOptions& options) : aops::ParallelLinearOp(options) {
  fallback_ops_.reserve(options_.out_channels.size());
  for (const int32_t out_channels : options_.out_channels) {
    aops::LinearOpOptions child_options{.in_channels = options_.in_channels,
                                        .out_channels = out_channels,
                                        .bias = options_.bias,
                                        .impl_type = options_.impl_type,
                                        .kai_w4a32_decode_thread_cap = options_.kai_w4a32_decode_thread_cap,
                                        .kai_w4a32_prefill_thread_cap = options_.kai_w4a32_prefill_thread_cap};
    child_options.setThreads(options_.getThreads());
    fallback_ops_.push_back(std::make_unique<CPULinearOp>(child_options));
  }
}

void CPUParallelLinearOp::load(const ParameterFile::ptr_t& ploader) {
  aops::ParallelLinearOp::load(ploader);
  for (size_t index = 0; index < fallback_ops_.size(); ++index) {
    fallback_ops_[index]->weight() = weights_[index];
    if (options_.bias) { fallback_ops_[index]->bias() = biases_[index]; }
  }
}

Tensor CPUParallelLinearOp::acquireKaiWorkspace(int32_t workspace_size, int m) {
  if (m != 1) { return Tensor::empty({workspace_size}, kInt8, kCPU).alloc(); }

  if (kai_decode_workspace_.isNil() || kai_decode_workspace_.numel() < static_cast<size_t>(workspace_size)) {
    kai_decode_workspace_ = Tensor::empty({workspace_size}, kInt8, kCPU).alloc();
  }
  return kai_decode_workspace_;
}

bool CPUParallelLinearOp::tryForwardSharedInputKai(const Tensor& input, std::vector<Tensor>& outputs) {
  constexpr size_t kMaximumSharedProjections = 3;
  constexpr auto kRequiredImpl = aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32;

  if (input.isNil() || input.device() != kCPU || input.dtype() != kFloat32 || !input.isContiguous() || input.rank() < 2
      || input.size(-1) != options_.in_channels || options_.bias || options_.impl_type != kRequiredImpl
      || outputs.size() != weights_.size() || weights_.size() < 2 || weights_.size() > kMaximumSharedProjections) {
    return false;
  }
  for (size_t index = 0; index + 2 < input.shape().size(); ++index) {
    if (input.shape()[index] != 1) { return false; }
  }
  const int32_t m = input.size(-2);
  if (m <= 0) { return false; }
  for (size_t index = 0; index < weights_.size(); ++index) {
    if (weights_[index].isNil() || weights_[index].device() != kCPU || outputs[index].isNil()
        || outputs[index].dtype() != kFloat32 || outputs[index].device() != kCPU || !outputs[index].isContiguous()
        || outputs[index].rank() != input.rank() || outputs[index].size(-2) != m
        || outputs[index].size(-1) != options_.out_channels[index]) {
      return false;
    }
    for (size_t dimension = 0; dimension + 2 < input.shape().size(); ++dimension) {
      if (outputs[index].shape()[dimension] != input.shape()[dimension]) { return false; }
    }
  }

  const auto plan =
      parallel_linear::planKaiW4A32SharedInput(m, options_.in_channels, options_.getThreads(),
                                               options_.kai_w4a32_decode_thread_cap, options_.kai_w4a32_prefill_thread_cap);
  if (!plan.supported() || plan.workspace_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) { return false; }

  std::array<parallel_linear::SharedInputProjection, kMaximumSharedProjections> projections{};
  for (size_t index = 0; index < weights_.size(); ++index) {
    projections[index] = {
        .dst = outputs[index].ptr<mllm_fp32_t>(),
        .packed_weight_bias = reinterpret_cast<const uint8_t*>(weights_[index].ptr<mllm_byte_t>()),
        .n = options_.out_channels[index],
    };
  }

  auto workspace = acquireKaiWorkspace(static_cast<int32_t>(plan.workspace_size), m);
  return parallel_linear::runKaiW4A32SharedInput(plan, input.ptr<mllm_fp32_t>(), projections.data(), weights_.size(),
                                                 workspace.ptr<void>(), m, options_.in_channels);
}

void CPUParallelLinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& input = inputs[0];
  if (tryForwardSharedInputKai(input, outputs)) { return; }
  for (size_t index = 0; index < fallback_ops_.size(); ++index) {
    // Keep fallback Linear execution aligned if the parent op's thread policy
    // is adjusted after construction.
    fallback_ops_[index]->options().setThreads(options_.getThreads());
    std::vector<Tensor> child_outputs = {outputs[index]};
    fallback_ops_[index]->forward({input}, child_outputs);
  }
}

}  // namespace mllm::cpu
