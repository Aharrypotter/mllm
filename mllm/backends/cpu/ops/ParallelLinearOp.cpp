// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/ParallelLinearOp.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "mllm/backends/cpu/kernels/Kernels.hpp"

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

Tensor CPUParallelLinearOp::acquireKaiWorkspace(int32_t workspace_size) {
  if (kai_workspace_.isNil() || kai_workspace_.numel() < static_cast<size_t>(workspace_size)) {
    kai_workspace_ = Tensor::empty({workspace_size}, kInt8, kCPU).alloc();
  }
  return kai_workspace_;
}

bool CPUParallelLinearOp::tryForwardSharedInputKai(const Tensor& input, std::vector<Tensor>& outputs) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
  constexpr size_t kMaximumSharedProjections = 3;
  constexpr auto kRequiredImpl = aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32;
  using KaiHelper = ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk;
  constexpr auto kDecodeTile = KaiHelper::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32;
  constexpr auto kPrefillTile = KaiHelper::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32;

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
  const auto tile = m == 1 ? kDecodeTile : kPrefillTile;
  if (m > 1 && !detail::shouldUseKaiW4A32I8mmPrefill(m)) {
    // The generic shared-input dot-product prefill path has not passed the
    // mobile product screen; preserve the established per-projection fallback.
    return false;
  }
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

  const int32_t thread_count = detail::kaiW4A32ThreadCount(m, options_.getThreads(), options_.kai_w4a32_decode_thread_cap,
                                                           options_.kai_w4a32_prefill_thread_cap);
  std::array<KaiHelper::SharedInputProjection, kMaximumSharedProjections> projections{};
  for (size_t index = 0; index < weights_.size(); ++index) {
    projections[index] = {
        .dst = outputs[index].ptr<mllm_fp32_t>(),
        .packed_weight_bias = reinterpret_cast<const uint8_t*>(weights_[index].ptr<mllm_byte_t>()),
        .n = options_.out_channels[index],
    };
  }

  KaiHelper kai_helper;
  const size_t workspace_size = kai_helper.workspace_size(m, options_.in_channels, tile);
  if (workspace_size == 0 || workspace_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) { return false; }
  auto workspace = acquireKaiWorkspace(static_cast<int32_t>(workspace_size));
  if (!kai_helper.matmul_shared_input(input.ptr<mllm_fp32_t>(), projections.data(), weights_.size(), workspace.ptr<void>(), m,
                                      options_.in_channels, tile, thread_count)) {
    return false;
  }

  static const bool trace_activation = [] {
    const char* value = std::getenv("MLLM_KAI_SHARED_INPUT_TRACE");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  if (trace_activation) {
    const uint32_t projection_group = static_cast<uint32_t>(weights_.size() - 2);
    const uint32_t activation_bit = 1U << (2U * projection_group + static_cast<uint32_t>(m > 1));
    static std::atomic<uint32_t> activated_groups{0};
    const uint32_t previous = activated_groups.fetch_or(activation_bit, std::memory_order_relaxed);
    if ((previous & activation_bit) == 0) {
      std::fprintf(stderr, "MLLM_KAI_SHARED_INPUT_ACTIVATED rhs=%zu m=%d k=%d threads=%d tile=%s\n", weights_.size(), m,
                   options_.in_channels, thread_count, m == 1 ? "dotprod_1x8" : "i8mm_4x8");
    }
  }
  return true;
#else
  (void)input;
  (void)outputs;
  return false;
#endif
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
