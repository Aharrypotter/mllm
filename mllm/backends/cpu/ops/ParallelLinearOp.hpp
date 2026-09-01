// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "mllm/backends/cpu/ops/LinearOp.hpp"
#include "mllm/core/aops/ParallelLinearOp.hpp"

namespace mllm::cpu {

class CPUParallelLinearOp final : public aops::ParallelLinearOp {
 public:
  explicit CPUParallelLinearOp(const aops::ParallelLinearOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  bool tryForwardSharedInputKai(const Tensor& input, std::vector<Tensor>& outputs);
  Tensor acquireKaiWorkspace(int32_t workspace_size, int m);

  std::vector<std::unique_ptr<CPULinearOp>> fallback_ops_;
  // Only the decode-sized (m == 1) buffer is retained; a prefill workspace is
  // two orders of magnitude larger and must not stay resident for the rest of
  // the process, which CPULinearOp already avoids the same way.
  Tensor kai_decode_workspace_;
};

class CPUParallelLinearOpFactory : public TypedOpFactory<OpTypes::kParallelLinear, aops::ParallelLinearOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ParallelLinearOpOptions& options) override {
    return std::make_shared<CPUParallelLinearOp>(options);
  }
};

}  // namespace mllm::cpu
