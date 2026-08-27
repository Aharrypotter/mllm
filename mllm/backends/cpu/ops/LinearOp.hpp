// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/LinearOp.hpp"

namespace mllm::cpu {

class CPULinearOp final : public aops::LinearOp {
 public:
  explicit CPULinearOp(const aops::LinearOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  Tensor acquireKaiWorkspace(int32_t workspace_size, int m);

  [[nodiscard]] int kaiW4A32ThreadCount(int m) const;

  Tensor kai_decode_workspace_;
};

class CPULinearOpFactory : public TypedOpFactory<OpTypes::kLinear, aops::LinearOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::LinearOpOptions& options) override {
    return std::make_shared<CPULinearOp>(options);
  }
};

}  // namespace mllm::cpu
