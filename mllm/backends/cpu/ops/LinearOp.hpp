// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/LinearOp.hpp"

namespace mllm::cpu {

namespace detail {

constexpr bool shouldUseKaiW4A32I8mmPrefill(int m, bool disabled, bool cpu_supports_i8mm) {
  return m >= 4 && !disabled && cpu_supports_i8mm;
}

}  // namespace detail

class CPULinearOp final : public aops::LinearOp {
 public:
  explicit CPULinearOp(const aops::LinearOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  Tensor acquireKaiWorkspace(int32_t workspace_size, int m);

  Tensor kai_decode_workspace_;
};

class CPULinearOpFactory : public TypedOpFactory<OpTypes::kLinear, aops::LinearOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::LinearOpOptions& options) override {
    return std::make_shared<CPULinearOp>(options);
  }
};

}  // namespace mllm::cpu
