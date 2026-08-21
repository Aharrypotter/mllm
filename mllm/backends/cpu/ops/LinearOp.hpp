// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/LinearOp.hpp"

namespace mllm::cpu {

namespace detail {

constexpr bool shouldUseKaiW4A32I8mmPrefill(int m, bool disabled, bool cpu_supports_i8mm) {
  return m >= 4 && !disabled && cpu_supports_i8mm;
}

bool shouldUseKaiW4A32I8mmPrefill(int m);

constexpr int kaiW4A32ThreadCount(int m, int requested_threads, int decode_thread_cap, int prefill_thread_cap) {
  // Every dynamic-input W4A32 KAI tile interprets the optional caps through
  // this helper; zero keeps the repository-wide requested thread count.
  const int cap = m == 1 ? decode_thread_cap : prefill_thread_cap;
  return cap > 0 && cap < requested_threads ? cap : requested_threads;
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
