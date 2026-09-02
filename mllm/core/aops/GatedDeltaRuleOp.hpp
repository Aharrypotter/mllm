// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct GatedDeltaRuleOpOptions : public BaseOpOptions<GatedDeltaRuleOpOptions> {
  bool state_inplace = false;
};

// Stateful grouped-head gated delta recurrence.
// Inputs: q/k [B, S, Hk, Dk], v [B, S, Hv, Dv], a/b [B, S, Hv],
// A_log/dt_bias [Hv], state [B, Hv, Dv, Dk].
// Outputs: output [B, S, Hv, Dv], updated_state [B, Hv, Dv, Dk].
class GatedDeltaRuleOp : public BaseOp {
 public:
  explicit GatedDeltaRuleOp(const GatedDeltaRuleOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;
  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  [[nodiscard]] const GatedDeltaRuleOpOptions& options() const { return options_; }

 protected:
  GatedDeltaRuleOpOptions options_;
};

}  // namespace mllm::aops
