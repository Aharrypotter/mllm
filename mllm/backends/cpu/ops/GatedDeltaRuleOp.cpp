// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/GatedDeltaRuleOp.hpp"

#include <cstring>

#include "mllm/backends/cpu/kernels/common/gdn/gated_delta_net.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::cpu {

CPUGatedDeltaRuleOp::CPUGatedDeltaRuleOp(const aops::GatedDeltaRuleOpOptions& options) : aops::GatedDeltaRuleOp(options) {}

void CPUGatedDeltaRuleOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  for (const auto& input : inputs) { MLLM_RT_ASSERT(input.isContiguous()); }
  const auto& q = inputs[0];
  const auto& v = inputs[2];
  auto& output = outputs[0];
  auto& updated_state = outputs[1];
  if (!options_.state_inplace) { std::memcpy(updated_state.ptr<float>(), inputs[7].ptr<float>(), inputs[7].bytes()); }
  gdn::gatedDeltaRuleF32(inputs[0].ptr<float>(), inputs[1].ptr<float>(), inputs[2].ptr<float>(), inputs[3].ptr<float>(),
                         inputs[4].ptr<float>(), inputs[5].ptr<float>(), inputs[6].ptr<float>(), updated_state.ptr<float>(),
                         output.ptr<float>(), q.shape()[0], q.shape()[1], q.shape()[2], v.shape()[2], q.shape()[3],
                         v.shape()[3], options_.getThreads());
}

}  // namespace mllm::cpu
