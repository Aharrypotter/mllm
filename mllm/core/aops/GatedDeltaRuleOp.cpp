// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/GatedDeltaRuleOp.hpp"

#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

GatedDeltaRuleOp::GatedDeltaRuleOp(const GatedDeltaRuleOpOptions& options)
    : BaseOp(OpTypes::kGatedDeltaRule), options_(options) {}

void GatedDeltaRuleOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void GatedDeltaRuleOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto* ir_ctx = static_cast<ir::IRContext*>(trace_context);
  const auto input_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  const auto output_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::GatedDeltaRuleOp>(shared_from_this(), input_irs, output_irs);
}

void GatedDeltaRuleOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("GatedDeltaRuleOp::forward not implemented in aops base.");
}

void GatedDeltaRuleOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 8);
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  const auto& a = inputs[3];
  const auto& b = inputs[4];
  const auto& a_log = inputs[5];
  const auto& dt_bias = inputs[6];
  const auto& state = inputs[7];

  MLLM_RT_ASSERT_EQ(q.rank(), 4);
  MLLM_RT_ASSERT_EQ(k.rank(), 4);
  MLLM_RT_ASSERT_EQ(v.rank(), 4);
  const int32_t batch = q.shape()[0];
  const int32_t sequence = q.shape()[1];
  const int32_t key_heads = q.shape()[2];
  const int32_t key_dim = q.shape()[3];
  const int32_t value_heads = v.shape()[2];
  const int32_t value_dim = v.shape()[3];
  MLLM_RT_ASSERT_EQ(k.shape(), q.shape());
  MLLM_RT_ASSERT_EQ(v.shape()[0], batch);
  MLLM_RT_ASSERT_EQ(v.shape()[1], sequence);
  MLLM_RT_ASSERT(key_heads > 0 && value_heads > 0 && value_heads % key_heads == 0);
  MLLM_RT_ASSERT_EQ(a.shape(), (Tensor::shape_t{batch, sequence, value_heads}));
  MLLM_RT_ASSERT_EQ(b.shape(), (Tensor::shape_t{batch, sequence, value_heads}));
  MLLM_RT_ASSERT_EQ(a_log.numel(), static_cast<std::size_t>(value_heads));
  MLLM_RT_ASSERT_EQ(dt_bias.numel(), static_cast<std::size_t>(value_heads));
  MLLM_RT_ASSERT_EQ(state.shape(), (Tensor::shape_t{batch, value_heads, value_dim, key_dim}));
  for (const auto& tensor : inputs) {
    MLLM_RT_ASSERT_EQ(tensor.dtype(), kFloat32);
    MLLM_RT_ASSERT_EQ(tensor.device(), q.device());
  }

  outputs.emplace_back(Tensor::empty({batch, sequence, value_heads, value_dim}, q.dtype(), q.device()));
  outputs.emplace_back(options_.state_inplace ? state : Tensor::empty(state.shape(), state.dtype(), state.device()));
}

void GatedDeltaRuleOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.state_inplace) {
    outputs[0].alloc();
  } else {
    BaseOp::setup(inputs, outputs);
  }
}

}  // namespace mllm::aops
