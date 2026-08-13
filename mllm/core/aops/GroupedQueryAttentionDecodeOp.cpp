// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/GroupedQueryAttentionDecodeOp.hpp"

#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

GroupedQueryAttentionDecodeOp::GroupedQueryAttentionDecodeOp(const GroupedQueryAttentionDecodeOpOptions& options)
    : BaseOp(OpTypes::kGroupedQueryAttentionDecode), options_(options) {}

void GroupedQueryAttentionDecodeOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void GroupedQueryAttentionDecodeOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                                          std::vector<Tensor>& outputs) {
  auto* ir_ctx = static_cast<ir::IRContext*>(trace_context);
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::GroupedQueryAttentionDecodeOp>(shared_from_this(), i_irs, o_irs);
}

void GroupedQueryAttentionDecodeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("GroupedQueryAttentionDecodeOp::forward not implemented in aops base.");
}

void GroupedQueryAttentionDecodeOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 3);
  const auto& query = inputs[0];
  const auto& key = inputs[1];
  const auto& value = inputs[2];
  const auto& q_shape = query.shape();
  const auto& k_shape = key.shape();
  const auto& v_shape = value.shape();

  MLLM_RT_ASSERT_EQ(q_shape.size(), 4);
  MLLM_RT_ASSERT_EQ(k_shape.size(), 4);
  MLLM_RT_ASSERT_EQ(v_shape.size(), 4);
  MLLM_RT_ASSERT_EQ(q_shape[0], k_shape[0]);
  MLLM_RT_ASSERT_EQ(q_shape[0], v_shape[0]);
  MLLM_RT_ASSERT_EQ(q_shape[2], 1);
  MLLM_RT_ASSERT_EQ(k_shape[1], v_shape[1]);
  MLLM_RT_ASSERT_EQ(k_shape[2], v_shape[2]);
  MLLM_RT_ASSERT_EQ(q_shape[3], k_shape[3]);
  MLLM_RT_ASSERT(q_shape[1] > 0 && k_shape[1] > 0 && q_shape[1] % k_shape[1] == 0);
  MLLM_RT_ASSERT(k_shape[2] > 0 && v_shape[3] > 0);
  MLLM_RT_ASSERT_EQ(query.dtype(), kFloat32);
  MLLM_RT_ASSERT_EQ(query.dtype(), key.dtype());
  MLLM_RT_ASSERT_EQ(query.dtype(), value.dtype());
  MLLM_RT_ASSERT_EQ(query.device(), key.device());
  MLLM_RT_ASSERT_EQ(query.device(), value.device());

  outputs.emplace_back(Tensor::empty({q_shape[0], q_shape[1], 1, v_shape[3]}, query.dtype(), query.device()));
}

void GroupedQueryAttentionDecodeOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::aops
