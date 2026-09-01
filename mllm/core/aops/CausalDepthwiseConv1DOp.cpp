// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/CausalDepthwiseConv1DOp.hpp"

#include <stdexcept>

#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

CausalDepthwiseConv1DOp::CausalDepthwiseConv1DOp(const CausalDepthwiseConv1DOpOptions& options)
    : BaseOp(OpTypes::kCausalDepthwiseConv1D), options_(options) {}

void CausalDepthwiseConv1DOp::load(const ParameterFile::ptr_t& ploader) {
  weight_ = ploader->pull(getName() + ".weight");
  if (options_.bias) { bias_ = ploader->pull(getName() + ".bias"); }
  if (ploader->version() == ModelFileVersion::kV1) {
    weight_ = weight_.view({options_.channels, 1, options_.kernel_size});
    if (options_.bias) { bias_ = bias_.view({options_.channels}); }
  }
}

void CausalDepthwiseConv1DOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto* ir_ctx = static_cast<ir::IRContext*>(trace_context);
  if (weight_ && !ir_ctx->lookupSymbolTable(getName() + ".weight")) {
    ir::IRWriterGuard guard(ir_ctx, ir_ctx->lookupSymbolTable("init")->cast_<ir::graph::SubGraphOp>()->getTopRegion());
    ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(weight_));
    if (options_.bias) { ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(bias_)); }
  }
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::CausalDepthwiseConv1DOp>(shared_from_this(), i_irs, o_irs);
}

void CausalDepthwiseConv1DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("CausalDepthwiseConv1DOp::forward not implemented in aops base.");
}

void CausalDepthwiseConv1DOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (inputs.size() != 2) { throw std::invalid_argument("CausalDepthwiseConv1D expects input and state"); }
  const auto& input = inputs[0];
  const auto& state = inputs[1];
  if (options_.channels <= 0 || options_.kernel_size <= 1) {
    throw std::invalid_argument("CausalDepthwiseConv1D options require positive channels and kernel_size > 1");
  }
  if (input.rank() != 3 || input.shape()[2] != options_.channels) {
    throw std::invalid_argument("CausalDepthwiseConv1D input must have [B, S, C] shape");
  }
  const Tensor::shape_t expected_state = {input.shape()[0], options_.channels, options_.kernel_size - 1};
  if (state.shape() != expected_state) {
    throw std::invalid_argument("CausalDepthwiseConv1D state must have [B, C, K - 1] shape");
  }
  for (const auto& tensor : inputs) {
    if (tensor.dtype() != kFloat32 || tensor.device() != input.device()) {
      throw std::invalid_argument("CausalDepthwiseConv1D requires float32 inputs on one device");
    }
  }
  outputs.emplace_back(Tensor::empty(input.shape(), input.dtype(), input.device()));
  outputs.emplace_back(options_.state_inplace ? state : Tensor::empty(state.shape(), state.dtype(), state.device()));
}

void CausalDepthwiseConv1DOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.state_inplace) {
    outputs[0].alloc();
  } else {
    BaseOp::setup(inputs, outputs);
  }
}

ParameterFile::ptr_t CausalDepthwiseConv1DOp::getParams() {
  auto params = ParameterFile::create();
  params->push(getName() + ".weight", weight_);
  if (options_.bias) { params->push(getName() + ".bias", bias_); }
  return params;
}

}  // namespace mllm::aops
