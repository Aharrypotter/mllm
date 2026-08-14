// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/ParallelLinearOp.hpp"

#include <stdexcept>

#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

ParallelLinearOp::ParallelLinearOp(const ParallelLinearOpOptions& options)
    : BaseOp(OpTypes::kParallelLinear), options_(options) {}

std::string ParallelLinearOp::projectionParameterName(size_t index, const char* suffix) const {
  const auto separator = getName().rfind('.');
  const std::string parent = separator == std::string::npos ? std::string{} : getName().substr(0, separator + 1);
  return parent + options_.projection_names.at(index) + suffix;
}

void ParallelLinearOp::load(const ParameterFile::ptr_t& ploader) {
  if (options_.projection_names.size() != options_.out_channels.size() || options_.projection_names.size() < 2) {
    throw std::invalid_argument("ParallelLinear requires matching projection names and at least two outputs");
  }
  weights_.clear();
  biases_.clear();
  weights_.reserve(options_.projection_names.size());
  if (options_.bias) { biases_.reserve(options_.projection_names.size()); }
  for (size_t index = 0; index < options_.projection_names.size(); ++index) {
    auto weight = ploader->pull(projectionParameterName(index, ".weight"));
    if (ploader->version() == ModelFileVersion::kV1
        && (options_.impl_type == LinearImplTypes::kDefault || options_.impl_type == LinearImplTypes::kBLAS
            || options_.impl_type == LinearImplTypes::kGGUF || options_.impl_type == LinearImplTypes::kMllmBlas)) {
      weight = weight.view({options_.out_channels[index], options_.in_channels});
    }
    weights_.push_back(std::move(weight));
    if (options_.bias) {
      auto bias = ploader->pull(projectionParameterName(index, ".bias"));
      if (ploader->version() == ModelFileVersion::kV1) { bias = bias.view({options_.out_channels[index]}); }
      biases_.push_back(std::move(bias));
    }
  }
}

void ParallelLinearOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto* ir_ctx = static_cast<ir::IRContext*>(trace_context);
  if (!weights_.empty()) {
    ir::IRWriterGuard guard(ir_ctx, ir_ctx->lookupSymbolTable("init")->cast_<ir::graph::SubGraphOp>()->getTopRegion());
    for (size_t index = 0; index < weights_.size(); ++index) {
      const auto weight_name = projectionParameterName(index, ".weight");
      if (!ir_ctx->lookupSymbolTable(weight_name)) {
        ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(weights_[index]));
      }
      if (options_.bias) {
        const auto bias_name = projectionParameterName(index, ".bias");
        if (!ir_ctx->lookupSymbolTable(bias_name)) {
          ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(biases_[index]));
        }
      }
    }
  }
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::ParallelLinearOp>(shared_from_this(), i_irs, o_irs);
}

void ParallelLinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("ParallelLinearOp::forward not implemented in aops base.");
}

void ParallelLinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (inputs.size() != 1 || inputs[0].rank() < 2 || inputs[0].size(-1) != options_.in_channels) {
    throw std::invalid_argument("ParallelLinear expects one [..., M, K] input with the configured K");
  }
  if (options_.in_channels <= 0 || options_.out_channels.size() < 2
      || options_.projection_names.size() != options_.out_channels.size()) {
    throw std::invalid_argument("ParallelLinear options are incomplete");
  }
  for (const int32_t channels : options_.out_channels) {
    if (channels <= 0) { throw std::invalid_argument("ParallelLinear output channels must be positive"); }
    auto shape = inputs[0].shape();
    shape.back() = channels;
    outputs.emplace_back(Tensor::empty(shape, inputs[0].dtype(), inputs[0].device()));
  }
}

void ParallelLinearOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

ParameterFile::ptr_t ParallelLinearOp::getParams() {
  auto params = ParameterFile::create();
  for (size_t index = 0; index < weights_.size(); ++index) {
    params->push(projectionParameterName(index, ".weight"), weights_[index]);
    if (options_.bias) { params->push(projectionParameterName(index, ".bias"), biases_[index]); }
  }
  return params;
}

}  // namespace mllm::aops
