// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/GroupedQueryAttentionOp.hpp"

namespace mllm::cpu {

class CPUGroupedQueryAttentionOp final : public aops::GroupedQueryAttentionOp {
 public:
  explicit CPUGroupedQueryAttentionOp(const aops::GroupedQueryAttentionOpOptions& options);
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUGroupedQueryAttentionOpFactory
    : public TypedOpFactory<OpTypes::kGroupedQueryAttention, aops::GroupedQueryAttentionOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::GroupedQueryAttentionOpOptions& options) override {
    return std::make_shared<CPUGroupedQueryAttentionOp>(options);
  }
};

}  // namespace mllm::cpu
