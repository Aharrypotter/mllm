// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/GroupedQueryAttentionDecodeOp.hpp"

namespace mllm::cpu {

class CPUGroupedQueryAttentionDecodeOp final : public aops::GroupedQueryAttentionDecodeOp {
 public:
  explicit CPUGroupedQueryAttentionDecodeOp(const aops::GroupedQueryAttentionDecodeOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUGroupedQueryAttentionDecodeOpFactory
    : public TypedOpFactory<OpTypes::kGroupedQueryAttentionDecode, aops::GroupedQueryAttentionDecodeOpOptions> {
 protected:
  std::shared_ptr<BaseOp> createOpImpl(const aops::GroupedQueryAttentionDecodeOpOptions& options) override {
    return std::make_shared<CPUGroupedQueryAttentionDecodeOp>(options);
  }
};

}  // namespace mllm::cpu
