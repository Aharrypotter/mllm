// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct GroupedQueryAttentionDecodeOpOptions : public BaseOpOptions<GroupedQueryAttentionDecodeOpOptions> {};

// Single-token grouped-query attention over native KV-head BHSD cache views.
// Inputs: query [B, Hq, 1, Dqk], key [B, Hkv, S, Dqk],
// value [B, Hkv, S, Dv]. Output: [B, Hq, 1, Dv].
class GroupedQueryAttentionDecodeOp : public BaseOp {
 public:
  explicit GroupedQueryAttentionDecodeOp(const GroupedQueryAttentionDecodeOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const GroupedQueryAttentionDecodeOpOptions& options() const { return options_; }

 protected:
  GroupedQueryAttentionDecodeOpOptions options_;
};

}  // namespace mllm::aops
