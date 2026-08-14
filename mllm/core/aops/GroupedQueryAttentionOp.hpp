// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "mllm/core/BaseOp.hpp"

namespace mllm::aops {

enum class GroupedQueryAttentionImplementation : int32_t {
  kDirectStrided = 0,
};

inline const char* groupedQueryAttentionImplementation2Str(GroupedQueryAttentionImplementation implementation) {
  switch (implementation) {
    case GroupedQueryAttentionImplementation::kDirectStrided: return "DirectStrided";
  }
  return "Unknown";
}

inline GroupedQueryAttentionImplementation str2GroupedQueryAttentionImplementation(const std::string& value) {
  if (value == "DirectStrided") return GroupedQueryAttentionImplementation::kDirectStrided;
  throw std::invalid_argument("Unknown GroupedQueryAttention implementation: " + value);
}

struct GroupedQueryAttentionOpOptions : public BaseOpOptions<GroupedQueryAttentionOpOptions> {
  GroupedQueryAttentionImplementation implementation = GroupedQueryAttentionImplementation::kDirectStrided;
};

class GroupedQueryAttentionOp : public BaseOp {
 public:
  explicit GroupedQueryAttentionOp(const GroupedQueryAttentionOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;
  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const GroupedQueryAttentionOpOptions& options() const { return options_; }

 protected:
  GroupedQueryAttentionOpOptions options_;
};

}  // namespace mllm::aops
