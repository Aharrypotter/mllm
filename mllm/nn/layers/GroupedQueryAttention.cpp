// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/GroupedQueryAttention.hpp"

namespace mllm::nn {

GroupedQueryAttention::GroupedQueryAttention()
    : Layer(OpTypes::kGroupedQueryAttention, aops::GroupedQueryAttentionOpOptions{}) {}

GroupedQueryAttention::GroupedQueryAttention(const aops::GroupedQueryAttentionOpOptions& options)
    : Layer(OpTypes::kGroupedQueryAttention, options) {}

}  // namespace mllm::nn
