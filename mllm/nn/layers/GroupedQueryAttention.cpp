// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/GroupedQueryAttention.hpp"

namespace mllm::nn {

GroupedQueryAttention::GroupedQueryAttention()
    : Layer(OpTypes::kGroupedQueryAttention, aops::GroupedQueryAttentionOpOptions{}) {}

GroupedQueryAttention::GroupedQueryAttention(aops::GroupedQueryAttentionImplementation implementation)
    : Layer(OpTypes::kGroupedQueryAttention, aops::GroupedQueryAttentionOpOptions{.implementation = implementation}) {}

}  // namespace mllm::nn
