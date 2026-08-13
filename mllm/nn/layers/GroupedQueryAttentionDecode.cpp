// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/GroupedQueryAttentionDecode.hpp"

namespace mllm::nn {

GroupedQueryAttentionDecode::GroupedQueryAttentionDecode()
    : GroupedQueryAttentionDecode(aops::GroupedQueryAttentionDecodeOpOptions{}) {}

GroupedQueryAttentionDecode::GroupedQueryAttentionDecode(const aops::GroupedQueryAttentionDecodeOpOptions& options)
    : Layer(OpTypes::kGroupedQueryAttentionDecode, options) {}

}  // namespace mllm::nn
