// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

namespace mllm::cpu::parallel_linear {

enum class SharedInputKernel : uint8_t {
  kUnsupported,
  kKaiDotprod,
  kKaiI8mm,
};

struct SharedInputProjection {
  float* dst = nullptr;
  const uint8_t* packed_weight_bias = nullptr;
  int32_t n = 0;
};

struct SharedInputPlan {
  SharedInputKernel kernel = SharedInputKernel::kUnsupported;
  size_t workspace_size = 0;
  int32_t thread_count = 0;

  [[nodiscard]] bool supported() const { return kernel != SharedInputKernel::kUnsupported; }
};

[[nodiscard]] SharedInputPlan planKaiW4A32SharedInput(int32_t m, int32_t k, int32_t requested_threads,
                                                      int32_t decode_thread_cap, int32_t prefill_thread_cap);

bool runKaiW4A32SharedInput(const SharedInputPlan& plan, const float* input, const SharedInputProjection* projections,
                            size_t projection_count, void* workspace, int32_t m, int32_t k);

}  // namespace mllm::cpu::parallel_linear
