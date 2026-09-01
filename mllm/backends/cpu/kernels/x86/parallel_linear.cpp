// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/common/parallel_linear/shared_input.hpp"

namespace mllm::cpu::parallel_linear {

SharedInputPlan planKaiW4A32SharedInput(int32_t m, int32_t k, int32_t requested_threads, int32_t decode_thread_cap,
                                        int32_t prefill_thread_cap) {
  (void)m;
  (void)k;
  (void)requested_threads;
  (void)decode_thread_cap;
  (void)prefill_thread_cap;
  return {};
}

bool runKaiW4A32SharedInput(const SharedInputPlan& plan, const float* input, const SharedInputProjection* projections,
                            size_t projection_count, void* workspace, int32_t m, int32_t k) {
  (void)plan;
  (void)input;
  (void)projections;
  (void)projection_count;
  (void)workspace;
  (void)m;
  (void)k;
  return false;
}

}  // namespace mllm::cpu::parallel_linear
