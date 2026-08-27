// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/cpu/kernels/common/linear/kai_w4a32_dispatch.hpp"
#include "mllm/backends/cpu/kernels/common/parallel_linear/shared_input.hpp"

#include "KernelTestHelper.hpp"

class ParallelLinearKernelTest : public KernelTest {
 public:
  ParallelLinearKernelTest() = default;
  ~ParallelLinearKernelTest() override = default;

  bool testKaiW4A32DispatchPolicy() {
    using mllm::cpu::kai_w4a32::shouldUseI8mmPrefill;
    using mllm::cpu::kai_w4a32::threadCount;

    if (shouldUseI8mmPrefill(3, false, true) || shouldUseI8mmPrefill(4, true, true) || shouldUseI8mmPrefill(4, false, false)
        || !shouldUseI8mmPrefill(4, false, true)) {
      return false;
    }
    if (threadCount(1, 8, 4, 6) != 4 || threadCount(28, 8, 4, 6) != 6 || threadCount(1, 2, 4, 6) != 2
        || threadCount(28, 8, 0, 0) != 8) {
      return false;
    }

    const auto decode_plan = mllm::cpu::parallel_linear::planKaiW4A32SharedInput(1, 64, 8, 4, 6);
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
    return decode_plan.supported() && decode_plan.kernel == mllm::cpu::parallel_linear::SharedInputKernel::kKaiDotprod
           && decode_plan.workspace_size > 0 && decode_plan.thread_count == 4;
#else
    return !decode_plan.supported();
#endif
  }
};
