// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/common/parallel_linear/shared_input.hpp"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>

#include "mllm/backends/cpu/kernels/arm/linear/kai.hpp"
#include "mllm/backends/cpu/kernels/common/linear/kai_w4a32_dispatch.hpp"

namespace mllm::cpu::parallel_linear {

namespace {

constexpr size_t kMaximumSharedProjections = 3;
using KaiHelper = arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk;
constexpr auto kDecodeTile = KaiHelper::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32;
constexpr auto kPrefillTile = KaiHelper::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32;

bool traceActivationEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("MLLM_KAI_SHARED_INPUT_TRACE");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  return enabled;
}

void traceActivation(const SharedInputPlan& plan, size_t projection_count, int32_t m, int32_t k) {
  if (!traceActivationEnabled()) { return; }

  const uint32_t projection_group = static_cast<uint32_t>(projection_count - 2);
  const uint32_t activation_bit = 1U << (2U * projection_group + static_cast<uint32_t>(m > 1));
  static std::atomic<uint32_t> activated_groups{0};
  const uint32_t previous = activated_groups.fetch_or(activation_bit, std::memory_order_relaxed);
  if ((previous & activation_bit) == 0) {
    std::fprintf(stderr, "MLLM_KAI_SHARED_INPUT_ACTIVATED rhs=%zu m=%d k=%d threads=%d tile=%s\n", projection_count, m, k,
                 plan.thread_count, plan.kernel == SharedInputKernel::kKaiDotprod ? "dotprod_1x8" : "i8mm_4x8");
  }
}

}  // namespace

SharedInputPlan planKaiW4A32SharedInput(int32_t m, int32_t k, int32_t requested_threads, int32_t decode_thread_cap,
                                        int32_t prefill_thread_cap) {
  if (m <= 0 || k <= 0 || requested_threads <= 0) { return {}; }
  if (m > 1 && !kai_w4a32::shouldUseI8mmPrefill(m)) { return {}; }

  const auto tile = m == 1 ? kDecodeTile : kPrefillTile;
  KaiHelper helper;
  const size_t workspace_size = helper.workspace_size(m, k, tile);
  const int32_t thread_count = kai_w4a32::threadCount(m, requested_threads, decode_thread_cap, prefill_thread_cap);
  if (workspace_size == 0 || thread_count <= 0) { return {}; }

  return {.kernel = m == 1 ? SharedInputKernel::kKaiDotprod : SharedInputKernel::kKaiI8mm,
          .workspace_size = workspace_size,
          .thread_count = thread_count};
}

bool runKaiW4A32SharedInput(const SharedInputPlan& plan, const float* input, const SharedInputProjection* projections,
                            size_t projection_count, void* workspace, int32_t m, int32_t k) {
  if (!plan.supported() || input == nullptr || projections == nullptr || projection_count < 2
      || projection_count > kMaximumSharedProjections || workspace == nullptr || m <= 0 || k <= 0 || plan.thread_count <= 0
      || (m == 1 && plan.kernel != SharedInputKernel::kKaiDotprod) || (m > 1 && plan.kernel != SharedInputKernel::kKaiI8mm)) {
    return false;
  }

  std::array<KaiHelper::SharedInputProjection, kMaximumSharedProjections> kai_projections{};
  for (size_t index = 0; index < projection_count; ++index) {
    if (projections[index].dst == nullptr || projections[index].packed_weight_bias == nullptr || projections[index].n <= 0) {
      return false;
    }
    kai_projections[index] = {
        .dst = projections[index].dst, .packed_weight_bias = projections[index].packed_weight_bias, .n = projections[index].n};
  }

  const auto tile = plan.kernel == SharedInputKernel::kKaiDotprod ? kDecodeTile : kPrefillTile;
  KaiHelper helper;
  if (!helper.matmul_shared_input(input, kai_projections.data(), projection_count, workspace, m, k, tile, plan.thread_count)) {
    return false;
  }
  traceActivation(plan, projection_count, m, k);
  return true;
}

}  // namespace mllm::cpu::parallel_linear
