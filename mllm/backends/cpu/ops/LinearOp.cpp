// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#if defined(__linux__)
#include <sys/auxv.h>
#endif

#include "mllm/backends/cpu/ops/LinearOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/Context.hpp"

namespace mllm::cpu {

namespace {

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

using KaiW4A32Helper = ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk;
using KaiW4A32Tile = KaiW4A32Helper::Tiles;

constexpr KaiW4A32Tile kKaiW4A32DotProdTile = KaiW4A32Tile::qai8dxp1x8_qsi4c32p8x8_1x8x32;
constexpr KaiW4A32Tile kKaiW4A32I8mmTile = KaiW4A32Tile::qai8dxp4x8_qsi4c32p8x8_4x8x32;

bool environmentFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

bool cpuSupportsI8mm() {
#if defined(__linux__) && defined(__aarch64__)
  constexpr unsigned long kHwcap2I8mm = 1UL << 13;
  static const bool supported = (getauxval(AT_HWCAP2) & kHwcap2I8mm) != 0;
  return supported;
#else
  return false;
#endif
}

KaiW4A32Tile selectKaiW4A32PrefillTile(int m) {
  static const bool disabled = environmentFlagEnabled("MLLM_KAI_PREFILL_I8MM_DISABLE");
  if (detail::shouldUseKaiW4A32I8mmPrefill(m, disabled, cpuSupportsI8mm())) { return kKaiW4A32I8mmTile; }
  return kKaiW4A32DotProdTile;
}

void traceKaiW4A32PrefillTile(KaiW4A32Tile tile, int m, int k, int n, int threads) {
  if (m < 4) { return; }
  static const bool trace = environmentFlagEnabled("MLLM_KAI_PREFILL_I8MM_TRACE");
  if (!trace) { return; }

  const uint32_t activation_bit = tile == kKaiW4A32I8mmTile ? 1U : 2U;
  static std::atomic<uint32_t> activated_tiles{0};
  const uint32_t previous = activated_tiles.fetch_or(activation_bit, std::memory_order_relaxed);
  if ((previous & activation_bit) != 0) { return; }

  if (tile == kKaiW4A32I8mmTile) {
    std::fprintf(stderr, "MLLM_KAI_PREFILL_I8MM_ACTIVATED m=%d k=%d n=%d threads=%d\n", m, k, n, threads);
  } else {
    const char* reason = environmentFlagEnabled("MLLM_KAI_PREFILL_I8MM_DISABLE") ? "disabled" : "unsupported";
    std::fprintf(stderr, "MLLM_KAI_PREFILL_DOTPROD_FALLBACK reason=%s m=%d k=%d n=%d threads=%d\n", reason, m, k, n, threads);
  }
}

#endif

}  // namespace

CPULinearOp::CPULinearOp(const aops::LinearOpOptions& options) : LinearOp(options) {}

void CPULinearOp::setKaiW4A32ThreadCaps(int decode_thread_cap, int prefill_thread_cap) {
  if (decode_thread_cap <= 0 || prefill_thread_cap <= 0) {
    throw std::invalid_argument("KAI W4A32 thread caps must be positive");
  }
  kai_w4a32_decode_thread_cap_ = decode_thread_cap;
  kai_w4a32_prefill_thread_cap_ = prefill_thread_cap;
}

int CPULinearOp::kaiW4A32ThreadCount(int m) const {
  return detail::kaiW4A32ThreadCount(
      m, options_.getThreads(), kai_w4a32_decode_thread_cap_, kai_w4a32_prefill_thread_cap_);
}

Tensor CPULinearOp::acquireKaiWorkspace(int32_t workspace_size, int m) {
  if (m != 1) { return Tensor::empty({workspace_size}, kInt8, kCPU).alloc(); }

  if (kai_decode_workspace_.isNil() || kai_decode_workspace_.numel() < static_cast<size_t>(workspace_size)) {
    kai_decode_workspace_ = Tensor::empty({workspace_size}, kInt8, kCPU).alloc();
  }
  return kai_decode_workspace_;
}

bool CPULinearOp::tryForwardSharedInputKaiM1(const Tensor& input, const BaseOp::ptr_t* linear_ops, size_t linear_op_count,
                                             std::vector<Tensor>& outputs) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
  constexpr size_t kMaximumSharedProjections = 3;
  constexpr auto kRequiredImpl = aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32;
  using KaiHelper = ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk;
  constexpr auto kTile = KaiHelper::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32;

  if (Context::instance().thisThread()->trace_mode || input.isNil() || input.device() != kCPU || input.dtype() != kFloat32
      || !input.isContiguous() || input.rank() < 2 || input.size(-2) != 1 || input.size(-1) <= 0 || !outputs.empty()
      || linear_ops == nullptr || linear_op_count < 2 || linear_op_count > kMaximumSharedProjections) {
    return false;
  }

  const auto input_shape = input.shape();
  for (size_t index = 0; index + 2 < input_shape.size(); ++index) {
    if (input_shape[index] != 1) { return false; }
  }

  const int32_t K = input.size(-1);
  int32_t thread_count = 0;
  std::array<CPULinearOp*, kMaximumSharedProjections> ops{};
  for (size_t index = 0; index < linear_op_count; ++index) {
    auto* op = dynamic_cast<CPULinearOp*>(linear_ops[index].get());
    if (op == nullptr || op->getDevice() != kCPU || op->options_.impl_type != kRequiredImpl || op->options_.bias
        || op->options_.in_channels != K || op->options_.out_channels <= 0 || op->weight_.isNil()
        || op->weight_.device() != kCPU || op->options_.getThreads() <= 0) {
      return false;
    }
    if (index == 0) {
      thread_count = op->kaiW4A32ThreadCount(1);
    } else if (op->kaiW4A32ThreadCount(1) != thread_count) {
      return false;
    }
    ops[index] = op;
  }

  std::vector<Tensor> prepared_outputs;
  prepared_outputs.reserve(linear_op_count);
  for (size_t index = 0; index < linear_op_count; ++index) {
    auto output_shape = input_shape;
    output_shape.back() = ops[index]->options_.out_channels;
    prepared_outputs.emplace_back(Tensor::empty(output_shape, kFloat32, kCPU).alloc());
  }

  std::array<KaiHelper::SharedInputProjection, kMaximumSharedProjections> projections{};
  for (size_t index = 0; index < linear_op_count; ++index) {
    projections[index] = {
        .dst = prepared_outputs[index].ptr<mllm_fp32_t>(),
        .packed_weight_bias = reinterpret_cast<const uint8_t*>(ops[index]->weight_.ptr<mllm_byte_t>()),
        .n = ops[index]->options_.out_channels,
    };
  }

  KaiHelper kai_helper;
  const size_t workspace_size = kai_helper.workspace_size(1, K, kTile);
  if (workspace_size == 0 || workspace_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) { return false; }
  auto workspace = ops[0]->acquireKaiWorkspace(static_cast<int32_t>(workspace_size), 1);
  if (!kai_helper.matmul_shared_input_m1(input.ptr<mllm_fp32_t>(), projections.data(), linear_op_count, workspace.ptr<void>(),
                                         K, kTile, thread_count)) {
    return false;
  }

  static const bool trace_activation = [] {
    const char* value = std::getenv("MLLM_KAI_SHARED_INPUT_TRACE");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  if (trace_activation) {
    const uint32_t activation_bit = 1U << static_cast<uint32_t>(linear_op_count - 2);
    static std::atomic<uint32_t> activated_groups{0};
    const uint32_t previous = activated_groups.fetch_or(activation_bit, std::memory_order_relaxed);
    if ((previous & activation_bit) == 0) {
      std::fprintf(stderr, "MLLM_KAI_SHARED_INPUT_ACTIVATED rhs=%zu k=%d threads=%d\n", linear_op_count, K, thread_count);
    }
  }

  outputs = std::move(prepared_outputs);
  return true;
#else
  (void)input;
  (void)linear_ops;
  (void)linear_op_count;
  (void)outputs;
  return false;
#endif
}

void CPULinearOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      switch (options_.impl_type) {
        case aops::LinearImplTypes::kBLAS:
        case aops::LinearImplTypes::kGGUF:
        case aops::LinearImplTypes::kMllmBlas:
        case aops::LinearImplTypes::kMllmBlas_KAI_SGEMM_NT_NT_NEON:
        case aops::LinearImplTypes::kMllmBlas_KAI_SGEMM_NT_T_SME:
        case aops::LinearImplTypes::kDefault: {
          weight_ = weight_.view({options_.out_channels, options_.in_channels});
          if (options_.bias) {
            bias_ = ploader->pull(getName() + ".bias");
            bias_ = bias_.view({options_.out_channels});
          }
          break;
        }
        default: {
          // No need to view.
          MLLM_EMPTY_SCOPE
          break;
        }
      }
      break;
    }
    case ModelFileVersion::kUserTemporary:
    case ModelFileVersion::kV2: {
      weight_ = ploader->pull(getName() + ".weight");
      switch (options_.impl_type) {
        case aops::LinearImplTypes::kBLAS:
        case aops::LinearImplTypes::kGGUF:
        case aops::LinearImplTypes::kMllmBlas:
        case aops::LinearImplTypes::kMllmBlas_KAI_SGEMM_NT_NT_NEON:
        case aops::LinearImplTypes::kMllmBlas_KAI_SGEMM_NT_T_SME:
        case aops::LinearImplTypes::kDefault: {
          if (options_.bias) {
            bias_ = ploader->pull(getName() + ".bias");
            bias_ = bias_.view({options_.out_channels});
          }
          break;
        }
        default: {
          // No need to view.
          MLLM_EMPTY_SCOPE
          break;
        }
      }
      break;
    }
    default: NYI("Unsupported model file version")
  }

  // Prepare data:
  auto impl_type = options_.impl_type;
  if (impl_type == aops::LinearImplTypes::kDefault) {
#if defined(MLLM_USE_BLAS)
    impl_type = aops::LinearImplTypes::kBLAS;
#else
    // FIXME, When we need kMllmBlas_KAI_SGEMM_NT_NT_NEON. set it.
#endif
  }

  switch (impl_type) {
    case aops::LinearImplTypes::kMllmBlas_KAI_SGEMM_NT_NT_NEON: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      ::mllm::cpu::arm::KaiLinear_fp32_fp32_fp32p_mxk_kxn kai_helper;
      weight_ = weight_.view({options_.out_channels, options_.in_channels});
      auto transposed_weight = weight_.transpose(0, 1);
      int32_t packed_weight_size = kai_helper.quant_pack_rhs_size(transposed_weight.size(0), transposed_weight.size(1));
      auto packed_weight = Tensor::empty({packed_weight_size}, kInt8, kCPU).alloc().setName(weight_.name()).setMemType(kGlobal);
      kai_helper.quant_pack_rhs_offline(packed_weight.ptr<mllm_byte_t>(), transposed_weight.ptr<mllm_fp32_t>(),
                                        bias_ ? bias_.ptr<mllm_fp32_t>() : nullptr, transposed_weight.size(0),
                                        transposed_weight.size(1));
      MLLM_INFO("Packing fp32 weight and bias to kai's fp32 format");
      weight_ = packed_weight;
#endif
      break;
    }
    default: {
      // No need to postprocess.
      MLLM_EMPTY_SCOPE
      break;
    }
  }
}

void CPULinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& o = outputs[0];

  auto input_shape = input.shape();
  MLLM_RT_ASSERT(input_shape.size() >= 2);

  // In Linear
  // inputs is always: [..., S, in_channels]
  // outputs is always: [out_channels, in_channels]
  int M = input_shape[input_shape.size() - 2];
  int K = input_shape[input_shape.size() - 1];
  int N = options_.out_channels;
  MLLM_RT_ASSERT_EQ(K, options_.in_channels);

  auto impl_type = options_.impl_type;
  if (impl_type == aops::LinearImplTypes::kDefault) {
#if defined(MLLM_USE_BLAS)
    impl_type = aops::LinearImplTypes::kBLAS;
#else
    if (K >= 4) {
      impl_type = aops::LinearImplTypes::kGGUF;
    } else
    // All fallback to mllm blas
    {
      impl_type = aops::LinearImplTypes::kMllmBlas;
    }
#endif
  }

  int batch_count = 1;
  for (size_t i = 0; i < input_shape.size() - 2; ++i) { batch_count *= input_shape[i]; }

  switch (impl_type) {
    case aops::LinearImplTypes::kBLAS: {
#if defined(MLLM_USE_BLAS)
      MLLM_RT_ASSERT_EQ(input.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(weight_.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(o.dtype(), kFloat32);
      if (bias_) { MLLM_RT_ASSERT_EQ(bias_.dtype(), kFloat32); }
      if (batch_count == 1) {
        blas::matmul_fp32(input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_fp32_t>(), o.ptr<mllm_fp32_t>(),
                          bias_ ? bias_.ptr<mllm_fp32_t>() : nullptr, M, N, K, false, true);
      } else {
        blas::batch_matmul_fp32(input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_fp32_t>(), o.ptr<mllm_fp32_t>(),
                                bias_ ? bias_.ptr<mllm_fp32_t>() : nullptr, batch_count, M, N, K,
                                input.stride()[input_shape.size() - 3], 0, o.stride()[o.shape().size() - 3], false, true);
      }
#else
      NYI("BLAS not supported. Pls set MLLM_USE_BLAS=ON to enable BLAS supports in cmake.");
#endif
      break;
    }

// The code below is for ARM64/ARM.
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
    case aops::LinearImplTypes::kMllmBlas_KAI_SGEMM_NT_NT_NEON: {
      auto M = input.shape()[input.shape().size() - 2];
      auto K = options_.in_channels;
      auto N = options_.out_channels;

      ::mllm::cpu::arm::KaiLinear_fp32_fp32_fp32p_mxk_kxn kai_helper;
      kai_helper.matmul(o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_byte_t>(), nullptr, M, K, N,
                        options_.getThreads());

      break;
    }
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32: {
    __mllm_label_kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32:
      auto M = input.shape()[input.shape().size() - 2];
      auto K = options_.in_channels;
      auto N = options_.out_channels;

      ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32);
      auto workspace = acquireKaiWorkspace(work_space_size, M);

      kai_helper.matmul(o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_byte_t>(), workspace.ptr<void>(), M, K,
                        N, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32,
                        options_.getThreads());
      return;
    }
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32: {
    __mllm_label_kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32:
      auto M = input.shape()[input.shape().size() - 2];
      auto K = options_.in_channels;
      auto N = options_.out_channels;

      KaiW4A32Helper kai_helper;
      const auto tile = selectKaiW4A32PrefillTile(M);
      const int thread_count = kaiW4A32ThreadCount(M);
      traceKaiW4A32PrefillTile(tile, M, K, N, thread_count);

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(M, K, tile);
      auto workspace = acquireKaiWorkspace(work_space_size, M);

      kai_helper.matmul(o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_byte_t>(), workspace.ptr<void>(), M, K,
                        N, tile, thread_count);
      return;
    }
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_8x4x32: {
      auto M = input.shape()[input.shape().size() - 2];
      auto K = options_.in_channels;
      auto N = options_.out_channels;

      if (M == 1) { goto __mllm_label_kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32; }

      ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32);
      auto workspace = acquireKaiWorkspace(work_space_size, M);

      kai_helper.matmul(o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_byte_t>(), workspace.ptr<void>(), M, K,
                        N, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32,
                        options_.getThreads());
      return;
    }
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_16x4x32: {
      auto M = input.shape()[input.shape().size() - 2];
      auto K = options_.in_channels;
      auto N = options_.out_channels;

      ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32);
      auto workspace = acquireKaiWorkspace(work_space_size, M);

      kai_helper.matmul(o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_byte_t>(), workspace.ptr<void>(), M, K,
                        N, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32,
                        options_.getThreads());
      return;
    }
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32: {
      auto M = input.shape()[input.shape().size() - 2];
      auto K = options_.in_channels;
      auto N = options_.out_channels;

      if (M == 1) { goto __mllm_label_kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32; }

      ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32);
      auto workspace = acquireKaiWorkspace(work_space_size, M);

      kai_helper.matmul(o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_byte_t>(), workspace.ptr<void>(), M, K,
                        N, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32,
                        options_.getThreads());
      return;
    }
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4: {
      auto M = input.shape()[input.shape().size() - 2];
      auto K = options_.in_channels;
      auto N = options_.out_channels;

      ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper;

      // FIXME:
      // Can be optimized for better performance.
      int32_t work_space_size = kai_helper.workspace_size(
          M, K, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4);
      auto workspace = acquireKaiWorkspace(work_space_size, M);

      kai_helper.matmul(o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_byte_t>(), workspace.ptr<void>(), M, K,
                        N, ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4,
                        options_.getThreads());
      return;
    }
#endif
    case aops::LinearImplTypes::kGGUF: {
      // use ggml matmul, which first try llamafile_sgemm, then fallback to ggml matmul
      auto thread_count = options_.getThreads();
      auto* bias_ptr = options_.bias ? &bias_ : nullptr;
      mllm::cpu::ggml::mat_mul(input, weight_, o, options_.bias, bias_ptr, false, true, thread_count);
      break;
    }
    case aops::LinearImplTypes::kMllmBlas: {
      MLLM_RT_ASSERT_EQ(input.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(weight_.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(o.dtype(), kFloat32);
      if (bias_) { MLLM_RT_ASSERT_EQ(bias_.dtype(), kFloat32); }
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      NYI("LinearImplTypes not supported in x86 yet");

#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      if (batch_count == 1) {
        arm::mllm_blas_matmul_fp32(M, K, N, o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_fp32_t>(),
                                   options_.bias ? bias_.ptr<mllm_fp32_t>() : nullptr, false, true, options_.getThreads());
      } else {
        arm::mllm_blas_batch_matmul_fp32(batch_count, M, K, N, o.stride()[o.shape().size() - 3],
                                         input.stride()[input.rank() - 3], 0, 0, o.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(),
                                         weight_.ptr<mllm_fp32_t>(), options_.bias ? bias_.ptr<mllm_fp32_t>() : nullptr, false,
                                         true, options_.getThreads());
      }
#else
// TODO Other arch
#endif
      break;
    }
    default: {
      NYI("LinearImplTypes not supported");
      break;
    }
  }
}

void CPULinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.isRedirect()) {
    outputs.emplace_back(inputs[1]);
    return;
  }
  const auto& i = inputs[0];
  auto i_shape = i.shape();

  MLLM_RT_ASSERT_EQ(i_shape[i_shape.size() - 1], options_.in_channels);

  auto o_shape = i_shape;
  o_shape[o_shape.size() - 1] = options_.out_channels;

  DataTypes o_dtype = i.dtype();

  switch (options_.impl_type) {
    case aops::LinearImplTypes::kKaiLinear_fp16_fp16_fp16p_mxk_kxn:
    case aops::LinearImplTypes::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x8_qai4c32p4x8_1x4:
    case aops::LinearImplTypes::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x4_qai4c32p4x4_8x4:
    case aops::LinearImplTypes::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x8_qai4c32p4x8_8x4_i8mm: {
      o_dtype = kFloat16;
      break;
    }
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32:
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32:
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_8x4x32:
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_16x4x32:
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32:
    case aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4:
    case aops::LinearImplTypes::kKaiLinear_f32_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa:
    case aops::LinearImplTypes::kKaiLinear_f32_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot:
    case aops::LinearImplTypes::kKaiLinear_f32_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod:
    case aops::LinearImplTypes::kKaiLinear_f32_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod:
    case aops::LinearImplTypes::kKaiLinear_f32_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod:
    case aops::LinearImplTypes::kKaiLinear_f32_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm: {
      o_dtype = kFloat32;
      break;
    }
    case aops::LinearImplTypes::kGGUF: {
      o_dtype = kFloat32;
      break;
    }
    case aops::LinearImplTypes::kQNN_LPBQ_w4a16o16_G32:
    case aops::LinearImplTypes::kQNN_LPBQ_w4a16o16_G64: {
      if (o_shape[0] == 1) { o_shape.erase(o_shape.begin()); }
      o_dtype = kUInt16PerTensorAsy;
      break;
    }
    default: o_dtype = i.dtype();
  }

  outputs.emplace_back(Tensor::empty(o_shape, o_dtype, i.device()));
}

}  // namespace mllm::cpu
