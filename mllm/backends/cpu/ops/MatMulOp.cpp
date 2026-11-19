// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Parallel.hpp"
#include "mllm/core/aops/MatMulOp.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/backends/cpu/ops/MatMulOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/matmul.hpp"
#include "mllm/backends/cpu/kernels/common/blas.hpp"
#include "mllm/backends/cpu/kernels/common/llamafile/llamafile_sgemm.hpp"

namespace mllm::cpu {

CPUMatMulOp::CPUMatMulOp(const aops::MatMulOpOptions& options) : aops::MatMulOp(options) {}

void CPUMatMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& lhs = inputs[0];
  auto& rhs = inputs[1];
  auto& o = outputs[0];

  auto transpose_a = options_.transpose_a;
  auto transpose_b = options_.transpose_b;

  auto lhs_shape = lhs.shape();
  auto rhs_shape = rhs.shape();

  MLLM_RT_ASSERT(lhs_shape.size() >= 2);
  MLLM_RT_ASSERT(rhs_shape.size() >= 2);

  const int lhs_rows = lhs_shape[lhs_shape.size() - 2];
  const int lhs_cols = lhs_shape[lhs_shape.size() - 1];
  const int rhs_rows = rhs_shape[rhs_shape.size() - 2];
  const int rhs_cols = rhs_shape[rhs_shape.size() - 1];
  const int M = transpose_a ? lhs_cols : lhs_rows;
  const int N = transpose_b ? rhs_rows : rhs_cols;
  const int K = transpose_a ? lhs_rows : lhs_cols;
  const int K_from_rhs = transpose_b ? rhs_cols : rhs_rows;
  MLLM_RT_ASSERT_EQ(K, K_from_rhs);

  int batch_count = 1;
  for (size_t i = 0; i < lhs_shape.size() - 2; ++i) { batch_count *= lhs_shape[i]; }
  int rhs_batch_count = 1;
  for (size_t i = 0; i < rhs_shape.size() - 2; ++i) { rhs_batch_count *= rhs_shape[i]; }
  MLLM_RT_ASSERT_EQ(batch_count, rhs_batch_count);

  auto mt = options_.matmul_type;
  if (mt == aops::MatMulOpType::kDefault) {
#if defined(MLLM_USE_BLAS)
    mt = aops::MatMulOpType::kBLAS;
#else
    if (!transpose_a && transpose_b && M >= 4) {
      // TODO: GGUF matmul should be correct when M < 4
      mt = aops::MatMulOpType::kGGUF;
    } else
    // All fallback to mllm blas
    {
      mt = aops::MatMulOpType::kMllmBlas;
    }
#endif
  }

  switch (mt) {
    case aops::MatMulOpType::kDefault: {
      NYI("Auto matmul type inference failed");
      break;
    }
    case aops::MatMulOpType::kGGUF: {
      // llamafile uses column-major order, so we actually perform K^T x Q
      if (lhs.isContiguousN(0)) {
        mllm::cpu::ggml::mat_mul(lhs, rhs, o, false, nullptr, transpose_a, transpose_b, options_.getThreads());
      } else {
        NYI("GGUF implementation for non-contiguous tensors not yet implemented");
      }
      break;
    }
    case aops::MatMulOpType::kBLAS: {
#if defined(MLLM_USE_BLAS)
      MLLM_RT_ASSERT_EQ(lhs.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(rhs.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(o.dtype(), kFloat32);
      if (batch_count == 1) {
        blas::matmul_fp32(lhs.ptr<mllm_fp32_t>(), rhs.ptr<mllm_fp32_t>(), o.ptr<mllm_fp32_t>(), nullptr, M, N, K, transpose_a,
                          transpose_b);
      } else {
        blas::batch_matmul_fp32(lhs.ptr<mllm_fp32_t>(), rhs.ptr<mllm_fp32_t>(), o.ptr<mllm_fp32_t>(), nullptr, batch_count, M,
                                N, K, lhs.stride()[lhs_shape.size() - 3], rhs.stride()[rhs_shape.size() - 3],
                                o.stride()[o.shape().size() - 3], transpose_a, transpose_b);
      }
#else
      NYI("BLAS not supported. Pls set MLLM_USE_BLAS=ON to enable BLAS supports in cmake.");
#endif
      break;
    }
    case aops::MatMulOpType::kCausalScaledQK: {
#if defined(__ARM_NEON)
      MLLM_RT_ASSERT(!transpose_a && transpose_b && M > 1 && N >= M && K > 0 && K % 4 == 0);
      MLLM_RT_ASSERT(lhs_shape.size() == 4 && rhs_shape.size() == 4 && lhs_shape[0] == 1 && rhs_shape[0] == 1);
      MLLM_RT_ASSERT(lhs.dtype() == kFloat32 && rhs.dtype() == kFloat32 && o.dtype() == kFloat32);
      const auto query_strides = lhs.stride();
      const auto key_strides = rhs.stride();
      const auto out_strides = o.stride();
      MLLM_RT_ASSERT(query_strides[3] == 1 && query_strides[2] == K && key_strides[3] == 1 && key_strides[2] == K);
      MLLM_RT_ASSERT(out_strides[3] == 1 && out_strides[2] == N);
      MLLM_RT_ASSERT(query_strides[1] >= static_cast<int64_t>(M) * K && key_strides[1] >= static_cast<int64_t>(N) * K
                        && out_strides[1] >= static_cast<int64_t>(M) * N);
      const auto threads = options_.getThreads();
      MLLM_RT_ASSERT(threads > 0);
      const float scale = 1.f / sqrtf(K);
      for (int head = 0; head < batch_count; ++head) {
        const auto* query = lhs.ptr<float>() + static_cast<int64_t>(head) * query_strides[1];
        const auto* key = rhs.ptr<float>() + static_cast<int64_t>(head) * key_strides[1];
        auto* out = o.ptr<float>() + static_cast<int64_t>(head) * out_strides[1];
        MLLM_CONDITIONAL_PARALLEL_FOR(threads > 1, threads, id, 0, threads, 1, {
          llamafile_causal_scaled_qk_f32(N, M, K, key, query, out, scale, id, threads);
        });
      }
      static const bool diagnostic = [] {
        const char* value = std::getenv("MLLM_QWEN_NPU_CAUSAL_QK_DIAGNOSTIC");
        return value && std::strcmp(value, "1") == 0;
      }();
      if (diagnostic) std::fprintf(stderr, "SHADOWNPU_CAUSAL_QK prefix=%d queries=%d heads=%d\n", N, M, batch_count);
#else
      NYI("CausalScaledQK requires ARM NEON");
#endif
      break;
    }
    case aops::MatMulOpType::kMllmBlas: {
      auto thread_count = options_.getThreads();
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      if (lhs.dtype() == kFloat32 && rhs.dtype() == kFloat32 && o.dtype() == kFloat32) {
        if (batch_count == 1) {
          arm::mllm_blas_matmul_fp32(M, K, N, o.ptr<mllm_fp32_t>(), lhs.ptr<mllm_fp32_t>(), rhs.ptr<mllm_fp32_t>(), nullptr,
                                     transpose_a, transpose_b, thread_count);
        } else {
          arm::mllm_blas_batch_matmul_fp32(batch_count, M, K, N, o.stride()[o.shape().size() - 3],
                                           lhs.stride()[lhs_shape.size() - 3], rhs.stride()[rhs_shape.size() - 3], 0,
                                           o.ptr<mllm_fp32_t>(), lhs.ptr<mllm_fp32_t>(), rhs.ptr<mllm_fp32_t>(), nullptr,
                                           transpose_a, transpose_b, thread_count);
        }
      }
#elif defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      if (lhs.dtype() == kFloat32 && rhs.dtype() == kFloat32 && o.dtype() == kFloat32) {
        if (batch_count == 1) {
          x86::mllm_blas_matmul_fp32(M, K, N, o.ptr<mllm_fp32_t>(), lhs.ptr<mllm_fp32_t>(), rhs.ptr<mllm_fp32_t>(), nullptr,
                                     transpose_a, transpose_b, thread_count);
        } else {
          x86::mllm_blas_batch_matmul_fp32(batch_count, M, K, N, o.stride()[o.shape().size() - 3],
                                           lhs.stride()[lhs_shape.size() - 3], rhs.stride()[rhs_shape.size() - 3], 0,
                                           o.ptr<mllm_fp32_t>(), lhs.ptr<mllm_fp32_t>(), rhs.ptr<mllm_fp32_t>(), nullptr,
                                           transpose_a, transpose_b, thread_count);
        }
      }
#else
      NYI("MllmBlas only support MLLM_HOST_ARCH_ARM64, MLLM_HOST_ARCH_ARM, MLLM_HOST_ARCH_X86_64 or MLLM_HOST_ARCH_X86 right "
          "now.")
#endif
      break;
    }
    default: {
      break;
    }
  }
}

}  // namespace mllm::cpu
