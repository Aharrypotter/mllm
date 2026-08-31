// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cmath>
#include <cstring>
#include <vector>

#include "KernelTestHelper.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q4.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q8.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/vec_dot.hpp"
#include "mllm/backends/cpu/kernels/common/llamafile/llamafile_sgemm.hpp"

class ArmQ4KernelTest : public KernelTest {
 public:
  bool q4ScaleEncodingIsPortableIEEEBinary16() {
    std::vector<float> input(QK4_0, 0.0f);
    input[0] = 4.0f;
    mllm::block_q4_0 block{};
    mllm::cpu::quantize_row_q4_0(input.data(), &block, QK4_0);

    uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, &block.d, sizeof(scale_bits));
    if (scale_bits != 0xb800U) { return false; }  // IEEE binary16(-0.5)

    scale_bits = 0x3800U;  // IEEE binary16(+0.5)
    std::memcpy(&block.d, &scale_bits, sizeof(scale_bits));
    std::fill(std::begin(block.qs), std::end(block.qs), 0x99U);
    std::vector<float> dequantized(QK4_0, 0.0f);
    mllm::cpu::dequantize_row_q4_0(&block, dequantized.data(), QK4_0);
    for (const float value : dequantized) {
      if (value != 0.5f) { return false; }
    }
    return true;
  }

  bool scalarQ4Q8DotDoesNotOverwriteNeighbors() {
    constexpr int kElements = 1536;
    std::vector<float> lhs(kElements);
    std::vector<float> rhs(kElements);
    for (int i = 0; i < kElements; ++i) {
      lhs[i] = std::sin(0.013f * static_cast<float>(i + 3)) * 0.75f;
      rhs[i] = std::cos(0.007f * static_cast<float>(i + 5)) * 0.5f;
    }

    std::vector<mllm::block_q4_0> lhs_q4(kElements / QK4_0);
    std::vector<mllm::block_q8_0> rhs_q8(kElements / QK8_0);
    std::vector<float> lhs_dequant(kElements);
    std::vector<float> rhs_dequant(kElements);
    mllm::cpu::quantize_row_q4_0(lhs.data(), lhs_q4.data(), kElements);
    mllm::cpu::quantize_row_q8_0(rhs.data(), rhs_q8.data(), kElements);
    mllm::cpu::dequantize_row_q4_0(lhs_q4.data(), lhs_dequant.data(), kElements);
    mllm::cpu::dequantize_row_q8_0(rhs_q8.data(), rhs_dequant.data(), kElements);

    float expected = 0.0f;
    for (int i = 0; i < kElements; ++i) { expected += lhs_dequant[i] * rhs_dequant[i]; }

    struct GuardedScalar {
      float left;
      float value;
      float right[4];
    } guarded{12345.25f, NAN, {-54321.5f, 777.25f, -888.5f, 999.75f}};

    mllm::cpu::ggml::vec_dot_q4_0_q8_0(kElements, &guarded.value, lhs_q4.data(), rhs_q8.data());
    const bool guards_ok = guarded.left == 12345.25f && guarded.right[0] == -54321.5f && guarded.right[1] == 777.25f
                           && guarded.right[2] == -888.5f && guarded.right[3] == 999.75f;
    const float tolerance = 2.0e-3f + 2.0e-4f * std::abs(expected);
    return guards_ok && std::isfinite(guarded.value) && std::abs(guarded.value - expected) <= tolerance;
  }

  bool singleRowQ4Q8LlamafileMatchesReference() {
    constexpr int kM = 1;
    constexpr int kN = 257;
    constexpr int kK = 1536;
    constexpr int kThreads = 4;
    std::vector<float> input(kM * kK);
    std::vector<float> weight(kN * kK);
    for (int k = 0; k < kK; ++k) { input[k] = std::sin(0.013f * static_cast<float>(k + 3)) * 0.75f; }
    for (int n = 0; n < kN; ++n) {
      for (int k = 0; k < kK; ++k) {
        weight[n * kK + k] = std::cos(0.007f * static_cast<float>((n + 2) * (k + 1))) * 0.5f
                             + std::sin(0.017f * static_cast<float>(n + k + 7)) * 0.125f;
      }
    }

    std::vector<mllm::block_q8_0> input_q8(kM * kK / QK8_0);
    std::vector<mllm::block_q4_0> weight_q4(kN * kK / QK4_0);
    std::vector<float> input_dequant(kM * kK);
    std::vector<float> weight_dequant(kN * kK);
    mllm::cpu::quantize_row_q8_0(input.data(), input_q8.data(), kK);
    mllm::cpu::dequantize_row_q8_0(input_q8.data(), input_dequant.data(), kK);
    for (int n = 0; n < kN; ++n) {
      mllm::cpu::quantize_row_q4_0(weight.data() + n * kK, weight_q4.data() + n * kK / QK4_0, kK);
      mllm::cpu::dequantize_row_q4_0(weight_q4.data() + n * kK / QK4_0, weight_dequant.data() + n * kK, kK);
    }

    std::vector<float> actual(kN, NAN);
    bool serviced = true;
    for (int thread = 0; thread < kThreads; ++thread) {
      serviced = serviced
                 && mllm::cpu::llamafile_sgemm(kN, kM, kK / QK8_0, weight_q4.data(), kK / QK4_0, input_q8.data(), kK / QK8_0,
                                               actual.data(), kN, thread, kThreads, MLLM_TYPE_Q4_0, MLLM_TYPE_Q8_0,
                                               MLLM_TYPE_F32, nullptr, MLLM_TYPE_F32);
    }
    if (!serviced) { return false; }

    for (int n = 0; n < kN; ++n) {
      float expected = 0.0f;
      for (int k = 0; k < kK; ++k) { expected += input_dequant[k] * weight_dequant[n * kK + k]; }
      const float tolerance = 2.0e-3f + 2.0e-4f * std::abs(expected);
      if (!std::isfinite(actual[n]) || std::abs(actual[n] - expected) > tolerance) { return false; }
    }
    return true;
  }
};
