// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/LinearShadowCorrectionOp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::cpu {

namespace {

bool scalarFlag(const Tensor& tensor) {
  switch (tensor.dtype()) {
    case kBool:
    case kUInt8: return tensor.ptr<uint8_t>()[0] != 0;
    case kInt8: return tensor.ptr<int8_t>()[0] != 0;
    case kInt16: return tensor.ptr<int16_t>()[0] != 0;
    case kInt32: return tensor.ptr<int32_t>()[0] != 0;
    default: NYI("Unsupported clip flag dtype");
  }
  return false;
}

}  // namespace

CPULinearShadowCorrectionOp::CPULinearShadowCorrectionOp(const aops::LinearShadowCorrectionOpOptions& options)
    : LinearShadowCorrectionOp(options) {}

void CPULinearShadowCorrectionOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& original_input = inputs[0];
  const auto& quantized_output = inputs[1];
  const auto& baseline = inputs[2];
  auto& corrected = outputs[0];

  std::memcpy(corrected.ptr<void>(), baseline.ptr<void>(), baseline.bytes());

  const bool clip_input = scalarFlag(clip_input_);
  const bool clip_output = scalarFlag(clip_output_);
  if (clip_input && clip_output) { return; }

  constexpr int32_t kQuantMin = -32768;
  constexpr int32_t kQuantMax = 32767;
  const float weight_step = weight_scale_.ptr<float>()[0];
  const float input_step = input_scale_.ptr<float>()[0] / static_cast<float>(kQuantMax);
  const float output_step = output_scale_.ptr<float>()[0] / static_cast<float>(kQuantMax);
  MLLM_RT_ASSERT(std::isfinite(weight_step) && weight_step > 0.0F);
  MLLM_RT_ASSERT(std::isfinite(input_step) && input_step > 0.0F);
  MLLM_RT_ASSERT(std::isfinite(output_step) && output_step > 0.0F);

  const auto* x = original_input.ptr<float>();
  const auto* q_output = quantized_output.ptr<int16_t>();
  const auto* base = baseline.ptr<float>();
  auto* output = corrected.ptr<float>();
  const auto* weight = weight_.ptr<int8_t>();
  const int32_t in_channels = options_.in_channels;
  const int32_t out_channels = options_.out_channels;
  const int64_t tokens = baseline.numel() / out_channels;

  for (int64_t token = 0; token < tokens; ++token) {
    const auto* token_input = x + token * in_channels;
    auto* token_output = output + token * out_channels;

    if (!clip_input) {
      for (int32_t input_channel = 0; input_channel < in_channels; ++input_channel) {
        const float unbounded_q = std::round(token_input[input_channel] / input_step);
        if (unbounded_q >= static_cast<float>(kQuantMin) && unbounded_q <= static_cast<float>(kQuantMax)) { continue; }
        const float clipped_q = std::clamp(unbounded_q, static_cast<float>(kQuantMin), static_cast<float>(kQuantMax));
        const float correction_scale = (unbounded_q - clipped_q) * input_step * weight_step;
        const auto* weight_row = weight + static_cast<int64_t>(input_channel) * out_channels;
        for (int32_t output_channel = 0; output_channel < out_channels; ++output_channel) {
          token_output[output_channel] += correction_scale * static_cast<float>(weight_row[output_channel]);
        }
      }
    }

    if (!clip_output) {
      const auto* token_q_output = q_output + token * out_channels;
      const auto* token_baseline = base + token * out_channels;
      for (int32_t output_channel = 0; output_channel < out_channels; ++output_channel) {
        const int32_t q_value = token_q_output[output_channel];
        if (q_value > kQuantMin && q_value < kQuantMax) { continue; }

        float full_sum = 0.0F;
        for (int32_t input_channel = 0; input_channel < in_channels; ++input_channel) {
          const float unbounded_q = std::round(token_input[input_channel] / input_step);
          const int8_t q_weight = weight[static_cast<int64_t>(input_channel) * out_channels + output_channel];
          full_sum += unbounded_q * input_step * static_cast<float>(q_weight) * weight_step;
        }
        const float restored_q = std::round(full_sum / output_step);
        token_output[output_channel] =
            token_baseline[output_channel] - static_cast<float>(q_value) * output_step + restored_q * output_step;
      }
    }
  }
}

}  // namespace mllm::cpu
