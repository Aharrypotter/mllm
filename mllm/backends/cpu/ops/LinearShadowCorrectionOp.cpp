// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/LinearShadowCorrectionOp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

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

  std::vector<int32_t> saturated_channels;
  if (!clip_output) { saturated_channels.reserve(out_channels); }
  std::vector<float> full_sums(clip_output ? 0 : out_channels);

  for (int64_t token = 0; token < tokens; ++token) {
    const auto* token_input = x + token * in_channels;
    const auto* token_q_output = q_output + token * out_channels;
    const auto* token_baseline = base + token * out_channels;
    auto* token_output = output + token * out_channels;

    saturated_channels.clear();
    if (!clip_output) {
      for (int32_t channel = 0; channel < out_channels; ++channel) {
        const int32_t code = token_q_output[channel];
        if (code <= kQuantMin || code >= kQuantMax) { saturated_channels.push_back(channel); }
      }
    }
    if (clip_input && saturated_channels.empty()) { continue; }
    std::fill_n(full_sums.begin(), saturated_channels.size(), 0.0F);

    // Keep each output's accumulation in input-channel order while visiting
    // weights from the same input row together.
    for (int32_t input_channel = 0; input_channel < in_channels; ++input_channel) {
      const float unbounded_q = std::round(token_input[input_channel] / input_step);
      const auto* weight_row = weight + static_cast<int64_t>(input_channel) * out_channels;
      if (!clip_input && !(unbounded_q >= static_cast<float>(kQuantMin) && unbounded_q <= static_cast<float>(kQuantMax))) {
        const float clipped_q = std::clamp(unbounded_q, static_cast<float>(kQuantMin), static_cast<float>(kQuantMax));
        const float correction_scale = (unbounded_q - clipped_q) * input_step * weight_step;
        for (int32_t channel = 0; channel < out_channels; ++channel) {
          token_output[channel] += correction_scale * static_cast<float>(weight_row[channel]);
        }
      }
      const float scaled_input = unbounded_q * input_step;
      for (size_t i = 0; i < saturated_channels.size(); ++i) {
        const float q_weight = weight_row[saturated_channels[i]];
        full_sums[i] += scaled_input * q_weight * weight_step;
      }
    }

    for (size_t i = 0; i < saturated_channels.size(); ++i) {
      const auto channel = saturated_channels[i];
      const float restored_q = std::round(full_sums[i] / output_step);
      token_output[channel] =
          token_baseline[channel] - static_cast<float>(token_q_output[channel]) * output_step + restored_q * output_step;
    }
  }
}

}  // namespace mllm::cpu
