// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstddef>
#include <cstdint>
#include <new>

#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/simple_reg.h"
#include "ShadowCorrectionPacketKernel.hpp"

BEGIN_PKG_OP_DEFINITION(PKG_ShadowCorrectionPacket);

namespace {

using namespace shadow_correction_packet;

bool isFlat(const Tensor& value) {
  constexpr auto incompatible = Tensor::tformat_is_not_flat | Tensor::tformat_is_chunked | Tensor::tformat_is_indirect;
  return (value.get_tensor_format_code() & incompatible) == 0;
}

GraphStatus packCorrection(Tensor& output, const Tensor& activation, const Tensor& codes, const Tensor& controls) {
  const auto [batch, height, width, depth] = activation.dims();
  const auto [code_batch, code_height, code_width, code_depth] = codes.dims();
  const auto [control_batch, control_height, control_width, control_depth] = controls.dims();
  if (batch != 1 || height * width != tile_rows || depth != in_width
      || code_batch != batch || code_height != height || code_width != width || code_depth != out_width
      || control_batch * control_height * control_width * control_depth != 3) {
    return GraphStatus::ErrorDimensions;
  }
  if (activation.get_dtype() != DType::Float32 || codes.get_dtype() != DType::QInt16 || codes.interface_offset() != 0
      || controls.get_dtype() != DType::Int32 || output.get_dtype() != DType::Int32
      || !isFlat(output) || !isFlat(activation) || !isFlat(codes) || !isFlat(controls)) {
    return GraphStatus::ErrorFatal;
  }

  constexpr std::size_t output_words_per_row = sizeof(correction_packet) / (tile_rows * sizeof(std::int32_t));
  constexpr std::size_t output_shape[] = {1, tile_rows, 1, output_words_per_row};
  if (!output.set_dims(output_shape) || output.total_storage_elements() < tile_rows * output_words_per_row
      || activation.total_storage_elements() < tile_rows * in_width
      || codes.total_storage_elements() < tile_rows * out_width || controls.total_storage_elements() < 3) {
    return GraphStatus::ErrorDimensions;
  }
  auto* destination = output.raw_data();
  const auto* rows = static_cast<const float*>(activation.raw_data_const());
  const auto* projected = static_cast<const std::int16_t*>(codes.raw_data_const());
  const auto* limits = static_cast<const std::uint32_t*>(controls.raw_data_const());
  if (!destination || !rows || !projected || !limits) { return GraphStatus::ErrorFatal; }

  auto* packet = new (destination) correction_packet;
  pack_shadow_correction_tile(rows, projected, limits[0], limits[1], limits[2], packet);
  return GraphStatus::Success;
}

}  // namespace

DEF_PACKAGE_OP(packCorrection, "ShadowCorrectionPacket")
DEF_PACKAGE_PARAM_ORDER("ShadowCorrectionPacket", "controls", true, nullptr)

END_PKG_OP_DEFINITION(PKG_ShadowCorrectionPacket);
