// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstddef>
#include <cstdint>
#include <new>

#include "HTP/core/log.h"
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
  const auto [out_batch, out_height, out_spatial_width, out_depth] = output.dims();
  if (batch != 1 || height * width != tile_rows || depth != in_width
      || code_batch != batch || code_height != height || code_width != width || code_depth != out_width
      || out_batch != 1 || out_height != tile_rows || out_spatial_width != 1 || out_depth != words_per_row
      || control_batch * control_height * control_width * control_depth != 3) {
    errlog("ShadowCorrectionPacket geometry activation=%zu:%zu:%zu:%zu codes=%zu:%zu:%zu:%zu controls=%zu:%zu:%zu:%zu output=%zu:%zu:%zu:%zu",
           batch, height, width, depth, code_batch, code_height, code_width, code_depth,
           control_batch, control_height, control_width, control_depth, out_batch, out_height, out_spatial_width, out_depth);
    return GraphStatus::ErrorDimensions;
  }
  const auto code_type = codes.get_dtype();
  const bool offset_binary = code_type == DType::QUInt16;
  if (activation.get_dtype() != DType::Float32 || (!offset_binary && code_type != DType::QInt16)
      || codes.interface_offset() != (offset_binary ? 32768 : 0)
      || controls.get_dtype() != DType::Int32 || output.get_dtype() != DType::Int32
      || !isFlat(output) || !isFlat(activation) || !isFlat(codes) || !isFlat(controls)) {
    errlog("ShadowCorrectionPacket tensor types=%u:%u:%u:%u formats=%x:%x:%x:%x code_offset=%d",
           static_cast<unsigned>(activation.get_dtype()), static_cast<unsigned>(codes.get_dtype()),
           static_cast<unsigned>(controls.get_dtype()), static_cast<unsigned>(output.get_dtype()),
           activation.get_tensor_format_code(), codes.get_tensor_format_code(), controls.get_tensor_format_code(),
           output.get_tensor_format_code(), codes.interface_offset());
    return GraphStatus::ErrorFatal;
  }

  if (output.total_storage_elements() < packet_words || activation.total_storage_elements() < tile_rows * in_width
      || codes.total_storage_elements() < tile_rows * out_width || controls.total_storage_elements() < 3) {
    errlog("ShadowCorrectionPacket storage elements activation=%zu codes=%zu controls=%zu output=%zu",
           activation.total_storage_elements(), codes.total_storage_elements(), controls.total_storage_elements(),
           output.total_storage_elements());
    return GraphStatus::ErrorDimensions;
  }
  auto* destination = output.raw_data();
  const auto* rows = static_cast<const float*>(activation.raw_data_const());
  const auto* projected = static_cast<const std::int16_t*>(codes.raw_data_const());
  const auto* limits = static_cast<const std::uint32_t*>(controls.raw_data_const());
  if (!destination || !rows || !projected || !limits) {
    errlog("ShadowCorrectionPacket missing data activation=%d codes=%d controls=%d output=%d",
           rows == nullptr, projected == nullptr, limits == nullptr, destination == nullptr);
    return GraphStatus::ErrorFatal;
  }

  auto* packet = new (destination) correction_packet;
  const auto encoding = offset_binary ? code_encoding::offset_binary : code_encoding::signed_fixed;
  pack_shadow_correction_tile(rows, projected, limits[0], limits[1], limits[2], packet, encoding);
  return GraphStatus::Success;
}

}  // namespace

DEF_PACKAGE_OP_AND_COST_AND_FLAGS(packCorrection, "ShadowCorrectionPacket", SNAIL, Flags::RESOURCE_HVX)
// The full M256 activation exceeds VTCM. The HVX kernel streams contiguous rows from DDR.
DEF_TENSOR_PROPERTIES(Op("ShadowCorrectionPacket", "activation", "codes", "controls"),
                      Flat("*", "activation", "codes", "controls"),
                      MainMemory("*", "activation", "codes", "controls"))
DEF_PACKAGE_PARAM_ORDER("ShadowCorrectionPacket", "controls", true, nullptr)

END_PKG_OP_DEFINITION(PKG_ShadowCorrectionPacket);
