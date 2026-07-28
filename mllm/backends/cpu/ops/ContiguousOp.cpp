// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <cstring>

#include "mllm/core/DataTypes.hpp"
#include "mllm/backends/cpu/ops/ContiguousOp.hpp"

namespace mllm::cpu {

CPUContiguousOp::CPUContiguousOp(const aops::ContiguousOpOptions& options) : aops::ContiguousOp(options) {}

void CPUContiguousOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  auto& o = outputs[0];

  const size_t ele_size = bytesOfType(i.dtype());
  const size_t total_elements = o.numel();
  const size_t total_bytes = total_elements * ele_size;

  if (total_elements == 0) { return; }

  char* dst_ptr = o.ptr<char>();
  const char* src_ptr = i.ptr<char>();

  const int32_t ndim = i.shape().size();

  if (ndim == 0 || i.isContiguous()) {
    std::memcpy(dst_ptr, src_ptr, total_bytes);
    return;
  }

  if (ndim == 1) {
    const int32_t stride_bytes = i.stride()[0] * ele_size;
    const int32_t size = i.shape()[0];

    const int32_t block_size = 8;
    const int32_t remainder = size % block_size;
    const int32_t end = size - remainder;

    for (int32_t j = 0; j < remainder; ++j) {
      std::memcpy(dst_ptr, src_ptr, ele_size);
      dst_ptr += ele_size;
      src_ptr += stride_bytes;
    }

    for (int32_t j = 0; j < end; j += block_size) {
      std::memcpy(dst_ptr, src_ptr, ele_size);
      dst_ptr += ele_size;
      src_ptr += stride_bytes;

      std::memcpy(dst_ptr, src_ptr, ele_size);
      dst_ptr += ele_size;
      src_ptr += stride_bytes;

      std::memcpy(dst_ptr, src_ptr, ele_size);
      dst_ptr += ele_size;
      src_ptr += stride_bytes;

      std::memcpy(dst_ptr, src_ptr, ele_size);
      dst_ptr += ele_size;
      src_ptr += stride_bytes;

      std::memcpy(dst_ptr, src_ptr, ele_size);
      dst_ptr += ele_size;
      src_ptr += stride_bytes;

      std::memcpy(dst_ptr, src_ptr, ele_size);
      dst_ptr += ele_size;
      src_ptr += stride_bytes;

      std::memcpy(dst_ptr, src_ptr, ele_size);
      dst_ptr += ele_size;
      src_ptr += stride_bytes;

      std::memcpy(dst_ptr, src_ptr, ele_size);
      dst_ptr += ele_size;
      src_ptr += stride_bytes;
    }
    return;
  }

  if (i.stride()[ndim - 1] == 1) {
    const size_t inner_size = i.shape()[ndim - 1];
    const size_t inner_bytes = inner_size * ele_size;
    const size_t outer_count = total_elements / inner_size;

    for (size_t idx = 0; idx < outer_count; ++idx) {
      int64_t offset = 0;
      size_t temp = idx;
      for (int d = ndim - 2; d >= 0; --d) {
        const int32_t dim_index = temp % i.shape()[d];
        temp /= i.shape()[d];
        offset += dim_index * i.stride()[d];
      }
      std::memcpy(dst_ptr, src_ptr + offset * ele_size, inner_bytes);
      dst_ptr += inner_bytes;
    }
    return;
  }

  std::vector<int32_t> indices(ndim, 0);
  std::vector<int32_t> strides_bytes(ndim);

  for (int d = 0; d < ndim; ++d) { strides_bytes[d] = i.stride()[d] * ele_size; }

  for (size_t count = 0; count < total_elements; ++count) {
    int64_t src_offset_bytes = 0;
    for (int d = 0; d < ndim; ++d) { src_offset_bytes += indices[d] * strides_bytes[d]; }

    std::memcpy(dst_ptr, src_ptr + src_offset_bytes, ele_size);
    dst_ptr += ele_size;

    for (int d = ndim - 1; d >= 0; --d) {
      if (++indices[d] < i.shape()[d]) { break; }
      indices[d] = 0;
    }
  }
}

}  // namespace mllm::cpu
