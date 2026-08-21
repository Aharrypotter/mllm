// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/aops/LinearOp.hpp"

namespace mllm::aops {

struct ParallelLinearOpOptions : public BaseOpOptions<ParallelLinearOpOptions> {
  int32_t in_channels = 0;
  std::vector<int32_t> out_channels;
  std::vector<std::string> projection_names;
  bool bias = false;
  LinearImplTypes impl_type = LinearImplTypes::kDefault;
  int32_t kai_w4a32_decode_thread_cap = 0;
  int32_t kai_w4a32_prefill_thread_cap = 0;
};

class ParallelLinearOp : public BaseOp {
 public:
  explicit ParallelLinearOp(const ParallelLinearOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;
  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  ParameterFile::ptr_t getParams() override;

  inline const ParallelLinearOpOptions& options() const { return options_; }

 protected:
  [[nodiscard]] std::string projectionParameterName(size_t index, const char* suffix) const;

  void validateProjectionNames() const;

  std::vector<Tensor> weights_;
  std::vector<Tensor> biases_;
  ParallelLinearOpOptions options_;
};

}  // namespace mllm::aops
