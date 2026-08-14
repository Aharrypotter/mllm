// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class CausalDepthwiseConv1DAccumulationOrder : int32_t {
  kCurrentFirst = 0,
  kHistoryFirst = 1,
};

inline const char* causalDepthwiseConv1DAccumulationOrder2Str(CausalDepthwiseConv1DAccumulationOrder order) {
  switch (order) {
    case CausalDepthwiseConv1DAccumulationOrder::kCurrentFirst: return "CurrentFirst";
    case CausalDepthwiseConv1DAccumulationOrder::kHistoryFirst: return "HistoryFirst";
  }
  return "Unknown";
}

inline CausalDepthwiseConv1DAccumulationOrder str2CausalDepthwiseConv1DAccumulationOrder(const std::string& value) {
  if (value == "CurrentFirst") return CausalDepthwiseConv1DAccumulationOrder::kCurrentFirst;
  if (value == "HistoryFirst") return CausalDepthwiseConv1DAccumulationOrder::kHistoryFirst;
  throw std::invalid_argument("Unknown CausalDepthwiseConv1D accumulation order: " + value);
}

struct CausalDepthwiseConv1DOpOptions : public BaseOpOptions<CausalDepthwiseConv1DOpOptions> {
  int32_t channels = 0;
  int32_t kernel_size = 0;
  bool bias = false;
  bool state_inplace = false;
  CausalDepthwiseConv1DAccumulationOrder accumulation_order = CausalDepthwiseConv1DAccumulationOrder::kCurrentFirst;
};

class CausalDepthwiseConv1DOp : public BaseOp {
 public:
  explicit CausalDepthwiseConv1DOp(const CausalDepthwiseConv1DOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;
  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  ParameterFile::ptr_t getParams() override;

  inline Tensor& weight() { return weight_; }
  inline Tensor& bias() { return bias_; }
  inline const CausalDepthwiseConv1DOpOptions& options() const { return options_; }

 protected:
  Tensor weight_;
  Tensor bias_;
  CausalDepthwiseConv1DOpOptions options_;
};

}  // namespace mllm::aops
