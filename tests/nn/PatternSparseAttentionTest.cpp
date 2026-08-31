// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "mllm/compile/ir/Trace.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/jit/binary/LinalgIRSerialization.hpp"
#include "mllm/compile/jit/interpreter/AopsFromJson.hpp"
#include "mllm/backends/qnn/QNNSequenceTiling.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/qwen_npu/ShadowNPUHeadSchedule.hpp"
#include "mllm/models/qwen_npu/modeling_qwen_npu.hpp"
#include "mllm/models/qwen_npu/modeling_qwen_npu_cpu.hpp"
#include "mllm/nn/Nn.hpp"

namespace {

using mllm::Tensor;

class PatternSparseAttentionTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

Tensor tensorFromValues(const Tensor::shape_t& shape, const std::vector<float>& values) {
  auto tensor = Tensor::empty(shape, mllm::kFloat32, mllm::kCPU).alloc();
  EXPECT_EQ(values.size(), static_cast<size_t>(tensor.numel()));
  std::copy(values.begin(), values.end(), tensor.ptr<float>());
  return tensor;
}

Tensor fp16TensorFromValues(const Tensor::shape_t& shape, const std::vector<float>& values) {
  auto tensor = Tensor::empty(shape, mllm::kFloat16, mllm::kCPU).alloc();
  EXPECT_EQ(values.size(), static_cast<size_t>(tensor.numel()));
  std::transform(values.begin(), values.end(), tensor.ptr<mllm::mllm_fp16_t>(),
                 [](float value) { return static_cast<mllm::mllm_fp16_t>(value); });
  return tensor;
}

float selectedSoftmaxValue(const std::vector<float>& scores, const std::vector<float>& values,
                           std::initializer_list<int32_t> selected) {
  // Android Release tests use -ffast-math, so keep this reference valid under
  // finite-math assumptions just like the production attention kernels.
  float maximum = std::numeric_limits<float>::lowest();
  for (const int32_t index : selected) { maximum = std::max(maximum, scores[index]); }
  float numerator = 0.0F;
  float denominator = 0.0F;
  for (const int32_t index : selected) {
    const float weight = std::exp(scores[index] - maximum);
    numerator += weight * values[index];
    denominator += weight;
  }
  return numerator / denominator;
}

mllm::aops::PatternSparseAttentionOpOptions optionsWith(float sparsity) {
  mllm::aops::PatternSparseAttentionOpOptions options;
  options.sparsity = sparsity;
  return options;
}

class PatternSparseAttentionTraceModule final : public mllm::nn::Module {
 public:
  explicit PatternSparseAttentionTraceModule(const mllm::aops::PatternSparseAttentionOpOptions& options)
      : Module("pattern_sparse_attention_trace") {
    attention_ = reg<mllm::nn::PatternSparseAttention>("attention", options);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<mllm::AnyValue>& args) override {
    if (inputs.size() == 4) { return {attention_(inputs[0], inputs[1], inputs[2], inputs[3])}; }
    return {attention_(inputs[0], inputs[1], inputs[2])};
  }

 private:
  mllm::nn::PatternSparseAttention attention_;
};

class SparseSoftmaxValueTraceModule final : public mllm::nn::Module {
 public:
  explicit SparseSoftmaxValueTraceModule(const mllm::aops::SparseSoftmaxValueOpOptions& options)
      : Module("sparse_softmax_value_trace") {
    attention_ = reg<mllm::nn::SparseSoftmaxValue>("attention", options);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<mllm::AnyValue>& args) override {
    return {attention_(inputs[0], inputs[1])};
  }

 private:
  mllm::nn::SparseSoftmaxValue attention_;
};

mllm::ir::linalg::PatternSparseAttentionOp::ptr_t findPatternSparseAttentionOp(const mllm::ir::node_ptr_t& node) {
  if (node->isa_<mllm::ir::linalg::PatternSparseAttentionOp>()) {
    return node->cast_<mllm::ir::linalg::PatternSparseAttentionOp>();
  }
  if (!node->isa_<mllm::ir::Op>()) { return nullptr; }
  for (const auto& region : node->cast_<mllm::ir::Op>()->regions()) {
    for (const auto& op : region->ops()) {
      if (auto found = findPatternSparseAttentionOp(op)) { return found; }
    }
  }
  return nullptr;
}

mllm::ir::linalg::SparseSoftmaxValueOp::ptr_t findSparseSoftmaxValueOp(const mllm::ir::node_ptr_t& node) {
  if (node->isa_<mllm::ir::linalg::SparseSoftmaxValueOp>()) { return node->cast_<mllm::ir::linalg::SparseSoftmaxValueOp>(); }
  if (!node->isa_<mllm::ir::Op>()) { return nullptr; }
  for (const auto& region : node->cast_<mllm::ir::Op>()->regions()) {
    for (const auto& op : region->ops()) {
      if (auto found = findSparseSoftmaxValueOp(op)) { return found; }
    }
  }
  return nullptr;
}

TEST_F(PatternSparseAttentionTest, UsesUniformHistoryAndLocalSuffix) {
  auto query = tensorFromValues({1, 1, 1, 1}, {1.0F});
  auto key = tensorFromValues({1, 1, 12, 1}, std::vector<float>(12, 0.0F));
  auto value = tensorFromValues({1, 1, 12, 1}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
  auto options = optionsWith(0.5F);
  options.causal_mask = false;
  options.local_ratio = 0.5F;
  PatternSparseAttentionTraceModule module(options);

  const auto output = module(query, key, value)[0];

  // Retain endpoint-uniform history {0,4,8} plus local suffix {9,10,11}.
  EXPECT_NEAR(*output.cptrAt<float>({0, 0, 0, 0}), 8.0F, 1.0e-6F);
}

TEST_F(PatternSparseAttentionTest, CausalOldPrefixMasksFutureKeys) {
  auto query = tensorFromValues({1, 1, 2, 1}, {1.0F, 1.0F});
  const std::vector<float> keys = {0.0F, 1.0F, 2.0F, 100.0F};
  const std::vector<float> values = {10.0F, 20.0F, 30.0F, 40.0F};
  auto key = tensorFromValues({1, 1, 4, 1}, keys);
  auto value = tensorFromValues({1, 1, 4, 1}, values);
  auto options = optionsWith(0.5F);
  options.local_ratio = 1.0F;
  PatternSparseAttentionTraceModule module(options);

  const auto output = module(query, key, value)[0];

  EXPECT_NEAR(*output.cptrAt<float>({0, 0, 0, 0}), selectedSoftmaxValue(keys, values, {1, 2}), 1.0e-5F);
  EXPECT_NEAR(*output.cptrAt<float>({0, 0, 1, 0}), selectedSoftmaxValue(keys, values, {2, 3}), 1.0e-5F);
}

TEST_F(PatternSparseAttentionTest, AppliesInverseSquareRootHeadScale) {
  auto query = tensorFromValues({1, 1, 1, 2}, {1.0F, 0.0F});
  auto key = tensorFromValues({1, 1, 4, 2}, {0.0F, 0.0F, 100.0F, 0.0F, 200.0F, 0.0F, 2.0F, 0.0F});
  auto value = tensorFromValues({1, 1, 4, 2}, {0.0F, 0.0F, 100.0F, 200.0F, 300.0F, 400.0F, 10.0F, 20.0F});
  auto options = optionsWith(0.5F);
  options.causal_mask = false;
  options.local_ratio = 0.5F;
  PatternSparseAttentionTraceModule module(options);

  const auto output = module(query, key, value)[0];
  const float scaled_score = 2.0F / std::sqrt(2.0F);
  const float selected_weight = std::exp(scaled_score) / (1.0F + std::exp(scaled_score));
  EXPECT_NEAR(*output.cptrAt<float>({0, 0, 0, 0}), selected_weight * 10.0F, 1.0e-5F);
  EXPECT_NEAR(*output.cptrAt<float>({0, 0, 0, 1}), selected_weight * 20.0F, 1.0e-5F);
}

TEST_F(PatternSparseAttentionTest, UsesIndependentHeadRetentions) {
  auto query = tensorFromValues({1, 2, 1, 1}, {0.0F, 0.0F});
  auto key = tensorFromValues({1, 2, 10, 1}, std::vector<float>(20, 0.0F));
  std::vector<float> values;
  for (int32_t index = 1; index <= 10; ++index) { values.push_back(static_cast<float>(index)); }
  for (int32_t index = 101; index <= 110; ++index) { values.push_back(static_cast<float>(index)); }
  auto value = tensorFromValues({1, 2, 10, 1}, values);
  auto options = optionsWith(0.8F);
  options.causal_mask = false;
  options.local_ratio = 1.0F;
  options.head_retentions = {0.2F, 0.6F};
  PatternSparseAttentionTraceModule module(options);

  const auto output = module(query, key, value)[0];

  EXPECT_NEAR(*output.cptrAt<float>({0, 0, 0, 0}), 9.5F, 1.0e-6F);
  EXPECT_NEAR(*output.cptrAt<float>({0, 1, 0, 0}), 107.5F, 1.0e-6F);
}

TEST_F(PatternSparseAttentionTest, DenseTokensKeepsTheWholeShortContext) {
  auto query = tensorFromValues({1, 1, 1, 1}, {1.0F});
  auto key = tensorFromValues({1, 1, 4, 1}, std::vector<float>(4, 0.0F));
  auto value = tensorFromValues({1, 1, 4, 1}, {10.0F, 20.0F, 30.0F, 40.0F});
  auto options = optionsWith(0.8F);
  options.causal_mask = false;
  options.random_pattern = true;
  options.dense_tokens = 4;
  PatternSparseAttentionTraceModule module(options);

  const auto output = module(query, key, value)[0];

  EXPECT_NEAR(*output.cptrAt<float>({0, 0, 0, 0}), 25.0F, 1.0e-6F);
}

TEST_F(PatternSparseAttentionTest, AcceptsFp16KeyAndValue) {
  auto query = tensorFromValues({1, 1, 1, 2}, {1.0F, 0.0F});
  auto key = fp16TensorFromValues({1, 1, 2, 2}, {0.0F, 0.0F, 2.0F, 0.0F});
  auto value = fp16TensorFromValues({1, 1, 2, 2}, {4.0F, 8.0F, 12.0F, 16.0F});
  auto options = optionsWith(0.0F);
  options.causal_mask = false;
  PatternSparseAttentionTraceModule module(options);

  const auto output = module(query, key, value)[0];
  const float score = 2.0F / std::sqrt(2.0F);
  const float selected_weight = std::exp(score) / (1.0F + std::exp(score));

  EXPECT_NEAR(*output.cptrAt<float>({0, 0, 0, 0}), (1.0F - selected_weight) * 4.0F + selected_weight * 12.0F, 1.0e-3F);
  EXPECT_NEAR(*output.cptrAt<float>({0, 0, 0, 1}), (1.0F - selected_weight) * 8.0F + selected_weight * 16.0F, 1.0e-3F);
}

TEST_F(PatternSparseAttentionTest, AcceptsFp16KeyAndFloat32Value) {
  auto query = tensorFromValues({1, 1, 1, 2}, {1.0F, 0.0F});
  auto key = fp16TensorFromValues({1, 1, 2, 2}, {0.0F, 0.0F, 2.0F, 0.0F});
  auto value = tensorFromValues({1, 1, 2, 2}, {1.0003F, 2.0003F, 3.0003F, 4.0003F});
  auto options = optionsWith(0.0F);
  options.causal_mask = false;
  PatternSparseAttentionTraceModule module(options);

  const auto output = module(query, key, value)[0];
  const float score = 2.0F / std::sqrt(2.0F);
  const float selected_weight = std::exp(score) / (1.0F + std::exp(score));

  EXPECT_NEAR(*output.cptrAt<float>({0, 0, 0, 0}), (1.0F - selected_weight) * 1.0003F + selected_weight * 3.0003F, 1.0e-6F);
  EXPECT_NEAR(*output.cptrAt<float>({0, 0, 0, 1}), (1.0F - selected_weight) * 2.0003F + selected_weight * 4.0003F, 1.0e-6F);
}

TEST_F(PatternSparseAttentionTest, HmxSelectorUsesSeparateFp16KeyAndFailsClosedUntilBackendIsBound) {
  auto query = tensorFromValues({1, 1, 1, 1}, {1.0F});
  auto key = tensorFromValues({1, 1, 1, 1}, {1.0F});
  auto value = tensorFromValues({1, 1, 1, 1}, {1.0F});
  auto selector_key = fp16TensorFromValues({1, 1, 1, 1}, {1.0F});
  auto options = optionsWith(0.5F);
  options.hmx_selector = true;
  PatternSparseAttentionTraceModule module(options);

  EXPECT_THROW((void)module(query, key, value, selector_key), std::runtime_error);
}

TEST_F(PatternSparseAttentionTest, QwenHmxSelectorKeyStagingMaterializesStridedStaticCacheSlice) {
  constexpr int32_t cache_capacity = 10;
  constexpr int32_t query_heads = 4;
  constexpr int32_t kv_heads = 2;
  constexpr int32_t sequence = 3;
  constexpr int32_t head_dim = 2;
  const std::vector<float> key_values = {
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F,
  };
  auto key = tensorFromValues({1, kv_heads, sequence, head_dim}, key_values);
  const std::vector<float> value_values = {
      21.0F, 22.0F, 23.0F, 24.0F, 25.0F, 26.0F, 31.0F, 32.0F, 33.0F, 34.0F, 35.0F, 36.0F,
  };
  auto value = tensorFromValues({1, kv_heads, sequence, head_dim}, value_values);
  mllm::nn::StaticCache cache(cache_capacity, 1, query_heads, kv_heads, head_dim, mllm::kFloat32, mllm::kFloat32, mllm::kCPU,
                              false);

  const auto cached_pair = cache.updateKVCache(0, key, value);
  const auto& cached = cached_pair[0];
  const auto& cached_value = cached_pair[1];
  ASSERT_EQ(cached.shape(), (Tensor::shape_t{1, query_heads, sequence, head_dim}));
  ASSERT_GT(cached.stride()[1], sequence * head_dim);

  auto staged = mllm::models::qwen_npu::stageHmxSelectorKey(cached);
  auto staged_value = mllm::models::qwen_npu::stageHmxSparseValue(cached_value);
  ASSERT_EQ(staged.dtype(), mllm::kFloat16);
  ASSERT_EQ(staged_value.dtype(), mllm::kFloat16);
  ASSERT_EQ(staged.shape(), cached.shape());
  ASSERT_EQ(staged_value.shape(), cached_value.shape());
  ASSERT_EQ(staged.stride()[1], sequence * head_dim);
  ASSERT_EQ(staged_value.stride()[1], sequence * head_dim);
  const auto staged_f32 = staged.to(mllm::kFloat32);
  const auto staged_value_f32 = staged_value.to(mllm::kFloat32);
  const int32_t repeats = query_heads / kv_heads;
  for (int32_t head = 0; head < query_heads; ++head) {
    const int32_t kv_head = head / repeats;
    for (int32_t token = 0; token < sequence; ++token) {
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        const size_t source_index = (static_cast<size_t>(kv_head) * sequence + token) * head_dim + dim;
        EXPECT_NEAR(*staged_f32.cptrAt<float>({0, head, token, dim}), key_values[source_index], 1.0e-3F);
        EXPECT_NEAR(*staged_value_f32.cptrAt<float>({0, head, token, dim}), value_values[source_index], 1.0e-3F);
      }
    }
  }
}

TEST_F(PatternSparseAttentionTest, QwenNpuUsesShadowNpuOnlyForPrefill) {
  mllm::models::qwen_npu::QwenNPUConfig config;
  config.hidden_size = 1536;
  config.num_attention_heads = 12;
  config.num_key_value_heads = 2;
  config.head_dim = 128;
  config.num_hidden_layers = 1;
  config.max_cache_length = 8;
  config.prefill_attention_sparsity = 0.5F;
  config.prefill_attention_pattern_sparse = true;
  config.prefill_attention_hmx_selector = true;
  config.prefill_attention_keep_dense_edge_layers = false;

  std::vector<float> prefill_query_values(12 * 2 * 128, 0.0F);
  std::vector<float> prefill_key_values(2 * 2 * 128, 0.0F);
  std::vector<float> prefill_value_values(2 * 2 * 128, 0.0F);
  prefill_query_values[0] = 1.0F;
  prefill_query_values[128] = 1.0F;
  prefill_key_values[128] = 1.0F;
  for (int32_t index = 0; index < 256; ++index) { prefill_value_values[index] = static_cast<float>(index + 1); }
  auto prefill_query = tensorFromValues({1, 12, 2, 128}, prefill_query_values);
  auto prefill_key = tensorFromValues({1, 2, 2, 128}, prefill_key_values);
  auto prefill_value = tensorFromValues({1, 2, 2, 128}, prefill_value_values);
  auto prefill_sin = tensorFromValues({1, 2, 128}, std::vector<float>(256, 0.0F));
  auto prefill_cos = tensorFromValues({1, 2, 128}, std::vector<float>(256, 1.0F));
  mllm::models::qwen_npu::QwenAttentionMatmul prefill_module("qwen_prefill", config, 0);
  mllm::nn::StaticCache prefill_cache(8, 1, 12, 2, 128, mllm::kFloat32, mllm::kFloat32, mllm::kCPU, false);
  prefill_module.getKVCache().setStaticCache(&prefill_cache);
  EXPECT_EQ(prefill_module.getKVCache().getStaticCache(), &prefill_cache);

  EXPECT_THROW((void)prefill_module(prefill_query, prefill_key, prefill_value, prefill_sin, prefill_cos), std::runtime_error);

  std::vector<float> decode_query_values(12 * 128, 0.0F);
  std::vector<float> decode_key_values(2 * 128, 0.0F);
  std::vector<float> decode_value_values(2 * 128, 0.0F);
  decode_query_values[0] = 1.0F;
  decode_key_values[0] = 1.0F;
  for (int32_t index = 0; index < 128; ++index) { decode_value_values[index] = static_cast<float>(3 + index * 2); }
  auto decode_query = tensorFromValues({1, 12, 1, 128}, decode_query_values);
  auto decode_key = tensorFromValues({1, 2, 1, 128}, decode_key_values);
  auto decode_value = tensorFromValues({1, 2, 1, 128}, decode_value_values);
  auto decode_sin = tensorFromValues({1, 1, 128}, std::vector<float>(128, 0.0F));
  auto decode_cos = tensorFromValues({1, 1, 128}, std::vector<float>(128, 1.0F));
  mllm::models::qwen_npu::QwenAttentionMatmul decode_module("qwen_decode", config, 0);
  mllm::nn::StaticCache decode_cache(8, 1, 12, 2, 128, mllm::kFloat32, mllm::kFloat32, mllm::kCPU, false);
  decode_module.getKVCache().setStaticCache(&decode_cache);
  EXPECT_EQ(decode_module.getKVCache().getStaticCache(), &decode_cache);

  const auto output = decode_module(decode_query, decode_key, decode_value, decode_sin, decode_cos)[0].to(mllm::kFloat32);
  EXPECT_EQ(output.shape(), (Tensor::shape_t{1, 1, 1536}));
  for (int32_t head = 0; head < 12; ++head) {
    for (int32_t dim = 0; dim < 128; ++dim) {
      const auto expected = decode_value_values[(head / 6) * 128 + dim];
      EXPECT_NEAR(*output.cptrAt<float>({0, 0, head * 128 + dim}), expected, 1.0e-3F);
    }
  }
}

TEST_F(PatternSparseAttentionTest, QwenHmxConfiguredDensePathKeepsCanonicalCacheFloat32) {
  mllm::models::qwen_npu::QwenNPUConfig hmx_config;
  hmx_config.hidden_size = 128;
  hmx_config.num_attention_heads = 1;
  hmx_config.num_key_value_heads = 1;
  hmx_config.num_hidden_layers = 1;
  hmx_config.max_cache_length = 8;
  hmx_config.prefill_attention_pattern_sparse = true;
  hmx_config.prefill_attention_hmx_selector = true;
  hmx_config.prefill_attention_keep_dense_edge_layers = false;
  hmx_config.prefill_attention_head_retentions = {{1.0F}};

  std::vector<float> query_values(3 * 128, 0.0F);
  std::vector<float> key_values(3 * 128, 0.0F);
  std::vector<float> value_values(3 * 128, 0.0F);
  for (int32_t token = 0; token < 3; ++token) {
    for (int32_t dim = 0; dim < 128; ++dim) {
      value_values[token * 128 + dim] = 1.0003F + static_cast<float>(token) * 0.1003F + static_cast<float>(dim) * 1.0e-5F;
    }
  }

  const auto query = tensorFromValues({1, 1, 3, 128}, query_values);
  const auto key = tensorFromValues({1, 1, 3, 128}, key_values);
  const auto value = tensorFromValues({1, 1, 3, 128}, value_values);
  const auto sin = tensorFromValues({1, 3, 128}, std::vector<float>(3 * 128, 0.0F));
  const auto cos = tensorFromValues({1, 3, 128}, std::vector<float>(3 * 128, 1.0F));

  mllm::models::qwen_npu::QwenAttentionMatmul hmx_module("qwen_hmx_all_dense", hmx_config, 0);
  mllm::nn::StaticCache hmx_cache(8, 1, 1, 1, 128, mllm::kFloat32, mllm::kFloat32, mllm::kCPU, false);
  hmx_module.getKVCache().setStaticCache(&hmx_cache);
  const auto hmx_output = hmx_module(query, key, value, sin, cos)[0].to(mllm::kFloat32);

  auto dense_config = hmx_config;
  dense_config.prefill_attention_pattern_sparse = false;
  dense_config.prefill_attention_hmx_selector = false;
  dense_config.prefill_attention_head_retentions.clear();
  mllm::models::qwen_npu::QwenAttentionMatmul dense_module("qwen_dense_reference", dense_config, 0);
  mllm::nn::StaticCache dense_cache(8, 1, 1, 1, 128, mllm::kFloat32, mllm::kFloat32, mllm::kCPU, false);
  dense_module.getKVCache().setStaticCache(&dense_cache);
  const auto dense_output = dense_module(query, key, value, sin, cos)[0].to(mllm::kFloat32);

  EXPECT_EQ(hmx_cache.getKCacheBuffer(0).dtype(), mllm::kFloat32);
  EXPECT_EQ(hmx_cache.getVCacheBuffer(0).dtype(), mllm::kFloat32);
  ASSERT_EQ(hmx_output.shape(), dense_output.shape());
  for (int32_t token = 0; token < 3; ++token) {
    for (int32_t dim = 0; dim < 128; ++dim) {
      EXPECT_NEAR(*hmx_output.cptrAt<float>({0, token, dim}), *dense_output.cptrAt<float>({0, token, dim}), 1.0e-5F);
    }
  }
}

TEST_F(PatternSparseAttentionTest, QwenNpuUsesSparseSoftmaxValueFallbackForPrefill) {
  mllm::models::qwen_npu::QwenNPUConfig config;
  config.hidden_size = 128;
  config.num_attention_heads = 1;
  config.num_key_value_heads = 1;
  config.head_dim = 128;
  config.num_hidden_layers = 1;
  config.max_cache_length = 8;
  config.prefill_attention_sparsity = 0.5F;
  config.prefill_attention_pattern_sparse = false;
  config.prefill_attention_keep_dense_edge_layers = false;

  std::vector<float> query_values(2 * 128, 0.0F);
  std::vector<float> key_values(2 * 128, 0.0F);
  std::vector<float> value_values(2 * 128, 0.0F);
  query_values[0] = 1.0F;
  query_values[128] = 1.0F;
  key_values[0] = 0.5F;
  key_values[128] = 1.0F;
  for (int32_t dim = 0; dim < 128; ++dim) {
    value_values[dim] = static_cast<float>(3 + dim);
    value_values[128 + dim] = static_cast<float>(7 + dim * 2);
  }

  auto query = tensorFromValues({1, 1, 2, 128}, query_values);
  auto key = tensorFromValues({1, 1, 2, 128}, key_values);
  auto value = tensorFromValues({1, 1, 2, 128}, value_values);
  auto sin = tensorFromValues({1, 2, 128}, std::vector<float>(2 * 128, 0.0F));
  auto cos = tensorFromValues({1, 2, 128}, std::vector<float>(2 * 128, 1.0F));

  mllm::models::qwen_npu::QwenAttentionMatmul module("qwen_sparse_softmax_value", config, 0);
  mllm::nn::StaticCache cache(8, 1, 1, 1, 128, mllm::kFloat32, mllm::kFloat32, mllm::kCPU, false);
  module.getKVCache().setStaticCache(&cache);
  const auto output = module(query, key, value, sin, cos)[0].to(mllm::kFloat32);

  EXPECT_EQ(output.shape(), (Tensor::shape_t{1, 2, 128}));
  for (int32_t dim = 0; dim < 128; ++dim) {
    EXPECT_NEAR(*output.cptrAt<float>({0, 0, dim}), value_values[dim], 1.0e-5F);
    EXPECT_NEAR(*output.cptrAt<float>({0, 1, dim}), value_values[128 + dim], 1.0e-5F);
  }
}

TEST_F(PatternSparseAttentionTest, ParsesExactHeadRetentionSchedule) {
  std::istringstream profile("1 1 1\n0.5 0.25 1\n");
  const auto schedule = mllm::models::qwen_npu::parseShadowNPUHeadSchedule(profile, "test-profile", 2, 3);

  EXPECT_EQ(schedule.retentions, (std::vector<std::vector<float>>{{1.0F, 1.0F, 1.0F}, {0.5F, 0.25F, 1.0F}}));
  EXPECT_NEAR(schedule.average_retention, 4.75F / 6.0F, 1.0e-6F);
  EXPECT_EQ(schedule.dense_heads, 4);
  EXPECT_EQ(schedule.normalized_dense_layers, "0");
}

TEST_F(PatternSparseAttentionTest, RejectsShortHeadRetentionSchedule) {
  std::istringstream profile("1 1 1\n0.5 0.25\n");
  EXPECT_THROW((void)mllm::models::qwen_npu::parseShadowNPUHeadSchedule(profile, "short-profile", 2, 3), std::invalid_argument);
}

TEST_F(PatternSparseAttentionTest, RejectsExtraOrInvalidHeadRetentionSchedule) {
  std::istringstream extra_profile("1 1 1\n0.5 0.25 1\n0.75\n");
  EXPECT_THROW((void)mllm::models::qwen_npu::parseShadowNPUHeadSchedule(extra_profile, "extra-profile", 2, 3),
               std::invalid_argument);

  std::istringstream invalid_profile("1 1 1\n0.5 0 1\n");
  EXPECT_THROW((void)mllm::models::qwen_npu::parseShadowNPUHeadSchedule(invalid_profile, "invalid-profile", 2, 3),
               std::invalid_argument);
}

TEST_F(PatternSparseAttentionTest, PlansExactQnnSequenceTiles) {
  const auto plan = mllm::qnn::planQnnSequenceTiling({1, 512, 1536}, {1, 256, 1536}, 256);
  EXPECT_EQ(plan.axis, 1U);
  EXPECT_EQ(plan.tile_count, 2);
  EXPECT_EQ(plan.compiled_shape, (Tensor::shape_t{1, 256, 1536}));
}

TEST_F(PatternSparseAttentionTest, CopiesQnnSequenceTilesAcrossOuterDimensions) {
  std::vector<float> values(2 * 4 * 3);
  for (size_t index = 0; index < values.size(); ++index) { values[index] = static_cast<float>(index); }
  auto logical = tensorFromValues({2, 4, 3}, values);
  auto tile = Tensor::empty({2, 2, 3}, mllm::kFloat32, mllm::kCPU).alloc();
  auto restored = Tensor::zeros({2, 4, 3}, mllm::kFloat32, mllm::kCPU);

  mllm::qnn::copyQnnSequenceTile(tile, 0, logical, 2, 2, 1);
  mllm::qnn::copyQnnSequenceTile(restored, 2, tile, 0, 2, 1);

  for (int32_t batch = 0; batch < 2; ++batch) {
    for (int32_t sequence = 0; sequence < 4; ++sequence) {
      for (int32_t dimension = 0; dimension < 3; ++dimension) {
        const float expected = sequence < 2 ? 0.0F : logical.at<float>({batch, sequence, dimension});
        EXPECT_FLOAT_EQ(restored.at<float>({batch, sequence, dimension}), expected);
      }
    }
  }
}

TEST_F(PatternSparseAttentionTest, RejectsAmbiguousQnnSequenceTiling) {
  EXPECT_THROW((void)mllm::qnn::planQnnSequenceTiling({2, 512, 16}, {1, 256, 16}, 256), std::invalid_argument);
  EXPECT_THROW((void)mllm::qnn::planQnnSequenceTiling({1, 500, 16}, {1, 256, 16}, 256), std::invalid_argument);
  EXPECT_THROW((void)mllm::qnn::planQnnSequenceTiling({1, 256, 16}, {1, 256, 16}, 256), std::invalid_argument);
}

TEST_F(PatternSparseAttentionTest, SparseSoftmaxValueBreaksTiesByLowestKeyIndex) {
  mllm::aops::SparseSoftmaxValueOpOptions options;
  options.sparsity = 0.5F;
  options.causal_mask = false;
  SparseSoftmaxValueTraceModule module(options);

  auto logits = tensorFromValues({1, 1, 1, 4}, {1.0F, 1.0F, 1.0F, 1.0F});
  auto value = tensorFromValues({1, 1, 4, 1}, {10.0F, 20.0F, 30.0F, 40.0F});
  const auto output = module(logits, value)[0];
  EXPECT_FLOAT_EQ(*output.cptrAt<float>({0, 0, 0, 0}), 15.0F);
}

TEST_F(PatternSparseAttentionTest, SparseSoftmaxValueUsesCausalPrefixAndFp16Value) {
  mllm::aops::SparseSoftmaxValueOpOptions options;
  options.sparsity = 0.75F;
  SparseSoftmaxValueTraceModule module(options);

  auto logits = tensorFromValues({1, 1, 2, 4}, {0.0F, 1.0F, 3.0F, 100.0F, 5.0F, 4.0F, 3.0F, 2.0F});
  auto value = fp16TensorFromValues({1, 1, 4, 2}, {10.0F, 20.0F, 11.0F, 21.0F, 12.0F, 22.0F, 13.0F, 23.0F});
  const auto output = module(logits, value)[0];

  EXPECT_FLOAT_EQ(*output.cptrAt<float>({0, 0, 0, 0}), 12.0F);
  EXPECT_FLOAT_EQ(*output.cptrAt<float>({0, 0, 0, 1}), 22.0F);
  EXPECT_FLOAT_EQ(*output.cptrAt<float>({0, 0, 1, 0}), 10.0F);
  EXPECT_FLOAT_EQ(*output.cptrAt<float>({0, 0, 1, 1}), 20.0F);
}

TEST_F(PatternSparseAttentionTest, SparseSoftmaxValueUsesIndependentHeadRetentions) {
  mllm::aops::SparseSoftmaxValueOpOptions options;
  options.sparsity = 0.5F;
  options.causal_mask = false;
  options.head_retentions = {0.25F, 0.5F};
  SparseSoftmaxValueTraceModule module(options);

  auto logits = tensorFromValues({1, 2, 1, 4}, {0.0F, 1.0F, 2.0F, 3.0F, 0.0F, 1.0F, 2.0F, 3.0F});
  auto value = tensorFromValues({1, 2, 4, 1}, {0.0F, 10.0F, 20.0F, 30.0F, 100.0F, 110.0F, 120.0F, 130.0F});
  const auto output = module(logits, value)[0];

  EXPECT_FLOAT_EQ(*output.cptrAt<float>({0, 0, 0, 0}), 30.0F);
  const float weight2 = std::exp(2.0F) / (std::exp(2.0F) + std::exp(3.0F));
  EXPECT_NEAR(*output.cptrAt<float>({0, 1, 0, 0}), weight2 * 120.0F + (1.0F - weight2) * 130.0F, 2.0e-5F);
}

TEST_F(PatternSparseAttentionTest, SparseSoftmaxValueTraceAndSerializationPreserveOptions) {
  mllm::aops::SparseSoftmaxValueOpOptions options;
  options.sparsity = 0.625F;
  options.causal_mask = false;
  options.dense_tokens = 16;
  options.head_retentions = {0.25F, 0.75F};
  SparseSoftmaxValueTraceModule module(options);
  auto ir_ctx = mllm::ir::trace(module, Tensor::empty({1, 2, 2, 4}, mllm::kFloat32, mllm::kCPU),
                                Tensor::empty({1, 2, 4, 3}, mllm::kFloat16, mllm::kCPU));
  auto ir_op = findSparseSoftmaxValueOp(ir_ctx->topLevelOp());
  ASSERT_NE(ir_op, nullptr);

  const auto encoded_options = mllm::jit::binary::dumpLinalgIROptions(ir_op);
  const nlohmann::json encoded = {{"op_type", "SparseSoftmaxValue"}, {"backend", "CPU"}, {"op_options", encoded_options}};
  const auto restored = mllm::jit::interpreter::aopsFromJson(encoded);
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->getOpType(), mllm::OpTypes::kSparseSoftmaxValue);
  const auto restored_sparse = std::dynamic_pointer_cast<mllm::aops::SparseSoftmaxValueOp>(restored);
  ASSERT_NE(restored_sparse, nullptr);
  EXPECT_FLOAT_EQ(restored_sparse->options().sparsity, options.sparsity);
  EXPECT_EQ(restored_sparse->options().causal_mask, options.causal_mask);
  EXPECT_EQ(restored_sparse->options().dense_tokens, options.dense_tokens);
  EXPECT_EQ(restored_sparse->options().head_retentions, options.head_retentions);
}

TEST_F(PatternSparseAttentionTest, TraceAndSerializationRoundTripPreservesOptions) {
  auto options = optionsWith(0.75F);
  options.local_ratio = 0.25F;
  options.prefix_tokens = 2;
  options.dense_tokens = 4;
  options.random_pattern = true;
  options.random_seed = 37;
  options.pack_reserve_tokens = 512;
  options.head_retentions = {0.25F, 0.5F};
  options.layer_id = 9;
  PatternSparseAttentionTraceModule module(options);
  auto ir_ctx = mllm::ir::trace(module, Tensor::empty({1, 2, 2, 4}, mllm::kFloat32, mllm::kCPU),
                                Tensor::empty({1, 2, 4, 4}, mllm::kFloat32, mllm::kCPU),
                                Tensor::empty({1, 2, 4, 3}, mllm::kFloat32, mllm::kCPU));
  auto ir_op = findPatternSparseAttentionOp(ir_ctx->topLevelOp());
  ASSERT_NE(ir_op, nullptr);
  ASSERT_NE(ir_op->getAOp(), nullptr);
  EXPECT_EQ(ir_op->getAOp()->getOpType(), mllm::OpTypes::kPatternSparseAttention);

  const auto encoded_options = mllm::jit::binary::dumpLinalgIROptions(ir_op);
  const nlohmann::json encoded = {{"op_type", "PatternSparseAttention"}, {"backend", "CPU"}, {"op_options", encoded_options}};
  const auto restored = mllm::jit::interpreter::aopsFromJson(encoded);
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->getOpType(), mllm::OpTypes::kPatternSparseAttention);
  const auto restored_pattern = std::dynamic_pointer_cast<mllm::aops::PatternSparseAttentionOp>(restored);
  ASSERT_NE(restored_pattern, nullptr);
  EXPECT_FLOAT_EQ(restored_pattern->options().sparsity, options.sparsity);
  EXPECT_FLOAT_EQ(restored_pattern->options().local_ratio, options.local_ratio);
  EXPECT_EQ(restored_pattern->options().prefix_tokens, options.prefix_tokens);
  EXPECT_EQ(restored_pattern->options().dense_tokens, options.dense_tokens);
  EXPECT_EQ(restored_pattern->options().random_pattern, options.random_pattern);
  EXPECT_EQ(restored_pattern->options().random_seed, options.random_seed);
  EXPECT_EQ(restored_pattern->options().pack_reserve_tokens, options.pack_reserve_tokens);
  EXPECT_EQ(restored_pattern->options().head_retentions, options.head_retentions);
  EXPECT_EQ(restored_pattern->options().layer_id, options.layer_id);
}

}  // namespace
