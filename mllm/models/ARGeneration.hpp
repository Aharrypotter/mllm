// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <iterator>
#include <chrono>
#include <unordered_set>

#include "mllm/core/Tensor.hpp"
#include "mllm/utils/AnyValue.hpp"
#include "mllm/compile/ir/Node.hpp"

namespace mllm::models {

using ARGenerationOutputPast = std::unordered_map<std::string, Tensor>;
using ARGenerationArgs = std::unordered_map<std::string, AnyValue>;
using IROutput = std::unordered_map<std::string, ir::IRContext::ptr_t>;

struct ARGenerationStep {
  int64_t current_step = -1;
  int64_t cur_token_id = 0;
};

// Snapshot of request-scoped autoregressive generation measurements.
// valid gates timing fields; completed reports whether generation reached EOS
// or the configured maximum length. All *_us fields are microseconds.
// prefill_tokens counts input sequence tokens, generated_tokens includes a
// sampled EOS token, and decode_steps excludes the first generated token.
struct ARGenerationPerformanceStats {
  bool valid = false;
  bool completed = false;
  int64_t total_duration_us = 0;
  int64_t prefill_duration_us = 0;
  int64_t decode_duration_us = 0;
  int64_t ttft_duration_us = 0;
  int64_t prefill_tokens = 0;
  int64_t generated_tokens = 0;
  int64_t decode_steps = 0;
};

class ARGeneration;
struct ARGenerationChatContext;

class ARGenerationChatIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = ARGenerationStep;
  using difference_type = std::ptrdiff_t;
  using pointer = const ARGenerationStep*;
  using reference = const ARGenerationStep&;

  ARGenerationChatIterator(ARGeneration& gen, const ARGenerationOutputPast& initial_input, const ARGenerationArgs& args);

  ARGenerationChatIterator();

  reference operator*() const;

  pointer operator->() const;

  ARGenerationChatIterator& operator++();

  bool operator==(const ARGenerationChatIterator& other) const;

  bool operator!=(const ARGenerationChatIterator& other) const;

 private:
  void step();

  ARGeneration* gen_ = nullptr;
  ARGenerationOutputPast current_input_;
  ARGenerationArgs args_;
  ARGenerationStep current_step_;
  bool finished_ = true;
  int64_t step_count_ = 0;

  float temperature_;
  int top_k_;
  float top_p_;
  int max_length_;
  int eos_token_id_;
  bool do_sample_;
};

struct ARGenerationChatContext {
  ARGenerationChatContext(ARGeneration& gen, const ARGenerationOutputPast& input, const ARGenerationArgs& args);

  ARGenerationChatIterator begin();

  ARGenerationChatIterator end();

 private:
  ARGeneration& gen_;
  ARGenerationOutputPast input_;
  ARGenerationArgs args_;
};

class ARGeneration {
 public:
  friend struct ARGenerationChatIterator;
  friend struct ARGenerationChatContext;

  virtual ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) = 0;

  virtual ARGenerationOutputPast generate(const ARGenerationOutputPast& input, const ARGenerationArgs& args);

  virtual void streamGenerate(const ARGenerationOutputPast& input, const ARGenerationArgs& args,
                              const std::function<void(int64_t)>& callback);

  virtual IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args);

  // Returns the current request's timing and token-count snapshot.
  [[nodiscard]] ARGenerationPerformanceStats perfStats() const;

  virtual void perfSummary();

  int64_t sampleGreedy(Tensor& logits);

  int64_t sampleTemperature(Tensor& logits, float temperature);

  int64_t sampleTopK(Tensor& logits, int k, float temperature);

  int64_t sampleTopP(Tensor& logits, float p, float temperature);

  ARGenerationChatContext chat(const ARGenerationOutputPast& input, const ARGenerationArgs& args = {});

  int64_t categoricalSample(const Tensor& probs);

  Tensor getLastLogits(Tensor& logits);

  int sampleFromDistribution(const std::vector<float>& probs);

  void prefillEventStartTimePoint();

  void prefillEventEndTimePoint();

  void decodeEventStartTimePoint();

  void decodeEventEndTimePoint();

  void customEventStartTimePoint(const std::string& name);

  void customEventEndTimePoint(const std::string& name);

 protected:
  using PerformanceClock = std::chrono::steady_clock;

  void firstTokenEventTimePoint();

  void generationEventEndTimePoint();

  [[nodiscard]] bool isEosToken(int64_t token_id, int64_t primary_eos_token_id) const;

  bool do_sample_ = false;
  int eos_token_id_ = -1;
  std::unordered_set<int64_t> additional_eos_token_ids_;
  int max_length_ = 1024;

  int64_t ar_steps_ = 0;
  int64_t ar_prefill_tokens_ = 0;

  PerformanceClock::time_point llm_prefill_start_time_;
  PerformanceClock::time_point llm_prefill_end_time_;
  PerformanceClock::time_point llm_first_token_time_;
  PerformanceClock::time_point llm_decode_start_time_;
  PerformanceClock::time_point llm_decode_end_time_;
  std::chrono::microseconds llm_decode_duration_{0};
  bool prefill_started_ = false;
  bool prefill_finished_ = false;
  bool first_token_recorded_ = false;
  bool decode_event_active_ = false;
  bool generation_completed_ = false;
  std::unordered_map<std::string, std::pair<PerformanceClock::time_point, PerformanceClock::time_point>> custom_event_time_;
  std::unordered_set<std::string> completed_custom_events_;
};

}  // namespace mllm::models
