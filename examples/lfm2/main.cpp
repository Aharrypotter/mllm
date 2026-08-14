#include <cstdio>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include <mllm/mllm.hpp>
#include <mllm/engine/Context.hpp>
#include <mllm/models/lfm2/modeling_lfm2.hpp>
#include <mllm/models/lfm2/tokenization_lfm2.hpp>

#include "benchmark_harness.hpp"

using mllm::Argparse;

namespace {

auto pythonJson(const nlohmann::ordered_json& value) -> std::string {
  if (value.is_array()) {
    std::string output = "[";
    for (size_t index = 0; index < value.size(); ++index) {
      if (index != 0) output += ", ";
      output += pythonJson(value[index]);
    }
    return output + "]";
  }
  if (value.is_object()) {
    std::string output = "{";
    size_t index = 0;
    for (const auto& [key, item] : value.items()) {
      if (index++ != 0) output += ", ";
      output += nlohmann::ordered_json(key).dump() + ": " + pythonJson(item);
    }
    return output + "}";
  }
  return value.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::strict);
}

}  // namespace

MLLM_MAIN({
  auto engine_args = mllm::engineArgAttach();
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Converted model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version: v1 or v2").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer JSON path").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Runtime config path").required(true);
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("Run one prompt non-interactively").required(false);
  auto& prompt_file = Argparse::add<std::string>("--prompt_file").help("Read one benchmark prompt from a file").required(false);
  auto& system_prompt = Argparse::add<std::string>("--system_prompt").help("Optional system prompt").required(false);
  auto& tools_json = Argparse::add<std::string>("--tools_json")
                         .help("JSON file containing one tool schema or an array of schemas")
                         .required(false);
  auto& max_new_tokens = Argparse::add<int>("-g|--max_new_tokens").help("Maximum generated tokens").required(false);
  auto& min_new_tokens =
      Argparse::add<int>("--min_new_tokens").help("Minimum generated tokens before EOS can stop generation").required(false);
  auto& print_token_ids = Argparse::add<bool>("--print_token_ids").help("Print generated token IDs").required(false);
  auto& benchmark_warmup =
      Argparse::add<int>("--benchmark_warmup").help("Unrecorded benchmark warmup requests").required(false);
  auto& benchmark_samples = Argparse::add<int>("--benchmark_samples").help("Measured benchmark requests").required(false);
  auto& benchmark_jsonl = Argparse::add<std::string>("--benchmark_jsonl").help("Fresh JSONL output path").required(false);
  auto& benchmark_variant =
      Argparse::add<std::string>("--benchmark_variant").help("Bound model artifact identity").required(false);
  auto& benchmark_source_manifest =
      Argparse::add<std::string>("--benchmark_source_manifest").help("Bound dirty-source manifest SHA-256").required(false);
  auto& expected_prompt_tokens =
      Argparse::add<int>("--expected_prompt_tokens").help("Fail if tokenized prompt length differs").required(false);

  for (int index = 1; index < argc; ++index) {
    if (std::string(argv[index]) == "-h" || std::string(argv[index]) == "--help") {
      Argparse::printHelp();
      return 0;
    }
  }
  Argparse::parse(argc, argv);
  mllm::configEngineWithArgs(engine_args);
  (void)help;

  mllm::ModelFileVersion file_version;
  if (model_version.get() == "v1") {
    file_version = mllm::ModelFileVersion::kV1;
  } else if (model_version.get() == "v2") {
    file_version = mllm::ModelFileVersion::kV2;
  } else {
    throw std::invalid_argument("model_version must be v1 or v2");
  }

  auto cfg = mllm::models::lfm2::Lfm2Config(config_path.get());
  int generation_limit = max_new_tokens.isSet() ? max_new_tokens.get() : 64;
  if (generation_limit <= 0 || generation_limit > cfg.max_cache_length) {
    throw std::invalid_argument("max_new_tokens must be between 1 and max_cache_length");
  }
  int minimum_generation = min_new_tokens.isSet() ? min_new_tokens.get() : 0;
  if (minimum_generation < 0 || minimum_generation > generation_limit) {
    throw std::invalid_argument("min_new_tokens must be between 0 and max_new_tokens");
  }
  const bool benchmark_mode = prompt_file.isSet() || benchmark_warmup.isSet() || benchmark_samples.isSet()
                              || benchmark_jsonl.isSet() || benchmark_variant.isSet() || benchmark_source_manifest.isSet()
                              || expected_prompt_tokens.isSet();
  if (prompt.isSet() && prompt_file.isSet()) throw std::invalid_argument("prompt and prompt_file are mutually exclusive");
  if (benchmark_mode) {
    if (!prompt_file.isSet() || !benchmark_samples.isSet() || !benchmark_jsonl.isSet() || !benchmark_variant.isSet()
        || !benchmark_source_manifest.isSet() || !expected_prompt_tokens.isSet()) {
      throw std::invalid_argument("benchmark mode requires prompt_file, benchmark_samples, benchmark_jsonl, "
                                  "benchmark_variant, benchmark_source_manifest, and expected_prompt_tokens");
    }
    if (benchmark_samples.get() <= 0 || (benchmark_warmup.isSet() && benchmark_warmup.get() < 0)) {
      throw std::invalid_argument("benchmark sample counts must be non-negative and measured samples must be positive");
    }
    if (generation_limit < 2) throw std::invalid_argument("benchmark max_new_tokens must be at least 2");
    if (tools_json.isSet() || system_prompt.isSet()) {
      throw std::invalid_argument("benchmark mode does not accept system_prompt or tools_json");
    }
    if (min_new_tokens.isSet()) throw std::invalid_argument("benchmark mode forces an exact generated-token count");
    const std::filesystem::path output_path(benchmark_jsonl.get());
    std::error_code error;
    if (std::filesystem::exists(output_path) && std::filesystem::file_size(output_path, error) != 0) {
      throw std::invalid_argument("benchmark_jsonl must be new or empty");
    }
  }
  const auto model_load_start = std::chrono::steady_clock::now();
  auto parameters = mllm::load(model_path.get(), file_version);
  mllm::models::lfm2::validateModelConfigMatch(cfg, parameters);
  auto tokenizer = mllm::models::lfm2::Lfm2Tokenizer(tokenizer_path.get());
  auto model = mllm::models::lfm2::Lfm2ForCausalLM(cfg);
  model.load(parameters);
  const auto model_load_end = std::chrono::steady_clock::now();
  const auto model_load_duration_us =
      std::chrono::duration_cast<std::chrono::microseconds>(model_load_end - model_load_start).count();
  fmt::print("LFM2.5-2.6B: {} layers ({} attention + {} short convolution)\n", cfg.num_hidden_layers, cfg.numAttentionLayers(),
             cfg.numConvLayers());

  int exit_code = 0;
  std::vector<std::string> raw_tools;
  if (tools_json.isSet()) {
    std::ifstream stream(tools_json.get(), std::ios::binary);
    if (!stream) throw std::invalid_argument("unable to read tools_json");
    nlohmann::ordered_json tools;
    stream >> tools;
    if (tools.is_object()) {
      raw_tools.push_back(pythonJson(tools));
    } else if (tools.is_array()) {
      for (const auto& tool : tools) { raw_tools.push_back(tool.is_string() ? tool.get<std::string>() : pythonJson(tool)); }
    } else {
      throw std::invalid_argument("tools_json must contain an object or an array of strings");
    }
  }
  if (benchmark_mode) {
    std::ifstream prompt_stream(prompt_file.get(), std::ios::binary);
    if (!prompt_stream) throw std::invalid_argument("unable to read prompt_file");
    std::string benchmark_prompt(std::istreambuf_iterator<char>(prompt_stream), {});
    while (!benchmark_prompt.empty() && (benchmark_prompt.back() == '\n' || benchmark_prompt.back() == '\r')) {
      benchmark_prompt.pop_back();
    }
    if (benchmark_prompt.empty()) throw std::invalid_argument("prompt_file must not be empty");
    const auto inputs = tokenizer.convertMessage({.prompt = benchmark_prompt});
    const auto prompt_tokens = inputs.at("sequence").shape()[1];
    if (prompt_tokens != expected_prompt_tokens.get()) {
      throw std::invalid_argument(
          fmt::format("prompt token count {} differs from expected {}", prompt_tokens, expected_prompt_tokens.get()));
    }
    if (prompt_tokens + generation_limit - 1 > cfg.max_cache_length) {
      throw std::invalid_argument("benchmark prompt plus generation exceeds max_cache_length");
    }

    std::ofstream jsonl(benchmark_jsonl.get(), std::ios::out | std::ios::trunc);
    if (!jsonl) throw std::invalid_argument("unable to open benchmark_jsonl");
    const int warmups = benchmark_warmup.isSet() ? benchmark_warmup.get() : 0;
    for (int request = 0; request < warmups + benchmark_samples.get(); ++request) {
      const bool warmup = request < warmups;
      model.resetState();
      const auto telemetry_before = mllm::examples::lfm2::benchmark::captureTelemetry();
      std::vector<int64_t> generated_token_ids;
      const auto request_start = std::chrono::steady_clock::now();
      model.streamGenerate(inputs,
                           {{"max_length", mllm::AnyValue(generation_limit)},
                            {"min_new_tokens", mllm::AnyValue(generation_limit)},
                            {"do_sample", mllm::AnyValue(false)}},
                           [&](int64_t token_id) { generated_token_ids.push_back(token_id); });
      const auto request_end = std::chrono::steady_clock::now();
      const auto stats = model.perfStats();
      std::vector<std::string> invalid_reasons;
      if (!stats.valid) invalid_reasons.push_back("invalid_performance_stats");
      if (!stats.completed) invalid_reasons.push_back("incomplete_generation");
      if (stats.prefill_tokens != prompt_tokens) invalid_reasons.push_back("prefill_token_count_mismatch");
      if (stats.generated_tokens != generation_limit || stats.decode_steps != generation_limit - 1
          || generated_token_ids.size() != static_cast<size_t>(generation_limit)) {
        invalid_reasons.push_back("generation_length_mismatch");
      }
      nlohmann::json record = {
          {"schema", "mllm.lfm25.product_benchmark.v1"},
          {"variant", benchmark_variant.get()},
          {"source_manifest_sha256", benchmark_source_manifest.get()},
          {"request_index", request},
          {"warmup", warmup},
          {"prompt_tokens", prompt_tokens},
          {"max_new_tokens", generation_limit},
          {"cpu_op_threads", mllm::Context::instance().getCpuOpThreads()},
          {"model_load_duration_us", model_load_duration_us},
          {"request_wall_duration_us",
           std::chrono::duration_cast<std::chrono::microseconds>(request_end - request_start).count()},
          {"generated_token_ids", generated_token_ids},
          {"telemetry_before", telemetry_before},
          {"telemetry_after", mllm::examples::lfm2::benchmark::captureTelemetry()},
          {"stats",
           {{"valid", stats.valid},
            {"completed", stats.completed},
            {"total_duration_us", stats.total_duration_us},
            {"prefill_duration_us", stats.prefill_duration_us},
            {"decode_duration_us", stats.decode_duration_us},
            {"ttft_duration_us", stats.ttft_duration_us},
            {"prefill_tokens", stats.prefill_tokens},
            {"generated_tokens", stats.generated_tokens},
            {"decode_steps", stats.decode_steps}}},
          {"invalid_reasons", invalid_reasons},
          {"status", invalid_reasons.empty() ? "ok" : "invalid"},
      };
      jsonl << record.dump() << '\n';
      jsonl.flush();
      if (!jsonl) throw std::runtime_error("failed to write benchmark_jsonl");
      if (!invalid_reasons.empty()) {
        exit_code = 2;
        break;
      }
    }
    if (exit_code == 0) fmt::print("Benchmark records: {}\n", benchmark_jsonl.get());
  } else
    while (true) {
      std::string prompt_text = prompt.isSet() ? prompt.get() : "";
      if (!prompt.isSet()) {
        fmt::print("Prompt (exit/quit to stop): ");
        if (!std::getline(std::cin, prompt_text) || prompt_text == "exit" || prompt_text == "quit") break;
      }
      if (prompt_text.empty()) {
        if (prompt.isSet()) exit_code = 1;
        if (prompt.isSet()) break;
        continue;
      }
      try {
        model.resetState();
        auto inputs = tokenizer.convertMessage(
            {.prompt = prompt_text, .system_prompt = system_prompt.isSet() ? system_prompt.get() : "", .tools = raw_tools});
        const auto prompt_tokens = inputs.at("sequence").shape()[1];
        if (prompt_tokens + generation_limit - 1 > cfg.max_cache_length) {
          throw std::invalid_argument("prompt plus generation exceeds max_cache_length");
        }
        if (prompt.isSet()) fmt::print("Prompt: {}\n", prompt_text);
        fmt::print("Response: ");
        mllm::models::lfm2::StreamingUtf8Decoder decoder;
        int generated_tokens = 0;
        for (const auto& step : model.chat(inputs, {{"max_length", mllm::AnyValue(generation_limit)},
                                                    {"min_new_tokens", mllm::AnyValue(minimum_generation)},
                                                    {"do_sample", mllm::AnyValue(false)}})) {
          ++generated_tokens;
          if (print_token_ids.isSet() && print_token_ids.get()) fmt::print(stderr, "TOKEN_ID:{}\n", step.cur_token_id);
          fmt::print("{}", decoder.append(tokenizer.detokenizeBytes(step.cur_token_id)));
          std::fflush(stdout);
        }
        fmt::print("{}\n", decoder.finish());
        fmt::print(stderr, "GENERATED_TOKEN_COUNT:{}\n", generated_tokens);
      } catch (const std::exception& error) {
        fmt::print(stderr, "LFM2 generation failed: {}\n", error.what());
        exit_code = 1;
      }
      if (prompt.isSet()) break;
    }
  return exit_code;
})
