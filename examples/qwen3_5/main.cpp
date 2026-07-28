#include <fmt/core.h>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

#include <mllm/mllm.hpp>
#include <mllm/models/qwen3_5/modeling_qwen3_5.hpp>
#include <mllm/models/qwen3_5/tokenization_qwen3_5.hpp>
#include <mllm/preprocessor/tokenizers/Unicode.hpp>
#include <mllm/utils/AnyValue.hpp>

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer JSON path").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("Run one prompt non-interactively").required(false);
  auto& max_new_tokens = Argparse::add<int>("-g|--max_new_tokens").help("Maximum generated tokens per prompt").required(false);
  auto& print_token_ids = Argparse::add<bool>("--print_token_ids").help("Print generated token IDs to stderr").required(false);

  // Argparse validates required options during parse(), so short-circuit help
  // before parsing to make `mllm-qwen3-5-runner --help` usable on its own.
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help") {
      Argparse::printHelp();
      return 0;
    }
  }

  Argparse::parse(argc, argv);

  (void)help;

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
#endif

  int exit_code = 0;
  {
    mllm::ModelFileVersion file_version;
    if (model_version.get() == "v1") {
      file_version = mllm::ModelFileVersion::kV1;
    } else if (model_version.get() == "v2") {
      file_version = mllm::ModelFileVersion::kV2;
    } else {
      throw std::invalid_argument("model_version must be either v1 or v2");
    }

    auto cfg = mllm::models::qwen3_5::Qwen3_5Config(config_path.get());
    auto tokenizer = mllm::models::qwen3_5::Qwen3_5Tokenizer(tokenizer_path.get());
    auto model = mllm::models::qwen3_5::Qwen3_5ForCausalLM(cfg);
    int generation_limit = max_new_tokens.isSet() ? max_new_tokens.get() : 64;
    if (generation_limit <= 0 || generation_limit > cfg.max_cache_length) {
      throw std::invalid_argument("max_new_tokens must be between 1 and max_cache_length");
    }
    if (prompt.isSet() && prompt.get().empty()) { throw std::invalid_argument("prompt must not be empty"); }

    fmt::print("Qwen3.5 0.8B: {} layers ({} full attention + {} GDN)\n", cfg.num_hidden_layers, cfg.numFullAttentionLayers(),
               cfg.numGDNLayers());

    auto param = mllm::load(model_path.get(), file_version);
    model.load(param);

    fmt::print("\n{:*^60}\n", prompt.isSet() ? " Qwen3.5 One-shot CLI " : " Qwen3.5 Interactive CLI ");
    if (!prompt.isSet()) { fmt::print("Enter 'exit' or 'quit' to end the session\n\n"); }

    while (true) {
      std::string prompt_text = prompt.isSet() ? prompt.get() : "";
      if (!prompt.isSet()) {
        fmt::print("Prompt text (or 'exit/quit'): ");
        if (!std::getline(std::cin, prompt_text) || prompt_text == "exit" || prompt_text == "quit") { break; }
      }
      if (prompt_text.empty()) { continue; }

      try {
        // Each prompt is an independent conversation. Both the full-attention
        // KV cache and every GDN recurrent/conv state must start empty.
        model.resetState();
        fmt::print("Processing...\n");
        auto inputs = tokenizer.convertMessage({.prompt = prompt_text});
        const auto prompt_length = inputs.at("sequence").shape()[1];
        if (prompt_length + generation_limit - 1 > cfg.max_cache_length) {
          throw std::invalid_argument(fmt::format("prompt token count ({}) plus max_new_tokens ({}) exceeds "
                                                  "max_cache_length ({})",
                                                  prompt_length, generation_limit, cfg.max_cache_length));
        }

        fmt::print("\nResponse: ");

        for (auto& step : model.chat(inputs, {{"max_length", mllm::AnyValue(generation_limit)}})) {
          if (print_token_ids.isSet() && print_token_ids.get()) { fmt::print(stderr, "TOKEN_ID:{}\n", step.cur_token_id); }
          fmt::print("{}", mllm::preprocessor::wideString2Utf8String(tokenizer.detokenize(step.cur_token_id)));
          std::fflush(stdout);
        }

        fmt::print("\n{}\n", std::string(60, '-'));
      } catch (const std::exception& e) {
        fmt::print("\nError: {}\n{}\n", e.what(), std::string(60, '-'));
        if (prompt.isSet()) { exit_code = 1; }
      }
      if (prompt.isSet()) { break; }
    }

    model.perfSummary();
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("qwen3_5.perf");
#endif

  mllm::print("\n");
  mllm::memoryReport();
  return exit_code;
})
