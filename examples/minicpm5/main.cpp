// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

#include <fmt/core.h>

#include <mllm/mllm.hpp>
#include <mllm/models/minicpm5/modeling_minicpm5.hpp>
#include <mllm/models/minicpm5/tokenization_minicpm5.hpp>

using mllm::Argparse;

MLLM_MAIN({
  auto engine_args = mllm::engineArgAttach();
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Converted model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version: v1 or v2").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Official tokenizer.json path").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Runtime config path").required(true);
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("Run one prompt non-interactively").required(false);
  auto& system = Argparse::add<std::string>("--system").help("Optional system message").required(false);
  auto& enable_thinking =
      Argparse::add<bool>("--enable_thinking").help("Open a thinking block in the generation prompt").required(false);
  auto& max_new_tokens = Argparse::add<int>("-g|--max_new_tokens").help("Maximum generated tokens per prompt").required(false);
  auto& print_token_ids = Argparse::add<bool>("--print_token_ids").help("Print generated token IDs to stderr").required(false);

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
    throw std::invalid_argument("model_version must be either v1 or v2");
  }

  int exit_code = 0;
  {
    const auto config = mllm::models::minicpm5::MiniCPM5Config(config_path.get());
    int generation_limit = max_new_tokens.isSet() ? max_new_tokens.get() : 64;
    if (generation_limit <= 0 || generation_limit > config.max_cache_length) {
      throw std::invalid_argument("max_new_tokens must be between 1 and max_cache_length");
    }
    if (prompt.isSet() && prompt.get().empty()) { throw std::invalid_argument("prompt must not be empty"); }

    auto parameters = mllm::load(model_path.get(), file_version);
    mllm::models::minicpm5::validateModelConfigMatch(config, parameters);
    auto tokenizer = mllm::models::minicpm5::MiniCPM5Tokenizer(tokenizer_path.get());
    auto model = mllm::models::minicpm5::MiniCPM5ForCausalLM(config);
    model.load(parameters);

    fmt::print("\n{:*^60}\n", prompt.isSet() ? " MiniCPM5-1B One-shot CLI " : " MiniCPM5-1B Interactive CLI ");
    if (!prompt.isSet()) fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

    while (true) {
      std::string prompt_text = prompt.isSet() ? prompt.get() : "";
      if (!prompt.isSet()) {
        fmt::print("Prompt text (or 'exit/quit'): ");
        if (!std::getline(std::cin, prompt_text) || prompt_text == "exit" || prompt_text == "quit") break;
      }
      if (prompt_text.empty()) continue;

      try {
        model.resetState();
        const auto inputs = tokenizer.convertMessage({
            .prompt = prompt_text,
            .system = system.isSet() ? system.get() : "",
            .enable_thinking = enable_thinking.isSet() && enable_thinking.get(),
        });
        const int32_t prompt_tokens = inputs.at("sequence").shape()[1];
        if (prompt_tokens + generation_limit - 1 > config.max_cache_length) {
          throw std::invalid_argument(fmt::format("prompt token count ({}) plus max_new_tokens ({}) exceeds "
                                                  "max_cache_length ({})",
                                                  prompt_tokens, generation_limit, config.max_cache_length));
        }

        fmt::print("\nResponse: ");
        mllm::models::minicpm5::MiniCPM5StreamingUtf8Decoder decoder;
        for (auto& step : model.chat(inputs, {{"max_length", mllm::AnyValue(generation_limit)}})) {
          if (print_token_ids.isSet() && print_token_ids.get()) fmt::print(stderr, "TOKEN_ID:{}\n", step.cur_token_id);
          fmt::print("{}", decoder.append(tokenizer.detokenizeBytes(step.cur_token_id)));
          std::fflush(stdout);
        }
        fmt::print("{}\n{}\n", decoder.finish(), std::string(60, '-'));
      } catch (const std::exception& error) {
        fmt::print("\nError: {}\n{}\n", error.what(), std::string(60, '-'));
        if (prompt.isSet()) exit_code = 1;
      }
      if (prompt.isSet()) break;
    }
    model.perfSummary();
  }

  mllm::print("\n");
  mllm::memoryReport();
  return exit_code;
})
