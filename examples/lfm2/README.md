# LFM2.5-2.6B on ARM CPU

This example supports the text-only
[`LiquidAI/LFM2.5-2.6B`](https://huggingface.co/LiquidAI/LFM2.5-2.6B)
checkpoint. The runtime binds the official 30-layer physical schedule: 22
stateful short-convolution layers and 8 full-attention layers.

The attention path uses 32 query heads and 8 native KV heads. Only the eight
full-attention layers allocate KV-cache slots; the cache never expands KV
history to query-head count. Each convolution layer independently retains the
two historical FP32 samples required by its three-tap causal kernel across
prefill and decode. Batch size is 1 and this
product configuration limits the runtime cache to 2048 tokens.

## Validate and convert

Run from the repository root. The first audit rejects architecture, physical
layer schedule, tensor-shape, or quantization-recipe drift before conversion.

```bash
python examples/lfm2/validate_checkpoint.py /path/to/LFM2.5-2.6B

python -m pymllm.mobile.utils.mllm_convertor \
  --input_path /path/to/LFM2.5-2.6B \
  --output_path /path/to/lfm2.5-2.6b-w4a32-kai.mllm \
  --model_name LFM2.5-2.6B \
  --cfg_path examples/lfm2/quant_cfg_2.6B_w4a32_kai.json \
  --pipeline w4a32_kai_pipeline \
  --format v2 \
  --verbose

python examples/lfm2/validate_converted_model.py \
  /path/to/lfm2.5-2.6b-w4a32-kai.mllm \
  /path/to/LFM2.5-2.6B
```

Linear weights use the existing KleidiAI dynamic-INT8-activation / INT4-weight
packing path. The depthwise convolution, norms, and lookup embedding remain
FP32. Because the checkpoint ties its output head to the embedding, conversion
retains `model.embed_tokens.weight` and creates the packed
`lm_head_out.weight` alias used by the runtime output projection.

## Build and run

```bash
cmake -S . -B build -DMLLM_ENABLE_EXAMPLE=ON
cmake --build build --target mllm-lfm2-runner -j

build/bin/mllm-lfm2-runner \
  --model_path /path/to/lfm2.5-2.6b-w4a32-kai.mllm \
  --model_version v2 \
  --tokenizer_path /path/to/LFM2.5-2.6B/tokenizer.json \
  --config_path examples/lfm2/config_2.6B_w4a32_kai.json \
  --prompt "Explain how to make mobile language-model inference reliable, covering correctness, state reset, artifact verification, and performance measurement." \
  --max_new_tokens 128 \
  --min_new_tokens 128 \
  --print_token_ids
```

For the Android ARM build, keep runtime OpenMP enabled but configure the CPU
backend without backend-wide OpenMP. LFM2.5 still selects the shared W4A32
I8MM prefill path and retained KAI decode workspace; avoiding OpenMP regions in
every backend operator preserves decode latency for its 8-attention / 22-conv
hybrid schedule.

```bash
cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DMLLM_CROSS_COMPILE=ON \
  -DMLLM_BUILD_ARM_BACKEND=ON \
  -DMLLM_ENABLE_EXAMPLE=ON \
  -DMLLM_ARM_CPU_BACKEND_USE_OPENMP=OFF

cmake --build build-android --target mllm-lfm2-runner -j
```

Omit `--prompt` for the interactive loop. Each prompt begins with
`resetState()`, clearing all eight logical attention slots and all 22
short-convolution histories. The tokenizer applies the checkpoint's byte-level
BPE contract and ends the generation prompt exactly at `<think>`. For a demo
receipt, use `demo_prompt.txt` unchanged on the host and Android, request a long
deterministic continuation with equal `max_new_tokens` and `min_new_tokens`,
and retain the printed prompt, response, token IDs, and generated-token count.

## Product benchmark records

Benchmark mode requires a file-backed prompt, its frozen token count, exact
model/source identities, and a fresh JSONL destination. It resets all model
state between requests, forces an exact generated-token count, and records
prefill, TTFT, decode, wall time, affinity, CPU-frequency, governor, and thermal
telemetry for every sample.

```bash
build/bin/mllm-lfm2-runner \
  --model_path /path/to/lfm2.5-2.6b-w4a32-kai.mllm \
  --model_version v2 \
  --tokenizer_path /path/to/LFM2.5-2.6B/tokenizer.json \
  --config_path examples/lfm2/config_2.6B_w4a32_kai.json \
  --prompt_file /path/to/prompt.txt \
  --expected_prompt_tokens PROMPT_TOKENS \
  --max_new_tokens 32 \
  --benchmark_warmup 1 \
  --benchmark_samples 5 \
  --benchmark_jsonl /path/to/fresh-results.jsonl \
  --benchmark_variant MODEL_SHA256 \
  --benchmark_source_manifest SOURCE_MANIFEST_SHA256
```

Use `--system_prompt` for the optional system role. `--tools_json` accepts
either one JSON tool-schema object or an array of schemas. Objects are rendered
with the pinned template's Python `json.dumps` spacing; string elements are
inserted byte-for-byte into `List of tools: [...]`. The runner returns the
model's tool-call text unchanged and does not parse or execute calls.

This implementation and its host tests establish local source/build
correctness. They do not by themselves claim output quality, Android
execution, or device performance.
