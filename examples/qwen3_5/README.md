# Qwen3.5 0.8B and 4B on ARM CPU

This example runs the text towers of
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) and
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B). Both checkpoints
also contain a vision tower and an MTP layer; those components are
intentionally excluded from the CPU model file.

| Model | Hidden size | Layers | GDN layers | Full-attention layers | GDN key/value heads |
|-------|------------:|-------:|-----------:|----------------------:|--------------------:|
| 0.8B  | 1024 | 24 | 18 | 6 | 16 / 16 |
| 4B    | 2560 | 32 | 24 | 8 | 16 / 32 |

Both mobile configurations limit the cache to 2048 tokens and support batch
size 1. The GDN recurrence and its depthwise-convolution history are stateful
across prefill and decode. `Qwen3_5ForCausalLM::resetState()` clears those
states together with the full-attention KV cache.

## Quantization

The user-facing W4A8 configuration uses dynamic INT8 activations with INT4
weights while retaining FP32 operator inputs and outputs. The existing
`w4a32_kai` configuration names and conversion pipeline are retained for
compatibility with mllm tooling.

Every `nn::Linear`, including the small GDN `in_proj_a` and `in_proj_b` gates,
is packed for the configured KAI runtime. Convolution weights, recurrent
parameters, embeddings, and norms stay in float32. The tied embedding matrix is
retained for token lookup and separately packed as `lm_head_out.weight` for
KAI.

## Convert a checkpoint

Run these commands from the repository root. The checkpoint audit verifies the
official text architecture, tensor shapes, and quantization coverage before
conversion.

### Qwen3.5-0.8B

```bash
python examples/qwen3_5/validate_checkpoint.py \
  /path/to/Qwen3.5-0.8B \
  --quant-config examples/qwen3_5/quant_cfg_0.8B_w4a32_kai.json

python -m pymllm.mobile.utils.mllm_convertor \
  --input_path /path/to/Qwen3.5-0.8B \
  --output_path /path/to/qwen3.5-0.8b-w4a32-kai.mllm \
  --model_name Qwen3.5-0.8B \
  --cfg_path examples/qwen3_5/quant_cfg_0.8B_w4a32_kai.json \
  --pipeline w4a32_kai_pipeline \
  --include_prefix model.language_model. \
  --format v2 \
  --verbose

python examples/qwen3_5/validate_converted_model.py \
  /path/to/qwen3.5-0.8b-w4a32-kai.mllm \
  /path/to/Qwen3.5-0.8B \
  --quant-config examples/qwen3_5/quant_cfg_0.8B_w4a32_kai.json \
  --model-name Qwen3.5-0.8B
```

### Qwen3.5-4B

```bash
python examples/qwen3_5/validate_checkpoint.py \
  /path/to/Qwen3.5-4B \
  --quant-config examples/qwen3_5/quant_cfg_4B_w4a32_kai.json

python -m pymllm.mobile.utils.mllm_convertor \
  --input_path /path/to/Qwen3.5-4B \
  --output_path /path/to/qwen3.5-4b-w4a32-kai.mllm \
  --model_name Qwen3.5-4B \
  --cfg_path examples/qwen3_5/quant_cfg_4B_w4a32_kai.json \
  --pipeline w4a32_kai_pipeline \
  --include_prefix model.language_model. \
  --format v2 \
  --verbose

python examples/qwen3_5/validate_converted_model.py \
  /path/to/qwen3.5-4b-w4a32-kai.mllm \
  /path/to/Qwen3.5-4B \
  --quant-config examples/qwen3_5/quant_cfg_4B_w4a32_kai.json \
  --model-name Qwen3.5-4B
```

Use the V2 model-file format for the 4B conversion. Its converted tensor data
is expected to exceed 4 GiB. On the supported 64-bit host and arm64 targets,
V2 stores descriptor sizes and offsets as 64-bit values. The converted-model
audit checks every descriptor and walks the full offset chain without loading
tensor payloads.

The official 4B conversion peaked at about 29.3 GiB RSS and produced a
4.9 GB model file. Use a host with at least 32 GiB of available memory and
about 16 GB of free disk for the checkpoint, output, and working margin.
Exact resource usage can vary by host and software environment.

## Run

Select the model and matching configuration:

```bash
# Qwen3.5-0.8B
mllm-qwen3-5-runner \
  --model_path /path/to/qwen3.5-0.8b-w4a32-kai.mllm \
  --model_version v2 \
  --tokenizer_path /path/to/Qwen3.5-0.8B/tokenizer.json \
  --config_path examples/qwen3_5/config_0.8B_w4a32_kai.json \
  --prompt "Give a one-sentence introduction." \
  --max_new_tokens 32

# Qwen3.5-4B
mllm-qwen3-5-runner \
  --model_path /path/to/qwen3.5-4b-w4a32-kai.mllm \
  --model_version v2 \
  --tokenizer_path /path/to/Qwen3.5-4B/tokenizer.json \
  --config_path examples/qwen3_5/config_4B_w4a32_kai.json \
  --prompt "Give a one-sentence introduction." \
  --max_new_tokens 32
```

Omit `--prompt` for the interactive loop. The CLI treats each prompt as an
independent conversation and resets all model state before inference.
