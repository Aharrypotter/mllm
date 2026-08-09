# Qwen3.5 on ARM CPU

This example supports the text towers of
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) and
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), plus single-image
inference for Qwen3.5-0.8B.

| Model | Text | Single image | Hidden size | GDN / full-attention layers |
| --- | --- | --- | ---: | ---: |
| 0.8B | Yes | Yes | 1024 | 18 / 6 |
| 4B | Yes | No | 2560 | 24 / 8 |

All configurations support batch size 1 and a maximum cache length of 2048
tokens. `Qwen3_5ForCausalLM::resetState()` clears the full-attention KV cache
and every GDN recurrent/convolution state before each prompt.

The multimodal path accepts one still image with aspect ratio at most 200. It
smart-resizes the image to a 65,536--262,144 pixel budget with both dimensions
divisible by 32. Video, multiple images, deep-stack visual features, MTP, and
multimodal benchmark mode are not supported.

## Quantization

The user-facing W4A8 configuration uses dynamic INT8 activations with INT4
weights while retaining FP32 operator inputs and outputs. The existing
`w4a32_kai` names and conversion pipeline remain for compatibility with mllm
tooling.

Text and vision `nn::Linear` weights use the KAI path. Vision Conv3D, learned
position embeddings, LayerNorm parameters, text embeddings, recurrent
parameters, and convolution weights remain FP32. The tied text embedding is
retained for token lookup and separately packed as `lm_head_out.weight`.

## Convert a checkpoint

Run commands from the repository root. Validate the checkpoint and conversion
recipe before creating a model file, then audit the complete V2 descriptor and
offset table after conversion.

### Qwen3.5-0.8B text-only

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

### Qwen3.5-0.8B single-image

```bash
python examples/qwen3_5/validate_checkpoint.py \
  /path/to/Qwen3.5-0.8B \
  --quant-config examples/qwen3_5/quant_cfg_0.8B_multimodal_w4a32_kai.json \
  --runtime-config examples/qwen3_5/config_0.8B_multimodal_w4a32_kai.json

python -m pymllm.mobile.utils.mllm_convertor \
  --input_path /path/to/Qwen3.5-0.8B \
  --output_path /path/to/qwen3.5-0.8b-multimodal-w4a32-kai.mllm \
  --model_name Qwen3.5-0.8B-Multimodal \
  --cfg_path examples/qwen3_5/quant_cfg_0.8B_multimodal_w4a32_kai.json \
  --pipeline w4a32_kai_pipeline \
  --include_prefix model.language_model. \
  --include_prefix model.visual. \
  --format v2 \
  --verbose

python examples/qwen3_5/validate_converted_model.py \
  /path/to/qwen3.5-0.8b-multimodal-w4a32-kai.mllm \
  /path/to/Qwen3.5-0.8B \
  --quant-config examples/qwen3_5/quant_cfg_0.8B_multimodal_w4a32_kai.json \
  --runtime-config examples/qwen3_5/config_0.8B_multimodal_w4a32_kai.json \
  --model-name Qwen3.5-0.8B-Multimodal
```

### Qwen3.5-4B text-only

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

The 4B converted tensor data exceeds 4 GiB. Use model-file V2 so descriptor
sizes and offsets remain 64-bit, and provide at least 32 GiB of available host
memory plus working disk space for conversion.

## Run

```bash
# Qwen3.5-0.8B single-image
mllm-qwen3-5-runner \
  --model_path /path/to/qwen3.5-0.8b-multimodal-w4a32-kai.mllm \
  --model_version v2 \
  --tokenizer_path /path/to/Qwen3.5-0.8B/tokenizer.json \
  --config_path examples/qwen3_5/config_0.8B_multimodal_w4a32_kai.json \
  --image_path /path/to/image.jpg \
  --prompt "Describe the image." \
  --max_new_tokens 32

# Qwen3.5-4B text-only
mllm-qwen3-5-runner \
  --model_path /path/to/qwen3.5-4b-w4a32-kai.mllm \
  --model_version v2 \
  --tokenizer_path /path/to/Qwen3.5-4B/tokenizer.json \
  --config_path examples/qwen3_5/config_4B_w4a32_kai.json \
  --prompt "Give a one-sentence introduction." \
  --max_new_tokens 32
```

Omit `--prompt` for the interactive loop. The same optional image is used for
each independent prompt in that process. Omit `--image_path` to run text-only
inference with either the text-only or multimodal 0.8B model.
