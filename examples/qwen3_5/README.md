# Qwen3.5 0.8B on ARM CPU

This example runs the text tower of `Qwen/Qwen3.5-0.8B`. The checkpoint also
contains a vision tower and an MTP layer; they are intentionally excluded from
the CPU model file.

The model uses six full-attention layers and eighteen Gated Delta Net (GDN)
layers. Both the GDN recurrence and its depthwise-convolution history are
stateful across prefill and decode. `Qwen3_5ForCausalLM::resetState()` clears
those states together with the full-attention KV cache.

## Convert the checkpoint

First verify that the checkpoint architecture, tensor shapes, and quantization
coverage match this 0.8B CPU implementation:

```bash
python examples/qwen3_5/validate_checkpoint.py \
  /path/to/Qwen3.5-0.8B
```

Run the converter from the repository root:

```bash
python -m pymllm.mobile.utils.mllm_convertor \
  --input_path /path/to/Qwen3.5-0.8B \
  --output_path /path/to/qwen3.5-0.8b-w4a32-kai.mllm \
  --model_name Qwen3.5-0.8B \
  --cfg_path examples/qwen3_5/quant_cfg_0.8B_w4a32_kai.json \
  --pipeline w4a32_kai_pipeline \
  --include_prefix model.language_model. \
  --format v2 \
  --verbose
```

The tied embedding matrix is retained for token lookup and separately packed as
`lm_head_out.weight` for KAI. Every `nn::Linear`, including the small GDN
`in_proj_a` and `in_proj_b` gates, is packed for the configured KAI runtime;
convolution weights, recurrent parameters, embeddings, and norms stay in
float32.

Linear uses dynamic INT8 activations with INT4 weights while retaining FP32
operator inputs and outputs.

Audit the resulting V2 descriptors without loading the tensor data:

```bash
python examples/qwen3_5/validate_converted_model.py \
  /path/to/qwen3.5-0.8b-w4a32-kai.mllm \
  /path/to/Qwen3.5-0.8B
```

## Run

```bash
mllm-qwen3-5-runner \
  --model_path /path/to/qwen3.5-0.8b-w4a32-kai.mllm \
  --model_version v2 \
  --tokenizer_path /path/to/Qwen3.5-0.8B/tokenizer.json \
  --config_path examples/qwen3_5/config_0.8B_w4a32_kai.json \
  --prompt "Give a one-sentence introduction." \
  --max_new_tokens 32
```

Omit `--prompt` for the interactive loop. The CLI treats each prompt as an
independent conversation and resets all model state before inference.
