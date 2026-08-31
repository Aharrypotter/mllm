#!/usr/bin/env python3
"""Compare Qwen2.5 FP32, rotated FP32, and exported W8A16 semantics.

This diagnostic reconstructs the fake-quantized PyTorch model from the exact
QNN export state.  In particular, exported int8 linear weights are transposed
back to PyTorch layout and dequantized with their stored per-tensor scales;
the stored FP32 Q/K/V biases and activation scales/clipping flags are reused
verbatim.  This keeps the oracle on the model-generation side of the mllm/QNN
runtime boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CHAT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--rotated-state", required=True, type=Path)
    parser.add_argument("--quantized-state", required=True, type=Path)
    parser.add_argument("--rotation", required=True, type=Path)
    parser.add_argument("--distribution", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--tools-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--layer-count",
        type=int,
        help="Restrict all semantic stages to the first N decoder layers",
    )
    parser.add_argument("--expected-prompt-tokens", type=int, default=512)
    parser.add_argument("--expected-text-sha256")
    parser.add_argument("--expected-token-ids-sha256")
    parser.add_argument(
        "--activation-contract",
        choices=("export-simulation", "legacy-qwen-runtime"),
        default="export-simulation",
        help=(
            "Activation precision contract to reconstruct. The legacy Qwen "
            "runtime uses W8A16 for Q/K/V and down_proj, and W8A8 for "
            "o_proj/gate_proj/up_proj."
        ),
    )
    parser.add_argument(
        "--w8a16-projection-mode",
        choices=(
            "export-post-bias",
            "export-post-bias-int32-bias",
            "separated-post-range",
            "separated-widened-range",
        ),
        default="export-post-bias",
        help=(
            "Q/K/V W8A16 boundary to reconstruct. The int32-bias mode models "
            "QNN accumulator-scale bias quantization before the exported post-bias "
            "activation fake quant. The separated modes model "
            "the mllm QNN path that quantizes the bias-free matmul before adding "
            "the exported Float32 bias and requantizing the post-bias result."
        ),
    )
    parser.add_argument(
        "--capture-hidden-states",
        action="store_true",
        help="Record embedding, per-layer residual, and final-norm summaries",
    )
    parser.add_argument(
        "--capture-prefix-top1",
        action="store_true",
        help="Apply the final norm and LM head to every captured decoder-layer output",
    )
    parser.add_argument(
        "--quantized-only",
        action="store_true",
        help="Skip the original and rotated FP32 probes for focused QNN-boundary diagnostics",
    )
    parser.add_argument(
        "--capture-layer-zero-boundaries",
        action="store_true",
        help="Record layer-0 normalization, projection, attention, and MLP boundaries",
    )
    parser.add_argument(
        "--boundary-dump-dir",
        type=Path,
        help="Write every exported layer Q/K/V and attention-input tensor as raw Float32",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(input_ids: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in input_ids.reshape(-1).tolist():
        digest.update(struct.pack("<q", int(value)))
    return digest.hexdigest()


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    values = value.detach().float().cpu()
    flat = values.reshape(-1)
    return {
        "shape": list(values.shape),
        "finite": int(torch.isfinite(flat).sum().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "argmax": int(flat.argmax().item()),
        "first": float(flat[0].item()),
    }


def dump_boundary_values(
    root: Path, values: dict[str, torch.Tensor]
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    files = []
    for name, value in sorted(values.items()):
        tensor = value.detach().float().cpu().contiguous()
        path = root / f"{name}.f32"
        tensor.numpy().tofile(path)
        files.append(
            {
                "boundary": name,
                "path": path.name,
                "shape": list(tensor.shape),
                "numel": tensor.numel(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    metadata = {"schema": 1, "dtype": "float32-le", "files": files}
    metadata_path = root / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "path": str(root),
        "file_count": len(files),
        "metadata_sha256": sha256_file(metadata_path),
    }


def untie_word_embeddings_for_state_load(model: torch.nn.Module) -> None:
    """Match the exporter before loading its rotated state.

    Qwen2.5 ships with tied token embedding and LM-head weights.  The rotation
    exporter explicitly unties them because the two tensors receive different
    transforms.  Loading that state into a still-tied Transformers model would
    silently copy both keys into the same storage and invalidate the oracle.
    """
    if model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(
            model.lm_head.weight.detach().clone(), requires_grad=False
        )
    model.config.tie_word_embeddings = False
    if model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr():
        raise RuntimeError("failed to untie token embedding and LM head")


def restrict_model_layers(model: torch.nn.Module, layer_count: int | None) -> int:
    total_layers = len(model.model.layers)
    if layer_count is None:
        return total_layers
    if layer_count < 0 or layer_count > total_layers:
        raise ValueError(f"--layer-count must be in [0, {total_layers}]")
    model.model.layers = torch.nn.ModuleList(list(model.model.layers[:layer_count]))
    model.config.num_hidden_layers = layer_count
    return layer_count


def state_for_layer_count(
    state: dict[str, torch.Tensor], layer_count: int, total_layers: int
) -> dict[str, torch.Tensor]:
    if layer_count == total_layers:
        return state
    layer_pattern = re.compile(r"^model\.layers\.(\d+)\.")
    return {
        name: tensor
        for name, tensor in state.items()
        if not (match := layer_pattern.match(name)) or int(match.group(1)) < layer_count
    }


def make_exported_linear(
    module: torch.nn.Linear,
    state: dict[str, torch.Tensor],
    name: str,
    linear_type: type[torch.nn.Module],
    *,
    n_scale_bits: int | None = None,
    n_bits: int | None = None,
    quantize_bias_to_int32_accumulator: bool = False,
) -> torch.nn.Module:
    weight_key = f"{name}.weight"
    scale_key = f"{weight_key}.scale"
    input_scale_key = f"{name}.input_scale"
    output_scale_key = f"{name}.output_scale"
    clip_input_key = f"{name}.clip_input"
    clip_output_key = f"{name}.clip_output"
    required = {
        weight_key,
        scale_key,
        input_scale_key,
        output_scale_key,
        clip_input_key,
        clip_output_key,
    }
    missing = sorted(required.difference(state))
    if missing:
        raise KeyError(f"missing exported tensors for {name}: {missing}")

    kwargs: dict[str, Any] = {
        "in_features": module.in_features,
        "out_features": module.out_features,
        "input_scale": scalar(state[input_scale_key]),
        "output_scale": scalar(state[output_scale_key]),
        "bias": module.bias is not None,
        "clip_top": {
            "input": bool(state[clip_input_key].item()),
            "output": bool(state[clip_output_key].item()),
        },
        "name": name,
    }
    if n_scale_bits is not None:
        kwargs["n_scale_bits"] = n_scale_bits
    if n_bits is not None:
        kwargs["n_bits"] = n_bits
    replacement = linear_type(**kwargs)

    exported_weight = state[weight_key]
    expected_export_shape = (module.in_features, module.out_features)
    if exported_weight.dtype != torch.int8:
        raise TypeError(f"{weight_key} is {exported_weight.dtype}, expected int8")
    if tuple(exported_weight.shape) != expected_export_shape:
        raise ValueError(
            f"{weight_key} shape {tuple(exported_weight.shape)} does not match "
            f"transposed QNN shape {expected_export_shape}"
        )
    device = module.weight.device
    dtype = module.weight.dtype
    replacement.weight = (
        exported_weight.transpose(-2, -1)
        .contiguous()
        .to(device=device, dtype=dtype)
        .mul_(scalar(state[scale_key]))
    )
    if module.bias is not None:
        bias_key = f"{name}.bias"
        if bias_key not in state:
            raise KeyError(f"missing exported bias {bias_key}")
        if state[bias_key].dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError(f"{bias_key} is {state[bias_key].dtype}, expected floating point")
        replacement_bias = state[bias_key].to(device=device, dtype=dtype)
        if quantize_bias_to_int32_accumulator:
            if n_scale_bits != 16 or n_bits != 16:
                raise ValueError(
                    "int32 accumulator bias simulation is only valid for W8A16 projections"
                )
            input_step = max(scalar(state[input_scale_key]), 1e-5) / 32767.0
            bias_step = input_step * scalar(state[scale_key])
            if not bias_step > 0.0:
                raise ValueError(f"invalid accumulator bias step for {name}: {bias_step}")
            replacement_bias = (
                replacement_bias.double()
                .div(bias_step)
                .round()
                .clamp(-(2**31), 2**31 - 1)
                .mul(bias_step)
                .to(dtype=dtype)
            )
        replacement.bias = replacement_bias
    return replacement


def make_separated_bias_w8ax_type(
    base_type: type[torch.nn.Module], *, widen_pre_bias_range: bool
) -> type[torch.nn.Module]:
    """Model mllm's separated QNN Conv2d + DequantizeAdd Q/K/V path."""

    class SeparatedBiasW8AX(base_type):
        @torch.no_grad()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            q_x = self.act_quant_input(x, scale=self.input_scale)
            matmul = torch.functional.F.linear(q_x, self.weight, None)

            pre_bias_range = self.output_scale.to(matmul.device)
            if widen_pre_bias_range and self.bias is not None:
                pre_bias_range = pre_bias_range + self.bias.detach().abs().max()
            pre_bias_scale = pre_bias_range.clamp(min=1e-5).div(32767)
            q_matmul = (
                matmul.div(pre_bias_scale)
                .round()
                .clamp(-32768, 32767)
                .mul(pre_bias_scale)
            )
            post_bias = q_matmul if self.bias is None else q_matmul + self.bias
            return self.act_quant_output(post_bias, scale=self.output_scale)

    mode = "widened" if widen_pre_bias_range else "post_range"
    SeparatedBiasW8AX.__name__ = f"SeparatedBiasW8AX_{mode}"
    return SeparatedBiasW8AX


def install_exported_w8a16(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    w8a8_type: type[torch.nn.Module],
    w8ax_type: type[torch.nn.Module],
    activation_contract: str,
    quantize_qkv_bias_to_int32_accumulator: bool,
) -> dict[str, int]:
    counts = {"w8a8": 0, "w8a16": 0}
    for layer_index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{layer_index}"
        attention = layer.self_attn
        for projection in ("q_proj", "k_proj", "v_proj"):
            name = f"{prefix}.self_attn.{projection}"
            original = getattr(attention, projection)
            setattr(
                attention,
                projection,
                make_exported_linear(
                    original,
                    state,
                    name,
                    w8ax_type,
                    n_scale_bits=16,
                    n_bits=16,
                    quantize_bias_to_int32_accumulator=quantize_qkv_bias_to_int32_accumulator,
                ),
            )
            counts["w8a16"] += 1
        name = f"{prefix}.self_attn.o_proj"
        attention.o_proj = make_exported_linear(
            attention.o_proj, state, name, w8a8_type
        )
        counts["w8a8"] += 1

        mlp = layer.mlp
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"{prefix}.mlp.{projection}"
            original = getattr(mlp, projection)
            use_w8a16 = (
                activation_contract == "legacy-qwen-runtime"
                and projection == "down_proj"
            )
            setattr(
                mlp,
                projection,
                make_exported_linear(
                    original,
                    state,
                    name,
                    w8ax_type if use_w8a16 else w8a8_type,
                    n_scale_bits=16 if use_w8a16 else None,
                    n_bits=16 if use_w8a16 else None,
                ),
            )
            counts["w8a16" if use_w8a16 else "w8a8"] += 1
    return counts


@torch.inference_mode()
def greedy_probe(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    capture_hidden_states: bool,
    capture_layer_zero_boundaries: bool,
    capture_prefix_top1: bool,
    boundary_dump_dir: Path | None,
) -> tuple[dict[str, Any], torch.Tensor]:
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    attention_mask = torch.ones_like(input_ids)
    boundary_values: dict[str, torch.Tensor] = {}
    dump_values: dict[str, torch.Tensor] = {}
    hook_handles: list[Any] = []
    if capture_layer_zero_boundaries:
        if not model.model.layers:
            raise ValueError("layer-0 boundary capture requires at least one decoder layer")
        layer = model.model.layers[0]

        def capture_output(name: str):
            def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
                boundary_values.setdefault(name, output.detach().float().cpu().clone())

            return hook

        def capture_input(name: str):
            def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                boundary_values.setdefault(name, inputs[0].detach().float().cpu().clone())

            return hook

        hook_handles.extend(
            [
                layer.input_layernorm.register_forward_hook(capture_output("input_layernorm")),
                layer.self_attn.q_proj.register_forward_hook(capture_output("q_proj")),
                layer.self_attn.k_proj.register_forward_hook(capture_output("k_proj")),
                layer.self_attn.v_proj.register_forward_hook(capture_output("v_proj")),
                layer.self_attn.o_proj.register_forward_pre_hook(capture_input("attention_output")),
                layer.self_attn.o_proj.register_forward_hook(capture_output("o_proj")),
                layer.post_attention_layernorm.register_forward_hook(capture_output("post_attention_layernorm")),
                layer.mlp.gate_proj.register_forward_hook(capture_output("gate_proj")),
                layer.mlp.up_proj.register_forward_hook(capture_output("up_proj")),
                layer.mlp.down_proj.register_forward_pre_hook(capture_input("down_proj_input")),
                layer.mlp.down_proj.register_forward_hook(capture_output("down_proj")),
            ]
        )
    if boundary_dump_dir is not None:
        for layer_index, layer in enumerate(model.model.layers):
            prefix = f"layer-{layer_index:02d}"

            def capture_dump_output(name: str):
                def hook(
                    _module: torch.nn.Module,
                    _inputs: tuple[Any, ...],
                    output: torch.Tensor,
                ) -> None:
                    dump_values.setdefault(
                        name, output.detach().float().cpu().contiguous()
                    )

                return hook

            def capture_dump_input(name: str):
                def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                    dump_values.setdefault(
                        name, inputs[0].detach().float().cpu().contiguous()
                    )

                return hook

            hook_handles.extend(
                [
                    layer.self_attn.q_proj.register_forward_hook(
                        capture_dump_output(f"{prefix}-q_proj")
                    ),
                    layer.self_attn.k_proj.register_forward_hook(
                        capture_dump_output(f"{prefix}-k_proj")
                    ),
                    layer.self_attn.v_proj.register_forward_hook(
                        capture_dump_output(f"{prefix}-v_proj")
                    ),
                    layer.self_attn.o_proj.register_forward_pre_hook(
                        capture_dump_input(f"{prefix}-attention_output")
                    ),
                    layer.post_attention_layernorm.register_forward_pre_hook(
                        capture_dump_input(f"{prefix}-post_attention_residual")
                    ),
                    layer.mlp.down_proj.register_forward_hook(
                        capture_dump_output(f"{prefix}-down_proj")
                    ),
                ]
            )
    start = time.perf_counter()
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            output_hidden_states=capture_hidden_states,
        )
    finally:
        for handle in hook_handles:
            handle.remove()
    first_logits = outputs.logits[:, -1, :].float().detach().cpu()
    first_hidden_states = outputs.hidden_states if capture_hidden_states else None
    logits = first_logits
    past_key_values = outputs.past_key_values
    generated: list[int] = []
    for step in range(max_new_tokens):
        next_token = logits.argmax(dim=-1).to(input_ids.device)
        generated.append(int(next_token.item()))
        if step + 1 == max_new_tokens:
            break
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=-1,
        )
        outputs = model(
            input_ids=next_token.view(1, 1),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :].float().detach().cpu()
        past_key_values = outputs.past_key_values
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    top_values, top_indices = torch.topk(first_logits[0], k=5)
    result = {
        "first_logits_finite": bool(torch.isfinite(first_logits).all().item()),
        "first_top1_id": int(first_logits.argmax(dim=-1).item()),
        "first_top1_text": tokenizer.decode([int(first_logits.argmax(dim=-1).item())]),
        "first_top5": [
            {
                "id": int(token_id),
                "logit": float(logit),
                "text": tokenizer.decode([int(token_id)]),
            }
            for token_id, logit in zip(top_indices.tolist(), top_values.tolist())
        ],
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
        "elapsed_ms": elapsed_ms,
    }
    if capture_hidden_states:
        hidden_summaries = []
        if first_hidden_states is None:
            raise RuntimeError("hidden-state capture was requested but the model returned none")
        for index, hidden_state in enumerate(first_hidden_states):
            values = hidden_state.detach().float().cpu()
            flat = values.reshape(-1)
            last_token = values[:, -1, :].reshape(-1)
            hidden_summaries.append(
                {
                    "index": index,
                    "shape": list(values.shape),
                    "finite": int(torch.isfinite(flat).sum().item()),
                    "min": float(flat.min().item()),
                    "max": float(flat.max().item()),
                    "argmax": int(flat.argmax().item()),
                    "first": float(flat[0].item()),
                    "last_token_min": float(last_token.min().item()),
                    "last_token_max": float(last_token.max().item()),
                    "last_token_argmax": int(last_token.argmax().item()),
                    "last_token_first": float(last_token[0].item()),
                }
            )
        result["hidden_state_summaries"] = hidden_summaries
        if capture_prefix_top1:
            prefix_top1 = []
            for depth, hidden_state in enumerate(first_hidden_states[1:], start=1):
                depth_logits = model.lm_head(
                    model.model.norm(hidden_state[:, -1:, :])
                ).float()
                depth_top1 = int(depth_logits[:, -1, :].argmax(dim=-1).item())
                prefix_top1.append(
                    {
                        "layer_count": depth,
                        "top1_id": depth_top1,
                        "top1_text": tokenizer.decode([depth_top1]),
                    }
                )
            result["prefix_top1"] = prefix_top1
    if capture_layer_zero_boundaries:
        result["layer_zero_boundary_summaries"] = {
            name: tensor_summary(value) for name, value in boundary_values.items()
        }
    if boundary_dump_dir is not None:
        if first_hidden_states is None:
            raise RuntimeError("boundary dump requires returned hidden states")
        dump_values["embedding"] = (
            first_hidden_states[0].detach().float().cpu().contiguous()
        )
        for layer_index, hidden_state in enumerate(first_hidden_states[1:-1]):
            dump_values[f"layer-{layer_index:02d}-residual"] = (
                hidden_state.detach().float().cpu().contiguous()
            )
        dump_values["final_norm"] = (
            first_hidden_states[-1].detach().float().cpu().contiguous()
        )
        result["boundary_dump"] = dump_boundary_values(
            boundary_dump_dir, dump_values
        )
    return result, first_logits


def logit_difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = (left - right).abs()
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this oracle requires CUDA")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.capture_prefix_top1 and not args.capture_hidden_states:
        raise ValueError("--capture-prefix-top1 requires --capture-hidden-states")
    if args.boundary_dump_dir is not None and not args.capture_hidden_states:
        raise ValueError("--boundary-dump-dir requires --capture-hidden-states")

    converter_root = args.tools_root / "qnn_convertor"
    if not converter_root.is_dir():
        raise FileNotFoundError(f"missing converter root: {converter_root}")
    sys.path.insert(0, str(converter_root))
    from utils.quantization_simulation import W8A8LinearStatic, W8AX_LinearStatic

    if args.w8a16_projection_mode in (
        "export-post-bias",
        "export-post-bias-int32-bias",
    ):
        w8ax_type = W8AX_LinearStatic
    else:
        w8ax_type = make_separated_bias_w8ax_type(
            W8AX_LinearStatic,
            widen_pre_bias_range=args.w8a16_projection_mode
            == "separated-widened-range",
        )

    prompt = args.prompt_file.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if args.expected_text_sha256 and prompt_sha256 != args.expected_text_sha256:
        raise ValueError(
            f"prompt hash mismatch: {prompt_sha256} != {args.expected_text_sha256}"
        )
    rendered = CHAT_TEMPLATE.format(prompt=prompt)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_tensors="pt",
    )
    if int(encoded.input_ids.shape[1]) != args.expected_prompt_tokens:
        raise ValueError(
            f"prompt token count {encoded.input_ids.shape[1]} != "
            f"{args.expected_prompt_tokens}"
        )
    ids_sha256 = token_ids_sha256(encoded.input_ids)
    if args.expected_token_ids_sha256 and ids_sha256 != args.expected_token_ids_sha256:
        raise ValueError(
            f"token id hash mismatch: {ids_sha256} != {args.expected_token_ids_sha256}"
        )

    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(device).eval()
    total_layers = len(model.model.layers)
    layer_count = restrict_model_layers(model, args.layer_count)
    input_ids = encoded.input_ids.to(device)

    original = None
    original_logits = None
    if not args.quantized_only:
        original, original_logits = greedy_probe(
            model,
            tokenizer,
            input_ids,
            args.max_new_tokens,
            args.capture_hidden_states,
            args.capture_layer_zero_boundaries,
            args.capture_prefix_top1,
            None,
        )
        print(json.dumps({"stage": "original_fp32", **original}, sort_keys=True), flush=True)

    untie_word_embeddings_for_state_load(model)
    rotated_state = torch.load(args.rotated_state, map_location="cpu", weights_only=True)
    rotated_state = state_for_layer_count(rotated_state, layer_count, total_layers)
    incompatible = model.load_state_dict(rotated_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"rotated state mismatch: {incompatible}")
    del rotated_state
    rotated = None
    rotated_logits = None
    if not args.quantized_only:
        rotated, rotated_logits = greedy_probe(
            model,
            tokenizer,
            input_ids,
            args.max_new_tokens,
            args.capture_hidden_states,
            args.capture_layer_zero_boundaries,
            args.capture_prefix_top1,
            None,
        )
        print(json.dumps({"stage": "rotated_fp32", **rotated}, sort_keys=True), flush=True)

    quantized_state = torch.load(args.quantized_state, map_location="cpu", weights_only=True)
    installation = install_exported_w8a16(
        model,
        quantized_state,
        W8A8LinearStatic,
        w8ax_type,
        args.activation_contract,
        args.w8a16_projection_mode == "export-post-bias-int32-bias",
    )
    del quantized_state
    exported_w8a16, exported_w8a16_logits = greedy_probe(
        model,
        tokenizer,
        input_ids,
        args.max_new_tokens,
        args.capture_hidden_states,
        args.capture_layer_zero_boundaries,
        args.capture_prefix_top1,
        args.boundary_dump_dir,
    )
    print(
        json.dumps({"stage": "exported_w8a16", **exported_w8a16}, sort_keys=True),
        flush=True,
    )

    artifacts = {
        "model": str(args.model),
        "rotated_state": {
            "path": str(args.rotated_state),
            "sha256": sha256_file(args.rotated_state),
        },
        "quantized_state": {
            "path": str(args.quantized_state),
            "sha256": sha256_file(args.quantized_state),
        },
        "rotation": {
            "path": str(args.rotation),
            "sha256": sha256_file(args.rotation),
        },
        "distribution": {
            "path": str(args.distribution),
            "sha256": sha256_file(args.distribution),
        },
        "prompt": {
            "path": str(args.prompt_file),
            "text_sha256": prompt_sha256,
            "rendered_token_count": int(encoded.input_ids.shape[1]),
            "token_ids_sha256_le_i64": ids_sha256,
        },
    }
    stages = {"exported_w8a16": exported_w8a16}
    comparisons = {}
    if not args.quantized_only:
        assert original is not None and original_logits is not None
        assert rotated is not None and rotated_logits is not None
        stages = {
            "original_fp32": original,
            "rotated_fp32": rotated,
            **stages,
        }
        comparisons = {
            "original_vs_rotated": {
                "top1_equal": original["first_top1_id"] == rotated["first_top1_id"],
                "generated_equal": original["generated_token_ids"]
                == rotated["generated_token_ids"],
                "first_logits": logit_difference(original_logits, rotated_logits),
            },
            "rotated_vs_exported_w8a16": {
                "top1_equal": rotated["first_top1_id"]
                == exported_w8a16["first_top1_id"],
                "generated_equal": rotated["generated_token_ids"]
                == exported_w8a16["generated_token_ids"],
                "first_logits": logit_difference(rotated_logits, exported_w8a16_logits),
            },
        }

    result = {
        "schema": 1,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name(0),
        "artifacts": artifacts,
        "installation": installation,
        "activation_contract": args.activation_contract,
        "w8a16_projection_mode": args.w8a16_projection_mode,
        "quantized_only": args.quantized_only,
        "layer_count": layer_count,
        "source_total_layers": total_layers,
        "rotated_state_loaded_with_untied_embeddings": True,
        "stages": stages,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)

    all_finite = all(
        stage["first_logits_finite"] for stage in result["stages"].values()
    )
    return 0 if all_finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
