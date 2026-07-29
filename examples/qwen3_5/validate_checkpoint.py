# Copyright (c) MLLM Team.
# Licensed under the MIT License.

"""Validate a Qwen3.5 text checkpoint and its mobile quantization recipe."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from safetensors import safe_open


def expected_text_shapes(text_config: dict) -> dict[str, list[int]]:
    hidden_size = text_config["hidden_size"]
    intermediate_size = text_config["intermediate_size"]
    head_dim = text_config["head_dim"]
    num_attention_heads = text_config["num_attention_heads"]
    num_key_value_heads = text_config["num_key_value_heads"]
    num_key_heads = text_config["linear_num_key_heads"]
    num_value_heads = text_config["linear_num_value_heads"]
    key_head_dim = text_config["linear_key_head_dim"]
    value_head_dim = text_config["linear_value_head_dim"]
    conv_kernel = text_config["linear_conv_kernel_dim"]

    key_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    conv_dim = key_dim * 2 + value_dim
    q_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    q_projection_size = q_size * (2 if text_config["attn_output_gate"] else 1)

    expected = {
        "model.language_model.embed_tokens.weight": [
            text_config["vocab_size"],
            hidden_size,
        ],
        "model.language_model.norm.weight": [hidden_size],
    }
    for layer_index, layer_type in enumerate(text_config["layer_types"]):
        prefix = f"model.language_model.layers.{layer_index}"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight": [hidden_size],
                f"{prefix}.post_attention_layernorm.weight": [hidden_size],
                f"{prefix}.mlp.gate_proj.weight": [
                    intermediate_size,
                    hidden_size,
                ],
                f"{prefix}.mlp.up_proj.weight": [
                    intermediate_size,
                    hidden_size,
                ],
                f"{prefix}.mlp.down_proj.weight": [
                    hidden_size,
                    intermediate_size,
                ],
            }
        )
        if layer_type == "linear_attention":
            expected.update(
                {
                    f"{prefix}.linear_attn.in_proj_qkv.weight": [
                        conv_dim,
                        hidden_size,
                    ],
                    f"{prefix}.linear_attn.in_proj_z.weight": [
                        value_dim,
                        hidden_size,
                    ],
                    f"{prefix}.linear_attn.in_proj_a.weight": [
                        num_value_heads,
                        hidden_size,
                    ],
                    f"{prefix}.linear_attn.in_proj_b.weight": [
                        num_value_heads,
                        hidden_size,
                    ],
                    f"{prefix}.linear_attn.conv1d.weight": [
                        conv_dim,
                        1,
                        conv_kernel,
                    ],
                    f"{prefix}.linear_attn.A_log": [num_value_heads],
                    f"{prefix}.linear_attn.dt_bias": [num_value_heads],
                    f"{prefix}.linear_attn.norm.weight": [value_head_dim],
                    f"{prefix}.linear_attn.out_proj.weight": [
                        hidden_size,
                        value_dim,
                    ],
                }
            )
        elif layer_type == "full_attention":
            expected.update(
                {
                    f"{prefix}.self_attn.q_proj.weight": [
                        q_projection_size,
                        hidden_size,
                    ],
                    f"{prefix}.self_attn.k_proj.weight": [
                        kv_size,
                        hidden_size,
                    ],
                    f"{prefix}.self_attn.v_proj.weight": [
                        kv_size,
                        hidden_size,
                    ],
                    f"{prefix}.self_attn.o_proj.weight": [
                        hidden_size,
                        q_size,
                    ],
                    f"{prefix}.self_attn.q_norm.weight": [head_dim],
                    f"{prefix}.self_attn.k_norm.weight": [head_dim],
                }
            )
        else:
            raise ValueError(f"Unsupported layer type {layer_type!r}")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--quant-config",
        type=Path,
        default=Path(__file__).with_name("quant_cfg_0.8B_w4a32_kai.json"),
    )
    args = parser.parse_args()

    with (args.checkpoint / "config.json").open() as config_file:
        config = json.load(config_file)
    with (args.checkpoint / "model.safetensors.index.json").open() as index_file:
        index = json.load(index_file)
    with args.quant_config.open() as quant_config_file:
        quant_config = json.load(quant_config_file)

    text_config = config["text_config"]
    if config.get("model_type") != "qwen3_5":
        raise AssertionError(
            f"Expected model_type='qwen3_5', got {config.get('model_type')!r}"
        )
    if not text_config.get(
        "tie_word_embeddings",
        config.get("tie_word_embeddings", True),
    ):
        raise AssertionError(
            "Qwen3.5 ARM CPU conversion currently requires tied word embeddings"
        )
    if len(text_config["layer_types"]) != text_config["num_hidden_layers"]:
        raise AssertionError("layer_types length must equal num_hidden_layers")
    expected_shapes = expected_text_shapes(text_config)
    weight_map = index["weight_map"]
    actual_text_keys = {
        name for name in weight_map if name.startswith("model.language_model.")
    }
    expected_text_keys = set(expected_shapes)
    missing = sorted(expected_text_keys - actual_text_keys)
    extra = sorted(actual_text_keys - expected_text_keys)
    if missing or extra:
        raise AssertionError(
            f"Text weight-name mismatch: missing={missing}, extra={extra}"
        )

    shard_names = sorted({weight_map[name] for name in actual_text_keys})
    dtype_counts: Counter[str] = Counter()
    with ExitStack() as stack:
        shards = {
            shard_name: stack.enter_context(
                safe_open(
                    args.checkpoint / shard_name,
                    framework="pt",
                    device="cpu",
                )
            )
            for shard_name in shard_names
        }

        for name, expected_shape in expected_shapes.items():
            tensor_slice = shards[weight_map[name]].get_slice(name)
            actual_shape = list(tensor_slice.get_shape())
            if actual_shape != expected_shape:
                raise AssertionError(
                    f"{name}: expected shape {expected_shape}, got {actual_shape}"
                )
            dtype_counts[str(tensor_slice.get_dtype())] += 1

        quantized_names: set[str] = set()
        pattern_counts: dict[str, int] = {}
        for pattern, entry in quant_config.items():
            regex = re.compile(pattern)
            matched_names = sorted(
                name for name in actual_text_keys if regex.fullmatch(name)
            )
            if not matched_names:
                raise AssertionError(
                    f"Quantization pattern matched no weights: {pattern}"
                )
            overlap = quantized_names.intersection(matched_names)
            if overlap:
                raise AssertionError(
                    f"Quantization pattern overlap for {pattern}: {sorted(overlap)}"
                )
            expected_shape = entry["hints"]["shape"]
            for name in matched_names:
                actual_shape = list(
                    shards[weight_map[name]].get_slice(name).get_shape()
                )
                if actual_shape != expected_shape:
                    raise AssertionError(
                        f"{pattern} matched {name}: config shape "
                        f"{expected_shape}, checkpoint shape {actual_shape}"
                    )
            quantized_names.update(matched_names)
            pattern_counts[pattern] = len(matched_names)

    summary = {
        "checkpoint": str(args.checkpoint),
        "text_parameters": len(actual_text_keys),
        "full_attention_layers": text_config["layer_types"].count("full_attention"),
        "linear_attention_layers": text_config["layer_types"].count("linear_attention"),
        "quantized_parameters": len(quantized_names),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "pattern_counts": pattern_counts,
    }

    kai_linear_pattern = re.compile(
        r"(?:"
        r"model\.language_model\.embed_tokens\.weight|"
        r".*\.mlp\.(?:gate_proj|up_proj|down_proj)\.weight|"
        r".*\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight|"
        r".*\.linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)\.weight"
        r")"
    )
    expected_quantized_names = {
        name for name in actual_text_keys if kai_linear_pattern.fullmatch(name)
    }
    missing_quantized_linears = sorted(expected_quantized_names - quantized_names)
    unexpected_quantized_parameters = sorted(quantized_names - expected_quantized_names)
    if missing_quantized_linears or unexpected_quantized_parameters:
        raise AssertionError(
            "KAI quantization coverage mismatch: "
            f"missing_linears={missing_quantized_linears}, "
            f"unexpected_parameters={unexpected_quantized_parameters}"
        )

    summary["expected_kai_linear_parameters"] = len(expected_quantized_names)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("QWEN35_CHECKPOINT_AUDIT_OK")


if __name__ == "__main__":
    main()
