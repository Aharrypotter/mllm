# Copyright (c) MLLM Team.
# Licensed under the MIT License.

"""Validate a Qwen3.5 checkpoint selection and its mobile quantization recipe."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from safetensors import safe_open


MODEL_VARIANTS = {
    "0.8B": {
        "model_name": "Qwen3.5-0.8B",
        "quant_config": "quant_cfg_0.8B_w4a32_kai.json",
        "runtime_config": "config_0.8B_w4a32_kai.json",
        "contract": {
            "attention_bias": False,
            "attn_output_gate": True,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "num_attention_heads": 8,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "tie_word_embeddings": True,
            "vocab_size": 248320,
        },
    },
    "4B": {
        "model_name": "Qwen3.5-4B",
        "quant_config": "quant_cfg_4B_w4a32_kai.json",
        "runtime_config": "config_4B_w4a32_kai.json",
        "contract": {
            "attention_bias": False,
            "attn_output_gate": True,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "num_attention_heads": 16,
            "num_hidden_layers": 32,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "tie_word_embeddings": True,
            "vocab_size": 248320,
        },
    },
}

KAI_HINT_CONTRACT = {
    "quant_method": "kai",
    "kai_matmul_triplet": "f32_qai8dxp_qsi4c32p",
    "kai_matmul_layout": "mxk_nxk",
    "kai_matmul_tile_cfg": "qai8dxp1x8_qsi4c32p8x8_1x8x32",
}
KAI_LINEAR_IMPL_TYPE = (
    "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_" "qai8dxp1x8_qsi4c32p8x8_1x8x32"
)
TIED_EMBEDDING_WEIGHT = "model.language_model.embed_tokens.weight"
OFFICIAL_VISION_CONTRACTS = {
    "0.8B": {
        "deepstack_visual_indexes": [],
        "depth": 12,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 768,
        "in_channels": 3,
        "intermediate_size": 3072,
        "num_heads": 12,
        "num_position_embeddings": 2304,
        "out_hidden_size": 1024,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "4B": {
        "deepstack_visual_indexes": [],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "intermediate_size": 4096,
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}
MULTIMODAL_TOKEN_CONTRACT = {
    "image_token_id": 248056,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
}


def _vision_contract_mismatches(
    vision_config: dict, model_size: str
) -> list[str]:
    contract = OFFICIAL_VISION_CONTRACTS[model_size]
    return [
        f"{field}={vision_config.get(field)!r} (expected {expected!r})"
        for field, expected in contract.items()
        if vision_config.get(field) != expected
    ]


def _expected_layer_types(num_hidden_layers: int, interval: int) -> list[str]:
    return [
        "full_attention" if (layer_index + 1) % interval == 0 else "linear_attention"
        for layer_index in range(num_hidden_layers)
    ]


def _variant_mismatches(text_config: dict, model_size: str) -> list[str]:
    contract = MODEL_VARIANTS[model_size]["contract"]
    mismatches = [
        f"{field}={text_config.get(field)!r} (expected {expected!r})"
        for field, expected in contract.items()
        if text_config.get(field) != expected
    ]
    expected_layer_types = _expected_layer_types(
        contract["num_hidden_layers"],
        contract["full_attention_interval"],
    )
    if text_config.get("layer_types") != expected_layer_types:
        mismatches.append(
            "layer_types does not match the official "
            f"{contract['full_attention_interval']}-layer hybrid schedule"
        )
    return mismatches


def resolve_model_size(text_config: dict, requested_size: str | None = None) -> str:
    """Resolve and validate a supported official Qwen3.5 text configuration."""

    if requested_size is not None:
        mismatches = _variant_mismatches(text_config, requested_size)
        if mismatches:
            raise AssertionError(
                f"Checkpoint does not match Qwen3.5-{requested_size}: "
                + "; ".join(mismatches)
            )
        return requested_size

    matches = [
        model_size
        for model_size in MODEL_VARIANTS
        if not _variant_mismatches(text_config, model_size)
    ]
    if len(matches) != 1:
        identity = {
            field: text_config.get(field)
            for field in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "linear_num_key_heads",
                "linear_num_value_heads",
            )
        }
        raise AssertionError(
            "Checkpoint does not match a supported official Qwen3.5 text "
            f"configuration (0.8B or 4B): {identity}"
        )
    return matches[0]


def default_quant_config_path(model_size: str) -> Path:
    return Path(__file__).with_name(MODEL_VARIANTS[model_size]["quant_config"])


def default_runtime_config_path(model_size: str) -> Path:
    return Path(__file__).with_name(MODEL_VARIANTS[model_size]["runtime_config"])


def model_name_for_size(model_size: str) -> str:
    return MODEL_VARIANTS[model_size]["model_name"]


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


def _expected_kai_linear_names(source_names: set[str]) -> set[str]:
    kai_linear_pattern = re.compile(
        r"(?:"
        r"model\.language_model\.embed_tokens\.weight|"
        r".*\.mlp\.(?:gate_proj|up_proj|down_proj)\.weight|"
        r".*\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight|"
        r".*\.linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)\.weight|"
        r"model\.visual\.(?:"
        r"blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12])|"
        r"merger\.linear_fc[12])\.(?:bias|weight)"
        r")"
    )
    return {name for name in source_names if kai_linear_pattern.fullmatch(name)}


def resolve_runtime_linear_impl_type(runtime_config: dict) -> str | None:
    """Mirror Qwen3_5Config's nested-then-top-level implementation lookup."""

    text_config = runtime_config.get("text_config", runtime_config)
    return text_config.get(
        "linear_impl_type",
        runtime_config.get("linear_impl_type"),
    )


def validate_kai_recipe_contract(
    text_config: dict,
    quant_config: dict,
    runtime_config: dict,
    vision_config: dict | None = None,
) -> tuple[set[str], dict[str, int]]:
    """Validate that a quantization recipe matches the configured KAI runtime."""

    checkpoint_size = resolve_model_size(text_config)
    if runtime_config.get("model_type") != "qwen3_5":
        raise AssertionError("KAI runtime config must declare model_type='qwen3_5'")
    runtime_text_config = runtime_config.get("text_config")
    if not isinstance(runtime_text_config, dict):
        raise AssertionError(
            "KAI runtime config must declare an object text_config, got "
            f"{type(runtime_text_config).__name__}"
        )
    runtime_size = resolve_model_size(runtime_text_config)
    if runtime_size != checkpoint_size:
        raise AssertionError(
            f"KAI runtime config is for Qwen3.5-{runtime_size}, "
            f"checkpoint is Qwen3.5-{checkpoint_size}"
        )
    linear_impl_type = resolve_runtime_linear_impl_type(runtime_config)
    if linear_impl_type != KAI_LINEAR_IMPL_TYPE:
        raise AssertionError(
            "KAI runtime linear_impl_type mismatch: "
            f"{linear_impl_type!r} != {KAI_LINEAR_IMPL_TYPE!r}"
        )

    include_vision = any("visual" in pattern for pattern in quant_config)
    runtime_vision_config = runtime_config.get("vision_config")
    if include_vision != isinstance(runtime_vision_config, dict):
        raise AssertionError(
            "KAI recipe and runtime config must either both include vision or both be text-only"
        )
    if include_vision:
        if not isinstance(vision_config, dict):
            raise AssertionError(
                "Multimodal KAI validation requires checkpoint vision_config"
            )
        checkpoint_vision_mismatches = _vision_contract_mismatches(
            vision_config, checkpoint_size
        )
        if checkpoint_vision_mismatches:
            raise AssertionError(
                f"Checkpoint vision_config does not match official Qwen3.5-{checkpoint_size}: "
                + "; ".join(checkpoint_vision_mismatches)
            )
        runtime_vision_mismatches = _vision_contract_mismatches(
            runtime_vision_config, checkpoint_size
        )
        if runtime_vision_mismatches:
            raise AssertionError(
                f"Runtime vision_config does not match official Qwen3.5-{checkpoint_size}: "
                + "; ".join(runtime_vision_mismatches)
            )

    source_shapes = expected_text_shapes(text_config)
    if include_vision:
        source_shapes.update(expected_vision_shapes(vision_config))
    source_names = set(source_shapes)
    quantized_names: set[str] = set()
    pattern_counts: dict[str, int] = {}

    for pattern, entry in quant_config.items():
        hints = entry.get("hints")
        if not isinstance(hints, dict):
            raise AssertionError(
                f"KAI quantization entry {pattern!r} must contain hints"
            )
        hint_mismatches = [
            f"{field}={hints.get(field)!r} (expected {expected!r})"
            for field, expected in KAI_HINT_CONTRACT.items()
            if hints.get(field) != expected
        ]
        if hint_mismatches:
            raise AssertionError(
                f"KAI recipe contract mismatch for {pattern}: "
                + "; ".join(hint_mismatches)
            )

        regex = re.compile(pattern)
        matched_names = sorted(name for name in source_names if regex.fullmatch(name))
        if not matched_names:
            raise AssertionError(f"Quantization pattern matched no weights: {pattern}")
        overlap = quantized_names.intersection(matched_names)
        if overlap:
            raise AssertionError(
                f"Quantization pattern overlap for {pattern}: {sorted(overlap)}"
            )
        configured_shape = hints.get("shape")
        for name in matched_names:
            expected_shape = source_shapes[name]
            if name.endswith(".bias"):
                shape_matches = (
                    isinstance(configured_shape, list)
                    and len(configured_shape) == 2
                    and [configured_shape[0]] == expected_shape
                )
            else:
                shape_matches = configured_shape == expected_shape
            if not shape_matches:
                raise AssertionError(
                    f"{pattern} matched {name}: config shape "
                    f"{configured_shape}, expected source shape {expected_shape}"
                )

        if TIED_EMBEDDING_WEIGHT in matched_names:
            if (
                matched_names != [TIED_EMBEDDING_WEIGHT]
                or hints.get("replace") is not False
                or hints.get("rename") != "lm_head_out.weight"
            ):
                raise AssertionError(
                    "Tied embedding KAI entry must retain embed_tokens.weight "
                    "and rename its packed copy to lm_head_out.weight"
                )
        elif hints.get("replace") is not True or hints.get("rename") is not None:
            raise AssertionError(
                f"KAI Linear entry {pattern} must set replace=true without rename"
            )

        quantized_names.update(matched_names)
        pattern_counts[pattern] = len(matched_names)

    expected_quantized_names = _expected_kai_linear_names(source_names)
    missing = sorted(expected_quantized_names - quantized_names)
    unexpected = sorted(quantized_names - expected_quantized_names)
    if missing or unexpected:
        raise AssertionError(
            "KAI quantization coverage mismatch: "
            f"missing_linears={missing}, unexpected_parameters={unexpected}"
        )
    return quantized_names, pattern_counts


def expected_vision_shapes(vision_config: dict) -> dict[str, list[int]]:
    hidden_size = vision_config["hidden_size"]
    intermediate_size = vision_config["intermediate_size"]
    out_hidden_size = vision_config["out_hidden_size"]
    patch_size = vision_config["patch_size"]
    temporal_patch_size = vision_config["temporal_patch_size"]
    merge_size = vision_config["spatial_merge_size"]

    expected = {
        "model.visual.patch_embed.proj.weight": [
            hidden_size,
            vision_config["in_channels"],
            temporal_patch_size,
            patch_size,
            patch_size,
        ],
        "model.visual.patch_embed.proj.bias": [hidden_size],
        "model.visual.pos_embed.weight": [
            vision_config["num_position_embeddings"],
            hidden_size,
        ],
        "model.visual.merger.norm.weight": [hidden_size],
        "model.visual.merger.norm.bias": [hidden_size],
        "model.visual.merger.linear_fc1.weight": [
            hidden_size * merge_size * merge_size,
            hidden_size * merge_size * merge_size,
        ],
        "model.visual.merger.linear_fc1.bias": [hidden_size * merge_size * merge_size],
        "model.visual.merger.linear_fc2.weight": [
            out_hidden_size,
            hidden_size * merge_size * merge_size,
        ],
        "model.visual.merger.linear_fc2.bias": [out_hidden_size],
    }
    for layer_index in range(vision_config["depth"]):
        prefix = f"model.visual.blocks.{layer_index}"
        expected.update(
            {
                f"{prefix}.norm1.weight": [hidden_size],
                f"{prefix}.norm1.bias": [hidden_size],
                f"{prefix}.norm2.weight": [hidden_size],
                f"{prefix}.norm2.bias": [hidden_size],
                f"{prefix}.attn.qkv.weight": [hidden_size * 3, hidden_size],
                f"{prefix}.attn.qkv.bias": [hidden_size * 3],
                f"{prefix}.attn.proj.weight": [hidden_size, hidden_size],
                f"{prefix}.attn.proj.bias": [hidden_size],
                f"{prefix}.mlp.linear_fc1.weight": [
                    intermediate_size,
                    hidden_size,
                ],
                f"{prefix}.mlp.linear_fc1.bias": [intermediate_size],
                f"{prefix}.mlp.linear_fc2.weight": [
                    hidden_size,
                    intermediate_size,
                ],
                f"{prefix}.mlp.linear_fc2.bias": [hidden_size],
            }
        )
    return expected


def validate_multimodal_config_contract(
    checkpoint_config: dict, runtime_config: dict
) -> None:
    """Validate a supported official multimodal checkpoint/runtime boundary."""

    checkpoint_text_config = checkpoint_config.get("text_config")
    runtime_text_config = runtime_config.get("text_config")
    if not isinstance(checkpoint_text_config, dict) or not isinstance(
        runtime_text_config, dict
    ):
        raise AssertionError(
            "Multimodal checkpoint and runtime must declare object text_config values"
        )
    checkpoint_size = resolve_model_size(checkpoint_text_config)
    runtime_size = resolve_model_size(runtime_text_config)
    if checkpoint_size != runtime_size:
        raise AssertionError(
            f"Multimodal runtime is Qwen3.5-{runtime_size}, checkpoint is Qwen3.5-{checkpoint_size}"
        )

    for field, expected in MULTIMODAL_TOKEN_CONTRACT.items():
        if checkpoint_config.get(field) != expected:
            raise AssertionError(
                f"Checkpoint {field}={checkpoint_config.get(field)!r}, expected {expected}"
            )
        if runtime_config.get(field) != expected:
            raise AssertionError(
                f"Runtime {field}={runtime_config.get(field)!r}, expected {expected}"
            )
    checkpoint_vision_config = checkpoint_config.get("vision_config")
    if not isinstance(checkpoint_vision_config, dict):
        raise AssertionError("Checkpoint must declare vision_config")
    checkpoint_vision_mismatches = _vision_contract_mismatches(
        checkpoint_vision_config, checkpoint_size
    )
    if checkpoint_vision_mismatches:
        raise AssertionError(
            f"Checkpoint vision_config does not match official Qwen3.5-{checkpoint_size}: "
            + "; ".join(checkpoint_vision_mismatches)
        )
    runtime_vision_config = runtime_config.get("vision_config")
    if not isinstance(runtime_vision_config, dict):
        raise AssertionError("Runtime must declare vision_config")
    runtime_vision_mismatches = _vision_contract_mismatches(
        runtime_vision_config, checkpoint_size
    )
    if runtime_vision_mismatches:
        raise AssertionError(
            f"Runtime vision_config does not match official Qwen3.5-{checkpoint_size}: "
            + "; ".join(runtime_vision_mismatches)
        )
    if runtime_config.get("image_min_pixels") != 256 * 256:
        raise AssertionError("Runtime image_min_pixels must be 256*256")
    if runtime_config.get("image_max_pixels") != 512 * 512:
        raise AssertionError("Runtime image_max_pixels must be 512*512")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--model-size",
        choices=tuple(MODEL_VARIANTS),
        help="Expected supported variant (default: detect 0.8B or 4B from config.json)",
    )
    parser.add_argument(
        "--quant-config",
        type=Path,
        help="Quantization recipe (default: recipe for the detected model size)",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        help="Runtime config (default: config for the detected model size)",
    )
    args = parser.parse_args()

    with (args.checkpoint / "config.json").open() as config_file:
        config = json.load(config_file)
    with (args.checkpoint / "model.safetensors.index.json").open() as index_file:
        index = json.load(index_file)

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
    model_size = resolve_model_size(text_config, args.model_size)
    quant_config_path = args.quant_config or default_quant_config_path(model_size)
    runtime_config_path = args.runtime_config or default_runtime_config_path(model_size)
    with quant_config_path.open() as quant_config_file:
        quant_config = json.load(quant_config_file)
    with runtime_config_path.open() as runtime_config_file:
        runtime_config = json.load(runtime_config_file)

    include_vision = any("visual" in pattern for pattern in quant_config)
    if include_vision:
        validate_multimodal_config_contract(config, runtime_config)
    expected_shapes = expected_text_shapes(text_config)
    if include_vision:
        expected_shapes.update(expected_vision_shapes(config["vision_config"]))
    quantized_names, pattern_counts = validate_kai_recipe_contract(
        text_config,
        quant_config,
        runtime_config,
        config.get("vision_config"),
    )
    weight_map = index["weight_map"]
    actual_text_keys = {
        name for name in weight_map if name.startswith("model.language_model.")
    }
    actual_selected_keys = set(actual_text_keys)
    if include_vision:
        actual_selected_keys.update(
            name for name in weight_map if name.startswith("model.visual.")
        )
    expected_selected_keys = set(expected_shapes)
    missing = sorted(expected_selected_keys - actual_selected_keys)
    extra = sorted(actual_selected_keys - expected_selected_keys)
    if missing or extra:
        raise AssertionError(
            f"Selected weight-name mismatch: missing={missing}, extra={extra}"
        )

    shard_names = sorted({weight_map[name] for name in actual_selected_keys})
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

        for pattern, entry in quant_config.items():
            regex = re.compile(pattern)
            matched_names = sorted(
                name for name in actual_selected_keys if regex.fullmatch(name)
            )
            expected_shape = entry["hints"]["shape"]
            for name in matched_names:
                actual_shape = list(
                    shards[weight_map[name]].get_slice(name).get_shape()
                )
                shape_for_entry = (
                    expected_shape if name.endswith(".weight") else [expected_shape[0]]
                )
                if actual_shape != shape_for_entry:
                    raise AssertionError(
                        f"{pattern} matched {name}: config shape "
                        f"{shape_for_entry}, checkpoint shape {actual_shape}"
                    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "model_size": model_size,
        "quant_config": str(quant_config_path),
        "runtime_config": str(runtime_config_path),
        "linear_impl_type": resolve_runtime_linear_impl_type(runtime_config),
        "text_parameters": len(actual_text_keys),
        "selected_parameters": len(actual_selected_keys),
        "includes_vision": include_vision,
        "full_attention_layers": text_config["layer_types"].count("full_attention"),
        "linear_attention_layers": text_config["layer_types"].count("linear_attention"),
        "quantized_parameters": len(quantized_names),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "pattern_counts": pattern_counts,
    }

    expected_quantized_names = _expected_kai_linear_names(actual_selected_keys)
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
