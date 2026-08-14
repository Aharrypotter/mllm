# Copyright (c) MLLM Team.
# Licensed under the MIT License.
"""Fail-closed audit for the official LFM2.5-2.6B checkpoint and W4A32 recipe."""

from __future__ import annotations

import argparse
import json
import re
from contextlib import ExitStack
from pathlib import Path

from safetensors import safe_open


OFFICIAL = {
    "architectures": ["Lfm2ForCausalLM"],
    "model_type": "lfm2",
    "hidden_size": 2048,
    "intermediate_size": 10752,
    "num_hidden_layers": 30,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "conv_L_cache": 3,
    "conv_bias": False,
    "block_auto_adjust_ff_dim": False,
    "norm_eps": 1e-5,
    "max_position_embeddings": 131072,
    "vocab_size": 128000,
    "tie_word_embeddings": True,
    "bos_token_id": 124894,
    "eos_token_id": 124900,
    "pad_token_id": 124893,
    "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
}
LAYER_TYPES = [
    "conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention",
    "conv", "conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv", "conv",
    "full_attention", "conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv",
]
KAI_FIELDS = {
    "quant_method": "kai",
    "kai_matmul_triplet": "f32_qai8dxp_qsi4c32p",
    "kai_matmul_layout": "mxk_nxk",
    "kai_matmul_tile_cfg": "qai8dxp1x8_qsi4c32p8x8_1x8x32",
}
KAI_IMPL = "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32"


def expected_shapes() -> dict[str, list[int]]:
    expected = {
        "model.embed_tokens.weight": [128000, 2048],
        "model.embedding_norm.weight": [2048],
    }
    for layer, layer_type in enumerate(LAYER_TYPES):
        prefix = f"model.layers.{layer}"
        expected.update(
            {
                f"{prefix}.operator_norm.weight": [2048],
                f"{prefix}.ffn_norm.weight": [2048],
                f"{prefix}.feed_forward.w1.weight": [10752, 2048],
                f"{prefix}.feed_forward.w3.weight": [10752, 2048],
                f"{prefix}.feed_forward.w2.weight": [2048, 10752],
            }
        )
        if layer_type == "conv":
            expected.update(
                {
                    f"{prefix}.conv.in_proj.weight": [6144, 2048],
                    f"{prefix}.conv.conv.weight": [2048, 1, 3],
                    f"{prefix}.conv.out_proj.weight": [2048, 2048],
                }
            )
        else:
            expected.update(
                {
                    f"{prefix}.self_attn.q_proj.weight": [2048, 2048],
                    f"{prefix}.self_attn.k_proj.weight": [512, 2048],
                    f"{prefix}.self_attn.v_proj.weight": [512, 2048],
                    f"{prefix}.self_attn.out_proj.weight": [2048, 2048],
                    f"{prefix}.self_attn.q_layernorm.weight": [64],
                    f"{prefix}.self_attn.k_layernorm.weight": [64],
                }
            )
    assert len(expected) == 266
    return expected


def validate_config(config: dict) -> None:
    mismatches = [
        f"{name}={config.get(name)!r}, expected {value!r}"
        for name, value in OFFICIAL.items()
        if config.get(name) != value
    ]
    if config.get("layer_types") != LAYER_TYPES:
        mismatches.append("layer_types differs from the official 30-layer physical schedule")
    if mismatches:
        raise AssertionError("Checkpoint contract mismatch: " + "; ".join(mismatches))


def validate_runtime_config(checkpoint: dict, runtime: dict) -> None:
    validate_config(runtime)
    for name in OFFICIAL:
        if runtime.get(name) != checkpoint.get(name):
            raise AssertionError(f"Runtime/checkpoint mismatch for {name}")
    if runtime.get("layer_types") != checkpoint.get("layer_types"):
        raise AssertionError("Runtime/checkpoint layer_types mismatch")
    if runtime.get("head_dim") != 64 or runtime.get("max_cache_length") != 2048:
        raise AssertionError("Runtime must bind head_dim=64 and max_cache_length=2048")
    if runtime.get("linear_impl_type") != KAI_IMPL:
        raise AssertionError("Runtime does not select the pinned KleidiAI W4A32 implementation")


def validate_recipe(recipe: dict, shapes: dict[str, list[int]]) -> None:
    patterns = [(re.compile(pattern), entry["hints"]) for pattern, entry in recipe.items()]
    matched_patterns = {pattern.pattern: 0 for pattern, _ in patterns}
    aliases: list[str] = []
    quantized: set[str] = set()
    for name, shape in shapes.items():
        matches = [(pattern, hints) for pattern, hints in patterns if pattern.fullmatch(name)]
        if len(matches) > 1:
            raise AssertionError(f"Multiple quantization rules match {name}")
        if not matches:
            continue
        pattern, hints = matches[0]
        matched_patterns[pattern.pattern] += 1
        quantized.add(name)
        if hints.get("shape") != shape:
            raise AssertionError(f"Recipe shape mismatch for {name}: {hints.get('shape')} != {shape}")
        for field, value in KAI_FIELDS.items():
            if hints.get(field) != value:
                raise AssertionError(f"Recipe {pattern.pattern} has invalid {field}")
        if hints.get("replace") is False:
            aliases.append(hints.get("rename"))
    unused = [pattern for pattern, count in matched_patterns.items() if count == 0]
    if unused:
        raise AssertionError(f"Quantization rules match no checkpoint tensor: {unused}")
    if aliases != ["lm_head_out.weight"]:
        raise AssertionError("Tied embedding must create exactly one lm_head_out.weight packed alias")
    intended_linears = {
        name
        for name, shape in shapes.items()
        if len(shape) == 2 and name != "model.embed_tokens.weight"
    }
    if not intended_linears.issubset(quantized):
        raise AssertionError(f"Unquantized intended Linear tensors: {sorted(intended_linears - quantized)}")
    allowed_non_linear = {"model.embed_tokens.weight"}
    unexpected = quantized - intended_linears - allowed_non_linear
    if unexpected:
        raise AssertionError(f"Recipe unexpectedly covers non-Linear tensors: {sorted(unexpected)}")


def checkpoint_shapes(checkpoint: Path) -> dict[str, list[int]]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
    elif (checkpoint / "model.safetensors").exists():
        shard_names = ["model.safetensors"]
    else:
        raise AssertionError("Checkpoint has no safetensors weights")
    actual: dict[str, list[int]] = {}
    with ExitStack() as stack:
        for shard_name in shard_names:
            handle = stack.enter_context(safe_open(checkpoint / shard_name, framework="pt", device="cpu"))
            for name in handle.keys():
                if name in actual:
                    raise AssertionError(f"Duplicate tensor across shards: {name}")
                actual[name] = list(handle.get_slice(name).get_shape())
    return actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--quant-config", type=Path, default=Path(__file__).with_name("quant_cfg_2.6B_w4a32_kai.json"))
    parser.add_argument("--runtime-config", type=Path, default=Path(__file__).with_name("config_2.6B_w4a32_kai.json"))
    args = parser.parse_args()

    checkpoint_config = json.loads((args.checkpoint / "config.json").read_text())
    runtime_config = json.loads(args.runtime_config.read_text())
    recipe = json.loads(args.quant_config.read_text())
    expected = expected_shapes()
    validate_config(checkpoint_config)
    validate_runtime_config(checkpoint_config, runtime_config)
    validate_recipe(recipe, expected)
    actual = checkpoint_shapes(args.checkpoint)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    wrong = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
    if missing or extra or wrong:
        raise AssertionError(f"Tensor contract mismatch: missing={missing}, extra={extra}, wrong_shapes={wrong}")
    print(json.dumps({"checkpoint": str(args.checkpoint), "parameters": len(actual), "attention_layers": 8,
                      "conv_layers": 22, "logical_cache_slots": 8}, indent=2, sort_keys=True))
    print("LFM2_CHECKPOINT_AUDIT_OK")


if __name__ == "__main__":
    main()
