# Copyright (c) MLLM Team.
# Licensed under the MIT License.

"""Validate a converted Qwen3.5 MLLM V2 model without loading tensor data."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path

from validate_checkpoint import (
    MODEL_VARIANTS,
    default_quant_config_path,
    default_runtime_config_path,
    expected_text_shapes,
    expected_vision_shapes,
    model_name_for_size,
    resolve_model_size,
    resolve_runtime_linear_impl_type,
    validate_kai_recipe_contract,
    validate_multimodal_config_contract,
)


MODEL_HEADER = struct.Struct("<II512sIQ")
PARAMETER_DESCRIPTOR = struct.Struct("<IIQQQ16i256s")
MODEL_MAGIC = 0x519A
MODEL_VERSION = 2
FLOAT32 = 0
BYTE = 134
# Exclusive upper bound of the unsigned 32-bit range; offsets at or above this
# value cannot be addressed by a 32-bit reader.
UINT32_LIMIT = 1 << 32


def _packed_size(out_channels: int, in_channels: int, tile_name: str) -> int:
    if tile_name != "qai8dxp1x8_qsi4c32p8x8_1x8x32":
        raise ValueError(f"Unsupported KAI tile configuration: {tile_name}")
    if out_channels <= 0 or in_channels <= 0 or in_channels % 32:
        raise ValueError(
            "KAI W4A32 dimensions must be positive and input channels divisible by 32"
        )

    nr = 8
    blocks_per_row = in_channels // 32
    bytes_per_nr_rows = nr * (blocks_per_row * (16 + 2) + 4 + 4)
    return math.ceil(out_channels / nr) * bytes_per_nr_rows


def _decode_c_string(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8")


def _expected_descriptors(
    config: dict, quant_config: dict
) -> dict[str, tuple[int, list[int], int]]:
    text_config = config.get("text_config", config)
    source_shapes = expected_text_shapes(text_config)
    if any("visual" in pattern for pattern in quant_config):
        vision_config = config.get("vision_config")
        if not isinstance(vision_config, dict):
            raise AssertionError(
                "Multimodal converted-model validation requires vision_config"
            )
        source_shapes.update(expected_vision_shapes(vision_config))
    patterns = [
        (re.compile(pattern), entry["hints"]) for pattern, entry in quant_config.items()
    ]
    expected: dict[str, tuple[int, list[int], int]] = {
        name: (FLOAT32, shape, math.prod(shape) * 4)
        for name, shape in source_shapes.items()
    }

    matched_sources: set[str] = set()
    for pattern, hints in patterns:
        matched = sorted(name for name in source_shapes if pattern.fullmatch(name))
        if not matched:
            raise AssertionError(
                f"Quantization pattern matched no selected tensors: {pattern.pattern}"
            )
        overlap = matched_sources.intersection(matched)
        if overlap:
            raise AssertionError(
                f"Multiple quantization patterns matched: {sorted(overlap)}"
            )
        weights = [name for name in matched if name.endswith(".weight")]
        if not weights:
            raise AssertionError(
                f"Quantization pattern matched no weight tensor: {pattern.pattern}"
            )
        for weight_name in weights:
            shape = source_shapes[weight_name]
            if list(hints["shape"]) != shape:
                raise AssertionError(
                    f"Quantization config shape mismatch for {weight_name}: "
                    f"{hints['shape']} != {shape}"
                )
            packed_size = _packed_size(
                out_channels=shape[0],
                in_channels=shape[1],
                tile_name=hints["kai_matmul_tile_cfg"],
            )
            packed_descriptor = (BYTE, [packed_size], packed_size)
            bias_name = weight_name.removesuffix(".weight") + ".bias"
            if hints["replace"]:
                expected[weight_name] = packed_descriptor
                if bias_name in matched:
                    expected.pop(bias_name)
            else:
                expected[hints["rename"]] = packed_descriptor
        matched_sources.update(matched)

    return expected


def _validate_descriptor_table(
    actual: dict[str, tuple[int, list[int], int, int]],
    expected: dict[str, tuple[int, list[int], int]],
    data_start: int,
    file_size: int,
) -> Counter:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise AssertionError(
            f"Parameter-name mismatch: missing={missing}, extra={extra}"
        )

    for name, expected_descriptor in expected.items():
        dtype, shape, size, _ = actual[name]
        if (dtype, shape, size) != expected_descriptor:
            raise AssertionError(
                f"{name}: expected {expected_descriptor}, got {(dtype, shape, size)}"
            )

    next_offset = data_start
    for name, (_, _, size, offset) in sorted(
        actual.items(), key=lambda item: item[1][3]
    ):
        if offset != next_offset:
            raise AssertionError(
                f"{name}: expected contiguous offset {next_offset}, got {offset}"
            )
        next_offset += size
    if next_offset != file_size:
        raise AssertionError(
            f"Tensor data ends at {next_offset}, but file size is {file_size}"
        )

    return Counter(dtype for dtype, _, _, _ in actual.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
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
    parser.add_argument(
        "--model-name",
        help="Expected V2 header model name (default: Qwen3.5-<detected size>)",
    )
    args = parser.parse_args()

    with (args.checkpoint / "config.json").open() as config_file:
        checkpoint_config = json.load(config_file)
    text_config = checkpoint_config["text_config"]
    model_size = resolve_model_size(text_config, args.model_size)
    quant_config_path = args.quant_config or default_quant_config_path(model_size)
    runtime_config_path = args.runtime_config or default_runtime_config_path(model_size)
    expected_model_name = args.model_name or model_name_for_size(model_size)
    with quant_config_path.open() as quant_config_file:
        quant_config = json.load(quant_config_file)
    with runtime_config_path.open() as runtime_config_file:
        runtime_config = json.load(runtime_config_file)
    include_vision = any("visual" in pattern for pattern in quant_config)
    if include_vision:
        validate_multimodal_config_contract(checkpoint_config, runtime_config)
        expected_model_name = args.model_name or f"Qwen3.5-{model_size}-Multimodal"
    validate_kai_recipe_contract(
        text_config,
        quant_config,
        runtime_config,
        checkpoint_config.get("vision_config"),
    )
    expected = _expected_descriptors(checkpoint_config, quant_config)

    file_size = args.model.stat().st_size
    with args.model.open("rb") as model_file:
        header_bytes = model_file.read(MODEL_HEADER.size)
        if len(header_bytes) != MODEL_HEADER.size:
            raise AssertionError("Truncated MLLM V2 model header")
        magic, version, raw_model_name, num_params, descriptor_offset = (
            MODEL_HEADER.unpack(header_bytes)
        )
        model_name = _decode_c_string(raw_model_name)
        if (magic, version, model_name, descriptor_offset) != (
            MODEL_MAGIC,
            MODEL_VERSION,
            expected_model_name,
            MODEL_HEADER.size,
        ):
            raise AssertionError(
                "Invalid model header: "
                f"magic={magic:#x}, version={version}, model_name={model_name!r}, "
                f"descriptor_offset={descriptor_offset}"
            )
        if num_params != len(expected):
            raise AssertionError(
                f"Expected {len(expected)} parameters, model declares {num_params}"
            )

        actual: dict[str, tuple[int, list[int], int, int]] = {}
        for expected_id in range(num_params):
            raw_descriptor = model_file.read(PARAMETER_DESCRIPTOR.size)
            if len(raw_descriptor) != PARAMETER_DESCRIPTOR.size:
                raise AssertionError(
                    f"Truncated parameter descriptor at index {expected_id}"
                )
            unpacked = PARAMETER_DESCRIPTOR.unpack(raw_descriptor)
            parameter_id, dtype, size, offset, shape_len = unpacked[:5]
            shape = list(unpacked[5:21])
            name = _decode_c_string(unpacked[21])
            if parameter_id != expected_id:
                raise AssertionError(
                    f"Parameter ID mismatch: expected {expected_id}, got {parameter_id}"
                )
            if shape_len > 16:
                raise AssertionError(f"{name}: invalid shape length {shape_len}")
            if name in actual:
                raise AssertionError(f"Duplicate parameter name: {name}")
            actual[name] = (dtype, shape[:shape_len], size, offset)

    data_start = MODEL_HEADER.size + num_params * PARAMETER_DESCRIPTOR.size
    dtype_counts = _validate_descriptor_table(
        actual,
        expected,
        data_start,
        file_size,
    )
    summary = {
        "model": str(args.model),
        "model_size": model_size,
        "quant_config": str(quant_config_path),
        "runtime_config": str(runtime_config_path),
        "linear_impl_type": resolve_runtime_linear_impl_type(runtime_config),
        "model_bytes": file_size,
        "parameters": num_params,
        "float32_parameters": dtype_counts[FLOAT32],
        "kai_packed_parameters": dtype_counts[BYTE],
        "crosses_uint32_offset": any(
            offset >= UINT32_LIMIT for _, _, _, offset in actual.values()
        ),
        "has_lm_head_out": "lm_head_out.weight" in actual,
        "visual_parameters": sum(name.startswith("model.visual.") for name in actual),
        "mtp_parameters": sum("mtp" in name for name in actual),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("QWEN35_CONVERTED_MODEL_AUDIT_OK")


if __name__ == "__main__":
    main()
