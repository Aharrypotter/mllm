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

from validate_checkpoint import expected_text_shapes


MODEL_HEADER = struct.Struct("<II512sIQ")
PARAMETER_DESCRIPTOR = struct.Struct("<IIQQQ16i256s")
MODEL_MAGIC = 0x519A
MODEL_VERSION = 2
FLOAT32 = 0
BYTE = 134


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
    text_config: dict, quant_config: dict
) -> dict[str, tuple[int, list[int], int]]:
    source_shapes = expected_text_shapes(text_config)
    patterns = [
        (re.compile(pattern), entry["hints"]) for pattern, entry in quant_config.items()
    ]
    expected: dict[str, tuple[int, list[int], int]] = {}

    for name, shape in source_shapes.items():
        matches = [hints for pattern, hints in patterns if pattern.fullmatch(name)]
        if len(matches) > 1:
            raise AssertionError(f"Multiple quantization patterns matched {name}")

        if not matches:
            expected[name] = (FLOAT32, shape, math.prod(shape) * 4)
            continue

        hints = matches[0]
        if list(hints["shape"]) != shape:
            raise AssertionError(
                f"Quantization config shape mismatch for {name}: "
                f"{hints['shape']} != {shape}"
            )
        packed_size = _packed_size(
            out_channels=shape[0],
            in_channels=shape[1],
            tile_name=hints["kai_matmul_tile_cfg"],
        )
        packed_descriptor = (BYTE, [packed_size], packed_size)
        if hints["replace"]:
            expected[name] = packed_descriptor
        else:
            expected[name] = (FLOAT32, shape, math.prod(shape) * 4)
            expected[hints["rename"]] = packed_descriptor

    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--quant-config",
        type=Path,
        default=Path(__file__).with_name("quant_cfg_0.8B_w4a32_kai.json"),
    )
    parser.add_argument("--model-name", default="Qwen3.5-0.8B")
    args = parser.parse_args()

    with (args.checkpoint / "config.json").open() as config_file:
        text_config = json.load(config_file)["text_config"]
    with args.quant_config.open() as quant_config_file:
        quant_config = json.load(quant_config_file)
    expected = _expected_descriptors(text_config, quant_config)

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
            args.model_name,
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

    data_start = MODEL_HEADER.size + num_params * PARAMETER_DESCRIPTOR.size
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

    dtype_counts = Counter(dtype for dtype, _, _, _ in actual.values())
    summary = {
        "model": str(args.model),
        "model_bytes": file_size,
        "parameters": num_params,
        "float32_parameters": dtype_counts[FLOAT32],
        "kai_packed_parameters": dtype_counts[BYTE],
        "has_lm_head_out": "lm_head_out.weight" in actual,
        "visual_or_mtp_parameters": sum(
            "visual" in name or "mtp" in name for name in actual
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("QWEN35_CONVERTED_MODEL_AUDIT_OK")


if __name__ == "__main__":
    main()
