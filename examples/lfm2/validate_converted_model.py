# Copyright (c) MLLM Team.
# Licensed under the MIT License.
"""Audit an LFM2.5-2.6B MLLM V2 descriptor table without loading tensor data."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from pathlib import Path

from validate_checkpoint import expected_shapes, validate_config, validate_recipe, validate_runtime_config


MODEL_HEADER = struct.Struct("<II512sIQ")
PARAMETER_DESCRIPTOR = struct.Struct("<IIQQQ16i256s")
MODEL_MAGIC = 0x519A
MODEL_VERSION = 2
FLOAT32 = 0
BYTE = 134


def packed_size(out_channels: int, in_channels: int) -> int:
    if in_channels <= 0 or in_channels % 32 or out_channels <= 0:
        raise AssertionError("Invalid KAI W4A32 matrix dimensions")
    blocks_per_row = in_channels // 32
    bytes_per_eight_rows = 8 * (blocks_per_row * 18 + 8)
    return math.ceil(out_channels / 8) * bytes_per_eight_rows


def expected_descriptors(recipe: dict) -> dict[str, tuple[int, list[int], int]]:
    source = expected_shapes()
    rules = [(re.compile(pattern), entry["hints"]) for pattern, entry in recipe.items()]
    expected: dict[str, tuple[int, list[int], int]] = {}
    for name, shape in source.items():
        matches = [hints for pattern, hints in rules if pattern.fullmatch(name)]
        if not matches:
            expected[name] = (FLOAT32, shape, math.prod(shape) * 4)
            continue
        if len(matches) != 1:
            raise AssertionError(f"Ambiguous recipe match for {name}")
        hints = matches[0]
        packed = packed_size(shape[0], shape[1])
        descriptor = (BYTE, [packed], packed)
        if hints["replace"]:
            expected[name] = descriptor
        else:
            expected[name] = (FLOAT32, shape, math.prod(shape) * 4)
            expected[hints["rename"]] = descriptor
    return expected


def c_string(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model-name", default="LFM2.5-2.6B")
    parser.add_argument("--quant-config", type=Path, default=Path(__file__).with_name("quant_cfg_2.6B_w4a32_kai.json"))
    parser.add_argument("--runtime-config", type=Path, default=Path(__file__).with_name("config_2.6B_w4a32_kai.json"))
    args = parser.parse_args()

    checkpoint = json.loads((args.checkpoint / "config.json").read_text())
    runtime = json.loads(args.runtime_config.read_text())
    recipe = json.loads(args.quant_config.read_text())
    validate_config(checkpoint)
    validate_runtime_config(checkpoint, runtime)
    validate_recipe(recipe, expected_shapes())
    expected = expected_descriptors(recipe)

    file_size = args.model.stat().st_size
    actual: dict[str, tuple[int, list[int], int, int]] = {}
    with args.model.open("rb") as stream:
        raw_header = stream.read(MODEL_HEADER.size)
        if len(raw_header) != MODEL_HEADER.size:
            raise AssertionError("Truncated MLLM V2 header")
        magic, version, raw_name, count, descriptor_offset = MODEL_HEADER.unpack(raw_header)
        if (magic, version, c_string(raw_name), descriptor_offset) != (
            MODEL_MAGIC,
            MODEL_VERSION,
            args.model_name,
            MODEL_HEADER.size,
        ):
            raise AssertionError("Invalid MLLM V2 model header")
        if count != len(expected):
            raise AssertionError(f"Expected {len(expected)} descriptors, file declares {count}")
        for parameter_id in range(count):
            raw = stream.read(PARAMETER_DESCRIPTOR.size)
            if len(raw) != PARAMETER_DESCRIPTOR.size:
                raise AssertionError(f"Truncated descriptor {parameter_id}")
            fields = PARAMETER_DESCRIPTOR.unpack(raw)
            actual_id, dtype, size, offset, rank = fields[:5]
            if actual_id != parameter_id or rank > 16:
                raise AssertionError(f"Invalid descriptor id/rank at {parameter_id}")
            name = c_string(fields[21])
            if name in actual:
                raise AssertionError(f"Duplicate descriptor: {name}")
            actual[name] = (dtype, list(fields[5:21])[:rank], size, offset)

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    wrong = sorted(name for name in set(expected) & set(actual) if actual[name][:3] != expected[name])
    if missing or extra or wrong:
        raise AssertionError(f"Converted tensor mismatch: missing={missing}, extra={extra}, wrong={wrong}")
    data_start = MODEL_HEADER.size + len(actual) * PARAMETER_DESCRIPTOR.size
    next_offset = data_start
    for name, (_, _, size, offset) in sorted(actual.items(), key=lambda item: item[1][3]):
        if offset != next_offset:
            raise AssertionError(f"Non-contiguous data before {name}: {offset} != {next_offset}")
        next_offset += size
    if next_offset != file_size:
        raise AssertionError(f"Tensor data ends at {next_offset}, file size is {file_size}")
    print(json.dumps({"model": str(args.model), "parameters": len(actual), "model_bytes": file_size,
                      "has_lm_head_out": "lm_head_out.weight" in actual,
                      "logical_cache_slots": 8}, indent=2, sort_keys=True))
    print("LFM2_CONVERTED_MODEL_AUDIT_OK")


if __name__ == "__main__":
    main()
