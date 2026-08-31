#!/usr/bin/env python3
"""Byte-compare exported W8A16 tensors with an mllm V2 model file."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import struct
from pathlib import Path
from typing import Any

import torch


MODEL_HEADER = struct.Struct("<ii512sIQ")
PARAMETER_DESCRIPTOR = struct.Struct("<IIQQQ16i256s")
MODEL_MAGIC = 0x519A
MODEL_VERSION = 2

TORCH_TO_MLLM_DTYPE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.int8: 16,
    torch.int16: 17,
    torch.int32: 18,
    torch.uint8: 129,
    torch.int64: 132,
    torch.bool: 129,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mllm-model", required=True, type=Path)
    parser.add_argument("--quantized-state", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.uint8)
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def parse_descriptors(mapped: mmap.mmap) -> tuple[str, list[dict[str, Any]]]:
    magic, version, raw_model_name, num_params, descriptor_offset = MODEL_HEADER.unpack_from(mapped, 0)
    if magic != MODEL_MAGIC or version != MODEL_VERSION:
        raise ValueError(f"not an mllm V2 model: magic={magic:#x} version={version}")
    model_name = raw_model_name.split(b"\0", 1)[0].decode("utf-8")
    descriptors = []
    for index in range(num_params):
        fields = PARAMETER_DESCRIPTOR.unpack_from(mapped, descriptor_offset + index * PARAMETER_DESCRIPTOR.size)
        parameter_id, parameter_type, parameter_size, parameter_offset, shape_len = fields[:5]
        if shape_len > 16:
            raise ValueError(f"parameter {index} has invalid shape length {shape_len}")
        shape = list(fields[5 : 5 + shape_len])
        name = fields[21].split(b"\0", 1)[0].decode("utf-8")
        descriptors.append(
            {
                "index": index,
                "parameter_id": parameter_id,
                "parameter_type": parameter_type,
                "parameter_size": parameter_size,
                "parameter_offset": parameter_offset,
                "shape": shape,
                "name": name,
            }
        )
    return model_name, descriptors


def main() -> int:
    args = parse_args()
    state = torch.load(args.quantized_state, map_location="cpu", weights_only=True)
    results: list[dict[str, Any]] = []
    with args.mllm_model.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        model_name, descriptors = parse_descriptors(mapped)
        selected = [descriptor for descriptor in descriptors if descriptor["name"].startswith(args.prefix)]
        if not selected:
            raise ValueError(f"no mllm tensors match prefix {args.prefix!r}")
        for descriptor in selected:
            name = descriptor["name"]
            item: dict[str, Any] = {**descriptor}
            tensor = state.get(name)
            if tensor is None:
                item["status"] = "missing_in_state"
                results.append(item)
                continue
            expected_type = TORCH_TO_MLLM_DTYPE.get(tensor.dtype)
            expected_bytes = tensor_bytes(tensor)
            offset = descriptor["parameter_offset"]
            size = descriptor["parameter_size"]
            actual_bytes = mapped[offset : offset + size]
            item.update(
                {
                    "state_dtype": str(tensor.dtype),
                    "state_shape": list(tensor.shape),
                    "state_size": len(expected_bytes),
                    "state_sha256": sha256_bytes(expected_bytes),
                    "mllm_sha256": sha256_bytes(actual_bytes),
                }
            )
            checks = {
                "dtype": expected_type == descriptor["parameter_type"],
                "shape": list(tensor.shape) == descriptor["shape"],
                "size": len(expected_bytes) == size,
                "bytes": expected_bytes == actual_bytes,
            }
            item["checks"] = checks
            item["status"] = "pass" if all(checks.values()) else "mismatch"
            results.append(item)

    result_names = {item["name"] for item in results}
    missing_in_mllm = sorted(name for name in state if name.startswith(args.prefix) and name not in result_names)
    summary = {
        "schema": 1,
        "model_name": model_name,
        "mllm_model": str(args.mllm_model),
        "quantized_state": str(args.quantized_state),
        "prefix": args.prefix,
        "selected": len(results),
        "passed": sum(item["status"] == "pass" for item in results),
        "mismatched": sum(item["status"] == "mismatch" for item in results),
        "missing_in_state": sum(item["status"] == "missing_in_state" for item in results),
        "missing_in_mllm": missing_in_mllm,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "MLLM_W8A16_SERIALIZATION_AUDIT "
        f"prefix={args.prefix} selected={summary['selected']} passed={summary['passed']} "
        f"mismatched={summary['mismatched']} missing_in_state={summary['missing_in_state']} "
        f"missing_in_mllm={len(missing_in_mllm)}"
    )
    return 0 if summary["passed"] == summary["selected"] and not missing_in_mllm else 1


if __name__ == "__main__":
    raise SystemExit(main())
