#!/usr/bin/env python3
"""Compare complete little-endian Float32 dumps of first-token logits."""

import argparse
from array import array
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys


def read_logits(path, count):
    data = path.read_bytes()
    if count < 2 or len(data) != 4 * count:
        raise ValueError(f"{path}: expected {count} Float32 values, found {len(data)} bytes")
    values = array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite logit at token {index}")
    return values, hashlib.sha256(data).hexdigest()


def top_ids(values):
    return heapq.nlargest(5, range(len(values)), key=lambda index: (values[index], -index))


def compare(baseline_path, candidate_path, count):
    baseline, baseline_hash = read_logits(baseline_path, count)
    candidate, candidate_hash = read_logits(candidate_path, count)
    baseline_top = top_ids(baseline)
    candidate_top = top_ids(candidate)
    differences = [abs(right - left) for left, right in zip(baseline, candidate)]
    error_squared = math.fsum(value * value for value in differences)
    baseline_squared = math.fsum(value * value for value in baseline)
    baseline_top1, baseline_top2 = baseline_top[:2]
    candidate_top1, candidate_top2 = candidate_top[:2]
    return {
        "values": count,
        "baseline_sha256": baseline_hash,
        "candidate_sha256": candidate_hash,
        "changed_values": sum(value != 0 for value in differences),
        "max_abs_error": max(differences),
        "rmse": math.sqrt(error_squared / count),
        "relative_l2": math.sqrt(error_squared / baseline_squared) if baseline_squared else None,
        "baseline_top_ids": baseline_top,
        "candidate_top_ids": candidate_top,
        "top1_equal": baseline_top1 == candidate_top1,
        "top5_overlap": len(set(baseline_top) & set(candidate_top)),
        "baseline_top1_margin": baseline[baseline_top1] - baseline[baseline_top2],
        "candidate_top1_margin": candidate[candidate_top1] - candidate[candidate_top2],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--expected-vocab", type=int, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.baseline, args.candidate, args.expected_vocab)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
