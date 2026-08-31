#!/usr/bin/env python3
"""Validate mllmV2 ShadowNPU A/B logs and summarize matched medians."""

from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


NAME = re.compile(r"(?P<mode>dense|sparse)-p(?P<prompt>\d+)-run\d+\.log$")


def fields(line: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)


@dataclass(frozen=True)
class Run:
    prompt_tokens: int
    mode: str
    prefill_ms: float
    sparse_total_ms: float
    actual_retention: float
    token_ids: tuple[int, ...]
    expected_code: str
    extracted_code: str


def parse_run(path: Path) -> Run:
    match = NAME.search(path.name)
    if match is None:
        raise ValueError(f"unexpected log name: {path}")
    text = path.read_text(encoding="utf-8", errors="replace").replace("\0", "")
    result_lines = [line for line in text.splitlines() if line.startswith("SHADOWNPU_RESULT")]
    token_lines = [line for line in text.splitlines() if line.startswith("SHADOWNPU_TOKEN_IDS")]
    quality_lines = [line for line in text.splitlines() if line.startswith("QUALITY_RESULT")]
    harness_lines = [line for line in text.splitlines() if line.startswith("SHADOWNPU_HARNESS_CONFIG")]
    if len(result_lines) != 1 or len(token_lines) != 1 or len(quality_lines) != 1 or len(harness_lines) != 1:
        raise ValueError(
            f"{path}: expected one harness, SHADOWNPU_RESULT, SHADOWNPU_TOKEN_IDS, and QUALITY_RESULT line"
        )
    if re.search(r"FastRPC.*(?:error|failed)|QNN.*(?:error|failed)", text, re.I):
        raise ValueError(f"{path}: FastRPC/QNN error found")

    result = fields(result_lines[0])
    mode = match.group("mode")
    harness = fields(harness_lines[0])
    expected_attention = harness.get("attention_mode", "dense" if mode == "dense" else "hmx-topk")
    if result.get("attention_mode") != expected_attention:
        raise ValueError(f"{path}: unexpected attention mode")
    prompt_tokens = int(match.group("prompt"))
    if int(result["prompt_tokens"]) != prompt_tokens:
        raise ValueError(f"{path}: tokenized prompt length does not match its bound label")
    if int(result.get("prefill_chunks", "0")) != prompt_tokens // 512 or prompt_tokens % 512 != 0:
        raise ValueError(f"{path}: QNN prefill did not use exact logical M512 chunks")
    if int(result.get("qnn_tile_sequence", "0")) != 256:
        raise ValueError(f"{path}: QNN graph was not compiled/executed as M256 tiles")
    qnn_graph_calls = int(result.get("qnn_graph_calls", "0"))
    qnn_tile_calls = int(result.get("qnn_tile_calls", "0"))
    if qnn_graph_calls <= 0 or qnn_tile_calls != 2 * qnn_graph_calls:
        raise ValueError(f"{path}: QNN M512-to-M256x2 execution counters are inconsistent")
    if float(result.get("qnn_execute_ms", "0")) <= 0.0:
        raise ValueError(f"{path}: QNN tiled execution counter did not advance")

    token_field = fields(token_lines[0]).get("ids", "")
    token_ids = tuple(int(value) for value in token_field.split(",") if value)
    if not token_ids or len(token_ids) != int(result["generated_tokens"]):
        raise ValueError(f"{path}: missing or inconsistent generated token IDs")
    quality = fields(quality_lines[0])
    if quality.get("retrieval_exact") not in {"1", "true"}:
        raise ValueError(f"{path}: strict access-code retrieval failed")
    expected_code = quality.get("expected_code", "")
    extracted_code = quality.get("extracted_code", "")
    if not expected_code or extracted_code != expected_code:
        raise ValueError(f"{path}: inconsistent strict retrieval receipt")

    actual_retention = float(result["actual_retention"])
    sparse_total_ms = float(result["sparse_total_ms"])
    hmx_select_ms = float(result["hmx_select_ms"])
    if mode == "dense":
        if actual_retention != 1.0 or sparse_total_ms != 0.0 or hmx_select_ms != 0.0:
            raise ValueError(f"{path}: dense baseline unexpectedly reported sparse work")
    else:
        if "FastRPC redirect" not in text:
            raise ValueError(f"{path}: sparse run did not observe the FastRPC session redirect")
        if not 0.0 < actual_retention < 1.0:
            raise ValueError(f"{path}: sparse run reported invalid actual retention")
        if sparse_total_ms <= 0.0 or hmx_select_ms <= 0.0:
            raise ValueError(f"{path}: sparse/HMX profiling counters did not advance")

    return Run(
        prompt_tokens=prompt_tokens,
        mode=mode,
        prefill_ms=float(result["prefill_ms"]),
        sparse_total_ms=sparse_total_ms,
        actual_retention=actual_retention,
        token_ids=token_ids,
        expected_code=expected_code,
        extracted_code=extracted_code,
    )


def summarize(paths: list[Path]) -> str:
    runs: dict[tuple[int, str], list[Run]] = {}
    for path in paths:
        run = parse_run(path)
        runs.setdefault((run.prompt_tokens, run.mode), []).append(run)

    lines = [
        "prompt_tokens\tdense_prefill_ms\tsparse_prefill_ms\tprefill_speedup\t"
        "sparse_attention_ms\tactual_retention\tretrieval_exact\ttoken_ids_match"
    ]
    for prompt in sorted({prompt for prompt, _ in runs}):
        dense = runs.get((prompt, "dense"), [])
        sparse = runs.get((prompt, "sparse"), [])
        if not dense or not sparse:
            raise ValueError(f"prompt {prompt}: both dense and sparse runs are required")
        retrieval_codes = {run.expected_code for run in dense + sparse}
        if len(retrieval_codes) != 1:
            raise ValueError(f"prompt {prompt}: dense/sparse expected access codes differ")
        token_ids_match = int(len({run.token_ids for run in dense + sparse}) == 1)
        if token_ids_match != 1:
            raise ValueError(f"prompt {prompt}: dense/sparse generated token IDs differ")
        dense_prefill = statistics.median(run.prefill_ms for run in dense)
        sparse_prefill = statistics.median(run.prefill_ms for run in sparse)
        sparse_attention = statistics.median(run.sparse_total_ms for run in sparse)
        actual_retention = statistics.median(run.actual_retention for run in sparse)
        lines.append(
            f"{prompt}\t{dense_prefill:.3f}\t{sparse_prefill:.3f}\t"
            f"{dense_prefill / sparse_prefill:.3f}\t{sparse_attention:.3f}\t"
            f"{actual_retention:.8f}\t1\t{token_ids_match}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        output = summarize(args.logs)
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
