#!/usr/bin/env python3
"""Generate exact-length ShadowNPU archive-retrieval prompts.

The default construction creates naturally exact-length prompts. The optional
``pr703-records-window`` construction reproduces PR #703's ``records`` input:
it tokenizes a longer archive, selects the first and last token windows, then
decodes the selected user tokens into text that mllmV2 can re-tokenize exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

from tokenizers import Tokenizer


CHAT_TEMPLATE_PREFIX = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
)
CHAT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
CHAT_TEMPLATE = CHAT_TEMPLATE_PREFIX + "{prompt}" + CHAT_TEMPLATE_SUFFIX
NEEDLE_PATTERN = re.compile(r"[A-Z]{3}-[0-9]{4}\Z")
PR703_HEAD = "7e16701819ed3540c1d6aa35a8bec0c731cb08de"
PR703_CONSTRUCTION = "pr703-records-window"
EXACT_CONSTRUCTION = "exact-records"


def token_ids(tokenizer: Tokenizer, prompt: str) -> list[int]:
    return tokenizer.encode(CHAT_TEMPLATE.format(prompt=prompt), add_special_tokens=False).ids


def make_exact_prompt(
    tokenizer: Tokenizer, target_tokens: int, needle: str
) -> tuple[str, list[int], dict[str, object]]:
    prefix = (
        f"Archive target record 0000 contains access code {needle}. "
        "Preserve this target exactly.\n"
    )
    padding_prefix = "Archive padding words:"
    suffix = (
        "\nDecoy warning: CPU-1842 is obsolete and is not the target.\n"
        "Question: What access code is stored in archive target record 0000? "
        "Answer with the target code only."
    )

    records: list[str] = []
    record_index = 1
    while True:
        record = (
            f"Archive record {record_index:04d}: routine mobile inference note, "
            f"checksum {(record_index * 7919) % 10000:04d}.\n"
        )
        candidate = prefix + "".join(records) + record + padding_prefix + suffix
        if len(token_ids(tokenizer, candidate)) > target_tokens:
            break
        records.append(record)
        record_index += 1

    base = prefix + "".join(records) + padding_prefix
    for padding_words in range(target_tokens + 1):
        prompt = base + " hello" * padding_words + suffix
        ids = token_ids(tokenizer, prompt)
        if len(ids) == target_tokens:
            return prompt, ids, {
                "source_records": len(records),
                "padding_words": padding_words,
                "source_prompt_tokens": len(ids),
                "head_tokens": len(ids),
                "tail_tokens": 0,
                "roundtrip_verified": True,
            }
        if len(ids) > target_tokens:
            break
    raise ValueError(f"could not construct an exact {target_tokens}-token prompt")


def make_pr703_source_prompt(target_tokens: int, needle: str) -> tuple[str, int]:
    """Reproduce benchmark_qwen_npu.cpp's ``quality_prompt == records`` text."""

    records = max(1024, target_tokens // 2)
    parts = [
        f"Archive target record 0000 contains access code {needle}. "
        "Preserve this target exactly.\n"
    ]
    for record_index in range(1, records + 1):
        parts.append(
            f"Archive record {record_index}: routine mobile inference note, "
            f"checksum {(record_index * 7919) % 10000}.\n"
        )
    parts.append(
        "Decoy warning: CPU-1842 is obsolete and is not the target.\n"
        "Question: What access code is stored in archive target record 0000? "
        "Answer with the target code only."
    )
    return "".join(parts), records


def make_pr703_records_window_prompt(
    tokenizer: Tokenizer, target_tokens: int, needle: str
) -> tuple[str, list[int], dict[str, object]]:
    source_prompt, records = make_pr703_source_prompt(target_tokens, needle)
    source_ids = token_ids(tokenizer, source_prompt)
    if len(source_ids) < target_tokens:
        raise ValueError(
            f"PR #703 source prompt produced only {len(source_ids)} tokens; "
            f"need {target_tokens}"
        )

    tail_tokens = min(256, target_tokens // 3)
    head_tokens = target_tokens - tail_tokens
    selected_ids = source_ids[:head_tokens] + source_ids[-tail_tokens:]

    prefix_ids = tokenizer.encode(
        CHAT_TEMPLATE_PREFIX, add_special_tokens=False
    ).ids
    suffix_ids = tokenizer.encode(
        CHAT_TEMPLATE_SUFFIX, add_special_tokens=False
    ).ids
    if selected_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("PR #703 selected tokens do not preserve the chat-template prefix")
    if selected_ids[-len(suffix_ids) :] != suffix_ids:
        raise ValueError("PR #703 selected tokens do not preserve the chat-template suffix")

    selected_user_ids = selected_ids[len(prefix_ids) : -len(suffix_ids)]
    prompt = tokenizer.decode(selected_user_ids, skip_special_tokens=False)
    roundtrip_ids = token_ids(tokenizer, prompt)
    if roundtrip_ids != selected_ids:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(roundtrip_ids, selected_ids, strict=False)
                )
                if actual != expected
            ),
            min(len(roundtrip_ids), len(selected_ids)),
        )
        raise ValueError(
            "PR #703 selected token window is not text-roundtrippable for mllmV2: "
            f"first mismatch at token {mismatch}, roundtrip count "
            f"{len(roundtrip_ids)}, expected {len(selected_ids)}"
        )

    return prompt, selected_ids, {
        "source_records": records,
        "padding_words": 0,
        "source_prompt_tokens": len(source_ids),
        "source_token_ids_sha256_le_i64": ids_sha256(source_ids),
        "head_tokens": head_tokens,
        "tail_tokens": tail_tokens,
        "roundtrip_verified": True,
    }


def make_prompt(
    tokenizer: Tokenizer, target_tokens: int, needle: str, construction: str
) -> tuple[str, list[int], dict[str, object]]:
    if construction == EXACT_CONSTRUCTION:
        return make_exact_prompt(tokenizer, target_tokens, needle)
    if construction == PR703_CONSTRUCTION:
        return make_pr703_records_window_prompt(tokenizer, target_tokens, needle)
    raise ValueError(f"unknown prompt construction: {construction}")


def ids_sha256(ids: list[int]) -> str:
    payload = b"".join(struct.pack("<q", value) for value in ids)
    return hashlib.sha256(payload).hexdigest()


def ids_fnv1a64_le_u32(ids: list[int]) -> int:
    """Match PR #703 benchmark_qwen_npu.cpp::hashTokens."""

    value = 1469598103934665603
    for token in ids:
        word = token & 0xFFFFFFFF
        for _ in range(4):
            value ^= word & 0xFF
            value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
            word >>= 8
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--needle", default="NPU-7391")
    parser.add_argument(
        "--construction",
        choices=[EXACT_CONSTRUCTION, PR703_CONSTRUCTION],
        default=EXACT_CONSTRUCTION,
        help="Prompt construction contract; use pr703-records-window for exact PR #703 A/B inputs",
    )
    args = parser.parse_args()

    if not NEEDLE_PATTERN.fullmatch(args.needle):
        parser.error("--needle must match [A-Z]{3}-[0-9]{4}")
    if any(length <= 0 or length % 512 != 0 for length in args.lengths):
        parser.error("--lengths must be positive multiples of 512")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "schema": 2,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "needle": args.needle,
        "chat_template": CHAT_TEMPLATE,
        "construction": args.construction,
        "token_ids_sha256_encoding": "little-endian signed Int64 per token",
        "token_ids_fnv1a64_encoding": "little-endian Uint32 per token; PR #703 hashTokens",
        "prompts": [],
    }
    if args.construction == PR703_CONSTRUCTION:
        metadata["reference"] = {
            "repository": "UbiquitousLearning/mllm",
            "pull_request": 703,
            "head": PR703_HEAD,
            "source": "examples/benchmark_qwen_npu.cpp",
            "quality_prompt": "records",
        }
    rows: list[dict[str, object]] = []
    for length in args.lengths:
        prompt, ids, construction_metadata = make_prompt(
            tokenizer, length, args.needle, args.construction
        )
        path = args.output_dir / f"prompt-{length}.txt"
        path.write_text(prompt, encoding="utf-8")
        row: dict[str, object] = {
            "path": path.name,
            "prompt_tokens": length,
            "token_ids_sha256_le_i64": ids_sha256(ids),
            "token_ids_fnv1a64_le_u32": ids_fnv1a64_le_u32(ids),
            "first_token_ids": ids[:16],
            "last_token_ids": ids[-16:],
            "text_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        row.update(construction_metadata)
        rows.append(row)
    metadata["prompts"] = rows
    (args.output_dir / "PROMPTS.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(
            f"{row['path']} tokens={row['prompt_tokens']} "
            f"text_sha256={row['text_sha256']} token_ids_sha256={row['token_ids_sha256_le_i64']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
