#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("generate_quality_prompts.py")
SPEC = importlib.util.spec_from_file_location("generate_quality_prompts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PROMPTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROMPTS)


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        del add_special_tokens
        return _Encoding([ord(character) for character in text])

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(value) for value in ids)


class GenerateQualityPromptsTest(unittest.TestCase):
    def test_pr703_source_matches_unpadded_cpp_records(self) -> None:
        prompt, records = PROMPTS.make_pr703_source_prompt(1024, "NPU-7391")

        self.assertEqual(records, 1024)
        self.assertIn("Archive record 1: routine mobile inference note, checksum 7919.\n", prompt)
        self.assertNotIn("Archive record 0001:", prompt)
        self.assertIn("Archive record 1024:", prompt)

    def test_pr703_window_roundtrips_through_mllmv2_text(self) -> None:
        tokenizer = _CharacterTokenizer()
        prompt, selected_ids, metadata = PROMPTS.make_pr703_records_window_prompt(
            tokenizer, 512, "NPU-7391"
        )
        source_prompt, _ = PROMPTS.make_pr703_source_prompt(512, "NPU-7391")
        source_ids = PROMPTS.token_ids(tokenizer, source_prompt)
        expected_ids = source_ids[:342] + source_ids[-170:]

        self.assertEqual(selected_ids, expected_ids)
        self.assertEqual(PROMPTS.token_ids(tokenizer, prompt), expected_ids)
        self.assertEqual(metadata["head_tokens"], 342)
        self.assertEqual(metadata["tail_tokens"], 170)
        self.assertTrue(metadata["roundtrip_verified"])

    def test_pr703_record_count_scales_like_cpp(self) -> None:
        self.assertEqual(PROMPTS.make_pr703_source_prompt(2048, "NPU-7391")[1], 1024)
        self.assertEqual(PROMPTS.make_pr703_source_prompt(4096, "NPU-7391")[1], 2048)


if __name__ == "__main__":
    unittest.main()
