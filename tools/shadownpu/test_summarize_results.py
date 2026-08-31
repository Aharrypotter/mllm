#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("summarize_results.py")
SPEC = importlib.util.spec_from_file_location("summarize_results", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SummarizeResultsTest(unittest.TestCase):
    def write_log(
        self,
        root: Path,
        mode: str,
        run: int,
        token_ids: str = "11,12",
        retrieval_exact: str = "1",
    ) -> Path:
        path = root / f"{mode}-p1024-run{run}.log"
        if mode == "dense":
            result = (
                "SHADOWNPU_HARNESS_CONFIG mode=dense attention_mode=dense prompt_tokens=1024 repetition=1\n"
                "SHADOWNPU_RESULT attention_mode=dense prompt_tokens=1024 prefill_ms=100.000 generated_tokens=2 "
                "actual_retention=1.00000000 hmx_select_ms=0.000 sparse_total_ms=0.000 prefill_chunks=2 "
                "qnn_tile_sequence=256 qnn_graph_calls=112 qnn_tile_calls=224 qnn_execute_ms=80.000\n"
            )
        else:
            result = (
                "SHADOWNPU_HARNESS_CONFIG mode=sparse attention_mode=hmx-topk prompt_tokens=1024 repetition=1\n"
                "FastRPC redirect: CDSP domain 3/session 0 -> domain 3/session 1\n"
                "SHADOWNPU_RESULT attention_mode=hmx-topk prompt_tokens=1024 prefill_ms=50.000 generated_tokens=2 "
                "actual_retention=0.20000000 hmx_select_ms=10.000 sparse_total_ms=40.000 prefill_chunks=2 "
                "qnn_tile_sequence=256 qnn_graph_calls=112 qnn_tile_calls=224 qnn_execute_ms=30.000\n"
            )
        path.write_text(
            result
            + f"SHADOWNPU_TOKEN_IDS ids={token_ids}\n"
            + f"QUALITY_RESULT expected_code=NPU-7391 extracted_code=NPU-7391 retrieval_exact={retrieval_exact}\n",
            encoding="utf-8",
        )
        return path

    def test_summarizes_matched_dense_sparse_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = MODULE.summarize([self.write_log(root, "dense", 1), self.write_log(root, "sparse", 1)])
        self.assertIn("1024\t100.000\t50.000\t2.000\t40.000\t0.20000000\t1\t1", output)

    def test_rejects_token_mismatch_even_when_both_retrieve_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = self.write_log(root, "dense", 1)
            sparse = self.write_log(root, "sparse", 1, token_ids="11,13")
            with self.assertRaisesRegex(ValueError, "generated token IDs differ"):
                MODULE.summarize([dense, sparse])

    def test_accepts_runtime_true_retrieval_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = self.write_log(root, "dense", 1, retrieval_exact="true")
            parsed = MODULE.parse_run(dense)
        self.assertEqual(parsed.extracted_code, "NPU-7391")

    def test_rejects_failed_strict_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sparse = self.write_log(root, "sparse", 1, retrieval_exact="0")
            with self.assertRaisesRegex(ValueError, "strict access-code retrieval failed"):
                MODULE.parse_run(sparse)

    def test_rejects_missing_qnn_tile_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = self.write_log(root, "dense", 1)
            dense.write_text(dense.read_text(encoding="utf-8").replace("qnn_tile_sequence=256", "qnn_tile_sequence=0"),
                             encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not compiled/executed as M256"):
                MODULE.parse_run(dense)

    def test_accepts_all_dense_fp16_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = self.write_log(root, "dense", 1)
            dense.write_text(
                dense.read_text(encoding="utf-8")
                .replace("attention_mode=dense", "attention_mode=hmx-topk")
                .replace("SHADOWNPU_TOKEN_IDS ids=11,12", "SHADOWNPU_TOKEN_IDS ids=21,22"),
                encoding="utf-8",
            )
            parsed = MODULE.parse_run(dense)
        self.assertEqual(parsed.mode, "dense")
        self.assertEqual(parsed.token_ids, (21, 22))


if __name__ == "__main__":
    unittest.main()
