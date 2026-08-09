# Copyright (c) MLLM Team.
# Licensed under the MIT License.

"""Focused product-contract tests for the Qwen3.5 conversion validators."""

from __future__ import annotations

import copy
import json
import re
import unittest
from collections import Counter
from pathlib import Path

from validate_checkpoint import (
    default_quant_config_path,
    default_runtime_config_path,
    expected_text_shapes,
    expected_vision_shapes,
    model_name_for_size,
    resolve_model_size,
    validate_kai_recipe_contract,
    validate_multimodal_config_contract,
)
from validate_converted_model import (
    BYTE,
    FLOAT32,
    MODEL_HEADER,
    PARAMETER_DESCRIPTOR,
    UINT32_LIMIT,
    _expected_descriptors,
    _validate_descriptor_table,
)


EXAMPLE_DIR = Path(__file__).resolve().parent


def _load_json(name: str) -> dict:
    with (EXAMPLE_DIR / name).open() as input_file:
        return json.load(input_file)


class Qwen35ValidatorTest(unittest.TestCase):
    def test_default_artifacts_resolve_for_both_supported_sizes(self) -> None:
        for model_size in ("0.8B", "4B"):
            with self.subTest(model_size=model_size):
                config = _load_json(f"config_{model_size}_w4a32_kai.json")
                self.assertEqual(
                    resolve_model_size(config["text_config"]),
                    model_size,
                )
                self.assertEqual(
                    default_quant_config_path(model_size),
                    EXAMPLE_DIR / f"quant_cfg_{model_size}_w4a32_kai.json",
                )
                self.assertEqual(
                    default_runtime_config_path(model_size),
                    EXAMPLE_DIR / f"config_{model_size}_w4a32_kai.json",
                )
                self.assertEqual(
                    model_name_for_size(model_size),
                    f"Qwen3.5-{model_size}",
                )

    def test_official_4b_quant_shapes_and_v2_offsets(self) -> None:
        config = _load_json("config_4B_w4a32_kai.json")
        quant_config = _load_json("quant_cfg_4B_w4a32_kai.json")
        text_config = config["text_config"]

        self.assertEqual(resolve_model_size(text_config, "4B"), "4B")
        quantized_names, _ = validate_kai_recipe_contract(
            text_config,
            quant_config,
            config,
        )
        self.assertEqual(len(quantized_names), 249)
        source_shapes = expected_text_shapes(text_config)
        self.assertEqual(
            source_shapes[
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
            ],
            [8192, 2560],
        )
        self.assertEqual(
            source_shapes["model.language_model.layers.3.self_attn.q_proj.weight"],
            [8192, 2560],
        )
        self.assertEqual(
            source_shapes["model.language_model.layers.3.self_attn.k_proj.weight"],
            [1024, 2560],
        )
        self.assertEqual(
            source_shapes["model.language_model.layers.0.linear_attn.in_proj_z.weight"],
            [4096, 2560],
        )
        self.assertEqual(
            source_shapes["model.language_model.layers.0.mlp.gate_proj.weight"],
            [9216, 2560],
        )
        self.assertEqual(
            source_shapes["model.language_model.embed_tokens.weight"],
            [248320, 2560],
        )

        for pattern, entry in quant_config.items():
            matches = [name for name in source_shapes if re.fullmatch(pattern, name)]
            self.assertTrue(matches, msg=f"unmatched quantization pattern: {pattern}")
            for name in matches:
                self.assertEqual(entry["hints"]["shape"], source_shapes[name])

        expected = _expected_descriptors(text_config, quant_config)
        data_start = MODEL_HEADER.size + len(expected) * PARAMETER_DESCRIPTOR.size
        next_offset = data_start
        actual: dict[str, tuple[int, list[int], int, int]] = {}
        for name, (dtype, shape, size) in expected.items():
            actual[name] = (dtype, shape, size, next_offset)
            next_offset += size

        self.assertEqual(len(expected), 427)
        self.assertEqual(next_offset, 4_923_030_836)
        self.assertGreater(next_offset, UINT32_LIMIT)
        self.assertTrue(
            any(offset >= UINT32_LIMIT for _, _, _, offset in actual.values())
        )

        dtype_counts = _validate_descriptor_table(
            actual,
            expected,
            data_start,
            next_offset,
        )
        self.assertEqual(dtype_counts[BYTE], 249)
        self.assertEqual(dtype_counts[FLOAT32], 178)

    def test_official_08b_multimodal_recipe_and_descriptors(self) -> None:
        config = _load_json("config_0.8B_multimodal_w4a32_kai.json")
        quant_config = _load_json("quant_cfg_0.8B_multimodal_w4a32_kai.json")

        checkpoint_config = copy.deepcopy(config)
        checkpoint_config["vision_config"].update(
            {"initializer_range": 0.02, "model_type": "qwen3_5"}
        )
        validate_multimodal_config_contract(checkpoint_config, config)
        quantized_names, _ = validate_kai_recipe_contract(
            checkpoint_config["text_config"],
            quant_config,
            config,
            checkpoint_config["vision_config"],
        )
        self.assertEqual(len(quantized_names), 287)
        self.assertEqual(len(expected_vision_shapes(config["vision_config"])), 153)

        expected = _expected_descriptors(config, quant_config)
        dtype_counts = Counter(dtype for dtype, _, _ in expected.values())
        self.assertEqual(len(expected), 424)
        self.assertEqual(dtype_counts[BYTE], 237)
        self.assertEqual(dtype_counts[FLOAT32], 187)
        self.assertEqual(
            sum(name.startswith("model.visual.") for name in expected),
            103,
        )

    def test_multimodal_recipe_rejects_text_runtime_and_contract_drift(self) -> None:
        config = _load_json("config_0.8B_multimodal_w4a32_kai.json")
        text_runtime = _load_json("config_0.8B_w4a32_kai.json")
        quant_config = _load_json("quant_cfg_0.8B_multimodal_w4a32_kai.json")

        with self.assertRaisesRegex(AssertionError, "both include vision"):
            validate_kai_recipe_contract(
                config["text_config"],
                quant_config,
                text_runtime,
                config["vision_config"],
            )

        mutated = copy.deepcopy(config)
        mutated["image_max_pixels"] = 1024 * 1024
        with self.assertRaisesRegex(AssertionError, "image_max_pixels"):
            validate_multimodal_config_contract(config, mutated)

    def test_variant_resolution_rejects_runtime_semantic_mismatches(self) -> None:
        text_config = _load_json("config_4B_w4a32_kai.json")["text_config"]
        mutations = (
            (("hidden_act",), "gelu"),
            (("rms_norm_eps",), 1e-05),
            (("eos_token_id",), 0),
            (("tie_word_embeddings",), False),
            (("mamba_ssm_dtype",), "bfloat16"),
            (("rope_parameters", "rope_theta"), 10000),
            (("rope_parameters", "partial_rotary_factor"), 1.0),
        )

        for path, value in mutations:
            with self.subTest(path=path, value=value):
                mutated = copy.deepcopy(text_config)
                target = mutated
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaisesRegex(
                    AssertionError,
                    "Checkpoint does not match Qwen3.5-4B",
                ):
                    resolve_model_size(mutated, "4B")

    def test_kai_recipe_contract_rejects_runtime_incompatible_hints(self) -> None:
        runtime_config = _load_json("config_4B_w4a32_kai.json")
        quant_config = _load_json("quant_cfg_4B_w4a32_kai.json")
        text_config = runtime_config["text_config"]
        first_pattern = next(iter(quant_config))

        hint_mutations = (
            ("quant_method", "not_kai"),
            ("kai_matmul_triplet", "wrong_triplet"),
            ("kai_matmul_layout", "wrong_layout"),
            ("kai_matmul_tile_cfg", "wrong_tile"),
            ("replace", False),
        )
        for field, value in hint_mutations:
            with self.subTest(field=field, value=value):
                mutated_quant_config = copy.deepcopy(quant_config)
                mutated_quant_config[first_pattern]["hints"][field] = value
                with self.assertRaises(AssertionError):
                    validate_kai_recipe_contract(
                        text_config,
                        mutated_quant_config,
                        runtime_config,
                    )

        mutated_runtime_config = copy.deepcopy(runtime_config)
        mutated_runtime_config["linear_impl_type"] = "WrongLinear"
        with self.assertRaisesRegex(
            AssertionError,
            "linear_impl_type mismatch",
        ):
            validate_kai_recipe_contract(
                text_config,
                quant_config,
                mutated_runtime_config,
            )

        nested_override = copy.deepcopy(runtime_config)
        nested_override["text_config"]["linear_impl_type"] = "Default"
        with self.assertRaisesRegex(
            AssertionError,
            "linear_impl_type mismatch",
        ):
            validate_kai_recipe_contract(
                text_config,
                quant_config,
                nested_override,
            )

        nested_only = copy.deepcopy(runtime_config)
        expected_linear_impl = nested_only.pop("linear_impl_type")
        nested_only["text_config"]["linear_impl_type"] = expected_linear_impl
        quantized_names, _ = validate_kai_recipe_contract(
            text_config,
            quant_config,
            nested_only,
        )
        self.assertEqual(len(quantized_names), 249)

    def test_kai_recipe_contract_rejects_missing_linear_coverage(self) -> None:
        runtime_config = _load_json("config_4B_w4a32_kai.json")
        quant_config = _load_json("quant_cfg_4B_w4a32_kai.json")
        text_config = runtime_config["text_config"]

        dropped_pattern = next(
            pattern for pattern in quant_config if "down_proj" in pattern
        )
        mutated_quant_config = copy.deepcopy(quant_config)
        del mutated_quant_config[dropped_pattern]
        with self.assertRaisesRegex(
            AssertionError,
            "KAI quantization coverage mismatch",
        ) as raised:
            validate_kai_recipe_contract(
                text_config,
                mutated_quant_config,
                runtime_config,
            )
        self.assertIn("down_proj", str(raised.exception))

    def test_kai_recipe_contract_rejects_non_linear_coverage(self) -> None:
        runtime_config = _load_json("config_4B_w4a32_kai.json")
        quant_config = _load_json("quant_cfg_4B_w4a32_kai.json")
        text_config = runtime_config["text_config"]
        source_shapes = expected_text_shapes(text_config)
        non_linear_name = "model.language_model.norm.weight"

        template_pattern = next(
            pattern for pattern in quant_config if "q_proj" in pattern
        )
        mutated_quant_config = copy.deepcopy(quant_config)
        extra_entry = copy.deepcopy(mutated_quant_config[template_pattern])
        extra_entry["hints"]["shape"] = source_shapes[non_linear_name]
        mutated_quant_config[f"^{re.escape(non_linear_name)}$"] = extra_entry
        with self.assertRaisesRegex(
            AssertionError,
            "KAI quantization coverage mismatch",
        ) as raised:
            validate_kai_recipe_contract(
                text_config,
                mutated_quant_config,
                runtime_config,
            )
        self.assertIn(non_linear_name, str(raised.exception))

    def test_kai_recipe_contract_rejects_invalid_tied_embedding_entry(self) -> None:
        runtime_config = _load_json("config_4B_w4a32_kai.json")
        quant_config = _load_json("quant_cfg_4B_w4a32_kai.json")
        text_config = runtime_config["text_config"]

        tied_pattern = next(
            pattern for pattern in quant_config if "embed_tokens" in pattern
        )
        hint_mutations = (
            ("replace", True),
            ("rename", "wrong_name.weight"),
            ("rename", None),
        )
        for field, value in hint_mutations:
            with self.subTest(field=field, value=value):
                mutated_quant_config = copy.deepcopy(quant_config)
                mutated_quant_config[tied_pattern]["hints"][field] = value
                with self.assertRaisesRegex(
                    AssertionError,
                    "Tied embedding KAI entry must retain embed_tokens.weight",
                ):
                    validate_kai_recipe_contract(
                        text_config,
                        mutated_quant_config,
                        runtime_config,
                    )

        mutated_quant_config = copy.deepcopy(quant_config)
        del mutated_quant_config[tied_pattern]["hints"]["rename"]
        with self.assertRaisesRegex(
            AssertionError,
            "Tied embedding KAI entry must retain embed_tokens.weight",
        ):
            validate_kai_recipe_contract(
                text_config,
                mutated_quant_config,
                runtime_config,
            )

    def test_kai_recipe_contract_rejects_malformed_runtime_text_config(self) -> None:
        runtime_config = _load_json("config_4B_w4a32_kai.json")
        quant_config = _load_json("quant_cfg_4B_w4a32_kai.json")
        text_config = runtime_config["text_config"]

        for mutation in ("missing", None, "not-an-object"):
            with self.subTest(mutation=mutation):
                mutated_runtime_config = copy.deepcopy(runtime_config)
                if mutation == "missing":
                    del mutated_runtime_config["text_config"]
                else:
                    mutated_runtime_config["text_config"] = mutation
                with self.assertRaisesRegex(
                    AssertionError,
                    "KAI runtime config must declare an object text_config",
                ):
                    validate_kai_recipe_contract(
                        text_config,
                        quant_config,
                        mutated_runtime_config,
                    )


if __name__ == "__main__":
    unittest.main()
