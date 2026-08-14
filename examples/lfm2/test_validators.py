# Copyright (c) MLLM Team.
# Licensed under the MIT License.
import json
import unittest
from pathlib import Path

import validate_checkpoint
import validate_converted_model


class Lfm2ValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(__file__).parent
        self.runtime = json.loads((self.directory / "config_2.6B_w4a32_kai.json").read_text())
        self.recipe = json.loads((self.directory / "quant_cfg_2.6B_w4a32_kai.json").read_text())

    def test_official_contract_has_266_tensors(self) -> None:
        validate_checkpoint.validate_config(self.runtime)
        shapes = validate_checkpoint.expected_shapes()
        self.assertEqual(len(shapes), 266)
        self.assertEqual(sum(kind == "full_attention" for kind in validate_checkpoint.LAYER_TYPES), 8)

    def test_quant_recipe_has_exact_tied_output_alias(self) -> None:
        validate_checkpoint.validate_recipe(self.recipe, validate_checkpoint.expected_shapes())
        descriptors = validate_converted_model.expected_descriptors(self.recipe)
        self.assertEqual(len(descriptors), 267)
        self.assertEqual(
            descriptors["model.embed_tokens.weight"][:2],
            (validate_converted_model.FLOAT32, [128000, 2048]),
        )
        self.assertEqual(descriptors["lm_head_out.weight"][0], validate_converted_model.BYTE)

    def test_schedule_drift_fails_closed(self) -> None:
        drifted = dict(self.runtime)
        drifted["layer_types"] = list(self.runtime["layer_types"])
        drifted["layer_types"][0] = "full_attention"
        with self.assertRaisesRegex(AssertionError, "layer_types"):
            validate_checkpoint.validate_config(drifted)


if __name__ == "__main__":
    unittest.main()
