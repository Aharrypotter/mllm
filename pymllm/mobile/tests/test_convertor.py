# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import pytest
import torch

from pymllm.mobile.convertor import load_model
from pymllm.mobile.convertor.model_file_v2 import (
    ModelFileV2Descriptor,
    ModelFileV2ParamsDescriptor,
)


def test_load_model_filters_torch_checkpoint_by_prefix(tmp_path):
    checkpoint_path = tmp_path / "model.bin"
    torch.save(
        {
            "model.layers.0.weight": torch.ones(1),
            "visual.blocks.0.weight": torch.zeros(1),
        },
        checkpoint_path,
    )

    state_dict = load_model(checkpoint_path, include_prefixes=("model.",))

    assert list(state_dict) == ["model.layers.0.weight"]


def test_model_file_v2_validates_utf8_name_capacity():
    with pytest.raises(ValueError, match="fewer than 512 UTF-8 bytes"):
        ModelFileV2Descriptor(model_name="模" * 171, num_params=0)


def test_model_file_v2_rejects_shape_above_descriptor_rank():
    with pytest.raises(ValueError, match="rank exceeds"):
        ModelFileV2ParamsDescriptor(
            param_id=0,
            param_type=0,
            param_size=1,
            param_offset=0,
            shape=[1] * 17,
            name="weight",
        )
