# Copyright (c) MLLM Team.
# Licensed under the MIT License.
from typing import Dict
import tvm_ffi
from ...ffi import (
    from_numpy,
    from_torch,
    Tensor,
    MLLM_FIND_NUMPY_AVAILABLE,
    MLLM_FIND_TORCH_AVAILABLE,
)
from ..quantize_pass import QuantizeBasePass, QuantizePlanPayload

if MLLM_FIND_TORCH_AVAILABLE:
    import torch
if MLLM_FIND_NUMPY_AVAILABLE:
    import numpy as np


class W4A32KAIQuantizePass(QuantizeBasePass):
    def __init__(self):
        super().__init__()

    def prepare(
        self, quantize_config, tensor_dict: Dict, **kwargs
    ) -> QuantizePlanPayload:
        replace: bool = quantize_config["replace"]
        rename: str = quantize_config["rename"] if not replace else None

        if len(tensor_dict) not in (1, 2):
            raise ValueError("KAI W4A32 expects one weight and an optional bias")
        weight_keys = [key for key in tensor_dict if key.endswith(".weight")]
        bias_keys = [key for key in tensor_dict if key.endswith(".bias")]
        if len(weight_keys) != 1:
            raise ValueError("KAI W4A32 expects exactly one weight tensor")
        if len(bias_keys) > 1 or set(weight_keys + bias_keys) != set(tensor_dict):
            raise ValueError("KAI W4A32 accepts only one weight and an optional bias")
        if not replace and not rename:
            raise ValueError("KAI W4A32 requires a non-empty rename when replace=false")

        actual_shape = list(tensor_dict[weight_keys[0]].shape)
        if len(actual_shape) != 2 or any(dimension <= 0 for dimension in actual_shape):
            raise ValueError("KAI W4A32 weight must be a non-empty rank-2 tensor")
        expected_shape = quantize_config.get("shape")
        if expected_shape is not None and list(expected_shape) != actual_shape:
            raise ValueError(
                f"KAI W4A32 expected weight shape {expected_shape}, got {actual_shape}"
            )
        if bias_keys and list(tensor_dict[bias_keys[0]].shape) != [actual_shape[0]]:
            raise ValueError(
                "KAI W4A32 bias shape must match the weight output channels"
            )

        ret = QuantizePlanPayload()
        ret.inputs_num = len(tensor_dict)
        ret.inputs_dict = tensor_dict

        if replace:
            ret.outputs_num = 1
            ret.outputs_dict = {weight_keys[0]: None}
        else:
            ret.outputs_num = 1
            ret.outputs_dict = {rename: None}
        return ret

    def match(self, quantize_config, tensor_dict: Dict, **kwargs) -> bool:
        if quantize_config.get("quant_method") != "kai":
            return False
        if quantize_config.get("kai_matmul_triplet") != "f32_qai8dxp_qsi4c32p":
            return False
        if quantize_config.get("kai_matmul_layout") != "mxk_nxk":
            return False
        return True

    def run(self, quantize_config, tensor_dict: Dict, **kwargs) -> Dict:
        normalized_tensor_dict = {}

        # Preprocess
        for k, v in tensor_dict.items():
            if isinstance(v, Tensor):
                normalized_tensor_dict.update({k: v})
            elif MLLM_FIND_TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                tmp_v: Tensor = from_torch(v.to(torch.float32))
                tmp_v.set_name(k)
                normalized_tensor_dict.update({k: tmp_v})
            elif MLLM_FIND_NUMPY_AVAILABLE and isinstance(v, np.ndarray):
                tmp_v: Tensor = from_numpy(v.astype(np.float32, copy=False))
                tmp_v.set_name(k)
                normalized_tensor_dict.update({k: tmp_v})
            else:
                raise TypeError(
                    f"KAI W4A32 does not support tensor type {type(v).__name__} for {k}"
                )

        # Processing
        tile_cfg_name = quantize_config["kai_matmul_tile_cfg"]
        weight: Tensor = Tensor()
        bias: Tensor = Tensor()
        weight_count = 0
        bias_count = 0
        for key in normalized_tensor_dict:
            if key.endswith(".weight"):
                weight = normalized_tensor_dict[key]
                weight_count += 1
            elif key.endswith(".bias"):
                bias = normalized_tensor_dict[key]
                bias_count += 1
        if weight_count != 1 or bias_count > 1:
            raise ValueError(
                "KAI W4A32 expects exactly one weight and at most one bias"
            )
        weight: Tensor = tvm_ffi.get_global_func(
            "mllm.quantize_pack.KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk"
        )(tile_cfg_name, weight, bias)

        replace: bool = quantize_config["replace"]
        rename: str = quantize_config["rename"] if not replace else None

        if replace:
            return {weight.name: weight}
        else:
            weight.set_name(rename)
            return {rename: weight}
