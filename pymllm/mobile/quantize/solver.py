# Copyright (c) MLLM Team.
# Licensed under the MIT License.
import re
from .quantize_pass import QuantizeBasePass, QuantizePlanPayload
from ..convertor import ModelFileV2
from typing import Dict, List, Any
from ..ffi import MLLM_FIND_NUMPY_AVAILABLE, MLLM_FIND_TORCH_AVAILABLE

if MLLM_FIND_TORCH_AVAILABLE:
    import torch
if MLLM_FIND_NUMPY_AVAILABLE:
    import numpy as np


class QuantizeSolver:
    def __init__(self):
        self.passes: List[QuantizeBasePass] = []

    def register_pass(self, pass_: QuantizeBasePass):
        self.passes.append(pass_)

    def _stream_quantize_write_v2(self, tensor_dict: Dict, writer: ModelFileV2) -> bool:
        pass

    def _build_quantize_plan(self, quant_cfg, tensor_dict: Dict) -> Dict:
        if quant_cfg is None:
            quant_cfg = {}

        param_groups: Dict[str, List[Any]] = {}
        claimed_inputs: Dict[str, str] = {}
        for k, v in quant_cfg.items():
            sub_group: Dict[str, QuantizePlanPayload] = {}
            hints = v["hints"]
            pattern = re.compile(k)
            for pk in sorted(tensor_dict):
                pv = tensor_dict[pk]
                if pattern.fullmatch(pk) is not None:
                    if pk in claimed_inputs:
                        raise ValueError(
                            f"Quantization patterns {claimed_inputs[pk]!r} and {k!r} "
                            f"both match {pk!r}"
                        )
                    claimed_inputs[pk] = k
                    # pk is model.linear_0.weight or model.linear_0.bias
                    # layer_name is model.linear_0
                    layer_name, _ = pk.rsplit(".", 1)
                    if layer_name not in sub_group:
                        sub_group[layer_name] = QuantizePlanPayload()
                    sub_group[layer_name].inputs_num += 1
                    sub_group[layer_name].inputs_dict.update({pk: pv})
            param_groups.update({k: [hints, sub_group, {}]})

        # Select exactly one pass and prepare its input/output contract.
        for group_regex in param_groups.keys():
            sub_group = param_groups[group_regex]
            for payload_name, payload in sub_group[1].items():
                matching_passes = [
                    pass_
                    for pass_ in self.passes
                    if pass_.match(sub_group[0], payload.inputs_dict)
                ]
                if not matching_passes:
                    raise ValueError(
                        f"No registered quantization pass matched {payload_name!r} "
                        f"for pattern {group_regex!r}"
                    )
                if len(matching_passes) > 1:
                    raise ValueError(
                        f"Multiple quantization passes matched {payload_name!r} "
                        f"for pattern {group_regex!r}"
                    )
                selected_pass = matching_passes[0]
                prepared_payload = selected_pass.prepare(
                    sub_group[0],
                    payload.inputs_dict,
                )
                if (
                    prepared_payload.inputs_num != len(prepared_payload.inputs_dict)
                    or prepared_payload.outputs_num
                    != len(prepared_payload.outputs_dict)
                    or any(
                        not isinstance(name, str) or not name
                        for name in prepared_payload.outputs_dict
                    )
                ):
                    raise ValueError(
                        f"Quantization pass produced an invalid plan for {payload_name!r}"
                    )
                sub_group[1][payload_name] = prepared_payload
                sub_group[2][payload_name] = selected_pass

        planned_names = set(tensor_dict)
        for sub_group in param_groups.values():
            for payload in sub_group[1].values():
                if sub_group[0]["replace"]:
                    planned_names.difference_update(payload.inputs_dict)
        for sub_group in param_groups.values():
            for payload in sub_group[1].values():
                for output_name in payload.outputs_dict:
                    if output_name in planned_names:
                        raise ValueError(
                            f"Quantization output {output_name!r} collides with "
                            "another model parameter"
                        )
                    planned_names.add(output_name)

        return param_groups

    def stream_quantize_params_size(
        self, quant_cfg, tensor_dict: Dict, **kwargs
    ) -> int:
        param_groups = self._build_quantize_plan(quant_cfg, tensor_dict)

        # Update params nums
        aux = set(tensor_dict.keys())

        for group_regex in param_groups.keys():
            sub_group = param_groups[group_regex]
            for payload_name, payload in sub_group[1].items():
                if sub_group[0]["replace"]:
                    for k in payload.inputs_dict.keys():
                        aux.remove(k)
                    for k in payload.outputs_dict.keys():
                        aux.add(k)
                else:
                    for k in payload.outputs_dict.keys():
                        aux.add(k)

        return len(aux)

    def stream_quantize(
        self, quant_cfg, tensor_dict: Dict, writer: ModelFileV2, **kwargs
    ) -> bool:
        if not isinstance(writer, ModelFileV2):
            raise NotImplementedError(
                "stream_quantize only support type: ModelFileV2 currently."
            )

        param_groups = self._build_quantize_plan(quant_cfg, tensor_dict)

        # Show Planned Info
        verbose = kwargs.get("verbose", False)
        if verbose:
            print("Planned Quantized Info:")
            for group_regex in param_groups.keys():
                print(f"{group_regex}:")
                sub_group = param_groups[group_regex]
                for payload_name, payload in sub_group[1].items():
                    print(" " * 4 + payload_name + ":")
                    print(" " * 8 + f"inputs num: {payload.inputs_num}")
                    print(" " * 8 + f"outputs num: {payload.outputs_num}")
                    print(" " * 8 + "params before quantization:")
                    for k in payload.inputs_dict.keys():
                        print(" " * 12 + k)
                    print(" " * 8 + "params after quantization:")
                    for k in payload.outputs_dict.keys():
                        print(" " * 12 + k)

        # Processing
        left_name = set(tensor_dict.keys())
        for group_regex in param_groups.keys():
            sub_group = param_groups[group_regex]
            for payload_name, payload in sub_group[1].items():
                selected_pass = sub_group[2][payload_name]
                prepared_payload = selected_pass.run(
                    sub_group[0],
                    payload.inputs_dict,
                )
                if set(prepared_payload) != set(payload.outputs_dict):
                    raise ValueError(
                        f"Quantization pass outputs changed after planning for {payload_name!r}"
                    )
                if sub_group[0]["replace"]:
                    for pk in payload.inputs_dict.keys():
                        left_name.remove(pk)
                        tensor_dict[pk] = None
                    for pk in sorted(prepared_payload):
                        pv = prepared_payload[pk]
                        if verbose:
                            print(pk)
                        writer.streaming_write(pk, pv)
                else:
                    for pk in sorted(prepared_payload):
                        pv = prepared_payload[pk]
                        writer.streaming_write(pk, pv)
                payload.inputs_dict = None
                payload.outputs_dict = None

        cast_left_2_fp32 = kwargs.get("cast_left_2_fp32", False)

        for k in sorted(left_name):
            if verbose:
                print(k)
            ttt = tensor_dict[k]
            if cast_left_2_fp32:
                if MLLM_FIND_TORCH_AVAILABLE and isinstance(ttt, torch.Tensor):
                    ttt = ttt.to(torch.float32)
                elif MLLM_FIND_NUMPY_AVAILABLE and isinstance(ttt, np.ndarray):
                    ttt = ttt.astype(np.float32, copy=False)
            writer.streaming_write(k, ttt)

        writer.finalize()
        return True
