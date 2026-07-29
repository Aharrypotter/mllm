# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import json
import argparse

from .. import convertor
from ..convertor.model_file_v2 import ModelFileV2
from ..quantize.solver import QuantizeSolver
from ..quantize.pipeline import BUILTIN_QUANTIZE_PIPELINE


def main():
    parser = argparse.ArgumentParser(description="MLLM Model Converter")
    parser.add_argument(
        "--input_path", type=str, help="Path to input model file", required=True
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to output model file", required=True
    )
    parser.add_argument("--model_name", type=str, help="Model name", required=True)
    parser.add_argument("--cfg_path", type=str, help="Quantization config file path")
    parser.add_argument(
        "--format",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Output format version (default: v2)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=sorted(BUILTIN_QUANTIZE_PIPELINE),
        help=f"Choose builtin pipeline in {BUILTIN_QUANTIZE_PIPELINE.keys()}",
    )
    parser.add_argument(
        "--include_prefix",
        action="append",
        default=[],
        help=(
            "Only convert parameters whose names start with this prefix. "
            "Repeat the option to keep multiple prefixes."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    if args.format != "v2":
        parser.error("this converter entry point currently supports only --format v2")
    if args.cfg_path is not None and args.pipeline is None:
        parser.error("--cfg_path requires --pipeline")
    if any(not prefix for prefix in args.include_prefix):
        parser.error("--include_prefix values must be non-empty")
    if args.verbose:
        print(f"Converting {args.input_path} to {args.output_path}")
        print(f"Output format: {args.format}")

    # Get params
    params = convertor.load_model(args.input_path, args.include_prefix)
    if args.include_prefix:
        if not params:
            parser.error(
                "--include_prefix did not match any parameters; refusing to write an empty model"
            )
        if args.verbose:
            print(
                f"Kept {len(params)} parameters matching prefixes: "
                + ", ".join(args.include_prefix)
            )

    # Build pipeline
    if args.cfg_path is None and args.pipeline is not None and args.format == "v2":
        cfg = None
        pipeline: QuantizeSolver = BUILTIN_QUANTIZE_PIPELINE[args.pipeline]()
        old_param_size = len(params)
        new_param_size = pipeline.stream_quantize_params_size(cfg, params)
        print(f"Params Num: Before: {old_param_size}, After: {new_param_size}")
        pipeline.stream_quantize(
            cfg,
            params,
            writer=ModelFileV2(
                args.output_path,
                args.model_name,
                "Streaming",
                max_params_descriptor_buffer_num=new_param_size,
            ),
            cast_left_2_fp32=True,
            verbose=args.verbose,
        )
    elif args.cfg_path is None and args.pipeline is None and args.format == "v2":
        cfg = None
        pipeline: QuantizeSolver = BUILTIN_QUANTIZE_PIPELINE["_raw"]()
        old_param_size = len(params)
        new_param_size = pipeline.stream_quantize_params_size(cfg, params)
        print(f"Params Num: Before: {old_param_size}, After: {new_param_size}")
        pipeline.stream_quantize(
            cfg,
            params,
            writer=ModelFileV2(
                args.output_path,
                args.model_name,
                "Streaming",
                max_params_descriptor_buffer_num=new_param_size,
            ),
            cast_left_2_fp32=False,
            verbose=args.verbose,
        )
    elif (
        args.cfg_path is not None and args.pipeline is not None and args.format == "v2"
    ):
        cfg = None
        with open(args.cfg_path) as f:
            cfg = json.load(f)
        pipeline: QuantizeSolver = BUILTIN_QUANTIZE_PIPELINE[args.pipeline]()

        old_param_size = len(params)
        new_param_size = pipeline.stream_quantize_params_size(cfg, params)

        print(f"Params Num: Before: {old_param_size}, After: {new_param_size}")

        pipeline.stream_quantize(
            cfg,
            params,
            writer=ModelFileV2(
                args.output_path,
                args.model_name,
                "Streaming",
                max_params_descriptor_buffer_num=new_param_size,
            ),
            cast_left_2_fp32=True,
            verbose=args.verbose,
        )
    else:
        parser.error("unsupported converter configuration")


if __name__ == "__main__":
    main()
