# ShadowNPU sparse attention on mllmV2

This port targets Qwen2.5-1.5B prefill on the current mllmV2 runtime. Decode
attention remains dense. The experiment contract is:

```text
AE head profile -> static scale calibration -> model-bound HMX bank
                -> static or measured per-head Q/K scales
                -> HMX INT8 QK -> CPU exact INT8 Top-k
                -> sparse softmax/value aggregation
```

The bank retains static per-head INT8 score requantization. Runtime Q/K scales
may use the selected static bucket or live per-head measurements; runtime output
scaling remains disabled. `TOPK_OVERSAMPLE=1`; rerank, outlier fallback, and
DSP INT32 Top-k are disabled in the accepted direct path.

## Evidence and device boundary

The three environments have non-interchangeable roles:

| Environment | Role | Current evidence |
|---|---|---|
| macOS arm64 host | Source integration, serialization, semantic and tool tests | Focused host gates passed for the source-v58 baseline; source-v65 was separately sealed after the dynamic-Q/K direct-path change |
| H20 server | Qualcomm Hexagon/QNN/Android cross-build and model-bound HMX v79 bank build | Source-v65/runtime-v92 Android/QNN build and artifact audit passed |
| OnePlus 13T (`PKX110`, SM8750) | The accepted ARM/Qualcomm NPU correctness target | Static direct execution passed at 1024 tokens; dynamic per-head Q/K plus static requantization passed at 2048 tokens in both ordinary and direct-three-stage execution. The performance path is currently rejected because the corrected sparse run remains slower than dense |

The Xiaomi 14 results preserved in PR #703 are historical source evidence.
They are not current mllmV2 or OnePlus 13T performance evidence.

The current correctness result is bound to these identities:

```text
remote/main base             50ad5a9b6fbea742e38b5b31776c187e50319c8e
source-v65 manifest          334167d213de078fc82114821852377c9f538aaa5271b1b06d2399665813c1da
source-v65 HMX selector      e72c50358967c8c545539867ad31c78de74270d7b597084d7533ba46f9c6423f
runtime-v92 evidence         b31c25c209e2e3257592c04ff7bdc55e767f56930e50088412e5732272f7c0f0
QNN context                  bd073ca2f8c70164233a1b3d7b38bc90048371dc10452fc2f140bfe0f3afe3fd
OnePlus 1024 direct evidence 01e6e1721b924d0d211060d68253189271faa69f09eee2edd2383d76343f3719
OnePlus 2048 direct evidence a7f526f888ac2cfd1951d2619977264e849a598039da9784ac9fab8951d57172
```

The accepted source-v64 direct correctness run used the packaged PR retain-0.2
profile, the exact 1024-token records input, and the cached M256 QNN context. It
produced token IDs `45,6325,12,22,18,24,16,151645`, exact retrieval of
`NPU-7391`, 225 finite logit summaries, 112 QNN graph calls, 224 QNN tile calls,
624 audited bucket decisions, 52 complete nine-bucket catalogs, and 52 valid
direct-three-stage timelines. The native crash count remained 31 before and
after. This is a correctness result; its timing is diagnostic rather than a
formal comparison with PR #703 or a dense baseline.

At 2048 tokens, the static-bucket direct path diverged from the dense token
oracle. Dense and exact Float32-QK sparse controls matched, while exact
HMX-versus-F32 Top-k recall localized the mismatch to static selector
quantization. Source-v65 permits live per-head Q/K scales in the fused H12
operator group while preserving the selected static per-head score-requant
scale and invalidating a packed K cache when its bound scale changes. The
runtime-v92 direct gate then reproduced the dense token IDs
`45,6325,12,22,18,24,16,151645`, exact `NPU-7391` retrieval, 225 finite logit
summaries, 224 QNN graph calls, 448 QNN tile calls, 1,248 bucket decisions, 104
complete catalogs, and 104 valid direct-stage timelines. Relevant QNN/FastRPC
error lines were zero and native crash counts remained 31 before and after.

That run reported 95,469.344 ms prefill, versus 55,538.968 ms for the frozen
dense 2048 control. The 24,887.541 ms selector/sparse interval is diagnostic,
not a formal A/B result. The candidate is not promoted to a matched timing
campaign while this product-screen boundary remains slower than dense.

The packaged profile at
`examples/profiles/shadownpu/qwen2-1.5B-retain-0.2.txt` is byte-identical to
PR #703 (SHA-256
`0425ffef895df428ff70cd013ac108f194831f9d0d77860f4a9ab44f21d29b32`).
It has average retention `0.20000000`, 24 dense heads, and dense layers `0-1`.
Do not describe a run with the qualification profile as a reproduction of the
packaged PR retention experiment.

The packaged Qwen2.5-1.5B contract is 28 layers, 12 query heads, head
dimension 128, logical HMX M512, and maximum N4160. The runner processes the
prompt as exact M512 prefill chunks. Each token-wise QNN subgraph is traced at
M256 and the mllmV2 QNN backend executes it twice through persistent shared
scratch buffers for each logical M512 chunk. The CPU attention/HMX stage still
sees the complete M512 query chunk and the accumulated KV cache. A Qwen1.5 checkpoint needs its own
config, context, profile, scale calibration, and operator bank; it must not
reuse the Qwen2.5 artifacts.

The tiling contract is explicit in the executable:

```text
--prefill-chunk 512 --qnn-sequence-tile 256
```

It no longer relies on PR #703's `MLLM_QNN_SEQUENCE_TILE` or
`MLLM_SKIP_QNN_CONTEXT_SAVE` environment variables. A live M256 build keeps
hundreds of finalized graphs resident and can exhaust HTP persistent graph
memory before the last Qwen layer. Generate the context explicitly in QNN HTP
prepare-only mode, seal it as a model/toolchain/device-bound artifact, and load
that immutable context for every functional or timed run:

```bash
mllm-qwen-npu ... \
  --attention-mode hmx-topk \
  --prefill-chunk 512 \
  --qnn-sequence-tile 256 \
  --prepare-qnn-context /new/path/qnn_context_m256.bin

mllm-qwen-npu ... \
  --attention-mode hmx-topk \
  --prefill-chunk 512 \
  --qnn-sequence-tile 256 \
  --qnn-context /sealed/path/qnn_context_m256.bin
```

Generation refuses to overwrite an existing path; HMX execution refuses to
fall back to a live build. A loaded finalized graph is bound by name but never
rebuilt. Any boundary whose rank, dtype, contiguity, or single expanded token
dimension does not match the cached M256 graph fails closed.

The QNN backend registers both halves of `LLaMAPackage` before graph creation.
By default it expects the ARM reference package as
`libQnnLLaMAPackage_CPU.so` and the Hexagon package as
`libQnnLLaMAPackage_HTP.so`. Device packaging may override their exact paths
with `MLLM_QNN_CPU_OP_PACKAGE` and `MLLM_QNN_HTP_OP_PACKAGE`; the two files must
not be confused even when the build tree originally emitted the same basename
under architecture-specific directories.

## Runtime paths

Two first-class mllmV2 operators are available:

- `PatternSparseAttention`: structured, random, or HMX-selected QK/softmax/PV
  over BHSD Q/K/V tensors. HMX mode passes F32 Q/K/V plus a separate fourth
  F16 selector-key input. Selected-index attention math continues to use the
  canonical F32 K/V cache; the F16 staging tensor is not reused for attention
  logits, dense prefill, or CPU decode.
- `SparseSoftmaxValue`: exact stable Top-k selection over precomputed F32 QK
  logits followed by softmax and F32/F16 value aggregation.

The Qwen runner exposes `--attention-mode f32-topk` as a correctness oracle
for the selection boundary. It loads the same layer-major head-retention
profile and the same M512/M256 QNN context contract as `hmx-topk`, but computes
F32 QK logits and exact CPU Top-k before sparse softmax/value aggregation. If
`f32-topk` passes while `hmx-topk` fails on otherwise identical inputs, the
retention schedule and sparse aggregation are separated from the INT8 HMX
selection approximation. `f32-topk` materializes dense logits and is not a
performance candidate.

`StaticCache` returns the live sequence as a logical BHSD view whose head
stride can still reflect the full cache capacity. A dtype conversion or other
linear consumer must not read that view as compact storage. The Qwen HMX path
therefore materializes the logical selector-key view with `contiguous()` before
converting it to F16. The regression test uses a deliberately capacity-strided
cache slice and verifies that every repeated GQA head survives staging.

Qwen configuration accepts an optional `shadownpu` object. The executable
overrides any embedded sparse mode so the dense baseline cannot accidentally
inherit sparse settings:

```json
{
  "shadownpu": {
    "sparsity": 0.8,
    "pattern_sparse": true,
    "hmx_selector": true,
    "keep_dense_edge_layers": true,
    "local_ratio": 0.5,
    "prefix_tokens": 0,
    "dense_tokens": 0,
    "pack_reserve_tokens": 4160
  }
}
```

The layer-major head-retention profile must contain exactly
`num_hidden_layers * num_attention_heads` finite values in `(0, 1]`.

## Host verification

From a configured host build:

```bash
./bin/Mllm-Test-Nn-PatternSparseAttention
./bin/Mllm-Test-CPUKernel \
  --gtest_filter='INT8TopKKernelTest.*:PatternSparseAttentionKernelTest.*:HMXPipelineSchedulerTest.*'

python3 tools/head_sparsity_profiler/test_head_sparsity_profiler.py
python3 tools/hmx_scale_calibrator/test_hmx_scale_calibrator.py
python3 tools/pipeline_latency_profiler/test_pipeline_latency_profiler.py
python3 tools/shadownpu/test_summarize_results.py
```

Off Android, an HMX request must fail closed. A host-only pass does not prove
Hexagon compilation, FastRPC loading, QNN/HMX coexistence, model quality, or
device performance.

## H20 build gate

Use the H20-hosted Qualcomm toolchain to build the Android/QNN runtime,
`mllm-qwen-npu`, the FastRPC redirect library, and a Qwen2.5-specific HMX v79
operator bank. The bank build must bind its source SHA, Hexagon SDK/tool
versions, architecture, model ID, H/M/K/N shape, calibrated scale profile,
manifest, and file hashes. Do not build or reuse a v75/Xiaomi bank for the
OnePlus SM8750 qualification.

The standalone bank entrypoint is:

```bash
HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK \
HMX_OPERATOR_HEADS=12 \
HMX_OPERATOR_M_VALUES=512 \
HMX_OPERATOR_N_VALUES=4160 \
tools/hmx_int8_matmul/scripts/build_operator_bank.sh v79 /path/to/output-bank
```

Exact toolchain and output paths must come from the live H20 profile; the
repository does not infer them.

QNN context generation is a separate OnePlus 13T gate because prepare-only
finalization still invokes the device HTP compiler. H20 cannot substitute for
that device step. Bind the resulting context hash to the source snapshot, QNN
runtime build ID, NPU model hash, op-package hashes, trace sequence 256, and the
OnePlus device identity before reuse.

The retention profile belongs to the CPU `QwenAttentionMatmul` path. Only the
projection and output/MLP modules are moved to QNN; the CPU attention input and
output shapes remain unchanged when retention values change. A profile change
therefore does not automatically require a new QNN context, but reuse needs a
fresh compatibility receipt: source/static graph-boundary proof, the exact new
profile marker, changed actual-retention counters, unchanged QNN graph/tile
counts, strict product quality, and no new native crash. Generate a fresh
context if any graph input/output shape, dtype, custom QNN operation, or that
compatibility gate changes or fails.

## OnePlus 13T qualification gate

After separately provisioning immutable, hash-verified artifacts in the
Termux environment, first run dense, F32 Top-k, and HMX Top-k correctness with
the same model, prompt, profile, compatible cached context, and runtime bank.
The source-v64/runtime-v91 result passed this sequence for the packaged PR
retain-0.2 profile with the exact 1024-token records input. The source-v65/
runtime-v92 result additionally passed the corrected dynamic-Q/K direct path
at 2048 tokens. Preserve the frozen source, runtime, context, profile,
prompt-token hash, HMX bank, and device receipts when extending this boundary.
Run a fresh increasing-length product screen and stop at the first token,
retrieval, or performance rejection. Only after every boundary is stable and
the candidate is not slower should an SSH-based matched A/B harness be run:

```bash
DEVICE_KEY=/path/to/oneplus-ssh-key \
DEVICE_HOST=oneplus-pkx110.taild842a2.ts.net \
DEVICE_ROOT=/data/data/com.termux/files/home/mllm-shadownpu-v2 \
QNN_CONTEXT=/data/data/com.termux/files/home/mllm-shadownpu-v2/run/qnn_context_m256.bin \
REPETITIONS=3 \
tools/shadownpu/run_qwen_ab.sh
```

The harness refuses a device other than OnePlus `PKX110`/`arm64-v8a`, checks
all required paths and the H12/M512/K128/N4160 manifest row, alternates run
order, binds tokenized prompt length, requires FastRPC redirect evidence, and
rejects dense/sparse generated-token mismatches. The summarizer also requires
`prefill_chunks = prompt_tokens / 512`, QNN tile size 256, and exactly two QNN
executions per logical graph call. It does not transfer files,
reboot the phone, alter clocks or thermal policy, or delete remote data.

## Current overlap status

The mllmV2 `PatternSparseAttention` operator now connects both whole-head and
row-range sparse callbacks to the HMX selector. With the Qwen2 1.5B H12 bank,
the validated direct path uses `MLLM_HMX_PIPELINE_GROUP_HEADS=12`, CPU-packed
per-head scales, one all-head long RPC, two cooperative Top-k workers, and the
`direct-three-stage` executor. Static Q/K bucket scales are valid for the
accepted 1024 input; the accepted 2048 path uses live per-head Q/K scales and
the same static per-head score requantization. A 20 percent row-range tail was
exercised on the Top-k assistance lane while the attention caller consumed the
complementary sparse rows.

The OnePlus correctness gate observed the expected direct timeline for every
target attention call, exact dense-oracle token equality, exact retrieval,
finite logits, zero relevant QNN/FastRPC error lines, and no new native crash.
This proves the target path is connected and product-correct for the frozen
source/model/context/input tuples. It does not reproduce PR #703 performance:
the corrected 2048 sparse path is still slower than dense, so a formal matched
A/B with additional repetitions is not currently justified.
