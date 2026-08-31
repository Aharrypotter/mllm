#!/usr/bin/env bash
set -euo pipefail

# Run a matched dense/HMX mllmV2 Qwen2.5-1.5B A/B on the already
# provisioned OnePlus 13T Termux environment. This script is execution-only:
# it never transfers files, reboots the phone, changes power policy, or
# deletes remote data.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
result_dir="${RESULT_DIR:-${repo_root}/results/shadownpu-qwen-v2}"

device_host="${DEVICE_HOST:-oneplus-pkx110.taild842a2.ts.net}"
device_user="${DEVICE_USER:-u0_a388}"
device_port="${DEVICE_PORT:-8022}"
device_root="${DEVICE_ROOT:-/data/data/com.termux/files/home/mllm-shadownpu-v2}"
run_dir="${RUN_DIR:-${device_root}/run}"
prompt_dir="${PROMPT_DIR:-${device_root}/prompts}"

bank="${HMX_BANK:-${device_root}/hmx-qwen2-v79-m512-h12-n4160}"
profile="${HEAD_PROFILE:-${device_root}/profiles/qwen2-1.5B-retain-0.2.txt}"
dense_profile="${DENSE_HEAD_PROFILE:-}"
benchmark="${BENCHMARK:-${device_root}/bin/mllm-qwen-npu}"
redirect="${SESSION_REDIRECT:-${device_root}/lib/libfastrpc_session_redirect.so}"
qnn_lib_dir="${QNN_LIB_DIR:-${device_root}/qnn-lib}"
qnn_cpu_op_package="${QNN_CPU_OP_PACKAGE:-${qnn_lib_dir}/libQnnLLaMAPackage_CPU.so}"
qnn_htp_op_package="${QNN_HTP_OP_PACKAGE:-${qnn_lib_dir}/libQnnLLaMAPackage_HTP.so}"
config="${CONFIG:-${device_root}/config/qwen2.5-1.5b-qnn.json}"
vocab="${VOCAB:-${device_root}/vocab/qwen2.5_vocab.mllm}"
merge="${MERGE:-${device_root}/vocab/qwen2.5_merges.txt}"
qnn_model="${QNN_MODEL:-${device_root}/models/Qwen2.5-1.5B-Instruct_rotated-noshadow.mllm}"
decoding_model="${DECODING_MODEL:-${device_root}/models/Qwen2.5-1.5B-Instruct_rotated-Q40.mllm}"
qnn_context="${QNN_CONTEXT:-${run_dir}/qnn_context_m256.bin}"
model_version="${MODEL_VERSION:-v1}"

prompts="${PROMPTS:-1024 2048 4096}"
repetition_count="${REPETITIONS:-3}"
decode_tokens="${DECODE_TOKENS:-8}"
dense_main_cpu="${DENSE_MAIN_CPU:-7}"
sparse_main_cpu="${SPARSE_MAIN_CPU:-2}"

if [[ ! "${repetition_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPETITIONS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${decode_tokens}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DECODE_TOKENS must be a positive integer" >&2
  exit 2
fi
if [[ "${model_version}" != "v1" && "${model_version}" != "v2" ]]; then
  echo "MODEL_VERSION must be v1 or v2" >&2
  exit 2
fi

read -r -a prompt_values <<<"${prompts}"
if (( ${#prompt_values[@]} == 0 )); then
  echo "PROMPTS must contain at least one positive integer" >&2
  exit 2
fi
for prompt_tokens in "${prompt_values[@]}"; do
  if [[ ! "${prompt_tokens}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PROMPTS must contain positive integers" >&2
    exit 2
  fi
done

remote_paths=("${device_root}" "${run_dir}" "${prompt_dir}" "${bank}" "${profile}" "${benchmark}" "${redirect}"
              "${qnn_lib_dir}" "${qnn_cpu_op_package}" "${qnn_htp_op_package}" "${config}" "${vocab}" "${merge}"
              "${qnn_model}" "${decoding_model}" "${qnn_context}")
if [[ -n "${dense_profile}" ]]; then remote_paths+=("${dense_profile}"); fi
for remote_path in "${remote_paths[@]}"; do
  if [[ ! "${remote_path}" =~ ^/[A-Za-z0-9._/+:-]+$ ]]; then
    echo "remote paths must be absolute and contain no whitespace or shell metacharacters: ${remote_path}" >&2
    exit 2
  fi
done

ssh_prefix=(ssh -o BatchMode=yes -p "${device_port}")
if [[ -n "${DEVICE_KEY:-}" ]]; then
  ssh_prefix+=(-i "${DEVICE_KEY}")
fi
ssh_prefix+=("${device_user}@${device_host}")

identity="$("${ssh_prefix[@]}" "/system/bin/getprop ro.product.manufacturer; /system/bin/getprop ro.product.model; /system/bin/getprop ro.product.cpu.abi")"
manufacturer="$(printf '%s\n' "${identity}" | sed -n '1p')"
model="$(printf '%s\n' "${identity}" | sed -n '2p')"
abi="$(printf '%s\n' "${identity}" | sed -n '3p')"
if [[ "${manufacturer}" != "OnePlus" || "${model}" != "PKX110" || "${abi}" != "arm64-v8a" ]]; then
  echo "refusing to run: expected OnePlus PKX110 arm64-v8a, observed ${identity//$'\n'/ }" >&2
  exit 2
fi

"${ssh_prefix[@]}" "test -x '${benchmark}' && test -d '${qnn_lib_dir}' && test -f '${bank}/manifest.tsv' && test -f '${profile}' && test -f '${redirect}' && test -f '${qnn_cpu_op_package}' && test -f '${qnn_htp_op_package}' && test -f '${config}' && test -f '${vocab}' && test -f '${merge}' && test -f '${qnn_model}' && test -f '${decoding_model}' && test -f '${qnn_context}' && ${dense_profile:+test -f '${dense_profile}' && }command -v taskset >/dev/null"
for prompt_tokens in "${prompt_values[@]}"; do
  "${ssh_prefix[@]}" "test -f '${prompt_dir}/prompt-${prompt_tokens}.txt'"
done

manifest_row="$("${ssh_prefix[@]}" "awk -F '\t' '\$1 == \"qwen2_1p5b\" && \$2 == 12 && \$5 == 512 && \$6 == 128 && \$7 == 4160 { operators[++count] = \$3; skels[count] = \$4 } END { if (count) { selected = int((count + 1) / 2); print operators[selected] \"\\t\" skels[selected] } }' '${bank}/manifest.tsv'" | tr -d '\r')"
if [[ -z "${manifest_row}" || "${manifest_row}" == *$'\n'* || "${manifest_row}" != *$'\t'* ]]; then
  echo "failed to select one qwen2_1p5b H12/M512/K128/N4160 operator/dispatcher pair" >&2
  exit 2
fi
manifest_operator="${manifest_row%%$'\t'*}"
manifest_dispatcher="${manifest_row#*$'\t'}"
operator="${HMX_OPERATOR:-${manifest_operator}}"
operator_path="${operator}"
dispatcher_path="${manifest_dispatcher}"
if [[ "${operator_path}" != /* ]]; then operator_path="${bank}/${operator_path}"; fi
if [[ "${dispatcher_path}" != /* ]]; then dispatcher_path="${bank}/${dispatcher_path}"; fi
"${ssh_prefix[@]}" "test -f '${operator_path}' && test -f '${dispatcher_path}'"

mkdir -p "${result_dir}"
logs=()

run_one() {
  local prompt_tokens="$1"
  local mode="$2"
  local repetition="$3"
  local log="${result_dir}/${mode}-p${prompt_tokens}-run${repetition}.log"
  local attention_mode="dense"
  local main_cpu="${dense_main_cpu}"
  local worker="0"
  local profile_argument=""
  if [[ "${mode}" == "dense" && -n "${dense_profile}" ]]; then
    attention_mode="hmx-topk"
    profile_argument="--head-retain-profile '${dense_profile}'"
  fi
  if [[ "${mode}" == "sparse" ]]; then
    attention_mode="hmx-topk"
    main_cpu="${sparse_main_cpu}"
    worker="1"
    profile_argument="--head-retain-profile '${profile}'"
  fi
  logs+=("${log}")
  printf 'SHADOWNPU_HARNESS_CONFIG device=OnePlus-PKX110 mode=%s attention_mode=%s prompt_tokens=%s repetition=%s operator=%s dispatcher=%s\n' \
    "${mode}" "${attention_mode}" "${prompt_tokens}" "${repetition}" "${operator_path}" "${dispatcher_path}" >"${log}"

  "${ssh_prefix[@]}" "cd '${run_dir}' && env \
LD_PRELOAD='${redirect}' \
LD_LIBRARY_PATH='${qnn_lib_dir}:${bank}' \
ADSP_LIBRARY_PATH='${qnn_lib_dir};${bank};/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp' \
MLLM_QNN_CPU_OP_PACKAGE='${qnn_cpu_op_package}' \
MLLM_QNN_HTP_OP_PACKAGE='${qnn_htp_op_package}' \
MLLM_QNN_CONTEXT_MEMORY_LIMIT_MB=1024 \
MLLM_ATTENTION_PROFILE=1 \
MLLM_HMX_INT8_OPERATOR_LIBRARY='${operator_path}' \
MLLM_HMX_INT8_OPERATOR_MANIFEST='${bank}/manifest.tsv' \
MLLM_HMX_INT8_OPERATOR_MODEL=qwen2_1p5b \
MLLM_HMX_INT8_CPU_SCALE_PROFILE=1 \
MLLM_HMX_INT8_CPU_PACK=1 \
MLLM_HMX_INT8_CPU_PACK_INTRA_HEAD=1 \
MLLM_HMX_CPU_PACK_CPU=5 \
MLLM_HMX_CPU_PACK_WORKERS=1 \
MLLM_HMX_CPU_PACK_OVERLAP=1 \
MLLM_HMX_CPU_PACK_SPIN_US=20000 \
MLLM_HMX_INT8_INCREMENTAL_K_SCALE=1 \
MLLM_HMX_INT8_LAYER_K_CACHE=1 \
MLLM_HMX_INT8_ASYNC_K_CACHE_STORE=1 \
MLLM_HMX_INT8_FUSE_PER_HEAD_BUCKETS=1 \
MLLM_HMX_INT8_PER_HEAD_BUCKET_GROUP_HEADS=12 \
MLLM_HMX_INT8_DYNAMIC_QK_SCALE=0 \
MLLM_HMX_INT8_DYNAMIC_OUTPUT_SCALE=0 \
MLLM_HMX_INT8_DSP_INT32_TOPK=0 \
MLLM_HMX_INT8_TOPK_OVERSAMPLE=1 \
MLLM_HMX_INT8_OUTLIER_FALLBACK=0 \
MLLM_HMX_PIPELINE_GROUP_HEADS=0 \
MLLM_HMX_PIPELINE_TOPK_CPU=4,6 \
MLLM_HMX_PIPELINE_TOPK_WORKERS=2 \
MLLM_HMX_PIPELINE_RPC_CPU=1 \
MLLM_HMX_PIPELINE_LONG_RPC=1 \
MLLM_CPU_ATTENTION_WORKER='${worker}' \
MLLM_CPU_ATTENTION_WORKER_CPU=7 \
MLLM_SPARSE_ATTENTION_THREADS=1 \
OMP_NUM_THREADS=4 \
KMP_AFFINITY='norespect,granularity=fine,proclist=[7,4,5,6],explicit' \
taskset -c '${main_cpu}' '${benchmark}' \
--config '${config}' \
--npu-model '${qnn_model}' \
--cpu-model '${decoding_model}' \
--qnn-context '${qnn_context}' \
--tokenizer '${vocab}' \
--merges '${merge}' \
--model-version '${model_version}' \
      --prompt-file '${prompt_dir}/prompt-${prompt_tokens}.txt' \
      --expected-prompt-tokens '${prompt_tokens}' \
      --expected-access-code NPU-7391 \
      --max-new-tokens '${decode_tokens}' \
--prefill-chunk 512 \
--qnn-sequence-tile 256 \
--attention-mode '${attention_mode}' ${profile_argument}" 2>&1 | tee -a "${log}"
}

for prompt_tokens in "${prompt_values[@]}"; do
  for ((repetition = 1; repetition <= repetition_count; ++repetition)); do
    if (( repetition % 2 == 1 )); then
      run_one "${prompt_tokens}" sparse "${repetition}"
      run_one "${prompt_tokens}" dense "${repetition}"
    else
      run_one "${prompt_tokens}" dense "${repetition}"
      run_one "${prompt_tokens}" sparse "${repetition}"
    fi
  done
done

python3 "${repo_root}/tools/shadownpu/summarize_results.py" \
  "${logs[@]}" --output "${result_dir}/summary.tsv"
