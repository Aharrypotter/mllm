#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
  echo 'Usage: bash run_qwen_v81_resident.sh COMMAND [ARG ...]' >&2
  exit 2
fi

export MLLM_QNN_CONTEXT_MEMORY_LIMIT_MB=0
export MLLM_QNN_CONTEXT_PERSISTENT_BINARY=1
exec "$@"
