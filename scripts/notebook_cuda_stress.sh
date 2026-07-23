#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CLQR_RUN_EXTENDED_STRESS=1
export CLQR_SKIP_BENCHMARK=1

exec "${script_dir}/notebook_cuda.sh" "$@"
