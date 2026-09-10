#!/usr/bin/env bash
set -euo pipefail

# Same implementation/fixtures, changing only dense scratch placement.
# Reuse one private cache for both builds; notebook_paper.py archives/cleans it.
if [[ $# != 2 || "$2" != --cuda ]]; then
  echo "usage: $0 NEW_OUTPUT_DIR --cuda" >&2
  exit 2
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -e "$1" ]]; then
  echo "output directory already exists: $1" >&2
  exit 2
fi
mkdir -p "$1"
output_dir="$(cd "$1" && pwd)"
: "${CLQR_RUN_EXTERNAL:=0}"
: "${CLQR_RUN_JAX:=0}"
: "${CLQR_RUN_ORIGINAL_TABLE:=0}"
source "$repo_dir/scripts/benchmark_options.sh"
export CLQR_PAPER_SUITE="${CLQR_PAPER_SUITE:-scratch}"
export CLQR_PAPER_CACHE_DIR="${CLQR_PAPER_CACHE_DIR:-$output_dir/cache}"
export CLQR_PAPER_BAZEL_ROOT="${CLQR_PAPER_BAZEL_ROOT:-$CLQR_PAPER_CACHE_DIR/bazel}"
export BAZELISK_HOME="${BAZELISK_HOME:-$CLQR_PAPER_CACHE_DIR/bazelisk}"
failed=0
for placement in auto global; do
  printf '\n=== FP64 scratch placement: %s ===\n' "$placement"
  if ! CLQR_CUDA_SCRATCH="$placement" bash "$repo_dir/scripts/paper_benchmarks.sh" \
      "$output_dir/$placement" --cuda; then
    failed=1
  fi
done
if ! python3 "$repo_dir/scripts/cuda_scratch_results.py" "$output_dir"; then
  failed=1
fi
exit "$failed"
