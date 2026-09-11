#!/usr/bin/env bash
# Source this before fetching dependencies or constructing benchmark targets.
# Each backend override takes precedence over the external-solvers master switch.
: "${CLQR_RUN_EXTERNAL:=1}"
: "${CLQR_RUN_VANROYE:=$CLQR_RUN_EXTERNAL}"
: "${CLQR_RUN_YANG:=$CLQR_RUN_EXTERNAL}"
: "${CLQR_RUN_LAINE:=$CLQR_RUN_EXTERNAL}"
: "${CLQR_RUN_CORRECTED_LAINE:=$CLQR_RUN_EXTERNAL}"
: "${CLQR_RUN_JAX:=1}"
: "${CLQR_RUN_TESTS:=1}"
: "${CLQR_RUN_SANITIZERS:=1}"
: "${CLQR_RUN_ORIGINAL_TABLE:=1}"
clqr_benchmark_options=(
  CLQR_RUN_EXTERNAL CLQR_RUN_VANROYE CLQR_RUN_YANG CLQR_RUN_LAINE
  CLQR_RUN_CORRECTED_LAINE CLQR_RUN_JAX CLQR_RUN_TESTS CLQR_RUN_SANITIZERS
  CLQR_RUN_ORIGINAL_TABLE
)
for clqr_option in "${clqr_benchmark_options[@]}"; do
  if [[ "${!clqr_option}" != 0 && "${!clqr_option}" != 1 ]]; then
    echo "$clqr_option must be 0 or 1" >&2
    return 2
  fi
  export "$clqr_option"
done
unset clqr_option
