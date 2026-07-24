#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d /kaggle/working ]]; then
  notebook_work_dir=/kaggle/working
else
  notebook_work_dir=/content
fi
jax_dir="${CLQR_JAX_DIR:-${notebook_work_dir}/constrained_lqr_jax}"
jax_revision="${CLQR_JAX_REVISION:-f867e0adf4ff165782a9b9bb3ebf1be6b66c856c}"
cuda_arch="${CLQR_CUDA_ARCH:-75}"
read -r -a precisions <<< "${CLQR_PRECISIONS:-FP64}"
native_test_timeout_seconds="${CLQR_NATIVE_TEST_TIMEOUT_SECONDS:-900}"

if [[ ! "${native_test_timeout_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CLQR_NATIVE_TEST_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi

run_native_test() {
  local sanitizer_tool="$1"
  local native_test_binary="$2"
  local -a command=()
  if [[ -n "${sanitizer_tool}" ]]; then
    command+=(
      compute-sanitizer
      --tool "${sanitizer_tool}"
      --error-exitcode 99
    )
  fi
  command+=("${native_test_binary}")
  if [[ "${native_test_binary}" == *adversarial_cuda_extended_test ]]; then
    command+=(--extended)
  fi
  timeout --signal=TERM "${native_test_timeout_seconds}s" "${command[@]}"
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/notebook_bazel.sh
source "${script_dir}/notebook_bazel.sh"
bazel_command="$(clqr_notebook_bazel "${repo_dir}" "${notebook_work_dir}")"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is unavailable; select a CUDA GPU runtime first." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; select a CUDA GPU runtime first." >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is unavailable; install coreutils first." >&2
  exit 2
fi

echo "=== Source revision ==="
git -C "${repo_dir}" rev-parse HEAD

echo "=== Host platform ==="
uname -a
if [[ -r /etc/os-release ]]; then
  cat /etc/os-release
fi

echo "=== CPU topology and identification ==="
lscpu
echo "effective logical CPUs: $(nproc)"
echo "all visible logical CPUs: $(nproc --all)"
if [[ -r /proc/self/status ]]; then
  sed -n "/^Cpus_allowed_list:/p;/^Mems_allowed_list:/p" /proc/self/status
fi

echo "=== System memory ==="
if [[ -r /proc/meminfo ]]; then
  sed -n "/^MemTotal:/p;/^SwapTotal:/p" /proc/meminfo
fi
free -h

echo "=== Build toolchain ==="
c++ --version
nvcc --version
(cd "${repo_dir}" && "${bazel_command}" --version)

echo "=== NVIDIA runtime summary ==="
nvidia-smi
echo "=== GPU identification and fixed specifications ==="
gpu_query="index,name,compute_cap,driver_version,vbios_version,pci.bus_id,memory.total,clocks.max.graphics,clocks.max.sm,clocks.max.memory"
if ! nvidia-smi --query-gpu="${gpu_query}" --format=csv; then
  echo "Extended GPU query unsupported; reporting portable fields instead."
  nvidia-smi \
    --query-gpu=index,name,driver_version,pci.bus_id,memory.total \
    --format=csv
fi

if [[ "${CLQR_RUN_JAX_CROSS_VALIDATION:-0}" == "1" ]]; then
  if [[ ! -d "${jax_dir}/.git" ]]; then
    git clone --filter=blob:none --no-checkout \
      https://github.com/joaospinto/constrained_lqr_jax.git "${jax_dir}"
  fi
  git -C "${jax_dir}" fetch --depth 1 origin "${jax_revision}"
  git -C "${jax_dir}" checkout --detach FETCH_HEAD
  python3 -m pip install --quiet -e "${jax_dir}"
fi

for precision in "${precisions[@]}"; do
  case "${precision}" in
    FP64|FP32) ;;
    *)
      echo "unsupported precision '${precision}'; use FP64 or FP32" >&2
      exit 2
      ;;
  esac
  precision_suffix="$(printf '%s' "${precision}" | tr '[:upper:]' '[:lower:]')"
  echo "=== ${precision} build and tests ==="
  echo "The native CUDA suite includes dimensions beyond the former compile-time capacities, zero controls, and a zero horizon."
  bazel_args=(
    "--config=${precision_suffix}"
    --config=cuda
    "--cuda_archs=sm_${cuda_arch}"
    "--jobs=$(nproc)"
  )
  host_test_targets=(
    //:clqr_test
    //:workspace_allocation_test
    //:python_binding_test
    //:jax_binding_test
    //:cuda_stub_test
  )
  if [[ "${CLQR_RUN_JAX_FFI_TEST:-0}" == "1" ]]; then
    host_test_targets+=(//:jax_cuda_binding_test)
  fi
  native_test_targets=(//:cuda_solver_test)
  native_test_binaries=("${repo_dir}/bazel-bin/cuda_solver_test")
  run_extended_stress=0
  if [[ "${CLQR_RUN_EXTENDED_STRESS:-0}" == "1" ]] &&
     { [[ "${precision}" == "FP64" ]] ||
       [[ "${CLQR_RUN_FP32_EXTENDED_STRESS:-0}" == "1" ]]; }; then
    run_extended_stress=1
  fi
  if [[ "${run_extended_stress}" == "1" ]]; then
    host_test_targets+=(
      //:adversarial_cpu_extended_test
      //:cuda_kernel_emulation_extended_test
    )
    native_test_targets+=(//:adversarial_cuda_extended_test)
    native_test_binaries+=(
      "${repo_dir}/bazel-bin/adversarial_cuda_extended_test"
    )
  else
    host_test_targets+=(
      //:adversarial_cpu_test
      //:cuda_kernel_emulation_test
    )
    native_test_targets+=(//:adversarial_cuda_test)
    native_test_binaries+=("${repo_dir}/bazel-bin/adversarial_cuda_test")
    if [[ "${precision}" == "FP32" ]] &&
       [[ "${CLQR_RUN_EXTENDED_STRESS:-0}" == "1" ]]; then
      echo "FP32 pathological extended stress is skipped by default."
      echo "Set CLQR_RUN_FP32_EXTENDED_STRESS=1 to opt in."
    fi
  fi
  build_targets=("${native_test_targets[@]}")
  if [[ "${CLQR_RUN_JAX_CROSS_VALIDATION:-0}" == "1" ]]; then
    build_targets+=(//:clqr_cuda_jax_fixture)
  fi
  if [[ "${CLQR_SKIP_BENCHMARK:-0}" != "1" ]]; then
    build_targets+=(//:clqr_cuda_benchmark)
  fi
  cd "${repo_dir}"
  if ! "${bazel_command}" test "${bazel_args[@]}" \
       --test_output=errors "${host_test_targets[@]}"; then
    exit 8
  fi
  "${bazel_command}" build "${bazel_args[@]}" "${build_targets[@]}"

  native_tests_passed=1
  for native_test_binary in "${native_test_binaries[@]}"; do
    if ! run_native_test "" "${native_test_binary}"; then
      native_tests_passed=0
      break
    fi
  done

  if [[ "${native_tests_passed}" != "1" ]] &&
     command -v compute-sanitizer >/dev/null 2>&1 &&
     [[ "${CLQR_SKIP_SANITIZER:-0}" != "1" ]]; then
    sanitizer_tools=(memcheck initcheck racecheck synccheck)
    for sanitizer_tool in "${sanitizer_tools[@]}"; do
      echo "=== ${precision} CUDA ${sanitizer_tool} failure diagnostic ==="
      for sanitizer_binary in "${native_test_binaries[@]}"; do
        run_native_test "${sanitizer_tool}" "${sanitizer_binary}" || true
      done
    done
  fi
  if [[ "${native_tests_passed}" != "1" ]]; then
    exit 8
  fi

  if command -v compute-sanitizer >/dev/null 2>&1 &&
     [[ "${CLQR_SKIP_SANITIZER:-0}" != "1" ]]; then
    read -r -a sanitizer_tools <<< \
      "${CLQR_SANITIZER_TOOLS:-memcheck initcheck racecheck synccheck}"
    for sanitizer_tool in "${sanitizer_tools[@]}"; do
      echo "=== ${precision} CUDA ${sanitizer_tool} check ==="
      for sanitizer_binary in "${native_test_binaries[@]}"; do
        run_native_test "${sanitizer_tool}" "${sanitizer_binary}"
      done
    done
  fi

  if [[ "${CLQR_RUN_JAX_CROSS_VALIDATION:-0}" == "1" ]]; then
    python3 "${repo_dir}/tests/cross_validate_jax.py" \
      "${repo_dir}/bazel-bin/clqr_cuda_jax_fixture"
  fi

  if [[ "${CLQR_SKIP_BENCHMARK:-0}" != "1" ]]; then
    "${repo_dir}/bazel-bin/clqr_cuda_benchmark" \
      --repeats "${CLQR_BENCHMARK_REPEATS:-5}"
  fi
done

if [[ "${CLQR_COMPARE_DIMENSION_BASELINE:-0}" == "1" ]]; then
  baseline_revision="${CLQR_DIMENSION_BASELINE_REVISION:-3d225eae7e8f7c24c42ddd1cfbf921ef4d540764}"
  candidate_revision="${CLQR_DIMENSION_CANDIDATE_REVISION:-511cd314445a421c35dcb77cd6154a70e99267e2}"
  dimension_base_args="--cuda_max_state_dimension=8 --cuda_max_control_dimension=4 --cuda_max_mixed_constraints=2 --cuda_max_state_constraints=2"
  for precision in "${precisions[@]}"; do
    echo "=== ${precision} isolated runtime-dimension comparison ==="
    CLQR_BASE_REVISION="${baseline_revision}" \
    CLQR_BASE_EXTRA_BAZEL_ARGS="${dimension_base_args}" \
    CLQR_CANDIDATE_REVISION="${candidate_revision}" \
    CLQR_PRECISION="${precision}" \
    CLQR_CUDA_ARCH="${cuda_arch}" \
    CLQR_BENCHMARK_REPEATS="${CLQR_COMPARISON_REPEATS:-11}" \
    CLQR_COMPARISON_ROUNDS="${CLQR_COMPARISON_ROUNDS:-3}" \
      bash "${repo_dir}/scripts/compare_cuda_revisions.sh"
  done
fi
