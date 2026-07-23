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
max_state_dimension="${CLQR_CUDA_MAX_STATE_DIMENSION:-8}"
max_control_dimension="${CLQR_CUDA_MAX_CONTROL_DIMENSION:-4}"
max_mixed_constraints="${CLQR_CUDA_MAX_MIXED_CONSTRAINTS:-2}"
max_state_constraints="${CLQR_CUDA_MAX_STATE_CONSTRAINTS:-2}"
read -r -a precisions <<< "${CLQR_PRECISIONS:-FP64}"

bazelisk_version=1.29.0
bazelisk_sha256=5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992
bazel_command="${CLQR_BAZEL:-}"
required_bazel_version="$(<"${repo_dir}/.bazelversion")"

if [[ -z "${bazel_command}" ]] && command -v bazelisk >/dev/null 2>&1; then
  bazel_command="$(command -v bazelisk)"
fi
if [[ -z "${bazel_command}" ]] && command -v bazel >/dev/null 2>&1 &&
   [[ "$(bazel --version 2>/dev/null)" == "bazel ${required_bazel_version}" ]]; then
  bazel_command="$(command -v bazel)"
fi
if [[ -z "${bazel_command}" ]]; then
  case "$(uname -m)" in
    x86_64|amd64) bazelisk_asset=bazelisk-linux-amd64 ;;
    *)
      echo "automatic Bazelisk bootstrap supports x86-64 Linux only; set CLQR_BAZEL" >&2
      exit 2
      ;;
  esac
  tool_dir="${notebook_work_dir}/clqr_tools"
  bazel_command="${tool_dir}/bazelisk-${bazelisk_version}"
  if [[ ! -x "${bazel_command}" ]] ||
     ! printf '%s  %s\n' "${bazelisk_sha256}" "${bazel_command}" |
       sha256sum --check --status; then
    mkdir -p "${tool_dir}"
    bazelisk_tmp="${bazel_command}.tmp"
    curl --fail --location --silent --show-error \
      "https://github.com/bazelbuild/bazelisk/releases/download/v${bazelisk_version}/${bazelisk_asset}" \
      --output "${bazelisk_tmp}"
    printf '%s  %s\n' "${bazelisk_sha256}" "${bazelisk_tmp}" |
      sha256sum --check
    chmod +x "${bazelisk_tmp}"
    mv "${bazelisk_tmp}" "${bazel_command}"
  fi
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is unavailable; select a CUDA GPU runtime first." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; select a CUDA GPU runtime first." >&2
  exit 2
fi

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
  echo "CUDA capacities: state=${max_state_dimension}, control=${max_control_dimension}, mixed=${max_mixed_constraints}, state constraints=${max_state_constraints}"
  bazel_args=(
    "--config=${precision_suffix}"
    --config=cuda
    "--cuda_archs=sm_${cuda_arch}"
    "--cuda_max_state_dimension=${max_state_dimension}"
    "--cuda_max_control_dimension=${max_control_dimension}"
    "--cuda_max_mixed_constraints=${max_mixed_constraints}"
    "--cuda_max_state_constraints=${max_state_constraints}"
    "--jobs=$(nproc)"
  )
  test_targets=(
    //:clqr_test
    //:workspace_allocation_test
    //:python_binding_test
    //:cuda_stub_test
    //:cuda_kernel_emulation_test
    //:cuda_solver_test
  )
  build_targets=(
    //:clqr_cuda_jax_fixture
    //:clqr_cuda_benchmark
  )
  cd "${repo_dir}"
  if ! "${bazel_command}" test "${bazel_args[@]}" \
       --test_output=errors "${test_targets[@]}"; then
    if command -v compute-sanitizer >/dev/null 2>&1 &&
       [[ "${CLQR_SKIP_SANITIZER:-0}" != "1" ]] &&
       [[ -x "${repo_dir}/bazel-bin/cuda_solver_test" ]]; then
      for sanitizer_tool in memcheck initcheck racecheck synccheck; do
        echo "=== ${precision} CUDA ${sanitizer_tool} failure diagnostic ==="
        compute-sanitizer --tool "${sanitizer_tool}" --error-exitcode 99 \
          "${repo_dir}/bazel-bin/cuda_solver_test" || true
      done
    fi
    exit 8
  fi
  "${bazel_command}" build "${bazel_args[@]}" "${build_targets[@]}"

  if command -v compute-sanitizer >/dev/null 2>&1 && \
     [[ "${CLQR_SKIP_SANITIZER:-0}" != "1" ]]; then
    read -r -a sanitizer_tools <<< \
      "${CLQR_SANITIZER_TOOLS:-memcheck initcheck racecheck synccheck}"
    for sanitizer_tool in "${sanitizer_tools[@]}"; do
      echo "=== ${precision} CUDA ${sanitizer_tool} check ==="
      compute-sanitizer --tool "${sanitizer_tool}" --error-exitcode 99 \
        "${repo_dir}/bazel-bin/cuda_solver_test"
    done
  fi

  if [[ "${CLQR_RUN_JAX_CROSS_VALIDATION:-0}" == "1" ]]; then
    python3 "${repo_dir}/tests/cross_validate_jax.py" \
      "${repo_dir}/bazel-bin/clqr_cuda_jax_fixture"
  fi

  "${repo_dir}/bazel-bin/clqr_cuda_benchmark" \
    --repeats "${CLQR_BENCHMARK_REPEATS:-5}"
done
