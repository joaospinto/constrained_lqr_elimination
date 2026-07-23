#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d /kaggle/working ]]; then
  notebook_work_dir=/kaggle/working
else
  notebook_work_dir=/content
fi
build_root="${CLQR_CUDA_BUILD_DIR:-${notebook_work_dir}/clqr_cuda_build}"
jax_dir="${CLQR_JAX_DIR:-${notebook_work_dir}/constrained_lqr_jax}"
jax_revision="${CLQR_JAX_REVISION:-f867e0adf4ff165782a9b9bb3ebf1be6b66c856c}"
cuda_arch="${CLQR_CUDA_ARCH:-75}"
max_state_dimension="${CLQR_CUDA_MAX_STATE_DIMENSION:-8}"
max_control_dimension="${CLQR_CUDA_MAX_CONTROL_DIMENSION:-4}"
max_mixed_constraints="${CLQR_CUDA_MAX_MIXED_CONSTRAINTS:-2}"
max_state_constraints="${CLQR_CUDA_MAX_STATE_CONSTRAINTS:-2}"
read -r -a precisions <<< "${CLQR_PRECISIONS:-FP64}"

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
cmake --version
c++ --version
nvcc --version

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

if [[ "${CLQR_SKIP_JAX:-0}" != "1" ]]; then
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
  build_dir="${build_root}_${precision_suffix}"
  echo "=== ${precision} build and tests ==="
  echo "CUDA capacities: state=${max_state_dimension}, control=${max_control_dimension}, mixed=${max_mixed_constraints}, state constraints=${max_state_constraints}"
  cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCLQR_ENABLE_CUDA=ON \
    -DCLQR_BUILD_TESTS=ON \
    -DCLQR_BUILD_BENCHMARKS=ON \
    -DCLQR_PRECISION="${precision}" \
    -DCLQR_CUDA_MAX_STATE_DIMENSION="${max_state_dimension}" \
    -DCLQR_CUDA_MAX_CONTROL_DIMENSION="${max_control_dimension}" \
    -DCLQR_CUDA_MAX_MIXED_CONSTRAINTS="${max_mixed_constraints}" \
    -DCLQR_CUDA_MAX_STATE_CONSTRAINTS="${max_state_constraints}" \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
  cmake --build "${build_dir}" --parallel "$(nproc)"
  if ! ctest --test-dir "${build_dir}" --output-on-failure; then
    if command -v compute-sanitizer >/dev/null 2>&1 &&
       [[ "${CLQR_SKIP_SANITIZER:-0}" != "1" ]]; then
      for sanitizer_tool in memcheck initcheck racecheck synccheck; do
        echo "=== ${precision} CUDA ${sanitizer_tool} failure diagnostic ==="
        compute-sanitizer --tool "${sanitizer_tool}" --error-exitcode 99 \
          "${build_dir}/clqr_cuda_test" || true
      done
    fi
    exit 8
  fi

  if command -v compute-sanitizer >/dev/null 2>&1 && \
     [[ "${CLQR_SKIP_SANITIZER:-0}" != "1" ]]; then
    read -r -a sanitizer_tools <<< \
      "${CLQR_SANITIZER_TOOLS:-memcheck initcheck racecheck synccheck}"
    for sanitizer_tool in "${sanitizer_tools[@]}"; do
      echo "=== ${precision} CUDA ${sanitizer_tool} check ==="
      compute-sanitizer --tool "${sanitizer_tool}" --error-exitcode 99 \
        "${build_dir}/clqr_cuda_test"
    done
  fi

  if [[ "${CLQR_SKIP_JAX:-0}" != "1" ]]; then
    python3 "${repo_dir}/tests/cross_validate_jax.py" \
      "${build_dir}/clqr_cuda_jax_fixture"
  fi

  "${build_dir}/clqr_cuda_benchmark" \
    --repeats "${CLQR_BENCHMARK_REPEATS:-5}"
done
