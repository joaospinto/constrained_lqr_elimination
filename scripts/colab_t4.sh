#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${CLQR_CUDA_BUILD_DIR:-/content/clqr_cuda_build}"
jax_dir="${CLQR_JAX_DIR:-/content/constrained_lqr_jax}"
cuda_arch="${CLQR_CUDA_ARCH:-75}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is unavailable; select a Colab GPU runtime first." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; select a Colab GPU runtime first." >&2
  exit 2
fi

echo "=== CUDA compiler ==="
nvcc --version
echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "=== CPU allocation ==="
lscpu | sed -n "/^Architecture:/p;/^CPU(s):/p;/^Model name:/p;/^Thread(s) per core:/p;/^Core(s) per socket:/p;/^Socket(s):/p"

cmake -S "${repo_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCLQR_ENABLE_CUDA=ON \
  -DCLQR_BUILD_TESTS=ON \
  -DCLQR_BUILD_BENCHMARKS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel 2
ctest --test-dir "${build_dir}" --output-on-failure

if [[ "${CLQR_SKIP_JAX:-0}" != "1" ]]; then
  if [[ ! -d "${jax_dir}/.git" ]]; then
    git clone --depth 1 https://github.com/joaospinto/constrained_lqr_jax.git "${jax_dir}"
  fi
  python3 -m pip install --quiet -e "${jax_dir}"
  python3 "${repo_dir}/tests/cross_validate_jax.py" \
    "${build_dir}/clqr_cuda_jax_fixture"
fi

"${build_dir}/clqr_cuda_benchmark" --repeats "${CLQR_BENCHMARK_REPEATS:-5}"
