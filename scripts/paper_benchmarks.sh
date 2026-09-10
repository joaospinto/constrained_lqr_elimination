#!/usr/bin/env bash
set -euo pipefail

# Reproduce the revised paper comparisons on one host. Results and third-party
# checkouts are kept outside the source tree; no historical evidence is vendored.
# Usage: bash scripts/paper_benchmarks.sh OUTPUT_DIR [--cuda]
if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != --cuda ) ]]; then
  echo "usage: $0 OUTPUT_DIR [--cuda]" >&2
  exit 2
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$1"
cuda_run=0
if [[ $# -eq 2 ]]; then cuda_run=1; fi
if [[ -e "$output_dir" ]]; then
  echo "output directory already exists; choose a new directory: $output_dir" >&2
  exit 2
fi
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
jobs="${CLQR_JOBS:-4}"
repeats="${CLQR_BENCHMARK_REPEATS:-11}"
suite="${CLQR_PAPER_SUITE:-all}"
for value in "$jobs" "$repeats"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "jobs and repeats must be positive integers" >&2
    exit 2
  fi
done
available_kb="$(df -Pk "$output_dir" | awk 'NR==2 {print $4}')"
if (( available_kb < 3 * 1024 * 1024 )); then
  echo "at least 3 GiB of free disk space is required for the reference builds" >&2
  exit 2
fi
for command in git cmake ctest c++ python3; do
  command -v "$command" >/dev/null || { echo "missing tool: $command" >&2; exit 2; }
done
if (( cuda_run )); then
  command -v nvcc >/dev/null || { echo "missing CUDA compiler: nvcc" >&2; exit 2; }
  command -v compute-sanitizer >/dev/null || { echo "missing compute-sanitizer" >&2; exit 2; }
  nvidia-smi >/dev/null
  cuda_arch="${CLQR_CUDA_ARCH:-}"
  if [[ -z "$cuda_arch" ]]; then
    gpu_caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/[[:space:]]//g' | sort -u)"
    if [[ ! "$gpu_caps" =~ ^[0-9]+\.[0-9]+$ ]]; then
      echo "Cannot select one GPU architecture; set CLQR_CUDA_ARCH for the device being tested." >&2
      exit 2
    fi
    cuda_arch="${gpu_caps/./}"
  fi
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cache_dir="${CLQR_PAPER_CACHE_DIR:-$output_dir/cache}"
mkdir -p "$cache_dir"
cache_dir="$(cd "$cache_dir" && pwd)"
available_kb="$(df -Pk "$cache_dir" | awk 'NR==2 {print $4}')"
if (( available_kb < 3 * 1024 * 1024 )); then
  echo "reference cache filesystem needs at least 3 GiB free: $cache_dir" >&2
  exit 2
fi
deps_dir="$cache_dir"
mkdir -p "$deps_dir"

checkout() {
  local name="$1" url="$2" revision="$3"
  local sparse_path="${4:-}"
  if [[ -e "$deps_dir/$name" ]]; then
    if [[ "$(git -C "$deps_dir/$name" rev-parse HEAD)" != "$revision" ||
          -n "$(git -C "$deps_dir/$name" status --porcelain)" ]]; then
      echo "reference cache must be clean and at the pinned revision: $deps_dir/$name" >&2
      exit 2
    fi
    return
  fi
  git init -q "$deps_dir/$name"
  git -C "$deps_dir/$name" remote add origin "$url"
  if [[ -n "$sparse_path" ]]; then
    git -C "$deps_dir/$name" config remote.origin.promisor true
    git -C "$deps_dir/$name" config remote.origin.partialclonefilter blob:none
    git -C "$deps_dir/$name" fetch --depth=1 --filter=blob:none origin "$revision"
    git -C "$deps_dir/$name" sparse-checkout init --cone
    git -C "$deps_dir/$name" sparse-checkout set "$sparse_path"
  else
    git -C "$deps_dir/$name" fetch --depth=1 origin "$revision"
  fi
  git -C "$deps_dir/$name" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
  if [[ "$(git -C "$deps_dir/$name" rev-parse HEAD)" != "$revision" ]]; then
    echo "unexpected revision for $name" >&2
    exit 1
  fi
}
# BLASFEO is an upstream dependency of Vanroye's reference, not of CLQR.
checkout blasfeo https://github.com/giaf/blasfeo.git 60493284741b3e4e5b0bf25155221d4b5f0b2232
checkout generalization_riccati https://github.com/lvanroye/generalization_riccati.git 8bf4b5684219a6035e93b61d8ef6ae4afd1f4e19
checkout factor_graph https://github.com/ShuoYangRobotics/equality-constraint-LQR-compare.git 905dfc5397a1b6c12e8cf9925127c9cb228a9353
checkout gtsam https://github.com/borglab/gtsam.git 8be5fe667ccf6b2c37593eeb8965917fd560db38
# Source-only working tree: the author's repository also contains old build
# artifacts. Keep the original numerical source unchanged and out of this repo.
checkout laine_author https://github.com/forrestlaine/parallel_lqr.git e24442fc853bd58fcab630c7b8b96116a53d151a src

blasfeo_target=GENERIC
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) blasfeo_target=ARMV8A_APPLE_M1 ;;
  Linux-x86_64)
    if LC_ALL=C lscpu | awk '/Flags:/ && /avx2/ {found=1} END {exit !found}'; then
      blasfeo_target=X64_INTEL_HASWELL
    fi ;;
esac
{
  printf '=== Source revision and working tree ===\n'
  git -C "$repo_dir" rev-parse HEAD
  git -C "$repo_dir" status --short
  printf '=== Host platform ===\n'
  date -u
  uname -a
  if [[ -r /etc/os-release ]]; then sed -n '1,160p' /etc/os-release; fi
  printf '=== CPU topology, affinity and memory ===\n'
  if command -v lscpu >/dev/null; then lscpu; fi
  if command -v nproc >/dev/null; then
    printf 'Effective logical CPUs: '; nproc
    printf 'All logical CPUs: '; nproc --all
  fi
  if [[ -r /proc/self/status ]]; then
    sed -n '/^Cpus_allowed_list:/p;/^Mems_allowed_list:/p' /proc/self/status
  fi
  if [[ -r /proc/meminfo ]]; then sed -n '/^MemTotal:/p;/^SwapTotal:/p' /proc/meminfo; fi
  if command -v free >/dev/null; then free -h; fi
  if [[ "$(uname -s)" == Darwin ]]; then
    sysctl hw.model hw.memsize hw.physicalcpu hw.logicalcpu machdep.cpu.brand_string
  fi
  printf '=== Toolchain ===\n'
  c++ --version
  cmake --version
  python3 --version
  if command -v nvcc >/dev/null; then nvcc --version; fi
  if command -v compute-sanitizer >/dev/null; then compute-sanitizer --version; fi
  printf 'BLASFEO target: %s\n' "$blasfeo_target"
  printf 'C++ comparison flags: Release -O3 -DNDEBUG -march=native\n'
  printf 'OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s MKL_NUM_THREADS=%s\n' \
    "$OMP_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$MKL_NUM_THREADS"
  printf 'Build jobs: %s; suite: %s; repetitions: %s\n' "$jobs" "$suite" "$repeats"
  printf '=== GPU and driver ===\n'
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi
    nvidia-smi -q
    nvidia-smi --query-gpu=index,name,uuid,compute_cap,driver_version,pci.bus_id,memory.total \
      --format=csv > "$output_dir/gpu.csv"
  fi
  if (( cuda_run )); then
    printf 'CUDA build architecture: sm_%s\n' "$cuda_arch"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-not set}"
  fi
  printf '=== Disk ===\n'
  df -h "$output_dir"
} > "$output_dir/platform.txt"

# Compile and validate before building references or timing long sweeps.
source "$repo_dir/scripts/notebook_bazel.sh"
bazel_command="$(clqr_notebook_bazel "$repo_dir" "$output_dir")"
bazel_cmd=("$bazel_command")
if [[ -n "${CLQR_PAPER_BAZEL_ROOT:-}" ]]; then
  bazel_cmd+=(--output_user_root="$CLQR_PAPER_BAZEL_ROOT")
fi
trap '"${bazel_cmd[@]}" shutdown || true' EXIT
# Use the subcommand: --version is not accepted after explicit startup options.
"${bazel_cmd[@]}" version >> "$output_dir/platform.txt"
cd "$repo_dir"
bazel_args=(--config=fp64 --jobs="$jobs" --cxxopt=-march=native)
targets=(//:clqr_paper_jax_cpu_benchmark //:clqr_paper_fixture)
if (( cuda_run )); then
  bazel_args+=(--config=cuda --cuda_archs="sm_${cuda_arch}")
  targets+=(//:clqr_paper_cuda_benchmark //:clqr_paper_jax_cuda_benchmark
            //:cuda_solver_test //:adversarial_cuda_extended_test)
fi
"${bazel_cmd[@]}" build "${bazel_args[@]}" "${targets[@]}"
bazel-bin/clqr_paper_fixture --suite "$suite" > "$output_dir/cases.json"
"${bazel_cmd[@]}" test "${bazel_args[@]}" --test_output=errors \
  //:clqr_test //:workspace_allocation_test //:scaling_problem_test \
  //:reduced_objective_test //:paper_jax_fixture_test \
  //:cuda_kernel_emulation_extended_test //:paper_results_test //:notebook_paper_test \
  //:corrected_laine_benchmark_test \
  //benchmarks/reference:quadratic_factor_test \
  //benchmarks/reference:stationarity_audit_test \
  //external_algorithms/corrected_laine_tomlin:all

# Build the authors' factor-graph dependency without unused modules or Boost.
cmake -S "$deps_dir/gtsam" -B "$cache_dir/gtsam-build" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGTSAM_USE_BOOST_FEATURES=OFF -DGTSAM_ENABLE_BOOST_SERIALIZATION=OFF \
  -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_UNSTABLE=OFF -DGTSAM_BUILD_PYTHON=OFF \
  -DGTSAM_WITH_TBB=OFF -DGTSAM_WITH_EIGEN_MKL=OFF \
  -DGTSAM_WITH_EIGEN_MKL_OPENMP=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=ON \
  -DGTSAM_INSTALL_CPPUNITLITE=OFF
cmake --build "$cache_dir/gtsam-build" --target gtsam -j "$jobs"
cmake -S "$repo_dir/benchmarks/reference" -B "$cache_dir/reference-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBLASFEO_SOURCE_DIR="$deps_dir/blasfeo" -DTARGET="$blasfeo_target" \
  -DGEN_RICCATI_SOURCE_DIR="$deps_dir/generalization_riccati" \
  -DFACTOR_GRAPH_SOURCE_DIR="$deps_dir/factor_graph" \
  -DLAINE_SOURCE_DIR="$deps_dir/laine_author" \
  -DEIGEN_SOURCE_DIR="$deps_dir/gtsam/gtsam/3rdparty/Eigen" \
  -DCLQR_COMPARE_LAINE_CORRECTED=ON \
  -DGTSAM_DIR="$cache_dir/gtsam-build"
cmake --build "$cache_dir/reference-build" -j "$jobs"
ctest --test-dir "$cache_dir/reference-build" --output-on-failure

# Numerical benchmark outcomes are data, not pass/fail gates. Nonzero exits
# still report harness failures, regression-test failures, and sanitizer errors.
failed=0
measure() {
  local name="$1"
  shift
  if "$@" > "$output_dir/$name.csv" 2> "$output_dir/$name.stderr"; then
    printf 'Completed %s\n' "$name"
  else
    printf 'CHECK FAILED: %s (see CSV and stderr)\n' "$name" >&2
    failed=1
  fi
}
check_log() {
  local name="$1"
  shift
  if "$@" > "$output_dir/$name.log" 2>&1; then
    printf 'Passed %s\n' "$name"
  else
    printf 'CHECK FAILED: %s (see its log)\n' "$name" >&2
    failed=1
  fi
}
bench_args=(--suite "$suite" --repeats "$repeats")
measure references "$cache_dir/reference-build/clqr_reference_benchmark" "${bench_args[@]}"
# Opposite-order rounds make scheduling/clock drift visible.
measure cpu_round1 "$cache_dir/reference-build/clqr_cpu_benchmark" "${bench_args[@]}"
measure laine_round1 "$cache_dir/reference-build/clqr_laine_benchmark" "${bench_args[@]}"
measure laine_corrected_round1 "$cache_dir/reference-build/clqr_laine_corrected_benchmark" "${bench_args[@]}"
measure laine_corrected_round2 "$cache_dir/reference-build/clqr_laine_corrected_benchmark" "${bench_args[@]}"
measure laine_round2 "$cache_dir/reference-build/clqr_laine_benchmark" "${bench_args[@]}"
measure cpu_round2 "$cache_dir/reference-build/clqr_cpu_benchmark" "${bench_args[@]}"

measure jax_cpu bazel-bin/clqr_paper_jax_cpu_benchmark "${bench_args[@]}"
if (( cuda_run )); then
  check_log cuda_validation bazel-bin/cuda_solver_test
  check_log cuda_extended_validation bazel-bin/adversarial_cuda_extended_test --extended
  for sanitizer in memcheck initcheck racecheck synccheck; do
    check_log "cuda_regression_${sanitizer}" compute-sanitizer --tool "$sanitizer" \
      --error-exitcode=99 bazel-bin/cuda_solver_test
    check_log "cuda_${sanitizer}" compute-sanitizer --tool "$sanitizer" \
      --error-exitcode=99 bazel-bin/clqr_paper_cuda_benchmark --suite smoke --repeats 1
  done
  measure cuda_host bazel-bin/clqr_paper_cuda_benchmark "${bench_args[@]}"
  measure cuda_resident bazel-bin/clqr_paper_jax_cuda_benchmark "${bench_args[@]}" \
    --capacity-report "$output_dir/cuda_host.csv"
  # Refresh the original table with its original build flags and fixture,
  # separately from the native-tuned dense comparison above.
  "${bazel_cmd[@]}" build --config=fp64 --config=cuda \
    --cuda_archs="sm_${cuda_arch}" --jobs="$jobs" //:clqr_cuda_benchmark
  measure original_table bazel-bin/clqr_cuda_benchmark --repeats "$repeats"
fi
"${bazel_cmd[@]}" shutdown
summary_args=(--suite "$suite" --require-laine)
if (( cuda_run )); then summary_args+=(--cuda); fi
if ! python3 scripts/paper_results.py "$output_dir" "${summary_args[@]}" \
    > "$output_dir/summary.stdout" 2> "$output_dir/summary.stderr"; then
  echo "CHECK FAILED: result summary (see summary.stderr and summary.json)" >&2
  failed=1
fi
printf '\nResults: %s\n' "$output_dir"
df -h "$output_dir"
exit "$failed"
