#!/usr/bin/env bash
set -euo pipefail

# Build two revisions in isolation, alternate benchmark order, and report
# median phase timings. Run correctness and sanitizer validation first.

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d /kaggle/working ]]; then
  notebook_work_dir=/kaggle/working
elif [[ -d /content ]]; then
  notebook_work_dir=/content
else
  notebook_work_dir="${TMPDIR:-/tmp}"
fi

base_revision="${CLQR_BASE_REVISION:-origin/main}"
candidate_revision="${CLQR_CANDIDATE_REVISION:-HEAD}"
precision="${CLQR_PRECISION:-FP64}"
cuda_arch="${CLQR_CUDA_ARCH:-60}"
repeats="${CLQR_BENCHMARK_REPEATS:-11}"
rounds="${CLQR_COMPARISON_ROUNDS:-3}"
keep_output="${CLQR_KEEP_COMPARE_OUTPUT:-0}"

case "${precision}" in
  FP64|FP32) ;;
  *)
    echo "unsupported precision '${precision}'; use FP64 or FP32" >&2
    exit 2
    ;;
esac
for value_name in repeats rounds; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer" >&2
    exit 2
  fi
done
if [[ "${keep_output}" != "0" && "${keep_output}" != "1" ]]; then
  echo "CLQR_KEEP_COMPARE_OUTPUT must be 0 or 1" >&2
  exit 2
fi
precision_suffix="$(printf '%s' "${precision}" | tr '[:upper:]' '[:lower:]')"
if command -v nproc >/dev/null 2>&1; then
  parallel_jobs="$(nproc)"
else
  parallel_jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1\n')"
fi

# shellcheck source=scripts/notebook_bazel.sh
source "${repo_dir}/scripts/notebook_bazel.sh"
bazel_command="$(clqr_notebook_bazel "${repo_dir}" "${notebook_work_dir}")"

resolve_revision() {
  local revision="$1"
  local label="$2"
  local commit
  if commit="$(git -C "${repo_dir}" rev-parse --verify \
      "${revision}^{commit}" 2>/dev/null)"; then
    printf '%s\n' "${commit}"
    return
  fi
  git -C "${repo_dir}" fetch --depth 1 origin "${revision}"
  commit="$(git -C "${repo_dir}" rev-parse FETCH_HEAD)"
  git -C "${repo_dir}" update-ref "refs/clqr-compare/${label}" "${commit}"
  printf '%s\n' "${commit}"
}

base_commit="$(resolve_revision "${base_revision}" baseline)"
candidate_commit="$(resolve_revision "${candidate_revision}" candidate)"
if [[ "${base_commit}" == "${candidate_commit}" ]]; then
  echo "baseline and candidate resolve to the same commit ${base_commit}" >&2
  exit 2
fi

run_root="${CLQR_COMPARE_RUN_ROOT:-${notebook_work_dir}/clqr_cuda_compare_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "${run_root}" ]]; then
  echo "comparison output already exists: ${run_root}" >&2
  exit 2
fi
mkdir -p "${run_root}"

cleanup() {
  if [[ "${keep_output}" == "0" && -d "${run_root}" ]]; then
    chmod -R u+w "${run_root}" 2>/dev/null || true
    rm -rf -- "${run_root}"
  fi
}
trap cleanup EXIT

bazel_args=(
  "--config=${precision_suffix}"
  --config=cuda
  "--cuda_archs=sm_${cuda_arch}"
  "--jobs=${parallel_jobs}"
)

build_revision() {
  local label="$1"
  local commit="$2"
  local source_dir="${run_root}/${label}-source"
  local archive="${run_root}/${label}.tar"
  local output_base="${run_root}/${label}-bazel"
  local build_log="${run_root}/${label}-build.log"

  echo "Building ${label} ${commit}..." >&2
  mkdir -p "${source_dir}"
  git -C "${repo_dir}" archive --format=tar --output="${archive}" "${commit}"
  tar -xf "${archive}" -C "${source_dir}"
  rm -f -- "${archive}"
  if ! (
    cd "${source_dir}"
    "${bazel_command}" "--output_base=${output_base}" build \
      "${bazel_args[@]}" //:clqr_cuda_benchmark
  ) >"${build_log}" 2>&1; then
    echo "${label} build failed:" >&2
    tail -n 80 "${build_log}" >&2
    return 1
  fi
}

run_benchmark() {
  local label="$1"
  local round="$2"
  local source_dir="${run_root}/${label}-source"
  local report="${run_root}/${label}-${round}.csv"
  echo "Benchmarking ${label}, round ${round}/${rounds}..." >&2
  "${source_dir}/bazel-bin/clqr_cuda_benchmark" --repeats "${repeats}" \
    >"${report}"
}

build_revision baseline "${base_commit}"
build_revision candidate "${candidate_commit}"

for ((round = 1; round <= rounds; ++round)); do
  if ((round % 2 == 1)); then
    run_benchmark baseline "${round}"
    run_benchmark candidate "${round}"
  else
    run_benchmark candidate "${round}"
    run_benchmark baseline "${round}"
  fi
done

python3 - "${run_root}" "${base_commit}" "${candidate_commit}" <<'PY'
import csv
import glob
import statistics
import sys


def read_report(path):
    with open(path, encoding="utf-8") as stream:
        lines = [line for line in stream if line and not line.startswith("#")]
    return {int(row["N"]): row for row in csv.DictReader(lines)}


def aggregate(root, label):
    reports = [
        read_report(path)
        for path in sorted(glob.glob(f"{root}/{label}-*.csv"))
    ]
    horizons = set.intersection(*(set(report) for report in reports))
    result = {}
    for horizon in sorted(horizons):
        keys = reports[0][horizon].keys()
        result[horizon] = {
            key: statistics.median(
                float(report[horizon][key]) for report in reports
            )
            for key in keys
            if key != "N"
        }
    return result


root, base_commit, candidate_commit = sys.argv[1:]
baseline = aggregate(root, "baseline")
candidate = aggregate(root, "candidate")
print(f"# baseline={base_commit}")
print(f"# candidate={candidate_commit}")
print("# ratios are baseline/candidate; values above one favor the candidate")
print(
    "N,base_wall_ms,candidate_wall_ms,wall_ratio,"
    "base_kernel_ms,candidate_kernel_ms,kernel_ratio,"
    "feasibility_ratio,reduction_ratio,riccati_ratio,"
    "reconstruction_ratio,multiplier_ratio,"
    "base_cpp_kkt,candidate_cpp_kkt,base_cuda_kkt,candidate_cuda_kkt"
)
for horizon in sorted(baseline.keys() & candidate.keys()):
    base = baseline[horizon]
    cand = candidate[horizon]

    def ratio(field):
        return base[field] / cand[field]

    print(
        f"{horizon},"
        f"{base['cuda_wall_ms']:.6f},{cand['cuda_wall_ms']:.6f},"
        f"{ratio('cuda_wall_ms'):.4f},"
        f"{base['cuda_kernel_ms']:.6f},{cand['cuda_kernel_ms']:.6f},"
        f"{ratio('cuda_kernel_ms'):.4f},"
        f"{ratio('feasibility_ms'):.4f},{ratio('reduction_ms'):.4f},"
        f"{ratio('riccati_ms'):.4f},{ratio('reconstruction_ms'):.4f},"
        f"{ratio('multiplier_ms'):.4f},"
        f"{base['cpp_kkt_residual']:.6e},"
        f"{cand['cpp_kkt_residual']:.6e},"
        f"{base['cuda_kkt_residual']:.6e},"
        f"{cand['cuda_kkt_residual']:.6e}"
    )
PY

if [[ "${keep_output}" == "1" ]]; then
  echo "Raw reports and build trees retained in ${run_root}" >&2
else
  echo "Temporary build trees will now be removed." >&2
fi
