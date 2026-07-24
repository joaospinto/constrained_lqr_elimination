#!/usr/bin/env bash
# Compile two revisions with the same focused constrained-CPU benchmark,
# alternate process order, and report timing and primal/KKT residual evidence.
set -euo pipefail

repo_dir="$(git rev-parse --show-toplevel)"
base_revision="${CLQR_BASE_REVISION:-b92d127}"
candidate_revision="${CLQR_CANDIDATE_REVISION:-HEAD}"
repeats="${CLQR_BENCHMARK_REPEATS:-101}"
rounds="${CLQR_COMPARISON_ROUNDS:-7}"
precisions="${CLQR_PRECISIONS:-FP64 FP32}"
cxx="${CLQR_CXX:-c++}"

git -C "${repo_dir}" rev-parse --verify "${base_revision}^{commit}" >/dev/null
git -C "${repo_dir}" rev-parse --verify \
  "${candidate_revision}^{commit}" >/dev/null

comparison_dir="$(mktemp -d "${TMPDIR:-/tmp}/clqr-cpu-qr-compare.XXXXXX")"
cleanup() {
  rm -rf "${comparison_dir}"
}
trap cleanup EXIT

extract_revision() {
  local revision="$1"
  local destination="$2"
  mkdir -p "${destination}"
  git -C "${repo_dir}" archive "${revision}" | tar -xf - -C "${destination}"
}

extract_revision "${base_revision}" "${comparison_dir}/baseline"
extract_revision "${candidate_revision}" "${comparison_dir}/candidate"

build_revision() {
  local source_dir="$1"
  local output="$2"
  local precision="$3"
  local precision_flag="-UCLQR_USE_FLOAT"
  if [[ "${precision}" == "FP32" ]]; then
    precision_flag="-DCLQR_USE_FLOAT=1"
  elif [[ "${precision}" != "FP64" ]]; then
    echo "unsupported precision: ${precision}" >&2
    exit 2
  fi
  "${cxx}" -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Werror \
    "${precision_flag}" \
    -I"${source_dir}/include" \
    -I"${source_dir}/tests" \
    "${source_dir}/src/clqr.cc" \
    "${source_dir}/src/linalg.cc" \
    "${repo_dir}/benchmarks/cpu_constraint_revision_benchmark.cc" \
    -o "${output}"
}

for precision in ${precisions}; do
  build_revision "${comparison_dir}/baseline" \
    "${comparison_dir}/baseline-${precision}" "${precision}"
  build_revision "${comparison_dir}/candidate" \
    "${comparison_dir}/candidate-${precision}" "${precision}"
done

run_revision() {
  local label="$1"
  local precision="$2"
  local round="$3"
  "${comparison_dir}/${label}-${precision}" --repeats "${repeats}" \
    >"${comparison_dir}/${label}-${precision}-${round}.csv"
}

for precision in ${precisions}; do
  for ((round = 1; round <= rounds; ++round)); do
    if ((round % 2 == 1)); then
      run_revision baseline "${precision}" "${round}"
      run_revision candidate "${precision}" "${round}"
    else
      run_revision candidate "${precision}" "${round}"
      run_revision baseline "${precision}" "${round}"
    fi
  done
done

python3 - "${comparison_dir}" "${rounds}" "${precisions}" \
  "${base_revision}" "${candidate_revision}" <<'PY'
import csv
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rounds = int(sys.argv[2])
precisions = sys.argv[3].split()
base_revision = sys.argv[4]
candidate_revision = sys.argv[5]
labels = ("baseline", "candidate")

rows = {}
for precision in precisions:
    for label in labels:
        for round_index in range(1, rounds + 1):
            path = root / f"{label}-{precision}-{round_index}.csv"
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    rows.setdefault((precision, row["case"], label), []).append(
                        row
                    )

print(f"# baseline={base_revision}")
print(f"# candidate={candidate_revision}")
print("# ratios are baseline/candidate; values above one favor the candidate")
print(
    "precision,case,baseline_status,candidate_status,"
    "baseline_message,candidate_message,"
    "baseline_median_us,candidate_median_us,time_ratio,"
    "baseline_primal,candidate_primal,baseline_kkt,candidate_kkt"
)
for precision in precisions:
    cases = sorted(
        key[1] for key in rows if key[0] == precision and key[2] == "baseline"
    )
    for case in cases:
        baseline = rows[(precision, case, "baseline")]
        candidate = rows[(precision, case, "candidate")]
        baseline_statuses = {row["status"] for row in baseline}
        candidate_statuses = {row["status"] for row in candidate}
        baseline_messages = {row["message"] for row in baseline}
        candidate_messages = {row["message"] for row in candidate}
        if len(baseline_statuses) != 1 or len(candidate_statuses) != 1:
            raise RuntimeError(f"non-deterministic status for {precision} {case}")
        baseline_time = statistics.median(
            float(row["median_us"]) for row in baseline
        )
        candidate_time = statistics.median(
            float(row["median_us"]) for row in candidate
        )
        baseline_primal = max(
            float(row["primal_residual"]) for row in baseline
        )
        candidate_primal = max(
            float(row["primal_residual"]) for row in candidate
        )
        baseline_kkt = max(float(row["kkt_residual"]) for row in baseline)
        candidate_kkt = max(float(row["kkt_residual"]) for row in candidate)
        print(
            f"{precision},{case},"
            f"{next(iter(baseline_statuses))},"
            f"{next(iter(candidate_statuses))},"
            f"{next(iter(baseline_messages))},"
            f"{next(iter(candidate_messages))},"
            f"{baseline_time:.3f},{candidate_time:.3f},"
            f"{baseline_time / candidate_time:.4f},"
            f"{baseline_primal:.6e},{candidate_primal:.6e},"
            f"{baseline_kkt:.6e},{candidate_kkt:.6e}"
        )
PY
