#!/usr/bin/env python3
"""Check benchmark data integrity and report measurements without accuracy gates."""

import argparse
import csv
import io
import json
import math
from pathlib import Path


KEY_FIELDS = ("family", "N", "n", "m", "mixed_rows", "state_rows", "seed")
SOURCES = {
    "clqr_cpu": "cpu_round1.csv",
    "gen_riccati": "references.csv",
    "factor_graph": "references.csv",
    "clqr_jax_cpu": "jax_cpu.csv",
    "clqr_cuda": "cuda_host.csv",
    "clqr_jax_cuda": "cuda_resident.csv",
}
OPTIONAL_SOURCES = {
    "laine_author": "laine_round1.csv",
}
PRIMAL_ONLY = {"factor_graph", "laine_author"}


def read_csv(path):
    return list(csv.DictReader(line for line in path.read_text().splitlines()
                               if line and not line.startswith("#")))


def key(row):
    return (row["family"], *(int(row[field]) for field in KEY_FIELDS[1:]))


def numeric(row, field):
    value = row.get(field, "")
    return float(value) if value else math.nan


def indexed(rows, backend):
    result = {}
    for row in rows:
        if row["backend"] != backend:
            continue
        identity = key(row)
        if identity in result:
            raise ValueError(f"duplicate {backend} case: {identity}")
        status = row["status"]
        if status not in ("ok", "inaccurate", "unsupported", "failed"):
            raise ValueError(f"unknown benchmark status: {status}")
        if status in ("ok", "inaccurate"):
            for field in ("median_ms", "p10_ms", "p90_ms"):
                if not math.isfinite(numeric(row, field)) or numeric(row, field) < 0:
                    raise ValueError(f"invalid {backend} {field}: {identity}")
            if not numeric(row, "p10_ms") <= numeric(row, "median_ms") <= numeric(row, "p90_ms"):
                raise ValueError(f"unordered timing quantiles: {identity}")
            errors = ["primal_error", "relative_objective_error", "feasibility_inf"]
            if backend not in PRIMAL_ONLY:
                errors.append("kkt_inf")
            if "planted_dual_stationarity_inf" in row:
                errors.append("planted_dual_stationarity_inf")
            for field in errors:
                if row.get(field, "") == "" or numeric(row, field) < 0:
                    raise ValueError(f"missing/invalid {backend} {field}: {identity}")
                if status == "ok" and not math.isfinite(numeric(row, field)):
                    raise ValueError(f"nonfinite successful {backend} {field}: {identity}")
            if backend == "clqr_cuda" and (not math.isfinite(numeric(row, "kernel_ms")) or
                                            numeric(row, "kernel_ms") < 0):
                raise ValueError(f"invalid CUDA kernel time: {identity}")
            if status == "ok" and (numeric(row, "primal_error") >= 1e-6 or
                                    numeric(row, "relative_objective_error") >= 1e-8 or
                                    numeric(row, "feasibility_inf") >= 1e-8 or
                                    (backend not in PRIMAL_ONLY and numeric(row, "kkt_inf") >= 1e-8) or
                                    ("planted_dual_stationarity_inf" in row and
                                     numeric(row, "planted_dual_stationarity_inf") >= 1e-8)):
                raise ValueError(f"status contradicts accuracy measurements: {identity}")
        result[identity] = row
    if not result:
        raise ValueError(f"no rows for {backend}")
    return result


def validate_cases(data, manifest):
    identities = set(data["clqr_cpu"])
    if any(set(rows) != identities for rows in data.values()):
        raise ValueError("backends have different cases/seeds; refusing to combine them")
    if len({identity[-1] for identity in identities}) != 1:
        raise ValueError("one run must use one declared seed")
    declared = [(row["family"], *(int(row[field]) for field in KEY_FIELDS[1:-1]))
                for row in manifest]
    if len(set(declared)) != len(declared) or {identity[:-1] for identity in identities} != set(declared):
        raise ValueError("measurements do not match the C++ fixture manifest")
    return identities


def time_cell(row, field="median_ms"):
    if row["status"] == "unsupported":
        return r"\textsc{oom}"
    if row["status"] == "failed":
        return r"\textsc{fail}"
    value = numeric(row, field)
    if not math.isfinite(value):
        raise ValueError(f"missing measured {field}")
    suffix = r"$^{\dagger}$" if row["status"] == "inaccurate" else ""
    return f"{value:.3f}" + suffix


def selected(identity):
    family, horizon, n, _, mixed, state, _ = identity
    return ((family == "combined" and horizon in (128, 2048, 32768)) or
            (family == "dimension" and n in (16, 32, 64)) or
            (family == "mixed_rows" and mixed in (0, 2, 6)) or
            (family == "state_rows" and state in (2, 6)))


def comparison_table(data):
    """A predeclared subset; the complete sweep remains in the run CSVs."""
    output = io.StringIO()
    print(r"\begin{table*}[!t]", file=output)
    print(r"\centering\footnotesize\setlength{\tabcolsep}{3pt}", file=output)
    print(r"\caption{Same-host FP64 dense comparisons, medians in ms. "
          r"$p_m,p_s$ are mixed/state equality counts; $m=n/2$. "
          r"Each call refactors. CPU columns use prepared representations; "
          r"CUDA wall includes host packing/transfers, whereas JAX retains numerical inputs/outputs on the GPU. "
          r"$\dagger$: accuracy threshold exceeded; \textsc{oom}: out of shared memory; "
          r"\textsc{fail}: unsuccessful solve.}", file=output)
    print(r"\label{tab:dense-comparisons}", file=output)
    print(r"\begin{tabular}{|r|r|r|r|r|r|r|r|r|r|}\hline", file=output)
    print(r"\multicolumn{4}{|c|}{Problem} & \multicolumn{3}{c|}{CPU} & "
          r"\multicolumn{3}{c|}{Tesla P100} \\\hline", file=output)
    print(r"$N$ & $n$ & $p_m$ & $p_s$ & Ours & Vanroye & Factor graph & "
          r"Kernels & Wall & JAX wall \\\hline", file=output)
    previous = None
    for identity in data["clqr_cpu"]:
        if not selected(identity):
            continue
        family, horizon, n, _, mixed, state, _ = identity
        group = "rows" if family.endswith("_rows") else family
        if previous and previous != group:
            print(r"\hline", file=output)
        previous = group
        cells = [str(horizon), str(n), str(mixed), str(state)]
        cells += [time_cell(data[backend][identity]) for backend in
                  ("clqr_cpu", "gen_riccati", "factor_graph")]
        cells += [time_cell(data["clqr_cuda"][identity], "kernel_ms"),
                  time_cell(data["clqr_cuda"][identity]),
                  time_cell(data["clqr_jax_cuda"][identity])]
        print(" & ".join(cells) + r" \\", file=output)
    print(r"\hline\end{tabular}\end{table*}", file=output)
    return output.getvalue()


def summarize(data):
    summary = {}
    for backend, cases in data.items():
        counts = {status: sum(row["status"] == status for row in cases.values())
                  for status in ("ok", "inaccurate", "unsupported", "failed")}
        maxima, nonfinite = {}, {}
        fields = ["primal_error", "relative_objective_error", "feasibility_inf", "kkt_inf"]
        if any("planted_dual_stationarity_inf" in row for row in cases.values()):
            fields.append("planted_dual_stationarity_inf")
        for field in fields:
            values = [numeric(row, field) for row in cases.values()]
            finite = [value for value in values if math.isfinite(value)]
            maxima[field] = max(finite) if finite else None
            nonfinite[field] = sum(row["status"] in ("ok", "inaccurate") and
                                   row.get(field, "") != "" and
                                   not math.isfinite(numeric(row, field))
                                   for row in cases.values())
        # Primal-only APIs do not return multipliers; their NaN is missing
        # information, not a measured numerical failure.
        if backend in PRIMAL_ONLY:
            nonfinite.pop("kkt_inf")
        summary[backend] = dict(counts=counts, maxima=maxima,
                               nonfinite_measurements=nonfinite)
    return summary


def scientific(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("a residual must be finite and nonnegative")
    if value == 0:
        return "$0$"
    mantissa, exponent = f"{value:.2e}".split("e")
    return "$" + mantissa + rf"{{\times}}10^{{{int(exponent)}}}$"


def original_rows(rows):
    output = []
    for row in rows:
        values = [numeric(row, field) for field in
                  ("cpp_cpu_ms", "cpp_kkt_residual", "cuda_kernel_ms",
                   "cuda_wall_ms", "cuda_kkt_residual")]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("original table contains missing/invalid measurements")
        cpu, cpu_kkt, kernel, wall, gpu_kkt = values
        if min(cpu, kernel, wall) <= 0:
            raise ValueError("original table timings must be positive")
        output.append(f"{int(row['N'])} & {cpu:.3f} & {scientific(cpu_kkt)} & "
                      f"{kernel:.3f} & {wall:.3f} & {scientific(gpu_kkt)}" + r" \\")
    return "\n".join(output) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--suite", default="all",
                        choices=("all", "smoke", "horizon", "dimension", "constraints"))
    args = parser.parse_args(argv)
    data = {}
    for backend, name in SOURCES.items():
        if "cuda" in backend and not args.cuda:
            continue
        data[backend] = indexed(read_csv(args.results / name), backend)
    for backend, name in OPTIONAL_SOURCES.items():
        if (args.results / name).is_file():
            data[backend] = indexed(read_csv(args.results / name), backend)
    identities = validate_cases(data, json.loads((args.results / "cases.json").read_text()))
    report = summarize(data)
    (args.results / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.cuda:
        # This table is explicitly labeled Tesla P100: do not silently insert
        # another GPU/host's measurements into the paper.
        platform = (args.results / "platform.txt").read_text()
        if "Tesla P100" not in platform or "Linux" not in platform:
            raise ValueError("the paper table requires the recorded Linux/P100 host")
        if args.suite == "all":
            if sum(selected(identity) for identity in identities) != 11:
                raise ValueError("paper table requires the complete all-suite run")
            (args.results / "comparison_table.tex").write_text(comparison_table(data))
        original = read_csv(args.results / "original_table.csv")
        expected = [2 ** exponent for exponent in range(5, 15)]
        if [int(row["N"]) for row in original] != expected:
            raise ValueError("original table must include every horizon 32..16384")
        if any((int(row["n"]), int(row["m"]), int(row["p"])) != (8, 4, 2)
               for row in original):
            raise ValueError("original table has unexpected local dimensions")
        (args.results / "original_table_rows.tex").write_text(original_rows(original))
        last = original[-1]
        wall = numeric(last, "cuda_wall_ms")
        kernel = numeric(last, "cuda_kernel_ms")
        packing = sum(numeric(last, field) for field in
                      ("input_pack_ms", "upload_ms", "download_ms"))
        report["original_table"] = dict(
            horizon=int(last["N"]), kernel_speedup=numeric(last, "cpp_cpu_ms") / kernel,
            wall_speedup=numeric(last, "cpp_cpu_ms") / wall, wall_kernel_gap_ms=wall - kernel,
            pack_transfer_share_of_gap=packing / (wall - kernel) if wall > kernel else None,
            max_kkt_inf=max(numeric(row, field) for row in original
                            for field in ("cpp_kkt_residual", "cuda_kkt_residual")))
        (args.results / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    # Numerical outcomes are measurements, including third-party failures.
    # Missing/malformed data still raises above; accuracy is not an exit gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
