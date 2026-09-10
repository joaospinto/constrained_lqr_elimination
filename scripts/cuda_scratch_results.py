#!/usr/bin/env python3
"""Join automatic/forced-global scratch timings; retain numerical failures."""

import argparse
import csv
import math

from pathlib import Path

try:
    from scripts.paper_results import KEY_FIELDS, indexed, read_csv, numeric
except ModuleNotFoundError:
    from paper_results import KEY_FIELDS, indexed, read_csv, numeric


def compare(results):
    auto = indexed(read_csv(results / "auto/cuda_host.csv"), "clqr_cuda")
    global_ = indexed(read_csv(results / "global/cuda_host.csv"), "clqr_cuda")
    if auto.keys() != global_.keys():
        raise ValueError("scratch runs contain different cases")
    rows = []
    metrics = ("status", "median_ms", "p10_ms", "p90_ms", "kernel_ms",
               "setup_solve_ms", "primal_error", "dual_error_inf", "kkt_inf")
    for key, a in auto.items():
        g = global_[key]
        row = {field: a[field] for field in KEY_FIELDS}
        for mode, measurement in (("auto", a), ("global", g)):
            row.update({f"{mode}_{field}": measurement[field] for field in metrics})
        for metric, label in (("median_ms", "wall"), ("kernel_ms", "kernel")):
            numerator, denominator = numeric(g, metric), numeric(a, metric)
            row[f"global_over_auto_{label}"] = (
                numerator / denominator
                if math.isfinite(numerator) and math.isfinite(denominator) and denominator > 0
                else "")
        rows.append(row)
    with (results / "scratch_comparison.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote scratch_comparison.csv; ratios above one mean forced-global is slower.")
    print("Raw CPU times, all errors, options, and platform details are in auto/ and global/.")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    compare(parser.parse_args().results)
