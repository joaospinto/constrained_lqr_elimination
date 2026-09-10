"""Exercise solver-dual reporting end to end, without accuracy-gating a reference."""

import csv
import math
import subprocess
import sys


def check(executable):
    run = subprocess.run([executable, "--suite", "smoke", "--repeats", "1"],
                         check=True, capture_output=True, text=True, timeout=30)
    rows = list(csv.DictReader(line for line in run.stdout.splitlines()
                              if line and not line.startswith("#")))
    author = [row for row in rows if row["backend"] == "laine_author"]
    if len(author) != 4:
        raise AssertionError("author benchmark did not emit all smoke cases")
    for row in author:
        feasibility = float(row["feasibility_inf"])
        stationarity = float(row["stationarity_inf"])
        kkt = float(row["kkt_inf"])
        # The audit maps nonfinite returned multipliers to infinity. NaN here
        # instead means that no dual result was supplied. Check reporting, not
        # that the third-party solver meets an accuracy threshold.
        if any(math.isnan(value) for value in (feasibility, stationarity, kkt)):
            raise AssertionError(f"missing measured author residuals: {row}")
        if kkt != max(feasibility, stationarity):
            raise AssertionError(f"KKT does not use returned duals: {row}")
        if kkt >= 1e-8 and row["status"] != "inaccurate":
            raise AssertionError(f"author dual error was hidden: {row}")
    # The pinned original leaves a zero-horizon multiplier at zero. The
    # fixture's optimal dual would instead certify the primal: ensure it is
    # not being substituted for the author's output.
    zero = next(row for row in author if row["family"] == "zero")
    if not float(zero["kkt_inf"]) > 1e-3 or not float(zero["planted_dual_stationarity_inf"]) < 1e-12:
        raise AssertionError("solver duals and the planted certificate were conflated")
    print("Original Laine benchmark reports timed primal/dual solves and actual KKT errors")


if __name__ == "__main__":
    check(sys.argv[1])
