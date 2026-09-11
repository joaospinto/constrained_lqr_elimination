"""Check executable isolation without accuracy-gating a third-party method."""

import csv
import subprocess
import sys


def check(executable, backend):
    options = [executable, "--suite", "smoke", "--case-index", "1", "--repeats", "1"]
    run = subprocess.run(options, check=True, capture_output=True, text=True, timeout=30)
    rows = list(csv.DictReader(line for line in run.stdout.splitlines()
                              if line and not line.startswith("#")))
    assert len(rows) == 1 and rows[0]["backend"] == backend, rows
    assert rows[0]["N"] == "17" and rows[0]["n"] == "8", rows
    assert None not in rows[0] and all(value is not None for value in rows[0].values()), rows
    assert rows[0]["status"] in ("ok", "inaccurate"), rows
    for other in {"clqr_cpu", "gen_riccati", "factor_graph"} - {backend}:
        rejected = subprocess.run(options + ["--backend", other],
                                  capture_output=True, text=True, timeout=30)
        assert rejected.returncode != 0, f"{backend} executable accepted {other}"
        assert "backend is not linked" in rejected.stderr, rejected.stderr
    print(f"{backend}: only its own backend is available")


if __name__ == "__main__":
    check(*sys.argv[1:])
