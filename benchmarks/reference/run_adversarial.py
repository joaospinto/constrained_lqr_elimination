#!/usr/bin/env python3
"""Compare native CPU solvers on the shared unit-test fixtures, without gates."""

import argparse
import csv
import io
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--independent", action="store_true",
                        help="use the corrected Laine-Tomlin implementation's independent fixtures")
    args = parser.parse_args()
    native = args.build.resolve() / "clqr_adversarial_benchmark"
    if not native.is_file():
        parser.error("build clqr_adversarial_benchmark first")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    extra = ["--independent"] if args.independent else []
    cases = subprocess.check_output([native, "--list", *extra], text=True).splitlines()
    variants = [(native, "clqr_cpu", "clqr_cpu"),
                (native, "gen_riccati", "gen_riccati")]
    for target, backends in (
        ("clqr_adversarial_factor_graph_benchmark", ("factor_graph",)),
        ("clqr_adversarial_laine_benchmark", ("laine_author",)),
        ("clqr_adversarial_laine_corrected_benchmark", ("laine_corrected",)),
    ):
        executable = args.build.resolve() / target
        if executable.is_file():
            variants.extend((executable, backend, backend) for backend in backends)
    args.output.mkdir(parents=True, exist_ok=False)
    for round_index in (1, 2):
        rows = []
        ordered = variants if round_index == 1 else variants[::-1]
        for case in cases:
            for executable, backend, label in ordered:
                command = [executable, "--case", case, "--backend", backend,
                           "--repeats", str(args.repeats), *extra]
                try:
                    result = subprocess.run(command, capture_output=True,
                                            text=True, timeout=30)
                    stdout, stderr = result.stdout, result.stderr
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    stdout, stderr, code = "", "30-second timeout", 124
                parsed = list(csv.DictReader(io.StringIO(stdout))) if code == 0 else []
                if len(parsed) == 1 and parsed[0].get("backend") == label:
                    row = parsed[0]
                else:
                    row = dict(backend=label, case=case,
                               status="timeout" if code == 124 else "process_error",
                               diagnostic=f"exit={code}: {stderr.strip()}")
                rows.append(row)
                # Keep crash diagnostics, but do not abort unrelated solvers/cases.
                if stderr or code:
                    (args.output / f"round{round_index}-{label}-{case}.log").write_text(
                        stdout + "\n" + stderr)
            print(f"round {round_index}: {case}", flush=True)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (args.output / f"round{round_index}.csv").open("w") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
