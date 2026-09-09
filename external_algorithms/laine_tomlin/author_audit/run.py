#!/usr/bin/env python3
"""Run original-author cases in isolated processes; retain all outcomes."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--implicit", action="store_true")
    parser.add_argument("--multipliers", action="store_true")
    args = parser.parse_args()
    executable = args.executable.resolve()
    cases = subprocess.check_output([executable, "--list"], text=True).splitlines()
    rows = []
    flags = (["--implicit"] if args.implicit else []) + (["--multipliers"] if args.multipliers else [])
    for case in cases:
        try:
            r = subprocess.run([executable, case, *flags], capture_output=True,
                               text=True, timeout=args.timeout)
            row = dict(case=case, returncode=r.returncode,
                       stdout=r.stdout.strip(), stderr=r.stderr.strip())
        except subprocess.TimeoutExpired:
            row = dict(case=case, returncode=124, stdout="", stderr="timeout")
        rows.append(row)
        print(row["stdout"] or f"{case},process_error,{row['returncode']}", flush=True)
    with args.output.open("x") as output:
        json.dump(rows, output, indent=2)


if __name__ == "__main__":
    main()
