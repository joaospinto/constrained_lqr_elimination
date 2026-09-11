#!/usr/bin/env python3
"""Run each paper case in isolation, retaining completed rows after crashes."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

if __package__:
    from . import benchmark_progress
else:
    # Direct execution also works with PYTHONSAFEPATH/-P (as in Bazel tests).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import benchmark_progress


IDENTITY = ("family", "N", "n", "m", "mixed_rows", "state_rows")


def run(command, *, manifest, backends, name, output, errors, jax=False,
        seed=20260907, interval=30.0):
    cases = json.loads(Path(manifest).read_text())
    # Fast solves can produce sample lists larger than CSV's 128 KiB default.
    csv.field_size_limit(sys.maxsize)
    if not cases or not backends or (jax and len(backends) != 1):
        raise ValueError("a nonempty case list and backend list are required")
    if len(set(backends)) != len(backends):
        raise ValueError("duplicate backends")
    case_dir = Path(output).parent / (name + "_cases")
    case_dir.mkdir()
    rows = []
    failed = False
    started = time.monotonic()
    total = len(cases) * len(backends)

    def report(message):
        print(f"[{name}] {message} (sweep elapsed {time.monotonic() - started:.1f}s)",
              flush=True)

    report(f"SWEEP START: {total} cases; backends={','.join(backends)}")
    with Path(errors).open("w") as diagnostics:
        for backend in backends:
            for case in cases:
                label = f"{name}/{backend}/{case['index']}"
                raw = case_dir / f"{backend}-{case['index']}.csv"
                log = raw.with_suffix(".stderr")
                invocation = [*command, "--case-index", str(case["index"]),
                              "--seed", str(seed)]
                if not jax:
                    invocation += ["--backend", backend]
                report(f"CASE {len(rows) + 1}/{total}: {backend} "
                       f"N={case['N']} n={case['n']} m={case['m']}")
                code = benchmark_progress.run(invocation, name=label,
                                              output=raw, errors=log, interval=interval)
                diagnostics.write(log.read_text(errors="replace"))
                row = dict(backend=backend, seed=seed,
                           **{field: case[field] for field in IDENTITY})
                try:
                    if code:
                        raise ValueError(f"child exited with code {code}; see {log.name}")
                    measured = list(csv.DictReader(
                        line for line in raw.read_text().splitlines()
                        if line and not line.startswith("#")))
                    if len(measured) != 1 or any(str(measured[0].get(k)) != str(v)
                                                 for k, v in row.items()):
                        raise ValueError("child must return exactly its requested backend/case/seed")
                    if None in measured[0] or any(v is None for v in measured[0].values()):
                        raise ValueError("truncated or malformed CSV row")
                    row = measured[0]
                except (ValueError, csv.Error) as error:
                    failed = True
                    row.update(status="failed", repeats=0, diagnostic=str(error))
                    diagnostics.write(f"{label}: {error}\n")
                diagnostics.flush()
                rows.append(row)
                # Tiny result metadata only; never hold problem arrays here.
                # Replace atomically after each case, including a fatal exit.
                fields = list(dict.fromkeys(key for result in rows for key in result))
                partial = Path(output).with_suffix(".partial")
                with partial.open("w") as destination:
                    writer = csv.DictWriter(destination, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                partial.replace(output)
                report(f"SAVED {len(rows)}/{total}: {backend} "
                       f"status={row.get('status', 'unknown')}")
    report(f"SWEEP COMPLETE: {len(rows)}/{total} cases saved to {output}")
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--backends", required=True, nargs="+")
    parser.add_argument("--name", required=True)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("--jax", action="store_true")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    interval = float(os.environ.get("CLQR_PROGRESS_INTERVAL_SECONDS", "30"))
    if not command or not math.isfinite(interval) or interval <= 0:
        parser.error("a command and a finite positive heartbeat interval are required")
    return run(command, manifest=args.manifest, backends=args.backends,
               name=args.name, output=args.stdout, errors=args.stderr,
               jax=args.jax, seed=args.seed, interval=interval)


if __name__ == "__main__":
    raise SystemExit(main())
