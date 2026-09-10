#!/usr/bin/env python3
"""Run the paper driver with a private, disposable cache and archive its output."""

import argparse
from pathlib import Path
import shutil
import subprocess
import os
import tempfile
import zipfile


def run(source, work_dir, *, cuda=True, suite="all", repeats=11, jobs=4,
        scratch_comparison=False):
    if shutil.disk_usage(work_dir).free < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free is required before starting the comparison")
    root = Path(tempfile.mkdtemp(prefix="clqr-paper-", dir=work_dir))
    results, cache = root / "results", root / "cache"
    env = dict(os.environ, CLQR_PAPER_CACHE_DIR=str(cache),
               CLQR_PAPER_BAZEL_ROOT=str(cache / "bazel"),
               BAZELISK_HOME=str(cache / "bazelisk"),
               CLQR_PAPER_SUITE=suite, CLQR_BENCHMARK_REPEATS=str(repeats),
               CLQR_JOBS=str(jobs),
               PYTHONDONTWRITEBYTECODE="1")
    driver = "cuda_scratch_benchmarks.sh" if scratch_comparison else "paper_benchmarks.sh"
    command = ["bash", str(source / "scripts" / driver), str(results)]
    if cuda:
        command.append("--cuda")
    print(f"Run directory: {root}", flush=True)
    completed = False
    code = 1
    try:
        with (root / "driver.log").open("w") as log:
            with subprocess.Popen(command, cwd=source, env=env, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                    completed = True
                except BaseException:
                    # Do not delete a cache after an interrupted process: its
                    # children may still be exiting. Normal driver exits run
                    # the Bazel shutdown trap before returning here.
                    process.terminate()
                    raise
    finally:
        archive = root / "paper-results.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
            # Never traverse the potentially large dependency/build cache.
            candidates = [root / "driver.log", *sorted(results.rglob("*"))]
            for path in candidates:
                relative = path.relative_to(root)
                if path.is_file() and "clqr_tools" not in relative.parts:
                    output.write(path, relative)
        if completed and cache.is_dir():
            shutil.rmtree(cache)
            print("Removed this run's downloaded dependencies and build cache; results retained.")
        print(f"Driver exit code: {code}")
        print(f"Free disk: {shutil.disk_usage(work_dir).free / 1024**3:.1f} GiB")
        print(f"Please return: {archive}", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--scratch-comparison", action="store_true",
                        help="compare automatic and forced-global CUDA scratch, without external solvers")
    parser.add_argument("--suite",
                        choices=("all", "smoke", "horizon", "dimension", "constraints", "scratch"))
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.repeats < 1 or args.jobs < 1:
        parser.error("repeats and jobs must be positive")
    if args.scratch_comparison and args.cpu_only:
        parser.error("--scratch-comparison requires CUDA")
    suite = args.suite or ("scratch" if args.scratch_comparison else "all")
    return run(Path(__file__).resolve().parents[1], args.work_dir.resolve(),
               cuda=not args.cpu_only, suite=suite, repeats=args.repeats, jobs=args.jobs,
               scratch_comparison=args.scratch_comparison)


if __name__ == "__main__":
    raise SystemExit(main())
