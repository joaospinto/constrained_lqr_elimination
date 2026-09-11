"""Synchronized native JAX-FFI timing with inputs/outputs resident on device."""

import argparse
import csv
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import time

import jax
import numpy as np

from benchmarks import paper_fixture


def _runfiles():
    value = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
    if value and Path(value).is_dir():
        return Path(value)
    # The two-stage py_binary launcher may set argv[0] to the Python entry
    # point inside the runfiles tree rather than the top-level executable.
    for source in (Path(__file__).absolute(), Path(sys.argv[0]).absolute()):
        for parent in source.parents:
            if parent.name.endswith(".runfiles") and parent.is_dir():
                return parent
    raise RuntimeError("run this benchmark with its Bazel launcher")


def _extension(root, name):
    candidates = list(root.rglob(name + ".so")) or list(root.rglob(name + ".*.so"))
    if not candidates:
        raise RuntimeError(f"missing {name} extension in runfiles")
    spec = importlib.util.spec_from_file_location(name, candidates[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _capacity_limits(path, seed):
    if path is None:
        return set()
    with path.open() as source:
        rows = csv.DictReader(line for line in source if not line.startswith("#"))
        return {(row["family"], *(int(row[field]) for field in
                  ("N", "n", "m", "mixed_rows", "state_rows")))
                for row in rows if row["backend"] == "clqr_cuda" and
                row["status"] == "unsupported" and int(row["seed"]) == seed}


def main(default_platform="cpu", argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("cpu", "cuda"), default=default_platform)
    parser.add_argument("--suite", default="all")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--capacity-report", type=Path,
                        help="same-run native CUDA CSV; skip sizes rejected by its workspace planner")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.capacity_report and args.platform != "cuda":
        parser.error("a capacity report applies only to the CUDA backend")
    capacity_limits = _capacity_limits(args.capacity_report, args.seed)
    root = _runfiles()
    _extension(root, "_clqr")
    cpu = _extension(root, "_clqr_jax_cpu")
    if np.dtype(cpu.scalar_dtype) != np.dtype("float64"):
        raise RuntimeError("paper benchmarks require --config=fp64")
    cuda = _extension(root, "_clqr_cuda") if args.platform == "cuda" else None
    package = next(root.rglob("python/clqr/__init__.py"))
    sys.path.insert(0, str(package.parents[1]))
    module = importlib.import_module("clqr.jax")
    fixture = next(root.rglob("clqr_paper_fixture"))
    jax.config.update("jax_enable_x64", True)
    device = jax.devices("gpu" if cuda else "cpu")[0]
    if cuda and not module.cuda_registered:
        raise RuntimeError("native CLQR CUDA FFI is not registered")

    print(f"# native JAX FFI, FP64, device={device}; inputs and outputs remain resident during timing")
    print("# compile, upload, host validation, and download are excluded; each solve includes fresh numerical factorization")
    print("# at least 3 warmups and 100 ms; each repetition blocks on all outputs")
    print("# primal_error and dual_error_inf: absolute infinity-norm errors against the known planted optimum")
    fields = ["backend", "family", "N", "n", "m", "mixed_rows", "state_rows", "seed",
              "status", "repeats", "median_ms", "p10_ms", "p90_ms", "primal_error",
              "relative_objective_error", "feasibility_inf", "stationarity_inf", "kkt_inf",
              "scalar_device_to_host_bytes", "scalar_host_to_device_bytes", "metadata_device_to_host_bytes",
              "planted_dual_stationarity_inf", "dual_error_inf"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    protocol_failed = False
    cases = paper_fixture.cases(fixture, args.suite)
    for index, case in enumerate(cases, 1):
        started = time.perf_counter()

        def progress(phase):
            print(f"[case {index}/{len(cases)}] clqr_jax_{args.platform} "
                  f"N={case['N']} n={case['n']} m={case['m']} {phase} "
                  f"(elapsed {time.perf_counter() - started:.1f}s)",
                  file=sys.stderr, flush=True)

        row = dict(backend="clqr_jax_" + args.platform, seed=args.seed, repeats=args.repeats,
                   **{key: value for key, value in case.items() if key != "index"})
        identity = (case["family"], *(case[field] for field in
                    ("N", "n", "m", "mixed_rows", "state_rows")))
        if identity in capacity_limits:
            # The host and FFI paths use the same native workspace planner.
            # This is an explicit skip, not a claimed successful JAX run.
            row.update(status="unsupported", repeats=0)
            print(f"# skipped {identity}: same-run native CUDA workspace capacity rejection")
            writer.writerow(row)
            sys.stdout.flush()
            progress("DONE status=unsupported (native CUDA capacity report)")
            continue
        solve = inputs = result = host = p = expected_x = expected_u = expected_dual = None
        try:
            progress("generating fixture")
            p, expected_x, expected_u, expected_dual = paper_fixture.problem(fixture, args.suite, case["index"], args.seed)
            progress("packing and uploading inputs")
            host = module.pack_problem(p)
            inputs = jax.device_put(host, device)
            jax.block_until_ready(inputs)
            progress("compiling JAX solve")
            solve = jax.jit(module.solve).lower(inputs).compile()
            progress("warmup")
            start = time.perf_counter()
            warmed = 0
            while warmed < 3 or time.perf_counter() - start < 0.1:
                result = jax.block_until_ready(solve(inputs))
                warmed += 1
            times = []
            progress(f"timing {args.repeats} resident solves")
            for _ in range(args.repeats):
                start = time.perf_counter_ns()
                result = jax.block_until_ready(solve(inputs))
                times.append((time.perf_counter_ns() - start) / 1e6)
            progress("validation")
            # Read the handler's audit before transferring the results for validation.
            if cuda:
                transfers = cuda.last_transfer_audit()
                expected_metadata = host.factors.dimensions.nbytes
                if transfers["scalar_device_to_host_bytes"] or transfers["scalar_host_to_device_bytes"]:
                    protocol_failed = True
                    raise RuntimeError("unexpected scalar transfer inside resident solve")
                if transfers["metadata_device_to_host_bytes"] != expected_metadata:
                    protocol_failed = True
                    raise RuntimeError("unexpected metadata transfer size")
                row.update({key: transfers[key] for key in fields if key.endswith("_bytes")})
            result = jax.device_get(result)
            if int(result.diagnostics[0]) != 0:
                raise RuntimeError(f"solver diagnostics {result.diagnostics.tolist()}")
            residuals = paper_fixture.audit(p, result, expected_x, expected_u, expected_dual)
            row.update(residuals)
            passed = (residuals["primal_error"] < 1e-6 and
                      residuals["relative_objective_error"] < 1e-8 and residuals["kkt_inf"] < 1e-8 and
                      residuals["planted_dual_stationarity_inf"] < 1e-8)
            times.sort()
            row.update(status="ok" if passed else "inaccurate", median_ms=times[len(times) // 2],
                       p10_ms=times[len(times) // 10], p90_ms=times[(len(times) - 1) * 9 // 10])
        except Exception as error:
            row["status"] = "failed"
            print(f"# {case['family']} N={case['N']} n={case['n']}: {str(error).replace(chr(10), ' ')}", flush=True)
        finally:
            # Release inputs/executables on failures as well as successful runs.
            del solve, inputs, result, host, p, expected_x, expected_u, expected_dual
            jax.clear_caches()
        writer.writerow(row)
        sys.stdout.flush()
        progress(f"DONE status={row['status']}")
    # Accuracy and solver rejections are recorded above. A violated timing/
    # transfer protocol is a harness failure, not a numerical measurement.
    return int(protocol_failed)
