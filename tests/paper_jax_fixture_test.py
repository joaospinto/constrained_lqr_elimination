import contextlib
import csv
import io
import copy
import math
from types import SimpleNamespace
from unittest import mock

import jax
import numpy as np
from benchmarks import paper_fixture
from benchmarks.paper_jax_benchmark import _runfiles

from benchmarks.paper_jax_benchmark import _capacity_limits, main


def check_capacity_report():
    class Report:
        def open(self):
            return io.StringIO(
                "# native workspace planner result\n"
                "backend,family,N,n,m,mixed_rows,state_rows,seed,status\n"
                "clqr_cuda,dimension,128,64,32,8,8,7,unsupported\n"
                "clqr_cuda,dimension,128,32,16,4,4,7,failed\n"
                "clqr_cpu,dimension,128,16,8,2,2,7,unsupported\n"
                "clqr_cuda,dimension,128,24,12,3,3,8,unsupported\n")

    assert _capacity_limits(None, 7) == set()
    assert _capacity_limits(Report(), 7) == {("dimension", 128, 64, 32, 8, 8)}

if __name__ == "__main__":
    check_capacity_report()
    output = io.StringIO()
    progress = io.StringIO()
    compiled_calls = []
    original_jit = jax.jit

    def counting_jit(function, *args, **kwargs):
        wrapped = original_jit(function, *args, **kwargs)
        if function.__module__ != "clqr.jax" or function.__name__ != "solve":
            return wrapped

        def lower(*inputs, **options):
            lowered = wrapped.lower(*inputs, **options)

            def compile_counted():
                compiled = lowered.compile()
                compiled_calls.append(0)
                index = len(compiled_calls) - 1

                def counted(*values):
                    compiled_calls[index] += 1
                    return compiled(*values)

                return counted

            return SimpleNamespace(compile=compile_counted)

        return SimpleNamespace(lower=lower)

    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(progress), \
            mock.patch.object(jax, "jit", side_effect=counting_jit):
        code = main("cpu", ["--suite", "smoke", "--repeats", "1"])
    print(output.getvalue(), end="")
    rows = list(csv.DictReader(line for line in output.getvalue().splitlines()
                               if line and not line.startswith("#")))
    # Benchmarks report numerical outcomes without failing their process.
    # This regression test must still enforce correctness on its smoke cases.
    assert code == 0
    assert len(rows) == 4
    # Exactly one warmup and one measured invocation, including the tiny
    # zero-horizon case: no minimum-duration warmup loop remains.
    assert compiled_calls == [2, 2, 2, 2], compiled_calls
    assert progress.getvalue().count("warmup (1 solve)") == len(rows)
    assert all(row["status"] == "ok" for row in rows), rows
    assert all(float(row["dual_error_inf"]) < 1e-8 for row in rows), rows
    assert "[case 1/4] clqr_jax_cpu N=1 n=4 m=2 generating fixture" in progress.getvalue()
    assert "[case 4/4]" in progress.getvalue()
    assert "compiling JAX solve" in progress.getvalue()
    assert "DONE status=ok" in progress.getvalue()
    assert "[case" not in output.getvalue()
    # An unsuccessful warmup must be reported, not repeated and timed.
    failed_output = io.StringIO()
    failed_progress = io.StringIO()
    failed_solve = mock.Mock(return_value=SimpleNamespace(
        diagnostics=np.array([3, 0, 0], dtype=np.int32)))
    failed_jit = SimpleNamespace(lower=lambda *a: SimpleNamespace(
        compile=lambda: failed_solve))
    with contextlib.redirect_stdout(failed_output), contextlib.redirect_stderr(failed_progress), \
            mock.patch.object(jax, "jit", return_value=failed_jit), \
            mock.patch.object(jax, "block_until_ready", side_effect=lambda value: value):
        code = main("cpu", ["--suite", "smoke", "--case-index", "0", "--repeats", "2"])
    assert code == 0
    assert failed_solve.call_count == 1
    assert "warmup solver diagnostics [3, 0, 0]" in failed_output.getvalue()
    assert "timing resident solves" not in failed_progress.getvalue()
    failed_rows = list(csv.DictReader(line for line in failed_output.getvalue().splitlines()
                                    if line and not line.startswith("#")))
    assert len(failed_rows) == 1 and failed_rows[0]["status"] == "failed"
    assert failed_rows[0]["repeats"] == "0"
    fixture = next(_runfiles().rglob("clqr_paper_fixture"))
    for index in (0, 2, 3):
        p, x, u, dual = paper_fixture.problem(fixture, "smoke", index, 20260907)
        optimum = SimpleNamespace(states=x, controls=u, **vars(dual))
        metrics = paper_fixture.audit(p, optimum, x, u, dual)
        assert metrics["primal_error"] == metrics["dual_error_inf"] == 0
        assert metrics["kkt_inf"] < 1e-12, metrics
        changed = copy.deepcopy(optimum)
        changed.initial_multiplier[0] += 0.25
        metrics = paper_fixture.audit(p, changed, x, u, dual)
        assert abs(metrics["dual_error_inf"] - 0.25) < 1e-15
        assert metrics["primal_error"] == 0
        assert metrics["stationarity_inf"] > 0.24
        # Nonfinite outputs are recorded as errors, not dropped or exceptions
        # that erase timing/other measurements.
        changed.initial_multiplier[0] = np.nan
        metrics = paper_fixture.audit(p, changed, x, u, dual)
        assert math.isinf(metrics["dual_error_inf"])
        assert math.isinf(metrics["kkt_inf"])
