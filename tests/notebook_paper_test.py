import contextlib
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

from scripts import notebook_paper
from scripts import benchmark_progress
from scripts import paper_sweep


class NotebookTest(unittest.TestCase):
    def test_progress_streams_diagnostics_without_polluting_csv(self):
        for code in (0, 7):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                output, errors = work / "results.csv", work / "results.stderr"
                command = [sys.executable, "-c",
                           "import sys,time; "
                           "print('backend,median_ms\\nclqr_cpu,1.25', flush=True); "
                           "print('[case 1/4] clqr_cpu N=1 n=4 timing', "
                           "file=sys.stderr, flush=True); "
                           f"time.sleep(0.15); sys.exit({code})"]
                console = io.StringIO()
                with contextlib.redirect_stdout(console):
                    actual = benchmark_progress.run(
                        command, name="cpu_round1", output=output, errors=errors,
                        interval=0.025)
                self.assertEqual(actual, code)
                self.assertEqual(output.read_text(),
                                 "backend,median_ms\nclqr_cpu,1.25\n")
                self.assertEqual(errors.read_text(),
                                 "[case 1/4] clqr_cpu N=1 n=4 timing\n")
                self.assertIn(errors.read_text(), console.getvalue())
                self.assertIn("[cpu_round1] RUNNING", console.getvalue())
                self.assertIn("COMPLETE" if code == 0 else "FAILED: exit code 7",
                              console.getvalue())
                self.assertNotIn("backend,median_ms", console.getvalue())

    def test_progress_streams_validation_stdout_and_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "validation.log"
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                code = benchmark_progress.run(
                    [sys.executable, "-c",
                     "import sys; print('case passed', flush=True); "
                     "print('diagnostic', file=sys.stderr, flush=True)"],
                    name="validation", output=output)
            self.assertEqual(code, 0)
            self.assertEqual(output.read_text(), "case passed\ndiagnostic\n")
            self.assertIn(output.read_text(), console.getvalue())

    @unittest.skipUnless(os.name == "posix", "POSIX process signal semantics")
    def test_progress_reports_killed_process(self):
        with tempfile.TemporaryDirectory() as directory:
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                code = benchmark_progress.run(
                    [sys.executable, "-c",
                     "import os,signal; os.kill(os.getpid(), signal.SIGKILL)"],
                    name="references", output=Path(directory) / "results.csv",
                    errors=Path(directory) / "results.stderr")
            self.assertEqual(code, 137)
            self.assertIn("FAILED: terminated by SIGKILL (9)", console.getvalue())

    def test_backend_switches_skip_fetch_build_and_execution(self):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        configurations = (
            ({"CLQR_RUN_EXTERNAL": "0"}, set(), False),
            ({"CLQR_RUN_EXTERNAL": "0", "CLQR_RUN_VANROYE": "1"},
             {"blasfeo"}, False),
            ({"CLQR_RUN_YANG": "0"},
             {"blasfeo", "eigen", "laine_author"}, False),
            ({}, {"blasfeo", "gtsam", "factor_graph", "laine_author"}, False),
            ({"CLQR_RUN_CPU": "0", "CLQR_RUN_EXTERNAL": "0", "CLQR_RUN_VANROYE": "1"},
             {"blasfeo"}, False),
            ({"CLQR_RUN_CPU": "0", "CLQR_RUN_EXTERNAL": "0", "CLQR_RUN_JAX_CPU": "1"},
             set(), False),
            # Explicit backend overrides also win over a disabled JAX group.
            *((dict(CLQR_RUN_EXTERNAL="0", CLQR_RUN_CPU=str(cpu),
                    CLQR_RUN_JAX_CPU=str(jax_cpu), CLQR_RUN_JAX_GPU=str(jax_gpu),
                    CLQR_RUN_TESTS=str(int(not cpu and not jax_cpu and jax_gpu)),
                    CLQR_RUN_SANITIZERS=str(int(not cpu and not jax_cpu and jax_gpu))), set(), True)
              for cpu, jax_cpu, jax_gpu in itertools.product((0, 1), repeat=3)),
        )
        for overrides, expected, cuda in configurations:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                source = work / "source"
                (source / "scripts").mkdir(parents=True)
                for name in ("paper_benchmarks.sh", "benchmark_options.sh",
                             "notebook_bazel.sh", "benchmark_progress.py", "paper_sweep.py",
                             "paper_results.py"):
                    shutil.copy(root / "scripts" / name, source / "scripts" / name)
                (source / ".bazelversion").write_text("9.1.1\n")
                (source / "bazel-bin").mkdir()
                tools = work / "tools"
                tools.mkdir()
                stub = work / "stub.py"
                # Stand-ins exercise the real shell driver without network,
                # nested Bazel, or reference compilation.
                stub.write_text('''import os, sys, json
from pathlib import Path
name, *args = sys.argv[1:]
with open(os.environ["CALL_LOG"], "a") as f:
    f.write(json.dumps([name, *args]) + "\\n")
if name == "python3" and args[0].endswith(("benchmark_progress.py", "paper_sweep.py", "paper_results.py")):
    os.execv(sys.executable, [sys.executable, *args])
case = dict(index=0, family="horizon", N=32, n=8, m=4, mixed_rows=1, state_rows=2)
if name == "clqr_paper_fixture":
    print(json.dumps([case]))
backend = {"clqr_paper_jax_cpu_benchmark": "clqr_jax_cpu",
           "clqr_paper_jax_cuda_benchmark": "clqr_jax_cuda"}.get(name)
if backend or "--backend" in args:
    import csv
    row = dict(backend=backend or args[args.index("--backend")+1], seed="20260907",
               median_ms=1.0, p10_ms=0.9, p90_ms=1.1, kernel_ms=0.8,
               primal_error=1e-12, relative_objective_error=1e-15,
               feasibility_inf=1e-12, kkt_inf=1e-12,
               status="ok", repeats=1, **{k:v for k,v in case.items() if k != "index"})
    writer = csv.DictWriter(sys.stdout, fieldnames=row.keys())
    writer.writeheader()
    writer.writerow(row)
if name == "git" and args[0] == "init":
    Path(args[-1]).mkdir(parents=True)
if name == "git" and "fetch" in args:
    (Path(args[1]) / "revision").write_text(args[-1])
if name == "git" and "rev-parse" in args:
    revision = Path(args[1]) / "revision"
    print(revision.read_text() if revision.exists() else "test-revision")
if name == "df":
    print("Filesystem 1024-blocks Used Available Capacity Mounted")
    print("mock 20000000 100 19999900 1% /mock")
if name == "nvidia-smi":
    if "--query-gpu=compute_cap" in args:
        print("6.0")
    else:
        print("name,compute_cap\\nTest GPU,6.0")
if name == "cmake" and "-B" in args:
    build = Path(args[args.index("-B") + 1])
    build.mkdir(parents=True, exist_ok=True)
    for exe in ("clqr_cpu_benchmark", "clqr_vanroye_benchmark", "clqr_yang_benchmark",
                "clqr_laine_benchmark", "clqr_laine_corrected_benchmark"):
        target = build / exe
        target.write_text(Path(os.environ["STUB_LAUNCHER"]).read_text())
        target.chmod(0o755)
''')
                launcher = tools / "launcher"
                launcher.write_text(
                    '#!/bin/sh\nexec "' + sys.executable + '" "' + str(stub) +
                    '" "${0##*/}" "$@"\n')
                launcher.chmod(0o755)
                for name in ("git", "cmake", "ctest", "bazel", "python3", "df", "sysctl", "tar",
                             "nvcc", "nvidia-smi", "compute-sanitizer"):
                    (tools / name).symlink_to(launcher)
                for name in ("clqr_paper_fixture", "clqr_paper_cpu_benchmark", "clqr_paper_jax_cpu_benchmark",
                             "clqr_paper_cuda_benchmark", "clqr_paper_jax_cuda_benchmark",
                             "cuda_solver_test", "adversarial_cuda_extended_test"):
                    (source / "bazel-bin" / name).symlink_to(launcher)
                log = work / "calls.jsonl"
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith("CLQR_")}
                env.update(PATH=str(tools) + os.pathsep + env["PATH"],
                           CLQR_BAZEL=str(tools / "bazel"), CALL_LOG=str(log),
                           STUB_LAUNCHER=str(launcher), CLQR_RUN_JAX="0",
                           CLQR_RUN_TESTS="0", CLQR_RUN_SANITIZERS="0",
                           CLQR_RUN_ORIGINAL_TABLE="0", CLQR_PAPER_SUITE="smoke")
                env.update(overrides)
                run_cpu = env.get("CLQR_RUN_CPU", "1") == "1"
                run_jax_cpu = env.get("CLQR_RUN_JAX_CPU", env["CLQR_RUN_JAX"]) == "1"
                run_jax_gpu = cuda and env.get("CLQR_RUN_JAX_GPU", env["CLQR_RUN_JAX"]) == "1"
                run_checks = env["CLQR_RUN_TESTS"] == "1"
                command = ["bash", str(source / "scripts/paper_benchmarks.sh"), str(work / "results")]
                if cuda:
                    command.append("--cuda")
                result = subprocess.run(command, env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual("[cpu_round1/clqr_cpu/0] START" in result.stdout, run_cpu)
                self.assertNotIn("cpu_round2", result.stdout)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                measured = [call for call in calls if "--case-index" in call and call[0] != "python3"]
                jax_backends = {"clqr_paper_jax_cpu_benchmark": "clqr_jax_cpu",
                                "clqr_paper_jax_cuda_benchmark": "clqr_jax_cuda"}
                backends = [jax_backends.get(call[0]) or call[call.index("--backend") + 1]
                            for call in measured]
                self.assertEqual(backends.count("clqr_cpu"), int(run_cpu))
                self.assertEqual(backends.count("clqr_jax_cpu"), int(run_jax_cpu))
                self.assertEqual(backends.count("clqr_jax_cuda"), int(run_jax_gpu))
                self.assertEqual(backends.count("clqr_cuda"), int(cuda))
                self.assertEqual(len(set(backends)), len(backends))
                self.assertTrue(all("--case-index" in call for call in measured))
                self.assertTrue(all("--min-seconds" in call for call in measured))
                self.assertTrue(all("--repeats" not in call for call in measured))
                for backend, executable, label in (
                        ("gen_riccati", "clqr_vanroye_benchmark", "vanroye"),
                        ("factor_graph", "clqr_yang_benchmark", "yang")):
                    selected = [call for call, actual in zip(measured, backends) if actual == backend]
                    self.assertTrue(all(call[0] == executable for call in selected))
                    if selected:
                        self.assertIn(f"[{label}] SWEEP START: 1 cases; backend={backend}", result.stdout)
                        self.assertTrue((work / "results" / (label + ".csv")).is_file())
                self.assertNotIn("clqr_reference_benchmark", log.read_text())
                fetched = {Path(call[-1]).name for call in calls
                           if call[:2] == ["git", "init"]}
                self.assertEqual(fetched, expected)
                vanroye_target = "//benchmarks/reference:vanroye_sources"
                self.assertEqual(any(vanroye_target in call for call in calls),
                                 "blasfeo" in expected)
                self.assertEqual(any(call[0] == "tar" for call in calls),
                                 "blasfeo" in expected)
                cmake_calls = [call for call in calls if call[0] == "cmake"]
                self.assertEqual(bool(cmake_calls), bool(expected))
                self.assertEqual(any("gtsam-build" in " ".join(call)
                                     for call in cmake_calls), "gtsam" in expected)
                self.assertFalse(any(call[0] == "ctest" for call in calls))
                build_calls = [call for call in calls if call[:2] == ["bazel", "build"]]
                for label, selected in (("clqr_paper_cpu_benchmark", run_cpu),
                                        ("clqr_paper_jax_cpu_benchmark", run_jax_cpu),
                                        ("clqr_paper_jax_cuda_benchmark", run_jax_gpu)):
                    self.assertEqual(any("//:" + label in call for call in build_calls), selected)
                if expected:
                    self.assertTrue(any(f"-DCLQR_COMPARE_CPU={'ON' if run_cpu else 'OFF'}" in call
                                        for call in cmake_calls))
                if cuda and run_checks:
                    self.assertIn("Passed cuda_validation", result.stdout)
                    self.assertIn("Passed cuda_rank_tolerance", result.stdout)
                    for tool in ("memcheck", "initcheck", "racecheck", "synccheck"):
                        self.assertIn(f"Passed cuda_regression_{tool}", result.stdout)
                        self.assertIn(f"Passed cuda_{tool}", result.stdout)
                summary = json.loads((work / "results/summary.json").read_text())
                self.assertEqual(set(summary) - {"platform", "gpus"}, set(backends))
                self.assertFalse((work / "results/original_table.csv").exists())
                for method, checkout in (("laine", "laine_author"),
                                          ("laine_corrected", "laine_author")):
                    self.assertEqual(any(call[0] == f"clqr_{method}_benchmark" for call in calls),
                                     checkout in expected)
                options = (work / "results/benchmark_options.txt").read_text()
                self.assertIn("CLQR_RUN_JAX=0", options)
                self.assertIn(f"CLQR_RUN_CPU={int(run_cpu)}", options)
                self.assertIn(f"CLQR_RUN_JAX_CPU={int(run_jax_cpu)}", options)

    @unittest.skipUnless(os.name == "posix", "POSIX process signal semantics")
    def test_sweep_preserves_later_cases_after_fatal_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = work / "cases.json"
            cases = [dict(index=i, family="horizon", N=32 * 2**i, n=8, m=4,
                          mixed_rows=1, state_rows=2) for i in range(3)]
            manifest.write_text(json.dumps(cases))
            child = """import csv, json, os, signal, sys
args = sys.argv
index = int(args[args.index('--case-index')+1])
if index == 1: os.kill(os.getpid(), signal.SIGKILL)
case = json.loads(open(args[1]).read())[index]
case.pop('index')
row = dict(backend=args[args.index('--backend')+1], seed=20260907,
           status='ok', repeats=40000, median_ms=0.025,
           solve_samples_ms=';'.join(['0.025'] * 40000), **case)
writer = csv.DictWriter(sys.stdout, fieldnames=row.keys())
writer.writeheader()
writer.writerow(row)
"""
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                code = paper_sweep.run(
                    [sys.executable, "-c", child, str(manifest)], manifest=manifest,
                    backend="clqr_cpu", name="cpu_round1", output=work / "result.csv",
                    errors=work / "result.stderr")
            import csv
            with (work / "result.csv").open() as output:
                rows = list(csv.DictReader(output))
            self.assertEqual(code, 1)
            self.assertEqual([row["status"] for row in rows], ["ok", "failed", "ok"])
            self.assertEqual([int(row["N"]) for row in rows], [32, 64, 128])
            self.assertIn("137", rows[1]["diagnostic"])
            self.assertEqual(len(rows[0]["solve_samples_ms"].split(';')), 40000)
            self.assertIn("CASE 2/3: clqr_cpu N=64 n=8 m=4", console.getvalue())
            self.assertIn("SAVED 2/3: clqr_cpu status=failed", console.getvalue())
            self.assertIn("SAVED 3/3: clqr_cpu status=ok", console.getvalue())
            self.assertIn("SWEEP COMPLETE: 3/3 cases saved", console.getvalue())

    def test_invalid_benchmark_switch_is_rejected(self):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        for name in ("CLQR_RUN_EXTERNAL", "CLQR_RUN_CPU", "CLQR_RUN_JAX_CPU", "CLQR_RUN_JAX_GPU"):
            with self.subTest(option=name):
                env = {key: value for key, value in os.environ.items() if not key.startswith("CLQR_")}
                env[name] = "sometimes"
                result = subprocess.run(
                    ["bash", "-c", 'source "$1"', "test", str(root / "scripts/benchmark_options.sh")],
                    env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{name} must be 0 or 1", result.stderr)

    def test_notebook_and_shell_share_jax_defaults_and_override_precedence(self):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        notebook = json.loads((root / "notebooks/kaggle_paper_comparison.ipynb").read_text())
        code = "".join(next(cell["source"] for cell in notebook["cells"]
                            if cell["cell_type"] == "code"))
        defaults = code[:code.index("# Use an uploaded snapshot")]
        for overrides, expected in (
                ({}, ("1", "1", "1")),
                ({"CLQR_RUN_JAX": "0"}, ("1", "0", "0")),
                ({"CLQR_RUN_JAX": "0", "CLQR_RUN_CPU": "0", "CLQR_RUN_JAX_GPU": "1"},
                 ("0", "0", "1")),
                ({"CLQR_RUN_JAX": "1", "CLQR_RUN_JAX_CPU": "0", "CLQR_RUN_JAX_GPU": "0"},
                 ("1", "0", "0"))):
            with self.subTest(overrides=overrides):
                env = {key: value for key, value in os.environ.items() if not key.startswith("CLQR_")}
                env.update(overrides)
                flags = ("CLQR_RUN_CPU", "CLQR_RUN_JAX_CPU", "CLQR_RUN_JAX_GPU")
                with mock.patch.dict(os.environ, env, clear=True):
                    exec(compile(defaults, "notebook-defaults", "exec"), {})
                    self.assertEqual(tuple(os.environ[name] for name in flags), expected)
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c",
                     'source "$1"; printf "%s\\n" "$CLQR_RUN_CPU" "$CLQR_RUN_JAX_CPU" "$CLQR_RUN_JAX_GPU"',
                     "test", str(root / "scripts/benchmark_options.sh")], env=env,
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(tuple(result.stdout.splitlines()), expected)

    def test_no_active_backend_is_rejected_before_build(self):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            env = {key: value for key, value in os.environ.items() if not key.startswith("CLQR_")}
            env.update(CLQR_RUN_CPU="0", CLQR_RUN_EXTERNAL="0", CLQR_RUN_JAX_CPU="0",
                       CLQR_RUN_JAX_GPU="1")
            result = subprocess.run(["bash", str(root / "scripts/paper_benchmarks.sh"), str(output)],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("GPU backends require --cuda", result.stderr)
            self.assertFalse(output.exists())

    def test_bazel_version_with_and_without_startup_options(self):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        driver = (root / "scripts/paper_benchmarks.sh").read_text()
        start = driver.index('bazel_command="$(clqr_notebook_bazel')
        end = driver.index('\ncd "$repo_dir"', start)
        # Execute the driver's actual command construction with a stand-in
        # that reports its argv. This does not start nested Bazel servers.
        shell = 'clqr_notebook_bazel() { printf "%s\\n" "$CLQR_BAZEL"; }\n'
        shell += driver[start:end]
        for explicit_root in (False, True):
            with self.subTest(explicit_root=explicit_root), \
                 tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                bazel = work / "fake bazel"
                bazel.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
                bazel.chmod(0o755)
                cache = str(work / "private cache") if explicit_root else ""
                env = dict(os.environ, CLQR_BAZEL=str(bazel),
                           CLQR_PAPER_BAZEL_ROOT=cache, repo_dir=str(root),
                           output_dir=str(work))
                subprocess.run(["bash", "-euo", "pipefail", "-c", shell],
                               env=env, check=True, capture_output=True, text=True)
                expected = (["--output_user_root=" + cache] if explicit_root else [])
                self.assertEqual((work / "platform.txt").read_text().splitlines(),
                                 expected + ["version"])

    def test_driver_selects_jax_and_rank_regressions(self):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        driver = (root / "scripts/paper_benchmarks.sh").read_text()
        start = driver.index('if (( CLQR_RUN_TESTS )); then\n  tests=')
        end = driver.index("\n# Build the authors' factor-graph dependency", start)
        shell = 'capture_bazel() { printf "%s\\n" "$@"; }\n'
        shell += 'bazel_cmd=(capture_bazel); bazel_args=(--config=fp64)\n'
        shell += driver[start:end]
        for run_tests, jax_cpu, jax_gpu, cuda in itertools.product((0, 1), repeat=4):
            with self.subTest(tests=run_tests, jax_cpu=jax_cpu, jax_gpu=jax_gpu, cuda=cuda):
                env = dict(os.environ, CLQR_RUN_TESTS=str(run_tests), CLQR_RUN_CPU="0",
                           CLQR_RUN_JAX_CPU=str(jax_cpu), CLQR_RUN_JAX_GPU=str(jax_gpu),
                           cuda_run=str(cuda), CLQR_RUN_CORRECTED_LAINE="0",
                           CLQR_RUN_YANG="0", CLQR_RUN_LAINE="0")
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", shell], env=env,
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                targets = result.stdout.splitlines()
                for name in ("cpu_rank_tolerance_test", "cuda_feasibility_rank_test",
                             "cuda_stage_layout_test", "cuda_buffer_test", "cuda_packing_pool_test"):
                    self.assertEqual("//:" + name in targets, bool(run_tests))
                for name in ("paper_jax_fixture_test", "jax_ffi_problem_test"):
                    self.assertEqual("//:" + name in targets,
                                     bool(run_tests and (jax_cpu or (cuda and jax_gpu))))
                self.assertEqual("//:jax_binding_test" in targets, bool(run_tests and jax_cpu))
                for name in ("jax_cuda_binding_test", "jax_cuda_transfer_audit_test"):
                    self.assertEqual("//:" + name in targets, bool(run_tests and jax_gpu and cuda))
        self.assertIn(
            "check_log cuda_rank_tolerance bazel-bin/cuda_solver_test --rank-regression",
            driver)

    def test_results_survive_success_and_failure(self):
        for code in (0, 1):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                source = work / "source"
                source.mkdir()
                untouched = source / "uncommitted.txt"
                untouched.write_text("keep this")
                neighbor = work / "other-cache"
                neighbor.mkdir()
                (neighbor / "keep").write_text("keep this too")

                def launch(command, **kwargs):
                    self.assertEqual(command[0], "bash")
                    self.assertEqual(command[-1], "--cuda")
                    self.assertEqual(kwargs["cwd"], source)
                    env = kwargs["env"]
                    results = Path(command[2])
                    cache = Path(env["CLQR_PAPER_CACHE_DIR"])
                    self.assertEqual(Path(env["CLQR_PAPER_BAZEL_ROOT"]), cache / "bazel")
                    self.assertEqual(Path(env["BAZELISK_HOME"]), cache / "bazelisk")
                    self.assertEqual(env["CLQR_PAPER_SUITE"], "smoke")
                    cache.mkdir()
                    (cache / "discard").write_text("generated cache")
                    results.mkdir()
                    (results / "measurements.csv").write_text("backend,kkt_inf\nclqr_cpu,1e-12\n")
                    (results / "clqr_tools").mkdir()
                    (results / "clqr_tools/bazelisk").write_text("generated tool")
                    process = mock.MagicMock()
                    process.stdout = io.StringIO("test driver output\n")
                    process.wait.return_value = code
                    process.__enter__.return_value = process
                    return process

                with mock.patch.object(notebook_paper.subprocess, "Popen", side_effect=launch), \
                     mock.patch.object(notebook_paper.shutil, "disk_usage",
                                       return_value=SimpleNamespace(free=10 * 1024**3)), \
                     contextlib.redirect_stdout(io.StringIO()):
                    actual = notebook_paper.run(source, work, suite="smoke", repeats=1)
                self.assertEqual(actual, code)
                root, = work.glob("clqr-paper-*")
                self.assertFalse((root / "cache").exists())
                self.assertTrue((root / "results/measurements.csv").is_file())
                self.assertEqual(untouched.read_text(), "keep this")
                self.assertTrue((neighbor / "keep").is_file())
                with zipfile.ZipFile(root / "paper-results.zip") as archive:
                    self.assertEqual(set(archive.namelist()),
                                     {"driver.log", "results/measurements.csv"})
                    self.assertEqual(archive.read("driver.log"), b"test driver output\n")

    def test_low_space_stops_before_starting(self):
        with mock.patch.object(notebook_paper.shutil, "disk_usage",
                               return_value=SimpleNamespace(free=1024**3)), \
             mock.patch.object(notebook_paper.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "5 GiB"):
                notebook_paper.run(Path("source"), Path("unused"))
            launch.assert_not_called()

    def test_notebook_is_fresh_and_executable(self):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        notebook = json.loads((root / "notebooks/kaggle_paper_comparison.ipynb").read_text())
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(cells), 1)
        cell = cells[0]
        self.assertEqual(cell["outputs"], [])
        self.assertIsNone(cell["execution_count"])
        code = "".join(cell["source"])
        compile(code, "kaggle_paper_comparison.ipynb", "exec")
        self.assertIn('get("CLQR_REVISION", "main")', code)
        self.assertIn("notebook_paper.py", code)
        for option in ("CPU", "EXTERNAL", "JAX", "TESTS", "SANITIZERS", "ORIGINAL_TABLE"):
            self.assertIn(f'setdefault("CLQR_RUN_{option}", "1")', code)
        for option in ("JAX_CPU", "JAX_GPU"):
            self.assertIn(f'setdefault("CLQR_RUN_{option}", os.environ["CLQR_RUN_JAX"])', code)
        for method in ("VANROYE", "YANG", "LAINE", "CORRECTED_LAINE"):
            self.assertIn(f'"{method}"', code)


if __name__ == "__main__":
    unittest.main()
