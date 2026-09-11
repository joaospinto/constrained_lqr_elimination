import contextlib
import io
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
            ({"CLQR_RUN_EXTERNAL": "0"}, set()),
            ({"CLQR_RUN_EXTERNAL": "0", "CLQR_RUN_VANROYE": "1"},
             {"blasfeo", "generalization_riccati"}),
            ({"CLQR_RUN_YANG": "0"},
             {"blasfeo", "generalization_riccati", "eigen", "laine_author"}),
            ({}, {"blasfeo", "generalization_riccati", "gtsam", "factor_graph", "laine_author"}),
        )
        for overrides, expected in configurations:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                source = work / "source"
                (source / "scripts").mkdir(parents=True)
                for name in ("paper_benchmarks.sh", "benchmark_options.sh",
                             "notebook_bazel.sh", "benchmark_progress.py"):
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
if name == "python3" and args[0].endswith("benchmark_progress.py"):
    os.execv(sys.executable, [sys.executable, *args])
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
if name == "cmake" and "-B" in args:
    build = Path(args[args.index("-B") + 1])
    build.mkdir(parents=True, exist_ok=True)
    for exe in ("clqr_cpu_benchmark", "clqr_reference_benchmark",
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
                for name in ("git", "cmake", "ctest", "bazel", "python3", "df", "sysctl"):
                    (tools / name).symlink_to(launcher)
                for name in ("clqr_paper_fixture", "clqr_paper_cpu_benchmark", "clqr_paper_jax_cpu_benchmark"):
                    (source / "bazel-bin" / name).symlink_to(launcher)
                log = work / "calls.jsonl"
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith("CLQR_")}
                env.update(PATH=str(tools) + os.pathsep + env["PATH"],
                           CLQR_BAZEL=str(tools / "bazel"), CALL_LOG=str(log),
                           STUB_LAUNCHER=str(launcher), CLQR_RUN_JAX="0",
                           CLQR_RUN_TESTS="0", CLQR_RUN_SANITIZERS="0",
                           CLQR_RUN_ORIGINAL_TABLE="0", **overrides)
                result = subprocess.run(["bash", str(source / "scripts/paper_benchmarks.sh"),
                                         str(work / "results")], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("[cpu_round1] START", result.stdout)
                self.assertIn("[cpu_round2] COMPLETE", result.stdout)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                fetched = {Path(call[-1]).name for call in calls
                           if call[:2] == ["git", "init"]}
                self.assertEqual(fetched, expected)
                cmake_calls = [call for call in calls if call[0] == "cmake"]
                self.assertEqual(bool(cmake_calls), bool(expected))
                self.assertEqual(any("gtsam-build" in " ".join(call)
                                     for call in cmake_calls), "gtsam" in expected)
                self.assertFalse(any(call[0] == "ctest" for call in calls))
                self.assertFalse(any("jax" in " ".join(call) for call in calls))
                for method, checkout in (("laine", "laine_author"),
                                          ("laine_corrected", "laine_author")):
                    self.assertEqual(any(call[0] == f"clqr_{method}_benchmark" for call in calls),
                                     checkout in expected)
                options = (work / "results/benchmark_options.txt").read_text()
                self.assertIn("CLQR_RUN_JAX=0", options)

    def test_invalid_benchmark_switch_is_rejected(self):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        result = subprocess.run(
            ["bash", "-c", 'source "$1"', "test", str(root / "scripts/benchmark_options.sh")],
            env=dict(os.environ, CLQR_RUN_EXTERNAL="sometimes"), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CLQR_RUN_EXTERNAL must be 0 or 1", result.stderr)

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
        for option in ("EXTERNAL", "JAX", "TESTS", "SANITIZERS", "ORIGINAL_TABLE"):
            self.assertIn(f'setdefault("CLQR_RUN_{option}", "1")', code)
        for method in ("VANROYE", "YANG", "LAINE", "CORRECTED_LAINE"):
            self.assertIn(f'"{method}"', code)


if __name__ == "__main__":
    unittest.main()
