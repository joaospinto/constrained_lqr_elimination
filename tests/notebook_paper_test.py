import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

from scripts import notebook_paper


class NotebookTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
