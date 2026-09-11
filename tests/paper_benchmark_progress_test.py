"""Exercise progress reporting with the real native FP64 smoke benchmark."""

import csv
import math
import os
from pathlib import Path
import subprocess
import unittest


class NativeProgressTest(unittest.TestCase):
    def test_progress_is_separate_from_measurements(self):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        result = subprocess.run(
            [str(root / "clqr_paper_cpu_benchmark"),
             "--suite", "smoke", "--repeats", "1"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = list(csv.DictReader(
            line for line in result.stdout.splitlines()
            if line and not line.startswith("#")))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["status"] == "ok" for row in rows), rows)
        self.assertTrue(all(row["repeats"] == "1" for row in rows))
        self.assertNotIn("[case", result.stdout)
        for index, row in enumerate(rows, 1):
            prefix = (f"[case {index}/4] clqr_cpu N={row['N']} "
                      f"n={row['n']} m={row['m']}")
            for phase in ("setup", "warmup (1 solve)", "timing prepared solves: 1 calls",
                          "validation", "timing setup+solve: 1 calls", "DONE status=ok"):
                self.assertIn(prefix + " " + phase, result.stderr)
        self.assertIn("[case 1/4] generating fixture", result.stderr)
        self.assertEqual(result.stderr.count("warmup (1 solve)"), len(rows))
        self.assertIn("exactly one untimed warmup solve", result.stdout)
        for row in rows:
            self.assertTrue(math.isnan(float(row["p10_ms"])))
            self.assertTrue(math.isnan(float(row["p90_ms"])))
            self.assertEqual(len(row["solve_samples_ms"].split(";")), 1)
            self.assertEqual(row["setup_solve_repeats"], "1")

    def test_duration_sampling_and_backend_selection(self):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        binary = str(root / "clqr_paper_cpu_benchmark")
        for seconds in (0.005, 1e-9):
            result = subprocess.run(
                [binary, "--suite", "smoke", "--case-index", "0",
                 "--backend", "clqr_cpu", "--min-seconds", str(seconds)],
                capture_output=True, text=True, check=True, timeout=30)
            rows = list(csv.DictReader(line for line in result.stdout.splitlines()
                                       if line and not line.startswith("#")))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["backend"], "clqr_cpu")
            self.assertEqual(row["status"], "ok")
            for field, count in (("solve_samples_ms", "repeats"),
                                 ("setup_solve_samples_ms", "setup_solve_repeats")):
                samples = [float(x) for x in row[field].split(";")]
                self.assertEqual(len(samples), int(row[count]))
                self.assertGreaterEqual(sum(samples) + 1e-9, seconds * 1000)
                self.assertLess(sum(samples[:-1]), seconds * 1000)
                if seconds == 1e-9:
                    self.assertEqual(len(samples), 1)
        result = subprocess.run([binary, "--backend", "laine_author"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
