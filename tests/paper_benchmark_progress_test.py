"""Exercise progress reporting with the real native FP64 smoke benchmark."""

import csv
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
            for phase in ("setup", "warmup", "timing 1 prepared solves",
                          "validation", "timing 1 setup+solve calls", "DONE status=ok"):
                self.assertIn(prefix + " " + phase, result.stderr)
        self.assertIn("[case 1/4] generating fixture", result.stderr)


if __name__ == "__main__":
    unittest.main()
