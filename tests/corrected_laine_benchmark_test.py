"""Check the actual benchmark adapter, including its dual sign/row mapping."""

import csv
import io
import math
import os
from pathlib import Path
import subprocess
import unittest


class BenchmarkTest(unittest.TestCase):
    def test_returned_duals_in_original_coordinates(self):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        output = subprocess.check_output(
            [str(root / "clqr_laine_corrected_benchmark"),
             "--suite", "smoke", "--repeats", "1"], text=True)
        rows = list(csv.DictReader(io.StringIO("\n".join(
            line for line in output.splitlines() if not line.startswith("#")))))
        corrected = [r for r in rows if r["backend"] == "laine_corrected"]
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(corrected), len(rows))  # No implicit CPU sweep.
        self.assertEqual({r["family"] for r in corrected},
                         {"state", "mixed", "combined", "zero"})
        for row in corrected:
            with self.subTest(family=row["family"]):
                self.assertEqual(row["status"], "ok")
                for field in ("primal_error", "dual_error_inf", "stationarity_inf",
                              "feasibility_inf", "kkt_inf"):
                    value = float(row[field])
                    self.assertTrue(math.isfinite(value), field)
                    self.assertLess(value, 1e-8, field)


if __name__ == "__main__":
    unittest.main()
