import csv
import io
import math
import os
from pathlib import Path
import subprocess
import unittest


class AdversarialBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        cls.executable = root / "clqr_adversarial_cpu_benchmark"

    def run_case(self, name):
        result = subprocess.run([self.executable, "--case", name, "--repeats", "1"],
                                text=True, capture_output=True, check=True)
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        self.assertEqual(len(rows), 1)
        self.assertNotIn(None, rows[0])  # Detect malformed CSV column counts.
        return rows[0]

    def test_measured_solve(self):
        row = self.run_case("single-stage")
        self.assertEqual(row["status"], "returned")
        self.assertGreater(float(row["median_ms"]), 0)
        self.assertGreaterEqual(int(row["batch"]), 1)
        for field in ("kkt_inf", "unit_kkt_inf", "dense_difference"):
            self.assertLess(float(row[field]), 1e-10)

    def test_numerical_rejection_is_data(self):
        row = self.run_case("singular-reduced-hessian")
        self.assertEqual(row["status"], "rejected")
        self.assertTrue(math.isnan(float(row["kkt_inf"])))
        self.assertIn("Hessian", row["diagnostic"])

    def test_infeasibility_is_data(self):
        row = self.run_case("infeasible-initial")
        self.assertEqual(row["status"], "rejected")
        self.assertEqual(row["expected"], "infeasible")
        self.assertTrue(row["diagnostic"])

    def test_unknown_case_is_harness_error(self):
        result = subprocess.run([self.executable, "--case", "nonexistent"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
