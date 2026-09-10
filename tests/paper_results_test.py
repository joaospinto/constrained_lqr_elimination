import copy
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts import paper_results as results


def row(backend="clqr_cpu"):
    return dict(backend=backend, family="horizon", N="128", n="8", m="4",
                mixed_rows="1", state_rows="2", seed="20260907", status="ok",
                median_ms="1.2", p10_ms="1.1", p90_ms="1.3",
                primal_error="1e-12", relative_objective_error="1e-15",
                feasibility_inf="1e-14", kkt_inf="1e-11", kernel_ms="1.0",
                dual_error_inf="1e-11")


class ResultsTest(unittest.TestCase):
    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            results.indexed([row(), row()], "clqr_cpu")

    def test_false_success_rejected(self):
        for field, value in (("primal_error", "0.001"), ("kkt_inf", "nan"),
                             ("median_ms", "-1"), ("feasibility_inf", "inf")):
            changed = row()
            changed[field] = value
            with self.assertRaises(ValueError):
                results.indexed([changed], "clqr_cpu")

    def test_reference_without_duals(self):
        for backend in results.PRIMAL_ONLY:
            value = row(backend)
            value["kkt_inf"] = "nan"
            value["dual_error_inf"] = "nan"
            value["planted_dual_stationarity_inf"] = "1e-12"
            data = results.indexed([value], backend)
            summary = results.summarize({backend: data})[backend]
            self.assertIsNone(summary["maxima"]["kkt_inf"])
            self.assertEqual(summary["maxima"]["planted_dual_stationarity_inf"], 1e-12)
            self.assertNotIn("kkt_inf", summary["nonfinite_measurements"])
            self.assertNotIn("dual_error_inf", summary["nonfinite_measurements"])
            self.assertIsNone(summary["maxima"]["dual_error_inf"])

    def test_planted_certificate_cannot_hide_inaccuracy(self):
        value = row("laine_author")
        value.update(planted_dual_stationarity_inf="0.3")
        with self.assertRaisesRegex(ValueError, "contradicts"):
            results.indexed([value], "laine_author")
        value["status"] = "inaccurate"
        data = results.indexed([value], "laine_author")
        self.assertEqual(results.summarize({"laine_author": data})
                         ["laine_author"]["counts"]["inaccurate"], 1)

    def test_author_dual_errors_are_not_primal_only(self):
        for residual in ("nan", "inf", "0.3"):
            with self.subTest(residual=residual):
                value = row("laine_author")
                value.update(kkt_inf=residual, planted_dual_stationarity_inf="1e-12")
                with self.assertRaises(ValueError):
                    results.indexed([value], "laine_author")
                # A numerical failure remains benchmark data, not a failed run.
                value["status"] = "inaccurate"
                data = results.indexed([value], "laine_author")
                report = results.summarize({"laine_author": data})["laine_author"]
                self.assertEqual(report["counts"]["inaccurate"], 1)
                self.assertEqual(report["nonfinite_measurements"]["kkt_inf"],
                                 int(residual in ("nan", "inf")))

    def test_manifest_detects_case_missing_from_every_backend(self):
        value = row()
        data = {"clqr_cpu": results.indexed([value], "clqr_cpu")}
        manifest = [{field: value[field] for field in results.KEY_FIELDS[:-1]}]
        self.assertEqual(results.validate_cases(data, manifest), {results.key(value)})
        manifest.append(dict(manifest[0], N="32768"))
        with self.assertRaisesRegex(ValueError, "manifest"):
            results.validate_cases(data, manifest)

    def test_backends_cannot_mix_seeds(self):
        cpu, gpu = row(), row("clqr_cuda")
        gpu["seed"] = "20260908"
        data = {"clqr_cpu": results.indexed([cpu], "clqr_cpu"),
                "clqr_cuda": results.indexed([gpu], "clqr_cuda")}
        with self.assertRaisesRegex(ValueError, "different cases/seeds"):
            results.validate_cases(data, [])

    def test_missing_kernel_time_rejected(self):
        value = row("clqr_cuda")
        value["kernel_ms"] = "nan"
        with self.assertRaisesRegex(ValueError, "kernel time"):
            results.indexed([value], "clqr_cuda")

    def test_accuracy_failures_remain_visible(self):
        value = row()
        value.update(status="inaccurate", primal_error="2e-6")
        data = results.indexed([value], "clqr_cpu")
        self.assertEqual(results.time_cell(value), r"1.200$^{\dagger}$")
        self.assertEqual(results.summarize({"clqr_cpu": data})
                         ["clqr_cpu"]["counts"]["inaccurate"], 1)
        value["status"] = "unsupported"
        self.assertEqual(results.time_cell(value), r"\textsc{oom}")
        value["status"] = "failed"
        self.assertEqual(results.time_cell(value), r"\textsc{fail}")

    def test_numerical_outcomes_do_not_fail_the_summary(self):
        for status in ("inaccurate", "failed"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                files = {}
                for backend, name in (results.SOURCES | results.OPTIONAL_SOURCES).items():
                    if "cuda" in backend:
                        continue
                    value = row(backend)
                    # Include third-party failures, not only our own solver.
                    value.update(status=status, primal_error="2e-6")
                    if backend in results.PRIMAL_ONLY:
                        value["kkt_inf"] = "nan"
                    files.setdefault(name, []).append(value)
                for name, rows in files.items():
                    with (path / name).open("w") as stream:
                        writer = csv.DictWriter(stream, fieldnames=rows[0])
                        writer.writeheader()
                        writer.writerows(rows)
                manifest = [{field: row()[field] for field in results.KEY_FIELDS[:-1]}]
                (path / "cases.json").write_text(json.dumps(manifest))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(results.main([str(path), "--suite", "smoke"]), 0)
                report = json.loads((path / "summary.json").read_text())
                self.assertEqual(report["factor_graph"]["counts"][status], 1)
                self.assertEqual(report["clqr_cpu"]["counts"][status], 1)
                self.assertEqual(report["laine_author"]["counts"][status], 1)
                self.assertEqual(report["laine_corrected"]["counts"][status], 1)
                measured = results.read_csv(path / "measurements.csv")
                self.assertTrue(all(r["reference_solution"] == "planted_optimum" for r in measured))
                self.assertEqual(len(measured), len(files) + 1)

    def test_nonfinite_errors_are_reported_not_discarded(self):
        value = row()
        value.update(status="inaccurate", primal_error="inf", kkt_inf="nan")
        data = results.indexed([value], "clqr_cpu")
        report = results.summarize({"clqr_cpu": data})["clqr_cpu"]
        self.assertIsNone(report["maxima"]["primal_error"])
        self.assertEqual(report["nonfinite_measurements"]["primal_error"], 1)
        self.assertEqual(report["nonfinite_measurements"]["kkt_inf"], 1)
        json.dumps(report, allow_nan=False)

    def test_cuda_report_retains_actual_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            files = {}
            for backend, name in (results.SOURCES | results.OPTIONAL_SOURCES).items():
                value = row(backend)
                if backend in results.PRIMAL_ONLY:
                    value.update(kkt_inf="nan", dual_error_inf="nan")
                files.setdefault(name, []).append(value)
            original = dict(N="32", n="8", m="4", p="2", cpp_cpu_ms="3",
                            cpp_kkt_residual="2e-14", cuda_kernel_ms="1",
                            cuda_wall_ms="2", cuda_kkt_residual="3e-13",
                            input_pack_ms="0.4", upload_ms="0.1", download_ms="0.1")
            files["original_table.csv"] = [dict(original, N=str(2**exponent))
                                            for exponent in range(5, 15)]
            for name, rows in files.items():
                with (path / name).open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
            manifest = [{field: row()[field] for field in results.KEY_FIELDS[:-1]}]
            (path / "cases.json").write_text(json.dumps(manifest))
            platform = "CPU: test host\nGPU: NVIDIA B200\nCUDA build architecture: sm_100\n"
            (path / "platform.txt").write_text(platform)
            # nvidia-smi separates both its headers and values with comma-space.
            (path / "gpu.csv").write_text("index, name, compute_cap\n0, NVIDIA B200, 10.0\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(results.main([str(path), "--suite", "smoke",
                                               "--cuda", "--require-laine"]), 0)
            report = json.loads((path / "summary.json").read_text())
            self.assertEqual(report["platform"], platform)
            self.assertEqual(report["gpus"],
                             [dict(index="0", name="NVIDIA B200", compute_cap="10.0")])

    def test_table_keeps_independent_columns(self):
        data = {}
        for index, backend in enumerate(results.SOURCES | results.OPTIONAL_SOURCES):
            value = row(backend)
            value["median_ms"] = str(index + 1)
            data[backend] = {results.key(value): value}
        text = results.comparison_table(data)
        self.assertIn("128 & 8 & 1.000 & 2.000 & 3.000 & 7.000 & 8.000 & 1.000 & 5.000 & 6.000", text)
        self.assertIn(r"\multicolumn{5}{c|}{CPU}", text)
        self.assertIn(r"\multicolumn{3}{c|}{GPU}", text)
        self.assertIn(r"6.000 \\", text)

    def test_fixed_ratio_table_selection(self):
        identities = [("horizon", N, 8, 4, 1, 2, 7)
                      for N in (128, 512, 2048, 8192, 32768)]
        identities += [("dimension", 128, n, n // 2, n // 8, n // 4, 7)
                       for n in (8, 16, 32, 64)]
        self.assertEqual(sum(map(results.selected, identities)), 8)
        self.assertFalse(results.selected(("dimension", 128, 32, 16, 4, 4, 7)))

    def test_dual_coordinate_errors_are_data(self):
        value = row()
        value["dual_error_inf"] = "1.0"
        # A coordinate difference alone is not a KKT error gate.
        data = results.indexed([value], "clqr_cpu")
        self.assertEqual(results.summarize({"clqr_cpu": data})
                         ["clqr_cpu"]["maxima"]["dual_error_inf"], 1.0)
        value["dual_error_inf"] = "nan"
        with self.assertRaises(ValueError):
            results.indexed([value], "clqr_cpu")

    def test_original_rows_reject_missing_cpu_values(self):
        value = dict(N="32", cpp_cpu_ms=".12345", cpp_kkt_residual="2.04e-14",
                     cuda_kernel_ms="1.86", cuda_wall_ms="2.02", cuda_kkt_residual="3e-13")
        text = results.original_rows([value])
        self.assertIn(r"0.123 & $2.04{\times}10^{-14}$ & 1.860 & 2.020", text)
        changed = copy.copy(value)
        changed["cpp_kkt_residual"] = "nan"
        with self.assertRaises(ValueError):
            results.original_rows([changed])


if __name__ == "__main__":
    unittest.main()
