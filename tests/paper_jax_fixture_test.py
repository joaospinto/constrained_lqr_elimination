import contextlib
import csv
import io

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
    with contextlib.redirect_stdout(output):
        code = main("cpu", ["--suite", "smoke", "--repeats", "1"])
    print(output.getvalue(), end="")
    rows = list(csv.DictReader(line for line in output.getvalue().splitlines()
                               if line and not line.startswith("#")))
    # Benchmarks report numerical outcomes without failing their process.
    # This regression test must still enforce correctness on its smoke cases.
    assert code == 0
    assert len(rows) == 4
    assert all(row["status"] == "ok" for row in rows), rows
