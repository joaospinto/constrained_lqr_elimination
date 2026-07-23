"""Cross-validate the CUDA backend against both JAX elimination solvers."""

from __future__ import annotations

import argparse
import json
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from constrained_lqr_jax import solver as jax_solver
from constrained_lqr_jax.helpers import compute_residual
from constrained_lqr_jax.types import FactorizationInputs, SolveInputs

jax.config.update("jax_enable_x64", True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", help="path to clqr_cuda_jax_fixture")
    parser.add_argument(
        "--cpu-fixture",
        action="store_true",
        help="ask the fixture executable for its CPU reference output",
    )
    args = parser.parse_args()
    command = [args.fixture]
    if args.cpu_fixture:
        command.append("--cpu")
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"fixture exited with status {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    raw = completed.stdout
    data = json.loads(raw)
    precision = data["precision"]
    comparison_tolerance = 2e-2 if precision == "FP32" else 3e-6
    residual_tolerance = 3e-6
    storage_dtype = np.float32 if precision == "FP32" else np.float64

    def reference_array(value: object) -> jax.Array:
        # Reconstruct the selected C++ storage values first, then promote them
        # for a high-accuracy JAX reference solve. This also preserves exact
        # FP32 row dependencies that decimal JSON parses can perturb in FP64.
        return jnp.asarray(np.asarray(value, dtype=storage_dtype), dtype=jnp.float64)
    print(
        "JAX backend:",
        jax.default_backend(),
        "; devices:",
        ", ".join(str(device) for device in jax.devices()),
        "; fixture precision:",
        precision,
    )
    A = reference_array(data["A"])
    B = reference_array(data["B"])
    Q = reference_array(data["Q"])
    M = reference_array(data["M"])
    R = reference_array(data["R"])
    D = reference_array(data["D"])
    E = reference_array(data["E"])
    q = reference_array(data["q"])
    r = reference_array(data["r"])
    c = reference_array(data["c"])
    d = reference_array(data["d"])
    horizon, n = A.shape[:2]
    p = D.shape[1]
    factorization = FactorizationInputs(
        A=A,
        B=B,
        Q=Q,
        M=M,
        R=R,
        D=D,
        E=E,
        Delta=jnp.zeros((horizon + 1, n, n), dtype=A.dtype),
        Sigma=jnp.zeros((horizon + 1, p, p), dtype=A.dtype),
    )
    solve_inputs = SolveInputs(q=q, r=r, c=c, d=d)
    cuda_x = np.asarray(data["X"])
    cuda_u = np.asarray(data["U"])
    fixture_name = "C++ CPU fixture" if args.cpu_fixture else "CUDA"
    sequential_solver = getattr(
        jax_solver, "factor_and_solve_elimination", jax_solver.factor_and_solve
    )
    for name, solver in (
        ("JAX sequential solver", sequential_solver),
        ("JAX parallel elimination", jax_solver.factor_and_solve_parallel_elimination),
    ):
        solution = solver(factorization, solve_inputs)
        np.testing.assert_allclose(
            solution.X,
            cuda_x,
            atol=comparison_tolerance,
            rtol=comparison_tolerance,
        )
        np.testing.assert_allclose(
            solution.U,
            cuda_u,
            atol=comparison_tolerance,
            rtol=comparison_tolerance,
        )
        residual = float(
            jnp.max(
                jnp.abs(compute_residual(factorization, solve_inputs, solution))
            )
        )
        if residual > residual_tolerance:
            raise AssertionError(f"{name} KKT residual is {residual:.3e}")
        print(f"{name}: matched {fixture_name}; KKT residual={residual:.3e}")


if __name__ == "__main__":
    main()
