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
    raw = subprocess.run(
        command, check=True, text=True, capture_output=True
    ).stdout
    data = json.loads(raw)
    A = jnp.asarray(data["A"], dtype=jnp.float64)
    B = jnp.asarray(data["B"], dtype=jnp.float64)
    Q = jnp.asarray(data["Q"], dtype=jnp.float64)
    M = jnp.asarray(data["M"], dtype=jnp.float64)
    R = jnp.asarray(data["R"], dtype=jnp.float64)
    D = jnp.asarray(data["D"], dtype=jnp.float64)
    E = jnp.asarray(data["E"], dtype=jnp.float64)
    q = jnp.asarray(data["q"], dtype=jnp.float64)
    r = jnp.asarray(data["r"], dtype=jnp.float64)
    c = jnp.asarray(data["c"], dtype=jnp.float64)
    d = jnp.asarray(data["d"], dtype=jnp.float64)
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
        np.testing.assert_allclose(solution.X, cuda_x, atol=3e-6, rtol=3e-6)
        np.testing.assert_allclose(solution.U, cuda_u, atol=3e-6, rtol=3e-6)
        residual = float(
            jnp.max(
                jnp.abs(compute_residual(factorization, solve_inputs, solution))
            )
        )
        if residual > 3e-6:
            raise AssertionError(f"{name} KKT residual is {residual:.3e}")
        print(f"{name}: matched {fixture_name}; KKT residual={residual:.3e}")


if __name__ == "__main__":
    main()
