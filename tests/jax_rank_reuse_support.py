"""Shared rank-changing fixture and original-KKT checks for JAX bindings."""

import numpy as np


def rank_reuse_problem(dtype):
    return {
        "initial_state": np.array([0.0, 0.5], dtype=dtype),
        "stages": [
            {
                "A": np.eye(2, dtype=dtype),
                "B": np.eye(2, dtype=dtype),
                "c": np.zeros(2, dtype=dtype),
                "R": np.eye(2, dtype=dtype),
                "M": np.zeros((2, 2), dtype=dtype),
                "r": np.zeros(2, dtype=dtype),
                "C": np.array([[0.0, 1.0]], dtype=dtype),
                "D": np.array([[0.0, 1.0]], dtype=dtype),
                "E": np.array([[1.0, 0.0]], dtype=dtype),
            }
            for _ in range(3)
        ],
        "Q": [np.eye(2, dtype=dtype) for _ in range(4)],
        "q": [np.zeros(2, dtype=dtype) for _ in range(4)],
    }


def assert_uniform_solution_kkt(packed, solution, atol):
    """Check all original KKT equations for a uniform-width packed fixture.

    Redundant rows allow different valid duals; test stationarity using the
    returned multipliers instead of comparing their individual values.
    """
    f = packed.factors._replace(**{
        name: np.asarray(value) for name, value in packed.factors._asdict().items()
    })
    b = packed.rhs._replace(**{
        name: np.asarray(value) for name, value in packed.rhs._asdict().items()
    })
    s = solution._replace(**{
        name: np.asarray(value) for name, value in solution._asdict().items()
    })
    for name, value in s._asdict().items():
        assert np.all(np.isfinite(value)), f"nonfinite {name}"
    x, u = s.states, s.controls
    lam, mu, nu = s.dynamics_multipliers, s.mixed_multipliers, s.state_multipliers
    mv = lambda a, v: np.einsum("tij,tj->ti", a, v)
    mtv = lambda a, v: np.einsum("tji,tj->ti", a, v)
    gx = mv(f.Q, x) + b.q
    gx[:-1] += mv(f.M, u) - mtv(f.A, lam) + mtv(f.C, mu) + mtv(f.E, nu)
    gx[0] += s.initial_multiplier
    gx[1:] += lam
    gx[-1] += f.terminal_E.T @ s.terminal_state_multiplier
    gu = mv(f.R, u) + b.r + mtv(f.M, x[:-1]) - mtv(f.B, lam) + mtv(f.D, mu)
    residuals = {
        "initial state": x[0] - b.initial_state,
        "dynamics": x[1:] - mv(f.A, x[:-1]) - mv(f.B, u) - b.c,
        "mixed constraints": mv(f.C, x[:-1]) + mv(f.D, u) + b.d,
        "state constraints": mv(f.E, x[:-1]) + b.e,
        "terminal constraints": f.terminal_E @ x[-1] + b.terminal_e,
        "state stationarity": gx,
        "control stationarity": gu,
    }
    objective = (0.5 * np.sum(x * mv(f.Q, x)) + np.sum(b.q * x)
                 + 0.5 * np.sum(u * mv(f.R, u)) + np.sum(b.r * u)
                 + np.sum(x[:-1] * mv(f.M, u)))
    residuals["objective"] = s.objective - objective
    for name, residual in residuals.items():
        np.testing.assert_allclose(residual, 0.0, rtol=0, atol=atol, err_msg=name)
