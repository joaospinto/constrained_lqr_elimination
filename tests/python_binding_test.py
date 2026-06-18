import importlib.util
import os
import pathlib
import sys

import numpy as np


def _load_extension():
    root = pathlib.Path(os.environ["TEST_SRCDIR"])
    candidates = list(root.rglob("_clqr.so"))
    if not candidates:
        raise RuntimeError("could not find _clqr.so in Bazel runfiles")
    spec = importlib.util.spec_from_file_location("_clqr", candidates[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules["_clqr"] = module
    spec.loader.exec_module(module)
    return module


def test_python_solve():
    clqr = _load_extension()
    problem = {
        "initial_state": np.array([1.0], dtype=np.float64),
        "stages": [
            {
                "A": np.array([[1.0]], dtype=np.float64),
                "B": np.array([[1.0]], dtype=np.float64),
                "c": np.array([0.0], dtype=np.float64),
                "Q": np.array([[1.0]], dtype=np.float64),
                "R": np.array([[2.0]], dtype=np.float64),
                "M": np.array([[0.0]], dtype=np.float64),
                "q": np.array([0.0], dtype=np.float64),
                "r": np.array([0.0], dtype=np.float64),
                "C": np.array([[1.0]], dtype=np.float64),
                "D": np.array([[1.0]], dtype=np.float64),
                "d": np.array([0.0], dtype=np.float64),
            }
        ],
        "terminal_Q": np.array([[1.0]], dtype=np.float64),
        "terminal_q": np.array([0.0], dtype=np.float64),
    }
    result = clqr.solve(problem)
    assert result["status"] == "optimal", result
    np.testing.assert_allclose(result["states"][0], np.array([1.0]), atol=1e-9)
    np.testing.assert_allclose(result["controls"][0], np.array([-1.0]), atol=1e-9)
    np.testing.assert_allclose(result["states"][1], np.array([0.0]), atol=1e-9)
    np.testing.assert_allclose(result["initial_multiplier"], np.array([-3.0]), atol=1e-9)
    np.testing.assert_allclose(result["dynamics_multipliers"][0], np.array([0.0]), atol=1e-9)
    np.testing.assert_allclose(result["mixed_multipliers"][0], np.array([2.0]), atol=1e-9)
    assert result["state_multipliers"][0].shape == (0,)
    assert result["terminal_state_multiplier"].shape == (0,)


def test_python_multiplier_shapes_multistage():
    clqr = _load_extension()
    x0 = np.array([0.5, -0.2], dtype=np.float64)
    u0 = np.array([0.3], dtype=np.float64)
    A0 = np.array([[1.0, 0.1], [0.0, 1.0]], dtype=np.float64)
    B0 = np.array([[0.2], [1.0]], dtype=np.float64)
    c0 = np.array([0.0, 0.0], dtype=np.float64)
    x1 = A0 @ x0 + B0 @ u0 + c0

    u1 = np.array([-0.4], dtype=np.float64)
    A1 = np.array([[0.9, 0.0], [0.1, 1.0]], dtype=np.float64)
    B1 = np.array([[1.0], [0.3]], dtype=np.float64)
    c1 = np.array([0.1, -0.1], dtype=np.float64)
    x2 = A1 @ x1 + B1 @ u1 + c1

    C0 = np.array([[1.0, -1.0]], dtype=np.float64)
    D0 = np.array([[0.5]], dtype=np.float64)
    E1 = np.array([[1.0, 0.0]], dtype=np.float64)
    terminal_E = np.array([[0.0, 1.0]], dtype=np.float64)
    problem = {
        "initial_state": x0,
        "stages": [
            {
                "A": A0,
                "B": B0,
                "c": c0,
                "Q": np.eye(2, dtype=np.float64),
                "R": np.array([[2.0]], dtype=np.float64),
                "M": np.zeros((2, 1), dtype=np.float64),
                "q": np.array([0.1, -0.1], dtype=np.float64),
                "r": np.array([0.2], dtype=np.float64),
                "C": C0,
                "D": D0,
                "d": -(C0 @ x0 + D0 @ u0),
            },
            {
                "A": A1,
                "B": B1,
                "c": c1,
                "Q": 1.2 * np.eye(2, dtype=np.float64),
                "R": np.array([[1.5]], dtype=np.float64),
                "M": np.zeros((2, 1), dtype=np.float64),
                "q": np.array([0.0, 0.2], dtype=np.float64),
                "r": np.array([-0.1], dtype=np.float64),
                "E": E1,
                "e": -(E1 @ x1),
            },
        ],
        "terminal_Q": 2.0 * np.eye(2, dtype=np.float64),
        "terminal_q": np.array([0.0, 0.0], dtype=np.float64),
        "terminal_E": terminal_E,
        "terminal_e": -(terminal_E @ x2),
    }
    result = clqr.solve(problem)
    assert result["status"] == "optimal", result
    assert len(result["states"]) == 3
    assert len(result["controls"]) == 2
    assert result["initial_multiplier"].shape == (2,)
    assert [x.shape for x in result["dynamics_multipliers"]] == [(2,), (2,)]
    assert [x.shape for x in result["mixed_multipliers"]] == [(1,), (0,)]
    assert [x.shape for x in result["state_multipliers"]] == [(0,), (1,)]
    assert result["terminal_state_multiplier"].shape == (1,)


if __name__ == "__main__":
    test_python_solve()
    test_python_multiplier_shapes_multistage()
