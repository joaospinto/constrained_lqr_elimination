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


if __name__ == "__main__":
    test_python_solve()
