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
                "R": np.array([[2.0]], dtype=np.float64),
                "M": np.array([[0.0]], dtype=np.float64),
                "r": np.array([0.0], dtype=np.float64),
                "C": np.array([[1.0]], dtype=np.float64),
                "D": np.array([[1.0]], dtype=np.float64),
                "d": np.array([0.0], dtype=np.float64),
            }
        ],
        "Q": [
            np.array([[1.0]], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
        ],
        "q": [
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        ],
    }
    result = clqr.solve(problem)
    assert result["status"] == "optimal", result
    assert result["newton_kkt_singular"] is False
    assert result["newton_kkt_wrong_inertia"] is False
    assert result["newton_kkt_diagnostic"] == ""
    np.testing.assert_allclose(result["states"][0], np.array([1.0]), atol=1e-9)
    np.testing.assert_allclose(result["controls"][0], np.array([-1.0]), atol=1e-9)
    np.testing.assert_allclose(result["states"][1], np.array([0.0]), atol=1e-9)
    np.testing.assert_allclose(
        result["initial_multiplier"], np.array([-3.0]), atol=1e-9
    )
    np.testing.assert_allclose(
        result["dynamics_multipliers"][0], np.array([0.0]), atol=1e-9
    )
    np.testing.assert_allclose(
        result["mixed_multipliers"][0], np.array([2.0]), atol=1e-9
    )
    assert result["state_multipliers"][0].shape == (0,)
    assert result["terminal_state_multiplier"].shape == (0,)


def _unconstrained_factor_problem():
    return {
        "initial_state": np.array([0.8, -0.3], dtype=np.float64),
        "stages": [
            {
                "A": np.array([[1.0, 0.2], [0.0, 0.9]], dtype=np.float64),
                "B": np.array([[0.1], [0.7]], dtype=np.float64),
                "c": np.array([0.05, -0.1], dtype=np.float64),
                "R": np.array([[1.7]], dtype=np.float64),
                "M": np.array([[0.05], [-0.03]], dtype=np.float64),
                "r": np.array([0.12], dtype=np.float64),
            },
            {
                "A": np.array([[0.95, 0.0], [0.1, 1.05]], dtype=np.float64),
                "B": np.array([[0.3], [0.4]], dtype=np.float64),
                "c": np.array([-0.02, 0.08], dtype=np.float64),
                "R": np.array([[2.2]], dtype=np.float64),
                "M": np.array([[0.02], [0.04]], dtype=np.float64),
                "r": np.array([-0.2], dtype=np.float64),
            },
        ],
        "Q": [
            np.array([[2.0, 0.1], [0.1, 1.5]], dtype=np.float64),
            np.array([[1.4, 0.0], [0.0, 1.8]], dtype=np.float64),
            np.array([[2.5, 0.2], [0.2, 3.0]], dtype=np.float64),
        ],
        "q": [
            np.array([0.2, -0.15], dtype=np.float64),
            np.array([-0.1, 0.25], dtype=np.float64),
            np.array([0.3, -0.35], dtype=np.float64),
        ],
    }


def _assert_primal_results_close(actual, expected):
    assert actual["status"] == expected["status"] == "optimal", (actual, expected)
    assert len(actual["states"]) == len(expected["states"])
    assert len(actual["controls"]) == len(expected["controls"])
    for got, want in zip(actual["states"], expected["states"]):
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)
    for got, want in zip(actual["controls"], expected["controls"]):
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        actual["initial_multiplier"],
        expected["initial_multiplier"],
        rtol=2e-5,
        atol=2e-6,
    )
    for got, want in zip(
        actual["dynamics_multipliers"], expected["dynamics_multipliers"]
    ):
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)
    for key in ("mixed_multipliers", "state_multipliers"):
        for got, want in zip(actual[key], expected[key]):
            np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        actual["terminal_state_multiplier"],
        expected["terminal_state_multiplier"],
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        actual["objective"], expected["objective"], rtol=2e-5, atol=2e-6
    )


def test_python_reusable_factorization():
    clqr = _load_extension()
    problem = _unconstrained_factor_problem()
    factors = clqr.factor(problem)
    solve_rhs = clqr.rhs(problem)
    assert factors.stage_count == 2
    assert factors.workspace_bytes > 0
    assert solve_rhs.stage_count == 2

    first = factors.solve(solve_rhs)
    _assert_primal_results_close(first, clqr.solve(problem))

    changed = _unconstrained_factor_problem()
    changed["initial_state"] = np.array([-0.45, 0.65], dtype=np.float64)
    changed["stages"][0]["c"] = np.array([-0.2, 0.3], dtype=np.float64)
    changed["q"][0] = np.array([-0.4, 0.55], dtype=np.float64)
    changed["stages"][0]["r"] = np.array([0.35], dtype=np.float64)
    changed["stages"][1]["c"] = np.array([0.15, -0.25], dtype=np.float64)
    changed["q"][1] = np.array([0.45, -0.5], dtype=np.float64)
    changed["stages"][1]["r"] = np.array([-0.3], dtype=np.float64)
    changed["q"][-1] = np.array([0.7, -0.8], dtype=np.float64)

    solve_rhs.initial_state = changed["initial_state"]
    for index in range(2):
        stage_rhs = solve_rhs.stage(index)
        stage_rhs.c = changed["stages"][index]["c"]
        stage_rhs.r = changed["stages"][index]["r"]
        solve_rhs.set_q(index, changed["q"][index])
    solve_rhs.set_q(2, changed["q"][-1])
    second = factors.solve(solve_rhs)
    _assert_primal_results_close(second, clqr.solve(changed))

    # The factorization owns its matrix data and does not observe later changes
    # to the input mapping.
    changed["stages"][0]["A"][0, 0] = 20.0
    repeated = factors.solve(solve_rhs)
    _assert_primal_results_close(repeated, second)

    solve_rhs.set_q(0, np.zeros(1, dtype=np.float64))
    invalid = factors.solve(solve_rhs)
    assert invalid["status"] == "invalid_input", invalid
    assert "q entry shape mismatch" in invalid["message"]


def test_python_constrained_factorization():
    clqr = _load_extension()
    problem = {
        "initial_state": np.array([1.0], dtype=np.float64),
        "stages": [
            {
                "A": np.array([[1.0]], dtype=np.float64),
                "B": np.array([[1.0]], dtype=np.float64),
                "c": np.array([0.0], dtype=np.float64),
                "R": np.array([[2.0]], dtype=np.float64),
                "M": np.array([[0.0]], dtype=np.float64),
                "r": np.array([0.0], dtype=np.float64),
                "C": np.array([[1.0]], dtype=np.float64),
                "D": np.array([[1.0]], dtype=np.float64),
                "d": np.array([0.0], dtype=np.float64),
                "E": np.array([[1.0]], dtype=np.float64),
                "e": np.array([-1.0], dtype=np.float64),
            }
        ],
        "Q": [
            np.array([[1.0]], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
        ],
        "q": [
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        ],
        "terminal_E": np.array([[1.0]], dtype=np.float64),
        "terminal_e": np.array([0.0], dtype=np.float64),
    }
    factors = clqr.factor(problem)
    rhs = clqr.rhs(problem)
    _assert_primal_results_close(factors.solve(rhs), clqr.solve(problem))

    changed = {
        **problem,
        "initial_state": np.array([0.4], dtype=np.float64),
        "stages": [{**problem["stages"][0]}],
        "q": [
            np.array([0.2], dtype=np.float64),
            np.array([-0.3], dtype=np.float64),
        ],
        "terminal_e": np.array([0.1], dtype=np.float64),
    }
    changed["stages"][0]["c"] = np.array([0.2], dtype=np.float64)
    changed["stages"][0]["r"] = np.array([-0.1], dtype=np.float64)
    changed["stages"][0]["d"] = np.array([0.3], dtype=np.float64)
    changed["stages"][0]["e"] = np.array([-0.4], dtype=np.float64)
    rhs.initial_state = changed["initial_state"]
    rhs.set_q(0, changed["q"][0])
    rhs.set_q(1, changed["q"][1])
    rhs.terminal_e = changed["terminal_e"]
    stage_rhs = rhs.stage(0)
    stage_rhs.c = changed["stages"][0]["c"]
    stage_rhs.r = changed["stages"][0]["r"]
    stage_rhs.d = changed["stages"][0]["d"]
    stage_rhs.e = changed["stages"][0]["e"]
    _assert_primal_results_close(factors.solve(rhs), clqr.solve(changed))

    stage_rhs.e = np.array([-0.5], dtype=np.float64)
    infeasible = factors.solve(rhs)
    assert infeasible["status"] == "infeasible", infeasible
    stage_rhs.e = changed["stages"][0]["e"]
    stage_rhs.d = np.zeros(0, dtype=np.float64)
    invalid = factors.solve(rhs)
    assert invalid["status"] == "invalid_input", invalid
    assert "d shape mismatch" in invalid["message"]


def test_python_factorization_reports_indefinite_factor():
    clqr = _load_extension()
    problem = {
        "initial_state": np.array([0.4], dtype=np.float64),
        "stages": [
            {
                "A": np.array([[0.8]], dtype=np.float64),
                "B": np.array([[0.2, -0.3]], dtype=np.float64),
                "c": np.array([0.1], dtype=np.float64),
                "R": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
                "M": np.array([[0.5, -0.25]], dtype=np.float64),
                "r": np.array([0.3, -0.6], dtype=np.float64),
            }
        ],
        "Q": [
            np.array([[1.2]], dtype=np.float64),
            np.array([[0.0]], dtype=np.float64),
        ],
        "q": [
            np.array([-0.2], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        ],
    }
    factors = clqr.factor(problem)
    factored = factors.solve(clqr.rhs(problem))
    ordinary = clqr.solve(problem)
    _assert_primal_results_close(factored, ordinary)
    assert factored["newton_kkt_singular"] is False
    assert factored["newton_kkt_wrong_inertia"] is True
    assert "wrong inertia" in factored["newton_kkt_diagnostic"]


def test_python_factorization_rejects_nonfinite_inputs():
    clqr = _load_extension()
    nonfinite_matrix = _unconstrained_factor_problem()
    nonfinite_matrix["Q"][0][0, 0] = np.nan
    try:
        clqr.factor(nonfinite_matrix)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("nonfinite factor matrix unexpectedly accepted")

    for tolerance in (np.inf, 0.0):
        try:
            clqr.factor(_unconstrained_factor_problem(), tolerance=tolerance)
        except ValueError as error:
            assert "tolerance" in str(error)
        else:
            raise AssertionError("invalid factor tolerance unexpectedly accepted")

    problem = _unconstrained_factor_problem()
    factors = clqr.factor(problem)
    solve_rhs = clqr.rhs(problem)
    solve_rhs.stage(0).c = np.array([np.inf, 0.0], dtype=np.float64)
    result = factors.solve(solve_rhs)
    assert result["status"] == "invalid_input", result
    assert "finite" in result["message"]


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
                "R": np.array([[2.0]], dtype=np.float64),
                "M": np.zeros((2, 1), dtype=np.float64),
                "r": np.array([0.2], dtype=np.float64),
                "C": C0,
                "D": D0,
                "d": -(C0 @ x0 + D0 @ u0),
            },
            {
                "A": A1,
                "B": B1,
                "c": c1,
                "R": np.array([[1.5]], dtype=np.float64),
                "M": np.zeros((2, 1), dtype=np.float64),
                "r": np.array([-0.1], dtype=np.float64),
                "E": E1,
                "e": -(E1 @ x1),
            },
        ],
        "Q": [
            np.eye(2, dtype=np.float64),
            1.2 * np.eye(2, dtype=np.float64),
            2.0 * np.eye(2, dtype=np.float64),
        ],
        "q": [
            np.array([0.1, -0.1], dtype=np.float64),
            np.array([0.0, 0.2], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
        ],
        "terminal_E": terminal_E,
        "terminal_e": -(terminal_E @ x2),
    }
    result = clqr.solve(problem)
    assert result["status"] == "optimal", result
    assert result["newton_kkt_singular"] is True
    assert result["newton_kkt_wrong_inertia"] is False
    assert len(result["states"]) == 3
    assert len(result["controls"]) == 2
    assert result["initial_multiplier"].shape == (2,)
    assert [x.shape for x in result["dynamics_multipliers"]] == [(2,), (2,)]
    assert [x.shape for x in result["mixed_multipliers"]] == [(1,), (0,)]
    assert [x.shape for x in result["state_multipliers"]] == [(0,), (1,)]
    assert result["terminal_state_multiplier"].shape == (1,)


if __name__ == "__main__":
    test_python_solve()
    test_python_reusable_factorization()
    test_python_constrained_factorization()
    test_python_factorization_reports_indefinite_factor()
    test_python_factorization_rejects_nonfinite_inputs()
    test_python_multiplier_shapes_multistage()
