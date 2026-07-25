import importlib
import importlib.util
import os
import pathlib
import sys

import jax
import numpy as np


def _load_extension(root, name):
    candidates = list(root.rglob(f"{name}.so"))
    if not candidates:
        candidates = list(root.rglob(f"{name}.*.so"))
    if not candidates:
        raise RuntimeError(f"could not find {name} in Bazel runfiles")
    spec = importlib.util.spec_from_file_location(name, candidates[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_modules():
    root = pathlib.Path(os.environ["TEST_SRCDIR"])
    _load_extension(root, "_clqr")
    cpu = _load_extension(root, "_clqr_jax_cpu")
    metal = _load_extension(root, "_clqr_metal")
    packages = list(root.rglob("python/clqr/__init__.py"))
    if not packages:
        raise RuntimeError("could not find python/clqr in Bazel runfiles")
    python_root = str(packages[0].parents[1])
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    module = importlib.import_module("clqr.jax")
    assert np.dtype(cpu.scalar_dtype) == np.dtype(metal.scalar_dtype)
    assert np.dtype(metal.scalar_dtype) == np.dtype(np.float32)
    assert module.metal_registered
    return module


def _problem(dtype, initial_state=1.0):
    return {
        "initial_state": np.array([initial_state], dtype=dtype),
        "stages": [
            {
                "A": np.array([[1.0]], dtype=dtype),
                "B": np.array([[1.0]], dtype=dtype),
                "c": np.array([0.0], dtype=dtype),
                "Q": np.array([[1.0]], dtype=dtype),
                "R": np.array([[2.0]], dtype=dtype),
                "M": np.array([[0.0]], dtype=dtype),
                "q": np.array([0.0], dtype=dtype),
                "r": np.array([0.0], dtype=dtype),
                "C": np.array([[1.0]], dtype=dtype),
                "D": np.array([[1.0]], dtype=dtype),
                "d": np.array([0.0], dtype=dtype),
            }
        ],
        "terminal_Q": np.array([[1.0]], dtype=dtype),
        "terminal_q": np.array([0.0], dtype=dtype),
    }


def _heterogeneous_problem(dtype):
    return {
        "initial_state": np.array([0.4, -0.2], dtype=dtype),
        "stages": [
            {
                "A": np.array([[1.0, 0.3]], dtype=dtype),
                "B": np.array([[0.5]], dtype=dtype),
                "c": np.array([0.1], dtype=dtype),
                "Q": np.eye(2, dtype=dtype),
                "R": np.array([[2.0]], dtype=dtype),
                "M": np.zeros((2, 1), dtype=dtype),
                "q": np.array([0.1, -0.1], dtype=dtype),
                "r": np.array([0.2], dtype=dtype),
            },
            {
                "A": np.array([[1.0], [-0.5]], dtype=dtype),
                "B": np.zeros((2, 0), dtype=dtype),
                "c": np.array([0.0, 0.1], dtype=dtype),
                "Q": np.array([[1.5]], dtype=dtype),
                "R": np.zeros((0, 0), dtype=dtype),
                "M": np.zeros((1, 0), dtype=dtype),
                "q": np.array([0.3], dtype=dtype),
                "r": np.zeros((0,), dtype=dtype),
            },
        ],
        "terminal_Q": 2.0 * np.eye(2, dtype=dtype),
        "terminal_q": np.array([-0.1, 0.2], dtype=dtype),
    }


_NUMERICAL_FIELDS = (
    "objective",
    "states",
    "controls",
    "initial_multiplier",
    "dynamics_multipliers",
    "mixed_multipliers",
    "state_multipliers",
    "terminal_state_multiplier",
)


def _max_abs(value):
    array = np.asarray(value)
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


def _max_primal_residual(problem, result):
    states = np.asarray(result.states)
    controls = np.asarray(result.controls)
    residual = _max_abs(
        states[0, : len(problem["initial_state"])]
        - np.asarray(problem["initial_state"])
    )
    for index, stage in enumerate(problem["stages"]):
        A = np.asarray(stage["A"])
        B = np.asarray(stage["B"])
        n = A.shape[1]
        next_n = A.shape[0]
        m = B.shape[1]
        x = states[index, :n]
        u = controls[index, :m]
        residual = max(
            residual,
            _max_abs(
                states[index + 1, :next_n] - A @ x - B @ u - np.asarray(stage["c"])
            ),
        )
        C = np.asarray(stage.get("C", np.zeros((0, n), dtype=x.dtype)))
        D = np.asarray(stage.get("D", np.zeros((C.shape[0], m), dtype=x.dtype)))
        d = np.asarray(stage.get("d", np.zeros((C.shape[0],), dtype=x.dtype)))
        E = np.asarray(stage.get("E", np.zeros((0, n), dtype=x.dtype)))
        e = np.asarray(stage.get("e", np.zeros((E.shape[0],), dtype=x.dtype)))
        residual = max(
            residual,
            _max_abs(C @ x + D @ u + d),
            _max_abs(E @ x + e),
        )
    terminal_E = np.asarray(
        problem.get(
            "terminal_E",
            np.zeros((0, np.asarray(problem["terminal_Q"]).shape[0])),
        )
    )
    terminal_e = np.asarray(problem.get("terminal_e", np.zeros((terminal_E.shape[0],))))
    return max(
        residual,
        _max_abs(terminal_E @ states[-1, : terminal_E.shape[1]] + terminal_e),
    )


def _max_kkt_residual(problem, result):
    states = np.asarray(result.states)
    controls = np.asarray(result.controls)
    initial_multiplier = np.asarray(result.initial_multiplier)
    dynamics_multipliers = np.asarray(result.dynamics_multipliers)
    mixed_multipliers = np.asarray(result.mixed_multipliers)
    state_multipliers = np.asarray(result.state_multipliers)
    terminal_multiplier = np.asarray(result.terminal_state_multiplier)
    stages = problem["stages"]
    residual = _max_abs(
        states[0, : len(problem["initial_state"])]
        - np.asarray(problem["initial_state"])
    )
    for index, stage in enumerate(stages):
        A = np.asarray(stage["A"])
        B = np.asarray(stage["B"])
        c = np.asarray(stage["c"])
        Q = np.asarray(stage["Q"])
        R = np.asarray(stage["R"])
        M = np.asarray(stage["M"])
        q = np.asarray(stage["q"])
        r = np.asarray(stage["r"])
        n = A.shape[1]
        next_n = A.shape[0]
        m = B.shape[1]
        x = states[index, :n]
        u = controls[index, :m]
        next_x = states[index + 1, :next_n]
        dynamics = dynamics_multipliers[index, :next_n]
        residual = max(residual, _max_abs(next_x - A @ x - B @ u - c))

        C = np.asarray(stage.get("C", np.zeros((0, n), dtype=x.dtype)))
        D = np.asarray(stage.get("D", np.zeros((C.shape[0], m), dtype=x.dtype)))
        d = np.asarray(stage.get("d", np.zeros((C.shape[0],), dtype=x.dtype)))
        mixed = mixed_multipliers[index, : C.shape[0]]
        residual = max(residual, _max_abs(C @ x + D @ u + d))

        E = np.asarray(stage.get("E", np.zeros((0, n), dtype=x.dtype)))
        e = np.asarray(stage.get("e", np.zeros((E.shape[0],), dtype=x.dtype)))
        state = state_multipliers[index, : E.shape[0]]
        residual = max(residual, _max_abs(E @ x + e))

        state_gradient = Q @ x + M @ u + q - A.T @ dynamics
        state_gradient += (
            initial_multiplier[:n]
            if index == 0
            else dynamics_multipliers[index - 1, :n]
        )
        state_gradient += C.T @ mixed + E.T @ state
        residual = max(residual, _max_abs(state_gradient))
        control_gradient = M.T @ x + R @ u + r - B.T @ dynamics
        control_gradient += D.T @ mixed
        residual = max(residual, _max_abs(control_gradient))

    terminal_Q = np.asarray(problem["terminal_Q"])
    terminal_q = np.asarray(problem["terminal_q"])
    terminal_n = terminal_Q.shape[0]
    terminal_x = states[-1, :terminal_n]
    terminal_E = np.asarray(
        problem.get("terminal_E", np.zeros((0, terminal_n), dtype=terminal_x.dtype))
    )
    terminal_e = np.asarray(
        problem.get(
            "terminal_e",
            np.zeros((terminal_E.shape[0],), dtype=terminal_x.dtype),
        )
    )
    residual = max(residual, _max_abs(terminal_E @ terminal_x + terminal_e))
    terminal_gradient = terminal_Q @ terminal_x + terminal_q
    terminal_gradient += (
        initial_multiplier[:terminal_n]
        if not stages
        else dynamics_multipliers[-1, :terminal_n]
    )
    terminal_gradient += terminal_E.T @ terminal_multiplier[: terminal_E.shape[0]]
    return max(residual, _max_abs(terminal_gradient))


def _compare_cpu_and_metal(
    clqr_jax,
    problem,
    atol=2e-4,
    *,
    compare_multipliers=True,
    kkt_tolerance=2e-3,
):
    packed = clqr_jax.pack_problem(problem, dtype=np.float32)
    cpu = jax.jit(clqr_jax.solve)(packed)
    metal = jax.jit(lambda value: clqr_jax.solve(value, backend="metal"))(packed)
    if int(cpu.status) != int(metal.status):
        state_difference = _max_abs(np.asarray(cpu.states) - np.asarray(metal.states))
        control_difference = _max_abs(
            np.asarray(cpu.controls) - np.asarray(metal.controls)
        )
        primal_residual = _max_primal_residual(problem, metal)
        raise AssertionError(
            f"status: CPU={int(cpu.status)}, Metal={int(metal.status)}, "
            f"diagnostics={np.asarray(metal.diagnostics).tolist()}, "
            f"state difference={state_difference:.6g}, "
            f"control difference={control_difference:.6g}, "
            f"primal residual={primal_residual:.6g}"
        )
    # Native device paths do not form the CPU-only diagnostic Newton KKT, so
    # compare the public status code and all numerical solution fields.
    fields = _NUMERICAL_FIELDS if compare_multipliers else _NUMERICAL_FIELDS[:3]
    residual = _max_kkt_residual(problem, metal)
    for name in fields:
        try:
            np.testing.assert_allclose(
                getattr(metal, name),
                getattr(cpu, name),
                atol=atol,
                rtol=atol,
                err_msg=name,
            )
        except AssertionError as error:
            raise AssertionError(
                f"{name}; Metal full KKT residual is {residual:.6g}: {error}"
            ) from error
    if residual > kkt_tolerance:
        raise AssertionError(f"Metal full KKT residual is {residual:.6g}")
    return packed, metal


def _random_feasible_problem(seed, horizon):
    rng = np.random.default_rng(seed)
    dtype = np.float32
    state_dimension = 2
    control_dimension = 2
    states = rng.normal(scale=0.3, size=(horizon + 1, state_dimension))
    controls = rng.normal(scale=0.2, size=(horizon, control_dimension))
    stages = []
    for index in range(horizon):
        A = 0.45 * np.eye(state_dimension) + rng.normal(
            scale=0.04, size=(state_dimension, state_dimension)
        )
        B = rng.normal(scale=0.2, size=(state_dimension, control_dimension))
        q_factor = rng.normal(scale=0.25, size=(state_dimension, state_dimension))
        r_factor = rng.normal(scale=0.25, size=(control_dimension, control_dimension))
        stage = {
            "A": A.astype(dtype),
            "B": B.astype(dtype),
            "c": (states[index + 1] - A @ states[index] - B @ controls[index]).astype(
                dtype
            ),
            "Q": (q_factor @ q_factor.T + 0.7 * np.eye(state_dimension)).astype(dtype),
            "R": (r_factor @ r_factor.T + 1.0 * np.eye(control_dimension)).astype(
                dtype
            ),
            "M": rng.normal(
                scale=0.025,
                size=(state_dimension, control_dimension),
            ).astype(dtype),
            "q": rng.normal(scale=0.1, size=state_dimension).astype(dtype),
            "r": rng.normal(scale=0.1, size=control_dimension).astype(dtype),
        }
        if index % 2 == 0:
            C = rng.normal(scale=0.4, size=(1, state_dimension))
            D = rng.normal(scale=0.4, size=(1, control_dimension))
            stage.update(
                C=C.astype(dtype),
                D=D.astype(dtype),
                d=(-(C @ states[index] + D @ controls[index])).astype(dtype),
            )
        if index > 0 and index % 3 == 1:
            E = rng.normal(scale=0.4, size=(1, state_dimension))
            stage.update(
                E=E.astype(dtype),
                e=(-(E @ states[index])).astype(dtype),
            )
        stages.append(stage)
    terminal_factor = rng.normal(scale=0.25, size=(state_dimension, state_dimension))
    terminal_E = rng.normal(scale=0.4, size=(1, state_dimension))
    return {
        "initial_state": states[0].astype(dtype),
        "stages": stages,
        "terminal_Q": (
            terminal_factor @ terminal_factor.T + 0.8 * np.eye(state_dimension)
        ).astype(dtype),
        "terminal_q": rng.normal(scale=0.1, size=state_dimension).astype(dtype),
        "terminal_E": terminal_E.astype(dtype),
        "terminal_e": (-(terminal_E @ states[-1])).astype(dtype),
    }


def _wide_constrained_problem():
    dtype = np.float32
    horizon = 5
    n = 8
    m = 4
    p = 2
    A = 0.9 * np.eye(n, dtype=dtype)
    A[1:, :-1] += 0.01 * np.eye(n - 1, dtype=dtype)
    B = np.zeros((n, m), dtype=dtype)
    B[:m, :] = 0.1 * np.eye(m, dtype=dtype)
    D = np.zeros((p, m), dtype=dtype)
    D[:, :p] = np.eye(p, dtype=dtype)
    stages = []
    for _ in range(horizon):
        stages.append(
            {
                "A": A.copy(),
                "B": B.copy(),
                "c": np.zeros(n, dtype=dtype),
                "Q": np.eye(n, dtype=dtype),
                "R": 2.0 * np.eye(m, dtype=dtype),
                "M": np.zeros((n, m), dtype=dtype),
                "q": np.zeros(n, dtype=dtype),
                "r": np.zeros(m, dtype=dtype),
                "C": np.zeros((p, n), dtype=dtype),
                "D": D.copy(),
                "d": np.zeros(p, dtype=dtype),
            }
        )
    return {
        "initial_state": np.linspace(0.1, 0.8, n, dtype=dtype),
        "stages": stages,
        "terminal_Q": 1.5 * np.eye(n, dtype=dtype),
        "terminal_q": np.zeros(n, dtype=dtype),
    }


def _wide_benign_problem():
    problem = _wide_constrained_problem()
    for stage in problem["stages"]:
        del stage["C"]
        del stage["D"]
        del stage["d"]
    return problem


def _sliced_primal_leaf_problem(horizon, constraint_kind):
    dtype = np.float32
    n = 8
    m = 4
    p = 2
    state = np.linspace(0.1, 0.8, n, dtype=dtype)
    A = 0.9 * np.eye(n, dtype=dtype)
    B = np.zeros((n, m), dtype=dtype)
    B[p : p + m, :] = 0.1 * np.eye(m, dtype=dtype)
    E = np.zeros((p, n), dtype=dtype)
    E[:, :p] = np.eye(p, dtype=dtype)
    stages = []
    for _ in range(horizon):
        stage = {
            "A": A.copy(),
            "B": B.copy(),
            "c": 0.1 * state,
            "Q": np.eye(n, dtype=dtype),
            "R": 2.0 * np.eye(m, dtype=dtype),
            "M": np.zeros((n, m), dtype=dtype),
            "q": np.zeros(n, dtype=dtype),
            "r": np.zeros(m, dtype=dtype),
        }
        if constraint_kind == "state-only":
            stage["E"] = E.copy()
            stage["e"] = -(E @ state)
        elif constraint_kind == "mixed":
            stage["C"] = E.copy()
            stage["D"] = np.zeros((p, m), dtype=dtype)
            stage["d"] = -(E @ state)
        elif constraint_kind == "rank-deficient":
            repeated = np.stack((E[0], E[0]))
            stage["E"] = repeated
            stage["e"] = -(repeated @ state)
        elif constraint_kind == "scaled":
            scales = np.asarray((1.0e-3, 1.0e3), dtype=dtype)
            scaled = E * scales[:, None]
            stage["E"] = scaled
            stage["e"] = -(scaled @ state)
        else:
            raise ValueError(constraint_kind)
        stages.append(stage)
    return {
        "initial_state": state,
        "stages": stages,
        "terminal_Q": 1.5 * np.eye(n, dtype=dtype),
        "terminal_q": np.zeros(n, dtype=dtype),
    }


def test_metal_eager_jit_and_vmap():
    clqr_jax = _load_modules()
    try:
        clqr_jax.solve(_problem(np.float64), backend="metal")
    except ValueError as error:
        assert "requires explicit float32 inputs" in str(error)
    else:
        raise AssertionError("Metal silently cast a float64 mapping")

    mixed_dtype = _problem(np.float32)
    mixed_dtype["stages"][0]["q"] = np.array([0.0], dtype=np.float64)
    try:
        clqr_jax.solve(mixed_dtype, backend="metal")
    except ValueError as error:
        assert "problem.stages[0].q uses float64" in str(error)
    else:
        raise AssertionError("Metal silently cast a nested float64 leaf")

    dtype = np.dtype(np.float32)
    packed = clqr_jax.pack_problem(_problem(dtype), dtype=dtype)

    def metal_solve(value):
        return clqr_jax.solve(value, backend="metal")

    eager = metal_solve(packed)
    compiled = jax.jit(metal_solve)(packed)
    assert int(eager.status) == clqr_jax.SolveStatus.OPTIMAL
    np.testing.assert_allclose(eager.states[:, 0], [1.0, 0.0], atol=3e-5)
    np.testing.assert_allclose(eager.controls[:, 0], [-1.0], atol=3e-5)
    np.testing.assert_allclose(eager.mixed_multipliers[:, 0], [2.0], atol=3e-5)
    for actual, expected in zip(compiled, eager):
        np.testing.assert_allclose(actual, expected, atol=3e-5)

    changed = packed._replace(
        rhs=packed.rhs._replace(initial_state=2.0 * packed.rhs.initial_state)
    )
    batched = jax.tree.map(lambda *leaves: np.stack(leaves), packed, changed)
    result = jax.jit(jax.vmap(metal_solve))(batched)
    np.testing.assert_allclose(result.states[:, 0, 0], [1.0, 2.0], atol=3e-5)


def test_metal_heterogeneous_zero_control_and_reuse():
    clqr_jax = _load_modules()
    packed, first = _compare_cpu_and_metal(clqr_jax, _heterogeneous_problem(np.float32))
    assert int(first.status) == clqr_jax.SolveStatus.OPTIMAL
    # Put a NaN in an entry that occupies an alignment gap in the smaller
    # layout below. Reusing the larger thread-local arena must not leak that
    # stale value into the smaller solve's finite-input scan.
    invalid_A = np.asarray(packed.factors.A).copy()
    invalid_A[0, 0, 1] = np.nan
    invalid = packed._replace(factors=packed.factors._replace(A=invalid_A))
    invalid_result = clqr_jax.solve(invalid, backend="metal")
    assert int(invalid_result.status) == clqr_jax.SolveStatus.INVALID_INPUT

    zero_control = {
        "initial_state": np.array([0.25], dtype=np.float32),
        "stages": [
            {
                "A": np.array([[0.9]], dtype=np.float32),
                "B": np.zeros((1, 0), dtype=np.float32),
                "c": np.array([0.1], dtype=np.float32),
                "Q": np.array([[1.0]], dtype=np.float32),
                "R": np.zeros((0, 0), dtype=np.float32),
                "M": np.zeros((1, 0), dtype=np.float32),
                "q": np.array([0.0], dtype=np.float32),
                "r": np.zeros((0,), dtype=np.float32),
            }
        ],
        "terminal_Q": np.array([[2.0]], dtype=np.float32),
        "terminal_q": np.array([0.1], dtype=np.float32),
    }
    _compare_cpu_and_metal(clqr_jax, zero_control)
    second = jax.jit(lambda value: clqr_jax.solve(value, backend="metal"))(packed)
    for actual, expected in zip(second, first):
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)


def test_metal_zero_horizon_and_all_constraint_families():
    clqr_jax = _load_modules()
    zero_horizon = {
        "initial_state": np.array([0.5, -0.25], dtype=np.float32),
        "stages": [],
        "terminal_Q": np.eye(2, dtype=np.float32),
        "terminal_q": np.array([0.1, -0.2], dtype=np.float32),
    }
    _, result = _compare_cpu_and_metal(clqr_jax, zero_horizon)
    assert result.controls.shape == (0, 0)

    constrained = {
        "initial_state": np.array([0.5, -0.25], dtype=np.float32),
        "stages": [
            {
                "A": np.eye(2, dtype=np.float32),
                "B": np.eye(2, dtype=np.float32),
                "c": np.zeros(2, dtype=np.float32),
                "Q": np.array([[1.2, 0.1], [0.1, 1.5]], dtype=np.float32),
                "R": np.array([[2.0, 0.2], [0.2, 1.7]], dtype=np.float32),
                "M": np.array([[0.03, 0.01], [-0.02, 0.04]], dtype=np.float32),
                "q": np.array([0.1, -0.2], dtype=np.float32),
                "r": np.array([-0.1, 0.05], dtype=np.float32),
                "C": np.zeros((1, 2), dtype=np.float32),
                "D": np.array([[1.0, 0.0]], dtype=np.float32),
                "d": np.array([-0.1], dtype=np.float32),
            },
            {
                "A": np.eye(2, dtype=np.float32),
                "B": np.eye(2, dtype=np.float32),
                "c": np.zeros(2, dtype=np.float32),
                "Q": np.array([[1.4, -0.1], [-0.1, 1.1]], dtype=np.float32),
                "R": np.array([[1.8, 0.1], [0.1, 2.1]], dtype=np.float32),
                "M": np.array([[0.02, -0.03], [0.01, 0.02]], dtype=np.float32),
                "q": np.array([-0.03, 0.08], dtype=np.float32),
                "r": np.array([0.04, -0.02], dtype=np.float32),
                "E": np.array([[0.0, 1.0]], dtype=np.float32),
                "e": np.array([-0.2], dtype=np.float32),
            },
        ],
        "terminal_Q": np.array([[1.7, 0.05], [0.05, 1.3]], dtype=np.float32),
        "terminal_q": np.array([0.02, -0.04], dtype=np.float32),
        "terminal_E": np.array([[1.0, 0.0]], dtype=np.float32),
        "terminal_e": np.array([-0.3], dtype=np.float32),
    }
    _, result = _compare_cpu_and_metal(clqr_jax, constrained, atol=4e-4)
    assert int(result.status) == clqr_jax.SolveStatus.OPTIMAL

    _, result = _compare_cpu_and_metal(
        clqr_jax,
        _wide_benign_problem(),
        atol=4e-3,
        kkt_tolerance=4e-3,
    )
    assert int(result.status) == clqr_jax.SolveStatus.OPTIMAL

    # Keep genuinely wide mixed-constraint coverage without making the
    # secondary FP32 path fail on a noisy multiplier-recovery boundary.
    wide_constrained = _wide_constrained_problem()
    packed = clqr_jax.pack_problem(wide_constrained, dtype=np.float32)
    cpu = jax.jit(clqr_jax.solve)(packed)
    metal = jax.jit(lambda value: clqr_jax.solve(value, backend="metal"))(packed)
    assert int(cpu.status) == clqr_jax.SolveStatus.OPTIMAL
    assert int(metal.status) in (
        clqr_jax.SolveStatus.OPTIMAL,
        clqr_jax.SolveStatus.NUMERICAL_FAILURE,
    )
    np.testing.assert_allclose(metal.states, cpu.states, atol=8e-3, rtol=8e-3)
    np.testing.assert_allclose(metal.controls, cpu.controls, atol=3e-2, rtol=3e-2)
    assert _max_primal_residual(wide_constrained, metal) <= 2e-6
    if int(metal.status) == clqr_jax.SolveStatus.OPTIMAL:
        assert _max_kkt_residual(wide_constrained, metal) <= 5e-2


def test_metal_rank_deficiency_and_failure_statuses():
    clqr_jax = _load_modules()
    redundant = _problem(np.float32)
    redundant["stages"][0]["E"] = np.array([[1.0], [2.0]], dtype=np.float32)
    redundant["stages"][0]["e"] = np.array([-1.0, -2.0], dtype=np.float32)
    _, result = _compare_cpu_and_metal(
        clqr_jax, redundant, atol=5e-4, compare_multipliers=False
    )
    assert int(result.status) == clqr_jax.SolveStatus.OPTIMAL

    infeasible = {
        "initial_state": np.array([1.0], dtype=np.float32),
        "stages": [],
        "terminal_Q": np.array([[1.0]], dtype=np.float32),
        "terminal_q": np.array([0.0], dtype=np.float32),
        "terminal_E": np.array([[1.0]], dtype=np.float32),
        "terminal_e": np.array([-2.0], dtype=np.float32),
    }
    packed = clqr_jax.pack_problem(infeasible, dtype=np.float32)
    result = clqr_jax.solve(packed, backend="metal")
    assert int(result.status) == clqr_jax.SolveStatus.INFEASIBLE
    np.testing.assert_array_equal(result.diagnostics[1:], [0, 0])

    nonfinite = clqr_jax.pack_problem(_problem(np.float32), dtype=np.float32)
    nonfinite_q = np.asarray(nonfinite.rhs.q).copy()
    nonfinite_q[0, 0] = np.nan
    nonfinite = nonfinite._replace(rhs=nonfinite.rhs._replace(q=nonfinite_q))
    result = clqr_jax.solve(nonfinite, backend="metal")
    assert int(result.status) == clqr_jax.SolveStatus.INVALID_INPUT

    singular = _problem(np.float32)
    singular["stages"][0]["R"] = np.array([[0.0]], dtype=np.float32)
    singular["stages"][0]["C"] = np.zeros((0, 1), dtype=np.float32)
    singular["stages"][0]["D"] = np.zeros((0, 1), dtype=np.float32)
    singular["stages"][0]["d"] = np.zeros((0,), dtype=np.float32)
    packed = clqr_jax.pack_problem(singular, dtype=np.float32)
    result = clqr_jax.solve(packed, backend="metal")
    assert int(result.status) == clqr_jax.SolveStatus.NUMERICAL_FAILURE


def test_metal_deterministic_random_properties():
    clqr_jax = _load_modules()
    accuracy_limit_tolerances = {
        11: (1e-4, 4e-4, 5e-3),
        19: (2e-3, 5e-3, 3e-2),
        29: (5e-3, 1e-2, 2e-1),
    }
    cases = (
        (13, 2),
        (3, 3),
        (17, 3),
        (7, 4),
        (11, 5),
        (19, 7),
        (29, 9),
    )
    for seed, horizon in cases:
        problem = _random_feasible_problem(seed, horizon)
        if seed in accuracy_limit_tolerances:
            packed = clqr_jax.pack_problem(problem, dtype=np.float32)
            cpu = jax.jit(clqr_jax.solve)(packed)
            metal = jax.jit(lambda value: clqr_jax.solve(value, backend="metal"))(
                packed
            )
            assert int(cpu.status) == clqr_jax.SolveStatus.OPTIMAL
            assert int(metal.status) in (
                clqr_jax.SolveStatus.OPTIMAL,
                clqr_jax.SolveStatus.NUMERICAL_FAILURE,
            )
            state_tolerance, control_tolerance, kkt_tolerance = (
                accuracy_limit_tolerances[seed]
            )
            np.testing.assert_allclose(
                metal.states,
                cpu.states,
                atol=state_tolerance,
                rtol=state_tolerance,
            )
            np.testing.assert_allclose(
                metal.controls,
                cpu.controls,
                atol=control_tolerance,
                rtol=control_tolerance,
            )
            assert _max_primal_residual(problem, metal) < 2e-6
            if int(metal.status) == clqr_jax.SolveStatus.OPTIMAL:
                assert _max_kkt_residual(problem, metal) < kkt_tolerance
            # These FP32 accuracy-limit cases may either return an honest
            # numerical failure or an accurate KKT point. Their dense KKT
            # condition numbers are approximately 7.88e2, 3.56e3, and 4.37e5.
            continue
        try:
            _, result = _compare_cpu_and_metal(
                clqr_jax,
                problem,
                atol=3e-3,
                kkt_tolerance=3e-3,
            )
        except AssertionError as error:
            raise AssertionError(
                f"property seed={seed}, horizon={horizon}: {error}"
            ) from error
        assert int(result.status) == clqr_jax.SolveStatus.OPTIMAL


def test_metal_long_horizon_and_dimension_bounds():
    clqr_jax = _load_modules()
    stage = {
        "A": np.array([[0.98]], dtype=np.float32),
        "B": np.array([[0.2]], dtype=np.float32),
        "c": np.array([0.01], dtype=np.float32),
        "Q": np.array([[1.0]], dtype=np.float32),
        "R": np.array([[2.0]], dtype=np.float32),
        "M": np.array([[0.01]], dtype=np.float32),
        "q": np.array([0.02], dtype=np.float32),
        "r": np.array([-0.01], dtype=np.float32),
    }
    for horizon in (127, 257, 1025):
        problem = {
            "initial_state": np.array([0.4], dtype=np.float32),
            "stages": [
                {key: value.copy() for key, value in stage.items()}
                for _ in range(horizon)
            ],
            "terminal_Q": np.array([[1.5]], dtype=np.float32),
            "terminal_q": np.array([0.1], dtype=np.float32),
        }
        _, result = _compare_cpu_and_metal(
            clqr_jax,
            problem,
            atol=3e-3,
            kkt_tolerance=3e-3,
        )
        assert int(result.status) == clqr_jax.SolveStatus.OPTIMAL

    packed = clqr_jax.pack_problem(_problem(np.float32), dtype=np.float32)
    bad_dimensions = np.asarray(packed.factors.dimensions).copy()
    bad_dimensions[0] = 2
    invalid = packed._replace(
        factors=packed.factors._replace(dimensions=bad_dimensions)
    )
    try:
        np.asarray(clqr_jax.solve(invalid, backend="metal").states)
    except Exception as error:
        assert "state dimension exceeds the padded state capacity" in str(error)
    else:
        raise AssertionError("invalid active dimensions were accepted")

    n = 128
    m = 64
    oversized_scratch = {
        "initial_state": np.zeros(n, dtype=np.float32),
        "stages": [
            {
                "A": 0.9 * np.eye(n, dtype=np.float32),
                "B": np.zeros((n, m), dtype=np.float32),
                "c": np.zeros(n, dtype=np.float32),
                "Q": np.eye(n, dtype=np.float32),
                "R": np.eye(m, dtype=np.float32),
                "M": np.zeros((n, m), dtype=np.float32),
                "q": np.zeros(n, dtype=np.float32),
                "r": np.zeros(m, dtype=np.float32),
            }
        ],
        "terminal_Q": np.eye(n, dtype=np.float32),
        "terminal_q": np.zeros(n, dtype=np.float32),
    }
    packed = clqr_jax.pack_problem(oversized_scratch, dtype=np.float32)
    try:
        np.asarray(clqr_jax.solve(packed, backend="metal").states)
    except Exception as error:
        assert "scratch exceeds device threadgroup memory" in str(error)
    else:
        raise AssertionError("oversized Metal threadgroup scratch was accepted")


def test_metal_sliced_primal_leaf_boundaries_and_constraints():
    clqr_jax = _load_modules()
    for horizon in (65, 127):
        for constraint_kind in (
            "state-only",
            "mixed",
            "rank-deficient",
            "scaled",
        ):
            problem = _sliced_primal_leaf_problem(horizon, constraint_kind)
            _, result = _compare_cpu_and_metal(
                clqr_jax,
                problem,
                atol=5e-3,
                compare_multipliers=False,
                kkt_tolerance=5e-2,
            )
            assert int(result.status) == clqr_jax.SolveStatus.OPTIMAL


if __name__ == "__main__":
    test_metal_eager_jit_and_vmap()
    test_metal_heterogeneous_zero_control_and_reuse()
    test_metal_zero_horizon_and_all_constraint_families()
    test_metal_rank_deficiency_and_failure_statuses()
    test_metal_deterministic_random_properties()
    test_metal_long_horizon_and_dimension_bounds()
    test_metal_sliced_primal_leaf_boundaries_and_constraints()
