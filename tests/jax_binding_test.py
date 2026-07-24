import importlib
import importlib.util
import os
import pathlib
import sys

import jax
import numpy as np


def _load_modules():
    root = pathlib.Path(os.environ["TEST_SRCDIR"])
    extension = _load_extension(root, "_clqr")
    _load_extension(root, "_clqr_jax_cpu")

    packages = list(root.rglob("python/clqr/__init__.py"))
    if not packages:
        raise RuntimeError("could not find python/clqr in Bazel runfiles")
    python_root = str(packages[0].parents[1])
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    clqr_jax = importlib.import_module("clqr.jax")
    return extension, clqr_jax


def _load_extension(root, name):
    extensions = list(root.rglob(f"{name}.so"))
    if not extensions:
        extensions = list(root.rglob(f"{name}.*.so"))
    if not extensions:
        raise RuntimeError(f"could not find {name} in Bazel runfiles")
    spec = importlib.util.spec_from_file_location(name, extensions[0])
    extension = importlib.util.module_from_spec(spec)
    sys.modules[name] = extension
    spec.loader.exec_module(extension)
    return extension


def _one_stage_problem(dtype):
    return {
        "initial_state": np.array([1.0], dtype=dtype),
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


def test_eager_and_jit():
    extension, clqr_jax = _load_modules()
    dtype = np.dtype(clqr_jax.scalar_dtype)
    if dtype == np.dtype(np.float64):
        jax.config.update("jax_enable_x64", True)
    packed = clqr_jax.pack_problem(_one_stage_problem(dtype), dtype=dtype)

    eager = clqr_jax.solve(packed)
    compiled = jax.jit(clqr_jax.solve)(packed)
    atol = 2e-5 if dtype == np.dtype(np.float32) else 1e-9
    assert int(eager.diagnostics[0]) == 0
    np.testing.assert_allclose(eager.states[:, 0], [1.0, 0.0], atol=atol)
    np.testing.assert_allclose(eager.controls[:, 0], [-1.0], atol=atol)
    np.testing.assert_allclose(eager.mixed_multipliers[:, 0], [2.0], atol=atol)
    for actual, expected in zip(compiled, eager):
        np.testing.assert_allclose(actual, expected, atol=atol)


def test_new_rhs_reuses_compiled_shape():
    extension, clqr_jax = _load_modules()
    dtype = np.dtype(clqr_jax.scalar_dtype)
    if dtype == np.dtype(np.float64):
        jax.config.update("jax_enable_x64", True)
    packed = clqr_jax.pack_problem(_one_stage_problem(dtype), dtype=dtype)
    solve = jax.jit(clqr_jax.solve)
    first = solve(packed)
    changed_rhs = packed.rhs._replace(
        initial_state=np.array([2.0], dtype=dtype)
    )
    second = solve(packed._replace(rhs=changed_rhs))
    assert int(first.diagnostics[0]) == 0
    assert int(second.diagnostics[0]) == 0
    np.testing.assert_allclose(first.states[0, 0], 1.0)
    np.testing.assert_allclose(second.states[0, 0], 2.0)
    np.testing.assert_allclose(second.controls[0, 0], -2.0)


def test_direct_mapping_and_sequential_vmap():
    _, clqr_jax = _load_modules()
    dtype = np.dtype(clqr_jax.scalar_dtype)
    if dtype == np.dtype(np.float64):
        jax.config.update("jax_enable_x64", True)
    problem = _one_stage_problem(dtype)
    direct = clqr_jax.solve(problem)
    assert int(direct.status) == clqr_jax.SolveStatus.OPTIMAL

    packed = clqr_jax.pack_problem(problem, dtype=dtype)
    second = packed._replace(
        rhs=packed.rhs._replace(
            initial_state=2.0 * packed.rhs.initial_state
        )
    )
    batched = jax.tree.map(lambda *leaves: np.stack(leaves), packed, second)
    result = jax.jit(jax.vmap(clqr_jax.solve))(batched)
    assert result.states.shape == (2, 2, 1)
    np.testing.assert_allclose(result.states[:, 0, 0], [1.0, 2.0])
    np.testing.assert_allclose(result.controls[:, 0, 0], [-1.0, -2.0])


def test_active_dimensions_are_bounds_checked():
    _, clqr_jax = _load_modules()
    dtype = np.dtype(clqr_jax.scalar_dtype)
    if dtype == np.dtype(np.float64):
        jax.config.update("jax_enable_x64", True)
    packed = clqr_jax.pack_problem(_one_stage_problem(dtype), dtype=dtype)
    invalid_dimensions = np.asarray(packed.factors.dimensions).copy()
    invalid_dimensions[0] = 2
    invalid = packed._replace(
        factors=packed.factors._replace(dimensions=invalid_dimensions)
    )
    try:
        np.asarray(clqr_jax.solve(invalid).states)
    except Exception as error:
        assert "state dimension exceeds the padded state capacity" in str(error)
    else:
        raise AssertionError("invalid active dimensions were accepted")


def test_heterogeneous_dimensions_match_python_solver():
    extension, clqr_jax = _load_modules()
    dtype = np.dtype(clqr_jax.scalar_dtype)
    if dtype == np.dtype(np.float64):
        jax.config.update("jax_enable_x64", True)
    problem = {
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
    packed = clqr_jax.pack_problem(problem, dtype=dtype)
    result = jax.jit(clqr_jax.solve)(packed)
    reference_problem = {
        key: (
            [
                {
                    stage_key: np.asarray(stage_value, dtype=np.float64)
                    for stage_key, stage_value in stage.items()
                }
                for stage in value
            ]
            if key == "stages"
            else np.asarray(value, dtype=np.float64)
        )
        for key, value in problem.items()
    }
    reference = extension.solve(reference_problem)
    atol = 3e-5 if dtype == np.dtype(np.float32) else 1e-9
    assert int(result.status) == clqr_jax.SolveStatus.OPTIMAL
    assert reference["status"] == "optimal"
    np.testing.assert_allclose(result.objective[0], reference["objective"], atol=atol)
    for node, state in enumerate(reference["states"]):
        np.testing.assert_allclose(
            result.states[node, : state.size], state, atol=atol
        )
    for stage, control in enumerate(reference["controls"]):
        np.testing.assert_allclose(
            result.controls[stage, : control.size], control, atol=atol
        )
    np.testing.assert_array_equal(
        np.asarray(packed.factors.dimensions),
        np.array([2, 1, 2, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32),
    )


def test_zero_horizon_and_zero_capacities():
    extension, clqr_jax = _load_modules()
    dtype = np.dtype(clqr_jax.scalar_dtype)
    if dtype == np.dtype(np.float64):
        jax.config.update("jax_enable_x64", True)
    problem = {
        "initial_state": np.array([0.5, -0.25], dtype=dtype),
        "stages": [],
        "terminal_Q": np.eye(2, dtype=dtype),
        "terminal_q": np.array([0.1, -0.2], dtype=dtype),
    }
    packed = clqr_jax.pack_problem(problem, dtype=dtype)
    result = jax.jit(clqr_jax.solve)(packed)
    assert int(result.diagnostics[0]) == 0
    assert result.controls.shape == (0, 0)
    assert result.mixed_multipliers.shape == (0, 0)
    np.testing.assert_allclose(result.states[0], problem["initial_state"])


if __name__ == "__main__":
    test_eager_and_jit()
    test_new_rhs_reuses_compiled_shape()
    test_direct_mapping_and_sequential_vmap()
    test_active_dimensions_are_bounds_checked()
    test_heterogeneous_dimensions_match_python_solver()
    test_zero_horizon_and_zero_capacities()
