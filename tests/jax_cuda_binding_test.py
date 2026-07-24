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
    cuda = _load_extension(root, "_clqr_cuda")
    packages = list(root.rglob("python/clqr/__init__.py"))
    if not packages:
        raise RuntimeError("could not find python/clqr in Bazel runfiles")
    python_root = str(packages[0].parents[1])
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    module = importlib.import_module("clqr.jax")
    assert cuda.scalar_dtype == cpu.scalar_dtype
    assert module.cuda_registered
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


def _cuda_device():
    devices = [
        device for device in jax.devices() if device.platform in ("gpu", "cuda")
    ]
    if not devices:
        raise RuntimeError("jax_cuda_binding_test requires a CUDA device")
    return devices[0]


def test_cuda_eager_jit_and_new_rhs():
    clqr_jax = _load_modules()
    device = _cuda_device()
    dtype = np.dtype(clqr_jax.scalar_dtype)
    if dtype == np.dtype(np.float64):
        jax.config.update("jax_enable_x64", True)
    packed = clqr_jax.pack_problem(_problem(dtype), dtype=dtype)
    packed = jax.device_put(packed, device)
    solve = jax.jit(clqr_jax.solve)
    atol = 3e-5 if dtype == np.dtype(np.float32) else 1e-9
    first = solve(packed)
    assert int(first.diagnostics[0]) == 0
    np.testing.assert_allclose(first.states[:, 0], [1.0, 0.0], atol=atol)
    np.testing.assert_allclose(first.controls[:, 0], [-1.0], atol=atol)

    changed = packed._replace(
        rhs=packed.rhs._replace(initial_state=2.0 * packed.rhs.initial_state)
    )
    second = solve(changed)
    assert int(second.diagnostics[0]) == 0
    np.testing.assert_allclose(second.states[0, 0], 2.0, atol=atol)
    np.testing.assert_allclose(second.controls[0, 0], -2.0, atol=atol)


def test_cuda_matches_cpu_for_heterogeneous_dimensions():
    clqr_jax = _load_modules()
    device = _cuda_device()
    dtype = np.dtype(clqr_jax.scalar_dtype)
    if dtype == np.dtype(np.float64):
        jax.config.update("jax_enable_x64", True)
    packed = clqr_jax.pack_problem(_heterogeneous_problem(dtype), dtype=dtype)
    solve = jax.jit(clqr_jax.solve)
    cpu = solve(jax.device_put(packed, jax.devices("cpu")[0]))
    cuda = solve(jax.device_put(packed, device))
    atol = 2e-4 if dtype == np.dtype(np.float32) else 1e-9
    for actual, expected in zip(cuda, cpu):
        np.testing.assert_allclose(actual, expected, atol=atol)


def test_cuda_sequential_vmap():
    clqr_jax = _load_modules()
    device = _cuda_device()
    dtype = np.dtype(clqr_jax.scalar_dtype)
    if dtype == np.dtype(np.float64):
        jax.config.update("jax_enable_x64", True)
    first = clqr_jax.pack_problem(_problem(dtype), dtype=dtype)
    second = first._replace(
        rhs=first.rhs._replace(initial_state=2.0 * first.rhs.initial_state)
    )
    batched = jax.tree.map(lambda *leaves: np.stack(leaves), first, second)
    batched = jax.device_put(batched, device)
    result = jax.jit(jax.vmap(clqr_jax.solve))(batched)
    atol = 3e-5 if dtype == np.dtype(np.float32) else 1e-9
    assert result.states.shape == (2, 2, 1)
    np.testing.assert_allclose(result.states[:, 0, 0], [1.0, 2.0], atol=atol)
    np.testing.assert_allclose(result.controls[:, 0, 0], [-1.0, -2.0], atol=atol)


def test_cuda_zero_horizon():
    clqr_jax = _load_modules()
    device = _cuda_device()
    dtype = np.dtype(clqr_jax.scalar_dtype)
    if dtype == np.dtype(np.float64):
        jax.config.update("jax_enable_x64", True)
    problem = {
        "initial_state": np.array([0.5, -0.25], dtype=dtype),
        "stages": [],
        "terminal_Q": np.eye(2, dtype=dtype),
        "terminal_q": np.array([0.1, -0.2], dtype=dtype),
    }
    packed = jax.device_put(
        clqr_jax.pack_problem(problem, dtype=dtype), device
    )
    result = jax.jit(clqr_jax.solve)(packed)
    assert int(result.status) == clqr_jax.SolveStatus.OPTIMAL
    assert result.controls.shape == (0, 0)
    np.testing.assert_allclose(result.states[0], problem["initial_state"])


if __name__ == "__main__":
    test_cuda_eager_jit_and_new_rhs()
    test_cuda_matches_cpu_for_heterogeneous_dimensions()
    test_cuda_sequential_vmap()
    test_cuda_zero_horizon()
