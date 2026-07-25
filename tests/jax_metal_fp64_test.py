import importlib
import importlib.util
import os
import pathlib
import platform
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


def test_metal_precision_contract():
    root = pathlib.Path(os.environ["TEST_SRCDIR"])
    _load_extension(root, "_clqr_jax_cpu")
    metal = _load_extension(root, "_clqr_metal")
    packages = list(root.rglob("python/clqr/__init__.py"))
    if not packages:
        raise RuntimeError("could not find python/clqr in Bazel runfiles")
    sys.path.insert(0, str(packages[0].parents[1]))
    clqr_jax = importlib.import_module("clqr.jax")
    if np.dtype(clqr_jax.scalar_dtype) == np.dtype(np.float32):
        apple_silicon = platform.machine().lower() in ("arm64", "aarch64")
        assert metal.supported == apple_silicon
        assert clqr_jax.metal_registered == apple_silicon
        assert ("clqr_metal_solve_f32" in metal.ffi_registrations()) == apple_silicon
        if not apple_silicon:
            assert "Apple-silicon" in metal.unsupported_reason
        return

    assert np.dtype(clqr_jax.scalar_dtype) == np.dtype(np.float64)
    assert not metal.supported
    assert not clqr_jax.metal_registered
    assert "does not support float64" in clqr_jax.metal_unsupported_reason
    jax.config.update("jax_enable_x64", True)
    problem = {
        "initial_state": np.array([1.0], dtype=np.float64),
        "stages": [],
        "terminal_Q": np.array([[1.0]], dtype=np.float64),
        "terminal_q": np.array([0.0], dtype=np.float64),
    }
    packed = clqr_jax.pack_problem(problem, dtype=np.float64)
    try:
        clqr_jax.solve(packed, backend="metal")
    except RuntimeError as error:
        assert "does not support float64" in str(error)
    else:
        raise AssertionError("FP64 Metal solve was not rejected")


if __name__ == "__main__":
    test_metal_precision_contract()
