"""JAX interface for the constrained LQR elimination solver.

Problems are padded once into fixed-shape arrays. Active dimensions are stored
separately, so the same compiled call supports heterogeneous stage dimensions.
The factor matrices and right-hand sides are separate pytrees in preparation
for a future numerical factor/solve API; the current ``solve`` call performs
both operations. On CUDA, scalar inputs and outputs remain device-resident;
only the compact active-dimension vector is read by the host to prepare and
reuse the native solver workspace.

Automatic differentiation is not implemented yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import _clqr_jax_cpu


class SolveStatus(IntEnum):
    OPTIMAL = 0
    INFEASIBLE = 1
    INVALID_INPUT = 2
    NUMERICAL_FAILURE = 3


class FactorizationInputs(NamedTuple):
    """Padded factor matrices; ``Q[-1]`` is the terminal cost matrix."""

    dimensions: Any
    A: Any
    B: Any
    Q: Any
    R: Any
    M: Any
    C: Any
    D: Any
    E: Any
    terminal_E: Any


class SolveInputs(NamedTuple):
    """Replaceable padded vectors; ``q[-1]`` is the terminal cost vector."""

    c: Any
    q: Any
    r: Any
    d: Any
    e: Any
    terminal_e: Any
    initial_state: Any


class PackedProblem(NamedTuple):
    """A fixed-shape, JAX-compatible constrained LQR problem."""

    factors: FactorizationInputs
    rhs: SolveInputs


class Solution(NamedTuple):
    """Padded CLQR solution returned as JAX arrays."""

    diagnostics: Any
    objective: Any
    states: Any
    controls: Any
    initial_multiplier: Any
    dynamics_multipliers: Any
    mixed_multipliers: Any
    state_multipliers: Any
    terminal_state_multiplier: Any

    @property
    def status(self) -> Any:
        return self.diagnostics[0]

    @property
    def newton_kkt_singular(self) -> Any:
        """CPU Newton-KKT diagnostic; CUDA/Metal currently leave this unset."""
        return self.diagnostics[1] != 0

    @property
    def newton_kkt_wrong_inertia(self) -> Any:
        """CPU Newton-KKT diagnostic; CUDA/Metal currently leave this unset."""
        return self.diagnostics[2] != 0


_SCALAR_DTYPE = np.dtype(_clqr_jax_cpu.scalar_dtype)
_TARGET_NAME = f"clqr_solve_f{_SCALAR_DTYPE.itemsize * 8}"
_METAL_TARGET_NAME = f"clqr_metal_solve_f{_SCALAR_DTYPE.itemsize * 8}"
scalar_dtype = _SCALAR_DTYPE

for _name, _capsule in _clqr_jax_cpu.ffi_registrations().items():
    jax.ffi.register_ffi_target(_name, _capsule, platform="cpu")

try:
    import _clqr_cuda
except ModuleNotFoundError as error:
    if error.name != "_clqr_cuda":
        raise
    cuda_registered = False
else:
    if np.dtype(_clqr_cuda.scalar_dtype) != _SCALAR_DTYPE:
        raise ImportError(
            "_clqr_jax_cpu and _clqr_cuda use different scalar precisions"
        )
    for _name, _capsule in _clqr_cuda.ffi_registrations().items():
        jax.ffi.register_ffi_target(_name, _capsule, platform="CUDA")
    cuda_registered = True

try:
    import _clqr_metal
except ModuleNotFoundError as error:
    if error.name != "_clqr_metal":
        raise
    metal_registered = False
    metal_unsupported_reason = "the optional _clqr_metal extension is absent"
else:
    if np.dtype(_clqr_metal.scalar_dtype) != _SCALAR_DTYPE:
        raise ImportError(
            "_clqr_jax_cpu and _clqr_metal use different scalar precisions"
        )
    metal_registered = bool(_clqr_metal.supported)
    metal_unsupported_reason = _clqr_metal.unsupported_reason
    if metal_registered:
        for _name, _capsule in _clqr_metal.ffi_registrations().items():
            jax.ffi.register_ffi_target(_name, _capsule, platform="cpu")


def _array(value: Any, name: str, dtype: np.dtype, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=dtype, order="C")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; got shape {array.shape}")
    return array


def _optional_matrix(
    mapping: Mapping[str, Any],
    key: str,
    shape: tuple[int, int],
    dtype: np.dtype,
) -> np.ndarray:
    if key not in mapping:
        return np.zeros(shape, dtype=dtype)
    array = _array(mapping[key], key, dtype, 2)
    if array.shape[1] != shape[1]:
        raise ValueError(f"{key} must have {shape[1]} columns; got shape {array.shape}")
    return array


def _optional_vector(
    mapping: Mapping[str, Any],
    key: str,
    size: int,
    dtype: np.dtype,
) -> np.ndarray:
    if key not in mapping:
        return np.zeros((size,), dtype=dtype)
    array = _array(mapping[key], key, dtype, 1)
    if array.shape != (size,):
        raise ValueError(f"{key} must have shape {(size,)}; got {array.shape}")
    return array


def _expect_shape(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")


def pack_problem(
    problem: Mapping[str, Any], *, dtype: Any | None = None
) -> PackedProblem:
    """Validate and pad the canonical dictionary schema."""

    if not isinstance(problem, Mapping):
        raise TypeError("problem must be a mapping")
    scalar_dtype = _SCALAR_DTYPE if dtype is None else np.dtype(dtype)
    if scalar_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")

    stages_value = problem.get("stages")
    if not isinstance(stages_value, Sequence):
        raise TypeError("problem['stages'] must be a sequence")
    stages = list(stages_value)

    try:
        Q_values = list(problem["Q"])
        q_values = list(problem["q"])
    except KeyError as error:
        raise KeyError(f"missing required key {error.args[0]!r}") from error
    except TypeError as error:
        raise TypeError("problem['Q'] and problem['q'] must be sequences") from error
    if len(Q_values) != len(stages) + 1:
        raise ValueError("problem['Q'] must contain N + 1 entries")
    if len(q_values) != len(stages) + 1:
        raise ValueError("problem['q'] must contain N + 1 entries")

    terminal_Q = _array(Q_values[-1], "Q[-1]", scalar_dtype, 2)
    if terminal_Q.shape[0] != terminal_Q.shape[1]:
        raise ValueError("Q[-1] must be square")
    terminal_n = terminal_Q.shape[0]
    terminal_q = _array(q_values[-1], "q[-1]", scalar_dtype, 1)
    _expect_shape(terminal_q, (terminal_n,), "q[-1]")
    terminal_E = _optional_matrix(problem, "terminal_E", (0, terminal_n), scalar_dtype)
    terminal_e = _optional_vector(
        problem, "terminal_e", terminal_E.shape[0], scalar_dtype
    )
    initial_state = _array(
        problem.get("initial_state"), "initial_state", scalar_dtype, 1
    )

    parsed: list[dict[str, np.ndarray]] = []
    state_dimensions: list[int] = []
    control_dimensions: list[int] = []
    mixed_dimensions: list[int] = []
    state_constraint_dimensions: list[int] = []
    for index, stage_value in enumerate(stages):
        if not isinstance(stage_value, Mapping):
            raise TypeError(f"stage {index} must be a mapping")
        stage = stage_value
        prefix = f"stages[{index}]"
        A = _array(stage.get("A"), f"{prefix}.A", scalar_dtype, 2)
        B = _array(stage.get("B"), f"{prefix}.B", scalar_dtype, 2)
        if A.shape[0] != B.shape[0]:
            raise ValueError(f"{prefix}.A and B must have the same row count")
        next_n, n = A.shape
        m = B.shape[1]
        c = _array(stage.get("c"), f"{prefix}.c", scalar_dtype, 1)
        Q = _array(Q_values[index], f"Q[{index}]", scalar_dtype, 2)
        R = _array(stage.get("R"), f"{prefix}.R", scalar_dtype, 2)
        M = _array(stage.get("M"), f"{prefix}.M", scalar_dtype, 2)
        q = _array(q_values[index], f"q[{index}]", scalar_dtype, 1)
        r = _array(stage.get("r"), f"{prefix}.r", scalar_dtype, 1)
        _expect_shape(c, (next_n,), f"{prefix}.c")
        _expect_shape(Q, (n, n), f"Q[{index}]")
        _expect_shape(R, (m, m), f"{prefix}.R")
        _expect_shape(M, (n, m), f"{prefix}.M")
        _expect_shape(q, (n,), f"q[{index}]")
        _expect_shape(r, (m,), f"{prefix}.r")

        C = _optional_matrix(stage, "C", (0, n), scalar_dtype)
        D = _optional_matrix(stage, "D", (C.shape[0], m), scalar_dtype)
        if D.shape != (C.shape[0], m):
            raise ValueError(
                f"{prefix}.D must have shape {(C.shape[0], m)}; got {D.shape}"
            )
        d = _optional_vector(stage, "d", C.shape[0], scalar_dtype)
        E = _optional_matrix(stage, "E", (0, n), scalar_dtype)
        e = _optional_vector(stage, "e", E.shape[0], scalar_dtype)

        if index == 0:
            state_dimensions.append(n)
        elif state_dimensions[-1] != n:
            raise ValueError(
                f"{prefix}.A has {n} columns but the preceding state "
                f"dimension is {state_dimensions[-1]}"
            )
        state_dimensions.append(next_n)
        control_dimensions.append(m)
        mixed_dimensions.append(C.shape[0])
        state_constraint_dimensions.append(E.shape[0])
        parsed.append(
            dict(
                A=A,
                B=B,
                c=c,
                Q=Q,
                R=R,
                M=M,
                q=q,
                r=r,
                C=C,
                D=D,
                d=d,
                E=E,
                e=e,
            )
        )

    if stages:
        if state_dimensions[-1] != terminal_n:
            raise ValueError(
                "Q[-1] dimension must equal the final stage state dimension"
            )
    else:
        state_dimensions = [terminal_n]
    if initial_state.shape != (state_dimensions[0],):
        raise ValueError(
            "initial_state must have shape "
            f"{(state_dimensions[0],)}; got {initial_state.shape}"
        )

    N = len(stages)
    nx = max(state_dimensions, default=0)
    nu = max(control_dimensions, default=0)
    nc = max(mixed_dimensions, default=0)
    ne = max(state_constraint_dimensions, default=0)
    nt = terminal_E.shape[0]

    arrays = {
        "A": np.zeros((N, nx, nx), dtype=scalar_dtype),
        "B": np.zeros((N, nx, nu), dtype=scalar_dtype),
        "c": np.zeros((N, nx), dtype=scalar_dtype),
        "Q": np.zeros((N + 1, nx, nx), dtype=scalar_dtype),
        "R": np.zeros((N, nu, nu), dtype=scalar_dtype),
        "M": np.zeros((N, nx, nu), dtype=scalar_dtype),
        "q": np.zeros((N + 1, nx), dtype=scalar_dtype),
        "r": np.zeros((N, nu), dtype=scalar_dtype),
        "C": np.zeros((N, nc, nx), dtype=scalar_dtype),
        "D": np.zeros((N, nc, nu), dtype=scalar_dtype),
        "d": np.zeros((N, nc), dtype=scalar_dtype),
        "E": np.zeros((N, ne, nx), dtype=scalar_dtype),
        "e": np.zeros((N, ne), dtype=scalar_dtype),
    }
    for index, stage in enumerate(parsed):
        n = state_dimensions[index]
        next_n = state_dimensions[index + 1]
        m = control_dimensions[index]
        mixed = mixed_dimensions[index]
        state_constraints = state_constraint_dimensions[index]
        arrays["A"][index, :next_n, :n] = stage["A"]
        arrays["B"][index, :next_n, :m] = stage["B"]
        arrays["c"][index, :next_n] = stage["c"]
        arrays["Q"][index, :n, :n] = stage["Q"]
        arrays["R"][index, :m, :m] = stage["R"]
        arrays["M"][index, :n, :m] = stage["M"]
        arrays["q"][index, :n] = stage["q"]
        arrays["r"][index, :m] = stage["r"]
        arrays["C"][index, :mixed, :n] = stage["C"]
        arrays["D"][index, :mixed, :m] = stage["D"]
        arrays["d"][index, :mixed] = stage["d"]
        arrays["E"][index, :state_constraints, :n] = stage["E"]
        arrays["e"][index, :state_constraints] = stage["e"]

    arrays["Q"][N, :terminal_n, :terminal_n] = terminal_Q
    arrays["q"][N, :terminal_n] = terminal_q
    padded_terminal_E = np.zeros((nt, nx), dtype=scalar_dtype)
    padded_terminal_E[:, :terminal_n] = terminal_E
    padded_initial_state = np.zeros((nx,), dtype=scalar_dtype)
    padded_initial_state[: state_dimensions[0]] = initial_state
    dimensions = np.asarray(
        state_dimensions
        + control_dimensions
        + mixed_dimensions
        + state_constraint_dimensions
        + [nt],
        dtype=np.int32,
    )

    return PackedProblem(
        factors=FactorizationInputs(
            dimensions=dimensions,
            A=arrays["A"],
            B=arrays["B"],
            Q=arrays["Q"],
            R=arrays["R"],
            M=arrays["M"],
            C=arrays["C"],
            D=arrays["D"],
            E=arrays["E"],
            terminal_E=padded_terminal_E,
        ),
        rhs=SolveInputs(
            c=arrays["c"],
            q=arrays["q"],
            r=arrays["r"],
            d=arrays["d"],
            e=arrays["e"],
            terminal_e=terminal_e,
            initial_state=padded_initial_state,
        ),
    )


def _require_mapping_dtype(value: Any, expected: np.dtype, path: str) -> None:
    """Reject explicitly typed mapping leaves that Metal would silently cast."""

    if isinstance(value, Mapping):
        for name, child in value.items():
            _require_mapping_dtype(child, expected, f"{path}.{name}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
        for index, child in enumerate(value):
            _require_mapping_dtype(child, expected, f"{path}[{index}]")
        return
    dtype = getattr(value, "dtype", None)
    if dtype is not None and np.dtype(dtype) != expected:
        raise ValueError(
            f"{path} uses {np.dtype(dtype)}; backend='metal' requires "
            f"explicit {expected} inputs"
        )


def solve(
    problem: PackedProblem | Mapping[str, Any],
    *,
    tolerance: float | None = None,
    backend: str | None = None,
) -> Solution:
    """Solve a packed CLQR problem using the backend of its JAX arrays.

    CPU arrays call the sequential C++ implementation. CUDA arrays call the
    optional CUDA extension. ``backend="metal"`` explicitly dispatches the
    native FP32 Metal implementation from the CPU JAX platform.
    """

    if isinstance(problem, Mapping):
        if backend == "metal":
            _require_mapping_dtype(problem, _SCALAR_DTYPE, "problem")
        packed = pack_problem(problem)
    else:
        packed = problem
    if not isinstance(packed, PackedProblem):
        raise TypeError("problem must be a PackedProblem or problem mapping")
    factors = jax.tree.map(jnp.asarray, packed.factors)
    rhs = jax.tree.map(jnp.asarray, packed.rhs)
    dtype = np.dtype(factors.A.dtype)
    if dtype != _SCALAR_DTYPE:
        raise ValueError(
            f"the installed CLQR extension uses {_SCALAR_DTYPE}, got {dtype}"
        )
    if factors.dimensions.dtype != jnp.int32:
        raise ValueError("packed dimensions must use int32")

    N, nx, _ = factors.A.shape
    nu = factors.B.shape[2]
    nc = factors.C.shape[1]
    ne = factors.E.shape[1]
    nt = factors.terminal_E.shape[0]
    result_shapes = (
        jax.ShapeDtypeStruct((3,), jnp.int32),
        jax.ShapeDtypeStruct((1,), dtype),
        jax.ShapeDtypeStruct((N + 1, nx), dtype),
        jax.ShapeDtypeStruct((N, nu), dtype),
        jax.ShapeDtypeStruct((nx,), dtype),
        jax.ShapeDtypeStruct((N, nx), dtype),
        jax.ShapeDtypeStruct((N, nc), dtype),
        jax.ShapeDtypeStruct((N, ne), dtype),
        jax.ShapeDtypeStruct((nt,), dtype),
    )
    scalar = np.float32 if dtype == np.dtype(np.float32) else np.float64
    if tolerance is None:
        tolerance = 1e-5 if dtype == np.dtype(np.float32) else 1e-9
    if backend not in (None, "metal"):
        raise ValueError("backend must be None or 'metal'")
    if backend == "metal" and not metal_registered:
        raise RuntimeError(
            f"the CLQR Metal backend is unavailable: {metal_unsupported_reason}"
        )
    target_name = _METAL_TARGET_NAME if backend == "metal" else _TARGET_NAME
    call = jax.ffi.ffi_call(target_name, result_shapes, vmap_method="sequential")
    output = call(
        factors.dimensions,
        factors.A,
        factors.B,
        rhs.c,
        factors.Q,
        factors.R,
        factors.M,
        rhs.q,
        rhs.r,
        factors.C,
        factors.D,
        rhs.d,
        factors.E,
        rhs.e,
        factors.terminal_E,
        rhs.terminal_e,
        rhs.initial_state,
        tolerance=scalar(tolerance),
    )
    return Solution(*output)


__all__ = [
    "FactorizationInputs",
    "PackedProblem",
    "Solution",
    "SolveStatus",
    "SolveInputs",
    "cuda_registered",
    "metal_registered",
    "metal_unsupported_reason",
    "pack_problem",
    "scalar_dtype",
    "solve",
]
