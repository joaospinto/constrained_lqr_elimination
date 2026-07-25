# Constrained LQR Elimination

CPU and CUDA implementations of equality-constrained finite-horizon
LQR via affine constraint elimination, followed by an unconstrained LQR solve
in reduced coordinates, plus a native Metal implementation exposed through
the JAX binding.

The implemented problem form is:

```text
min sum_i 1/2 x_i' Q_i x_i + 1/2 u_i' R_i u_i + x_i' M_i u_i + q_i' x_i + r_i' u_i
    + 1/2 x_N' Q_N x_N + q_N' x_N

s.t. x_0 = initial_state
     x_{i+1} = A_i x_i + B_i u_i + c_i
     C_i x_i + D_i u_i + d_i = 0
     E_i x_i + e_i = 0
     E_N x_N + e_N = 0
```

The constrained solver performs a single right-to-left elimination sweep. It first
parameterizes any terminal state equality as `x_N = T_N z_N + t_N`, then visits stages
from `N - 1` down to `0`. At each stage it:

1. applies the already-carried parameterization of `x_{i+1}` to the dynamics, turning
   the eliminated next-state rows into additional mixed constraints on `(x_i, u_i)`;
2. eliminates the stage mixed constraints by parameterizing the control as
   `u_i = Y_i x_i + Z_i v_i + y_i`, appending any residual state-only constraints to
   node `i`;
3. parameterizes the resulting node-`i` state equality as `x_i = T_i z_i + t_i` and
   carries that basis left to stage `i - 1`.

After the sweep, all explicit equality constraints have been folded into affine state and
control maps, so the reduced problem is an unconstrained LQR solved by a standard Riccati
backward/forward pass. The final state and control trajectories are mapped back through the
stored affine maps, and the original multipliers are recovered by applying the transposes of
the recorded elimination operations to the reduced stationarity covectors. Redundant equality
rows mark the Newton-KKT system as singular; a reduced control Hessian with the wrong inertia
is reported separately when the candidate solve can still proceed.

The C++ API uses `clqr::Scalar` throughout. FP64 is the default; FP32 is a
separate, pure-precision build selected at compile time. Because the scalar
type is part of the C++ ABI, libraries and clients must use the same precision
configuration.

## CUDA backend

The CUDA backend implements the parallel form of the same reduction. For a
horizon of `N` stages, it:

1. contracts and expands a balanced tree of affine feasibility relations to
   compute every state parameterization `x_i = T_i z_i + t_i`;
2. eliminates each stage's controls independently in those coordinates;
3. solves the reduced unconstrained LQR with a balanced conditional-value
   scan and reconstructs the primal trajectory with an affine-map scan; and
4. recovers the original equality multipliers with a balanced dual-relation
   scan, reusing the parameterizations produced by the primal solve.

Every horizon-dependent device buffer has `O(N)` storage. Problem data,
trajectories, parameterizations, reduced stages, feedback records, multiplier
records, and scan coefficients are packed from the runtime dimensions of the
individual stages. Kernels loop over and factor only active state, control,
constraint, and reduced dimensions; no padded dense algebra is performed and
there are no build-time dimension capacities.

Every horizon-dependent device dependency is a balanced-tree reduction or
expansion. Feasibility propagation, the conditional-value solve, primal
reconstruction, and multiplier recovery each use at most a constant multiple
of `ceil(log2(N + 1))` kernel rounds plus a constant number of independent
per-stage kernels. No device kernel iterates over the horizon. Consequently,
for fixed local dimensions, the device algorithm has `O(log N)` parallel time,
`O(N)` work, and `O(N)` storage. End-to-end API wall time additionally includes
serial host packing, compact-storage planning, transfers, and result
construction; these costs are reported separately from pure kernel time.

The conditional-value scan requires every reduced stage control Hessian to be
positive definite, as in the standard convex LQR setting. A problem violating
this regularity condition is reported as a numerical failure; the CUDA backend
does not contain a horizon-sequential Riccati compatibility path.

The public CUDA API is in `clqr/cuda.h`. The ordinary owning interface validates
the problem and copies the result into caller-owned vectors:

```cpp
clqr::cuda::Workspace workspace;
workspace.Reserve(problem);
clqr::cuda::Solution result;
clqr::cuda::Solve(problem, workspace, result);
```

For repeated solves with unchanged dimensions and matrix/vector shapes, reserve
once and use the prepared, workspace-backed interface:

```cpp
clqr::cuda::Workspace workspace;
workspace.Reserve(problem);

// Numerical entries may change between calls, but the structure must not.
clqr::cuda::SolutionView result =
    clqr::cuda::SolvePreparedView(problem, workspace);
```

`SolutionView` avoids per-solve result allocation and copying. It remains valid
until the workspace is reused, reserved for another structure, moved, or
destroyed. Call `Materialize` or `SolvePrepared` when an owning result is
required. `SolveView` provides the same non-owning representation while
retaining structural validation on every call.

A CPU-only build provides the same symbols through a stub library:
`clqr::cuda::Available()` returns false and `Solve` reports an invalid-input
result without loading the CUDA runtime.

Build and test the native backend with Bazel. For example, target a P100
(`sm_60`) in FP64:

```sh
bazel test //:cuda_solver_test \
  --config=fp64 \
  --config=cuda-benchmark \
  --cuda_archs=sm_60 \
  --test_output=errors
```

`--config=fp32` builds both the CPU reference and CUDA backend entirely in
FP32. Local dense workspaces are sized at runtime. A solve is accepted whenever
its packed allocations and the largest active per-block factorization fit the
selected device's portable, non-opt-in per-block shared-memory limit. Staying
within that default limit avoids architecture-specific launch attributes and
excessive per-block storage that would sharply reduce occupancy. If a
factorization does not fit, `Solve` returns a deterministic diagnostic
describing the required resource; there is no capacity flag to rebuild.

The CUDA benchmark uses `SolvePreparedView`, reuses reserved storage, and
reports both end-to-end wall time and pure kernel time. Wall time includes host
packing, all transfers, synchronization, kernels, and construction of the
workspace-backed view. `cuda_kernel_ms` sums CUDA event intervals containing
only kernels; `upload_ms` and `download_ms` cover the bulk packed inputs and
outputs. The remaining columns separate input packing, compact-layout updates,
device objective reduction, API setup, and synchronization or phase-control
overhead. Multiplier consistency rejection is disabled only while timing so
the final KKT residual can be reported rather than turning a numerical
threshold crossing into a missing row.

For a reproducible native-CUDA validation and benchmark run, open
[`notebooks/kaggle_cuda_benchmark.ipynb`](notebooks/kaggle_cuda_benchmark.ipynb)
in a fresh Kaggle GPU notebook. It records the machine specification, builds
the CPU and CUDA implementations in the same precision, runs the CPU,
kernel-emulation, native-CUDA, and Compute Sanitizer tests, and benchmarks all
configured horizons. It bootstraps the Bazel version pinned in `.bazelversion`
when needed. The canonical shell driver is
[`scripts/notebook_cuda.sh`](scripts/notebook_cuda.sh); the former
`scripts/colab_t4.sh` name remains as a compatibility wrapper.

After validating a candidate, compare it against `main` with alternating
benchmark order:

```sh
CLQR_CANDIDATE_REVISION=feature/my-candidate \
CLQR_CUDA_ARCH=60 bash scripts/compare_cuda_revisions.sh
```

The comparison defaults to three rounds of eleven warmed FP64 solves and
reports CPU, wall, kernel, phase, packing/transfer, and CPU/CUDA KKT-residual
comparisons. The default CSV includes absolute candidate packing/transfer time
and its share of the wall--kernel gap. Temporary build trees are removed
automatically; set `CLQR_KEEP_COMPARE_OUTPUT=1` to retain them.

Revision-specific build flags can be supplied through
`CLQR_BASE_EXTRA_BAZEL_ARGS` and `CLQR_CANDIDATE_EXTRA_BAZEL_ARGS`, without
shell evaluation. The emitted report records both revisions and both
revision-specific build argument strings.

The benchmark does not install or time the JAX implementation. To run the
additional solution-level JAX cross-validation diagnostic, set
`CLQR_RUN_JAX_CROSS_VALIDATION=1`; `CLQR_JAX_REVISION` can override its pinned
reference revision.

For correctness stress testing rather than timing, use
[`notebooks/kaggle_cuda_stress.ipynb`](notebooks/kaggle_cuda_stress.ipynb).
Its shell driver, [`scripts/notebook_cuda_stress.sh`](scripts/notebook_cuda_stress.sh),
runs the extended fixed-seed and long-horizon corpus in FP64 and the standard
representative corpus in FP32 through the sequential C++ solver, kernel
emulation, and native CUDA. It applies Compute Sanitizer memcheck, initcheck,
racecheck, and synccheck to both native suites. Set
`CLQR_RUN_FP32_EXTENDED_STRESS=1` to add the explicitly non-gating,
pathological long-horizon FP32 corpus. The report includes the exact revision,
host, compiler, CUDA toolkit, driver, and GPU descriptions.
The normal CI-sized adversarial suite is `//:adversarial_cpu_test`; opt into the
broader host and emulation suites with:

```sh
bazel test //:adversarial_cpu_extended_test \
  //:cuda_kernel_emulation_extended_test \
  --config=fp64 \
  --test_output=errors
```

## CPU backend and native APIs

Build and test either precision:

```sh
bazel test //... --config=fp64
bazel test //... --config=fp32
```

Run the C++ timing benchmark:

```sh
bazel run -c opt //:clqr_benchmark
```

Pass an integer scale factor to increase the per-case iteration counts:

```sh
bazel run -c opt //:clqr_benchmark -- 10
```

Sample workspace-API results from `bazel-bin/clqr_benchmark 5` at `9e85b6f`,
after building `//:clqr_benchmark -c opt` on arm64 macOS with Apple clang
21.0.0. The benchmark reserves workspace once per problem and times repeated
solves. It also reports `max_us`; local scheduler spikes can make maxima
unrepresentative, so median and p90 are usually better summary statistics.

| Case | Iterations | Mean | Median | P90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| `N=16 n=4 m=2 p=0` | 1000 | `3.551 us` | `3.542 us` | `3.583 us` | `3.416 us` | `5.917 us` |
| `N=16 n=4 m=2 p=1` | 1000 | `9.768 us` | `9.750 us` | `9.833 us` | `9.625 us` | `12.334 us` |
| `N=16 n=4 m=2 p=2` | 1000 | `9.980 us` | `9.750 us` | `10.625 us` | `9.167 us` | `34.625 us` |
| `N=16 n=6 m=3 p=0` | 500 | `7.714 us` | `7.542 us` | `8.042 us` | `7.292 us` | `30.292 us` |
| `N=16 n=6 m=3 p=1` | 500 | `16.033 us` | `15.542 us` | `17.000 us` | `15.000 us` | `37.875 us` |
| `N=16 n=6 m=3 p=2` | 500 | `16.004 us` | `15.708 us` | `16.583 us` | `15.083 us` | `37.875 us` |
| `N=32 n=6 m=3 p=0` | 250 | `15.429 us` | `15.083 us` | `16.000 us` | `14.542 us` | `37.084 us` |
| `N=32 n=6 m=3 p=1` | 250 | `33.072 us` | `32.416 us` | `35.125 us` | `30.875 us` | `53.167 us` |
| `N=32 n=6 m=3 p=2` | 250 | `32.632 us` | `31.958 us` | `33.833 us` | `31.041 us` | `55.917 us` |
| `N=64 n=6 m=3 p=0` | 100 | `30.888 us` | `30.000 us` | `32.959 us` | `29.334 us` | `46.625 us` |
| `N=64 n=6 m=3 p=1` | 100 | `64.560 us` | `64.000 us` | `66.042 us` | `61.958 us` | `92.333 us` |
| `N=64 n=6 m=3 p=2` | 100 | `65.601 us` | `65.083 us` | `67.333 us` | `62.458 us` | `88.250 us` |
| `N=128 n=8 m=4 p=0` | 50 | `112.403 us` | `111.000 us` | `113.042 us` | `109.250 us` | `133.416 us` |
| `N=128 n=8 m=4 p=1` | 50 | `202.968 us` | `199.125 us` | `219.167 us` | `196.667 us` | `222.125 us` |
| `N=128 n=8 m=4 p=2` | 50 | `195.794 us` | `192.958 us` | `212.541 us` | `190.250 us` | `219.500 us` |

All sample cases reported `singular_count=0` and `wrong_inertia_count=0`.

For revision-to-revision evidence on the constrained CPU path, including
adversarial row scaling, redundant rows, primal/KKT residuals, and alternating
benchmark order, run:

```sh
CLQR_BASE_REVISION=<baseline> \
CLQR_CANDIDATE_REVISION=<candidate> \
scripts/compare_cpu_constraint_revisions.sh
```

The focused comparison defaults to FP64 and FP32, 101 timed solves per process,
and seven alternating-order rounds.

C++ users include `clqr/clqr.h` and call `clqr::Solve` with a workspace:

```cpp
clqr::Workspace workspace;
workspace.Reserve(problem);  // one allocation owned by workspace
clqr::SolutionView result = clqr::Solve(problem, workspace);
```

For fixed-size uniform unconstrained problems, the required byte count is `constexpr`:

```cpp
constexpr std::size_t kBytes = clqr::Workspace::RequiredBytesUniform(64, 6, 3);
alignas(std::max_align_t) std::array<unsigned char, kBytes> memory{};
clqr::Workspace workspace(memory.data(), memory.size());
clqr::SolutionView result = clqr::Solve(problem, workspace);  // zero heap allocations
```

There is also a constexpr size helper for uniform constrained workspace solves:

```cpp
constexpr std::size_t kBytes =
    clqr::Workspace::RequiredBytesUniformConstrained(16, 4, 2, 1);
```

The workspace API covers constrained and unconstrained problems. Unconstrained problems use
the raw Riccati path directly. Constrained problems activate the workspace arena and run the
constraint-elimination algorithm, including the reduced Riccati solve and multiplier recovery.

For repeated unconstrained solves with fixed matrices and new right-hand
sides, factor the matrix-dependent Riccati recursion once:

```cpp
clqr::Factorization factors = clqr::Factor(problem);
clqr::SolveRhs rhs = clqr::ExtractRhs(problem);
clqr::Workspace solve_workspace;
solve_workspace.Reserve(factors);

clqr::SolutionView first = clqr::Solve(factors, rhs, solve_workspace);
rhs.initial_state = next_initial_state;
rhs.stages[0].q = next_state_gradient;
clqr::SolutionView second = clqr::Solve(factors, rhs, solve_workspace);
```

`Factor` owns the fixed `A`, `B`, `Q`, `R`, `M`, and terminal `Q` data.
Each factored `Solve` accepts new `c`, `q`, `r`, terminal `q`, and initial
state values with unchanged dimensions, and performs no heap allocation when
given a reserved workspace. The first API slice deliberately rejects equality
constraints; reusable CUDA/JAX factorization and constrained factorization
remain follow-up work.

As with the ordinary workspace API, every buffer and string referenced by the
returned `SolutionView` is workspace-backed and remains valid only until that
workspace is reused or destroyed. Copy any values that must survive the next
solve.

### Native NumPy binding

The native C++ `Problem`, `SolveRhs`, and NumPy dictionary APIs intentionally
keep `terminal_Q` and `terminal_q` as separate fields. This schema mirrors the
C++ types and is distinct from the packed JAX API below.

The Python extension is built by the Bazel target `//:_clqr`; the shared-object output is
addressable as `//:_clqr.so`. It exposes the `_clqr` module directly:

```python
import _clqr

result = _clqr.solve({
    "initial_state": ...,
    "stages": [
        {
            "A": ..., "B": ..., "c": ...,
            "Q": ..., "R": ..., "M": ..., "q": ..., "r": ...,
            "C": ..., "D": ..., "d": ...,  # optional
            "E": ..., "e": ...,            # optional
        },
    ],
    "terminal_Q": ...,
    "terminal_q": ...,
    "terminal_E": ...,  # optional
    "terminal_e": ...,  # optional
})
```

For repeated unconstrained solves, the Python extension exposes the same
matrix/RHS split as C++:

```python
factors = _clqr.factor(problem)
solve_rhs = _clqr.rhs(problem)

first = factors.solve(solve_rhs)
solve_rhs.initial_state = next_initial_state
solve_rhs.stage(0).c = next_dynamics_offset
solve_rhs.stage(0).q = next_state_gradient
solve_rhs.stage(0).r = next_control_gradient
solve_rhs.terminal_q = next_terminal_gradient
second = factors.solve(solve_rhs)
```

`Factorization` owns the matrices and internally reuses its native solve
workspace. Assigning an RHS property copies that array into owned native
storage; result arrays are newly owned NumPy arrays. This factorized Python
path currently rejects equality-constrained problems.

The lightweight package wrapper in `python/clqr/__init__.py` re-exports the same `solve`
function, as well as `factor` and `rhs`, once `_clqr` is on `PYTHONPATH`.

The Python boundary accepts and returns NumPy-compatible `float64` arrays; an
FP32 extension converts them to and from `clqr::Scalar` internally. The result
dict contains `status`, `message`,
`newton_kkt_singular`, `newton_kkt_wrong_inertia`, `newton_kkt_diagnostic`, `objective`,
`states`, `controls`, `initial_multiplier`, `dynamics_multipliers`, `mixed_multipliers`,
`state_multipliers`, and `terminal_state_multiplier`. The multiplier signs correspond to the
constraints exactly as written above. The Newton-KKT diagnostic fields are reported separately
from `status`; when the reduced solve can proceed, a candidate solution is still returned.

## JAX bindings

`python/clqr/jax.py` exposes the solver through JAX's typed FFI:

```bash
bazel build //:_clqr //:_clqr_jax_cpu //:_clqr_cuda --config=cuda
PYTHONPATH="$PWD/python:$PWD/bazel-bin" python your_program.py
```

```python
import jax

from clqr.jax import FactorizationInputs, PackedProblem, SolveInputs, solve

# Canonical JAX storage: Q and q include the terminal entry at index N.
packed = PackedProblem(
    factors=FactorizationInputs(
        dimensions=dimensions,
        A=A,
        B=B,
        Q=Q,  # shape (N + 1, nx, nx)
        R=R,
        M=M,
        C=C,
        D=D,
        E=E,
        terminal_E=terminal_E,
    ),
    rhs=SolveInputs(
        c=c,
        q=q,  # shape (N + 1, nx)
        r=r,
        d=d,
        e=e,
        terminal_e=terminal_e,
        initial_state=initial_state,
    ),
)
result = jax.jit(solve)(jax.device_put(packed, jax.devices()[0]))
```

`PackedProblem.factors` contains the matrices and active dimensions;
`PackedProblem.rhs` contains the vectors and initial state. Arrays are padded
only at the interface, while each backend stage still operates on its active
runtime dimensions. `dimensions` concatenates the `N + 1` state dimensions,
the `N` control dimensions, the `N` mixed-constraint counts, the `N`
state-constraint counts, and the terminal-constraint count. `factors.Q` and
`rhs.q` have `N + 1` entries, with their last entries holding the terminal
cost; the other stage arrays have `N` entries. The optional `pack_problem`
adapter converts a native NumPy dictionary with separate terminal fields into
this representation. The split lets callers replace any RHS vector without
changing the compiled JAX shape. It is not yet a numerical factor/solve split:
the current call refactors after either part changes.

CPU arrays dispatch to the sequential C++ solver. When `//:_clqr_cuda` is
installed, CUDA arrays dispatch to the CUDA solver on the device selected by
JAX. Scalar inputs and outputs remain device-resident, the bridge preserves
JAX stream ordering, and the native CUDA workspace is reused for unchanged
dimensions. Only the compact active-dimension vector is copied to the host for
workspace planning; there is no bulk scalar round trip through host memory.

On Apple silicon, the optional `_clqr_metal` extension provides the same
elimination/scan numerical design as CUDA in a native FP32 Metal
implementation. Select it explicitly because JAX exposes Apple Metal hosts
through its CPU platform:

```python
metal_solve = jax.jit(lambda value: solve(value, backend="metal"))
result = metal_solve(packed)
```

Metal uses one command buffer and one compute encoder per solve, with reusable
shared buffers on unified memory. It is FP32-only: an FP64 build reports the
unsupported precision before dispatch, and `backend="metal"` never silently
falls back to CPU. Mapping inputs with an explicit non-FP32 dtype are rejected
rather than silently cast.

Build and test it on macOS with:

```bash
bazel test //:jax_metal_binding_test \
  //:jax_metal_source_audit_test \
  --config=fp32 \
  --test_output=errors
bazel test //:jax_metal_fp64_test --config=fp64 --test_output=errors
```

Hosted macOS CI builds the native target and source-audits its GPU path because
hosted runners do not guarantee a usable Apple-family Metal device. The full
binding test above runs natively on supported Apple silicon.

The typed FFI lives in optional `_clqr_jax_cpu`, `_clqr_cuda`, and
`_clqr_metal` extensions; the existing `_clqr` Python binding remains
independent of JAX.

The raw FFI call supports eager execution, `jax.jit`, and sequential `jax.vmap`.
Automatic differentiation and sharded-problem rules are not implemented.
Build/test the CPU binding with `//:jax_binding_test`. On a CUDA 12 machine,
`//:jax_cuda_binding_test --config=cuda` exercises GPU dispatch; the notebook
driver includes it when `CLQR_RUN_JAX_FFI_TEST=1`.
