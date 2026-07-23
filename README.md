# Constrained LQR Elimination

CPU and CUDA implementations of equality-constrained finite-horizon
LQR via affine constraint elimination, followed by an unconstrained LQR solve
in reduced coordinates.

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

The benchmark does not install or time the JAX implementation. To run the
additional solution-level JAX cross-validation diagnostic, set
`CLQR_RUN_JAX_CROSS_VALIDATION=1`; `CLQR_JAX_REVISION` can override its pinned
reference revision.

For correctness stress testing rather than timing, use
[`notebooks/kaggle_cuda_stress.ipynb`](notebooks/kaggle_cuda_stress.ipynb).
Its shell driver, [`scripts/notebook_cuda_stress.sh`](scripts/notebook_cuda_stress.sh),
runs the extended fixed-seed and long-horizon corpus in FP64 and FP32 through
the sequential C++ solver and native CUDA, runs the shared standard and
selected extended corpus through CUDA kernel emulation, and applies Compute
Sanitizer memcheck, initcheck, racecheck, and synccheck to both native suites.
The report includes the exact revision, host, compiler, CUDA toolkit, driver,
and GPU descriptions.
The normal CI-sized adversarial suite is `//:adversarial_cpu_test`; opt into the
broader host and emulation suites with:

```sh
bazel test //:adversarial_cpu_extended_test \
  //:cuda_kernel_emulation_extended_test \
  --config=fp64 \
  --test_output=errors
```

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

Sample workspace-API results from `bazel-bin/clqr_benchmark 5` after building
`//:clqr_benchmark -c opt` on arm64 macOS with clang 22.1.1. The benchmark reserves
workspace once per problem and times repeated solves. It also reports `max_us`; local
scheduler spikes can make maxima unrepresentative, so median and p90 are usually
better summary statistics.

| Case | Iterations | Mean | Median | P90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| `N=16 n=4 m=2 p=0` | 1000 | `13.8 us` | `14.2 us` | `16.5 us` | `8.88 us` | `104 us` |
| `N=16 n=4 m=2 p=1` | 1000 | `22.6 us` | `20.3 us` | `23.6 us` | `18.6 us` | `160 us` |
| `N=16 n=4 m=2 p=2` | 1000 | `18.7 us` | `18.1 us` | `19.7 us` | `16.7 us` | `91.1 us` |
| `N=16 n=6 m=3 p=0` | 500 | `24.0 us` | `23.1 us` | `24.3 us` | `22.3 us` | `93.9 us` |
| `N=16 n=6 m=3 p=1` | 500 | `37.5 us` | `35.7 us` | `38.3 us` | `35.1 us` | `112 us` |
| `N=16 n=6 m=3 p=2` | 500 | `37.6 us` | `35.9 us` | `38.1 us` | `34.9 us` | `124 us` |
| `N=32 n=6 m=3 p=0` | 250 | `47.6 us` | `46.2 us` | `50.0 us` | `44.5 us` | `86.6 us` |
| `N=32 n=6 m=3 p=1` | 250 | `77.7 us` | `74.8 us` | `78.6 us` | `71.9 us` | `227 us` |
| `N=32 n=6 m=3 p=2` | 250 | `78.3 us` | `73.9 us` | `88.7 us` | `71.1 us` | `180 us` |
| `N=64 n=6 m=3 p=0` | 100 | `91.5 us` | `90.0 us` | `94.6 us` | `88.9 us` | `116 us` |
| `N=64 n=6 m=3 p=1` | 100 | `159 us` | `151 us` | `178 us` | `143 us` | `295 us` |
| `N=64 n=6 m=3 p=2` | 100 | `151 us` | `144 us` | `158 us` | `140 us` | `406 us` |
| `N=128 n=8 m=4 p=0` | 50 | `400 us` | `382 us` | `482 us` | `363 us` | `585 us` |
| `N=128 n=8 m=4 p=1` | 50 | `521 us` | `513 us` | `566 us` | `497 us` | `643 us` |
| `N=128 n=8 m=4 p=2` | 50 | `524 us` | `511 us` | `574 us` | `481 us` | `752 us` |

All sample cases reported `singular_count=0` and `wrong_inertia_count=0`.

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

The lightweight package wrapper in `python/clqr/__init__.py` re-exports the same `solve`
function once `_clqr` is on `PYTHONPATH`.

The Python boundary accepts and returns NumPy-compatible `float64` arrays; an
FP32 extension converts them to and from `clqr::Scalar` internally. The result
dict contains `status`, `message`,
`newton_kkt_singular`, `newton_kkt_wrong_inertia`, `newton_kkt_diagnostic`, `objective`,
`states`, `controls`, `initial_multiplier`, `dynamics_multipliers`, `mixed_multipliers`,
`state_multipliers`, and `terminal_state_multiplier`. The multiplier signs correspond to the
constraints exactly as written above. The Newton-KKT diagnostic fields are reported separately
from `status`; when the reduced solve can proceed, a candidate solution is still returned.
