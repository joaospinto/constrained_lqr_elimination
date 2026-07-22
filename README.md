# Constrained LQR Elimination

Sequential CPU implementation of equality-constrained finite-horizon LQR via stagewise affine
constraint elimination, followed by a standard Riccati solve on the reduced unconstrained problem.

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
stored affine maps, and the original multipliers are recovered by a backward pass over the
original KKT stationarity equations. Redundant equality rows mark the Newton-KKT system as
singular; a reduced control Hessian with the wrong inertia is reported separately when the
candidate solve can still proceed.

## Optional CUDA backend

The CMake build has an optional CUDA backend for problems whose state, control, mixed-
constraint, and state-constraint dimensions are each at most 16. The limit bounds kernel
storage; every relation, state parameterization, control parameterization, and Riccati element
also carries its active dimension. Consequently, a reduction from `(n_i, m_i)` to
`(r_i, s_i)` causes the later kernels to perform `r_i`- and `s_i`-dimensional algebra rather
than padded `n_i`- and `m_i`-dimensional algebra.

The GPU algorithm consists of:

1. a reverse associative scan of affine endpoint relations to compute all feasible-state
   parameterizations `x_i = T_i z_i + t_i`;
2. independent per-stage control elimination and construction of the reduced LQR;
3. an associative conditional-value scan for the reduced Riccati solve, followed by an
   affine prefix scan for state rollout; and
4. balanced contraction and expansion of the original stationarity relations to recover one
   globally consistent set of multipliers.

Rank decisions use row equilibration and partial pivoting. Redundant equalities are accepted,
and free multiplier components are set to zero. If a reduced stage cost `R_i` is not positive
definite, the backend retains the same reduced coordinates but uses a GPU-side sequential
Riccati fallback; this avoids making positive definiteness of each `R_i` a new correctness
requirement.

The public numeric type is `clqr::Scalar`. CMake selects it consistently for the C++ API,
CPU solver, and CUDA backend with `CLQR_PRECISION=FP64` (the default) or `FP32`; use separate
build directories because the choice changes the library ABI. FP32 uses precision-appropriate
rank and consistency thresholds. For short FP32 problems with dependent equality rows, it
conservatively uses the sequential CUDA Riccati path and host multiplier recovery; full-rank
instances retain the parallel GPU path.

The CUDA benchmark is precision matched: an FP32 build reports FP32 sequential C++ CPU times
against FP32 CUDA times, while an FP64 build compares the corresponding FP64 implementations.

The default build remains CUDA-free:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Build for a Tesla T4 (compute capability 7.5) with:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DCLQR_ENABLE_CUDA=ON -DCLQR_PRECISION=FP64 \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
build-cuda/clqr_cuda_benchmark
```

For FP32, configure a distinct directory with `-DCLQR_PRECISION=FP32`.

The CUDA test compares states, controls, and objective values against the existing C++ solver,
then checks the complete primal-dual KKT residual. It covers unconstrained, state-only, mixed,
terminal, nonuniform, rank-deficient/redundant, independently rescaled, infeasible,
sequential-fallback, zero-horizon, zero-control, more-constraints-than-controls, and declared
dimension-limit cases.
CUDA-free test builds additionally execute every numerical kernel in a one-thread CPU-emulation
mode, comparing both Riccati paths and the recovered KKT point with the C++ solver.

On Google Colab or Kaggle, select a GPU runtime, clone this repository, and run the complete
hardware report, build, C++/CUDA tests, JAX cross-validation, and benchmark with one command:

```sh
bash scripts/colab_t4.sh
```

The script automatically uses `/content` on Colab and `/kaggle/working` on Kaggle. It builds,
tests, cross-validates, and benchmarks FP64 and FP32 in separate directories. It defaults to
architecture 75 and five benchmark repetitions. Override these with
`CLQR_CUDA_ARCH`, `CLQR_BENCHMARK_REPEATS`, or a space-separated `CLQR_PRECISIONS` selection.
Set `CLQR_JAX_DIR` to reuse an existing checkout of `joaospinto/constrained_lqr_jax`, or
`CLQR_SKIP_JAX=1` to omit only that cross-check. When `compute-sanitizer` is available, the
script also runs its memory, initialization, shared-memory race, and synchronization checkers
for each selected precision. Set `CLQR_SKIP_SANITIZER=1` to omit those passes, or set
`CLQR_SANITIZER_TOOLS` to a space-separated subset.

Build and test:

```sh
bazel test //...
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

Arrays are NumPy-compatible `float64` arrays. The result dict contains `status`, `message`,
`newton_kkt_singular`, `newton_kkt_wrong_inertia`, `newton_kkt_diagnostic`, `objective`,
`states`, `controls`, `initial_multiplier`, `dynamics_multipliers`, `mixed_multipliers`,
`state_multipliers`, and `terminal_state_multiplier`. The multiplier signs correspond to the
constraints exactly as written above. The Newton-KKT diagnostic fields are reported separately
from `status`; when the reduced solve can proceed, a candidate solution is still returned.
