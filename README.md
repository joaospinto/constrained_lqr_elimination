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

All native and Python problem interfaces use one cost schema. For a problem
with `N` stages, `Q` and `q` each contain `N + 1` entries: indices `0` through
`N - 1` are the stage state costs and index `N` is the terminal state cost.
Each `Stage` contains `A`, `B`, `c`, `R`, `M`, `r`, and its optional equality
constraints.

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
FP32. Local dense workspaces are sized at runtime. Workspace preparation queries
the selected GPU's shared-memory capacity and each kernel's static usage, and
opts into larger dynamic shared-memory allocations when needed and supported.
Each launch still requests only its planned scratch size. Kernels whose dense
scratch exceeds the per-block limit use separate per-block slices of reusable
global device workspace instead. Small static shared scalars remain shared.
The fallback preserves the algorithm and numerical choices, but its memory
traffic can cost performance. There is no capacity flag to rebuild; total
device-memory availability still limits the problem size.

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
[`scripts/notebook_cuda.sh`](scripts/notebook_cuda.sh).

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

### Same-machine paper comparisons

`bash scripts/paper_benchmarks.sh /path/to/new-results` builds pinned copies
of the generalized-Riccati, factor-graph, and author-written Laine reference implementations, and
compares them with the sequential solver on identical FP64 instances. Add
`--cuda` on a CUDA machine to include native host-input and device-resident
JAX timings and refresh the original horizon table. The dense comparison
uses native-architecture Release builds for every C++ implementation.
The CUDA run also executes native FP64 regressions and all four Compute
Sanitizer tools on the standard CUDA suite and dense smoke fixtures before
collecting GPU timings.
Requires CMake, a C++ compiler, Git, and at least 3 GiB free disk
space; `CLQR_JOBS` defaults to 4. Set `CLQR_PAPER_SUITE=smoke` for a short run.

The driver streams the current method, case index/count, dimensions, and phase
(setup, warm-up, timing, or validation), plus elapsed time. During long commands,
it prints a heartbeat every 30 seconds; override this with
`CLQR_PROGRESS_INTERVAL_SECONDS`. Progress stays out of the CSV files and is
retained in the notebook's `driver.log`.

Downloads, builds, and measurements can be selected independently with these
environment variables (only `0` and `1` are accepted):

| Variable | Default | Controls |
| --- | --- | --- |
| `CLQR_RUN_EXTERNAL` | `1` | Default for the four external-method switches below |
| `CLQR_RUN_VANROYE` | inherits `CLQR_RUN_EXTERNAL` | Vanroye and its BLASFEO dependency |
| `CLQR_RUN_YANG` | inherits `CLQR_RUN_EXTERNAL` | Yang and its GTSAM dependency |
| `CLQR_RUN_LAINE` | inherits `CLQR_RUN_EXTERNAL` | Author-written Laine–Tomlin |
| `CLQR_RUN_CORRECTED_LAINE` | inherits `CLQR_RUN_EXTERNAL` | Corrected Laine–Tomlin |
| `CLQR_RUN_JAX` | `1` | JAX builds, tests, and timings |
| `CLQR_RUN_TESTS` | `1` | Regression tests |
| `CLQR_RUN_SANITIZERS` | `1` | CUDA Compute Sanitizer runs |
| `CLQR_RUN_ORIGINAL_TABLE` | `1` | Additional original-table timing sweep |

For a native CPU/CUDA-only measurement, preserving correctness checks:

```sh
CLQR_RUN_EXTERNAL=0 CLQR_RUN_JAX=0 CLQR_RUN_ORIGINAL_TABLE=0 \
  bash scripts/paper_benchmarks.sh /path/to/new-results --cuda
```

Individual overrides win: `CLQR_RUN_EXTERNAL=0 CLQR_RUN_VANROYE=1`
includes only Vanroye among the external methods. Laine comparisons without
Yang fetch Eigen headers directly, not GTSAM. Selected settings are saved in
`benchmark_options.txt`; disabled backends are not required by the summary.
The notebook and desktop runner inherit the same environment variables.

The [paper comparison notebook](notebooks/kaggle_paper_comparison.ipynb)
runs this workflow from a fresh Kaggle GPU session using an uploaded
`clqr-source.bundle` snapshot or, by default, current `origin/main`,
archives the results and logs, and removes its own downloaded/build cache.
It records CPU topology/affinity/memory and GPU model, compute capability,
memory, driver, clocks, and CUDA/compiler versions in `platform.txt` and
`gpu.csv`. CUDA architecture is detected rather than fixed to P100; set
`CLQR_CUDA_ARCH` explicitly when choosing among heterogeneous GPUs.

The same archiving/cleanup runner works on a Linux CUDA desktop, including
Blackwell, without a notebook. With Git, CMake, a C++ compiler, Python 3, a
GPU-compatible CUDA toolkit (including `compute-sanitizer`), Internet access,
and at least 5 GiB free, run from the repository root:

```sh
mkdir -p ../clqr-desktop-results
python3 -u scripts/notebook_paper.py \
  --work-dir "$(cd ../clqr-desktop-results && pwd)" \
  --suite all --jobs 4
```

It prints the `paper-results.zip` path and removes only that run's private
dependency/build cache. GPU architecture and system details are detected and
recorded; no P100-specific architecture flag is needed.

To create a source bundle, run these commands from a full-history clone with
local `main` at the revision you want to test, choosing an output path outside
the repository:

```sh
git bundle create /path/to/clqr-source.bundle main
git bundle verify /path/to/clqr-source.bundle
```

Upload `clqr-source.bundle` as a Kaggle dataset and attach it to the notebook.
The bundle contains the committed history of `main`, not uncommitted edits,
build outputs, or external dependencies, and is not checked into the repository.
Internet is still required for pinned dependencies. Without an attached bundle,
the notebook fetches current `origin/main` instead.

The default `all` suite crosses every power-of-two horizon from 32 through
32768 with every state dimension $n=8,16,24,32,48,64$ (66 cases), always with
$m=n/2$, $p_s=n/4$, and $p_m=n/8$. State-only rows at the fixed initial
state are omitted to avoid introducing artificial redundancy. Large cases can
take substantially longer or exceed available memory; reported failures remain
in the results. A separate
`CLQR_PAPER_SUITE=constraints` diagnostic varies constraint counts;
`CLQR_PAPER_SUITE=dimension` measures all six state dimensions at $N=128$;
`CLQR_PAPER_SUITE=horizon` restricts the horizon sweep to $n=8,16$.
The archiving runner accepts the corresponding `--suite` options. Smaller
diagnostics are opt-in: complete-grid runs never silently omit a pair.
All measured cases remain in the CSVs even when the generated paper table
selects only $n=8,16$.
The corrected Laine–Tomlin implementation is included separately from the
author's original, using the optimized native dense kernels.
Each native or JAX paper-comparison case performs exactly one untimed warmup
solve before measuring repeated-use latency. Individual solves are timed until
their cumulative measured time reaches one second, finishing the current solve
and imposing no minimum repetition count. Set `CLQR_BENCHMARK_SECONDS` to change
that target; an explicit `CLQR_BENCHMARK_REPEATS` or runner `--repeats` selects
fixed-count sampling for diagnostics. Prepared solve and setup-plus-solve each
have their own duration target. CSVs retain actual sample counts and individual
times in chronological order; variability is unavailable for a single sample.
Each solve refactors; setup and setup-plus-solve times are reported separately.
Primal, original-objective, and available
original KKT residuals are audited outside the timing interval. The original
Laine adapter includes the author's multiplier recovery in each timed solve
and reports KKT residuals using those returned multipliers. Corrected Laine–Tomlin
also recovers its own multipliers within timing. The factor-graph adapter
returns only primals, so its unavailable dual residuals are
reported as `nan`. A separate `planted_dual_stationarity_inf` column checks
each returned primal against the fixture's known optimal multipliers, without
attributing those multipliers to the solver. `primal_error` and `dual_error_inf`
are absolute infinity-norm differences from the known planted optimal solution,
not from an arbitrarily chosen competing solver. Missing dual outputs are `nan`.
The unified `measurements.csv` retains all these columns; no new measurements
are inserted into the paper automatically. Dense-comparison numerical errors and solver rejections
remain visible in the CSV and summary without failing the benchmark run. Missing or malformed
data, regression-test failures, and sanitizer errors still produce a nonzero
exit status. No runtimes from different hosts are
combined. Reference code and results stay in the chosen output directory.
For repeated local runs, `CLQR_PAPER_CACHE_DIR` reuses dependency checkouts
and builds; the driver verifies their pinned revisions and rejects dirty
reference sources. Each run still writes to a new results directory.
The driver also produces `summary.json`; a complete CUDA run produces LaTeX
table files. The summary checks matching cases/seeds, preserves failed or
inaccurate rows, and uses the first comparison round rather than selecting the
fastest round. Each backend is measured once by default; set
`CLQR_BENCHMARK_ROUNDS=2` for an additional, reverse-order CPU comparison round.
Each paper benchmark executable contains one backend and only its dependencies;
the shared harness source is compiled separately for each method. The script
runs separate Vanroye, Yang, and Laine sweeps, with no implicit CPU baseline.
The driver isolates each backend/case in a
subprocess, retaining its raw CSV/log and updating the combined CSV after every
case; a killed process cannot discard the remaining cases in its sweep.
The separate original-table reproduction retains its fixed-count protocol.

For a local comparison on the shared adversarial unit-test fixtures, build
the `clqr_adversarial_benchmark` CMake target, then run
`python3 benchmarks/reference/run_adversarial.py
<build-directory> <new-results-directory>`. Both rounds report unscaled
original KKT residuals as well as the unit tests' row-normalized residuals.
If the factor-graph and Laine adversarial targets are built, the runner includes
them automatically, isolating crashes and preserving numerical failures.

See the [reference comparison scope](benchmarks/reference/README.md) for
algorithm assumptions, adapter limitations, and output differences.

The core C++ library and public API remain dependency-free. Its native dense
kernels use fixed-size SIMD tiles with no heap allocation or matrix-packing
workspace. External solvers (and their BLASFEO/GTSAM dependencies) are fetched
and built only by the explicit comparison driver, not by library builds or
`bazel test //...`. Eigen is a Bazel development dependency used only by the
reference-adapter tests; it is omitted for downstream library consumers.
`bazel build //:clqr --ignore_dev_dependency` checks that separation.
CUDA-resident timing uses the native JAX FFI, excludes compilation and initial placement,
blocks on every result, and audits that the solver performs no bulk scalar
host round trip. Its inputs come directly from the C++ fixture generator.
Sizes rejected by the native CUDA workspace planner are explicitly skipped
in the JAX run using that same run's capacity report; these are not counted
as successful solves.

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

Sample workspace-API results from `bazel-bin/clqr_benchmark 5`, after building
`//:clqr_benchmark -c opt` on arm64 macOS with Apple clang 21.0.0. The
benchmark reserves workspace once per problem and times repeated solves. It
also reports `max_us`; local scheduler spikes can make maxima unrepresentative,
so median and p90 are usually better summary statistics.

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

For revision-to-revision comparison of the constrained CPU path, including
adversarial row scaling, redundant rows, primal/KKT residuals, and alternating
benchmark order, run:

```sh
CLQR_BASE_REVISION=<baseline> \
CLQR_CANDIDATE_REVISION=<candidate> \
scripts/compare_cpu_constraint_revisions.sh
```

The focused comparison defaults to FP64 and FP32, 101 timed solves per process,
and seven alternating-order rounds.

C++ users include `clqr/clqr.h` and call `clqr::Solve` with a solve
workspace:

```cpp
clqr::SolveWorkspace workspace;
workspace.reserve(problem);  // one allocation owned by workspace
clqr::SolutionView result = clqr::Solve(problem, workspace);
```

For fixed-size uniform unconstrained problems, the required byte count is `constexpr`:

```cpp
constexpr std::size_t kBytes = clqr::SolveWorkspace::num_bytes(64, 6, 3);
alignas(std::max_align_t) std::array<unsigned char, kBytes> memory{};
clqr::SolveWorkspace workspace;
workspace.mem_assign(problem, memory.data());
clqr::SolutionView result = clqr::Solve(problem, workspace);  // zero heap allocations
```

There is also a constexpr size helper for uniform constrained workspace solves:

```cpp
constexpr std::size_t kBytes =
    clqr::SolveWorkspace::num_bytes(16, 4, 2, 1);
```

The workspace API covers constrained and unconstrained problems. Unconstrained problems use
the raw Riccati path directly. Constrained problems activate the workspace arena and run the
constraint-elimination algorithm, including the reduced Riccati solve and multiplier recovery.

For repeated solves with fixed matrices and new right-hand sides, cache the
constraint elimination and reduced Riccati factorization once:

```cpp
clqr::Factorization factors = clqr::Factor(problem);
clqr::SolveRhs rhs = clqr::ExtractRhs(problem);
clqr::SolveWorkspace solve_workspace;
solve_workspace.reserve(factors);

clqr::SolutionView first = clqr::Solve(factors, rhs, solve_workspace);
rhs.initial_state = next_initial_state;
rhs.q[0] = next_state_gradient;
rhs.q.back() = next_terminal_gradient;
rhs.stages[0].d = next_mixed_constraint_offset;
rhs.stages[0].e = next_state_constraint_offset;
rhs.terminal_e = next_terminal_constraint_offset;
clqr::SolutionView second = clqr::Solve(factors, rhs, solve_workspace);
```

`Factor` owns the fixed `A`, `B`, `Q`, `R`, `M`, `C`, `D`, `E`, and
`terminal_E` data. Each factored `Solve` accepts new `c`, all `N + 1` entries
of `q`, `r`, `d`, `e`, `terminal_e`, and the initial state with unchanged
dimensions. It performs no heap allocation when given a reserved workspace.

The convenience `Factor(problem)` call owns and dynamically allocates its
persistent cache. For a completely caller-allocated factor/solve path, use
the `num_bytes` / `mem_assign` pattern:

```cpp
constexpr std::size_t kFactorBytes =
    clqr::FactorizationWorkspace::num_bytes(16, 4, 2, 1);
constexpr std::size_t kSolveBytes =
    clqr::SolveWorkspace::num_bytes(16, 4, 2, 1);

alignas(std::max_align_t)
    std::array<unsigned char, kFactorBytes> factor_memory{};
alignas(std::max_align_t)
    std::array<unsigned char, kSolveBytes> solve_memory{};

clqr::FactorizationWorkspace factor_workspace;
factor_workspace.mem_assign(problem, factor_memory.data());
clqr::Factorization factors =
    clqr::Factor(problem, factor_workspace);  // zero heap allocations

clqr::SolveWorkspace solve_workspace;
solve_workspace.mem_assign(factors, solve_memory.data());
clqr::SolveRhs rhs = clqr::ExtractRhs(problem);
clqr::SolutionView result =
    clqr::Solve(factors, rhs, solve_workspace);  // zero heap allocations
```

The constexpr overloads take capacity bounds for a uniform problem: horizon,
state dimension, control dimension, mixed rows per stage, state-only rows per
stage, and terminal rows. Runtime `num_bytes(problem)` supports heterogeneous
dimensions and returns a tighter bound. `FactorizationWorkspace::reserve`
provides the corresponding one-allocation owning path.

An externally backed `FactorizationWorkspace` must outlive its
`Factorization`. Destroy the factorization before reassigning, reserving, or
reusing that workspace. The factorization remains immutable and may be shared
by concurrent solves, provided each solve has its own `SolveWorkspace`.

As with the ordinary workspace API, every buffer and string referenced by the
returned `SolutionView` is workspace-backed and remains valid only until that
workspace is reused or destroyed. Copy any values that must survive the next
solve.

### Native NumPy binding

The Python extension is built by the Bazel target `//:_clqr`; the shared-object output is
addressable as `//:_clqr.so`. It exposes the `_clqr` module directly:

```python
import _clqr

result = _clqr.solve({
    "initial_state": ...,
    "stages": [
        {
            "A": ..., "B": ..., "c": ...,
            "R": ..., "M": ..., "r": ...,
            "C": ..., "D": ..., "d": ...,  # optional
            "E": ..., "e": ...,            # optional
        },
    ],
    "Q": [Q_0, ..., Q_N],
    "q": [q_0, ..., q_N],
    "terminal_E": ...,  # optional
    "terminal_e": ...,  # optional
})
```

For repeated solves, the Python extension exposes the same matrix/RHS split as
C++:

```python
factors = _clqr.factor(problem)
solve_rhs = _clqr.rhs(problem)

first = factors.solve(solve_rhs)
solve_rhs.initial_state = next_initial_state
solve_rhs.stage(0).c = next_dynamics_offset
solve_rhs.stage(0).r = next_control_gradient
solve_rhs.stage(0).d = next_mixed_constraint_offset
solve_rhs.stage(0).e = next_state_constraint_offset
solve_rhs.terminal_e = next_terminal_constraint_offset
solve_rhs.set_q(0, next_state_gradient)
solve_rhs.set_q(solve_rhs.q_count - 1, next_terminal_gradient)
second = factors.solve(solve_rhs)
```

`Factorization` owns the matrices and internally reuses its native solve
workspace. Assigning an RHS property copies that array into owned native
storage; result arrays are newly owned NumPy arrays.

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
adapter validates and pads the same dictionary schema used by the native
NumPy binding. The split lets callers replace any RHS vector without changing
the compiled JAX shape. It is not a numerical factor/solve split: the current
call refactors after either part changes.

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
