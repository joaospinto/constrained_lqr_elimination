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
| `N=16 n=4 m=2 p=0` | 1000 | `16.1 us` | `18.0 us` | `18.6 us` | `11.4 us` | `42.2 us` |
| `N=16 n=4 m=2 p=1` | 1000 | `33.9 us` | `32.8 us` | `42.5 us` | `27.9 us` | `132 us` |
| `N=16 n=4 m=2 p=2` | 1000 | `26.5 us` | `25.9 us` | `27.0 us` | `24.6 us` | `82.7 us` |
| `N=16 n=6 m=3 p=0` | 500 | `25.5 us` | `25.2 us` | `25.9 us` | `25.0 us` | `37.9 us` |
| `N=16 n=6 m=3 p=1` | 500 | `48.3 us` | `47.4 us` | `49.7 us` | `46.3 us` | `99.4 us` |
| `N=16 n=6 m=3 p=2` | 500 | `46.1 us` | `45.4 us` | `47.5 us` | `44.7 us` | `77.4 us` |
| `N=32 n=6 m=3 p=0` | 250 | `49.6 us` | `48.2 us` | `51.6 us` | `47.9 us` | `82.1 us` |
| `N=32 n=6 m=3 p=1` | 250 | `93.4 us` | `92.1 us` | `96.7 us` | `91.2 us` | `118 us` |
| `N=32 n=6 m=3 p=2` | 250 | `93.2 us` | `92.8 us` | `95.9 us` | `90.9 us` | `111 us` |
| `N=64 n=6 m=3 p=0` | 100 | `97.1 us` | `95.8 us` | `101 us` | `94.7 us` | `109 us` |
| `N=64 n=6 m=3 p=1` | 100 | `186 us` | `183 us` | `195 us` | `180 us` | `284 us` |
| `N=64 n=6 m=3 p=2` | 100 | `184 us` | `180 us` | `194 us` | `179 us` | `260 us` |
| `N=128 n=8 m=4 p=0` | 50 | `375 us` | `374 us` | `385 us` | `366 us` | `394 us` |
| `N=128 n=8 m=4 p=1` | 50 | `595 us` | `589 us` | `612 us` | `577 us` | `702 us` |
| `N=128 n=8 m=4 p=2` | 50 | `574 us` | `568 us` | `591 us` | `562 us` | `638 us` |

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
