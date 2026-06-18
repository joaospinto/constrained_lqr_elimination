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

Sample results from `bazel-bin/clqr_benchmark 5` after building `//:clqr_benchmark -c opt`
on arm64 macOS with clang 22.1.1. The benchmark also reports `max_us`; local scheduler
spikes can make maxima unrepresentative, so median and p90 are usually better summary
statistics.

| Case | Iterations | Mean | Median | P90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| `N=16 n=4 m=2 p=0` | 1000 | `31.8 us` | `31.3 us` | `34.8 us` | `27.8 us` | `162 us` |
| `N=16 n=4 m=2 p=1` | 1000 | `47.1 us` | `46.9 us` | `49.1 us` | `43.2 us` | `226 us` |
| `N=16 n=4 m=2 p=2` | 1000 | `43.7 us` | `41.5 us` | `47.8 us` | `39.6 us` | `152 us` |
| `N=16 n=6 m=3 p=0` | 500 | `49.8 us` | `49.1 us` | `50.3 us` | `46.6 us` | `189 us` |
| `N=16 n=6 m=3 p=1` | 500 | `72.9 us` | `71.9 us` | `73.4 us` | `70.2 us` | `202 us` |
| `N=16 n=6 m=3 p=2` | 500 | `76.2 us` | `70.5 us` | `74.7 us` | `69.4 us` | `333 us` |
| `N=32 n=6 m=3 p=0` | 250 | `109 us` | `94.8 us` | `101 us` | `93.8 us` | `438 us` |
| `N=32 n=6 m=3 p=1` | 250 | `217 us` | `143 us` | `350 us` | `141 us` | `9.51 ms` |
| `N=32 n=6 m=3 p=2` | 250 | `144 us` | `142 us` | `143 us` | `140 us` | `301 us` |
| `N=64 n=6 m=3 p=0` | 100 | `190 us` | `188 us` | `189 us` | `187 us` | `347 us` |
| `N=64 n=6 m=3 p=1` | 100 | `284 us` | `280 us` | `283 us` | `278 us` | `490 us` |
| `N=64 n=6 m=3 p=2` | 100 | `283 us` | `279 us` | `281 us` | `278 us` | `477 us` |
| `N=128 n=8 m=4 p=0` | 50 | `490 us` | `485 us` | `495 us` | `478 us` | `661 us` |
| `N=128 n=8 m=4 p=1` | 50 | `721 us` | `710 us` | `718 us` | `705 us` | `916 us` |
| `N=128 n=8 m=4 p=2` | 50 | `704 us` | `692 us` | `704 us` | `687 us` | `979 us` |

All sample cases reported `singular_count=0` and `wrong_inertia_count=0`.

C++ users include `clqr/clqr.h` and call `clqr::Solve`.

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
