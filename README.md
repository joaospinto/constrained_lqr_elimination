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

Sample results from `bazel run -c opt //:clqr_benchmark` on arm64 macOS with clang 22.1.1:

| Case | Iterations | Mean | Min | Max |
|---|---:|---:|---:|---:|
| `N=16 n=4 m=2 p=0` | 200 | `127 us` | `96.9 us` | `583 us` |
| `N=16 n=4 m=2 p=1` | 200 | `119 us` | `110 us` | `366 us` |
| `N=16 n=4 m=2 p=2` | 200 | `119 us` | `105 us` | `1.19 ms` |
| `N=16 n=6 m=3 p=0` | 100 | `142 us` | `133 us` | `252 us` |
| `N=16 n=6 m=3 p=1` | 100 | `168 us` | `153 us` | `484 us` |
| `N=16 n=6 m=3 p=2` | 100 | `157 us` | `149 us` | `237 us` |
| `N=32 n=6 m=3 p=0` | 50 | `290 us` | `262 us` | `744 us` |
| `N=32 n=6 m=3 p=1` | 50 | `340 us` | `305 us` | `977 us` |
| `N=32 n=6 m=3 p=2` | 50 | `315 us` | `303 us` | `417 us` |
| `N=64 n=6 m=3 p=0` | 20 | `575 us` | `537 us` | `1.15 ms` |
| `N=64 n=6 m=3 p=1` | 20 | `628 us` | `599 us` | `778 us` |
| `N=64 n=6 m=3 p=2` | 20 | `641 us` | `602 us` | `884 us` |
| `N=128 n=8 m=4 p=0` | 10 | `1.48 ms` | `1.20 ms` | `3.83 ms` |
| `N=128 n=8 m=4 p=1` | 10 | `1.60 ms` | `1.41 ms` | `2.34 ms` |
| `N=128 n=8 m=4 p=2` | 10 | `1.45 ms` | `1.33 ms` | `2.07 ms` |

All sample cases reported `singular_count=0` and `wrong_inertia_count=0`.

C++ users include `clqr/clqr.h` and call `clqr::Solve`.

The Python extension target is `//:_clqr.so`. It exposes:

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

Arrays are NumPy-compatible `float64` arrays. The result dict contains `status`, `message`,
`newton_kkt_singular`, `newton_kkt_wrong_inertia`, `newton_kkt_diagnostic`, `objective`,
`states`, `controls`, `initial_multiplier`, `dynamics_multipliers`, `mixed_multipliers`,
`state_multipliers`, and `terminal_state_multiplier`. The multiplier signs correspond to the
constraints exactly as written above. The Newton-KKT diagnostic fields are reported separately
from `status`; when the reduced solve can proceed, a candidate solution is still returned.
