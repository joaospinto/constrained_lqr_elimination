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

Sample results from `bazel run -c opt //:clqr_benchmark -- 5` on arm64 macOS with
clang 22.1.1:

| Case | Iterations | Mean | Min | Max |
|---|---:|---:|---:|---:|
| `N=16 n=4 m=2 p=0` | 1000 | `68.2 us` | `56.2 us` | `739 us` |
| `N=16 n=4 m=2 p=1` | 1000 | `92.8 us` | `78.2 us` | `1.69 ms` |
| `N=16 n=4 m=2 p=2` | 1000 | `86.9 us` | `75.4 us` | `1.74 ms` |
| `N=16 n=6 m=3 p=0` | 500 | `97.0 us` | `76.9 us` | `1.70 ms` |
| `N=16 n=6 m=3 p=1` | 500 | `121 us` | `105 us` | `2.91 ms` |
| `N=16 n=6 m=3 p=2` | 500 | `113 us` | `103 us` | `1.76 ms` |
| `N=32 n=6 m=3 p=0` | 250 | `173 us` | `151 us` | `1.11 ms` |
| `N=32 n=6 m=3 p=1` | 250 | `300 us` | `217 us` | `3.97 ms` |
| `N=32 n=6 m=3 p=2` | 250 | `250 us` | `209 us` | `1.35 ms` |
| `N=64 n=6 m=3 p=0` | 100 | `333 us` | `299 us` | `1.37 ms` |
| `N=64 n=6 m=3 p=1` | 100 | `460 us` | `417 us` | `805 us` |
| `N=64 n=6 m=3 p=2` | 100 | `460 us` | `417 us` | `2.04 ms` |
| `N=128 n=8 m=4 p=0` | 50 | `753 us` | `662 us` | `2.34 ms` |
| `N=128 n=8 m=4 p=1` | 50 | `1.06 ms` | `938 us` | `2.65 ms` |
| `N=128 n=8 m=4 p=2` | 50 | `1.00 ms` | `929 us` | `1.87 ms` |

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
