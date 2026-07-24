# CPU constraint-elimination QR evidence

This comparison isolates commit `9641f73` (column-pivoted Householder QR
constraint elimination) from its direct parent `b92d127` (RREF elimination).
The final QR candidate also includes `33d8d2c`, which normalizes independently
scaled mixed-constraint columns during multiplier recovery. It does not use
regularization, iterative refinement, or changed solver tolerances.

The benchmark compiles both revisions with the same benchmark source, pinned
fixture/oracle header from the candidate checkout, and compiler flags. It
reserves one workspace per problem, performs five untimed warmups, and times
101 solves per process. Seven process rounds alternate baseline/candidate
order; the table reports the median of the seven within-process medians. Primal
and KKT residuals are computed outside the timed region by that common oracle.

Run:

```sh
CLQR_BASE_REVISION=b92d127 \
CLQR_CANDIDATE_REVISION=33d8d2c \
CLQR_BENCHMARK_REPEATS=101 \
CLQR_COMPARISON_ROUNDS=7 \
scripts/compare_cpu_constraint_revisions.sh
```

Recorded 2026-07-24 on arm64 macOS (Darwin 25.5.0, Apple clang 21.0.0,
`-O3 -DNDEBUG`). Ratios above one favor QR.

## FP64 (primary)

| Case | RREF | QR | RREF/QR | RREF KKT | QR KKT |
|---|---:|---:|---:|---:|---:|
| accuracy-limit N17 | 17.083 us | 18.000 us | 0.9491 | 3.359e-11 | 4.780e-11 |
| alternating N64 | 171.916 us | 185.250 us | 0.9280 | 3.122e-10 | 3.949e-13 |
| mixed N32 | 44.250 us | 50.375 us | 0.8784 | 1.319e-14 | 7.304e-15 |
| more mixed than controls N5 | 7.166 us | 8.500 us | 0.8431 | 1.066e-14 | 1.915e-15 |
| redundant N32 | 45.459 us | 53.000 us | 0.8577 | 2.548e-15 | 7.317e-16 |
| independently scaled N32 | 41.083 us | 50.500 us | 0.8135 | 4.297e-15 | 4.604e-15 |
| terminal-only N64 | 59.667 us | 59.958 us | 0.9951 | 1.876e-14 | 3.791e-14 |

QR costs 5-19% on cases that repeatedly eliminate mixed constraints (23% for
the independently-scaled stress case). The strongest FP64 accuracy changes are
a 791x lower KKT residual for alternating constraints, 5.6x for more mixed rows
than controls, and 3.5x for redundant rows. All reported FP64 residuals remain
at or below `4.8e-11`.

The uncorrected QR commit `9641f73` returned `numerical_failure` for the
independently-scaled N32 case in both precisions (`mixed-only multiplier
recovery`, FP64 stage 31 and FP32 stage 11). Commit `33d8d2c` restores an
optimal solution through scale-invariant column normalization; FP64 KKT is
`4.604e-15`.

## FP32 (secondary)

| Case | RREF | QR | RREF/QR | RREF KKT | QR KKT |
|---|---:|---:|---:|---:|---:|
| accuracy-limit N17 | 17.291 us | 18.458 us | 0.9368 | 1.981e-2 | 9.018e-3 |
| alternating N64 | 162.125 us | 174.833 us | 0.9273 | 6.087e-2 | 1.543e-4 |
| mixed N32 | 40.667 us | 46.875 us | 0.8676 | 6.692e-6 | 5.929e-6 |
| more mixed than controls N5 | 7.041 us | 8.542 us | 0.8243 | 9.015e-6 | 5.782e-6 |
| redundant N32 | 42.333 us | 50.375 us | 0.8404 | 6.216e-7 | 3.153e-7 |
| independently scaled N32 | 38.375 us | 47.709 us | 0.8044 | 2.629e-6 | 1.142e-5 |
| terminal-only N64 | 56.167 us | 56.583 us | 0.9926 | 2.331e-5 | 1.208e-5 |

All listed FP64 and FP32 cases return `optimal`. The benchmark reports raw
measurements and does not turn FP32 residual variation into a pass/fail gate.
