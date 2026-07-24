# CPU constraint-elimination QR evidence

This comparison isolates commit `9641f73` (column-pivoted Householder QR
constraint elimination) from its direct parent `b92d127` (RREF elimination).
The final QR candidate also normalizes independently scaled mixed-constraint
columns during multiplier recovery. It does not use regularization, iterative
refinement, or changed solver tolerances.

The benchmark compiles both revisions with the same benchmark source, pinned
fixture/oracle header from the candidate checkout, and compiler flags. It
reserves one workspace per problem, performs five untimed warmups, and times
101 solves per process. Seven process rounds alternate baseline/candidate
order; the table reports the median of the seven within-process medians. Primal
and KKT residuals are computed outside the timed region by that common oracle.

Run:

```sh
CLQR_BASE_REVISION=b92d127 \
CLQR_CANDIDATE_REVISION=HEAD \
CLQR_BENCHMARK_REPEATS=101 \
CLQR_COMPARISON_ROUNDS=7 \
scripts/compare_cpu_constraint_revisions.sh
```

Recorded 2026-07-24 on arm64 macOS (Darwin 25.5.0, Apple clang 21.0.0,
`-O3 -DNDEBUG`). Ratios above one favor QR.

## FP64 (primary)

| Case | RREF | QR | RREF/QR | RREF KKT | QR KKT |
|---|---:|---:|---:|---:|---:|
| accuracy-limit N17 | 17.084 us | 18.083 us | 0.9448 | 3.359e-11 | 4.780e-11 |
| alternating N64 | 173.250 us | 185.542 us | 0.9338 | 3.122e-10 | 3.949e-13 |
| mixed N32 | 44.292 us | 50.417 us | 0.8785 | 1.319e-14 | 7.304e-15 |
| more mixed than controls N5 | 7.167 us | 8.542 us | 0.8390 | 1.066e-14 | 1.915e-15 |
| redundant N32 | 45.917 us | 52.834 us | 0.8691 | 2.548e-15 | 7.317e-16 |
| independently scaled N32 | 41.209 us | 51.000 us | 0.8080 | 4.297e-15 | 4.604e-15 |
| terminal-only N64 | 59.750 us | 60.291 us | 0.9910 | 1.876e-14 | 3.791e-14 |

QR costs 5-19% on cases that repeatedly eliminate mixed constraints (23% for
the independently-scaled stress case). The strongest FP64 accuracy changes are
a 791x lower KKT residual for alternating constraints, 5.6x for more mixed rows
than controls, and 3.5x for redundant rows. All reported FP64 residuals remain
at or below `4.8e-11`.

The uncorrected QR commit `9641f73` returned `numerical_failure` for the
independently-scaled N32 case in both precisions (`mixed-only multiplier
recovery`, FP64 stage 31 and FP32 stage 11). Scale-invariant column
normalization restores an optimal solution; FP64 KKT is `4.604e-15`.

## FP32 (secondary)

| Case | RREF | QR | RREF/QR | RREF KKT | QR KKT |
|---|---:|---:|---:|---:|---:|
| accuracy-limit N17 | 17.750 us | 18.500 us | 0.9595 | 1.981e-2 | 9.018e-3 |
| alternating N64 | 161.709 us | 175.250 us | 0.9227 | 6.087e-2 | 1.543e-4 |
| mixed N32 | 40.750 us | 46.916 us | 0.8686 | 6.692e-6 | 5.929e-6 |
| more mixed than controls N5 | 7.041 us | 8.500 us | 0.8284 | 9.015e-6 | 5.782e-6 |
| redundant N32 | 43.000 us | 50.708 us | 0.8480 | 6.216e-7 | 3.153e-7 |
| independently scaled N32 | 39.000 us | 48.125 us | 0.8104 | 2.629e-6 | 1.142e-5 |
| terminal-only N64 | 56.750 us | 56.833 us | 0.9985 | 2.331e-5 | 1.208e-5 |

All listed FP64 and FP32 cases return `optimal`. The benchmark reports raw
measurements and does not turn FP32 residual variation into a pass/fail gate.
