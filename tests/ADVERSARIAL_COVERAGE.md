# Adversarial validation

`adversarial_test_support.h` is the single source of deterministic fixtures,
the nonfinite-safe scaled primal/KKT residual checks, and the structurally
independent dense-KKT primal oracle used by the adversarial suites. The dense
oracle assembles a monolithic KKT system independently from either solver but
shares the library's low-level RREF and linear-system primitives.

| Property | Sequential C++ | Kernel emulation | Native CUDA |
|---|---:|---:|---:|
| Empty and zero-horizon problems | yes | yes | yes |
| Feasible zero-horizon terminal equalities | yes | yes | yes |
| Horizon 1, powers of two, and non-powers of two | yes | yes | yes |
| Horizons 31/32/33, 63/65, 127, 257, and 1025 | extended | through 257 | extended |
| Nonuniform state/control dimensions | yes | yes | yes |
| Zero controls | yes | yes | yes |
| Mixed rows exceeding control dimension | yes | yes | yes |
| Redundant/rank-deficient equalities | yes | yes | yes |
| Independently scaled rows | yes | yes | yes |
| Subnormal/largest-finite row scaling and unrepresentable pullbacks | yes | no | no |
| Initial, stagewise, and terminal state equalities | yes | yes | yes |
| Initial/terminal infeasibility | yes | no (public solver path only) | yes |
| Nonunique equality multipliers | yes | yes | yes |
| Non-positive-definite reduced Hessian | yes | kernel-level suite | yes |
| Invalid shapes | yes | no (public packing path only) | yes |
| Workspace reuse across changing shapes | yes | n/a | yes |
| Fixed-seed structural property corpus | extended | representative subset | extended |
| Dense-KKT primal oracle | small well-conditioned cases | CPU comparison | small well-conditioned cases |
| FP32 and FP64 | yes | yes | yes |
| ASan/UBSan | yes | yes | no |
| memcheck/initcheck/racecheck/synccheck | n/a | n/a | stress notebook |

JAX binding validation is separate from the native table. The CPU and CUDA
binding suites cover eager execution, `jax.jit`, sequential `jax.vmap`, changed
right-hand sides, heterogeneous dimensions, zero controls, and zero horizons.
The native Metal binding suite additionally covers all constraint families,
rank deficiency and failure statuses, deterministic property cases, workspace
reuse, and horizons through 1025 under its FP32-only precision contract.

The normal CI-sized suite is:

```sh
bazel test //... --config=fp64
bazel test //... --config=fp32
```

The extended deterministic CPU/property suite is:

```sh
bazel test //:adversarial_cpu_extended_test --config=fp64
bazel test //:adversarial_cpu_extended_test --config=fp32
```

The extended emulation suite adds scan-boundary horizons through 257 and the
selected fixed-seed property cases:

```sh
bazel test //:cuda_kernel_emulation_extended_test \
  --config=fp64 \
  --test_output=errors
```

FP32 emulation uses the same multiplier-consistency gate and a quantitative
`3e-2` KKT residual gate. CPU and native CUDA retain their independent KKT
gates. The deliberately duplicate-row exact-JAX
fixture remains FP64-only; the shared redundant-row cases cover FP32.

The hard generated inputs are not replaced by friendlier seeds. In particular,
horizon-17 seed 27 has a full-rank 71-by-88 equality Jacobian and full-rank
159-by-159 KKT matrix at the FP32 rank threshold, but the CUDA FP32 dual
recovery rejects it on the reference build rather than returning its roughly
`4e-1` KKT residual. Such accuracy-limit cases may either produce their
documented safe rejection or complete and satisfy their quantitative KKT gate.
The rejecting tree node is intentionally not prescribed: equivalent
floating-point instruction orderings can detect the same documented
phase/diagnostic at different nodes.

With pivoted-QR constraint elimination, the same deliberately ill-conditioned
seed has a largest observed Linux FP32 CPU state-stationarity residual of about
`4.8e-2`. Only that fixture uses three times the ordinary FP32 KKT tolerance;
its primal and dense-reference gates are unchanged. The same fixture-specific
scale is applied by sequential CPU, emulation, and native CUDA. Separately
named stable fixtures at the same scan boundaries retain the ordinary KKT
gate except for the 257-stage FP32 emulation fixture, which permits twice the
ordinary tolerance: direct Cholesky reuse produces a roughly `5.1e-2`
residual there while avoiding a second factorization of every reduced control
Hessian. FP64 must solve both sets with the ordinary gate.

The original 1025-stage alternating seed is also a documented FP32
accuracy-limit fixture. Its conditional-value interface becomes too
ill-conditioned for a reliable pure-FP32 solve on the reference P100, so
native CUDA may reject it as a numerical failure. An instruction ordering that
does complete must still satisfy the ordinary quantitative KKT gate. The FP64
expectation remains an optimal solve with the ordinary gate.

On a CUDA machine, run the extended native target explicitly:

```sh
bazel build //:adversarial_cuda_extended_test \
  --config=fp64 --config=cuda \
  --cuda_archs=sm_60
./bazel-bin/adversarial_cuda_extended_test --extended
```

The notebook driver likewise builds native tests with Bazel and executes them
directly and sequentially. This avoids Bazel test-runner CUDA library/sandbox
differences and prevents independent GPU test processes from overlapping.

The stress driver runs the full extended corpus for FP64 and the standard
representative native and Compute Sanitizer suites for FP32:

```sh
CLQR_PRECISIONS="FP64 FP32" bash scripts/notebook_cuda_stress.sh
```

Pathological extended FP32 long-horizon cases are explicit, non-gating
diagnostics. Opt in with:

```sh
CLQR_PRECISIONS="FP64 FP32" \
CLQR_RUN_FP32_EXTENDED_STRESS=1 \
bash scripts/notebook_cuda_stress.sh
```

This opt-in contains the documented 1025-stage FP32 numerical-limit fixture.
It does not change FP64 coverage or the standard and stable FP32 gates.

The reproducible machine-report and Compute Sanitizer workflow is
`notebooks/kaggle_cuda_stress.ipynb`.
