# Laine–Tomlin reference implementation

This is **our independent C++ reimplementation** of the sequential constrained
dynamic-programming method of Forrest Laine and Claire Tomlin, not their source
code and not the CLQR elimination algorithm. It uses FP64 and Eigen storage,
SVDs, and Cholesky factorizations. Matrix products and multiple-RHS triangular
solves use CLQR's optimized native CPU kernels; it does not call or modify
the CPU, CUDA, or Metal CLQR solvers. The fixture test
alone imports CLQR problem generators and compares the two implementations.

Reference: *Efficient Computation of Feedback Control for Equality-Constrained
LQR*, ICRA 2019, [DOI](https://doi.org/10.1109/ICRA.2019.8793566). The equations
audited here are from the publicly accessible
[arXiv v2 preprint](https://arxiv.org/abs/1807.00794v2), titled *Efficient
Computation of Feedback Control for Constrained Systems*. Equation numbers and
corrections below refer specifically to that version; the IEEE proceedings text
has not been independently checked.

## Recurrence and corrections

For each stage, combine its quadratic cost with the next value function
`0.5*x_next' V x_next + v' x_next` and substitute `x_next = A*x+B*u+c`.
With `Muu = R+B' V B`, `Mux = S'+B' V A`, the linear control term is
`mu = r+B' (v+V*c)`. Stack the local constraints and the propagated
constraints-to-go as `Nu*u+Nx*x+b=0`.

An SVD of `Nu` gives a particular control `up=Kp*x+kp`, with
`Kp=-Nu† Nx`, `kp=-Nu† b`, and an orthonormal null-space basis `Z`.
The remaining minimization is over `u=up+Z*w`:

```
W = Z' Muu Z
K = Kp - Z solve(W, Z' (Mux + Muu Kp))
k = kp - Z solve(W, Z' (mu  + Muu kp))
```

These expressions follow directly by setting the derivative with respect to
`w` to zero. The propagated constraint is
`(I-Nu Nu†)(Nx*x+b)=0`; its independent rows are compressed before continuing
backward. Substituting the complete policy into the quadratic gives the next
value function. Thus, by backward induction, the policy minimizes each feasible
suffix whenever the free-control Hessian `W` is positive definite. The forward
pass starts only if the supplied initial state satisfies the remaining
constraints.

Three details matter when transcribing the preprint:

- Equations (13), (15), (17), and (18) omit the coupling of the particular
  control with the free-control cost. Orthogonal `P` and `Z` do **not** imply
  `Z' Muu P=0`. The `Muu Kp` and `Muu kp` terms above are necessary.
- Equation (12)'s linear cost terms omit the affine dynamics contribution
  `V*c`. Both state and control gradients require it.
- Equation (24)'s quadratic expression must be stored symmetrically:
  `Vnew=Mxx+Mux' K+K' Mux+K' Muu K`. Simply storing `2*Mux' K` as a matrix
  can corrupt subsequent products despite representing the same quadratic form.

The analytic regression `min 0.5*u'[[2,1],[1,2]]u` subject to `u[0]=1`
has solution `(1,-0.5)` and cost `0.75`. Literal equation (18) instead yields
`(1,0)`, cost `1`, and free-control stationarity residual `1`. This is not a
floating-point effect. A second test checks the omitted affine dynamics term.

The preprint's full-column-rank test on `[H,h]` is sufficient, but not necessary,
for infeasibility. We check the left-null residual of `H*x+h=0` and compatibility
with the initial state, including inconsistent constant rows.

## Public implementations inspected

- [Laine's `parallel_lqr`](https://github.com/forrestlaine/parallel_lqr/blob/e24442fc853bd58fcab630c7b8b96116a53d151a/src/trajectory.cpp#L531)
  contains the correct particular/free coupling and affine dynamics terms in
  `perform_constrained_dynamic_programming_backup`. This supports the corrected
  recurrence above. The repository has no license file, and this routine does
  not compress redundant propagated rows. The optional adapter below tests
  its original backward and forward routines without editing that source.
- [Yang et al.'s comparison port](https://github.com/ShuoYangRobotics/equality-constraint-LQR-compare/blob/905dfc5397a1b6c12e8cf9925127c9cb228a9353/src/EcLqr_laine-impl.h)
  indexes vectors after `reserve` without `resize` and omits the cost coupling
  in the partially constrained branch.
- [Safe-PDP's `EQCLQR`](https://github.com/wanxinjin/Safe-PDP/blob/main/SafePDP/SafePDP.py)
  includes affine dynamics corrections, but counts singular values with
  `(S > threshold).size` rather than counting true entries, and omits the
  particular/free cost coupling.
- [`Constraint_LQR`](https://github.com/ggory15/Constraint_LQR/blob/master/BackPass.py)
  is a restricted Python example, not a general partially constrained solver.

The code here is newly written from the dynamic-programming subproblem and
corrected algebra, rather than copied from those repositories.

The author-written implementation has a separate finite-precision rank issue.
For `x[0]=1`, `x[t+1]=x[t]+u[t]`, unit state/control quadratic costs, and
`x[1]+u[1]=0`, the two-stage optimum is `u=(-2/3,-1/3)`. Adding the redundant
row `3*x[1]+3*u[1]=0` leaves the mathematical problem unchanged. In the original
code, the SVD left-null projection can leave a coefficient around `1e-16`;
the next purely relative rank test treats it as a new constraint, giving
`u=(-1,0)` instead. The objective increases from `5/6` to `1`. This reproduces
with Apple Clang and GCC, and Eigen 3.3 and 3.4. Our implementation removes
projection residuals at the rank-tolerance scale before normalizing and
compressing the propagated constraints. Its regression checks both controls
against the analytic answer and an independent dense oracle.

Constraint compression is explicitly prescribed after equations (20)–(21)
and in the preprint's complexity analysis; it is not a new method introduced
here. The inspected author code omits that step. Its uncompressed rows can
accumulate with the horizon, unlike the bounded representation used here.

Thus the corrected dynamic-programming algebra and the original code's
numerical robustness are distinct questions. The published arXiv expressions
are not correct as written; the author-written recursion includes the missing
terms but is not reliable for all redundant inputs. Zero-control cases also
reach an empty-matrix Eigen reduction, and the original routines do not return
an infeasibility status. These limitations are not silently repaired in the
author benchmark.

## Interface and scope

`Problem` holds stage-varying dynamics, quadratic/linear costs, affine equality
constraints, and the initial state. `Q` and `q` have `N+1` entries including the
terminal cost. `Stage::S` is the state/control cross-cost matrix. State-only rows
are represented with zero rows in `D`. State and control dimensions may vary;
zero controls and zero horizon are supported.

```cpp
#include "external_algorithms/laine_tomlin/solver.h"

laine_tomlin::Problem problem; // populate dynamics, costs, and constraints
auto solution = laine_tomlin::Solve(problem);
if (solution.status == laine_tomlin::Status::kOptimal) {
  // solution.states, solution.controls
  // feedback: u[t] = solution.K[t] * x[t] + solution.k[t]
}
```

The solver returns **primals and policies, not multipliers**. It does not
regularize a singular/indefinite free-control Hessian or apply iterative
refinement. Its `kOptimal` status denotes successful numerical recursion, not a
separate whole-horizon KKT certificate. Tests audit original, unscaled
constraints and optimality independently.

SVDs determine rank and apply pseudoinverses without explicitly forming an
inverse. A Cholesky solve handles the free-control cost. Equality rows are
equilibrated; this preserves the feasible optimum but changes the least-squares
extension of the policy outside the feasible set. `Options` exposes the rank
and feasibility tolerances. Constraint compression bounds propagated rows by
the state dimension. Work is linear in the horizon for bounded stage dimensions
and constraint counts, with cubic dense stage algebra. Storage is linear in the
horizon for policies and trajectories; this reference uses dynamic allocations.
The shared arithmetic kernels allocate no memory, but that does not make the
reference solver allocation-free. Its Eigen matrices, decompositions, and
returned trajectories still allocate. Product routing uses existing strides,
including column-major storage, without packing the matrices.

## Validation and local comparison

From the repository root:

```sh
bazel test --config=fp64 //external_algorithms/laine_tomlin:all
bazel run --config=fp64 //external_algorithms/laine_tomlin:compare -- --benchmark
```

`dense_test` checks native-kernel routing for both storage orders, transposes,
blocks, strided vectors/maps, and zero-sized products. `solver_test` checks
analytic counterexamples, zero-sized and inconsistent
problems, bounded constraint propagation at horizon 2048, and 256 deterministic
random convex problems against an independently assembled dense QP. These tests
do not link CLQR. `fixture_test` exercises the 74 shared adversarial cases and
cross-checks CLQR on the same 256 random problems. It uses an independent
sparse QR of the original constraint Jacobian transpose to
audit stationarity. The audit's multipliers are **not solver outputs**, are
computed outside timing, and never change the primal solution.

The optional benchmark prints both solvers' times and residuals without failing
on numerical outcomes. Each timed solve includes fresh factorization and primal
reconstruction; CLQR also recovers multipliers. Input conversion and validation
audits are untimed. This reference allocates per solve; CLQR uses its reserved
workspace. These are not equal-output or allocation-free performance numbers.
The indefinite-Hessian fixture is intentionally rejected here, whereas CLQR
supports returning an indefinite Newton/KKT direction.

## Testing the original author code

Use a separate checkout of `forrestlaine/parallel_lqr` at
`e24442fc853bd58fcab630c7b8b96116a53d151a` and an existing Eigen header directory.
No author source is vendored; the checkout must remain unchanged. Build-only
compatibility supplies `<functional>` and the old `eigen3/Eigen` include path.

```sh
cmake -S external_algorithms/laine_tomlin/author_audit -B "$audit_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLAINE_SOURCE_DIR="$author_checkout" -DEIGEN_SOURCE_DIR="$eigen_headers"
cmake --build "$audit_build" -j2
ctest --test-dir "$audit_build" --output-on-failure
python3 external_algorithms/laine_tomlin/author_audit/run.py \
  "$audit_build/author_audit" "$results/author.json"
```

The runner isolates crashes and timeouts and preserves all outcomes, including
numerical disagreements. Its CSV lines inside the JSON contain case, outcome,
single-call milliseconds (not benchmark medians), original feasibility,
independently audited stationarity, primal difference, CLQR KKT residual where
available, expected mathematical outcome, and optional author-multiplier
stationarity. Use `--implicit` to exercise equivalent parameterized endpoint
inputs. `--multipliers` additionally audits the author's `compute_multipliers`
and `set_lq_multipliers` outputs; the default audit and timed benchmark assess
primals, without attributing external QR multipliers to the author solver.
`-DAUDIT_EIGEN_ASSERTIONS=ON` enables Eigen assertions in the original sources.

For the identical dense cases used by the paper's comparison harness, run
`bazel run --config=fp64 //:clqr_laine_reimplementation_benchmark -- --suite dimension`.
The optional CMake harness also provides this target when configured with
`-DCLQR_COMPARE_LAINE_REIMPLEMENTATION=ON` and `EIGEN_SOURCE_DIR`; no author
checkout is required. It prints `clqr_cpu` and `laine_reimplementation`
separately. The latter has no reported-dual KKT
column; `planted_dual_stationarity_inf` checks each returned primal using the
fixture's known optimal multipliers. This untimed, linear-horizon certificate
is not a result returned by those solvers. Local runs do not replace the
paper's Kaggle measurements.
