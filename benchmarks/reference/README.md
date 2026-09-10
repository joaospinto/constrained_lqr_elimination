# Reference comparisons

The paper driver builds pinned, separately downloaded reference sources and
times identical FP64 problems on one machine. Only that explicit driver
fetches the reference checkouts; CMake accepts existing paths and never
fetches them. Neither library builds nor `bazel test //...` builds any
external solver. Eigen is marked `dev_dependency = True` for the optional
adapter tests; downstream Bazel consumers do not inherit it.
No third-party solver source is vendored here.

| Implementation | Mathematical scope relevant here | Timed output |
| --- | --- | --- |
| CLQR CPU | Stagewise equality constraints, including redundant rows; varying dimensions | Primal trajectory and original multipliers |
| [Vanroye et al.](https://github.com/lvanroye/generalization_riccati) | Positive-definite reduced Hessian and full-row-rank global equality Jacobian | Primal trajectory and original multipliers |
| [Yang et al., factor graph](https://github.com/ShuoYangRobotics/equality-constraint-LQR-compare) | Equality-constrained dynamic programming represented by Gaussian factors | Primal trajectory |
| [Laine's original C++ solver](https://github.com/forrestlaine/parallel_lqr) | Constrained dynamic programming; the adapter supports fixed dimensions and executes the original recursion unchanged | Primal trajectory, feedback policies, and original multipliers |

Method assumptions are not accuracy guarantees for every numerical
implementation. The Laine comparison uses the author's pinned source unchanged;
its results are audited rather than assumed accurate.

The [factor-graph paper](https://arxiv.org/abs/2011.01360) formulates
positive-definite state/control costs and
also supports cross-time constraints. Our harness uses stage-local constraints
to match CLQR's API. Its cost adapter accepts dense positive-definite costs
and semidefinite costs whose gradient lies in the Hessian's range, without
regularization. Other costs are explicitly reported as adapter-unsupported,
not as a failure of the authors' elimination routine. The CMake tests exercise
the unmodified routine on coupled semidefinite and zero-cost examples.

## Measurement contract

Every timed call refactors; these are not cached-right-hand-side solves.
Representation conversion and storage reservation are measured separately.
`setup_solve_ms` includes construction and a solve, but excludes destruction.
CLQR and Vanroye use preallocated storage. The factor-graph adapter prepares
cost square roots during setup. Thus prepared-solve and setup-plus-solve times answer different
questions, and the solvers do not all provide the same outputs.

The original Laine adapter calls `compute_multipliers()` and
`set_lq_multipliers()` inside every timed solve. Its `stationarity_inf` and
`kkt_inf` use those returned multipliers without repair or replacement;
inaccurate or nonfinite duals remain reported numerical outcomes.
Reported-dual residuals are `nan` for primal-only adapters.
For dense planted fixtures, `planted_dual_stationarity_inf` separately
checks each returned primal using known optimal fixture multipliers. This is a
linear-horizon validation certificate, not a solver-produced dual estimate.
Original feasibility, distance to the planted optimum, and objective error are
also reported. All validation is outside the timed interval.

`run_adversarial.py` runs each solver/case in a separate process, in two
opposite-order rounds. If built, the Laine and factor-graph targets are included
alongside CLQR and Vanroye. On these unplanted fixtures, primal-only outputs have
an independent sparse-QR stationarity audit (`audited_stationarity_inf`), again
outside timing and never substituted for solver output. Crashes, timeouts,
rejections, and inaccurate solutions remain visible; no failed rows are removed
from comparisons. A stationary result on an indefinite-Hessian case is not a
certificate of a minimum.

Use `scripts/paper_benchmarks.sh` for the dense comparison, or configure this
CMake project with existing reference checkouts. The Laine targets require
`LAINE_SOURCE_DIR` and `EIGEN_SOURCE_DIR`; the driver uses GTSAM's bundled Eigen
for both factor-graph and Laine adapters. Local CPU measurements must not be
inserted into tables labeled with the Kaggle/P100 host.

## Other related methods

[Callens and Decré (2026)](https://arxiv.org/abs/2608.29238) exploit zero-block
structure inside the LU decomposition used by the generalized Riccati
recursion. This is a further optimization of the Vanroye/Fatrop family, not a
separate rank-redundancy treatment. The pinned Vanroye baseline here does not
include that optimization; the harness does not measure its additional speedup.

[Jallet et al. (RSS 2024)](https://www.roboticsproceedings.org/rss20/p002.html)
provide serial and parallel factorizations in
[ALIGATOR](https://github.com/Simple-Robotics/aligator). Their proximal LQ
equations contain a positive dual regularization parameter: the equality
residual equals that parameter times a multiplier correction. A single such
solve is therefore not generally an exact solve of the unregularized problem.
The [terminal kernel](https://github.com/Simple-Robotics/aligator/blob/f0b6ceb6b8ace0cf6d9dc09ef14ba2cd7f3a7f0c/include/aligator/gar/riccati-kernel.hxx)
divides terminal constraints by this parameter; setting it to zero is not a
general exact-equality benchmark configuration. ALIGATOR is not currently
timed by this harness. Comparing a regularized solve requires reporting the
regularization and original residuals; comparing converged proximal iterations
requires timing all iterations.

[Lefebvre and Vantilborgh (2026)](https://doi.org/10.1002/oca.70069) develop
parallel conditional-value/domain propagation for equality-constrained control.
No verified author implementation is currently included in this harness.
It remains a mathematical comparison, not a measured
runtime claim. Likewise, no author-written Sideris–Rodriguez implementation is
currently included. Absence from these measurements is not evidence that an
algorithm is slower or unavailable elsewhere.
