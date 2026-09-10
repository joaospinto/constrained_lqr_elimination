#ifndef EXTERNAL_ALGORITHMS_CORRECTED_LAINE_TOMLIN_SOLVER_H_
#define EXTERNAL_ALGORITHMS_CORRECTED_LAINE_TOMLIN_SOLVER_H_

// GCC can diagnose Eigen 3.4's internal triangular-product buffer during SVD
// instantiation. Scope this upstream-template exception to Eigen's headers;
// uninitialized-variable diagnostics remain enabled for our implementation.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <Eigen/Core>
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <string>
#include <vector>

// Our corrected FP64 Laine-Tomlin implementation; see README.md for provenance and
// corrections to the equations in arXiv:1807.00794v2. No CLQR solver
// dependency.
namespace corrected_laine_tomlin {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

struct Stage {
  Matrix A, B;
  Vector c;    // x[t+1] = A*x[t] + B*u[t] + c
  Matrix R, S; // stage cross cost: x[t]'*S*u[t]
  Vector r;
  Matrix C, D;
  Vector d; // C*x[t] + D*u[t] + d = 0 (includes state-only rows)
};

struct Problem {
  std::vector<Stage> stages;
  std::vector<Matrix> Q; // N+1 entries, including terminal cost
  std::vector<Vector> q;
  Matrix terminal_C;
  Vector terminal_d;
  Vector initial_state;
};

struct Options {
  // Relative SVD rank threshold; rows are equilibrated before decomposition.
  double rank_tolerance = 1e-12;
  double feasibility_tolerance = 1e-9;
};

enum class Status { kOptimal, kInfeasible, kInvalidInput, kNumericalFailure };

struct Result {
  Status status = Status::kInvalidInput;
  std::string message;
  std::vector<Matrix> K;
  std::vector<Vector> k; // u[t] = K[t]*x[t] + k[t]
  std::vector<Vector> states, controls;
  double objective = 0;
  // Largest compressed constraint-to-go row count (never exceeds state dim).
  Eigen::Index max_constraint_rows = 0;
};

// Requires positive-definite Hessians in the free-control directions. Returns
// primal trajectories and feedback, not multipliers. No regularization,
// iterative refinement, CLQR fallback, or dense whole-horizon solve.
Result Solve(const Problem &problem, const Options &options = {});

} // namespace corrected_laine_tomlin
#endif
