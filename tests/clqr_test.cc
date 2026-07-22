#include "clqr/clqr.h"
#include "../src/multiplier_recovery.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using clqr::Matrix;
using clqr::Problem;
using clqr::Scalar;
using clqr::SolutionView;
using clqr::Solve;
using clqr::SolveStatus;
using clqr::Stage;
using clqr::Vector;
using clqr::VectorView;
using clqr::Workspace;

#ifdef CLQR_USE_FLOAT
constexpr Scalar kTol = 2e-4f;
constexpr Scalar kLinearSolveTolerance = 1e-7f;
constexpr Scalar kTrajectoryMultiplierTolerance = 1e-2f;
#else
constexpr Scalar kTol = 1e-7;
constexpr Scalar kLinearSolveTolerance = 1e-10;
constexpr Scalar kTrajectoryMultiplierTolerance = kTol;
#endif

struct Solution {
  SolveStatus status = SolveStatus::kInvalidInput;
  std::string message;
  std::vector<Vector> states;
  std::vector<Vector> controls;
  Vector initial_multiplier;
  std::vector<Vector> dynamics_multipliers;
  std::vector<Vector> mixed_multipliers;
  std::vector<Vector> state_multipliers;
  Vector terminal_state_multiplier;
  bool newton_kkt_singular = false;
  bool newton_kkt_wrong_inertia = false;
  std::string newton_kkt_diagnostic;
  Scalar objective = 0.0;
};

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

void ExpectNear(Scalar actual, Scalar expected, Scalar tol,
                const std::string& message) {
  if (std::abs(actual - expected) > tol) {
    std::cerr << "FAIL: " << message << ": actual=" << actual
              << " expected=" << expected << "\n";
    std::exit(1);
  }
}

void ExpectVectorNear(const Vector& actual, const Vector& expected, Scalar tol,
                      const std::string& message) {
  Expect(actual.size() == expected.size(), message + " size mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    ExpectNear(actual[i], expected[i], tol,
               message + "[" + std::to_string(i) + "]");
  }
}

void ExpectVectorViewNear(const VectorView& actual, const Vector& expected,
                          Scalar tol, const std::string& message) {
  Expect(actual.size == expected.size(), message + " size mismatch");
  for (std::size_t i = 0; i < actual.size; ++i) {
    ExpectNear(actual[i], expected[i], tol,
               message + "[" + std::to_string(i) + "]");
  }
}

Vector CopyVectorView(const VectorView& view) {
  Vector out(view.size);
  for (std::size_t i = 0; i < view.size; ++i) out[i] = view[i];
  return out;
}

Solution CopySolutionView(const SolutionView& view) {
  Solution out;
  out.status = view.status;
  out.message = view.message;
  out.states.resize(view.state_count);
  for (std::size_t i = 0; i < view.state_count; ++i)
    out.states[i] = CopyVectorView(view.states[i]);
  out.controls.resize(view.control_count);
  for (std::size_t i = 0; i < view.control_count; ++i) {
    out.controls[i] = CopyVectorView(view.controls[i]);
  }
  out.initial_multiplier = CopyVectorView(view.initial_multiplier);
  out.dynamics_multipliers.resize(view.dynamics_multiplier_count);
  for (std::size_t i = 0; i < view.dynamics_multiplier_count; ++i) {
    out.dynamics_multipliers[i] = CopyVectorView(view.dynamics_multipliers[i]);
  }
  out.mixed_multipliers.resize(view.mixed_multiplier_count);
  for (std::size_t i = 0; i < view.mixed_multiplier_count; ++i) {
    out.mixed_multipliers[i] = CopyVectorView(view.mixed_multipliers[i]);
  }
  out.state_multipliers.resize(view.state_multiplier_count);
  for (std::size_t i = 0; i < view.state_multiplier_count; ++i) {
    out.state_multipliers[i] = CopyVectorView(view.state_multipliers[i]);
  }
  out.terminal_state_multiplier =
      CopyVectorView(view.terminal_state_multiplier);
  out.objective = view.objective;
  out.newton_kkt_singular = view.newton_kkt_singular;
  out.newton_kkt_wrong_inertia = view.newton_kkt_wrong_inertia;
  out.newton_kkt_diagnostic = view.newton_kkt_diagnostic;
  return out;
}

Solution SolveWithWorkspace(
    const Problem& p,
    const clqr::SolveOptions& options = clqr::SolveOptions{}) {
  Workspace workspace;
  workspace.Reserve(p, options);
  return CopySolutionView(Solve(p, workspace, options));
}

void ExpectDiagnostics(const Solution& sol, bool singular, bool wrong_inertia,
                       const std::string& message) {
  Expect(sol.newton_kkt_singular == singular, message + " singular diagnostic");
  Expect(sol.newton_kkt_wrong_inertia == wrong_inertia,
         message + " inertia diagnostic");
}

Scalar MaxAbsVector(const Vector& x) {
  Scalar out = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i)
    out = std::max(out, std::abs(x[i]));
  return out;
}

Scalar MaxAbsDifference(const Vector& a, const Vector& b) {
  Expect(a.size() == b.size(), "difference size mismatch");
  Scalar out = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    out = std::max(out, std::abs(a[i] - b[i]));
  return out;
}

void Accumulate(Matrix matrix, Vector multiplier, Vector* into) {
  Expect(matrix.rows() == multiplier.size(),
         "accumulate multiplier size mismatch");
  Expect(matrix.cols() == into->size(), "accumulate target size mismatch");
  Vector term = Transpose(matrix) * multiplier;
  for (std::size_t i = 0; i < into->size(); ++i) (*into)[i] += term[i];
}

Scalar MaxKktStationarityResidual(const Problem& p, const Solution& sol) {
  const std::size_t N = p.stages.size();
  Scalar residual = 0.0;
  if (N == 0) {
    Vector grad =
        p.terminal_Q * sol.states[0] + p.terminal_q + sol.initial_multiplier;
    Accumulate(p.terminal_E, sol.terminal_state_multiplier, &grad);
    return MaxAbsVector(grad);
  }
  for (std::size_t i = 0; i < N; ++i) {
    const Stage& s = p.stages[i];
    Vector x_grad = s.Q * sol.states[i] + s.M * sol.controls[i] + s.q -
                    Transpose(s.A) * sol.dynamics_multipliers[i];
    if (i == 0) {
      x_grad = x_grad + sol.initial_multiplier;
    } else {
      x_grad = x_grad + sol.dynamics_multipliers[i - 1];
    }
    Accumulate(s.C, sol.mixed_multipliers[i], &x_grad);
    Accumulate(s.E, sol.state_multipliers[i], &x_grad);
    residual = std::max(residual, MaxAbsVector(x_grad));

    Vector u_grad = Transpose(s.M) * sol.states[i] + s.R * sol.controls[i] +
                    s.r - Transpose(s.B) * sol.dynamics_multipliers[i];
    Accumulate(s.D, sol.mixed_multipliers[i], &u_grad);
    residual = std::max(residual, MaxAbsVector(u_grad));
  }
  Vector terminal_grad = p.terminal_Q * sol.states[N] + p.terminal_q +
                         sol.dynamics_multipliers[N - 1];
  Accumulate(p.terminal_E, sol.terminal_state_multiplier, &terminal_grad);
  residual = std::max(residual, MaxAbsVector(terminal_grad));
  return residual;
}

Scalar MaxKktPrimalResidual(const Problem& p, const Solution& sol) {
  const std::size_t N = p.stages.size();
  Scalar residual = MaxAbsDifference(sol.states[0], p.initial_state);
  for (std::size_t i = 0; i < N; ++i) {
    const Stage& s = p.stages[i];
    residual = std::max(residual,
                        MaxAbsVector(sol.states[i + 1] - s.A * sol.states[i] -
                                     s.B * sol.controls[i] - s.c));
    if (s.C.rows() > 0) {
      residual = std::max(residual, MaxAbsVector(s.C * sol.states[i] +
                                                 s.D * sol.controls[i] + s.d));
    }
    if (s.E.rows() > 0) {
      residual = std::max(residual, MaxAbsVector(s.E * sol.states[i] + s.e));
    }
  }
  if (p.terminal_E.rows() > 0) {
    residual = std::max(
        residual, MaxAbsVector(p.terminal_E * sol.states[N] + p.terminal_e));
  }
  return residual;
}

Scalar MaxKktResidual(const Problem& p, const Solution& sol) {
  return std::max(MaxKktStationarityResidual(p, sol),
                  MaxKktPrimalResidual(p, sol));
}

std::vector<std::size_t> StateOffsets(const Problem& p) {
  std::vector<std::size_t> out(p.stages.size() + 1);
  std::size_t offset = 0;
  for (std::size_t i = 0; i < p.stages.size(); ++i) {
    out[i] = offset;
    offset += p.stages[i].A.cols();
  }
  out[p.stages.size()] = offset;
  return out;
}

std::vector<std::size_t> ControlOffsets(const Problem& p,
                                        std::size_t state_vars) {
  std::vector<std::size_t> out(p.stages.size());
  std::size_t offset = state_vars;
  for (std::size_t i = 0; i < p.stages.size(); ++i) {
    out[i] = offset;
    offset += p.stages[i].B.cols();
  }
  return out;
}

struct DenseKktSolution {
  std::vector<Vector> x;
  std::vector<Vector> u;
};

enum class ConstraintMode {
  kUnconstrained,
  kStateOnly,
  kFullMixed,
  kMixed,
  kRankDeficientMixed,
  kTerminalState,
};

void AddConstraintRow(std::vector<Vector>* rows, std::vector<Scalar>* rhs,
                      Vector row, Scalar constant) {
  rows->push_back(row);
  rhs->push_back(-constant);
}

DenseKktSolution SolveKkt(const Problem& p) {
  const std::size_t N = p.stages.size();
  std::vector<std::size_t> xoff = StateOffsets(p);
  std::size_t state_vars = xoff[N] + p.terminal_Q.rows();
  std::vector<std::size_t> uoff = ControlOffsets(p, state_vars);
  std::size_t vars = state_vars;
  for (const Stage& s : p.stages) vars += s.B.cols();

  Matrix H(vars, vars);
  Vector h(vars);
  for (std::size_t i = 0; i < N; ++i) {
    const Stage& s = p.stages[i];
    for (std::size_t r = 0; r < s.Q.rows(); ++r) {
      h[xoff[i] + r] += s.q[r];
      for (std::size_t c = 0; c < s.Q.cols(); ++c)
        H(xoff[i] + r, xoff[i] + c) += s.Q(r, c);
    }
    for (std::size_t r = 0; r < s.R.rows(); ++r) {
      h[uoff[i] + r] += s.r[r];
      for (std::size_t c = 0; c < s.R.cols(); ++c)
        H(uoff[i] + r, uoff[i] + c) += s.R(r, c);
    }
    for (std::size_t r = 0; r < s.M.rows(); ++r) {
      for (std::size_t c = 0; c < s.M.cols(); ++c) {
        H(xoff[i] + r, uoff[i] + c) += s.M(r, c);
        H(uoff[i] + c, xoff[i] + r) += s.M(r, c);
      }
    }
  }
  for (std::size_t r = 0; r < p.terminal_Q.rows(); ++r) {
    h[xoff[N] + r] += p.terminal_q[r];
    for (std::size_t c = 0; c < p.terminal_Q.cols(); ++c) {
      H(xoff[N] + r, xoff[N] + c) += p.terminal_Q(r, c);
    }
  }

  std::vector<Vector> rows;
  std::vector<Scalar> rhs;
  for (std::size_t r = 0; r < p.initial_state.size(); ++r) {
    Vector row(vars);
    row[xoff[0] + r] = 1.0;
    AddConstraintRow(&rows, &rhs, row, -p.initial_state[r]);
  }
  for (std::size_t i = 0; i < N; ++i) {
    const Stage& s = p.stages[i];
    for (std::size_t r = 0; r < s.A.rows(); ++r) {
      Vector row(vars);
      row[xoff[i + 1] + r] = 1.0;
      for (std::size_t c = 0; c < s.A.cols(); ++c)
        row[xoff[i] + c] -= s.A(r, c);
      for (std::size_t c = 0; c < s.B.cols(); ++c)
        row[uoff[i] + c] -= s.B(r, c);
      AddConstraintRow(&rows, &rhs, row, -s.c[r]);
    }
    for (std::size_t r = 0; r < s.C.rows(); ++r) {
      Vector row(vars);
      for (std::size_t c = 0; c < s.C.cols(); ++c)
        row[xoff[i] + c] += s.C(r, c);
      for (std::size_t c = 0; c < s.D.cols(); ++c)
        row[uoff[i] + c] += s.D(r, c);
      AddConstraintRow(&rows, &rhs, row, s.d[r]);
    }
    for (std::size_t r = 0; r < s.E.rows(); ++r) {
      Vector row(vars);
      for (std::size_t c = 0; c < s.E.cols(); ++c)
        row[xoff[i] + c] += s.E(r, c);
      AddConstraintRow(&rows, &rhs, row, s.e[r]);
    }
  }
  for (std::size_t r = 0; r < p.terminal_E.rows(); ++r) {
    Vector row(vars);
    for (std::size_t c = 0; c < p.terminal_E.cols(); ++c)
      row[xoff[N] + c] += p.terminal_E(r, c);
    AddConstraintRow(&rows, &rhs, row, p.terminal_e[r]);
  }

  Matrix constraints_aug(rows.size(), vars + 1);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    for (std::size_t c = 0; c < vars; ++c) constraints_aug(r, c) = rows[r][c];
    constraints_aug(r, vars) = rhs[r];
  }
  clqr::RrefResult independent =
      clqr::Rref(constraints_aug, vars, kLinearSolveTolerance);
  rows.clear();
  rhs.clear();
  for (std::size_t row : independent.pivot_rows) {
    Vector constraint(vars);
    for (std::size_t c = 0; c < vars; ++c)
      constraint[c] = independent.matrix(row, c);
    rows.push_back(constraint);
    rhs.push_back(independent.matrix(row, vars));
  }

  const std::size_t constraints = rows.size();
  Matrix kkt(vars + constraints, vars + constraints);
  Vector b(vars + constraints);
  for (std::size_t i = 0; i < vars; ++i) {
    b[i] = -h[i];
    for (std::size_t j = 0; j < vars; ++j) kkt(i, j) = H(i, j);
  }
  for (std::size_t r = 0; r < constraints; ++r) {
    b[vars + r] = rhs[r];
    for (std::size_t c = 0; c < vars; ++c) {
      kkt(vars + r, c) = rows[r][c];
      kkt(c, vars + r) = rows[r][c];
    }
  }
  Vector z = clqr::SolveLinearSystem(kkt, b, kLinearSolveTolerance);

  DenseKktSolution out;
  out.x.resize(N + 1);
  out.u.resize(N);
  for (std::size_t i = 0; i < N; ++i) {
    out.x[i] = Vector(p.stages[i].A.cols());
    for (std::size_t r = 0; r < out.x[i].size(); ++r)
      out.x[i][r] = z[xoff[i] + r];
    out.u[i] = Vector(p.stages[i].B.cols());
    for (std::size_t r = 0; r < out.u[i].size(); ++r)
      out.u[i][r] = z[uoff[i] + r];
  }
  out.x[N] = Vector(p.terminal_Q.rows());
  for (std::size_t r = 0; r < out.x[N].size(); ++r)
    out.x[N][r] = z[xoff[N] + r];
  return out;
}

Scalar DeterministicValue(int seed, std::size_t i, std::size_t j = 0) {
  const Scalar x = static_cast<Scalar>(seed + 17 * static_cast<int>(i) +
                                       31 * static_cast<int>(j));
  return 0.5 * std::sin(0.37 * x) + 0.25 * std::cos(0.19 * x);
}

Vector GeneratedVector(std::size_t size, int seed, Scalar scale = 1.0) {
  Vector out(size);
  for (std::size_t i = 0; i < size; ++i)
    out[i] = scale * DeterministicValue(seed, i);
  return out;
}

Matrix GeneratedMatrix(std::size_t rows, std::size_t cols, int seed,
                       Scalar scale = 1.0) {
  Matrix out(rows, cols);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j)
      out(i, j) = scale * DeterministicValue(seed, i, j);
  }
  return out;
}

Matrix PositiveDefinite(std::size_t size, int seed, Scalar diagonal) {
  Matrix g = GeneratedMatrix(size, size, seed, 0.25);
  Matrix out = Transpose(g) * g;
  for (std::size_t i = 0; i < size; ++i) out(i, i) += diagonal;
  return out;
}

Scalar RowDot(const Matrix& a, std::size_t row, const Vector& x) {
  Scalar out = 0.0;
  for (std::size_t col = 0; col < x.size(); ++col) out += a(row, col) * x[col];
  return out;
}

void SetMixedConstraintFromNominal(Stage* stage, const Vector& x,
                                   const Vector& u) {
  stage->d = Vector(stage->C.rows());
  for (std::size_t row = 0; row < stage->C.rows(); ++row) {
    stage->d[row] = -(RowDot(stage->C, row, x) + RowDot(stage->D, row, u));
  }
}

void SetStateConstraintFromNominal(Matrix* E, Vector* e, const Vector& x) {
  *e = Vector(E->rows());
  for (std::size_t row = 0; row < E->rows(); ++row)
    (*e)[row] = -RowDot(*E, row, x);
}

Problem GeneratedFeasibleProblem(int seed, std::size_t N, std::size_t n,
                                 std::size_t m, std::size_t p,
                                 ConstraintMode mode) {
  Problem problem;
  std::vector<Vector> x(N + 1);
  std::vector<Vector> u(N);
  for (std::size_t i = 0; i <= N; ++i)
    x[i] = GeneratedVector(n, seed + 100 + static_cast<int>(i), 0.8);
  for (std::size_t i = 0; i < N; ++i)
    u[i] = GeneratedVector(m, seed + 200 + static_cast<int>(i), 0.7);
  problem.initial_state = x[0];
  problem.stages.resize(N);
  for (std::size_t i = 0; i < N; ++i) {
    Stage& stage = problem.stages[i];
    stage.A = GeneratedMatrix(n, n, seed + 10 * static_cast<int>(i), 0.2);
    for (std::size_t row = 0; row < n; ++row) stage.A(row, row) += 0.8;
    stage.B = GeneratedMatrix(n, m, seed + 300 + 10 * static_cast<int>(i), 0.3);
    stage.c = x[i + 1] - stage.A * x[i] - stage.B * u[i];
    stage.Q = PositiveDefinite(n, seed + 400 + static_cast<int>(i), 1.0);
    stage.R = PositiveDefinite(m, seed + 500 + static_cast<int>(i), 1.5);
    stage.M = GeneratedMatrix(n, m, seed + 600 + static_cast<int>(i), 0.05);
    stage.q = GeneratedVector(n, seed + 700 + static_cast<int>(i), 0.3);
    stage.r = GeneratedVector(m, seed + 800 + static_cast<int>(i), 0.3);
    stage.C = Matrix(0, n);
    stage.D = Matrix(0, m);
    stage.d = Vector(0);
    stage.E = Matrix(0, n);
    stage.e = Vector(0);

    if (p > 0 && (mode == ConstraintMode::kFullMixed ||
                  mode == ConstraintMode::kRankDeficientMixed ||
                  (mode == ConstraintMode::kMixed && i % 2 == 0))) {
      stage.C = GeneratedMatrix(p, n, seed + 900 + static_cast<int>(i), 0.5);
      stage.D = GeneratedMatrix(p, m, seed + 1000 + static_cast<int>(i), 0.5);
      if (mode == ConstraintMode::kRankDeficientMixed && p >= 2) {
        for (std::size_t col = 0; col < n; ++col)
          stage.C(1, col) = 2.0 * stage.C(0, col);
        for (std::size_t col = 0; col < m; ++col)
          stage.D(1, col) = 2.0 * stage.D(0, col);
      }
      SetMixedConstraintFromNominal(&stage, x[i], u[i]);
      if (mode == ConstraintMode::kRankDeficientMixed && p >= 2)
        stage.d[1] = 2.0 * stage.d[0];
    } else if (p > 0 && (mode == ConstraintMode::kStateOnly ||
                         (mode == ConstraintMode::kMixed && i % 2 == 1))) {
      stage.E = GeneratedMatrix(p, n, seed + 1100 + static_cast<int>(i), 0.5);
      SetStateConstraintFromNominal(&stage.E, &stage.e, x[i]);
    }
  }
  problem.terminal_Q = PositiveDefinite(n, seed + 1200, 1.5);
  problem.terminal_q = GeneratedVector(n, seed + 1300, 0.3);
  problem.terminal_E = Matrix(0, n);
  problem.terminal_e = Vector(0);
  if (p > 0 && mode == ConstraintMode::kTerminalState) {
    problem.terminal_E = GeneratedMatrix(p, n, seed + 1400, 0.5);
    SetStateConstraintFromNominal(&problem.terminal_E, &problem.terminal_e,
                                  x[N]);
  }
  return problem;
}

Problem BaseProblem() {
  Problem p;
  p.initial_state = Vector{1.0, -0.5};
  p.stages.resize(2);

  p.stages[0].A = Matrix(2, 2, {1.0, 0.2, 0.0, 1.0});
  p.stages[0].B = Matrix(2, 2, {0.0, 0.1, 1.0, 0.2});
  p.stages[0].c = Vector{0.1, -0.2};
  p.stages[0].Q = Matrix(2, 2, {2.0, 0.1, 0.1, 1.0});
  p.stages[0].R = Matrix(2, 2, {3.0, 0.2, 0.2, 2.0});
  p.stages[0].M = Matrix(2, 2, {0.1, -0.2, 0.0, 0.15});
  p.stages[0].q = Vector{0.2, -0.1};
  p.stages[0].r = Vector{0.3, -0.4};
  p.stages[0].C = Matrix(0, 2);
  p.stages[0].D = Matrix(0, 2);
  p.stages[0].d = Vector(0);
  p.stages[0].E = Matrix(0, 2);
  p.stages[0].e = Vector(0);

  p.stages[1].A = Matrix(2, 2, {0.9, -0.1, 0.3, 1.1});
  p.stages[1].B = Matrix(2, 1, {0.2, 0.7});
  p.stages[1].c = Vector{-0.1, 0.05};
  p.stages[1].Q = Matrix(2, 2, {1.5, 0.0, 0.0, 1.2});
  p.stages[1].R = Matrix(1, 1, {2.5});
  p.stages[1].M = Matrix(2, 1, {0.05, -0.1});
  p.stages[1].q = Vector{-0.3, 0.2};
  p.stages[1].r = Vector{0.25};
  p.stages[1].C = Matrix(0, 2);
  p.stages[1].D = Matrix(0, 1);
  p.stages[1].d = Vector(0);
  p.stages[1].E = Matrix(0, 2);
  p.stages[1].e = Vector(0);

  p.terminal_Q = Matrix(2, 2, {4.0, 0.2, 0.2, 3.0});
  p.terminal_q = Vector{0.1, -0.2};
  p.terminal_E = Matrix(0, 2);
  p.terminal_e = Vector(0);
  return p;
}

void CheckAgainstKkt(const Problem& p, const std::string& name,
                     bool expect_singular = false,
                     bool expect_wrong_inertia = false) {
  Solution sol = SolveWithWorkspace(p);
  Expect(sol.status == SolveStatus::kOptimal,
         name + " solver status: " + sol.message);
  ExpectDiagnostics(sol, expect_singular, expect_wrong_inertia, name);
  DenseKktSolution kkt = SolveKkt(p);
  for (std::size_t i = 0; i < sol.states.size(); ++i) {
    ExpectVectorNear(sol.states[i], kkt.x[i], kTol,
                     name + " x" + std::to_string(i));
  }
  for (std::size_t i = 0; i < sol.controls.size(); ++i) {
    ExpectVectorNear(sol.controls[i], kkt.u[i], kTol,
                     name + " u" + std::to_string(i));
  }
  ExpectNear(MaxKktResidual(p, sol), 0.0, kTol, name + " full KKT residual");
}

void UnconstrainedMatchesKkt() {
  CheckAgainstKkt(BaseProblem(), "unconstrained");
}

void WorkspaceUnconstrainedMatchesKkt() {
  Problem p =
      GeneratedFeasibleProblem(321, 4, 3, 2, 0, ConstraintMode::kUnconstrained);
  constexpr std::size_t kBytes = Workspace::RequiredBytesUniform(4, 3, 2);
  static_assert(kBytes > 0, "workspace size must be positive");
  Expect(Workspace::RequiredBytes(p) == kBytes,
         "constexpr workspace byte count");

  alignas(std::max_align_t) std::array<unsigned char, kBytes> memory{};
  Workspace workspace(memory.data(), memory.size());
  SolutionView view = Solve(p, workspace);
  Solution copied = CopySolutionView(view);
  DenseKktSolution kkt = SolveKkt(p);
  Expect(view.status == SolveStatus::kOptimal,
         std::string("workspace status: ") + view.message);
  Expect(view.state_count == kkt.x.size(), "workspace state count");
  Expect(view.control_count == kkt.u.size(), "workspace control count");
  for (std::size_t i = 0; i < view.state_count; ++i) {
    ExpectVectorViewNear(view.states[i], kkt.x[i], kTol,
                         "workspace state " + std::to_string(i));
  }
  for (std::size_t i = 0; i < view.control_count; ++i) {
    ExpectVectorViewNear(view.controls[i], kkt.u[i], kTol,
                         "workspace control " + std::to_string(i));
  }
  ExpectNear(MaxKktResidual(p, copied), 0.0, kTol,
             "workspace full KKT residual");

  std::vector<unsigned char> too_small(Workspace::RequiredBytes(p) - 1);
  Workspace small_workspace(too_small.data(), too_small.size());
  SolutionView small = Solve(p, small_workspace);
  Expect(small.status == SolveStatus::kInvalidInput,
         "undersized workspace status");
}

void WorkspaceConstrainedMatchesKkt() {
  const std::vector<Problem> problems = {
      GeneratedFeasibleProblem(410, 4, 3, 2, 1, ConstraintMode::kStateOnly),
      GeneratedFeasibleProblem(411, 4, 3, 2, 2, ConstraintMode::kFullMixed),
      GeneratedFeasibleProblem(412, 5, 3, 2, 1, ConstraintMode::kTerminalState),
  };
  for (std::size_t case_index = 0; case_index < problems.size(); ++case_index) {
    const Problem& p = problems[case_index];
    Workspace workspace;
    workspace.Reserve(p);
    SolutionView view = Solve(p, workspace);
    Solution copied = CopySolutionView(view);
    const std::string name =
        "workspace constrained " + std::to_string(case_index);
    Expect(view.status == SolveStatus::kOptimal,
           name + " status: " + std::string(view.message));
    DenseKktSolution kkt = SolveKkt(p);
    for (std::size_t i = 0; i < view.state_count; ++i) {
      ExpectVectorViewNear(view.states[i], kkt.x[i], kTol,
                           name + " state " + std::to_string(i));
    }
    for (std::size_t i = 0; i < view.control_count; ++i) {
      ExpectVectorViewNear(view.controls[i], kkt.u[i], kTol,
                           name + " control " + std::to_string(i));
    }
    ExpectNear(MaxKktResidual(p, copied), 0.0, kTol,
               name + " full KKT residual");
  }
}

void MixedConstraintMatchesKkt() {
  Problem p = BaseProblem();
  p.stages[0].C = Matrix(1, 2, {1.0, -2.0});
  p.stages[0].D = Matrix(1, 2, {0.0, 1.0});
  p.stages[0].d = Vector{0.4};
  CheckAgainstKkt(p, "mixed");
}

void RankDeficientMixedConstraintMatchesKkt() {
  Problem p = BaseProblem();
  p.stages[0].C = Matrix(2, 2, {1.0, 0.0, 2.0, 0.0});
  p.stages[0].D = Matrix(2, 2, {1.0, 0.0, 2.0, 0.0});
  p.stages[0].d = Vector{-0.2, -0.4};
  CheckAgainstKkt(p, "rank-deficient mixed", true);
}

void StateConstraintMatchesKkt() {
  Problem p = BaseProblem();
  p.terminal_E = Matrix(1, 2, {1.0, -0.5});
  p.terminal_e = Vector{-0.2};
  CheckAgainstKkt(p, "terminal state");
}

void GeneratedCasesMatchKkt() {
  struct Case {
    const char* name;
    std::size_t N;
    std::size_t n;
    std::size_t m;
    std::size_t p;
    ConstraintMode mode;
  };
  const std::vector<Case> cases = {
      {"generated unconstrained", 4, 3, 2, 0, ConstraintMode::kUnconstrained},
      {"generated state-only", 4, 3, 2, 1, ConstraintMode::kStateOnly},
      {"generated state-only narrow-control", 6, 3, 1, 1,
       ConstraintMode::kStateOnly},
      {"generated full mixed", 4, 3, 2, 2, ConstraintMode::kFullMixed},
      {"generated mixed alternating", 6, 4, 2, 2, ConstraintMode::kMixed},
      {"generated p greater than m", 4, 3, 1, 2, ConstraintMode::kFullMixed},
      {"generated rank-deficient mixed", 4, 3, 2, 2,
       ConstraintMode::kRankDeficientMixed},
      {"generated terminal state", 5, 3, 2, 1, ConstraintMode::kTerminalState},
      {"generated single stage", 1, 2, 2, 1, ConstraintMode::kFullMixed},
  };
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const Case& c = cases[i];
    const bool expect_singular =
        c.mode == ConstraintMode::kStateOnly ||
        c.mode == ConstraintMode::kMixed ||
        c.mode == ConstraintMode::kRankDeficientMixed ||
        (c.mode == ConstraintMode::kFullMixed && c.p > c.m);
    CheckAgainstKkt(GeneratedFeasibleProblem(100 + static_cast<int>(i), c.N,
                                             c.n, c.m, c.p, c.mode),
                    c.name, expect_singular);
  }
}

void TrajectoryMultiplierRecoveryMatchesKkt() {
  Problem problem =
      GeneratedFeasibleProblem(519, 64, 4, 2, 1, ConstraintMode::kStateOnly);
  Solution solution = SolveWithWorkspace(problem);
  Expect(solution.status == SolveStatus::kOptimal,
         "trajectory multiplier source solve status");
  clqr::internal::MultiplierRecovery recovery =
      clqr::internal::RecoverMultipliersForTrajectory(
          problem, solution.states, solution.controls, kLinearSolveTolerance);
  Expect(recovery.success,
         "trajectory multiplier recovery: " + recovery.message);
  solution.initial_multiplier = std::move(recovery.initial);
  solution.dynamics_multipliers = std::move(recovery.dynamics);
  solution.mixed_multipliers = std::move(recovery.mixed);
  solution.state_multipliers = std::move(recovery.state);
  solution.terminal_state_multiplier = std::move(recovery.terminal);
  ExpectNear(MaxKktResidual(problem, solution), 0.0,
             kTrajectoryMultiplierTolerance,
             "trajectory multiplier recovery KKT residual");
}

void WrongInertiaReportedWithCandidate() {
  Problem p;
  p.initial_state = Vector{0.0};
  p.stages.resize(1);
  p.stages[0].A = Matrix(1, 1, {1.0});
  p.stages[0].B = Matrix(1, 1, {1.0});
  p.stages[0].c = Vector{0.0};
  p.stages[0].Q = Matrix(1, 1, {0.0});
  p.stages[0].R = Matrix(1, 1, {-1.0});
  p.stages[0].M = Matrix(1, 1, {0.0});
  p.stages[0].q = Vector{0.0};
  p.stages[0].r = Vector{-2.0};
  p.stages[0].C = Matrix(0, 1);
  p.stages[0].D = Matrix(0, 1);
  p.stages[0].d = Vector(0);
  p.stages[0].E = Matrix(0, 1);
  p.stages[0].e = Vector(0);
  p.terminal_Q = Matrix(1, 1, {0.0});
  p.terminal_q = Vector{0.0};
  p.terminal_E = Matrix(0, 1);
  p.terminal_e = Vector(0);

  CheckAgainstKkt(p, "wrong inertia", false, true);
}

void SingularReducedHessianReported() {
  Problem p;
  p.initial_state = Vector{0.0};
  p.stages.resize(1);
  p.stages[0].A = Matrix(1, 1, {1.0});
  p.stages[0].B = Matrix(1, 1, {1.0});
  p.stages[0].c = Vector{0.0};
  p.stages[0].Q = Matrix(1, 1, {0.0});
  p.stages[0].R = Matrix(1, 1, {0.0});
  p.stages[0].M = Matrix(1, 1, {0.0});
  p.stages[0].q = Vector{0.0};
  p.stages[0].r = Vector{0.0};
  p.stages[0].C = Matrix(0, 1);
  p.stages[0].D = Matrix(0, 1);
  p.stages[0].d = Vector(0);
  p.stages[0].E = Matrix(0, 1);
  p.stages[0].e = Vector(0);
  p.terminal_Q = Matrix(1, 1, {0.0});
  p.terminal_q = Vector{0.0};
  p.terminal_E = Matrix(0, 1);
  p.terminal_e = Vector(0);

  Solution sol = SolveWithWorkspace(p);
  Expect(sol.status == SolveStatus::kNumericalFailure,
         "singular reduced Hessian status");
  ExpectDiagnostics(sol, true, false, "singular reduced Hessian");
}

void InfeasibleConstraintDetected() {
  Problem p = BaseProblem();
  p.stages[0].E = Matrix(2, 2, {1.0, 0.0, 1.0, 0.0});
  p.stages[0].e = Vector{-1.0, -2.0};
  Solution sol = SolveWithWorkspace(p);
  Expect(sol.status == SolveStatus::kInfeasible, "infeasible status");
}

}  // namespace

int main() {
  UnconstrainedMatchesKkt();
  WorkspaceUnconstrainedMatchesKkt();
  WorkspaceConstrainedMatchesKkt();
  MixedConstraintMatchesKkt();
  RankDeficientMixedConstraintMatchesKkt();
  StateConstraintMatchesKkt();
  GeneratedCasesMatchKkt();
  TrajectoryMultiplierRecoveryMatchesKkt();
  WrongInertiaReportedWithCandidate();
  SingularReducedHessianReported();
  InfeasibleConstraintDetected();
  std::cout << "all C++ tests passed\n";
  return 0;
}
