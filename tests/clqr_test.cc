#include "clqr/clqr.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using clqr::Matrix;
using clqr::Problem;
using clqr::Solution;
using clqr::Solve;
using clqr::SolveStatus;
using clqr::Stage;
using clqr::Vector;

constexpr double kTol = 1e-7;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

void ExpectNear(double actual, double expected, double tol, const std::string& message) {
  if (std::abs(actual - expected) > tol) {
    std::cerr << "FAIL: " << message << ": actual=" << actual << " expected=" << expected
              << "\n";
    std::exit(1);
  }
}

void ExpectVectorNear(const Vector& actual, const Vector& expected, double tol,
                      const std::string& message) {
  Expect(actual.size() == expected.size(), message + " size mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    ExpectNear(actual[i], expected[i], tol, message + "[" + std::to_string(i) + "]");
  }
}

double MaxAbsVector(const Vector& x) {
  double out = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) out = std::max(out, std::abs(x[i]));
  return out;
}

double MaxAbsDifference(const Vector& a, const Vector& b) {
  Expect(a.size() == b.size(), "difference size mismatch");
  double out = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) out = std::max(out, std::abs(a[i] - b[i]));
  return out;
}

void Accumulate(Matrix matrix, Vector multiplier, Vector* into) {
  Expect(matrix.rows() == multiplier.size(), "accumulate multiplier size mismatch");
  Expect(matrix.cols() == into->size(), "accumulate target size mismatch");
  Vector term = Transpose(matrix) * multiplier;
  for (std::size_t i = 0; i < into->size(); ++i) (*into)[i] += term[i];
}

double MaxKktStationarityResidual(const Problem& p, const Solution& sol) {
  const std::size_t N = p.stages.size();
  double residual = 0.0;
  if (N == 0) {
    Vector grad = p.terminal_Q * sol.states[0] + p.terminal_q + sol.initial_multiplier;
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

    Vector u_grad = Transpose(s.M) * sol.states[i] + s.R * sol.controls[i] + s.r -
                    Transpose(s.B) * sol.dynamics_multipliers[i];
    Accumulate(s.D, sol.mixed_multipliers[i], &u_grad);
    residual = std::max(residual, MaxAbsVector(u_grad));
  }
  Vector terminal_grad =
      p.terminal_Q * sol.states[N] + p.terminal_q + sol.dynamics_multipliers[N - 1];
  Accumulate(p.terminal_E, sol.terminal_state_multiplier, &terminal_grad);
  residual = std::max(residual, MaxAbsVector(terminal_grad));
  return residual;
}

double MaxKktPrimalResidual(const Problem& p, const Solution& sol) {
  const std::size_t N = p.stages.size();
  double residual = MaxAbsDifference(sol.states[0], p.initial_state);
  for (std::size_t i = 0; i < N; ++i) {
    const Stage& s = p.stages[i];
    residual = std::max(
        residual,
        MaxAbsVector(sol.states[i + 1] - s.A * sol.states[i] - s.B * sol.controls[i] - s.c));
    if (s.C.rows() > 0) {
      residual =
          std::max(residual, MaxAbsVector(s.C * sol.states[i] + s.D * sol.controls[i] + s.d));
    }
    if (s.E.rows() > 0) {
      residual = std::max(residual, MaxAbsVector(s.E * sol.states[i] + s.e));
    }
  }
  if (p.terminal_E.rows() > 0) {
    residual = std::max(residual, MaxAbsVector(p.terminal_E * sol.states[N] + p.terminal_e));
  }
  return residual;
}

double MaxKktResidual(const Problem& p, const Solution& sol) {
  return std::max(MaxKktStationarityResidual(p, sol), MaxKktPrimalResidual(p, sol));
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

std::vector<std::size_t> ControlOffsets(const Problem& p, std::size_t state_vars) {
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

void AddConstraintRow(std::vector<Vector>* rows, std::vector<double>* rhs, Vector row,
                      double constant) {
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
      for (std::size_t c = 0; c < s.Q.cols(); ++c) H(xoff[i] + r, xoff[i] + c) += s.Q(r, c);
    }
    for (std::size_t r = 0; r < s.R.rows(); ++r) {
      h[uoff[i] + r] += s.r[r];
      for (std::size_t c = 0; c < s.R.cols(); ++c) H(uoff[i] + r, uoff[i] + c) += s.R(r, c);
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
  std::vector<double> rhs;
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
      for (std::size_t c = 0; c < s.A.cols(); ++c) row[xoff[i] + c] -= s.A(r, c);
      for (std::size_t c = 0; c < s.B.cols(); ++c) row[uoff[i] + c] -= s.B(r, c);
      AddConstraintRow(&rows, &rhs, row, -s.c[r]);
    }
    for (std::size_t r = 0; r < s.C.rows(); ++r) {
      Vector row(vars);
      for (std::size_t c = 0; c < s.C.cols(); ++c) row[xoff[i] + c] += s.C(r, c);
      for (std::size_t c = 0; c < s.D.cols(); ++c) row[uoff[i] + c] += s.D(r, c);
      AddConstraintRow(&rows, &rhs, row, s.d[r]);
    }
    for (std::size_t r = 0; r < s.E.rows(); ++r) {
      Vector row(vars);
      for (std::size_t c = 0; c < s.E.cols(); ++c) row[xoff[i] + c] += s.E(r, c);
      AddConstraintRow(&rows, &rhs, row, s.e[r]);
    }
  }
  for (std::size_t r = 0; r < p.terminal_E.rows(); ++r) {
    Vector row(vars);
    for (std::size_t c = 0; c < p.terminal_E.cols(); ++c) row[xoff[N] + c] += p.terminal_E(r, c);
    AddConstraintRow(&rows, &rhs, row, p.terminal_e[r]);
  }

  Matrix constraints_aug(rows.size(), vars + 1);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    for (std::size_t c = 0; c < vars; ++c) constraints_aug(r, c) = rows[r][c];
    constraints_aug(r, vars) = rhs[r];
  }
  clqr::RrefResult independent = clqr::Rref(constraints_aug, vars, 1e-10);
  rows.clear();
  rhs.clear();
  for (std::size_t row : independent.pivot_rows) {
    Vector constraint(vars);
    for (std::size_t c = 0; c < vars; ++c) constraint[c] = independent.matrix(row, c);
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
  Vector z = clqr::SolveLinearSystem(kkt, b, 1e-10);

  DenseKktSolution out;
  out.x.resize(N + 1);
  out.u.resize(N);
  for (std::size_t i = 0; i < N; ++i) {
    out.x[i] = Vector(p.stages[i].A.cols());
    for (std::size_t r = 0; r < out.x[i].size(); ++r) out.x[i][r] = z[xoff[i] + r];
    out.u[i] = Vector(p.stages[i].B.cols());
    for (std::size_t r = 0; r < out.u[i].size(); ++r) out.u[i][r] = z[uoff[i] + r];
  }
  out.x[N] = Vector(p.terminal_Q.rows());
  for (std::size_t r = 0; r < out.x[N].size(); ++r) out.x[N][r] = z[xoff[N] + r];
  return out;
}

double DeterministicValue(int seed, std::size_t i, std::size_t j = 0) {
  const double x = static_cast<double>(seed + 17 * static_cast<int>(i) +
                                      31 * static_cast<int>(j));
  return 0.5 * std::sin(0.37 * x) + 0.25 * std::cos(0.19 * x);
}

Vector GeneratedVector(std::size_t size, int seed, double scale = 1.0) {
  Vector out(size);
  for (std::size_t i = 0; i < size; ++i) out[i] = scale * DeterministicValue(seed, i);
  return out;
}

Matrix GeneratedMatrix(std::size_t rows, std::size_t cols, int seed, double scale = 1.0) {
  Matrix out(rows, cols);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) out(i, j) = scale * DeterministicValue(seed, i, j);
  }
  return out;
}

Matrix PositiveDefinite(std::size_t size, int seed, double diagonal) {
  Matrix g = GeneratedMatrix(size, size, seed, 0.25);
  Matrix out = Transpose(g) * g;
  for (std::size_t i = 0; i < size; ++i) out(i, i) += diagonal;
  return out;
}

double RowDot(const Matrix& a, std::size_t row, const Vector& x) {
  double out = 0.0;
  for (std::size_t col = 0; col < x.size(); ++col) out += a(row, col) * x[col];
  return out;
}

void SetMixedConstraintFromNominal(Stage* stage, const Vector& x, const Vector& u) {
  stage->d = Vector(stage->C.rows());
  for (std::size_t row = 0; row < stage->C.rows(); ++row) {
    stage->d[row] = -(RowDot(stage->C, row, x) + RowDot(stage->D, row, u));
  }
}

void SetStateConstraintFromNominal(Matrix* E, Vector* e, const Vector& x) {
  *e = Vector(E->rows());
  for (std::size_t row = 0; row < E->rows(); ++row) (*e)[row] = -RowDot(*E, row, x);
}

Problem GeneratedFeasibleProblem(int seed, std::size_t N, std::size_t n, std::size_t m,
                                 std::size_t p, ConstraintMode mode) {
  Problem problem;
  std::vector<Vector> x(N + 1);
  std::vector<Vector> u(N);
  for (std::size_t i = 0; i <= N; ++i) x[i] = GeneratedVector(n, seed + 100 + static_cast<int>(i), 0.8);
  for (std::size_t i = 0; i < N; ++i) u[i] = GeneratedVector(m, seed + 200 + static_cast<int>(i), 0.7);
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
        for (std::size_t col = 0; col < n; ++col) stage.C(1, col) = 2.0 * stage.C(0, col);
        for (std::size_t col = 0; col < m; ++col) stage.D(1, col) = 2.0 * stage.D(0, col);
      }
      SetMixedConstraintFromNominal(&stage, x[i], u[i]);
      if (mode == ConstraintMode::kRankDeficientMixed && p >= 2) stage.d[1] = 2.0 * stage.d[0];
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
    SetStateConstraintFromNominal(&problem.terminal_E, &problem.terminal_e, x[N]);
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

void CheckAgainstKkt(const Problem& p, const std::string& name) {
  Solution sol = Solve(p);
  Expect(sol.status == SolveStatus::kOptimal, name + " solver status: " + sol.message);
  DenseKktSolution kkt = SolveKkt(p);
  for (std::size_t i = 0; i < sol.states.size(); ++i) {
    ExpectVectorNear(sol.states[i], kkt.x[i], kTol, name + " x" + std::to_string(i));
  }
  for (std::size_t i = 0; i < sol.controls.size(); ++i) {
    ExpectVectorNear(sol.controls[i], kkt.u[i], kTol, name + " u" + std::to_string(i));
  }
  ExpectNear(MaxKktResidual(p, sol), 0.0, kTol, name + " full KKT residual");
}

void UnconstrainedMatchesKkt() { CheckAgainstKkt(BaseProblem(), "unconstrained"); }

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
  CheckAgainstKkt(p, "rank-deficient mixed");
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
      {"generated state-only narrow-control", 6, 3, 1, 1, ConstraintMode::kStateOnly},
      {"generated full mixed", 4, 3, 2, 2, ConstraintMode::kFullMixed},
      {"generated mixed alternating", 6, 4, 2, 2, ConstraintMode::kMixed},
      {"generated p greater than m", 4, 3, 1, 2, ConstraintMode::kFullMixed},
      {"generated rank-deficient mixed", 4, 3, 2, 2, ConstraintMode::kRankDeficientMixed},
      {"generated terminal state", 5, 3, 2, 1, ConstraintMode::kTerminalState},
      {"generated single stage", 1, 2, 2, 1, ConstraintMode::kFullMixed},
  };
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const Case& c = cases[i];
    CheckAgainstKkt(GeneratedFeasibleProblem(100 + static_cast<int>(i), c.N, c.n, c.m, c.p,
                                             c.mode),
                    c.name);
  }
}

void InfeasibleConstraintDetected() {
  Problem p = BaseProblem();
  p.stages[0].E = Matrix(2, 2, {1.0, 0.0, 1.0, 0.0});
  p.stages[0].e = Vector{-1.0, -2.0};
  Solution sol = Solve(p);
  Expect(sol.status == SolveStatus::kInfeasible, "infeasible status");
}

}  // namespace

int main() {
  UnconstrainedMatchesKkt();
  MixedConstraintMatchesKkt();
  RankDeficientMixedConstraintMatchesKkt();
  StateConstraintMatchesKkt();
  GeneratedCasesMatchKkt();
  InfeasibleConstraintDetected();
  std::cout << "all C++ tests passed\n";
  return 0;
}
