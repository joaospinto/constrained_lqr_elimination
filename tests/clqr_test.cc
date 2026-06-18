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
  ExpectNear(MaxKktStationarityResidual(p, sol), 0.0, kTol, name + " multiplier KKT residual");
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
  InfeasibleConstraintDetected();
  std::cout << "all C++ tests passed\n";
  return 0;
}
