#ifndef LAINE_TOMLIN_TEST_PROBLEM_H_
#define LAINE_TOMLIN_TEST_PROBLEM_H_
#include "solver.h"
#include <random>
namespace laine_tomlin::test {
using Eigen::Index;
inline Problem Empty(std::size_t N, Index n, Index m) {
  Problem p;
  p.Q.assign(N + 1, Matrix::Identity(n, n));
  p.q.assign(N + 1, Vector::Zero(n));
  p.initial_state = Vector::Zero(n);
  p.terminal_C.resize(0, n);
  p.terminal_d.resize(0);
  p.stages.resize(N);
  for (auto &s : p.stages) {
    s.A = Matrix::Identity(n, n);
    s.B = Matrix::Zero(n, m);
    s.c = Vector::Zero(n);
    s.R = Matrix::Identity(m, m);
    s.S = Matrix::Zero(n, m);
    s.r = Vector::Zero(m);
    s.C.resize(0, n);
    s.D.resize(0, m);
    s.d.resize(0);
  }
  return p;
}

// Independent whole-horizon oracle: assemble ALL states and controls, project
// the dense equality-constrained quadratic into the global constraint
// nullspace. No dynamic programming, no CLQR code, and no policies from the
// tested solver.
struct Dense {
  Matrix Hessian, J;
  Vector gradient, b, optimum;
  std::vector<Index> xoff, uoff;
};

inline Dense Assemble(const Problem &p) {
  Dense d;
  Index nvar = 0, rows = p.initial_state.size() + p.terminal_d.size();
  for (std::size_t t = 0; t < p.Q.size(); ++t) {
    d.xoff.push_back(nvar);
    nvar += p.Q[t].rows();
    if (t < p.stages.size()) {
      d.uoff.push_back(nvar);
      nvar += p.stages[t].B.cols();
      rows += p.stages[t].A.rows() + p.stages[t].d.size();
    }
  }
  d.Hessian = Matrix::Zero(nvar, nvar);
  d.gradient = Vector::Zero(nvar);
  d.J = Matrix::Zero(rows, nvar);
  d.b = Vector::Zero(rows);
  Index row = p.initial_state.size();
  d.J.topLeftCorner(row, row).setIdentity();
  d.b.head(row) = -p.initial_state;
  for (std::size_t t = 0; t < p.Q.size(); ++t) {
    const Index n = p.Q[t].rows(), xoff = d.xoff[t];
    d.Hessian.block(xoff, xoff, n, n) = p.Q[t];
    d.gradient.segment(xoff, n) = p.q[t];
    if (t == p.stages.size())
      break;
    const auto &s = p.stages[t];
    const Index m = s.B.cols(), uoff = d.uoff[t], next = s.A.rows();
    d.Hessian.block(uoff, uoff, m, m) = s.R;
    d.Hessian.block(xoff, uoff, n, m) = s.S;
    d.Hessian.block(uoff, xoff, m, n) = s.S.transpose();
    d.gradient.segment(uoff, m) = s.r;
    d.J.block(row, xoff, next, n) = s.A;
    d.J.block(row, uoff, next, m) = s.B;
    d.J.block(row, d.xoff[t + 1], next, next) = -Matrix::Identity(next, next);
    d.b.segment(row, next) = s.c;
    row += next;
    d.J.block(row, xoff, s.d.size(), n) = s.C;
    d.J.block(row, uoff, s.d.size(), m) = s.D;
    d.b.segment(row, s.d.size()) = s.d;
    row += s.d.size();
  }
  d.J.bottomRightCorner(p.terminal_d.size(), p.Q.back().rows()) = p.terminal_C;
  d.b.tail(p.terminal_d.size()) = p.terminal_d;
  return d;
}

inline Problem RandomProblem(unsigned seed, bool uniform = false) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> normal;
  auto random = [&](Index rows, Index cols) {
    Matrix a(rows, cols);
    for (Index j = 0; j < cols; ++j)
      for (Index i = 0; i < rows; ++i)
        a(i, j) = normal(rng) * 0.3;
    return a;
  };
  const std::size_t N = 1 + seed % 9;
  Problem p;
  p.Q.resize(N + 1);
  p.q.resize(N + 1);
  p.stages.resize(N);
  for (std::size_t t = 0; t <= N; ++t) {
    const Index n = 1 + (uniform ? seed : seed + t) % 4;
    p.Q[t] = Matrix::Identity(n, n);
    p.q[t] = random(n, 1);
  }
  p.initial_state = random(p.Q[0].rows(), 1);
  Vector x = p.initial_state;
  for (std::size_t t = 0; t < N; ++t) {
    auto &s = p.stages[t];
    const Index n = p.Q[t].rows(), next = p.Q[t + 1].rows(),
                m = uniform ? 1 + seed % 4 : (seed + t) % 5;
    s.A = random(next, n);
    s.B = random(next, m);
    s.c = random(next, 1);
    Matrix G = random(n + m, n + m);
    Matrix cost = G.transpose() * G + Matrix::Identity(n + m, n + m);
    p.Q[t] = cost.topLeftCorner(n, n);
    s.R = cost.bottomRightCorner(m, m);
    s.S = cost.topRightCorner(n, m);
    s.r = random(m, 1);
    const Vector u = random(m, 1);
    const Index rows = (seed + 2 * t) % 6;
    s.C = random(rows, n);
    s.D = random(rows, m);
    if (rows > 1) {
      s.C.row(1) = 2 * s.C.row(0);
      if (m)
        s.D.row(1) = 2 * s.D.row(0);
    }
    if (rows > 2 && seed % 2 && m)
      s.D.row(2).setZero();
    s.d = -s.C * x - s.D * u;
    // Independently rescale equivalent rows without changing the problem.
    for (Index i = 0; i < rows; ++i) {
      const double scale = std::pow(10.0, static_cast<int>((seed + i) % 9) - 4);
      s.C.row(i) *= scale;
      if (m)
        s.D.row(i) *= scale;
      s.d[i] *= scale;
    }
    x = s.A * x + s.B * u + s.c;
  }
  p.terminal_C = random(seed % 4, x.size());
  p.terminal_d = -p.terminal_C * x;
  return p;
}
} // namespace laine_tomlin::test
#endif
