#ifndef CLQR_BENCHMARKS_REFERENCE_EIGEN_PROBLEM_H_
#define CLQR_BENCHMARKS_REFERENCE_EIGEN_PROBLEM_H_

#include <Eigen/Core>
#include <utility>
#include <vector>
#include "clqr/clqr.h"

// Input conversion shared by external adapters and untimed audits only.
// This file contains no factorization or solve implementation.
namespace clqr::benchmark::reference {
using Eigen::Index;
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

struct Trajectory {
  std::vector<Vector> states, controls;
};

namespace conversion {
inline Matrix Convert(const clqr::Matrix &a) {
  Matrix b(a.rows(), a.cols());
  for (Index i = 0; i < b.rows(); ++i)
    for (Index j = 0; j < b.cols(); ++j)
      b(i, j) = a(i, j);
  return b;
}
inline Vector Convert(const clqr::Vector &a) {
  Vector b(a.size());
  for (Index i = 0; i < b.size(); ++i)
    b[i] = a[i];
  return b;
}
inline Problem Convert(const clqr::Problem &p) {
  Problem q;
  for (const auto &Q : p.Q)
    q.Q.push_back(Convert(Q));
  for (const auto &v : p.q)
    q.q.push_back(Convert(v));
  q.initial_state = Convert(p.initial_state);
  q.terminal_C = Convert(p.terminal_E);
  q.terminal_d = Convert(p.terminal_e);
  for (const auto &s : p.stages) {
    Stage t;
    t.A = Convert(s.A);
    t.B = Convert(s.B);
    t.c = Convert(s.c);
    t.R = Convert(s.R);
    t.S = Convert(s.M);
    t.r = Convert(s.r);
    t.C.resize(s.C.rows() + s.E.rows(), s.C.cols());
    t.D = Matrix::Zero(s.C.rows() + s.E.rows(), s.D.cols());
    t.d.resize(s.d.size() + s.e.size());
    t.C.topRows(s.C.rows()) = Convert(s.C);
    t.C.bottomRows(s.E.rows()) = Convert(s.E);
    t.D.topRows(s.D.rows()) = Convert(s.D);
    t.d.head(s.d.size()) = Convert(s.d);
    t.d.tail(s.e.size()) = Convert(s.e);
    q.stages.push_back(std::move(t));
  }
  return q;
}

}  // namespace conversion
}  // namespace clqr::benchmark::reference
#endif
