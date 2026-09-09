#ifndef LAINE_TOMLIN_PROBLEM_CONVERSION_H_
#define LAINE_TOMLIN_PROBLEM_CONVERSION_H_
#include "clqr/clqr.h"
#include "solver.h"
namespace laine_tomlin::conversion {
using Eigen::Index;
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
inline laine_tomlin::Problem Convert(const clqr::Problem &p) {
  laine_tomlin::Problem q;
  for (const auto &Q : p.Q)
    q.Q.push_back(Convert(Q));
  for (const auto &v : p.q)
    q.q.push_back(Convert(v));
  q.initial_state = Convert(p.initial_state);
  q.terminal_C = Convert(p.terminal_E);
  q.terminal_d = Convert(p.terminal_e);
  for (const auto &s : p.stages) {
    laine_tomlin::Stage t;
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

// For independent random-fixture cross-checks only. Coefficients are copied;
// zero-control rows may remain in C/D rather than being split into E.
inline clqr::Problem ToClqr(const laine_tomlin::Problem &p) {
  auto matrix = [](const Matrix &a) {
    clqr::Matrix b(a.rows(), a.cols());
    for (Index i = 0; i < a.rows(); ++i)
      for (Index j = 0; j < a.cols(); ++j)
        b(i, j) = a(i, j);
    return b;
  };
  auto vector = [](const Vector &a) {
    clqr::Vector b(a.size());
    for (Index i = 0; i < a.size(); ++i)
      b[i] = a[i];
    return b;
  };
  clqr::Problem q;
  for (const auto &Q : p.Q)
    q.Q.push_back(matrix(Q));
  for (const auto &v : p.q)
    q.q.push_back(vector(v));
  q.initial_state = vector(p.initial_state);
  q.terminal_E = matrix(p.terminal_C);
  q.terminal_e = vector(p.terminal_d);
  for (const auto &s : p.stages) {
    clqr::Stage t;
    t.A = matrix(s.A);
    t.B = matrix(s.B);
    t.c = vector(s.c);
    t.R = matrix(s.R);
    t.M = matrix(s.S);
    t.r = vector(s.r);
    t.C = matrix(s.C);
    t.D = matrix(s.D);
    t.d = vector(s.d);
    t.E = clqr::Matrix(0, s.A.cols());
    q.stages.push_back(std::move(t));
  }
  return q;
}
} // namespace laine_tomlin::conversion
#endif
