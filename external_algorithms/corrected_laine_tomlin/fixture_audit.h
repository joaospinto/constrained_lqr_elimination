#ifndef CORRECTED_LAINE_TOMLIN_FIXTURE_AUDIT_H_
#define CORRECTED_LAINE_TOMLIN_FIXTURE_AUDIT_H_
#include "problem_conversion.h"
#include "solver.h"
#include "tests/adversarial_test_support.h"
// Eigen 3.4 has an unused debug counter in SparseCore/TriangularSolver.h.
// Scope the upstream-header exception; project warnings remain errors.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include <Eigen/SparseQR>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace corrected_laine_tomlin::audit {
using Eigen::Index;
using corrected_laine_tomlin::Matrix;
using corrected_laine_tomlin::Vector;

using corrected_laine_tomlin::conversion::Convert;

inline double Max(const Matrix &a) {
  return !a.allFinite() ? std::numeric_limits<double>::infinity()
         : a.size()     ? a.cwiseAbs().maxCoeff()
                        : 0;
}

inline std::string CsvMessage(std::string s) {
  for (char &c : s)
    if (c == ',' || c == '\n' || c == '\r')
      c = ' ';
  return s;
}

// Direct original-coordinate KKT audit of solver-produced multipliers.
inline std::pair<double, double> AuditReturnedMultipliers(const Problem &p,
                                                         const Result &r) {
  const auto N = p.stages.size();
  if (r.states.size() != N + 1 || r.controls.size() != N ||
      r.constraints.size() != N || r.dynamics.size() != N ||
      r.initial.size() != p.initial_state.size() ||
      r.terminal.size() != p.terminal_d.size())
    throw std::runtime_error("invalid returned primal/dual shapes");
  double feasibility = Max(r.states[0] - p.initial_state), stationarity = 0;
  for (std::size_t t = 0; t <= N; ++t) {
    const Vector &x = r.states[t];
    Vector gx = p.Q[t] * x + p.q[t] - (t ? r.dynamics[t - 1] : r.initial);
    if (t == N) {
      gx += p.terminal_C.transpose() * r.terminal;
      feasibility = std::max(feasibility, Max(p.terminal_C * x + p.terminal_d));
    } else {
      const auto &s = p.stages[t];
      const auto &u = r.controls[t];
      if (r.dynamics[t].size() != s.A.rows() ||
          r.constraints[t].size() != s.C.rows())
        throw std::runtime_error("invalid returned stage dual shapes");
      gx += s.S * u + s.A.transpose() * r.dynamics[t] +
            s.C.transpose() * r.constraints[t];
      const Vector gu = s.R * u + s.r + s.S.transpose() * x +
                        s.B.transpose() * r.dynamics[t] +
                        s.D.transpose() * r.constraints[t];
      stationarity = std::max(stationarity, Max(gu));
      feasibility = std::max({feasibility,
          Max(s.A * x + s.B * u + s.c - r.states[t + 1]),
          Max(s.C * x + s.D * u + s.d)});
    }
    stationarity = std::max(stationarity, Max(gx));
  }
  return {feasibility, stationarity};
}

// Untimed independent optimality audit, NOT solver output: assemble original
// sparse J' and find multipliers with sparse QR. This never changes the primal.
inline std::pair<double, double> Audit(const corrected_laine_tomlin::Problem &p,
                                       const corrected_laine_tomlin::Result &r) {
  std::vector<Index> xo, uo;
  Index vars = 0, cons = p.initial_state.size() + p.terminal_d.size();
  for (std::size_t t = 0; t < p.Q.size(); ++t) {
    xo.push_back(vars);
    vars += p.Q[t].rows();
    if (t < p.stages.size()) {
      uo.push_back(vars);
      vars += p.stages[t].B.cols();
      cons += p.stages[t].A.rows() + p.stages[t].d.size();
    }
  }
  if (!vars)
    return {0, 0};
  using Sparse = Eigen::SparseMatrix<double>;
  std::vector<Eigen::Triplet<double>> entries;
  auto block = [&](Index var, Index constraint, const Matrix &a) {
    for (Index i = 0; i < a.rows(); ++i)
      for (Index j = 0; j < a.cols(); ++j)
        if (a(i, j) != 0)
          entries.emplace_back(var + j, constraint + i, a(i, j));
  };
  Index row = p.initial_state.size();
  block(0, 0, Matrix::Identity(row, row));
  double feasibility = Max(r.states[0] - p.initial_state);
  Vector gradient = Vector::Zero(vars);
  for (std::size_t t = 0; t < p.Q.size(); ++t) {
    const Vector &x = r.states[t];
    gradient.segment(xo[t], x.size()) += p.Q[t] * x + p.q[t];
    if (t == p.stages.size())
      break;
    const auto &s = p.stages[t];
    const Vector &u = r.controls[t];
    gradient.segment(xo[t], x.size()) += s.S * u;
    gradient.segment(uo[t], u.size()) = s.R * u + s.S.transpose() * x + s.r;
    block(xo[t], row, s.A);
    block(uo[t], row, s.B);
    block(xo[t + 1], row, -Matrix::Identity(s.A.rows(), s.A.rows()));
    row += s.A.rows();
    block(xo[t], row, s.C);
    block(uo[t], row, s.D);
    row += s.d.size();
    feasibility =
        std::max({feasibility, Max(s.A * x + s.B * u + s.c - r.states[t + 1]),
                  Max(s.C * x + s.D * u + s.d)});
  }
  block(xo.back(), row, p.terminal_C);
  feasibility =
      std::max(feasibility, Max(p.terminal_C * r.states.back() + p.terminal_d));
  if (!cons)
    return {feasibility, Max(gradient)};
  Sparse Jt(vars, cons);
  Jt.setFromTriplets(entries.begin(), entries.end());
  // Column scaling changes the multiplier coordinates, not attainable
  // gradients.
  for (Index j = 0; j < Jt.cols(); ++j) {
    double norm = 0;
    for (Sparse::InnerIterator it(Jt, j); it; ++it)
      norm = std::hypot(norm, it.value());
    if (norm > 0)
      for (Sparse::InnerIterator it(Jt, j); it; ++it)
        it.valueRef() /= norm;
  }
  Eigen::SparseQR<Sparse, Eigen::COLAMDOrdering<int>> qr;
  qr.setPivotThreshold(1e-12);
  qr.compute(Jt);
  if (qr.info() != Eigen::Success)
    throw std::runtime_error("sparse dual audit failed");
  const Vector dual = qr.solve(-gradient);
  return {feasibility, Max(gradient + Jt * dual)};
}

} // namespace corrected_laine_tomlin::audit
#endif
