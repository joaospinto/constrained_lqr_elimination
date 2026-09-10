#ifndef CLQR_BENCHMARKS_REFERENCE_STATIONARITY_AUDIT_H_
#define CLQR_BENCHMARKS_REFERENCE_STATIONARITY_AUDIT_H_
#include "benchmarks/reference/eigen_problem.h"
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
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace clqr::benchmark::reference::audit {
using Eigen::Index;
using clqr::benchmark::reference::Matrix;
using clqr::benchmark::reference::Vector;

inline double Max(const Matrix &a) {
  return !a.allFinite() ? std::numeric_limits<double>::infinity()
         : a.size()     ? a.cwiseAbs().maxCoeff()
                        : 0;
}

// Untimed independent optimality audit, NOT solver output: assemble original
// sparse J' and find multipliers with sparse QR. This never changes the primal.
inline std::pair<double, double> Audit(const Problem &p, const Trajectory &r) {
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

} // namespace clqr::benchmark::reference::audit
#endif
