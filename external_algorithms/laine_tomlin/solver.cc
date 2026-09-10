#include "solver.h"
#include "dense.h"

#include <Eigen/Cholesky>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <utility>

namespace laine_tomlin {
namespace {
using Eigen::Index;
using dense::AddProduct;
using dense::Product;
using RowMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <class Derived>
double MaxAbs(const Eigen::MatrixBase<Derived> &value) {
  return value.size() ? value.cwiseAbs().maxCoeff() : 0;
}

bool Shape(const Matrix &a, Index rows, Index cols) {
  return a.rows() == rows && a.cols() == cols && a.allFinite();
}

bool Symmetric(const Matrix &a) {
  return a.rows() == a.cols() && a.allFinite() &&
         MaxAbs(a - a.transpose()) <= 1e-12 * std::max(1.0, MaxAbs(a));
}

bool Valid(const Problem &p, const Options &o) {
  const auto N = p.stages.size();
  if (!(o.rank_tolerance > 0 && o.rank_tolerance < 1) ||
      !(o.feasibility_tolerance > 0) ||
      !std::isfinite(o.feasibility_tolerance) || p.Q.size() != N + 1 ||
      p.q.size() != N + 1)
    return false;
  for (std::size_t t = 0; t <= N; ++t) {
    if (!Symmetric(p.Q[t]) || !Shape(p.q[t], p.Q[t].rows(), 1))
      return false;
    if (t == N)
      break;
    const auto &s = p.stages[t];
    const Index n = p.Q[t].rows(), next = p.Q[t + 1].rows();
    const Index m = s.B.cols(), rows = s.C.rows();
    if (!Shape(s.A, next, n) || !Shape(s.B, next, m) || !Shape(s.c, next, 1) ||
        !Shape(s.R, m, m) || !Symmetric(s.R) || !Shape(s.S, n, m) ||
        !Shape(s.r, m, 1) || !Shape(s.C, rows, n) || !Shape(s.D, rows, m) ||
        !Shape(s.d, rows, 1))
      return false;
  }
  return Shape(p.initial_state, p.Q.front().rows(), 1) &&
         Shape(p.terminal_C, p.terminal_d.size(), p.Q.back().rows()) &&
         p.terminal_d.allFinite();
}

// Normalize equality rows, including rows whose only nonzero is their offset.
// Scaling changes only the off-feasible-set least-squares extension of a
// policy.
void Equilibrate(Matrix &a, Matrix &b, Vector &d) {
  for (Index i = 0; i < d.size(); ++i) {
    const double scale = std::max(a.cols() ? MaxAbs(a.row(i)) : 0,
                                  b.cols() ? MaxAbs(b.row(i)) : 0);
    const double divisor = scale > 0 ? scale : std::abs(d[i]);
    if (divisor > 0) {
      if (a.cols())
        a.row(i) /= divisor;
      if (b.cols())
        b.row(i) /= divisor;
      d[i] /= divisor;
    }
  }
}

Index Rank(const Vector &singular, double threshold) {
  if (!singular.size())
    return 0;
  return (singular.array() > threshold).count();
}

// Compress H*x+h=0 with an SVD, retaining an orthonormal row basis. Unlike a
// full-rank-[H,h] test, the left-null residual also detects 0*x+nonzero=0.
Status Compress(Matrix &H, Vector &h, const Options &o) {
  Matrix empty = Matrix::Zero(H.rows(), 0);
  Equilibrate(H, empty, h);
  if (!H.allFinite() || !h.allFinite())
    return Status::kNumericalFailure;
  if (H.rows() == 0)
    return Status::kOptimal;
  if (H.cols() == 0) {
    if (MaxAbs(h) > o.feasibility_tolerance)
      return Status::kInfeasible;
    H.resize(0, 0);
    h.resize(0);
    return Status::kOptimal;
  }
  Eigen::JacobiSVD<Matrix> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Index rank =
      Rank(svd.singularValues(), o.rank_tolerance * svd.singularValues()[0]);
  const auto U = svd.matrixU().leftCols(rank);
  const Vector projected = Product(U.transpose(), h);
  if (MaxAbs(h - Product(U, projected)) >
      o.feasibility_tolerance * std::max(1.0, MaxAbs(h)))
    return Status::kInfeasible;
  H = svd.matrixV().leftCols(rank).transpose();
  h = projected.cwiseQuotient(svd.singularValues().head(rank));
  return H.allFinite() && h.allFinite() ? Status::kOptimal
                                        : Status::kNumericalFailure;
}

Result Failure(Status status, const std::string &message) {
  Result r;
  r.status = status;
  r.message = message;
  return r;
}
} // namespace

Result Solve(const Problem &p, const Options &o) {
  if (!Valid(p, o))
    return Failure(Status::kInvalidInput, "invalid shape, values, or options");
  const std::size_t N = p.stages.size();
  Result result;
  result.K.resize(N);
  result.k.resize(N);
  Matrix V = p.Q.back(), H = p.terminal_C;
  Vector v = p.q.back(), h = p.terminal_d;
  const Status terminal_status = Compress(H, h, o);
  if (terminal_status != Status::kOptimal)
    return Failure(terminal_status,
                   "inconsistent or nonfinite terminal constraints");
  result.max_constraint_rows = H.rows();

  for (std::size_t t = N; t-- > 0;) {
    const auto &s = p.stages[t];
    const Index n = s.A.cols(), m = s.B.cols(), local = s.C.rows();
    const Matrix VA = Product(V, s.A), VB = Product(V, s.B);
    const Vector shifted = v + Product(V, s.c);
    const Matrix Mxx = p.Q[t] + Product(s.A.transpose(), VA);
    const Matrix Mux = s.S.transpose() + Product(s.B.transpose(), VA);
    Matrix Muu = s.R + Product(s.B.transpose(), VB);
    Muu = (0.5 * (Muu + Muu.transpose())).eval();
    const Vector mx = p.q[t] + Product(s.A.transpose(), shifted);
    const Vector mu = s.r + Product(s.B.transpose(), shifted);
    Matrix Nx(local + H.rows(), n), Nu(local + H.rows(), m);
    Vector b(local + H.rows());
    if (local) {
      if (n)
        Nx.topRows(local) = s.C;
      if (m)
        Nu.topRows(local) = s.D;
      b.head(local) = s.d;
    }
    if (H.rows()) {
      if (n)
        Nx.bottomRows(H.rows()) = Product(H, s.A);
      if (m)
        Nu.bottomRows(H.rows()) = Product(H, s.B);
      b.tail(H.rows()) = h + Product(H, s.c);
    }
    Equilibrate(Nx, Nu, b);
    if (!Mxx.allFinite() || !Mux.allFinite() || !Muu.allFinite() ||
        !mx.allFinite() || !mu.allFinite() || !Nx.allFinite() ||
        !Nu.allFinite() || !b.allFinite())
      return Failure(Status::kNumericalFailure, "nonfinite stage reduction");

    Matrix Kp = Matrix::Zero(m, n), Z = Matrix::Identity(m, m);
    Vector kp = Vector::Zero(m);
    Matrix next_H = Nx;
    Vector next_h = b;
    if (Nu.rows() && m) {
      // Thin U avoids a quadratic allocation in the number of constraint rows;
      // full V supplies all control null-space directions.
      Eigen::JacobiSVD<Matrix> svd(Nu,
                                   Eigen::ComputeThinU | Eigen::ComputeFullV);
      // Rows have unit coefficient scale; the absolute floor prevents treating
      // cancellation noise from propagated constraints as an actionable pivot.
      const Index rank =
          Rank(svd.singularValues(),
               o.rank_tolerance * std::max(1.0, svd.singularValues()[0]));
      const auto U = svd.matrixU().leftCols(rank);
      const Matrix UNx = Product(U.transpose(), Nx);
      const Vector Ub = Product(U.transpose(), b);
      const Matrix inverse_action =
          svd.matrixV().leftCols(rank) *
          svd.singularValues().head(rank).cwiseInverse().asDiagonal();
      dense::Multiply(Kp, inverse_action, UNx, -1, 0);
      dense::Multiply(kp, inverse_action, Ub, -1, 0);
      AddProduct(next_H, U, UNx, -1);
      AddProduct(next_h, U, Ub, -1);
      // Residual components at the SVD rounding level must not be promoted to
      // unit constraints by row equilibration in Compress.
      for (Index i = 0; i < next_H.rows(); ++i) {
        const double original =
            std::max({1.0, n ? MaxAbs(Nx.row(i)) : 0, std::abs(b[i])});
        if (std::max(n ? MaxAbs(next_H.row(i)) : 0, std::abs(next_h[i])) <=
            o.rank_tolerance * original) {
          if (n)
            next_H.row(i).setZero();
          next_h[i] = 0;
        }
      }
      Z = svd.matrixV().rightCols(m - rank);
    }
    Matrix &K = result.K[t];
    Vector &k = result.k[t];
    K = Kp;
    k = kp;
    if (Z.cols()) {
      const Matrix MZ = Product(Muu, Z);
      Matrix reduced = Product(Z.transpose(), MZ);
      reduced = (0.5 * (reduced + reduced.transpose())).eval();
      Eigen::LLT<Matrix> llt(reduced);
      if (llt.info() != Eigen::Success)
        return Failure(Status::kNumericalFailure,
                       "non-positive-definite free-control Hessian at stage " +
                           std::to_string(t));
      RowMatrix rhs(Z.cols(), n + 1);
      // The Muu*Kp and Muu*kp terms are essential: Euclidean-orthogonal
      // constrained/free directions need not be Muu-orthogonal.
      const Matrix matrix_rhs = Mux + Product(Muu, Kp);
      const Vector vector_rhs = mu + Product(Muu, kp);
      rhs.leftCols(n) = Product(Z.transpose(), matrix_rhs);
      rhs.col(n) = Product(Z.transpose(), vector_rhs);
      const RowMatrix lower = llt.matrixL();
      clqr::detail::NativeCholeskySolve(lower.data(), lower.rows(), rhs.cols(),
                                       rhs.data());
      AddProduct(K, Z, rhs.leftCols(n), -1);
      AddProduct(k, Z, rhs.col(n), -1);
    }
    const Matrix cross = Product(Mux.transpose(), K);
    V = Mxx + cross + cross.transpose() + Product(K.transpose(), Product(Muu, K));
    V = (0.5 * (V + V.transpose())).eval();
    const Vector shifted_control = mu + Product(Muu, k);
    v = mx + Product(Mux.transpose(), k) + Product(K.transpose(), shifted_control);
    if (!V.allFinite() || !v.allFinite() || !K.allFinite() || !k.allFinite())
      return Failure(Status::kNumericalFailure,
                     "nonfinite backward recurrence");
    H = std::move(next_H);
    h = std::move(next_h);
    const Status constraint_status = Compress(H, h, o);
    if (constraint_status != Status::kOptimal)
      return Failure(
          constraint_status,
          "inconsistent or nonfinite propagated constraints at stage " +
              std::to_string(t));
    result.max_constraint_rows = std::max(result.max_constraint_rows, H.rows());
  }
  if (MaxAbs(Product(H, p.initial_state) + h) >
      o.feasibility_tolerance *
          std::max({1.0, MaxAbs(h), MaxAbs(p.initial_state)}))
    return Failure(Status::kInfeasible,
                   "initial state violates constraints-to-go");

  result.states.resize(N + 1);
  result.controls.resize(N);
  result.states[0] = p.initial_state;
  for (std::size_t t = 0; t < N; ++t) {
    const auto &s = p.stages[t];
    const Vector &x = result.states[t];
    Vector &u = result.controls[t];
    u = Product(result.K[t], x) + result.k[t];
    result.states[t + 1] = Product(s.A, x) + Product(s.B, u) + s.c;
    result.objective +=
        0.5 * x.dot(Product(p.Q[t], x)) + p.q[t].dot(x) +
        x.dot(Product(s.S, u)) + 0.5 * u.dot(Product(s.R, u)) + s.r.dot(u);
    if (!u.allFinite() || !result.states[t + 1].allFinite())
      return Failure(Status::kNumericalFailure, "nonfinite trajectory");
  }
  const Vector &last = result.states.back();
  result.objective +=
      0.5 * last.dot(Product(p.Q.back(), last)) + p.q.back().dot(last);
  if (!std::isfinite(result.objective))
    return Failure(Status::kNumericalFailure, "nonfinite objective");
  result.status = Status::kOptimal;
  result.message = "optimal";
  return result;
}
} // namespace laine_tomlin
