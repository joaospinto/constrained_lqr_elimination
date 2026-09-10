#ifndef CLQR_BENCHMARK_QUADRATIC_FACTOR_H_
#define CLQR_BENCHMARK_QUADRATIC_FACTOR_H_

// GCC's optimized Eigen 3.4 matrix-vector templates trigger this diagnostic.
// Keep the exception inside the third-party headers, not our adapter code.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <cmath>
#include <limits>
#include <stdexcept>

namespace clqr::benchmark {

struct QuadraticFactor {
  Eigen::MatrixXd jacobian;
  Eigen::VectorXd rhs;
};

// Encode .5*z'*H*z + g'*z as .5*||J*z-b||^2 plus a constant.
// This is input conversion, not a change to the factor-graph solver. A
// semidefinite cost is representable precisely when g lies in range(H).
inline QuadraticFactor MakeQuadraticFactor(const Eigen::MatrixXd &h,
                                           const Eigen::VectorXd &g) {
  const Eigen::Index n = h.rows();
  if (h.cols() != n || g.size() != n || !h.allFinite() || !g.allFinite())
    throw std::invalid_argument("invalid quadratic factor dimensions or data");
  if (!n)
    return {Eigen::MatrixXd(0, 0), Eigen::VectorXd(0)};
  // Preserve the existing SPD conversion and its setup cost.
  Eigen::LLT<Eigen::MatrixXd> factor(h);
  if (factor.info() == Eigen::Success)
    return {factor.matrixL().transpose(), -factor.matrixL().solve(g)};

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(h);
  if (spectrum.info() != Eigen::Success)
    throw std::runtime_error("quadratic factor eigendecomposition failed");
  const double relative = 64 * std::numeric_limits<double>::epsilon() * n;
  const double threshold =
      relative * spectrum.eigenvalues().cwiseAbs().maxCoeff();
  if (spectrum.eigenvalues().minCoeff() < -threshold)
    throw std::runtime_error(
        "unsupported by least-squares cost adapter: indefinite stage cost");
  const Eigen::VectorXd projected = spectrum.eigenvectors().transpose() * g;
  const double gradient_threshold = relative * g.norm();
  QuadraticFactor out{Eigen::MatrixXd::Zero(n, n), Eigen::VectorXd::Zero(n)};
  for (Eigen::Index i = 0; i < n; ++i) {
    const double value = spectrum.eigenvalues()[i];
    if (value <= threshold) {
      if (std::abs(projected[i]) > gradient_threshold)
        throw std::runtime_error("unsupported by least-squares cost adapter: "
                                 "gradient outside cost range");
      continue;
    }
    const double root = std::sqrt(value);
    out.jacobian.row(i) = root * spectrum.eigenvectors().col(i).transpose();
    out.rhs[i] = -projected[i] / root;
  }
  return out;
}

} // namespace clqr::benchmark
#endif
