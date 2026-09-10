#include "benchmarks/reference/quadratic_factor.h"

#ifdef CLQR_BENCHMARK_GTSAM
#include <EcLqr_fg.h>
#endif

#include <iostream>
#include <string_view>

namespace {
using clqr::benchmark::MakeQuadraticFactor;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void Require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void CheckEncoding(const MatrixXd &h, const VectorXd &g) {
  const auto [j, b] = MakeQuadraticFactor(h, g);
  Require((j.transpose() * j - h).norm() <= 1e-12 * h.norm(),
          "quadratic coefficient changed");
  Require((j.transpose() * b + g).norm() <= 1e-12 * g.norm(),
          "linear coefficient changed");
}

void CheckUnsupported(const MatrixXd &h, const VectorXd &g,
                      std::string_view reason) {
  try {
    (void)MakeQuadraticFactor(h, g);
  } catch (const std::runtime_error &e) {
    Require(std::string_view(e.what()).find(reason) != std::string_view::npos,
            "wrong cost-adapter diagnostic");
    return;
  }
  throw std::runtime_error("unsupported cost was silently changed");
}

#ifdef CLQR_BENCHMARK_GTSAM
// Run the authors' unchanged elimination/forward solve on a coupled PSD cost.
// x0=1, x1=x0+u0, cost .5*(x0+u0)^2 - 2*(x0+u0).
// Its unique optimum is u0=1, x1=2 despite the singular stage Hessian.
void CheckAuthorSolve() {
  using gtsam::Symbol;
  using gtsam::noiseModel::Constrained;
  using gtsam::noiseModel::Unit;
  gtsam::GaussianFactorGraph graph;
  const MatrixXd one = MatrixXd::Ones(1, 1);
  const VectorXd zero = VectorXd::Zero(1);
  graph.add(Symbol('x', 0), one, VectorXd::Ones(1), Constrained::All(1));
  graph.add(Symbol('x', 0), one, Symbol('u', 0), one, Symbol('x', 1), -one,
            zero, Constrained::All(1));
  const auto [j, b] =
      MakeQuadraticFactor(MatrixXd::Ones(2, 2), VectorXd::Constant(2, -2));
  graph.add(Symbol('x', 0), j.leftCols(1), Symbol('u', 0), j.rightCols(1), b,
            Unit::Create(2));
  const auto result = ecLqr::fgSolFromBn(ecLqr::BnFromGfg(graph, 1));
  Require(std::abs(result.at(Symbol('u', 0))[0] - 1) < 1e-12,
          "PSD cost author control");
  Require(std::abs(result.at(Symbol('x', 1))[0] - 2) < 1e-12,
          "PSD cost author state");

  // With no cost and two redundant fixed-state rows, the zero factor must
  // remain zero: no damping or artificial objective is introduced.
  gtsam::GaussianFactorGraph fixed;
  MatrixXd rows(2, 1);
  rows << 1, 3;
  fixed.add(Symbol('x', 0), rows, (2 * rows).col(0), Constrained::All(2));
  const auto [zj, zb] = MakeQuadraticFactor(MatrixXd::Zero(1, 1), zero);
  fixed.add(Symbol('x', 0), zj, zb, Unit::Create(1));
  const auto fixed_result = ecLqr::fgSolFromBn(ecLqr::BnFromGfg(fixed, 0));
  Require(std::abs(fixed_result.at(Symbol('x', 0))[0] - 2) < 1e-12,
          "zero-cost redundant author solve");
}
#endif
} // namespace

int main() {
  CheckEncoding(MatrixXd::Identity(3, 3), VectorXd::LinSpaced(3, -1, 2));
  CheckEncoding(MatrixXd::Ones(3, 3), VectorXd::Constant(3, 2));
  CheckEncoding(MatrixXd::Zero(2, 2), VectorXd::Zero(2));
  CheckEncoding(MatrixXd(0, 0), VectorXd(0));
  for (double scale : {1e-12, 1.0, 1e12}) {
    MatrixXd h = MatrixXd::Zero(2, 2);
    h(1, 1) = scale;
    VectorXd g(2);
    g << 0, -3 * scale;
    CheckEncoding(h, g);
    g[0] = scale;
    CheckUnsupported(h, g, "gradient outside cost range");
    h(0, 0) = -scale;
    CheckUnsupported(h, g, "indefinite stage cost");
  }
#ifdef CLQR_BENCHMARK_GTSAM
  CheckAuthorSolve();
#endif
  std::cout << "Quadratic cost conversion tests passed\n";
}
