#include "solver.h"
#include "test_problem.h"

#include <Eigen/Cholesky>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
using namespace laine_tomlin;
using Eigen::Index;
using namespace laine_tomlin::test;

void Check(bool ok, const std::string &what) {
  if (!ok)
    throw std::runtime_error(what);
}
double Max(const Matrix &a) {
  return !a.allFinite() ? std::numeric_limits<double>::infinity()
         : a.size()     ? a.cwiseAbs().maxCoeff()
                        : 0;
}

double worst_primal = 0, worst_stationarity = 0, worst_difference = 0;
void Verify(const Problem &p, const std::string &name) {
  const Result r = Solve(p);
  Check(r.status == Status::kOptimal, name + ": " + r.message);
  Dense d = Assemble(p);
  if (!d.Hessian.rows())
    return;
  Matrix scaled = d.J;
  Vector b = d.b;
  for (Index i = 0; i < b.size(); ++i) {
    const double norm = scaled.row(i).norm();
    if (norm > 0) {
      scaled.row(i) /= norm;
      b[i] /= norm;
    }
  }
  Vector optimum = Vector::Zero(d.Hessian.rows());
  Matrix Z = Matrix::Identity(d.Hessian.rows(), d.Hessian.rows());
  if (scaled.rows()) {
    Eigen::JacobiSVD<Matrix> svd(scaled,
                                 Eigen::ComputeThinU | Eigen::ComputeFullV);
    svd.setThreshold(1e-12);
    optimum = svd.solve(-b);
    Z = svd.matrixV().rightCols(d.Hessian.rows() - svd.rank());
  }
  if (Z.cols()) {
    const Matrix reduced = Z.transpose() * d.Hessian * Z;
    Eigen::LLT<Matrix> llt(reduced);
    Check(llt.info() == Eigen::Success, name + ": oracle not strictly convex");
    optimum -=
        Z *
        llt.solve(Vector(Z.transpose() * (d.Hessian * optimum + d.gradient)));
  }
  Check(Max(scaled * optimum + b) < 1e-8, name + ": dense oracle infeasible");
  Vector actual(d.Hessian.rows());
  for (std::size_t t = 0; t < r.states.size(); ++t) {
    actual.segment(d.xoff[t], r.states[t].size()) = r.states[t];
    if (t < r.controls.size()) {
      actual.segment(d.uoff[t], r.controls[t].size()) = r.controls[t];
      Check(Max(r.controls[t] - r.K[t] * r.states[t] - r.k[t]) < 1e-12,
            name + ": feedback policy mismatch");
    }
  }
  // Duals reconstructed solely for the audit; not attributed to this solver.
  const Vector gradient = d.Hessian * actual + d.gradient;
  Vector lambda_scaled = Vector::Zero(scaled.rows());
  if (scaled.rows()) {
    Eigen::JacobiSVD<Matrix> dual_svd(
        scaled.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
    dual_svd.setThreshold(1e-12);
    lambda_scaled = dual_svd.solve(-gradient);
  }
  const double stationarity =
      Max(gradient + scaled.transpose() * lambda_scaled);
  const double primal = Max(d.J * actual + d.b);
  const double difference = Max(actual - optimum);
  worst_primal = std::max(worst_primal, primal);
  worst_stationarity = std::max(worst_stationarity, stationarity);
  worst_difference = std::max(worst_difference, difference);
  Check(difference < 2e-7 * std::max(1.0, Max(optimum)),
        name + ": dense solution difference=" + std::to_string(difference));
  Check(stationarity < 2e-7,
        name + ": stationarity=" + std::to_string(stationarity));
  Check(primal < 2e-7, name + ": primal residual=" + std::to_string(primal));
  const double objective =
      0.5 * actual.dot(d.Hessian * actual) + d.gradient.dot(actual);
  Check(std::abs(objective - r.objective) <
            1e-10 * std::max(1.0, std::abs(objective)),
        name + ": objective");
}

void Counterexamples() {
  Problem p = Empty(1, 1, 2);
  p.stages[0].R << 2, 1, 1, 2;
  p.stages[0].C = Matrix::Zero(1, 1);
  p.stages[0].D = Matrix::Zero(1, 2);
  p.stages[0].D(0, 0) = 1;
  p.stages[0].d = Vector::Constant(1, -1);
  const Result r = Solve(p);
  Check(r.status == Status::kOptimal &&
            std::abs(r.controls[0][1] + 0.5) < 1e-14,
        "constrained/free Hessian coupling");
  Check(std::abs(r.objective - 0.75) < 1e-14, "analytic optimum objective");
  // Literal arXiv (18) has mu=0, hence u=(1,0): its free gradient is 1.
  const Vector literal = (Vector(2) << 1, 0).finished();
  Check(std::abs((p.stages[0].R * literal)[1] - 1) < 1e-14,
        "literal formula counterexample");
  Verify(p, "control offset coupling");
  p.stages[0].C(0, 0) = -1;
  p.stages[0].d[0] = 0;
  p.initial_state[0] = 2;
  Check(std::abs(Solve(p).K[0](1, 0) + 0.5) < 1e-14, "feedback coupling");
  Verify(p, "feedback coupling");

  p = Empty(1, 1, 1);
  p.stages[0].B(0, 0) = 1;
  p.stages[0].c[0] = 1;
  Check(std::abs(Solve(p).controls[0][0] + 0.5) < 1e-14,
        "affine dynamics contribution");
  Verify(p, "affine dynamics");

  // A redundant constraint produces a cancellation residual in the left-null
  // projection. It must not become a new exact constraint one stage earlier.
  p = Empty(2, 1, 1);
  p.initial_state[0] = 1;
  for (auto &s : p.stages)
    s.B(0, 0) = 1;
  p.stages[1].C.resize(2, 1);
  p.stages[1].C << 1, 3;
  p.stages[1].D = p.stages[1].C;
  p.stages[1].d = Vector::Zero(2);
  const Result redundant = Solve(p);
  Check(redundant.status == Status::kOptimal &&
            std::abs(redundant.controls[0][0] + 2.0 / 3.0) < 1e-14 &&
            std::abs(redundant.controls[1][0] + 1.0 / 3.0) < 1e-14,
        "redundant row must not constrain predecessor state");
  Verify(p, "redundant constraint cancellation");
}

void Edges() {
  Verify(Empty(0, 0, 0), "empty zero horizon");
  Verify(Empty(0, 2, 0), "zero horizon");
  Verify(Empty(5, 2, 0), "zero controls");
  Verify(Empty(5, 0, 2), "zero states");
  Problem p = Empty(2, 2, 1);
  p.terminal_C = Matrix::Zero(1, 2);
  p.terminal_d = Vector::Ones(1);
  Check(Solve(p).status == Status::kInfeasible,
        "inconsistent constant terminal row");
  p.terminal_C(0, 0) = 1;
  Check(Solve(p).status == Status::kInfeasible, "incompatible initial state");
  p.terminal_d[0] = 0;
  Verify(p, "redundant propagated state row");
  p.stages[0].R(0, 0) = -1;
  Check(Solve(p).status == Status::kNumericalFailure,
        "indefinite free cost rejected");
  p.stages[0].R(0, 0) = 0;
  Check(Solve(p).status == Status::kNumericalFailure,
        "singular free cost rejected");
  p.stages[0].R(0, 0) = std::numeric_limits<double>::quiet_NaN();
  Check(Solve(p).status == Status::kInvalidInput, "NaN rejected");
  p = Empty(1, 2, 1);
  p.q.clear();
  Check(Solve(p).status == Status::kInvalidInput, "shape rejected");
  p = Empty(1, 2, 1);
  p.Q[0](0, 1) = 1;
  Check(Solve(p).status == Status::kInvalidInput, "asymmetric cost rejected");
  p = Empty(1, 1, 1);
  p.stages[0].B(0, 0) = 1e200;
  Check(Solve(p).status == Status::kNumericalFailure,
        "overflow rejected before decomposition");

  // Uncontrollable duplicate rows must not accumulate with the horizon.
  p = Empty(2048, 2, 0);
  for (auto &s : p.stages) {
    s.C = Matrix::Ones(6, 2);
    s.D.resize(6, 0);
    s.d = Vector::Zero(6);
  }
  const Result r = Solve(p);
  Check(r.status == Status::kOptimal && r.max_constraint_rows == 1,
        "linear-horizon constraint compression");
}
} // namespace

int main() {
  try {
    Counterexamples();
    Edges();
    for (unsigned seed = 0; seed < 128; ++seed)
      Verify(RandomProblem(seed), "seed " + std::to_string(seed));
    for (unsigned seed = 0; seed < 128; ++seed)
      Verify(RandomProblem(seed, true), "uniform seed " + std::to_string(seed));
    std::cout << "Laine-Tomlin: analytic, edge, 256 dense-oracle cases passed; "
              << "max original feasibility=" << worst_primal
              << ", audited stationarity=" << worst_stationarity
              << ", dense primal difference=" << worst_difference << '\n';
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
