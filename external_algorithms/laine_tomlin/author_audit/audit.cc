// Adapter for the original author-written solver, not our reimplementation.
// Only input coefficients are assigned here. Every DP and forward operation
// executes in the separately downloaded, unmodified trajectory.cpp.
#include "../fixture_audit.h"
#include "../test_problem.h"
#include "adapter.h"
#include <Eigen/SVD>
#include <functional>

#include <chrono>
#include <iomanip>
#include <memory>
#include <optional>
#include <string>

namespace {
using namespace laine_tomlin;
using namespace laine_tomlin::audit;

using laine_tomlin::author::MakeTrajectory;

Result Run(trajectory::Trajectory &t) {
  t.compute_feedback_policies();
  t.compute_state_control_dependencies();
  t.set_open_loop_traj();
  Result r;
  r.states = t.open_loop_states;
  r.controls = t.open_loop_controls;
  r.K = t.current_state_feedback_matrices;
  r.k = t.feedforward_controls;
  return r;
}

// Audit the author's own multipliers separately from the QR certificate.
double AuthorStationarity(const Problem &p, const Result &r,
                          trajectory::Trajectory &t) {
  t.compute_multipliers();
  t.set_lq_multipliers();
  double error = 0;
  for (std::size_t k = 0; k < p.stages.size(); ++k) {
    const auto &s = p.stages[k];
    const Vector mu = t.lq_running_constraint_multipliers[k].head(s.d.size());
    const auto &next = t.lq_dynamics_multipliers[k + 1];
    error = std::max(
        {error,
         Max(p.Q[k] * r.states[k] + p.q[k] + s.S * r.controls[k] +
             s.C.transpose() * mu + t.lq_dynamics_multipliers[k] -
             s.A.transpose() * next),
         Max(s.R * r.controls[k] + s.r + s.S.transpose() * r.states[k] +
             s.D.transpose() * mu - s.B.transpose() * next)});
  }
  return std::max(error, Max(p.Q.back() * r.states.back() + p.q.back() +
                             t.lq_dynamics_multipliers.back() +
                             p.terminal_C.transpose() *
                                 t.lq_terminal_constraint_multiplier.head(
                                     p.terminal_d.size())));
}

Problem Analytic(const std::string &name) {
  if (name == "analytic-redundant-feedback" ||
      name == "analytic-single-feedback") {
    auto p = laine_tomlin::test::Empty(2, 1, 1);
    p.initial_state[0] = 1;
    for (auto &s : p.stages)
      s.B(0, 0) = 1;
    auto &s = p.stages[1];
    s.C.resize(2, 1);
    s.C << 1, 3;
    s.D = s.C;
    s.d = Vector::Zero(2);
    if (name == "analytic-single-feedback") {
      s.C.conservativeResize(1, 1);
      s.D.conservativeResize(1, 1);
      s.d.conservativeResize(1);
    }
    return p;
  }
  Problem p;
  p.initial_state = Vector::Zero(1);
  p.Q = {Matrix::Identity(1, 1), Matrix::Identity(1, 1)};
  p.q = {Vector::Zero(1), Vector::Zero(1)};
  p.terminal_C.resize(0, 1);
  p.terminal_d.resize(0);
  Stage s;
  s.A = Matrix::Identity(1, 1);
  s.c = Vector::Zero(1);
  if (name == "analytic-affine") {
    s.B = Matrix::Ones(1, 1);
    s.c[0] = 1;
    s.R = Matrix::Identity(1, 1);
    s.S = Matrix::Zero(1, 1);
    s.r = Vector::Zero(1);
    s.C.resize(0, 1);
    s.D.resize(0, 1);
    s.d.resize(0);
  } else {
    s.B = Matrix::Zero(1, 2);
    s.R.resize(2, 2);
    s.R << 2, 1, 1, 2;
    s.S = Matrix::Zero(1, 2);
    s.r = Vector::Zero(2);
    s.C = Matrix::Zero(1, 1);
    s.D.resize(1, 2);
    s.D << 1, 0;
    s.d = Vector::Constant(1, -1);
    if (name == "analytic-feedback") {
      s.C(0, 0) = -1;
      s.d[0] = 0;
      p.initial_state[0] = 2;
    }
  }
  p.stages.push_back(s);
  return p;
}
double DenseDifference(const Problem &p, const Result &r) {
  auto d = laine_tomlin::test::Assemble(p);
  Matrix scaled = d.J;
  Vector b = d.b;
  for (Eigen::Index i = 0; i < b.size(); ++i) {
    double norm = scaled.row(i).norm();
    if (norm > 0) {
      scaled.row(i) /= norm;
      b[i] /= norm;
    }
  }
  Eigen::JacobiSVD<Matrix> svd(scaled,
                               Eigen::ComputeThinU | Eigen::ComputeFullV);
  svd.setThreshold(1e-12);
  Vector optimum = svd.solve(-b);
  Matrix Z = svd.matrixV().rightCols(d.Hessian.rows() - svd.rank());
  if (Z.cols()) {
    Matrix reduced = Z.transpose() * d.Hessian * Z;
    Eigen::LLT<Matrix> factor(reduced);
    if (factor.info() != Eigen::Success)
      throw std::runtime_error("dense oracle not positive definite");
    optimum -= Z * factor.solve(Vector(Z.transpose() *
                                       (d.Hessian * optimum + d.gradient)));
  }
  if (Max(scaled * optimum + b) > 1e-8)
    throw std::runtime_error("dense oracle infeasible");
  double diff = 0;
  for (std::size_t k = 0; k < r.states.size(); ++k) {
    diff = std::max(diff, Max(r.states[k] -
                              optimum.segment(d.xoff[k], r.states[k].size())));
    if (k < r.controls.size())
      diff =
          std::max(diff, Max(r.controls[k] -
                             optimum.segment(d.uoff[k], r.controls[k].size())));
  }
  return diff;
}
} // namespace

int main(int argc, char **argv) {
  std::cout << std::setprecision(12);
  auto cases = clqr::test::adversarial::StandardCases();
  auto ext = clqr::test::adversarial::ExtendedCases();
  cases.insert(cases.end(), ext.begin(), ext.end());
  if (argc < 2)
    return 2;
  bool implicit = false, multipliers = false;
  for (int i = 2; i < argc; ++i) {
    if (std::string(argv[i]) == "--implicit")
      implicit = true;
    else if (std::string(argv[i]) == "--multipliers")
      multipliers = true;
    else
      return 2;
  }
  const std::string name = argv[1];
  if (name == "--list") {
    for (auto s : {"analytic-coupling", "analytic-feedback", "analytic-affine",
                   "analytic-redundant-feedback", "analytic-single-feedback"})
      std::cout << s << '\n';
    for (const auto &c : cases)
      std::cout << c.name << '\n';
    for (int seed = 0; seed < 128; ++seed)
      std::cout << "random-" << seed << '\n';
    for (int N : {16, 32, 64, 128, 256, 512})
      std::cout << "redundant-chain-" << N << '\n';
    return 0;
  }
  try {
    Problem p;
    std::optional<clqr::Problem> random_native;
    const clqr::Problem *native = nullptr;
    std::string expected = "optimal";
    if (name.starts_with("analytic-"))
      p = Analytic(name);
    else if (name.starts_with("random-")) {
      p = laine_tomlin::test::RandomProblem(std::stoul(name.substr(7)), true);
      random_native = laine_tomlin::conversion::ToClqr(p);
      native = &*random_native;
    } else if (name.starts_with("redundant-chain-")) {
      p = laine_tomlin::test::Empty(std::stoul(name.substr(16)), 1, 1);
      for (auto &s : p.stages) {
        s.C = Matrix::Ones(1, 1);
        s.D = Matrix::Zero(1, 1);
        s.d = Vector::Zero(1);
      }
    } else {
      auto it = std::find_if(cases.begin(), cases.end(),
                             [&](const auto &c) { return c.name == name; });
      if (it == cases.end())
        throw std::runtime_error("unknown case");
      if (it->cpu_status == clqr::SolveStatus::kInvalidInput) {
        std::cout << name << ",unsupported,invalid-input\n";
        return 0;
      }
      native = &it->problem;
      if (it->cpu_status != clqr::SolveStatus::kOptimal)
        expected = "nonoptimal";
      if (name == "indefinite-reduced-hessian")
        expected = "unbounded";
      p = Convert(it->problem);
    }
    auto t = MakeTrajectory(p, implicit);
    auto start = std::chrono::steady_clock::now();
    auto r = Run(*t);
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start)
                    .count();
    for (const auto &x : r.states)
      if (!x.allFinite())
        throw std::runtime_error("nonfinite state");
    for (const auto &u : r.controls)
      if (!u.allFinite())
        throw std::runtime_error("nonfinite control");
    auto [feas, stat] = Audit(p, r);
    double author_stationarity = std::numeric_limits<double>::quiet_NaN();
    if (multipliers)
      author_stationarity = AuthorStationarity(p, r, *t);
    double diff = std::numeric_limits<double>::quiet_NaN();
    double core_kkt = std::numeric_limits<double>::quiet_NaN();
    if (name.starts_with("random-"))
      diff = DenseDifference(p, r);
    if (name.starts_with("redundant-chain-")) {
      diff = 0;
      for (const auto &x : r.states)
        diff = std::max(diff, Max(x));
      for (const auto &u : r.controls)
        diff = std::max(diff, Max(u));
    }
    if (native && expected == "optimal") {
      if (!std::isfinite(diff))
        diff = 0;
      clqr::Workspace workspace;
      workspace.Reserve(*native);
      auto q = clqr::Solve(*native, workspace);
      if (q.status != clqr::SolveStatus::kOptimal)
        throw std::runtime_error("CLQR comparison did not solve");
      auto point = clqr::test::adversarial::CopyCpuSolution(q);
      core_kkt = clqr::test::adversarial::MaxKktResidual(*native, point);
      for (std::size_t k = 0; k < r.states.size(); ++k) {
        diff = std::max(diff, Max(r.states[k] - Convert(point.states[k])));
        if (k < r.controls.size())
          diff =
              std::max(diff, Max(r.controls[k] - Convert(point.controls[k])));
      }
    }
    if (name.starts_with("analytic-")) {
      std::cerr << name << " u=" << r.controls[0].transpose()
                << " K=" << r.K[0].transpose() << '\n';
      Vector answer = Vector::Constant(p.stages[0].B.cols(), -0.5);
      if (name != "analytic-affine")
        answer[0] = 1;
      if (name == "analytic-feedback")
        answer *= 2;
      diff = Max(r.controls[0] - answer);
      if (name == "analytic-redundant-feedback" ||
          name == "analytic-single-feedback") {
        diff = std::max(std::abs(r.controls[0][0] + 2.0 / 3.0),
                        std::abs(r.controls[1][0] + 1.0 / 3.0));
        std::cerr << "u1=" << r.controls[1].transpose()
                  << " dense_difference=" << DenseDifference(p, r) << '\n';
        if (name == "analytic-redundant-feedback") {
          const auto &s = p.stages[1];
          Eigen::JacobiSVD<Matrix> svd(s.D, Eigen::ComputeFullU |
                                                Eigen::ComputeFullV);
          const Matrix transformed = svd.matrixU().transpose() * s.C;
          const Matrix residual = transformed.bottomRows(1);
          Eigen::JacobiSVD<Matrix> next(residual * p.stages[0].B,
                                        Eigen::ComputeFullU |
                                            Eigen::ComputeFullV);
          std::cerr << std::setprecision(17)
                    << "separate projection recomputation=" << residual
                    << "; next control rank=" << next.rank() << '\n';
        }
      }
    }
    std::cout << name << ",returned," << ms << ',' << feas << ',' << stat << ','
              << diff << ',' << core_kkt << ',' << expected << ','
              << author_stationarity << '\n';
    return (expected == "optimal" &&
            (feas > 1e-7 || stat > 1e-7 || diff > 1e-7))
               ? 1
               : 0;
  } catch (const std::exception &e) {
    std::cout << name << ",exception," << CsvMessage(e.what()) << '\n';
    return 1;
  }
}
