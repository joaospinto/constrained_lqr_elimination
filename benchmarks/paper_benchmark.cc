#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "benchmarks/paper_cases.h"
#include "benchmarks/scaling_problem.h"
#ifdef CLQR_BENCHMARK_ADVERSARIAL
#include "tests/adversarial_test_support.h"
#endif

#ifdef CLQR_BENCHMARK_GEN_RICCATI
#include <gen_riccati.hpp>
#endif
#ifdef CLQR_BENCHMARK_GTSAM
#include <EcLqr_fg.h>
#include "benchmarks/reference/quadratic_factor.h"
#endif
#ifdef CLQR_BENCHMARK_CUDA
#include "clqr/cuda.h"
#endif
#ifdef CLQR_BENCHMARK_LAINE
#include "benchmarks/reference/laine_author_adapter.h"
#endif
#ifdef CLQR_BENCHMARK_LAINE_CORRECTED
#include "external_algorithms/corrected_laine_tomlin/problem_conversion.h"
#endif
#ifdef CLQR_BENCHMARK_INDEPENDENT_FIXTURES
#include "external_algorithms/corrected_laine_tomlin/test_problem.h"
#include "external_algorithms/corrected_laine_tomlin/problem_conversion.h"
#endif
#if defined(CLQR_BENCHMARK_ADVERSARIAL) && \
    (defined(CLQR_BENCHMARK_LAINE) || defined(CLQR_BENCHMARK_GTSAM) || \
     defined(CLQR_BENCHMARK_LAINE_CORRECTED))
#include "benchmarks/reference/stationarity_audit.h"
#endif

namespace {
using clqr::Matrix;
using clqr::Problem;
using clqr::Vector;
using Clock = std::chrono::steady_clock;
using Trajectory = std::pair<std::vector<Vector>, std::vector<Vector>>;

using clqr::benchmark::Multipliers;

Vector CopyVector(const clqr::VectorView &v) {
  Vector out(v.size);
  for (std::size_t j = 0; j < v.size; ++j)
    out[j] = v[j];
  return out;
}

template <class View> Multipliers CopyMultipliers(const View &v) {
  Multipliers out;
  out.initial = CopyVector(v.initial_multiplier);
  out.terminal = CopyVector(v.terminal_state_multiplier);
  for (std::size_t i = 0; i < v.dynamics_multiplier_count; ++i)
    out.dynamics.push_back(CopyVector(v.dynamics_multipliers[i]));
  for (std::size_t i = 0; i < v.mixed_multiplier_count; ++i)
    out.mixed.push_back(CopyVector(v.mixed_multipliers[i]));
  for (std::size_t i = 0; i < v.state_multiplier_count; ++i)
    out.state.push_back(CopyVector(v.state_multipliers[i]));
  return out;
}

// Unlike std::max-based reductions, a nonfinite entry must not disappear.
double InfinityNorm(const Vector &v) {
  double out = 0;
  for (std::size_t j = 0; j < v.size(); ++j) {
    if (!std::isfinite(v[j]))
      return std::numeric_limits<double>::infinity();
    out = std::max(out, std::abs(double(v[j])));
  }
  return out;
}

struct Residuals {
  double feasibility = 0;
  double stationarity = std::numeric_limits<double>::quiet_NaN();
};

Residuals Audit(const Problem &p, const Trajectory &trajectory,
                const std::optional<Multipliers> &dual) {
  const auto &[x, u] = trajectory;
  Residuals out;
  out.feasibility = InfinityNorm(x.front() - p.initial_state);
  if (dual)
    out.stationarity = 0;
  for (std::size_t i = 0; i < p.stages.size(); ++i) {
    const auto &s = p.stages[i];
    out.feasibility =
        std::max({out.feasibility,
                  InfinityNorm(x[i + 1] - s.A * x[i] - s.B * u[i] - s.c),
                  InfinityNorm(s.C * x[i] + s.D * u[i] + s.d),
                  InfinityNorm(s.E * x[i] + s.e)});
    if (dual) {
      const Vector gx = p.Q[i] * x[i] + s.M * u[i] + p.q[i] -
                        clqr::Transpose(s.A) * dual->dynamics[i] +
                        (i == 0 ? dual->initial : dual->dynamics[i - 1]) +
                        clqr::Transpose(s.C) * dual->mixed[i] +
                        clqr::Transpose(s.E) * dual->state[i];
      const Vector gu = clqr::Transpose(s.M) * x[i] + s.R * u[i] + s.r -
                        clqr::Transpose(s.B) * dual->dynamics[i] +
                        clqr::Transpose(s.D) * dual->mixed[i];
      out.stationarity =
          std::max({out.stationarity, InfinityNorm(gx), InfinityNorm(gu)});
    }
  }
  out.feasibility = std::max(
      out.feasibility, InfinityNorm(p.terminal_E * x.back() + p.terminal_e));
  if (dual)
    out.stationarity =
        std::max(out.stationarity,
                 InfinityNorm(p.Q.back() * x.back() + p.q.back() +
                              (p.stages.empty() ? dual->initial
                                                : dual->dynamics.back()) +
                              clqr::Transpose(p.terminal_E) * dual->terminal));
  return out;
}

double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

template <class View> Trajectory Copy(const View &result) {
  Trajectory out;
  for (std::size_t i = 0; i < result.state_count; ++i) {
    Vector x(result.states[i].size);
    for (std::size_t j = 0; j < x.size(); ++j)
      x[j] = result.states[i][j];
    out.first.push_back(std::move(x));
  }
  for (std::size_t i = 0; i < result.control_count; ++i) {
    Vector u(result.controls[i].size);
    for (std::size_t j = 0; j < u.size(); ++j)
      u[j] = result.controls[i][j];
    out.second.push_back(std::move(u));
  }
  return out;
}

class CpuSolver {
public:
  explicit CpuSolver(const Problem &p) : p_(p) { workspace_.Reserve(p); }
  void Solve() {
    result_ = clqr::Solve(p_, workspace_);
    if (result_.status != clqr::SolveStatus::kOptimal)
      throw std::runtime_error(result_.message);
  }
  Trajectory Result() const { return Copy(result_); }
  Multipliers DualResult() const { return CopyMultipliers(result_); }

private:
  const Problem &p_;
  clqr::Workspace workspace_;
  clqr::SolutionView result_;
};

#ifdef CLQR_BENCHMARK_CUDA
class CudaSolver {
public:
  explicit CudaSolver(const Problem &p) : p_(p) {
    workspace_.Reserve(p, options_);
  }
  void Solve() {
    result_ = clqr::cuda::SolvePreparedView(p_, workspace_, options_);
    if (result_.status != clqr::SolveStatus::kOptimal)
      throw std::runtime_error(result_.message);
  }
  Trajectory Result() const { return Copy(result_); }
  Multipliers DualResult() const { return CopyMultipliers(result_); }
  double KernelMs() const {
    const auto &t = result_.timings;
    return t.feasibility_ms + t.reduction_ms + t.riccati_ms +
           t.reconstruction_ms + t.multiplier_ms + t.objective_ms;
  }

private:
  const Problem &p_;
  clqr::cuda::Options options_;
  clqr::cuda::Workspace workspace_;
  clqr::cuda::SolutionView result_;
};
#endif

#ifdef CLQR_BENCHMARK_GTSAM
gtsam::Matrix EigenMatrix(const Matrix &a) {
  gtsam::Matrix out(a.rows(), a.cols());
  for (std::size_t j = 0; j < a.rows(); ++j)
    for (std::size_t k = 0; k < a.cols(); ++k)
      out(j, k) = a(j, k);
  return out;
}
gtsam::Vector EigenVector(const Vector &v) {
  gtsam::Vector out(v.size());
  for (std::size_t j = 0; j < v.size(); ++j)
    out[j] = v[j];
  return out;
}

class FactorGraphSolver {
public:
  explicit FactorGraphSolver(const Problem &p) : horizon_(p.stages.size()) {
    using gtsam::Symbol;
    using gtsam::noiseModel::Constrained;
    using gtsam::noiseModel::Unit;
    const auto eye = [](int n) { return gtsam::Matrix::Identity(n, n).eval(); };
    graph_.add(Symbol('x', 0), eye(p.initial_state.size()),
               EigenVector(p.initial_state),
               Constrained::All(p.initial_state.size()));
    for (std::size_t i = 0; i <= horizon_; ++i) {
      const int n = p.Q[i].rows();
      const int m = i < horizon_ ? p.stages[i].R.rows() : 0;
      gtsam::Matrix h = gtsam::Matrix::Zero(n + m, n + m);
      gtsam::Vector gradient(n + m);
      h.topLeftCorner(n, n) = EigenMatrix(p.Q[i]);
      gradient.head(n) = EigenVector(p.q[i]);
      if (i < horizon_) {
        const auto &s = p.stages[i];
        h.bottomRightCorner(m, m) = EigenMatrix(s.R);
        h.topRightCorner(n, m) = EigenMatrix(s.M);
        h.bottomLeftCorner(m, n) = h.topRightCorner(n, m).transpose();
        gradient.tail(m) = EigenVector(s.r);
        graph_.add(Symbol('x', i), EigenMatrix(s.A), Symbol('u', i),
                   EigenMatrix(s.B), Symbol('x', i + 1), -eye(s.A.rows()),
                   -EigenVector(s.c), Constrained::All(s.A.rows()));
        if (s.C.rows())
          graph_.add(Symbol('x', i), EigenMatrix(s.C), Symbol('u', i),
                     EigenMatrix(s.D), -EigenVector(s.d),
                     Constrained::All(s.C.rows()));
      }
      // H = L L'; J=L' and b=-L^{-1}g encode the full quadratic,
      // including cross terms and linear gradients, up to an additive constant.
      // The authors' supplied graph builder only handles time-invariant,
      // diagonal state/control costs. Extend only the input conversion here;
      // use their unmodified backward elimination and forward solve below.
      const auto [jacobian, rhs] =
          clqr::benchmark::MakeQuadraticFactor(h, gradient);
      if (m)
        graph_.add(Symbol('x', i), jacobian.leftCols(n), Symbol('u', i),
                   jacobian.rightCols(m), rhs, Unit::Create(n + m));
      else
        graph_.add(Symbol('x', i), jacobian, rhs, Unit::Create(n));
      const Matrix &e = i < horizon_ ? p.stages[i].E : p.terminal_E;
      const Vector &offset = i < horizon_ ? p.stages[i].e : p.terminal_e;
      if (e.rows())
        graph_.add(Symbol('x', i), EigenMatrix(e), -EigenVector(offset),
                   Constrained::All(e.rows()));
    }
  }
  void Solve() {
    result_ = ecLqr::fgSolFromBn(ecLqr::BnFromGfg(graph_, horizon_));
  }
  Trajectory Result() const {
    Trajectory out;
    for (std::size_t i = 0; i <= horizon_; ++i) {
      const auto &x = result_.at(gtsam::Symbol('x', i));
      Vector state(x.size());
      for (int j = 0; j < x.size(); ++j)
        state[j] = x[j];
      out.first.push_back(std::move(state));
      if (i < horizon_) {
        const auto &u = result_.at(gtsam::Symbol('u', i));
        Vector control(u.size());
        for (int j = 0; j < u.size(); ++j)
          control[j] = u[j];
        out.second.push_back(std::move(control));
      }
    }
    return out;
  }

private:
  std::size_t horizon_;
  gtsam::GaussianFactorGraph graph_;
  gtsam::VectorValues result_;
};
#endif

#ifdef CLQR_BENCHMARK_GEN_RICCATI
class RiccatiSolver {
public:
  explicit RiccatiSolver(const Problem &p)
      : p_(p), ocp_(MakeOcp(p)), solver_(ocp_), ux_(sum(ocp_.nu + ocp_.nx), 1),
        lam_(sum(ocp_.ng + ocp_.nx) - ocp_.nx[0], 1) {
    solver_.it_ref = false;
  }
  void Solve() {
    const int status = solver_.solve_pd_sys_normal(ocp_, 0.0, ux_[0], lam_[0]);
    if (status != 0)
      throw std::runtime_error("gen_riccati status=" + std::to_string(status));
  }
  Trajectory Result() const {
    Trajectory out;
    int offset = 0;
    for (int i = 0; i < ocp_.K; ++i) {
      Vector u(ocp_.nu[i]), x(ocp_.nx[i]);
      for (int j = 0; j < ocp_.nu[i]; ++j)
        u[j] = ux_[0].at(offset++);
      for (int j = 0; j < ocp_.nx[i]; ++j)
        x[j] = ux_[0].at(offset++);
      if (i + 1 < ocp_.K)
        out.second.push_back(std::move(u));
      out.first.push_back(std::move(x));
    }
    return out;
  }
  Multipliers DualResult() const {
    Multipliers out;
    out.initial = Vector(ocp_.nx[0]);
    for (int j = 0; j < ocp_.nx[0]; ++j)
      out.initial[j] = lam_[0].at(j);
    for (int i = 0; i < ocp_.K; ++i) {
      int offset = solver_.g_offs[i] + (i == 0 ? ocp_.nx[0] : 0);
      if (i + 1 < ocp_.K) {
        const auto &s = p_.stages[i];
        Vector dynamics(ocp_.nx[i + 1]), mixed(s.C.rows()), state(s.E.rows());
        // The reference uses A*x+B*u+c-x_next=0, opposite to our dynamics sign.
        for (std::size_t j = 0; j < dynamics.size(); ++j)
          dynamics[j] = -lam_[0].at(solver_.dyn_eq_offs[i] + j);
        for (std::size_t j = 0; j < mixed.size(); ++j)
          mixed[j] = lam_[0].at(offset++);
        for (std::size_t j = 0; j < state.size(); ++j)
          state[j] = lam_[0].at(offset++);
        out.dynamics.push_back(std::move(dynamics));
        out.mixed.push_back(std::move(mixed));
        out.state.push_back(std::move(state));
      } else {
        out.terminal = Vector(p_.terminal_e.size());
        for (std::size_t j = 0; j < out.terminal.size(); ++j)
          out.terminal[j] = lam_[0].at(offset++);
      }
    }
    return out;
  }

private:
  static gen_riccati::COCP MakeOcp(const Problem &p) {
    const std::size_t N = p.stages.size();
    gen_riccati::NumericVector nx(N + 1), nu(N + 1), ng(N + 1);
    for (std::size_t i = 0; i <= N; ++i) {
      nx[i] = p.Q[i].rows();
      nu[i] = i < N ? p.stages[i].R.rows() : 0;
      ng[i] = i < N ? p.stages[i].C.rows() + p.stages[i].E.rows()
                    : p.terminal_E.rows();
      if (i == 0)
        ng[i] += nx[i];
    }
    gen_riccati::COCP ocp(N + 1, nu, nx, ng);
    for (std::size_t i = 0; i <= N; ++i) {
      auto h = ocp.RSQrqt[i];
      auto eq = ocp.Ggt[i];
      for (int j = 0; j <= nu[i] + nx[i]; ++j) {
        for (int k = 0; k < nu[i] + nx[i]; ++k)
          h.at(j, k) = 0;
        for (int k = 0; k < ng[i]; ++k)
          eq.at(j, k) = 0;
      }
      for (int j = 0; j < nx[i]; ++j) {
        for (int k = 0; k < nx[i]; ++k)
          h.at(nu[i] + j, nu[i] + k) = p.Q[i](j, k);
        h.at(nu[i] + nx[i], nu[i] + j) = p.q[i][j];
      }
      int col = 0;
      if (i == 0) {
        for (int j = 0; j < nx[i]; ++j) {
          eq.at(nu[i] + j, col) = 1;
          eq.at(nu[i] + nx[i], col++) = -p.initial_state[j];
        }
      }
      if (i < N) {
        const auto &s = p.stages[i];
        for (int j = 0; j < nu[i]; ++j) {
          for (int k = 0; k < nu[i]; ++k)
            h.at(j, k) = s.R(j, k);
          for (int k = 0; k < nx[i]; ++k) {
            h.at(j, nu[i] + k) = s.M(k, j);
            h.at(nu[i] + k, j) = s.M(k, j);
          }
          h.at(nu[i] + nx[i], j) = s.r[j];
        }
        auto dynamics = ocp.BAbt[i];
        for (int k = 0; k < nx[i + 1]; ++k) {
          for (int j = 0; j < nu[i]; ++j)
            dynamics.at(j, k) = s.B(k, j);
          for (int j = 0; j < nx[i]; ++j)
            dynamics.at(nu[i] + j, k) = s.A(k, j);
          dynamics.at(nu[i] + nx[i], k) = s.c[k];
        }
        for (std::size_t k = 0; k < s.C.rows(); ++k, ++col) {
          for (int j = 0; j < nu[i]; ++j)
            eq.at(j, col) = s.D(k, j);
          for (int j = 0; j < nx[i]; ++j)
            eq.at(nu[i] + j, col) = s.C(k, j);
          eq.at(nu[i] + nx[i], col) = s.d[k];
        }
      }
      const Matrix &e = i < N ? p.stages[i].E : p.terminal_E;
      const Vector &offset = i < N ? p.stages[i].e : p.terminal_e;
      for (std::size_t k = 0; k < e.rows(); ++k, ++col) {
        for (int j = 0; j < nx[i]; ++j)
          eq.at(nu[i] + j, col) = e(k, j);
        eq.at(nu[i] + nx[i], col) = offset[k];
      }
    }
    return ocp;
  }
  const Problem &p_;
  gen_riccati::COCP ocp_;
  gen_riccati::OCPLSRiccati solver_;
  gen_riccati::FatropMemoryVecBF ux_, lam_;
};
#endif

#if defined(CLQR_BENCHMARK_LAINE) || defined(CLQR_BENCHMARK_LAINE_CORRECTED)
Trajectory EigenTrajectory(const std::vector<Eigen::VectorXd> &x,
                           const std::vector<Eigen::VectorXd> &u) {
  Trajectory out;
  auto copy = [](const auto &source, auto &target) {
    for (const auto &v : source) {
      Vector w(v.size());
      for (Eigen::Index j = 0; j < v.size(); ++j)
        w[j] = v[j];
      target.push_back(std::move(w));
    }
  };
  copy(x, out.first);
  copy(u, out.second);
  return out;
}
#endif

#ifdef CLQR_BENCHMARK_LAINE
class LaineAuthorSolver {
public:
  explicit LaineAuthorSolver(const Problem &p)
      : original_(p), problem_(clqr::benchmark::reference::conversion::Convert(p)),
        trajectory_(clqr::benchmark::reference::author::MakeTrajectory(problem_, false)) {}
  void Solve() { clqr::benchmark::reference::author::Solve(*trajectory_); }
  Trajectory Result() const {
    return EigenTrajectory(trajectory_->open_loop_states,
                           trajectory_->open_loop_controls);
  }
  Multipliers DualResult() const {
    return clqr::benchmark::reference::author::CopyMultipliers(original_, *trajectory_);
  }
private:
  const Problem &original_;
  clqr::benchmark::reference::Problem problem_;
  std::unique_ptr<trajectory::Trajectory> trajectory_;
};
#endif

#ifdef CLQR_BENCHMARK_LAINE_CORRECTED
class LaineCorrectedSolver {
public:
  explicit LaineCorrectedSolver(const Problem &p)
      : original_(p), problem_(corrected_laine_tomlin::conversion::Convert(p)) {}
  void Solve() {
    result_ = corrected_laine_tomlin::Solve(problem_);
    if (result_.status != corrected_laine_tomlin::Status::kOptimal)
      throw std::runtime_error(result_.message);
  }
  Trajectory Result() const {
    return EigenTrajectory(result_.states, result_.controls);
  }
  Multipliers DualResult() const {
    auto copy = [](const auto &v) {
      Vector out(v.size());
      for (Eigen::Index i = 0; i < v.size(); ++i)
        out[i] = v[i];
      return out;
    };
    Multipliers out;
    // CLQR's benchmark uses x[0]-initial_state and x[t+1]-A*x-B*u-c,
    // opposite to this solver's documented boundary/dynamics signs.
    out.initial = clqr::Scale(copy(result_.initial), clqr::Scalar{-1});
    out.terminal = copy(result_.terminal);
    for (std::size_t t = 0; t < original_.stages.size(); ++t) {
      const auto &s = original_.stages[t];
      out.dynamics.push_back(clqr::Scale(copy(result_.dynamics[t]), clqr::Scalar{-1}));
      out.mixed.push_back(copy(result_.constraints[t].head(s.C.rows())));
      out.state.push_back(copy(result_.constraints[t].tail(s.E.rows())));
    }
    return out;
  }
private:
  const Problem &original_;
  corrected_laine_tomlin::Problem problem_;
  corrected_laine_tomlin::Result result_;
};
#endif

template <class Solver>
void Run(const clqr::benchmark::PaperCase &c,
         const clqr::benchmark::ScalingProblem &data, const char *backend,
         int repeats, double min_seconds, std::uint64_t seed,
         std::size_t index, std::size_t count) {
  const auto started = Clock::now();
  const auto progress = [&](const std::string &phase) {
    std::cerr << "[case " << index + 1 << '/' << count << "] " << backend
              << " N=" << c.horizon << " n=" << c.n << " m=" << c.m
              << " " << phase << " (elapsed "
              << std::chrono::duration_cast<std::chrono::seconds>(
                     Clock::now() - started).count()
              << "s)\n";
  };
  const auto more_samples = [&](std::size_t count, double total_ms) {
    return repeats > 0 ? count < static_cast<std::size_t>(repeats)
                       : total_ms < 1000 * min_seconds;
  };
  const auto sample_list = [](const std::vector<double>& samples) {
    std::ostringstream out;
    out << std::setprecision(12);
    for (std::size_t i = 0; i < samples.size(); ++i) {
      if (i) out << ';';
      out << samples[i];
    }
    return out.str();
  };
  progress("setup");
  std::cout << backend << ',' << c.family << ',' << c.horizon << ',' << c.n
            << ',' << c.m << ',' << c.mixed << ',' << c.state << ',' << seed
            << ',';
  try {
    const auto begin = Clock::now();
    std::optional<Solver> solver(std::in_place, data.problem);
    const double setup_ms = Milliseconds(begin);
    progress("warmup (1 solve)");
    solver->Solve();
    std::vector<double> times, kernel_times, setup_solve_times;
    const std::string sampling = repeats > 0 ? std::to_string(repeats) + " calls"
        : "at least " + std::to_string(min_seconds) + " measured seconds";
    progress("timing prepared solves: " + sampling);
    double total_ms = 0;
    while (more_samples(times.size(), total_ms)) {
      const auto start = Clock::now();
      solver->Solve();
      times.push_back(Milliseconds(start));
      total_ms += times.back();
      if constexpr (requires { solver->KernelMs(); })
        kernel_times.push_back(solver->KernelMs());
    }
    progress("validation");
    const auto result = solver->Result();
    std::optional<Multipliers> dual;
    if constexpr (requires { solver->DualResult(); })
      dual = solver->DualResult();
    const Residuals residuals = Audit(data.problem, result, dual);
    // A linear-horizon certificate using the known planted dual. This checks
    // every backend's returned primal, including primal-only references,
    // without an expensive global factorization or fabricated solver duals.
    const double planted_dual_stationarity =
        Audit(data.problem, result, data.dual).stationarity;
    const double error = std::max(
        clqr::benchmark::MaxDifference(result.first, data.states),
        clqr::benchmark::MaxDifference(result.second, data.controls));
    const double dual_error = dual
        ? clqr::benchmark::MaxDifference(*dual, data.dual)
        : std::numeric_limits<double>::quiet_NaN();
    const long double reference = clqr::benchmark::OriginalObjective(
        data.problem, data.states, data.controls);
    const long double objective = clqr::benchmark::OriginalObjective(
        data.problem, result.first, result.second);
    const long double obj_error =
        std::abs(objective - reference) / std::max(1.0L, std::abs(reference));
    const bool valid = std::isfinite(error) && std::isfinite(obj_error) &&
                       error < 1e-6 && obj_error < 1e-8 &&
                       residuals.feasibility < 1e-8 &&
                       (!dual || residuals.stationarity < 1e-8) &&
                       planted_dual_stationarity < 1e-8;
    // Representation preparation can itself involve numerical work (e.g. cost
    // square roots for the factor graph). Also report a repeated setup+solve
    // measurement so that the prepared-representation timing does not hide it.
    // Destruction and extraction into a common output container are excluded.
    // Do not keep a second (potentially multi-GiB) workspace resident while
    // measuring fresh allocation. The validation outputs above own their data.
    solver.reset();
    progress("timing setup+solve: " + sampling);
    total_ms = 0;
    while (more_samples(setup_solve_times.size(), total_ms)) {
      const auto start = Clock::now();
      Solver fresh(data.problem);
      fresh.Solve();
      setup_solve_times.push_back(Milliseconds(start));
      total_ms += setup_solve_times.back();
    }
    const auto solve_samples = sample_list(times);
    const auto setup_samples = sample_list(setup_solve_times);
    const auto kernel_samples = sample_list(kernel_times);
    std::sort(times.begin(), times.end());
    std::sort(kernel_times.begin(), kernel_times.end());
    std::sort(setup_solve_times.begin(), setup_solve_times.end());
    std::cout << (valid ? "ok" : "inaccurate") << ',' << times.size() << ','
              << setup_ms << ',' << times[times.size() / 2] << ','
              << (times.size() > 1 ? times[times.size() / 10]
                                  : std::numeric_limits<double>::quiet_NaN()) << ','
              << (times.size() > 1 ? times[(times.size() - 1) * 9 / 10]
                                  : std::numeric_limits<double>::quiet_NaN()) << ',' << error << ','
              << objective << ',' << obj_error << ','
              << (kernel_times.empty()
                      ? std::numeric_limits<double>::quiet_NaN()
                      : kernel_times[kernel_times.size() / 2])
              << ',' << residuals.feasibility << ',' << residuals.stationarity
              << ','
              << (dual ? std::max(residuals.feasibility, residuals.stationarity)
                       : std::numeric_limits<double>::quiet_NaN())
              << ',' << setup_solve_times[setup_solve_times.size() / 2]
              << ',' << planted_dual_stationarity << ',' << dual_error
              << ',' << setup_solve_times.size() << ',' << solve_samples
              << ',' << setup_samples << ',' << kernel_samples << '\n';
    progress(valid ? "DONE status=ok" : "DONE status=inaccurate");
  } catch (const std::exception &e) {
    std::string message = e.what();
    for (char &ch : message)
      if (ch == ',' || ch == '\n')
        ch = ' ';
    const bool unsupported =
        std::string(backend) == "clqr_cuda" &&
        message.find("exceeding device shared-memory resources") !=
            std::string::npos;
    std::cout << (unsupported ? "unsupported" : "failed")
              << ",0,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,,,\n# "
              << message << '\n';
    progress(std::string("DONE status=") + (unsupported ? "unsupported: " : "failed: ") +
             message);
  }
  std::cout.flush();
}
} // namespace

#ifdef CLQR_BENCHMARK_ADVERSARIAL
namespace {
using clqr::test::adversarial::TestCase;

template <class Solver>
void RunAdversarial(const TestCase &c, const char *backend, int repeats) {
  const auto &p = c.problem;
  std::size_t n = 0, m = 0;
  for (const auto &q : p.Q)
    n = std::max(n, q.rows());
  for (const auto &s : p.stages)
    m = std::max(m, s.R.rows());
  std::cout << backend << ',' << c.name << ',' << p.stages.size() << ','
            << n << ',' << m << ',' << clqr::StatusName(c.cpu_status) << ',';
  try {
    // Malformed shapes are a public-API validation test, not a valid input to
    // the reference's unchecked packed-matrix interface.
    if (c.cpu_status == clqr::SolveStatus::kInvalidInput &&
        std::string(backend) != "clqr_cpu")
      throw std::runtime_error("unsupported: malformed input shape");
    Solver solver(p);
    const auto warmup_start = Clock::now();
    int warmed = 0;
    while (warmed < 3 || Milliseconds(warmup_start) < 50.0) {
      solver.Solve();
      ++warmed;
    }
    // Batch sub-millisecond solves to avoid timer overhead dominating tiny
    // fixtures. Every invocation refactors; this is not a cached-RHS solve.
    const int batch = std::clamp(
        static_cast<int>(warmed / Milliseconds(warmup_start)), 1, 10000);
    std::vector<double> times;
    for (int i = 0; i < repeats; ++i) {
      const auto start = Clock::now();
      for (int j = 0; j < batch; ++j)
        solver.Solve();
      times.push_back(Milliseconds(start) / batch);
    }
    std::sort(times.begin(), times.end());
    const auto trajectory = solver.Result();
    std::optional<Multipliers> dual;
    if constexpr (requires { solver.DualResult(); })
      dual = solver.DualResult();
    const auto residuals = Audit(p, trajectory, dual);
    clqr::test::adversarial::KktPoint point;
    point.states = trajectory.first;
    point.controls = trajectory.second;
    std::string worst;
    double unit_kkt = std::numeric_limits<double>::quiet_NaN();
    double audited_stationarity = std::numeric_limits<double>::quiet_NaN();
    if (dual) {
      point.initial_multiplier = dual->initial;
      point.dynamics_multipliers = dual->dynamics;
      point.mixed_multipliers = dual->mixed;
      point.state_multipliers = dual->state;
      point.terminal_state_multiplier = dual->terminal;
      unit_kkt = clqr::test::adversarial::MaxKktResidual(p, point, &worst);
    }
#if defined(CLQR_BENCHMARK_LAINE) || defined(CLQR_BENCHMARK_GTSAM) || defined(CLQR_BENCHMARK_LAINE_CORRECTED)
    else {
      clqr::benchmark::reference::Trajectory audited;
      for (const auto &x : trajectory.first)
        audited.states.push_back(clqr::benchmark::reference::conversion::Convert(x));
      for (const auto &u : trajectory.second)
        audited.controls.push_back(clqr::benchmark::reference::conversion::Convert(u));
      audited_stationarity = clqr::benchmark::reference::audit::Audit(
          clqr::benchmark::reference::conversion::Convert(p), audited).second;
    }
#endif
    double dense_difference = std::numeric_limits<double>::quiet_NaN();
    if (c.dense_reference) {
      const auto dense = clqr::test::adversarial::SolveDenseKkt(p);
      dense_difference =
          clqr::test::adversarial::MaxPrimalDifference(point, dense);
    }
    // Report every residual, even for fixtures whose unit test does not impose
    // a full KKT threshold. A poor numerical result is data, not a process error.
    const double kkt =
        dual ? std::max(residuals.feasibility, residuals.stationarity)
             : std::numeric_limits<double>::quiet_NaN();
    const bool finite =
        std::isfinite(residuals.feasibility) &&
        std::isfinite(dual ? residuals.stationarity : audited_stationarity);
    std::cout << (finite ? "returned" : "nonfinite") << ','
              << repeats << ',' << batch << ',' << times[times.size() / 2]
              << ',' << times[times.size() / 10] << ','
              << times[(times.size() - 1) * 9 / 10] << ','
              << residuals.feasibility << ',' << residuals.stationarity << ','
              << kkt << ',' << unit_kkt << ',' << audited_stationarity << ','
              << dense_difference << ','
              << worst << ",\n";
  } catch (const std::exception &error) {
    std::string message = error.what();
    for (char &ch : message)
      if (ch == ',' || ch == '\n')
        ch = ' ';
    std::cout << "rejected,0,0,nan,nan,nan,nan,nan,nan,nan,nan,nan,,"
              << message << '\n';
  }
  std::cout.flush();
}

int AdversarialMain(int argc, char **argv) {
  std::string name, backend;
  int repeats = 21;
  bool list = false;
#ifdef CLQR_BENCHMARK_INDEPENDENT_FIXTURES
  bool independent = false;
#endif
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
#ifdef CLQR_BENCHMARK_INDEPENDENT_FIXTURES
    if (arg == "--independent") {
      independent = true;
      continue;
    }
#endif
    if (arg == "--list") {
      list = true;
      continue;
    }
    if (i + 1 == argc)
      throw std::invalid_argument("missing option value");
    if (arg == "--case")
      name = argv[++i];
    else if (arg == "--backend")
      backend = argv[++i];
    else if (arg == "--repeats")
      repeats = std::stoi(argv[++i]);
    else
      throw std::invalid_argument("unknown option: " + arg);
  }
  if (repeats < 1)
    throw std::invalid_argument("repeats must be positive");
  std::vector<std::string> backends = {"clqr_cpu"};
#ifdef CLQR_BENCHMARK_GEN_RICCATI
  backends.push_back("gen_riccati");
#endif
#ifdef CLQR_BENCHMARK_GTSAM
  backends.push_back("factor_graph");
#endif
#ifdef CLQR_BENCHMARK_LAINE
  backends.push_back("laine_author");
#endif
#ifdef CLQR_BENCHMARK_LAINE_CORRECTED
  backends.push_back("laine_corrected");
#endif
  if (!backend.empty() &&
      std::find(backends.begin(), backends.end(), backend) == backends.end())
    throw std::invalid_argument("unknown backend: " + backend);
  if (backend.empty() && !list) {
    if (backends.size() != 1)
      throw std::invalid_argument("select exactly one backend with --backend");
    backend = backends.front();
  }
#ifndef CLQR_BENCHMARK_GEN_RICCATI
  if (backend == "gen_riccati")
    throw std::invalid_argument("gen_riccati is not linked in this build");
#endif
  auto cases = clqr::test::adversarial::StandardCases();
  const auto extended = clqr::test::adversarial::ExtendedCases();
  cases.insert(cases.end(), extended.begin(), extended.end());
#ifdef CLQR_BENCHMARK_INDEPENDENT_FIXTURES
  if (independent) {
    cases.clear();
    for (bool uniform : {false, true})
      for (unsigned seed = 0; seed < 128; ++seed)
        cases.push_back({std::string(uniform ? "uniform-" : "varying-") +
                             std::to_string(seed),
                         corrected_laine_tomlin::conversion::ToClqr(
                             corrected_laine_tomlin::test::RandomProblem(seed, uniform))});
  }
#endif
  if (!list)
    std::cout << "backend,case,N,n,m,expected,status,repeats,batch,median_ms,"
                 "p10_ms,p90_ms,feasibility_inf,stationarity_inf,kkt_inf,"
                 "unit_kkt_inf,audited_stationarity_inf,dense_difference,"
                 "worst,diagnostic\n"
              << std::setprecision(12);
  bool found = false;
  for (const auto &c : cases) {
    if (!name.empty() && c.name != name)
      continue;
    found = true;
    if (list) {
      std::cout << c.name << '\n';
      continue;
    }
    if (backend.empty() || backend == "clqr_cpu") {
      RunAdversarial<CpuSolver>(c, "clqr_cpu", repeats);
    }
#ifdef CLQR_BENCHMARK_GEN_RICCATI
    if (backend.empty() || backend == "gen_riccati")
      RunAdversarial<RiccatiSolver>(c, "gen_riccati", repeats);
#endif
#ifdef CLQR_BENCHMARK_GTSAM
    if (backend.empty() || backend == "factor_graph")
      RunAdversarial<FactorGraphSolver>(c, "factor_graph", repeats);
#endif
#ifdef CLQR_BENCHMARK_LAINE
    if (backend.empty() || backend == "laine_author")
      RunAdversarial<LaineAuthorSolver>(c, "laine_author", repeats);
#endif
#ifdef CLQR_BENCHMARK_LAINE_CORRECTED
    if (backend.empty() || backend == "laine_corrected")
      RunAdversarial<LaineCorrectedSolver>(c, "laine_corrected", repeats);
#endif
  }
  if (!found)
    throw std::invalid_argument("unknown case: " + name);
  return 0;
}
} // namespace
#endif

int main(int argc, char **argv) {
  static_assert(sizeof(clqr::Scalar) == sizeof(double),
                "paper comparisons require FP64");
#ifdef CLQR_BENCHMARK_ADVERSARIAL
  return AdversarialMain(argc, argv);
#endif
  std::string suite = "smoke";
  int repeats = 0;
  double min_seconds = 1;
  int case_index = -1;
  std::string backend;
  std::uint64_t seed = 20260907;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (i + 1 == argc)
      throw std::invalid_argument("missing option value");
    if (option == "--suite")
      suite = argv[++i];
    else if (option == "--repeats") {
      repeats = std::stoi(argv[++i]);
      if (repeats < 1)
        throw std::invalid_argument("repeats must be positive");
    } else if (option == "--min-seconds")
      min_seconds = std::stod(argv[++i]);
    else if (option == "--seed")
      seed = std::stoull(argv[++i]);
    else if (option == "--backend")
      backend = argv[++i];
    else if (option == "--case-index")
      case_index = std::stoi(argv[++i]);
    else
      throw std::invalid_argument("unknown option: " + option);
  }
  if (!std::isfinite(min_seconds) || min_seconds <= 0)
    throw std::invalid_argument("min-seconds must be finite and positive");
  std::vector<std::string> backends = {
#ifdef CLQR_BENCHMARK_GEN_RICCATI
    "gen_riccati",
#endif
#ifdef CLQR_BENCHMARK_GTSAM
    "factor_graph",
#endif
#ifdef CLQR_BENCHMARK_CUDA
    "clqr_cuda",
#endif
#ifdef CLQR_BENCHMARK_LAINE
    "laine_author",
#endif
#ifdef CLQR_BENCHMARK_LAINE_CORRECTED
    "laine_corrected",
#endif
  };
  if (backends.empty()) backends.push_back("clqr_cpu");
  if (backend.empty()) {
    if (backends.size() != 1)
      throw std::invalid_argument("select exactly one backend with --backend");
    backend = backends.front();
  }
  if (!backend.empty() &&
      std::find(backends.begin(), backends.end(), backend) == backends.end())
    throw std::invalid_argument("backend is not linked in this build: " + backend);
  const auto cases = clqr::benchmark::PaperCases(suite);
  if (case_index < -1 || case_index >= static_cast<int>(cases.size()))
    throw std::invalid_argument("case-index is outside the suite");
  std::cout << "# FP64; exactly one untimed warmup solve; setup includes "
               "representation/storage preparation; "
               "solve includes a fresh factorization and primal solve on every "
               "repetition.\n"
               "# All backends receive identical data. Validation and result "
               "conversion are untimed.\n"
               "# CLQR, generalized Riccati, and both Laine implementations "
               "also recover duals inside timing. Primal-only backends "
               "report no solver-dual KKT measurement.\n"
               "# Setup+solve includes representation construction and fresh "
               "workspace allocation; destruction is excluded.\n"
               "# Missing dual/kernel measurements are nan, not inferred from "
               "other solvers.\n"
               "# planted_dual_stationarity_inf audits each returned primal "
               "with the fixture's known optimal dual, not solver output.\n"
               "# primal_error and dual_error_inf are absolute infinity-norm "
               "errors against the known planted optimum, in original coordinates.\n"
               "# Samples are individually timed; sample-list columns retain "
               "chronological times in ms separated by semicolons. "
               "Timing quantiles are unavailable for a single sample.\n";
  std::cout << "# min_seconds=" << min_seconds << "; fixed_repeats=" << repeats
            << " (0 means duration-based); each timing phase uses its own budget.\n"
               "backend,family,N,n,m,mixed_rows,state_rows,seed,status,repeats,"
               "setup_ms,"
               "median_ms,p10_ms,p90_ms,primal_error,objective,relative_"
               "objective_error,kernel_ms,"
               "feasibility_inf,stationarity_inf,kkt_inf,setup_solve_ms,"
               "planted_dual_stationarity_inf,dual_error_inf,setup_solve_repeats,"
               "solve_samples_ms,setup_solve_samples_ms,kernel_samples_ms\n";
  std::cout << std::setprecision(12);
  std::cout.flush();
  for (std::size_t index = 0; index < cases.size(); ++index) {
    if (case_index >= 0 && index != static_cast<std::size_t>(case_index))
      continue;
    const auto &c = cases[index];
    std::cerr << "[case " << index + 1 << '/' << cases.size()
              << "] generating fixture N=" << c.horizon << " n=" << c.n
              << " m=" << c.m << '\n';
    const auto data = clqr::benchmark::MakeScalingProblem(
        c.horizon, c.n, c.m, c.mixed, c.state, seed);
    if (backend == "clqr_cpu")
      Run<CpuSolver>(c, data, "clqr_cpu", repeats, min_seconds, seed, index, cases.size());
#ifdef CLQR_BENCHMARK_GEN_RICCATI
    if (backend == "gen_riccati")
      Run<RiccatiSolver>(c, data, "gen_riccati", repeats, min_seconds, seed, index, cases.size());
#endif
#ifdef CLQR_BENCHMARK_GTSAM
    if (backend == "factor_graph")
      Run<FactorGraphSolver>(c, data, "factor_graph", repeats, min_seconds, seed, index, cases.size());
#endif
#ifdef CLQR_BENCHMARK_CUDA
    if (backend == "clqr_cuda")
      Run<CudaSolver>(c, data, "clqr_cuda", repeats, min_seconds, seed, index, cases.size());
#endif
#ifdef CLQR_BENCHMARK_LAINE
    if (backend == "laine_author")
      Run<LaineAuthorSolver>(c, data, "laine_author", repeats, min_seconds, seed, index, cases.size());
#endif
#ifdef CLQR_BENCHMARK_LAINE_CORRECTED
    if (backend == "laine_corrected")
      Run<LaineCorrectedSolver>(c, data, "laine_corrected", repeats, min_seconds, seed, index, cases.size());
#endif
  }
  // Keep all numerical outcomes in the CSV, without turning them into test
  // failures. Regression tests and sanitizer checks are separate executables.
  return 0;
}
