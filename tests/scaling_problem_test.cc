#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "benchmarks/scaling_problem.h"
#include "benchmarks/paper_cases.h"
#include "tests/adversarial_test_support.h"

namespace {
void CheckKkt(const clqr::Problem &p, const clqr::SolutionView &result,
              double tolerance, const std::string &name) {
  if (result.status != clqr::SolveStatus::kOptimal)
    throw std::runtime_error(result.message);
  std::string equation;
  const auto point = clqr::test::adversarial::CopyCpuSolution(result);
  const double residual =
      clqr::test::adversarial::MaxKktResidual(p, point, &equation);
  if (!std::isfinite(residual) || residual > tolerance)
    throw std::runtime_error(name + " KKT " + equation + ": " +
                             std::to_string(residual));
}
} // namespace

int main() {
  static_assert(sizeof(clqr::Scalar) == sizeof(double),
                "Paper fixtures require FP64");
  constexpr double tolerance = 2e-8;
  const auto cases = clqr::benchmark::PaperCases("all");
  if (cases.size() != 15)
    throw std::runtime_error("paper sweep case count");
  const auto horizons = clqr::benchmark::PaperCases("horizon");
  if (horizons.size() != 11)
    throw std::runtime_error("paper horizon count");
  for (std::size_t i = 0; i < horizons.size(); ++i)
    if (horizons[i].horizon != (std::size_t{32} << i))
      throw std::runtime_error("paper horizon sweep skips a power of two");
  for (const auto &c : cases)
    if (c.n % 8 || c.m != c.n / 2 || c.mixed != c.n / 8 ||
        c.state != c.n / 4 ||
        (c.family != "horizon" && c.family != "dimension"))
      throw std::runtime_error("paper dimension ratios");
  const auto reference = clqr::benchmark::MakeScalingProblem(2, 8, 4, 1, 2);
  auto changed_dual = reference.dual;
  changed_dual.dynamics[1][0] += 0.25;
  if (std::abs(clqr::benchmark::MaxDifference(changed_dual, reference.dual) - 0.25) > 1e-15)
    throw std::runtime_error("original-coordinate dual error");
  changed_dual.initial[0] = std::numeric_limits<double>::quiet_NaN();
  if (!std::isinf(clqr::benchmark::MaxDifference(changed_dual, reference.dual)))
    throw std::runtime_error("nonfinite dual error hidden");
  for (const std::size_t n : {4, 8, 16, 32}) {
    const std::size_t m = n / 2;
    for (const std::size_t horizon : {0, 1, 17}) {
      for (int kind = 0; kind < 3; ++kind) {
        const auto data = clqr::benchmark::MakeScalingProblem(
            horizon, n, m, kind == 0 ? 0 : m / 2, kind == 1 ? 0 : m / 2);
        clqr::test::adversarial::KktPoint planted;
        planted.states = data.states;
        planted.controls = data.controls;
        planted.initial_multiplier = data.dual.initial;
        planted.dynamics_multipliers = data.dual.dynamics;
        planted.mixed_multipliers = data.dual.mixed;
        planted.state_multipliers = data.dual.state;
        planted.terminal_state_multiplier = data.dual.terminal;
        if (clqr::test::adversarial::MaxKktResidual(data.problem, planted) >
            1e-12)
          throw std::runtime_error("fixture's planted primal-dual certificate");
        clqr::Workspace workspace;
        workspace.Reserve(data.problem);
        const auto result = clqr::Solve(data.problem, workspace);
        CheckKkt(data.problem, result, tolerance,
                 "N=" + std::to_string(horizon) + " n=" + std::to_string(n) +
                     " kind=" + std::to_string(kind));
        double error = 0;
        for (std::size_t i = 0; i <= horizon; ++i) {
          for (std::size_t j = 0; j < n; ++j)
            error = std::max(error, std::abs(double(result.states[i][j] -
                                                    data.states[i][j])));
          if (i < horizon)
            for (std::size_t j = 0; j < m; ++j)
              error = std::max(error, std::abs(double(result.controls[i][j] -
                                                      data.controls[i][j])));
        }
        const long double reference = clqr::benchmark::OriginalObjective(
            data.problem, data.states, data.controls);
        const double objective_error =
            static_cast<double>(std::abs(result.objective - reference) /
                                std::max(1.0L, std::abs(reference)));
        if (!std::isfinite(error) || !std::isfinite(objective_error) ||
            error > tolerance || objective_error > tolerance) {
          std::cerr << "N=" << horizon << " n=" << n << " kind=" << kind
                    << " primal=" << error << " objective=" << objective_error
                    << '\n';
          return 1;
        }
      }
    }
  }
  // State elimination can make the reduced open-loop dynamics unstable even
  // though the original A is contractive and the optimum is well behaved.
  // Replaying A^T backwards to recover costates used to amplify roundoff to
  // 1e30 on the first case. Reuse feedback in both solve paths. The last case
  // also checks cancellation when all reduced controls have been eliminated.
  for (const auto &data :
       {clqr::benchmark::MakeScalingProblem(512, 16, 8, 0, 6),
        clqr::benchmark::MakeScalingProblem(128, 64, 32, 8, 8),
        clqr::benchmark::MakeScalingProblem(17, 8, 4, 2, 2)}) {
    clqr::Workspace full_workspace;
    full_workspace.Reserve(data.problem);
    const std::string name = "N=" + std::to_string(data.problem.stages.size()) +
                             " n=" + std::to_string(data.problem.Q[0].rows());
    CheckKkt(data.problem, clqr::Solve(data.problem, full_workspace), 1e-8,
             name + " full");
    const auto factorization = clqr::Factor(data.problem);
    clqr::Workspace cached_workspace;
    cached_workspace.Reserve(factorization);
    auto rhs = clqr::ExtractRhs(data.problem);
    CheckKkt(data.problem, clqr::Solve(factorization, rhs, cached_workspace),
             1e-8, name + " cached");
    auto changed = data.problem;
    for (std::size_t i = 0; i < changed.q.size(); ++i)
      for (std::size_t j = 0; j < changed.q[i].size(); ++j)
        changed.q[i][j] += clqr::Scalar{0.01};
    rhs = clqr::ExtractRhs(changed);
    CheckKkt(changed, clqr::Solve(factorization, rhs, cached_workspace), 1e-8,
             name + " changed cached");
    CheckKkt(changed, clqr::Solve(changed, full_workspace), 1e-8,
             name + " changed full");
  }
  std::cout
      << "All scaling fixtures recover the planted optimum and objective.\n";
}
