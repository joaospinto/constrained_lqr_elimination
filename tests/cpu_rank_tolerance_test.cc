#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "benchmarks/scaling_problem.h"
#include "tests/adversarial_test_support.h"

namespace {
namespace adversarial = clqr::test::adversarial;
double maximum_kkt = 0;
std::size_t checked = 0;

void Check(const clqr::Problem &problem, const clqr::SolutionView &solution,
           const std::string &name) {
  if (solution.status != clqr::SolveStatus::kOptimal)
    throw std::runtime_error(name + ": " + solution.message);
  std::string equation;
  const auto point = adversarial::CopyCpuSolution(solution);
  const double residual =
      adversarial::MaxKktResidual(problem, point, &equation);
  if (!std::isfinite(residual) || residual > 1e-8)
    throw std::runtime_error(name + ": " + equation +
                             " KKT=" + std::to_string(residual));
  maximum_kkt = std::max(maximum_kkt, residual);
  ++checked;
}

void CheckTolerances(const clqr::Problem &problem, const std::string &name) {
  for (const double tolerance : {1e-12, 1e-11, 1e-10, 1e-9, 1e-8}) {
    clqr::SolveOptions options;
    options.tolerance = tolerance;
    std::cout << name << " rank tolerance=" << tolerance << '\n';
    {
      clqr::Workspace workspace;
      workspace.Reserve(problem, options);
      Check(problem, clqr::Solve(problem, workspace, options), name + " full");
    }
    {
      const auto factorization = clqr::Factor(problem, options);
      if (factorization.status() != clqr::SolveStatus::kOptimal)
        throw std::runtime_error(name + ": factorization failed");
      clqr::Workspace workspace;
      workspace.Reserve(factorization);
      Check(problem,
            clqr::Solve(factorization, clqr::ExtractRhs(problem), workspace),
            name + " cached");
    }
  }
}
} // namespace

int main() {
  static_assert(sizeof(clqr::Scalar) == sizeof(double));
  for (const std::size_t n : {8, 16, 24, 32, 48, 64}) {
    for (const std::size_t horizon : {32, 128}) {
      const auto data =
          clqr::benchmark::MakeScalingProblem(horizon, n, n / 2, n / 8, n / 4);
      CheckTolerances(data.problem, "N=" + std::to_string(horizon) +
                                        " n=" + std::to_string(n));
    }
  }
  // The same fixture exposed rank-threshold sensitivity in CUDA's scan.
  const auto regression =
      clqr::benchmark::MakeScalingProblem(2048, 32, 16, 4, 8);
  CheckTolerances(regression.problem, "N=2048 n=32");
  for (const auto &test : adversarial::StandardCases()) {
    if (test.name == "redundant-rank-deficient" ||
        test.name == "independently-scaled-rows" ||
        test.name == "more-mixed-than-controls" ||
        test.name == "fully-constrained-nonunique-multipliers" ||
        test.name == "free-fixed-free-state" ||
        test.name == "zero-horizon-terminal")
      CheckTolerances(test.problem, test.name);
  }
  std::cout << checked
            << " CPU full/cached solves passed; maximum KKT=" << maximum_kkt
            << '\n';
}
