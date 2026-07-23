#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "adversarial_test_support.h"

namespace {

using clqr::Problem;
using clqr::Scalar;
using clqr::SolutionView;
using clqr::SolveStatus;
using clqr::Vector;
using clqr::Workspace;
using clqr::test::adversarial::DensePrimal;
using clqr::test::adversarial::KktPoint;
using clqr::test::adversarial::TestCase;

void Expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void NonfiniteOracleCase() {
  const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  const Vector nonfinite{nan};
  Expect(std::isinf(clqr::test::adversarial::TestMaxAbs(nonfinite)),
         "nonfinite max-absolute residual is infinite");

  const Problem problem = clqr::test::adversarial::UniformProblem(
      9000, 0, 1, 0, 0, clqr::test::adversarial::Pattern::kNone);
  KktPoint point;
  point.states = {nonfinite};
  point.initial_multiplier = nonfinite;
  Expect(std::isinf(
             clqr::test::adversarial::MaxPrimalResidual(problem, point)),
         "nonfinite primal residual is infinite");
  Expect(std::isinf(clqr::test::adversarial::MaxKktResidual(problem, point)),
         "nonfinite KKT residual is infinite");
  DensePrimal dense;
  dense.states = {Vector{Scalar{0}}};
  Expect(std::isinf(
             clqr::test::adversarial::MaxPrimalDifference(point, dense)),
         "nonfinite dense-oracle difference is infinite");
}

void RunCase(const TestCase &test_case, Workspace *reusable_workspace) {
  const Problem &problem = test_case.problem;
  reusable_workspace->Reserve(problem);
  const SolutionView solution = clqr::Solve(problem, *reusable_workspace);
  Expect(solution.status == test_case.cpu_status,
         test_case.name + " status: " + solution.message);
  if (solution.status != SolveStatus::kOptimal) {
    Expect(std::string(solution.message) != clqr::StatusName(solution.status),
           test_case.name + " preserves its detailed diagnostic");
    std::cout << "case: " << test_case.name << " passed ("
              << clqr::StatusName(solution.status) << ")\n";
    return;
  }

  const KktPoint point = clqr::test::adversarial::CopyCpuSolution(solution);
  const Scalar primal_residual =
      clqr::test::adversarial::MaxPrimalResidual(problem, point);
  Expect(primal_residual <= clqr::test::adversarial::kPrimalTolerance *
                                test_case.tolerance_scale,
         test_case.name +
             " primal residual=" + std::to_string(primal_residual));
  if (test_case.check_full_kkt) {
    std::string worst;
    const Scalar residual =
        clqr::test::adversarial::MaxKktResidual(problem, point, &worst);
    Expect(residual <= clqr::test::adversarial::kKktTolerance *
                           test_case.tolerance_scale *
                           test_case.kkt_tolerance_scale,
           test_case.name + " KKT residual=" + std::to_string(residual) +
               " in " + worst);
  }
  if (test_case.dense_reference) {
    const DensePrimal dense = clqr::test::adversarial::SolveDenseKkt(problem);
    const Scalar difference =
        clqr::test::adversarial::MaxPrimalDifference(point, dense);
    Expect(difference <= clqr::test::adversarial::kDenseTolerance *
                             test_case.tolerance_scale,
           test_case.name +
               " dense-KKT primal difference=" + std::to_string(difference));
  }
  std::cout << "case: " << test_case.name << " passed\n";
}

} // namespace

int main(int argc, char **argv) {
  bool extended = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--extended") {
      extended = true;
    } else {
      std::cerr << "unknown argument: " << argv[i] << '\n';
      return 2;
    }
  }

  NonfiniteOracleCase();
  std::vector<TestCase> cases = clqr::test::adversarial::StandardCases();
  if (extended) {
    std::vector<TestCase> more = clqr::test::adversarial::ExtendedCases();
    cases.insert(cases.end(), more.begin(), more.end());
  }
  Workspace reusable_workspace;
  for (const TestCase &test_case : cases)
    RunCase(test_case, &reusable_workspace);
  std::cout << "all " << cases.size() << " adversarial CPU cases passed\n";
  return 0;
}
