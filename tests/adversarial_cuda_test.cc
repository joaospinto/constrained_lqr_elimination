#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "adversarial_test_support.h"
#include "clqr/cuda.h"

namespace {

using clqr::Problem;
using clqr::Scalar;
using clqr::SolveStatus;
using clqr::test::adversarial::DensePrimal;
using clqr::test::adversarial::KktPoint;
using clqr::test::adversarial::TestCase;

void Expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

KktPoint CopyCudaSolution(const clqr::cuda::Solution &source) {
  KktPoint result;
  result.states = source.states;
  result.controls = source.controls;
  result.initial_multiplier = source.initial_multiplier;
  result.dynamics_multipliers = source.dynamics_multipliers;
  result.mixed_multipliers = source.mixed_multipliers;
  result.state_multipliers = source.state_multipliers;
  result.terminal_state_multiplier = source.terminal_state_multiplier;
  result.objective = source.objective;
  return result;
}

void RunCase(const TestCase &test_case, clqr::cuda::Workspace *workspace,
             clqr::cuda::Solution *solution) {
  if (test_case.cuda_status == SolveStatus::kInvalidInput) {
    *solution = clqr::cuda::Solve(test_case.problem);
  } else {
    workspace->Reserve(test_case.problem);
    clqr::cuda::Solve(test_case.problem, *workspace, *solution);
  }
  const bool accurate_success =
      test_case.allow_accurate_cuda_success &&
      solution->status == SolveStatus::kOptimal;
  Expect(solution->status == test_case.cuda_status || accurate_success,
         test_case.name + " status: " + solution->message);
  if (solution->status != SolveStatus::kOptimal) {
    std::cout << "case: " << test_case.name << " passed ("
              << clqr::StatusName(solution->status) << ")\n";
    return;
  }

  if (test_case.name == "free-fixed-free-state") {
    Expect(solution->reduced_state_dimensions == std::vector<int>({1, 0, 1}),
           test_case.name + " reduced state dimensions");
  }
  const KktPoint point = CopyCudaSolution(*solution);
  const Scalar primal_residual =
      clqr::test::adversarial::MaxPrimalResidual(test_case.problem, point);
  Expect(primal_residual <= clqr::test::adversarial::kPrimalTolerance *
                                test_case.tolerance_scale,
         test_case.name +
             " primal residual=" + std::to_string(primal_residual));
  if (test_case.check_full_kkt || accurate_success) {
    std::string worst;
    const Scalar residual = clqr::test::adversarial::MaxKktResidual(
        test_case.problem, point, &worst);
    Expect(residual <= clqr::test::adversarial::kKktTolerance *
                           test_case.tolerance_scale *
                           test_case.kkt_tolerance_scale,
           test_case.name + " KKT residual=" + std::to_string(residual) +
               " in " + worst);
  }
  if (test_case.dense_reference) {
    const DensePrimal dense =
        clqr::test::adversarial::SolveDenseKkt(test_case.problem);
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
  if (!clqr::cuda::Available()) {
    std::cerr << "CUDA unavailable: " << clqr::cuda::DeviceDescription()
              << '\n';
    return 2;
  }
  std::vector<TestCase> cases = clqr::test::adversarial::StandardCases();
  if (extended) {
    std::vector<TestCase> more = clqr::test::adversarial::ExtendedCases();
    cases.insert(cases.end(), more.begin(), more.end());
  }
  clqr::cuda::Workspace workspace;
  clqr::cuda::Solution solution;
  for (const TestCase &test_case : cases)
    RunCase(test_case, &workspace, &solution);
  std::cout << "all " << cases.size()
            << " executed adversarial CUDA cases passed\n";
  return 0;
}
