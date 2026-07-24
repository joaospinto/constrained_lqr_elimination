#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "adversarial_test_support.h"
#include "clqr/clqr.h"

namespace {

using clqr::Problem;
using clqr::Scalar;
using clqr::SolutionView;
using clqr::SolveStatus;
using clqr::Workspace;
using clqr::test::adversarial::KktPoint;
using clqr::test::adversarial::Pattern;

struct BenchmarkCase {
  std::string name;
  Problem problem;
};

int ParsePositiveInteger(const char *text, const char *argument) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0) {
    std::cerr << argument << " must be a positive integer\n";
    std::exit(2);
  }
  return static_cast<int>(value);
}

std::string CsvToken(std::string value) {
  for (char &character : value) {
    if (character == ' ' || character == ',')
      character = '_';
  }
  return value;
}

std::vector<BenchmarkCase> Cases() {
  using clqr::test::adversarial::UniformProblem;
  std::vector<BenchmarkCase> cases;
  cases.push_back({"mixed-N32-n6-m3-p2",
                   UniformProblem(5001, 32, 6, 3, 2, Pattern::kMixed)});
  cases.push_back({"alternating-N64-n6-m3-p2",
                   UniformProblem(5002, 64, 6, 3, 2, Pattern::kAlternating)});
  // Reuse the adversarial suite's fixed seed for the p > m case so both
  // precisions have a quantitatively checked, well-conditioned comparison.
  cases.push_back({"more-mixed-N5-n4-m1-p3",
                   UniformProblem(42, 5, 4, 1, 3, Pattern::kMixed)});
  cases.push_back({"redundant-N32-n6-m3-p3",
                   UniformProblem(5004, 32, 6, 3, 3, Pattern::kRedundant)});
  cases.push_back({"scaled-N32-n6-m3-p3",
                   UniformProblem(5005, 32, 6, 3, 3, Pattern::kScaled)});
  cases.push_back({"terminal-N64-n6-m3-p3",
                   UniformProblem(5006, 64, 6, 3, 3, Pattern::kTerminal)});
  cases.push_back({"accuracy-limit-N17-n3-m2-p1",
                   UniformProblem(27, 17, 3, 2, 1, Pattern::kAlternating)});
  return cases;
}

void RunCase(const BenchmarkCase &benchmark_case, int warmups, int repeats) {
  Workspace workspace;
  workspace.Reserve(benchmark_case.problem);
  SolutionView result;
  for (int repetition = 0; repetition < warmups; ++repetition) {
    result = clqr::Solve(benchmark_case.problem, workspace);
    if (result.status != SolveStatus::kOptimal)
      break;
  }

  std::vector<double> elapsed_us;
  elapsed_us.reserve(static_cast<std::size_t>(repeats));
  volatile Scalar objective_checksum = Scalar{0};
  if (result.status == SolveStatus::kOptimal) {
    for (int repetition = 0; repetition < repeats; ++repetition) {
      const auto begin = std::chrono::steady_clock::now();
      result = clqr::Solve(benchmark_case.problem, workspace);
      const auto end = std::chrono::steady_clock::now();
      if (result.status != SolveStatus::kOptimal)
        break;
      elapsed_us.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
      objective_checksum += result.objective;
    }
  }

  Scalar primal_residual = std::numeric_limits<Scalar>::infinity();
  Scalar kkt_residual = std::numeric_limits<Scalar>::infinity();
  std::string worst_equation = "not_available";
  if (result.status == SolveStatus::kOptimal) {
    const KktPoint point = clqr::test::adversarial::CopyCpuSolution(result);
    primal_residual = clqr::test::adversarial::MaxPrimalResidual(
        benchmark_case.problem, point);
    kkt_residual = clqr::test::adversarial::MaxKktResidual(
        benchmark_case.problem, point, &worst_equation);
  }

  double median_us = std::numeric_limits<double>::infinity();
  double p10_us = median_us;
  double p90_us = median_us;
  if (!elapsed_us.empty()) {
    std::sort(elapsed_us.begin(), elapsed_us.end());
    median_us = elapsed_us[elapsed_us.size() / 2];
    p10_us = elapsed_us[static_cast<std::size_t>(
        std::floor(0.1 * static_cast<double>(elapsed_us.size() - 1)))];
    p90_us = elapsed_us[static_cast<std::size_t>(
        std::floor(0.9 * static_cast<double>(elapsed_us.size() - 1)))];
  }

  std::cout << clqr::kPrecisionName << ',' << benchmark_case.name << ','
            << clqr::StatusName(result.status) << ','
            << CsvToken(result.message) << ',' << elapsed_us.size() << ','
            << std::setprecision(12) << median_us << ',' << p10_us << ','
            << p90_us << ',' << primal_residual << ',' << kkt_residual << ','
            << CsvToken(worst_equation) << ',' << objective_checksum << '\n';
}

} // namespace

int main(int argc, char **argv) {
  int warmups = 5;
  int repeats = 101;
  for (int argument = 1; argument < argc; ++argument) {
    const std::string option = argv[argument];
    if (option == "--warmups" && argument + 1 < argc) {
      warmups = ParsePositiveInteger(argv[++argument], "--warmups");
    } else if (option == "--repeats" && argument + 1 < argc) {
      repeats = ParsePositiveInteger(argv[++argument], "--repeats");
    } else {
      std::cerr << "usage: " << argv[0]
                << " [--warmups COUNT] [--repeats COUNT]\n";
      return 2;
    }
  }

  std::cout << "precision,case,status,message,repeats,median_us,p10_us,p90_us,"
               "primal_residual,kkt_residual,worst_equation,"
               "objective_checksum\n";
  for (const BenchmarkCase &benchmark_case : Cases())
    RunCase(benchmark_case, warmups, repeats);
  return 0;
}
