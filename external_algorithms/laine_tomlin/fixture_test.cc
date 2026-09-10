#include "solver.h"
#include "tests/adversarial_test_support.h"

#include "fixture_audit.h"
#include "test_problem.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {
using namespace laine_tomlin::audit;
template <class F> double Time(F solve) {
  using Clock = std::chrono::steady_clock;
  auto ms = [](auto start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
  };
  const auto start = Clock::now();
  int count = 0;
  do {
    solve();
    ++count;
  } while (count < 3 || ms(start) < 50);
  const int batch = std::clamp(static_cast<int>(count / ms(start)), 1, 10000);
  std::vector<double> times;
  for (int i = 0; i < 21; ++i) {
    const auto begin = Clock::now();
    for (int j = 0; j < batch; ++j)
      solve();
    times.push_back(ms(begin) / batch);
  }
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}
} // namespace

int main(int argc, char **argv) {
  const bool benchmark =
      argc == 2 && std::string_view(argv[1]) == "--benchmark";
  if (argc > 1 && !benchmark)
    return 2;
  auto cases = clqr::test::adversarial::StandardCases();
  const auto extended = clqr::test::adversarial::ExtendedCases();
  cases.insert(cases.end(), extended.begin(), extended.end());
  if (!benchmark) {
    for (bool uniform : {false, true})
      for (unsigned seed = 0; seed < 128; ++seed)
        cases.push_back(
            {std::string(uniform ? "uniform-" : "varying-") +
                 std::to_string(seed),
             laine_tomlin::conversion::ToClqr(
                 laine_tomlin::test::RandomProblem(seed, uniform))});
  }
  int failures = 0, returned = 0;
  std::cout << std::setprecision(12)
            << "case,status,laine_corrected_ms,clqr_ms,feasibility_inf,audited_"
               "stationarity_inf,clqr_kkt_inf,primal_difference,diagnostic\n";
  for (const auto &c : cases) {
    try {
      const auto p = Convert(c.problem);
      auto r = laine_tomlin::Solve(p);
      auto expected = c.cpu_status;
      // CLQR can return indefinite Newton directions; this reference minimizes.
      if (c.name == "indefinite-reduced-hessian")
        expected = clqr::SolveStatus::kNumericalFailure;
      const auto mapped = expected == clqr::SolveStatus::kOptimal
                              ? laine_tomlin::Status::kOptimal
                          : expected == clqr::SolveStatus::kInfeasible
                              ? laine_tomlin::Status::kInfeasible
                          : expected == clqr::SolveStatus::kInvalidInput
                              ? laine_tomlin::Status::kInvalidInput
                              : laine_tomlin::Status::kNumericalFailure;
      if (r.status != mapped)
        ++failures;
      if (r.status != laine_tomlin::Status::kOptimal) {
        std::cout << c.name << ",rejected,nan,nan,nan,nan,nan,nan,"
                  << CsvMessage(r.message) << '\n';
        continue;
      }
      ++returned;
      clqr::Workspace workspace;
      workspace.Reserve(c.problem);
      auto native = clqr::Solve(c.problem, workspace);
      if (native.status != clqr::SolveStatus::kOptimal)
        throw std::runtime_error(native.message);
      double lt_ms = std::numeric_limits<double>::quiet_NaN(), cpu_ms = lt_ms;
      if (benchmark) {
        lt_ms = Time([&] { r = laine_tomlin::Solve(p); });
        cpu_ms = Time([&] { native = clqr::Solve(c.problem, workspace); });
      }
      const auto [primal, stationarity] = Audit(p, r);
      const auto point = clqr::test::adversarial::CopyCpuSolution(native);
      std::string worst;
      const auto cpu_kkt =
          clqr::test::adversarial::MaxKktResidual(c.problem, point, &worst);
      double difference = 0;
      for (std::size_t t = 0; t < r.states.size(); ++t) {
        difference =
            std::max(difference, Max(r.states[t] - Convert(point.states[t])));
        if (t < r.controls.size())
          difference = std::max(
              difference, Max(r.controls[t] - Convert(point.controls[t])));
      }
      if (primal > 1e-7 || stationarity > 1e-7 || difference > 1e-5 ||
          !std::isfinite(cpu_kkt) || cpu_kkt > 1e-7)
        ++failures;
      if (!benchmark) {
        const auto factors = clqr::Factor(c.problem);
        if (factors.status() != clqr::SolveStatus::kOptimal)
          throw std::runtime_error(factors.message());
        clqr::Workspace cached_workspace;
        cached_workspace.Reserve(factors);
        const auto cached =
            clqr::Solve(factors, clqr::ExtractRhs(c.problem), cached_workspace);
        if (cached.status != clqr::SolveStatus::kOptimal)
          throw std::runtime_error(cached.message);
        std::string cached_worst;
        const double cached_kkt = clqr::test::adversarial::MaxKktResidual(
            c.problem, clqr::test::adversarial::CopyCpuSolution(cached),
            &cached_worst);
        if (!std::isfinite(cached_kkt) || cached_kkt > 1e-7) {
          ++failures;
          std::cerr << c.name << " cached KKT=" << cached_kkt << " at "
                    << cached_worst << '\n';
        }
      }
      std::cout << c.name << ",returned," << lt_ms << ',' << cpu_ms << ','
                << primal << ',' << stationarity << ',' << cpu_kkt << ','
                << difference << ',' << CsvMessage(worst) << '\n';
    } catch (const std::exception &e) {
      ++failures;
      std::cout << c.name << ",exception,nan,nan,nan,nan,nan,nan,"
                << CsvMessage(e.what()) << '\n';
    }
  }
  std::cerr << returned << " returned trajectories in " << cases.size()
            << " fixtures; " << failures << " validation discrepancies\n";
  // Benchmark outcomes are measurements, not regression-test gates.
  return benchmark ? 0 : failures != 0;
}
