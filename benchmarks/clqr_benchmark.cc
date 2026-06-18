#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "clqr/clqr.h"

namespace {

struct Dimensions {
  std::size_t horizon;
  std::size_t states;
  std::size_t controls;
  int base_iterations;
};

double DeterministicValue(int seed, std::size_t i, std::size_t j = 0) {
  const double x = static_cast<double>(seed + 17 * static_cast<int>(i) +
                                      31 * static_cast<int>(j));
  return 0.5 * std::sin(0.37 * x) + 0.25 * std::cos(0.19 * x);
}

clqr::Vector GeneratedVector(std::size_t size, int seed, double scale = 1.0) {
  clqr::Vector out(size);
  for (std::size_t i = 0; i < size; ++i) out[i] = scale * DeterministicValue(seed, i);
  return out;
}

clqr::Matrix GeneratedMatrix(std::size_t rows, std::size_t cols, int seed,
                             double scale = 1.0) {
  clqr::Matrix out(rows, cols);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      out(i, j) = scale * DeterministicValue(seed, i, j);
    }
  }
  return out;
}

clqr::Matrix PositiveDefinite(std::size_t size, int seed, double diagonal) {
  clqr::Matrix g = GeneratedMatrix(size, size, seed, 0.15);
  clqr::Matrix out = clqr::Transpose(g) * g;
  for (std::size_t i = 0; i < size; ++i) out(i, i) += diagonal;
  return out;
}

double RowDot(const clqr::Matrix& a, std::size_t row, const clqr::Vector& x) {
  double out = 0.0;
  for (std::size_t col = 0; col < x.size(); ++col) out += a(row, col) * x[col];
  return out;
}

clqr::Problem MakeFeasibleMixedProblem(int seed, std::size_t horizon, std::size_t states,
                                       std::size_t controls, std::size_t constraints) {
  clqr::Problem problem;
  std::vector<clqr::Vector> x(horizon + 1);
  std::vector<clqr::Vector> u(horizon);
  for (std::size_t i = 0; i <= horizon; ++i) {
    x[i] = GeneratedVector(states, seed + 100 + static_cast<int>(i), 0.5);
  }
  for (std::size_t i = 0; i < horizon; ++i) {
    u[i] = GeneratedVector(controls, seed + 200 + static_cast<int>(i), 0.4);
  }

  problem.initial_state = x[0];
  problem.stages.resize(horizon);
  for (std::size_t i = 0; i < horizon; ++i) {
    clqr::Stage& stage = problem.stages[i];
    stage.A = GeneratedMatrix(states, states, seed + 10 * static_cast<int>(i), 0.1);
    for (std::size_t row = 0; row < states; ++row) stage.A(row, row) += 0.9;
    stage.B = GeneratedMatrix(states, controls, seed + 300 + 10 * static_cast<int>(i), 0.2);
    stage.c = x[i + 1] - stage.A * x[i] - stage.B * u[i];
    stage.Q = PositiveDefinite(states, seed + 400 + static_cast<int>(i), 1.0);
    stage.R = PositiveDefinite(controls, seed + 500 + static_cast<int>(i), 1.5);
    stage.M = GeneratedMatrix(states, controls, seed + 600 + static_cast<int>(i), 0.03);
    stage.q = GeneratedVector(states, seed + 700 + static_cast<int>(i), 0.2);
    stage.r = GeneratedVector(controls, seed + 800 + static_cast<int>(i), 0.2);
    stage.C = clqr::Matrix(0, states);
    stage.D = clqr::Matrix(0, controls);
    stage.d = clqr::Vector(0);
    stage.E = clqr::Matrix(0, states);
    stage.e = clqr::Vector(0);

    if (constraints > 0 && i % 3 == 1) {
      stage.C = GeneratedMatrix(constraints, states, seed + 900 + static_cast<int>(i), 0.25);
      stage.D =
          GeneratedMatrix(constraints, controls, seed + 1000 + static_cast<int>(i), 0.25);
      stage.d = clqr::Vector(constraints);
      for (std::size_t row = 0; row < constraints; ++row) {
        stage.d[row] = -(RowDot(stage.C, row, x[i]) + RowDot(stage.D, row, u[i]));
      }
    }
  }
  problem.terminal_Q = PositiveDefinite(states, seed + 1200, 1.5);
  problem.terminal_q = GeneratedVector(states, seed + 1300, 0.2);
  problem.terminal_E = clqr::Matrix(0, states);
  problem.terminal_e = clqr::Vector(0);
  return problem;
}

int IterationScale(int argc, char** argv) {
  if (argc <= 1) return 1;
  const int scale = std::atoi(argv[1]);
  return std::max(scale, 1);
}

void RunCase(const std::string& name, const clqr::Problem& problem, int iterations) {
  clqr::SolveOptions options;
  options.tolerance = 1e-9;

  clqr::Solution warmup = clqr::Solve(problem, options);
  if (warmup.status != clqr::SolveStatus::kOptimal) {
    std::cout << name << ",status=warmup_failed,message=\"" << warmup.message << "\"\n";
    return;
  }

  double checksum = 0.0;
  std::vector<double> times_us;
  times_us.reserve(iterations);
  int singular_count = 0;
  int wrong_inertia_count = 0;
  for (int i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    clqr::Solution solution = clqr::Solve(problem, options);
    const auto end = std::chrono::steady_clock::now();
    if (solution.status != clqr::SolveStatus::kOptimal) {
      std::cout << name << ",status=failed,message=\"" << solution.message << "\"\n";
      return;
    }
    const double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
    times_us.push_back(elapsed_us);
    checksum += solution.objective;
    singular_count += solution.newton_kkt_singular ? 1 : 0;
    wrong_inertia_count += solution.newton_kkt_wrong_inertia ? 1 : 0;
  }

  std::vector<double> sorted_times = times_us;
  std::sort(sorted_times.begin(), sorted_times.end());
  double total_us = 0.0;
  for (double time_us : times_us) total_us += time_us;
  const double min_us = sorted_times.front();
  const double max_us = sorted_times.back();
  const double median_us = sorted_times[sorted_times.size() / 2];
  const double p90_us = sorted_times[static_cast<std::size_t>(
      std::floor(0.9 * static_cast<double>(sorted_times.size() - 1)))];

  std::cout << name << ",iterations=" << iterations << ",mean_us=" << total_us / iterations
            << ",median_us=" << median_us << ",p90_us=" << p90_us
            << ",min_us=" << min_us << ",max_us=" << max_us
            << ",objective_checksum=" << checksum << ",singular_count=" << singular_count
            << ",wrong_inertia_count=" << wrong_inertia_count << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  const int scale = IterationScale(argc, argv);
  const std::vector<Dimensions> dimensions = {
      {16, 4, 2, 200},
      {16, 6, 3, 100},
      {32, 6, 3, 50},
      {64, 6, 3, 20},
      {128, 8, 4, 10},
  };

  std::cout << "case,iterations,mean_us,median_us,p90_us,min_us,max_us,"
               "objective_checksum,singular_count,wrong_inertia_count\n";
  int seed = 1;
  for (const Dimensions& dim : dimensions) {
    for (std::size_t constraints = 0; constraints <= 2; ++constraints) {
      const std::string name = "N=" + std::to_string(dim.horizon) +
                               " n=" + std::to_string(dim.states) +
                               " m=" + std::to_string(dim.controls) +
                               " p=" + std::to_string(constraints) + " mixed";
      RunCase(name,
              MakeFeasibleMixedProblem(seed, dim.horizon, dim.states, dim.controls,
                                       constraints),
              dim.base_iterations * scale);
      ++seed;
    }
  }
  return 0;
}
