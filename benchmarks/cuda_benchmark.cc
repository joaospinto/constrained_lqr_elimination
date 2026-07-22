#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "clqr/cuda.h"

namespace {

using Clock = std::chrono::steady_clock;
using clqr::Matrix;
using clqr::Problem;
using clqr::Scalar;
using clqr::Stage;
using clqr::Vector;

Scalar Value(int seed, std::size_t row, std::size_t col = 0) {
  const Scalar x = static_cast<Scalar>(seed * 97 + row * 31 + col * 47);
  return std::sin(0.017 * x) + 0.3 * std::cos(0.029 * x);
}

Matrix GeneratedMatrix(std::size_t rows, std::size_t cols, int seed,
                       Scalar scale) {
  Matrix out(rows, cols);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col)
      out(row, col) = scale * Value(seed, row, col);
  }
  return out;
}

Vector GeneratedVector(std::size_t size, int seed, Scalar scale) {
  Vector out(size);
  for (std::size_t row = 0; row < size; ++row)
    out[row] = scale * Value(seed, row);
  return out;
}

Matrix PositiveDefinite(std::size_t size, int seed, Scalar diagonal) {
  Matrix g = GeneratedMatrix(size, size, seed, 0.15);
  Matrix out = clqr::Transpose(g) * g;
  for (std::size_t row = 0; row < size; ++row) out(row, row) += diagonal;
  return out;
}

Scalar RowDot(const Matrix& matrix, std::size_t row, const Vector& vector) {
  Scalar value = 0.0;
  for (std::size_t col = 0; col < vector.size(); ++col)
    value += matrix(row, col) * vector[col];
  return value;
}

Problem StateOnlyProblem(std::size_t horizon, std::size_t n, std::size_t m,
                         std::size_t p) {
  Problem problem;
  std::vector<Vector> x(horizon + 1);
  std::vector<Vector> u(horizon);
  for (std::size_t i = 0; i <= horizon; ++i)
    x[i] = GeneratedVector(n, 1000 + static_cast<int>(i), 0.5);
  for (std::size_t i = 0; i < horizon; ++i)
    u[i] = GeneratedVector(m, 2000 + static_cast<int>(i), 0.4);
  problem.initial_state = x[0];
  problem.stages.resize(horizon);
  for (std::size_t i = 0; i < horizon; ++i) {
    Stage& stage = problem.stages[i];
    stage.A = GeneratedMatrix(n, n, 3000 + static_cast<int>(i), 0.08);
    for (std::size_t row = 0; row < n; ++row) stage.A(row, row) += 0.9;
    stage.B = GeneratedMatrix(n, m, 4000 + static_cast<int>(i), 0.15);
    stage.c = x[i + 1] - stage.A * x[i] - stage.B * u[i];
    stage.Q = PositiveDefinite(n, 5000 + static_cast<int>(i), 1.0);
    stage.R = PositiveDefinite(m, 6000 + static_cast<int>(i), 1.5);
    stage.M = GeneratedMatrix(n, m, 7000 + static_cast<int>(i), 0.02);
    stage.q = GeneratedVector(n, 8000 + static_cast<int>(i), 0.1);
    stage.r = GeneratedVector(m, 9000 + static_cast<int>(i), 0.1);
    stage.C = Matrix(0, n);
    stage.D = Matrix(0, m);
    stage.d = Vector(0);
    stage.E = GeneratedMatrix(p, n, 10000 + static_cast<int>(i), 0.3);
    stage.e = Vector(p);
    for (std::size_t row = 0; row < p; ++row)
      stage.e[row] = -RowDot(stage.E, row, x[i]);
  }
  problem.terminal_Q = PositiveDefinite(n, 11000, 1.5);
  problem.terminal_q = GeneratedVector(n, 12000, 0.1);
  problem.terminal_E = Matrix(0, n);
  problem.terminal_e = Vector(0);
  return problem;
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double DeviceTotal(const clqr::cuda::Timings& timings) {
  return timings.upload_ms + timings.feasibility_ms + timings.reduction_ms +
         timings.riccati_ms + timings.reconstruction_ms +
         timings.multiplier_ms + timings.download_ms;
}

}  // namespace

int main(int argc, char** argv) {
  if (!clqr::cuda::Available()) {
    std::cerr << "No CUDA device available: " << clqr::cuda::DeviceDescription()
              << "\n";
    return 2;
  }
  int repeats = 5;
  std::size_t cpu_max_horizon = 128;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string option = argv[i];
    if (option == "--repeats") {
      repeats = std::max(1, std::atoi(argv[i + 1]));
    } else if (option == "--cpu-max-horizon") {
      cpu_max_horizon =
          static_cast<std::size_t>(std::max(0, std::atoi(argv[i + 1])));
    } else {
      std::cerr << "unknown option: " << option << '\n';
      return 2;
    }
  }
  const std::vector<std::size_t> horizons{32, 64, 128, 256, 512, 1024, 2048};
  constexpr std::size_t n = 8;
  constexpr std::size_t m = 4;
  constexpr std::size_t p = 2;
  std::cout << "# device=" << clqr::cuda::DeviceDescription() << "\n";
  std::cout << "# precision=" << clqr::kPrecisionName << "\n";
  std::cout << "# cpp_cpu_ms and CUDA timings use the same scalar precision.\n";
  std::cout
      << "# CUDA wall time includes allocation; cuda_device_ms is the sum "
         "of event-timed transfer and kernel phases.\n";
  std::cout << "# CPU columns are nan above N=" << cpu_max_horizon
            << "; the current general CPU multiplier-workspace bound is cubic "
               "in N for state-only equalities.\n";
  std::cout << "N,n,m,p,repeats,cpp_cpu_ms,cuda_wall_ms,cuda_device_ms,"
               "wall_speedup,feasibility_ms,reduction_ms,riccati_ms,"
               "reconstruction_ms,multiplier_ms,min_reduced_n,"
               "min_reduced_m,parallel_riccati\n";
  for (std::size_t horizon : horizons) {
    Problem problem = StateOnlyProblem(horizon, n, m, p);
    clqr::Workspace workspace;
    const bool run_cpu = horizon <= cpu_max_horizon;
    if (run_cpu) {
      workspace.Reserve(problem);
      for (int warmup = 0; warmup < 2; ++warmup) {
        const clqr::SolutionView cpu = clqr::Solve(problem, workspace);
        if (cpu.status != clqr::SolveStatus::kOptimal) {
          std::cerr << "CPU warmup failed at N=" << horizon << ": "
                    << cpu.message << "\n";
          return 1;
        }
      }
    }
    clqr::cuda::Solution gpu = clqr::cuda::Solve(problem);
    if (gpu.status != clqr::SolveStatus::kOptimal) {
      std::cerr << "CUDA warmup failed at N=" << horizon << ": " << gpu.message
                << "\n";
      return 1;
    }

    std::vector<double> cpu_times;
    std::vector<double> gpu_wall_times;
    std::vector<double> gpu_device_times;
    std::vector<double> feasibility;
    std::vector<double> reduction;
    std::vector<double> riccati;
    std::vector<double> reconstruction;
    std::vector<double> multiplier;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      if (run_cpu) {
        const auto cpu_start = Clock::now();
        const clqr::SolutionView cpu = clqr::Solve(problem, workspace);
        const auto cpu_stop = Clock::now();
        if (cpu.status != clqr::SolveStatus::kOptimal) return 1;
        cpu_times.push_back(
            std::chrono::duration<double, std::milli>(cpu_stop - cpu_start)
                .count());
      }

      const auto gpu_start = Clock::now();
      gpu = clqr::cuda::Solve(problem);
      const auto gpu_stop = Clock::now();
      if (gpu.status != clqr::SolveStatus::kOptimal) return 1;
      gpu_wall_times.push_back(
          std::chrono::duration<double, std::milli>(gpu_stop - gpu_start)
              .count());
      gpu_device_times.push_back(DeviceTotal(gpu.timings));
      feasibility.push_back(gpu.timings.feasibility_ms);
      reduction.push_back(gpu.timings.reduction_ms);
      riccati.push_back(gpu.timings.riccati_ms);
      reconstruction.push_back(gpu.timings.reconstruction_ms);
      multiplier.push_back(gpu.timings.multiplier_ms);
    }
    const double cpu_ms =
        run_cpu ? Median(cpu_times) : std::numeric_limits<double>::quiet_NaN();
    const double gpu_wall_ms = Median(gpu_wall_times);
    int min_reduced_n = std::numeric_limits<int>::max();
    int min_reduced_m = std::numeric_limits<int>::max();
    for (int value : gpu.reduced_state_dimensions)
      min_reduced_n = std::min(min_reduced_n, value);
    for (int value : gpu.reduced_control_dimensions)
      min_reduced_m = std::min(min_reduced_m, value);
    std::cout << horizon << ',' << n << ',' << m << ',' << p << ',' << repeats
              << ',' << std::fixed << std::setprecision(6) << cpu_ms << ','
              << gpu_wall_ms << ',' << Median(gpu_device_times) << ','
              << cpu_ms / gpu_wall_ms << ',' << Median(feasibility) << ','
              << Median(reduction) << ',' << Median(riccati) << ','
              << Median(reconstruction) << ',' << Median(multiplier) << ','
              << min_reduced_n << ',' << min_reduced_m << ','
              << (gpu.used_parallel_riccati ? "yes" : "no") << '\n';
  }
  return 0;
}
