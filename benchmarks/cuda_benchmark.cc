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
#include "cuda_benchmark_problem.h"

namespace {

using Clock = std::chrono::steady_clock;
using clqr::Matrix;
using clqr::Problem;
using clqr::Scalar;
using clqr::Stage;
using clqr::Vector;

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double KernelTotal(const clqr::cuda::Timings &timings) {
  return timings.feasibility_ms + timings.reduction_ms +
         timings.riccati_ms + timings.reconstruction_ms +
         timings.multiplier_ms;
}

double EventTotal(const clqr::cuda::Timings &timings) {
  return timings.upload_ms + KernelTotal(timings) + timings.download_ms;
}

Scalar BenchmarkMaxAbs(const Vector &vector) {
  Scalar value = Scalar{0};
  for (std::size_t i = 0; i < vector.size(); ++i)
    value = std::max(value, std::abs(vector[i]));
  return value;
}

Scalar MaxScaledMixedResidual(const Stage &stage, const Vector &state,
                              const Vector &control) {
  Scalar residual = Scalar{0};
  for (std::size_t row = 0; row < stage.C.rows(); ++row) {
    Scalar value = stage.d[row];
    Scalar scale = std::max(Scalar{1}, std::abs(stage.d[row]));
    for (std::size_t col = 0; col < stage.C.cols(); ++col) {
      value += stage.C(row, col) * state[col];
      scale = std::max(scale, std::abs(stage.C(row, col)));
    }
    for (std::size_t col = 0; col < stage.D.cols(); ++col) {
      value += stage.D(row, col) * control[col];
      scale = std::max(scale, std::abs(stage.D(row, col)));
    }
    residual = std::max(residual, std::abs(value) / scale);
  }
  return residual;
}

Scalar MaxScaledStateResidual(const Matrix &matrix, const Vector &offset,
                              const Vector &state) {
  Scalar residual = Scalar{0};
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    Scalar value = offset[row];
    Scalar scale = std::max(Scalar{1}, std::abs(offset[row]));
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
      value += matrix(row, col) * state[col];
      scale = std::max(scale, std::abs(matrix(row, col)));
    }
    residual = std::max(residual, std::abs(value) / scale);
  }
  return residual;
}

void AddTransposeProduct(const Matrix &matrix, const Vector &multiplier,
                         Vector *target) {
  for (std::size_t col = 0; col < matrix.cols(); ++col) {
    for (std::size_t row = 0; row < matrix.rows(); ++row)
      (*target)[col] += matrix(row, col) * multiplier[row];
  }
}

Scalar MaxKktResidual(const Problem &problem,
                      const clqr::cuda::Solution &solution) {
  Scalar residual = Scalar{0};
  const std::size_t horizon = problem.stages.size();
  for (std::size_t i = 0; i < horizon; ++i) {
    const Stage &stage = problem.stages[i];
    residual = std::max(
        residual,
        BenchmarkMaxAbs(solution.states[i + 1] - stage.A * solution.states[i] -
                        stage.B * solution.controls[i] - stage.c));
    residual =
        std::max(residual, MaxScaledMixedResidual(stage, solution.states[i],
                                                  solution.controls[i]));
    residual = std::max(
        residual, MaxScaledStateResidual(stage.E, stage.e, solution.states[i]));

    Vector gx = stage.Q * solution.states[i] + stage.M * solution.controls[i] +
                stage.q -
                clqr::Transpose(stage.A) * solution.dynamics_multipliers[i];
    gx = gx + (i == 0 ? solution.initial_multiplier
                      : solution.dynamics_multipliers[i - 1]);
    AddTransposeProduct(stage.C, solution.mixed_multipliers[i], &gx);
    AddTransposeProduct(stage.E, solution.state_multipliers[i], &gx);
    residual = std::max(residual, BenchmarkMaxAbs(gx));

    Vector gu = clqr::Transpose(stage.M) * solution.states[i] +
                stage.R * solution.controls[i] + stage.r -
                clqr::Transpose(stage.B) * solution.dynamics_multipliers[i];
    AddTransposeProduct(stage.D, solution.mixed_multipliers[i], &gu);
    residual = std::max(residual, BenchmarkMaxAbs(gu));
  }
  residual = std::max(residual, BenchmarkMaxAbs(solution.states.front() -
                                                problem.initial_state));
  residual = std::max(residual, MaxScaledStateResidual(problem.terminal_E,
                                                       problem.terminal_e,
                                                       solution.states.back()));
  Vector terminal_gradient =
      problem.terminal_Q * solution.states.back() + problem.terminal_q;
  terminal_gradient =
      terminal_gradient + (horizon == 0 ? solution.initial_multiplier
                                        : solution.dynamics_multipliers.back());
  AddTransposeProduct(problem.terminal_E, solution.terminal_state_multiplier,
                      &terminal_gradient);
  return std::max(residual, BenchmarkMaxAbs(terminal_gradient));
}

} // namespace

int main(int argc, char **argv) {
  if (!clqr::cuda::Available()) {
    std::cerr << "No CUDA device available: " << clqr::cuda::DeviceDescription()
              << "\n";
    return 2;
  }
  int repeats = 5;
  std::size_t cpu_max_horizon = std::numeric_limits<std::size_t>::max();
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
  const std::vector<std::size_t> horizons{32,   64,   128,  256,  512,
                                          1024, 2048, 4096, 8192, 16384};
  constexpr std::size_t n = 8;
  constexpr std::size_t m = 4;
  constexpr std::size_t p = 2;
  std::cout << "# device=" << clqr::cuda::DeviceDescription() << "\n";
  std::cout << "# precision=" << clqr::kPrecisionName << "\n";
  std::cout << "# CUDA capacities: state=" << clqr::cuda::kMaxStateDimension
            << ", control=" << clqr::cuda::kMaxControlDimension
            << ", mixed=" << clqr::cuda::kMaxMixedConstraints
            << ", state constraints=" << clqr::cuda::kMaxStateConstraints
            << "\n";
  std::cout << "# cpp_cpu_ms and CUDA timings use the same scalar precision.\n";
  std::cout
      << "# CUDA wall time reuses reserved storage and includes packing, "
         "transfers, kernels, and result construction; cuda_kernel_ms excludes "
         "host-device transfers.\n";
  if (cpu_max_horizon == std::numeric_limits<std::size_t>::max()) {
    std::cout << "# The sequential C++ solver is timed at every horizon.\n";
  } else {
    std::cout << "# CPU columns are nan above N=" << cpu_max_horizon << ".\n";
  }
  std::cout << "# Multiplier consistency rejection is disabled while timing; "
               "kkt_residual reports the final repeated solution's "
               "accuracy.\n";
  std::cout << "N,n,m,p,repeats,cpp_cpu_ms,cuda_wall_ms,cuda_kernel_ms,"
               "wall_speedup,kernel_speedup,host_overhead_ms,upload_ms,"
               "feasibility_ms,reduction_ms,riccati_ms,reconstruction_ms,"
               "multiplier_ms,download_ms,min_reduced_n,"
               "min_reduced_m,parallel_riccati,cuda_kkt_residual\n";
  clqr::cuda::Workspace cuda_workspace;
  int completed_horizons = 0;
  for (std::size_t horizon : horizons) {
    Problem problem = clqr::benchmark::StateOnlyProblem(horizon, n, m, p);
    clqr::Workspace workspace;
    const bool run_cpu = horizon <= cpu_max_horizon;
    if (run_cpu) {
      workspace.Reserve(problem);
      const clqr::SolutionView cpu = clqr::Solve(problem, workspace);
      if (cpu.status != clqr::SolveStatus::kOptimal) {
        std::cerr << "CPU warmup failed at N=" << horizon << ": " << cpu.message
                  << "\n";
        return 1;
      }
    }
    clqr::cuda::Options cuda_options;
    cuda_options.enforce_multiplier_consistency = false;
    cuda_workspace.Reserve(problem, cuda_options);
    clqr::cuda::Solution gpu;
    clqr::cuda::Solve(problem, cuda_workspace, gpu, cuda_options);
    if (gpu.status != clqr::SolveStatus::kOptimal) {
      std::cout << "# CUDA warmup failed at N=" << horizon << ": "
                << gpu.message << "\n";
      continue;
    }

    std::vector<double> cpu_times;
    std::vector<double> gpu_wall_times;
    std::vector<double> gpu_kernel_times;
    std::vector<double> host_overheads;
    std::vector<double> feasibility;
    std::vector<double> upload;
    std::vector<double> reduction;
    std::vector<double> riccati;
    std::vector<double> reconstruction;
    std::vector<double> multiplier;
    std::vector<double> download;
    bool repeat_failed = false;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      if (run_cpu) {
        const auto cpu_start = Clock::now();
        const clqr::SolutionView cpu = clqr::Solve(problem, workspace);
        const auto cpu_stop = Clock::now();
        if (cpu.status != clqr::SolveStatus::kOptimal)
          return 1;
        cpu_times.push_back(
            std::chrono::duration<double, std::milli>(cpu_stop - cpu_start)
                .count());
      }

      const auto gpu_start = Clock::now();
      clqr::cuda::Solve(problem, cuda_workspace, gpu, cuda_options);
      const auto gpu_stop = Clock::now();
      if (gpu.status != clqr::SolveStatus::kOptimal) {
        std::cout << "# CUDA timed solve failed at N=" << horizon
                  << ", repeat=" << repeat << ": " << gpu.message << "\n";
        repeat_failed = true;
        break;
      }
      const double gpu_wall_time =
          std::chrono::duration<double, std::milli>(gpu_stop - gpu_start)
              .count();
      const double gpu_kernel_time = KernelTotal(gpu.timings);
      gpu_wall_times.push_back(gpu_wall_time);
      gpu_kernel_times.push_back(gpu_kernel_time);
      host_overheads.push_back(gpu_wall_time - EventTotal(gpu.timings));
      upload.push_back(gpu.timings.upload_ms);
      feasibility.push_back(gpu.timings.feasibility_ms);
      reduction.push_back(gpu.timings.reduction_ms);
      riccati.push_back(gpu.timings.riccati_ms);
      reconstruction.push_back(gpu.timings.reconstruction_ms);
      multiplier.push_back(gpu.timings.multiplier_ms);
      download.push_back(gpu.timings.download_ms);
    }
    if (repeat_failed)
      continue;
    const double cpu_ms =
        run_cpu ? Median(cpu_times) : std::numeric_limits<double>::quiet_NaN();
    const double gpu_wall_ms = Median(gpu_wall_times);
    const double gpu_kernel_ms = Median(gpu_kernel_times);
    int min_reduced_n = std::numeric_limits<int>::max();
    int min_reduced_m = std::numeric_limits<int>::max();
    for (int value : gpu.reduced_state_dimensions)
      min_reduced_n = std::min(min_reduced_n, value);
    for (int value : gpu.reduced_control_dimensions)
      min_reduced_m = std::min(min_reduced_m, value);
    const Scalar kkt_residual = MaxKktResidual(problem, gpu);
    std::cout << horizon << ',' << n << ',' << m << ',' << p << ',' << repeats
              << ',' << std::fixed << std::setprecision(6) << cpu_ms << ','
              << gpu_wall_ms << ',' << gpu_kernel_ms << ','
              << cpu_ms / gpu_wall_ms << ',' << cpu_ms / gpu_kernel_ms << ','
              << Median(host_overheads) << ',' << Median(upload) << ','
              << Median(feasibility) << ',' << Median(reduction) << ','
              << Median(riccati) << ',' << Median(reconstruction) << ','
              << Median(multiplier) << ',' << Median(download) << ','
              << min_reduced_n << ',' << min_reduced_m << ','
              << (gpu.used_parallel_riccati ? "yes" : "no") << ','
              << std::scientific << kkt_residual << '\n';
    ++completed_horizons;
  }
  return completed_horizons == 0 ? 1 : 0;
}
