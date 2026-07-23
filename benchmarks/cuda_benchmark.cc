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
  return timings.feasibility_ms + timings.reduction_ms + timings.riccati_ms +
         timings.reconstruction_ms + timings.multiplier_ms +
         timings.objective_ms;
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

Vector CopyVectorView(const clqr::VectorView &view) {
  Vector copy(view.size);
  for (std::size_t index = 0; index < view.size; ++index)
    copy[index] = view[index];
  return copy;
}

std::vector<Vector> CopyVectorViews(const clqr::VectorView *views,
                                    std::size_t count) {
  std::vector<Vector> copies;
  copies.reserve(count);
  for (std::size_t index = 0; index < count; ++index)
    copies.push_back(CopyVectorView(views[index]));
  return copies;
}

clqr::cuda::Solution CopyCpuSolution(const clqr::SolutionView &view) {
  clqr::cuda::Solution copy;
  copy.states = CopyVectorViews(view.states, view.state_count);
  copy.controls = CopyVectorViews(view.controls, view.control_count);
  copy.initial_multiplier = CopyVectorView(view.initial_multiplier);
  copy.dynamics_multipliers = CopyVectorViews(
      view.dynamics_multipliers, view.dynamics_multiplier_count);
  copy.mixed_multipliers =
      CopyVectorViews(view.mixed_multipliers, view.mixed_multiplier_count);
  copy.state_multipliers =
      CopyVectorViews(view.state_multipliers, view.state_multiplier_count);
  copy.terminal_state_multiplier =
      CopyVectorView(view.terminal_state_multiplier);
  return copy;
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
  std::cout << "# dimensions are runtime-sized; benchmark problem: n=" << n
            << ", m=" << m << ", mixed/state constraints=" << p << "\n";
  std::cout << "# per-stage records are compact runtime slices; dense kernel "
               "workspaces use active-dimension dynamic shared memory.\n";
  std::cout << "# cpp_cpu_ms and CUDA timings use the same scalar precision.\n";
  std::cout
      << "# CUDA wall time reuses reserved storage and includes packing, "
         "transfers, kernels, synchronization, and construction of a "
         "workspace-backed result view.\n"
         "# cuda_kernel_ms sums pure kernel event times and excludes all "
         "host-device transfers; upload_ms and download_ms cover the bulk "
         "packed inputs and outputs.\n"
         "# objective_ms is the device-side objective reduction and is "
         "included in cuda_kernel_ms.\n"
         "# other_wall_ms is wall time minus bulk upload, kernels, and bulk "
         "download; it includes host packing/view construction, "
         "phase-control "
         "transfers, synchronization, and API setup/validation.\n"
         "# api_overhead_ms is time outside the backend's internal timer "
         "(validation and device/workspace setup); internal_other_ms is "
         "internal wall time not covered by bulk-transfer or kernel events.\n"
         "# The timed CUDA path uses a prepared workspace: numerical values "
         "may change, while dimensions and shapes remain fixed.\n";
  if (cpu_max_horizon == std::numeric_limits<std::size_t>::max()) {
    std::cout << "# The sequential C++ solver is timed at every horizon.\n";
  } else {
    std::cout << "# CPU columns are nan above N=" << cpu_max_horizon << ".\n";
  }
  std::cout << "# Multiplier consistency rejection is disabled while timing; "
               "the KKT residuals report the final repeated solutions' "
               "accuracy.\n";
  std::cout << "N,n,m,p,repeats,cpp_cpu_ms,cuda_wall_ms,cuda_internal_ms,"
               "cuda_kernel_ms,wall_speedup,kernel_speedup,other_wall_ms,"
               "api_overhead_ms,internal_other_ms,input_pack_ms,layout_ms,"
               "result_ms,objective_ms,synchronizations,"
               "unattributed_internal_ms,upload_ms,"
               "feasibility_ms,reduction_ms,riccati_ms,reconstruction_ms,"
               "multiplier_ms,download_ms,min_reduced_n,"
               "min_reduced_m,cpp_kkt_residual,cuda_kkt_residual\n";
  clqr::cuda::Workspace cuda_workspace;
  int completed_horizons = 0;
  for (std::size_t horizon : horizons) {
    Problem problem = clqr::benchmark::StateOnlyProblem(horizon, n, m, p);
    clqr::Workspace workspace;
    clqr::SolutionView cpu_view;
    const bool run_cpu = horizon <= cpu_max_horizon;
    if (run_cpu) {
      workspace.Reserve(problem);
      cpu_view = clqr::Solve(problem, workspace);
      if (cpu_view.status != clqr::SolveStatus::kOptimal) {
        std::cerr << "CPU warmup failed at N=" << horizon << ": "
                  << cpu_view.message << "\n";
        return 1;
      }
    }
    clqr::cuda::Options cuda_options;
    cuda_options.enforce_multiplier_consistency = false;
    cuda_workspace.Reserve(problem, cuda_options);
    clqr::cuda::Solution gpu;
    clqr::cuda::SolutionView gpu_view =
        clqr::cuda::SolvePreparedView(problem, cuda_workspace, cuda_options);
    if (gpu_view.status != clqr::SolveStatus::kOptimal) {
      std::cout << "# CUDA warmup failed at N=" << horizon << ": "
                << gpu_view.message << "\n";
      continue;
    }

    std::vector<double> cpu_times;
    std::vector<double> gpu_wall_times;
    std::vector<double> gpu_internal_times;
    std::vector<double> gpu_kernel_times;
    std::vector<double> other_wall_times;
    std::vector<double> api_overhead_times;
    std::vector<double> internal_other_times;
    std::vector<double> input_pack;
    std::vector<double> layout;
    std::vector<double> result;
    std::vector<double> objective;
    std::vector<double> unattributed_internal;
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
        cpu_view = clqr::Solve(problem, workspace);
        const auto cpu_stop = Clock::now();
        if (cpu_view.status != clqr::SolveStatus::kOptimal)
          return 1;
        cpu_times.push_back(
            std::chrono::duration<double, std::milli>(cpu_stop - cpu_start)
                .count());
      }

      const auto gpu_start = Clock::now();
      gpu_view =
          clqr::cuda::SolvePreparedView(problem, cuda_workspace, cuda_options);
      const auto gpu_stop = Clock::now();
      if (gpu_view.status != clqr::SolveStatus::kOptimal) {
        std::cout << "# CUDA timed solve failed at N=" << horizon
                  << ", repeat=" << repeat << ": " << gpu_view.message << "\n";
        repeat_failed = true;
        break;
      }
      const double gpu_wall_time =
          std::chrono::duration<double, std::milli>(gpu_stop - gpu_start)
              .count();
      const double gpu_kernel_time = KernelTotal(gpu_view.timings);
      gpu_wall_times.push_back(gpu_wall_time);
      gpu_internal_times.push_back(gpu_view.timings.total_ms);
      gpu_kernel_times.push_back(gpu_kernel_time);
      other_wall_times.push_back(gpu_wall_time - EventTotal(gpu_view.timings));
      api_overhead_times.push_back(gpu_wall_time - gpu_view.timings.total_ms);
      internal_other_times.push_back(gpu_view.timings.total_ms -
                                     EventTotal(gpu_view.timings));
      input_pack.push_back(gpu_view.timings.input_pack_ms);
      layout.push_back(gpu_view.timings.layout_ms);
      result.push_back(gpu_view.timings.result_ms);
      objective.push_back(gpu_view.timings.objective_ms);
      unattributed_internal.push_back(
          gpu_view.timings.total_ms - EventTotal(gpu_view.timings) -
          gpu_view.timings.input_pack_ms - gpu_view.timings.layout_ms -
          gpu_view.timings.result_ms);
      upload.push_back(gpu_view.timings.upload_ms);
      feasibility.push_back(gpu_view.timings.feasibility_ms);
      reduction.push_back(gpu_view.timings.reduction_ms);
      riccati.push_back(gpu_view.timings.riccati_ms);
      reconstruction.push_back(gpu_view.timings.reconstruction_ms);
      multiplier.push_back(gpu_view.timings.multiplier_ms);
      download.push_back(gpu_view.timings.download_ms);
    }
    if (repeat_failed)
      continue;
    const double cpu_ms =
        run_cpu ? Median(cpu_times) : std::numeric_limits<double>::quiet_NaN();
    const double gpu_wall_ms = Median(gpu_wall_times);
    const double gpu_internal_ms = Median(gpu_internal_times);
    const double gpu_kernel_ms = Median(gpu_kernel_times);
    int min_reduced_n = std::numeric_limits<int>::max();
    int min_reduced_m = std::numeric_limits<int>::max();
    for (std::size_t index = 0; index < gpu_view.reduced_state_dimensions.size;
         ++index)
      min_reduced_n =
          std::min(min_reduced_n, gpu_view.reduced_state_dimensions[index]);
    for (std::size_t index = 0;
         index < gpu_view.reduced_control_dimensions.size; ++index)
      min_reduced_m =
          std::min(min_reduced_m, gpu_view.reduced_control_dimensions[index]);
    clqr::cuda::Materialize(gpu_view, gpu);
    const Scalar cpu_kkt_residual =
        run_cpu ? MaxKktResidual(problem, CopyCpuSolution(cpu_view))
                : std::numeric_limits<Scalar>::quiet_NaN();
    const Scalar cuda_kkt_residual = MaxKktResidual(problem, gpu);
    std::cout << horizon << ',' << n << ',' << m << ',' << p << ',' << repeats
              << ',' << std::fixed << std::setprecision(6) << cpu_ms << ','
              << gpu_wall_ms << ',' << gpu_internal_ms << ',' << gpu_kernel_ms
              << ',' << cpu_ms / gpu_wall_ms << ',' << cpu_ms / gpu_kernel_ms
              << ',' << Median(other_wall_times) << ','
              << Median(api_overhead_times) << ','
              << Median(internal_other_times) << ',' << Median(input_pack)
              << ',' << Median(layout) << ',' << Median(result) << ','
              << Median(objective) << ','
              << gpu_view.timings.synchronization_count << ','
              << Median(unattributed_internal) << ',' << Median(upload) << ','
              << Median(feasibility) << ',' << Median(reduction) << ','
              << Median(riccati) << ',' << Median(reconstruction) << ','
              << Median(multiplier) << ',' << Median(download) << ','
              << min_reduced_n << ',' << min_reduced_m << ','
              << std::scientific << cpu_kkt_residual << ','
              << cuda_kkt_residual << '\n';
    ++completed_horizons;
  }
  return completed_horizons == 0 ? 1 : 0;
}
