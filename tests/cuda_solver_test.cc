#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "clqr/cuda.h"

namespace {

using clqr::Matrix;
using clqr::Problem;
using clqr::SolveStatus;
using clqr::Stage;
using clqr::Vector;

constexpr double kTolerance = 3e-6;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

double DeterministicValue(int seed, std::size_t i, std::size_t j = 0) {
  const double x = static_cast<double>(
      (seed + 17) * 131 + static_cast<int>(i) * 37 + static_cast<int>(j) * 53);
  return std::sin(0.013 * x) + 0.35 * std::cos(0.031 * x);
}

Vector GeneratedVector(std::size_t size, int seed, double scale = 1.0) {
  Vector out(size);
  for (std::size_t i = 0; i < size; ++i)
    out[i] = scale * DeterministicValue(seed, i);
  return out;
}

Matrix GeneratedMatrix(std::size_t rows, std::size_t cols, int seed,
                       double scale = 1.0) {
  Matrix out(rows, cols);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col)
      out(row, col) = scale * DeterministicValue(seed, row, col);
  }
  return out;
}

Matrix PositiveDefinite(std::size_t size, int seed, double diagonal) {
  Matrix g = GeneratedMatrix(size, size, seed, 0.2);
  Matrix out = clqr::Transpose(g) * g;
  for (std::size_t i = 0; i < size; ++i) out(i, i) += diagonal;
  return out;
}

double RowDot(const Matrix& matrix, std::size_t row, const Vector& vector) {
  double value = 0.0;
  for (std::size_t col = 0; col < vector.size(); ++col)
    value += matrix(row, col) * vector[col];
  return value;
}

enum class ConstraintMode {
  kNone,
  kState,
  kMixed,
  kAlternating,
  kRedundantMixed,
  kTerminal,
};

Problem GeneratedProblem(int seed, std::size_t horizon, std::size_t n,
                         std::size_t m, std::size_t p, ConstraintMode mode) {
  Problem problem;
  std::vector<Vector> nominal_x(horizon + 1);
  std::vector<Vector> nominal_u(horizon);
  for (std::size_t i = 0; i <= horizon; ++i)
    nominal_x[i] = GeneratedVector(n, seed + 100 + static_cast<int>(i), 0.6);
  for (std::size_t i = 0; i < horizon; ++i)
    nominal_u[i] = GeneratedVector(m, seed + 200 + static_cast<int>(i), 0.5);
  problem.initial_state = nominal_x[0];
  problem.stages.resize(horizon);
  for (std::size_t i = 0; i < horizon; ++i) {
    Stage& stage = problem.stages[i];
    stage.A = GeneratedMatrix(n, n, seed + 10 * static_cast<int>(i), 0.15);
    for (std::size_t row = 0; row < n; ++row) stage.A(row, row) += 0.75;
    stage.B = GeneratedMatrix(n, m, seed + 300 + static_cast<int>(i), 0.25);
    stage.c =
        nominal_x[i + 1] - stage.A * nominal_x[i] - stage.B * nominal_u[i];
    stage.Q = PositiveDefinite(n, seed + 400 + static_cast<int>(i), 1.0);
    stage.R = PositiveDefinite(m, seed + 500 + static_cast<int>(i), 1.4);
    stage.M = GeneratedMatrix(n, m, seed + 600 + static_cast<int>(i), 0.04);
    stage.q = GeneratedVector(n, seed + 700 + static_cast<int>(i), 0.2);
    stage.r = GeneratedVector(m, seed + 800 + static_cast<int>(i), 0.2);
    stage.C = Matrix(0, n);
    stage.D = Matrix(0, m);
    stage.d = Vector(0);
    stage.E = Matrix(0, n);
    stage.e = Vector(0);

    const bool mixed = mode == ConstraintMode::kMixed ||
                       mode == ConstraintMode::kRedundantMixed ||
                       (mode == ConstraintMode::kAlternating && i % 2 == 0);
    const bool state = mode == ConstraintMode::kState ||
                       (mode == ConstraintMode::kAlternating && i % 2 == 1);
    if (mixed && p > 0) {
      stage.C = GeneratedMatrix(p, n, seed + 900 + static_cast<int>(i), 0.45);
      stage.D = GeneratedMatrix(p, m, seed + 1000 + static_cast<int>(i), 0.45);
      if (mode == ConstraintMode::kRedundantMixed && p >= 2) {
        for (std::size_t col = 0; col < n; ++col)
          stage.C(1, col) = 2.0 * stage.C(0, col);
        for (std::size_t col = 0; col < m; ++col)
          stage.D(1, col) = 2.0 * stage.D(0, col);
      }
      stage.d = Vector(p);
      for (std::size_t row = 0; row < p; ++row) {
        stage.d[row] = -(RowDot(stage.C, row, nominal_x[i]) +
                         RowDot(stage.D, row, nominal_u[i]));
      }
    } else if (state && p > 0) {
      stage.E = GeneratedMatrix(p, n, seed + 1100 + static_cast<int>(i), 0.45);
      stage.e = Vector(p);
      for (std::size_t row = 0; row < p; ++row)
        stage.e[row] = -RowDot(stage.E, row, nominal_x[i]);
    }
  }
  problem.terminal_Q = PositiveDefinite(n, seed + 1200, 1.5);
  problem.terminal_q = GeneratedVector(n, seed + 1300, 0.2);
  problem.terminal_E = Matrix(0, n);
  problem.terminal_e = Vector(0);
  if (mode == ConstraintMode::kTerminal && p > 0) {
    problem.terminal_E = GeneratedMatrix(p, n, seed + 1400, 0.45);
    problem.terminal_e = Vector(p);
    for (std::size_t row = 0; row < p; ++row)
      problem.terminal_e[row] =
          -RowDot(problem.terminal_E, row, nominal_x.back());
  }
  return problem;
}

Problem NonuniformProblem() {
  const std::vector<std::size_t> dimensions{3, 4, 2, 5};
  const std::vector<std::size_t> controls{2, 3, 1};
  Problem problem;
  std::vector<Vector> nominal_x;
  std::vector<Vector> nominal_u;
  for (std::size_t i = 0; i < dimensions.size(); ++i)
    nominal_x.push_back(
        GeneratedVector(dimensions[i], 2100 + static_cast<int>(i), 0.5));
  for (std::size_t i = 0; i < controls.size(); ++i)
    nominal_u.push_back(
        GeneratedVector(controls[i], 2200 + static_cast<int>(i), 0.4));
  problem.initial_state = nominal_x.front();
  problem.stages.resize(controls.size());
  for (std::size_t i = 0; i < controls.size(); ++i) {
    Stage& stage = problem.stages[i];
    const std::size_t n = dimensions[i];
    const std::size_t next = dimensions[i + 1];
    const std::size_t m = controls[i];
    stage.A = GeneratedMatrix(next, n, 2300 + static_cast<int>(i), 0.2);
    stage.B = GeneratedMatrix(next, m, 2400 + static_cast<int>(i), 0.25);
    stage.c =
        nominal_x[i + 1] - stage.A * nominal_x[i] - stage.B * nominal_u[i];
    stage.Q = PositiveDefinite(n, 2500 + static_cast<int>(i), 1.0);
    stage.R = PositiveDefinite(m, 2600 + static_cast<int>(i), 1.2);
    stage.M = GeneratedMatrix(n, m, 2700 + static_cast<int>(i), 0.03);
    stage.q = GeneratedVector(n, 2800 + static_cast<int>(i), 0.2);
    stage.r = GeneratedVector(m, 2900 + static_cast<int>(i), 0.2);
    stage.C = Matrix(0, n);
    stage.D = Matrix(0, m);
    stage.d = Vector(0);
    stage.E = Matrix(0, n);
    stage.e = Vector(0);
    if (i == 0) {
      stage.C = GeneratedMatrix(1, n, 3000, 0.4);
      stage.D = GeneratedMatrix(1, m, 3010, 0.4);
      stage.d = Vector{-(RowDot(stage.C, 0, nominal_x[i]) +
                         RowDot(stage.D, 0, nominal_u[i]))};
    } else if (i == 1) {
      stage.E = GeneratedMatrix(1, n, 3020, 0.4);
      stage.e = Vector{-RowDot(stage.E, 0, nominal_x[i])};
    }
  }
  problem.terminal_Q = PositiveDefinite(dimensions.back(), 3100, 1.4);
  problem.terminal_q = GeneratedVector(dimensions.back(), 3110, 0.2);
  problem.terminal_E = GeneratedMatrix(1, dimensions.back(), 3120, 0.4);
  problem.terminal_e = Vector{-RowDot(problem.terminal_E, 0, nominal_x.back())};
  return problem;
}

double TestMaxAbs(const Vector& vector) {
  double value = 0.0;
  for (std::size_t i = 0; i < vector.size(); ++i)
    value = std::max(value, std::abs(vector[i]));
  return value;
}

void AddTransposeProduct(const Matrix& matrix, const Vector& multiplier,
                         Vector* target) {
  for (std::size_t col = 0; col < matrix.cols(); ++col) {
    for (std::size_t row = 0; row < matrix.rows(); ++row)
      (*target)[col] += matrix(row, col) * multiplier[row];
  }
}

double MaxKktResidual(const Problem& problem,
                      const clqr::cuda::Solution& solution) {
  double residual = 0.0;
  const std::size_t horizon = problem.stages.size();
  for (std::size_t i = 0; i < horizon; ++i) {
    const Stage& stage = problem.stages[i];
    residual = std::max(
        residual,
        TestMaxAbs(solution.states[i + 1] - stage.A * solution.states[i] -
                   stage.B * solution.controls[i] - stage.c));
    if (stage.C.rows() > 0)
      residual = std::max(residual,
                          TestMaxAbs(stage.C * solution.states[i] +
                                     stage.D * solution.controls[i] + stage.d));
    if (stage.E.rows() > 0)
      residual = std::max(residual,
                          TestMaxAbs(stage.E * solution.states[i] + stage.e));

    Vector gx = stage.Q * solution.states[i] + stage.M * solution.controls[i] +
                stage.q -
                clqr::Transpose(stage.A) * solution.dynamics_multipliers[i];
    gx = gx + (i == 0 ? solution.initial_multiplier
                      : solution.dynamics_multipliers[i - 1]);
    AddTransposeProduct(stage.C, solution.mixed_multipliers[i], &gx);
    AddTransposeProduct(stage.E, solution.state_multipliers[i], &gx);
    residual = std::max(residual, TestMaxAbs(gx));

    Vector gu = clqr::Transpose(stage.M) * solution.states[i] +
                stage.R * solution.controls[i] + stage.r -
                clqr::Transpose(stage.B) * solution.dynamics_multipliers[i];
    AddTransposeProduct(stage.D, solution.mixed_multipliers[i], &gu);
    residual = std::max(residual, TestMaxAbs(gu));
  }
  Vector initial_error = solution.states.front() - problem.initial_state;
  residual = std::max(residual, TestMaxAbs(initial_error));
  if (problem.terminal_E.rows() > 0)
    residual = std::max(residual,
                        TestMaxAbs(problem.terminal_E * solution.states.back() +
                                   problem.terminal_e));
  Vector terminal_gradient =
      problem.terminal_Q * solution.states.back() + problem.terminal_q;
  terminal_gradient =
      terminal_gradient +
      (horizon == 0 ? solution.initial_multiplier
                    : solution.dynamics_multipliers.back());
  AddTransposeProduct(problem.terminal_E, solution.terminal_state_multiplier,
                      &terminal_gradient);
  return std::max(residual, TestMaxAbs(terminal_gradient));
}

void CompareWithCpu(const Problem& problem, const std::string& name,
                    bool expect_parallel = true,
                    const Problem* cpu_reference_problem = nullptr) {
  std::cout << "case: " << name << std::endl;
  const Problem& cpu_problem =
      cpu_reference_problem == nullptr ? problem : *cpu_reference_problem;
  clqr::Workspace workspace;
  workspace.Reserve(cpu_problem);
  const clqr::SolutionView cpu = clqr::Solve(cpu_problem, workspace);
  const clqr::cuda::Solution gpu = clqr::cuda::Solve(problem);
  Expect(cpu.status == SolveStatus::kOptimal,
         name + " CPU status: " + cpu.message);
  Expect(gpu.status == SolveStatus::kOptimal,
         name + " CUDA status: " + gpu.message);
  Expect(gpu.used_parallel_riccati == expect_parallel, name + " Riccati path");
  Expect(gpu.states.size() == cpu.state_count, name + " state count");
  Expect(gpu.controls.size() == cpu.control_count, name + " control count");
  for (std::size_t i = 0; i < cpu.state_count; ++i) {
    Expect(gpu.states[i].size() == cpu.states[i].size, name + " state size");
    for (std::size_t row = 0; row < cpu.states[i].size; ++row) {
      Expect(std::abs(gpu.states[i][row] - cpu.states[i][row]) <= kTolerance,
             name + " state mismatch at " + std::to_string(i) + "," +
                 std::to_string(row));
    }
  }
  for (std::size_t i = 0; i < cpu.control_count; ++i) {
    Expect(gpu.controls[i].size() == cpu.controls[i].size,
           name + " control size");
    for (std::size_t row = 0; row < cpu.controls[i].size; ++row) {
      Expect(
          std::abs(gpu.controls[i][row] - cpu.controls[i][row]) <= kTolerance,
          name + " control mismatch at " + std::to_string(i) + "," +
              std::to_string(row));
    }
  }
  Expect(std::abs(gpu.objective - cpu.objective) <=
             kTolerance * (1.0 + std::abs(cpu.objective)),
         name + " objective mismatch");
  Expect(MaxKktResidual(problem, gpu) <= 2e-5,
         name + " full primal-dual KKT residual");
  for (std::size_t i = 0; i < gpu.reduced_state_dimensions.size(); ++i) {
    Expect(gpu.reduced_state_dimensions[i] <=
               static_cast<int>(gpu.states[i].size()),
           name + " reduced state dimension bound");
  }
  for (std::size_t i = 0; i < gpu.reduced_control_dimensions.size(); ++i) {
    Expect(gpu.reduced_control_dimensions[i] <=
               static_cast<int>(gpu.controls[i].size()),
           name + " reduced control dimension bound");
  }
}

Problem RiccatiFallbackProblem() {
  Problem problem;
  problem.initial_state = Vector{0.4};
  problem.stages.resize(1);
  Stage& stage = problem.stages[0];
  stage.A = Matrix(1, 1, {1.0});
  stage.B = Matrix(1, 1, {1.0});
  stage.c = Vector{0.0};
  stage.Q = Matrix(1, 1, {0.0});
  stage.R = Matrix(1, 1, {0.0});
  stage.M = Matrix(1, 1, {0.0});
  stage.q = Vector{0.0};
  stage.r = Vector{-0.3};
  stage.C = Matrix(0, 1);
  stage.D = Matrix(0, 1);
  stage.d = Vector(0);
  stage.E = Matrix(0, 1);
  stage.e = Vector(0);
  problem.terminal_Q = Matrix(1, 1, {2.0});
  problem.terminal_q = Vector{-0.2};
  problem.terminal_E = Matrix(0, 1);
  problem.terminal_e = Vector(0);
  return problem;
}

Problem ZeroHorizonProblem() {
  Problem problem;
  problem.initial_state = Vector{0.4, -0.2, 0.7};
  problem.terminal_Q = PositiveDefinite(3, 3200, 1.2);
  problem.terminal_q = GeneratedVector(3, 3210, 0.2);
  problem.terminal_E = Matrix(0, 3);
  problem.terminal_e = Vector(0);
  return problem;
}

Problem MaximumConstraintProblem() {
  constexpr int seed = 110;
  constexpr std::size_t dimension = clqr::cuda::kMaxStateDimension;
  Problem problem =
      GeneratedProblem(seed, 1, dimension, dimension, 0, ConstraintMode::kNone);
  Stage& stage = problem.stages[0];
  const Vector nominal_u = GeneratedVector(dimension, seed + 200, 0.5);

  stage.C = GeneratedMatrix(dimension, dimension, 3300, 0.1);
  stage.D = clqr::Identity(dimension);
  stage.d = Vector(dimension);
  for (std::size_t row = 0; row < dimension; ++row) {
    stage.d[row] =
        -(RowDot(stage.C, row, problem.initial_state) + nominal_u[row]);
  }
  stage.E = clqr::Identity(dimension);
  stage.e = Vector(dimension);
  for (std::size_t row = 0; row < dimension; ++row)
    stage.e[row] = -problem.initial_state[row];
  return problem;
}

Problem RescaledMixedRowsProblem(const Problem& unscaled) {
  Problem problem = unscaled;
  for (Stage& stage : problem.stages) {
    for (std::size_t row = 0; row < stage.C.rows(); ++row) {
      const double scale = row == 0 ? 1e-7 : (row == 2 ? 1e7 : 1.0);
      for (std::size_t col = 0; col < stage.C.cols(); ++col)
        stage.C(row, col) *= scale;
      for (std::size_t col = 0; col < stage.D.cols(); ++col)
        stage.D(row, col) *= scale;
      stage.d[row] *= scale;
    }
  }
  return problem;
}

void InfeasibleCase() {
  Problem problem = GeneratedProblem(90, 2, 3, 2, 0, ConstraintMode::kNone);
  problem.stages[0].E = Matrix(2, 3, {1.0, 0.0, 0.0, 1.0, 0.0, 0.0});
  problem.stages[0].e = Vector{-1.0, -2.0};
  const clqr::cuda::Solution solution = clqr::cuda::Solve(problem);
  Expect(solution.status == SolveStatus::kInfeasible,
         "inconsistent redundant rows are infeasible");
}

void InvalidDeviceCases() {
  const Problem problem = ZeroHorizonProblem();
  clqr::cuda::Options options;
  options.device = -1;
  Expect(
      clqr::cuda::Solve(problem, options).status == SolveStatus::kInvalidInput,
      "negative CUDA device index is invalid input");
  options.device = 1000000;
  Expect(
      clqr::cuda::Solve(problem, options).status == SolveStatus::kInvalidInput,
      "out-of-range CUDA device index is invalid input");
}

}  // namespace

int main() {
  if (!clqr::cuda::Available()) {
    std::cout << "CUDA test skipped: " << clqr::cuda::DeviceDescription()
              << "\n";
    return 0;
  }
  std::cout << "testing " << clqr::cuda::DeviceDescription() << "\n";
  CompareWithCpu(GeneratedProblem(1, 5, 4, 3, 0, ConstraintMode::kNone),
                 "unconstrained");
  CompareWithCpu(GeneratedProblem(2, 6, 4, 2, 1, ConstraintMode::kState),
                 "state-only");
  CompareWithCpu(GeneratedProblem(3, 5, 4, 3, 2, ConstraintMode::kMixed),
                 "mixed");
  CompareWithCpu(GeneratedProblem(4, 7, 5, 3, 2, ConstraintMode::kAlternating),
                 "alternating");
  CompareWithCpu(
      GeneratedProblem(5, 5, 4, 2, 2, ConstraintMode::kRedundantMixed),
      "rank-deficient redundant");
  CompareWithCpu(GeneratedProblem(6, 4, 4, 2, 2, ConstraintMode::kTerminal),
                 "terminal constraints");
  CompareWithCpu(NonuniformProblem(), "nonuniform dimensions");
  CompareWithCpu(RiccatiFallbackProblem(), "Riccati fallback", false);
  CompareWithCpu(ZeroHorizonProblem(), "zero horizon");
  CompareWithCpu(GeneratedProblem(100, 3, clqr::cuda::kMaxStateDimension,
                                  clqr::cuda::kMaxControlDimension, 0,
                                  ConstraintMode::kNone),
                 "maximum active dimensions");
  CompareWithCpu(GeneratedProblem(101, 4, 3, 0, 1, ConstraintMode::kState),
                 "zero control dimension");
  CompareWithCpu(GeneratedProblem(102, 5, 4, 1, 3, ConstraintMode::kMixed),
                 "more mixed rows than controls");
  const Problem unscaled_rows =
      GeneratedProblem(120, 4, 4, 2, 3, ConstraintMode::kMixed);
  // Keep the CPU reference well scaled so this case specifically tests
  // whether CUDA rank decisions are invariant to independent row scaling.
  CompareWithCpu(RescaledMixedRowsProblem(unscaled_rows),
                 "independently rescaled rows", true, &unscaled_rows);
  CompareWithCpu(MaximumConstraintProblem(), "maximum constraint dimensions");
  InfeasibleCase();
  InvalidDeviceCases();
  std::cout << "all CUDA cross-validation tests passed\n";
  return 0;
}
