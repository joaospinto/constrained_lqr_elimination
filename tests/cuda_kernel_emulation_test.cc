#define CLQR_CUDA_EMULATION
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../src/cuda_solver.cu"
#include "cuda_jax_problem.h"

namespace {

using clqr::Matrix;
using clqr::Problem;
using clqr::Stage;
using clqr::Vector;
using namespace clqr::cuda;
using namespace clqr::cuda::detail;

constexpr double kTolerance = 1e-9;
constexpr double kMultiplierTolerance = kMinimumMultiplierTolerance;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

double Value(int seed, std::size_t row, std::size_t col = 0) {
  const double x = static_cast<double>(seed * 83 + row * 29 + col * 43);
  return std::sin(0.019 * x) + 0.25 * std::cos(0.037 * x);
}

Matrix GeneratedMatrix(std::size_t rows, std::size_t cols, int seed,
                       double scale) {
  Matrix out(rows, cols);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col)
      out(row, col) = scale * Value(seed, row, col);
  return out;
}

Vector GeneratedVector(std::size_t size, int seed, double scale) {
  Vector out(size);
  for (std::size_t row = 0; row < size; ++row)
    out[row] = scale * Value(seed, row);
  return out;
}

Matrix PositiveDefinite(std::size_t size, int seed, double diagonal) {
  Matrix g = GeneratedMatrix(size, size, seed, 0.18);
  Matrix out = clqr::Transpose(g) * g;
  for (std::size_t row = 0; row < size; ++row) out(row, row) += diagonal;
  return out;
}

double RowDot(const Matrix& matrix, std::size_t row, const Vector& vector) {
  double value = 0.0;
  for (std::size_t col = 0; col < vector.size(); ++col)
    value += matrix(row, col) * vector[col];
  return value;
}

Problem MakeProblem() {
  constexpr std::size_t horizon = 5;
  constexpr std::size_t n = 4;
  constexpr std::size_t m = 3;
  constexpr std::size_t p = 2;
  Problem problem;
  std::vector<Vector> x(horizon + 1);
  std::vector<Vector> u(horizon);
  for (std::size_t i = 0; i <= horizon; ++i)
    x[i] = GeneratedVector(n, 100 + static_cast<int>(i), 0.5);
  for (std::size_t i = 0; i < horizon; ++i)
    u[i] = GeneratedVector(m, 200 + static_cast<int>(i), 0.4);
  problem.initial_state = x[0];
  problem.stages.resize(horizon);
  for (std::size_t i = 0; i < horizon; ++i) {
    Stage& stage = problem.stages[i];
    stage.A = GeneratedMatrix(n, n, 300 + static_cast<int>(i), 0.12);
    for (std::size_t row = 0; row < n; ++row) stage.A(row, row) += 0.8;
    stage.B = GeneratedMatrix(n, m, 400 + static_cast<int>(i), 0.22);
    stage.c = x[i + 1] - stage.A * x[i] - stage.B * u[i];
    stage.Q = PositiveDefinite(n, 500 + static_cast<int>(i), 1.0);
    stage.R = PositiveDefinite(m, 600 + static_cast<int>(i), 1.4);
    stage.M = GeneratedMatrix(n, m, 700 + static_cast<int>(i), 0.025);
    stage.q = GeneratedVector(n, 800 + static_cast<int>(i), 0.15);
    stage.r = GeneratedVector(m, 900 + static_cast<int>(i), 0.15);
    stage.C = Matrix(0, n);
    stage.D = Matrix(0, m);
    stage.d = Vector(0);
    stage.E = Matrix(0, n);
    stage.e = Vector(0);
    if (i % 2 == 0) {
      stage.C = GeneratedMatrix(p, n, 1000 + static_cast<int>(i), 0.3);
      stage.D = GeneratedMatrix(p, m, 1100 + static_cast<int>(i), 0.3);
      for (std::size_t col = 0; col < n; ++col)
        stage.C(1, col) = 2.0 * stage.C(0, col);
      for (std::size_t col = 0; col < m; ++col)
        stage.D(1, col) = 2.0 * stage.D(0, col);
      stage.d = Vector(p);
      stage.d[0] = -(RowDot(stage.C, 0, x[i]) + RowDot(stage.D, 0, u[i]));
      stage.d[1] = 2.0 * stage.d[0];
    } else {
      stage.E = GeneratedMatrix(p, n, 1200 + static_cast<int>(i), 0.3);
      for (std::size_t col = 0; col < n; ++col)
        stage.E(1, col) = -3.0 * stage.E(0, col);
      stage.e = Vector(p);
      stage.e[0] = -RowDot(stage.E, 0, x[i]);
      stage.e[1] = -3.0 * stage.e[0];
    }
  }
  problem.terminal_Q = PositiveDefinite(n, 1300, 1.5);
  problem.terminal_q = GeneratedVector(n, 1400, 0.15);
  problem.terminal_E = GeneratedMatrix(1, n, 1500, 0.3);
  problem.terminal_e = Vector{-RowDot(problem.terminal_E, 0, x.back())};
  return problem;
}

Problem UniformProblem(int seed, std::size_t horizon, std::size_t n,
                       std::size_t m) {
  Problem problem;
  std::vector<Vector> x(horizon + 1);
  std::vector<Vector> u(horizon);
  for (std::size_t i = 0; i <= horizon; ++i)
    x[i] = GeneratedVector(n, seed + 100 + static_cast<int>(i), 0.5);
  for (std::size_t i = 0; i < horizon; ++i)
    u[i] = GeneratedVector(m, seed + 200 + static_cast<int>(i), 0.4);
  problem.initial_state = x.front();
  problem.stages.resize(horizon);
  for (std::size_t i = 0; i < horizon; ++i) {
    Stage& stage = problem.stages[i];
    stage.A = GeneratedMatrix(n, n, seed + 300 + static_cast<int>(i), 0.08);
    for (std::size_t row = 0; row < n; ++row) stage.A(row, row) += 0.9;
    stage.B = GeneratedMatrix(n, m, seed + 400 + static_cast<int>(i), 0.2);
    stage.c = x[i + 1] - stage.A * x[i] - stage.B * u[i];
    stage.Q = PositiveDefinite(n, seed + 500 + static_cast<int>(i), 1.0);
    stage.R = PositiveDefinite(m, seed + 600 + static_cast<int>(i), 1.4);
    stage.M = GeneratedMatrix(n, m, seed + 700 + static_cast<int>(i), 0.02);
    stage.q = GeneratedVector(n, seed + 800 + static_cast<int>(i), 0.15);
    stage.r = GeneratedVector(m, seed + 900 + static_cast<int>(i), 0.15);
    stage.C = Matrix(0, n);
    stage.D = Matrix(0, m);
    stage.d = Vector(0);
    stage.E = Matrix(0, n);
    stage.e = Vector(0);
  }
  problem.terminal_Q = PositiveDefinite(n, seed + 1000, 1.5);
  problem.terminal_q = GeneratedVector(n, seed + 1100, 0.15);
  problem.terminal_E = Matrix(0, n);
  problem.terminal_e = Vector(0);
  return problem;
}

Problem ZeroHorizonProblem() {
  Problem problem;
  problem.initial_state = Vector{0.4, -0.2, 0.7};
  problem.terminal_Q = PositiveDefinite(3, 1600, 1.2);
  problem.terminal_q = GeneratedVector(3, 1610, 0.2);
  problem.terminal_E = Matrix(0, 3);
  problem.terminal_e = Vector(0);
  return problem;
}

Problem ZeroControlStateConstraintProblem() {
  constexpr int seed = 1900;
  constexpr std::size_t horizon = 4;
  constexpr std::size_t n = 3;
  Problem problem = UniformProblem(seed, horizon, n, 0);
  for (std::size_t i = 0; i < horizon; ++i) {
    Stage& stage = problem.stages[i];
    const Vector nominal_x =
        GeneratedVector(n, seed + 100 + static_cast<int>(i), 0.5);
    stage.E = GeneratedMatrix(1, n, seed + 1200 + static_cast<int>(i), 0.3);
    stage.e = Vector{-RowDot(stage.E, 0, nominal_x)};
  }
  return problem;
}

Problem MaximumConstraintProblem() {
  constexpr int seed = 1700;
  constexpr std::size_t dimension = kMaxStateDimension;
  Problem problem = UniformProblem(seed, 1, dimension, dimension);
  Stage& stage = problem.stages[0];
  const Vector nominal_u = GeneratedVector(dimension, seed + 200, 0.4);
  stage.C = GeneratedMatrix(dimension, dimension, seed + 1200, 0.1);
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

Problem MoreMixedRowsThanControlsProblem() {
  constexpr int seed = 2000;
  constexpr std::size_t horizon = 4;
  constexpr std::size_t n = 4;
  constexpr std::size_t m = 1;
  constexpr std::size_t rows = 3;
  Problem problem = UniformProblem(seed, horizon, n, m);
  for (std::size_t i = 0; i < horizon; ++i) {
    Stage& stage = problem.stages[i];
    const Vector nominal_x =
        GeneratedVector(n, seed + 100 + static_cast<int>(i), 0.5);
    const Vector nominal_u =
        GeneratedVector(m, seed + 200 + static_cast<int>(i), 0.4);
    stage.C = GeneratedMatrix(rows, n, seed + 1200 + static_cast<int>(i), 0.3);
    stage.D = GeneratedMatrix(rows, m, seed + 1300 + static_cast<int>(i), 0.3);
    stage.d = Vector(rows);
    for (std::size_t row = 0; row < rows; ++row) {
      const double scale = row == 0 ? 1e-7 : (row == 2 ? 1e7 : 1.0);
      for (std::size_t col = 0; col < n; ++col) stage.C(row, col) *= scale;
      for (std::size_t col = 0; col < m; ++col) stage.D(row, col) *= scale;
      stage.d[row] =
          -(RowDot(stage.C, row, nominal_x) + RowDot(stage.D, row, nominal_u));
    }
  }
  return problem;
}

template <std::size_t Size>
void PackMatrix(const Matrix& source, double (&target)[Size],
                std::size_t stride) {
  for (std::size_t row = 0; row < source.rows(); ++row)
    for (std::size_t col = 0; col < source.cols(); ++col)
      target[row * stride + col] = source(row, col);
}

template <std::size_t Size>
void PackVector(const Vector& source, double (&target)[Size]) {
  for (std::size_t row = 0; row < source.size(); ++row)
    target[row] = source[row];
}

PackedStage Pack(const Stage& source) {
  PackedStage out;
  out.n = static_cast<int>(source.A.cols());
  out.next_n = static_cast<int>(source.A.rows());
  out.m = static_cast<int>(source.B.cols());
  out.mixed = static_cast<int>(source.C.rows());
  out.state = static_cast<int>(source.E.rows());
  PackMatrix(source.A, out.A, kMaxStateDimension);
  PackMatrix(source.B, out.B, kMaxControlDimension);
  PackVector(source.c, out.c);
  PackMatrix(source.Q, out.Q, kMaxStateDimension);
  PackMatrix(source.R, out.R, kMaxControlDimension);
  PackMatrix(source.M, out.M, kMaxControlDimension);
  PackVector(source.q, out.q);
  PackVector(source.r, out.r);
  PackMatrix(source.C, out.C, kMaxStateDimension);
  PackMatrix(source.D, out.D, kMaxControlDimension);
  PackVector(source.d, out.d);
  PackMatrix(source.E, out.E, kMaxStateDimension);
  PackVector(source.e, out.e);
  return out;
}

PackedTerminal Pack(const Problem& problem) {
  PackedTerminal out;
  out.n = static_cast<int>(problem.terminal_Q.rows());
  out.state = static_cast<int>(problem.terminal_E.rows());
  PackMatrix(problem.terminal_Q, out.Q, kMaxStateDimension);
  PackVector(problem.terminal_q, out.q);
  PackMatrix(problem.terminal_E, out.E, kMaxStateDimension);
  PackVector(problem.terminal_e, out.e);
  return out;
}

template <typename Function>
void Launch(int blocks, Function function) {
  threadIdx.x = 0;
  blockDim.x = 1;
  gridDim.x = blocks;
  for (int block = 0; block < blocks; ++block) {
    blockIdx.x = block;
    function();
  }
}

double MaxResidual(const Problem& problem, const std::vector<double>& states,
                   const std::vector<double>& controls,
                   const std::vector<double>& initial_multiplier,
                   const std::vector<double>& dynamics,
                   const std::vector<double>& mixed,
                   const std::vector<double>& state_multipliers,
                   const std::vector<double>& terminal_multiplier) {
  double residual = 0.0;
  const int horizon = static_cast<int>(problem.stages.size());
  for (int i = 0; i < horizon; ++i) {
    const Stage& s = problem.stages[i];
    const double* x = states.data() + i * kMaxStateDimension;
    const double* xp = states.data() + (i + 1) * kMaxStateDimension;
    const double* u = controls.data() + i * kMaxControlDimension;
    const double* right = dynamics.data() + i * kMaxStateDimension;
    const double* left = i == 0
                             ? initial_multiplier.data()
                             : dynamics.data() + (i - 1) * kMaxStateDimension;
    for (std::size_t row = 0; row < s.A.rows(); ++row) {
      double value = xp[row] - s.c[row];
      for (std::size_t col = 0; col < s.A.cols(); ++col)
        value -= s.A(row, col) * x[col];
      for (std::size_t col = 0; col < s.B.cols(); ++col)
        value -= s.B(row, col) * u[col];
      residual = std::max(residual, std::abs(value));
    }
    for (std::size_t row = 0; row < s.C.rows(); ++row) {
      double value = s.d[row];
      for (std::size_t col = 0; col < s.C.cols(); ++col)
        value += s.C(row, col) * x[col];
      for (std::size_t col = 0; col < s.D.cols(); ++col)
        value += s.D(row, col) * u[col];
      residual = std::max(residual, std::abs(value));
    }
    for (std::size_t row = 0; row < s.E.rows(); ++row) {
      double value = s.e[row];
      for (std::size_t col = 0; col < s.E.cols(); ++col)
        value += s.E(row, col) * x[col];
      residual = std::max(residual, std::abs(value));
    }
    for (std::size_t row = 0; row < s.A.cols(); ++row) {
      double value = s.q[row] + left[row];
      for (std::size_t col = 0; col < s.Q.cols(); ++col)
        value += s.Q(row, col) * x[col];
      for (std::size_t col = 0; col < s.M.cols(); ++col)
        value += s.M(row, col) * u[col];
      for (std::size_t next = 0; next < s.A.rows(); ++next)
        value -= s.A(next, row) * right[next];
      for (std::size_t constraint = 0; constraint < s.C.rows(); ++constraint)
        value +=
            s.C(constraint, row) * mixed[i * kMaxMixedConstraints + constraint];
      for (std::size_t constraint = 0; constraint < s.E.rows(); ++constraint)
        value += s.E(constraint, row) *
                 state_multipliers[i * kMaxStateConstraints + constraint];
      residual = std::max(residual, std::abs(value));
    }
    for (std::size_t row = 0; row < s.B.cols(); ++row) {
      double value = s.r[row];
      for (std::size_t col = 0; col < s.M.rows(); ++col)
        value += s.M(col, row) * x[col];
      for (std::size_t col = 0; col < s.R.cols(); ++col)
        value += s.R(row, col) * u[col];
      for (std::size_t next = 0; next < s.B.rows(); ++next)
        value -= s.B(next, row) * right[next];
      for (std::size_t constraint = 0; constraint < s.D.rows(); ++constraint)
        value +=
            s.D(constraint, row) * mixed[i * kMaxMixedConstraints + constraint];
      residual = std::max(residual, std::abs(value));
    }
  }
  const double* terminal = states.data() + horizon * kMaxStateDimension;
  const double* left =
      horizon == 0 ? initial_multiplier.data()
                   : dynamics.data() + (horizon - 1) * kMaxStateDimension;
  for (std::size_t row = 0; row < problem.terminal_Q.rows(); ++row) {
    double value = problem.terminal_q[row] + left[row];
    for (std::size_t col = 0; col < problem.terminal_Q.cols(); ++col)
      value += problem.terminal_Q(row, col) * terminal[col];
    for (std::size_t constraint = 0; constraint < problem.terminal_E.rows();
         ++constraint)
      value +=
          problem.terminal_E(constraint, row) * terminal_multiplier[constraint];
    residual = std::max(residual, std::abs(value));
  }
  return residual;
}

void RunEmulation(const Problem& problem, const std::string& name,
                  bool expect_reduced_state, bool expect_reduced_control) {
  const int horizon = static_cast<int>(problem.stages.size());
  const int nodes = horizon + 1;
  std::vector<PackedStage> stages;
  for (const Stage& stage : problem.stages) stages.push_back(Pack(stage));
  const PackedTerminal terminal = Pack(problem);
  std::vector<double> initial(kMaxStateDimension);
  for (std::size_t row = 0; row < problem.initial_state.size(); ++row)
    initial[row] = problem.initial_state[row];
  DeviceStatus status;

  std::vector<Relation> relation_a(nodes), relation_b(nodes);
  Launch(nodes, [&] {
    BuildPrimalLeavesKernel(stages.data(), horizon, &terminal, kTolerance,
                            relation_a.data(), &status);
  });
  Relation* suffix = relation_a.data();
  Relation* relation_output = relation_b.data();
  for (int offset = 1; offset < nodes; offset *= 2) {
    std::fill(relation_output, relation_output + nodes, Relation{});
    Launch(nodes, [&] {
      SuffixRelationsKernel(suffix, nodes, offset, kTolerance, relation_output,
                            &status);
    });
    std::swap(suffix, relation_output);
  }
  std::vector<StateParam> state_params(nodes);
  Launch(nodes, [&] {
    StateParamKernel(suffix, nodes, state_params.data(), &status, kTolerance);
  });
  Expect(status.code == kDeviceOk, "emulated feasibility scan");

  std::vector<ControlParam> control_params(horizon);
  std::vector<ReducedStage> reduced(horizon);
  ReducedTerminal reduced_terminal;
  std::vector<double> reduced_initial(kMaxStateDimension);
  Launch(horizon, [&] {
    ReduceStagesKernel(stages.data(), suffix, state_params.data(), horizon,
                       kTolerance, control_params.data(), reduced.data(),
                       &status);
  });
  Launch(1, [&] {
    ReduceTerminalKernel(&terminal, state_params.data(), horizon,
                         &reduced_terminal);
  });
  Launch(1, [&] {
    InitialReducedStateKernel(state_params.data(), initial.data(),
                              reduced_initial.data(), kTolerance, &status);
  });
  Expect(status.code == kDeviceOk,
         name + " emulated independent reduction (stage=" +
             std::to_string(status.stage) +
             ", detail=" + std::to_string(status.detail) + ")");
  bool reduced_a_state = false;
  bool reduced_a_control = false;
  for (const StateParam& param : state_params)
    reduced_a_state |= param.reduced_dim < param.physical_dim;
  for (const ControlParam& param : control_params)
    reduced_a_control |= param.reduced_dim < param.physical_dim;
  if (expect_reduced_state)
    Expect(reduced_a_state, name + " exercises smaller state dimensions");
  if (expect_reduced_control)
    Expect(reduced_a_control, name + " exercises smaller control dimensions");

  std::vector<ValueElement> value_a(nodes), value_b(nodes);
  std::vector<Feedback> feedback(horizon);
  int parallel_ok = 1;
  Launch(nodes, [&] {
    BuildValueElementsKernel(reduced.data(), &reduced_terminal, horizon,
                             kTolerance, value_a.data(), &parallel_ok, &status);
  });
  Expect(parallel_ok == 1, "parallel value base applicability");
  ValueElement* value_suffix = value_a.data();
  ValueElement* value_output = value_b.data();
  for (int offset = 1; offset < nodes; offset *= 2) {
    std::fill(value_output, value_output + nodes, ValueElement{});
    Launch(nodes, [&] {
      SuffixValueElementsKernel(value_suffix, nodes, offset, kTolerance,
                                value_output, &parallel_ok);
    });
    std::swap(value_suffix, value_output);
  }
  Expect(parallel_ok == 1, "parallel value scan");
  Launch(horizon, [&] {
    FeedbackKernel(reduced.data(), value_suffix, horizon, kTolerance,
                   feedback.data(), &status);
  });
  Expect(status.code == kDeviceOk, "emulated feedback solve");

  std::vector<ValueElement> sequential_values(nodes);
  std::vector<Feedback> sequential_feedback(horizon);
  int sequential_base_ok = 1;
  Launch(nodes, [&] {
    BuildValueElementsKernel(reduced.data(), &reduced_terminal, horizon,
                             kTolerance, sequential_values.data(),
                             &sequential_base_ok, &status);
  });
  Launch(1, [&] {
    SequentialRiccatiKernel(reduced.data(), horizon, kTolerance,
                            sequential_values.data(),
                            sequential_feedback.data(), &status);
  });
  Expect(status.code == kDeviceOk, "emulated sequential Riccati fallback");
  for (int i = 0; i < horizon; ++i) {
    for (int row = 0; row < feedback[i].control_dim; ++row) {
      Expect(
          std::abs(feedback[i].k[row] - sequential_feedback[i].k[row]) < 2e-8,
          "parallel and sequential feedforward terms agree");
      for (int col = 0; col < feedback[i].state_dim; ++col) {
        Expect(
            std::abs(feedback[i].K[row * kMaxStateDimension + col] -
                     sequential_feedback[i].K[row * kMaxStateDimension + col]) <
                2e-8,
            "parallel and sequential feedback terms agree");
      }
    }
  }

  std::vector<AffineMap> map_a(horizon), map_b(horizon);
  Launch(horizon, [&] {
    InitializeAffineMapsKernel(feedback.data(), horizon, map_a.data());
  });
  AffineMap* prefix = map_a.data();
  AffineMap* map_output = map_b.data();
  for (int offset = 1; offset < horizon; offset *= 2) {
    std::fill(map_output, map_output + horizon, AffineMap{});
    Launch(horizon, [&] {
      PrefixAffineMapsKernel(prefix, horizon, offset, map_output, &status);
    });
    std::swap(prefix, map_output);
  }
  std::vector<double> reduced_states(nodes * kMaxStateDimension);
  std::vector<double> states(nodes * kMaxStateDimension);
  std::vector<double> controls(horizon * kMaxControlDimension);
  Launch(nodes, [&] {
    EvaluateReducedStatesKernel(prefix, horizon, reduced_initial.data(),
                                reduced_states.data());
  });
  Launch(nodes, [&] {
    ReconstructPrimalKernel(state_params.data(), control_params.data(),
                            feedback.data(), reduced_states.data(), horizon,
                            states.data(), controls.data());
  });
  Expect(status.code == kDeviceOk, "emulated affine rollout");

  clqr::Workspace workspace;
  workspace.Reserve(problem);
  const clqr::SolutionView cpu = clqr::Solve(problem, workspace);
  Expect(cpu.status == clqr::SolveStatus::kOptimal, "CPU reference status");
  for (int i = 0; i < nodes; ++i) {
    for (std::size_t row = 0; row < cpu.states[i].size; ++row) {
      Expect(std::abs(states[i * kMaxStateDimension + row] -
                      cpu.states[i][row]) < 2e-7,
             "emulated state matches CPU before dual recovery at " +
                 std::to_string(i) + "," + std::to_string(row) + ": emulated=" +
                 std::to_string(states[i * kMaxStateDimension + row]) +
                 ", CPU=" + std::to_string(cpu.states[i][row]));
    }
  }
  for (int i = 0; i < horizon; ++i) {
    for (std::size_t row = 0; row < cpu.controls[i].size; ++row) {
      Expect(std::abs(controls[i * kMaxControlDimension + row] -
                      cpu.controls[i][row]) < 2e-7,
             "emulated control matches CPU before dual recovery at " +
                 std::to_string(i) + "," + std::to_string(row) + ": emulated=" +
                 std::to_string(controls[i * kMaxControlDimension + row]) +
                 ", CPU=" + std::to_string(cpu.controls[i][row]));
    }
  }

  int padded = 1;
  while (padded < nodes) padded *= 2;
  std::vector<int> level_offsets{0};
  std::vector<int> level_counts{padded};
  int total = padded;
  while (level_counts.back() > 1) {
    level_offsets.push_back(total);
    level_counts.push_back(level_counts.back() / 2);
    total += level_counts.back();
  }
  std::vector<Relation> dual_tree(total);
  std::vector<NodeValue> dual_values(total);
  Launch(padded, [&] {
    BuildDualLeavesKernel(stages.data(), &terminal, horizon, padded,
                          states.data(), controls.data(), kMultiplierTolerance,
                          dual_tree.data(), &status);
  });
  for (std::size_t level = 0; level + 1 < level_counts.size(); ++level) {
    Launch(level_counts[level + 1], [&] {
      ReduceDualTreeLevelKernel(dual_tree.data(), level_offsets[level],
                                level_offsets[level + 1],
                                level_counts[level + 1], kMultiplierTolerance,
                                dual_tree.data(), &status);
    });
  }
  const int root = level_offsets.back();
  Launch(1, [&] {
    SolveDualRootKernel(dual_tree.data() + root, dual_values.data() + root,
                        &status, kMultiplierTolerance);
  });
  for (int level = static_cast<int>(level_counts.size()) - 2; level >= 0;
       --level) {
    Launch(level_counts[level + 1], [&] {
      ExpandDualTreeLevelKernel(
          dual_tree.data(), level_offsets[level], level_offsets[level + 1],
          level_counts[level + 1], kMultiplierTolerance, dual_values.data(),
          dual_values.data(), &status);
    });
  }
  std::vector<double> initial_multiplier(kMaxStateDimension);
  std::vector<double> dynamics(horizon * kMaxStateDimension);
  std::vector<double> mixed(horizon * kMaxMixedConstraints);
  std::vector<double> state_multipliers(horizon * kMaxStateConstraints);
  std::vector<double> terminal_multiplier(kMaxStateConstraints);
  Launch(nodes, [&] {
    RecoverLocalMultipliersKernel(
        stages.data(), &terminal, horizon, states.data(), controls.data(),
        dual_values.data(), kMultiplierTolerance, initial_multiplier.data(),
        dynamics.data(), mixed.data(), state_multipliers.data(),
        terminal_multiplier.data(), &status);
  });
  Expect(status.code == kDeviceOk,
         "emulated multiplier recovery (stage=" + std::to_string(status.stage) +
             ", detail=" + std::to_string(status.detail) + ")");

  for (int i = 0; i < nodes; ++i) {
    for (std::size_t row = 0; row < cpu.states[i].size; ++row) {
      Expect(std::abs(states[i * kMaxStateDimension + row] -
                      cpu.states[i][row]) < 2e-7,
             "emulated state matches CPU");
    }
  }
  for (int i = 0; i < horizon; ++i) {
    for (std::size_t row = 0; row < cpu.controls[i].size; ++row) {
      Expect(std::abs(controls[i * kMaxControlDimension + row] -
                      cpu.controls[i][row]) < 2e-7,
             "emulated control matches CPU");
    }
  }
  const double residual =
      MaxResidual(problem, states, controls, initial_multiplier, dynamics,
                  mixed, state_multipliers, terminal_multiplier);
  Expect(residual < 2e-7, "emulated full KKT residual");
  std::cout << name
            << " CUDA kernel emulation matched CPU; KKT residual=" << residual
            << '\n';
}

}  // namespace

int main() {
  RunEmulation(MakeProblem(), "rank-deficient constrained", true, true);
  RunEmulation(ZeroHorizonProblem(), "zero-horizon", false, false);
  RunEmulation(
      UniformProblem(1800, 3, kMaxStateDimension, kMaxControlDimension),
      "maximum-active-dimension", false, false);
  RunEmulation(ZeroControlStateConstraintProblem(), "zero-control", true,
               false);
  RunEmulation(MaximumConstraintProblem(), "maximum-constraint", true, true);
  RunEmulation(MoreMixedRowsThanControlsProblem(), "more-mixed-than-controls",
               true, true);
  RunEmulation(clqr::test::MakeJaxCrossValidationProblem(), "exact-JAX-fixture",
               true, false);
  return 0;
}
