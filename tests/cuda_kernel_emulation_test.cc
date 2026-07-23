#define CLQR_CUDA_EMULATION
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "../benchmarks/cuda_benchmark_problem.h"
#include "../src/cuda_solver.cu"
#include "adversarial_test_support.h"
#include "cuda_jax_problem.h"

namespace {

using clqr::Matrix;
using clqr::Problem;
using clqr::Scalar;
using clqr::Stage;
using clqr::Vector;
using namespace clqr::cuda;
using namespace clqr::cuda::detail;

// Test-only backing strides deliberately exceed the former production
// capacities (8/8/8/8). They are not solver limits.
constexpr int kTestStateCapacity = 24;
constexpr int kTestControlCapacity = 12;
constexpr int kTestMixedCapacity = 10;
constexpr int kTestStateConstraintCapacity = 10;
constexpr int kTestDualCapacity = kTestStateCapacity + kTestMixedCapacity;
constexpr int kTestRelationRows = 2 * kTestStateCapacity;
constexpr int kTestRelationEntries =
    kTestRelationRows * (2 * kTestStateCapacity) + kTestRelationRows;
constexpr int kTestValueEntries =
    3 * kTestStateCapacity * kTestStateCapacity + 2 * kTestStateCapacity;
constexpr int kTestMapEntries =
    kTestStateCapacity * kTestStateCapacity + kTestStateCapacity;
constexpr int kTestDualRelationEntries =
    2 * kTestDualCapacity * (2 * kTestDualCapacity) + 2 * kTestDualCapacity;
constexpr int kTestDualValueEntries = 2 * kTestDualCapacity;

#ifdef CLQR_USE_FLOAT
constexpr Scalar kTolerance = 1e-5f;
// The scan and sequential CPU paths can choose different FP32 pivots on the
// longer generated problems.  Keep their raw trajectory comparison looser than
// the independently checked feasibility and KKT gates below.
constexpr Scalar kPrimalComparisonTolerance = 5e-2f;
constexpr Scalar kKktComparisonTolerance = 3e-2f;
constexpr Scalar kLongHorizonKktComparisonTolerance = 3e-2f;
#else
constexpr Scalar kTolerance = 1e-9;
constexpr Scalar kPrimalComparisonTolerance = 2e-7;
constexpr Scalar kKktComparisonTolerance = 2e-7;
constexpr Scalar kLongHorizonKktComparisonTolerance = 2e-5;
#endif

struct AllowedDeviceFailure {
  int code;
  const char *phase;
  int detail;
};

const AllowedDeviceFailure *
AllowedFailureForCase(const std::string &name) {
#ifdef CLQR_USE_FLOAT
  static const std::pair<const char *, AllowedDeviceFailure> failures[]{
      {"shared-ill-conditioned-horizon-17",
       {kDeviceNumericalFailure, "multiplier recovery", 17}},
      {"shared-extended-horizon-32",
       {kDeviceNumericalFailure, "multiplier recovery", 17}},
      {"shared-extended-horizon-63",
       {kDeviceNumericalFailure, "multiplier recovery", 17}},
      {"shared-extended-horizon-65",
       {kDeviceNumericalFailure, "multiplier recovery", 17}},
      {"shared-extended-horizon-127",
       {kDeviceNumericalFailure, "value scan", 20}},
      {"shared-extended-horizon-257",
       {kDeviceNumericalFailure, "multiplier recovery", 17}},
      {"shared-property-seed-1",
       {kDeviceNumericalFailure, "independent reduction", 7}},
      {"shared-property-seed-5",
       {kDeviceInfeasible, "independent reduction", 6}},
  };
  for (const auto &failure : failures) {
    if (name == failure.first)
      return &failure.second;
  }
#else
  (void)name;
#endif
  return nullptr;
}

void Expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

bool FinishAllowedDeviceFailure(const DeviceStatus &status,
                                const std::string &name,
                                const char *phase,
                                const AllowedDeviceFailure *allowed) {
  if (status.code == kDeviceOk)
    return false;
  Expect(allowed != nullptr, name + " unexpected " + phase + " failure");
  Expect(status.code == allowed->code &&
             std::string(phase) == allowed->phase &&
             status.detail == allowed->detail,
         name + " unexpected device rejection during " + phase + " (stage=" +
             std::to_string(status.stage) +
             ", detail=" + std::to_string(status.detail) + ")");
  std::cout << name
            << " CUDA kernel emulation passed (expected FP32 device rejection "
               "during "
            << phase << " at stage " << status.stage << ", detail "
            << status.detail << ")\n";
  return true;
}

void TinyCoefficientRrefCase() {
  constexpr Scalar rank_tolerance = Scalar{1e-4};
  Scalar matrix[]{rank_tolerance * rank_tolerance, Scalar{1}};
  int pivot_columns[1]{};
  int pivot_rows[1]{};
  int rank = -1;
  int best_row = -1;
  Scalar factors[1]{};
  RrefBlock(matrix, 1, 2, 1, rank_tolerance, pivot_columns, pivot_rows, &rank,
            &best_row, factors, rank_tolerance);
  Expect(rank == 0, "RREF does not amplify a roundoff-level coefficient");
  Expect(matrix[0] == Scalar{0},
         "RREF removes a coefficient below its rank tolerance");
}

void PivotedLuMultiRhsCase() {
#ifdef CLQR_USE_FLOAT
  constexpr Scalar solve_tolerance = 1e-5f;
  constexpr Scalar comparison_tolerance = 2e-5f;
  const Scalar row_scales[]{1e-3f, 1e3f, 1.0f};
#else
  constexpr Scalar solve_tolerance = 1e-9;
  constexpr Scalar comparison_tolerance = 2e-12;
  const Scalar row_scales[]{1e-12, 1e12, 1.0};
#endif
  constexpr int dimension = 3;
  constexpr int right_hand_side_count = 3;
  constexpr int columns = dimension + right_hand_side_count;
  const Scalar coefficient[dimension * dimension]{
      Scalar{0}, Scalar{2}, Scalar{-1}, Scalar{1}, Scalar{-2},
      Scalar{3}, Scalar{4}, Scalar{1},  Scalar{2}};
  const Scalar expected[dimension * right_hand_side_count]{
      Scalar{0.25}, Scalar{-0.4}, Scalar{0.8},  Scalar{-0.7}, Scalar{0.3},
      Scalar{0.1},  Scalar{0.5},  Scalar{-0.2}, Scalar{-0.6}};
  Scalar augmented[dimension * columns]{};
  for (int row = 0; row < dimension; ++row) {
    for (int col = 0; col < dimension; ++col) {
      augmented[row * columns + col] =
          row_scales[row] * coefficient[row * dimension + col];
    }
    for (int rhs = 0; rhs < right_hand_side_count; ++rhs) {
      for (int col = 0; col < dimension; ++col) {
        augmented[row * columns + dimension + rhs] +=
            augmented[row * columns + col] *
            expected[col * right_hand_side_count + rhs];
      }
    }
  }
  Scalar factors[dimension]{};
  int best_row = -1;
  threadIdx.x = 0;
  blockDim.x = 1;
  Expect(SolveGeneralMultipleRhsBlock(augmented, dimension, columns,
                                      solve_tolerance, factors, &best_row),
         "pivoted LU multi-RHS solve");
  for (int row = 0; row < dimension; ++row) {
    for (int rhs = 0; rhs < right_hand_side_count; ++rhs) {
      const int entry = row * right_hand_side_count + rhs;
      Expect(std::abs(augmented[row * columns + dimension + rhs] -
                      expected[entry]) < comparison_tolerance,
             "pivoted LU multi-RHS solution entry " + std::to_string(entry));
    }
  }

  Scalar singular[]{Scalar{1}, Scalar{2}, Scalar{3},
                    Scalar{2}, Scalar{4}, Scalar{6}};
  Scalar singular_factors[2]{};
  best_row = -1;
  Expect(!SolveGeneralMultipleRhsBlock(singular, 2, 3, solve_tolerance,
                                       singular_factors, &best_row),
         "pivoted LU rejects a singular coefficient matrix");
}

void OrthogonalEchelonCase() {
#ifdef CLQR_USE_FLOAT
  constexpr Scalar comparison_tolerance = 2e-4f;
#else
  constexpr Scalar comparison_tolerance = 2e-11;
#endif
  constexpr int rows = 4;
  constexpr int variables = 3;
  constexpr int columns = variables + 1;
  Scalar matrix[rows * columns]{
      Scalar{1e6}, Scalar{2e6},  Scalar{0},    Scalar{5e6},
      Scalar{0},   Scalar{1e-6}, Scalar{1e-6}, Scalar{4e-6},
      Scalar{3},   Scalar{9},    Scalar{3},    Scalar{27},
      Scalar{0},   Scalar{0},    Scalar{0},    Scalar{0}};
  int pivot_columns[variables]{};
  int permutation[variables]{};
  int rank = -1;
  int best_column = -1;
  Scalar reflector[rows]{};
  Scalar matrix_scale = Scalar{0};
  threadIdx.x = 0;
  blockDim.x = 1;

  OrthogonalEchelonBlock(matrix, rows, columns, variables, variables,
                         kTolerance, pivot_columns, permutation, &rank,
                         &best_column, reflector, &matrix_scale);

  Expect(rank == 2, "orthogonal echelon detects the scaled rank");
  bool pivot[variables]{};
  for (int row = 0; row < rank; ++row) {
    const int col = pivot_columns[row];
    Expect(col >= 0 && col < variables && !pivot[col],
           "orthogonal echelon returns distinct valid pivots");
    pivot[col] = true;
    for (int other = 0; other < rank; ++other) {
      const Scalar expected = other == row ? Scalar{1} : Scalar{0};
      Expect(std::abs(matrix[other * columns + col] - expected) <
                 comparison_tolerance,
             "orthogonal echelon normalizes its pivot block");
    }
  }
  const Scalar solution[variables]{Scalar{1}, Scalar{2}, Scalar{2}};
  for (int row = 0; row < rank; ++row) {
    Scalar residual = -matrix[row * columns + variables];
    for (int col = 0; col < variables; ++col)
      residual += matrix[row * columns + col] * solution[col];
    Expect(std::abs(residual) < comparison_tolerance,
           "orthogonal echelon preserves the solution relation");
  }
  for (int row = rank; row < rows; ++row) {
    Scalar residual = Scalar{0};
    for (int col = 0; col < columns; ++col)
      residual = std::max(residual, std::abs(matrix[row * columns + col]));
    Expect(residual < comparison_tolerance,
           "orthogonal echelon removes consistent dependent rows");
  }
}

void IllConditionedPositiveDefiniteMultiRhsCase() {
#ifdef CLQR_USE_FLOAT
  constexpr Scalar small_eigenvalue = 5e-4f;
  constexpr Scalar uniform_scale = 1e-3f;
  constexpr Scalar solve_tolerance = 1e-5f;
  constexpr Scalar comparison_tolerance = 2e-3f;
#else
  constexpr Scalar small_eigenvalue = 5e-8;
  constexpr Scalar uniform_scale = 1e-12;
  constexpr Scalar solve_tolerance = 1e-9;
  constexpr Scalar comparison_tolerance = 2e-8;
#endif
  constexpr int dimension = 3;
  constexpr int right_hand_side_count = 3;
  // The leading block has eigenvalues 1 and small_eigenvalue, with a
  // nontrivial rotation so that the test exercises both triangular solves.
  const Scalar matrix[dimension * dimension]{
      Scalar{0.64} + Scalar{0.36} * small_eigenvalue,
      Scalar{0.48} * (Scalar{1} - small_eigenvalue),
      Scalar{0},
      Scalar{0.48} * (Scalar{1} - small_eigenvalue),
      Scalar{0.36} + Scalar{0.64} * small_eigenvalue,
      Scalar{0},
      Scalar{0},
      Scalar{0},
      Scalar{0.2}};
  const Scalar expected[dimension * right_hand_side_count]{
      Scalar{0.25}, Scalar{-0.4}, Scalar{0.8},  Scalar{-0.7}, Scalar{0.3},
      Scalar{0.1},  Scalar{0.5},  Scalar{-0.2}, Scalar{-0.6}};
  const auto solve_and_check = [&](Scalar scale, const std::string &name) {
    Scalar scaled_matrix[dimension * dimension]{};
    Scalar right_hand_sides[dimension * right_hand_side_count]{};
    for (int row = 0; row < dimension; ++row) {
      for (int col = 0; col < dimension; ++col)
        scaled_matrix[row * dimension + col] =
            scale * matrix[row * dimension + col];
      for (int rhs = 0; rhs < right_hand_side_count; ++rhs) {
        for (int col = 0; col < dimension; ++col) {
          right_hand_sides[row * right_hand_side_count + rhs] +=
              scaled_matrix[row * dimension + col] *
              expected[col * right_hand_side_count + rhs];
        }
      }
    }
    Scalar cholesky[dimension * dimension]{};
    int positive_definite = 0;
    threadIdx.x = 0;
    blockDim.x = 1;
    Expect(FactorPositiveDefiniteBlock(scaled_matrix, dimension, dimension,
                                       solve_tolerance, cholesky,
                                       &positive_definite),
           name + " ill-conditioned SPD factorization");
    SolvePositiveDefiniteMultipleRhsBlock(cholesky, dimension, right_hand_sides,
                                          right_hand_side_count,
                                          right_hand_side_count);
    for (int entry = 0; entry < dimension * right_hand_side_count; ++entry) {
      Expect(std::abs(right_hand_sides[entry] - expected[entry]) <
                 comparison_tolerance,
             name + " ill-conditioned SPD multi-RHS solution entry " +
                 std::to_string(entry));
    }
  };
  solve_and_check(Scalar{1}, "unit-scale");
  solve_and_check(uniform_scale, "uniformly-scaled");
}

void InvalidValueElementCopyCase() {
  Scalar input_j = Scalar{17};
  Scalar output_j = Scalar{23};
  ValueElement input{};
  input.left_dim = -1;
  input.right_dim = 0;
  input.J = &input_j;
  ValueElement output{};
  output.J = &output_j;

  CopyValueElementBlock(input, &output);

  Expect(output.left_dim == -1 && output.right_dim == 0,
         "copy preserves an invalid value-element sentinel");
  Expect(output_j == Scalar{23},
         "copy does not access storage for an invalid value element");
}

void FreeFixedFreeValueCompositionCase() {
  Scalar first_j[]{2, Scalar{0.1}, Scalar{0.1}, 3};
  Scalar first_eta[]{Scalar{0.4}, Scalar{-0.2}};
  Scalar second_c[]{4, Scalar{0.2}, Scalar{-0.1},
                    Scalar{0.2}, 5, Scalar{0.3},
                    Scalar{-0.1}, Scalar{0.3}, 6};
  Scalar second_b[]{Scalar{0.7}, Scalar{-0.5}, Scalar{0.25}};
  ValueElement first{};
  first.left_dim = 2;
  first.right_dim = 0;
  first.J = first_j;
  first.eta = first_eta;
  ValueElement second{};
  second.left_dim = 0;
  second.right_dim = 3;
  second.C = second_c;
  second.b = second_b;

  Scalar output_a[6];
  Scalar output_j[4]{};
  Scalar output_eta[2]{};
  Scalar output_c[9]{};
  Scalar output_b[3]{};
  std::fill(std::begin(output_a), std::end(output_a), Scalar{17});
  ValueElement output{};
  output.A = output_a;
  output.J = output_j;
  output.eta = output_eta;
  output.C = output_c;
  output.b = output_b;
  DeviceStatus status{kDeviceOk, -1, 0};
  Scalar augmented[1]{};
  Scalar factors[1]{};
  int best_row = -1;
  threadIdx.x = 0;
  blockDim.x = 1;

  ComposeValueElementsBlock(first, second, kTolerance, &output, &status, 0,
                            augmented, factors, &best_row);

  Expect(status.code == kDeviceOk,
         "free-fixed-free value composition status");
  Expect(output.left_dim == 2 && output.right_dim == 3,
         "free-fixed-free value composition dimensions");
  for (Scalar value : output_a)
    Expect(value == Scalar{0},
           "free-fixed-free value composition publishes a zero cross map");
  for (std::size_t i = 0; i < std::size(first_j); ++i)
    Expect(output_j[i] == first_j[i],
           "free-fixed-free value composition preserves the left Hessian");
  for (std::size_t i = 0; i < std::size(first_eta); ++i)
    Expect(output_eta[i] == first_eta[i],
           "free-fixed-free value composition preserves the left gradient");
  for (std::size_t i = 0; i < std::size(second_c); ++i)
    Expect(output_c[i] == second_c[i],
           "free-fixed-free value composition preserves the right curvature");
  for (std::size_t i = 0; i < std::size(second_b); ++i)
    Expect(output_b[i] == second_b[i],
           "free-fixed-free value composition preserves the right offset");
}

void DualRelationLeafScratchSizeCase() {
  ScratchSize without_constraint_scales;
  without_constraint_scales.Add<Scalar>(4);
  without_constraint_scales.Add<Scalar>(1);
  without_constraint_scales.Add<int>(1);
  without_constraint_scales.Add<int>(1);
  Expect(DualRelationLeafScratchBytes(4, 1, 1) ==
             without_constraint_scales.bytes + sizeof(Scalar),
         "dual-relation leaf scratch includes state-constraint scales");

  ScratchSize more_constraints_than_rows;
  more_constraints_than_rows.Add<Scalar>(6);
  more_constraints_than_rows.Add<Scalar>(1);
  more_constraints_than_rows.Add<Scalar>(3);
  more_constraints_than_rows.Add<int>(1);
  more_constraints_than_rows.Add<int>(3);
  Expect(DualRelationLeafScratchBytes(6, 1, 3) ==
             more_constraints_than_rows.bytes,
         "dual-relation leaf scratch sizes the QR permutation at runtime");
}

Problem PathologicalScratchProblem() {
  constexpr std::size_t n = 45;
  Problem problem;
  problem.initial_state = Vector(n);
  problem.stages.resize(1);
  Stage &stage = problem.stages[0];
  stage.A = Matrix(0, n);
  stage.B = Matrix(0, 0);
  stage.c = Vector(0);
  stage.Q = Matrix(n, n);
  for (std::size_t row = 0; row < n; ++row)
    stage.Q(row, row) = Scalar{1};
  stage.R = Matrix(0, 0);
  stage.M = Matrix(n, 0);
  stage.q = Vector(n);
  stage.r = Vector(0);
  stage.C = Matrix(0, n);
  stage.D = Matrix(0, 0);
  stage.d = Vector(0);
  stage.E = Matrix(0, n);
  stage.e = Vector(0);
  problem.terminal_Q = Matrix(0, 0);
  problem.terminal_q = Vector(0);
  problem.terminal_E = Matrix(0, 0);
  problem.terminal_e = Vector(0);
  return problem;
}

void ScratchPlannerTopologyCase() {
  constexpr std::size_t kUsableP100SharedBytes = 48 * 1024 - 256;
  const ScratchRequirements pathological =
      PlanScratch(PathologicalScratchProblem());
  Expect(pathological.Maximum() <= kUsableP100SharedBytes,
         "topology-aware scratch accepts a 45-to-0 state transition");
  Expect(DenseEliminationScratchBytes(4 * 45, 3 * 45 + 1,
                                      "legacy synthetic workspace") >
             kUsableP100SharedBytes,
         "the former synthetic cross-maximum would reject the 45-to-0 case");

  constexpr std::size_t n = 8;
  const Problem uniform_problem = clqr::benchmark::StateOnlyProblem(8, n, 4, 2);
  const ScratchRequirements uniform = PlanScratch(uniform_problem);
  const ScanShape uniform_relation = MakeScanShape(n, n);
  const ScanShape terminal_relation = MakeScanShape(n, 0);
  ScratchSize value_leaf;
  value_leaf.Add<Scalar>(4 * 4);
  value_leaf.Add<Scalar>(4 * (2 * n + 1));
  ScratchSize feedback;
  feedback.Add<Scalar>(4 * (4 + n + 1));
  feedback.Add<Scalar>(4 * 4);
  ScratchSize dual_parameter;
  dual_parameter.Add<Scalar>(12 * 9);
  dual_parameter.Add<Scalar>(0);
  dual_parameter.Add<Scalar>(12);
  dual_parameter.Add<Scalar>(n * n);
  dual_parameter.Add<Scalar>(n);
  dual_parameter.Add<Scalar>(n);
  dual_parameter.Add<int>(n);
  Expect(uniform.primal_relation ==
             DenseEliminationScratchBytes(4 * n, 3 * n + 1,
                                          "uniform primal workspace"),
         "uniform n=8 primal-relation launch scratch is unchanged");
  Expect(uniform.primal_leaf == DenseEliminationScratchBytes(
                                    10, 21, "uniform primal-leaf workspace") &&
             uniform.primal_relation_final ==
                 RelationFinalizeScratchBytes(
                     uniform_relation, &uniform_relation, terminal_relation) &&
             uniform.state_parameter == n * sizeof(int) &&
             uniform.stage_reduction ==
                 DenseEliminationScratchBytes(
                     8, 13, "uniform stage-reduction workspace"),
         "uniform n=8 feasibility launch scratch is unchanged");
  Expect(uniform.value_compose ==
             GeneralSolveScratchBytes(n, 3 * n + 1, "uniform value workspace"),
         "uniform n=8 value-composition launch scratch uses LU workspace");
  Expect(uniform.value_leaf == value_leaf.bytes &&
             uniform.value_finalize ==
                 ValueFinalizeScratchBytes(uniform_relation, &uniform_relation,
                                           terminal_relation) &&
             uniform.feedback == feedback.bytes &&
             uniform.affine_finalize ==
                 AffineFinalizeScratchBytes(uniform_relation, &uniform_relation,
                                            uniform_relation),
         "uniform n=8 Riccati/reconstruction launch scratch matches active "
         "dimensions");
  Expect(uniform.dual_relation ==
             DenseEliminationScratchBytes(4 * n, 3 * n + 1,
                                          "uniform dual workspace"),
         "uniform n=8 dual-relation launch scratch is unchanged");
  Expect(uniform.dual_parameter == dual_parameter.bytes &&
             uniform.dual_relation_leaf ==
                 DualRelationLeafScratchBytes(8 * 19, 8, 2) &&
             uniform.dual_root == DualRootScratchBytes(terminal_relation) &&
             uniform.dual_expand ==
                 DualExpandScratchBytes(uniform_relation, uniform_relation),
         "uniform n=8 dual-expansion launch scratch is unchanged");

  std::vector<int> cached_key;
  ScratchRequirements cached_scratch;
  int plan_builds =
      RefreshScratchPlan(uniform_problem, &cached_key, &cached_scratch);
  for (int repeat = 0; repeat < 100; ++repeat) {
    plan_builds +=
        RefreshScratchPlan(uniform_problem, &cached_key, &cached_scratch);
  }
  Expect(plan_builds == 1 && cached_scratch.Maximum() == uniform.Maximum(),
         "same-shape workspace reuse builds the scratch plan only once");
  const Problem changed_problem =
      clqr::benchmark::StateOnlyProblem(8, n + 1, 4, 2);
  plan_builds +=
      RefreshScratchPlan(changed_problem, &cached_key, &cached_scratch);
  Expect(plan_builds == 2 &&
             cached_scratch.Maximum() == PlanScratch(changed_problem).Maximum(),
         "a dimension change rebuilds the cached scratch plan");
}

Scalar Value(int seed, std::size_t row, std::size_t col = 0) {
  const Scalar x = static_cast<Scalar>(seed * 83 + row * 29 + col * 43);
  return std::sin(0.019 * x) + 0.25 * std::cos(0.037 * x);
}

Matrix GeneratedMatrix(std::size_t rows, std::size_t cols, int seed,
                       Scalar scale) {
  Matrix out(rows, cols);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col)
      out(row, col) = scale * Value(seed, row, col);
  return out;
}

Vector GeneratedVector(std::size_t size, int seed, Scalar scale) {
  Vector out(size);
  for (std::size_t row = 0; row < size; ++row)
    out[row] = scale * Value(seed, row);
  return out;
}

Matrix PositiveDefinite(std::size_t size, int seed, Scalar diagonal) {
  Matrix g = GeneratedMatrix(size, size, seed, 0.18);
  Matrix out = clqr::Transpose(g) * g;
  for (std::size_t row = 0; row < size; ++row)
    out(row, row) += diagonal;
  return out;
}

Scalar RowDot(const Matrix &matrix, std::size_t row, const Vector &vector) {
  Scalar value = 0.0;
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
    Stage &stage = problem.stages[i];
    stage.A = GeneratedMatrix(n, n, 300 + static_cast<int>(i), 0.12);
    for (std::size_t row = 0; row < n; ++row)
      stage.A(row, row) += 0.8;
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
    Stage &stage = problem.stages[i];
    stage.A = GeneratedMatrix(n, n, seed + 300 + static_cast<int>(i), 0.08);
    for (std::size_t row = 0; row < n; ++row)
      stage.A(row, row) += 0.9;
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
    Stage &stage = problem.stages[i];
    const Vector nominal_x =
        GeneratedVector(n, seed + 100 + static_cast<int>(i), 0.5);
    stage.E = GeneratedMatrix(1, n, seed + 1200 + static_cast<int>(i), 0.3);
    stage.e = Vector{-RowDot(stage.E, 0, nominal_x)};
  }
  return problem;
}

Problem ExactDualRelationScratchProblem() {
  constexpr int seed = 1950;
  constexpr std::size_t horizon = 4;
  Problem problem = UniformProblem(seed, horizon, 1, 0);
  for (std::size_t i = 1; i < horizon; ++i) {
    problem.stages[i].E = Matrix(1, 1, {Scalar{1}});
    problem.stages[i].e =
        Vector{-GeneratedVector(1, seed + 100 + static_cast<int>(i), 0.5)[0]};
  }
  problem.terminal_E = Matrix(1, 1, {Scalar{1}});
  problem.terminal_e = Vector{
      -GeneratedVector(1, seed + 100 + static_cast<int>(horizon), 0.5)[0]};
  return problem;
}

Problem HeterogeneousDimensionProblem() {
  constexpr int seed = 1975;
  const std::vector<std::size_t> dimensions{24, 1, 23, 0};
  const std::vector<std::size_t> controls{2, 1, 0};
  const std::size_t horizon = controls.size();
  Problem problem;
  std::vector<Vector> x(horizon + 1);
  std::vector<Vector> u(horizon);
  for (std::size_t i = 0; i <= horizon; ++i)
    x[i] = GeneratedVector(dimensions[i], seed + 100 + static_cast<int>(i),
                           Scalar{0.25});
  for (std::size_t i = 0; i < horizon; ++i)
    u[i] = GeneratedVector(controls[i], seed + 200 + static_cast<int>(i),
                           Scalar{0.2});
  problem.initial_state = x.front();
  problem.stages.resize(horizon);
  for (std::size_t i = 0; i < horizon; ++i) {
    Stage &stage = problem.stages[i];
    const std::size_t n = dimensions[i];
    const std::size_t next = dimensions[i + 1];
    const std::size_t m = controls[i];
    stage.A = GeneratedMatrix(next, n, seed + 300 + static_cast<int>(i),
                              Scalar{0.05});
    stage.B = GeneratedMatrix(next, m, seed + 400 + static_cast<int>(i),
                              Scalar{0.08});
    stage.c = x[i + 1] - stage.A * x[i] - stage.B * u[i];
    stage.Q = PositiveDefinite(n, seed + 500 + static_cast<int>(i), Scalar{1});
    stage.R =
        PositiveDefinite(m, seed + 600 + static_cast<int>(i), Scalar{1.5});
    stage.M =
        GeneratedMatrix(n, m, seed + 700 + static_cast<int>(i), Scalar{0.01});
    stage.q =
        GeneratedVector(n, seed + 800 + static_cast<int>(i), Scalar{0.05});
    stage.r =
        GeneratedVector(m, seed + 900 + static_cast<int>(i), Scalar{0.05});
    stage.C = Matrix(0, n);
    stage.D = Matrix(0, m);
    stage.d = Vector(0);
    stage.E = Matrix(0, n);
    stage.e = Vector(0);
  }
  problem.terminal_Q =
      PositiveDefinite(dimensions.back(), seed + 1000, Scalar{1.5});
  problem.terminal_q =
      GeneratedVector(dimensions.back(), seed + 1100, Scalar{0.05});
  problem.terminal_E = Matrix(0, dimensions.back());
  problem.terminal_e = Vector(0);
  return problem;
}

Problem MaximumConstraintProblem() {
  constexpr int seed = 1700;
  constexpr std::size_t n = kTestStateCapacity;
  constexpr std::size_t m = kTestControlCapacity;
  constexpr std::size_t constraints =
      std::min(kTestMixedCapacity, kTestStateConstraintCapacity);
  Problem problem = UniformProblem(seed, 1, n, m);
  Stage &stage = problem.stages[0];
  const Vector nominal_u = GeneratedVector(m, seed + 200, 0.4);
  stage.C = GeneratedMatrix(constraints, n, seed + 1200, 0.1);
  stage.D = GeneratedMatrix(constraints, m, seed + 1210, 0.1);
  stage.d = Vector(constraints);
  for (std::size_t row = 0; row < constraints; ++row) {
    stage.d[row] = -(RowDot(stage.C, row, problem.initial_state) +
                     RowDot(stage.D, row, nominal_u));
  }
  stage.E = GeneratedMatrix(constraints, n, seed + 1220, 0.1);
  stage.e = Vector(constraints);
  for (std::size_t row = 0; row < constraints; ++row)
    stage.e[row] = -RowDot(stage.E, row, problem.initial_state);
  return problem;
}

Problem MoreMixedRowsThanControlsProblem() {
  constexpr int seed = 2000;
  constexpr std::size_t horizon = 4;
  constexpr std::size_t n = 4;
  constexpr std::size_t m = 1;
  constexpr std::size_t rows = std::min<std::size_t>(3, kTestMixedCapacity);
  Problem problem = UniformProblem(seed, horizon, n, m);
  for (std::size_t i = 0; i < horizon; ++i) {
    Stage &stage = problem.stages[i];
    const Vector nominal_x =
        GeneratedVector(n, seed + 100 + static_cast<int>(i), 0.5);
    const Vector nominal_u =
        GeneratedVector(m, seed + 200 + static_cast<int>(i), 0.4);
    stage.C = GeneratedMatrix(rows, n, seed + 1200 + static_cast<int>(i), 0.3);
    stage.D = GeneratedMatrix(rows, m, seed + 1300 + static_cast<int>(i), 0.3);
    stage.d = Vector(rows);
    for (std::size_t row = 0; row < rows; ++row) {
      const Scalar scale =
          rows >= 3 && row == 0 ? 1e-7 : (row == 2 ? 1e7 : 1.0);
      for (std::size_t col = 0; col < n; ++col)
        stage.C(row, col) *= scale;
      for (std::size_t col = 0; col < m; ++col)
        stage.D(row, col) *= scale;
      stage.d[row] =
          -(RowDot(stage.C, row, nominal_x) + RowDot(stage.D, row, nominal_u));
    }
  }
  return problem;
}

Problem LongHorizonStateConstraintProblem() {
  return clqr::benchmark::StateOnlyProblem(
      16384, std::min<std::size_t>(8, kTestStateCapacity),
      std::min<std::size_t>(4, kTestControlCapacity),
      std::min<std::size_t>(2, kTestStateConstraintCapacity));
}

void PackMatrix(const Matrix &source, Scalar **cursor, const Scalar **target) {
  *target = *cursor;
  for (std::size_t row = 0; row < source.rows(); ++row)
    for (std::size_t col = 0; col < source.cols(); ++col)
      (*cursor)[row * source.cols() + col] = source(row, col);
  *cursor += source.rows() * source.cols();
}

void PackVector(const Vector &source, Scalar **cursor, const Scalar **target) {
  *target = *cursor;
  for (std::size_t row = 0; row < source.size(); ++row)
    (*cursor)[row] = source[row];
  *cursor += source.size();
}

std::size_t PackedEntries(const Stage &source) {
  const auto matrix_entries = [](const Matrix &matrix) {
    return matrix.rows() * matrix.cols();
  };
  return matrix_entries(source.A) + matrix_entries(source.B) + source.c.size() +
         matrix_entries(source.Q) + matrix_entries(source.R) +
         matrix_entries(source.M) + source.q.size() + source.r.size() +
         matrix_entries(source.C) + matrix_entries(source.D) + source.d.size() +
         matrix_entries(source.E) + source.e.size();
}

std::size_t PackedEntries(const Problem &problem) {
  std::size_t entries = problem.terminal_Q.rows() * problem.terminal_Q.cols() +
                        problem.terminal_q.size() +
                        problem.terminal_E.rows() * problem.terminal_E.cols() +
                        problem.terminal_e.size();
  for (const Stage &stage : problem.stages)
    entries += PackedEntries(stage);
  return entries;
}

PackedStage Pack(const Stage &source, Scalar **cursor) {
  PackedStage out{};
  out.n = static_cast<int>(source.A.cols());
  out.next_n = static_cast<int>(source.A.rows());
  out.m = static_cast<int>(source.B.cols());
  out.mixed = static_cast<int>(source.C.rows());
  out.state = static_cast<int>(source.E.rows());
  PackMatrix(source.A, cursor, &out.A);
  PackMatrix(source.B, cursor, &out.B);
  PackVector(source.c, cursor, &out.c);
  PackMatrix(source.Q, cursor, &out.Q);
  PackMatrix(source.R, cursor, &out.R);
  PackMatrix(source.M, cursor, &out.M);
  PackVector(source.q, cursor, &out.q);
  PackVector(source.r, cursor, &out.r);
  PackMatrix(source.C, cursor, &out.C);
  PackMatrix(source.D, cursor, &out.D);
  PackVector(source.d, cursor, &out.d);
  PackMatrix(source.E, cursor, &out.E);
  PackVector(source.e, cursor, &out.e);
  return out;
}

PackedTerminal Pack(const Problem &problem, Scalar **cursor) {
  PackedTerminal out{};
  out.n = static_cast<int>(problem.terminal_Q.rows());
  out.state = static_cast<int>(problem.terminal_E.rows());
  PackMatrix(problem.terminal_Q, cursor, &out.Q);
  PackVector(problem.terminal_q, cursor, &out.q);
  PackMatrix(problem.terminal_E, cursor, &out.E);
  PackVector(problem.terminal_e, cursor, &out.e);
  return out;
}

template <typename Function> void Launch(int blocks, Function function) {
  threadIdx.x = 0;
  blockDim.x = 1;
  gridDim.x = blocks;
  for (int block = 0; block < blocks; ++block) {
    blockIdx.x = block;
    function();
  }
}

void FiniteInputValidationCase() {
  Scalar problem_data[]{Scalar{1},
                        std::numeric_limits<Scalar>::quiet_NaN()};
  Scalar initial_state[]{Scalar{2}};
  DeviceStatus status{};
  Launch(2, [&] {
    CheckFiniteInputsKernel(problem_data, std::size(problem_data),
                            initial_state, std::size(initial_state), &status);
  });
  Expect(status.code == kDeviceInvalidInput && status.stage == -1 &&
             status.detail == 21,
         "device input validation rejects a non-finite problem coefficient");

  problem_data[1] = Scalar{3};
  initial_state[0] = std::numeric_limits<Scalar>::infinity();
  status = DeviceStatus{};
  Launch(2, [&] {
    CheckFiniteInputsKernel(problem_data, std::size(problem_data),
                            initial_state, std::size(initial_state), &status);
  });
  Expect(status.code == kDeviceInvalidInput,
         "device input validation rejects a non-finite initial state");

  initial_state[0] = Scalar{4};
  status = DeviceStatus{};
  Launch(2, [&] {
    CheckFiniteInputsKernel(problem_data, std::size(problem_data),
                            initial_state, std::size(initial_state), &status);
  });
  Expect(status.code == kDeviceOk,
         "device input validation accepts finite values");
}

void DeviceObjectiveCase() {
  Scalar Q[]{Scalar{2}, Scalar{1}, Scalar{1}, Scalar{4}};
  Scalar R[]{Scalar{3}};
  Scalar M[]{Scalar{1}, Scalar{-2}};
  Scalar q[]{Scalar{0.5}, Scalar{-1}};
  Scalar r[]{Scalar{0.25}};
  PackedStage stage{};
  stage.n = 2;
  stage.m = 1;
  stage.Q = Q;
  stage.R = R;
  stage.M = M;
  stage.q = q;
  stage.r = r;
  Scalar terminal_Q[]{Scalar{5}};
  Scalar terminal_q[]{Scalar{2}};
  PackedTerminal terminal{};
  terminal.n = 1;
  terminal.Q = terminal_Q;
  terminal.q = terminal_q;
  Scalar states[]{Scalar{1}, Scalar{2}, Scalar{3}};
  Scalar controls[]{Scalar{4}};
  int state_offsets[]{0, 2, 3};
  int control_offsets[]{0, 1};
  Scalar objective_tree[3]{};
  DeviceStatus status{};

  Launch(2, [&] {
    BuildObjectiveTermsKernel(&stage, 1, &terminal, states, controls,
                              state_offsets, control_offsets, objective_tree,
                              &status);
  });
  Launch(1, [&] {
    ReduceObjectiveTreeLevelKernel(objective_tree, 0, 2, 2, &status);
  });
  Expect(status.code == kDeviceOk &&
             std::abs(objective_tree[2] - Scalar{51}) < Scalar{1e-5},
         "device objective reduction matches the dense quadratic objective");
}

void NonPositiveDefiniteReducedControlCostCase() {
  ReducedStage stage{};
  Scalar stage_a[1]{};
  Scalar stage_b[1]{};
  Scalar stage_c[1]{};
  Scalar stage_q_matrix[1]{Scalar{1}};
  Scalar stage_r_matrix[1]{Scalar{-1}};
  Scalar stage_m[1]{};
  Scalar stage_q[1]{};
  Scalar stage_r[1]{};
  stage.A = stage_a;
  stage.B = stage_b;
  stage.c = stage_c;
  stage.Q = stage_q_matrix;
  stage.R = stage_r_matrix;
  stage.M = stage_m;
  stage.q = stage_q;
  stage.r = stage_r;
  stage.n = 1;
  stage.next_n = 1;
  stage.m = 1;
  ReducedTerminal terminal{};
  Scalar terminal_q_matrix[1]{Scalar{1}};
  Scalar terminal_q[1]{};
  terminal.Q = terminal_q_matrix;
  terminal.q = terminal_q;
  terminal.n = 1;
  std::vector<ValueElement> elements(2);
  std::vector<Scalar> storage(2 * kTestValueEntries);
  for (int node = 0; node < 2; ++node) {
    BindValueElementScratch(&elements[node],
                            storage.data() + static_cast<std::size_t>(node) *
                                                 kTestValueEntries,
                            kTestStateCapacity, kTestStateCapacity);
  }
  DeviceStatus status{kDeviceOk, -1, 0};
  Launch(2, [&] {
    BuildValueElementsKernel(&stage, &terminal, 1, kTolerance, elements.data(),
                             &status);
  });
  Expect(status.code == kDeviceNumericalFailure,
         "non-positive-definite reduced control cost status");
  Expect(status.stage == 0 && status.detail == 19,
         "non-positive-definite reduced control cost diagnostic");
}

Scalar MaxResidual(const Problem &problem, const std::vector<Scalar> &states,
                   const std::vector<Scalar> &controls,
                   const std::vector<Scalar> &initial_multiplier,
                   const std::vector<Scalar> &dynamics,
                   const std::vector<Scalar> &mixed,
                   const std::vector<Scalar> &state_multipliers,
                   const std::vector<Scalar> &terminal_multiplier,
                   std::string *worst = nullptr) {
  Scalar residual = 0.0;
  const auto update = [&](Scalar candidate, std::string equation) {
    if (!std::isfinite(candidate)) {
      residual = std::numeric_limits<Scalar>::infinity();
      if (worst != nullptr)
        *worst = std::move(equation);
      return;
    }
    candidate = std::abs(candidate);
    if (candidate > residual) {
      residual = candidate;
      if (worst != nullptr)
        *worst = std::move(equation);
    }
  };
  const int horizon = static_cast<int>(problem.stages.size());
  for (int i = 0; i < horizon; ++i) {
    const Stage &s = problem.stages[i];
    const Scalar *x = states.data() + i * kTestStateCapacity;
    const Scalar *xp = states.data() + (i + 1) * kTestStateCapacity;
    const Scalar *u = controls.data() + i * kTestControlCapacity;
    const Scalar *right = dynamics.data() + i * kTestStateCapacity;
    const Scalar *left = i == 0
                             ? initial_multiplier.data()
                             : dynamics.data() + (i - 1) * kTestStateCapacity;
    for (std::size_t row = 0; row < s.A.rows(); ++row) {
      Scalar value = xp[row] - s.c[row];
      for (std::size_t col = 0; col < s.A.cols(); ++col)
        value -= s.A(row, col) * x[col];
      for (std::size_t col = 0; col < s.B.cols(); ++col)
        value -= s.B(row, col) * u[col];
      update(value, "dynamics at stage " + std::to_string(i));
    }
    for (std::size_t row = 0; row < s.C.rows(); ++row) {
      Scalar value = s.d[row];
      Scalar scale = std::max(Scalar{1}, std::abs(s.d[row]));
      for (std::size_t col = 0; col < s.C.cols(); ++col) {
        value += s.C(row, col) * x[col];
        scale = std::max(scale, std::abs(s.C(row, col)));
      }
      for (std::size_t col = 0; col < s.D.cols(); ++col) {
        value += s.D(row, col) * u[col];
        scale = std::max(scale, std::abs(s.D(row, col)));
      }
      update(value / scale, "mixed feasibility at stage " + std::to_string(i));
    }
    for (std::size_t row = 0; row < s.E.rows(); ++row) {
      Scalar value = s.e[row];
      Scalar scale = std::max(Scalar{1}, std::abs(s.e[row]));
      for (std::size_t col = 0; col < s.E.cols(); ++col) {
        value += s.E(row, col) * x[col];
        scale = std::max(scale, std::abs(s.E(row, col)));
      }
      update(value / scale, "state feasibility at stage " + std::to_string(i));
    }
    for (std::size_t row = 0; row < s.A.cols(); ++row) {
      Scalar value = s.q[row] + left[row];
      for (std::size_t col = 0; col < s.Q.cols(); ++col)
        value += s.Q(row, col) * x[col];
      for (std::size_t col = 0; col < s.M.cols(); ++col)
        value += s.M(row, col) * u[col];
      for (std::size_t next = 0; next < s.A.rows(); ++next)
        value -= s.A(next, row) * right[next];
      for (std::size_t constraint = 0; constraint < s.C.rows(); ++constraint)
        value +=
            s.C(constraint, row) * mixed[i * kTestMixedCapacity + constraint];
      for (std::size_t constraint = 0; constraint < s.E.rows(); ++constraint)
        value +=
            s.E(constraint, row) *
            state_multipliers[i * kTestStateConstraintCapacity + constraint];
      update(value, "state stationarity at stage " + std::to_string(i));
    }
    for (std::size_t row = 0; row < s.B.cols(); ++row) {
      Scalar value = s.r[row];
      for (std::size_t col = 0; col < s.M.rows(); ++col)
        value += s.M(col, row) * x[col];
      for (std::size_t col = 0; col < s.R.cols(); ++col)
        value += s.R(row, col) * u[col];
      for (std::size_t next = 0; next < s.B.rows(); ++next)
        value -= s.B(next, row) * right[next];
      for (std::size_t constraint = 0; constraint < s.D.rows(); ++constraint)
        value +=
            s.D(constraint, row) * mixed[i * kTestMixedCapacity + constraint];
      update(value, "control stationarity at stage " + std::to_string(i));
    }
  }
  const Scalar *terminal = states.data() + horizon * kTestStateCapacity;
  const Scalar *left =
      horizon == 0 ? initial_multiplier.data()
                   : dynamics.data() + (horizon - 1) * kTestStateCapacity;
  for (std::size_t row = 0; row < problem.terminal_Q.rows(); ++row) {
    Scalar value = problem.terminal_q[row] + left[row];
    for (std::size_t col = 0; col < problem.terminal_Q.cols(); ++col)
      value += problem.terminal_Q(row, col) * terminal[col];
    for (std::size_t constraint = 0; constraint < problem.terminal_E.rows();
         ++constraint)
      value +=
          problem.terminal_E(constraint, row) * terminal_multiplier[constraint];
    update(value, "terminal stationarity");
  }
  return residual;
}

void RunEmulation(const Problem &problem, const std::string &name,
                  bool expect_reduced_state, bool expect_reduced_control,
                  bool compare_cpu = true,
                  Scalar kkt_tolerance_scale = Scalar{1}) {
  const AllowedDeviceFailure *allowed_failure = AllowedFailureForCase(name);
  const int horizon = static_cast<int>(problem.stages.size());
  const int nodes = horizon + 1;
  std::vector<Scalar> packed_data(PackedEntries(problem));
  Scalar *packed_cursor = packed_data.data();
  std::vector<PackedStage> stages;
  for (const Stage &stage : problem.stages)
    stages.push_back(Pack(stage, &packed_cursor));
  const PackedTerminal terminal = Pack(problem, &packed_cursor);
  Expect(packed_cursor == packed_data.data() + packed_data.size(),
         "compact problem packing uses the exact allocation");
  std::vector<Scalar> initial(kTestStateCapacity);
  for (std::size_t row = 0; row < problem.initial_state.size(); ++row)
    initial[row] = problem.initial_state[row];
  DeviceStatus status{};
  Scalar feasibility_consistency_tolerance =
      std::max(kTolerance, kMinimumFeasibilityConsistencyTolerance);

  std::vector<int> node_level_offsets{0};
  std::vector<int> node_level_counts{nodes};
  int node_tree_size = nodes;
  while (node_level_counts.back() > 1) {
    node_level_offsets.push_back(node_tree_size);
    node_level_counts.push_back((node_level_counts.back() + 1) / 2);
    node_tree_size += node_level_counts.back();
  }
  const int feasibility_scan_levels =
      static_cast<int>(node_level_counts.size()) - 1;
  feasibility_consistency_tolerance = std::max(
      kTolerance, kMinimumFeasibilityConsistencyTolerance *
                      static_cast<Scalar>(feasibility_scan_levels + 2));
  std::vector<Relation> relation_a(nodes),
      relation_b(std::max(node_tree_size - nodes, 1));
  std::vector<Scalar> relation_a_storage(static_cast<std::size_t>(nodes) *
                                         kTestRelationEntries);
  std::vector<Scalar> relation_b_storage(relation_b.size() *
                                         kTestRelationEntries);
  for (int node = 0; node < nodes; ++node) {
    BindRelationScratch(&relation_a[node],
                        relation_a_storage.data() +
                            static_cast<std::size_t>(node) *
                                kTestRelationEntries,
                        kTestStateCapacity, kTestStateCapacity);
  }
  for (std::size_t node = 0; node < relation_b.size(); ++node) {
    BindRelationScratch(&relation_b[node],
                        relation_b_storage.data() + node * kTestRelationEntries,
                        kTestStateCapacity, kTestStateCapacity);
  }
  Launch(nodes, [&] {
    BuildPrimalLeavesKernel(stages.data(), horizon, &terminal, kTolerance,
                            feasibility_consistency_tolerance,
                            relation_a.data(), &status);
  });
  if (nodes > 1) {
    const int first_parent_count = node_level_counts[1];
    Launch(first_parent_count, [&] {
      ReduceRelationLeavesKernel(relation_a.data(), nodes, first_parent_count,
                                 kTolerance, feasibility_consistency_tolerance,
                                 relation_b.data(), &status);
    });
    for (std::size_t level = 1; level + 1 < node_level_counts.size(); ++level) {
      Launch(node_level_counts[level + 1], [&] {
        ReduceRelationTreeLevelKernel(
            relation_b.data(), node_level_offsets[level] - nodes,
            node_level_offsets[level + 1] - nodes, node_level_counts[level],
            node_level_counts[level + 1], kTolerance,
            feasibility_consistency_tolerance, &status);
      });
    }
    Launch(1, [&] {
      InitializeRelationContextRootKernel(relation_b.data(),
                                          node_level_offsets.back() - nodes);
    });
    for (int level = static_cast<int>(node_level_counts.size()) - 2; level >= 1;
         --level) {
      Launch(node_level_counts[level + 1], [&] {
        ExpandRelationContextLevelKernel(
            relation_b.data(), node_level_offsets[level] - nodes,
            node_level_offsets[level + 1] - nodes, node_level_counts[level],
            node_level_counts[level + 1], kTolerance,
            feasibility_consistency_tolerance, &status);
      });
    }
    Launch(first_parent_count, [&] {
      FinalizeRelationSuffixFromParentsKernel(
          relation_a.data(), nodes, relation_b.data(), first_parent_count,
          kTolerance, feasibility_consistency_tolerance, &status);
    });
  }
  Relation *suffix = relation_a.data();
  std::vector<StateParam> state_params(nodes);
  std::vector<int> state_free_columns(static_cast<std::size_t>(nodes) *
                                      kTestStateCapacity);
  std::vector<Scalar> state_t(static_cast<std::size_t>(nodes) *
                              kTestStateCapacity);
  std::vector<Scalar> state_T(static_cast<std::size_t>(nodes) *
                              kTestStateCapacity * kTestStateCapacity);
  for (int node = 0; node < nodes; ++node) {
    state_params[node].free_columns =
        state_free_columns.data() +
        static_cast<std::size_t>(node) * kTestStateCapacity;
    state_params[node].T = state_T.data() + static_cast<std::size_t>(node) *
                                                kTestStateCapacity *
                                                kTestStateCapacity;
    state_params[node].t =
        state_t.data() + static_cast<std::size_t>(node) * kTestStateCapacity;
  }
  Launch(nodes, [&] {
    StateParamKernel(suffix, nodes, state_params.data(), nullptr, &status,
                     kTolerance);
  });
  if (FinishAllowedDeviceFailure(status, name, "feasibility scan",
                                 allowed_failure))
    return;

  std::vector<ControlParam> control_params(horizon);
  std::vector<ReducedStage> reduced(horizon);
  std::vector<int> control_free_columns(static_cast<std::size_t>(horizon) *
                                        kTestControlCapacity);
  std::vector<Scalar> control_Y(static_cast<std::size_t>(horizon) *
                                kTestControlCapacity * kTestStateCapacity);
  std::vector<Scalar> control_Z(static_cast<std::size_t>(horizon) *
                                kTestControlCapacity * kTestControlCapacity);
  std::vector<Scalar> control_y(static_cast<std::size_t>(horizon) *
                                kTestControlCapacity);
  std::vector<Scalar> reduced_A(static_cast<std::size_t>(horizon) *
                                kTestStateCapacity * kTestStateCapacity);
  std::vector<Scalar> reduced_B(static_cast<std::size_t>(horizon) *
                                kTestStateCapacity * kTestControlCapacity);
  std::vector<Scalar> reduced_c(static_cast<std::size_t>(horizon) *
                                kTestStateCapacity);
  std::vector<Scalar> reduced_Q(static_cast<std::size_t>(horizon) *
                                kTestStateCapacity * kTestStateCapacity);
  std::vector<Scalar> reduced_R(static_cast<std::size_t>(horizon) *
                                kTestControlCapacity * kTestControlCapacity);
  std::vector<Scalar> reduced_M(static_cast<std::size_t>(horizon) *
                                kTestStateCapacity * kTestControlCapacity);
  std::vector<Scalar> reduced_q(static_cast<std::size_t>(horizon) *
                                kTestStateCapacity);
  std::vector<Scalar> reduced_r(static_cast<std::size_t>(horizon) *
                                kTestControlCapacity);
  for (int stage = 0; stage < horizon; ++stage) {
    const std::size_t index = static_cast<std::size_t>(stage);
    control_params[stage].free_columns =
        control_free_columns.data() + index * kTestControlCapacity;
    control_params[stage].Y =
        control_Y.data() + index * kTestControlCapacity * kTestStateCapacity;
    control_params[stage].Z =
        control_Z.data() + index * kTestControlCapacity * kTestControlCapacity;
    control_params[stage].y = control_y.data() + index * kTestControlCapacity;
    reduced[stage].A =
        reduced_A.data() + index * kTestStateCapacity * kTestStateCapacity;
    reduced[stage].B =
        reduced_B.data() + index * kTestStateCapacity * kTestControlCapacity;
    reduced[stage].c = reduced_c.data() + index * kTestStateCapacity;
    reduced[stage].Q =
        reduced_Q.data() + index * kTestStateCapacity * kTestStateCapacity;
    reduced[stage].R =
        reduced_R.data() + index * kTestControlCapacity * kTestControlCapacity;
    reduced[stage].M =
        reduced_M.data() + index * kTestStateCapacity * kTestControlCapacity;
    reduced[stage].q = reduced_q.data() + index * kTestStateCapacity;
    reduced[stage].r = reduced_r.data() + index * kTestControlCapacity;
  }
  ReducedTerminal reduced_terminal{};
  std::vector<Scalar> reduced_terminal_Q(
      static_cast<std::size_t>(kTestStateCapacity) * kTestStateCapacity);
  std::vector<Scalar> reduced_terminal_q(kTestStateCapacity);
  reduced_terminal.Q = reduced_terminal_Q.data();
  reduced_terminal.q = reduced_terminal_q.data();
  std::vector<Scalar> reduced_initial(kTestStateCapacity);
  Launch(horizon, [&] {
    ReduceStagesKernel(stages.data(), suffix, state_params.data(), horizon,
                       kTolerance, feasibility_consistency_tolerance,
                       control_params.data(), reduced.data(), nullptr, &status);
  });
  Launch(1, [&] {
    ReduceTerminalKernel(&terminal, state_params.data(), horizon,
                         &reduced_terminal);
  });
  Launch(1, [&] {
    InitialReducedStateKernel(state_params.data(), initial.data(),
                              reduced_initial.data(), kTolerance, &status);
  });
  if (FinishAllowedDeviceFailure(status, name, "independent reduction",
                                 allowed_failure))
    return;
  bool reduced_a_state = false;
  bool reduced_a_control = false;
  for (const StateParam &param : state_params)
    reduced_a_state |= param.reduced_dim < param.physical_dim;
  for (const ControlParam &param : control_params)
    reduced_a_control |= param.reduced_dim < param.physical_dim;
  if (expect_reduced_state)
    Expect(reduced_a_state, name + " exercises smaller state dimensions");
  if (expect_reduced_control)
    Expect(reduced_a_control, name + " exercises smaller control dimensions");

  std::vector<ValueElement> value_a(nodes),
      value_b(std::max(node_tree_size - nodes, 1));
  std::vector<Scalar> value_a_storage(static_cast<std::size_t>(nodes) *
                                      kTestValueEntries);
  std::vector<Scalar> value_b_storage(value_b.size() * kTestValueEntries);
  std::fill(value_a_storage.begin(), value_a_storage.end(), Scalar{17});
  std::fill(value_b_storage.begin(), value_b_storage.end(), Scalar{19});
  for (int node = 0; node < nodes; ++node) {
    BindValueElementScratch(&value_a[node],
                            value_a_storage.data() +
                                static_cast<std::size_t>(node) *
                                    kTestValueEntries,
                            kTestStateCapacity, kTestStateCapacity);
  }
  for (std::size_t node = 0; node < value_b.size(); ++node) {
    BindValueElementScratch(&value_b[node],
                            value_b_storage.data() + node * kTestValueEntries,
                            kTestStateCapacity, kTestStateCapacity);
  }
  std::vector<Feedback> feedback(horizon);
  std::vector<Scalar> feedback_K(static_cast<std::size_t>(horizon) *
                                 kTestControlCapacity * kTestStateCapacity);
  std::vector<Scalar> feedback_k(static_cast<std::size_t>(horizon) *
                                 kTestControlCapacity);
  std::vector<Scalar> feedback_transition(static_cast<std::size_t>(horizon) *
                                          kTestStateCapacity *
                                          kTestStateCapacity);
  std::vector<Scalar> feedback_offset(static_cast<std::size_t>(horizon) *
                                      kTestStateCapacity);
  for (int stage = 0; stage < horizon; ++stage) {
    const std::size_t index = static_cast<std::size_t>(stage);
    feedback[stage].K =
        feedback_K.data() + index * kTestControlCapacity * kTestStateCapacity;
    feedback[stage].k = feedback_k.data() + index * kTestControlCapacity;
    feedback[stage].transition =
        feedback_transition.data() +
        index * kTestStateCapacity * kTestStateCapacity;
    feedback[stage].offset =
        feedback_offset.data() + index * kTestStateCapacity;
  }
  Launch(nodes, [&] {
    BuildValueElementsKernel(reduced.data(), &reduced_terminal, horizon,
                             kTolerance, value_a.data(), &status);
  });
  if (FinishAllowedDeviceFailure(status, name, "value base",
                                 allowed_failure))
    return;
  ValueElement *value_suffix = value_a.data();
  if (nodes > 1) {
    const int first_parent_count = node_level_counts[1];
    Launch(first_parent_count, [&] {
      ReduceValueLeavesKernel(value_a.data(), nodes, first_parent_count,
                              kTolerance, &status, value_b.data());
    });
    for (std::size_t level = 1; level + 1 < node_level_counts.size(); ++level) {
      Launch(node_level_counts[level + 1], [&] {
        ReduceValueTreeLevelKernel(
            value_b.data(), node_level_offsets[level] - nodes,
            node_level_offsets[level + 1] - nodes, node_level_counts[level],
            node_level_counts[level + 1], kTolerance, &status);
      });
    }
    Launch(1, [&] {
      InitializeValueContextRootKernel(value_b.data(),
                                       node_level_offsets.back() - nodes);
    });
    for (int level = static_cast<int>(node_level_counts.size()) - 2; level >= 1;
         --level) {
      Launch(node_level_counts[level + 1], [&] {
        ExpandValueContextLevelKernel(
            value_b.data(), node_level_offsets[level] - nodes,
            node_level_offsets[level + 1] - nodes, node_level_counts[level],
            node_level_counts[level + 1], kTolerance, &status);
      });
    }
    Launch(first_parent_count, [&] {
      FinalizeValueSuffixFromParentsKernel(value_a.data(), nodes,
                                           value_b.data(), first_parent_count,
                                           kTolerance, &status);
    });
  }
  if (FinishAllowedDeviceFailure(status, name, "value scan",
                                 allowed_failure))
    return;
  Launch(horizon, [&] {
    FeedbackKernel(reduced.data(), value_suffix, horizon, kTolerance,
                   feedback.data(), &status);
  });
  if (FinishAllowedDeviceFailure(status, name, "feedback solve",
                                 allowed_failure))
    return;

  std::vector<int> stage_level_offsets{0};
  std::vector<int> stage_level_counts{std::max(horizon, 1)};
  int stage_tree_size = stage_level_counts.front();
  while (stage_level_counts.back() > 1) {
    stage_level_offsets.push_back(stage_tree_size);
    stage_level_counts.push_back((stage_level_counts.back() + 1) / 2);
    stage_tree_size += stage_level_counts.back();
  }
  std::vector<AffineMap> map_a(horizon),
      map_b(std::max(stage_tree_size - std::max(horizon, 1), 1));
  std::vector<Scalar> map_a_storage(static_cast<std::size_t>(horizon) *
                                    kTestMapEntries);
  std::vector<Scalar> map_b_storage(map_b.size() * kTestMapEntries);
  for (int stage = 0; stage < horizon; ++stage) {
    BindAffineMapScratch(&map_a[stage],
                         map_a_storage.data() +
                             static_cast<std::size_t>(stage) * kTestMapEntries,
                         kTestStateCapacity, kTestStateCapacity);
  }
  for (std::size_t node = 0; node < map_b.size(); ++node) {
    BindAffineMapScratch(&map_b[node],
                         map_b_storage.data() + node * kTestMapEntries,
                         kTestStateCapacity, kTestStateCapacity);
  }
  Launch(horizon, [&] {
    InitializeAffineMapsKernel(feedback.data(), horizon, map_a.data(), &status);
  });
  AffineMap *prefix = map_a.data();
  if (horizon > 1) {
    const int first_parent_count = stage_level_counts[1];
    Launch(first_parent_count, [&] {
      ReduceAffineLeavesKernel(map_a.data(), horizon, first_parent_count,
                               map_b.data(), &status);
    });
    for (std::size_t level = 1; level + 1 < stage_level_counts.size();
         ++level) {
      Launch(stage_level_counts[level + 1], [&] {
        ReduceAffineTreeLevelKernel(
            map_b.data(), stage_level_offsets[level] - horizon,
            stage_level_offsets[level + 1] - horizon, stage_level_counts[level],
            stage_level_counts[level + 1], &status);
      });
    }
    Launch(1, [&] {
      InitializeAffineContextRootKernel(map_b.data(),
                                        stage_level_offsets.back() - horizon);
    });
    for (int level = static_cast<int>(stage_level_counts.size()) - 2;
         level >= 1; --level) {
      Launch(stage_level_counts[level + 1], [&] {
        ExpandAffineContextLevelKernel(
            map_b.data(), stage_level_offsets[level] - horizon,
            stage_level_offsets[level + 1] - horizon, stage_level_counts[level],
            stage_level_counts[level + 1], &status);
      });
    }
    Launch(first_parent_count, [&] {
      FinalizeAffinePrefixFromParentsKernel(map_a.data(), horizon, map_b.data(),
                                            first_parent_count, &status);
    });
  }
  std::vector<int> reduced_state_offsets(nodes + 1);
  std::vector<int> state_offsets(nodes + 1);
  std::vector<int> control_offsets(horizon + 1);
  for (int index = 0; index < nodes; ++index) {
    reduced_state_offsets[index + 1] =
        reduced_state_offsets[index] + state_params[index].reduced_dim;
    state_offsets[index] = index * kTestStateCapacity;
  }
  state_offsets[nodes] = nodes * kTestStateCapacity;
  for (int index = 0; index <= horizon; ++index)
    control_offsets[index] = index * kTestControlCapacity;
  std::vector<Scalar> reduced_states(reduced_state_offsets.back());
  std::vector<Scalar> reduced_controls(static_cast<std::size_t>(horizon) *
                                       kTestControlCapacity);
  std::vector<Scalar> states(nodes * kTestStateCapacity);
  std::vector<Scalar> controls(horizon * kTestControlCapacity);
  Launch(nodes, [&] {
    ReconstructPrimalKernel(prefix, state_params.data(), control_params.data(),
                            feedback.data(), reduced_initial.data(),
                            reduced_state_offsets.data(), state_offsets.data(),
                            control_offsets.data(), horizon,
                            reduced_states.data(), reduced_controls.data(),
                            states.data(), controls.data(), &status);
  });
  if (FinishAllowedDeviceFailure(status, name, "affine rollout",
                                 allowed_failure))
    return;

  if (compare_cpu && allowed_failure == nullptr) {
    clqr::Workspace workspace;
    workspace.Reserve(problem);
    const clqr::SolutionView cpu = clqr::Solve(problem, workspace);
    Expect(cpu.status == clqr::SolveStatus::kOptimal,
           name + " CPU reference status=" +
               std::to_string(static_cast<int>(cpu.status)) +
               ", message=" + cpu.message);
    for (int i = 0; i < nodes; ++i) {
      for (std::size_t row = 0; row < cpu.states[i].size; ++row) {
        Expect(std::abs(states[i * kTestStateCapacity + row] -
                        cpu.states[i][row]) < kPrimalComparisonTolerance,
               "emulated state matches CPU before dual recovery at " +
                   std::to_string(i) + "," + std::to_string(row) +
                   ": emulated=" +
                   std::to_string(states[i * kTestStateCapacity + row]) +
                   ", CPU=" + std::to_string(cpu.states[i][row]));
      }
    }
    for (int i = 0; i < horizon; ++i) {
      for (std::size_t row = 0; row < cpu.controls[i].size; ++row) {
        Expect(std::abs(controls[i * kTestControlCapacity + row] -
                        cpu.controls[i][row]) < kPrimalComparisonTolerance,
               "emulated control matches CPU before dual recovery at " +
                   std::to_string(i) + "," + std::to_string(row) +
                   ": emulated=" +
                   std::to_string(controls[i * kTestControlCapacity + row]) +
                   ", CPU=" + std::to_string(cpu.controls[i][row]));
      }
    }
  }

  std::vector<Scalar> initial_multiplier(kTestStateCapacity);
  std::vector<Scalar> dynamics(horizon * kTestStateCapacity);
  std::vector<Scalar> mixed(horizon * kTestMixedCapacity);
  std::vector<Scalar> state_multipliers(horizon * kTestStateConstraintCapacity);
  std::vector<Scalar> terminal_multiplier(kTestStateConstraintCapacity);
  std::vector<int> dynamics_offsets(horizon + 1);
  std::vector<int> mixed_offsets(horizon + 1);
  std::vector<int> state_constraint_offsets(horizon + 1);
  for (int index = 0; index <= horizon; ++index) {
    dynamics_offsets[index] = index * kTestStateCapacity;
    mixed_offsets[index] = index * kTestMixedCapacity;
    state_constraint_offsets[index] = index * kTestStateConstraintCapacity;
  }
  const Scalar multiplier_rank_tolerance = kMinimumMultiplierRankTolerance;
  const Scalar multiplier_consistency_tolerance =
      std::max(multiplier_rank_tolerance,
               kMultiplierConsistencyTolerancePerTreeLevel *
                   static_cast<Scalar>(stage_level_counts.size()));
  std::vector<DualParam> dual_params(horizon);
  std::vector<StateDualParam> state_dual_params(horizon);
  std::vector<int> dual_free_columns(static_cast<std::size_t>(horizon) *
                                     kTestDualCapacity);
  std::vector<Scalar> dual_basis(static_cast<std::size_t>(horizon) *
                                 kTestDualCapacity * kTestDualCapacity);
  std::vector<Scalar> dual_offset(static_cast<std::size_t>(horizon) *
                                  kTestDualCapacity);
  std::vector<Scalar> state_dual_offset(static_cast<std::size_t>(horizon) *
                                        kTestStateConstraintCapacity);
  std::vector<Scalar> state_dual_left(static_cast<std::size_t>(horizon) *
                                      kTestStateConstraintCapacity *
                                      kTestDualCapacity);
  std::vector<Scalar> state_dual_right(static_cast<std::size_t>(horizon) *
                                       kTestStateConstraintCapacity *
                                       kTestDualCapacity);
  for (int stage = 0; stage < horizon; ++stage) {
    const std::size_t index = static_cast<std::size_t>(stage);
    dual_params[stage].free_columns =
        dual_free_columns.data() + index * kTestDualCapacity;
    dual_params[stage].basis =
        dual_basis.data() + index * kTestDualCapacity * kTestDualCapacity;
    dual_params[stage].offset = dual_offset.data() + index * kTestDualCapacity;
    state_dual_params[stage].offset =
        state_dual_offset.data() + index * kTestStateConstraintCapacity;
    state_dual_params[stage].left =
        state_dual_left.data() +
        index * kTestStateConstraintCapacity * kTestDualCapacity;
    state_dual_params[stage].right =
        state_dual_right.data() +
        index * kTestStateConstraintCapacity * kTestDualCapacity;
  }
  std::vector<DualRelation> dual_tree(stage_tree_size);
  std::vector<DualNodeValue> dual_values(stage_tree_size);
  std::vector<Scalar> dual_tree_storage(
      static_cast<std::size_t>(stage_tree_size) * kTestDualRelationEntries);
  std::vector<Scalar> dual_value_storage(
      static_cast<std::size_t>(stage_tree_size) * kTestDualValueEntries);
  for (int node = 0; node < stage_tree_size; ++node) {
    BindDualRelationScratch(&dual_tree[node],
                            dual_tree_storage.data() +
                                static_cast<std::size_t>(node) *
                                    kTestDualRelationEntries,
                            kTestDualCapacity, kTestDualCapacity);
    BindDualValueScratch(&dual_values[node],
                         dual_value_storage.data() +
                             static_cast<std::size_t>(node) *
                                 kTestDualValueEntries,
                         kTestDualCapacity);
  }
  int dual_scan_needed = 0;
  if (horizon > 0) {
    Launch(horizon, [&] {
      BuildDualParametersKernel(
          stages.data(), state_params.data(), value_suffix,
          reduced_states.data(), states.data(), controls.data(),
          reduced_state_offsets.data(), state_offsets.data(),
          control_offsets.data(), horizon, multiplier_rank_tolerance,
          multiplier_consistency_tolerance, dual_params.data(),
          &dual_scan_needed, nullptr, &status);
    });
    Launch(horizon, [&] {
      BuildDualParameterRelationsKernel(
          stages.data(), &terminal, dual_params.data(), horizon, states.data(),
          controls.data(), state_offsets.data(), control_offsets.data(),
          multiplier_rank_tolerance, multiplier_consistency_tolerance,
          dual_tree.data(), &dual_scan_needed, state_dual_params.data(),
          &status);
    });
    for (std::size_t level = 0; level + 1 < stage_level_counts.size();
         ++level) {
      Launch(stage_level_counts[level + 1], [&] {
        ReduceDualTreeLevelKernel(
            dual_tree.data(), stage_level_offsets[level],
            stage_level_offsets[level + 1], stage_level_counts[level],
            stage_level_counts[level + 1], multiplier_rank_tolerance,
            multiplier_consistency_tolerance, dual_tree.data(),
            &dual_scan_needed, &status);
      });
    }
    const int root = stage_level_offsets.back();
    Launch(1, [&] {
      SolveDualRootKernel(dual_tree.data() + root, dual_values.data() + root,
                          &dual_scan_needed, &status,
                          multiplier_rank_tolerance);
    });
    for (int level = static_cast<int>(stage_level_counts.size()) - 2;
         level >= 0; --level) {
      Launch(stage_level_counts[level + 1], [&] {
        ExpandDualTreeLevelKernel(
            dual_tree.data(), stage_level_offsets[level],
            stage_level_offsets[level + 1], stage_level_counts[level],
            stage_level_counts[level + 1], multiplier_rank_tolerance,
            multiplier_consistency_tolerance, dual_values.data(),
            dual_values.data(), &dual_scan_needed, &status);
      });
    }
    Launch(horizon, [&] {
      RecoverParameterizedMultipliersKernel(
          dual_params.data(), state_dual_params.data(), dual_values.data(),
          dynamics_offsets.data(), mixed_offsets.data(),
          state_constraint_offsets.data(), horizon, dynamics.data(),
          mixed.data(), state_multipliers.data(), terminal_multiplier.data(),
          &status);
    });
  }
  Launch(1, [&] {
    RecoverInitialMultiplierKernel(
        stages.data(), &terminal, horizon, states.data(), controls.data(),
        dynamics.data(), mixed.data(), state_offsets.data(),
        control_offsets.data(), dynamics_offsets.data(), mixed_offsets.data(),
        state_constraint_offsets.data(), initial_multiplier.data(),
        state_multipliers.data(), terminal_multiplier.data(), &status);
  });
  if (FinishAllowedDeviceFailure(status, name, "multiplier recovery",
                                 allowed_failure))
    return;

  std::string worst_residual;
  const Scalar residual = MaxResidual(
      problem, states, controls, initial_multiplier, dynamics, mixed,
      state_multipliers, terminal_multiplier, &worst_residual);
  const Scalar kkt_tolerance = horizon >= 256
                                   ? kLongHorizonKktComparisonTolerance
                                   : kKktComparisonTolerance;
  Expect(std::isfinite(residual) &&
             residual < kkt_tolerance * kkt_tolerance_scale,
         name + " emulated full KKT residual: " + std::to_string(residual) +
             " in " + worst_residual);
  const char *validation = compare_cpu ? "matched CPU" : "completed";
  std::cout << name << " CUDA kernel emulation "
            << validation << "; KKT residual=" << residual << '\n';
}

bool FitsAdversarialEmulationStorage(const Problem &problem) {
  if (problem.terminal_Q.rows() > static_cast<std::size_t>(kTestStateCapacity) ||
      problem.terminal_E.rows() >
          static_cast<std::size_t>(kTestStateConstraintCapacity)) {
    return false;
  }
  for (const Stage &stage : problem.stages) {
    if (stage.A.cols() > static_cast<std::size_t>(kTestStateCapacity) ||
        stage.A.rows() > static_cast<std::size_t>(kTestStateCapacity) ||
        stage.B.cols() > static_cast<std::size_t>(kTestControlCapacity) ||
        stage.C.rows() > static_cast<std::size_t>(kTestMixedCapacity) ||
        stage.E.rows() >
            static_cast<std::size_t>(kTestStateConstraintCapacity)) {
      return false;
    }
  }
  return true;
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
  TinyCoefficientRrefCase();
  FiniteInputValidationCase();
  DeviceObjectiveCase();
  PivotedLuMultiRhsCase();
  OrthogonalEchelonCase();
  IllConditionedPositiveDefiniteMultiRhsCase();
  InvalidValueElementCopyCase();
  FreeFixedFreeValueCompositionCase();
  DualRelationLeafScratchSizeCase();
  ScratchPlannerTopologyCase();
  NonPositiveDefiniteReducedControlCostCase();
  RunEmulation(MakeProblem(), "rank-deficient constrained", true, true);
  RunEmulation(ZeroHorizonProblem(), "zero-horizon", false, false);
  RunEmulation(
      UniformProblem(1800, 3, kTestStateCapacity, kTestControlCapacity),
      "maximum-active-dimension", false, false);
  RunEmulation(ZeroControlStateConstraintProblem(), "zero-control", true,
               false);
  RunEmulation(ExactDualRelationScratchProblem(), "exact-dual-relation-scratch",
               true, false);
  RunEmulation(HeterogeneousDimensionProblem(), "heterogeneous-dimensions",
               false, false);
  RunEmulation(MaximumConstraintProblem(), "maximum-constraint", true, true);
  RunEmulation(MoreMixedRowsThanControlsProblem(), "more-mixed-than-controls",
               true, true);
  RunEmulation(clqr::benchmark::StateOnlyProblem(3, 3, 1, 1),
               "short-horizon-state", true, false);
#ifdef CLQR_USE_FLOAT
  std::cout << "exact-JAX-fixture CUDA kernel emulation skipped in FP32 "
               "(its deliberately duplicate state rows make multiplier "
               "recovery fail the native consistency gate)\n";
#else
  RunEmulation(clqr::test::MakeJaxCrossValidationProblem(), "exact-JAX-fixture",
               true, false);
#endif
  RunEmulation(LongHorizonStateConstraintProblem(), "long-horizon-state", true,
               false, false);
  std::vector<clqr::test::adversarial::TestCase> cases =
      clqr::test::adversarial::StandardCases();
  if (extended) {
    std::vector<clqr::test::adversarial::TestCase> more =
        clqr::test::adversarial::ExtendedCases();
    cases.insert(cases.end(), more.begin(), more.end());
  }
  std::size_t executed = 0;
  for (const clqr::test::adversarial::TestCase &test_case : cases) {
    if (!test_case.emulate ||
        (test_case.cuda_status != clqr::SolveStatus::kOptimal &&
         test_case.cuda_status != clqr::SolveStatus::kNumericalFailure &&
         test_case.cuda_status != clqr::SolveStatus::kInfeasible) ||
        !FitsAdversarialEmulationStorage(test_case.problem)) {
      continue;
    }
    RunEmulation(test_case.problem, "shared-" + test_case.name, false, false,
                 true, test_case.tolerance_scale *
                           test_case.kkt_tolerance_scale);
    ++executed;
  }
  std::cout << "all " << executed
            << " selected shared adversarial emulation cases passed\n";
  return 0;
}
