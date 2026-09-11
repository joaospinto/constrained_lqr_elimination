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
#include "../benchmarks/paper_cases.h"
#include "../benchmarks/scaling_problem.h"
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

std::size_t MaximumScratchBytes(const ScratchRequirements &scratch) {
  return std::max({scratch.primal_leaf, scratch.primal_relation,
                   scratch.primal_relation_final, scratch.state_parameter,
                   scratch.stage_reduction, scratch.terminal_reduction,
                   scratch.value_leaf, scratch.value_compose,
                   scratch.value_finalize, scratch.feedback,
                   scratch.affine_terms, scratch.affine_finalize,
                   scratch.dual_parameter, scratch.dual_relation_leaf,
                   scratch.dual_relation, scratch.dual_root,
                   scratch.dual_expand});
}

template <typename T>
concept HasAffineRightEndpoint = requires(T value) { value.b; };

template <typename T>
concept HasAffineLeftEndpoint = requires(T value) { value.eta; };

static_assert(!HasAffineRightEndpoint<ValueElement>);
static_assert(!HasAffineLeftEndpoint<ValueElement>);

// Test-only backing strides deliberately exceed the former production
// capacities (8/8/8/8). They are not solver limits.
#ifdef CLQR_EMULATION_RANK_REGRESSION
constexpr int kTestStateCapacity = 32;
constexpr int kTestControlCapacity = 16;
#else
constexpr int kTestStateCapacity = 24;
constexpr int kTestControlCapacity = 12;
#endif
constexpr int kTestMixedCapacity = 10;
constexpr int kTestStateConstraintCapacity = 10;
constexpr int kTestDualCapacity = kTestStateCapacity + kTestMixedCapacity;
constexpr int kTestRelationRows = 2 * kTestStateCapacity;
constexpr int kTestRelationEntries =
    kTestRelationRows * (2 * kTestStateCapacity) + kTestRelationRows;
constexpr int kTestValueEntries = 3 * kTestStateCapacity * kTestStateCapacity;
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

const AllowedDeviceFailure *AllowedFailureForCase(const std::string &name) {
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
      {"shared-stable-extended-horizon-257",
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

void CheckReducedObjectiveIdentity(
    const Problem& problem, const std::vector<StateParam>& states,
    const std::vector<ControlParam>& controls,
    const std::vector<ReducedStage>& stages, const ReducedTerminal& terminal,
    const std::string& name) {
  const auto matrix = [](const Scalar* values, int rows, int cols) {
    Matrix out(rows, cols);
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j) out(i, j) = values[i * cols + j];
    return out;
  };
  const auto vector = [](const Scalar* values, int rows) {
    Vector out(rows);
    for (int i = 0; i < rows; ++i) out[i] = values[i];
    return out;
  };
  const std::size_t N = stages.size();
  Problem reduced;
  reduced.stages.resize(N);
  reduced.Q.resize(N + 1);
  reduced.q.resize(N + 1);
  std::vector<Vector> x(N + 1), z(N + 1), t(N + 1), u(N), v(N), y(N);
  for (std::size_t i = 0; i <= N; ++i) {
    t[i] = vector(states[i].t, states[i].physical_dim);
    const int n = states[i].reduced_dim;
    reduced.Q[i] = matrix(i == N ? terminal.Q : stages[i].Q, n, n);
    reduced.q[i] = vector(i == N ? terminal.q : stages[i].q, n);
    if (i < N) {
      y[i] = vector(controls[i].y, controls[i].physical_dim);
      auto& s = reduced.stages[i];
      s.M = matrix(stages[i].M, n, stages[i].m);
      s.R = matrix(stages[i].R, stages[i].m, stages[i].m);
      s.r = vector(stages[i].r, stages[i].m);
    }
  }
  const long double kappa = clqr::benchmark::OriginalObjective(problem, t, y);
  clqr::benchmark::Random random(7901);
  long double maximum = 0;
  for (int probe = 0; probe < 4; ++probe) {
    for (std::size_t i = 0; i <= N; ++i) {
      const auto& s = states[i];
      z[i] = random.Vec(s.reduced_dim, probe == 0 ? Scalar{0} : Scalar{0.7});
      x[i] = matrix(s.T, s.physical_dim, s.reduced_dim) * z[i] + t[i];
      if (i < N) {
        const auto& c = controls[i];
        v[i] = random.Vec(c.reduced_dim, probe == 0 ? Scalar{0} : Scalar{0.4});
        u[i] = matrix(c.Y, c.physical_dim, c.state_dim) * z[i] +
               matrix(c.Z, c.physical_dim, c.reduced_dim) * v[i] + y[i];
      }
    }
    const long double original = clqr::benchmark::OriginalObjective(problem, x, u);
    const long double transformed = clqr::benchmark::OriginalObjective(reduced, z, v);
    const long double relative = std::abs(original - transformed - kappa) /
        std::max({1.0L, std::abs(original), std::abs(transformed), std::abs(kappa)});
#ifdef CLQR_USE_FLOAT
    constexpr long double tolerance = 5e-4L;
#else
    constexpr long double tolerance = 5e-11L;
#endif
    Expect(std::isfinite(relative) && relative <= tolerance,
           name + " reduced-plus-constant objective identity, relative error=" +
               std::to_string(static_cast<double>(relative)));
    maximum = std::max(maximum, relative);
  }
  std::cout << name << " objective identity: max relative discrepancy=" << maximum << '\n';
}

bool FinishAllowedDeviceFailure(const DeviceStatus &status,
                                const std::string &name, const char *phase,
                                const AllowedDeviceFailure *allowed) {
  if (status.code == kDeviceOk)
    return false;
  Expect(allowed != nullptr,
         name + " unexpected " + phase +
             " failure (stage=" + std::to_string(status.stage) +
             ", detail=" + std::to_string(status.detail) + ")");
  Expect(status.code == allowed->code && std::string(phase) == allowed->phase &&
             status.detail == allowed->detail,
         name + " unexpected device rejection during " + phase +
             " (stage=" + std::to_string(status.stage) +
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

void DualResidualOrthogonalEchelonCase() {
#ifdef CLQR_USE_FLOAT
  constexpr Scalar comparison_tolerance = 2e-4f;
#else
  constexpr Scalar comparison_tolerance = 2e-11;
#endif
  constexpr int rows = 4;
  constexpr int eliminated = 2;
  constexpr int left_dim = 2;
  constexpr int right_dim = 1;
  constexpr int variables = eliminated + left_dim + right_dim;
  constexpr int columns = variables + 1;
  Scalar matrix[rows * columns]{
      Scalar{0},    Scalar{0},    Scalar{1e6}, Scalar{2e6}, Scalar{0},
      Scalar{5e6},  Scalar{0},    Scalar{0},   Scalar{0},   Scalar{1e-6},
      Scalar{1e-6}, Scalar{4e-6}, Scalar{0},   Scalar{0},   Scalar{3},
      Scalar{9},    Scalar{3},    Scalar{27},  Scalar{0},   Scalar{0},
      Scalar{0},    Scalar{0},    Scalar{0},   Scalar{0}};
  int pivot_columns[rows]{};
  int permutation[variables]{};
  int rank = -1;
  int best_column = -1;
  Scalar reflector[rows]{};
  Scalar matrix_scale = Scalar{0};
  threadIdx.x = 0;
  blockDim.x = 1;

  OrthogonalEchelonBlock(matrix, rows, columns, variables, variables,
                         kTolerance, pivot_columns, permutation, &rank,
                         &best_column, reflector, &matrix_scale,
                         kMinimumDualRelationRowScale);

  Expect(rank == 2, "dual residual QR detects the outer relation rank");
  for (int row = 0; row < rank; ++row) {
    Expect(pivot_columns[row] >= eliminated,
           "dual residual QR does not pivot eliminated zero columns");
  }
  constexpr int relation_capacity = left_dim + right_dim;
  constexpr int relation_entries = relation_capacity * left_dim +
                                   relation_capacity * right_dim +
                                   relation_capacity;
  Scalar relation_storage[relation_entries]{};
  DualRelation relation{};
  BindDualRelationScratch(&relation, relation_storage, left_dim, right_dim);
  ExtractResidualRelation(matrix, columns, rank, pivot_columns, eliminated,
                          left_dim, right_dim, &relation);
  Expect(relation.rows == 2 && relation.left_dim == left_dim &&
             relation.right_dim == right_dim,
         "dual residual QR extracts the expected relation shape");
  const Scalar solution[left_dim + right_dim]{Scalar{1}, Scalar{2}, Scalar{2}};
  for (int row = 0; row < relation.rows; ++row) {
    Scalar residual = -relation.rhs[row];
    for (int col = 0; col < left_dim; ++col)
      residual += relation.left[row * left_dim + col] * solution[col];
    for (int col = 0; col < right_dim; ++col) {
      residual +=
          relation.right[row * right_dim + col] * solution[left_dim + col];
    }
    Expect(std::abs(residual) < comparison_tolerance,
           "dual residual QR preserves the affine outer relation");
  }

  constexpr int inconsistent_rows = 2;
  constexpr int inconsistent_variables = 3;
  constexpr int inconsistent_columns = inconsistent_variables + 1;
  Scalar inconsistent[inconsistent_rows * inconsistent_columns]{
      Scalar{0}, Scalar{1}, Scalar{0}, Scalar{1},
      Scalar{0}, Scalar{1}, Scalar{0}, Scalar{2}};
  int inconsistent_pivots[inconsistent_rows]{};
  int inconsistent_permutation[inconsistent_variables]{};
  int inconsistent_rank = -1;
  Scalar inconsistent_reflector[inconsistent_rows]{};
  OrthogonalEchelonBlock(inconsistent, inconsistent_rows, inconsistent_columns,
                         inconsistent_variables, inconsistent_variables,
                         kTolerance, inconsistent_pivots,
                         inconsistent_permutation, &inconsistent_rank,
                         &best_column, inconsistent_reflector, &matrix_scale,
                         kMinimumDualRelationRowScale);
  Expect(inconsistent_rank == 1,
         "dual residual QR identifies a repeated coefficient row");
  Expect(InconsistentRref(inconsistent, inconsistent_rows, inconsistent_columns,
                          inconsistent_variables, kTolerance, kTolerance),
         "dual residual QR retains the orthogonal consistency residual");

#ifdef CLQR_USE_FLOAT
  constexpr Scalar marginal_residual = 8e-2f;
#else
  constexpr Scalar marginal_residual = 4e-6;
#endif
  Scalar marginal[4]{Scalar{1}, Scalar{0}, Scalar{1}, marginal_residual};
  int marginal_pivots[1]{};
  int marginal_permutation[1]{};
  int marginal_rank = -1;
  Scalar marginal_reflector[2]{};
  OrthogonalEchelonBlock(marginal, 2, 2, 1, 1, kMinimumMultiplierRankTolerance,
                         marginal_pivots, marginal_permutation, &marginal_rank,
                         &best_column, marginal_reflector, &matrix_scale,
                         kMinimumDualRelationRowScale);
  const Scalar tree_tolerance =
      kMultiplierConsistencyTolerancePerTreeLevel * Scalar{7};
  const Scalar leaf_tolerance = kMultiplierConsistencyTolerancePerTreeLevel;
  Expect(!InconsistentRref(marginal, 2, 2, 1, kMinimumMultiplierRankTolerance,
                           tree_tolerance),
         "tree-accumulated tolerance admits a marginal leaf residual");
  Expect(InconsistentRref(marginal, 2, 2, 1, kMinimumMultiplierRankTolerance,
                          leaf_tolerance),
         "per-leaf tolerance rejects a marginal leaf residual");
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
  Scalar second_c[]{4, Scalar{0.2}, Scalar{-0.1}, Scalar{0.2},
                    5, Scalar{0.3}, Scalar{-0.1}, Scalar{0.3},
                    6};
  ValueElement first{};
  first.left_dim = 2;
  first.right_dim = 0;
  first.J = first_j;
  ValueElement second{};
  second.left_dim = 0;
  second.right_dim = 3;
  second.C = second_c;

  Scalar output_a[6];
  Scalar output_j[4]{};
  Scalar output_c[9]{};
  std::fill(std::begin(output_a), std::end(output_a), Scalar{17});
  ValueElement output{};
  output.A = output_a;
  output.J = output_j;
  output.C = output_c;
  DeviceStatus status{kDeviceOk, -1, 0};
  Scalar augmented[1]{};
  Scalar factors[1]{};
  Scalar product[1]{};
  int best_row = -1;
  threadIdx.x = 0;
  blockDim.x = 1;

  ComposeValueElementsBlock(first, second, kTolerance, &output, &status, 0,
                            augmented, factors, product, &best_row);

  Expect(status.code == kDeviceOk, "free-fixed-free value composition status");
  Expect(output.left_dim == 2 && output.right_dim == 3,
         "free-fixed-free value composition dimensions");
  for (Scalar value : output_a)
    Expect(value == Scalar{0},
           "free-fixed-free value composition publishes a zero cross map");
  for (std::size_t i = 0; i < std::size(first_j); ++i)
    Expect(output_j[i] == first_j[i],
           "free-fixed-free value composition preserves the left Hessian");
  for (std::size_t i = 0; i < std::size(second_c); ++i)
    Expect(output_c[i] == second_c[i],
           "free-fixed-free value composition preserves the right curvature");
}

void NonuniformValueCompositionCase() {
  constexpr int left = 2;
  constexpr int shared = 3;
  constexpr int right = 4;
  constexpr int columns = 2 * shared + left;
  Scalar first_a[]{Scalar{1.0}, Scalar{0.2}, Scalar{-0.3},
                   Scalar{0.8}, Scalar{0.4}, Scalar{-0.5}};
  Scalar first_c[]{Scalar{0.2},   Scalar{0.02}, Scalar{-0.01},
                   Scalar{0.02},  Scalar{0.3},  Scalar{0.04},
                   Scalar{-0.01}, Scalar{0.04}, Scalar{0.25}};
  Scalar first_j[]{Scalar{1.4}, Scalar{0.1}, Scalar{0.1}, Scalar{1.1}};
  Scalar second_a[]{Scalar{0.6}, Scalar{-0.2}, Scalar{0.1},  Scalar{0.3},
                    Scalar{0.7}, Scalar{-0.4}, Scalar{-0.5}, Scalar{0.2},
                    Scalar{0.8}, Scalar{0.9},  Scalar{0.1},  Scalar{-0.3}};
  Scalar second_c[]{Scalar{0.4},   Scalar{0.03},  Scalar{-0.02}, Scalar{0.01},
                    Scalar{0.03},  Scalar{0.5},   Scalar{0.04},  Scalar{-0.01},
                    Scalar{-0.02}, Scalar{0.04},  Scalar{0.6},   Scalar{0.02},
                    Scalar{0.01},  Scalar{-0.01}, Scalar{0.02},  Scalar{0.45}};
  Scalar second_j[]{Scalar{0.7},   Scalar{0.05}, Scalar{-0.02},
                    Scalar{0.05},  Scalar{0.9},  Scalar{0.03},
                    Scalar{-0.02}, Scalar{0.03}, Scalar{0.8}};
  ValueElement first{left, shared, first_a, first_c, first_j};
  ValueElement second{shared, right, second_a, second_c, second_j};

  Scalar output_a[right * left]{};
  Scalar output_c[right * right]{};
  Scalar output_j[left * left]{};
  ValueElement output{0, 0, output_a, output_c, output_j};
  Scalar augmented[shared * columns]{};
  Scalar factors[shared]{};
  Scalar product[shared * right]{};
  DeviceStatus status{kDeviceOk, -1, 0};
  int best_row = -1;
  threadIdx.x = 0;
  blockDim.x = 1;
  ComposeValueElementsBlock(first, second, kTolerance, &output, &status, 0,
                            augmented, factors, product, &best_row);
  Expect(status.code == kDeviceOk,
         "nonuniform staged value composition status");

  Scalar reference_augmented[shared * columns]{};
  for (int row = 0; row < shared; ++row) {
    for (int col = 0; col < shared; ++col) {
      Scalar value = row == col ? Scalar{1} : Scalar{0};
      for (int k = 0; k < shared; ++k)
        value += first_c[row * shared + k] * second_j[k * shared + col];
      reference_augmented[row * columns + col] = value;
    }
    for (int col = 0; col < left; ++col)
      reference_augmented[row * columns + shared + col] =
          first_a[row * left + col];
    for (int col = 0; col < shared; ++col)
      reference_augmented[row * columns + shared + left + col] =
          first_c[row * shared + col];
  }
  Scalar reference_factors[shared]{};
  int reference_best_row = -1;
  Expect(SolveGeneralMultipleRhsBlock(reference_augmented, shared, columns,
                                      kTolerance, reference_factors,
                                      &reference_best_row),
         "nonuniform reference value solve");

  Scalar reference_a[right * left]{};
  Scalar reference_c[right * right]{};
  Scalar reference_j[left * left]{};
  for (int row = 0; row < right; ++row) {
    for (int col = 0; col < left; ++col) {
      for (int k = 0; k < shared; ++k) {
        reference_a[row * left + col] +=
            second_a[row * shared + k] *
            reference_augmented[k * columns + shared + col];
      }
    }
  }
  for (int row = 0; row < right; ++row) {
    for (int col = 0; col < right; ++col) {
      Scalar value = second_c[row * right + col];
      for (int p = 0; p < shared; ++p) {
        for (int q = 0; q < shared; ++q) {
          value += second_a[row * shared + p] *
                   reference_augmented[p * columns + shared + left + q] *
                   second_a[col * shared + q];
        }
      }
      reference_c[row * right + col] = value;
    }
  }
  for (int row = 0; row < left; ++row) {
    for (int col = 0; col < left; ++col) {
      Scalar value = first_j[row * left + col];
      for (int p = 0; p < shared; ++p) {
        for (int q = 0; q < shared; ++q) {
          value += first_a[p * left + row] * second_j[p * shared + q] *
                   reference_augmented[q * columns + shared + col];
        }
      }
      reference_j[row * left + col] = value;
    }
  }
  for (int row = 0; row < right; ++row) {
    for (int col = row + 1; col < right; ++col) {
      const Scalar value = Scalar{0.5} * (reference_c[row * right + col] +
                                          reference_c[col * right + row]);
      reference_c[row * right + col] = value;
      reference_c[col * right + row] = value;
    }
  }
  for (int row = 0; row < left; ++row) {
    for (int col = row + 1; col < left; ++col) {
      const Scalar value = Scalar{0.5} * (reference_j[row * left + col] +
                                          reference_j[col * left + row]);
      reference_j[row * left + col] = value;
      reference_j[col * left + row] = value;
    }
  }
#ifdef CLQR_USE_FLOAT
  constexpr Scalar comparison_tolerance = Scalar{2e-6};
#else
  constexpr Scalar comparison_tolerance = Scalar{2e-14};
#endif
  for (int entry = 0; entry < right * left; ++entry)
    Expect(std::abs(output_a[entry] - reference_a[entry]) <
               comparison_tolerance,
           "nonuniform staged value A entry " + std::to_string(entry));
  for (int entry = 0; entry < right * right; ++entry)
    Expect(std::abs(output_c[entry] - reference_c[entry]) <
               comparison_tolerance,
           "nonuniform staged value C entry " + std::to_string(entry));
  for (int entry = 0; entry < left * left; ++entry)
    Expect(std::abs(output_j[entry] - reference_j[entry]) <
               comparison_tolerance,
           "nonuniform staged value J entry " + std::to_string(entry));
}

void StagedFeedbackSystemCase() {
  constexpr int n = 3;
  constexpr int next_n = 4;
  constexpr int m = 2;
  constexpr int columns = m + n;
  Scalar A[]{Scalar{0.8}, Scalar{-0.2}, Scalar{0.1},  Scalar{0.3},
             Scalar{0.7}, Scalar{-0.1}, Scalar{-0.4}, Scalar{0.2},
             Scalar{0.9}, Scalar{0.5},  Scalar{0.1},  Scalar{-0.3}};
  Scalar B[]{Scalar{0.6},  Scalar{-0.2}, Scalar{0.1}, Scalar{0.7},
             Scalar{-0.5}, Scalar{0.4},  Scalar{0.8}, Scalar{0.3}};
  Scalar R[]{Scalar{1.4}, Scalar{0.1}, Scalar{0.1}, Scalar{1.2}};
  Scalar M[]{Scalar{0.02}, Scalar{-0.03}, Scalar{0.04},
             Scalar{0.01}, Scalar{-0.02}, Scalar{0.05}};
  Scalar J[]{Scalar{0.9},   Scalar{0.03},  Scalar{-0.02}, Scalar{0.01},
             Scalar{0.03},  Scalar{0.8},   Scalar{0.04},  Scalar{-0.01},
             Scalar{-0.02}, Scalar{0.04},  Scalar{0.7},   Scalar{0.02},
             Scalar{0.01},  Scalar{-0.01}, Scalar{0.02},  Scalar{0.6}};
  ReducedStage stage{};
  stage.n = n;
  stage.next_n = next_n;
  stage.m = m;
  stage.A = A;
  stage.B = B;
  stage.R = R;
  stage.M = M;
  ValueElement next{};
  next.left_dim = next_n;
  next.J = J;
  Scalar augmented[m * columns]{};
  Scalar product[next_n * columns]{};
  threadIdx.x = 0;
  blockDim.x = 1;
  BuildMatrixFeedbackSystem(stage, next, augmented, product, columns);

#ifdef CLQR_USE_FLOAT
  constexpr Scalar comparison_tolerance = Scalar{2e-6};
#else
  constexpr Scalar comparison_tolerance = Scalar{2e-14};
#endif
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < columns; ++col) {
      Scalar expected = col < m ? R[row * m + col] : -M[(col - m) * m + row];
      for (int a = 0; a < next_n; ++a) {
        for (int b = 0; b < next_n; ++b) {
          const Scalar operand = col < m ? B[b * m + col] : A[b * n + col - m];
          const Scalar term = B[a * m + row] * J[a * next_n + b] * operand;
          expected += col < m ? term : -term;
        }
      }
      Expect(std::abs(augmented[row * columns + col] - expected) <
                 comparison_tolerance,
             "staged feedback system entry " +
                 std::to_string(row * columns + col));
    }
  }
}

void TerminalReductionScratchPaddingCase() {
  Scalar terminal_Q[]{Scalar{2}};
  Scalar terminal_q[]{Scalar{-0.2}};
  PackedTerminal terminal{};
  terminal.n = 1;
  terminal.Q = terminal_Q;
  terminal.q = terminal_q;
  Scalar transform[]{Scalar{1}};
  Scalar offset[]{Scalar{0.4}};
  StateParam param{};
  param.physical_dim = 1;
  param.reduced_dim = 1;
  param.T = transform;
  param.t = offset;
  Scalar reduced_Q[1]{};
  Scalar reduced_q[1]{};
  ReducedTerminal reduced{};
  reduced.Q = reduced_Q;
  reduced.q = reduced_q;
  threadIdx.x = 0;
  blockDim.x = 1;

  ReduceTerminalKernel(&terminal, &param, 0, &reduced);

  Expect(g_emulated_block_scratch_bytes == kSharedVectorAccessBytes,
         "scalar terminal reduction requests one complete shared-memory "
         "transaction");
  Expect(std::abs(reduced_Q[0] - Scalar{2}) < kTolerance &&
             std::abs(reduced_q[0] - Scalar{0.6}) < kTolerance,
         "scalar terminal reduction preserves the staged Hessian and "
         "gradient");

  // Exercise the production global pointer path with explicit guards and
  // neighboring block slices, not just the emulation-owned backing store.
  constexpr std::size_t stride = 16;
  alignas(16) unsigned char guarded[5 * stride];
  std::fill(std::begin(guarded), std::end(guarded), 0xa5);
  blockIdx.x = 2;
  ReduceTerminalKernel<true>(&terminal, &param, 0, &reduced,
                             guarded + stride, stride);
  for (std::size_t i = 0; i < sizeof(guarded); ++i)
    if (i < 3 * stride || i >= 4 * stride)
      Expect(guarded[i] == 0xa5, "global scratch changed another block or guard");
  Expect(std::abs(reduced_Q[0] - Scalar{2}) < kTolerance &&
             std::abs(reduced_q[0] - Scalar{0.6}) < kTolerance,
         "explicit global scratch preserves terminal reduction");
  blockIdx.x = 0;
}

void DualRelationLeafScratchSizeCase() {
  ScratchSize without_constraint_scales;
  without_constraint_scales.Add<Scalar>(2);
  without_constraint_scales.Add<Scalar>(1);
  without_constraint_scales.Add<int>(1);
  without_constraint_scales.Add<int>(1);
  Expect(DualRelationLeafScratchBytes(2, 1, 1, 1) ==
             without_constraint_scales.bytes + sizeof(Scalar),
         "dual-relation leaf scratch includes state-constraint scales");

  ScratchSize more_columns_than_rows;
  more_columns_than_rows.Add<Scalar>(6);
  more_columns_than_rows.Add<Scalar>(1);
  more_columns_than_rows.Add<Scalar>(1);
  more_columns_than_rows.Add<int>(1);
  more_columns_than_rows.Add<int>(5);
  Expect(DualRelationLeafScratchBytes(6, 1, 1, 5) ==
             more_columns_than_rows.bytes,
         "dual-relation leaf scratch sizes the full QR permutation at runtime");
}

Problem PathologicalScratchProblem() {
  constexpr std::size_t n = 45;
  Problem problem;
  problem.initial_state = Vector(n);
  problem.stages.resize(1);
  problem.Q.resize(2);
  problem.q.resize(2);
  Stage &stage = problem.stages[0];
  stage.A = Matrix(0, n);
  stage.B = Matrix(0, 0);
  stage.c = Vector(0);
  problem.Q[0] = Matrix(n, n);
  for (std::size_t row = 0; row < n; ++row)
    problem.Q[0](row, row) = Scalar{1};
  stage.R = Matrix(0, 0);
  stage.M = Matrix(n, 0);
  problem.q[0] = Vector(n);
  stage.r = Vector(0);
  stage.C = Matrix(0, n);
  stage.D = Matrix(0, 0);
  stage.d = Vector(0);
  stage.E = Matrix(0, n);
  stage.e = Vector(0);
  problem.Q.back() = Matrix(0, 0);
  problem.q.back() = Vector(0);
  problem.terminal_E = Matrix(0, 0);
  problem.terminal_e = Vector(0);
  return problem;
}

void ScratchPlannerTopologyCase() {
  const auto optin_problem =
      clqr::benchmark::MakeScalingProblem(8, 24, 12, 3, 6).problem;
  const std::size_t optin_bytes =
      sizeof(Scalar) * (16 * 24 * 24 + 10 * 24) + 8 * 4 * 24 + 40;
  Expect(MaximumScratchBytes(PlanScratch(optin_problem)) == optin_bytes,
         "native opt-in fixture uses the predicted shared-memory footprint");
  constexpr std::size_t kUsableP100SharedBytes = 48 * 1024 - 256;
  const ScratchRequirements pathological =
      PlanScratch(PathologicalScratchProblem());
  Expect(MaximumScratchBytes(pathological) <= kUsableP100SharedBytes,
         "topology-aware scratch accepts a 45-to-0 state transition");
  Expect(DenseEliminationScratchBytes(4 * 45, 3 * 45 + 1,
                                      "legacy synthetic workspace") >
             kUsableP100SharedBytes,
         "the former synthetic cross-maximum would reject the 45-to-0 case");

  constexpr std::size_t n = 8;
  const ScanShape nonuniform_first = MakeScanShape(2, 3);
  const ScanShape nonuniform_second = MakeScanShape(3, 4);
  ScratchSize nonuniform_value_compose;
  nonuniform_value_compose.Add<Scalar>(3 * (2 * 3 + 2));
  nonuniform_value_compose.Add<Scalar>(3);
  nonuniform_value_compose.Add<Scalar>(3 * 4);
  Expect(ValueComposeScratchBytes(nonuniform_first, nonuniform_second,
                                  "nonuniform value workspace") ==
             nonuniform_value_compose.bytes,
         "nonuniform value-composition scratch uses the exact largest staged "
         "product");
  bool staged_overflow_rejected = false;
  try {
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const ScanShape large_first{maximum / 4, 2, 0, true};
    const ScanShape large_second{2, maximum / 2 + 1, 0, true};
    (void)ValueComposeScratchBytes(large_first, large_second,
                                   "overflowing staged value workspace");
  } catch (const std::invalid_argument &) {
    staged_overflow_rejected = true;
  }
  Expect(staged_overflow_rejected,
         "value-composition planner rejects staged-product size overflow");

  Expect(StageHessianTransformScratchBytes(
             1, 1, "scalar terminal Hessian workspace") ==
             kSharedVectorAccessBytes,
         "scalar Hessian transform planner covers one complete shared-memory "
         "transaction");
  ScratchSize relation_matrix;
  relation_matrix.Add<Scalar>(7 * 11);
  ScratchSize elimination_tail;
  elimination_tail.Add<Scalar>(7);
  elimination_tail.Add<int>(7);
  elimination_tail.Add<int>(7);
  ScratchSize dynamics_tail;
  dynamics_tail.Add<Scalar>(5 * 3);
  dynamics_tail.Add<Scalar>(5);
  Expect(StageRelationReductionScratchBytes(
             7, 11, 5, 3, "nonuniform stage-reduction workspace") ==
             relation_matrix.bytes +
                 std::max(elimination_tail.bytes, dynamics_tail.bytes),
         "stage-reduction scratch exactly aliases nonoverlapping relation and "
         "elimination lifetimes");
  bool transform_overflow_rejected = false;
  try {
    (void)StageHessianTransformScratchBytes(
        std::numeric_limits<std::size_t>::max() / 2 + 1, 2,
        "overflowing Hessian workspace");
  } catch (const std::invalid_argument &) {
    transform_overflow_rejected = true;
  }
  Expect(transform_overflow_rejected,
         "Hessian-transform planner rejects staged-product size overflow");
  bool transform_padding_overflow_rejected = false;
  try {
    (void)StageHessianTransformScratchBytes(
        std::numeric_limits<std::size_t>::max(), 1,
        "overflowing padded Hessian workspace");
  } catch (const std::invalid_argument &) {
    transform_padding_overflow_rejected = true;
  }
  Expect(transform_padding_overflow_rejected,
         "Hessian-transform planner rejects transaction-padding overflow");

  const ScratchRequirements scalar =
      PlanScratch(clqr::benchmark::StateOnlyProblem(1, 1, 1, 0));
  Expect(scalar.terminal_reduction == kSharedVectorAccessBytes,
         "scalar terminal launch allocates one complete shared-memory "
         "transaction");

  const Problem uniform_problem = clqr::benchmark::StateOnlyProblem(8, n, 4, 2);
  const ScratchRequirements uniform = PlanScratch(uniform_problem);
  const ScanShape uniform_relation = MakeScanShape(n, n);
  const ScanShape terminal_relation = MakeScanShape(n, 0);
  ScratchSize value_leaf;
  value_leaf.Add<Scalar>(4 * 4);
  value_leaf.Add<Scalar>(4 * (2 * n));
  ScratchSize value_compose;
  value_compose.Add<Scalar>(n * (3 * n));
  value_compose.Add<Scalar>(n);
  value_compose.Add<Scalar>(n * n);
  ScratchSize feedback;
  feedback.Add<Scalar>(4 * (4 + n));
  feedback.Add<Scalar>(n * (4 + n));
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
  Expect(
      uniform.primal_leaf == DenseEliminationScratchBytes(
                                 10, 21, "uniform primal-leaf workspace") &&
          uniform.primal_relation_final ==
              RelationFinalizeScratchBytes(uniform_relation, &uniform_relation,
                                           terminal_relation) &&
          uniform.state_parameter == StateParameterScratchBytes(n, n) &&
          uniform.stage_reduction ==
              std::max(StageRelationReductionScratchBytes(
                           8, 13, n, n, "uniform stage-reduction workspace"),
                       StageHessianTransformScratchBytes(
                           n + 4, n + 4, "uniform stage Hessian workspace")) &&
          uniform.terminal_reduction ==
              StageHessianTransformScratchBytes(
                  n, n, "uniform terminal Hessian workspace"),
      "uniform n=8 reduction scratch exactly covers staged dynamics and "
      "Hessian products");
  Expect(uniform.value_compose == value_compose.bytes,
         "uniform n=8 value-composition scratch includes the exact staged "
         "matrix product");
  Expect(uniform.value_leaf == value_leaf.bytes &&
             uniform.value_finalize ==
                 ValueFinalizeScratchBytes(uniform_relation, &uniform_relation,
                                           terminal_relation) &&
             uniform.feedback == feedback.bytes &&
             uniform.affine_terms == n * sizeof(Scalar) &&
             uniform.affine_finalize ==
                 AffineFinalizeScratchBytes(uniform_relation, &uniform_relation,
                                            uniform_relation),
         "uniform n=8 Riccati/reconstruction launch scratch matches active "
         "dimensions");
  const ScratchRequirements odd =
      PlanScratch(clqr::benchmark::StateOnlyProblem(7, 5, 3, 1));
  Expect(odd.affine_terms == AffineTermsScratchBytes(5) &&
             odd.affine_terms >= 5 * sizeof(Scalar),
         "odd affine scratch dimensions include a complete shared-memory "
         "transaction");
  Expect(uniform.dual_relation ==
             DenseEliminationScratchBytes(4 * n, 3 * n + 1,
                                          "uniform dual workspace"),
         "uniform n=8 dual-relation launch scratch is unchanged");
  Expect(uniform.dual_parameter == dual_parameter.bytes &&
             uniform.dual_relation_leaf ==
                 DualRelationLeafScratchBytes(8 * 19, 8, 2, 18) &&
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
  Expect(plan_builds == 1 &&
             MaximumScratchBytes(cached_scratch) == MaximumScratchBytes(uniform),
         "same-shape workspace reuse builds the scratch plan only once");
  const Problem changed_problem =
      clqr::benchmark::StateOnlyProblem(8, n + 1, 4, 2);
  plan_builds +=
      RefreshScratchPlan(changed_problem, &cached_key, &cached_scratch);
  Expect(plan_builds == 2 &&
             MaximumScratchBytes(cached_scratch) ==
                 MaximumScratchBytes(PlanScratch(changed_problem)),
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
  problem.Q.resize(horizon + 1);
  problem.q.resize(horizon + 1);
  for (std::size_t i = 0; i < horizon; ++i) {
    Stage &stage = problem.stages[i];
    stage.A = GeneratedMatrix(n, n, 300 + static_cast<int>(i), 0.12);
    for (std::size_t row = 0; row < n; ++row)
      stage.A(row, row) += 0.8;
    stage.B = GeneratedMatrix(n, m, 400 + static_cast<int>(i), 0.22);
    stage.c = x[i + 1] - stage.A * x[i] - stage.B * u[i];
    problem.Q[i] = PositiveDefinite(n, 500 + static_cast<int>(i), 1.0);
    stage.R = PositiveDefinite(m, 600 + static_cast<int>(i), 1.4);
    stage.M = GeneratedMatrix(n, m, 700 + static_cast<int>(i), 0.025);
    problem.q[i] = GeneratedVector(n, 800 + static_cast<int>(i), 0.15);
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
  problem.Q.back() = PositiveDefinite(n, 1300, 1.5);
  problem.q.back() = GeneratedVector(n, 1400, 0.15);
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
  problem.Q.resize(horizon + 1);
  problem.q.resize(horizon + 1);
  for (std::size_t i = 0; i < horizon; ++i) {
    Stage &stage = problem.stages[i];
    stage.A = GeneratedMatrix(n, n, seed + 300 + static_cast<int>(i), 0.08);
    for (std::size_t row = 0; row < n; ++row)
      stage.A(row, row) += 0.9;
    stage.B = GeneratedMatrix(n, m, seed + 400 + static_cast<int>(i), 0.2);
    stage.c = x[i + 1] - stage.A * x[i] - stage.B * u[i];
    problem.Q[i] = PositiveDefinite(n, seed + 500 + static_cast<int>(i), 1.0);
    stage.R = PositiveDefinite(m, seed + 600 + static_cast<int>(i), 1.4);
    stage.M = GeneratedMatrix(n, m, seed + 700 + static_cast<int>(i), 0.02);
    problem.q[i] = GeneratedVector(n, seed + 800 + static_cast<int>(i), 0.15);
    stage.r = GeneratedVector(m, seed + 900 + static_cast<int>(i), 0.15);
    stage.C = Matrix(0, n);
    stage.D = Matrix(0, m);
    stage.d = Vector(0);
    stage.E = Matrix(0, n);
    stage.e = Vector(0);
  }
  problem.Q.back() = PositiveDefinite(n, seed + 1000, 1.5);
  problem.q.back() = GeneratedVector(n, seed + 1100, 0.15);
  problem.terminal_E = Matrix(0, n);
  problem.terminal_e = Vector(0);
  return problem;
}

enum class AffineSource { kStateCost, kControlCost, kDynamics };

Problem SingleAffineSourceProblem(AffineSource source) {
  Problem problem = UniformProblem(1550 + static_cast<int>(source), 7, 4, 3);
  for (std::size_t i = 0; i < problem.stages.size(); ++i) {
    Stage &stage = problem.stages[i];
    const Vector dynamics_offset = stage.c;
    const Vector state_gradient = problem.q[i];
    const Vector control_gradient = stage.r;
    stage.c = source == AffineSource::kDynamics ? dynamics_offset
                                                : Vector(stage.c.size());
    problem.q[i] = source == AffineSource::kStateCost
                       ? state_gradient
                       : Vector(problem.q[i].size());
    stage.r = source == AffineSource::kControlCost ? control_gradient
                                                   : Vector(stage.r.size());
  }
  if (source != AffineSource::kStateCost)
    problem.q.back() = Vector(problem.q.back().size());
  return problem;
}

Problem ZeroHorizonProblem() {
  Problem problem;
  problem.initial_state = Vector{0.4, -0.2, 0.7};
  problem.Q = {PositiveDefinite(3, 1600, 1.2)};
  problem.q = {GeneratedVector(3, 1610, 0.2)};
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
  problem.Q.resize(horizon + 1);
  problem.q.resize(horizon + 1);
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
    problem.Q[i] =
        PositiveDefinite(n, seed + 500 + static_cast<int>(i), Scalar{1});
    stage.R =
        PositiveDefinite(m, seed + 600 + static_cast<int>(i), Scalar{1.5});
    stage.M =
        GeneratedMatrix(n, m, seed + 700 + static_cast<int>(i), Scalar{0.01});
    problem.q[i] =
        GeneratedVector(n, seed + 800 + static_cast<int>(i), Scalar{0.05});
    stage.r =
        GeneratedVector(m, seed + 900 + static_cast<int>(i), Scalar{0.05});
    stage.C = Matrix(0, n);
    stage.D = Matrix(0, m);
    stage.d = Vector(0);
    stage.E = Matrix(0, n);
    stage.e = Vector(0);
  }
  problem.Q.back() =
      PositiveDefinite(dimensions.back(), seed + 1000, Scalar{1.5});
  problem.q.back() =
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

std::size_t PackedEntries(const Stage &source, const Matrix &Q,
                          const Vector &q) {
  const auto matrix_entries = [](const Matrix &matrix) {
    return matrix.rows() * matrix.cols();
  };
  return matrix_entries(source.A) + matrix_entries(source.B) + source.c.size() +
         matrix_entries(Q) + matrix_entries(source.R) +
         matrix_entries(source.M) + q.size() + source.r.size() +
         matrix_entries(source.C) + matrix_entries(source.D) + source.d.size() +
         matrix_entries(source.E) + source.e.size();
}

std::size_t PackedEntries(const Problem &problem) {
  std::size_t entries = problem.Q.back().rows() * problem.Q.back().cols() +
                        problem.q.back().size() +
                        problem.terminal_E.rows() * problem.terminal_E.cols() +
                        problem.terminal_e.size();
  for (std::size_t i = 0; i < problem.stages.size(); ++i)
    entries += PackedEntries(problem.stages[i], problem.Q[i], problem.q[i]);
  return entries;
}

PackedStage Pack(const Stage &source, const Matrix &Q, const Vector &q,
                 Scalar **cursor) {
  PackedStage out{};
  out.n = static_cast<int>(source.A.cols());
  out.next_n = static_cast<int>(source.A.rows());
  out.m = static_cast<int>(source.B.cols());
  out.mixed = static_cast<int>(source.C.rows());
  out.state = static_cast<int>(source.E.rows());
  PackMatrix(source.A, cursor, &out.A);
  PackMatrix(source.B, cursor, &out.B);
  PackVector(source.c, cursor, &out.c);
  PackMatrix(Q, cursor, &out.Q);
  PackMatrix(source.R, cursor, &out.R);
  PackMatrix(source.M, cursor, &out.M);
  PackVector(q, cursor, &out.q);
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
  out.n = static_cast<int>(problem.Q.back().rows());
  out.state = static_cast<int>(problem.terminal_E.rows());
  PackMatrix(problem.Q.back(), cursor, &out.Q);
  PackVector(problem.q.back(), cursor, &out.q);
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

bool g_test_global_scratch = false;
int g_test_first_block = 0;

template <typename Function> void LaunchScratch(int blocks, Function function) {
  if (!g_test_global_scratch) {
    Launch(blocks, function);
    return;
  }
  // Deliberately small and non-power-of-two: exercise offsets and partial
  // launches at every tree level, independently of the test horizon.
  for (int first = 0; first < blocks; first += 3) {
    g_test_first_block = first;
    Launch(std::min(3, blocks - first), function);
  }
  g_test_first_block = 0;
}

#define EMULATED_SCRATCH_KERNEL(kernel, ...)                                   \
  do {                                                                         \
    if (g_test_global_scratch)                                                 \
      kernel<true>(__VA_ARGS__, nullptr, 0, g_test_first_block);               \
    else                                                                       \
      kernel<false>(__VA_ARGS__);                                              \
  } while (false)

void FiniteInputValidationCase() {
  Scalar problem_data[]{Scalar{1}, std::numeric_limits<Scalar>::quiet_NaN()};
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
  LaunchScratch(2, [&] {
    EMULATED_SCRATCH_KERNEL(BuildValueElementsKernel, &stage, &terminal, 1,
                            kTolerance, elements.data(), &status);
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
      Scalar value = problem.q[i][row] + left[row];
      for (std::size_t col = 0; col < problem.Q[i].cols(); ++col)
        value += problem.Q[i](row, col) * x[col];
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
  for (std::size_t row = 0; row < problem.Q.back().rows(); ++row) {
    Scalar value = problem.q.back()[row] + left[row];
    for (std::size_t col = 0; col < problem.Q.back().cols(); ++col)
      value += problem.Q.back()(row, col) * terminal[col];
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
                  Scalar kkt_tolerance_scale = Scalar{1},
                  Scalar rank_tolerance = kTolerance) {
  const AllowedDeviceFailure *allowed_failure = AllowedFailureForCase(name);
  const int horizon = static_cast<int>(problem.stages.size());
  const int nodes = horizon + 1;
  std::vector<Scalar> packed_data(PackedEntries(problem));
  Scalar *packed_cursor = packed_data.data();
  std::vector<PackedStage> stages;
  for (std::size_t i = 0; i < problem.stages.size(); ++i)
    stages.push_back(
        Pack(problem.stages[i], problem.Q[i], problem.q[i], &packed_cursor));
  const PackedTerminal terminal = Pack(problem, &packed_cursor);
  Expect(packed_cursor == packed_data.data() + packed_data.size(),
         "compact problem packing uses the exact allocation");
  std::vector<Scalar> initial(kTestStateCapacity);
  for (std::size_t row = 0; row < problem.initial_state.size(); ++row)
    initial[row] = problem.initial_state[row];
  DeviceStatus status{};
  Scalar feasibility_consistency_tolerance =
      std::max(rank_tolerance, kMinimumFeasibilityConsistencyTolerance);

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
      rank_tolerance, kMinimumFeasibilityConsistencyTolerance *
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
  LaunchScratch(nodes, [&] {
    EMULATED_SCRATCH_KERNEL(BuildPrimalLeavesKernel, stages.data(), horizon,
                            &terminal, rank_tolerance,
                            feasibility_consistency_tolerance,
                            relation_a.data(), &status);
  });
  if (nodes > 1) {
    const int first_parent_count = node_level_counts[1];
    LaunchScratch(first_parent_count, [&] {
      EMULATED_SCRATCH_KERNEL(ReduceRelationLeavesKernel, relation_a.data(),
                              nodes, first_parent_count, rank_tolerance,
                              feasibility_consistency_tolerance,
                              relation_b.data(), &status);
    });
    for (std::size_t level = 1; level + 1 < node_level_counts.size(); ++level) {
      LaunchScratch(node_level_counts[level + 1], [&] {
        EMULATED_SCRATCH_KERNEL(
            ReduceRelationTreeLevelKernel, relation_b.data(),
            node_level_offsets[level] - nodes,
            node_level_offsets[level + 1] - nodes, node_level_counts[level],
            node_level_counts[level + 1], rank_tolerance,
            feasibility_consistency_tolerance, &status);
      });
    }
    Launch(1, [&] {
      InitializeRelationContextRootKernel(relation_b.data(),
                                          node_level_offsets.back() - nodes);
    });
    for (int level = static_cast<int>(node_level_counts.size()) - 2; level >= 1;
         --level) {
      LaunchScratch(node_level_counts[level + 1], [&] {
        EMULATED_SCRATCH_KERNEL(
            ExpandRelationContextLevelKernel, relation_b.data(),
            node_level_offsets[level] - nodes,
            node_level_offsets[level + 1] - nodes, node_level_counts[level],
            node_level_counts[level + 1], rank_tolerance,
            feasibility_consistency_tolerance, &status);
      });
    }
    LaunchScratch(first_parent_count, [&] {
      EMULATED_SCRATCH_KERNEL(FinalizeRelationSuffixFromParentsKernel,
                              relation_a.data(), nodes, relation_b.data(),
                              first_parent_count, rank_tolerance,
                              feasibility_consistency_tolerance, &status);
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
  LaunchScratch(nodes, [&] {
    EMULATED_SCRATCH_KERNEL(StateParamKernel, suffix, nodes,
                            state_params.data(), nullptr, &status,
                            rank_tolerance);
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
  LaunchScratch(horizon, [&] {
    EMULATED_SCRATCH_KERNEL(
        ReduceStagesKernel, stages.data(), suffix, state_params.data(), horizon,
        rank_tolerance, feasibility_consistency_tolerance,
        control_params.data(), reduced.data(), nullptr, &status);
  });
  LaunchScratch(1, [&] {
    EMULATED_SCRATCH_KERNEL(ReduceTerminalKernel, &terminal,
                            state_params.data(), horizon, &reduced_terminal);
  });
  Launch(1, [&] {
    InitialReducedStateKernel(state_params.data(), initial.data(),
                              reduced_initial.data(), rank_tolerance, &status);
  });
  if (FinishAllowedDeviceFailure(status, name, "independent reduction",
                                 allowed_failure))
    return;
  CheckReducedObjectiveIdentity(problem, state_params, control_params, reduced,
                                reduced_terminal, name);
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
  std::vector<Scalar> feedback_control_factor(
      static_cast<std::size_t>(horizon) * kTestControlCapacity *
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
    feedback[stage].control_factor =
        feedback_control_factor.data() +
        index * kTestControlCapacity * kTestControlCapacity;
    feedback[stage].transition =
        feedback_transition.data() +
        index * kTestStateCapacity * kTestStateCapacity;
    feedback[stage].offset =
        feedback_offset.data() + index * kTestStateCapacity;
  }
  LaunchScratch(nodes, [&] {
    EMULATED_SCRATCH_KERNEL(BuildValueElementsKernel, reduced.data(),
                            &reduced_terminal, horizon, rank_tolerance,
                            value_a.data(), &status);
  });
  if (FinishAllowedDeviceFailure(status, name, "value base", allowed_failure))
    return;
  ValueElement *value_suffix = value_a.data();
  g_value_matrix_combinations = 0;
  g_value_matrix_factorizations = 0;
  if (nodes > 1) {
    const int first_parent_count = node_level_counts[1];
    LaunchScratch(first_parent_count, [&] {
      EMULATED_SCRATCH_KERNEL(ReduceValueLeavesKernel, value_a.data(), nodes,
                              first_parent_count, rank_tolerance, &status,
                              value_b.data());
    });
    for (std::size_t level = 1; level + 1 < node_level_counts.size(); ++level) {
      LaunchScratch(node_level_counts[level + 1], [&] {
        EMULATED_SCRATCH_KERNEL(
            ReduceValueTreeLevelKernel, value_b.data(),
            node_level_offsets[level] - nodes,
            node_level_offsets[level + 1] - nodes, node_level_counts[level],
            node_level_counts[level + 1], rank_tolerance, &status);
      });
    }
    Launch(1, [&] {
      InitializeValueContextRootKernel(value_b.data(),
                                       node_level_offsets.back() - nodes);
    });
    for (int level = static_cast<int>(node_level_counts.size()) - 2; level >= 1;
         --level) {
      LaunchScratch(node_level_counts[level + 1], [&] {
        EMULATED_SCRATCH_KERNEL(
            ExpandValueContextLevelKernel, value_b.data(),
            node_level_offsets[level] - nodes,
            node_level_offsets[level + 1] - nodes, node_level_counts[level],
            node_level_counts[level + 1], rank_tolerance, &status);
      });
    }
    LaunchScratch(first_parent_count, [&] {
      EMULATED_SCRATCH_KERNEL(FinalizeValueSuffixFromParentsKernel,
                              value_a.data(), nodes, value_b.data(),
                              first_parent_count, rank_tolerance, &status);
    });
  }
  if (FinishAllowedDeviceFailure(status, name, "value scan", allowed_failure))
    return;
  Expect(g_value_matrix_factorizations == g_value_matrix_combinations,
         name + " uses exactly one LU factorization per matrix composition");
  LaunchScratch(horizon, [&] {
    EMULATED_SCRATCH_KERNEL(MatrixFeedbackKernel, reduced.data(), value_suffix,
                            horizon, rank_tolerance, feedback.data(), &status);
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
  const auto run_affine_prefix_scan = [&] {
    if (horizon <= 1)
      return;
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
    LaunchScratch(first_parent_count, [&] {
      EMULATED_SCRATCH_KERNEL(FinalizeAffinePrefixFromParentsKernel,
                              map_a.data(), horizon, map_b.data(),
                              first_parent_count, &status);
    });
  };

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

  std::vector<Scalar> reduced_value_linear(reduced_state_offsets.back());
  LaunchScratch(horizon, [&] {
    EMULATED_SCRATCH_KERNEL(InitializeCostateMapsKernel, reduced.data(),
                            value_suffix, feedback.data(), horizon,
                            map_a.data(), &status);
  });
  run_affine_prefix_scan();
  Launch(nodes, [&] {
    RecoverCostatesKernel(map_a.data(), &reduced_terminal,
                          reduced_state_offsets.data(), horizon,
                          reduced_value_linear.data(), &status);
  });
  LaunchScratch(horizon, [&] {
    EMULATED_SCRATCH_KERNEL(FinalizeFeedbackKernel, reduced.data(),
                            value_suffix, reduced_value_linear.data(),
                            reduced_state_offsets.data(), horizon,
                            feedback.data(), &status);
  });
  if (FinishAllowedDeviceFailure(status, name, "affine Riccati recovery",
                                 allowed_failure))
    return;

  Launch(horizon, [&] {
    InitializeAffineMapsKernel(feedback.data(), horizon, map_a.data(), &status);
  });
  AffineMap *prefix = map_a.data();
  run_affine_prefix_scan();
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
  const Scalar multiplier_leaf_consistency_tolerance = std::max(
      multiplier_rank_tolerance, kMultiplierConsistencyTolerancePerTreeLevel);
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
    LaunchScratch(horizon, [&] {
      EMULATED_SCRATCH_KERNEL(
          BuildDualParametersKernel, stages.data(), state_params.data(),
          value_suffix, reduced_value_linear.data(), reduced_states.data(),
          states.data(), controls.data(), reduced_state_offsets.data(),
          state_offsets.data(), control_offsets.data(), horizon,
          multiplier_rank_tolerance, multiplier_consistency_tolerance,
          dual_params.data(), &dual_scan_needed, nullptr, &status);
    });
    LaunchScratch(horizon, [&] {
      EMULATED_SCRATCH_KERNEL(
          BuildDualParameterRelationsKernel, stages.data(), &terminal,
          dual_params.data(), horizon, states.data(), controls.data(),
          state_offsets.data(), control_offsets.data(),
          multiplier_rank_tolerance, multiplier_leaf_consistency_tolerance,
          dual_tree.data(), &dual_scan_needed, state_dual_params.data(),
          &status);
    });
    for (std::size_t level = 0; level + 1 < stage_level_counts.size();
         ++level) {
      LaunchScratch(stage_level_counts[level + 1], [&] {
        EMULATED_SCRATCH_KERNEL(
            ReduceDualTreeLevelKernel, dual_tree.data(),
            stage_level_offsets[level], stage_level_offsets[level + 1],
            stage_level_counts[level], stage_level_counts[level + 1],
            multiplier_rank_tolerance, multiplier_consistency_tolerance,
            dual_tree.data(), &dual_scan_needed, &status);
      });
    }
    const int root = stage_level_offsets.back();
    LaunchScratch(1, [&] {
      EMULATED_SCRATCH_KERNEL(SolveDualRootKernel, dual_tree.data() + root,
                              dual_values.data() + root, &dual_scan_needed,
                              &status, multiplier_rank_tolerance);
    });
    for (int level = static_cast<int>(stage_level_counts.size()) - 2;
         level >= 0; --level) {
      LaunchScratch(stage_level_counts[level + 1], [&] {
        EMULATED_SCRATCH_KERNEL(
            ExpandDualTreeLevelKernel, dual_tree.data(),
            stage_level_offsets[level], stage_level_offsets[level + 1],
            stage_level_counts[level], stage_level_counts[level + 1],
            multiplier_rank_tolerance, multiplier_consistency_tolerance,
            dual_values.data(), dual_values.data(), &dual_scan_needed, &status);
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
  std::cout << name << " CUDA kernel emulation " << validation
            << "; KKT residual=" << residual << '\n';
}

void CoordinatePivotingCase() {
  Scalar matrix[] = {Scalar{1e-4}, Scalar{1},  Scalar{0},    Scalar{0.2},
                     Scalar{0.3},  Scalar{0},  Scalar{1e-4}, Scalar{1},
                     Scalar{0.1},  Scalar{0.4}};
  Scalar factors[2];
  int pivots[2], pivot_rows[2], rank = 0, best = 0;
  Launch(1, [&] {
    RrefBlock(matrix, 2, 5, 4, kTolerance, pivots, pivot_rows, &rank, &best,
              factors, Scalar{0}, 3);
  });
  Expect(rank == 2 && pivots[0] == 1 && pivots[1] == 2,
         "control parameterization chooses a well-scaled pivot subset");
  for (const Scalar value : matrix)
    Expect(std::abs(value) <= Scalar{1},
           "control coordinate coefficients stay bounded");

  Scalar left[] = {Scalar{1}, Scalar{1024}, Scalar{0}};
  Scalar rhs[] = {Scalar{-256}};
  Relation relation{};
  relation.left_dim = 3;
  relation.right_dim = 0;
  relation.rows = 1;
  relation.left = left;
  relation.rhs = rhs;
  Scalar T[9], t[3];
  int free_columns[3];
  StateParam param{};
  param.T = T;
  param.t = t;
  param.free_columns = free_columns;
  DeviceStatus status{};
  LaunchScratch(1, [&] {
    EMULATED_SCRATCH_KERNEL(StateParamKernel, &relation, 1, &param, nullptr,
                            &status, kTolerance);
  });
  Expect(status.code == kDeviceOk && param.reduced_dim == 2 &&
             free_columns[0] == 0 && free_columns[1] == 2,
         "state parameterization repivots an ill-scaled canonical relation");
  Expect(g_emulated_block_scratch_bytes == StateParameterScratchBytes(3, 1),
         "state parameterization scratch matches the actual typed layout");
  for (int col = 0; col < 2; ++col) {
    Scalar residual = 0;
    for (int row = 0; row < 3; ++row) {
      Expect(std::abs(T[row * 2 + col]) <= Scalar{1},
             "state basis stays bounded");
      residual += left[row] * T[row * 2 + col];
    }
    Expect(std::abs(residual) < kTolerance,
           "repivoted state basis spans the same nullspace");
  }
  Scalar offset_residual = -rhs[0];
  for (int row = 0; row < 3; ++row)
    offset_residual += left[row] * t[row];
  Expect(std::abs(offset_residual) < kTolerance,
         "repivoted affine offset satisfies the original equation");

  relation.rows = 0;
  int dimensions[2] = {-1, -1};
  LaunchScratch(1, [&] {
    EMULATED_SCRATCH_KERNEL(StateParamKernel, &relation, 1, &param, dimensions,
                            &status, kTolerance);
  });
  Expect(status.code == kDeviceOk && dimensions[0] == 3 && dimensions[1] == 3,
         "unconstrained state dimensions are unchanged");
  for (int row = 0; row < 3; ++row) {
    Expect(t[row] == Scalar{0} && free_columns[row] == row,
           "unconstrained state coordinates have zero offset");
    for (int col = 0; col < 3; ++col)
      Expect(T[row * 3 + col] == (row == col ? Scalar{1} : Scalar{0}),
             "unconstrained state coordinates are identity");
  }
}

bool FitsAdversarialEmulationStorage(const Problem &problem) {
  if (problem.Q.back().rows() > static_cast<std::size_t>(kTestStateCapacity) ||
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
#ifdef CLQR_EMULATION_RANK_REGRESSION
  if (argc == 2 && std::string(argv[1]) == "--rank-regression") {
    const auto data = clqr::benchmark::MakeScalingProblem(2048, 32, 16, 4, 8);
    // Fixed-column elimination lost a state constraint at stage 734 for
    // 1e-10, despite succeeding at 1e-9. Check the entire solve, including
    // original-coordinate feasibility and multiplier stationarity.
    for (const Scalar tolerance : {1e-12, 1e-11, 1e-10, 1e-9, 1e-8}) {
      std::cout << "rank tolerance=" << tolerance << '\n';
      RunEmulation(data.problem, "paper-N2048-n32", false, false, true,
                   Scalar{1e-8} / kLongHorizonKktComparisonTolerance,
                   tolerance);
    }
    return 0;
  }
#endif
  bool extended = false;
  bool paper = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--extended") {
      extended = true;
    } else if (std::string(argv[i]) == "--paper") {
      paper = true;
    } else {
      std::cerr << "unknown argument: " << argv[i] << '\n';
      return 2;
    }
  }
  if (paper || extended) {
#ifdef CLQR_USE_FLOAT
    if (paper) {
      std::cerr << "paper fixture emulation requires FP64\n";
      return 2;
    }
#else
    std::size_t executed = 0;
    for (const auto &test : clqr::benchmark::PaperCases("all")) {
      // Keep host-emulation storage bounded; this is not a GPU capacity claim.
      if (test.horizon > 512 || test.n > kTestStateCapacity)
        continue;
      const auto data = clqr::benchmark::MakeScalingProblem(
          test.horizon, test.n, test.m, test.mixed, test.state);
      Expect(FitsAdversarialEmulationStorage(data.problem),
             "paper case fits the emulation backing storage");
      const Scalar base_tolerance = test.horizon >= 256
                                        ? kLongHorizonKktComparisonTolerance
                                        : kKktComparisonTolerance;
      RunEmulation(
          data.problem,
          "paper-" + test.family + "-N" + std::to_string(test.horizon) + "-n" +
              std::to_string(test.n) + "-pm" + std::to_string(test.mixed) +
              "-ps" + std::to_string(test.state),
          false, false, true, Scalar{1e-8} / base_tolerance);
      ++executed;
    }
    std::cout << "all " << executed << " paper CUDA emulation cases passed\n";
    if (paper)
      return 0;
#endif
  }
  TinyCoefficientRrefCase();
  CoordinatePivotingCase();
  FiniteInputValidationCase();
  DeviceObjectiveCase();
  PivotedLuMultiRhsCase();
  OrthogonalEchelonCase();
  DualResidualOrthogonalEchelonCase();
  IllConditionedPositiveDefiniteMultiRhsCase();
  InvalidValueElementCopyCase();
  FreeFixedFreeValueCompositionCase();
  NonuniformValueCompositionCase();
  StagedFeedbackSystemCase();
  TerminalReductionScratchPaddingCase();
  DualRelationLeafScratchSizeCase();
  ScratchPlannerTopologyCase();
  NonPositiveDefiniteReducedControlCostCase();
  RunEmulation(MakeProblem(), "rank-deficient constrained", true, true);
  g_test_global_scratch = true;
  RunEmulation(MakeProblem(), "bounded-global-rank-deficient", true, true);
  RunEmulation(HeterogeneousDimensionProblem(), "bounded-global-heterogeneous",
               false, false);
  RunEmulation(ZeroHorizonProblem(), "bounded-global-zero-horizon", false,
               false);
  RunEmulation(ExactDualRelationScratchProblem(), "bounded-global-dual-scan",
               true, false);
  g_test_global_scratch = false;
  RunEmulation(SingleAffineSourceProblem(AffineSource::kStateCost),
               "state-gradient-only", false, false);
  RunEmulation(SingleAffineSourceProblem(AffineSource::kControlCost),
               "control-gradient-only", false, false);
  RunEmulation(SingleAffineSourceProblem(AffineSource::kDynamics),
               "dynamics-offset-only", false, false);
  RunEmulation(ZeroHorizonProblem(), "zero-horizon", false, false);
  RunEmulation(
      UniformProblem(1800, 3, kTestStateCapacity, kTestControlCapacity),
      "maximum-active-dimension", false, false);
  RunEmulation(clqr::benchmark::MakeScalingProblem(8, 24, 12, 3, 6).problem,
               "opt-in-shared-memory-fixture", true, true);
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
                 true,
                 test_case.tolerance_scale * test_case.kkt_tolerance_scale);
    ++executed;
  }
  std::cout << "all " << executed
            << " selected shared adversarial emulation cases passed\n";
  return 0;
}
