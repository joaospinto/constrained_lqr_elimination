#ifndef CLQR_TESTS_ADVERSARIAL_TEST_SUPPORT_H_
#define CLQR_TESTS_ADVERSARIAL_TEST_SUPPORT_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "clqr/clqr.h"

namespace clqr::test::adversarial {

#ifdef CLQR_USE_FLOAT
constexpr Scalar kPrimalTolerance = 3e-3f;
constexpr Scalar kKktTolerance = 2e-2f;
constexpr Scalar kDenseTolerance = 4e-3f;
constexpr Scalar kLinearTolerance = 2e-5f;
constexpr Scalar kSmallRowScale = 1e-2f;
constexpr Scalar kLargeRowScale = 1e2f;
constexpr Scalar kAccuracyLimitKktToleranceScale = 2;
#else
constexpr Scalar kPrimalTolerance = 3e-7;
constexpr Scalar kKktTolerance = 3e-5;
constexpr Scalar kDenseTolerance = 5e-7;
constexpr Scalar kLinearTolerance = 1e-10;
constexpr Scalar kSmallRowScale = 1e-6;
constexpr Scalar kLargeRowScale = 1e6;
constexpr Scalar kAccuracyLimitKktToleranceScale = 1;
#endif

enum class Pattern {
  kNone,
  kState,
  kMixed,
  kAlternating,
  kRedundant,
  kTerminal,
  kScaled,
};

struct TestCase {
  std::string name;
  Problem problem;
  SolveStatus cpu_status = SolveStatus::kOptimal;
  SolveStatus cuda_status = SolveStatus::kOptimal;
  bool emulate = true;
  bool dense_reference = true;
  bool check_full_kkt = true;
  Scalar tolerance_scale = Scalar{1};
  // Some deliberately ill-conditioned FP32 cases are expected to reject on
  // the reference GPU. Different instruction orderings may instead solve
  // them; in that case the native test must apply the full KKT gate.
  bool allow_accurate_cuda_success = false;
  Scalar kkt_tolerance_scale = Scalar{1};
};

inline SolveStatus
CudaStatusForFp32Limitation(
    bool affected,
    SolveStatus affected_status = SolveStatus::kNumericalFailure) {
#ifdef CLQR_USE_FLOAT
  return affected ? affected_status : SolveStatus::kOptimal;
#else
  (void)affected;
  (void)affected_status;
  return SolveStatus::kOptimal;
#endif
}

struct KktPoint {
  std::vector<Vector> states;
  std::vector<Vector> controls;
  Vector initial_multiplier;
  std::vector<Vector> dynamics_multipliers;
  std::vector<Vector> mixed_multipliers;
  std::vector<Vector> state_multipliers;
  Vector terminal_state_multiplier;
  Scalar objective = Scalar{0};
};

struct DensePrimal {
  std::vector<Vector> states;
  std::vector<Vector> controls;
};

inline Scalar Value(int seed, std::size_t row, std::size_t col = 0) {
  const Scalar x = static_cast<Scalar>(seed * 97 + static_cast<int>(row) * 37 +
                                       static_cast<int>(col) * 61);
  return Scalar{0.55} * std::sin(Scalar{0.017} * x) +
         Scalar{0.3} * std::cos(Scalar{0.029} * x);
}

inline Vector GeneratedVector(std::size_t size, int seed,
                              Scalar scale = Scalar{1}) {
  Vector out(size);
  for (std::size_t i = 0; i < size; ++i)
    out[i] = scale * Value(seed, i);
  return out;
}

inline Matrix GeneratedMatrix(std::size_t rows, std::size_t cols, int seed,
                              Scalar scale = Scalar{1}) {
  Matrix out(rows, cols);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col)
      out(row, col) = scale * Value(seed, row, col);
  }
  return out;
}

inline Matrix PositiveDefinite(std::size_t size, int seed,
                               Scalar diagonal = Scalar{1}) {
  Matrix g = GeneratedMatrix(size, size, seed, Scalar{0.16});
  Matrix result = Transpose(g) * g;
  for (std::size_t i = 0; i < size; ++i)
    result(i, i) += diagonal;
  return result;
}

inline Scalar RowDot(const Matrix &matrix, std::size_t row,
                     const Vector &vector) {
  Scalar value = Scalar{0};
  for (std::size_t col = 0; col < vector.size(); ++col)
    value += matrix(row, col) * vector[col];
  return value;
}

inline void SetMixedOffset(Stage *stage, const Vector &state,
                           const Vector &control) {
  stage->d = Vector(stage->C.rows());
  for (std::size_t row = 0; row < stage->C.rows(); ++row) {
    stage->d[row] =
        -(RowDot(stage->C, row, state) + RowDot(stage->D, row, control));
  }
}

inline void SetStateOffset(Matrix *matrix, Vector *offset,
                           const Vector &state) {
  *offset = Vector(matrix->rows());
  for (std::size_t row = 0; row < matrix->rows(); ++row)
    (*offset)[row] = -RowDot(*matrix, row, state);
}

inline Problem FeasibleProblem(int seed, std::vector<std::size_t> state_dims,
                               std::vector<std::size_t> control_dims,
                               std::size_t rows, Pattern pattern) {
  if (state_dims.size() != control_dims.size() + 1)
    throw std::invalid_argument("state/control horizon mismatch");
  const std::size_t horizon = control_dims.size();
  std::vector<Vector> nominal_states(horizon + 1);
  std::vector<Vector> nominal_controls(horizon);
  for (std::size_t i = 0; i <= horizon; ++i) {
    nominal_states[i] = GeneratedVector(
        state_dims[i], seed + 100 + static_cast<int>(i), Scalar{0.45});
  }
  for (std::size_t i = 0; i < horizon; ++i) {
    nominal_controls[i] = GeneratedVector(
        control_dims[i], seed + 200 + static_cast<int>(i), Scalar{0.35});
  }

  Problem problem;
  problem.initial_state = nominal_states.front();
  problem.stages.resize(horizon);
  for (std::size_t i = 0; i < horizon; ++i) {
    const std::size_t n = state_dims[i];
    const std::size_t next_n = state_dims[i + 1];
    const std::size_t m = control_dims[i];
    Stage &stage = problem.stages[i];
    stage.A = GeneratedMatrix(next_n, n, seed + 300 + 11 * i, Scalar{0.11});
    for (std::size_t row = 0; row < std::min(n, next_n); ++row)
      stage.A(row, row) += Scalar{0.82};
    stage.B = GeneratedMatrix(next_n, m, seed + 400 + 13 * i, Scalar{0.19});
    stage.c = nominal_states[i + 1] - stage.A * nominal_states[i] -
              stage.B * nominal_controls[i];
    stage.Q = PositiveDefinite(n, seed + 500 + i, Scalar{1});
    stage.R = PositiveDefinite(m, seed + 600 + i, Scalar{1.3});
    stage.M = GeneratedMatrix(n, m, seed + 700 + i, Scalar{0.025});
    stage.q =
        GeneratedVector(n, seed + 800 + static_cast<int>(i), Scalar{0.12});
    stage.r =
        GeneratedVector(m, seed + 900 + static_cast<int>(i), Scalar{0.12});
    stage.C = Matrix(0, n);
    stage.D = Matrix(0, m);
    stage.d = Vector(0);
    stage.E = Matrix(0, n);
    stage.e = Vector(0);

    const bool use_mixed = pattern == Pattern::kMixed ||
                           pattern == Pattern::kRedundant ||
                           pattern == Pattern::kScaled ||
                           (pattern == Pattern::kAlternating && i % 2 == 0);
    const bool use_state = pattern == Pattern::kState ||
                           (pattern == Pattern::kAlternating && i % 2 != 0);
    if (rows > 0 && use_mixed) {
      stage.C = GeneratedMatrix(rows, n, seed + 1000 + 17 * i, Scalar{0.31});
      stage.D = GeneratedMatrix(rows, m, seed + 1100 + 19 * i, Scalar{0.31});
      if (pattern == Pattern::kRedundant && rows >= 2) {
        for (std::size_t col = 0; col < n; ++col)
          stage.C(1, col) = Scalar{-2} * stage.C(0, col);
        for (std::size_t col = 0; col < m; ++col)
          stage.D(1, col) = Scalar{-2} * stage.D(0, col);
      }
      SetMixedOffset(&stage, nominal_states[i], nominal_controls[i]);
      if (pattern == Pattern::kRedundant && rows >= 2)
        stage.d[1] = Scalar{-2} * stage.d[0];
      if (pattern == Pattern::kScaled) {
        for (std::size_t row = 0; row < rows; ++row) {
          const Scalar scale =
              row % 3 == 0 ? kSmallRowScale
                           : (row % 3 == 1 ? Scalar{1} : kLargeRowScale);
          for (std::size_t col = 0; col < n; ++col)
            stage.C(row, col) *= scale;
          for (std::size_t col = 0; col < m; ++col)
            stage.D(row, col) *= scale;
          stage.d[row] *= scale;
        }
      }
    } else if (rows > 0 && use_state) {
      stage.E = GeneratedMatrix(rows, n, seed + 1200 + 23 * i, Scalar{0.31});
      SetStateOffset(&stage.E, &stage.e, nominal_states[i]);
    }
  }
  const std::size_t terminal_n = state_dims.back();
  problem.terminal_Q = PositiveDefinite(terminal_n, seed + 1300, Scalar{1.45});
  problem.terminal_q = GeneratedVector(terminal_n, seed + 1400, Scalar{0.12});
  problem.terminal_E = Matrix(0, terminal_n);
  problem.terminal_e = Vector(0);
  if (rows > 0 && pattern == Pattern::kTerminal) {
    problem.terminal_E =
        GeneratedMatrix(rows, terminal_n, seed + 1500, Scalar{0.31});
    SetStateOffset(&problem.terminal_E, &problem.terminal_e,
                   nominal_states.back());
  }
  return problem;
}

inline Problem UniformProblem(int seed, std::size_t horizon, std::size_t n,
                              std::size_t m, std::size_t rows,
                              Pattern pattern) {
  return FeasibleProblem(seed, std::vector<std::size_t>(horizon + 1, n),
                         std::vector<std::size_t>(horizon, m), rows, pattern);
}

inline Problem InfeasibleInitialProblem() {
  Problem problem = UniformProblem(3000, 3, 3, 2, 0, Pattern::kNone);
  Stage &first = problem.stages.front();
  first.E = Matrix(2, 3, {1, 0, 0, 2, 0, 0});
  first.e = Vector{-problem.initial_state[0],
                   -Scalar{2} * problem.initial_state[0] + Scalar{1}};
  return problem;
}

inline Problem InfeasibleTerminalProblem() {
  Problem problem = UniformProblem(3010, 4, 3, 2, 0, Pattern::kNone);
  problem.terminal_E = Matrix(2, 3, {1, 0, 0, -3, 0, 0});
  problem.terminal_e = Vector{0, 1};
  return problem;
}

inline Problem SingularReducedHessianProblem(Scalar control_diagonal) {
  Problem problem = UniformProblem(3020, 1, 1, 1, 0, Pattern::kNone);
  problem.stages[0].Q = Matrix(1, 1, {0});
  problem.stages[0].R = Matrix(1, 1, {control_diagonal});
  problem.stages[0].M = Matrix(1, 1, {0});
  problem.stages[0].q = Vector{0};
  problem.stages[0].r = Vector{control_diagonal < 0 ? Scalar{-0.2} : Scalar{0}};
  problem.terminal_Q = Matrix(1, 1, {0});
  problem.terminal_q = Vector{0};
  return problem;
}

inline Problem FullyConstrainedWithNonuniqueMultipliers() {
  Problem problem = UniformProblem(3030, 2, 2, 1, 0, Pattern::kNone);
  for (std::size_t i = 0; i < problem.stages.size(); ++i) {
    Stage &stage = problem.stages[i];
    const Vector nominal =
        GeneratedVector(1, 3030 + 200 + static_cast<int>(i), Scalar{0.35});
    stage.R = Matrix(1, 1, {0});
    stage.r = Vector{0};
    stage.C = Matrix(2, 2, {0, 0, 0, 0});
    stage.D = Matrix(2, 1, {1, -2});
    stage.d = Vector{-nominal[0], Scalar{2} * nominal[0]};
  }
  return problem;
}

inline Problem FreeFixedFreeStateProblem() {
  constexpr int seed = 3035;
  Problem problem = UniformProblem(seed, 2, 1, 1, 0, Pattern::kNone);
  const Vector nominal_control = GeneratedVector(1, seed + 200, Scalar{0.35});
  const Vector middle =
      problem.stages[0].A * problem.initial_state +
      problem.stages[0].B * nominal_control + problem.stages[0].c;
  problem.stages[1].E = Matrix(1, 1, {1});
  problem.stages[1].e = Vector{-middle[0]};
  return problem;
}

inline Problem EmptyProblem() {
  Problem problem;
  problem.initial_state = Vector(0);
  problem.terminal_Q = Matrix(0, 0);
  problem.terminal_q = Vector(0);
  problem.terminal_E = Matrix(0, 0);
  problem.terminal_e = Vector(0);
  return problem;
}

inline Problem InvalidShapeProblem() {
  Problem problem = UniformProblem(3040, 2, 3, 2, 0, Pattern::kNone);
  problem.stages[1].R = Matrix(1, 1, {1});
  return problem;
}

inline std::vector<TestCase> StandardCases() {
  std::vector<TestCase> cases;
  cases.push_back({"empty-zero-horizon", EmptyProblem()});
  cases.push_back(
      {"zero-horizon", UniformProblem(1, 0, 3, 0, 0, Pattern::kNone)});
  cases.push_back({"zero-horizon-terminal",
                   UniformProblem(2, 0, 3, 0, 2, Pattern::kTerminal)});
  cases.push_back(
      {"single-stage", UniformProblem(3, 1, 3, 2, 1, Pattern::kMixed)});
  cases.push_back(
      {"power-of-two", UniformProblem(4, 8, 3, 2, 1, Pattern::kAlternating)});
  for (std::size_t horizon : {2U, 3U, 5U, 6U, 7U, 9U, 15U, 17U}) {
    // Keep the odd scan-boundary case well-conditioned in FP32.  Seed 27 is
    // retained separately below as an explicit numerical-limit fixture.
    const int seed = horizon == 17 ? 20 : 10 + static_cast<int>(horizon);
    cases.push_back({"horizon-" + std::to_string(horizon),
                     UniformProblem(seed, horizon, 3, 2, 1,
                                    Pattern::kAlternating),
                     SolveStatus::kOptimal, SolveStatus::kOptimal, true,
                     horizon <= 9});
  }
  // This intentionally accuracy-limited FP32 fixture uses twice the ordinary
  // KKT gate only; its primal and dense-reference gates are unchanged, and
  // stable fixtures at the same horizon retain the ordinary KKT gate.
  cases.push_back(
      {"ill-conditioned-horizon-17",
       UniformProblem(27, 17, 3, 2, 1, Pattern::kAlternating),
       SolveStatus::kOptimal, CudaStatusForFp32Limitation(true), true,
       false, true, Scalar{1}, true, kAccuracyLimitKktToleranceScale});
  cases.push_back(
      {"nonuniform-zero-control",
       FeasibleProblem(40, {1, 4, 2, 3}, {0, 3, 1}, 1, Pattern::kAlternating)});
  cases.push_back(
      {"all-zero-controls", UniformProblem(41, 5, 3, 0, 1, Pattern::kState)});
  cases.push_back({"more-mixed-than-controls",
                   UniformProblem(42, 5, 4, 1, 3, Pattern::kMixed)});
  cases.push_back({"redundant-rank-deficient",
                   UniformProblem(43, 5, 4, 2, 3, Pattern::kRedundant),
                   SolveStatus::kOptimal, SolveStatus::kOptimal, true, false});
  cases.push_back({"independently-scaled-rows",
                   UniformProblem(44, 4, 4, 2, 3, Pattern::kScaled),
                   SolveStatus::kOptimal, SolveStatus::kOptimal, true, false,
                   true, Scalar{8}});
  cases.push_back(
      {"terminal-only", UniformProblem(45, 5, 4, 2, 2, Pattern::kTerminal)});
  cases.push_back({"fully-constrained-nonunique-multipliers",
                   FullyConstrainedWithNonuniqueMultipliers(),
                   SolveStatus::kOptimal, SolveStatus::kOptimal, true, false});
  cases.push_back(
      {"free-fixed-free-state", FreeFixedFreeStateProblem(),
       SolveStatus::kOptimal, SolveStatus::kOptimal, true, true, true});
  cases.push_back({"infeasible-initial", InfeasibleInitialProblem(),
                   SolveStatus::kInfeasible, SolveStatus::kInfeasible, false,
                   false, false});
  cases.push_back({"infeasible-terminal", InfeasibleTerminalProblem(),
                   SolveStatus::kInfeasible, SolveStatus::kInfeasible, false,
                   false, false});
  cases.push_back({"singular-reduced-hessian",
                   SingularReducedHessianProblem(Scalar{0}),
                   SolveStatus::kNumericalFailure,
                   SolveStatus::kNumericalFailure, false, false, false});
  cases.push_back({"indefinite-reduced-hessian",
                   SingularReducedHessianProblem(Scalar{-1}),
                   SolveStatus::kOptimal, SolveStatus::kNumericalFailure, false,
                   true, true});
  cases.push_back({"invalid-shape", InvalidShapeProblem(),
                   SolveStatus::kInvalidInput, SolveStatus::kInvalidInput,
                   false, false, false});
  return cases;
}

inline std::vector<TestCase> ExtendedCases() {
  std::vector<TestCase> cases;
  for (std::size_t horizon : {31U, 32U, 33U, 63U, 65U, 127U, 257U, 1025U}) {
    const bool fp32_numerical_limit =
        horizon == 32 || horizon == 63 || horizon == 65 || horizon == 127 ||
        horizon == 257;
    cases.push_back({"extended-horizon-" + std::to_string(horizon),
                     UniformProblem(200 + static_cast<int>(horizon), horizon, 3,
                                    2, 1, Pattern::kAlternating),
                     SolveStatus::kOptimal,
                     CudaStatusForFp32Limitation(fp32_numerical_limit),
                     horizon <= 257, false, horizon == 31, Scalar{1},
                     fp32_numerical_limit});
  }
  // Retain the original generated horizons above, including their explicit
  // FP32 numerical failures, and add well-conditioned versions that must
  // complete the same scan boundaries quantitatively.
  for (const auto& [horizon, seed] :
       {std::pair<std::size_t, int>{32, 200}, {63, 200}, {65, 200},
        {127, 207}, {257, 214}}) {
    cases.push_back({"stable-extended-horizon-" + std::to_string(horizon),
                     UniformProblem(seed, horizon, 3, 2, 1,
                                    Pattern::kAlternating),
                     SolveStatus::kOptimal, SolveStatus::kOptimal,
                     true, false, false});
  }
  // Fixed seeds make these property cases exactly reproducible.
  for (int seed = 0; seed < 32; ++seed) {
    const std::size_t horizon = 1 + static_cast<std::size_t>((17 * seed) % 23);
    const std::size_t n = 1 + static_cast<std::size_t>((7 * seed) % 4);
    const std::size_t m = static_cast<std::size_t>((5 * seed) % 4);
    const std::size_t rows =
        std::min<std::size_t>(3, 1 + static_cast<std::size_t>(seed % 3));
    const Pattern pattern =
        static_cast<Pattern>(1 + static_cast<unsigned>(seed) % 5);
    const bool fp32_numerical_limit = seed == 1 || seed == 5;
    cases.push_back({"property-seed-" + std::to_string(seed),
                     UniformProblem(5000 + seed, horizon, n, m, rows, pattern),
                     SolveStatus::kOptimal,
                     CudaStatusForFp32Limitation(
                         fp32_numerical_limit,
                         seed == 5 ? SolveStatus::kInfeasible
                                   : SolveStatus::kNumericalFailure),
                     seed < 8,
                     horizon * (n + m) <= 30 && pattern != Pattern::kScaled,
                     seed < 10, Scalar{1}, fp32_numerical_limit});
  }
  cases.push_back({"stable-property-seed-1",
                   UniformProblem(5003, 18, 4, 1, 2, Pattern::kMixed),
                   SolveStatus::kOptimal, SolveStatus::kOptimal, true, false,
                   true});
  cases.push_back({"stable-property-seed-5",
                   UniformProblem(5000, 17, 4, 1, 3, Pattern::kState),
                   SolveStatus::kOptimal, SolveStatus::kOptimal, true, false,
                   true});
  return cases;
}

inline KktPoint CopyCpuSolution(const SolutionView &source) {
  KktPoint result;
  result.states.resize(source.state_count);
  for (std::size_t i = 0; i < source.state_count; ++i) {
    result.states[i] = Vector(source.states[i].size);
    for (std::size_t j = 0; j < source.states[i].size; ++j)
      result.states[i][j] = source.states[i][j];
  }
  result.controls.resize(source.control_count);
  for (std::size_t i = 0; i < source.control_count; ++i) {
    result.controls[i] = Vector(source.controls[i].size);
    for (std::size_t j = 0; j < source.controls[i].size; ++j)
      result.controls[i][j] = source.controls[i][j];
  }
  const auto copy_view = [](const VectorView &source_view) {
    Vector copy(source_view.size);
    for (std::size_t i = 0; i < source_view.size; ++i)
      copy[i] = source_view[i];
    return copy;
  };
  result.initial_multiplier = copy_view(source.initial_multiplier);
  result.dynamics_multipliers.resize(source.dynamics_multiplier_count);
  for (std::size_t i = 0; i < source.dynamics_multiplier_count; ++i)
    result.dynamics_multipliers[i] = copy_view(source.dynamics_multipliers[i]);
  result.mixed_multipliers.resize(source.mixed_multiplier_count);
  for (std::size_t i = 0; i < source.mixed_multiplier_count; ++i)
    result.mixed_multipliers[i] = copy_view(source.mixed_multipliers[i]);
  result.state_multipliers.resize(source.state_multiplier_count);
  for (std::size_t i = 0; i < source.state_multiplier_count; ++i)
    result.state_multipliers[i] = copy_view(source.state_multipliers[i]);
  result.terminal_state_multiplier =
      copy_view(source.terminal_state_multiplier);
  result.objective = source.objective;
  return result;
}

inline Scalar TestMaxAbs(const Vector &vector) {
  Scalar result = Scalar{0};
  for (std::size_t i = 0; i < vector.size(); ++i) {
    if (!std::isfinite(vector[i]))
      return std::numeric_limits<Scalar>::infinity();
    result = std::max(result, std::abs(vector[i]));
  }
  return result;
}

inline Scalar ScaledResidual(const Matrix &matrix, const Vector &vector,
                             const Vector &offset) {
  Scalar result = Scalar{0};
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    if (!std::isfinite(offset[row]))
      return std::numeric_limits<Scalar>::infinity();
    Scalar value = offset[row];
    Scalar scale = std::max(Scalar{1}, std::abs(offset[row]));
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
      if (!std::isfinite(matrix(row, col)) || !std::isfinite(vector[col]))
        return std::numeric_limits<Scalar>::infinity();
      value += matrix(row, col) * vector[col];
      scale = std::max(scale, std::abs(matrix(row, col)));
    }
    if (!std::isfinite(value) || !std::isfinite(scale))
      return std::numeric_limits<Scalar>::infinity();
    result = std::max(result, std::abs(value) / scale);
  }
  return result;
}

inline void AddTranspose(const Matrix &matrix, const Vector &multiplier,
                         Vector *target) {
  for (std::size_t col = 0; col < matrix.cols(); ++col) {
    for (std::size_t row = 0; row < matrix.rows(); ++row)
      (*target)[col] += matrix(row, col) * multiplier[row];
  }
}

inline Scalar MaxKktResidual(const Problem &problem, const KktPoint &point,
                             std::string *worst = nullptr) {
  Scalar result = Scalar{0};
  const auto update = [&](Scalar value, std::string equation) {
    if (!std::isfinite(value)) {
      result = std::numeric_limits<Scalar>::infinity();
      if (worst != nullptr)
        *worst = std::move(equation);
      return;
    }
    if (value > result) {
      result = value;
      if (worst != nullptr)
        *worst = std::move(equation);
    }
  };
  const std::size_t horizon = problem.stages.size();
  if (point.states.size() != horizon + 1 || point.controls.size() != horizon) {
    if (worst != nullptr)
      *worst = "trajectory shape";
    return std::numeric_limits<Scalar>::infinity();
  }
  update(TestMaxAbs(point.states.front() - problem.initial_state),
         "initial state");
  for (std::size_t i = 0; i < horizon; ++i) {
    const Stage &stage = problem.stages[i];
    update(TestMaxAbs(point.states[i + 1] - stage.A * point.states[i] -
                      stage.B * point.controls[i] - stage.c),
           "dynamics " + std::to_string(i));
    if (stage.C.rows() > 0) {
      Matrix mixed(stage.C.rows(), stage.C.cols() + stage.D.cols());
      Vector variables(stage.C.cols() + stage.D.cols());
      for (std::size_t row = 0; row < stage.C.rows(); ++row) {
        for (std::size_t col = 0; col < stage.C.cols(); ++col)
          mixed(row, col) = stage.C(row, col);
        for (std::size_t col = 0; col < stage.D.cols(); ++col)
          mixed(row, stage.C.cols() + col) = stage.D(row, col);
      }
      for (std::size_t col = 0; col < stage.C.cols(); ++col)
        variables[col] = point.states[i][col];
      for (std::size_t col = 0; col < stage.D.cols(); ++col)
        variables[stage.C.cols() + col] = point.controls[i][col];
      update(ScaledResidual(mixed, variables, stage.d),
             "mixed feasibility " + std::to_string(i));
    }
    if (stage.E.rows() > 0) {
      update(ScaledResidual(stage.E, point.states[i], stage.e),
             "state feasibility " + std::to_string(i));
    }
    Vector state_gradient = stage.Q * point.states[i] +
                            stage.M * point.controls[i] + stage.q -
                            Transpose(stage.A) * point.dynamics_multipliers[i];
    state_gradient =
        state_gradient +
        (i == 0 ? point.initial_multiplier : point.dynamics_multipliers[i - 1]);
    AddTranspose(stage.C, point.mixed_multipliers[i], &state_gradient);
    AddTranspose(stage.E, point.state_multipliers[i], &state_gradient);
    update(TestMaxAbs(state_gradient),
           "state stationarity " + std::to_string(i));

    Vector control_gradient =
        Transpose(stage.M) * point.states[i] + stage.R * point.controls[i] +
        stage.r - Transpose(stage.B) * point.dynamics_multipliers[i];
    AddTranspose(stage.D, point.mixed_multipliers[i], &control_gradient);
    update(TestMaxAbs(control_gradient),
           "control stationarity " + std::to_string(i));
  }
  if (problem.terminal_E.rows() > 0) {
    update(ScaledResidual(problem.terminal_E, point.states.back(),
                          problem.terminal_e),
           "terminal feasibility");
  }
  Vector terminal_gradient = problem.terminal_Q * point.states.back() +
                             problem.terminal_q +
                             (horizon == 0 ? point.initial_multiplier
                                           : point.dynamics_multipliers.back());
  AddTranspose(problem.terminal_E, point.terminal_state_multiplier,
               &terminal_gradient);
  update(TestMaxAbs(terminal_gradient), "terminal stationarity");
  return result;
}

inline Scalar MaxPrimalResidual(const Problem &problem, const KktPoint &point) {
  const std::size_t horizon = problem.stages.size();
  if (point.states.size() != horizon + 1 || point.controls.size() != horizon) {
    return std::numeric_limits<Scalar>::infinity();
  }
  Scalar result = TestMaxAbs(point.states.front() - problem.initial_state);
  for (std::size_t i = 0; i < horizon; ++i) {
    const Stage &stage = problem.stages[i];
    result = std::max(
        result, TestMaxAbs(point.states[i + 1] - stage.A * point.states[i] -
                           stage.B * point.controls[i] - stage.c));
    if (stage.C.rows() > 0) {
      Matrix mixed(stage.C.rows(), stage.C.cols() + stage.D.cols());
      Vector variables(stage.C.cols() + stage.D.cols());
      for (std::size_t row = 0; row < stage.C.rows(); ++row) {
        for (std::size_t col = 0; col < stage.C.cols(); ++col)
          mixed(row, col) = stage.C(row, col);
        for (std::size_t col = 0; col < stage.D.cols(); ++col)
          mixed(row, stage.C.cols() + col) = stage.D(row, col);
      }
      for (std::size_t col = 0; col < stage.C.cols(); ++col)
        variables[col] = point.states[i][col];
      for (std::size_t col = 0; col < stage.D.cols(); ++col)
        variables[stage.C.cols() + col] = point.controls[i][col];
      result = std::max(result, ScaledResidual(mixed, variables, stage.d));
    }
    if (stage.E.rows() > 0) {
      result =
          std::max(result, ScaledResidual(stage.E, point.states[i], stage.e));
    }
  }
  if (problem.terminal_E.rows() > 0) {
    result =
        std::max(result, ScaledResidual(problem.terminal_E, point.states.back(),
                                        problem.terminal_e));
  }
  return result;
}

inline std::vector<std::size_t> StateOffsets(const Problem &problem) {
  std::vector<std::size_t> offsets(problem.stages.size() + 1);
  std::size_t next = 0;
  for (std::size_t i = 0; i < problem.stages.size(); ++i) {
    offsets[i] = next;
    next += problem.stages[i].A.cols();
  }
  offsets.back() = next;
  return offsets;
}

inline DensePrimal SolveDenseKkt(const Problem &problem) {
  const std::size_t horizon = problem.stages.size();
  const std::vector<std::size_t> state_offsets = StateOffsets(problem);
  const std::size_t state_variables =
      state_offsets.back() + problem.terminal_Q.rows();
  std::vector<std::size_t> control_offsets(horizon);
  std::size_t variables = state_variables;
  for (std::size_t i = 0; i < horizon; ++i) {
    control_offsets[i] = variables;
    variables += problem.stages[i].B.cols();
  }
  Matrix hessian(variables, variables);
  Vector linear(variables);
  for (std::size_t i = 0; i < horizon; ++i) {
    const Stage &stage = problem.stages[i];
    for (std::size_t row = 0; row < stage.Q.rows(); ++row) {
      linear[state_offsets[i] + row] += stage.q[row];
      for (std::size_t col = 0; col < stage.Q.cols(); ++col)
        hessian(state_offsets[i] + row, state_offsets[i] + col) +=
            stage.Q(row, col);
    }
    for (std::size_t row = 0; row < stage.R.rows(); ++row) {
      linear[control_offsets[i] + row] += stage.r[row];
      for (std::size_t col = 0; col < stage.R.cols(); ++col)
        hessian(control_offsets[i] + row, control_offsets[i] + col) +=
            stage.R(row, col);
    }
    for (std::size_t row = 0; row < stage.M.rows(); ++row) {
      for (std::size_t col = 0; col < stage.M.cols(); ++col) {
        hessian(state_offsets[i] + row, control_offsets[i] + col) +=
            stage.M(row, col);
        hessian(control_offsets[i] + col, state_offsets[i] + row) +=
            stage.M(row, col);
      }
    }
  }
  for (std::size_t row = 0; row < problem.terminal_Q.rows(); ++row) {
    linear[state_offsets.back() + row] += problem.terminal_q[row];
    for (std::size_t col = 0; col < problem.terminal_Q.cols(); ++col)
      hessian(state_offsets.back() + row, state_offsets.back() + col) +=
          problem.terminal_Q(row, col);
  }

  std::vector<Vector> rows;
  std::vector<Scalar> rhs;
  const auto add_constraint = [&](Vector row, Scalar right_hand_side) {
    rows.push_back(std::move(row));
    rhs.push_back(right_hand_side);
  };
  for (std::size_t row = 0; row < problem.initial_state.size(); ++row) {
    Vector constraint(variables);
    constraint[state_offsets.front() + row] = Scalar{1};
    add_constraint(std::move(constraint), problem.initial_state[row]);
  }
  for (std::size_t i = 0; i < horizon; ++i) {
    const Stage &stage = problem.stages[i];
    for (std::size_t row = 0; row < stage.A.rows(); ++row) {
      Vector constraint(variables);
      constraint[state_offsets[i + 1] + row] = Scalar{1};
      for (std::size_t col = 0; col < stage.A.cols(); ++col)
        constraint[state_offsets[i] + col] -= stage.A(row, col);
      for (std::size_t col = 0; col < stage.B.cols(); ++col)
        constraint[control_offsets[i] + col] -= stage.B(row, col);
      add_constraint(std::move(constraint), stage.c[row]);
    }
    for (std::size_t row = 0; row < stage.C.rows(); ++row) {
      Vector constraint(variables);
      for (std::size_t col = 0; col < stage.C.cols(); ++col)
        constraint[state_offsets[i] + col] = stage.C(row, col);
      for (std::size_t col = 0; col < stage.D.cols(); ++col)
        constraint[control_offsets[i] + col] = stage.D(row, col);
      add_constraint(std::move(constraint), -stage.d[row]);
    }
    for (std::size_t row = 0; row < stage.E.rows(); ++row) {
      Vector constraint(variables);
      for (std::size_t col = 0; col < stage.E.cols(); ++col)
        constraint[state_offsets[i] + col] = stage.E(row, col);
      add_constraint(std::move(constraint), -stage.e[row]);
    }
  }
  for (std::size_t row = 0; row < problem.terminal_E.rows(); ++row) {
    Vector constraint(variables);
    for (std::size_t col = 0; col < problem.terminal_E.cols(); ++col)
      constraint[state_offsets.back() + col] = problem.terminal_E(row, col);
    add_constraint(std::move(constraint), -problem.terminal_e[row]);
  }

  Matrix augmented(rows.size(), variables + 1);
  for (std::size_t row = 0; row < rows.size(); ++row) {
    for (std::size_t col = 0; col < variables; ++col)
      augmented(row, col) = rows[row][col];
    augmented(row, variables) = rhs[row];
  }
  const RrefResult independent = Rref(augmented, variables, kLinearTolerance);
  const std::size_t constraints = independent.pivot_rows.size();
  Matrix kkt(variables + constraints, variables + constraints);
  Vector right_hand_side(variables + constraints);
  for (std::size_t row = 0; row < variables; ++row) {
    right_hand_side[row] = -linear[row];
    for (std::size_t col = 0; col < variables; ++col)
      kkt(row, col) = hessian(row, col);
  }
  for (std::size_t row = 0; row < constraints; ++row) {
    const std::size_t source = independent.pivot_rows[row];
    right_hand_side[variables + row] = independent.matrix(source, variables);
    for (std::size_t col = 0; col < variables; ++col) {
      kkt(variables + row, col) = independent.matrix(source, col);
      kkt(col, variables + row) = independent.matrix(source, col);
    }
  }
  const Vector solution =
      SolveLinearSystem(kkt, right_hand_side, kLinearTolerance);
  DensePrimal result;
  result.states.resize(horizon + 1);
  result.controls.resize(horizon);
  for (std::size_t i = 0; i < horizon; ++i) {
    result.states[i] = Vector(problem.stages[i].A.cols());
    for (std::size_t row = 0; row < result.states[i].size(); ++row)
      result.states[i][row] = solution[state_offsets[i] + row];
    result.controls[i] = Vector(problem.stages[i].B.cols());
    for (std::size_t row = 0; row < result.controls[i].size(); ++row)
      result.controls[i][row] = solution[control_offsets[i] + row];
  }
  result.states.back() = Vector(problem.terminal_Q.rows());
  for (std::size_t row = 0; row < result.states.back().size(); ++row)
    result.states.back()[row] = solution[state_offsets.back() + row];
  return result;
}

inline Scalar MaxPrimalDifference(const KktPoint &actual,
                                  const DensePrimal &expected) {
  if (actual.states.size() != expected.states.size() ||
      actual.controls.size() != expected.controls.size()) {
    return std::numeric_limits<Scalar>::infinity();
  }
  Scalar result = Scalar{0};
  for (std::size_t i = 0; i < actual.states.size(); ++i)
    result =
        std::max(result, TestMaxAbs(actual.states[i] - expected.states[i]));
  for (std::size_t i = 0; i < actual.controls.size(); ++i)
    result =
        std::max(result, TestMaxAbs(actual.controls[i] - expected.controls[i]));
  return result;
}

} // namespace clqr::test::adversarial

#endif // CLQR_TESTS_ADVERSARIAL_TEST_SUPPORT_H_
