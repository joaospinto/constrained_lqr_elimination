#ifndef CLQR_TESTS_CUDA_JAX_PROBLEM_H_
#define CLQR_TESTS_CUDA_JAX_PROBLEM_H_

#include <cmath>
#include <cstddef>
#include <vector>

#include "clqr/clqr.h"

namespace clqr {
namespace test {
namespace detail {

inline Scalar JaxFixtureValue(int seed, std::size_t row, std::size_t col = 0) {
  const Scalar x = static_cast<Scalar>(seed * 97 + row * 31 + col * 47);
  return std::sin(0.017 * x) + 0.3 * std::cos(0.029 * x);
}

inline Matrix JaxFixtureMatrix(std::size_t rows, std::size_t cols, int seed,
                               Scalar scale) {
  Matrix out(rows, cols);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col)
      out(row, col) = scale * JaxFixtureValue(seed, row, col);
  return out;
}

inline Vector JaxFixtureVector(std::size_t size, int seed, Scalar scale) {
  Vector out(size);
  for (std::size_t row = 0; row < size; ++row)
    out[row] = scale * JaxFixtureValue(seed, row);
  return out;
}

inline Matrix JaxFixturePositiveDefinite(std::size_t size, int seed,
                                         Scalar diagonal) {
  Matrix g = JaxFixtureMatrix(size, size, seed, 0.15);
  Matrix out = Transpose(g) * g;
  for (std::size_t row = 0; row < size; ++row) out(row, row) += diagonal;
  return out;
}

inline Scalar RowDot(const Matrix& matrix, std::size_t row,
                     const Vector& vector) {
  Scalar value = 0.0;
  for (std::size_t col = 0; col < vector.size(); ++col)
    value += matrix(row, col) * vector[col];
  return value;
}

}  // namespace detail

inline Problem MakeJaxCrossValidationProblem() {
  constexpr std::size_t horizon = 7;
  constexpr std::size_t n = 4;
  constexpr std::size_t m = 3;
  constexpr std::size_t p = 2;
  Problem problem;
  std::vector<Vector> x(horizon + 1);
  std::vector<Vector> u(horizon);
  for (std::size_t i = 0; i <= horizon; ++i)
    x[i] = detail::JaxFixtureVector(n, 101 + static_cast<int>(i), 0.5);
  for (std::size_t i = 0; i < horizon; ++i)
    u[i] = detail::JaxFixtureVector(m, 201 + static_cast<int>(i), 0.4);
  problem.initial_state = x.front();
  problem.stages.resize(horizon);
  for (std::size_t i = 0; i < horizon; ++i) {
    Stage& stage = problem.stages[i];
    stage.A = detail::JaxFixtureMatrix(n, n, 301 + static_cast<int>(i), 0.1);
    for (std::size_t row = 0; row < n; ++row) stage.A(row, row) += 0.85;
    stage.B = detail::JaxFixtureMatrix(n, m, 401 + static_cast<int>(i), 0.2);
    stage.c = x[i + 1] - stage.A * x[i] - stage.B * u[i];
    stage.Q =
        detail::JaxFixturePositiveDefinite(n, 501 + static_cast<int>(i), 1.0);
    stage.R =
        detail::JaxFixturePositiveDefinite(m, 601 + static_cast<int>(i), 1.3);
    stage.M = detail::JaxFixtureMatrix(n, m, 701 + static_cast<int>(i), 0.025);
    stage.q = detail::JaxFixtureVector(n, 801 + static_cast<int>(i), 0.15);
    stage.r = detail::JaxFixtureVector(m, 901 + static_cast<int>(i), 0.15);
    stage.C = Matrix(0, n);
    stage.D = Matrix(0, m);
    stage.d = Vector(0);
    stage.E = detail::JaxFixtureMatrix(p, n, 1001 + static_cast<int>(i), 0.3);
    for (std::size_t col = 0; col < n; ++col)
      stage.E(1, col) = 2.0 * stage.E(0, col);
    stage.e = Vector(p);
    stage.e[0] = -detail::RowDot(stage.E, 0, x[i]);
    stage.e[1] = 2.0 * stage.e[0];
  }
  problem.terminal_Q = detail::JaxFixturePositiveDefinite(n, 1101, 1.4);
  problem.terminal_q = detail::JaxFixtureVector(n, 1201, 0.15);
  problem.terminal_E = Matrix(0, n);
  problem.terminal_e = Vector(0);
  return problem;
}

}  // namespace test
}  // namespace clqr

#endif  // CLQR_TESTS_CUDA_JAX_PROBLEM_H_
