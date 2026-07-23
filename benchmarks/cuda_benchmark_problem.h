#ifndef CLQR_BENCHMARKS_CUDA_BENCHMARK_PROBLEM_H_
#define CLQR_BENCHMARKS_CUDA_BENCHMARK_PROBLEM_H_

#include <cmath>
#include <cstddef>
#include <vector>

#include "clqr/clqr.h"

namespace clqr {
namespace benchmark {

inline Scalar Value(int seed, std::size_t row, std::size_t col = 0) {
  const Scalar x = static_cast<Scalar>(seed * 97 + row * 31 + col * 47);
  return std::sin(0.017 * x) + 0.3 * std::cos(0.029 * x);
}

inline Matrix GeneratedMatrix(std::size_t rows, std::size_t cols, int seed,
                              Scalar scale) {
  Matrix out(rows, cols);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col)
      out(row, col) = scale * Value(seed, row, col);
  }
  return out;
}

inline Vector GeneratedVector(std::size_t size, int seed, Scalar scale) {
  Vector out(size);
  for (std::size_t row = 0; row < size; ++row)
    out[row] = scale * Value(seed, row);
  return out;
}

inline Matrix PositiveDefinite(std::size_t size, int seed, Scalar diagonal) {
  Matrix g = GeneratedMatrix(size, size, seed, Scalar{0.15});
  Matrix out = Transpose(g) * g;
  for (std::size_t row = 0; row < size; ++row) out(row, row) += diagonal;
  return out;
}

inline Scalar RowDot(const Matrix& matrix, std::size_t row,
                     const Vector& vector) {
  Scalar value = Scalar{0};
  for (std::size_t col = 0; col < vector.size(); ++col)
    value += matrix(row, col) * vector[col];
  return value;
}

inline Problem StateOnlyProblem(std::size_t horizon, std::size_t n,
                                std::size_t m, std::size_t p) {
  Problem problem;
  std::vector<Vector> x(horizon + 1);
  std::vector<Vector> u(horizon);
  for (std::size_t i = 0; i <= horizon; ++i)
    x[i] = GeneratedVector(n, 1000 + static_cast<int>(i), Scalar{0.5});
  for (std::size_t i = 0; i < horizon; ++i)
    u[i] = GeneratedVector(m, 2000 + static_cast<int>(i), Scalar{0.4});
  problem.initial_state = x[0];
  problem.stages.resize(horizon);
  for (std::size_t i = 0; i < horizon; ++i) {
    Stage& stage = problem.stages[i];
    stage.A = GeneratedMatrix(n, n, 3000 + static_cast<int>(i),
                              Scalar{0.08});
    for (std::size_t row = 0; row < n; ++row) stage.A(row, row) += 0.9;
    stage.B = GeneratedMatrix(n, m, 4000 + static_cast<int>(i),
                              Scalar{0.15});
    stage.c = x[i + 1] - stage.A * x[i] - stage.B * u[i];
    stage.Q = PositiveDefinite(n, 5000 + static_cast<int>(i), Scalar{1.0});
    stage.R = PositiveDefinite(m, 6000 + static_cast<int>(i), Scalar{1.5});
    stage.M = GeneratedMatrix(n, m, 7000 + static_cast<int>(i),
                              Scalar{0.02});
    stage.q = GeneratedVector(n, 8000 + static_cast<int>(i), Scalar{0.1});
    stage.r = GeneratedVector(m, 9000 + static_cast<int>(i), Scalar{0.1});
    stage.C = Matrix(0, n);
    stage.D = Matrix(0, m);
    stage.d = Vector(0);
    stage.E = GeneratedMatrix(p, n, 10000 + static_cast<int>(i),
                              Scalar{0.3});
    stage.e = Vector(p);
    for (std::size_t row = 0; row < p; ++row)
      stage.e[row] = -RowDot(stage.E, row, x[i]);
  }
  problem.terminal_Q = PositiveDefinite(n, 11000, Scalar{1.5});
  problem.terminal_q = GeneratedVector(n, 12000, Scalar{0.1});
  problem.terminal_E = Matrix(0, n);
  problem.terminal_e = Vector(0);
  return problem;
}

}  // namespace benchmark
}  // namespace clqr

#endif  // CLQR_BENCHMARKS_CUDA_BENCHMARK_PROBLEM_H_
