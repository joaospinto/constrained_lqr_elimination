#include "src/cpu_dense.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
template <class T>
T Value(std::size_t i) {
  return static_cast<T>(std::sin(double(i + 1)));
}

template <class T>
void Near(T actual, long double expected, long double scale = 1) {
  if (!std::isfinite(actual) ||
      std::abs(static_cast<long double>(actual) - expected) >
          64 * std::numeric_limits<T>::epsilon() * std::max(1.L, scale))
    throw std::runtime_error("native dense kernel mismatch");
}

template <class T>
void Products() {
  for (const std::size_t rows : {0, 1, 3, 4, 5, 8, 17, 33})
    for (const std::size_t cols : {0, 1, 3, 4, 5, 8, 19, 37})
      for (const std::size_t shared : {0, 1, 3, 4, 7, 16, 33})
        for (const bool ta : {false, true})
          for (const bool tb : {false, true}) {
            const std::size_t lda = (ta ? rows : shared) + 3;
            const std::size_t ldb = (tb ? shared : cols) + 2;
            const std::size_t ldc = cols + 5;
            std::vector<T> a((ta ? shared : rows) * lda);
            std::vector<T> b((tb ? cols : shared) * ldb);
            for (std::size_t i = 0; i < a.size(); ++i) a[i] = Value<T>(i);
            for (std::size_t i = 0; i < b.size(); ++i) b[i] = Value<T>(i + 19);
            for (const T alpha : {T{0}, T{1}, T{-0.7}})
              for (const T beta : {T{0}, T{1}, T{-0.3}}) {
                std::vector<T> out(rows * ldc, T{-999});
                for (std::size_t i = 0; i < rows; ++i)
                  for (std::size_t j = 0; j < cols; ++j)
                    out[i * ldc + j] = beta == 0
                                           ? std::numeric_limits<T>::quiet_NaN()
                                           : Value<T>(i * ldc + j);
                const auto original = out;
                clqr::detail::NativeGemm(ta, tb, rows, cols, shared, alpha,
                                         shared && alpha ? a.data() : nullptr,
                                         lda,
                                         shared && alpha ? b.data() : nullptr,
                                         ldb, beta, out.data(), ldc);
                for (std::size_t i = 0; i < rows; ++i) {
                  for (std::size_t j = 0; j < cols; ++j) {
                    long double expected =
                        beta == 0 ? 0
                                  : static_cast<long double>(beta) *
                                        original[i * ldc + j];
                    long double scale = std::abs(expected);
                    for (std::size_t k = 0; k < shared; ++k) {
                      const long double term =
                          static_cast<long double>(alpha) *
                          a[ta ? k * lda + i : i * lda + k] *
                          b[tb ? j * ldb + k : k * ldb + j];
                      expected += term;
                      scale += std::abs(term);
                    }
                    Near(out[i * ldc + j], expected, scale);
                  }
                  for (std::size_t j = cols; j < ldc; ++j)
                    Near(out[i * ldc + j], -999);
                }
              }
          }
}

template <class T>
void TriangularSolves() {
  for (const std::size_t order : {0, 1, 3, 4, 5, 8, 33})
    for (const std::size_t cols : {0, 1, 3, 4, 5, 8, 17, 32}) {
      std::vector<T> lower(order * order, std::numeric_limits<T>::quiet_NaN());
      std::vector<T> expected(order * cols), intermediate(order * cols, T{0});
      std::vector<T> rhs(order * cols, T{0});
      for (std::size_t i = 0; i < order; ++i) {
        for (std::size_t j = 0; j <= i; ++j)
          lower[i * order + j] =
              i == j ? T{2} : T{0.05} * Value<T>(i * order + j);
        for (std::size_t j = 0; j < cols; ++j)
          expected[i * cols + j] = Value<T>(i * cols + j);
      }
      for (std::size_t i = 0; i < order; ++i)
        for (std::size_t j = 0; j < cols; ++j)
          for (std::size_t k = i; k < order; ++k)
            intermediate[i * cols + j] +=
                lower[k * order + i] * expected[k * cols + j];
      for (std::size_t i = 0; i < order; ++i)
        for (std::size_t j = 0; j < cols; ++j)
          for (std::size_t k = 0; k <= i; ++k)
            rhs[i * cols + j] +=
                lower[i * order + k] * intermediate[k * cols + j];
      clqr::detail::NativeCholeskySolve(lower.data(), order, cols, rhs.data());
      for (std::size_t i = 0; i < rhs.size(); ++i) Near(rhs[i], expected[i]);
    }
}
}  // namespace

int main() {
  Products<float>();
  Products<double>();
  TriangularSolves<float>();
  TriangularSolves<double>();
  std::cout << "FP32/FP64 native products and triangular solves passed.\n";
}
