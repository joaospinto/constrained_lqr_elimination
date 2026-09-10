#ifndef CLQR_SRC_CPU_DENSE_H_
#define CLQR_SRC_CPU_DENSE_H_

#include <cstddef>
#include <cstring>

#include "clqr/linalg.h"

namespace clqr::detail {

// Row-major products with independent leading dimensions. Inputs may overlap
// each other, but must not overlap the output. Fixed-size output tiles expose
// independent accumulations to SIMD without reassociating the inner product.
// No packing buffers, heap allocations, or dimension-specific instantiations.
template <bool TransposeA, bool TransposeB, int Rows, int Cols, class T>
inline void NativeGemmTile(std::size_t shared, T alpha, const T* a,
                           std::size_t lda, const T* b, std::size_t ldb, T beta,
                           T* out, std::size_t ldc) {
#if defined(__GNUC__) || defined(__clang__)
  // A 128-bit compiler vector maps to NEON on ARM and SSE on x86. memcpy
  // permits ordinary scalar alignment; no aligned allocation is required.
  constexpr int lanes = 16 / sizeof(T);
  if constexpr (Cols % lanes == 0) {
    typedef T Pack __attribute__((vector_size(16)));
    constexpr int packs = Cols / lanes;
    Pack sums[Rows][packs] = {};
    if (beta != T{0})
      for (int i = 0; i < Rows; ++i)
        for (int j = 0; j < packs; ++j) {
          Pack value;
          std::memcpy(&value, out + i * ldc + j * lanes, sizeof(Pack));
          sums[i][j] = beta * value;
        }
    for (std::size_t k = 0; k < shared; ++k) {
      Pack right[packs] = {};
      for (int j = 0; j < packs; ++j) {
        if constexpr (TransposeB) {
          T gathered[lanes];
          for (int lane = 0; lane < lanes; ++lane)
            gathered[lane] = b[(j * lanes + lane) * ldb + k];
          std::memcpy(&right[j], gathered, sizeof(Pack));
        } else {
          std::memcpy(&right[j], b + k * ldb + j * lanes, sizeof(Pack));
        }
      }
      for (int i = 0; i < Rows; ++i) {
        const T value = alpha * a[TransposeA ? k * lda + i : i * lda + k];
        for (int j = 0; j < packs; ++j) sums[i][j] += value * right[j];
      }
    }
    for (int i = 0; i < Rows; ++i)
      for (int j = 0; j < packs; ++j)
        std::memcpy(out + i * ldc + j * lanes, &sums[i][j], sizeof(Pack));
    return;
  }
#endif
  T sums[Rows][Cols] = {};
  for (int i = 0; i < Rows; ++i)
    for (int j = 0; j < Cols; ++j)
      sums[i][j] = beta == T{0} ? T{0} : beta * out[i * ldc + j];
  for (std::size_t k = 0; k < shared; ++k) {
    for (int i = 0; i < Rows; ++i) {
      const T value = alpha * a[TransposeA ? k * lda + i : i * lda + k];
      for (int j = 0; j < Cols; ++j)
        sums[i][j] += value * b[TransposeB ? j * ldb + k : k * ldb + j];
    }
  }
  for (int i = 0; i < Rows; ++i)
    for (int j = 0; j < Cols; ++j) out[i * ldc + j] = sums[i][j];
}

template <bool TransposeA, bool TransposeB, class T>
inline void NativeGemmImpl(std::size_t rows, std::size_t cols,
                           std::size_t shared, T alpha, const T* a,
                           std::size_t lda, const T* b, std::size_t ldb, T beta,
                           T* out, std::size_t ldc) {
  std::size_t i = 0;
  for (; i + 4 <= rows; i += 4) {
    const T* ai = a + (TransposeA ? i : i * lda);
    std::size_t j = 0;
    for (; j + 8 <= cols; j += 8)
      NativeGemmTile<TransposeA, TransposeB, 4, 8>(
          shared, alpha, ai, lda, b + (TransposeB ? j * ldb : j), ldb, beta,
          out + i * ldc + j, ldc);
    for (; j + 4 <= cols; j += 4)
      NativeGemmTile<TransposeA, TransposeB, 4, 4>(
          shared, alpha, ai, lda, b + (TransposeB ? j * ldb : j), ldb, beta,
          out + i * ldc + j, ldc);
    for (; j < cols; ++j)
      NativeGemmTile<TransposeA, TransposeB, 4, 1>(
          shared, alpha, ai, lda, b + (TransposeB ? j * ldb : j), ldb, beta,
          out + i * ldc + j, ldc);
  }
  for (; i < rows; ++i) {
    const T* ai = a + (TransposeA ? i : i * lda);
    std::size_t j = 0;
    for (; j + 4 <= cols; j += 4)
      NativeGemmTile<TransposeA, TransposeB, 1, 4>(
          shared, alpha, ai, lda, b + (TransposeB ? j * ldb : j), ldb, beta,
          out + i * ldc + j, ldc);
    for (; j < cols; ++j)
      NativeGemmTile<TransposeA, TransposeB, 1, 1>(
          shared, alpha, ai, lda, b + (TransposeB ? j * ldb : j), ldb, beta,
          out + i * ldc + j, ldc);
  }
}

template <class T>
inline void NativeGemm(bool transpose_a, bool transpose_b, std::size_t rows,
                       std::size_t cols, std::size_t shared, T alpha,
                       const T* a, std::size_t lda, const T* b, std::size_t ldb,
                       T beta, T* out, std::size_t ldc) {
  if (rows == 0 || cols == 0) return;
  if (shared == 0 || alpha == T{0}) {
    for (std::size_t i = 0; i < rows; ++i)
      for (std::size_t j = 0; j < cols; ++j)
        out[i * ldc + j] = beta == T{0} ? T{0} : beta * out[i * ldc + j];
    return;
  }
  if (transpose_a) {
    if (transpose_b)
      NativeGemmImpl<true, true>(rows, cols, shared, alpha, a, lda, b, ldb,
                                 beta, out, ldc);
    else
      NativeGemmImpl<true, false>(rows, cols, shared, alpha, a, lda, b, ldb,
                                  beta, out, ldc);
  } else {
    if (transpose_b)
      NativeGemmImpl<false, true>(rows, cols, shared, alpha, a, lda, b, ldb,
                                  beta, out, ldc);
    else
      NativeGemmImpl<false, false>(rows, cols, shared, alpha, a, lda, b, ldb,
                                   beta, out, ldc);
  }
}

template <bool Transpose, int Cols, class T>
inline void NativeTriangularTile(const T* lower, std::size_t order,
                                 std::size_t columns, std::size_t row, T* rhs) {
  T values[Cols];
  for (int col = 0; col < Cols; ++col) values[col] = rhs[row * columns + col];
  const std::size_t begin = Transpose ? row + 1 : 0;
  const std::size_t end = Transpose ? order : row;
  for (std::size_t k = begin; k < end; ++k) {
    const T value = lower[Transpose ? k * order + row : row * order + k];
    for (int col = 0; col < Cols; ++col)
      values[col] -= value * rhs[k * columns + col];
  }
  for (int col = 0; col < Cols; ++col)
    rhs[row * columns + col] = values[col] / lower[row * order + row];
}

template <class T>
inline void NativeCholeskySolve(const T* lower, std::size_t order,
                                std::size_t columns, T* rhs) {
  for (std::size_t row = 0; row < order; ++row) {
    std::size_t col = 0;
    for (; col + 4 <= columns; col += 4)
      NativeTriangularTile<false, 4>(lower, order, columns, row, rhs + col);
    for (; col < columns; ++col)
      NativeTriangularTile<false, 1>(lower, order, columns, row, rhs + col);
  }
  for (std::size_t rev = 0; rev < order; ++rev) {
    const std::size_t row = order - 1 - rev;
    std::size_t col = 0;
    for (; col + 4 <= columns; col += 4)
      NativeTriangularTile<true, 4>(lower, order, columns, row, rhs + col);
    for (; col < columns; ++col)
      NativeTriangularTile<true, 1>(lower, order, columns, row, rhs + col);
  }
}

}  // namespace clqr::detail
#endif  // CLQR_SRC_CPU_DENSE_H_
