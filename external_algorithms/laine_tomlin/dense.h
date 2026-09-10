#ifndef EXTERNAL_ALGORITHMS_LAINE_TOMLIN_DENSE_H_
#define EXTERNAL_ALGORITHMS_LAINE_TOMLIN_DENSE_H_

#include <Eigen/Core>
#include "src/cpu_dense.h"

namespace laine_tomlin::dense {

// Eigen owns the reference solver's storage; CLQR supplies only its allocation-
// free arithmetic kernels. A column-major output is computed as C'=B'*A',
// avoiding matrix conversion or packing. Transposes and blocks retain strides.
template <class Derived> auto Layout(const Eigen::MatrixBase<Derived>& a) {
  struct Result { bool row_major; std::size_t leading; };
  if (a.innerStride() == 1)
    return Result{Derived::IsRowMajor, std::size_t(a.outerStride())};
  // A strided column/row vector is also a row/column-major one-column/row
  // matrix. This covers a column of the row-major multiple-RHS solution.
  if (a.cols() == 1)
    return Result{true, std::size_t(a.innerStride())};
  return Result{false, std::size_t(a.innerStride())};
}

template <class Out, class Left, class Right>
void Multiply(Eigen::MatrixBase<Out>& out, const Eigen::MatrixBase<Left>& a,
              const Eigen::MatrixBase<Right>& b, double alpha, double beta) {
  eigen_assert(a.cols() == b.rows() && out.rows() == a.rows() &&
               out.cols() == b.cols());
  if ((a.innerStride() != 1 && a.rows() != 1 && a.cols() != 1) ||
      (b.innerStride() != 1 && b.rows() != 1 && b.cols() != 1) ||
      out.innerStride() != 1) {
    // General non-unit-inner-stride maps are not used by the recurrence.
    // Support them directly without allocating a packed copy.
    for (Eigen::Index i = 0; i < out.rows(); ++i)
      for (Eigen::Index j = 0; j < out.cols(); ++j) {
        double value = beta == 0 ? 0 : beta * out(i, j);
        for (Eigen::Index k = 0; k < a.cols(); ++k)
          value += alpha * a(i, k) * b(k, j);
        out(i, j) = value;
      }
    return;
  }
  const auto left = Layout(a);
  const auto right = Layout(b);
  if constexpr (Out::IsRowMajor) {
    clqr::detail::NativeGemm(!left.row_major, !right.row_major, a.rows(),
        b.cols(), a.cols(), alpha, a.derived().data(), left.leading,
        b.derived().data(), right.leading, beta, out.derived().data(), out.outerStride());
  } else {
    clqr::detail::NativeGemm(right.row_major, left.row_major, b.cols(),
        a.rows(), a.cols(), alpha, b.derived().data(), right.leading,
        a.derived().data(), left.leading, beta, out.derived().data(), out.outerStride());
  }
}

template <class Left, class Right>
auto Product(const Eigen::MatrixBase<Left>& a, const Eigen::MatrixBase<Right>& b) {
  Eigen::Matrix<double, Left::RowsAtCompileTime, Right::ColsAtCompileTime>
      out(a.rows(), b.cols());
  Multiply(out, a, b, 1, 0);
  return out;
}

template <class Out, class Left, class Right>
void AddProduct(Eigen::MatrixBase<Out>& out, const Eigen::MatrixBase<Left>& a,
                const Eigen::MatrixBase<Right>& b, double alpha = 1) {
  Multiply(out, a, b, alpha, 1);
}

} // namespace laine_tomlin::dense
#endif
