#include "dense.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using Matrix = Eigen::MatrixXd;
using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <class A, class B> void Check(const A& a, const B& b) {
  const Matrix expected = a * b;
  const auto check = [&](const auto& actual, const auto& reference) {
    if (!actual.allFinite() ||
        (actual.size() && (actual - reference).cwiseAbs().maxCoeff() >
             1e-12 * std::max(1.0, reference.cwiseAbs().maxCoeff())))
      throw std::runtime_error("native/Eigen product mismatch");
  };
  check(laine_tomlin::dense::Product(a, b), expected);
  RowMatrix out = RowMatrix::Constant(a.rows(), b.cols(), 0.25);
  laine_tomlin::dense::Multiply(out, a, b, -0.5, 2);
  check(out, (0.5 - 0.5 * expected.array()).matrix());
}
} // namespace

int main() {
  for (int rows : {0, 1, 3, 8})
    for (int cols : {0, 1, 5, 9})
      for (int shared : {0, 1, 4, 7}) {
        Matrix a = Matrix::Random(rows, shared), b = Matrix::Random(shared, cols);
        RowMatrix ar = a, br = b;
        Check(a, b);
        Check(a, br);
        Check(ar, b);
        Check(ar, br);
        Check(b.transpose(), a.transpose());
      }
  Matrix a = Matrix::Random(9, 7), b = Matrix::Random(7, 11);
  Check(a.block(1, 2, 5, 3), b.block(2, 1, 3, 4));
  RowMatrix row_b = b;
  Check(a, row_b.col(4));
  Check(a.row(3), b);
  using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
  const Eigen::Map<const Matrix, 0, Stride> strided(a.data(), 4, 3, Stride(18, 2));
  Check(strided, b.topRows(3));
  std::cout << "Native matrix routing tests passed\n";
}
