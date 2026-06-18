#include "clqr/linalg.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace clqr {
namespace {

thread_local WorkspaceArena* active_workspace_arena = nullptr;

void Check(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

}  // namespace

WorkspaceArena* ActiveWorkspaceArena() { return active_workspace_arena; }

void SetActiveWorkspaceArena(WorkspaceArena* arena) { active_workspace_arena = arena; }

Vector::Vector(std::size_t size) : data_(size, 0.0) {}

Vector::Vector(std::initializer_list<double> values) : data_(values) {}

void Vector::reserve(std::size_t size) { data_.reserve(size); }

void Vector::resize(std::size_t size) { data_.assign(size, 0.0); }

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

Matrix::Matrix(std::size_t rows, std::size_t cols,
               std::initializer_list<double> values)
    : rows_(rows), cols_(cols), data_(values) {
  Check(data_.size() == rows * cols, "matrix initializer has wrong size");
}

void Matrix::reserve(std::size_t rows, std::size_t cols) { data_.reserve(rows * cols); }

void Matrix::resize(std::size_t rows, std::size_t cols) {
  rows_ = rows;
  cols_ = cols;
  data_.assign(rows * cols, 0.0);
}

double& Matrix::operator()(std::size_t row, std::size_t col) {
  return data_[row * cols_ + col];
}

double Matrix::operator()(std::size_t row, std::size_t col) const {
  return data_[row * cols_ + col];
}

Matrix Identity(std::size_t size) {
  Matrix out(size, size);
  for (std::size_t i = 0; i < size; ++i) out(i, i) = 1.0;
  return out;
}

Matrix Zeros(std::size_t rows, std::size_t cols) { return Matrix(rows, cols); }

Vector Zeros(std::size_t size) { return Vector(size); }

Matrix Transpose(const Matrix& a) {
  Matrix out(a.cols(), a.rows());
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) out(j, i) = a(i, j);
  }
  return out;
}

Matrix operator+(const Matrix& a, const Matrix& b) {
  Check(a.rows() == b.rows() && a.cols() == b.cols(), "matrix add shape mismatch");
  Matrix out(a.rows(), a.cols());
  for (std::size_t i = 0; i < out.data().size(); ++i) out.data()[i] = a.data()[i] + b.data()[i];
  return out;
}

Matrix operator-(const Matrix& a, const Matrix& b) {
  Check(a.rows() == b.rows() && a.cols() == b.cols(), "matrix subtract shape mismatch");
  Matrix out(a.rows(), a.cols());
  for (std::size_t i = 0; i < out.data().size(); ++i) out.data()[i] = a.data()[i] - b.data()[i];
  return out;
}

Matrix operator*(const Matrix& a, const Matrix& b) {
  Check(a.cols() == b.rows(), "matrix multiply shape mismatch");
  Matrix out(a.rows(), b.cols());
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t k = 0; k < a.cols(); ++k) {
      const double aik = a(i, k);
      for (std::size_t j = 0; j < b.cols(); ++j) out(i, j) += aik * b(k, j);
    }
  }
  return out;
}

Vector operator+(const Vector& a, const Vector& b) {
  Check(a.size() == b.size(), "vector add shape mismatch");
  Vector out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
  return out;
}

Vector operator-(const Vector& a, const Vector& b) {
  Check(a.size() == b.size(), "vector subtract shape mismatch");
  Vector out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] - b[i];
  return out;
}

Vector operator*(const Matrix& a, const Vector& x) {
  Check(a.cols() == x.size(), "matrix-vector multiply shape mismatch");
  Vector out(a.rows());
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) out[i] += a(i, j) * x[j];
  }
  return out;
}

Matrix Scale(const Matrix& a, double alpha) {
  Matrix out(a.rows(), a.cols());
  for (std::size_t i = 0; i < out.data().size(); ++i) out.data()[i] = alpha * a.data()[i];
  return out;
}

Vector Scale(const Vector& x, double alpha) {
  Vector out(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) out[i] = alpha * x[i];
  return out;
}

Matrix HorizontalConcat(const Matrix& a, const Matrix& b) {
  Check(a.rows() == b.rows(), "horizontal concat row mismatch");
  Matrix out(a.rows(), a.cols() + b.cols());
  for (std::size_t i = 0; i < out.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) out(i, j) = a(i, j);
    for (std::size_t j = 0; j < b.cols(); ++j) out(i, a.cols() + j) = b(i, j);
  }
  return out;
}

Matrix VerticalConcat(const Matrix& a, const Matrix& b) {
  Check(a.cols() == b.cols(), "vertical concat column mismatch");
  Matrix out(a.rows() + b.rows(), a.cols());
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) out(i, j) = a(i, j);
  }
  for (std::size_t i = 0; i < b.rows(); ++i) {
    for (std::size_t j = 0; j < b.cols(); ++j) out(a.rows() + i, j) = b(i, j);
  }
  return out;
}

Vector Concat(const Vector& a, const Vector& b) {
  Vector out(a.size() + b.size());
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i];
  for (std::size_t i = 0; i < b.size(); ++i) out[a.size() + i] = b[i];
  return out;
}

Matrix Rows(const Matrix& a, const WorkspaceVector<std::size_t>& rows) {
  Matrix out(rows.size(), a.cols());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) out(i, j) = a(rows[i], j);
  }
  return out;
}

Matrix Cols(const Matrix& a, const WorkspaceVector<std::size_t>& cols) {
  Matrix out(a.rows(), cols.size());
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < cols.size(); ++j) out(i, j) = a(i, cols[j]);
  }
  return out;
}

Vector Entries(const Vector& a, const WorkspaceVector<std::size_t>& rows) {
  Vector out(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) out[i] = a[rows[i]];
  return out;
}

Matrix RemoveRows(const Matrix& a, const WorkspaceVector<std::size_t>& remove) {
  Matrix out(a.rows() - remove.size(), a.cols());
  std::size_t row = 0;
  for (std::size_t i = 0; i < a.rows(); ++i) {
    bool removed = false;
    for (std::size_t index : remove) {
      if (index == i) {
        removed = true;
        break;
      }
    }
    if (removed) continue;
    for (std::size_t j = 0; j < a.cols(); ++j) out(row, j) = a(i, j);
    ++row;
  }
  return out;
}

double Dot(const Vector& a, const Vector& b) {
  Check(a.size() == b.size(), "dot product shape mismatch");
  double out = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) out += a[i] * b[i];
  return out;
}

double MaxAbs(const Matrix& a) {
  double out = 0.0;
  for (double value : a.data()) out = std::max(out, std::abs(value));
  return out;
}

double MaxAbs(const Vector& a) {
  double out = 0.0;
  for (double value : a.data()) out = std::max(out, std::abs(value));
  return out;
}

bool IsNearlyZero(double value, double tolerance) { return std::abs(value) <= tolerance; }

bool AllFinite(const Matrix& a) {
  for (double value : a.data()) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

bool AllFinite(const Vector& a) {
  for (double value : a.data()) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

Vector SolveLinearSystem(Matrix a, Vector b, double tolerance) {
  Check(a.rows() == a.cols(), "linear solve matrix must be square");
  Check(a.rows() == b.size(), "linear solve rhs shape mismatch");
  const std::size_t n = a.rows();
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot = col;
    double best = std::abs(a(col, col));
    for (std::size_t row = col + 1; row < n; ++row) {
      const double candidate = std::abs(a(row, col));
      if (candidate > best) {
        best = candidate;
        pivot = row;
      }
    }
    if (best <= tolerance) throw std::runtime_error("singular linear system");
    if (pivot != col) {
      for (std::size_t j = col; j < n; ++j) std::swap(a(col, j), a(pivot, j));
      std::swap(b[col], b[pivot]);
    }
    const double pivot_value = a(col, col);
    for (std::size_t j = col; j < n; ++j) a(col, j) /= pivot_value;
    b[col] /= pivot_value;
    for (std::size_t row = 0; row < n; ++row) {
      if (row == col) continue;
      const double factor = a(row, col);
      if (IsNearlyZero(factor, tolerance)) continue;
      for (std::size_t j = col; j < n; ++j) a(row, j) -= factor * a(col, j);
      b[row] -= factor * b[col];
    }
  }
  return b;
}

Matrix SolveLinearSystem(Matrix a, Matrix b, double tolerance) {
  Check(a.rows() == a.cols(), "linear solve matrix must be square");
  Check(a.rows() == b.rows(), "linear solve rhs shape mismatch");
  const std::size_t n = a.rows();
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot = col;
    double best = std::abs(a(col, col));
    for (std::size_t row = col + 1; row < n; ++row) {
      const double candidate = std::abs(a(row, col));
      if (candidate > best) {
        best = candidate;
        pivot = row;
      }
    }
    if (best <= tolerance) throw std::runtime_error("singular linear system");
    if (pivot != col) {
      for (std::size_t j = col; j < n; ++j) std::swap(a(col, j), a(pivot, j));
      for (std::size_t j = 0; j < b.cols(); ++j) std::swap(b(col, j), b(pivot, j));
    }
    const double pivot_value = a(col, col);
    for (std::size_t j = col; j < n; ++j) a(col, j) /= pivot_value;
    for (std::size_t j = 0; j < b.cols(); ++j) b(col, j) /= pivot_value;
    for (std::size_t row = 0; row < n; ++row) {
      if (row == col) continue;
      const double factor = a(row, col);
      if (IsNearlyZero(factor, tolerance)) continue;
      for (std::size_t j = col; j < n; ++j) a(row, j) -= factor * a(col, j);
      for (std::size_t j = 0; j < b.cols(); ++j) b(row, j) -= factor * b(col, j);
    }
  }
  return b;
}

RrefResult Rref(Matrix a, std::size_t pivot_column_limit, double tolerance) {
  RrefResult result;
  result.pivot_columns.reserve(std::min(a.rows(), pivot_column_limit));
  result.pivot_rows.reserve(std::min(a.rows(), pivot_column_limit));
  std::size_t pivot_row = 0;
  const std::size_t limit = std::min(pivot_column_limit, a.cols());
  for (std::size_t col = 0; col < limit && pivot_row < a.rows(); ++col) {
    std::size_t best_row = pivot_row;
    double best = std::abs(a(best_row, col));
    for (std::size_t row = pivot_row + 1; row < a.rows(); ++row) {
      const double candidate = std::abs(a(row, col));
      if (candidate > best) {
        best = candidate;
        best_row = row;
      }
    }
    if (best <= tolerance) continue;
    if (best_row != pivot_row) {
      for (std::size_t j = 0; j < a.cols(); ++j) std::swap(a(pivot_row, j), a(best_row, j));
    }
    const double pivot_value = a(pivot_row, col);
    for (std::size_t j = col; j < a.cols(); ++j) a(pivot_row, j) /= pivot_value;
    for (std::size_t row = 0; row < a.rows(); ++row) {
      if (row == pivot_row) continue;
      const double factor = a(row, col);
      if (IsNearlyZero(factor, tolerance)) continue;
      for (std::size_t j = col; j < a.cols(); ++j) a(row, j) -= factor * a(pivot_row, j);
    }
    result.pivot_columns.push_back(col);
    result.pivot_rows.push_back(pivot_row);
    ++pivot_row;
  }
  for (double& value : a.data()) {
    if (IsNearlyZero(value, tolerance)) value = 0.0;
  }
  result.matrix = std::move(a);
  return result;
}

std::string Shape(const Matrix& a) {
  std::ostringstream os;
  os << a.rows() << "x" << a.cols();
  return os.str();
}

std::ostream& operator<<(std::ostream& os, const Vector& x) {
  os << "[";
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (i) os << ", ";
    os << std::setprecision(12) << x[i];
  }
  os << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Matrix& a) {
  os << "[";
  for (std::size_t i = 0; i < a.rows(); ++i) {
    if (i) os << "; ";
    for (std::size_t j = 0; j < a.cols(); ++j) {
      if (j) os << ", ";
      os << std::setprecision(12) << a(i, j);
    }
  }
  os << "]";
  return os;
}

}  // namespace clqr
