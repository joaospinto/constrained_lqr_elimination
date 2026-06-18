#ifndef CLQR_LINALG_H_
#define CLQR_LINALG_H_

#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <string>
#include <vector>

namespace clqr {

class Vector {
 public:
  Vector() = default;
  explicit Vector(std::size_t size);
  Vector(std::initializer_list<double> values);

  std::size_t size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }
  void resize(std::size_t size);

  double& operator[](std::size_t i) { return data_[i]; }
  double operator[](std::size_t i) const { return data_[i]; }

  const std::vector<double>& data() const { return data_; }
  std::vector<double>& data() { return data_; }

 private:
  std::vector<double> data_;
};

class Matrix {
 public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t cols);
  Matrix(std::size_t rows, std::size_t cols, std::initializer_list<double> values);

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
  bool empty() const { return rows_ == 0 || cols_ == 0; }
  void resize(std::size_t rows, std::size_t cols);

  double& operator()(std::size_t row, std::size_t col);
  double operator()(std::size_t row, std::size_t col) const;

  const std::vector<double>& data() const { return data_; }
  std::vector<double>& data() { return data_; }

 private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<double> data_;
};

struct RrefResult {
  Matrix matrix;
  std::vector<std::size_t> pivot_columns;
  std::vector<std::size_t> pivot_rows;
};

Matrix Identity(std::size_t size);
Matrix Zeros(std::size_t rows, std::size_t cols);
Vector Zeros(std::size_t size);

Matrix Transpose(const Matrix& a);
Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator-(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& a, const Matrix& b);
Vector operator+(const Vector& a, const Vector& b);
Vector operator-(const Vector& a, const Vector& b);
Vector operator*(const Matrix& a, const Vector& x);
Matrix Scale(const Matrix& a, double alpha);
Vector Scale(const Vector& x, double alpha);

Matrix HorizontalConcat(const Matrix& a, const Matrix& b);
Matrix VerticalConcat(const Matrix& a, const Matrix& b);
Vector Concat(const Vector& a, const Vector& b);
Matrix Rows(const Matrix& a, const std::vector<std::size_t>& rows);
Matrix Cols(const Matrix& a, const std::vector<std::size_t>& cols);
Vector Entries(const Vector& a, const std::vector<std::size_t>& rows);
Matrix RemoveRows(const Matrix& a, const std::vector<std::size_t>& remove);

double Dot(const Vector& a, const Vector& b);
double MaxAbs(const Matrix& a);
double MaxAbs(const Vector& a);
bool IsNearlyZero(double value, double tolerance);
bool AllFinite(const Matrix& a);
bool AllFinite(const Vector& a);

Vector SolveLinearSystem(Matrix a, Vector b, double tolerance);
Matrix SolveLinearSystem(Matrix a, Matrix b, double tolerance);
RrefResult Rref(Matrix a, std::size_t pivot_column_limit, double tolerance);

std::string Shape(const Matrix& a);
std::ostream& operator<<(std::ostream& os, const Vector& x);
std::ostream& operator<<(std::ostream& os, const Matrix& a);

}  // namespace clqr

#endif  // CLQR_LINALG_H_
