#ifndef CLQR_LINALG_H_
#define CLQR_LINALG_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <new>
#include <ostream>
#include <string>
#include <vector>

namespace clqr {

class WorkspaceArena {
 public:
  WorkspaceArena() = default;
  WorkspaceArena(void* memory, std::size_t bytes) { Reset(memory, bytes); }

  void Reset(void* memory, std::size_t bytes) {
    data_ = reinterpret_cast<unsigned char*>(memory);
    size_ = bytes;
    used_ = 0;
  }
  void Clear() { used_ = 0; }
  void* Allocate(std::size_t bytes, std::size_t alignment) {
    const std::size_t aligned = Align(used_, alignment);
    if (aligned > size_ || bytes > size_ - aligned) throw std::bad_alloc();
    void* out = data_ + aligned;
    used_ = aligned + bytes;
    return out;
  }
  bool Owns(const void* ptr) const {
    const auto* p = reinterpret_cast<const unsigned char*>(ptr);
    return p >= data_ && p < data_ + size_;
  }
  std::size_t used() const { return used_; }
  std::size_t size() const { return size_; }

 private:
  static constexpr std::size_t Align(std::size_t offset, std::size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
  }

  unsigned char* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t used_ = 0;
};

WorkspaceArena* ActiveWorkspaceArena();
void SetActiveWorkspaceArena(WorkspaceArena* arena);

class ScopedWorkspaceArena {
 public:
  explicit ScopedWorkspaceArena(WorkspaceArena* arena)
      : previous_(ActiveWorkspaceArena()) {
    SetActiveWorkspaceArena(arena);
  }
  ~ScopedWorkspaceArena() { SetActiveWorkspaceArena(previous_); }

 private:
  WorkspaceArena* previous_ = nullptr;
};

template <typename T>
class WorkspaceAllocator {
 public:
  using value_type = T;

  WorkspaceAllocator() = default;
  template <typename U>
  WorkspaceAllocator(const WorkspaceAllocator<U>&) {}

  T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();
    const std::size_t bytes = n * sizeof(T);
    WorkspaceArena* arena = ActiveWorkspaceArena();
    if (arena != nullptr) return static_cast<T*>(arena->Allocate(bytes, alignof(T)));
    return static_cast<T*>(::operator new(bytes));
  }

  void deallocate(T* ptr, std::size_t) noexcept {
    WorkspaceArena* arena = ActiveWorkspaceArena();
    if (arena != nullptr && arena->Owns(ptr)) return;
    ::operator delete(ptr);
  }

  template <typename U>
  bool operator==(const WorkspaceAllocator<U>&) const noexcept {
    return true;
  }
  template <typename U>
  bool operator!=(const WorkspaceAllocator<U>&) const noexcept {
    return false;
  }
};

template <typename T>
using WorkspaceVector = std::vector<T, WorkspaceAllocator<T>>;

class Vector {
 public:
  using Storage = WorkspaceVector<double>;

  Vector() = default;
  explicit Vector(std::size_t size);
  Vector(std::initializer_list<double> values);

  std::size_t size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }
  void reserve(std::size_t size);
  void resize(std::size_t size);

  double& operator[](std::size_t i) { return data_[i]; }
  double operator[](std::size_t i) const { return data_[i]; }

  const Storage& data() const { return data_; }
  Storage& data() { return data_; }

 private:
  Storage data_;
};

class Matrix {
 public:
  using Storage = WorkspaceVector<double>;

  Matrix() = default;
  Matrix(std::size_t rows, std::size_t cols);
  Matrix(std::size_t rows, std::size_t cols, std::initializer_list<double> values);

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
  bool empty() const { return rows_ == 0 || cols_ == 0; }
  void reserve(std::size_t rows, std::size_t cols);
  void resize(std::size_t rows, std::size_t cols);

  double& operator()(std::size_t row, std::size_t col);
  double operator()(std::size_t row, std::size_t col) const;

  const Storage& data() const { return data_; }
  Storage& data() { return data_; }

 private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  Storage data_;
};

struct RrefResult {
  Matrix matrix;
  WorkspaceVector<std::size_t> pivot_columns;
  WorkspaceVector<std::size_t> pivot_rows;
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
Matrix Rows(const Matrix& a, const WorkspaceVector<std::size_t>& rows);
Matrix Cols(const Matrix& a, const WorkspaceVector<std::size_t>& cols);
Vector Entries(const Vector& a, const WorkspaceVector<std::size_t>& rows);
Matrix RemoveRows(const Matrix& a, const WorkspaceVector<std::size_t>& remove);

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
