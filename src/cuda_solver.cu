#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "clqr/cuda.h"
#include "cuda_internal.h"

namespace clqr {
namespace cuda {
namespace detail {
namespace {

constexpr int kThreads = 128;
constexpr int kMaxRrefRows = 2 * kMaxRelationRows;
constexpr int kMaxRrefColumns = kMaxDualColumns;
constexpr int kMaxRrefEntries = kMaxRrefRows * kMaxRrefColumns;

__device__ inline double DeviceAbs(double x) { return x < 0.0 ? -x : x; }

__device__ inline bool DeviceFinite(double x) {
  return x >= -DBL_MAX && x <= DBL_MAX;
}

__device__ void SetFailure(DeviceStatus* status, int code, int stage,
                           int detail) {
  if (atomicCAS(&status->code, kDeviceOk, code) == kDeviceOk) {
    status->stage = stage;
    status->detail = detail;
  }
}

// A global failure can be reported by another block at any time.  Sampling it
// independently in every thread before a later __syncthreads() can therefore
// make only part of a block return.  Have one thread sample the flag and
// broadcast a block-uniform decision instead.
__device__ bool BlockEnabled(const DeviceStatus* status, int* enabled) {
  if (threadIdx.x == 0) *enabled = status->code == kDeviceOk;
  __syncthreads();
  return *enabled != 0;
}

__device__ bool BlockEnabled(const int* flag, int* enabled) {
  if (threadIdx.x == 0) *enabled = *flag != 0;
  __syncthreads();
  return *enabled != 0;
}

// Scale each nonzero equation before pivoting, then use partial row pivoting.
// This makes rank decisions invariant to independent equation rescaling while
// retaining the deterministic free-column convention of the CPU RREF path.
__device__ void RrefBlock(double* matrix, int rows, int columns,
                          int pivot_limit, double tolerance, int* pivot_columns,
                          int* pivot_rows, int* rank, int* best_row,
                          double* factors) {
  for (int row = threadIdx.x; row < rows; row += blockDim.x) {
    double scale = 0.0;
    for (int col = 0; col < pivot_limit; ++col) {
      scale = fmax(scale, DeviceAbs(matrix[row * columns + col]));
    }
    if (scale > 0.0) {
      for (int col = 0; col < columns; ++col)
        matrix[row * columns + col] /= scale;
    }
  }
  if (threadIdx.x == 0) *rank = 0;
  __syncthreads();

  for (int col = 0; col < pivot_limit; ++col) {
    if (threadIdx.x == 0) {
      *best_row = -1;
      double best = tolerance;
      for (int row = *rank; row < rows; ++row) {
        const double candidate = DeviceAbs(matrix[row * columns + col]);
        if (candidate > best) {
          best = candidate;
          *best_row = row;
        }
      }
    }
    __syncthreads();
    if (*best_row < 0) continue;

    const int pivot_row = *rank;
    if (*best_row != pivot_row) {
      for (int j = threadIdx.x; j < columns; j += blockDim.x) {
        const double tmp = matrix[pivot_row * columns + j];
        matrix[pivot_row * columns + j] = matrix[*best_row * columns + j];
        matrix[*best_row * columns + j] = tmp;
      }
    }
    __syncthreads();

    const double pivot = matrix[pivot_row * columns + col];
    for (int j = col + threadIdx.x; j < columns; j += blockDim.x) {
      matrix[pivot_row * columns + j] /= pivot;
    }
    __syncthreads();

    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
      factors[row] = row == pivot_row ? 0.0 : matrix[row * columns + col];
    }
    __syncthreads();
    for (int index = threadIdx.x; index < rows * (columns - col);
         index += blockDim.x) {
      const int row = index / (columns - col);
      const int j = col + index % (columns - col);
      if (row != pivot_row) {
        matrix[row * columns + j] -=
            factors[row] * matrix[pivot_row * columns + j];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      pivot_columns[pivot_row] = col;
      pivot_rows[pivot_row] = pivot_row;
      ++(*rank);
    }
    __syncthreads();
    if (*rank == rows) break;
  }

  for (int index = threadIdx.x; index < rows * columns; index += blockDim.x) {
    if (DeviceAbs(matrix[index]) <= tolerance) matrix[index] = 0.0;
  }
  __syncthreads();
}

__device__ bool InconsistentRref(const double* matrix, int rows, int columns,
                                 int lhs_columns, double tolerance) {
  for (int row = 0; row < rows; ++row) {
    bool zero = true;
    for (int col = 0; col < lhs_columns; ++col) {
      if (DeviceAbs(matrix[row * columns + col]) > tolerance) {
        zero = false;
        break;
      }
    }
    if (zero && DeviceAbs(matrix[row * columns + lhs_columns]) > tolerance)
      return true;
  }
  return false;
}

__device__ void ExtractResidualRelation(const double* matrix, int columns,
                                        int rank, const int* pivot_columns,
                                        int eliminated_columns, int left_dim,
                                        int right_dim, Relation* output) {
  if (threadIdx.x == 0) {
    int eliminated_rank = 0;
    while (eliminated_rank < rank &&
           pivot_columns[eliminated_rank] < eliminated_columns) {
      ++eliminated_rank;
    }
    output->left_dim = left_dim;
    output->right_dim = right_dim;
    output->rows = rank - eliminated_rank;
  }
  __syncthreads();
  const int eliminated_rank = rank - output->rows;
  const int outer_dim = left_dim + right_dim;
  for (int index = threadIdx.x; index < output->rows * outer_dim;
       index += blockDim.x) {
    const int row = index / outer_dim;
    const int col = index % outer_dim;
    const double value =
        matrix[(eliminated_rank + row) * columns + eliminated_columns + col];
    if (col < left_dim) {
      output->left[row * kMaxStateDimension + col] = value;
    } else {
      output->right[row * kMaxStateDimension + col - left_dim] = value;
    }
  }
  for (int row = threadIdx.x; row < output->rows; row += blockDim.x) {
    output->rhs[row] = matrix[(eliminated_rank + row) * columns + columns - 1];
  }
  __syncthreads();
}

__global__ void BuildPrimalLeavesKernel(const PackedStage* stages,
                                        int stage_count,
                                        const PackedTerminal* terminal_ptr,
                                        double tolerance, Relation* leaves,
                                        DeviceStatus* status) {
  const int index = blockIdx.x;
  if (index > stage_count) return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled)) return;
  __shared__ double matrix[kMaxRrefEntries];
  __shared__ double factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int rows;
  __shared__ int columns;
  __shared__ int eliminated;
  __shared__ int left_dim;
  __shared__ int right_dim;
  __shared__ int local_ok;

  const PackedTerminal& terminal = *terminal_ptr;

  if (threadIdx.x == 0) {
    if (index == stage_count) {
      rows = terminal.state;
      columns = terminal.n + 1;
      eliminated = 0;
      left_dim = terminal.n;
      right_dim = 0;
    } else {
      const PackedStage& s = stages[index];
      rows = s.mixed + s.state + s.next_n;
      columns = s.m + s.n + s.next_n + 1;
      eliminated = s.m;
      left_dim = s.n;
      right_dim = s.next_n;
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = 0.0;
  __syncthreads();
  if (index == stage_count) {
    for (int linear = threadIdx.x; linear < terminal.state * terminal.n;
         linear += blockDim.x) {
      const int row = linear / terminal.n;
      const int col = linear % terminal.n;
      matrix[row * columns + col] = terminal.E[row * kMaxStateDimension + col];
    }
    for (int row = threadIdx.x; row < terminal.state; row += blockDim.x) {
      matrix[row * columns + terminal.n] = -terminal.e[row];
    }
  } else {
    const PackedStage& s = stages[index];
    for (int linear = threadIdx.x; linear < s.mixed * s.m;
         linear += blockDim.x) {
      const int row = linear / s.m;
      const int col = linear % s.m;
      matrix[row * columns + col] = s.D[row * kMaxControlDimension + col];
    }
    for (int linear = threadIdx.x; linear < s.mixed * s.n;
         linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      matrix[row * columns + s.m + col] = s.C[row * kMaxStateDimension + col];
    }
    for (int row = threadIdx.x; row < s.mixed; row += blockDim.x) {
      matrix[row * columns + columns - 1] = -s.d[row];
    }
    for (int linear = threadIdx.x; linear < s.state * s.n;
         linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      matrix[(s.mixed + row) * columns + s.m + col] =
          s.E[row * kMaxStateDimension + col];
    }
    for (int row = threadIdx.x; row < s.state; row += blockDim.x) {
      matrix[(s.mixed + row) * columns + columns - 1] = -s.e[row];
    }
    const int dynamics_row = s.mixed + s.state;
    for (int linear = threadIdx.x; linear < s.next_n * s.m;
         linear += blockDim.x) {
      const int row = linear / s.m;
      const int col = linear % s.m;
      matrix[(dynamics_row + row) * columns + col] =
          -s.B[row * kMaxControlDimension + col];
    }
    for (int linear = threadIdx.x; linear < s.next_n * s.n;
         linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      matrix[(dynamics_row + row) * columns + s.m + col] =
          -s.A[row * kMaxStateDimension + col];
    }
    for (int row = threadIdx.x; row < s.next_n; row += blockDim.x) {
      matrix[(dynamics_row + row) * columns + s.m + s.n + row] = 1.0;
      matrix[(dynamics_row + row) * columns + columns - 1] = s.c[row];
    }
  }
  __syncthreads();
  RrefBlock(matrix, rows, columns, columns - 1, tolerance, pivot_columns,
            pivot_rows, &rank, &best_row, factors);
  if (threadIdx.x == 0) {
    local_ok = !InconsistentRref(matrix, rows, columns, columns - 1, tolerance);
    if (!local_ok) SetFailure(status, kDeviceInfeasible, index, 1);
  }
  __syncthreads();
  if (!local_ok) return;
  ExtractResidualRelation(matrix, columns, rank, pivot_columns, eliminated,
                          left_dim, right_dim, &leaves[index]);
}

__device__ void ComposeRelationsBlock(const Relation& first,
                                      const Relation& second, double tolerance,
                                      Relation* output, DeviceStatus* status,
                                      int stage, double* matrix,
                                      double* factors, int* pivot_columns,
                                      int* pivot_rows, int* rank, int* best_row,
                                      int* local_ok) {
  if (first.right_dim != second.left_dim) {
    if (threadIdx.x == 0) SetFailure(status, kDeviceNumericalFailure, stage, 2);
    return;
  }
  const int shared = first.right_dim;
  const int rows = first.rows + second.rows;
  const int columns = shared + first.left_dim + second.right_dim + 1;
  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = 0.0;
  __syncthreads();
  for (int linear = threadIdx.x; linear < first.rows * shared;
       linear += blockDim.x) {
    const int row = linear / shared;
    const int col = linear % shared;
    matrix[row * columns + col] = first.right[row * kMaxStateDimension + col];
  }
  for (int linear = threadIdx.x; linear < first.rows * first.left_dim;
       linear += blockDim.x) {
    const int row = linear / first.left_dim;
    const int col = linear % first.left_dim;
    matrix[row * columns + shared + col] =
        first.left[row * kMaxStateDimension + col];
  }
  for (int row = threadIdx.x; row < first.rows; row += blockDim.x) {
    matrix[row * columns + columns - 1] = first.rhs[row];
  }
  for (int linear = threadIdx.x; linear < second.rows * shared;
       linear += blockDim.x) {
    const int row = linear / shared;
    const int col = linear % shared;
    matrix[(first.rows + row) * columns + col] =
        second.left[row * kMaxStateDimension + col];
  }
  for (int linear = threadIdx.x; linear < second.rows * second.right_dim;
       linear += blockDim.x) {
    const int row = linear / second.right_dim;
    const int col = linear % second.right_dim;
    matrix[(first.rows + row) * columns + shared + first.left_dim + col] =
        second.right[row * kMaxStateDimension + col];
  }
  for (int row = threadIdx.x; row < second.rows; row += blockDim.x) {
    matrix[(first.rows + row) * columns + columns - 1] = second.rhs[row];
  }
  __syncthreads();
  RrefBlock(matrix, rows, columns, columns - 1, tolerance, pivot_columns,
            pivot_rows, rank, best_row, factors);
  if (threadIdx.x == 0) {
    *local_ok =
        !InconsistentRref(matrix, rows, columns, columns - 1, tolerance);
    if (!*local_ok) SetFailure(status, kDeviceInfeasible, stage, 3);
  }
  __syncthreads();
  if (!*local_ok) return;
  ExtractResidualRelation(matrix, columns, *rank, pivot_columns, shared,
                          first.left_dim, second.right_dim, output);
}

__device__ void CopyRelationBlock(const Relation& input, Relation* output) {
  if (threadIdx.x == 0) {
    output->left_dim = input.left_dim;
    output->right_dim = input.right_dim;
    output->rows = input.rows;
  }
  for (int linear = threadIdx.x; linear < input.rows * input.left_dim;
       linear += blockDim.x) {
    const int row = linear / input.left_dim;
    const int col = linear % input.left_dim;
    output->left[row * kMaxStateDimension + col] =
        input.left[row * kMaxStateDimension + col];
  }
  for (int linear = threadIdx.x; linear < input.rows * input.right_dim;
       linear += blockDim.x) {
    const int row = linear / input.right_dim;
    const int col = linear % input.right_dim;
    output->right[row * kMaxStateDimension + col] =
        input.right[row * kMaxStateDimension + col];
  }
  for (int row = threadIdx.x; row < input.rows; row += blockDim.x)
    output->rhs[row] = input.rhs[row];
}

__global__ void SuffixRelationsKernel(const Relation* input, int count,
                                      int offset, double tolerance,
                                      Relation* output, DeviceStatus* status) {
  const int index = blockIdx.x;
  if (index >= count) return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled)) return;
  if (index + offset >= count) {
    CopyRelationBlock(input[index], &output[index]);
    return;
  }
  __shared__ double matrix[kMaxRrefEntries];
  __shared__ double factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;
  ComposeRelationsBlock(input[index], input[index + offset], tolerance,
                        &output[index], status, index, matrix, factors,
                        pivot_columns, pivot_rows, &rank, &best_row, &local_ok);
}

__global__ void StateParamKernel(const Relation* suffix, int count,
                                 StateParam* params, DeviceStatus* status,
                                 double tolerance) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count || status->code != kDeviceOk) return;
  const Relation& relation = suffix[index];
  if (relation.right_dim != 0 || relation.rows > relation.left_dim) {
    SetFailure(status, kDeviceNumericalFailure, index, 4);
    return;
  }
  StateParam& out = params[index];
  out.physical_dim = relation.left_dim;
  bool pivot[kMaxStateDimension]{};
  int pivot_row[kMaxStateDimension];
  for (int i = 0; i < kMaxStateDimension; ++i) pivot_row[i] = -1;
  for (int row = 0; row < relation.rows; ++row) {
    int column = -1;
    for (int col = 0; col < relation.left_dim; ++col) {
      if (DeviceAbs(relation.left[row * kMaxStateDimension + col]) >
          tolerance) {
        column = col;
        break;
      }
    }
    if (column < 0 || pivot[column]) {
      SetFailure(status, kDeviceNumericalFailure, index, 5);
      return;
    }
    pivot[column] = true;
    pivot_row[column] = row;
  }
  int reduced = 0;
  for (int col = 0; col < relation.left_dim; ++col) {
    if (!pivot[col]) out.free_columns[reduced++] = col;
  }
  out.reduced_dim = reduced;
  for (int col = 0; col < relation.left_dim; ++col) {
    if (pivot[col]) {
      const int row = pivot_row[col];
      const double diagonal = relation.left[row * kMaxStateDimension + col];
      out.t[col] = relation.rhs[row] / diagonal;
      for (int free = 0; free < reduced; ++free) {
        out.T[col * kMaxStateDimension + free] =
            -relation.left[row * kMaxStateDimension + out.free_columns[free]] /
            diagonal;
      }
    }
  }
  for (int free = 0; free < reduced; ++free) {
    out.T[out.free_columns[free] * kMaxStateDimension + free] = 1.0;
  }
}

}  // namespace
}  // namespace detail
}  // namespace cuda
}  // namespace clqr

#ifndef CLQR_CUDA_EMULATION
namespace clqr {
namespace cuda {
namespace detail {
namespace {

__global__ void ReduceStagesKernel(const PackedStage*, const Relation*,
                                   const StateParam*, int, double,
                                   ControlParam*, ReducedStage*, DeviceStatus*);
__global__ void ReduceTerminalKernel(const PackedTerminal*, const StateParam*,
                                     int, ReducedTerminal*);
__global__ void InitialReducedStateKernel(const StateParam*, const double*,
                                          double*, double, DeviceStatus*);
__global__ void BuildValueElementsKernel(const ReducedStage*,
                                         const ReducedTerminal*, int, double,
                                         ValueElement*, int*, DeviceStatus*);
__global__ void SuffixValueElementsKernel(const ValueElement*, int, int, double,
                                          ValueElement*, int*);
__global__ void FeedbackKernel(const ReducedStage*, const ValueElement*, int,
                               double, Feedback*, DeviceStatus*);
__global__ void SequentialRiccatiKernel(const ReducedStage*, int, double,
                                        ValueElement*, Feedback*,
                                        DeviceStatus*);
__global__ void InitializeAffineMapsKernel(const Feedback*, int, AffineMap*);
__global__ void PrefixAffineMapsKernel(const AffineMap*, int, int, AffineMap*,
                                       DeviceStatus*);
__global__ void EvaluateReducedStatesKernel(const AffineMap*, int,
                                            const double*, double*);
__global__ void ReconstructPrimalKernel(const StateParam*, const ControlParam*,
                                        const Feedback*, const double*, int,
                                        double*, double*);
__global__ void BuildDualLeavesKernel(const PackedStage*, const PackedTerminal*,
                                      int, int, const double*, const double*,
                                      double, Relation*, DeviceStatus*);
__global__ void ReduceDualTreeLevelKernel(const Relation*, int, int, int,
                                          double, Relation*, DeviceStatus*);
__global__ void SolveDualRootKernel(const Relation*, NodeValue*, DeviceStatus*,
                                    double);
__global__ void ExpandDualTreeLevelKernel(const Relation*, int, int, int,
                                          double, const NodeValue*, NodeValue*,
                                          DeviceStatus*);
__global__ void RecoverLocalMultipliersKernel(const PackedStage*,
                                              const PackedTerminal*, int,
                                              const double*, const double*,
                                              const NodeValue*, double, double*,
                                              double*, double*, double*,
                                              double*, DeviceStatus*);

void CudaCheck(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return;
  throw std::runtime_error(std::string(operation) + ": " +
                           cudaGetErrorString(error));
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { Allocate(count); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Release();
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }
  ~DeviceBuffer() { Release(); }

  void Allocate(std::size_t count) {
    Release();
    count_ = std::max<std::size_t>(count, 1);
    CudaCheck(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
              "cudaMalloc");
  }
  void Release() {
    if (data_ != nullptr) cudaFree(data_);
    data_ = nullptr;
    count_ = 0;
  }
  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t count() const { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

double TimeGpu(const std::function<void()>& function) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CudaCheck(cudaEventCreate(&start), "cudaEventCreate(start)");
  try {
    CudaCheck(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    CudaCheck(cudaEventRecord(start), "cudaEventRecord(start)");
    function();
    CudaCheck(cudaGetLastError(), "CUDA kernel launch");
    CudaCheck(cudaEventRecord(stop), "cudaEventRecord(stop)");
    CudaCheck(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float milliseconds = 0.0f;
    CudaCheck(cudaEventElapsedTime(&milliseconds, start, stop),
              "cudaEventElapsedTime");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return milliseconds;
  } catch (...) {
    if (stop != nullptr) cudaEventDestroy(stop);
    cudaEventDestroy(start);
    throw;
  }
}

bool Finite(const Matrix& matrix) {
  for (double value : matrix.data()) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

bool Finite(const Vector& vector) {
  for (double value : vector.data()) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}

void ValidateCudaProblem(const Problem& problem, const Options& options) {
  Require(std::isfinite(options.tolerance) && options.tolerance > 0.0,
          "CUDA tolerance must be finite and positive");
  Require(options.device >= 0, "CUDA device index must be nonnegative");
  // node_count, its next power of two, and the resulting 2*padded-1 tree size
  // are all stored in signed ints.
  Require(problem.stages.size() <=
              static_cast<std::size_t>(std::numeric_limits<int>::max() / 2),
          "too many stages for CUDA tree indices");
  const std::size_t count = problem.stages.size();
  Require(problem.terminal_Q.rows() == problem.terminal_Q.cols(),
          "terminal_Q must be square");
  Require(problem.terminal_Q.rows() <= kMaxStateDimension,
          "terminal state dimension exceeds CUDA limit");
  Require(problem.terminal_q.size() == problem.terminal_Q.rows(),
          "terminal_q shape mismatch");
  Require(problem.terminal_E.cols() == problem.terminal_Q.rows(),
          "terminal_E shape mismatch");
  Require(problem.terminal_E.rows() <= kMaxStateConstraints,
          "terminal constraint count exceeds CUDA limit");
  Require(problem.terminal_e.size() == problem.terminal_E.rows(),
          "terminal_e shape mismatch");
  Require(Finite(problem.terminal_Q) && Finite(problem.terminal_q) &&
              Finite(problem.terminal_E) && Finite(problem.terminal_e) &&
              Finite(problem.initial_state),
          "problem contains a non-finite terminal or initial value");
  if (count == 0) {
    Require(problem.initial_state.size() == problem.terminal_Q.rows(),
            "initial_state and terminal state dimensions differ");
    return;
  }
  Require(problem.initial_state.size() == problem.stages.front().A.cols(),
          "initial_state and first stage dimensions differ");
  for (std::size_t i = 0; i < count; ++i) {
    const Stage& s = problem.stages[i];
    const std::size_t n = s.A.cols();
    const std::size_t next = s.A.rows();
    const std::size_t m = s.B.cols();
    Require(n <= kMaxStateDimension && next <= kMaxStateDimension,
            "state dimension exceeds CUDA limit at stage " + std::to_string(i));
    Require(
        m <= kMaxControlDimension,
        "control dimension exceeds CUDA limit at stage " + std::to_string(i));
    Require(s.C.rows() <= kMaxMixedConstraints,
            "mixed constraint count exceeds CUDA limit at stage " +
                std::to_string(i));
    Require(s.E.rows() <= kMaxStateConstraints,
            "state constraint count exceeds CUDA limit at stage " +
                std::to_string(i));
    Require(s.B.rows() == next && s.c.size() == next,
            "dynamics shape mismatch at stage " + std::to_string(i));
    Require(s.Q.rows() == n && s.Q.cols() == n && s.q.size() == n,
            "state-cost shape mismatch at stage " + std::to_string(i));
    Require(s.R.rows() == m && s.R.cols() == m && s.r.size() == m,
            "control-cost shape mismatch at stage " + std::to_string(i));
    Require(s.M.rows() == n && s.M.cols() == m,
            "cross-cost shape mismatch at stage " + std::to_string(i));
    Require(s.C.cols() == n && s.D.rows() == s.C.rows() && s.D.cols() == m &&
                s.d.size() == s.C.rows(),
            "mixed-constraint shape mismatch at stage " + std::to_string(i));
    Require(s.E.cols() == n && s.e.size() == s.E.rows(),
            "state-constraint shape mismatch at stage " + std::to_string(i));
    const std::size_t expected_next = i + 1 == count
                                          ? problem.terminal_Q.rows()
                                          : problem.stages[i + 1].A.cols();
    Require(
        next == expected_next,
        "neighboring state dimensions differ at stage " + std::to_string(i));
    Require(
        Finite(s.A) && Finite(s.B) && Finite(s.c) && Finite(s.Q) &&
            Finite(s.R) && Finite(s.M) && Finite(s.q) && Finite(s.r) &&
            Finite(s.C) && Finite(s.D) && Finite(s.d) && Finite(s.E) &&
            Finite(s.e),
        "problem contains a non-finite value at stage " + std::to_string(i));
  }
}

template <std::size_t Size>
void PackMatrix(const Matrix& source, double (&target)[Size],
                std::size_t stride) {
  for (std::size_t row = 0; row < source.rows(); ++row) {
    for (std::size_t col = 0; col < source.cols(); ++col) {
      target[row * stride + col] = source(row, col);
    }
  }
}

template <std::size_t Size>
void PackVector(const Vector& source, double (&target)[Size]) {
  for (std::size_t i = 0; i < source.size(); ++i) target[i] = source[i];
}

PackedStage PackStage(const Stage& source) {
  PackedStage out;
  out.n = static_cast<int>(source.A.cols());
  out.next_n = static_cast<int>(source.A.rows());
  out.m = static_cast<int>(source.B.cols());
  out.mixed = static_cast<int>(source.C.rows());
  out.state = static_cast<int>(source.E.rows());
  PackMatrix(source.A, out.A, kMaxStateDimension);
  PackMatrix(source.B, out.B, kMaxControlDimension);
  PackVector(source.c, out.c);
  PackMatrix(source.Q, out.Q, kMaxStateDimension);
  PackMatrix(source.R, out.R, kMaxControlDimension);
  PackMatrix(source.M, out.M, kMaxControlDimension);
  PackVector(source.q, out.q);
  PackVector(source.r, out.r);
  PackMatrix(source.C, out.C, kMaxStateDimension);
  PackMatrix(source.D, out.D, kMaxControlDimension);
  PackVector(source.d, out.d);
  PackMatrix(source.E, out.E, kMaxStateDimension);
  PackVector(source.e, out.e);
  return out;
}

PackedTerminal PackTerminal(const Problem& problem) {
  PackedTerminal out;
  out.n = static_cast<int>(problem.terminal_Q.rows());
  out.state = static_cast<int>(problem.terminal_E.rows());
  PackMatrix(problem.terminal_Q, out.Q, kMaxStateDimension);
  PackVector(problem.terminal_q, out.q);
  PackMatrix(problem.terminal_E, out.E, kMaxStateDimension);
  PackVector(problem.terminal_e, out.e);
  return out;
}

DeviceStatus ReadStatus(const DeviceBuffer<DeviceStatus>& status) {
  DeviceStatus host;
  CudaCheck(
      cudaMemcpy(&host, status.get(), sizeof(host), cudaMemcpyDeviceToHost),
      "copy CUDA status");
  return host;
}

std::string DeviceFailureMessage(const DeviceStatus& status) {
  std::ostringstream out;
  if (status.code == kDeviceInfeasible) {
    out << "CUDA feasibility elimination found an inconsistent relation";
  } else {
    out << "CUDA backend encountered a rank or consistency failure";
  }
  if (status.stage >= 0) out << " at stage/node " << status.stage;
  out << " (diagnostic " << status.detail << ")";
  return out.str();
}

bool ApplyDeviceFailure(const DeviceStatus& status, Solution* solution) {
  if (status.code == kDeviceOk) return false;
  solution->status = status.code == kDeviceInfeasible
                         ? SolveStatus::kInfeasible
                         : SolveStatus::kNumericalFailure;
  solution->message = DeviceFailureMessage(status);
  return true;
}

double ObjectiveFromPacked(const std::vector<PackedStage>& stages,
                           const PackedTerminal& terminal,
                           const std::vector<double>& states,
                           const std::vector<double>& controls) {
  double objective = 0.0;
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const PackedStage& s = stages[i];
    const double* x = states.data() + i * kMaxStateDimension;
    const double* u = controls.data() + i * kMaxControlDimension;
    for (int row = 0; row < s.n; ++row) {
      objective += s.q[row] * x[row];
      for (int col = 0; col < s.n; ++col)
        objective +=
            0.5 * x[row] * s.Q[row * kMaxStateDimension + col] * x[col];
      for (int col = 0; col < s.m; ++col)
        objective += x[row] * s.M[row * kMaxControlDimension + col] * u[col];
    }
    for (int row = 0; row < s.m; ++row) {
      objective += s.r[row] * u[row];
      for (int col = 0; col < s.m; ++col)
        objective +=
            0.5 * u[row] * s.R[row * kMaxControlDimension + col] * u[col];
    }
  }
  const double* x = states.data() + stages.size() * kMaxStateDimension;
  for (int row = 0; row < terminal.n; ++row) {
    objective += terminal.q[row] * x[row];
    for (int col = 0; col < terminal.n; ++col)
      objective +=
          0.5 * x[row] * terminal.Q[row * kMaxStateDimension + col] * x[col];
  }
  return objective;
}

Solution SolveImpl(const Problem& problem, const Options& options) {
  ValidateCudaProblem(problem, options);
  int device_count = 0;
  CudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  Require(options.device < device_count, "CUDA device index is out of range");
  CudaCheck(cudaSetDevice(options.device), "cudaSetDevice");
  const auto total_start = std::chrono::steady_clock::now();
  Solution solution;
  const int stage_count = static_cast<int>(problem.stages.size());
  const int node_count = stage_count + 1;
  std::vector<PackedStage> host_stages;
  host_stages.reserve(problem.stages.size());
  for (const Stage& stage : problem.stages)
    host_stages.push_back(PackStage(stage));
  const PackedTerminal terminal = PackTerminal(problem);
  std::vector<double> host_initial(kMaxStateDimension, 0.0);
  for (std::size_t i = 0; i < problem.initial_state.size(); ++i)
    host_initial[i] = problem.initial_state[i];

  DeviceBuffer<PackedStage> device_stages(stage_count);
  DeviceBuffer<PackedTerminal> device_terminal(1);
  DeviceBuffer<double> device_initial(kMaxStateDimension);
  DeviceBuffer<DeviceStatus> device_status(1);
  DeviceBuffer<Relation> relation_a(node_count);
  DeviceBuffer<Relation> relation_b(node_count);
  DeviceBuffer<StateParam> state_params(node_count);
  DeviceBuffer<ControlParam> control_params(stage_count);
  DeviceBuffer<ReducedStage> reduced_stages(stage_count);
  DeviceBuffer<ReducedTerminal> reduced_terminal(1);
  DeviceBuffer<double> reduced_initial(kMaxStateDimension);

  solution.timings.upload_ms = TimeGpu([&] {
    if (stage_count > 0) {
      CudaCheck(cudaMemcpy(device_stages.get(), host_stages.data(),
                           host_stages.size() * sizeof(PackedStage),
                           cudaMemcpyHostToDevice),
                "upload stages");
    }
    CudaCheck(cudaMemcpy(device_terminal.get(), &terminal,
                         sizeof(PackedTerminal), cudaMemcpyHostToDevice),
              "upload terminal data");
    CudaCheck(
        cudaMemcpy(device_initial.get(), host_initial.data(),
                   kMaxStateDimension * sizeof(double), cudaMemcpyHostToDevice),
        "upload initial state");
    CudaCheck(cudaMemset(device_status.get(), 0, sizeof(DeviceStatus)),
              "clear CUDA status");
    CudaCheck(cudaMemset(relation_a.get(), 0, node_count * sizeof(Relation)),
              "clear primal relations A");
    CudaCheck(cudaMemset(relation_b.get(), 0, node_count * sizeof(Relation)),
              "clear primal relations B");
    CudaCheck(
        cudaMemset(state_params.get(), 0, node_count * sizeof(StateParam)),
        "clear state parameters");
    if (stage_count > 0) {
      CudaCheck(cudaMemset(control_params.get(), 0,
                           stage_count * sizeof(ControlParam)),
                "clear control parameters");
      CudaCheck(cudaMemset(reduced_stages.get(), 0,
                           stage_count * sizeof(ReducedStage)),
                "clear reduced stages");
    }
    CudaCheck(cudaMemset(reduced_terminal.get(), 0, sizeof(ReducedTerminal)),
              "clear reduced terminal");
    CudaCheck(cudaMemset(reduced_initial.get(), 0,
                         kMaxStateDimension * sizeof(double)),
              "clear reduced initial state");
  });

  Relation* suffix = relation_a.get();
  solution.timings.feasibility_ms = TimeGpu([&] {
    BuildPrimalLeavesKernel<<<node_count, kThreads>>>(
        device_stages.get(), stage_count, device_terminal.get(),
        options.tolerance, relation_a.get(), device_status.get());
    Relation* input = relation_a.get();
    Relation* output = relation_b.get();
    for (int offset = 1; offset < node_count; offset *= 2) {
      CudaCheck(cudaMemset(output, 0, node_count * sizeof(Relation)),
                "clear suffix relation output");
      SuffixRelationsKernel<<<node_count, kThreads>>>(input, node_count, offset,
                                                      options.tolerance, output,
                                                      device_status.get());
      std::swap(input, output);
    }
    suffix = input;
    const int blocks = (node_count + kThreads - 1) / kThreads;
    StateParamKernel<<<blocks, kThreads>>>(
        suffix, node_count, state_params.get(), device_status.get(),
        options.tolerance);
  });
  DeviceStatus status = ReadStatus(device_status);
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }

  solution.timings.reduction_ms = TimeGpu([&] {
    if (stage_count > 0) {
      ReduceStagesKernel<<<stage_count, kThreads>>>(
          device_stages.get(), suffix, state_params.get(), stage_count,
          options.tolerance, control_params.get(), reduced_stages.get(),
          device_status.get());
    }
    ReduceTerminalKernel<<<1, kThreads>>>(device_terminal.get(),
                                          state_params.get(), stage_count,
                                          reduced_terminal.get());
    InitialReducedStateKernel<<<1, kThreads>>>(
        state_params.get(), device_initial.get(), reduced_initial.get(),
        options.tolerance, device_status.get());
  });
  status = ReadStatus(device_status);
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }

  DeviceBuffer<ValueElement> value_a(node_count);
  DeviceBuffer<ValueElement> value_b(node_count);
  DeviceBuffer<Feedback> feedback(stage_count);
  DeviceBuffer<int> parallel_ok(1);
  ValueElement* value_suffix = value_a.get();
  int host_parallel_ok = 1;
  solution.timings.riccati_ms = TimeGpu([&] {
    CudaCheck(cudaMemset(value_a.get(), 0, node_count * sizeof(ValueElement)),
              "clear value elements A");
    CudaCheck(cudaMemset(value_b.get(), 0, node_count * sizeof(ValueElement)),
              "clear value elements B");
    if (stage_count > 0)
      CudaCheck(cudaMemset(feedback.get(), 0, stage_count * sizeof(Feedback)),
                "clear feedback");
    CudaCheck(cudaMemcpy(parallel_ok.get(), &host_parallel_ok, sizeof(int),
                         cudaMemcpyHostToDevice),
              "initialize parallel Riccati flag");
    BuildValueElementsKernel<<<node_count, kThreads>>>(
        reduced_stages.get(), reduced_terminal.get(), stage_count,
        options.tolerance, value_a.get(), parallel_ok.get(),
        device_status.get());
  });
  CudaCheck(cudaMemcpy(&host_parallel_ok, parallel_ok.get(), sizeof(int),
                       cudaMemcpyDeviceToHost),
            "read parallel Riccati flag");

  if (host_parallel_ok != 0) {
    solution.timings.riccati_ms += TimeGpu([&] {
      ValueElement* input = value_a.get();
      ValueElement* output = value_b.get();
      for (int offset = 1; offset < node_count; offset *= 2) {
        CudaCheck(cudaMemset(output, 0, node_count * sizeof(ValueElement)),
                  "clear value suffix output");
        SuffixValueElementsKernel<<<node_count, kThreads>>>(
            input, node_count, offset, options.tolerance, output,
            parallel_ok.get());
        std::swap(input, output);
      }
      value_suffix = input;
    });
    CudaCheck(cudaMemcpy(&host_parallel_ok, parallel_ok.get(), sizeof(int),
                         cudaMemcpyDeviceToHost),
              "read value scan status");
  }

  if (host_parallel_ok != 0) {
    solution.used_parallel_riccati = true;
    solution.timings.riccati_ms += TimeGpu([&] {
      if (stage_count > 0) {
        FeedbackKernel<<<stage_count, kThreads>>>(
            reduced_stages.get(), value_suffix, stage_count, options.tolerance,
            feedback.get(), device_status.get());
      }
    });
  } else {
    solution.used_parallel_riccati = false;
    solution.timings.riccati_ms += TimeGpu([&] {
      CudaCheck(cudaMemset(value_a.get(), 0, node_count * sizeof(ValueElement)),
                "clear sequential value storage");
      host_parallel_ok = 1;
      CudaCheck(cudaMemcpy(parallel_ok.get(), &host_parallel_ok, sizeof(int),
                           cudaMemcpyHostToDevice),
                "reset Riccati flag");
      BuildValueElementsKernel<<<node_count, kThreads>>>(
          reduced_stages.get(), reduced_terminal.get(), stage_count,
          options.tolerance, value_a.get(), parallel_ok.get(),
          device_status.get());
      if (stage_count > 0) {
        SequentialRiccatiKernel<<<1, kThreads>>>(
            reduced_stages.get(), stage_count, options.tolerance, value_a.get(),
            feedback.get(), device_status.get());
      }
      value_suffix = value_a.get();
    });
  }
  status = ReadStatus(device_status);
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }

  DeviceBuffer<AffineMap> map_a(stage_count);
  DeviceBuffer<AffineMap> map_b(stage_count);
  DeviceBuffer<double> reduced_states(static_cast<std::size_t>(node_count) *
                                      kMaxStateDimension);
  DeviceBuffer<double> states(static_cast<std::size_t>(node_count) *
                              kMaxStateDimension);
  DeviceBuffer<double> controls(
      static_cast<std::size_t>(std::max(stage_count, 1)) *
      kMaxControlDimension);
  AffineMap* prefix = map_a.get();
  solution.timings.reconstruction_ms = TimeGpu([&] {
    CudaCheck(cudaMemset(reduced_states.get(), 0,
                         static_cast<std::size_t>(node_count) *
                             kMaxStateDimension * sizeof(double)),
              "clear reduced states");
    CudaCheck(cudaMemset(states.get(), 0,
                         static_cast<std::size_t>(node_count) *
                             kMaxStateDimension * sizeof(double)),
              "clear physical states");
    CudaCheck(cudaMemset(controls.get(), 0,
                         static_cast<std::size_t>(std::max(stage_count, 1)) *
                             kMaxControlDimension * sizeof(double)),
              "clear physical controls");
    if (stage_count > 0) {
      CudaCheck(cudaMemset(map_a.get(), 0, stage_count * sizeof(AffineMap)),
                "clear affine maps A");
      CudaCheck(cudaMemset(map_b.get(), 0, stage_count * sizeof(AffineMap)),
                "clear affine maps B");
      InitializeAffineMapsKernel<<<stage_count, kThreads>>>(
          feedback.get(), stage_count, map_a.get());
      AffineMap* input = map_a.get();
      AffineMap* output = map_b.get();
      for (int offset = 1; offset < stage_count; offset *= 2) {
        CudaCheck(cudaMemset(output, 0, stage_count * sizeof(AffineMap)),
                  "clear affine prefix output");
        PrefixAffineMapsKernel<<<stage_count, kThreads>>>(
            input, stage_count, offset, output, device_status.get());
        std::swap(input, output);
      }
      prefix = input;
    }
    const int state_blocks = (node_count + kThreads - 1) / kThreads;
    EvaluateReducedStatesKernel<<<state_blocks, kThreads>>>(
        prefix, stage_count, reduced_initial.get(), reduced_states.get());
    ReconstructPrimalKernel<<<node_count, kThreads>>>(
        state_params.get(), control_params.get(), feedback.get(),
        reduced_states.get(), stage_count, states.get(), controls.get());
  });
  status = ReadStatus(device_status);
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }

  // Multiplier recovery uses a balanced relation tree. It chooses zero for
  // genuinely free multiplier components, which is sufficient even when the
  // original equality rows are redundant.
  int padded = 1;
  while (padded < node_count) padded *= 2;
  std::vector<int> level_offsets{0};
  std::vector<int> level_counts{padded};
  int total_nodes = padded;
  while (level_counts.back() > 1) {
    level_offsets.push_back(total_nodes);
    level_counts.push_back(level_counts.back() / 2);
    total_nodes += level_counts.back();
  }
  DeviceBuffer<Relation> dual_tree(total_nodes);
  DeviceBuffer<NodeValue> dual_values(total_nodes);
  DeviceBuffer<double> initial_multiplier(kMaxStateDimension);
  DeviceBuffer<double> dynamics_multipliers(
      static_cast<std::size_t>(std::max(stage_count, 1)) * kMaxStateDimension);
  DeviceBuffer<double> mixed_multipliers(
      static_cast<std::size_t>(std::max(stage_count, 1)) *
      kMaxMixedConstraints);
  DeviceBuffer<double> state_multipliers(
      static_cast<std::size_t>(std::max(stage_count, 1)) *
      kMaxStateConstraints);
  DeviceBuffer<double> terminal_multiplier(kMaxStateConstraints);
  solution.timings.multiplier_ms = TimeGpu([&] {
    CudaCheck(cudaMemset(dual_tree.get(), 0, total_nodes * sizeof(Relation)),
              "clear dual relation tree");
    CudaCheck(cudaMemset(dual_values.get(), 0, total_nodes * sizeof(NodeValue)),
              "clear dual tree values");
    CudaCheck(cudaMemset(initial_multiplier.get(), 0,
                         kMaxStateDimension * sizeof(double)),
              "clear initial multiplier");
    CudaCheck(cudaMemset(dynamics_multipliers.get(), 0,
                         static_cast<std::size_t>(std::max(stage_count, 1)) *
                             kMaxStateDimension * sizeof(double)),
              "clear dynamics multipliers");
    CudaCheck(cudaMemset(mixed_multipliers.get(), 0,
                         static_cast<std::size_t>(std::max(stage_count, 1)) *
                             kMaxMixedConstraints * sizeof(double)),
              "clear mixed multipliers");
    CudaCheck(cudaMemset(state_multipliers.get(), 0,
                         static_cast<std::size_t>(std::max(stage_count, 1)) *
                             kMaxStateConstraints * sizeof(double)),
              "clear state multipliers");
    CudaCheck(cudaMemset(terminal_multiplier.get(), 0,
                         kMaxStateConstraints * sizeof(double)),
              "clear terminal multipliers");
    BuildDualLeavesKernel<<<padded, kThreads>>>(
        device_stages.get(), device_terminal.get(), stage_count, padded,
        states.get(), controls.get(), options.tolerance, dual_tree.get(),
        device_status.get());
    for (std::size_t level = 0; level + 1 < level_counts.size(); ++level) {
      ReduceDualTreeLevelKernel<<<level_counts[level + 1], kThreads>>>(
          dual_tree.get(), level_offsets[level], level_offsets[level + 1],
          level_counts[level + 1], options.tolerance, dual_tree.get(),
          device_status.get());
    }
    const int root_offset = level_offsets.back();
    SolveDualRootKernel<<<1, 1>>>(dual_tree.get() + root_offset,
                                  dual_values.get() + root_offset,
                                  device_status.get(), options.tolerance);
    for (int level = static_cast<int>(level_counts.size()) - 2; level >= 0;
         --level) {
      ExpandDualTreeLevelKernel<<<level_counts[level + 1], kThreads>>>(
          dual_tree.get(), level_offsets[level], level_offsets[level + 1],
          level_counts[level + 1], options.tolerance, dual_values.get(),
          dual_values.get(), device_status.get());
    }
    RecoverLocalMultipliersKernel<<<node_count, kThreads>>>(
        device_stages.get(), device_terminal.get(), stage_count, states.get(),
        controls.get(), dual_values.get(), options.tolerance,
        initial_multiplier.get(), dynamics_multipliers.get(),
        mixed_multipliers.get(), state_multipliers.get(),
        terminal_multiplier.get(), device_status.get());
  });
  status = ReadStatus(device_status);
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }

  std::vector<double> host_states(static_cast<std::size_t>(node_count) *
                                  kMaxStateDimension);
  std::vector<double> host_controls(
      static_cast<std::size_t>(std::max(stage_count, 1)) *
      kMaxControlDimension);
  std::vector<double> host_initial_multiplier(kMaxStateDimension);
  std::vector<double> host_dynamics(
      static_cast<std::size_t>(std::max(stage_count, 1)) * kMaxStateDimension);
  std::vector<double> host_mixed(
      static_cast<std::size_t>(std::max(stage_count, 1)) *
      kMaxMixedConstraints);
  std::vector<double> host_state_multipliers(
      static_cast<std::size_t>(std::max(stage_count, 1)) *
      kMaxStateConstraints);
  std::vector<double> host_terminal_multiplier(kMaxStateConstraints);
  std::vector<StateParam> host_state_params(node_count);
  std::vector<ControlParam> host_control_params(stage_count);
  solution.timings.download_ms = TimeGpu([&] {
    CudaCheck(
        cudaMemcpy(host_states.data(), states.get(),
                   host_states.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "download states");
    CudaCheck(cudaMemcpy(host_controls.data(), controls.get(),
                         host_controls.size() * sizeof(double),
                         cudaMemcpyDeviceToHost),
              "download controls");
    CudaCheck(
        cudaMemcpy(host_initial_multiplier.data(), initial_multiplier.get(),
                   host_initial_multiplier.size() * sizeof(double),
                   cudaMemcpyDeviceToHost),
        "download initial multiplier");
    CudaCheck(cudaMemcpy(host_dynamics.data(), dynamics_multipliers.get(),
                         host_dynamics.size() * sizeof(double),
                         cudaMemcpyDeviceToHost),
              "download dynamics multipliers");
    CudaCheck(
        cudaMemcpy(host_mixed.data(), mixed_multipliers.get(),
                   host_mixed.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "download mixed multipliers");
    CudaCheck(cudaMemcpy(host_state_multipliers.data(), state_multipliers.get(),
                         host_state_multipliers.size() * sizeof(double),
                         cudaMemcpyDeviceToHost),
              "download state multipliers");
    CudaCheck(
        cudaMemcpy(host_terminal_multiplier.data(), terminal_multiplier.get(),
                   host_terminal_multiplier.size() * sizeof(double),
                   cudaMemcpyDeviceToHost),
        "download terminal multiplier");
    CudaCheck(cudaMemcpy(host_state_params.data(), state_params.get(),
                         host_state_params.size() * sizeof(StateParam),
                         cudaMemcpyDeviceToHost),
              "download state dimensions");
    if (stage_count > 0) {
      CudaCheck(cudaMemcpy(host_control_params.data(), control_params.get(),
                           host_control_params.size() * sizeof(ControlParam),
                           cudaMemcpyDeviceToHost),
                "download control dimensions");
    }
  });

  solution.states.resize(node_count);
  solution.reduced_state_dimensions.resize(node_count);
  for (int i = 0; i < node_count; ++i) {
    const int n = host_state_params[i].physical_dim;
    solution.states[i] = Vector(n);
    for (int row = 0; row < n; ++row)
      solution.states[i][row] =
          host_states[static_cast<std::size_t>(i) * kMaxStateDimension + row];
    solution.reduced_state_dimensions[i] = host_state_params[i].reduced_dim;
  }
  solution.controls.resize(stage_count);
  solution.reduced_control_dimensions.resize(stage_count);
  solution.dynamics_multipliers.resize(stage_count);
  solution.mixed_multipliers.resize(stage_count);
  solution.state_multipliers.resize(stage_count);
  for (int i = 0; i < stage_count; ++i) {
    const PackedStage& s = host_stages[i];
    solution.controls[i] = Vector(s.m);
    for (int row = 0; row < s.m; ++row)
      solution.controls[i][row] =
          host_controls[static_cast<std::size_t>(i) * kMaxControlDimension +
                        row];
    solution.reduced_control_dimensions[i] = host_control_params[i].reduced_dim;
    solution.dynamics_multipliers[i] = Vector(s.next_n);
    for (int row = 0; row < s.next_n; ++row)
      solution.dynamics_multipliers[i][row] =
          host_dynamics[static_cast<std::size_t>(i) * kMaxStateDimension + row];
    solution.mixed_multipliers[i] = Vector(s.mixed);
    for (int row = 0; row < s.mixed; ++row)
      solution.mixed_multipliers[i][row] =
          host_mixed[static_cast<std::size_t>(i) * kMaxMixedConstraints + row];
    solution.state_multipliers[i] = Vector(s.state);
    for (int row = 0; row < s.state; ++row)
      solution.state_multipliers[i][row] =
          host_state_multipliers[static_cast<std::size_t>(i) *
                                     kMaxStateConstraints +
                                 row];
  }
  solution.initial_multiplier = Vector(problem.initial_state.size());
  for (std::size_t row = 0; row < problem.initial_state.size(); ++row)
    solution.initial_multiplier[row] = host_initial_multiplier[row];
  solution.terminal_state_multiplier = Vector(terminal.state);
  for (int row = 0; row < terminal.state; ++row)
    solution.terminal_state_multiplier[row] = host_terminal_multiplier[row];
  solution.objective =
      ObjectiveFromPacked(host_stages, terminal, host_states, host_controls);
  solution.status = SolveStatus::kOptimal;
  solution.message = solution.used_parallel_riccati
                         ? "optimal (parallel CUDA Riccati scan)"
                         : "optimal (CUDA sequential Riccati fallback)";
  solution.timings.total_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - total_start)
          .count();
  return solution;
}

}  // namespace
}  // namespace detail

bool Available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string DeviceDescription(int device) {
  cudaDeviceProp properties{};
  const cudaError_t error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess)
    return std::string("CUDA unavailable: ") + cudaGetErrorString(error);
  std::ostringstream out;
  out << properties.name << " (compute " << properties.major << "."
      << properties.minor << ", "
      << static_cast<double>(properties.totalGlobalMem) / (1024.0 * 1024.0)
      << " MiB)";
  return out.str();
}

Solution Solve(const Problem& problem, const Options& options) {
  try {
    if (!Available()) {
      Solution out;
      out.status = SolveStatus::kInvalidInput;
      out.message = "no CUDA device is available";
      return out;
    }
    return detail::SolveImpl(problem, options);
  } catch (const std::invalid_argument& error) {
    Solution out;
    out.status = SolveStatus::kInvalidInput;
    out.message = error.what();
    return out;
  } catch (const std::exception& error) {
    Solution out;
    out.status = SolveStatus::kNumericalFailure;
    out.message = error.what();
    return out;
  }
}

}  // namespace cuda
}  // namespace clqr
#endif  // CLQR_CUDA_EMULATION

namespace clqr {
namespace cuda {
namespace detail {
namespace {

__global__ void InitializeAffineMapsKernel(const Feedback* feedback, int count,
                                           AffineMap* maps) {
  const int index = blockIdx.x;
  if (index >= count) return;
  const Feedback& fb = feedback[index];
  if (threadIdx.x == 0) {
    maps[index].left_dim = fb.state_dim;
    maps[index].right_dim = fb.next_state_dim;
  }
  for (int linear = threadIdx.x; linear < fb.next_state_dim * fb.state_dim;
       linear += blockDim.x) {
    const int row = linear / fb.state_dim;
    const int col = linear % fb.state_dim;
    maps[index].linear[row * kMaxStateDimension + col] =
        fb.transition[row * kMaxStateDimension + col];
  }
  for (int row = threadIdx.x; row < fb.next_state_dim; row += blockDim.x)
    maps[index].offset[row] = fb.offset[row];
}

__global__ void PrefixAffineMapsKernel(const AffineMap* input, int count,
                                       int offset, AffineMap* output,
                                       DeviceStatus* status) {
  const int index = blockIdx.x;
  if (index >= count || status->code != kDeviceOk) return;
  if (index < offset) {
    const AffineMap& map = input[index];
    if (threadIdx.x == 0) {
      output[index].left_dim = map.left_dim;
      output[index].right_dim = map.right_dim;
    }
    for (int linear = threadIdx.x; linear < map.right_dim * map.left_dim;
         linear += blockDim.x) {
      const int row = linear / map.left_dim;
      const int col = linear % map.left_dim;
      output[index].linear[row * kMaxStateDimension + col] =
          map.linear[row * kMaxStateDimension + col];
    }
    for (int row = threadIdx.x; row < map.right_dim; row += blockDim.x)
      output[index].offset[row] = map.offset[row];
    return;
  }
  const AffineMap& first = input[index - offset];
  const AffineMap& second = input[index];
  if (first.right_dim != second.left_dim) {
    SetFailure(status, kDeviceNumericalFailure, index, 11);
    return;
  }
  if (threadIdx.x == 0) {
    output[index].left_dim = first.left_dim;
    output[index].right_dim = second.right_dim;
  }
  for (int linear = threadIdx.x; linear < second.right_dim * first.left_dim;
       linear += blockDim.x) {
    const int row = linear / first.left_dim;
    const int col = linear % first.left_dim;
    double value = 0.0;
    for (int k = 0; k < first.right_dim; ++k) {
      value += second.linear[row * kMaxStateDimension + k] *
               first.linear[k * kMaxStateDimension + col];
    }
    output[index].linear[row * kMaxStateDimension + col] = value;
  }
  for (int row = threadIdx.x; row < second.right_dim; row += blockDim.x) {
    double value = second.offset[row];
    for (int k = 0; k < first.right_dim; ++k) {
      value += second.linear[row * kMaxStateDimension + k] * first.offset[k];
    }
    output[index].offset[row] = value;
  }
}

__global__ void EvaluateReducedStatesKernel(const AffineMap* prefix,
                                            int stage_count,
                                            const double* initial,
                                            double* reduced_states) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > stage_count) return;
  if (index == 0) {
    for (int col = 0; col < kMaxStateDimension; ++col)
      reduced_states[col] = initial[col];
    return;
  }
  const AffineMap& map = prefix[index - 1];
  for (int row = 0; row < map.right_dim; ++row) {
    double value = map.offset[row];
    for (int col = 0; col < map.left_dim; ++col) {
      value += map.linear[row * kMaxStateDimension + col] * initial[col];
    }
    reduced_states[index * kMaxStateDimension + row] = value;
  }
}

__global__ void ReconstructPrimalKernel(const StateParam* state_params,
                                        const ControlParam* control_params,
                                        const Feedback* feedback,
                                        const double* reduced_states,
                                        int stage_count, double* states,
                                        double* controls) {
  const int index = blockIdx.x;
  if (index > stage_count) return;
  const StateParam& state = state_params[index];
  const double* z = reduced_states + index * kMaxStateDimension;
  for (int x = threadIdx.x; x < state.physical_dim; x += blockDim.x) {
    double value = state.t[x];
    for (int col = 0; col < state.reduced_dim; ++col) {
      value += state.T[x * kMaxStateDimension + col] * z[col];
    }
    states[index * kMaxStateDimension + x] = value;
  }
  if (index == stage_count) return;
  const ControlParam& control = control_params[index];
  const Feedback& fb = feedback[index];
  __shared__ double v[kMaxControlDimension];
  for (int row = threadIdx.x; row < control.reduced_dim; row += blockDim.x) {
    double value = fb.k[row];
    for (int col = 0; col < fb.state_dim; ++col) {
      value += fb.K[row * kMaxStateDimension + col] * z[col];
    }
    v[row] = value;
  }
  __syncthreads();
  for (int u = threadIdx.x; u < control.physical_dim; u += blockDim.x) {
    double value = control.y[u];
    for (int col = 0; col < control.state_dim; ++col) {
      value += control.Y[u * kMaxStateDimension + col] * z[col];
    }
    for (int col = 0; col < control.reduced_dim; ++col) {
      value += control.Z[u * kMaxControlDimension + col] * v[col];
    }
    controls[index * kMaxControlDimension + u] = value;
  }
}

__global__ void BuildDualLeavesKernel(const PackedStage* stages,
                                      const PackedTerminal* terminal_ptr,
                                      int stage_count, int padded_count,
                                      const double* states,
                                      const double* controls, double tolerance,
                                      Relation* tree, DeviceStatus* status) {
  const int index = blockIdx.x;
  if (index >= padded_count) return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled)) return;
  if (index > stage_count) {
    if (threadIdx.x == 0) {
      tree[index].left_dim = 0;
      tree[index].right_dim = 0;
      tree[index].rows = 0;
    }
    return;
  }
  __shared__ double matrix[kMaxRrefEntries];
  __shared__ double factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int rows;
  __shared__ int columns;
  __shared__ int eliminated;
  __shared__ int left_dim;
  __shared__ int right_dim;
  __shared__ int local_ok;
  const PackedTerminal& terminal = *terminal_ptr;
  if (threadIdx.x == 0) {
    if (index == stage_count) {
      rows = terminal.n;
      eliminated = terminal.state;
      left_dim = terminal.n;
      right_dim = 0;
      columns = eliminated + left_dim + 1;
    } else {
      const PackedStage& s = stages[index];
      rows = s.n + s.m;
      eliminated = s.mixed + s.state;
      left_dim = s.n;
      right_dim = s.next_n;
      columns = eliminated + left_dim + right_dim + 1;
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = 0.0;
  __syncthreads();

  if (index == stage_count) {
    const double* x = states + index * kMaxStateDimension;
    for (int linear = threadIdx.x; linear < terminal.n * terminal.state;
         linear += blockDim.x) {
      const int row = linear / terminal.state;
      const int constraint = linear % terminal.state;
      matrix[row * columns + constraint] =
          terminal.E[constraint * kMaxStateDimension + row];
    }
    for (int row = threadIdx.x; row < terminal.n; row += blockDim.x) {
      matrix[row * columns + eliminated + row] = 1.0;
      double gradient = terminal.q[row];
      for (int col = 0; col < terminal.n; ++col) {
        gradient += terminal.Q[row * kMaxStateDimension + col] * x[col];
      }
      matrix[row * columns + columns - 1] = -gradient;
    }
  } else {
    const PackedStage& s = stages[index];
    const double* x = states + index * kMaxStateDimension;
    const double* u = controls + index * kMaxControlDimension;
    for (int linear = threadIdx.x; linear < s.n * s.mixed;
         linear += blockDim.x) {
      const int row = linear / s.mixed;
      const int constraint = linear % s.mixed;
      matrix[row * columns + constraint] =
          s.C[constraint * kMaxStateDimension + row];
    }
    for (int linear = threadIdx.x; linear < s.n * s.state;
         linear += blockDim.x) {
      const int row = linear / s.state;
      const int constraint = linear % s.state;
      matrix[row * columns + s.mixed + constraint] =
          s.E[constraint * kMaxStateDimension + row];
    }
    for (int linear = threadIdx.x; linear < s.m * s.mixed;
         linear += blockDim.x) {
      const int row = linear / s.mixed;
      const int constraint = linear % s.mixed;
      matrix[(s.n + row) * columns + constraint] =
          s.D[constraint * kMaxControlDimension + row];
    }
    for (int row = threadIdx.x; row < s.n; row += blockDim.x) {
      matrix[row * columns + eliminated + row] = 1.0;
      for (int next = 0; next < s.next_n; ++next) {
        matrix[row * columns + eliminated + s.n + next] =
            -s.A[next * kMaxStateDimension + row];
      }
      double gradient = s.q[row];
      for (int col = 0; col < s.n; ++col)
        gradient += s.Q[row * kMaxStateDimension + col] * x[col];
      for (int col = 0; col < s.m; ++col)
        gradient += s.M[row * kMaxControlDimension + col] * u[col];
      matrix[row * columns + columns - 1] = -gradient;
    }
    for (int row = threadIdx.x; row < s.m; row += blockDim.x) {
      for (int next = 0; next < s.next_n; ++next) {
        matrix[(s.n + row) * columns + eliminated + s.n + next] =
            -s.B[next * kMaxControlDimension + row];
      }
      double gradient = s.r[row];
      for (int col = 0; col < s.n; ++col)
        gradient += s.M[col * kMaxControlDimension + row] * x[col];
      for (int col = 0; col < s.m; ++col)
        gradient += s.R[row * kMaxControlDimension + col] * u[col];
      matrix[(s.n + row) * columns + columns - 1] = -gradient;
    }
  }
  __syncthreads();
  RrefBlock(matrix, rows, columns, columns - 1, tolerance, pivot_columns,
            pivot_rows, &rank, &best_row, factors);
  if (threadIdx.x == 0) {
    local_ok = !InconsistentRref(matrix, rows, columns, columns - 1, tolerance);
    if (!local_ok) SetFailure(status, kDeviceNumericalFailure, index, 12);
  }
  __syncthreads();
  if (!local_ok) return;
  ExtractResidualRelation(matrix, columns, rank, pivot_columns, eliminated,
                          left_dim, right_dim, &tree[index]);
}

__global__ void ReduceDualTreeLevelKernel(const Relation* tree,
                                          int child_offset, int parent_offset,
                                          int parent_count, double tolerance,
                                          Relation* mutable_tree,
                                          DeviceStatus* status) {
  const int index = blockIdx.x;
  if (index >= parent_count) return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled)) return;
  __shared__ double matrix[kMaxRrefEntries];
  __shared__ double factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;
  ComposeRelationsBlock(
      tree[child_offset + 2 * index], tree[child_offset + 2 * index + 1],
      tolerance, &mutable_tree[parent_offset + index], status, index, matrix,
      factors, pivot_columns, pivot_rows, &rank, &best_row, &local_ok);
}

__global__ void SolveDualRootKernel(const Relation* relation, NodeValue* value,
                                    DeviceStatus* status, double tolerance) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || status->code != kDeviceOk) return;
  if (relation->right_dim != 0) {
    SetFailure(status, kDeviceNumericalFailure, 0, 13);
    return;
  }
  value->left_dim = relation->left_dim;
  value->right_dim = 0;
  bool used[kMaxStateDimension]{};
  for (int row = 0; row < relation->rows; ++row) {
    int pivot = -1;
    for (int col = 0; col < relation->left_dim; ++col) {
      if (DeviceAbs(relation->left[row * kMaxStateDimension + col]) >
          tolerance) {
        pivot = col;
        break;
      }
    }
    if (pivot < 0 || used[pivot]) {
      SetFailure(status, kDeviceNumericalFailure, 0, 14);
      return;
    }
    used[pivot] = true;
    value->left[pivot] =
        relation->rhs[row] / relation->left[row * kMaxStateDimension + pivot];
  }
}

__global__ void ExpandDualTreeLevelKernel(const Relation* tree,
                                          int child_offset, int parent_offset,
                                          int parent_count, double tolerance,
                                          const NodeValue* parent_values,
                                          NodeValue* values,
                                          DeviceStatus* status) {
  const int index = blockIdx.x;
  if (index >= parent_count) return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled)) return;
  const Relation& left = tree[child_offset + 2 * index];
  const Relation& right = tree[child_offset + 2 * index + 1];
  const NodeValue& parent = parent_values[parent_offset + index];
  if (left.left_dim != parent.left_dim || right.right_dim != parent.right_dim ||
      left.right_dim != right.left_dim) {
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, index, 15);
    return;
  }
  const int shared = left.right_dim;
  const int rows = left.rows + right.rows;
  const int columns = shared + 1;
  __shared__ double matrix[kMaxRrefRows * (kMaxStateDimension + 1)];
  __shared__ double factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;
  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = 0.0;
  __syncthreads();
  for (int linear = threadIdx.x; linear < left.rows * shared;
       linear += blockDim.x) {
    const int row = linear / shared;
    const int col = linear % shared;
    matrix[row * columns + col] = left.right[row * kMaxStateDimension + col];
  }
  for (int row = threadIdx.x; row < left.rows; row += blockDim.x) {
    double rhs = left.rhs[row];
    for (int col = 0; col < left.left_dim; ++col) {
      rhs -= left.left[row * kMaxStateDimension + col] * parent.left[col];
    }
    matrix[row * columns + shared] = rhs;
  }
  for (int linear = threadIdx.x; linear < right.rows * shared;
       linear += blockDim.x) {
    const int row = linear / shared;
    const int col = linear % shared;
    matrix[(left.rows + row) * columns + col] =
        right.left[row * kMaxStateDimension + col];
  }
  for (int row = threadIdx.x; row < right.rows; row += blockDim.x) {
    double rhs = right.rhs[row];
    for (int col = 0; col < right.right_dim; ++col) {
      rhs -= right.right[row * kMaxStateDimension + col] * parent.right[col];
    }
    matrix[(left.rows + row) * columns + shared] = rhs;
  }
  __syncthreads();
  RrefBlock(matrix, rows, columns, shared, tolerance, pivot_columns, pivot_rows,
            &rank, &best_row, factors);
  if (threadIdx.x == 0) {
    local_ok = !InconsistentRref(matrix, rows, columns, shared, tolerance);
    if (!local_ok) SetFailure(status, kDeviceNumericalFailure, index, 16);
  }
  __syncthreads();
  if (!local_ok) return;
  if (threadIdx.x == 0) {
    NodeValue& left_value = values[child_offset + 2 * index];
    NodeValue& right_value = values[child_offset + 2 * index + 1];
    left_value.left_dim = left.left_dim;
    left_value.right_dim = shared;
    right_value.left_dim = shared;
    right_value.right_dim = right.right_dim;
    for (int col = 0; col < left.left_dim; ++col)
      left_value.left[col] = parent.left[col];
    for (int col = 0; col < right.right_dim; ++col)
      right_value.right[col] = parent.right[col];
    double shared_value[kMaxStateDimension]{};
    for (int p = 0; p < rank; ++p) {
      if (pivot_columns[p] < shared)
        shared_value[pivot_columns[p]] = matrix[p * columns + shared];
    }
    for (int col = 0; col < shared; ++col) {
      left_value.right[col] = shared_value[col];
      right_value.left[col] = shared_value[col];
    }
  }
}

__global__ void RecoverLocalMultipliersKernel(
    const PackedStage* stages, const PackedTerminal* terminal_ptr,
    int stage_count, const double* states, const double* controls,
    const NodeValue* leaf_values, double tolerance, double* initial_multiplier,
    double* dynamics_multipliers, double* mixed_multipliers,
    double* state_multipliers, double* terminal_multiplier,
    DeviceStatus* status) {
  const int index = blockIdx.x;
  if (index > stage_count) return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled)) return;
  __shared__ double
      matrix[kMaxRrefRows * (kMaxMixedConstraints + kMaxStateConstraints + 1)];
  __shared__ double factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int rows;
  __shared__ int variables;
  __shared__ int local_ok;
  const PackedTerminal& terminal = *terminal_ptr;
  if (threadIdx.x == 0) {
    if (index == stage_count) {
      rows = terminal.n;
      variables = terminal.state;
    } else {
      rows = stages[index].n + stages[index].m;
      variables = stages[index].mixed + stages[index].state;
    }
  }
  __syncthreads();
  const int columns = variables + 1;
  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = 0.0;
  __syncthreads();
  const NodeValue& endpoints = leaf_values[index];

  if (index == stage_count) {
    const double* x = states + index * kMaxStateDimension;
    for (int linear = threadIdx.x; linear < terminal.n * terminal.state;
         linear += blockDim.x) {
      const int row = linear / terminal.state;
      const int constraint = linear % terminal.state;
      matrix[row * columns + constraint] =
          terminal.E[constraint * kMaxStateDimension + row];
    }
    for (int row = threadIdx.x; row < terminal.n; row += blockDim.x) {
      double rhs = -terminal.q[row] - endpoints.left[row];
      for (int col = 0; col < terminal.n; ++col)
        rhs -= terminal.Q[row * kMaxStateDimension + col] * x[col];
      matrix[row * columns + variables] = rhs;
    }
  } else {
    const PackedStage& s = stages[index];
    const double* x = states + index * kMaxStateDimension;
    const double* u = controls + index * kMaxControlDimension;
    for (int linear = threadIdx.x; linear < s.n * s.mixed;
         linear += blockDim.x) {
      const int row = linear / s.mixed;
      const int constraint = linear % s.mixed;
      matrix[row * columns + constraint] =
          s.C[constraint * kMaxStateDimension + row];
    }
    for (int linear = threadIdx.x; linear < s.n * s.state;
         linear += blockDim.x) {
      const int row = linear / s.state;
      const int constraint = linear % s.state;
      matrix[row * columns + s.mixed + constraint] =
          s.E[constraint * kMaxStateDimension + row];
    }
    for (int linear = threadIdx.x; linear < s.m * s.mixed;
         linear += blockDim.x) {
      const int row = linear / s.mixed;
      const int constraint = linear % s.mixed;
      matrix[(s.n + row) * columns + constraint] =
          s.D[constraint * kMaxControlDimension + row];
    }
    for (int row = threadIdx.x; row < s.n; row += blockDim.x) {
      double rhs = -s.q[row] - endpoints.left[row];
      for (int col = 0; col < s.n; ++col)
        rhs -= s.Q[row * kMaxStateDimension + col] * x[col];
      for (int col = 0; col < s.m; ++col)
        rhs -= s.M[row * kMaxControlDimension + col] * u[col];
      for (int next = 0; next < s.next_n; ++next)
        rhs += s.A[next * kMaxStateDimension + row] * endpoints.right[next];
      matrix[row * columns + variables] = rhs;
    }
    for (int row = threadIdx.x; row < s.m; row += blockDim.x) {
      double rhs = -s.r[row];
      for (int col = 0; col < s.n; ++col)
        rhs -= s.M[col * kMaxControlDimension + row] * x[col];
      for (int col = 0; col < s.m; ++col)
        rhs -= s.R[row * kMaxControlDimension + col] * u[col];
      for (int next = 0; next < s.next_n; ++next)
        rhs += s.B[next * kMaxControlDimension + row] * endpoints.right[next];
      matrix[(s.n + row) * columns + variables] = rhs;
    }
  }
  __syncthreads();
  RrefBlock(matrix, rows, columns, variables, tolerance, pivot_columns,
            pivot_rows, &rank, &best_row, factors);
  if (threadIdx.x == 0) {
    local_ok = !InconsistentRref(matrix, rows, columns, variables, tolerance);
    if (!local_ok) SetFailure(status, kDeviceNumericalFailure, index, 17);
  }
  __syncthreads();
  if (!local_ok) return;
  if (threadIdx.x == 0) {
    double solution[kMaxMixedConstraints + kMaxStateConstraints]{};
    for (int p = 0; p < rank; ++p) {
      if (pivot_columns[p] < variables)
        solution[pivot_columns[p]] = matrix[p * columns + variables];
    }
    if (index == stage_count) {
      for (int row = 0; row < terminal.state; ++row)
        terminal_multiplier[row] = solution[row];
      if (stage_count == 0) {
        for (int row = 0; row < terminal.n; ++row)
          initial_multiplier[row] = endpoints.left[row];
      }
    } else {
      const PackedStage& s = stages[index];
      for (int row = 0; row < s.mixed; ++row)
        mixed_multipliers[index * kMaxMixedConstraints + row] = solution[row];
      for (int row = 0; row < s.state; ++row)
        state_multipliers[index * kMaxStateConstraints + row] =
            solution[s.mixed + row];
      if (index == 0) {
        for (int row = 0; row < s.n; ++row)
          initial_multiplier[row] = endpoints.left[row];
      }
      for (int row = 0; row < s.next_n; ++row)
        dynamics_multipliers[index * kMaxStateDimension + row] =
            endpoints.right[row];
    }
  }
}

}  // namespace
}  // namespace detail
}  // namespace cuda
}  // namespace clqr

namespace clqr {
namespace cuda {
namespace detail {
namespace {

__global__ void BuildValueElementsKernel(const ReducedStage* stages,
                                         const ReducedTerminal* terminal_ptr,
                                         int stage_count, double tolerance,
                                         ValueElement* elements,
                                         int* parallel_ok,
                                         DeviceStatus* status) {
  const int index = blockIdx.x;
  if (index > stage_count) return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled)) return;
  if (index == stage_count) {
    const ReducedTerminal& terminal = *terminal_ptr;
    if (threadIdx.x == 0) {
      elements[index].left_dim = terminal.n;
      elements[index].right_dim = 0;
    }
    for (int linear = threadIdx.x; linear < terminal.n * terminal.n;
         linear += blockDim.x) {
      const int row = linear / terminal.n;
      const int col = linear % terminal.n;
      elements[index].J[row * kMaxStateDimension + col] =
          terminal.Q[row * kMaxStateDimension + col];
    }
    for (int row = threadIdx.x; row < terminal.n; row += blockDim.x) {
      elements[index].eta[row] = terminal.q[row];
    }
    return;
  }

  const ReducedStage& s = stages[index];
  ValueElement& out = elements[index];
  if (threadIdx.x == 0) {
    out.left_dim = s.n;
    out.right_dim = s.next_n;
  }
  if (s.m == 0) {
    for (int linear = threadIdx.x; linear < s.next_n * s.n;
         linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      out.A[row * kMaxStateDimension + col] =
          s.A[row * kMaxStateDimension + col];
    }
    for (int row = threadIdx.x; row < s.next_n; row += blockDim.x)
      out.b[row] = s.c[row];
    for (int linear = threadIdx.x; linear < s.n * s.n; linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      out.J[row * kMaxStateDimension + col] =
          s.Q[row * kMaxStateDimension + col];
    }
    for (int row = threadIdx.x; row < s.n; row += blockDim.x)
      out.eta[row] = s.q[row];
    return;
  }

  __shared__ double
      augmented[kMaxControlDimension * (2 * kMaxControlDimension)];
  __shared__ double factors[kMaxRrefRows];
  __shared__ double cholesky[kMaxControlDimension * kMaxControlDimension];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int positive_definite;
  const int columns = 2 * s.m;
  for (int linear = threadIdx.x; linear < s.m * columns; linear += blockDim.x) {
    const int row = linear / columns;
    const int col = linear % columns;
    augmented[linear] = col < s.m ? s.R[row * kMaxControlDimension + col]
                                  : (col - s.m == row ? 1.0 : 0.0);
  }
  for (int linear = threadIdx.x; linear < s.m * s.m; linear += blockDim.x) {
    const int row = linear / s.m;
    const int col = linear % s.m;
    cholesky[linear] = 0.5 * (s.R[row * kMaxControlDimension + col] +
                              s.R[col * kMaxControlDimension + row]);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    positive_definite = 1;
    double scale = 1.0;
    for (int i = 0; i < s.m; ++i)
      scale = fmax(scale, DeviceAbs(cholesky[i * s.m + i]));
    for (int i = 0; i < s.m && positive_definite; ++i) {
      double diagonal = cholesky[i * s.m + i];
      for (int k = 0; k < i; ++k) {
        diagonal -= cholesky[i * s.m + k] * cholesky[i * s.m + k];
      }
      if (!(diagonal > tolerance * scale) || !DeviceFinite(diagonal)) {
        positive_definite = 0;
        break;
      }
      cholesky[i * s.m + i] = sqrt(diagonal);
      for (int row = i + 1; row < s.m; ++row) {
        double value = cholesky[row * s.m + i];
        for (int k = 0; k < i; ++k) {
          value -= cholesky[row * s.m + k] * cholesky[i * s.m + k];
        }
        cholesky[row * s.m + i] = value / cholesky[i * s.m + i];
      }
    }
    if (!positive_definite) atomicExch(parallel_ok, 0);
  }
  __syncthreads();
  if (!positive_definite) return;

  RrefBlock(augmented, s.m, columns, s.m, tolerance, pivot_columns, pivot_rows,
            &rank, &best_row, factors);
  if (rank != s.m) {
    if (threadIdx.x == 0) atomicExch(parallel_ok, 0);
    return;
  }

  // R inverse occupies the right half of augmented.
  for (int linear = threadIdx.x; linear < s.n * s.n; linear += blockDim.x) {
    const int a = linear / s.n;
    const int b = linear % s.n;
    double value = s.Q[a * kMaxStateDimension + b];
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value -= s.M[a * kMaxControlDimension + u] *
                 augmented[u * columns + s.m + v] *
                 s.M[b * kMaxControlDimension + v];
      }
    }
    out.J[a * kMaxStateDimension + b] = value;
  }
  for (int a = threadIdx.x; a < s.n; a += blockDim.x) {
    double value = s.q[a];
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value -= s.M[a * kMaxControlDimension + u] *
                 augmented[u * columns + s.m + v] * s.r[v];
      }
    }
    out.eta[a] = value;
  }
  for (int linear = threadIdx.x; linear < s.next_n * s.n;
       linear += blockDim.x) {
    const int row = linear / s.n;
    const int col = linear % s.n;
    double value = s.A[row * kMaxStateDimension + col];
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value -= s.B[row * kMaxControlDimension + u] *
                 augmented[u * columns + s.m + v] *
                 s.M[col * kMaxControlDimension + v];
      }
    }
    out.A[row * kMaxStateDimension + col] = value;
  }
  for (int row = threadIdx.x; row < s.next_n; row += blockDim.x) {
    double value = s.c[row];
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value -= s.B[row * kMaxControlDimension + u] *
                 augmented[u * columns + s.m + v] * s.r[v];
      }
    }
    out.b[row] = value;
  }
  for (int linear = threadIdx.x; linear < s.next_n * s.next_n;
       linear += blockDim.x) {
    const int a = linear / s.next_n;
    const int b = linear % s.next_n;
    double value = 0.0;
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value += s.B[a * kMaxControlDimension + u] *
                 augmented[u * columns + s.m + v] *
                 s.B[b * kMaxControlDimension + v];
      }
    }
    out.C[a * kMaxStateDimension + b] = value;
  }
}

__device__ void ComposeValueElementsBlock(
    const ValueElement& first, const ValueElement& second, double tolerance,
    ValueElement* output, int* parallel_ok, double* augmented, double* factors,
    int* pivot_columns, int* pivot_rows, int* rank, int* best_row) {
  const int shared = first.right_dim;
  if (shared != second.left_dim) {
    if (threadIdx.x == 0) atomicExch(parallel_ok, 0);
    return;
  }
  const int left = first.left_dim;
  const int right = second.right_dim;
  if (threadIdx.x == 0) {
    output->left_dim = left;
    output->right_dim = right;
  }
  if (shared == 0) {
    for (int linear = threadIdx.x; linear < left * left; linear += blockDim.x) {
      const int row = linear / left;
      const int col = linear % left;
      output->J[row * kMaxStateDimension + col] =
          first.J[row * kMaxStateDimension + col];
    }
    for (int row = threadIdx.x; row < left; row += blockDim.x)
      output->eta[row] = first.eta[row];
    for (int linear = threadIdx.x; linear < right * right;
         linear += blockDim.x) {
      const int row = linear / right;
      const int col = linear % right;
      output->C[row * kMaxStateDimension + col] =
          second.C[row * kMaxStateDimension + col];
    }
    for (int row = threadIdx.x; row < right; row += blockDim.x)
      output->b[row] = second.b[row];
    return;
  }

  const int rhs_columns = left + 1 + shared;
  const int columns = shared + rhs_columns;
  for (int linear = threadIdx.x; linear < shared * columns;
       linear += blockDim.x) {
    const int row = linear / columns;
    const int col = linear % columns;
    double value = 0.0;
    if (col < shared) {
      value = row == col ? 1.0 : 0.0;
      for (int k = 0; k < shared; ++k) {
        value += first.C[row * kMaxStateDimension + k] *
                 second.J[k * kMaxStateDimension + col];
      }
    } else if (col < shared + left) {
      value = first.A[row * kMaxStateDimension + col - shared];
    } else if (col == shared + left) {
      value = first.b[row];
      for (int k = 0; k < shared; ++k) {
        value -= first.C[row * kMaxStateDimension + k] * second.eta[k];
      }
    } else {
      value = first.C[row * kMaxStateDimension + col - shared - left - 1];
    }
    augmented[linear] = value;
  }
  __syncthreads();
  RrefBlock(augmented, shared, columns, shared, tolerance, pivot_columns,
            pivot_rows, rank, best_row, factors);
  if (*rank != shared) {
    if (threadIdx.x == 0) atomicExch(parallel_ok, 0);
    return;
  }

  // A = A2*S^{-1}*A1 and b = A2*S^{-1}(b1+C1*eta2)+b2.
  for (int linear = threadIdx.x; linear < right * left; linear += blockDim.x) {
    const int row = linear / left;
    const int col = linear % left;
    double value = 0.0;
    for (int k = 0; k < shared; ++k) {
      value += second.A[row * kMaxStateDimension + k] *
               augmented[k * columns + shared + col];
    }
    output->A[row * kMaxStateDimension + col] = value;
  }
  for (int row = threadIdx.x; row < right; row += blockDim.x) {
    double value = second.b[row];
    for (int k = 0; k < shared; ++k) {
      value += second.A[row * kMaxStateDimension + k] *
               augmented[k * columns + shared + left];
    }
    output->b[row] = value;
  }
  for (int linear = threadIdx.x; linear < right * right; linear += blockDim.x) {
    const int row = linear / right;
    const int col = linear % right;
    double value = second.C[row * kMaxStateDimension + col];
    for (int p = 0; p < shared; ++p) {
      for (int q = 0; q < shared; ++q) {
        value += second.A[row * kMaxStateDimension + p] *
                 augmented[p * columns + shared + left + 1 + q] *
                 second.A[col * kMaxStateDimension + q];
      }
    }
    output->C[row * kMaxStateDimension + col] = value;
  }
  for (int linear = threadIdx.x; linear < left * left; linear += blockDim.x) {
    const int row = linear / left;
    const int col = linear % left;
    double value = first.J[row * kMaxStateDimension + col];
    for (int p = 0; p < shared; ++p) {
      for (int q = 0; q < shared; ++q) {
        value += first.A[p * kMaxStateDimension + row] *
                 second.J[p * kMaxStateDimension + q] *
                 augmented[q * columns + shared + col];
      }
    }
    output->J[row * kMaxStateDimension + col] = value;
  }
  for (int row = threadIdx.x; row < left; row += blockDim.x) {
    double value = first.eta[row];
    for (int p = 0; p < shared; ++p) {
      double dual = second.eta[p];
      for (int q = 0; q < shared; ++q) {
        dual += second.J[p * kMaxStateDimension + q] *
                augmented[q * columns + shared + left];
      }
      value += first.A[p * kMaxStateDimension + row] * dual;
    }
    output->eta[row] = value;
  }
  __syncthreads();
  for (int linear = threadIdx.x; linear < left * left; linear += blockDim.x) {
    const int row = linear / left;
    const int col = linear % left;
    if (row < col) {
      const double value = 0.5 * (output->J[row * kMaxStateDimension + col] +
                                  output->J[col * kMaxStateDimension + row]);
      output->J[row * kMaxStateDimension + col] = value;
      output->J[col * kMaxStateDimension + row] = value;
    }
  }
  for (int linear = threadIdx.x; linear < right * right; linear += blockDim.x) {
    const int row = linear / right;
    const int col = linear % right;
    if (row < col) {
      const double value = 0.5 * (output->C[row * kMaxStateDimension + col] +
                                  output->C[col * kMaxStateDimension + row]);
      output->C[row * kMaxStateDimension + col] = value;
      output->C[col * kMaxStateDimension + row] = value;
    }
  }
}

__device__ void CopyValueElementBlock(const ValueElement& input,
                                      ValueElement* output) {
  if (threadIdx.x == 0) {
    output->left_dim = input.left_dim;
    output->right_dim = input.right_dim;
  }
  for (int linear = threadIdx.x; linear < input.right_dim * input.left_dim;
       linear += blockDim.x) {
    const int row = linear / input.left_dim;
    const int col = linear % input.left_dim;
    output->A[row * kMaxStateDimension + col] =
        input.A[row * kMaxStateDimension + col];
  }
  for (int row = threadIdx.x; row < input.right_dim; row += blockDim.x)
    output->b[row] = input.b[row];
  for (int linear = threadIdx.x; linear < input.right_dim * input.right_dim;
       linear += blockDim.x) {
    const int row = linear / input.right_dim;
    const int col = linear % input.right_dim;
    output->C[row * kMaxStateDimension + col] =
        input.C[row * kMaxStateDimension + col];
  }
  for (int row = threadIdx.x; row < input.left_dim; row += blockDim.x)
    output->eta[row] = input.eta[row];
  for (int linear = threadIdx.x; linear < input.left_dim * input.left_dim;
       linear += blockDim.x) {
    const int row = linear / input.left_dim;
    const int col = linear % input.left_dim;
    output->J[row * kMaxStateDimension + col] =
        input.J[row * kMaxStateDimension + col];
  }
}

__global__ void SuffixValueElementsKernel(const ValueElement* input, int count,
                                          int offset, double tolerance,
                                          ValueElement* output,
                                          int* parallel_ok) {
  const int index = blockIdx.x;
  if (index >= count) return;
  __shared__ int block_enabled;
  if (!BlockEnabled(parallel_ok, &block_enabled)) return;
  if (index + offset >= count) {
    CopyValueElementBlock(input[index], &output[index]);
    return;
  }
  __shared__ double
      augmented[kMaxStateDimension * (3 * kMaxStateDimension + 1)];
  __shared__ double factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  ComposeValueElementsBlock(input[index], input[index + offset], tolerance,
                            &output[index], parallel_ok, augmented, factors,
                            pivot_columns, pivot_rows, &rank, &best_row);
}

__device__ void BuildFeedbackSystem(const ReducedStage& s,
                                    const ValueElement& next, double* augmented,
                                    int columns) {
  for (int linear = threadIdx.x; linear < s.m * columns; linear += blockDim.x) {
    const int row = linear / columns;
    const int col = linear % columns;
    double value = 0.0;
    if (col < s.m) {
      value = s.R[row * kMaxControlDimension + col];
      for (int a = 0; a < s.next_n; ++a) {
        for (int b = 0; b < s.next_n; ++b) {
          value += s.B[a * kMaxControlDimension + row] *
                   next.J[a * kMaxStateDimension + b] *
                   s.B[b * kMaxControlDimension + col];
        }
      }
    } else if (col < s.m + s.n) {
      const int x = col - s.m;
      value = -s.M[x * kMaxControlDimension + row];
      for (int a = 0; a < s.next_n; ++a) {
        for (int b = 0; b < s.next_n; ++b) {
          value -= s.B[a * kMaxControlDimension + row] *
                   next.J[a * kMaxStateDimension + b] *
                   s.A[b * kMaxStateDimension + x];
        }
      }
    } else {
      value = -s.r[row];
      for (int a = 0; a < s.next_n; ++a) {
        double future = next.eta[a];
        for (int b = 0; b < s.next_n; ++b) {
          future += next.J[a * kMaxStateDimension + b] * s.c[b];
        }
        value -= s.B[a * kMaxControlDimension + row] * future;
      }
    }
    augmented[linear] = value;
  }
}

__device__ void ExtractFeedback(const ReducedStage& s, const double* augmented,
                                int columns, Feedback* feedback) {
  if (threadIdx.x == 0) {
    feedback->state_dim = s.n;
    feedback->next_state_dim = s.next_n;
    feedback->control_dim = s.m;
  }
  for (int linear = threadIdx.x; linear < s.m * s.n; linear += blockDim.x) {
    const int row = linear / s.n;
    const int col = linear % s.n;
    feedback->K[row * kMaxStateDimension + col] =
        augmented[row * columns + s.m + col];
  }
  for (int row = threadIdx.x; row < s.m; row += blockDim.x) {
    feedback->k[row] = augmented[row * columns + s.m + s.n];
  }
  __syncthreads();
  for (int linear = threadIdx.x; linear < s.next_n * s.n;
       linear += blockDim.x) {
    const int row = linear / s.n;
    const int col = linear % s.n;
    double value = s.A[row * kMaxStateDimension + col];
    for (int u = 0; u < s.m; ++u) {
      value += s.B[row * kMaxControlDimension + u] *
               feedback->K[u * kMaxStateDimension + col];
    }
    feedback->transition[row * kMaxStateDimension + col] = value;
  }
  for (int row = threadIdx.x; row < s.next_n; row += blockDim.x) {
    double value = s.c[row];
    for (int u = 0; u < s.m; ++u) {
      value += s.B[row * kMaxControlDimension + u] * feedback->k[u];
    }
    feedback->offset[row] = value;
  }
}

__global__ void FeedbackKernel(const ReducedStage* stages,
                               const ValueElement* suffix, int stage_count,
                               double tolerance, Feedback* feedback,
                               DeviceStatus* status) {
  const int index = blockIdx.x;
  if (index >= stage_count) return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled)) return;
  const ReducedStage& s = stages[index];
  const ValueElement& next = suffix[index + 1];
  Feedback& out = feedback[index];
  if (s.m == 0) {
    if (threadIdx.x == 0) {
      out.state_dim = s.n;
      out.next_state_dim = s.next_n;
      out.control_dim = 0;
    }
    for (int linear = threadIdx.x; linear < s.next_n * s.n;
         linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      out.transition[row * kMaxStateDimension + col] =
          s.A[row * kMaxStateDimension + col];
    }
    for (int row = threadIdx.x; row < s.next_n; row += blockDim.x)
      out.offset[row] = s.c[row];
    return;
  }
  __shared__ double augmented[kMaxControlDimension *
                              (kMaxControlDimension + kMaxStateDimension + 1)];
  __shared__ double factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  const int columns = s.m + s.n + 1;
  BuildFeedbackSystem(s, next, augmented, columns);
  __syncthreads();
  RrefBlock(augmented, s.m, columns, s.m, tolerance, pivot_columns, pivot_rows,
            &rank, &best_row, factors);
  if (rank != s.m) {
    if (threadIdx.x == 0) SetFailure(status, kDeviceNumericalFailure, index, 9);
    return;
  }
  ExtractFeedback(s, augmented, columns, &out);
}

__global__ void SequentialRiccatiKernel(const ReducedStage* stages,
                                        int stage_count, double tolerance,
                                        ValueElement* suffix,
                                        Feedback* feedback,
                                        DeviceStatus* status) {
  if (blockIdx.x != 0) return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled)) return;
  __shared__ double augmented[kMaxControlDimension *
                              (kMaxControlDimension + kMaxStateDimension + 1)];
  __shared__ double factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;

  for (int index = stage_count - 1; index >= 0; --index) {
    const ReducedStage& s = stages[index];
    const ValueElement& next = suffix[index + 1];
    Feedback& fb = feedback[index];
    if (s.m == 0) {
      if (threadIdx.x == 0) {
        fb.state_dim = s.n;
        fb.next_state_dim = s.next_n;
        fb.control_dim = 0;
      }
      for (int linear = threadIdx.x; linear < s.next_n * s.n;
           linear += blockDim.x) {
        const int row = linear / s.n;
        const int col = linear % s.n;
        fb.transition[row * kMaxStateDimension + col] =
            s.A[row * kMaxStateDimension + col];
      }
      for (int row = threadIdx.x; row < s.next_n; row += blockDim.x)
        fb.offset[row] = s.c[row];
    } else {
      const int columns = s.m + s.n + 1;
      BuildFeedbackSystem(s, next, augmented, columns);
      __syncthreads();
      RrefBlock(augmented, s.m, columns, s.m, tolerance, pivot_columns,
                pivot_rows, &rank, &best_row, factors);
      if (rank != s.m) {
        if (threadIdx.x == 0)
          SetFailure(status, kDeviceNumericalFailure, index, 10);
        return;
      }
      ExtractFeedback(s, augmented, columns, &fb);
    }
    __syncthreads();

    ValueElement& current = suffix[index];
    if (threadIdx.x == 0) {
      current.left_dim = s.n;
      current.right_dim = 0;
    }
    for (int linear = threadIdx.x; linear < s.n * s.n; linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      double value = s.Q[row * kMaxStateDimension + col];
      for (int a = 0; a < s.next_n; ++a) {
        for (int b = 0; b < s.next_n; ++b) {
          value += s.A[a * kMaxStateDimension + row] *
                   next.J[a * kMaxStateDimension + b] *
                   s.A[b * kMaxStateDimension + col];
        }
      }
      for (int u = 0; u < s.m; ++u) {
        double cross = s.M[row * kMaxControlDimension + u];
        for (int a = 0; a < s.next_n; ++a) {
          for (int b = 0; b < s.next_n; ++b) {
            cross += s.A[a * kMaxStateDimension + row] *
                     next.J[a * kMaxStateDimension + b] *
                     s.B[b * kMaxControlDimension + u];
          }
        }
        value += cross * fb.K[u * kMaxStateDimension + col];
      }
      current.J[row * kMaxStateDimension + col] = value;
    }
    for (int row = threadIdx.x; row < s.n; row += blockDim.x) {
      double value = s.q[row];
      for (int a = 0; a < s.next_n; ++a) {
        double future = next.eta[a];
        for (int b = 0; b < s.next_n; ++b) {
          future += next.J[a * kMaxStateDimension + b] * s.c[b];
        }
        value += s.A[a * kMaxStateDimension + row] * future;
      }
      for (int u = 0; u < s.m; ++u) {
        double cross = s.M[row * kMaxControlDimension + u];
        for (int a = 0; a < s.next_n; ++a) {
          for (int b = 0; b < s.next_n; ++b) {
            cross += s.A[a * kMaxStateDimension + row] *
                     next.J[a * kMaxStateDimension + b] *
                     s.B[b * kMaxControlDimension + u];
          }
        }
        value += cross * fb.k[u];
      }
      current.eta[row] = value;
    }
    __syncthreads();
    for (int linear = threadIdx.x; linear < s.n * s.n; linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      if (row < col) {
        const double value = 0.5 * (current.J[row * kMaxStateDimension + col] +
                                    current.J[col * kMaxStateDimension + row]);
        current.J[row * kMaxStateDimension + col] = value;
        current.J[col * kMaxStateDimension + row] = value;
      }
    }
    __syncthreads();
  }
}

}  // namespace
}  // namespace detail
}  // namespace cuda
}  // namespace clqr

namespace clqr {
namespace cuda {
namespace detail {
namespace {

__global__ void ReduceStagesKernel(
    const PackedStage* stages, const Relation* suffix,
    const StateParam* state_params, int stage_count, double tolerance,
    ControlParam* control_params, ReducedStage* reduced, DeviceStatus* status) {
  const int index = blockIdx.x;
  if (index >= stage_count) return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled)) return;
  const PackedStage& s = stages[index];
  const StateParam& current = state_params[index];
  const StateParam& next = state_params[index + 1];
  const Relation& next_relation = suffix[index + 1];
  __shared__ double matrix[kMaxStageConstraintRows * kMaxStageReductionColumns];
  __shared__ double factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int rows;
  __shared__ int columns;
  __shared__ int control_rank;
  __shared__ int local_ok;

  if (threadIdx.x == 0) {
    rows = s.mixed + next_relation.rows;
    columns = s.m + current.reduced_dim + 1;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = 0.0;
  __syncthreads();

  // Original mixed equalities after x = T*z + t.
  for (int linear = threadIdx.x; linear < s.mixed * s.m; linear += blockDim.x) {
    const int row = linear / s.m;
    const int col = linear % s.m;
    matrix[row * columns + col] = s.D[row * kMaxControlDimension + col];
  }
  for (int linear = threadIdx.x; linear < s.mixed * current.reduced_dim;
       linear += blockDim.x) {
    const int row = linear / current.reduced_dim;
    const int z = linear % current.reduced_dim;
    double value = 0.0;
    for (int x = 0; x < s.n; ++x) {
      value += s.C[row * kMaxStateDimension + x] *
               current.T[x * kMaxStateDimension + z];
    }
    matrix[row * columns + s.m + z] = value;
  }
  for (int row = threadIdx.x; row < s.mixed; row += blockDim.x) {
    double value = -s.d[row];
    for (int x = 0; x < s.n; ++x) {
      value -= s.C[row * kMaxStateDimension + x] * current.t[x];
    }
    matrix[row * columns + columns - 1] = value;
  }

  // The successor must belong to its propagated feasible-state set.
  for (int linear = threadIdx.x; linear < next_relation.rows * s.m;
       linear += blockDim.x) {
    const int row = linear / s.m;
    const int u = linear % s.m;
    double value = 0.0;
    for (int xp = 0; xp < s.next_n; ++xp) {
      value += next_relation.left[row * kMaxStateDimension + xp] *
               s.B[xp * kMaxControlDimension + u];
    }
    matrix[(s.mixed + row) * columns + u] = value;
  }
  for (int linear = threadIdx.x;
       linear < next_relation.rows * current.reduced_dim;
       linear += blockDim.x) {
    const int row = linear / current.reduced_dim;
    const int z = linear % current.reduced_dim;
    double value = 0.0;
    for (int xp = 0; xp < s.next_n; ++xp) {
      double at = 0.0;
      for (int x = 0; x < s.n; ++x) {
        at += s.A[xp * kMaxStateDimension + x] *
              current.T[x * kMaxStateDimension + z];
      }
      value += next_relation.left[row * kMaxStateDimension + xp] * at;
    }
    matrix[(s.mixed + row) * columns + s.m + z] = value;
  }
  for (int row = threadIdx.x; row < next_relation.rows; row += blockDim.x) {
    double value = next_relation.rhs[row];
    for (int xp = 0; xp < s.next_n; ++xp) {
      double affine = s.c[xp];
      for (int x = 0; x < s.n; ++x) {
        affine += s.A[xp * kMaxStateDimension + x] * current.t[x];
      }
      value -= next_relation.left[row * kMaxStateDimension + xp] * affine;
    }
    matrix[(s.mixed + row) * columns + columns - 1] = value;
  }
  __syncthreads();

  RrefBlock(matrix, rows, columns, columns - 1, tolerance, pivot_columns,
            pivot_rows, &rank, &best_row, factors);
  if (threadIdx.x == 0) {
    local_ok = 1;
    if (InconsistentRref(matrix, rows, columns, columns - 1, tolerance)) {
      SetFailure(status, kDeviceInfeasible, index, 6);
      local_ok = 0;
    }
    control_rank = 0;
    while (control_rank < rank && pivot_columns[control_rank] < s.m)
      ++control_rank;
    if (control_rank < rank) {
      // The suffix relation says every current reduced state is feasible, so
      // elimination must not discover an additional condition on z.
      SetFailure(status, kDeviceNumericalFailure, index, 7);
      local_ok = 0;
    }
    if (local_ok) {
      ControlParam& cp = control_params[index];
      cp.physical_dim = s.m;
      cp.state_dim = current.reduced_dim;
      cp.reduced_dim = s.m - control_rank;
      bool pivot[kMaxControlDimension]{};
      for (int p = 0; p < control_rank; ++p) pivot[pivot_columns[p]] = true;
      int free = 0;
      for (int u = 0; u < s.m; ++u) {
        if (!pivot[u]) cp.free_columns[free++] = u;
      }
      for (int p = 0; p < control_rank; ++p) {
        const int u = pivot_columns[p];
        cp.y[u] = matrix[p * columns + columns - 1];
        for (int z = 0; z < current.reduced_dim; ++z) {
          cp.Y[u * kMaxStateDimension + z] = -matrix[p * columns + s.m + z];
        }
        for (int v = 0; v < cp.reduced_dim; ++v) {
          cp.Z[u * kMaxControlDimension + v] =
              -matrix[p * columns + cp.free_columns[v]];
        }
      }
      for (int v = 0; v < cp.reduced_dim; ++v) {
        cp.Z[cp.free_columns[v] * kMaxControlDimension + v] = 1.0;
      }
      ReducedStage& rs = reduced[index];
      rs.n = current.reduced_dim;
      rs.next_n = next.reduced_dim;
      rs.m = cp.reduced_dim;
    }
  }
  __syncthreads();
  if (!local_ok) return;

  const ControlParam& cp = control_params[index];
  ReducedStage& rs = reduced[index];

  // Reduced dynamics, selecting the free physical coordinates at node i+1.
  for (int linear = threadIdx.x; linear < rs.next_n * rs.n;
       linear += blockDim.x) {
    const int row = linear / rs.n;
    const int z = linear % rs.n;
    const int xp = next.free_columns[row];
    double value = 0.0;
    for (int x = 0; x < s.n; ++x) {
      value += s.A[xp * kMaxStateDimension + x] *
               current.T[x * kMaxStateDimension + z];
    }
    for (int u = 0; u < s.m; ++u) {
      value +=
          s.B[xp * kMaxControlDimension + u] * cp.Y[u * kMaxStateDimension + z];
    }
    rs.A[row * kMaxStateDimension + z] = value;
  }
  for (int linear = threadIdx.x; linear < rs.next_n * rs.m;
       linear += blockDim.x) {
    const int row = linear / rs.m;
    const int v = linear % rs.m;
    const int xp = next.free_columns[row];
    double value = 0.0;
    for (int u = 0; u < s.m; ++u) {
      value += s.B[xp * kMaxControlDimension + u] *
               cp.Z[u * kMaxControlDimension + v];
    }
    rs.B[row * kMaxControlDimension + v] = value;
  }
  for (int row = threadIdx.x; row < rs.next_n; row += blockDim.x) {
    const int xp = next.free_columns[row];
    double value = s.c[xp] - next.t[xp];
    for (int x = 0; x < s.n; ++x) {
      value += s.A[xp * kMaxStateDimension + x] * current.t[x];
    }
    for (int u = 0; u < s.m; ++u) {
      value += s.B[xp * kMaxControlDimension + u] * cp.y[u];
    }
    rs.c[row] = value;
  }

  // Reduced quadratic and bilinear terms.
  for (int linear = threadIdx.x; linear < rs.n * rs.n; linear += blockDim.x) {
    const int a = linear / rs.n;
    const int b = linear % rs.n;
    double value = 0.0;
    for (int x = 0; x < s.n; ++x) {
      for (int y = 0; y < s.n; ++y) {
        value += current.T[x * kMaxStateDimension + a] *
                 s.Q[x * kMaxStateDimension + y] *
                 current.T[y * kMaxStateDimension + b];
      }
      for (int u = 0; u < s.m; ++u) {
        value += current.T[x * kMaxStateDimension + a] *
                 s.M[x * kMaxControlDimension + u] *
                 cp.Y[u * kMaxStateDimension + b];
        value += cp.Y[u * kMaxStateDimension + a] *
                 s.M[x * kMaxControlDimension + u] *
                 current.T[x * kMaxStateDimension + b];
      }
    }
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value += cp.Y[u * kMaxStateDimension + a] *
                 s.R[u * kMaxControlDimension + v] *
                 cp.Y[v * kMaxStateDimension + b];
      }
    }
    rs.Q[a * kMaxStateDimension + b] = value;
  }
  for (int linear = threadIdx.x; linear < rs.m * rs.m; linear += blockDim.x) {
    const int a = linear / rs.m;
    const int b = linear % rs.m;
    double value = 0.0;
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value += cp.Z[u * kMaxControlDimension + a] *
                 s.R[u * kMaxControlDimension + v] *
                 cp.Z[v * kMaxControlDimension + b];
      }
    }
    rs.R[a * kMaxControlDimension + b] = value;
  }
  for (int linear = threadIdx.x; linear < rs.n * rs.m; linear += blockDim.x) {
    const int z = linear / rs.m;
    const int v = linear % rs.m;
    double value = 0.0;
    for (int x = 0; x < s.n; ++x) {
      for (int u = 0; u < s.m; ++u) {
        value += current.T[x * kMaxStateDimension + z] *
                 s.M[x * kMaxControlDimension + u] *
                 cp.Z[u * kMaxControlDimension + v];
      }
    }
    for (int u = 0; u < s.m; ++u) {
      for (int w = 0; w < s.m; ++w) {
        value += cp.Y[u * kMaxStateDimension + z] *
                 s.R[u * kMaxControlDimension + w] *
                 cp.Z[w * kMaxControlDimension + v];
      }
    }
    rs.M[z * kMaxControlDimension + v] = value;
  }

  for (int z = threadIdx.x; z < rs.n; z += blockDim.x) {
    double value = 0.0;
    for (int x = 0; x < s.n; ++x) {
      double gx = s.q[x];
      for (int y = 0; y < s.n; ++y)
        gx += s.Q[x * kMaxStateDimension + y] * current.t[y];
      for (int u = 0; u < s.m; ++u)
        gx += s.M[x * kMaxControlDimension + u] * cp.y[u];
      value += current.T[x * kMaxStateDimension + z] * gx;
    }
    for (int u = 0; u < s.m; ++u) {
      double gu = s.r[u];
      for (int x = 0; x < s.n; ++x)
        gu += s.M[x * kMaxControlDimension + u] * current.t[x];
      for (int v = 0; v < s.m; ++v)
        gu += s.R[u * kMaxControlDimension + v] * cp.y[v];
      value += cp.Y[u * kMaxStateDimension + z] * gu;
    }
    rs.q[z] = value;
  }
  for (int v = threadIdx.x; v < rs.m; v += blockDim.x) {
    double value = 0.0;
    for (int u = 0; u < s.m; ++u) {
      double gu = s.r[u];
      for (int x = 0; x < s.n; ++x)
        gu += s.M[x * kMaxControlDimension + u] * current.t[x];
      for (int w = 0; w < s.m; ++w)
        gu += s.R[u * kMaxControlDimension + w] * cp.y[w];
      value += cp.Z[u * kMaxControlDimension + v] * gu;
    }
    rs.r[v] = value;
  }
}

__global__ void ReduceTerminalKernel(const PackedTerminal* terminal_ptr,
                                     const StateParam* state_params,
                                     int terminal_index,
                                     ReducedTerminal* reduced) {
  const PackedTerminal& terminal = *terminal_ptr;
  const StateParam& param = state_params[terminal_index];
  if (threadIdx.x == 0) reduced->n = param.reduced_dim;
  for (int linear = threadIdx.x; linear < param.reduced_dim * param.reduced_dim;
       linear += blockDim.x) {
    const int a = linear / param.reduced_dim;
    const int b = linear % param.reduced_dim;
    double value = 0.0;
    for (int x = 0; x < terminal.n; ++x) {
      for (int y = 0; y < terminal.n; ++y) {
        value += param.T[x * kMaxStateDimension + a] *
                 terminal.Q[x * kMaxStateDimension + y] *
                 param.T[y * kMaxStateDimension + b];
      }
    }
    reduced->Q[a * kMaxStateDimension + b] = value;
  }
  for (int a = threadIdx.x; a < param.reduced_dim; a += blockDim.x) {
    double value = 0.0;
    for (int x = 0; x < terminal.n; ++x) {
      double gx = terminal.q[x];
      for (int y = 0; y < terminal.n; ++y) {
        gx += terminal.Q[x * kMaxStateDimension + y] * param.t[y];
      }
      value += param.T[x * kMaxStateDimension + a] * gx;
    }
    reduced->q[a] = value;
  }
}

__global__ void InitialReducedStateKernel(const StateParam* state_params,
                                          const double* initial_state,
                                          double* reduced_initial,
                                          double tolerance,
                                          DeviceStatus* status) {
  const StateParam& param = state_params[0];
  for (int z = threadIdx.x; z < param.reduced_dim; z += blockDim.x) {
    const int physical = param.free_columns[z];
    reduced_initial[z] = initial_state[physical] - param.t[physical];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double scale = 1.0;
    double residual = 0.0;
    for (int x = 0; x < param.physical_dim; ++x) {
      double value = param.t[x];
      for (int z = 0; z < param.reduced_dim; ++z) {
        value += param.T[x * kMaxStateDimension + z] * reduced_initial[z];
      }
      scale = fmax(scale, DeviceAbs(initial_state[x]));
      residual = fmax(residual, DeviceAbs(value - initial_state[x]));
    }
    if (residual > 20.0 * tolerance * scale) {
      SetFailure(status, kDeviceInfeasible, 0, 8);
    }
  }
}

}  // namespace
}  // namespace detail
}  // namespace cuda
}  // namespace clqr
