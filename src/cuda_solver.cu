#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
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
#ifdef CLQR_USE_FLOAT
constexpr Scalar kMinimumFeasibilityConsistencyTolerance = 1e-4f;
constexpr Scalar kMinimumMultiplierRankTolerance = 3e-3f;
constexpr Scalar kMultiplierConsistencyTolerancePerTreeLevel = 2e-2f;
#else
constexpr Scalar kMinimumFeasibilityConsistencyTolerance = Scalar{0};
constexpr Scalar kMinimumMultiplierRankTolerance = 1e-7;
constexpr Scalar kMultiplierConsistencyTolerancePerTreeLevel = 1e-6;
#endif
constexpr int kMaxRrefRows =
    ConstexprMax(2 * kMaxRelationRows,
                 ConstexprMax(kMaxMixedConstraints + kMaxStateConstraints +
                                  kMaxStateDimension,
                              kMaxStateDimension + kMaxControlDimension));
constexpr int kMaxRrefColumns = kMaxDualColumns;
constexpr int kMaxRrefEntries = kMaxRrefRows * kMaxRrefColumns;

constexpr Scalar kScalarMax = std::numeric_limits<Scalar>::max();

__device__ inline Scalar DeviceAbs(Scalar x) { return x < Scalar{0} ? -x : x; }

__device__ inline bool DeviceFinite(Scalar x) {
  return x >= -kScalarMax && x <= kScalarMax;
}

#ifdef CLQR_CUDA_EMULATION
constexpr int kActiveWarpWidth = 1;
#else
constexpr int kActiveWarpWidth = 32;
#endif

__device__ Scalar WarpSum(Scalar value) {
#ifdef CLQR_CUDA_EMULATION
  return value;
#else
  constexpr unsigned kFullWarp = 0xffffffffu;
  for (int offset = 16; offset > 0; offset /= 2)
    value += __shfl_down_sync(kFullWarp, value, offset);
  return __shfl_sync(kFullWarp, value, 0);
#endif
}

__device__ Scalar WarpMaximum(Scalar value) {
#ifdef CLQR_CUDA_EMULATION
  return value;
#else
  constexpr unsigned kFullWarp = 0xffffffffu;
  for (int offset = 16; offset > 0; offset /= 2)
    value = fmax(value, __shfl_down_sync(kFullWarp, value, offset));
  return __shfl_sync(kFullWarp, value, 0);
#endif
}

__device__ void WarpSynchronize() {
#ifndef CLQR_CUDA_EMULATION
  __syncwarp();
#endif
}

__device__ void SetFailure(DeviceStatus *status, int code, int stage,
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
__device__ bool BlockEnabled(const DeviceStatus *status, int *enabled) {
  if (threadIdx.x == 0)
    *enabled = status->code == kDeviceOk;
  __syncthreads();
  return *enabled != 0;
}

__device__ bool BlockEnabled(const int *flag, int *enabled) {
  if (threadIdx.x == 0)
    *enabled = *flag != 0;
  __syncthreads();
  return *enabled != 0;
}

// Scale each nonzero equation before pivoting, then use partial row pivoting.
// This makes rank decisions invariant to independent equation rescaling while
// retaining the deterministic free-column convention of the CPU RREF path.
__device__ void RrefBlock(Scalar *matrix, int rows, int columns,
                          int pivot_limit, Scalar tolerance, int *pivot_columns,
                          int *pivot_rows, int *rank, int *best_row,
                          Scalar *factors) {
  for (int row = threadIdx.x; row < rows; row += blockDim.x) {
    Scalar scale = Scalar{0};
    for (int col = 0; col < pivot_limit; ++col) {
      scale = fmax(scale, DeviceAbs(matrix[row * columns + col]));
    }
    if (scale > Scalar{0}) {
      for (int col = 0; col < columns; ++col)
        matrix[row * columns + col] /= scale;
    }
  }
  if (threadIdx.x == 0)
    *rank = 0;
  __syncthreads();

  for (int col = 0; col < pivot_limit; ++col) {
    if (threadIdx.x == 0) {
      *best_row = -1;
      Scalar best = tolerance;
      for (int row = *rank; row < rows; ++row) {
        const Scalar candidate = DeviceAbs(matrix[row * columns + col]);
        if (candidate > best) {
          best = candidate;
          *best_row = row;
        }
      }
    }
    __syncthreads();
    const int selected_row = *best_row;
    // A no-pivot iteration skips the barriers below.  Ensure every thread has
    // consumed best_row before thread 0 reuses it in the next iteration.
    __syncthreads();
    if (selected_row < 0)
      continue;

    const int pivot_row = *rank;
    if (selected_row != pivot_row) {
      for (int j = threadIdx.x; j < columns; j += blockDim.x) {
        const Scalar tmp = matrix[pivot_row * columns + j];
        matrix[pivot_row * columns + j] = matrix[selected_row * columns + j];
        matrix[selected_row * columns + j] = tmp;
      }
    }
    __syncthreads();

    const Scalar pivot = matrix[pivot_row * columns + col];
    // All threads must load the pivot before any thread normalizes its entry.
    __syncthreads();
    for (int j = col + threadIdx.x; j < columns; j += blockDim.x) {
      matrix[pivot_row * columns + j] /= pivot;
    }
    __syncthreads();

    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
      factors[row] = row == pivot_row ? Scalar{0} : matrix[row * columns + col];
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
    if (*rank == rows)
      break;
  }

  for (int index = threadIdx.x; index < rows * columns; index += blockDim.x) {
    if (DeviceAbs(matrix[index]) <= tolerance)
      matrix[index] = Scalar{0};
  }
  __syncthreads();
}

__device__ bool InconsistentRref(const Scalar *matrix, int rows, int columns,
                                 int lhs_columns, Scalar lhs_tolerance,
                                 Scalar rhs_tolerance) {
  for (int row = 0; row < rows; ++row) {
    bool zero = true;
    for (int col = 0; col < lhs_columns; ++col) {
      if (DeviceAbs(matrix[row * columns + col]) > lhs_tolerance) {
        zero = false;
        break;
      }
    }
    if (zero && DeviceAbs(matrix[row * columns + lhs_columns]) > rhs_tolerance)
      return true;
  }
  return false;
}

__device__ bool InconsistentRref(const Scalar *matrix, int rows, int columns,
                                 int lhs_columns, Scalar tolerance) {
  return InconsistentRref(matrix, rows, columns, lhs_columns, tolerance,
                          tolerance);
}

// Measure the right-hand side relative to each equation's largest coefficient
// before the matrix is overwritten, so long multiplier chains are tested on a
// scale-independent residual.
__device__ Scalar ConditionedRhsScale(const Scalar *matrix, int rows,
                                      int columns, int lhs_columns,
                                      Scalar rank_tolerance) {
  Scalar scale = Scalar{1};
  for (int row = 0; row < rows; ++row) {
    Scalar lhs_scale = Scalar{0};
    for (int col = 0; col < lhs_columns; ++col) {
      const Scalar value = DeviceAbs(matrix[row * columns + col]);
      if (value > lhs_scale)
        lhs_scale = value;
    }
    Scalar rhs_scale = DeviceAbs(matrix[row * columns + lhs_columns]);
    if (lhs_scale > rank_tolerance)
      rhs_scale /= lhs_scale;
    if (rhs_scale > scale)
      scale = rhs_scale;
  }
  return scale;
}

__device__ void ExtractResidualRelation(const Scalar *matrix, int columns,
                                        int rank, const int *pivot_columns,
                                        int eliminated_columns, int left_dim,
                                        int right_dim, Relation *output) {
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
    const Scalar value =
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

// Eliminate the leading variables by orthogonally projecting the equations
// onto their left nullspace, then retain an orthonormal basis of the resulting
// affine relation.  Pivoted, reorthogonalized modified Gram--Schmidt is used in
// both directions.  Stage dimensions are bounded, so the serial work within a
// block is constant with respect to the horizon while independent tree nodes
// remain fully parallel.
__device__ void EliminateRelationOrthogonally(
    Scalar *matrix, int rows, int columns, int eliminated_columns, int left_dim,
    int right_dim, Scalar rank_tolerance, Scalar consistency_tolerance,
    Relation *relation, int *local_ok) {
  if (threadIdx.x < kActiveWarpWidth) {
    const int lane = threadIdx.x;
    const Scalar rank_threshold_squared = rank_tolerance * rank_tolerance;
    Scalar rhs_scale = Scalar{1};
    for (int row = lane; row < rows; row += kActiveWarpWidth)
      rhs_scale =
          fmax(rhs_scale, DeviceAbs(matrix[row * columns + columns - 1]));
    rhs_scale = WarpMaximum(rhs_scale);

    int eliminated_rank = 0;
    for (int basis = 0; basis < eliminated_columns; ++basis) {
      int best_column = basis;
      Scalar best_norm_squared = Scalar{-1};
      for (int candidate = basis; candidate < eliminated_columns; ++candidate) {
        Scalar norm_squared = Scalar{0};
        for (int row = lane; row < rows; row += kActiveWarpWidth) {
          const Scalar value = matrix[row * columns + candidate];
          norm_squared += value * value;
        }
        norm_squared = WarpSum(norm_squared);
        if (norm_squared > best_norm_squared) {
          best_norm_squared = norm_squared;
          best_column = candidate;
        }
      }
      if (!(best_norm_squared > rank_threshold_squared))
        break;
      if (best_column != basis) {
        for (int row = lane; row < rows; row += kActiveWarpWidth) {
          const Scalar value = matrix[row * columns + basis];
          matrix[row * columns + basis] = matrix[row * columns + best_column];
          matrix[row * columns + best_column] = value;
        }
        WarpSynchronize();
      }
      for (int pass = 0; pass < 2; ++pass) {
        for (int previous = 0; previous < basis; ++previous) {
          Scalar projection = Scalar{0};
          for (int row = lane; row < rows; row += kActiveWarpWidth) {
            projection += matrix[row * columns + basis] *
                          matrix[row * columns + previous];
          }
          projection = WarpSum(projection);
          for (int row = lane; row < rows; row += kActiveWarpWidth) {
            matrix[row * columns + basis] -=
                projection * matrix[row * columns + previous];
          }
          WarpSynchronize();
        }
      }
      Scalar norm_squared = Scalar{0};
      for (int row = lane; row < rows; row += kActiveWarpWidth) {
        const Scalar value = matrix[row * columns + basis];
        norm_squared += value * value;
      }
      norm_squared = WarpSum(norm_squared);
      if (!(norm_squared > rank_threshold_squared))
        break;
      const Scalar inverse_norm = Scalar{1} / sqrt(norm_squared);
      for (int row = lane; row < rows; row += kActiveWarpWidth)
        matrix[row * columns + basis] *= inverse_norm;
      WarpSynchronize();
      for (int candidate = basis + 1; candidate < eliminated_columns;
           ++candidate) {
        for (int pass = 0; pass < 2; ++pass) {
          Scalar projection = Scalar{0};
          for (int row = lane; row < rows; row += kActiveWarpWidth)
            projection += matrix[row * columns + candidate] *
                          matrix[row * columns + basis];
          projection = WarpSum(projection);
          for (int row = lane; row < rows; row += kActiveWarpWidth)
            matrix[row * columns + candidate] -=
                projection * matrix[row * columns + basis];
          WarpSynchronize();
        }
      }
      ++eliminated_rank;
    }

    // Apply the orthogonal projector to the outer coefficients and right-hand
    // side. The remaining rows are precisely the equations that cannot be
    // satisfied by choosing the eliminated variables.
    for (int col = eliminated_columns; col < columns; ++col) {
      for (int basis = 0; basis < eliminated_rank; ++basis) {
        for (int pass = 0; pass < 2; ++pass) {
          Scalar projection = Scalar{0};
          for (int row = lane; row < rows; row += kActiveWarpWidth)
            projection +=
                matrix[row * columns + col] * matrix[row * columns + basis];
          projection = WarpSum(projection);
          for (int row = lane; row < rows; row += kActiveWarpWidth)
            matrix[row * columns + col] -=
                projection * matrix[row * columns + basis];
          WarpSynchronize();
        }
      }
    }

    const int outer_columns = left_dim + right_dim;
    int relation_rank = 0;
    while (relation_rank < rows) {
      int best_row = relation_rank;
      Scalar best_norm_squared = Scalar{-1};
      for (int candidate = relation_rank; candidate < rows; ++candidate) {
        Scalar norm_squared = Scalar{0};
        for (int col = lane; col < outer_columns; col += kActiveWarpWidth) {
          const Scalar value =
              matrix[candidate * columns + eliminated_columns + col];
          norm_squared += value * value;
        }
        norm_squared = WarpSum(norm_squared);
        if (norm_squared > best_norm_squared) {
          best_norm_squared = norm_squared;
          best_row = candidate;
        }
      }
      if (!(best_norm_squared > rank_threshold_squared))
        break;
      if (best_row != relation_rank) {
        for (int outer = lane; outer <= outer_columns;
             outer += kActiveWarpWidth) {
          const int col = eliminated_columns + outer;
          const Scalar value = matrix[relation_rank * columns + col];
          matrix[relation_rank * columns + col] =
              matrix[best_row * columns + col];
          matrix[best_row * columns + col] = value;
        }
        WarpSynchronize();
      }
      for (int pass = 0; pass < 2; ++pass) {
        for (int previous = 0; previous < relation_rank; ++previous) {
          Scalar projection = Scalar{0};
          for (int col = lane; col < outer_columns; col += kActiveWarpWidth) {
            projection +=
                matrix[relation_rank * columns + eliminated_columns + col] *
                matrix[previous * columns + eliminated_columns + col];
          }
          projection = WarpSum(projection);
          for (int col = lane; col <= outer_columns; col += kActiveWarpWidth) {
            matrix[relation_rank * columns + eliminated_columns + col] -=
                projection *
                matrix[previous * columns + eliminated_columns + col];
          }
          WarpSynchronize();
        }
      }
      Scalar norm_squared = Scalar{0};
      for (int col = lane; col < outer_columns; col += kActiveWarpWidth) {
        const Scalar value =
            matrix[relation_rank * columns + eliminated_columns + col];
        norm_squared += value * value;
      }
      norm_squared = WarpSum(norm_squared);
      if (!(norm_squared > rank_threshold_squared))
        break;
      const Scalar inverse_norm = Scalar{1} / sqrt(norm_squared);
      for (int col = lane; col <= outer_columns; col += kActiveWarpWidth)
        matrix[relation_rank * columns + eliminated_columns + col] *=
            inverse_norm;
      WarpSynchronize();
      for (int row = relation_rank + 1; row < rows; ++row) {
        for (int pass = 0; pass < 2; ++pass) {
          Scalar projection = Scalar{0};
          for (int col = lane; col < outer_columns; col += kActiveWarpWidth) {
            projection +=
                matrix[row * columns + eliminated_columns + col] *
                matrix[relation_rank * columns + eliminated_columns + col];
          }
          projection = WarpSum(projection);
          for (int col = lane; col <= outer_columns; col += kActiveWarpWidth) {
            matrix[row * columns + eliminated_columns + col] -=
                projection *
                matrix[relation_rank * columns + eliminated_columns + col];
          }
          WarpSynchronize();
        }
      }
      ++relation_rank;
    }

    Scalar maximum_residual = Scalar{0};
    for (int row = relation_rank + lane; row < rows; row += kActiveWarpWidth)
      maximum_residual = fmax(maximum_residual,
                              DeviceAbs(matrix[row * columns + columns - 1]));
    maximum_residual = WarpMaximum(maximum_residual);
    const bool okay = maximum_residual <= consistency_tolerance * rhs_scale;
    if (lane == 0) {
      *local_ok = okay;
      relation->left_dim = left_dim;
      relation->right_dim = right_dim;
      relation->rows = relation_rank;
    }
    if (okay) {
      const int outer_entries = relation_rank * outer_columns;
      for (int entry = lane; entry < outer_entries; entry += kActiveWarpWidth) {
        const int row = entry / outer_columns;
        const int col = entry % outer_columns;
        const Scalar value = matrix[row * columns + eliminated_columns + col];
        if (col < left_dim) {
          relation->left[row * kMaxStateDimension + col] = value;
        } else {
          relation->right[row * kMaxStateDimension + col - left_dim] = value;
        }
      }
      for (int row = lane; row < relation_rank; row += kActiveWarpWidth)
        relation->rhs[row] = matrix[row * columns + columns - 1];
    }
  }
  __syncthreads();
}

// Solve an overdetermined system with column-pivoted, reorthogonalized QR.
// Free variables are set to zero in the pivoted coordinates.  The coefficient
// matrix is overwritten by the orthonormal columns.
__device__ void SolveSystemOrthogonally(Scalar *matrix, int rows, int columns,
                                        int variables, Scalar rank_tolerance,
                                        Scalar consistency_tolerance,
                                        Scalar rhs_scale, Scalar *residual_rhs,
                                        Scalar *upper, Scalar *rhs_projection,
                                        Scalar *solution, int *permutation,
                                        int *rank, int *local_ok) {
  if (threadIdx.x < kActiveWarpWidth) {
    const int lane = threadIdx.x;
    for (int index = lane; index < variables * variables;
         index += kActiveWarpWidth)
      upper[index] = Scalar{0};
    for (int variable = lane; variable < variables;
         variable += kActiveWarpWidth) {
      permutation[variable] = variable;
      rhs_projection[variable] = Scalar{0};
      solution[variable] = Scalar{0};
    }
    for (int row = lane; row < rows; row += kActiveWarpWidth)
      residual_rhs[row] = matrix[row * columns + variables];
    WarpSynchronize();

    int computed_rank = 0;
    const Scalar rank_threshold_squared = rank_tolerance * rank_tolerance;
    for (int basis = 0; basis < variables; ++basis) {
      int best_column = basis;
      Scalar best_norm_squared = Scalar{-1};
      for (int candidate = basis; candidate < variables; ++candidate) {
        Scalar norm_squared = Scalar{0};
        for (int row = lane; row < rows; row += kActiveWarpWidth) {
          const Scalar value = matrix[row * columns + candidate];
          norm_squared += value * value;
        }
        norm_squared = WarpSum(norm_squared);
        if (norm_squared > best_norm_squared) {
          best_norm_squared = norm_squared;
          best_column = candidate;
        }
      }
      if (!(best_norm_squared > rank_threshold_squared))
        break;
      if (best_column != basis) {
        for (int row = lane; row < rows; row += kActiveWarpWidth) {
          const Scalar value = matrix[row * columns + basis];
          matrix[row * columns + basis] = matrix[row * columns + best_column];
          matrix[row * columns + best_column] = value;
        }
        for (int previous = lane; previous < basis;
             previous += kActiveWarpWidth) {
          const Scalar value = upper[previous * variables + basis];
          upper[previous * variables + basis] =
              upper[previous * variables + best_column];
          upper[previous * variables + best_column] = value;
        }
        if (lane == 0) {
          const int variable = permutation[basis];
          permutation[basis] = permutation[best_column];
          permutation[best_column] = variable;
        }
        WarpSynchronize();
      }

      Scalar norm_squared = Scalar{0};
      for (int row = lane; row < rows; row += kActiveWarpWidth) {
        const Scalar value = matrix[row * columns + basis];
        norm_squared += value * value;
      }
      norm_squared = WarpSum(norm_squared);
      if (!(norm_squared > rank_threshold_squared))
        break;
      const Scalar norm = sqrt(norm_squared);
      if (lane == 0)
        upper[basis * variables + basis] = norm;
      for (int row = lane; row < rows; row += kActiveWarpWidth)
        matrix[row * columns + basis] /= norm;
      WarpSynchronize();

      for (int pass = 0; pass < 2; ++pass) {
        Scalar projection = Scalar{0};
        for (int row = lane; row < rows; row += kActiveWarpWidth)
          projection += matrix[row * columns + basis] * residual_rhs[row];
        projection = WarpSum(projection);
        if (lane == 0)
          rhs_projection[basis] += projection;
        for (int row = lane; row < rows; row += kActiveWarpWidth)
          residual_rhs[row] -= projection * matrix[row * columns + basis];
        WarpSynchronize();
      }
      for (int candidate = basis + 1; candidate < variables; ++candidate) {
        for (int pass = 0; pass < 2; ++pass) {
          Scalar projection = Scalar{0};
          for (int row = lane; row < rows; row += kActiveWarpWidth) {
            projection += matrix[row * columns + basis] *
                          matrix[row * columns + candidate];
          }
          projection = WarpSum(projection);
          if (lane == 0)
            upper[basis * variables + candidate] += projection;
          for (int row = lane; row < rows; row += kActiveWarpWidth) {
            matrix[row * columns + candidate] -=
                projection * matrix[row * columns + basis];
          }
          WarpSynchronize();
        }
      }
      ++computed_rank;
    }

    Scalar maximum_residual = Scalar{0};
    for (int row = lane; row < rows; row += kActiveWarpWidth)
      maximum_residual = fmax(maximum_residual, DeviceAbs(residual_rhs[row]));
    maximum_residual = WarpMaximum(maximum_residual);
    const bool okay = maximum_residual <= consistency_tolerance * rhs_scale;
    WarpSynchronize();
    if (lane == 0) {
      *rank = computed_rank;
      *local_ok = okay;
    }
    if (okay && lane == 0) {
      Scalar pivoted_solution[kMaxStateDimension]{};
      for (int reverse = 0; reverse < computed_rank; ++reverse) {
        const int row = computed_rank - 1 - reverse;
        Scalar value = rhs_projection[row];
        for (int col = row + 1; col < computed_rank; ++col)
          value -= upper[row * variables + col] * pivoted_solution[col];
        pivoted_solution[row] = value / upper[row * variables + row];
      }
      for (int variable = 0; variable < computed_rank; ++variable)
        solution[permutation[variable]] = pivoted_solution[variable];
    }
  }
  __syncthreads();
}

__global__ void
BuildPrimalLeavesKernel(const PackedStage *stages, int stage_count,
                        const PackedTerminal *terminal_ptr,
                        Scalar rank_tolerance, Scalar consistency_tolerance,
                        Relation *leaves, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index > stage_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  __shared__ Scalar matrix[kMaxRrefEntries];
  __shared__ Scalar factors[kMaxRrefRows];
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

  const PackedTerminal &terminal = *terminal_ptr;

  if (threadIdx.x == 0) {
    if (index == stage_count) {
      rows = terminal.state;
      columns = terminal.n + 1;
      eliminated = 0;
      left_dim = terminal.n;
      right_dim = 0;
    } else {
      const PackedStage &s = stages[index];
      rows = s.mixed + s.state + s.next_n;
      columns = s.m + s.n + s.next_n + 1;
      eliminated = s.m;
      left_dim = s.n;
      right_dim = s.next_n;
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = Scalar{0};
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
    const PackedStage &s = stages[index];
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
      matrix[(dynamics_row + row) * columns + s.m + s.n + row] = Scalar{1};
      matrix[(dynamics_row + row) * columns + columns - 1] = s.c[row];
    }
  }
  __syncthreads();
  RrefBlock(matrix, rows, columns, columns - 1, rank_tolerance, pivot_columns,
            pivot_rows, &rank, &best_row, factors);
  if (threadIdx.x == 0) {
    local_ok = !InconsistentRref(matrix, rows, columns, columns - 1,
                                 consistency_tolerance);
    if (!local_ok)
      SetFailure(status, kDeviceInfeasible, index, 1);
  }
  __syncthreads();
  if (!local_ok)
    return;
  ExtractResidualRelation(matrix, columns, rank, pivot_columns, eliminated,
                          left_dim, right_dim, &leaves[index]);
}

__device__ void ComposeRelationsBlock(
    const Relation &first, const Relation &second, Scalar rank_tolerance,
    Scalar consistency_tolerance, Relation *output, DeviceStatus *status,
    int stage, int inconsistency_code, int inconsistency_detail, Scalar *matrix,
    Scalar *factors, int *pivot_columns, int *pivot_rows, int *rank,
    int *best_row, int *local_ok, bool orthonormalize_output = false) {
  if (first.right_dim != second.left_dim) {
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, stage, 2);
    return;
  }
  const int shared = first.right_dim;
  const int rows = first.rows + second.rows;
  const int columns = shared + first.left_dim + second.right_dim + 1;
  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = Scalar{0};
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
  if (orthonormalize_output) {
    EliminateRelationOrthogonally(matrix, rows, columns, shared, first.left_dim,
                                  second.right_dim, rank_tolerance,
                                  consistency_tolerance, output, local_ok);
    if (threadIdx.x == 0 && !*local_ok)
      SetFailure(status, inconsistency_code, stage, inconsistency_detail);
    __syncthreads();
    return;
  }
  RrefBlock(matrix, rows, columns, columns - 1, rank_tolerance, pivot_columns,
            pivot_rows, rank, best_row, factors);
  if (threadIdx.x == 0) {
    *local_ok = !InconsistentRref(matrix, rows, columns, columns - 1,
                                  consistency_tolerance);
    if (!*local_ok)
      SetFailure(status, inconsistency_code, stage, inconsistency_detail);
  }
  __syncthreads();
  if (!*local_ok)
    return;
  ExtractResidualRelation(matrix, columns, *rank, pivot_columns, shared,
                          first.left_dim, second.right_dim, output);
}

__device__ void CopyRelationBlock(const Relation &input, Relation *output) {
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

__device__ bool InvalidScanRelation(const Relation &relation) {
  return relation.left_dim < 0;
}

__device__ void SetInvalidScanRelation(Relation *relation) {
  if (threadIdx.x == 0) {
    relation->left_dim = -1;
    relation->right_dim = 0;
    relation->rows = 0;
  }
}

__device__ void
ComposeScanRelationBlock(const Relation &first, const Relation &second,
                         Scalar rank_tolerance, Scalar consistency_tolerance,
                         Relation *output, DeviceStatus *status, int stage,
                         int inconsistency_detail, Scalar *matrix,
                         Scalar *factors, int *pivot_columns, int *pivot_rows,
                         int *rank, int *best_row, int *local_ok) {
  if (InvalidScanRelation(first)) {
    if (InvalidScanRelation(second)) {
      SetInvalidScanRelation(output);
    } else {
      CopyRelationBlock(second, output);
    }
    return;
  }
  if (InvalidScanRelation(second)) {
    CopyRelationBlock(first, output);
    return;
  }
  ComposeRelationsBlock(first, second, rank_tolerance, consistency_tolerance,
                        output, status, stage, kDeviceInfeasible,
                        inconsistency_detail, matrix, factors, pivot_columns,
                        pivot_rows, rank, best_row, local_ok);
}

__global__ void SeedRelationTreeKernel(const Relation *leaves, int count,
                                       Relation *tree) {
  const int index = blockIdx.x;
  if (index >= count)
    return;
  CopyRelationBlock(leaves[index], &tree[index]);
}

__global__ void ReduceRelationTreeLevelKernel(Relation *tree, int child_offset,
                                              int parent_offset,
                                              int child_count, int parent_count,
                                              Scalar rank_tolerance,
                                              Scalar consistency_tolerance,
                                              DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  const int left = child_offset + 2 * index;
  if (2 * index + 1 >= child_count) {
    CopyRelationBlock(tree[left], &tree[parent_offset + index]);
    return;
  }
  const int right = left + 1;
  __shared__ Scalar matrix[kMaxRrefRows * kMaxRelationColumns];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;
  ComposeRelationsBlock(tree[left], tree[right], rank_tolerance,
                        consistency_tolerance, &tree[parent_offset + index],
                        status, index, kDeviceInfeasible, 19, matrix, factors,
                        pivot_columns, pivot_rows, &rank, &best_row, &local_ok);
}

__global__ void InitializeRelationContextRootKernel(Relation *tree,
                                                    int root_offset) {
  if (blockIdx.x == 0)
    SetInvalidScanRelation(&tree[root_offset]);
}

__global__ void ExpandRelationContextLevelKernel(
    Relation *tree, int child_offset, int parent_offset, int child_count,
    int parent_count, Scalar rank_tolerance, Scalar consistency_tolerance,
    DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  const int left = child_offset + 2 * index;
  const Relation &parent_context = tree[parent_offset + index];
  if (2 * index + 1 >= child_count) {
    CopyRelationBlock(parent_context, &tree[left]);
    return;
  }
  const int right = left + 1;
  __shared__ Scalar matrix[kMaxRrefRows * kMaxRelationColumns];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;
  ComposeScanRelationBlock(tree[right], parent_context, rank_tolerance,
                           consistency_tolerance, &tree[left], status, index,
                           20, matrix, factors, pivot_columns, pivot_rows,
                           &rank, &best_row, &local_ok);
  __syncthreads();
  CopyRelationBlock(parent_context, &tree[right]);
}

__global__ void FinalizeRelationSuffixKernel(Relation *leaves, int count,
                                             const Relation *exclusive_contexts,
                                             Scalar rank_tolerance,
                                             Scalar consistency_tolerance,
                                             DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  const Relation &after = exclusive_contexts[index];
  if (InvalidScanRelation(after))
    return;
  __shared__ Scalar matrix[kMaxRrefRows * kMaxRelationColumns];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;
  ComposeRelationsBlock(leaves[index], after, rank_tolerance,
                        consistency_tolerance, &leaves[index], status, index,
                        kDeviceInfeasible, 21, matrix, factors, pivot_columns,
                        pivot_rows, &rank, &best_row, &local_ok);
}

__global__ void StateParamKernel(const Relation *suffix, int count,
                                 StateParam *params, DeviceStatus *status,
                                 Scalar tolerance) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count || status->code != kDeviceOk)
    return;
  const Relation &relation = suffix[index];
  if (relation.right_dim != 0 || relation.rows > relation.left_dim) {
    SetFailure(status, kDeviceNumericalFailure, index, 4);
    return;
  }
  StateParam &out = params[index];
  out.physical_dim = relation.left_dim;
  for (int row = 0; row < relation.left_dim; ++row) {
    out.t[row] = Scalar{0};
    for (int col = 0; col < relation.left_dim; ++col)
      out.T[row * kMaxStateDimension + col] = Scalar{0};
  }
  bool pivot[kMaxStateDimension]{};
  int pivot_row[kMaxStateDimension];
  for (int i = 0; i < kMaxStateDimension; ++i)
    pivot_row[i] = -1;
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
    if (!pivot[col])
      out.free_columns[reduced++] = col;
  }
  out.reduced_dim = reduced;
  for (int col = 0; col < relation.left_dim; ++col) {
    if (pivot[col]) {
      const int row = pivot_row[col];
      const Scalar diagonal = relation.left[row * kMaxStateDimension + col];
      out.t[col] = relation.rhs[row] / diagonal;
      for (int free = 0; free < reduced; ++free) {
        out.T[col * kMaxStateDimension + free] =
            -relation.left[row * kMaxStateDimension + out.free_columns[free]] /
            diagonal;
      }
    }
  }
  for (int free = 0; free < reduced; ++free) {
    out.T[out.free_columns[free] * kMaxStateDimension + free] = Scalar{1};
  }
}

__global__ void PackReducedDimensionsKernel(const StateParam *state_params,
                                            const ControlParam *control_params,
                                            int stage_count,
                                            int *state_dimensions,
                                            int *control_dimensions) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > stage_count)
    return;
  state_dimensions[2 * index] = state_params[index].physical_dim;
  state_dimensions[2 * index + 1] = state_params[index].reduced_dim;
  if (index < stage_count) {
    control_dimensions[2 * index] = control_params[index].physical_dim;
    control_dimensions[2 * index + 1] = control_params[index].reduced_dim;
  }
}

} // namespace
} // namespace detail
} // namespace cuda
} // namespace clqr

#ifndef CLQR_CUDA_EMULATION
namespace clqr {
namespace cuda {
namespace detail {
namespace {

__global__ void ReduceStagesKernel(const PackedStage *, const Relation *,
                                   const StateParam *, int, Scalar, Scalar,
                                   ControlParam *, ReducedStage *,
                                   DeviceStatus *);
__global__ void ReduceTerminalKernel(const PackedTerminal *, const StateParam *,
                                     int, ReducedTerminal *);
__global__ void InitialReducedStateKernel(const StateParam *, const Scalar *,
                                          Scalar *, Scalar, DeviceStatus *);
__global__ void BuildValueElementsKernel(const ReducedStage *,
                                         const ReducedTerminal *, int, Scalar,
                                         ValueElement *, int *, DeviceStatus *);
__global__ void SeedValueTreeKernel(const ValueElement *, int, ValueElement *);
__global__ void ReduceValueTreeLevelKernel(ValueElement *, int, int, int, int,
                                           Scalar, int *);
__global__ void InitializeValueContextRootKernel(ValueElement *, int);
__global__ void ExpandValueContextLevelKernel(ValueElement *, int, int, int,
                                              int, Scalar, int *);
__global__ void FinalizeValueSuffixKernel(ValueElement *, int,
                                          const ValueElement *, Scalar, int *);
__global__ void FeedbackKernel(const ReducedStage *, const ValueElement *, int,
                               Scalar, Feedback *, DeviceStatus *);
__global__ void SequentialRiccatiKernel(const ReducedStage *, int, Scalar,
                                        ValueElement *, Feedback *,
                                        DeviceStatus *);
__global__ void InitializeAffineMapsKernel(const Feedback *, int, AffineMap *);
__global__ void SeedAffineTreeKernel(const AffineMap *, int, AffineMap *);
__global__ void ReduceAffineTreeLevelKernel(AffineMap *, int, int, int, int,
                                            DeviceStatus *);
__global__ void InitializeAffineContextRootKernel(AffineMap *, int);
__global__ void ExpandAffineContextLevelKernel(AffineMap *, int, int, int, int,
                                               DeviceStatus *);
__global__ void FinalizeAffinePrefixKernel(AffineMap *, int, const AffineMap *,
                                           DeviceStatus *);
__global__ void EvaluateReducedStatesKernel(const AffineMap *, int,
                                            const Scalar *, Scalar *);
__global__ void ReconstructPrimalKernel(const StateParam *,
                                        const ControlParam *, const Feedback *,
                                        const Scalar *, int, Scalar *,
                                        Scalar *);
__global__ void BuildDualLeavesKernel(const PackedStage *,
                                      const PackedTerminal *, int, int,
                                      const Scalar *, const Scalar *, Scalar,
                                      Scalar, Relation *, DeviceStatus *);
__global__ void ReduceDualTreeLevelKernel(const Relation *, int, int, int, int,
                                          Scalar, Scalar, Relation *,
                                          DeviceStatus *);
__global__ void SolveDualRootKernel(const Relation *, NodeValue *,
                                    DeviceStatus *, Scalar);
__global__ void ExpandDualTreeLevelKernel(const Relation *, int, int, int, int,
                                          Scalar, Scalar, const NodeValue *,
                                          NodeValue *, DeviceStatus *);
__global__ void
RecoverLocalMultipliersKernel(const PackedStage *, const PackedTerminal *, int,
                              const Scalar *, const Scalar *, const NodeValue *,
                              Scalar, Scalar, Scalar *, Scalar *, Scalar *,
                              Scalar *, Scalar *, DeviceStatus *);

void CudaCheck(cudaError_t error, const char *operation) {
  if (error == cudaSuccess)
    return;
  throw std::runtime_error(std::string(operation) + ": " +
                           cudaGetErrorString(error));
}

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { Allocate(count); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
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
    CudaCheck(cudaMalloc(reinterpret_cast<void **>(&data_), count_ * sizeof(T)),
              "cudaMalloc");
  }
  void Reserve(std::size_t count) {
    const std::size_t required = std::max<std::size_t>(count, 1);
    if (count_ < required)
      Allocate(required);
  }
  void Release() {
    if (data_ != nullptr)
      cudaFree(data_);
    data_ = nullptr;
    count_ = 0;
  }
  T *get() { return data_; }
  const T *get() const { return data_; }
  std::size_t count() const { return count_; }

private:
  T *data_ = nullptr;
  std::size_t count_ = 0;
};

template <typename T> class PinnedBuffer {
public:
  PinnedBuffer() = default;
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;
  ~PinnedBuffer() { Release(); }

  void Reserve(std::size_t count) {
    const std::size_t required = std::max<std::size_t>(count, 1);
    if (capacity_ >= required)
      return;
    Release();
    CudaCheck(
        cudaMallocHost(reinterpret_cast<void **>(&data_), required * sizeof(T)),
        "cudaMallocHost");
    capacity_ = required;
  }
  void Resize(std::size_t count) {
    Reserve(count);
    size_ = count;
  }
  void Release() {
    if (data_ != nullptr)
      cudaFreeHost(data_);
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }
  T *data() { return data_; }
  const T *data() const { return data_; }
  T *begin() { return data_; }
  T *end() { return data_ + size_; }
  T &operator[](std::size_t index) { return data_[index]; }
  const T &operator[](std::size_t index) const { return data_[index]; }
  std::size_t size() const { return size_; }

private:
  T *data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

struct WorkspaceStorage {
  int device = -1;
  cudaEvent_t event_start = nullptr;
  cudaEvent_t event_stop = nullptr;
  DeviceBuffer<PackedStage> device_stages;
  DeviceBuffer<PackedTerminal> device_terminal;
  DeviceBuffer<Scalar> device_initial;
  DeviceBuffer<DeviceStatus> device_status;
  DeviceBuffer<Relation> relation_leaves;
  DeviceBuffer<Relation> relation_scan;
  DeviceBuffer<StateParam> state_params;
  DeviceBuffer<ControlParam> control_params;
  DeviceBuffer<ReducedStage> reduced_stages;
  DeviceBuffer<ReducedTerminal> reduced_terminal;
  DeviceBuffer<Scalar> reduced_initial;
  DeviceBuffer<ValueElement> value_leaves;
  DeviceBuffer<ValueElement> value_scan;
  DeviceBuffer<Feedback> feedback;
  DeviceBuffer<int> parallel_ok;
  DeviceBuffer<AffineMap> map_leaves;
  DeviceBuffer<AffineMap> map_scan;
  DeviceBuffer<Scalar> reduced_states;
  DeviceBuffer<Scalar> states;
  DeviceBuffer<Scalar> controls;
  DeviceBuffer<int> state_dimensions;
  DeviceBuffer<int> control_dimensions;
  DeviceBuffer<Relation> dual_tree;
  DeviceBuffer<NodeValue> dual_values;
  DeviceBuffer<Scalar> initial_multiplier;
  DeviceBuffer<Scalar> dynamics_multipliers;
  DeviceBuffer<Scalar> mixed_multipliers;
  DeviceBuffer<Scalar> state_multipliers;
  DeviceBuffer<Scalar> terminal_multiplier;

  PinnedBuffer<PackedStage> host_stages;
  PinnedBuffer<PackedTerminal> host_terminal;
  PinnedBuffer<Scalar> host_initial;
  PinnedBuffer<Scalar> host_states;
  PinnedBuffer<Scalar> host_controls;
  PinnedBuffer<Scalar> host_initial_multiplier;
  PinnedBuffer<Scalar> host_dynamics;
  PinnedBuffer<Scalar> host_mixed;
  PinnedBuffer<Scalar> host_state_multipliers;
  PinnedBuffer<Scalar> host_terminal_multiplier;
  PinnedBuffer<int> host_state_dimensions;
  PinnedBuffer<int> host_control_dimensions;
  PinnedBuffer<int> host_parallel_ok;
  PinnedBuffer<DeviceStatus> host_status;
  std::vector<int> node_level_offsets;
  std::vector<int> node_level_counts;
  std::vector<int> stage_level_offsets;
  std::vector<int> stage_level_counts;

  ~WorkspaceStorage() {
    if (device >= 0)
      cudaSetDevice(device);
    if (event_stop != nullptr)
      cudaEventDestroy(event_stop);
    if (event_start != nullptr)
      cudaEventDestroy(event_start);
  }

  void Reserve(int requested_device, int stage_count, int node_count) {
    if (device >= 0 && device != requested_device) {
      throw std::invalid_argument(
          "a CUDA workspace cannot be reused across devices");
    }
    device = requested_device;
    if (event_start == nullptr)
      CudaCheck(cudaEventCreate(&event_start), "cudaEventCreate(start)");
    if (event_stop == nullptr)
      CudaCheck(cudaEventCreate(&event_stop), "cudaEventCreate(stop)");
    int total_tree_nodes = 0;
    for (int level_count = node_count;; level_count = (level_count + 1) / 2) {
      total_tree_nodes += level_count;
      if (level_count == 1)
        break;
    }
    int total_stage_tree_nodes = 0;
    for (int level_count = std::max(stage_count, 1);;
         level_count = (level_count + 1) / 2) {
      total_stage_tree_nodes += level_count;
      if (level_count == 1)
        break;
    }
    device_stages.Reserve(stage_count);
    device_terminal.Reserve(1);
    device_initial.Reserve(kMaxStateDimension);
    device_status.Reserve(1);
    relation_leaves.Reserve(node_count);
    relation_scan.Reserve(total_tree_nodes);
    state_params.Reserve(node_count);
    control_params.Reserve(stage_count);
    reduced_stages.Reserve(stage_count);
    reduced_terminal.Reserve(1);
    reduced_initial.Reserve(kMaxStateDimension);
    value_leaves.Reserve(node_count);
    value_scan.Reserve(total_tree_nodes);
    feedback.Reserve(stage_count);
    parallel_ok.Reserve(1);
    map_leaves.Reserve(stage_count);
    map_scan.Reserve(total_stage_tree_nodes);
    reduced_states.Reserve(static_cast<std::size_t>(node_count) *
                           kMaxStateDimension);
    states.Reserve(static_cast<std::size_t>(node_count) * kMaxStateDimension);
    controls.Reserve(static_cast<std::size_t>(std::max(stage_count, 1)) *
                     kMaxControlDimension);
    state_dimensions.Reserve(static_cast<std::size_t>(2) * node_count);
    control_dimensions.Reserve(static_cast<std::size_t>(2) * stage_count);
    dual_tree.Reserve(total_tree_nodes);
    dual_values.Reserve(total_tree_nodes);
    initial_multiplier.Reserve(kMaxStateDimension);
    dynamics_multipliers.Reserve(
        static_cast<std::size_t>(std::max(stage_count, 1)) *
        kMaxStateDimension);
    mixed_multipliers.Reserve(
        static_cast<std::size_t>(std::max(stage_count, 1)) *
        kMaxMixedConstraints);
    state_multipliers.Reserve(
        static_cast<std::size_t>(std::max(stage_count, 1)) *
        kMaxStateConstraints);
    terminal_multiplier.Reserve(kMaxStateConstraints);

    host_stages.Resize(stage_count);
    host_terminal.Resize(1);
    host_initial.Resize(kMaxStateDimension);
    host_states.Resize(static_cast<std::size_t>(node_count) *
                       kMaxStateDimension);
    host_controls.Resize(static_cast<std::size_t>(std::max(stage_count, 1)) *
                         kMaxControlDimension);
    host_initial_multiplier.Resize(kMaxStateDimension);
    host_dynamics.Resize(static_cast<std::size_t>(std::max(stage_count, 1)) *
                         kMaxStateDimension);
    host_mixed.Resize(static_cast<std::size_t>(std::max(stage_count, 1)) *
                      kMaxMixedConstraints);
    host_state_multipliers.Resize(
        static_cast<std::size_t>(std::max(stage_count, 1)) *
        kMaxStateConstraints);
    host_terminal_multiplier.Resize(kMaxStateConstraints);
    host_state_dimensions.Resize(static_cast<std::size_t>(2) * node_count);
    host_control_dimensions.Resize(static_cast<std::size_t>(2) * stage_count);
    host_parallel_ok.Resize(1);
    host_status.Resize(1);
  }
};

template <typename Function>
double TimeGpu(WorkspaceStorage &workspace, Function &&function) {
  CudaCheck(cudaEventRecord(workspace.event_start), "cudaEventRecord(start)");
  function();
  CudaCheck(cudaGetLastError(), "CUDA kernel launch");
  CudaCheck(cudaEventRecord(workspace.event_stop), "cudaEventRecord(stop)");
  CudaCheck(cudaEventSynchronize(workspace.event_stop), "cudaEventSynchronize");
  float milliseconds = 0.0f;
  CudaCheck(cudaEventElapsedTime(&milliseconds, workspace.event_start,
                                 workspace.event_stop),
            "cudaEventElapsedTime");
  return milliseconds;
}

bool Finite(const Matrix &matrix) {
  for (Scalar value : matrix.data()) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

bool Finite(const Vector &vector) {
  for (Scalar value : vector.data()) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

void Require(bool condition, const std::string &message) {
  if (!condition)
    throw std::invalid_argument(message);
}

void ValidateCudaProblem(const Problem &problem, const Options &options) {
  Require(std::isfinite(options.tolerance) && options.tolerance > Scalar{0},
          "CUDA tolerance must be finite and positive");
  Require(options.device >= 0, "CUDA device index must be nonnegative");
  // Node counts, scan padding, and compact-tree offsets are stored in signed
  // ints.
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
    const Stage &s = problem.stages[i];
    const std::size_t n = s.A.cols();
    const std::size_t next = s.A.rows();
    const std::size_t m = s.B.cols();
    Require(n <= kMaxStateDimension && next <= kMaxStateDimension,
            "state dimension exceeds CUDA limit at stage " + std::to_string(i));
    Require(m <= kMaxControlDimension,
            "control dimension exceeds CUDA limit at stage " +
                std::to_string(i));
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
    Require(next == expected_next,
            "neighboring state dimensions differ at stage " +
                std::to_string(i));
    Require(Finite(s.A) && Finite(s.B) && Finite(s.c) && Finite(s.Q) &&
                Finite(s.R) && Finite(s.M) && Finite(s.q) && Finite(s.r) &&
                Finite(s.C) && Finite(s.D) && Finite(s.d) && Finite(s.E) &&
                Finite(s.e),
            "problem contains a non-finite value at stage " +
                std::to_string(i));
  }
}

template <std::size_t Size>
void PackMatrix(const Matrix &source, Scalar (&target)[Size],
                std::size_t stride) {
  for (std::size_t row = 0; row < source.rows(); ++row) {
    for (std::size_t col = 0; col < source.cols(); ++col) {
      target[row * stride + col] = source(row, col);
    }
  }
}

template <std::size_t Size>
void PackVector(const Vector &source, Scalar (&target)[Size]) {
  for (std::size_t i = 0; i < source.size(); ++i)
    target[i] = source[i];
}

PackedStage PackStage(const Stage &source) {
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

PackedTerminal PackTerminal(const Problem &problem) {
  PackedTerminal out;
  out.n = static_cast<int>(problem.terminal_Q.rows());
  out.state = static_cast<int>(problem.terminal_E.rows());
  PackMatrix(problem.terminal_Q, out.Q, kMaxStateDimension);
  PackVector(problem.terminal_q, out.q);
  PackMatrix(problem.terminal_E, out.E, kMaxStateDimension);
  PackVector(problem.terminal_e, out.e);
  return out;
}

std::string DeviceFailureMessage(const DeviceStatus &status) {
  std::ostringstream out;
  if (status.code == kDeviceInfeasible) {
    out << "CUDA feasibility elimination found an inconsistent relation";
  } else {
    out << "CUDA backend encountered a rank or consistency failure";
  }
  if (status.stage >= 0)
    out << " at stage/node " << status.stage;
  out << " (diagnostic " << status.detail << ")";
  return out.str();
}

bool ApplyDeviceFailure(const DeviceStatus &status, Solution *solution) {
  if (status.code == kDeviceOk)
    return false;
  solution->status = status.code == kDeviceInfeasible
                         ? SolveStatus::kInfeasible
                         : SolveStatus::kNumericalFailure;
  solution->message = DeviceFailureMessage(status);
  return true;
}

Scalar ObjectiveFromPacked(const PackedStage *stages, std::size_t stage_count,
                           const PackedTerminal &terminal, const Scalar *states,
                           const Scalar *controls) {
  Scalar objective = Scalar{0};
  for (std::size_t i = 0; i < stage_count; ++i) {
    const PackedStage &s = stages[i];
    const Scalar *x = states + i * kMaxStateDimension;
    const Scalar *u = controls + i * kMaxControlDimension;
    for (int row = 0; row < s.n; ++row) {
      objective += s.q[row] * x[row];
      for (int col = 0; col < s.n; ++col)
        objective +=
            Scalar{0.5} * x[row] * s.Q[row * kMaxStateDimension + col] * x[col];
      for (int col = 0; col < s.m; ++col)
        objective += x[row] * s.M[row * kMaxControlDimension + col] * u[col];
    }
    for (int row = 0; row < s.m; ++row) {
      objective += s.r[row] * u[row];
      for (int col = 0; col < s.m; ++col)
        objective += Scalar{0.5} * u[row] *
                     s.R[row * kMaxControlDimension + col] * u[col];
    }
  }
  const Scalar *x = states + stage_count * kMaxStateDimension;
  for (int row = 0; row < terminal.n; ++row) {
    objective += terminal.q[row] * x[row];
    for (int col = 0; col < terminal.n; ++col)
      objective += Scalar{0.5} * x[row] *
                   terminal.Q[row * kMaxStateDimension + col] * x[col];
  }
  return objective;
}

Solution &SolveImpl(const Problem &problem, WorkspaceStorage &workspace,
                    Solution &solution, const Options &options) {
  ValidateCudaProblem(problem, options);
  int device_count = 0;
  CudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  Require(options.device < device_count, "CUDA device index is out of range");
  CudaCheck(cudaSetDevice(options.device), "cudaSetDevice");
  const auto total_start = std::chrono::steady_clock::now();
  solution.status = SolveStatus::kInvalidInput;
  solution.message.clear();
  solution.used_parallel_riccati = false;
  solution.objective = Scalar{0};
  solution.timings = Timings{};
  const int stage_count = static_cast<int>(problem.stages.size());
  const int node_count = stage_count + 1;
  workspace.Reserve(options.device, stage_count, node_count);
  auto &level_offsets = workspace.node_level_offsets;
  auto &level_counts = workspace.node_level_counts;
  level_offsets.assign(1, 0);
  level_counts.assign(1, node_count);
  int node_tree_size = node_count;
  while (level_counts.back() > 1) {
    level_offsets.push_back(node_tree_size);
    level_counts.push_back((level_counts.back() + 1) / 2);
    node_tree_size += level_counts.back();
  }
  const int feasibility_scan_levels = static_cast<int>(level_counts.size()) - 1;
  auto &stage_level_offsets = workspace.stage_level_offsets;
  auto &stage_level_counts = workspace.stage_level_counts;
  stage_level_offsets.assign(1, 0);
  stage_level_counts.assign(1, std::max(stage_count, 1));
  int stage_tree_size = stage_level_counts.front();
  while (stage_level_counts.back() > 1) {
    stage_level_offsets.push_back(stage_tree_size);
    stage_level_counts.push_back((stage_level_counts.back() + 1) / 2);
    stage_tree_size += stage_level_counts.back();
  }
  auto &host_stages = workspace.host_stages;
  for (std::size_t index = 0; index < problem.stages.size(); ++index)
    host_stages[index] = PackStage(problem.stages[index]);
  PackedTerminal &terminal = workspace.host_terminal[0];
  terminal = PackTerminal(problem);
  auto &host_initial = workspace.host_initial;
  std::fill(host_initial.begin(), host_initial.end(), Scalar{0});
  for (std::size_t i = 0; i < problem.initial_state.size(); ++i)
    host_initial[i] = problem.initial_state[i];

  auto &device_stages = workspace.device_stages;
  auto &device_terminal = workspace.device_terminal;
  auto &device_initial = workspace.device_initial;
  auto &device_status = workspace.device_status;
  auto &relation_a = workspace.relation_leaves;
  auto &relation_b = workspace.relation_scan;
  auto &state_params = workspace.state_params;
  auto &control_params = workspace.control_params;
  auto &reduced_stages = workspace.reduced_stages;
  auto &reduced_terminal = workspace.reduced_terminal;
  auto &reduced_initial = workspace.reduced_initial;

  solution.timings.upload_ms = TimeGpu(workspace, [&] {
    if (stage_count > 0) {
      CudaCheck(cudaMemcpyAsync(device_stages.get(), host_stages.data(),
                                host_stages.size() * sizeof(PackedStage),
                                cudaMemcpyHostToDevice),
                "upload stages");
    }
    CudaCheck(cudaMemcpyAsync(device_terminal.get(), &terminal,
                              sizeof(PackedTerminal), cudaMemcpyHostToDevice),
              "upload terminal data");
    CudaCheck(cudaMemcpyAsync(device_initial.get(), host_initial.data(),
                              kMaxStateDimension * sizeof(Scalar),
                              cudaMemcpyHostToDevice),
              "upload initial state");
    CudaCheck(cudaMemsetAsync(device_status.get(), 0, sizeof(DeviceStatus)),
              "clear CUDA status");
  });

  Scalar feasibility_consistency_tolerance = std::max(
      options.tolerance, kMinimumFeasibilityConsistencyTolerance *
                             static_cast<Scalar>(feasibility_scan_levels + 2));
  solution.timings.feasibility_ms = TimeGpu(workspace, [&] {
    BuildPrimalLeavesKernel<<<node_count, kThreads>>>(
        device_stages.get(), stage_count, device_terminal.get(),
        options.tolerance, feasibility_consistency_tolerance, relation_a.get(),
        device_status.get());
    SeedRelationTreeKernel<<<node_count, kThreads>>>(
        relation_a.get(), node_count, relation_b.get());
    for (std::size_t level = 0; level + 1 < level_counts.size(); ++level) {
      ReduceRelationTreeLevelKernel<<<level_counts[level + 1], kThreads>>>(
          relation_b.get(), level_offsets[level], level_offsets[level + 1],
          level_counts[level], level_counts[level + 1], options.tolerance,
          feasibility_consistency_tolerance, device_status.get());
    }
    InitializeRelationContextRootKernel<<<1, kThreads>>>(relation_b.get(),
                                                         level_offsets.back());
    for (int level = static_cast<int>(level_counts.size()) - 2; level >= 0;
         --level) {
      ExpandRelationContextLevelKernel<<<level_counts[level + 1], kThreads>>>(
          relation_b.get(), level_offsets[level], level_offsets[level + 1],
          level_counts[level], level_counts[level + 1], options.tolerance,
          feasibility_consistency_tolerance, device_status.get());
    }
    FinalizeRelationSuffixKernel<<<node_count, kThreads>>>(
        relation_a.get(), node_count, relation_b.get(), options.tolerance,
        feasibility_consistency_tolerance, device_status.get());
    const int blocks = (node_count + kThreads - 1) / kThreads;
    StateParamKernel<<<blocks, kThreads>>>(
        relation_a.get(), node_count, state_params.get(), device_status.get(),
        options.tolerance);
    CudaCheck(cudaMemcpyAsync(workspace.host_status.data(), device_status.get(),
                              sizeof(DeviceStatus), cudaMemcpyDeviceToHost),
              "read feasibility status");
  });
  Relation *suffix = relation_a.get();
  DeviceStatus status = workspace.host_status[0];
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }

  solution.timings.reduction_ms = TimeGpu(workspace, [&] {
    if (stage_count > 0) {
      ReduceStagesKernel<<<stage_count, kThreads>>>(
          device_stages.get(), suffix, state_params.get(), stage_count,
          options.tolerance, feasibility_consistency_tolerance,
          control_params.get(), reduced_stages.get(), device_status.get());
    }
    ReduceTerminalKernel<<<1, kThreads>>>(device_terminal.get(),
                                          state_params.get(), stage_count,
                                          reduced_terminal.get());
    InitialReducedStateKernel<<<1, kThreads>>>(
        state_params.get(), device_initial.get(), reduced_initial.get(),
        options.tolerance, device_status.get());
    CudaCheck(cudaMemcpyAsync(workspace.host_status.data(), device_status.get(),
                              sizeof(DeviceStatus), cudaMemcpyDeviceToHost),
              "read reduction status");
  });
  status = workspace.host_status[0];
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }

  auto &value_a = workspace.value_leaves;
  auto &value_b = workspace.value_scan;
  auto &feedback = workspace.feedback;
  auto &parallel_ok = workspace.parallel_ok;
  ValueElement *value_suffix = value_a.get();
  int &host_parallel_ok = workspace.host_parallel_ok[0];
  host_parallel_ok = 1;
  solution.timings.riccati_ms = TimeGpu(workspace, [&] {
    CudaCheck(cudaMemcpyAsync(parallel_ok.get(), &host_parallel_ok, sizeof(int),
                              cudaMemcpyHostToDevice),
              "initialize parallel Riccati flag");
    BuildValueElementsKernel<<<node_count, kThreads>>>(
        reduced_stages.get(), reduced_terminal.get(), stage_count,
        options.tolerance, value_a.get(), parallel_ok.get(),
        device_status.get());
    CudaCheck(cudaMemcpyAsync(&host_parallel_ok, parallel_ok.get(), sizeof(int),
                              cudaMemcpyDeviceToHost),
              "read parallel Riccati flag");
  });

  if (host_parallel_ok != 0) {
    solution.timings.riccati_ms += TimeGpu(workspace, [&] {
      SeedValueTreeKernel<<<node_count, kThreads>>>(value_a.get(), node_count,
                                                    value_b.get());
      for (std::size_t level = 0; level + 1 < level_counts.size(); ++level) {
        ReduceValueTreeLevelKernel<<<level_counts[level + 1], kThreads>>>(
            value_b.get(), level_offsets[level], level_offsets[level + 1],
            level_counts[level], level_counts[level + 1], options.tolerance,
            parallel_ok.get());
      }
      InitializeValueContextRootKernel<<<1, kThreads>>>(value_b.get(),
                                                        level_offsets.back());
      for (int level = static_cast<int>(level_counts.size()) - 2; level >= 0;
           --level) {
        ExpandValueContextLevelKernel<<<level_counts[level + 1], kThreads>>>(
            value_b.get(), level_offsets[level], level_offsets[level + 1],
            level_counts[level], level_counts[level + 1], options.tolerance,
            parallel_ok.get());
      }
      FinalizeValueSuffixKernel<<<node_count, kThreads>>>(
          value_a.get(), node_count, value_b.get(), options.tolerance,
          parallel_ok.get());
      value_suffix = value_a.get();
      CudaCheck(cudaMemcpyAsync(&host_parallel_ok, parallel_ok.get(),
                                sizeof(int), cudaMemcpyDeviceToHost),
                "read value scan status");
    });
  }

  if (host_parallel_ok != 0) {
    solution.used_parallel_riccati = true;
    solution.timings.riccati_ms += TimeGpu(workspace, [&] {
      if (stage_count > 0) {
        FeedbackKernel<<<stage_count, kThreads>>>(
            reduced_stages.get(), value_suffix, stage_count, options.tolerance,
            feedback.get(), device_status.get());
      }
      CudaCheck(cudaMemcpyAsync(workspace.host_status.data(),
                                device_status.get(), sizeof(DeviceStatus),
                                cudaMemcpyDeviceToHost),
                "read Riccati status");
    });
  } else {
    solution.used_parallel_riccati = false;
    solution.timings.riccati_ms += TimeGpu(workspace, [&] {
      host_parallel_ok = 1;
      CudaCheck(cudaMemcpyAsync(parallel_ok.get(), &host_parallel_ok,
                                sizeof(int), cudaMemcpyHostToDevice),
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
      CudaCheck(cudaMemcpyAsync(workspace.host_status.data(),
                                device_status.get(), sizeof(DeviceStatus),
                                cudaMemcpyDeviceToHost),
                "read Riccati status");
    });
  }
  status = workspace.host_status[0];
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }

  auto &map_a = workspace.map_leaves;
  auto &map_b = workspace.map_scan;
  auto &reduced_states = workspace.reduced_states;
  auto &states = workspace.states;
  auto &controls = workspace.controls;
  auto &state_dimensions = workspace.state_dimensions;
  auto &control_dimensions = workspace.control_dimensions;
  AffineMap *prefix = map_a.get();
  solution.timings.reconstruction_ms = TimeGpu(workspace, [&] {
    if (stage_count > 0) {
      InitializeAffineMapsKernel<<<stage_count, kThreads>>>(
          feedback.get(), stage_count, map_a.get());
      SeedAffineTreeKernel<<<stage_count, kThreads>>>(map_a.get(), stage_count,
                                                      map_b.get());
      for (std::size_t level = 0; level + 1 < stage_level_counts.size();
           ++level) {
        ReduceAffineTreeLevelKernel<<<stage_level_counts[level + 1],
                                      kThreads>>>(
            map_b.get(), stage_level_offsets[level],
            stage_level_offsets[level + 1], stage_level_counts[level],
            stage_level_counts[level + 1], device_status.get());
      }
      InitializeAffineContextRootKernel<<<1, kThreads>>>(
          map_b.get(), stage_level_offsets.back());
      for (int level = static_cast<int>(stage_level_counts.size()) - 2;
           level >= 0; --level) {
        ExpandAffineContextLevelKernel<<<stage_level_counts[level + 1],
                                         kThreads>>>(
            map_b.get(), stage_level_offsets[level],
            stage_level_offsets[level + 1], stage_level_counts[level],
            stage_level_counts[level + 1], device_status.get());
      }
      FinalizeAffinePrefixKernel<<<stage_count, kThreads>>>(
          map_a.get(), stage_count, map_b.get(), device_status.get());
      prefix = map_a.get();
    }
    const int state_blocks = (node_count + kThreads - 1) / kThreads;
    EvaluateReducedStatesKernel<<<state_blocks, kThreads>>>(
        prefix, stage_count, reduced_initial.get(), reduced_states.get());
    ReconstructPrimalKernel<<<node_count, kThreads>>>(
        state_params.get(), control_params.get(), feedback.get(),
        reduced_states.get(), stage_count, states.get(), controls.get());
    PackReducedDimensionsKernel<<<state_blocks, kThreads>>>(
        state_params.get(), control_params.get(), stage_count,
        state_dimensions.get(), control_dimensions.get());
    CudaCheck(cudaMemcpyAsync(workspace.host_status.data(), device_status.get(),
                              sizeof(DeviceStatus), cudaMemcpyDeviceToHost),
              "read reconstruction status");
  });
  status = workspace.host_status[0];
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }

  // Multiplier recovery uses a balanced relation tree. It chooses zero for
  // genuinely free multiplier components, which is sufficient even when the
  // original equality rows are redundant. Its right-hand side is formed from
  // the reconstructed primal solution, so use a separate numerical floor for
  // rank/consistency decisions after primal roundoff has accumulated.
  const Scalar multiplier_rank_tolerance =
      std::max(options.tolerance, kMinimumMultiplierRankTolerance);
  const Scalar multiplier_consistency_tolerance =
      options.enforce_multiplier_consistency
          ? std::max(multiplier_rank_tolerance,
                     kMultiplierConsistencyTolerancePerTreeLevel *
                         level_counts.size())
          : kScalarMax;
  auto &dual_tree = workspace.dual_tree;
  auto &dual_values = workspace.dual_values;
  auto &initial_multiplier = workspace.initial_multiplier;
  auto &dynamics_multipliers = workspace.dynamics_multipliers;
  auto &mixed_multipliers = workspace.mixed_multipliers;
  auto &state_multipliers = workspace.state_multipliers;
  auto &terminal_multiplier = workspace.terminal_multiplier;
  solution.timings.multiplier_ms = TimeGpu(workspace, [&] {
    BuildDualLeavesKernel<<<node_count, kThreads>>>(
        device_stages.get(), device_terminal.get(), stage_count, node_count,
        states.get(), controls.get(), multiplier_rank_tolerance,
        multiplier_consistency_tolerance, dual_tree.get(), device_status.get());
    for (std::size_t level = 0; level + 1 < level_counts.size(); ++level) {
      ReduceDualTreeLevelKernel<<<level_counts[level + 1], kThreads>>>(
          dual_tree.get(), level_offsets[level], level_offsets[level + 1],
          level_counts[level], level_counts[level + 1],
          multiplier_rank_tolerance, multiplier_consistency_tolerance,
          dual_tree.get(), device_status.get());
    }
    const int root_offset = level_offsets.back();
    SolveDualRootKernel<<<1, 1>>>(
        dual_tree.get() + root_offset, dual_values.get() + root_offset,
        device_status.get(), multiplier_rank_tolerance);
    for (int level = static_cast<int>(level_counts.size()) - 2; level >= 0;
         --level) {
      ExpandDualTreeLevelKernel<<<level_counts[level + 1], kThreads>>>(
          dual_tree.get(), level_offsets[level], level_offsets[level + 1],
          level_counts[level], level_counts[level + 1],
          multiplier_rank_tolerance, multiplier_consistency_tolerance,
          dual_values.get(), dual_values.get(), device_status.get());
    }
    RecoverLocalMultipliersKernel<<<node_count, kThreads>>>(
        device_stages.get(), device_terminal.get(), stage_count, states.get(),
        controls.get(), dual_values.get(), multiplier_rank_tolerance,
        multiplier_consistency_tolerance, initial_multiplier.get(),
        dynamics_multipliers.get(), mixed_multipliers.get(),
        state_multipliers.get(), terminal_multiplier.get(),
        device_status.get());
    CudaCheck(cudaMemcpyAsync(workspace.host_status.data(), device_status.get(),
                              sizeof(DeviceStatus), cudaMemcpyDeviceToHost),
              "read multiplier status");
  });
  status = workspace.host_status[0];
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }

  auto &host_states = workspace.host_states;
  auto &host_controls = workspace.host_controls;
  auto &host_initial_multiplier = workspace.host_initial_multiplier;
  auto &host_dynamics = workspace.host_dynamics;
  auto &host_mixed = workspace.host_mixed;
  auto &host_state_multipliers = workspace.host_state_multipliers;
  auto &host_terminal_multiplier = workspace.host_terminal_multiplier;
  auto &host_state_dimensions = workspace.host_state_dimensions;
  auto &host_control_dimensions = workspace.host_control_dimensions;
  solution.timings.download_ms = TimeGpu(workspace, [&] {
    CudaCheck(cudaMemcpyAsync(host_states.data(), states.get(),
                              host_states.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
              "download states");
    CudaCheck(cudaMemcpyAsync(host_controls.data(), controls.get(),
                              host_controls.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
              "download controls");
    CudaCheck(cudaMemcpyAsync(host_initial_multiplier.data(),
                              initial_multiplier.get(),
                              host_initial_multiplier.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
              "download initial multiplier");
    CudaCheck(cudaMemcpyAsync(host_dynamics.data(), dynamics_multipliers.get(),
                              host_dynamics.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
              "download dynamics multipliers");
    CudaCheck(cudaMemcpyAsync(host_mixed.data(), mixed_multipliers.get(),
                              host_mixed.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
              "download mixed multipliers");
    CudaCheck(cudaMemcpyAsync(host_state_multipliers.data(),
                              state_multipliers.get(),
                              host_state_multipliers.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
              "download state multipliers");
    CudaCheck(cudaMemcpyAsync(host_terminal_multiplier.data(),
                              terminal_multiplier.get(),
                              host_terminal_multiplier.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
              "download terminal multiplier");
    CudaCheck(cudaMemcpyAsync(host_state_dimensions.data(),
                              state_dimensions.get(),
                              host_state_dimensions.size() * sizeof(int),
                              cudaMemcpyDeviceToHost),
              "download state dimensions");
    if (stage_count > 0) {
      CudaCheck(cudaMemcpyAsync(host_control_dimensions.data(),
                                control_dimensions.get(),
                                host_control_dimensions.size() * sizeof(int),
                                cudaMemcpyDeviceToHost),
                "download control dimensions");
    }
  });

  solution.states.resize(node_count);
  solution.reduced_state_dimensions.resize(node_count);
  for (int i = 0; i < node_count; ++i) {
    const int n = host_state_dimensions[2 * i];
    solution.states[i].resize(n);
    for (int row = 0; row < n; ++row)
      solution.states[i][row] =
          host_states[static_cast<std::size_t>(i) * kMaxStateDimension + row];
    solution.reduced_state_dimensions[i] = host_state_dimensions[2 * i + 1];
  }
  solution.controls.resize(stage_count);
  solution.reduced_control_dimensions.resize(stage_count);
  solution.dynamics_multipliers.resize(stage_count);
  solution.mixed_multipliers.resize(stage_count);
  solution.state_multipliers.resize(stage_count);
  for (int i = 0; i < stage_count; ++i) {
    const PackedStage &s = host_stages[i];
    solution.controls[i].resize(s.m);
    for (int row = 0; row < s.m; ++row)
      solution.controls[i][row] =
          host_controls[static_cast<std::size_t>(i) * kMaxControlDimension +
                        row];
    solution.reduced_control_dimensions[i] = host_control_dimensions[2 * i + 1];
    solution.dynamics_multipliers[i].resize(s.next_n);
    for (int row = 0; row < s.next_n; ++row)
      solution.dynamics_multipliers[i][row] =
          host_dynamics[static_cast<std::size_t>(i) * kMaxStateDimension + row];
    solution.mixed_multipliers[i].resize(s.mixed);
    for (int row = 0; row < s.mixed; ++row)
      solution.mixed_multipliers[i][row] =
          host_mixed[static_cast<std::size_t>(i) * kMaxMixedConstraints + row];
    solution.state_multipliers[i].resize(s.state);
    for (int row = 0; row < s.state; ++row)
      solution.state_multipliers[i][row] =
          host_state_multipliers[static_cast<std::size_t>(i) *
                                     kMaxStateConstraints +
                                 row];
  }
  solution.initial_multiplier.resize(problem.initial_state.size());
  for (std::size_t row = 0; row < problem.initial_state.size(); ++row)
    solution.initial_multiplier[row] = host_initial_multiplier[row];
  solution.terminal_state_multiplier.resize(terminal.state);
  for (int row = 0; row < terminal.state; ++row)
    solution.terminal_state_multiplier[row] = host_terminal_multiplier[row];
  solution.objective =
      ObjectiveFromPacked(host_stages.data(), host_stages.size(), terminal,
                          host_states.data(), host_controls.data());
  solution.status = SolveStatus::kOptimal;
  solution.message = solution.used_parallel_riccati
                         ? "optimal (parallel CUDA Riccati scan)"
                         : "optimal (CUDA sequential Riccati fallback)";
  if (!options.enforce_multiplier_consistency)
    solution.message += "; multiplier consistency unchecked";
  solution.timings.total_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - total_start)
          .count();
  return solution;
}

} // namespace
} // namespace detail

struct Workspace::Impl {
  detail::WorkspaceStorage storage;
};

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() {
  if (impl_ && impl_->storage.device >= 0)
    cudaSetDevice(impl_->storage.device);
}
Workspace::Workspace(Workspace &&) noexcept = default;
Workspace &Workspace::operator=(Workspace &&other) noexcept {
  if (this != &other) {
    if (impl_ && impl_->storage.device >= 0)
      cudaSetDevice(impl_->storage.device);
    impl_ = std::move(other.impl_);
  }
  return *this;
}

void Workspace::Reserve(const Problem &problem, const Options &options) {
  if (!impl_)
    impl_ = std::make_unique<Impl>();
  detail::ValidateCudaProblem(problem, options);
  int device_count = 0;
  detail::CudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  detail::Require(options.device < device_count,
                  "CUDA device index is out of range");
  detail::CudaCheck(cudaSetDevice(options.device), "cudaSetDevice");
  const int stage_count = static_cast<int>(problem.stages.size());
  const int node_count = stage_count + 1;
  impl_->storage.Reserve(options.device, stage_count, node_count);
}

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

Solution &Solve(const Problem &problem, Workspace &workspace, Solution &result,
                const Options &options) {
  try {
    if (!Available()) {
      result.status = SolveStatus::kInvalidInput;
      result.message = "no CUDA device is available";
      return result;
    }
    if (!workspace.impl_)
      workspace.impl_ = std::make_unique<Workspace::Impl>();
    return detail::SolveImpl(problem, workspace.impl_->storage, result,
                             options);
  } catch (const std::invalid_argument &error) {
    result.status = SolveStatus::kInvalidInput;
    result.message = error.what();
    return result;
  } catch (const std::exception &error) {
    result.status = SolveStatus::kNumericalFailure;
    result.message = error.what();
    return result;
  }
}

Solution Solve(const Problem &problem, const Options &options) {
  Workspace workspace;
  Solution result;
  Solve(problem, workspace, result, options);
  return result;
}

} // namespace cuda
} // namespace clqr
#endif // CLQR_CUDA_EMULATION

namespace clqr {
namespace cuda {
namespace detail {
namespace {

__global__ void InitializeAffineMapsKernel(const Feedback *feedback, int count,
                                           AffineMap *maps) {
  const int index = blockIdx.x;
  if (index >= count)
    return;
  const Feedback &fb = feedback[index];
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

__device__ void CopyAffineMapBlock(const AffineMap &input, AffineMap *output) {
  if (threadIdx.x == 0) {
    output->left_dim = input.left_dim;
    output->right_dim = input.right_dim;
  }
  for (int linear = threadIdx.x; linear < input.right_dim * input.left_dim;
       linear += blockDim.x) {
    const int row = linear / input.left_dim;
    const int col = linear % input.left_dim;
    output->linear[row * kMaxStateDimension + col] =
        input.linear[row * kMaxStateDimension + col];
  }
  for (int row = threadIdx.x; row < input.right_dim; row += blockDim.x)
    output->offset[row] = input.offset[row];
}

__device__ bool InvalidScanAffineMap(const AffineMap &map) {
  return map.left_dim < 0;
}

__device__ void SetInvalidScanAffineMap(AffineMap *map) {
  if (threadIdx.x == 0) {
    map->left_dim = -1;
    map->right_dim = 0;
  }
}

__device__ void ComposeAffineMapsBlock(const AffineMap &first,
                                       const AffineMap &second,
                                       AffineMap *output, DeviceStatus *status,
                                       int index) {
  if (InvalidScanAffineMap(first)) {
    if (InvalidScanAffineMap(second)) {
      SetInvalidScanAffineMap(output);
    } else {
      CopyAffineMapBlock(second, output);
    }
    return;
  }
  if (InvalidScanAffineMap(second)) {
    CopyAffineMapBlock(first, output);
    return;
  }
  if (first.right_dim != second.left_dim) {
    SetFailure(status, kDeviceNumericalFailure, index, 11);
    return;
  }
  if (threadIdx.x == 0) {
    output->left_dim = first.left_dim;
    output->right_dim = second.right_dim;
  }
  for (int linear = threadIdx.x; linear < second.right_dim * first.left_dim;
       linear += blockDim.x) {
    const int row = linear / first.left_dim;
    const int col = linear % first.left_dim;
    Scalar value = Scalar{0};
    for (int k = 0; k < first.right_dim; ++k) {
      value += second.linear[row * kMaxStateDimension + k] *
               first.linear[k * kMaxStateDimension + col];
    }
    output->linear[row * kMaxStateDimension + col] = value;
  }
  for (int row = threadIdx.x; row < second.right_dim; row += blockDim.x) {
    Scalar value = second.offset[row];
    for (int k = 0; k < first.right_dim; ++k) {
      value += second.linear[row * kMaxStateDimension + k] * first.offset[k];
    }
    output->offset[row] = value;
  }
}

__global__ void SeedAffineTreeKernel(const AffineMap *leaves, int count,
                                     AffineMap *tree) {
  const int index = blockIdx.x;
  if (index >= count)
    return;
  CopyAffineMapBlock(leaves[index], &tree[index]);
}

__global__ void ReduceAffineTreeLevelKernel(AffineMap *tree, int child_offset,
                                            int parent_offset, int child_count,
                                            int parent_count,
                                            DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  const int left = child_offset + 2 * index;
  if (2 * index + 1 >= child_count) {
    CopyAffineMapBlock(tree[left], &tree[parent_offset + index]);
    return;
  }
  const int right = left + 1;
  ComposeAffineMapsBlock(tree[left], tree[right], &tree[parent_offset + index],
                         status, index);
}

__global__ void InitializeAffineContextRootKernel(AffineMap *tree,
                                                  int root_offset) {
  if (blockIdx.x == 0)
    SetInvalidScanAffineMap(&tree[root_offset]);
}

__global__ void
ExpandAffineContextLevelKernel(AffineMap *tree, int child_offset,
                               int parent_offset, int child_count,
                               int parent_count, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  const int left = child_offset + 2 * index;
  const AffineMap &parent_context = tree[parent_offset + index];
  if (2 * index + 1 >= child_count) {
    CopyAffineMapBlock(parent_context, &tree[left]);
    return;
  }
  const int right = left + 1;
  ComposeAffineMapsBlock(parent_context, tree[left], &tree[right], status,
                         index);
  __syncthreads();
  CopyAffineMapBlock(parent_context, &tree[left]);
}

__global__ void FinalizeAffinePrefixKernel(AffineMap *leaves, int count,
                                           const AffineMap *exclusive_scan,
                                           DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  const AffineMap &before = exclusive_scan[index];
  if (InvalidScanAffineMap(before))
    return;
  __shared__ AffineMap composed;
  ComposeAffineMapsBlock(before, leaves[index], &composed, status, index);
  __syncthreads();
  CopyAffineMapBlock(composed, &leaves[index]);
}

__global__ void EvaluateReducedStatesKernel(const AffineMap *prefix,
                                            int stage_count,
                                            const Scalar *initial,
                                            Scalar *reduced_states) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > stage_count)
    return;
  if (index == 0) {
    for (int col = 0; col < kMaxStateDimension; ++col)
      reduced_states[col] = initial[col];
    return;
  }
  const AffineMap &map = prefix[index - 1];
  for (int row = 0; row < map.right_dim; ++row) {
    Scalar value = map.offset[row];
    for (int col = 0; col < map.left_dim; ++col) {
      value += map.linear[row * kMaxStateDimension + col] * initial[col];
    }
    reduced_states[index * kMaxStateDimension + row] = value;
  }
}

__global__ void ReconstructPrimalKernel(const StateParam *state_params,
                                        const ControlParam *control_params,
                                        const Feedback *feedback,
                                        const Scalar *reduced_states,
                                        int stage_count, Scalar *states,
                                        Scalar *controls) {
  const int index = blockIdx.x;
  if (index > stage_count)
    return;
  const StateParam &state = state_params[index];
  const Scalar *z = reduced_states + index * kMaxStateDimension;
  for (int x = threadIdx.x; x < state.physical_dim; x += blockDim.x) {
    Scalar value = state.t[x];
    for (int col = 0; col < state.reduced_dim; ++col) {
      value += state.T[x * kMaxStateDimension + col] * z[col];
    }
    states[index * kMaxStateDimension + x] = value;
  }
  if (index == stage_count)
    return;
  const ControlParam &control = control_params[index];
  const Feedback &fb = feedback[index];
  __shared__ Scalar v[kMaxControlDimension];
  for (int row = threadIdx.x; row < control.reduced_dim; row += blockDim.x) {
    Scalar value = fb.k[row];
    for (int col = 0; col < fb.state_dim; ++col) {
      value += fb.K[row * kMaxStateDimension + col] * z[col];
    }
    v[row] = value;
  }
  __syncthreads();
  for (int u = threadIdx.x; u < control.physical_dim; u += blockDim.x) {
    Scalar value = control.y[u];
    for (int col = 0; col < control.state_dim; ++col) {
      value += control.Y[u * kMaxStateDimension + col] * z[col];
    }
    for (int col = 0; col < control.reduced_dim; ++col) {
      value += control.Z[u * kMaxControlDimension + col] * v[col];
    }
    controls[index * kMaxControlDimension + u] = value;
  }
}

__global__ void BuildDualLeavesKernel(
    const PackedStage *stages, const PackedTerminal *terminal_ptr,
    int stage_count, int padded_count, const Scalar *states,
    const Scalar *controls, Scalar rank_tolerance, Scalar consistency_tolerance,
    Relation *tree, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= padded_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  if (index > stage_count) {
    if (threadIdx.x == 0) {
      tree[index].left_dim = 0;
      tree[index].right_dim = 0;
      tree[index].rows = 0;
    }
    return;
  }
  __shared__ Scalar matrix[kMaxRrefEntries];
  __shared__ int rows;
  __shared__ int columns;
  __shared__ int eliminated;
  __shared__ int left_dim;
  __shared__ int right_dim;
  __shared__ int local_ok;
  __shared__ Scalar
      constraint_scales[kMaxMixedConstraints + kMaxStateConstraints];
  const PackedTerminal &terminal = *terminal_ptr;
  if (threadIdx.x == 0) {
    if (index == stage_count) {
      rows = terminal.n;
      eliminated = terminal.state;
      left_dim = terminal.n;
      right_dim = 0;
      columns = eliminated + left_dim + 1;
    } else {
      const PackedStage &s = stages[index];
      rows = s.n + s.m;
      eliminated = s.mixed + s.state;
      left_dim = s.n;
      right_dim = s.next_n;
      columns = eliminated + left_dim + right_dim + 1;
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = Scalar{0};
  if (index == stage_count) {
    for (int constraint = threadIdx.x; constraint < terminal.state;
         constraint += blockDim.x) {
      Scalar scale = Scalar{0};
      for (int row = 0; row < terminal.n; ++row) {
        scale =
            fmax(scale,
                 DeviceAbs(terminal.E[constraint * kMaxStateDimension + row]));
      }
      constraint_scales[constraint] = scale > Scalar{0} ? scale : Scalar{1};
    }
  } else {
    const PackedStage &s = stages[index];
    for (int constraint = threadIdx.x; constraint < s.mixed;
         constraint += blockDim.x) {
      Scalar scale = Scalar{0};
      for (int row = 0; row < s.n; ++row) {
        scale =
            fmax(scale, DeviceAbs(s.C[constraint * kMaxStateDimension + row]));
      }
      for (int row = 0; row < s.m; ++row) {
        scale = fmax(scale,
                     DeviceAbs(s.D[constraint * kMaxControlDimension + row]));
      }
      constraint_scales[constraint] = scale > Scalar{0} ? scale : Scalar{1};
    }
    for (int constraint = threadIdx.x; constraint < s.state;
         constraint += blockDim.x) {
      Scalar scale = Scalar{0};
      for (int row = 0; row < s.n; ++row) {
        scale =
            fmax(scale, DeviceAbs(s.E[constraint * kMaxStateDimension + row]));
      }
      constraint_scales[s.mixed + constraint] =
          scale > Scalar{0} ? scale : Scalar{1};
    }
  }
  __syncthreads();

  if (index == stage_count) {
    const Scalar *x = states + index * kMaxStateDimension;
    for (int linear = threadIdx.x; linear < terminal.n * terminal.state;
         linear += blockDim.x) {
      const int row = linear / terminal.state;
      const int constraint = linear % terminal.state;
      matrix[row * columns + constraint] =
          terminal.E[constraint * kMaxStateDimension + row] /
          constraint_scales[constraint];
    }
    for (int row = threadIdx.x; row < terminal.n; row += blockDim.x) {
      matrix[row * columns + eliminated + row] = Scalar{1};
      Scalar gradient = terminal.q[row];
      for (int col = 0; col < terminal.n; ++col) {
        gradient += terminal.Q[row * kMaxStateDimension + col] * x[col];
      }
      matrix[row * columns + columns - 1] = -gradient;
    }
  } else {
    const PackedStage &s = stages[index];
    const Scalar *x = states + index * kMaxStateDimension;
    const Scalar *u = controls + index * kMaxControlDimension;
    for (int linear = threadIdx.x; linear < s.n * s.mixed;
         linear += blockDim.x) {
      const int row = linear / s.mixed;
      const int constraint = linear % s.mixed;
      matrix[row * columns + constraint] =
          s.C[constraint * kMaxStateDimension + row] /
          constraint_scales[constraint];
    }
    for (int linear = threadIdx.x; linear < s.n * s.state;
         linear += blockDim.x) {
      const int row = linear / s.state;
      const int constraint = linear % s.state;
      matrix[row * columns + s.mixed + constraint] =
          s.E[constraint * kMaxStateDimension + row] /
          constraint_scales[s.mixed + constraint];
    }
    for (int linear = threadIdx.x; linear < s.m * s.mixed;
         linear += blockDim.x) {
      const int row = linear / s.mixed;
      const int constraint = linear % s.mixed;
      matrix[(s.n + row) * columns + constraint] =
          s.D[constraint * kMaxControlDimension + row] /
          constraint_scales[constraint];
    }
    for (int row = threadIdx.x; row < s.n; row += blockDim.x) {
      matrix[row * columns + eliminated + row] = Scalar{1};
      for (int next = 0; next < s.next_n; ++next) {
        matrix[row * columns + eliminated + s.n + next] =
            -s.A[next * kMaxStateDimension + row];
      }
      Scalar gradient = s.q[row];
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
      Scalar gradient = s.r[row];
      for (int col = 0; col < s.n; ++col)
        gradient += s.M[col * kMaxControlDimension + row] * x[col];
      for (int col = 0; col < s.m; ++col)
        gradient += s.R[row * kMaxControlDimension + col] * u[col];
      matrix[(s.n + row) * columns + columns - 1] = -gradient;
    }
  }
  __syncthreads();
  EliminateRelationOrthogonally(matrix, rows, columns, eliminated, left_dim,
                                right_dim, rank_tolerance,
                                consistency_tolerance, &tree[index], &local_ok);
  if (threadIdx.x == 0 && !local_ok)
    SetFailure(status, kDeviceNumericalFailure, index, 12);
}

__global__ void
ReduceDualTreeLevelKernel(const Relation *tree, int child_offset,
                          int parent_offset, int child_count, int parent_count,
                          Scalar rank_tolerance, Scalar consistency_tolerance,
                          Relation *mutable_tree, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  if (2 * index + 1 >= child_count) {
    CopyRelationBlock(tree[child_offset + 2 * index],
                      &mutable_tree[parent_offset + index]);
    return;
  }
  __shared__ Scalar matrix[kMaxRrefEntries];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;
  ComposeRelationsBlock(
      tree[child_offset + 2 * index], tree[child_offset + 2 * index + 1],
      rank_tolerance, consistency_tolerance,
      &mutable_tree[parent_offset + index], status, index,
      kDeviceNumericalFailure, 18, matrix, factors, pivot_columns, pivot_rows,
      &rank, &best_row, &local_ok, true);
}

__global__ void SolveDualRootKernel(const Relation *relation, NodeValue *value,
                                    DeviceStatus *status, Scalar tolerance) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || status->code != kDeviceOk)
    return;
  if (relation->right_dim != 0) {
    SetFailure(status, kDeviceNumericalFailure, 0, 13);
    return;
  }
  value->left_dim = relation->left_dim;
  value->right_dim = 0;
  // The contraction stores an orthonormal row basis.  Its transpose therefore
  // maps the affine right-hand side to the minimum-norm root endpoint.
  for (int col = 0; col < relation->left_dim; ++col) {
    Scalar endpoint = Scalar{0};
    for (int row = 0; row < relation->rows; ++row) {
      endpoint +=
          relation->left[row * kMaxStateDimension + col] * relation->rhs[row];
    }
    value->left[col] = endpoint;
  }
  for (int row = 0; row < relation->rows; ++row) {
    Scalar residual = -relation->rhs[row];
    for (int col = 0; col < relation->left_dim; ++col) {
      residual +=
          relation->left[row * kMaxStateDimension + col] * value->left[col];
    }
    if (DeviceAbs(residual) >
        tolerance * fmax(Scalar{1}, DeviceAbs(relation->rhs[row]))) {
      SetFailure(status, kDeviceNumericalFailure, 0, 14);
      return;
    }
  }
}

__global__ void ExpandDualTreeLevelKernel(
    const Relation *tree, int child_offset, int parent_offset, int child_count,
    int parent_count, Scalar rank_tolerance, Scalar consistency_tolerance,
    const NodeValue *parent_values, NodeValue *values, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  if (2 * index + 1 >= child_count) {
    const NodeValue &parent = parent_values[parent_offset + index];
    NodeValue &child = values[child_offset + 2 * index];
    if (threadIdx.x == 0) {
      child.left_dim = parent.left_dim;
      child.right_dim = parent.right_dim;
    }
    for (int entry = threadIdx.x; entry < parent.left_dim; entry += blockDim.x)
      child.left[entry] = parent.left[entry];
    for (int entry = threadIdx.x; entry < parent.right_dim; entry += blockDim.x)
      child.right[entry] = parent.right[entry];
    return;
  }
  const Relation &left = tree[child_offset + 2 * index];
  const Relation &right = tree[child_offset + 2 * index + 1];
  const NodeValue &parent = parent_values[parent_offset + index];
  if (left.left_dim != parent.left_dim || right.right_dim != parent.right_dim ||
      left.right_dim != right.left_dim) {
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, index, 15);
    return;
  }
  const int shared = left.right_dim;
  const int rows = left.rows + right.rows;
  const int columns = shared + 1;
  __shared__ Scalar matrix[kMaxRrefRows * (kMaxStateDimension + 1)];
  __shared__ Scalar residual_rhs[kMaxRrefRows];
  __shared__ Scalar upper[kMaxStateDimension * kMaxStateDimension];
  __shared__ Scalar rhs_projection[kMaxStateDimension];
  __shared__ Scalar shared_solution[kMaxStateDimension];
  __shared__ int permutation[kMaxStateDimension];
  __shared__ int rank;
  __shared__ int local_ok;
  __shared__ Scalar conditioned_rhs_scale;
  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = Scalar{0};
  __syncthreads();
  for (int linear = threadIdx.x; linear < left.rows * shared;
       linear += blockDim.x) {
    const int row = linear / shared;
    const int col = linear % shared;
    matrix[row * columns + col] = left.right[row * kMaxStateDimension + col];
  }
  for (int row = threadIdx.x; row < left.rows; row += blockDim.x) {
    Scalar rhs = left.rhs[row];
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
    Scalar rhs = right.rhs[row];
    for (int col = 0; col < right.right_dim; ++col) {
      rhs -= right.right[row * kMaxStateDimension + col] * parent.right[col];
    }
    matrix[(left.rows + row) * columns + shared] = rhs;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    conditioned_rhs_scale =
        ConditionedRhsScale(matrix, rows, columns, shared, rank_tolerance);
  }
  __syncthreads();
  SolveSystemOrthogonally(matrix, rows, columns, shared, rank_tolerance,
                          consistency_tolerance, conditioned_rhs_scale,
                          residual_rhs, upper, rhs_projection, shared_solution,
                          permutation, &rank, &local_ok);
  if (threadIdx.x == 0 && !local_ok)
    SetFailure(status, kDeviceNumericalFailure, index, 16);
  __syncthreads();
  if (!local_ok)
    return;
  if (threadIdx.x == 0) {
    NodeValue &left_value = values[child_offset + 2 * index];
    NodeValue &right_value = values[child_offset + 2 * index + 1];
    left_value.left_dim = left.left_dim;
    left_value.right_dim = shared;
    right_value.left_dim = shared;
    right_value.right_dim = right.right_dim;
    for (int col = 0; col < left.left_dim; ++col)
      left_value.left[col] = parent.left[col];
    for (int col = 0; col < right.right_dim; ++col)
      right_value.right[col] = parent.right[col];
    for (int col = 0; col < shared; ++col) {
      left_value.right[col] = shared_solution[col];
      right_value.left[col] = shared_solution[col];
    }
  }
}

__global__ void RecoverLocalMultipliersKernel(
    const PackedStage *stages, const PackedTerminal *terminal_ptr,
    int stage_count, const Scalar *states, const Scalar *controls,
    const NodeValue *leaf_values, Scalar rank_tolerance,
    Scalar consistency_tolerance, Scalar *initial_multiplier,
    Scalar *dynamics_multipliers, Scalar *mixed_multipliers,
    Scalar *state_multipliers, Scalar *terminal_multiplier,
    DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index > stage_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  __shared__ Scalar
      matrix[kMaxRrefRows * (kMaxMixedConstraints + kMaxStateConstraints + 1)];
  __shared__ Scalar
      original_matrix[(kMaxStateDimension + kMaxControlDimension) *
                      (kMaxMixedConstraints + kMaxStateConstraints + 1)];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ Scalar local_solution[kMaxMixedConstraints + kMaxStateConstraints];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int rows;
  __shared__ int variables;
  __shared__ int local_ok;
  __shared__ Scalar
      constraint_scales[kMaxMixedConstraints + kMaxStateConstraints];
  const PackedTerminal &terminal = *terminal_ptr;
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
    matrix[i] = Scalar{0};
  if (index == stage_count) {
    for (int constraint = threadIdx.x; constraint < terminal.state;
         constraint += blockDim.x) {
      Scalar scale = Scalar{0};
      for (int row = 0; row < terminal.n; ++row) {
        scale =
            fmax(scale,
                 DeviceAbs(terminal.E[constraint * kMaxStateDimension + row]));
      }
      constraint_scales[constraint] = scale > Scalar{0} ? scale : Scalar{1};
    }
  } else {
    const PackedStage &s = stages[index];
    for (int constraint = threadIdx.x; constraint < s.mixed;
         constraint += blockDim.x) {
      Scalar scale = Scalar{0};
      for (int row = 0; row < s.n; ++row) {
        scale =
            fmax(scale, DeviceAbs(s.C[constraint * kMaxStateDimension + row]));
      }
      for (int row = 0; row < s.m; ++row) {
        scale = fmax(scale,
                     DeviceAbs(s.D[constraint * kMaxControlDimension + row]));
      }
      constraint_scales[constraint] = scale > Scalar{0} ? scale : Scalar{1};
    }
    for (int constraint = threadIdx.x; constraint < s.state;
         constraint += blockDim.x) {
      Scalar scale = Scalar{0};
      for (int row = 0; row < s.n; ++row) {
        scale =
            fmax(scale, DeviceAbs(s.E[constraint * kMaxStateDimension + row]));
      }
      constraint_scales[s.mixed + constraint] =
          scale > Scalar{0} ? scale : Scalar{1};
    }
  }
  __syncthreads();
  const NodeValue &endpoints = leaf_values[index];

  if (index == stage_count) {
    const Scalar *x = states + index * kMaxStateDimension;
    for (int linear = threadIdx.x; linear < terminal.n * terminal.state;
         linear += blockDim.x) {
      const int row = linear / terminal.state;
      const int constraint = linear % terminal.state;
      matrix[row * columns + constraint] =
          terminal.E[constraint * kMaxStateDimension + row] /
          constraint_scales[constraint];
    }
    for (int row = threadIdx.x; row < terminal.n; row += blockDim.x) {
      Scalar rhs = -terminal.q[row] - endpoints.left[row];
      for (int col = 0; col < terminal.n; ++col)
        rhs -= terminal.Q[row * kMaxStateDimension + col] * x[col];
      matrix[row * columns + variables] = rhs;
    }
  } else {
    const PackedStage &s = stages[index];
    const Scalar *x = states + index * kMaxStateDimension;
    const Scalar *u = controls + index * kMaxControlDimension;
    for (int linear = threadIdx.x; linear < s.n * s.mixed;
         linear += blockDim.x) {
      const int row = linear / s.mixed;
      const int constraint = linear % s.mixed;
      matrix[row * columns + constraint] =
          s.C[constraint * kMaxStateDimension + row] /
          constraint_scales[constraint];
    }
    for (int linear = threadIdx.x; linear < s.n * s.state;
         linear += blockDim.x) {
      const int row = linear / s.state;
      const int constraint = linear % s.state;
      matrix[row * columns + s.mixed + constraint] =
          s.E[constraint * kMaxStateDimension + row] /
          constraint_scales[s.mixed + constraint];
    }
    for (int linear = threadIdx.x; linear < s.m * s.mixed;
         linear += blockDim.x) {
      const int row = linear / s.mixed;
      const int constraint = linear % s.mixed;
      matrix[(s.n + row) * columns + constraint] =
          s.D[constraint * kMaxControlDimension + row] /
          constraint_scales[constraint];
    }
    for (int row = threadIdx.x; row < s.n; row += blockDim.x) {
      Scalar rhs = -s.q[row] - endpoints.left[row];
      for (int col = 0; col < s.n; ++col)
        rhs -= s.Q[row * kMaxStateDimension + col] * x[col];
      for (int col = 0; col < s.m; ++col)
        rhs -= s.M[row * kMaxControlDimension + col] * u[col];
      for (int next = 0; next < s.next_n; ++next)
        rhs += s.A[next * kMaxStateDimension + row] * endpoints.right[next];
      matrix[row * columns + variables] = rhs;
    }
    for (int row = threadIdx.x; row < s.m; row += blockDim.x) {
      Scalar rhs = -s.r[row];
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
  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    original_matrix[i] = matrix[i];
  __syncthreads();
  RrefBlock(matrix, rows, columns, variables, rank_tolerance, pivot_columns,
            pivot_rows, &rank, &best_row, factors);
  if (threadIdx.x == 0) {
    for (int variable = 0; variable < variables; ++variable)
      local_solution[variable] = Scalar{0};
    for (int p = 0; p < rank; ++p) {
      if (pivot_columns[p] < variables) {
        local_solution[pivot_columns[p]] = matrix[p * columns + variables];
      }
    }
    local_ok = 1;
    for (int row = 0; row < rows; ++row) {
      Scalar value = Scalar{0};
      Scalar scale = DeviceAbs(original_matrix[row * columns + variables]);
      for (int variable = 0; variable < variables; ++variable) {
        const Scalar term = original_matrix[row * columns + variable] *
                            local_solution[variable];
        value += term;
        scale += DeviceAbs(term);
      }
      if (scale < Scalar{1})
        scale = Scalar{1};
      const Scalar residual =
          DeviceAbs(value - original_matrix[row * columns + variables]);
      if (residual > consistency_tolerance * scale) {
        local_ok = 0;
        break;
      }
    }
    if (!local_ok)
      SetFailure(status, kDeviceNumericalFailure, index, 17);
  }
  __syncthreads();
  if (!local_ok)
    return;
  if (threadIdx.x == 0) {
    if (index == stage_count) {
      for (int row = 0; row < terminal.state; ++row)
        terminal_multiplier[row] = local_solution[row] / constraint_scales[row];
      if (stage_count == 0) {
        for (int row = 0; row < terminal.n; ++row)
          initial_multiplier[row] = endpoints.left[row];
      }
    } else {
      const PackedStage &s = stages[index];
      for (int row = 0; row < s.mixed; ++row)
        mixed_multipliers[index * kMaxMixedConstraints + row] =
            local_solution[row] / constraint_scales[row];
      for (int row = 0; row < s.state; ++row)
        state_multipliers[index * kMaxStateConstraints + row] =
            local_solution[s.mixed + row] / constraint_scales[s.mixed + row];
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

} // namespace
} // namespace detail
} // namespace cuda
} // namespace clqr

namespace clqr {
namespace cuda {
namespace detail {
namespace {

__global__ void BuildValueElementsKernel(const ReducedStage *stages,
                                         const ReducedTerminal *terminal_ptr,
                                         int stage_count, Scalar tolerance,
                                         ValueElement *elements,
                                         int *parallel_ok,
                                         DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index > stage_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  if (index == stage_count) {
    const ReducedTerminal &terminal = *terminal_ptr;
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

  const ReducedStage &s = stages[index];
  ValueElement &out = elements[index];
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

  __shared__ Scalar
      augmented[kMaxControlDimension * (2 * kMaxControlDimension)];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ Scalar cholesky[kMaxControlDimension * kMaxControlDimension];
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
                                  : (col - s.m == row ? Scalar{1} : Scalar{0});
  }
  for (int linear = threadIdx.x; linear < s.m * s.m; linear += blockDim.x) {
    const int row = linear / s.m;
    const int col = linear % s.m;
    cholesky[linear] = Scalar{0.5} * (s.R[row * kMaxControlDimension + col] +
                                      s.R[col * kMaxControlDimension + row]);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    positive_definite = 1;
    Scalar scale = Scalar{1};
    for (int i = 0; i < s.m; ++i)
      scale = fmax(scale, DeviceAbs(cholesky[i * s.m + i]));
    for (int i = 0; i < s.m && positive_definite; ++i) {
      Scalar diagonal = cholesky[i * s.m + i];
      for (int k = 0; k < i; ++k) {
        diagonal -= cholesky[i * s.m + k] * cholesky[i * s.m + k];
      }
      if (!(diagonal > tolerance * scale) || !DeviceFinite(diagonal)) {
        positive_definite = 0;
        break;
      }
      cholesky[i * s.m + i] = sqrt(diagonal);
      for (int row = i + 1; row < s.m; ++row) {
        Scalar value = cholesky[row * s.m + i];
        for (int k = 0; k < i; ++k) {
          value -= cholesky[row * s.m + k] * cholesky[i * s.m + k];
        }
        cholesky[row * s.m + i] = value / cholesky[i * s.m + i];
      }
    }
    if (!positive_definite)
      atomicExch(parallel_ok, 0);
  }
  __syncthreads();
  if (!positive_definite)
    return;

  RrefBlock(augmented, s.m, columns, s.m, tolerance, pivot_columns, pivot_rows,
            &rank, &best_row, factors);
  if (rank != s.m) {
    if (threadIdx.x == 0)
      atomicExch(parallel_ok, 0);
    return;
  }

  // R inverse occupies the right half of augmented.
  for (int linear = threadIdx.x; linear < s.n * s.n; linear += blockDim.x) {
    const int a = linear / s.n;
    const int b = linear % s.n;
    Scalar value = s.Q[a * kMaxStateDimension + b];
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
    Scalar value = s.q[a];
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
    Scalar value = s.A[row * kMaxStateDimension + col];
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
    Scalar value = s.c[row];
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
    Scalar value = Scalar{0};
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
    const ValueElement &first, const ValueElement &second, Scalar tolerance,
    ValueElement *output, int *parallel_ok, Scalar *augmented, Scalar *factors,
    int *pivot_columns, int *pivot_rows, int *rank, int *best_row) {
  const int shared = first.right_dim;
  if (shared != second.left_dim) {
    if (threadIdx.x == 0)
      atomicExch(parallel_ok, 0);
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
    Scalar value = Scalar{0};
    if (col < shared) {
      value = row == col ? Scalar{1} : Scalar{0};
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
    if (threadIdx.x == 0)
      atomicExch(parallel_ok, 0);
    return;
  }

  // A = A2*S^{-1}*A1 and b = A2*S^{-1}(b1+C1*eta2)+b2.
  for (int linear = threadIdx.x; linear < right * left; linear += blockDim.x) {
    const int row = linear / left;
    const int col = linear % left;
    Scalar value = Scalar{0};
    for (int k = 0; k < shared; ++k) {
      value += second.A[row * kMaxStateDimension + k] *
               augmented[k * columns + shared + col];
    }
    output->A[row * kMaxStateDimension + col] = value;
  }
  for (int row = threadIdx.x; row < right; row += blockDim.x) {
    Scalar value = second.b[row];
    for (int k = 0; k < shared; ++k) {
      value += second.A[row * kMaxStateDimension + k] *
               augmented[k * columns + shared + left];
    }
    output->b[row] = value;
  }
  for (int linear = threadIdx.x; linear < right * right; linear += blockDim.x) {
    const int row = linear / right;
    const int col = linear % right;
    Scalar value = second.C[row * kMaxStateDimension + col];
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
    Scalar value = first.J[row * kMaxStateDimension + col];
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
    Scalar value = first.eta[row];
    for (int p = 0; p < shared; ++p) {
      Scalar dual = second.eta[p];
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
      const Scalar value =
          Scalar{0.5} * (output->J[row * kMaxStateDimension + col] +
                         output->J[col * kMaxStateDimension + row]);
      output->J[row * kMaxStateDimension + col] = value;
      output->J[col * kMaxStateDimension + row] = value;
    }
  }
  for (int linear = threadIdx.x; linear < right * right; linear += blockDim.x) {
    const int row = linear / right;
    const int col = linear % right;
    if (row < col) {
      const Scalar value =
          Scalar{0.5} * (output->C[row * kMaxStateDimension + col] +
                         output->C[col * kMaxStateDimension + row]);
      output->C[row * kMaxStateDimension + col] = value;
      output->C[col * kMaxStateDimension + row] = value;
    }
  }
}

__device__ void CopyValueElementBlock(const ValueElement &input,
                                      ValueElement *output) {
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

__device__ bool InvalidScanValueElement(const ValueElement &element) {
  return element.left_dim < 0;
}

__device__ void SetInvalidScanValueElement(ValueElement *element) {
  if (threadIdx.x == 0) {
    element->left_dim = -1;
    element->right_dim = 0;
  }
}

__device__ void
ComposeScanValueBlock(const ValueElement &first, const ValueElement &second,
                      Scalar tolerance, ValueElement *output, int *parallel_ok,
                      Scalar *augmented, Scalar *factors, int *pivot_columns,
                      int *pivot_rows, int *rank, int *best_row) {
  if (InvalidScanValueElement(first)) {
    if (InvalidScanValueElement(second)) {
      SetInvalidScanValueElement(output);
    } else {
      CopyValueElementBlock(second, output);
    }
    return;
  }
  if (InvalidScanValueElement(second)) {
    CopyValueElementBlock(first, output);
    return;
  }
  ComposeValueElementsBlock(first, second, tolerance, output, parallel_ok,
                            augmented, factors, pivot_columns, pivot_rows, rank,
                            best_row);
}

__global__ void SeedValueTreeKernel(const ValueElement *leaves, int count,
                                    ValueElement *tree) {
  const int index = blockIdx.x;
  if (index >= count)
    return;
  CopyValueElementBlock(leaves[index], &tree[index]);
}

__global__ void ReduceValueTreeLevelKernel(ValueElement *tree, int child_offset,
                                           int parent_offset, int child_count,
                                           int parent_count, Scalar tolerance,
                                           int *parallel_ok) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(parallel_ok, &block_enabled))
    return;
  const int left = child_offset + 2 * index;
  if (2 * index + 1 >= child_count) {
    CopyValueElementBlock(tree[left], &tree[parent_offset + index]);
    return;
  }
  const int right = left + 1;
  __shared__ Scalar
      augmented[kMaxStateDimension * (3 * kMaxStateDimension + 1)];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  ComposeValueElementsBlock(tree[left], tree[right], tolerance,
                            &tree[parent_offset + index], parallel_ok,
                            augmented, factors, pivot_columns, pivot_rows,
                            &rank, &best_row);
}

__global__ void InitializeValueContextRootKernel(ValueElement *tree,
                                                 int root_offset) {
  if (blockIdx.x == 0)
    SetInvalidScanValueElement(&tree[root_offset]);
}

__global__ void ExpandValueContextLevelKernel(
    ValueElement *tree, int child_offset, int parent_offset, int child_count,
    int parent_count, Scalar tolerance, int *parallel_ok) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(parallel_ok, &block_enabled))
    return;
  const int left = child_offset + 2 * index;
  const ValueElement &parent_context = tree[parent_offset + index];
  if (2 * index + 1 >= child_count) {
    CopyValueElementBlock(parent_context, &tree[left]);
    return;
  }
  const int right = left + 1;
  __shared__ Scalar
      augmented[kMaxStateDimension * (3 * kMaxStateDimension + 1)];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  ComposeScanValueBlock(tree[right], parent_context, tolerance, &tree[left],
                        parallel_ok, augmented, factors, pivot_columns,
                        pivot_rows, &rank, &best_row);
  __syncthreads();
  CopyValueElementBlock(parent_context, &tree[right]);
}

__global__ void
FinalizeValueSuffixKernel(ValueElement *leaves, int count,
                          const ValueElement *exclusive_contexts,
                          Scalar tolerance, int *parallel_ok) {
  const int index = blockIdx.x;
  if (index >= count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(parallel_ok, &block_enabled))
    return;
  const ValueElement &after = exclusive_contexts[index];
  if (InvalidScanValueElement(after))
    return;
  __shared__ Scalar
      augmented[kMaxStateDimension * (3 * kMaxStateDimension + 1)];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ ValueElement composed;
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  ComposeValueElementsBlock(leaves[index], after, tolerance, &composed,
                            parallel_ok, augmented, factors, pivot_columns,
                            pivot_rows, &rank, &best_row);
  __syncthreads();
  CopyValueElementBlock(composed, &leaves[index]);
}

__device__ void BuildFeedbackSystem(const ReducedStage &s,
                                    const ValueElement &next, Scalar *augmented,
                                    int columns) {
  for (int linear = threadIdx.x; linear < s.m * columns; linear += blockDim.x) {
    const int row = linear / columns;
    const int col = linear % columns;
    Scalar value = Scalar{0};
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
        Scalar future = next.eta[a];
        for (int b = 0; b < s.next_n; ++b) {
          future += next.J[a * kMaxStateDimension + b] * s.c[b];
        }
        value -= s.B[a * kMaxControlDimension + row] * future;
      }
    }
    augmented[linear] = value;
  }
}

__device__ void ExtractFeedback(const ReducedStage &s, const Scalar *augmented,
                                int columns, Feedback *feedback) {
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
    Scalar value = s.A[row * kMaxStateDimension + col];
    for (int u = 0; u < s.m; ++u) {
      value += s.B[row * kMaxControlDimension + u] *
               feedback->K[u * kMaxStateDimension + col];
    }
    feedback->transition[row * kMaxStateDimension + col] = value;
  }
  for (int row = threadIdx.x; row < s.next_n; row += blockDim.x) {
    Scalar value = s.c[row];
    for (int u = 0; u < s.m; ++u) {
      value += s.B[row * kMaxControlDimension + u] * feedback->k[u];
    }
    feedback->offset[row] = value;
  }
}

__global__ void FeedbackKernel(const ReducedStage *stages,
                               const ValueElement *suffix, int stage_count,
                               Scalar tolerance, Feedback *feedback,
                               DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= stage_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  const ReducedStage &s = stages[index];
  const ValueElement &next = suffix[index + 1];
  Feedback &out = feedback[index];
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
  __shared__ Scalar augmented[kMaxControlDimension *
                              (kMaxControlDimension + kMaxStateDimension + 1)];
  __shared__ Scalar factors[kMaxRrefRows];
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
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, index, 9);
    return;
  }
  ExtractFeedback(s, augmented, columns, &out);
}

__global__ void SequentialRiccatiKernel(const ReducedStage *stages,
                                        int stage_count, Scalar tolerance,
                                        ValueElement *suffix,
                                        Feedback *feedback,
                                        DeviceStatus *status) {
  if (blockIdx.x != 0)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  __shared__ Scalar augmented[kMaxControlDimension *
                              (kMaxControlDimension + kMaxStateDimension + 1)];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;

  for (int index = stage_count - 1; index >= 0; --index) {
    const ReducedStage &s = stages[index];
    const ValueElement &next = suffix[index + 1];
    Feedback &fb = feedback[index];
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

    ValueElement &current = suffix[index];
    if (threadIdx.x == 0) {
      current.left_dim = s.n;
      current.right_dim = 0;
    }
    for (int linear = threadIdx.x; linear < s.n * s.n; linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      Scalar value = s.Q[row * kMaxStateDimension + col];
      for (int a = 0; a < s.next_n; ++a) {
        for (int b = 0; b < s.next_n; ++b) {
          value += s.A[a * kMaxStateDimension + row] *
                   next.J[a * kMaxStateDimension + b] *
                   s.A[b * kMaxStateDimension + col];
        }
      }
      for (int u = 0; u < s.m; ++u) {
        Scalar cross = s.M[row * kMaxControlDimension + u];
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
      Scalar value = s.q[row];
      for (int a = 0; a < s.next_n; ++a) {
        Scalar future = next.eta[a];
        for (int b = 0; b < s.next_n; ++b) {
          future += next.J[a * kMaxStateDimension + b] * s.c[b];
        }
        value += s.A[a * kMaxStateDimension + row] * future;
      }
      for (int u = 0; u < s.m; ++u) {
        Scalar cross = s.M[row * kMaxControlDimension + u];
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
        const Scalar value =
            Scalar{0.5} * (current.J[row * kMaxStateDimension + col] +
                           current.J[col * kMaxStateDimension + row]);
        current.J[row * kMaxStateDimension + col] = value;
        current.J[col * kMaxStateDimension + row] = value;
      }
    }
    __syncthreads();
  }
}

} // namespace
} // namespace detail
} // namespace cuda
} // namespace clqr

namespace clqr {
namespace cuda {
namespace detail {
namespace {

__global__ void
ReduceStagesKernel(const PackedStage *stages, const Relation *suffix,
                   const StateParam *state_params, int stage_count,
                   Scalar rank_tolerance, Scalar consistency_tolerance,
                   ControlParam *control_params, ReducedStage *reduced,
                   DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= stage_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  const PackedStage &s = stages[index];
  const StateParam &current = state_params[index];
  const StateParam &next = state_params[index + 1];
  const Relation &next_relation = suffix[index + 1];
  __shared__ Scalar matrix[kMaxStageConstraintRows * kMaxStageReductionColumns];
  __shared__ Scalar factors[kMaxRrefRows];
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
    matrix[i] = Scalar{0};
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
    Scalar value = Scalar{0};
    for (int x = 0; x < s.n; ++x) {
      value += s.C[row * kMaxStateDimension + x] *
               current.T[x * kMaxStateDimension + z];
    }
    matrix[row * columns + s.m + z] = value;
  }
  for (int row = threadIdx.x; row < s.mixed; row += blockDim.x) {
    Scalar value = -s.d[row];
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
    Scalar value = Scalar{0};
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
    Scalar value = Scalar{0};
    for (int xp = 0; xp < s.next_n; ++xp) {
      Scalar at = Scalar{0};
      for (int x = 0; x < s.n; ++x) {
        at += s.A[xp * kMaxStateDimension + x] *
              current.T[x * kMaxStateDimension + z];
      }
      value += next_relation.left[row * kMaxStateDimension + xp] * at;
    }
    matrix[(s.mixed + row) * columns + s.m + z] = value;
  }
  for (int row = threadIdx.x; row < next_relation.rows; row += blockDim.x) {
    Scalar value = next_relation.rhs[row];
    for (int xp = 0; xp < s.next_n; ++xp) {
      Scalar affine = s.c[xp];
      for (int x = 0; x < s.n; ++x) {
        affine += s.A[xp * kMaxStateDimension + x] * current.t[x];
      }
      value -= next_relation.left[row * kMaxStateDimension + xp] * affine;
    }
    matrix[(s.mixed + row) * columns + columns - 1] = value;
  }
  __syncthreads();

  // Normalize raw mixed equations by their original scale, while treating a
  // successor equation produced entirely at the roundoff level as zero.  A
  // second relative row test removes constraints that vanish after applying
  // the current feasible-state parameterization.  This must happen before the
  // generic RREF equilibration, which would otherwise magnify cancellation
  // noise into a unit pivot.
  for (int row = threadIdx.x; row < rows; row += blockDim.x) {
    Scalar scale = Scalar{0};
    if (row < s.mixed) {
      for (int x = 0; x < s.n; ++x)
        scale = fmax(scale, DeviceAbs(s.C[row * kMaxStateDimension + x]));
      for (int u = 0; u < s.m; ++u)
        scale = fmax(scale, DeviceAbs(s.D[row * kMaxControlDimension + u]));
      scale = fmax(scale, DeviceAbs(s.d[row]));
    } else {
      for (int col = 0; col < columns; ++col)
        scale = fmax(scale, DeviceAbs(matrix[row * columns + col]));
    }
    factors[row] = scale;
  }
  __syncthreads();
  for (int linear = threadIdx.x; linear < rows * columns;
       linear += blockDim.x) {
    const int row = linear / columns;
    const Scalar scale = factors[row];
    if (row >= s.mixed && scale <= rank_tolerance) {
      matrix[linear] = Scalar{0};
    } else if (scale > Scalar{0}) {
      matrix[linear] /= scale;
    }
  }
  __syncthreads();
  for (int row = threadIdx.x; row < rows; row += blockDim.x) {
    Scalar scale = Scalar{0};
    for (int col = 0; col < columns; ++col)
      scale = fmax(scale, DeviceAbs(matrix[row * columns + col]));
    factors[row] = scale;
  }
  __syncthreads();
  for (int linear = threadIdx.x; linear < rows * columns;
       linear += blockDim.x) {
    if (factors[linear / columns] <= rank_tolerance)
      matrix[linear] = Scalar{0};
  }
  __syncthreads();

  RrefBlock(matrix, rows, columns, columns - 1, rank_tolerance, pivot_columns,
            pivot_rows, &rank, &best_row, factors);
  if (threadIdx.x == 0) {
    local_ok = 1;
    if (InconsistentRref(matrix, rows, columns, columns - 1,
                         consistency_tolerance)) {
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
      ControlParam &cp = control_params[index];
      cp.physical_dim = s.m;
      cp.state_dim = current.reduced_dim;
      cp.reduced_dim = s.m - control_rank;
      bool pivot[kMaxControlDimension]{};
      for (int p = 0; p < control_rank; ++p)
        pivot[pivot_columns[p]] = true;
      int free = 0;
      for (int u = 0; u < s.m; ++u) {
        if (!pivot[u])
          cp.free_columns[free++] = u;
      }
      ReducedStage &rs = reduced[index];
      rs.n = current.reduced_dim;
      rs.next_n = next.reduced_dim;
      rs.m = cp.reduced_dim;
    }
  }
  __syncthreads();
  if (!local_ok)
    return;

  ControlParam &initialized_control = control_params[index];
  for (int u = threadIdx.x; u < initialized_control.physical_dim;
       u += blockDim.x)
    initialized_control.y[u] = Scalar{0};
  for (int linear = threadIdx.x; linear < initialized_control.physical_dim *
                                              initialized_control.state_dim;
       linear += blockDim.x)
    initialized_control.Y[linear] = Scalar{0};
  for (int linear = threadIdx.x; linear < initialized_control.physical_dim *
                                              initialized_control.reduced_dim;
       linear += blockDim.x) {
    const int u = linear / initialized_control.reduced_dim;
    const int v = linear % initialized_control.reduced_dim;
    initialized_control.Z[u * kMaxControlDimension + v] = Scalar{0};
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int p = 0; p < control_rank; ++p) {
      const int u = pivot_columns[p];
      initialized_control.y[u] = matrix[p * columns + columns - 1];
      for (int z = 0; z < current.reduced_dim; ++z) {
        initialized_control.Y[u * kMaxStateDimension + z] =
            -matrix[p * columns + s.m + z];
      }
      for (int v = 0; v < initialized_control.reduced_dim; ++v) {
        initialized_control.Z[u * kMaxControlDimension + v] =
            -matrix[p * columns + initialized_control.free_columns[v]];
      }
    }
    for (int v = 0; v < initialized_control.reduced_dim; ++v) {
      initialized_control
          .Z[initialized_control.free_columns[v] * kMaxControlDimension + v] =
          Scalar{1};
    }
  }
  __syncthreads();

  const ControlParam &cp = control_params[index];
  ReducedStage &rs = reduced[index];

  // Reduced dynamics, selecting the free physical coordinates at node i+1.
  for (int linear = threadIdx.x; linear < rs.next_n * rs.n;
       linear += blockDim.x) {
    const int row = linear / rs.n;
    const int z = linear % rs.n;
    const int xp = next.free_columns[row];
    Scalar value = Scalar{0};
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
    Scalar value = Scalar{0};
    for (int u = 0; u < s.m; ++u) {
      value += s.B[xp * kMaxControlDimension + u] *
               cp.Z[u * kMaxControlDimension + v];
    }
    rs.B[row * kMaxControlDimension + v] = value;
  }
  for (int row = threadIdx.x; row < rs.next_n; row += blockDim.x) {
    const int xp = next.free_columns[row];
    Scalar value = s.c[xp] - next.t[xp];
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
    Scalar value = Scalar{0};
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
    Scalar value = Scalar{0};
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
    Scalar value = Scalar{0};
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
    Scalar value = Scalar{0};
    for (int x = 0; x < s.n; ++x) {
      Scalar gx = s.q[x];
      for (int y = 0; y < s.n; ++y)
        gx += s.Q[x * kMaxStateDimension + y] * current.t[y];
      for (int u = 0; u < s.m; ++u)
        gx += s.M[x * kMaxControlDimension + u] * cp.y[u];
      value += current.T[x * kMaxStateDimension + z] * gx;
    }
    for (int u = 0; u < s.m; ++u) {
      Scalar gu = s.r[u];
      for (int x = 0; x < s.n; ++x)
        gu += s.M[x * kMaxControlDimension + u] * current.t[x];
      for (int v = 0; v < s.m; ++v)
        gu += s.R[u * kMaxControlDimension + v] * cp.y[v];
      value += cp.Y[u * kMaxStateDimension + z] * gu;
    }
    rs.q[z] = value;
  }
  for (int v = threadIdx.x; v < rs.m; v += blockDim.x) {
    Scalar value = Scalar{0};
    for (int u = 0; u < s.m; ++u) {
      Scalar gu = s.r[u];
      for (int x = 0; x < s.n; ++x)
        gu += s.M[x * kMaxControlDimension + u] * current.t[x];
      for (int w = 0; w < s.m; ++w)
        gu += s.R[u * kMaxControlDimension + w] * cp.y[w];
      value += cp.Z[u * kMaxControlDimension + v] * gu;
    }
    rs.r[v] = value;
  }
}

__global__ void ReduceTerminalKernel(const PackedTerminal *terminal_ptr,
                                     const StateParam *state_params,
                                     int terminal_index,
                                     ReducedTerminal *reduced) {
  const PackedTerminal &terminal = *terminal_ptr;
  const StateParam &param = state_params[terminal_index];
  if (threadIdx.x == 0)
    reduced->n = param.reduced_dim;
  for (int linear = threadIdx.x; linear < param.reduced_dim * param.reduced_dim;
       linear += blockDim.x) {
    const int a = linear / param.reduced_dim;
    const int b = linear % param.reduced_dim;
    Scalar value = Scalar{0};
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
    Scalar value = Scalar{0};
    for (int x = 0; x < terminal.n; ++x) {
      Scalar gx = terminal.q[x];
      for (int y = 0; y < terminal.n; ++y) {
        gx += terminal.Q[x * kMaxStateDimension + y] * param.t[y];
      }
      value += param.T[x * kMaxStateDimension + a] * gx;
    }
    reduced->q[a] = value;
  }
}

__global__ void InitialReducedStateKernel(const StateParam *state_params,
                                          const Scalar *initial_state,
                                          Scalar *reduced_initial,
                                          Scalar tolerance,
                                          DeviceStatus *status) {
  const StateParam &param = state_params[0];
  for (int z = threadIdx.x; z < param.reduced_dim; z += blockDim.x) {
    const int physical = param.free_columns[z];
    reduced_initial[z] = initial_state[physical] - param.t[physical];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    Scalar scale = Scalar{1};
    Scalar residual = Scalar{0};
    for (int x = 0; x < param.physical_dim; ++x) {
      Scalar value = param.t[x];
      for (int z = 0; z < param.reduced_dim; ++z) {
        value += param.T[x * kMaxStateDimension + z] * reduced_initial[z];
      }
      scale = fmax(scale, DeviceAbs(initial_state[x]));
      residual = fmax(residual, DeviceAbs(value - initial_state[x]));
    }
    if (residual > Scalar{20} * tolerance * scale) {
      SetFailure(status, kDeviceInfeasible, 0, 8);
    }
  }
}

} // namespace
} // namespace detail
} // namespace cuda
} // namespace clqr
