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

// Each stage/node carries a small dense problem.  A single warp avoids the
// four-warp synchronization and occupancy cost of the former 128-thread
// blocks while retaining enough lanes for all supported active dimensions.
constexpr int kThreads = 32;
#ifdef CLQR_USE_FLOAT
constexpr Scalar kMinimumFeasibilityConsistencyTolerance = 1e-4f;
constexpr Scalar kMinimumMultiplierRankTolerance = 1e-4f;
constexpr Scalar kMultiplierConsistencyTolerancePerTreeLevel = 2e-2f;
#else
constexpr Scalar kMinimumFeasibilityConsistencyTolerance = Scalar{0};
constexpr Scalar kMinimumMultiplierRankTolerance = 1e-7;
constexpr Scalar kMultiplierConsistencyTolerancePerTreeLevel = 1e-6;
#endif
constexpr int kMaxRrefRows = ConstexprMax(
    4 * kMaxDualParameterDimension,
    ConstexprMax(2 * kMaxRelationRows,
                 ConstexprMax(kMaxMixedConstraints + kMaxStateConstraints +
                                  kMaxStateDimension,
                              kMaxStateDimension + kMaxControlDimension)));
constexpr int kMaxRrefColumns = kMaxDualColumns;
constexpr int kMaxRrefEntries = kMaxRrefRows * kMaxRrefColumns;
constexpr int kMaxValueElementEntries =
    3 * kMaxStateDimension * kMaxStateDimension + 2 * kMaxStateDimension;
constexpr int kMaxAffineMapEntries =
    kMaxStateDimension * kMaxStateDimension + kMaxStateDimension;
constexpr int kMaxRelationScratchEntries =
    2 * kMaxRelationRows * kMaxStateDimension + kMaxRelationRows;
constexpr int kMaxDualRelationScratchEntries =
    4 * kMaxDualParameterDimension * kMaxDualParameterDimension +
    2 * kMaxDualParameterDimension;
constexpr int kMaxDualValueScratchEntries = 2 * kMaxDualParameterDimension;
constexpr std::size_t kConservativeSharedMemoryLimit = 48 * 1024;
constexpr std::size_t kMaximumRrefSharedBytes =
    static_cast<std::size_t>(kMaxRrefEntries + 2 * kMaxRrefRows) *
        sizeof(Scalar) +
    static_cast<std::size_t>(2 * kMaxRrefRows + 16) * sizeof(int) +
    static_cast<std::size_t>(kMaxRelationScratchEntries) * sizeof(Scalar) +
    static_cast<std::size_t>(kMaxDualParameterDimension) * sizeof(Scalar);
static_assert(
    kMaximumRrefSharedBytes <= kConservativeSharedMemoryLimit,
    "CUDA dimension capacities exceed the portable 48 KiB per-block shared-"
    "memory budget; configure smaller CLQR_CUDA_MAX_* values");

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

__device__ void BindValueElementScratch(ValueElement *element,
                                        Scalar *storage) {
  element->A = storage;
  element->b = element->A + kMaxStateDimension * kMaxStateDimension;
  element->C = element->b + kMaxStateDimension;
  element->eta = element->C + kMaxStateDimension * kMaxStateDimension;
  element->J = element->eta + kMaxStateDimension;
}

__device__ void BindAffineMapScratch(AffineMap *map, Scalar *storage) {
  map->linear = storage;
  map->offset = map->linear + kMaxStateDimension * kMaxStateDimension;
}

__device__ void BindRelationScratch(Relation *relation, Scalar *storage) {
  relation->left = storage;
  relation->right = relation->left + kMaxRelationRows * kMaxStateDimension;
  relation->rhs = relation->right + kMaxRelationRows * kMaxStateDimension;
}

__device__ void BindDualRelationScratch(DualRelation *relation,
                                        Scalar *storage) {
  relation->left = storage;
  relation->right = relation->left +
                    2 * kMaxDualParameterDimension * kMaxDualParameterDimension;
  relation->rhs = relation->right +
                  2 * kMaxDualParameterDimension * kMaxDualParameterDimension;
}

__device__ void BindDualValueScratch(DualNodeValue *value, Scalar *storage) {
  value->left = storage;
  value->right = value->left + kMaxDualParameterDimension;
}

__device__ void SetFailure(DeviceStatus *status, int code, int stage,
                           int detail) {
  if (atomicCAS(&status->code, kDeviceOk, code) == kDeviceOk) {
    status->stage = stage;
    status->detail = detail;
  }
}

// A global failure can be reported by another block at any time.  Sampling it
// independently in every lane before a later warp synchronization can
// therefore make only part of a warp return.  Have one lane sample the flag
// and broadcast a warp-uniform decision instead.
__device__ bool BlockEnabled(const DeviceStatus *status, int *enabled) {
  if (threadIdx.x == 0)
    *enabled = status->code == kDeviceOk;
  WarpSynchronize();
  return *enabled != 0;
}

__device__ bool BlockEnabled(const int *flag, int *enabled) {
  if (threadIdx.x == 0)
    *enabled = *flag != 0;
  WarpSynchronize();
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
  WarpSynchronize();

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
    WarpSynchronize();
    const int selected_row = *best_row;
    // A no-pivot iteration skips the barriers below.  Ensure every thread has
    // consumed best_row before thread 0 reuses it in the next iteration.
    WarpSynchronize();
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
    WarpSynchronize();

    const Scalar pivot = matrix[pivot_row * columns + col];
    // All threads must load the pivot before any thread normalizes its entry.
    WarpSynchronize();
    for (int j = col + threadIdx.x; j < columns; j += blockDim.x) {
      matrix[pivot_row * columns + j] /= pivot;
    }
    WarpSynchronize();

    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
      factors[row] = row == pivot_row ? Scalar{0} : matrix[row * columns + col];
    }
    WarpSynchronize();
    for (int index = threadIdx.x; index < rows * (columns - col);
         index += blockDim.x) {
      const int row = index / (columns - col);
      const int j = col + index % (columns - col);
      if (row != pivot_row) {
        matrix[row * columns + j] -=
            factors[row] * matrix[pivot_row * columns + j];
      }
    }
    WarpSynchronize();
    if (threadIdx.x == 0) {
      pivot_columns[pivot_row] = col;
      pivot_rows[pivot_row] = pivot_row;
      ++(*rank);
    }
    WarpSynchronize();
    if (*rank == rows)
      break;
  }

  for (int index = threadIdx.x; index < rows * columns; index += blockDim.x) {
    if (DeviceAbs(matrix[index]) <= tolerance)
      matrix[index] = Scalar{0};
  }
  WarpSynchronize();
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

template <typename RelationType>
__device__ void ExtractResidualRelation(const Scalar *matrix, int columns,
                                        int rank, const int *pivot_columns,
                                        int eliminated_columns, int left_dim,
                                        int right_dim, RelationType *output) {
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
  WarpSynchronize();
  const int eliminated_rank = rank - output->rows;
  const int outer_dim = left_dim + right_dim;
  for (int index = threadIdx.x; index < output->rows * outer_dim;
       index += blockDim.x) {
    const int row = index / outer_dim;
    const int col = index % outer_dim;
    const Scalar value =
        matrix[(eliminated_rank + row) * columns + eliminated_columns + col];
    if (col < left_dim) {
      output->left[row * left_dim + col] = value;
    } else {
      output->right[row * right_dim + col - left_dim] = value;
    }
  }
  for (int row = threadIdx.x; row < output->rows; row += blockDim.x) {
    output->rhs[row] = matrix[(eliminated_rank + row) * columns + columns - 1];
  }
  WarpSynchronize();
}

// Eliminate the leading variables by orthogonally projecting the equations
// onto their left nullspace, then retain an orthonormal basis of the resulting
// affine relation.  Pivoted, reorthogonalized modified Gram--Schmidt is used in
// both directions.  Stage dimensions are bounded, so the serial work within a
// block is constant with respect to the horizon while independent tree nodes
// remain fully parallel.
template <typename RelationType>
__device__ void EliminateRelationOrthogonally(
    Scalar *matrix, int rows, int columns, int eliminated_columns, int left_dim,
    int right_dim, Scalar rank_tolerance, Scalar consistency_tolerance,
    RelationType *relation, int *local_ok) {
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
          relation->left[row * left_dim + col] = value;
        } else {
          relation->right[row * right_dim + col - left_dim] = value;
        }
      }
      for (int row = lane; row < relation_rank; row += kActiveWarpWidth)
        relation->rhs[row] = matrix[row * columns + columns - 1];
    }
  }
  WarpSynchronize();
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
      Scalar pivoted_solution[kMaxDualParameterDimension]{};
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
  WarpSynchronize();
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
  WarpSynchronize();

  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = Scalar{0};
  WarpSynchronize();
  if (index == stage_count) {
    for (int linear = threadIdx.x; linear < terminal.state * terminal.n;
         linear += blockDim.x) {
      const int row = linear / terminal.n;
      const int col = linear % terminal.n;
      matrix[row * columns + col] = terminal.E[row * terminal.n + col];
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
      matrix[row * columns + col] = s.D[row * s.m + col];
    }
    for (int linear = threadIdx.x; linear < s.mixed * s.n;
         linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      matrix[row * columns + s.m + col] = s.C[row * s.n + col];
    }
    for (int row = threadIdx.x; row < s.mixed; row += blockDim.x) {
      matrix[row * columns + columns - 1] = -s.d[row];
    }
    for (int linear = threadIdx.x; linear < s.state * s.n;
         linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      matrix[(s.mixed + row) * columns + s.m + col] = s.E[row * s.n + col];
    }
    for (int row = threadIdx.x; row < s.state; row += blockDim.x) {
      matrix[(s.mixed + row) * columns + columns - 1] = -s.e[row];
    }
    const int dynamics_row = s.mixed + s.state;
    for (int linear = threadIdx.x; linear < s.next_n * s.m;
         linear += blockDim.x) {
      const int row = linear / s.m;
      const int col = linear % s.m;
      matrix[(dynamics_row + row) * columns + col] = -s.B[row * s.m + col];
    }
    for (int linear = threadIdx.x; linear < s.next_n * s.n;
         linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      matrix[(dynamics_row + row) * columns + s.m + col] =
          -s.A[row * s.n + col];
    }
    for (int row = threadIdx.x; row < s.next_n; row += blockDim.x) {
      matrix[(dynamics_row + row) * columns + s.m + s.n + row] = Scalar{1};
      matrix[(dynamics_row + row) * columns + columns - 1] = s.c[row];
    }
  }
  WarpSynchronize();
  RrefBlock(matrix, rows, columns, columns - 1, rank_tolerance, pivot_columns,
            pivot_rows, &rank, &best_row, factors);
  if (threadIdx.x == 0) {
    local_ok = 1;
    if (!local_ok)
      SetFailure(status, kDeviceInfeasible, index, 1);
  }
  WarpSynchronize();
  if (!local_ok)
    return;
  ExtractResidualRelation(matrix, columns, rank, pivot_columns, eliminated,
                          left_dim, right_dim, &leaves[index]);
}

template <typename RelationType>
__device__ void
ComposeRelationsBlock(const RelationType &first, const RelationType &second,
                      Scalar rank_tolerance, Scalar consistency_tolerance,
                      RelationType *output, DeviceStatus *status, int stage,
                      int inconsistency_code, int inconsistency_detail,
                      Scalar *matrix, Scalar *factors, int *pivot_columns,
                      int *pivot_rows, int *rank, int *best_row, int *local_ok,
                      bool orthonormalize_output = false) {
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
  WarpSynchronize();
  for (int linear = threadIdx.x; linear < first.rows * shared;
       linear += blockDim.x) {
    const int row = linear / shared;
    const int col = linear % shared;
    matrix[row * columns + col] = first.right[row * first.right_dim + col];
  }
  for (int linear = threadIdx.x; linear < first.rows * first.left_dim;
       linear += blockDim.x) {
    const int row = linear / first.left_dim;
    const int col = linear % first.left_dim;
    matrix[row * columns + shared + col] =
        first.left[row * first.left_dim + col];
  }
  for (int row = threadIdx.x; row < first.rows; row += blockDim.x) {
    matrix[row * columns + columns - 1] = first.rhs[row];
  }
  for (int linear = threadIdx.x; linear < second.rows * shared;
       linear += blockDim.x) {
    const int row = linear / shared;
    const int col = linear % shared;
    matrix[(first.rows + row) * columns + col] =
        second.left[row * second.left_dim + col];
  }
  for (int linear = threadIdx.x; linear < second.rows * second.right_dim;
       linear += blockDim.x) {
    const int row = linear / second.right_dim;
    const int col = linear % second.right_dim;
    matrix[(first.rows + row) * columns + shared + first.left_dim + col] =
        second.right[row * second.right_dim + col];
  }
  for (int row = threadIdx.x; row < second.rows; row += blockDim.x) {
    matrix[(first.rows + row) * columns + columns - 1] = second.rhs[row];
  }
  WarpSynchronize();
  if (orthonormalize_output) {
    EliminateRelationOrthogonally(matrix, rows, columns, shared, first.left_dim,
                                  second.right_dim, rank_tolerance,
                                  consistency_tolerance, output, local_ok);
    if (threadIdx.x == 0 && !*local_ok)
      SetFailure(status, inconsistency_code, stage, inconsistency_detail);
    WarpSynchronize();
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
  WarpSynchronize();
  if (!*local_ok)
    return;
  ExtractResidualRelation(matrix, columns, *rank, pivot_columns, shared,
                          first.left_dim, second.right_dim, output);
}

template <typename RelationType>
__device__ void CopyRelationBlock(const RelationType &input,
                                  RelationType *output) {
  if (threadIdx.x == 0) {
    output->left_dim = input.left_dim;
    output->right_dim = input.right_dim;
    output->rows = input.rows;
  }
  for (int linear = threadIdx.x; linear < input.rows * input.left_dim;
       linear += blockDim.x) {
    const int row = linear / input.left_dim;
    const int col = linear % input.left_dim;
    output->left[row * input.left_dim + col] =
        input.left[row * input.left_dim + col];
  }
  for (int linear = threadIdx.x; linear < input.rows * input.right_dim;
       linear += blockDim.x) {
    const int row = linear / input.right_dim;
    const int col = linear % input.right_dim;
    output->right[row * input.right_dim + col] =
        input.right[row * input.right_dim + col];
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

__global__ void
ReduceRelationLeavesKernel(const Relation *leaves, int count, int parent_count,
                           Scalar rank_tolerance, Scalar consistency_tolerance,
                           Relation *parents, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  const int left = 2 * index;
  if (left + 1 >= count) {
    CopyRelationBlock(leaves[left], &parents[index]);
    return;
  }
  __shared__ Scalar matrix[kMaxRrefRows * kMaxRelationColumns];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;
  ComposeRelationsBlock(leaves[left], leaves[left + 1], rank_tolerance,
                        consistency_tolerance, &parents[index], status, index,
                        kDeviceInfeasible, 19, matrix, factors, pivot_columns,
                        pivot_rows, &rank, &best_row, &local_ok);
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
  WarpSynchronize();
  CopyRelationBlock(parent_context, &tree[right]);
}

__global__ void FinalizeRelationSuffixFromParentsKernel(
    Relation *leaves, int count, const Relation *parent_contexts,
    int parent_count, Scalar rank_tolerance, Scalar consistency_tolerance,
    DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  const int left = 2 * index;
  const Relation &parent = parent_contexts[index];
  __shared__ Relation composed;
  __shared__ Scalar composed_storage[kMaxRelationScratchEntries];
  __shared__ Scalar matrix[kMaxRrefRows * kMaxRelationColumns];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;
  if (threadIdx.x == 0)
    BindRelationScratch(&composed, composed_storage);
  WarpSynchronize();
  if (left + 1 >= count) {
    if (!InvalidScanRelation(parent)) {
      ComposeScanRelationBlock(leaves[left], parent, rank_tolerance,
                               consistency_tolerance, &composed, status, left,
                               21, matrix, factors, pivot_columns, pivot_rows,
                               &rank, &best_row, &local_ok);
      WarpSynchronize();
      CopyRelationBlock(composed, &leaves[left]);
    }
    return;
  }
  const int right = left + 1;
  if (!InvalidScanRelation(parent)) {
    ComposeScanRelationBlock(leaves[right], parent, rank_tolerance,
                             consistency_tolerance, &composed, status, right,
                             21, matrix, factors, pivot_columns, pivot_rows,
                             &rank, &best_row, &local_ok);
    WarpSynchronize();
    CopyRelationBlock(composed, &leaves[right]);
    WarpSynchronize();
  }
  ComposeRelationsBlock(leaves[left], leaves[right], rank_tolerance,
                        consistency_tolerance, &composed, status, left,
                        kDeviceInfeasible, 21, matrix, factors, pivot_columns,
                        pivot_rows, &rank, &best_row, &local_ok);
  WarpSynchronize();
  CopyRelationBlock(composed, &leaves[left]);
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
  }
  bool pivot[kMaxStateDimension]{};
  int pivot_row[kMaxStateDimension];
  for (int i = 0; i < kMaxStateDimension; ++i)
    pivot_row[i] = -1;
  for (int row = 0; row < relation.rows; ++row) {
    int column = -1;
    for (int col = 0; col < relation.left_dim; ++col) {
      if (DeviceAbs(relation.left[row * relation.left_dim + col]) > tolerance) {
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
  for (int row = 0; row < relation.left_dim; ++row)
    for (int col = 0; col < reduced; ++col)
      out.T[row * reduced + col] = Scalar{0};
  for (int col = 0; col < relation.left_dim; ++col) {
    if (pivot[col]) {
      const int row = pivot_row[col];
      const Scalar diagonal = relation.left[row * relation.left_dim + col];
      out.t[col] = relation.rhs[row] / diagonal;
      for (int free = 0; free < reduced; ++free) {
        out.T[col * reduced + free] =
            -relation.left[row * relation.left_dim + out.free_columns[free]] /
            diagonal;
      }
    }
  }
  for (int free = 0; free < reduced; ++free) {
    out.T[out.free_columns[free] * reduced + free] = Scalar{1};
  }
}

__global__ void PackStateDimensionsKernel(const StateParam *state_params,
                                          int count, int *state_dimensions) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  state_dimensions[2 * index] = state_params[index].physical_dim;
  state_dimensions[2 * index + 1] = state_params[index].reduced_dim;
}

__global__ void PackControlDimensionsKernel(const ControlParam *control_params,
                                            int count,
                                            int *control_dimensions) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  control_dimensions[2 * index] = control_params[index].physical_dim;
  control_dimensions[2 * index + 1] = control_params[index].reduced_dim;
}

__global__ void PackDualDimensionsKernel(const DualParam *dual_params,
                                         int count, int *dual_dimensions) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  dual_dimensions[index] = dual_params[index].free_dim;
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
__global__ void ReduceValueLeavesKernel(const ValueElement *, int, int, Scalar,
                                        int *, ValueElement *);
__global__ void ReduceValueTreeLevelKernel(ValueElement *, int, int, int, int,
                                           Scalar, int *);
__global__ void InitializeValueContextRootKernel(ValueElement *, int);
__global__ void ExpandValueContextLevelKernel(ValueElement *, int, int, int,
                                              int, Scalar, int *);
__global__ void FinalizeValueSuffixFromParentsKernel(ValueElement *, int,
                                                     const ValueElement *, int,
                                                     Scalar, int *);
__global__ void FeedbackKernel(const ReducedStage *, const ValueElement *, int,
                               Scalar, Feedback *, DeviceStatus *);
__global__ void SequentialRiccatiKernel(const ReducedStage *, int, Scalar,
                                        ValueElement *, Feedback *,
                                        DeviceStatus *);
__global__ void InitializeAffineMapsKernel(const Feedback *, int, AffineMap *);
__global__ void ReduceAffineLeavesKernel(const AffineMap *, int, int,
                                         AffineMap *, DeviceStatus *);
__global__ void ReduceAffineTreeLevelKernel(AffineMap *, int, int, int, int,
                                            DeviceStatus *);
__global__ void InitializeAffineContextRootKernel(AffineMap *, int);
__global__ void ExpandAffineContextLevelKernel(AffineMap *, int, int, int, int,
                                               DeviceStatus *);
__global__ void FinalizeAffinePrefixFromParentsKernel(AffineMap *, int,
                                                      const AffineMap *, int,
                                                      DeviceStatus *);
__global__ void EvaluateReducedStatesKernel(const AffineMap *, int,
                                            const StateParam *, const Scalar *,
                                            const int *, Scalar *);
__global__ void ReconstructPrimalKernel(const StateParam *,
                                        const ControlParam *, const Feedback *,
                                        const Scalar *, const int *,
                                        const int *, const int *, int, Scalar *,
                                        Scalar *);
__global__ void BuildDualParametersKernel(
    const PackedStage *, const StateParam *, const ValueElement *,
    const Scalar *, const Scalar *, const Scalar *, const int *, const int *,
    const int *, int, Scalar, Scalar, DualParam *, int *, DeviceStatus *);
__global__ void BuildDualParameterRelationsKernel(
    const PackedStage *, const PackedTerminal *, const DualParam *, int,
    const Scalar *, const Scalar *, const int *, const int *, Scalar, Scalar,
    DualRelation *, const int *, StateDualParam *, DeviceStatus *);
__global__ void RecoverParameterizedDynamicsAndMixedKernel(
    const DualParam *, const DualNodeValue *, const int *, const int *, int,
    Scalar *, Scalar *);
__global__ void
RecoverStateMultipliersFromParametersKernel(const StateDualParam *,
                                            const DualNodeValue *, const int *,
                                            int, Scalar *, Scalar *);
__global__ void RecoverInitialMultiplierKernel(
    const PackedStage *, const PackedTerminal *, int, const Scalar *,
    const Scalar *, const Scalar *, const Scalar *, const int *, const int *,
    const int *, const int *, const int *, Scalar *, Scalar *, Scalar *);
__global__ void ReduceDualTreeLevelKernel(const DualRelation *, int, int, int,
                                          int, Scalar, Scalar, DualRelation *,
                                          const int *, DeviceStatus *);
__global__ void SolveDualRootKernel(const DualRelation *, DualNodeValue *,
                                    const int *, DeviceStatus *, Scalar);
__global__ void ExpandDualTreeLevelKernel(const DualRelation *, int, int, int,
                                          int, Scalar, Scalar,
                                          const DualNodeValue *,
                                          DualNodeValue *, const int *,
                                          DeviceStatus *);
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
  DeviceBuffer<Scalar> device_problem_data;
  DeviceBuffer<Scalar> device_initial;
  DeviceBuffer<DeviceStatus> device_status;
  DeviceBuffer<Relation> relation_leaves;
  DeviceBuffer<Relation> relation_scan;
  DeviceBuffer<Scalar> relation_data;
  DeviceBuffer<StateParam> state_params;
  DeviceBuffer<ControlParam> control_params;
  DeviceBuffer<ReducedStage> reduced_stages;
  DeviceBuffer<ReducedTerminal> reduced_terminal;
  DeviceBuffer<Scalar> reduced_initial;
  DeviceBuffer<ValueElement> value_leaves;
  DeviceBuffer<ValueElement> value_scan;
  DeviceBuffer<Scalar> value_data;
  DeviceBuffer<Feedback> feedback;
  DeviceBuffer<int> parallel_ok;
  DeviceBuffer<AffineMap> map_leaves;
  DeviceBuffer<AffineMap> map_scan;
  DeviceBuffer<Scalar> map_data;
  DeviceBuffer<Scalar> reduced_states;
  DeviceBuffer<Scalar> states;
  DeviceBuffer<Scalar> controls;
  DeviceBuffer<int> state_offsets;
  DeviceBuffer<int> reduced_state_offsets;
  DeviceBuffer<int> control_offsets;
  DeviceBuffer<int> dynamics_offsets;
  DeviceBuffer<int> mixed_offsets;
  DeviceBuffer<int> state_constraint_offsets;
  DeviceBuffer<int> state_dimensions;
  DeviceBuffer<int> control_dimensions;
  DeviceBuffer<DualParam> dual_params;
  DeviceBuffer<int> dual_dimensions;
  DeviceBuffer<int> dual_scan_needed;
  DeviceBuffer<StateDualParam> state_dual_params;
  DeviceBuffer<DualRelation> dual_tree;
  DeviceBuffer<DualNodeValue> dual_values;
  DeviceBuffer<Scalar> dual_relation_data;
  DeviceBuffer<Scalar> dual_value_data;
  DeviceBuffer<Scalar> initial_multiplier;
  DeviceBuffer<Scalar> dynamics_multipliers;
  DeviceBuffer<Scalar> mixed_multipliers;
  DeviceBuffer<Scalar> state_multipliers;
  DeviceBuffer<Scalar> terminal_multiplier;

  PinnedBuffer<PackedStage> host_stages;
  PinnedBuffer<PackedTerminal> host_terminal;
  PinnedBuffer<Scalar> host_problem_data;
  PinnedBuffer<Scalar> host_initial;
  PinnedBuffer<Scalar> host_states;
  PinnedBuffer<Scalar> host_controls;
  PinnedBuffer<Scalar> host_initial_multiplier;
  PinnedBuffer<Scalar> host_dynamics;
  PinnedBuffer<Scalar> host_mixed;
  PinnedBuffer<Scalar> host_state_multipliers;
  PinnedBuffer<Scalar> host_terminal_multiplier;
  PinnedBuffer<int> host_state_offsets;
  PinnedBuffer<int> host_reduced_state_offsets;
  PinnedBuffer<int> host_control_offsets;
  PinnedBuffer<int> host_dynamics_offsets;
  PinnedBuffer<int> host_mixed_offsets;
  PinnedBuffer<int> host_state_constraint_offsets;
  PinnedBuffer<int> host_state_dimensions;
  PinnedBuffer<int> host_control_dimensions;
  PinnedBuffer<Relation> host_relation_leaves;
  PinnedBuffer<Relation> host_relation_scan;
  PinnedBuffer<ValueElement> host_value_leaves;
  PinnedBuffer<ValueElement> host_value_scan;
  PinnedBuffer<AffineMap> host_map_leaves;
  PinnedBuffer<AffineMap> host_map_scan;
  PinnedBuffer<int> host_parallel_ok;
  PinnedBuffer<int> host_dual_scan_needed;
  PinnedBuffer<int> host_dual_dimensions;
  PinnedBuffer<DualRelation> host_dual_tree;
  PinnedBuffer<DualNodeValue> host_dual_values;
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

  void Reserve(int requested_device, int stage_count, int node_count,
               std::size_t state_entries, std::size_t control_entries,
               std::size_t mixed_entries, std::size_t state_constraint_entries,
               std::size_t problem_data_entries, int initial_state_dimension,
               int terminal_constraint_dimension) {
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
    device_problem_data.Reserve(problem_data_entries);
    device_initial.Reserve(initial_state_dimension);
    device_status.Reserve(1);
    relation_leaves.Reserve(node_count);
    relation_scan.Reserve(total_tree_nodes - node_count);
    state_params.Reserve(node_count);
    control_params.Reserve(stage_count);
    reduced_stages.Reserve(stage_count);
    reduced_terminal.Reserve(1);
    reduced_initial.Reserve(initial_state_dimension);
    value_leaves.Reserve(node_count);
    value_scan.Reserve(total_tree_nodes - node_count);
    feedback.Reserve(stage_count);
    parallel_ok.Reserve(1);
    map_leaves.Reserve(stage_count);
    map_scan.Reserve(total_stage_tree_nodes - std::max(stage_count, 1));
    states.Reserve(state_entries);
    controls.Reserve(control_entries);
    state_offsets.Reserve(node_count + 1);
    reduced_state_offsets.Reserve(node_count + 1);
    control_offsets.Reserve(stage_count + 1);
    dynamics_offsets.Reserve(stage_count + 1);
    mixed_offsets.Reserve(stage_count + 1);
    state_constraint_offsets.Reserve(stage_count + 1);
    state_dimensions.Reserve(static_cast<std::size_t>(2) * node_count);
    control_dimensions.Reserve(static_cast<std::size_t>(2) * stage_count);
    dual_params.Reserve(stage_count);
    dual_dimensions.Reserve(stage_count);
    dual_scan_needed.Reserve(1);
    state_dual_params.Reserve(stage_count);
    initial_multiplier.Reserve(initial_state_dimension);
    dynamics_multipliers.Reserve(state_entries - initial_state_dimension);
    mixed_multipliers.Reserve(mixed_entries);
    state_multipliers.Reserve(state_constraint_entries);
    terminal_multiplier.Reserve(terminal_constraint_dimension);

    host_stages.Resize(stage_count);
    host_terminal.Resize(1);
    host_problem_data.Resize(problem_data_entries);
    host_initial.Resize(initial_state_dimension);
    host_states.Resize(state_entries);
    host_controls.Resize(control_entries);
    host_initial_multiplier.Resize(initial_state_dimension);
    host_dynamics.Resize(state_entries - initial_state_dimension);
    host_mixed.Resize(mixed_entries);
    host_state_multipliers.Resize(state_constraint_entries);
    host_terminal_multiplier.Resize(terminal_constraint_dimension);
    host_state_offsets.Resize(node_count + 1);
    host_reduced_state_offsets.Resize(node_count + 1);
    host_control_offsets.Resize(stage_count + 1);
    host_dynamics_offsets.Resize(stage_count + 1);
    host_mixed_offsets.Resize(stage_count + 1);
    host_state_constraint_offsets.Resize(stage_count + 1);
    host_state_dimensions.Resize(static_cast<std::size_t>(2) * node_count);
    host_control_dimensions.Resize(static_cast<std::size_t>(2) * stage_count);
    host_relation_leaves.Resize(node_count);
    host_relation_scan.Resize(total_tree_nodes - node_count);
    host_value_leaves.Resize(node_count);
    host_value_scan.Resize(total_tree_nodes - node_count);
    host_map_leaves.Resize(stage_count);
    host_map_scan.Resize(total_stage_tree_nodes - std::max(stage_count, 1));
    host_parallel_ok.Resize(1);
    host_dual_scan_needed.Resize(1);
    host_dual_dimensions.Resize(stage_count);
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

void PackMatrix(const Matrix &source, Scalar **host_cursor,
                Scalar **device_cursor, const Scalar **target) {
  *target = *device_cursor;
  for (std::size_t row = 0; row < source.rows(); ++row) {
    for (std::size_t col = 0; col < source.cols(); ++col) {
      (*host_cursor)[row * source.cols() + col] = source(row, col);
    }
  }
  const std::size_t entries = source.rows() * source.cols();
  *host_cursor += entries;
  *device_cursor += entries;
}

void PackVector(const Vector &source, Scalar **host_cursor,
                Scalar **device_cursor, const Scalar **target) {
  *target = *device_cursor;
  for (std::size_t i = 0; i < source.size(); ++i)
    (*host_cursor)[i] = source[i];
  *host_cursor += source.size();
  *device_cursor += source.size();
}

void PackStage(const Stage &source, Scalar **host_cursor,
               Scalar **device_cursor, PackedStage *out) {
  out->n = static_cast<int>(source.A.cols());
  out->next_n = static_cast<int>(source.A.rows());
  out->m = static_cast<int>(source.B.cols());
  out->mixed = static_cast<int>(source.C.rows());
  out->state = static_cast<int>(source.E.rows());
  PackMatrix(source.A, host_cursor, device_cursor, &out->A);
  PackMatrix(source.B, host_cursor, device_cursor, &out->B);
  PackVector(source.c, host_cursor, device_cursor, &out->c);
  PackMatrix(source.Q, host_cursor, device_cursor, &out->Q);
  PackMatrix(source.R, host_cursor, device_cursor, &out->R);
  PackMatrix(source.M, host_cursor, device_cursor, &out->M);
  PackVector(source.q, host_cursor, device_cursor, &out->q);
  PackVector(source.r, host_cursor, device_cursor, &out->r);
  PackMatrix(source.C, host_cursor, device_cursor, &out->C);
  PackMatrix(source.D, host_cursor, device_cursor, &out->D);
  PackVector(source.d, host_cursor, device_cursor, &out->d);
  PackMatrix(source.E, host_cursor, device_cursor, &out->E);
  PackVector(source.e, host_cursor, device_cursor, &out->e);
}

void PackTerminal(const Problem &problem, Scalar **host_cursor,
                  Scalar **device_cursor, PackedTerminal *out) {
  out->n = static_cast<int>(problem.terminal_Q.rows());
  out->state = static_cast<int>(problem.terminal_E.rows());
  PackMatrix(problem.terminal_Q, host_cursor, device_cursor, &out->Q);
  PackVector(problem.terminal_q, host_cursor, device_cursor, &out->q);
  PackMatrix(problem.terminal_E, host_cursor, device_cursor, &out->E);
  PackVector(problem.terminal_e, host_cursor, device_cursor, &out->e);
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

struct CompactEntryCounts {
  std::size_t states = 0;
  std::size_t controls = 0;
  std::size_t mixed = 0;
  std::size_t state_constraints = 0;
  std::size_t problem_data = 0;
};

CompactEntryCounts CountCompactEntries(const Problem &problem) {
  CompactEntryCounts counts;
  counts.states = problem.initial_state.size();
  for (const Stage &stage : problem.stages) {
    const std::size_t n = stage.A.cols();
    const std::size_t next_n = stage.A.rows();
    const std::size_t m = stage.B.cols();
    const std::size_t mixed = stage.C.rows();
    const std::size_t state_constraints = stage.E.rows();
    counts.states += stage.A.rows();
    counts.controls += m;
    counts.mixed += mixed;
    counts.state_constraints += state_constraints;
    counts.problem_data += next_n * n + next_n * m + next_n + n * n + m * m +
                           n * m + n + m + mixed * n + mixed * m + mixed +
                           state_constraints * n + state_constraints;
  }
  const std::size_t terminal_n = problem.terminal_Q.rows();
  const std::size_t terminal_constraints = problem.terminal_E.rows();
  counts.problem_data += terminal_n * terminal_n + terminal_n +
                         terminal_constraints * terminal_n +
                         terminal_constraints;
  constexpr std::size_t kMaxOffset =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  Require(counts.states <= kMaxOffset && counts.controls <= kMaxOffset &&
              counts.mixed <= kMaxOffset &&
              counts.state_constraints <= kMaxOffset,
          "CUDA compact buffer offsets exceed 32-bit indexing");
  return counts;
}

void BuildCompactOffsets(const Problem &problem, WorkspaceStorage *workspace) {
  auto &state = workspace->host_state_offsets;
  auto &control = workspace->host_control_offsets;
  auto &dynamics = workspace->host_dynamics_offsets;
  auto &mixed = workspace->host_mixed_offsets;
  auto &state_constraint = workspace->host_state_constraint_offsets;
  state[0] = 0;
  control[0] = 0;
  dynamics[0] = 0;
  mixed[0] = 0;
  state_constraint[0] = 0;
  for (std::size_t index = 0; index < problem.stages.size(); ++index) {
    const Stage &stage = problem.stages[index];
    state[index + 1] = state[index] + static_cast<int>(stage.A.cols());
    control[index + 1] = control[index] + static_cast<int>(stage.B.cols());
    dynamics[index + 1] = dynamics[index] + static_cast<int>(stage.A.rows());
    mixed[index + 1] = mixed[index] + static_cast<int>(stage.C.rows());
    state_constraint[index + 1] =
        state_constraint[index] + static_cast<int>(stage.E.rows());
  }
  state[problem.stages.size() + 1] =
      state[problem.stages.size()] +
      static_cast<int>(problem.terminal_Q.rows());
}

struct ScanShape {
  int left = -1;
  int right = 0;
};

ScanShape ComposeScanShapes(const ScanShape &first, const ScanShape &second) {
  if (first.left < 0)
    return second;
  if (second.left < 0)
    return first;
  Require(first.right == second.left,
          "internal CUDA scan layout has incompatible dimensions");
  return {first.left, second.right};
}

struct ValueCapacity {
  std::size_t a = 0;
  std::size_t b = 0;
  std::size_t c = 0;
  std::size_t eta = 0;
  std::size_t j = 0;

  void Include(const ScanShape &shape) {
    if (shape.left < 0)
      return;
    const std::size_t left = static_cast<std::size_t>(shape.left);
    const std::size_t right = static_cast<std::size_t>(shape.right);
    a = std::max(a, right * left);
    b = std::max(b, right);
    c = std::max(c, right * right);
    eta = std::max(eta, left);
    j = std::max(j, left * left);
  }

  std::size_t Entries() const { return a + b + c + eta + j; }
};

struct RelationCapacity {
  std::size_t left = 0;
  std::size_t right = 0;
  std::size_t rhs = 0;

  void Include(const ScanShape &shape) {
    if (shape.left < 0)
      return;
    const std::size_t left_dimension = static_cast<std::size_t>(shape.left);
    const std::size_t right_dimension = static_cast<std::size_t>(shape.right);
    const std::size_t rows = left_dimension + right_dimension;
    left = std::max(left, rows * left_dimension);
    right = std::max(right, rows * right_dimension);
    rhs = std::max(rhs, rows);
  }

  std::size_t Entries() const { return left + right + rhs; }
};

struct MapCapacity {
  std::size_t linear = 0;
  std::size_t offset = 0;

  void Include(const ScanShape &shape) {
    if (shape.left < 0)
      return;
    linear =
        std::max(linear, static_cast<std::size_t>(shape.left) * shape.right);
    offset = std::max(offset, static_cast<std::size_t>(shape.right));
  }

  std::size_t Entries() const { return linear + offset; }
};

template <typename Capacity>
void PlanSuffixScanStorage(const std::vector<ScanShape> &leaves,
                           const std::vector<int> &level_offsets,
                           const std::vector<int> &level_counts,
                           std::vector<Capacity> *leaf_capacity,
                           std::vector<Capacity> *internal_capacity) {
  const int leaf_count = static_cast<int>(leaves.size());
  const int internal_count =
      level_offsets.back() + level_counts.back() - leaf_count;
  leaf_capacity->assign(leaf_count, Capacity{});
  internal_capacity->assign(internal_count, Capacity{});
  for (int leaf = 0; leaf < leaf_count; ++leaf)
    (*leaf_capacity)[leaf].Include(leaves[leaf]);

  std::vector<ScanShape> reductions(internal_count);
  std::vector<ScanShape> contexts(internal_count);
  for (std::size_t level = 1; level < level_counts.size(); ++level) {
    const int offset = level_offsets[level] - leaf_count;
    const int child_count = level_counts[level - 1];
    const int child_offset = level_offsets[level - 1] - leaf_count;
    for (int parent = 0; parent < level_counts[level]; ++parent) {
      const int child = 2 * parent;
      const ScanShape first =
          level == 1 ? leaves[child] : reductions[child_offset + child];
      ScanShape result = first;
      if (child + 1 < child_count) {
        const ScanShape second = level == 1
                                     ? leaves[child + 1]
                                     : reductions[child_offset + child + 1];
        result = ComposeScanShapes(first, second);
      }
      reductions[offset + parent] = result;
      (*internal_capacity)[offset + parent].Include(result);
    }
  }

  if (leaf_count == 1)
    return;
  contexts[level_offsets.back() - leaf_count] = ScanShape{};
  for (int level = static_cast<int>(level_counts.size()) - 2; level >= 1;
       --level) {
    const int child_offset = level_offsets[level] - leaf_count;
    const int parent_offset = level_offsets[level + 1] - leaf_count;
    for (int parent = 0; parent < level_counts[level + 1]; ++parent) {
      const int child = 2 * parent;
      const ScanShape parent_context = contexts[parent_offset + parent];
      if (child + 1 >= level_counts[level]) {
        contexts[child_offset + child] = parent_context;
        (*internal_capacity)[child_offset + child].Include(parent_context);
        continue;
      }
      const ScanShape left_context = ComposeScanShapes(
          reductions[child_offset + child + 1], parent_context);
      contexts[child_offset + child] = left_context;
      contexts[child_offset + child + 1] = parent_context;
      (*internal_capacity)[child_offset + child].Include(left_context);
      (*internal_capacity)[child_offset + child + 1].Include(parent_context);
    }
  }

  const int parent_offset = level_offsets[1] - leaf_count;
  for (int parent = 0; parent < level_counts[1]; ++parent) {
    const int child = 2 * parent;
    const ScanShape parent_context = contexts[parent_offset + parent];
    if (child + 1 >= leaf_count) {
      (*leaf_capacity)[child].Include(
          ComposeScanShapes(leaves[child], parent_context));
      continue;
    }
    const ScanShape right_suffix =
        ComposeScanShapes(leaves[child + 1], parent_context);
    const ScanShape left_suffix =
        ComposeScanShapes(leaves[child], right_suffix);
    (*leaf_capacity)[child + 1].Include(right_suffix);
    (*leaf_capacity)[child].Include(left_suffix);
  }
}

void BindValueStorage(ValueElement *element, Scalar **cursor,
                      const ValueCapacity &capacity) {
  element->left_dim = -1;
  element->right_dim = 0;
  element->A = *cursor;
  *cursor += capacity.a;
  element->b = *cursor;
  *cursor += capacity.b;
  element->C = *cursor;
  *cursor += capacity.c;
  element->eta = *cursor;
  *cursor += capacity.eta;
  element->J = *cursor;
  *cursor += capacity.j;
}

void BindRelationStorage(Relation *relation, Scalar **cursor,
                         const RelationCapacity &capacity) {
  relation->left_dim = -1;
  relation->right_dim = 0;
  relation->rows = 0;
  relation->left = *cursor;
  *cursor += capacity.left;
  relation->right = *cursor;
  *cursor += capacity.right;
  relation->rhs = *cursor;
  *cursor += capacity.rhs;
}

void BindMapStorage(AffineMap *map, Scalar **cursor,
                    const MapCapacity &capacity) {
  map->left_dim = -1;
  map->right_dim = 0;
  map->linear = *cursor;
  *cursor += capacity.linear;
  map->offset = *cursor;
  *cursor += capacity.offset;
}

void PrepareRelationStorage(const Problem &problem,
                            WorkspaceStorage *workspace) {
  const int stage_count = static_cast<int>(problem.stages.size());
  const int node_count = stage_count + 1;
  std::vector<ScanShape> leaves(node_count);
  for (int stage = 0; stage < stage_count; ++stage) {
    leaves[stage] = {static_cast<int>(problem.stages[stage].A.cols()),
                     static_cast<int>(problem.stages[stage].A.rows())};
  }
  leaves[stage_count] = {static_cast<int>(problem.terminal_Q.rows()), 0};
  std::vector<RelationCapacity> leaf_capacity;
  std::vector<RelationCapacity> internal_capacity;
  PlanSuffixScanStorage(leaves, workspace->node_level_offsets,
                        workspace->node_level_counts, &leaf_capacity,
                        &internal_capacity);
  std::size_t entries = 0;
  for (const RelationCapacity &capacity : leaf_capacity)
    entries += capacity.Entries();
  for (const RelationCapacity &capacity : internal_capacity)
    entries += capacity.Entries();
  workspace->relation_data.Reserve(entries);
  Scalar *cursor = workspace->relation_data.get();
  for (int node = 0; node < node_count; ++node)
    BindRelationStorage(&workspace->host_relation_leaves[node], &cursor,
                        leaf_capacity[node]);
  for (std::size_t node = 0; node < internal_capacity.size(); ++node)
    BindRelationStorage(&workspace->host_relation_scan[node], &cursor,
                        internal_capacity[node]);
  Require(cursor == workspace->relation_data.get() + entries,
          "internal CUDA relation layout size mismatch");
}

void PrepareValueStorage(WorkspaceStorage *workspace, int stage_count) {
  const int node_count = stage_count + 1;
  const auto &level_offsets = workspace->node_level_offsets;
  const auto &level_counts = workspace->node_level_counts;
  std::vector<ScanShape> leaves(node_count);
  for (int node = 0; node < node_count; ++node) {
    const int left = workspace->host_state_dimensions[2 * node + 1];
    const int right = node == stage_count
                          ? 0
                          : workspace->host_state_dimensions[2 * node + 3];
    leaves[node] = {left, right};
  }

  const int internal_count =
      static_cast<int>(workspace->host_value_scan.size());
  std::vector<ValueCapacity> leaf_capacity;
  std::vector<ValueCapacity> internal_capacity;
  PlanSuffixScanStorage(leaves, level_offsets, level_counts, &leaf_capacity,
                        &internal_capacity);

  std::size_t entries = 0;
  for (const ValueCapacity &capacity : leaf_capacity)
    entries += capacity.Entries();
  for (const ValueCapacity &capacity : internal_capacity)
    entries += capacity.Entries();
  workspace->value_data.Reserve(entries);
  Scalar *cursor = workspace->value_data.get();
  for (int node = 0; node < node_count; ++node)
    BindValueStorage(&workspace->host_value_leaves[node], &cursor,
                     leaf_capacity[node]);
  for (int node = 0; node < internal_count; ++node)
    BindValueStorage(&workspace->host_value_scan[node], &cursor,
                     internal_capacity[node]);
  Require(cursor == workspace->value_data.get() + entries,
          "internal CUDA value layout size mismatch");
}

void PrepareMapStorage(WorkspaceStorage *workspace, int stage_count) {
  if (stage_count == 0)
    return;
  const auto &level_offsets = workspace->stage_level_offsets;
  const auto &level_counts = workspace->stage_level_counts;
  std::vector<ScanShape> leaves(stage_count);
  std::vector<MapCapacity> leaf_capacity(stage_count);
  for (int stage = 0; stage < stage_count; ++stage) {
    leaves[stage] = {workspace->host_state_dimensions[2 * stage + 1],
                     workspace->host_state_dimensions[2 * stage + 3]};
    leaf_capacity[stage].Include(leaves[stage]);
  }

  const int internal_count = static_cast<int>(workspace->host_map_scan.size());
  std::vector<ScanShape> reductions(internal_count);
  std::vector<ScanShape> contexts(internal_count);
  std::vector<MapCapacity> internal_capacity(internal_count);
  for (std::size_t level = 1; level < level_counts.size(); ++level) {
    const int offset = level_offsets[level] - stage_count;
    const int child_count = level_counts[level - 1];
    const int child_offset = level_offsets[level - 1] - stage_count;
    for (int parent = 0; parent < level_counts[level]; ++parent) {
      const int child = 2 * parent;
      const ScanShape first =
          level == 1 ? leaves[child] : reductions[child_offset + child];
      ScanShape result = first;
      if (child + 1 < child_count) {
        const ScanShape second = level == 1
                                     ? leaves[child + 1]
                                     : reductions[child_offset + child + 1];
        result = ComposeScanShapes(first, second);
      }
      reductions[offset + parent] = result;
      internal_capacity[offset + parent].Include(result);
    }
  }

  if (stage_count > 1) {
    contexts[level_offsets.back() - stage_count] = ScanShape{};
    for (int level = static_cast<int>(level_counts.size()) - 2; level >= 1;
         --level) {
      const int child_offset = level_offsets[level] - stage_count;
      const int parent_offset = level_offsets[level + 1] - stage_count;
      for (int parent = 0; parent < level_counts[level + 1]; ++parent) {
        const int child = 2 * parent;
        const ScanShape parent_context = contexts[parent_offset + parent];
        contexts[child_offset + child] = parent_context;
        internal_capacity[child_offset + child].Include(parent_context);
        if (child + 1 < level_counts[level]) {
          const ScanShape right_context = ComposeScanShapes(
              parent_context, reductions[child_offset + child]);
          contexts[child_offset + child + 1] = right_context;
          internal_capacity[child_offset + child + 1].Include(right_context);
        }
      }
    }

    const int parent_offset = level_offsets[1] - stage_count;
    for (int parent = 0; parent < level_counts[1]; ++parent) {
      const int child = 2 * parent;
      const ScanShape parent_context = contexts[parent_offset + parent];
      const ScanShape left_prefix =
          ComposeScanShapes(parent_context, leaves[child]);
      leaf_capacity[child].Include(left_prefix);
      if (child + 1 < stage_count) {
        leaf_capacity[child + 1].Include(
            ComposeScanShapes(left_prefix, leaves[child + 1]));
      }
    }
  }

  std::size_t entries = 0;
  for (const MapCapacity &capacity : leaf_capacity)
    entries += capacity.Entries();
  for (const MapCapacity &capacity : internal_capacity)
    entries += capacity.Entries();
  workspace->map_data.Reserve(entries);
  Scalar *cursor = workspace->map_data.get();
  for (int stage = 0; stage < stage_count; ++stage)
    BindMapStorage(&workspace->host_map_leaves[stage], &cursor,
                   leaf_capacity[stage]);
  for (int node = 0; node < internal_count; ++node)
    BindMapStorage(&workspace->host_map_scan[node], &cursor,
                   internal_capacity[node]);
  Require(cursor == workspace->map_data.get() + entries,
          "internal CUDA map layout size mismatch");
}

struct DualValueCapacity {
  std::size_t left = 0;
  std::size_t right = 0;

  void Include(const ScanShape &shape) {
    left = std::max(left, static_cast<std::size_t>(shape.left));
    right = std::max(right, static_cast<std::size_t>(shape.right));
  }

  std::size_t Entries() const { return left + right; }
};

void BindDualRelationStorage(DualRelation *relation, Scalar **cursor,
                             const RelationCapacity &capacity) {
  relation->left_dim = -1;
  relation->right_dim = 0;
  relation->rows = 0;
  relation->left = *cursor;
  *cursor += capacity.left;
  relation->right = *cursor;
  *cursor += capacity.right;
  relation->rhs = *cursor;
  *cursor += capacity.rhs;
}

void BindDualValueStorage(DualNodeValue *value, Scalar **cursor,
                          const DualValueCapacity &capacity) {
  value->left_dim = -1;
  value->right_dim = 0;
  value->left = *cursor;
  *cursor += capacity.left;
  value->right = *cursor;
  *cursor += capacity.right;
}

void PrepareDualStorage(WorkspaceStorage *workspace, int stage_count) {
  const auto &level_offsets = workspace->stage_level_offsets;
  const auto &level_counts = workspace->stage_level_counts;
  const int tree_size = level_offsets.back() + level_counts.back();
  std::vector<ScanShape> shapes(tree_size);
  std::vector<RelationCapacity> relation_capacity(tree_size);
  std::vector<DualValueCapacity> value_capacity(tree_size);
  for (int stage = 0; stage < stage_count; ++stage) {
    const int right = stage + 1 == stage_count
                          ? 0
                          : workspace->host_dual_dimensions[stage + 1];
    shapes[stage] = {workspace->host_dual_dimensions[stage], right};
    relation_capacity[stage].Include(shapes[stage]);
    value_capacity[stage].Include(shapes[stage]);
  }
  for (std::size_t level = 1; level < level_counts.size(); ++level) {
    const int child_offset = level_offsets[level - 1];
    const int parent_offset = level_offsets[level];
    for (int parent = 0; parent < level_counts[level]; ++parent) {
      const int child = 2 * parent;
      ScanShape shape = shapes[child_offset + child];
      if (child + 1 < level_counts[level - 1]) {
        shape = ComposeScanShapes(shape, shapes[child_offset + child + 1]);
      }
      shapes[parent_offset + parent] = shape;
      relation_capacity[parent_offset + parent].Include(shape);
      value_capacity[parent_offset + parent].Include(shape);
    }
  }

  std::size_t relation_entries = 0;
  std::size_t value_entries = 0;
  for (int node = 0; node < tree_size; ++node) {
    relation_entries += relation_capacity[node].Entries();
    value_entries += value_capacity[node].Entries();
  }
  workspace->dual_tree.Reserve(tree_size);
  workspace->dual_values.Reserve(tree_size);
  workspace->dual_relation_data.Reserve(relation_entries);
  workspace->dual_value_data.Reserve(value_entries);
  workspace->host_dual_tree.Resize(tree_size);
  workspace->host_dual_values.Resize(tree_size);
  Scalar *relation_cursor = workspace->dual_relation_data.get();
  Scalar *value_cursor = workspace->dual_value_data.get();
  for (int node = 0; node < tree_size; ++node) {
    BindDualRelationStorage(&workspace->host_dual_tree[node], &relation_cursor,
                            relation_capacity[node]);
    BindDualValueStorage(&workspace->host_dual_values[node], &value_cursor,
                         value_capacity[node]);
  }
  Require(relation_cursor ==
              workspace->dual_relation_data.get() + relation_entries,
          "internal CUDA dual-relation layout size mismatch");
  Require(value_cursor == workspace->dual_value_data.get() + value_entries,
          "internal CUDA dual-value layout size mismatch");
}

Scalar ObjectiveFromCompact(const Problem &problem, const int *state_offsets,
                            const int *control_offsets, const Scalar *states,
                            const Scalar *controls) {
  Scalar objective = Scalar{0};
  for (std::size_t i = 0; i < problem.stages.size(); ++i) {
    const Stage &s = problem.stages[i];
    const Scalar *x = states + state_offsets[i];
    const Scalar *u = controls + control_offsets[i];
    for (std::size_t row = 0; row < s.Q.rows(); ++row) {
      objective += s.q[row] * x[row];
      for (std::size_t col = 0; col < s.Q.cols(); ++col)
        objective += Scalar{0.5} * x[row] * s.Q(row, col) * x[col];
      for (std::size_t col = 0; col < s.M.cols(); ++col)
        objective += x[row] * s.M(row, col) * u[col];
    }
    for (std::size_t row = 0; row < s.R.rows(); ++row) {
      objective += s.r[row] * u[row];
      for (std::size_t col = 0; col < s.R.cols(); ++col)
        objective += Scalar{0.5} * u[row] * s.R(row, col) * u[col];
    }
  }
  const Scalar *x = states + state_offsets[problem.stages.size()];
  for (std::size_t row = 0; row < problem.terminal_Q.rows(); ++row) {
    objective += problem.terminal_q[row] * x[row];
    for (std::size_t col = 0; col < problem.terminal_Q.cols(); ++col)
      objective += Scalar{0.5} * x[row] * problem.terminal_Q(row, col) * x[col];
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
  const CompactEntryCounts entry_counts = CountCompactEntries(problem);
  workspace.Reserve(
      options.device, stage_count, node_count, entry_counts.states,
      entry_counts.controls, entry_counts.mixed, entry_counts.state_constraints,
      entry_counts.problem_data, static_cast<int>(problem.initial_state.size()),
      static_cast<int>(problem.terminal_E.rows()));
  BuildCompactOffsets(problem, &workspace);
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
  PrepareRelationStorage(problem, &workspace);
  auto &host_stages = workspace.host_stages;
  Scalar *host_problem_cursor = workspace.host_problem_data.data();
  Scalar *device_problem_cursor = workspace.device_problem_data.get();
  for (std::size_t index = 0; index < problem.stages.size(); ++index)
    PackStage(problem.stages[index], &host_problem_cursor,
              &device_problem_cursor, &host_stages[index]);
  PackedTerminal &terminal = workspace.host_terminal[0];
  PackTerminal(problem, &host_problem_cursor, &device_problem_cursor,
               &terminal);
  Require(host_problem_cursor == workspace.host_problem_data.data() +
                                     workspace.host_problem_data.size(),
          "internal CUDA problem-packing size mismatch");
  auto &host_initial = workspace.host_initial;
  std::fill(host_initial.begin(), host_initial.end(), Scalar{0});
  for (std::size_t i = 0; i < problem.initial_state.size(); ++i)
    host_initial[i] = problem.initial_state[i];

  auto &device_stages = workspace.device_stages;
  auto &device_terminal = workspace.device_terminal;
  auto &device_initial = workspace.device_initial;
  auto &device_status = workspace.device_status;
  auto &state_offsets = workspace.state_offsets;
  auto &reduced_state_offsets = workspace.reduced_state_offsets;
  auto &control_offsets = workspace.control_offsets;
  auto &dynamics_offsets = workspace.dynamics_offsets;
  auto &mixed_offsets = workspace.mixed_offsets;
  auto &state_constraint_offsets = workspace.state_constraint_offsets;
  auto &relation_a = workspace.relation_leaves;
  auto &relation_b = workspace.relation_scan;
  auto &state_params = workspace.state_params;
  auto &control_params = workspace.control_params;
  auto &state_dimensions = workspace.state_dimensions;
  auto &control_dimensions = workspace.control_dimensions;
  auto &reduced_stages = workspace.reduced_stages;
  auto &reduced_terminal = workspace.reduced_terminal;
  auto &reduced_initial = workspace.reduced_initial;

  solution.timings.upload_ms = TimeGpu(workspace, [&] {
    CudaCheck(cudaMemcpyAsync(
                  relation_a.get(), workspace.host_relation_leaves.data(),
                  workspace.host_relation_leaves.size() * sizeof(Relation),
                  cudaMemcpyHostToDevice),
              "upload compact relation leaves");
    if (workspace.host_relation_scan.size() > 0) {
      CudaCheck(cudaMemcpyAsync(
                    relation_b.get(), workspace.host_relation_scan.data(),
                    workspace.host_relation_scan.size() * sizeof(Relation),
                    cudaMemcpyHostToDevice),
                "upload compact relation tree");
    }
    if (stage_count > 0) {
      CudaCheck(cudaMemcpyAsync(device_stages.get(), host_stages.data(),
                                host_stages.size() * sizeof(PackedStage),
                                cudaMemcpyHostToDevice),
                "upload stages");
    }
    CudaCheck(cudaMemcpyAsync(device_terminal.get(), &terminal,
                              sizeof(PackedTerminal), cudaMemcpyHostToDevice),
              "upload terminal data");
    CudaCheck(
        cudaMemcpyAsync(workspace.device_problem_data.get(),
                        workspace.host_problem_data.data(),
                        workspace.host_problem_data.size() * sizeof(Scalar),
                        cudaMemcpyHostToDevice),
        "upload compact problem data");
    CudaCheck(cudaMemcpyAsync(device_initial.get(), host_initial.data(),
                              host_initial.size() * sizeof(Scalar),
                              cudaMemcpyHostToDevice),
              "upload initial state");
    CudaCheck(cudaMemcpyAsync(state_offsets.get(),
                              workspace.host_state_offsets.data(),
                              workspace.host_state_offsets.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
              "upload state offsets");
    CudaCheck(cudaMemcpyAsync(
                  control_offsets.get(), workspace.host_control_offsets.data(),
                  workspace.host_control_offsets.size() * sizeof(int),
                  cudaMemcpyHostToDevice),
              "upload control offsets");
    CudaCheck(
        cudaMemcpyAsync(dynamics_offsets.get(),
                        workspace.host_dynamics_offsets.data(),
                        workspace.host_dynamics_offsets.size() * sizeof(int),
                        cudaMemcpyHostToDevice),
        "upload dynamics-multiplier offsets");
    CudaCheck(cudaMemcpyAsync(mixed_offsets.get(),
                              workspace.host_mixed_offsets.data(),
                              workspace.host_mixed_offsets.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
              "upload mixed offsets");
    CudaCheck(cudaMemcpyAsync(state_constraint_offsets.get(),
                              workspace.host_state_constraint_offsets.data(),
                              workspace.host_state_constraint_offsets.size() *
                                  sizeof(int),
                              cudaMemcpyHostToDevice),
              "upload state-constraint offsets");
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
    if (node_count > 1) {
      const int first_parent_count = level_counts[1];
      ReduceRelationLeavesKernel<<<first_parent_count, kThreads>>>(
          relation_a.get(), node_count, first_parent_count, options.tolerance,
          feasibility_consistency_tolerance, relation_b.get(),
          device_status.get());
      for (std::size_t level = 1; level + 1 < level_counts.size(); ++level) {
        ReduceRelationTreeLevelKernel<<<level_counts[level + 1], kThreads>>>(
            relation_b.get(), level_offsets[level] - node_count,
            level_offsets[level + 1] - node_count, level_counts[level],
            level_counts[level + 1], options.tolerance,
            feasibility_consistency_tolerance, device_status.get());
      }
      InitializeRelationContextRootKernel<<<1, kThreads>>>(
          relation_b.get(), level_offsets.back() - node_count);
      for (int level = static_cast<int>(level_counts.size()) - 2; level >= 1;
           --level) {
        ExpandRelationContextLevelKernel<<<level_counts[level + 1], kThreads>>>(
            relation_b.get(), level_offsets[level] - node_count,
            level_offsets[level + 1] - node_count, level_counts[level],
            level_counts[level + 1], options.tolerance,
            feasibility_consistency_tolerance, device_status.get());
      }
      FinalizeRelationSuffixFromParentsKernel<<<first_parent_count, kThreads>>>(
          relation_a.get(), node_count, relation_b.get(), first_parent_count,
          options.tolerance, feasibility_consistency_tolerance,
          device_status.get());
    }
    const int blocks = (node_count + kThreads - 1) / kThreads;
    StateParamKernel<<<blocks, kThreads>>>(
        relation_a.get(), node_count, state_params.get(), device_status.get(),
        options.tolerance);
    PackStateDimensionsKernel<<<blocks, kThreads>>>(
        state_params.get(), node_count, state_dimensions.get());
    CudaCheck(cudaMemcpyAsync(workspace.host_status.data(), device_status.get(),
                              sizeof(DeviceStatus), cudaMemcpyDeviceToHost),
              "read feasibility status");
    CudaCheck(
        cudaMemcpyAsync(workspace.host_state_dimensions.data(),
                        state_dimensions.get(),
                        workspace.host_state_dimensions.size() * sizeof(int),
                        cudaMemcpyDeviceToHost),
        "download reduced state dimensions");
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
  workspace.host_reduced_state_offsets[0] = 0;
  for (int index = 0; index < node_count; ++index) {
    workspace.host_reduced_state_offsets[index + 1] =
        workspace.host_reduced_state_offsets[index] +
        workspace.host_state_dimensions[2 * index + 1];
  }
  workspace.reduced_states.Reserve(
      workspace.host_reduced_state_offsets[node_count]);

  solution.timings.reduction_ms = TimeGpu(workspace, [&] {
    CudaCheck(cudaMemcpyAsync(reduced_state_offsets.get(),
                              workspace.host_reduced_state_offsets.data(),
                              workspace.host_reduced_state_offsets.size() *
                                  sizeof(int),
                              cudaMemcpyHostToDevice),
              "upload reduced-state offsets");
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
    if (stage_count > 0) {
      const int control_blocks = (stage_count + kThreads - 1) / kThreads;
      PackControlDimensionsKernel<<<control_blocks, kThreads>>>(
          control_params.get(), stage_count, control_dimensions.get());
      CudaCheck(cudaMemcpyAsync(workspace.host_control_dimensions.data(),
                                control_dimensions.get(),
                                workspace.host_control_dimensions.size() *
                                    sizeof(int),
                                cudaMemcpyDeviceToHost),
                "download reduced control dimensions");
    }
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

  PrepareValueStorage(&workspace, stage_count);
  PrepareMapStorage(&workspace, stage_count);

  auto &value_a = workspace.value_leaves;
  auto &value_b = workspace.value_scan;
  auto &feedback = workspace.feedback;
  auto &parallel_ok = workspace.parallel_ok;
  ValueElement *value_suffix = value_a.get();
  int &host_parallel_ok = workspace.host_parallel_ok[0];
  host_parallel_ok = 1;
  solution.timings.riccati_ms = TimeGpu(workspace, [&] {
    CudaCheck(cudaMemcpyAsync(value_a.get(), workspace.host_value_leaves.data(),
                              workspace.host_value_leaves.size() *
                                  sizeof(ValueElement),
                              cudaMemcpyHostToDevice),
              "upload compact value leaves");
    if (workspace.host_value_scan.size() > 0) {
      CudaCheck(cudaMemcpyAsync(value_b.get(), workspace.host_value_scan.data(),
                                workspace.host_value_scan.size() *
                                    sizeof(ValueElement),
                                cudaMemcpyHostToDevice),
                "upload compact value tree");
    }
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
      if (node_count > 1) {
        const int first_parent_count = level_counts[1];
        ReduceValueLeavesKernel<<<first_parent_count, kThreads>>>(
            value_a.get(), node_count, first_parent_count, options.tolerance,
            parallel_ok.get(), value_b.get());
        for (std::size_t level = 1; level + 1 < level_counts.size(); ++level) {
          ReduceValueTreeLevelKernel<<<level_counts[level + 1], kThreads>>>(
              value_b.get(), level_offsets[level] - node_count,
              level_offsets[level + 1] - node_count, level_counts[level],
              level_counts[level + 1], options.tolerance, parallel_ok.get());
        }
        InitializeValueContextRootKernel<<<1, kThreads>>>(
            value_b.get(), level_offsets.back() - node_count);
        for (int level = static_cast<int>(level_counts.size()) - 2; level >= 1;
             --level) {
          ExpandValueContextLevelKernel<<<level_counts[level + 1], kThreads>>>(
              value_b.get(), level_offsets[level] - node_count,
              level_offsets[level + 1] - node_count, level_counts[level],
              level_counts[level + 1], options.tolerance, parallel_ok.get());
        }
        FinalizeValueSuffixFromParentsKernel<<<first_parent_count, kThreads>>>(
            value_a.get(), node_count, value_b.get(), first_parent_count,
            options.tolerance, parallel_ok.get());
      }
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
  AffineMap *prefix = map_a.get();
  solution.timings.reconstruction_ms = TimeGpu(workspace, [&] {
    if (stage_count > 0) {
      CudaCheck(
          cudaMemcpyAsync(map_a.get(), workspace.host_map_leaves.data(),
                          workspace.host_map_leaves.size() * sizeof(AffineMap),
                          cudaMemcpyHostToDevice),
          "upload compact affine leaves");
      if (workspace.host_map_scan.size() > 0) {
        CudaCheck(
            cudaMemcpyAsync(map_b.get(), workspace.host_map_scan.data(),
                            workspace.host_map_scan.size() * sizeof(AffineMap),
                            cudaMemcpyHostToDevice),
            "upload compact affine tree");
      }
      InitializeAffineMapsKernel<<<stage_count, kThreads>>>(
          feedback.get(), stage_count, map_a.get());
      if (stage_count > 1) {
        const int first_parent_count = stage_level_counts[1];
        ReduceAffineLeavesKernel<<<first_parent_count, kThreads>>>(
            map_a.get(), stage_count, first_parent_count, map_b.get(),
            device_status.get());
        for (std::size_t level = 1; level + 1 < stage_level_counts.size();
             ++level) {
          ReduceAffineTreeLevelKernel<<<stage_level_counts[level + 1],
                                        kThreads>>>(
              map_b.get(), stage_level_offsets[level] - stage_count,
              stage_level_offsets[level + 1] - stage_count,
              stage_level_counts[level], stage_level_counts[level + 1],
              device_status.get());
        }
        InitializeAffineContextRootKernel<<<1, kThreads>>>(
            map_b.get(), stage_level_offsets.back() - stage_count);
        for (int level = static_cast<int>(stage_level_counts.size()) - 2;
             level >= 1; --level) {
          ExpandAffineContextLevelKernel<<<stage_level_counts[level + 1],
                                           kThreads>>>(
              map_b.get(), stage_level_offsets[level] - stage_count,
              stage_level_offsets[level + 1] - stage_count,
              stage_level_counts[level], stage_level_counts[level + 1],
              device_status.get());
        }
        FinalizeAffinePrefixFromParentsKernel<<<first_parent_count, kThreads>>>(
            map_a.get(), stage_count, map_b.get(), first_parent_count,
            device_status.get());
      }
      prefix = map_a.get();
    }
    const int state_blocks = (node_count + kThreads - 1) / kThreads;
    EvaluateReducedStatesKernel<<<state_blocks, kThreads>>>(
        prefix, stage_count, state_params.get(), reduced_initial.get(),
        reduced_state_offsets.get(), reduced_states.get());
    ReconstructPrimalKernel<<<state_blocks, kThreads>>>(
        state_params.get(), control_params.get(), feedback.get(),
        reduced_states.get(), reduced_state_offsets.get(), state_offsets.get(),
        control_offsets.get(), stage_count, states.get(), controls.get());
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

  // First recover the part of each dynamics/mixed multiplier fixed by the
  // reduced costate and control stationarity.  Only genuinely free components
  // remain in the balanced relation tree; in the common full-column-rank case
  // that tree has zero-dimensional endpoints.  State-only and endpoint
  // multipliers then follow independently from state stationarity.
  const Scalar multiplier_rank_tolerance =
      std::max(options.tolerance, kMinimumMultiplierRankTolerance);
  const Scalar multiplier_consistency_tolerance =
      options.enforce_multiplier_consistency
          ? std::max(multiplier_rank_tolerance,
                     kMultiplierConsistencyTolerancePerTreeLevel *
                         stage_level_counts.size())
          : kScalarMax;
  auto &dual_params = workspace.dual_params;
  auto &dual_dimensions = workspace.dual_dimensions;
  auto &dual_scan_needed = workspace.dual_scan_needed;
  auto &state_dual_params = workspace.state_dual_params;
  auto &dual_tree = workspace.dual_tree;
  auto &dual_values = workspace.dual_values;
  auto &initial_multiplier = workspace.initial_multiplier;
  auto &dynamics_multipliers = workspace.dynamics_multipliers;
  auto &mixed_multipliers = workspace.mixed_multipliers;
  auto &state_multipliers = workspace.state_multipliers;
  auto &terminal_multiplier = workspace.terminal_multiplier;
  int &host_dual_scan_needed = workspace.host_dual_scan_needed[0];
  host_dual_scan_needed = 0;
  solution.timings.multiplier_ms = 0.0;
  if (stage_count > 0) {
    solution.timings.multiplier_ms += TimeGpu(workspace, [&] {
      CudaCheck(cudaMemsetAsync(dual_scan_needed.get(), 0, sizeof(int)),
                "initialize dual scan flag");
      BuildDualParametersKernel<<<stage_count, kThreads>>>(
          device_stages.get(), state_params.get(), value_suffix,
          reduced_states.get(), states.get(), controls.get(),
          reduced_state_offsets.get(), state_offsets.get(),
          control_offsets.get(), stage_count, multiplier_rank_tolerance,
          multiplier_consistency_tolerance, dual_params.get(),
          dual_scan_needed.get(), device_status.get());
      const int dual_dimension_blocks = (stage_count + kThreads - 1) / kThreads;
      PackDualDimensionsKernel<<<dual_dimension_blocks, kThreads>>>(
          dual_params.get(), stage_count, dual_dimensions.get());
      CudaCheck(cudaMemcpyAsync(&host_dual_scan_needed, dual_scan_needed.get(),
                                sizeof(int), cudaMemcpyDeviceToHost),
                "read dual scan flag");
      CudaCheck(
          cudaMemcpyAsync(workspace.host_dual_dimensions.data(),
                          dual_dimensions.get(),
                          workspace.host_dual_dimensions.size() * sizeof(int),
                          cudaMemcpyDeviceToHost),
          "read dual dimensions");
      CudaCheck(cudaMemcpyAsync(workspace.host_status.data(),
                                device_status.get(), sizeof(DeviceStatus),
                                cudaMemcpyDeviceToHost),
                "read dual parameter status");
    });
    status = workspace.host_status[0];
    if (ApplyDeviceFailure(status, &solution)) {
      solution.timings.total_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - total_start)
              .count();
      return solution;
    }
    if (host_dual_scan_needed != 0) {
      PrepareDualStorage(&workspace, stage_count);
    }
    DualRelation *dual_relations =
        host_dual_scan_needed != 0 ? dual_tree.get() : nullptr;
    DualNodeValue *dual_leaf_values =
        host_dual_scan_needed != 0 ? dual_values.get() : nullptr;
    solution.timings.multiplier_ms += TimeGpu(workspace, [&] {
      if (host_dual_scan_needed != 0) {
        CudaCheck(cudaMemcpyAsync(
                      dual_tree.get(), workspace.host_dual_tree.data(),
                      workspace.host_dual_tree.size() * sizeof(DualRelation),
                      cudaMemcpyHostToDevice),
                  "upload compact dual relation tree");
        CudaCheck(cudaMemcpyAsync(
                      dual_values.get(), workspace.host_dual_values.data(),
                      workspace.host_dual_values.size() * sizeof(DualNodeValue),
                      cudaMemcpyHostToDevice),
                  "upload compact dual value tree");
      }
      BuildDualParameterRelationsKernel<<<stage_count, kThreads>>>(
          device_stages.get(), device_terminal.get(), dual_params.get(),
          stage_count, states.get(), controls.get(), state_offsets.get(),
          control_offsets.get(), multiplier_rank_tolerance,
          multiplier_consistency_tolerance, dual_relations,
          dual_scan_needed.get(), state_dual_params.get(), device_status.get());
      if (host_dual_scan_needed != 0) {
        for (std::size_t level = 0; level + 1 < stage_level_counts.size();
             ++level) {
          ReduceDualTreeLevelKernel<<<stage_level_counts[level + 1],
                                      kThreads>>>(
              dual_tree.get(), stage_level_offsets[level],
              stage_level_offsets[level + 1], stage_level_counts[level],
              stage_level_counts[level + 1], multiplier_rank_tolerance,
              multiplier_consistency_tolerance, dual_tree.get(),
              dual_scan_needed.get(), device_status.get());
        }
        const int root_offset = stage_level_offsets.back();
        SolveDualRootKernel<<<1, kThreads>>>(
            dual_tree.get() + root_offset, dual_values.get() + root_offset,
            dual_scan_needed.get(), device_status.get(),
            multiplier_rank_tolerance);
        for (int level = static_cast<int>(stage_level_counts.size()) - 2;
             level >= 0; --level) {
          ExpandDualTreeLevelKernel<<<stage_level_counts[level + 1],
                                      kThreads>>>(
              dual_tree.get(), stage_level_offsets[level],
              stage_level_offsets[level + 1], stage_level_counts[level],
              stage_level_counts[level + 1], multiplier_rank_tolerance,
              multiplier_consistency_tolerance, dual_values.get(),
              dual_values.get(), dual_scan_needed.get(), device_status.get());
        }
      }
      const int recovery_blocks = (stage_count + kThreads - 1) / kThreads;
      RecoverParameterizedDynamicsAndMixedKernel<<<recovery_blocks, kThreads>>>(
          dual_params.get(), dual_leaf_values, dynamics_offsets.get(),
          mixed_offsets.get(), stage_count, dynamics_multipliers.get(),
          mixed_multipliers.get());
      RecoverStateMultipliersFromParametersKernel<<<recovery_blocks,
                                                    kThreads>>>(
          state_dual_params.get(), dual_leaf_values,
          state_constraint_offsets.get(), stage_count, state_multipliers.get(),
          terminal_multiplier.get());
    });
  }
  solution.timings.multiplier_ms += TimeGpu(workspace, [&] {
    RecoverInitialMultiplierKernel<<<1, kThreads>>>(
        device_stages.get(), device_terminal.get(), stage_count, states.get(),
        controls.get(), dynamics_multipliers.get(), mixed_multipliers.get(),
        state_offsets.get(), control_offsets.get(), dynamics_offsets.get(),
        mixed_offsets.get(), state_constraint_offsets.get(),
        initial_multiplier.get(), state_multipliers.get(),
        terminal_multiplier.get());
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
  });

  solution.states.resize(node_count);
  solution.reduced_state_dimensions.resize(node_count);
  for (int i = 0; i < node_count; ++i) {
    const int n = host_state_dimensions[2 * i];
    solution.states[i].resize(n);
    for (int row = 0; row < n; ++row)
      solution.states[i][row] =
          host_states[workspace.host_state_offsets[i] + row];
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
          host_controls[workspace.host_control_offsets[i] + row];
    solution.reduced_control_dimensions[i] = host_control_dimensions[2 * i + 1];
    solution.dynamics_multipliers[i].resize(s.next_n);
    for (int row = 0; row < s.next_n; ++row)
      solution.dynamics_multipliers[i][row] =
          host_dynamics[workspace.host_dynamics_offsets[i] + row];
    solution.mixed_multipliers[i].resize(s.mixed);
    for (int row = 0; row < s.mixed; ++row)
      solution.mixed_multipliers[i][row] =
          host_mixed[workspace.host_mixed_offsets[i] + row];
    solution.state_multipliers[i].resize(s.state);
    for (int row = 0; row < s.state; ++row)
      solution.state_multipliers[i][row] =
          host_state_multipliers[workspace.host_state_constraint_offsets[i] +
                                 row];
  }
  solution.initial_multiplier.resize(problem.initial_state.size());
  for (std::size_t row = 0; row < problem.initial_state.size(); ++row)
    solution.initial_multiplier[row] = host_initial_multiplier[row];
  solution.terminal_state_multiplier.resize(terminal.state);
  for (int row = 0; row < terminal.state; ++row)
    solution.terminal_state_multiplier[row] = host_terminal_multiplier[row];
  solution.objective =
      ObjectiveFromCompact(problem, workspace.host_state_offsets.data(),
                           workspace.host_control_offsets.data(),
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
  const detail::CompactEntryCounts entry_counts =
      detail::CountCompactEntries(problem);
  impl_->storage.Reserve(
      options.device, stage_count, node_count, entry_counts.states,
      entry_counts.controls, entry_counts.mixed, entry_counts.state_constraints,
      entry_counts.problem_data, static_cast<int>(problem.initial_state.size()),
      static_cast<int>(problem.terminal_E.rows()));
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
    maps[index].linear[row * fb.state_dim + col] =
        fb.transition[row * fb.state_dim + col];
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
    output->linear[row * input.left_dim + col] =
        input.linear[row * input.left_dim + col];
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
      value += second.linear[row * second.left_dim + k] *
               first.linear[k * first.left_dim + col];
    }
    output->linear[row * first.left_dim + col] = value;
  }
  for (int row = threadIdx.x; row < second.right_dim; row += blockDim.x) {
    Scalar value = second.offset[row];
    for (int k = 0; k < first.right_dim; ++k) {
      value += second.linear[row * second.left_dim + k] * first.offset[k];
    }
    output->offset[row] = value;
  }
}

__global__ void ReduceAffineLeavesKernel(const AffineMap *leaves, int count,
                                         int parent_count, AffineMap *parents,
                                         DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  const int left = 2 * index;
  if (left + 1 >= count) {
    CopyAffineMapBlock(leaves[left], &parents[index]);
    return;
  }
  ComposeAffineMapsBlock(leaves[left], leaves[left + 1], &parents[index],
                         status, index);
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
  WarpSynchronize();
  CopyAffineMapBlock(parent_context, &tree[left]);
}

__global__ void
FinalizeAffinePrefixFromParentsKernel(AffineMap *leaves, int count,
                                      const AffineMap *parent_contexts,
                                      int parent_count, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  const int left = 2 * index;
  const AffineMap &parent = parent_contexts[index];
  __shared__ AffineMap composed;
  __shared__ Scalar composed_storage[kMaxAffineMapEntries];
  if (threadIdx.x == 0)
    BindAffineMapScratch(&composed, composed_storage);
  WarpSynchronize();
  if (left + 1 >= count) {
    if (!InvalidScanAffineMap(parent)) {
      ComposeAffineMapsBlock(parent, leaves[left], &composed, status, left);
      WarpSynchronize();
      CopyAffineMapBlock(composed, &leaves[left]);
    }
    return;
  }
  const int right = left + 1;
  if (!InvalidScanAffineMap(parent)) {
    ComposeAffineMapsBlock(parent, leaves[left], &composed, status, left);
    WarpSynchronize();
    CopyAffineMapBlock(composed, &leaves[left]);
    WarpSynchronize();
  }
  ComposeAffineMapsBlock(leaves[left], leaves[right], &composed, status, right);
  WarpSynchronize();
  CopyAffineMapBlock(composed, &leaves[right]);
}

__global__ void EvaluateReducedStatesKernel(const AffineMap *prefix,
                                            int stage_count,
                                            const StateParam *state_params,
                                            const Scalar *initial,
                                            const int *reduced_state_offsets,
                                            Scalar *reduced_states) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > stage_count)
    return;
  if (index == 0) {
    for (int col = 0; col < state_params[0].reduced_dim; ++col)
      reduced_states[reduced_state_offsets[0] + col] = initial[col];
    return;
  }
  const AffineMap &map = prefix[index - 1];
  for (int row = 0; row < map.right_dim; ++row) {
    Scalar value = map.offset[row];
    for (int col = 0; col < map.left_dim; ++col) {
      value += map.linear[row * map.left_dim + col] * initial[col];
    }
    reduced_states[reduced_state_offsets[index] + row] = value;
  }
}

__global__ void
ReconstructPrimalKernel(const StateParam *state_params,
                        const ControlParam *control_params,
                        const Feedback *feedback, const Scalar *reduced_states,
                        const int *reduced_state_offsets,
                        const int *state_offsets, const int *control_offsets,
                        int stage_count, Scalar *states, Scalar *controls) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > stage_count)
    return;
  const StateParam &state = state_params[index];
  const Scalar *z = reduced_states + reduced_state_offsets[index];
  for (int x = 0; x < state.physical_dim; ++x) {
    Scalar value = state.t[x];
    for (int col = 0; col < state.reduced_dim; ++col) {
      value += state.T[x * state.reduced_dim + col] * z[col];
    }
    states[state_offsets[index] + x] = value;
  }
  if (index == stage_count)
    return;
  const ControlParam &control = control_params[index];
  const Feedback &fb = feedback[index];
  Scalar v[kMaxControlDimension];
  for (int row = 0; row < control.reduced_dim; ++row) {
    Scalar value = fb.k[row];
    for (int col = 0; col < fb.state_dim; ++col) {
      value += fb.K[row * fb.state_dim + col] * z[col];
    }
    v[row] = value;
  }
  for (int u = 0; u < control.physical_dim; ++u) {
    Scalar value = control.y[u];
    for (int col = 0; col < control.state_dim; ++col) {
      value += control.Y[u * control.state_dim + col] * z[col];
    }
    for (int col = 0; col < control.reduced_dim; ++col) {
      value += control.Z[u * control.reduced_dim + col] * v[col];
    }
    controls[control_offsets[index] + u] = value;
  }
}

__global__ void BuildDualParametersKernel(
    const PackedStage *stages, const StateParam *state_params,
    const ValueElement *value_suffix, const Scalar *reduced_states,
    const Scalar *states, const Scalar *controls,
    const int *reduced_state_offsets, const int *state_offsets,
    const int *control_offsets, int stage_count, Scalar rank_tolerance,
    Scalar consistency_tolerance, DualParam *params, int *scan_needed,
    DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= stage_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  const PackedStage &stage = stages[index];
  const StateParam &next = state_params[index + 1];
  const ValueElement &next_value = value_suffix[index + 1];
  const int variables = stage.next_n + stage.mixed;
  const int rows = next.reduced_dim + stage.m;
  const int columns = variables + 1;
  __shared__ Scalar matrix[kMaxRrefRows * kMaxDualColumns];
  __shared__ Scalar constraint_scales[kMaxMixedConstraints];
  __shared__ Scalar residual_rhs[kMaxRrefRows];
  __shared__ Scalar
      upper[kMaxDualParameterDimension * kMaxDualParameterDimension];
  __shared__ Scalar rhs_projection[kMaxDualParameterDimension];
  __shared__ Scalar solution[kMaxDualParameterDimension];
  __shared__ int permutation[kMaxDualParameterDimension];
  __shared__ int rank;
  __shared__ int local_ok;
  __shared__ Scalar conditioned_rhs_scale;

  for (int entry = threadIdx.x; entry < rows * columns; entry += blockDim.x)
    matrix[entry] = Scalar{0};
  for (int constraint = threadIdx.x; constraint < stage.mixed;
       constraint += blockDim.x) {
    Scalar scale = Scalar{0};
    for (int state = 0; state < stage.n; ++state)
      scale = fmax(scale, DeviceAbs(stage.C[constraint * stage.n + state]));
    for (int control = 0; control < stage.m; ++control)
      scale = fmax(scale, DeviceAbs(stage.D[constraint * stage.m + control]));
    constraint_scales[constraint] = scale > Scalar{0} ? scale : Scalar{1};
  }
  WarpSynchronize();

  for (int linear = threadIdx.x; linear < next.reduced_dim * stage.next_n;
       linear += blockDim.x) {
    const int row = linear / stage.next_n;
    const int state = linear % stage.next_n;
    matrix[row * columns + state] = next.T[state * next.reduced_dim + row];
  }
  const Scalar *next_reduced_state =
      reduced_states + reduced_state_offsets[index + 1];
  for (int row = threadIdx.x; row < next.reduced_dim; row += blockDim.x) {
    Scalar costate = -next_value.eta[row];
    for (int col = 0; col < next.reduced_dim; ++col)
      costate -= next_value.J[row * next_value.left_dim + col] *
                 next_reduced_state[col];
    matrix[row * columns + variables] = costate;
  }
  for (int linear = threadIdx.x; linear < stage.m * stage.next_n;
       linear += blockDim.x) {
    const int control = linear / stage.next_n;
    const int state = linear % stage.next_n;
    matrix[(next.reduced_dim + control) * columns + state] =
        stage.B[state * stage.m + control];
  }
  for (int linear = threadIdx.x; linear < stage.m * stage.mixed;
       linear += blockDim.x) {
    const int control = linear / stage.mixed;
    const int constraint = linear % stage.mixed;
    matrix[(next.reduced_dim + control) * columns + stage.next_n + constraint] =
        -stage.D[constraint * stage.m + control] /
        constraint_scales[constraint];
  }
  const Scalar *state = states + state_offsets[index];
  const Scalar *control = controls + control_offsets[index];
  for (int row = threadIdx.x; row < stage.m; row += blockDim.x) {
    Scalar gradient = stage.r[row];
    for (int col = 0; col < stage.n; ++col)
      gradient += stage.M[col * stage.m + row] * state[col];
    for (int col = 0; col < stage.m; ++col)
      gradient += stage.R[row * stage.m + col] * control[col];
    matrix[(next.reduced_dim + row) * columns + variables] = gradient;
  }
  WarpSynchronize();
  if (threadIdx.x == 0) {
    conditioned_rhs_scale =
        ConditionedRhsScale(matrix, rows, columns, variables, rank_tolerance);
  }
  WarpSynchronize();
  SolveSystemOrthogonally(matrix, rows, columns, variables, rank_tolerance,
                          consistency_tolerance, conditioned_rhs_scale,
                          residual_rhs, upper, rhs_projection, solution,
                          permutation, &rank, &local_ok);
  if (threadIdx.x == 0) {
    if (!local_ok) {
      SetFailure(status, kDeviceNumericalFailure, index, 24);
    } else {
      DualParam &out = params[index];
      out.state_dim = stage.next_n;
      out.mixed_dim = stage.mixed;
      out.physical_dim = variables;
      out.free_dim = variables - rank;
      for (int free = 0; free < out.free_dim; ++free)
        out.free_columns[free] = permutation[rank + free];
      if (out.free_dim > 0)
        atomicExch(scan_needed, 1);
      for (int variable = 0; variable < variables; ++variable) {
        const Scalar scale = variable < stage.next_n
                                 ? Scalar{1}
                                 : constraint_scales[variable - stage.next_n];
        out.offset[variable] = solution[variable] / scale;
        for (int free = 0; free < out.free_dim; ++free)
          out.basis[variable * out.free_dim + free] = Scalar{0};
      }
      for (int free = 0; free < out.free_dim; ++free) {
        Scalar pivoted_basis[kMaxDualParameterDimension]{};
        pivoted_basis[rank + free] = Scalar{1};
        for (int reverse = 0; reverse < rank; ++reverse) {
          const int row = rank - 1 - reverse;
          Scalar value = -upper[row * variables + rank + free];
          for (int col = row + 1; col < rank; ++col) {
            value -= upper[row * variables + col] * pivoted_basis[col];
          }
          pivoted_basis[row] = value / upper[row * variables + row];
        }
        for (int position = 0; position < variables; ++position) {
          const int variable = permutation[position];
          const Scalar scale = variable < stage.next_n
                                   ? Scalar{1}
                                   : constraint_scales[variable - stage.next_n];
          out.basis[variable * out.free_dim + free] =
              pivoted_basis[position] / scale;
        }
      }
    }
  }
  WarpSynchronize();
}

__global__ void BuildDualParameterRelationsKernel(
    const PackedStage *stages, const PackedTerminal *terminal_ptr,
    const DualParam *params, int stage_count, const Scalar *states,
    const Scalar *controls, const int *state_offsets,
    const int *control_offsets, Scalar rank_tolerance,
    Scalar consistency_tolerance, DualRelation *relations,
    const int *scan_needed, StateDualParam *state_params,
    DeviceStatus *status) {
  const int relation_index = blockIdx.x;
  if (relation_index >= stage_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  const int node = relation_index + 1;
  const bool is_terminal = node == stage_count;
  const PackedTerminal &terminal = *terminal_ptr;
  const DualParam &left = params[node - 1];
  const DualParam *right = is_terminal ? nullptr : &params[node];
  const int state_dim = is_terminal ? terminal.n : stages[node].n;
  const int state_constraints =
      is_terminal ? terminal.state : stages[node].state;
  const int right_dim = is_terminal ? 0 : right->free_dim;
  const int rows = state_dim;
  const int columns = state_constraints + left.free_dim + right_dim + 1;
  __shared__ Scalar matrix[kMaxRrefRows * kMaxDualColumns];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ Scalar constraint_scales[kMaxStateConstraints];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;

  for (int entry = threadIdx.x; entry < rows * columns; entry += blockDim.x)
    matrix[entry] = Scalar{0};
  for (int constraint = threadIdx.x; constraint < state_constraints;
       constraint += blockDim.x) {
    Scalar scale = Scalar{0};
    for (int state = 0; state < state_dim; ++state) {
      const Scalar value =
          is_terminal ? terminal.E[constraint * terminal.n + state]
                      : stages[node].E[constraint * stages[node].n + state];
      scale = fmax(scale, DeviceAbs(value));
    }
    constraint_scales[constraint] = scale > Scalar{0} ? scale : Scalar{1};
  }
  WarpSynchronize();
  for (int linear = threadIdx.x; linear < rows * state_constraints;
       linear += blockDim.x) {
    const int state = linear / state_constraints;
    const int constraint = linear % state_constraints;
    const Scalar value =
        is_terminal ? terminal.E[constraint * terminal.n + state]
                    : stages[node].E[constraint * stages[node].n + state];
    matrix[state * columns + constraint] =
        value / constraint_scales[constraint];
  }
  for (int linear = threadIdx.x; linear < state_dim * left.free_dim;
       linear += blockDim.x) {
    const int state = linear / left.free_dim;
    const int free = linear % left.free_dim;
    matrix[state * columns + state_constraints + free] =
        left.basis[state * left.free_dim + free];
  }
  if (!is_terminal) {
    const PackedStage &stage = stages[node];
    for (int linear = threadIdx.x; linear < state_dim * right_dim;
         linear += blockDim.x) {
      const int state = linear / right_dim;
      const int free = linear % right_dim;
      Scalar value = Scalar{0};
      for (int next_state = 0; next_state < stage.next_n; ++next_state) {
        value -= stage.A[next_state * stage.n + state] *
                 right->basis[next_state * right_dim + free];
      }
      for (int constraint = 0; constraint < stage.mixed; ++constraint) {
        value += stage.C[constraint * stage.n + state] *
                 right->basis[(stage.next_n + constraint) * right_dim + free];
      }
      matrix[state * columns + state_constraints + left.free_dim + free] =
          value;
    }
  }
  const Scalar *state = states + state_offsets[node];
  for (int row = threadIdx.x; row < state_dim; row += blockDim.x) {
    Scalar rhs = -left.offset[row];
    if (is_terminal) {
      rhs -= terminal.q[row];
      for (int col = 0; col < terminal.n; ++col)
        rhs -= terminal.Q[row * terminal.n + col] * state[col];
    } else {
      const PackedStage &stage = stages[node];
      const Scalar *control = controls + control_offsets[node];
      rhs -= stage.q[row];
      for (int col = 0; col < stage.n; ++col)
        rhs -= stage.Q[row * stage.n + col] * state[col];
      for (int col = 0; col < stage.m; ++col)
        rhs -= stage.M[row * stage.m + col] * control[col];
      for (int next_state = 0; next_state < stage.next_n; ++next_state)
        rhs += stage.A[next_state * stage.n + row] * right->offset[next_state];
      for (int constraint = 0; constraint < stage.mixed; ++constraint) {
        rhs -= stage.C[constraint * stage.n + row] *
               right->offset[stage.next_n + constraint];
      }
    }
    matrix[row * columns + columns - 1] = rhs;
  }
  WarpSynchronize();
  RrefBlock(matrix, rows, columns, columns - 1, rank_tolerance, pivot_columns,
            pivot_rows, &rank, &best_row, factors);
  if (threadIdx.x == 0) {
    // Any zero-left/nonzero-rhs row is retained in the residual relation and
    // checked by the global contraction.  Rejecting it locally is incorrect:
    // adjacent free dual coordinates can still satisfy that relation.
    StateDualParam &out = state_params[relation_index];
    out.constraint_dim = state_constraints;
    out.left_dim = left.free_dim;
    out.right_dim = right_dim;
    for (int constraint = 0; constraint < state_constraints; ++constraint) {
      out.offset[constraint] = Scalar{0};
      for (int free = 0; free < left.free_dim; ++free)
        out.left[constraint * left.free_dim + free] = Scalar{0};
      for (int free = 0; free < right_dim; ++free)
        out.right[constraint * right_dim + free] = Scalar{0};
    }
    for (int pivot = 0; pivot < rank; ++pivot) {
      const int constraint = pivot_columns[pivot];
      if (constraint >= state_constraints)
        break;
      const Scalar inverse_scale = Scalar{1} / constraint_scales[constraint];
      out.offset[constraint] =
          matrix[pivot * columns + columns - 1] * inverse_scale;
      for (int free = 0; free < left.free_dim; ++free) {
        out.left[constraint * left.free_dim + free] =
            -matrix[pivot * columns + state_constraints + free] * inverse_scale;
      }
      for (int free = 0; free < right_dim; ++free) {
        out.right[constraint * right_dim + free] =
            -matrix[pivot * columns + state_constraints + left.free_dim +
                    free] *
            inverse_scale;
      }
    }
  }
  WarpSynchronize();
  if (*scan_needed != 0) {
    ExtractResidualRelation(matrix, columns, rank, pivot_columns,
                            state_constraints, left.free_dim, right_dim,
                            &relations[relation_index]);
  }
}

__global__ void RecoverParameterizedDynamicsAndMixedKernel(
    const DualParam *params, const DualNodeValue *leaf_values,
    const int *dynamics_offsets, const int *mixed_offsets, int stage_count,
    Scalar *dynamics_multipliers, Scalar *mixed_multipliers) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= stage_count)
    return;
  const DualParam &param = params[index];
  for (int row = 0; row < param.physical_dim; ++row) {
    Scalar result = param.offset[row];
    for (int free = 0; free < param.free_dim; ++free) {
      result += param.basis[row * param.free_dim + free] *
                leaf_values[index].left[free];
    }
    if (row < param.state_dim) {
      dynamics_multipliers[dynamics_offsets[index] + row] = result;
    } else {
      mixed_multipliers[mixed_offsets[index] + row - param.state_dim] = result;
    }
  }
}

__global__ void RecoverStateMultipliersFromParametersKernel(
    const StateDualParam *params, const DualNodeValue *leaf_values,
    const int *state_constraint_offsets, int stage_count,
    Scalar *state_multipliers, Scalar *terminal_multiplier) {
  const int relation_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (relation_index >= stage_count)
    return;
  const int node = relation_index + 1;
  const StateDualParam &param = params[relation_index];
  for (int constraint = 0; constraint < param.constraint_dim; ++constraint) {
    Scalar multiplier = param.offset[constraint];
    for (int free = 0; free < param.left_dim; ++free) {
      multiplier += param.left[constraint * param.left_dim + free] *
                    leaf_values[relation_index].left[free];
    }
    for (int free = 0; free < param.right_dim; ++free) {
      multiplier += param.right[constraint * param.right_dim + free] *
                    leaf_values[relation_index].right[free];
    }
    if (node == stage_count) {
      terminal_multiplier[constraint] = multiplier;
    } else {
      state_multipliers[state_constraint_offsets[node] + constraint] =
          multiplier;
    }
  }
}

__global__ void RecoverInitialMultiplierKernel(
    const PackedStage *stages, const PackedTerminal *terminal_ptr,
    int stage_count, const Scalar *states, const Scalar *controls,
    const Scalar *dynamics_multipliers, const Scalar *mixed_multipliers,
    const int *state_offsets, const int *control_offsets,
    const int *dynamics_offsets, const int *mixed_offsets,
    const int *state_constraint_offsets, Scalar *initial_multiplier,
    Scalar *state_multipliers, Scalar *terminal_multiplier) {
  if (blockIdx.x != 0)
    return;
  const PackedTerminal &terminal = *terminal_ptr;
  if (stage_count == 0) {
    for (int row = threadIdx.x; row < terminal.state; row += blockDim.x)
      terminal_multiplier[row] = Scalar{0};
    for (int row = threadIdx.x; row < terminal.n; row += blockDim.x) {
      Scalar value = -terminal.q[row];
      for (int col = 0; col < terminal.n; ++col)
        value -=
            terminal.Q[row * terminal.n + col] * states[state_offsets[0] + col];
      initial_multiplier[row] = value;
    }
    return;
  }
  const PackedStage &stage = stages[0];
  for (int row = threadIdx.x; row < stage.state; row += blockDim.x)
    state_multipliers[state_constraint_offsets[0] + row] = Scalar{0};
  for (int row = threadIdx.x; row < stage.n; row += blockDim.x) {
    Scalar value = -stage.q[row];
    for (int col = 0; col < stage.n; ++col)
      value -= stage.Q[row * stage.n + col] * states[state_offsets[0] + col];
    for (int col = 0; col < stage.m; ++col)
      value -=
          stage.M[row * stage.m + col] * controls[control_offsets[0] + col];
    for (int col = 0; col < stage.next_n; ++col)
      value += stage.A[col * stage.n + row] *
               dynamics_multipliers[dynamics_offsets[0] + col];
    for (int constraint = 0; constraint < stage.mixed; ++constraint) {
      value -= stage.C[constraint * stage.n + row] *
               mixed_multipliers[mixed_offsets[0] + constraint];
    }
    initial_multiplier[row] = value;
  }
}

__global__ void
ReduceDualTreeLevelKernel(const DualRelation *tree, int child_offset,
                          int parent_offset, int child_count, int parent_count,
                          Scalar rank_tolerance, Scalar consistency_tolerance,
                          DualRelation *mutable_tree, const int *scan_needed,
                          DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count || *scan_needed == 0)
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

__global__ void SolveDualRootKernel(const DualRelation *relation,
                                    DualNodeValue *value,
                                    const int *scan_needed,
                                    DeviceStatus *status, Scalar tolerance) {
  if (blockIdx.x != 0 || *scan_needed == 0)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  if (relation->right_dim != 0) {
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, 0, 13);
    return;
  }
  const int variables = relation->left_dim;
  const int columns = variables + 1;
  __shared__ Scalar matrix[kMaxRrefRows * (kMaxDualParameterDimension + 1)];
  __shared__ Scalar residual_rhs[kMaxRrefRows];
  __shared__ Scalar
      upper[kMaxDualParameterDimension * kMaxDualParameterDimension];
  __shared__ Scalar rhs_projection[kMaxDualParameterDimension];
  __shared__ Scalar solution[kMaxDualParameterDimension];
  __shared__ int permutation[kMaxDualParameterDimension];
  __shared__ int rank;
  __shared__ int local_ok;
  __shared__ Scalar rhs_scale;
  for (int linear = threadIdx.x; linear < relation->rows * variables;
       linear += blockDim.x) {
    const int row = linear / variables;
    const int col = linear % variables;
    matrix[row * columns + col] =
        relation->left[row * relation->left_dim + col];
  }
  for (int row = threadIdx.x; row < relation->rows; row += blockDim.x)
    matrix[row * columns + variables] = relation->rhs[row];
  WarpSynchronize();
  if (threadIdx.x == 0)
    rhs_scale = ConditionedRhsScale(matrix, relation->rows, columns, variables,
                                    tolerance);
  WarpSynchronize();
  SolveSystemOrthogonally(matrix, relation->rows, columns, variables, tolerance,
                          tolerance, rhs_scale, residual_rhs, upper,
                          rhs_projection, solution, permutation, &rank,
                          &local_ok);
  if (!local_ok) {
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, 0, 14);
    return;
  }
  if (threadIdx.x == 0) {
    value->left_dim = variables;
    value->right_dim = 0;
  }
  for (int col = threadIdx.x; col < variables; col += blockDim.x)
    value->left[col] = solution[col];
}

__global__ void ExpandDualTreeLevelKernel(
    const DualRelation *tree, int child_offset, int parent_offset,
    int child_count, int parent_count, Scalar rank_tolerance,
    Scalar consistency_tolerance, const DualNodeValue *parent_values,
    DualNodeValue *values, const int *scan_needed, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count || *scan_needed == 0)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(status, &block_enabled))
    return;
  if (2 * index + 1 >= child_count) {
    const DualNodeValue &parent = parent_values[parent_offset + index];
    DualNodeValue &child = values[child_offset + 2 * index];
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
  const DualRelation &left = tree[child_offset + 2 * index];
  const DualRelation &right = tree[child_offset + 2 * index + 1];
  const DualNodeValue &parent = parent_values[parent_offset + index];
  if (left.left_dim != parent.left_dim || right.right_dim != parent.right_dim ||
      left.right_dim != right.left_dim) {
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, index, 15);
    return;
  }
  const int shared = left.right_dim;
  const int rows = left.rows + right.rows;
  const int columns = shared + 1;
  __shared__ Scalar matrix[kMaxRrefRows * (kMaxDualParameterDimension + 1)];
  __shared__ Scalar residual_rhs[kMaxRrefRows];
  __shared__ Scalar
      upper[kMaxDualParameterDimension * kMaxDualParameterDimension];
  __shared__ Scalar rhs_projection[kMaxDualParameterDimension];
  __shared__ Scalar shared_solution[kMaxDualParameterDimension];
  __shared__ int permutation[kMaxDualParameterDimension];
  __shared__ int rank;
  __shared__ int local_ok;
  __shared__ Scalar conditioned_rhs_scale;
  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = Scalar{0};
  WarpSynchronize();
  for (int linear = threadIdx.x; linear < left.rows * shared;
       linear += blockDim.x) {
    const int row = linear / shared;
    const int col = linear % shared;
    matrix[row * columns + col] = left.right[row * left.right_dim + col];
  }
  for (int row = threadIdx.x; row < left.rows; row += blockDim.x) {
    Scalar rhs = left.rhs[row];
    for (int col = 0; col < left.left_dim; ++col) {
      rhs -= left.left[row * left.left_dim + col] * parent.left[col];
    }
    matrix[row * columns + shared] = rhs;
  }
  for (int linear = threadIdx.x; linear < right.rows * shared;
       linear += blockDim.x) {
    const int row = linear / shared;
    const int col = linear % shared;
    matrix[(left.rows + row) * columns + col] =
        right.left[row * right.left_dim + col];
  }
  for (int row = threadIdx.x; row < right.rows; row += blockDim.x) {
    Scalar rhs = right.rhs[row];
    for (int col = 0; col < right.right_dim; ++col) {
      rhs -= right.right[row * right.right_dim + col] * parent.right[col];
    }
    matrix[(left.rows + row) * columns + shared] = rhs;
  }
  WarpSynchronize();
  if (threadIdx.x == 0) {
    conditioned_rhs_scale =
        ConditionedRhsScale(matrix, rows, columns, shared, rank_tolerance);
  }
  WarpSynchronize();
  SolveSystemOrthogonally(matrix, rows, columns, shared, rank_tolerance,
                          consistency_tolerance, conditioned_rhs_scale,
                          residual_rhs, upper, rhs_projection, shared_solution,
                          permutation, &rank, &local_ok);
  if (threadIdx.x == 0 && !local_ok)
    SetFailure(status, kDeviceNumericalFailure, index, 16);
  WarpSynchronize();
  if (!local_ok)
    return;
  if (threadIdx.x == 0) {
    DualNodeValue &left_value = values[child_offset + 2 * index];
    DualNodeValue &right_value = values[child_offset + 2 * index + 1];
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
      elements[index].J[row * terminal.n + col] =
          terminal.Q[row * terminal.n + col];
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
      out.A[row * s.n + col] = s.A[row * s.n + col];
    }
    for (int row = threadIdx.x; row < s.next_n; row += blockDim.x)
      out.b[row] = s.c[row];
    for (int linear = threadIdx.x; linear < s.n * s.n; linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      out.J[row * s.n + col] = s.Q[row * s.n + col];
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
    augmented[linear] = col < s.m ? s.R[row * s.m + col]
                                  : (col - s.m == row ? Scalar{1} : Scalar{0});
  }
  for (int linear = threadIdx.x; linear < s.m * s.m; linear += blockDim.x) {
    const int row = linear / s.m;
    const int col = linear % s.m;
    cholesky[linear] =
        Scalar{0.5} * (s.R[row * s.m + col] + s.R[col * s.m + row]);
  }
  WarpSynchronize();
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
  WarpSynchronize();
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
    Scalar value = s.Q[a * s.n + b];
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value -= s.M[a * s.m + u] * augmented[u * columns + s.m + v] *
                 s.M[b * s.m + v];
      }
    }
    out.J[a * s.n + b] = value;
  }
  for (int a = threadIdx.x; a < s.n; a += blockDim.x) {
    Scalar value = s.q[a];
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value -= s.M[a * s.m + u] * augmented[u * columns + s.m + v] * s.r[v];
      }
    }
    out.eta[a] = value;
  }
  for (int linear = threadIdx.x; linear < s.next_n * s.n;
       linear += blockDim.x) {
    const int row = linear / s.n;
    const int col = linear % s.n;
    Scalar value = s.A[row * s.n + col];
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value -= s.B[row * s.m + u] * augmented[u * columns + s.m + v] *
                 s.M[col * s.m + v];
      }
    }
    out.A[row * s.n + col] = value;
  }
  for (int row = threadIdx.x; row < s.next_n; row += blockDim.x) {
    Scalar value = s.c[row];
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value -= s.B[row * s.m + u] * augmented[u * columns + s.m + v] * s.r[v];
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
        value += s.B[a * s.m + u] * augmented[u * columns + s.m + v] *
                 s.B[b * s.m + v];
      }
    }
    out.C[a * s.next_n + b] = value;
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
      output->J[row * left + col] = first.J[row * first.left_dim + col];
    }
    for (int row = threadIdx.x; row < left; row += blockDim.x)
      output->eta[row] = first.eta[row];
    for (int linear = threadIdx.x; linear < right * right;
         linear += blockDim.x) {
      const int row = linear / right;
      const int col = linear % right;
      output->C[row * right + col] = second.C[row * second.right_dim + col];
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
        value += first.C[row * first.right_dim + k] *
                 second.J[k * second.left_dim + col];
      }
    } else if (col < shared + left) {
      value = first.A[row * first.left_dim + col - shared];
    } else if (col == shared + left) {
      value = first.b[row];
      for (int k = 0; k < shared; ++k) {
        value -= first.C[row * first.right_dim + k] * second.eta[k];
      }
    } else {
      value = first.C[row * first.right_dim + col - shared - left - 1];
    }
    augmented[linear] = value;
  }
  WarpSynchronize();
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
      value += second.A[row * second.left_dim + k] *
               augmented[k * columns + shared + col];
    }
    output->A[row * left + col] = value;
  }
  for (int row = threadIdx.x; row < right; row += blockDim.x) {
    Scalar value = second.b[row];
    for (int k = 0; k < shared; ++k) {
      value += second.A[row * second.left_dim + k] *
               augmented[k * columns + shared + left];
    }
    output->b[row] = value;
  }
  for (int linear = threadIdx.x; linear < right * right; linear += blockDim.x) {
    const int row = linear / right;
    const int col = linear % right;
    Scalar value = second.C[row * second.right_dim + col];
    for (int p = 0; p < shared; ++p) {
      for (int q = 0; q < shared; ++q) {
        value += second.A[row * second.left_dim + p] *
                 augmented[p * columns + shared + left + 1 + q] *
                 second.A[col * second.left_dim + q];
      }
    }
    output->C[row * right + col] = value;
  }
  for (int linear = threadIdx.x; linear < left * left; linear += blockDim.x) {
    const int row = linear / left;
    const int col = linear % left;
    Scalar value = first.J[row * first.left_dim + col];
    for (int p = 0; p < shared; ++p) {
      for (int q = 0; q < shared; ++q) {
        value += first.A[p * first.left_dim + row] *
                 second.J[p * second.left_dim + q] *
                 augmented[q * columns + shared + col];
      }
    }
    output->J[row * left + col] = value;
  }
  for (int row = threadIdx.x; row < left; row += blockDim.x) {
    Scalar value = first.eta[row];
    for (int p = 0; p < shared; ++p) {
      Scalar dual = second.eta[p];
      for (int q = 0; q < shared; ++q) {
        dual += second.J[p * second.left_dim + q] *
                augmented[q * columns + shared + left];
      }
      value += first.A[p * first.left_dim + row] * dual;
    }
    output->eta[row] = value;
  }
  WarpSynchronize();
  for (int linear = threadIdx.x; linear < left * left; linear += blockDim.x) {
    const int row = linear / left;
    const int col = linear % left;
    if (row < col) {
      const Scalar value = Scalar{0.5} * (output->J[row * left + col] +
                                          output->J[col * left + row]);
      output->J[row * left + col] = value;
      output->J[col * left + row] = value;
    }
  }
  for (int linear = threadIdx.x; linear < right * right; linear += blockDim.x) {
    const int row = linear / right;
    const int col = linear % right;
    if (row < col) {
      const Scalar value = Scalar{0.5} * (output->C[row * right + col] +
                                          output->C[col * right + row]);
      output->C[row * right + col] = value;
      output->C[col * right + row] = value;
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
    output->A[row * input.left_dim + col] = input.A[row * input.left_dim + col];
  }
  for (int row = threadIdx.x; row < input.right_dim; row += blockDim.x)
    output->b[row] = input.b[row];
  for (int linear = threadIdx.x; linear < input.right_dim * input.right_dim;
       linear += blockDim.x) {
    const int row = linear / input.right_dim;
    const int col = linear % input.right_dim;
    output->C[row * input.right_dim + col] =
        input.C[row * input.right_dim + col];
  }
  for (int row = threadIdx.x; row < input.left_dim; row += blockDim.x)
    output->eta[row] = input.eta[row];
  for (int linear = threadIdx.x; linear < input.left_dim * input.left_dim;
       linear += blockDim.x) {
    const int row = linear / input.left_dim;
    const int col = linear % input.left_dim;
    output->J[row * input.left_dim + col] = input.J[row * input.left_dim + col];
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

__global__ void ReduceValueLeavesKernel(const ValueElement *leaves, int count,
                                        int parent_count, Scalar tolerance,
                                        int *parallel_ok,
                                        ValueElement *parents) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(parallel_ok, &block_enabled))
    return;
  const int left = 2 * index;
  if (left + 1 >= count) {
    CopyValueElementBlock(leaves[left], &parents[index]);
    return;
  }
  __shared__ Scalar
      augmented[kMaxStateDimension * (3 * kMaxStateDimension + 1)];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  ComposeValueElementsBlock(leaves[left], leaves[left + 1], tolerance,
                            &parents[index], parallel_ok, augmented, factors,
                            pivot_columns, pivot_rows, &rank, &best_row);
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
  WarpSynchronize();
  CopyValueElementBlock(parent_context, &tree[right]);
}

__global__ void FinalizeValueSuffixFromParentsKernel(
    ValueElement *leaves, int count, const ValueElement *parent_contexts,
    int parent_count, Scalar tolerance, int *parallel_ok) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  __shared__ int block_enabled;
  if (!BlockEnabled(parallel_ok, &block_enabled))
    return;
  const int left = 2 * index;
  const ValueElement &parent = parent_contexts[index];
  __shared__ Scalar
      augmented[kMaxStateDimension * (3 * kMaxStateDimension + 1)];
  __shared__ Scalar factors[kMaxRrefRows];
  __shared__ ValueElement composed;
  __shared__ Scalar composed_storage[kMaxValueElementEntries];
  __shared__ int pivot_columns[kMaxRrefRows];
  __shared__ int pivot_rows[kMaxRrefRows];
  __shared__ int rank;
  __shared__ int best_row;
  if (threadIdx.x == 0)
    BindValueElementScratch(&composed, composed_storage);
  WarpSynchronize();
  if (left + 1 >= count) {
    if (!InvalidScanValueElement(parent)) {
      ComposeScanValueBlock(leaves[left], parent, tolerance, &composed,
                            parallel_ok, augmented, factors, pivot_columns,
                            pivot_rows, &rank, &best_row);
      WarpSynchronize();
      CopyValueElementBlock(composed, &leaves[left]);
    }
    return;
  }
  const int right = left + 1;
  if (!InvalidScanValueElement(parent)) {
    ComposeScanValueBlock(leaves[right], parent, tolerance, &composed,
                          parallel_ok, augmented, factors, pivot_columns,
                          pivot_rows, &rank, &best_row);
    WarpSynchronize();
    CopyValueElementBlock(composed, &leaves[right]);
    WarpSynchronize();
  }
  ComposeValueElementsBlock(leaves[left], leaves[right], tolerance, &composed,
                            parallel_ok, augmented, factors, pivot_columns,
                            pivot_rows, &rank, &best_row);
  WarpSynchronize();
  CopyValueElementBlock(composed, &leaves[left]);
}

__device__ void BuildFeedbackSystem(const ReducedStage &s,
                                    const ValueElement &next, Scalar *augmented,
                                    int columns) {
  for (int linear = threadIdx.x; linear < s.m * columns; linear += blockDim.x) {
    const int row = linear / columns;
    const int col = linear % columns;
    Scalar value = Scalar{0};
    if (col < s.m) {
      value = s.R[row * s.m + col];
      for (int a = 0; a < s.next_n; ++a) {
        for (int b = 0; b < s.next_n; ++b) {
          value += s.B[a * s.m + row] * next.J[a * next.left_dim + b] *
                   s.B[b * s.m + col];
        }
      }
    } else if (col < s.m + s.n) {
      const int x = col - s.m;
      value = -s.M[x * s.m + row];
      for (int a = 0; a < s.next_n; ++a) {
        for (int b = 0; b < s.next_n; ++b) {
          value -= s.B[a * s.m + row] * next.J[a * next.left_dim + b] *
                   s.A[b * s.n + x];
        }
      }
    } else {
      value = -s.r[row];
      for (int a = 0; a < s.next_n; ++a) {
        Scalar future = next.eta[a];
        for (int b = 0; b < s.next_n; ++b) {
          future += next.J[a * next.left_dim + b] * s.c[b];
        }
        value -= s.B[a * s.m + row] * future;
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
    feedback->K[row * s.n + col] = augmented[row * columns + s.m + col];
  }
  for (int row = threadIdx.x; row < s.m; row += blockDim.x) {
    feedback->k[row] = augmented[row * columns + s.m + s.n];
  }
  WarpSynchronize();
  for (int linear = threadIdx.x; linear < s.next_n * s.n;
       linear += blockDim.x) {
    const int row = linear / s.n;
    const int col = linear % s.n;
    Scalar value = s.A[row * s.n + col];
    for (int u = 0; u < s.m; ++u) {
      value += s.B[row * s.m + u] * feedback->K[u * s.n + col];
    }
    feedback->transition[row * s.n + col] = value;
  }
  for (int row = threadIdx.x; row < s.next_n; row += blockDim.x) {
    Scalar value = s.c[row];
    for (int u = 0; u < s.m; ++u) {
      value += s.B[row * s.m + u] * feedback->k[u];
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
      out.transition[row * s.n + col] = s.A[row * s.n + col];
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
  WarpSynchronize();
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
        fb.transition[row * s.n + col] = s.A[row * s.n + col];
      }
      for (int row = threadIdx.x; row < s.next_n; row += blockDim.x)
        fb.offset[row] = s.c[row];
    } else {
      const int columns = s.m + s.n + 1;
      BuildFeedbackSystem(s, next, augmented, columns);
      WarpSynchronize();
      RrefBlock(augmented, s.m, columns, s.m, tolerance, pivot_columns,
                pivot_rows, &rank, &best_row, factors);
      if (rank != s.m) {
        if (threadIdx.x == 0)
          SetFailure(status, kDeviceNumericalFailure, index, 10);
        return;
      }
      ExtractFeedback(s, augmented, columns, &fb);
    }
    WarpSynchronize();

    ValueElement &current = suffix[index];
    if (threadIdx.x == 0) {
      current.left_dim = s.n;
      current.right_dim = 0;
    }
    for (int linear = threadIdx.x; linear < s.n * s.n; linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      Scalar value = s.Q[row * s.n + col];
      for (int a = 0; a < s.next_n; ++a) {
        for (int b = 0; b < s.next_n; ++b) {
          value += s.A[a * s.n + row] * next.J[a * next.left_dim + b] *
                   s.A[b * s.n + col];
        }
      }
      for (int u = 0; u < s.m; ++u) {
        Scalar cross = s.M[row * s.m + u];
        for (int a = 0; a < s.next_n; ++a) {
          for (int b = 0; b < s.next_n; ++b) {
            cross += s.A[a * s.n + row] * next.J[a * next.left_dim + b] *
                     s.B[b * s.m + u];
          }
        }
        value += cross * fb.K[u * s.n + col];
      }
      current.J[row * current.left_dim + col] = value;
    }
    for (int row = threadIdx.x; row < s.n; row += blockDim.x) {
      Scalar value = s.q[row];
      for (int a = 0; a < s.next_n; ++a) {
        Scalar future = next.eta[a];
        for (int b = 0; b < s.next_n; ++b) {
          future += next.J[a * next.left_dim + b] * s.c[b];
        }
        value += s.A[a * s.n + row] * future;
      }
      for (int u = 0; u < s.m; ++u) {
        Scalar cross = s.M[row * s.m + u];
        for (int a = 0; a < s.next_n; ++a) {
          for (int b = 0; b < s.next_n; ++b) {
            cross += s.A[a * s.n + row] * next.J[a * next.left_dim + b] *
                     s.B[b * s.m + u];
          }
        }
        value += cross * fb.k[u];
      }
      current.eta[row] = value;
    }
    WarpSynchronize();
    for (int linear = threadIdx.x; linear < s.n * s.n; linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      if (row < col) {
        const Scalar value =
            Scalar{0.5} * (current.J[row * current.left_dim + col] +
                           current.J[col * current.left_dim + row]);
        current.J[row * current.left_dim + col] = value;
        current.J[col * current.left_dim + row] = value;
      }
    }
    WarpSynchronize();
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
  WarpSynchronize();
  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = Scalar{0};
  WarpSynchronize();

  // Original mixed equalities after x = T*z + t.
  for (int linear = threadIdx.x; linear < s.mixed * s.m; linear += blockDim.x) {
    const int row = linear / s.m;
    const int col = linear % s.m;
    matrix[row * columns + col] = s.D[row * s.m + col];
  }
  for (int linear = threadIdx.x; linear < s.mixed * current.reduced_dim;
       linear += blockDim.x) {
    const int row = linear / current.reduced_dim;
    const int z = linear % current.reduced_dim;
    Scalar value = Scalar{0};
    for (int x = 0; x < s.n; ++x) {
      value += s.C[row * s.n + x] * current.T[x * current.reduced_dim + z];
    }
    matrix[row * columns + s.m + z] = value;
  }
  for (int row = threadIdx.x; row < s.mixed; row += blockDim.x) {
    Scalar value = -s.d[row];
    for (int x = 0; x < s.n; ++x) {
      value -= s.C[row * s.n + x] * current.t[x];
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
      value += next_relation.left[row * next_relation.left_dim + xp] *
               s.B[xp * s.m + u];
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
        at += s.A[xp * s.n + x] * current.T[x * current.reduced_dim + z];
      }
      value += next_relation.left[row * next_relation.left_dim + xp] * at;
    }
    matrix[(s.mixed + row) * columns + s.m + z] = value;
  }
  for (int row = threadIdx.x; row < next_relation.rows; row += blockDim.x) {
    Scalar value = next_relation.rhs[row];
    for (int xp = 0; xp < s.next_n; ++xp) {
      Scalar affine = s.c[xp];
      for (int x = 0; x < s.n; ++x) {
        affine += s.A[xp * s.n + x] * current.t[x];
      }
      value -= next_relation.left[row * next_relation.left_dim + xp] * affine;
    }
    matrix[(s.mixed + row) * columns + columns - 1] = value;
  }
  WarpSynchronize();

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
        scale = fmax(scale, DeviceAbs(s.C[row * s.n + x]));
      for (int u = 0; u < s.m; ++u)
        scale = fmax(scale, DeviceAbs(s.D[row * s.m + u]));
      scale = fmax(scale, DeviceAbs(s.d[row]));
    } else {
      for (int col = 0; col < columns; ++col)
        scale = fmax(scale, DeviceAbs(matrix[row * columns + col]));
    }
    factors[row] = scale;
  }
  WarpSynchronize();
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
  WarpSynchronize();
  for (int row = threadIdx.x; row < rows; row += blockDim.x) {
    Scalar scale = Scalar{0};
    for (int col = 0; col < columns; ++col)
      scale = fmax(scale, DeviceAbs(matrix[row * columns + col]));
    factors[row] = scale;
  }
  WarpSynchronize();
  for (int linear = threadIdx.x; linear < rows * columns;
       linear += blockDim.x) {
    if (factors[linear / columns] <= rank_tolerance)
      matrix[linear] = Scalar{0};
  }
  WarpSynchronize();

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
  WarpSynchronize();
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
    initialized_control.Z[u * initialized_control.reduced_dim + v] = Scalar{0};
  }
  WarpSynchronize();
  if (threadIdx.x == 0) {
    for (int p = 0; p < control_rank; ++p) {
      const int u = pivot_columns[p];
      initialized_control.y[u] = matrix[p * columns + columns - 1];
      for (int z = 0; z < current.reduced_dim; ++z) {
        initialized_control.Y[u * initialized_control.state_dim + z] =
            -matrix[p * columns + s.m + z];
      }
      for (int v = 0; v < initialized_control.reduced_dim; ++v) {
        initialized_control.Z[u * initialized_control.reduced_dim + v] =
            -matrix[p * columns + initialized_control.free_columns[v]];
      }
    }
    for (int v = 0; v < initialized_control.reduced_dim; ++v) {
      initialized_control.Z[initialized_control.free_columns[v] *
                                initialized_control.reduced_dim +
                            v] = Scalar{1};
    }
  }
  WarpSynchronize();

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
      value += s.A[xp * s.n + x] * current.T[x * current.reduced_dim + z];
    }
    for (int u = 0; u < s.m; ++u) {
      value += s.B[xp * s.m + u] * cp.Y[u * cp.state_dim + z];
    }
    rs.A[row * rs.n + z] = value;
  }
  for (int linear = threadIdx.x; linear < rs.next_n * rs.m;
       linear += blockDim.x) {
    const int row = linear / rs.m;
    const int v = linear % rs.m;
    const int xp = next.free_columns[row];
    Scalar value = Scalar{0};
    for (int u = 0; u < s.m; ++u) {
      value += s.B[xp * s.m + u] * cp.Z[u * cp.reduced_dim + v];
    }
    rs.B[row * rs.m + v] = value;
  }
  for (int row = threadIdx.x; row < rs.next_n; row += blockDim.x) {
    const int xp = next.free_columns[row];
    Scalar value = s.c[xp] - next.t[xp];
    for (int x = 0; x < s.n; ++x) {
      value += s.A[xp * s.n + x] * current.t[x];
    }
    for (int u = 0; u < s.m; ++u) {
      value += s.B[xp * s.m + u] * cp.y[u];
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
        value += current.T[x * current.reduced_dim + a] * s.Q[x * s.n + y] *
                 current.T[y * current.reduced_dim + b];
      }
      for (int u = 0; u < s.m; ++u) {
        value += current.T[x * current.reduced_dim + a] * s.M[x * s.m + u] *
                 cp.Y[u * cp.state_dim + b];
        value += cp.Y[u * cp.state_dim + a] * s.M[x * s.m + u] *
                 current.T[x * current.reduced_dim + b];
      }
    }
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value += cp.Y[u * cp.state_dim + a] * s.R[u * s.m + v] *
                 cp.Y[v * cp.state_dim + b];
      }
    }
    rs.Q[a * rs.n + b] = value;
  }
  for (int linear = threadIdx.x; linear < rs.m * rs.m; linear += blockDim.x) {
    const int a = linear / rs.m;
    const int b = linear % rs.m;
    Scalar value = Scalar{0};
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value += cp.Z[u * cp.reduced_dim + a] * s.R[u * s.m + v] *
                 cp.Z[v * cp.reduced_dim + b];
      }
    }
    rs.R[a * rs.m + b] = value;
  }
  for (int linear = threadIdx.x; linear < rs.n * rs.m; linear += blockDim.x) {
    const int z = linear / rs.m;
    const int v = linear % rs.m;
    Scalar value = Scalar{0};
    for (int x = 0; x < s.n; ++x) {
      for (int u = 0; u < s.m; ++u) {
        value += current.T[x * current.reduced_dim + z] * s.M[x * s.m + u] *
                 cp.Z[u * cp.reduced_dim + v];
      }
    }
    for (int u = 0; u < s.m; ++u) {
      for (int w = 0; w < s.m; ++w) {
        value += cp.Y[u * cp.state_dim + z] * s.R[u * s.m + w] *
                 cp.Z[w * cp.reduced_dim + v];
      }
    }
    rs.M[z * rs.m + v] = value;
  }

  for (int z = threadIdx.x; z < rs.n; z += blockDim.x) {
    Scalar value = Scalar{0};
    for (int x = 0; x < s.n; ++x) {
      Scalar gx = s.q[x];
      for (int y = 0; y < s.n; ++y)
        gx += s.Q[x * s.n + y] * current.t[y];
      for (int u = 0; u < s.m; ++u)
        gx += s.M[x * s.m + u] * cp.y[u];
      value += current.T[x * current.reduced_dim + z] * gx;
    }
    for (int u = 0; u < s.m; ++u) {
      Scalar gu = s.r[u];
      for (int x = 0; x < s.n; ++x)
        gu += s.M[x * s.m + u] * current.t[x];
      for (int v = 0; v < s.m; ++v)
        gu += s.R[u * s.m + v] * cp.y[v];
      value += cp.Y[u * cp.state_dim + z] * gu;
    }
    rs.q[z] = value;
  }
  for (int v = threadIdx.x; v < rs.m; v += blockDim.x) {
    Scalar value = Scalar{0};
    for (int u = 0; u < s.m; ++u) {
      Scalar gu = s.r[u];
      for (int x = 0; x < s.n; ++x)
        gu += s.M[x * s.m + u] * current.t[x];
      for (int w = 0; w < s.m; ++w)
        gu += s.R[u * s.m + w] * cp.y[w];
      value += cp.Z[u * cp.reduced_dim + v] * gu;
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
        value += param.T[x * param.reduced_dim + a] *
                 terminal.Q[x * terminal.n + y] *
                 param.T[y * param.reduced_dim + b];
      }
    }
    reduced->Q[a * reduced->n + b] = value;
  }
  for (int a = threadIdx.x; a < param.reduced_dim; a += blockDim.x) {
    Scalar value = Scalar{0};
    for (int x = 0; x < terminal.n; ++x) {
      Scalar gx = terminal.q[x];
      for (int y = 0; y < terminal.n; ++y) {
        gx += terminal.Q[x * terminal.n + y] * param.t[y];
      }
      value += param.T[x * param.reduced_dim + a] * gx;
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
  WarpSynchronize();
  if (threadIdx.x == 0) {
    Scalar scale = Scalar{1};
    Scalar residual = Scalar{0};
    for (int x = 0; x < param.physical_dim; ++x) {
      Scalar value = param.t[x];
      for (int z = 0; z < param.reduced_dim; ++z) {
        value += param.T[x * param.reduced_dim + z] * reduced_initial[z];
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
