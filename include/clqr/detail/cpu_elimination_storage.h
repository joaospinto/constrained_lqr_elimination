#ifndef CLQR_DETAIL_CPU_ELIMINATION_STORAGE_H_
#define CLQR_DETAIL_CPU_ELIMINATION_STORAGE_H_

#include "clqr/linalg.h"

namespace clqr::detail {

struct StateMap {
  Matrix linear;
  Vector offset;
};
struct ControlMap {
  Matrix state_linear;
  Matrix control_linear;
  Vector offset;
};
struct OrthogonalOperations {
  // Reflector k occupies columns [k, rows); columns [0,k) store R(0:k,k).
  Matrix reflectors;
  Vector betas, diagonal, row_scales;
  std::size_t rank = 0;
};
struct AffineStateBasis {
  Matrix T;
  Vector offset;
  OrthogonalOperations rhs_operations;
  WorkspaceVector<std::size_t> free_rows, pivot_rows;
  bool active = false, infeasible = false, redundant = false;
  std::string message;
};
struct MixedElimination {
  Matrix Y, Z;
  Vector y;
  Matrix state_C;
  Vector state_d;
  OrthogonalOperations rhs_operations;
  WorkspaceVector<std::size_t> pivot_columns, state_transformed_rows;
  bool active = false, infeasible = false, redundant = false;
  std::string message;
};
struct EliminationStageTrace {
  MixedElimination mixed;
};

constexpr std::size_t StorageMin(std::size_t a, std::size_t b) {
  return a < b ? a : b;
}
constexpr std::size_t StorageMax(std::size_t a, std::size_t b) {
  return a > b ? a : b;
}

// Counts the arrays, not the matrix/vector objects. Each nonempty allocation
// can waste at most alignment-1 bytes; round the complete region as well.
// The explicit allocation counts below also cover FP32/index interleaving.
constexpr std::size_t EliminationArrayBytes(std::size_t scalars,
                                            std::size_t indices,
                                            std::size_t allocations) {
  constexpr std::size_t alignment = alignof(std::size_t);
  static_assert(alignof(Scalar) <= alignment);
  const std::size_t bytes = scalars * sizeof(Scalar) +
                            indices * sizeof(std::size_t) +
                            allocations * (alignment - alignof(Scalar));
  return (bytes + alignment - 1) & ~(alignment - 1);
}

constexpr std::size_t OrthogonalScalars(std::size_t rows, std::size_t columns) {
  const std::size_t rank = StorageMin(rows, columns);
  return rank * rows + 2 * rank + rows;
}

// The retained arrays after one stage is eliminated. a is the number of
// next-state pivots: these add mixed rows but remove dynamics rows. For the
// current state/control we allow zero rank (the largest reduced matrices),
// while retaining the full pre-rank QR capacity. No full-row-rank assumption.
constexpr std::size_t EliminatedStageBytes(std::size_t n, std::size_t next_n,
                                           std::size_t m, std::size_t mixed,
                                           std::size_t state, std::size_t a) {
  const std::size_t p = mixed + a;
  const std::size_t s = state + p;
  std::size_t scalars =
      (next_n - a) * (n + m + 1) + n * n + 2 * n * m + m * m + n + 2 * m;
  std::size_t indices = 0;
  if (p != 0) {
    // Control map Z; mixed trace Y/Z/y and QR operations. The residual
    // relation itself is discarded after the state basis consumes it.
    scalars += m * m + m * n + m * m + m + OrthogonalScalars(p, m);
    indices += StorageMin(p, m) + p;
  }
  if (s != 0) {
    // State map and basis, and the state QR operations. Free+pivot indices
    // have total length n when copied into persistent storage.
    scalars += 2 * (n * n + n) + OrthogonalScalars(s, n);
    indices += n;
  }
  return EliminationArrayBytes(scalars, indices, 32);
}

constexpr std::size_t EliminatedStageBound(std::size_t n, std::size_t next_n,
                                           std::size_t m, std::size_t mixed,
                                           std::size_t state,
                                           std::size_t next_pivots) {
  // Between p=m and p+state=n the scalar count is convex quadratic or
  // linear in a. Thus a maximum occurs at an endpoint or one of these two
  // breakpoints. a=1 covers the activation of otherwise empty constraints.
  const std::size_t candidates[] = {
      0, StorageMin(1, next_pivots), next_pivots,
      StorageMin(next_pivots, m > mixed ? m - mixed : 0),
      StorageMin(next_pivots, n > state + mixed ? n - state - mixed : 0)};
  std::size_t bytes = 0;
  for (std::size_t a : candidates)
    bytes =
        StorageMax(bytes, EliminatedStageBytes(n, next_n, m, mixed, state, a));
  return bytes;
}

constexpr std::size_t EliminatedTerminalBytes(std::size_t n, std::size_t rows) {
  std::size_t scalars = n * n + n;
  if (rows != 0) scalars += 2 * (n * n + n) + OrthogonalScalars(rows, n);
  return EliminationArrayBytes(scalars, rows != 0 ? n : 0, 10);
}

// Allocation ledger for a complete stage in the monotonic scratch arena.
// Unlike retained storage, all intermediates are counted, but this region
// is reused at every stage. n/m/next_n are unreduced dimensions.
constexpr std::size_t EliminationStageScratchBytes(
    std::size_t n, std::size_t next_n, std::size_t m, std::size_t mixed,
    std::size_t state, std::size_t next_pivots) {
  const std::size_t p = mixed + next_pivots;
  const std::size_t s = state + p;
  // Input stage, Q/q, and initial control map.
  std::size_t scalars = next_n * (n + m + 1) + n * n + 2 * n * m + m * m + n +
                        2 * m + mixed * (n + m + 1) + state * (n + 1);
  std::size_t indices = 0;
  if (next_pivots != 0) {
    // Old dynamics, selected rows, propagated constraints, concatenations.
    scalars += 3 * next_n * (n + m + 1) + p * (n + m + 1);
  }
  if (p != 0) {
    // ControlBasis: augmented/restored QR input, orthogonal trace, Y/Z/y,
    // residual state relation; permutation and both copies of index lists.
    scalars += 2 * p * (m + n + 1) + OrthogonalScalars(p, m) + m * n + m * m +
               m + p * (n + 1);
    indices += 2 * StorageMin(p, m) + 2 * m + 2 * p;
    // Mixed substitution products, replacement matrices/maps and appended E/e.
    scalars += n + 3 * n * m + 3 * m * m + 3 * m + next_n * m + s * (n + 1);
  }
  if (s != 0) {
    // StateBasis, then current-state cost/dynamics/map substitution.
    scalars += 2 * s * (n + 1) + OrthogonalScalars(s, n) + n * n + n;
    indices += 2 * n + 2 * StorageMin(s, n);
    scalars += 8 * n * n + 5 * n * m + m * m + 2 * next_n * n + 5 * n + 5 * m +
               2 * next_n;
  }
  return EliminationArrayBytes(scalars, indices, 96);
}

constexpr std::size_t EliminationTerminalScratchBytes(std::size_t n,
                                                      std::size_t rows) {
  std::size_t scalars = n * n + n + rows * (n + 1);
  std::size_t indices = 0;
  if (rows != 0) {
    scalars += 2 * rows * (n + 1) + OrthogonalScalars(rows, n) + n * n + n +
               7 * n * n + 5 * n;
    indices += 2 * n + 2 * StorageMin(n, rows);
  }
  return EliminationArrayBytes(scalars, indices, 32);
}

constexpr std::size_t MixedRecoveryScratchBytes(std::size_t m, std::size_t p) {
  return EliminationArrayBytes(
      2 * p + 2 * m * (p + 1) + OrthogonalScalars(m, p), p + StorageMin(m, p),
      10);
}

constexpr std::size_t PullbackScratchBytes(std::size_t n, std::size_t next_n,
                                           std::size_t m, std::size_t p,
                                           std::size_t e) {
  // Sum of all vector expressions in one RecoverEliminatedMultipliers stage:
  // bar_c, bar_y and its products, bar_offset and products, bar_e/bar_d,
  // selected residual rows, output slices, and reconstructed dynamics.
  return EliminationArrayBytes(4 * next_n + 5 * m + 7 * n +
                                   2 * (e + p + next_n) + 3 * (p + next_n) + e +
                                   p,
                               0, 28);
}

constexpr std::size_t InitialParametrizationBytes(std::size_t n) {
  // Difference vector, augmented/restored QR, trace, free/pivot/permutation
  // indices, affine basis and offset (including the rank-zero case).
  return EliminationArrayBytes(4 * n * n + 7 * n, 3 * n, 12);
}

constexpr std::size_t ReplayMatrixScratchBytes(std::size_t n,
                                               std::size_t next_n,
                                               std::size_t m) {
  // All products and copies in BuildConstrainedReplayCache, one stage only.
  return EliminationArrayBytes(
      4 * next_n * n + 2 * next_n * m + 7 * n * n + 6 * n * m + m * m, 0, 24);
}

constexpr std::size_t ReplayRhsScratchBytes(std::size_t n, std::size_t next_n,
                                            std::size_t m, std::size_t p,
                                            std::size_t e) {
  return EliminationArrayBytes(4 * n + 8 * m + 8 * next_n + 5 * p + 3 * e, 0,
                               24);
}

}  // namespace clqr::detail
#endif  // CLQR_DETAIL_CPU_ELIMINATION_STORAGE_H_
