#ifndef CLQR_SRC_METAL_DUAL_OBJECTIVE_SOURCE_H_
#define CLQR_SRC_METAL_DUAL_OBJECTIVE_SOURCE_H_

namespace clqr::metal::detail {

inline constexpr char kMetalDualObjectiveSourcePart0[] = R"CLQR_METAL(
#include <metal_stdlib>

using namespace metal;

// The layout is an exact MSL mirror of src/metal_layout.h.  All persistent
// records contain offsets, never device pointers.
struct KernelParams {
  uint stage_count;
  uint state_capacity;
  uint control_capacity;
  uint mixed_capacity;
  uint state_constraint_capacity;
  uint terminal_constraint_capacity;
  float rank_tolerance;
  float consistency_tolerance;

  uint input_A;
  uint input_B;
  uint input_c;
  uint input_Q;
  uint input_R;
  uint input_M;
  uint input_q;
  uint input_r;
  uint input_C;
  uint input_D;
  uint input_d;
  uint input_E;
  uint input_e;
  uint input_terminal_E;
  uint input_terminal_e;
  uint input_initial_state;

  uint output_objective;
  uint output_states;
  uint output_controls;
  uint output_initial_multiplier;
  uint output_dynamics_multipliers;
  uint output_mixed_multipliers;
  uint output_state_multipliers;
  uint output_terminal_state_multiplier;

  uint relation_data;
  uint relation_stride;
  uint state_param_data;
  uint state_param_stride;
  uint control_param_data;
  uint control_param_stride;
  uint reduced_stage_data;
  uint reduced_stage_stride;
  uint reduced_terminal_data;
  uint value_data;
  uint value_stride;
  uint feedback_data;
  uint feedback_stride;
  uint affine_data;
  uint affine_stride;
  uint reduced_initial;
  uint reduced_costates;
  uint reduced_states;
  uint reduced_controls;
  uint dual_param_data;
  uint dual_param_stride;
  uint state_dual_param_data;
  uint state_dual_param_stride;
  uint dual_relation_data;
  uint dual_relation_stride;
  uint dual_node_data;
  uint dual_node_stride;
  uint objective_tree;
  uint float_scratch;
  uint float_scratch_stride;

  uint status;
  uint relation_meta;
  uint state_param_meta;
  uint state_param_free_columns;
  uint control_param_meta;
  uint control_param_free_columns;
  uint reduced_stage_meta;
  uint value_meta;
  uint feedback_meta;
  uint affine_meta;
  uint dual_param_meta;
  uint dual_param_free_columns;
  uint state_dual_param_meta;
  uint dual_relation_meta;
  uint dual_node_meta;
  uint int_scratch;
  uint int_scratch_stride;

  uint child_offset;
  uint parent_offset;
  uint child_count;
  uint parent_count;
  uint temporary_offset;
  uint phase_detail;
  uint padding0;
  uint padding1;
  uint padding2;
};

constant int kDeviceOk = 0;
constant int kDeviceInfeasible = 1;
constant int kDeviceNumericalFailure = 2;
constant int kDeviceInvalidInput = 3;
constant float kMinimumDualRelationRowScale = 1.0e-6f;

inline bool enabled(device int *metadata, constant KernelParams &p) {
  device atomic_int *code =
      reinterpret_cast<device atomic_int *>(metadata + p.status);
  return atomic_load_explicit(code, memory_order_relaxed) == kDeviceOk;
}

inline void fail(device int *metadata, constant KernelParams &p, int code,
                 int stage, int detail) {
  device atomic_int *status =
      reinterpret_cast<device atomic_int *>(metadata + p.status);
  int expected = kDeviceOk;
  bool claimed = false;
  while (expected == kDeviceOk &&
         !(claimed = atomic_compare_exchange_weak_explicit(
               status, &expected, code, memory_order_relaxed,
               memory_order_relaxed))) {
  }
  if (claimed) {
    metadata[p.status + 1u] = stage;
    metadata[p.status + 2u] = detail;
  }
}

inline uint control_dimension_offset(constant KernelParams &p) {
  return p.stage_count + 1u;
}
inline uint mixed_dimension_offset(constant KernelParams &p) {
  return 2u * p.stage_count + 1u;
}
inline uint state_constraint_dimension_offset(constant KernelParams &p) {
  return 3u * p.stage_count + 1u;
}
inline uint terminal_constraint_dimension_offset(constant KernelParams &p) {
  return 4u * p.stage_count + 1u;
}
inline int state_dim(device const int *dimensions, uint node) {
  return dimensions[node];
}
inline int control_dim(device const int *dimensions, constant KernelParams &p,
                       uint stage) {
  return dimensions[control_dimension_offset(p) + stage];
}
inline int mixed_dim(device const int *dimensions, constant KernelParams &p,
                     uint stage) {
  return dimensions[mixed_dimension_offset(p) + stage];
}
inline int state_constraint_dim(device const int *dimensions,
                                constant KernelParams &p, uint node) {
  return node == p.stage_count
             ? dimensions[terminal_constraint_dimension_offset(p)]
             : dimensions[state_constraint_dimension_offset(p) + node];
}

inline uint dual_capacity(constant KernelParams &p) {
  return p.state_capacity + p.mixed_capacity;
}
inline uint state_constraint_capacity(constant KernelParams &p) {
  return max(p.state_constraint_capacity, p.terminal_constraint_capacity);
}
inline uint dual_scan_flag(constant KernelParams &p) {
  // reduced_stage_meta reserves 3*stage_count stage entries and one flag.
  return p.reduced_stage_meta + 3u * p.stage_count;
}

inline device float *scratch(device float *workspace,
                             constant KernelParams &p, uint slot) {
  return workspace + p.float_scratch + slot * p.float_scratch_stride;
}
inline device int *integer_scratch(device int *metadata,
                                   constant KernelParams &p, uint slot) {
  return metadata + p.int_scratch + slot * p.int_scratch_stride;
}

// StateParam:
//   data = T[nx,nx], t[nx] (T row stride nx)
//   meta = physical_dim, reduced_dim.
inline device const float *state_T(device const float *workspace,
                                   constant KernelParams &p, uint node) {
  return workspace + p.state_param_data + node * p.state_param_stride;
}
inline int reduced_state_dim(device const int *metadata,
                             constant KernelParams &p, uint node) {
  return metadata[p.state_param_meta + 2u * node + 1u];
}

// ValueElement uses A[nx,nx], C[nx,nx], J[nx,nx], with active compact
// matrices in each fixed-capacity block.  The completed suffix scan leaves
// node suffixes in slots [0, stage_count].
inline device const float *value_J(device const float *workspace,
                                   constant KernelParams &p, uint node) {
  return workspace + p.value_data + node * p.value_stride +
         2u * p.state_capacity * p.state_capacity;
}
inline int value_left_dim(device const int *metadata,
                          constant KernelParams &p, uint node) {
  return metadata[p.value_meta + 2u * node];
}

// DualParam:
//   basis[dual_capacity,dual_capacity], offset[dual_capacity]
//   meta = state_dim, mixed_dim, physical_dim, free_dim.
inline device float *dual_basis(device float *workspace,
                                constant KernelParams &p, uint stage) {
  return workspace + p.dual_param_data + stage * p.dual_param_stride;
}
inline device const float *dual_basis(device const float *workspace,
                                      constant KernelParams &p, uint stage) {
  return workspace + p.dual_param_data + stage * p.dual_param_stride;
}
inline device float *dual_offset(device float *workspace,
                                 constant KernelParams &p, uint stage) {
  return dual_basis(workspace, p, stage) +
         dual_capacity(p) * dual_capacity(p);
}
inline device const float *dual_offset(device const float *workspace,
                                       constant KernelParams &p, uint stage) {
  return dual_basis(workspace, p, stage) +
         dual_capacity(p) * dual_capacity(p);
}
inline uint dual_meta_base(constant KernelParams &p, uint stage) {
  return p.dual_param_meta + 4u * stage;
}
inline int dual_state_dim(device const int *metadata, constant KernelParams &p,
                          uint stage) {
  return metadata[dual_meta_base(p, stage)];
}
inline int dual_physical_dim(device const int *metadata,
                             constant KernelParams &p, uint stage) {
  return metadata[dual_meta_base(p, stage) + 2u];
}
inline int dual_free_dim(device const int *metadata, constant KernelParams &p,
                         uint stage) {
  return metadata[dual_meta_base(p, stage) + 3u];
}

// StateDualParam:
//   offset[max_constraints],
//   left[max_constraints,dual_capacity],
//   right[max_constraints,dual_capacity]
//   meta = constraint_dim, left_dim, right_dim.
inline device float *state_dual_offset(device float *workspace,
                                       constant KernelParams &p, uint slot) {
  return workspace + p.state_dual_param_data +
         slot * p.state_dual_param_stride;
}
inline device const float *state_dual_offset(
    device const float *workspace, constant KernelParams &p, uint slot) {
  return workspace + p.state_dual_param_data +
         slot * p.state_dual_param_stride;
}
inline device float *state_dual_left(device float *workspace,
                                     constant KernelParams &p, uint slot) {
  return state_dual_offset(workspace, p, slot) +
         state_constraint_capacity(p);
}
inline device const float *state_dual_left(
    device const float *workspace, constant KernelParams &p, uint slot) {
  return state_dual_offset(workspace, p, slot) +
         state_constraint_capacity(p);
}
inline device float *state_dual_right(device float *workspace,
                                      constant KernelParams &p, uint slot) {
  return state_dual_left(workspace, p, slot) +
         state_constraint_capacity(p) * dual_capacity(p);
}
inline device const float *state_dual_right(
    device const float *workspace, constant KernelParams &p, uint slot) {
  return state_dual_left(workspace, p, slot) +
         state_constraint_capacity(p) * dual_capacity(p);
}
inline uint state_dual_meta_base(constant KernelParams &p, uint slot) {
  return p.state_dual_param_meta + 3u * slot;
}

// A dual relation is stored as two [2*dual_capacity,dual_capacity] blocks
// followed by rhs[2*dual_capacity].  This is the fixed-capacity counterpart
// of CUDA's exact compact relation arena.
inline uint dual_relation_rows_capacity(constant KernelParams &p) {
  return 2u * dual_capacity(p);
}
inline uint dual_relation_left_base(constant KernelParams &p, uint slot) {
  return p.dual_relation_data + slot * p.dual_relation_stride;
}
inline uint dual_relation_right_base(constant KernelParams &p, uint slot) {
  return dual_relation_left_base(p, slot) +
         dual_relation_rows_capacity(p) * dual_capacity(p);
}
inline uint dual_relation_rhs_base(constant KernelParams &p, uint slot) {
  return dual_relation_right_base(p, slot) +
         dual_relation_rows_capacity(p) * dual_capacity(p);
}
inline uint dual_relation_meta_base(constant KernelParams &p, uint slot) {
  return p.dual_relation_meta + 3u * slot;
}
inline int dual_relation_left_dim(device const int *metadata,
                                  constant KernelParams &p, uint slot) {
  return metadata[dual_relation_meta_base(p, slot)];
}
inline int dual_relation_right_dim(device const int *metadata,
                                   constant KernelParams &p, uint slot) {
  return metadata[dual_relation_meta_base(p, slot) + 1u];
}
inline int dual_relation_rows(device const int *metadata,
                              constant KernelParams &p, uint slot) {
  return metadata[dual_relation_meta_base(p, slot) + 2u];
}
inline void set_dual_relation_shape(device int *metadata,
                                    constant KernelParams &p, uint slot,
                                    int left, int right, int rows) {
  const uint meta = dual_relation_meta_base(p, slot);
  metadata[meta] = left;
  metadata[meta + 1u] = right;
  metadata[meta + 2u] = rows;
}

// DualNodeValue stores left[dual_capacity], right[dual_capacity].
inline device float *dual_node_left(device float *workspace,
                                    constant KernelParams &p, uint slot) {
  return workspace + p.dual_node_data + slot * p.dual_node_stride;
}
inline device const float *dual_node_left(device const float *workspace,
                                          constant KernelParams &p,
                                          uint slot) {
  return workspace + p.dual_node_data + slot * p.dual_node_stride;
}
inline device float *dual_node_right(device float *workspace,
                                     constant KernelParams &p, uint slot) {
  return dual_node_left(workspace, p, slot) + dual_capacity(p);
}
inline device const float *dual_node_right(device const float *workspace,
                                           constant KernelParams &p,
                                           uint slot) {
  return dual_node_left(workspace, p, slot) + dual_capacity(p);
}
inline uint dual_node_meta_base(constant KernelParams &p, uint slot) {
  return p.dual_node_meta + 2u * slot;
}

inline float conditioned_rhs_scale(device const float *matrix, int rows,
                                   int columns, int lhs_columns,
                                   float rank_tolerance) {
  float scale = 1.0f;
  for (int row = 0; row < rows; ++row) {
    float lhs_scale = 0.0f;
    for (int col = 0; col < lhs_columns; ++col)
      lhs_scale = fmax(lhs_scale, fabs(matrix[row * columns + col]));
    float rhs_scale = fabs(matrix[row * columns + lhs_columns]);
    if (lhs_scale > rank_tolerance)
      rhs_scale /= lhs_scale;
    scale = fmax(scale, rhs_scale);
  }
  return scale;
}

inline float conditioned_rhs_scale_threadgroup(
    threadgroup const float *matrix, int rows, int columns, int lhs_columns,
    float rank_tolerance) {
  float scale = 1.0f;
  for (int row = 0; row < rows; ++row) {
    float lhs_scale = 0.0f;
    for (int col = 0; col < lhs_columns; ++col)
      lhs_scale = fmax(lhs_scale, fabs(matrix[row * columns + col]));
    float rhs_scale = fabs(matrix[row * columns + lhs_columns]);
    if (lhs_scale > rank_tolerance)
      rhs_scale /= lhs_scale;
    scale = fmax(scale, rhs_scale);
  }
  return scale;
}


inline bool inconsistent_relation(device const float *matrix, int rows,
                                  int columns, int lhs_columns,
                                  float lhs_tolerance,
                                  float rhs_tolerance) {
  for (int row = 0; row < rows; ++row) {
    bool zero = true;
    for (int col = 0; col < lhs_columns; ++col) {
      if (fabs(matrix[row * columns + col]) > lhs_tolerance) {
        zero = false;
        break;
      }
    }
    if (zero && fabs(matrix[row * columns + lhs_columns]) > rhs_tolerance)
      return true;
  }
  return false;
}

inline bool inconsistent_relation_threadgroup(
    threadgroup const float *matrix, int rows, int columns, int lhs_columns,
    float lhs_tolerance, float rhs_tolerance) {
  for (int row = 0; row < rows; ++row) {
    bool zero = true;
    for (int col = 0; col < lhs_columns; ++col) {
      if (fabs(matrix[row * columns + col]) > lhs_tolerance) {
        zero = false;
        break;
      }
    }
    if (zero && fabs(matrix[row * columns + lhs_columns]) > rhs_tolerance)
      return true;
  }
  return false;
}

// Column-pivoted, twice-reorthogonalized modified Gram--Schmidt.  Free
// variables are zero in pivoted coordinates.  This is the serial small-system
// form of CUDA SolveSystemOrthogonally; every Metal grid thread owns one
// independent stage/tree node.
inline bool solve_system_orthogonally(
    device float *matrix, int rows, int columns, int variables,
    float rank_tolerance, float consistency_tolerance, float rhs_scale,
    device float *residual_rhs, device float *upper,
    device float *rhs_projection, device float *solution,
    device int *permutation, thread int &rank) {
  for (int i = 0; i < variables * variables; ++i)
    upper[i] = 0.0f;
  for (int variable = 0; variable < variables; ++variable) {
    permutation[variable] = variable;
    rhs_projection[variable] = 0.0f;
    solution[variable] = 0.0f;
  }
  for (int row = 0; row < rows; ++row)
    residual_rhs[row] = matrix[row * columns + variables];

  rank = 0;
  const float threshold_squared = rank_tolerance * rank_tolerance;
  for (int basis = 0; basis < variables; ++basis) {
    int best_column = basis;
    float best_norm_squared = -1.0f;
    for (int candidate = basis; candidate < variables; ++candidate) {
      float norm_squared = 0.0f;
      for (int row = 0; row < rows; ++row) {
        const float value = matrix[row * columns + candidate];
        norm_squared = fma(value, value, norm_squared);
      }
      if (norm_squared > best_norm_squared) {
        best_norm_squared = norm_squared;
        best_column = candidate;
      }
    }
    if (!(best_norm_squared > threshold_squared))
      break;
    if (best_column != basis) {
      for (int row = 0; row < rows; ++row) {
        const float value = matrix[row * columns + basis];
        matrix[row * columns + basis] =
            matrix[row * columns + best_column];
        matrix[row * columns + best_column] = value;
      }
      for (int previous = 0; previous < basis; ++previous) {
        const float value = upper[previous * variables + basis];
        upper[previous * variables + basis] =
            upper[previous * variables + best_column];
        upper[previous * variables + best_column] = value;
      }
      const int variable = permutation[basis];
      permutation[basis] = permutation[best_column];
      permutation[best_column] = variable;
    }

    float norm_squared = 0.0f;
    for (int row = 0; row < rows; ++row) {
      const float value = matrix[row * columns + basis];
      norm_squared = fma(value, value, norm_squared);
    }
    if (!(norm_squared > threshold_squared))
      break;
    const float norm = sqrt(norm_squared);
    upper[basis * variables + basis] = norm;
    for (int row = 0; row < rows; ++row)
      matrix[row * columns + basis] /= norm;

    for (int pass = 0; pass < 2; ++pass) {
      float projection = 0.0f;
      for (int row = 0; row < rows; ++row)
        projection = fma(matrix[row * columns + basis], residual_rhs[row],
                         projection);
      rhs_projection[basis] += projection;
      for (int row = 0; row < rows; ++row)
        residual_rhs[row] -= projection * matrix[row * columns + basis];
    }
    for (int candidate = basis + 1; candidate < variables; ++candidate) {
      for (int pass = 0; pass < 2; ++pass) {
        float projection = 0.0f;
        for (int row = 0; row < rows; ++row)
          projection =
              fma(matrix[row * columns + basis],
                  matrix[row * columns + candidate], projection);
        upper[basis * variables + candidate] += projection;
        for (int row = 0; row < rows; ++row)
          matrix[row * columns + candidate] -=
              projection * matrix[row * columns + basis];
      }
    }
    ++rank;
  }

  float maximum_residual = 0.0f;
  for (int row = 0; row < rows; ++row)
    maximum_residual = fmax(maximum_residual, fabs(residual_rhs[row]));
  if (maximum_residual > consistency_tolerance * rhs_scale)
    return false;

  for (int reverse = 0; reverse < rank; ++reverse) {
    const int row = rank - 1 - reverse;
    float value = rhs_projection[row];
    for (int col = row + 1; col < rank; ++col)
      value = fma(-upper[row * variables + col], rhs_projection[col],
                  value);
    rhs_projection[row] = value / upper[row * variables + row];
  }
  for (int variable = 0; variable < rank; ++variable)
    solution[permutation[variable]] = rhs_projection[variable];
  return true;
}

// Address-space specialization for one scalar stage per SIMD lane.  Every
// lane owns a disjoint threadgroup-memory slice, so this preserves the scalar
// operation order without barriers or cross-lane reassociation.
inline bool solve_system_orthogonally_threadgroup(
    threadgroup float *matrix, int rows, int columns, int variables,
    float rank_tolerance, float consistency_tolerance, float rhs_scale,
    threadgroup float *residual_rhs, threadgroup float *upper,
    threadgroup float *rhs_projection, threadgroup float *solution,
    threadgroup int *permutation, thread int &rank) {
  for (int i = 0; i < variables * variables; ++i)
    upper[i] = 0.0f;
  for (int variable = 0; variable < variables; ++variable) {
    permutation[variable] = variable;
    rhs_projection[variable] = 0.0f;
    solution[variable] = 0.0f;
  }
  for (int row = 0; row < rows; ++row)
    residual_rhs[row] = matrix[row * columns + variables];

  rank = 0;
  const float threshold_squared = rank_tolerance * rank_tolerance;
  for (int basis = 0; basis < variables; ++basis) {
    int best_column = basis;
    float best_norm_squared = -1.0f;
    for (int candidate = basis; candidate < variables; ++candidate) {
      float norm_squared = 0.0f;
      for (int row = 0; row < rows; ++row) {
        const float value = matrix[row * columns + candidate];
        norm_squared = fma(value, value, norm_squared);
      }
      if (norm_squared > best_norm_squared) {
        best_norm_squared = norm_squared;
        best_column = candidate;
      }
    }
    if (!(best_norm_squared > threshold_squared))
      break;
    if (best_column != basis) {
      for (int row = 0; row < rows; ++row) {
        const float value = matrix[row * columns + basis];
        matrix[row * columns + basis] =
            matrix[row * columns + best_column];
        matrix[row * columns + best_column] = value;
      }
      for (int previous = 0; previous < basis; ++previous) {
        const float value = upper[previous * variables + basis];
        upper[previous * variables + basis] =
            upper[previous * variables + best_column];
        upper[previous * variables + best_column] = value;
      }
      const int variable = permutation[basis];
      permutation[basis] = permutation[best_column];
      permutation[best_column] = variable;
    }

    float norm_squared = 0.0f;
    for (int row = 0; row < rows; ++row) {
      const float value = matrix[row * columns + basis];
      norm_squared = fma(value, value, norm_squared);
    }
    if (!(norm_squared > threshold_squared))
      break;
    const float norm = sqrt(norm_squared);
    upper[basis * variables + basis] = norm;
    for (int row = 0; row < rows; ++row)
      matrix[row * columns + basis] /= norm;

    for (int pass = 0; pass < 2; ++pass) {
      float projection = 0.0f;
      for (int row = 0; row < rows; ++row)
        projection = fma(matrix[row * columns + basis], residual_rhs[row],
                         projection);
      rhs_projection[basis] += projection;
      for (int row = 0; row < rows; ++row)
        residual_rhs[row] -= projection * matrix[row * columns + basis];
    }
    for (int candidate = basis + 1; candidate < variables; ++candidate) {
      for (int pass = 0; pass < 2; ++pass) {
        float projection = 0.0f;
        for (int row = 0; row < rows; ++row)
          projection =
              fma(matrix[row * columns + basis],
                  matrix[row * columns + candidate], projection);
        upper[basis * variables + candidate] += projection;
        for (int row = 0; row < rows; ++row)
          matrix[row * columns + candidate] -=
              projection * matrix[row * columns + basis];
      }
    }
    ++rank;
  }

  float maximum_residual = 0.0f;
  for (int row = 0; row < rows; ++row)
    maximum_residual = fmax(maximum_residual, fabs(residual_rhs[row]));
  if (maximum_residual > consistency_tolerance * rhs_scale)
    return false;

  for (int reverse = 0; reverse < rank; ++reverse) {
    const int row = rank - 1 - reverse;
    float value = rhs_projection[row];
    for (int col = row + 1; col < rank; ++col)
      value = fma(-upper[row * variables + col], rhs_projection[col],
                  value);
    rhs_projection[row] = value / upper[row * variables + row];
  }
  for (int variable = 0; variable < rank; ++variable)
    solution[permutation[variable]] = rhs_projection[variable];
  return true;
}

// Column-pivoted Householder QR with the same equilibration, pivot
// restoration, and threshold conventions as CUDA OrthogonalEchelonBlock.
inline int orthogonal_echelon(
    device float *matrix, int rows, int columns, int pivot_limit,
    int equilibration_limit, float tolerance, device int *pivot_columns,
    device int *permutation, device float *reflector,
    float minimum_row_scale) {
  for (int row = 0; row < rows; ++row) {
    float scale = 0.0f;
    for (int col = 0; col < equilibration_limit; ++col)
      scale = fmax(scale, fabs(matrix[row * columns + col]));
    if (scale > minimum_row_scale)
      for (int col = 0; col < columns; ++col)
        matrix[row * columns + col] /= scale;
  }
  for (int col = 0; col < pivot_limit; ++col)
    permutation[col] = col;

  float matrix_scale = 0.0f;
  for (int col = 0; col < pivot_limit; ++col) {
    float squared_norm = 0.0f;
    for (int row = 0; row < rows; ++row) {
      const float value = matrix[row * columns + col];
      squared_norm = fma(value, value, squared_norm);
    }
    matrix_scale = fmax(matrix_scale, sqrt(squared_norm));
  }
  const float threshold = tolerance * fmax(1.0f, matrix_scale);
  int rank = 0;
  const int maximum_rank = min(rows, pivot_limit);
  for (int basis = 0; basis < maximum_rank; ++basis) {
    int selected_column = basis;
    float best_norm_squared = -1.0f;
    for (int candidate = basis; candidate < pivot_limit; ++candidate) {
      float squared_norm = 0.0f;
      for (int row = basis; row < rows; ++row) {
        const float value = matrix[row * columns + candidate];
        squared_norm = fma(value, value, squared_norm);
      }
      if (squared_norm > best_norm_squared) {
        best_norm_squared = squared_norm;
        selected_column = candidate;
      }
    }
    const float best_norm = sqrt(best_norm_squared);
    if (!(best_norm > threshold))
      break;
    if (selected_column != basis) {
      for (int row = 0; row < rows; ++row) {
        const float value = matrix[row * columns + basis];
        matrix[row * columns + basis] =
            matrix[row * columns + selected_column];
        matrix[row * columns + selected_column] = value;
      }
      const int variable = permutation[basis];
      permutation[basis] = permutation[selected_column];
      permutation[selected_column] = variable;
    }

    const float diagonal = matrix[basis * columns + basis];
    const float alpha = diagonal < 0.0f ? best_norm : -best_norm;
    for (int row = basis; row < rows; ++row)
      reflector[row] = matrix[row * columns + basis];
    reflector[basis] -= alpha;
    matrix[basis * columns + basis] = alpha;
    float reflector_norm_squared = 0.0f;
    for (int row = basis; row < rows; ++row)
      reflector_norm_squared =
          fma(reflector[row], reflector[row], reflector_norm_squared);
    const float beta = reflector_norm_squared > 0.0f
                           ? 2.0f / reflector_norm_squared
                           : 0.0f;
    if (!(beta > 0.0f))
      break;
    for (int col = basis + 1; col < columns; ++col) {
      float projection = 0.0f;
      for (int row = basis; row < rows; ++row)
        projection =
            fma(reflector[row], matrix[row * columns + col], projection);
      projection *= beta;
      for (int row = basis; row < rows; ++row)
        matrix[row * columns + col] =
            fma(-reflector[row], projection, matrix[row * columns + col]);
    }
    pivot_columns[basis] = permutation[basis];
    rank = basis + 1;
    for (int row = basis + 1; row < rows; ++row)
      matrix[row * columns + basis] = 0.0f;
  }

  for (int reverse = 0; reverse < rank; ++reverse) {
    const int row = rank - 1 - reverse;
    const float diagonal = matrix[row * columns + row];
    for (int col = row; col < columns; ++col)
      matrix[row * columns + col] /= diagonal;
    for (int upper = 0; upper < row; ++upper) {
      const float factor = matrix[upper * columns + row];
      for (int col = row; col < columns; ++col)
        matrix[upper * columns + col] =
            fma(-factor, matrix[row * columns + col],
                matrix[upper * columns + col]);
    }
  }

  for (int original = 0; original < pivot_limit; ++original) {
    int selected = original;
    while (permutation[selected] != original)
      ++selected;
    if (selected == original)
      continue;
    for (int row = 0; row < rows; ++row) {
      const float value = matrix[row * columns + original];
      matrix[row * columns + original] =
          matrix[row * columns + selected];
      matrix[row * columns + selected] = value;
    }
    const int variable = permutation[original];
    permutation[original] = permutation[selected];
    permutation[selected] = variable;
  }
  for (int index = 0; index < rows * columns; ++index) {
    const int row = index / columns;
    const int col = index - row * columns;
    if ((row >= rank && col < pivot_limit) ||
        fabs(matrix[index]) <= tolerance)
      matrix[index] = 0.0f;
  }
  return rank;
}

inline int orthogonal_echelon_threadgroup(
    threadgroup float *matrix, int rows, int columns, int pivot_limit,
    int equilibration_limit, float tolerance, threadgroup int *pivot_columns,
    threadgroup int *permutation, threadgroup float *reflector,
    float minimum_row_scale) {
  for (int row = 0; row < rows; ++row) {
    float scale = 0.0f;
    for (int col = 0; col < equilibration_limit; ++col)
      scale = fmax(scale, fabs(matrix[row * columns + col]));
    if (scale > minimum_row_scale)
      for (int col = 0; col < columns; ++col)
        matrix[row * columns + col] /= scale;
  }
  for (int col = 0; col < pivot_limit; ++col)
    permutation[col] = col;

  float matrix_scale = 0.0f;
  for (int col = 0; col < pivot_limit; ++col) {
    float squared_norm = 0.0f;
    for (int row = 0; row < rows; ++row) {
      const float value = matrix[row * columns + col];
      squared_norm = fma(value, value, squared_norm);
    }
    matrix_scale = fmax(matrix_scale, sqrt(squared_norm));
  }
  const float threshold = tolerance * fmax(1.0f, matrix_scale);
  int rank = 0;
  const int maximum_rank = min(rows, pivot_limit);
  for (int basis = 0; basis < maximum_rank; ++basis) {
    int selected_column = basis;
    float best_norm_squared = -1.0f;
    for (int candidate = basis; candidate < pivot_limit; ++candidate) {
      float squared_norm = 0.0f;
      for (int row = basis; row < rows; ++row) {
        const float value = matrix[row * columns + candidate];
        squared_norm = fma(value, value, squared_norm);
      }
      if (squared_norm > best_norm_squared) {
        best_norm_squared = squared_norm;
        selected_column = candidate;
      }
    }
    const float best_norm = sqrt(best_norm_squared);
    if (!(best_norm > threshold))
      break;
    if (selected_column != basis) {
      for (int row = 0; row < rows; ++row) {
        const float value = matrix[row * columns + basis];
        matrix[row * columns + basis] =
            matrix[row * columns + selected_column];
        matrix[row * columns + selected_column] = value;
      }
      const int variable = permutation[basis];
      permutation[basis] = permutation[selected_column];
      permutation[selected_column] = variable;
    }

    const float diagonal = matrix[basis * columns + basis];
    const float alpha = diagonal < 0.0f ? best_norm : -best_norm;
    for (int row = basis; row < rows; ++row)
      reflector[row] = matrix[row * columns + basis];
    reflector[basis] -= alpha;
    matrix[basis * columns + basis] = alpha;
    float reflector_norm_squared = 0.0f;
    for (int row = basis; row < rows; ++row)
      reflector_norm_squared =
          fma(reflector[row], reflector[row], reflector_norm_squared);
    const float beta = reflector_norm_squared > 0.0f
                           ? 2.0f / reflector_norm_squared
                           : 0.0f;
    if (!(beta > 0.0f))
      break;
    for (int col = basis + 1; col < columns; ++col) {
      float projection = 0.0f;
      for (int row = basis; row < rows; ++row)
        projection =
            fma(reflector[row], matrix[row * columns + col], projection);
      projection *= beta;
      for (int row = basis; row < rows; ++row)
        matrix[row * columns + col] =
            fma(-reflector[row], projection, matrix[row * columns + col]);
    }
    pivot_columns[basis] = permutation[basis];
    rank = basis + 1;
    for (int row = basis + 1; row < rows; ++row)
      matrix[row * columns + basis] = 0.0f;
  }

  for (int reverse = 0; reverse < rank; ++reverse) {
    const int row = rank - 1 - reverse;
    const float diagonal = matrix[row * columns + row];
    for (int col = row; col < columns; ++col)
      matrix[row * columns + col] /= diagonal;
    for (int upper = 0; upper < row; ++upper) {
      const float factor = matrix[upper * columns + row];
      for (int col = row; col < columns; ++col)
        matrix[upper * columns + col] =
            fma(-factor, matrix[row * columns + col],
                matrix[upper * columns + col]);
    }
  }

  for (int original = 0; original < pivot_limit; ++original) {
    int selected = original;
    while (permutation[selected] != original)
      ++selected;
    if (selected == original)
      continue;
    for (int row = 0; row < rows; ++row) {
      const float value = matrix[row * columns + original];
      matrix[row * columns + original] =
          matrix[row * columns + selected];
      matrix[row * columns + selected] = value;
    }
    const int variable = permutation[original];
    permutation[original] = permutation[selected];
    permutation[selected] = variable;
  }
  for (int index = 0; index < rows * columns; ++index) {
    const int row = index / columns;
    const int col = index - row * columns;
    if ((row >= rank && col < pivot_limit) ||
        fabs(matrix[index]) <= tolerance)
      matrix[index] = 0.0f;
  }
  return rank;
}


inline void clear_dual_relation(device float *workspace,
                                device int *metadata,
                                constant KernelParams &p, uint slot,
                                int left_dim, int right_dim, int rows) {
  set_dual_relation_shape(metadata, p, slot, left_dim, right_dim, rows);
  const uint left = dual_relation_left_base(p, slot);
  const uint right = dual_relation_right_base(p, slot);
  const uint rhs = dual_relation_rhs_base(p, slot);
  const uint row_capacity = dual_relation_rows_capacity(p);
  const uint d = dual_capacity(p);
  for (uint i = 0; i < row_capacity * d; ++i) {
    workspace[left + i] = 0.0f;
    workspace[right + i] = 0.0f;
  }
  for (uint row = 0; row < row_capacity; ++row)
    workspace[rhs + row] = 0.0f;
}

inline void copy_dual_relation(device float *workspace, device int *metadata,
                               constant KernelParams &p, uint source,
                               uint destination) {
  if (source == destination)
    return;
  const int left_dim = dual_relation_left_dim(metadata, p, source);
  const int right_dim = dual_relation_right_dim(metadata, p, source);
  const int rows = dual_relation_rows(metadata, p, source);
  clear_dual_relation(workspace, metadata, p, destination, left_dim,
                      right_dim, rows);
  const uint d = dual_capacity(p);
  const uint source_left = dual_relation_left_base(p, source);
  const uint source_right = dual_relation_right_base(p, source);
  const uint source_rhs = dual_relation_rhs_base(p, source);
  const uint destination_left = dual_relation_left_base(p, destination);
  const uint destination_right = dual_relation_right_base(p, destination);
  const uint destination_rhs = dual_relation_rhs_base(p, destination);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < left_dim; ++col)
      workspace[destination_left + uint(row) * d + uint(col)] =
          workspace[source_left + uint(row) * d + uint(col)];
    for (int col = 0; col < right_dim; ++col)
      workspace[destination_right + uint(row) * d + uint(col)] =
          workspace[source_right + uint(row) * d + uint(col)];
    workspace[destination_rhs + uint(row)] =
        workspace[source_rhs + uint(row)];
  }
}

// Orthogonal elimination for relation-tree composition.  This is the serial
// form of CUDA EliminateRelationOrthogonally: twice-reorthogonalized MGS for
// the shared columns followed by a twice-reorthogonalized row basis for the
// outer affine relation.
inline bool eliminate_relation_orthogonally(
    device float *matrix, int rows, int columns, int eliminated_columns,
    int left_dim, int right_dim, float rank_tolerance,
    float consistency_tolerance, device float *workspace,
    device int *metadata, constant KernelParams &p, uint output_slot) {
  const float threshold_squared = rank_tolerance * rank_tolerance;
  float rhs_scale = 1.0f;
  for (int row = 0; row < rows; ++row)
    rhs_scale = fmax(rhs_scale, fabs(matrix[row * columns + columns - 1]));

  int eliminated_rank = 0;
  for (int basis = 0; basis < eliminated_columns; ++basis) {
    int best_column = basis;
    float best_norm_squared = -1.0f;
    for (int candidate = basis; candidate < eliminated_columns; ++candidate) {
      float norm_squared = 0.0f;
      for (int row = 0; row < rows; ++row) {
        const float value = matrix[row * columns + candidate];
        norm_squared = fma(value, value, norm_squared);
      }
      if (norm_squared > best_norm_squared) {
        best_norm_squared = norm_squared;
        best_column = candidate;
      }
    }
    if (!(best_norm_squared > threshold_squared))
      break;
    if (best_column != basis) {
      for (int row = 0; row < rows; ++row) {
        const float value = matrix[row * columns + basis];
        matrix[row * columns + basis] =
            matrix[row * columns + best_column];
        matrix[row * columns + best_column] = value;
      }
    }
    for (int pass = 0; pass < 2; ++pass)
      for (int previous = 0; previous < basis; ++previous) {
        float projection = 0.0f;
        for (int row = 0; row < rows; ++row)
          projection =
              fma(matrix[row * columns + basis],
                  matrix[row * columns + previous], projection);
        for (int row = 0; row < rows; ++row)
          matrix[row * columns + basis] =
              fma(-projection, matrix[row * columns + previous],
                  matrix[row * columns + basis]);
      }
    float norm_squared = 0.0f;
    for (int row = 0; row < rows; ++row) {
      const float value = matrix[row * columns + basis];
      norm_squared = fma(value, value, norm_squared);
    }
    if (!(norm_squared > threshold_squared))
      break;
    const float inverse_norm = 1.0f / sqrt(norm_squared);
    for (int row = 0; row < rows; ++row)
      matrix[row * columns + basis] *= inverse_norm;
    for (int candidate = basis + 1; candidate < eliminated_columns;
         ++candidate)
      for (int pass = 0; pass < 2; ++pass) {
        float projection = 0.0f;
        for (int row = 0; row < rows; ++row)
          projection =
              fma(matrix[row * columns + candidate],
                  matrix[row * columns + basis], projection);
        for (int row = 0; row < rows; ++row)
          matrix[row * columns + candidate] =
              fma(-projection, matrix[row * columns + basis],
                  matrix[row * columns + candidate]);
      }
    ++eliminated_rank;
  }
  for (int col = eliminated_columns; col < columns; ++col)
    for (int basis = 0; basis < eliminated_rank; ++basis)
      for (int pass = 0; pass < 2; ++pass) {
        float projection = 0.0f;
        for (int row = 0; row < rows; ++row)
          projection =
              fma(matrix[row * columns + col],
                  matrix[row * columns + basis], projection);
        for (int row = 0; row < rows; ++row)
          matrix[row * columns + col] =
              fma(-projection, matrix[row * columns + basis],
                  matrix[row * columns + col]);
      }

  const int outer_columns = left_dim + right_dim;
  int relation_rank = 0;
  while (relation_rank < rows) {
    int best_row = relation_rank;
    float best_norm_squared = -1.0f;
    for (int candidate = relation_rank; candidate < rows; ++candidate) {
      float norm_squared = 0.0f;
      for (int col = 0; col < outer_columns; ++col) {
        const float value =
            matrix[candidate * columns + eliminated_columns + col];
        norm_squared = fma(value, value, norm_squared);
      }
      if (norm_squared > best_norm_squared) {
        best_norm_squared = norm_squared;
        best_row = candidate;
      }
    }
    if (!(best_norm_squared > threshold_squared))
      break;
    if (best_row != relation_rank) {
      for (int outer = 0; outer <= outer_columns; ++outer) {
        const int col = eliminated_columns + outer;
        const float value = matrix[relation_rank * columns + col];
        matrix[relation_rank * columns + col] =
            matrix[best_row * columns + col];
        matrix[best_row * columns + col] = value;
      }
    }
    for (int pass = 0; pass < 2; ++pass)
      for (int previous = 0; previous < relation_rank; ++previous) {
        float projection = 0.0f;
        for (int col = 0; col < outer_columns; ++col)
          projection = fma(
              matrix[relation_rank * columns + eliminated_columns + col],
              matrix[previous * columns + eliminated_columns + col],
              projection);
        for (int col = 0; col <= outer_columns; ++col)
          matrix[relation_rank * columns + eliminated_columns + col] =
              fma(-projection,
                  matrix[previous * columns + eliminated_columns + col],
                  matrix[relation_rank * columns + eliminated_columns + col]);
      }
    float norm_squared = 0.0f;
    for (int col = 0; col < outer_columns; ++col) {
      const float value =
          matrix[relation_rank * columns + eliminated_columns + col];
      norm_squared = fma(value, value, norm_squared);
    }
    if (!(norm_squared > threshold_squared))
      break;
    const float inverse_norm = 1.0f / sqrt(norm_squared);
    for (int col = 0; col <= outer_columns; ++col)
      matrix[relation_rank * columns + eliminated_columns + col] *=
          inverse_norm;
    for (int row = relation_rank + 1; row < rows; ++row)
      for (int pass = 0; pass < 2; ++pass) {
        float projection = 0.0f;
        for (int col = 0; col < outer_columns; ++col)
          projection = fma(
              matrix[row * columns + eliminated_columns + col],
              matrix[relation_rank * columns + eliminated_columns + col],
              projection);
        for (int col = 0; col <= outer_columns; ++col)
          matrix[row * columns + eliminated_columns + col] =
              fma(-projection,
                  matrix[relation_rank * columns + eliminated_columns + col],
                  matrix[row * columns + eliminated_columns + col]);
      }
    ++relation_rank;
  }

  float maximum_residual = 0.0f;
  for (int row = relation_rank; row < rows; ++row)
    maximum_residual =
        fmax(maximum_residual, fabs(matrix[row * columns + columns - 1]));
  if (maximum_residual > consistency_tolerance * rhs_scale)
    return false;

  clear_dual_relation(workspace, metadata, p, output_slot, left_dim, right_dim,
                      relation_rank);
  const uint left = dual_relation_left_base(p, output_slot);
  const uint right = dual_relation_right_base(p, output_slot);
  const uint rhs = dual_relation_rhs_base(p, output_slot);
  const uint d = dual_capacity(p);
  for (int row = 0; row < relation_rank; ++row) {
    for (int col = 0; col < left_dim; ++col)
      workspace[left + uint(row) * d + uint(col)] =
          matrix[row * columns + eliminated_columns + col];
    for (int col = 0; col < right_dim; ++col)
      workspace[right + uint(row) * d + uint(col)] =
          matrix[row * columns + eliminated_columns + left_dim + col];
    workspace[rhs + uint(row)] =
        matrix[row * columns + columns - 1];
  }
  return true;
}

inline bool compose_dual_relations(
    device float *workspace, device int *metadata, constant KernelParams &p,
    uint first_slot, uint second_slot, uint output_slot, uint scratch_slot,
    int failure_stage, int failure_detail) {
  const int first_left_dim =
      dual_relation_left_dim(metadata, p, first_slot);
  const int shared = dual_relation_right_dim(metadata, p, first_slot);
  const int first_rows = dual_relation_rows(metadata, p, first_slot);
  const int second_left_dim =
      dual_relation_left_dim(metadata, p, second_slot);
  const int second_right_dim =
      dual_relation_right_dim(metadata, p, second_slot);
  const int second_rows = dual_relation_rows(metadata, p, second_slot);
  if (shared != second_left_dim) {
    fail(metadata, p, kDeviceNumericalFailure, failure_stage, 2);
    return false;
  }
  const int rows = first_rows + second_rows;
  const int columns = shared + first_left_dim + second_right_dim + 1;
  device float *matrix = scratch(workspace, p, scratch_slot);
  for (int i = 0; i < rows * columns; ++i)
    matrix[i] = 0.0f;
  const uint d = dual_capacity(p);
  const uint first_left = dual_relation_left_base(p, first_slot);
  const uint first_right = dual_relation_right_base(p, first_slot);
  const uint first_rhs = dual_relation_rhs_base(p, first_slot);
  const uint second_left = dual_relation_left_base(p, second_slot);
  const uint second_right = dual_relation_right_base(p, second_slot);
  const uint second_rhs = dual_relation_rhs_base(p, second_slot);
  for (int row = 0; row < first_rows; ++row) {
    for (int col = 0; col < shared; ++col)
      matrix[row * columns + col] =
          workspace[first_right + uint(row) * d + uint(col)];
    for (int col = 0; col < first_left_dim; ++col)
      matrix[row * columns + shared + col] =
          workspace[first_left + uint(row) * d + uint(col)];
    matrix[row * columns + columns - 1] =
        workspace[first_rhs + uint(row)];
  }
  for (int row = 0; row < second_rows; ++row) {
    const int output_row = first_rows + row;
    for (int col = 0; col < shared; ++col)
      matrix[output_row * columns + col] =
          workspace[second_left + uint(row) * d + uint(col)];
    for (int col = 0; col < second_right_dim; ++col)
      matrix[output_row * columns + shared + first_left_dim + col] =
          workspace[second_right + uint(row) * d + uint(col)];
    matrix[output_row * columns + columns - 1] =
        workspace[second_rhs + uint(row)];
  }
  if (!eliminate_relation_orthogonally(
          matrix, rows, columns, shared, first_left_dim, second_right_dim,
          p.rank_tolerance, p.consistency_tolerance, workspace, metadata, p,
          output_slot)) {
    fail(metadata, p, kDeviceNumericalFailure, failure_stage, failure_detail);
    return false;
  }
  return true;
}

)CLQR_METAL";

inline constexpr char kMetalDualObjectiveSourcePart1[] = R"CLQR_METAL(

kernel void clqr_build_dual_parameters(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  (void)input;
  if (gid >= p.stage_count || !enabled(metadata, p))
    return;
  const int n = state_dim(dimensions, gid);
  const int next_n = state_dim(dimensions, gid + 1u);
  const int m = control_dim(dimensions, p, gid);
  const int mixed = mixed_dim(dimensions, p, gid);
  const int next_reduced = reduced_state_dim(metadata, p, gid + 1u);
  const int variables = next_n + mixed;
  const int rows = next_reduced + m;
  const int columns = variables + 1;
  device float *matrix = scratch(workspace, p, gid);
  device float *constraint_scales = matrix + rows * columns;
  device float *residual_rhs = constraint_scales + mixed;
  device float *upper = residual_rhs + rows;
  device float *rhs_projection = upper + variables * variables;
  device float *solution = rhs_projection + variables;
  device int *permutation = integer_scratch(metadata, p, gid);
  for (int entry = 0; entry < rows * columns; ++entry)
    matrix[entry] = 0.0f;

  const uint nx = p.state_capacity;
  const uint nu = p.control_capacity;
  const uint nc = p.mixed_capacity;
  const uint C = p.input_C + gid * nc * nx;
  const uint D = p.input_D + gid * nc * nu;
  for (int constraint = 0; constraint < mixed; ++constraint) {
    float scale = 0.0f;
    for (int state = 0; state < n; ++state)
      scale = fmax(scale, fabs(input[C + uint(constraint) * nx +
                                      uint(state)]));
    for (int control = 0; control < m; ++control)
      scale = fmax(scale, fabs(input[D + uint(constraint) * nu +
                                      uint(control)]));
    constraint_scales[constraint] = scale > 0.0f ? scale : 1.0f;
  }

  device const float *next_T = state_T(workspace, p, gid + 1u);
  for (int row = 0; row < next_reduced; ++row)
    for (int state = 0; state < next_n; ++state)
      matrix[row * columns + state] =
          next_T[uint(state) * nx + uint(row)];
  device const float *next_reduced_state =
      workspace + p.reduced_states + (gid + 1u) * nx;
  device const float *next_linear =
      workspace + p.reduced_costates + (gid + 1u) * nx;
  device const float *next_J = value_J(workspace, p, gid + 1u);
  const int next_value_dim = value_left_dim(metadata, p, gid + 1u);
  for (int row = 0; row < next_reduced; ++row) {
    float costate = -next_linear[row];
    for (int col = 0; col < next_reduced; ++col)
      costate -= next_J[row * next_value_dim + col] *
                 next_reduced_state[col];
    matrix[row * columns + variables] = costate;
  }
  const uint B = p.input_B + gid * nx * nu;
  for (int control = 0; control < m; ++control) {
    for (int state = 0; state < next_n; ++state)
      matrix[(next_reduced + control) * columns + state] =
          input[B + uint(state) * nu + uint(control)];
    for (int constraint = 0; constraint < mixed; ++constraint)
      matrix[(next_reduced + control) * columns + next_n + constraint] =
          -input[D + uint(constraint) * nu + uint(control)] /
          constraint_scales[constraint];
  }

  device const float *x = output + p.output_states + gid * nx;
  device const float *u = output + p.output_controls + gid * nu;
  const uint M = p.input_M + gid * nx * nu;
  const uint R = p.input_R + gid * nu * nu;
  const uint r = p.input_r + gid * nu;
  for (int row = 0; row < m; ++row) {
    float gradient = input[r + uint(row)];
    for (int col = 0; col < n; ++col)
      gradient =
          fma(input[M + uint(col) * nu + uint(row)], x[col], gradient);
    for (int col = 0; col < m; ++col)
      gradient =
          fma(input[R + uint(row) * nu + uint(col)], u[col], gradient);
    matrix[(next_reduced + row) * columns + variables] = gradient;
  }
  const float rhs_scale = conditioned_rhs_scale(
      matrix, rows, columns, variables, p.rank_tolerance);
  int rank = 0;
  if (!solve_system_orthogonally(
          matrix, rows, columns, variables, p.rank_tolerance,
          p.consistency_tolerance, rhs_scale, residual_rhs, upper,
          rhs_projection, solution, permutation, rank)) {
    fail(metadata, p, kDeviceNumericalFailure, int(gid), 24);
    return;
  }

  const int free_dim = variables - rank;
  const uint meta = dual_meta_base(p, gid);
  metadata[meta] = next_n;
  metadata[meta + 1u] = mixed;
  metadata[meta + 2u] = variables;
  metadata[meta + 3u] = free_dim;
  const uint dcap = dual_capacity(p);
  device float *basis = dual_basis(workspace, p, gid);
  device float *offset = dual_offset(workspace, p, gid);
  for (uint i = 0; i < dcap * dcap; ++i)
    basis[i] = 0.0f;
  for (uint i = 0; i < dcap; ++i)
    offset[i] = 0.0f;
  for (int free = 0; free < free_dim; ++free)
    metadata[p.dual_param_free_columns + gid * dcap + uint(free)] =
        permutation[rank + free];
  if (free_dim > 0)
    atomic_store_explicit(
        reinterpret_cast<device atomic_int *>(metadata + dual_scan_flag(p)),
        1, memory_order_relaxed);
  for (int variable = 0; variable < variables; ++variable) {
    const float scale =
        variable < next_n ? 1.0f : constraint_scales[variable - next_n];
    offset[variable] = solution[variable] / scale;
  }
  for (int free = 0; free < free_dim; ++free) {
    for (int position = 0; position < variables; ++position)
      solution[position] = 0.0f;
    solution[rank + free] = 1.0f;
    for (int reverse = 0; reverse < rank; ++reverse) {
      const int row = rank - 1 - reverse;
      float value = -upper[row * variables + rank + free];
      for (int col = row + 1; col < rank; ++col)
        value =
            fma(-upper[row * variables + col], solution[col], value);
      solution[row] = value / upper[row * variables + row];
    }
    for (int position = 0; position < variables; ++position) {
      const int variable = permutation[position];
      const float scale =
          variable < next_n ? 1.0f : constraint_scales[variable - next_n];
      basis[uint(variable) * dcap + uint(free)] =
          solution[position] / scale;
    }
  }
}

kernel void clqr_build_dual_parameters_threadgroup_sliced(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    threadgroup float *local_float [[threadgroup(0)]],
    threadgroup int *local_int [[threadgroup(1)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]]) {
  const uint gid = group * lane_count + lane;
  if (gid >= p.stage_count || !enabled(metadata, p))
    return;
  const int n = state_dim(dimensions, gid);
  const int next_n = state_dim(dimensions, gid + 1u);
  const int m = control_dim(dimensions, p, gid);
  const int mixed = mixed_dim(dimensions, p, gid);
  const int next_reduced = reduced_state_dim(metadata, p, gid + 1u);
  const int variables = next_n + mixed;
  const int rows = next_reduced + m;
  const int columns = variables + 1;
  const uint nx = p.state_capacity;
  const uint nu = p.control_capacity;
  const uint nc = p.mixed_capacity;
  const uint dcap = dual_capacity(p);
  const uint nx_plus_nu = nx + nu;
  const uint per_lane_float_entries =
      nx_plus_nu * (dcap + 1u) + nc + nx_plus_nu + dcap * dcap +
      2u * dcap;
  const uint per_lane_float_stride =
      (per_lane_float_entries + 3u) & ~3u;
  const uint per_lane_integer_stride = (dcap + 7u) & ~3u;
  threadgroup float *matrix =
      local_float + lane * per_lane_float_stride;
  threadgroup float *constraint_scales = matrix + rows * columns;
  threadgroup float *residual_rhs = constraint_scales + mixed;
  threadgroup float *upper = residual_rhs + rows;
  threadgroup float *rhs_projection = upper + variables * variables;
  threadgroup float *solution = rhs_projection + variables;
  threadgroup int *permutation =
      local_int + lane * per_lane_integer_stride;
  for (int entry = 0; entry < rows * columns; ++entry)
    matrix[entry] = 0.0f;

  const uint C = p.input_C + gid * nc * nx;
  const uint D = p.input_D + gid * nc * nu;
  for (int constraint = 0; constraint < mixed; ++constraint) {
    float scale = 0.0f;
    for (int state = 0; state < n; ++state)
      scale = fmax(scale, fabs(input[C + uint(constraint) * nx +
                                      uint(state)]));
    for (int control = 0; control < m; ++control)
      scale = fmax(scale, fabs(input[D + uint(constraint) * nu +
                                      uint(control)]));
    constraint_scales[constraint] = scale > 0.0f ? scale : 1.0f;
  }

  device const float *next_T = state_T(workspace, p, gid + 1u);
  for (int row = 0; row < next_reduced; ++row)
    for (int state = 0; state < next_n; ++state)
      matrix[row * columns + state] =
          next_T[uint(state) * nx + uint(row)];
  device const float *next_reduced_state =
      workspace + p.reduced_states + (gid + 1u) * nx;
  device const float *next_linear =
      workspace + p.reduced_costates + (gid + 1u) * nx;
  device const float *next_J = value_J(workspace, p, gid + 1u);
  const int next_value_dim = value_left_dim(metadata, p, gid + 1u);
  for (int row = 0; row < next_reduced; ++row) {
    float costate = -next_linear[row];
    for (int col = 0; col < next_reduced; ++col)
      costate -= next_J[row * next_value_dim + col] *
                 next_reduced_state[col];
    matrix[row * columns + variables] = costate;
  }
  const uint B = p.input_B + gid * nx * nu;
  for (int control = 0; control < m; ++control) {
    for (int state = 0; state < next_n; ++state)
      matrix[(next_reduced + control) * columns + state] =
          input[B + uint(state) * nu + uint(control)];
    for (int constraint = 0; constraint < mixed; ++constraint)
      matrix[(next_reduced + control) * columns + next_n + constraint] =
          -input[D + uint(constraint) * nu + uint(control)] /
          constraint_scales[constraint];
  }

  device const float *x = output + p.output_states + gid * nx;
  device const float *u = output + p.output_controls + gid * nu;
  const uint M = p.input_M + gid * nx * nu;
  const uint R = p.input_R + gid * nu * nu;
  const uint r = p.input_r + gid * nu;
  for (int row = 0; row < m; ++row) {
    float gradient = input[r + uint(row)];
    for (int col = 0; col < n; ++col)
      gradient =
          fma(input[M + uint(col) * nu + uint(row)], x[col], gradient);
    for (int col = 0; col < m; ++col)
      gradient =
          fma(input[R + uint(row) * nu + uint(col)], u[col], gradient);
    matrix[(next_reduced + row) * columns + variables] = gradient;
  }
  const float rhs_scale = conditioned_rhs_scale_threadgroup(
      matrix, rows, columns, variables, p.rank_tolerance);
  int rank = 0;
  if (!solve_system_orthogonally_threadgroup(
          matrix, rows, columns, variables, p.rank_tolerance,
          p.consistency_tolerance, rhs_scale, residual_rhs, upper,
          rhs_projection, solution, permutation, rank)) {
    fail(metadata, p, kDeviceNumericalFailure, int(gid), 24);
    return;
  }

  const int free_dim = variables - rank;
  const uint meta = dual_meta_base(p, gid);
  metadata[meta] = next_n;
  metadata[meta + 1u] = mixed;
  metadata[meta + 2u] = variables;
  metadata[meta + 3u] = free_dim;
  device float *basis = dual_basis(workspace, p, gid);
  device float *offset = dual_offset(workspace, p, gid);
  for (uint i = 0; i < dcap * dcap; ++i)
    basis[i] = 0.0f;
  for (uint i = 0; i < dcap; ++i)
    offset[i] = 0.0f;
  for (int free = 0; free < free_dim; ++free)
    metadata[p.dual_param_free_columns + gid * dcap + uint(free)] =
        permutation[rank + free];
  if (free_dim > 0)
    atomic_store_explicit(
        reinterpret_cast<device atomic_int *>(metadata + dual_scan_flag(p)),
        1, memory_order_relaxed);
  for (int variable = 0; variable < variables; ++variable) {
    const float scale =
        variable < next_n ? 1.0f : constraint_scales[variable - next_n];
    offset[variable] = solution[variable] / scale;
  }
  for (int free = 0; free < free_dim; ++free) {
    for (int position = 0; position < variables; ++position)
      solution[position] = 0.0f;
    solution[rank + free] = 1.0f;
    for (int reverse = 0; reverse < rank; ++reverse) {
      const int row = rank - 1 - reverse;
      float value = -upper[row * variables + rank + free];
      for (int col = row + 1; col < rank; ++col)
        value =
            fma(-upper[row * variables + col], solution[col], value);
      solution[row] = value / upper[row * variables + row];
    }
    for (int position = 0; position < variables; ++position) {
      const int variable = permutation[position];
      const float scale =
          variable < next_n ? 1.0f : constraint_scales[variable - next_n];
      basis[uint(variable) * dcap + uint(free)] =
          solution[position] / scale;
    }
  }
}

kernel void clqr_build_dual_relation_leaves(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.stage_count || !enabled(metadata, p))
    return;
  const uint node = gid + 1u;
  const bool terminal = node == p.stage_count;
  const int n = state_dim(dimensions, node);
  const int constraints = state_constraint_dim(dimensions, p, node);
  const int left_free = dual_free_dim(metadata, p, node - 1u);
  const int right_free =
      terminal ? 0 : dual_free_dim(metadata, p, node);
  const int columns = constraints + left_free + right_free + 1;
  device float *matrix = scratch(workspace, p, gid);
  device float *factors = matrix + n * columns;
  device float *constraint_scales = factors + n;
  device float *reflector = constraint_scales + constraints;
  device int *pivot_columns = integer_scratch(metadata, p, gid);
  device int *permutation = pivot_columns + n;
  for (int i = 0; i < n * columns; ++i)
    matrix[i] = 0.0f;

  const uint nx = p.state_capacity;
  const uint nu = p.control_capacity;
  const uint nc = p.mixed_capacity;
  const uint ne = p.state_constraint_capacity;
  const uint E = terminal ? p.input_terminal_E
                          : p.input_E + node * ne * nx;
  for (int constraint = 0; constraint < constraints; ++constraint) {
    float scale = 0.0f;
    for (int state = 0; state < n; ++state)
      scale = fmax(scale, fabs(input[E + uint(constraint) * nx +
                                      uint(state)]));
    constraint_scales[constraint] = scale > 0.0f ? scale : 1.0f;
  }
  for (int state = 0; state < n; ++state)
    for (int constraint = 0; constraint < constraints; ++constraint)
      matrix[state * columns + constraint] =
          input[E + uint(constraint) * nx + uint(state)] /
          constraint_scales[constraint];

  const uint dcap = dual_capacity(p);
  device const float *left_basis =
      dual_basis(workspace, p, node - 1u);
  for (int state = 0; state < n; ++state)
    for (int free = 0; free < left_free; ++free)
      matrix[state * columns + constraints + free] =
          left_basis[uint(state) * dcap + uint(free)];

  if (!terminal) {
    const int next_n = state_dim(dimensions, node + 1u);
    const int mixed = mixed_dim(dimensions, p, node);
    device const float *right_basis = dual_basis(workspace, p, node);
    const uint A = p.input_A + node * nx * nx;
    const uint C = p.input_C + node * nc * nx;
    for (int state = 0; state < n; ++state)
      for (int free = 0; free < right_free; ++free) {
        float value = 0.0f;
        for (int next_state = 0; next_state < next_n; ++next_state)
          value = fma(-input[A + uint(next_state) * nx + uint(state)],
                      right_basis[uint(next_state) * dcap + uint(free)],
                      value);
        for (int constraint = 0; constraint < mixed; ++constraint)
          value = fma(input[C + uint(constraint) * nx + uint(state)],
                      right_basis[uint(next_n + constraint) * dcap +
                                  uint(free)],
                      value);
        matrix[state * columns + constraints + left_free + free] = value;
      }
  }

  device const float *x = output + p.output_states + node * nx;
  device const float *left_offset =
      dual_offset(workspace, p, node - 1u);
  for (int row = 0; row < n; ++row) {
    float rhs = -left_offset[row];
    if (terminal) {
      const uint Q = p.input_Q + node * nx * nx;
      const uint q = p.input_q + node * nx;
      rhs -= input[q + uint(row)];
      for (int col = 0; col < n; ++col)
        rhs = fma(-input[Q + uint(row) * nx + uint(col)], x[col], rhs);
    } else {
      const int m = control_dim(dimensions, p, node);
      const int next_n = state_dim(dimensions, node + 1u);
      const int mixed = mixed_dim(dimensions, p, node);
      const uint Q = p.input_Q + node * nx * nx;
      const uint q = p.input_q + node * nx;
      const uint M = p.input_M + node * nx * nu;
      const uint A = p.input_A + node * nx * nx;
      const uint C = p.input_C + node * nc * nx;
      device const float *u = output + p.output_controls + node * nu;
      device const float *right_offset =
          dual_offset(workspace, p, node);
      rhs -= input[q + uint(row)];
      for (int col = 0; col < n; ++col)
        rhs = fma(-input[Q + uint(row) * nx + uint(col)], x[col], rhs);
      for (int col = 0; col < m; ++col)
        rhs = fma(-input[M + uint(row) * nu + uint(col)], u[col], rhs);
      for (int next_state = 0; next_state < next_n; ++next_state)
        rhs = fma(input[A + uint(next_state) * nx + uint(row)],
                  right_offset[next_state], rhs);
      for (int constraint = 0; constraint < mixed; ++constraint)
        rhs = fma(-input[C + uint(constraint) * nx + uint(row)],
                  right_offset[next_n + constraint], rhs);
    }
    matrix[row * columns + columns - 1] = rhs;
  }

  int rank = orthogonal_echelon(
      matrix, n, columns, constraints, columns - 1, p.rank_tolerance,
      pivot_columns, permutation, reflector, kMinimumDualRelationRowScale);
  const int constraint_rank = rank;
  const uint state_meta = state_dual_meta_base(p, gid);
  metadata[state_meta] = constraints;
  metadata[state_meta + 1u] = left_free;
  metadata[state_meta + 2u] = right_free;
  const uint ccap = state_constraint_capacity(p);
  device float *parameter_offset = state_dual_offset(workspace, p, gid);
  device float *parameter_left = state_dual_left(workspace, p, gid);
  device float *parameter_right = state_dual_right(workspace, p, gid);
  for (uint i = 0; i < ccap; ++i)
    parameter_offset[i] = 0.0f;
  for (uint i = 0; i < ccap * dcap; ++i) {
    parameter_left[i] = 0.0f;
    parameter_right[i] = 0.0f;
  }
  for (int pivot = 0; pivot < constraint_rank; ++pivot) {
    const int constraint = pivot_columns[pivot];
    const float inverse_scale = 1.0f / constraint_scales[constraint];
    parameter_offset[constraint] =
        matrix[pivot * columns + columns - 1] * inverse_scale;
    for (int free = 0; free < left_free; ++free)
      parameter_left[uint(constraint) * dcap + uint(free)] =
          -matrix[pivot * columns + constraints + free] * inverse_scale;
    for (int free = 0; free < right_free; ++free)
      parameter_right[uint(constraint) * dcap + uint(free)] =
          -matrix[pivot * columns + constraints + left_free + free] *
          inverse_scale;
  }

  device float *residual_matrix = matrix + constraint_rank * columns;
  const int residual_rows = n - constraint_rank;
  rank = orthogonal_echelon(
      residual_matrix, residual_rows, columns, columns - 1, columns - 1,
      p.rank_tolerance, pivot_columns, permutation, reflector,
      kMinimumDualRelationRowScale);
  if (inconsistent_relation(residual_matrix, residual_rows, columns,
                            columns - 1, p.rank_tolerance,
                            p.consistency_tolerance)) {
    fail(metadata, p, kDeviceNumericalFailure, int(gid), 17);
    return;
  }
  if (atomic_load_explicit(
          reinterpret_cast<device atomic_int *>(metadata +
                                                 dual_scan_flag(p)),
          memory_order_relaxed) == 0)
    return;
  int eliminated_rank = 0;
  while (eliminated_rank < rank &&
         pivot_columns[eliminated_rank] < constraints)
    ++eliminated_rank;
  const int relation_rows = rank - eliminated_rank;
  clear_dual_relation(workspace, metadata, p, gid, left_free, right_free,
                      relation_rows);
  const uint relation_left = dual_relation_left_base(p, gid);
  const uint relation_right = dual_relation_right_base(p, gid);
  const uint relation_rhs = dual_relation_rhs_base(p, gid);
  for (int row = 0; row < relation_rows; ++row) {
    for (int col = 0; col < left_free; ++col)
      workspace[relation_left + uint(row) * dcap + uint(col)] =
          residual_matrix[(eliminated_rank + row) * columns +
                          constraints + col];
    for (int col = 0; col < right_free; ++col)
      workspace[relation_right + uint(row) * dcap + uint(col)] =
          residual_matrix[(eliminated_rank + row) * columns +
                          constraints + left_free + col];
    workspace[relation_rhs + uint(row)] =
        residual_matrix[(eliminated_rank + row) * columns + columns - 1];
  }
}

kernel void clqr_build_dual_relation_leaves_threadgroup_sliced(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    threadgroup float *local_float [[threadgroup(0)]],
    threadgroup int *local_int [[threadgroup(1)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]]) {
  const uint gid = group * lane_count + lane;
  if (gid >= p.stage_count || !enabled(metadata, p))
    return;
  const uint node = gid + 1u;
  const bool terminal = node == p.stage_count;
  const int n = state_dim(dimensions, node);
  const int constraints = state_constraint_dim(dimensions, p, node);
  const int left_free = dual_free_dim(metadata, p, node - 1u);
  const int right_free =
      terminal ? 0 : dual_free_dim(metadata, p, node);
  const int columns = constraints + left_free + right_free + 1;
  const uint dcap = dual_capacity(p);
  const uint state_capacity = state_constraint_capacity(p);
  const uint per_lane_float_entries =
      p.state_capacity * (state_capacity + 2u * dcap + 1u) +
      2u * p.state_capacity + state_capacity;
  const uint per_lane_float_stride =
      (per_lane_float_entries + 3u) & ~3u;
  const uint per_lane_integer_entries =
      p.state_capacity + state_capacity + 2u * dcap + 4u;
  const uint per_lane_integer_stride =
      (per_lane_integer_entries + 3u) & ~3u;
  threadgroup float *matrix =
      local_float + lane * per_lane_float_stride;
  threadgroup float *factors = matrix + n * columns;
  threadgroup float *constraint_scales = factors + n;
  threadgroup float *reflector = constraint_scales + constraints;
  threadgroup int *pivot_columns =
      local_int + lane * per_lane_integer_stride;
  threadgroup int *permutation = pivot_columns + n;
  for (int i = 0; i < n * columns; ++i)
    matrix[i] = 0.0f;

  const uint nx = p.state_capacity;
  const uint nu = p.control_capacity;
  const uint nc = p.mixed_capacity;
  const uint ne = p.state_constraint_capacity;
  const uint E = terminal ? p.input_terminal_E
                          : p.input_E + node * ne * nx;
  for (int constraint = 0; constraint < constraints; ++constraint) {
    float scale = 0.0f;
    for (int state = 0; state < n; ++state)
      scale = fmax(scale, fabs(input[E + uint(constraint) * nx +
                                      uint(state)]));
    constraint_scales[constraint] = scale > 0.0f ? scale : 1.0f;
  }
  for (int state = 0; state < n; ++state)
    for (int constraint = 0; constraint < constraints; ++constraint)
      matrix[state * columns + constraint] =
          input[E + uint(constraint) * nx + uint(state)] /
          constraint_scales[constraint];

  device const float *left_basis =
      dual_basis(workspace, p, node - 1u);
  for (int state = 0; state < n; ++state)
    for (int free = 0; free < left_free; ++free)
      matrix[state * columns + constraints + free] =
          left_basis[uint(state) * dcap + uint(free)];

  if (!terminal) {
    const int next_n = state_dim(dimensions, node + 1u);
    const int mixed = mixed_dim(dimensions, p, node);
    device const float *right_basis = dual_basis(workspace, p, node);
    const uint A = p.input_A + node * nx * nx;
    const uint C = p.input_C + node * nc * nx;
    for (int state = 0; state < n; ++state)
      for (int free = 0; free < right_free; ++free) {
        float value = 0.0f;
        for (int next_state = 0; next_state < next_n; ++next_state)
          value = fma(-input[A + uint(next_state) * nx + uint(state)],
                      right_basis[uint(next_state) * dcap + uint(free)],
                      value);
        for (int constraint = 0; constraint < mixed; ++constraint)
          value = fma(input[C + uint(constraint) * nx + uint(state)],
                      right_basis[uint(next_n + constraint) * dcap +
                                  uint(free)],
                      value);
        matrix[state * columns + constraints + left_free + free] = value;
      }
  }

  device const float *x = output + p.output_states + node * nx;
  device const float *left_offset =
      dual_offset(workspace, p, node - 1u);
  for (int row = 0; row < n; ++row) {
    float rhs = -left_offset[row];
    if (terminal) {
      const uint Q = p.input_Q + node * nx * nx;
      const uint q = p.input_q + node * nx;
      rhs -= input[q + uint(row)];
      for (int col = 0; col < n; ++col)
        rhs = fma(-input[Q + uint(row) * nx + uint(col)], x[col], rhs);
    } else {
      const int m = control_dim(dimensions, p, node);
      const int next_n = state_dim(dimensions, node + 1u);
      const int mixed = mixed_dim(dimensions, p, node);
      const uint Q = p.input_Q + node * nx * nx;
      const uint q = p.input_q + node * nx;
      const uint M = p.input_M + node * nx * nu;
      const uint A = p.input_A + node * nx * nx;
      const uint C = p.input_C + node * nc * nx;
      device const float *u = output + p.output_controls + node * nu;
      device const float *right_offset =
          dual_offset(workspace, p, node);
      rhs -= input[q + uint(row)];
      for (int col = 0; col < n; ++col)
        rhs = fma(-input[Q + uint(row) * nx + uint(col)], x[col], rhs);
      for (int col = 0; col < m; ++col)
        rhs = fma(-input[M + uint(row) * nu + uint(col)], u[col], rhs);
      for (int next_state = 0; next_state < next_n; ++next_state)
        rhs = fma(input[A + uint(next_state) * nx + uint(row)],
                  right_offset[next_state], rhs);
      for (int constraint = 0; constraint < mixed; ++constraint)
        rhs = fma(-input[C + uint(constraint) * nx + uint(row)],
                  right_offset[next_n + constraint], rhs);
    }
    matrix[row * columns + columns - 1] = rhs;
  }

  int rank = orthogonal_echelon_threadgroup(
      matrix, n, columns, constraints, columns - 1, p.rank_tolerance,
      pivot_columns, permutation, reflector, kMinimumDualRelationRowScale);
  const int constraint_rank = rank;
  const uint state_meta = state_dual_meta_base(p, gid);
  metadata[state_meta] = constraints;
  metadata[state_meta + 1u] = left_free;
  metadata[state_meta + 2u] = right_free;
  const uint ccap = state_constraint_capacity(p);
  device float *parameter_offset = state_dual_offset(workspace, p, gid);
  device float *parameter_left = state_dual_left(workspace, p, gid);
  device float *parameter_right = state_dual_right(workspace, p, gid);
  for (uint i = 0; i < ccap; ++i)
    parameter_offset[i] = 0.0f;
  for (uint i = 0; i < ccap * dcap; ++i) {
    parameter_left[i] = 0.0f;
    parameter_right[i] = 0.0f;
  }
  for (int pivot = 0; pivot < constraint_rank; ++pivot) {
    const int constraint = pivot_columns[pivot];
    const float inverse_scale = 1.0f / constraint_scales[constraint];
    parameter_offset[constraint] =
        matrix[pivot * columns + columns - 1] * inverse_scale;
    for (int free = 0; free < left_free; ++free)
      parameter_left[uint(constraint) * dcap + uint(free)] =
          -matrix[pivot * columns + constraints + free] * inverse_scale;
    for (int free = 0; free < right_free; ++free)
      parameter_right[uint(constraint) * dcap + uint(free)] =
          -matrix[pivot * columns + constraints + left_free + free] *
          inverse_scale;
  }

  threadgroup float *residual_matrix = matrix + constraint_rank * columns;
  const int residual_rows = n - constraint_rank;
  rank = orthogonal_echelon_threadgroup(
      residual_matrix, residual_rows, columns, columns - 1, columns - 1,
      p.rank_tolerance, pivot_columns, permutation, reflector,
      kMinimumDualRelationRowScale);
  if (inconsistent_relation_threadgroup(residual_matrix, residual_rows, columns,
                            columns - 1, p.rank_tolerance,
                            p.consistency_tolerance)) {
    fail(metadata, p, kDeviceNumericalFailure, int(gid), 17);
    return;
  }
  if (atomic_load_explicit(
          reinterpret_cast<device atomic_int *>(metadata +
                                                 dual_scan_flag(p)),
          memory_order_relaxed) == 0)
    return;
  int eliminated_rank = 0;
  while (eliminated_rank < rank &&
         pivot_columns[eliminated_rank] < constraints)
    ++eliminated_rank;
  const int relation_rows = rank - eliminated_rank;
  clear_dual_relation(workspace, metadata, p, gid, left_free, right_free,
                      relation_rows);
  const uint relation_left = dual_relation_left_base(p, gid);
  const uint relation_right = dual_relation_right_base(p, gid);
  const uint relation_rhs = dual_relation_rhs_base(p, gid);
  for (int row = 0; row < relation_rows; ++row) {
    for (int col = 0; col < left_free; ++col)
      workspace[relation_left + uint(row) * dcap + uint(col)] =
          residual_matrix[(eliminated_rank + row) * columns +
                          constraints + col];
    for (int col = 0; col < right_free; ++col)
      workspace[relation_right + uint(row) * dcap + uint(col)] =
          residual_matrix[(eliminated_rank + row) * columns +
                          constraints + left_free + col];
    workspace[relation_rhs + uint(row)] =
        residual_matrix[(eliminated_rank + row) * columns + columns - 1];
  }
}


kernel void clqr_reduce_dual_relations(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  (void)dimensions;
  (void)input;
  (void)output;
  if (gid >= p.parent_count || !enabled(metadata, p) ||
      atomic_load_explicit(
          reinterpret_cast<device atomic_int *>(metadata +
                                                 dual_scan_flag(p)),
          memory_order_relaxed) == 0)
    return;
  const uint left = p.child_offset + 2u * gid;
  const uint parent = p.parent_offset + gid;
  if (2u * gid + 1u >= p.child_count) {
    copy_dual_relation(workspace, metadata, p, left, parent);
    return;
  }
  compose_dual_relations(workspace, metadata, p, left, left + 1u, parent, gid,
                         int(gid), 18);
}

kernel void clqr_solve_dual_root(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  (void)dimensions;
  (void)input;
  (void)output;
  if (gid != 0u || !enabled(metadata, p) ||
      atomic_load_explicit(
          reinterpret_cast<device atomic_int *>(metadata +
                                                 dual_scan_flag(p)),
          memory_order_relaxed) == 0)
    return;
  const uint relation_slot = p.child_offset;
  const uint value_slot = p.parent_offset;
  const int left_dim =
      dual_relation_left_dim(metadata, p, relation_slot);
  const int right_dim =
      dual_relation_right_dim(metadata, p, relation_slot);
  const int rows = dual_relation_rows(metadata, p, relation_slot);
  if (right_dim != 0) {
    fail(metadata, p, kDeviceNumericalFailure, 0, 13);
    return;
  }
  const int columns = left_dim + 1;
  device float *matrix = scratch(workspace, p, 0);
  device float *residual_rhs = matrix + rows * columns;
  device float *upper = residual_rhs + rows;
  device float *rhs_projection = upper + left_dim * left_dim;
  device float *solution = rhs_projection + left_dim;
  device int *permutation = integer_scratch(metadata, p, 0);
  const uint d = dual_capacity(p);
  const uint left = dual_relation_left_base(p, relation_slot);
  const uint rhs = dual_relation_rhs_base(p, relation_slot);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < left_dim; ++col)
      matrix[row * columns + col] =
          workspace[left + uint(row) * d + uint(col)];
    matrix[row * columns + left_dim] = workspace[rhs + uint(row)];
  }
  const float rhs_scale = conditioned_rhs_scale(
      matrix, rows, columns, left_dim, p.rank_tolerance);
  int rank = 0;
  if (!solve_system_orthogonally(
          matrix, rows, columns, left_dim, p.rank_tolerance,
          p.rank_tolerance, rhs_scale, residual_rhs, upper, rhs_projection,
          solution, permutation, rank)) {
    fail(metadata, p, kDeviceNumericalFailure, 0, 14);
    return;
  }
  const uint node_meta = dual_node_meta_base(p, value_slot);
  metadata[node_meta] = left_dim;
  metadata[node_meta + 1u] = 0;
  device float *node_left = dual_node_left(workspace, p, value_slot);
  device float *node_right = dual_node_right(workspace, p, value_slot);
  for (uint i = 0; i < d; ++i) {
    node_left[i] = i < uint(left_dim) ? solution[i] : 0.0f;
    node_right[i] = 0.0f;
  }
}

kernel void clqr_expand_dual_relations(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  (void)dimensions;
  (void)input;
  (void)output;
  if (gid >= p.parent_count || !enabled(metadata, p) ||
      atomic_load_explicit(
          reinterpret_cast<device atomic_int *>(metadata +
                                                 dual_scan_flag(p)),
          memory_order_relaxed) == 0)
    return;
  const uint left_slot = p.child_offset + 2u * gid;
  const uint parent_slot = p.parent_offset + gid;
  const uint d = dual_capacity(p);
  const uint parent_meta = dual_node_meta_base(p, parent_slot);
  const int parent_left_dim = metadata[parent_meta];
  const int parent_right_dim = metadata[parent_meta + 1u];
  device const float *parent_left =
      dual_node_left(workspace, p, parent_slot);
  device const float *parent_right =
      dual_node_right(workspace, p, parent_slot);
  if (2u * gid + 1u >= p.child_count) {
    const uint child_meta = dual_node_meta_base(p, left_slot);
    metadata[child_meta] = parent_left_dim;
    metadata[child_meta + 1u] = parent_right_dim;
    device float *child_left = dual_node_left(workspace, p, left_slot);
    device float *child_right = dual_node_right(workspace, p, left_slot);
    for (uint i = 0; i < d; ++i) {
      child_left[i] =
          i < uint(parent_left_dim) ? parent_left[i] : 0.0f;
      child_right[i] =
          i < uint(parent_right_dim) ? parent_right[i] : 0.0f;
    }
    return;
  }

  const uint right_slot = left_slot + 1u;
  const int left_left_dim =
      dual_relation_left_dim(metadata, p, left_slot);
  const int shared = dual_relation_right_dim(metadata, p, left_slot);
  const int left_rows = dual_relation_rows(metadata, p, left_slot);
  const int right_left_dim =
      dual_relation_left_dim(metadata, p, right_slot);
  const int right_right_dim =
      dual_relation_right_dim(metadata, p, right_slot);
  const int right_rows = dual_relation_rows(metadata, p, right_slot);
  if (left_left_dim != parent_left_dim ||
      right_right_dim != parent_right_dim || shared != right_left_dim) {
    fail(metadata, p, kDeviceNumericalFailure, int(gid), 15);
    return;
  }
  const int rows = left_rows + right_rows;
  const int columns = shared + 1;
  device float *matrix = scratch(workspace, p, gid);
  device float *residual_rhs = matrix + rows * columns;
  device float *upper = residual_rhs + rows;
  device float *rhs_projection = upper + shared * shared;
  device float *solution = rhs_projection + shared;
  device int *permutation = integer_scratch(metadata, p, gid);
  for (int i = 0; i < rows * columns; ++i)
    matrix[i] = 0.0f;
  const uint left_coeff = dual_relation_left_base(p, left_slot);
  const uint left_shared = dual_relation_right_base(p, left_slot);
  const uint left_rhs = dual_relation_rhs_base(p, left_slot);
  const uint right_shared = dual_relation_left_base(p, right_slot);
  const uint right_coeff = dual_relation_right_base(p, right_slot);
  const uint right_rhs = dual_relation_rhs_base(p, right_slot);
  for (int row = 0; row < left_rows; ++row) {
    for (int col = 0; col < shared; ++col)
      matrix[row * columns + col] =
          workspace[left_shared + uint(row) * d + uint(col)];
    float rhs = workspace[left_rhs + uint(row)];
    for (int col = 0; col < left_left_dim; ++col)
      rhs -= workspace[left_coeff + uint(row) * d + uint(col)] *
             parent_left[col];
    matrix[row * columns + shared] = rhs;
  }
  for (int row = 0; row < right_rows; ++row) {
    const int output_row = left_rows + row;
    for (int col = 0; col < shared; ++col)
      matrix[output_row * columns + col] =
          workspace[right_shared + uint(row) * d + uint(col)];
    float rhs = workspace[right_rhs + uint(row)];
    for (int col = 0; col < right_right_dim; ++col)
      rhs -= workspace[right_coeff + uint(row) * d + uint(col)] *
             parent_right[col];
    matrix[output_row * columns + shared] = rhs;
  }
  const float rhs_scale = conditioned_rhs_scale(
      matrix, rows, columns, shared, p.rank_tolerance);
  int rank = 0;
  if (!solve_system_orthogonally(
          matrix, rows, columns, shared, p.rank_tolerance,
          p.consistency_tolerance, rhs_scale, residual_rhs, upper,
          rhs_projection, solution, permutation, rank)) {
    fail(metadata, p, kDeviceNumericalFailure, int(gid), 16);
    return;
  }
  const uint left_meta = dual_node_meta_base(p, left_slot);
  const uint right_meta = dual_node_meta_base(p, right_slot);
  metadata[left_meta] = left_left_dim;
  metadata[left_meta + 1u] = shared;
  metadata[right_meta] = shared;
  metadata[right_meta + 1u] = right_right_dim;
  device float *left_value_left =
      dual_node_left(workspace, p, left_slot);
  device float *left_value_right =
      dual_node_right(workspace, p, left_slot);
  device float *right_value_left =
      dual_node_left(workspace, p, right_slot);
  device float *right_value_right =
      dual_node_right(workspace, p, right_slot);
  for (uint i = 0; i < d; ++i) {
    left_value_left[i] =
        i < uint(left_left_dim) ? parent_left[i] : 0.0f;
    right_value_right[i] =
        i < uint(right_right_dim) ? parent_right[i] : 0.0f;
    const float shared_value = i < uint(shared) ? solution[i] : 0.0f;
    left_value_right[i] = shared_value;
    right_value_left[i] = shared_value;
  }
}

kernel void clqr_recover_parameterized_multipliers(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  (void)dimensions;
  (void)input;
  if (gid >= p.stage_count || !enabled(metadata, p))
    return;
  const uint nx = p.state_capacity;
  const uint nc = p.mixed_capacity;
  const uint ne = p.state_constraint_capacity;
  const uint d = dual_capacity(p);
  const int state_variables = dual_state_dim(metadata, p, gid);
  const int physical = dual_physical_dim(metadata, p, gid);
  const int free_dim = dual_free_dim(metadata, p, gid);
  device const float *basis = dual_basis(workspace, p, gid);
  device const float *offset = dual_offset(workspace, p, gid);
  device const float *leaf_left = dual_node_left(workspace, p, gid);
  for (int row = 0; row < physical; ++row) {
    float result = offset[row];
    for (int free = 0; free < free_dim; ++free)
      result += basis[uint(row) * d + uint(free)] * leaf_left[free];
    if (row < state_variables)
      output[p.output_dynamics_multipliers + gid * nx + uint(row)] =
          result;
    else
      output[p.output_mixed_multipliers + gid * nc +
             uint(row - state_variables)] = result;
  }

  const uint node = gid + 1u;
  const uint state_meta = state_dual_meta_base(p, gid);
  const int constraints = metadata[state_meta];
  const int left_dim = metadata[state_meta + 1u];
  const int right_dim = metadata[state_meta + 2u];
  device const float *parameter_offset =
      state_dual_offset(workspace, p, gid);
  device const float *parameter_left =
      state_dual_left(workspace, p, gid);
  device const float *parameter_right =
      state_dual_right(workspace, p, gid);
  device const float *leaf_right = dual_node_right(workspace, p, gid);
  for (int constraint = 0; constraint < constraints; ++constraint) {
    float multiplier = parameter_offset[constraint];
    for (int free = 0; free < left_dim; ++free)
      multiplier +=
          parameter_left[uint(constraint) * d + uint(free)] *
          leaf_left[free];
    for (int free = 0; free < right_dim; ++free)
      multiplier +=
          parameter_right[uint(constraint) * d + uint(free)] *
          leaf_right[free];
    if (node == p.stage_count)
      output[p.output_terminal_state_multiplier + uint(constraint)] =
          multiplier;
    else
      output[p.output_state_multipliers + node * ne +
             uint(constraint)] = multiplier;
  }
}

kernel void clqr_recover_initial_multiplier(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  (void)workspace;
  if (gid != 0u || !enabled(metadata, p))
    return;
  const uint nx = p.state_capacity;
  const uint nu = p.control_capacity;
  const uint nc = p.mixed_capacity;
  const uint ne = p.state_constraint_capacity;
  if (p.stage_count == 0u) {
    const int n = state_dim(dimensions, 0u);
    const uint Q = p.input_Q;
    const uint q = p.input_q;
    device const float *x = output + p.output_states;
    for (int row = 0; row < n; ++row) {
      float value = -input[q + uint(row)];
      for (int col = 0; col < n; ++col)
        value -= input[Q + uint(row) * nx + uint(col)] * x[col];
      output[p.output_initial_multiplier + uint(row)] = value;
    }
    return;
  }
  const int n = state_dim(dimensions, 0u);
  const int next_n = state_dim(dimensions, 1u);
  const int m = control_dim(dimensions, p, 0u);
  const int mixed = mixed_dim(dimensions, p, 0u);
  const uint Q = p.input_Q;
  const uint q = p.input_q;
  const uint M = p.input_M;
  const uint A = p.input_A;
  const uint C = p.input_C;
  device const float *x = output + p.output_states;
  device const float *u = output + p.output_controls;
  device const float *dynamics =
      output + p.output_dynamics_multipliers;
  device const float *mixed_multipliers =
      output + p.output_mixed_multipliers;
  for (int row = 0; row < n; ++row) {
    float value = -input[q + uint(row)];
    for (int col = 0; col < n; ++col)
      value -= input[Q + uint(row) * nx + uint(col)] * x[col];
    for (int col = 0; col < m; ++col)
      value -= input[M + uint(row) * nu + uint(col)] * u[col];
    for (int col = 0; col < next_n; ++col)
      value += input[A + uint(col) * nx + uint(row)] * dynamics[col];
    for (int constraint = 0; constraint < mixed; ++constraint)
      value -= input[C + uint(constraint) * nx + uint(row)] *
               mixed_multipliers[constraint];
    output[p.output_initial_multiplier + uint(row)] = value;
  }
  // The initial state equality is represented by initial_multiplier, so its
  // separate state-constraint multiplier is the canonical zero solution.
  const int initial_constraints = state_constraint_dim(dimensions, p, 0u);
  for (int row = 0; row < initial_constraints; ++row)
    output[p.output_state_multipliers + uint(row)] = 0.0f;
  (void)ne;
}

kernel void clqr_build_objective_terms(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid > p.stage_count || !enabled(metadata, p))
    return;
  const uint nx = p.state_capacity;
  const uint nu = p.control_capacity;
  const int n = state_dim(dimensions, gid);
  device const float *x = output + p.output_states + gid * nx;
  const uint Q = p.input_Q + gid * nx * nx;
  const uint q = p.input_q + gid * nx;
  float objective = 0.0f;
  for (int row = 0; row < n; ++row)
    objective += input[q + uint(row)] * x[row];
  for (int row = 0; row < n; ++row)
    for (int col = 0; col < n; ++col)
      objective += 0.5f * x[row] *
                   input[Q + uint(row) * nx + uint(col)] * x[col];
  if (gid < p.stage_count) {
    const int m = control_dim(dimensions, p, gid);
    device const float *u = output + p.output_controls + gid * nu;
    const uint R = p.input_R + gid * nu * nu;
    const uint M = p.input_M + gid * nx * nu;
    const uint r = p.input_r + gid * nu;
    for (int row = 0; row < n; ++row)
      for (int col = 0; col < m; ++col)
        objective += x[row] *
                     input[M + uint(row) * nu + uint(col)] * u[col];
    for (int row = 0; row < m; ++row)
      objective += input[r + uint(row)] * u[row];
    for (int row = 0; row < m; ++row)
      for (int col = 0; col < m; ++col)
        objective += 0.5f * u[row] *
                     input[R + uint(row) * nu + uint(col)] * u[col];
  }
  workspace[p.objective_tree + gid] = objective;
}

kernel void clqr_reduce_objective(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  (void)dimensions;
  (void)input;
  (void)output;
  if (gid >= p.parent_count || !enabled(metadata, p))
    return;
  const uint left = 2u * gid;
  float value = workspace[p.objective_tree + p.child_offset + left];
  if (left + 1u < p.child_count)
    value += workspace[p.objective_tree + p.child_offset + left + 1u];
  workspace[p.objective_tree + p.parent_offset + gid] = value;
}

kernel void clqr_finalize_objective(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  (void)dimensions;
  (void)input;
  if (gid == 0u && enabled(metadata, p))
    output[p.output_objective] =
        workspace[p.objective_tree + p.child_offset];
}
)CLQR_METAL";

} // namespace clqr::metal::detail

#endif // CLQR_SRC_METAL_DUAL_OBJECTIVE_SOURCE_H_
