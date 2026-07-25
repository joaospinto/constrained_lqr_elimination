#ifndef CLQR_SRC_METAL_PRIMAL_SCAN_SOURCE_H_
#define CLQR_SRC_METAL_PRIMAL_SCAN_SOURCE_H_

namespace clqr::metal::detail {

inline constexpr char kMetalPrimalScanSource[] = R"CLQR_METAL(
#include <metal_stdlib>

using namespace metal;

// This file deliberately contains no device pointers in persistent metadata.
// Every matrix is addressed by a 32-bit element offset into one of the four
// arenas supplied by the host.  Besides making the layout ABI-stable, this
// avoids rebasing pointers when a reusable workspace grows.

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

enum DeviceCode : int {
  kDeviceOk = 0,
  kDeviceInfeasible = 1,
  kDeviceNumericalFailure = 2,
  kDeviceInvalidInput = 3,
};

inline uint relation_rows_capacity(constant KernelParams &p) {
  return 2u * p.state_capacity;
}

inline uint relation_left_offset(constant KernelParams &p, uint slot) {
  return p.relation_data + slot * p.relation_stride;
}

inline uint relation_right_offset(constant KernelParams &p, uint slot) {
  return relation_left_offset(p, slot) +
         relation_rows_capacity(p) * p.state_capacity;
}

inline uint relation_rhs_offset(constant KernelParams &p, uint slot) {
  return relation_right_offset(p, slot) +
         relation_rows_capacity(p) * p.state_capacity;
}

inline int relation_left_dim(device const int *metadata,
                             constant KernelParams &p, uint slot) {
  return metadata[p.relation_meta + 3u * slot];
}

inline int relation_right_dim(device const int *metadata,
                              constant KernelParams &p, uint slot) {
  return metadata[p.relation_meta + 3u * slot + 1u];
}

inline int relation_rows(device const int *metadata, constant KernelParams &p,
                         uint slot) {
  return metadata[p.relation_meta + 3u * slot + 2u];
}

inline void set_relation_shape(device int *metadata, constant KernelParams &p,
                               uint slot, int left_dim, int right_dim,
                               int rows) {
  metadata[p.relation_meta + 3u * slot] = left_dim;
  metadata[p.relation_meta + 3u * slot + 1u] = right_dim;
  metadata[p.relation_meta + 3u * slot + 2u] = rows;
}

inline bool invalid_relation(device const int *metadata,
                             constant KernelParams &p, uint slot) {
  return relation_left_dim(metadata, p, slot) < 0;
}

inline bool status_ok(device int *metadata, constant KernelParams &p) {
  device atomic_int *code =
      reinterpret_cast<device atomic_int *>(metadata + p.status);
  return atomic_load_explicit(code, memory_order_relaxed) == kDeviceOk;
}

inline void set_failure(device int *metadata, constant KernelParams &p,
                        int code_value, int stage, int detail) {
  device atomic_int *code =
      reinterpret_cast<device atomic_int *>(metadata + p.status);
  int expected = kDeviceOk;
  bool claimed = false;
  while (expected == kDeviceOk &&
         !(claimed = atomic_compare_exchange_weak_explicit(
               code, &expected, code_value, memory_order_relaxed,
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

// Row-equilibrated RREF with deterministic partial row pivoting.  The pivot
// convention and strict comparison match the CUDA implementation.
inline int rref_serial(device float *matrix, int rows, int columns,
                       int pivot_limit, float tolerance,
                       device int *pivot_columns) {
  for (int row = 0; row < rows; ++row) {
    float scale = 0.0f;
    for (int col = 0; col < pivot_limit; ++col)
      scale = fmax(scale, fabs(matrix[row * columns + col]));
    if (scale > 0.0f) {
      for (int col = 0; col < columns; ++col)
        matrix[row * columns + col] /= scale;
    }
  }

  int rank = 0;
  for (int col = 0; col < pivot_limit && rank < rows; ++col) {
    int best_row = -1;
    float best = tolerance;
    for (int row = rank; row < rows; ++row) {
      const float candidate = fabs(matrix[row * columns + col]);
      if (candidate > best) {
        best = candidate;
        best_row = row;
      }
    }
    if (best_row < 0)
      continue;

    if (best_row != rank) {
      for (int j = 0; j < columns; ++j) {
        const float tmp = matrix[rank * columns + j];
        matrix[rank * columns + j] = matrix[best_row * columns + j];
        matrix[best_row * columns + j] = tmp;
      }
    }

    const float pivot = matrix[rank * columns + col];
    for (int j = col; j < columns; ++j)
      matrix[rank * columns + j] /= pivot;
    for (int row = 0; row < rows; ++row) {
      if (row == rank)
        continue;
      const float factor = matrix[row * columns + col];
      if (factor == 0.0f)
        continue;
      for (int j = col; j < columns; ++j)
        matrix[row * columns + j] -=
            factor * matrix[rank * columns + j];
    }
    pivot_columns[rank++] = col;
  }

  for (int index = 0; index < rows * columns; ++index) {
    if (fabs(matrix[index]) <= tolerance)
      matrix[index] = 0.0f;
  }
  return rank;
}

inline bool inconsistent_rref(device const float *matrix, int rows,
                              int columns, int lhs_columns,
                              float lhs_tolerance, float rhs_tolerance) {
  for (int row = 0; row < rows; ++row) {
    bool zero = true;
    for (int col = 0; col < lhs_columns; ++col) {
      if (fabs(matrix[row * columns + col]) > lhs_tolerance) {
        zero = false;
        break;
      }
    }
    if (zero &&
        fabs(matrix[row * columns + lhs_columns]) > rhs_tolerance)
      return true;
  }
  return false;
}

inline void set_invalid_relation(device int *metadata,
                                 constant KernelParams &p, uint slot) {
  set_relation_shape(metadata, p, slot, -1, 0, 0);
}

inline void extract_residual_relation(
    device const float *matrix, int columns, int rank,
    device const int *pivot_columns, int eliminated_columns, int left_dim,
    int right_dim, device float *workspace, device int *metadata,
    constant KernelParams &p, uint output_slot) {
  int eliminated_rank = 0;
  while (eliminated_rank < rank &&
         pivot_columns[eliminated_rank] < eliminated_columns)
    ++eliminated_rank;
  const int output_rows = rank - eliminated_rank;
  set_relation_shape(metadata, p, output_slot, left_dim, right_dim,
                     output_rows);
  const uint left = relation_left_offset(p, output_slot);
  const uint right = relation_right_offset(p, output_slot);
  const uint rhs = relation_rhs_offset(p, output_slot);
  for (int row = 0; row < output_rows; ++row) {
    for (int col = 0; col < left_dim; ++col) {
      workspace[left + uint(row) * p.state_capacity + uint(col)] =
          matrix[(eliminated_rank + row) * columns +
                 eliminated_columns + col];
    }
    for (int col = 0; col < right_dim; ++col) {
      workspace[right + uint(row) * p.state_capacity + uint(col)] =
          matrix[(eliminated_rank + row) * columns +
                 eliminated_columns + left_dim + col];
    }
    workspace[rhs + uint(row)] =
        matrix[(eliminated_rank + row) * columns + columns - 1];
  }
}

// Address-space specializations for one independent scalar leaf per SIMD
// lane. Each lane owns a disjoint threadgroup-memory slice, so every pivot
// decision and arithmetic operation follows clqr_build_primal_leaves exactly.
inline int rref_serial_threadgroup(
    threadgroup float *matrix, int rows, int columns, int pivot_limit,
    float tolerance, threadgroup int *pivot_columns) {
  for (int row = 0; row < rows; ++row) {
    float scale = 0.0f;
    for (int col = 0; col < pivot_limit; ++col)
      scale = fmax(scale, fabs(matrix[row * columns + col]));
    if (scale > 0.0f) {
      for (int col = 0; col < columns; ++col)
        matrix[row * columns + col] /= scale;
    }
  }

  int rank = 0;
  for (int col = 0; col < pivot_limit && rank < rows; ++col) {
    int best_row = -1;
    float best = tolerance;
    for (int row = rank; row < rows; ++row) {
      const float candidate = fabs(matrix[row * columns + col]);
      if (candidate > best) {
        best = candidate;
        best_row = row;
      }
    }
    if (best_row < 0)
      continue;

    if (best_row != rank) {
      for (int j = 0; j < columns; ++j) {
        const float tmp = matrix[rank * columns + j];
        matrix[rank * columns + j] = matrix[best_row * columns + j];
        matrix[best_row * columns + j] = tmp;
      }
    }

    const float pivot = matrix[rank * columns + col];
    for (int j = col; j < columns; ++j)
      matrix[rank * columns + j] /= pivot;
    for (int row = 0; row < rows; ++row) {
      if (row == rank)
        continue;
      const float factor = matrix[row * columns + col];
      if (factor == 0.0f)
        continue;
      for (int j = col; j < columns; ++j)
        matrix[row * columns + j] -=
            factor * matrix[rank * columns + j];
    }
    pivot_columns[rank++] = col;
  }

  for (int index = 0; index < rows * columns; ++index) {
    if (fabs(matrix[index]) <= tolerance)
      matrix[index] = 0.0f;
  }
  return rank;
}

inline bool inconsistent_rref_threadgroup(
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
    if (zero &&
        fabs(matrix[row * columns + lhs_columns]) > rhs_tolerance)
      return true;
  }
  return false;
}

inline void extract_residual_relation_threadgroup(
    threadgroup const float *matrix, int columns, int rank,
    threadgroup const int *pivot_columns, int eliminated_columns, int left_dim,
    int right_dim, device float *workspace, device int *metadata,
    constant KernelParams &p, uint output_slot) {
  int eliminated_rank = 0;
  while (eliminated_rank < rank &&
         pivot_columns[eliminated_rank] < eliminated_columns)
    ++eliminated_rank;
  const int output_rows = rank - eliminated_rank;
  set_relation_shape(metadata, p, output_slot, left_dim, right_dim,
                     output_rows);
  const uint left = relation_left_offset(p, output_slot);
  const uint right = relation_right_offset(p, output_slot);
  const uint rhs = relation_rhs_offset(p, output_slot);
  for (int row = 0; row < output_rows; ++row) {
    for (int col = 0; col < left_dim; ++col) {
      workspace[left + uint(row) * p.state_capacity + uint(col)] =
          matrix[(eliminated_rank + row) * columns +
                 eliminated_columns + col];
    }
    for (int col = 0; col < right_dim; ++col) {
      workspace[right + uint(row) * p.state_capacity + uint(col)] =
          matrix[(eliminated_rank + row) * columns +
                 eliminated_columns + left_dim + col];
    }
    workspace[rhs + uint(row)] =
        matrix[(eliminated_rank + row) * columns + columns - 1];
  }
}

// The balanced primal reduction is the highest-volume feasibility operation.
// Give each relation composition one complete 32-lane SIMDgroup.  Reductions
// that affect rank or pivot selection deliberately remain on lane zero, while
// independent rows and matrix entries are distributed across the other lanes.
// Consequently every ordered inner arithmetic loop is identical to
// rref_serial; only independent work is concurrent.
constant uint kPrimalRelationSimdWidth = 32u;

inline int rref_cooperative(
    device float *matrix, int rows, int columns, int pivot_limit,
    float tolerance, device int *pivot_columns, uint lane,
    threadgroup int *shared_state) {
  const int lane_index = int(lane);
  const int lane_count = int(kPrimalRelationSimdWidth);
  for (int row = lane_index; row < rows; row += lane_count) {
    float scale = 0.0f;
    for (int col = 0; col < pivot_limit; ++col)
      scale = fmax(scale, fabs(matrix[row * columns + col]));
    if (scale > 0.0f) {
      for (int col = 0; col < columns; ++col)
        matrix[row * columns + col] /= scale;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  if (lane == 0u)
    shared_state[0] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int col = 0; col < pivot_limit; ++col) {
    if (lane == 0u) {
      const int rank = shared_state[0];
      int best_row = -2;
      if (rank < rows) {
        best_row = -1;
        float best = tolerance;
        for (int row = rank; row < rows; ++row) {
          const float candidate = fabs(matrix[row * columns + col]);
          if (candidate > best) {
            best = candidate;
            best_row = row;
          }
        }
      }
      shared_state[1] = best_row;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int rank = shared_state[0];
    const int best_row = shared_state[1];
    if (best_row == -2)
      break;
    if (best_row < 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      continue;
    }

    if (best_row != rank) {
      for (int j = lane_index; j < columns; j += lane_count) {
        const float tmp = matrix[rank * columns + j];
        matrix[rank * columns + j] = matrix[best_row * columns + j];
        matrix[best_row * columns + j] = tmp;
      }
    }
    threadgroup_barrier(mem_flags::mem_device);

    const float pivot = matrix[rank * columns + col];
    for (int j = col + lane_index; j < columns; j += lane_count)
      matrix[rank * columns + j] /= pivot;
    threadgroup_barrier(mem_flags::mem_device);

    for (int row = lane_index; row < rows; row += lane_count) {
      if (row == rank)
        continue;
      const float factor = matrix[row * columns + col];
      if (factor == 0.0f)
        continue;
      for (int j = col; j < columns; ++j)
        matrix[row * columns + j] -=
            factor * matrix[rank * columns + j];
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (lane == 0u) {
      pivot_columns[rank] = col;
      shared_state[0] = rank + 1;
    }
    threadgroup_barrier(mem_flags::mem_device |
                        mem_flags::mem_threadgroup);
  }

  for (int index = lane_index; index < rows * columns;
       index += lane_count) {
    if (fabs(matrix[index]) <= tolerance)
      matrix[index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_device);
  return shared_state[0];
}

inline void copy_relation_cooperative(
    device float *workspace, device int *metadata, constant KernelParams &p,
    uint input_slot, uint output_slot, uint lane) {
  if (input_slot == output_slot)
    return;
  const int left_dim = relation_left_dim(metadata, p, input_slot);
  const int right_dim = relation_right_dim(metadata, p, input_slot);
  const int rows = relation_rows(metadata, p, input_slot);
  if (left_dim < 0) {
    if (lane == 0u)
      set_invalid_relation(metadata, p, output_slot);
    return;
  }
  if (lane == 0u)
    set_relation_shape(metadata, p, output_slot, left_dim, right_dim, rows);
  const uint input_left = relation_left_offset(p, input_slot);
  const uint input_right = relation_right_offset(p, input_slot);
  const uint input_rhs = relation_rhs_offset(p, input_slot);
  const uint output_left = relation_left_offset(p, output_slot);
  const uint output_right = relation_right_offset(p, output_slot);
  const uint output_rhs = relation_rhs_offset(p, output_slot);
  const int lane_index = int(lane);
  const int lane_count = int(kPrimalRelationSimdWidth);
  for (int row = lane_index; row < rows; row += lane_count) {
    for (int col = 0; col < left_dim; ++col) {
      workspace[output_left + uint(row) * p.state_capacity + uint(col)] =
          workspace[input_left + uint(row) * p.state_capacity + uint(col)];
    }
    for (int col = 0; col < right_dim; ++col) {
      workspace[output_right + uint(row) * p.state_capacity + uint(col)] =
          workspace[input_right + uint(row) * p.state_capacity + uint(col)];
    }
    workspace[output_rhs + uint(row)] = workspace[input_rhs + uint(row)];
  }
}

inline void extract_residual_relation_cooperative(
    device const float *matrix, int columns, int rank,
    device const int *pivot_columns, int eliminated_columns, int left_dim,
    int right_dim, device float *workspace, device int *metadata,
    constant KernelParams &p, uint output_slot, uint lane,
    threadgroup int *shared_state) {
  if (lane == 0u) {
    int eliminated_rank = 0;
    while (eliminated_rank < rank &&
           pivot_columns[eliminated_rank] < eliminated_columns)
      ++eliminated_rank;
    shared_state[2] = eliminated_rank;
    shared_state[3] = rank - eliminated_rank;
    set_relation_shape(metadata, p, output_slot, left_dim, right_dim,
                       shared_state[3]);
  }
  threadgroup_barrier(mem_flags::mem_device |
                      mem_flags::mem_threadgroup);

  const int eliminated_rank = shared_state[2];
  const int output_rows = shared_state[3];
  const uint left = relation_left_offset(p, output_slot);
  const uint right = relation_right_offset(p, output_slot);
  const uint rhs = relation_rhs_offset(p, output_slot);
  const int lane_index = int(lane);
  const int lane_count = int(kPrimalRelationSimdWidth);
  for (int row = lane_index; row < output_rows; row += lane_count) {
    for (int col = 0; col < left_dim; ++col) {
      workspace[left + uint(row) * p.state_capacity + uint(col)] =
          matrix[(eliminated_rank + row) * columns +
                 eliminated_columns + col];
    }
    for (int col = 0; col < right_dim; ++col) {
      workspace[right + uint(row) * p.state_capacity + uint(col)] =
          matrix[(eliminated_rank + row) * columns +
                 eliminated_columns + left_dim + col];
    }
    workspace[rhs + uint(row)] =
        matrix[(eliminated_rank + row) * columns + columns - 1];
  }
}

inline bool compose_relations_cooperative(
    device float *workspace, device int *metadata, constant KernelParams &p,
    uint first_slot, uint second_slot, uint output_slot, uint scratch_index,
    int failure_stage, int failure_code, int failure_detail, uint lane,
    threadgroup int *shared_state) {
  const int first_left_dim = relation_left_dim(metadata, p, first_slot);
  const int shared = relation_right_dim(metadata, p, first_slot);
  const int first_rows = relation_rows(metadata, p, first_slot);
  const int second_left_dim = relation_left_dim(metadata, p, second_slot);
  const int second_right_dim = relation_right_dim(metadata, p, second_slot);
  const int second_rows = relation_rows(metadata, p, second_slot);
  if (shared != second_left_dim) {
    if (lane == 0u)
      set_failure(metadata, p, kDeviceNumericalFailure, failure_stage, 2);
    return false;
  }

  const int rows = first_rows + second_rows;
  const int columns = shared + first_left_dim + second_right_dim + 1;
  device float *matrix =
      workspace + p.float_scratch + scratch_index * p.float_scratch_stride;
  device int *pivot_columns =
      metadata + p.int_scratch + scratch_index * p.int_scratch_stride;
  const int lane_index = int(lane);
  const int lane_count = int(kPrimalRelationSimdWidth);
  for (int index = lane_index; index < rows * columns;
       index += lane_count)
    matrix[index] = 0.0f;
  threadgroup_barrier(mem_flags::mem_device);

  const uint first_left = relation_left_offset(p, first_slot);
  const uint first_right = relation_right_offset(p, first_slot);
  const uint first_rhs = relation_rhs_offset(p, first_slot);
  const uint second_left = relation_left_offset(p, second_slot);
  const uint second_right = relation_right_offset(p, second_slot);
  const uint second_rhs = relation_rhs_offset(p, second_slot);
  for (int row = lane_index; row < first_rows; row += lane_count) {
    for (int col = 0; col < shared; ++col) {
      matrix[row * columns + col] =
          workspace[first_right +
                    uint(row) * p.state_capacity + uint(col)];
    }
    for (int col = 0; col < first_left_dim; ++col) {
      matrix[row * columns + shared + col] =
          workspace[first_left +
                    uint(row) * p.state_capacity + uint(col)];
    }
    matrix[row * columns + columns - 1] =
        workspace[first_rhs + uint(row)];
  }
  for (int row = lane_index; row < second_rows; row += lane_count) {
    const int output_row = first_rows + row;
    for (int col = 0; col < shared; ++col) {
      matrix[output_row * columns + col] =
          workspace[second_left +
                    uint(row) * p.state_capacity + uint(col)];
    }
    for (int col = 0; col < second_right_dim; ++col) {
      matrix[output_row * columns + shared + first_left_dim + col] =
          workspace[second_right +
                    uint(row) * p.state_capacity + uint(col)];
    }
    matrix[output_row * columns + columns - 1] =
        workspace[second_rhs + uint(row)];
  }
  threadgroup_barrier(mem_flags::mem_device);

  const int rank =
      rref_cooperative(matrix, rows, columns, columns - 1, p.rank_tolerance,
                       pivot_columns, lane, shared_state);
  if (lane == 0u) {
    shared_state[1] =
        inconsistent_rref(matrix, rows, columns, columns - 1,
                          p.rank_tolerance, p.consistency_tolerance)
            ? 1
            : 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (shared_state[1] != 0) {
    if (lane == 0u)
      set_failure(metadata, p, failure_code, failure_stage, failure_detail);
    return false;
  }

  extract_residual_relation_cooperative(
      matrix, columns, rank, pivot_columns, shared, first_left_dim,
      second_right_dim, workspace, metadata, p, output_slot, lane,
      shared_state);
  return true;
}

inline bool compose_scan_relations_cooperative(
    device float *workspace, device int *metadata, constant KernelParams &p,
    uint first_slot, uint second_slot, uint output_slot, uint scratch_index,
    int failure_stage, int failure_detail, uint lane,
    threadgroup int *shared_state) {
  if (invalid_relation(metadata, p, first_slot)) {
    if (invalid_relation(metadata, p, second_slot)) {
      if (lane == 0u)
        set_invalid_relation(metadata, p, output_slot);
    } else {
      copy_relation_cooperative(workspace, metadata, p, second_slot,
                                output_slot, lane);
    }
    return true;
  }
  if (invalid_relation(metadata, p, second_slot)) {
    copy_relation_cooperative(workspace, metadata, p, first_slot, output_slot,
                              lane);
    return true;
  }
  return compose_relations_cooperative(
      workspace, metadata, p, first_slot, second_slot, output_slot,
      scratch_index, failure_stage, kDeviceInfeasible, failure_detail, lane,
      shared_state);
}

kernel void clqr_check_finite_inputs(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  (void)dimensions;
  (void)output;
  (void)workspace;
  const uint input_entries = p.input_initial_state + p.state_capacity;
  if (gid < input_entries && !isfinite(input[gid]))
    set_failure(metadata, p, kDeviceInvalidInput, -1, 21);
}

kernel void clqr_build_primal_leaves(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  (void)output;
  if (gid > p.stage_count || !status_ok(metadata, p))
    return;

  const bool terminal = gid == p.stage_count;
  const int n = dimensions[gid];
  const int next_n = terminal ? 0 : dimensions[gid + 1u];
  const int m =
      terminal ? 0 : dimensions[control_dimension_offset(p) + gid];
  const int mixed =
      terminal ? 0 : dimensions[mixed_dimension_offset(p) + gid];
  const int state_constraints =
      terminal
          ? dimensions[terminal_constraint_dimension_offset(p)]
          : dimensions[state_constraint_dimension_offset(p) + gid];
  const int rows =
      terminal ? state_constraints : mixed + state_constraints + next_n;
  const int columns = terminal ? n + 1 : m + n + next_n + 1;
  device float *matrix =
      workspace + p.float_scratch + gid * p.float_scratch_stride;
  device int *pivot_columns =
      metadata + p.int_scratch + gid * p.int_scratch_stride;
  for (int index = 0; index < rows * columns; ++index)
    matrix[index] = 0.0f;

  const uint nx = p.state_capacity;
  const uint nu = p.control_capacity;
  const uint nc = p.mixed_capacity;
  const uint ne = p.state_constraint_capacity;
  if (terminal) {
    for (int row = 0; row < state_constraints; ++row) {
      for (int col = 0; col < n; ++col) {
        matrix[row * columns + col] =
            input[p.input_terminal_E + uint(row) * nx + uint(col)];
      }
      matrix[row * columns + n] =
          -input[p.input_terminal_e + uint(row)];
    }
  } else {
    const uint stage = gid;
    const uint A = p.input_A + stage * nx * nx;
    const uint B = p.input_B + stage * nx * nu;
    const uint c = p.input_c + stage * nx;
    const uint C = p.input_C + stage * nc * nx;
    const uint D = p.input_D + stage * nc * nu;
    const uint d = p.input_d + stage * nc;
    const uint E = p.input_E + stage * ne * nx;
    const uint e = p.input_e + stage * ne;
    for (int row = 0; row < mixed; ++row) {
      for (int col = 0; col < m; ++col)
        matrix[row * columns + col] =
            input[D + uint(row) * nu + uint(col)];
      for (int col = 0; col < n; ++col)
        matrix[row * columns + m + col] =
            input[C + uint(row) * nx + uint(col)];
      matrix[row * columns + columns - 1] = -input[d + uint(row)];
    }
    for (int row = 0; row < state_constraints; ++row) {
      const int output_row = mixed + row;
      for (int col = 0; col < n; ++col)
        matrix[output_row * columns + m + col] =
            input[E + uint(row) * nx + uint(col)];
      matrix[output_row * columns + columns - 1] =
          -input[e + uint(row)];
    }
    const int dynamics_row = mixed + state_constraints;
    for (int row = 0; row < next_n; ++row) {
      for (int col = 0; col < m; ++col)
        matrix[(dynamics_row + row) * columns + col] =
            -input[B + uint(row) * nu + uint(col)];
      for (int col = 0; col < n; ++col)
        matrix[(dynamics_row + row) * columns + m + col] =
            -input[A + uint(row) * nx + uint(col)];
      matrix[(dynamics_row + row) * columns + m + n + row] = 1.0f;
      matrix[(dynamics_row + row) * columns + columns - 1] =
          input[c + uint(row)];
    }
  }

  const int rank =
      rref_serial(matrix, rows, columns, columns - 1, p.rank_tolerance,
                  pivot_columns);
  if (inconsistent_rref(matrix, rows, columns, columns - 1,
                        p.rank_tolerance, p.consistency_tolerance)) {
    set_failure(metadata, p, kDeviceInfeasible, int(gid), 1);
    return;
  }
  extract_residual_relation(matrix, columns, rank, pivot_columns, m, n,
                            next_n, workspace, metadata, p, gid);
}

kernel void clqr_build_primal_leaves_threadgroup_sliced(
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
  (void)output;
  const uint gid = group * lane_count + lane;
  if (gid > p.stage_count || !status_ok(metadata, p))
    return;

  const bool terminal = gid == p.stage_count;
  const int n = dimensions[gid];
  const int next_n = terminal ? 0 : dimensions[gid + 1u];
  const int m =
      terminal ? 0 : dimensions[control_dimension_offset(p) + gid];
  const int mixed =
      terminal ? 0 : dimensions[mixed_dimension_offset(p) + gid];
  const int state_constraints =
      terminal
          ? dimensions[terminal_constraint_dimension_offset(p)]
          : dimensions[state_constraint_dimension_offset(p) + gid];
  const int rows =
      terminal ? state_constraints : mixed + state_constraints + next_n;
  const int columns = terminal ? n + 1 : m + n + next_n + 1;
  const uint max_rows =
      max(p.mixed_capacity + p.state_constraint_capacity + p.state_capacity,
          p.terminal_constraint_capacity);
  const uint max_columns =
      p.control_capacity + 2u * p.state_capacity + 1u;
  const uint per_lane_float_entries = max_rows * max_columns;
  const uint per_lane_float_stride =
      (per_lane_float_entries + 3u) & ~3u;
  const uint per_lane_integer_entries =
      p.control_capacity + 2u * p.state_capacity;
  const uint per_lane_integer_stride =
      (per_lane_integer_entries + 3u) & ~3u;
  threadgroup float *matrix =
      local_float + lane * per_lane_float_stride;
  threadgroup int *pivot_columns =
      local_int + lane * per_lane_integer_stride;
  for (int index = 0; index < rows * columns; ++index)
    matrix[index] = 0.0f;

  const uint nx = p.state_capacity;
  const uint nu = p.control_capacity;
  const uint nc = p.mixed_capacity;
  const uint ne = p.state_constraint_capacity;
  if (terminal) {
    for (int row = 0; row < state_constraints; ++row) {
      for (int col = 0; col < n; ++col) {
        matrix[row * columns + col] =
            input[p.input_terminal_E + uint(row) * nx + uint(col)];
      }
      matrix[row * columns + n] =
          -input[p.input_terminal_e + uint(row)];
    }
  } else {
    const uint stage = gid;
    const uint A = p.input_A + stage * nx * nx;
    const uint B = p.input_B + stage * nx * nu;
    const uint c = p.input_c + stage * nx;
    const uint C = p.input_C + stage * nc * nx;
    const uint D = p.input_D + stage * nc * nu;
    const uint d = p.input_d + stage * nc;
    const uint E = p.input_E + stage * ne * nx;
    const uint e = p.input_e + stage * ne;
    for (int row = 0; row < mixed; ++row) {
      for (int col = 0; col < m; ++col)
        matrix[row * columns + col] =
            input[D + uint(row) * nu + uint(col)];
      for (int col = 0; col < n; ++col)
        matrix[row * columns + m + col] =
            input[C + uint(row) * nx + uint(col)];
      matrix[row * columns + columns - 1] = -input[d + uint(row)];
    }
    for (int row = 0; row < state_constraints; ++row) {
      const int output_row = mixed + row;
      for (int col = 0; col < n; ++col)
        matrix[output_row * columns + m + col] =
            input[E + uint(row) * nx + uint(col)];
      matrix[output_row * columns + columns - 1] =
          -input[e + uint(row)];
    }
    const int dynamics_row = mixed + state_constraints;
    for (int row = 0; row < next_n; ++row) {
      for (int col = 0; col < m; ++col)
        matrix[(dynamics_row + row) * columns + col] =
            -input[B + uint(row) * nu + uint(col)];
      for (int col = 0; col < n; ++col)
        matrix[(dynamics_row + row) * columns + m + col] =
            -input[A + uint(row) * nx + uint(col)];
      matrix[(dynamics_row + row) * columns + m + n + row] = 1.0f;
      matrix[(dynamics_row + row) * columns + columns - 1] =
          input[c + uint(row)];
    }
  }

  const int rank =
      rref_serial_threadgroup(matrix, rows, columns, columns - 1,
                              p.rank_tolerance, pivot_columns);
  if (inconsistent_rref_threadgroup(
          matrix, rows, columns, columns - 1, p.rank_tolerance,
          p.consistency_tolerance)) {
    set_failure(metadata, p, kDeviceInfeasible, int(gid), 1);
    return;
  }
  extract_residual_relation_threadgroup(
      matrix, columns, rank, pivot_columns, m, n, next_n, workspace, metadata,
      p, gid);
}

// A reduction invocation combines [child_offset + 2*i,
// child_offset + 2*i+1] into parent_offset+i.
kernel void clqr_reduce_primal_relations(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  (void)dimensions;
  (void)input;
  (void)output;
  threadgroup int shared_state[4];
  const uint gid = threadgroup_position.x;
  if (lane == 0u)
    shared_state[0] =
        gid < p.parent_count && status_ok(metadata, p) ? 1 : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (shared_state[0] == 0)
    return;
  const uint left = p.child_offset + 2u * gid;
  const uint result = p.parent_offset + gid;
  if (2u * gid + 1u >= p.child_count) {
    copy_relation_cooperative(workspace, metadata, p, left, result, lane);
    return;
  }
  compose_relations_cooperative(
      workspace, metadata, p, left, left + 1u, result, gid, int(gid),
      kDeviceInfeasible, 19, lane, shared_state);
}

// The parent slots contain suffix contexts.  Reduction values are still in
// the child slots.  The child slots are replaced in place by their contexts.
kernel void clqr_expand_primal_suffix_context(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  (void)dimensions;
  (void)input;
  (void)output;
  threadgroup int shared_state[4];
  const uint gid = threadgroup_position.x;
  if (lane == 0u) {
    if (gid < p.parent_count && p.parent_count == 1u)
      set_invalid_relation(metadata, p, p.parent_offset);
    shared_state[0] =
        gid < p.parent_count && status_ok(metadata, p) ? 1 : 0;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (shared_state[0] == 0)
    return;
  const uint left = p.child_offset + 2u * gid;
  const uint parent = p.parent_offset + gid;
  if (2u * gid + 1u >= p.child_count) {
    copy_relation_cooperative(workspace, metadata, p, parent, left, lane);
    return;
  }
  const uint right = left + 1u;
  if (!compose_scan_relations_cooperative(
          workspace, metadata, p, right, parent, left, gid, int(gid), 20,
          lane, shared_state))
    return;
  copy_relation_cooperative(workspace, metadata, p, parent, right, lane);
}

// Finalize the two leaves owned by each parent context.  The right suffix is
// formed first; the left suffix then consumes it.  A scratch relation slot is
// required because an active parent may alias neither leaf, while the global
// matrix scratch is independently strided by gid.
kernel void clqr_finalize_primal_suffix(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  (void)dimensions;
  (void)input;
  (void)output;
  threadgroup int shared_state[4];
  const uint gid = threadgroup_position.x;
  if (lane == 0u) {
    if (gid < p.parent_count && p.parent_count == 1u &&
        p.child_count <= 2u)
      set_invalid_relation(metadata, p, p.parent_offset);
    shared_state[0] =
        gid < p.parent_count && status_ok(metadata, p) ? 1 : 0;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (shared_state[0] == 0)
    return;
  const uint left = p.child_offset + 2u * gid;
  const uint right = left + 1u;
  const uint parent = p.parent_offset + gid;
  const uint temporary = p.temporary_offset + gid;
  if (2u * gid + 1u >= p.child_count) {
    if (!invalid_relation(metadata, p, parent) &&
        compose_scan_relations_cooperative(
            workspace, metadata, p, left, parent, temporary, gid, int(left),
            21, lane, shared_state)) {
      threadgroup_barrier(mem_flags::mem_device);
      copy_relation_cooperative(workspace, metadata, p, temporary, left,
                                lane);
    }
    return;
  }

  if (!invalid_relation(metadata, p, parent)) {
    if (!compose_scan_relations_cooperative(
            workspace, metadata, p, right, parent, temporary, gid, int(right),
            21, lane, shared_state))
      return;
    threadgroup_barrier(mem_flags::mem_device);
    copy_relation_cooperative(workspace, metadata, p, temporary, right, lane);
    threadgroup_barrier(mem_flags::mem_device);
  }
  if (!compose_relations_cooperative(
          workspace, metadata, p, left, right, temporary, gid, int(left),
          kDeviceInfeasible, 21, lane, shared_state))
    return;
  threadgroup_barrier(mem_flags::mem_device);
  copy_relation_cooperative(workspace, metadata, p, temporary, left, lane);
}

// State parameters use a padded layout:
//   data(slot) = T[state_capacity,state_capacity], t[state_capacity]
//   meta(slot) = physical_dim, reduced_dim
//   free_columns(slot) = state_capacity int entries
// T's row stride is state_capacity, independent of the active reduced size.
kernel void clqr_extract_state_parameters(
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
  const uint count = p.stage_count + 1u;
  if (gid >= count || !status_ok(metadata, p))
    return;
  const uint slot = p.child_offset + gid;
  const int left_dim = relation_left_dim(metadata, p, slot);
  const int right_dim = relation_right_dim(metadata, p, slot);
  const int rows = relation_rows(metadata, p, slot);
  if (right_dim != 0 || rows > left_dim || left_dim < 0) {
    set_failure(metadata, p, kDeviceNumericalFailure, int(gid), 4);
    return;
  }

  const uint nx = p.state_capacity;
  const uint data = p.state_param_data + gid * p.state_param_stride;
  const uint T = data;
  const uint t = data + nx * nx;
  const uint meta = p.state_param_meta + 2u * gid;
  const uint free_columns = p.state_param_free_columns + gid * nx;
  device int *pivot_row =
      metadata + p.int_scratch + gid * p.int_scratch_stride;
  for (int col = 0; col < left_dim; ++col) {
    pivot_row[col] = -1;
    workspace[t + uint(col)] = 0.0f;
  }
  for (int row = 0; row < left_dim; ++row)
    for (uint col = 0; col < nx; ++col)
      workspace[T + uint(row) * nx + col] = 0.0f;

  const uint relation_left = relation_left_offset(p, slot);
  const uint relation_rhs = relation_rhs_offset(p, slot);
  for (int row = 0; row < rows; ++row) {
    int column = -1;
    for (int col = 0; col < left_dim; ++col) {
      if (fabs(workspace[relation_left + uint(row) * nx + uint(col)]) >
          p.rank_tolerance) {
        column = col;
        break;
      }
    }
    if (column < 0 || pivot_row[column] >= 0) {
      set_failure(metadata, p, kDeviceNumericalFailure, int(gid), 5);
      return;
    }
    pivot_row[column] = row;
  }

  int reduced = 0;
  for (int col = 0; col < left_dim; ++col) {
    if (pivot_row[col] < 0)
      metadata[free_columns + uint(reduced++)] = col;
  }
  metadata[meta] = left_dim;
  metadata[meta + 1u] = reduced;

  for (int col = 0; col < left_dim; ++col) {
    if (pivot_row[col] < 0)
      continue;
    const int row = pivot_row[col];
    const float diagonal =
        workspace[relation_left + uint(row) * nx + uint(col)];
    workspace[t + uint(col)] =
        workspace[relation_rhs + uint(row)] / diagonal;
    for (int free = 0; free < reduced; ++free) {
      const int free_column = metadata[free_columns + uint(free)];
      workspace[T + uint(col) * nx + uint(free)] =
          -workspace[relation_left + uint(row) * nx + uint(free_column)] /
          diagonal;
    }
  }
  for (int free = 0; free < reduced; ++free) {
    const int free_column = metadata[free_columns + uint(free)];
    workspace[T + uint(free_column) * nx + uint(free)] = 1.0f;
  }
}
)CLQR_METAL";

} // namespace clqr::metal::detail

#endif // CLQR_SRC_METAL_PRIMAL_SCAN_SOURCE_H_
