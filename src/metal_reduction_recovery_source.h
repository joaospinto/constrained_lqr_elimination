#ifndef CLQR_SRC_METAL_REDUCTION_RECOVERY_SOURCE_H_
#define CLQR_SRC_METAL_REDUCTION_RECOVERY_SOURCE_H_

namespace clqr::metal::detail {

inline constexpr char kMetalReductionRecoverySource[] = R"CLQR_METAL(
#include <metal_stdlib>

using namespace metal;

// Keep this declaration byte-for-byte layout-compatible with
// clqr::metal::detail::KernelParams in metal_layout.h. All scalar offsets are
// float elements and all metadata offsets are int32 elements.
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

inline int state_dimension(device const int *dimensions, uint node) {
  return dimensions[node];
}

inline int control_dimension(device const int *dimensions,
                             constant KernelParams &p, uint stage) {
  return dimensions[control_dimension_offset(p) + stage];
}

inline int mixed_dimension(device const int *dimensions,
                           constant KernelParams &p, uint stage) {
  return dimensions[mixed_dimension_offset(p) + stage];
}

inline uint relation_rows_capacity(constant KernelParams &p) {
  return 2u * p.state_capacity;
}

inline uint relation_left(device const int *metadata,
                          constant KernelParams &p, uint slot) {
  (void)metadata;
  return p.relation_data + slot * p.relation_stride;
}

inline uint relation_right(device const int *metadata,
                           constant KernelParams &p, uint slot) {
  return relation_left(metadata, p, slot) +
         relation_rows_capacity(p) * p.state_capacity;
}

inline uint relation_rhs(device const int *metadata,
                         constant KernelParams &p, uint slot) {
  return relation_right(metadata, p, slot) +
         relation_rows_capacity(p) * p.state_capacity;
}

inline int relation_left_dim(device const int *metadata,
                             constant KernelParams &p, uint slot) {
  return metadata[p.relation_meta + 3u * slot];
}

inline int relation_rows(device const int *metadata, constant KernelParams &p,
                         uint slot) {
  return metadata[p.relation_meta + 3u * slot + 2u];
}

inline uint state_param_T(constant KernelParams &p, uint node) {
  return p.state_param_data + node * p.state_param_stride;
}

inline uint state_param_t(constant KernelParams &p, uint node) {
  return state_param_T(p, node) + p.state_capacity * p.state_capacity;
}

inline int state_param_physical(device const int *metadata,
                                constant KernelParams &p, uint node) {
  return metadata[p.state_param_meta + 2u * node];
}

inline int state_param_reduced(device const int *metadata,
                               constant KernelParams &p, uint node) {
  return metadata[p.state_param_meta + 2u * node + 1u];
}

inline uint state_param_free_columns(constant KernelParams &p, uint node) {
  return p.state_param_free_columns + node * p.state_capacity;
}

inline uint control_param_Y(constant KernelParams &p, uint stage) {
  return p.control_param_data + stage * p.control_param_stride;
}

inline uint control_param_Z(constant KernelParams &p, uint stage) {
  return control_param_Y(p, stage) +
         p.control_capacity * p.state_capacity;
}

inline uint control_param_y(constant KernelParams &p, uint stage) {
  return control_param_Z(p, stage) +
         p.control_capacity * p.control_capacity;
}

inline uint control_param_free_columns(constant KernelParams &p, uint stage) {
  return p.control_param_free_columns + stage * p.control_capacity;
}

inline device int *control_param_meta(device int *metadata,
                                      constant KernelParams &p, uint stage) {
  return metadata + p.control_param_meta + 3u * stage;
}

inline uint reduced_stage_base(constant KernelParams &p, uint stage) {
  return p.reduced_stage_data + stage * p.reduced_stage_stride;
}

inline uint reduced_stage_A(constant KernelParams &p, uint stage) {
  return reduced_stage_base(p, stage);
}

inline uint reduced_stage_B(constant KernelParams &p, uint stage) {
  return reduced_stage_A(p, stage) +
         p.state_capacity * p.state_capacity;
}

inline uint reduced_stage_c(constant KernelParams &p, uint stage) {
  return reduced_stage_B(p, stage) +
         p.state_capacity * p.control_capacity;
}

inline uint reduced_stage_Q(constant KernelParams &p, uint stage) {
  return reduced_stage_c(p, stage) + p.state_capacity;
}

inline uint reduced_stage_R(constant KernelParams &p, uint stage) {
  return reduced_stage_Q(p, stage) +
         p.state_capacity * p.state_capacity;
}

inline uint reduced_stage_M(constant KernelParams &p, uint stage) {
  return reduced_stage_R(p, stage) +
         p.control_capacity * p.control_capacity;
}

inline uint reduced_stage_q(constant KernelParams &p, uint stage) {
  return reduced_stage_M(p, stage) +
         p.state_capacity * p.control_capacity;
}

inline uint reduced_stage_r(constant KernelParams &p, uint stage) {
  return reduced_stage_q(p, stage) + p.state_capacity;
}

inline device int *reduced_stage_meta(device int *metadata,
                                      constant KernelParams &p, uint stage) {
  return metadata + p.reduced_stage_meta + 3u * stage;
}

inline uint reduced_terminal_Q(constant KernelParams &p) {
  return p.reduced_terminal_data;
}

inline uint reduced_terminal_q(constant KernelParams &p) {
  return p.reduced_terminal_data + p.state_capacity * p.state_capacity;
}

inline device int *value_meta(device int *metadata, constant KernelParams &p,
                              uint slot) {
  return metadata + p.value_meta + 2u * slot;
}

inline uint value_J(constant KernelParams &p, uint slot) {
  return p.value_data + slot * p.value_stride +
         2u * p.state_capacity * p.state_capacity;
}

inline uint feedback_K(constant KernelParams &p, uint stage) {
  return p.feedback_data + stage * p.feedback_stride;
}

inline uint feedback_k(constant KernelParams &p, uint stage) {
  return feedback_K(p, stage) +
         p.control_capacity * p.state_capacity;
}

inline uint feedback_factor(constant KernelParams &p, uint stage) {
  return feedback_k(p, stage) + p.control_capacity;
}

inline uint feedback_transition(constant KernelParams &p, uint stage) {
  return feedback_factor(p, stage) +
         p.control_capacity * p.control_capacity;
}

inline uint feedback_offset(constant KernelParams &p, uint stage) {
  return feedback_transition(p, stage) +
         p.state_capacity * p.state_capacity;
}

inline device int *feedback_meta(device int *metadata,
                                 constant KernelParams &p, uint stage) {
  return metadata + p.feedback_meta + 3u * stage;
}

inline uint affine_linear(constant KernelParams &p, uint slot) {
  return p.affine_data + slot * p.affine_stride;
}

inline uint affine_offset(constant KernelParams &p, uint slot) {
  return affine_linear(p, slot) +
         p.state_capacity * p.state_capacity;
}

inline device int *affine_meta(device int *metadata, constant KernelParams &p,
                               uint slot) {
  return metadata + p.affine_meta + 2u * slot;
}

// One 32-lane threadgroup performs the same deterministic RREF as CUDA.
// Lane zero owns every pivot/rank decision; lanes only split independent
// rows or entries, so every scalar dot product keeps its original order.
inline void rref_threadgroup(
    threadgroup float *matrix, int rows, int columns, int pivot_limit,
    float tolerance, threadgroup int *pivot_columns,
    threadgroup float *factors, threadgroup int *rank,
    threadgroup int *best_row, uint lane, uint lane_count) {
  for (int row = int(lane); row < rows; row += int(lane_count)) {
    float scale = 0.0f;
    for (int col = 0; col < pivot_limit; ++col)
      scale = fmax(scale, fabs(matrix[row * columns + col]));
    if (scale > 0.0f)
      for (int col = 0; col < columns; ++col)
        matrix[row * columns + col] /= scale;
  }
  if (lane == 0u)
    *rank = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int col = 0; col < pivot_limit; ++col) {
    if (lane == 0u) {
      *best_row = -1;
      float best = tolerance;
      for (int row = *rank; row < rows; ++row) {
        const float candidate = fabs(matrix[row * columns + col]);
        if (candidate > best) {
          best = candidate;
          *best_row = row;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int selected_row = *best_row;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (selected_row < 0)
      continue;

    const int pivot_row = *rank;
    if (selected_row != pivot_row) {
      for (int j = int(lane); j < columns; j += int(lane_count)) {
        const float temporary = matrix[pivot_row * columns + j];
        matrix[pivot_row * columns + j] =
            matrix[selected_row * columns + j];
        matrix[selected_row * columns + j] = temporary;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float pivot = matrix[pivot_row * columns + col];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = col + int(lane); j < columns; j += int(lane_count))
      matrix[pivot_row * columns + j] /= pivot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int row = int(lane); row < rows; row += int(lane_count))
      factors[row] =
          row == pivot_row ? 0.0f : matrix[row * columns + col];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int active_columns = columns - col;
    for (int index = int(lane); index < rows * active_columns;
         index += int(lane_count)) {
      const int row = index / active_columns;
      const int j = col + index % active_columns;
      if (row != pivot_row)
        matrix[row * columns + j] =
            fma(-factors[row], matrix[pivot_row * columns + j],
                matrix[row * columns + j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
      pivot_columns[pivot_row] = col;
      ++(*rank);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (*rank == rows)
      break;
  }

  for (int index = int(lane); index < rows * columns;
       index += int(lane_count))
    if (fabs(matrix[index]) <= tolerance)
      matrix[index] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void solve_positive_definite(device const float *lower, int dimension,
                                    device float *rhs, int rhs_stride,
                                    int rhs_count) {
  for (int column = 0; column < rhs_count; ++column) {
    for (int row = 0; row < dimension; ++row) {
      float value = rhs[row * rhs_stride + column];
      for (int col = 0; col < row; ++col)
        value = fma(-lower[row * dimension + col],
                    rhs[col * rhs_stride + column], value);
      rhs[row * rhs_stride + column] =
          value / lower[row * dimension + row];
    }
    for (int reverse = 0; reverse < dimension; ++reverse) {
      const int row = dimension - 1 - reverse;
      float value = rhs[row * rhs_stride + column];
      for (int col = row + 1; col < dimension; ++col)
        value = fma(-lower[col * dimension + row],
                    rhs[col * rhs_stride + column], value);
      rhs[row * rhs_stride + column] =
          value / lower[row * dimension + row];
    }
  }
}

kernel void clqr_reduce_stages(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    threadgroup float *local_float [[threadgroup(0)]],
    threadgroup int *local_int [[threadgroup(1)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  (void)output;
  const uint gid = group.x;
  const uint lane_count = group_size.x;
  if (gid >= p.stage_count)
    return;
  if (lane == 0u)
    local_int[0] = enabled(metadata, p) ? 1 : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_int[0] == 0)
    return;

  const int n = state_dimension(dimensions, gid);
  const int next_n = state_dimension(dimensions, gid + 1u);
  const int physical_m = control_dimension(dimensions, p, gid);
  const int mixed = mixed_dimension(dimensions, p, gid);
  const int current_reduced = state_param_reduced(metadata, p, gid);
  const int next_reduced = state_param_reduced(metadata, p, gid + 1u);
  const uint suffix_slot = gid + 1u;
  const int suffix_rows = relation_rows(metadata, p, suffix_slot);
  const int suffix_left_dim =
      relation_left_dim(metadata, p, suffix_slot);
  if (suffix_left_dim != next_n) {
    if (lane == 0u)
      fail(metadata, p, kDeviceNumericalFailure, int(gid), 7);
    return;
  }

  const int rows = mixed + suffix_rows;
  const int columns = physical_m + current_reduced + 1;
  threadgroup float *matrix = local_float;
  threadgroup float *row_factors = matrix + rows * columns;
  threadgroup int *pivot_columns = local_int + 4;
  threadgroup int *rank = local_int + 1;
  threadgroup int *best_row = local_int + 2;
  threadgroup int *control_rank_ptr = local_int + 3;
  for (int index = int(lane); index < rows * columns;
       index += int(lane_count))
    matrix[index] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint nx = p.state_capacity;
  const uint nu = p.control_capacity;
  const uint nc = p.mixed_capacity;
  const uint A = p.input_A + gid * nx * nx;
  const uint B = p.input_B + gid * nx * nu;
  const uint c = p.input_c + gid * nx;
  const uint Q = p.input_Q + gid * nx * nx;
  const uint R = p.input_R + gid * nu * nu;
  const uint M = p.input_M + gid * nx * nu;
  const uint q = p.input_q + gid * nx;
  const uint r = p.input_r + gid * nu;
  const uint C = p.input_C + gid * nc * nx;
  const uint D = p.input_D + gid * nc * nu;
  const uint d = p.input_d + gid * nc;

  const uint current_T = state_param_T(p, gid);
  const uint current_t = state_param_t(p, gid);
  const uint next_t = state_param_t(p, gid + 1u);
  const uint next_free = state_param_free_columns(p, gid + 1u);
  const uint suffix_left = relation_left(metadata, p, suffix_slot);
  const uint suffix_rhs = relation_rhs(metadata, p, suffix_slot);
  // The host reserves the larger of the RREF-plus-successor scratch below
  // and the post-RREF block-transform scratch. Keep A*T and c+A*t past RREF
  // so reduced dynamics can reuse the exact same staged products.
  threadgroup float *suffix_AT = row_factors + rows;
  threadgroup float *suffix_affine =
      suffix_AT + next_n * current_reduced;
  for (int linear = int(lane); linear < next_n * current_reduced;
       linear += int(lane_count)) {
    const int xp = linear / current_reduced;
    const int z = linear % current_reduced;
    float value = 0.0f;
    for (int x = 0; x < n; ++x)
      value = fma(input[A + uint(xp) * nx + uint(x)],
                  workspace[current_T + uint(x) * nx + uint(z)], value);
    suffix_AT[linear] = value;
  }
  for (int xp = int(lane); xp < next_n; xp += int(lane_count)) {
    float value = input[c + uint(xp)];
    for (int x = 0; x < n; ++x)
      value = fma(input[A + uint(xp) * nx + uint(x)],
                  workspace[current_t + uint(x)], value);
    suffix_affine[xp] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Original mixed equalities after x=Tz+t.
  for (int linear = int(lane); linear < mixed * physical_m;
       linear += int(lane_count)) {
    const int row = linear / physical_m;
    const int u = linear % physical_m;
    matrix[row * columns + u] =
        input[D + uint(row) * nu + uint(u)];
  }
  for (int linear = int(lane); linear < mixed * current_reduced;
       linear += int(lane_count)) {
    const int row = linear / current_reduced;
    const int z = linear % current_reduced;
    float value = 0.0f;
    for (int x = 0; x < n; ++x)
      value = fma(input[C + uint(row) * nx + uint(x)],
                  workspace[current_T + uint(x) * nx + uint(z)], value);
    matrix[row * columns + physical_m + z] = value;
  }
  for (int row = int(lane); row < mixed; row += int(lane_count)) {
    float value = -input[d + uint(row)];
    for (int x = 0; x < n; ++x)
      value = fma(-input[C + uint(row) * nx + uint(x)],
                  workspace[current_t + uint(x)], value);
    matrix[row * columns + columns - 1] = value;
  }

  // The successor must satisfy the already-scanned suffix relation.
  for (int linear = int(lane); linear < suffix_rows * physical_m;
       linear += int(lane_count)) {
    const int row = linear / physical_m;
    const int u = linear % physical_m;
    float value = 0.0f;
    for (int xp = 0; xp < next_n; ++xp)
      value = fma(workspace[suffix_left + uint(row) * nx + uint(xp)],
                  input[B + uint(xp) * nu + uint(u)], value);
    matrix[(mixed + row) * columns + u] = value;
  }
  for (int linear = int(lane); linear < suffix_rows * current_reduced;
       linear += int(lane_count)) {
    const int row = linear / current_reduced;
    const int z = linear % current_reduced;
    float value = 0.0f;
    for (int xp = 0; xp < next_n; ++xp)
      value = fma(workspace[suffix_left + uint(row) * nx + uint(xp)],
                  suffix_AT[xp * current_reduced + z], value);
    matrix[(mixed + row) * columns + physical_m + z] = value;
  }
  for (int row = int(lane); row < suffix_rows; row += int(lane_count)) {
    const int output_row = mixed + row;
    float value = workspace[suffix_rhs + uint(row)];
    for (int xp = 0; xp < next_n; ++xp) {
      value = fma(-workspace[suffix_left + uint(row) * nx + uint(xp)],
                  suffix_affine[xp], value);
    }
    matrix[output_row * columns + columns - 1] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Preserve CUDA's two preliminary cancellation filters before generic
  // row equilibration inside RREF.
  for (int row = int(lane); row < rows; row += int(lane_count)) {
    float scale = 0.0f;
    if (row < mixed) {
      for (int x = 0; x < n; ++x)
        scale = fmax(
            scale, fabs(input[C + uint(row) * nx + uint(x)]));
      for (int u = 0; u < physical_m; ++u)
        scale = fmax(
            scale, fabs(input[D + uint(row) * nu + uint(u)]));
      scale = fmax(scale, fabs(input[d + uint(row)]));
    } else {
      for (int col = 0; col < columns; ++col)
        scale = fmax(scale, fabs(matrix[row * columns + col]));
    }
    row_factors[row] = scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int linear = int(lane); linear < rows * columns;
       linear += int(lane_count)) {
    const int row = linear / columns;
    const float scale = row_factors[row];
    if (row >= mixed && scale <= p.rank_tolerance)
      matrix[linear] = 0.0f;
    else if (scale > 0.0f)
      matrix[linear] /= scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int row = int(lane); row < rows; row += int(lane_count)) {
    float scale = 0.0f;
    for (int col = 0; col < columns; ++col)
      scale = fmax(scale, fabs(matrix[row * columns + col]));
    row_factors[row] = scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int linear = int(lane); linear < rows * columns;
       linear += int(lane_count))
    if (row_factors[linear / columns] <= p.rank_tolerance)
      matrix[linear] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  rref_threadgroup(matrix, rows, columns, columns - 1, p.rank_tolerance,
                   pivot_columns, row_factors, rank, best_row, lane,
                   lane_count);
  if (lane == 0u) {
    local_int[0] = 1;
    for (int row = 0; row < rows && local_int[0] != 0; ++row) {
      bool zero = true;
      for (int col = 0; col < columns - 1; ++col) {
        if (fabs(matrix[row * columns + col]) > p.rank_tolerance) {
          zero = false;
          break;
        }
      }
      if (zero &&
          fabs(matrix[row * columns + columns - 1]) >
              p.consistency_tolerance) {
        fail(metadata, p, kDeviceInfeasible, int(gid), 6);
        local_int[0] = 0;
      }
    }
    *control_rank_ptr = 0;
    while (*control_rank_ptr < *rank &&
           pivot_columns[*control_rank_ptr] < physical_m)
      ++(*control_rank_ptr);
    if (*control_rank_ptr < *rank) {
      fail(metadata, p, kDeviceNumericalFailure, int(gid), 7);
      local_int[0] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_int[0] == 0)
    return;
  const int control_rank = *control_rank_ptr;

  const int reduced_m = physical_m - control_rank;
  device int *cp_meta = control_param_meta(metadata, p, gid);
  device int *rs_meta = reduced_stage_meta(metadata, p, gid);
  const uint cp_free = control_param_free_columns(p, gid);
  if (lane == 0u) {
    cp_meta[0] = physical_m;
    cp_meta[1] = current_reduced;
    cp_meta[2] = reduced_m;
    rs_meta[0] = current_reduced;
    rs_meta[1] = next_reduced;
    rs_meta[2] = reduced_m;
    int free = 0;
    for (int u = 0; u < physical_m; ++u) {
      bool pivot = false;
      for (int row = 0; row < control_rank; ++row)
        pivot = pivot || pivot_columns[row] == u;
      if (!pivot)
        metadata[cp_free + uint(free++)] = u;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  const uint cp_Y = control_param_Y(p, gid);
  const uint cp_Z = control_param_Z(p, gid);
  const uint cp_y = control_param_y(p, gid);
  for (int u = int(lane); u < physical_m; u += int(lane_count))
    workspace[cp_y + uint(u)] = 0.0f;
  for (int linear = int(lane); linear < physical_m * current_reduced;
       linear += int(lane_count)) {
    const int u = linear / current_reduced;
    const int z = linear % current_reduced;
    workspace[cp_Y + uint(u) * nx + uint(z)] = 0.0f;
  }
  for (int linear = int(lane); linear < physical_m * reduced_m;
       linear += int(lane_count)) {
    const int u = linear / reduced_m;
    const int v = linear % reduced_m;
    workspace[cp_Z + uint(u) * nu + uint(v)] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (lane == 0u) {
    for (int row = 0; row < control_rank; ++row) {
      const int u = pivot_columns[row];
      workspace[cp_y + uint(u)] =
          matrix[row * columns + columns - 1];
      for (int z = 0; z < current_reduced; ++z)
        workspace[cp_Y + uint(u) * nx + uint(z)] =
            -matrix[row * columns + physical_m + z];
      for (int v = 0; v < reduced_m; ++v)
        workspace[cp_Z + uint(u) * nu + uint(v)] =
            -matrix[row * columns + metadata[cp_free + uint(v)]];
    }
    for (int v = 0; v < reduced_m; ++v) {
      const int u = metadata[cp_free + uint(v)];
      workspace[cp_Z + uint(u) * nu + uint(v)] = 1.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  const uint rs_A = reduced_stage_A(p, gid);
  const uint rs_B = reduced_stage_B(p, gid);
  const uint rs_c = reduced_stage_c(p, gid);
  const uint rs_Q = reduced_stage_Q(p, gid);
  const uint rs_R = reduced_stage_R(p, gid);
  const uint rs_M = reduced_stage_M(p, gid);
  const uint rs_q = reduced_stage_q(p, gid);
  const uint rs_r = reduced_stage_r(p, gid);

  for (int linear = int(lane); linear < next_reduced * current_reduced;
       linear += int(lane_count)) {
    const int row = linear / current_reduced;
    const int z = linear % current_reduced;
    const int xp = metadata[next_free + uint(row)];
    float value = suffix_AT[xp * current_reduced + z];
    for (int u = 0; u < physical_m; ++u)
      value = fma(input[B + uint(xp) * nu + uint(u)],
                  workspace[cp_Y + uint(u) * nx + uint(z)], value);
    workspace[rs_A + uint(row) * nx + uint(z)] = value;
  }
  for (int linear = int(lane); linear < next_reduced * reduced_m;
       linear += int(lane_count)) {
    const int row = linear / reduced_m;
    const int v = linear % reduced_m;
    const int xp = metadata[next_free + uint(row)];
    float value = 0.0f;
    for (int u = 0; u < physical_m; ++u)
      value = fma(input[B + uint(xp) * nu + uint(u)],
                  workspace[cp_Z + uint(u) * nu + uint(v)], value);
    workspace[rs_B + uint(row) * nu + uint(v)] = value;
  }
  for (int row = int(lane); row < next_reduced;
       row += int(lane_count)) {
    const int xp = metadata[next_free + uint(row)];
    float value =
        suffix_affine[xp] - workspace[next_t + uint(xp)];
    for (int u = 0; u < physical_m; ++u)
      value = fma(input[B + uint(xp) * nu + uint(u)],
                  workspace[cp_y + uint(u)], value);
    workspace[rs_c + uint(row)] = value;
  }

  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  // Apply the dense block transform
  //
  //   L = [ T  0 ],  H = [ Q  M ],
  //       [ Y  Z ]       [ M' R ]
  //
  // as U=H*L followed by L'*U. This is cubic in the active dimensions;
  // forming every output entry directly from two contracted physical
  // indices would be quartic. The RREF storage is dead at this point and is
  // deliberately reused for the two blocks of U and H*[t;y]+[q;r].
  const int physical_rows = n + physical_m;
  threadgroup float *block_z = local_float;
  threadgroup float *block_v =
      block_z + physical_rows * current_reduced;
  threadgroup float *physical_gradient =
      block_v + physical_rows * reduced_m;

  for (int linear = int(lane); linear < n * current_reduced;
       linear += int(lane_count)) {
    const int x = linear / current_reduced;
    const int z = linear % current_reduced;
    float value = 0.0f;
    for (int y = 0; y < n; ++y)
      value = fma(input[Q + uint(x) * nx + uint(y)],
                  workspace[current_T + uint(y) * nx + uint(z)], value);
    for (int u = 0; u < physical_m; ++u)
      value = fma(input[M + uint(x) * nu + uint(u)],
                  workspace[cp_Y + uint(u) * nx + uint(z)], value);
    block_z[x * current_reduced + z] = value;
  }
  for (int linear = int(lane); linear < physical_m * current_reduced;
       linear += int(lane_count)) {
    const int u = linear / current_reduced;
    const int z = linear % current_reduced;
    float value = 0.0f;
    for (int x = 0; x < n; ++x)
      value = fma(input[M + uint(x) * nu + uint(u)],
                  workspace[current_T + uint(x) * nx + uint(z)], value);
    for (int v = 0; v < physical_m; ++v)
      value = fma(input[R + uint(u) * nu + uint(v)],
                  workspace[cp_Y + uint(v) * nx + uint(z)], value);
    block_z[(n + u) * current_reduced + z] = value;
  }
  for (int linear = int(lane); linear < n * reduced_m;
       linear += int(lane_count)) {
    const int x = linear / reduced_m;
    const int v = linear % reduced_m;
    float value = 0.0f;
    for (int u = 0; u < physical_m; ++u)
      value = fma(input[M + uint(x) * nu + uint(u)],
                  workspace[cp_Z + uint(u) * nu + uint(v)], value);
    block_v[x * reduced_m + v] = value;
  }
  for (int linear = int(lane); linear < physical_m * reduced_m;
       linear += int(lane_count)) {
    const int u = linear / reduced_m;
    const int v = linear % reduced_m;
    float value = 0.0f;
    for (int physical_v = 0; physical_v < physical_m; ++physical_v)
      value = fma(input[R + uint(u) * nu + uint(physical_v)],
                  workspace[cp_Z + uint(physical_v) * nu + uint(v)], value);
    block_v[(n + u) * reduced_m + v] = value;
  }
  for (int x = int(lane); x < n; x += int(lane_count)) {
    float value = input[q + uint(x)];
    for (int y = 0; y < n; ++y)
      value = fma(input[Q + uint(x) * nx + uint(y)],
                  workspace[current_t + uint(y)], value);
    for (int u = 0; u < physical_m; ++u)
      value = fma(input[M + uint(x) * nu + uint(u)],
                  workspace[cp_y + uint(u)], value);
    physical_gradient[x] = value;
  }
  for (int u = int(lane); u < physical_m; u += int(lane_count)) {
    float value = input[r + uint(u)];
    for (int x = 0; x < n; ++x)
      value = fma(input[M + uint(x) * nu + uint(u)],
                  workspace[current_t + uint(x)], value);
    for (int v = 0; v < physical_m; ++v)
      value = fma(input[R + uint(u) * nu + uint(v)],
                  workspace[cp_y + uint(v)], value);
    physical_gradient[n + u] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int linear = int(lane); linear < current_reduced * current_reduced;
       linear += int(lane_count)) {
    const int a = linear / current_reduced;
    const int b = linear % current_reduced;
    float value = 0.0f;
    for (int x = 0; x < n; ++x)
      value = fma(workspace[current_T + uint(x) * nx + uint(a)],
                  block_z[x * current_reduced + b], value);
    for (int u = 0; u < physical_m; ++u)
      value = fma(workspace[cp_Y + uint(u) * nx + uint(a)],
                  block_z[(n + u) * current_reduced + b], value);
    workspace[rs_Q + uint(a) * nx + uint(b)] = value;
  }
  for (int linear = int(lane); linear < current_reduced * reduced_m;
       linear += int(lane_count)) {
    const int z = linear / reduced_m;
    const int v = linear % reduced_m;
    float value = 0.0f;
    for (int x = 0; x < n; ++x)
      value = fma(workspace[current_T + uint(x) * nx + uint(z)],
                  block_v[x * reduced_m + v], value);
    for (int u = 0; u < physical_m; ++u)
      value = fma(workspace[cp_Y + uint(u) * nx + uint(z)],
                  block_v[(n + u) * reduced_m + v], value);
    workspace[rs_M + uint(z) * nu + uint(v)] = value;
  }
  for (int linear = int(lane); linear < reduced_m * reduced_m;
       linear += int(lane_count)) {
    const int a = linear / reduced_m;
    const int b = linear % reduced_m;
    float value = 0.0f;
    for (int u = 0; u < physical_m; ++u)
      value = fma(workspace[cp_Z + uint(u) * nu + uint(a)],
                  block_v[(n + u) * reduced_m + b], value);
    workspace[rs_R + uint(a) * nu + uint(b)] = value;
  }
  for (int z = int(lane); z < current_reduced; z += int(lane_count)) {
    float value = 0.0f;
    for (int x = 0; x < n; ++x)
      value = fma(workspace[current_T + uint(x) * nx + uint(z)],
                  physical_gradient[x], value);
    for (int u = 0; u < physical_m; ++u)
      value = fma(workspace[cp_Y + uint(u) * nx + uint(z)],
                  physical_gradient[n + u], value);
    workspace[rs_q + uint(z)] = value;
  }
  for (int v = int(lane); v < reduced_m; v += int(lane_count)) {
    float value = 0.0f;
    for (int u = 0; u < physical_m; ++u)
      value = fma(workspace[cp_Z + uint(u) * nu + uint(v)],
                  physical_gradient[n + u], value);
    workspace[rs_r + uint(v)] = value;
  }
}

kernel void clqr_reduce_terminal(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    threadgroup float *local_float [[threadgroup(0)]],
    threadgroup int *local_int [[threadgroup(1)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  (void)output;
  if (group.x != 0u)
    return;
  if (lane == 0u)
    local_int[0] = enabled(metadata, p) ? 1 : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_int[0] == 0)
    return;
  const uint lane_count = group_size.x;
  const uint terminal = p.stage_count;
  const int n = state_dimension(dimensions, terminal);
  const int reduced = state_param_reduced(metadata, p, terminal);
  if (lane == 0u)
    metadata[p.reduced_stage_meta + 3u * terminal] = reduced;
  const uint nx = p.state_capacity;
  const uint T = state_param_T(p, terminal);
  const uint t = state_param_t(p, terminal);
  const uint Q = p.input_Q + terminal * nx * nx;
  const uint q = p.input_q + terminal * nx;
  const uint out_Q = reduced_terminal_Q(p);
  const uint out_q = reduced_terminal_q(p);
  threadgroup float *QT = local_float;
  threadgroup float *gradient = QT + n * reduced;
  for (int linear = int(lane); linear < n * reduced;
       linear += int(lane_count)) {
    const int x = linear / reduced;
    const int b = linear % reduced;
    float value = 0.0f;
    for (int y = 0; y < n; ++y)
      value = fma(input[Q + uint(x) * nx + uint(y)],
                  workspace[T + uint(y) * nx + uint(b)], value);
    QT[linear] = value;
  }
  for (int x = int(lane); x < n; x += int(lane_count)) {
    float value = input[q + uint(x)];
    for (int y = 0; y < n; ++y)
      value = fma(input[Q + uint(x) * nx + uint(y)],
                  workspace[t + uint(y)], value);
    gradient[x] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int linear = int(lane); linear < reduced * reduced;
       linear += int(lane_count)) {
    const int a = linear / reduced;
    const int b = linear % reduced;
    float value = 0.0f;
    for (int x = 0; x < n; ++x)
      value = fma(workspace[T + uint(x) * nx + uint(a)],
                  QT[x * reduced + b], value);
    workspace[out_Q + uint(a) * nx + uint(b)] = value;
  }
  for (int a = int(lane); a < reduced; a += int(lane_count)) {
    float value = 0.0f;
    for (int x = 0; x < n; ++x)
      value = fma(workspace[T + uint(x) * nx + uint(a)],
                  gradient[x], value);
    workspace[out_q + uint(a)] = value;
  }
}

kernel void clqr_initial_reduced_state(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  (void)output;
  if (gid != 0u || !enabled(metadata, p))
    return;
  const int physical = state_param_physical(metadata, p, 0);
  const int reduced = state_param_reduced(metadata, p, 0);
  const uint nx = p.state_capacity;
  const uint T = state_param_T(p, 0);
  const uint t = state_param_t(p, 0);
  const uint free_columns = state_param_free_columns(p, 0);
  for (int z = 0; z < reduced; ++z) {
    const int x = metadata[free_columns + uint(z)];
    workspace[p.reduced_initial + uint(z)] =
        input[p.input_initial_state + uint(x)] -
        workspace[t + uint(x)];
  }
  float scale = 1.0f;
  float residual = 0.0f;
  for (int x = 0; x < physical; ++x) {
    float value = workspace[t + uint(x)];
    for (int z = 0; z < reduced; ++z)
      value = fma(workspace[T + uint(x) * nx + uint(z)],
                  workspace[p.reduced_initial + uint(z)], value);
    const float expected = input[p.input_initial_state + uint(x)];
    scale = fmax(scale, fabs(expected));
    residual = fmax(residual, fabs(value - expected));
  }
  if (residual > 20.0f * p.rank_tolerance * scale)
    fail(metadata, p, kDeviceInfeasible, 0, 8);
}

kernel void clqr_matrix_feedback(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    threadgroup float *local_float [[threadgroup(0)]],
    threadgroup int *local_int [[threadgroup(1)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  (void)dimensions;
  (void)input;
  (void)output;
  const uint gid = group.x;
  const uint lane_count = group_size.x;
  if (gid >= p.stage_count)
    return;
  if (lane == 0u)
    local_int[0] = enabled(metadata, p) ? 1 : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_int[0] == 0)
    return;
  device int *stage = reduced_stage_meta(metadata, p, gid);
  const int n = stage[0];
  const int next_n = stage[1];
  const int m = stage[2];
  device int *next_value = value_meta(metadata, p, gid + 1u);
  if (next_value[0] != next_n) {
    if (lane == 0u)
      fail(metadata, p, kDeviceNumericalFailure, int(gid), 9);
    return;
  }
  device int *fb = feedback_meta(metadata, p, gid);
  if (lane == 0u) {
    fb[0] = n;
    fb[1] = next_n;
    fb[2] = m;
  }
  const uint nx = p.state_capacity;
  const uint nu = p.control_capacity;
  const uint A = reduced_stage_A(p, gid);
  const uint B = reduced_stage_B(p, gid);
  const uint R = reduced_stage_R(p, gid);
  const uint M = reduced_stage_M(p, gid);
  const uint J = value_J(p, gid + 1u);
  const uint K = feedback_K(p, gid);
  const uint factor = feedback_factor(p, gid);
  const uint transition = feedback_transition(p, gid);

  if (m == 0) {
    for (int linear = int(lane); linear < next_n * n;
         linear += int(lane_count)) {
      const int row = linear / n;
      const int col = linear % n;
      workspace[transition + uint(row) * nx + uint(col)] =
          workspace[A + uint(row) * nx + uint(col)];
    }
    return;
  }

  const int columns = m + n;
  // Stage J*[B,A] once. The direct form contracted both indices of J for
  // every output entry, making the feedback construction quartic.
  threadgroup float *products = local_float;
  threadgroup float *augmented = products + next_n * columns;
  for (int linear = int(lane); linear < next_n * columns;
       linear += int(lane_count)) {
    const int row = linear / columns;
    const int col = linear % columns;
    float value = 0.0f;
    if (col < m) {
      for (int b = 0; b < next_n; ++b)
        value = fma(workspace[J + uint(row) * uint(next_n) + uint(b)],
                    workspace[B + uint(b) * nu + uint(col)], value);
    } else {
      const int x = col - m;
      for (int b = 0; b < next_n; ++b)
        value = fma(workspace[J + uint(row) * uint(next_n) + uint(b)],
                    workspace[A + uint(b) * nx + uint(x)], value);
    }
    products[linear] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int linear = int(lane); linear < m * columns;
       linear += int(lane_count)) {
    const int row = linear / columns;
    const int col = linear % columns;
    if (col < m) {
      float value = workspace[R + uint(row) * nu + uint(col)];
      for (int a = 0; a < next_n; ++a)
        value = fma(workspace[B + uint(a) * nu + uint(row)],
                    products[a * columns + col], value);
      augmented[linear] = value;
    } else {
      const int x = col - m;
      float value = -workspace[M + uint(x) * nu + uint(row)];
      for (int a = 0; a < next_n; ++a)
        value = fma(-workspace[B + uint(a) * nu + uint(row)],
                    products[a * columns + m + x], value);
      augmented[linear] = value;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int linear = int(lane); linear < m * m;
       linear += int(lane_count)) {
    const int row = linear / m;
    const int col = linear % m;
    workspace[factor + uint(linear)] =
        0.5f * (augmented[row * columns + col] +
                augmented[col * columns + row]);
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (lane == 0u) {
    local_int[1] = 1;
    float scale = 0.0f;
    for (int diagonal = 0; diagonal < m; ++diagonal)
      scale = fmax(
          scale, fabs(workspace[factor + uint(diagonal * m + diagonal)]));
    if (!(scale > 0.0f) || !isfinite(scale))
      local_int[1] = 0;
    for (int col = 0; col < m && local_int[1] != 0; ++col) {
      float diagonal = workspace[factor + uint(col * m + col)];
      for (int k = 0; k < col; ++k)
        diagonal = fma(-workspace[factor + uint(col * m + k)],
                       workspace[factor + uint(col * m + k)], diagonal);
      if (!(diagonal > p.rank_tolerance * scale) ||
          !isfinite(diagonal)) {
        local_int[1] = 0;
        break;
      }
      workspace[factor + uint(col * m + col)] = sqrt(diagonal);
      for (int row = col + 1; row < m; ++row) {
        float value = workspace[factor + uint(row * m + col)];
        for (int k = 0; k < col; ++k)
          value = fma(-workspace[factor + uint(row * m + k)],
                      workspace[factor + uint(col * m + k)], value);
        workspace[factor + uint(row * m + col)] =
            value / workspace[factor + uint(col * m + col)];
      }
    }
    if (local_int[1] == 0)
      fail(metadata, p, kDeviceNumericalFailure, int(gid), 9);
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (local_int[1] == 0)
    return;

  for (int rhs = int(lane); rhs < n; rhs += int(lane_count)) {
    for (int row = 0; row < m; ++row) {
      float value = augmented[row * columns + m + rhs];
      for (int col = 0; col < row; ++col)
        value = fma(-workspace[factor + uint(row * m + col)],
                    augmented[col * columns + m + rhs], value);
      augmented[row * columns + m + rhs] =
          value / workspace[factor + uint(row * m + row)];
    }
    for (int reverse = 0; reverse < m; ++reverse) {
      const int row = m - 1 - reverse;
      float value = augmented[row * columns + m + rhs];
      for (int col = row + 1; col < m; ++col)
        value = fma(-workspace[factor + uint(col * m + row)],
                    augmented[col * columns + m + rhs], value);
      augmented[row * columns + m + rhs] =
          value / workspace[factor + uint(row * m + row)];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int linear = int(lane); linear < m * n;
       linear += int(lane_count)) {
    const int row = linear / n;
    const int col = linear % n;
    workspace[K + uint(row) * nx + uint(col)] =
        augmented[row * columns + m + col];
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (int linear = int(lane); linear < next_n * n;
       linear += int(lane_count)) {
    const int row = linear / n;
    const int col = linear % n;
    float value = workspace[A + uint(row) * nx + uint(col)];
    for (int u = 0; u < m; ++u)
      value = fma(workspace[B + uint(row) * nu + uint(u)],
                  workspace[K + uint(u) * nx + uint(col)], value);
    workspace[transition + uint(row) * nx + uint(col)] = value;
  }
}

// Store f_i:p_{i+1}->p_i in reverse stage order. The generic affine-prefix
// scan then produces every costate suffix without a sequential Riccati pass.
kernel void clqr_initialize_costate_maps(
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
  if (gid >= p.stage_count || !enabled(metadata, p))
    return;
  device int *stage = reduced_stage_meta(metadata, p, gid);
  const int n = stage[0];
  const int next_n = stage[1];
  const int m = stage[2];
  const uint slot = p.stage_count - 1u - gid;
  device int *map = affine_meta(metadata, p, slot);
  map[0] = next_n;
  map[1] = n;
  const uint nx = p.state_capacity;
  const uint nu = p.control_capacity;
  const uint transition = feedback_transition(p, gid);
  const uint K = feedback_K(p, gid);
  const uint c = reduced_stage_c(p, gid);
  const uint q = reduced_stage_q(p, gid);
  const uint r = reduced_stage_r(p, gid);
  const uint J = value_J(p, gid + 1u);
  const uint linear = affine_linear(p, slot);
  const uint offset = affine_offset(p, slot);
  device float *future =
      workspace + p.float_scratch + gid * p.float_scratch_stride;
  for (int row = 0; row < n; ++row)
    for (int col = 0; col < next_n; ++col)
      workspace[linear + uint(row) * uint(next_n) + uint(col)] =
          workspace[transition + uint(col) * nx + uint(row)];
  for (int row = 0; row < next_n; ++row) {
    float value = 0.0f;
    for (int col = 0; col < next_n; ++col)
      value = fma(workspace[J + uint(row) * uint(next_n) + uint(col)],
                  workspace[c + uint(col)], value);
    future[row] = value;
  }
  for (int row = 0; row < n; ++row) {
    float value = workspace[q + uint(row)];
    for (int u = 0; u < m; ++u)
      value = fma(workspace[K + uint(u) * nx + uint(row)],
                  workspace[r + uint(u)], value);
    for (int next_row = 0; next_row < next_n; ++next_row)
      value = fma(
          workspace[transition + uint(next_row) * nx + uint(row)],
          future[next_row], value);
    workspace[offset + uint(row)] = value;
  }
}

kernel void clqr_recover_costates(
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
  if (gid > p.stage_count || !enabled(metadata, p))
    return;
  const uint nx = p.state_capacity;
  const uint target = p.reduced_costates + gid * nx;
  if (gid == p.stage_count) {
    const int terminal_n =
        metadata[p.reduced_stage_meta + 3u * p.stage_count];
    for (int row = 0; row < terminal_n; ++row)
      workspace[target + uint(row)] =
          workspace[reduced_terminal_q(p) + uint(row)];
    return;
  }
  const uint slot = p.stage_count - 1u - gid;
  device int *map = affine_meta(metadata, p, slot);
  const int left = map[0];
  const int right = map[1];
  const uint linear = affine_linear(p, slot);
  const uint offset = affine_offset(p, slot);
  for (int row = 0; row < right; ++row) {
    float value = workspace[offset + uint(row)];
    for (int col = 0; col < left; ++col)
      value = fma(
          workspace[linear + uint(row) * uint(left) + uint(col)],
          workspace[reduced_terminal_q(p) + uint(col)], value);
    workspace[target + uint(row)] = value;
  }
}

kernel void clqr_finalize_feedback(
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
  if (gid >= p.stage_count || !enabled(metadata, p))
    return;
  device int *stage = reduced_stage_meta(metadata, p, gid);
  const int next_n = stage[1];
  const int m = stage[2];
  const uint nx = p.state_capacity;
  const uint nu = p.control_capacity;
  const uint B = reduced_stage_B(p, gid);
  const uint c = reduced_stage_c(p, gid);
  const uint r = reduced_stage_r(p, gid);
  const uint J = value_J(p, gid + 1u);
  const uint k = feedback_k(p, gid);
  const uint factor = feedback_factor(p, gid);
  const uint offset = feedback_offset(p, gid);
  device float *future =
      workspace + p.float_scratch + gid * p.float_scratch_stride;
  const uint next_costate =
      p.reduced_costates + (gid + 1u) * nx;
  for (int row = 0; row < next_n; ++row) {
    float value = workspace[next_costate + uint(row)];
    for (int col = 0; col < next_n; ++col)
      value = fma(workspace[J + uint(row) * uint(next_n) + uint(col)],
                  workspace[c + uint(col)], value);
    future[row] = value;
  }
  for (int u = 0; u < m; ++u) {
    float value = -workspace[r + uint(u)];
    for (int row = 0; row < next_n; ++row)
      value = fma(-workspace[B + uint(row) * nu + uint(u)], future[row],
                  value);
    workspace[k + uint(u)] = value;
  }
  if (m > 0)
    solve_positive_definite(workspace + factor, m, workspace + k, 1, 1);
  for (int row = 0; row < next_n; ++row) {
    float value = workspace[c + uint(row)];
    for (int u = 0; u < m; ++u)
      value = fma(workspace[B + uint(row) * nu + uint(u)],
                  workspace[k + uint(u)], value);
    workspace[offset + uint(row)] = value;
  }
}

// Overwrite the costate-map leaves with the closed-loop state maps used by
// the same generic affine-prefix scan for primal reconstruction.
kernel void clqr_initialize_state_maps(
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
  if (gid >= p.stage_count || !enabled(metadata, p))
    return;
  device int *fb = feedback_meta(metadata, p, gid);
  const int n = fb[0];
  const int next_n = fb[1];
  device int *map = affine_meta(metadata, p, gid);
  map[0] = n;
  map[1] = next_n;
  const uint nx = p.state_capacity;
  const uint transition = feedback_transition(p, gid);
  const uint source_offset = feedback_offset(p, gid);
  const uint linear = affine_linear(p, gid);
  const uint target_offset = affine_offset(p, gid);
  for (int row = 0; row < next_n; ++row) {
    for (int col = 0; col < n; ++col)
      workspace[linear + uint(row) * uint(n) + uint(col)] =
          workspace[transition + uint(row) * nx + uint(col)];
    workspace[target_offset + uint(row)] =
        workspace[source_offset + uint(row)];
  }
}

kernel void clqr_reconstruct_primal(
    device const int *dimensions [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device float *workspace [[buffer(3)]],
    device int *metadata [[buffer(4)]],
    constant KernelParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  (void)input;
  if (gid > p.stage_count || !enabled(metadata, p))
    return;
  const uint nx = p.state_capacity;
  const uint nu = p.control_capacity;
  const int physical = state_param_physical(metadata, p, gid);
  const int reduced = state_param_reduced(metadata, p, gid);
  const uint z = p.reduced_states + gid * nx;
  if (gid == 0u) {
    for (int col = 0; col < reduced; ++col)
      workspace[z + uint(col)] =
          workspace[p.reduced_initial + uint(col)];
  } else {
    device int *map = affine_meta(metadata, p, gid - 1u);
    const int left = map[0];
    const int right = map[1];
    const uint linear = affine_linear(p, gid - 1u);
    const uint offset = affine_offset(p, gid - 1u);
    if (right != reduced) {
      fail(metadata, p, kDeviceNumericalFailure, int(gid), 11);
      return;
    }
    for (int row = 0; row < right; ++row) {
      float value = workspace[offset + uint(row)];
      for (int col = 0; col < left; ++col)
        value = fma(
            workspace[linear + uint(row) * uint(left) + uint(col)],
            workspace[p.reduced_initial + uint(col)], value);
      workspace[z + uint(row)] = value;
    }
  }

  const uint T = state_param_T(p, gid);
  const uint t = state_param_t(p, gid);
  const uint state_output = p.output_states + gid * nx;
  for (uint x = 0; x < nx; ++x)
    output[state_output + x] = 0.0f;
  for (int x = 0; x < physical; ++x) {
    float value = workspace[t + uint(x)];
    for (int col = 0; col < reduced; ++col)
      value = fma(workspace[T + uint(x) * nx + uint(col)],
                  workspace[z + uint(col)], value);
    output[state_output + uint(x)] = value;
  }
  if (gid == p.stage_count)
    return;

  device int *cp = control_param_meta(metadata, p, gid);
  const int physical_m = cp[0];
  const int control_state_dim = cp[1];
  const int reduced_m = cp[2];
  const uint v = p.reduced_controls + gid * nu;
  const uint K = feedback_K(p, gid);
  const uint k = feedback_k(p, gid);
  for (int row = 0; row < reduced_m; ++row) {
    float value = workspace[k + uint(row)];
    for (int col = 0; col < control_state_dim; ++col)
      value = fma(workspace[K + uint(row) * nx + uint(col)],
                  workspace[z + uint(col)], value);
    workspace[v + uint(row)] = value;
  }
  const uint Y = control_param_Y(p, gid);
  const uint Z = control_param_Z(p, gid);
  const uint y = control_param_y(p, gid);
  const uint control_output = p.output_controls + gid * nu;
  for (uint u = 0; u < nu; ++u)
    output[control_output + u] = 0.0f;
  for (int u = 0; u < physical_m; ++u) {
    float value = workspace[y + uint(u)];
    for (int col = 0; col < control_state_dim; ++col)
      value = fma(workspace[Y + uint(u) * nx + uint(col)],
                  workspace[z + uint(col)], value);
    for (int col = 0; col < reduced_m; ++col)
      value = fma(workspace[Z + uint(u) * nu + uint(col)],
                  workspace[v + uint(col)], value);
    output[control_output + uint(u)] = value;
  }
}
)CLQR_METAL";

} // namespace clqr::metal::detail

#endif // CLQR_SRC_METAL_REDUCTION_RECOVERY_SOURCE_H_
