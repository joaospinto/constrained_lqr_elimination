#ifndef CLQR_SRC_METAL_VALUE_AFFINE_SOURCE_H_
#define CLQR_SRC_METAL_VALUE_AFFINE_SOURCE_H_

namespace clqr::metal::detail {

inline constexpr char kMetalValueAffineSource[] = R"CLQR_METAL(
#include <metal_stdlib>
using namespace metal;

struct KernelParams {
  uint stage_count;
  uint state_capacity;
  uint control_capacity;
  uint mixed_capacity;
  uint state_constraint_capacity;
  uint terminal_constraint_capacity;
  float rank_tolerance;
  float consistency_tolerance;

  uint input_A; uint input_B; uint input_c; uint input_Q;
  uint input_R; uint input_M; uint input_q; uint input_r;
  uint input_C; uint input_D; uint input_d; uint input_E;
  uint input_e; uint input_terminal_E; uint input_terminal_e;
  uint input_initial_state;

  uint output_objective; uint output_states; uint output_controls;
  uint output_initial_multiplier; uint output_dynamics_multipliers;
  uint output_mixed_multipliers; uint output_state_multipliers;
  uint output_terminal_state_multiplier;

  uint relation_data; uint relation_stride;
  uint state_param_data; uint state_param_stride;
  uint control_param_data; uint control_param_stride;
  uint reduced_stage_data; uint reduced_stage_stride;
  uint reduced_terminal_data;
  uint value_data; uint value_stride;
  uint feedback_data; uint feedback_stride;
  uint affine_data; uint affine_stride;
  uint reduced_initial; uint reduced_costates;
  uint reduced_states; uint reduced_controls;
  uint dual_param_data; uint dual_param_stride;
  uint state_dual_param_data; uint state_dual_param_stride;
  uint dual_relation_data; uint dual_relation_stride;
  uint dual_node_data; uint dual_node_stride;
  uint objective_tree; uint float_scratch; uint float_scratch_stride;

  uint status; uint relation_meta; uint state_param_meta;
  uint state_param_free_columns; uint control_param_meta;
  uint control_param_free_columns; uint reduced_stage_meta;
  uint value_meta; uint feedback_meta; uint affine_meta;
  uint dual_param_meta; uint dual_param_free_columns;
  uint state_dual_param_meta; uint dual_relation_meta;
  uint dual_node_meta; uint int_scratch; uint int_scratch_stride;

  uint child_offset; uint parent_offset; uint child_count; uint parent_count;
  uint temporary_offset; uint phase_detail;
  uint padding0; uint padding1; uint padding2;
};

constant int kOk = 0;
constant int kInfeasible = 1;
constant int kNumericalFailure = 2;
constant int kInvalidInput = 3;

inline bool enabled(device int *iw, constant KernelParams &p) {
  return atomic_load_explicit(
             reinterpret_cast<device atomic_int *>(iw + p.status),
             memory_order_relaxed) == kOk;
}

inline void fail(device int *iw, constant KernelParams &p, int code, int stage,
                 int detail) {
  device atomic_int *status =
      reinterpret_cast<device atomic_int *>(iw + p.status);
  int expected = kOk;
  bool claimed = false;
  while (expected == kOk &&
         !(claimed = atomic_compare_exchange_weak_explicit(
               status, &expected, code, memory_order_relaxed,
               memory_order_relaxed))) {
  }
  if (claimed) {
    iw[p.status + 1] = stage;
    iw[p.status + 2] = detail;
  }
}

inline device float *reduced_stage(device float *w,
                                   constant KernelParams &p, uint stage) {
  return w + p.reduced_stage_data + stage * p.reduced_stage_stride;
}
inline device int *reduced_stage_meta(device int *iw,
                                      constant KernelParams &p, uint stage) {
  return iw + p.reduced_stage_meta + 3 * stage;
}

inline device float *value_A(device float *w, constant KernelParams &p,
                             uint slot) {
  return w + p.value_data + slot * p.value_stride;
}
inline device float *value_C(device float *w, constant KernelParams &p,
                             uint slot) {
  uint n2 = p.state_capacity * p.state_capacity;
  return value_A(w, p, slot) + n2;
}
inline device float *value_J(device float *w, constant KernelParams &p,
                             uint slot) {
  uint n2 = p.state_capacity * p.state_capacity;
  return value_A(w, p, slot) + 2 * n2;
}
inline device int *value_meta(device int *iw, constant KernelParams &p,
                              uint slot) {
  return iw + p.value_meta + 2 * slot;
}

inline device float *scratch(device float *w, constant KernelParams &p,
                             uint gid) {
  return w + p.float_scratch + gid * p.float_scratch_stride;
}

inline bool factor_positive_definite(device const float *matrix, int stride,
                                     int n, float tolerance,
                                     device float *lower) {
  for (int i = 0; i < n * n; ++i) lower[i] = 0.0f;
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col <= row; ++col) {
      float value = matrix[row * stride + col];
      for (int k = 0; k < col; ++k)
        value = fma(-lower[row * n + k], lower[col * n + k], value);
      if (row == col) {
        if (!(value > tolerance) || !isfinite(value)) return false;
        lower[row * n + col] = sqrt(value);
      } else {
        lower[row * n + col] = value / lower[col * n + col];
      }
    }
  }
  return true;
}

inline void set_invalid_value(device int *iw, constant KernelParams &p,
                              uint slot) {
  device int *meta = value_meta(iw, p, slot);
  meta[0] = -1;
  meta[1] = 0;
}

inline bool invalid_value(device int *iw, constant KernelParams &p,
                          uint slot) {
  return value_meta(iw, p, slot)[0] < 0;
}

inline void cooperative_barrier() {
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}

// Cooperative counterpart to solve_general_multiple_rhs for one value
// composition owned by one threadgroup. Lane zero retains the serial scale
// and pivot searches so the factorization makes exactly the same decisions.
// The lanes only distribute independent element updates and right-hand sides.
inline bool solve_general_multiple_rhs_cooperative(
    device float *augmented, int n, int columns, float tolerance,
    device float *row_scales, uint lane, uint lane_count,
    threadgroup int *solve_ok) {
  if (lane == 0) {
    *solve_ok = 1;
    for (int row = 0; row < n; ++row) {
      float scale = 0.0f;
      for (int col = 0; col < n; ++col)
        scale = max(scale, abs(augmented[row * columns + col]));
      if (!(scale > 0.0f)) {
        *solve_ok = 0;
        break;
      }
      row_scales[row] = scale;
    }
  }
  cooperative_barrier();
  if (*solve_ok == 0) return false;

  for (uint index = lane; index < uint(n * columns); index += lane_count) {
    int row = int(index) / columns;
    augmented[index] /= row_scales[row];
  }
  cooperative_barrier();

  for (int pivot_col = 0; pivot_col < n; ++pivot_col) {
    if (lane == 0) {
      int best_row = -1;
      float best = tolerance;
      for (int row = pivot_col; row < n; ++row) {
        float candidate = abs(augmented[row * columns + pivot_col]);
        if (candidate > best) {
          best = candidate;
          best_row = row;
        }
      }
      // Zero denotes failure; otherwise encode the row as row + 1.
      *solve_ok = best_row + 1;
    }
    cooperative_barrier();
    if (*solve_ok == 0) return false;

    int best_row = *solve_ok - 1;
    if (best_row != pivot_col) {
      for (uint col = lane; col < uint(columns); col += lane_count) {
        float temporary = augmented[pivot_col * columns + int(col)];
        augmented[pivot_col * columns + int(col)] =
            augmented[best_row * columns + int(col)];
        augmented[best_row * columns + int(col)] = temporary;
      }
    }
    cooperative_barrier();

    float pivot = augmented[pivot_col * columns + pivot_col];
    for (uint row = uint(pivot_col + 1) + lane; row < uint(n);
         row += lane_count) {
      float factor = augmented[int(row) * columns + pivot_col] / pivot;
      augmented[int(row) * columns + pivot_col] = factor;
    }
    cooperative_barrier();

    uint remaining_rows = uint(n - pivot_col - 1);
    uint remaining_columns = uint(columns - pivot_col - 1);
    uint update_count = remaining_rows * remaining_columns;
    for (uint index = lane; index < update_count; index += lane_count) {
      int row = pivot_col + 1 + int(index / remaining_columns);
      int col = pivot_col + 1 + int(index % remaining_columns);
      float factor = augmented[row * columns + pivot_col];
      augmented[row * columns + col] =
          fma(-factor, augmented[pivot_col * columns + col],
              augmented[row * columns + col]);
    }
    cooperative_barrier();
  }

  int rhs_count = columns - n;
  for (uint rhs_index = lane; rhs_index < uint(rhs_count);
       rhs_index += lane_count) {
    for (int row = 0; row < n; ++row) {
      float value = augmented[row * columns + n + int(rhs_index)];
      for (int col = 0; col < row; ++col)
        value =
            fma(-augmented[row * columns + col],
                augmented[col * columns + n + int(rhs_index)], value);
      augmented[row * columns + n + int(rhs_index)] = value;
    }
    for (int reverse = 0; reverse < n; ++reverse) {
      int row = n - 1 - reverse;
      float value = augmented[row * columns + n + int(rhs_index)];
      for (int col = row + 1; col < n; ++col)
        value =
            fma(-augmented[row * columns + col],
                augmented[col * columns + n + int(rhs_index)], value);
      augmented[row * columns + n + int(rhs_index)] =
          value / augmented[row * columns + row];
    }
  }
  cooperative_barrier();
  return true;
}

inline bool solve_general_multiple_rhs_cooperative_threadgroup(
    threadgroup float *augmented, int n, int columns, float tolerance,
    threadgroup float *row_scales, uint lane, uint lane_count,
    threadgroup int *solve_ok) {
  if (lane == 0) {
    *solve_ok = 1;
    for (int row = 0; row < n; ++row) {
      float scale = 0.0f;
      for (int col = 0; col < n; ++col)
        scale = max(scale, abs(augmented[row * columns + col]));
      if (!(scale > 0.0f)) {
        *solve_ok = 0;
        break;
      }
      row_scales[row] = scale;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (*solve_ok == 0) return false;

  for (uint index = lane; index < uint(n * columns); index += lane_count) {
    int row = int(index) / columns;
    augmented[index] /= row_scales[row];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int pivot_col = 0; pivot_col < n; ++pivot_col) {
    if (lane == 0) {
      int best_row = -1;
      float best = tolerance;
      for (int row = pivot_col; row < n; ++row) {
        float candidate = abs(augmented[row * columns + pivot_col]);
        if (candidate > best) {
          best = candidate;
          best_row = row;
        }
      }
      *solve_ok = best_row + 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (*solve_ok == 0) return false;

    int best_row = *solve_ok - 1;
    if (best_row != pivot_col) {
      for (uint col = lane; col < uint(columns); col += lane_count) {
        float temporary = augmented[pivot_col * columns + int(col)];
        augmented[pivot_col * columns + int(col)] =
            augmented[best_row * columns + int(col)];
        augmented[best_row * columns + int(col)] = temporary;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float pivot = augmented[pivot_col * columns + pivot_col];
    for (uint row = uint(pivot_col + 1) + lane; row < uint(n);
         row += lane_count) {
      float factor = augmented[int(row) * columns + pivot_col] / pivot;
      augmented[int(row) * columns + pivot_col] = factor;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint remaining_rows = uint(n - pivot_col - 1);
    uint remaining_columns = uint(columns - pivot_col - 1);
    uint update_count = remaining_rows * remaining_columns;
    for (uint index = lane; index < update_count; index += lane_count) {
      int row = pivot_col + 1 + int(index / remaining_columns);
      int col = pivot_col + 1 + int(index % remaining_columns);
      float factor = augmented[row * columns + pivot_col];
      augmented[row * columns + col] =
          fma(-factor, augmented[pivot_col * columns + col],
              augmented[row * columns + col]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  int rhs_count = columns - n;
  for (uint rhs_index = lane; rhs_index < uint(rhs_count);
       rhs_index += lane_count) {
    for (int row = 0; row < n; ++row) {
      float value = augmented[row * columns + n + int(rhs_index)];
      for (int col = 0; col < row; ++col)
        value =
            fma(-augmented[row * columns + col],
                augmented[col * columns + n + int(rhs_index)], value);
      augmented[row * columns + n + int(rhs_index)] = value;
    }
    for (int reverse = 0; reverse < n; ++reverse) {
      int row = n - 1 - reverse;
      float value = augmented[row * columns + n + int(rhs_index)];
      for (int col = row + 1; col < n; ++col)
        value =
            fma(-augmented[row * columns + col],
                augmented[col * columns + n + int(rhs_index)], value);
      augmented[row * columns + n + int(rhs_index)] =
          value / augmented[row * columns + row];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return true;
}

inline void copy_value_cooperative(
    device float *w, device int *iw, constant KernelParams &p,
    uint input_slot, uint output_slot, uint lane, uint lane_count) {
  device int *input_meta = value_meta(iw, p, input_slot);
  device int *output_meta = value_meta(iw, p, output_slot);
  int left = input_meta[0];
  int right = input_meta[1];
  if (lane == 0) {
    output_meta[0] = left;
    output_meta[1] = right;
  }
  cooperative_barrier();
  if (left >= 0) {
    device float *in_A = value_A(w, p, input_slot);
    device float *in_C = value_C(w, p, input_slot);
    device float *in_J = value_J(w, p, input_slot);
    device float *out_A = value_A(w, p, output_slot);
    device float *out_C = value_C(w, p, output_slot);
    device float *out_J = value_J(w, p, output_slot);
    for (uint i = lane; i < uint(right * left); i += lane_count)
      out_A[i] = in_A[i];
    for (uint i = lane; i < uint(right * right); i += lane_count)
      out_C[i] = in_C[i];
    for (uint i = lane; i < uint(left * left); i += lane_count)
      out_J[i] = in_J[i];
  }
  cooperative_barrier();
}

inline bool compose_value_cooperative(
    device float *w, device int *iw, constant KernelParams &p,
    uint first_slot, uint second_slot, uint output_slot, uint scratch_slot,
    int diagnostic_stage, uint lane, uint lane_count,
    threadgroup int *solve_ok) {
  bool first_invalid = invalid_value(iw, p, first_slot);
  bool second_invalid = invalid_value(iw, p, second_slot);
  if (first_invalid) {
    if (second_invalid) {
      if (lane == 0) set_invalid_value(iw, p, output_slot);
      cooperative_barrier();
    } else {
      copy_value_cooperative(w, iw, p, second_slot, output_slot, lane,
                             lane_count);
    }
    return true;
  }
  if (second_invalid) {
    copy_value_cooperative(w, iw, p, first_slot, output_slot, lane,
                           lane_count);
    return true;
  }

  device int *first_meta = value_meta(iw, p, first_slot);
  device int *second_meta = value_meta(iw, p, second_slot);
  int left = first_meta[0];
  int shared = first_meta[1];
  int right = second_meta[1];
  if (lane == 0) {
    *solve_ok = shared == second_meta[0] ? 1 : 0;
    if (*solve_ok == 0)
      fail(iw, p, kNumericalFailure, diagnostic_stage, 20);
    else {
      device int *output_meta = value_meta(iw, p, output_slot);
      output_meta[0] = left;
      output_meta[1] = right;
    }
  }
  cooperative_barrier();
  if (*solve_ok == 0) return false;

  device float *first_A = value_A(w, p, first_slot);
  device float *first_C = value_C(w, p, first_slot);
  device float *first_J = value_J(w, p, first_slot);
  device float *second_A = value_A(w, p, second_slot);
  device float *second_C = value_C(w, p, second_slot);
  device float *second_J = value_J(w, p, second_slot);
  device float *output_A = value_A(w, p, output_slot);
  device float *output_C = value_C(w, p, output_slot);
  device float *output_J = value_J(w, p, output_slot);
  if (shared == 0) {
    for (uint i = lane; i < uint(right * left); i += lane_count)
      output_A[i] = 0.0f;
    for (uint i = lane; i < uint(left * left); i += lane_count)
      output_J[i] = first_J[i];
    for (uint i = lane; i < uint(right * right); i += lane_count)
      output_C[i] = second_C[i];
    cooperative_barrier();
    return true;
  }

  int rhs_columns = left + shared;
  int columns = shared + rhs_columns;
  device float *augmented = scratch(w, p, scratch_slot);
  device float *factors = augmented + shared * columns;
  uint augmented_size = uint(shared * columns);
  for (uint index = lane; index < augmented_size; index += lane_count) {
    int row = int(index) / columns;
    int col = int(index) - row * columns;
    float value = 0.0f;
    if (col < shared) {
      value = row == col ? 1.0f : 0.0f;
      for (int k = 0; k < shared; ++k)
        value = fma(first_C[row * shared + k],
                    second_J[k * shared + col], value);
    } else if (col < shared + left) {
      value = first_A[row * left + col - shared];
    } else {
      value = first_C[row * shared + col - shared - left];
    }
    augmented[index] = value;
  }
  cooperative_barrier();
  bool factored = solve_general_multiple_rhs_cooperative(
      augmented, shared, columns, p.rank_tolerance, factors, lane,
      lane_count, solve_ok);
  if (!factored) {
    if (lane == 0)
      fail(iw, p, kNumericalFailure, diagnostic_stage, 20);
    cooperative_barrier();
    return false;
  }

  for (uint index = lane; index < uint(right * left);
       index += lane_count) {
    int row = int(index) / left;
    int col = int(index) - row * left;
    float value = 0.0f;
    for (int k = 0; k < shared; ++k)
      value = fma(second_A[row * shared + k],
                  augmented[k * columns + shared + col], value);
    output_A[index] = value;
  }

  // Form the two quadratic updates as pairs of ordinary matrix products.
  // This reassociates FP32 arithmetic, but reduces each update from O(n^4)
  // scalar work to O(n^3).  The solved augmented system is no longer needed
  // after the intermediates below have consumed it, so its existing storage
  // can hold the final J update without increasing the scratch plan.
  cooperative_barrier();
  for (uint index = lane; index < uint(right * shared);
       index += lane_count) {
    int row = int(index) / shared;
    int col = int(index) - row * shared;
    float value = 0.0f;
    for (int a = 0; a < shared; ++a)
      value = fma(second_A[row * shared + a],
                  augmented[a * columns + shared + left + col], value);
    output_C[index] = value;
  }
  cooperative_barrier();
  for (uint index = lane; index < uint(right * right);
       index += lane_count) {
    int row = int(index) / right;
    int col = int(index) - row * right;
    float value = second_C[index];
    for (int b = 0; b < shared; ++b)
      value =
          fma(output_C[row * shared + b],
              second_A[col * shared + b], value);
    output_J[index] = value;
  }
  cooperative_barrier();
  for (uint index = lane; index < uint(right * right);
       index += lane_count)
    output_C[index] = output_J[index];
  cooperative_barrier();

  for (uint index = lane; index < uint(shared * left);
       index += lane_count) {
    int row = int(index) / left;
    int col = int(index) - row * left;
    float value = 0.0f;
    for (int b = 0; b < shared; ++b)
      value = fma(second_J[row * shared + b],
                  augmented[b * columns + shared + col], value);
    output_J[index] = value;
  }
  cooperative_barrier();
  for (uint index = lane; index < uint(left * left);
       index += lane_count) {
    int row = int(index) / left;
    int col = int(index) - row * left;
    float value = first_J[index];
    for (int a = 0; a < shared; ++a)
      value =
          fma(first_A[a * left + row], output_J[a * left + col], value);
    augmented[index] = value;
  }
  cooperative_barrier();
  for (uint index = lane; index < uint(left * left);
       index += lane_count)
    output_J[index] = augmented[index];
  cooperative_barrier();

  for (uint index = lane; index < uint(left * left);
       index += lane_count) {
    int row = int(index) / left;
    int col = int(index) - row * left;
    if (col > row) {
      float value =
          0.5f * (output_J[index] + output_J[col * left + row]);
      output_J[index] = value;
      output_J[col * left + row] = value;
    }
  }
  for (uint index = lane; index < uint(right * right);
       index += lane_count) {
    int row = int(index) / right;
    int col = int(index) - row * right;
    if (col > row) {
      float value =
          0.5f * (output_C[index] + output_C[col * right + row]);
      output_C[index] = value;
      output_C[col * right + row] = value;
    }
  }
  cooperative_barrier();
  return true;
}

inline bool compose_value_cooperative_threadgroup(
    device float *w, device int *iw, constant KernelParams &p,
    uint first_slot, uint second_slot, uint output_slot, int diagnostic_stage,
    uint lane, uint lane_count, threadgroup int *solve_ok,
    threadgroup float *local) {
  bool first_invalid = invalid_value(iw, p, first_slot);
  bool second_invalid = invalid_value(iw, p, second_slot);
  if (first_invalid) {
    if (second_invalid) {
      if (lane == 0) set_invalid_value(iw, p, output_slot);
      cooperative_barrier();
    } else {
      copy_value_cooperative(w, iw, p, second_slot, output_slot, lane,
                             lane_count);
    }
    return true;
  }
  if (second_invalid) {
    copy_value_cooperative(w, iw, p, first_slot, output_slot, lane,
                           lane_count);
    return true;
  }

  device int *first_meta = value_meta(iw, p, first_slot);
  device int *second_meta = value_meta(iw, p, second_slot);
  int left = first_meta[0];
  int shared = first_meta[1];
  int right = second_meta[1];
  if (lane == 0) {
    *solve_ok = shared == second_meta[0] ? 1 : 0;
    if (*solve_ok == 0)
      fail(iw, p, kNumericalFailure, diagnostic_stage, 20);
    else {
      device int *output_meta = value_meta(iw, p, output_slot);
      output_meta[0] = left;
      output_meta[1] = right;
    }
  }
  cooperative_barrier();
  if (*solve_ok == 0) return false;

  device float *first_A = value_A(w, p, first_slot);
  device float *first_C = value_C(w, p, first_slot);
  device float *first_J = value_J(w, p, first_slot);
  device float *second_A = value_A(w, p, second_slot);
  device float *second_C = value_C(w, p, second_slot);
  device float *second_J = value_J(w, p, second_slot);
  device float *output_A = value_A(w, p, output_slot);
  device float *output_C = value_C(w, p, output_slot);
  device float *output_J = value_J(w, p, output_slot);
  if (shared == 0) {
    for (uint i = lane; i < uint(right * left); i += lane_count)
      output_A[i] = 0.0f;
    for (uint i = lane; i < uint(left * left); i += lane_count)
      output_J[i] = first_J[i];
    for (uint i = lane; i < uint(right * right); i += lane_count)
      output_C[i] = second_C[i];
    cooperative_barrier();
    return true;
  }

  int rhs_columns = left + shared;
  int columns = shared + rhs_columns;
  threadgroup float *augmented = local;
  threadgroup float *factors = augmented + shared * columns;
  uint augmented_size = uint(shared * columns);
  for (uint index = lane; index < augmented_size; index += lane_count) {
    int row = int(index) / columns;
    int col = int(index) - row * columns;
    float value = 0.0f;
    if (col < shared) {
      value = row == col ? 1.0f : 0.0f;
      for (int k = 0; k < shared; ++k)
        value = fma(first_C[row * shared + k],
                    second_J[k * shared + col], value);
    } else if (col < shared + left) {
      value = first_A[row * left + col - shared];
    } else {
      value = first_C[row * shared + col - shared - left];
    }
    augmented[index] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool factored = solve_general_multiple_rhs_cooperative_threadgroup(
      augmented, shared, columns, p.rank_tolerance, factors, lane,
      lane_count, solve_ok);
  if (!factored) {
    if (lane == 0)
      fail(iw, p, kNumericalFailure, diagnostic_stage, 20);
    cooperative_barrier();
    return false;
  }

  for (uint index = lane; index < uint(right * left);
       index += lane_count) {
    int row = int(index) / left;
    int col = int(index) - row * left;
    float value = 0.0f;
    for (int k = 0; k < shared; ++k)
      value = fma(second_A[row * shared + k],
                  augmented[k * columns + shared + col], value);
    output_A[index] = value;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = lane; index < uint(right * shared);
       index += lane_count) {
    int row = int(index) / shared;
    int col = int(index) - row * shared;
    float value = 0.0f;
    for (int a = 0; a < shared; ++a)
      value = fma(second_A[row * shared + a],
                  augmented[a * columns + shared + left + col], value);
    output_C[index] = value;
  }
  cooperative_barrier();
  for (uint index = lane; index < uint(right * right);
       index += lane_count) {
    int row = int(index) / right;
    int col = int(index) - row * right;
    float value = second_C[index];
    for (int b = 0; b < shared; ++b)
      value =
          fma(output_C[row * shared + b],
              second_A[col * shared + b], value);
    output_J[index] = value;
  }
  cooperative_barrier();
  for (uint index = lane; index < uint(right * right);
       index += lane_count)
    output_C[index] = output_J[index];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint index = lane; index < uint(shared * left);
       index += lane_count) {
    int row = int(index) / left;
    int col = int(index) - row * left;
    float value = 0.0f;
    for (int b = 0; b < shared; ++b)
      value = fma(second_J[row * shared + b],
                  augmented[b * columns + shared + col], value);
    output_J[index] = value;
  }
  cooperative_barrier();
  for (uint index = lane; index < uint(left * left);
       index += lane_count) {
    int row = int(index) / left;
    int col = int(index) - row * left;
    float value = first_J[index];
    for (int a = 0; a < shared; ++a)
      value =
          fma(first_A[a * left + row], output_J[a * left + col], value);
    augmented[index] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = lane; index < uint(left * left);
       index += lane_count)
    output_J[index] = augmented[index];
  cooperative_barrier();

  for (uint index = lane; index < uint(left * left);
       index += lane_count) {
    int row = int(index) / left;
    int col = int(index) - row * left;
    if (col > row) {
      float value =
          0.5f * (output_J[index] + output_J[col * left + row]);
      output_J[index] = value;
      output_J[col * left + row] = value;
    }
  }
  for (uint index = lane; index < uint(right * right);
       index += lane_count) {
    int row = int(index) / right;
    int col = int(index) - row * right;
    if (col > row) {
      float value =
          0.5f * (output_C[index] + output_C[col * right + row]);
      output_C[index] = value;
      output_C[col * right + row] = value;
    }
  }
  cooperative_barrier();
  return true;
}

kernel void clqr_build_value_leaves(
    device const int *dims [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], device float *w [[buffer(3)]],
    device int *iw [[buffer(4)]], constant KernelParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  (void)dims; (void)input; (void)output;
  uint gid = group.x;
  threadgroup int solve_ok;
  if (gid > p.stage_count) return;
  if (lane == 0) solve_ok = enabled(iw, p) ? 1 : 0;
  cooperative_barrier();
  if (solve_ok == 0) return;
  int n = gid == p.stage_count
              ? iw[p.reduced_stage_meta + 3 * p.stage_count]
              : reduced_stage_meta(iw, p, gid)[0];
  device int *meta = value_meta(iw, p, gid);
  if (lane == 0) {
    meta[0] = n;
    meta[1] =
        gid == p.stage_count ? 0 : reduced_stage_meta(iw, p, gid)[1];
  }
  cooperative_barrier();
  device float *out_A = value_A(w, p, gid);
  device float *out_C = value_C(w, p, gid);
  device float *out_J = value_J(w, p, gid);
  uint nx = p.state_capacity;
  if (gid == p.stage_count) {
    device const float *Q = w + p.reduced_terminal_data;
    for (uint index = lane; index < uint(n * n); index += group_size.x) {
      int row = int(index) / n;
      int col = int(index) - row * n;
      out_J[index] = Q[row * nx + col];
    }
    return;
  }
  device int *stage_meta = reduced_stage_meta(iw, p, gid);
  int next_n = stage_meta[1];
  int m = stage_meta[2];
  device const float *stage = reduced_stage(w, p, gid);
  device const float *A = stage;
  device const float *B = A + nx * nx;
  device const float *stage_c = B + nx * p.control_capacity;
  device const float *Q = stage_c + nx;
  device const float *R = Q + nx * nx;
  device const float *M = R + p.control_capacity * p.control_capacity;
  if (m == 0) {
    for (uint index = lane; index < uint(next_n * n);
         index += group_size.x) {
      int row = int(index) / n;
      int col = int(index) - row * n;
      out_A[index] = A[row * nx + col];
    }
    for (uint index = lane; index < uint(n * n); index += group_size.x) {
      int row = int(index) / n;
      int col = int(index) - row * n;
      out_J[index] = Q[row * nx + col];
    }
    for (uint i = lane; i < uint(next_n * next_n); i += group_size.x)
      out_C[i] = 0.0f;
    return;
  }
  int rhs_count = n + next_n;
  device float *local = scratch(w, p, gid);
  device float *lower = local;
  device float *rhs = lower + m * m;
  if (lane == 0) {
    solve_ok =
        factor_positive_definite(R, p.control_capacity, m, p.rank_tolerance,
                                 lower)
            ? 1
            : 0;
    if (solve_ok == 0)
      fail(iw, p, kNumericalFailure, gid, 19);
  }
  cooperative_barrier();
  if (solve_ok == 0) return;
  for (uint index = lane; index < uint(m * rhs_count);
       index += group_size.x) {
    int row = int(index) / rhs_count;
    int col = int(index) - row * rhs_count;
    rhs[index] =
        col < n ? M[col * p.control_capacity + row]
                : B[(col - n) * p.control_capacity + row];
  }
  cooperative_barrier();
  for (uint column = lane; column < uint(rhs_count);
       column += group_size.x) {
    for (int row = 0; row < m; ++row) {
      float value = rhs[row * rhs_count + int(column)];
      for (int k = 0; k < row; ++k)
        value = fma(-lower[row * m + k],
                    rhs[k * rhs_count + int(column)], value);
      rhs[row * rhs_count + int(column)] = value / lower[row * m + row];
    }
    for (int reverse = 0; reverse < m; ++reverse) {
      int row = m - 1 - reverse;
      float value = rhs[row * rhs_count + int(column)];
      for (int k = row + 1; k < m; ++k)
        value = fma(-lower[k * m + row],
                    rhs[k * rhs_count + int(column)], value);
      rhs[row * rhs_count + int(column)] = value / lower[row * m + row];
    }
  }
  cooperative_barrier();
  for (uint index = lane; index < uint(n * n); index += group_size.x) {
    int row = int(index) / n;
    int col = int(index) - row * n;
    float value = Q[row * nx + col];
    for (int u = 0; u < m; ++u)
      value = fma(-M[row * p.control_capacity + u],
                  rhs[u * rhs_count + col], value);
    out_J[index] = value;
  }
  for (uint index = lane; index < uint(next_n * n);
       index += group_size.x) {
    int row = int(index) / n;
    int col = int(index) - row * n;
    float value = A[row * nx + col];
    for (int u = 0; u < m; ++u)
      value = fma(-B[row * p.control_capacity + u],
                  rhs[u * rhs_count + col], value);
    out_A[index] = value;
  }
  for (uint index = lane; index < uint(next_n * next_n);
       index += group_size.x) {
    int row = int(index) / next_n;
    int col = int(index) - row * next_n;
    float value = 0.0f;
    for (int u = 0; u < m; ++u)
      value = fma(B[row * p.control_capacity + u],
                  rhs[u * rhs_count + n + col], value);
    out_C[index] = value;
  }
}

kernel void clqr_reduce_value(
    device const int *dims [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], device float *w [[buffer(3)]],
    device int *iw [[buffer(4)]], constant KernelParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  (void)dims; (void)input; (void)output;
  uint gid = group.x;
  threadgroup int solve_ok;
  if (gid >= p.parent_count) return;
  if (lane == 0) solve_ok = enabled(iw, p) ? 1 : 0;
  cooperative_barrier();
  if (solve_ok == 0) return;
  uint left = p.child_offset + 2 * gid;
  uint parent = p.parent_offset + gid;
  if (2 * gid + 1 >= p.child_count) {
    copy_value_cooperative(w, iw, p, left, parent, lane, group_size.x);
    return;
  }
  compose_value_cooperative(w, iw, p, left, left + 1, parent, gid,
                            int(p.parent_offset + gid), lane, group_size.x,
                            &solve_ok);
}

kernel void clqr_reduce_value_threadgroup(
    device const int *dims [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], device float *w [[buffer(3)]],
    device int *iw [[buffer(4)]], constant KernelParams &p [[buffer(5)]],
    threadgroup float *local [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  (void)dims; (void)input; (void)output;
  uint gid = group.x;
  threadgroup int solve_ok;
  if (gid >= p.parent_count) return;
  if (lane == 0) solve_ok = enabled(iw, p) ? 1 : 0;
  cooperative_barrier();
  if (solve_ok == 0) return;
  uint left = p.child_offset + 2 * gid;
  uint parent = p.parent_offset + gid;
  if (2 * gid + 1 >= p.child_count) {
    copy_value_cooperative(w, iw, p, left, parent, lane, group_size.x);
    return;
  }
  compose_value_cooperative_threadgroup(
      w, iw, p, left, left + 1, parent, int(p.parent_offset + gid), lane,
      group_size.x, &solve_ok, local);
}

kernel void clqr_expand_value_context(
    device const int *dims [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], device float *w [[buffer(3)]],
    device int *iw [[buffer(4)]], constant KernelParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  (void)dims; (void)input; (void)output;
  uint gid = group.x;
  threadgroup int solve_ok;
  if (gid >= p.parent_count) return;
  if (lane == 0) {
    if (p.parent_count == 1 && p.child_count <= 2)
      set_invalid_value(iw, p, p.parent_offset);
    solve_ok = enabled(iw, p) ? 1 : 0;
  }
  cooperative_barrier();
  if (solve_ok == 0) return;
  uint left = p.child_offset + 2 * gid;
  uint parent = p.parent_offset + gid;
  if (2 * gid + 1 >= p.child_count) {
    copy_value_cooperative(w, iw, p, parent, left, lane, group_size.x);
    return;
  }
  if (compose_value_cooperative(w, iw, p, left + 1, parent, left, gid,
                                int(left), lane, group_size.x, &solve_ok))
    copy_value_cooperative(w, iw, p, parent, left + 1, lane, group_size.x);
}

kernel void clqr_expand_value_context_threadgroup(
    device const int *dims [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], device float *w [[buffer(3)]],
    device int *iw [[buffer(4)]], constant KernelParams &p [[buffer(5)]],
    threadgroup float *local [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  (void)dims; (void)input; (void)output;
  uint gid = group.x;
  threadgroup int solve_ok;
  if (gid >= p.parent_count) return;
  if (lane == 0) {
    if (p.parent_count == 1 && p.child_count <= 2)
      set_invalid_value(iw, p, p.parent_offset);
    solve_ok = enabled(iw, p) ? 1 : 0;
  }
  cooperative_barrier();
  if (solve_ok == 0) return;
  uint left = p.child_offset + 2 * gid;
  uint parent = p.parent_offset + gid;
  if (2 * gid + 1 >= p.child_count) {
    copy_value_cooperative(w, iw, p, parent, left, lane, group_size.x);
    return;
  }
  if (compose_value_cooperative_threadgroup(
          w, iw, p, left + 1, parent, left, int(left), lane, group_size.x,
          &solve_ok, local))
    copy_value_cooperative(w, iw, p, parent, left + 1, lane, group_size.x);
}

kernel void clqr_finalize_value_suffix(
    device const int *dims [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], device float *w [[buffer(3)]],
    device int *iw [[buffer(4)]], constant KernelParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  (void)dims; (void)input; (void)output;
  uint gid = group.x;
  threadgroup int solve_ok;
  if (gid >= p.parent_count) return;
  if (lane == 0) {
    if (p.parent_count == 1 && p.child_count <= 2)
      set_invalid_value(iw, p, p.parent_offset);
    solve_ok = enabled(iw, p) ? 1 : 0;
  }
  cooperative_barrier();
  if (solve_ok == 0) return;
  uint left = 2 * gid;
  uint right = left + 1;
  uint parent = p.parent_offset + gid;
  uint temporary = p.temporary_offset + gid;
  if (right >= p.child_count) {
    if (!invalid_value(iw, p, parent)) {
      if (compose_value_cooperative(w, iw, p, left, parent, temporary, gid,
                                    int(left), lane, group_size.x, &solve_ok))
        copy_value_cooperative(w, iw, p, temporary, left, lane,
                               group_size.x);
    }
    return;
  }
  if (!invalid_value(iw, p, parent)) {
    if (!compose_value_cooperative(w, iw, p, right, parent, temporary, gid,
                                   int(right), lane, group_size.x, &solve_ok))
      return;
    copy_value_cooperative(w, iw, p, temporary, right, lane, group_size.x);
  }
  if (compose_value_cooperative(w, iw, p, left, right, temporary, gid,
                                int(left), lane, group_size.x, &solve_ok))
    copy_value_cooperative(w, iw, p, temporary, left, lane, group_size.x);
}

kernel void clqr_finalize_value_suffix_threadgroup(
    device const int *dims [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], device float *w [[buffer(3)]],
    device int *iw [[buffer(4)]], constant KernelParams &p [[buffer(5)]],
    threadgroup float *local [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  (void)dims; (void)input; (void)output;
  uint gid = group.x;
  threadgroup int solve_ok;
  if (gid >= p.parent_count) return;
  if (lane == 0) {
    if (p.parent_count == 1 && p.child_count <= 2)
      set_invalid_value(iw, p, p.parent_offset);
    solve_ok = enabled(iw, p) ? 1 : 0;
  }
  cooperative_barrier();
  if (solve_ok == 0) return;
  uint left = 2 * gid;
  uint right = left + 1;
  uint parent = p.parent_offset + gid;
  uint temporary = p.temporary_offset + gid;
  if (right >= p.child_count) {
    if (!invalid_value(iw, p, parent)) {
      if (compose_value_cooperative_threadgroup(
              w, iw, p, left, parent, temporary, int(left), lane,
              group_size.x, &solve_ok, local))
        copy_value_cooperative(w, iw, p, temporary, left, lane,
                               group_size.x);
    }
    return;
  }
  if (!invalid_value(iw, p, parent)) {
    if (!compose_value_cooperative_threadgroup(
            w, iw, p, right, parent, temporary, int(right), lane,
            group_size.x, &solve_ok, local))
      return;
    copy_value_cooperative(w, iw, p, temporary, right, lane, group_size.x);
  }
  if (compose_value_cooperative_threadgroup(
          w, iw, p, left, right, temporary, int(left), lane, group_size.x,
          &solve_ok, local))
    copy_value_cooperative(w, iw, p, temporary, left, lane, group_size.x);
}

inline device float *affine_linear(device float *w, constant KernelParams &p,
                                   uint slot) {
  return w + p.affine_data + slot * p.affine_stride;
}
inline device float *affine_offset(device float *w, constant KernelParams &p,
                                   uint slot) {
  return affine_linear(w, p, slot) +
         p.state_capacity * p.state_capacity;
}
inline device int *affine_meta(device int *iw, constant KernelParams &p,
                               uint slot) {
  return iw + p.affine_meta + 2 * slot;
}
inline bool invalid_affine(device int *iw, constant KernelParams &p,
                           uint slot) {
  return affine_meta(iw, p, slot)[0] < 0;
}
inline void set_invalid_affine(device int *iw, constant KernelParams &p,
                               uint slot) {
  affine_meta(iw, p, slot)[0] = -1;
  affine_meta(iw, p, slot)[1] = 0;
}
inline void copy_affine_cooperative(
    device float *w, device int *iw, constant KernelParams &p, uint source,
    uint target, uint lane, uint lane_count) {
  device int *sm = affine_meta(iw, p, source);
  device int *tm = affine_meta(iw, p, target);
  int left = sm[0], right = sm[1];
  if (lane == 0) {
    tm[0] = left;
    tm[1] = right;
  }
  cooperative_barrier();
  if (left >= 0) {
    device float *sl = affine_linear(w, p, source);
    device float *so = affine_offset(w, p, source);
    device float *tl = affine_linear(w, p, target);
    device float *to = affine_offset(w, p, target);
    for (uint i = lane; i < uint(right * left); i += lane_count)
      tl[i] = sl[i];
    for (uint i = lane; i < uint(right); i += lane_count)
      to[i] = so[i];
  }
  cooperative_barrier();
}

inline bool compose_affine_cooperative(
    device float *w, device int *iw, constant KernelParams &p, uint first,
    uint second, uint target, int stage, uint lane, uint lane_count,
    threadgroup int *compose_ok) {
  bool first_invalid = invalid_affine(iw, p, first);
  bool second_invalid = invalid_affine(iw, p, second);
  if (first_invalid) {
    if (second_invalid) {
      if (lane == 0) set_invalid_affine(iw, p, target);
      cooperative_barrier();
    } else {
      copy_affine_cooperative(w, iw, p, second, target, lane, lane_count);
    }
    return true;
  }
  if (second_invalid) {
    copy_affine_cooperative(w, iw, p, first, target, lane, lane_count);
    return true;
  }
  device int *fm = affine_meta(iw, p, first);
  device int *sm = affine_meta(iw, p, second);
  int left = fm[0], shared = fm[1], right = sm[1];
  if (lane == 0) {
    *compose_ok = shared == sm[0] ? 1 : 0;
    if (*compose_ok == 0)
      fail(iw, p, kNumericalFailure, stage, 11);
    else {
      device int *tm = affine_meta(iw, p, target);
      tm[0] = left;
      tm[1] = right;
    }
  }
  cooperative_barrier();
  if (*compose_ok == 0) return false;
  device float *fl = affine_linear(w, p, first);
  device float *fo = affine_offset(w, p, first);
  device float *sl = affine_linear(w, p, second);
  device float *so = affine_offset(w, p, second);
  device float *tl = affine_linear(w, p, target);
  device float *to = affine_offset(w, p, target);
  for (uint index = lane; index < uint(right * left);
       index += lane_count) {
    int row = int(index) / left;
    int col = int(index) - row * left;
    float value = 0.0f;
    for (int k = 0; k < shared; ++k)
      value = fma(sl[row * shared + k], fl[k * left + col], value);
    tl[index] = value;
  }
  for (uint row = lane; row < uint(right); row += lane_count) {
    float value = so[row];
    for (int k = 0; k < shared; ++k)
      value = fma(sl[row * shared + k], fo[k], value);
    to[row] = value;
  }
  cooperative_barrier();
  return true;
}

kernel void clqr_reduce_affine(
    device const int *dims [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], device float *w [[buffer(3)]],
    device int *iw [[buffer(4)]], constant KernelParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  (void)dims; (void)input; (void)output;
  uint gid = group.x;
  threadgroup int compose_ok;
  if (gid >= p.parent_count) return;
  if (lane == 0) compose_ok = enabled(iw, p) ? 1 : 0;
  cooperative_barrier();
  if (compose_ok == 0) return;
  uint left = p.child_offset + 2 * gid;
  uint parent = p.parent_offset + gid;
  if (2 * gid + 1 >= p.child_count)
    copy_affine_cooperative(w, iw, p, left, parent, lane, group_size.x);
  else
    compose_affine_cooperative(w, iw, p, left, left + 1, parent, int(parent),
                               lane, group_size.x, &compose_ok);
}

kernel void clqr_expand_affine_context(
    device const int *dims [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], device float *w [[buffer(3)]],
    device int *iw [[buffer(4)]], constant KernelParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  (void)dims; (void)input; (void)output;
  uint gid = group.x;
  threadgroup int compose_ok;
  if (gid >= p.parent_count) return;
  if (lane == 0) {
    if (p.parent_count == 1 && p.child_count <= 2)
      set_invalid_affine(iw, p, p.parent_offset);
    compose_ok = enabled(iw, p) ? 1 : 0;
  }
  cooperative_barrier();
  if (compose_ok == 0) return;
  uint left = p.child_offset + 2 * gid;
  uint parent = p.parent_offset + gid;
  if (2 * gid + 1 >= p.child_count) {
    copy_affine_cooperative(w, iw, p, parent, left, lane, group_size.x);
    return;
  }
  // Prefix context: right receives left composed after the parent context.
  if (compose_affine_cooperative(w, iw, p, parent, left, left + 1, int(left),
                                 lane, group_size.x, &compose_ok))
    copy_affine_cooperative(w, iw, p, parent, left, lane, group_size.x);
}

kernel void clqr_finalize_affine_prefix(
    device const int *dims [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], device float *w [[buffer(3)]],
    device int *iw [[buffer(4)]], constant KernelParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  (void)dims; (void)input; (void)output;
  uint gid = group.x;
  threadgroup int compose_ok;
  if (gid >= p.parent_count) return;
  if (lane == 0) {
    if (p.parent_count == 1 && p.child_count <= 2)
      set_invalid_affine(iw, p, p.parent_offset);
    compose_ok = enabled(iw, p) ? 1 : 0;
  }
  cooperative_barrier();
  if (compose_ok == 0) return;
  uint left = 2 * gid, right = left + 1;
  uint parent = p.parent_offset + gid;
  uint temporary = p.temporary_offset + gid;
  if (right >= p.child_count) {
    if (!invalid_affine(iw, p, parent)) {
      if (compose_affine_cooperative(w, iw, p, parent, left, temporary,
                                     int(left), lane, group_size.x,
                                     &compose_ok))
        copy_affine_cooperative(w, iw, p, temporary, left, lane,
                                group_size.x);
    }
    return;
  }
  if (!invalid_affine(iw, p, parent)) {
    if (!compose_affine_cooperative(w, iw, p, parent, left, temporary,
                                    int(left), lane, group_size.x,
                                    &compose_ok))
      return;
    copy_affine_cooperative(w, iw, p, temporary, left, lane, group_size.x);
  }
  if (compose_affine_cooperative(w, iw, p, left, right, temporary, int(right),
                                 lane, group_size.x, &compose_ok))
    copy_affine_cooperative(w, iw, p, temporary, right, lane, group_size.x);
}
)CLQR_METAL";

} // namespace clqr::metal::detail

#endif // CLQR_SRC_METAL_VALUE_AFFINE_SOURCE_H_
