#ifndef CLQR_SRC_METAL_LAYOUT_H_
#define CLQR_SRC_METAL_LAYOUT_H_

#include <cstdint>
#include <type_traits>

namespace clqr::metal::detail {

// Scalar arenas are addressed in float elements and metadata arenas in
// int32_t elements.  Keeping every offset 32-bit makes the exact same standard
// layout usable as an MSL constant-buffer argument.  The host planner rejects
// any solve whose arena would exceed that representable range.
struct KernelParams {
  std::uint32_t stage_count;
  std::uint32_t state_capacity;
  std::uint32_t control_capacity;
  std::uint32_t mixed_capacity;
  std::uint32_t state_constraint_capacity;
  std::uint32_t terminal_constraint_capacity;
  float rank_tolerance;
  float consistency_tolerance;

  // Padded scalar input arena.
  std::uint32_t input_A;
  std::uint32_t input_B;
  std::uint32_t input_c;
  std::uint32_t input_Q;
  std::uint32_t input_R;
  std::uint32_t input_M;
  std::uint32_t input_q;
  std::uint32_t input_r;
  std::uint32_t input_C;
  std::uint32_t input_D;
  std::uint32_t input_d;
  std::uint32_t input_E;
  std::uint32_t input_e;
  std::uint32_t input_terminal_E;
  std::uint32_t input_terminal_e;
  std::uint32_t input_initial_state;

  // Packed scalar output arena.
  std::uint32_t output_objective;
  std::uint32_t output_states;
  std::uint32_t output_controls;
  std::uint32_t output_initial_multiplier;
  std::uint32_t output_dynamics_multipliers;
  std::uint32_t output_mixed_multipliers;
  std::uint32_t output_state_multipliers;
  std::uint32_t output_terminal_state_multiplier;

  // Float workspace arenas and fixed strides.
  std::uint32_t relation_data;
  std::uint32_t relation_stride;
  std::uint32_t state_param_data;
  std::uint32_t state_param_stride;
  std::uint32_t control_param_data;
  std::uint32_t control_param_stride;
  std::uint32_t reduced_stage_data;
  std::uint32_t reduced_stage_stride;
  std::uint32_t reduced_terminal_data;
  std::uint32_t value_data;
  std::uint32_t value_stride;
  std::uint32_t feedback_data;
  std::uint32_t feedback_stride;
  std::uint32_t affine_data;
  std::uint32_t affine_stride;
  std::uint32_t reduced_initial;
  std::uint32_t reduced_costates;
  std::uint32_t reduced_states;
  std::uint32_t reduced_controls;
  std::uint32_t dual_param_data;
  std::uint32_t dual_param_stride;
  std::uint32_t state_dual_param_data;
  std::uint32_t state_dual_param_stride;
  std::uint32_t dual_relation_data;
  std::uint32_t dual_relation_stride;
  std::uint32_t dual_node_data;
  std::uint32_t dual_node_stride;
  std::uint32_t objective_tree;
  std::uint32_t float_scratch;
  std::uint32_t float_scratch_stride;

  // int32 metadata workspace.
  std::uint32_t status;
  std::uint32_t relation_meta;
  std::uint32_t state_param_meta;
  std::uint32_t state_param_free_columns;
  std::uint32_t control_param_meta;
  std::uint32_t control_param_free_columns;
  std::uint32_t reduced_stage_meta;
  std::uint32_t value_meta;
  std::uint32_t feedback_meta;
  std::uint32_t affine_meta;
  std::uint32_t dual_param_meta;
  std::uint32_t dual_param_free_columns;
  std::uint32_t state_dual_param_meta;
  std::uint32_t dual_relation_meta;
  std::uint32_t dual_node_meta;
  std::uint32_t int_scratch;
  std::uint32_t int_scratch_stride;

  // Reused tree invocation descriptor. Each command encoder receives its own
  // copied instance with these six fields changed.
  std::uint32_t child_offset;
  std::uint32_t parent_offset;
  std::uint32_t child_count;
  std::uint32_t parent_count;
  std::uint32_t temporary_offset;
  std::uint32_t phase_detail;
  std::uint32_t padding0;
  std::uint32_t padding1;
  std::uint32_t padding2;
};

static_assert(std::is_standard_layout_v<KernelParams>);
static_assert(sizeof(KernelParams) % 16 == 0,
              "Metal constant arguments must remain 16-byte aligned");

} // namespace clqr::metal::detail

#endif // CLQR_SRC_METAL_LAYOUT_H_
