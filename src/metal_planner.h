#ifndef CLQR_SRC_METAL_PLANNER_H_
#define CLQR_SRC_METAL_PLANNER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/metal_layout.h"

namespace clqr::metal::detail {

inline std::size_t CheckedSizeSum(std::initializer_list<std::size_t> terms,
                                  const char *description) {
  std::size_t result = 0;
  for (const std::size_t term : terms) {
    if (term > std::numeric_limits<std::size_t>::max() - result)
      throw std::length_error(std::string(description) + " overflows size_t");
    result += term;
  }
  return result;
}

inline std::size_t
CheckedSizeProduct(std::initializer_list<std::size_t> factors,
                   const char *description) {
  std::size_t result = 1;
  for (const std::size_t factor : factors) {
    if (factor != 0 &&
        result > std::numeric_limits<std::size_t>::max() / factor) {
      throw std::length_error(std::string(description) + " overflows size_t");
    }
    result *= factor;
  }
  return result;
}

inline std::size_t CheckedSizeAlignUp(std::size_t value, std::size_t alignment,
                                      const char *description) {
  if (alignment == 0)
    throw std::invalid_argument("Metal alignment must be nonzero");
  const std::size_t remainder = value % alignment;
  if (remainder == 0)
    return value;
  return CheckedSizeSum({value, alignment - remainder}, description);
}

inline std::uint32_t CheckedU32(std::size_t value, const char *description) {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::length_error(std::string(description) +
                            " exceeds 32-bit Metal offsets");
  }
  return static_cast<std::uint32_t>(value);
}

class ArenaSizer {
public:
  std::uint32_t Add(std::size_t count) {
    constexpr std::size_t kAlignmentElements = 4;
    const std::size_t aligned = CheckedSizeSum({size_, kAlignmentElements - 1},
                                               "Metal arena alignment") &
                                ~(kAlignmentElements - 1);
    const std::size_t next =
        CheckedSizeSum({aligned, count}, "Metal arena size");
    if (next > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error("Metal arena exceeds 32-bit element offsets");
    }
    size_ = next;
    return static_cast<std::uint32_t>(aligned);
  }

  std::size_t size() const { return size_; }

private:
  std::size_t size_ = 0;
};

struct TreePlan {
  std::vector<std::uint32_t> offsets;
  std::vector<std::uint32_t> counts;
  std::uint32_t slots = 0;
};

inline TreePlan BuildTreePlan(std::uint32_t leaf_count) {
  TreePlan plan;
  std::uint32_t count = leaf_count;
  while (count > 0) {
    plan.offsets.push_back(plan.slots);
    plan.counts.push_back(count);
    plan.slots =
        CheckedU32(CheckedSizeSum({plan.slots, count}, "Metal tree slot count"),
                   "Metal tree slot count");
    if (count == 1)
      break;
    count = count / 2 + count % 2;
  }
  return plan;
}

inline std::size_t PlanLaneSlicedThreadgroupLanes(
    std::size_t static_bytes, std::size_t maximum_bytes,
    std::size_t per_lane_bytes, std::size_t maximum_threads,
    std::size_t thread_execution_width) {
  if (per_lane_bytes == 0 || static_bytes > maximum_bytes)
    return 0;
  const std::size_t memory_lanes =
      (maximum_bytes - static_bytes) / per_lane_bytes;
  const std::size_t lanes =
      std::min({std::size_t(32), maximum_threads, memory_lanes});
  const std::size_t minimum_lanes =
      std::max(std::size_t(1), thread_execution_width / 2);
  return lanes >= minimum_lanes ? lanes : 0;
}

inline bool HasCooperativeThreadgroupOccupancy(
    std::size_t static_bytes, std::size_t dynamic_bytes,
    std::size_t maximum_bytes, std::size_t minimum_resident_groups) {
  if (minimum_resident_groups == 0)
    return false;
  const std::size_t per_group_limit = maximum_bytes / minimum_resident_groups;
  return static_bytes <= per_group_limit &&
         dynamic_bytes <= per_group_limit - static_bytes;
}

struct InvocationLayout {
  KernelParams params{};
  std::size_t input_floats = 0;
  std::size_t output_floats = 0;
  std::size_t workspace_floats = 0;
  std::size_t workspace_ints = 0;
  std::size_t reduced_stage_float_bytes = 0;
  std::size_t reduced_stage_integer_bytes = 0;
  std::size_t reduced_terminal_float_bytes = 0;
  std::size_t reduced_terminal_integer_bytes = 0;
  std::size_t feedback_float_bytes = 0;
  std::size_t feedback_integer_bytes = 0;
  std::size_t primal_leaf_float_bytes = 0;
  std::size_t primal_leaf_integer_bytes = 0;
  std::size_t value_composition_float_bytes = 0;
  std::size_t dual_parameter_float_bytes = 0;
  std::size_t dual_parameter_integer_bytes = 0;
  std::size_t dual_leaf_float_bytes = 0;
  std::size_t dual_leaf_integer_bytes = 0;
  std::size_t dual_relation_float_bytes = 0;
  std::size_t dual_relation_integer_bytes = 0;
  std::size_t dual_solve_float_bytes = 0;
  std::size_t dual_solve_integer_bytes = 0;
  TreePlan node_tree;
  TreePlan stage_tree;
};

inline InvocationLayout
PlanInvocation(std::uint32_t stage_count, std::uint32_t state_capacity,
               std::uint32_t control_capacity, std::uint32_t mixed_capacity,
               std::uint32_t state_constraint_capacity,
               std::uint32_t terminal_constraint_capacity, float tolerance) {
  InvocationLayout layout;
  KernelParams &p = layout.params;
  p.stage_count = stage_count;
  p.state_capacity = state_capacity;
  p.control_capacity = control_capacity;
  p.mixed_capacity = mixed_capacity;
  p.state_constraint_capacity = state_constraint_capacity;
  p.terminal_constraint_capacity = terminal_constraint_capacity;
  p.rank_tolerance = tolerance;
  p.consistency_tolerance = 20.0f * tolerance;

  const std::size_t N = stage_count;
  const std::size_t nx = state_capacity;
  const std::size_t nu = control_capacity;
  const std::size_t nc = mixed_capacity;
  const std::size_t ne = state_constraint_capacity;
  const std::size_t nt = terminal_constraint_capacity;
  const std::size_t N1 =
      CheckedSizeSum({N, 1}, "Metal stage count plus terminal");
  layout.node_tree =
      BuildTreePlan(CheckedU32(N1, "Metal stage count plus terminal"));
  layout.stage_tree = BuildTreePlan(stage_count);

  const auto sum = [](std::initializer_list<std::size_t> terms) {
    return CheckedSizeSum(terms, "Metal invocation layout");
  };
  const auto product = [](std::initializer_list<std::size_t> factors) {
    return CheckedSizeProduct(factors, "Metal invocation layout");
  };

  ArenaSizer input;
  p.input_A = input.Add(product({N, nx, nx}));
  p.input_B = input.Add(product({N, nx, nu}));
  p.input_c = input.Add(product({N, nx}));
  p.input_Q = input.Add(product({N1, nx, nx}));
  p.input_R = input.Add(product({N, nu, nu}));
  p.input_M = input.Add(product({N, nx, nu}));
  p.input_q = input.Add(product({N1, nx}));
  p.input_r = input.Add(product({N, nu}));
  p.input_C = input.Add(product({N, nc, nx}));
  p.input_D = input.Add(product({N, nc, nu}));
  p.input_d = input.Add(product({N, nc}));
  p.input_E = input.Add(product({N, ne, nx}));
  p.input_e = input.Add(product({N, ne}));
  p.input_terminal_E = input.Add(product({nt, nx}));
  p.input_terminal_e = input.Add(nt);
  p.input_initial_state = input.Add(nx);
  layout.input_floats = input.size();

  ArenaSizer output;
  p.output_objective = output.Add(1);
  p.output_states = output.Add(product({N1, nx}));
  p.output_controls = output.Add(product({N, nu}));
  p.output_initial_multiplier = output.Add(nx);
  p.output_dynamics_multipliers = output.Add(product({N, nx}));
  p.output_mixed_multipliers = output.Add(product({N, nc}));
  p.output_state_multipliers = output.Add(product({N, ne}));
  p.output_terminal_state_multiplier = output.Add(nt);
  layout.output_floats = output.size();

  const std::size_t relation_rows = product({2, nx});
  const std::size_t relation_slots = sum({layout.node_tree.slots, N / 2 + 1});
  const std::size_t stage_slots = sum({layout.stage_tree.slots, N / 2 + N % 2});
  ArenaSizer workspace;
  p.relation_stride = CheckedU32(
      sum({product({2, relation_rows, nx}), relation_rows}), "relation stride");
  p.relation_data = workspace.Add(product({relation_slots, p.relation_stride}));
  p.state_param_stride =
      CheckedU32(sum({product({nx, nx}), nx}), "state parameter stride");
  p.state_param_data = workspace.Add(product({N1, p.state_param_stride}));
  p.control_param_stride =
      CheckedU32(sum({product({nu, nx}), product({nu, nu}), nu}),
                 "control parameter stride");
  p.control_param_data = workspace.Add(product({N, p.control_param_stride}));
  p.reduced_stage_stride =
      CheckedU32(sum({product({2, nx, nx}), product({2, nx, nu}),
                      product({nu, nu}), product({2, nx}), nu}),
                 "reduced stage stride");
  p.reduced_stage_data = workspace.Add(product({N, p.reduced_stage_stride}));
  p.reduced_terminal_data = workspace.Add(sum({product({nx, nx}), nx}));
  p.value_stride = CheckedU32(product({3, nx, nx}), "value stride");
  p.value_data = workspace.Add(product({relation_slots, p.value_stride}));
  p.feedback_stride = CheckedU32(
      sum({product({nu, nx}), nu, product({nu, nu}), product({nx, nx}), nx}),
      "feedback stride");
  p.feedback_data = workspace.Add(product({N, p.feedback_stride}));
  p.affine_stride = CheckedU32(sum({product({nx, nx}), nx}), "affine stride");
  p.affine_data = workspace.Add(product({stage_slots, p.affine_stride}));
  p.reduced_initial = workspace.Add(nx);
  p.reduced_costates = workspace.Add(product({N1, nx}));
  p.reduced_states = workspace.Add(product({N1, nx}));
  p.reduced_controls = workspace.Add(product({N, nu}));

  const std::size_t dual_capacity = sum({nx, nc});
  p.dual_param_stride =
      CheckedU32(sum({product({dual_capacity, dual_capacity}), dual_capacity}),
                 "dual parameter stride");
  p.dual_param_data = workspace.Add(product({N, p.dual_param_stride}));
  const std::size_t state_dual_capacity = std::max(ne, nt);
  p.state_dual_param_stride = CheckedU32(
      product({state_dual_capacity, sum({1, product({2, dual_capacity})})}),
      "state dual parameter stride");
  p.state_dual_param_data =
      workspace.Add(product({N, p.state_dual_param_stride}));
  p.dual_relation_stride =
      CheckedU32(sum({product({4, dual_capacity, dual_capacity}),
                      product({2, dual_capacity})}),
                 "dual relation stride");
  p.dual_relation_data =
      workspace.Add(product({stage_slots, p.dual_relation_stride}));
  p.dual_node_stride =
      CheckedU32(product({2, dual_capacity}), "dual node stride");
  p.dual_node_data = workspace.Add(product({stage_slots, p.dual_node_stride}));
  p.objective_tree = workspace.Add(layout.node_tree.slots);

  const std::size_t two_nx = product({2, nx});
  const std::size_t nx_plus_nu = sum({nx, nu});
  const std::size_t two_dual = product({2, dual_capacity});
  const std::size_t primal_leaf_scratch =
      product({std::max(sum({nc, ne, nx}), nt), sum({nu, two_nx, 1})});
  const std::size_t primal_relation_scratch =
      product({product({4, nx}), sum({product({3, nx}), 1})});
  const std::size_t stage_reduction_rref_scratch = sum(
      {product({sum({nc, two_nx}), sum({nu, nx, 2})}), product({nx, nx}), nx});
  const std::size_t stage_reduction_block_scratch =
      sum({product({nx_plus_nu, nx_plus_nu}), nx, nu});
  const std::size_t stage_reduction_scratch =
      std::max(stage_reduction_rref_scratch, stage_reduction_block_scratch);
  const std::size_t terminal_reduction_scratch = sum({product({nx, nx}), nx});
  const std::size_t feedback_scratch = product({nx_plus_nu, nx_plus_nu});
  const auto threadgroup_bytes = [](std::size_t bytes) {
    constexpr std::size_t kMetalThreadgroupMemoryAlignment = 16;
    return CheckedSizeAlignUp(bytes, kMetalThreadgroupMemoryAlignment,
                              "Metal threadgroup memory");
  };
  layout.reduced_stage_float_bytes =
      threadgroup_bytes(product({sizeof(float), stage_reduction_scratch}));
  layout.reduced_stage_integer_bytes =
      threadgroup_bytes(product({sizeof(std::int32_t), sum({nc, two_nx, 4})}));
  layout.reduced_terminal_float_bytes =
      threadgroup_bytes(product({sizeof(float), terminal_reduction_scratch}));
  layout.reduced_terminal_integer_bytes =
      threadgroup_bytes(sizeof(std::int32_t));
  layout.feedback_float_bytes =
      threadgroup_bytes(product({sizeof(float), feedback_scratch}));
  layout.feedback_integer_bytes =
      threadgroup_bytes(product({sizeof(std::int32_t), 2}));
  layout.primal_leaf_float_bytes =
      threadgroup_bytes(product({sizeof(float), primal_leaf_scratch}));
  layout.primal_leaf_integer_bytes =
      threadgroup_bytes(product({sizeof(std::int32_t), sum({nu, two_nx})}));
  const std::size_t value_composition_scratch =
      sum({product({nx, product({3, nx})}), nx});
  layout.value_composition_float_bytes =
      threadgroup_bytes(product({sizeof(float), value_composition_scratch}));
  const std::size_t value_leaf_scratch =
      sum({product({nu, nu}), product({2, nu, nx})});
  const std::size_t dual_parameter_scratch =
      sum({product({nx_plus_nu, sum({dual_capacity, 1})}), nc, nx_plus_nu,
           product({dual_capacity, dual_capacity}), two_dual});
  const std::size_t dual_leaf_scratch =
      sum({product({nx, sum({state_dual_capacity, two_dual, 1})}), two_nx,
           state_dual_capacity});
  const std::size_t dual_relation_scratch = product(
      {product({4, dual_capacity}), sum({product({3, dual_capacity}), 1})});
  const std::size_t dual_solve_scratch =
      sum({product({5, dual_capacity, dual_capacity}),
           product({10, dual_capacity})});
  layout.dual_parameter_float_bytes =
      threadgroup_bytes(product({sizeof(float), dual_parameter_scratch}));
  layout.dual_parameter_integer_bytes = threadgroup_bytes(
      product({sizeof(std::int32_t), sum({dual_capacity, 4})}));
  layout.dual_leaf_float_bytes =
      threadgroup_bytes(product({sizeof(float), dual_leaf_scratch}));
  layout.dual_leaf_integer_bytes = threadgroup_bytes(product(
      {sizeof(std::int32_t), sum({nx, state_dual_capacity, two_dual, 4})}));
  layout.dual_relation_float_bytes =
      threadgroup_bytes(product({sizeof(float), dual_relation_scratch}));
  layout.dual_relation_integer_bytes = threadgroup_bytes(
      product({sizeof(std::int32_t), sum({dual_capacity, 4})}));
  layout.dual_solve_float_bytes =
      threadgroup_bytes(product({sizeof(float), dual_solve_scratch}));
  layout.dual_solve_integer_bytes = threadgroup_bytes(
      product({sizeof(std::int32_t), sum({dual_capacity, 4})}));
  const std::size_t float_scratch_stride = std::max(
      {primal_leaf_scratch, primal_relation_scratch, stage_reduction_scratch,
       terminal_reduction_scratch, feedback_scratch, value_composition_scratch,
       value_leaf_scratch, dual_parameter_scratch, dual_leaf_scratch,
       dual_relation_scratch, dual_solve_scratch, nx});
  p.float_scratch_stride =
      CheckedU32(float_scratch_stride, "float scratch stride");
  p.float_scratch = workspace.Add(product({N1, p.float_scratch_stride}));
  layout.workspace_floats = workspace.size();

  ArenaSizer integers;
  p.status = integers.Add(3);
  p.relation_meta = integers.Add(product({3, relation_slots}));
  p.state_param_meta = integers.Add(product({2, N1}));
  p.state_param_free_columns = integers.Add(product({N1, nx}));
  p.control_param_meta = integers.Add(product({3, N}));
  p.control_param_free_columns = integers.Add(product({N, nu}));
  p.reduced_stage_meta = integers.Add(sum({product({3, N}), 1}));
  p.value_meta = integers.Add(product({2, relation_slots}));
  p.feedback_meta = integers.Add(product({3, N}));
  p.affine_meta = integers.Add(product({2, stage_slots}));
  p.dual_param_meta = integers.Add(product({4, N}));
  p.dual_param_free_columns = integers.Add(product({N, dual_capacity}));
  p.state_dual_param_meta = integers.Add(product({3, N}));
  p.dual_relation_meta = integers.Add(product({3, stage_slots}));
  p.dual_node_meta = integers.Add(product({2, stage_slots}));
  const std::size_t int_scratch_stride =
      std::max({sum({nu, two_nx}), sum({nx, state_dual_capacity, two_dual}),
                dual_capacity});
  p.int_scratch_stride =
      CheckedU32(int_scratch_stride, "integer scratch stride");
  p.int_scratch = integers.Add(product({N1, p.int_scratch_stride}));
  layout.workspace_ints = integers.size();
  return layout;
}

} // namespace clqr::metal::detail

#endif // CLQR_SRC_METAL_PLANNER_H_
