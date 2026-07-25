#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "src/metal_planner.h"

namespace {

void Expect(bool condition) {
  if (!condition)
    throw std::runtime_error("Metal planner test expectation failed");
}

template <typename Function> void ExpectLengthError(Function &&function) {
  bool rejected = false;
  try {
    function();
  } catch (const std::length_error &) {
    rejected = true;
  }
  Expect(rejected);
}

} // namespace

int main() {
  using clqr::metal::detail::ArenaSizer;
  using clqr::metal::detail::CheckedSizeAlignUp;
  using clqr::metal::detail::CheckedSizeProduct;
  using clqr::metal::detail::CheckedSizeSum;
  using clqr::metal::detail::CheckedU32;
  using clqr::metal::detail::HasCooperativeThreadgroupOccupancy;
  using clqr::metal::detail::PlanInvocation;
  using clqr::metal::detail::PlanLaneSlicedThreadgroupLanes;

  const auto normal = PlanInvocation(1024, 8, 4, 2, 2, 2, 1.0e-5f);
  Expect(normal.params.stage_count == 1024);
  Expect(normal.params.state_capacity == 8);
  Expect(normal.params.control_capacity == 4);
  Expect(normal.node_tree.counts.front() == 1025);
  Expect(normal.stage_tree.counts.front() == 1024);
  Expect(normal.input_floats > 0);
  Expect(normal.output_floats > 0);
  Expect(normal.workspace_floats > 0);
  Expect(normal.workspace_ints > 0);
  Expect(normal.reduced_stage_float_bytes > 0);
  Expect(normal.reduced_stage_integer_bytes > 0);
  Expect(normal.reduced_terminal_float_bytes > 0);
  Expect(normal.reduced_terminal_integer_bytes == 16);
  Expect(normal.feedback_float_bytes > 0);
  Expect(normal.feedback_integer_bytes == 16);
  Expect(normal.primal_leaf_float_bytes == 1008);
  Expect(normal.primal_leaf_integer_bytes == 80);
  Expect(normal.value_composition_float_bytes == 800);
  Expect(normal.dual_parameter_float_bytes == 1072);
  Expect(normal.dual_parameter_integer_bytes == 64);
  Expect(normal.dual_leaf_float_bytes == 816);
  Expect(normal.dual_leaf_integer_bytes == 144);
  Expect(normal.dual_relation_float_bytes == 4960);
  Expect(normal.dual_relation_integer_bytes == 64);
  Expect(normal.dual_solve_float_bytes == 2400);
  Expect(normal.dual_solve_integer_bytes == 64);
  for (const std::size_t bytes :
       {normal.reduced_stage_float_bytes, normal.reduced_stage_integer_bytes,
        normal.reduced_terminal_float_bytes,
        normal.reduced_terminal_integer_bytes, normal.feedback_float_bytes,
        normal.feedback_integer_bytes, normal.primal_leaf_float_bytes,
        normal.primal_leaf_integer_bytes, normal.value_composition_float_bytes,
        normal.dual_parameter_float_bytes, normal.dual_parameter_integer_bytes,
        normal.dual_leaf_float_bytes, normal.dual_leaf_integer_bytes,
        normal.dual_relation_float_bytes, normal.dual_relation_integer_bytes,
        normal.dual_solve_float_bytes, normal.dual_solve_integer_bytes}) {
    Expect(bytes % 16 == 0);
  }

  constexpr std::uint32_t maximum = std::numeric_limits<std::uint32_t>::max();
  constexpr std::size_t size_max = std::numeric_limits<std::size_t>::max();
  Expect(CheckedSizeAlignUp(0, 16, "test alignment") == 0);
  Expect(CheckedSizeAlignUp(1, 16, "test alignment") == 16);
  Expect(CheckedSizeAlignUp(15, 16, "test alignment") == 16);
  Expect(CheckedSizeAlignUp(16, 16, "test alignment") == 16);
  Expect(CheckedSizeAlignUp(17, 16, "test alignment") == 32);
  ExpectLengthError(
      [=] { CheckedSizeAlignUp(size_max, 16, "test alignment"); });
  Expect(PlanLaneSlicedThreadgroupLanes(0, 32768, 1024, 32, 32) == 32);
  Expect(PlanLaneSlicedThreadgroupLanes(0, 16384, 1024, 32, 32) == 16);
  Expect(PlanLaneSlicedThreadgroupLanes(0, 15360, 1024, 32, 32) == 0);
  Expect(PlanLaneSlicedThreadgroupLanes(1, 0, 1024, 32, 32) == 0);
  Expect(PlanLaneSlicedThreadgroupLanes(0, 32768, 0, 32, 32) == 0);
  Expect(HasCooperativeThreadgroupOccupancy(0, 8192, 32768, 4));
  Expect(HasCooperativeThreadgroupOccupancy(1, 8191, 32768, 4));
  Expect(!HasCooperativeThreadgroupOccupancy(0, 8193, 32768, 4));
  Expect(!HasCooperativeThreadgroupOccupancy(8193, 0, 32768, 4));
  Expect(!HasCooperativeThreadgroupOccupancy(0, 0, 32768, 0));
  ExpectLengthError([=] { CheckedSizeSum({size_max, 1}, "test sum"); });
  ExpectLengthError([=] { CheckedSizeProduct({size_max, 2}, "test product"); });
  if constexpr (size_max > maximum) {
    ExpectLengthError(
        [=] { CheckedU32(std::size_t(maximum) + 1, "test offset"); });
  }
  ExpectLengthError([=] { PlanInvocation(maximum, 1, 1, 0, 0, 0, 1.0e-5f); });
  ExpectLengthError(
      [=] { PlanInvocation(maximum - 1, 1, 1, 0, 0, 0, 1.0e-5f); });
  ExpectLengthError([=] { PlanInvocation(1, maximum, 1, 0, 0, 0, 1.0e-5f); });
  ExpectLengthError([=] {
    PlanInvocation(1, maximum, maximum, maximum, maximum, maximum, 1.0e-5f);
  });

  ArenaSizer arena;
  arena.Add(maximum);
  ExpectLengthError([&arena] { arena.Add(1); });
  return 0;
}
