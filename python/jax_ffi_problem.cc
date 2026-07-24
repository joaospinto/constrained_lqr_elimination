#include "python/jax_ffi_problem.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace clqr::python {
namespace {

bool Fail(std::string message, std::string *error) {
  if (error != nullptr)
    *error = std::move(message);
  return false;
}

bool ValidDimension(std::int32_t value, std::size_t capacity) {
  return value >= 0 && static_cast<std::size_t>(value) <= capacity;
}

void CopyVector(const Scalar *source, std::size_t size, Vector *output) {
  output->resize(size);
  if (size == 0)
    return;
  std::copy_n(source, size, output->data().data());
}

void CopyMatrix(const Scalar *source, std::size_t rows, std::size_t columns,
                std::size_t source_stride, Matrix *output) {
  output->resize(rows, columns);
  if (rows == 0 || columns == 0)
    return;
  for (std::size_t row = 0; row < rows; ++row) {
    std::copy_n(source + row * source_stride, columns,
                output->data().data() + row * columns);
  }
}

void CopyView(const VectorView &source, Scalar *destination) {
  if (source.size == 0)
    return;
  std::copy_n(source.data, source.size, destination);
}

template <typename T> void FillZero(T *output, std::size_t size) {
  if (size == 0)
    return;
  std::fill_n(output, size, T{0});
}

template <typename T> T *Offset(T *pointer, std::size_t offset) {
  return offset == 0 ? pointer : pointer + offset;
}

} // namespace

bool BuildProblem(const PackedProblemBuffers &packed, Problem *problem,
                  std::string *error) {
  if (problem == nullptr)
    return Fail("problem output is null", error);
  if (packed.stage_count > (std::numeric_limits<std::size_t>::max() - 2) / 4) {
    return Fail("stage count is too large", error);
  }
  const std::size_t expected_dimensions = 4 * packed.stage_count + 2;
  if (packed.dimension_count != expected_dimensions) {
    return Fail("dimension vector must contain 4 * stages + 2 entries", error);
  }
  if (packed.dimensions == nullptr) {
    return Fail("dimension vector is null", error);
  }

  const std::size_t control_offset = packed.stage_count + 1;
  const std::size_t mixed_offset = 2 * packed.stage_count + 1;
  const std::size_t state_constraint_offset = 3 * packed.stage_count + 1;
  const std::size_t terminal_constraint_offset = 4 * packed.stage_count + 1;
  for (std::size_t node = 0; node <= packed.stage_count; ++node) {
    if (!ValidDimension(packed.dimensions[node], packed.state_capacity)) {
      return Fail("state dimension exceeds the padded state capacity", error);
    }
  }
  for (std::size_t stage = 0; stage < packed.stage_count; ++stage) {
    if (!ValidDimension(packed.dimensions[control_offset + stage],
                        packed.control_capacity)) {
      return Fail("control dimension exceeds the padded control capacity",
                  error);
    }
    if (!ValidDimension(packed.dimensions[mixed_offset + stage],
                        packed.mixed_capacity)) {
      return Fail("mixed-constraint dimension exceeds its padded capacity",
                  error);
    }
    if (!ValidDimension(packed.dimensions[state_constraint_offset + stage],
                        packed.state_constraint_capacity)) {
      return Fail("state-constraint dimension exceeds its padded capacity",
                  error);
    }
  }
  if (!ValidDimension(packed.dimensions[terminal_constraint_offset],
                      packed.terminal_constraint_capacity)) {
    return Fail("terminal-constraint dimension exceeds its padded capacity",
                error);
  }

  problem->stages.resize(packed.stage_count);
  const std::size_t nx = packed.state_capacity;
  const std::size_t nu = packed.control_capacity;
  const std::size_t nc = packed.mixed_capacity;
  const std::size_t ne = packed.state_constraint_capacity;
  for (std::size_t stage_index = 0; stage_index < packed.stage_count;
       ++stage_index) {
    const std::size_t n =
        static_cast<std::size_t>(packed.dimensions[stage_index]);
    const std::size_t next_n =
        static_cast<std::size_t>(packed.dimensions[stage_index + 1]);
    const std::size_t m = static_cast<std::size_t>(
        packed.dimensions[control_offset + stage_index]);
    const std::size_t mixed =
        static_cast<std::size_t>(packed.dimensions[mixed_offset + stage_index]);
    const std::size_t state_constraints = static_cast<std::size_t>(
        packed.dimensions[state_constraint_offset + stage_index]);

    const std::size_t matrix_state_offset = stage_index * nx * nx;
    const std::size_t matrix_control_offset = stage_index * nx * nu;
    const std::size_t control_square_offset = stage_index * nu * nu;
    const std::size_t state_vector_offset = stage_index * nx;
    const std::size_t control_vector_offset = stage_index * nu;
    const std::size_t mixed_state_offset = stage_index * nc * nx;
    const std::size_t mixed_control_offset = stage_index * nc * nu;
    const std::size_t mixed_vector_offset = stage_index * nc;
    const std::size_t constraint_state_offset = stage_index * ne * nx;
    const std::size_t constraint_vector_offset = stage_index * ne;

    Stage &stage = problem->stages[stage_index];
    CopyMatrix(Offset(packed.A, matrix_state_offset), next_n, n, nx, &stage.A);
    CopyMatrix(Offset(packed.B, matrix_control_offset), next_n, m, nu,
               &stage.B);
    CopyVector(Offset(packed.c, state_vector_offset), next_n, &stage.c);
    CopyMatrix(Offset(packed.Q, matrix_state_offset), n, n, nx, &stage.Q);
    CopyMatrix(Offset(packed.R, control_square_offset), m, m, nu, &stage.R);
    CopyMatrix(Offset(packed.M, matrix_control_offset), n, m, nu, &stage.M);
    CopyVector(Offset(packed.q, state_vector_offset), n, &stage.q);
    CopyVector(Offset(packed.r, control_vector_offset), m, &stage.r);
    CopyMatrix(Offset(packed.C, mixed_state_offset), mixed, n, nx, &stage.C);
    CopyMatrix(Offset(packed.D, mixed_control_offset), mixed, m, nu, &stage.D);
    CopyVector(Offset(packed.d, mixed_vector_offset), mixed, &stage.d);
    CopyMatrix(Offset(packed.E, constraint_state_offset), state_constraints, n,
               nx, &stage.E);
    CopyVector(Offset(packed.e, constraint_vector_offset), state_constraints,
               &stage.e);
  }

  const std::size_t initial_n = static_cast<std::size_t>(packed.dimensions[0]);
  const std::size_t terminal_n =
      static_cast<std::size_t>(packed.dimensions[packed.stage_count]);
  const std::size_t terminal_constraints =
      static_cast<std::size_t>(packed.dimensions[terminal_constraint_offset]);
  CopyVector(packed.initial_state, initial_n, &problem->initial_state);
  CopyMatrix(packed.terminal_Q, terminal_n, terminal_n, nx,
             &problem->terminal_Q);
  CopyVector(packed.terminal_q, terminal_n, &problem->terminal_q);
  CopyMatrix(packed.terminal_E, terminal_constraints, terminal_n, nx,
             &problem->terminal_E);
  CopyVector(packed.terminal_e, terminal_constraints, &problem->terminal_e);
  return true;
}

void WriteSolution(const PackedProblemBuffers &packed,
                   const SolutionView &solution,
                   const PackedSolutionBuffers &output) {
  output.diagnostics[0] = static_cast<std::int32_t>(solution.status);
  output.diagnostics[1] = solution.newton_kkt_singular ? 1 : 0;
  output.diagnostics[2] = solution.newton_kkt_wrong_inertia ? 1 : 0;
  output.objective[0] = solution.objective;

  const std::size_t nx = packed.state_capacity;
  const std::size_t nu = packed.control_capacity;
  const std::size_t nc = packed.mixed_capacity;
  const std::size_t ne = packed.state_constraint_capacity;
  FillZero(output.states, (packed.stage_count + 1) * nx);
  FillZero(output.controls, packed.stage_count * nu);
  FillZero(output.initial_multiplier, nx);
  FillZero(output.dynamics_multipliers, packed.stage_count * nx);
  FillZero(output.mixed_multipliers, packed.stage_count * nc);
  FillZero(output.state_multipliers, packed.stage_count * ne);
  FillZero(output.terminal_state_multiplier,
           packed.terminal_constraint_capacity);

  for (std::size_t node = 0; node < solution.state_count; ++node) {
    CopyView(solution.states[node], Offset(output.states, node * nx));
  }
  for (std::size_t stage = 0; stage < solution.control_count; ++stage) {
    CopyView(solution.controls[stage], Offset(output.controls, stage * nu));
  }
  CopyView(solution.initial_multiplier, output.initial_multiplier);
  for (std::size_t stage = 0; stage < solution.dynamics_multiplier_count;
       ++stage) {
    CopyView(solution.dynamics_multipliers[stage],
             Offset(output.dynamics_multipliers, stage * nx));
  }
  for (std::size_t stage = 0; stage < solution.mixed_multiplier_count;
       ++stage) {
    CopyView(solution.mixed_multipliers[stage],
             Offset(output.mixed_multipliers, stage * nc));
  }
  for (std::size_t stage = 0; stage < solution.state_multiplier_count;
       ++stage) {
    CopyView(solution.state_multipliers[stage],
             Offset(output.state_multipliers, stage * ne));
  }
  CopyView(solution.terminal_state_multiplier,
           output.terminal_state_multiplier);
}

} // namespace clqr::python
