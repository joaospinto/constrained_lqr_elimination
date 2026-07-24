#ifndef CLQR_PYTHON_JAX_FFI_PROBLEM_H_
#define CLQR_PYTHON_JAX_FFI_PROBLEM_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "clqr/clqr.h"

namespace clqr::python {

struct PackedProblemBuffers {
  std::size_t stage_count = 0;
  std::size_t state_capacity = 0;
  std::size_t control_capacity = 0;
  std::size_t mixed_capacity = 0;
  std::size_t state_constraint_capacity = 0;
  std::size_t terminal_constraint_capacity = 0;

  const std::int32_t *dimensions = nullptr;
  std::size_t dimension_count = 0;

  const Scalar *A = nullptr;
  const Scalar *B = nullptr;
  const Scalar *c = nullptr;
  const Scalar *Q = nullptr;
  const Scalar *R = nullptr;
  const Scalar *M = nullptr;
  const Scalar *q = nullptr;
  const Scalar *r = nullptr;
  const Scalar *C = nullptr;
  const Scalar *D = nullptr;
  const Scalar *d = nullptr;
  const Scalar *E = nullptr;
  const Scalar *e = nullptr;
  const Scalar *terminal_Q = nullptr;
  const Scalar *terminal_q = nullptr;
  const Scalar *terminal_E = nullptr;
  const Scalar *terminal_e = nullptr;
  const Scalar *initial_state = nullptr;
};

struct PackedSolutionBuffers {
  std::int32_t *diagnostics = nullptr;
  Scalar *objective = nullptr;
  Scalar *states = nullptr;
  Scalar *controls = nullptr;
  Scalar *initial_multiplier = nullptr;
  Scalar *dynamics_multipliers = nullptr;
  Scalar *mixed_multipliers = nullptr;
  Scalar *state_multipliers = nullptr;
  Scalar *terminal_state_multiplier = nullptr;
};

bool BuildProblem(const PackedProblemBuffers &packed, Problem *problem,
                  std::string *error);
void WriteSolution(const PackedProblemBuffers &packed,
                   const SolutionView &solution,
                   const PackedSolutionBuffers &output);

} // namespace clqr::python

#endif // CLQR_PYTHON_JAX_FFI_PROBLEM_H_
