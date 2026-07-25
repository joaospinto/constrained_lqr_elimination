#include "python/jax_ffi_problem.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool Zero(const clqr::Matrix& matrix) {
  for (clqr::Scalar value : matrix.data()) {
    if (value != clqr::Scalar{0}) return false;
  }
  return true;
}

bool Zero(const clqr::Vector& vector) {
  for (clqr::Scalar value : vector.data()) {
    if (value != clqr::Scalar{0}) return false;
  }
  return true;
}

}  // namespace

int main() {
  // states=[2,1,2], controls=[1,0], mixed=[0,0],
  // state constraints=[0,0], terminal constraints=0.
  std::vector<std::int32_t> dimensions = {2, 1, 2, 1, 0, 0, 0, 0, 0, 0};
  clqr::python::PackedProblemBuffers packed;
  packed.stage_count = 2;
  packed.state_capacity = 2;
  packed.control_capacity = 1;
  packed.mixed_capacity = 0;
  packed.state_constraint_capacity = 0;
  packed.terminal_constraint_capacity = 0;
  packed.dimensions = dimensions.data();
  packed.dimension_count = dimensions.size();

  clqr::Problem problem;
  std::string error;
  if (!clqr::python::BuildProblemStructure(packed, &problem, &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  if (problem.stages.size() != 2 || problem.stages[0].A.rows() != 1 ||
      problem.stages[0].A.cols() != 2 || problem.stages[0].B.cols() != 1 ||
      problem.stages[1].A.rows() != 2 || problem.stages[1].A.cols() != 1 ||
      problem.stages[1].B.cols() != 0 || problem.initial_state.size() != 2 ||
      problem.Q.size() != 3 || problem.q.size() != 3 ||
      problem.Q.back().rows() != 2 || problem.Q.back().cols() != 2 ||
      problem.terminal_E.rows() != 0) {
    std::cerr << "shape-only problem has the wrong structure\n";
    return 1;
  }
  if (!Zero(problem.stages[0].A) || !Zero(problem.stages[0].B) ||
      !Zero(problem.stages[0].c) || !Zero(problem.Q[0]) ||
      !Zero(problem.stages[0].R) || !Zero(problem.stages[0].M) ||
      !Zero(problem.q[0]) || !Zero(problem.stages[0].r) ||
      !Zero(problem.Q.back()) || !Zero(problem.q.back()) ||
      !Zero(problem.initial_state)) {
    std::cerr << "shape-only problem unexpectedly contains nonzero data\n";
    return 1;
  }

  dimensions[0] = 3;
  if (clqr::python::BuildProblemStructure(packed, &problem, &error)) {
    std::cerr << "out-of-capacity dimension was accepted\n";
    return 1;
  }
  std::cout << "JAX FFI shape-only problem test passed\n";
  return 0;
}
