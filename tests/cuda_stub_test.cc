#include <iostream>

#include "clqr/cuda.h"

int main() {
  if (clqr::cuda::Available()) {
    std::cerr << "CPU-only CUDA stub unexpectedly reports a device\n";
    return 1;
  }
  clqr::Problem problem;
  clqr::cuda::Workspace workspace;
  const clqr::cuda::SolutionView view =
      clqr::cuda::SolveView(problem, workspace);
  if (view.status != clqr::SolveStatus::kInvalidInput) {
    std::cerr << "CUDA stub view returned an unexpected status\n";
    return 1;
  }
  clqr::cuda::Solution materialized;
  materialized.states.push_back(clqr::Vector{1.0});
  materialized.controls.push_back(clqr::Vector{2.0});
  materialized.initial_multiplier = clqr::Vector{3.0};
  materialized.dynamics_multipliers.push_back(clqr::Vector{4.0});
  materialized.mixed_multipliers.push_back(clqr::Vector{5.0});
  materialized.state_multipliers.push_back(clqr::Vector{6.0});
  materialized.terminal_state_multiplier = clqr::Vector{7.0};
  materialized.reduced_state_dimensions.push_back(1);
  materialized.reduced_control_dimensions.push_back(1);
  clqr::cuda::Materialize(view, materialized);
  if (materialized.status != clqr::SolveStatus::kInvalidInput) {
    std::cerr << "CUDA stub materialization returned an unexpected status\n";
    return 1;
  }
  if (!materialized.states.empty() || !materialized.controls.empty() ||
      !materialized.initial_multiplier.empty() ||
      !materialized.dynamics_multipliers.empty() ||
      !materialized.mixed_multipliers.empty() ||
      !materialized.state_multipliers.empty() ||
      !materialized.terminal_state_multiplier.empty() ||
      !materialized.reduced_state_dimensions.empty() ||
      !materialized.reduced_control_dimensions.empty()) {
    std::cerr << "CUDA stub materialization retained stale output storage\n";
    return 1;
  }
  const clqr::cuda::Solution result = clqr::cuda::Solve(problem);
  if (result.status != clqr::SolveStatus::kInvalidInput) {
    std::cerr << "CUDA stub returned an unexpected status\n";
    return 1;
  }
  std::cout << "CUDA-free stub test passed\n";
  return 0;
}
