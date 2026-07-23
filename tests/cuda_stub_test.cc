#include <iostream>

#include "clqr/cuda.h"

int main() {
  if (clqr::cuda::Available()) {
    std::cerr << "CPU-only CUDA stub unexpectedly reports a device\n";
    return 1;
  }
  clqr::Problem problem;
  const clqr::cuda::Solution result = clqr::cuda::Solve(problem);
  if (result.status != clqr::SolveStatus::kInvalidInput) {
    std::cerr << "CUDA stub returned an unexpected status\n";
    return 1;
  }
  std::cout << "CUDA-free stub test passed\n";
  return 0;
}
