#ifndef CLQR_CUDA_H_
#define CLQR_CUDA_H_

#include <string>
#include <vector>

#include "clqr/clqr.h"

namespace clqr {
namespace cuda {

struct Options {
#ifdef CLQR_USE_FLOAT
  Scalar tolerance = 1e-5f;
#else
  Scalar tolerance = 1e-10;
#endif
  int device = 0;
};

struct Timings {
  double upload_ms = 0.0;
  double feasibility_ms = 0.0;
  double reduction_ms = 0.0;
  double riccati_ms = 0.0;
  double reconstruction_ms = 0.0;
  double multiplier_ms = 0.0;
  double download_ms = 0.0;
  double total_ms = 0.0;
};

struct Solution {
  SolveStatus status = SolveStatus::kInvalidInput;
  std::string message;
  std::vector<Vector> states;
  std::vector<Vector> controls;
  Vector initial_multiplier;
  std::vector<Vector> dynamics_multipliers;
  std::vector<Vector> mixed_multipliers;
  std::vector<Vector> state_multipliers;
  Vector terminal_state_multiplier;
  std::vector<int> reduced_state_dimensions;
  std::vector<int> reduced_control_dimensions;
  bool used_parallel_riccati = false;
  bool used_host_multiplier_recovery = false;
  Scalar objective = Scalar{0};
  Timings timings;
};

// The CUDA backend currently targets stage dimensions up to 16. Storage is
// bounded, but every kernel also carries the active per-stage dimensions and
// skips padded rows and columns.
constexpr int kMaxStateDimension = 16;
constexpr int kMaxControlDimension = 16;
constexpr int kMaxMixedConstraints = 16;
constexpr int kMaxStateConstraints = 16;

bool Available();
std::string DeviceDescription(int device = 0);
Solution Solve(const Problem& problem, const Options& options = Options{});

}  // namespace cuda
}  // namespace clqr

#endif  // CLQR_CUDA_H_
