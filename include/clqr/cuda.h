#ifndef CLQR_CUDA_H_
#define CLQR_CUDA_H_

#include <memory>
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
  // Keep enabled in applications. Benchmarks may disable this to inspect and
  // report the KKT residual of a best-effort multiplier reconstruction.
  bool enforce_multiplier_consistency = true;
};

struct Timings {
  // Bulk packed-input transfer, measured with CUDA events.
  double upload_ms = 0.0;
  // Computational phases below contain kernel event time only; phase-control
  // transfers and synchronization are intentionally excluded.
  double feasibility_ms = 0.0;
  double reduction_ms = 0.0;
  double riccati_ms = 0.0;
  double reconstruction_ms = 0.0;
  double multiplier_ms = 0.0;
  // Bulk packed-output transfer, measured with CUDA events.
  double download_ms = 0.0;
  // End-to-end host wall time, including packing, all transfers,
  // synchronization, kernels, and construction of the owning result.
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
  Scalar objective = Scalar{0};
  Timings timings;
};

// Capacities are compile-time parameters because they bound each kernel's
// small dense shared-memory workspace. Global problem and trajectory buffers
// are packed using the active dimensions of every individual stage.
#ifndef CLQR_CUDA_MAX_STATE_DIMENSION
#define CLQR_CUDA_MAX_STATE_DIMENSION 8
#endif
#ifndef CLQR_CUDA_MAX_CONTROL_DIMENSION
#define CLQR_CUDA_MAX_CONTROL_DIMENSION 8
#endif
#ifndef CLQR_CUDA_MAX_MIXED_CONSTRAINTS
#define CLQR_CUDA_MAX_MIXED_CONSTRAINTS 8
#endif
#ifndef CLQR_CUDA_MAX_STATE_CONSTRAINTS
#define CLQR_CUDA_MAX_STATE_CONSTRAINTS 8
#endif
constexpr int kMaxStateDimension = CLQR_CUDA_MAX_STATE_DIMENSION;
constexpr int kMaxControlDimension = CLQR_CUDA_MAX_CONTROL_DIMENSION;
constexpr int kMaxMixedConstraints = CLQR_CUDA_MAX_MIXED_CONSTRAINTS;
constexpr int kMaxStateConstraints = CLQR_CUDA_MAX_STATE_CONSTRAINTS;

class Workspace {
public:
  Workspace();
  ~Workspace();
  Workspace(Workspace &&) noexcept;
  Workspace &operator=(Workspace &&) noexcept;
  Workspace(const Workspace &) = delete;
  Workspace &operator=(const Workspace &) = delete;

  void Reserve(const Problem &problem, const Options &options = Options{});

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend Solution &Solve(const Problem &, Workspace &, Solution &,
                         const Options &);
};

bool Available();
std::string DeviceDescription(int device = 0);
Solution &Solve(const Problem &problem, Workspace &workspace, Solution &result,
                const Options &options = Options{});
Solution Solve(const Problem &problem, const Options &options = Options{});

} // namespace cuda
} // namespace clqr

#endif // CLQR_CUDA_H_
