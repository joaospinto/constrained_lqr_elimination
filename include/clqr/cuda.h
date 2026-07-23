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
  // Host-side packing of the numerical problem into reusable pinned storage.
  double input_pack_ms = 0.0;
  // Device time for bulk packed-input transfer and status initialization.
  // The first solve may also upload cached pointer-bearing metadata.
  double upload_ms = 0.0;
  // Computational phases below contain kernel event time only; phase-control
  // transfers and synchronization are intentionally excluded.
  double feasibility_ms = 0.0;
  double reduction_ms = 0.0;
  double riccati_ms = 0.0;
  double reconstruction_ms = 0.0;
  double multiplier_ms = 0.0;
  // Device time for packed-output, status, and objective transfers.
  double download_ms = 0.0;
  // Host work that updates compact layouts after active dimensions are known.
  double layout_ms = 0.0;
  // Host construction/copying of an owning per-stage result. This is zero for
  // SolveView and is added by Materialize/Solve.
  double result_ms = 0.0;
  // Device-side objective evaluation and deterministic tree reduction.
  double objective_ms = 0.0;
  // Host-blocking stream/event synchronizations performed by this solve.
  int synchronization_count = 0;
  // End-to-end host wall time, including packing, all transfers,
  // synchronization, kernels, and construction of the returned
  // representation.
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
  Scalar objective = Scalar{0};
  Timings timings;
};

struct StridedIntView {
  const int *data = nullptr;
  std::size_t size = 0;
  std::size_t stride = 1;

  const int &operator[](std::size_t index) const {
    return data[index * stride];
  }
};

// Non-owning host view backed by Workspace. It remains valid until that
// workspace is reserved for a different structure, reused by another solve,
// moved, or destroyed.
struct SolutionView {
  SolveStatus status = SolveStatus::kInvalidInput;
  const char *message = "";
  VectorView *states = nullptr;
  std::size_t state_count = 0;
  VectorView *controls = nullptr;
  std::size_t control_count = 0;
  VectorView initial_multiplier;
  VectorView *dynamics_multipliers = nullptr;
  std::size_t dynamics_multiplier_count = 0;
  VectorView *mixed_multipliers = nullptr;
  std::size_t mixed_multiplier_count = 0;
  VectorView *state_multipliers = nullptr;
  std::size_t state_multiplier_count = 0;
  VectorView terminal_state_multiplier;
  StridedIntView reduced_state_dimensions;
  StridedIntView reduced_control_dimensions;
  Scalar objective = Scalar{0};
  Timings timings;
};

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
  friend SolutionView SolveView(const Problem &, Workspace &, const Options &);
  friend SolutionView SolvePreparedView(const Problem &, Workspace &,
                                        const Options &);
  friend Solution &Solve(const Problem &, Workspace &, Solution &,
                         const Options &);
};

bool Available();
std::string DeviceDescription(int device = 0);
SolutionView SolveView(const Problem &problem, Workspace &workspace,
                       const Options &options = Options{});
// Skips repeated structural validation after Workspace::Reserve. The problem
// dimensions and matrix/vector shapes must remain unchanged; numerical values
// may change.
SolutionView SolvePreparedView(const Problem &problem, Workspace &workspace,
                               const Options &options = Options{});
Solution &Materialize(const SolutionView &view, Solution &result);
Solution &SolvePrepared(const Problem &problem, Workspace &workspace,
                        Solution &result, const Options &options = Options{});
Solution &Solve(const Problem &problem, Workspace &workspace, Solution &result,
                const Options &options = Options{});
Solution Solve(const Problem &problem, const Options &options = Options{});

} // namespace cuda
} // namespace clqr

#endif // CLQR_CUDA_H_
