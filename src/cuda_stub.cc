#include "clqr/cuda.h"

namespace clqr {
namespace cuda {

struct Workspace::Impl {};

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() = default;
Workspace::Workspace(Workspace &&) noexcept = default;
Workspace &Workspace::operator=(Workspace &&) noexcept = default;

void Workspace::Reserve(const Problem &, const Options &) {
  if (!impl_)
    impl_ = std::make_unique<Impl>();
}

bool Available() { return false; }

std::string DeviceDescription(int) { return "CUDA backend not built"; }

SolutionView SolveView(const Problem &, Workspace &, const Options &) {
  SolutionView result;
  result.status = SolveStatus::kInvalidInput;
  result.message = "CUDA backend not built; use Bazel --config=cuda and link "
                   "clqr_cuda";
  return result;
}

SolutionView SolvePreparedView(const Problem &problem, Workspace &workspace,
                               const Options &options) {
  return SolveView(problem, workspace, options);
}

Solution &Materialize(const SolutionView &view, Solution &result) {
  result.states.clear();
  result.controls.clear();
  result.initial_multiplier.resize(0);
  result.dynamics_multipliers.clear();
  result.mixed_multipliers.clear();
  result.state_multipliers.clear();
  result.terminal_state_multiplier.resize(0);
  result.reduced_state_dimensions.clear();
  result.reduced_control_dimensions.clear();
  result.status = view.status;
  result.message = view.message;
  result.objective = view.objective;
  result.timings = view.timings;
  return result;
}

Solution &Solve(const Problem &problem, Workspace &workspace, Solution &result,
                const Options &options) {
  Materialize(SolveView(problem, workspace, options), result);
  return result;
}

Solution &SolvePrepared(const Problem &problem, Workspace &workspace,
                        Solution &result, const Options &options) {
  return Materialize(SolvePreparedView(problem, workspace, options), result);
}

Solution Solve(const Problem &problem, const Options &options) {
  Workspace workspace;
  Solution result;
  Solve(problem, workspace, result, options);
  return result;
}

} // namespace cuda
} // namespace clqr
