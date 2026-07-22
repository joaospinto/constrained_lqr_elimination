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

Solution &Solve(const Problem &, Workspace &, Solution &result,
                const Options &) {
  result.status = SolveStatus::kInvalidInput;
  result.message =
      "CUDA backend not built; configure CMake with CLQR_ENABLE_CUDA=ON";
  return result;
}

Solution Solve(const Problem &problem, const Options &options) {
  Workspace workspace;
  Solution result;
  Solve(problem, workspace, result, options);
  return result;
}

} // namespace cuda
} // namespace clqr
