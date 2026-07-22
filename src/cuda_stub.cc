#include "clqr/cuda.h"

namespace clqr {
namespace cuda {

bool Available() { return false; }

std::string DeviceDescription(int) { return "CUDA backend not built"; }

Solution Solve(const Problem&, const Options&) {
  Solution out;
  out.status = SolveStatus::kInvalidInput;
  out.message =
      "CUDA backend not built; configure CMake with CLQR_ENABLE_CUDA=ON";
  return out;
}

}  // namespace cuda
}  // namespace clqr
