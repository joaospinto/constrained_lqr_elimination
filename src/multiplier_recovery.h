#ifndef CLQR_MULTIPLIER_RECOVERY_H_
#define CLQR_MULTIPLIER_RECOVERY_H_

#include <string>
#include <vector>

#include "clqr/clqr.h"

namespace clqr {
namespace internal {

struct MultiplierRecovery {
  bool success = false;
  std::string message;
  Vector initial;
  std::vector<Vector> dynamics;
  std::vector<Vector> mixed;
  std::vector<Vector> state;
  Vector terminal;
};

MultiplierRecovery RecoverMultipliersForTrajectory(
    const Problem& problem, const std::vector<Vector>& states,
    const std::vector<Vector>& controls, Scalar tolerance);

}  // namespace internal
}  // namespace clqr

#endif  // CLQR_MULTIPLIER_RECOVERY_H_
