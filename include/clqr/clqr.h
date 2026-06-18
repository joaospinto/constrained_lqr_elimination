#ifndef CLQR_CLQR_H_
#define CLQR_CLQR_H_

#include <string>
#include <vector>

#include "clqr/linalg.h"

namespace clqr {

struct Stage {
  Matrix A;
  Matrix B;
  Vector c;

  Matrix Q;
  Matrix R;
  Matrix M;
  Vector q;
  Vector r;

  Matrix C;
  Matrix D;
  Vector d;

  Matrix E;
  Vector e;
};

struct Problem {
  std::vector<Stage> stages;
  Matrix terminal_Q;
  Vector terminal_q;
  Matrix terminal_E;
  Vector terminal_e;
  Vector initial_state;
};

enum class SolveStatus {
  kOptimal,
  kInfeasible,
  kInvalidInput,
  kNumericalFailure,
};

struct SolveOptions {
  double tolerance = 1e-9;
  int max_elimination_passes = 100;
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
  bool newton_kkt_singular = false;
  bool newton_kkt_wrong_inertia = false;
  std::string newton_kkt_diagnostic;
  double objective = 0.0;
};

Solution Solve(const Problem& problem, const SolveOptions& options = SolveOptions{});
const char* StatusName(SolveStatus status);

}  // namespace clqr

#endif  // CLQR_CLQR_H_
