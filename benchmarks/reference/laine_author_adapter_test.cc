#include "benchmarks/reference/laine_author_adapter.h"

#include <iostream>

namespace ref = clqr::benchmark::reference;

int main() {
  ref::Problem p;
  p.initial_state = ref::Vector::Ones(1);
  p.Q = {ref::Matrix::Identity(1, 1), ref::Matrix::Identity(1, 1)};
  p.q = {ref::Vector::Zero(1), ref::Vector::Zero(1)};
  p.terminal_C.resize(0, 1);
  p.terminal_d.resize(0);
  ref::Stage s;
  s.A = s.B = s.R = ref::Matrix::Identity(1, 1);
  s.S = ref::Matrix::Zero(1, 1);
  s.c = s.r = ref::Vector::Zero(1);
  s.C.resize(0, 1);
  s.D.resize(0, 1);
  s.d.resize(0);
  p.stages.push_back(s);
  auto trajectory = ref::author::MakeTrajectory(p, false);
  // Repeated calls must refactor the same problem, without changing inputs.
  for (int repeat = 0; repeat < 3; ++repeat) {
    ref::author::Solve(*trajectory);
    if (std::abs(trajectory->open_loop_controls[0][0] + .5) > 1e-12 ||
        std::abs(trajectory->open_loop_states[1][0] - .5) > 1e-12)
      throw std::runtime_error("original Laine adapter simple solve mismatch");
  }
  std::cout << "Original Laine adapter smoke test passed\n";
}
