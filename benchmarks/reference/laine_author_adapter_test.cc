#include "benchmarks/reference/laine_author_adapter.h"

#include <iostream>
#include <limits>

namespace ref = clqr::benchmark::reference;

namespace {
void Near(double actual, double expected) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-11)
    throw std::runtime_error("original Laine multiplier mismatch: " +
                             std::to_string(actual) + " != " +
                             std::to_string(expected));
}

void TestTerminalMultipliers() {
  auto p = clqr::benchmark::MakeScalingProblem(1, 1, 1, 0, 0).problem;
  p.initial_state[0] = 1;
  p.Q[0](0, 0) = 2;
  p.Q[1](0, 0) = 3;
  p.q[0][0] = .3;
  p.q[1][0] = -.4;
  auto &s = p.stages[0];
  s.A(0, 0) = s.B(0, 0) = 1;
  s.c[0] = .25;
  s.R(0, 0) = 4;
  s.M(0, 0) = .2;
  s.r[0] = .6;
  p.terminal_E = clqr::Matrix(1, 1);
  p.terminal_E(0, 0) = 2;
  p.terminal_e = clqr::Vector(1);
  p.terminal_e[0] = -1;
  auto t = ref::author::MakeTrajectory(ref::conversion::Convert(p), false);
  for (int repeat = 0; repeat < 3; ++repeat) {
    ref::author::Solve(*t);
    const auto dual = ref::author::CopyMultipliers(p, *t);
    // The constraints fix x0=1, x1=.5, u0=-.75. Direct differentiation
    // gives these unique multipliers in the benchmark's sign convention.
    Near(t->open_loop_states[1][0], .5);
    Near(t->open_loop_controls[0][0], -.75);
    Near(dual.initial[0], -4.35);
    Near(dual.dynamics.at(0)[0], -2.2);
    Near(dual.terminal[0], .55);
    Near(2 + .2 * (-.75) + .3 + dual.initial[0] - dual.dynamics[0][0], 0);
    Near(.2 + 4 * (-.75) + .6 - dual.dynamics[0][0], 0);
    Near(3 * .5 - .4 + dual.dynamics[0][0] + 2 * dual.terminal[0], 0);
  }
}

void TestMultiplierLayout() {
  // Nonuniform active row counts distinguish mixed/state rows and inactive
  // allocated capacity. This tests copying only, not the author's numerics.
  auto p = clqr::benchmark::MakeScalingProblem(3, 3, 3, 1, 1).problem;
  auto t = ref::author::MakeTrajectory(ref::conversion::Convert(p), false);
  for (std::size_t i = 0; i < t->lq_dynamics_multipliers.size(); ++i)
    for (int j = 0; j < 3; ++j)
      t->lq_dynamics_multipliers[i][j] = 10 * i + j + 1;
  for (std::size_t i = 0; i < p.stages.size(); ++i) {
    t->lq_running_constraint_multipliers[i].resize(2);
    t->lq_running_constraint_multipliers[i].setConstant(999);
    t->lq_running_constraint_multipliers[i][0] = 100 + i;
    if (p.stages[i].e.size())
      t->lq_running_constraint_multipliers[i][1] = 200 + i;
  }
  t->lq_terminal_constraint_multiplier.setConstant(300);
  const auto dual = ref::author::CopyMultipliers(p, *t);
  for (int j = 0; j < 3; ++j)
    Near(dual.initial[j], j + 1);
  Near(dual.terminal[0], 300);
  if (dual.state.at(0).size() != 0)
    throw std::runtime_error("inactive author multiplier was copied");
  for (std::size_t i = 0; i < p.stages.size(); ++i) {
    for (int j = 0; j < 3; ++j)
      Near(dual.dynamics.at(i)[j], 10 * (i + 1) + j + 1);
    Near(dual.mixed.at(i)[0], 100 + i);
    if (p.stages[i].e.size())
      Near(dual.state.at(i)[0], 200 + i);
  }
  t->lq_dynamics_multipliers[1][0] = std::numeric_limits<double>::quiet_NaN();
  if (!std::isnan(ref::author::CopyMultipliers(p, *t).dynamics[0][0]))
    throw std::runtime_error("nonfinite author multiplier was hidden");
}
} // namespace

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
  TestTerminalMultipliers();
  TestMultiplierLayout();
  std::cout << "Original Laine primal/dual adapter tests passed\n";
}
