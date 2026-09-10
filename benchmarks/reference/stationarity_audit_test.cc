#include "benchmarks/reference/stationarity_audit.h"

#include <iostream>

namespace ref = clqr::benchmark::reference;

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  // min .5*(x0^2 + u0^2 + x1^2), x0=1, x1=x0+u0.
  clqr::Problem original;
  original.initial_state = clqr::Vector{1};
  original.Q = {clqr::Matrix(1, 1), clqr::Matrix(1, 1)};
  original.Q[0](0, 0) = original.Q[1](0, 0) = 1;
  original.q = {clqr::Vector(1), clqr::Vector(1)};
  original.terminal_E = clqr::Matrix(0, 1);
  original.stages.resize(1);
  auto& stage = original.stages[0];
  stage.A = stage.B = stage.R = clqr::Matrix(1, 1);
  stage.A(0, 0) = stage.B(0, 0) = stage.R(0, 0) = 1;
  stage.M = clqr::Matrix(1, 1);
  stage.c = stage.r = clqr::Vector(1);
  // Redundant state-only rows must be appended to mixed rows, with zero D.
  stage.C = stage.D = clqr::Matrix(0, 1);
  stage.E = clqr::Matrix(2, 1);
  stage.E(0, 0) = 1;
  stage.E(1, 0) = 4;
  stage.e = clqr::Vector{-1, -4};
  auto p = ref::conversion::Convert(original);
  Require(p.stages[0].C.rows() == 2 && p.stages[0].D.norm() == 0,
          "state-only conversion changed constraints");
  ref::Trajectory result{{ref::Vector::Ones(1), ref::Vector::Constant(1, .5)},
                         {ref::Vector::Constant(1, -.5)}};
  auto [feasibility, stationarity] = ref::audit::Audit(p, result);
  Require(feasibility < 1e-12 && stationarity < 1e-12,
          "optimal trajectory failed independent audit");
  result.states[1][0] = 1;
  result.controls[0][0] = 0;
  auto wrong = ref::audit::Audit(p, result);
  Require(wrong.first == 0 && wrong.second > .1,
          "audit accepted feasible but nonoptimal trajectory");
  result.states[0][0] = 2;
  Require(ref::audit::Audit(p, result).first >= 1,
          "audit missed infeasibility");

  // Zero horizon and fully fixed state: stationarity admits an initial dual.
  original.stages.clear();
  original.Q.resize(1);
  original.q.resize(1);
  p = ref::conversion::Convert(original);
  result = {{ref::Vector::Ones(1)}, {}};
  auto terminal = ref::audit::Audit(p, result);
  Require(terminal.first == 0 && terminal.second < 1e-12,
          "zero-horizon audit failed");
  std::cout << "Independent primal-only stationarity audit passed\n";
}
