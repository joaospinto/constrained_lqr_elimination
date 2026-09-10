#ifndef CLQR_BENCHMARKS_REFERENCE_LAINE_AUTHOR_ADAPTER_H_
#define CLQR_BENCHMARKS_REFERENCE_LAINE_AUTHOR_ADAPTER_H_
#include "benchmarks/reference/eigen_problem.h"
#include "benchmarks/scaling_problem.h"
#include "trajectory.h"
#include <functional>
#include <memory>
#include <stdexcept>
namespace clqr::benchmark::reference::author {
inline std::unique_ptr<trajectory::Trajectory> MakeTrajectory(const Problem &p,
                                                              bool implicit) {
  const int n = p.initial_state.size();
  const int m = p.stages.empty() ? 1 : p.stages[0].B.cols();
  if (!n)
    throw std::runtime_error("unsupported: zero state dimension");
  int rows = 0;
  for (const auto &s : p.stages) {
    if (s.A.rows() != n || s.A.cols() != n || s.B.cols() != m)
      throw std::runtime_error("unsupported: varying state/control dimensions");
    rows = std::max(rows, static_cast<int>(s.d.size()));
  }
  // These callbacks must not be evaluated: the exact linear/quadratic data
  // below already are the inputs to the author's LQR subproblem solver.
  std::function<void(const Vector *, const Vector *, Vector &)> dyn =
      [](auto, auto, auto &) {
        throw std::runtime_error("unexpected dynamics evaluation");
      };
  std::function<void(const Vector *, const Vector *, int, Vector &)> con =
      [](auto, auto, int, auto &) {
        throw std::runtime_error("unexpected constraint evaluation");
      };
  std::function<void(const Vector *, Vector &)> end = [](auto, auto &) {
    throw std::runtime_error("unexpected endpoint evaluation");
  };
  std::function<double(const Vector *, const Vector *)> cost =
      [](auto, auto) -> double {
    throw std::runtime_error("unexpected cost evaluation");
  };
  std::function<double(const Vector *)> final_cost = [](auto) -> double {
    throw std::runtime_error("unexpected terminal cost evaluation");
  };
  dynamics::Dynamics dynamics(&dyn);
  running_constraint::RunningConstraint inequalities(&con, 0);
  equality_constrained_running_constraint::EqualityConstrainedRunningConstraint
      equalities(&con, rows);
  endpoint_constraint::EndPointConstraint terminal_inequalities(&end, 0,
                                                                implicit);
  equality_constrained_endpoint_constraint::
      EqualityConstrainedEndPointConstraint terminal_equalities(
          &end, p.terminal_d.size(), implicit),
      initial(&end, n, implicit);
  running_cost::RunningCost running_cost(&cost);
  terminal_cost::TerminalCost terminal_cost(&final_cost);
  auto t = std::make_unique<trajectory::Trajectory>(
      p.Q.size(), n, m, &dynamics, &inequalities, &equalities,
      &terminal_inequalities, &terminal_equalities, &initial, &running_cost,
      &terminal_cost);
  t->initial_constraint_jacobian_state = Matrix::Identity(n, n);
  t->initial_constraint_affine_term =
      implicit ? Vector::Zero(n).eval() : (-p.initial_state).eval();
  t->lq_initial_state = implicit ? (-p.initial_state).eval() : p.initial_state;
  t->num_active_terminal_constraints = p.terminal_d.size();
  t->active_terminal_constraint_jacobian_state = p.terminal_C;
  t->active_terminal_constraint_affine_term =
      implicit ? Vector::Zero(p.terminal_d.size()).eval() : p.terminal_d;
  t->active_terminal_constraint_jacobian_terminal_projection =
      implicit
          ? Matrix::Identity(p.terminal_d.size(), p.terminal_d.size()).eval()
          : Matrix::Zero(p.terminal_d.size(), 0).eval();
  if (implicit)
    t->lq_terminal_state = p.terminal_d;
  t->terminal_cost_hessians_state_state = p.Q.back();
  t->terminal_cost_gradient_state = p.q.back();
  for (std::size_t k = 0; k < p.stages.size(); ++k) {
    const auto &s = p.stages[k];
    t->dynamics_jacobians_state[k] = s.A;
    t->dynamics_jacobians_control[k] = s.B;
    t->dynamics_affine_terms[k] = s.c;
    t->hamiltonian_hessians_state_state[k] = p.Q[k];
    t->hamiltonian_hessians_control_control[k] = s.R;
    t->hamiltonian_hessians_control_state[k] = s.S.transpose();
    t->hamiltonian_gradients_state[k] = p.q[k];
    t->hamiltonian_gradients_control[k] = s.r;
    t->num_active_constraints[k] = s.d.size();
    t->active_running_constraint_jacobians_state[k] = s.C;
    t->active_running_constraint_jacobians_control[k] = s.D;
    t->active_running_constraint_affine_terms[k] = s.d;
  }
  return t;
}
inline void Solve(trajectory::Trajectory &t) {
  t.compute_feedback_policies();
  t.compute_state_control_dependencies();
  t.set_open_loop_traj();
  t.compute_multipliers();
  t.set_lq_multipliers();
}

// Copy the author's returned duals, without recomputing or repairing them.
// His dynamics array starts with the initial-condition multiplier; subsequent
// entries have the same x[t+1]-A*x[t]-B*u[t]-c sign convention as our audit.
inline clqr::benchmark::Multipliers CopyMultipliers(
    const clqr::Problem &p, const trajectory::Trajectory &t) {
  const auto copy = [](const auto &v) {
    clqr::Vector out(v.size());
    for (Eigen::Index i = 0; i < v.size(); ++i)
      out[i] = v[i];
    return out;
  };
  clqr::benchmark::Multipliers out;
  out.initial = copy(t.lq_dynamics_multipliers.at(0));
  out.terminal = copy(t.lq_terminal_constraint_multiplier.head(p.terminal_e.size()));
  for (std::size_t i = 0; i < p.stages.size(); ++i) {
    out.dynamics.push_back(copy(t.lq_dynamics_multipliers.at(i + 1)));
    const auto &s = p.stages[i];
    const auto &mu = t.lq_running_constraint_multipliers.at(i);
    // Convert() stacks mixed rows first, followed by state-only rows. Ignore
    // inactive capacity beyond those rows, not any active multiplier.
    out.mixed.push_back(copy(mu.head(s.d.size())));
    out.state.push_back(copy(mu.segment(s.d.size(), s.e.size())));
  }
  return out;
}
} // namespace clqr::benchmark::reference::author
#endif
