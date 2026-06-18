#include "clqr/clqr.h"

#include <algorithm>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace clqr {
namespace {

struct StateMap {
  Matrix linear;
  Vector offset;
};

struct ControlMap {
  Matrix state_linear;
  Matrix control_linear;
  Vector offset;
};

struct WorkingProblem {
  std::vector<Stage> stages;
  Matrix terminal_Q;
  Vector terminal_q;
  Matrix terminal_E;
  Vector terminal_e;
};

struct WorkingState {
  WorkingProblem problem;
  std::vector<StateMap> state_maps;
  std::vector<ControlMap> control_maps;
};

struct AffineStateBasis {
  Matrix T;
  Vector offset;
  std::vector<std::size_t> free_rows;
  std::vector<std::size_t> pivot_rows;
  bool infeasible = false;
  std::string message;
};

void Check(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}

Matrix VectorAsColumn(const Vector& v) {
  Matrix out(v.size(), 1);
  for (std::size_t i = 0; i < v.size(); ++i) out(i, 0) = v[i];
  return out;
}

Matrix Symmetrize(const Matrix& a) {
  Check(a.rows() == a.cols(), "cannot symmetrize a rectangular matrix");
  Matrix out(a.rows(), a.cols());
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) out(i, j) = 0.5 * (a(i, j) + a(j, i));
  }
  return out;
}

void AppendStateConstraints(Stage& stage, const Matrix& E_extra, const Vector& e_extra) {
  if (E_extra.rows() == 0) return;
  if (stage.E.rows() == 0) {
    stage.E = E_extra;
    stage.e = e_extra;
  } else {
    stage.E = VerticalConcat(stage.E, E_extra);
    stage.e = Concat(stage.e, e_extra);
  }
}

void AppendMixedConstraints(Stage& stage, const Matrix& C_extra, const Matrix& D_extra,
                            const Vector& d_extra) {
  if (C_extra.rows() == 0) return;
  if (stage.C.rows() == 0) {
    stage.C = C_extra;
    stage.D = D_extra;
    stage.d = d_extra;
  } else {
    stage.C = VerticalConcat(stage.C, C_extra);
    stage.D = VerticalConcat(stage.D, D_extra);
    stage.d = Concat(stage.d, d_extra);
  }
}

void ValidateProblem(const Problem& problem) {
  const std::size_t N = problem.stages.size();
  Check(problem.initial_state.size() > 0 || N == 0, "initial_state must be set");
  Check(problem.terminal_Q.rows() == problem.terminal_Q.cols(), "terminal_Q must be square");
  Check(problem.terminal_q.size() == problem.terminal_Q.rows(), "terminal_q shape mismatch");
  Check(problem.terminal_E.cols() == problem.terminal_Q.rows(), "terminal_E shape mismatch");
  Check(problem.terminal_e.size() == problem.terminal_E.rows(), "terminal_e shape mismatch");
  if (N == 0) {
    Check(problem.initial_state.size() == problem.terminal_Q.rows(),
          "initial_state and terminal_Q shape mismatch");
    return;
  }
  Check(problem.stages.front().A.cols() == problem.initial_state.size(),
        "initial_state and first stage shape mismatch");
  for (std::size_t i = 0; i < N; ++i) {
    const Stage& s = problem.stages[i];
    const std::size_t n = s.A.cols();
    const std::size_t next_n = s.A.rows();
    const std::size_t m = s.B.cols();
    Check(s.B.rows() == next_n, "B row count must match A row count");
    Check(s.c.size() == next_n, "c shape mismatch");
    Check(s.Q.rows() == n && s.Q.cols() == n, "Q shape mismatch");
    Check(s.R.rows() == m && s.R.cols() == m, "R shape mismatch");
    Check(s.M.rows() == n && s.M.cols() == m, "M shape mismatch");
    Check(s.q.size() == n, "q shape mismatch");
    Check(s.r.size() == m, "r shape mismatch");
    Check(s.C.cols() == n, "C shape mismatch");
    Check(s.D.rows() == s.C.rows() && s.D.cols() == m, "D shape mismatch");
    Check(s.d.size() == s.C.rows(), "d shape mismatch");
    Check(s.E.cols() == n, "E shape mismatch");
    Check(s.e.size() == s.E.rows(), "e shape mismatch");
    const std::size_t expected_next =
        (i + 1 == N) ? problem.terminal_Q.rows() : problem.stages[i + 1].A.cols();
    Check(next_n == expected_next, "neighboring stage dimensions do not match");
  }
}

double Objective(const Problem& original, const std::vector<Vector>& xs,
                 const std::vector<Vector>& us) {
  double value = 0.0;
  for (std::size_t i = 0; i < original.stages.size(); ++i) {
    const Stage& s = original.stages[i];
    value += 0.5 * Dot(xs[i], s.Q * xs[i]);
    value += 0.5 * Dot(us[i], s.R * us[i]);
    value += Dot(xs[i], s.M * us[i]);
    value += Dot(s.q, xs[i]);
    value += Dot(s.r, us[i]);
  }
  value += 0.5 * Dot(xs.back(), original.terminal_Q * xs.back());
  value += Dot(original.terminal_q, xs.back());
  return value;
}

WorkingState Initialize(const Problem& problem) {
  WorkingState state;
  state.problem.stages = problem.stages;
  state.problem.terminal_Q = problem.terminal_Q;
  state.problem.terminal_q = problem.terminal_q;
  state.problem.terminal_E = problem.terminal_E;
  state.problem.terminal_e = problem.terminal_e;

  if (!state.problem.stages.empty()) {
    Stage& first = state.problem.stages.front();
    AppendStateConstraints(first, Identity(problem.initial_state.size()),
                           Scale(problem.initial_state, -1.0));
  } else if (problem.initial_state.size() > 0) {
    if (state.problem.terminal_E.rows() == 0) {
      state.problem.terminal_E = Identity(problem.initial_state.size());
      state.problem.terminal_e = Scale(problem.initial_state, -1.0);
    } else {
      state.problem.terminal_E =
          VerticalConcat(state.problem.terminal_E, Identity(problem.initial_state.size()));
      state.problem.terminal_e =
          Concat(state.problem.terminal_e, Scale(problem.initial_state, -1.0));
    }
  }

  const std::size_t num_states = problem.stages.size() + 1;
  state.state_maps.resize(num_states);
  for (std::size_t i = 0; i < num_states; ++i) {
    const std::size_t n = (i == problem.stages.size()) ? problem.terminal_Q.rows()
                                                       : problem.stages[i].A.cols();
    state.state_maps[i].linear = Identity(n);
    state.state_maps[i].offset = Vector(n);
  }
  state.control_maps.resize(problem.stages.size());
  for (std::size_t i = 0; i < problem.stages.size(); ++i) {
    const std::size_t n = problem.stages[i].A.cols();
    const std::size_t m = problem.stages[i].B.cols();
    state.control_maps[i].state_linear = Matrix(m, n);
    state.control_maps[i].control_linear = Identity(m);
    state.control_maps[i].offset = Vector(m);
  }
  return state;
}

AffineStateBasis StateBasis(const Matrix& E, const Vector& e, std::size_t n,
                            double tolerance) {
  AffineStateBasis out;
  if (E.rows() == 0) {
    out.T = Identity(n);
    out.offset = Vector(n);
    out.free_rows.resize(n);
    for (std::size_t i = 0; i < n; ++i) out.free_rows[i] = i;
    return out;
  }
  Matrix augmented = HorizontalConcat(E, VectorAsColumn(e));
  RrefResult rref = Rref(augmented, n, tolerance);
  std::unordered_set<std::size_t> pivot_set(rref.pivot_columns.begin(), rref.pivot_columns.end());
  for (std::size_t row = 0; row < rref.matrix.rows(); ++row) {
    bool zero_e = true;
    for (std::size_t col = 0; col < n; ++col) {
      if (!IsNearlyZero(rref.matrix(row, col), tolerance)) {
        zero_e = false;
        break;
      }
    }
    if (zero_e && !IsNearlyZero(rref.matrix(row, n), tolerance)) {
      out.infeasible = true;
      out.message = "inconsistent state equality constraints";
      return out;
    }
  }

  for (std::size_t col = 0; col < n; ++col) {
    if (pivot_set.find(col) == pivot_set.end()) out.free_rows.push_back(col);
  }
  out.pivot_rows = rref.pivot_columns;
  out.T = Matrix(n, out.free_rows.size());
  out.offset = Vector(n);
  for (std::size_t j = 0; j < out.free_rows.size(); ++j) out.T(out.free_rows[j], j) = 1.0;
  for (std::size_t pivot_index = 0; pivot_index < rref.pivot_columns.size(); ++pivot_index) {
    const std::size_t pivot_col = rref.pivot_columns[pivot_index];
    const std::size_t row = rref.pivot_rows[pivot_index];
    out.offset[pivot_col] = -rref.matrix(row, n);
    for (std::size_t free_index = 0; free_index < out.free_rows.size(); ++free_index) {
      out.T(pivot_col, free_index) = -rref.matrix(row, out.free_rows[free_index]);
    }
  }
  return out;
}

struct MixedElimination {
  Matrix Y;
  Matrix Z;
  Vector y;
  Matrix state_C;
  Vector state_d;
  bool infeasible = false;
  std::string message;
};

MixedElimination ControlBasis(const Matrix& C, const Matrix& D, const Vector& d,
                              double tolerance) {
  MixedElimination out;
  const std::size_t constraints = D.rows();
  const std::size_t m = D.cols();
  const std::size_t n = C.cols();
  if (constraints == 0) {
    out.Y = Matrix(m, n);
    out.Z = Identity(m);
    out.y = Vector(m);
    out.state_C = Matrix(0, n);
    out.state_d = Vector(0);
    return out;
  }
  Matrix augmented = HorizontalConcat(HorizontalConcat(D, C), VectorAsColumn(d));
  RrefResult rref = Rref(augmented, m, tolerance);
  std::unordered_set<std::size_t> pivot_set(rref.pivot_columns.begin(), rref.pivot_columns.end());
  std::vector<std::size_t> free_cols;
  for (std::size_t col = 0; col < m; ++col) {
    if (pivot_set.find(col) == pivot_set.end()) free_cols.push_back(col);
  }
  out.Y = Matrix(m, n);
  out.Z = Matrix(m, free_cols.size());
  out.y = Vector(m);
  for (std::size_t j = 0; j < free_cols.size(); ++j) out.Z(free_cols[j], j) = 1.0;
  for (std::size_t pivot_index = 0; pivot_index < rref.pivot_columns.size(); ++pivot_index) {
    const std::size_t pivot_col = rref.pivot_columns[pivot_index];
    const std::size_t row = rref.pivot_rows[pivot_index];
    out.y[pivot_col] = -rref.matrix(row, m + n);
    for (std::size_t x_col = 0; x_col < n; ++x_col) {
      out.Y(pivot_col, x_col) = -rref.matrix(row, m + x_col);
    }
    for (std::size_t free_index = 0; free_index < free_cols.size(); ++free_index) {
      out.Z(pivot_col, free_index) = -rref.matrix(row, free_cols[free_index]);
    }
  }

  std::vector<std::size_t> residual_rows;
  std::unordered_set<std::size_t> pivot_rows(rref.pivot_rows.begin(), rref.pivot_rows.end());
  for (std::size_t row = 0; row < constraints; ++row) {
    if (pivot_rows.find(row) != pivot_rows.end()) continue;
    bool zero_d = true;
    for (std::size_t col = 0; col < m; ++col) {
      if (!IsNearlyZero(rref.matrix(row, col), tolerance)) {
        zero_d = false;
        break;
      }
    }
    if (!zero_d) continue;
    bool any_state = false;
    for (std::size_t col = 0; col < n; ++col) {
      if (!IsNearlyZero(rref.matrix(row, m + col), tolerance)) {
        any_state = true;
        break;
      }
    }
    const bool any_const = !IsNearlyZero(rref.matrix(row, m + n), tolerance);
    if (!any_state && any_const) {
      out.infeasible = true;
      out.message = "inconsistent mixed equality constraints";
      return out;
    }
    if (any_state || any_const) residual_rows.push_back(row);
  }
  out.state_C = Matrix(residual_rows.size(), n);
  out.state_d = Vector(residual_rows.size());
  for (std::size_t i = 0; i < residual_rows.size(); ++i) {
    const std::size_t row = residual_rows[i];
    for (std::size_t col = 0; col < n; ++col) out.state_C(i, col) = rref.matrix(row, m + col);
    out.state_d[i] = rref.matrix(row, m + n);
  }
  return out;
}

bool EliminateMixedWithMaps(WorkingState& state, double tolerance, std::string* error) {
  bool changed = false;
  for (std::size_t i = 0; i < state.problem.stages.size(); ++i) {
    Stage& s = state.problem.stages[i];
    if (s.C.rows() == 0) continue;
    MixedElimination basis = ControlBasis(s.C, s.D, s.d, tolerance);
    if (basis.infeasible) {
      *error = basis.message;
      throw std::runtime_error(*error);
    }
    const Matrix old_Q = s.Q;
    const Matrix old_R = s.R;
    const Matrix old_M = s.M;
    const Matrix old_B = s.B;
    const Vector old_q = s.q;
    const Vector old_r = s.r;
    const Vector old_c = s.c;
    const Vector Ry = old_R * basis.y;

    s.Q = Symmetrize(old_Q + Transpose(basis.Y) * old_R * basis.Y +
                     old_M * basis.Y + Transpose(old_M * basis.Y));
    s.R = Symmetrize(Transpose(basis.Z) * old_R * basis.Z);
    s.M = old_M * basis.Z + Transpose(basis.Y) * old_R * basis.Z;
    s.q = old_q + Transpose(basis.Y) * (Ry + old_r) + old_M * basis.y;
    s.r = Transpose(basis.Z) * (Ry + old_r);
    s.A = s.A + old_B * basis.Y;
    s.B = old_B * basis.Z;
    s.c = old_c + old_B * basis.y;
    s.C = Matrix(0, s.A.cols());
    s.D = Matrix(0, s.B.cols());
    s.d = Vector(0);
    AppendStateConstraints(s, basis.state_C, basis.state_d);

    ControlMap old_map = state.control_maps[i];
    state.control_maps[i].state_linear = old_map.state_linear + old_map.control_linear * basis.Y;
    state.control_maps[i].control_linear = old_map.control_linear * basis.Z;
    state.control_maps[i].offset = old_map.offset + old_map.control_linear * basis.y;
    changed = true;
  }
  return changed;
}

bool AnyStateConstraints(const WorkingProblem& problem) {
  for (const Stage& s : problem.stages) {
    if (s.E.rows() > 0) return true;
  }
  return problem.terminal_E.rows() > 0;
}

bool AnyMixedConstraints(const WorkingProblem& problem) {
  for (const Stage& s : problem.stages) {
    if (s.C.rows() > 0) return true;
  }
  return false;
}

bool EliminateState(WorkingState& state, double tolerance, std::string* error) {
  if (!AnyStateConstraints(state.problem)) return false;
  const std::size_t N = state.problem.stages.size();
  std::vector<AffineStateBasis> bases(N + 1);
  for (std::size_t i = 0; i <= N; ++i) {
    const Matrix& E = (i == N) ? state.problem.terminal_E : state.problem.stages[i].E;
    const Vector& e = (i == N) ? state.problem.terminal_e : state.problem.stages[i].e;
    const std::size_t n = (i == N) ? state.problem.terminal_Q.rows()
                                   : state.problem.stages[i].A.cols();
    bases[i] = StateBasis(E, e, n, tolerance);
    if (bases[i].infeasible) {
      *error = bases[i].message;
      throw std::runtime_error(*error);
    }
  }

  for (std::size_t i = 0; i < N; ++i) {
    Stage& s = state.problem.stages[i];
    const AffineStateBasis& cur = bases[i];
    const AffineStateBasis& next = bases[i + 1];
    const Matrix old_A = s.A;
    const Matrix old_B = s.B;
    const Matrix old_Q = s.Q;
    const Matrix old_M = s.M;
    const Vector old_q = s.q;
    const Vector old_c = s.c;
    const Matrix full_A = old_A * cur.T;
    const Vector full_g = old_A * cur.offset + old_c;

    Matrix new_A = Rows(full_A, next.free_rows);
    Matrix new_B = Rows(old_B, next.free_rows);
    Vector new_c = Entries(full_g, next.free_rows);

    Matrix C_extra(next.pivot_rows.size(), cur.T.cols());
    Matrix D_extra(next.pivot_rows.size(), old_B.cols());
    Vector d_extra(next.pivot_rows.size());
    for (std::size_t row = 0; row < next.pivot_rows.size(); ++row) {
      const std::size_t pivot = next.pivot_rows[row];
      for (std::size_t col = 0; col < cur.T.cols(); ++col) {
        double value = full_A(pivot, col);
        for (std::size_t f = 0; f < next.free_rows.size(); ++f) {
          value -= next.T(pivot, f) * full_A(next.free_rows[f], col);
        }
        C_extra(row, col) = value;
      }
      for (std::size_t col = 0; col < old_B.cols(); ++col) {
        double value = old_B(pivot, col);
        for (std::size_t f = 0; f < next.free_rows.size(); ++f) {
          value -= next.T(pivot, f) * old_B(next.free_rows[f], col);
        }
        D_extra(row, col) = value;
      }
      double value = full_g[pivot] - next.offset[pivot];
      for (std::size_t f = 0; f < next.free_rows.size(); ++f) {
        value -= next.T(pivot, f) * full_g[next.free_rows[f]];
      }
      d_extra[row] = value;
    }

    s.Q = Symmetrize(Transpose(cur.T) * old_Q * cur.T);
    s.M = Transpose(cur.T) * old_M;
    s.q = Transpose(cur.T) * (old_q + old_Q * cur.offset);
    s.r = s.r + Transpose(old_M) * cur.offset;
    s.A = new_A;
    s.B = new_B;
    s.c = new_c;
    s.C = Matrix(0, cur.T.cols());
    s.D = Matrix(0, old_B.cols());
    s.d = Vector(0);
    s.E = Matrix(0, cur.T.cols());
    s.e = Vector(0);
    AppendMixedConstraints(s, C_extra, D_extra, d_extra);

    StateMap old_state_map = state.state_maps[i];
    state.state_maps[i].linear = old_state_map.linear * cur.T;
    state.state_maps[i].offset = old_state_map.offset + old_state_map.linear * cur.offset;

    ControlMap old_control_map = state.control_maps[i];
    state.control_maps[i].state_linear = old_control_map.state_linear * cur.T;
    state.control_maps[i].offset =
        old_control_map.offset + old_control_map.state_linear * cur.offset;
  }

  const AffineStateBasis& terminal = bases[N];
  const Matrix old_terminal_Q = state.problem.terminal_Q;
  const Vector old_terminal_q = state.problem.terminal_q;
  state.problem.terminal_Q = Symmetrize(Transpose(terminal.T) * old_terminal_Q * terminal.T);
  state.problem.terminal_q = Transpose(terminal.T) * (old_terminal_q + old_terminal_Q * terminal.offset);
  state.problem.terminal_E = Matrix(0, terminal.T.cols());
  state.problem.terminal_e = Vector(0);
  StateMap old_terminal_map = state.state_maps[N];
  state.state_maps[N].linear = old_terminal_map.linear * terminal.T;
  state.state_maps[N].offset = old_terminal_map.offset + old_terminal_map.linear * terminal.offset;
  return true;
}

struct ReducedSolution {
  std::vector<Vector> x;
  std::vector<Vector> u;
};

ReducedSolution SolveUnconstrained(const WorkingProblem& problem, double tolerance) {
  const std::size_t N = problem.stages.size();
  std::vector<Matrix> P(N + 1);
  std::vector<Vector> p(N + 1);
  std::vector<Matrix> K(N);
  std::vector<Vector> k(N);
  P[N] = problem.terminal_Q;
  p[N] = problem.terminal_q;
  for (std::size_t rev = 0; rev < N; ++rev) {
    const std::size_t i = N - 1 - rev;
    const Stage& s = problem.stages[i];
    const Vector pc = p[i + 1] + P[i + 1] * s.c;
    const Matrix Hxx = s.Q + Transpose(s.A) * P[i + 1] * s.A;
    const Matrix Huu = s.R + Transpose(s.B) * P[i + 1] * s.B;
    const Matrix Hxu = s.M + Transpose(s.A) * P[i + 1] * s.B;
    const Vector hx = s.q + Transpose(s.A) * pc;
    const Vector hu = s.r + Transpose(s.B) * pc;
    Matrix solve_hxu = SolveLinearSystem(Huu, Transpose(Hxu), tolerance);
    Vector solve_hu = SolveLinearSystem(Huu, hu, tolerance);
    K[i] = Scale(solve_hxu, -1.0);
    k[i] = Scale(solve_hu, -1.0);
    P[i] = Symmetrize(Hxx - Hxu * solve_hxu);
    p[i] = hx - Hxu * solve_hu;
  }

  ReducedSolution sol;
  sol.x.resize(N + 1);
  sol.u.resize(N);
  sol.x[0] = Vector(problem.stages.empty() ? problem.terminal_Q.rows()
                                           : problem.stages.front().A.cols());
  for (std::size_t i = 0; i < N; ++i) {
    sol.u[i] = K[i] * sol.x[i] + k[i];
    sol.x[i + 1] = problem.stages[i].A * sol.x[i] + problem.stages[i].B * sol.u[i] +
                   problem.stages[i].c;
  }
  return sol;
}

Solution Recover(const Problem& original, const WorkingState& state,
                 const ReducedSolution& reduced) {
  Solution out;
  out.status = SolveStatus::kOptimal;
  out.message = "optimal";
  out.states.resize(reduced.x.size());
  out.controls.resize(reduced.u.size());
  for (std::size_t i = 0; i < reduced.x.size(); ++i) {
    out.states[i] = state.state_maps[i].linear * reduced.x[i] + state.state_maps[i].offset;
  }
  for (std::size_t i = 0; i < reduced.u.size(); ++i) {
    out.controls[i] = state.control_maps[i].state_linear * reduced.x[i] +
                      state.control_maps[i].control_linear * reduced.u[i] +
                      state.control_maps[i].offset;
  }
  out.objective = Objective(original, out.states, out.controls);
  return out;
}

}  // namespace

const char* StatusName(SolveStatus status) {
  switch (status) {
    case SolveStatus::kOptimal:
      return "optimal";
    case SolveStatus::kInfeasible:
      return "infeasible";
    case SolveStatus::kInvalidInput:
      return "invalid_input";
    case SolveStatus::kNumericalFailure:
      return "numerical_failure";
  }
  return "unknown";
}

Solution Solve(const Problem& problem, const SolveOptions& options) {
  Solution out;
  try {
    ValidateProblem(problem);
    WorkingState state = Initialize(problem);
    std::string error;
    for (int pass = 0; pass < options.max_elimination_passes; ++pass) {
      const bool mixed = EliminateMixedWithMaps(state, options.tolerance, &error);
      const bool state_changed = EliminateState(state, options.tolerance, &error);
      if (!AnyMixedConstraints(state.problem) && !AnyStateConstraints(state.problem)) break;
      if (!mixed && !state_changed) {
        out.status = SolveStatus::kNumericalFailure;
        out.message = "constraint elimination made no progress";
        return out;
      }
      if (pass + 1 == options.max_elimination_passes) {
        out.status = SolveStatus::kNumericalFailure;
        out.message = "constraint elimination pass limit exceeded";
        return out;
      }
    }
    ReducedSolution reduced = SolveUnconstrained(state.problem, options.tolerance);
    return Recover(problem, state, reduced);
  } catch (const std::invalid_argument& e) {
    out.status = SolveStatus::kInvalidInput;
    out.message = e.what();
    return out;
  } catch (const std::runtime_error& e) {
    out.status = SolveStatus::kInfeasible;
    out.message = e.what();
    return out;
  } catch (const std::exception& e) {
    out.status = SolveStatus::kNumericalFailure;
    out.message = e.what();
    return out;
  }
}

}  // namespace clqr
