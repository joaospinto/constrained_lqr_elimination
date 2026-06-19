#include "clqr/clqr.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <exception>
#include <sstream>
#include <stdexcept>

namespace clqr {
namespace {

#if defined(__GNUC__) || defined(__clang__)
#define CLQR_RESTRICT __restrict__
#else
#define CLQR_RESTRICT
#endif

#if defined(__clang__)
#define CLQR_UNROLL _Pragma("clang loop unroll_count(2)")
#elif defined(__GNUC__)
#define CLQR_UNROLL _Pragma("GCC unroll 2")
#else
#define CLQR_UNROLL
#endif

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
  WorkspaceVector<Stage> stages;
  Matrix terminal_Q;
  Vector terminal_q;
  Matrix terminal_E;
  Vector terminal_e;
};

struct WorkingState {
  WorkingProblem problem;
  WorkspaceVector<StateMap> state_maps;
  WorkspaceVector<ControlMap> control_maps;
};

struct NewtonKktDiagnostics {
  bool singular = false;
  bool wrong_inertia = false;
  WorkspaceVector<std::string> messages;
};

struct Solution {
  SolveStatus status = SolveStatus::kInvalidInput;
  std::string message;
  WorkspaceVector<Vector> states;
  WorkspaceVector<Vector> controls;
  Vector initial_multiplier;
  WorkspaceVector<Vector> dynamics_multipliers;
  WorkspaceVector<Vector> mixed_multipliers;
  WorkspaceVector<Vector> state_multipliers;
  Vector terminal_state_multiplier;
  bool newton_kkt_singular = false;
  bool newton_kkt_wrong_inertia = false;
  std::string newton_kkt_diagnostic;
  double objective = 0.0;
};

struct RectangularSolve {
  Vector x;
  bool inconsistent = false;
  std::size_t rank = 0;
};

struct LinearParametrization {
  Matrix basis;
  Vector offset;
  bool inconsistent = false;
};

struct AffineSet {
  Matrix basis;
  Vector offset;
};

struct StageMultiplierMap {
  AffineSet future_y;
  Matrix local_basis;
  Vector local_offset;
  std::size_t future_parameter_size = 0;
  std::size_t mixed_size = 0;
  std::size_t state_size = 0;
};

struct AffineStateBasis {
  Matrix T;
  Vector offset;
  WorkspaceVector<std::size_t> free_rows;
  WorkspaceVector<std::size_t> pivot_rows;
  bool infeasible = false;
  bool redundant = false;
  std::string message;
};

void Check(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

std::size_t ControlLinearCols(const ControlMap& map, std::size_t identity_size) {
  return (map.control_linear.rows() == 0 && map.control_linear.cols() == 0)
             ? identity_size
             : map.control_linear.cols();
}

bool LazyIdentityStateMap(const StateMap& map) {
  return map.linear.rows() == 0 && map.linear.cols() == 0 && map.offset.size() == 0;
}

bool LazyIdentityControlLinear(const ControlMap& map) {
  return map.control_linear.rows() == 0 && map.control_linear.cols() == 0;
}

RectangularSolve SolveRectangularRref(const Matrix& a, const Vector& b, double tolerance) {
  Check(a.rows() == b.size(), "rectangular solve rhs shape mismatch");
  RectangularSolve out;
  out.x = Vector(a.cols());
  if (a.cols() == 0) {
    out.inconsistent = MaxAbs(b) > tolerance;
    return out;
  }
  Matrix augmented(a.rows(), a.cols() + 1);
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) augmented(i, j) = a(i, j);
    augmented(i, a.cols()) = b[i];
  }
  RrefResult rref = Rref(augmented, a.cols(), tolerance);
  out.rank = rref.pivot_columns.size();
  for (std::size_t i = 0; i < rref.pivot_columns.size(); ++i) {
    out.x[rref.pivot_columns[i]] = rref.matrix(rref.pivot_rows[i], a.cols());
  }
  for (std::size_t row = 0; row < rref.matrix.rows(); ++row) {
    bool zero_lhs = true;
    for (std::size_t col = 0; col < a.cols(); ++col) {
      if (!IsNearlyZero(rref.matrix(row, col), tolerance)) {
        zero_lhs = false;
        break;
      }
    }
    if (zero_lhs && !IsNearlyZero(rref.matrix(row, a.cols()), tolerance)) {
      out.inconsistent = true;
      break;
    }
  }
  return out;
}

RectangularSolve SolveMixedMultiplierRref(const Stage& stage, const Vector& x,
                                          const Vector& u, const Vector& y,
                                          double tolerance) {
  const std::size_t constraints = stage.C.rows();
  const std::size_t controls = stage.B.cols();
  RectangularSolve out;
  out.x = Vector(constraints);
  if (constraints == 0) return out;

  Matrix augmented(controls, constraints + 1);
  for (std::size_t control = 0; control < controls; ++control) {
    double value = -stage.r[control];
    CLQR_UNROLL
    for (std::size_t row = 0; row < stage.B.rows(); ++row) {
      value += stage.B(row, control) * y[row];
    }
    CLQR_UNROLL
    for (std::size_t row = 0; row < stage.M.rows(); ++row) {
      value -= stage.M(row, control) * x[row];
    }
    CLQR_UNROLL
    for (std::size_t col = 0; col < stage.R.cols(); ++col) {
      value -= stage.R(control, col) * u[col];
    }
    augmented(control, constraints) = value;
    for (std::size_t constraint = 0; constraint < constraints; ++constraint) {
      augmented(control, constraint) = stage.D(constraint, control);
    }
  }

  RrefResult rref = Rref(std::move(augmented), constraints, tolerance);
  out.rank = rref.pivot_columns.size();
  for (std::size_t i = 0; i < rref.pivot_columns.size(); ++i) {
    out.x[rref.pivot_columns[i]] = rref.matrix(rref.pivot_rows[i], constraints);
  }
  for (std::size_t row = 0; row < rref.matrix.rows(); ++row) {
    bool zero_lhs = true;
    for (std::size_t col = 0; col < constraints; ++col) {
      if (!IsNearlyZero(rref.matrix(row, col), tolerance)) {
        zero_lhs = false;
        break;
      }
    }
    if (zero_lhs && !IsNearlyZero(rref.matrix(row, constraints), tolerance)) {
      out.inconsistent = true;
      break;
    }
  }
  return out;
}

Vector Slice(const Vector& x, std::size_t offset, std::size_t size) {
  Vector out(size);
  for (std::size_t i = 0; i < size; ++i) out[i] = x[offset + i];
  return out;
}

bool Contains(const WorkspaceVector<std::size_t>& values, std::size_t needle) {
  for (std::size_t value : values) {
    if (value == needle) return true;
  }
  return false;
}

LinearParametrization ParametrizeLinearSystem(const Matrix& a, const Vector& b,
                                              double tolerance) {
  Check(a.rows() == b.size(), "linear parametrization rhs shape mismatch");
  LinearParametrization out;
  if (a.cols() == 0) {
    out.offset = Vector(0);
    out.basis = Matrix(0, 0);
    out.inconsistent = MaxAbs(b) > tolerance;
    return out;
  }
  Matrix augmented(a.rows(), a.cols() + 1);
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) augmented(i, j) = a(i, j);
    augmented(i, a.cols()) = b[i];
  }
  RrefResult rref = Rref(augmented, a.cols(), tolerance);
  WorkspaceVector<std::size_t> free_cols;
  free_cols.reserve(a.cols());
  for (std::size_t col = 0; col < a.cols(); ++col) {
    if (!Contains(rref.pivot_columns, col)) free_cols.push_back(col);
  }
  out.offset = Vector(a.cols());
  out.basis = Matrix(a.cols(), free_cols.size());
  for (std::size_t j = 0; j < free_cols.size(); ++j) out.basis(free_cols[j], j) = 1.0;
  for (std::size_t pivot_index = 0; pivot_index < rref.pivot_columns.size(); ++pivot_index) {
    const std::size_t pivot_col = rref.pivot_columns[pivot_index];
    const std::size_t row = rref.pivot_rows[pivot_index];
    out.offset[pivot_col] = rref.matrix(row, a.cols());
    for (std::size_t free_index = 0; free_index < free_cols.size(); ++free_index) {
      out.basis(pivot_col, free_index) = -rref.matrix(row, free_cols[free_index]);
    }
  }
  for (std::size_t row = 0; row < rref.matrix.rows(); ++row) {
    bool zero_lhs = true;
    for (std::size_t col = 0; col < a.cols(); ++col) {
      if (!IsNearlyZero(rref.matrix(row, col), tolerance)) {
        zero_lhs = false;
        break;
      }
    }
    if (zero_lhs && !IsNearlyZero(rref.matrix(row, a.cols()), tolerance)) {
      out.inconsistent = true;
      break;
    }
  }
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

void SymmetrizeInPlace(Matrix& a) {
  Check(a.rows() == a.cols(), "cannot symmetrize a rectangular matrix");
  for (std::size_t row = 0; row < a.rows(); ++row) {
    for (std::size_t col = row + 1; col < a.cols(); ++col) {
      const double value = 0.5 * (a(row, col) + a(col, row));
      a(row, col) = value;
      a(col, row) = value;
    }
  }
}

bool IsIdentity(const Matrix& a, double tolerance) {
  if (a.rows() != a.cols()) return false;
  for (std::size_t row = 0; row < a.rows(); ++row) {
    for (std::size_t col = 0; col < a.cols(); ++col) {
      const double expected = row == col ? 1.0 : 0.0;
      if (std::abs(a(row, col) - expected) > tolerance) return false;
    }
  }
  return true;
}

void AddDiagnostic(NewtonKktDiagnostics* diagnostics, const std::string& message) {
  if (diagnostics == nullptr) return;
  if (ActiveWorkspaceArena() != nullptr) return;
  diagnostics->messages.push_back(message);
}

void AddIndexedDiagnostic(NewtonKktDiagnostics* diagnostics, const char* prefix,
                          std::size_t index) {
  if (diagnostics == nullptr) return;
  if (ActiveWorkspaceArena() != nullptr) return;
  AddDiagnostic(diagnostics, std::string(prefix) + std::to_string(index));
}

std::string JoinMessages(const WorkspaceVector<std::string>& messages) {
  std::ostringstream out;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    if (i > 0) out << "; ";
    out << messages[i];
  }
  return out.str();
}

void AnalyzeReducedControlHessian(const Matrix& Huu, bool positive_definite,
                                  std::size_t stage, double tolerance,
                                  NewtonKktDiagnostics* diagnostics) {
  if (positive_definite) return;
  RrefResult rank = Rref(Huu, Huu.cols(), tolerance);
  if (rank.pivot_columns.size() < Huu.cols()) {
    diagnostics->singular = true;
    AddIndexedDiagnostic(diagnostics, "singular reduced control Hessian at stage ", stage);
    return;
  }
  diagnostics->wrong_inertia = true;
  AddIndexedDiagnostic(diagnostics, "reduced control Hessian has wrong inertia at stage ",
                       stage);
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

double Objective(const Problem& original, const WorkspaceVector<Vector>& xs,
                 const WorkspaceVector<Vector>& us) {
  double value = 0.0;
  for (std::size_t i = 0; i < original.stages.size(); ++i) {
    const Stage& s = original.stages[i];
    for (std::size_t row = 0; row < s.Q.rows(); ++row) {
      for (std::size_t col = 0; col < s.Q.cols(); ++col) {
        value += 0.5 * xs[i][row] * s.Q(row, col) * xs[i][col];
      }
    }
    for (std::size_t row = 0; row < s.R.rows(); ++row) {
      for (std::size_t col = 0; col < s.R.cols(); ++col) {
        value += 0.5 * us[i][row] * s.R(row, col) * us[i][col];
      }
    }
    for (std::size_t row = 0; row < s.M.rows(); ++row) {
      for (std::size_t col = 0; col < s.M.cols(); ++col) {
        value += xs[i][row] * s.M(row, col) * us[i][col];
      }
    }
    value += Dot(s.q, xs[i]);
    value += Dot(s.r, us[i]);
  }
  for (std::size_t row = 0; row < original.terminal_Q.rows(); ++row) {
    for (std::size_t col = 0; col < original.terminal_Q.cols(); ++col) {
      value += 0.5 * xs.back()[row] * original.terminal_Q(row, col) * xs.back()[col];
    }
  }
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

  const std::size_t num_states = problem.stages.size() + 1;
  state.state_maps.resize(num_states);
  state.control_maps.resize(problem.stages.size());
  for (std::size_t i = 0; i < problem.stages.size(); ++i) {
    const std::size_t n = problem.stages[i].A.cols();
    const std::size_t m = problem.stages[i].B.cols();
    state.control_maps[i].state_linear = Matrix(m, n);
    state.control_maps[i].offset = Vector(m);
  }
  return state;
}

LinearParametrization ReducedInitialState(const Problem& original, const StateMap& map,
                                          double tolerance) {
  if (LazyIdentityStateMap(map)) {
    LinearParametrization out;
    out.offset = original.initial_state;
    out.basis = Matrix(original.initial_state.size(), 0);
    return out;
  }
  if (IsIdentity(map.linear, tolerance) && MaxAbs(map.offset) <= tolerance) {
    LinearParametrization out;
    out.offset = original.initial_state;
    out.basis = Matrix(original.initial_state.size(), 0);
    return out;
  }
  return ParametrizeLinearSystem(map.linear, original.initial_state - map.offset, tolerance);
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
  Matrix augmented(E.rows(), n + 1);
  for (std::size_t row = 0; row < E.rows(); ++row) {
    for (std::size_t col = 0; col < n; ++col) augmented(row, col) = E(row, col);
    augmented(row, n) = e[row];
  }
  RrefResult rref = Rref(augmented, n, tolerance);
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
  out.redundant = rref.pivot_columns.size() < E.rows();

  out.free_rows.reserve(n);
  for (std::size_t col = 0; col < n; ++col) {
    if (!Contains(rref.pivot_columns, col)) out.free_rows.push_back(col);
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
  bool redundant = false;
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
  Matrix augmented(constraints, m + n + 1);
  for (std::size_t row = 0; row < constraints; ++row) {
    for (std::size_t col = 0; col < m; ++col) augmented(row, col) = D(row, col);
    for (std::size_t col = 0; col < n; ++col) augmented(row, m + col) = C(row, col);
    augmented(row, m + n) = d[row];
  }
  RrefResult rref = Rref(augmented, m, tolerance);
  WorkspaceVector<std::size_t> free_cols;
  free_cols.reserve(m);
  for (std::size_t col = 0; col < m; ++col) {
    if (!Contains(rref.pivot_columns, col)) free_cols.push_back(col);
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

  WorkspaceVector<std::size_t> residual_rows;
  residual_rows.reserve(constraints);
  for (std::size_t row = 0; row < constraints; ++row) {
    if (Contains(rref.pivot_rows, row)) continue;
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
    if (!any_state && !any_const) out.redundant = true;
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

bool EliminateMixedStageWithMaps(WorkingState& state, std::size_t i,
                                 double tolerance, std::string* error,
                                 NewtonKktDiagnostics* diagnostics) {
  Stage& s = state.problem.stages[i];
  if (s.C.rows() == 0) return false;
  MixedElimination basis = ControlBasis(s.C, s.D, s.d, tolerance);
  if (basis.infeasible) {
    *error = basis.message;
    throw std::runtime_error(*error);
  }
  if (basis.redundant) {
    diagnostics->singular = true;
    AddIndexedDiagnostic(diagnostics, "redundant mixed equality constraints at stage ", i);
  }
  const Matrix& old_Q = s.Q;
  const Matrix& old_R = s.R;
  const Matrix& old_M = s.M;
  const Matrix& old_A = s.A;
  const Matrix& old_B = s.B;
  const Vector& old_q = s.q;
  const Vector& old_r = s.r;
  const Vector& old_c = s.c;

  const std::size_t n = old_Q.rows();
  const std::size_t m = old_R.rows();
  const std::size_t next_n = old_B.rows();
  const std::size_t reduced_m = basis.Z.cols();
  Vector affine_control_gradient(m);
  for (std::size_t row = 0; row < m; ++row) {
    double value = 0.0;
    for (std::size_t col = 0; col < m; ++col) value += old_R(row, col) * basis.y[col];
    affine_control_gradient[row] = value + old_r[row];
  }

  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      double value = old_Q(row, col);
      for (std::size_t u = 0; u < m; ++u) {
        value += old_M(row, u) * basis.Y(u, col);
        value += old_M(col, u) * basis.Y(u, row);
        for (std::size_t v = 0; v < m; ++v) {
          value += basis.Y(u, row) * old_R(u, v) * basis.Y(v, col);
        }
      }
      s.Q(row, col) = value;
    }
  }
  SymmetrizeInPlace(s.Q);

  Matrix new_R(reduced_m, reduced_m);
  for (std::size_t row = 0; row < reduced_m; ++row) {
    for (std::size_t col = 0; col < reduced_m; ++col) {
      double value = 0.0;
      for (std::size_t u = 0; u < m; ++u) {
        for (std::size_t v = 0; v < m; ++v) {
          value += basis.Z(u, row) * old_R(u, v) * basis.Z(v, col);
        }
      }
      new_R(row, col) = value;
    }
  }
  SymmetrizeInPlace(new_R);

  Matrix new_M(n, reduced_m);
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t col = 0; col < reduced_m; ++col) {
      double value = 0.0;
      for (std::size_t u = 0; u < m; ++u) {
        value += old_M(row, u) * basis.Z(u, col);
        for (std::size_t v = 0; v < m; ++v) {
          value += basis.Y(u, row) * old_R(u, v) * basis.Z(v, col);
        }
      }
      new_M(row, col) = value;
    }
  }

  for (std::size_t row = 0; row < n; ++row) {
    double value = old_q[row];
    for (std::size_t u = 0; u < m; ++u) {
      value += basis.Y(u, row) * affine_control_gradient[u];
      value += old_M(row, u) * basis.y[u];
    }
    s.q[row] = value;
  }

  Vector new_r(reduced_m);
  for (std::size_t row = 0; row < reduced_m; ++row) {
    double value = 0.0;
    for (std::size_t u = 0; u < m; ++u) value += basis.Z(u, row) * affine_control_gradient[u];
    new_r[row] = value;
  }

  for (std::size_t row = 0; row < next_n; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      double value = old_A(row, col);
      for (std::size_t u = 0; u < m; ++u) value += old_B(row, u) * basis.Y(u, col);
      s.A(row, col) = value;
    }
  }
  Matrix new_B(next_n, reduced_m);
  for (std::size_t row = 0; row < next_n; ++row) {
    for (std::size_t col = 0; col < reduced_m; ++col) {
      double value = 0.0;
      for (std::size_t u = 0; u < m; ++u) value += old_B(row, u) * basis.Z(u, col);
      new_B(row, col) = value;
    }
  }
  for (std::size_t row = 0; row < next_n; ++row) {
    double value = old_c[row];
    for (std::size_t u = 0; u < m; ++u) value += old_B(row, u) * basis.y[u];
    s.c[row] = value;
  }
  s.R = std::move(new_R);
  s.M = std::move(new_M);
  s.r = std::move(new_r);
  s.B = std::move(new_B);
  s.C = Matrix(0, s.A.cols());
  s.D = Matrix(0, s.B.cols());
  s.d = Vector(0);
  AppendStateConstraints(s, basis.state_C, basis.state_d);

  const ControlMap& old_map = state.control_maps[i];
  const std::size_t old_control_cols = ControlLinearCols(old_map, m);
  Matrix new_state_linear(old_map.state_linear.rows(), basis.Y.cols());
  if (LazyIdentityControlLinear(old_map)) {
    for (std::size_t row = 0; row < old_map.state_linear.rows(); ++row) {
      for (std::size_t col = 0; col < basis.Y.cols(); ++col) {
        new_state_linear(row, col) = old_map.state_linear(row, col) + basis.Y(row, col);
      }
    }
  } else {
    for (std::size_t row = 0; row < old_map.state_linear.rows(); ++row) {
      for (std::size_t col = 0; col < basis.Y.cols(); ++col) {
        double value = old_map.state_linear(row, col);
        for (std::size_t u = 0; u < old_control_cols; ++u) {
          value += old_map.control_linear(row, u) * basis.Y(u, col);
        }
        new_state_linear(row, col) = value;
      }
    }
  }

  Matrix new_control_linear(old_map.offset.size(), basis.Z.cols());
  if (LazyIdentityControlLinear(old_map)) {
    for (std::size_t row = 0; row < old_map.offset.size(); ++row) {
      for (std::size_t col = 0; col < basis.Z.cols(); ++col) {
        new_control_linear(row, col) = basis.Z(row, col);
      }
    }
  } else {
    for (std::size_t row = 0; row < old_map.offset.size(); ++row) {
      for (std::size_t col = 0; col < basis.Z.cols(); ++col) {
        double value = 0.0;
        for (std::size_t u = 0; u < old_control_cols; ++u) {
          value += old_map.control_linear(row, u) * basis.Z(u, col);
        }
        new_control_linear(row, col) = value;
      }
    }
  }

  Vector new_offset(old_map.offset.size());
  if (LazyIdentityControlLinear(old_map)) {
    for (std::size_t row = 0; row < old_map.offset.size(); ++row) {
      new_offset[row] = old_map.offset[row] + basis.y[row];
    }
  } else {
    for (std::size_t row = 0; row < old_map.offset.size(); ++row) {
      double value = old_map.offset[row];
      for (std::size_t u = 0; u < old_control_cols; ++u) {
        value += old_map.control_linear(row, u) * basis.y[u];
      }
      new_offset[row] = value;
    }
  }
  state.control_maps[i].state_linear = std::move(new_state_linear);
  state.control_maps[i].control_linear = std::move(new_control_linear);
  state.control_maps[i].offset = std::move(new_offset);
  return true;
}

bool StateBasisIsIdentity(const AffineStateBasis& basis, double tolerance) {
  return basis.pivot_rows.empty() && IsIdentity(basis.T, tolerance) &&
         MaxAbs(basis.offset) <= tolerance;
}

void ApplyTerminalStateBasis(WorkingState& state, const AffineStateBasis& basis,
                             double tolerance) {
  if (StateBasisIsIdentity(basis, tolerance)) {
    state.problem.terminal_E = Matrix(0, state.problem.terminal_Q.rows());
    state.problem.terminal_e = Vector(0);
    return;
  }
  const Matrix old_terminal_Q = state.problem.terminal_Q;
  const Vector old_terminal_q = state.problem.terminal_q;
  state.problem.terminal_Q =
      Symmetrize(Transpose(basis.T) * old_terminal_Q * basis.T);
  state.problem.terminal_q =
      Transpose(basis.T) * (old_terminal_q + old_terminal_Q * basis.offset);
  state.problem.terminal_E = Matrix(0, basis.T.cols());
  state.problem.terminal_e = Vector(0);

  StateMap old_terminal_map = state.state_maps.back();
  if (LazyIdentityStateMap(old_terminal_map)) {
    state.state_maps.back().linear = basis.T;
    state.state_maps.back().offset = basis.offset;
  } else {
    state.state_maps.back().linear = old_terminal_map.linear * basis.T;
    state.state_maps.back().offset =
        old_terminal_map.offset + old_terminal_map.linear * basis.offset;
  }
}

void ApplyNextStateBasisToStage(WorkingState& state, std::size_t i,
                                const AffineStateBasis& next,
                                double tolerance) {
  if (StateBasisIsIdentity(next, tolerance)) return;
  Stage& s = state.problem.stages[i];
  const Matrix old_A = s.A;
  const Matrix old_B = s.B;
  const Vector old_c = s.c;

  Matrix new_A = Rows(old_A, next.free_rows);
  Matrix new_B = Rows(old_B, next.free_rows);
  Vector new_c = Entries(old_c, next.free_rows);

  Matrix C_extra(next.pivot_rows.size(), old_A.cols());
  Matrix D_extra(next.pivot_rows.size(), old_B.cols());
  Vector d_extra(next.pivot_rows.size());
  for (std::size_t row = 0; row < next.pivot_rows.size(); ++row) {
    const std::size_t pivot = next.pivot_rows[row];
    for (std::size_t col = 0; col < old_A.cols(); ++col) {
      double value = old_A(pivot, col);
      for (std::size_t f = 0; f < next.free_rows.size(); ++f) {
        value -= next.T(pivot, f) * old_A(next.free_rows[f], col);
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
    double value = old_c[pivot] - next.offset[pivot];
    for (std::size_t f = 0; f < next.free_rows.size(); ++f) {
      value -= next.T(pivot, f) * old_c[next.free_rows[f]];
    }
    d_extra[row] = value;
  }

  s.A = new_A;
  s.B = new_B;
  s.c = new_c;
  AppendMixedConstraints(s, C_extra, D_extra, d_extra);
}

void ApplyCurrentStateBasisToStage(WorkingState& state, std::size_t i,
                                   const AffineStateBasis& cur,
                                   double tolerance) {
  Stage& s = state.problem.stages[i];
  if (StateBasisIsIdentity(cur, tolerance)) {
    s.E = Matrix(0, s.A.cols());
    s.e = Vector(0);
    return;
  }
  const Matrix old_A = s.A;
  const Matrix old_Q = s.Q;
  const Matrix old_M = s.M;
  const Vector old_q = s.q;

  s.Q = Symmetrize(Transpose(cur.T) * old_Q * cur.T);
  s.M = Transpose(cur.T) * old_M;
  s.q = Transpose(cur.T) * (old_q + old_Q * cur.offset);
  s.r = s.r + Transpose(old_M) * cur.offset;
  s.A = old_A * cur.T;
  s.c = old_A * cur.offset + s.c;
  s.C = Matrix(0, cur.T.cols());
  s.D = Matrix(0, s.B.cols());
  s.d = Vector(0);
  s.E = Matrix(0, cur.T.cols());
  s.e = Vector(0);

  StateMap old_state_map = state.state_maps[i];
  if (LazyIdentityStateMap(old_state_map)) {
    state.state_maps[i].linear = cur.T;
    state.state_maps[i].offset = cur.offset;
  } else {
    state.state_maps[i].linear = old_state_map.linear * cur.T;
    state.state_maps[i].offset =
        old_state_map.offset + old_state_map.linear * cur.offset;
  }

  ControlMap old_control_map = state.control_maps[i];
  state.control_maps[i].state_linear = old_control_map.state_linear * cur.T;
  state.control_maps[i].offset =
      old_control_map.offset + old_control_map.state_linear * cur.offset;
}

void CheckStateBasis(const AffineStateBasis& basis, std::size_t node,
                     std::string* error, NewtonKktDiagnostics* diagnostics) {
  if (basis.infeasible) {
    *error = basis.message;
    throw std::runtime_error(*error);
  }
  if (basis.redundant) {
    diagnostics->singular = true;
    AddIndexedDiagnostic(diagnostics, "redundant state equality constraints at node ", node);
  }
}

void EliminateConstraintsRightToLeft(WorkingState& state, double tolerance,
                                     std::string* error,
                                     NewtonKktDiagnostics* diagnostics) {
  const std::size_t N = state.problem.stages.size();
  AffineStateBasis next;
  bool next_is_identity = true;
  if (state.problem.terminal_E.rows() > 0) {
    next = StateBasis(state.problem.terminal_E, state.problem.terminal_e,
                      state.problem.terminal_Q.rows(), tolerance);
    CheckStateBasis(next, N, error, diagnostics);
    ApplyTerminalStateBasis(state, next, tolerance);
    next_is_identity = StateBasisIsIdentity(next, tolerance);
  }

  for (std::size_t rev = 0; rev < N; ++rev) {
    const std::size_t i = N - 1 - rev;
    if (!next_is_identity) ApplyNextStateBasisToStage(state, i, next, tolerance);
    EliminateMixedStageWithMaps(state, i, tolerance, error, diagnostics);

    Stage& s = state.problem.stages[i];
    if (s.E.rows() == 0) {
      next = AffineStateBasis{};
      next_is_identity = true;
      continue;
    }
    AffineStateBasis cur = StateBasis(s.E, s.e, s.A.cols(), tolerance);
    CheckStateBasis(cur, i, error, diagnostics);
    ApplyCurrentStateBasisToStage(state, i, cur, tolerance);
    next_is_identity = StateBasisIsIdentity(cur, tolerance);
    next = std::move(cur);
  }
}

bool AnyOriginalConstraints(const Problem& problem) {
  if (problem.terminal_E.rows() > 0) return true;
  for (const Stage& stage : problem.stages) {
    if (stage.C.rows() > 0 || stage.E.rows() > 0) return true;
  }
  return false;
}

bool AnyOriginalStateConstraints(const Problem& problem) {
  if (problem.terminal_E.rows() > 0) return true;
  for (const Stage& stage : problem.stages) {
    if (stage.E.rows() > 0) return true;
  }
  return false;
}

bool StateMapsAreIdentity(const WorkingState& state, double tolerance) {
  for (const StateMap& map : state.state_maps) {
    if (LazyIdentityStateMap(map)) continue;
    if (!IsIdentity(map.linear, tolerance) || MaxAbs(map.offset) > tolerance) return false;
  }
  return true;
}

struct ReducedSolution {
  WorkspaceVector<Vector> x;
  WorkspaceVector<Vector> u;
};

struct RiccatiWorkspace {
  WorkspaceVector<unsigned char> storage;

  std::size_t* state_dim = nullptr;
  std::size_t* control_dim = nullptr;
  std::size_t* P_offset = nullptr;
  std::size_t* p_offset = nullptr;
  std::size_t* K_offset = nullptr;
  std::size_t* k_offset = nullptr;

  double* P = nullptr;
  double* p = nullptr;
  double* K = nullptr;
  double* k = nullptr;

  double* pc = nullptr;
  double* Hxx = nullptr;
  double* Huu = nullptr;
  double* Hxu = nullptr;
  double* A_T_P = nullptr;
  double* B_T_P = nullptr;
  double* hx = nullptr;
  double* hu = nullptr;
  double* lower = nullptr;
  double* solve_hxu = nullptr;
  double* solve_hu = nullptr;
  Matrix fallback_Huu;
  Matrix fallback_Hxu;
  Vector fallback_hu;

  struct Sizes {
    std::size_t max_state = 0;
    std::size_t max_control = 0;
    std::size_t total_P = 0;
    std::size_t total_p = 0;
    std::size_t total_K = 0;
    std::size_t total_k = 0;
    std::size_t bytes = 0;
  };

  static std::size_t Align(std::size_t offset, std::size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
  }

  template <typename T>
  static T* Slice(unsigned char* base, std::size_t* offset, std::size_t count) {
    *offset = Align(*offset, alignof(T));
    T* out = base == nullptr ? nullptr : reinterpret_cast<T*>(base + *offset);
    *offset += count * sizeof(T);
    return out;
  }

  static Sizes ComputeSizes(const WorkspaceVector<Stage>& stages) {
    const std::size_t N = stages.size();
    Sizes sizes;
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t n = stages[i].A.cols();
      const std::size_t m = stages[i].B.cols();
      sizes.max_state = std::max(sizes.max_state, n);
      sizes.max_control = std::max(sizes.max_control, m);
      sizes.total_P += n * n;
      sizes.total_p += n;
      sizes.total_K += m * n;
      sizes.total_k += m;
    }
    std::size_t terminal_state = 0;
    if (N > 0) {
      terminal_state = stages.back().A.rows();
    }
    sizes.max_state = std::max(sizes.max_state, terminal_state);
    sizes.total_P += terminal_state * terminal_state;
    sizes.total_p += terminal_state;

    Layout(nullptr, stages, sizes.total_P, sizes.total_p, sizes.total_K, sizes.total_k,
           sizes.max_state, sizes.max_control, nullptr, &sizes.bytes);
    return sizes;
  }

  static std::size_t RequiredBytes(const WorkspaceVector<Stage>& stages) {
    return ComputeSizes(stages).bytes;
  }

  void Reserve(const WorkspaceVector<Stage>& stages) {
    const Sizes sizes = ComputeSizes(stages);
    storage.assign(sizes.bytes, 0);
    Assign(stages, storage.data(), sizes.bytes);
  }

  void Assign(const WorkspaceVector<Stage>& stages, unsigned char* data,
              std::size_t bytes) {
    const Sizes sizes = ComputeSizes(stages);
    if (bytes < sizes.bytes) {
      throw std::invalid_argument("workspace is too small for Riccati solve");
    }
    std::size_t offset = 0;
    Layout(this, stages, sizes.total_P, sizes.total_p, sizes.total_K, sizes.total_k,
           sizes.max_state, sizes.max_control, data, &offset);

    std::size_t total_P = 0;
    std::size_t total_p = 0;
    std::size_t total_K = 0;
    std::size_t total_k = 0;
    const std::size_t N = stages.size();
    for (std::size_t i = 0; i < N; ++i) {
      state_dim[i] = stages[i].A.cols();
      control_dim[i] = stages[i].B.cols();
      P_offset[i] = total_P;
      p_offset[i] = total_p;
      K_offset[i] = total_K;
      k_offset[i] = total_k;
      total_P += state_dim[i] * state_dim[i];
      total_p += state_dim[i];
      total_K += control_dim[i] * state_dim[i];
      total_k += control_dim[i];
    }
    if (N > 0) state_dim[N] = stages.back().A.rows();
    P_offset[N] = total_P;
    p_offset[N] = total_p;
  }

  static void Layout(RiccatiWorkspace* workspace, const WorkspaceVector<Stage>& stages, std::size_t total_P,
                     std::size_t total_p, std::size_t total_K, std::size_t total_k,
                     std::size_t max_state, std::size_t max_control,
                     unsigned char* base, std::size_t* offset) {
    const std::size_t N = stages.size();
    std::size_t* state_dim = Slice<std::size_t>(base, offset, N + 1);
    std::size_t* control_dim = Slice<std::size_t>(base, offset, N);
    std::size_t* P_offset = Slice<std::size_t>(base, offset, N + 1);
    std::size_t* p_offset = Slice<std::size_t>(base, offset, N + 1);
    std::size_t* K_offset = Slice<std::size_t>(base, offset, N);
    std::size_t* k_offset = Slice<std::size_t>(base, offset, N);
    double* P = Slice<double>(base, offset, total_P);
    double* p = Slice<double>(base, offset, total_p);
    double* K = Slice<double>(base, offset, total_K);
    double* k = Slice<double>(base, offset, total_k);
    double* pc = Slice<double>(base, offset, max_state);
    double* Hxx = Slice<double>(base, offset, max_state * max_state);
    double* Huu = Slice<double>(base, offset, max_control * max_control);
    double* Hxu = Slice<double>(base, offset, max_state * max_control);
    double* A_T_P = Slice<double>(base, offset, max_state * max_state);
    double* B_T_P = Slice<double>(base, offset, max_control * max_state);
    double* hx = Slice<double>(base, offset, max_state);
    double* hu = Slice<double>(base, offset, max_control);
    double* lower = Slice<double>(base, offset, max_control * max_control);
    double* solve_hxu = Slice<double>(base, offset, max_control * max_state);
    double* solve_hu = Slice<double>(base, offset, max_control);
    if (workspace == nullptr) return;
    workspace->state_dim = state_dim;
    workspace->control_dim = control_dim;
    workspace->P_offset = P_offset;
    workspace->p_offset = p_offset;
    workspace->K_offset = K_offset;
    workspace->k_offset = k_offset;
    workspace->P = P;
    workspace->p = p;
    workspace->K = K;
    workspace->k = k;
    workspace->pc = pc;
    workspace->Hxx = Hxx;
    workspace->Huu = Huu;
    workspace->Hxu = Hxu;
    workspace->A_T_P = A_T_P;
    workspace->B_T_P = B_T_P;
    workspace->hx = hx;
    workspace->hu = hu;
    workspace->lower = lower;
    workspace->solve_hxu = solve_hxu;
    workspace->solve_hu = solve_hu;
  }

  double* PPtr(std::size_t i) { return P + P_offset[i]; }
  const double* PPtr(std::size_t i) const { return P + P_offset[i]; }
  double* pPtr(std::size_t i) { return p + p_offset[i]; }
  const double* pPtr(std::size_t i) const { return p + p_offset[i]; }
  double* KPtr(std::size_t i) { return K + K_offset[i]; }
  const double* KPtr(std::size_t i) const { return K + K_offset[i]; }
  double* kPtr(std::size_t i) { return k + k_offset[i]; }
  const double* kPtr(std::size_t i) const { return k + k_offset[i]; }
};

bool CholeskyFactorizeRaw(const double* CLQR_RESTRICT a, std::size_t n,
                          double tolerance, double* CLQR_RESTRICT lower) {
  for (std::size_t j = 0; j < n; ++j) {
    double diagonal = a[j * n + j];
    CLQR_UNROLL
    for (std::size_t k = 0; k < j; ++k) diagonal -= lower[j * n + k] * lower[j * n + k];
    if (diagonal <= tolerance) return false;
    lower[j * n + j] = std::sqrt(diagonal);
    for (std::size_t i = j + 1; i < n; ++i) {
      double value = a[i * n + j];
      CLQR_UNROLL
      for (std::size_t k = 0; k < j; ++k) value -= lower[i * n + k] * lower[j * n + k];
      lower[i * n + j] = value / lower[j * n + j];
    }
  }
  return true;
}

void MirrorLowerTriangleRaw(double* CLQR_RESTRICT a, std::size_t n) {
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t col = 0; col < row; ++col) {
      a[col * n + row] = a[row * n + col];
    }
  }
}

void SolveWithCholeskyRaw(const double* CLQR_RESTRICT lower,
                          const double* CLQR_RESTRICT Hxu,
                          const double* CLQR_RESTRICT hu, std::size_t n,
                          std::size_t m, double* CLQR_RESTRICT solve_hxu,
                          double* CLQR_RESTRICT solve_hu) {
  for (std::size_t control = 0; control < m; ++control) {
    CLQR_UNROLL
    for (std::size_t state = 0; state < n; ++state) {
      solve_hxu[control * n + state] = Hxu[state * m + control];
    }
    solve_hu[control] = hu[control];
  }
  for (std::size_t row = 0; row < m; ++row) {
    CLQR_UNROLL
    for (std::size_t col = 0; col < n; ++col) {
      double value = solve_hxu[row * n + col];
      CLQR_UNROLL
      for (std::size_t k = 0; k < row; ++k) {
        value -= lower[row * m + k] * solve_hxu[k * n + col];
      }
      solve_hxu[row * n + col] = value / lower[row * m + row];
    }
    double value = solve_hu[row];
    CLQR_UNROLL
    for (std::size_t k = 0; k < row; ++k) {
      value -= lower[row * m + k] * solve_hu[k];
    }
    solve_hu[row] = value / lower[row * m + row];
  }
  for (std::size_t rev = 0; rev < m; ++rev) {
    const std::size_t row = m - 1 - rev;
    CLQR_UNROLL
    for (std::size_t col = 0; col < n; ++col) {
      double value = solve_hxu[row * n + col];
      CLQR_UNROLL
      for (std::size_t k = row + 1; k < m; ++k) {
        value -= lower[k * m + row] * solve_hxu[k * n + col];
      }
      solve_hxu[row * n + col] = value / lower[row * m + row];
    }
    double value = solve_hu[row];
    CLQR_UNROLL
    for (std::size_t k = row + 1; k < m; ++k) {
      value -= lower[k * m + row] * solve_hu[k];
    }
    solve_hu[row] = value / lower[row * m + row];
  }
}

void CopyRawToMatrix(const double* data, std::size_t rows, std::size_t cols,
                     Matrix* out) {
  out->resize(rows, cols);
  for (std::size_t i = 0; i < rows * cols; ++i) out->data()[i] = data[i];
}

void CopyRawToVector(const double* data, std::size_t size, Vector* out) {
  out->resize(size);
  for (std::size_t i = 0; i < size; ++i) (*out)[i] = data[i];
}

template <typename T>
T* WorkspaceSlice(unsigned char* base, std::size_t* offset, std::size_t count) {
  *offset = RiccatiWorkspace::Align(*offset, alignof(T));
  T* out = base == nullptr ? nullptr : reinterpret_cast<T*>(base + *offset);
  *offset += count * sizeof(T);
  return out;
}

struct SolutionWorkspaceLayout {
  RiccatiWorkspace riccati;
  SolutionView view;
  double* state_data = nullptr;
  double* control_data = nullptr;
  double* initial_multiplier_data = nullptr;
  double* dynamics_multiplier_data = nullptr;
  double* mixed_multiplier_data = nullptr;
  double* state_multiplier_data = nullptr;
  double* terminal_state_multiplier_data = nullptr;
};

std::size_t TotalStateScalars(const Problem& problem) {
  if (problem.stages.empty()) return problem.initial_state.size();
  std::size_t total = problem.stages.front().A.cols();
  for (const Stage& stage : problem.stages) total += stage.A.rows();
  return total;
}

std::size_t TotalControlScalars(const Problem& problem) {
  std::size_t total = 0;
  for (const Stage& stage : problem.stages) total += stage.B.cols();
  return total;
}

std::size_t TotalDynamicsMultiplierScalars(const Problem& problem) {
  std::size_t total = 0;
  for (const Stage& stage : problem.stages) total += stage.A.rows();
  return total;
}

std::size_t TotalMixedMultiplierScalars(const Problem& problem) {
  std::size_t total = 0;
  for (const Stage& stage : problem.stages) total += stage.C.rows();
  return total;
}

std::size_t TotalStateMultiplierScalars(const Problem& problem) {
  std::size_t total = 0;
  for (const Stage& stage : problem.stages) total += stage.E.rows();
  return total;
}

void AddWorkspaceBytes(std::size_t* offset, std::size_t alignment, std::size_t bytes) {
  *offset = (*offset + alignment - 1) & ~(alignment - 1);
  *offset += bytes;
}

template <typename T>
void AddObjects(std::size_t* offset, std::size_t count) {
  AddWorkspaceBytes(offset, alignof(T), sizeof(T) * count);
}

void AddDoubles(std::size_t* offset, std::size_t count) {
  AddObjects<double>(offset, count);
}

void AddIndices(std::size_t* offset, std::size_t count) {
  AddObjects<std::size_t>(offset, count);
}

std::size_t MatrixScalars(const Matrix& matrix) { return matrix.rows() * matrix.cols(); }

std::size_t StageStorageScalars(const Stage& stage) {
  return MatrixScalars(stage.A) + MatrixScalars(stage.B) + stage.c.size() +
         MatrixScalars(stage.Q) + MatrixScalars(stage.R) + MatrixScalars(stage.M) +
         stage.q.size() + stage.r.size() + MatrixScalars(stage.C) +
         MatrixScalars(stage.D) + stage.d.size() + MatrixScalars(stage.E) +
         stage.e.size();
}

struct WorkspaceDimensionSummary {
  std::size_t stages = 0;
  std::size_t max_state = 0;
  std::size_t max_next_state = 0;
  std::size_t max_control = 0;
  std::size_t max_mixed_rows = 0;
  std::size_t max_state_rows = 0;
  std::size_t max_terminal_rows = 0;
  std::size_t total_state = 0;
  std::size_t total_control = 0;
  std::size_t total_dynamics = 0;
  std::size_t total_mixed = 0;
  std::size_t total_state_rows = 0;
  std::size_t terminal_state = 0;
};

WorkspaceDimensionSummary SummarizeWorkspaceDimensions(const Problem& problem) {
  WorkspaceDimensionSummary dims;
  dims.stages = problem.stages.size();
  dims.terminal_state = problem.terminal_Q.rows();
  dims.max_terminal_rows = problem.terminal_E.rows();
  dims.max_state = std::max(problem.initial_state.size(), problem.terminal_Q.rows());
  dims.max_next_state = problem.terminal_Q.rows();
  dims.total_state = TotalStateScalars(problem);
  dims.total_control = TotalControlScalars(problem);
  dims.total_dynamics = TotalDynamicsMultiplierScalars(problem);
  dims.total_mixed = TotalMixedMultiplierScalars(problem);
  dims.total_state_rows = TotalStateMultiplierScalars(problem);
  for (const Stage& stage : problem.stages) {
    dims.max_state = std::max(dims.max_state, stage.A.cols());
    dims.max_next_state = std::max(dims.max_next_state, stage.A.rows());
    dims.max_control = std::max(dims.max_control, stage.B.cols());
    dims.max_mixed_rows = std::max(dims.max_mixed_rows, stage.C.rows());
    dims.max_state_rows = std::max(dims.max_state_rows, stage.E.rows());
  }
  return dims;
}

void AddSolutionStorageBound(const Problem& problem, std::size_t* offset) {
  const std::size_t N = problem.stages.size();
  AddObjects<Vector>(offset, N + 1);
  AddObjects<Vector>(offset, N);
  AddObjects<Vector>(offset, N);
  AddObjects<Vector>(offset, N);
  AddObjects<Vector>(offset, N);
  AddDoubles(offset, TotalStateScalars(problem));
  AddDoubles(offset, TotalControlScalars(problem));
  AddDoubles(offset, TotalDynamicsMultiplierScalars(problem));
  AddDoubles(offset, TotalMixedMultiplierScalars(problem));
  AddDoubles(offset, TotalStateMultiplierScalars(problem));
  AddDoubles(offset, problem.terminal_E.rows());
  AddDoubles(offset, problem.initial_state.size());
}

void AddSolutionViewStorageBound(const Problem& problem, std::size_t* offset) {
  const std::size_t N = problem.stages.size();
  AddObjects<VectorView>(offset, N + 1);
  AddObjects<VectorView>(offset, N);
  AddObjects<VectorView>(offset, N);
  AddObjects<VectorView>(offset, N);
  AddObjects<VectorView>(offset, N);
}

void AddWorkingProblemStorageBound(const Problem& problem, std::size_t* offset) {
  const std::size_t N = problem.stages.size();
  AddObjects<Stage>(offset, N);
  for (const Stage& stage : problem.stages) AddDoubles(offset, StageStorageScalars(stage));
  AddDoubles(offset, MatrixScalars(problem.terminal_Q) + problem.terminal_q.size() +
                           MatrixScalars(problem.terminal_E) + problem.terminal_e.size());

  AddObjects<StateMap>(offset, N + 1);
  for (std::size_t i = 0; i <= N; ++i) {
    const std::size_t n =
        (i == N) ? problem.terminal_Q.rows() : problem.stages[i].A.cols();
    AddDoubles(offset, n * n + n);
  }
  AddObjects<ControlMap>(offset, N);
  for (const Stage& stage : problem.stages) {
    const std::size_t n = stage.A.cols();
    const std::size_t m = stage.B.cols();
    AddDoubles(offset, m * n + m * m + m);
  }
}

std::size_t MixedEliminationStageDoubles(const WorkspaceDimensionSummary& dims,
                                         std::size_t mixed_rows_bound,
                                         std::size_t state_rows_bound) {
  const std::size_t n = dims.max_state;
  const std::size_t next_n = dims.max_next_state;
  const std::size_t m = dims.max_control;
  const std::size_t p = std::max<std::size_t>(1, mixed_rows_bound);
  const std::size_t f = m;
  std::size_t scalars = 0;
  scalars += 2 * p * (m + n + 1);       // augmented RREF input and by-value RREF copy
  scalars += m * n + m * f + m;         // Y, Z, y
  scalars += p * n + p;                 // residual state constraints
  scalars += n * n + m * m + n * m + next_n * m + n + m + next_n;  // old copies
  scalars += 8 * n * n + 4 * m * m + 6 * n * m + 2 * next_n * n +
             2 * next_n * m + 3 * next_n + 4 * n + 4 * m;          // temporaries
  scalars += (state_rows_bound + p) * n + state_rows_bound + p;    // appended E/e
  scalars += 2 * (m * n) + 2 * (m * m) + 3 * m;                    // control maps
  return scalars;
}

std::size_t StateEliminationStageDoubles(const WorkspaceDimensionSummary& dims,
                                         std::size_t state_rows_bound,
                                         std::size_t state_pivot_bound) {
  const std::size_t n = dims.max_state;
  const std::size_t next_n = dims.max_next_state;
  const std::size_t m = dims.max_control;
  const std::size_t p = std::max<std::size_t>(1, state_rows_bound);
  const std::size_t pivots = std::max<std::size_t>(1, state_pivot_bound);
  std::size_t scalars = 0;
  scalars += 2 * p * (n + 1) + n * n + n;                          // basis RREF, T, offset
  scalars += 3 * next_n * n + 3 * next_n * m + 4 * next_n;          // dynamics and extras
  scalars += 8 * n * n + 4 * n * m + 2 * m * n + 4 * n + 2 * m;     // cost temporaries
  scalars += (pivots + dims.max_mixed_rows) * (n + m + 1);          // appended C/D/d
  scalars += 3 * n * n + 2 * n + 2 * m * n + m * m + m;             // state/control maps
  return scalars;
}

void AddEliminationStorageBound(const Problem& problem, const SolveOptions& options,
                                const WorkspaceDimensionSummary& dims,
                                std::size_t* offset) {
  (void)problem;
  (void)options;
  const std::size_t mixed_rows_bound = dims.max_mixed_rows + dims.max_next_state;
  const std::size_t state_rows_bound =
      std::max(dims.max_terminal_rows, dims.max_state_rows + mixed_rows_bound);
  const std::size_t state_pivot_bound = std::min(dims.max_state, state_rows_bound);
  const std::size_t mixed_stage_doubles =
      MixedEliminationStageDoubles(dims, mixed_rows_bound, state_rows_bound);
  const std::size_t state_stage_doubles =
      StateEliminationStageDoubles(dims, state_rows_bound, state_pivot_bound);
  AddObjects<AffineStateBasis>(offset, dims.stages + 1);
  AddDoubles(offset, (dims.stages + 1) *
                         (2 * state_rows_bound * (dims.max_state + 1) +
                          dims.max_state * dims.max_state + dims.max_state));
  AddIndices(offset, (dims.stages + 1) * 3 * dims.max_state);
  AddDoubles(offset, dims.stages * state_stage_doubles);
  AddDoubles(offset, dims.stages * mixed_stage_doubles);
  AddIndices(offset, dims.stages *
                         (3 * dims.max_control + mixed_rows_bound +
                          3 * dims.max_state));
}

void AddRecoveryStorageBound(const Problem& problem, const WorkspaceDimensionSummary& dims,
                             std::size_t* offset) {
  const std::size_t N = problem.stages.size();
  const std::size_t total_constraints =
      TotalMixedMultiplierScalars(problem) + TotalStateMultiplierScalars(problem) +
      problem.terminal_E.rows();
  const std::size_t parameter_bound =
      std::max<std::size_t>(1, total_constraints + dims.max_mixed_rows +
                                   dims.max_state_rows + dims.max_terminal_rows);
  AddObjects<StageMultiplierMap>(offset, N);
  AddDoubles(offset, dims.max_state * parameter_bound + dims.max_state);
  for (const Stage& stage : problem.stages) {
    const std::size_t n = stage.A.cols();
    const std::size_t m = stage.B.cols();
    const std::size_t cols = parameter_bound + stage.C.rows() + stage.E.rows();
    AddDoubles(offset, m * cols + m);                    // local system and rhs
    AddDoubles(offset, 2 * m * (cols + 1));              // RREF input/copy
    AddIndices(offset, 3 * std::max(m, cols));
    AddDoubles(offset, cols * cols + cols);              // local parametrization
    AddDoubles(offset, n * cols + n + cols);             // previous/local parameters
    AddDoubles(offset, stage.A.rows() + stage.C.rows() + stage.E.rows());
  }
  AddSolutionStorageBound(problem, offset);
}

std::size_t ConstrainedWorkspaceRequiredBytes(const Problem& problem,
                                              const SolveOptions& options) {
  const WorkspaceDimensionSummary dims = SummarizeWorkspaceDimensions(problem);
  std::size_t offset = 0;
  AddWorkingProblemStorageBound(problem, &offset);
  AddEliminationStorageBound(problem, options, dims, &offset);
  offset += RiccatiWorkspace::RequiredBytes(problem.stages);
  AddSolutionStorageBound(problem, &offset);
  AddRecoveryStorageBound(problem, dims, &offset);
  AddSolutionViewStorageBound(problem, &offset);
  return offset;
}

std::size_t WorkspaceRequiredBytesInternal(const Problem& problem,
                                           const SolveOptions& options = SolveOptions{}) {
  if (AnyOriginalConstraints(problem)) {
    return ConstrainedWorkspaceRequiredBytes(problem, options);
  }
  std::size_t offset = 0;
  offset = RiccatiWorkspace::RequiredBytes(problem.stages);
  const std::size_t N = problem.stages.size();
  WorkspaceSlice<VectorView>(nullptr, &offset, N + 1);
  WorkspaceSlice<VectorView>(nullptr, &offset, N);
  WorkspaceSlice<VectorView>(nullptr, &offset, N);
  WorkspaceSlice<VectorView>(nullptr, &offset, N);
  WorkspaceSlice<VectorView>(nullptr, &offset, N);
  WorkspaceSlice<double>(nullptr, &offset, TotalStateScalars(problem));
  WorkspaceSlice<double>(nullptr, &offset, TotalControlScalars(problem));
  WorkspaceSlice<double>(nullptr, &offset, problem.initial_state.size());
  WorkspaceSlice<double>(nullptr, &offset, TotalDynamicsMultiplierScalars(problem));
  WorkspaceSlice<double>(nullptr, &offset, TotalMixedMultiplierScalars(problem));
  WorkspaceSlice<double>(nullptr, &offset, TotalStateMultiplierScalars(problem));
  WorkspaceSlice<double>(nullptr, &offset, problem.terminal_E.rows());
  return offset;
}

SolutionWorkspaceLayout BindSolutionWorkspace(const Problem& problem, Workspace& workspace) {
  const std::size_t required = WorkspaceRequiredBytesInternal(problem);
  if (workspace.size() < required) {
    throw std::invalid_argument("workspace is too small");
  }
  const std::size_t riccati_bytes = RiccatiWorkspace::RequiredBytes(problem.stages);
  SolutionWorkspaceLayout layout;
  layout.riccati.Assign(problem.stages, workspace.data(), riccati_bytes);

  std::size_t offset = riccati_bytes;
  const std::size_t N = problem.stages.size();
  layout.view.states = WorkspaceSlice<VectorView>(workspace.data(), &offset, N + 1);
  layout.view.controls = WorkspaceSlice<VectorView>(workspace.data(), &offset, N);
  layout.view.dynamics_multipliers = WorkspaceSlice<VectorView>(workspace.data(), &offset, N);
  layout.view.mixed_multipliers = WorkspaceSlice<VectorView>(workspace.data(), &offset, N);
  layout.view.state_multipliers = WorkspaceSlice<VectorView>(workspace.data(), &offset, N);
  layout.state_data = WorkspaceSlice<double>(workspace.data(), &offset, TotalStateScalars(problem));
  layout.control_data = WorkspaceSlice<double>(workspace.data(), &offset, TotalControlScalars(problem));
  layout.initial_multiplier_data =
      WorkspaceSlice<double>(workspace.data(), &offset, problem.initial_state.size());
  layout.dynamics_multiplier_data =
      WorkspaceSlice<double>(workspace.data(), &offset, TotalDynamicsMultiplierScalars(problem));
  layout.mixed_multiplier_data =
      WorkspaceSlice<double>(workspace.data(), &offset, TotalMixedMultiplierScalars(problem));
  layout.state_multiplier_data =
      WorkspaceSlice<double>(workspace.data(), &offset, TotalStateMultiplierScalars(problem));
  layout.terminal_state_multiplier_data =
      WorkspaceSlice<double>(workspace.data(), &offset, problem.terminal_E.rows());

  layout.view.state_count = N + 1;
  layout.view.control_count = N;
  layout.view.dynamics_multiplier_count = N;
  layout.view.mixed_multiplier_count = N;
  layout.view.state_multiplier_count = N;

  double* state = layout.state_data;
  if (N == 0) {
    layout.view.states[0] = {state, problem.initial_state.size()};
  } else {
    layout.view.states[0] = {state, problem.stages.front().A.cols()};
    state += layout.view.states[0].size;
    for (std::size_t i = 0; i < N; ++i) {
      layout.view.states[i + 1] = {state, problem.stages[i].A.rows()};
      state += problem.stages[i].A.rows();
    }
  }

  double* control = layout.control_data;
  double* dynamics = layout.dynamics_multiplier_data;
  double* mixed = layout.mixed_multiplier_data;
  double* state_multiplier = layout.state_multiplier_data;
  for (std::size_t i = 0; i < N; ++i) {
    layout.view.controls[i] = {control, problem.stages[i].B.cols()};
    control += problem.stages[i].B.cols();
    layout.view.dynamics_multipliers[i] = {dynamics, problem.stages[i].A.rows()};
    dynamics += problem.stages[i].A.rows();
    layout.view.mixed_multipliers[i] = {mixed, problem.stages[i].C.rows()};
    mixed += problem.stages[i].C.rows();
    layout.view.state_multipliers[i] = {state_multiplier, problem.stages[i].E.rows()};
    state_multiplier += problem.stages[i].E.rows();
  }
  layout.view.initial_multiplier = {layout.initial_multiplier_data, problem.initial_state.size()};
  layout.view.terminal_state_multiplier =
      {layout.terminal_state_multiplier_data, problem.terminal_E.rows()};
  return layout;
}

bool ComputeUnconstrainedRiccatiInto(const WorkspaceVector<Stage>& stages,
                                     const Matrix& terminal_Q,
                                     const Vector& terminal_q,
                                     double tolerance,
                                     RiccatiWorkspace* workspace,
                                     NewtonKktDiagnostics* diagnostics) {
  const std::size_t N = stages.size();
  if (N == 0) return true;
  {
    double* terminal_P = workspace->PPtr(N);
    for (std::size_t i = 0; i < terminal_Q.data().size(); ++i) {
      terminal_P[i] = terminal_Q.data()[i];
    }
    double* terminal_p = workspace->pPtr(N);
    for (std::size_t i = 0; i < terminal_q.size(); ++i) terminal_p[i] = terminal_q[i];
  }
  for (std::size_t rev = 0; rev < N; ++rev) {
    const std::size_t i = N - 1 - rev;
    const Stage& s = stages[i];
    const std::size_t n = workspace->state_dim[i];
    const std::size_t next_n = workspace->state_dim[i + 1];
    const std::size_t m = workspace->control_dim[i];
    const double* CLQR_RESTRICT P_next = workspace->PPtr(i + 1);
    const double* CLQR_RESTRICT p_next = workspace->pPtr(i + 1);
    const double* CLQR_RESTRICT A_data = s.A.data().data();
    const double* CLQR_RESTRICT B_data = s.B.data().data();
    const double* CLQR_RESTRICT Q_data = s.Q.data().data();
    const double* CLQR_RESTRICT R_data = s.R.data().data();
    const double* CLQR_RESTRICT M_data = s.M.data().data();
    const double* CLQR_RESTRICT c_data = s.c.data().data();
    const double* CLQR_RESTRICT q_data = s.q.data().data();
    const double* CLQR_RESTRICT r_data = s.r.data().data();

    for (std::size_t row = 0; row < next_n; ++row) {
      double value = p_next[row];
      CLQR_UNROLL
      for (std::size_t col = 0; col < next_n; ++col) {
        value += P_next[row * next_n + col] * c_data[col];
      }
      workspace->pc[row] = value;
    }

    for (std::size_t row = 0; row < n; ++row) {
      double* CLQR_RESTRICT out = workspace->A_T_P + row * next_n;
      for (std::size_t p_row = 0; p_row < next_n; ++p_row) {
        double value = 0.0;
        const double* CLQR_RESTRICT p_data = P_next + p_row * next_n;
        CLQR_UNROLL
        for (std::size_t p_col = 0; p_col < next_n; ++p_col) {
          value += p_data[p_col] * A_data[p_col * n + row];
        }
        out[p_row] = value;
      }
    }
    for (std::size_t row = 0; row < n; ++row) {
      for (std::size_t col = 0; col <= row; ++col) {
        double value = Q_data[row * n + col];
        CLQR_UNROLL
        for (std::size_t shared = 0; shared < next_n; ++shared) {
          value += workspace->A_T_P[row * next_n + shared] *
                   A_data[shared * n + col];
        }
        workspace->Hxx[row * n + col] = value;
      }
    }

    for (std::size_t row = 0; row < m; ++row) {
      double* CLQR_RESTRICT out = workspace->B_T_P + row * next_n;
      for (std::size_t p_row = 0; p_row < next_n; ++p_row) {
        double value = 0.0;
        const double* CLQR_RESTRICT p_data = P_next + p_row * next_n;
        CLQR_UNROLL
        for (std::size_t p_col = 0; p_col < next_n; ++p_col) {
          value += p_data[p_col] * B_data[p_col * m + row];
        }
        out[p_row] = value;
      }
    }
    for (std::size_t row = 0; row < m; ++row) {
      for (std::size_t col = 0; col <= row; ++col) {
        double value = R_data[row * m + col];
        CLQR_UNROLL
        for (std::size_t shared = 0; shared < next_n; ++shared) {
          value += workspace->B_T_P[row * next_n + shared] *
                   B_data[shared * m + col];
        }
        workspace->Huu[row * m + col] = value;
      }
    }

    for (std::size_t row = 0; row < n; ++row) {
      for (std::size_t col = 0; col < m; ++col) {
        double value = M_data[row * m + col];
        CLQR_UNROLL
        for (std::size_t shared = 0; shared < next_n; ++shared) {
          value += workspace->A_T_P[row * next_n + shared] *
                   B_data[shared * m + col];
        }
        workspace->Hxu[row * m + col] = value;
      }
    }

    for (std::size_t row = 0; row < n; ++row) {
      workspace->hx[row] = q_data[row];
    }
    for (std::size_t row = 0; row < m; ++row) {
      workspace->hu[row] = r_data[row];
    }
    for (std::size_t shared = 0; shared < next_n; ++shared) {
      const double pc = workspace->pc[shared];
      const double* CLQR_RESTRICT A_row = A_data + shared * n;
      CLQR_UNROLL
      for (std::size_t col = 0; col < n; ++col) {
        workspace->hx[col] += A_row[col] * pc;
      }
      const double* CLQR_RESTRICT B_row = B_data + shared * m;
      CLQR_UNROLL
      for (std::size_t col = 0; col < m; ++col) {
        workspace->hu[col] += B_row[col] * pc;
      }
    }

    if (!CholeskyFactorizeRaw(workspace->Huu, m, tolerance, workspace->lower)) {
      MirrorLowerTriangleRaw(workspace->Huu, m);
      CopyRawToMatrix(workspace->Huu, m, m, &workspace->fallback_Huu);
      CopyRawToMatrix(workspace->Hxu, n, m, &workspace->fallback_Hxu);
      CopyRawToVector(workspace->hu, m, &workspace->fallback_hu);
      if (diagnostics != nullptr) {
        AnalyzeReducedControlHessian(workspace->fallback_Huu, false, i, tolerance,
                                     diagnostics);
      }
      if (diagnostics != nullptr && diagnostics->singular) return false;
      Matrix solve_hxu =
          SolveLinearSystem(workspace->fallback_Huu, Transpose(workspace->fallback_Hxu), tolerance);
      Vector solve_hu = SolveLinearSystem(workspace->fallback_Huu, workspace->fallback_hu, tolerance);
      for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
          workspace->solve_hxu[row * n + col] = solve_hxu(row, col);
        }
        workspace->solve_hu[row] = solve_hu[row];
      }
    } else {
      SolveWithCholeskyRaw(workspace->lower, workspace->Hxu, workspace->hu, n, m,
                           workspace->solve_hxu, workspace->solve_hu);
    }

    double* CLQR_RESTRICT K = workspace->KPtr(i);
    double* CLQR_RESTRICT k = workspace->kPtr(i);
    for (std::size_t row = 0; row < m; ++row) {
      CLQR_UNROLL
      for (std::size_t col = 0; col < n; ++col) {
        K[row * n + col] = -workspace->solve_hxu[row * n + col];
      }
      k[row] = -workspace->solve_hu[row];
    }

    double* CLQR_RESTRICT P = workspace->PPtr(i);
    for (std::size_t row = 0; row < n; ++row) {
      for (std::size_t col = 0; col <= row; ++col) {
        double value = workspace->Hxx[row * n + col];
        CLQR_UNROLL
        for (std::size_t control = 0; control < m; ++control) {
          value -= workspace->Hxu[row * m + control] *
                   workspace->solve_hxu[control * n + col];
        }
        P[row * n + col] = value;
        if (row != col) P[col * n + row] = value;
      }
    }

    double* CLQR_RESTRICT p = workspace->pPtr(i);
    for (std::size_t row = 0; row < n; ++row) {
      double value = workspace->hx[row];
      CLQR_UNROLL
      for (std::size_t control = 0; control < m; ++control) {
        value -= workspace->Hxu[row * m + control] * workspace->solve_hu[control];
      }
      p[row] = value;
    }
  }
  return true;
}

void RecoverUnconstrainedMultipliersInto(const Problem& original,
                                         const RiccatiWorkspace& riccati,
                                         SolutionView* out) {
  const std::size_t N = original.stages.size();
  if (N == 0) {
    for (std::size_t row = 0; row < original.terminal_Q.rows(); ++row) {
      double value = original.terminal_q[row];
      for (std::size_t col = 0; col < original.terminal_Q.cols(); ++col) {
        value += original.terminal_Q(row, col) * out->states[0][col];
      }
      out->initial_multiplier[row] = -value;
    }
    return;
  }
  for (std::size_t node = 0; node <= N; ++node) {
    const std::size_t n = riccati.state_dim[node];
    const double* P = riccati.PPtr(node);
    const double* p = riccati.pPtr(node);
    const VectorView x = out->states[node];
    VectorView multiplier =
        node == 0 ? out->initial_multiplier : out->dynamics_multipliers[node - 1];
    for (std::size_t row = 0; row < n; ++row) {
      double value = p[row];
      CLQR_UNROLL
      for (std::size_t col = 0; col < n; ++col) {
        value += P[row * n + col] * x[col];
      }
      multiplier[row] = -value;
    }
  }
}

bool SolveUnconstrainedIntoView(const Problem& problem, double tolerance,
                                SolutionWorkspaceLayout* layout,
                                NewtonKktDiagnostics* diagnostics) {
  SolutionView* out = &layout->view;
  for (std::size_t i = 0; i < problem.initial_state.size(); ++i) {
    out->states[0][i] = problem.initial_state[i];
  }
  if (!ComputeUnconstrainedRiccatiInto(problem.stages, problem.terminal_Q,
                                       problem.terminal_q, tolerance, &layout->riccati,
                                       diagnostics)) {
    out->status = SolveStatus::kNumericalFailure;
    out->message = "reduced control Hessian is not positive definite";
    out->newton_kkt_singular = diagnostics != nullptr && diagnostics->singular;
    out->newton_kkt_wrong_inertia = diagnostics != nullptr && diagnostics->wrong_inertia;
    return false;
  }

  double objective = 0.0;
  for (std::size_t i = 0; i < problem.stages.size(); ++i) {
    const Stage& s = problem.stages[i];
    const std::size_t n = layout->riccati.state_dim[i];
    const std::size_t m = layout->riccati.control_dim[i];
    const double* K = layout->riccati.KPtr(i);
    const double* k = layout->riccati.kPtr(i);
    for (std::size_t row = 0; row < m; ++row) {
      double value = k[row];
      CLQR_UNROLL
      for (std::size_t col = 0; col < n; ++col) value += K[row * n + col] * out->states[i][col];
      out->controls[i][row] = value;
    }
    for (std::size_t row = 0; row < s.A.rows(); ++row) {
      double value = s.c[row];
      CLQR_UNROLL
      for (std::size_t col = 0; col < s.A.cols(); ++col) {
        value += s.A(row, col) * out->states[i][col];
      }
      CLQR_UNROLL
      for (std::size_t col = 0; col < s.B.cols(); ++col) {
        value += s.B(row, col) * out->controls[i][col];
      }
      out->states[i + 1][row] = value;
    }
    const VectorView& x = out->states[i];
    const VectorView& u = out->controls[i];
    for (std::size_t row = 0; row < s.Q.rows(); ++row) {
      double value = 0.0;
      CLQR_UNROLL
      for (std::size_t col = 0; col < s.Q.cols(); ++col) {
        value += s.Q(row, col) * x[col];
      }
      objective += 0.5 * x[row] * value;
    }
    for (std::size_t row = 0; row < s.R.rows(); ++row) {
      double value = 0.0;
      CLQR_UNROLL
      for (std::size_t col = 0; col < s.R.cols(); ++col) {
        value += s.R(row, col) * u[col];
      }
      objective += 0.5 * u[row] * value;
    }
    for (std::size_t row = 0; row < s.M.rows(); ++row) {
      double value = 0.0;
      CLQR_UNROLL
      for (std::size_t col = 0; col < s.M.cols(); ++col) {
        value += s.M(row, col) * u[col];
      }
      objective += x[row] * value;
    }
    for (std::size_t row = 0; row < s.q.size(); ++row) objective += s.q[row] * x[row];
    for (std::size_t row = 0; row < s.r.size(); ++row) objective += s.r[row] * u[row];
  }
  RecoverUnconstrainedMultipliersInto(problem, layout->riccati, out);
  const VectorView& terminal = out->states[problem.stages.size()];
  for (std::size_t row = 0; row < problem.terminal_Q.rows(); ++row) {
    double value = 0.0;
    CLQR_UNROLL
    for (std::size_t col = 0; col < problem.terminal_Q.cols(); ++col) {
      value += problem.terminal_Q(row, col) * terminal[col];
    }
    objective += 0.5 * terminal[row] * value;
  }
  for (std::size_t row = 0; row < problem.terminal_q.size(); ++row) {
    objective += problem.terminal_q[row] * terminal[row];
  }
  out->objective = objective;
  out->status = SolveStatus::kOptimal;
  out->message = "optimal";
  return true;
}

VectorView MakeVectorView(Vector& vector) {
  return {vector.data().empty() ? nullptr : vector.data().data(), vector.size()};
}

VectorView* AllocateVectorViews(WorkspaceArena* arena, std::size_t count) {
  if (count == 0) return nullptr;
  return static_cast<VectorView*>(arena->Allocate(sizeof(VectorView) * count,
                                                  alignof(VectorView)));
}

SolutionView MakeSolutionViewFromSolution(Solution* solution, WorkspaceArena* arena) {
  SolutionView view;
  view.status = solution->status;
  view.message = StatusName(solution->status);
  view.state_count = solution->states.size();
  view.control_count = solution->controls.size();
  view.dynamics_multiplier_count = solution->dynamics_multipliers.size();
  view.mixed_multiplier_count = solution->mixed_multipliers.size();
  view.state_multiplier_count = solution->state_multipliers.size();
  view.states = AllocateVectorViews(arena, view.state_count);
  view.controls = AllocateVectorViews(arena, view.control_count);
  view.dynamics_multipliers = AllocateVectorViews(arena, view.dynamics_multiplier_count);
  view.mixed_multipliers = AllocateVectorViews(arena, view.mixed_multiplier_count);
  view.state_multipliers = AllocateVectorViews(arena, view.state_multiplier_count);
  for (std::size_t i = 0; i < view.state_count; ++i) view.states[i] = MakeVectorView(solution->states[i]);
  for (std::size_t i = 0; i < view.control_count; ++i) view.controls[i] = MakeVectorView(solution->controls[i]);
  for (std::size_t i = 0; i < view.dynamics_multiplier_count; ++i) {
    view.dynamics_multipliers[i] = MakeVectorView(solution->dynamics_multipliers[i]);
  }
  for (std::size_t i = 0; i < view.mixed_multiplier_count; ++i) {
    view.mixed_multipliers[i] = MakeVectorView(solution->mixed_multipliers[i]);
  }
  for (std::size_t i = 0; i < view.state_multiplier_count; ++i) {
    view.state_multipliers[i] = MakeVectorView(solution->state_multipliers[i]);
  }
  view.initial_multiplier = MakeVectorView(solution->initial_multiplier);
  view.terminal_state_multiplier = MakeVectorView(solution->terminal_state_multiplier);
  view.newton_kkt_singular = solution->newton_kkt_singular;
  view.newton_kkt_wrong_inertia = solution->newton_kkt_wrong_inertia;
  view.newton_kkt_diagnostic = "";
  view.objective = solution->objective;
  return view;
}

ReducedSolution SolveUnconstrained(const WorkspaceVector<Stage>& stages,
                                   const Matrix& terminal_Q,
                                   const Vector& terminal_q,
                                   const Vector& initial_state, double tolerance,
                                   NewtonKktDiagnostics* diagnostics) {
  const std::size_t N = stages.size();
  RiccatiWorkspace workspace;
  if (WorkspaceArena* arena = ActiveWorkspaceArena(); arena != nullptr) {
    const std::size_t bytes = RiccatiWorkspace::RequiredBytes(stages);
    auto* data = static_cast<unsigned char*>(arena->Allocate(bytes, alignof(double)));
    workspace.Assign(stages, data, bytes);
  } else {
    workspace.Reserve(stages);
  }
  if (N == 0) {
    ReducedSolution sol;
    sol.x.push_back(initial_state);
    return sol;
  }
  if (!ComputeUnconstrainedRiccatiInto(stages, terminal_Q, terminal_q, tolerance,
                                       &workspace, diagnostics)) {
    throw std::runtime_error("reduced control Hessian is not positive definite");
  }

  ReducedSolution sol;
  sol.x.resize(N + 1);
  sol.u.resize(N);
  sol.x[0] = initial_state;
  for (std::size_t i = 0; i < N; ++i) {
    const Stage& s = stages[i];
    const std::size_t n = workspace.state_dim[i];
    const std::size_t m = workspace.control_dim[i];
    const double* K = workspace.KPtr(i);
    const double* k = workspace.kPtr(i);
    sol.u[i].resize(m);
    for (std::size_t row = 0; row < m; ++row) {
      double value = k[row];
      for (std::size_t col = 0; col < n; ++col) {
        value += K[row * n + col] * sol.x[i][col];
      }
      sol.u[i][row] = value;
    }
    sol.x[i + 1].resize(s.A.rows());
    for (std::size_t row = 0; row < s.A.rows(); ++row) {
      double value = s.c[row];
      for (std::size_t col = 0; col < s.A.cols(); ++col) {
        value += s.A(row, col) * sol.x[i][col];
      }
      for (std::size_t col = 0; col < s.B.cols(); ++col) {
        value += s.B(row, col) * sol.u[i][col];
      }
      sol.x[i + 1][row] = value;
    }
  }
  return sol;
}

bool RecoverMultipliers(const Problem& original, Solution* out, double tolerance,
                        std::string* error) {
  const std::size_t N = original.stages.size();
  out->dynamics_multipliers.assign(N, Vector());
  out->mixed_multipliers.assign(N, Vector());
  out->state_multipliers.assign(N, Vector());

  if (N == 0) {
    const std::size_t n = original.terminal_Q.rows();
    const std::size_t pt = original.terminal_E.rows();
    Matrix k(n, n + pt);
    for (std::size_t row = 0; row < n; ++row) k(row, row) = 1.0;
    for (std::size_t row = 0; row < n; ++row) {
      for (std::size_t col = 0; col < pt; ++col) k(row, n + col) = original.terminal_E(col, row);
    }
    Vector rhs = Scale(original.terminal_Q * out->states[0] + original.terminal_q, -1.0);
    RectangularSolve solve = SolveRectangularRref(k, rhs, tolerance);
    if (solve.inconsistent) {
      *error = "inconsistent terminal multiplier recovery";
      return false;
    }
    out->initial_multiplier = Slice(solve.x, 0, n);
    out->terminal_state_multiplier = Slice(solve.x, n, pt);
    return true;
  }

  AffineSet future_y;
  future_y.basis = Scale(Transpose(original.terminal_E), -1.0);
  future_y.offset = Scale(original.terminal_Q * out->states[N] + original.terminal_q, -1.0);
  WorkspaceVector<StageMultiplierMap> maps(N);
  for (std::size_t rev = 0; rev < N; ++rev) {
    const std::size_t i = N - 1 - rev;
    const Stage& s = original.stages[i];
    const std::size_t n = s.A.cols();
    const std::size_t m = s.B.cols();
    const std::size_t mixed = s.C.rows();
    const std::size_t state = s.E.rows();
    const std::size_t future_params = future_y.basis.cols();

    Matrix u_system(m, future_params + mixed + state);
    for (std::size_t row = 0; row < m; ++row) {
      for (std::size_t col = 0; col < future_params; ++col) {
        double value = 0.0;
        for (std::size_t r = 0; r < s.B.rows(); ++r) value -= s.B(r, row) * future_y.basis(r, col);
        u_system(row, col) = value;
      }
    }
    for (std::size_t row = 0; row < m; ++row) {
      for (std::size_t col = 0; col < mixed; ++col) u_system(row, future_params + col) = s.D(col, row);
    }
    Vector u_rhs = Scale(Transpose(s.M) * out->states[i] + s.R * out->controls[i] + s.r,
                         -1.0) +
                   Transpose(s.B) * future_y.offset;
    LinearParametrization local = ParametrizeLinearSystem(u_system, u_rhs, tolerance);
    if (local.inconsistent) {
      *error = "inconsistent stage multiplier recovery at stage " + std::to_string(i);
      return false;
    }

    Matrix prev_from_local(n, future_params + mixed + state);
    for (std::size_t row = 0; row < n; ++row) {
      for (std::size_t col = 0; col < future_params; ++col) {
        double value = 0.0;
        for (std::size_t r = 0; r < s.A.rows(); ++r) value += s.A(r, row) * future_y.basis(r, col);
        prev_from_local(row, col) = value;
      }
      for (std::size_t col = 0; col < mixed; ++col) prev_from_local(row, future_params + col) = -s.C(col, row);
      for (std::size_t col = 0; col < state; ++col) {
        prev_from_local(row, future_params + mixed + col) = -s.E(col, row);
      }
    }
    Vector prev_offset = Scale(s.Q * out->states[i] + s.M * out->controls[i] + s.q,
                               -1.0) +
                         Transpose(s.A) * future_y.offset;

    maps[i].future_y = future_y;
    maps[i].local_basis = local.basis;
    maps[i].local_offset = local.offset;
    maps[i].future_parameter_size = future_params;
    maps[i].mixed_size = mixed;
    maps[i].state_size = state;

    future_y.basis = prev_from_local * local.basis;
    future_y.offset = prev_offset + prev_from_local * local.offset;
  }

  Vector parameter(future_y.basis.cols());
  out->initial_multiplier = future_y.offset;
  for (std::size_t i = 0; i < N; ++i) {
    const StageMultiplierMap& map = maps[i];
    Vector local = map.local_offset + map.local_basis * parameter;
    Vector future_parameter = Slice(local, 0, map.future_parameter_size);
    out->dynamics_multipliers[i] = map.future_y.offset + map.future_y.basis * future_parameter;
    out->mixed_multipliers[i] = Slice(local, map.future_parameter_size, map.mixed_size);
    out->state_multipliers[i] =
        Slice(local, map.future_parameter_size + map.mixed_size, map.state_size);
    parameter = future_parameter;
  }
  out->terminal_state_multiplier = parameter;
  return true;
}

void RecoverUnconstrainedMultipliers(const Problem& original, Solution* out) {
  const std::size_t N = original.stages.size();
  out->mixed_multipliers.assign(N, Vector(0));
  out->state_multipliers.assign(N, Vector(0));
  out->terminal_state_multiplier = Vector(0);
  if (N == 0) {
    out->dynamics_multipliers.clear();
    out->initial_multiplier = Vector(original.terminal_Q.rows());
    for (std::size_t row = 0; row < original.terminal_Q.rows(); ++row) {
      double value = original.terminal_q[row];
      for (std::size_t col = 0; col < original.terminal_Q.cols(); ++col) {
        value += original.terminal_Q(row, col) * out->states[0][col];
      }
      out->initial_multiplier[row] = -value;
    }
    return;
  }

  out->dynamics_multipliers.assign(N, Vector());
  out->dynamics_multipliers[N - 1] = Vector(original.terminal_Q.rows());
  for (std::size_t row = 0; row < original.terminal_Q.rows(); ++row) {
    double value = original.terminal_q[row];
    for (std::size_t col = 0; col < original.terminal_Q.cols(); ++col) {
      value += original.terminal_Q(row, col) * out->states[N][col];
    }
    out->dynamics_multipliers[N - 1][row] = -value;
  }
  for (std::size_t rev = 1; rev < N; ++rev) {
    const std::size_t i = N - 1 - rev;
    const Stage& s = original.stages[i + 1];
    out->dynamics_multipliers[i] = Vector(s.A.cols());
    for (std::size_t col = 0; col < s.A.cols(); ++col) {
      double value = -s.q[col];
      for (std::size_t row = 0; row < s.A.rows(); ++row) {
        value += s.A(row, col) * out->dynamics_multipliers[i + 1][row];
      }
      for (std::size_t q_col = 0; q_col < s.Q.cols(); ++q_col) {
        value -= s.Q(col, q_col) * out->states[i + 1][q_col];
      }
      for (std::size_t u_col = 0; u_col < s.M.cols(); ++u_col) {
        value -= s.M(col, u_col) * out->controls[i + 1][u_col];
      }
      out->dynamics_multipliers[i][col] = value;
    }
  }
  const Stage& first = original.stages[0];
  out->initial_multiplier = Vector(first.A.cols());
  for (std::size_t col = 0; col < first.A.cols(); ++col) {
    double value = -first.q[col];
    for (std::size_t row = 0; row < first.A.rows(); ++row) {
      value += first.A(row, col) * out->dynamics_multipliers[0][row];
    }
    for (std::size_t q_col = 0; q_col < first.Q.cols(); ++q_col) {
      value -= first.Q(col, q_col) * out->states[0][q_col];
    }
    for (std::size_t u_col = 0; u_col < first.M.cols(); ++u_col) {
      value -= first.M(col, u_col) * out->controls[0][u_col];
    }
    out->initial_multiplier[col] = value;
  }
}

bool RecoverMixedOnlyMultipliers(const Problem& original, Solution* out, double tolerance,
                                 std::string* error) {
  const std::size_t N = original.stages.size();
  out->dynamics_multipliers.assign(N, Vector());
  out->mixed_multipliers.assign(N, Vector());
  out->state_multipliers.assign(N, Vector(0));
  out->terminal_state_multiplier = Vector(0);

  if (N == 0) {
    out->initial_multiplier = Vector(original.terminal_Q.rows());
    for (std::size_t row = 0; row < original.terminal_Q.rows(); ++row) {
      double value = original.terminal_q[row];
      for (std::size_t col = 0; col < original.terminal_Q.cols(); ++col) {
        value += original.terminal_Q(row, col) * out->states[0][col];
      }
      out->initial_multiplier[row] = -value;
    }
    return true;
  }

  out->dynamics_multipliers[N - 1].resize(original.terminal_Q.rows());
  for (std::size_t row = 0; row < original.terminal_Q.rows(); ++row) {
    double value = original.terminal_q[row];
    for (std::size_t col = 0; col < original.terminal_Q.cols(); ++col) {
      value += original.terminal_Q(row, col) * out->states[N][col];
    }
    out->dynamics_multipliers[N - 1][row] = -value;
  }

  for (std::size_t rev = 0; rev < N; ++rev) {
    const std::size_t i = N - 1 - rev;
    const Stage& stage = original.stages[i];
    const Vector& y = out->dynamics_multipliers[i];
    if (stage.C.rows() > 0) {
      RectangularSolve solve =
          SolveMixedMultiplierRref(stage, out->states[i], out->controls[i], y, tolerance);
      if (solve.inconsistent) {
        *error = "inconsistent mixed-only multiplier recovery at stage " + std::to_string(i);
        return false;
      }
      out->mixed_multipliers[i] = std::move(solve.x);
    }

    Vector& previous = i == 0 ? out->initial_multiplier : out->dynamics_multipliers[i - 1];
    previous.resize(stage.A.cols());
    for (std::size_t state = 0; state < stage.A.cols(); ++state) {
      double value = -stage.q[state];
      for (std::size_t row = 0; row < stage.A.rows(); ++row) {
        value += stage.A(row, state) * y[row];
      }
      for (std::size_t col = 0; col < stage.Q.cols(); ++col) {
        value -= stage.Q(state, col) * out->states[i][col];
      }
      for (std::size_t control = 0; control < stage.M.cols(); ++control) {
        value -= stage.M(state, control) * out->controls[i][control];
      }
      for (std::size_t constraint = 0; constraint < stage.C.rows(); ++constraint) {
        value -= stage.C(constraint, state) * out->mixed_multipliers[i][constraint];
      }
      previous[state] = value;
    }
  }
  return true;
}

void ApplyDiagnostics(const NewtonKktDiagnostics& diagnostics, Solution* out) {
  out->newton_kkt_singular = diagnostics.singular;
  out->newton_kkt_wrong_inertia = diagnostics.wrong_inertia;
  out->newton_kkt_diagnostic =
      diagnostics.messages.empty() ? std::string() : JoinMessages(diagnostics.messages);
}

void ApplyControlMap(const ControlMap& map, const Vector& reduced_state,
                     const Vector& reduced_control, Vector* out) {
  out->resize(map.offset.size());
  if (LazyIdentityControlLinear(map)) {
    for (std::size_t row = 0; row < map.offset.size(); ++row) {
      double value = map.offset[row] + reduced_control[row];
      CLQR_UNROLL
      for (std::size_t col = 0; col < reduced_state.size(); ++col) {
        value += map.state_linear(row, col) * reduced_state[col];
      }
      (*out)[row] = value;
    }
  } else {
    for (std::size_t row = 0; row < map.offset.size(); ++row) {
      double value = map.offset[row];
      CLQR_UNROLL
      for (std::size_t col = 0; col < reduced_state.size(); ++col) {
        value += map.state_linear(row, col) * reduced_state[col];
      }
      CLQR_UNROLL
      for (std::size_t col = 0; col < reduced_control.size(); ++col) {
        value += map.control_linear(row, col) * reduced_control[col];
      }
      (*out)[row] = value;
    }
  }
}

void ApplyControlMapRaw(const ControlMap& map, const Vector& reduced_state,
                        const double* CLQR_RESTRICT reduced_control,
                        std::size_t reduced_control_size, Vector* out) {
  out->resize(map.offset.size());
  if (LazyIdentityControlLinear(map)) {
    for (std::size_t row = 0; row < map.offset.size(); ++row) {
      double value = map.offset[row] + reduced_control[row];
      CLQR_UNROLL
      for (std::size_t col = 0; col < reduced_state.size(); ++col) {
        value += map.state_linear(row, col) * reduced_state[col];
      }
      (*out)[row] = value;
    }
  } else {
    for (std::size_t row = 0; row < map.offset.size(); ++row) {
      double value = map.offset[row];
      CLQR_UNROLL
      for (std::size_t col = 0; col < reduced_state.size(); ++col) {
        value += map.state_linear(row, col) * reduced_state[col];
      }
      CLQR_UNROLL
      for (std::size_t col = 0; col < reduced_control_size; ++col) {
        value += map.control_linear(row, col) * reduced_control[col];
      }
      (*out)[row] = value;
    }
  }
}

Solution SolveMixedOnlyIdentityState(const Problem& original, const WorkingState& state,
                                     const Vector& initial_state, double tolerance,
                                     NewtonKktDiagnostics* diagnostics) {
  const WorkspaceVector<Stage>& stages = state.problem.stages;
  const std::size_t N = stages.size();
  RiccatiWorkspace workspace;
  if (WorkspaceArena* arena = ActiveWorkspaceArena(); arena != nullptr) {
    const std::size_t bytes = RiccatiWorkspace::RequiredBytes(stages);
    auto* data = static_cast<unsigned char*>(arena->Allocate(bytes, alignof(double)));
    workspace.Assign(stages, data, bytes);
  } else {
    workspace.Reserve(stages);
  }
  if (!ComputeUnconstrainedRiccatiInto(stages, state.problem.terminal_Q,
                                       state.problem.terminal_q, tolerance, &workspace,
                                       diagnostics)) {
    throw std::runtime_error("reduced control Hessian is not positive definite");
  }

  Solution out;
  out.status = SolveStatus::kOptimal;
  out.message = "optimal";
  if (diagnostics != nullptr) ApplyDiagnostics(*diagnostics, &out);
  out.states.resize(N + 1);
  out.controls.resize(N);
  out.states[0] = initial_state;
  double objective = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    const Stage& reduced_stage = stages[i];
    const Stage& original_stage = original.stages[i];
    const std::size_t n = workspace.state_dim[i];
    const std::size_t reduced_m = workspace.control_dim[i];
    const double* K = workspace.KPtr(i);
    const double* k = workspace.kPtr(i);
    for (std::size_t row = 0; row < reduced_m; ++row) {
      double value = k[row];
      CLQR_UNROLL
      for (std::size_t col = 0; col < n; ++col) value += K[row * n + col] * out.states[i][col];
      workspace.hu[row] = value;
    }

    if (original.stages[i].C.rows() == 0) {
      out.controls[i].resize(reduced_m);
      for (std::size_t row = 0; row < reduced_m; ++row) out.controls[i][row] = workspace.hu[row];
    } else {
      ApplyControlMapRaw(state.control_maps[i], out.states[i], workspace.hu, reduced_m,
                         &out.controls[i]);
    }

    out.states[i + 1].resize(reduced_stage.A.rows());
    for (std::size_t row = 0; row < reduced_stage.A.rows(); ++row) {
      double value = reduced_stage.c[row];
      CLQR_UNROLL
      for (std::size_t col = 0; col < reduced_stage.A.cols(); ++col) {
        value += reduced_stage.A(row, col) * out.states[i][col];
      }
      CLQR_UNROLL
      for (std::size_t col = 0; col < reduced_stage.B.cols(); ++col) {
        value += reduced_stage.B(row, col) * workspace.hu[col];
      }
      out.states[i + 1][row] = value;
    }

    const Vector& x = out.states[i];
    const Vector& u = out.controls[i];
    for (std::size_t row = 0; row < original_stage.Q.rows(); ++row) {
      double value = 0.0;
      CLQR_UNROLL
      for (std::size_t col = 0; col < original_stage.Q.cols(); ++col) {
        value += original_stage.Q(row, col) * x[col];
      }
      objective += 0.5 * x[row] * value;
    }
    for (std::size_t row = 0; row < original_stage.R.rows(); ++row) {
      double value = 0.0;
      CLQR_UNROLL
      for (std::size_t col = 0; col < original_stage.R.cols(); ++col) {
        value += original_stage.R(row, col) * u[col];
      }
      objective += 0.5 * u[row] * value;
    }
    for (std::size_t row = 0; row < original_stage.M.rows(); ++row) {
      double value = 0.0;
      CLQR_UNROLL
      for (std::size_t col = 0; col < original_stage.M.cols(); ++col) {
        value += original_stage.M(row, col) * u[col];
      }
      objective += x[row] * value;
    }
    for (std::size_t row = 0; row < original_stage.q.size(); ++row) {
      objective += original_stage.q[row] * x[row];
    }
    for (std::size_t row = 0; row < original_stage.r.size(); ++row) {
      objective += original_stage.r[row] * u[row];
    }
  }

  const Vector& terminal = out.states[N];
  for (std::size_t row = 0; row < original.terminal_Q.rows(); ++row) {
    double value = 0.0;
    CLQR_UNROLL
    for (std::size_t col = 0; col < original.terminal_Q.cols(); ++col) {
      value += original.terminal_Q(row, col) * terminal[col];
    }
    objective += 0.5 * terminal[row] * value;
  }
  for (std::size_t row = 0; row < original.terminal_q.size(); ++row) {
    objective += original.terminal_q[row] * terminal[row];
  }
  out.objective = objective;
  std::string error;
  if (!RecoverMixedOnlyMultipliers(original, &out, tolerance, &error)) {
    out.status = SolveStatus::kNumericalFailure;
    out.message = error;
  }
  return out;
}

Solution Recover(const Problem& original, const WorkingState& state,
                 ReducedSolution&& reduced, double tolerance,
                 const NewtonKktDiagnostics& diagnostics) {
  Solution out;
  out.status = SolveStatus::kOptimal;
  out.message = "optimal";
  ApplyDiagnostics(diagnostics, &out);
  const bool has_original_state_constraints = AnyOriginalStateConstraints(original);
  const bool identity_state_maps =
      !has_original_state_constraints && StateMapsAreIdentity(state, tolerance);
  if (identity_state_maps) {
    out.states = std::move(reduced.x);
  } else {
    out.states.resize(reduced.x.size());
  }
  out.controls.resize(reduced.u.size());
  if (!identity_state_maps) {
    for (std::size_t i = 0; i < reduced.x.size(); ++i) {
      const StateMap& map = state.state_maps[i];
      if (LazyIdentityStateMap(map)) {
        out.states[i] = reduced.x[i];
      } else {
        out.states[i] = map.linear * reduced.x[i] + map.offset;
      }
    }
  }
  for (std::size_t i = 0; i < reduced.u.size(); ++i) {
    if (identity_state_maps && original.stages[i].C.rows() == 0) {
      out.controls[i] = std::move(reduced.u[i]);
    } else {
      const Vector& reduced_state = identity_state_maps ? out.states[i] : reduced.x[i];
      ApplyControlMap(state.control_maps[i], reduced_state, reduced.u[i], &out.controls[i]);
    }
  }
  out.objective = Objective(original, out.states, out.controls);
  std::string error;
  const bool recovered =
      has_original_state_constraints ? RecoverMultipliers(original, &out, tolerance, &error)
                                     : RecoverMixedOnlyMultipliers(original, &out, tolerance, &error);
  if (!recovered) {
    out.status = SolveStatus::kNumericalFailure;
    out.message = error;
  }
  return out;
}

Solution RecoverUnmapped(const Problem& original, ReducedSolution&& reduced,
                         double tolerance, const NewtonKktDiagnostics& diagnostics) {
  Solution out;
  out.status = SolveStatus::kOptimal;
  out.message = "optimal";
  ApplyDiagnostics(diagnostics, &out);
  out.states = std::move(reduced.x);
  out.controls = std::move(reduced.u);
  out.objective = Objective(original, out.states, out.controls);
  (void)tolerance;
  RecoverUnconstrainedMultipliers(original, &out);
  return out;
}

}  // namespace

std::size_t Workspace::RequiredBytes(const Problem& problem) {
  return WorkspaceRequiredBytesInternal(problem, SolveOptions{});
}

std::size_t Workspace::RequiredBytes(const Problem& problem, const SolveOptions& options) {
  return WorkspaceRequiredBytesInternal(problem, options);
}

void Workspace::Reserve(const Problem& problem) {
  Reserve(problem, SolveOptions{});
}

void Workspace::Reserve(const Problem& problem, const SolveOptions& options) {
  const std::size_t required = RequiredBytes(problem, options);
  owned_.assign(required, 0);
  external_ = nullptr;
  data_ = owned_.data();
  size_ = owned_.size();
  arena_.Reset(data_, size_);
}

void Workspace::UseExternalMemory(void* memory, std::size_t bytes) {
  external_ = reinterpret_cast<unsigned char*>(memory);
  data_ = external_;
  size_ = bytes;
  arena_.Reset(data_, size_);
}

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

static Solution SolveInternalResult(const Problem& problem, const SolveOptions& options);

SolutionView Solve(const Problem& problem, Workspace& workspace,
                   const SolveOptions& options) {
  SolutionView out;
  NewtonKktDiagnostics diagnostics;
  try {
    ValidateProblem(problem);
    if (AnyOriginalConstraints(problem)) {
      const std::size_t required = WorkspaceRequiredBytesInternal(problem, options);
      if (workspace.size() < required) {
        throw std::invalid_argument("workspace is too small");
      }
      workspace.arena().Clear();
      ScopedWorkspaceArena scoped(&workspace.arena());
      Solution solution = SolveInternalResult(problem, options);
      return MakeSolutionViewFromSolution(&solution, &workspace.arena());
    }
    SolutionWorkspaceLayout layout = BindSolutionWorkspace(problem, workspace);
    SolveUnconstrainedIntoView(problem, options.tolerance, &layout, &diagnostics);
    layout.view.newton_kkt_singular = diagnostics.singular;
    layout.view.newton_kkt_wrong_inertia =
        layout.view.newton_kkt_wrong_inertia || diagnostics.wrong_inertia;
    return layout.view;
  } catch (const std::invalid_argument& e) {
    out.status = SolveStatus::kInvalidInput;
    out.message = e.what();
    return out;
  } catch (const std::exception& e) {
    out.status = SolveStatus::kNumericalFailure;
    out.message = e.what();
    return out;
  }
}

static Solution SolveInternalResult(const Problem& problem, const SolveOptions& options) {
  Solution out;
  NewtonKktDiagnostics diagnostics;
  try {
    ValidateProblem(problem);
    if (!AnyOriginalConstraints(problem)) {
      try {
        ReducedSolution reduced =
            SolveUnconstrained(problem.stages, problem.terminal_Q, problem.terminal_q,
                               problem.initial_state, options.tolerance, &diagnostics);
        return RecoverUnmapped(problem, std::move(reduced), options.tolerance, diagnostics);
      } catch (const std::runtime_error& e) {
        out.status = SolveStatus::kNumericalFailure;
        out.message = e.what();
        ApplyDiagnostics(diagnostics, &out);
        return out;
      }
    }
    WorkingState state = Initialize(problem);
    std::string error;
    EliminateConstraintsRightToLeft(state, options.tolerance, &error, &diagnostics);
    LinearParametrization initial =
        ReducedInitialState(problem, state.state_maps.front(), options.tolerance);
    if (initial.inconsistent) {
      out.status = SolveStatus::kInfeasible;
      out.message = "initial state is inconsistent with state equality constraints";
      ApplyDiagnostics(diagnostics, &out);
      return out;
    }
    if (!LazyIdentityStateMap(state.state_maps.front()) &&
        state.state_maps.front().linear.cols() < state.state_maps.front().linear.rows()) {
      diagnostics.singular = true;
      AddDiagnostic(&diagnostics, "initial state makes node 0 state constraints redundant");
    }
    try {
      const bool has_original_state_constraints = AnyOriginalStateConstraints(problem);
      const bool identity_state_maps =
          !has_original_state_constraints && StateMapsAreIdentity(state, options.tolerance);
      if (identity_state_maps) {
        return SolveMixedOnlyIdentityState(problem, state, initial.offset, options.tolerance,
                                           &diagnostics);
      }
      ReducedSolution reduced =
          SolveUnconstrained(state.problem.stages, state.problem.terminal_Q,
                             state.problem.terminal_q, initial.offset, options.tolerance,
                             &diagnostics);
      return Recover(problem, state, std::move(reduced), options.tolerance, diagnostics);
    } catch (const std::runtime_error& e) {
      out.status = SolveStatus::kNumericalFailure;
      out.message = e.what();
      ApplyDiagnostics(diagnostics, &out);
      return out;
    }
  } catch (const std::invalid_argument& e) {
    out.status = SolveStatus::kInvalidInput;
    out.message = e.what();
    ApplyDiagnostics(diagnostics, &out);
    return out;
  } catch (const std::runtime_error& e) {
    out.status = SolveStatus::kInfeasible;
    out.message = e.what();
    ApplyDiagnostics(diagnostics, &out);
    return out;
  } catch (const std::exception& e) {
    out.status = SolveStatus::kNumericalFailure;
    out.message = e.what();
    ApplyDiagnostics(diagnostics, &out);
    return out;
  }
}

}  // namespace clqr
