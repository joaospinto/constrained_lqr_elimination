#ifndef CLQR_CLQR_H_
#define CLQR_CLQR_H_

#include <cstddef>
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
  WorkspaceVector<Stage> stages;
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

struct VectorView {
  double* data = nullptr;
  std::size_t size = 0;

  double& operator[](std::size_t i) { return data[i]; }
  const double& operator[](std::size_t i) const { return data[i]; }
};

struct SolutionView {
  SolveStatus status = SolveStatus::kInvalidInput;
  const char* message = "";
  VectorView* states = nullptr;
  std::size_t state_count = 0;
  VectorView* controls = nullptr;
  std::size_t control_count = 0;
  VectorView initial_multiplier;
  VectorView* dynamics_multipliers = nullptr;
  std::size_t dynamics_multiplier_count = 0;
  VectorView* mixed_multipliers = nullptr;
  std::size_t mixed_multiplier_count = 0;
  VectorView* state_multipliers = nullptr;
  std::size_t state_multiplier_count = 0;
  VectorView terminal_state_multiplier;
  bool newton_kkt_singular = false;
  bool newton_kkt_wrong_inertia = false;
  const char* newton_kkt_diagnostic = "";
  double objective = 0.0;
};

class Workspace {
 public:
  Workspace() = default;
  Workspace(void* memory, std::size_t bytes) { UseExternalMemory(memory, bytes); }

  static std::size_t RequiredBytes(const Problem& problem);
  static std::size_t RequiredBytes(const Problem& problem, const SolveOptions& options);
  static constexpr std::size_t RequiredBytesUniform(std::size_t stages,
                                                    std::size_t state_dim,
                                                    std::size_t control_dim) {
    return RequiredBytesUniformWithTerminal(stages, state_dim, state_dim, control_dim);
  }
  static constexpr std::size_t RequiredBytesUniformWithTerminal(
      std::size_t stages, std::size_t state_dim, std::size_t terminal_state_dim,
      std::size_t control_dim) {
    const std::size_t total_state_scalars = stages * state_dim + terminal_state_dim;
    const std::size_t total_control_scalars = stages * control_dim;
    const std::size_t total_dynamics_scalars =
        stages == 0 ? 0 : (stages - 1) * state_dim + terminal_state_dim;
    const std::size_t total_p = total_state_scalars;
    const std::size_t total_P =
        stages * state_dim * state_dim + terminal_state_dim * terminal_state_dim;
    const std::size_t total_K = stages * control_dim * state_dim;
    const std::size_t total_k = total_control_scalars;
    const std::size_t max_state =
        state_dim > terminal_state_dim ? state_dim : terminal_state_dim;
    const std::size_t max_control = control_dim;

    std::size_t bytes = 0;
    bytes = AddAligned(bytes, alignof(std::size_t), sizeof(std::size_t) * (stages + 1));
    bytes = AddAligned(bytes, alignof(std::size_t), sizeof(std::size_t) * stages);
    bytes = AddAligned(bytes, alignof(std::size_t), sizeof(std::size_t) * (stages + 1));
    bytes = AddAligned(bytes, alignof(std::size_t), sizeof(std::size_t) * (stages + 1));
    bytes = AddAligned(bytes, alignof(std::size_t), sizeof(std::size_t) * stages);
    bytes = AddAligned(bytes, alignof(std::size_t), sizeof(std::size_t) * stages);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * total_P);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * total_p);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * total_K);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * total_k);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * max_state);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * max_state * max_state);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * max_control * max_control);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * max_state * max_control);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * max_state * max_state);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * max_control * max_state);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * max_state);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * max_control);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * max_control * max_control);
    bytes = AddAligned(bytes, alignof(double),
                       sizeof(double) * max_control * (max_state + 1));
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * max_control * max_state);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * max_control);

    bytes = AddAligned(bytes, alignof(VectorView),
                       sizeof(VectorView) * (stages + 1));
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * total_state_scalars);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * total_control_scalars);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * state_dim);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * total_dynamics_scalars);
    return bytes;
  }
  static constexpr std::size_t RequiredBytesUniformConstrained(
      std::size_t stages, std::size_t state_dim, std::size_t control_dim,
      std::size_t mixed_constraints_per_stage,
      std::size_t state_constraints_per_stage = 0,
      std::size_t terminal_constraints = 0,
      std::size_t max_elimination_passes = 100) {
    const std::size_t total_state_scalars = (stages + 1) * state_dim;
    const std::size_t total_control_scalars = stages * control_dim;
    const std::size_t total_dynamics_scalars = stages * state_dim;
    const std::size_t total_mixed_scalars = stages * mixed_constraints_per_stage;
    const std::size_t total_state_multiplier_scalars =
        stages * state_constraints_per_stage;
    const std::size_t mixed_rows_bound = mixed_constraints_per_stage + state_dim;
    const std::size_t state_rows_bound =
        Max(terminal_constraints, state_constraints_per_stage + mixed_rows_bound);
    const std::size_t state_pivot_bound = Min(state_dim, state_rows_bound);
    const bool has_stage_constraints =
        stages > 0 && (mixed_constraints_per_stage > 0 || state_constraints_per_stage > 0);
    const std::size_t rightmost_constrained_node =
        terminal_constraints > 0 ? stages : (has_stage_constraints ? stages - 1 : 0);
    const std::size_t pass_bound =
        Max(std::size_t{1}, Min(max_elimination_passes, rightmost_constrained_node + 1));
    const std::size_t generated_mixed_stage_ops =
        rightmost_constrained_node * (rightmost_constrained_node + 1) / 2;
    const std::size_t original_mixed_stage_ops =
        mixed_constraints_per_stage > 0 ? stages : std::size_t{0};
    const std::size_t mixed_stage_ops =
        original_mixed_stage_ops + generated_mixed_stage_ops;
    const std::size_t mixed_stage_doubles =
        2 * mixed_rows_bound * (control_dim + state_dim + 1) +
        control_dim * state_dim + control_dim * control_dim + control_dim +
        mixed_rows_bound * state_dim + mixed_rows_bound +
        state_dim * state_dim + control_dim * control_dim + state_dim * control_dim +
        state_dim * control_dim + state_dim + control_dim + state_dim +
        8 * state_dim * state_dim + 4 * control_dim * control_dim +
        6 * state_dim * control_dim + 2 * state_dim * state_dim +
        2 * state_dim * control_dim + 3 * state_dim + 4 * state_dim +
        4 * control_dim + (state_rows_bound + mixed_rows_bound) * state_dim +
        state_rows_bound + mixed_rows_bound + 2 * control_dim * state_dim +
        2 * control_dim * control_dim + 3 * control_dim;
    const std::size_t state_stage_doubles =
        2 * state_rows_bound * (state_dim + 1) + state_dim * state_dim + state_dim +
        3 * state_dim * state_dim + 3 * state_dim * control_dim + 4 * state_dim +
        8 * state_dim * state_dim + 4 * state_dim * control_dim +
        2 * control_dim * state_dim + 4 * state_dim + 2 * control_dim +
        (state_pivot_bound + mixed_constraints_per_stage) *
            (state_dim + control_dim + 1) +
        3 * state_dim * state_dim + 2 * state_dim + 2 * control_dim * state_dim +
        control_dim * control_dim + control_dim;
    const std::size_t parameter_bound =
        Max(std::size_t{1}, total_mixed_scalars + total_state_multiplier_scalars +
                                terminal_constraints + mixed_constraints_per_stage +
                                state_constraints_per_stage + terminal_constraints);

    std::size_t bytes = 0;
    bytes = AddAligned(bytes, alignof(Stage), sizeof(Stage) * stages);
    bytes = AddAligned(bytes, alignof(Vector), sizeof(Vector) * (10 * stages + 2));
    bytes = AddAligned(bytes, alignof(Matrix), sizeof(Matrix) * (8 * stages + 2));
    bytes = AddAligned(bytes, alignof(std::size_t),
                       sizeof(std::size_t) * (6 * stages + 3));
    bytes = AddAligned(bytes, alignof(double),
                       sizeof(double) * (80 * stages + 32));
    bytes = AddAligned(bytes, alignof(double),
                       sizeof(double) * stages *
                           (state_dim * state_dim + state_dim * control_dim + state_dim +
                            state_dim * state_dim + control_dim * control_dim +
                            state_dim * control_dim + state_dim + control_dim +
                            mixed_constraints_per_stage * state_dim +
                            mixed_constraints_per_stage * control_dim +
                            mixed_constraints_per_stage +
                            state_constraints_per_stage * state_dim +
                            state_constraints_per_stage));
    bytes = AddAligned(bytes, alignof(double),
                       sizeof(double) *
                           (state_dim * state_dim + state_dim +
                            terminal_constraints * state_dim + terminal_constraints));
    bytes = AddAligned(bytes, alignof(double),
                       sizeof(double) *
                           ((stages + 1) * (state_dim * state_dim + state_dim) +
                            stages * (control_dim * state_dim +
                                      control_dim * control_dim + control_dim)));
    bytes = AddAligned(bytes, alignof(double),
                       sizeof(double) *
                           (pass_bound * (stages + 1) *
                                (2 * state_rows_bound * (state_dim + 1) +
                                 state_dim * state_dim + state_dim) +
                            pass_bound * stages * state_stage_doubles +
                            mixed_stage_ops * mixed_stage_doubles));
    bytes = AddAligned(bytes, alignof(std::size_t),
                       sizeof(std::size_t) *
                           (pass_bound * (stages + 1) * 3 * state_dim +
                            mixed_stage_ops *
                                (3 * control_dim + mixed_rows_bound + 3 * state_dim)));
    bytes = AddAligned(bytes, alignof(Matrix),
                       sizeof(Matrix) * pass_bound * (4 * stages + 2));
    bytes = AddAligned(bytes, alignof(Vector),
                       sizeof(Vector) * pass_bound * (4 * stages + 2));
    bytes += RequiredBytesUniform(stages, state_dim, control_dim);
    bytes = AddAligned(bytes, alignof(VectorView),
                       sizeof(VectorView) * (stages + 1));
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * total_state_scalars);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * total_control_scalars);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * state_dim);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * total_dynamics_scalars);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * total_mixed_scalars);
    bytes = AddAligned(bytes, alignof(double),
                       sizeof(double) * total_state_multiplier_scalars);
    bytes = AddAligned(bytes, alignof(double), sizeof(double) * terminal_constraints);
    bytes = AddAligned(bytes, alignof(double),
                       sizeof(double) * stages *
                           (control_dim * (parameter_bound + mixed_constraints_per_stage +
                                           state_constraints_per_stage) +
                            control_dim + 2 * control_dim *
                                              (parameter_bound + mixed_constraints_per_stage +
                                               state_constraints_per_stage + 1) +
                            parameter_bound * parameter_bound + parameter_bound +
                            state_dim * parameter_bound + state_dim + parameter_bound));
    return bytes;
  }

  void Reserve(const Problem& problem);
  void Reserve(const Problem& problem, const SolveOptions& options);
  void UseExternalMemory(void* memory, std::size_t bytes);
  unsigned char* data() { return data_; }
  const unsigned char* data() const { return data_; }
  std::size_t size() const { return size_; }
  bool owns_memory() const { return external_ == nullptr; }
  WorkspaceArena& arena() { return arena_; }
  const WorkspaceArena& arena() const { return arena_; }

 private:
  static constexpr std::size_t Align(std::size_t offset, std::size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
  }
  static constexpr std::size_t AddAligned(std::size_t offset, std::size_t alignment,
                                          std::size_t bytes) {
    return Align(offset, alignment) + bytes;
  }
  static constexpr std::size_t Min(std::size_t a, std::size_t b) {
    return a < b ? a : b;
  }
  static constexpr std::size_t Max(std::size_t a, std::size_t b) {
    return a > b ? a : b;
  }

  std::vector<unsigned char> owned_;
  unsigned char* external_ = nullptr;
  unsigned char* data_ = nullptr;
  std::size_t size_ = 0;
  WorkspaceArena arena_;
};

SolutionView Solve(const Problem& problem, Workspace& workspace,
                   const SolveOptions& options = SolveOptions{});
const char* StatusName(SolveStatus status);

}  // namespace clqr

#endif  // CLQR_CLQR_H_
