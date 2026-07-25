#ifndef CLQR_CLQR_H_
#define CLQR_CLQR_H_

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "clqr/linalg.h"

namespace clqr {

struct Stage {
  Matrix A;
  Matrix B;
  Vector c;

  Matrix R;
  Matrix M;
  Vector r;

  Matrix C;
  Matrix D;
  Vector d;

  Matrix E;
  Vector e;
};

struct Problem {
  WorkspaceVector<Stage> stages;
  WorkspaceVector<Matrix> Q;
  WorkspaceVector<Vector> q;
  Matrix terminal_E;
  Vector terminal_e;
  Vector initial_state;
};

struct StageRhs {
  Vector c;
  Vector r;
  Vector d;
  Vector e;
};

struct SolveRhs {
  WorkspaceVector<StageRhs> stages;
  WorkspaceVector<Vector> q;
  Vector terminal_e;
  Vector initial_state;
};

SolveRhs ExtractRhs(const Problem& problem);

enum class SolveStatus {
  kOptimal,
  kInfeasible,
  kInvalidInput,
  kNumericalFailure,
};

struct SolveOptions {
#ifdef CLQR_USE_FLOAT
  Scalar tolerance = 1e-5f;
#else
  Scalar tolerance = 1e-9;
#endif
};

struct VectorView {
  Scalar* data = nullptr;
  std::size_t size = 0;

  Scalar& operator[](std::size_t i) { return data[i]; }
  const Scalar& operator[](std::size_t i) const { return data[i]; }
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
  Scalar objective = Scalar{0};
};

class Workspace;
class FactorizationWorkspace;

class Factorization {
 public:
  struct Impl;
  struct ImplDeleter {
    bool owns_memory = true;
    void operator()(Impl* impl) const noexcept;
  };

  Factorization();
  ~Factorization();
  Factorization(Factorization&&) noexcept;
  Factorization& operator=(Factorization&&) noexcept;
  Factorization(const Factorization&) = delete;
  Factorization& operator=(const Factorization&) = delete;

  SolveStatus status() const;
  const char* message() const;
  std::size_t stage_count() const;
  std::size_t RequiredSolveBytes() const;

 private:
  explicit Factorization(Impl* impl, bool owns_memory);

  std::unique_ptr<Impl, ImplDeleter> impl_;

  friend Factorization Factor(const Problem&, const SolveOptions&);
  friend Factorization Factor(const Problem&, FactorizationWorkspace&,
                              const SolveOptions&);
  friend SolutionView Solve(const Factorization&, const SolveRhs&, Workspace&);
};

Factorization Factor(const Problem& problem,
                     const SolveOptions& options = SolveOptions{});

class FactorizationWorkspace {
 public:
  FactorizationWorkspace() = default;
  FactorizationWorkspace(void* memory, std::size_t bytes) {
    mem_assign(memory, bytes);
  }
  FactorizationWorkspace(const FactorizationWorkspace&) = delete;
  FactorizationWorkspace& operator=(const FactorizationWorkspace&) = delete;
  FactorizationWorkspace(FactorizationWorkspace&&) = delete;
  FactorizationWorkspace& operator=(FactorizationWorkspace&&) = delete;

  static std::size_t num_bytes(const Problem& problem,
                               const SolveOptions& options = SolveOptions{});
  static constexpr std::size_t num_bytes(
      std::size_t stages, std::size_t state_dim, std::size_t control_dim,
      std::size_t mixed_constraints_per_stage = 0,
      std::size_t state_constraints_per_stage = 0,
      std::size_t terminal_constraints = 0);

  void reserve(const Problem& problem,
               const SolveOptions& options = SolveOptions{});
  std::size_t mem_assign(const Problem& problem, unsigned char* memory,
                         const SolveOptions& options = SolveOptions{});
  std::size_t mem_assign(void* memory, std::size_t bytes);

  unsigned char* data() { return data_; }
  const unsigned char* data() const { return data_; }
  std::size_t size() const { return size_; }
  std::size_t used() const { return arena_.used(); }
  bool owns_memory() const { return external_ == nullptr; }

 private:
  void ResetArena();

  std::vector<unsigned char> owned_;
  unsigned char* external_ = nullptr;
  unsigned char* data_ = nullptr;
  std::size_t size_ = 0;
  WorkspaceArena arena_;

  friend Factorization Factor(const Problem&, FactorizationWorkspace&,
                              const SolveOptions&);
};

Factorization Factor(const Problem& problem, FactorizationWorkspace& workspace,
                     const SolveOptions& options = SolveOptions{});

class Workspace {
 public:
  Workspace() = default;
  Workspace(void* memory, std::size_t bytes) {
    UseExternalMemory(memory, bytes);
  }

  static std::size_t RequiredBytes(const Problem& problem);
  static std::size_t RequiredBytes(const Problem& problem,
                                   const SolveOptions& options);
  static std::size_t RequiredBytes(const Factorization& factorization);
  static std::size_t num_bytes(const Problem& problem) {
    return RequiredBytes(problem);
  }
  static std::size_t num_bytes(const Problem& problem,
                               const SolveOptions& options) {
    return RequiredBytes(problem, options);
  }
  static std::size_t num_bytes(const Factorization& factorization) {
    return RequiredBytes(factorization);
  }
  static constexpr std::size_t num_bytes(std::size_t stages,
                                         std::size_t state_dim,
                                         std::size_t control_dim) {
    return RequiredBytesUniform(stages, state_dim, control_dim);
  }
  static constexpr std::size_t num_bytes(
      std::size_t stages, std::size_t state_dim, std::size_t control_dim,
      std::size_t mixed_constraints_per_stage,
      std::size_t state_constraints_per_stage = 0,
      std::size_t terminal_constraints = 0) {
    return RequiredBytesUniformConstrained(
        stages, state_dim, control_dim, mixed_constraints_per_stage,
        state_constraints_per_stage, terminal_constraints);
  }
  static constexpr std::size_t RequiredBytesUniform(std::size_t stages,
                                                    std::size_t state_dim,
                                                    std::size_t control_dim) {
    return RequiredBytesUniformWithTerminal(stages, state_dim, state_dim,
                                            control_dim);
  }
  static constexpr std::size_t RequiredBytesUniformWithTerminal(
      std::size_t stages, std::size_t state_dim, std::size_t terminal_state_dim,
      std::size_t control_dim) {
    const std::size_t largest_dimension =
        Max(Max(state_dim, terminal_state_dim), control_dim);
    if (!WorkspaceBoundInputsSafe(stages, largest_dimension))
      return std::numeric_limits<std::size_t>::max();
    const std::size_t total_state_scalars =
        stages * state_dim + terminal_state_dim;
    const std::size_t total_control_scalars = stages * control_dim;
    const std::size_t total_dynamics_scalars =
        stages == 0 ? 0 : (stages - 1) * state_dim + terminal_state_dim;
    const std::size_t total_p = total_state_scalars;
    const std::size_t total_P = stages * state_dim * state_dim +
                                terminal_state_dim * terminal_state_dim;
    const std::size_t total_K = stages * control_dim * state_dim;
    const std::size_t total_k = total_control_scalars;
    const std::size_t max_state =
        state_dim > terminal_state_dim ? state_dim : terminal_state_dim;
    const std::size_t max_control = control_dim;

    std::size_t bytes = 0;
    bytes = AddAligned(bytes, alignof(std::size_t),
                       sizeof(std::size_t) * (stages + 1));
    bytes =
        AddAligned(bytes, alignof(std::size_t), sizeof(std::size_t) * stages);
    bytes = AddAligned(bytes, alignof(std::size_t),
                       sizeof(std::size_t) * (stages + 1));
    bytes = AddAligned(bytes, alignof(std::size_t),
                       sizeof(std::size_t) * (stages + 1));
    bytes =
        AddAligned(bytes, alignof(std::size_t), sizeof(std::size_t) * stages);
    bytes =
        AddAligned(bytes, alignof(std::size_t), sizeof(std::size_t) * stages);
    bytes = AddAligned(bytes, alignof(Scalar), sizeof(Scalar) * total_P);
    bytes = AddAligned(bytes, alignof(Scalar), sizeof(Scalar) * total_p);
    bytes = AddAligned(bytes, alignof(Scalar), sizeof(Scalar) * total_K);
    bytes = AddAligned(bytes, alignof(Scalar), sizeof(Scalar) * total_k);
    bytes = AddAligned(bytes, alignof(Scalar), sizeof(Scalar) * max_state);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * max_state * max_state);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * max_control * max_control);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * max_state * max_control);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * max_state * max_state);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * max_control * max_state);
    bytes = AddAligned(bytes, alignof(Scalar), sizeof(Scalar) * max_state);
    bytes = AddAligned(bytes, alignof(Scalar), sizeof(Scalar) * max_control);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * max_control * max_control);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * max_control * max_state);
    bytes = AddAligned(bytes, alignof(Scalar), sizeof(Scalar) * max_control);
    bytes = AddAligned(bytes, alignof(std::size_t),
                       sizeof(std::size_t) * max_control);

    bytes = AddAligned(bytes, alignof(VectorView),
                       sizeof(VectorView) * (stages + 1));
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * total_state_scalars);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * total_control_scalars);
    bytes = AddAligned(bytes, alignof(Scalar), sizeof(Scalar) * state_dim);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * total_dynamics_scalars);
    return bytes;
  }
  static constexpr std::size_t RequiredBytesUniformConstrained(
      std::size_t stages, std::size_t state_dim, std::size_t control_dim,
      std::size_t mixed_constraints_per_stage,
      std::size_t state_constraints_per_stage = 0,
      std::size_t terminal_constraints = 0) {
    const std::size_t largest_dimension =
        Max(Max(state_dim, control_dim),
            Max(Max(mixed_constraints_per_stage, state_constraints_per_stage),
                terminal_constraints));
    if (!WorkspaceBoundInputsSafe(stages, largest_dimension))
      return std::numeric_limits<std::size_t>::max();
    const std::size_t total_state_scalars = (stages + 1) * state_dim;
    const std::size_t total_control_scalars = stages * control_dim;
    const std::size_t total_dynamics_scalars = stages * state_dim;
    const std::size_t total_mixed_scalars =
        stages * mixed_constraints_per_stage;
    const std::size_t total_state_multiplier_scalars =
        stages * state_constraints_per_stage;
    const std::size_t mixed_rows_bound =
        mixed_constraints_per_stage + state_dim;
    const std::size_t state_rows_bound = Max(
        terminal_constraints, state_constraints_per_stage + mixed_rows_bound);
    const std::size_t state_pivot_bound = Min(state_dim, state_rows_bound);
    const std::size_t mixed_stage_ops = stages;
    const std::size_t mixed_stage_scalars =
        2 * mixed_rows_bound *
            (control_dim + state_dim + 1 + mixed_rows_bound) +
        control_dim * state_dim + control_dim * control_dim + control_dim +
        control_dim * mixed_rows_bound + mixed_rows_bound * state_dim +
        mixed_rows_bound + mixed_rows_bound * mixed_rows_bound +
        state_dim * state_dim + control_dim * control_dim +
        state_dim * control_dim + state_dim * control_dim + state_dim +
        control_dim + state_dim + 8 * state_dim * state_dim +
        4 * control_dim * control_dim + 6 * state_dim * control_dim +
        2 * state_dim * state_dim + 2 * state_dim * control_dim +
        3 * state_dim + 4 * state_dim + 4 * control_dim +
        (state_rows_bound + mixed_rows_bound) * state_dim + state_rows_bound +
        mixed_rows_bound + 2 * control_dim * state_dim +
        2 * control_dim * control_dim + 3 * control_dim;
    const std::size_t state_stage_scalars =
        2 * state_rows_bound * (state_dim + 1 + state_rows_bound) +
        state_dim * state_dim + state_dim + state_dim * state_rows_bound +
        3 * state_dim * state_dim + 3 * state_dim * control_dim +
        4 * state_dim + 8 * state_dim * state_dim +
        4 * state_dim * control_dim + 2 * control_dim * state_dim +
        4 * state_dim + 2 * control_dim +
        (state_pivot_bound + mixed_constraints_per_stage) *
            (state_dim + control_dim + 1) +
        3 * state_dim * state_dim + 2 * state_dim +
        2 * control_dim * state_dim + control_dim * control_dim + control_dim;
    const std::size_t pullback_stage_scalars =
        40 * (2 * state_dim + control_dim + mixed_constraints_per_stage +
              state_constraints_per_stage + terminal_constraints + 1);
    std::size_t bytes = 0;
    bytes = AddAligned(bytes, alignof(Stage), sizeof(Stage) * stages);
    bytes =
        AddAligned(bytes, alignof(Vector), sizeof(Vector) * (10 * stages + 2));
    bytes =
        AddAligned(bytes, alignof(Matrix), sizeof(Matrix) * (8 * stages + 2));
    bytes = AddAligned(bytes, alignof(std::size_t),
                       sizeof(std::size_t) * (6 * stages + 3));
    bytes =
        AddAligned(bytes, alignof(Scalar), sizeof(Scalar) * (80 * stages + 32));
    bytes = AddAligned(
        bytes, alignof(Scalar),
        sizeof(Scalar) * stages *
            (state_dim * state_dim + state_dim * control_dim + state_dim +
             state_dim * state_dim + control_dim * control_dim +
             state_dim * control_dim + state_dim + control_dim +
             mixed_constraints_per_stage * state_dim +
             mixed_constraints_per_stage * control_dim +
             mixed_constraints_per_stage +
             state_constraints_per_stage * state_dim +
             state_constraints_per_stage));
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * (state_dim * state_dim + state_dim +
                                         terminal_constraints * state_dim +
                                         terminal_constraints));
    bytes = AddAligned(
        bytes, alignof(Scalar),
        sizeof(Scalar) * ((stages + 1) * (state_dim * state_dim + state_dim) +
                          stages * (control_dim * state_dim +
                                    control_dim * control_dim + control_dim)));
    bytes =
        AddAligned(bytes, alignof(Scalar),
                   sizeof(Scalar) *
                       ((stages + 1) * (2 * state_rows_bound * (state_dim + 1) +
                                        state_dim * state_dim + state_dim) +
                        stages * state_stage_scalars +
                        mixed_stage_ops * mixed_stage_scalars));
    bytes =
        AddAligned(bytes, alignof(std::size_t),
                   sizeof(std::size_t) *
                       ((stages + 1) * 3 * state_dim +
                        mixed_stage_ops * (3 * control_dim + mixed_rows_bound +
                                           3 * state_dim)));
    bytes =
        AddAligned(bytes, alignof(Matrix), sizeof(Matrix) * (4 * stages + 2));
    bytes =
        AddAligned(bytes, alignof(Vector), sizeof(Vector) * (4 * stages + 2));
    bytes = AddAligned(bytes, alignof(std::size_t),
                       RequiredBytesUniform(stages, state_dim, control_dim));
    bytes = AddAligned(bytes, alignof(VectorView),
                       sizeof(VectorView) * (stages + 1));
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(VectorView), sizeof(VectorView) * stages);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * total_state_scalars);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * total_control_scalars);
    bytes = AddAligned(bytes, alignof(Scalar), sizeof(Scalar) * state_dim);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * total_dynamics_scalars);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * total_mixed_scalars);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * total_state_multiplier_scalars);
    bytes = AddAligned(bytes, alignof(Scalar),
                       sizeof(Scalar) * terminal_constraints);
    bytes =
        AddAligned(bytes, alignof(Vector), sizeof(Vector) * (2 * stages + 1));
    bytes = AddAligned(
        bytes, alignof(Scalar),
        sizeof(Scalar) *
            (total_dynamics_scalars + total_state_scalars +
             stages * pullback_stage_scalars +
             4 * state_dim * (state_dim + terminal_constraints + 1)));
    if (stages == 0) {
      // Match the runtime constrained-workspace bound for the terminal-only
      // recovery path.  With no stage-proportional scratch, its dense affine
      // products and rectangular multiplier QR coexist in the arena.
      const std::size_t local_dimension = state_dim + terminal_constraints + 1;
      bytes =
          AddAligned(bytes, alignof(Scalar),
                     sizeof(Scalar) * 8 * local_dimension * local_dimension);
    }
    bytes =
        AddAligned(bytes, alignof(Vector), sizeof(Vector) * (5 * stages + 1));
    bytes = AddAligned(
        bytes, alignof(Scalar),
        sizeof(Scalar) * (total_state_scalars + total_control_scalars +
                          total_dynamics_scalars + total_mixed_scalars +
                          total_state_multiplier_scalars +
                          terminal_constraints + state_dim));
    return bytes;
  }

  void Reserve(const Problem& problem);
  void Reserve(const Problem& problem, const SolveOptions& options);
  void Reserve(const Factorization& factorization);
  void reserve(const Problem& problem) { Reserve(problem); }
  void reserve(const Problem& problem, const SolveOptions& options) {
    Reserve(problem, options);
  }
  void reserve(const Factorization& factorization) { Reserve(factorization); }
  std::size_t mem_assign(const Problem& problem, unsigned char* memory) {
    const std::size_t bytes = RequiredBytes(problem);
    UseExternalMemory(memory, bytes);
    return bytes;
  }
  std::size_t mem_assign(const Problem& problem, const SolveOptions& options,
                         unsigned char* memory) {
    const std::size_t bytes = RequiredBytes(problem, options);
    UseExternalMemory(memory, bytes);
    return bytes;
  }
  std::size_t mem_assign(const Factorization& factorization,
                         unsigned char* memory) {
    const std::size_t bytes = RequiredBytes(factorization);
    UseExternalMemory(memory, bytes);
    return bytes;
  }
  std::size_t mem_assign(void* memory, std::size_t bytes) {
    UseExternalMemory(memory, bytes);
    return bytes;
  }
  void UseExternalMemory(void* memory, std::size_t bytes);
  unsigned char* data() { return data_; }
  const unsigned char* data() const { return data_; }
  std::size_t size() const { return size_; }
  bool owns_memory() const { return external_ == nullptr; }
  WorkspaceArena& arena() { return arena_; }
  const WorkspaceArena& arena() const { return arena_; }

 private:
  const char* StoreMessage(const char* message);
  const char* StoreDiagnostic(const char* diagnostic);

  static constexpr std::size_t Align(std::size_t offset,
                                     std::size_t alignment) {
    if (offset > std::numeric_limits<std::size_t>::max() - (alignment - 1))
      return std::numeric_limits<std::size_t>::max();
    return (offset + alignment - 1) & ~(alignment - 1);
  }
  static constexpr std::size_t AddAligned(std::size_t offset,
                                          std::size_t alignment,
                                          std::size_t bytes) {
    const std::size_t aligned = Align(offset, alignment);
    if (aligned == std::numeric_limits<std::size_t>::max() ||
        bytes > std::numeric_limits<std::size_t>::max() - aligned)
      return std::numeric_limits<std::size_t>::max();
    return aligned + bytes;
  }
  static constexpr std::size_t Min(std::size_t a, std::size_t b) {
    return a < b ? a : b;
  }
  static constexpr std::size_t Max(std::size_t a, std::size_t b) {
    return a > b ? a : b;
  }
  static constexpr bool WorkspaceBoundInputsSafe(
      std::size_t stages, std::size_t largest_dimension) {
    constexpr std::size_t kCoefficientMargin = 1024;
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (stages == maximum || largest_dimension == maximum) return false;
    const std::size_t stage_count = stages + 1;
    const std::size_t dimension = largest_dimension + 1;
    if (dimension > maximum / dimension) return false;
    const std::size_t dimension_square = dimension * dimension;
    if (stage_count > maximum / dimension_square) return false;
    const std::size_t problem_scale = stage_count * dimension_square;
    return problem_scale <= maximum / kCoefficientMargin;
  }

  std::vector<unsigned char> owned_;
  unsigned char* external_ = nullptr;
  unsigned char* data_ = nullptr;
  std::size_t size_ = 0;
  WorkspaceArena arena_;
  std::array<char, 128> message_{};
  std::array<char, 256> diagnostic_{};

  friend SolutionView Solve(const Problem&, Workspace&, const SolveOptions&);
  friend SolutionView Solve(const Factorization&, const SolveRhs&, Workspace&);
};

constexpr std::size_t FactorizationWorkspace::num_bytes(
    std::size_t stages, std::size_t state_dim, std::size_t control_dim,
    std::size_t mixed_constraints_per_stage,
    std::size_t state_constraints_per_stage, std::size_t terminal_constraints) {
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  const std::size_t largest_dimension =
      state_dim > control_dim ? state_dim : control_dim;
  const std::size_t largest_constraint =
      mixed_constraints_per_stage > state_constraints_per_stage
          ? (mixed_constraints_per_stage > terminal_constraints
                 ? mixed_constraints_per_stage
                 : terminal_constraints)
          : (state_constraints_per_stage > terminal_constraints
                 ? state_constraints_per_stage
                 : terminal_constraints);
  const std::size_t largest = largest_dimension > largest_constraint
                                  ? largest_dimension
                                  : largest_constraint;
  if (stages == maximum || largest == maximum) return maximum;
  const std::size_t stage_count = stages + 1;
  const std::size_t dimension = largest + 1;
  if (dimension > maximum / dimension) return maximum;
  const std::size_t dimension_square = dimension * dimension;
  if (stage_count > maximum / dimension_square) return maximum;
  const std::size_t scale = stage_count * dimension_square;
  const bool constrained = mixed_constraints_per_stage != 0 ||
                           state_constraints_per_stage != 0 ||
                           terminal_constraints != 0;
  const std::size_t base =
      constrained
          ? Workspace::RequiredBytesUniformConstrained(
                stages, state_dim, control_dim, mixed_constraints_per_stage,
                state_constraints_per_stage, terminal_constraints)
          : Workspace::RequiredBytesUniform(stages, state_dim, control_dim);
  if (base == maximum) return maximum;
  // All explicit storage terms below have a coefficient below 256 relative
  // to stage_count * (largest_dimension + 1)^2.
  if (scale > maximum / 256 / sizeof(Scalar)) return maximum;

  const std::size_t n = state_dim;
  const std::size_t m = control_dim;
  const std::size_t p = mixed_constraints_per_stage;
  const std::size_t e = state_constraints_per_stage;
  const std::size_t t = terminal_constraints;
  const std::size_t cache_stage_scalars = 3 * n * n + 4 * n * m + 2 * m * m;
  std::size_t extra_scalars = stages * cache_stage_scalars + n * n;
  std::size_t extra_indices = stages * m + 2 * stages + 1;
  if (constrained) {
    const std::size_t matrix_only_stage_scalars =
        n * n + 2 * n * m + m * m + p * (n + m) + e * n + 2 * (n + m + p + e);
    const std::size_t matrix_only_node_scalars = n * n + 2 * n;
    const std::size_t replay_stage_scalars =
        11 * n * n + 8 * n * m + m * m;
    extra_scalars +=
        stages * (matrix_only_stage_scalars + replay_stage_scalars) +
        stage_count * matrix_only_node_scalars + t * n + 2 * t + 2 * n;
    if (t != 0) {
      const std::size_t rank = n < t ? n : t;
      extra_scalars += 8 * n * n + 6 * n + 2 * t * (n + 1) +
                       rank * t + 2 * rank + t;
      extra_indices += 2 * n + 2 * rank;
    }
    if (stages == 0) {
      extra_scalars +=
          n * (n + t) + 2 * n * (n + t + 1) + n * n + 3 * n;
      extra_indices += t + 2 * n;
    }
  }
  if (extra_scalars > maximum / sizeof(Scalar)) return maximum;
  const std::size_t scalar_bytes = extra_scalars * sizeof(Scalar);
  if (extra_indices > maximum / sizeof(std::size_t)) return maximum;
  const std::size_t index_bytes = extra_indices * sizeof(std::size_t);
  constexpr std::size_t kMetadataBytesPerStage = 1536;
  if (stage_count > maximum / kMetadataBytesPerStage) return maximum;
  const std::size_t metadata_bytes = stage_count * kMetadataBytesPerStage;
  if (scalar_bytes > maximum - index_bytes) return maximum;
  const std::size_t data_bytes = scalar_bytes + index_bytes;
  if (data_bytes > maximum - metadata_bytes) return maximum;
  const std::size_t extra = data_bytes + metadata_bytes;
  if (base > maximum - extra) return maximum;
  return base + extra;
}

using SolveWorkspace = Workspace;

SolutionView Solve(const Problem& problem, Workspace& workspace,
                   const SolveOptions& options = SolveOptions{});
SolutionView Solve(const Factorization& factorization, const SolveRhs& rhs,
                   Workspace& workspace);
const char* StatusName(SolveStatus status);

}  // namespace clqr

#endif  // CLQR_CLQR_H_
