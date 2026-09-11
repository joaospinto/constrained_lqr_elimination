#include "../benchmarks/scaling_problem.h"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#endif
#include "../src/clqr.cc"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <iostream>

namespace {
using namespace clqr;

void Expect(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void Equal(const Matrix& a, const Matrix& b) {
  Expect(a.rows() == b.rows() && a.cols() == b.cols() && a.data() == b.data(),
         "scratch changed a matrix");
}
void Equal(const Vector& a, const Vector& b) {
  Expect(a.data() == b.data(), "scratch changed a vector");
}

Problem RankCase(std::size_t n, std::size_t m, std::size_t rows,
                 std::size_t control_rank, std::size_t state_rank,
                 std::size_t terminal_rank, bool heterogeneous) {
  Problem p;
  p.initial_state = Vector(n);
  p.stages.resize(3);
  p.Q.resize(4);
  p.q.resize(4);
  for (std::size_t i = 0; i < 4; ++i) {
    const std::size_t ni = heterogeneous && i % 2 ? n + 2 : n;
    p.Q[i] = Identity(ni);
    p.q[i] = Vector(ni);
    if (i == 3) break;
    const std::size_t next = heterogeneous && (i + 1) % 2 ? n + 2 : n;
    Stage& s = p.stages[i];
    s.A = Matrix(next, ni);
    for (std::size_t j = 0; j < std::min(ni, next); ++j)
      s.A(j, j) = Scalar{0.5};
    s.B = Matrix(next, m);
    s.c = Vector(next);
    s.R = Identity(m);
    s.M = Matrix(ni, m);
    s.r = Vector(m);
    s.C = Matrix(rows, ni);
    s.D = Matrix(rows, m);
    s.d = Vector(rows);
    s.E = Matrix(rows, ni);
    s.e = Vector(rows);
    for (std::size_t j = 0; j < rows; ++j) {
      if (j < control_rank && j < m) s.D(j, j) = Scalar{1};
      if (ni && j >= control_rank) s.C(j, j % ni) = Scalar{0.25};
      if (j < state_rank && j < ni) s.E(j, j) = Scalar{1};
    }
  }
  p.terminal_E = Matrix(rows, p.Q.back().rows());
  p.terminal_e = Vector(rows);
  for (std::size_t j = 0; j < std::min(rows, terminal_rank); ++j)
    if (j < p.Q.back().rows()) p.terminal_E(j, j) = Scalar{1};
  return p;
}

void CheckLayout(const Problem& p) {
  auto expected = Initialize(p);
  NewtonKktDiagnostics diagnostics;
  std::string error;
  EliminateConstraintsRightToLeft(expected, SolveOptions{}.tolerance, &error,
                                  &diagnostics);
  std::size_t bytes = 0;
  AddEliminationStorageBound(p, &bytes);
  std::vector<std::max_align_t> storage(
      (bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t) + 2);
  auto* memory = reinterpret_cast<unsigned char*>(storage.data());
  std::memset(memory + bytes, 0xa5, 16);
  WorkspaceArena arena(memory, bytes);
  {
    ScopedWorkspaceArena active(&arena);
    const auto actual = InitializeAndEliminate(p, SolveOptions{}.tolerance,
                                               &error, &diagnostics);
    for (std::size_t i = 0; i <= p.stages.size(); ++i) {
      Equal(actual.problem.Q[i], expected.problem.Q[i]);
      Equal(actual.problem.q[i], expected.problem.q[i]);
      Equal(actual.state_maps[i].linear, expected.state_maps[i].linear);
      Equal(actual.state_maps[i].offset, expected.state_maps[i].offset);
      Equal(actual.state_bases[i].T, expected.state_bases[i].T);
      Equal(actual.state_bases[i].rhs_operations.reflectors,
            expected.state_bases[i].rhs_operations.reflectors);
      if (i == p.stages.size()) continue;
      const auto& a = actual.problem.stages[i];
      const auto& b = expected.problem.stages[i];
      Equal(a.A, b.A);
      Equal(a.B, b.B);
      Equal(a.c, b.c);
      Equal(a.R, b.R);
      Equal(a.M, b.M);
      Equal(a.r, b.r);
      Equal(actual.control_maps[i].state_linear,
            expected.control_maps[i].state_linear);
      Equal(actual.control_maps[i].control_linear,
            expected.control_maps[i].control_linear);
      Equal(actual.elimination_traces[i].mixed.rhs_operations.reflectors,
            expected.elimination_traces[i].mixed.rhs_operations.reflectors);
    }
  }
  for (std::size_t i = 0; i < 16; ++i)
    Expect(memory[bytes + i] == 0xa5, "elimination crossed its reservation");

  Workspace workspace;
  workspace.Reserve(p);
  const auto result = Solve(p, workspace);
  if (result.status != SolveStatus::kOptimal) {
    std::cerr << "N=" << p.stages.size() << " n=" << p.initial_state.size()
              << ": " << result.message << '\n';
    throw std::runtime_error("rank-pattern workspace solve failed");
  }

  FactorizationWorkspace factor_workspace;
  factor_workspace.reserve(p);
  const auto factors = Factor(p, factor_workspace);
  Expect(factors.status() == SolveStatus::kOptimal,
         "rank-pattern factorization failed");
  Workspace factored_workspace;
  factored_workspace.reserve(factors);
  const auto rhs = ExtractRhs(p);
  for (int repeat = 0; repeat < 2; ++repeat) {
    const auto factored = Solve(factors, rhs, factored_workspace);
    if (factored.status != SolveStatus::kOptimal)
      std::cerr << "cached N=" << p.stages.size()
                << " n=" << p.initial_state.size() << ": " << factored.message
                << '\n';
    Expect(factored.status == SolveStatus::kOptimal,
           "rank-pattern cached solve failed");
  }
}
}  // namespace

int main() {
  using namespace clqr;
  // Check the constant-time maximization against exhaustive pivot counts,
  // including both rank-capacity breakpoints and zero-dimensional controls.
  for (std::size_t n = 0; n <= 12; ++n)
    for (std::size_t m = 0; m <= 12; ++m)
      for (std::size_t p = 0; p <= 12; ++p)
        for (std::size_t e = 0; e <= 12; ++e) {
          const std::size_t bound =
              detail::EliminatedStageBound(n, 13, m, p, e, 13);
          for (std::size_t a = 0; a <= 13; ++a)
            Expect(detail::EliminatedStageBytes(n, 13, m, p, e, a) <= bound,
                   "rank envelope missed an interior maximum");
        }
  std::size_t cases = 0;
  for (std::size_t n : {1, 2, 4, 8})
    for (std::size_t m : {0, 1, 3, 8})
      for (std::size_t rows : {0, 1, 5, 17})
        for (std::size_t rank : {0, 1, 4, 8}) {
          CheckLayout(RankCase(n, m, rows, std::min(m, rank), std::min(n, rank),
                               std::min(n, rank), true));
          ++cases;
        }
  for (std::size_t n : {0, 1, 8})
    for (std::size_t rows : {0, 1, 64}) {
      auto p = RankCase(n, 0, rows, 0, 0, n, false);
      p.stages.clear();
      p.Q.resize(1);
      p.q.resize(1);
      CheckLayout(p);
      ++cases;
    }
  // Real dense fixtures and long horizons; only modest allocations locally.
#ifndef CLQR_USE_FLOAT
  for (std::size_t n : {8, 16, 32, 64}) {
    const auto p =
        benchmark::MakeScalingProblem(128, n, n / 2, n / 8, n / 4).problem;
    CheckLayout(p);
    Workspace workspace;
    workspace.Reserve(p);
    const auto result = Solve(p, workspace);
    Expect(result.status == SolveStatus::kOptimal, "scaling solve failed");
    std::cout << "N=128 n=" << n << " reserved_bytes=" << workspace.size()
              << " retained_bytes=" << workspace.arena().used() << '\n';
  }
  // Check linear growth after propagated rank bounds saturate. Calculate the
  // large reservation from this affine size law without allocating that case.
  std::size_t previous_solve = 0, previous_factor = 0, previous_cached = 0;
  std::size_t solve_step = 0, factor_step = 0, cached_step = 0;
  for (std::size_t N : {32, 64, 96}) {
    const auto p = benchmark::MakeScalingProblem(N, 64, 32, 8, 16).problem;
    const auto solve = Workspace::num_bytes(p);
    const auto factor = FactorizationWorkspace::num_bytes(p);
    const auto cached = ConstrainedFactoredSolveRequiredBytes(p);
    if (N == 96) {
      Expect(solve - previous_solve == solve_step &&
                 factor - previous_factor == factor_step &&
                 cached - previous_cached == cached_step,
             "workspace reservation is not affine in the horizon");
      std::cout << "N=32768 n=64 calculated_runtime_solve_bytes="
                << solve + ((32768 - N) / 32) * solve_step
                << " factor_bytes=" << factor + ((32768 - N) / 32) * factor_step
                << " cached_solve_bytes="
                << cached + ((32768 - N) / 32) * cached_step << '\n';
    }
    solve_step = solve - previous_solve;
    factor_step = factor - previous_factor;
    cached_step = cached - previous_cached;
    previous_solve = solve;
    previous_factor = factor;
    previous_cached = cached;
  }
#endif
  std::cout << cases
            << " rank/shape layouts checked; uniform largest-case bound="
            << Workspace::num_bytes(32768, 64, 32, 8, 16, 16) << " bytes\n";
}
