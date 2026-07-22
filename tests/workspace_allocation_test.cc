#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>
#include <vector>

#include "clqr/clqr.h"

namespace {

std::atomic<bool> g_count_allocations{false};
std::atomic<std::size_t> g_allocations{0};
std::atomic<std::size_t> g_bytes{0};

void CountAllocation(std::size_t size) {
  if (g_count_allocations.load(std::memory_order_relaxed)) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    g_bytes.fetch_add(size, std::memory_order_relaxed);
  }
}

void* Allocate(std::size_t size) {
  CountAllocation(size);
  if (void* ptr = std::malloc(size)) return ptr;
  throw std::bad_alloc();
}

void ResetCounters() {
  g_allocations.store(0, std::memory_order_relaxed);
  g_bytes.store(0, std::memory_order_relaxed);
}

void StartCounting() {
  ResetCounters();
  g_count_allocations.store(true, std::memory_order_relaxed);
}

void StopCounting() {
  g_count_allocations.store(false, std::memory_order_relaxed);
}

using clqr::Matrix;
using clqr::Problem;
using clqr::Scalar;
using clqr::SolutionView;
using clqr::Solve;
using clqr::SolveStatus;
using clqr::Stage;
using clqr::Vector;
using clqr::Workspace;

Problem MakeProblem(std::size_t stages, std::size_t state_dim,
                    std::size_t control_dim) {
  Problem problem;
  problem.initial_state = Vector(state_dim);
  for (std::size_t i = 0; i < state_dim; ++i) {
    problem.initial_state[i] = 0.1 * static_cast<Scalar>(i + 1);
  }
  problem.stages.resize(stages);
  for (std::size_t i = 0; i < stages; ++i) {
    Stage& stage = problem.stages[i];
    stage.A = Matrix(state_dim, state_dim);
    for (std::size_t row = 0; row < state_dim; ++row) {
      for (std::size_t col = 0; col < state_dim; ++col) {
        stage.A(row, col) =
            row == col ? 0.9 : 0.02 * static_cast<Scalar>(row + col + 1);
      }
    }
    stage.B = Matrix(state_dim, control_dim);
    for (std::size_t row = 0; row < state_dim; ++row) {
      for (std::size_t col = 0; col < control_dim; ++col) {
        stage.B(row, col) = 0.03 * static_cast<Scalar>((row + 1) * (col + 1));
      }
    }
    stage.c = Vector(state_dim);
    stage.Q = Matrix(state_dim, state_dim);
    for (std::size_t row = 0; row < state_dim; ++row) stage.Q(row, row) = 1.0;
    stage.R = Matrix(control_dim, control_dim);
    for (std::size_t row = 0; row < control_dim; ++row) stage.R(row, row) = 2.0;
    stage.M = Matrix(state_dim, control_dim);
    stage.q = Vector(state_dim);
    stage.r = Vector(control_dim);
    stage.C = Matrix(0, state_dim);
    stage.D = Matrix(0, control_dim);
    stage.d = Vector(0);
    stage.E = Matrix(0, state_dim);
    stage.e = Vector(0);
  }
  problem.terminal_Q = Matrix(state_dim, state_dim);
  for (std::size_t row = 0; row < state_dim; ++row)
    problem.terminal_Q(row, row) = 1.5;
  problem.terminal_q = Vector(state_dim);
  problem.terminal_E = Matrix(0, state_dim);
  problem.terminal_e = Vector(0);
  return problem;
}

Problem MakeConstrainedProblem(std::size_t stages, std::size_t state_dim,
                               std::size_t control_dim) {
  Problem problem = MakeProblem(stages, state_dim, control_dim);
  for (std::size_t i = 0; i < stages; ++i) {
    Stage& stage = problem.stages[i];
    stage.C = Matrix(1, state_dim);
    stage.D = Matrix(1, control_dim);
    stage.D(0, 0) = 1.0;
    stage.d = Vector(1);
  }
  return problem;
}

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

}  // namespace

void* operator new(std::size_t size) { return Allocate(size); }
void* operator new[](std::size_t size) { return Allocate(size); }
void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete[](void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete[](void* ptr, std::size_t) noexcept { std::free(ptr); }

int main() {
  Problem problem = MakeProblem(64, 6, 3);

  Workspace owned;
  StartCounting();
  owned.Reserve(problem);
  SolutionView owned_solution = Solve(problem, owned);
  StopCounting();
  const std::size_t owned_allocations =
      g_allocations.load(std::memory_order_relaxed);
  const std::size_t owned_bytes = g_bytes.load(std::memory_order_relaxed);
  Expect(owned_solution.status == SolveStatus::kOptimal,
         "owned workspace solve status");
  Expect(owned_allocations == 1,
         "owned workspace should allocate exactly once");

  std::vector<unsigned char> memory(Workspace::RequiredBytes(problem));
  Workspace external(memory.data(), memory.size());
  StartCounting();
  SolutionView external_solution = Solve(problem, external);
  StopCounting();
  const std::size_t external_allocations =
      g_allocations.load(std::memory_order_relaxed);
  const std::size_t external_bytes = g_bytes.load(std::memory_order_relaxed);
  Expect(external_solution.status == SolveStatus::kOptimal,
         "external workspace solve status");
  Expect(external_allocations == 0,
         "external workspace solve should not allocate");

  Problem constrained = MakeConstrainedProblem(16, 4, 2);
  constexpr std::size_t kConstrainedBytes =
      Workspace::RequiredBytesUniformConstrained(16, 4, 2, 1);
  static_assert(kConstrainedBytes > 0,
                "constrained workspace size must be positive");
  const std::size_t constrained_required =
      Workspace::RequiredBytes(constrained);
  Expect(Workspace::RequiredBytes(constrained) <= kConstrainedBytes,
         "constrained constexpr workspace byte count");
  Workspace constrained_owned;
  StartCounting();
  constrained_owned.Reserve(constrained);
  SolutionView constrained_owned_solution =
      Solve(constrained, constrained_owned);
  StopCounting();
  const std::size_t constrained_owned_allocations =
      g_allocations.load(std::memory_order_relaxed);
  const std::size_t constrained_owned_bytes =
      g_bytes.load(std::memory_order_relaxed);
  Expect(constrained_owned_solution.status == SolveStatus::kOptimal,
         "constrained owned workspace solve status");

  std::vector<unsigned char> constrained_memory(
      Workspace::RequiredBytes(constrained));
  Workspace constrained_external(constrained_memory.data(),
                                 constrained_memory.size());
  StartCounting();
  SolutionView constrained_external_solution =
      Solve(constrained, constrained_external);
  StopCounting();
  const std::size_t constrained_external_allocations =
      g_allocations.load(std::memory_order_relaxed);
  const std::size_t constrained_external_bytes =
      g_bytes.load(std::memory_order_relaxed);
  Expect(constrained_external_solution.status == SolveStatus::kOptimal,
         "constrained external workspace solve status");

  const std::size_t constrained_32 =
      Workspace::RequiredBytes(MakeConstrainedProblem(32, 4, 2));
  const std::size_t constrained_64 =
      Workspace::RequiredBytes(MakeConstrainedProblem(64, 4, 2));
  Expect(constrained_64 > constrained_32,
         "constrained workspace should grow with the horizon");
  Expect(constrained_64 <= 3 * constrained_32,
         "constrained workspace should scale linearly with the horizon");

  Problem terminal_constrained = MakeProblem(8, 3, 2);
  terminal_constrained.terminal_E = Matrix(1, 3);
  terminal_constrained.terminal_E(0, 0) = 1.0;
  terminal_constrained.terminal_e = Vector(1);
  const std::size_t terminal_required =
      Workspace::RequiredBytes(terminal_constrained);
  std::vector<unsigned char> terminal_memory(terminal_required);
  Workspace terminal_external(terminal_memory.data(), terminal_memory.size());
  StartCounting();
  SolutionView terminal_solution =
      Solve(terminal_constrained, terminal_external);
  StopCounting();
  const std::size_t terminal_external_allocations =
      g_allocations.load(std::memory_order_relaxed);
  const std::size_t terminal_external_bytes =
      g_bytes.load(std::memory_order_relaxed);
  Expect(terminal_solution.status == SolveStatus::kOptimal,
         "terminal constrained workspace solve status");

  std::cout << "owned_allocations=" << owned_allocations
            << " owned_bytes=" << owned_bytes
            << " external_allocations=" << external_allocations
            << " external_bytes=" << external_bytes
            << " constrained_required=" << constrained_required
            << " constrained_constexpr=" << kConstrainedBytes
            << " constrained_owned_allocations="
            << constrained_owned_allocations
            << " constrained_owned_bytes=" << constrained_owned_bytes
            << " constrained_external_allocations="
            << constrained_external_allocations
            << " constrained_external_bytes=" << constrained_external_bytes
            << " terminal_required=" << terminal_required
            << " terminal_external_allocations="
            << terminal_external_allocations
            << " terminal_external_bytes=" << terminal_external_bytes << "\n";
  Expect(constrained_owned_allocations == 1,
         "constrained owned workspace should allocate exactly once");
  Expect(constrained_external_allocations == 0,
         "constrained external workspace solve should not allocate");
  Expect(terminal_external_allocations == 0,
         "terminal constrained external workspace solve should not allocate");
  return 0;
}
