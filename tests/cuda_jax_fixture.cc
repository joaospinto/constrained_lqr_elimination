#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "clqr/cuda.h"
#include "cuda_jax_problem.h"

namespace {

using clqr::Matrix;
using clqr::Problem;
using clqr::Vector;

void PrintVector(const Vector& vector) {
  std::cout << '[';
  for (std::size_t i = 0; i < vector.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << vector[i];
  }
  std::cout << ']';
}

void PrintMatrix(const Matrix& matrix) {
  std::cout << '[';
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    if (row) std::cout << ',';
    std::cout << '[';
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
      if (col) std::cout << ',';
      std::cout << matrix(row, col);
    }
    std::cout << ']';
  }
  std::cout << ']';
}

template <typename Function>
void PrintSequence(std::size_t count, Function function) {
  std::cout << '[';
  for (std::size_t i = 0; i < count; ++i) {
    if (i) std::cout << ',';
    function(i);
  }
  std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) {
  const Problem problem = clqr::test::MakeJaxCrossValidationProblem();
  clqr::cuda::Solution solution;
  if (argc == 2 && std::string(argv[1]) == "--cpu") {
    clqr::Workspace workspace;
    workspace.Reserve(problem);
    const clqr::SolutionView cpu = clqr::Solve(problem, workspace);
    solution.status = cpu.status;
    solution.message = cpu.message;
    solution.states.resize(cpu.state_count);
    for (std::size_t i = 0; i < cpu.state_count; ++i) {
      solution.states[i] = Vector(cpu.states[i].size);
      for (std::size_t row = 0; row < cpu.states[i].size; ++row)
        solution.states[i][row] = cpu.states[i][row];
    }
    solution.controls.resize(cpu.control_count);
    for (std::size_t i = 0; i < cpu.control_count; ++i) {
      solution.controls[i] = Vector(cpu.controls[i].size);
      for (std::size_t row = 0; row < cpu.controls[i].size; ++row)
        solution.controls[i][row] = cpu.controls[i][row];
    }
  } else {
    if (!clqr::cuda::Available()) return 2;
    solution = clqr::cuda::Solve(problem);
  }
  if (solution.status != clqr::SolveStatus::kOptimal) {
    std::cerr << solution.message << '\n';
    return 1;
  }
  const std::size_t horizon = problem.stages.size();
  const std::size_t n = problem.initial_state.size();
  const std::size_t p = problem.stages.front().E.rows();
  std::cout << std::setprecision(
      std::numeric_limits<clqr::Scalar>::max_digits10);
  std::cout << "{\"precision\":\"" << clqr::kPrecisionName << "\",\"A\":";
  PrintSequence(horizon,
                [&](std::size_t i) { PrintMatrix(problem.stages[i].A); });
  std::cout << ",\"B\":";
  PrintSequence(horizon,
                [&](std::size_t i) { PrintMatrix(problem.stages[i].B); });
  std::cout << ",\"Q\":";
  PrintSequence(horizon + 1, [&](std::size_t i) {
    PrintMatrix(i == horizon ? problem.terminal_Q : problem.stages[i].Q);
  });
  std::cout << ",\"M\":";
  PrintSequence(horizon,
                [&](std::size_t i) { PrintMatrix(problem.stages[i].M); });
  std::cout << ",\"R\":";
  PrintSequence(horizon,
                [&](std::size_t i) { PrintMatrix(problem.stages[i].R); });
  std::cout << ",\"D\":";
  PrintSequence(horizon + 1, [&](std::size_t i) {
    if (i < horizon) {
      PrintMatrix(problem.stages[i].E);
    } else {
      PrintMatrix(Matrix(p, n));
    }
  });
  std::cout << ",\"E\":";
  PrintSequence(horizon, [&](std::size_t i) {
    PrintMatrix(Matrix(p, problem.stages[i].B.cols()));
  });
  std::cout << ",\"q\":";
  PrintSequence(horizon + 1, [&](std::size_t i) {
    PrintVector(i == horizon ? problem.terminal_q : problem.stages[i].q);
  });
  std::cout << ",\"r\":";
  PrintSequence(horizon,
                [&](std::size_t i) { PrintVector(problem.stages[i].r); });
  std::cout << ",\"c\":";
  PrintSequence(horizon + 1, [&](std::size_t i) {
    PrintVector(i == 0 ? problem.initial_state : problem.stages[i - 1].c);
  });
  std::cout << ",\"d\":";
  PrintSequence(horizon + 1, [&](std::size_t i) {
    if (i < horizon) {
      Vector rhs(p);
      for (std::size_t row = 0; row < p; ++row)
        rhs[row] = -problem.stages[i].e[row];
      PrintVector(rhs);
    } else {
      PrintVector(Vector(p));
    }
  });
  std::cout << ",\"X\":";
  PrintSequence(solution.states.size(),
                [&](std::size_t i) { PrintVector(solution.states[i]); });
  std::cout << ",\"U\":";
  PrintSequence(solution.controls.size(),
                [&](std::size_t i) { PrintVector(solution.controls[i]); });
  std::cout << "}\n";
  return 0;
}
