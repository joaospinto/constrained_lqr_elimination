#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "clqr/cuda.h"

namespace {

using clqr::Matrix;
using clqr::Problem;
using clqr::Stage;
using clqr::Vector;

double Value(int seed, std::size_t row, std::size_t col = 0) {
  const double x = static_cast<double>(seed * 97 + row * 31 + col * 47);
  return std::sin(0.017 * x) + 0.3 * std::cos(0.029 * x);
}

Matrix GeneratedMatrix(std::size_t rows, std::size_t cols, int seed,
                       double scale) {
  Matrix out(rows, cols);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col)
      out(row, col) = scale * Value(seed, row, col);
  return out;
}

Vector GeneratedVector(std::size_t size, int seed, double scale) {
  Vector out(size);
  for (std::size_t row = 0; row < size; ++row)
    out[row] = scale * Value(seed, row);
  return out;
}

Matrix PositiveDefinite(std::size_t size, int seed, double diagonal) {
  Matrix g = GeneratedMatrix(size, size, seed, 0.15);
  Matrix out = clqr::Transpose(g) * g;
  for (std::size_t row = 0; row < size; ++row) out(row, row) += diagonal;
  return out;
}

double RowDot(const Matrix& matrix, std::size_t row, const Vector& vector) {
  double value = 0.0;
  for (std::size_t col = 0; col < vector.size(); ++col)
    value += matrix(row, col) * vector[col];
  return value;
}

Problem MakeProblem() {
  constexpr std::size_t horizon = 7;
  constexpr std::size_t n = 4;
  constexpr std::size_t m = 3;
  constexpr std::size_t p = 2;
  Problem problem;
  std::vector<Vector> x(horizon + 1);
  std::vector<Vector> u(horizon);
  for (std::size_t i = 0; i <= horizon; ++i)
    x[i] = GeneratedVector(n, 101 + static_cast<int>(i), 0.5);
  for (std::size_t i = 0; i < horizon; ++i)
    u[i] = GeneratedVector(m, 201 + static_cast<int>(i), 0.4);
  problem.initial_state = x[0];
  problem.stages.resize(horizon);
  for (std::size_t i = 0; i < horizon; ++i) {
    Stage& stage = problem.stages[i];
    stage.A = GeneratedMatrix(n, n, 301 + static_cast<int>(i), 0.1);
    for (std::size_t row = 0; row < n; ++row) stage.A(row, row) += 0.85;
    stage.B = GeneratedMatrix(n, m, 401 + static_cast<int>(i), 0.2);
    stage.c = x[i + 1] - stage.A * x[i] - stage.B * u[i];
    stage.Q = PositiveDefinite(n, 501 + static_cast<int>(i), 1.0);
    stage.R = PositiveDefinite(m, 601 + static_cast<int>(i), 1.3);
    stage.M = GeneratedMatrix(n, m, 701 + static_cast<int>(i), 0.025);
    stage.q = GeneratedVector(n, 801 + static_cast<int>(i), 0.15);
    stage.r = GeneratedVector(m, 901 + static_cast<int>(i), 0.15);
    stage.C = Matrix(0, n);
    stage.D = Matrix(0, m);
    stage.d = Vector(0);
    stage.E = GeneratedMatrix(p, n, 1001 + static_cast<int>(i), 0.3);
    for (std::size_t col = 0; col < n; ++col)
      stage.E(1, col) = 2.0 * stage.E(0, col);
    stage.e = Vector(p);
    stage.e[0] = -RowDot(stage.E, 0, x[i]);
    stage.e[1] = 2.0 * stage.e[0];
  }
  problem.terminal_Q = PositiveDefinite(n, 1101, 1.4);
  problem.terminal_q = GeneratedVector(n, 1201, 0.15);
  problem.terminal_E = Matrix(0, n);
  problem.terminal_e = Vector(0);
  return problem;
}

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
  const Problem problem = MakeProblem();
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
  std::cout << std::setprecision(17);
  std::cout << "{\"A\":";
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
