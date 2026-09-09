#ifndef CLQR_BENCHMARKS_SCALING_PROBLEM_H_
#define CLQR_BENCHMARKS_SCALING_PROBLEM_H_

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "clqr/clqr.h"

namespace clqr::benchmark {

// Integer arithmetic fixes the random stream independently of the C++ standard
// library. Unlike a sum of separable sinusoids, this does not impose a low-rank
// structure on the generated dense matrices.
class Random {
public:
  explicit Random(std::uint64_t seed) : state_(seed) {}
  Scalar Next() {
    state_ += UINT64_C(0x9e3779b97f4a7c15);
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    z ^= z >> 31;
    return static_cast<Scalar>(2.0 * static_cast<double>(z >> 11) * 0x1.0p-53 -
                               1.0);
  }
  Matrix Mat(std::size_t rows, std::size_t cols, Scalar scale) {
    Matrix a(rows, cols);
    for (std::size_t i = 0; i < rows; ++i)
      for (std::size_t j = 0; j < cols; ++j)
        a(i, j) = scale * Next();
    return a;
  }
  Vector Vec(std::size_t rows, Scalar scale) {
    Vector v(rows);
    for (std::size_t i = 0; i < rows; ++i)
      v[i] = scale * Next();
    return v;
  }

private:
  std::uint64_t state_;
};

struct ScalingProblem {
  Problem problem;
  std::vector<Vector> states;
  std::vector<Vector> controls;
};

// Plant a primal-dual optimum, not merely a feasible trajectory. The Hessian is
// I + G'G at every stage, so the optimum is unique. Offsets enforce
// feasibility; gradients enforce stationarity with independently generated
// multipliers. The initial state is fixed, so omit state-only rows at node
// zero: retaining them would make the global equality Jacobian redundant by
// construction and exclude otherwise suitable full-row-rank reference solvers.
inline ScalingProblem MakeScalingProblem(std::size_t horizon, std::size_t n,
                                         std::size_t m, std::size_t mixed_rows,
                                         std::size_t state_rows,
                                         std::uint64_t seed = 20260907) {
  if (n == 0 || state_rows > n || mixed_rows + state_rows > m)
    throw std::invalid_argument("scaling family needs n > 0, state_rows <= n, "
                                "mixed_rows + state_rows <= m");
  Random rng(seed);
  ScalingProblem out;
  Problem &p = out.problem;
  out.states.resize(horizon + 1);
  out.controls.resize(horizon);
  std::vector<Vector> lambda(horizon + 1);
  for (std::size_t i = 0; i <= horizon; ++i) {
    out.states[i] = rng.Vec(n, Scalar{0.5});
    lambda[i] = rng.Vec(n, Scalar{0.2});
    if (i < horizon)
      out.controls[i] = rng.Vec(m, Scalar{0.4});
  }
  p.initial_state = out.states.front();
  p.stages.resize(horizon);
  p.Q.resize(horizon + 1);
  p.q.resize(horizon + 1);
  for (std::size_t i = 0; i < horizon; ++i) {
    Stage &s = p.stages[i];
    s.A = rng.Mat(n, n, Scalar{0.15} / std::sqrt(Scalar(n)));
    for (std::size_t j = 0; j < n; ++j)
      s.A(j, j) += Scalar{0.7};
    s.B = rng.Mat(n, m, Scalar{1} / std::sqrt(Scalar(m > 0 ? m : 1)));
    s.c = out.states[i + 1] - s.A * out.states[i] - s.B * out.controls[i];
    Matrix g = rng.Mat(n + m, n + m, Scalar{0.3} / std::sqrt(Scalar(n + m)));
    Matrix h = Transpose(g) * g;
    for (std::size_t j = 0; j < n + m; ++j)
      h(j, j) += Scalar{1};
    p.Q[i] = Matrix(n, n);
    s.M = Matrix(n, m);
    s.R = Matrix(m, m);
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < n; ++k)
        p.Q[i](j, k) = h(j, k);
      for (std::size_t k = 0; k < m; ++k)
        s.M(j, k) = h(j, n + k);
    }
    for (std::size_t j = 0; j < m; ++j)
      for (std::size_t k = 0; k < m; ++k)
        s.R(j, k) = h(n + j, n + k);
    s.C = rng.Mat(mixed_rows, n, Scalar{1} / std::sqrt(Scalar(n)));
    s.D = rng.Mat(mixed_rows, m, Scalar{1});
    s.d = Scale(s.C * out.states[i] + s.D * out.controls[i], Scalar{-1});
    s.E = rng.Mat(i == 0 ? 0 : state_rows, n, Scalar{1});
    s.e = Scale(s.E * out.states[i], Scalar{-1});
    const Vector mu = rng.Vec(mixed_rows, Scalar{0.2});
    const Vector eta = rng.Vec(s.E.rows(), Scalar{0.2});
    p.q[i] = Scale(p.Q[i] * out.states[i] + s.M * out.controls[i], Scalar{-1}) +
             Transpose(s.A) * lambda[i + 1] - lambda[i] - Transpose(s.C) * mu -
             Transpose(s.E) * eta;
    s.r = Scale(Transpose(s.M) * out.states[i] + s.R * out.controls[i],
                Scalar{-1}) +
          Transpose(s.B) * lambda[i + 1] - Transpose(s.D) * mu;
  }
  Matrix g = rng.Mat(n, n, Scalar{0.3} / std::sqrt(Scalar(n)));
  p.Q.back() = Transpose(g) * g;
  for (std::size_t j = 0; j < n; ++j)
    p.Q.back()(j, j) += Scalar{1};
  p.terminal_E = rng.Mat(horizon == 0 ? 0 : state_rows, n, Scalar{1});
  p.terminal_e = Scale(p.terminal_E * out.states.back(), Scalar{-1});
  p.q.back() =
      Scale(p.Q.back() * out.states.back(), Scalar{-1}) - lambda.back() -
      Transpose(p.terminal_E) * rng.Vec(p.terminal_E.rows(), Scalar{0.2});
  return out;
}

// Evaluate the original objective independently of every solver, with a wider
// accumulator on hosts where long double offers one. Never include this audit
// in a solver timing interval.
inline long double OriginalObjective(const Problem &p,
                                     const std::vector<Vector> &x,
                                     const std::vector<Vector> &u) {
  long double value = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    for (std::size_t j = 0; j < x[i].size(); ++j) {
      value += static_cast<long double>(p.q[i][j]) * x[i][j];
      for (std::size_t k = 0; k < x[i].size(); ++k)
        value += 0.5L * x[i][j] * p.Q[i](j, k) * x[i][k];
      if (i < u.size())
        for (std::size_t k = 0; k < u[i].size(); ++k)
          value +=
              static_cast<long double>(x[i][j]) * p.stages[i].M(j, k) * u[i][k];
    }
    if (i < u.size()) {
      for (std::size_t j = 0; j < u[i].size(); ++j) {
        value += static_cast<long double>(p.stages[i].r[j]) * u[i][j];
        for (std::size_t k = 0; k < u[i].size(); ++k)
          value += 0.5L * u[i][j] * p.stages[i].R(j, k) * u[i][k];
      }
    }
  }
  return value;
}

} // namespace clqr::benchmark
#endif
