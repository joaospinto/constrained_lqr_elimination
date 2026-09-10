// Inspect the actual CPU reduction without exposing implementation details in
// the public API or adding objective-audit work to the timed solver.
#include "../benchmarks/scaling_problem.h"

// This test compiles the implementation in a single translation unit. Its
// private anonymous-namespace types do not cross a translation-unit boundary;
// GCC only diagnoses them here because the .cc file is included as a header.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#endif
#include "../src/clqr.cc"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <iostream>

int main() {
  using namespace clqr;
  long double worst = 0;
  std::size_t checks = 0;
#ifdef CLQR_USE_FLOAT
  constexpr long double threshold = 5e-4L;
#else
  constexpr long double threshold = 5e-11L;
#endif
  for (const std::size_t n : {4, 8, 16}) {
    const std::size_t m = n / 2;
    for (const std::size_t horizon : {0, 1, 17}) {
      for (int kind = 0; kind < 3; ++kind) {
        const auto fixture = benchmark::MakeScalingProblem(
            horizon, n, m, kind == 0 ? 0 : m / 2, kind == 1 ? 0 : m / 2);
        const Problem &p = fixture.problem;
        auto state = Initialize(p);
        std::string error;
        NewtonKktDiagnostics diagnostics;
        EliminateConstraintsRightToLeft(state, SolveOptions{}.tolerance, &error,
                                        &diagnostics);
        Problem reduced;
        reduced.stages = state.problem.stages;
        reduced.Q = state.problem.Q;
        reduced.q = state.problem.q;
        benchmark::Random random(7901);
        std::vector<Vector> x(horizon + 1), z(horizon + 1),
            offsets(horizon + 1), u(horizon), v(horizon),
            control_offsets(horizon);
        for (std::size_t i = 0; i <= horizon; ++i) {
          offsets[i] = LazyIdentityStateMap(state.state_maps[i])
                           ? Vector(n)
                           : state.state_maps[i].offset;
          if (i < horizon)
            control_offsets[i] = state.control_maps[i].offset;
        }
        const long double kappa =
            benchmark::OriginalObjective(p, offsets, control_offsets);
        for (int probe = 0; probe < 4; ++probe) {
          for (std::size_t i = 0; i <= horizon; ++i) {
            z[i] = random.Vec(reduced.Q[i].rows(),
                              probe == 0 ? Scalar{0} : Scalar{0.7});
            const auto &map = state.state_maps[i];
            x[i] = LazyIdentityStateMap(map) ? z[i]
                                             : map.linear * z[i] + map.offset;
            if (i < horizon) {
              v[i] = random.Vec(reduced.stages[i].R.rows(),
                                probe == 0 ? Scalar{0} : Scalar{0.4});
              ApplyControlMap(state.control_maps[i], z[i], v[i], &u[i]);
            }
          }
          const long double original = benchmark::OriginalObjective(p, x, u);
          const long double transformed =
              benchmark::OriginalObjective(reduced, z, v);
          const long double relative =
              std::abs(original - transformed - kappa) /
              std::max({1.0L, std::abs(original), std::abs(transformed),
                        std::abs(kappa)});
          if (!std::isfinite(relative) || relative > threshold) {
            std::cerr << "objective identity N=" << horizon << " n=" << n
                      << " kind=" << kind << " probe=" << probe
                      << " relative_error=" << relative << '\n';
            return 1;
          }
          worst = std::max(worst, relative);
          ++checks;
        }
      }
    }
  }
  std::cout << checks << " reduced-plus-constant objective identities passed; "
            << "max relative discrepancy=" << worst << '\n';
}
