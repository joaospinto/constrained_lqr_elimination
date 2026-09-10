// Stream a benchmark instance to Python without duplicating the generator or
// retaining large datasets on disk. Array dimensions precede little-endian FP64
// data; the small case listing is JSON. Neither operation is part of a solve.
#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "benchmarks/paper_cases.h"
#include "benchmarks/scaling_problem.h"

namespace {
void Word(std::uint64_t value) {
  std::cout.write(reinterpret_cast<const char *>(&value), sizeof(value));
}
void Values(const clqr::Scalar *data, std::size_t size) {
  if (size)
    std::cout.write(reinterpret_cast<const char *>(data), sizeof(*data) * size);
}
void Array(const clqr::Matrix &value) {
  Word(value.rows());
  Word(value.cols());
  Values(value.data().data(), value.data().size());
}
void Array(const clqr::Vector &value) {
  Word(value.size());
  Values(value.data().data(), value.size());
}
} // namespace

int main(int argc, char **argv) {
  static_assert(std::endian::native == std::endian::little);
  static_assert(sizeof(clqr::Scalar) == sizeof(double));
  std::string suite = "all";
  std::uint64_t seed = 20260907;
  int index = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (i + 1 == argc)
      throw std::invalid_argument("missing option value");
    if (option == "--suite")
      suite = argv[++i];
    else if (option == "--index")
      index = std::stoi(argv[++i]);
    else if (option == "--seed")
      seed = std::stoull(argv[++i]);
    else
      throw std::invalid_argument("unknown option: " + option);
  }
  const auto cases = clqr::benchmark::PaperCases(suite);
  if (index < 0) {
    std::cout << '[';
    for (std::size_t i = 0; i < cases.size(); ++i) {
      const auto &c = cases[i];
      if (i)
        std::cout << ',';
      std::cout << "{\"index\":" << i << ",\"family\":\"" << c.family
                << "\",\"N\":" << c.horizon << ",\"n\":" << c.n
                << ",\"m\":" << c.m << ",\"mixed_rows\":" << c.mixed
                << ",\"state_rows\":" << c.state << '}';
    }
    std::cout << "]\n";
    return 0;
  }
  const auto &c = cases.at(static_cast<std::size_t>(index));
  const auto data = clqr::benchmark::MakeScalingProblem(c.horizon, c.n, c.m,
                                                        c.mixed, c.state, seed);
  const auto &p = data.problem;
  std::cout.write("CLQRB002", 8);
  Word(c.horizon);
  Word(seed);
  Array(p.initial_state);
  for (std::size_t i = 0; i <= c.horizon; ++i) {
    Array(p.Q[i]);
    Array(p.q[i]);
  }
  Array(p.terminal_E);
  Array(p.terminal_e);
  for (const auto &s : p.stages) {
    Array(s.A);
    Array(s.B);
    Array(s.c);
    Array(s.R);
    Array(s.M);
    Array(s.r);
    Array(s.C);
    Array(s.D);
    Array(s.d);
    Array(s.E);
    Array(s.e);
  }
  for (const auto &x : data.states)
    Array(x);
  for (const auto &u : data.controls)
    Array(u);
  Array(data.dual.initial);
  Array(data.dual.terminal);
  for (const auto &lambda : data.dual.dynamics)
    Array(lambda);
  for (const auto &mu : data.dual.mixed)
    Array(mu);
  for (const auto &eta : data.dual.state)
    Array(eta);
  if (!std::cout)
    throw std::runtime_error("fixture output failed");
}
