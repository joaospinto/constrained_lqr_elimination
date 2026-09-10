#ifndef CLQR_BENCHMARKS_PAPER_CASES_H_
#define CLQR_BENCHMARKS_PAPER_CASES_H_

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace clqr::benchmark {
struct PaperCase {
  std::string family;
  std::size_t horizon, n, m, mixed, state;
};

inline std::vector<PaperCase> PaperCases(const std::string &suite) {
  std::vector<PaperCase> cases;
  if (suite == "smoke")
    return {{"state", 1, 4, 2, 0, 1},
            {"mixed", 17, 8, 4, 2, 0},
            {"combined", 17, 8, 4, 1, 1},
            {"zero", 0, 4, 2, 0, 0}};
  if (suite == "horizon" || suite == "all")
    for (const std::size_t N : {128, 512, 2048, 8192, 16384, 32768}) {
      cases.push_back({"state", N, 8, 4, 0, 2});
      cases.push_back({"combined", N, 8, 4, 1, 1});
    }
  if (suite == "dimension" || suite == "all")
    for (const std::size_t n : {8, 16, 24, 32, 48, 64})
      cases.push_back({"dimension", 128, n, n / 2, n / 8, n / 8});
  if (suite == "constraints" || suite == "all")
    for (const std::size_t p : {0, 2, 4, 6}) {
      cases.push_back({"mixed_rows", 512, 16, 8, p, 0});
      cases.push_back({"state_rows", 512, 16, 8, 0, p});
    }
  if (cases.empty())
    throw std::invalid_argument("unknown paper benchmark suite: " + suite);
  return cases;
}
} // namespace clqr::benchmark
#endif
