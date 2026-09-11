#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../src/cuda_stage_layout.h"

namespace {
using namespace clqr::cuda::detail;
using clqr::Scalar;

void Expect(bool condition) {
  if (!condition)
    throw std::runtime_error("CUDA phase layout check failed");
}

template <typename Layout> void CheckLayout(Layout layout) {
  DenseLayoutCursor plan;
  layout(plan, [](auto *, std::size_t) {});
  std::vector<std::byte> storage(plan.bytes());
  DenseLayoutCursor bind(storage.data());
  std::size_t end = 0;
  const auto check = [&](auto *pointer, std::size_t count) {
    using T = std::remove_pointer_t<decltype(pointer)>;
    if (count == 0)
      return;
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    Expect(address % alignof(T) == 0);
    const std::size_t begin =
        reinterpret_cast<std::byte *>(pointer) - storage.data();
    Expect(begin >= end && count * sizeof(T) <= storage.size() - begin);
    end = begin + count * sizeof(T);
    std::fill_n(pointer, count, T{1});
  };
  layout(bind, check);
  Expect(bind.bytes() == plan.bytes());
}

void CheckDimensions(std::size_t n, std::size_t next, std::size_t m,
                     std::size_t p) {
  CheckLayout([&](DenseLayoutCursor &cursor, auto check) {
    const auto s = LayoutStateParameter(cursor, n);
    check(s.free_columns, n);
    check(s.T, n * n);
    check(s.t, n);
    const auto c = LayoutControlParameter(cursor, m, n);
    check(c.free_columns, m);
    check(c.Y, m * n);
    check(c.Z, m * m);
    check(c.y, m);
    const auto r = LayoutReducedStage(cursor, n, next, m);
    check(r.A, next * n);
    check(r.B, next * m);
    check(r.c, next);
    check(r.Q, n * n);
    check(r.R, m * m);
    check(r.M, n * m);
    check(r.q, n);
    check(r.r, m);
    const auto t = LayoutReducedTerminal(cursor, next);
    check(t.Q, next * next);
    check(t.q, next);
    const auto f = LayoutFeedback(cursor, n, next, m);
    check(f.K, m * n);
    check(f.k, m);
    check(f.control_factor, m * m);
    check(f.transition, next * n);
    check(f.offset, next);
    const auto d = LayoutDualParameter(cursor, next + p);
    check(d.free_columns, next + p);
    check(d.basis, (next + p) * (next + p));
    check(d.offset, next + p);
    const auto sd = LayoutStateDualParameter(cursor, p, n, m);
    check(sd.offset, p);
    check(sd.left, p * n);
    check(sd.right, p * m);
  });
}

void CheckLargePhaseReuse() {
  constexpr std::size_t N = 32768, n = 64, m = 32, reduced = 48, control = 8;
  DenseLayoutCursor state, reduction, feedback, dual;
  for (std::size_t i = 0; i < N; ++i) {
    LayoutStateParameter(state, n);
    LayoutControlParameter(reduction, m, reduced);
    LayoutReducedStage(reduction, reduced, reduced, m);
    LayoutFeedback(feedback, reduced, reduced, control);
    LayoutDualParameter(dual, n + n / 8);
  }
  LayoutStateParameter(state, n);
  LayoutReducedTerminal(reduction, reduced);
  // Even the minimum full-width feasibility tree provides this many scalar
  // slots. Both following phases fit without keeping another dense allocation.
  Expect(reduction.bytes() + feedback.bytes() < N * 4 * n * n * sizeof(Scalar));
  Expect(dual.bytes() < reduction.bytes());
  std::cout << "N=" << N << " n=" << n << " state_bytes=" << state.bytes()
            << " reduction_bytes=" << reduction.bytes()
            << " feedback_bytes=" << feedback.bytes()
            << " dual_parameter_bytes=" << dual.bytes() << '\n';
}
} // namespace

int main() {
  for (std::size_t n : {0, 1, 3, 8, 32, 64})
    for (std::size_t next : {0, 1, 5, 16, 64})
      for (std::size_t m : {0, 1, 3, 32})
        CheckDimensions(n, next, m, (n + m) % 11);
  bool overflow = false;
  try {
    DenseLayoutCursor cursor;
    LayoutReducedStage(cursor, std::numeric_limits<std::size_t>::max(), 2, 1);
  } catch (const std::invalid_argument &) {
    overflow = true;
  }
  Expect(overflow);
  CheckLargePhaseReuse();
}
