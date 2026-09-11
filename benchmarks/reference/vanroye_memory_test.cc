#include "blasfeo_wrapper.hpp"

#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace {
std::size_t requested = 0;
bool reject = false;
void *TestMalloc(std::size_t bytes) {
  requested = bytes;
  // Never let a broken regression test allocate the multi-GiB test buffers.
  if (reject || bytes > 1024 * 1024)
    return nullptr;
  return std::malloc(bytes);
}
void Expect(bool condition) {
  if (!condition)
    throw std::runtime_error("Vanroye allocation regression");
}
template <class Construct> void ExpectRejected(Construct construct) {
  reject = true;
  bool threw = false;
  try {
    construct();
  } catch (const std::bad_alloc &) {
    threw = true;
  }
  Expect(threw);
  reject = false;
}
} // namespace

// Exercise the actual wrapper with allocation failure injection. All numerical
// functions still link against BLASFEO; no external source is vendored.
#define malloc TestMalloc
#include "blasfeo_wrapper.cpp"
#undef malloc

int main() {
  using namespace gen_riccati;
  const std::size_t count = 32768;
  ExpectRejected([&] { FatropMemoryMatBF matrices(97, 96, count); });
  const std::size_t headers = count * sizeof(MAT);
  const std::size_t expected =
      (headers + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE * CACHE_LINE_SIZE +
      CACHE_LINE_SIZE + count * MEMSIZE_MAT(97, 96);
  Expect(expected > std::numeric_limits<int>::max() && requested == expected);
  ExpectRejected([] { FatropMemoryVecBF vectors(65536, 4096); });
  Expect(requested > std::numeric_limits<int>::max());
  ExpectRejected([] { MemoryPermMat permutations(64, 10000000); });
  Expect(requested == 10000000ULL * (sizeof(PermMat) + 64 * sizeof(int)));

  FatropMemoryMatBF matrices(3, 4, 2);
  matrices[1].at(2, 3) = 7;
  Expect(matrices[1].at(2, 3) == 7);
  ExpectRejected([&] { matrices.set_up(); });
  matrices.set_up();
  Expect(matrices[1].at(2, 3) == 0);
  FatropMemoryVecBF vectors(4, 2);
  vectors[1].at(3) = 5;
  Expect(vectors[1].at(3) == 5);
  ExpectRejected([&] { vectors.set_up(); });
  vectors.set_up();
  MemoryPermMat permutations(4, 2);
  ExpectRejected([&] { permutations.set_up(); });
  permutations.set_up();
}
