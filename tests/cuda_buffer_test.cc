#include <cuda_runtime.h>

#include <cstdlib>
#include <limits>
#include <stdexcept>

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
namespace mock {
bool fail = false;
bool fail_info = false;
int allocations = 0;
int frees = 0;
} // namespace mock
cudaError_t cudaMalloc(void **out, std::size_t bytes) {
  ++mock::allocations;
  *out = mock::fail ? nullptr : std::malloc(bytes);
  return *out == nullptr ? 2 : cudaSuccess;
}
cudaError_t cudaFree(void *data) {
  ++mock::frees;
  std::free(data);
  return cudaSuccess;
}
const char *cudaGetErrorString(cudaError_t) { return "out of memory"; }
cudaError_t cudaMemGetInfo(std::size_t *free_bytes, std::size_t *total_bytes) {
  *free_bytes = 256;
  *total_bytes = 1024;
  return mock::fail_info ? 2 : cudaSuccess;
}

#include "src/cuda_buffer.h"

void Expect(bool condition) {
  if (!condition)
    throw std::runtime_error("CUDA buffer test failed");
}

int main() {
  clqr::cuda::detail::DeviceBuffer<double> buffer;
  for (int attempt = 1; attempt <= 2; ++attempt) {
    mock::fail = true;
    bool failed = false;
    try {
      buffer.Reserve(64);
    } catch (const clqr::cuda::detail::DeviceAllocationError &error) {
      failed = true;
      const std::string message(error.what());
      Expect(message.find("requested_bytes=512") != std::string::npos);
      Expect(message.find("free_bytes=256") != std::string::npos);
      Expect(message.find("total_bytes=1024") != std::string::npos);
    }
    Expect(failed && buffer.get() == nullptr && buffer.count() == 0);
    Expect(mock::allocations == attempt);
  }
  mock::fail = false;
  buffer.Reserve(64);
  Expect(buffer.get() != nullptr && buffer.count() == 64);
  buffer.Reserve(64);
  Expect(mock::allocations == 3);
  mock::fail = true;
  mock::fail_info = true;
  try {
    buffer.Reserve(128);
  } catch (const clqr::cuda::detail::DeviceAllocationError &error) {
    const std::string message(error.what());
    Expect(message.find("requested_bytes=1024") != std::string::npos);
    Expect(message.find("free_bytes=") == std::string::npos);
  }
  Expect(buffer.get() == nullptr && buffer.count() == 0 && mock::frees == 1);
  mock::fail = false;
  buffer.Reserve(128);
  Expect(buffer.get() != nullptr && buffer.count() == 128 &&
         mock::allocations == 5);
  bool overflow = false;
  try {
    buffer.Reserve(std::numeric_limits<std::size_t>::max());
  } catch (const std::invalid_argument &) {
    overflow = true;
  }
  Expect(overflow && buffer.get() == nullptr && buffer.count() == 0);
  double borrowed[4]{};
  buffer.Bind(borrowed, 4);
  buffer.Release();
  Expect(mock::frees == 2);
}
