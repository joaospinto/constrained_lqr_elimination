#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

// Exercise the production host-side configuration without a CUDA device.
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
enum cudaDeviceAttr {
  cudaDevAttrMaxSharedMemoryPerBlock,
  cudaDevAttrMaxSharedMemoryPerBlockOptin,
  cudaDevAttrMultiProcessorCount,
};
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize };
struct cudaFuncAttributes {
  std::size_t sharedSizeBytes = 0;
  int maxDynamicSharedSizeBytes = 48 * 1024;
};

namespace mock {
void FirstKernel() {}
void SecondKernel() {}
using Kernel = void (*)();
constexpr int ordinary = 48 * 1024;
constexpr int optin = 99 * 1024;
int device = 0;
int optin_capacity = optin;
int set_calls = 0;
int last_set = 0;
int queried_device = -1;
int multiprocessors = 56;
int resident_blocks = 32;
int occupancy_queries = 0;
cudaError_t error = cudaSuccess;
cudaError_t set_error = cudaSuccess;
cudaError_t occupancy_error = cudaSuccess;
std::array<std::array<cudaFuncAttributes, 2>, 2> attributes{};

cudaFuncAttributes &Attributes(Kernel kernel) {
  return attributes[device][kernel == FirstKernel ? 0 : 1];
}

void Reset() {
  device = 0;
  optin_capacity = optin;
  set_calls = 0;
  last_set = 0;
  queried_device = -1;
  multiprocessors = 56;
  resident_blocks = 32;
  occupancy_queries = 0;
  error = cudaSuccess;
  set_error = cudaSuccess;
  occupancy_error = cudaSuccess;
  attributes = {};
}
} // namespace mock

const char *cudaGetErrorString(cudaError_t) { return "mock CUDA error"; }

cudaError_t cudaDeviceGetAttribute(int *out, cudaDeviceAttr attribute,
                                   int device) {
  mock::queried_device = device;
  switch (attribute) {
  case cudaDevAttrMaxSharedMemoryPerBlock:
    *out = mock::ordinary;
    break;
  case cudaDevAttrMaxSharedMemoryPerBlockOptin:
    *out = mock::optin_capacity;
    break;
  case cudaDevAttrMultiProcessorCount:
    *out = mock::multiprocessors;
    break;
  }
  return mock::error;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int *out, mock::Kernel kernel, int threads, std::size_t shared_bytes) {
  if (kernel != mock::SecondKernel || threads != 32 || shared_bytes != 0)
    std::abort();
  ++mock::occupancy_queries;
  *out = mock::resident_blocks;
  return mock::occupancy_error;
}

cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *out,
                                  mock::Kernel kernel) {
  *out = mock::Attributes(kernel);
  return mock::error;
}

cudaError_t cudaFuncSetAttribute(mock::Kernel kernel,
                                 cudaFuncAttribute attribute, int value) {
  if (attribute != cudaFuncAttributeMaxDynamicSharedMemorySize)
    std::abort();
  ++mock::set_calls;
  mock::last_set = value;
  if (mock::set_error == cudaSuccess)
    mock::Attributes(kernel).maxDynamicSharedSizeBytes = value;
  return mock::set_error;
}

#include "../src/cuda_shared_memory.h"

namespace {
using clqr::cuda::detail::ConfigureKernelSharedMemory;
using clqr::cuda::detail::DeviceSharedMemoryCapacity;
using clqr::cuda::detail::ForEachGlobalScratchLaunch;
using clqr::cuda::detail::PlanKernelScratch;

void Expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void Configure(std::size_t bytes, int capacity = mock::optin) {
  ConfigureKernelSharedMemory(mock::FirstKernel, "FirstKernel", bytes,
                              capacity);
}

void ExpectRejected(std::size_t bytes, int capacity) {
  bool rejected = false;
  try {
    Configure(bytes, capacity);
  } catch (const std::invalid_argument &error) {
    const std::string message = error.what();
    rejected = message.find("FirstKernel") != std::string::npos &&
               message.find(std::to_string(bytes)) != std::string::npos &&
               message.find("static shared memory") != std::string::npos;
  }
  Expect(rejected, "oversized scratch gives a kernel-specific diagnostic");
}

void DeviceCapacityCase() {
  mock::Reset();
  Expect(DeviceSharedMemoryCapacity(1) == mock::optin &&
             mock::queried_device == 1,
         "query the selected GPU's opt-in capacity");
  mock::optin_capacity = 0;
  Expect(DeviceSharedMemoryCapacity(0) == mock::ordinary,
         "a device without extra opt-in memory retains its ordinary limit");
  mock::optin_capacity = mock::ordinary;
  Expect(DeviceSharedMemoryCapacity(0) == mock::ordinary,
         "P100-like capacity remains 48 KiB");
}

void ExactCapacityCase() {
  mock::Reset();
  auto &attributes = mock::Attributes(mock::FirstKernel);
  attributes.sharedSizeBytes = 128;
  attributes.maxDynamicSharedSizeBytes = mock::ordinary - 128;
  Configure(0);
  Configure(mock::ordinary - 128);
  Expect(mock::set_calls == 0, "small problems do not opt in");
  Configure(mock::ordinary - 127);
  Expect(mock::set_calls == 1 && mock::last_set == mock::optin - 128,
         "opt in above the kernel's ordinary dynamic limit");
  Configure(mock::optin - 128);
  Configure(1024);
  Expect(mock::set_calls == 1,
         "exact fit and smaller workspaces never lower the enabled capacity");
  ExpectRejected(mock::optin - 127, mock::optin);
  ExpectRejected(std::numeric_limits<std::size_t>::max(), mock::optin);
  attributes.sharedSizeBytes = mock::optin + 1;
  ExpectRejected(0, mock::optin);
  Expect(mock::set_calls == 1, "invalid requests do not change attributes");
}

void BenchmarkDimensionsCase() {
  mock::Reset();
  // FP64 uniform n=24 primal-relation finalization, including its record.
  constexpr std::size_t bytes = 128 * 24 * 24 + 112 * 24 + 40;
  ExpectRejected(bytes, mock::ordinary);
  Configure(bytes);
  Expect(mock::set_calls == 1, "n=24 fits the larger opt-in budget");
  ExpectRejected(128 * 32 * 32 + 112 * 32 + 40, mock::optin);
}

void KernelAndDeviceIsolationCase() {
  mock::Reset();
  Configure(70 * 1024);
  ConfigureKernelSharedMemory(mock::SecondKernel, "SecondKernel", 60 * 1024,
                              mock::optin);
  Expect(mock::set_calls == 2, "each kernel must opt in independently");
  mock::device = 1;
  Configure(65 * 1024);
  Expect(mock::set_calls == 3, "each GPU must opt in independently");
  mock::device = 0;
  Configure(80 * 1024);
  Expect(mock::set_calls == 3, "an enabled GPU/kernel keeps its full capacity");
}

void ApiErrorCase() {
  mock::Reset();
  mock::error = 1;
  bool query_failed = false;
  bool configure_failed = false;
  try {
    (void)DeviceSharedMemoryCapacity(0);
  } catch (const std::runtime_error &) {
    query_failed = true;
  }
  try {
    Configure(60 * 1024);
  } catch (const std::runtime_error &) {
    configure_failed = true;
  }
  Expect(query_failed && configure_failed && mock::set_calls == 0,
         "CUDA query failures are propagated before configuring a launch");
  mock::Reset();
  mock::set_error = 1;
  bool set_failed = false;
  try {
    Configure(60 * 1024);
  } catch (const std::runtime_error &) {
    set_failed = true;
  }
  Expect(set_failed && mock::set_calls == 1 &&
             mock::Attributes(mock::FirstKernel).maxDynamicSharedSizeBytes ==
                 mock::ordinary,
         "a failed opt-in is not silently accepted");
  mock::set_error = cudaSuccess;
  Configure(60 * 1024);
  Expect(mock::set_calls == 2, "retry a failed configuration");
}

void GlobalFallbackCase() {
  mock::Reset();
  auto plan = PlanKernelScratch(mock::FirstKernel, mock::SecondKernel, "test",
                                1024, mock::ordinary, 0, 32);
  Expect(plan.shared_bytes == 1024 && plan.global_stride == 0 &&
             plan.GlobalBytes(100) == 0 && mock::occupancy_queries == 0,
         "fitting kernels do not allocate global scratch");
  plan = PlanKernelScratch(mock::FirstKernel, mock::SecondKernel, "test", 76001,
                           mock::ordinary, 1, 32);
  Expect(plan.shared_bytes == 0 && plan.global_stride == 76016 &&
             plan.GlobalBytes(65) == 76016 * 65 && mock::set_calls == 0 &&
             plan.global_block_limit == 56 * 32 && mock::queried_device == 1,
         "oversized kernels use aligned per-block global slices");
  plan = PlanKernelScratch(mock::FirstKernel, mock::SecondKernel, "test", 76001,
                           mock::optin, 0, 32);
  Expect(plan.shared_bytes == 76001 && plan.global_stride == 0 &&
             mock::set_calls == 1,
         "opt-in shared memory is preferred when it fits");
  plan = PlanKernelScratch(mock::FirstKernel, mock::SecondKernel, "test", 76001,
                           mock::ordinary, 0, 32);
  Expect(
      plan.GlobalBytes(std::numeric_limits<std::size_t>::max()) ==
          76016 * 56 * 32,
      "global scratch does not grow with the horizon beyond one device wave");
  bool overflow = false;
  try {
    auto oversized = plan;
    oversized.global_stride = std::numeric_limits<std::size_t>::max();
    (void)oversized.GlobalBytes(2);
  } catch (const std::invalid_argument &) {
    overflow = true;
  }
  Expect(overflow, "global allocation multiplication is checked");
  overflow = false;
  try {
    (void)PlanKernelScratch(mock::FirstKernel, mock::SecondKernel, "test",
                            std::numeric_limits<std::size_t>::max(),
                            mock::ordinary, 0, 32);
  } catch (const std::invalid_argument &) {
    overflow = true;
  }
  Expect(overflow, "global alignment addition is checked");
}

void BoundedLaunchCase() {
  mock::Reset();
  mock::multiprocessors = 2;
  mock::resident_blocks = 3;
  auto plan = PlanKernelScratch(mock::FirstKernel, mock::SecondKernel, "test",
                                76001, mock::ordinary, 0, 32);
  for (int blocks : {0, 1, 5, 6, 7, 17, 32, 33, 257, 32769}) {
    int covered = 0;
    int launches = 0;
    ForEachGlobalScratchLaunch(plan, blocks, [&](int first, int count) {
      Expect(first == covered && count > 0 && count <= 6,
             "scratch launches cover contiguous, disjoint logical blocks");
      Expect(plan.GlobalBytes(blocks) >= plan.global_stride * count,
             "each physical block has its own allocated scratch slice");
      covered += count;
      ++launches;
    });
    Expect(covered == blocks && launches == (blocks + 5) / 6,
           "bounded launches cover zero, odd, and full-wave extents exactly");
  }
  for (int bad_count : {0, -1, std::numeric_limits<int>::max()}) {
    mock::multiprocessors = bad_count;
    bool rejected = false;
    try {
      (void)PlanKernelScratch(mock::FirstKernel, mock::SecondKernel, "test",
                              76001, mock::ordinary, 0, 32);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    Expect(rejected, "invalid or overflowing device concurrency is rejected");
  }
  mock::Reset();
  mock::occupancy_error = 1;
  bool rejected = false;
  try {
    (void)PlanKernelScratch(mock::FirstKernel, mock::SecondKernel, "test",
                            76001, mock::ordinary, 0, 32);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  Expect(rejected, "kernel occupancy query failure is propagated");
}
} // namespace

int main() {
  DeviceCapacityCase();
  ExactCapacityCase();
  BenchmarkDimensionsCase();
  KernelAndDeviceIsolationCase();
  ApiErrorCase();
  GlobalFallbackCase();
  BoundedLaunchCase();
  std::cout << "CUDA shared-memory configuration tests passed\n";
}
