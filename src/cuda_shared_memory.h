#ifndef CLQR_CUDA_SHARED_MEMORY_H_
#define CLQR_CUDA_SHARED_MEMORY_H_

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace clqr::cuda::detail {

struct KernelScratchLaunch {
  std::size_t shared_bytes = 0;
  std::size_t global_stride = 0;

  std::size_t GlobalBytes(std::size_t blocks) const {
    if (blocks && global_stride >
                      std::numeric_limits<std::size_t>::max() / blocks)
      throw std::invalid_argument("CUDA global scratch allocation overflows");
    return global_stride * blocks;
  }
};

inline void CheckSharedMemoryApi(cudaError_t error, const char *operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
}

inline int DeviceSharedMemoryCapacity(int device) {
  int ordinary = 0;
  int optin = 0;
  CheckSharedMemoryApi(
      cudaDeviceGetAttribute(&ordinary, cudaDevAttrMaxSharedMemoryPerBlock,
                             device),
      "query CUDA shared-memory limit");
  CheckSharedMemoryApi(
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             device),
      "query CUDA opt-in shared-memory limit");
  // Older devices can expose no extra opt-in capacity.
  return std::max(ordinary, optin);
}

template <typename Kernel>
void ConfigureKernelSharedMemory(Kernel kernel, const char *name,
                                 std::size_t required_bytes,
                                 int device_capacity) {
  cudaFuncAttributes attributes{};
  CheckSharedMemoryApi(cudaFuncGetAttributes(&attributes, kernel),
                       "query CUDA kernel shared-memory attributes");
  const std::size_t capacity =
      static_cast<std::size_t>(std::max(device_capacity, 0));
  if (attributes.sharedSizeBytes > capacity ||
      required_bytes > capacity - attributes.sharedSizeBytes) {
    throw std::invalid_argument(
        std::string(name) + " requires " + std::to_string(required_bytes) +
        " bytes of dynamic per-block workspace plus " +
        std::to_string(attributes.sharedSizeBytes) +
        " bytes of static shared memory, exceeding device shared-memory "
        "resources (" +
        std::to_string(capacity) + " bytes per block)");
  }
  if (required_bytes > static_cast<std::size_t>(
                           std::max(attributes.maxDynamicSharedSizeBytes, 0))) {
    // The attribute is shared by all workspaces using this kernel/device.
    // Always enable its full legal capacity, never a smaller workspace's
    // requirement: concurrent preparation must not lower another one's limit.
    // Launches still allocate only their exact planned dynamic byte count.
    const int dynamic_capacity =
        static_cast<int>(capacity - attributes.sharedSizeBytes);
    CheckSharedMemoryApi(
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             dynamic_capacity),
        "enable CUDA kernel opt-in shared memory");
  }
}

template <typename Kernel>
KernelScratchLaunch PlanKernelScratch(Kernel shared_kernel, Kernel global_kernel,
                                     const char *name, std::size_t bytes,
                                     int device_capacity, bool force_global) {
  cudaFuncAttributes attributes{};
  CheckSharedMemoryApi(cudaFuncGetAttributes(&attributes, shared_kernel),
                       "query CUDA shared-scratch kernel attributes");
  const auto capacity = static_cast<std::size_t>(std::max(device_capacity, 0));
  if ((!force_global || bytes == 0) &&
      attributes.sharedSizeBytes <= capacity &&
      bytes <= capacity - attributes.sharedSizeBytes) {
    ConfigureKernelSharedMemory(shared_kernel, name, bytes, device_capacity);
    return {bytes, 0};
  }
  // The fallback retains small static shared scalars, but its dense scratch
  // lives in separately owned global memory. No dynamic shared allocation.
  ConfigureKernelSharedMemory(global_kernel, name, 0, device_capacity);
  constexpr std::size_t alignment = 16;
  if (bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1))
    throw std::invalid_argument("CUDA global scratch alignment overflows");
  return {0, std::max(alignment, (bytes + alignment - 1) / alignment * alignment)};
}

} // namespace clqr::cuda::detail

#endif // CLQR_CUDA_SHARED_MEMORY_H_
