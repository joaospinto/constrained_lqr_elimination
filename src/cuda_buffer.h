#ifndef CLQR_CUDA_BUFFER_H_
#define CLQR_CUDA_BUFFER_H_

#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace clqr::cuda::detail {

class DeviceAllocationError : public std::runtime_error {
public:
  DeviceAllocationError(cudaError_t error, std::size_t requested_bytes)
      : std::runtime_error(Message(error, requested_bytes)) {}

private:
  static std::string Message(cudaError_t error, std::size_t requested_bytes) {
    std::string message =
        std::string("cudaMalloc: ") + cudaGetErrorString(error) +
        "; requested_bytes=" + std::to_string(requested_bytes);
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess)
      message += "; free_bytes=" + std::to_string(free_bytes) +
                 "; total_bytes=" + std::to_string(total_bytes);
    return message;
  }
};

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { Allocate(count); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)),
        owns_(std::exchange(other.owns_, false)) {}
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      Release();
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
      owns_ = std::exchange(other.owns_, false);
    }
    return *this;
  }
  ~DeviceBuffer() { Release(); }

  void Allocate(std::size_t count) {
    Release();
    const std::size_t required = std::max<std::size_t>(count, 1);
    if (required > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::invalid_argument("CUDA device allocation size overflows");
    T *replacement = nullptr;
    const cudaError_t error = cudaMalloc(
        reinterpret_cast<void **>(&replacement), required * sizeof(T));
    if (error != cudaSuccess)
      throw DeviceAllocationError(error, required * sizeof(T));
    // Publish capacity only after allocation succeeds. A failed allocation
    // must not make Reserve believe a null buffer can hold the next solve.
    data_ = replacement;
    count_ = required;
    owns_ = true;
  }
  void Reserve(std::size_t count) {
    const std::size_t required = std::max<std::size_t>(count, 1);
    if (count_ < required)
      Allocate(required);
  }
  void Bind(T *data, std::size_t count) {
    Release();
    data_ = data;
    count_ = std::max<std::size_t>(count, 1);
    owns_ = false;
  }
  // Reuse phase-disjoint storage when it fits; otherwise retain a separately
  // owned allocation. A previous borrowed view may refer to a resized arena,
  // so it must not satisfy an owned fallback reservation by its old capacity.
  void ReserveReusing(T *scratch, std::size_t scratch_count,
                      std::size_t count) {
    if (count <= scratch_count) {
      Bind(scratch, count);
    } else {
      if (!owns_)
        Release();
      Reserve(count);
    }
  }
  void Release() {
    if (owns_ && data_ != nullptr)
      cudaFree(data_);
    data_ = nullptr;
    count_ = 0;
    owns_ = false;
  }
  T *get() { return data_; }
  const T *get() const { return data_; }
  std::size_t count() const { return count_; }

private:
  T *data_ = nullptr;
  std::size_t count_ = 0;
  bool owns_ = false;
};

} // namespace clqr::cuda::detail
#endif // CLQR_CUDA_BUFFER_H_
