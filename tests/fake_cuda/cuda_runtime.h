#ifndef CLQR_TESTS_FAKE_CUDA_RUNTIME_H_
#define CLQR_TESTS_FAKE_CUDA_RUNTIME_H_

// Minimal declarations used only by the macOS host-only CUDA syntax check.
// A real CUDA build always finds NVIDIA's cuda_runtime.h before this directory.

#include <cstddef>

#define __host__
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))

struct __attribute__((device_builtin)) uint3 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};
struct __attribute__((device_builtin)) dim3 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
  constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
  constexpr dim3(uint3 value) : x(value.x), y(value.y), z(value.z) {}
  constexpr operator uint3() const { return uint3{x, y, z}; }
};

extern const __device__ uint3 threadIdx;
extern const __device__ uint3 blockIdx;
extern const __device__ dim3 blockDim;
extern const __device__ dim3 gridDim;

extern "C" __device__ int atomicCAS(int *, int, int);
extern "C" __device__ int atomicExch(int *, int);
extern "C" __device__ void __syncthreads();
extern "C" __device__ void __syncwarp(unsigned int = 0xffffffffu);
template <typename T>
__device__ T __shfl_down_sync(unsigned int, T, unsigned int);
template <typename T> __device__ T __shfl_sync(unsigned int, T, int);
__device__ double fmax(double, double);
__device__ float fmax(float, float);
__device__ double sqrt(double);
__device__ float sqrt(float);

using cudaError_t = int;
using cudaEvent_t = void *;
using cudaStream_t = void *;
constexpr cudaError_t cudaSuccess = 0;
constexpr int cudaMemcpyHostToDevice = 1;
constexpr int cudaMemcpyDeviceToHost = 2;

struct cudaDeviceProp {
  char name[256];
  std::size_t totalGlobalMem;
  int major;
  int minor;
};

extern "C" cudaError_t cudaMalloc(void **, std::size_t);
extern "C" cudaError_t cudaMallocHost(void **, std::size_t);
extern "C" cudaError_t cudaFree(void *);
extern "C" cudaError_t cudaFreeHost(void *);
extern "C" cudaError_t cudaMemcpy(void *, const void *, std::size_t, int);
extern "C" cudaError_t cudaMemcpyAsync(void *, const void *, std::size_t, int,
                                       cudaStream_t = nullptr);
extern "C" cudaError_t cudaMemset(void *, int, std::size_t);
extern "C" cudaError_t cudaMemsetAsync(void *, int, std::size_t,
                                       cudaStream_t = nullptr);
extern "C" cudaError_t cudaSetDevice(int);
extern "C" cudaError_t cudaGetDeviceCount(int *);
extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp *, int);
extern "C" const char *cudaGetErrorString(cudaError_t);
extern "C" cudaError_t cudaGetLastError();
extern "C" cudaError_t cudaEventCreate(cudaEvent_t *);
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t);
extern "C" cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr);
extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t);
extern "C" cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);

extern "C" cudaError_t cudaConfigureCall(dim3, dim3, std::size_t = 0,
                                         cudaStream_t = nullptr);
extern "C" cudaError_t cudaSetupArgument(const void *, std::size_t,
                                         std::size_t);
extern "C" cudaError_t cudaLaunch(const void *);

#endif // CLQR_TESTS_FAKE_CUDA_RUNTIME_H_
