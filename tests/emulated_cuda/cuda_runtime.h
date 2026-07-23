#ifndef CLQR_TESTS_EMULATED_CUDA_RUNTIME_H_
#define CLQR_TESTS_EMULATED_CUDA_RUNTIME_H_

#define __device__
#define __global__
#define __host__
#define __shared__

struct EmulatedCudaIndex {
  int x = 0;
  int y = 0;
  int z = 0;
};

inline EmulatedCudaIndex threadIdx;
inline EmulatedCudaIndex blockIdx;
inline EmulatedCudaIndex blockDim{1, 1, 1};
inline EmulatedCudaIndex gridDim{1, 1, 1};

inline void __syncthreads() {}
inline int atomicCAS(int *address, int compare, int value) {
  const int old = *address;
  if (old == compare)
    *address = value;
  return old;
}
inline int atomicExch(int *address, int value) {
  const int old = *address;
  *address = value;
  return old;
}

#endif // CLQR_TESTS_EMULATED_CUDA_RUNTIME_H_
