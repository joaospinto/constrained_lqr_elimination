#ifndef CLQR_PYTHON_JAX_FFI_CUDA_H_
#define CLQR_PYTHON_JAX_FFI_CUDA_H_

#include <cstdint>

#include "xla/ffi/api/c_api.h"

extern "C" XLA_FFI_Error *ClqrCudaFfi(XLA_FFI_CallFrame *);
void ClqrCudaGetLastTransferAudit(std::uint64_t *scalar_device_to_host_bytes,
                                  std::uint64_t *scalar_host_to_device_bytes,
                                  std::uint64_t *metadata_device_to_host_bytes);

#endif // CLQR_PYTHON_JAX_FFI_CUDA_H_
