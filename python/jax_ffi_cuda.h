#ifndef CLQR_PYTHON_JAX_FFI_CUDA_H_
#define CLQR_PYTHON_JAX_FFI_CUDA_H_

#include "xla/ffi/api/c_api.h"

extern "C" XLA_FFI_Error *ClqrCudaFfi(XLA_FFI_CallFrame *);

#endif // CLQR_PYTHON_JAX_FFI_CUDA_H_
