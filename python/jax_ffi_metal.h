#ifndef CLQR_PYTHON_JAX_FFI_METAL_H_
#define CLQR_PYTHON_JAX_FFI_METAL_H_

#include "xla/ffi/api/c_api.h"

extern "C" XLA_FFI_Error *ClqrMetalFfi(XLA_FFI_CallFrame *call_frame);

#endif // CLQR_PYTHON_JAX_FFI_METAL_H_
