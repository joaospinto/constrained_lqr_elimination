#ifndef CLQR_CUDA_DEVICE_IO_H_
#define CLQR_CUDA_DEVICE_IO_H_

#include <cstddef>
#include <cstdint>

#include "clqr/cuda.h"

namespace clqr::cuda::detail {

// Padded device buffers used by the JAX FFI. Active dimensions are described
// by the owning shape-only Problem passed to SolvePackedDevice. Matrix rows
// use the capacities below as their device-side leading dimensions.
struct PaddedDeviceProblem {
  std::size_t state_capacity = 0;
  std::size_t control_capacity = 0;
  std::size_t mixed_capacity = 0;
  std::size_t state_constraint_capacity = 0;
  std::size_t terminal_constraint_capacity = 0;
  const Scalar *A = nullptr;
  const Scalar *B = nullptr;
  const Scalar *c = nullptr;
  const Scalar *Q = nullptr;
  const Scalar *R = nullptr;
  const Scalar *M = nullptr;
  const Scalar *q = nullptr;
  const Scalar *r = nullptr;
  const Scalar *C = nullptr;
  const Scalar *D = nullptr;
  const Scalar *d = nullptr;
  const Scalar *E = nullptr;
  const Scalar *e = nullptr;
  const Scalar *terminal_E = nullptr;
  const Scalar *terminal_e = nullptr;
  const Scalar *initial_state = nullptr;
};

struct PaddedDeviceSolution {
  std::int32_t *diagnostics = nullptr;
  Scalar *objective = nullptr;
  Scalar *states = nullptr;
  Scalar *controls = nullptr;
  Scalar *initial_multiplier = nullptr;
  Scalar *dynamics_multipliers = nullptr;
  Scalar *mixed_multipliers = nullptr;
  Scalar *state_multipliers = nullptr;
  Scalar *terminal_state_multiplier = nullptr;
};

struct DeviceTransferAudit {
  std::size_t scalar_device_to_host_bytes = 0;
  std::size_t scalar_host_to_device_bytes = 0;
  std::size_t metadata_device_to_host_bytes = 0;
  std::size_t metadata_host_to_device_bytes = 0;
};

// Executes the same native CUDA solver used by SolvePreparedView, but imports
// and exports padded scalar arrays entirely on the supplied device stream.
// Only compact structural/status metadata may cross to the host. The first
// call reserves `workspace`; subsequent calls require the same structure,
// matching SolvePreparedView's contract.
// Device allocation failures throw with requested/free/total byte diagnostics;
// the workspace can be retried and the FFI forwards the error to its caller.
SolveStatus SolvePackedDevice(const Problem &structure, Workspace &workspace,
                              const PaddedDeviceProblem &input,
                              const PaddedDeviceSolution &output,
                              void *cuda_stream,
                              const Options &options = Options{},
                              DeviceTransferAudit *audit = nullptr);

} // namespace clqr::cuda::detail

#endif // CLQR_CUDA_DEVICE_IO_H_
