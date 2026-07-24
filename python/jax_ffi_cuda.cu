#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "clqr/cuda.h"
#include "python/jax_ffi_cuda.h"
#include "python/jax_ffi_problem.h"
#include "src/cuda_device_io.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

#ifdef CLQR_USE_FLOAT
constexpr ffi::DataType kScalarType = ffi::DataType::F32;
#else
constexpr ffi::DataType kScalarType = ffi::DataType::F64;
#endif

using ScalarBufferR1 = ffi::BufferR1<kScalarType>;
using ScalarBufferR2 = ffi::BufferR2<kScalarType>;
using ScalarBufferR3 = ffi::BufferR3<kScalarType>;
using ScalarResultR1 = ffi::ResultBufferR1<kScalarType>;
using ScalarResultR2 = ffi::ResultBufferR2<kScalarType>;

void CudaCheck(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename Buffer>
bool HasShape(const Buffer &buffer,
              std::initializer_list<std::int64_t> expected) {
  const auto actual = buffer.dimensions();
  if (actual.size() != expected.size())
    return false;
  std::size_t index = 0;
  for (const std::int64_t dimension : expected) {
    if (actual[index++] != dimension)
      return false;
  }
  return true;
}

template <typename T> class PinnedBuffer {
public:
  PinnedBuffer() = default;
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;

  ~PinnedBuffer() {
    if (data_ != nullptr)
      cudaFreeHost(data_);
  }

  void Reserve(std::size_t count) {
    if (count <= capacity_)
      return;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::length_error("JAX staging byte count overflows");
    }
    T *replacement = nullptr;
    CudaCheck(cudaMallocHost(reinterpret_cast<void **>(&replacement),
                             count * sizeof(T)),
              "cudaMallocHost for JAX staging");
    if (data_ != nullptr) {
      const cudaError_t free_status = cudaFreeHost(data_);
      if (free_status != cudaSuccess) {
        cudaFreeHost(replacement);
        CudaCheck(free_status, "cudaFreeHost");
      }
    }
    data_ = replacement;
    capacity_ = count;
  }

  T *data() { return data_; }
  const T *data() const { return data_; }

private:
  T *data_ = nullptr;
  std::size_t capacity_ = 0;
};

struct Staging {
  PinnedBuffer<std::int32_t> input_dimensions;
  cudaEvent_t output_ready = nullptr;
  bool output_pending = false;

  ~Staging() {
    if (output_pending)
      cudaEventSynchronize(output_ready);
    if (output_ready != nullptr)
      cudaEventDestroy(output_ready);
  }

  // A workspace owns the compact device buffers read by the asynchronous
  // padded-output scatter. A later invocation on another XLA stream must not
  // repack or reallocate those buffers until that export has completed.
  void WaitForPreviousOutput() {
    if (!output_pending)
      return;
    CudaCheck(cudaEventSynchronize(output_ready),
              "wait for prior device-resident JAX output");
    output_pending = false;
  }

  void RecordOutput(cudaStream_t stream) {
    if (output_ready == nullptr) {
      CudaCheck(cudaEventCreateWithFlags(&output_ready, cudaEventDisableTiming),
                "create device-resident JAX output event");
    }
    CudaCheck(cudaEventRecord(output_ready, stream),
              "record device-resident JAX output event");
    output_pending = true;
  }
};

struct DeviceState {
  int device = 0;
  Staging staging;
  clqr::Problem problem;
  clqr::cuda::Workspace workspace;
  std::vector<std::int32_t> structure_key;

  ~DeviceState() { cudaSetDevice(device); }
};

struct ThreadState {
  std::unordered_map<int, std::unique_ptr<DeviceState>> devices;
};

thread_local ThreadState thread_state;

std::atomic<std::uint64_t> last_scalar_device_to_host_bytes{0};
std::atomic<std::uint64_t> last_scalar_host_to_device_bytes{0};
std::atomic<std::uint64_t> last_metadata_device_to_host_bytes{0};

ffi::Error SolveCudaImpl(
    cudaStream_t stream, clqr::Scalar tolerance,
    ffi::BufferR1<ffi::DataType::S32> dimensions, ScalarBufferR3 A,
    ScalarBufferR3 B, ScalarBufferR2 c, ScalarBufferR3 Q, ScalarBufferR3 R,
    ScalarBufferR3 M, ScalarBufferR2 q, ScalarBufferR2 r, ScalarBufferR3 C,
    ScalarBufferR3 D, ScalarBufferR2 d, ScalarBufferR3 E, ScalarBufferR2 e,
    ScalarBufferR2 terminal_E, ScalarBufferR1 terminal_e,
    ScalarBufferR1 initial_state,
    ffi::ResultBufferR1<ffi::DataType::S32> diagnostics,
    ScalarResultR1 objective, ScalarResultR2 states, ScalarResultR2 controls,
    ScalarResultR1 initial_multiplier, ScalarResultR2 dynamics_multipliers,
    ScalarResultR2 mixed_multipliers, ScalarResultR2 state_multipliers,
    ScalarResultR1 terminal_state_multiplier) {
  try {
    const auto a_shape = A.dimensions();
    if (a_shape.size() != 3 || a_shape[1] != a_shape[2]) {
      return ffi::Error::InvalidArgument(
          "A must have shape (stages, max_state, max_state)");
    }
    const std::int64_t stage_count = a_shape[0];
    const std::int64_t state_capacity = a_shape[1];
    if (stage_count < 0 || state_capacity < 0) {
      return ffi::Error::InvalidArgument("negative padded problem dimension");
    }
    const auto b_shape = B.dimensions();
    if (b_shape.size() != 3 || b_shape[0] != stage_count ||
        b_shape[1] != state_capacity || b_shape[2] < 0) {
      return ffi::Error::InvalidArgument(
          "B must have shape (stages, max_state, max_control)");
    }
    const std::int64_t control_capacity = b_shape[2];
    const auto c_constraint_shape = C.dimensions();
    const auto e_constraint_shape = E.dimensions();
    const auto terminal_constraint_shape = terminal_E.dimensions();
    if (c_constraint_shape.size() != 3 ||
        c_constraint_shape[0] != stage_count ||
        c_constraint_shape[2] != state_capacity ||
        e_constraint_shape.size() != 3 ||
        e_constraint_shape[0] != stage_count ||
        e_constraint_shape[2] != state_capacity ||
        terminal_constraint_shape.size() != 2 ||
        terminal_constraint_shape[1] != state_capacity) {
      return ffi::Error::InvalidArgument(
          "constraint matrices have inconsistent padded shapes");
    }
    const std::int64_t mixed_capacity = c_constraint_shape[1];
    const std::int64_t state_constraint_capacity = e_constraint_shape[1];
    const std::int64_t terminal_constraint_capacity =
        terminal_constraint_shape[0];

    if (!HasShape(dimensions, {4 * stage_count + 2}) ||
        !HasShape(c, {stage_count, state_capacity}) ||
        !HasShape(Q, {stage_count + 1, state_capacity, state_capacity}) ||
        !HasShape(R, {stage_count, control_capacity, control_capacity}) ||
        !HasShape(M, {stage_count, state_capacity, control_capacity}) ||
        !HasShape(q, {stage_count + 1, state_capacity}) ||
        !HasShape(r, {stage_count, control_capacity}) ||
        !HasShape(D, {stage_count, mixed_capacity, control_capacity}) ||
        !HasShape(d, {stage_count, mixed_capacity}) ||
        !HasShape(e, {stage_count, state_constraint_capacity}) ||
        !HasShape(terminal_e, {terminal_constraint_capacity}) ||
        !HasShape(initial_state, {state_capacity})) {
      return ffi::Error::InvalidArgument(
          "one or more packed input arrays have inconsistent shapes");
    }
    if (!HasShape(*diagnostics, {3}) || !HasShape(*objective, {1}) ||
        !HasShape(*states, {stage_count + 1, state_capacity}) ||
        !HasShape(*controls, {stage_count, control_capacity}) ||
        !HasShape(*initial_multiplier, {state_capacity}) ||
        !HasShape(*dynamics_multipliers, {stage_count, state_capacity}) ||
        !HasShape(*mixed_multipliers, {stage_count, mixed_capacity}) ||
        !HasShape(*state_multipliers,
                  {stage_count, state_constraint_capacity}) ||
        !HasShape(*terminal_state_multiplier, {terminal_constraint_capacity})) {
      return ffi::Error::InvalidArgument(
          "one or more packed output arrays have inconsistent shapes");
    }

    int device = 0;
    CudaCheck(cudaGetDevice(&device), "cudaGetDevice");
    auto &device_pointer = thread_state.devices[device];
    if (!device_pointer) {
      device_pointer = std::make_unique<DeviceState>();
      device_pointer->device = device;
    }
    DeviceState &device_state = *device_pointer;
    Staging &staging = device_state.staging;
    staging.WaitForPreviousOutput();
    staging.input_dimensions.Reserve(dimensions.element_count());
    if (dimensions.element_count() > 0) {
      CudaCheck(cudaMemcpyAsync(
                    staging.input_dimensions.data(), dimensions.typed_data(),
                    dimensions.element_count() * sizeof(std::int32_t),
                    cudaMemcpyDeviceToHost, stream),
                "copy JAX dimensions to pinned host staging");
    }

    clqr::python::PackedProblemBuffers packed;
    packed.stage_count = static_cast<std::size_t>(stage_count);
    packed.state_capacity = static_cast<std::size_t>(state_capacity);
    packed.control_capacity = static_cast<std::size_t>(control_capacity);
    packed.mixed_capacity = static_cast<std::size_t>(mixed_capacity);
    packed.state_constraint_capacity =
        static_cast<std::size_t>(state_constraint_capacity);
    packed.terminal_constraint_capacity =
        static_cast<std::size_t>(terminal_constraint_capacity);
    packed.dimensions = staging.input_dimensions.data();
    packed.dimension_count = dimensions.element_count();
    CudaCheck(cudaStreamSynchronize(stream),
              "wait for JAX CLQR dimension metadata");
    last_metadata_device_to_host_bytes.store(dimensions.element_count() *
                                                 sizeof(std::int32_t),
                                             std::memory_order_relaxed);
    last_scalar_device_to_host_bytes.store(0, std::memory_order_relaxed);
    last_scalar_host_to_device_bytes.store(0, std::memory_order_relaxed);

    const std::vector<std::int32_t> structure_key(
        packed.dimensions, packed.dimensions + packed.dimension_count);
    if (device_state.structure_key != structure_key) {
      std::string error;
      if (!clqr::python::BuildProblemStructure(packed, &device_state.problem,
                                               &error)) {
        return ffi::Error::InvalidArgument(std::move(error));
      }
      device_state.workspace = clqr::cuda::Workspace{};
      device_state.structure_key = structure_key;
    }

    clqr::cuda::detail::PaddedDeviceProblem device_input;
    device_input.state_capacity = packed.state_capacity;
    device_input.control_capacity = packed.control_capacity;
    device_input.mixed_capacity = packed.mixed_capacity;
    device_input.state_constraint_capacity = packed.state_constraint_capacity;
    device_input.terminal_constraint_capacity =
        packed.terminal_constraint_capacity;
    device_input.A = A.typed_data();
    device_input.B = B.typed_data();
    device_input.c = c.typed_data();
    device_input.Q = Q.typed_data();
    device_input.R = R.typed_data();
    device_input.M = M.typed_data();
    device_input.q = q.typed_data();
    device_input.r = r.typed_data();
    device_input.C = C.typed_data();
    device_input.D = D.typed_data();
    device_input.d = d.typed_data();
    device_input.E = E.typed_data();
    device_input.e = e.typed_data();
    device_input.terminal_E = terminal_E.typed_data();
    device_input.terminal_e = terminal_e.typed_data();
    device_input.initial_state = initial_state.typed_data();

    clqr::cuda::detail::PaddedDeviceSolution device_output;
    device_output.diagnostics = diagnostics->typed_data();
    device_output.objective = objective->typed_data();
    device_output.states = states->typed_data();
    device_output.controls = controls->typed_data();
    device_output.initial_multiplier = initial_multiplier->typed_data();
    device_output.dynamics_multipliers = dynamics_multipliers->typed_data();
    device_output.mixed_multipliers = mixed_multipliers->typed_data();
    device_output.state_multipliers = state_multipliers->typed_data();
    device_output.terminal_state_multiplier =
        terminal_state_multiplier->typed_data();

    clqr::cuda::Options options;
    options.device = device;
    options.tolerance = tolerance;
    clqr::cuda::detail::DeviceTransferAudit audit;
    clqr::cuda::detail::SolvePackedDevice(
        device_state.problem, device_state.workspace, device_input,
        device_output, reinterpret_cast<void *>(stream), options, &audit);
    last_scalar_device_to_host_bytes.store(audit.scalar_device_to_host_bytes,
                                           std::memory_order_relaxed);
    last_scalar_host_to_device_bytes.store(audit.scalar_host_to_device_bytes,
                                           std::memory_order_relaxed);
    staging.RecordOutput(stream);
    return ffi::Error::Success();
  } catch (const std::exception &exception) {
    return ffi::Error::Internal(std::string("CLQR CUDA FFI failed: ") +
                                exception.what());
  }
}

} // namespace

void ClqrCudaGetLastTransferAudit(
    std::uint64_t *scalar_device_to_host_bytes,
    std::uint64_t *scalar_host_to_device_bytes,
    std::uint64_t *metadata_device_to_host_bytes) {
  if (scalar_device_to_host_bytes != nullptr) {
    *scalar_device_to_host_bytes =
        last_scalar_device_to_host_bytes.load(std::memory_order_relaxed);
  }
  if (scalar_host_to_device_bytes != nullptr) {
    *scalar_host_to_device_bytes =
        last_scalar_host_to_device_bytes.load(std::memory_order_relaxed);
  }
  if (metadata_device_to_host_bytes != nullptr) {
    *metadata_device_to_host_bytes =
        last_metadata_device_to_host_bytes.load(std::memory_order_relaxed);
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ClqrCudaFfi, SolveCudaImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<clqr::Scalar>("tolerance")
                                  .Arg<ffi::BufferR1<ffi::DataType::S32>>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR2>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR2>()
                                  .Arg<ScalarBufferR2>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR2>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR2>()
                                  .Arg<ScalarBufferR2>()
                                  .Arg<ScalarBufferR1>()
                                  .Arg<ScalarBufferR1>()
                                  .Ret<ffi::BufferR1<ffi::DataType::S32>>()
                                  .Ret<ScalarBufferR1>()
                                  .Ret<ScalarBufferR2>()
                                  .Ret<ScalarBufferR2>()
                                  .Ret<ScalarBufferR1>()
                                  .Ret<ScalarBufferR2>()
                                  .Ret<ScalarBufferR2>()
                                  .Ret<ScalarBufferR2>()
                                  .Ret<ScalarBufferR1>());
