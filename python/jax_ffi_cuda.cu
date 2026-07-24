#include <cuda_runtime.h>

#include <algorithm>
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

std::size_t CheckedSum(std::initializer_list<std::size_t> values,
                       const char *description) {
  std::size_t total = 0;
  for (std::size_t value : values) {
    if (value > std::numeric_limits<std::size_t>::max() - total) {
      throw std::invalid_argument(std::string(description) +
                                  " element count overflows");
    }
    total += value;
  }
  return total;
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
  PinnedBuffer<clqr::Scalar> input_scalars;
  PinnedBuffer<std::int32_t> output_diagnostics;
  PinnedBuffer<clqr::Scalar> output_scalars;
  cudaEvent_t output_ready = nullptr;
  bool output_pending = false;

  ~Staging() {
    if (output_pending)
      cudaEventSynchronize(output_ready);
    if (output_ready != nullptr)
      cudaEventDestroy(output_ready);
  }

  void WaitForPreviousOutput() {
    if (!output_pending)
      return;
    CudaCheck(cudaEventSynchronize(output_ready),
              "wait for prior JAX output staging");
    output_pending = false;
  }

  void RecordOutput(cudaStream_t stream) {
    if (output_ready == nullptr) {
      CudaCheck(cudaEventCreateWithFlags(&output_ready, cudaEventDisableTiming),
                "create JAX output staging event");
    }
    CudaCheck(cudaEventRecord(output_ready, stream),
              "record JAX output staging event");
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

template <typename Buffer>
const clqr::Scalar *EnqueueInput(const Buffer &source,
                                 clqr::Scalar **destination,
                                 cudaStream_t stream) {
  const std::size_t count = source.element_count();
  clqr::Scalar *result = *destination;
  if (count > 0) {
    CudaCheck(cudaMemcpyAsync(result, source.typed_data(),
                              count * sizeof(clqr::Scalar),
                              cudaMemcpyDeviceToHost, stream),
              "copy JAX input to pinned host staging");
    *destination += count;
  }
  return result;
}

template <typename Result>
void EnqueueOutput(const clqr::Scalar **source, Result &destination,
                   cudaStream_t stream) {
  const std::size_t count = destination->element_count();
  if (count > 0) {
    CudaCheck(cudaMemcpyAsync(destination->typed_data(), *source,
                              count * sizeof(clqr::Scalar),
                              cudaMemcpyHostToDevice, stream),
              "copy pinned CLQR result to JAX output");
    *source += count;
  }
}

template <typename T> T *Offset(T *pointer, std::size_t offset) {
  return offset == 0 ? pointer : pointer + offset;
}

void CopyVector(const clqr::VectorView &source, clqr::Scalar *destination) {
  if (source.size == 0)
    return;
  std::copy_n(source.data, source.size, destination);
}

void WriteCudaSolution(const clqr::python::PackedProblemBuffers &packed,
                       const clqr::cuda::SolutionView &solution,
                       const clqr::python::PackedSolutionBuffers &output) {
  output.diagnostics[0] = static_cast<std::int32_t>(solution.status);
  output.diagnostics[1] = 0;
  output.diagnostics[2] = 0;
  output.objective[0] = solution.objective;

  const std::size_t nx = packed.state_capacity;
  const std::size_t nu = packed.control_capacity;
  const std::size_t nc = packed.mixed_capacity;
  const std::size_t ne = packed.state_constraint_capacity;
  if ((packed.stage_count + 1) * nx > 0) {
    std::fill_n(output.states, (packed.stage_count + 1) * nx, clqr::Scalar{0});
  }
  if (packed.stage_count * nu > 0) {
    std::fill_n(output.controls, packed.stage_count * nu, clqr::Scalar{0});
  }
  if (nx > 0) {
    std::fill_n(output.initial_multiplier, nx, clqr::Scalar{0});
    std::fill_n(output.dynamics_multipliers, packed.stage_count * nx,
                clqr::Scalar{0});
  }
  if (packed.stage_count * nc > 0) {
    std::fill_n(output.mixed_multipliers, packed.stage_count * nc,
                clqr::Scalar{0});
  }
  if (packed.stage_count * ne > 0) {
    std::fill_n(output.state_multipliers, packed.stage_count * ne,
                clqr::Scalar{0});
  }
  if (packed.terminal_constraint_capacity > 0) {
    std::fill_n(output.terminal_state_multiplier,
                packed.terminal_constraint_capacity, clqr::Scalar{0});
  }

  for (std::size_t node = 0; node < solution.state_count; ++node) {
    CopyVector(solution.states[node], Offset(output.states, node * nx));
  }
  for (std::size_t stage = 0; stage < solution.control_count; ++stage) {
    CopyVector(solution.controls[stage], Offset(output.controls, stage * nu));
  }
  CopyVector(solution.initial_multiplier, output.initial_multiplier);
  for (std::size_t stage = 0; stage < solution.dynamics_multiplier_count;
       ++stage) {
    CopyVector(solution.dynamics_multipliers[stage],
               Offset(output.dynamics_multipliers, stage * nx));
  }
  for (std::size_t stage = 0; stage < solution.mixed_multiplier_count;
       ++stage) {
    CopyVector(solution.mixed_multipliers[stage],
               Offset(output.mixed_multipliers, stage * nc));
  }
  for (std::size_t stage = 0; stage < solution.state_multiplier_count;
       ++stage) {
    CopyVector(solution.state_multipliers[stage],
               Offset(output.state_multipliers, stage * ne));
  }
  CopyVector(solution.terminal_state_multiplier,
             output.terminal_state_multiplier);
}

ffi::Error SolveCudaImpl(
    cudaStream_t stream, clqr::Scalar tolerance,
    ffi::BufferR1<ffi::DataType::S32> dimensions, ScalarBufferR3 A,
    ScalarBufferR3 B, ScalarBufferR2 c, ScalarBufferR3 Q, ScalarBufferR3 R,
    ScalarBufferR3 M, ScalarBufferR2 q, ScalarBufferR2 r, ScalarBufferR3 C,
    ScalarBufferR3 D, ScalarBufferR2 d, ScalarBufferR3 E, ScalarBufferR2 e,
    ScalarBufferR2 terminal_Q, ScalarBufferR1 terminal_q,
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
        !HasShape(Q, {stage_count, state_capacity, state_capacity}) ||
        !HasShape(R, {stage_count, control_capacity, control_capacity}) ||
        !HasShape(M, {stage_count, state_capacity, control_capacity}) ||
        !HasShape(q, {stage_count, state_capacity}) ||
        !HasShape(r, {stage_count, control_capacity}) ||
        !HasShape(D, {stage_count, mixed_capacity, control_capacity}) ||
        !HasShape(d, {stage_count, mixed_capacity}) ||
        !HasShape(e, {stage_count, state_constraint_capacity}) ||
        !HasShape(terminal_Q, {state_capacity, state_capacity}) ||
        !HasShape(terminal_q, {state_capacity}) ||
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
    const std::size_t input_scalar_count =
        CheckedSum({A.element_count(), B.element_count(), c.element_count(),
                    Q.element_count(), R.element_count(), M.element_count(),
                    q.element_count(), r.element_count(), C.element_count(),
                    D.element_count(), d.element_count(), E.element_count(),
                    e.element_count(), terminal_Q.element_count(),
                    terminal_q.element_count(), terminal_E.element_count(),
                    terminal_e.element_count(), initial_state.element_count()},
                   "JAX CLQR input");
    staging.input_dimensions.Reserve(dimensions.element_count());
    staging.input_scalars.Reserve(input_scalar_count);
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
    clqr::Scalar *input_cursor = staging.input_scalars.data();
    packed.A = EnqueueInput(A, &input_cursor, stream);
    packed.B = EnqueueInput(B, &input_cursor, stream);
    packed.c = EnqueueInput(c, &input_cursor, stream);
    packed.Q = EnqueueInput(Q, &input_cursor, stream);
    packed.R = EnqueueInput(R, &input_cursor, stream);
    packed.M = EnqueueInput(M, &input_cursor, stream);
    packed.q = EnqueueInput(q, &input_cursor, stream);
    packed.r = EnqueueInput(r, &input_cursor, stream);
    packed.C = EnqueueInput(C, &input_cursor, stream);
    packed.D = EnqueueInput(D, &input_cursor, stream);
    packed.d = EnqueueInput(d, &input_cursor, stream);
    packed.E = EnqueueInput(E, &input_cursor, stream);
    packed.e = EnqueueInput(e, &input_cursor, stream);
    packed.terminal_Q = EnqueueInput(terminal_Q, &input_cursor, stream);
    packed.terminal_q = EnqueueInput(terminal_q, &input_cursor, stream);
    packed.terminal_E = EnqueueInput(terminal_E, &input_cursor, stream);
    packed.terminal_e = EnqueueInput(terminal_e, &input_cursor, stream);
    packed.initial_state = EnqueueInput(initial_state, &input_cursor, stream);
    CudaCheck(cudaStreamSynchronize(stream), "wait for JAX CLQR input staging");

    std::string error;
    if (!clqr::python::BuildProblem(packed, &device_state.problem, &error)) {
      return ffi::Error::InvalidArgument(std::move(error));
    }
    clqr::cuda::Options options;
    options.device = device;
    options.tolerance = tolerance;
    const std::vector<std::int32_t> structure_key(
        packed.dimensions, packed.dimensions + packed.dimension_count);
    clqr::cuda::SolutionView solution;
    if (device_state.structure_key == structure_key) {
      solution = clqr::cuda::SolvePreparedView(
          device_state.problem, device_state.workspace, options);
    } else {
      solution = clqr::cuda::SolveView(device_state.problem,
                                       device_state.workspace, options);
      if (solution.state_count == packed.stage_count + 1 &&
          solution.control_count == packed.stage_count) {
        device_state.structure_key = structure_key;
      } else {
        device_state.structure_key.clear();
      }
    }

    const std::size_t output_scalar_count = CheckedSum(
        {objective->element_count(), states->element_count(),
         controls->element_count(), initial_multiplier->element_count(),
         dynamics_multipliers->element_count(),
         mixed_multipliers->element_count(), state_multipliers->element_count(),
         terminal_state_multiplier->element_count()},
        "JAX CLQR output");
    staging.output_diagnostics.Reserve(diagnostics->element_count());
    staging.output_scalars.Reserve(output_scalar_count);
    clqr::python::PackedSolutionBuffers host_output;
    host_output.diagnostics = staging.output_diagnostics.data();
    clqr::Scalar *output_cursor = staging.output_scalars.data();
    host_output.objective = output_cursor;
    output_cursor += objective->element_count();
    host_output.states = output_cursor;
    output_cursor += states->element_count();
    host_output.controls = output_cursor;
    output_cursor += controls->element_count();
    host_output.initial_multiplier = output_cursor;
    output_cursor += initial_multiplier->element_count();
    host_output.dynamics_multipliers = output_cursor;
    output_cursor += dynamics_multipliers->element_count();
    host_output.mixed_multipliers = output_cursor;
    output_cursor += mixed_multipliers->element_count();
    host_output.state_multipliers = output_cursor;
    output_cursor += state_multipliers->element_count();
    host_output.terminal_state_multiplier = output_cursor;
    WriteCudaSolution(packed, solution, host_output);

    CudaCheck(cudaMemcpyAsync(
                  diagnostics->typed_data(), staging.output_diagnostics.data(),
                  diagnostics->element_count() * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice, stream),
              "copy CLQR diagnostics to JAX output");
    const clqr::Scalar *copy_cursor = staging.output_scalars.data();
    EnqueueOutput(&copy_cursor, objective, stream);
    EnqueueOutput(&copy_cursor, states, stream);
    EnqueueOutput(&copy_cursor, controls, stream);
    EnqueueOutput(&copy_cursor, initial_multiplier, stream);
    EnqueueOutput(&copy_cursor, dynamics_multipliers, stream);
    EnqueueOutput(&copy_cursor, mixed_multipliers, stream);
    EnqueueOutput(&copy_cursor, state_multipliers, stream);
    EnqueueOutput(&copy_cursor, terminal_state_multiplier, stream);
    staging.RecordOutput(stream);
    return ffi::Error::Success();
  } catch (const std::exception &exception) {
    return ffi::Error::Internal(std::string("CLQR CUDA FFI failed: ") +
                                exception.what());
  }
}

} // namespace

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
