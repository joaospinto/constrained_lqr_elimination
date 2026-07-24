#include <cstdint>
#include <exception>
#include <initializer_list>
#include <string>

#include "python/jax_ffi_cpu.h"
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

struct ThreadState {
  clqr::Problem problem;
  clqr::Workspace workspace;
};

thread_local ThreadState thread_state;

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

ffi::Error SolveCpuImpl(
    clqr::Scalar tolerance, ffi::BufferR1<ffi::DataType::S32> dimensions,
    ScalarBufferR3 A, ScalarBufferR3 B, ScalarBufferR2 c, ScalarBufferR3 Q,
    ScalarBufferR3 R, ScalarBufferR3 M, ScalarBufferR2 q, ScalarBufferR2 r,
    ScalarBufferR3 C, ScalarBufferR3 D, ScalarBufferR2 d, ScalarBufferR3 E,
    ScalarBufferR2 e, ScalarBufferR2 terminal_Q, ScalarBufferR1 terminal_q,
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

    clqr::python::PackedProblemBuffers packed;
    packed.stage_count = static_cast<std::size_t>(stage_count);
    packed.state_capacity = static_cast<std::size_t>(state_capacity);
    packed.control_capacity = static_cast<std::size_t>(control_capacity);
    packed.mixed_capacity = static_cast<std::size_t>(mixed_capacity);
    packed.state_constraint_capacity =
        static_cast<std::size_t>(state_constraint_capacity);
    packed.terminal_constraint_capacity =
        static_cast<std::size_t>(terminal_constraint_capacity);
    packed.dimensions = dimensions.typed_data();
    packed.dimension_count = dimensions.element_count();
    packed.A = A.typed_data();
    packed.B = B.typed_data();
    packed.c = c.typed_data();
    packed.Q = Q.typed_data();
    packed.R = R.typed_data();
    packed.M = M.typed_data();
    packed.q = q.typed_data();
    packed.r = r.typed_data();
    packed.C = C.typed_data();
    packed.D = D.typed_data();
    packed.d = d.typed_data();
    packed.E = E.typed_data();
    packed.e = e.typed_data();
    packed.terminal_Q = terminal_Q.typed_data();
    packed.terminal_q = terminal_q.typed_data();
    packed.terminal_E = terminal_E.typed_data();
    packed.terminal_e = terminal_e.typed_data();
    packed.initial_state = initial_state.typed_data();

    std::string error;
    if (!clqr::python::BuildProblem(packed, &thread_state.problem, &error)) {
      return ffi::Error::InvalidArgument(std::move(error));
    }
    clqr::SolveOptions options;
    options.tolerance = tolerance;
    const std::size_t workspace_bytes =
        clqr::Workspace::RequiredBytes(thread_state.problem, options);
    if (thread_state.workspace.size() < workspace_bytes)
      thread_state.workspace.Reserve(thread_state.problem, options);
    const clqr::SolutionView solution = clqr::Solve(
        thread_state.problem, thread_state.workspace, options);

    clqr::python::PackedSolutionBuffers output;
    output.diagnostics = diagnostics->typed_data();
    output.objective = objective->typed_data();
    output.states = states->typed_data();
    output.controls = controls->typed_data();
    output.initial_multiplier = initial_multiplier->typed_data();
    output.dynamics_multipliers = dynamics_multipliers->typed_data();
    output.mixed_multipliers = mixed_multipliers->typed_data();
    output.state_multipliers = state_multipliers->typed_data();
    output.terminal_state_multiplier = terminal_state_multiplier->typed_data();
    clqr::python::WriteSolution(packed, solution, output);
    return ffi::Error::Success();
  } catch (const std::exception &exception) {
    return ffi::Error::Internal(std::string("CLQR CPU FFI failed: ") +
                                exception.what());
  }
}

} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(ClqrCpuFfi, SolveCpuImpl,
                              ffi::Ffi::Bind()
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
