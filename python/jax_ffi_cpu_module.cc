#include <nanobind/nanobind.h>

#include <type_traits>

#include "python/jax_ffi_cpu.h"

namespace nb = nanobind;

namespace {

nb::dict FfiRegistrations() {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, decltype(ClqrCpuFfi),
                                      XLA_FFI_CallFrame *>,
                "CLQR must expose a typed XLA FFI handler");
  nb::dict registrations;
#ifdef CLQR_USE_FLOAT
  constexpr const char *name = "clqr_solve_f32";
#else
  constexpr const char *name = "clqr_solve_f64";
#endif
  registrations[nb::str(name)] =
      nb::capsule(reinterpret_cast<void *>(ClqrCpuFfi));
  return registrations;
}

} // namespace

NB_MODULE(_clqr_jax_cpu, module) {
  module.doc() = "CPU registration for the CLQR JAX FFI.";
  module.def("ffi_registrations", &FfiRegistrations);
#ifdef CLQR_USE_FLOAT
  module.attr("scalar_dtype") = "float32";
#else
  module.attr("scalar_dtype") = "float64";
#endif
}
