#include <nanobind/nanobind.h>

#include <type_traits>

#include "python/jax_ffi_metal.h"

namespace nb = nanobind;

namespace {

nb::dict FfiRegistrations() {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, decltype(ClqrMetalFfi),
                                      XLA_FFI_CallFrame *>,
                "CLQR Metal must expose a typed XLA FFI handler");
  nb::dict registrations;
#if defined(CLQR_USE_FLOAT) && (defined(__arm64__) || defined(__aarch64__))
  registrations[nb::str("clqr_metal_solve_f32")] =
      nb::capsule(reinterpret_cast<void *>(ClqrMetalFfi));
#endif
  return registrations;
}

} // namespace

NB_MODULE(_clqr_metal, module) {
  module.doc() = "Native Metal registration for the CLQR JAX FFI.";
  module.def("ffi_registrations", &FfiRegistrations);
#ifdef CLQR_USE_FLOAT
  module.attr("scalar_dtype") = "float32";
#if defined(__arm64__) || defined(__aarch64__)
  module.attr("supported") = true;
  module.attr("unsupported_reason") = nb::none();
#else
  module.attr("supported") = false;
  module.attr("unsupported_reason") =
      "the CLQR Metal backend requires an Apple-silicon host";
#endif
#else
  module.attr("scalar_dtype") = "float64";
  module.attr("supported") = false;
  module.attr("unsupported_reason") =
      "Apple Metal does not support float64; rebuild CLQR with "
      "--config=fp32 to use backend='metal'";
#endif
}
