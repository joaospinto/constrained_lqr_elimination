#include <nanobind/nanobind.h>

#include <cstdint>
#include <type_traits>

#include "python/jax_ffi_cuda.h"

namespace nb = nanobind;

namespace {

nb::dict FfiRegistrations() {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, decltype(ClqrCudaFfi),
                                      XLA_FFI_CallFrame *>,
                "CLQR must expose a typed XLA FFI handler");
  nb::dict registrations;
#ifdef CLQR_USE_FLOAT
  constexpr const char *name = "clqr_solve_f32";
#else
  constexpr const char *name = "clqr_solve_f64";
#endif
  registrations[nb::str(name)] =
      nb::capsule(reinterpret_cast<void *>(ClqrCudaFfi));
  return registrations;
}

nb::dict LastTransferAudit() {
  std::uint64_t scalar_device_to_host_bytes = 0;
  std::uint64_t scalar_host_to_device_bytes = 0;
  std::uint64_t metadata_device_to_host_bytes = 0;
  ClqrCudaGetLastTransferAudit(&scalar_device_to_host_bytes,
                               &scalar_host_to_device_bytes,
                               &metadata_device_to_host_bytes);
  nb::dict audit;
  audit["scalar_device_to_host_bytes"] = scalar_device_to_host_bytes;
  audit["scalar_host_to_device_bytes"] = scalar_host_to_device_bytes;
  audit["metadata_device_to_host_bytes"] = metadata_device_to_host_bytes;
  return audit;
}

} // namespace

NB_MODULE(_clqr_cuda, module) {
  module.doc() = "CUDA registration for the CLQR JAX FFI.";
  module.def("ffi_registrations", &FfiRegistrations);
  module.def("last_transfer_audit", &LastTransferAudit);
#ifdef CLQR_USE_FLOAT
  module.attr("scalar_dtype") = "float32";
#else
  module.attr("scalar_dtype") = "float64";
#endif
}
