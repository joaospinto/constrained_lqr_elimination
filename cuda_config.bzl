"""Validated build settings for the CUDA backend's static capacities."""

load("@bazel_skylib//rules:common_settings.bzl", "int_flag")

CUDA_CAPACITIES = [
    ("cuda_max_state_dimension", "CLQR_CUDA_MAX_STATE_DIMENSION", 8),
    ("cuda_max_control_dimension", "CLQR_CUDA_MAX_CONTROL_DIMENSION", 8),
    ("cuda_max_mixed_constraints", "CLQR_CUDA_MAX_MIXED_CONSTRAINTS", 8),
    ("cuda_max_state_constraints", "CLQR_CUDA_MAX_STATE_CONSTRAINTS", 8),
]

_MIN_CAPACITY = 1
_MAX_CAPACITY = 16

def declare_cuda_capacity_flags():
    """Declares the public flags and their select-compatible conditions."""
    for flag_name, _, default_value in CUDA_CAPACITIES:
        int_flag(
            name = flag_name,
            build_setting_default = default_value,
        )
        for value in range(_MIN_CAPACITY, _MAX_CAPACITY + 1):
            native.config_setting(
                name = "%s_%d" % (flag_name, value),
                flag_values = {":" + flag_name: str(value)},
            )

def cuda_capacity_defines():
    """Returns matching preprocessor definitions for every capacity flag."""
    result = []
    for flag_name, define_name, _ in CUDA_CAPACITIES:
        result += select({
            ":%s_%d" % (flag_name, value): [
                "%s=%d" % (define_name, value),
            ]
            for value in range(_MIN_CAPACITY, _MAX_CAPACITY + 1)
        })
    return result
