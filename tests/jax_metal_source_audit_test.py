import os
import pathlib


def _source(name):
    root = pathlib.Path(os.environ["TEST_SRCDIR"])
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} in runfiles; found {matches}")
    return matches[0].read_text()


def test_metal_ffi_uses_one_precise_native_gpu_submission():
    source = _source("jax_ffi_metal.cc")
    for forbidden_cpu_path in (
        '#include "clqr/clqr.h"',
        "BuildProblem",
        "SolvePreparedView",
        "iterative_refinement",
    ):
        assert forbidden_cpu_path not in source
    assert "options.fastMathEnabled = NO;" in source
    assert "options.fastMathEnabled = YES;" not in source
    assert "supportsFamily:MTLGPUFamilyApple1" in source
    assert "device_.hasUnifiedMemory" in source
    assert "kRequiredThreadExecutionWidth = 32" in source
    assert "MTLResourceStorageModeShared" in source
    assert "class PackedInputWriter" in source
    assert "std::memset(workspace.inputs.contents()" not in source
    assert "std::memset(workspace.outputs.contents()" in source
    assert "std::memset(workspace.int_workspace.contents()" in source
    assert "clear_multiplier_outputs" not in source
    assert "initialize_dual_scan" not in source
    assert "std::getenv" not in source
    assert "CLQR_METAL_FORCE_GLOBAL_PRIMAL_LEAVES" not in source
    assert source.count("[runtime.queue() commandBuffer]") == 1
    assert source.count("computeCommandEncoder]") == 1
    assert source.count("[metal_command_buffer commit]") == 1
    assert source.count("[metal_command_buffer waitUntilCompleted]") == 1
    assert source.count("setBuffer:workspace.inputs.get()") == 1
    assert source.count("setBuffer:workspace.outputs.get()") == 1
    assert source.count("setBuffer:workspace.float_workspace.get()") == 1
    assert "memoryBarrierWithScope:MTLBarrierScopeBuffers" in source
    assert source.count("EncodeCooperativeReductionTree(") == 4
    assert source.count("EncodeCooperativeTreeContexts(") == 4
    assert source.count("EncodeCooperativeKernelWithThreadgroupMemory(") >= 3

    planner = _source("metal_planner.h")
    assert "CheckedSizeSum" in planner
    assert "CheckedSizeProduct" in planner
    assert "CheckedSizeAlignUp" in planner
    assert "CheckedU32" in planner
    assert "PlanLaneSlicedThreadgroupLanes" in planner
    assert "thread_execution_width / 2" in planner
    assert "reduced_stage_float_bytes" in planner
    assert "feedback_float_bytes" in planner
    for scratch in (
        "primal_leaf_float_bytes",
        "primal_leaf_integer_bytes",
        "value_composition_float_bytes",
        "dual_parameter_float_bytes",
        "dual_parameter_integer_bytes",
        "dual_leaf_float_bytes",
        "dual_leaf_integer_bytes",
        "dual_relation_float_bytes",
        "dual_relation_integer_bytes",
        "dual_solve_float_bytes",
        "dual_solve_integer_bytes",
    ):
        assert scratch in planner

    primal = _source("metal_primal_scan_source.h")
    assert "clqr_build_primal_leaves_threadgroup_sliced" in primal
    primal_leaf_slice = primal.split(
        "kernel void clqr_build_primal_leaves_threadgroup_sliced", 1
    )[1].split("kernel void clqr_reduce_primal_relations", 1)[0]
    assert "threadgroup_barrier" not in primal_leaf_slice
    assert "group * lane_count + lane" in primal_leaf_slice
    assert "p.float_scratch" not in primal_leaf_slice
    assert "p.int_scratch" not in primal_leaf_slice
    assert (
        "p.mixed_capacity + p.state_constraint_capacity + p.state_capacity"
        in primal_leaf_slice
    )
    assert "p.control_capacity + 2u * p.state_capacity + 1u" in primal_leaf_slice
    assert "product({sizeof(float), primal_leaf_scratch})" in planner
    assert "sum({nu, two_nx})" in planner

    dual = _source("metal_dual_objective_source.h")
    assert "clqr_clear_multiplier_outputs" not in dual
    assert "clqr_initialize_dual_scan" not in dual
    assert "clqr_build_dual_parameters_threadgroup_sliced" in dual
    assert "clqr_build_dual_relation_leaves_threadgroup_sliced" in dual
    parameter_slice = dual.split(
        "kernel void clqr_build_dual_parameters_threadgroup_sliced", 1
    )[1].split("kernel void clqr_build_dual_relation_leaves", 1)[0]
    leaf_slice = dual.split(
        "kernel void clqr_build_dual_relation_leaves_threadgroup_sliced", 1
    )[1].split("kernel void clqr_reduce_dual_relations", 1)[0]
    assert "threadgroup_barrier" not in parameter_slice
    assert "threadgroup_barrier" not in leaf_slice
    assert "group * lane_count + lane" in parameter_slice
    assert "group * lane_count + lane" in leaf_slice
    assert "EncodeLaneSlicedKernelWithThreadgroupMemory" in source
    assert "PlanLaneSlicedThreadgroupLanes(" in source
    assert "runtime.build_primal_leaves_threadgroup_sliced()" in source
    assert "layout.primal_leaf_float_bytes" in source
    assert "layout.primal_leaf_integer_bytes" in source
    assert "CLQR_METAL_THREADGROUP_SLICED_PRIMAL_LEAVES" not in source
    assert "CLQR_METAL_THREADGROUP_SLICED_DUAL_PARAMETERS" not in source
    assert "CLQR_METAL_THREADGROUP_SLICED_DUAL_LEAVES" not in source

    value = _source("metal_value_affine_source.h")
    solve_slice = value.split(
        "inline bool solve_general_multiple_rhs_cooperative_threadgroup", 1
    )[1].split("inline void copy_value_cooperative", 1)[0]
    composition_slice = value.split(
        "inline bool compose_value_cooperative_threadgroup", 1
    )[1].split("kernel void clqr_build_value_leaves", 1)[0]
    assert "threadgroup float *augmented = local;" in composition_slice
    assert (
        "solve_general_multiple_rhs_cooperative_threadgroup" in composition_slice
    )
    assert "scratch(w, p" not in composition_slice
    assert "mem_flags::mem_device" not in solve_slice
    assert "mem_flags::mem_threadgroup" in solve_slice
    for kernel in (
        "clqr_reduce_value_threadgroup",
        "clqr_expand_value_context_threadgroup",
        "clqr_finalize_value_suffix_threadgroup",
    ):
        assert f"kernel void {kernel}" in value
        assert f'"{kernel}"' in source
    assert "HasCooperativeThreadgroupOccupancy" in source
    assert "kMinimumResidentThreadgroups = 4" in source
    assert "runtime.reduce_value_threadgroup()" in source
    assert "runtime.expand_value_context_threadgroup()" in source
    assert "runtime.finalize_value_suffix_threadgroup()" in source
    assert "layout.value_composition_float_bytes" in source
    assert "CLQR_METAL_FORCE_GLOBAL_VALUE_COMPOSITION" not in source
    assert "sum({product({nx, product({3, nx})}), nx})" in planner
    assert (
        "product({sizeof(float), value_composition_scratch})" in planner
    )


def test_metal_dense_products_remain_staged_and_cubic():
    value = _source("metal_value_affine_source.h")
    reduction = _source("metal_reduction_recovery_source.h")
    planner = _source("metal_planner.h")
    host = _source("jax_ffi_metal.cc")

    # Conditional-value composition stages each quadratic correction as two
    # ordinary products and reuses the dead augmented solve storage.
    assert (
        "Form the two quadratic updates as pairs of ordinary matrix products." in value
    )
    assert "output_C[row * shared + b]" in value
    assert "output_J[index] = value;" in value

    # Stage reduction forms A*T once, then applies H*L followed by L'*U for
    # the reduced Hessian instead of directly contracting two physical axes.
    assert "Keep A*T and c+A*t past RREF" in reduction
    assert "threadgroup float *suffix_AT" in reduction
    assert "Apply the dense block transform" in reduction
    assert "as U=H*L followed by L'*U" in reduction
    assert "threadgroup float *block_z" in reduction
    assert "threadgroup float *block_v" in reduction

    # Terminal reduction and feedback likewise stage Q*T and J*[B,A].
    assert "threadgroup float *QT = local_float;" in reduction
    assert "QT[x * reduced + b]" in reduction
    assert "Stage J*[B,A] once." in reduction
    assert "threadgroup float *products = local_float;" in reduction
    assert "products[a * columns + col]" in reduction

    # The planner and launch-side dynamic threadgroup allocations must cover
    # those exact intermediates.
    assert "stage_reduction_block_scratch" in planner
    assert "product({nx_plus_nu, nx_plus_nu})" in planner
    assert "terminal_reduction_scratch" in planner
    assert "sum({product({nx, nx}), nx})" in planner
    assert "feedback_scratch" in planner
    assert "product({nx_plus_nu, nx_plus_nu})" in planner
    assert "reduced_terminal_float_bytes" in host
    assert "feedback_float_bytes" in host

    # Diagnostic direct-product branches were useful for A/B, but must not
    # remain as production alternatives.
    for source in (value, reduction):
        assert "CLQR_METAL_DIRECT_VALUE_PRODUCTS" not in source
        assert "p.padding0" not in source
        assert "direct_product" not in source


if __name__ == "__main__":
    test_metal_ffi_uses_one_precise_native_gpu_submission()
    test_metal_dense_products_remain_staged_and_cubic()
