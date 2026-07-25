import os
import pathlib


def _cuda_source():
    root = pathlib.Path(os.environ["TEST_SRCDIR"])
    matches = list(root.rglob("src/cuda_solver.cu"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one cuda_solver.cu in runfiles; found {matches}")
    return matches[0].read_text()


def _between(source, first, second):
    return source.split(first, 1)[1].split(second, 1)[0]


def _between_last(source, first, second):
    return source.rsplit(first, 1)[1].split(second, 1)[0]


def test_conditional_value_composition_is_staged_cubic():
    source = _cuda_source()
    body = _between(
        source, "ComposeValueElementsBlock(", "__device__ bool InvalidScanValueElement"
    )
    assert "for (int p =" not in body
    assert "for (int q =" not in body
    assert "product[row * right + col] = value;" in body
    assert "product[row * left + col] = value;" in body
    assert "second.A[row * second.left_dim + k] *" in body
    assert "product[k * right + col]" in body
    assert "first.A[k * first.left_dim + row] *" in body
    assert "product[k * left + col]" in body
    assert "ValueProductScratchEntries" in source
    assert "ScratchCheckedProduct(first.right," in source


def test_feedback_hessian_products_are_staged_cubic():
    source = _cuda_source()
    body = _between(
        source, "BuildMatrixFeedbackSystem(", "__device__ void ExtractMatrixFeedback"
    )
    assert "for (int a =" not in body
    assert "for (int b =" not in body
    assert "product[row * columns + col] = value;" in body
    assert "next.J[row * next.left_dim + k] * operand" in body
    assert "s.B[k * s.m + row] * product[k * columns + col]" in body
    assert 'next, ScratchCheckedSum({m, n}, "feedback workspace")' in source


def test_stage_reduction_products_are_staged_cubic():
    source = _cuda_source()
    body = _between_last(
        source, "ReduceStagesKernel(", "__global__ void ReduceTerminalKernel"
    )
    assert "Scalar at = Scalar{0};" not in body
    assert "dynamics_state[xp * current.reduced_dim + z]" in body
    assert "dynamics_offset[xp]" in body
    assert "Scalar *hessian_times_map" in body
    assert "StageHessianEntry(s, row, inner)" in body
    assert "StageReductionMapEntry(s, current, cp, inner, col)" in body
    assert "hessian_times_map[inner * reduced_variables + col]" in body
    for old_direct_product in (
        "current.T[x * current.reduced_dim + a] * s.Q",
        "cp.Y[u * cp.state_dim + a] * s.R",
        "cp.Z[u * cp.reduced_dim + a] * s.R",
        "current.T[x * current.reduced_dim + z] * s.M",
    ):
        assert old_direct_product not in body
    assert "StageRelationReductionScratchBytes" in source
    assert "StageHessianTransformScratchBytes" in source
    assert "SharedScalarEntries(transform_capacity_entries)" in body
    assert "SharedScalarEntries(transform_entries)" in body
    assert "Take<Scalar>(shared_transform_entries)" in body


def test_terminal_reduction_product_is_staged_cubic():
    source = _cuda_source()
    body = _between_last(
        source, "ReduceTerminalKernel(", "__global__ void InitialReducedStateKernel"
    )
    assert "Scalar *hessian_times_map" in body
    assert "hessian_times_map[inner * param.reduced_dim + col]" in body
    assert (
        "param.T[x * param.reduced_dim + a] *\n"
        "                 terminal.Q[x * terminal.n + y]"
    ) not in body
    assert "scratch.terminal_reduction" in source
    assert "SharedScalarEntries(transform_entries)" in body
    assert "scratch_size.Add<Scalar>(shared_transform_entries)" in body
    assert "scratch.Take<Scalar>(shared_transform_entries)" in body
    assert "linear < shared_transform_entries" in body


def test_hessian_transform_planner_pads_shared_access_footprint():
    source = _cuda_source()
    body = _between(
        source,
        "StageHessianTransformScratchBytes(",
        "struct ScanShape",
    )
    assert "ScratchCheckedProduct(physical_variables, reduced_variables" in body
    assert "ScratchCheckedSharedScalarEntries(entries, description)" in body


if __name__ == "__main__":
    test_conditional_value_composition_is_staged_cubic()
    test_feedback_hessian_products_are_staged_cubic()
    test_stage_reduction_products_are_staged_cubic()
    test_terminal_reduction_product_is_staged_cubic()
    test_hessian_transform_planner_pads_shared_access_footprint()
