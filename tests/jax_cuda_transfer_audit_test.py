import os
import pathlib


def _source(name):
    root = pathlib.Path(os.environ["TEST_SRCDIR"])
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} in runfiles; found {matches}")
    return matches[0].read_text()


def test_cuda_ffi_has_no_bulk_scalar_host_staging():
    source = _source("jax_ffi_cuda.cu")
    assert "cudaMemcpyHostToDevice" not in source
    assert source.count("cudaMemcpyDeviceToHost") == 1
    copy = source.index("cudaMemcpyDeviceToHost")
    assert "dimensions.typed_data()" in source[max(0, copy - 300) : copy]
    for removed_staging_path in (
        "input_scalars",
        "output_scalars",
        "WriteCudaSolution",
        "SolvePreparedView",
    ):
        assert removed_staging_path not in source
    assert "staging.WaitForPreviousOutput();" in source
    assert "staging.RecordOutput(stream);" in source


if __name__ == "__main__":
    test_cuda_ffi_has_no_bulk_scalar_host_staging()
