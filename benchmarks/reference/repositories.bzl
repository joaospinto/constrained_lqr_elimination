"""Pinned, optional reference sources; never dependencies of the public library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _reference_sources_impl(_ctx):
    http_archive(
        name = "vanroye_source",
        urls = ["https://codeload.github.com/lvanroye/generalization_riccati/tar.gz/8bf4b5684219a6035e93b61d8ef6ae4afd1f4e19"],
        sha256 = "035c3bf9ce31b799ad97d2c34d1b597886460ca34b53ceb51ac05b8c7f41c6b0",
        type = "tar.gz",
        strip_prefix = "generalization_riccati-8bf4b5684219a6035e93b61d8ef6ae4afd1f4e19/src/gen_riccati",
        patches = ["//benchmarks/reference:vanroye_memory.patch"],
        patch_args = ["-p1"],
        build_file_content = """
filegroup(
    name = "sources",
    srcs = glob(["*.cpp", "*.hpp"]),
    visibility = ["//visibility:public"],
)
exports_files(["gen_riccati.cpp"])
""",
    )

reference_sources = module_extension(implementation = _reference_sources_impl)
