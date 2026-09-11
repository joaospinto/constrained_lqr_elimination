import os
import pathlib
import re
import unittest


class SharedMemoryLaunchAudit(unittest.TestCase):
    def test_every_dynamic_launch_has_matching_configuration(self):
        source = (
            pathlib.Path(os.environ["TEST_SRCDIR"])
            / os.environ["TEST_WORKSPACE"]
            / "src/cuda_solver.cu"
        ).read_text()
        source = source.replace("\\\n", "")
        configurations = re.findall(r"X\((\w+),\s*(\w+),\s*\w+\)", source)
        self.assertEqual(len(configurations), len(set(configurations)))
        launches = set(re.findall(r"CLQR_LAUNCH_SCRATCH\(\s*(\w+),", source))
        launches.discard("kernel")
        self.assertTrue(launches)
        self.assertEqual({kernel for kernel, _ in configurations}, launches)
        definitions = re.findall(
            r"template <bool GlobalScratch = false>\s*"
            r"__global__ void (\w+)\([^{};]*CLQR_SCRATCH_PARAMS\)\s*{", source
        )
        self.assertEqual(set(definitions), launches)
        self.assertNotRegex(source, r"<<<[^>]*scratch\.\w+")
        self.assertIn(
            "kernel<true><<<count, workspace.threads, 0, stream>>>", source
        )
        self.assertTrue(
            re.search(r"ForEachGlobalScratchLaunch\(\s*clqr_launch, blocks,", source),
            "global scratch launches must use the bounded launch iterator",
        )
        self.assertIn("clqr_launch.global_stride, first)", source)
        self.assertIn("workspace.global_scratch.get()", source)
        self.assertIn("plans->kernel.GlobalBytes(blocks)", source)
        self.assertNotIn("kStaticSharedMemoryAllowance", source)

        # Scratch offsets use the physical block; logical stage/tree indices
        # include the launch offset. Catch an unconverted scratch kernel.
        for kernel in definitions:
            match = re.search(
                rf"__global__ void {kernel}\([^{{}};]*CLQR_SCRATCH_PARAMS\)\s*{{",
                source,
            )
            start = match.end()
            depth = 1
            end = start
            while depth:
                depth += (source[end] == "{") - (source[end] == "}")
                end += 1
            body = source[start:end]
            self.assertNotIn(
                "blockIdx.x", body.replace("(first_block + blockIdx.x)", ""), kernel
            )


if __name__ == "__main__":
    unittest.main()
