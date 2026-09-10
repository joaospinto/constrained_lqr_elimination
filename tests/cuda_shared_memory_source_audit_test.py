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
        configurations = re.findall(r"X\((\w+),\s*(\w+),\s*\w+\)", source)
        self.assertEqual(len(configurations), len(set(configurations)))
        launches = set(re.findall(r"CLQR_LAUNCH_SCRATCH\((\w+),", source))
        launches.discard("kernel")
        self.assertTrue(launches)
        self.assertEqual({kernel for kernel, _ in configurations}, launches)
        definitions = re.findall(
            r"template <bool GlobalScratch = kDefaultGlobalScratch>\s*"
            r"__global__ void (\w+)\([^{};]*CLQR_SCRATCH_PARAMS\)\s*{", source
        )
        self.assertEqual(set(definitions), launches)
        self.assertNotRegex(source, r"<<<[^>]*scratch\.\w+")
        self.assertIn("kernel<true><<<blocks, kThreads, 0, stream>>>", source)
        self.assertIn("workspace.global_scratch.get()", source)
        self.assertIn("plans->kernel.GlobalBytes(blocks)", source)
        self.assertNotIn("kStaticSharedMemoryAllowance", source)


if __name__ == "__main__":
    unittest.main()
