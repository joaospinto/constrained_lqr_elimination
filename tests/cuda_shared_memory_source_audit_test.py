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
        configurations = re.findall(
            r"CLQR_CONFIGURE_SCRATCH\((\w+),\s*(\w+)\);", source
        )
        self.assertEqual(len(configurations), len(set(configurations)))
        launches = set()
        for kernel, arguments in re.findall(r"(\w+)\s*<<<(.*?)>>>", source, re.S):
            dynamic_bytes = arguments.split(",")[2].strip()
            if dynamic_bytes == "0":
                continue
            self.assertRegex(dynamic_bytes, r"^scratch\.\w+$")
            launches.add((kernel, dynamic_bytes.removeprefix("scratch.")))
        self.assertTrue(launches)
        self.assertEqual(set(configurations), launches)
        self.assertEqual(
            source.count("ConfigureScratchMemory(workspace->scratch, device);"), 1
        )
        self.assertNotIn("kStaticSharedMemoryAllowance", source)


if __name__ == "__main__":
    unittest.main()
