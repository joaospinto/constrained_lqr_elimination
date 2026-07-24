# CUDA runtime-dimension evidence

This historical comparison isolates the removal of the CUDA build-time
dimension capacities. It is not a benchmark of the later, fully optimized
`main` stack:

- baseline: `3d225eae7e8f7c24c42ddd1cfbf921ef4d540764`
- candidate: `511cd314445a421c35dcb77cd6154a70e99267e2`
- baseline flags:
  `--cuda_max_state_dimension=8 --cuda_max_control_dimension=4`
  `--cuda_max_mixed_constraints=2 --cuda_max_state_constraints=2`
- candidate flags: none

The test ran in FP64 on a Kaggle Tesla P100-PCIE-16GB with driver 580.159.04
(driver-reported CUDA 13.0). Each revision used eleven warmed solves per
horizon and three alternating-order rounds; the report contains the median
from those rounds. Ratios are baseline divided by candidate, so values above
one favor the runtime-sized candidate.

| N | Base CPU (ms) | Candidate CPU (ms) | Base wall (ms) | Candidate wall (ms) | Wall ratio | Base kernel (ms) | Candidate kernel (ms) | Kernel ratio | Base CUDA KKT | Candidate CUDA KKT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.312373 | 0.350792 | 2.072685 | 2.091153 | 0.9912 | 1.843840 | 1.863456 | 0.9895 | 3.148419e-13 | 3.148419e-13 |
| 64 | 0.613575 | 0.639666 | 2.327848 | 2.338272 | 0.9955 | 2.070752 | 2.091584 | 0.9900 | 2.873257e-13 | 2.873257e-13 |
| 128 | 1.203865 | 1.242835 | 2.629761 | 2.635322 | 0.9979 | 2.301152 | 2.328544 | 0.9882 | 5.594136e-13 | 5.594136e-13 |
| 256 | 2.497240 | 2.594146 | 3.082544 | 3.006837 | 1.0252 | 2.635328 | 2.587296 | 1.0186 | 7.336354e-13 | 7.336354e-13 |
| 512 | 5.224646 | 5.389416 | 3.924261 | 3.744177 | 1.0481 | 3.206592 | 3.102880 | 1.0334 | 6.682432e-13 | 6.682432e-13 |
| 1024 | 11.755314 | 11.879444 | 5.262845 | 4.898135 | 1.0745 | 3.997376 | 3.704992 | 1.0789 | 8.175682e-13 | 8.175682e-13 |
| 2048 | 25.939994 | 27.550174 | 8.080390 | 7.354615 | 1.0987 | 5.695488 | 5.071872 | 1.1230 | 8.175682e-13 | 8.175682e-13 |
| 4096 | 55.057698 | 56.696532 | 13.454825 | 12.051717 | 1.1164 | 8.404288 | 7.282624 | 1.1540 | 1.123551e-12 | 1.123551e-12 |
| 8192 | 109.603611 | 115.503529 | 24.667694 | 21.610901 | 1.1414 | 13.996576 | 11.653120 | 1.2011 | 1.123551e-12 | 1.123551e-12 |
| 16384 | 220.356398 | 228.253599 | 46.499297 | 40.056983 | 1.1608 | 25.528641 | 20.678944 | 1.2345 | 1.425970e-12 | 1.425970e-12 |

At the three smallest horizons, wall time changes by less than 0.9% and kernel
time by less than 1.2%. From N=256 onward, the runtime-sized implementation is
faster in both wall and kernel time. At N=16384 the baseline/candidate speedup
is 1.1608x wall and 1.2345x kernel. CUDA KKT residuals are identical at the
reported precision for every horizon and remain at or below 1.425970e-12.

The runtime-dimension change does not modify the sequential CPU solver. CPU
timings are retained for completeness, but their variation is not attributed
to this CUDA-only change. The historical benchmark schema did not emit CPU KKT
or input-packing fields, so those values are `nan` in the
[raw report](cuda_runtime_dimension_p100.csv).
