# PMPP labs verification report

- Generated: 2026-11-20T02:41:00
- System: macOS-27.0.1-arm64-arm-64bit-Mach-O (arm64), Python 3.13.4, cmake version 4.0.2
- Backends with a usable device: **emu, metal**
- Reference solutions: **10 of 10 passed**

## What this report does and does not establish

- A reference solution that passes on `metal`, `cuda` or `hip` was compiled and run on that
  real device and matched a CPU reference. Nothing else counts as verified on that backend.
- `emu` is a CPU emulator of the CUDA subset. It checks kernel logic (indexing, shared memory,
  barriers, bounds) with deterministic thread schedules and AddressSanitizer. It says nothing
  about performance, and passing it does not prove a kernel runs on a GPU.
- The CUDA front-end check parses `.cu` files with clang against hand-written stub headers.
  It is a syntax and host/device-rule check only.
- Benchmark numbers are from this machine at this moment. They are not expectations.

## Backends

| Backend | Built | Device | Reason / device name |
|---|---|---|---|
| emu | yes | yes | pmpp CPU emulator (not a GPU) |
| metal | yes | yes | Apple M3 Max |
| cuda | skipped | - | no CUDA compiler (nvcc) found |
| hip | skipped | - | no HIP compiler found (is ROCm installed? set ROCM_PATH) |

### Device properties reported by `emu`

| Property | Value |
|---|---|
| name | pmpp CPU emulator (not a GPU) |
| totalGlobalMem | 0 |
| multiProcessorCount | 1 |
| warpSize | 1 |
| maxThreadsPerBlock | 1024 |
| maxThreadsDim.x | 1024 |
| maxThreadsDim.y | 1024 |
| maxThreadsDim.z | 1024 |
| maxGridSize.x | 65535 |
| maxGridSize.y | 65535 |
| maxGridSize.z | 65535 |
| sharedMemPerBlock | 49152 |
| regsPerBlock | 0 |
| computeCapability.major | 0 |
| computeCapability.minor | 0 |
| sharedMemPerMultiprocessor | 49152 |
| regsPerMultiprocessor | 0 |
| maxThreadsPerMultiProcessor | 1024 |
| maxBlocksPerMultiProcessor | 1 |
| note | synthetic values describing the CPU emulator, not a GPU |

### Device properties reported by `metal`

| Property | Value |
|---|---|
| name | Apple M3 Max |
| hasUnifiedMemory | 1 |
| maxThreadsPerThreadgroup.width | 1024 |
| maxThreadsPerThreadgroup.height | 1024 |
| maxThreadsPerThreadgroup.depth | 1024 |
| maxThreadgroupMemoryLength | 32768 |
| maxBufferLength | 22613000192 |
| recommendedMaxWorkingSetSize | 30150672384 |
| physicalMemory | 38654705664 |

## Reference solutions

| Lab | Backend | Tile | Build | Tests passed | Verdict |
|---|---|---|---|---|---|
| ch05/lab1_roofline | emu | 16 | ok | 21/21 | pass |
| ch05/lab1_roofline | metal | 16 | ok | 9/9 | pass |
| ch05/lab2_tiled | emu | 16 | ok | 9/9 | pass |
| ch05/lab2_tiled | emu | 8 | ok | 9/9 | pass |
| ch05/lab2_tiled | emu | 32 | ok | 9/9 | pass |
| ch05/lab2_tiled | metal | 16 | ok | 10/10 | pass |
| ch05/lab3_gemm | emu | 16 | ok | 24/24 | pass |
| ch05/lab3_gemm | metal | 16 | ok | 10/10 | pass |
| ch05/lab4_racehunt | emu | 16 | ok | 18/18 | pass |
| ch05/lab4_racehunt | metal | 16 | ok | 8/8 | pass |

## Starters (negative control) and your work

An untouched starter must build and must fail its tests; if one passes, the tests are too weak.

| Lab | Backend | State of work/ | Build | Tests passed | Reading |
|---|---|---|---|---|---|
| ch05/lab1_roofline | emu | untouched | ok | 6/21 | starter fails as it should |
| ch05/lab1_roofline | metal | untouched | ok | 2/9 | starter fails as it should |
| ch05/lab2_tiled | emu | untouched | ok | 0/9 | starter fails as it should |
| ch05/lab2_tiled | metal | untouched | ok | 0/10 | starter fails as it should |
| ch05/lab3_gemm | emu | untouched | ok | 0/24 | starter fails as it should |
| ch05/lab3_gemm | metal | untouched | ok | 0/10 | starter fails as it should |
| ch05/lab4_racehunt | emu | untouched | ok | 3/6 (exit -6: crashed or aborted) | starter fails as it should |
| ch05/lab4_racehunt | metal | untouched | ok | 3/8 | starter fails as it should |

## Benchmarks (this machine, this run)

### ch05/lab1_roofline, solution, metal, TILE_WIDTH=16

| name | bytes | seconds | gb_per_s | flop | gflops | width | bandwidth_gb_per_s | compute_gflops | threshold_flop_per_byte | naive_model_ceiling_gflops | naive_gflops | naive_width |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bandwidth | 1.5e+09 | 0.005626 | 266.6 |  |  |  |  |  |  |  |  |  |
| compute |  | 0.08398 |  | 1.374e+11 | 1637 |  |  |  |  |  |  |  |
| naive_matmul |  | 0.1999 |  |  | 687.5 | 4096 |  |  |  |  |  |  |
| roofline |  |  |  |  |  |  | 266.6 | 1637 | 6.138 | 66.66 | 687.5 | 4096 |

### ch05/lab2_tiled, solution, metal, TILE_WIDTH=16

| name | width | seconds | gflops | tile | speedup_vs_naive |
|---|---|---|---|---|---|
| naive | 4096 | 0.1996 | 688.4 |  |  |
| tiled | 4096 | 0.362 | 379.6 | 4 | 0.5515 |
| tiled | 4096 | 0.1023 | 1344 | 8 | 1.952 |
| tiled | 4096 | 0.1157 | 1188 | 16 | 1.726 |
| tiled | 4096 | 0.1031 | 1333 | 32 | 1.936 |

### ch05/lab3_gemm, solution, metal, TILE_WIDTH=16

| name | batch | k | l | tile | seconds | gflops | rows_per_s |
|---|---|---|---|---|---|---|---|
| linear_layer | 1 | 1024 | 4096 | 16 | 0.000184 | 45.59 | 5435 |
| linear_layer | 4 | 1024 | 4096 | 16 | 0.0002087 | 160.8 | 1.917e+04 |
| linear_layer | 16 | 1024 | 4096 | 16 | 0.0001728 | 776.8 | 9.26e+04 |
| linear_layer | 64 | 1024 | 4096 | 16 | 0.000535 | 1004 | 1.196e+05 |
| linear_layer | 256 | 1024 | 4096 | 16 | 0.001527 | 1407 | 1.677e+05 |
| linear_layer | 1024 | 1024 | 4096 | 16 | 0.006086 | 1411 | 1.683e+05 |

## CUDA front-end check (clang, stub headers, nothing run)

Compiler: Apple clang version 21.0.0 (clang-2100.3.34.2)

| File | Parses as CUDA |
|---|---|
| labs/ch05/lab1_roofline/solution/kernel.cu | yes |
| labs/ch05/lab1_roofline/starter/kernel.cu | yes |
| labs/ch05/lab1_roofline/work/kernel.cu | yes |
| labs/ch05/lab2_tiled/solution/kernel.cu | yes |
| labs/ch05/lab2_tiled/starter/kernel.cu | yes |
| labs/ch05/lab2_tiled/work/kernel.cu | yes |
| labs/ch05/lab3_gemm/solution/kernel.cu | yes |
| labs/ch05/lab3_gemm/starter/kernel.cu | yes |
| labs/ch05/lab3_gemm/work/kernel.cu | yes |
| labs/ch05/lab4_racehunt/solution/kernel.cu | yes |
| labs/ch05/lab4_racehunt/starter/kernel.cu | yes |
| labs/ch05/lab4_racehunt/work/kernel.cu | yes |
| labs/ch05/shared/naive_matmul.cu | yes |

