# PMPP labs verification report

- Generated: 2026-11-21T12:07:03
- System: macOS-27.0.1-arm64-arm-64bit-Mach-O (arm64), Python 3.13.4, cmake version 4.0.2
- Backends with a usable device: **emu, race, metal**
- Reference solutions: **226 of 226 passed**

## What this report does and does not establish

- A reference solution that passes on `metal`, `cuda` or `hip` was compiled and run on that
  real device and matched a CPU reference. Nothing else counts as verified on that backend.
- `emu` is a CPU emulator of the CUDA subset. It checks kernel logic (indexing, shared memory,
  barriers, bounds) with deterministic thread schedules and AddressSanitizer. It says nothing
  about performance, and passing it does not prove a kernel runs on a GPU.
- `race` is the same emulator with its threads running concurrently under ThreadSanitizer. A data
  race it reports is real for the schedule that ran; a clean run does not prove there is none. It
  has no bounds checking, and it too says nothing about performance. Labs opt in to it.
- The CUDA front-end check parses `.cu` files with clang against hand-written stub headers.
  It is a syntax and host/device-rule check only.
- Benchmark numbers are from this machine at this moment. They are not expectations.

## Backends

| Backend | Built | Device | Reason / device name |
|---|---|---|---|
| emu | yes | yes | pmpp CPU emulator (not a GPU) |
| race | yes | yes | pmpp CPU emulator, race-detection flavour (not a GPU) |
| metal | yes | yes | Apple M3 Max |
| cuda | skipped | - | no CUDA compiler (nvcc) found |
| hip | skipped | - | no HIP compiler found (is ROCm installed? set ROCM_PATH) |

### Device properties reported by `emu`

| Property | Value |
|---|---|
| name | pmpp CPU emulator (not a GPU) |
| totalGlobalMem | 0 |
| multiProcessorCount | 1 |
| warpSize | 32 |
| maxThreadsPerBlock | 1024 |
| maxThreadsDim.x | 1024 |
| maxThreadsDim.y | 1024 |
| maxThreadsDim.z | 64 |
| maxGridSize.x | 2147483647 |
| maxGridSize.y | 65535 |
| maxGridSize.z | 65535 |
| sharedMemPerBlock | 49152 |
| regsPerBlock | 0 |
| totalConstMem | 0 |
| l2CacheSize | 0 |
| computeCapability.major | 0 |
| computeCapability.minor | 0 |
| sharedMemPerMultiprocessor | 98304 |
| regsPerMultiprocessor | 0 |
| maxThreadsPerMultiProcessor | 1024 |
| maxBlocksPerMultiProcessor | 1 |
| asyncEngineCount | 0 |
| concurrentKernels | 0 |
| note | synthetic values describing the CPU emulator, not a GPU |

### Device properties reported by `race`

| Property | Value |
|---|---|
| name | pmpp CPU emulator, race-detection flavour (not a GPU) |
| warpSize | 32 |
| maxThreadsPerBlock | 1024 |
| maxThreadsDim.x | 1024 |
| maxThreadsDim.y | 1024 |
| maxThreadsDim.z | 64 |
| maxGridSize.x | 2147483647 |
| maxGridSize.y | 65535 |
| maxGridSize.z | 65535 |
| sharedMemPerBlock | 49152 |
| multiProcessorCount | 1 |
| note | synthetic values describing the CPU emulator (race-detection flavour), not a GPU |

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

## Builds not made

| Lab | Variant | Backend | Why |
|---|---|---|---|
| ch11/lab5_singlepass | work | metal | no Metal build for this lab: work/ has no kernel.metal and launch.mm (the lab is CUDA-family only; its README says why) |
| ch11/lab5_singlepass | solution | metal | no Metal build for this lab: solution/ has no kernel.metal and launch.mm (the lab is CUDA-family only; its README says why) |
| ch12/lab5_inplace | work | metal | no Metal build for this lab: work/ has no kernel.metal and launch.mm (the lab is CUDA-family only; its README says why) |
| ch12/lab5_inplace | solution | metal | no Metal build for this lab: solution/ has no kernel.metal and launch.mm (the lab is CUDA-family only; its README says why) |

## Reference solutions

| Lab | Backend | Tile | Build | Tests passed | Verdict |
|---|---|---|---|---|---|
| appA/lab1_sumerror | emu | 16 | ok | 51/51 | pass |
| appA/lab1_sumerror | race | 16 | ok | 17/17 | pass |
| appA/lab1_sumerror | metal | 16 | ok | 39/39 | pass |
| appC/lab1_feeding | emu | 16 | ok | 144/144 | pass |
| appC/lab1_feeding | metal | 16 | ok | 60/60 | pass |
| ch02/lab1_vecadd | emu | 16 | ok | 60/60 | pass |
| ch02/lab1_vecadd | metal | 16 | ok | 10/10 | pass |
| ch02/lab2_timing | emu | 16 | ok | 39/39 | pass |
| ch02/lab2_timing | metal | 16 | ok | 10/10 | pass |
| ch02/lab3_elementwise | emu | 16 | ok | 78/78 | pass |
| ch02/lab3_elementwise | metal | 16 | ok | 24/24 | pass |
| ch02/lab4_bughunt | emu | 16 | ok | 21/21 | pass |
| ch02/lab4_bughunt | metal | 16 | ok | 7/7 | pass |
| ch03/lab1_grayscale | emu | 16 | ok | 33/33 | pass |
| ch03/lab1_grayscale | metal | 16 | ok | 12/12 | pass |
| ch03/lab2_blur | emu | 16 | ok | 30/30 | pass |
| ch03/lab2_blur | metal | 16 | ok | 11/11 | pass |
| ch03/lab3_naive_gemm | emu | 16 | ok | 30/30 | pass |
| ch03/lab3_naive_gemm | race | 16 | ok | 10/10 | pass |
| ch03/lab3_naive_gemm | metal | 16 | ok | 12/12 | pass |
| ch03/lab4_granularity | emu | 16 | ok | 45/45 | pass |
| ch03/lab4_granularity | race | 16 | ok | 15/15 | pass |
| ch03/lab4_granularity | metal | 16 | ok | 21/21 | pass |
| ch03/lab5_maphunt | emu | 16 | ok | 27/27 | pass |
| ch03/lab5_maphunt | race | 16 | ok | 10/10 | pass |
| ch03/lab5_maphunt | metal | 16 | ok | 10/10 | pass |
| ch04/lab1_launchconfig | emu | 16 | ok | 42/42 | pass |
| ch04/lab1_launchconfig | metal | 16 | ok | 13/13 | pass |
| ch04/lab2_divergence | emu | 16 | ok | 60/60 | pass |
| ch04/lab2_divergence | metal | 16 | ok | 20/20 | pass |
| ch04/lab3_barrierhunt | emu | 16 | ok | 30/30 | pass |
| ch04/lab3_barrierhunt | race | 16 | ok | 10/10 | pass |
| ch04/lab3_barrierhunt | metal | 16 | ok | 10/10 | pass |
| ch04/lab4_blocksize | emu | 16 | ok | 45/45 | pass |
| ch04/lab4_blocksize | metal | 16 | ok | 7/7 | pass |
| ch05/lab1_roofline | emu | 16 | ok | 21/21 | pass |
| ch05/lab1_roofline | metal | 16 | ok | 9/9 | pass |
| ch05/lab2_tiled | emu | 16 | ok | 9/9 | pass |
| ch05/lab2_tiled | emu | 8 | ok | 9/9 | pass |
| ch05/lab2_tiled | emu | 32 | ok | 9/9 | pass |
| ch05/lab2_tiled | race | 16 | ok | 5/5 | pass |
| ch05/lab2_tiled | metal | 16 | ok | 10/10 | pass |
| ch05/lab3_gemm | emu | 16 | ok | 24/24 | pass |
| ch05/lab3_gemm | race | 16 | ok | 8/8 | pass |
| ch05/lab3_gemm | metal | 16 | ok | 10/10 | pass |
| ch05/lab4_racehunt | emu | 16 | ok | 18/18 | pass |
| ch05/lab4_racehunt | race | 16 | ok | 6/6 | pass |
| ch05/lab4_racehunt | metal | 16 | ok | 8/8 | pass |
| ch05/lab5_dynamic | emu | 16 | ok | 48/48 | pass |
| ch05/lab5_dynamic | race | 16 | ok | 16/16 | pass |
| ch05/lab5_dynamic | metal | 16 | ok | 24/24 | pass |
| ch06/lab1_coalescing | emu | 16 | ok | 111/111 | pass |
| ch06/lab1_coalescing | metal | 16 | ok | 42/42 | pass |
| ch06/lab2_vector | emu | 16 | ok | 87/87 | pass |
| ch06/lab2_vector | metal | 16 | ok | 35/35 | pass |
| ch06/lab3_banks | emu | 16 | ok | 48/48 | pass |
| ch06/lab3_banks | emu | 32 | ok | 42/42 | pass |
| ch06/lab3_banks | race | 16 | ok | 16/16 | pass |
| ch06/lab3_banks | metal | 16 | ok | 38/38 | pass |
| ch06/lab4_linear | emu | 16 | ok | 48/48 | pass |
| ch06/lab4_linear | emu | 32 | ok | 48/48 | pass |
| ch06/lab4_linear | race | 16 | ok | 16/16 | pass |
| ch06/lab4_linear | metal | 16 | ok | 44/44 | pass |
| ch06/lab5_doublebuf | emu | 16 | ok | 12/12 | pass |
| ch06/lab5_doublebuf | emu | 8 | ok | 12/12 | pass |
| ch06/lab5_doublebuf | emu | 32 | ok | 9/9 | pass |
| ch06/lab5_doublebuf | race | 16 | ok | 3/3 | pass |
| ch06/lab5_doublebuf | metal | 16 | ok | 9/9 | pass |
| ch07/lab1_constconv | emu | 16 | ok | 45/45 | pass |
| ch07/lab1_constconv | metal | 16 | ok | 21/21 | pass |
| ch07/lab2_tiledconv | emu | 32 | ok | 105/105 | pass |
| ch07/lab2_tiledconv | emu | 8 | ok | 105/105 | pass |
| ch07/lab2_tiledconv | emu | 16 | ok | 105/105 | pass |
| ch07/lab2_tiledconv | race | 32 | ok | 35/35 | pass |
| ch07/lab2_tiledconv | metal | 32 | ok | 251/251 | pass |
| ch08/lab1_stencil3d | emu | 32 | ok | 126/126 | pass |
| ch08/lab1_stencil3d | emu | 16 | ok | 126/126 | pass |
| ch08/lab1_stencil3d | emu | 8 | ok | 126/126 | pass |
| ch08/lab1_stencil3d | race | 32 | ok | 28/28 | pass |
| ch08/lab1_stencil3d | metal | 32 | ok | 85/85 | pass |
| ch09/lab1_racehunt | emu | 16 | ok | 36/36 | pass |
| ch09/lab1_racehunt | race | 16 | ok | 12/12 | pass |
| ch09/lab1_racehunt | metal | 16 | ok | 16/16 | pass |
| ch09/lab2_privatised | emu | 16 | ok | 42/42 | pass |
| ch09/lab2_privatised | race | 16 | ok | 14/14 | pass |
| ch09/lab2_privatised | metal | 16 | ok | 18/18 | pass |
| ch09/lab3_tokencounts | emu | 16 | ok | 90/90 | pass |
| ch09/lab3_tokencounts | race | 16 | ok | 30/30 | pass |
| ch09/lab3_tokencounts | metal | 16 | ok | 80/80 | pass |
| ch10/lab1_sum | emu | 16 | ok | 48/48 | pass |
| ch10/lab1_sum | race | 16 | ok | 8/8 | pass |
| ch10/lab1_sum | metal | 16 | ok | 24/24 | pass |
| ch10/lab2_reducehunt | emu | 16 | ok | 21/21 | pass |
| ch10/lab2_reducehunt | race | 16 | ok | 7/7 | pass |
| ch10/lab2_reducehunt | metal | 16 | ok | 9/9 | pass |
| ch10/lab3_softmax_argmax | emu | 16 | ok | 111/111 | pass |
| ch10/lab3_softmax_argmax | race | 16 | ok | 15/15 | pass |
| ch10/lab3_softmax_argmax | metal | 16 | ok | 48/48 | pass |
| ch11/lab1_blockscan | emu | 16 | ok | 24/24 | pass |
| ch11/lab1_blockscan | race | 16 | ok | 4/4 | pass |
| ch11/lab1_blockscan | metal | 16 | ok | 20/20 | pass |
| ch11/lab2_coarsened | emu | 16 | ok | 60/60 | pass |
| ch11/lab2_coarsened | race | 16 | ok | 10/10 | pass |
| ch11/lab2_coarsened | metal | 16 | ok | 44/44 | pass |
| ch11/lab3_scanhunt | emu | 16 | ok | 48/48 | pass |
| ch11/lab3_scanhunt | race | 16 | ok | 3/3 | pass |
| ch11/lab3_scanhunt | metal | 16 | ok | 20/20 | pass |
| ch11/lab4_global | emu | 16 | ok | 18/18 | pass |
| ch11/lab4_global | race | 16 | ok | 3/3 | pass |
| ch11/lab4_global | metal | 16 | ok | 9/9 | pass |
| ch11/lab5_singlepass | emu | 16 | ok | 12/12 | pass |
| ch11/lab5_singlepass | race | 16 | ok | 2/2 | pass |
| ch11/lab6_topp | emu | 16 | ok | 108/108 | pass |
| ch11/lab6_topp | race | 16 | ok | 17/17 | pass |
| ch11/lab6_topp | metal | 16 | ok | 47/47 | pass |
| ch12/lab1_unstable | emu | 16 | ok | 84/84 | pass |
| ch12/lab1_unstable | race | 16 | ok | 25/25 | pass |
| ch12/lab1_unstable | metal | 16 | ok | 37/37 | pass |
| ch12/lab2_filterhunt | emu | 16 | ok | 81/81 | pass |
| ch12/lab2_filterhunt | race | 16 | ok | 24/24 | pass |
| ch12/lab2_filterhunt | metal | 16 | ok | 33/33 | pass |
| ch12/lab3_stable | emu | 16 | ok | 231/231 | pass |
| ch12/lab3_stable | race | 16 | ok | 62/62 | pass |
| ch12/lab3_stable | metal | 16 | ok | 112/112 | pass |
| ch12/lab4_candidates | emu | 16 | ok | 39/39 | pass |
| ch12/lab4_candidates | race | 16 | ok | 5/5 | pass |
| ch12/lab4_candidates | metal | 16 | ok | 17/17 | pass |
| ch12/lab5_inplace | emu | 16 | ok | 108/108 | pass |
| ch12/lab5_inplace | race | 16 | ok | 30/30 | pass |
| ch13/lab1_merge | emu | 16 | ok | 90/90 | pass |
| ch13/lab1_merge | race | 16 | ok | 2/2 | pass |
| ch13/lab1_merge | metal | 16 | ok | 34/34 | pass |
| ch14/lab1_twosorts | emu | 16 | ok | 57/57 | pass |
| ch14/lab1_twosorts | race | 16 | ok | 15/15 | pass |
| ch14/lab1_twosorts | metal | 16 | ok | 23/23 | pass |
| ch14/lab2_multibit | emu | 16 | ok | 48/48 | pass |
| ch14/lab2_multibit | race | 16 | ok | 15/15 | pass |
| ch14/lab2_multibit | metal | 16 | ok | 19/19 | pass |
| ch14/lab3_sortrows | emu | 16 | ok | 24/24 | pass |
| ch14/lab3_sortrows | race | 16 | ok | 5/5 | pass |
| ch14/lab3_sortrows | metal | 16 | ok | 11/11 | pass |
| ch15/lab1_coarse | emu | 16 | ok | 30/30 | pass |
| ch15/lab1_coarse | race | 16 | ok | 2/2 | pass |
| ch15/lab1_coarse | metal | 16 | ok | 14/14 | pass |
| ch15/lab2_regtile | emu | 16 | ok | 30/30 | pass |
| ch15/lab2_regtile | race | 16 | ok | 2/2 | pass |
| ch15/lab2_regtile | metal | 16 | ok | 14/14 | pass |
| ch15/lab3_remap | emu | 16 | ok | 60/60 | pass |
| ch15/lab3_remap | race | 16 | ok | 4/4 | pass |
| ch15/lab3_remap | metal | 16 | ok | 28/28 | pass |
| ch15/lab4_padded | emu | 16 | ok | 60/60 | pass |
| ch15/lab4_padded | race | 16 | ok | 4/4 | pass |
| ch15/lab4_padded | metal | 16 | ok | 28/28 | pass |
| ch15/lab5_pipelined | emu | 16 | ok | 60/60 | pass |
| ch15/lab5_pipelined | race | 16 | ok | 4/4 | pass |
| ch15/lab5_pipelined | metal | 16 | ok | 28/28 | pass |
| ch15/lab6_linear | emu | 16 | ok | 63/63 | pass |
| ch15/lab6_linear | race | 16 | ok | 7/7 | pass |
| ch15/lab6_linear | metal | 16 | ok | 33/33 | pass |
| ch15/lab7_bughunt | emu | 16 | ok | 60/60 | pass |
| ch15/lab7_bughunt | race | 16 | ok | 6/6 | pass |
| ch15/lab7_bughunt | metal | 16 | ok | 28/28 | pass |
| ch16/lab1_wavefront | emu | 16 | ok | 63/63 | pass |
| ch16/lab1_wavefront | race | 16 | ok | 9/9 | pass |
| ch16/lab1_wavefront | metal | 16 | ok | 17/17 | pass |
| ch17/lab1_spmv | emu | 16 | ok | 228/228 | pass |
| ch17/lab1_spmv | race | 16 | ok | 19/19 | pass |
| ch17/lab1_spmv | metal | 16 | ok | 104/104 | pass |
| ch18/lab1_frontier | emu | 16 | ok | 105/105 | pass |
| ch18/lab1_frontier | race | 16 | ok | 35/35 | pass |
| ch18/lab1_frontier | metal | 16 | ok | 19/19 | pass |
| ch19/lab1_convgemm | emu | 16 | ok | 96/96 | pass |
| ch19/lab1_convgemm | race | 16 | ok | 12/12 | pass |
| ch19/lab1_convgemm | metal | 16 | ok | 44/44 | pass |
| ch20/lab1_causal_softmax | emu | 16 | ok | 36/36 | pass |
| ch20/lab1_causal_softmax | race | 16 | ok | 4/4 | pass |
| ch20/lab1_causal_softmax | metal | 16 | ok | 24/24 | pass |
| ch20/lab2_flash | emu | 32 | ok | 12/12 | pass |
| ch20/lab2_flash | emu | 16 | ok | 12/12 | pass |
| ch20/lab2_flash | race | 32 | ok | 2/2 | pass |
| ch20/lab2_flash | metal | 32 | ok | 24/24 | pass |
| ch20/lab3_flash_hunt | emu | 32 | ok | 12/12 | pass |
| ch20/lab3_flash_hunt | race | 32 | ok | 2/2 | pass |
| ch20/lab3_flash_hunt | metal | 32 | ok | 24/24 | pass |
| ch20/lab4_decode | emu | 16 | ok | 57/57 | pass |
| ch20/lab4_decode | race | 16 | ok | 18/18 | pass |
| ch20/lab4_decode | metal | 16 | ok | 38/38 | pass |
| ch20/lab5_gqa_decode | emu | 16 | ok | 57/57 | pass |
| ch20/lab5_gqa_decode | race | 16 | ok | 18/18 | pass |
| ch20/lab5_gqa_decode | metal | 16 | ok | 43/43 | pass |
| ch21/lab1_dcs | emu | 16 | ok | 75/75 | pass |
| ch21/lab1_dcs | race | 16 | ok | 25/25 | pass |
| ch21/lab1_dcs | metal | 16 | ok | 30/30 | pass |
| ch22/lab1_gemv | emu | 16 | ok | 117/117 | pass |
| ch22/lab1_gemv | race | 16 | ok | 10/10 | pass |
| ch22/lab1_gemv | metal | 16 | ok | 66/66 | pass |
| ch22/lab2_batching | emu | 16 | ok | 45/45 | pass |
| ch22/lab2_batching | metal | 16 | ok | 21/21 | pass |
| ch23/lab1_overlap | emu | 16 | ok | 384/384 | pass |
| ch23/lab1_overlap | metal | 16 | ok | 148/148 | pass |
| x1/lab1_topk | emu | 16 | ok | 81/81 | pass |
| x1/lab1_topk | race | 16 | ok | 17/17 | pass |
| x1/lab1_topk | metal | 16 | ok | 64/64 | pass |
| x1/lab2_filters | emu | 16 | ok | 81/81 | pass |
| x1/lab2_filters | race | 16 | ok | 15/15 | pass |
| x1/lab2_filters | metal | 16 | ok | 56/56 | pass |
| x1/lab3_sampler | emu | 16 | ok | 48/48 | pass |
| x1/lab3_sampler | race | 16 | ok | 25/25 | pass |
| x1/lab3_sampler | metal | 16 | ok | 105/105 | pass |
| x1/lab4_filterhunt | emu | 16 | ok | 30/30 | pass |
| x1/lab4_filterhunt | race | 16 | ok | 6/6 | pass |
| x1/lab4_filterhunt | metal | 16 | ok | 20/20 | pass |
| x2/lab1_q8gemv | emu | 16 | ok | 24/24 | pass |
| x2/lab1_q8gemv | metal | 16 | ok | 13/13 | pass |
| x2/lab2_q4gemv | emu | 16 | ok | 30/30 | pass |
| x2/lab2_q4gemv | metal | 16 | ok | 16/16 | pass |
| x2/lab3_qhunt | emu | 16 | ok | 24/24 | pass |
| x2/lab3_qhunt | race | 16 | ok | 8/8 | pass |
| x2/lab3_qhunt | metal | 16 | ok | 8/8 | pass |
| x2/lab4_m9 | emu | 16 | ok | 57/57 | pass |
| x2/lab4_m9 | metal | 16 | ok | 43/43 | pass |
| x3/lab1_llama_pieces | emu | 16 | ok | 132/132 | pass |
| x3/lab1_llama_pieces | race | 16 | ok | 28/28 | pass |
| x3/lab1_llama_pieces | metal | 16 | ok | 44/44 | pass |
| x3/lab2_tinyllama | emu | 16 | ok | 93/93 | pass |
| x3/lab2_tinyllama | metal | 16 | ok | 42/42 | pass |

## Starters (negative control) and your work

An untouched starter must build and must fail its tests; if one passes, the tests are too weak.

| Lab | Backend | State of work/ | Build | Tests passed | Reading |
|---|---|---|---|---|---|
| appA/lab1_sumerror | emu | untouched | ok | 3/51 | starter fails as it should |
| appA/lab1_sumerror | race | untouched | ok | 1/17 | starter fails as it should |
| appA/lab1_sumerror | metal | untouched | ok | 1/39 | starter fails as it should |
| appC/lab1_feeding | emu | untouched | ok | 0/144 | starter fails as it should |
| appC/lab1_feeding | metal | untouched | ok | 0/60 | starter fails as it should |
| ch02/lab1_vecadd | emu | untouched | ok | 0/60 | starter fails as it should |
| ch02/lab1_vecadd | metal | untouched | ok | 0/10 | starter fails as it should |
| ch02/lab2_timing | emu | untouched | ok | 15/39 | starter fails as it should |
| ch02/lab2_timing | metal | untouched | ok | 0/10 | starter fails as it should |
| ch02/lab3_elementwise | emu | untouched | ok | 0/78 | starter fails as it should |
| ch02/lab3_elementwise | metal | untouched | ok | 0/24 | starter fails as it should |
| ch02/lab4_bughunt | emu | untouched | ok | 0/21 | starter fails as it should |
| ch02/lab4_bughunt | metal | untouched | ok | 0/7 | starter fails as it should |
| ch03/lab1_grayscale | emu | untouched | ok | 0/33 | starter fails as it should |
| ch03/lab1_grayscale | metal | untouched | ok | 0/12 | starter fails as it should |
| ch03/lab2_blur | emu | untouched | ok | 0/30 | starter fails as it should |
| ch03/lab2_blur | metal | untouched | ok | 0/11 | starter fails as it should |
| ch03/lab3_naive_gemm | emu | untouched | ok | 0/30 | starter fails as it should |
| ch03/lab3_naive_gemm | race | untouched | ok | 0/10 | starter fails as it should |
| ch03/lab3_naive_gemm | metal | untouched | ok | 0/12 | starter fails as it should |
| ch03/lab4_granularity | emu | untouched | ok | 15/45 | starter fails as it should |
| ch03/lab4_granularity | race | untouched | ok | 5/15 | starter fails as it should |
| ch03/lab4_granularity | metal | untouched | ok | 7/21 | starter fails as it should |
| ch03/lab5_maphunt | emu | untouched | ok | 3/6 (exit -6: crashed or aborted) | starter fails as it should |
| ch03/lab5_maphunt | race | untouched | ok | 1/10 (ThreadSanitizer: 1 data race report) | starter fails as it should |
| ch03/lab5_maphunt | metal | untouched | ok | 1/10 | starter fails as it should |
| ch04/lab1_launchconfig | emu | untouched | ok | 0/42 | starter fails as it should |
| ch04/lab1_launchconfig | metal | untouched | ok | 0/13 | starter fails as it should |
| ch04/lab2_divergence | emu | untouched | ok | 0/60 | starter fails as it should |
| ch04/lab2_divergence | metal | untouched | ok | 0/20 | starter fails as it should |
| ch04/lab3_barrierhunt | emu | untouched | ok | 0/3 (exit -6: crashed or aborted) | starter fails as it should |
| ch04/lab3_barrierhunt | race | untouched | ok | 0/10 (ThreadSanitizer: 1 data race report) | starter fails as it should |
| ch04/lab3_barrierhunt | metal | untouched | ok | 1/10 | starter fails as it should |
| ch04/lab4_blocksize | emu | untouched | ok | 0/45 | starter fails as it should |
| ch04/lab4_blocksize | metal | untouched | ok | 0/7 | starter fails as it should |
| ch05/lab1_roofline | emu | untouched | ok | 6/21 | starter fails as it should |
| ch05/lab1_roofline | metal | untouched | ok | 2/9 | starter fails as it should |
| ch05/lab2_tiled | emu | untouched | ok | 0/9 | starter fails as it should |
| ch05/lab2_tiled | race | untouched | ok | 0/5 | starter fails as it should |
| ch05/lab2_tiled | metal | untouched | ok | 0/10 | starter fails as it should |
| ch05/lab3_gemm | emu | untouched | ok | 0/24 | starter fails as it should |
| ch05/lab3_gemm | race | untouched | ok | 0/8 | starter fails as it should |
| ch05/lab3_gemm | metal | untouched | ok | 0/10 | starter fails as it should |
| ch05/lab4_racehunt | emu | untouched | ok | 3/6 (exit -6: crashed or aborted) | starter fails as it should |
| ch05/lab4_racehunt | race | untouched | ok | 1/6 (ThreadSanitizer: 109 data race reports) | starter fails as it should |
| ch05/lab4_racehunt | metal | untouched | ok | 3/8 | starter fails as it should |
| ch05/lab5_dynamic | emu | untouched | ok | 0/48 | starter fails as it should |
| ch05/lab5_dynamic | race | untouched | ok | 0/16 | starter fails as it should |
| ch05/lab5_dynamic | metal | untouched | ok | 0/24 | starter fails as it should |
| ch06/lab1_coalescing | emu | untouched | ok | 0/111 | starter fails as it should |
| ch06/lab1_coalescing | metal | untouched | ok | 0/42 | starter fails as it should |
| ch06/lab2_vector | emu | untouched | ok | 3/87 | starter fails as it should |
| ch06/lab2_vector | metal | untouched | ok | 1/35 | starter fails as it should |
| ch06/lab3_banks | emu | untouched | ok | 6/48 | starter fails as it should |
| ch06/lab3_banks | race | untouched | ok | 2/16 | starter fails as it should |
| ch06/lab3_banks | metal | untouched | ok | 2/38 | starter fails as it should |
| ch06/lab4_linear | emu | untouched | ok | 24/48 | starter fails as it should |
| ch06/lab4_linear | race | untouched | ok | 8/16 | starter fails as it should |
| ch06/lab4_linear | metal | untouched | ok | 22/44 | starter fails as it should |
| ch06/lab5_doublebuf | emu | untouched | ok | 6/12 | starter fails as it should |
| ch06/lab5_doublebuf | race | untouched | ok | 1/3 (ThreadSanitizer: 32 data race reports) | starter fails as it should |
| ch06/lab5_doublebuf | metal | untouched | ok | 5/9 | starter fails as it should |
| ch07/lab1_constconv | emu | untouched | ok | 0/45 | starter fails as it should |
| ch07/lab1_constconv | metal | untouched | ok | 0/21 | starter fails as it should |
| ch07/lab2_tiledconv | emu | untouched | ok | 9/57 (exit -6: crashed or aborted) | starter fails as it should |
| ch07/lab2_tiledconv | race | untouched | ok | 3/35 | starter fails as it should |
| ch07/lab2_tiledconv | metal | untouched | ok | 7/251 | starter fails as it should |
| ch08/lab1_stencil3d | emu | untouched | ok | 36/126 | starter fails as it should |
| ch08/lab1_stencil3d | race | untouched | ok | 8/28 | starter fails as it should |
| ch08/lab1_stencil3d | metal | untouched | ok | 20/85 | starter fails as it should |
| ch09/lab1_racehunt | emu | untouched | ok | 19/36 | starter fails as it should |
| ch09/lab1_racehunt | race | untouched | ok | 1/12 (ThreadSanitizer: 3 data race reports) | starter fails as it should |
| ch09/lab1_racehunt | metal | untouched | ok | 1/16 | starter fails as it should |
| ch09/lab2_privatised | emu | untouched | ok | 6/42 | starter fails as it should |
| ch09/lab2_privatised | race | untouched | ok | 2/14 | starter fails as it should |
| ch09/lab2_privatised | metal | untouched | ok | 2/18 | starter fails as it should |
| ch09/lab3_tokencounts | emu | untouched | ok | 9/90 | starter fails as it should |
| ch09/lab3_tokencounts | race | untouched | ok | 3/30 | starter fails as it should |
| ch09/lab3_tokencounts | metal | untouched | ok | 8/80 | starter fails as it should |
| ch10/lab1_sum | emu | untouched | ok | 12/48 | starter fails as it should |
| ch10/lab1_sum | race | untouched | ok | 2/8 | starter fails as it should |
| ch10/lab1_sum | metal | untouched | ok | 8/24 | starter fails as it should |
| ch10/lab2_reducehunt | emu | untouched | ok | 0/0 (exit -6: crashed or aborted) | starter fails as it should |
| ch10/lab2_reducehunt | race | untouched | ok | 0/7 (ThreadSanitizer: 2 data race reports) | starter fails as it should |
| ch10/lab2_reducehunt | metal | untouched | ok | 2/9 | starter fails as it should |
| ch10/lab3_softmax_argmax | emu | untouched | ok | 18/111 | starter fails as it should |
| ch10/lab3_softmax_argmax | race | untouched | ok | 2/15 | starter fails as it should |
| ch10/lab3_softmax_argmax | metal | untouched | ok | 8/48 | starter fails as it should |
| ch11/lab1_blockscan | emu | untouched | ok | 12/24 | starter fails as it should |
| ch11/lab1_blockscan | race | untouched | ok | 2/4 | starter fails as it should |
| ch11/lab1_blockscan | metal | untouched | ok | 8/20 | starter fails as it should |
| ch11/lab2_coarsened | emu | untouched | ok | 24/60 | starter fails as it should |
| ch11/lab2_coarsened | race | untouched | ok | 4/10 | starter fails as it should |
| ch11/lab2_coarsened | metal | untouched | ok | 18/46 | starter fails as it should |
| ch11/lab3_scanhunt | emu | untouched | ok | 16/48 | starter fails as it should |
| ch11/lab3_scanhunt | race | untouched | ok | 0/3 (ThreadSanitizer: 17 data race reports) | starter fails as it should |
| ch11/lab3_scanhunt | metal | untouched | ok | 8/20 | starter fails as it should |
| ch11/lab4_global | emu | untouched | ok | 6/18 | starter fails as it should |
| ch11/lab4_global | race | untouched | ok | 1/3 | starter fails as it should |
| ch11/lab4_global | metal | untouched | ok | 3/9 | starter fails as it should |
| ch11/lab5_singlepass | emu | untouched | ok | 6/12 | starter fails as it should |
| ch11/lab5_singlepass | race | untouched | ok | 1/2 | starter fails as it should |
| ch11/lab6_topp | emu | untouched | ok | 18/108 | starter fails as it should |
| ch11/lab6_topp | race | untouched | ok | 2/17 | starter fails as it should |
| ch11/lab6_topp | metal | untouched | ok | 8/47 | starter fails as it should |
| ch12/lab1_unstable | emu | untouched | ok | 3/84 | starter fails as it should |
| ch12/lab1_unstable | race | untouched | ok | 1/25 | starter fails as it should |
| ch12/lab1_unstable | metal | untouched | ok | 1/37 | starter fails as it should |
| ch12/lab2_filterhunt | emu | untouched | ok | 27/27 (exit -6: crashed or aborted) | starter fails as it should |
| ch12/lab2_filterhunt | race | untouched | ok | 2/24 (ThreadSanitizer: 3 data race reports) | starter fails as it should |
| ch12/lab2_filterhunt | metal | untouched | ok | 4/33 | starter fails as it should |
| ch12/lab3_stable | emu | untouched | ok | 24/231 | starter fails as it should |
| ch12/lab3_stable | race | untouched | ok | 5/62 | starter fails as it should |
| ch12/lab3_stable | metal | untouched | ok | 10/112 | starter fails as it should |
| ch12/lab4_candidates | emu | untouched | ok | 9/39 | starter fails as it should |
| ch12/lab4_candidates | race | untouched | ok | 1/5 | starter fails as it should |
| ch12/lab4_candidates | metal | untouched | ok | 4/17 | starter fails as it should |
| ch12/lab5_inplace | emu | untouched | ok | 0/108 | starter fails as it should |
| ch12/lab5_inplace | race | untouched | ok | 0/30 | starter fails as it should |
| ch13/lab1_merge | emu | untouched | ok | 6/90 | starter fails as it should |
| ch13/lab1_merge | race | untouched | ok | 0/2 | starter fails as it should |
| ch13/lab1_merge | metal | untouched | ok | 2/34 | starter fails as it should |
| ch14/lab1_twosorts | emu | untouched | ok | 21/57 | starter fails as it should |
| ch14/lab1_twosorts | race | untouched | ok | 4/15 | starter fails as it should |
| ch14/lab1_twosorts | metal | untouched | ok | 8/23 | starter fails as it should |
| ch14/lab2_multibit | emu | untouched | ok | 3/48 | starter fails as it should |
| ch14/lab2_multibit | race | untouched | ok | 1/15 | starter fails as it should |
| ch14/lab2_multibit | metal | untouched | ok | 1/19 | starter fails as it should |
| ch14/lab3_sortrows | emu | untouched | ok | 0/24 | starter fails as it should |
| ch14/lab3_sortrows | race | untouched | ok | 0/5 | starter fails as it should |
| ch14/lab3_sortrows | metal | untouched | ok | 0/11 | starter fails as it should |
| ch15/lab1_coarse | emu | untouched | ok | 0/30 | starter fails as it should |
| ch15/lab1_coarse | race | untouched | ok | 0/2 | starter fails as it should |
| ch15/lab1_coarse | metal | untouched | ok | 0/14 | starter fails as it should |
| ch15/lab2_regtile | emu | untouched | ok | 3/30 | starter fails as it should |
| ch15/lab2_regtile | race | untouched | ok | 0/2 | starter fails as it should |
| ch15/lab2_regtile | metal | untouched | ok | 1/14 | starter fails as it should |
| ch15/lab3_remap | emu | untouched | ok | 0/60 | starter fails as it should |
| ch15/lab3_remap | race | untouched | ok | 0/4 | starter fails as it should |
| ch15/lab3_remap | metal | untouched | ok | 0/28 | starter fails as it should |
| ch15/lab4_padded | emu | untouched | ok | 6/60 | starter fails as it should |
| ch15/lab4_padded | race | untouched | ok | 0/4 | starter fails as it should |
| ch15/lab4_padded | metal | untouched | ok | 2/28 | starter fails as it should |
| ch15/lab5_pipelined | emu | untouched | ok | 6/60 | starter fails as it should |
| ch15/lab5_pipelined | race | untouched | ok | 0/4 | starter fails as it should |
| ch15/lab5_pipelined | metal | untouched | ok | 2/28 | starter fails as it should |
| ch15/lab6_linear | emu | untouched | ok | 0/63 | starter fails as it should |
| ch15/lab6_linear | race | untouched | ok | 0/7 | starter fails as it should |
| ch15/lab6_linear | metal | untouched | ok | 4/33 | starter fails as it should |
| ch15/lab7_bughunt | emu | untouched | ok | 3/3 (exit -6: crashed or aborted) | starter fails as it should |
| ch15/lab7_bughunt | race | untouched | ok | 0/6 (ThreadSanitizer: 563 data race reports) | starter fails as it should |
| ch15/lab7_bughunt | metal | untouched | ok | 4/28 | starter fails as it should |
| ch16/lab1_wavefront | emu | untouched | ok | 3/63 | starter fails as it should |
| ch16/lab1_wavefront | race | untouched | ok | 1/9 | starter fails as it should |
| ch16/lab1_wavefront | metal | untouched | ok | 1/17 | starter fails as it should |
| ch17/lab1_spmv | emu | untouched | ok | 39/228 | starter fails as it should |
| ch17/lab1_spmv | race | untouched | ok | 3/19 | starter fails as it should |
| ch17/lab1_spmv | metal | untouched | ok | 15/104 | starter fails as it should |
| ch18/lab1_frontier | emu | untouched | ok | 15/57 | starter fails as it should |
| ch18/lab1_frontier | race | untouched | ok | 5/19 | starter fails as it should |
| ch18/lab1_frontier | metal | untouched | ok | 5/19 | starter fails as it should |
| ch19/lab1_convgemm | emu | untouched | ok | 24/96 | starter fails as it should |
| ch19/lab1_convgemm | race | untouched | ok | 3/12 | starter fails as it should |
| ch19/lab1_convgemm | metal | untouched | ok | 11/44 | starter fails as it should |
| ch20/lab1_causal_softmax | emu | untouched | ok | 0/36 | starter fails as it should |
| ch20/lab1_causal_softmax | race | untouched | ok | 0/4 | starter fails as it should |
| ch20/lab1_causal_softmax | metal | untouched | ok | 0/24 | starter fails as it should |
| ch20/lab2_flash | emu | untouched | ok | 0/12 | starter fails as it should |
| ch20/lab2_flash | race | untouched | ok | 0/2 | starter fails as it should |
| ch20/lab2_flash | metal | untouched | ok | 0/1 | starter fails as it should |
| ch20/lab3_flash_hunt | emu | untouched | ok | 3/12 | starter fails as it should |
| ch20/lab3_flash_hunt | race | untouched | ok | 0/2 (ThreadSanitizer: 2 data race reports) | starter fails as it should |
| ch20/lab3_flash_hunt | metal | untouched | ok | 4/24 | starter fails as it should |
| ch20/lab4_decode | emu | untouched | ok | 3/57 | starter fails as it should |
| ch20/lab4_decode | race | untouched | ok | 1/18 | starter fails as it should |
| ch20/lab4_decode | metal | untouched | ok | 1/38 | starter fails as it should |
| ch20/lab5_gqa_decode | emu | untouched | ok | 6/57 | starter fails as it should |
| ch20/lab5_gqa_decode | race | untouched | ok | 2/18 | starter fails as it should |
| ch20/lab5_gqa_decode | metal | untouched | ok | 2/43 | starter fails as it should |
| ch21/lab1_dcs | emu | untouched | ok | 15/75 | starter fails as it should |
| ch21/lab1_dcs | race | untouched | ok | 5/25 | starter fails as it should |
| ch21/lab1_dcs | metal | untouched | ok | 6/30 | starter fails as it should |
| ch22/lab1_gemv | emu | untouched | ok | 0/117 | starter fails as it should |
| ch22/lab1_gemv | race | untouched | ok | 0/10 (ThreadSanitizer: 3 data race reports) | starter fails as it should |
| ch22/lab1_gemv | metal | untouched | ok | 0/66 | starter fails as it should |
| ch22/lab2_batching | emu | untouched | ok | 0/45 | starter fails as it should |
| ch22/lab2_batching | metal | untouched | ok | 0/21 | starter fails as it should |
| ch23/lab1_overlap | emu | untouched | ok | 0/384 | starter fails as it should |
| ch23/lab1_overlap | metal | untouched | ok | 0/148 | starter fails as it should |
| x1/lab1_topk | emu | untouched | ok | 3/81 | starter fails as it should |
| x1/lab1_topk | race | untouched | ok | 1/17 | starter fails as it should |
| x1/lab1_topk | metal | untouched | ok | 3/64 | starter fails as it should |
| x1/lab2_filters | emu | untouched | ok | 3/81 | starter fails as it should |
| x1/lab2_filters | race | untouched | ok | 1/15 | starter fails as it should |
| x1/lab2_filters | metal | untouched | ok | 4/56 | starter fails as it should |
| x1/lab3_sampler | emu | untouched | ok | 3/48 | starter fails as it should |
| x1/lab3_sampler | race | untouched | ok | 1/25 | starter fails as it should |
| x1/lab3_sampler | metal | untouched | ok | 0/105 | starter fails as it should |
| x1/lab4_filterhunt | emu | untouched | ok | 3/30 | starter fails as it should |
| x1/lab4_filterhunt | race | untouched | ok | 1/6 (ThreadSanitizer: 1 data race report) | starter fails as it should |
| x1/lab4_filterhunt | metal | untouched | ok | 2/20 | starter fails as it should |
| x2/lab1_q8gemv | emu | untouched | ok | 6/24 | starter fails as it should |
| x2/lab1_q8gemv | metal | untouched | ok | 2/13 | starter fails as it should |
| x2/lab2_q4gemv | emu | untouched | ok | 9/30 | starter fails as it should |
| x2/lab2_q4gemv | metal | untouched | ok | 4/16 | starter fails as it should |
| x2/lab3_qhunt | emu | untouched | ok | 3/24 | starter fails as it should |
| x2/lab3_qhunt | race | untouched | ok | 1/8 (ThreadSanitizer: 4 data race reports) | starter fails as it should |
| x2/lab3_qhunt | metal | untouched | ok | 1/8 | starter fails as it should |
| x2/lab4_m9 | emu | untouched | ok | 0/57 | starter fails as it should |
| x2/lab4_m9 | metal | untouched | ok | 0/43 | starter fails as it should |
| x3/lab1_llama_pieces | emu | untouched | ok | 18/120 (exit -6: crashed or aborted) | starter fails as it should |
| x3/lab1_llama_pieces | race | untouched | ok | 4/28 (ThreadSanitizer: 1 data race report) | starter fails as it should |
| x3/lab1_llama_pieces | metal | untouched | ok | 8/44 | starter fails as it should |
| x3/lab2_tinyllama | emu | untouched | ok | 63/93 | starter fails as it should |
| x3/lab2_tinyllama | metal | untouched | ok | 21/42 | starter fails as it should |

## Emulator self-tests

These test the emulator itself (`labs/selftest`), not any lab. A row marked *must be reported* is a
planted data race: it is right only when ThreadSanitizer reports it.

| Self-test | Backend | Build | Cases passed | Expectation | Verdict |
|---|---|---|---|---|---|
| warp | emu | ok | 39/39 | all cases pass, no race report | pass |
| warp | race | ok | 13/13 | all cases pass, no race report | pass |
| atomics | emu | ok | 15/15 | all cases pass, no race report | pass |
| atomics | race | ok | 5/5 | all cases pass, no race report | pass |
| constant | emu | ok | 9/9 | all cases pass, no race report | pass |
| constant | race | ok | 3/3 | all cases pass, no race report | pass |
| shared | emu | ok | 18/18 | all cases pass, no race report | pass |
| shared | race | ok | 6/6 | all cases pass, no race report | pass |
| cgroups | emu | ok | 15/15 | all cases pass, no race report | pass |
| cgroups | race | ok | 5/5 | all cases pass, no race report | pass |
| hostapi | emu | ok | 21/21 | all cases pass, no race report | pass |
| hostapi | race | ok | 7/7 | all cases pass, no race report | pass |
| intrinsics | emu | ok | 6/6 | all cases pass, no race report | pass |
| intrinsics | race | ok | 2/2 | all cases pass, no race report | pass |
| halfprec | emu | ok | 9/9 | all cases pass, no race report | pass |
| halfprec | race | ok | 3/3 | all cases pass, no race report | pass |
| race_hist_atomic | emu | ok | 9/9 | all cases pass, no race report | pass |
| race_hist_atomic | race | ok | 3/3 | all cases pass, no race report | pass |
| race_hist_plain | emu | ok | 3/3 | all cases pass, no race report | pass |
| race_hist_plain | race | ok | 0/1 (ThreadSanitizer: 1 data race report) | data race *must be reported* | race reported, as it must be |
| race_negatives | race | ok | 0/4 (ThreadSanitizer: 5 data race reports) | data race *must be reported* | race reported, as it must be |

ThreadSanitizer key lines for `selftest_race_hist_plain.race`:

```
WARNING: ThreadSanitizer: data race
  Write of size 4 by thread T15:
      kHistPlain(unsigned char const*, unsigned int*, int) race_hist_plain.cu:11
  Previous write of size 4 by thread T11:
      kHistPlain(unsigned char const*, unsigned int*, int) race_hist_plain.cu:11
  Location is heap block of size 16 allocated by main thread:
SUMMARY: ThreadSanitizer: data race race_hist_plain.cu:11 in kHistPlain(unsigned char const*, unsigned int*, int)
```

ThreadSanitizer key lines for `selftest_race_negatives.race`:

```
WARNING: ThreadSanitizer: data race
  Write of size 4 by thread T39:
      kNoBarrier(int const*, int*) race_negatives.cu:9
  Previous read of size 4 by thread T26:
      kNoBarrier(int const*, int*) race_negatives.cu:11
  Location is heap block of size 256 allocated by main thread:
SUMMARY: ThreadSanitizer: data race race_negatives.cu:9 in kNoBarrier(int const*, int*)
WARNING: ThreadSanitizer: data race
  Read of size 4 by thread T225:
      kBlocksRace(int const*, int*) race_negatives.cu:29
  Previous write of size 4 by thread T129:
      kBlocksRace(int const*, int*) race_negatives.cu:29
  Location is heap block of size 4 allocated by main thread:
SUMMARY: ThreadSanitizer: data race race_negatives.cu:29 in kBlocksRace(int const*, int*)
WARNING: ThreadSanitizer: data race
  Read of size 4 by thread T407:
      kShuffleIsNoBarrier(int*) race_negatives.cu:46
  Previous write of size 4 by thread T408:
      kShuffleIsNoBarrier(int*) race_negatives.cu:44
  Location is heap block of size 128 allocated by main thread:
SUMMARY: ThreadSanitizer: data race race_negatives.cu:46 in kShuffleIsNoBarrier(int*)
WARNING: ThreadSanitizer: data race
  Write of size 4 by thread T417:
      kBetweenBarriers(int*) race_negatives.cu:66
  Previous read of size 4 by thread T480:
      kBetweenBarriers(int*) race_negatives.cu:65
  Location is heap block of size 256 allocated by main thread:
SUMMARY: ThreadSanitizer: data race race_negatives.cu:66 in kBetweenBarriers(int*)
(1 more ThreadSanitizer report not shown: 1 on the lines above, 0 elsewhere)
```

## Benchmarks (this machine, this run)

### appA/lab1_sumerror, solution, metal, TILE_WIDTH=16

| name | rounding | keeps_denormals | fuses_mul_add | method | acc | data | n | coarse | block | result | reference | sum_abs | abs_err | rel_err | ulps | matches_model | model_rounding | seconds | gb_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe | nearest | False | True |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| accuracy |  |  |  | serial | float | ones | 65536 | 1 | 1 | 65536 | 65536 | 65536 | 0 | 0 | 0 | True | nearest |  |  |
| accuracy |  |  |  | serial | float | uniform | 65536 | 1 | 1 | 3.277e+04 | 3.277e+04 | 3.277e+04 | 0.0198 | 6.043e-07 | 5.07 | True | nearest |  |  |
| accuracy |  |  |  | serial | float | mixed | 65536 | 1 | 1 | 2066 | 2066 | 2066 | 0.07595 | 3.676e-05 | 311.1 | True | nearest |  |  |
| accuracy |  |  |  | serial | float | mixed-asc | 65536 | 1 | 1 | 2058 | 2058 | 2058 | 0.00213 | 1.035e-06 | 8.725 | True | nearest |  |  |
| accuracy |  |  |  | serial | float | mixed-desc | 65536 | 1 | 1 | 2055 | 2055 | 2055 | 0.1855 | 9.026e-05 | 759.6 | True | nearest |  |  |
| accuracy |  |  |  | serial | float | cancel | 65536 | 1 | 1 | -12.65 | -12.65 | 2043 | 4.842e-06 | 2.37e-09 | 5.077 | True | nearest |  |  |
| accuracy |  |  |  | serial | float | softmax | 65536 | 1 | 1 | 7240 | 7240 | 7240 | 0.1022 | 1.411e-05 | 209.2 | True | nearest |  |  |
| accuracy |  |  |  | serial | float | big-small | 65536 | 1 | 1 | 1 | 1.004 | 1.004 | 0.003906 | 0.003891 | 3.277e+04 | True | nearest |  |  |
| accuracy |  |  |  | tree | float | ones | 65536 | 16 | 256 | 65536 | 65536 | 65536 | 0 | 0 | 0 | True | nearest |  |  |
| accuracy |  |  |  | tree | float | uniform | 65536 | 16 | 256 | 3.277e+04 | 3.277e+04 | 3.277e+04 | 0.0002728 | 8.323e-09 | 0.06982 | True | nearest |  |  |
| accuracy |  |  |  | tree | float | mixed | 65536 | 16 | 256 | 2066 | 2066 | 2066 | 1.863e-05 | 9.017e-09 | 0.07631 | True | nearest |  |  |
| accuracy |  |  |  | tree | float | mixed-asc | 65536 | 16 | 256 | 2058 | 2058 | 2058 | 6.707e-05 | 3.26e-08 | 0.2747 | True | nearest |  |  |
| accuracy |  |  |  | tree | float | mixed-desc | 65536 | 16 | 256 | 2055 | 2055 | 2055 | 8.582e-05 | 4.177e-08 | 0.3515 | True | nearest |  |  |
| accuracy |  |  |  | tree | float | cancel | 65536 | 16 | 256 | -12.65 | -12.65 | 2043 | 1.834e-06 | 8.978e-10 | 1.923 | True | nearest |  |  |
| accuracy |  |  |  | tree | float | softmax | 65536 | 16 | 256 | 7240 | 7240 | 7240 | 0.0001129 | 1.56e-08 | 0.2313 | True | nearest |  |  |
| accuracy |  |  |  | tree | float | big-small | 65536 | 16 | 256 | 1.004 | 1.004 | 1.004 | 8.941e-07 | 8.906e-07 | 7.5 | True | nearest |  |  |
| accuracy |  |  |  | kahan | float | ones | 65536 | 16 | 256 | 65536 | 65536 | 65536 | 0 | 0 | 0 | True | nearest |  |  |
| accuracy |  |  |  | kahan | float | uniform | 65536 | 16 | 256 | 3.277e+04 | 3.277e+04 | 3.277e+04 | 0.003633 | 1.109e-07 | 0.9302 | True | nearest |  |  |
| accuracy |  |  |  | kahan | float | mixed | 65536 | 16 | 256 | 2066 | 2066 | 2066 | 1.863e-05 | 9.017e-09 | 0.07631 | True | nearest |  |  |
| accuracy |  |  |  | kahan | float | mixed-asc | 65536 | 16 | 256 | 2058 | 2058 | 2058 | 0.0001771 | 8.605e-08 | 0.7253 | True | nearest |  |  |
| accuracy |  |  |  | kahan | float | mixed-desc | 65536 | 16 | 256 | 2055 | 2055 | 2055 | 8.582e-05 | 4.177e-08 | 0.3515 | True | nearest |  |  |
| accuracy |  |  |  | kahan | float | cancel | 65536 | 16 | 256 | -12.65 | -12.65 | 2043 | 7.349e-08 | 3.598e-11 | 0.07706 | True | nearest |  |  |
| accuracy |  |  |  | kahan | float | softmax | 65536 | 16 | 256 | 7240 | 7240 | 7240 | 0.0001129 | 1.56e-08 | 0.2313 | True | nearest |  |  |
| accuracy |  |  |  | kahan | float | big-small | 65536 | 16 | 256 | 1.004 | 1.004 | 1.004 | 5.96e-08 | 5.937e-08 | 0.5 | True | nearest |  |  |
| accuracy |  |  |  | serial | half | ones | 65536 | 1 | 1 | 2048 | 65536 | 65536 | 63488 | 0.9688 | 1984 | True | nearest |  |  |
| accuracy |  |  |  | serial | half | uniform | 65536 | 1 | 1 | 2048 | 3.277e+04 | 3.277e+04 | 3.072e+04 | 0.9375 | 960.1 | True | nearest |  |  |
| accuracy |  |  |  | serial | half | mixed | 65536 | 1 | 1 | 512 | 2066 | 2066 | 1554 | 0.7522 | 777 | True | nearest |  |  |
| accuracy |  |  |  | serial | half | mixed-asc | 65536 | 1 | 1 | 512 | 2058 | 2058 | 1546 | 0.7512 | 772.9 | True | nearest |  |  |
| accuracy |  |  |  | serial | half | mixed-desc | 65536 | 1 | 1 | 512 | 2055 | 2055 | 1543 | 0.7508 | 771.4 | True | nearest |  |  |
| accuracy |  |  |  | serial | half | cancel | 65536 | 1 | 1 | -12.84 | -12.65 | 2043 | 0.1852 | 9.066e-05 | 23.7 | True | nearest |  |  |
| accuracy |  |  |  | serial | half | softmax | 65536 | 1 | 1 | 2048 | 7240 | 7240 | 5192 | 0.7171 | 1298 | True | nearest |  |  |
| accuracy |  |  |  | serial | half | big-small | 65536 | 1 | 1 | 1 | 33 | 33 | 32 | 0.9697 | 1024 | True | nearest |  |  |
| accuracy |  |  |  | tree | half | ones | 65536 | 16 | 256 | None | 65536 | 65536 | None | None | None | True | nearest |  |  |
| accuracy |  |  |  | tree | half | uniform | 65536 | 16 | 256 | 32768 | 3.277e+04 | 3.277e+04 | 3.802 | 0.000116 | 0.1188 | True | nearest |  |  |
| accuracy |  |  |  | tree | half | mixed | 65536 | 16 | 256 | 2066 | 2066 | 2066 | 0.05361 | 2.595e-05 | 0.0268 | True | nearest |  |  |
| accuracy |  |  |  | tree | half | mixed-asc | 65536 | 16 | 256 | 2056 | 2058 | 2058 | 1.719 | 0.0008354 | 0.8595 | True | nearest |  |  |
| accuracy |  |  |  | tree | half | mixed-desc | 65536 | 16 | 256 | 2056 | 2055 | 2055 | 1.267 | 0.0006166 | 0.6335 | True | nearest |  |  |
| accuracy |  |  |  | tree | half | cancel | 65536 | 16 | 256 | -12.64 | -12.65 | 2043 | 0.01014 | 4.967e-06 | 1.299 | True | nearest |  |  |
| accuracy |  |  |  | tree | half | softmax | 65536 | 16 | 256 | 7240 | 7240 | 7240 | 0.2622 | 3.621e-05 | 0.06555 | True | nearest |  |  |
| accuracy |  |  |  | tree | half | big-small | 65536 | 16 | 256 | 33 | 33 | 33 | 0.0004883 | 1.48e-05 | 0.01562 | True | nearest |  |  |
| accuracy |  |  |  | kahan | half | ones | 65536 | 16 | 256 | None | 65536 | 65536 | None | None | None | True | nearest |  |  |
| accuracy |  |  |  | kahan | half | uniform | 65536 | 16 | 256 | 32768 | 3.277e+04 | 3.277e+04 | 3.802 | 0.000116 | 0.1188 | True | nearest |  |  |
| accuracy |  |  |  | kahan | half | mixed | 65536 | 16 | 256 | 2066 | 2066 | 2066 | 0.05361 | 2.595e-05 | 0.0268 | True | nearest |  |  |
| accuracy |  |  |  | kahan | half | mixed-asc | 65536 | 16 | 256 | 2056 | 2058 | 2058 | 1.719 | 0.0008354 | 0.8595 | True | nearest |  |  |
| accuracy |  |  |  | kahan | half | mixed-desc | 65536 | 16 | 256 | 2056 | 2055 | 2055 | 1.267 | 0.0006166 | 0.6335 | True | nearest |  |  |
| accuracy |  |  |  | kahan | half | cancel | 65536 | 16 | 256 | -12.66 | -12.65 | 2043 | 0.00548 | 2.683e-06 | 0.7015 | True | nearest |  |  |
| accuracy |  |  |  | kahan | half | softmax | 65536 | 16 | 256 | 7240 | 7240 | 7240 | 0.2622 | 3.621e-05 | 0.06555 | True | nearest |  |  |
| accuracy |  |  |  | kahan | half | big-small | 65536 | 16 | 256 | 33 | 33 | 33 | 0.0004883 | 1.48e-05 | 0.01562 | True | nearest |  |  |
| accuracy |  |  |  | serial | bfloat | ones | 65536 | 1 | 1 | 256 | 65536 | 65536 | 65280 | 0.9961 | 127.5 | True | nearest |  |  |
| accuracy |  |  |  | serial | bfloat | uniform | 65536 | 1 | 1 | 256 | 3.277e+04 | 3.277e+04 | 3.252e+04 | 0.9922 | 127 | True | nearest |  |  |
| accuracy |  |  |  | serial | bfloat | mixed | 65536 | 1 | 1 | 64 | 2066 | 2066 | 2002 | 0.969 | 125.1 | True | nearest |  |  |
| accuracy |  |  |  | serial | bfloat | mixed-asc | 65536 | 1 | 1 | 64 | 2058 | 2058 | 1994 | 0.9689 | 124.6 | True | nearest |  |  |
| accuracy |  |  |  | serial | bfloat | mixed-desc | 65536 | 1 | 1 | 64 | 2055 | 2055 | 1991 | 0.9689 | 124.4 | True | nearest |  |  |
| accuracy |  |  |  | serial | bfloat | cancel | 65536 | 1 | 1 | -13.81 | -12.66 | 2043 | 1.149 | 0.0005624 | 18.38 | True | nearest |  |  |
| accuracy |  |  |  | serial | bfloat | softmax | 65536 | 1 | 1 | 256 | 7240 | 7240 | 6984 | 0.9646 | 218.2 | True | nearest |  |  |
| accuracy |  |  |  | serial | bfloat | big-small | 65536 | 1 | 1 | 1 | 257 | 257 | 256 | 0.9961 | 128 | True | nearest |  |  |
| accuracy |  |  |  | tree | bfloat | ones | 65536 | 16 | 256 | 65536 | 65536 | 65536 | 0 | 0 | 0 | True | nearest |  |  |
| accuracy |  |  |  | tree | bfloat | uniform | 65536 | 16 | 256 | 32768 | 3.277e+04 | 3.277e+04 | 3.772 | 0.0001151 | 0.01473 | True | nearest |  |  |
| accuracy |  |  |  | tree | bfloat | mixed | 65536 | 16 | 256 | 2064 | 2066 | 2066 | 2.003 | 0.0009697 | 0.1252 | True | nearest |  |  |
| accuracy |  |  |  | tree | bfloat | mixed-asc | 65536 | 16 | 256 | 2048 | 2058 | 2058 | 9.703 | 0.004715 | 0.6064 | True | nearest |  |  |
| accuracy |  |  |  | tree | bfloat | mixed-desc | 65536 | 16 | 256 | 2048 | 2055 | 2055 | 6.712 | 0.003267 | 0.4195 | True | nearest |  |  |
| accuracy |  |  |  | tree | bfloat | cancel | 65536 | 16 | 256 | -12.75 | -12.66 | 2043 | 0.08627 | 4.223e-05 | 1.38 | True | nearest |  |  |
| accuracy |  |  |  | tree | bfloat | softmax | 65536 | 16 | 256 | 7200 | 7240 | 7240 | 39.67 | 0.005479 | 1.24 | True | nearest |  |  |
| accuracy |  |  |  | tree | bfloat | big-small | 65536 | 16 | 256 | 256 | 257 | 257 | 0.9961 | 0.003876 | 0.498 | True | nearest |  |  |
| accuracy |  |  |  | kahan | bfloat | ones | 65536 | 16 | 256 | 65536 | 65536 | 65536 | 0 | 0 | 0 | True | nearest |  |  |
| accuracy |  |  |  | kahan | bfloat | uniform | 65536 | 16 | 256 | 32768 | 3.277e+04 | 3.277e+04 | 3.772 | 0.0001151 | 0.01473 | True | nearest |  |  |
| accuracy |  |  |  | kahan | bfloat | mixed | 65536 | 16 | 256 | 2064 | 2066 | 2066 | 2.003 | 0.0009697 | 0.1252 | True | nearest |  |  |
| accuracy |  |  |  | kahan | bfloat | mixed-asc | 65536 | 16 | 256 | 2048 | 2058 | 2058 | 9.703 | 0.004715 | 0.6064 | True | nearest |  |  |
| accuracy |  |  |  | kahan | bfloat | mixed-desc | 65536 | 16 | 256 | 2048 | 2055 | 2055 | 6.712 | 0.003267 | 0.4195 | True | nearest |  |  |
| accuracy |  |  |  | kahan | bfloat | cancel | 65536 | 16 | 256 | -12.62 | -12.66 | 2043 | 0.03873 | 1.896e-05 | 0.6197 | True | nearest |  |  |
| accuracy |  |  |  | kahan | bfloat | softmax | 65536 | 16 | 256 | 7264 | 7240 | 7240 | 24.33 | 0.003361 | 0.7604 | True | nearest |  |  |
| accuracy |  |  |  | kahan | bfloat | big-small | 65536 | 16 | 256 | 256 | 257 | 257 | 0.9961 | 0.003876 | 0.498 | True | nearest |  |  |
| coarse |  |  |  | tree | float |  | 4194304 | 1 | 256 |  |  |  |  | 3.944e-08 | 0.3309 |  |  | 0.0001414 | 118.6 |
| coarse |  |  |  | kahan | float |  | 4194304 | 1 | 256 |  |  |  |  | 7.974e-08 | 0.6691 |  |  | 0.0001472 | 114 |
| coarse |  |  |  | tree | float |  | 4194304 | 4 | 256 |  |  |  |  | 7.974e-08 | 0.6691 |  |  | 5.113e-05 | 328.2 |
| coarse |  |  |  | kahan | float |  | 4194304 | 4 | 256 |  |  |  |  | 7.974e-08 | 0.6691 |  |  | 5.45e-05 | 307.8 |
| coarse |  |  |  | tree | float |  | 4194304 | 16 | 256 |  |  |  |  | 3.944e-08 | 0.3309 |  |  | 3.975e-05 | 422.1 |
| coarse |  |  |  | kahan | float |  | 4194304 | 16 | 256 |  |  |  |  | 1.586e-07 | 1.331 |  |  | 4.062e-05 | 413 |
| coarse |  |  |  | tree | float |  | 4194304 | 64 | 256 |  |  |  |  | 7.974e-08 | 0.6691 |  |  | 3.721e-05 | 450.9 |
| coarse |  |  |  | kahan | float |  | 4194304 | 64 | 256 |  |  |  |  | 7.974e-08 | 0.6691 |  |  | 3.75e-05 | 447.4 |
| coarse |  |  |  | tree | float |  | 4194304 | 256 | 256 |  |  |  |  | 3.944e-08 | 0.3309 |  |  | 7.867e-05 | 213.3 |
| coarse |  |  |  | kahan | float |  | 4194304 | 256 | 256 |  |  |  |  | 7.974e-08 | 0.6691 |  |  | 8.05e-05 | 208.4 |
| coarse |  |  |  | tree | float |  | 4194304 | 1024 | 256 |  |  |  |  | 7.974e-08 | 0.6691 |  |  | 0.0002829 | 59.31 |
| coarse |  |  |  | kahan | float |  | 4194304 | 1024 | 256 |  |  |  |  | 7.974e-08 | 0.6691 |  |  | 0.0002872 | 58.41 |
| coarse |  |  |  | tree | float |  | 4194304 | 4096 | 256 |  |  |  |  | 3.944e-08 | 0.3309 |  |  | 0.001084 | 15.47 |
| coarse |  |  |  | kahan | float |  | 4194304 | 4096 | 256 |  |  |  |  | 7.974e-08 | 0.6691 |  |  | 0.001102 | 15.23 |
| time |  |  |  | serial | float |  | 1048576 | 1 | 1 |  |  |  |  |  |  |  |  | 0.07059 | 0.05941 |
| time |  |  |  | tree | float |  | 16777216 | 16 | 256 |  |  |  |  |  |  |  |  | 0.0002393 | 280.5 |
| time |  |  |  | kahan | float |  | 16777216 | 16 | 256 |  |  |  |  |  |  |  |  | 0.0002411 | 278.3 |
| time |  |  |  | serial | half |  | 1048576 | 1 | 1 |  |  |  |  |  |  |  |  | 0.07048 | 0.05951 |
| time |  |  |  | tree | half |  | 16777216 | 16 | 256 |  |  |  |  |  |  |  |  | 0.0002359 | 284.5 |
| time |  |  |  | kahan | half |  | 16777216 | 16 | 256 |  |  |  |  |  |  |  |  | 0.000238 | 282 |
| time |  |  |  | serial | bfloat |  | 1048576 | 1 | 1 |  |  |  |  |  |  |  |  | 0.07185 | 0.05838 |
| time |  |  |  | tree | bfloat |  | 16777216 | 16 | 256 |  |  |  |  |  |  |  |  | 0.0002394 | 280.4 |
| time |  |  |  | kahan | bfloat |  | 16777216 | 16 | 256 |  |  |  |  |  |  |  |  | 0.0002346 | 286 |

### appC/lab1_feeding, solution, metal, TILE_WIDTH=16

| name | shared_same | nocopy_same | path | workload | rows | cols | count | passes | table_bytes | touched_bytes | seconds | gpu_copy_s | gpu_kernel_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| addresses | False | False |  |  |  |  |  |  |  |  |  |  |  |
| feed |  |  | private | dense1 | 262144 | 256 | 262144 | 1 | 268435456 | 268435456 | 0.06418 | 0.004145 | 0.002999 |
| feed |  |  | nocopy | dense1 | 262144 | 256 | 262144 | 1 | 268435456 | 268435456 | 0.01936 | 0 | 0.003052 |
| feed |  |  | shared | dense1 | 262144 | 256 | 262144 | 1 | 268435456 | 268435456 | 0.04544 | 0 | 0.003276 |
| feed |  |  | private | dense8 | 262144 | 256 | 262144 | 8 | 268435456 | 2.147e+09 | 0.08342 | 0.004132 | 0.02343 |
| feed |  |  | nocopy | dense8 | 262144 | 256 | 262144 | 8 | 268435456 | 2.147e+09 | 0.03925 | 0 | 0.0235 |
| feed |  |  | shared | dense8 | 262144 | 256 | 262144 | 8 | 268435456 | 2.147e+09 | 0.06586 | 0 | 0.02337 |
| feed |  |  | private | sparse1 | 262144 | 256 | 4096 | 1 | 268435456 | 4194304 | 0.02724 | 0.002105 | 4.017e-05 |
| feed |  |  | nocopy | sparse1 | 262144 | 256 | 4096 | 1 | 268435456 | 4194304 | 0.007541 | 0 | 5.354e-05 |
| feed |  |  | shared | sparse1 | 262144 | 256 | 4096 | 1 | 268435456 | 4194304 | 0.02011 | 0 | 5.412e-05 |

### ch02/lab1_vecadd, solution, metal, TILE_WIDTH=16

| name | n | cpu_s | offload_s | speedup_vs_cpu |
|---|---|---|---|---|
| end_to_end | 65536 | 6.75e-06 | 0.000718 | 0.009401 |
| end_to_end | 1048576 | 0.0001655 | 0.001813 | 0.09129 |
| end_to_end | 16777216 | 0.001809 | 0.01565 | 0.1157 |

### ch02/lab2_timing, solution, metal, TILE_WIDTH=16

| name | n | wrap_s | encode_s | wait_s | readback_s | gpu_s | total_s | seconds | bytes_copied |
|---|---|---|---|---|---|---|---|---|---|
| phases | 65536 | 5.546e-05 | 5.042e-06 | 0.0002222 | 1.604e-05 | 7.667e-06 | 0.0002987 |  |  |
| resident | 65536 |  |  |  |  | 7.333e-06 |  | 0.0001879 | 0 |
| cpu | 65536 |  |  |  |  |  |  | 6.792e-06 |  |
| phases | 1048576 | 0.0005234 | 4.75e-06 | 0.0003892 | 0.0002229 | 3.688e-05 | 0.00114 |  |  |
| resident | 1048576 |  |  |  |  | 3.05e-05 |  | 0.0002415 | 0 |
| cpu | 1048576 |  |  |  |  |  |  | 0.0001055 |  |
| phases | 16777216 | 0.007834 | 2.413e-05 | 0.003477 | 0.004488 | 0.0008277 | 0.01582 |  |  |
| resident | 16777216 |  |  |  |  | 0.0007285 |  | 0.001405 | 0 |
| cpu | 16777216 |  |  |  |  |  |  | 0.001811 |  |

### ch02/lab3_elementwise, solution, metal, TILE_WIDTH=16

| name | kernel | elements | seconds | gb_per_s | rows | d | n | resident_s | roundtrips_s | ratio | roundtrip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| op | addBias | 67108864 | 0.00201 | 400.7 |  |  |  |  |  |  |  |
| op | scaleShift | 67108864 | 0.001997 | 268.8 |  |  |  |  |  |  |  |
| op | addResidual | 67108864 | 0.002895 | 278.2 |  |  |  |  |  |  |  |
| chain |  |  |  |  | 16384 | 4096 | 67108864 | 0.06642 | 0.1356 | 2.042 | host arrays copied into shared buffers, results copied out |

### ch03/lab1_grayscale, solution, metal, TILE_WIDTH=16

| name | block_w | block_h | width | height | seconds | mpix_per_s |
|---|---|---|---|---|---|---|
| grayscale | 8 | 8 | 16384 | 16384 | 0.006124 | 4.383e+04 |
| grayscale | 16 | 16 | 16384 | 16384 | 0.005032 | 5.335e+04 |
| grayscale | 32 | 32 | 16384 | 16384 | 0.00445 | 6.032e+04 |
| grayscale | 256 | 1 | 16384 | 16384 | 0.003993 | 6.723e+04 |
| grayscale | 1 | 256 | 16384 | 16384 | 0.01373 | 1.955e+04 |

### ch03/lab2_blur, solution, metal, TILE_WIDTH=16

| name | radius | width | height | seconds | mpix_per_s |
|---|---|---|---|---|---|
| blur | 1 | 16384 | 16384 | 0.009992 | 2.686e+04 |
| blur | 2 | 16384 | 16384 | 0.02461 | 1.091e+04 |
| blur | 4 | 16384 | 16384 | 0.07687 | 3492 |
| blur | 8 | 16384 | 16384 | 0.2657 | 1010 |

### ch03/lab3_naive_gemm, solution, metal, TILE_WIDTH=16

| name | width | seconds | gflops | time_vs_half_width | batch | k | l | block | rows_per_s |
|---|---|---|---|---|---|---|---|---|---|
| square | 256 | 8.242e-05 | 407.1 |  |  |  |  |  |  |
| square | 512 | 0.000512 | 524.2 | 6.213 |  |  |  |  |  |
| square | 1024 | 0.002556 | 840.3 | 4.991 |  |  |  |  |  |
| linear_layer |  | 0.0002586 | 32.44 |  | 1 | 1024 | 4096 | 16 | 3867 |
| linear_layer |  | 0.0002539 | 528.7 |  | 16 | 1024 | 4096 | 16 | 6.302e+04 |
| linear_layer |  | 0.002476 | 867.3 |  | 256 | 1024 | 4096 | 16 | 1.034e+05 |

### ch03/lab4_granularity, solution, metal, TILE_WIDTH=16

| name | mode | width | threads | launched | seconds | gflops | time_vs_element |
|---|---|---|---|---|---|---|---|
| granularity | element | 1024 | 1048576 | 1048576 | 0.002427 | 884.7 | 1 |
| granularity | row | 1024 | 1024 | 1024 | 0.1647 | 13.04 | 67.85 |
| granularity | col | 1024 | 1024 | 1024 | 0.158 | 13.6 | 65.07 |

### ch04/lab1_launchconfig, solution, metal, TILE_WIDTH=16

| name | device | maxThreadsPerThreadgroup_width | maxThreadsPerThreadgroup_height | maxThreadsPerThreadgroup_depth | maxThreadgroupMemoryLength | hasUnifiedMemory | threadExecutionWidth | maxTotalThreadsPerThreadgroup | probe_threadExecutionWidth | probe_maxTotalThreadsPerThreadgroup | config | batch | in | out | block_x | block_y | grid_x | grid_y | launches_per_run | seconds | gflops | derived_over_fixed | threadgroup_x | threadgroup_y | width_in_kernel | simdgroups_per_threadgroup | consecutive_runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| queried | Apple M3 Max | 1024 | 1024 | 1024 | 32768 | True | 32 | 1024 | 32 | 1024 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| linear |  |  |  |  |  |  |  |  |  |  | derived | 1 | 1024 | 4096 | 32 | 1 | 128 | 1 | 256 | 0.0002685 | 31.24 |  |  |  |  |  |  |
| linear |  |  |  |  |  |  |  |  |  |  | fixed16 | 1 | 1024 | 4096 | 16 | 16 | 256 | 1 | 256 | 0.0002531 | 33.14 |  |  |  |  |  |  |
| speedup |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  | 0.9428 |  |  |  |  |  |
| linear |  |  |  |  |  |  |  |  |  |  | derived | 64 | 1024 | 4096 | 32 | 32 | 128 | 2 | 64 | 0.001064 | 504.4 |  |  |  |  |  |  |
| linear |  |  |  |  |  |  |  |  |  |  | fixed16 | 64 | 1024 | 4096 | 16 | 16 | 256 | 4 | 64 | 0.0008527 | 629.6 |  |  |  |  |  |  |
| speedup |  |  |  |  |  |  |  |  |  |  |  | 64 |  |  |  |  |  |  |  |  |  | 0.8011 |  |  |  |  |  |
| probe |  |  |  |  |  |  | 32 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 32 | 32 | 32 | 32 | True |
| probe |  |  |  |  |  |  | 32 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1024 | 1 | 32 | 32 | True |
| probe |  |  |  |  |  |  | 32 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 16 | 16 | 32 | 8 | True |

### ch04/lab2_divergence, solution, metal, TILE_WIDTH=16

| name | pattern | width | threads_per_block | n | iters | fraction_b | seconds | sorted_seconds | ratio_to_sorted | ratio_to_branch | order | mean_trips | total_steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| branch | all A | 32 | 256 | 1048576 | 65536 | 0 | 0.0693 | 0.0693 | 1 |  |  |  |  |
| select | all A | 32 | 256 | 1048576 | 65536 | 0 | 0.1479 | 0.0693 | 2.134 | 2.134 |  |  |  |
| branch | all B | 32 | 256 | 1048576 | 65536 | 1 | 0.0705 | 0.0705 | 1 |  |  |  |  |
| select | all B | 32 | 256 | 1048576 | 65536 | 1 | 0.1472 | 0.0705 | 2.088 | 2.088 |  |  |  |
| branch | first half of the array A, second half B | 32 | 256 | 1048576 | 65536 | 0.5 | 0.0698 | 0.0698 | 1 |  |  |  |  |
| select | first half of the array A, second half B | 32 | 256 | 1048576 | 65536 | 0.5 | 0.1454 | 0.0698 | 2.082 | 2.082 |  |  |  |
| branch | alternating SIMD-groups | 32 | 256 | 1048576 | 65536 | 0.5 | 0.07387 | 0.0698 | 1.058 |  |  |  |  |
| select | alternating SIMD-groups | 32 | 256 | 1048576 | 65536 | 0.5 | 0.1452 | 0.0698 | 2.08 | 1.965 |  |  |  |
| branch | half of each SIMD-group's lanes | 32 | 256 | 1048576 | 65536 | 0.5 | 0.1445 | 0.0698 | 2.071 |  |  |  |  |
| select | half of each SIMD-group's lanes | 32 | 256 | 1048576 | 65536 | 0.5 | 0.1482 | 0.0698 | 2.124 | 1.026 |  |  |  |
| branch | alternating lanes | 32 | 256 | 1048576 | 65536 | 0.5 | 0.1444 | 0.0698 | 2.068 |  |  |  |  |
| select | alternating lanes | 32 | 256 | 1048576 | 65536 | 0.5 | 0.1455 | 0.0698 | 2.085 | 1.008 |  |  |  |
| branch | one lane per SIMD-group | 32 | 256 | 1048576 | 65536 | 0.03125 | 0.1444 | 0.06983 | 2.067 |  |  |  |  |
| select | one lane per SIMD-group | 32 | 256 | 1048576 | 65536 | 0.03125 | 0.1472 | 0.06983 | 2.108 | 1.02 |  |  |  |
| branch | random half (seeded) | 32 | 256 | 1048576 | 65536 | 0.5 | 0.1446 | 0.06978 | 2.072 |  |  |  |  |
| select | random half (seeded) | 32 | 256 | 1048576 | 65536 | 0.5 | 0.1473 | 0.06978 | 2.112 | 1.019 |  |  |  |
| loop |  | 32 | 256 | 1048576 |  |  | 0.07245 |  | 1 |  | sorted | 65536 | 6.872e+10 |
| loop |  | 32 | 256 | 1048576 |  |  | 0.1419 |  | 1.959 |  | shuffled | 65536 | 6.872e+10 |
| loop |  | 32 | 256 | 1048576 |  |  | 0.07283 |  | 1.005 |  | all equal to the mean | 65536 | 6.872e+10 |

### ch04/lab4_blocksize, solution, metal, TILE_WIDTH=16

| name | outputs | lookups_per_output | table_floats | table_bytes | width | max_threads_per_block | launches_per_run | threads_per_block | multiple | seconds | median_seconds | lookups_per_s | fastest_threads_per_block | fastest_seconds | slowest_threads_per_block | slowest_seconds | slowest_over_fastest | largest_median_over_best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| problem | 1000000 | 16 | 64000000 | 256000000 | 32 | 1024 | 16 |  |  |  |  |  |  |  |  |  |  |  |
| sweep |  |  |  |  | 32 |  |  | 32 | 1 | 0.004435 | 0.004498 | 3.608e+09 |  |  |  |  |  |  |
| sweep |  |  |  |  | 32 |  |  | 64 | 2 | 0.004433 | 0.004498 | 3.609e+09 |  |  |  |  |  |  |
| sweep |  |  |  |  | 32 |  |  | 96 | 3 | 0.004403 | 0.004455 | 3.634e+09 |  |  |  |  |  |  |
| sweep |  |  |  |  | 32 |  |  | 128 | 4 | 0.004457 | 0.004466 | 3.59e+09 |  |  |  |  |  |  |
| sweep |  |  |  |  | 32 |  |  | 192 | 6 | 0.004382 | 0.004467 | 3.651e+09 |  |  |  |  |  |  |
| sweep |  |  |  |  | 32 |  |  | 256 | 8 | 0.004474 | 0.004493 | 3.576e+09 |  |  |  |  |  |  |
| sweep |  |  |  |  | 32 |  |  | 384 | 12 | 0.00447 | 0.004517 | 3.579e+09 |  |  |  |  |  |  |
| sweep |  |  |  |  | 32 |  |  | 512 | 16 | 0.004394 | 0.004481 | 3.641e+09 |  |  |  |  |  |  |
| sweep |  |  |  |  | 32 |  |  | 768 | 24 | 0.004423 | 0.004489 | 3.618e+09 |  |  |  |  |  |  |
| sweep |  |  |  |  | 32 |  |  | 1024 | 32 | 0.004483 | 0.004535 | 3.569e+09 |  |  |  |  |  |  |
| summary |  |  |  |  |  |  |  |  |  |  |  |  | 192 | 0.004382 | 1024 | 0.004483 | 1.023 | 1.02 |

### ch05/lab1_roofline, solution, metal, TILE_WIDTH=16

| name | bytes | seconds | gb_per_s | flop | gflops | width | bandwidth_gb_per_s | compute_gflops | threshold_flop_per_byte | naive_model_ceiling_gflops | naive_gflops | naive_width |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bandwidth | 1.5e+09 | 0.005564 | 269.6 |  |  |  |  |  |  |  |  |  |
| compute |  | 0.06876 |  | 1.374e+11 | 1999 |  |  |  |  |  |  |  |
| naive_matmul |  | 0.1613 |  |  | 852.3 | 4096 |  |  |  |  |  |  |
| roofline |  |  |  |  |  |  | 269.6 | 1999 | 7.414 | 67.4 | 852.3 | 4096 |

### ch05/lab2_tiled, solution, metal, TILE_WIDTH=16

| name | width | seconds | gflops | tile | speedup_vs_naive |
|---|---|---|---|---|---|
| naive | 4096 | 0.159 | 864.2 |  |  |
| tiled | 4096 | 0.3075 | 447 | 4 | 0.5172 |
| tiled | 4096 | 0.08397 | 1637 | 8 | 1.894 |
| tiled | 4096 | 0.09082 | 1513 | 16 | 1.751 |
| tiled | 4096 | 0.08402 | 1636 | 32 | 1.893 |

### ch05/lab3_gemm, solution, metal, TILE_WIDTH=16

| name | batch | k | l | tile | seconds | gflops | rows_per_s |
|---|---|---|---|---|---|---|---|
| linear_layer | 1 | 1024 | 4096 | 16 | 0.0002223 | 37.74 | 4499 |
| linear_layer | 4 | 1024 | 4096 | 16 | 0.0002201 | 152.4 | 1.817e+04 |
| linear_layer | 16 | 1024 | 4096 | 16 | 0.0002259 | 594.2 | 7.084e+04 |
| linear_layer | 64 | 1024 | 4096 | 16 | 0.0004107 | 1307 | 1.558e+05 |
| linear_layer | 256 | 1024 | 4096 | 16 | 0.001525 | 1408 | 1.678e+05 |
| linear_layer | 1024 | 1024 | 4096 | 16 | 0.006005 | 1431 | 1.705e+05 |

### ch05/lab5_dynamic, solution, metal, TILE_WIDTH=16

| name | tile | width | seconds | gflops | shared_bytes_per_block |
|---|---|---|---|---|---|
| tiled_dynamic | 4 | 512 | 0.001329 | 202 | 128 |
| tiled_dynamic | 8 | 512 | 0.0005862 | 457.9 | 512 |
| tiled_dynamic | 16 | 512 | 0.0004931 | 544.4 | 2048 |
| tiled_dynamic | 32 | 512 | 0.0004674 | 574.3 | 8192 |
| tiled_dynamic | 4 | 1024 | 0.007933 | 270.7 | 128 |
| tiled_dynamic | 8 | 1024 | 0.003346 | 641.8 | 512 |
| tiled_dynamic | 16 | 1024 | 0.002852 | 752.9 | 2048 |
| tiled_dynamic | 32 | 1024 | 0.002824 | 760.5 | 8192 |
| tiled_dynamic | 4 | 2048 | 0.06312 | 272.2 | 128 |
| tiled_dynamic | 8 | 2048 | 0.02669 | 643.6 | 512 |
| tiled_dynamic | 16 | 2048 | 0.02258 | 761 | 2048 |
| tiled_dynamic | 32 | 2048 | 0.02276 | 754.9 | 8192 |
| tiled_dynamic | 4 | 4096 | 0.522 | 263.3 | 128 |
| tiled_dynamic | 8 | 4096 | 0.2163 | 635.4 | 512 |
| tiled_dynamic | 16 | 4096 | 0.1824 | 753.6 | 2048 |
| tiled_dynamic | 32 | 4096 | 0.1937 | 709.5 | 8192 |

### ch06/lab1_coalescing, solution, metal, TILE_WIDTH=16

| name | n | seconds | gb_per_s | stride | relative_to_stride1 | kernel | rows | cols | width |
|---|---|---|---|---|---|---|---|---|---|
| copy | 33554432 | 0.0009706 | 276.6 |  |  |  |  |  |  |
| strided | 33554432 | 0.001004 | 267.5 | 1 | 1 |  |  |  |  |
| strided | 16777216 | 0.0007403 | 181.3 | 2 | 0.6778 |  |  |  |  |
| strided | 8388608 | 0.0006115 | 109.7 | 4 | 0.4103 |  |  |  |  |
| strided | 4194304 | 0.0005304 | 63.27 | 8 | 0.2365 |  |  |  |  |
| strided | 2097152 | 0.0004838 | 34.68 | 16 | 0.1296 |  |  |  |  |
| strided | 1048576 | 0.0002398 | 34.98 | 32 | 0.1308 |  |  |  |  |
| strided | 524288 | 0.000121 | 34.65 | 64 | 0.1295 |  |  |  |  |
| transpose |  | 0.0005012 | 267.8 |  |  | read_rows | 4096 | 4096 |  |
| transpose |  | 0.0004945 | 271.4 |  |  | write_rows | 4096 | 4096 |  |
| simd_width |  |  |  |  |  |  |  |  | 32 |

### ch06/lab2_vector, solution, metal, TILE_WIDTH=16

| name | n | seconds | gb_per_s | speedup_vs_scalar |
|---|---|---|---|---|
| vecadd | 67108864 | 0.002981 | 270.1 | 1 |
| vecadd | 67108864 | 0.003276 | 245.8 | 0.9101 |
| vecadd | 67108864 | 0.002994 | 269 | 0.9956 |

### ch06/lab3_banks, solution, metal, TILE_WIDTH=16

| name | width | stride | lanes | iters | seconds | ns_per_access | relative_to_stride1 | n | gb_per_s | tile | pitch | rows | cols |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| simd_width | 32 |  |  |  |  |  |  |  |  |  |  |  |  |
| bank_probe |  | 1 | 32 | 262144 | 0.01482 | 28.27 | 1 |  |  |  |  |  |  |
| bank_probe |  | 2 | 32 | 262144 | 0.0152 | 28.99 | 1.026 |  |  |  |  |  |  |
| bank_probe |  | 4 | 32 | 262144 | 0.01615 | 30.8 | 1.09 |  |  |  |  |  |  |
| bank_probe |  | 7 | 32 | 262144 | 0.01634 | 31.17 | 1.103 |  |  |  |  |  |  |
| bank_probe |  | 8 | 32 | 262144 | 0.01634 | 31.17 | 1.103 |  |  |  |  |  |  |
| bank_probe |  | 12 | 32 | 262144 | 0.01634 | 31.17 | 1.103 |  |  |  |  |  |  |
| bank_probe |  | 16 | 32 | 262144 | 0.01767 | 33.7 | 1.192 |  |  |  |  |  |  |
| bank_probe |  | 24 | 32 | 262144 | 0.01672 | 31.89 | 1.128 |  |  |  |  |  |  |
| bank_probe |  | 31 | 32 | 262144 | 0.01634 | 31.17 | 1.103 |  |  |  |  |  |  |
| bank_probe |  | 32 | 32 | 262144 | 0.0203 | 38.73 | 1.37 |  |  |  |  |  |  |
| bank_probe |  | 33 | 32 | 262144 | 0.01596 | 30.44 | 1.077 |  |  |  |  |  |  |
| copy |  |  |  |  | 0.002017 |  |  | 67108864 | 266.2 |  |  |  |  |
| transpose |  |  |  |  | 0.002056 |  |  |  | 261.2 | 16 | 16 | 8192 | 8192 |
| transpose |  |  |  |  | 0.002056 |  |  |  | 261.1 | 16 | 17 | 8192 | 8192 |
| transpose |  |  |  |  | 0.00205 |  |  |  | 261.9 | 32 | 32 | 8192 | 8192 |
| transpose |  |  |  |  | 0.002049 |  |  |  | 262 | 32 | 33 | 8192 | 8192 |

### ch06/lab4_linear, solution, metal, TILE_WIDTH=16

| name | kernel | tile | batch | in | out | seconds | gflops | speedup_vs_baseline |
|---|---|---|---|---|---|---|---|---|
| linear | baseline | 16 | 1 | 1024 | 4096 | 0.0001871 | 44.83 |  |
| linear | tuned | 16 | 1 | 1024 | 4096 | 0.0001904 | 44.06 | 0.9829 |
| linear | baseline | 16 | 16 | 1024 | 4096 | 0.0001887 | 711.1 |  |
| linear | tuned | 16 | 16 | 1024 | 4096 | 0.000196 | 684.8 | 0.963 |
| linear | baseline | 16 | 64 | 1024 | 4096 | 0.000608 | 883 |  |
| linear | tuned | 16 | 64 | 1024 | 4096 | 0.0006328 | 848.4 | 0.9608 |
| linear | baseline | 16 | 256 | 1024 | 4096 | 0.001659 | 1295 |  |
| linear | tuned | 16 | 256 | 1024 | 4096 | 0.001662 | 1292 | 0.9979 |
| linear | baseline | 16 | 1024 | 1024 | 4096 | 0.005981 | 1436 |  |
| linear | tuned | 16 | 1024 | 1024 | 4096 | 0.006298 | 1364 | 0.9497 |
| linear | baseline | 32 | 1 | 1024 | 4096 | 0.0002391 | 35.09 |  |
| linear | tuned | 32 | 1 | 1024 | 4096 | 0.0002538 | 33.06 | 0.9422 |
| linear | baseline | 32 | 16 | 1024 | 4096 | 0.00024 | 559.3 |  |
| linear | tuned | 32 | 16 | 1024 | 4096 | 0.0002564 | 523.4 | 0.9358 |
| linear | baseline | 32 | 64 | 1024 | 4096 | 0.0004477 | 1199 |  |
| linear | tuned | 32 | 64 | 1024 | 4096 | 0.000494 | 1087 | 0.9061 |
| linear | baseline | 32 | 256 | 1024 | 4096 | 0.001473 | 1458 |  |
| linear | tuned | 32 | 256 | 1024 | 4096 | 0.001677 | 1281 | 0.8783 |
| linear | baseline | 32 | 1024 | 1024 | 4096 | 0.005858 | 1466 |  |
| linear | tuned | 32 | 1024 | 1024 | 4096 | 0.006699 | 1282 | 0.8745 |

### ch06/lab5_doublebuf, solution, metal, TILE_WIDTH=16

| name | kernel | tile | width | seconds | gflops | speedup_vs_two_barriers |
|---|---|---|---|---|---|---|
| matmul | two_barriers | 16 | 4096 | 0.1632 | 842.1 | 1 |
| matmul | double_buffered | 16 | 4096 | 0.1841 | 746.6 | 0.8866 |
| matmul | two_barriers | 32 | 4096 | 0.1709 | 804.1 | 1 |
| matmul | double_buffered | 32 | 4096 | 0.1643 | 836.4 | 1.04 |

### ch07/lab1_constconv, solution, metal, TILE_WIDTH=16

| name | kernel | threadExecutionWidth | maxTotalThreadsPerThreadgroup | radius | m | width | height | seconds | gflops | speedup_vs_basic | model_intensity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pipeline | conv_const | 32 | 1024 |  |  |  |  |  |  |  |  |
| basic |  |  |  | 1 | 3 | 8192 | 8192 | 0.003099 | 389.8 | 1 | 0.25 |
| const |  |  |  | 1 | 3 | 8192 | 8192 | 0.002653 | 455.4 | 1.168 | 0.5 |
| byvalue |  |  |  | 1 | 3 | 8192 | 8192 | 0.002652 | 455.5 | 1.169 | 0.5 |
| basic |  |  |  | 2 | 5 | 8192 | 8192 | 0.008529 | 393.4 | 1 | 0.25 |
| const |  |  |  | 2 | 5 | 8192 | 8192 | 0.009465 | 354.5 | 0.9011 | 0.5 |
| byvalue |  |  |  | 2 | 5 | 8192 | 8192 | 0.01312 | 255.8 | 0.6501 | 0.5 |
| basic |  |  |  | 3 | 7 | 8192 | 8192 | 0.0263 | 250.1 | 1 | 0.25 |
| const |  |  |  | 3 | 7 | 8192 | 8192 | 0.02532 | 259.8 | 1.039 | 0.5 |
| byvalue |  |  |  | 3 | 7 | 8192 | 8192 | 0.02824 | 232.9 | 0.9312 | 0.5 |
| basic |  |  |  | 5 | 11 | 8192 | 8192 | 0.04483 | 362.3 | 1 | 0.25 |
| const |  |  |  | 5 | 11 | 8192 | 8192 | 0.05718 | 284 | 0.784 | 0.5 |
| byvalue |  |  |  | 5 | 11 | 8192 | 8192 | 0.05696 | 285.1 | 0.787 | 0.5 |
| basic |  |  |  | 7 | 15 | 8192 | 8192 | 0.08584 | 351.8 | 1 | 0.25 |
| const |  |  |  | 7 | 15 | 8192 | 8192 | 0.0933 | 323.7 | 0.9201 | 0.5 |
| byvalue |  |  |  | 7 | 15 | 8192 | 8192 | 0.09289 | 325.1 | 0.9241 | 0.5 |

### ch07/lab2_tiledconv, solution, metal, TILE_WIDTH=32

| name | tile | out_tile | radius | m | width | height | seconds | gflops | speedup_vs_const | model_intensity | model_traffic_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| const | 16 | 16 | 1 | 3 | 8192 | 8192 | 0.003045 | 396.7 | 1 |  |  |
| tiled_halo | 8 | 6 | 1 | 3 | 8192 | 8192 | 0.008709 | 138.7 | 0.3497 | 1.62 | 3.24 |
| tiled_cached | 8 | 8 | 1 | 3 | 8192 | 8192 | 0.007007 | 172.4 | 0.4346 |  |  |
| tiled_halo | 16 | 14 | 1 | 3 | 8192 | 8192 | 0.003896 | 310.1 | 0.7817 | 1.951 | 3.903 |
| tiled_cached | 16 | 16 | 1 | 3 | 8192 | 8192 | 0.006407 | 188.5 | 0.4753 |  |  |
| tiled_halo | 32 | 30 | 1 | 3 | 8192 | 8192 | 0.0029 | 416.5 | 1.05 | 2.105 | 4.21 |
| tiled_cached | 32 | 32 | 1 | 3 | 8192 | 8192 | 0.007837 | 154.1 | 0.3886 |  |  |
| const | 16 | 16 | 2 | 5 | 8192 | 8192 | 0.008351 | 401.8 | 1 |  |  |
| tiled_halo | 8 | 4 | 2 | 5 | 8192 | 8192 | 0.01735 | 193.4 | 0.4814 | 2.5 | 5 |
| tiled_cached | 8 | 8 | 2 | 5 | 8192 | 8192 | 0.01793 | 187.2 | 0.4658 |  |  |
| tiled_halo | 16 | 12 | 2 | 5 | 8192 | 8192 | 0.004838 | 693.5 | 1.726 | 4.5 | 9 |
| tiled_cached | 16 | 16 | 2 | 5 | 8192 | 8192 | 0.01866 | 179.9 | 0.4476 |  |  |
| tiled_halo | 32 | 28 | 2 | 5 | 8192 | 8192 | 0.004223 | 794.5 | 1.977 | 5.42 | 10.84 |
| tiled_cached | 32 | 32 | 2 | 5 | 8192 | 8192 | 0.02026 | 165.6 | 0.4122 |  |  |
| const | 16 | 16 | 3 | 7 | 8192 | 8192 | 0.01602 | 410.4 | 1 |  |  |
| tiled_halo | 8 | 2 | 3 | 7 | 8192 | 8192 | 0.08281 | 79.42 | 0.1935 | 1.441 | 2.882 |
| tiled_cached | 8 | 8 | 3 | 7 | 8192 | 8192 | 0.03236 | 203.3 | 0.4953 |  |  |
| tiled_halo | 16 | 10 | 3 | 7 | 8192 | 8192 | 0.01417 | 464.2 | 1.131 | 6.882 | 13.76 |
| tiled_cached | 16 | 16 | 3 | 7 | 8192 | 8192 | 0.03264 | 201.5 | 0.491 |  |  |
| tiled_halo | 32 | 26 | 3 | 7 | 8192 | 8192 | 0.01136 | 579 | 1.411 | 9.742 | 19.48 |
| tiled_cached | 32 | 32 | 3 | 7 | 8192 | 8192 | 0.02999 | 219.3 | 0.5343 |  |  |
| const | 16 | 16 | 5 | 11 | 8192 | 8192 | 0.03503 | 463.7 | 1 |  |  |
| tiled_halo | 16 | 6 | 5 | 11 | 8192 | 8192 | 0.06021 | 269.7 | 0.5817 | 7.459 | 14.92 |
| tiled_cached | 16 | 16 | 5 | 11 | 8192 | 8192 | 0.07645 | 212.4 | 0.4581 |  |  |
| tiled_halo | 32 | 22 | 5 | 11 | 8192 | 8192 | 0.03547 | 457.9 | 0.9875 | 19.42 | 38.84 |
| tiled_cached | 32 | 32 | 5 | 11 | 8192 | 8192 | 0.0652 | 249.1 | 0.5372 |  |  |
| const | 16 | 16 | 7 | 15 | 8192 | 8192 | 0.06272 | 481.5 | 1 |  |  |
| tiled_halo | 16 | 2 | 7 | 15 | 8192 | 8192 | 0.306 | 98.68 | 0.205 | 1.731 | 3.462 |
| tiled_cached | 16 | 16 | 7 | 15 | 8192 | 8192 | 0.1314 | 229.9 | 0.4774 |  |  |
| tiled_halo | 32 | 18 | 7 | 15 | 8192 | 8192 | 0.06234 | 484.4 | 1.006 | 27.04 | 54.08 |
| tiled_cached | 32 | 32 | 7 | 15 | 8192 | 8192 | 0.1278 | 236.4 | 0.491 |  |  |

### ch08/lab1_stencil3d, solution, metal, TILE_WIDTH=32

| name | n | seconds | gpts | gbps_compulsory | kernel | tile | max_threads | speedup_vs_basic | depth | slabs | speedup_vs_planes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| basic | 512 | 0.004876 | 27.2 | 217.6 |  |  |  |  |  |  |  |
| limits |  |  |  |  | cube | 8 | 1024 |  |  |  |  |
| cube | 512 | 0.007086 | 18.72 | 149.8 |  | 8 |  | 0.6882 |  |  |  |
| limits |  |  |  |  | regtile | 8 | 1024 |  |  |  |  |
| planes | 512 | 0.005268 | 25.18 | 201.5 |  | 8 |  | 0.9257 | 6 |  |  |
| regtile | 512 | 0.005827 | 22.76 | 182.1 |  | 8 |  | 0.8368 | 6 | 86 | 0.9039 |
| regtile | 512 | 0.005913 | 22.43 | 179.5 |  | 8 |  | 0.8246 | 8 | 64 | 0.8908 |
| regtile | 512 | 0.005984 | 22.17 | 177.3 |  | 8 |  | 0.8148 | 16 | 32 | 0.8802 |
| regtile | 512 | 0.006878 | 19.29 | 154.3 |  | 8 |  | 0.7089 | 64 | 8 | 0.7658 |
| regtile | 512 | 0.008969 | 14.79 | 118.3 |  | 8 |  | 0.5437 | 512 | 1 | 0.5873 |
| limits |  |  |  |  | regtile | 16 | 1024 |  |  |  |  |
| planes | 512 | 0.00523 | 25.36 | 202.9 |  | 16 |  | 0.9324 | 14 |  |  |
| regtile | 512 | 0.005375 | 24.68 | 197.4 |  | 16 |  | 0.9072 | 14 | 37 | 0.9729 |
| regtile | 512 | 0.005362 | 24.74 | 197.9 |  | 16 |  | 0.9094 | 8 | 64 | 0.9754 |
| regtile | 512 | 0.005378 | 24.66 | 197.3 |  | 16 |  | 0.9067 | 16 | 32 | 0.9724 |
| regtile | 512 | 0.005901 | 22.48 | 179.8 |  | 16 |  | 0.8264 | 64 | 8 | 0.8863 |
| regtile | 512 | 0.007189 | 18.45 | 147.6 |  | 16 |  | 0.6783 | 512 | 1 | 0.7275 |
| limits |  |  |  |  | regtile | 32 | 1024 |  |  |  |  |
| planes | 512 | 0.004811 | 27.57 | 220.6 |  | 32 |  | 1.014 | 30 |  |  |
| regtile | 512 | 0.005253 | 25.25 | 202 |  | 32 |  | 0.9283 | 30 | 18 | 0.9158 |
| regtile | 512 | 0.004929 | 26.91 | 215.3 |  | 32 |  | 0.9893 | 8 | 64 | 0.976 |
| regtile | 512 | 0.005031 | 26.37 | 211 |  | 32 |  | 0.9693 | 16 | 32 | 0.9563 |
| regtile | 512 | 0.00564 | 23.52 | 188.2 |  | 32 |  | 0.8646 | 64 | 8 | 0.853 |
| regtile | 512 | 0.005783 | 22.94 | 183.5 |  | 32 |  | 0.8433 | 512 | 1 | 0.8319 |

### ch09/lab1_racehunt, solution, metal, TILE_WIDTH=16

| name | data | pixels | seconds | gpixels_per_s |
|---|---|---|---|---|
| basic | uniform | 67108864 | 0.004755 | 14.11 |
| pool | uniform | 67108864 | 0.00339 | 19.8 |
| basic | constant | 67108864 | 0.05881 | 1.141 |
| pool | constant | 67108864 | 0.00432 | 15.53 |

### ch09/lab2_privatised, solution, metal, TILE_WIDTH=16

| name | data | coarse | pixels | seconds | gpixels_per_s |
|---|---|---|---|---|---|
| basic | uniform | 1 | 67108864 | 0.004755 | 14.11 |
| contiguous | uniform | 16 | 67108864 | 0.0002297 | 292.1 |
| shared | uniform | 1 | 67108864 | 0.0008829 | 76.01 |
| shared | uniform | 4 | 67108864 | 0.0003625 | 185.1 |
| shared | uniform | 16 | 67108864 | 0.000295 | 227.5 |
| shared | uniform | 64 | 67108864 | 0.0002941 | 228.2 |
| runs | uniform | 16 | 67108864 | 0.0003158 | 212.5 |
| basic | constant | 1 | 67108864 | 0.05884 | 1.141 |
| contiguous | constant | 16 | 67108864 | 0.001967 | 34.12 |
| shared | constant | 1 | 67108864 | 0.002206 | 30.41 |
| shared | constant | 4 | 67108864 | 0.001938 | 34.63 |
| shared | constant | 16 | 67108864 | 0.001923 | 34.91 |
| shared | constant | 64 | 67108864 | 0.001928 | 34.82 |
| runs | constant | 16 | 67108864 | 0.0003445 | 194.8 |
| basic | sky | 1 | 67108864 | 0.05765 | 1.164 |
| contiguous | sky | 16 | 67108864 | 0.001941 | 34.57 |
| shared | sky | 1 | 67108864 | 0.002205 | 30.43 |
| shared | sky | 4 | 67108864 | 0.001934 | 34.7 |
| shared | sky | 16 | 67108864 | 0.001905 | 35.22 |
| shared | sky | 64 | 67108864 | 0.001839 | 36.5 |
| runs | sky | 16 | 67108864 | 0.0003459 | 194 |

### ch09/lab3_tokencounts, solution, metal, TILE_WIDTH=16

| name | data | batch | len | vocab | seconds | tokens_per_s | blocks_per_row | pool_bytes | elements_per_s |
|---|---|---|---|---|---|---|---|---|---|
| count | uniform | 8 | 4096 | 151936 | 5.625e-06 | 5.825e+09 |  |  |  |
| count_pool | uniform | 8 | 4096 | 151936 | 0.0005924 | 5.532e+07 | 16 | 77791232 |  |
| append | uniform | 8 | 4096 | 151936 | 2.75e-06 | 2.909e+06 |  |  |  |
| penalties | uniform | 8 | 4096 | 151936 | 1.871e-05 |  |  |  | 6.497e+10 |
| count | zipf | 8 | 4096 | 151936 | 5.792e-06 | 5.658e+09 |  |  |  |
| count_pool | zipf | 8 | 4096 | 151936 | 0.0005794 | 5.655e+07 | 16 | 77791232 |  |
| append | zipf | 8 | 4096 | 151936 | 2.75e-06 | 2.909e+06 |  |  |  |
| penalties | zipf | 8 | 4096 | 151936 | 1.9e-05 |  |  |  | 6.397e+10 |

### ch10/lab1_sum, solution, metal, TILE_WIDTH=16

| name | kernel | block | rows | seconds | gb_per_s | n | coarse | blocks | runs | distinct |
|---|---|---|---|---|---|---|---|---|---|---|
| rows | simple | 256 | 32768 | 0.0008458 | 79.35 |  |  |  |  |  |
| rows | convergent | 256 | 32768 | 0.0003654 | 183.7 |  |  |  |  |  |
| rows | shared | 256 | 32768 | 0.0002275 | 295 |  |  |  |  |  |
| rows | warp | 256 | 32768 | 0.0002159 | 310.9 |  |  |  |  |  |
| any |  | 256 |  | 0.0008944 | 300.1 | 67108864 | 1 | 131072 |  |  |
| any |  | 256 |  | 0.0008937 | 300.3 | 67108864 | 2 | 65536 |  |  |
| any |  | 256 |  | 0.0008963 | 299.5 | 67108864 | 4 | 32768 |  |  |
| any |  | 256 |  | 0.0008972 | 299.2 | 67108864 | 8 | 16384 |  |  |
| any |  | 256 |  | 0.0008979 | 299 | 67108864 | 16 | 8192 |  |  |
| any |  | 256 |  | 0.0009103 | 294.9 | 67108864 | 32 | 4096 |  |  |
| any |  | 256 |  | 0.0009209 | 291.5 | 67108864 | 64 | 2048 |  |  |
| any |  | 256 |  | 0.0009571 | 280.5 | 67108864 | 128 | 1024 |  |  |
| determinism |  |  |  |  |  | 16777216 |  |  | 20 | 20 |

### ch10/lab3_softmax_argmax, solution, metal, TILE_WIDTH=16

| name | kernel | rows | cols | seconds | gb_per_s |
|---|---|---|---|---|---|
| softmax | naive | 1 | 32000 | 0.006236 | 0.04105 |
| softmax | block | 1 | 32000 | 3.895e-05 | 6.573 |
| softmax | naive | 8 | 32000 | 0.00678 | 0.3021 |
| softmax | block | 8 | 32000 | 4.023e-05 | 50.9 |
| softmax | naive | 64 | 32000 | 0.008301 | 1.974 |
| softmax | block | 64 | 32000 | 8.173e-05 | 200.5 |
| softmax | naive | 1 | 128256 | 0.02602 | 0.03943 |
| softmax | block | 1 | 128256 | 0.0002072 | 4.951 |
| softmax | naive | 8 | 128256 | 0.02767 | 0.2967 |
| softmax | block | 8 | 128256 | 0.0002297 | 35.73 |
| softmax | naive | 64 | 128256 | 0.03527 | 1.862 |
| softmax | block | 64 | 128256 | 0.0005942 | 110.5 |
| argmax | naive | 1 | 32000 | 0.002419 |  |
| argmax | block | 1 | 32000 | 2.119e-05 |  |
| argmax | naive | 8 | 32000 | 0.002506 |  |
| argmax | block | 8 | 32000 | 1.969e-05 |  |
| argmax | naive | 64 | 32000 | 0.002746 |  |
| argmax | block | 64 | 32000 | 2.59e-05 |  |
| argmax | naive | 1 | 128256 | 0.009689 |  |
| argmax | block | 1 | 128256 | 7.314e-05 |  |
| argmax | naive | 8 | 128256 | 0.01008 |  |
| argmax | block | 8 | 128256 | 7.799e-05 |  |
| argmax | naive | 64 | 128256 | 0.01175 |  |
| argmax | block | 64 | 128256 | 0.0001761 |  |

### ch11/lab1_blockscan, solution, metal, TILE_WIDTH=16

| name | kernel | block | n | seconds | gb_per_s | share_of_copy |
|---|---|---|---|---|---|---|
| scan | copy | 256 | 16777216 | 0.0004859 | 276.2 | 1 |
| scan | two_barrier | 256 | 16777216 | 0.0006033 | 222.5 | 0.8053 |
| scan | double_buffer | 256 | 16777216 | 0.000624 | 215.1 | 0.7786 |
| scan | warp_shuffle | 256 | 16777216 | 0.0004869 | 275.7 | 0.9979 |
| scan | warp_prefix | 256 | 16777216 | 0.0004905 | 273.6 | 0.9906 |
| scan | copy | 512 | 16777216 | 0.0004853 | 276.6 | 1 |
| scan | two_barrier | 512 | 16777216 | 0.0006727 | 199.5 | 0.7214 |
| scan | double_buffer | 512 | 16777216 | 0.000689 | 194.8 | 0.7043 |
| scan | warp_shuffle | 512 | 16777216 | 0.0004899 | 274 | 0.9906 |
| scan | warp_prefix | 512 | 16777216 | 0.0004876 | 275.2 | 0.9951 |
| scan | copy | 1024 | 16777216 | 0.0004875 | 275.3 | 1 |
| scan | two_barrier | 1024 | 16777216 | 0.0007659 | 175.2 | 0.6365 |
| scan | double_buffer | 1024 | 16777216 | 0.0007714 | 174 | 0.6319 |
| scan | warp_shuffle | 1024 | 16777216 | 0.0004913 | 273.2 | 0.9922 |
| scan | warp_prefix | 1024 | 16777216 | 0.0004884 | 274.8 | 0.9981 |

### ch11/lab2_coarsened, solution, metal, TILE_WIDTH=16

| name | kernel | block | n | seconds | gb_per_s | share_of_copy | coarse |
|---|---|---|---|---|---|---|---|
| scan | copy | 256 | 67108864 | 0.002001 | 268.3 | 1 |  |
| scan | coarsened | 256 | 67108864 | 0.00237 | 226.6 | 0.8446 | 1 |
| scan | register_tiled | 256 | 67108864 | 0.002367 | 226.9 | 0.8457 | 1 |
| scan | thread_major | 256 | 67108864 | 0.002023 | 265.4 | 0.9892 | 1 |
| scan | coarsened | 256 | 67108864 | 0.002027 | 264.9 | 0.9874 | 2 |
| scan | register_tiled | 256 | 67108864 | 0.002023 | 265.3 | 0.9892 | 2 |
| scan | thread_major | 256 | 67108864 | 0.002036 | 263.7 | 0.9829 | 2 |
| scan | coarsened | 256 | 67108864 | 0.002052 | 261.7 | 0.9755 | 4 |
| scan | register_tiled | 256 | 67108864 | 0.002032 | 264.2 | 0.9849 | 4 |
| scan | thread_major | 256 | 67108864 | 0.002028 | 264.8 | 0.987 | 4 |
| scan | coarsened | 256 | 67108864 | 0.002055 | 261.2 | 0.9739 | 8 |
| scan | register_tiled | 256 | 67108864 | 0.002052 | 261.6 | 0.9753 | 8 |
| scan | thread_major | 256 | 67108864 | 0.002037 | 263.6 | 0.9826 | 8 |
| scan | coarsened | 256 | 67108864 | 0.002183 | 246 | 0.9169 | 16 |
| scan | register_tiled | 256 | 67108864 | 0.00213 | 252 | 0.9394 | 16 |
| scan | thread_major | 256 | 67108864 | 0.002034 | 263.9 | 0.9838 | 16 |
| scan | copy | 1024 | 67108864 | 0.002002 | 268.1 | 1 |  |
| scan | coarsened | 1024 | 67108864 | 0.002138 | 251.1 | 0.9366 | 1 |
| scan | register_tiled | 1024 | 67108864 | 0.002141 | 250.7 | 0.9351 | 1 |
| scan | thread_major | 1024 | 67108864 | 0.002024 | 265.3 | 0.9894 | 1 |
| scan | coarsened | 1024 | 67108864 | 0.002043 | 262.8 | 0.98 | 2 |
| scan | register_tiled | 1024 | 67108864 | 0.002038 | 263.4 | 0.9823 | 2 |
| scan | thread_major | 1024 | 67108864 | 0.002039 | 263.3 | 0.9819 | 2 |
| scan | coarsened | 1024 | 67108864 | 0.002036 | 263.7 | 0.9835 | 4 |
| scan | register_tiled | 1024 | 67108864 | 0.002043 | 262.8 | 0.9803 | 4 |
| scan | thread_major | 1024 | 67108864 | 0.002037 | 263.6 | 0.9831 | 4 |
| scan | thread_major | 1024 | 67108864 | 0.002037 | 263.6 | 0.9832 | 8 |
| scan | thread_major | 1024 | 67108864 | 0.002049 | 262 | 0.977 | 16 |

### ch11/lab4_global, solution, metal, TILE_WIDTH=16

| name | kernel | block | segment | coarse | n | seconds | elements_per_s | book_bytes_per_element | gb_per_s | share_of_copy |
|---|---|---|---|---|---|---|---|---|---|---|
| global_scan | copy | 256 | 1024 | 4 | 67108864 | 0.002004 | 3.349e+10 | 8 | 267.9 | 1 |
| global_scan | scan_scan_add | 256 | 1024 | 4 | 67108864 | 0.003971 | 1.69e+10 | 16.01 | 270.6 | 0.5047 |
| global_scan | reduce_scan_scan | 256 | 1024 | 4 | 67108864 | 0.002932 | 2.288e+10 | 12.01 | 274.9 | 0.6833 |
| global_scan | copy | 1024 | 4096 | 4 | 67108864 | 0.002001 | 3.354e+10 | 8 | 268.3 | 1 |
| global_scan | scan_scan_add | 1024 | 4096 | 4 | 67108864 | 0.003976 | 1.688e+10 | 16 | 270.1 | 0.5033 |
| global_scan | reduce_scan_scan | 1024 | 4096 | 4 | 67108864 | 0.00294 | 2.283e+10 | 12 | 274 | 0.6807 |

### ch11/lab6_topp, solution, metal, TILE_WIDTH=16

| name | kernel | rows | cols | seconds | gb_per_s | share_of_copy | speedup_vs_naive |
|---|---|---|---|---|---|---|---|
| top_p_mask | copy | 1 | 128256 | 4.021e-06 | 287.1 |  |  |
| top_p_mask | naive | 1 | 128256 | 0.009537 | 0.121 | 0.0004216 |  |
| top_p_mask | yours | 1 | 128256 | 0.00016 | 7.215 | 0.02513 | 59.61 |
| top_p_mask | copy | 8 | 128256 | 1.31e-05 | 704.8 |  |  |
| top_p_mask | naive | 8 | 128256 | 0.0103 | 0.8963 | 0.001272 |  |
| top_p_mask | yours | 8 | 128256 | 0.0002065 | 44.71 | 0.06344 | 49.88 |
| top_p_mask | copy | 64 | 128256 | 0.000233 | 317.1 |  |  |
| top_p_mask | naive | 64 | 128256 | 0.01468 | 5.031 | 0.01587 |  |
| top_p_mask | yours | 64 | 128256 | 0.0003676 | 201 | 0.6338 | 39.95 |
| top_p_mask | copy | 256 | 32000 | 0.0002597 | 283.9 |  |  |
| top_p_mask | naive | 256 | 32000 | 0.003851 | 19.15 | 0.06744 |  |
| top_p_mask | yours | 256 | 32000 | 0.0003318 | 222.2 | 0.7828 | 11.61 |

### ch12/lab1_unstable, solution, metal, TILE_WIDTH=16

| name | fraction | n | seconds | gbps | gkeys_per_s |
|---|---|---|---|---|---|
| copy | 1 | 67108864 | 0.002006 | 267.6 | 33.45 |
| atomic | 0 | 67108864 | 0.0008922 | 300.9 | 75.21 |
| warp | 0 | 67108864 | 0.0008945 | 300.1 | 75.03 |
| private | 0 | 67108864 | 0.0008935 | 300.4 | 75.11 |
| atomic | 0.001 | 67108864 | 0.000893 | 300.9 | 75.15 |
| warp | 0.001 | 67108864 | 0.0008976 | 299.4 | 74.77 |
| private | 0.001 | 67108864 | 0.0008968 | 299.6 | 74.83 |
| atomic | 0.01 | 67108864 | 0.0009039 | 299.9 | 74.25 |
| warp | 0.01 | 67108864 | 0.0008989 | 301.6 | 74.66 |
| private | 0.01 | 67108864 | 0.0008992 | 301.5 | 74.63 |
| atomic | 0.03 | 67108864 | 0.001246 | 221.9 | 53.86 |
| warp | 0.03 | 67108864 | 0.001247 | 221.7 | 53.82 |
| private | 0.03 | 67108864 | 0.0009684 | 285.5 | 69.3 |
| atomic | 0.1 | 67108864 | 0.001882 | 156.9 | 35.66 |
| warp | 0.1 | 67108864 | 0.001882 | 156.9 | 35.66 |
| private | 0.1 | 67108864 | 0.001069 | 276.3 | 62.79 |
| atomic | 0.25 | 67108864 | 0.001966 | 170.7 | 34.13 |
| warp | 0.25 | 67108864 | 0.001966 | 170.7 | 34.14 |
| private | 0.25 | 67108864 | 0.001251 | 268.3 | 53.65 |
| atomic | 0.5 | 67108864 | 0.001998 | 201.5 | 33.58 |
| warp | 0.5 | 67108864 | 0.001998 | 201.5 | 33.59 |
| private | 0.5 | 67108864 | 0.001536 | 262.1 | 43.69 |
| atomic | 0.75 | 67108864 | 0.00215 | 218.5 | 31.22 |
| warp | 0.75 | 67108864 | 0.00203 | 231.4 | 33.06 |
| private | 0.75 | 67108864 | 0.001782 | 263.6 | 37.66 |
| atomic | 1 | 67108864 | 0.002035 | 263.9 | 32.98 |
| warp | 1 | 67108864 | 0.002037 | 263.5 | 32.94 |
| private | 1 | 67108864 | 0.002042 | 263 | 32.87 |

### ch12/lab3_stable, solution, metal, TILE_WIDTH=16

| name | n | seconds | gbps | fraction | kept_fraction | coarse | model_bytes_per_key | copy_equivalent_bytes_per_key |
|---|---|---|---|---|---|---|---|---|
| copy | 134217728 | 0.004028 | 266.6 |  |  |  |  |  |
| flags | 134217728 | 0.01388 | 38.72 | 0.001 | 0.001001 | 0 | 28 | 27.57 |
| staged | 134217728 | 0.006511 | 82.54 | 0.001 | 0.001001 | 1 | 8.004 | 12.93 |
| staged | 134217728 | 0.005984 | 89.8 | 0.001 | 0.001001 | 2 | 8.004 | 11.89 |
| staged | 134217728 | 0.005728 | 93.83 | 0.001 | 0.001001 | 4 | 8.004 | 11.38 |
| staged | 134217728 | 0.005627 | 95.51 | 0.001 | 0.001001 | 8 | 8.004 | 11.18 |
| staged | 134217728 | 0.0069 | 77.88 | 0.001 | 0.001001 | 16 | 8.004 | 13.7 |
| flags | 134217728 | 0.01447 | 37.47 | 0.01 | 0.009998 | 0 | 28.04 | 28.75 |
| staged | 134217728 | 0.006588 | 82.31 | 0.01 | 0.009998 | 1 | 8.04 | 13.08 |
| staged | 134217728 | 0.006024 | 90.01 | 0.01 | 0.009998 | 2 | 8.04 | 11.97 |
| staged | 134217728 | 0.005771 | 93.96 | 0.01 | 0.009998 | 4 | 8.04 | 11.46 |
| staged | 134217728 | 0.005658 | 95.84 | 0.01 | 0.009998 | 8 | 8.04 | 11.24 |
| staged | 134217728 | 0.006937 | 78.17 | 0.01 | 0.009998 | 16 | 8.04 | 13.78 |
| flags | 134217728 | 0.01714 | 34.45 | 0.1 | 0.1 | 0 | 28.4 | 34.05 |
| staged | 134217728 | 0.006657 | 88.71 | 0.1 | 0.1 | 1 | 8.4 | 13.22 |
| staged | 134217728 | 0.00615 | 96.02 | 0.1 | 0.1 | 2 | 8.4 | 12.22 |
| staged | 134217728 | 0.005894 | 100.2 | 0.1 | 0.1 | 4 | 8.4 | 11.71 |
| staged | 134217728 | 0.005775 | 102.3 | 0.1 | 0.1 | 8 | 8.4 | 11.47 |
| staged | 134217728 | 0.007187 | 82.17 | 0.1 | 0.1 | 16 | 8.4 | 14.27 |
| flags | 134217728 | 0.0189 | 42.61 | 0.5 | 0.5 | 0 | 30 | 37.54 |
| staged | 134217728 | 0.006828 | 118 | 0.5 | 0.5 | 1 | 10 | 13.56 |
| staged | 134217728 | 0.0063 | 127.8 | 0.5 | 0.5 | 2 | 10 | 12.51 |
| staged | 134217728 | 0.006036 | 133.4 | 0.5 | 0.5 | 4 | 10 | 11.99 |
| staged | 134217728 | 0.005908 | 136.3 | 0.5 | 0.5 | 8 | 10 | 11.73 |
| staged | 134217728 | 0.007798 | 103.3 | 0.5 | 0.5 | 16 | 10 | 15.49 |
| flags | 134217728 | 0.01977 | 51.6 | 0.9 | 0.9 | 0 | 31.6 | 39.27 |
| staged | 134217728 | 0.006976 | 146.2 | 0.9 | 0.9 | 1 | 11.6 | 13.85 |
| staged | 134217728 | 0.006449 | 158.2 | 0.9 | 0.9 | 2 | 11.6 | 12.81 |
| staged | 134217728 | 0.006181 | 165 | 0.9 | 0.9 | 4 | 11.6 | 12.28 |
| staged | 134217728 | 0.006056 | 168.4 | 0.9 | 0.9 | 8 | 11.6 | 12.03 |
| staged | 134217728 | 0.008202 | 124.4 | 0.9 | 0.9 | 16 | 11.6 | 16.29 |
| flags | 134217728 | 0.01992 | 53.89 | 1 | 1 | 0 | 32 | 39.57 |
| staged | 134217728 | 0.006978 | 153.9 | 1 | 1 | 1 | 12 | 13.86 |
| staged | 134217728 | 0.006452 | 166.4 | 1 | 1 | 2 | 12 | 12.82 |
| staged | 134217728 | 0.00622 | 172.6 | 1 | 1 | 4 | 12 | 12.35 |
| staged | 134217728 | 0.006091 | 176.3 | 1 | 1 | 8 | 12 | 12.1 |
| staged | 134217728 | 0.008221 | 130.6 | 1 | 1 | 16 | 12 | 16.33 |

### ch12/lab4_candidates, solution, metal, TILE_WIDTH=16

| name | kernel | rows | cols | fraction | kept | seconds | gb_per_s | rows_per_s |
|---|---|---|---|---|---|---|---|---|
| compaction | serial | 1 | 32000 | 0.001 | 32 | 0.002492 | 0.05147 | 401.3 |
| compaction | block | 1 | 32000 | 0.001 | 32 | 0.0001065 | 1.205 | 9391 |
| compaction | serial | 1 | 32000 | 0.01 | 320 | 0.002472 | 0.05282 | 404.5 |
| compaction | block | 1 | 32000 | 0.01 | 320 | 0.0001066 | 1.224 | 9378 |
| compaction | serial | 1 | 32000 | 0.1 | 3201 | 0.002519 | 0.06097 | 396.9 |
| compaction | block | 1 | 32000 | 0.1 | 3201 | 0.0001083 | 1.418 | 9231 |
| compaction | serial | 1 | 32000 | 0.5 | 16001 | 0.002736 | 0.09357 | 365.5 |
| compaction | block | 1 | 32000 | 0.5 | 16001 | 0.0001125 | 2.276 | 8892 |
| compaction | serial | 8 | 32000 | 0.001 | 258 | 0.002561 | 0.4007 | 3124 |
| compaction | block | 8 | 32000 | 0.001 | 258 | 0.0001084 | 9.465 | 7.379e+04 |
| compaction | serial | 8 | 32000 | 0.01 | 2560 | 0.002615 | 0.3995 | 3059 |
| compaction | block | 8 | 32000 | 0.01 | 2560 | 0.0001086 | 9.617 | 7.365e+04 |
| compaction | serial | 8 | 32000 | 0.1 | 25604 | 0.003014 | 0.4078 | 2655 |
| compaction | block | 8 | 32000 | 0.1 | 25604 | 0.0001106 | 11.11 | 7.231e+04 |
| compaction | serial | 8 | 32000 | 0.5 | 128002 | 0.003366 | 0.6085 | 2377 |
| compaction | block | 8 | 32000 | 0.5 | 128002 | 0.0001125 | 18.21 | 7.114e+04 |
| compaction | serial | 64 | 32000 | 0.001 | 2064 | 0.002837 | 2.893 | 2.256e+04 |
| compaction | block | 64 | 32000 | 0.001 | 2064 | 0.0001269 | 64.71 | 5.045e+05 |
| compaction | serial | 64 | 32000 | 0.01 | 20491 | 0.003041 | 2.748 | 2.105e+04 |
| compaction | block | 64 | 32000 | 0.01 | 20491 | 0.0001249 | 66.93 | 5.126e+05 |
| compaction | serial | 64 | 32000 | 0.1 | 204817 | 0.003904 | 2.518 | 1.639e+04 |
| compaction | block | 64 | 32000 | 0.1 | 204817 | 0.0001327 | 74.11 | 4.824e+05 |
| compaction | serial | 64 | 32000 | 0.5 | 1024023 | 0.004139 | 3.958 | 1.546e+04 |
| compaction | block | 64 | 32000 | 0.5 | 1024023 | 0.0001387 | 118.1 | 4.614e+05 |
| compaction | serial | 1 | 128256 | 0.001 | 129 | 0.00988 | 0.05203 | 101.2 |
| compaction | block | 1 | 128256 | 0.001 | 129 | 0.0004324 | 1.189 | 2313 |
| compaction | serial | 1 | 128256 | 0.01 | 1283 | 0.009905 | 0.05283 | 101 |
| compaction | block | 1 | 128256 | 0.01 | 1283 | 0.000433 | 1.209 | 2310 |
| compaction | serial | 1 | 128256 | 0.1 | 12826 | 0.01009 | 0.06099 | 99.07 |
| compaction | block | 1 | 128256 | 0.1 | 12826 | 0.0004393 | 1.401 | 2276 |
| compaction | serial | 1 | 128256 | 0.5 | 64129 | 0.01097 | 0.09351 | 91.13 |
| compaction | block | 1 | 128256 | 0.5 | 64129 | 0.000447 | 2.296 | 2237 |
| compaction | serial | 8 | 128256 | 0.001 | 1033 | 0.01032 | 0.3987 | 775.5 |
| compaction | block | 8 | 128256 | 0.001 | 1033 | 0.0004276 | 9.617 | 1.871e+04 |
| compaction | serial | 8 | 128256 | 0.01 | 10268 | 0.01052 | 0.398 | 760.6 |
| compaction | block | 8 | 128256 | 0.01 | 10268 | 0.000428 | 9.782 | 1.869e+04 |
| compaction | serial | 8 | 128256 | 0.1 | 102617 | 0.01212 | 0.4064 | 660.2 |
| compaction | block | 8 | 128256 | 0.1 | 102617 | 0.0004365 | 11.28 | 1.833e+04 |
| compaction | serial | 8 | 128256 | 0.5 | 513035 | 0.01377 | 0.5961 | 580.9 |
| compaction | block | 8 | 128256 | 0.5 | 513035 | 0.0004804 | 17.09 | 1.665e+04 |
| compaction | serial | 64 | 128256 | 0.001 | 8257 | 0.01208 | 2.724 | 5299 |
| compaction | block | 64 | 128256 | 0.001 | 8257 | 0.0005459 | 60.27 | 1.172e+05 |
| compaction | serial | 64 | 128256 | 0.01 | 82172 | 0.0129 | 2.596 | 4960 |
| compaction | block | 64 | 128256 | 0.01 | 82172 | 0.0005483 | 61.08 | 1.167e+05 |
| compaction | serial | 64 | 128256 | 0.1 | 820916 | 0.0165 | 2.387 | 3878 |
| compaction | block | 64 | 128256 | 0.1 | 820916 | 0.0005621 | 70.1 | 1.139e+05 |
| compaction | serial | 64 | 128256 | 0.5 | 4104257 | 0.01663 | 3.948 | 3848 |
| compaction | block | 64 | 128256 | 0.5 | 4104257 | 0.000635 | 103.4 | 1.008e+05 |

### ch13/lab1_merge, solution, metal, TILE_WIDTH=16

| name | per_thread | m | n | total | seconds | gb_per_s |
|---|---|---|---|---|---|---|
| basic | 1 | 8388608 | 8388608 | 16777216 | 0.006105 | 43.97 |
| basic | 2 | 8388608 | 8388608 | 16777216 | 0.003389 | 79.2 |
| basic | 4 | 8388608 | 8388608 | 16777216 | 0.003425 | 78.36 |
| basic | 8 | 8388608 | 8388608 | 16777216 | 0.004135 | 64.92 |
| basic | 16 | 8388608 | 8388608 | 16777216 | 0.004824 | 55.65 |
| tiled | 4 | 8388608 | 8388608 | 16777216 | 0.001308 | 205.3 |
| tiled | 8 | 8388608 | 8388608 | 16777216 | 0.001155 | 232.3 |
| tiled | 16 | 8388608 | 8388608 | 16777216 | 0.001177 | 228 |

### ch14/lab1_twosorts, solution, metal, TILE_WIDTH=16

| name | n | steps | seconds | wall_seconds | mkeys_per_s |
|---|---|---|---|---|---|
| oddeven | 1024 | 978 | 0.0762 |  |  |
| oddeven | 4096 | 4032 | 0.3322 |  |  |
| radix1 | 1024 |  | 0.000679 | 0.0008914 | 1.508 |
| cpu_stable_sort | 1024 |  | 1.563e-05 |  | 65.54 |
| radix1 | 4096 |  | 0.0005996 | 0.0008211 | 6.831 |
| cpu_stable_sort | 4096 |  | 7.95e-05 |  | 51.52 |
| radix1 | 65536 |  | 0.0007171 | 0.001039 | 91.39 |
| cpu_stable_sort | 65536 |  | 0.002037 |  | 32.18 |
| radix1 | 1048576 |  | 0.005686 | 0.006341 | 184.4 |
| cpu_stable_sort | 1048576 |  | 0.04349 |  | 24.11 |
| radix1 | 4194304 |  | 0.01759 | 0.01844 | 238.5 |
| cpu_stable_sort | 4194304 |  | 0.1909 |  | 21.97 |

### ch14/lab2_multibit, solution, metal, TILE_WIDTH=16

| name | n | seconds | mkeys_per_s | radix_bits | keys_per_thread | passes | counted_gbps |
|---|---|---|---|---|---|---|---|
| onebit_global | 8388608 | 0.04158 | 201.8 |  |  |  |  |
| local | 8388608 | 0.03424 | 245 | 1 | 1 | 32 | 250.8 |
| local | 8388608 | 0.02205 | 380.5 | 2 | 1 | 16 | 194.8 |
| local | 8388608 | 0.01724 | 486.6 | 4 | 1 | 8 | 124.6 |
| local | 8388608 | 0.01642 | 510.7 | 6 | 1 | 6 | 98.06 |
| local | 8388608 | 0.01693 | 495.6 | 8 | 1 | 4 | 63.43 |
| local | 8388608 | 0.01294 | 648.2 | 4 | 2 | 8 | 165.9 |
| local | 8388608 | 0.01082 | 775 | 4 | 4 | 8 | 198.4 |
| local | 8388608 | 0.01011 | 829.9 | 4 | 8 | 8 | 212.5 |
| local | 8388608 | 0.0115 | 729.7 | 8 | 2 | 4 | 93.41 |
| local | 8388608 | 0.009999 | 839 | 8 | 4 | 4 | 107.4 |
| local | 8388608 | 0.008821 | 950.9 | 8 | 8 | 4 | 121.7 |
| cpu_stable_sort | 8388608 | 0.4084 | 20.54 |  |  |  |  |

### ch14/lab3_sortrows, solution, metal, TILE_WIDTH=16

| name | rows | cols | seconds | rows_per_s | mkeys_per_s |
|---|---|---|---|---|---|
| sort_rows | 1 | 32000 | 0.0009821 | 1018 | 32.58 |
| cpu_stable_sort_rows | 1 | 32000 | 0.001828 | 546.9 |  |
| sort_rows | 8 | 32000 | 0.001014 | 7892 | 252.5 |
| cpu_stable_sort_rows | 8 | 32000 | 0.01503 | 532.2 |  |
| sort_rows | 64 | 32000 | 0.001621 | 3.949e+04 | 1264 |
| cpu_stable_sort_rows | 64 | 32000 | 0.1211 | 528.4 |  |
| sort_rows | 1 | 151936 | 0.004518 | 221.3 | 33.63 |
| cpu_stable_sort_rows | 1 | 151936 | 0.01136 | 88.06 |  |
| sort_rows | 8 | 151936 | 0.005275 | 1516 | 230.4 |
| cpu_stable_sort_rows | 8 | 151936 | 0.0919 | 87.05 |  |
| sort_rows | 64 | 151936 | 0.009317 | 6869 | 1044 |
| cpu_stable_sort_rows | 64 | 151936 | 0.7347 | 87.11 |  |

### ch15/lab1_coarse, solution, metal, TILE_WIDTH=16

| name | max_threads_per_threadgroup | simd_width | static_threadgroup_bytes | kernel | shape | m | k | n | seconds | gflops |
|---|---|---|---|---|---|---|---|---|---|---|
| pipeline | 256 | 32 | 8192 |  |  |  |  |  |  |  |
| tiled_5_3 |  |  |  | 5.3 | square | 1024 | 1024 | 1024 | 0.001991 | 1078 |
| stage |  |  |  | 15.1 | square | 1024 | 1024 | 1024 | 0.001559 | 1377 |
| vendor |  |  |  | vendor | square | 1024 | 1024 | 1024 | 0.0003258 | 6592 |
| tiled_5_3 |  |  |  | 5.3 | square | 2048 | 2048 | 2048 | 0.01205 | 1426 |
| stage |  |  |  | 15.1 | square | 2048 | 2048 | 2048 | 0.007243 | 2372 |
| vendor |  |  |  | vendor | square | 2048 | 2048 | 2048 | 0.001956 | 8781 |
| tiled_5_3 |  |  |  | 5.3 | square | 4096 | 4096 | 4096 | 0.0992 | 1386 |
| stage |  |  |  | 15.1 | square | 4096 | 4096 | 4096 | 0.05516 | 2492 |
| vendor |  |  |  | vendor | square | 4096 | 4096 | 4096 | 0.01533 | 8965 |
| tiled_5_3 |  |  |  | 5.3 | layer | 1 | 1024 | 4096 | 0.0001255 | 66.84 |
| stage |  |  |  | 15.1 | layer | 1 | 1024 | 4096 | 0.0006228 | 13.47 |
| vendor |  |  |  | vendor | layer | 1 | 1024 | 4096 | 3.083e-05 | 272.1 |
| tiled_5_3 |  |  |  | 5.3 | layer | 16 | 1024 | 4096 | 0.0001282 | 1047 |
| stage |  |  |  | 15.1 | layer | 16 | 1024 | 4096 | 0.0006279 | 213.8 |
| vendor |  |  |  | vendor | layer | 16 | 1024 | 4096 | 3.788e-05 | 3544 |
| tiled_5_3 |  |  |  | 5.3 | layer | 256 | 1024 | 4096 | 0.001526 | 1407 |
| stage |  |  |  | 15.1 | layer | 256 | 1024 | 4096 | 0.00129 | 1665 |
| vendor |  |  |  | vendor | layer | 256 | 1024 | 4096 | 0.0002904 | 7396 |
| tiled_5_3 |  |  |  | 5.3 | layer | 1024 | 1024 | 4096 | 0.006004 | 1431 |
| stage |  |  |  | 15.1 | layer | 1024 | 1024 | 4096 | 0.003586 | 2395 |
| vendor |  |  |  | vendor | layer | 1024 | 1024 | 4096 | 0.0009987 | 8601 |
| tiled_5_3 |  |  |  | 5.3 | short_k | 4096 | 128 | 4096 | 0.003051 | 1408 |
| stage |  |  |  | 15.1 | short_k | 4096 | 128 | 4096 | 0.001822 | 2357 |
| vendor |  |  |  | vendor | short_k | 4096 | 128 | 4096 | 0.000501 | 8573 |

### ch15/lab2_regtile, solution, metal, TILE_WIDTH=16

| name | max_threads_per_threadgroup | simd_width | static_threadgroup_bytes | kernel | shape | m | k | n | seconds | gflops |
|---|---|---|---|---|---|---|---|---|---|---|
| pipeline | 256 | 32 | 8192 |  |  |  |  |  |  |  |
| tiled_5_3 |  |  |  | 5.3 | square | 1024 | 1024 | 1024 | 0.001604 | 1339 |
| previous |  |  |  | 15.1 | square | 1024 | 1024 | 1024 | 0.001273 | 1686 |
| stage |  |  |  | 15.2 | square | 1024 | 1024 | 1024 | 0.001256 | 1710 |
| vendor |  |  |  | vendor | square | 1024 | 1024 | 1024 | 0.0002793 | 7689 |
| tiled_5_3 |  |  |  | 5.3 | square | 2048 | 2048 | 2048 | 0.01211 | 1419 |
| previous |  |  |  | 15.1 | square | 2048 | 2048 | 2048 | 0.007341 | 2340 |
| stage |  |  |  | 15.2 | square | 2048 | 2048 | 2048 | 0.007465 | 2301 |
| vendor |  |  |  | vendor | square | 2048 | 2048 | 2048 | 0.001954 | 8792 |
| tiled_5_3 |  |  |  | 5.3 | square | 4096 | 4096 | 4096 | 0.1005 | 1368 |
| previous |  |  |  | 15.1 | square | 4096 | 4096 | 4096 | 0.05513 | 2493 |
| stage |  |  |  | 15.2 | square | 4096 | 4096 | 4096 | 0.05694 | 2414 |
| vendor |  |  |  | vendor | square | 4096 | 4096 | 4096 | 0.01534 | 8961 |
| tiled_5_3 |  |  |  | 5.3 | layer | 1 | 1024 | 4096 | 0.0001254 | 66.91 |
| previous |  |  |  | 15.1 | layer | 1 | 1024 | 4096 | 0.0006253 | 13.41 |
| stage |  |  |  | 15.2 | layer | 1 | 1024 | 4096 | 0.0006346 | 13.22 |
| vendor |  |  |  | vendor | layer | 1 | 1024 | 4096 | 2.863e-05 | 293.1 |
| tiled_5_3 |  |  |  | 5.3 | layer | 16 | 1024 | 4096 | 0.0001283 | 1046 |
| previous |  |  |  | 15.1 | layer | 16 | 1024 | 4096 | 0.0006269 | 214.1 |
| stage |  |  |  | 15.2 | layer | 16 | 1024 | 4096 | 0.0006358 | 211.1 |
| vendor |  |  |  | vendor | layer | 16 | 1024 | 4096 | 3.75e-05 | 3579 |
| tiled_5_3 |  |  |  | 5.3 | layer | 256 | 1024 | 4096 | 0.001536 | 1398 |
| previous |  |  |  | 15.1 | layer | 256 | 1024 | 4096 | 0.001288 | 1668 |
| stage |  |  |  | 15.2 | layer | 256 | 1024 | 4096 | 0.001322 | 1624 |
| vendor |  |  |  | vendor | layer | 256 | 1024 | 4096 | 0.000289 | 7432 |
| tiled_5_3 |  |  |  | 5.3 | layer | 1024 | 1024 | 4096 | 0.006004 | 1431 |
| previous |  |  |  | 15.1 | layer | 1024 | 1024 | 4096 | 0.003583 | 2397 |
| stage |  |  |  | 15.2 | layer | 1024 | 1024 | 4096 | 0.00372 | 2309 |
| vendor |  |  |  | vendor | layer | 1024 | 1024 | 4096 | 0.0009964 | 8621 |
| tiled_5_3 |  |  |  | 5.3 | short_k | 4096 | 128 | 4096 | 0.003051 | 1408 |
| previous |  |  |  | 15.1 | short_k | 4096 | 128 | 4096 | 0.001829 | 2349 |
| stage |  |  |  | 15.2 | short_k | 4096 | 128 | 4096 | 0.00192 | 2237 |
| vendor |  |  |  | vendor | short_k | 4096 | 128 | 4096 | 0.0005006 | 8580 |

### ch15/lab3_remap, solution, metal, TILE_WIDTH=16

| name | max_threads_per_threadgroup | simd_width | static_threadgroup_bytes | kernel | shape | m | k | n | seconds | gflops | width |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pipeline | 256 | 32 | 8192 |  |  |  |  |  |  |  |  |
| tiled_5_3 |  |  |  | 5.3 | square | 1024 | 1024 | 1024 | 0.001524 | 1409 |  |
| previous |  |  |  | 15.2 | square | 1024 | 1024 | 1024 | 0.001246 | 1723 |  |
| stage |  |  |  | 15.3 | square | 1024 | 1024 | 1024 | 0.001185 | 1812 | 32 |
| vendor |  |  |  | vendor | square | 1024 | 1024 | 1024 | 0.0002791 | 7694 |  |
| tiled_5_3 |  |  |  | 5.3 | square | 2048 | 2048 | 2048 | 0.01202 | 1429 |  |
| previous |  |  |  | 15.2 | square | 2048 | 2048 | 2048 | 0.007482 | 2296 |  |
| stage |  |  |  | 15.3 | square | 2048 | 2048 | 2048 | 0.007106 | 2418 | 32 |
| vendor |  |  |  | vendor | square | 2048 | 2048 | 2048 | 0.001954 | 8794 |  |
| tiled_5_3 |  |  |  | 5.3 | square | 4096 | 4096 | 4096 | 0.09964 | 1379 |  |
| previous |  |  |  | 15.2 | square | 4096 | 4096 | 4096 | 0.05657 | 2429 |  |
| stage |  |  |  | 15.3 | square | 4096 | 4096 | 4096 | 0.05406 | 2542 | 32 |
| vendor |  |  |  | vendor | square | 4096 | 4096 | 4096 | 0.01534 | 8961 |  |
| tiled_5_3 |  |  |  | 5.3 | layer | 1 | 1024 | 4096 | 0.0001255 | 66.82 |  |
| previous |  |  |  | 15.2 | layer | 1 | 1024 | 4096 | 0.0006295 | 13.33 |  |
| stage |  |  |  | 15.3 | layer | 1 | 1024 | 4096 | 0.0006118 | 13.71 | 32 |
| vendor |  |  |  | vendor | layer | 1 | 1024 | 4096 | 2.9e-05 | 289.3 |  |
| tiled_5_3 |  |  |  | 5.3 | layer | 16 | 1024 | 4096 | 0.0001285 | 1045 |  |
| previous |  |  |  | 15.2 | layer | 16 | 1024 | 4096 | 0.0006325 | 212.2 |  |
| stage |  |  |  | 15.3 | layer | 16 | 1024 | 4096 | 0.0006125 | 219.1 | 32 |
| vendor |  |  |  | vendor | layer | 16 | 1024 | 4096 | 3.758e-05 | 3571 |  |
| tiled_5_3 |  |  |  | 5.3 | layer | 256 | 1024 | 4096 | 0.001525 | 1408 |  |
| previous |  |  |  | 15.2 | layer | 256 | 1024 | 4096 | 0.001332 | 1612 |  |
| stage |  |  |  | 15.3 | layer | 256 | 1024 | 4096 | 0.001248 | 1721 | 32 |
| vendor |  |  |  | vendor | layer | 256 | 1024 | 4096 | 0.0002895 | 7418 |  |
| tiled_5_3 |  |  |  | 5.3 | layer | 1024 | 1024 | 4096 | 0.006006 | 1430 |  |
| previous |  |  |  | 15.2 | layer | 1024 | 1024 | 4096 | 0.003714 | 2313 |  |
| stage |  |  |  | 15.3 | layer | 1024 | 1024 | 4096 | 0.003535 | 2430 | 32 |
| vendor |  |  |  | vendor | layer | 1024 | 1024 | 4096 | 0.0009975 | 8612 |  |
| tiled_5_3 |  |  |  | 5.3 | short_k | 4096 | 128 | 4096 | 0.003052 | 1407 |  |
| previous |  |  |  | 15.2 | short_k | 4096 | 128 | 4096 | 0.001923 | 2234 |  |
| stage |  |  |  | 15.3 | short_k | 4096 | 128 | 4096 | 0.001773 | 2422 | 32 |
| vendor |  |  |  | vendor | short_k | 4096 | 128 | 4096 | 0.0005035 | 8530 |  |

### ch15/lab4_padded, solution, metal, TILE_WIDTH=16

| name | max_threads_per_threadgroup | simd_width | static_threadgroup_bytes | kernel | shape | m | k | n | seconds | gflops | width |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pipeline | 256 | 32 | 8704 |  |  |  |  |  |  |  |  |
| tiled_5_3 |  |  |  | 5.3 | square | 1024 | 1024 | 1024 | 0.001604 | 1339 |  |
| previous |  |  |  | 15.3 | square | 1024 | 1024 | 1024 | 0.001252 | 1715 | 32 |
| stage |  |  |  | 15.4 | square | 1024 | 1024 | 1024 | 0.00102 | 2104 | 32 |
| vendor |  |  |  | vendor | square | 1024 | 1024 | 1024 | 0.0002783 | 7717 |  |
| tiled_5_3 |  |  |  | 5.3 | square | 2048 | 2048 | 2048 | 0.01201 | 1430 |  |
| previous |  |  |  | 15.3 | square | 2048 | 2048 | 2048 | 0.007144 | 2405 | 32 |
| stage |  |  |  | 15.4 | square | 2048 | 2048 | 2048 | 0.006292 | 2730 | 32 |
| vendor |  |  |  | vendor | square | 2048 | 2048 | 2048 | 0.001954 | 8791 |  |
| tiled_5_3 |  |  |  | 5.3 | square | 4096 | 4096 | 4096 | 0.09789 | 1404 |  |
| previous |  |  |  | 15.3 | square | 4096 | 4096 | 4096 | 0.05429 | 2532 | 32 |
| stage |  |  |  | 15.4 | square | 4096 | 4096 | 4096 | 0.04594 | 2992 | 32 |
| vendor |  |  |  | vendor | square | 4096 | 4096 | 4096 | 0.01533 | 8966 |  |
| tiled_5_3 |  |  |  | 5.3 | layer | 1 | 1024 | 4096 | 0.0001257 | 66.75 |  |
| previous |  |  |  | 15.3 | layer | 1 | 1024 | 4096 | 0.0006057 | 13.85 | 32 |
| stage |  |  |  | 15.4 | layer | 1 | 1024 | 4096 | 0.000553 | 15.17 | 32 |
| vendor |  |  |  | vendor | layer | 1 | 1024 | 4096 | 2.892e-05 | 290.1 |  |
| tiled_5_3 |  |  |  | 5.3 | layer | 16 | 1024 | 4096 | 0.0001271 | 1056 |  |
| previous |  |  |  | 15.3 | layer | 16 | 1024 | 4096 | 0.000612 | 219.3 | 32 |
| stage |  |  |  | 15.4 | layer | 16 | 1024 | 4096 | 0.0005524 | 243 | 32 |
| vendor |  |  |  | vendor | layer | 16 | 1024 | 4096 | 3.796e-05 | 3536 |  |
| tiled_5_3 |  |  |  | 5.3 | layer | 256 | 1024 | 4096 | 0.001525 | 1408 |  |
| previous |  |  |  | 15.3 | layer | 256 | 1024 | 4096 | 0.001252 | 1715 | 32 |
| stage |  |  |  | 15.4 | layer | 256 | 1024 | 4096 | 0.001046 | 2054 | 32 |
| vendor |  |  |  | vendor | layer | 256 | 1024 | 4096 | 0.0002889 | 7433 |  |
| tiled_5_3 |  |  |  | 5.3 | layer | 1024 | 1024 | 4096 | 0.006005 | 1430 |  |
| previous |  |  |  | 15.3 | layer | 1024 | 1024 | 4096 | 0.003588 | 2394 | 32 |
| stage |  |  |  | 15.4 | layer | 1024 | 1024 | 4096 | 0.003173 | 2707 | 32 |
| vendor |  |  |  | vendor | layer | 1024 | 1024 | 4096 | 0.0009965 | 8620 |  |
| tiled_5_3 |  |  |  | 5.3 | short_k | 4096 | 128 | 4096 | 0.003051 | 1408 |  |
| previous |  |  |  | 15.3 | short_k | 4096 | 128 | 4096 | 0.001787 | 2404 | 32 |
| stage |  |  |  | 15.4 | short_k | 4096 | 128 | 4096 | 0.001641 | 2617 | 32 |
| vendor |  |  |  | vendor | short_k | 4096 | 128 | 4096 | 0.0004997 | 8596 |  |

### ch15/lab5_pipelined, solution, metal, TILE_WIDTH=16

| name | max_threads_per_threadgroup | simd_width | static_threadgroup_bytes | kernel | shape | m | k | n | seconds | gflops | width |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pipeline | 256 | 32 | 17408 |  |  |  |  |  |  |  |  |
| tiled_5_3 |  |  |  | 5.3 | square | 1024 | 1024 | 1024 | 0.001526 | 1408 |  |
| previous |  |  |  | 15.4 | square | 1024 | 1024 | 1024 | 0.001025 | 2095 | 32 |
| stage |  |  |  | 15.5 | square | 1024 | 1024 | 1024 | 0.002407 | 892.3 | 32 |
| vendor |  |  |  | vendor | square | 1024 | 1024 | 1024 | 0.0002777 | 7732 |  |
| tiled_5_3 |  |  |  | 5.3 | square | 2048 | 2048 | 2048 | 0.01378 | 1246 |  |
| previous |  |  |  | 15.4 | square | 2048 | 2048 | 2048 | 0.007201 | 2386 | 32 |
| stage |  |  |  | 15.5 | square | 2048 | 2048 | 2048 | 0.01794 | 957.4 | 32 |
| vendor |  |  |  | vendor | square | 2048 | 2048 | 2048 | 0.002354 | 7297 |  |
| tiled_5_3 |  |  |  | 5.3 | square | 4096 | 4096 | 4096 | 0.1551 | 885.9 |  |
| previous |  |  |  | 15.4 | square | 4096 | 4096 | 4096 | 0.07716 | 1781 | 32 |
| stage |  |  |  | 15.5 | square | 4096 | 4096 | 4096 | 0.1742 | 789 | 32 |
| vendor |  |  |  | vendor | square | 4096 | 4096 | 4096 | 0.02069 | 6644 |  |
| tiled_5_3 |  |  |  | 5.3 | layer | 1 | 1024 | 4096 | 0.0001847 | 45.41 |  |
| previous |  |  |  | 15.4 | layer | 1 | 1024 | 4096 | 0.0008149 | 10.29 | 32 |
| stage |  |  |  | 15.5 | layer | 1 | 1024 | 4096 | 0.002122 | 3.952 | 32 |
| vendor |  |  |  | vendor | layer | 1 | 1024 | 4096 | 5.112e-05 | 164.1 |  |
| tiled_5_3 |  |  |  | 5.3 | layer | 16 | 1024 | 4096 | 0.000166 | 808.5 |  |
| previous |  |  |  | 15.4 | layer | 16 | 1024 | 4096 | 0.0007168 | 187.2 | 32 |
| stage |  |  |  | 15.5 | layer | 16 | 1024 | 4096 | 0.00184 | 72.96 | 32 |
| vendor |  |  |  | vendor | layer | 16 | 1024 | 4096 | 4.829e-05 | 2779 |  |
| tiled_5_3 |  |  |  | 5.3 | layer | 256 | 1024 | 4096 | 0.001787 | 1202 |  |
| previous |  |  |  | 15.4 | layer | 256 | 1024 | 4096 | 0.001346 | 1596 | 32 |
| stage |  |  |  | 15.5 | layer | 256 | 1024 | 4096 | 0.002885 | 744.4 | 32 |
| vendor |  |  |  | vendor | layer | 256 | 1024 | 4096 | 0.0003364 | 6384 |  |
| tiled_5_3 |  |  |  | 5.3 | layer | 1024 | 1024 | 4096 | 0.007401 | 1161 |  |
| previous |  |  |  | 15.4 | layer | 1024 | 1024 | 4096 | 0.003735 | 2300 | 32 |
| stage |  |  |  | 15.5 | layer | 1024 | 1024 | 4096 | 0.009862 | 871 | 32 |
| vendor |  |  |  | vendor | layer | 1024 | 1024 | 4096 | 0.001244 | 6907 |  |
| tiled_5_3 |  |  |  | 5.3 | short_k | 4096 | 128 | 4096 | 0.003985 | 1078 |  |
| previous |  |  |  | 15.4 | short_k | 4096 | 128 | 4096 | 0.002077 | 2068 | 32 |
| stage |  |  |  | 15.5 | short_k | 4096 | 128 | 4096 | 0.00484 | 887.3 | 32 |
| vendor |  |  |  | vendor | short_k | 4096 | 128 | 4096 | 0.0006444 | 6665 |  |

### ch15/lab6_linear, solution, metal, TILE_WIDTH=16

| name | kernel | width | threadExecutionWidth | maxTotalThreadsPerThreadgroup | staticThreadgroupMemoryLength | batch | in | out | seconds | gflops | rounds | tile | library | math |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pipeline | m3 | 32 | 32 | 256 | 17664 |  |  |  |  |  |  |  |  |  |
| m2 | m2 |  |  |  |  | 1 | 1024 | 4096 | 0.0001266 | 66.25 | 100 | 16 |  |  |
| m3 | m3 | 32 |  |  |  | 1 | 1024 | 4096 | 0.00149 | 5.631 | 100 |  |  |  |
| vendor | vendor |  |  |  |  | 1 | 1024 | 4096 | 2.6e-05 | 322.6 | 100 |  | MPSMatrixMultiplication | MPSKernelOptionsAllowReducedPrecision not set (FP32 storage, higher internal precision) |
| m2 | m2 |  |  |  |  | 4 | 1024 | 4096 | 0.000127 | 264.2 | 100 | 16 |  |  |
| m3 | m3 | 32 |  |  |  | 4 | 1024 | 4096 | 0.001492 | 22.49 | 100 |  |  |  |
| vendor | vendor |  |  |  |  | 4 | 1024 | 4096 | 5.4e-05 | 621.4 | 100 |  | MPSMatrixMultiplication | MPSKernelOptionsAllowReducedPrecision not set (FP32 storage, higher internal precision) |
| m2 | m2 |  |  |  |  | 16 | 1024 | 4096 | 0.000131 | 1025 | 100 | 16 |  |  |
| m3 | m3 | 32 |  |  |  | 16 | 1024 | 4096 | 0.001486 | 90.32 | 100 |  |  |  |
| vendor | vendor |  |  |  |  | 16 | 1024 | 4096 | 3.808e-05 | 3524 | 100 |  | MPSMatrixMultiplication | MPSKernelOptionsAllowReducedPrecision not set (FP32 storage, higher internal precision) |
| m2 | m2 |  |  |  |  | 64 | 1024 | 4096 | 0.0004216 | 1273 | 100 | 16 |  |  |
| m3 | m3 | 32 |  |  |  | 64 | 1024 | 4096 | 0.001486 | 361.3 | 100 |  |  |  |
| vendor | vendor |  |  |  |  | 64 | 1024 | 4096 | 9.371e-05 | 5729 | 100 |  | MPSMatrixMultiplication | MPSKernelOptionsAllowReducedPrecision not set (FP32 storage, higher internal precision) |
| m2 | m2 |  |  |  |  | 256 | 1024 | 4096 | 0.001653 | 1299 | 100 | 16 |  |  |
| m3 | m3 | 32 |  |  |  | 256 | 1024 | 4096 | 0.002592 | 828.4 | 100 |  |  |  |
| vendor | vendor |  |  |  |  | 256 | 1024 | 4096 | 0.0003296 | 6515 | 100 |  | MPSMatrixMultiplication | MPSKernelOptionsAllowReducedPrecision not set (FP32 storage, higher internal precision) |
| m2 | m2 |  |  |  |  | 1024 | 1024 | 4096 | 0.008577 | 1001 | 32 | 16 |  |  |
| m3 | m3 | 32 |  |  |  | 1024 | 1024 | 4096 | 0.01115 | 770.2 | 32 |  |  |  |
| vendor | vendor |  |  |  |  | 1024 | 1024 | 4096 | 0.001409 | 6099 | 32 |  | MPSMatrixMultiplication | MPSKernelOptionsAllowReducedPrecision not set (FP32 storage, higher internal precision) |
| m2 | m2 |  |  |  |  | 1 | 4096 | 4096 | 0.0005387 | 62.28 | 100 | 16 |  |  |
| m3 | m3 | 32 |  |  |  | 1 | 4096 | 4096 | 0.006071 | 5.527 | 100 |  |  |  |
| vendor | vendor |  |  |  |  | 1 | 4096 | 4096 | 0.0002255 | 148.8 | 100 |  | MPSMatrixMultiplication | MPSKernelOptionsAllowReducedPrecision not set (FP32 storage, higher internal precision) |
| m2 | m2 |  |  |  |  | 16 | 4096 | 4096 | 0.0005394 | 995.4 | 100 | 16 |  |  |
| m3 | m3 | 32 |  |  |  | 16 | 4096 | 4096 | 0.00599 | 89.62 | 100 |  |  |  |
| vendor | vendor |  |  |  |  | 16 | 4096 | 4096 | 0.0002796 | 1920 | 100 |  | MPSMatrixMultiplication | MPSKernelOptionsAllowReducedPrecision not set (FP32 storage, higher internal precision) |
| m2 | m2 |  |  |  |  | 256 | 4096 | 4096 | 0.006387 | 1345 | 39 | 16 |  |  |
| m3 | m3 | 32 |  |  |  | 256 | 4096 | 4096 | 0.01003 | 856.4 | 39 |  |  |  |
| vendor | vendor |  |  |  |  | 256 | 4096 | 4096 | 0.001292 | 6649 | 39 |  | MPSMatrixMultiplication | MPSKernelOptionsAllowReducedPrecision not set (FP32 storage, higher internal precision) |

### ch16/lab1_wavefront, solution, metal, TILE_WIDTH=16

| name | lseq | tile | pad | launches | seconds | gcups | speedup_vs_baseline |
|---|---|---|---|---|---|---|---|
| baseline | 8192 | 0 | 0 | 16383 | 0.04087 | 1.642 | 1 |
| square | 8192 | 8 | 0 | 2047 | 0.02027 | 3.311 | 2.016 |
| square | 8192 | 8 | 1 | 2047 | 0.0175 | 3.834 | 2.335 |
| square | 8192 | 16 | 0 | 1023 | 0.01753 | 3.829 | 2.332 |
| square | 8192 | 16 | 1 | 1023 | 0.01839 | 3.649 | 2.223 |
| square | 8192 | 32 | 0 | 511 | 0.01513 | 4.434 | 2.701 |
| square | 8192 | 32 | 1 | 511 | 0.01522 | 4.409 | 2.685 |
| square | 8192 | 64 | 0 | 255 | 0.01594 | 4.21 | 2.564 |
| square | 8192 | 64 | 1 | 255 | 0.01895 | 3.542 | 2.157 |
| hyper | 8192 | 8 | 0 | 3071 | 0.0191 | 3.513 | 2.14 |
| hyper | 8192 | 8 | 1 | 3071 | 0.01734 | 3.869 | 2.357 |
| hyper | 8192 | 16 | 0 | 1535 | 0.01348 | 4.978 | 3.032 |
| hyper | 8192 | 16 | 1 | 1535 | 0.01326 | 5.062 | 3.083 |
| hyper | 8192 | 32 | 0 | 767 | 0.01192 | 5.631 | 3.43 |
| hyper | 8192 | 32 | 1 | 767 | 0.01161 | 5.781 | 3.521 |
| hyper | 8192 | 64 | 0 | 383 | 0.01467 | 4.576 | 2.787 |
| hyper | 8192 | 64 | 1 | 383 | 0.01024 | 6.554 | 3.992 |

### ch17/lab1_spmv, solution, metal, TILE_WIDTH=16

| name | matrix | format | rows | nnz | width | stride | coo_entries | seconds | matrix_bytes | pad_fraction | gbps | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spmv | uniform | csr | 2097157 | 33554512 | None | None | 0 | 0.00198 | 276824728 | 0 | 152.5 |  |
| spmv | uniform | ell | 2097157 | 33554512 | 16 | 2097157 | 0 | 0.0009933 | 276824724 | 0 | 304 |  |
| spmv | uniform | ell_aligned | 2097157 | 33554512 | 16 | 2097184 | 0 | 0.0009881 | 276828180 | 1.287e-05 | 305.6 |  |
| spmv | spread | csr | 2097157 | 33551172 | None | None | 0 | 0.002476 | 276798008 | 0 | 122 |  |
| spmv | spread | ell | 2097157 | 33551172 | 31 | 2097157 | 0 | 0.00176 | 528483564 | 0.4839 | 171.6 |  |
| spmv | spread | ell_aligned | 2097157 | 33551172 | 31 | 2097184 | 0 | 0.001782 | 528490260 | 0.4839 | 169.4 |  |
| spmv | spread | hybrid | 2097157 | 33551172 | 16 | 2097157 | 8116995 | 0.00143 | 374228664 | 0.242 | 211.2 | 16 |
| spmv | spread_sorted | csr | 2097157 | 33551172 | None | None | 0 | 0.003343 | 276798008 | 0 | 90.34 |  |
| spmv | spread_sorted | ell | 2097157 | 33551172 | 31 | 2097157 | 0 | 0.001297 | 528483564 | 0.4839 | 232.8 |  |
| spmv | spread_sorted | ell_aligned | 2097157 | 33551172 | 31 | 2097184 | 0 | 0.001347 | 528490260 | 0.4839 | 224.2 |  |
| spmv | skewed | csr | 2097157 | 34537792 | None | None | 0 | 0.001711 | 284690968 | 0 | 181.1 |  |
| spmv | skewed | ell | 2097157 | 34537792 | 256 | 2097157 | 0 | 0.001583 | 4.303e+09 | 0.9357 | 195.8 |  |
| spmv | skewed | ell_aligned | 2097157 | 34537792 | 256 | 2097184 | 0 | 0.001561 | 4.303e+09 | 0.9357 | 198.5 |  |
| spmv | skewed | hybrid | 2097157 | 34537792 | 16 | 2097157 | 983280 | 0.001084 | 288624084 | 0 | 285.9 | 16 |

### ch18/lab1_frontier, solution, metal, TILE_WIDTH=16

| name | graph | vertices | edges | levels | seconds | us_per_level | medges_per_s |
|---|---|---|---|---|---|---|---|
| push | grid | 1048576 | 4190208 | 2046 | 0.4636 | 226.5 | 9.039 |
| frontier | grid | 1048576 | 4190208 | 2046 | 0.3358 | 164.1 | 12.48 |
| privatised | grid | 1048576 | 4190208 | 2046 | 0.3282 | 160.4 | 12.77 |
| floor | grid | 1048576 | 4190208 | 2046 | 0.2988 | 146 | 14.02 |
| push | rmat | 1048576 | 4194304 | 6 | 0.007591 | 1084 | 552.1 |
| frontier | rmat | 1048576 | 4194304 | 6 | 0.01239 | 1770 | 338.1 |
| privatised | rmat | 1048576 | 4194304 | 6 | 0.01374 | 1963 | 304.9 |
| floor | rmat | 1048576 | 4194304 | 6 | 0.001258 | 179.7 | 3332 |

### ch19/lab1_convgemm, solution, metal, TILE_WIDTH=16

| name | shape | n | c | h | w | m | k | seconds | gflops | unroll_seconds | gemm_seconds | scratch_bytes | expansion | gemm_source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| direct | early | 8 | 3 | 224 | 224 | 64 | 3 | 0.002123 | 641.8 |  |  |  |  |  |
| unroll_gemm | early | 8 | 3 | 224 | 224 | 64 | 3 | 0.001691 | 805.8 | 0.0002704 | 0.001364 | 5322672 | 8.84 | m2 reference |
| implicit | early | 8 | 3 | 224 | 224 | 64 | 3 | 0.001714 | 794.8 |  |  |  |  |  |
| direct | middle | 8 | 64 | 56 | 56 | 64 | 3 | 0.003313 | 519.2 |  |  |  |  |  |
| unroll_gemm | middle | 8 | 64 | 56 | 56 | 64 | 3 | 0.001854 | 927.5 | 0.0003643 | 0.001517 | 6718464 | 8.369 | m2 reference |
| implicit | middle | 8 | 64 | 56 | 56 | 64 | 3 | 0.002169 | 792.9 |  |  |  |  |  |
| direct | late | 8 | 256 | 14 | 14 | 256 | 3 | 0.003881 | 350.1 |  |  |  |  |  |
| unroll_gemm | late | 8 | 256 | 14 | 14 | 256 | 3 | 0.001916 | 709.1 | 0.0001232 | 0.001679 | 1327104 | 6.612 | m2 reference |
| implicit | late | 8 | 256 | 14 | 14 | 256 | 3 | 0.002241 | 606.4 |  |  |  |  |  |
| direct | pointwise | 8 | 256 | 14 | 14 | 256 | 1 | 0.001223 | 168 |  |  |  |  |  |
| unroll_gemm | pointwise | 8 | 256 | 14 | 14 | 256 | 1 | 0.0004339 | 473.7 | 5.112e-05 | 0.0003763 | 200704 | 1 | m2 reference |
| implicit | pointwise | 8 | 256 | 14 | 14 | 256 | 1 | 0.0004233 | 485.5 |  |  |  |  |  |
| direct | small | 64 | 6 | 14 | 14 | 16 | 5 | 0.0001259 | 244 |  |  |  |  |  |
| unroll_gemm | small | 64 | 6 | 14 | 14 | 16 | 5 | 0.001256 | 24.45 | 0.0002767 | 0.0009533 | 60000 | 12.76 | m2 reference |
| implicit | small | 64 | 6 | 14 | 14 | 16 | 5 | 8.3e-05 | 370.1 |  |  |  |  |  |

### ch20/lab1_causal_softmax, solution, metal, TILE_WIDTH=16

| name | n | heads | d | softmax_seconds | softmax_gb_per_s | score_bytes | s_seconds | o_seconds | total_seconds | softmax_share |
|---|---|---|---|---|---|---|---|---|---|---|
| unfused_attention | 1024 | 8 | 128 | 0.0001771 | 284.4 | 67108864 | 0.002888 | 0.001818 | 0.004883 | 0.03626 |
| unfused_attention | 2048 | 8 | 128 | 0.0008361 | 240.8 | 268435456 | 0.009338 | 0.008023 | 0.0182 | 0.04594 |
| unfused_attention | 4096 | 8 | 128 | 0.003417 | 235.7 | 1.074e+09 | 0.06093 | 0.05445 | 0.1188 | 0.02877 |

### ch20/lab2_flash, solution, metal, TILE_WIDTH=32

| name | n | heads | d | b_c | flash_seconds | flash_gflops | unfused_score_bytes | unfused_seconds | unfused_gflops | flash_over_unfused |
|---|---|---|---|---|---|---|---|---|---|---|
| attention | 512 | 8 | 128 | 16 | 0.001958 | 274.3 | 16777216 | 0.001326 | 404.9 | 1.476 |
| attention | 1024 | 8 | 128 | 16 | 0.006352 | 338.1 | 67108864 | 0.00493 | 435.6 | 1.289 |
| attention | 2048 | 8 | 128 | 16 | 0.02361 | 363.9 | 268435456 | 0.017 | 505.4 | 1.389 |
| attention | 4096 | 8 | 128 | 16 | 0.1092 | 314.5 | 1.074e+09 | 0.07513 | 457.3 | 1.454 |

### ch20/lab4_decode, solution, metal, TILE_WIDTH=16

| name | batch | len | h_q | h_kv | head_dim | cache_bytes | seconds | median_seconds | ms | gb_per_s | bytes_moved | us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 1 | 512 | 32 | 32 | 128 | 16777216 | 7.95e-05 | 8.15e-05 | 0.0795 | 211 |  |  |
| decode | 1 | 2048 | 32 | 32 | 128 | 67108864 | 0.0002247 | 0.000343 | 0.2247 | 298.7 |  |  |
| decode | 1 | 8192 | 32 | 32 | 128 | 268435456 | 0.00105 | 0.001205 | 1.05 | 255.6 |  |  |
| decode | 1 | 32768 | 32 | 32 | 128 | 1.074e+09 | 0.003835 | 0.003857 | 3.835 | 280 |  |  |
| decode | 8 | 512 | 32 | 32 | 128 | 134217728 | 0.0004511 | 0.0004591 | 0.4511 | 297.5 |  |  |
| decode | 8 | 2048 | 32 | 32 | 128 | 536870912 | 0.001836 | 0.001853 | 1.836 | 292.4 |  |  |
| decode | 8 | 8192 | 32 | 32 | 128 | 2.147e+09 | 0.007394 | 0.007405 | 7.394 | 290.4 |  |  |
| decode | 8 | 32768 | 32 | 32 | 128 | 8.59e+09 | 0.03043 | 0.03112 | 30.43 | 282.3 |  |  |
| append | 1 |  |  | 32 |  |  | 5.375e-06 |  |  |  | 65536 | 5.375 |
| append | 8 |  |  | 32 |  |  | 4.417e-06 |  |  |  | 524288 | 4.417 |

### ch20/lab5_gqa_decode, solution, metal, TILE_WIDTH=16

| name | batch | len | h_q | h_kv | g_q | blocks | cache_bytes | seconds | median_seconds | ms | gb_per_s | speedup_vs_perhead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| perhead | 1 | 8192 | 32 | 32 | 1 | 32 | 268435456 | 0.0009588 | 0.001209 | 0.9588 | 280 |  |
| grouped | 1 | 8192 | 32 | 32 | 1 | 32 | 268435456 | 0.0009571 | 0.0009647 | 0.9571 | 280.5 | 1.002 |
| perhead | 1 | 8192 | 32 | 8 | 4 | 32 | 67108864 | 0.000599 | 0.0007235 | 0.599 | 112 |  |
| grouped | 1 | 8192 | 32 | 8 | 4 | 8 | 67108864 | 0.0009212 | 0.0009834 | 0.9212 | 72.85 | 0.6502 |
| perhead | 1 | 8192 | 32 | 4 | 8 | 32 | 33554432 | 0.0004146 | 0.0004247 | 0.4146 | 80.93 |  |
| grouped | 1 | 8192 | 32 | 4 | 8 | 4 | 33554432 | 0.00133 | 0.001333 | 1.33 | 25.24 | 0.3118 |
| perhead | 4 | 8192 | 32 | 32 | 1 | 128 | 1.074e+09 | 0.003824 | 0.003843 | 3.824 | 280.8 |  |
| grouped | 4 | 8192 | 32 | 32 | 1 | 128 | 1.074e+09 | 0.003838 | 0.003851 | 3.838 | 279.8 | 0.9966 |
| perhead | 4 | 8192 | 32 | 8 | 4 | 128 | 268435456 | 0.001192 | 0.001208 | 1.192 | 225.3 |  |
| grouped | 4 | 8192 | 32 | 8 | 4 | 32 | 268435456 | 0.0009421 | 0.0009475 | 0.9421 | 284.9 | 1.265 |
| perhead | 4 | 8192 | 32 | 4 | 8 | 128 | 134217728 | 0.0009757 | 0.0009785 | 0.9757 | 137.6 |  |
| grouped | 4 | 8192 | 32 | 4 | 8 | 16 | 134217728 | 0.001354 | 0.001358 | 1.354 | 99.11 | 0.7204 |

### ch21/lab1_dcs, solution, metal, TILE_WIDTH=16

| name | comparison | nx | ny | atoms | chunk | seconds | pairs_per_s | group_width |
|---|---|---|---|---|---|---|---|---|
| scatter | A | 256 | 256 | 1024 | 256 | 0.01335 | 5.028e+09 | 32 |
| gather | A | 256 | 256 | 1024 | 256 | 0.0002569 | 2.613e+11 | 32 |
| gather | B | 1024 | 1024 | 4096 | 64 | 0.01211 | 3.546e+11 | 32 |
| coarse_adjacent | B | 1024 | 1024 | 4096 | 64 | 0.005355 | 8.02e+11 | 32 |
| coarse_strided | B | 1024 | 1024 | 4096 | 64 | 0.005401 | 7.953e+11 | 32 |
| gather | B | 1024 | 1024 | 4096 | 256 | 0.0116 | 3.704e+11 | 32 |
| coarse_adjacent | B | 1024 | 1024 | 4096 | 256 | 0.004955 | 8.669e+11 | 32 |
| coarse_strided | B | 1024 | 1024 | 4096 | 256 | 0.005048 | 8.507e+11 | 32 |
| gather | B | 1024 | 1024 | 4096 | 512 | 0.01151 | 3.731e+11 | 32 |
| coarse_adjacent | B | 1024 | 1024 | 4096 | 512 | 0.004878 | 8.805e+11 | 32 |
| coarse_strided | B | 1024 | 1024 | 4096 | 512 | 0.004952 | 8.673e+11 | 32 |
| gather | B | 1024 | 1024 | 4096 | 1024 | 0.01143 | 3.758e+11 | 32 |
| coarse_adjacent | B | 1024 | 1024 | 4096 | 1024 | 0.004831 | 8.89e+11 | 32 |
| coarse_strided | B | 1024 | 1024 | 4096 | 1024 | 0.0049 | 8.766e+11 | 32 |

### ch22/lab1_gemv, solution, metal, TILE_WIDTH=16

| name | kernel | splits | batch | in | out | seconds | weight_gb_per_s |
|---|---|---|---|---|---|---|---|
| gemv | gather | 1 | 1 | 4096 | 4096 | 0.0003224 | 208.2 |
| gemv | splitk | 2 | 1 | 4096 | 4096 | 0.0002703 | 248.3 |
| gemv | splitk | 4 | 1 | 4096 | 4096 | 0.0002497 | 268.7 |
| gemv | splitk | 8 | 1 | 4096 | 4096 | 0.0002419 | 277.4 |
| gemv | splitk | 16 | 1 | 4096 | 4096 | 0.0002615 | 256.7 |
| gemv | gather | 1 | 4 | 4096 | 4096 | 0.0005049 | 132.9 |
| gemv | splitk | 2 | 4 | 4096 | 4096 | 0.0003232 | 207.6 |
| gemv | splitk | 4 | 4 | 4096 | 4096 | 0.0002707 | 247.9 |
| gemv | splitk | 8 | 4 | 4096 | 4096 | 0.0002648 | 253.4 |
| gemv | splitk | 16 | 4 | 4096 | 4096 | 0.0002845 | 235.9 |
| gemv | gather | 1 | 8 | 4096 | 4096 | 0.0008324 | 80.62 |
| gemv | splitk | 2 | 8 | 4096 | 4096 | 0.000461 | 145.6 |
| gemv | splitk | 4 | 8 | 4096 | 4096 | 0.0003447 | 194.7 |
| gemv | splitk | 8 | 8 | 4096 | 4096 | 0.0003175 | 211.4 |
| gemv | splitk | 16 | 8 | 4096 | 4096 | 0.0003277 | 204.8 |
| gemv | gather | 1 | 1 | 4096 | 16 | 7.492e-05 | 3.499 |
| gemv | splitk | 2 | 1 | 4096 | 16 | 4.012e-05 | 6.533 |
| gemv | splitk | 4 | 1 | 4096 | 16 | 2.258e-05 | 11.61 |
| gemv | splitk | 8 | 1 | 4096 | 16 | 1.173e-05 | 22.35 |
| gemv | splitk | 16 | 1 | 4096 | 16 | 7.672e-06 | 34.17 |
| gemv | gather | 1 | 4 | 4096 | 16 | 0.0001107 | 2.368 |
| gemv | splitk | 2 | 4 | 4096 | 16 | 5.908e-05 | 4.437 |
| gemv | splitk | 4 | 4 | 4096 | 16 | 3.292e-05 | 7.962 |
| gemv | splitk | 8 | 4 | 4096 | 16 | 1.687e-05 | 15.54 |
| gemv | splitk | 16 | 4 | 4096 | 16 | 1.023e-05 | 25.61 |
| gemv | gather | 1 | 8 | 4096 | 16 | 0.0001603 | 1.636 |
| gemv | splitk | 2 | 8 | 4096 | 16 | 8.527e-05 | 3.074 |
| gemv | splitk | 4 | 8 | 4096 | 16 | 4.669e-05 | 5.615 |
| gemv | splitk | 8 | 8 | 4096 | 16 | 2.385e-05 | 10.99 |
| gemv | splitk | 16 | 8 | 4096 | 16 | 1.496e-05 | 17.52 |
| gemv | gather | 1 | 1 | 14336 | 4096 | 0.001307 | 179.7 |
| gemv | splitk | 2 | 1 | 14336 | 4096 | 0.0009894 | 237.4 |
| gemv | splitk | 4 | 1 | 14336 | 4096 | 0.0008663 | 271.1 |
| gemv | splitk | 8 | 1 | 14336 | 4096 | 0.0008277 | 283.8 |
| gemv | splitk | 16 | 1 | 14336 | 4096 | 0.0008037 | 292.2 |
| gemv | splitk | 32 | 1 | 14336 | 4096 | 0.0008172 | 287.4 |
| gemv | gather | 1 | 4 | 14336 | 4096 | 0.002173 | 108.1 |
| gemv | splitk | 2 | 4 | 14336 | 4096 | 0.001484 | 158.3 |
| gemv | splitk | 4 | 4 | 14336 | 4096 | 0.001082 | 217 |
| gemv | splitk | 8 | 4 | 14336 | 4096 | 0.0009331 | 251.7 |
| gemv | splitk | 16 | 4 | 14336 | 4096 | 0.0009025 | 260.3 |
| gemv | splitk | 32 | 4 | 14336 | 4096 | 0.0009236 | 254.3 |
| gemv | gather | 1 | 8 | 14336 | 4096 | 0.004345 | 54.06 |
| gemv | splitk | 2 | 8 | 14336 | 4096 | 0.002684 | 87.51 |
| gemv | splitk | 4 | 8 | 14336 | 4096 | 0.001751 | 134.1 |
| gemv | splitk | 8 | 8 | 14336 | 4096 | 0.001341 | 175.2 |
| gemv | splitk | 16 | 8 | 14336 | 4096 | 0.001195 | 196.6 |
| gemv | splitk | 32 | 8 | 14336 | 4096 | 0.001161 | 202.3 |

### ch22/lab2_batching, solution, metal, TILE_WIDTH=16

| name | gemm | batch | seconds | median_s | tokens_per_s | weight_bytes | weight_gb_per_s | model_flop_per_byte | d | layers | vocab | piece |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step | m2 | 1 | 0.004169 | 0.004214 | 239.9 | 658538496 | 158 | 0.4994 | 4096 | 2 | 32000 |  |
| step | m2 | 2 | 0.004169 | 0.004191 | 479.7 | 658538496 | 158 | 0.9976 | 4096 | 2 | 32000 |  |
| step | m2 | 4 | 0.004188 | 0.004672 | 955.1 | 658538496 | 157.2 | 1.99 | 4096 | 2 | 32000 |  |
| step | m2 | 8 | 0.004834 | 0.005119 | 1655 | 658538496 | 136.2 | 3.962 | 4096 | 2 | 32000 |  |
| step | m2 | 16 | 0.004523 | 0.004734 | 3538 | 658538496 | 145.6 | 7.849 | 4096 | 2 | 32000 |  |
| step | m2 | 32 | 0.008095 | 0.008557 | 3953 | 658538496 | 81.36 | 15.41 | 4096 | 2 | 32000 |  |
| step | m2 | 64 | 0.01571 | 0.01659 | 4073 | 658538496 | 41.91 | 29.72 | 4096 | 2 | 32000 |  |
| step | m2 | 128 | 0.03122 | 0.03222 | 4100 | 658538496 | 21.09 | 55.48 | 4096 | 2 | 32000 |  |
| step | m2 | 256 | 0.06446 | 0.06492 | 3971 | 658538496 | 10.22 | 97.91 | 4096 | 2 | 32000 |  |
| step | m2 | 512 | 0.1252 | 0.1276 | 4090 | 658538496 | 5.261 | 158.6 | 4096 | 2 | 32000 |  |
| step | m2 | 1024 | 0.2546 | 0.2554 | 4022 | 658538496 | 2.587 | 229.7 | 4096 | 2 | 32000 |  |
| step | m3 | 1 | 0.04644 | 0.04834 | 21.53 | 658538496 | 14.18 | 0.4994 | 4096 | 2 | 32000 |  |
| step | m3 | 2 | 0.04629 | 0.04813 | 43.21 | 658538496 | 14.23 | 0.9976 | 4096 | 2 | 32000 |  |
| step | m3 | 4 | 0.04597 | 0.04782 | 87.01 | 658538496 | 14.32 | 1.99 | 4096 | 2 | 32000 |  |
| step | m3 | 8 | 0.04602 | 0.04777 | 173.8 | 658538496 | 14.31 | 3.962 | 4096 | 2 | 32000 |  |
| step | m3 | 16 | 0.04647 | 0.04831 | 344.3 | 658538496 | 14.17 | 7.849 | 4096 | 2 | 32000 |  |
| step | m3 | 32 | 0.04618 | 0.04767 | 693 | 658538496 | 14.26 | 15.41 | 4096 | 2 | 32000 |  |
| step | m3 | 64 | 0.04607 | 0.04796 | 1389 | 658538496 | 14.29 | 29.72 | 4096 | 2 | 32000 |  |
| step | m3 | 128 | 0.04643 | 0.04875 | 2757 | 658538496 | 14.18 | 55.48 | 4096 | 2 | 32000 |  |
| step | m3 | 256 | 0.08417 | 0.08632 | 3041 | 658538496 | 7.824 | 97.91 | 4096 | 2 | 32000 |  |
| step | m3 | 512 | 0.1573 | 0.1612 | 3255 | 658538496 | 4.187 | 158.6 | 4096 | 2 | 32000 |  |
| step | m3 | 1024 | 0.3258 | 0.3274 | 3143 | 658538496 | 2.021 | 229.7 | 4096 | 2 | 32000 |  |
| step | gemv | 1 | 0.002585 | 0.00263 | 386.8 | 658538496 | 254.7 | 0.4994 | 4096 | 2 | 32000 |  |
| step | gemv | 2 | 0.002656 | 0.002703 | 753.1 | 658538496 | 248 | 0.9976 | 4096 | 2 | 32000 |  |
| step | gemv | 4 | 0.002885 | 0.002915 | 1387 | 658538496 | 228.3 | 1.99 | 4096 | 2 | 32000 |  |
| step | gemv | 8 | 0.004143 | 0.004208 | 1931 | 658538496 | 158.9 | 3.962 | 4096 | 2 | 32000 |  |
| one_by_one | m2 | 1 | 0.004178 |  | 239.3 |  |  |  |  |  |  |  |
| one_by_one | m2 | 2 | 0.008379 |  | 238.7 |  |  |  |  |  |  |  |
| one_by_one | m2 | 4 | 0.01675 |  | 238.8 |  |  |  |  |  |  |  |
| one_by_one | m2 | 8 | 0.03367 |  | 237.6 |  |  |  |  |  |  |  |
| one_by_one | m2 | 16 | 0.06886 |  | 232.4 |  |  |  |  |  |  |  |
| one_by_one | m2 | 32 | 0.1395 |  | 229.3 |  |  |  |  |  |  |  |
| one_by_one | m3 | 1 | 0.04673 |  | 21.4 |  |  |  |  |  |  |  |
| one_by_one | m3 | 2 | 0.09225 |  | 21.68 |  |  |  |  |  |  |  |
| one_by_one | m3 | 4 | 0.1887 |  | 21.2 |  |  |  |  |  |  |  |
| one_by_one | m3 | 8 | 0.3812 |  | 20.99 |  |  |  |  |  |  |  |
| one_by_one | m3 | 16 | 0.7623 |  | 20.99 |  |  |  |  |  |  |  |
| one_by_one | m3 | 32 | 1.529 |  | 20.93 |  |  |  |  |  |  |  |
| one_by_one | gemv | 1 | 0.002595 |  | 385.3 |  |  |  |  |  |  |  |
| one_by_one | gemv | 2 | 0.005197 |  | 384.9 |  |  |  |  |  |  |  |
| one_by_one | gemv | 4 | 0.01042 |  | 383.8 |  |  |  |  |  |  |  |
| one_by_one | gemv | 8 | 0.02076 |  | 385.3 |  |  |  |  |  |  |  |
| one_by_one | gemv | 16 | 0.04181 |  | 382.7 |  |  |  |  |  |  |  |
| one_by_one | gemv | 32 | 0.0851 |  | 376 |  |  |  |  |  |  |  |
| piece | m2 | 1 | 3e-06 |  |  |  |  |  |  |  |  | embed |
| piece | m2 | 1 | 0.0005138 |  |  |  |  |  |  |  |  | linear |
| piece | m2 | 1 | 2.958e-06 |  |  |  |  |  |  |  |  | bias |
| piece | m2 | 1 | 2.542e-06 |  |  |  |  |  |  |  |  | residual |
| piece | m2 | 1 | 0.003078 |  |  |  |  |  |  |  |  | lm_head |
| piece | m2 | 1 | 5.104e-05 |  |  |  |  |  |  |  |  | softmax |
| piece | m2 | 1 | 3.25e-05 |  |  |  |  |  |  |  |  | argmax |
| piece | m2 | 16 | 4e-06 |  |  |  |  |  |  |  |  | embed |
| piece | m2 | 16 | 0.0005344 |  |  |  |  |  |  |  |  | linear |
| piece | m2 | 16 | 3.75e-06 |  |  |  |  |  |  |  |  | bias |
| piece | m2 | 16 | 3.625e-06 |  |  |  |  |  |  |  |  | residual |
| piece | m2 | 16 | 0.003163 |  |  |  |  |  |  |  |  | lm_head |
| piece | m2 | 16 | 5.192e-05 |  |  |  |  |  |  |  |  | softmax |
| piece | m2 | 16 | 3.329e-05 |  |  |  |  |  |  |  |  | argmax |
| piece | m2 | 256 | 1.663e-05 |  |  |  |  |  |  |  |  | embed |
| piece | m2 | 256 | 0.006273 |  |  |  |  |  |  |  |  | linear |
| piece | m2 | 256 | 1.354e-05 |  |  |  |  |  |  |  |  | bias |
| piece | m2 | 256 | 1.642e-05 |  |  |  |  |  |  |  |  | residual |
| piece | m2 | 256 | 0.04903 |  |  |  |  |  |  |  |  | lm_head |
| piece | m2 | 256 | 0.0004467 |  |  |  |  |  |  |  |  | softmax |
| piece | m2 | 256 | 0.0001072 |  |  |  |  |  |  |  |  | argmax |
| piece | m2 | 1024 | 7.425e-05 |  |  |  |  |  |  |  |  | embed |
| piece | m2 | 1024 | 0.02474 |  |  |  |  |  |  |  |  | linear |
| piece | m2 | 1024 | 4.213e-05 |  |  |  |  |  |  |  |  | bias |
| piece | m2 | 1024 | 0.0001093 |  |  |  |  |  |  |  |  | residual |
| piece | m2 | 1024 | 0.1979 |  |  |  |  |  |  |  |  | lm_head |
| piece | m2 | 1024 | 0.001814 |  |  |  |  |  |  |  |  | softmax |
| piece | m2 | 1024 | 0.0004605 |  |  |  |  |  |  |  |  | argmax |
| piece | m3 | 1 | 3e-06 |  |  |  |  |  |  |  |  | embed |
| piece | m3 | 1 | 0.006008 |  |  |  |  |  |  |  |  | linear |
| piece | m3 | 1 | 2.875e-06 |  |  |  |  |  |  |  |  | bias |
| piece | m3 | 1 | 2.5e-06 |  |  |  |  |  |  |  |  | residual |
| piece | m3 | 1 | 0.03371 |  |  |  |  |  |  |  |  | lm_head |
| piece | m3 | 1 | 5.104e-05 |  |  |  |  |  |  |  |  | softmax |
| piece | m3 | 1 | 3.292e-05 |  |  |  |  |  |  |  |  | argmax |
| piece | m3 | 16 | 4e-06 |  |  |  |  |  |  |  |  | embed |
| piece | m3 | 16 | 0.006004 |  |  |  |  |  |  |  |  | linear |
| piece | m3 | 16 | 3.708e-06 |  |  |  |  |  |  |  |  | bias |
| piece | m3 | 16 | 3.5e-06 |  |  |  |  |  |  |  |  | residual |
| piece | m3 | 16 | 0.03367 |  |  |  |  |  |  |  |  | lm_head |
| piece | m3 | 16 | 5.262e-05 |  |  |  |  |  |  |  |  | softmax |
| piece | m3 | 16 | 3.387e-05 |  |  |  |  |  |  |  |  | argmax |
| piece | m3 | 256 | 1.662e-05 |  |  |  |  |  |  |  |  | embed |
| piece | m3 | 256 | 0.009596 |  |  |  |  |  |  |  |  | linear |
| piece | m3 | 256 | 1.321e-05 |  |  |  |  |  |  |  |  | bias |
| piece | m3 | 256 | 1.65e-05 |  |  |  |  |  |  |  |  | residual |
| piece | m3 | 256 | 0.06197 |  |  |  |  |  |  |  |  | lm_head |
| piece | m3 | 256 | 0.0004342 |  |  |  |  |  |  |  |  | softmax |
| piece | m3 | 256 | 0.0001073 |  |  |  |  |  |  |  |  | argmax |
| piece | m3 | 1024 | 7.412e-05 |  |  |  |  |  |  |  |  | embed |
| piece | m3 | 1024 | 0.0339 |  |  |  |  |  |  |  |  | linear |
| piece | m3 | 1024 | 4.208e-05 |  |  |  |  |  |  |  |  | bias |
| piece | m3 | 1024 | 0.000109 |  |  |  |  |  |  |  |  | residual |
| piece | m3 | 1024 | 0.2445 |  |  |  |  |  |  |  |  | lm_head |
| piece | m3 | 1024 | 0.001818 |  |  |  |  |  |  |  |  | softmax |
| piece | m3 | 1024 | 0.0004585 |  |  |  |  |  |  |  |  | argmax |
| piece | gemv | 1 | 3e-06 |  |  |  |  |  |  |  |  | embed |
| piece | gemv | 1 | 0.0003227 |  |  |  |  |  |  |  |  | linear |
| piece | gemv | 1 | 2.917e-06 |  |  |  |  |  |  |  |  | bias |
| piece | gemv | 1 | 2.542e-06 |  |  |  |  |  |  |  |  | residual |
| piece | gemv | 1 | 0.001809 |  |  |  |  |  |  |  |  | lm_head |
| piece | gemv | 1 | 5.112e-05 |  |  |  |  |  |  |  |  | softmax |
| piece | gemv | 1 | 3.25e-05 |  |  |  |  |  |  |  |  | argmax |
| piece | gemv | 8 | 3.542e-06 |  |  |  |  |  |  |  |  | embed |
| piece | gemv | 8 | 0.0007105 |  |  |  |  |  |  |  |  | linear |
| piece | gemv | 8 | 3.25e-06 |  |  |  |  |  |  |  |  | bias |
| piece | gemv | 8 | 3.042e-06 |  |  |  |  |  |  |  |  | residual |
| piece | gemv | 8 | 0.002952 |  |  |  |  |  |  |  |  | lm_head |
| piece | gemv | 8 | 5.233e-05 |  |  |  |  |  |  |  |  | softmax |
| piece | gemv | 8 | 3.379e-05 |  |  |  |  |  |  |  |  | argmax |

### ch23/lab1_overlap, solution, metal, TILE_WIDTH=16

| name | create_s | fresh_fill_s | fill_s | bytes | kernel_s | drain_s | degree | host | seconds | gpu_s | buffers | chunks | overlap_observed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| firsttouch | 1.8e-05 | 0.01322 | 0.004279 | 268435456 |  |  |  |  |  |  |  |  |  |
| parts |  |  | 0.004279 | 268435456 | 0.009776 | 0.00427 | 128 |  |  |  |  |  |  |
| serial |  |  |  | 268435456 |  |  | 128 | shared | 0.01927 | 0.009777 |  |  |  |
| inflight |  |  |  | 268435456 |  |  | 128 |  | 0.02134 | 0.009824 | 1 | 4 | False |
| inflight |  |  |  | 268435456 |  |  | 128 |  | 0.02325 | 0.009985 | 1 | 16 | False |
| inflight |  |  |  | 268435456 |  |  | 128 |  | 0.03298 | 0.01044 | 1 | 64 | False |
| inflight |  |  |  | 268435456 |  |  | 128 |  | 0.0136 | 0.01215 | 2 | 4 | True |
| inflight |  |  |  | 268435456 |  |  | 128 |  | 0.01259 | 0.011 | 2 | 16 | True |
| inflight |  |  |  | 268435456 |  |  | 128 |  | 0.01808 | 0.01372 | 2 | 64 | True |
| inflight |  |  |  | 268435456 |  |  | 128 |  | 0.01261 | 0.01603 | 3 | 4 | True |
| inflight |  |  |  | 268435456 |  |  | 128 |  | 0.0108 | 0.01307 | 3 | 16 | True |
| inflight |  |  |  | 268435456 |  |  | 128 |  | 0.01145 | 0.0111 | 3 | 64 | True |
| inflight |  |  |  | 268435456 |  |  | 128 |  | 0.01264 | 0.01703 | 4 | 4 | True |
| inflight |  |  |  | 268435456 |  |  | 128 |  | 0.01069 | 0.01904 | 4 | 16 | True |
| inflight |  |  |  | 268435456 |  |  | 128 |  | 0.01135 | 0.01262 | 4 | 64 | True |

### x1/lab1_topk, solution, metal, TILE_WIDTH=16

| name | rows | cols | k | seconds | rows_per_s | bytes_per_key | calls | sort_over_select | cpu_over_select |
|---|---|---|---|---|---|---|---|---|---|
| topk_select | 1 | 32000 | 1 | 9.217e-05 | 1.085e+04 | 20 | 32 |  |  |
| full_sort_then_first_k | 1 | 32000 | 1 | 0.0009871 | 1013 | 80 | 32 | 10.71 |  |
| cpu_partial_sort | 1 | 32000 | 1 | 1.249e-05 | 8.009e+04 |  |  |  | 0.1355 |
| topk_select | 1 | 32000 | 40 | 0.0002193 | 4561 | 20.01 | 16 |  |  |
| full_sort_then_first_k | 1 | 32000 | 40 | 0.0009871 | 1013 | 80 | 32 | 4.502 |  |
| cpu_partial_sort | 1 | 32000 | 40 | 1.766e-05 | 5.664e+04 |  |  |  | 0.08052 |
| topk_select | 1 | 32000 | 1024 | 0.0002479 | 4034 | 20.26 | 16 |  |  |
| full_sort_then_first_k | 1 | 32000 | 1024 | 0.0009871 | 1013 | 80 | 32 | 3.982 |  |
| cpu_partial_sort | 1 | 32000 | 1024 | 8.39e-05 | 1.192e+04 |  |  |  | 0.3385 |
| topk_select | 8 | 32000 | 1 | 0.0002326 | 3.44e+04 | 20 | 16 |  |  |
| full_sort_then_first_k | 8 | 32000 | 1 | 0.001 | 7999 | 80 | 32 | 4.3 |  |
| cpu_partial_sort | 8 | 32000 | 1 | 9.651e-05 | 8.289e+04 |  |  |  | 0.415 |
| topk_select | 8 | 32000 | 40 | 0.0002448 | 3.269e+04 | 20.01 | 16 |  |  |
| full_sort_then_first_k | 8 | 32000 | 40 | 0.001 | 7999 | 80 | 32 | 4.086 |  |
| cpu_partial_sort | 8 | 32000 | 40 | 0.0001474 | 5.427e+04 |  |  |  | 0.6023 |
| topk_select | 8 | 32000 | 1024 | 0.0002473 | 3.234e+04 | 20.26 | 16 |  |  |
| full_sort_then_first_k | 8 | 32000 | 1024 | 0.001 | 7999 | 80 | 32 | 4.043 |  |
| cpu_partial_sort | 8 | 32000 | 1024 | 0.001074 | 7449 |  |  |  | 4.342 |
| topk_select | 64 | 32000 | 1 | 0.0002995 | 2.137e+05 | 20 | 8 |  |  |
| full_sort_then_first_k | 64 | 32000 | 1 | 0.001629 | 3.929e+04 | 80 | 16 | 5.439 |  |
| cpu_partial_sort | 64 | 32000 | 1 | 0.0007753 | 8.255e+04 |  |  |  | 2.588 |
| topk_select | 64 | 32000 | 40 | 0.0003681 | 1.739e+05 | 20.01 | 4 |  |  |
| full_sort_then_first_k | 64 | 32000 | 40 | 0.001629 | 3.929e+04 | 80 | 16 | 4.426 |  |
| cpu_partial_sort | 64 | 32000 | 40 | 0.001241 | 5.158e+04 |  |  |  | 3.371 |
| topk_select | 64 | 32000 | 1024 | 0.0003118 | 2.052e+05 | 20.26 | 8 |  |  |
| full_sort_then_first_k | 64 | 32000 | 1024 | 0.001629 | 3.929e+04 | 80 | 16 | 5.224 |  |
| cpu_partial_sort | 64 | 32000 | 1024 | 0.01021 | 6269 |  |  |  | 32.74 |
| topk_select | 1 | 151936 | 1 | 0.00111 | 900.7 | 20 | 2 |  |  |
| full_sort_then_first_k | 1 | 151936 | 1 | 0.004546 | 220 | 80 | 8 | 4.095 |  |
| cpu_partial_sort | 1 | 151936 | 1 | 5.788e-05 | 1.728e+04 |  |  |  | 0.05214 |
| topk_select | 1 | 151936 | 40 | 0.001245 | 803.3 | 20 | 2 |  |  |
| full_sort_then_first_k | 1 | 151936 | 40 | 0.004546 | 220 | 80 | 8 | 3.652 |  |
| cpu_partial_sort | 1 | 151936 | 40 | 6.681e-05 | 1.497e+04 |  |  |  | 0.05367 |
| topk_select | 1 | 151936 | 1024 | 0.001178 | 849.2 | 20.05 | 2 |  |  |
| full_sort_then_first_k | 1 | 151936 | 1024 | 0.004546 | 220 | 80 | 8 | 3.861 |  |
| cpu_partial_sort | 1 | 151936 | 1024 | 0.0001902 | 5259 |  |  |  | 0.1615 |
| topk_select | 8 | 151936 | 1 | 0.00122 | 6560 | 20 | 2 |  |  |
| full_sort_then_first_k | 8 | 151936 | 1 | 0.005248 | 1524 | 80 | 4 | 4.303 |  |
| cpu_partial_sort | 8 | 151936 | 1 | 0.0004527 | 1.767e+04 |  |  |  | 0.3712 |
| topk_select | 8 | 151936 | 40 | 0.001928 | 4149 | 20 | 1 |  |  |
| full_sort_then_first_k | 8 | 151936 | 40 | 0.005248 | 1524 | 80 | 4 | 2.722 |  |
| cpu_partial_sort | 8 | 151936 | 40 | 0.0005268 | 1.519e+04 |  |  |  | 0.2732 |
| topk_select | 8 | 151936 | 1024 | 0.001336 | 5988 | 20.05 | 1 |  |  |
| full_sort_then_first_k | 8 | 151936 | 1024 | 0.005248 | 1524 | 80 | 4 | 3.928 |  |
| cpu_partial_sort | 8 | 151936 | 1024 | 0.002095 | 3818 |  |  |  | 1.568 |
| topk_select | 64 | 151936 | 1 | 0.001813 | 3.531e+04 | 20 | 2 |  |  |
| full_sort_then_first_k | 64 | 151936 | 1 | 0.009338 | 6853 | 80 | 4 | 5.152 |  |
| cpu_partial_sort | 64 | 151936 | 1 | 0.003634 | 1.761e+04 |  |  |  | 2.005 |
| topk_select | 64 | 151936 | 40 | 0.00255 | 2.51e+04 | 20 | 1 |  |  |
| full_sort_then_first_k | 64 | 151936 | 40 | 0.009338 | 6853 | 80 | 4 | 3.662 |  |
| cpu_partial_sort | 64 | 151936 | 40 | 0.004284 | 1.494e+04 |  |  |  | 1.68 |
| topk_select | 64 | 151936 | 1024 | 0.002555 | 2.505e+04 | 20.05 | 1 |  |  |
| full_sort_then_first_k | 64 | 151936 | 1024 | 0.009338 | 6853 | 80 | 4 | 3.655 |  |
| cpu_partial_sort | 64 | 151936 | 1024 | 0.01767 | 3622 |  |  |  | 6.915 |

### x1/lab2_filters, solution, metal, TILE_WIDTH=16

| name | rows | cols | k | seconds | sort_share | gb_per_s | bytes_per_token | speedup_vs_unfused |
|---|---|---|---|---|---|---|---|---|
| sort_rows | 1 | 40 | 40 | 7.012e-05 |  |  |  |  |
| chain_composed | 1 | 40 | 40 | 5.417e-05 | 1.294 |  |  |  |
| filters_unfused | 1 | 40 | 40 | 7.862e-06 |  | 0.1475 | 29 |  |
| filters_fused | 1 | 40 | 40 | 3.55e-06 |  | 0.1352 | 12 | 2.215 |
| sort_rows | 64 | 40 | 40 | 5.631e-05 |  |  |  |  |
| chain_composed | 64 | 40 | 40 | 5.463e-05 | 1.031 |  |  |  |
| filters_unfused | 64 | 40 | 40 | 8.691e-06 |  | 8.542 | 29 |  |
| filters_fused | 64 | 40 | 40 | 2.843e-06 |  | 10.81 | 12 | 3.057 |
| sort_rows | 1 | 1024 | 1024 | 6.405e-05 |  |  |  |  |
| chain_composed | 1 | 1024 | 1024 | 7.78e-05 | 0.8232 |  |  |  |
| filters_unfused | 1 | 1024 | 1024 | 8.992e-06 |  | 3.303 | 29 |  |
| filters_fused | 1 | 1024 | 1024 | 3.202e-06 |  | 3.837 | 12 | 2.808 |
| sort_rows | 64 | 1024 | 1024 | 6.511e-05 |  |  |  |  |
| chain_composed | 64 | 1024 | 1024 | 8.813e-05 | 0.7388 |  |  |  |
| filters_unfused | 64 | 1024 | 1024 | 9.405e-06 |  | 202.1 | 29 |  |
| filters_fused | 64 | 1024 | 1024 | 5.069e-06 |  | 155.2 | 12 | 1.856 |
| sort_rows | 1 | 32000 | 0 | 0.0009988 |  |  |  |  |
| chain_composed | 1 | 32000 | 0 | 0.001094 | 0.9127 |  |  |  |
| filters_unfused | 1 | 32000 | 0 | 0.0001053 |  | 8.811 | 29 |  |
| filters_fused | 1 | 32000 | 0 | 4.819e-05 |  | 7.968 | 12 | 2.185 |
| sort_rows | 64 | 32000 | 0 | 0.002174 |  |  |  |  |
| chain_composed | 64 | 32000 | 0 | 0.001905 | 1.141 |  |  |  |
| filters_unfused | 64 | 32000 | 0 | 0.0001879 |  | 316.2 | 29 |  |
| filters_fused | 64 | 32000 | 0 | 8.387e-05 |  | 293 | 12 | 2.24 |
| sort_rows | 1 | 151936 | 0 | 0.004678 |  |  |  |  |
| chain_composed | 1 | 151936 | 0 | 0.005177 | 0.9036 |  |  |  |
| filters_unfused | 1 | 151936 | 0 | 0.0005225 |  | 8.433 | 29 |  |
| filters_fused | 1 | 151936 | 0 | 0.0002361 |  | 7.723 | 12 | 2.213 |
| sort_rows | 64 | 151936 | 0 | 0.009315 |  |  |  |  |
| chain_composed | 64 | 151936 | 0 | 0.01071 | 0.8697 |  |  |  |
| filters_unfused | 64 | 151936 | 0 | 0.001366 |  | 206.4 | 29 |  |
| filters_fused | 64 | 151936 | 0 | 0.0005962 |  | 195.7 | 12 | 2.292 |

### x1/lab3_sampler, solution, metal, TILE_WIDTH=16

| name | rows | vocab | top_k | seconds | tokens_per_s | calls | gpu_over_cpu | stage_0_s | stage_1_s | stage_2_s | stage_3_s | stage_4_s | stage_5_s | stage_6_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sample_step | 1 | 32000 | 40 | 0.000539 | 1855 | 2 | 14.81 | 9.496e-06 | 0.000247 | 4.189e-05 | 2.192e-06 | 2.441e-06 | 2.655e-06 | 4.388e-06 |
| cpu_sample_step | 1 | 32000 | 40 | 3.638e-05 | 2.748e+04 |  |  |  |  |  |  |  |  |  |
| sample_step | 8 | 32000 | 40 | 0.0003238 | 2.471e+04 | 8 | 1.389 | 1.171e-05 | 0.0002436 | 4.305e-05 | 2.355e-06 | 2.917e-06 | 3.269e-06 | 4.145e-06 |
| cpu_sample_step | 8 | 32000 | 40 | 0.0002331 | 3.432e+04 |  |  |  |  |  |  |  |  |  |
| sample_step | 64 | 32000 | 40 | 0.0004394 | 1.456e+05 | 8 | 0.2197 | 7.704e-05 | 0.0002918 | 5.605e-05 | 2.258e-06 | 3.264e-06 | 3.191e-06 | 6.019e-06 |
| cpu_sample_step | 64 | 32000 | 40 | 0.002 | 3.2e+04 |  |  |  |  |  |  |  |  |  |
| sample_step | 1 | 151936 | 40 | 0.001068 | 935.9 | 2 | 8.976 | 9.274e-06 | 0.001025 | 4.885e-05 | 2.236e-06 | 2.869e-06 | 3.113e-06 | 5.309e-06 |
| cpu_sample_step | 1 | 151936 | 40 | 0.000119 | 8401 |  |  |  |  |  |  |  |  |  |
| sample_step | 8 | 151936 | 40 | 0.001254 | 6381 | 2 | 1.295 | 4.237e-05 | 0.00117 | 4.67e-05 | 2.318e-06 | 2.999e-06 | 3.085e-06 | 5.554e-06 |
| cpu_sample_step | 8 | 151936 | 40 | 0.0009681 | 8263 |  |  |  |  |  |  |  |  |  |
| sample_step | 64 | 151936 | 40 | 0.002456 | 2.605e+04 | 1 | 0.3103 | 0.0008208 | 0.001587 | 5.61e-05 | 2.707e-06 | 3.147e-06 | 3.377e-06 | 5.088e-06 |
| cpu_sample_step | 64 | 151936 | 40 | 0.007917 | 8084 |  |  |  |  |  |  |  |  |  |
| sample_step_whole_row | 1 | 151936 | 0 | 0.006117 | 163.5 | 1 | 0.5318 | 9.216e-06 | 0.001002 | 0.00466 | 0.0002505 | 0.0001889 | 7.925e-05 | 5.435e-06 |
| cpu_sample_step_whole_row | 1 | 151936 | 0 | 0.0115 | 86.94 |  |  |  |  |  |  |  |  |  |
| sample_step_whole_row | 8 | 151936 | 0 | 0.007759 | 1031 | 1 | 0.08187 | 4.407e-05 | 0.001065 | 0.005287 | 0.0003081 | 0.0002603 | 8.395e-05 | 5.462e-06 |
| cpu_sample_step_whole_row | 8 | 151936 | 0 | 0.09478 | 84.41 |  |  |  |  |  |  |  |  |  |

### x2/lab1_q8gemv, solution, metal, TILE_WIDTH=16

| name | kernel | format | batch | in | out | seconds | weight_bytes | weight_gb_per_s | speedup_vs_f32 |
|---|---|---|---|---|---|---|---|---|---|
| qgemv | gather | f32 | 1 | 4096 | 4096 | 0.0003495 | 67108864 | 192 | 1 |
| qgemv | q8_0 | q8_0 | 1 | 4096 | 4096 | 8.526e-05 | 17825792 | 209.1 | 4.099 |
| qgemv | gather | f32 | 4 | 4096 | 4096 | 0.0005051 | 67108864 | 132.9 | 1 |
| qgemv | q8_0 | q8_0 | 4 | 4096 | 4096 | 0.0001287 | 17825792 | 138.5 | 3.925 |
| qgemv | gather | f32 | 8 | 4096 | 4096 | 0.0008184 | 67108864 | 82 | 1 |
| qgemv | q8_0 | q8_0 | 8 | 4096 | 4096 | 0.0002417 | 17825792 | 73.76 | 3.386 |
| qgemv | gather | f32 | 1 | 4096 | 14336 | 0.000919 | 234881024 | 255.6 | 1 |
| qgemv | q8_0 | q8_0 | 1 | 4096 | 14336 | 0.0002348 | 62390272 | 265.7 | 3.913 |
| qgemv | gather | f32 | 4 | 4096 | 14336 | 0.001061 | 234881024 | 221.4 | 1 |
| qgemv | q8_0 | q8_0 | 4 | 4096 | 14336 | 0.0003175 | 62390272 | 196.5 | 3.343 |
| qgemv | gather | f32 | 8 | 4096 | 14336 | 0.001586 | 234881024 | 148.1 | 1 |
| qgemv | q8_0 | q8_0 | 8 | 4096 | 14336 | 0.0006669 | 62390272 | 93.56 | 2.379 |
| qgemv | gather | f32 | 1 | 14336 | 4096 | 0.00131 | 234881024 | 179.2 | 1 |
| qgemv | q8_0 | q8_0 | 1 | 14336 | 4096 | 0.0003179 | 62390272 | 196.3 | 4.123 |
| qgemv | gather | f32 | 4 | 14336 | 4096 | 0.002218 | 234881024 | 105.9 | 1 |
| qgemv | q8_0 | q8_0 | 4 | 14336 | 4096 | 0.0006292 | 62390272 | 99.15 | 3.524 |
| qgemv | gather | f32 | 8 | 14336 | 4096 | 0.004238 | 234881024 | 55.42 | 1 |
| qgemv | q8_0 | q8_0 | 8 | 14336 | 4096 | 0.001066 | 62390272 | 58.55 | 3.977 |

### x2/lab2_q4gemv, solution, metal, TILE_WIDTH=16

| name | kernel | format | batch | in | out | seconds | weight_bytes | weight_gb_per_s | speedup_vs_f32 |
|---|---|---|---|---|---|---|---|---|---|
| qgemv | gather | f32 | 1 | 4096 | 4096 | 0.0003506 | 67108864 | 191.4 | 1 |
| qgemv | q8_0 | q8_0 | 1 | 4096 | 4096 | 8.256e-05 | 17825792 | 215.9 | 4.246 |
| qgemv | q4_0_unfused | q4_0 | 1 | 4096 | 4096 | 0.0006236 | 9437184 | 15.13 | 0.5621 |
| qgemv | q4_0 | q4_0 | 1 | 4096 | 4096 | 4.912e-05 | 9437184 | 192.1 | 7.137 |
| qgemv | gather | f32 | 4 | 4096 | 4096 | 0.0005096 | 67108864 | 131.7 | 1 |
| qgemv | q8_0 | q8_0 | 4 | 4096 | 4096 | 0.0001282 | 17825792 | 139 | 3.974 |
| qgemv | q4_0_unfused | q4_0 | 4 | 4096 | 4096 | 0.0008174 | 9437184 | 11.55 | 0.6235 |
| qgemv | q4_0 | q4_0 | 4 | 4096 | 4096 | 0.000119 | 9437184 | 79.33 | 4.284 |
| qgemv | gather | f32 | 8 | 4096 | 4096 | 0.0008265 | 67108864 | 81.2 | 1 |
| qgemv | q8_0 | q8_0 | 8 | 4096 | 4096 | 0.0002437 | 17825792 | 73.15 | 3.392 |
| qgemv | q4_0_unfused | q4_0 | 8 | 4096 | 4096 | 0.001121 | 9437184 | 8.422 | 0.7376 |
| qgemv | q4_0 | q4_0 | 8 | 4096 | 4096 | 0.0002264 | 9437184 | 41.69 | 3.651 |
| qgemv | gather | f32 | 1 | 4096 | 14336 | 0.0008791 | 234881024 | 267.2 | 1 |
| qgemv | q8_0 | q8_0 | 1 | 4096 | 14336 | 0.0002322 | 62390272 | 268.7 | 3.786 |
| qgemv | q4_0_unfused | q4_0 | 1 | 4096 | 14336 | 0.001975 | 33030144 | 16.73 | 0.4451 |
| qgemv | q4_0 | q4_0 | 1 | 4096 | 14336 | 0.0001386 | 33030144 | 238.3 | 6.342 |
| qgemv | gather | f32 | 4 | 4096 | 14336 | 0.001076 | 234881024 | 218.4 | 1 |
| qgemv | q8_0 | q8_0 | 4 | 4096 | 14336 | 0.0003182 | 62390272 | 196.1 | 3.381 |
| qgemv | q4_0_unfused | q4_0 | 4 | 4096 | 14336 | 0.002217 | 33030144 | 14.9 | 0.4851 |
| qgemv | q4_0 | q4_0 | 4 | 4096 | 14336 | 0.0003366 | 33030144 | 98.13 | 3.196 |
| qgemv | gather | f32 | 8 | 4096 | 14336 | 0.001593 | 234881024 | 147.5 | 1 |
| qgemv | q8_0 | q8_0 | 8 | 4096 | 14336 | 0.00067 | 62390272 | 93.12 | 2.377 |
| qgemv | q4_0_unfused | q4_0 | 8 | 4096 | 14336 | 0.002773 | 33030144 | 11.91 | 0.5743 |
| qgemv | q4_0 | q4_0 | 8 | 4096 | 14336 | 0.0007315 | 33030144 | 45.16 | 2.178 |
| qgemv | gather | f32 | 1 | 14336 | 4096 | 0.001336 | 234881024 | 175.8 | 1 |
| qgemv | q8_0 | q8_0 | 1 | 14336 | 4096 | 0.000313 | 62390272 | 199.4 | 4.269 |
| qgemv | q4_0_unfused | q4_0 | 1 | 14336 | 4096 | 0.002506 | 33030144 | 13.18 | 0.5333 |
| qgemv | q4_0 | q4_0 | 1 | 14336 | 4096 | 0.000165 | 33030144 | 200.2 | 8.099 |
| qgemv | gather | f32 | 4 | 14336 | 4096 | 0.002218 | 234881024 | 105.9 | 1 |
| qgemv | q8_0 | q8_0 | 4 | 14336 | 4096 | 0.0006173 | 62390272 | 101.1 | 3.593 |
| qgemv | q4_0_unfused | q4_0 | 4 | 14336 | 4096 | 0.003384 | 33030144 | 9.759 | 0.6553 |
| qgemv | q4_0 | q4_0 | 4 | 14336 | 4096 | 0.0004801 | 33030144 | 68.79 | 4.619 |
| qgemv | gather | f32 | 8 | 14336 | 4096 | 0.004233 | 234881024 | 55.48 | 1 |
| qgemv | q8_0 | q8_0 | 8 | 14336 | 4096 | 0.001075 | 62390272 | 58.02 | 3.937 |
| qgemv | q4_0_unfused | q4_0 | 8 | 14336 | 4096 | 0.005565 | 33030144 | 5.935 | 0.7607 |
| qgemv | q4_0 | q4_0 | 8 | 14336 | 4096 | 0.0008521 | 33030144 | 38.77 | 4.968 |

### x2/lab3_qhunt, solution, metal, TILE_WIDTH=16

| name | path | batch | in | out | seconds_quantise | seconds_gemv | seconds_total | weight_bytes | copies | source |
|---|---|---|---|---|---|---|---|---|---|---|
| qhunt | int8_q8_1 | 1 | 4096 | 4096 | 8e-06 | 0.0001247 | 0.0001128 | 9437184 | 54 | this lab |
| qhunt | float_q4_0 | 1 | 4096 | 4096 | 0 | 7.433e-05 | 7.433e-05 | 9437184 | 54 | the reference solution of lab X2.2 |
| qhunt | int8_q8_1 | 8 | 4096 | 4096 | 4.208e-06 | 0.0008105 | 0.000812 | 9437184 | 54 | this lab |
| qhunt | float_q4_0 | 8 | 4096 | 4096 | 0 | 0.0002961 | 0.0002961 | 9437184 | 54 | the reference solution of lab X2.2 |
| qhunt | int8_q8_1 | 1 | 14336 | 4096 | 4.917e-06 | 0.0003608 | 0.0003618 | 33030144 | 15 | this lab |
| qhunt | float_q4_0 | 1 | 14336 | 4096 | 0 | 0.000183 | 0.000183 | 33030144 | 15 | the reference solution of lab X2.2 |
| qhunt | int8_q8_1 | 8 | 14336 | 4096 | 4.25e-06 | 0.001887 | 0.001889 | 33030144 | 15 | this lab |
| qhunt | float_q4_0 | 8 | 14336 | 4096 | 0 | 0.0008851 | 0.0008851 | 33030144 | 15 | the reference solution of lab X2.2 |

### x2/lab4_m9, solution, metal, TILE_WIDTH=16

| name | format_layers | format_head | batch | seconds | median_s | tokens_per_s | weight_bytes | weight_gb_per_s | speedup_vs_f32 | logits_rel_rms | agree_steps | d | layers | vocab |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qstep | f32 | f32 | 1 | 0.002588 | 0.002629 | 386.4 | 658538496 | 254.5 | 1 | 0 | 32 | 4096 | 2 | 32000 |
| qstep | f32 | f32 | 2 | 0.002648 | 0.002694 | 755.3 | 658538496 | 248.7 | 1 | 0 | 32 | 4096 | 2 | 32000 |
| qstep | f32 | f32 | 4 | 0.002881 | 0.002933 | 1388 | 658538496 | 228.6 | 1 | 0 | 32 | 4096 | 2 | 32000 |
| qstep | f32 | f32 | 8 | 0.004148 | 0.004242 | 1929 | 658538496 | 158.7 | 1 | 0 | 32 | 4096 | 2 | 32000 |
| qstep | q8_0 | q8_0 | 1 | 0.0007135 | 0.0007278 | 1401 | 174948352 | 245.2 | 3.627 | 0.005241 | 32 | 4096 | 2 | 32000 |
| qstep | q8_0 | q8_0 | 2 | 0.0007386 | 0.0007458 | 2708 | 174948352 | 236.9 | 3.585 | 0.005237 | 32 | 4096 | 2 | 32000 |
| qstep | q8_0 | q8_0 | 4 | 0.0009815 | 0.0009966 | 4075 | 174948352 | 178.2 | 2.936 | 0.00522 | 32 | 4096 | 2 | 32000 |
| qstep | q8_0 | q8_0 | 8 | 0.001972 | 0.00201 | 4056 | 174948352 | 88.7 | 2.103 | 0.005205 | 32 | 4096 | 2 | 32000 |
| qstep | q4_0 | q4_0 | 1 | 0.0004722 | 0.0004788 | 2118 | 92635136 | 196.2 | 5.481 | 0.09104 | 32 | 4096 | 2 | 32000 |
| qstep | q4_0 | q4_0 | 2 | 0.0005541 | 0.0005656 | 3609 | 92635136 | 167.2 | 4.779 | 0.09093 | 32 | 4096 | 2 | 32000 |
| qstep | q4_0 | q4_0 | 4 | 0.001039 | 0.001066 | 3849 | 92635136 | 89.14 | 2.772 | 0.09088 | 32 | 4096 | 2 | 32000 |
| qstep | q4_0 | q4_0 | 8 | 0.002251 | 0.002277 | 3554 | 92635136 | 41.16 | 1.843 | 0.09059 | 32 | 4096 | 2 | 32000 |
| qstep | q4_0 | f32 | 1 | 0.002013 | 0.002035 | 496.8 | 543195136 | 269.8 | 1.286 | 0.02381 | 32 | 4096 | 2 | 32000 |
| qstep | q4_0 | f32 | 2 | 0.002066 | 0.002087 | 968 | 543195136 | 262.9 | 1.282 | 0.02382 | 32 | 4096 | 2 | 32000 |
| qstep | q4_0 | f32 | 4 | 0.002279 | 0.002298 | 1755 | 543195136 | 238.4 | 1.264 | 0.02389 | 32 | 4096 | 2 | 32000 |
| qstep | q4_0 | f32 | 8 | 0.003325 | 0.003367 | 2406 | 543195136 | 163.4 | 1.248 | 0.02412 | 32 | 4096 | 2 | 32000 |

### x3/lab1_llama_pieces, solution, metal, TILE_WIDTH=16

| name | rows | d | bytes | seconds | gb_per_s | calls | lab51_gb_per_s | heads | head_dim |
|---|---|---|---|---|---|---|---|---|---|
| rmsnorm | 1 | 576 | 6912 | 3.942e-06 | 1.753 | 512 | 271.1 |  |  |
| rmsnorm | 16 | 576 | 76032 | 2.548e-06 | 29.84 | 1024 | 271.1 |  |  |
| rmsnorm | 256 | 576 | 1181952 | 4.67e-06 | 253.1 | 512 | 271.1 |  |  |
| rmsnorm | 1 | 4096 | 49152 | 7.608e-06 | 6.461 | 512 | 271.1 |  |  |
| rmsnorm | 16 | 4096 | 540672 | 6.413e-06 | 84.31 | 256 | 271.1 |  |  |
| rmsnorm | 256 | 4096 | 8404992 | 1.734e-05 | 484.7 | 128 | 271.1 |  |  |
| rope | 1 | 576 | 4868 | 2.928e-06 | 1.663 | 1024 | 271.1 | 9 | 64 |
| rope | 16 | 576 | 77888 | 2.935e-06 | 26.54 | 1024 | 271.1 | 9 | 64 |
| rope | 256 | 576 | 1246208 | 5.672e-06 | 219.7 | 512 | 271.1 | 9 | 64 |
| rope | 1 | 4096 | 33284 | 2.705e-06 | 12.31 | 1024 | 271.1 | 32 | 128 |
| rope | 16 | 4096 | 532544 | 3.915e-06 | 136 | 512 | 271.1 | 32 | 128 |
| rope | 256 | 4096 | 8520704 | 2.006e-05 | 424.8 | 128 | 271.1 | 32 | 128 |
| swiglu | 1 | 576 | 27648 | 1.695e-06 | 16.31 | 2048 | 271.1 |  |  |
| swiglu | 16 | 576 | 442368 | 1.978e-06 | 223.7 | 1024 | 271.1 |  |  |
| swiglu | 256 | 576 | 7077888 | 8.014e-06 | 883.2 | 256 | 271.1 |  |  |
| swiglu | 1 | 4096 | 196608 | 1.778e-06 | 110.6 | 2048 | 271.1 |  |  |
| swiglu | 16 | 4096 | 3145728 | 4.189e-06 | 751 | 512 | 271.1 |  |  |
| swiglu | 256 | 4096 | 50331648 | 0.0001548 | 325 | 16 | 271.1 |  |  |

### x3/lab2_tinyllama, solution, metal, TILE_WIDTH=16

| name | d | hidden | layers | heads | kv_heads | vocab | seq_len | weight_bytes | gemm | attention | batch | position | seconds | median_s | tokens_per_s | weight_bytes_per_token | weight_gb_per_s | kv_bytes | group | share | lab51_gb_per_s | ceiling_tokens_per_s | fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model | 2048 | 5632 | 8 | 16 | 4 | 32000 | 1024 | 1.705e+09 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| step |  |  |  |  |  |  |  |  | gemv | grouped | 1 | 16 | 0.007848 | 0.008093 | 127.4 | 1.705e+09 | 217.3 | 557056 |  |  |  |  |  |
| step |  |  |  |  |  |  |  |  | gemv | grouped | 4 | 16 | 0.009115 | 0.009474 | 438.8 | 426280960 | 187.1 | 2228224 |  |  |  |  |  |
| step |  |  |  |  |  |  |  |  | gemv | grouped | 1 | 512 | 0.008219 | 0.008392 | 121.7 | 1.705e+09 | 207.5 | 16809984 |  |  |  |  |  |
| step |  |  |  |  |  |  |  |  | gemv | grouped | 4 | 512 | 0.009501 | 0.009768 | 421 | 426280960 | 179.5 | 67239936 |  |  |  |  |  |
| step |  |  |  |  |  |  |  |  | m2 | grouped | 1 | 16 | 0.01411 | 0.01416 | 70.85 | 1.705e+09 | 120.8 | 557056 |  |  |  |  |  |
| step |  |  |  |  |  |  |  |  | m2 | grouped | 4 | 16 | 0.0142 | 0.01425 | 281.8 | 426280960 | 120.1 | 2228224 |  |  |  |  |  |
| step |  |  |  |  |  |  |  |  | m2 | grouped | 1 | 512 | 0.01433 | 0.01442 | 69.78 | 1.705e+09 | 119 | 16809984 |  |  |  |  |  |
| step |  |  |  |  |  |  |  |  | m2 | grouped | 4 | 512 | 0.0145 | 0.01454 | 275.9 | 426280960 | 117.6 | 67239936 |  |  |  |  |  |
| step |  |  |  |  |  |  |  |  | m3 | grouped | 1 | 16 | 0.1436 | 0.1442 | 6.965 | 1.705e+09 | 11.88 | 557056 |  |  |  |  |  |
| step |  |  |  |  |  |  |  |  | m3 | grouped | 4 | 16 | 0.1436 | 0.1437 | 27.86 | 426280960 | 11.87 | 2228224 |  |  |  |  |  |
| step |  |  |  |  |  |  |  |  | m3 | grouped | 1 | 512 | 0.144 | 0.1455 | 6.943 | 1.705e+09 | 11.84 | 16809984 |  |  |  |  |  |
| step |  |  |  |  |  |  |  |  | m3 | grouped | 4 | 512 | 0.144 | 0.145 | 27.77 | 426280960 | 11.84 | 67239936 |  |  |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 1 | 16 | 0.006245 |  |  |  |  |  | matmuls | 0.83 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 1 | 16 | 8.567e-05 |  |  |  |  |  | attention | 0.01138 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 1 | 16 | 0.0002399 |  |  |  |  |  | norms+rope+swiglu+residual | 0.03188 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 1 | 16 | 0.000954 |  |  |  |  |  | embed+lm head+argmax+advance | 0.1268 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 4 | 16 | 0.008481 |  |  |  |  |  | matmuls | 0.8617 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 4 | 16 | 8.833e-05 |  |  |  |  |  | attention | 0.008975 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 4 | 16 | 0.0002465 |  |  |  |  |  | norms+rope+swiglu+residual | 0.02505 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 4 | 16 | 0.001026 |  |  |  |  |  | embed+lm head+argmax+advance | 0.1043 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 1 | 512 | 0.006277 |  |  |  |  |  | matmuls | 0.8084 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 1 | 512 | 0.0002907 |  |  |  |  |  | attention | 0.03744 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 1 | 512 | 0.0002411 |  |  |  |  |  | norms+rope+swiglu+residual | 0.03106 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 1 | 512 | 0.0009558 |  |  |  |  |  | embed+lm head+argmax+advance | 0.1231 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 4 | 512 | 0.008432 |  |  |  |  |  | matmuls | 0.8426 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 4 | 512 | 0.000297 |  |  |  |  |  | attention | 0.02968 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 4 | 512 | 0.0002466 |  |  |  |  |  | norms+rope+swiglu+residual | 0.02464 |  |  |  |
| piece |  |  |  |  |  |  |  |  | gemv |  | 4 | 512 | 0.001031 |  |  |  |  |  | embed+lm head+argmax+advance | 0.1031 |  |  |  |
| ceiling |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 127.4 | 1.705e+09 |  |  |  |  | 271.1 | 159 | 0.8014 |

## CUDA front-end check (clang, stub headers, nothing run)

Compiler: Apple clang version 21.0.0 (clang-2100.3.34.2)

| File | Parses as CUDA |
|---|---|
| labs/appA/lab1_sumerror/solution/kernel.cu | yes |
| labs/appA/lab1_sumerror/starter/kernel.cu | yes |
| labs/appA/lab1_sumerror/work/kernel.cu | yes |
| labs/appA/shared/fpprobe.cu | yes |
| labs/appC/lab1_feeding/solution/kernel.cu | yes |
| labs/appC/lab1_feeding/starter/kernel.cu | yes |
| labs/appC/lab1_feeding/work/kernel.cu | yes |
| labs/appC/shared/probe.cu | yes |
| labs/appC/snippets/c1_helper_generic.cu | yes |
| labs/appC/snippets/c3_chase.cu | yes |
| labs/appC/snippets/c6_sortfile_cpu.cu | yes |
| labs/appC/snippets/c6_sortfile_explicit.cu | yes |
| labs/appC/snippets/c6_sortfile_managed.cu | yes |
| labs/appC/snippets/wrap_layer_table.cu | yes |
| labs/ch02/lab1_vecadd/solution/kernel.cu | yes |
| labs/ch02/lab1_vecadd/starter/kernel.cu | yes |
| labs/ch02/lab1_vecadd/work/kernel.cu | yes |
| labs/ch02/lab2_timing/solution/kernel.cu | yes |
| labs/ch02/lab2_timing/starter/kernel.cu | yes |
| labs/ch02/lab2_timing/work/kernel.cu | yes |
| labs/ch02/lab3_elementwise/solution/kernel.cu | yes |
| labs/ch02/lab3_elementwise/starter/kernel.cu | yes |
| labs/ch02/lab3_elementwise/work/kernel.cu | yes |
| labs/ch02/lab4_bughunt/solution/kernel.cu | yes |
| labs/ch02/lab4_bughunt/starter/kernel.cu | yes |
| labs/ch02/lab4_bughunt/work/kernel.cu | yes |
| labs/ch02/snippets/count_above_race.cu | yes |
| labs/ch02/snippets/ex9_foo.cu | yes |
| labs/ch02/snippets/saxpy_stub_defect.cu | yes |
| labs/ch02/snippets/vecadd_sequential.cu | yes |
| labs/ch02/snippets/vecadd_stub.cu | yes |
| labs/ch03/lab1_grayscale/solution/kernel.cu | yes |
| labs/ch03/lab1_grayscale/starter/kernel.cu | yes |
| labs/ch03/lab1_grayscale/work/kernel.cu | yes |
| labs/ch03/lab2_blur/solution/kernel.cu | yes |
| labs/ch03/lab2_blur/starter/kernel.cu | yes |
| labs/ch03/lab2_blur/work/kernel.cu | yes |
| labs/ch03/lab3_naive_gemm/solution/kernel.cu | yes |
| labs/ch03/lab3_naive_gemm/starter/kernel.cu | yes |
| labs/ch03/lab3_naive_gemm/work/kernel.cu | yes |
| labs/ch03/lab4_granularity/solution/kernel.cu | yes |
| labs/ch03/lab4_granularity/starter/kernel.cu | yes |
| labs/ch03/lab4_granularity/work/kernel.cu | yes |
| labs/ch03/lab5_maphunt/solution/kernel.cu | yes |
| labs/ch03/lab5_maphunt/starter/kernel.cu | yes |
| labs/ch03/lab5_maphunt/work/kernel.cu | yes |
| labs/ch03/snippets/blur_kernel.cu | yes |
| labs/ch03/snippets/blur_unsigned.cu | yes |
| labs/ch03/snippets/check/blur_test.cu | yes |
| labs/ch03/snippets/check/grayscale_test.cu | yes |
| labs/ch03/snippets/check/launch_checked_test.cu | yes |
| labs/ch03/snippets/check/matvec_test.cu | yes |
| labs/ch03/snippets/check/scale_rows_test.cu | yes |
| labs/ch03/snippets/grayscale_as_printed.cu | yes |
| labs/ch03/snippets/grayscale_kernel.cu | yes |
| labs/ch03/snippets/launch_1d.cu | yes |
| labs/ch03/snippets/launch_checked.cu | yes |
| labs/ch03/snippets/launch_fig31.cu | yes |
| labs/ch03/snippets/matmul_kernel.cu | yes |
| labs/ch03/snippets/matvec_options.cu | yes |
| labs/ch03/snippets/scale_rows_defect.cu | yes |
| labs/ch03/snippets/vec_add_1d.cu | yes |
| labs/ch04/lab1_launchconfig/solution/kernel.cu | yes |
| labs/ch04/lab1_launchconfig/starter/kernel.cu | yes |
| labs/ch04/lab1_launchconfig/work/kernel.cu | yes |
| labs/ch04/lab2_divergence/solution/kernel.cu | yes |
| labs/ch04/lab2_divergence/starter/kernel.cu | yes |
| labs/ch04/lab2_divergence/work/kernel.cu | yes |
| labs/ch04/lab3_barrierhunt/solution/kernel.cu | yes |
| labs/ch04/lab3_barrierhunt/starter/kernel.cu | yes |
| labs/ch04/lab3_barrierhunt/work/kernel.cu | yes |
| labs/ch04/lab4_blocksize/provided/occupancy_api.cu | yes |
| labs/ch04/lab4_blocksize/provided/occupancy_none.cu | yes |
| labs/ch04/lab4_blocksize/solution/kernel.cu | yes |
| labs/ch04/lab4_blocksize/starter/kernel.cu | yes |
| labs/ch04/lab4_blocksize/work/kernel.cu | yes |
| labs/ch04/snippets/barrier_early_return.cu | yes |
| labs/ch04/snippets/barrier_guarded.cu | yes |
| labs/ch04/snippets/barrier_loop.cu | yes |
| labs/ch04/snippets/barrier_per_thread.cu | yes |
| labs/ch04/snippets/barrier_uniform.cu | yes |
| labs/ch04/snippets/device_loop.cu | yes |
| labs/ch04/snippets/foo_kernel.cu | yes |
| labs/ch04/snippets/incorrect_barrier.cu | yes |
| labs/ch04/snippets/warp_tail_sum.cu | yes |
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
| labs/ch05/lab5_dynamic/solution/kernel.cu | yes |
| labs/ch05/lab5_dynamic/starter/kernel.cu | yes |
| labs/ch05/lab5_dynamic/work/kernel.cu | yes |
| labs/ch05/shared/naive_matmul.cu | yes |
| labs/ch05/snippets/transpose_tiles_defect.cu | yes |
| labs/ch05/snippets/where_does_it_live.cu | yes |
| labs/ch06/lab1_coalescing/provided/copy.cu | yes |
| labs/ch06/lab1_coalescing/solution/kernel.cu | yes |
| labs/ch06/lab1_coalescing/starter/kernel.cu | yes |
| labs/ch06/lab1_coalescing/work/kernel.cu | yes |
| labs/ch06/lab2_vector/provided/vecadd.cu | yes |
| labs/ch06/lab2_vector/solution/kernel.cu | yes |
| labs/ch06/lab2_vector/starter/kernel.cu | yes |
| labs/ch06/lab2_vector/work/kernel.cu | yes |
| labs/ch06/lab3_banks/provided/bank_probe.cu | yes |
| labs/ch06/lab3_banks/provided/copy.cu | yes |
| labs/ch06/lab3_banks/solution/kernel.cu | yes |
| labs/ch06/lab3_banks/starter/kernel.cu | yes |
| labs/ch06/lab3_banks/work/kernel.cu | yes |
| labs/ch06/lab4_linear/provided/linear_baseline.cu | yes |
| labs/ch06/lab4_linear/solution/kernel.cu | yes |
| labs/ch06/lab4_linear/starter/kernel.cu | yes |
| labs/ch06/lab4_linear/work/kernel.cu | yes |
| labs/ch06/lab5_doublebuf/provided/tiled_two_barriers.cu | yes |
| labs/ch06/lab5_doublebuf/solution/kernel.cu | yes |
| labs/ch06/lab5_doublebuf/starter/kernel.cu | yes |
| labs/ch06/lab5_doublebuf/work/kernel.cu | yes |
| labs/ch06/snippets/s61_foo_kernel.cu | yes |
| labs/ch06/snippets/s65_coarsen_toy.cu | yes |
| labs/ch06/snippets/s66_unroll.cu | yes |
| labs/ch06/snippets/s66_where_x_lives.cu | yes |
| labs/ch06/snippets/s67_exchange.cu | yes |
| labs/ch06/snippets/wrap_scores.cu | yes |
| labs/ch07/lab1_constconv/solution/kernel.cu | yes |
| labs/ch07/lab1_constconv/starter/kernel.cu | yes |
| labs/ch07/lab1_constconv/work/kernel.cu | yes |
| labs/ch07/lab2_tiledconv/solution/kernel.cu | yes |
| labs/ch07/lab2_tiledconv/starter/kernel.cu | yes |
| labs/ch07/lab2_tiledconv/work/kernel.cu | yes |
| labs/ch07/shared/conv_const.cu | yes |
| labs/ch07/snippets/check/s75_halo_toy_test.cu | yes |
| labs/ch07/snippets/check/wrap_causal_conv1d_test.cu | yes |
| labs/ch07/snippets/s75_halo_toy.cu | yes |
| labs/ch07/snippets/wrap_causal_conv1d.cu | yes |
| labs/ch08/lab1_stencil3d/solution/kernel.cu | yes |
| labs/ch08/lab1_stencil3d/starter/kernel.cu | yes |
| labs/ch08/lab1_stencil3d/work/kernel.cu | yes |
| labs/ch08/shared/planes.cu | yes |
| labs/ch08/shared/tiled_cube.cu | yes |
| labs/ch08/snippets/regtile_two_planes.cu | yes |
| labs/ch08/snippets/rolling_window_indexed.cu | yes |
| labs/ch08/snippets/zmarch_planes.cu | yes |
| labs/ch08/snippets/zmarch_registers.cu | yes |
| labs/ch09/lab1_racehunt/solution/kernel.cu | yes |
| labs/ch09/lab1_racehunt/starter/kernel.cu | yes |
| labs/ch09/lab1_racehunt/work/kernel.cu | yes |
| labs/ch09/lab2_privatised/provided/ref_kernels.cu | yes |
| labs/ch09/lab2_privatised/solution/kernel.cu | yes |
| labs/ch09/lab2_privatised/starter/kernel.cu | yes |
| labs/ch09/lab2_privatised/work/kernel.cu | yes |
| labs/ch09/lab3_tokencounts/provided/count_pool.cu | yes |
| labs/ch09/lab3_tokencounts/solution/kernel.cu | yes |
| labs/ch09/lab3_tokencounts/starter/kernel.cu | yes |
| labs/ch09/lab3_tokencounts/work/kernel.cu | yes |
| labs/ch09/snippets/check/privatised_check.cu | yes |
| labs/ch09/snippets/check/wrap_early_exit_test.cu | yes |
| labs/ch09/snippets/privatised_variants.cu | yes |
| labs/ch09/snippets/wrap_early_exit.cu | yes |
| labs/ch10/lab1_sum/solution/kernel.cu | yes |
| labs/ch10/lab1_sum/starter/kernel.cu | yes |
| labs/ch10/lab1_sum/work/kernel.cu | yes |
| labs/ch10/lab2_reducehunt/solution/kernel.cu | yes |
| labs/ch10/lab2_reducehunt/starter/kernel.cu | yes |
| labs/ch10/lab2_reducehunt/work/kernel.cu | yes |
| labs/ch10/lab3_softmax_argmax/solution/kernel.cu | yes |
| labs/ch10/lab3_softmax_argmax/starter/kernel.cu | yes |
| labs/ch10/lab3_softmax_argmax/work/kernel.cu | yes |
| labs/ch10/shared/inplace_sum.cu | yes |
| labs/ch10/shared/rowwise_naive.cu | yes |
| labs/ch10/snippets/check/reduction_check.cu | yes |
| labs/ch10/snippets/check/s107_hybrid_test.cu | yes |
| labs/ch10/snippets/check/s107_warp_reduce_test.cu | yes |
| labs/ch10/snippets/check/wrap_centre_test.cu | yes |
| labs/ch10/snippets/s103_inplace.cu | yes |
| labs/ch10/snippets/s104_mirrored.cu | yes |
| labs/ch10/snippets/s106_barrier_hunt.cu | yes |
| labs/ch10/snippets/s107_hybrid.cu | yes |
| labs/ch10/snippets/s107_warp_reduce_as_printed.cu | yes |
| labs/ch10/snippets/wrap_centre.cu | yes |
| labs/ch11/lab1_blockscan/solution/kernel.cu | yes |
| labs/ch11/lab1_blockscan/starter/kernel.cu | yes |
| labs/ch11/lab1_blockscan/work/kernel.cu | yes |
| labs/ch11/lab2_coarsened/solution/kernel.cu | yes |
| labs/ch11/lab2_coarsened/starter/kernel.cu | yes |
| labs/ch11/lab2_coarsened/work/kernel.cu | yes |
| labs/ch11/lab3_scanhunt/solution/kernel.cu | yes |
| labs/ch11/lab3_scanhunt/starter/kernel.cu | yes |
| labs/ch11/lab3_scanhunt/work/kernel.cu | yes |
| labs/ch11/lab4_global/solution/kernel.cu | yes |
| labs/ch11/lab4_global/starter/kernel.cu | yes |
| labs/ch11/lab4_global/work/kernel.cu | yes |
| labs/ch11/lab5_singlepass/solution/kernel.cu | yes |
| labs/ch11/lab5_singlepass/starter/kernel.cu | yes |
| labs/ch11/lab5_singlepass/work/kernel.cu | yes |
| labs/ch11/lab6_topp/solution/kernel.cu | yes |
| labs/ch11/lab6_topp/starter/kernel.cu | yes |
| labs/ch11/lab6_topp/work/kernel.cu | yes |
| labs/ch11/shared-g11/bs_copy.cu | yes |
| labs/ch11/shared-g11/bs_threadmajor.cu | yes |
| labs/ch11/shared-g11/bs_two_barrier.cu | yes |
| labs/ch11/shared-g12/topp_copy.cu | yes |
| labs/ch11/shared-g12/topp_naive.cu | yes |
| labs/ch11/shared-g13/global_copy.cu | yes |
| labs/ch11/snippets/check/s1110_check.cu | yes |
| labs/ch11/snippets/check/s112_check.cu | yes |
| labs/ch11/snippets/check/s114_check.cu | yes |
| labs/ch11/snippets/check/wrap_mass_before_test.cu | yes |
| labs/ch11/snippets/s1110_brent_kung.cu | yes |
| labs/ch11/snippets/s112_kogge_stone.cu | yes |
| labs/ch11/snippets/s112_race_variants.cu | yes |
| labs/ch11/snippets/s114_blockscan_as_printed.cu | yes |
| labs/ch11/snippets/s116_coarsened_as_printed.cu | yes |
| labs/ch11/snippets/s117_regtiled_as_printed.cu | yes |
| labs/ch11/snippets/s119_flag_protocol.cu | yes |
| labs/ch11/snippets/wrap_mass_before.cu | yes |
| labs/ch12/lab1_unstable/provided/copy_keys.cu | yes |
| labs/ch12/lab1_unstable/solution/kernel.cu | yes |
| labs/ch12/lab1_unstable/starter/kernel.cu | yes |
| labs/ch12/lab1_unstable/work/kernel.cu | yes |
| labs/ch12/lab2_filterhunt/solution/kernel.cu | yes |
| labs/ch12/lab2_filterhunt/starter/kernel.cu | yes |
| labs/ch12/lab2_filterhunt/work/kernel.cu | yes |
| labs/ch12/lab3_stable/solution/kernel.cu | yes |
| labs/ch12/lab3_stable/starter/kernel.cu | yes |
| labs/ch12/lab3_stable/work/kernel.cu | yes |
| labs/ch12/lab4_candidates/provided/rowwise_serial.cu | yes |
| labs/ch12/lab4_candidates/solution/kernel.cu | yes |
| labs/ch12/lab4_candidates/starter/kernel.cu | yes |
| labs/ch12/lab4_candidates/work/kernel.cu | yes |
| labs/ch12/lab5_inplace/solution/kernel.cu | yes |
| labs/ch12/lab5_inplace/starter/kernel.cu | yes |
| labs/ch12/lab5_inplace/work/kernel.cu | yes |
| labs/ch12/shared-g12/scan_u32.cu | yes |
| labs/ch12/snippets/check/s123_check.cu | yes |
| labs/ch12/snippets/check/s124_privatised_check.cu | yes |
| labs/ch12/snippets/check/wrap_pairs_test.cu | yes |
| labs/ch12/snippets/s123_fig12_3.cu | yes |
| labs/ch12/snippets/s123_fig12_4_cg.cu | yes |
| labs/ch12/snippets/s124_privatised_variants.cu | yes |
| labs/ch12/snippets/wrap_pairs.cu | yes |
| labs/ch13/lab1_merge/solution/kernel.cu | yes |
| labs/ch13/lab1_merge/starter/kernel.cu | yes |
| labs/ch13/lab1_merge/work/kernel.cu | yes |
| labs/ch13/snippets/check/merge_basic_test.cu | yes |
| labs/ch13/snippets/merge_basic.cu | yes |
| labs/ch14/lab1_twosorts/solution/kernel.cu | yes |
| labs/ch14/lab1_twosorts/starter/kernel.cu | yes |
| labs/ch14/lab1_twosorts/work/kernel.cu | yes |
| labs/ch14/lab2_multibit/solution/kernel.cu | yes |
| labs/ch14/lab2_multibit/starter/kernel.cu | yes |
| labs/ch14/lab2_multibit/work/kernel.cu | yes |
| labs/ch14/lab3_sortrows/solution/kernel.cu | yes |
| labs/ch14/lab3_sortrows/starter/kernel.cu | yes |
| labs/ch14/lab3_sortrows/work/kernel.cu | yes |
| labs/ch14/shared-g11/onebit_global.cu | yes |
| labs/ch14/shared-g11/scan.cu | yes |
| labs/ch14/snippets/check/s142_check.cu | yes |
| labs/ch14/snippets/check/s144_check.cu | yes |
| labs/ch14/snippets/check/s145_check.cu | yes |
| labs/ch14/snippets/check/wrap_check.cu | yes |
| labs/ch14/snippets/s142_oddeven.cu | yes |
| labs/ch14/snippets/s144_floatkey.cu | yes |
| labs/ch14/snippets/s145_radix_iter.cu | yes |
| labs/ch14/snippets/wrap_blocksort.cu | yes |
| labs/ch15/lab1_coarse/solution/kernel.cu | yes |
| labs/ch15/lab1_coarse/starter/kernel.cu | yes |
| labs/ch15/lab1_coarse/work/kernel.cu | yes |
| labs/ch15/lab2_regtile/solution/kernel.cu | yes |
| labs/ch15/lab2_regtile/starter/kernel.cu | yes |
| labs/ch15/lab2_regtile/work/kernel.cu | yes |
| labs/ch15/lab3_remap/solution/kernel.cu | yes |
| labs/ch15/lab3_remap/starter/kernel.cu | yes |
| labs/ch15/lab3_remap/work/kernel.cu | yes |
| labs/ch15/lab4_padded/solution/kernel.cu | yes |
| labs/ch15/lab4_padded/starter/kernel.cu | yes |
| labs/ch15/lab4_padded/work/kernel.cu | yes |
| labs/ch15/lab5_pipelined/solution/kernel.cu | yes |
| labs/ch15/lab5_pipelined/starter/kernel.cu | yes |
| labs/ch15/lab5_pipelined/work/kernel.cu | yes |
| labs/ch15/lab6_linear/provided/occupancy.cu | yes |
| labs/ch15/lab6_linear/provided/vendor_linear.cu | yes |
| labs/ch15/lab6_linear/solution/kernel.cu | yes |
| labs/ch15/lab6_linear/starter/kernel.cu | yes |
| labs/ch15/lab6_linear/work/kernel.cu | yes |
| labs/ch15/lab7_bughunt/solution/kernel.cu | yes |
| labs/ch15/lab7_bughunt/starter/kernel.cu | yes |
| labs/ch15/lab7_bughunt/work/kernel.cu | yes |
| labs/ch15/shared/stage_occupancy.cu | yes |
| labs/ch15/shared/vendor_blas.cu | yes |
| labs/ch15/snippets/s156_padded_offset.cu | yes |
| labs/ch15/snippets/s158_barrier_before_load.cu | yes |
| labs/ch15/snippets/s158_no_epilogue.cu | yes |
| labs/ch15/snippets/s158_no_loop_barrier.cu | yes |
| labs/ch15/snippets/s158_pipeline_toy.cu | yes |
| labs/ch15/snippets/s158_single_buffer.cu | yes |
| labs/ch15/snippets/s158_wrong_tile.cu | yes |
| labs/ch15/snippets/wrap_load_first.cu | yes |
| labs/ch16/lab1_wavefront/solution/kernel.cu | yes |
| labs/ch16/lab1_wavefront/starter/kernel.cu | yes |
| labs/ch16/lab1_wavefront/work/kernel.cu | yes |
| labs/ch16/shared/baseline.cu | yes |
| labs/ch16/snippets/check/fw_bottomup_test.cu | yes |
| labs/ch16/snippets/fw_bottomup.cu | yes |
| labs/ch17/lab1_spmv/provided/coo_spmv.cu | yes |
| labs/ch17/lab1_spmv/solution/kernel.cu | yes |
| labs/ch17/lab1_spmv/starter/kernel.cu | yes |
| labs/ch17/lab1_spmv/work/kernel.cu | yes |
| labs/ch17/snippets/check/spmv_check.cu | yes |
| labs/ch17/snippets/check/spmv_jds_check.cu | yes |
| labs/ch17/snippets/spmv_coo.cu | yes |
| labs/ch17/snippets/spmv_csc.cu | yes |
| labs/ch17/snippets/spmv_csr.cu | yes |
| labs/ch17/snippets/spmv_ell.cu | yes |
| labs/ch17/snippets/spmv_jds.cu | yes |
| labs/ch18/lab1_frontier/provided/ref_kernels.cu | yes |
| labs/ch18/lab1_frontier/solution/kernel.cu | yes |
| labs/ch18/lab1_frontier/starter/kernel.cu | yes |
| labs/ch18/lab1_frontier/work/kernel.cu | yes |
| labs/ch18/snippets/check/s183_check.cu | yes |
| labs/ch18/snippets/check/s186_coop_check.cu | yes |
| labs/ch18/snippets/check/s186_privatised_check.cu | yes |
| labs/ch18/snippets/s183_variants.cu | yes |
| labs/ch18/snippets/s186_coop_swap.cu | yes |
| labs/ch18/snippets/s186_privatised_variants.cu | yes |
| labs/ch19/lab1_convgemm/provided/implicit_gemm.cu | yes |
| labs/ch19/lab1_convgemm/solution/kernel.cu | yes |
| labs/ch19/lab1_convgemm/starter/kernel.cu | yes |
| labs/ch19/lab1_convgemm/work/kernel.cu | yes |
| labs/ch19/snippets/wrap_backward_scatter.cu | yes |
| labs/ch20/lab1_causal_softmax/solution/kernel.cu | yes |
| labs/ch20/lab1_causal_softmax/starter/kernel.cu | yes |
| labs/ch20/lab1_causal_softmax/work/kernel.cu | yes |
| labs/ch20/lab2_flash/solution/kernel.cu | yes |
| labs/ch20/lab2_flash/starter/kernel.cu | yes |
| labs/ch20/lab2_flash/work/kernel.cu | yes |
| labs/ch20/lab3_flash_hunt/solution/kernel.cu | yes |
| labs/ch20/lab3_flash_hunt/starter/kernel.cu | yes |
| labs/ch20/lab3_flash_hunt/work/kernel.cu | yes |
| labs/ch20/lab4_decode/solution/kernel.cu | yes |
| labs/ch20/lab4_decode/starter/kernel.cu | yes |
| labs/ch20/lab4_decode/work/kernel.cu | yes |
| labs/ch20/lab5_gqa_decode/solution/kernel.cu | yes |
| labs/ch20/lab5_gqa_decode/starter/kernel.cu | yes |
| labs/ch20/lab5_gqa_decode/work/kernel.cu | yes |
| labs/ch20/snippets/s203_softmax_corrected.cu | yes |
| labs/ch21/lab1_dcs/solution/kernel.cu | yes |
| labs/ch21/lab1_dcs/starter/kernel.cu | yes |
| labs/ch21/lab1_dcs/work/kernel.cu | yes |
| labs/ch21/shared/coarse_adjacent.cu | yes |
| labs/ch21/snippets/bias_gather.cu | yes |
| labs/ch21/snippets/check/dcs_test.cu | yes |
| labs/ch21/snippets/coarsen_adjacent.cu | yes |
| labs/ch21/snippets/coarsen_printed.cu | yes |
| labs/ch21/snippets/coarsen_strided.cu | yes |
| labs/ch21/snippets/gather.cu | yes |
| labs/ch21/snippets/scatter.cu | yes |
| labs/ch22/lab1_gemv/solution/kernel.cu | yes |
| labs/ch22/lab1_gemv/starter/kernel.cu | yes |
| labs/ch22/lab1_gemv/work/kernel.cu | yes |
| labs/ch22/lab2_batching/provided/step_timer.cu | yes |
| labs/ch22/lab2_batching/solution/kernel.cu | yes |
| labs/ch22/lab2_batching/starter/kernel.cu | yes |
| labs/ch22/lab2_batching/work/kernel.cu | yes |
| labs/ch22/snippets/check/conv_check.cu | yes |
| labs/ch22/snippets/gather_conv.cu | yes |
| labs/ch22/snippets/scatter_conv_plain.cu | yes |
| labs/ch23/lab1_overlap/probe.cu | yes |
| labs/ch23/lab1_overlap/solution/kernel.cu | yes |
| labs/ch23/lab1_overlap/starter/kernel.cu | yes |
| labs/ch23/lab1_overlap/work/kernel.cu | yes |
| labs/ch23/snippets/check/staged_loop_test.cu | yes |
| labs/ch23/snippets/nccl_exchange.cu | yes |
| labs/ch23/snippets/overlap_loop.cu | yes |
| labs/ch23/snippets/rerecord.cu | yes |
| labs/ch23/snippets/staged_loop.cu | yes |
| labs/selftest/atomics.cu | yes |
| labs/selftest/cgroups.cu | yes |
| labs/selftest/constant.cu | yes |
| labs/selftest/halfprec.cu | yes |
| labs/selftest/hostapi.cu | yes |
| labs/selftest/intrinsics.cu | yes |
| labs/selftest/race_hist_atomic.cu | yes |
| labs/selftest/race_hist_plain.cu | yes |
| labs/selftest/race_negatives.cu | yes |
| labs/selftest/shared.cu | yes |
| labs/selftest/warp.cu | yes |
| labs/x1/lab1_topk/solution/kernel.cu | yes |
| labs/x1/lab1_topk/starter/kernel.cu | yes |
| labs/x1/lab1_topk/work/kernel.cu | yes |
| labs/x1/lab2_filters/solution/kernel.cu | yes |
| labs/x1/lab2_filters/starter/kernel.cu | yes |
| labs/x1/lab2_filters/work/kernel.cu | yes |
| labs/x1/lab3_sampler/solution/kernel.cu | yes |
| labs/x1/lab3_sampler/starter/kernel.cu | yes |
| labs/x1/lab3_sampler/work/kernel.cu | yes |
| labs/x1/lab4_filterhunt/solution/kernel.cu | yes |
| labs/x1/lab4_filterhunt/starter/kernel.cu | yes |
| labs/x1/lab4_filterhunt/work/kernel.cu | yes |
| labs/x1/shared-g11/minp_g11.cu | yes |
| labs/x1/shared-g11/penalties_g11.cu | yes |
| labs/x1/shared-g11/softmax_g11.cu | yes |
| labs/x1/shared-g11/sortrows_g11.cu | yes |
| labs/x1/shared-g11/topk_g11.cu | yes |
| labs/x1/shared-g11/topp_g11.cu | yes |
| labs/x1/shared-g12/softmax_g12.cu | yes |
| labs/x1/shared-g12/sortrows_g12.cu | yes |
| labs/x1/shared-g12/topp_g12.cu | yes |
| labs/x1/snippets/check/draw_check.cu | yes |
| labs/x1/snippets/check/philox_kat.cu | yes |
| labs/x1/snippets/draw_search.cu | yes |
| labs/x1/snippets/philox.cu | yes |
| labs/x2/lab1_q8gemv/solution/kernel.cu | yes |
| labs/x2/lab1_q8gemv/starter/kernel.cu | yes |
| labs/x2/lab1_q8gemv/work/kernel.cu | yes |
| labs/x2/lab2_q4gemv/provided/unfused.cu | yes |
| labs/x2/lab2_q4gemv/solution/kernel.cu | yes |
| labs/x2/lab2_q4gemv/starter/kernel.cu | yes |
| labs/x2/lab2_q4gemv/work/kernel.cu | yes |
| labs/x2/lab3_qhunt/solution/kernel.cu | yes |
| labs/x2/lab3_qhunt/starter/kernel.cu | yes |
| labs/x2/lab3_qhunt/work/kernel.cu | yes |
| labs/x2/lab4_m9/provided/step_timer.cu | yes |
| labs/x2/lab4_m9/solution/kernel.cu | yes |
| labs/x2/lab4_m9/starter/kernel.cu | yes |
| labs/x2/lab4_m9/work/kernel.cu | yes |
| labs/x2/shared-g12/gemv_q4_0_g12.cu | yes |
| labs/x2/shared-g12/gemv_q8_0_g12.cu | yes |
| labs/x2/snippets/check/accumulators_check.cu | yes |
| labs/x3/lab1_llama_pieces/solution/kernel.cu | yes |
| labs/x3/lab1_llama_pieces/starter/kernel.cu | yes |
| labs/x3/lab1_llama_pieces/work/kernel.cu | yes |
| labs/x3/lab2_tinyllama/provided/m9_linear.cu | yes |
| labs/x3/lab2_tinyllama/provided/step_timer.cu | yes |
| labs/x3/lab2_tinyllama/solution/kernel.cu | yes |
| labs/x3/lab2_tinyllama/starter/kernel.cu | yes |
| labs/x3/lab2_tinyllama/work/kernel.cu | yes |

