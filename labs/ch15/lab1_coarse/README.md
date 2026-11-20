# Lab 15.1: the 128 x 128 coarsened kernel

Book: section 15.3 (pp. 352-357), Figs. 15.3 to 15.7. Milestone M3, stage 1 of 5.

Write the book's large-tile matrix multiplication, `C[M x N] = A[M x K] * B[K x N]`,
row-major, any sizes. A block of 256 threads computes a 128 x 128 tile of C. Each thread
owns an 8 x 8 patch of that tile and keeps it in registers, while the inner dimension
streams through shared memory 8 columns of A and 8 rows of B at a time. Four helpers
(`clear`, `loadTile`, `mm`, `writeTile`), the kernel `gemmCoarseKernel`, and its launch
`launchGemmCoarse`.

Where the lab differs from the page, on purpose (the solution's header says why):

- The block is one-dimensional, 256 threads. The prose says 16 x 16, but Fig. 15.3 reads
  only `threadIdx.x`.
- `A_s` and `B_s` are declared once, at the top of the kernel.
- Sizes are `int`, and the arguments are `(A, B, C, M, K, N)`: lab 5.3's order (A's rows,
  the inner dimension, B's columns), so milestone M2's call sites work unchanged.
- The helpers are `static`. Later stages link this lab's solution into their benchmark,
  and the CPU emulator's linker would silently merge two non-static inline functions
  with the same name.
- `__launch_bounds__(256)` is ours. It tells the compiler the block size; without it HIP
  compiles for the largest block the device accepts and may squeeze registers
  (`notes/today/ch15-facts.md` B3).

## Files

- Edit `work/kernel.cu` and `work/kernel.metal` + `work/launch.mm`.
- `../shared/` holds the tests and the benchmark that labs 15.1 to 15.5 share
  (`stage_driver.hpp`), the vendor GEMM calls, and the build logic that copies lab 5.3's
  solution into this lab's benchmark. You do not edit it.

## Commands

    python3 labs/pmpp.py run 15.1 --backend emu   # start here
    python3 labs/pmpp.py run 15.1                 # every backend you have
    python3 labs/pmpp.py run 15.1 --bench         # real devices only

On the emulator a missing barrier gives a wrong answer under every thread order, and
AddressSanitizer stops at the first out-of-bounds read of A or B and names the line.

## What the tests check

- Shapes chosen to break one thing each: a single element, one block with one tile pair,
  three tile pairs, partial tiles in every direction with K and N not multiples of 4, one
  decode row across three column blocks, K = 1, four full blocks, a narrow N, and K = 0
  (an empty sum: C must come out all zeros). GPUs add a 1024 square and three large
  awkward shapes. The K = 0 case has 30 seconds; a kernel still running then fails with
  the likely reason, a tile-loop bound that wrapped around (lab 15.5 is where that bites).
- C is filled with NaN before each run and followed by a guard band of 128 rows holding a
  sentinel. An element your kernel never writes stays NaN; a store past the last row of C
  changes the guard band. The failure message says which.
- Results are compared with a CPU reference within a tolerance that scales with the size
  of the products summed (Metal does not promise strict IEEE arithmetic).
- On Metal, the pipeline must accept 256 threads per threadgroup. Metal may set a lower
  limit than the kernel asks for, so the driver checks before it runs anything.

## Before you benchmark

Write down two predictions.

1. **The speed-up over lab 5.3's kernel at 2048 x 2048 x 2048.** Weigh three things. The
   model says 4 FLOP/B for lab 5.3's 16 x 16 tiles and 32 FLOP/B for 128 x 128 tiles: 8
   times less global traffic requested. But the inner loop of Fig. 15.6 still reads two
   values from shared memory for every multiply-add, exactly as lab 5.3's does. And on the
   machine the course was piloted on, caches already served much of lab 5.3's traffic, so
   tiling gained far less than its model said.
2. **At batch 1 on the 1024 -> 4096 layer, faster or slower than lab 5.3?** Count the
   blocks each kernel launches (the benchmark prints them: 256 for lab 5.3, 32 here) and
   compare with your device's number of SMs from `python3 labs/pmpp.py doctor` (CUDA and
   HIP report `multiProcessorCount`; Metal reports no such count).

`--bench` times lab 5.3's solution (T = 16), your kernel, and the vendor's GEMM (cuBLAS,
hipBLAS or MPS), in turn, best of 5 runs each, on squares of 1024, 2048 and 4096, the
1024 -> 4096 layer at batch 1, 16, 256 and 1024, and the short-K shape 4096 x 128 x 4096.
Every kernel is first checked against the CPU reference. It also prints what the device
says about your kernel: on CUDA and HIP the blocks per SM the occupancy API reports (the
book's estimate is one, p. 363), on Metal the pipeline's thread limit, SIMD-group width
and threadgroup memory. The numbers are measurements of your machine.

## Beyond the book

Not in the book; from `notes/today/ch15-facts.md`.

- Metal: the helpers are `METAL_FUNC`, Apple's macro for an always-inline function from
  its shipped headers; the MSL specification has no inline attribute. Loops are unrolled
  with `#pragma clang loop unroll(full)`, which Apple's compiler accepts and MLX uses, but
  Apple documents no unroll control (C4). `[[max_total_threads_per_threadgroup(256)]]`
  is the counterpart of `__launch_bounds__` (C3).
- The vendor baselines: cuBLAS is the book's (p. 368); hipBLAS and Metal Performance
  Shaders are ours (B6, C6). cuBLAS and hipBLAS are column-major, so the row-major
  product is computed as `C^T = B^T A^T` (A10). All three compute in FP32 unless told
  otherwise; the benchmark prints the math mode. The cuBLAS and hipBLAS calls have not
  been built on a CUDA or ROCm machine yet: if they do not build, configure with
  `-DPMPP_CH15_VENDOR=OFF` (and `-DPMPP_CH15_OCCUPANCY_API=OFF` for the occupancy query),
  as lab 4.4 does for its occupancy call.
