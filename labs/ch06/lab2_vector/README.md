# Lab 6.2: vector loads, with the boundaries done right

Book: section 6.3 (pp. 138-139), Fig. 6.12 and Exercise 4 (p. 155); section 6.5
(pp. 141-143) for the coarsening loop.

Fig. 6.12 rewrites vector addition so that each thread adds four pairs of floats with one
16-byte load per input and one 16-byte store. It launches a quarter of the threads and
leaves the boundaries to Exercise 4: as printed it reads past the end whenever n is not a
multiple of 4. Section 6.5 then calls the same kernel an example of thread coarsening.
This lab separates the two ideas. You write two kernels in which each thread owns the
same four elements, `4i` to `4i+3`, and time them against one float per thread.

**Your part:**

- `vecAdd4Kernel` and `launchVecAdd4`: four elements per instruction. A full chunk is
  added with one `float4` load from `x`, one from `y` and one `float4` store to `z`
  (the book's cast). A partial last chunk (1 to 3 elements) falls back to one float at a
  time. A thread whose chunk starts past the end does nothing. Launch
  `ceil(ceil(n / 4) / 256)` blocks of 256 threads.
- `vecAddCoarseKernel` and `launchVecAddCoarse`: four elements per thread, one float per
  access. This is section 6.5's coarsening loop with a factor of 4: `iStart` is 4 times
  the global thread index, the loop runs over `c` from 0 to 3 under `#pragma unroll`, and
  each element is checked against n.

## Files

- Edit `work/kernel.cu` (CUDA, HIP, emulator): the two kernels and their launches.
- Edit `work/kernel.metal` (`vec_add4`, `vec_add_coarse`) and `work/launch.mm`
  (`encodeVecAdd4`, `encodeVecAddCoarse`). `vec_add4` declares its three buffers as
  `float4` for full chunks and takes the same three buffers again as plain `float` for
  the tail, so `encodeVecAdd4` binds each buffer twice, at two indices.
- Provided, not edited: the one-float-per-thread baseline in `provided/vecadd.cu`,
  `provided/vecadd.metal` and `provided/vecadd_launch.mm`.

## Commands

- Test: `python3 labs/pmpp.py run 6.2 --backend emu`, then `python3 labs/pmpp.py run 6.2`.
- Measure: `python3 labs/pmpp.py run 6.2 --bench` (real devices only).

## What the tests do

Both kernels run for n = 1, 2, 3, 4, 5, 7, 8, 9, 255, 256, 257, 1023, 1024 and 1025, and
on a GPU also for 2^20, 2^20 + 3 and 10,000,001. Every remainder of n mod 4 is covered,
with one block, several blocks, and a partial last block.

- Each array holds exactly n floats, with nothing spare after it. On the emulator a
  `float4` access that reaches past the end is an AddressSanitizer error that names the
  line. On a GPU the same mistake may go unnoticed, which is why you start on `emu`.
- `z` starts as NaN, so an element your kernel never writes is reported, and the message
  says whether it was in a full chunk or in the partial last one.
- The result is compared with a CPU reference, with a tolerance.
- The pointers are the allocations themselves, so they are 16-byte aligned. No test
  passes a misaligned pointer such as `x + 1`: the emulator cannot tell a misaligned
  `float4` access from an aligned one, so such a test could not fail there.

## Alignment, per backend

The cast needs a pointer aligned to 16 bytes. What each vendor documents
(`notes/today/ch06-facts.md` and `notes/today/infra-facts.md`):

- CUDA: NVIDIA's reference guides do not describe the cast. NVIDIA's developer blog does,
  and requires the cast pointer to be aligned to the vector's size (A11). `cudaMalloc`
  returns memory aligned to at least 256 bytes (A3).
- HIP: `float4` exists with the CUDA name and has 16-byte alignment in the header; no AMD
  page describes the cast either (B5, infra B6).
- Metal: the documented way is to declare the buffer `device const float4*` (C3,
  infra C7), which is what `vec_add4` does. A pointer cast compiles, and MLX and
  llama.cpp use one, but the specification does not describe it. Apple also says that
  `uint` indices can stop loads being vectorised, so the Metal kernels index with `int`
  (C3). How many accesses the GPU actually makes for a `float4` is not documented.

## Before you benchmark, predict

Write these down first.

1. The speed-up of `vec4` over `scalar` at the largest size the benchmark reaches. The
   book's argument is that two vector loads replace eight scalar loads, 75 % fewer load
   instructions for the same data (p. 139). The book also says the scalar kernel is
   already coalesced and runs at full occupancy (p. 138), and it moves 12 bytes for every
   addition. Decide how much of the saving in instructions becomes time.
2. `coarse4` against `scalar`: faster, the same, or slower? Before you answer, apply
   section 6.1's rule to one load instruction of each kernel: how far apart are the
   addresses of neighbouring threads? (Compare Exercise 1, pp. 154-155.)

## Reading the output

The benchmark grows n from 2^22 until the baseline takes over 5 ms, up to 2^26, keeping
the three arrays within a quarter of the memory your device reports. It prints one row
per kernel at that n:

- `time ms`: the best of five runs.
- `GB/s`: 12 bytes per element divided by the time. It counts the bytes the addition
  needs, not what the memory system moved.
- `vs scalar`: the baseline's time divided by this kernel's.

The numbers are measurements of three kernels on your machine at one size. Section 6.3 of
the lesson reads the same rows.

## Go further

PyTorch's elementwise kernels for CUDA and ROCm do what Exercise 4 asks, with one step
more: they do not assume alignment. `aten/src/ATen/native/cuda/MemoryAccess.cuh` checks
each tensor's address at run time (`can_vectorize_up_to` returns 8, 4, 2 or 1) and picks
the vector width from it, falling back to scalar accesses (ch06 facts D4). llama.cpp's
CUDA and HIP backend copies 16 bytes at a time through the same kind of cast and has a
helper that checks a tensor's alignment first (D1). As an exercise of your own: what would
`vecAdd4Kernel` need in order to accept `x + 1`?
