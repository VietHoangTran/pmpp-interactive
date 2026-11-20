# Lab 11.2: coarsening and register tiling

Book: sections 11.6 to 11.8 (pp. 267-273), Figs. 11.11 to 11.13, and exercise 4 (p. 286).

You write the coarsened block scan of Fig. 11.12 and its register-tiled form, Fig. 11.13,
and time them against a provided variant that skips shared memory and a copy kernel. The
scan of the per-thread totals is your `blockScan` from lab 11.1.

The contract: block b scans its own segment of `COARSE_FACTOR*blockDim.x` elements,
inclusive, with +, and knows nothing of the other blocks. n can be anything from 1 up. An
element at or past n loads 0 and is not stored. Blocks are powers of two from 64 to 1024.
`COARSE_FACTOR` is 1, 2, 4, 8 or 16.

**Your part:**

- **(a) `CoarsenedScanKernel`** (Fig. 11.12). The block loads its segment into shared
  memory in `COARSE_FACTOR` passes. In each pass, thread t loads element t of the next
  `blockDim.x` elements, so the threads of a warp read consecutive addresses. Then each
  thread scans its own run of `COARSE_FACTOR` consecutive elements, sequentially, in
  shared memory. `blockScan` scans the run totals, each thread adds the total of the
  runs before its own, and the block stores in passes, as it loaded.
- **(b) `RegisterTiledScanKernel`** (Fig. 11.13). The same, but the run lives in a local
  array `buffer_r[COARSE_FACTOR]`. Each element is read from shared memory once, and the
  loops over it carry `#pragma unroll`. Shared memory stays only to make the global
  loads and stores contiguous (p. 272).
- Corrected from the page: the book's kernels never use `N`, so they are right only when
  n is a multiple of the segment. Yours check every global load and store, and an
  element out of range contributes 0.
- `COARSE_FACTOR` is a template parameter (`template <unsigned int COARSE_FACTOR>`), so
  one build tests and times all five factors. The launchers take the factor as their last
  argument and pick the instantiation with a `switch`. On Metal it is a preprocessor
  macro, set when the driver builds the pipeline: one pipeline per factor.
- The segment buffer is dynamic shared memory, `extern __shared__ float buffer_s[];`,
  sized by the third `<<< >>>` argument. On the page it is a static array of
  `COARSE_FACTOR*BLOCK_DIM` with a compile-time `BLOCK_DIM`. Sized for the largest block,
  a static array would reserve four times what a 256-thread block uses, and the
  benchmark would measure that. With `COARSE_FACTOR*blockDim.x` at most 8192 (factor 16
  only with blocks up to 512), the total stays within the 48 KiB a block may use without
  opting in (ch05 facts A3).

## Files

- Edit `work/kernel.cu` (CUDA, HIP and both emulators): `CoarsenedScanKernel`,
  `RegisterTiledScanKernel`, your `blockScan` and its helpers copied from lab 11.1, and
  `launchCoarsenedScan` and `launchRegisterTiledScan`.
- Edit `work/kernel.metal` (`coarsened_scan`, `register_tiled_scan`, and your
  `block_scan` from lab 11.1) and `work/launch.mm` (`encodeCoarsenedScan`,
  `encodeRegisterTiledScan`). Each encoder returns false, without dispatching, for a
  configuration its pipeline cannot run: more threads than
  `maxTotalThreadsPerThreadgroup`, a block that is not whole SIMD-groups (at most
  `width` of them), or more threadgroup memory than the device's
  `maxThreadgroupMemoryLength`.
- Provided, not edited, in `../shared-g11/`: `bs_threadmajor.cu` and `.metal` (the
  thread-major variant, below), `bs_copy.cu` and `.metal`, and `bs_launch.mm`.

The provided **thread-major** kernel is Fig. 11.13 without the staging: each thread loads
its own run of `COARSE_FACTOR` consecutive elements straight from global memory and stores
it straight back. In one load instruction the lanes of a warp touch addresses
`COARSE_FACTOR` floats apart, which is the pattern p. 269 warns against. It is not a straw
man. llama.cpp's CUDA cumsum loads this way, and MLX's Metal scan has each thread read
its own contiguous values too (ch11 facts D1, D3). Your benchmark shows what it costs
on your device.

## Commands

- Test: `python3 labs/pmpp.py run 11.2 --backend emu` first, then `python3 labs/pmpp.py run 11.2`.
- Shared updates: `python3 labs/pmpp.py run 11.2 --backend race`.
- Measure: `python3 labs/pmpp.py run 11.2 --bench` (real devices only).
- The reference solution: add `--solution`.

## What the tests do

- n = 1, 2, 33, 1000, 4097, 65537 and 2^22 + 3, at factors 1, 2, 4, 8 and 16 and blocks
  of 64, 256 and 1024. A configuration whose memory does not fit is skipped, and the
  driver prints why. The emulator runs n up to 4097 at factors 1, 4 and 8 with blocks of
  64 and 256. `race` runs n = 1000 at block 64.
- The same three kinds of data and the same tolerance as lab 11.1, per segment: all ones
  and integers from -3 to 3 must match exactly. Values uniform in [-1, 1] must be within
  1e-6 times the sum of |x| over the segment up to that element, plus 1e-6, of Fig.
  11.1's loop run in double.
- Checked explicitly before the rest: the first element of every segment (the scan
  restarts there), the last element of every segment (its total), and the last element
  of a partial last segment.
- The output starts as NaN. Guard words follow it, up to the end of the last segment and
  16 more, so a store at or past n is reported.
- The provided thread-major and copy kernels are tested too.

What each backend can catch:

- `emu` runs the threads one at a time with garbage in shared memory, under
  AddressSanitizer. Each of these fails there on its own: a load not checked against n
  (AddressSanitizer names the line), a store not checked against n (the guard words), a
  segment loaded in one order and read in the other, and a missing barrier after the
  load or before the store.
- `race` runs the threads at the same time under ThreadSanitizer and reports a missing
  barrier with both source lines.
- Metal is a real GPU. It shows whether your kernels are fast, not only right.

## Where the backends differ (not in the book)

- **HIP**: `extern __shared__`, templates and `#pragma unroll` keep their spelling (ch05
  facts B4; infra facts B7). The lab 11.1 mask and warp-width rules still apply to your
  `blockScan`. The driver reads the shared-memory limit from the device
  (`sharedMemPerBlock`), which differs on AMD. None of this has been built or run with
  ROCm here: unverified.
- **Metal**: dynamic threadgroup memory is a `threadgroup float*` kernel parameter marked
  `[[threadgroup(0)]]`. Its size is set with `setThreadgroupMemoryLength:atIndex:`, in a
  multiple of 16 bytes (ch05 facts C8). The limit is the device's
  `maxThreadgroupMemoryLength`, and the kernel's static arrays count against it too
  (`staticThreadgroupMemoryLength`, ch05 facts C3). That is why some factor and block
  pairs that run on CUDA are skipped on Metal. MSL documents no unroll control (infra
  facts, "Where the backends really differ"). The run is a fixed-size local array with
  constant trip counts, and the compiler decides.
- On Metal you can also build `block_scan` on `simd_prefix_inclusive_sum` (lab 11.1,
  (b2)). The tests do not care which one you use.

## Before you benchmark, predict

Write these down first. The lesson asks for them under the ids in brackets.

1. Which factor from 1 to 16 will be fastest at block 256 (`ch11/labs/p-lab2-best`).
2. What share of your copy bandwidth the register-tiled kernel will reach at its best
   factor (`ch11/11.8/p-copy`, asked in section 11.8).
3. How many times faster your register-tiled kernel (b), staged through shared memory, will
   be than the provided thread-major kernel at the same factor, which loads each run straight
   from global memory (`ch11/11.6/p-staging`, asked in section 11.6 at block 256, factor 4).

The book's model: with P threads and N elements, coarsening brings the work from about
N·log2(N) down to about 2N + P·log2(P), and the steps to about 2N/P + log2(P)
(pp. 269-270). A scan moves at least 8 bytes per element, and a memory-bound kernel that
reaches more than 80% of bandwidth with no redundant accesses counts as well optimized
(pp. 272-273). Here the yardstick is your own copy kernel.

## Reading the output

The benchmark scans n = 2^26 floats (fewer if the input and output would take more than a
quarter of the memory your device reports), at blocks of 256 and 1024. For each block
size it times the copy kernel, then, at each factor from 1 to 16, your coarsened and
register-tiled kernels and the thread-major one. A configuration that does not fit is
printed as skipped, with the reason.

- `time ms` is the best of five runs after one untimed warm-up. A quarter of a second of
  untimed copies comes first, so the first kernel timed does not also pay for the device
  waking up.
- `GB/s` counts 8 bytes per element (one 4-byte read, one 4-byte write) divided by that
  time: the bytes a scan must move, not what the memory system moved.
- `of copy` is the kernel's rate as a share of the copy kernel's at the same block size.

The same rows go to the lesson pages (sections 11.6, 11.7 and 11.8 and the labs page). The
numbers are measurements of this machine at this size. They are not a property of the
kernels.

## Go further (untested)

Exercise 4 (p. 286), starting from your register-tiled kernel:

- **Vector loads** for the global-to-shared and shared-to-global passes. The book's
  technique on CUDA casts `float*` to `float4*`. No vendor documents that cast (infra
  facts A6, B6). On Metal, declare the buffer `device const float4*` instead, which is
  the documented way to read four floats at once (infra facts C7). Each thread then moves
  four adjacent elements per pass, so the passes and their indices change. n need not be
  a multiple of 4: handle the tail.
- **Padding** for the strided shared-memory accesses. Thread t reads `buffer_s[t*F + c]`,
  so neighbouring lanes are F floats apart. On CUDA's 32 banks of 4 bytes (infra facts
  A10) that is a bank conflict for every even F (our arithmetic). Pad the array so that
  index i lives at `i + i/32`, and measure whether it matters. Apple documents nothing
  about threadgroup-memory banks (infra facts C13), so on Metal the measurement is the
  only answer.
