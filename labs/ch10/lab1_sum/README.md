# Lab 10.1: one sum, three kernels

Book: sections 10.6 to 10.10 (pp. 236-248), Figs. 10.10, 10.13, 10.16, 10.18 and 10.20,
and exercise 5 (p. 250).

You write three sum reductions and time them against the book's first two, which are
provided. The two-stage block reduction you write in (b) is the building block that
lab 10.3 reuses for the softmax and argmax of milestone M4.

**Your part:**

- **(a) `SharedMemorySumReductionKernel`** (Fig. 10.10). Block b reduces row b of
  `2*blockDim.x` floats into `out[b]`: each thread adds its two elements as it loads them,
  then a tree in shared memory finishes the row. Fig. 10.10 writes `*output` but has no
  `output` parameter; restore it.
- **(b) `warp_reduce` and `TwoStageWarpLevelSumReductionKernel`** (Figs. 10.13 and 10.16),
  made portable. Same rows as (a). Every warp reduces its values with shuffles, lane 0 of
  each warp hands its sum over through shared memory, one barrier, and warp 0 reduces
  those sums.
  - The lane mask gets its own type: `unsigned int` on CUDA, 64 bits on HIP. Never the
    literal `0xffffffff`.
  - The shuffle loop starts at `warpSize/2`, and lane and warp come from `threadIdx.x`
    and `warpSize`, not from a constant 32.
  - The per-warp array is sized for the narrowest documented warp: `1024/32` entries.
  - The stage-2 read is guarded. Lanes with no warp behind them use 0.
- **(c) `CoarsenedSumReductionKernel`** (Figs. 10.18 and 10.20, completed, plus exercise 5).
  Any n. The coarsening factor is a run-time argument, and each thread first adds its
  `2*coarse` elements serially, `blockDim.x` apart. An element past the end contributes 0.
  The same two-stage block reduction follows, and then one
  `cuda::atomic_ref<float, cuda::thread_scope_device>` `fetch_add(..., cuda::memory_order_relaxed)`
  per block, the book's spelling (under HIP, the legacy `atomicAdd`: see below). The
  launcher sets `*out` to 0 (`cudaMemset`) and launches
  `ceil(n / (2*coarse*block))` blocks.

Every launcher takes the block size, a power of two from 64 to 1024, so the tests can try
several.

## Files

- Edit `work/kernel.cu` (CUDA, HIP and both emulators): the three kernels, `warp_reduce`,
  and `launchSharedMemorySum`, `launchTwoStageWarpSum` and `launchCoarsenedSum`.
- Edit `work/kernel.metal` (`shared_memory_sum`, `warp_reduce`, `two_stage_warp_sum`,
  `coarsened_sum`) and `work/launch.mm` (`encodeSharedMemorySum`, `encodeTwoStageWarpSum`,
  `encodeCoarsenedSum`). Each encode function returns false, without dispatching, for a
  block its pipeline cannot run: more threads than the pipeline's
  `maxTotalThreadsPerThreadgroup`, or, for the two SIMD-group kernels, a block that is not
  a whole number of SIMD-groups of `threadExecutionWidth` lanes or has more SIMD-groups
  than one SIMD-group has lanes.
- Provided, not edited: the book's `SimpleSumReductionKernel` (Fig. 10.5) and
  `ConvergentSumReductionKernel` (Fig. 10.8) in `../shared/inplace_sum.cu`,
  `../shared/inplace_sum.metal` and `../shared/inplace_launch.mm`. Each block reduces one
  row, in place.

## Commands

- Test: `python3 labs/pmpp.py run 10.1 --backend emu` first, then `python3 labs/pmpp.py run 10.1`.
- Atomics: `python3 labs/pmpp.py run 10.1 --backend race`.
- Measure: `python3 labs/pmpp.py run 10.1 --bench` (real devices only).
- The reference solution: add `--solution`.

## What the tests do

- (a), (b) and both provided kernels: 1, 3 and 37 rows at blocks of 64, 128, 256 and 1024
  threads (emulator: 1 and 3 rows at 64 and 256; `race`: 2 rows at 64; Metal: the blocks
  the pipeline allows, and the driver prints any it skips). The provided kernels are tested
  too, so a broken provided file shows.
- (c): n = 1, 2, 31, 32, 33, 255, 256, 257, 1000, 4097, 100003 and 2^22 (the last two on
  real devices only), at coarse 1, 2, 3 and 8, with blocks of 64 and 256. On `race`:
  n = 33, 1000 and 4097 with blocks of 64, so that most cases have at least 4 blocks
  running at the same time.
- Three kinds of data. All ones and integers from -3 to 3 have exact float sums whatever
  the order of the additions, so they must match exactly. Values uniform in [-1, 1] must
  be within 1e-6 times the sum of |x|, plus 1e-6. The absolute part is there because
  CUDA's float atomics on global memory flush denormals to zero (ch10 facts A6).
- Outputs start as NaN, so a row your kernel never writes is reported. Every case of (c)
  is launched twice on the same output, and the launcher must reset `*out` to 0 each time.

What each backend can catch:

- `emu` fills shared memory with garbage for every block and runs the threads one at a
  time in three fixed orders. A misplaced barrier, an unguarded stage-2 read, or a thread
  that returns before the shuffles fails there on every run, though not always under all
  three orders.
- `race` runs blocks concurrently under ThreadSanitizer. A plain `*output += partialSum`
  in place of the atomic is reported with its source line. `emu` cannot see a missing
  atomic.
- Metal is a real GPU. On the machine this lab was built on, an unguarded stage-2 read
  passed there, because the unwritten threadgroup memory happened to hold zeros. Start on
  `emu`.
- Nothing on a Mac can catch the two HIP mistakes: a `0xffffffff` mask and a loop that
  starts at 16. Both emulators model 32-lane warps, and Metal has no masks. HIP rejects
  the 32-bit mask at compile time, because its `_sync` functions `static_assert` a 64-bit
  mask. On a 64-lane wavefront, a loop that starts at 16 sums each half of the wavefront
  separately (ch10 facts B1; infra facts B3). Check those two by reading your code.

## Where the backends differ (not in the book)

- **HIP**: the mask type above. `warpSize` is 64 on gfx9 (Instinct) and 32 on later AMD
  parts. A 1024-thread block then has 16 warps, which is why the stage-2 read needs its
  guard (ch10 facts B2). `<cuda/atomic>` comes from libhipcxx on HIP, which the course's
  HIP build links only where ROCm provides it (infra facts B2), so the file includes it only when `__HIPCC__`
  is not defined, and under HIP the commit is the legacy `atomicAdd(output, value)`, which
  is also relaxed and device-wide (infra facts A1, B1). None of this has been built or run
  with ROCm here: unverified.
- **Metal**: no masks; `simd_shuffle_down(v, delta)`. Metal does not document how a
  threadgroup is split into SIMD-groups, so the SIMD-group index and the lane come from
  `[[simdgroup_index_in_threadgroup]]` and `[[thread_index_in_simdgroup]]`, not from
  `t / width` (ch04 facts C2). The width is not documented either: read
  `[[threads_per_simdgroup]]`. Float atomics need `device atomic_float*` and
  `atomic_fetch_add_explicit` with `memory_order_relaxed`: Metal 3, Apple7 and later
  (infra facts C1). Metal has no `cudaMemset` inside a compute pass. The buffer is in
  shared storage and its command buffer has not been committed yet, so `launch.mm` writes
  the 0 through `out.contents` (ch05 facts C6).
- The provided Metal kernels pass partial sums through device memory, so their barrier is
  `threadgroup_barrier(mem_flags::mem_device)` (ch05 facts C2).

## Before you benchmark, predict

Write these down first. The lesson asks for them under the ids in brackets.

1. How many times faster than the simple kernel the two-stage kernel will run per row
   (`ch10/labs/p-lab1-warp`).
2. Which coarsening factor, from 1 to 128, will be fastest on one array of 2^26 floats
   or more (the driver doubles n on a fast device, up to 2^28, and starts lower only
   where memory is short; see "Reading the output") (`ch10/10.10/p-coarse`).
3. How many distinct results 20 runs of the atomic version will give on the same input
   (`ch10/10.9/p-distinct`).

The book's analysis for 256 elements says 30% against 66% execution-resource efficiency
(simple against convergent, pp. 231 and 234), and 141 against 36 against 9 global memory
requests (simple, convergent, shared memory, pp. 235 and 237). Your device decides how much
of that becomes time.

## Reading the output

The benchmark prints three tables. The same rows go to the lesson pages (sections 10.8,
10.9, 10.10 and the labs page).

- **rows**: 2^24 floats as rows of 512 (block 256), for the four row kernels. Each timed
  run gets its own freshly uploaded copy of the input, because the provided kernels
  overwrite theirs. A quarter of a second of untimed work comes first, so the first
  kernel timed does not also pay for the device waking up. `time ms` is the best of five
  runs. `GB/s` is the input's bytes (4 per float) divided by that time: the bytes a sum
  must read, not what the memory system moved.
- **any length**: the coarsened kernel at coarse 1 to 128 on one array. n starts at 2^26
  and doubles while coarse 1 takes under half a millisecond, up to 2^28 and a quarter of
  the memory your device reports. `blocks` is how many blocks were launched. On CUDA and
  HIP the time includes the launcher's `cudaMemset` of one float. On Metal it does not,
  because the zero is written from the CPU.
- **determinism**: 20 runs of the coarsened kernel on the same 2^24 floats at coarse 1,
  and the number of distinct bit patterns among the 20 results.

The numbers are measurements of this machine at these sizes. They are not a property of
the kernels.

## Go further

The determinism table is the difference between the book's two multi-block designs. An
atomic commit adds the block sums in whatever order the blocks finish. Partial sums
written per block and reduced by a second kernel keep one order (pp. 245-246). CUB names
the two `not_guaranteed` and `run_to_run`, and makes `run_to_run` the default for
`cub::DeviceReduce` (ch10 facts A13). Try it: write the per-block sums of (c) to an array
and reduce that array with your kernel (b), then count distinct results again. It matters
for an LLM whenever the same prompt must give the same tokens. A softmax denominator that
changes in its last bits can flip a near-tie in greedy decoding.
