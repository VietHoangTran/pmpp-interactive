# Lab 11.1: one block scan, three ways

Book: sections 11.2 to 11.4 (pp. 254-265), Figs. 11.3, 11.5 and 11.8 to 11.10.

You write two block-level inclusive scans and time them against the book's two-barrier
kernel (Fig. 11.3), which is provided. The block scan you write in (b) is the building
block that labs 11.2, 11.3 and 11.6 reuse.

Every kernel has the same contract. Block b scans its own segment of `blockDim.x`
elements, inclusive, with +, and knows nothing of the other blocks. n can be anything
from 1 up. A thread past the end loads 0, the identity of +, takes part in every barrier
and shuffle, and stores nothing. The block size is a power of two from 64 to 1024,
chosen at run time, so there is no `SEG_SIZE`: use `blockDim.x`, and size shared arrays
for the largest block.

**Your part:**

- **(a) `KoggeStoneDoubleBufferKernel`** (Fig. 11.5). Start from the provided Fig. 11.3
  and remove its second barrier with two shared buffers that swap roles every iteration.
  That leaves one barrier per iteration. Threads that do not add must still copy their
  value across, and the result is read from `inBuffer_s` after the last swap.
- **(b) `warpScan`, `blockScan` and `WarpScanKernel`** (Figs. 11.8 to 11.10), made
  portable:
  - The lane mask gets its own type: `unsigned int` on CUDA, 64 bits on HIP. Never the
    literal `0xffffffff`.
  - The shuffle loop runs while `stride < warpSize`. `laneIdx()` and `warpIdx()` come
    from `threadIdx.x` and `warpSize`, not from a constant 32.
  - The array of warp totals is declared in the kernel, sized for the narrowest
    documented warp (`1024/32` entries), and passed to `blockScan` as a pointer. Fig. 11.9
    declares it inside `blockScan`, which Metal refuses and HIP does not document.
  - The stage-2 read is guarded. Lanes with no warp behind them feed 0.
  - Every thread calls `blockScan`, in range or not.
- **Metal only, (b2) `warp_scan_prefix`**: the same block scan with each SIMD-group scan
  done by one call to `simd_prefix_inclusive_sum`, so you can compare the one-call
  function with your loop.

## Files

- Edit `work/kernel.cu` (CUDA, HIP and both emulators): `KoggeStoneDoubleBufferKernel`,
  `warpScan`, `blockScan`, `WarpScanKernel`, `launchKoggeStoneDB` and `launchWarpScan`.
- Edit `work/kernel.metal` (`ks_double_buffer`, `warp_scan`, `block_scan_shuffle`,
  `warp_scan_shuffle`, `block_scan_prefix`, `warp_scan_prefix`) and `work/launch.mm`
  (`encodeKoggeStoneDB`, `encodeWarpScanShuffle`, `encodeWarpScanPrefix`). Each encode
  function returns false, without dispatching, for a block its pipeline cannot run: more
  threads than the pipeline's `maxTotalThreadsPerThreadgroup`, or, for the two
  SIMD-group kernels, a block that is not a whole number of SIMD-groups of
  `threadExecutionWidth` lanes or has more SIMD-groups than one SIMD-group has lanes.
- Provided, not edited, in `../shared-g11/`: `bs_two_barrier.cu` and `.metal` (Fig. 11.3,
  the baseline), `bs_copy.cu` and `.metal` (a plain copy, for the bandwidth baseline),
  and `bs_launch.mm`.

## Commands

- Test: `python3 labs/pmpp.py run 11.1 --backend emu` first, then `python3 labs/pmpp.py run 11.1`.
- Barriers: `python3 labs/pmpp.py run 11.1 --backend race`.
- Measure: `python3 labs/pmpp.py run 11.1 --bench` (real devices only).
- The reference solution: add `--solution`.

## What the tests do

- n = 1, 31, 32, 33, 255, 256, 257, 1000, 4097, 100003 and 2^22, at blocks of 64, 128,
  256 and 1024. The emulator runs n up to 4097 at blocks of 64 and 256. `race` runs
  n = 33 and 1000 at block 64, which is 1 and 16 blocks, so several blocks run at once.
  On Metal the SIMD-group kernels run at the blocks the pipeline allows, and the driver
  prints any it skips.
- Three kinds of data. With all ones, element i must become `i mod block + 1`. Integers
  from -3 to 3 have exact prefix sums in any order of addition, so they must match
  exactly too. Values uniform in [-1, 1] are compared with Fig. 11.1's loop run in
  double over each segment, within 1e-6 times the sum of |x| over the segment up to that
  element, plus 1e-6.
- The output starts as NaN, so an element your kernel never writes is reported. Guard
  words follow the output up to the end of the last block and 16 more. A thread past n
  that stores is reported.
- The provided kernels are tested too, so a broken provided file shows.

What each backend can catch:

- `emu` fills shared memory with garbage for every block and runs the threads one at a
  time in three fixed orders. A missing copy branch, a missing barrier in (a), and
  `threadIdx.x` in place of the lane in `warpScan` each fail there on every run.
- `race` runs the threads of a block, and several blocks, at the same time under
  ThreadSanitizer. A missing barrier is reported with both source lines.
- An unguarded stage-2 read passes on every backend here. In a scan, the garbage
  a spare lane reads only flows into the lanes after it, and nobody reads those.
  What the guard prevents is a read past the end of the array. On a 64-lane wavefront,
  lanes 32 to 63 would read past the 32 entries (our reasoning, from ch11 facts B1).
  Keep it.
- Nothing on a Mac can catch two HIP mistakes. Both emulators model 32-lane warps, and
  Metal has no masks. So check these two by reading your code:
  - A `0xffffffff` mask. HIP rejects it at compile time, because its `_sync` functions
    `static_assert` a 64-bit mask (ch11 facts B1, infra facts B3).
  - A literal 32 for the warp width. On a 64-lane wavefront, a loop that stops at
    stride 16 scans each 32-lane half on its own. The result is right only if every
    other count (the lane, the warp, the last-lane test) also uses 32. Mix the two and
    it is wrong (ch11 facts B1).

## Where the backends differ (not in the book)

- **HIP**: the mask type above. `warpSize` is 64 on gfx9 (Instinct) and 32 on later AMD
  parts, and since ROCm 7.0 it is not a compile-time constant (ch05 facts B5, ch10 facts
  B2). Lanes below the stride get their own value back from `__shfl_up_sync`, as on CUDA
  (ch11 facts A3, B1). None of this has been built or run with ROCm here: unverified.
- **Metal**: no masks; `simd_shuffle_up(v, delta)` leaves the lower `delta` lanes
  unchanged, the same rule as CUDA. Metal does not document how a threadgroup is split
  into SIMD-groups, so the lane, the SIMD-group index and the width come from
  `[[thread_index_in_simdgroup]]`, `[[simdgroup_index_in_threadgroup]]` and
  `[[threads_per_simdgroup]]` (infra facts C5). A helper function cannot declare
  threadgroup memory: the compiler says "variables in the threadgroup address space
  cannot be declared in a non-qualified function" (ch11 facts C2).
- `simd_prefix_inclusive_sum` sums over the *active* threads of the SIMD-group. A thread
  that returned early would neither add nor count, which is one more reason threads past
  n stay in the kernel and feed 0. Apple does not document the order of its additions,
  so its last bits may differ from your loop's (ch11 facts C1).
- CUDA has no one-call warp scan for `float`. The library routes are `cub::WarpScan` and
  `cooperative_groups::inclusive_scan` on a tile (ch11 facts A4).

## Before you benchmark, predict

Write these down first. The lesson asks for them under the ids in brackets.

1. At blocks of 1024, how many times faster than the two-barrier kernel your
   warp-shuffle scan will run (`ch11/labs/p-lab1-warp`).
2. How many times faster than the two-barrier kernel double buffering alone will be
   (`ch11/11.3/p-double`, asked in section 11.3).
3. On Metal: will your shuffle loop be faster, slower or the same as the one-call
   `simd_prefix_inclusive_sum` version? By how much?

The book counts barriers: two per iteration in Fig. 11.3, one per iteration in Fig. 11.5,
two in all for Fig. 11.9 (pp. 258-265). It also says a scan moves at least 8 bytes per
element, and that even a copy does not reach the published bandwidth (pp. 272-273). How
much of the barrier count becomes time depends on your device.

## Reading the output

The benchmark scans n = 2^24 floats (fewer if your device reports less than four times
the 128 MiB of input and output that needs), one segment per block, at blocks of 256, 512 and 1024. For each
block size it times the copy kernel first, then the two-barrier kernel, your (a), your
(b) and, on Metal, (b2).

- `time ms` is the best of five runs after one untimed warm-up. A quarter of a second of
  untimed copies comes first, so the first kernel timed does not also pay for the device
  waking up.
- `GB/s` counts 8 bytes per element (one 4-byte read, one 4-byte write) divided by that
  time: the bytes a scan must move, not what the memory system moved.
- `of copy` is the kernel's rate as a share of the copy kernel's at the same block size.
  The copy is the most this machine gave a kernel that moves the same bytes, so it is
  the yardstick here, not the published peak.

The same rows go to the lesson pages (sections 11.3 and 11.4 and the labs page). The
numbers are measurements of this machine at this size. They are not a property of the
kernels.

## Go further

- Make the warp scan identity-filling on Metal: `simd_shuffle_and_fill_up(val, 0.0f, d)`
  copies the identity into the lower lanes, so the lane test disappears. It needs Apple8
  or later, so not M1 (ch11 facts C1). MLX uses it for 8-byte types (ch11 facts D3).
- Turn the inclusive scan into an exclusive one, as top-p sampling needs (section 11.1):
  shift by one lane with one more shuffle, or subtract the input. One of the two returns
  exactly what your inclusive scan computed; the other adds a rounding error of its own.
  Which, and why does it matter at a top-p boundary?
