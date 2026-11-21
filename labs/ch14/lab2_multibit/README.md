# Lab 14.2: multi-bit radix sort with local sorting

Book: sections 14.6 to 14.8 (pp. 339-345, Figs. 14.8, 14.9, 14.12 and 14.13) and exercises 1,
2 and 3 (p. 347). The book gives no code for any of these; the figures give the design. A bare
item such as (A4) or (C4) below is in `notes/today/ch14-facts.md`.

This lab's baseline is lab 14.1's part B, provided ready-made in `labs/ch14/shared-g11/`, so
you can start here; lab 14.1 is optional.

One stable LSD radix sort of (key, value) pairs that does all three exercises at once: local
sorting in shared memory (exercise 1), a radix of r bits (exercise 2) and c keys per thread
(exercise 3). r runs from 1 to 8 and c from 1 to 8, both chosen at run time. A block has 256
threads and owns a tile of 256c consecutive pairs; the last tile may be partial, and its
padding is neither counted nor written.

One pass over the digit `(key >> shift) & (2^r - 1)` is three launches, as section 14.6 lays
it out:

1. `radixLocalSort(keysIn, valsIn, keysLocal, valsLocal, table, localStarts, n, shift,
   radixBits, keysPerThread)`. Load the tile into shared memory, coalesced. Sort it by the
   digit with r stable one-bit splits, least significant digit bit first (p. 342). Each split
   is a block-level exclusive scan: each thread counts the ones among its c keys, a scan of
   the 256 thread totals gives the ones before them (chapter 11's scan-scan-add inside a
   block), and Fig. 14.6's two formulas, applied inside the tile, give every pair its new
   place. Then write the sorted tile back to the same tile position, the block's 2^r bucket
   sizes into the digit-major table `table[d * numBlocks + block]` (Figs. 14.9 and 14.12: all
   blocks' counts for digit 0, then for digit 1, ...), and where each bucket starts inside
   the tile into `localStarts[block * 2^r + d]`.
2. The provided `launchExclusiveScan(table, 2^r * numBlocks, scratch)`: every local bucket's
   global start.
3. `radixScatterLocal(keysLocal, valsLocal, keysOut, valsOut, tableScanned, localStarts, n,
   shift, radixBits, keysPerThread)`: the pair at tile position p with digit d goes to
   `tableScanned[d * numBlocks + block] + (p - localStarts[block * 2^r + d])`. Consecutive
   threads take consecutive tile positions, so inside one bucket they write consecutive
   addresses. That is the coalescing section 14.6 is after.

`launchRadixSortMulti(keys, vals, tmpKeys, tmpVals, table, localStarts, scanScratch, n,
radixBits, keysPerThread)` runs ceil(32 / r) passes and leaves the result in `keys`/`vals`.
When r does not divide 32 (r = 3, 5, 6, 7), the last pass has only the bits that remain; a
shift by 32 or more is undefined in C++, so the last digit width must shrink, not the mask
alone. The kernels take `keysPerThread` as their last parameter (the sizes of everything
depend on it).

## Files

- Edit `work/kernel.cu` (CUDA, HIP and both emulators): `radixLocalSort`,
  `radixScatterLocal`, `launchRadixSortMulti`. Do the block-level scan in shared memory with
  barriers. A warp-shuffle scan needs a lane mask, and on HIP the mask must be a 64-bit
  integer on every target (`notes/today/infra-facts.md` B3); the shared-memory form has no
  mask to get wrong. Never assume a warp width.
- Edit `work/kernel.metal` (`radix_local_sort`, `radix_scatter_local`; the signatures are
  given). The block-level scan may use `simd_prefix_exclusive_sum` and `simd_sum` inside each
  SIMD-group (infra-facts C2) with one threadgroup slot per SIMD-group for their totals. The
  SIMD-group width is not a documented constant (infra-facts C5): `launch.mm` reads it from
  the pipeline's `threadExecutionWidth`, and the kernel has it as `[[threads_per_simdgroup]]`.
  A scan done this way adds up the thread totals in (SIMD-group, lane) order, and how a
  threadgroup is split into SIMD-groups is implementation-defined (`notes/today/ch04-facts.md`
  C2). So in the splits, give each thread the tile positions of its rank in that order,
  `sg * width + lane`, not of `thread_position_in_threadgroup`; the coalesced loads and stores
  may keep the thread position.
- Edit `work/launch.mm` (`encodeRadixSortMulti`): every pass, the provided scan included, in
  the one serial compute encoder it is given (`notes/today/ch02-facts.md` C9). Threadgroup
  memory is the tile (256c x 8 bytes) at `[[threadgroup(0)]]`, the SIMD-group totals at
  `[[threadgroup(1)]]`, and whatever the kernel declares itself
  (`staticThreadgroupMemoryLength`). Lengths are multiples of 16 bytes and the sum must fit
  `maxThreadgroupMemoryLength` (`notes/today/ch05-facts.md` C1, C3, C8); return false before
  encoding anything when it does not.
- Provided, not edited: `labs/ch14/shared-g11/scan.*` (the scan of lab 14.1) and
  `onebit_global.cu`, `onebit_global.metal`, `onebit_global_launch.mm`: lab 14.1's design,
  one bit per pass with a scan of n + 1 bits and every key written straight to its global
  place. It is the baseline for section 14.6's claim.

## Commands

- Test: `python3 labs/pmpp.py run 14.2 --backend emu` first (several minutes: every GPU
  thread runs on the CPU), then `python3 labs/pmpp.py run 14.2`.
- Shared-memory races: `python3 labs/pmpp.py run 14.2 --backend race`. The tile is
  rewritten by every split, so a missing barrier lets a thread read a half-written tile.
  Planted on 2026-11-20 in a copy of the reference (the `__syncthreads()` at the end of the
  split loop removed), it gave:
  - `emu`: r = 1 passes (step 3's barrier covers the gap); r = 2, c = 1 fails in the first
    thread order with a wrong order, then AddressSanitizer reports a heap-buffer-overflow at
    the tile write `vals_s[dst] = ...`.
  - `race` (re-run on 2026-11-21 after the two-token barrier fix, four runs, each stopped in
    under a minute): r = 1 passes; r = 2, c = 1 and c = 2 fail. Every run now names the
    missing barrier's pair: a write of the tile in one split (`vals_s[dst] = ...`, or
    `keys_s[dst]`) against a read of it at the start of the next (`myVals[k] = vals_s[p]`);
    some runs also report the scan's buffer `scan_s` reused across splits and a write-write
    race on `start_s` in step 3. The run then stops with a ThreadSanitizer SEGV at
    `keys_s[dst] = ...` (a corrupted index; the race backend has no bounds checks). When this
    defect was first planted, the race flavour shared one ThreadSanitizer token between all of
    a block's `__syncthreads()`, and only the `start_s` race was reported; it now alternates
    two tokens per barrier (`labs/ch14/snippets/README.md`, the wrap kernel). Still, a clean
    race run is not evidence that every barrier is there. Run `emu` first.
- Measure: `python3 labs/pmpp.py run 14.2 --bench` (real devices only).
- The reference solution: add `--solution`.

## What the tests check

- The provided baseline, so a broken provided file shows up as itself.
- On a GPU, every r in 1, 2, 3, 4, 5, 8 (3 and 5 do not divide 32) with every c in 1, 2, 4,
  at n = 1, 255, 256, 257, 1000, 4097, 65537 and 2^20, with eight key patterns: full-range
  random, 0..15 (heavy ties), all equal, sorted, reversed, powers of two, "every key shares
  its low byte" (on the first passes each block puts all its keys into one bucket, so the
  table is mostly zeros) and "keys below 256" (every digit above bit 7 is zero).
- The CPU emulators pay for every thread of every block of up to 96 launches per sort, so
  there each (r, c) with c in 1, 2 (except r = 1, c = 2, the slowest) runs at n = 1, 257 and
  1000 with key patterns that rotate across the configurations, plus every pattern at r = 4,
  c = 2, n = 600 (two tiles), the two sizes around one tile at r = 5, the two zero-heavy
  patterns at r = 8, c = 2, and (emu only) n = 4097 at r = 8, c = 1, whose table of 4352
  entries needs a scan of several tiles.
- The keys must equal `std::stable_sort`'s and the values, which are the input positions,
  must equal its permutation exactly: among equal keys the positions must increase.
- Before every run the driver writes 0xCDCDCDCD into the whole table, `localStarts`, the
  scan's scratch, the temporary buffers and 64 guard slots past the end of every pair array.
  A bucket count that is not written for this pass, or a write into the padding, fails.
- A configuration the device cannot run (threadgroup or shared memory over the limit) is
  printed as skipped with the reason; it is not counted as a pass.

## Before you benchmark

Make three predictions (the pages ask for them):

1. Will local sorting at r = 1, c = 1 beat the provided direct scatter, and by what factor?
   (`ch14/14.6/p-local-gain`, on page 14.6)
2. Which of r = 1, 2, 4, 6 and 8 is fastest at c = 1? (`ch14/14.7/p-best-r`, on page 14.7)
3. How many times faster than one CPU thread's `std::stable_sort` is your best configuration
   at 2^22 pairs? (`ch14/labs/p-lab2-cpu`)

The explorers on pages 14.7 and 14.8 show what the model expects. Count for yourself:

- **Passes**: ceil(32 / r). Each pass reads and writes every pair twice (local sort, then
  scatter): 32 bytes per pair per pass. The baseline moves about 44 bytes per pair per pass
  (lab 14.1's count) but writes in two runs per warp at most; it has 32 passes.
- **Runs per warp in the scatter**: with 2^r buckets, a warp's keys fall into up to
  min(2^r, warp width) separate runs of addresses (A10). Local sorting gathers each bucket's
  keys together inside the tile, so a run is as long as the tile's bucket: about
  256c / 2^r keys for random keys.
- **Table size**: 2^r x ceil(n / 256c) entries, written, scanned and read every pass. At
  r = 8, c = 1 that is as many entries as there are keys.
- **Shared memory**: 256c x 8 bytes for the tile plus the scan and bucket arrays; at c = 8
  the tile alone is 16 KiB. Where it runs out depends on your device (`python3 labs/pmpp.py
  doctor`).

The benchmark starts at n = 2^22 and doubles n while the baseline takes under 20 ms (within
an eighth of the device's memory). It times the baseline (`onebit_global`), your sort
(`local`) at r = 1, 2, 4, 6, 8 with c = 1 and at c = 1, 2, 4, 8 with r = 4 and r = 8,
skipping what the device cannot run, and `std::stable_sort` of the same pairs on one CPU
thread with `std::chrono` (a host measurement). Each row is the best of 5 runs after a
warm-up, on random full-range keys, and every timed result is checked before it is reported.
Time is the device time of the command buffer on Metal and the wall time around the launches
and the final synchronisation on CUDA and HIP. "Counted GB/s" is counted, not measured
traffic: 32 bytes per pair per pass, divided by the time, with the table and the scan left
out. The numbers are measurements of your machine, not targets.

## Portability

- HIP: the reference scans in shared memory, with no warp intrinsics and no lane masks.
  `extern __shared__` keeps its spelling (`notes/today/ch05-facts.md` B4). Not compiled or
  run on AMD hardware here: unverified.
- Metal: SIMD-group prefix sums need GPU family Apple7, which every Apple silicon Mac is
  (infra-facts C2). The kernels never assume a SIMD-group width, and the local sort ranks its
  threads by SIMD-group and lane wherever positions must follow the scan's order, because the
  split of a threadgroup into SIMD-groups is the implementation's (ch04-facts C2). The
  provided scan (`shared-g11/scan.metal`) does the same. Metal documents no forward
  progress between threadgroups (C4), which is why every pass is three dispatches and not a
  single-pass sort with look-back.
- What the libraries choose (A4): CUB's radix sort uses 8-bit digits in its single-pass
  (Onesweep) path and 7-bit digits in its multi-pass path for keys wider than a byte, with a
  per-architecture keys-per-thread setting; rocPRIM uses 8 bits in 374 of its 384 tuned
  Onesweep configurations (B3); PyTorch's MPS radix sort uses 8 bits for 16- and 32-bit keys
  (D6). Those are their tunings for their kernels, not a rule for yours.
