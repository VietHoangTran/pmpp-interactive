# Lab 11.4: a scan of any length, hierarchically

Book: section 11.9 (pp. 273-276), Figs. 11.14 and 11.15.

Labs 11.1 and 11.2 scan each block's segment on its own. Here you join the segments into
one inclusive scan of all n elements, with the two three-kernel designs the book draws:

- **scan-scan-add** (Fig. 11.14). Kernel 1 scans every segment and writes each segment's
  total. Kernel 2 scans the totals. Kernel 3 adds the scanned total of all earlier
  segments to every element of a segment. The book counts 16 + 8/S bytes of global
  memory per element, with S elements per segment (p. 274).
- **reduce-scan-scan** (Fig. 11.15). Kernel 1 only reduces every segment to its total
  (a chapter 10 reduction). Kernel 2 scans the totals. Kernel 3 adds the earlier
  segments' total to the segment's first element and then scans the segment, which
  gives the final values directly. The book counts 12 + 12/S bytes per element (p. 275).

The book prints no code for either, and its kernel 2 scans the totals "in a single block"
(p. 273). That caps n at S × S elements. Here kernel 2 is the whole design again, applied
to the totals, until they fit in one segment. So any n works.

Contract. A block of `block` threads owns a segment of S = `COARSE_FACTOR` × `block`
consecutive elements. `COARSE_FACTOR` is 4, defined once per backend (`kernel.cu` for
CUDA, `launch.mm` for Metal); the driver reads it through `coarseFactor()`. The block
size is a power of two from 64 to 1024. n runs from 1 to below 2^31. An element at or
past n loads 0, the identity of +, and is not stored. `scratch` holds
`scratchFloats(n, block)` floats, and nothing is assumed about what it holds on entry.

**Your part:**

- **`scanSegment`**: your `RegisterTiledScanKernel` from lab 11.2, made into a device
  function, with two additions. It adds a `carry` to the segment's first element before
  the scan (Fig. 11.15's kernel 3), and it returns the segment's total.
- **`ScanSegmentsKernel` and `AddBlockSumsKernel`**: kernels 1 and 3 of Fig. 11.14.
- **`ReduceSegmentsKernel` and `ScanWithCarryKernel`**: kernels 1 and 3 of Fig. 11.15.
  The reduction covers the same segment the scan does, with the same coalesced loads.
- **`scratchFloats`, `launchScanScanAdd`, `launchReduceScanScan`**: how much scratch
  the recursion needs, and the launchers. Each launcher calls itself on the totals for
  kernel 2. Lay the scratch out level by level.
- **Metal**: the same kernels in `kernel.metal`, and in `launch.mm` `scratchFloats`,
  `encodeScanScanAdd` and `encodeReduceScanScan`. They write several dispatches into the
  one compute encoder the driver opened. Each dispatch sees what the one before it
  wrote. Apple documents two reasons: the plain `computeCommandEncoder` "dispatches its
  compute commands serially", and buffers made by the device are hazard-tracked by
  default (ch02 facts C9). Threadgroups never talk to each other inside a dispatch.

## Files

- Edit `work/kernel.cu` (CUDA, HIP and both emulators): `scanSegment`, the four kernels,
  `scratchFloats`, `launchScanScanAdd` and `launchReduceScanScan`.
- Edit `work/kernel.metal` (`scan_segment`, `scan_segments`, `add_block_sums`,
  `reduce_segments`, `scan_with_carry`) and `work/launch.mm` (`scratchFloats`,
  `encodeScanScanAdd`, `encodeReduceScanScan`). Each encode function returns false,
  without dispatching, for a block one of its pipelines cannot run.
- Provided in the starter, not part of the exercise: `warpScan` and `blockScan` from
  lab 11.1, `warpReduce` from lab 10.1, and on Metal the `fits` check. Swap in your own
  if you like.
- Provided, not edited, in `../shared-g13/`: `global_copy.cu` and `.metal` (a plain
  copy, the bandwidth baseline) and `global_launch.mm`.

## Commands

- Test: `python3 labs/pmpp.py run 11.4 --backend emu` first, then `python3 labs/pmpp.py run 11.4`.
- Barriers under ThreadSanitizer: `python3 labs/pmpp.py run 11.4 --backend race`.
- Measure: `python3 labs/pmpp.py run 11.4 --bench` (real devices only).
- The reference solution: add `--solution`.

The emulator run is slow, most of it in the three-level case below.

## What the tests do

- On real devices: n = 1, 2, 255, 256, 257, 1000, 4096, 4097, 65536, 70001, 2^20 + 1 and
  2^24 + 7, at blocks of 64, 256 and 1024. At block 64 (S = 256), 2^24 + 7 is four
  levels: 65,537 segments, then 257, then 2, then 1.
- On `emu`: n up to 4097 at blocks of 64 and 256, plus n = 70001 at block 64. That is
  274 segments, whose totals make 2 segments, whose totals make 1: three levels of the
  recursion. The emulator is slow at that size, so it runs it with the integers only.
- On `race`: n = 1000 and 5000 at block 64, which is 4 and 20 blocks, several at once.
- Three kinds of data. All ones, and integers from -3 to 3, have exact prefix sums in any
  order of addition while the sum of |x| stays at or below 2^24, so there they must match
  exactly. Past that (all ones at 2^24 + 7), and for values uniform in [-1, 1], the
  result is compared with Fig. 11.1's loop run in double, within 1e-5 times the sum of
  |x| up to that element, plus 1e-6. Why that much: each output goes through at most
  about 16 rounded additions per level of the hierarchy. With 4 levels, the worst case
  is 64 × 2^-24 ≈ 3.8e-6 times that sum.
- Each design runs twice on the same buffers. The first launch starts with the output
  and the scratch full of NaN, so a total read before it is written shows up. The second
  launch gets new data and whatever the first left in the scratch, so a kernel that adds
  to the scratch instead of overwriting it shows up too.
- Guard words follow the output and the scratch. A thread past n that stores is
  reported, and so is a launcher that uses more scratch than `scratchFloats` asked for.
  The input must come back unchanged.
- The provided copy kernel is tested too, so a broken provided file shows.

What each backend can catch:

- `emu` runs the threads one at a time in three fixed orders, and fills shared memory
  with garbage for every block. A wrong index, a missing barrier and an out-of-bounds
  access fail there every time.
- `race` runs the threads of a block, and several blocks, at the same time under
  ThreadSanitizer. It reports a missing barrier inside your kernels. Nothing in this lab
  should pass data between blocks inside one launch. Every hand-off happens between
  launches, so it has nothing else to report.
- The emulators are slow and model no time. Only a real device measures what the two
  designs cost.

## Where the backends differ (not in the book)

- **HIP**: as in lab 11.1, the lane mask needs its own 64-bit type, and `warpSize` is 64
  on gfx9 and 32 on later AMD parts (ch11 facts B1). Nothing here uses atomics. Not
  built or run with ROCm: unverified.
- **Metal**: this is Metal's global scan. Metal documents no forward progress between
  threadgroups, and an Apple engineer called the single-pass scan's assumptions
  undefined behaviour (ch11 facts C5). So lab 11.5's single kernel has no Metal version.
  llama.cpp ships the same three-dispatch scan-scan-add on Metal, and PyTorch's MPS
  backend a three-pass reduce, scan and carry (ch11 facts D1, D4).
- **Metal**: a buffer can be bound at a byte offset (`setBuffer:offset:atIndex:`), which
  is how one scratch buffer holds every level. For a `float` array the offset must be a
  multiple of 4 (ch06 facts C3). There is no null buffer to say "no totals". So where
  CUDA could pass `nullptr`, Metal gets a separate kernel.
- **Libraries**: CUB's `DeviceScan` does a single pass with decoupled look-back, and
  rocPRIM's scan does the same; MPSGraph has `cumulativeSum` (ch11 facts A8, B2, C6).

## Before you benchmark, predict

Write these down first. The lesson asks for the first under the id in brackets.

1. How many times faster than scan-scan-add will reduce-scan-scan run at n = 2^26 on
   your device? The byte count says 16/12, about 1.33 (`ch11/11.9/p-rss`, asked in
   section 11.9).
2. What share of the copy kernel's rate will each design reach? If both moved their
   bytes as fast as the copy moves its 8, scan-scan-add would take twice the copy's time
   and reduce-scan-scan one and a half times.

## Reading the output

The benchmark scans n = 2^26 floats. It uses fewer if your device reports less than
four times the 512 MB of input and output. It runs at blocks of 256 and 1024 (S = 1024
and 4096). For each block size it times the copy, then scan-scan-add, then
reduce-scan-scan, and checks each result before reporting it.

- `time ms` is the best of five runs after one untimed warm-up. It covers every kernel
  of every level. A quarter of a second of untimed copies comes first, so the first
  kernel timed does not also pay for the device waking up.
- `Gelem/s` is n divided by the time.
- `book B/elem` is the book's count of global-memory bytes per element (pp. 272-275):
  8 for the copy, 16 + 8/S and 12 + 12/S for the designs. `GB/s book` is that count
  times n, divided by the time. It is a model of the traffic, not a measurement of it,
  so a design can show more GB/s by this count than the copy does.
- `of copy` is the copy's time as a share of the design's.
- The last line for each block size gives the measured ratio of the two designs beside
  the ratio of the book's byte counts.

The same rows go to section 11.9 and to the labs page, beside lab 11.5's. The numbers
are measurements of this machine at this size. They are not a property of the designs.
Run the benchmark more than once before you compare with your prediction. If the runs
disagree, that spread is part of what you measured.

## Go further

- p. 276 warns that some chapter 10 reductions assume a commutative operator. Which line
  of your `ReduceSegmentsKernel` assumes it? Look at which elements one thread combines.
  How would you rewrite it for an operator that is associative but not commutative?
- Raise `COARSE_FACTOR` to 8. S doubles, so there are fewer levels, and the 8/S and 12/S
  terms halve. Does the time move? Mind the shared-memory limit in the comment at the
  top of `kernel.cu`.
- Make the scan exclusive (section 11.1), as memory allocation and top-p sampling need.
  Which kernels change in each design?
