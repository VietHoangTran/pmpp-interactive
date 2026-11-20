# Lab 4.2: what a divergent warp costs

Book: section 4.5 (pp. 79-82), Figs. 4.9 and 4.10. The book shows the two shapes as
fragments inside the figures, not as kernels. These three kernels are ours.

Section 4.5 says a warp whose threads take different paths is executed in passes, one
per path, with the other threads inactive. It gives no price. This lab measures one on
your device, with a kernel whose branch pattern the driver controls.

**Your part:** three kernels and their launches. Every array has `n` elements and thread
`i` of the grid owns element `i`.

- `branchKernel`: `out[i]` is `pathA(in[i], iters)` where `which[i] == 0`, and
  `pathB(in[i], iters)` everywhere else. Each thread runs one of the two. This is
  Fig. 4.9's shape, with data in place of `threadIdx.x < 24`.
- `selectKernel`: the same `out[i]`, written so that no thread's control flow depends
  on `which[i]`.
- `loopKernel`: `out[i]` is `in[i]` after `trips[i]` steps of `stepA`. This is
  Fig. 4.10's shape. `trips[i]` can be 0.

## Files

- Edit `work/kernel.cu` (CUDA, HIP, emulator): the three kernels and `launchBranch`,
  `launchSelect`, `launchLoop`.
- Edit `work/kernel.metal` (`branch_kernel`, `select_kernel`, `loop_kernel`) and
  `work/launch.mm` (`encodeBranch`, `encodeSelect`, `encodeLoop`).
- Provided beside them, not to be edited: `paths.h` and `paths.metal.h`. They define
  `pathA`, `pathB`, `stepA` and `stepB`. The two paths cost the same, `iters` steps
  each. Read the comment in `paths.h` for why they are loops and not one-line
  expressions.
- Provided, used by the driver only: `provided/simd_probe.metal`.

Two Metal details:

- `kernel.metal` says `#include "paths.metal.h"`, but the driver compiles it at run
  time, and Apple documents that source compiled that way "may only import the Metal
  default library" (`notes/today/ch05-facts.md`, C7). The driver pastes the header in
  before compiling. Compiler messages still name `kernel.metal` and its line numbers.
- Dispatch whole threadgroups (`dispatchThreadgroups`), as the comment in `launch.mm`
  explains.

## Commands

- Test: `python3 labs/pmpp.py run 4.2 --backend emu`, then `python3 labs/pmpp.py run 4.2`.
- Measure: `python3 labs/pmpp.py run 4.2 --bench` (real devices only).

The tests run eight patterns of A and B (all A, all B, first half of the array,
alternating warps, half of each warp's lanes, alternating lanes, one lane per warp, a
seeded random half) at block sizes of 1, 2 and 8 widths, with array lengths that are and
are not multiples of the block size. `selectKernel` must also agree with `branchKernel`.
`loopKernel` is tested with trip counts of 0, a sorted ramp, the same ramp shuffled, and
a constant. A passing test says the values are right. It cannot tell whether your
`selectKernel` really avoids the branch: only the benchmark shows that.

## Where the width comes from

The patterns are built from the width your device reports, never from a literal 32.

- CUDA and HIP: `warpSize` from the device properties. A thread's warp is its linear
  index in the block divided by the width. NVIDIA documents that cut. AMD documents the
  linear order and uses the same lane arithmetic in its examples
  (`notes/today/ch04-facts.md`, A8 and B6).
- Metal: `threadExecutionWidth` of the compiled pipeline. The specification leaves the
  cut of a threadgroup into SIMD-groups "implementation-defined" (C2), so the driver
  does not compute it. It dispatches the provided probe kernel in the same shape, reads
  back each thread's SIMD-group and lane, and builds "one lane per SIMD-group" and the
  other patterns from that. The test output says what the probe found. The probe is a
  different kernel from yours, so this is evidence about your device, not a guarantee
  about your pipeline. The comment in `provided/simd_probe.metal` has the details.
- The emulator does not run the threads of a warp in step and has no cost model. It runs
  every pattern for pretend widths of 4, 32 and 64 to check your kernels' logic, and it
  cannot run the benchmark.

## Before you benchmark, predict

Write both answers down first.

1. With half the lanes of every warp on each path, how many times slower will
   `branchKernel` be than on the sorted pattern with the same amount of work (first half
   of the array A, second half B)?
2. On that same half-and-half pattern, will computing both paths and selecting be faster
   or slower than the branch?

Then two more to reason about before you look. By the book's model (p. 80), how many
passes does a warp make when one lane takes path B and all the others take A? And for
`loopKernel`: the same trip counts, once sorted along the array and once shuffled. Which
is slower, and why does every count replaced by the mean belong in the comparison?

## Reading the output

The benchmark grows `iters` until the all-A case takes long enough to time, then prints
one row per pattern:

- `branch ms`: `branchKernel` on that pattern.
- `sorted, same mix`: `branchKernel` on a pattern with the same number of A and B, all
  the A in front. The amount of arithmetic is the same; only the layout differs.
- `branch / sorted`: the cost of the layout. This is the number your first prediction
  was about.
- `select ms` and `select / branch`: your second prediction.

Three rows follow for `loopKernel`: trip counts sorted, the same counts shuffled, and
every count equal to the mean. The width and the block size used are in the header line
and in every recorded row.

What the vendors document, to set against what you measured
(`notes/today/ch04-facts.md`):

- NVIDIA: a diverged warp "executes each branch path taken, disabling threads that are
  not on that path". No cost figure is given (A9).
- AMD: "the hardware serializes both branches, executing each path with only the
  relevant threads active", and predication (what `selectKernel` does) is one of the
  remedies AMD lists (B7).
- Apple documents neither how the threads of a SIMD-group that take different paths are
  executed nor what it costs (C4). On Metal your table is the evidence, and no document
  says what it should look like.

The numbers are measurements of these three kernels on your machine at one size. They
are not a property of branches in general: a branch whose two sides are a few
instructions long may be compiled into a select, and no vendor documents when (A9, C4).
