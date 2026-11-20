# Lab 20.3: flash attention race hunt

Book: section 20.5 (pp. 492-504), Figs. 20.9 to 20.16. A bare item such as (A3) is in
`notes/today/ch20-facts.md`.

`work/kernel.cu` and `work/kernel.metal` hold lab 20.2's flash attention kernel (the same
contract, the same tests) with **three independent defects**, the same three in both files.
It compiles. Find and fix all three, and keep a record of which backend showed you which.

You know what kind of defect each one is, not where it is:

1. a missing barrier between the lanes of one warp;
2. a missing barrier across the whole block;
3. a wrong value in the online softmax's bookkeeping: no synchronisation involved.

If you have not done lab 20.2, read its README first: "The design" and "Where this differs
from the page" describe the correct kernel.

## Before you run anything, predict

Copy this table and fill in each cell with "fails", "reported" (ThreadSanitizer names the
lines although the numbers may be right) or "passes". The labs page asks for your answer
for defect 1 as `ch20/labs/q-detector`.

| Defect | `emu` | `race` | your GPU |
|---|---|---|---|
| 1. warp barrier | | | |
| 2. block barrier | | | |
| 3. bookkeeping | | | |

What you need to reason about it:

- `emu` runs one thread at a time. A thread runs until its next synchronisation point
  (`__syncthreads()`, any warp `_sync` call, a shuffle included), then the next thread runs,
  in three fixed orders. A shuffle is a rendezvous of the whole warp: no lane leaves it
  before every lane has arrived.
- `race` runs the threads at once under ThreadSanitizer. As on a GPU, shuffles do not order
  memory there (A3); `__syncwarp()` and `__syncthreads()` do.
- Your GPU runs warps in whatever order and overlap it likes, and the hardware may keep a
  warp's lanes together whether or not the code asks for it (on Apple GPUs, whether a
  SIMD-group runs in lockstep is not documented, C4).

## Method

The same as lab 5.4's:

1. `python3 labs/pmpp.py run 20.3 --backend emu`. Its thread orders are deterministic, so a
   defect it can see fails every time, and AddressSanitizer stops at an out-of-bounds
   access and names the line. It is slow on this kernel: start it and read the code while
   it runs.
2. Fix one thing, run again. The failures change as you go.
3. `python3 labs/pmpp.py run 20.3 --backend race`. ThreadSanitizer names the two source
   lines whose write and read collide, which says where a barrier is missing. A function
   the compiler inlined may show as `flashAttentionKernel` with no line number; the other
   side of the pair still names its line. It sees only collisions that happened in that
   run, and reports each pair of lines once per run, so a later test of the same kernel can
   pass: a clean report is not a proof.
4. Then your GPU: `python3 labs/pmpp.py run 20.3`. Note which defects it showed you and
   which it hid. A GPU that passes is not evidence either.
5. When all three backends pass, compare `work/` with `solution/`, where each fixed line is
   marked.

Useful while hunting: the "late max" data makes the running max rise in every tile, so
anything wrong with a rescale shows there first; the failure message names the row, the
head and the data recipe.

## Files

- `work/kernel.cu` (CUDA; also built for HIP and the two emulators) and
  `work/kernel.metal`: yours to fix.
- `work/launch.mm`: lab 20.2's reference Metal host side (it chooses `B_c` for your device,
  compiles `kernel.metal` with it, checks the pipeline and dispatches). It has no defect.
- `main.cpp`: the driver. The tests, the CPU reference and the tolerances are lab 20.2's,
  in `../shared-g11/flash_test.hpp`: the same shapes on a GPU, small ones on the emulators
  (`emu`: n = 1, 33, 64, 70; `race`: n = 33 and 40).

Nothing to benchmark here: lab 20.2 times the same kernel.
