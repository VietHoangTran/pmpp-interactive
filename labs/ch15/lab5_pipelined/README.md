# Lab 15.5: double buffering and software pipelining

Book: section 15.8 (pp. 364-368), Figs. 15.12 to 15.14. Milestone M3, stage 5 of 5: the
last stage in the book's layout, so this lab's `stage` row is the top rung of the ladder
that began with lab 5.3. Lab 15.6 is the milestone itself.

`work/` holds lab 15.4's solution, renamed (`gemmPipelinedKernel`, `launchGemmPipelined`,
`launchGemmPipelinedWidth`, `gemm_pipelined`, `encodeGemmPipelined`), with its tile loop
taken out. Put it back so that iteration `tile` computes with pair `tile` while it loads
pair `tile + 1` into a second set of buffers, with one barrier per iteration instead of
two:

- two pairs of shared tiles, `A_s[2][...]` and `B_s[2][...]` (TODO 1);
- a prologue that loads pair 0 and waits; a loop over `tile < numTiles - 1` that runs `mm`
  on pair `tile`, then loads pair `tile + 1` into the other buffers, then waits; an
  epilogue that runs `mm` on the last pair (TODO 2).

Fig. 15.14 is a skeleton: its arguments are elided, its buffers undeclared, and its
pointer types do not match the one-dimensional tiles of Fig. 15.3 (`notes/ch15.md`,
"Unclear" 8). The book swaps a current and a next pointer at the end of each iteration;
choosing a pair's buffers by its parity, `tile & 1`, as lab 6.5 does, is the same
alternation. Two cases the figure leaves to you: what `numTiles - 1` does when K is 0
(an unsigned count wraps around), and which buffers the epilogue reads.

## Files

- `work/kernel.cu` and `work/kernel.metal`: the double buffers and the tile loop.
  `work/launch.mm` is complete: pipelining changes the kernel, not the launch.

## Method

1. Before you write anything, answer from Fig. 15.12: which of lab 15.4's two barriers
   guards a true dependence and which a false one? Why can a thread that runs ahead
   never overwrite a pair that its neighbours are still reading, not even two
   iterations later?
2. `python3 labs/pmpp.py run 15.5 --backend emu`, then `--backend race`, then the rest.

## What the tests check

Lab 15.4's tests, at both widths. On a GPU every case is launched 10 times with C refilled
with NaN before each launch, and a failure says in how many launches the result was
wrong. `(128, 8, 128)` is a single tile pair (the loop runs zero times) and
`(128, 24, 128)` three pairs. `(5, 0, 9)` has no tile pair at all: C must come out all
zeros. It runs under a 30-second watchdog, so the unsigned wrap-around fails there with
that reason instead of running for hours. Both forms of the mistake were checked to fail
on their own: an unsigned `numTiles` with no guard fails on `emu` and Metal through the
watchdog; a signed one with no guard, whose epilogue then reads the buffers nobody
loaded, fails on `emu` (garbage in C) and passed on Metal, where those buffers happened
to hold zeros. The race backend runs one 256-thread block at a time under
ThreadSanitizer on `(128, 24, 128)` and `(130, 9, 131)`.

The barrier inside the loop was checked to fail on its own when removed:

- on the emulator, every case with more than one tile pair fails, under all three thread
  orders; the single-pair cases pass, because their loop never runs;
- on the race backend, ThreadSanitizer reports the race between the store in `loadTile`
  and the read in `mm`, with both source lines;
- on Metal, the only GPU it was checked on, some cases failed in only a few of the 10
  launches, and some passed all 10.

The untouched starter builds and fails: without the loop, C comes out 0.

## Before you benchmark

Predict **the speed-up over lab 15.4 at 2048 x 2048 x 2048.** First read lab 15.4's
`occupancy` row (CUDA, HIP) or `pipeline` row (Metal). If two or more blocks fit on an SM,
other blocks were already overlapping their loads with your block's arithmetic, and the
gain from overlapping within a block shrinks (p. 364). Weigh too:

- The overlap is the compiler's to make (p. 367). The source only puts the loads of pair
  `tile + 1` and the arithmetic of pair `tile` in the same iteration with no barrier
  between them; nothing makes the compiler interleave them.
- Shared memory per block doubles, to 17,408 bytes with the padding
  (`notes/today/ch15-facts.md` E1), which can cost occupancy where shared memory is what
  limits it.

Then try the other order inside the loop: load pair `tile + 1` first, then run `mm` on
pair `tile`. It is just as correct (the tests will tell you) and it changes what the
compiler is handed. Measure both orders on your device. Whatever you find is a
measurement of your device and your compiler, not a rule.

## Beyond the book

Not in the book; from `notes/today/ch15-facts.md`.

- CUDA (compute capability 8.0 and later) can copy from global to shared memory without
  passing through registers (LDGSTS, `cp.async`), and Hopper adds TMA for whole tiles.
  `cuda::memcpy_async` does not guarantee LDGSTS, and the book's claim that nvcc fuses a
  vector load and store into one by itself could not be verified (A7, A8). The loop here
  is the synchronous form of the same idea.
- HIP documents no usable asynchronous copy into LDS for this; the software form of this
  lab is the one to use there (B5). Metal documents none at all (`ch06-facts.md` C7).
- CUTLASS double-buffers at two levels: the shared tiles, as here, and the register
  strips within each thread (A12).
- Metal: Xcode's pipeline statistics report how long threads wait at barriers
  ("Synchronization (barrier)", C3), the place to look for what one barrier fewer buys.
