# Lab 4.3: barrier hunt

Book: section 4.3 (the rules for `__syncthreads()`, p. 72), section 4.4 (p. 74: within a
block, assume the threads run in any order) and exercise 5 (p. 91). The kernel is ours,
not a listing from the book.

`work/kernel.cu` and `work/kernel.metal` hold a kernel that reverses each block's slice of
an array in place. Block b owns elements `b*blockDim.x` up to `(b+1)*blockDim.x`, and the
last slice stops where the array stops. Every thread reads its element into a register,
then writes it to the mirrored position of its slice. It compiles, and it has **three
independent defects**. Find and fix all three.

- Edit `work/kernel.cu` (`reverseBlocksKernel`; the launch `launchReverseBlocks(a, n,
  threadsPerBlock)` below it is correct) and `work/kernel.metal` (`reverse_blocks`).
  `work/launch.mm` is correct too. The same three defects are in both kernels.
- Test: `python3 labs/pmpp.py run 4.3 --backend emu`, then `python3 labs/pmpp.py run 4.3`.

The driver chooses the block size: 32, 48, 64 and 96 threads, then one, two and four times
the width your device reports (`warpSize` on CUDA and HIP, the pipeline's
`threadExecutionWidth` on Metal). Each block size is tested with array lengths that are
multiples of it and with lengths that are not.

## Method

1. Emulator first: `python3 labs/pmpp.py run 4.3 --backend emu`. It runs one thread until
   that thread reaches a barrier or returns, then the next, under three fixed orders. A
   kernel that needs a barrier and has none is wrong there every time.
2. Fix one thing, run again. The failures change as you go. Use what the passing and
   failing cases have in common: which block sizes, and whether the length is a multiple.
3. Know what each kind of failure looks like on the emulator:
   - **Wrong values.** The message names the first wrong element, its block and its thread.
   - **A write outside the array.** The emulator's buffer is exactly the array, so
     AddressSanitizer stops the run at the first one. Read three things near the top of
     its report: `WRITE`, the first frame (`kernel.cu:` and a line number), and the
     sentence saying how many bytes before or after the region the address is. The tests
     after the stop are not run.
4. `python3 labs/pmpp.py run 4.3 --backend race` runs the emulated threads concurrently
   under ThreadSanitizer. A read and a write of the same element with no barrier between
   them are reported with both source lines (`python3 labs/pmpp.py doctor` says whether
   your compiler can build the `race` backend).
5. Then your GPU: `python3 labs/pmpp.py run 4.3`. There the array sits between sentinel
   elements, each shape is launched 40 times on fresh data, and a failing case says in how
   many of the 40 it was wrong.

The Metal barrier takes a flag that names the memory whose accesses it orders. Chapter 5's
kernels pass `mem_flags::mem_threadgroup`. Decide whether that is the right flag here.

## The experiment (exercise 5)

Exercise 5 puts a claim to you: with 32 threads in a block, `__syncthreads()` can be left
out. Test it on your own device once all tests pass.

Before you run it, write down two predictions. With the barrier removed and a block
exactly one warp wide (the case marked `1 x width`), will your GPU's tests fail? And with
a block four warps wide (`4 x width`)?

1. Delete the barrier from your fixed `work/kernel.cu` (or `work/kernel.metal`).
2. `python3 labs/pmpp.py run 4.3 --backend emu`: every case fails, under every order.
3. `python3 labs/pmpp.py run 4.3` on your GPU, five times. For each block size, note in
   how many of the five runs a case failed, and in how many of its 40 launches.
4. Put the barrier back.

Whatever your table shows, read it against what the vendors document, because that is
what the next device is entitled to do:

- The width is not a constant. NVIDIA calls `warpSize` "commonly 32"; under HIP it is 64
  on AMD's gfx9 devices and 32 on gfx10 and above; Apple documents no value at all, only
  where to read it (`notes/today/ch04-facts.md`, A8, B6, C2).
- NVIDIA does not promise that the threads of one warp stay in step: from compute
  capability 7.0 on, each thread has its own program counter, and NVIDIA's advice is to
  synchronise explicitly (A10). AMD does promise lockstep within a warp (B5). Apple says
  nothing either way (C2).
- A block wider than one warp is several warps, and no vendor orders them against each
  other without a barrier.

A case that passed 40 times showed you 40 schedules. It did not show you that the kernel
is correct. The emulator's three orders are schedules too, chosen to be unkind.

Nothing to benchmark here.
