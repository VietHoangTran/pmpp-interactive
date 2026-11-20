# Lab 6.5: double buffering, with a race detector watching

Book: section 6.7 (pp. 146-147), applied to the tiled kernel of section 5.4 (Fig. 5.9,
p. 108) that you wrote in lab 5.2.

`work/kernel.cu` and `work/kernel.metal` hold lab 5.2's tiled matrix multiplication with
its second barrier deleted. It compiles, and it is wrong. Make it correct with **one
barrier per phase**, without putting the second barrier back: give each operand two
tiles, let the phases alternate between the two pairs, keep one barrier after the loads
and none after the dot product. Square matrices whose width is a multiple of
`TILE_WIDTH`, as in lab 5.2.

- Edit `work/kernel.cu` (`doubleBufferedMatmulKernel`) and `work/kernel.metal`
  (`double_buffered_matmul`). The launches, `launchDoubleBufferedMatmul` in `kernel.cu`
  and `encodeDoubleBufferedMatmul` in `work/launch.mm`, are lab 5.2's and are correct:
  double buffering changes the kernel, not the launch.
- `provided/` holds lab 5.2's two-barrier kernel for CUDA and Metal, renamed. The
  benchmark times your kernel against it. You do not edit it.
- `TILE_WIDTH` is defined by the build (16 by default; 8 and 32 are also built for the
  benchmark; Metal gets 16 and, where the device takes a 32 x 32 threadgroup, 32 at run
  time). Do not hard-code 16.

## Method

1. **Before you change anything**, run the starter on the race backend:

       python3 labs/pmpp.py run 6.5 --backend race

   The emulated threads run concurrently under ThreadSanitizer. Its report names two lines
   of `work/kernel.cu`: a store into a tile, and the multiply-add that reads the tiles.
   Write down, in your own words, which thread does what, in which order, for that report
   to be possible, and which of section 6.7's two kinds of dependence it is. (`python3
   labs/pmpp.py doctor` says whether your compiler can build the race backend.)
2. `python3 labs/pmpp.py run 6.5 --backend emu`. The starter passes at width 16 and fails
   at 32 and 64, under all three thread orders. Explain the width 16 case before you go on.
3. Fix `work/kernel.cu`. Run `--backend race` and `--backend emu` until both are clean.
4. Make the same change in `work/kernel.metal`, then `python3 labs/pmpp.py run 6.5` on
   every backend you have. On a GPU each case is launched 10 times on the same inputs,
   with the output refilled with NaN before each launch, and a failing case says in how
   many launches it was wrong.
5. Answer TODO 4 in the comment: why can a thread that runs ahead not overwrite a tile its
   neighbours are still reading, not even two phases later?

Lab 5.2's loop writes first and reads second, so the barrier that guards the false
(write-after-read) dependence is its second one. In the book's toy loop on p. 146 the
read comes first, and there it is the first one. The book swaps two buffer pointers at
the end of each iteration (p. 147); choosing the pair from the phase number does the same
job.

## What the tests catch

Each of these was checked to fail on its own, on the emulator (wrong values under every
thread order), on the race backend (a ThreadSanitizer report as well as wrong values) and
on Metal:

- **The starter**: one pair of tiles and no second barrier. A thread that finishes its dot
  product loads the next phase's elements over slots its neighbours have not read yet
  (write-after-read). On the emulator it fails from two phases on; with one phase there
  is no next phase to overwrite anything.
- **Two pairs of tiles, no barrier at all**: a thread reads slots its neighbours have not
  written yet (read-after-write). Fails even with one phase.
- **The barrier after the dot product instead of after the loads**: the same
  read-after-write. Nothing makes a thread wait for its neighbours' loads before it
  reads.
- **The dot product reading pair `(ph + 1) & 1`**: it reads the pair the other phases
  load, not the one this phase loaded.

On a GPU some of these pass some launches. The emulators do not let them.

## A clean run is evidence, not proof

The race backend examines only the schedules that happened, and ThreadSanitizer reports
each pair of source lines once per run. The emulator's three thread orders are three
schedules, chosen to be unkind. Ten launches on a GPU are ten schedules. A kernel that
passes all of them has not been shown correct. The argument in TODO 4 is what shows it.

## Before you benchmark

    python3 labs/pmpp.py run 6.5 --bench

times the provided two-barrier kernel and yours on the same matrices, alternating, at each
tile width your backend builds (CUDA and HIP: 8, 16 and 32, as separate programs; Metal:
16, and 32 where the device takes it). The width grows from 512 until the two-barrier
kernel takes more than 50 ms, up to 4096. Real devices only.

Write down a prediction first: **at T = 16, what speed-up do you expect from one barrier
per phase instead of two?** Give a number, and the reason. Two things to weigh: how many
barriers a phase has against the T multiply-adds each thread does in it, and what the
change spent (twice the shared memory per block, which chapter 5 showed can cost
occupancy).

Then compare with real code. MLX's production Metal GEMM keeps two barriers per K step,
the book's "before" loop (`notes/today/ch06-facts.md`, D3). Whatever your measurement says,
it is a measurement of your device and this kernel, not a rule.

## Beyond the book

Not in the book; from `notes/today/ch06-facts.md`.

- The two-buffer, one-barrier loop needs only the block barrier, so the same `.cu` file
  compiles for HIP (B8). On Metal it is `threadgroup_barrier(mem_flags::mem_threadgroup)`
  once per phase (C7).
- Two pairs of 32 x 32 `float` tiles are 16 KB, twice lab 5.2's. The driver compares what
  the kernel needs with what the device reports (`sharedMemPerBlock` on CUDA and HIP, the
  pipeline's threadgroup memory against `maxThreadgroupMemoryLength` on Metal) instead of
  assuming a limit.
- NVIDIA documents double buffering today mainly together with asynchronous copies: the
  next tile is fetched into one buffer while the threads compute on the other (A16). That
  is where chapter 15 takes the idea. AMD documents an asynchronous copy into LDS only for
  CDNA 3 and 4, through a compiler intrinsic (B8). Metal documents no asynchronous copy
  into threadgroup memory at all (C7), so on Metal the synchronous form in this lab is the
  one you have.
