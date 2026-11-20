# Lab 7.2: tiled convolution with halos

Book: sections 7.5 and 7.6 (pp. 172-179), Figs. 7.12 and 7.15.

A 2D convolution of a `width x height` image of floats, row-major, with a
`(2r+1) x (2r+1)` filter applied without flipping, ghost cells counted as 0. Two tiled
kernels, both reading the filter from constant memory:

1. **Part A, write it: the halo kernel** (Fig. 7.12). The block is the input tile:
   `IN_TILE_DIM x IN_TILE_DIM` threads, each loading one element, ghost cells stored as
   zeros, then one barrier. In the compute phase the outer `FILTER_RADIUS` ring of
   threads sits idle, and the others each produce one element of an
   `OUT_TILE_DIM x OUT_TILE_DIM` output tile, where `OUT_TILE_DIM = IN_TILE_DIM - 2r`.
   Write `convTiledHaloKernel` and `launchHalo` (its block and grid).
2. **Part B, repair it: the cached-halo kernel** (Fig. 7.15). Shared memory holds only
   the tile itself, the same size as the output tile; a tap that falls outside it is
   read from global memory, where the book expects neighbouring blocks to have left it
   in the L2 cache. `convCachedKernel` compiles and has **two defects**. One of them is
   printed in the book.

The radius is a template parameter, `template <int FILTER_RADIUS>`, so the filter loops
have fixed trip counts. A provided switch picks the compiled kernel for the radius the
driver asks for (1 to 7), the way llama.cpp's `ssm_conv` picks its filter width
(`notes/today/ch07-facts.md` D3). `IN_TILE_DIM` and `TILE_DIM` are the build's
`TILE_WIDTH`.

- Edit `work/kernel.cu` (part A: `convTiledHaloKernel`, `launchHalo`; part B:
  `convCachedKernel`), `work/kernel.metal` (`conv_tiled_halo`, `conv_tiled_cached`, the
  same two defects) and `work/launch.mm` (`encodeConvTiledHalo`). The parts start at the
  lines `// @part halo` and `// @part cached`. Do not edit `main.cpp` or
  `../shared/`, or the provided functions after `// @part provided`.
- `TILE_WIDTH` is defined by the build: 32 by default, and 8 and 16 are also built for
  the benchmark. Metal compiles 8, 16 and 32 at run time, passing `TILE_WIDTH` and
  `FILTER_RADIUS` as macros (a function constant cannot size a threadgroup array, C3); a
  tile is tested only where its pipeline's `maxTotalThreadsPerThreadgroup` allows
  `tile x tile` threads, and otherwise the driver says what it skipped and why. On HIP a
  32 x 32 block can be refused (B6); the test then says so. Do not hard-code 32.
- Test: `python3 labs/pmpp.py run 7.2`. Start with `--backend emu`.
- Measure: `python3 labs/pmpp.py run 7.2 --bench` (real devices only).

## What the tests check

Both kernels against a CPU reference (zero ghost cells, no flip, accumulated in double),
with a tolerance scaled by the sum of `|f * x|` for each output, because Metal does not
default to strict IEEE arithmetic. `P` is filled with NaN before every launch, so an
output nobody writes fails, and on a real device a guard region after `P` must come
back untouched.

- The provided `const` kernel the benchmark compares against, on 33x17, radii 1 to 7
  (1 to 3 on the emulator). These pass from the start.
- The book's 7 x 7 example with its 5 x 5 filter (Fig. 7.4, r = 2): the whole output,
  and `P[2][2] = 321` and `P[1][0] = 112` (pp. 163-164).
- Random data with random (so asymmetric) filters, for every radius from 1 to
  `min(7, (tile - 2) / 2)`, which keeps the halo kernel's output tile at 2 or more:
  1x1, 7x7, 30x30, 33x17, 100x61, 257x129 and 1000x999 (width x height). On the
  emulator, radii 1 to 3 at 1x1, 7x7, 13x9, 33x17 and 40x31. Most of these widths and
  heights are not multiples of the tile, so tiles overhang the right and bottom edges.

## Part A: things to get right

- `threadIdx` and `blockIdx` are unsigned, and so are Metal's position attributes.
  `threadIdx.x - FILTER_RADIUS` is an unsigned expression: for thread 0 it is a huge
  number, not -2. Copy the indices to `int` before you subtract anything.
- **Every thread reaches the barrier.** The ring threads compute nothing, but they load
  halo elements the others read. Put the compute guard after the barrier, and never
  `return` before it. NVIDIA and Apple say a barrier waits only for threads that have
  not exited, and the emulator models that, so an early return may not hang anything;
  the damage it does is a halo element nobody loaded (`notes/today/ch04-facts.md` A6
  and C3).
- **Store the ghost zeros.** The compute loop has no bounds test: that is the point of
  materialising ghost cells. Shared memory you did not write holds whatever was there.
  The emulator fills it with `0xCD` bytes for every block, so it fails every time; a GPU
  makes no promise either way, and its uninitialised shared memory may happen to read as
  zero, so on a device this mistake can pass. Run on `emu`.
- Without the barrier, the emulator fails under all three thread orders: in order,
  thread `(r, r)` runs its whole compute loop before the threads after it have loaded
  anything.

## Part B: a repair

Method, as in lab 5.4:

1. `python3 labs/pmpp.py run 7.2 --backend emu` first. AddressSanitizer stops the run at
   the first out-of-bounds access and names the line, and the tests after that stop do
   not run. Part A's tests come first; they fail until you have written part A, which
   does not stop the run.
2. Fix one thing, run again. The failures change as you go.
3. Then run on your GPU, and note which defect the GPU showed you and how.

Each defect fails on its own. One gives wrong outputs almost everywhere. The other gives
wrong outputs only near the right edge of the image, in tiles that overhang it (so not
at tile 8 on the 1000-wide image, which is 125 tiles across), and on the last row it
reads past the end of `N`.

**A third thing to notice, which no test can catch.** The in-tile test in the starter is
the book's: `threadIdx.x - FILTER_RADIUS + fCol >= 0 && ... < TILE_DIM`. The left side
is unsigned, so `>= 0` is always true. The kernel is still correct, by accident: a tap
left of the tile wraps to a value above four billion, and the `< TILE_DIM` half rejects
it (`notes/today/ch07-facts.md` A8, checked by compiling it). Fix it anyway, by testing
`int` copies of the indices, because the next edit that drops the upper bound, or tests
a different upper bound, breaks it silently. Clang reports the always-true comparison
under `-Wtautological-unsigned-zero-compare`; the default build does not.

## Where this lab differs from the page

- **Fig. 7.15, line 21.** The shared-memory read omits the `- FILTER_RADIUS` that the
  test just before it uses (`notes/ch07.md`, Unclear in the PDF).
- **The `>= 0` tests on unsigned values** in Fig. 7.15, above.
- **`F_c` and `F`.** Figs. 7.12 and 7.15 declare `F_c` and read `F`. Here it is `F_c`
  throughout, flat (15 x 15 floats for the largest radius), indexed
  `fRow * (2*FILTER_RADIUS+1) + fCol`. `N_s` has its element type, literals are `0.0f`,
  and `N` is `const`.
- **The radius** is a template parameter chosen by a switch, not one `#define`.
- **Launch code.** The book shows no host code for these kernels; `launchHalo` and the
  provided `launchCached` are ours. The block of Fig. 7.12 is `IN_TILE_DIM` square and
  the grid has one block per output tile, which is what the book's indexing implies.
- **The constant symbol lives in `kernel.cu`**, beside the kernels and the
  `cudaMemcpyToSymbol` call: an `extern __constant__` declaration in another file needs
  relocatable device code on CUDA and HIP (A1, B3).

## The experiment

Do this once all tests pass. Before you run `--bench`, write down two predictions:

1. Section 7.5's formula (p. 176) with a 32 x 32 input tile and a 5 x 5 filter
   (`t = 28`, `m = 5`) gives 5.42 FLOP/B, against 0.5 FLOP/B for a constant-memory
   kernel that reads every input from global memory (p. 171). So the tiled kernel asks
   for about 5.42 / 0.5 = **10.8 times less global traffic**. How many times faster than
   the constant-memory kernel will it run on your device?
2. At tile 32 and r = 2, will the cached kernel be faster than the halo kernel? Write
   down the ratio you expect: `tiled_cached` GFLOP/s divided by `tiled_halo` GFLOP/s.

The 10.8 is about requested bytes. Caches already serve many of the constant-memory
kernel's neighbour reads (p. 168), and a kernel that asks for little memory traffic is
then limited by something else, so the speed-up you measure can be far from 10.8 in
either direction. Section 7.6's kernel bets that the halo reads hit a cache; whether
they do on your GPU is the measurement (`notes/today/ch07-facts.md` C4 for Apple, B5 for
AMD).

What `--bench` does: it doubles the side of a square image from 2048 until the provided
constant-memory kernel at r = 2 takes over 50 ms, up to 8192 x 8192, keeping `N` and `P`
within an eighth of the memory the device reports (and 2 GiB). For each tile width (on
CUDA and HIP, the build's: `pmpp.py run 7.2 --bench` runs the t8, t16 and default
builds; on Metal 8, 16 and 32) and r in {1, 2, 3, 5, 7} where the halo kernel's output
tile is at least 2, it checks each kernel on a 100x61 image and then times it: the best
of at least 5 runs, repeated until the runs add up to about a quarter of a second. A
kernel whose check fails is reported, not timed.

- `const` is the provided baseline (`../shared/conv_const.*`): book Fig. 7.9, one
  thread per output, the filter in constant memory, the same compile-time radius as your
  kernels, and 16 x 16 blocks in every build.
- `out` is the output tile: `tile - 2r` for `tiled_halo`, `tile` for `tiled_cached`.
- GFLOP/s counts `2 * m^2` per output, ghost taps included (the book's count, p. 167).
- `x const` is the constant-memory kernel's time divided by this kernel's.
- `model` is section 7.5's FLOP/B for the halo kernel at `t = out`, and `ratio` is that
  divided by 0.5: the formula's factor of requested bytes, which section 7.5 plots against
  the measured `x const`.
- On CUDA and HIP each timed run includes the provided `cudaMemcpyToSymbol` of the
  filter (at most 900 bytes), for all three kernels alike.

Every number it prints is a measurement of your device. Nothing here says what it should
be.
