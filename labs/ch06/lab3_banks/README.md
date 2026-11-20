# Lab 6.3: provoke a bank conflict, then pad it away

Book: section 6.4 (pp. 139-141) and Exercise 5 (p. 155); section 6.1 (pp. 129-130,
Fig. 6.4) for turning a tile in shared memory.

Section 6.4's example is the corner-turning store: the threads of a warp write down a
column of a 32 x 32 shared tile, every one of them lands in bank 0, and the hardware
serves them one after another. One unused column of padding moves each row by one bank
and the conflict is gone (p. 140). The book shows the two declarations and nothing else.
Here you write a kernel whose whole job is that turn, a matrix transpose, and measure what
the padding buys on your device.

**Your part:** `tiledTransposeKernel(in, out, rows, cols, pitch)` and
`launchTiledTranspose`. `in` is rows x cols and `out` is cols x rows, both row-major, for
any sizes.

- One `TILE_WIDTH x TILE_WIDTH` block per tile. The shared tile is declared for you, with
  room for `TILE_WIDTH + 1` words per row, and element (r, c) of the tile is
  `tile[r * pitch + c]`.
- Read the input tile row by row, so that neighbouring threads read neighbouring
  addresses (coalesced, section 6.1). One barrier. Write the output tile row by row, which
  means each thread takes its value from the shared tile down a column. That column read
  is where the conflict is when the pitch is a multiple of the bank count.
- Check bounds for partial tiles, and make sure every thread reaches the barrier.
- `pitch` is a run-time argument. The driver calls your kernel with `pitch = TILE_WIDTH`
  (unpadded) and `pitch = TILE_WIDTH + 1` (padded), so one kernel serves both layouts.

## Files

- Edit `work/kernel.cu` (CUDA, HIP, emulator): `tiledTransposeKernel` and
  `launchTiledTranspose`. `TILE_WIDTH` is defined by the build (16; 32 is also built for
  the benchmark), so do not hard-code 16.
- Edit `work/kernel.metal` (`tiled_transpose`) and `work/launch.mm`
  (`encodeTiledTranspose`). Metal gets the tile width at run time: 16, and 32 where the
  pipeline accepts 1024 threads per threadgroup.
- Provided, not edited: the two instruments, `provided/bank_probe.cu`,
  `provided/bank_probe.metal`, `provided/copy.cu`, `provided/copy.metal` and
  `provided/bank_probe_launch.mm`.

## Commands

- Test: `python3 labs/pmpp.py run 6.3 --backend emu`, then `python3 labs/pmpp.py run 6.3`.
- Race check: `python3 labs/pmpp.py run 6.3 --backend race` runs the emulated threads
  concurrently under ThreadSanitizer (tile 16 only), which names the two lines of a
  missing or misplaced barrier. A clean run is evidence, not proof.
- Measure: `python3 labs/pmpp.py run 6.3 --bench` (real devices only).

## What the tests do

Your transpose runs on 1 x 1, 7 x 5, 16 x 16, 17 x 33, 64 x 48, 100 x 1 and 1 x 100, and
on a GPU also on 1000 x 1000 and 513 x 2049, each at both pitches. `out` starts as NaN, so
an element you never write is reported, and the comparison is exact: a transpose moves
values and computes nothing. On the emulator a missing barrier fails in at least one of
its three thread orders, and AddressSanitizer stops at an access outside `in` or `out`
and names the line.

The tests cannot see which pitch you used: the output is the same either way. Only the
benchmark can, and only on a device where the layout matters.

Two short checks of the instruments run first: the copy, and the probe at stride 1, whose
output value is known.

## The instruments

- **Bank probe.** One block of exactly one warp: `warpSize` lanes on CUDA and HIP, the
  `threadExecutionWidth` of the probe's own pipeline on Metal. Lane `l` reads and writes
  word `(l * stride) & 1023` of a 1024-float shared array, Exercise 5's array and index,
  many times over, each step depending on the one before. The benchmark runs strides 1,
  2, 4, 7, 8, 12, 16, 24, 31, 32 and 33 and reports the time per warp-wide access (one
  read or one write) and the ratio to stride 1. What it times is a chain of dependent
  accesses by one warp, so the latency of each access and the loop around it are in every
  number. The comment in `provided/bank_probe.cu` says how it keeps the compiler from
  removing the accesses.
- **Copy.** Reads and writes the same bytes as the transpose with no shared memory and no
  turn. A transpose that runs about as fast has little left to gain from its layout.

The emulator runs the probe only to check its output. It models no banks and no time.

## First, Exercise 5 by hand

For strides 32, 31, 24, 16, 12, 8 and 7, work out which banks the 32 lanes of a warp hit
and how many lanes share each bank, with the book's layout of 32 banks of 4 bytes
(p. 139). Then find the rule that gives the answer for any stride without listing the
lanes. The probe's rows are in the same order, so you can set your answers beside them.

## Before you benchmark, predict

Write both down first.

1. How many times longer the probe takes per access at stride 32 than at stride 1.
   NVIDIA's model says 32 passes against 1, but the probe also times latency and loop
   overhead, so decide how much of that 32 should show. AMD and Apple need not follow
   NVIDIA's model at all (below).
2. The ratio of transpose time at pitch T to pitch T + 1, for T = 16 and for T = 32. For
   T = 16, first work out the conflict of the column read at pitch 16 and at pitch 17:
   a warp of 32 threads covers two rows of a 16-wide block.

## What the vendors document

From `notes/today/ch06-facts.md` and `notes/today/infra-facts.md`, to set against what you
measured:

- NVIDIA: 32 banks, each 4 bytes wide. A request with conflicts is split into as many
  conflict-free requests as needed. Threads that read the same 32-bit word do not
  conflict. The padded tile is NVIDIA's own worked example (A6, A12, infra A10). Nsight
  Compute counts conflicts in `l1tex__data_bank_conflicts_pipe_lsu.sum` (A12).
- AMD: 32 banks (CDNA 1 to 3) or 64 (CDNA 4, RDNA 2 to 4), 4 bytes each, serving 16 lanes
  per cycle, so a 64-wide wavefront's access takes several cycles with no conflict at all.
  AMD's guide uses the same 32 x 33 padding (B4, infra B10). With 64 lanes at stride 32,
  lanes 32 to 63 wrap round onto the words of lanes 0 to 31 (our arithmetic), and AMD
  says a broadcast from one address to several threads is handled without conflicts:
  read that row with care.
- Apple: no bank structure is documented for threadgroup memory (C2, infra C13). Apple's
  only advice on access order is to have neighbouring threads of a quad group touch
  neighbouring elements, and to avoid many threads reading the same location. On Metal
  the probe is how you find out.

## Reading the output

- `SIMD width`: the warp width the probe used, and where it came from.
- Bank probe: one row per stride with the time, the nanoseconds per warp-wide access, and
  the time relative to stride 1. The step count grows until stride 1 takes over 10 ms.
- Copy: the matrix side grows from 1024 until the copy takes over 5 ms, up to 8192,
  keeping both matrices within a quarter of the memory your device reports.
- Transpose: one row per tile width and pitch, with GB/s (8 bytes per element: one read,
  one write) and `vs copy`, the copy's time divided by the transpose's.
- On CUDA and HIP, `TILE_WIDTH` 32 is a separate build that `--bench` also runs. The
  probe and the copy are measured once, by the `TILE_WIDTH` 16 build.

The numbers are measurements of your machine.

## Go further

The fix scales up in real kernels. llama.cpp's quantised matrix multiplication for CUDA
pads each block in shared memory by 16 bytes "to avoid shared memory bank conflicts" and
checks the padding with a `static_assert` (ch06 facts D1). MLX pads its Metal GEMM tiles
by 16 bytes and lays the tile out by the operand's storage order, which is corner turning
plus padding (D3). Why would a kernel whose threads move 16 bytes at a time pad by 16
bytes rather than by one word?
