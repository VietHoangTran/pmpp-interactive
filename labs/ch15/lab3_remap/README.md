# Lab 15.3: coalesced stores, four 4 x 4 sub-tiles and float4 stores

Book: section 15.5 (pp. 360-361), Fig. 15.10; Exercises 3 and 2 (p. 370). The book gives
no code for this section. Milestone M3, stage 3 of 5.

In labs 15.1 and 15.2 each thread stores an 8 x 8 patch, so in one store instruction
neighbouring threads of a warp write addresses 8 floats apart. `work/` holds lab 15.2's
solution, renamed and made a template over the warp width `W`. Re-map the threads as
section 15.5 describes, and store each row of a sub-tile with one `float4` store.

The layout, for a warp of W threads (the book's numbers are W = 32):

- a warp's W lanes form 8 lane-rows of `LC = W / 8` lanes;
- each warp's tile is split into 2 x 2 quadrants of 32 rows by `4 * LC` columns, and a
  lane owns the 4 x 4 sub-tile at the same place in all four quadrants: its logical 8 x 8
  tile;
- a warp's tile is therefore 64 x `8 * LC`, and `128 / (8 * LC)` warps sit across the
  block tile.

For W = 32 that is the book's 8 warps as 2 x 4, warp tiles of 64 x 32, quadrants of
32 x 16, lanes as 8 x 4. For W = 64 (AMD CDNA) the same rules give 4 warps as 2 x 2, warp
tiles of 64 x 64 and quadrants of 32 x 32 (`notes/today/ch15-facts.md` B1). Work out
lane 5 of warp 6 by hand for both widths before you write code.

## Files

- `work/kernel.cu`: the mapping (TODO 1), `mm` over the four sub-tiles (TODO 2),
  `writeTile` with `float4` stores (TODO 3), and `launchGemmRemap` (TODO 4), which reads
  the device's warp width once and calls `launchGemmRemapWidth`.
  `launchGemmRemapWidth`, which launches the instance for a given width, is written for
  you: the tests call it with both widths, and the benchmark's occupancy query refers to
  both instances, so the starter must create them.
- `work/kernel.metal` (`gemm_remap`) and `work/launch.mm` (`encodeGemmRemap`): the same
  in Metal. The host compiles the kernel with `SIMD_WIDTH` defined, and you bind C a
  second time as `device float4*` for the vector stores, as in lab 6.2.

Never write the warp width as a number inside the kernel. On HIP it differs between
devices and "can not be assumed to be a compile-time constant on the host"; AMD's
recommended route is the one used here, a host-chosen template instance (B1).

## Commands

    python3 labs/pmpp.py run 15.3 --backend emu
    python3 labs/pmpp.py run 15.3
    python3 labs/pmpp.py run 15.3 --bench

## What the tests check

Lab 15.1's tests, each run with the layout for W = 32 and for W = 64 on every backend,
the emulators included: the layout is index arithmetic, so whether it is right does not
depend on the hardware's width. On CUDA and HIP the launch that reads the device's width
runs as well. On Metal the driver builds `SIMD_WIDTH` 32 and 64 for the tests, and for
the benchmark the width the pipeline reports. The untouched starter builds and fails:
`writeTile` stores nothing, so C stays NaN.

Shapes with N = 1, 3, 7, 9, 131 and 511 need the scalar fallback. One thing the emulator cannot
see: a `float4` store to an address that is not 16-byte aligned runs fine on a CPU, so
dropping the alignment condition still passes on `emu`. On Metal the same mistake writes
to the wrong elements (the `float4` view's index rounds down) and fails. For the book's
pointer cast on CUDA, NVIDIA's developer blog requires a 16-byte-aligned address
(`notes/today/ch06-facts.md` A11), and `cudaMalloc` returns at least 256-byte-aligned
memory (A3), so N a multiple of 4 is enough. On HIP, C comes from `hipMalloc`, for which
AMD's memory reference states no alignment (ch15-facts B8): that the same rule is enough
there is unverified.

## Before you benchmark

Predict **the speed-up over lab 15.2 on the short-K shape 4096 x 128 x 4096, and at
2048 x 2048 x 2048.** Weigh:

- The stores happen once per output element, however long K is. With K = 128 there are
  16 tile pairs to load and compute for every tile stored; with K = 2048 there are 256. So
  a better store pattern matters most when K is short.
- The remap fixes the stores and removes the B strip's 4-way bank conflict, but it raises
  the A strip's conflict from 2-way to 8-way on 32 banks: eight lane-rows read rows of
  `A_s` 4 apart, 32 words apart (E1). Lab 15.4 adds the padding that removes it.
- On Metal, Apple documents no coalescing rule: nothing documented predicts the result.

## Beyond the book

- Not every library solves section 15.5 this way. CUTLASS keeps the thread layout that is
  best for the arithmetic and stages the output tile through shared memory in a separate
  epilogue, then stores it with striped, coalesced accesses (A12); llama.cpp's Metal
  kernel does the same for edge tiles (D2).
- Metal: the `float4` store goes through a second binding of C declared
  `device float4*`, Apple's documented vector form, instead of the book's pointer cast
  (`notes/today/ch06-facts.md` C3).
