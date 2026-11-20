# Lab 15.4: padding and vector loads

Book: section 15.6 (pp. 361-363), Fig. 15.11; Exercise 1 (p. 370); the case for vector
loads at low occupancy (section 15.7, p. 364). Milestone M3, stage 4 of 5.

`work/` holds lab 15.3's solution, renamed (`gemmPaddedKernel`, `launchGemmPadded`,
`launchGemmPaddedWidth`, `gemm_padded`, `encodeGemmPadded`), with `A_s` unpadded and the
body of `loadTile` removed. Two changes:

1. **Padding.** Give `A_s` the leading dimension `bK + 1` everywhere it is indexed: its
   declaration, the shared store in `loadTile`, the row stride of the strip reads in `mm`,
   and the thread's start offset in the call to `mm`. The book names the first three and
   not the last (p. 363; `notes/ch15.md`, "Unclear" 4). Leave it out and see what fails.
2. **Vector loads** (Exercise 1). Each tile of this kernel is 1024 floats, 256 `float4`s:
   one `float4` load per thread per tile, when it is safe. The A tile's rows are
   16-byte aligned when K is a multiple of 4, the B tile's when N is (on CUDA, A and B
   come from `cudaMalloc`, aligned to at least 256 bytes; on HIP they come from
   `hipMalloc`, whose alignment no facts file documents, so the same rule there is
   unverified), and the four columns must lie inside the matrix. Otherwise keep the scalar path that fills zeros past the matrix. Store into
   shared memory one float at a time: a padded row of `A_s` is 9 floats, so a `float4`
   store there would not be aligned, and NVIDIA documents no bank rule for 128-bit shared
   accesses (`notes/today/ch15-facts.md` A2).

The book writes the declaration `A_s[bM*(bK + 1)]`. The CPU emulator's translator rejects
parentheses inside a `__shared__` declaration, so give the padded leading dimension a name
(`#define LDA_S (bK + 1)`) and write `A_s[bM * LDA_S]`.

## Files

- `work/kernel.cu`: the padded declaration, every index into `A_s`, and `loadTile`.
- `work/kernel.metal` (`gemm_padded`) and `work/launch.mm` (`encodeGemmPadded`): the same in
  Metal, where the vector loads go through second bindings of A and B declared
  `device const float4*`, as C already is for the stores.

## Commands

    python3 labs/pmpp.py run 15.4 --backend emu
    python3 labs/pmpp.py run 15.4
    python3 labs/pmpp.py run 15.4 --bench

## What the tests check

Lab 15.3's tests, at both widths. Shapes with K or N not a multiple of 4 (1, 3, 7, 9, 129,
131, 511) run the scalar fallback; the others run the vector path. The untouched
starter builds and fails: `loadTile` writes nothing, so `mm` multiplies whatever was in
shared memory.

The missing start offset is caught: with the padded declaration and leading dimension
but `&A_s[tRow * bK]`, every test shape with 5 rows or more fails, on the emulator and on
Metal. The shapes with 1, 3 or 4 rows pass, because all their rows belong to threads
whose sub-tiles start at row 0 of the tile, where the two offsets agree. A test that
only used small matrices would have missed it.

## Before you benchmark

Predict **the speed-up over lab 15.3 at 2048 x 2048 x 2048.** Weigh:

- On 32 banks the A-strip conflict of lab 15.3's layout goes from 8-way to none: rows 4
  apart start 36 words apart, in banks 0, 4, ..., 28 (Fig. 15.11; E1).
- Vector loads cut the load instructions per tile pair from 8 to 2 per thread. The book's
  argument is that a kernel running few threads per SM issues too few memory requests to
  use the bandwidth, and wider requests make up for it (p. 364).
- On AMD devices with 64 banks the conflict figures are our model; AMD does not document
  how a 64-lane access is split (B2).
- On Metal, Apple documents no threadgroup-memory banks at all (C2): this is a measurement
  with no prediction from the vendor.

## Beyond the book

- Which leading dimensions remove the A-strip conflict (our arithmetic, E1): any odd one
  on 32 banks; any not divisible by 4 on 64 banks. MLX pads its Metal tiles by 16 bytes,
  a practice with no Apple documentation behind it (C2, D3).
