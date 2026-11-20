# Lab 20.2: flash attention forward (milestone M6, part two)

Book: section 20.5 (pp. 492-504): the online softmax of Eqs. 20.3 to 20.6, the tiling of
Fig. 20.8 and the kernel of Figs. 20.9 to 20.16; Exercises 1 and 3 (p. 511). The batch and
head dimensions, any `n`, the logsumexp `L`, the skipped tiles and everything on Metal are
not on the page: they are ours, and the reasons are below. A bare item such as (A3) or (C1)
is in `notes/today/ch20-facts.md`.

Write causal attention as one kernel, so that the `n x n` scores of lab 20.1 never exist in
device memory:

```
launchFlashAttention(Q, K, V, O, L, batch, heads, n, scale)
```

| Tensor | Shape |
|---|---|
| `Q`, `K`, `V` (in), `O` (out) | `[batch, heads, n, 128]` |
| `L` (out) | `[batch, heads, n]` |

Row-major `float`, batch first, head dimension 128 (`HEAD_DIM`, the book's d).

## The contract

- `n >= 1`, any value: not only multiples of 32. `batch * heads` blocks along grid y.
- Causal: row `r` sees the columns `c <= r` (square, so the book's mask and lab 20.1's
  lower-right one agree).
- With `s = scale * Q K^T`, `m` the row's largest visible `s` and `D` the sum of
  `exp(s - m)` over the visible columns: `O = (sum of exp(s - m) V) / D` and `L = m + log(D)`,
  the same `L` lab 20.1 writes for the same scores.
- No buffer of `n x n` anything. The driver allocates only Q, K, V, O and L, and checks it.

## The design (the book's)

- **One block of 512 threads per panel of `B_r = 32` rows of Q**: 16 warps, 2 rows each
  (`B_r_warp`). Grid x walks the panels with the book's outer loop (`i` starts at
  `blockIdx.x` and strides by `gridDim.x`, p. 497); grid y walks the (sequence, head) pairs,
  the "additional dimensions of thread blocks" p. 495 mentions.
- **Storage** (p. 497): each thread keeps its slice of the warp's Q rows and O rows, and
  the running max `m_i` and denominator `D_i` of its two rows, in local arrays meant to stay
  in registers. The block shares three tiles: `KT_j` (a tile of K, transposed and padded),
  `S_i` (the scores of the panel against that tile, overwritten in place by P) and `V_j`.
- **Lanes are interleaved** (Fig. 20.10): lane L holds columns L, L + 32, L + 64, L + 96 of
  a Q or O row. To compute a score, every lane needs every element of the Q row, so the
  owner broadcasts each one with a shuffle (Fig. 20.12).
- **Two loops.** Over tiles j: load `KT_j` and `V_j`, barrier, then each warp walks its two
  rows (`ii`): compute the segment of S and its max, update `m` and rescale `D`, turn S into
  P in place and add its sum to `D`, rescale O and add P times `V_j`. Then a second barrier,
  before the next load overwrites the tiles (Fig. 20.9 line 39). After the last tile,
  `store_O` divides by `D` once.
- **`B_c`**, the tile width, is `TILE_WIDTH` from the build: 32 as on the page (the
  default) and 16, also built for the benchmark. Each lane owns at most one column of a
  tile, so with `B_c = 16` lanes 16 to 31 have none: they still take part in every shuffle
  and bring the identity (`-FLT_MAX` for the max, 0 for the sum).

**Provided, already corrected**: the constants, the lane-mask type and the 32-lane logical
warp, `load_Q`, `load_KT_and_V` (zero-filled past `n`, padded through `addr()`) and
`store_O` (bounds test, writes `O = O~ / D` and `L = m + log(D)`).

**Yours**: `initialize` (Exercise 3), `compute_S_and_max`, `update_m_and_D`,
`compute_P_and_update_D`, `compute_O`, the kernel's two loops with both block barriers, and
the launch (Exercise 1). On Metal, also the part of `launch.mm` described under "Metal".

## Where this differs from the page

1. **A warp barrier the page leaves out** (A3). In `compute_P_and_update_D` each lane writes
   one element of its warp's P row; in `compute_O` every lane reads the whole row. The
   shuffles in between do not order memory ("These intrinsics do not imply a memory
   barrier"), so nothing guarantees the reads see the writes. `__syncwarp()` after the P
   write does. llama.cpp has the same barrier at the same point (D1). Lab 20.3 is about it.
2. **No `NEG_INFINITY`** (A5). No CUDA header defines it, and IEEE gives NaN for
   `(-inf) - (-inf)`. Masks are tested by index, so no masked score is ever used; the
   identity of max is `-FLT_MAX`; the running max starts at `-FLT_MAX/2`, as llama.cpp's
   does, so that no subtraction can overflow. Exercise 3's answer for O and D, 0, is
   FlashAttention-2's (A7).
3. **No CUB.** `cub::Max()` and `cub::Sum()` are gone (infra A11), and the page's single
   `WarpReduce::TempStorage` shared by all 16 warps is not CUB's documented use (A2). A
   butterfly of `__shfl_xor_sync` leaves the max or the sum in every lane, so the page's
   broadcast from lane 0 disappears.
4. **A typed mask and an explicit width on every shuffle.** `0xFFFFFFFF` does not compile
   for HIP (B1). On a 64-wide AMD wavefront the book's 32-lane warp is half a wavefront,
   and HIP's shuffle, whose width defaults to 64, would read across the halves; passing
   width 32 keeps each exchange inside its logical warp (B2).
5. **`L` where the page writes `out_D`.** The denominator alone cannot rebuild P without
   the row max, which the page never writes; FlashAttention-2 stores `m + log(D)` (A7).
6. **Any `n`.** The page divides N by `B_r` and `B_c`, so N must be a multiple of both
   (notes, "Unclear in the PDF"). Here the tile counts round up, the loaders fill rows past
   `n` with zeros, the mask also hides columns `>= n`, and `store_O` skips rows `>= n`.
7. **Tiles declared in the kernel** and passed to the helpers. Metal rejects threadgroup
   arrays in helper functions (C2), and AMD documents only kernel scope (B3).
8. **Tiles wholly above the diagonal may be skipped**, as FlashAttention-2 does (A7); the
   page runs every tile for every panel. The reference skips them. Optional: a kernel that
   runs them passes, and only costs time.
9. The page's `d` is `HEAD_DIM` here (a macro named `d` would rewrite every `d` in the
   file), and `exp`, `log` are written `expf`, `logf`.

## Metal

The same structure, with Metal's vocabulary (the table at the top of `kernel.metal`):
`simd_broadcast` for the Q broadcast (the source lane is the same in every lane, as it
requires), `simd_max` and `simd_sum` for the reductions (the result lands in every lane),
`simdgroup_barrier(mem_flags::mem_threadgroup)` for the warp barrier (C4) and
`threadgroup_barrier(mem_flags::mem_threadgroup)` for the block ones.

`launch.mm` is yours too:

- **Choose `B_c` from the device.** The book's tiles at `B_c = 32` take 37,376 bytes;
  Apple documents 32 KB of threadgroup memory (C1), and creating the pipeline fails. Take
  the largest `B_c` in {32, 16, 8} whose tiles fit `maxThreadgroupMemoryLength` (16 gives
  18,688 bytes, F1).
- **Compile with it.** `kernel.metal` reads `TILE_WIDTH`; pass it through
  `MTLCompileOptions.preprocessorMacros` (C6).
- **Check the pipeline.** The lane layout needs SIMD-groups of exactly 32
  (`threadExecutionWidth`; Apple documents no fixed width, C3) and 512 threads per
  threadgroup (`maxTotalThreadsPerThreadgroup`, a property of the pipeline). If either
  fails, return nil with a message that says which.
- **Dispatch** one threadgroup of 512 per panel along x and per (sequence, head) along y.

## Files

- `work/kernel.cu` (CUDA; also built for HIP and the two emulators). The TODOs are in
  order.
- `work/kernel.metal` and `work/launch.mm` (Metal).
- `main.cpp`: the driver. The tests, the CPU reference and the tolerances are in
  `../shared-g11/flash_test.hpp`, shared with lab 20.3.
- Provided for the benchmark, not edited: `../shared-g11/unfused.cmake` copies lab 15.6's
  reference `launchLinear` (M3), lab 20.1's reference `launchCausalSoftmax` and lab 5.3's
  reference `launchGemm` (M2) into the build, renamed. If one of those solutions is
  missing, the unfused rows are absent.

## Commands

- Test: `python3 labs/pmpp.py run 20.2 --backend emu` first (slow: each case takes a while;
  it runs four), then `--backend race` (below), then `python3 labs/pmpp.py run 20.2` on
  your GPU.
- Measure: `python3 labs/pmpp.py run 20.2 --bench`. On CUDA and HIP this also runs the
  `B_c = 16` build.

## What the tests check

- **Shapes**: `n` = 1, 2, 31, 32, 33, 63, 64, 65, 100, 257, 1000, 2048 (multiples and
  non-multiples of `B_r` and `B_c`), each with (batch, heads) = (1, 1) and (2, 3). `emu`
  runs `n` = 1, 33 (two heads), 64 and 70; `race` runs `n` = 33 and 40.
- **Data**, one recipe per head, every recipe in every shape:
  - Q, K, V uniform in [-1, 1];
  - **late max**: the score of column c grows by about 0.04 per column, so the running max
    rises in every tile and every rescale matters. A missing or misplaced rescale fails
    here;
  - every score offset by about +90. `exp(90)` is beyond float's range, so a kernel that
    does not subtract the max overflows (an offset whose exponential still fits in a float
    would not catch it).
  - V with a different level in every column, so that a column mix-up in O shows.
- **Checks**: every row of O and L of every head against a double CPU reference, with the
  tolerances below; O and L start as NaN, so an element nobody writes is caught; 64 canary
  values after O and after L must be untouched; and no device buffer of `n x n` floats.

## Tolerances, and why

The reference (`flash::refRow`) computes each row in double from the same float inputs.
Each row gets its own bound, from its own data. With u = 2^-24 (one rounding) and, for the
row:

- `A`, the largest `scale * sum |q k|` over its visible columns (how large a dot product's
  terms are);
- `X`, the largest `m - s` (how far down the exponents reach);
- `T`, the number of tiles the row spans;

the relative error of the un-normalised output and of `D` is at most about

```
e = 2 (d + 2) u A  +  u X  +  2u (3 + 2X)  +  2u (3T + 2X)  +  (B_c + 2T + 8) u
```

term by term:

- **Scores.** A 128-term dot product accumulated in order, then scaled: at most
  `(d + 1)` roundings of terms no larger than `A` in total, and `s - m` carries the error
  of two scores: `2 (d + 2) u A` absolute in the exponent.
- **`s - m`** is rounded once: `u X`.
- **The exponential.** The largest bound of the three backends is Metal's fast `exp`,
  `3 + floor(|2x|)` ulp (ch10-facts C4); CUDA's `expf` is within 2 ulp (ch10-facts A11).
  An ulp is at most `2u` relative: `2u (3 + 2X)`.
- **The rescale factors** `exp(old max - new max)`, one per tile: their arguments add up
  to at most `X` over the row, and each has the same ulp bound: `2u (3T + 2X)`.
- **The sums.** All terms are positive. `P V` adds `B_c` products per tile, each tile's
  error relative to that tile's share; the running `O` and `D` are rescaled and added to
  once per tile (`2T`); the butterfly (5 steps), the division and a little margin make the
  8.

The checks allow twice that, because the numerator and the denominator both carry it:

- O: `|got - ref| <= 2 e * sum_c p_c |v_c| + 1e-7` for each column, where `p` is the
  reference probability. The absolute part only matters for an output near 0.
- L: `|got - ref| <= 2 e + 1e-5 + 2^-22 |ref|`. `log` turns the relative error of `D` into
  the same absolute error; `log`'s own bound is in none of our facts files (unverified:
  `1e-5` allows for it); `m` and the final addition round once each.

This is a first-order estimate, not a proof; the factor 2 is its margin. A forgotten
rescale, a wrong max or a missing barrier misses it by orders of magnitude. HIP publishes
measured errors rather than bounds (ch10-facts B4).

## The race backend

`emu` runs one thread at a time, and a thread runs until its next synchronisation point.
Every shuffle is a rendezvous of the whole warp, so by the time any lane leaves the sum
butterfly, every lane has written its P: on `emu` the missing warp barrier of item 1 above
never shows. `race` runs the threads at once under ThreadSanitizer, and shuffles there,
as on a GPU, do not order memory: it reports the P write (with its line) and the
`compute_O` read as a race. It also reports a missing block barrier between the tiles,
naming the tile load and the read it collides with. A clean run is not a proof. Lab 20.3
makes you find both.

## Before you benchmark, predict

The benchmark times flash attention against lab 20.1's unfused pipeline at 8 heads,
d = 128, n = 512, 1024, 2048 and 4096, batch 1:

- flash: your kernel, one launch for all heads;
- unfused: per head, `S = Q K^T` with M3 (lab 15.6's reference), lab 20.1's reference
  causal softmax, `O = P V` with M2 (lab 5.3's reference).

It prints milliseconds per call (the best of 5 timed runs, each repeating the call until it
lasts at least 2 ms), causal attention GFLOP/s counted as `2 n^2 d` per head (half of the
`4 n^2 d` of the two full products, since half of S is masked; the unfused path computes the
masked half anyway), their ratio, and the bytes the unfused path allocates for S and P.
Both outputs are checked before their times are reported. Sizes that do not fit in a
quarter of the device memory reported are skipped, and the line says so.

**Prediction `ch20/labs/p-lab2-ratio`.** Before you run it: at n = 4096, flash time
divided by unfused time on your device.

Two forces pull in opposite directions; the lesson does not say which wins on your device.

- Flash removes S and P from device memory. Per head the unfused path writes S, reads it
  in the softmax, writes P and reads it again in `P V`: at least `2 n^2 x 4` bytes written
  and as many read, 128 MiB per head at n = 4096.
- Flash's two products are the book's simple warp loops: one shuffle per multiply-add in
  the scores, scalar FMAs from shared memory in `P V`. The unfused path runs them on your
  tuned M3 and M2 GEMMs. The book's footnote 8 (pp. 500-501) says these products could
  use chapter 15's more advanced optimisations; production kernels use matrix hardware
  (A6, D2).

The numbers are measurements of your machine, at these sizes, not targets.

## Portability

- HIP: `kernel.cu` builds through `pmpp/vendors/hip.h`. The 32-lane logical warp, the
  typed mask and the explicit width make two logical warps share a 64-wide wavefront
  without mixing (B2). Not compiled or run on AMD hardware here: unverified. The padding
  `addr()` is tuned for 32 banks; AMD has 32 or 64 (B3).
- Metal: see above. The `B_c` your device gets is printed at the start of the test run.
- `kernel.cu` declares `expf` and `logf` under `PMPP_CUDA_FRONTEND_CHECK`, a macro only the
  course's CUDA front-end check defines.
