# Lab 14.3: sorted candidates, every row descending and stable (project milestone M5)

Book: radix sort as a chain of stable partitions (section 14.4, pp. 334-336), the destination
formulas of Fig. 14.6 (p. 338) and the kernel of Fig. 14.7 (p. 339), the local sort in shared
memory (14.6, pp. 339-342), a digit of r bits as r one-bit local passes (14.7, pp. 342-344),
several keys per thread (14.8, pp. 344-345). The book prints no code for 14.6 to 14.8 (they are
exercises 1 to 3, p. 347). You also reuse chapter 9's shared-memory histogram and chapter 11's
block scan. Not in the book, and ours: the float key, one block per row, the running offset per
bucket, and the gather. A bare item such as (A3) or (C1) below is in `notes/today/ch14-facts.md`.

Write one host function and the kernels behind it:

```cpp
void launchSortRowsDesc(const float* probs, float* sortedProbs, int* sortedIdx,
                        unsigned int* scratch, int rows, int cols);
```

For every row of `probs`, a `[rows x cols]` float32 tensor (row-major, batch first),
`sortedIdx` gets the column indices in descending order of value, equal values by increasing
column, and `sortedProbs` gets the values in that order. `scratch` holds `4 x rows x cols`
unsigned ints for you to use. This is the input to top-p's cumulative sum in module X1: sort,
then scan.

- Edit `work/kernel.cu` (CUDA; also built for HIP and the emulators): `descKeys`,
  `sortRowsPass`, `gatherRows`, `launchSortRowsDesc`. Edit `work/kernel.metal` (`desc_keys`,
  `sort_rows_pass`, `gather_rows`) and `work/launch.mm` (`encodeSortRowsDesc`, which encodes
  the whole sort into the one compute encoder it is given).
- Provided in those files: the block scan (chapter 11's scan-scan-add) and its helpers.
- Test: `python3 labs/pmpp.py run 14.3`. Start with `--backend emu`, which takes a few minutes.
- Races: `python3 labs/pmpp.py run 14.3 --backend race` (see "The race backend" below).
- Measure: `python3 labs/pmpp.py run 14.3 --bench`, after the prediction below.

## The contract

- `rows >= 1`, `cols >= 1`, any value: not only multiples of 256 or of your tile.
- Any float: negatives, `-0.0` and `+0.0`, infinities, `FLT_MAX`, subnormals, NaN.
- The order is that of this key, sorted ascending: take the 32 bits `u` of the float; if
  `u` is `-0.0` (`0x80000000`) make it `+0.0`; then set the sign bit of a non-negative value
  and invert every bit of a negative one; then invert the whole key. On numbers that is
  descending numeric order with the two zeros equal. On NaN it is the bits: `+NaN` comes
  before everything (even `+inf`) and `-NaN` after everything (even `-inf`). This is the
  transform CUB documents for its float keys, including the zero step and the NaN rule
  (A3; the bit patterns are worked in E1).
- Stable: equal keys keep increasing column order. Descending comes from inverting the key,
  not from reversing an ascending result, which would list ties by decreasing column (A3).
- `sortedProbs[r][p]` is exactly the 32 bits of `probs[r][sortedIdx[r][p]]`. `probs` is not
  modified. Nothing is written outside the three output buffers.

### Why bits, not floats

The kernels never perform a float operation. The launcher hands them `probs` and
`sortedProbs` as `unsigned int*` (in CUDA a cast; in Metal the buffers are bound as
`device const uint*` and `device uint*`), so the `-0.0` test, the sign test and the copy are
all integer work. That matters on Metal: both of its candidate default math modes allow "no
signed zeros", fast math leaves NaN and INF undefined, and subnormals may be flushed to zero in
arithmetic (C3). A float-side `x == 0.0f ? 0.0f : x` may be rewritten; an integer test may not.
On HIP, `__float_as_uint` exists in the header but is not in AMD's published math reference
(B6). The unsigned view needs no bit-cast intrinsic at all, so it is the form that works
everywhere. (CUDA documents `__float_as_uint`, A8.)

For this milestone's own input, softmax probabilities, every value is at least `+0.0`, so the
sign branch never fires (E1). The tests still include negatives, both zeros and infinities,
because the same kernel sorts logits.

## The design (ours)

One block of 256 threads per row, the shape of lab 10.3. Three kernels:

1. `descKeys`: each value's key (above), and its column as the index.
2. `sortRowsPass`, launched once per digit of r bits (your choice; 8 is suggested, so four
   passes over the 32 bits, least significant digit first), reading one key and index buffer
   of `scratch` and writing the other. Each block:
   - counts its row's digits in a shared-memory histogram of 2^r bins, with atomics
     (chapter 9);
   - scans the 2^r counts (exclusive): where each bucket starts in the output row;
   - walks the row in tiles of 256c keys, c keys per thread. Each tile is sorted in shared
     memory by the digit with r stable one-bit splits, using Fig. 14.6's formulas and the
     provided block scan; then each key is written to its bucket's running offset, and the
     offsets move on by the bucket sizes of the tile.
3. `gatherRows`: `sortedIdx` from the final index buffer, `sortedProbs` by copying the bits of
   `probs` at those indices.

Why the pass is stable across tiles: the tiles are visited in order by one block, and a
bucket's offset only moves forward, so a key of an earlier tile always lands before a key of a
later tile with the same digit. That running offset does the job of the book's table of bucket sizes per
block (Fig. 14.9): with one block per row there is only one block to account for. It also
removes Fig. 14.7's grid-wide scan, which the book calls but never defines (notes/ch14.md,
"Unclear in the PDF", item 6), and its read of `bits[N]`, one past the N entries the figure
shows (item 4).

Within a tile, consecutive threads write consecutive positions of one bucket: section 14.6's
coalesced write. With 2^8 buckets and a few thousand keys per tile, a bucket holds a few keys
per tile on random data, so the writes are far from fully coalesced: that is the cost of a
large radix that section 14.7 describes (and A10 counts in sectors).

## What the tests check

- Shapes: rows 1, 3, 4 and 7 by cols 1, 2, 31, 255, 256, 257, 1000, 4096, 32000, 50257 and
  151936 (the last two are the vocabulary sizes of GPT-2 and Qwen3-8B, ch09-facts D7). On
  `emu`, rows 1 and 3 by cols 1, 2, 31, 255, 256, 257, 1000 and 4099, with one batch per row
  count per size, the recipes rotating with the size, because every case runs under three
  thread orders and each order costs tens of thousands of thread switches per row. On `race`,
  rows 2 by cols 1, 31, 257, 1000 and 2100 (two tiles even at 8 keys per thread), every recipe.
- Data, mixed within a batch so that a kernel reading or writing the wrong row is caught:
  - softmax probabilities of logits uniform in [-12, 12];
  - the same with planted exact ties: columns 0 and cols-1 (equal to the row's maximum, which
    makes a three-way tie), 255 and 256, 5 and 261 (one thread's stride apart), and runs of 40
    equal values across every multiple of 256 from 512 to 8192. A tile is 256c keys, so for
    any whole c from 1 to 32 a run crosses a tile boundary once the row is longer than
    256c + 20 columns;
  - every value equal (the answer is 0, 1, ..., cols-1);
  - one-hot (the zeros must stay in column order behind the one);
  - logits with `-0.0` and `+0.0` interleaved every 11 columns, and (as far as the row has
    room) two each of `-inf`, `+inf`, `-FLT_MAX`, `FLT_MAX`, `FLT_MIN`, `-FLT_MIN` and four
    subnormals of both signs;
  - the same with up to four NaNs, two of each sign, two of them with payloads.
- `sortedIdx` equals the reference exactly (that is the stability check); `sortedProbs` equals
  the gathered bits exactly; each output row is a permutation; 64 guard entries past the end of
  `sortedProbs`, `sortedIdx` and `scratch` are untouched; `probs` is unchanged.
- No tolerance: a sort does no arithmetic and the values are copied as bits.
- The reference, in `main.cpp`: `std::stable_sort` of the column indices on the integer key.
  Not Accelerate's `vDSP_vsorti`, which Apple documents as not stable (C2). On every row without
  NaN the driver also checks its reference against `std::stable_sort` with `>` on the floats,
  which treats the two zeros as equal and keeps input order (A11). With NaN, `>` is not a strict
  weak ordering, so the key is the only reference.

## The race backend

`emu` runs one thread at a time, so it cannot see two threads touching the same memory with
nothing ordering them if the result happens to come out right. `race` runs a block's threads
at once under ThreadSanitizer and reports such pairs with both source lines. In this lab the
places to get right are the histogram (atomics, and a barrier between zeroing, counting and
reading), the tile buffer (every split reads it and then rewrites it), the bucket starts of a
tile, and the running offsets (read by the scatter, then advanced, then read by the next tile).

## The library route, per backend

The book's advice is to use a library sort unless you have a reason not to (p. 347). For this
operation, a stable descending sort of (key, index) pairs per row:

- CUDA: `cub::DeviceSegmentedRadixSort::SortPairsDescending`, or
  `cub::DeviceSegmentedSort::StableSortPairsDescending`. The plain
  `DeviceSegmentedSort::SortPairsDescending` is documented as not stable (A5). The values are
  the token indices, exactly the shape of this lab's output (A2).
- HIP: the same names under `hipcub::` in both hipCUB copies of ROCm 10.2 (B1). In hipCUB
  4.7.0 they run rocPRIM's segmented radix sort, and `StableSortPairsDescending` simply calls
  `SortPairsDescending` (B1); rocPRIM documents its device radix sort as stable (B2).
- Metal: no sort in MSL and no kernel-level library sort. MPSGraph's
  `argSort(_:axis:descending:name:)` (macOS 13) sorts along an axis, and Apple states no
  stability or tie order for it (C1). If equal probabilities must come out by increasing token
  index, you write the kernel, as here. PyTorch's MPS backend has its own: a radix sort with
  this chapter's count, scan and scatter design in Metal and 8-bit digits for 32-bit keys. Its
  key differs from this lab's: `-0.0` and `+0.0` stay distinct, and NaN goes last (D6, our
  reading of its source).

### Why this kernel matters on HIP for llama.cpp

llama.cpp's CUDA `argsort` sorts short rows with a bitonic network in shared memory and long
rows with CUB. Its HIP build defines no CUB path at all, and the bitonic path needs the row,
padded to a power of two, in shared memory: a 151,936-column row pads to 262,144 ints, 1 MiB.
By our reading of `supports_op`, a vocabulary-sized `argsort` is therefore reported unsupported
on HIP and runs on another backend (D1). A row-wise sort whose shared memory does not grow with
the row, like this one, is what that path lacks (our reading, and not compiled or run on AMD
hardware here).

## Before you benchmark

Write down two predictions (the lesson asks for them as `ch14/labs/p-lab3-batch1`):

1. At batch 1 and 151,936 columns, how many times faster than one CPU thread's
   `std::stable_sort` will your kernel be? A number below 1 means slower.
2. Will that ratio grow or shrink at batch 64?

Section 14.8's bridge and chapter 10's section 10.10 explain why one block per row leaves most
of a device idle at batch 1.

The benchmark sorts softmax probabilities at rows 1, 8 and 64 by cols 32,000 and 151,936. Time
per call is the best of 5 timed runs on the device, each repeating the whole sort (keys, every
pass, gather) back to back until the run lasts at least 20 ms (runs of one short call were
seen to vary by up to a factor of two); it reports rows per second and millions of keys per
second. The baseline is `std::stable_sort` with `>` on each row in turn plus the gather, on one
CPU thread, timed on the host with a steady clock (best of 5). Every timed result is checked
before it is reported, and a shape larger than a quarter of the device memory reported is
skipped. The numbers are measurements of your machine, not targets.

## Portability

- HIP: `kernel.cu` builds through `pmpp/vendors/hip.h` with the typed shuffle mask and
  `warpSize`; with 64-wide warps a block of 256 has 4 warps. `<cuda/atomic>` on HIP comes from
  libhipcxx, part of ROCm from 10.1 (infra-facts B2). Not compiled or run on AMD hardware here:
  unverified.
- Metal: `launch.mm` reads the SIMD-group width from the pipeline (`threadExecutionWidth`),
  checks that 256 threads make whole SIMD-groups with no more groups than lanes, sizes the
  per-SIMD-group scan memory from it (a multiple of 16 bytes, ch05-facts C8), and checks the
  kernel's threadgroup arrays fit. The threadgroup histogram is `atomic_uint` (ch09-facts C1).
  The kernel ranks its threads by SIMD-group and lane, because the split of a threadgroup into
  SIMD-groups is the implementation's (ch04-facts C2) and the scan runs in that order. All the
  dispatches go into one compute encoder: a plain one is serial, and device buffers are
  hazard-tracked, so each pass sees the one before (ch02-facts C9).
- The CUDA front-end check needs the shim include directory that `labs/pmpp.py` passes
  (`-I labs/common/include/pmpp/emu/shim`) for `<cuda/atomic>`.
