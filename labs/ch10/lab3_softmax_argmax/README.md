# Lab 10.3: softmax and argmax over logits (project milestone M4)

Book: the two-stage warp-level reduction (section 10.8, Fig. 10.16, p. 243), the coarsening
loop (section 10.10, Fig. 10.20, p. 247) and exercise 4's max reduction (p. 250). The
softmax, the argmax and the decoder are not in the book: they are ours, built from those
parts. A bare item such as (A11) or (D2) below is in `notes/today/ch10-facts.md`.

Write three row-wise kernels over logits stored as `[rows x cols]` float32, row-major,
batch first:

- `launchRowMax(x, out, rows, cols)`: the largest value of each row.
- `launchSoftmaxRows(x, y, rows, cols, scale)`: `y = softmax(scale * x)` for each row,
  computed stably: a max reduction, then a sum of `exp(scale * x - max)`, then the
  division. `scale` is 1/temperature when you sample (module X1) and 1/sqrt(d) in
  attention (chapter 20).
- `launchArgmaxRows(x, idx, rows, cols)`: the index of the largest logit of each row, ties
  to the lowest index. This is the greedy-decoding step.

One block of 256 threads per row. Each thread walks its row with a loop strided by the block
size (coalesced), then the block reduces with lab 10.1's two-stage warp reduction: typed
mask, `warpSize`, stage 2 guarded.

- Edit `work/kernel.cu` (CUDA; also built for HIP and the emulators) and
  `work/kernel.metal` + `work/launch.mm` (Metal). The names are fixed: the driver calls
  `launchRowMax`, `launchSoftmaxRows`, `launchArgmaxRows`, the Metal kernels `row_max`,
  `softmax_rows`, `argmax_rows`, and the encoders `encodeRowMax`, `encodeSoftmaxRows`,
  `encodeArgmaxRows`.
- Provided, not edited: `labs/ch10/shared/rowwise_naive.cu`, `rowwise_naive.metal`
  and `rowwise_naive_launch.mm`, one thread per row, the benchmark baseline.
- Test: `python3 labs/pmpp.py run 10.3`. Start with `--backend emu`.
- Races: `python3 labs/pmpp.py run 10.3 --backend race` (see "The race backend" below).
- Measure: `python3 labs/pmpp.py run 10.3 --bench`.

## The contract

- `rows >= 1`, `cols >= 1`, any value: not only multiples of 32 or 256.
- Every logit is finite. NaN and infinity are outside the tested contract (last section).
- Ties go to the lowest index. The tree combines (value, index) pairs in an order you do
  not control, so the combine must give the same pair in any order: the larger value wins,
  and of two equal values the lower index wins. That rule is commutative and associative
  on finite values, which is what sections 10.4 and 10.9 ask of an operator. PyTorch
  documents the same rule for `torch.argmax` (notes/today/ch10-facts.md D4); llama.cpp's
  CUDA argmax, its Metal argmax and its CPU sampler each break ties differently (D2), so
  the same greedy step can pick different tokens on different backends.
- The identity of max is `-FLT_MAX`, not the book's -infinity (p. 222). Under Metal's
  fast math the behaviour of infinity is undefined (C3), and one finite identity keeps the
  three backends on one contract. The argmax identity is the pair (`-FLT_MAX`, `INT_MAX`):
  it loses every tie, even against a logit equal to `-FLT_MAX`.

## What the tests check

- Shapes: rows 1, 4 and 7 by cols 1, 2, 31, 32, 33, 255, 256, 257, 1000, 4096, 32000,
  50257 and 128256 (the last three are vocabulary-sized). The emulator uses rows 1 and 3
  and cols up to 4099; the race backend rows 2 and cols up to 1000.
- Data, row by row, mixed within a batch so that a kernel reading the wrong row is caught:
  logit-like values uniform in [-12, 12]; the same offset by +90 and by -90 (exp(90) is
  beyond float's range, so a softmax that skips the max subtraction returns infinity or
  NaN: appendix A, pp. 590-591); all-equal rows (argmax 0, softmax 1/cols); all-negative
  rows (catch a max identity of 0); a row of `-FLT_MAX`; and planted exact maxima at
  columns (1, 2), (0, cols-1), (5, 261), (6, 261), (3, 40) and (cols-2, cols-1). The pairs
  put the tie in neighbouring lanes, at both ends, twice in one thread's loop, with the
  higher index on the lower lane, in two warps, and at the end of the row. Between them they
  catch "keep my own pair on a tie", ">= in the thread loop" and "the highest index wins".
- Scales 1, 0.5 and 0.125 for softmax.
- Row max exact, argmax exact, softmax within the tolerance below, and nothing written past
  the end of any output.
- Greedy decoding (below).
- The provided baseline is tested too, so that a broken provided file shows up here.

## Tolerances, and why

Max and argmax are compared exactly: a max does no arithmetic.

Softmax is compared per element with `|got - ref| <= 1e-20 + 1e-4 * ref`, where `ref` is
computed in double on the CPU from the same float inputs, and every row must sum to 1
within `1e-4`. The relative part comes from the error bounds, at the largest row tested
(128,256 columns, exponent arguments between -24 and 0):

- `exp`: CUDA's `expf` is within 2 ulp (ch10-facts A11). Metal's precise `exp` is within
  4 ulp, its fast `exp` within `3 + floor(|2x|)` ulp (C4), which is 51 ulp, about 6e-6
  relative, at x = -24. HIP publishes measured differences against the C++ library, not
  bounds (1 ulp for `expf` on [-10, 10], B4); beyond that range it is unverified.
- The argument `scale * x - max` is itself rounded once: up to 1.4e-6 relative in `exp`.
- The sum: each thread adds about 500 terms in turn, then two tree stages, at most about
  540 roundings of 2^-24 each, so 3.2e-5 relative (every term is positive).
- One reciprocal and one multiplication: 1.2e-7.

That totals about 4e-5 with Metal's fast `exp` (6.1e-6 + 1.4e-6 + 3.2e-5 + 1.2e-7) and
3.4e-5 with CUDA's `expf`; `1e-4` leaves more than twice the larger. A missing max subtraction, a wrong scale or a wrong denominator miss
by orders of magnitude. The absolute term is far below the smallest probability the test
rows produce (about 1e-14), so in practice the check is relative; it is there so that a
subnormal result, which a GPU may flush to zero, would not fail a row on its own.

Write `expf`, not `__expf`: the bound above uses `expf`'s figure, and `__expf`'s error
grows with |x| (2 + floor(|1.173 x|) ulp, A11). The provided baseline adds a whole row in
one thread, one rounding per column at worst, so the driver holds it to `1e-4 + cols * 2^-24`.

## The race backend

`emu` runs one thread at a time and cannot see a race that happens to leave the right
values. `race` runs the threads of a block at once under ThreadSanitizer and reports two
threads touching the same memory with nothing ordering them, with both source lines. The
case this lab is built for: two block reductions in a row that share one shared array. The
second writes the array while slower warps may still be reading the first one's result, so
something must order them (llama.cpp and PyTorch both do, D1, D4). On a GPU that defect
can pass every test, because whether a fast warp overwrites the result before a slow one has
read it depends on scheduling.

## Greedy decoding (the milestone)

The driver also runs a toy greedy decoder on your argmax. The "model" is a deterministic
lookup table of logits, `V x V`: after token t the logits are row t, and every row has its
largest value at two columns, an exact tie. For a batch of 4 sequences, each step copies the
rows of the current tokens into a `[4 x V]` logits buffer, calls `launchArgmaxRows`, and
feeds the indices back as the next tokens. After 16 steps the sequences must match the CPU's.

That is greedy decoding end to end for a table, not for a language model. A real checkpoint
comes at milestone M10; the kernel it will call is this one.

## Before you benchmark

Write down two predictions (the lesson asks for them as `ch10/labs/p-lab3-naive`):

1. At batch 1 and 128,256 columns, how many times faster will your block-per-row softmax
   be than one thread per row?
2. Will the gap grow or shrink at batch 64?

Section 10.10's explorer shows why one block per row leaves most of a GPU idle at batch 1.

The benchmark times softmax (scale 1) and argmax at rows 1, 8 and 64 by cols 32,000 and
128,256, for your kernels ("block") and the baseline ("naive"). Time per call is the best of
5 timed runs, each repeating the call until it lasts at least 2 ms. GB/s counts the minimum
traffic: 8 bytes per element for softmax (read x once, write y once) and 4 for argmax (read
x once). Your softmax reads each row three times, so it moves more than it is credited with.
Every timed result is checked before it is reported. The numbers are measurements of your
machine, not targets.

## NaN and infinity

Outside the tested contract, for three reasons:

- Metal's fast math, which may be the runtime's default (Apple documents the default three
  different ways, ch05-facts C7), leaves the behaviour of NaN and infinity undefined (C3). A
  test that passed there would prove nothing.
- CUDA's `fmaxf` drops a NaN operand (A10), while a comparison such as `a > b ? a : b`
  returns its second operand whenever either is NaN: the same reduction gives different
  answers depending on how its max is written.
- Libraries disagree, so there is no single answer to test against: PyTorch and MLX let NaN
  win argmax (D3, D4); llama.cpp's CUDA argmax, by our reading of its source, skips NaN and
  returns -1 for a row of -infinity (D2); CUB's `Max` ignores NaN while its `ArgMax` does
  not say (A13).

Masked logits set to -infinity are the case that matters in practice (attention masks,
banned tokens). They come back with the mask and softmax kernel of chapter 20.

## Portability

- HIP: `kernel.cu` builds through `pmpp/vendors/hip.h` with the typed mask and `warpSize`;
  with 64-wide warps a block of 256 has 4 warps, and stage 2 pads the other lanes. Not
  compiled or run on AMD hardware here: unverified.
- Metal: `launch.mm` reads the SIMD-group width from the pipeline (`threadExecutionWidth`),
  checks that 256 threads make whole SIMD-groups with no more groups than lanes, and sizes
  the per-SIMD-group threadgroup memory from it (a multiple of 16 bytes, ch05-facts C8).
- `kernel.cu` declares `expf` under `PMPP_CUDA_FRONTEND_CHECK`, a macro only the course's
  CUDA front-end check defines: that check has no CUDA math declarations of its own yet.
