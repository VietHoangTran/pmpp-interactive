# Lab 20.1: causal mask and softmax (milestone M6, part one)

Book: section 20.3 (pp. 486-488), the fused mask-and-softmax kernel of Fig. 20.4 (p. 487).
The reductions inside it are lab 10.3's (milestone M4). The batch dimension, the
lower-right mask for `nq < nk` and the logsumexp `L` are not on the page: they are ours,
and the reasons are below. A bare item such as (A1) or (D5) is in
`notes/today/ch20-facts.md`.

Write the middle stage of unfused attention: given the scores `S = Q K^T`, produce
`P = softmax(scale * S + M)`, where `M` is the causal mask, and the logsumexp of every row.

```
launchCausalSoftmax(S, P, L, batch, nq, nk, scale)
```

| Tensor | Shape |
|---|---|
| `S` (in), `P` (out) | `[batch, nq, nk]` |
| `L` (out) | `[batch, nq]` |

Row-major `float`, batch first. `batch` counts (sequence, head) pairs: one item per head of
every sequence.

## The contract

- `1 <= nq <= nk`, `batch >= 1`, every score finite.
- **The mask is aligned to the lower right.** Row `r` sees the columns `c <= r + (nk - nq)`.
  With `nq == nk` that is the book's rule, column at most row (p. 484). With `nq == 1` the
  single row sees every column: one decode step, which needs no mask (Fig. 20.6).
  FlashAttention (since 2.1) and MLX align the mask this way; PyTorch's `is_causal` aligns
  it to the upper left when `nq < nk`, which would let a one-token query see only key 0
  (A8, D4, D5).
- With `m` the largest `scale * S` over the visible columns of a row and
  `s = sum of exp(scale * S - m)` over them:
  - `P = exp(scale * S - m) / s` on the visible columns, **exactly 0** on the masked ones;
  - `L = m + log(s)`, the logsumexp of the row's visible scaled scores.
- A masked score is never read. Its value is irrelevant, and the tests make it 1e30 to
  prove the point.

`L` is what lab 20.2's flash kernel must reproduce for the same scores, so its tests can
compare the two.

## The design (Fig. 20.4's)

- **One block of 256 threads per row.** Grid x is the row (the book's `blockIdx.x`),
  grid y the item. Thread `t` handles columns `t, t + 256, t + 512, ...`: thread
  coarsening, coalesced on every pass (p. 488).
- **The mask is the loop bound.** The max and the sum loops stop at the last visible
  column, so masked entries are never touched; the output loop writes explicit zeros for
  them. That is the book's trick, and it is why no infinity appears anywhere.
- **Three passes over the visible part of the row**: the max, then the sum of
  `exp(scale * S - m)`, then the write of P. Between the max and the sum, a barrier.
- **The reductions are given to you**: lab 10.3's two-stage block reductions,
  `blockReduceMax` and `blockReduceSum`, copied unchanged and credited in the file. They
  return the result to every thread through `scratch[0]`.

You write the masked kernel (its three passes, the barrier between the reductions, and
`L`) and the launch.

## Where this differs from the page

The page needs four corrections, and makes two choices we change:

1. **`gridDim.y` and `gridDim.z` are 1, not 0** (p. 487 says 0). Unused grid dimensions
   default to 1 on CUDA and HIP (A1). Here grid y is used anyway: it walks the batch.
2. **No `NEG_INFINITY`.** No CUDA header defines that name (A5). The identity of max is
   `-FLT_MAX`, finite, as in lab 10.3: under Metal's fast math the behaviour of infinity is
   undefined (C5), and one finite contract serves all three backends.
3. **No `cub::Max()` or `cub::Sum()`.** Those functors were removed in CCCL 3.0 (infra A11),
   and hipCUB spells things differently (B1). The M4 reductions replace `BlockReduce` and
   the book's `max_or_sum` broadcast: they already hand the result to every thread.
4. **`L` in place of `D`.** The book writes the denominator `D` "for reuse during
   training" (p. 488). `D` alone cannot rebuild P, because the row max it was taken
   against is not saved. FlashAttention-2 stores the logsumexp `m + log(D)` instead (A7),
   and so does this lab.

The two choices: the kernel applies `scale` itself (Fig. 20.4 reads S already scaled; the
book allows either place, p. 486), so that S can come straight from the M3 linear layer;
and `exp` and `log` are written `expf` and `logf` in `kernel.cu`, single precision said out
loud (A5).

## Files

- `work/kernel.cu` (CUDA; also built for HIP and the two emulators):
  `causalSoftmaxKernel` and `launchCausalSoftmax`. The M4 block reductions at the top are
  provided; the TODOs are in order.
- `work/kernel.metal`: `causal_softmax`, the same kernel with `simd_max` / `simd_sum` and
  threadgroup memory (lab 10.3's Metal reductions, provided). `work/launch.mm`:
  `encodeCausalSoftmax` checks the pipeline, binds the arguments and sizes the scratch;
  you write the dispatch.
- `main.cpp`: the driver. The CPU reference and the tolerances are in
  `../shared-g11/unfused.hpp` (namespace `causal`), shared with lab 20.2's benchmark.
- Provided for the benchmark, not edited: `../shared-g11/unfused.cmake` copies lab 15.6's
  reference `launchLinear` (M3) and lab 5.3's reference `launchGemm` (M2) into the build,
  renamed. If either lab's solution is missing, its rows are absent.

## Commands

- Test: `python3 labs/pmpp.py run 20.1 --backend emu` first, then `--backend race` (below),
  then `python3 labs/pmpp.py run 20.1` on your GPU.
- Measure: `python3 labs/pmpp.py run 20.1 --bench`.

## What the tests check

- **Shapes**, `(nq, nk)`: (1, 1), (1, 2), (1, 4097), (7, 100), (31, 31), (32, 32), (33, 33),
  (255, 255), (256, 256), (257, 257), (1000, 1000), (2048, 2048), each at batch 1 and 3.
  Rows of 1 and 2 columns leave most threads idle; 255, 256, 257 sit on either side of one
  pass of the block; (7, 100) has `nq < nk`; 4097 columns take 17 passes. `emu` runs
  batch 1 and 2 with `nk` up to 300; `race` runs batch 1 with `nk` up to 100.
- **Data**, one recipe per row, mixed within a batch so that a kernel that reads the wrong
  row or the wrong item is caught:
  - scores uniform in [-8, 8];
  - the same offset by +90 and by -90. `exp(90)` is beyond float's range, so a kernel that
    skips the max subtraction returns infinity or NaN;
  - the row's largest score at its last visible column (an off-by-one in the bound misses
    it);
  - 1e30 in every masked column: a kernel that reads a masked score for its max or its sum
    gets a row of zeros or NaN;
  - every score equal (P is `1 / visible`).
- **Scales** 1 and `1/sqrt(128)` (`emu` runs the second, `race` the first).
- **Checks**: P within the tolerance below and every row summing to 1; masked P exactly 0;
  L within its tolerance; P and L start as NaN, so an element nobody writes is caught; 64
  canary values after P and after L must be untouched.

## Tolerances, and why

The reference is computed in double on the CPU from the same float scores
(`causal::refRow`). The bounds follow lab 10.3's derivation, at the worst case tested:
4097 columns, exponent arguments down to about -17 at scale 1.

**P on a visible column:** `|got - ref| <= 1e-20 + 5e-5 * ref`, and the row sums to 1
within `5e-5`. The relative error of one P is at most the sum of:

- its own `exp`: CUDA's `expf` is within 2 ulp (ch10-facts A11), Metal's precise `exp`
  within 4 ulp, its fast `exp` within `3 + floor(|2x|)` ulp (ch10-facts C4): 37 ulp at
  x = -17, about 4.4e-6;
- its argument: `scale * S` is rounded once before `m` is subtracted, up to half an ulp
  of 98 for the +90 rows, 3.8e-6 absolute, so 3.8e-6 relative after `exp`;
- the sum `s`: every term carries the same two errors (at most 8.2e-6 together), plus 17
  additions in a thread and two 5-step shuffle trees, 27 roundings of 2^-24, 1.6e-6
  (every term is positive);
- one reciprocal and one multiplication, 1.2e-7.

That is at most about 1.8e-5; `5e-5` leaves more than twice that. A missing max
subtraction, a wrong scale, a wrong bound or a masked score in the sum misses by orders of
magnitude. HIP publishes measured differences, not bounds (ch10-facts B4); beyond that,
unverified. The absolute term only keeps a subnormal probability, which a GPU may flush to
zero, from failing a row on its own.

**L:** `|got - ref| <= 5e-5 + 2^-21 * |ref|`. `log(s)` turns the relative error of `s` (at
most 1e-5 by the list above) into the same absolute error, and `log` adds its own, which
our facts files do not bound (unverified: 5e-5 leaves room for a few ulp of a value below
9). `m` is exact up to the rounding of `scale * S` (2^-24 relative) and the final addition
rounds once more (2^-24 relative): `2^-21` covers both four times over.

Write `expf` and `logf`, not `__expf`: the bound uses `expf`'s figure.

## The race backend

`emu` runs one thread at a time and cannot see a race that happens to leave the right
values. `race` runs the threads of a block at once under ThreadSanitizer. The case it is
here for is lab 10.3's: `blockReduceMax` and `blockReduceSum` share `scratch`, and every
thread reads the max from `scratch[0]` just before the sum reduction writes `scratch[0]`
again. Without something ordering the two, a fast warp can overwrite the max before a slow
warp has read it. On a GPU that defect can pass every test; ThreadSanitizer names the two
lines. A clean run is not a proof.

## Before you benchmark, predict

The benchmark runs the unfused attention pipeline at 8 heads, d = 128, N = 1024, 2048
and 4096, one launch per head:

1. `S = Q K^T`: M3, lab 15.6's reference `launchLinear` (the weight layout `(out, in)` is
   exactly K's `(n, d)`);
2. your causal softmax, scale `1/sqrt(d)`, `nq = nk = N`;
3. `O = P V`: M2, lab 5.3's reference `launchGemm`.

It prints each stage's time (the best of 5 timed runs, each repeating the stage until it
lasts at least 2 ms), the softmax's share of the total, its GB/s counted on its minimum
traffic (the visible half of S read once, P written once), and the bytes S and P occupy.
Every stage's output is checked before its time is reported. Sizes that need more than a
quarter of the memory your device reports are skipped.

**Prediction `ch20/labs/p-lab1-share`.** Before you run it: at N = 4096, d = 128, what
percentage of the three-stage pipeline's time will your mask-and-softmax kernel take?

Hint: the S product does `2 N^2 d` FLOPs (M3 computes all of S, masked half included).
The softmax moves about `10 N^2` bytes: the visible half of S is `2 N^2` bytes, read three
times, and P is `4 N^2` bytes, written once. Divide each by the rate it is limited by, your
compute ceiling and your bandwidth from lab 5.1; their ratio sets the share. If both
kernels ran at their ceilings, a device whose ridge point is R FLOP/B would spend
`2 d / (10 R)` times as long in the product as in the softmax. Then ask how far M3 and
M2 are from the compute ceiling, and your softmax from the bandwidth one.

The numbers are measurements of your machine, at these sizes, not targets.

## Portability

- HIP: `kernel.cu` builds through `pmpp/vendors/hip.h`; the provided reductions use the
  typed lane mask and `warpSize`, so a 64-wide wavefront gives 4 warps per block of 256.
  Not compiled or run on AMD hardware here: unverified.
- Metal: `launch.mm` reads the SIMD-group width from the pipeline (`threadExecutionWidth`)
  and refuses to dispatch unless 256 threads make whole SIMD-groups and no more
  SIMD-groups than a SIMD-group has lanes (stage 2 gives each group's result one lane).
- `kernel.cu` declares `expf` and `logf` under `PMPP_CUDA_FRONTEND_CHECK`, a macro only the
  course's CUDA front-end check defines: that check has no CUDA math declarations of its
  own.
