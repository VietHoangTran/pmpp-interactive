# Lab X1.2: top-p and min-p, composed from earlier labs, then fused (project milestone M5)

Not in the book: module X1 is ours (the book has no sampling at all, PLAN.md). The pieces are
the book's: chapter 10's two-stage reduction (Figs. 10.13 and 10.16, pp. 240 and 243) and
chapter 11's warp and block scans (Figs. 11.8 and 11.9, p. 264), which labs 10.3, 11.6 and 14.3
built into a softmax, a top-p mask and a row sort. The order of the two filters, the top-p rule
and the min-p test are llama.cpp's; the fusion, the prefix argument and the error bound are
ours. A bare item such as (D1) below is in `notes/today/x1-facts.md`; others name their file.
What llama.cpp's source does was checked in the facts pass on 2026-11-20.

After the model's last matmul, a sampler turns a row of logits into one token. Top-p keeps the
smallest set of the most likely tokens whose probability reaches p; min-p keeps the tokens
whose probability is at least min-p times the largest. llama.cpp's default chain runs top-p,
then min-p, then temperature, then the draw (D1). This lab builds the top-p and min-p stage
twice: once by composing what you already have, once as one kernel.

**Part A, compose.** Write

```cpp
void launchFilterRows(const float* candLogits, int* sortedPos, float* sortedLogits,
                      float* sortedProbs, int* n, void* workspace,
                      int rows, int K, float topP, float minP);
```

from lab 14.3's `launchSortRowsDesc`, lab 10.3's `launchSoftmaxRows` and lab 11.6's
`launchTopPMask` (their reference solutions are provided), plus one kernel of your own,
`minPCount`, with its launcher

```cpp
void launchMinPCount(const float* sortedLogits, const int* keptTopP, int* n,
                     int rows, int K, float minP);
```

which counts the min-p survivors of each sorted row and writes `n` from that count and lab
11.6's `keptTopP`. The benchmark also calls it on its own, to time the chain after the sort.

**Part B, fuse.** Write

```cpp
void launchFilterSortedFused(const float* sortedLogits, float* sortedProbs, int* n,
                             int rows, int K, float topP, float minP);
```

one kernel, `filterSortedFused`, one block of 256 threads per row, that starts from rows that
are already sorted and produces the same `sortedProbs` and `n` while reading the logits twice
and writing the probabilities once.

## Files and commands

- Edit `work/kernel.cu` (CUDA; also built for HIP and the emulators): `minPCount`,
  `launchMinPCount`, `launchFilterRows`, `filterSortedFused`, `launchFilterSortedFused`.
  Edit `work/kernel.metal` (`min_p_count`, `filter_sorted_fused`) and `work/launch.mm`
  (`encodeMinPCount`, `encodeFilterRows`, `encodeFilterSortedFused`). The names are fixed.
- Provided in `work/kernel.cu`: the warp sum, the warp scan and the block sum of chapters 10
  and 11, as `static` functions. Keep your own helpers `static` too: the provided copies below
  define functions with some of the same names, and the emulator links everything into one
  program.
- Provided, not edited, in `labs/x1/shared-g12/`: `sortrows_g12.cu`, `softmax_g12.cu` and
  `topp_g12.cu`, copies of the reference solutions of labs 14.3, 10.3 and 11.6 under file
  names of their own, with `laneIdx`, `warpIdx` and `warpsInBlock` made `static` so that the
  three link into one emulator program. `app/src/chapters/x1/copies-g12.test.ts` checks that
  nothing else differs from the originals. On Metal the driver loads the originals'
  `solution/kernel.metal` from their own directories and links their `solution/launch.mm`.
- Test: `python3 labs/pmpp.py run x1/lab2`. Start with `--backend emu` (several minutes: lab
  14.3's sort alone is dozens of block scans per row there).
- Races: `python3 labs/pmpp.py run x1/lab2 --backend race`.
- Measure: `python3 labs/pmpp.py run x1/lab2 --bench`, after the predictions below.

## The contract

Every row holds K finite logits; `rows >= 1`, `K >= 1`; row-major, batch first. For each row:

- **Part A only.** `sortedPos` holds the positions of the row sorted by logit in descending
  order, equal logits by lower position (stable). `sortedLogits[i]` is the same 32 bits as
  `candLogits[sortedPos[i]]`.
- `sortedProbs` is the softmax at temperature 1 of the sorted row:
  `exp(l_i - l_0) / sum_j exp(l_j - l_0)`, where `l_0`, the first sorted logit, is the largest.
- `n = max(1, min(a, b))`, where
  - `a`, top-p: the number of tokens `i` with `i == 0`, or `topP >= 1`, or the mass ranked
    before `i` (`sortedProbs[0] + ... + sortedProbs[i-1]`) below `topP`. That is lab 11.6's
    keep count.
  - `b`, min-p: the number of tokens with `sortedLogits[i] >= sortedLogits[0] + lnMinP`,
    where `lnMinP = logf(minP)` is computed once, on the host, as a float. Every token when
    `minP <= 0`.
- The inputs are not modified, and nothing is written past the end of any output.

Why each clause:

- **Top-p keeps the token that crosses p.** llama.cpp's host and graph samplers both keep
  exactly the tokens whose mass before them is below p (`notes/today/ch11-facts.md` D2),
  which is the top-p paper's "smallest set whose mass is at least p" (D9). `p >= 1` keeps
  everything because llama.cpp returns early there, and because a float scan's mass before
  the last tokens can round to 1. Lab 11.6's README has the rest.
- **Min-p on logits, against the largest plus a logarithm.** llama.cpp's host min-p computes
  `max_logit + logf(p)` once and keeps `logit >= min_logit` (`ch12-facts.md` D3). In exact
  arithmetic that is `p_i >= minP * p_max`, the min-p paper's rule (D9), with no softmax
  needed. The comparison is `>=`: a logit exactly at the threshold survives. Computing
  `logf` once on the host makes the threshold the same float on every backend.
- **Top-p, then min-p.** That is llama.cpp's default order (D1); mlx-lm runs the same two in
  the same order (D8). Both are applied before temperature there.
- **`n` is a minimum.** Both filters keep a prefix of the sorted row: top-p because the mass
  before a token only grows down the row, min-p because the logits only fall. Min-p's
  threshold uses the largest logit, which top-p never removes, so running min-p after top-p
  sees the same threshold as running it on the whole row. A token survives both exactly when
  it is in the shorter prefix. `max(1, ...)` because the largest logit always survives top-p
  and survives min-p for any `minP <= 1`; for `minP > 1` nothing passes min-p, and a sampler
  must still return a token.
- **The sort is stable**, so of two equal logits the one at the lower position comes first.
  At a top-p boundary that is the tie rule (lab 11.6: "ties are broken by position, that is,
  by the sort"). mlx-lm keeps every token tied with the boundary instead (ch11-facts D3).

### Part A's workspace

`workspace` is one allocation the driver makes, laid out in this order:

| Piece | Size | Used by |
|---|---|---|
| sort scratch | `4 * rows * K` uint32 | lab 14.3 |
| `massBefore` | `rows * K` float32 | lab 11.6 |
| `keptTopP` | `rows` int32 | lab 11.6, then your `minPCount` |
| `keep` | `rows * K` uint8 | lab 11.6 |

`21 * rows * K + 4 * rows` bytes in all; `keep` comes last so that nothing after it needs
alignment. On Metal the same four pieces are four buffers, because lab 11.6's
`encodeTopPMask` binds each of its buffers at offset 0.

## Suggested design for part B

One block of 256 threads per row, lab 11.6's shape:

- The row is sorted, so its maximum is element 0: no max pass.
- Pass 1: the sum of `exp(l - l_0)`, coarsened over the row, then chapter 10's two stages
  (the provided `blockSum`).
- Pass 2: chunks of 1,024 tokens (4 per thread), with the mass of earlier chunks carried in a
  register. Each warp reads its slice of logits coalesced, computes each probability, writes
  it to `sortedProbs` and stages it in shared memory; that is also where the logit is in a
  register, so decide min-p there. Each lane scans its run of 4 consecutive probabilities in
  registers, the block scans the run totals (warp scans, then warp 0 over the warp totals),
  and each token's mass before it, earlier chunks included, decides top-p while it is in a
  register.
- Count both decisions per thread, add the counts with one two-stage reduction, and let one
  thread write `n`.
- Shared arrays are reused: by the reduction and the scan, and by one chunk and the next.
  For each, find the last read and the next write, and put a barrier between them.

On Metal, `simd_sum`, `simd_prefix_exclusive_sum` and `simd_prefix_inclusive_sum` do in one
call what the shuffle loops do (`infra-facts.md` C2, ch11-facts C1).

## What the tests check

Both parts run on the same inputs: part A on the candidates, part B on the reference's sorted
logits.

- Shapes: rows 1, 3 and 7 by K = 1, 2, 31, 40, 255, 256, 257, 1,000, 1,024, 1,025, 4,096,
  32,000 and 151,936 (40 is llama.cpp's default top-k, D1; 151,936 is a vocabulary size,
  `ch09-facts.md` D7). The emulator runs rows 1 and 3 and K up to 4,099, one batch per
  setting; the race backend 2 rows of K = 1, 33, 257, 1,025 and 2,100. On both, part A runs at
  two sizes only (K = 40 and 1,025; race: 33 and 1,025), on the planted min-p row and on
  softmax rows at p = 0.9, min-p = 0.05, because lab 14.3's sort is most of their time.
  Rows of different kinds are mixed in one batch, so a kernel that reads the wrong row fails.
- **Exact rows, compared with no tolerance.** A group of `G = 2^j` logits of 0 and the rest
  -1000, in shuffled positions. With `exp(0) = 1` and `exp(-1000) = 0` every probability is
  `2^-j` or 0 and every partial sum is exact, in any order of addition. A correctly rounded
  `exp` gives exactly those two values; the vendors' bounds of a few ulp (ch10-facts A11, C4)
  do not promise them. They held on Metal and on the emulators when this lab was built, and are
  unverified on CUDA and HIP.
  With `p = (t + 1) / G` the inclusive sum at token t is exactly p: token t must be kept and
  token t + 1 dropped. t is placed at token 0, inside a lane's run of 4 (token 129), at the
  last token of the first chunk of 1,024 (token 1,023), at the first token of the second chunk
  and inside it. `p = 0` and `1e-7` keep one token; `p = 1` and `1.5` keep every token, the
  -1000 tail included.
- **Min-p at its threshold.** A row whose largest logit is 0, with one logit planted at
  exactly `logf(minP)` (0.05 and 0.1) as the host computes it, which must be kept, and the
  float just below it, which must not; `p = 1`. Adding 0 is exact under any rounding mode or
  reassociation, so these counts are exact on every backend. `minP = 1` keeps only the logits
  tied with the largest.
- **Softmax rows**: logits uniform in [-12, 12], peaked (one at 12, the rest in [-12, 0]) and
  nearly flat (in [-1, 1]), at p = 0.5, 0.9 and 0.95 and min-p = 0, 0.05 and 0.1. Long rows
  hold ties (the generator has 65,536 levels). `sortedProbs` must be within the band below
  of the reference; `n` must equal the reference's wherever the reference's mass before each
  token and each logit's distance from the min-p threshold lie outside the band, and either
  neighbour is accepted inside it.
- Always: `sortedPos` is a permutation in the stable order and `sortedLogits` holds the
  candidates' bits; `1 <= n <= K`; part B's `n` equals part A's wherever the band does not
  allow a choice; outputs start as NaN or -1 so that anything never written shows; 64 guard
  entries past every output and past the workspace are untouched; inputs unchanged.
- The provided labs 14.3, 10.3 and 11.6 are tested on their own too, so that a broken
  provided file shows up as that and not as a failure of yours.

## The tolerance band, and why

The reference is computed in double from the float logits: the probabilities, and the mass
before each token as their exact running sum. Probabilities are compared with
`|got - ref| <= 1e-4 * ref`, the mass (through the top-p decision) with the same relative band:
lab 11.6's band, re-derived here for the probabilities a GPU computes itself.

Worst case at the largest row tested, 151,936 logits, 256 threads, exponent arguments in
[-24, 0], every term positive (so each rounding is at most `2^-24` of a partial sum that is at
most the final one):

- `exp`: CUDA's `expf` is within 2 ulp (`ch10-facts.md` A11); Metal's fast `exp` within
  `3 + floor(|2x|)` ulp (ch10-facts C4), 51 ulp or about `6.1e-6` relative at x = -24. HIP
  publishes measured differences, not bounds (ch10-facts B4): unverified beyond them.
- The argument `l - l_0` is rounded once: up to `24 * 2^-24`, about `1.4e-6` relative in `exp`.
- The sum: each thread adds up to 594 terms in turn (593 roundings; the first, 0 + x, is
  exact), then two tree stages. Apple does not document the order of `simd_sum`, so allow 63
  roundings for each stage: 719 in all, `4.3e-5`.
- One reciprocal and one multiplication: `1.2e-7`.

So a probability is within about `5.1e-5` relative with Metal's fast `exp` (`3.7e-5` with
CUDA's tree of 5 + 3 levels and `expf`). The mass before a token is a sum of such
probabilities, all positive, so it carries the same relative error, plus the scan's own
roundings: 3 in a lane's run, 63 and 63 for the two scan stages (again any order on Metal),
148 for the carry over 149 chunks of 1,024, and 3 that combine the parts: 280 roundings,
`1.7e-5`. In all about `6.8e-5`, inside `1e-4`. The composed chain has the same bound: lab
10.3's softmax and lab 11.6's scan add in the same shapes. A wrong denominator, an inclusive
scan or a missing carry miss by whole probabilities.

Min-p has a band of its own on softmax rows: the GPU adds the same two floats as the host
(`l_0 + lnMinP`), but a compiler allowed to reassociate may compare `l - l_0` with `lnMinP`
instead, which can differ in the last bit. Logits within `1e-6 * (|l_0| + |lnMinP|)` of the
threshold may go either way. Where `l_0` is 0 (the exact and planted rows) there is no band.

Why not compare `n` exactly everywhere: a float scan's result depends on its addition tree,
which differs between your kernel, lab 11.6's and the reference (ch11-facts A8, D4), and
Metal does not even promise round-to-nearest (C5). Inside the band either answer is a correct
sampler.

## Before you benchmark

Two predictions (the labs page asks for them as `x1/labs/p-lab2-fused` and
`x1/labs/p-lab2-sort`):

1. **At 64 rows of 151,936, how many times faster will the fused kernel be than the composed
   chain without the sort?** Count the bytes each design must move per token first: the
   composed chain's softmax reads the logits three times and writes the probabilities, lab
   11.6 reads them and writes the mass and the keep byte, min-p reads the logits again; the
   fused kernel reads the logits twice and writes the probabilities once. Then decide whether
   both run at the same bandwidth.
2. **What share of the composed chain's time, sort included, will the sort take at K =
   151,936, and at K = 40?** Section X1.5's explorer is the place to reason about it: how much
   work does lab 14.3 do per row, whatever K is, and how much do the filters do?

The benchmark runs rows 1 and 64 by K = 40, 1,024, 32,000 and 151,936 on softmax-shaped rows at
llama.cpp's default top-p 0.95 and min-p 0.05 (D1). It times lab 14.3's sort alone, your whole
`launchFilterRows` (sort included), the composed chain after the sort (softmax, top-p mask,
your min-p count, on the same sorted logits), and your fused kernel on those sorted logits, not
timed with the sort. Time per call is the best of 5 timed runs, each repeating the call until
it lasts at least 2 ms. It prints milliseconds per call, GB/s at the minimum bytes each filter
design must move (29 bytes per token composed, 12 fused), the sort's share of the composed
chain, and the fused kernel's speed-up over the composed chain after the sort. Every timed
result is checked before it is reported. The numbers are measurements of your machine, not
targets.

A shape of 1 row is at most 600 KB per buffer, and a timed run repeats the call on the same
buffers; read the 64-row shapes for bandwidth.

## Portability

- HIP: the typed lane mask and `warpSize` let `kernel.cu` build through `pmpp/vendors/hip.h`;
  with 64-wide warps a block of 256 has 4 warps. Not compiled or run on AMD hardware here:
  unverified.
- Metal: `launch.mm` reads the SIMD-group width from the pipeline (`threadExecutionWidth`)
  and refuses 256 threads that are not whole SIMD-groups, or more SIMD-groups than a
  SIMD-group has lanes. Metal's default math mode allows reassociation and its rounding mode
  may be round-toward-zero (`ch05-facts.md` C7, x1-facts C5): the band above allows any order
  inside a SIMD-group, and the exact rows are exact in any order and either rounding mode.
- The "keep everything" min-p threshold is `-FLT_MAX`, not `-INFINITY`: under Metal's fast
  math the behaviour of infinity is undefined (`ch10-facts.md` C3), and `-FLT_MAX` is below
  every finite logit on every backend.
- NaN and infinite logits are outside the contract.

## Where this goes

This is milestone M5's top-p and min-p stage. Lab X1.3 composes the same stage from its own
copies inside the whole sampler; going further, swap this lab's fused kernel in for the
composed stage there and measure the difference on a whole sampling step. Lab X1.4 hunts three
defects in the fused kernel you have just written.
