# Lab 11.6: top-p keep mask (project milestone M5, the top-p piece)

Book: the exclusive scan (section 11.1, p. 253) computed with lab 11.3's row scan, which is
the coarsened, register-tiled scan of Figs. 11.12 and 11.13 (pp. 268 and 271) on the warp and
block scans of Figs. 11.8 and 11.9 (p. 264). Top-p sampling is not in the book: the rule, the
mask and the count are ours. A bare item such as (D2) below is in
`notes/today/ch11-facts.md`.

Top-p (nucleus) sampling keeps the smallest set of most likely tokens whose probability
reaches p, and samples from those. After the probabilities are sorted, that set is a prefix
of each row, and finding it is a scan. Write one row-wise kernel:

`launchTopPMask(probs, massBefore, keep, kept, rows, cols, p)`

- `probs`: `[rows x cols]` float32, row-major, batch first. Each row is a softmax output
  sorted in descending order and summing to 1 within rounding.
- `massBefore`: `[rows x cols]` float32. The mass ranked strictly before each token: an
  exclusive scan of the row.
- `keep`: `[rows x cols]` uint8, 1 = keep, 0 = drop.
- `kept`: `[rows]` int32, the number of ones in each row of `keep`.

Files:

- Edit `work/kernel.cu` (`TopPMaskKernel`, `launchTopPMask`; CUDA, also built for HIP and the
  emulators) and `work/kernel.metal` (`top_p_mask`) + `work/launch.mm` (`encodeTopPMask`). The
  names are fixed.
- Provided, not edited, in `labs/ch11/shared-g12/`: `topp_naive.cu` and `.metal` (one thread
  per row, Fig. 11.1's loop: the baseline), `topp_copy.cu` and `.metal` (a copy with the same
  9 bytes per token: the yardstick) and `topp_launch.mm`.
- Test: `python3 labs/pmpp.py run 11.6`. Start with `--backend emu`.
- Races: `python3 labs/pmpp.py run 11.6 --backend race`.
- Measure: `python3 labs/pmpp.py run 11.6 --bench`.

## The contract

For every row and token i:

- `massBefore[i] = probs[0] + ... + probs[i-1]`, and `massBefore[0] = 0` exactly.
- `keep[i] = 1` exactly when `i == 0`, or `p >= 1`, or `massBefore[i] < p`.
- `kept = ` the number of ones in the row of `keep`.

Why each clause:

- **The token that crosses p is kept.** llama.cpp's host sampler adds each token to a running
  sum and stops once the sum reaches p, keeping the token that reached it; its graph sampler
  does the same with a cumulative sum and one extra token; transformers and vLLM sort in
  ascending order and drop a token when its probability plus the mass ranked below it is at
  most 1 - p (D2, D4). With rows that sum to 1 all of them keep exactly the tokens whose mass
  before them is below p (our derivation, D2 and D4). That
  is section 11.1's exclusive scan, and the off-by-one between "exclusive < p" and
  "inclusive < p" is the whole difference between keeping and dropping the crossing token.
- **Token 0 is always kept.** Its mass before is 0, so `massBefore < p` keeps it for any
  p > 0; the clause also covers p = 0. vLLM likewise keeps at least one token whatever p is
  (D4).
- **p >= 1 keeps every token.** llama.cpp returns early for `p >= 1.0f` (D2). Without that,
  the float mass before the last tokens can round to 1 or just above it, and p = 1 would drop
  tokens that a user asking for "no top-p" expects to keep.
- **Ties are broken by position**, that is, by the sort: of two tokens with equal
  probability on either side of p, the one sorted first is kept. mlx-lm instead keeps every
  token tied with the boundary token (D3). That is a stated alternative, not this lab's rule.

The keep mask need not be a prefix. Two neighbouring masses are computed along different
addition trees, so inside the tolerance band below a 0 can be followed by a 1. `kept` counts
ones, not the length of a prefix, and chapter 12's compaction does not assume one.

## Suggested design

Lab 11.3's fixed row scan (one block of 256 threads per row, chunks of 4 per thread, a carried
prefix), with the threshold and the count fused in, so the probabilities are read once:

- While each token's mass is still in a register, decide keep, write it, and count it.
- After the chunk loop, add the threads' counts with chapter 10's two-stage reduction, and let
  one thread write `kept`.
- On Metal, `simd_prefix_exclusive_sum` and `simd_prefix_inclusive_sum` do in one call what
  Fig. 11.8's loop does, and `simd_sum` adds the counts (C1, `infra-facts.md` C2). The
  threadgroup arrays must be declared in the kernel (C2).

## What the tests check

- Shapes: rows 1, 4 and 7 by cols 1, 2, 31, 32, 33, 255, 256, 257, 1000, 4096, 32000, 50257
  and 128256 (the last three are vocabulary-sized; GPT-2's is 50,257, ch09-facts D7). The
  emulator uses rows 1 and 3 and cols up to 4099; the race backend rows 2 and cols 1, 33,
  257, 1000 and 2100 (three chunks of 1024, so the carried prefix is exercised). Rows of
  different kinds are mixed in one batch, so a kernel that reads the wrong row is caught.
- **Exact boundaries.** Dyadic rows: every probability is a whole multiple of 2^-24 and the
  row sums to exactly 1, so every partial sum is exact in float, in any order of addition.
  p is 0.5 or 0.75, and each row is built so that the inclusive mass of one token is exactly
  p: at token 0 (p equal to the first probability: exactly 1 token kept), inside a lane's run
  of 4, at the end of a chunk of 1024, a third of the way in, before a zero tail, and four
  tokens of 0.25 followed by zeros (ties straddling p: p = 0.5 keeps 2). The token whose
  inclusive mass equals p must be kept and the next one dropped. These rows are compared
  exactly, which catches both "inclusive < p" and "massBefore <= p".
- **Softmax rows**, at p = 0.1, 0.5, 0.9, 0.95 and 0.99 (the emulators: 0.5 and 0.99): the
  softmax of logits uniform in [-12, 12], computed in double, rounded to float and sorted; a
  peaked row (one logit 40 above the rest); a nearly flat row; and a row whose last half is
  exactly 0. `massBefore` must lie within the band below of the reference, the keep decision
  must be right wherever the reference is farther from p than the band, and either decision
  is accepted inside it.
- **Edge values of p**: 0 and 1e-7 keep exactly token 0; 1 and 1.5 keep every token, including
  the zero tails whose mass before them is exactly 1 (the emulators: 0 and 1).
- Always: `massBefore[r][0]` is exactly 0; `keep` holds only 0 and 1; `keep[i]` agrees with the
  rule applied to the `massBefore[i]` you wrote; `kept` equals the row's count of ones. The
  outputs start as NaN, 0xAB and -1, so anything never written shows, and guard elements past
  the end of each output must be untouched.
- The provided baseline and copy are tested too, so that a broken provided file shows up here.

## The tolerance band, and why

The reference is Fig. 11.1's loop in double over the float probabilities, made exclusive: the
exact mass before each token, to about 1e-11 relative. Your `massBefore[i]` must satisfy
`|got - ref| <= 1e-4 * ref`.

The bound for the suggested design, as lab 10.3 derived its own: every probability is at least
0, so every partial sum along the way is at most `ref`, and each rounding adds at most
`2^-24` of it. A token's mass passes through at most

- 3 additions in its lane's run of 4 (the first, 0 + x, is exact);
- 63 in the SIMD-group or warp scan, and 63 in the scan of the warp totals: a Kogge-Stone
  loop needs only log2 of the width, but Apple does not document the order of
  `simd_prefix_*_sum` (C1), so we allow any order of up to 64 terms;
- 126 in the carry, one per chunk of 1024 at 128,256 tokens;
- 3 that combine the parts;

258 roundings, so at most `(1 + 2^-24)^258 - 1`, about `1.5e-5` relative. `1e-4` leaves six
times that, and covers a design whose chunks are four times smaller. A wrong index, a missing
token or an inclusive scan miss by a whole probability, which the band catches wherever that
probability is larger than `1e-4` of the mass before it. The dyadic rows have no band at all.

Why not compare bit for bit: a floating-point scan's result depends on its addition tree.
CUB, Thrust, rocPRIM and PyTorch's CUDA cumsum document that results may vary from run to run
unless a deterministic mode is asked for (A8, B2, D4), and transformers and vLLM compute the same
rule from an ascending scan compared with 1 - p, so a token within rounding of p can land on
different sides in different libraries (D4). Inside the band, either answer is a correct
sampler.

The provided baseline adds a whole row in one thread, one rounding per token, so it is held to
`1e-4 + cols * 2^-24`.

## Before you benchmark

Write down your prediction (the labs page asks for it as `ch11/labs/p-lab6-copy`): **at 64 rows
of 128,256 tokens, what share of your copy bandwidth will your kernel reach?** Two things to
weigh: the copy runs one thread per token, 32,064 blocks of 256 here; the suggested design
runs one block per row, 64 blocks, each walking 126 chunks one after another with block
barriers in each. Section 11.8 calls a memory-bound kernel that reaches more than 80% of the
bandwidth, with no redundant accesses, well optimized (p. 273).

The benchmark times the copy, the one-thread-per-row baseline and your kernel at rows x cols =
1 x 128,256, 8 x 128,256, 64 x 128,256 and 256 x 32,000, on sorted softmax rows at p = 0.9.
Time per call is the best of 5 timed runs, each repeating the call until it lasts at least
2 ms. It prints milliseconds per call, GB/s at 9 bytes per token (read 4, write 4 + 1), your
share of the copy's bandwidth, and your speed-up over the baseline. Every timed result is
checked before it is reported. The numbers are measurements of your machine, not targets.

A shape of 1 or 8 rows is a few megabytes, and a timed run repeats the call on the same
buffers many times. If the copy's GB/s there is higher than at 64 rows, those repeats are
being served from something faster than main memory, and "of copy" compares against that.
Read the share at 64 x 128,256 and 256 x 32,000 for the bandwidth question; read the small
shapes for what one block per row does at batch 1.

## Portability

- HIP: the typed lane mask and `warpSize` let `kernel.cu` build through `pmpp/vendors/hip.h`;
  with 64-wide warps a block of 256 has 4 warps. Not compiled or run on AMD hardware here:
  unverified.
- Metal: `launch.mm` reads the SIMD-group width from the pipeline (`threadExecutionWidth`)
  and refuses 256 threads that are not whole SIMD-groups, or more SIMD-groups than a
  SIMD-group has lanes. Metal's default math mode allows reassociation (ch05-facts C7): the
  band above already allows any order inside a SIMD-group, and the dyadic rows are exact in
  any order.
- NaN and infinity are outside the contract: the rows are softmax outputs.

## Where this goes

This is the top-p piece of milestone M5 (lab 9.3's token counts were the first). Chapter 12 compacts the kept tokens into a short list,
chapter 14 sorts the logits that this lab receives already sorted, and module X1 draws the
sample from the nucleus: the sausage picture of section 11.1 (p. 252), with a uniform random
number in place of the knife.
