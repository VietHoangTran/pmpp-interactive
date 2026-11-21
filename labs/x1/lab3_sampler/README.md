# Lab X1.3: milestone M5, a batched GPU sampler

Not in the book: sampling is ours (PLAN.md, module X1). Every stage but three is an earlier lab's
kernel: lab 9.3's penalties, lab 10.3's softmax and argmax, lab X1.1's top-k select, lab 14.3's
sort, lab 11.6's top-p mask. You write what none of them has: a counter-based random number per
row, the repetition penalty, the final draw, and the composition. A bare item such as (A4) or
(E2) below is in `notes/today/x1-facts.md`; others name their file.

```cpp
size_t sampleWorkspaceBytes(int rows, int vocab, int topK);
void launchSampleRows(const float* logits, const unsigned int* counts,
                      const unsigned long long* seeds, const unsigned long long* streams,
                      const unsigned long long* steps, int* tokens, float* uOut, void* workspace,
                      int rows, int vocab, int topK, float topP, float minP, float temperature,
                      float repeatPenalty, float frequencyPenalty, float presencePenalty);
```

`[rows x vocab]` logits and token counts in, one token id per row out. Each row brings its own
seed, stream and step, and the token is a function of those and of the data: run it again and you
get the same token, on any backend, for every draw that is not within rounding of a boundary
(below). Milestone M10 (lab X3.2) calls it on a GPU when this lab's solution is present.

- Edit `work/kernel.cu` (CUDA; also built for HIP and the emulators): `philox4x32_10`,
  `philoxKatKernel`, `uniformFromBits`, `applyRepetition`, `drawRows`, `sampleWorkspaceBytes`,
  `launchSampleRows`. The launchers `launchPhiloxKat`, `launchApplyRepetition`, `launchDrawRows`
  and `launchDrawRowsU` (the test entry: u from a buffer) are written for you.
- Edit `work/kernel.metal` (`philox4x32_10`, `philox_kat`, `uniform_from_bits`,
  `apply_repetition`, `draw_rows`) and `work/launch.mm` (`encodePhiloxKat`,
  `encodeApplyRepetition`, `encodeDrawRows`, `encodeSampleRows`).
- Provided in those files: the workspace copy (`copyRows`, `copy_rows`, `encodeCopyRows`) and
  the warp helpers.
- Provided and not edited, in `labs/x1/shared-g11/`: copies of lab 9.3 (`penalties_g11.cu`),
  lab 10.3 (`softmax_g11.cu`), lab 11.6 (`topp_g11.cu`), lab 14.3 (`sortrows_g11.cu`) and lab
  X1.1 (`topk_g11.cu`, `.metal`, `_launch.mm`), each with a distinct file stem and its helper
  functions static so they can be linked together; `app/src/chapters/x1/copies-g11.test.ts`
  checks that nothing else differs from the originals. And `minp_g11.cu`, `.metal`, `_launch.mm`:
  min-p as a count on a sorted row, and how many candidates survive (below). On Metal the earlier
  chapters' own `solution/kernel.metal` files are loaded by path and their `launch.mm` files
  linked in; `sampler_metal.h` declares their encoders, the pipelines and the workspace.
- Test: `python3 labs/pmpp.py run x1/lab3`. Start with `--backend emu` (several minutes).
- Races: `python3 labs/pmpp.py run x1/lab3 --backend race`.
- Measure: `python3 labs/pmpp.py run x1/lab3 --bench`, after the predictions below.

## The chain

llama.cpp's default order (D1): penalties, top-k, top-p, min-p, then temperature, then the draw.
Truncation happens before temperature; Hugging Face transformers does it the other way round
(D4), and mlx-lm runs top-p before top-k (D8). M5 follows llama.cpp.

| Stage | Kernel | Writes, per row |
|---|---|---|
| copy | `copyRows` (provided) | the logits, `vocab` floats: the caller's are read-only |
| repetition | yours | seen tokens: a logit `<= 0` times the penalty, a positive one divided by it |
| frequency, presence | lab 9.3 `launchApplyPenalties` | `logit -= count * f + (count > 0) * p` |
| greedy (`temperature <= 0`) | lab 10.3 `launchArgmaxRows` | the token: the largest logit, lowest id on ties; then stop |
| top-k | lab X1.1 `launchTopKRows` | `K` (column, logit) pairs, column order, ties to the lowest columns |
| sort | lab 14.3 `launchSortRowsDesc` | the `K` logits largest first, stable, and their positions |
| softmax | lab 10.3 `launchSoftmaxRows`, scale 1 | `K` probabilities |
| top-p | lab 11.6 `launchTopPMask` | `kept`: keep position `i` when `i == 0`, `p >= 1`, or the mass before `i` is below `p` |
| min-p, combine | `launchMinPCombine` (provided) | `n = max(1, min(kept, countMinP))`, where `countMinP` counts the sorted logits at least the first plus `ln(minP)`; `minP <= 0` disables it |
| draw | yours | the token |

`K` is `topK`, or `vocab` when `topK <= 0` or `topK >= vocab` (llama.cpp: "<= 0 to use vocab
size", D1): the slow path, a whole-row select and sort. The workspace holds every intermediate:
the penalised logits (`rows x vocab` floats), the select's columns and logits and its scratch
(`4 x rows` uints), the sorted logits and their positions, the sort's scratch (`4 x rows x K`
uints), the probabilities, lab 11.6's mass and keep flags (`rows x K` bytes) and counts, and
`n`. `sampleWorkspaceBytes` says how many bytes that is; on Metal, where the earlier labs'
encoders bind every buffer at offset 0, `sampler_metal.h` makes one buffer per intermediate
instead.

Ties go to the lowest token id at every stage: the select keeps the lowest columns and hands
them over in column order, lab 14.3's sort is stable, argmax prefers the lowest index.

## The pieces you write

**Philox4x32-10** (A4): ten rounds on a 128-bit counter and a 64-bit key. A round multiplies
counter word 0 by `0xD2511F53` and word 2 by `0xCD9E8D57`, each a full 32 x 32 to 64-bit
product, and forms `(hi1 ^ w1 ^ k0, lo1, hi0 ^ w3 ^ k1, lo0)`; between rounds the key grows by
`0x9E3779B9` and `0xBB67AE85`. Write the product as an `unsigned long long` (Metal: `ulong`)
multiplication: `__umulhi` exists on CUDA and HIP (A5, B3), and the course's emulator and
front-end stub accept it too, but the product form needs no intrinsic and is the same source on
Metal. Random123's code is distributed under a BSD-style notice
(Copyright 2010-2011, D. E. Shaw Research), reproduced in the solution; cuRAND's headers are under
NVIDIA's licence and are not used. MSL has no random number function at all (C1).

**M5's mapping** (ours): counter = (step low, step high, stream low, stream high), key = (seed
low, seed high), word 0. By our reading of cuRAND's header that is the first `curand()` after
`curand_init(seed, stream, 4 * step, &s)` (A3, unverified on a device).

**u** = `(w >> 8) * 2^-24`. An integer below 2^24 converts to float exactly and a power of two
scales exactly, so u is the same float on every backend under any math mode, in
`[0, 1 - 2^-24]`, never 1 (E2). cuRAND's and rocRAND's own conversions differ from each other
for about 0.4 % of inputs (B2), which is one reason M5 converts the bits itself.

**Repetition** (D2): for a token with count > 0, a logit `<= 0` is multiplied by the penalty and
a positive one divided. llama.cpp's comment: "The academic publication that described this
technique actually just only divided, but that would cause tokens with negative logits to become
more likely, which is obviously wrong."

**The draw**, one block per row as FlashInfer's sampling kernel does (A6): weights
`w_i = exp((l_i - l_0) * invT)` over the `n` survivors (`l_0` the largest, `invT = 1 / T` computed
once on the host); the total by a block reduction; the target `u * total`; then the row in chunks
of `blockDim.x * 4` weights, four consecutive ones per thread, with the sum of earlier chunks
carried along. A chunk whose total cannot reach past the target is skipped after its scan. In the
chunk that does, the winner is the first position with a positive weight whose inclusive sum is
strictly greater than the target: the first such lane of a warp by ballot, the lowest warp that
has one. If rounding leaves no such position, the last positive weight wins (FlashInfer has the
same fallback, A6). With u in `[0, 1)` and "strictly greater", a zero weight can never win (A2).

## Which draws must match

The CPU reference in `main.cpp` samples every row in double precision, after penalties applied in
float in llama.cpp's order (so an unseen token keeps its logit bit for bit). A device computes the
same chain in float, and Metal's default math mode may also contract, reassociate and round toward
zero (C5, ch05-facts C7), so a device can land on the other side of a boundary that the CPU
passes closely. The reference therefore records, for every decision, how far it is from flipping,
and the test compares tokens only for **decisive** draws. The bands below are ours; they are
deliberately generous, so that a correct kernel never fails, and a draw inside one is counted and
printed, never failed. More than half the draws of every case must be decisive, or the test
reports a problem with its own data.

- A penalised logit: an unseen token's value is exact everywhere. A seen token's went through a
  multiplication or division and a subtraction that a device may round differently or fuse:
  band `e = 8 ulp` of the largest magnitude involved.
- Top-k: undecided if some kept token minus its band is at or below some dropped token plus its
  band (one of them penalised). Equal unpenalised values are not undecided: the lowest id wins on
  every backend.
- Top-p and min-p: each is a prefix count. A position whose mass before it is within
  `2^-24 x (512 + K / 256)` of `p` (float softmax, a block-reduced total, lab 11.6's chunked scan
  with a carried prefix), or whose logit is within `2 ulp + e` of the min-p threshold, could go
  either way; the draw is undecided only if that can change `n`.
- Order: two adjacent survivors whose bands overlap could be sorted the other way; a draw that
  lands on either is undecided.
- The draw: undecided if `u x total` is within `total x (2^-14 + n x 2^-22) + 2 invT (e_0 + max e)`
  of a boundary of the cumulative weights. The relative part covers float `exp` and the chunked
  scan; the rest covers penalised logits.

The tests print how many draws fell inside a band, with the reason for the first one.

## What the tests check

1. **Philox known answers**, on the device: Random123's three 10-round vectors (A4): counter and
   key all zero give `6627e8d5 e169c58d bc57ac4c 9b00dbd8`; all ones give
   `408f276d 41c83b0e a20bc7c6 6d5451fd`; the digits of pi give
   `d16cfe09 94fdcceb 5001e420 24126ea1`. The driver checks its own host Philox against them too.
2. **u bit for bit** for 4,096 (seed, stream, step) triples through `drawRows` with one candidate
   per row (64 on `emu`, 32 on `race`), starting with seed 0, stream 0, step 0, whose word 0 must be
   `0x6627e8d5`. Seeds, streams and steps above 2^32 are included.
3. **The draw alone** through `launchDrawRowsU` with planted u: equal weights (`exp(0) = 1`, so
   every sum is an exact integer) with `u = m x 2^-24` and `u x n` exact, which must give position
   `floor(u x n)`, including the seams of the first chunks (positions 1023 and 1024 of a
   2,048-weight row, 2047 and 2048 of 4,096); zero weights (`exp(-1e4)` underflows to 0) that must
   never win, at u = 0, at a target equal to a sum, and at `u = 1 - 2^-24`; one survivor; and
   `u = 1`, which cannot come from the conversion but puts the target at the total, as rounding
   can, so that the fallback must pick the last positive weight. Candidates past `n` hold a huge
   weight that must never be read, and the positions are mapped to ids through a permutation.
4. **Repetition** on 2 rows of 1,000 logits (300 on `emu`) of both signs, exact zeros of both signs and counts 0
   to 3: penalty 1.1 and 2.0 within 2 ulp of llama.cpp's rule (a zero stays zero), penalty 1 and
   unseen tokens bit-identical.
5. **The whole sampler** against the CPU reference: rows 1, 3 and 8 by vocab 1,000, 4,099, 32,000
   and 151,936; settings llama.cpp's defaults (top-k 40, top-p 0.95, min-p 0.05, T 0.8) with
   penalties 1.1, 0.1, 0.1, and then each of top-k 1, top-k 0 (whole row), top-p 1, min-p 0,
   T 0 (greedy, lowest id), T 2, and no penalties; 64 seeds x 4 steps per case, one stream per
   row. The logits have a planted tie of two unseen head tokens, zeros of both signs among seen
   tokens, and in odd rows two unseen tokens tied at the maximum. Decisive draws must match
   exactly; u must match bit for bit (and is not written when T <= 0); every token must be a
   vocabulary id; 64 guard entries past `tokens` and `uOut` and (CUDA, HIP, emulators) past
   `sampleWorkspaceBytes` are untouched; the logits and counts are unchanged. On `emu`: rows 1
   and 3 by vocab 1,000 and 4,099, top-k at most 40 (no whole-row case), every fourth (shape,
   setting), 1 seed x 1 step. On `race`: rows 2 by vocab 300 and 1,000, every setting, 2 seeds x
   2 steps. The toy loop below uses a vocabulary of 1,000 (300 on `emu`).
6. **A toy decode loop**: 4 rows, 16 steps (on `emu` 2 rows, 4 steps), logits from a fixed table, counts kept on the device
   by lab 9.3's `appendTokens` after every step, at llama.cpp's defaults with penalties. The CPU
   first finds a seed whose draws are all decisive along its own trajectory; your token
   sequences must equal it.
7. **Tokens digests**: FNV-1a hashes of the decisive draws of a fixed case at the defaults, rows
   of 1,000. On a GPU the full case is 8 rows x 16 seeds x 8 steps (1,024 draws), written to the
   run's JSON as `tokens_digest` with `decisive_draws` and `draws`. Its core, the first 2 rows x
   2 seeds x 2 steps (8 draws), is hashed on its own as `core_digest` with `core_draws`; the
   emulators, too slow for the full case, run and write the core only. The same digest on two
   backends means the same tokens on those draws; the core compares every backend, the full
   digest the GPUs.

## Before you benchmark

Write down two predictions (the lesson asks for them as `x1/labs/p-lab3-cpu` and
`x1/labs/p-lab3-batch`):

1. At batch 1 and 151,936 tokens with llama.cpp's defaults, what is the GPU's time per sampling
   step divided by one CPU thread's? Below 1 means the GPU wins.
2. How many times longer will a step take at batch 64 than at batch 1?

Reason from one block per row (chapter 10, section 10.10: at batch 1 one multiprocessor works
and the rest wait) and from which stages read the whole vocabulary: the copy and the penalties
read and write every logit, the select reads the row up to five times; everything after top-k
touches 40 values per row.

The benchmark runs rows 1, 8 and 64 by vocab 32,000 and 151,936 at llama.cpp's defaults with the
penalties on, and the whole-row path (top-k 0) at rows 1 and 8 by 151,936. It reports:

- ms per sampling step for `launchSampleRows` (on Metal `encodeSampleRows`) and tokens per second
  (rows divided by that time);
- each stage timed alone: penalties (copy, repetition, frequency and presence), top-k select,
  sort of k, softmax, top-p, min-p and combine, the draw;
- one CPU thread running the same chain in float (llama.cpp samples on the host by default, D1;
  top-k by `std::partial_sort`, ch14-facts D3), and the ratio GPU / CPU.

Each time is the best of 5 runs, each repeating the call until it lasts at least 2 ms; every
timed result is checked against the CPU reference first (decisive draws), and a shape larger than
a quarter of the device memory reported is skipped. Metal times are the command buffer's GPU time
(`GPUEndTime - GPUStartTime`); CUDA and HIP times are host wall-clock time around the launches and
a synchronise; the CPU's is a steady clock on the host. The numbers are measurements of your
machine, not targets.

## Portability

- HIP: `kernel.cu` is written to build through `pmpp/vendors/hip.h`. The lane mask is 64 bits on every AMD
  target (infra-facts B3), and the find-first-set matches it (`__ffsll`). The 64-bit product form
  of Philox needs no intrinsic. hipRAND exists with cuRAND's names, but its `hiprand_uniform` on
  AMD converts bits differently from `curand_uniform` (B2): with M5's own Philox and conversion
  the u is the same text on every backend, and the known-answer test checks it. Not compiled or
  run on AMD hardware here: unverified.
- Metal: Philox in MSL with `ulong` products (Metal 2.2 and later, ch14-facts C3); `mulhi` would
  give the high half too (C4). `simd_ballot` returns a 64-bit vote whose bits above the SIMD-group
  width are undefined (ch12-facts C2), so the draw masks it to the lanes present before `ctz`.
  Positions come from `sg * width + lane` (ch04-facts C2). `launch.mm` binds a stand-in buffer
  where `uIn` or `uOut` is absent and passes flags instead; the workspace is one buffer per
  intermediate (above), so the CUDA-side guard past the workspace has no Metal counterpart.
- The CUDA front-end check needs the shim include directory that `labs/pmpp.py` passes
  (`-I labs/common/include/pmpp/emu/shim`) for the copies that include `<cuda/atomic>`.
