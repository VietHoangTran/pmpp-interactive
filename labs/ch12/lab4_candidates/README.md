# Lab 12.4: candidate compaction (a piece of project milestone M5)

Book: the stable filter (section 12.5, Figs. 12.7 and 12.8, pp. 297-298) and the warp vote
with its binary prefix sum (section 12.3, Fig. 12.3, pp. 292-293). Carrying (column, value)
pairs, one block per row and the sampling use are not in the book: they are ours. A bare
item such as (C4) or (D2) below is in `notes/today/ch12-facts.md`.

A sampler that keeps only some tokens (top-k, top-p, min-p) first needs, for each row of
logits, the list of tokens that pass a threshold. Write that list. For a `[rows x cols]`
float32 logits tensor, row-major, batch first, and one threshold per row:

- `launchCompactCandidates(logits, threshold, outIdx, outVal, counts, rows, cols)` writes,
  for every row r, the pairs (column, logit) with `logit >= threshold[r]`, in ascending
  column order, to `outIdx[r * cols + 0 ..)` and `outVal[r * cols + 0 ..)`, and their
  number to `counts[r]`.

One block of 256 threads per row. The block walks its row in rounds of `blockDim.x`
columns, one column per thread per round, so each round reads one contiguous run
(coalesced). In each round a thread finds the position of its kept pair from three numbers:
the pairs kept in earlier rounds, the pairs kept by earlier warps this round, and the pairs
kept by lower lanes of its own warp. The last comes from a warp vote and the binary prefix
sum of Fig. 12.3; the middle one from a scan of the warp totals in shared memory; the first
is a running count that every thread keeps. No global atomics, and no block ever waits for
another (next section). Staging a round's pairs in shared memory before writing them, as
Fig. 12.9 does, is allowed but not required.

- Edit `work/kernel.cu` (CUDA; also built for HIP and the emulators) and
  `work/kernel.metal` + `work/launch.mm` (Metal). The names are fixed: the driver calls
  `launchCompactCandidates`, the Metal kernel `compact_candidates`, and the encoder
  `encodeCompactCandidates`.
- Provided, not edited: `provided/rowwise_serial.cu`, `rowwise_serial.metal` and
  `rowwise_serial.mm` (`launchCompactSerial`, `compact_serial`, `encodeCompactSerial`), one
  thread per row running llama.cpp's host min-p loop (D3), the benchmark baseline.
- Test: `python3 labs/pmpp.py run 12.4`. Start with `--backend emu`.
- Races: `python3 labs/pmpp.py run 12.4 --backend race` (see "The race backend" below).
- Measure: `python3 labs/pmpp.py run 12.4 --bench`.

## Why one block per row, and no atomics

- **Order.** The book's unstable filters (Figs. 12.2 to 12.6) take output slots from an
  atomic counter, so the order of the survivors depends on which thread got there first
  (ch09 A3). A sampler that walks its candidates in order, or that breaks ties by column,
  needs the same answer every run, so this lab asks for the stable filter: every kept
  pair's slot is the number of kept pairs before it (section 12.5). PyTorch's `torch.topk`
  refuses atomics for the same reason (D4).
- **Waiting.** The book's stable filter does its scan across the whole grid
  (`gridExclusiveScan`, Fig. 12.8), where a block waits for the blocks before it. CUDA
  documents when that wait is safe (A10); AMD documents no progress guarantee between
  workgroups (B6) and Apple none between threadgroups (C4). One block per row needs no
  waiting at all: the row's running count never leaves the block. It is the design
  llama.cpp's Metal top-k uses (one threadgroup per row, D2), and it fits a
  batch-by-vocabulary layout.
- **Deterministic is not stable.** llama.cpp's Metal top-k compaction is deterministic but
  writes its survivors in lane-major order, not column order (our reading of the source,
  D2). The tests here check column order.

## The contract

- `rows >= 1`, `cols >= 1`, any value: not only multiples of 32 or 256.
- Every logit and every threshold is finite. NaN and infinity are outside the tested
  contract, as in lab 10.3 (last section).
- Each row has `cols` slots in `outIdx` and `outVal`, enough for the case where every
  column passes. Slots at and past `counts[r]` are not written: the caller may have put
  something there.
- `outIdx` is `int`, `outVal` is `float`, `counts` is `unsigned int`.

## What the tests check

- Shapes: rows 1, 4 and 7 by cols 1, 2, 31, 32, 33, 255, 256, 257, 1000, 4096, 32000,
  50257 and 128256 (the last three are vocabulary-sized). The emulator uses rows 1 and 3 and
  cols up to 4099; the race backend rows 2 and cols up to 1000.
- Logits uniform in [-12, 12]. Every row of a batch has its own threshold, so a kernel that
  reads the wrong row's threshold keeps the wrong number of pairs:
  - just above the row's max (nothing kept), equal to the max (only the maxima), just below
    the min (everything kept);
  - the CPU quantiles that keep about 1, 10 and 50 percent;
  - min-p 0.1, computed on the host as llama.cpp does, `max + ln 0.1` (D3);
  - values planted exactly equal to the threshold at columns (1, 2), (31, 32), (63, 64),
    (255, 256) and (0, cols - 1): neighbouring lanes, across a 32-lane and a 64-lane warp
    boundary, across a round boundary, and at both ends. They catch `>` for `>=` and an
    off-by-one at every seam;
  - a row whose survivors are all in its last round of 256 columns, which catches a loop
    that skips the last, partial round and a count written from the wrong round.
- For every row: `counts[r]` exact; `outIdx[0 .. counts[r])` equal to the CPU list, which
  is strictly ascending; `outVal` equal bit for bit to the logits at those columns; every
  slot past `counts[r]` unchanged (index -1, and the value's bits checked on the host);
  nothing written past the end of any output.
- The provided baseline is tested too, so that a broken provided file shows up here.

The comparison is exact, with no tolerance, which this course's labs normally avoid on Metal
(its default math mode is not strict IEEE, ch05-facts C7). Here no arithmetic touches a
value: the threshold is computed on the host, the kernel compares two finite floats and
copies one. The math modes relax arithmetic and the handling of NaN, INF and signed zeros;
a comparison and a copy involve none of those (our reading of C7), so exact equality is
the right test.

## The race backend

`emu` runs one thread at a time and cannot see a race that happens to leave the right
values. `race` runs the threads of a block at once under ThreadSanitizer. The case this
lab is built for: the warp totals of one round live in the same shared array as the totals
of the next. A warp that has finished its round can write its next total while a slower
warp is still adding up this round's, so something must order the two. On a GPU that
mistake can pass every test, because whether it bites depends on scheduling. When this lab
was built, the solution with that ordering removed passed every `emu` test, and `race`
reported it.

## The milestone: how module X1 composes this

This lab writes candidates for a threshold it is given. Where the threshold comes from is
the sampler's business:

- **Min-p.** Lab 10.3's row max `m` gives the threshold `m + ln p`: a token passes when
  `exp(logit - m) >= p`, that is when its probability is at least p times the largest one.
  llama.cpp computes exactly this, `max_logit + logf(p)`, on logits, so no softmax is needed
  to filter (D3).
- **Top-k.** The threshold is the k-th largest logit, which comes from a selection
  (module X1, section X1.4: a radix select, measured against chapter 14's sort). Ties at the threshold are all kept, in column order, so
  `counts[r]` can exceed k. To keep exactly k with llama.cpp's tie rule, "the lowest columns
  win" (D1), drop the surplus pairs whose value equals the threshold, starting from the end
  of the list: the ties are in column order, so the ones dropped are the highest columns.
  Do not just cut the list at k: a pair above the threshold can come after a tie.
- **Top-p** (ours). Once the candidates are sorted by logit (chapter 14), a scan of their
  probabilities (chapter 11) finds where the cumulative mass reaches p; what is kept is a
  prefix of the sorted list.
- **Why a list and a count, not a mask.** Samplers built as compute graphs (llama.cpp's
  backend samplers, mlx-lm) keep the full row and set the losers to minus infinity (D3,
  D6). MLX says why of itself: it has "limited support for operations for which output
  shapes are dependent on input data" (D6). A hand-written kernel can return a list and a
  count, which is what this lab does.

## Before you benchmark

Write down two predictions (the lesson asks for them as `ch12/labs/p-lab4-batch1` and
`ch12/labs/p-lab4-fraction`):

1. At 1 row of 128,256 columns with 1 percent kept: the provided thread-per-row time
   divided by your block-per-row time.
2. At 64 rows: your block-per-row time at 50 percent kept divided by its time at
   1 percent kept.

Section 10.10's point applies here too: at batch 1, one block per row uses one block of a
GPU that has room for many. Ask also what changes between 1 and 50 percent: the number of
bytes written, or the number of rounds?

The benchmark times rows 1, 8 and 64 by cols 32,000 and 128,256, with thresholds from CPU
quantiles that keep 0.1, 1, 10 and 50 percent, for your kernel ("block") and the baseline
("serial"). Time per call is the best of 5 timed runs, each repeating the call until it
lasts at least 2 ms. GB/s counts the minimum traffic: 4 bytes per logit read, 8 per kept
pair written and 4 per row for the count. Rows per second is rows divided by the time per
call. Every timed result is checked before it is reported. The numbers are measurements of
your machine, not targets.

## NaN and infinity

Outside the tested contract, for the reasons lab 10.3 gives: under Metal's fast math the
behaviour of NaN and infinity is undefined (ch10-facts C3), and Apple documents the default
math mode three different ways (ch05-facts C7). On the CPU, with IEEE comparisons, `>=`
never keeps a NaN logit and a NaN threshold keeps nothing; under fast math a GPU need not
agree. Masked logits of minus infinity come back in chapter 20.

## Portability

- HIP: `kernel.cu` builds through `pmpp/vendors/hip.h`. The vote is `__ballot_sync` with a
  mask of its own type, 64 bits on HIP (infra-facts B3), and the popcount must match that
  type: `__popc` takes 32 bits, so on a 64-wide wavefront it would silently count only
  lanes 0 to 31; `__popcll` counts all 64 (B1). The lanes-below mask is
  `((lane_mask_t)1 << lane) - 1`, never `1 << lane`, which is undefined past lane 31 (A3,
  B1). llama.cpp guards the same trap by deleting the popcount for any other type (D1).
  Not compiled or run on AMD hardware here: unverified.
- Fig. 12.3's `__activemask()` inside the branch: NVIDIA documents that it does not tell
  which lanes took the branch (A1), AMD's header says it does (B1), Apple says nothing
  (C3). The vote here is taken by every lane, in uniform control flow, where the three
  agree.
- Metal: `simd_ballot` returns a 64-bit `simd_vote` whatever the SIMD-group width, and on
  hardware with fewer than 64 lanes its top bits are undefined (C2). The rank,
  `popcount(vote & lanesBelow)`, never reads them; a SIMD-group's total does, so count it
  with `simd_sum(keep ? 1u : 0u)` instead (C1, C2). `simd_prefix_exclusive_sum` ranks in one
  call (C1).
- Metal: `launch.mm` reads the SIMD-group width from the pipeline (`threadExecutionWidth`),
  checks that 256 threads make whole SIMD-groups with no more groups than lanes (ch11-facts
  C2: what a scan of the totals by one SIMD-group needs), and sizes the threadgroup array
  of totals from it, a multiple of 16 bytes (ch05-facts C8).
