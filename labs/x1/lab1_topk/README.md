# Lab X1.1: top-k by radix select, measured against a full sort (project milestone M5)

Not in the book: sampling is ours (PLAN.md, module X1). The parts are the book's: chapter 9's
shared-memory histogram with atomics (Fig. 9.10, pp. 213-214), the most-significant-digit idea of
section 14.9 (p. 346), the warp vote and popcount rank of section 12.3 (Fig. 12.3, pp. 292-293)
inside section 12.5's stable filter (pp. 297-298), and chapter 11's warp scan (Fig. 11.8, p. 264).
Section X1.4 of the lesson explains the algorithm and steps through it. A bare item such as (A3)
or (D2) below is in `notes/today/ch14-facts.md`; others name their file.

Write one host function and the two kernels behind it:

```cpp
void launchTopKRows(const float* logits, int* outIdx, float* outVal, unsigned int* scratch,
                    int rows, int cols, int k);
```

For every row of `logits`, a `[rows x cols]` float32 tensor (row-major, batch first), write the
k largest logits and their columns, without sorting the row. Lab X1.3, the whole sampler, calls a
copy of the reference solution of this lab as its top-k stage.

- Edit `work/kernel.cu` (CUDA; also built for HIP and the emulators): `topkThreshold`,
  `topkGather`, `launchTopKRows`. Edit `work/kernel.metal` (`topk_threshold`, `topk_gather`) and
  `work/launch.mm` (`encodeTopKRows`, which encodes both into the one compute encoder it is given).
- Provided in those files: lab 12.4's warp helpers (the typed lane mask, a popcount and a
  find-first-set that match it, `laneIdx`, `warpIdx`), and lab 14.3's key (`orderedKey`).
- Provided and not edited: lab 14.3's sort, the benchmark's baseline. On CUDA, HIP and the
  emulators it is `labs/x1/shared-g11/sortrows_g11.cu`, a copy of lab 14.3's
  `solution/kernel.cu` whose only change is that `laneIdx` and `warpIdx` are static (so that your
  file may define them too); `app/src/chapters/x1/copies-g11.test.ts` checks that nothing else
  differs. On Metal, lab 14.3's own `solution/kernel.metal` and `launch.mm` are used.
- Test: `python3 labs/pmpp.py run x1/lab1`. Start with `--backend emu` (about three minutes).
- Races: `python3 labs/pmpp.py run x1/lab1 --backend race`.
- Measure: `python3 labs/pmpp.py run x1/lab1 --bench`, after the prediction below.

## The contract

- `rows >= 1`, `cols >= 1`, `k >= 1`, any sizes. Every logit finite; `-0.0` and `+0.0` allowed.
- Let `kk = min(k, cols)`. Row `r`'s output is `outIdx[r * k + j]` and `outVal[r * k + j]` for
  `j < kk`: the columns of the `kk` largest logits of the row, in **ascending column order**, and
  their logits. The row stride of the output is `k`, not `kk`.
- "Largest" is the order of lab 14.3's key: take the 32 bits `u` of the float, make `-0.0`
  (`0x80000000`) into `+0.0`, then set the sign bit of a non-negative value and invert every bit
  of a negative one. Read as unsigned integers, the keys are in the order of the logits, with
  the two zeros equal (CUB documents the transform, A3).
- Ties: when several columns hold the `kk`-th value, the lowest columns are kept. This is what
  llama.cpp's CUDA radix select and Metal `kernel_top_k` do (D2); Hugging Face transformers keeps
  every tied token instead and can return more than k (x1-facts D4).
- `outVal` holds the 32 bits of the logit, copied. Slots `kk .. k-1` of a row, anything past
  `rows * k`, past `4 x rows` entries of `scratch`, and the logits themselves stay untouched.
- The kernels never perform a float operation: the launcher hands them the logits as unsigned
  integers, as lab 14.3 does, so no math mode can rewrite the `-0.0` test (C3).

## The design (ours)

One block of 256 threads per row, two kernels.

1. `topkThreshold`, up to four passes inside one launch, one per byte of the key, most
   significant first. Each pass zeroes a 256-bin histogram in shared memory, walks the row
   (thread `t` takes columns `t, t + 256, ...`) and, for every key whose bytes decided so far
   equal the prefix found so far, atomically counts its current byte. Then it walks the bins
   from 255 down until the running count reaches the keys still needed: one warp does this,
   `warpSize` bins at a time, with a warp scan and a ballot. Keys in higher bins are in, keys in
   lower bins are out, the crossing bin's byte joins the prefix, and what is still needed drops by
   the keys above it. After four passes the prefix is `K*`, the `kk`-th largest key. When every
   key of the crossing bin is needed earlier, the set is already decided and the kernel stops
   (section X1.4's stepper shows this): `K*` is then the prefix with its undecided bits zero, the
   smallest key that bin can hold, and the budget is the whole bin, so the gather keeps all of
   it. Either way it leaves in `scratch` `K*` and how many keys equal to `K*` are still needed
   (the tie budget).
2. `topkGather`, one pass in rounds of 256 columns. A key is above (`key > K*`), a tie
   (`key == K*`), or out. A tie is kept while the number of ties before it in
   the row, in column order, is below the budget; a kept key's slot is the number of kept keys
   before it. Both ranks come from a ballot and a popcount of the lanes below inside each warp,
   warp totals in shared memory, and running counts carried from round to round: lab 12.4's
   stable filter, with a tie budget.

The output is in column order because ranks are in column order: no sort anywhere.

## What the tests check

- Shapes: rows 1, 4 and 7 by cols 1, 2, 31, 255, 256, 257, 1000, 4096, 32000, 50257 and 151936
  (GPT-2's and Qwen3's vocabulary sizes, ch09-facts D7), and for each, k = 1, 2, 40, 256, 1000,
  cols and cols + 5 (that is, larger than the row) where k <= cols + 5. On `emu`, rows 1 and 3 by
  cols up to 4099, k = 1, 40, 256 (from 257 columns), 1000 (4099 only) and cols + 5, one batch per
  row count per case with the data rotating, because every case runs under three thread orders.
  On `race`, rows 2 by cols 1, 33, 257, 1000 and 2100 with the same k, every kind of data.
- Data, mixed within each batch so that a kernel reading or writing the wrong row is caught:
  - logits uniform in [-12, 12] (65,536 levels, so with many ties);
  - peaked softmax-shaped rows (positive probabilities);
  - every logit equal (the answer is columns 0 .. kk-1);
  - exact ties at the k-th value planted at columns 31 and 32, 63 and 64, 255 and 256, 0 and
    cols-1, and runs of 40 equal values across a warp seam (12..51), the first round seam
    (236..275) and a later one (2028..2067), with a tie budget of 1, half or all but one of them;
  - `-0.0` and `+0.0` interleaved every third column with the k-th value a zero;
  - keys sharing their top 8, 16 and 24 bits (which force 2, 3 and 4 passes; the last has only
    256 distinct values);
  - keys whose bytes are all 0x00 or 0xFF, and magnitudes near `FLT_MAX` of both signs (so the
    crossing bin is at either end of the walk);
  - negative logits only.
- Exact comparison, no tolerance: `outIdx` equals the reference; `outVal` is bit-identical to the
  logits at those columns; slots past `kk` keep the sentinel the driver wrote (-1 and a NaN bit
  pattern); 64 guard entries past `outIdx`, `outVal` and `scratch` are untouched; the logits are
  unchanged.
- The reference, in `main.cpp`: `std::stable_sort` of the column indices by key, largest first,
  then the first `kk`, then sorted by column.
- The provided sort baseline is checked too, on a few shapes: its first k of each row, put in
  column order, must be the reference.

## Before you benchmark

Write down a prediction (the lesson asks for it as `x1/labs/p-lab1-sort`): at 64 rows of 151,936
logits and k = 40, how many times faster will your select be than lab 14.3's full sort?

Section X1.4 counts the bytes each design must move per key: the sort's four passes each read the
4-byte key to count it and read and write the 8-byte (key, index) pair to scatter it, 80 bytes;
the select reads the key four times to count and once to gather, 20 bytes, plus 8 bytes for each
kept pair. That is about 4 times less traffic. Say whether you expect the time ratio above or
below 4, and why. Three things to weigh: the sort's scatter goes to 256 buckets, so its writes are
poorly coalesced (lab 14.3's README); the select's atomics in later passes all land on the few
bins near the crossing; and both designs run one block per row, so at 64 rows neither fills a
large device, and at 1 row both use one multiprocessor (chapter 10, section 10.10).

The benchmark runs rows 1, 8 and 64 by cols 32,000 and 151,936 by k = 1, 40 and 1,024, on logits
uniform in [-12, 12] with 2^24 levels. It reports, per call:

- your select (both kernels), timed on the device, best of 5 runs, each repeating the call until
  it lasts at least 2 ms;
- lab 14.3's whole sort (keys, four passes, gather), best of 5 runs of at least 20 ms (lab 14.3
  found shorter runs vary); the top k are the first k of each sorted row, so taking them is free;
- one CPU thread running `std::partial_sort` of (id, logit) pairs per row, as llama.cpp's host
  top-k does for k up to 128 (D3), timed on the host with a steady clock;
- rows per second, the bytes per key above, and the ratios sort/select and CPU/select.

Every timed result is checked before it is reported (`std::partial_sort` is not stable, so its
result is checked by value), and a shape larger than a quarter of the device memory reported is
skipped. The numbers are measurements of your machine, not targets.

## Portability

- HIP: `kernel.cu` builds through `pmpp/vendors/hip.h`. The vote is 64 bits on every AMD target
  and `__ballot_sync` and `__shfl_*_sync` take a 64-bit mask (infra-facts B3), so the lane mask
  has its own type and the popcount and find-first-set match it (`__popcll`, `__ffsll`); `__popc`
  on a 64-bit vote would silently count only the low 32 lanes (ch12-facts B1). With 64-wide
  warps the bin walk takes four steps of 64 bins instead of eight of 32. Not compiled or run on
  AMD hardware here: unverified.
- Metal: the histogram is `threadgroup atomic_uint`, read with `atomic_load_explicit`
  (ch09-facts C1). `simd_ballot` returns a 64-bit vote whose bits above the SIMD-group width are
  undefined (ch12-facts C2): the rank `popcount(vote & lanes below)` never reads them, the count
  comes from `simd_sum`, and the first set lane is found after masking the vote to the lanes that
  exist. Threads take their columns from `sg * width + lane`, because the scans run in SIMD-group
  order and how a threadgroup splits into SIMD-groups is the implementation's (ch04-facts C2).
  `launch.mm` reads the width from the pipeline (`threadExecutionWidth`) and refuses to dispatch
  unless 256 threads make whole SIMD-groups and the width divides the 256 bins.
- The library route: `cub::DeviceTopK` (unordered output, ties unspecified) and
  `DeviceBatchedTopK` on CUDA; `rocprim::topk` on HIP with no hipCUB wrapper; MPSGraph `topK` on
  Metal with no stated tie order (A7, B4, C1). None of them promises this lab's column order and
  lowest-column ties, which is why M5 has its own.
- The CUDA front-end check needs the shim include directory that `labs/pmpp.py` passes
  (`-I labs/common/include/pmpp/emu/shim`) for `<cuda/atomic>`.
