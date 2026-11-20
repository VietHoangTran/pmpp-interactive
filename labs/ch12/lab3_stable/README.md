# Lab 12.3: a stable filter by scan

Book: section 12.5 (pp. 297-298, Figs. 12.7 and 12.8), section 12.6 (pp. 298-300,
Figs. 12.9 and 12.10) and exercises 1, 3 and 4 (p. 302). A bare item such as (A1) or (D4)
below is in `notes/today/ch12-facts.md`.

Write two stable, out-of-place filters of `unsigned int` keys. A key is kept when
`key < threshold`. The kept keys come out in input order, packed from `output[0]`, and
`*outputSize` receives their count. The caller guarantees `n >= 1`: with no keys there is no
last key to write the count, and the launchers would launch zero blocks, which HIP rejects
(`notes/today/ch02-facts.md` B4), so neither launcher has to handle it. Neither filter lets one block wait for another, so both
run on CUDA, HIP and Metal. (Fig. 12.8 calls a grid-wide scan inside one kernel, which needs
blocks to wait for earlier blocks; only CUDA documents that as safe, A10, B6, C4. Lab 12.5
builds that version.)

**Part A, exercise 1: three steps.**

- `keepFlagsKernel`: `keep[i]` is 1 for a kept key, 0 otherwise.
- The provided scan: `offset` = the exclusive scan of `keep`.
- `scatterKernel`: a kept key goes to `output[offset[i]]`, and the thread of the last key
  writes `*outputSize = offset[n-1] + keep[n-1]`.
- `launchStableFilterFlags(input, output, n, threshold, outputSize, keep, offset, scanScratch)`
  launches the three in order.

**Part B, exercises 3 and 4 in reduce-then-scan form.** Block `b` owns a tile of
`256 * coarse` keys and walks it in `coarse` rounds of 256 consecutive keys, so every
round's loads are coalesced.

- `countKeptKernel`: in every round each warp counts its kept keys with one vote and a
  popcount; the warp totals are added in shared memory; one thread writes `blockCounts[b]`.
- The provided scan turns `blockCounts` into `blockOffsets`.
- `stagedScatterKernel`: the block reads its tile again, round by round. Each kept key's
  place in a shared-memory staging buffer is (kept keys of earlier rounds) + (kept keys of
  earlier warps in this round) + (kept lanes below it in its warp). The last term is the
  binary prefix sum of p. 293: the vote, ANDed with a mask of the lanes below, popcounted.
  After the rounds, consecutive threads copy the staged run to `output[blockOffsets[b] + j]`.
  The last block writes `*outputSize`. The staging buffer is `extern __shared__`,
  `256 * coarse * 4` bytes, 16 KiB at coarse 16.
- `launchStableFilterStaged(input, output, n, threshold, outputSize, blockCounts,
  blockOffsets, scanScratch, coarse)`.

Part B is the shape of PyTorch's `torch.nonzero_static` on CUDA (count per block, scan the
counts, scatter, D4) and of its MPS `nonzero` (three dispatches, D5).

Rules for the votes in part B's CUDA code:

- Vote with `__ballot_sync` over the whole warp, called by every lane, kept or not, past the
  end or not. In that uniform control flow the result is exactly the set of kept lanes. Do
  not call `__activemask()` inside the branch as Fig. 12.3 does: NVIDIA now warns that it
  "cannot be used to determine which warp lanes execute a given branch" (A1).
- Use the provided `lane_mask_t` for every mask and `maskPopc` to count one. On HIP the mask
  is 64 bits and `__popc` would silently drop the upper half (B1). Never write `0xffffffff`.
- Lane and warp come from `threadIdx.x` and `warpSize`, never from a constant 32.

On Metal the vote and popcount become `simd_sum(keep ? 1u : 0u)` and the rank becomes
`simd_prefix_exclusive_sum(keep ? 1u : 0u)` (C1), called by every thread.

## Files

- Edit `work/kernel.cu` (CUDA, HIP and both emulators).
- Edit `work/kernel.metal` (`keep_flags`, `scatter_flags`, `count_kept`, `staged_scatter`;
  the signatures are given) and `work/launch.mm` (`encodeStableFilterFlags`,
  `encodeStableFilterStaged`). Each encoder puts several dispatches, the provided scan in the
  middle, into the one compute encoder it receives. An encoder made with plain
  `computeCommandEncoder` runs its dispatches in order and each sees the previous one's
  writes (`notes/today/ch02-facts.md` C9): the Metal counterpart of three launches on one
  CUDA stream. Read the SIMD-group width from the pipeline's `threadExecutionWidth`, size the
  threadgroup memory with `setThreadgroupMemoryLength:atIndex:` (multiples of 16 bytes,
  `ch05-facts.md` C8), and return false if it does not fit the device's
  `maxThreadgroupMemoryLength`.
- Provided, not edited: `labs/ch12/shared-g12/scan_u32.cu`, `scan_u32.metal` and
  `scan_u32_launch.mm`. A hierarchical exclusive scan of `uint32`: every block scans a tile
  of 1024, the tile totals are scanned the same way, and an add pass finishes
  (scan-scan-add, section 11.9, Fig. 11.14, pp. 273-274). No block waits for another.
  `scanScratchElems(n)` gives its scratch size; the driver allocates the scratch. The same
  files hold the copy kernel the benchmark compares against.

## Commands

- Test: `python3 labs/pmpp.py run 12.3 --backend emu` first (a few minutes: every GPU thread
  runs on the CPU), then `python3 labs/pmpp.py run 12.3`.
- Shared-memory races: `python3 labs/pmpp.py run 12.3 --backend race`. The staging buffer and
  the per-warp totals are rewritten every round; a missing barrier shows up here with both
  source lines, even when the answer happened to come out right.
- Measure: `python3 labs/pmpp.py run 12.3 --bench` (real devices only).
- The reference solution: add `--solution`.

## What the tests check

- The provided scan on its own, against a CPU exclusive scan, out of place and once in place.
- n = 1, 31, 32, 33, 255, 256, 257, 1000, 4099, 65537, 2^20 and 3 x 2^20 + 17 on a GPU. The
  emulator runs part A up to 4099 plus 65537 (at half kept only) and part B up to 4099; the
  race backend runs n up to 2048.
- Part A, and part B at coarse 1, 2, 4, 8 and 16. Coarse 16 runs only where its staging
  buffer fits the device's shared memory; otherwise the driver prints that it skipped it.
- Kept fractions 0, 0.001, 0.5, 0.97 and 1, and five patterns: only the first key, only the
  last key, every other key, runs of 12 across every tile boundary, and one run exactly
  filling one tile.
- Kept keys are all different and not in increasing order, so a key out of place is caught.
  The count must be exact, `output[0..count)` must equal the CPU stable filter element by
  element, and every slot past the count, 64 past the end of the array included, must still
  hold the sentinel the driver wrote. The driver fills `keep`, `offset`, `blockCounts`,
  `blockOffsets` and the scan's scratch with garbage before every run.

## Before you benchmark

Write down two predictions (the labs page records them as `ch12/labs/p-lab3-staged` and
`ch12/labs/p-lab3-coarse`):

1. At coarse 4 and half the keys kept, part A's time divided by part B's time.
2. Which coarsening factor will be fastest at half kept.

The model is section 12.5's byte count per key, at kept fraction f:

- Part A's named arrays: the input read twice (8), `keep` written once and read twice (12),
  `offset` written once and read once (8), the kept keys written (4f). 28 + 4f, so 30 bytes
  per key at half kept.
- Part B: the input read twice (8) and the kept keys written (4f); the per-block counts are
  a few bytes per thousand keys. 8 + 4f, about 10 at half kept.

The model charges the scan 8 bytes per element (read `keep`, write `offset`). The provided
scan really moves 16 (its add pass reads and writes `offset` again, p. 274), so part A's real
traffic is about 8 bytes per key above its model. Caches decide the rest: part B's second
read of a tile may come from a cache, and part A's arrays may not. Your device decides.

The benchmark starts at n = 2^24 and doubles n until a plain copy of the input takes 3 ms
(within an eighth of the device's memory). For kept fractions 0.001, 0.01, 0.1, 0.5, 0.9
and 1 it times part A ("flags") and part B ("staged") at every coarsening factor that fits.
Each row gives the time (best of 5 runs), GB/s over the minimum traffic `4n + 4 x kept + 4`
bytes, the model's bytes per key, and "at copy speed": how many bytes per key the copy
kernel would have moved in the same time. The last two columns are the comparison to make:
the model's count against what your device charged. Every timed result is checked before it
is reported. The numbers are measurements of your machine, not targets.

## Portability

- HIP: the votes use the 64-bit `lane_mask_t` and `__popcll` through `maskPopc`, and lanes
  come from `warpSize`; with 64-wide warps a block of 256 has 4 warps. Not compiled or run
  on AMD hardware here: unverified.
- Metal: the SIMD-group functions need GPU family Apple7, which every Apple silicon Mac is
  (infra-facts C2). The kernels never assume a SIMD-group width: the per-SIMD-group array is
  sized by `launch.mm` from `threadExecutionWidth`.
- The three-step design needs no forward progress between blocks, which is why it is the
  portable baseline (A10, B6, C4). Lab 12.5 trades that for reading every key once.
