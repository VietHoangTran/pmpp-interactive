# Lab 12.5: in-place stable filter in a single pass (optional)

Book: exercises 2, 3 and 4 (p. 302) and section 12.7 (p. 300, Fig. 12.11), on chapter 11's
dynamic block index and single look-back (Figs. 11.17 and 11.18, pp. 278-281). A bare item
such as (A9) below is in `notes/today/ch12-facts.md`.

CUDA family only: there is no Metal version (see "Why there is no Metal version").

Write the stable filter of lab 12.3 (keep a key when `key < threshold`, kept keys in input
order) as one kernel that reads every key once, and make it work in place: the driver calls
it with `output == input` as well as with a separate output.

`stableFilterSinglePassKernel`, one block per tile of `BLOCK_DIM * coarse` keys:

1. Thread 0 takes the block's index from `*blockCounter` with a relaxed device-scope
   `fetch_add` and shares it in shared memory. The block uses it, not `blockIdx.x`.
2. The block loads its whole tile into dynamic shared memory in coalesced rounds. Barrier:
   every read of this block's input is done.
3. Round by round it compacts the kept keys, in order, at the front of the same shared
   array: a warp vote gives each kept lane its rank, a scan of the warp totals gives the rest,
   as in lab 12.3's staged scatter. The block now knows its count.
4. Single look-back, as in Fig. 11.17: thread 0 waits until `flags[bid - 1]` is set (an
   acquire), reads `inclusive[bid - 1]`, the kept keys of all earlier blocks, stores
   `inclusive[bid]`, then sets `flags[bid]` (a release). Barrier.
5. The block writes its kept keys at that offset, consecutive threads on consecutive keys.
   The last block writes `*outputSize`.

`launchStableFilterSinglePass(input, output, n, threshold, outputSize, blockCounter, flags,
inclusive, coarse)` launches it. The driver zeroes `*blockCounter` and the flags before
every call; nothing in the kernel may assume they start at zero otherwise. The caller
guarantees `n >= 1`: with no keys there is no last block to write the count, and the
launcher would launch zero blocks, which HIP rejects (`notes/today/ch02-facts.md` B4). No `__restrict__`: `input` and `output` may be the same
array.

## Files

- Edit `work/kernel.cu` (CUDA, HIP and both emulators). The lane mask, `maskPopc`, the lane
  helpers and `spinPause()` are provided at the top: call `spinPause()` once per turn of the
  wait loop. It sleeps briefly on a GPU (`__nanosleep` on CUDA, ch11-facts A7;
  `__builtin_amdgcn_s_sleep(1)` on AMD, where rocPRIM found early MI100 parts hang in a bare
  spin, ch11-facts B3) and does nothing in the emulators.
- `BLOCK_DIM` is 256, and 64 on the race backend (below).

## Commands

- Logic: `python3 labs/pmpp.py run 12.5 --backend emu`.
- Ordering: `python3 labs/pmpp.py run 12.5 --backend race`.
- On a CUDA or HIP device: `python3 labs/pmpp.py run 12.5`, and `--bench` to measure.
- The reference solution: add `--solution`.

If a case runs longer than 120 seconds the driver stops and reports it: a block is almost
certainly waiting for a flag that is never set (a wrong index, a flag never stored, the
counter not used). Nothing hangs forever.

## Why it is correct in place: two conditions

Section 12.7's rule: never write a slot before the key that lived there has been read. A key
only ever moves to a lower index, so a block's writes land in its own tile or in earlier
tiles. Within the block, the barrier after the load orders its reads before its writes.
Across blocks, the book's argument is the scan chain of Fig. 12.11: for blocks i < j,
read(i) comes before scan(i), which comes before scan(j), which comes before write(j). Two
things must hold for that chain to mean anything on real hardware.

**1. The chain must order memory, not only time (A9).** The flag store must be a release and
the flag load an acquire, and every thread's read of the tile must come before the release,
which is what the barrier after the load gives you. CUB's select kernel says exactly this in
a comment and switches its look-back from relaxed to acquire/release when the selection may
be in place (A9). Here the flag also guards `inclusive[]`, a plain array, so acquire and
release are needed even out of place: that is NVIDIA's message-passing pattern
(ch11-facts A5). In place they protect the keys as well.

**2. Waiting on another block must be allowed to end (A10).** CUDA's execution model now says
that once a thread of a block has made progress, every thread of its block eventually makes
progress. A block that took index b - 1 from the counter has made progress, so the block
waiting for it waits for a block that will finish. Wait for `blockIdx.x - 1` instead and
nothing promises that block has even started; with more blocks than the device holds at once,
that can deadlock. This is why step 1 uses the counter (the book's reason, p. 280).

On HIP, AMD documents no forward-progress rule for workgroups (B6, ch11-facts B3). Its own
tutorial says blocks start in increasing order on current hardware but that the API does not
guarantee it, and rocPRIM needed per-architecture workarounds for both the block order and
the spin. The counter is the defensive choice there. **HIP: unverified** until a
`reports/verify-*.md` from an AMD machine shows this lab passing.

## Why there is no Metal version

Apple documents no forward progress between threadgroups (C4), and an Apple engineer,
answering on Apple's developer forum, called the chained wait of a single-pass scan undefined
behaviour (ch11-facts C5). There is also no grid barrier or cooperative launch to fall back
on. The documented-safe in-place design on Metal is one threadgroup walking its range in
order, with only threadgroup barriers, which lab 12.4's kernel already is for each row. The
portable multi-dispatch filter is lab 12.3.

## What the emulators can and cannot show

- `emu` runs blocks one after another in index order, so every block finds its predecessor's
  flag already set: the look-back never waits, and the in-place hazard cannot appear. It
  checks the logic: indices, the compaction, barriers within a block, bounds (under
  AddressSanitizer).
- `race` runs blocks of 64 threads, four at a time, as real OS threads under
  ThreadSanitizer, with real atomics and real acquire and release. Several blocks run at
  once there, so a block can reach its look-back before the block it waits for is done, and
  an access to `inclusive[]` or to the keys that nothing orders is reported with both source
  lines. A clean run does not prove there is no race: only the schedules that occurred are
  checked.
- Neither says anything about speed, and passing them does not prove the kernel runs on a
  GPU. The CUDA and HIP rows of `./verify.sh`'s report do.

### Optional experiment: take the order away

Once your kernel passes, make the flag load and the flag store `cuda::memory_order_relaxed`
and run `python3 labs/pmpp.py run 12.5 --backend race`. Predict first what will be reported.
When this lab was built, ThreadSanitizer reported two races, the same two in each of four
runs: a block's read of `inclusive[bid - 1]` against the earlier block's write of it, and,
in place, one block's write of an output key against another block's read of the input key
that lived in that slot. The answers still came out right in most cases: on the emulator, as
on a GPU, a missing order is a hazard that a run may or may not hit. Put the acquire and the
release back.

## What the tests check

- n = 1, 33, 257, 1000, 4099, 65537, 2^20 and 3 x 2^20 + 17 on a device; up to 65537 on
  `emu` (65537 at half kept only); 1, 33, 257, 1000 and 2048 on `race`.
- Coarse 1, 2 and 4. Every case runs out of place and in place.
- Kept fractions 0, 0.5 and 1, plus two patterns that make blocks write over earlier blocks'
  input: only block 1's keys kept (every one of them lands on block 0's slots), and only odd
  blocks' keys kept.
- The count exactly, and the first `count` elements equal to the CPU stable filter element by
  element. Out of place, every slot past the count must still hold the sentinel the driver
  wrote. In place, the array past the count is unspecified, as Thrust's `remove_if` leaves it
  (A8), and is not checked.

## Before you benchmark (a CUDA or HIP device)

Write down one prediction (the lesson asks for it as `ch12/labs/p-lab5-singlepass`): at
coarse 4 and half the keys kept, lab 12.3 part B's time divided by this kernel's time, out of
place.

The model: the single pass reads each key once, 4 + 4f bytes per key at kept fraction f
(6 at half kept), where lab 12.3 part B reads it twice, 8 + 4f (10). Against that, the
look-back chain makes every block's hand-off wait for the one before it, and the chain is as
long as there are blocks.

The benchmark times n = 2^24 at kept fractions 0.01, 0.5 and 1, coarse 1, 4 and 8, out of
place and in place. Each timed run starts from the original keys and zeroed flags; that reset
is not timed. Rows `singlepass` give the time (best of 5 runs) and GB/s over the minimum
traffic, `4n + 4 x kept` bytes. The labs page sets them beside lab 12.3's `staged` rows from
the same device. Every timed result is checked first. The numbers are measurements of your
machine, not targets.
