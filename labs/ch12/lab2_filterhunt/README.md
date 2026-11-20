# Lab 12.2: filter hunt

Book: sections 12.2 to 12.4 (pp. 290-297), Figs. 12.2, 12.3 and 12.6. The defects are ours,
not the book's.

`work/kernel.cu` and `work/kernel.metal` hold the three unstable filters of lab 12.1,
keep = key < threshold, written correctly except for **three independent defects**. They
compile. Find and fix all three.

- `filterAtomicKernel` (Metal: `filter_atomic`): one counter update per kept key.
- `filterWarpAggregatedKernel` (Metal: `filter_warp_aggregated`): one atomic per warp, from a
  vote that every lane takes.
- `filterPrivatisedKernel` (Metal: `filter_privatised`): a private list and counter in
  shared memory, one public atomic per block.

Names, signatures and contract are lab 12.1's: `launchFilterAtomic`,
`launchFilterWarpAggregated`, `launchFilterPrivatised(input, output, n, threshold,
outputSize)`, and on Metal `encodeFilterAtomic` and friends. The caller sets `*outputSize`
to 0 before every call.

- Edit `work/kernel.cu` (the three kernels; the launchers below them are correct) and
  `work/kernel.metal`. The same three defects are in both files. `work/launch.mm` is correct.
- Not defects: `lane_mask_t` and the typed helpers `maskPopc` and `maskFfs`, the vote that
  every lane takes before any thread returns, the lane index from `warpSize`, the legacy
  `atomicAdd` under `__HIPCC__`, and on Metal the SIMD-group functions called from every
  thread. Lab 12.1's README says why for each.
- Test: `python3 labs/pmpp.py run 12.2 --backend emu`, then `--backend race`, then
  `python3 labs/pmpp.py run 12.2` for your GPU.

## What the tests do

Lab 12.1's checks, at smaller sizes. For every input the driver checks that `*outputSize`
equals the number of keys a sequential filter keeps, that `output[0, count)` sorted equals
the sequential filter's list sorted, and that every slot from `count` to `n` still holds the
sentinel `0xFFFFFFFF` it wrote before the call. Then it sets `*outputSize` to 0 again, calls
the kernel a second time and checks the same things.

- Sizes 1, 31, 32, 33, 255, 256, 257, 1000, 4099; on a GPU also 65537 and 2^20. The `race`
  backend stops at 1000.
- Uniform keys in [0, 2^24) with fractions kept 0, 0.001, 0.03, 0.5, 0.97 and 1, and the
  patterns of lab 12.1 (every key, no key, only the last key, only key 31 of every 32, only
  keys 32 to 63 of every 64, every other key, runs of 40).
- On a GPU every input gets 5 rounds of the two calls, and a failure says in how many rounds
  it was wrong. On the emulators, one round (under each of `emu`'s three thread orders).

## Your prediction

Before you run anything, read `work/kernel.cu` and write down the defects you can find by
reading. For each one, write down which backend you expect to report it: `emu`, `race`,
your GPU, or none, and why. The labs page asks for one number from that
(`ch12/labs/p-lab2-emu`): of the three defects, how many will `emu` catch? Then run `emu`,
`race` and your GPU, and fill in this table as you go:

| Defect | `emu` | `race` | your GPU |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

## Method

1. Emulator first: `python3 labs/pmpp.py run 12.2 --backend emu`. It runs one thread until
   that thread reaches a barrier, a warp `_sync` call or its end, then the next, under three
   fixed orders. Know what each kind of failure looks like there:
   - **A wrong count**: `*outputSize is 3; the sequential filter keeps 12`.
   - **A hole or a spill**: `slot 0 of [0, 1) was never written (it still holds the
     sentinel)`, or `slot 19, at or past the count 19, was written`. The count was right, so
     the keys went to the wrong places.
   - **A write outside the output array.** Each device buffer is one exact heap block on the
     emulator, so AddressSanitizer stops the run at the first such write. Read `WRITE`, the
     first frame (`kernel.cu:` and a line number), and how many bytes after the region the
     address is. The tests after the stop are not run: fix that one, then run again.
   - **A failure that depends on the thread order.** One defect fails under "threads in
     order" and passes under "threads reversed". In the in-order schedule thread 0 runs
     first from each barrier to the next; in the reversed one it runs last. Ask what thread
     0 does between two barriers that the rest of the block must have finished first.
2. Fix one thing, run again. The failures change as you go.
3. `python3 labs/pmpp.py run 12.2 --backend race` runs the emulated threads truly
   concurrently under ThreadSanitizer. Two threads touching the same memory without
   synchronisation are reported with both accesses and their `kernel.cu:` lines, whether or
   not the result came out wrong. One defect is invisible to `emu`: no emulated thread is
   ever interrupted in the middle of a read, an add and a write, so nothing is lost there.
   One access of a pair can be inside the emulator's own atomic (`pmpp_emu::atomic_rmw ...
   fetch_add`): that is an `atomic_ref` call of yours, racing with a plain access on the
   other line. ThreadSanitizer reports each pair of lines once per run, so a case after the
   first report can pass with the defect still there.
4. Then your GPU: `python3 labs/pmpp.py run 12.2`. Note which defects it showed you, and
   which it hid. Look at which sizes fail, too: a defect between warps can stay hidden
   while the whole input fits in one warp.

If a fix needs an atomic: on CUDA, the book's `cuda::atomic_ref<unsigned int, scope>` and
`fetch_add(..., cuda::memory_order_relaxed)`, with the scope chosen from the threads that
share the counter; on HIP, the legacy `atomicAdd` beside it under `#if
defined(__HIPCC__)`, because `<cuda/atomic>` comes from libhipcxx there, which the course's
HIP build links only where ROCm provides it (`notes/today/infra-facts.md` B1, B2); on Metal, atomic functions take
atomic types only, so the fix changes the parameter to `device atomic_uint*` and uses
`atomic_fetch_add_explicit` (infra-facts C1). The host still allocates a plain 32-bit counter.

## Reading your table

Whatever your GPU showed, read it against what the vendors document, because that is what
the next device is entitled to do:

- Atomics on one address are serialised in no fixed order (`notes/today/ch12-facts.md`,
  the table "Already settled elsewhere", which points to ch09 A3 and B1). That is what makes
  Fig. 12.2 unstable, and it is all the atomic promises. A plain read, add and write gives
  no promise at all: two threads can read the same count.
- Without a barrier, nothing says the rest of the block has finished a step. NVIDIA: on
  every current GPU "independent thread scheduling allows full concurrency between threads,
  regardless of warp" (`notes/today/ch12-facts.md` A1). Apple: the SIMD-groups of a
  threadgroup "make independent forward progress with respect to each other, in the absence
  of threadgroup barrier operations" (`notes/today/infra-facts.md` C5). Threads that happen
  to keep together on your device are not a promise.
- No GPU tool would have named the global counter. NVIDIA's `compute-sanitizer --tool
  racecheck` checks shared memory only, so an unsynchronised global counter is invisible to
  it, while the private counter in shared memory is in its reach; AMD and Apple document no
  race detector (ch12-facts, "Already settled elsewhere", pointing to ch09 A10, B6, C5). The
  `race` backend is ThreadSanitizer on CPU threads: a clean run there shows the schedules
  that happened, not all of them.
- One of the three defects is not a race at all. It is wrong on every backend, every time,
  and no schedule can hide it.

A case that passed 5 rounds on your GPU showed you 10 launches on one device. It did not show
you that the kernel is correct.

Nothing to benchmark here.
