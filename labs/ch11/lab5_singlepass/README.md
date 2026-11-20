# Lab 11.5: single-pass scan with lookback (CUDA family, optional)

Book: section 11.9 (pp. 276-282), Figs. 11.16(b), 11.17 and 11.18.

Lab 11.4 joins the segments with several kernels and pays for it in traffic: 16 or 12
bytes per element where a scan needs 8 (pp. 272-276). Here one kernel does the whole
scan. Each block scans its segment. Its last thread, the leader, waits until the block
before it has published the sum of everything up to its end. The leader then publishes
the same for its own block, and the block adds the earlier sum to its elements. The
input is loaded once and the output stored once.

Two things make the waiting safe on CUDA:

- **The block index comes from a counter, not from `blockIdx.x`** (p. 280). Thread 0
  claims the next index with an atomic increment. A block can then only wait for a block
  that has already started. Under the forward-progress rules of NVIDIA's programming
  guide, a block that has started keeps making progress, so the wait ends (ch11 facts A6
  works this through). Nothing promises that the block with `blockIdx.x - 1` has
  started at all.
- **The flag is a device-scope atomic with release and acquire** (Fig. 11.17). The
  leader stores its running total in `partialSums[bid]` with a plain store, then sets
  `flags[bid]` with a release store. The next leader reads the flag with an acquire load
  and only then reads `partialSums[bid - 1]` with a plain load. This is NVIDIA's own
  "message passing" example (ch11 facts A5).

**No Metal version.** Metal documents no forward progress between threadgroups. An Apple
engineer, asked about exactly this chained scan, called its assumptions undefined
behaviour (ch11 facts C5). On Metal, lab 11.4 is the global scan.

**Your part** (in `work/kernel.cu`):

- **The dynamic block index** at the top of `SinglePassScanKernel` (Fig. 11.18, lines
  04-11): one thread claims it from `*blockCounter`, and the block shares it.
- **`interBlockScan`** (Fig. 11.17), with the leader as the last thread. Wait with
  `cuda::atomic_ref<unsigned int, cuda::thread_scope_device>` and
  `load(cuda::memory_order_acquire)`. Publish with a plain store to `partialSums[bid]`,
  then `store(1, cuda::memory_order_release)` on the flag. Return the earlier blocks' sum
  to every thread.
- **`launchSinglePassScan`**: zero the counter and the flags (`cudaMemset`) before every
  launch.

The block-level scan underneath (labs 11.1 and 11.2) is provided. So is the carry added
before the store.

Where this differs from the page:

- Fig. 11.18's parameter list is missing the comma before `unsigned int N` (p. 281).
- Fig. 11.18 never uses N. Every load and store here is checked against it.
- Fig. 11.17 reads the flag with `fetch_add(0, acquire)` and sets it with
  `fetch_add(1, release)`. NVIDIA's example uses `load` and `store`, the read-only and
  write-only forms with the same ordering (ch11 facts A5).
- Fig. 11.17 declares `prevBlockPartialSum_s` inside the device function. The kernel
  declares it here and passes a pointer, as with the warp totals in lab 11.1 (ch11 facts
  B1, C2).
- The book implies that the flags start at zero (p. 278) and says that the counter is
  "zero-initialized" (p. 280). The launcher does both before every launch.

## Files

- Edit `work/kernel.cu` (CUDA, HIP and both emulators): `SinglePassScanKernel`,
  `interBlockScan` and `launchSinglePassScan`. There is no `kernel.metal` or `launch.mm`.
- Provided, not edited: `../shared-g13/global_copy.cu` (a plain copy, the bandwidth
  baseline).

## Commands

- Test: `python3 labs/pmpp.py run 11.5 --backend emu`, then
  `python3 labs/pmpp.py run 11.5 --backend race`. On a CUDA machine,
  `python3 labs/pmpp.py run 11.5`.
- Measure: `python3 labs/pmpp.py run 11.5 --bench` (CUDA or HIP devices only).
- The reference solution: add `--solution`.

A block that waits for a flag nobody sets never stops waiting. On `emu` and `race` the
run then hangs until `pmpp.py` gives up, after 15 minutes; press Ctrl-C instead. The
usual causes: a missing release store, a wait on `flags[bid]` instead of `flags[bid - 1]`,
or a block 0 that waits.

## What the tests do

- On real devices: n = 1, 33, 1000, 4097, 100003 and 2^22, at blocks of 64 and 256. On
  `emu`: n up to 4097 at blocks of 64 and 256. On `race`: n = 33, 1000 and 4097 at block
  64, which is 1, 4 and 17 blocks, so at least four run at once.
- Data and tolerance as in lab 11.4: all ones and integers from -3 to 3 must match
  exactly; values uniform in [-1, 1] must lie within 1e-5 times the sum of |x| up to the
  element, plus 1e-6, of Fig. 11.1's loop run in double.
- Each case runs twice on the same buffers. Before the first launch every flag is set,
  `partialSums` and the output are NaN, and the counter holds one more than the number
  of blocks, as if left over from other work. The second launch gets new data and
  whatever the first one left.
- If the flags are not zeroed, a leader can read `partialSums[bid - 1]` before that
  block has written it.
- If the counter is not zeroed, blocks claim indices past the last block. `partialSums`
  and `flags` are followed by enough non-zero guard words that this fails with a message
  instead of a wait that never ends.
- After every launch the counter must equal the number of blocks. A kernel that never
  claims its index from the counter fails here, and only here.

## What the emulators can and cannot see

Checked when this lab was written, by planting each mistake in the reference solution:

- `race` reports a relaxed flag. With the acquire load made relaxed, the release store
  made relaxed, or both, ThreadSanitizer reported a data race between the read of
  `partialSums[bid - 1]` and the write of `partialSums[bid]`, every time it was run.
  `emu` passed all three.
- `race` reports flags that the launcher does not zero (a data race on `partialSums`).
  `emu` passes them.
- Both report a counter that the launcher does not zero, and a kernel that never claims
  its index from the counter.

What neither can see:

- `emu` runs blocks one after another in index order. The lookback never waits there,
  and the dynamic index always equals `blockIdx.x`.
- Thread scopes are ignored. A flag declared with `cuda::thread_scope_block` passes on
  both emulators. On a GPU it is a data race, which is undefined behaviour, because the
  flag is read by another block (ch11 facts A5, ch09 facts A2).
- Neither can produce the deadlock of waiting on the block with `blockIdx.x - 1`. A
  kernel that uses `blockIdx.x` produced right outputs on both when this was checked.
  The counter check above catches it only when the kernel never touches the counter: a
  kernel that claims an index and then uses `blockIdx.x` anyway passes every test here.
  Read your code for that one. The section 11.9 simulator shows the deadlock.
- A clean `race` run does not prove the absence of a race. ThreadSanitizer examines only
  the schedules that happened. Passing here does not prove the kernel runs on a GPU.

## Where the backends differ (not in the book)

- **HIP**: `cuda::atomic_ref` comes from libhipcxx (ROCm 10.1 and later), keeping the
  CUDA spelling. libhipcxx passes the memory order through to AMD's atomics (ch11 facts
  B4, infra facts B2). The course's HIP build links libhipcxx where ROCm provides it
  (unverified: no ROCm machine has built it yet). AMD documents no forward-progress guarantee
  between workgroups. AMD's own rocPRIM claims block indices with an atomic counter only
  on some architectures, and needed a sleep in its wait loop on early MI100 parts (ch11
  facts B3). Unverified on HIP until a `reports/verify-*.md` from an AMD machine shows it
  passing.
- **Metal**: none (ch11 facts C5), as above.
- **Libraries**: CUB's `DeviceScan` uses decoupled look-back, exercise 5's design, and
  says it "typically proceeds at "memcpy" speeds" (ch11 facts A1, A8). CUB indexes
  tiles by `blockIdx.x`, relying on launch order, which the CUDA guide does not promise
  (ch11 facts A6, our reading).

## Before you benchmark, predict

Write this down first. The labs page asks for it under the id in brackets.

- How many times faster than lab 11.4's scan-scan-add will the single-pass kernel run at
  n = 2^26? The byte count says 2 (16 bytes per element against 8)
  (`ch11/labs/p-lab5-single`).
- And what share of the copy kernel's rate will it reach? Every block waits for the one
  before it, a chain through all n/S blocks (p. 279).

## Reading the output

The benchmark scans n = 2^26 floats. It uses fewer if your device reports less than four
times the 512 MB of input and output. It runs at blocks of 256 and 1024 (S = 1024 and
4096), times the copy and then the single-pass kernel, and checks the scan before
reporting it.

- `time ms` is the best of five runs after one untimed warm-up. On CUDA and HIP it is
  host time around the launcher and a synchronise, so it includes the launcher's two
  `cudaMemset` calls.
- `GB/s` counts 8 bytes per element, the book's count for a single pass, divided by that
  time.
- `of copy` is the copy's time as a share of the kernel's.

Run lab 11.4's benchmark on the same device to compare. The labs page puts both labs'
rows side by side. The numbers are measurements of that machine at that size. Run each
benchmark more than once; if the runs disagree, that spread is part of the measurement.

## Go further (CUDA devices; not tested here)

- **Back-off.** CUDA has `__nanosleep(ns)` for exactly this kind of wait, and CUB calls it
  in its look-back (ch11 facts A7). Sleep a little inside the empty wait loop and
  measure. HIP documents no counterpart (ch11 facts B4), and neither the emulators nor
  the front-end check's stubs declare it, so keep it behind
  `#if defined(PMPP_BACKEND_CUDA)`, which the course's build defines for the CUDA
  backend only.
- **Decoupled look-back** (exercise 5, pp. 279-280). Each block publishes its own total
  first, with a second state, and the leader walks back past blocks whose running total
  is not ready yet, adding their own totals, until it finds one that is. The driver
  checks only the output, the counter and the guard words, so it accepts any flag
  values.
