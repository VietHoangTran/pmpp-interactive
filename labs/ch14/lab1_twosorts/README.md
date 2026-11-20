# Lab 14.1: two first sorts, odd-even and one-bit radix

Book: section 14.2 (pp. 331-333, Fig. 14.2) and section 14.5 (pp. 336-339, Figs. 14.6 and
14.7). A bare item such as (A9) or (C4) below is in `notes/today/ch14-facts.md`.

Two sorts of 32-bit unsigned keys, written the way the chapter builds them, and timed against
each other and against one CPU thread.

**Part A: odd-even sort (Fig. 14.2).**

- `oddEvenStep(data, hasChanged, n, isOddStep)`: thread t owns the pair (i, i + 1) with
  i = 2t on an even step and 2t + 1 on an odd step. It puts the pair in order and, if it
  swapped, sets `*hasChanged` to 1 with an atomic store (`atomicExch(hasChanged, 1u)`, or
  `cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(*hasChanged).store(1u,
  cuda::memory_order_relaxed)`). It returns at once when n < 2 or when its pair runs past
  the end.
- `launchOddEvenSort(data, flag, n)`: the host loop the book describes but does not show.
  Even and odd steps alternate, starting with an even step. Clear the flag, run an even step
  and the odd step after it, then read the flag. Stop only when that pair of steps made no
  swap. Return the number of steps launched.

**Part B: a one-bit LSD radix sort of (key, value) pairs (Figs. 14.6 and 14.7).** Fig. 14.7 is
one kernel that calls `gridExclusiveScan` between extracting the bits and moving the keys.
The book never defines that function, and a scan across blocks inside one kernel needs the
blocks to wait for one another: a cooperative launch on CUDA and HIP, nothing documented on
Metal (C4). So a pass here is three launches:

- `radixExtractBits(keys, bits, n, bit)`: `bits[i]` = bit `bit` of `keys[i]`, and
  `bits[n] = 0`.
- the provided `launchExclusiveScan(bits, n + 1, scratch)`: afterwards `bits[i]` is the
  number of ones before key i and `bits[n]` the number of ones in total.
- `radixScatter(keysIn, valsIn, keysOut, valsOut, onesBefore, n, bit)`: the two destination
  formulas of Fig. 14.6; the value moves with its key.
- `launchRadixSortOneBit(keys, vals, tmpKeys, tmpVals, bits, scanScratch, n)`: 32 passes,
  least significant bit first, reading one pair of buffers and writing the other. The sorted
  pairs must end in `keys`/`vals`.

The values are the keys' input positions. In lab 14.3 they are token ids. With positions as
values the test can see stability directly: among equal keys the positions must increase.

## Five traps on the pages

Each of these turns a literal reading of the chapter into a wrong or fragile program. Work
out the fix yourself; the reference solution says in its comments where it differs from the
page.

1. **The N + 1 scan** (`notes/ch14.md`, "Unclear in the PDF", item 4). Fig. 14.7 reads
   `bits[N]` as the total number of ones. That entry only exists, and only holds the total,
   if the exclusive scan runs over N + 1 entries. Fig. 14.6 draws N. What matters is the
   extra entry, not what you put in it: an exclusive scan never carries its last input into
   any output. Write 0 there anyway, so the scanned array is "the bits, then nothing".
2. **The "zero bucket" slip** (item 3). On p. 337 the sentence that introduces the second
   formula says it is for keys mapping to the zero bucket. The formula is the one bucket's.
   Use the formulas as the figure labels them.
3. **The unsigned `n - 1` wrap.** Fig. 14.2 guards with `i < N - 1`. With unsigned N, N - 1
   at N = 0 is the largest unsigned value and every thread passes. The driver never launches
   your kernel with n = 0, so no test catches it; write the guard so it cannot wrap anyway.
4. **The stopping rule** (item 7). The book says to keep going "until no elements are
   swapped". One quiet step proves nothing: `[1, 3, 2, 4]` makes no swap on its first, even,
   step and is not sorted. Two steps in a row, one even and one odd in either order, both
   quiet, show that every adjacent pair is in order. This lab reads the flag once per
   even-odd pair, so it sees only the quiet pairs that start on an even step: it may launch
   one step more than necessary (10 instead of 9 for `[8, 1, 2, 3, 4, 5, 6, 7]`) in exchange
   for half the reads. The tests include the trap and a largest-first input.
5. **The flag race** (p. 333). Many threads may set `hasChanged` in the same step. The book
   calls the race benign because they all write 1, then warns that the C++ memory model
   makes it undefined behaviour anyway. NVIDIA's memory model says the same (A9). Use an
   atomic store. The `race` backend reports a plain store as a write-write data race with
   both source lines; `emu` cannot see it, because it never interrupts a thread between
   synchronisation points.

## Files

- Edit `work/kernel.cu` (CUDA, HIP and both emulators): `oddEvenStep`, `launchOddEvenSort`,
  `radixExtractBits`, `radixScatter`, `launchRadixSortOneBit`.
- Edit `work/kernel.metal` (`odd_even_step`, `radix_extract_bits`, `radix_scatter`; the
  signatures are given). The flag is a `device atomic_uint*` set with
  `atomic_store_explicit(..., memory_order_relaxed)`: Metal's atomic functions only take
  atomic-typed pointers (`notes/today/infra-facts.md` C1).
- Edit `work/launch.mm`:
  - `runOddEvenSort(queue, step, data, flag, n)` commits one command buffer per even-odd pair
    of steps, waits for it, and reads the flag through `flag.contents`. A shared-storage
    buffer needs no copy, but the CPU may read it only after the GPU work that wrote it has
    completed (`notes/today/ch05-facts.md` C6; `waitUntilCompleted`, infra-facts C12).
    Returns the step count, or -1 if a command buffer failed.
  - `encodeRadixSortOneBit(enc, extract, scatter, keys, vals, tmpKeys, tmpVals, bits,
    scanScratch, n)` puts all 32 passes, the provided scan's dispatches included, into the
    one compute encoder it receives. An encoder made with plain `computeCommandEncoder` runs
    its dispatches in order and each sees the previous one's writes
    (`notes/today/ch02-facts.md` C9).
- Provided, not edited: `labs/ch14/shared-g11/scan.cu`, `scan.metal`, `scan_launch.mm`
  (ours). An in-place exclusive scan of unsigned ints for any count, built as section 11.9
  builds one (scan-scan-add, Fig. 11.14, pp. 273-274): block scans, a recursive scan of the
  block totals, an add. No block waits for another. `scanScratchCount(count)` gives its
  scratch size; the driver allocates the scratch.

## Commands

- Test: `python3 labs/pmpp.py run 14.1 --backend emu` first (a few minutes: every GPU thread
  runs on the CPU), then `python3 labs/pmpp.py run 14.1`.
- The flag race: `python3 labs/pmpp.py run 14.1 --backend race`.
- Measure: `python3 labs/pmpp.py run 14.1 --bench` (real devices only).
- The reference solution: add `--solution`.

## What the tests check

- The provided scan on its own against a CPU exclusive scan, so a broken provided file shows
  up as itself.
- Odd-even: n = 1 (checked together with 2), 3, 4, 8, 33, 256 and 1000; the emulator stops
  at 256, the race backend at 64. Inputs: random, reversed, already sorted, all equal,
  largest first (the book's worst case, p. 333) and the trap above. The result must be
  sorted and a permutation of the input; the step count must be even (whole pairs), at least
  2 for n of 2 or more, and at most n + 2 rounded up to whole pairs (n + 2 for even n, n + 3
  for odd n: the sort needs at most n steps, then the pair holding the last swap must finish
  and one quiet pair follow). Guard slots past the end hold 0, so a pair that reaches past
  the last element swaps a key into them and is caught.
- Radix: n = 1 (checked together with 2), 31, 32, 33, 1000, 4097, 2^16 and 2^20; the emulator
  stops at 4097, the race backend at 1000. Keys: full-range random (the top bit is set about
  half the time, so a signed key or a pass that stops before bit 31 fails), 0..15 (heavy
  ties), all equal, sorted, reversed, powers of two. The keys must equal `std::stable_sort`'s
  and the values must equal its permutation exactly. The emulator runs every key pattern at
  n = 1000 and two at each other size; it pays for every thread of every block of the
  96 or more launches a sort makes.
- Before every run the driver writes 0xCDCDCDCD into every slot of `bits`, the scan's
  scratch, the temporary buffers and 64 guard slots past the end of every array. A pass that
  writes past n is caught; on the emulator AddressSanitizer stops one that writes past the
  allocation and names the line.

## Before you benchmark

Write down two predictions (the labs page asks for them):

1. At n = 4096, how many times slower than your one-bit radix sort will your odd-even sort be?
   (`ch14/labs/p-lab1-oddeven`)
2. At 2^22 pairs, how many times faster than one CPU thread's `std::stable_sort` will your
   one-bit radix sort be? (`ch14/labs/p-lab1-cpu`)

Write down why first, from two counts:

- **Launches and host round trips per sort.** Odd-even launches one kernel per step, about n
  of them (at most n + 3), and the host reads the flag back once per pair of steps, so about
  n / 2 round trips that each wait for the GPU to finish. The radix sort launches
  32 x (2 + the scan's launches) kernels and never waits in the middle. The scan of n + 1
  entries is 1 launch up to 1024 entries, 3 up to 1024 x 1024, 5 beyond.
- **Bytes per pass.** One odd-even step reads every key and writes the ones it swaps. One
  radix pass reads the keys and writes the bits (8 bytes per pair), scans n + 1 entries (the
  provided scan reads and writes them twice when there is more than one tile: 16), reads
  keys, values and bits and writes keys and values (20). Count it for 32 passes, against
  `std::stable_sort`'s O(n log n) comparisons and moves.

The benchmark times odd-even at n = 1024 and 4096 (wall time of the whole host loop: launches
and flag read-backs), the one-bit radix sort at n = 1024, 4096, 2^16, 2^20 and 2^22, and
`std::stable_sort` of the same (key, position) pairs on one CPU thread with `std::chrono` (a
host measurement). Radix rows give the device time of the command buffer on Metal and the
wall time around the launches and the final synchronisation on CUDA and HIP, plus a "wall"
column, the host's clock on every backend; "odd-even/radix" divides the two wall times. Each
row is the best of 5 runs after a warm-up, and every timed result is checked before it is
reported. The numbers are measurements of your machine, not targets.

## Portability

- HIP: `atomicExch` has the same name and arguments (infra-facts B1); the kernels use no warp
  intrinsics, so no lane masks. Not compiled or run on AMD hardware here: unverified.
- Metal: the flag is `device atomic_uint*` (infra-facts C1); one command buffer per pair of
  steps, read after `waitUntilCompleted` (ch05-facts C6); all radix passes in one serial
  encoder (ch02-facts C9). The three-launch pass needs no inter-threadgroup progress, which
  Metal does not document (C4).
- The book's own advice (p. 347) is to use a library sort, `cub::DeviceRadixSort` on CUDA
  (stable, pairs, descending: A2) or `hipcub::` on HIP (B1). Metal has no kernel-level library
  sort (C1). This lab is for understanding what such a sort does per pass.
