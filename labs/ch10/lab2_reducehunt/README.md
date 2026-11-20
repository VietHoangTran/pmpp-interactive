# Lab 10.2: reduction hunt

Book: sections 10.7 to 10.9 (pp. 237-246): `warp_reduce` (Fig. 10.13), the two-stage
kernel (Fig. 10.16) and the multi-block commit (Fig. 10.18), extended to any length as
exercise 5 asks (p. 250). The defects are ours, not the book's, except one that the
book's own Fig. 10.16 has as printed.

`work/kernel.cu` and `work/kernel.metal` hold `reduceAnyKernel`, a sum over any length
`n`. Blocks have 128 threads. Each thread adds two elements, so a block covers 256; each
warp reduces its values with shuffles, warp 0 reduces the per-warp sums, and thread 0 of
every block adds the block's total into `*out`, which the launcher sets to 0 first. It
compiles. It has **four independent defects**. Find and fix all four.

- Edit `work/kernel.cu` (`reduceAnyKernel`; `launchReduceAny(in, out, n)` below it is
  correct) and `work/kernel.metal` (`reduce_any`). `work/launch.mm` is correct too. The
  same four defects are in both kernels.
- Not defects: the lane mask has its own type and the shuffle loop starts at
  `warpSize/2`, where the book writes `0xffffffff` and `WARP_SIZE/2` with a `WARP_SIZE`
  of 32. HIP rejects a 32-bit mask and
  AMD warps may be 64 wide (`notes/today/ch10-facts.md` B1). On Metal the SIMD-group
  width comes from `[[threads_per_simdgroup]]`, because Apple documents no value.
- Test: `python3 labs/pmpp.py run 10.2 --backend emu`, then `--backend race`, then
  `python3 labs/pmpp.py run 10.2` for your GPU.

The driver tests lengths that are multiples of 256 and lengths that are not (on the
emulators at most 4097). Each case is launched twice on the same output buffer with
different data. The data are whole numbers from 1 to 8 and every partial sum stays below
2^24, so every order of addition gives exactly the same float: a wrong sum is a defect,
never rounding.

## Your prediction

Before you run anything, read `work/kernel.cu` and write down the defects you can find by
reading. For each one, write down whether you expect your GPU to show it (a wrong sum)
or only the emulators to catch it, and why. The labs page asks for one number from that
(`ch10/labs/p-lab2-gpu`): of the nine lengths your GPU runs, how many will the kernel as
shipped get right in all 20 launches? Then run `emu`, `race` and your GPU, and
compare. Fill in this table as you go:

| Defect | `emu` | `race` | your GPU |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |

## Method

1. Emulator first: `python3 labs/pmpp.py run 10.2 --backend emu`. It runs one thread until
   that thread reaches a barrier, a shuffle or its end, then the next, under three fixed
   orders, so a missing barrier is wrong there every time.
2. Fix one thing, run again. The failures change as you go. Use what the passing and
   failing cases have in common: which lengths, which thread orders.
3. Know what each kind of failure looks like on the emulator:
   - **A shuffle that reads a lane taking no part in it.** The emulator hands back `0xCD`
     bytes, which as a float is a large negative number, and prints a note naming the
     shuffle's line and the lane it read.
   - **A read outside an array.** Each device buffer, and each block's `__shared__`
     memory, is one exact heap block there, so AddressSanitizer stops the run at the
     first such read.
     Read `READ`, the first frame (`kernel.cu:` and a line number), and how many bytes
     after the region the address is. The tests after the stop are not run.
   - **Shared memory nobody has written yet** holds `0xCD` bytes, filled afresh for
     every block: wrong values again, not a stop.
4. `python3 labs/pmpp.py run 10.2 --backend race` runs the emulated threads, and two
   blocks at a time, truly concurrently under ThreadSanitizer. Two threads touching the
   same memory without synchronisation are reported with both source lines, whether or
   not the sum came out wrong. One defect is invisible to `emu` and shows only here.
   ThreadSanitizer reports each pair of lines once per run, so a case after the first
   report can pass with the defect still there.
5. Then your GPU: `python3 labs/pmpp.py run 10.2`. Each case is launched 20 times (ten
   pairs) and a failing case says in how many it was wrong. Note which defects your GPU
   showed you, and which it hid.

If a fix needs an atomic: on CUDA, the book's `cuda::atomic_ref` (Fig. 10.18, which
needs `#include <cuda/atomic>`) or the legacy `atomicAdd`; on HIP, the legacy `atomicAdd`,
because `<cuda/atomic>` comes from libhipcxx there and the course's HIP build does not
link it (`notes/today/infra-facts.md` B2; the reference solution chooses with
`#if defined(__HIPCC__)`); on Metal, atomic
functions take only atomic types, so the pointer itself must be declared atomic
(`notes/today/infra-facts.md` C1).

## Reading your table

Whatever your GPU showed, read it against what the vendors document, because that is
what the next device is entitled to do (`notes/today/ch10-facts.md` and
`infra-facts.md`):

- Reading a lane that is not taking part in a shuffle is undefined. NVIDIA: "If the
  target thread is inactive, the retrieved value is undefined" (ch10 A3). Apple: "A
  thread needs to only read data from another active thread in the SIMD-group" (infra C2).
  An undefined value can be the right one, every time, on one device.
- Shared or threadgroup memory starts with whatever was there. Nothing promises zeros.
- Without a barrier, warps make progress independently of each other (NVIDIA: "different
  warps execute independently", ch04 A9; Apple: SIMD-groups "make independent forward
  progress", infra C5). The barrier is also what orders one warp's writes before another
  warp's reads (ch10 A1).
- Two threads reading and writing the same location without synchronisation is a data
  race, and NVIDIA calls it undefined behaviour: "Reading from or writing to the same
  memory location without memory fences or synchronization results in undefined
  behavior" (ch10 A1). That holds for two blocks as much as for two warps.

A case that passed 20 times on your GPU showed you 20 schedules and one device's answer to
an undefined read. It did not show you that the kernel is correct.

Nothing to benchmark here.
