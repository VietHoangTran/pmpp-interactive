# Lab 11.3: scan race hunt

Book: the race that Kogge-Stone's second barrier prevents (section 11.2, pp. 257-258), the
true and false dependences of section 11.3 (pp. 258-259), the warp and block scans (section
11.4, Figs. 11.8 and 11.9, p. 264) and the coarsened, register-tiled scan (sections 11.6 and
11.7, Figs. 11.12 and 11.13, pp. 268 and 271). The defects are ours. A bare item such as (D4)
below is in `notes/today/ch11-facts.md`.

`work/kernel.cu` and `work/kernel.metal` hold a row-wise **exclusive** scan in the shape
llama.cpp, MLX and PyTorch ship for batched rows (D1, D3, D4): `out[r][j]` is the sum of
`in[r][0..j-1]`, and `out[r][0]` is 0.

- One block per row. The block walks its row in chunks of `COARSE_FACTOR * blockDim.x`
  elements (4 per thread) and carries the sum of the chunks before.
- Each warp loads its slice of the chunk with coalesced reads into shared memory, then each
  lane reads back 4 consecutive elements and scans them sequentially in registers. Staging
  through shared memory is the book's (Figs. 11.12 and 11.13 stage the whole segment);
  staging per warp is ours. llama.cpp and MLX do not stage: each thread loads its own run
  directly (D1, D3).
- The lanes' run totals are scanned with the shuffle loop of Fig. 11.8; the last lane of each
  warp writes the warp's total into `warpSums_s`; warp 0 scans those totals (Fig. 11.9). The
  same `warpSums_s` is reused by every chunk.
- Each lane adds what comes before its run, and the warp stores its slice back, coalesced.

It compiles. It has **three independent defects**. Find and fix all three.

- Edit `work/kernel.cu` (`ExclusiveRowScanKernel`; `launchExclusiveRowScan(in, out, rows,
  cols, block)` below it is correct) and `work/kernel.metal` (`exclusive_row_scan`).
  `work/launch.mm` (`encodeExclusiveRowScan`) is correct too. The same three defects are in
  both kernels.
- Not defects: the typed lane mask and `warpSize` where the book writes `0xffffffff` and 32
  (HIP rejects a 32-bit mask and AMD warps may have 64 lanes, B1); the bounds checks, which
  Figs. 11.12 and 11.13 do not have; each warp staging its own slice instead of the whole
  block segment; the previous lane's total coming from a shuffle instead of a shared array;
  and, on Metal, the lane index passed into the helper, which cannot read thread attributes
  or declare threadgroup memory (C2).
- Test: `python3 labs/pmpp.py run 11.3 --backend emu`, then `--backend race`, then
  `python3 labs/pmpp.py run 11.3` for your GPU.

## What the tests check

- Blocks of 64 and 256 threads; rows 1 and 3 by columns 1, 33, 1000, 4097 and 32000. With 64
  threads a chunk is 256 elements, so 1000 columns is 4 chunks and 32000 is 125. The emulator
  stops at 4097 columns; the race backend runs 2 rows of 1, 33 and 1000 with 64 threads.
- Three kinds of row, mixed in one batch so that reading the wrong row fails: all ones (so
  `out[r][j] = j`), whole numbers from -3 to 3, and values uniform in [-1, 1].
- Ones and whole numbers are compared exactly. Every partial sum is a whole number far below
  2^24, which a float holds exactly, so every order of addition gives the same float: a
  difference is a defect, never rounding.
- `out[r][0]` must be exactly 0, the identity of +.
- The output starts as NaN, so an element nobody wrote shows as NaN, and 64 guard words past
  the end must be untouched.
- On a device each case runs 10 times on fresh data, and a failure says how many launches
  were wrong.

## Your prediction

Before you run anything, read `work/kernel.cu` and write down the defects you can find by
reading. For each one, write down whether your GPU will show it (a wrong value) or only the
emulators will catch it, and why. The labs page asks for one number from that
(`ch11/labs/p-lab3-gpu`): **running only on your GPU, how many of the three defects will make a
test fail?** Then run `emu`, `race` and your GPU, fixing one defect at a time, and fill in:

| Defect | `emu` | `race` | your GPU |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

## Method

1. Emulator first: `python3 labs/pmpp.py run 11.3 --backend emu`. It runs one thread until that
   thread reaches a barrier, a shuffle or its end, then the next, under three fixed orders.
2. Fix one thing, run again. The failures change as you go. Use what the passing and failing
   cases have in common: which block size, how many chunks, which kind of row, which column
   goes wrong first.
3. Know what the emulator's failures look like:
   - **Shared memory nobody has written yet** holds `0xCD` bytes, filled afresh for every
     block. As a float that is about -4.3e8.
   - **A value that is exactly double what it should be**, or off by a whole run's total,
     points at arithmetic, not timing: it fails in every thread order and on every backend.
   - What `emu` cannot show: it resolves a barrier or a shuffle only when no thread can run
     any further. So every thread finishes its stretch of work up to its next barrier or
     shuffle before any thread starts the stretch after that. A defect whose damage needs a
     fast warp to be a whole stretch ahead of a slow one stays hidden there.
4. `python3 labs/pmpp.py run 11.3 --backend race` runs the emulated threads truly concurrently
   under ThreadSanitizer. Two threads touching the same memory with nothing ordering them are
   reported with both source lines, whether or not a value came out wrong. ThreadSanitizer
   reports each pair of lines once per run, so a case after the first report can pass with
   the defect still there.
5. Then your GPU: `python3 labs/pmpp.py run 11.3`. Note which defects it showed you and which
   it hid.

## Reading your table

Whatever your GPU showed, read it against what the vendors document, because that is what the
next device is entitled to do:

- Shuffle-up leaves the lowest `delta` lanes with their own value, on CUDA, HIP and Metal alike
  (A3, B1, C1). Whether a lane may use what it received depends on its position in the warp.
- Lanes of one warp: NVIDIA warns that "the ability for threads to diverge and reconverge at
  sub-warp granularity makes such assumptions invalid" about lockstep, and asks for
  `__syncwarp()`, which "provides memory ordering among the participating threads within a
  warp" (`notes/today/ch04-facts.md` A10). AMD documents lockstep for its lanes (ch04 B5); Apple
  provides `simdgroup_barrier`, "an execution and memory barrier" (`infra-facts.md` C4). Code
  that passes because lanes happened to run in lockstep has a data race all the same.
- Different warps run independently (NVIDIA, ch04 A9; Apple: SIMD-groups "make independent
  forward progress ... in the absence of threadgroup barrier operations", infra C5). A warp
  that has finished its chunk may be anywhere in the next one.
- "Reading from or writing to the same memory location without memory fences or
  synchronization results in undefined behavior" (`ch10-facts.md` A1). A write after a read
  is a race too: the book's second barrier in Fig. 11.3 exists only to order one (pp. 257-258),
  and PyTorch's MPS scan has a barrier whose comment says that without it "long
  (multi-iteration) scans race" (D4).

A case that passed 10 times on your GPU showed you 10 schedules of one device. It did not show
you that the kernel is correct.

## The tolerance for uniform rows

Whole-number rows are exact. A uniform row is compared with
`|got - ref| <= 2 * depth * 2^-24 * (|x_0| + ... + |x_{j-1}|) + 1e-6`, where `ref` is the sum
in double (Fig. 11.1's loop, made exclusive) and `depth = chunks per row + 16`. That is lab
11.1's tolerance (`1e-6` times the sum of magnitudes, plus `1e-6`, within one segment of at
most 1024) with the relative term grown for the carry: about `2e-6` for a row of one chunk,
`1.7e-5` for 125 chunks.

Why it grows: every output is the inputs added along a tree, and each addition rounds by at
most `2^-24` relative. An input reaches its output through at most 3 additions in its lane's
run, 6 in the warp scan (warps of up to 64 lanes), 5 in the scan of warp totals, one per chunk
in the carry, and 3 that combine the parts: `chunks + 17` at most, and the first addition of a
run (0 + x) is exact. So the error is at most about `depth * 2^-24` times the sum of the
magnitudes. The factor 2 spares rounding in the bound itself. A defect here moves values by
whole elements, orders of magnitude more.

## Portability

- HIP: the typed lane mask and `warpSize` let `kernel.cu` build through `pmpp/vendors/hip.h`.
  With 64-wide warps, a block of 64 is one warp: an arithmetic defect that needs a second
  warp shows only with 256 threads. Not compiled or run on AMD hardware here: unverified.
- Metal: `launch.mm` reads the SIMD-group width from the pipeline (`threadExecutionWidth`) and
  refuses a threadgroup that is not whole SIMD-groups, or has more SIMD-groups than a
  SIMD-group has lanes.

Lab 11.6 builds its top-p kernel on the fixed version of this scan.

Nothing to benchmark here.
