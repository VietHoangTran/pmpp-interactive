# Lab X1.4: filter hunt, three defects in a fused top-p and min-p kernel

Not in the book: module X1 is ours, and so are the defects. The kernel is lab X1.2's part B, so
its pieces are the book's two-stage reduction (Figs. 10.13 and 10.16, pp. 240 and 243) and its
warp and block scans (Figs. 11.8 and 11.9, p. 264), and its rules are llama.cpp's
(`notes/today/x1-facts.md` D1, `ch11-facts.md` D2, `ch12-facts.md` D3). Lab X1.2's README
states the contract in full; in short, for each row of K logits sorted in descending order:

- `sortedProbs` = the softmax at temperature 1 of the row;
- `n = max(1, min(a, b))`, where `a` counts the tokens whose mass ranked before them is below
  `topP` (token 0 always, every token when `topP >= 1`), and `b` counts the logits at or
  above `logit[0] + logf(minP)` (every token when `minP <= 0`).

`work/kernel.cu` and `work/kernel.metal` hold `filterSortedFused` / `filter_sorted_fused`
in the shape lab X1.2 suggests: one block of 256 threads per row; pass 1 adds
`exp(l - l_0)` with the two-stage block sum; pass 2 walks the row in chunks of 1,024 with
the mass of earlier chunks carried in a register, writes each probability, decides min-p on
each logit and top-p on each token's mass, and counts both; one reduction turns the counts
into `n`. The shared array `warpSums_s` (`simdSums_s` on Metal) holds the block sum first,
then each chunk's warp totals.

It compiles. It has **three independent defects**. Find and fix all three.

- Edit `work/kernel.cu` (`filterSortedFused`; `launchFilterSortedFused` below it is correct)
  and `work/kernel.metal` (`filter_sorted_fused`). `work/launch.mm`
  (`encodeFilterSortedFused`) is correct. The same three defects are in both kernels.
- Not defects: the typed lane mask and `warpSize` where the book writes `0xffffffff` and 32
  (HIP rejects a 32-bit mask and AMD warps may have 64 lanes, `infra-facts.md` B3); the
  threshold `-FLT_MAX` that keeps every token when `minP <= 0` (`-INFINITY` is undefined
  under Metal's fast math, `ch10-facts.md` C3); `logf(minP)` computed once on the host and
  passed in, as llama.cpp computes it; no max pass, because the row is sorted; each warp
  staging its own slice of the chunk; the first lane's exclusive value set to 0 after the
  shuffle; `max(1, ...)`; the two counts sharing one reduction; and the helpers being
  `static`.
- Test: `python3 labs/pmpp.py run x1/lab4 --backend emu`, then `--backend race`, then
  `python3 labs/pmpp.py run x1/lab4` for your GPU.

## What the tests check

Lab X1.2's part B checks, at smaller sizes (the driver includes lab X1.2's `main.cpp`):

- Rows 1, 3 and 7 by K = 1, 31, 257, 1,000, 1,024, 1,025, 2,100, 4,096, 4,099 and 32,000 on a
  GPU, where every batch runs ten times on fresh outputs and a failure says which launch was
  wrong. The emulator runs rows 1 and 3 by K = 1,000, 1,024, 1,025, 2,100 and 4,099, the race
  backend 2 rows of K = 33, 1,025 and 2,100. Rows longer than one chunk of 1,024 are there on
  every backend on purpose.
- **Exact rows**, compared with no tolerance: a group of `2^j` logits of 0 and the rest -1000,
  so every probability is a power of two or 0 and every partial sum is exact in any order
  (provided `exp(0)` is exactly 1 and `exp(-1000)` exactly 0, which lab X1.2's README
  discusses).
  p sits exactly on the inclusive sum at one token (token 0, inside a lane's run of 4, the
  last token of the first chunk, the first of the second, inside the second): that token must
  be kept and the next dropped. p = 0 keeps one token, p = 1 every token. A logit planted at
  exactly `0 + logf(minP)` is kept, the float just below it is not.
- **Softmax rows**: probabilities within `1e-4` relative of the reference in double, and `n`
  equal to the reference's wherever the band does not allow a choice (lab X1.2's README
  derives the band).
- Outputs start as NaN and -1, and 64 guard entries past each output must be untouched.

## Your prediction

Before you run anything, read `work/kernel.cu` and write down the defects you can find by
reading. For each one, write down which backend you expect to report it: `emu`, `race`, your
GPU, or none of them, and why. The labs page asks for one number from that
(`x1/labs/p-lab4-emu`): **of the three defects, how many will `emu` report?** Then run `emu`,
`race` and your GPU, fixing one defect at a time, and fill in:

| Defect | What it is | `emu` | `race` | your GPU |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

## Method

1. Emulator first: `python3 labs/pmpp.py run x1/lab4 --backend emu`. It runs one thread until
   that thread reaches a barrier, a shuffle or its end, then the next, under three fixed
   orders (in order, reversed, shuffled).
2. Fix one thing, run again. The failures change as you go. Use what the failing cases have in
   common: which K, which kind of row, which p, and the message's hint. The driver names a
   known mistake when `n` matches what that mistake would give.
3. Know what the emulator can and cannot show:
   - **Arithmetic** (a wrong sum, a wrong comparison, a value added in the wrong place) is
     wrong in every thread order and on every backend. Exact rows turn it into a wrong count
     with no rounding to hide behind.
   - **Shared memory nobody has written yet** holds `0xCD` bytes, filled afresh for every
     block.
   - What `emu` cannot show: it resolves a barrier or a shuffle only when no thread can run
     any further, so every thread finishes its stretch of work up to its next barrier or
     shuffle before any thread starts the stretch after it. A defect whose damage needs one
     warp to be a whole stretch ahead of another stays hidden there, in all three orders.
4. `python3 labs/pmpp.py run x1/lab4 --backend race` runs the emulated threads truly
   concurrently under ThreadSanitizer. Two threads touching the same memory with nothing
   ordering them are reported with both source lines, whether or not a value came out wrong.
   ThreadSanitizer reports each pair of lines once per run, so a case after the first report
   can pass with the defect still there.
5. Then your GPU: `python3 labs/pmpp.py run x1/lab4`. Note which defects it showed you and
   which it hid.

## Reading your table

Whatever your GPU showed, read it against what the vendors document, because that is what the
next device is entitled to do:

- Different warps run independently (NVIDIA, `ch04-facts.md` A9; Apple: SIMD-groups "make
  independent forward progress ... in the absence of threadgroup barrier operations",
  `infra-facts.md` C5). A warp that has read a value from shared memory may be anywhere in the
  next phase while another has not read it yet.
- "Reading from or writing to the same memory location without memory fences or
  synchronization results in undefined behavior" (`ch10-facts.md` A1). A write after a read is
  a race too: lab 10.3's softmax has a barrier between two block reductions that share one
  array for exactly that reason, and llama.cpp and PyTorch place the same barrier
  (ch10-facts D1, D4).

A case that passed ten times on your GPU showed you ten schedules of one device. It did not
show you that the kernel is correct.

## Portability

- HIP: the typed lane mask and `warpSize` are there so that `kernel.cu` can build through `pmpp/vendors/hip.h`;
  with 64-wide warps a block of 256 has 4 warps. Not compiled or run on AMD hardware here:
  unverified.
- Metal: `launch.mm` reads the SIMD-group width from the pipeline (`threadExecutionWidth`) and
  refuses 256 threads that are not whole SIMD-groups, or more SIMD-groups than a SIMD-group has
  lanes.

Nothing to benchmark here.
