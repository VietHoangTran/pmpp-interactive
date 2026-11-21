# Lab 13.1: merge two sorted arrays with co-rank partitioning

Book: sections 13.2 to 13.6 (pp. 305-320), Figs. 13.2, 13.5 and 13.9, and the idea of
Exercise 3 (p. 328). Part (b) has the shape that CUB, rocPRIM and MLX ship, which is not
the book's Figs. 13.11 to 13.13 (`notes/today/ch13-facts.md` A7, B2, D1).

A and B are sorted arrays of key-value pairs, `struct KV { int key; int val; }` (in
`merge.h`). You write the first `total` outputs of their stable merge to C, comparing
keys only. Ties go to A, and elements from the same array keep their order: the book's
rule (pp. 304-305). `total = m + n` is the whole merge. A smaller `total` is ours, not the
book's: it is the merge of two top-k lists, where only the first k outputs are wanted.

**Your part:**

- **(a) The basic kernel** (Figs. 13.2, 13.5, 13.9, corrected).
  - `co_rank(k, A, m, B, n)`: of the first k outputs, how many come from A. A binary
    search on `.key`, with min and max written as ternaries and the bounds tests ahead of
    the reads.
  - `merge_sequential`: the two-phase sequential merge, with `<=`.
  - `mergeBasicKernel`: each thread owns `perThread` consecutive outputs. Its first
    output `k_curr` and the next thread's `k_next` are limited to `total`, and a thread
    with `k_curr >= total` returns. Then two `co_rank` calls and one `merge_sequential`.
  - `launchMergeBasic`: blocks of 128 threads (`MERGE_BLOCK`), enough of them for `total`
    outputs, counted with integer ceiling division. Nothing is launched when `total` is 0.
- **(b) Staged through shared memory**, one tile of `128 * perThread` outputs per block.
  - `partitionKernel` stores `bounds[b] = co_rank(min(b * tile, total), A, m, B, n)`
    for every tile boundary `b = 0 .. numTiles`.
  - `mergeTiledKernel`: block b takes `A[bounds[b] .. bounds[b+1])` and the B range that
    makes up the rest of its outputs (found by subtraction). It copies both into one
    `extern __shared__ KV tile[]` with consecutive threads on consecutive elements, has
    one `__syncthreads()`, and then each thread runs `co_rank` on the shared copies and
    `merge_sequential` from shared memory to C. Every thread reaches the barrier: no
    early return.
  - `launchMergeTiled`: two launches, the partition and then the merge, with
    `tile * sizeof(KV)` bytes of dynamic shared memory.

## Files

- Edit `work/kernel.cu` (CUDA, HIP and both emulators): `co_rank`, `merge_sequential`,
  `mergeBasicKernel`, `partitionKernel`, `mergeTiledKernel`, `launchMergeBasic`,
  `launchMergeTiled`. `coRankKernel` and `launchCoRank` are provided, so the tests can
  call your `co_rank` on its own.
- Edit `work/kernel.metal` (`co_rank`, `merge_sequential`, `merge_basic`,
  `merge_partition`, `merge_tiled`; `co_rank_test` is provided) and `work/launch.mm`
  (`encodeMergeBasic`, `encodeMergeTiled`; `encodeCoRank` is provided). Each encode
  function returns false, without encoding anything, when the pipeline's
  `maxTotalThreadsPerThreadgroup` is below 128, or (tiled) when the tile does not fit in
  the device's `maxThreadgroupMemoryLength`. The driver prints anything it skips.
- Do not edit `merge.h` or `main.cpp`.

In Metal, `co_rank` and `merge_sequential` read device memory in `merge_basic` and
threadgroup memory in `merge_tiled`. Metal's address spaces are disjoint and every pointer
names one, so the single `const KV*` helper that serves CUDA cannot. The starter makes
each helper a template on the pointer type, as PyTorch's MPS merge sort does; MLX writes
two overloads instead (ch13-facts C3). The MSL specification also does not restate C++'s
rule that `&&` skips its right operand (ch13-facts C2), so in Metal test the bounds first
and read inside the `if`.

`encodeMergeTiled` puts two dispatches in one encoder. The runtime opens the default
serial kind of encoder, and buffers made by the device are hazard-tracked, so the merge
starts after the partition has written `bounds` (`notes/today/ch02-facts.md` C9).
`setThreadgroupMemoryLength` needs a multiple of 16 bytes (ch05-facts C8): `tile * 8`
always is.

## Commands

- Test: `python3 labs/pmpp.py run 13.1 --backend emu` first, then `python3 labs/pmpp.py run 13.1`.
- The tiled kernel's shared tile: `python3 labs/pmpp.py run 13.1 --backend race`.
- Measure: `python3 labs/pmpp.py run 13.1 --bench` (real devices only).
- The reference solution: add `--solution`.

## What the tests do

Every check is exact equality. A merge copies elements and does no arithmetic, so no
tolerance applies.

1. **`co_rank` alone**, through the provided kernel. On the book's running example
   (keys A = 1 7 8 9 10, B = 7 10 10 12) every k from 0 to 9; the expected values come
   from `std::merge`, not from a table. Then random pairs of arrays (lengths 0 to 40,
   keys 0 to 3, so many ties) at every k.
2. **Both merges** on: the running example; (0, 0), where nothing may be launched and
   nothing may crash; (0, 37); (37, 0); (1, 1) with equal keys; (100, 100) with every key
   equal; (64, 64) with A entirely below B, and the reverse; (1000, 7); (7, 1000); random
   (300, 257) and (4093, 3001), once with keys in [0, 8) and once with distinct keys.
   Real devices add (2^20, 2^20) and (3,000,000, 100,000). Each at `perThread` 1, 3 and 8
   (devices add 16), and at `total` = m + n, (m + n) / 2, 17 and 1, those that are at
   most m + n.

A[i] has value i and B[j] has value 1,000,000,000 + j, so every output says where it came
from. The reference is `std::merge` comparing keys only, which the C++ standard guarantees
to be stable with the first range first (ch13-facts A6). An exact match of `C[0 .. total)`
therefore also checks stability. C starts filled with the sentinel `{INT_MIN, -1}` and has
64 more elements than m + n: an output you never wrote, and anything written at or past
`C[total]`, are both reported.

What each backend can catch:

- `emu` runs the threads one at a time in three fixed orders, fills shared memory with
  garbage for every block, and stops at the first out-of-bounds read or write with the
  source line (AddressSanitizer). A read of `A[-1]` or `B[n]` stops it at once. A missing
  barrier gives garbage outputs.
- `race` runs only the tiled kernel, small: the running example and (300, 257) at
  `perThread` 4 and 2 (2 and 3 blocks of 128 threads), with threads truly concurrent under
  ThreadSanitizer. A missing barrier is reported as a data race on the tile.
- Real devices run the same checks at larger sizes. Nothing in this lab checks their
  reads for bounds, so a wrong bounds test shows there only if it changes an output or
  the device reports a fault.

## Reading a failure

- *co_rank(k=6) returned 4, expected 5 (ties go to A)*: your search gives the answer a
  merge would give if ties went to B, at every k of that case (the hint appears only
  then). Look at which of your two tests allows equal keys.
- *C[6] came from B[1]; a stable merge takes A[4] first (both have key 10)*: right keys,
  wrong order. Either `co_rank` or `merge_sequential` breaks ties the wrong way.
- *C[k] was never written*: a thread that should have produced it did not run, or
  produced fewer outputs than it owns. Count the threads your launch creates.
- *C[k] was written, but only the first total = T outputs may be*: something is limited
  to m + n where it should be limited to `total`.
- *C[k] holds {key -842150451, ...}, which is no input element*: a value read from
  shared memory that no thread had written yet (the emulator fills it with the byte 0xCD).
- *AddressSanitizer: heap-buffer-overflow ... in co_rank*: a read past either end of A or
  B, usually a bounds test that comes after the read it should protect.

## Where this lab differs from the page

- **Integer ceiling division** (ch13-facts A4). Fig. 13.9 line 03 computes the per-thread
  count as `ceil((m+n)/(blockDim.x*gridDim.x))`. Both operands are integers, so the
  division has already discarded the fraction and `ceil` changes nothing: some outputs
  are never produced. Here the per-thread count is an argument and the launcher computes
  the grid from it as `(total + per*128 - 1) / (per*128)`.
- **`__device__`** on `co_rank` and `merge_sequential`. The listings show no qualifier;
  on CUDA and HIP an unqualified function is host-only (ch13-facts, "Already settled").
- **`k_curr`**. The book limits `k_next` only, and relies on the launch for threads past
  the end. Here a thread whose first output is at or past `total` returns.
- **KV pairs** instead of `int`, so stability can be tested, and `const` inputs.
- **`total`**, at most m + n (ours).
- **Exact loading**, not the 2x tile of section 13.6. The book's tiled kernel loads a
  full tile of A and a full tile of B each iteration and uses only half of what it loads
  (p. 318). This lab computes every block's co-rank first, in its own launch (thread 0
  and two barriers do it in Fig. 13.11), so each block loads exactly the elements it will
  use: the alternative on p. 316 and Exercise 3's idea. One tile per block, so there is
  no while-loop and no `A_consumed` bookkeeping.
- **No circular buffer** (section 13.7). Exact loading leaves nothing over to keep.
- In Metal: the template on the pointer type and the explicit bounds guards (above).

## Beyond the book

- The libraries call `co_rank` "merge path" (CUB, rocPRIM) or "merge partition" (MLX,
  PyTorch MPS) (ch13-facts A1). Part (b) is their design: a partition pass, then exact
  loading into shared memory (A7, B2, D1). PyTorch's MPS merge has no partition pass:
  every thread of a block computes the block's partition itself, in registers, which its
  comment calls "same result, no barriers" (D2).
- `cub::DeviceMerge` is documented as unstable, `thrust::merge` as stable, and Thrust's
  CUDA backend calls CUB (A7). If stability matters to you, check what the library
  promises, not what its code happens to do.
- The `total < m + n` mode is how llama.cpp's Metal backend computes top-k for short
  rows: it keeps the top k of each block, then merges the lists pairwise with a kernel
  shaped like Fig. 13.9 that stops at `top_k` (D3). A top-k list is sorted largest
  first, so it merges with the keys negated or the comparison reversed. Module X1 builds
  its top-k as a radix select instead (section X1.4, "Top-k is a selection, not a sort"),
  so nothing there depends on this mode.

## The experiment

Once every test passes, and before you run `--bench`, write down two numbers. The labs
page records them for you.

1. For the basic kernel, how many times faster (or slower) is `perThread` 8 than
   `perThread` 1? At 1, every output pays for two searches of log2 depth, but neighbouring
   threads store to neighbouring addresses. At 8, one search is shared by 8 outputs, but
   at each step of the merge loop the threads of a warp (a SIMD-group on Metal) touch
   addresses 8 elements apart.
2. At `perThread` 8, how many times faster is the tiled kernel than the basic one?

Then run `python3 labs/pmpp.py run 13.1 --bench` and compare. The lesson's model counts
accesses and ignores caches, and each thread of the basic kernel walks its own slice in
order, so a cache may absorb much of the scattering. Neither answer is promised.

What `--bench` does: keys are uniform random 31-bit integers, sorted, with m = n. It grows
m from 2^20 until the basic kernel at `perThread` 8 takes at least 20 ms, or each input
reaches 2^23 elements (or the arrays would pass a quarter of the memory your device
reports), with `total = m + n`. Then it times the basic kernel at `perThread` 1, 2, 4, 8
and 16 and the tiled kernel at 4, 8 and 16 (best of 5 runs after a warm-up), checks every
result against the CPU, and prints milliseconds and GB/s. GB/s is
`2 x total x 8 bytes / time`: reading each element once and writing it once, the least
traffic any merge can make, not what the kernel actually moved. The tiled time includes
its partition launch. Compare GB/s with the bandwidth row lab 5.1 measured on the same
device. A row the device cannot run is printed as skipped, with the reason.
