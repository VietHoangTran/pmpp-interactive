# Lab 18.1: frontier BFS with a privatised queue

Book: sections 18.5 to 18.7 (pp. 438-448), Figs. 18.14, 18.15 and 18.17.

Write breadth-first search the way section 18.6 builds it: one launch per level, one thread
per vertex of the previous frontier, and each block collecting the vertices it finds in a
private frontier in shared memory before copying them to the public frontier with one atomic
per block. Then, optionally, fold every level into one launch with grid-wide barriers
(section 18.7).

The graph is in CSR form (p. 427): the out-neighbours of `v` are `dst[srcPtrs[v]]` to
`dst[srcPtrs[v + 1] - 1]`. `level` is an `unsigned int` array; `level[root]` is 0 and every
other entry is `UNVISITED` when your code starts. Afterwards every vertex the root can reach
holds its level, and every other vertex is still `UNVISITED`.

**Part A (required, every backend).**

- `visitVertexAtomically` (Metal: `visit_vertex`), Fig. 18.14. "Still unvisited? Then it is
  mine, at this level" as one compare-and-swap, true for exactly one of the threads that try.
- `bfsPrivatisedKernel` (Metal: `bfs_privatised`), Fig. 18.15. A private frontier of
  `PRIVATE_FRONTIER_CAPACITY` slots and a private counter in shared memory; a block-scope
  atomic on the private counter; an overflow path that appends straight to the public
  frontier when the private one is full; three block barriers; one device-scope atomic per
  block to reserve its range of the public frontier; a commit in which neighbouring threads
  write neighbouring addresses.

**Part B (optional; CUDA, HIP and the emulators).**

- `bfsCooperativeKernel`, Fig. 18.17: the level loop moves into the kernel. The grid is
  smaller than a large frontier, so each thread strides through the previous frontier by
  `grid.num_threads()`, and three `grid.sync()` calls end each level.
- The marked lines of `launchBfsCooperative`: size the grid so that every block is resident
  at once (an occupancy query times the number of SMs, pp. 445-446), and launch it with
  `cudaLaunchCooperativeKernel`.

Metal has no part B: it documents no grid-wide barrier, no cooperative launch and no
occupancy query from which to size a resident grid (`notes/today/ch18-facts.md` C1, ch04
C3 and C5). On Metal the search is one dispatch per level.

## Files and commands

- Edit `work/kernel.cu` (both parts) and `work/kernel.metal` (part A).
- Written for you: `launchBfsPrivatised` in `work/kernel.cu` (zero the counter, launch
  `ceil(numPrevFrontier / BLOCK_DIM)` blocks, read the count back, swap the frontiers), the
  device-support check at the top of `launchBfsCooperative`, and `work/launch.mm` (the same
  per-level loop for Metal: one command buffer per level, `waitUntilCompleted`, then a read
  of the count). You need not edit `launch.mm`.
- `provided/ref_kernels.cu`, `.metal` and `ref_launch.mm` hold the benchmark's baselines,
  which you do not edit: `push` (Fig. 18.6, every vertex every level, with relaxed atomics
  in place of the page's benign race), `frontier` (Figs. 18.12 and 18.14, one public
  atomic per vertex found) and `floor` (an empty kernel). They are tested too.
- `BLOCK_DIM` is 128 and `PRIVATE_FRONTIER_CAPACITY` is 256 (1 KB of threadgroup memory on
  Metal), both defined by the build. The capacity is larger than the block, so a block can
  hold more entries than it has threads.
- Test: `python3 labs/pmpp.py run 18.1`. Start with `--backend emu`, then `--backend race`.
- Measure (GPU only): `python3 labs/pmpp.py run 18.1 --bench`.

On HIP, `<cuda/atomic>` comes from libhipcxx, which this course's HIP build does not rely on
(`notes/today/infra-facts.md` B2), and libhipcxx implements `compare_exchange_strong` with a
weak compare-and-swap at block and device scope (ch18-facts B4). So beside each
`cuda::atomic_ref`, write the legacy call under `#if defined(__HIPCC__)`: `atomicCAS`
(strong, and it returns the old value, not a bool), `atomicAdd` and `atomicExch`, all
relaxed and valid on shared and global memory (infra B1), as in lab 12.1. The
cooperative-groups header has a different path on HIP; the `#if` for it is written for you.

On Metal the only compare-and-exchange is `atomic_compare_exchange_weak_explicit`, which may
fail although the values were equal (ch18-facts C4). After a real failure `expected` holds
the value in memory; after a spurious one it still holds what you passed. Loop on that.
Atomic functions take atomic types only, so `level` is `device atomic_uint*` and the private
counter is a `threadgroup atomic_uint`, zeroed with `atomic_store_explicit`. `vertex` is a
reserved word in MSL; the files call it `vertexId`.

## Where this differs from the page

- **Line 29 of Fig. 18.15** (line 35 of Fig. 18.17). On overflow the page puts the private
  counter back to the capacity with a plain store, `numCurrFrontier_s =
  PRIVATE_FRONTIER_CAPACITY;`, while other threads of the block may be doing `fetch_add` on
  it. That is a data race in CUDA's memory model, which makes it undefined behaviour,
  whatever the hardware does (ch18-facts A7). Store through the same `atomic_ref` (relaxed),
  or drop the store and clamp the count with `min` after the barrier. Our enumeration of
  every interleaving for small cases found the arithmetic right with the atomic store (E1,
  our check, not a vendor's). On Metal the counter is an atomic type, so only the atomic
  store can be written.
- **The sentinel.** The page suggests `UNVISITED` be "`cuda::std::numeric_limits<unsigned>`"
  (p. 433), which names the trait without `::max()` (A9).
- **`UINT_MAX`.** Here `UNVISITED` is `UINT_MAX` from `<climits>`: the same value, and it
  needs no library, where `<cuda/std/limits>` exists on CUDA and in libhipcxx but not in the
  CPU emulator. Metal spells it `numeric_limits<uint>::max()` (C5).
- **Hoisted `__shared__`.** Fig. 18.17 declares its shared variables inside the level loop.
  A `__shared__` variable lives as long as the block, not one pass of a loop, so declare
  them at the top of the kernel; the reset at the top of each level is what empties the
  private frontier.
- **Six arguments.** The page calls `cudaLaunchCooperativeKernel` with four arguments. CUDA
  accepts that (its last two parameters have defaults, A1); HIP does not (B1). Pass all six:
  `(..., kernelArgs, 0, 0)`.
- **64-bit ranks.** `grid.thread_rank()` and `grid.num_threads()` return `unsigned long long`
  on CUDA (A4) and 32-bit values on HIP (B2). Cast them to `unsigned int` once; the page
  narrows them silently in the loop header.
- **Guard words on the public frontier.** No listing checks the public frontier's capacity,
  and the page shows no allocation for it (notes "Unclear in the PDF"). Here each public
  frontier buffer has room for `numVertices` entries, which is all a search that visits each
  vertex once can need, and the driver writes guard words after both buffers and checks
  them after every search. A vertex added twice can push a frontier past its end; the guard
  words say so.
- **Raw pointers.** `srcPtrs`, `dst` and `numVertices` are passed as they are; the book's
  `CSRGraph` struct is never defined. Every listing in the chapter is called `bfs_kernel`;
  the names here are ours.

## What the tests do

Every case compares `level` with a sequential BFS in `main.cpp` exactly (levels are whole
numbers, so the tolerance is zero), and checks the guard words after both public frontier
buffers. For the one-launch-per-level searches it also checks the number of launches and
each level's frontier size against the sequential search's count of vertices at that level:
more means a vertex was added twice, fewer means one was lost.

Graphs: the book's Fig. 18.1 graph from roots 0 and 2 (the levels of Fig. 18.4); the
Exercise 1 graph from vertex 0 (our reading of the drawing on p. 450); a single vertex; a
root with no out-edges; two components; self-loops and repeated edges; a chain (300
vertices on a GPU, 100 on the emulators); a grid with 4 neighbours per vertex (24 x 24 on
the emulators, 1024 x 1024 on a GPU); "fan-out", where 128 children find 512 grandchildren
in one block against 256 private slots, so many threads overflow at once and the commit has
more entries than the block has threads, and a copy with 256 children that puts two blocks
in flight; "fan-out shared", where children compete for the same grandchildren; a
"stale-expected trap", 0 to {1, 2}, 1 to {3}, 2 to {3, 4}, where the thread for vertex 2
loses vertex 3 and must still win vertex 4; a hub of degree 5,000, where one thread
overflows alone; and two R-MAT graphs from fixed seeds (small on the emulators). Part B runs
the same graphs through `launchBfsCooperative`; on `emu` that is 8 blocks of 128 threads,
the emulator's 1,024-thread cooperative limit. Part B's tests are skipped, with a message,
while `launchBfsCooperative` returns `BFS_COOP_NOT_WRITTEN`, on Metal, and on a device that
reports no cooperative launch.

What each backend can show you:

- `emu` runs one thread at a time until it reaches a barrier or its end, in three orders.
  Most missing block or grid barriers are wrong there under at least one order, but not
  all: for some, the emulator's schedule happens to produce the right answer, and only
  `race` reports them. An expected value reused across calls is wrong there,
  and AddressSanitizer stops at an out-of-bounds access in shared or global memory and names
  the line. It cannot see a missing compare-and-swap or the page's plain store at line 29:
  nobody is interrupted between a check and a store.
- `race` runs the threads truly concurrently under ThreadSanitizer and reports two threads
  touching the same memory without synchronisation, with both source lines. That is where
  a missing atomic shows.
- Your GPU shows duplicates or lost vertices only when the schedule produces them, and a
  missing barrier can pass there.

## Before you benchmark

`--bench` builds two graphs with about the same number of edges (shrunk if they would take
more than a quarter of the memory the device reports):

- **grid**: 1024 x 1024, 4 neighbours, root at a corner. 2,046 levels, and no frontier is
  larger than 1,024 vertices: a road network in miniature.
- **rmat**: R-MAT from a fixed seed (the generator and its parameters are ours and are in
  `main.cpp`). A handful of levels with large frontiers: a social network in miniature.

For each graph it times whole searches, best of 5 after one untimed run, with `level` reset
outside the timing, and checks every result before reporting it: `push`, `frontier`, your
`privatised`, your `cooperative` (CUDA and HIP only), and `floor`. `floor` runs the same
host loop as the per-level searches around an empty kernel, once per level of that graph:
zero the counter, launch, read 4 bytes back (on Metal: an empty dispatch, `waitUntilCompleted`,
a read of the count). It is the fixed cost of a level with no search in it. Each row gives
milliseconds, microseconds per level and millions of edges traversed per second.

Write down three numbers first:

1. On **grid**, the share of your privatised search's time that `floor` accounts for
   (`floor` seconds divided by `privatised` seconds). Close to 1 would mean the fixed cost
   of a level dominates, which is what p. 445 says about graphs like road networks.
2. On **rmat**, how many times faster `privatised` is than `frontier`: the relief from
   contention on one public counter that privatisation buys (p. 442).
3. On CUDA or HIP, how many times faster `cooperative` is than `privatised` on **grid**, and
   on **rmat**.

The labs page records (1), (2) and the grid half of (3).

Run `--bench` more than once. A search of 2,046 tiny levels is mostly waiting for the host
and the device to hand work back and forth, and that varies from run to run; if two runs
disagree about the ratio, the difference between them is part of the answer. The book gives
no numbers for any of this. The numbers are measurements of your device, not of GPUs in
general.
