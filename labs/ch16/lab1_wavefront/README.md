# Lab 16.1: a tiled wavefront, square tiles then sheared tiles

Book: section 16.6 (pp. 384-389, Figs. 16.8 to 16.12), section 16.7 (pp. 389-396,
Figs. 16.15 to 16.20), and the exercise on p. 386.

The table is the Smith-Waterman scoring matrix. `H` is (L_seq + 1) x (L_seq + 1) ints,
row-major, with row 0 and column 0 zero. Sequence `a` indexes rows, `b` columns:

    H[r][q] = max(0, H[r-1][q-1] + s, H[r][q-1] + SW_GAP, H[r-1][q] + SW_GAP)
    s = SW_MATCH if a[r-1] == b[q-1], else SW_MISMATCH

`sw.h` holds the book's example values (+3, -3 and a gap of 2, stored as -2; p. 382).
The book does not explain them, and nothing in this lab depends on them. A cell needs
its north, west and north-west neighbours, so each anti-diagonal is a wavefront: its
cells can all run once the two anti-diagonals before it are done.

Provided: the CPU reference (in `main.cpp`) and a baseline in `../shared/` that launches
once per anti-diagonal, one thread per cell, every input read from global memory:
2 L_seq - 1 launches (p. 384; the book leaves its code as an exercise). You write two
tiled versions that need far fewer launches.

## Part 1: square tiles (section 16.6)

One W x W tile per block of W threads, held in shared memory, and one launch per
*front* (an anti-diagonal of tiles): 2T - 1 launches for T = ceil(L_seq / W) tiles per side.

- `max4`, `sqLoadN`, `sqLoadW`, `sqLoadNW`: Fig. 16.10 with a row stride. Tile element
  (row, col) is `swTile[row * stride + col]`, with `stride = W + pad`.
- `swSquareKernel(H, a, b, L, d, firstRow, stride)`, L = L_seq + 1: the block's tile row is
  `firstRow + blockIdx.x` and its tile column is `d` minus that. In the compute loop the
  thread index is the tile row. There are 2W - 1 steps; a thread computes its cell only
  when the cell is inside the tile and inside the table, and every thread passes one
  `__syncthreads()` after every step, outside that test.
- `sqStoreTile`: here the thread index is the tile column, so each tile row reaches `H`
  as one contiguous, coalesced piece (p. 389).
- `launchSwSquare(H, a, b, Lseq, W, pad)`: exactly the tiles of each front (the exercise
  on p. 386; the book launches T blocks every time and lets the spare ones exit),
  blocks of W threads, `W * stride` ints of dynamic shared memory.

## Part 2: sheared tiles (section 16.7)

Shift each row of a tile one column further left than the row above (a shear with
factor m = -1, Fig. 16.14). Every step inside a tile is then a full anti-diagonal of W
cells, and a tile takes W steps instead of 2W - 1. The price: a tile row must now run
two tile columns behind the row above, so there are 3T - 1 fronts, and the edges of the
table cut tiles.

- `hyInitTile`: zero the tile, then a barrier (Fig. 16.19). Cells that fall outside the
  table are never computed, but their slots are read as neighbours.
- `hyLoadN`, `hyLoadW`, `hyLoadNW`: Fig. 16.17. The tile is stored as a square, the cell
  of step `q_tile` in column `q_tile`, so the neighbours are not where they were. The
  book derives the north one and leaves the other two to you (p. 394).
- `hyStoreTile`: Fig. 16.18, with the shear, skipping cells outside the table on every side.
- `swHyperKernel`: Fig. 16.16, same parameters as the square kernel.
- `launchSwHyper`: exactly the tiles of each of the 3T - 1 fronts.

Write all index arithmetic in `int`. The book's `const int tile_col = d - blockIdx.x`
subtracts unsigned values and counts on the conversion to `int` giving a negative
number, which C++17 leaves to the compiler (`notes/today/ch16-facts.md` A3).

## Metal

`kernel.metal` holds `sw_square` and `sw_hyper`. The tile is a kernel parameter,
`threadgroup int* swTile [[threadgroup(0)]]`, handed to `static inline` helpers with the
CUDA names: an MSL helper cannot declare threadgroup memory itself (ch16-facts C3). The
barrier is `threadgroup_barrier(mem_flags::mem_threadgroup)`.

In `launch.mm`, `encodeSwSquare` and `encodeSwHyper` write one dispatch per front into
the encoder they are given: `setBytes` with that front's `SwParams`, then
`setThreadgroupMemoryLength` (a multiple of 16 bytes, ch05-facts C8), then the
threadgroups. The encoder runs its dispatches in order and the buffers are
hazard-tracked, so each front sees what the earlier ones stored (ch02-facts C9). That
is Metal's form of the book's relaunch between fronts.

## Files

- `work/kernel.cu`: the helpers, `swSquareKernel`, `swHyperKernel`, `launchSwSquare`,
  `launchSwHyper`.
- `work/kernel.metal`: `sw_square`, `sw_hyper` and their helpers.
- `work/launch.mm`: `encodeSwSquare`, `encodeSwHyper`.
- Read-only: `main.cpp` (tests, benchmark, CPU reference), `sw.h`, `CMakeLists.txt`,
  `../shared/baseline.cu`, `../shared/baseline.metal`, `../shared/baseline_launch.mm`.

## Commands

    python3 labs/pmpp.py run 16.1 --backend emu     # start here
    python3 labs/pmpp.py run 16.1 --backend race    # threads at the same time, under ThreadSanitizer
    python3 labs/pmpp.py run 16.1                   # every backend found
    python3 labs/pmpp.py run 16.1 --bench           # also benchmark (real devices only)

## What the tests check

Every cell of the table, row 0 and column 0 included, must equal the CPU reference
exactly, and 256 ints allocated after the table must stay untouched. Before each run the
interior of the table holds 1234567, so a cell nobody wrote is reported as "never
written". Both kernels with pad 0 and 1, and the baseline:

| Backend | L_seq | W |
|---|---|---|
| emu | 1, 2, 7, 8, 9, 31, 33, 50 | 3, 4, 5, 8, 16 |
| race | 9, 20 | 4, 8 |
| metal, cuda, hip | 1, 31, 32, 33, 100, 1000, 2049 | 8, 16, 32, 64 |

On a real device, a W that needs more threads or more shared (threadgroup) memory per
group than the device reports is skipped with a message.

Inputs for each length: random sequences over 4 symbols (3 seeds), two identical
sequences, two with no match at all (an all-zero table), and a pair whose first column
matches often (`b[0]` is `a`'s most frequent symbol). The last one is there for the
sheared kernel: the left edge of the table cuts sheared tiles, and a cell in column 1
reads neighbours from tile slots that no thread computes. On the emulator, shared memory
nobody wrote holds `0xCD` bytes, a large negative int, and the maximum with 0 hides most of
it: a missing `hyInitTile` shows only where a column-1 cell's match should win, which this
pair provides. On a GPU nothing promises what an unwritten slot holds, so the same defect
may pass there, on some runs or on all of them: rely on the emulator for this one. On the
emulator the three longest lengths run three of these inputs (random seed 1, identical,
first column), because the emulator pays for every barrier of every thread.

Each of these defects, alone in an otherwise correct solution, fails on `emu`: the
barrier removed from the step loop (either kernel); `<` instead of `<=` for the sheared
tiles' last column; the `hyInitTile` call removed; `q >= 1` dropped from the sheared
compute test; `sqLoadNW` without its `q_tile == 0` case; one tile short in each front's
block count (either kernel); and `stride = W + 1` with only `W * W` ints of shared memory.
So does a barrier moved inside the bounds test, in either kernel, under all three thread
orders. Three of these stop the run with an AddressSanitizer report that names the line
(the dropped `q >= 1`, the missing `q_tile == 0` case, the short shared memory); the others
show as wrong cells.

## Before you benchmark

The benchmark grows L_seq from 1024, doubling, until the square kernel at W = 32 takes at
least 20 ms or L_seq reaches 8192. It then times the baseline, and both kernels at W = 8,
16, 32 and 64 (where the device allows) with pad 0 and 1. Each row shows the launches,
the best time and GCUPS (cell updates per second / 1e9); the last lines are the three
ratios below, measured. Run it at least twice: a difference that does not survive a
second run is noise. Write down three numbers first:

1. **How many times faster are square tiles at W = 32 than the baseline?** For: 2T - 1
   launches instead of 2 L_seq - 1, and neighbours from shared memory. Against: a front
   holds at most T tiles, one block of W threads each, and inside a tile a step has
   between 1 and W cells to compute, about W/2 on average (Eq. 16.3).
2. **Sheared time divided by square time, at W = 32, unpadded.** For: W steps per tile
   instead of 2W - 1, every thread busy in every step (Eq. 16.4). Against: 3T - 1 launches
   instead of 2T - 1, at most about T/2 + 1 tiles per front instead of T, extra incomplete
   tiles at the edges (ch16-facts E2), and every step writes and reads a column of the
   stored square, W ints apart from one thread to the next.
3. **Sheared time with one gap per row divided by without, at W = 32.** Under NVIDIA's
   bank model (32 banks of 4 bytes), the unpadded column is a 31- to 32-way conflict at
   W = 32, and one gap per row removes it (ch16-facts E1, our arithmetic on that model).
   AMD has 32 or 64 banks depending on the architecture (B3). Apple documents no
   threadgroup-memory banks at all (infra-facts C13), so on Metal this is a measurement
   with no vendor model behind it.

Then say which way padding moves the **square** kernel. Its accesses run along an
anti-diagonal of the tile, W - 1 ints apart from one thread to the next, which the same
NVIDIA model finds conflict-free unpadded; one gap per row makes that W ints, which the
model finds conflicting, up to 32-way at W = 32 (E1). The book pads only the sheared tile.

## Where this differs from the book

- All index arithmetic is `int` (above).
- Each launch covers exactly the tiles of one front, through `firstRow` (the exercise on
  p. 386).
- Padding is one gap per tile row, through `stride`. The book's `pad()` macro inserts a
  gap every `NUM_BANKS` ints, which is one per row only when W equals the number of
  banks; at W = 64 on NVIDIA it leaves a 2-way conflict where one gap per row leaves none
  (`notes/ch16.md`, "Unclear in the PDF"; ch16-facts E1).
- The sheared store tests `q >= 1`. Fig. 16.18 tests only `q < L` with an unsigned `L`,
  so a negative `q` fails by wrapping around; with an `int` `L` it would write into the
  row above.
- One gap constant. The book names two penalties with the same value, and attaches the
  names one way in Eq. 16.2 and the other way in Fig. 16.6 and the kernel (`notes/ch16.md`,
  "Unclear in the PDF"); with equal values it makes no difference.
- The listings leave `MATCH`, `MISMATCH`, `DELETION`, `INSERTION` and the sequence type `T`
  undefined; here they are the constants of `sw.h` and `int`.

## Beyond the book

- Exercise 6 replaces `max4` with the DPX intrinsic `__vimax3_s32_relu(nw + s, w + gap,
  n + gap)`, which gives the same value for every input. It is CUDA only; NVIDIA's
  feature table lists it as native on compute capability 9.0 and 10.x and as several
  instructions elsewhere, and NVIDIA's own CCCL source reports a compiler bug in
  dependent chains of it before CUDA 13.4 (ch16-facts A2). HIP has no DPX functions (B2).
  Metal's closest form is `max(max3(a, b, c), 0)` (Metal 2.1 and later, C2). Not tested
  here; if you try it, keep `max4` as the fallback and compare the two.
