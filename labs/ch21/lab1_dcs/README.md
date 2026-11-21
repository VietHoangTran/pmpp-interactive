# Lab 21.1: direct Coulomb summation, scatter, gather and coarsened

Book: sections 21.2 to 21.4 (pp. 515-522), Figs. 21.5, 21.6, 21.8 and 21.10, and
Exercise 1 (p. 528).

One slice of an electrostatic potential map, computed three ways. Every point of the slice
gets the sum, over all atoms, of charge / distance. The atoms reach the device in chunks
small enough for constant memory, and each launch adds one chunk's contribution
(p. 517). The physics does not matter here. What matters is the order of decisions the
chapter takes: scatter (one thread per input, atomics on every output), gather (one thread
per output, no atomics), coarsening (one thread per four outputs, operands reused from
registers), and which four outputs a coarsened thread owns, which decides whether its
writes are coalesced.

## Your part

Three kernels and their launch functions, in `work/kernel.cu`:

1. The `__constant__` array for one chunk: `CHUNK_MAX` atoms of four floats (x, y, z,
   charge).
2. `dcsScatterKernel` and `launchDcsScatter`: Fig. 21.5. One thread per atom of the chunk,
   in 1D blocks of `BLOCK_X` threads. Each thread walks the whole slice and adds its atom
   to every point with a float `atomicAdd`. The listing has no atom count; your kernel
   needs one, and must leave if its atom is past the end of the chunk.
3. `dcsGatherKernel` and `launchDcsGather`: Fig. 21.6. One thread per point, in
   `BLOCK_X x BLOCK_Y` blocks. It sums the chunk in a register and adds the sum to its
   point with one `+=`. Test `i` and `j`: the slice is not padded to whole blocks.
4. `dcsCoarsenedKernel` and `launchDcsCoarsened`: Fig. 21.10. Each thread owns `COARSEN`
   (four) points of a row, `blockDim.x` apart: thread `t` of a block takes points
   `t`, `t + blockDim.x`, `t + 2 blockDim.x` and `t + 3 blockDim.x` of the block's span of
   `4 blockDim.x` points. Read each atom once, share `dy*dy + dz*dz` and the charge across
   the four points, and test each point against the end of the row before you write it.

Each launch function is the host loop of p. 517, which is Exercise 1: copy at most `chunk`
atoms into the constant array with `cudaMemcpyToSymbol`, launch, repeat until every atom
has been added. The last chunk is usually partial. `dcs.h` gives the contract:

- the grid is `nz x ny x nx` floats, row-major, x fastest, as a device pointer;
- the atoms are `numatoms x 4` floats in **host** memory;
- the launch **adds** to every point of slice `k` and leaves the other slices alone;
- `1 <= chunk <= CHUNK_MAX`.

The same three kernels go in `work/kernel.metal` (`dcs_scatter`, `dcs_gather`,
`dcs_coarsened`) and their chunk loops in `work/launch.mm` (`encodeDcsScatter`,
`encodeDcsGather`, `encodeDcsCoarsened`).

## Files

- Edit `work/kernel.cu` (CUDA, HIP and both emulators), `work/kernel.metal` and
  `work/launch.mm`.
- `dcs.h`: the sizes and the launch prototypes. `CHUNK_MAX` is 1,024 atoms, 16 KB of
  constant memory per file; `BLOCK_X` is 64 and `BLOCK_Y` 4; `COARSEN` is 4. Do not
  hard-code these values.
- Provided, not edited: `../shared/coarse_adjacent.cu`, `coarse_adjacent.metal` and
  `coarse_adjacent_launch.mm`. That is Fig. 21.8 with its line 05 corrected (see below):
  each thread owns four **adjacent** points. It is the baseline your strided kernel is
  timed against, and it has its own constant array, `atoms_adj`, so the two cannot clash.
  It is tested too.
- `main.cpp`: the tests, the CPU reference and the benchmark. You do not edit it.

## Commands

- Test: `python3 labs/pmpp.py run 21.1 --backend emu` first, then `--backend race`, then
  `python3 labs/pmpp.py run 21.1` for your GPU.
- Measure: `python3 labs/pmpp.py run 21.1 --bench` (real devices only).
- Start again: `python3 labs/pmpp.py reset 21.1 --yes`.

## On Metal

- There is no symbol to copy to. All the atoms sit in one buffer of `float4`s, 16 bytes
  per atom, and a chunk is that buffer bound again at the chunk's first atom:
  `setBuffer:atoms offset:16 * start atIndex:1`. The kernel takes it as
  `constant float4* atoms`. Apple documents no constant cache behind that address space
  and no size limit for it (`notes/today/ch21-facts.md` C1).
- Metal's atomic functions take atomic types only. The scatter kernel declares the grid
  `device atomic_float*` and adds with
  `atomic_fetch_add_explicit(&grid[p], v, memory_order_relaxed)`; the gather and
  coarsened kernels declare the same buffer `device float*`. Float atomics need an
  Apple7 (M1) or later GPU (C2).
- The dispatches of one plain compute encoder run in the order they were encoded and see
  each other's writes (`notes/today/ch02-facts.md` C9), so the chunk loop needs no
  barrier.
- Dispatch the coarsened kernel with `dispatchThreadgroups:threadsPerThreadgroup:`, whole
  threadgroups of `BLOCK_X x BLOCK_Y`, `ceil(nx / (4 x BLOCK_X))` of them across.
  `dispatchThreads` trims the threadgroups at the edge of the grid (ch05-facts C5), and a
  trimmed threadgroup's missing threads own points that nobody would compute. Read the
  stride from `[[dispatch_threads_per_threadgroup]]`, the size you dispatched. Scatter and
  gather can use `dispatchThreads` with exactly one thread per atom or per point.

## On HIP

The same `kernel.cu` is written to build for HIP unchanged (unverified: no HIP machine has
built it yet), because it uses only what `pmpp/vendors/hip.h` carries over unchanged: threads and blocks, constant memory, a legacy atomic and runtime calls;
`cudaMemcpyToSymbol` is mapped to `hipMemcpyToSymbol`. (Warp `_sync` intrinsics,
cooperative groups and half-precision headers would not port unchanged.) Float `atomicAdd` on AMD compiles to a safe operation, possibly a
compare-and-swap loop, unless the build passes `-munsafe-fp-atomics`, and RDNA1 and CDNA1
have no hardware float atomic add (`notes/today/infra-facts.md` B1, ch21-facts B1). The
course's HIP build does not pass the flag, so your scatter measurement on AMD is of the
safe form. AMD documents no size for constant memory; its constant cache (the scalar
cache) helps only when every lane of a wavefront reads the same address, which the gather
kernels do and the scatter kernel does not.

## What the tests do

Every case runs one full slice through your launch function and compares each point of
slice `k` with a sum in double in `main.cpp`. The tolerance at each point is
`1e-4 x (sum over atoms of |charge| / distance)`: it allows for Metal's fast math
(ch21-facts C3) and for float atomics adding in no fixed order, so that scatter's sums
can differ from run to run in the last bits (A6). Never bitwise.

- Slices, `nx x ny`: 37 x 11 and 300 x 5. Neither is a multiple of `4 x BLOCK_X`, so the
  last block of every row overhangs; 300 is one whole span of 256 points and 44 more.
- 150 atoms in chunks of 64: three launches, of 64, 64 and 22 atoms. Also all 150 in one
  launch of a `CHUNK_MAX` chunk, and a single atom.
- Slice 1 of 3 and slice 2 of 3: every other slice holds a sentinel that must survive bit
  for bit. Slice 2 is the last, so a write past the end of a row there leaves the
  allocation.
- Charges uniform in [-1, 1], both signs. Every atom's z is half-way between two slice
  planes, so no distance is zero. Spacing 0.5, and 0.25 in one case.
- On a GPU also 1000 x 333 with 2,000 atoms in chunks of 1,024.
- All four kernels (your three and the provided one) on every case, and a check that
  gather and coarsened agree with each other within the same tolerance.

The grid is allocated with exactly `nz x ny x nx` floats, so on `emu` a write past its end
stops the run with an AddressSanitizer report naming the line.

## What each backend can catch

- `emu` runs one thread at a time and never interrupts one in the middle of a `+=`, so it
  **cannot see a missing atomic**: scatter with a plain `+=` passes there. It catches wrong
  indices, a missing bound (AddressSanitizer), stale atoms from a missing atom-count test,
  `=` where `+=` belongs (the three-launch cases), and a missing slice offset (the
  sentinel cases).
- `race` runs the threads truly concurrently under ThreadSanitizer. Scatter with a plain
  `+=` is reported there with its source line, whether or not the sums came out wrong,
  and so is a coarsened kernel whose threads own overlapping points. It has no bounds
  checking.
- A GPU shows a missing atomic only as wrong sums, and only when two threads happened to
  collide in time.
- The 37-wide cases cannot see a wrong `dx1`, `dx2` or `dx3` in the coarsened kernel: a
  37-point row holds only the first point of each thread. The 300-wide cases can.

## Before you benchmark, predict

Write these down first; the labs page records them.

1. **Scatter against gather** (`p-scatter`): how many times faster will gather run than
   scatter on comparison A on your GPU? The book says atomics "significantly" slow the
   scatter kernel and gives no number (p. 518); no vendor gives one either (ch21-facts
   A6). Think about what every thread of a chunk is doing at the same moment.
2. **Strided against adjacent** (`p-strided`): how many times faster will your strided
   kernel run than the provided adjacent one at chunk 64, and will that gap grow or
   shrink by chunk 1,024? On NVIDIA, the documented rule counts 32-byte sectors per warp:
   the four write statements of the adjacent kernel touch 64 sectors, those of the
   strided kernel 16 (ch21-facts A8, E2). But each launch reads and writes each point
   once, while the atom loop runs `chunk` times per point (E3). Our own prediction,
   unverified, is that the gap shrinks as the chunk grows. AMD documents grouping
   a wavefront's addresses into 64- or 128-byte lines (B3, E2), which predicts the line
   counts but not the time; there the wavefront can be 64 lanes wide and the float atomics
   depend on a compiler flag (B1). Apple documents no coalescing rule and no cost for its
   atomics (C2, C4).

## The benchmark

Real devices only. Each time is one whole slice, every chunk included, the best of
`--reps` runs (5 by default), with the kernels taking turns. On CUDA and HIP that includes
the `cudaMemcpyToSymbol` copies; on Metal the atoms are already in a buffer and a chunk is
a rebinding.

- **A**, scatter against gather: a 256 x 256 slice, 1,024 atoms, chunk 256. Kept small
  because scatter has only one thread per atom of a chunk and each of them walks the
  whole slice.
- **B**, gather, `coarse_adjacent` and `coarse_strided` (yours): a 1024 x 1024 slice, 4,096
  atoms, at chunk 64, 256, 512 and 1,024. 512 atoms are 8 KB, the constant-cache working
  set per SM that NVIDIA documents (A5); whether anything changes there is unverified, and
  this is where you would see it.

Each row reports the kernel (`scatter`, `gather`, `coarse_adjacent`, `coarse_strided`),
`nx`, `ny`, the atoms, the chunk, seconds per slice, atom-point pairs per second, and the
group width: `warpSize` on CUDA and HIP, the `dcs_coarsened` pipeline's
`threadExecutionWidth` on Metal. The explorers on the lesson's pages 21.2 and 21.5 offer
Metal's recorded width under "Apple". The numbers are measurements of your machine, not
properties of the kernels.

## Optional, on CUDA: see the sectors

Nsight Compute's memory tables report sectors per request; for 32-bit accesses by a full
warp, 4 is the optimum (ch21-facts A8). Profile `dcsCoarsenedKernel` and
`dcsCoarseAdjacentKernel` and compare their global stores. (Not run here: the machine
that wrote this lab has no CUDA device.)

## Where this lab differs from the book's pages

- **Fig. 21.8, line 05.** As printed, `i = blockIdx.x * blockDim.x*COARSEN_FACTOR +
  threadIdx.x` adds an unscaled `threadIdx.x` while each thread writes `i` to `i+3`, so
  neighbouring threads own overlapping points and the last `3 x blockDim.x - 3` points of
  each block's span are written by nobody (notes, "Unclear in the PDF"; ch21-facts E2).
  The provided baseline scales `threadIdx.x` by the factor, as the text on p. 521
  describes. The same line 05 is correct in Fig. 21.10, whose ownership is strided.
- The slice is an `int k`, and its z is `k x gridspacing`. Fig. 21.6 passes a float `z`
  and computes `k = z / gridspacing`, which truncates; Figs. 21.3 and 21.4 index the slice
  in two other ways.
- The sizes are `nx` and `ny`, where the book passes a `dim3 grid`.
- None of the book's kernels tests a bound, and Fig. 21.5 has no atom count at all.
  Fig. 21.9 draws a grid padded to whole blocks, which the text never discusses; this
  grid is not padded, so every kernel here tests its bounds.
- The chapter does not name the call that fills constant memory (notes, Exercise 1); it
  is `cudaMemcpyToSymbol`.
- The book's `atomicAdd` on `float` is current CUDA. NVIDIA now calls it a legacy atomic
  and recommends `cuda::atomic_ref`, which chapter 9 used (infra-facts A1, A2).

## Go further

PyTorch makes this chapter's choice for the gradient of an embedding table, and makes it
differently on two backends. On CUDA (and ROCm) it sorts the token ids, which bins the
rows, then gives each thread one output and writes once: gather, deterministic. It takes
a fused atomic path only when a heuristic says the gather would have poor parallelism and
deterministic algorithms were not requested. On MPS the Metal kernel is Fig. 21.5's pattern, one thread per input and an
atomic add into the gradient, and PyTorch raises an error for it in deterministic mode
(ch21-facts D1, D2). From what you measured in comparison A and what A6 says about
the order of float atomics, which of the two would you ship, and when?
