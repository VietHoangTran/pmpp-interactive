# Lab 9.2: privatised in shared memory, coarsened and run-merged

Book: sections 9.4 to 9.6 (pp. 211-219), Figs. 9.10, 9.14 and 9.15.

Write two 256-bin image histograms that keep a private copy of the bins in shared memory:

- `histogramShared` (Metal: `histogram_shared`), Fig. 9.14. Zero the block's private bins
  with a strided loop, barrier, count `coarseFactor` pixels per thread with interleaved
  indexing and block-scope atomics on the private bins, barrier, then add each non-zero
  private bin to the public bins with a device-scope atomic.
- `histogramRuns` (Metal: `histogram_runs`), Fig. 9.15. The same, plus a run counter per
  thread: consecutive pixels of one thread with the same value are counted in a register
  and cost one atomic when the value changes.

The coarsening factor is a run-time argument, `coarseFactor`. The book uses a compile-time
macro, `COARSE_FACTOR`; an argument lets one build be measured at several factors.

- Edit `work/kernel.cu` (the two kernels) and `work/kernel.metal` (the same two in Metal).
  The launchers below the kernels and `work/launch.mm` are written for you: 256 threads
  per block (on Metal, fewer if the pipeline cannot run 256), and ceil(n / (coarseFactor x
  256)) blocks.
- `provided/` holds the benchmark's two comparison kernels, which you do not edit:
  `refHistogramBasic` (Fig. 9.6, one thread per pixel) and `refHistogramContiguous`
  (Fig. 9.12, shared-memory privatisation with contiguous indexing). They are tested too.
- On Metal, atomic functions take atomic types only. The private bins are a `threadgroup
  atomic_uint` array declared in the kernel body, zeroed with `atomic_store_explicit` and
  read in the commit with `atomic_load_explicit`; `work/kernel.metal` has the vocabulary
  table. The MSL specification shows no example of a threadgroup atomic array declared that
  way; it is the form shipping code uses (`notes/today/ch09-facts.md` C1). If your Metal
  compiler rejects it, use a `threadgroup atomic_uint* [[threadgroup(0)]]` parameter and
  `setThreadgroupMemoryLength`, as PyTorch's MPS histogram does (the parameter in
  `HistogramKernel.metal`, the length set in `HistogramKernel.mm`; D2).
- Test: `python3 labs/pmpp.py run 9.2`. Start with `--backend emu`, then `--backend race`.
- Measure (GPU only): `python3 labs/pmpp.py run 9.2 --bench`.

## What the tests do

Every case compares all 256 counts with the sequential loop of Fig. 9.2 in `main.cpp`,
exactly. The driver zeroes `bins` before every launch.

- Images, width x height: 1 x 1, 255 x 1, 256 x 1, 257 x 1, 64 x 64, 100 x 100; on a GPU
  also 1024 x 1024 and 4096 x 1025. Most leave the last block with fewer pixels than
  threads.
- Coarsening factors 1, 3, 4 and 16.
- Four kinds of image, from fixed seeds: uniform random bytes, a constant image (every pixel
  200), a sky (every row one value, changing slowly from row to row), and stripes (runs of 8
  equal bytes).
- On a GPU every configuration is launched 5 times.

What each backend can and cannot show you here:

- Shared memory nobody zeroed holds `0xCD` bytes on `emu` and `race`, fresh for every
  block, so a missing zeroing loop is wrong there every time. A GPU may hand you zeros, and
  then the tests pass on it. Passing there is not evidence.
- A missing barrier is wrong on `emu` under at least one of its three thread orders, and
  `race` reports it with the lines involved.
- `race` runs a block of 256 threads on its own (it runs blocks larger than 128 threads
  one at a time), so it cannot see two blocks racing in the commit. A plain `+=` there may
  show on your GPU as lost counts.
- Two mistakes pass every test. A zeroing or commit loop that assumes one thread per bin
  (`b < blockDim.x`, or no loop at all) gives the right counts, because the launchers use
  256 threads and there are 256 bins; it is still wrong for any other block size, so write
  the loops as the book does (lab 9.1 tests its pool kernel at 64 and 128 threads). And
  contiguous indexing (Fig. 9.12) gives the right counts: which pixels a thread takes
  changes the speed, not the result. Only the benchmark shows it.

## Before you benchmark

`--bench` times, on 2^26 pixels (8192 x 8192, so a sky row is long; fewer rows if the
device reports too little memory), uniform bytes, the constant image and the sky:
`basic` (provided), `contiguous` (provided, coarse 16), your `shared` at coarse 1, 4, 16
and 64, and your `runs` at coarse 16. It prints Gpixels/s for each, best of 5 runs, and the
three comparisons below. Write down first:

1. How many times faster `shared` at coarse 16 is than `basic` on the constant image
   (section 9.4 asks this too).
2. How many times faster interleaved (`shared`) is than `contiguous` at coarse 16 on uniform
   bytes (section 9.5 asks this too). The book's argument is coalescing: in one step, the
   threads of a warp should read neighbouring bytes (p. 216). NVIDIA documents that rule as
   32-byte transactions per warp (A7) and AMD as cache-line requests (B3). Apple documents
   no coalescing rule (C4), so on a Mac this measurement is the only evidence you have.
3. Whether `runs` beats `shared` on the sky, and on uniform bytes (the labs page records
   this prediction).

The numbers are measurements of your device, not of GPUs in general. Zeroing `bins` before
each run is not timed.
