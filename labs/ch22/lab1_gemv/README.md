# Lab 22.1: two ways to cut a decode-time matrix-vector product

Book: section 22.2 (pp. 532-536), above all the criteria on p. 533, and section 22.5
(p. 539) for why decode is a matrix-vector product. Chapter 22 prints no code, so both
kernels, the defect and the benchmark are ours.

At decode time a handful of tokens goes through each linear layer:
`Y[batch x out] = X[batch x in] * W^T`, with W stored (out, in) row-major. That is the
layout and argument order of the project's `launchLinear` (labs 6.4 and 15.6). With a
batch of at most 8 (`MAX_BATCH`), this is a few matrix-vector products that share W
(p. 539), and its cost is reading W once. This lab cuts it into per-warp work in two
ways and measures which way your device prefers.

- **(a) Output-centric, a gather (yours).** One warp owns output column j. Its lanes
  stride along `in` together, reading row j of W contiguously, and each lane keeps one
  running sum per batch row in registers. Each sum is reduced across the warp with
  shuffles, and lane 0 writes `Y[b][j]`. Nobody else writes that element, so there are no
  atomics. The kernel has `out` warps' worth of parallelism.
- **(b) Split-K, input-centric along `in` (provided, with one defect).** `in` is cut into S
  chunks, and one warp takes one (chunk, output) pair. It reduces its chunk's partial
  sums the same way and adds them into `Y[b][j]`, which S warps share. The launch zeroes
  Y first. This is a scatter, and p. 533 says what a scatter needs: several threads may
  update the same output at once, so atomics are needed to avoid lost updates. In
  exchange it has S times as many warps.

The book calls gather usually preferable on CUDA devices, because each thread
accumulates in its own registers and nothing needs an atomic. But it also says gather
against scatter is not the only criterion. It names the amount of parallelism a
decomposition exposes, how easy it is to tell which inputs feed which output, and the
load balance it gives (p. 533). Here the last two are the same for both cuts.
Parallelism is where they differ. The book's own example of paying for atomics to gain
parallelism and balance is SpMV in COO format (pp. 535-536). This lab measures that
trade on layer shapes you will meet in a real model.

## Your part

1. **(a)** Write `gemvGatherKernel` and `launchGemvGather` in `work/kernel.cu`, and
   `gemv_gather` in `work/kernel.metal` plus the dispatch in `encodeGemvGather` in
   `work/launch.mm`.
   - Never a literal 32 for the warp width: `warpSize` in the kernel, the device property
     in the launcher (provided: `deviceWarpSize()`), the pipeline's
     `threadExecutionWidth` on Metal. Chapter 10's `warpSum`, with its own lane-mask type,
     is provided.
   - Blocks (threadgroups) of `WARPS_PER_BLOCK` = 4 warps (SIMD-groups); one warp per
     output.
2. **(b)** Find the defect in `gemvSplitKKernel` (Metal: `gemv_splitk`). Run `emu` first
   and see it pass, then `race`. Write down why `emu` cannot see it. Fix it:
   - CUDA and HIP: `atomicAdd(&Y[...], total)`.
   - Metal: atomic functions take atomic types only, so the kernel's `Y` parameter becomes
     `device atomic_float*`, and the add becomes
     `atomic_fetch_add_explicit(..., memory_order_relaxed)` (infra-facts C1).
     `launch.mm` binds the same buffer whatever its type in the kernel.
3. Benchmark both decompositions, after writing down your prediction (below).

## Files

- `work/kernel.cu` (CUDA, HIP and both emulators): `gemvGatherKernel` and
  `launchGemvGather` are yours. Fix the defect in `gemvSplitKKernel`.
  `launchGemvSplitK` is provided and correct: it zeroes Y with `cudaMemset`, then
  launches `ceil(S x out / 4)` blocks of 4 warps.
- `work/kernel.metal`: `gemv_gather` is yours. Fix `gemv_splitk`. `gemv_zero` is
  provided.
- `work/launch.mm`: the dispatch in `encodeGemvGather` is yours. `encodeGemvSplitK` is
  provided: it dispatches `gemv_zero` and then `gemv_splitk` in the same serial encoder.
- `main.cpp`: the driver and the CPU reference `cpuLinear`. You do not need to edit it.

Names and signatures are fixed. Lab 22.2 links them when this lab's solution is present:

```cpp
void launchGemvGather(const float* X, const float* W, float* Y, int batch, int in, int out);
void launchGemvSplitK(const float* X, const float* W, float* Y, int batch, int in, int out, int splits);
// Metal: kernels gemv_gather, gemv_splitk, gemv_zero. Buffers: X 0, W 1, Y 2, batch 3, in 4, out 5, splits 6.
void encodeGemvGather(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
                      uint32_t batch, uint32_t in, uint32_t out);
void encodeGemvSplitK(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> splitPipeline,
                      id<MTLComputePipelineState> zeroPipeline, id<MTLBuffer> X, id<MTLBuffer> W,
                      id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out, uint32_t splits);
```

Both encode functions encode into the encoder they are given and commit nothing, so
several launches can share one command buffer.

## Commands

- Test: `python3 labs/pmpp.py run 22.1 --backend emu` first (about a minute), then
  `--backend race`, then `python3 labs/pmpp.py run 22.1` for your GPU.
- Measure: `python3 labs/pmpp.py run 22.1 --bench` (real devices only).
- The reference solution: add `--solution`. Start again: `python3 labs/pmpp.py reset 22.1 --yes`.

## What the tests do

- Shapes (batch, in, out): (1, 1, 1), (1, 31, 5), (2, 33, 7), (3, 100, 64), (8, 257, 130),
  (5, 64, 1). On a GPU also (1, 4096, 16), (8, 4096, 4096), (4, 14336, 4096) and
  (8, 1000, 4099).
- Split counts: 1, 2, 3, 7, 32 and one split per warp width of input, `ceil(in / width)`.
  Some counts leave chunks empty (S = 32 at in = 33 gives 17 chunks of 2 and 15 empty
  ones), and some do not divide `in`. Each case name says how the chunks fall.
- The emulators model 32-lane warps only. There the driver also runs the split count for
  a 64-lane warp, `ceil(in / 64)`, which changes which lane reads what and how many chunks
  are empty.
- On the emulators, the split-K cases of the two widest shapes run with fewer outputs:
  (3, 100, 9) and (8, 257, 5). Every shuffle there is a switch between threads, and a
  split-K launch has S x out warps. The chunking is along `in`, which is kept.
- X and W hold exactly their floats, so on `emu` a read past either is an
  AddressSanitizer report naming the line. Y starts as NaN, so an element nobody writes
  fails. It is followed by a guard band of sentinels that must survive.
- Every case launches twice without resetting Y. A split-K launch that does not zero Y
  itself fails the second time.
- Results are compared with `y = x W^T` computed in double on the CPU, within 1e-4 times
  (1 + the sum of |x w| for that output), as in lab 15.6. Never bitwise: atomics add the
  partial sums in any order.
- On a GPU, one more case: split-K at (1, 4096, 16) with S = 64, launched five times. It
  has 64 warps adding into each of 16 outputs, the most contended case. It fails if any
  launch is wrong, and says in how many.

## What each backend can catch

- **`emu`** runs one emulated thread until it reaches a synchronisation point (a
  shuffle, a barrier) or returns, then the next. It catches wrong indices, reads past X
  or W, writes past Y, and a lane that skips a shuffle. It **cannot** see the split-K
  defect: lane 0's read, add and write of `Y[b][j]` has no synchronisation point inside
  it, so no other thread runs in the middle, and the sum is right every time.
- **`race`** runs the threads of a block, and several blocks, at the same time under
  ThreadSanitizer. It reports two unsynchronised accesses to the same memory with their
  source lines. Here both accesses are on the line of the read-modify-write. It can only
  see two warps collide when they run at the same time. It runs blocks of up to 128
  threads concurrently, and it cannot see the same thread index of two blocks that ran one
  after the other on one OS thread (`labs/README.md`, the header of
  `common/include/pmpp/emu/cuda_emu.h`). That is why split-K numbers its warps so that the
  S warps of one output are neighbours: with blocks of 4 warps, S = 4 puts one output's
  4 warps in each block, and S = 16 fills 4 blocks per output. The race cases are
  (2, 64, 8) at S = 4 and S = 16 first, then a few more. ThreadSanitizer reports each pair
  of lines once per run, so a later case can pass with the defect still there.
- **Metal** has no race detector here. A lost update shows only as a wrong sum, and only
  when two SIMD-groups happen to collide in time. On the machine this lab was built on,
  the plain version gave wrong sums in most split-K cases and in 5 of 5 contention
  launches. That is timing, not a guarantee. NVIDIA's rule for its own GPUs is that
  "Reading from or writing to the same memory location without memory fences or
  synchronization results in undefined behavior" (ch10 facts A1). A passing run does not
  make the kernel correct.
- **Nothing on a Mac checks a 64-lane warp.** Both emulators model 32-lane warps, and
  Metal reports its own width, so a literal 32 can pass everywhere here. Check by reading.
  The shuffle loop starts at `warpSize / 2`, the mask has its own type, and the stride is
  `warpSize`.

## Where the backends differ (not in the book)

- **HIP**: the `_sync` intrinsics `static_assert` a 64-bit mask on every target, hence
  `lane_mask_t` (infra-facts B3). `warpSize` is 64 on gfx9 (Instinct) and 32 on later
  AMD parts (ch05 facts B5). Float `atomicAdd` exists, but by default it compiles to a
  safe operation, possibly a compare-and-swap loop, unless the build passes
  `-munsafe-fp-atomics` (infra-facts B1). The course's HIP build does not. None of this
  has been built or run with ROCm here: unverified.
- **Metal**: SIMD-group index and lane come from `[[simdgroup_index_in_threadgroup]]` and
  `[[thread_index_in_simdgroup]]`, because Metal leaves the split of a threadgroup into
  SIMD-groups to the implementation (ch04 facts C2). The SIMD-group width is the
  pipeline's `threadExecutionWidth` (ch03 facts C9), and `launch.mm` makes every
  threadgroup a whole number of SIMD-groups. `simd_sum` adds across a SIMD-group with no
  mask and needs an Apple7 GPU or later (infra-facts C2). Float atomics need
  `device atomic_float*`, Metal 3 and Apple7 or later, and `memory_order_relaxed`
  (infra-facts C1). If `gemv_splitk` does not build on your GPU, the driver reports that
  as the reason each split-K case fails.
- **Zeroing on Metal**: a compute pass has no `cudaMemset`. Lab 10.1 wrote the zero from
  the CPU, which works only before the command buffer is committed and only for one
  launch per command buffer. Here `gemv_zero` is a dispatch of its own, in the same serial
  encoder as `gemv_splitk`. A plain compute encoder runs its dispatches "in dispatched
  order", and device buffers are hazard-tracked, so `gemv_splitk` sees the zeros
  (ch02 facts C9).

## Before you benchmark, predict

Write these down first:

1. On the 4096 -> 16 layer at batch 1, how many times faster the best split will be than
   gather.
2. On the 4096 -> 4096 layer at batch 1, the same ratio. Below 1 means gather wins.

Things to weigh:

- **Threads exposed.** Gather has one warp per output: 16 warps on the narrow layer.
  Split-K has S x 16.
- **How many warps your device can keep resident.** On CUDA and HIP,
  `python3 labs/pmpp.py doctor` reports `multiProcessorCount`,
  `maxThreadsPerMultiProcessor` and `warpSize`, and the benchmark prints the product.
  Metal has no occupancy API, and Xcode's GPU tools show occupancy instead (ch06 facts
  C5, ch15 facts C3).
- **Atomics per output**: S.
- **Bytes of W read**: identical for every S. Each chunk-warp reads its own part of
  row j once.
- **On HIP**, a float `atomicAdd` that may be a compare-and-swap loop (infra-facts B1).

## Reading the output

- Layers: 4096 -> 4096; 4096 -> 16, a narrow layer such as a router; 14336 -> 4096, an
  MLP down-projection. The last two shapes are ours. Batches 1, 4 and 8.
- Kernels: gather (S = 1), then split-K at S = 2, 4, 8, ... up to `in / (8 x width)`, so
  that every chunk holds at least 8 warp-widths of input.
- The kernels take turns: one untimed round, then five rounds, and each kernel keeps its
  best round.
- One timed submission launches the kernel once per copy of W, going through copies that
  together hold at least 256 MB (at most 1024 copies). A launch therefore does not find
  its weights where the previous launch left them. The header line of each layer says how
  many copies. `time (us)` is per launch. For split-K it includes the zeroing
  (`cudaMemset`, or the `gemv_zero` dispatch on Metal), because that is part of the
  launch. CUDA and HIP time the submission on the host clock, around the launches and
  the final synchronisation. Metal uses the command buffer's GPU start and end times.
- `W GB/s` is 4 x in x out bytes divided by the time: the bytes of W a matrix-vector
  product must read, not everything the memory system moved. The labs page compares it
  with lab 5.1's measured bandwidth, the speed limit for a matrix-vector product.
- `warps` is S x out. The last line of each batch names the best S and its speed
  relative to gather.
- Rows go to the report as `gemv` rows: `{kernel, splits, batch, in, out, seconds,
  weight_gb_per_s}`.

Then read your numbers against p. 533. Where does gather's lack of atomics win, and
where does split-K's extra parallelism? Which shape turns the answer around, and why? The
numbers are measurements of your device at these shapes. They are not properties of the
kernels.

## Go further

- **Where a real backend draws the line** (ours, from ch22-facts D6): llama.cpp has
  separate small-batch matrix-vector kernels and chooses them by the number of tokens in
  the batch. On CUDA and HIP that is `mmvf` and `mmvq` up to 8 tokens, with per-GPU
  exceptions. On Metal it is `mul_mv_ext` for 2 to 8 tokens. Lab 22.2's benchmark adds
  this lab's gather as its small-batch path for batches up to 8, when this lab's solution
  is present.
- **A third cut, without atomics** (ours): give each output a whole block of S warps.
  Each warp reduces its chunk, and the block combines the S partial sums in shared memory
  with chapter 10's two-stage reduction. This exposes the same parallelism as split-K
  for S up to the warps in a block, and needs no atomic and no zeroing. Write it and add
  it to the benchmark.
- **Determinism**: with atomics, the S partial sums of an output are added in whatever
  order the warps finish, so the last bits of Y can change from run to run. Gather has
  one order. Count distinct results over 20 runs, as lab 10.1 does, and think about what
  that means for greedy decoding.
