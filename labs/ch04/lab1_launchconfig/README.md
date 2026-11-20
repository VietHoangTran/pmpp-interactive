# Lab 4.1: ask the device, then size the launch (project milestone M1, second half)

Book: section 4.8 (pp. 87-89) for the queries, section 4.7 (pp. 85-86) for the slot
arithmetic, section 4.4 (p. 75) for how a block is cut into warps. The kernel is lab 3.3's
naive GEMM with a linear layer's names; it is given.

Lab 3.3 launched that kernel with 16 x 16 blocks because the book's figure does. Here you
query the device yourself and derive the block and the grid from what it reports. This
completes M1.

## CUDA, HIP and the emulator: `work/kernel.cu`

Three functions. The kernel `linearKernel`, the launch statement `launchLinearWith` and
the functions at the end of the file are given.

1. `queryDevices(info, maxDevices)`: the loop of p. 88. Ask for the device count, then for
   each device's properties, and copy the fields `DeviceInfo` names (`launch_config.h`,
   given, not edited). Check both return codes. The page does not.
2. `chooseLaunch(d, rows, cols, &grid, &block)`: a pure function, no CUDA call inside, so
   the tests can hand it descriptions of devices you do not own. It must return a
   configuration that satisfies every rule below, or `false` if none exists.
3. `launchLinear(X, W, Y, batch, in, out)`: query, choose, launch through
   `launchLinearWith`. The driver reads back the configuration that call recorded.

The rules for `chooseLaunch`. The tests check each one on its own and name the one that
failed:

- the block is within `maxThreadsPerBlock`, and within `maxThreadsDim` on each axis;
- its thread count is a whole number of warps (`warpSize`, never a literal);
- blocks of that size can fill the thread slots of an SM. Section 4.7 gives two ways to
  fall short: the block is so small that the block slots run out first, or its size does
  not divide the thread slots. A slot field of 0 means the device did not report it, and
  the rule that needs it is skipped;
- the block is no taller than `rows`, and `block.z` is 1;
- the grid covers `rows x cols` with no block to spare, `grid.z` is 1, and the grid is
  within `maxGridSize`.

Several blocks satisfy all of this. Which one you return is your design decision, and the
benchmark tells you what it was worth. These rules are this course's starting point for a
kernel with one thread per output, not a vendor's recommendation.

Where p. 89 no longer matches the headers (`notes/today/ch04-facts.md`, A16 and A18):

- `devProp.clockRate` was removed from `cudaDeviceProp` in CUDA 13.0. A file that reads it
  does not compile there. HIP still has the member. Do not read it.
- The page describes `regsPerBlock` as the registers of an SM. That figure is
  `regsPerMultiprocessor`. `regsPerBlock` is the limit for one block.
- `maxThreadsPerMultiProcessor` and `maxBlocksPerMultiProcessor` are not on the page. They
  are the thread slots and block slots of section 4.7.
- The page's loop compares an `unsigned int` index with the `int` count and ignores both
  return values. NVIDIA's own version of the loop uses `int`, and its guide says to check
  every return value.

On HIP the same file is built through `pmpp/vendors/hip.h`: every field `DeviceInfo` names
exists in `hipDeviceProp_t` with the same spelling (`notes/today/ch05-facts.md`, B2). Some
mean something else there (ch04 facts B9): `major` and `minor` are an AMD architecture
number, `maxBlocksPerMultiProcessor` is computed from the thread slots and is not a limit
of its own, and AMD's reference page says some fields can read 0. That is why a 0 skips
its rule instead of dividing by it.

## Metal: `work/kernel.metal` and `work/launch.mm`

Metal splits the answer in two. The device knows its name and its limits for the simplest
possible kernel. The width of a SIMD-group and the real thread limit belong to a compiled
kernel, so they are read from the pipeline state. The device has no property for a core
count, a register count or the threads resident at once (facts C1, C5).

1. `kernel.metal`: the probe kernel `simd_probe`. Each thread records its
   `[[simdgroup_index_in_threadgroup]]`, its `[[thread_index_in_simdgroup]]` and
   `[[threads_per_simdgroup]]`. On CUDA the first two are arithmetic on the thread index,
   because the cut of a block into warps is documented. Metal's specification calls the
   cut of a threadgroup into SIMD-groups implementation-defined (facts C2), so a kernel
   asks. The linear kernel is given.
2. `launch.mm`, `queryMetal(device, pipeline)`: fill `MetalInfo`. Four fields come from the
   device, two from the pipeline.
3. `launch.mm`, `encodeLinear(...)`: size the threadgroup by Apple's recipe (facts C6):
   width `threadExecutionWidth`, height `maxTotalThreadsPerThreadgroup` divided by that
   width, then no taller than the batch. Dispatch with `dispatchThreads`, which takes the
   grid in threads and makes the edge threadgroups smaller. The given kernel has no bounds
   check, so `dispatchThreadgroups` would write outside Y. `dispatchThreads` needs GPU
   family Apple4 or later; the driver checks and says so if your device is older.
4. `launch.mm`, `encodeProbe(...)`: dispatch the probe on the grid and threadgroup size
   the driver hands you.

## Commands

- Test: `python3 labs/pmpp.py run 4.1`   Start with `--backend emu`.
- Measure: `python3 labs/pmpp.py run 4.1 --bench` (real devices only)
- The reference: add `--solution`.

## What the tests do

- `queryDevices`: every field of device 0 must equal what the harness reads on its own.
- `chooseLaunch`: nine synthetic devices crossed with four output shapes (1 x 4096,
  3 x 40, 64 x 4096, 1000 x 1000). The devices are numbers chosen to exercise one rule
  each: the book's example limits, a device with 1536 thread slots and 16 block slots, a
  width of 64 under a 256-thread limit, `maxThreadsDim` smaller than `maxThreadsPerBlock`,
  slot fields of 0 at widths 32 and 64, and a grid limit too small for most of the
  shapes. None describes a product. Whether a configuration exists at all is found by
  trying every block within the limits. Where none does, the answer must be `false`.
- `launchLinear`: the result against a CPU reference with a tolerance, for outputs of
  1 x 1000, 3 x 40, 37 x 53 and 64 x 257, and the configuration it used against the rules
  for the real device. Y is pre-filled, so an output you never write is reported. On a
  real device a value stored past the end of Y is reported too; on the emulator
  AddressSanitizer stops at the first access outside a buffer and names the line.
- Metal: `MetalInfo` against the driver's own reading, for three pipelines: the two
  kernels of `kernel.metal`, and `linear_layer` compiled again with its thread limit set
  below the device's (through `MTLComputePipelineDescriptor.maxTotalThreadsPerThreadgroup`,
  facts C5). Apple describes the device's figure as what it supports "for a trivial
  shader", and these two kernels are close to trivial, so their pipelines may report the
  same number as the device. The third cannot, and shows a value read from the wrong
  object. The threadgroup
  `encodeLinear` returns, for both linear pipelines, must be at most that pipeline's
  `maxTotalThreadsPerThreadgroup`, a multiple of its `threadExecutionWidth` and no taller
  than the batch. The probe's records must satisfy
  what the specification promises (a lane below the width, no two threads of a threadgroup
  with the same SIMD-group and lane, every SIMD-group full except possibly the last) and
  nothing more: the tests do not assume how the cut is made.

The emulator's device properties are synthetic. They describe the emulator, not a GPU,
and its warp width is fixed.

## Before you benchmark

The benchmark times a linear layer with 1024 inputs and 4096 outputs (the book's example
layer of section 20.6, which lab 5.3 also uses) at batch 1 and batch 64, once with your
derived configuration and once with fixed 16 x 16 blocks, in alternation, and reports seconds and GFLOP/s for each. Write down two predictions first:

1. At batch 1, how many times faster will your derived configuration be than fixed
   16 x 16 blocks on your GPU? On CUDA and HIP a 16 x 16 block at batch 1 has 15 rows of
   threads with nothing to do. On Metal `dispatchThreads` clips the threadgroup to the
   grid, so that request runs as 16 x 1. Decide whether that changes your answer.
2. The width you expect your device to report.

The benchmark also prints what you queried, the block and grid you derived, and on Metal a
summary of the probe: the width the kernel saw, the SIMD-groups per threadgroup, and
whether the SIMD-groups were consecutive runs of the linear thread index in those
dispatches. A "yes" there is what your GPU did, not something Metal promises.

The numbers are measurements of your machine. If the derived configuration is slower than
16 x 16 at some batch size, that is a finding about the rules, and the reason to keep the
choice in one function you can change.
