# Lab 4.4: block size on a kernel that waits on memory

Book: section 4.6 (pp. 82-85) for why a waiting warp costs nothing while another is ready,
section 4.7 (pp. 85-87) for the occupancy arithmetic, exercises 6 to 8 (pp. 91-92). The
kernel is ours, not a listing from the book.

Write a gather in the shape of an embedding bag:

    out[i] = sum over j < k of table[idx[i*k + j]]      for i < n

The indices are random and the table is large, so nearly every step of the loop is a load
from global memory that nothing predicts. This is the kind of kernel section 4.6 is about:
it spends its time waiting, and what it needs from the SM is other warps to run meanwhile.
It is also the shape of a token-embedding lookup at the bottom of a model.

- Edit `work/kernel.cu`:
  - `gatherSumKernel` and `launchGatherSum(table, idx, out, n, k, threadsPerBlock)`: one
    thread per output on a 1D grid, the block size a run-time argument.
  - `modelBlocksPerSm(limits, threadsPerBlock, regsPerThread)` and `modelOccupancy(...)`:
    the book's arithmetic over block slots, thread slots and registers. Work the book's
    cases by hand first, then write the function that gives the same answers. A block
    takes thread slots in whole warps, the width comes from `limits.warpSize`, and a limit
    of 0 means the device did not report it: skip the rule that needs it. `sm_limits.h`
    holds the struct; it is given and not edited.
- Edit `work/kernel.metal` (`gather_sum`) and `work/launch.mm` (`encodeGatherSum`, with
  the threadgroup size as an argument). There is no model to write for Metal: Apple
  documents no per-core limits, no register count and no occupancy function
  (`notes/today/ch04-facts.md`, C5).
- `provided/occupancy_api.cu` is the driver's. On CUDA and HIP it asks the vendor's
  runtime how many blocks of your kernel fit on one SM.
- Test: `python3 labs/pmpp.py run 4.4`   Start with `--backend emu`.
- Measure: `python3 labs/pmpp.py run 4.4 --bench` (real devices only)

## What the tests do

The kernel is compared with a CPU reference, with a tolerance, for several n, k and block
sizes built from the width your device reports: one width, one and a half, two, three, the
largest block the device takes, and 7 threads. Most leave a partial last block. `out` is
pre-filled, so an output you never write is reported. On a real device a value stored past
the end of `out` is reported; on the emulator AddressSanitizer stops at the first access
outside a buffer and names the line.

The model runs on limits nobody has to own (CUDA, HIP and the emulator):

- the book's seven worked block sizes on its example device: 100% for 1024 down to 64
  threads per block, 50% for 32, 75% for 768;
- registers on the same device: 31 and 33 registers per thread at 512 threads per block
  (4 blocks, then 3), and 64 registers per thread (50% at any block size);
- exercises 6, 7 and 8;
- a device with 1536 thread slots and 16 block slots, where 768 gives 100% and 1024 gives
  66.7%. Current GPUs differ in these limits (facts A12), so the book's answers hold on
  its example and not everywhere;
- blocks that are not whole warps (48 threads take two warps where a warp is 32), and a
  width of 64;
- limits of 0, and blocks that cannot run at all.

Registers are the model's weakest part. The book multiplies threads by registers per
thread. NVIDIA says registers are partitioned among the warps and does not publish the
granularity; AMD allocates them per wavefront in granules and has a second register pool
that no device field shows (facts A13, B10). For a real kernel the vendor's occupancy
function is the authority, which is why the benchmark prints it beside your model.

## Before you benchmark

The benchmark runs the same gather (a table sized from your device's memory, between 32 MB
and 256 MB) at 1, 2, 3, 4, 6, 8, 12, 16, 24 and 32 times the device's width, up to the
largest block the device or the pipeline accepts. The sizes are timed in rounds. For each
it reports the best and the median time and the lookups per second.

Write down your prediction first: which block size will be slowest on your device, and by
what factor compared with the fastest?

On CUDA and HIP two more columns follow: the blocks per SM and the occupancy your model
gives for the limits the device reports, and the same from the vendor's function for your
compiled kernel. Where they disagree, find out which resource your model left out.

- Your model is given no register count unless you provide one: no device field holds it.
  The compiler prints it (`nvcc --resource-usage`, `hipcc --resource-usage`). Pass it with
  `PMPP_REGS_PER_THREAD=<n> python3 labs/pmpp.py run 4.4 --bench`.
- On AMD, `maxBlocksPerMultiProcessor` is computed from the thread slots and is not a
  limit of its own, the block size is rounded up to whole wavefronts, and the register
  field counts vector registers only (facts B9, B10). The benchmark says so when it runs
  on HIP.
- `provided/occupancy_api.cu` has not been compiled by nvcc or a HIP compiler. If it does
  not build with your toolchain, configure once with
  `cmake -S labs -B labs/build -DPMPP_CH04_OCCUPANCY_API=OFF` and the benchmark runs
  without the vendor's columns.

On Metal the benchmark is a timing sweep and nothing else. There is no occupancy figure to
set beside it, so what you can say is what you measured.

The last two lines of the output give the slowest size over the fastest, and for scale how
far the median run of a single size was from its own best run. A gap between two sizes
that is no larger than that second number is the machine, not the block size. The numbers
are measurements of your device; a flat sweep is a result too.
