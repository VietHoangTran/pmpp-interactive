# Lab 23.1: hide the copies behind the kernel

Book: section 23.3 (pp. 553-563): streams and the fourth launch parameter (p. 556),
`cudaStreamSynchronize` (p. 559), `cudaMemcpyAsync` (p. 560), pinned memory (pp. 562-563).
The kernel, the chunked pipeline, the Metal version and every measurement are ours: the
book overlaps a halo exchange over the network with a stencil's internal rows, and shows
no host-device pipeline. The staged copy-and-execute pattern is NVIDIA's (Best Practices
Guide 10.1.2, `notes/today/ch23-facts.md` A7); the Metal pattern is Apple's (C5).

Chapter 2's round trip does three things one after another: copy the input to the device,
run the kernel, copy the output back. While the link copies, the GPU's cores wait; while
the cores compute, the link waits. Section 23.3's remedy, applied to one GPU: cut the work
into chunks and let chunk 2's copy run while chunk 1's kernel does. This lab makes you
write that pipeline and then measure how much of it your device delivers.

## The kernel

`polyKernel`: `out[i] = p(in[i])`, where p is a polynomial of degree `degree` with fixed
coefficients (0.5 for even powers, -0.25 for odd), evaluated by Horner's rule: `degree`
multiply-adds per element. It stands in for an elementwise layer whose cost you can dial.
The benchmark dials it so that the kernel takes about as long as copying the input in,
which is where overlap matters most.

## Your part

The chunk rule (both backends): chunk k is `[k*n/chunks, (k+1)*n/chunks)`, integer
division. Skip empty chunks (n can be smaller than `chunks`). Number the chunks you queue
j = 0, 1, 2, ... in order.

**CUDA, HIP and the emulator**, in `work/kernel.cu`: `runStaged(job, streams, chunks, order)`.

1. Create `streams` streams with `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)`.
2. Queue chunk j in stream `j mod streams`: `cudaMemcpyAsync` of its part of `hostIn` into
   `d_in`, `launchPoly` from `d_in` into `d_out`, `cudaMemcpyAsync` of its part of `d_out`
   into `hostOut`. Same offset everywhere.
3. `Order::Depth` queues chunk by chunk (copy in 0, kernel 0, copy out 0, copy in 1, ...).
   `Order::Breadth` queues every copy in, then every kernel, then every copy out.
4. Synchronise every stream, destroy each once, return 0. On a CUDA error, print it and
   return 1 (the `CHECK` macro).

The driver owns `d_in`, `d_out` and the host buffers, which are pinned (`cudaMallocHost`)
or pageable (ordinary heap memory) depending on the test.

**Metal**, in `work/launch.mm`: `runInFlight(job, buffers, chunks)`, Apple's documented
pattern for overlapping CPU and GPU work (C5). K = `buffers`.

1. K slots, each a pair of shared buffers long enough for the largest chunk, from the
   provided `slotBuffer`.
2. A dispatch semaphore that starts at K.
3. For chunk j: wait on the semaphore; if slot `j mod K` still holds an earlier chunk,
   `waitFor` its command buffer and copy its output to `hostOut`; copy the chunk's input
   from `hostIn` into the slot; encode with `encodePoly` in a command buffer of its own;
   add a completion handler that only signals the semaphore; commit.
4. After the last chunk, `waitFor` the slots still in flight, oldest first, and drain them.

The drain stays on your thread. Apple documents when a completion handler runs ("the GPU
device calls immediately after the GPU finishes running the commands", infra-facts C12),
and the facts files record nothing about which thread it runs on, so the handler does only
what Apple's sample does there: give the token back (C5).

## Files

- Edit, below the line `your code below`: `work/kernel.cu` (`runStaged`) and
  `work/launch.mm` (`runInFlight`).
- Provided above that line:
  - `kernel.cu`: `polyKernel`; `launchPoly(d_in, d_out, count, degree, stream)`, 256 threads
    per block, which records each launch for the tests and refuses a count below 1 (a grid
    of zero blocks is a launch error on CUDA; it refuses on every backend so that no test
    depends on what the emulator does with one); `runSerial`, the chapter 2 baseline.
  - `launch.mm`: `encodePoly(encoder, inBuf, inOffset, outBuf, outOffset, count, degree)`,
    which records the buffers it was given and refuses a range past a buffer's end;
    `waitFor`, which waits, reports a failed command buffer and adds up GPU time;
    `runSerialMetal`, the baseline in one command buffer; `slotBuffer`.
- Provided, not edited: `work/kernel.metal` (complete), `overlap.h` (the jobs, `Order`,
  every declaration), `main.cpp` (driver, CPU reference, calibration, benchmark),
  `probe.cu` (the CUDA and HIP instruments: copy-engine count, pinned allocation, timed
  parts).

## Commands

- Test: `python3 labs/pmpp.py run 23.1 --backend emu` first, then `python3 labs/pmpp.py run 23.1`.
- Measure: `python3 labs/pmpp.py run 23.1 --bench` (real devices only).
- The reference solution: add `--solution`.

## Where this lab differs from the book

- **Non-blocking streams.** The book creates its streams with `cudaStreamCreate` (p. 556)
  or with `cudaStreamDefault` and a priority (p. 558). Those are blocking streams: anything
  queued in the legacy default stream waits for them and makes them wait (ch23-facts A3,
  A9). `runSerial` and the driver's timing probes use the default stream, so this lab's
  streams are `cudaStreamNonBlocking`.
- **No priorities.** There is one kind of work here. Priorities are a hint for kernels only,
  with no effect on copies (A2), and ROCm ignores them by default (B2).
- **No events.** The book needs events because three kernels in three streams share one
  norm (pp. 559-560). Here each chunk's three operations stay in one stream, whose order is
  all the ordering they need.
- **"Requires" pinned memory.** The book says `cudaMemcpyAsync()` requires pinned host
  memory (p. 563). NVIDIA's reference is softer: from pageable memory the call still works
  but "will revert to a synchronous behavior which will not overlap with other work" (A6);
  HIP's header says the same (B4). The tests run both kinds; the benchmark measures the cost.
- **No copies on Metal.** On Apple silicon a shared buffer is the CPU's and the GPU's memory
  at once, and no copy engine is documented (C4). The work left to hide is the CPU's: the
  fill (writing a chunk's input into a shared buffer) and the drain (reading its output).
  One CPU thread does both, so the pipeline has two lanes, not three: the CPU's, fill +
  drain, and the GPU's. However well the buffers overlap, a run takes at least the longer
  lane (the stepper's bound, n times max(fill + drain, kernel) per chunk). The best case,
  lanes of equal length, is half the serial time; per-chunk costs, the wait for each slot
  and the two sides sharing one memory eat into that; the `inflight` rows measure how much.

## What the tests do

Shapes: n = 1, 7, 1000, 4099 and 65536 floats; real devices add the benchmark's size (see
below). Degree 7. Inputs in [-0.75, 0.75], from a fixed seed.

- **CUDA, HIP, emulator.** (streams, chunks) = (1, 1), (1, 4), (2, 2), (3, 5), (4, 16),
  (8, 3); for n = 7 also (2, 16) and (3, 16), where empty chunks sit between full ones; at
  the benchmark's size also (4, 64). Both orders, pinned and pageable host buffers (the
  same allocation on the emulator).
- **Metal.** K = 1, 2, 3, 4 with chunks = 1, 4, 2, 5, 16, 3, and 64 at the benchmark's size.

For each case:

1. `hostOut` equals the CPU's Horner evaluation in `main.cpp` within a relative tolerance
   (Metal does not default to strict IEEE arithmetic); `hostOut` starts as NaN, so a chunk
   never copied back shows as NaN. `hostIn` is unchanged.
2. CUDA family, from `launchPoly`'s record: every element computed exactly once, one launch
   per non-empty chunk on the rule's boundaries, input and output at the same offset, no
   refused launch, none in the default stream, `min(streams, non-empty chunks)` distinct
   streams, and neighbouring chunks in different streams when `streams` is 2 or more.
3. Metal, from `encodePoly`'s record: one dispatch per non-empty chunk with the rule's
   sizes, at most K distinct input buffers, neighbouring chunks in different input buffers
   when K is 2 or more, one `waitFor` per command buffer, and chunk j encoded after at most
   `max(0, j - K + 1)` waits. That last check is what makes it a pipeline: waiting for more
   than the slot needs leaves the GPU idle while the CPU fills.
4. Emulator only: AddressSanitizer stops a chunk that runs past an array with its source
   line, and a stream destroyed twice as a double free with both lines. A stream never
   destroyed fails the case through AddressSanitizer's count of live heap bytes, which
   must be the same after the call as before it. (LeakSanitizer would report it at exit,
   but on the Mac this lab was built on it says "detect_leaks is not supported on this
   platform".)

The provided baselines are checked too, quietly: a case appears only if one fails. The
untouched starter returns -1 from `runStaged` and `runInFlight`, so every case fails.

## What the tests cannot see

- **Anything about time, on the emulator.** Every stream call there is synchronous: a call
  has finished when it returns. So a `runStaged` with no `cudaStreamSynchronize` before
  returning, one with a `cudaDeviceSynchronize` after every chunk, and pageable buffers all
  pass. On the emulator, with the reference solution changed in each of the first two ways,
  every case passed. Only a benchmark on a GPU can show the second and the third. On a GPU
  the first is a race: the call returns while copies are still landing in `hostOut`, and
  whether a test catches it depends on timing.
- **The copies' streams.** Only launches are recorded. A copy queued in another stream than
  its chunk's kernel is a race on a GPU and invisible on the emulator.
- **Depth against breadth.** Both orders issue the same operations; nothing recorded tells
  them apart. The benchmark may.
- **`k*n` in an `int`.** At the test sizes it never overflows. At the benchmark's size with
  64 chunks it can, which is why real devices run one 64-chunk case there.
- **The semaphore.** Waiting on the slot's own command buffer is enough for correctness, so a
  `runInFlight` without the semaphore passes. The semaphore is Apple's documented rate
  control (C5); keep it, and keep the semaphore balanced: on the machine this lab was built
  on, releasing a dispatch semaphore whose count is below its starting value ended the
  process.

## Before you benchmark, predict

Use the stepper in the lesson's section 23.3. On CUDA and HIP the driver makes the kernel
take about as long as the copy in, so start from its "equal stages" preset. On Metal it makes
the kernel take about as long as fill + drain together. Write these down first; the lesson
asks for the first three under the ids in brackets.

1. CUDA or HIP: serial time divided by staged time at 4 streams, 16 chunks, depth-first,
   pinned (`ch23/labs/p-staged`). On CUDA, first look up `asyncEngineCount` in the
   benchmark's first lines (the `parts` row) and set the stepper's copy engines to match.
   On HIP that number means nothing (B3).
2. The same ratio with pageable host buffers (`ch23/labs/p-pageable`).
3. Metal: in-flight time with 1 buffer divided by in-flight time with 3 buffers, at 16
   chunks, with the stepper's Metal view, a kernel equal to fill + drain
   (`ch23/labs/p-metal`). Then, unrecorded: how close will 3 buffers come to half the
   serial time?
4. Not recorded: will breadth-first differ from depth-first on your GPU? The stepper says it
   might, with one copy engine, if that engine serves copies in the order they were issued
   (E2). NVIDIA documents neither that order nor how its copy queues change it (A7, A8).

After the run, the stepper's "your device" preset places your measured staged rows beside
the model.

## Reading the output

The size is as many floats as fit in a sixteenth of the memory the device reports
(`totalGlobalMem` on CUDA and HIP; the smaller of `maxBufferLength` and
`recommendedMaxWorkingSetSize` on Metal), up to 64 Mi floats (256 MiB per array). The cap is
a parameter of this lab, there to keep a run short, not a fact about any device. The first
lines print the size chosen and why.

- **CUDA and HIP** print `asyncEngineCount` and `concurrentKernels`. NVIDIA: with one copy
  engine, a copy in and a copy out cannot run at the same time; with two they can (A7). On
  ROCm `asyncEngineCount` is a runtime constant on every device (B3).
- **Calibration.** The copy in of the whole array is timed (pinned, events, best of
  several), then the kernel at degrees 1, 2, 4, ... until it takes twice as long, then one
  degree between the last two. The degree whose kernel time is closest to the copy in is
  used for every row and printed. On Metal the kernel is matched to fill + drain instead:
  the CPU does both, one after the other on one thread, so that sum is its lane, and the
  kernel is timed inside the provided `runSerialMetal`, exactly as the `serial` row is. At low
  degrees the kernel's time may stay flat: reading and writing the array, not the arithmetic,
  sets it (chapter 5's ratio). If the time never grows, or a low degree takes longer than a
  high one, something else was using the GPU (or, on Metal, the CPU) during the run; on the
  machine this lab was built on, that happened while other programs were busy. Run again
  before you compare anything.
- **`parts`**: each part alone over the whole array. CUDA and HIP: `h2d_s`, `kernel_s`,
  `d2h_s`, plus `async_engines` and `concurrent_kernels`. Metal: `fill_s`, `kernel_s`
  (GPU time of the serial path's one command buffer), `drain_s`.
- **`firsttouch`** (Metal): making a shared buffer of the array's size, and its first fill,
  against a fill of a buffer already used. On the machine this lab was built on, the first
  fill took several times as long, almost all of it in the first write to each page. That is
  why `slotBuffer` keeps your slots from one call to the next: with buffers made inside each
  call, the rows below would mostly time page faults.
- **`serial`**: the provided baseline, `host` = `pinned` or `pageable` (CUDA, HIP) or
  `shared` (Metal).
- **`staged`** (CUDA, HIP): `streams` 1, 2, 3, 4, 8 by `chunks` 4, 16, 64 by `order`, pinned;
  plus 4 streams, 16 chunks, depth, pageable.
- **`inflight`** (Metal): `buffers` 1 to 4 by `chunks` 4, 16, 64, with `gpu_s`, the sum of
  `GPUEndTime - GPUStartTime` over the command buffers passed to `waitFor` (ch05-facts C9),
  and `overlap_observed`: whether the CPU encoded a chunk, right after filling it, while the
  GPU was running another chunk's command buffer. Both clocks are mach time. That is an
  observation of your machine; Apple documents the pattern, not the outcome.
- Every row's `seconds` is the wall-clock time of one whole call (stream creation
  included), the best of several calls after one untimed warm-up. The warm-up's and the last
  call's outputs are compared with the serial path's, which is itself checked against the
  CPU on a sample; a wrong one is reported without a time.
- The last lines divide the times your predictions asked about.

The numbers are measurements of your machine at one size and one degree, not properties of
the method.

## Where the backends differ (not in the book)

- **CUDA**: `asyncEngineCount` copy engines (A7). Streams map onto 8 hardware work queues
  by default (`CUDA_DEVICE_MAX_CONNECTIONS`); more active streams than queues can make
  unrelated streams wait for each other (A8). Watch the 8-stream rows.
- **HIP**: the same calls through `pmpp/vendors/hip.h` (`hipStreamCreateWithFlags`,
  `hipMemcpyAsync`, `hipHostMalloc`). At most 4 hardware queues per device by default;
  more streams share them round-robin (ch22-facts B1). An async copy from pageable memory
  is performed synchronously (B4). `asyncEngineCount` is a constant (B3). Not run on AMD
  hardware here.
- **Metal**: no host-device copy for shared buffers and no copy engine (C4); several
  buffers in flight with a semaphore and a completion handler is the documented overlap
  (C5). A queue holds at most 64 uncompleted command buffers by default (C1); this lab
  keeps at most 4 in flight.

Where you meet this again: PyTorch's `tensor.to("cuda", non_blocking=True)` is a
`cudaMemcpyAsync` (ch02-facts D4), and PyTorch's own notes say that from pageable memory
"the call may block the host" (appC-facts D1). llama.cpp's CUDA backend creates its streams
with `cudaStreamNonBlocking` (ch23-facts D3).

## Go further

- Give `runStaged` a third order: copy in chunk j + 1 before launching chunk j. Where does
  it land between depth and breadth on your GPU?
- Time the 8-stream rows again with `CUDA_DEVICE_MAX_CONNECTIONS=4` and with `=16` (A8).
- On Metal, move the drain into the completion handler, and say what you would need to know
  about the handler's thread before you trusted it.

## Status of the reference solution

Verified on Metal (tests and benchmark), emulator-checked, front-end-checked for
`kernel.cu` and `probe.cu`. The front-end check sees `probeEngines`'s emulator branch only:
the stub header declares neither `asyncEngineCount` nor `concurrentKernels`. Unverified on
CUDA and on HIP: no machine with either has run it.
