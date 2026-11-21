# Lab C.1: three ways to feed a kernel

Book: appendix C, sections C.2 (pp. 624-625), C.3 (p. 625), C.6 (pp. 626-627, Fig. C.1)
and C.8 (p. 628). The kernel is an embedding lookup in the book's sense (ch. 20, p. 480).
The passes, the Metal paths, the probes and every measurement are ours: the appendix
shows no code for C.2 and gives no timings.

Appendix C is a history of where a kernel's data may live and which pointer reaches it.
This lab gives one kernel three homes on each backend and times the whole trip, the one
Fig. C.1 makes: the CPU produces a table, the kernel reads it, the CPU consumes the
result. The kernel is provided. You write the host side.

## The kernel

`gatherAdd`: `out[r*cols + c] += table[ids[r]*cols + c]`. Row r of the result is row
`ids[r]` of the table: a token-embedding lookup, where the table is a vocabulary of
embedding rows and the ids are token ids. It runs `passes` times on the same data, so the
result is `passes` times the gathered rows. The passes stand in for a kernel that reads its
input more than once.

## Your part

Every path receives a `FeedJob` (`feeding.h`): the table's `rows` and `cols`, `count` ids
in host memory the driver owns, `passes`, and two callbacks. `produce` writes the table, as
`fread` does in Fig. C.1; `consume` reads the result, as `use_data` does. Each path:

1. allocates the table, the ids and the result in its own kind of memory;
2. calls `produce` exactly once to write the table, and copies the ids in;
3. starts the result at zero;
4. launches the kernel `passes` times, then makes sure the last pass has finished;
5. calls `consume` exactly once, with a pointer the CPU may read;
6. frees what it allocated and returns 0, or prints what failed and returns 1.

**CUDA, HIP and the emulator**, in `work/kernel.cu`:

- `feedExplicit`: the book's simple model (p. 623). `produce` writes into `malloc` memory;
  `cudaMalloc` the three arrays; `cudaMemcpy` the table and ids in; `cudaMemset` the
  result; the passes; `cudaMemcpy` the result back; `consume`.
- `feedMapped`: zero-copy (C.2), which NVIDIA now calls mapped memory. `cudaHostAlloc`
  with `cudaHostAllocMapped` for all three arrays; `produce` writes in place; the kernel
  gets the same pointers; synchronise; `consume` in place.
- `feedManaged`: Fig. C.1. `cudaMallocManaged` with `cudaMemAttachGlobal` for all three;
  `produce`, the passes, `cudaDeviceSynchronize`, `consume`, all through the same pointers.

**Metal**, in `work/launch.mm`:

- `feedPrivateCopy`: the explicit model. `produce` writes a shared staging buffer; a blit,
  in a command buffer of its own, copies it into a private buffer; the passes run on
  private buffers; a second blit copies the result into a shared buffer; `consume`.
- `feedNoCopy`: page-aligned host memory, its length rounded up to whole pages, wrapped
  with `newBufferWithBytesNoCopy` (ch02-facts C2). llama.cpp hands its memory-mapped
  model weights to Metal this way (appC-facts D4).
- `feedShared`: `newBufferWithLength` with shared storage; `produce` writes through
  `contents`.

## Files

- Edit `work/kernel.cu` and `work/launch.mm`, below the line `your code below`.
- Provided, not edited:
  - above that line, the kernel `gatherAddKernel` and `launchGatherAdd` (one pass, 256
    threads per block); `encodeGatherAdd` (one pass into an open compute encoder) and
    `waitFor` (waits for a command buffer, reports a failure, adds up its GPU time as copy
    or kernel work). Both launchers record the table they were given, for the tests;
  - `work/kernel.metal`, the Metal kernel, complete;
  - `feeding.h`, the job and every declaration;
  - `main.cpp`, the driver, with the CPU reference;
  - `../shared/probe.cu`, the CUDA and HIP instruments of the benchmark.

## Commands

- Test: `python3 labs/pmpp.py run appC/lab1_feeding --backend emu` first, then
  `python3 labs/pmpp.py run appC/lab1_feeding`. The runner also accepts the lab's number,
  `C.1`, in place of the directory.
- Measure: `python3 labs/pmpp.py run appC/lab1_feeding --bench` (real devices only).
- The reference solution: add `--solution`.

## Where this lab differs from the book

- **Mapped memory needs no second pointer today.** The book says the `cudaHostAlloc`
  pointer cannot be passed to a kernel, and host code must get a different one from
  `cudaHostGetDevicePointer()` (p. 624). That was CUDA 2.2. Every 64-bit CUDA process now
  has unified addressing, and NVIDIA documents `cudaHostAlloc` memory as reachable from
  the device through the same pointer, with or without the mapped flag, and
  `cudaHostGetDevicePointer()` as unnecessary for it (appC-facts A3, A5). The test checks
  that the kernel receives the very pointer `produce` wrote through. The emulator does
  not provide `cudaHostGetDevicePointer`, so a call to it does not build there. Keep the
  mapped flag: on HIP it makes the memory fine-grained (B2).
- **Two names for pinned memory.** The book allocates pinned memory with `cudaHostAlloc`
  here and with `cudaMallocHost` in chapter 23, without relating them. They are one
  allocation, with and without flags: `cudaHostAllocDefault` "causes cudaHostAlloc() to
  emulate cudaMallocHost()" (infra-facts A8).
- **Fig. C.1 as printed** drops the semicolon after `cudaDeviceSynchronize()` and elides
  the launch configuration. `feedManaged` has both and checks every call.
- **Two meanings of "unified memory".** The book's (CUDA 6) is a pool of pages that the
  runtime migrates between two physical memories. Apple's is one physical memory that
  nothing migrates (appC-facts, terminology note). Metal has no managed mode on Apple
  silicon (ch05-facts C6), so its third path is a shared buffer.

## What the tests do

Shapes, rows x cols of float: 1 x 1, 7 x 3, 64 x 33 and 130 x 17; real devices add
4096 x 256. Two sets of ids: dense (every row once, in order) and sparse (a row in eight,
rounded up, drawn from a fixed seed with repeats, always including the last row, where an
off-by-one shows). One pass and three passes. For each path and case:

1. The result `consume` saw is `passes` times the gathered rows of the CPU reference in
   `main.cpp`, within a tolerance. Every correct value is also a whole number exactly
   (the table holds whole numbers and the kernel only adds), and that is checked too.
2. `produce` and `consume` were each called exactly once, with pointers that are not null.
3. Structure. For `feedMapped`, `feedManaged`, `feedNoCopy` and `feedShared`, the table
   the provided launcher recorded is the memory `produce` wrote: one pointer serves both
   sides. For `feedExplicit` and `feedPrivateCopy` it is not, so a copy happened, and on
   Metal the buffer the kernel read has private storage.
4. `feedNoCopy`: the pointer `produce` wrote through is a multiple of the page size
   (`getpagesize()`).

The untouched starter returns -1 from every function, so every case fails.

## What the tests can and cannot see

- **A missing synchronise.** On the emulator every call is synchronous, and
  `cudaHostAlloc` and `cudaMallocManaged` are plain host allocations. A `feedMapped` or
  `feedManaged` with no `cudaDeviceSynchronize` before `consume` passes there, and no
  timing effect shows. Only a real device shows the mistake: as a race on Linux with full
  unified memory (`consume` reads while the GPU may still be writing), and, for managed
  memory, as a segmentation fault where `ConcurrentManagedAccess` is 0, which includes
  every Windows and WSL system (appC-facts A10).
- **A result that does not start at zero.** The emulator's fresh allocations hold bytes
  that read as a float of about -3e-16. Added to a table value, that vanishes, except
  where the table holds 0. The whole-number check of test 1 catches it wherever a gathered
  row holds a 0, which happens in the 64 x 33 and 130 x 17 cases. On Metal,
  `newBufferWithLength` clears what it allocates (ch02-facts C2); `posix_memalign` memory
  does not have to be clear. On the machine this lab was built on, leaving the `memset`
  out of `feedNoCopy` failed most of its cases.
- **Consuming before the wait, on Metal.** On the machine this lab was built on, the
  shared path's result then read as zeros and every case failed. Metal does not promise
  that outcome; it only makes ordering your job (ch05-facts C6).
- **Alignment, on Metal.** On the same machine, `newBufferWithBytesNoCopy` accepted memory
  that was not page-aligned, and the results were right. Apple's documentation asks for
  page-aligned memory (ch02-facts C2), and test 4 is the one that fails.
- Leaks are not checked, and neither is whether each free matches its allocation (on the
  emulator they are all `free`).
- The tests cannot tell whether a path is fast. That is the benchmark's job.

## Before you benchmark, predict

A tool to predict with, the C.2 break-even rule. It is ours, derived from the book's
advice to use zero-copy memory only for data a kernel reads "occasionally and sparsely"
(p. 625). On a discrete GPU the explicit path moves the whole table across the
interconnect once and then reads device memory. The mapped path moves only what the
kernel touches, but crosses the interconnect on every touch. Device memory is much
faster than the link (the book says the link is under a tenth, p. 625; the benchmark's
probe row measures both on your machine). So mapped wins, roughly, only when the kernel
reads less than about one table's worth over its life. The benchmark prints that amount as
`touched`, the table bytes the passes read divided by the table's size: 1 for `dense1`,
8 for `dense8`, 1/64 for `sparse1`.

Write these down first. The lesson asks for the first three under the ids in brackets.

1. CUDA or HIP, `sparse1`: explicit time divided by mapped time (`appC/labs/p-sparse`).
2. CUDA or HIP, `dense8`: mapped time divided by explicit time (`appC/labs/p-repeat`).
3. Metal, `dense8`: private-copy time divided by shared time, and whether no-copy will
   differ from shared at all (`appC/labs/p-metal`). On Apple silicon no interconnect
   separates the CPU's memory from the GPU's (appC-facts C2). Apple says Metal "may apply
   additional optimizations to private resources" and does not say which.
4. Look up your managed paradigm (the CUDA and HIP benchmark prints it first; the
   lesson's page for C.7 explains the four) and say which path `managed` will resemble:
   explicit after the first pass on Linux with full unified memory, where a page usually
   migrates to whichever processor uses it (A11); a bulk move of all managed
   memory when the GPU starts, and back when you synchronise, on Windows and WSL (A10);
   mapped on AMD without XNACK, where `hipMallocManaged` "behaves the same as
   hipMallocHost" (B5).

## Reading the output

The table has 256 floats per row and as many rows as fit in an eighth of the memory the
device reports (`totalGlobalMem` on CUDA and HIP; the smaller of `maxBufferLength` and
`recommendedMaxWorkingSetSize` on Metal), up to 256 MiB. The 256 MiB cap is a parameter
of this lab, there to keep a run short, not a fact about any device. The first lines
print the bytes chosen and why.

- CUDA and HIP print the unified-memory paradigm: the three attributes of appC-facts A14
  and the name of the paradigm they make. Then the probe: a pinned host-to-device
  `cudaMemcpy` and a device-to-device `cudaMemcpy` of the table's size, timed with events,
  best of several. Device bandwidth counts the bytes read plus the bytes written. Their
  ratio is your machine's answer to the book's "less than 10%". `feedExplicit` copies
  from `malloc` memory, not pinned memory, so its copy may run below the probe's figure
  (ch02-facts A16).
- Metal prints whether a buffer's CPU address (`contents`) equals its GPU address
  (`gpuAddress`, macOS 13 and later), for a shared and a no-copy buffer. Apple documents
  neither answer: this is an observation of your machine (appC-facts C3).
- Then one row per workload and path. `seconds` is the whole feed call, the best of
  several calls after one untimed warm-up call: allocation, `produce`, copies, passes,
  synchronise, `consume` and free. On Metal, `GPU copy` and `GPU kernels` add up
  `GPUEndTime - GPUStartTime` of the command buffers that call passed to `waitFor` as copy
  and kernel work (ch05-facts C9).
- Allocation is inside the time on purpose: Fig. C.1's function allocates too. A host
  allocator may hand back memory it already holds from the previous call, while
  `newBufferWithLength` clears what it hands out (ch02-facts C2), so part of a difference
  between two paths can come from allocation rather than from copies. Look at the GPU
  columns before you decide.
- The last lines divide the times your predictions asked about.

The warm-up call's result and the last call's result are checked; a wrong one is reported
without a time. The numbers are measurements of your machine at one table size, not
properties of the paths.

## Where the backends differ (not in the book)

- **CUDA**: mapped memory is pinned host memory, one pointer for both sides under unified
  addressing, every access crossing the interconnect (A3). Managed memory faults and
  migrates page by page where `ConcurrentManagedAccess` is 1 (A11), and moves in bulk
  where it is 0 (A10).
- **HIP**: the same calls through `pmpp/vendors/hip.h`. `hipHostMalloc` with the mapped
  flag gives fine-grained memory (B2). `hipMallocManaged` migrates only with XNACK on
  (`HSA_XNACK=1`) and a driver with HMM; otherwise it behaves as pinned host memory (B5).
  `cudaMallocManaged` and `cudaMemAttachGlobal` reach HIP through two stopgap defines in
  `feeding.h` until the shared header maps them. Not run on AMD hardware here.
- **Metal**: one memory. A shared buffer needs no copy (ch05-facts C6); a private buffer
  has no CPU address (ch02-facts C2); a no-copy buffer wraps your own page-aligned memory
  (C2). No managed mode and no migration on GPU page faults are documented for Apple
  silicon (appC-facts C5).

## Go further

- `feedExplicit` stages through `malloc` memory. Stage through `cudaMallocHost` instead
  and compare its copy with the probe's figure. Is a staging buffer still worth having
  once it is pinned, or should `produce` write the pinned buffer and the copy go straight
  from there?
- Prefetch the table to the device in `feedManaged` before the first pass. The call
  changed in CUDA 13 to take a `cudaMemLocation` (appC-facts A14), and HIP's default
  header still takes an `int` device (B6), so the line needs an `#if`. Neither emulator
  has the call: this one is for real devices only.

## Status of the reference solution

Verified on Metal (tests and benchmark), emulator-checked, front-end-checked for
`kernel.cu` and `../shared/probe.cu`. The front-end check sees the paradigm query's
emulator branch only: its CUDA and HIP branches use device attributes the stub header does
not declare. Unverified on CUDA and on HIP: no machine with either has run it.
