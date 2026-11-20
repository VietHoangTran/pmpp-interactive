# Lab 2.2: where does the time go?

Book: section 2.3, p. 28: the remark that "transparent" outsourcing can be very
inefficient because of the copying back and forth, and that real code keeps large data on
the device. The book gives no timings. The measurement and everything about Metal are
ours.

Instrument the round trip of lab 2.1 and measure that remark on your own machine. The
starter already contains a working `vecAdd` under a new name (the reference solution of
lab 2.1; paste your own if you prefer it). It computes the right sums and measures
nothing.

**CUDA, HIP and the emulator:** `work/kernel.cu`

- Complete `vecAddTimed(A_h, B_h, C_h, n, syncAfterLaunch, phase_s)`. `phase_s[0..4]`
  receive the seconds spent in allocation, copies in, kernel, copy back, free.
- `syncAfterLaunch != 0`: the kernel phase ends after a checked `cudaDeviceSynchronize()`.
  `syncAfterLaunch == 0`: the clock is read right after the launch, with no synchronise.
  That is the experiment. The sums must be right either way.
- `launchVecAdd` and the kernel are provided and working. The driver times `launchVecAdd`
  itself, with its own synchronise, on arrays it has already put on the device. That is
  the "resident" measurement.

You decide where each clock read and each synchronise goes. The vendors document two
traps (`notes/today/ch02-facts.md`, items A3, A5 and B1):

- A launch returns before the grid has run. A timer around a launch alone measures only
  the launch.
- A copy or a free may hide waiting. NVIDIA calls `cudaMemcpy` synchronous "for most use
  cases", and an older edition of its guide listed small host-to-device copies as
  asynchronous. `cudaFree` "may perform implicit synchronization". `hipFree` always
  synchronises first. Work that is still running when such a call starts is charged to
  the phase that call is in.

**Metal:** `work/launch.mm` (`work/kernel.metal` is provided complete)

- Complete `vecAddTimed(device, queue, pipeline, A_h, B_h, C_h, n, phase_s)`.
  `phase_s[0..3]` receive the seconds spent creating the buffers and copying the host
  arrays in, encoding and committing, waiting, and copying the result out. `phase_s[4]`
  receives the GPU time the command buffer reports (`GPUEndTime - GPUStartTime`).
- Write `vecAddResident(queue, pipeline, A, B, C, n)`: the same launch on buffers the
  driver created with shared storage and filled through `contents`. The driver reads the
  result in place.
- There is no `syncAfterLaunch` on Metal. On CUDA the blocking copy back waits for the
  kernel, so leaving the synchronise out only moves time from one phase to another. On
  Metal nothing waits for you: reading `contents` before the wait has returned is an
  ordering error, not a timing choice.

`main.cpp` is the driver; you do not edit it.

- Test: `python3 labs/pmpp.py run 2.2`   Start with `--backend emu`.
- Measure: `python3 labs/pmpp.py run 2.2 --bench` (real devices only)

## What the tests can and cannot see

- The sums are right, with both values of `syncAfterLaunch`, for lengths that are not
  multiples of the block size.
- Every phase was written, is finite and is not negative.
- Your phases add up to no more than the driver's own clock around the call (with 5%
  and 50 microseconds of slack) and to at least half of it. On Metal that is the four
  host phases; the GPU time lies inside the wait, so it is checked on its own: not 0,
  and not longer than the call.
- `launchVecAdd` (CUDA, HIP, emulator) and `vecAddResident` (Metal) give the right sums
  on arrays the driver owns.

These hold on the emulator too, where every call is synchronous. Whether a synchronise is
in the right place is not a test, and on the emulator it cannot be one. Only the
benchmark on a real device shows it.

## Before you benchmark

Write down two numbers.

1. At the largest size, what percent of the end-to-end time is the kernel?
2. CUDA and HIP: with the synchronise removed, what percent of its synchronised value
   will your kernel phase read, and which phase will grow instead?
   Metal: how does the time of the resident path compare with the total of the
   host-array path?

The benchmark runs three lengths (2^16, 2^20 and 2^24 elements, fewer if the device
reports too little memory). Each row is the fastest of several calls after one untimed
warm-up call. On CUDA and HIP it prints your five phases twice, with and without the
synchronise, and puts your kernel phase next to the driver's time for `launchVecAdd` on
resident data: if the two differ by much, a synchronise is missing or misplaced. On Metal
it prints your four phases and the GPU time, then the resident path with the number of
bytes that path copies. Both print the CPU loop.

## Status of the reference solution

Verified on Metal, emulator-checked, front-end-checked. Unverified on CUDA and on HIP:
no machine with either has run it yet. The synchronise placement in `solution/kernel.cu`
follows the documentation cited above and has not been measured on a CUDA or HIP device.
