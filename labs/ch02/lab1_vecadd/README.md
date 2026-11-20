# Lab 2.1: vector addition, end to end

Book: sections 2.3 to 2.6 (pp. 26-39). The kernel is Fig. 2.10 (p. 35), the host function
is Fig. 2.13 (p. 38), the error check is the sidebar on p. 32. The Metal version is ours,
not the book's.

Write the whole program of the chapter. The driver plays `main` from Fig. 2.4: it owns
three ordinary host arrays and calls your `vecAdd`. It never sees a device pointer.

**CUDA, HIP and the emulator:** `work/kernel.cu`

- `CHECK(call)`: the pattern of p. 32 as a macro, written once. The snippet on the page
  does not compile as printed; find out why before you copy it. Yours prints the error
  string, the file and the line, frees nothing, and makes the enclosing function return a
  non-zero value. No `exit()`: the driver tests what happens after a failure.
- `vecAddKernel(A, B, C, n)`: one pair-wise addition per thread.
- `vecAdd(A_h, B_h, C_h, n, threadsPerBlock)`: allocate three device arrays, copy A and B
  in, compute the block count for the block size you are given, launch, check the launch,
  wait, copy C back, free. Every runtime call goes through `CHECK`. Return 0, or non-zero
  when anything failed.

**Metal:** `work/kernel.metal` and `work/launch.mm`

- `vec_add`: the kernel.
- `vecAdd(device, queue, pipeline, A_h, B_h, C_h, n)`: create the three buffers, encode,
  dispatch exactly `n` threads, commit, wait, check, copy the result out. Metal has no
  error codes: check every creation for nil, and the command buffer's status and error
  after the wait.
- `makeVecAddPipeline` is provided. It turns the text of `kernel.metal` into a pipeline
  state. The driver calls it once and keeps the result.

Names and signatures are fixed. `main.cpp` is the driver; you do not edit it.

- Test: `python3 labs/pmpp.py run 2.1`   Start with `--backend emu`.
- Measure: `python3 labs/pmpp.py run 2.1 --bench` (real devices only)

## What the tests look for

The inputs are in [0.5, 1.5], so no correct sum is near 0, and `C_h` starts as NaN.
Lengths sit on both sides of each block size: 1, 31, 32, 33, 255, 256, 257, 1000, 4097
(real devices add 2^20 + 3). On CUDA, HIP and the emulator each length runs with 32 and
with 256 threads per block, and one case uses the largest block the device reports.

- NaN left in `C_h`: nothing was copied back to those elements.
- A wrong tail: too few blocks. Device memory that no thread wrote holds a marker pattern
  on the emulator, whatever was there before on CUDA, and zero on Metal.
- An AddressSanitizer report on the emulator: an access outside an array, with the source
  line. A missing guard, a size in elements where bytes were meant, and a free before the
  copy back all end here. CUDA documents no bounds check on device arrays, which is why
  the first run is on the emulator.
- CUDA, HIP and the emulator only: one call asks for a block twice as large as the device
  allows. It must return non-zero without ending the program, and the valid call made
  right after it must return 0 with the right sums.

## Before you benchmark

Write down one number: at the largest size, how many times faster or slower is your
end-to-end `vecAdd` than the plain CPU loop? Above 1 means the GPU version wins.

The benchmark times the CPU loop of Fig. 2.4 and one whole call to your `vecAdd` for
three lengths (2^16, 2^20 and 2^24 elements, fewer if the device reports too little
memory), best of several calls after one untimed warm-up call. The warm-up keeps the
start-up of the runtime out of the timing. It prints the ratio. Lab 2.2 then shows where
the time went.

## Status of the reference solution

Verified on Metal, emulator-checked, front-end-checked. Unverified on CUDA and on HIP:
no machine with either has run it yet.
