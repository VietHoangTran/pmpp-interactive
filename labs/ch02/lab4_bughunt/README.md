# Lab 2.4: bug hunt, a stub that almost works

Book: sections 2.4 to 2.6 (pp. 28-39): Fig. 2.10 (the kernel), Fig. 2.13 (the host
function), the sidebar on error checking (p. 32). The program itself is ours.

`work/` holds a complete program that compiles. `scalePrefix` takes a host array of `n`
floats, multiplies the first `len` of them by `s` on the device, and must leave the
padding from `len` to `n` exactly as it was. It is the shape of applying an operation
to the valid part of a padded batch.

- `work/kernel.cu` (CUDA / HIP / emulator) has **four independent defects**: the kernel
  `scalePrefixKernel` and the host function `scalePrefix` are both in it.
- `work/kernel.metal` + `work/launch.mm` (Metal) have **three**. The Metal kernel is
  dispatched in whole threadgroups, as llama.cpp's Metal backend does, so it computes
  its own index the way a CUDA kernel does.

Each test fills the array with values between 1 and 2, calls `scalePrefix` with
`s = 3`, and examines the whole array: prefix scaled, padding bit for bit unchanged.
The sizes come in an order: first arrays with padding behind the prefix, then arrays
with none, then (CUDA family) a launch the device cannot run.

Method:

1. `python3 labs/pmpp.py run 2.4 --backend emu` first. The emulator runs the kernel
   under AddressSanitizer, which stops at the first access outside a device array and
   names the line.
2. Fix one thing, run again. The failures change as you go: one defect can hide
   another until it is fixed.
3. Then run on your GPU: `python3 labs/pmpp.py run 2.4`. Note which defects the GPU
   showed you by itself and which it hid.

Before you run anything on your GPU, predict: of the defects, how many will the GPU
report by itself, through an error code or a crash, as opposed to a wrong value that
only a test against the CPU reference reveals? Write the number down, then compare it
with what the emulator reported.

Nothing to benchmark here.

## After the hunt

Read this once all tests pass. It is about what each backend does with these
mistakes, and it names them.

- **Threads past `len`.** While the surplus threads still land inside the array they
  scale padding, which every backend shows as a wrong value. When `len` is `n` and not
  a multiple of the block size (the case `n = 1000`), they write past the end of the
  device array. The emulator stops there and names the kernel line. A GPU is not
  guaranteed to report it (`notes/today/ch02-facts.md`, items A6 and C5): on a GPU that
  case may simply pass. This defect is invisible while the block count is rounded
  down, because then no thread is surplus.
- **Zero blocks.** With the block count rounded down, a prefix shorter than one block
  launches nothing. The emulator rejects that launch as an invalid configuration. HIP
  documents an error for it (item B4). NVIDIA's documentation does not say what CUDA
  does (item A8). On Metal, a dispatch of zero threadgroups ran nothing and reported
  nothing when this lab was built; the facts file has no Apple statement about it. The
  tests therefore accept one outcome only: the right values and a return value of 0.
- **A byte count that is an element count.** Three quarters of the array never reach
  the device. What the kernel finds there instead depends on the backend: the
  emulator's fill pattern, whatever the memory held on CUDA (`cudaMalloc` does not
  clear it, item A2), zeros on Metal (`newBufferWithLength` clears it, item C2).
- **The unchecked launch** (CUDA family only). A launch statement has no value. NVIDIA's
  and AMD's pattern is to read the error state right after it (items A6 and B2). On the
  emulator the `cudaDeviceSynchronize` that follows says nothing about a refused launch,
  so the unchecked version returns 0 and the last test fails. Whether CUDA's or HIP's
  synchronise would report the refusal is not stated in the documentation we checked,
  so on a GPU that test may pass with the defect still in place. Metal has no
  counterpart: its dispatch calls return nothing and there is no error state to read
  (item C5), which is why the Metal version has three defects.

## Status of the reference solution

Verified on Metal, emulator-checked, front-end-checked. Unverified on CUDA and on HIP:
no machine with either has run it yet. Each defect, put back into the solution on its
own, was shown to fail: all four on the emulator, the three Metal ones on Metal.
