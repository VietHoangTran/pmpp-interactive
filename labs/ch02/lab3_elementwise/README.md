# Lab 2.3: the elementwise operations a transformer needs (project milestone M0)

Book: sections 2.3 to 2.6 (pp. 26-39). The kernel shape is Fig. 2.10 (p. 35), the
affine map is Exercise 9 (Fig. 2.15, p. 42), the shared helper is the
`__host__ __device__` row of Fig. 2.11 (p. 36) and Exercise 10. The book applies none
of this to bias, scale or residual, and shows no Metal: those parts are ours.

Write three kernels, the launch function beside each, and one host function that
chains them on the device:

- `addBiasKernel`: `Y[r][j] = X[r][j] + bias[j]` for a `rows x d` tensor and a bias of
  `d` values. One thread per element. The grid is flat, the tensor is not.
- `scaleShiftKernel`: `Y[i] = alpha * X[i] + beta`, through `scaleShift`, a helper that
  the compiler builds twice, once for the host and once for the device.
- `addResidualKernel`: `X[i] += R[i]`, in place.
- `launchAddBias`, `launchScaleShift`, `launchAddResidual`: each computes its own block
  count for 256 threads per block and launches. Nothing else.
- `blockTail`: `y = alpha * (x + bias) + beta + r` for host arrays. One set of copies
  in, three launches back to back, one check and one wait, one copy out, then free.

These follow the conventions every later milestone uses: row-major `float` tensors with
the batch dimension first, raw device pointers and explicit sizes as kernel arguments,
a `launch<Name>` function beside each kernel, the CPU reference in `main.cpp`. A launch
function only computes its grid and launches. Waiting and reading the error state are
the caller's job, so that a caller can queue several launches on data that stays on
the device. `launchAddBias`, `launchScaleShift` and `launchAddResidual` are the pieces
later milestones call.

**CUDA, HIP and the emulator:** `work/kernel.cu`. `scaleShiftOnHost` and the `CHECK`
macro are provided.

**Metal:** `work/kernel.metal` (the helper `scale_shift` and the kernels `add_bias`,
`scale_shift_k`, `add_residual`) and `work/launch.mm` (`encodeAddBias`,
`encodeScaleShift`, `encodeAddResidual`, and `blockTail`). Each encode function writes
one dispatch into an encoder the caller opened. `blockTail` puts all three dispatches
into one command buffer and waits once. Apple documents that the plain
`computeCommandEncoder` runs its dispatches serially and that buffers created from the
device are hazard-tracked by default, so each dispatch sees what the one before it wrote
(`notes/today/ch02-facts.md`, item C9).

On Metal there is no `scaleShiftOnHost`, and no test for it. Every function in a
`.metal` file is GPU code, written in a separate language and compiled separately from
the host program, and Apple documents no way to build one definition for both the CPU
and the GPU (`notes/today/ch02-facts.md`, item C7). `scale_shift` in `kernel.metal`
plays the part of a `__device__` helper only. Projects that need the same definitions
on both sides of Metal, llama.cpp among them, share a plain header of constants and
structs, not functions (item D2).

Names and signatures are fixed. `main.cpp` is the driver; you do not edit it.

- Test: `python3 labs/pmpp.py run 2.3`   Start with `--backend emu`.
- Measure: `python3 labs/pmpp.py run 2.3 --bench` (real devices only)

## What the tests look for

Each launch function is tested alone, on tensors the driver already keeps on the
device, for shapes `1 x 1`, `1 x 257`, `3 x 100`, `4 x 256` and `7 x 33` (real devices
add `64 x 4099`). Only one of those has a multiple of 256 elements.

- Every input is between 1 and 4 and every correct output is well away from 0. Outputs
  start as NaN, so an element no thread wrote is reported as such.
- The bias differs from column to column, and every shape but `1 x 1` has fewer rows
  than columns: a bias indexed by the row, or a row and column swapped, gives wrong
  values.
- Inputs are read back after the call and must be unchanged. `addResidual` is checked
  in place.
- The scale kernel is compared with the CPU reference and, on CUDA, HIP and the
  emulator, element by element with `scaleShiftOnHost`, within the harness tolerance.
  Not bit for bit: a compiler may fuse the multiply and the add into one operation that
  rounds once, and the host and device compilations need not agree on that.
- A thread past the end of a tensor: on the emulator, AddressSanitizer stops the run
  and names the kernel line. On a GPU nothing is guaranteed to report it, so there the
  driver puts a block of marker values behind every tensor and checks they survived.
- `blockTail` is compared with the CPU composition of the three operations on the same
  shapes. It must return 0, leave its inputs alone, and overwrite every element of
  `y_h` (which starts as NaN).

## Before you benchmark

Write down one number: at the size the benchmark picks, how many times faster is one
resident chain of three operations than the same three operations as three round trips?

Section 2.3 counted 9 tensor-sized transfers against 5 for a chain in which every
operation brings a whole tensor of its own. Here the bias is one row and the scale is
two numbers, so first count the tensor-sized copies each variant of this chain makes.
Your device decides what that count is in time.

The benchmark grows the tensor (4096 columns, rows doubling) until one kernel takes long
enough to time, up to 2^26 elements, and stops early if an allocation fails. It then:

- times each kernel alone on data already on the device and prints the global-memory
  traffic it asks for per second (12 bytes per element for the two adds, 8 for the
  scale). That is traffic requested, not traffic observed: `addBias` asks for the same
  bias row over and over;
- times one call of your `blockTail` against the same three operations done as three
  separate offloads that the driver composes from your launch (or encode) functions,
  each with its own copy in and copy out, and prints the ratio.

On Metal no bus is crossed. The round trips copy host arrays into shared buffers and
the results back out each time (item C3): what is measured there is the cost of an
interface that takes host arrays. Data kept in shared buffers would not be copied at
all.

## Status of the reference solution

Verified on Metal, emulator-checked, front-end-checked. Unverified on CUDA and on HIP:
no machine with either has run it yet.
