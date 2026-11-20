# Lab 7.1: the filter in constant memory, three ways

Book: sections 7.2 to 7.4 (pp. 164-172): Fig. 7.7 (p. 166), Fig. 7.9 (p. 170) and the
by-value alternative on p. 172.

The basic convolution kernel loads one filter weight and one input element from global
memory for every multiply-add, so the book's model gives it 0.25 FLOP/B (p. 167). Every
thread reads the same weights in the same order, so section 7.4 moves the filter into
constant memory, where all threads of a warp reading one address are served by the
constant cache, and the model doubles to 0.5 FLOP/B (p. 171). The book then names an
alternative that needs no explicit copy call: put the weights in a struct and pass it by value
with the launch (p. 172). You write all three and measure whether the doubling shows up
on your device.

**Your part:** three kernels and their launches. All three compute the same output: one
thread per output element, zero ghost cells, the filter not flipped (the book's formula,
p. 162; PyTorch's `F.conv2d(x, w, padding=r)` computes the same thing), a run-time
radius from 0 to 7, and a guard for threads past the edge of the image.

1. `convBasicKernel` and `launchConvBasic(N, F, P, r, width, height)`: Fig. 7.7, with
   `F` a device pointer. The printed listing indexes the plain pointers `F` and `P` as
   2D arrays and does not compile; flatten both. The book's kernel has no output guard
   because it assumes the grid fits the image exactly; yours needs one.
2. `F_c`, `convConstKernel` and `launchConvConst(N, F_h, P, r, width, height)`:
   Fig. 7.9. Declare `F_c` in constant memory, flat, with room for the largest filter.
   `F_h` is the filter in host memory; the launch copies its `(2r+1)^2` weights into
   `F_c` with `cudaMemcpyToSymbol`, then launches.
3. `convByValueKernel` and `launchConvByValue(N, F_h, P, r, width, height)`: the
   alternative on p. 172. Fill the provided `struct Filter` (904 bytes) on the host and
   pass it to the kernel as an ordinary `const Filter` argument.

Keep the `__constant__` declaration, the kernel and the copy in `kernel.cu`. The book
says a kernel in another file needs an `extern` declaration (p. 169); in nvcc's default
whole-program mode that is an error unless everything is built with `-rdc=true`
(HIP: `-fgpu-rdc`). CUDA also has a `__grid_constant__` qualifier for parameters like
the struct; we found no HIP documentation of it, so this file does without it.

On Metal there is no symbol to copy to: a program-scope `constant` array must be
initialised in the shader source. The three ways become three kinds of kernel argument:

| | CUDA | Metal |
|---|---|---|
| basic | `const float* F` | `device const float* F`, a buffer |
| const | `__constant__ float F_c[...]` + `cudaMemcpyToSymbol` | `constant float* F`, a buffer in the constant address space |
| byvalue | `const Filter f`, passed by value | `constant Filter& f`, filled with `setBytes` (under 4 KB) |

## Files

- Edit `work/kernel.cu` (CUDA, HIP, emulator). `TILE_WIDTH` is defined by the build
  (16); blocks are `TILE_WIDTH x TILE_WIDTH`. Do not hard-code 16.
- Edit `work/kernel.metal` (`conv_basic`, `conv_const`, `conv_by_value`) and
  `work/launch.mm` (`encodeConvBasic`, `encodeConvConst`, `encodeConvByValue`). The
  driver passes 16 as `tileWidth`.
- Each file is split by lines reading `// @part basic`, `// @part const` and
  `// @part byvalue`. Leave them in: the lessons use them to show one part at a time.
- `main.cpp` holds the tests, the CPU reference and the benchmark. You do not edit it.

## Commands

- Test: `python3 labs/pmpp.py run 7.1 --backend emu` first, then `python3 labs/pmpp.py run 7.1`.
- Measure: `python3 labs/pmpp.py run 7.1 --bench` (real devices only).
- Start again: `python3 labs/pmpp.py reset 7.1 --yes`.

## What the tests do

For each of the three kernels:

- The book's 7 x 7 image and 5 x 5 filter (Fig. 7.4). `P[2][2]` must be 321 and
  `P[1][0]` must be 112, the two outputs the book works out (pp. 163-164), and every
  other output must match the reference.
- Random images with random filters at radii 0, 1, 2, 3 and 7, on images of 1 x 1,
  5 x 9, 16 x 16, 33 x 17, 100 x 61 and 257 x 129 (width x height) on a GPU, and
  1 x 1, 5 x 9, 7 x 7 and 33 x 17 on the emulator. A random filter is not symmetric,
  so a flipped filter, or one read as `F[fCol][fRow]`, fails here although it passes the
  book's symmetric example. Radius 7 on a 5 x 9 image is almost all ghost taps.

`P` starts as NaN, so an output you never write is reported. The reference is a direct
convolution accumulated in double; each output's tolerance is scaled by the sum of
`|f * x|` over its taps, because Metal does not default to strict IEEE arithmetic.

On the emulator, AddressSanitizer stops at a read outside `N` (a missing ghost-cell
test) or a write outside `P` (a missing output guard) and names the line. On a GPU the
same mistakes show up only as wrong values.

## Before you benchmark, predict

Write these down first.

1. At r = 2 (a 5 x 5 filter), how many times faster than the basic kernel will the
   constant-memory kernel run on your device? The book's model says 2x, from 0.25 to
   0.5 FLOP/B. That holds only if both kernels are limited by memory bandwidth and
   nothing else is cached: the model counts every read of `N` as a trip to DRAM,
   although neighbouring threads read mostly the same input elements (p. 168).
2. On Metal, will the `setBytes` struct beat the constant buffer? Apple documents no
   constant cache. What it has said, in a 2016 talk, is that the `constant` address
   space is meant for heavily reused read-only data, and that data whose reads are
   statically bounded, such as a fixed-size struct passed by reference, can be preloaded
   into registers. The talk was about the iPhone GPUs of the time; whether your Mac does
   this is not documented.

## The benchmark

For each radius in 1, 2, 3, 5 and 7, the image grows from 2048 x 2048 until the basic
kernel takes over 50 ms, up to 8192 x 8192 and within a quarter of the memory your
device reports. All three kernels then run on that image with the same random filter,
best of `--reps` runs (5 by default). Each row gives the time, GFLOP/s, the speed-up over
the basic kernel, and the book's model intensity for that kernel. GFLOP/s counts
`2 x width x height x m^2` per run with `m = 2r + 1`, ghost taps included: the book's
count (p. 167), not the work the kernels actually do near the edges.

On Metal a first row reports the `conv_const` pipeline's `threadExecutionWidth` and
`maxTotalThreadsPerThreadgroup`.

To place each kernel against its model, take the bandwidth lab 5.1 measured on the same
device. The model's ceiling is 0.25 times that figure (in GFLOP/s) for the basic kernel
and 0.5 times it for the other two. A kernel well above its ceiling tells you the
model's "every read goes to DRAM" assumption does not describe your device: something is
serving the reuse. The numbers are measurements of your machine.

## Optional, on CUDA: where did the filter go?

`-Xptxas=-v` makes ptxas list each kernel's register, shared and constant memory use
(CUDA Best Practices Guide). For example:

```sh
nvcc -c -Xptxas=-v -DTILE_WIDTH=16 labs/ch07/lab1_constconv/work/kernel.cu -o /dev/null
```

See where the 900 bytes of `F_c` show up for `convConstKernel`, and whether anything
of the 904-byte `Filter` shows up for `convByValueKernel`. NVIDIA documents that kernel
parameters are passed through constant memory, up to 32,764 bytes since CUDA 12.1 on
Volta and later GPUs. (This command was not run here: the machine that wrote
this lab has no CUDA toolkit.)

## What the vendors document

From `notes/today/ch07-facts.md`, to set beside what you measure:

- NVIDIA: constant memory is cached. If all threads of a warp read the same address it
  "can be as fast as a register access"; different addresses within a warp are
  served one after another (A3). The filter's index does not depend on the thread,
  which is the best case. Kernel parameters travel through constant memory (A4).
- AMD: `__constant__` keeps its name and `cudaMemcpyToSymbol` becomes
  `hipMemcpyToSymbol` (mapped by the build). The cache behind it is the scalar cache,
  which helps only when the read is uniform across the wavefront. We found no AMD
  document giving a size for constant memory or a limit on kernel arguments (B1, B2).
- Apple: no constant cache and no size limit for constant buffers are documented;
  `setBytes` takes under 4 KB (C1, C2).

## Where this lab differs from the book's pages

- Fig. 7.7 indexes `float* F` and `float* P` with two subscripts and gives `inRow` and
  `inCol` no type. Both are fixed here, and the output guard is added.
- Fig. 7.9 declares `F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1]` for one compile-time
  radius. Here `F_c` is flat and sized for radius 7, so one build serves every radius.
- The book describes `cudaMemcpyToSymbol` as telling the runtime that the data will not
  change during the kernel. NVIDIA documents it as a copy to any device symbol, constant
  or not; it is the `__constant__` qualifier that makes `F_c` read-only in kernels (A2).

## Go further

Real backends use the by-value route. llama.cpp's CUDA `conv2d_kernel`
takes its sizes as a `const conv_params` argument, and its Metal kernels take a
`constant` struct reference filled with `setBytes`. Neither llama.cpp's CUDA backend nor
PyTorch's native CUDA kernels use `__constant__` at all, and llama.cpp keeps its
quantisation lookup tables in global memory on CUDA and in the `constant` address space
on Metal (D4). Those tables are indexed by each thread's own quantised value. From what
NVIDIA and AMD say about their constant caches, why is that a poor fit for constant
memory, when a convolution filter is a good one?
