# Lab 15.6: the fast linear layer (milestone M3)

Book: sections 15.3 to 15.8 (pp. 352-368) and Exercises 1 to 3 (p. 370).

Write `launchLinear(X, W, Y, batch, in, out)`: `Y[batch x out] = X[batch x in] * W^T`,
with W stored (out, in) row-major, as `nn.Linear` and ggml store a weight. It is lab
6.4's signature exactly, so this kernel drops in where M2 was. Inside goes your lab 15.5
kernel, adapted to the project's layout. In the book's names A = X (M = batch, K = in),
B = W^T (N = out), C = Y.

## Your part

Start from your lab 15.5 kernel. `clear`, `loadTile` (now for the X tile), `mm` and
`writeTile` carry over unchanged. Three things are new:

1. **The W tile.** B = W^T is stored column-major: W's rows are B's columns. Write
   `loadTileT`, which reads the W tile the way W is stored (128 rows of W, 8 floats along
   k each, one float4 per thread when `in % 4 == 0`, scalars with zero fill otherwise)
   and stores it transposed into `B_s[k][n]`. Then `mm` reads a B strip as one
   contiguous row of `B_s`, as in the book and in lab 15.5.
2. **The pitch of `B_s`, `LDB_S`.** The starter sets it to `bN`, which is correct and
   conflicts. Pick a better one (below).
3. **Names.** The kernel is `fastLinearKernel<WARP>`: the warp-width template parameter
   lab 15.5 called W is WARP here, because W is the weight. On Metal the kernel is
   `linear_m3`.

Handle `in == 0` (Fig. 15.14's `numTiles - 1` must not wrap). `launchLinearWidth`,
`launchLinear` and `launch.mm` are complete.

## Beyond the book (ours)

The book multiplies two row-major matrices and gives no code for vector loads, vector
stores or the section 15.5 layout. Everything that adapts lab 15.5 to the project layout
is ours:

- **Storing W transposed.** Without it, `mm` would read a B strip down a column of a
  shared tile, with a stride. Turning the tile in shared memory keeps the strip read a
  contiguous row; this is lab 6.4's corner turning at the 128 x 8 scale.
- **`LDB_S = bN + 4 = 132`.** The transposed store writes down columns of `B_s`. With
  NVIDIA's 32 banks of 4 bytes (`notes/today/ch15-facts.md` A2), lane t of warp w in the
  float4 mapping stores word `(4(t % 2) + j) * pitch + 16w + t / 2`. With pitch 132 that
  is `16(t % 2) + t / 2 + 4j + 16w` mod 32: 32 different banks. Fig. 15.5's scalar
  mapping, which this kernel does not use but a reader might (lane t stores
  n = 32s + 4w + t / 8, k = t % 8), gives `4(t % 8) + t / 8 + 4w` mod 32: 32 different
  banks again. With pitch 128 the same stores are 2-way and 8-way conflicts. Section
  15.6's explorer, third mode, draws both mappings. The rows stay 16-byte
  aligned and the B strip stays contiguous. 132 floats is 128 plus 16 bytes, the padding
  MLX gives its tiles. Apple documents no banks (C2), so on Metal this is a measurement.
- **The warp width.** As in lab 15.5: 32, or 64 for a 64-wide AMD wavefront, where the
  block becomes 2 x 2 wavefront tiles of 64 x 64, lanes 8 x 8, quadrants 32 x 32
  (ch15-facts B1). `launchLinear` reads `warpSize` from the device; on Metal the driver
  compiles `SIMD_WIDTH` from the pipeline's `threadExecutionWidth` (C1).
- **float4 paths.** Loads need `in % 4 == 0`, stores `out % 4 == 0`, and pointers that are
  the allocations themselves: 16-byte aligned on CUDA, where `cudaMalloc` aligns to at
  least 256 bytes (ch06-facts A3); on HIP, AMD's memory reference states no alignment for
  `hipMalloc` (ch15-facts B8), so the float4 paths there are unverified. On Metal they go through buffers declared
  `device float4*`, Metal's documented route, not a pointer cast (ch06-facts C3).
- **The 4096 -> 4096 layer** in the benchmark is our choice, a 7B-class hidden size.

Where the code differs from the page, as in lab 15.5: a one-dimensional block of 256
threads (notes, "Unclear" 7); every index into the padded A tile uses `bK + 1`, the row
offset included (notes, "Unclear" 4); Fig. 15.14 made concrete, with buffers chosen by
parity (notes, "Unclear" 8).

## Files

- `work/kernel.cu` (CUDA, HIP, emulators): the helpers, `fastLinearKernel<WARP>`.
- `work/kernel.metal`: `linear_m3`. `work/launch.mm`: `encodeLinear`, complete. It binds
  Y, X and W a second time, as float4, at indices 4, 5 and 6.
- Provided, not edited, in `provided/`: `vendor_linear.cu` (cuBLAS, hipBLAS),
  `vendor_linear_mps.mm` (MPS), `occupancy.cu` (registers and blocks per SM, CUDA and
  HIP) and `baselines.cmake`, which also copies lab 6.4's reference solution into the
  build directory with its names changed (`launchLinearM2`, `encodeLinearM2`) to serve as
  the m2 baseline.

## Commands

- Test: `python3 labs/pmpp.py run 15.6 --backend emu` first, then `--backend race`, then
  `python3 labs/pmpp.py run 15.6` on your GPU.
- Measure: `python3 labs/pmpp.py run 15.6 --bench`.

## What the tests do

- Shapes (batch, in, out): (1,1,1), (1,40,260), (3,3,3), (128,8,128), (128,24,128),
  (130,9,131), (5,1,9), (17,33,9), (256,32,256), (3,0,5); on a GPU also (4,300,1000),
  (257,129,511), (1024,1024,1024), (8,4096,4096). `race` runs (128,24,128), (130,9,131)
  and (1,40,260).
- `(3,0,5)` is `in == 0`: Y must come out all zeros. It runs under a 30-second watchdog, so
  a `numTiles - 1` that wraps around (an unsigned `numTiles` with no guard) fails with
  that reason, on `emu` and on Metal, instead of running for hours.
- Every shape at warp widths 32 and 64, on every backend: the width only changes which
  thread owns which outputs, so both must give the same Y. One more case runs
  `launchLinear` (Metal: the pipeline for the device's width) as the project calls it.
- X and W hold exactly their floats, so on `emu` a read past either is an
  AddressSanitizer error that names the line. Y starts as NaN and is followed by a guard
  band whose floats must still hold their sentinel afterwards.
- The result is compared with `y = x W^T` computed on the CPU in double, with a tolerance
  scaled by the sum of |x w| per output. On a GPU each case runs twice, Y back to NaN in
  between.
- On a GPU the provided vendor call is checked against the same reference. A failure
  there is the course's bug, not yours.
- The CPU reference, `cpuLinear`, is in this lab's `main.cpp`, as the project's
  conventions ask; `../shared/linear_test.hpp` declares it and holds the test problem.

## Before you benchmark, predict

The benchmark times three kernels on lab 5.3's and 6.4's 1024 -> 4096 layer at batch 1,
4, 16, 64, 256 and 1024, and on a 4096 -> 4096 layer at batch 1, 16 and 256: **m2** (lab
6.4's kernel, T = 16), **m3** (yours) and **vendor** (cuBLAS, hipBLAS or
`MPSMatrixMultiplication`). They take turns in every round and the best round counts.

Write down two things first:

1. At batch 256 on the 1024 -> 4096 layer, your M3's speed as a percentage of the vendor
   library's.
2. At batch 1, the ranking of m2, m3 and vendor.

Things to weigh:

- At batch 1 all three must read the whole weight matrix once; section 15.2's intensity,
  0.5 m n / (m + n), is under 0.5 FLOP/B when m = 1. And a 128-row block tile computes
  128 rows of which 127 are padding: how many blocks does the 1024 -> 4096 layer give at
  batch 1, and how much of the GPU do they occupy?
- A library may pick a different kernel for each shape. llama.cpp itself switches to
  matrix-vector kernels up to batch 8 (`MMVQ_MAX_BATCH_SIZE`, `MMVF_MAX_BATCH_SIZE`,
  ch15-facts D1), and PyTorch's own comment calls its gemv "bw bound" (D4).
- The vendor libraries use techniques this chapter only names (section 15.9). All three
  compute FP32 by default (A10, B6, C6), so the gap is technique, not precision.
- On Metal, Apple documents no banks and no register counts (C2, C3), and advises against
  many threads reading one threadgroup address, which section 15.5's layout does on
  purpose (C2). MLX's own GEMM uses smaller tiles and 64 to 128 threads per threadgroup
  (D3). These are facts to weigh, not reasons the documentation gives.
- Every round reuses the same X and W, so a cache large enough to hold W can serve it.

The benchmark also prints one row about your compiled kernel: on CUDA and HIP the
registers per thread and blocks per SM the runtime reports (section 15.7's argument,
measured), on Metal the pipeline's `threadExecutionWidth`, thread limit and threadgroup
memory. It prints the vendor library's math mode. Its figures are measurements of your
machine, not expectations.

## Notes on the vendor libraries

- cuBLAS and hipBLAS are column-major. The call is `cublasSgemm(h, CUBLAS_OP_T,
  CUBLAS_OP_N, out, batch, in, &1, W, in, X, in, &0, Y, out)`: Y^T = W X^T read
  column-major (our derivation, `provided/vendor_linear.cu`). hipBLAS takes the same
  arguments. MPS is row-major and needs only `transposeRight: YES`.
- The handle's math mode is never changed. On NVIDIA, `NVIDIA_TF32_OVERRIDE=0` turns TF32
  off everywhere; on compute capability 10.0 and 10.3 the BF16x9 emulation can be switched
  on without code changes through `CUBLAS_EMULATE_SINGLE_PRECISION`, and then the library
  runs a different algorithm (A10). The benchmark prints both variables.
- `provided/vendor_linear.cu` and `provided/occupancy.cu` have not been compiled for
  CUDA or HIP (unverified). If either does not build, configure with
  `-DPMPP_CH15_VENDOR_BLAS=OFF` or `-DPMPP_CH15_OCCUPANCY_API=OFF` and the lab runs
  without those rows. Without lab 6.4's solution there are no m2 rows.
