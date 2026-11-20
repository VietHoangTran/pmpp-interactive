# Lab 19.1: a convolutional layer two ways

Book: section 19.2 (pp. 460-463, Figs. 19.5 and 19.7) and section 19.3 (pp. 463-472,
Eq. 19.4, Eq. 19.5, Fig. 19.11).

Compute one convolutional layer, the book's (stride 1, no padding), two ways, and time
them against each other and against the implicit GEMM of Fig. 19.11.

- X is N x C x H x W, F is M x C x K x K, Y is N x M x H_out x W_out (NCHW), all
  row-major `float`. H_out = H - K + 1, W_out = W - K + 1, and P = H_out * W_out is the
  number of pixels in one output map.

## Your part

**(1) Direct.** `convDirectKernel` and `launchConvDirect(X, F, Y, N, C, H, W, M, K)`:
Fig. 19.7, corrected. One thread per output pixel, blocks of `TILE_WIDTH x TILE_WIDTH`
threads, grid `(M, T, N)`, with the tile index linearised row by row as in Fig. 19.5 but
with ceiling division. Every output is bounds-checked, and you linearise the page's
`X[n, c, h + p, w + q]` by hand.

**(2) Unroll + GEMM.** `unrollKernel` and `launchUnroll(X, n, Bt, C, H, W, K)` write
sample n's unrolled input transposed, as Bt = B^T: P rows, one per output pixel, each
holding that pixel's C*K*K patch values in the book's (c, p, q) order. Then
`launchConvGemm(X, F, Y, Bt, N, C, H, W, M, K)`: for each sample, unroll into the scratch
`Bt` and make one call of the project's M2 GEMM, lab 6.4's

```
launchLinear(X, W, Y, batch, in, out)   // Y[batch x out] = X[batch x in] * W^T, W stored (out, in)
```

Choose its six arguments so that the sample's Y comes out M x P (NCHW) and F is never
copied. That one call is the point of the lab: the project's GEMM signature composes.

**Metal.** `conv_direct` and `unroll_input` go in `kernel.metal`; `encodeConvDirect`,
`encodeUnroll` and `encodeConvGemm` in `launch.mm`. `encodeConvGemm` loops over the
samples in one compute encoder and calls the provided `encodeLinearAt`, which binds lab
6.4's `linear` kernel with a byte offset into each buffer. A plain compute encoder runs
its dispatches in order and the buffers are hazard-tracked, so one scratch serves every
sample (`notes/today/ch02-facts.md` C9).

The corrected Fig. 19.11 is provided and timed beside your two paths; you do not write it.

## Where the code differs from the page

- Fig. 19.5 divides `W_out / TILE_WIDTH`, which leaves 12 rows and 12 columns of LeNet
  C1's 28 x 28 output uncomputed. Round up, and let the threads with no pixel do nothing.
- Fig. 19.7 has no bounds check and its signature lacks H, W and M, which the linearised
  indices need. The page also names its `dim3` variables `blockDim` and `gridDim`; the
  lab uses `dimBlock` and `dimGrid`, as lab 6.4 does, so they do not hide the built-in
  variables of those names.
- The book gives no unroll kernel. It describes explicit unrolling and its cost (p. 467)
  and moves on to the implicit version. The kernel here is ours.
- Fig. 19.11 (provided, `provided/implicit_gemm.*`): the phase count rounds up (the page's
  `(C*K*K) / TILE_WIDTH` runs one phase for C*K*K = 25 and drops 9 of the 25 terms), the
  F load, the X gather and the store are bounds-checked with zero fill (the book leaves
  this as an exercise, p. 470), and the host code the book does not show launches
  `(ceil(P / 16), ceil(M / 16), N)` blocks.

## Files

- `work/kernel.cu` (CUDA, HIP, emulators): `convDirectKernel`, `launchConvDirect`,
  `unrollKernel`, `launchUnroll`, `launchConvGemm`.
- `work/kernel.metal`: `conv_direct`, `unroll_input`.
- `work/launch.mm`: `encodeConvDirect`, `encodeUnroll`, `encodeConvGemm`. Its last
  function, `convGemmScratchProblem`, is provided: before the driver allocates the scratch
  it checks one sample's B^T against the device's `maxBufferLength` (Metal caps one buffer
  there, at least 256 MB by Apple's documentation, `notes/today/ch19-facts.md` C2) and
  fails with a message instead of allocating.
- Provided, not edited, in `provided/`:
  - `gemm.cmake` picks which lab 6.4 kernel your GEMM path calls (below).
  - `linear_at.mm`: `encodeLinearAt`, lab 6.4's `encodeLinear` with byte offsets.
  - `implicit_gemm.cu`, `implicit_gemm.metal`, `implicit_gemm_launch.mm`: the corrected
    Fig. 19.11, as `launchConvImplicitGemm` and `encodeConvImplicitGemm`.
- `main.cpp`: the driver. `cpuConvLayer`, the CPU reference, is Fig. 19.4 with its
  indices linearised and the sum kept in double.

**Which GEMM.** By default the path calls lab 6.4's reference solution. To call your own
lab 6.4 kernel, set `PMPP_CH19_GEMM=work`; `PMPP_CH19_GEMM=m2` or unset goes back. The
choice is read when the labs are configured, which `pmpp.py` does before every run. The
tests and the benchmark print which one ran ("m2 reference" or "m2 yours").

## Commands

- Test: `python3 labs/pmpp.py run 19.1 --backend emu` first. It runs under
  AddressSanitizer, so a read or write past X, F, Bt or Y stops with the source line.
  Then `--backend race`, then `python3 labs/pmpp.py run 19.1` on your GPU.
- Measure: `python3 labs/pmpp.py run 19.1 --bench`.
- With your own M2: `PMPP_CH19_GEMM=work python3 labs/pmpp.py run 19.1 --bench`.

## What the tests do

Shapes (N, C, H, W, M, K), on every backend:

- The book's example, (1, 3, 3, 3, 2, 2), with Fig. 19.2's integers. Y must be exactly
  `14 20 15 24 / 12 24 17 26`: small integers are exact in float, so no tolerance.
- The same twice, (2, 3, 3, 3, 2, 2).
- LeNet C1, (2, 1, 32, 32, 6, 5): C*K*K = 25, a 28 x 28 output and M = 6, none a
  multiple of 16. These are the shapes the printed Figs. 19.5 and 19.11 get wrong.
- LeNet C3, (2, 6, 14, 14, 16, 5): C*K*K = 150, a 10 x 10 output.
- Pointwise, (2, 8, 7, 9, 5, 1). One output pixel, (1, 4, 5, 5, 3, 5), so P = 1.
  Not square, (1, 2, 9, 17, 3, 3). C*K*K a multiple of 16, (1, 16, 6, 6, 16, 2).
- On a GPU also (4, 64, 56, 56, 64, 3), (8, 256, 14, 14, 256, 3) and (2, 3, 224, 224, 64, 3).
- `race` runs the book's example, (1, 2, 9, 17, 3, 3) and (1, 6, 14, 14, 16, 5): the GEMM
  and the implicit kernel share tiles across barriers.

Four checks per shape:

- **(a) The unroll alone.** Every element of every sample's B^T is compared exactly with
  a CPU unroll, so an index mistake is reported as an unroll error, with the row, column
  and X element it should have been, and not as a wrong GEMM.
- **(b) The direct path** and **(c) the unroll + GEMM path**, against `cpuConvLayer`
  with a tolerance scaled by the sum of |x f| per output. If (a) passed and (c) fails, the
  message points at the arguments of your GEMM call.
- **(d) The provided implicit GEMM.** A failure there is the course's bug, and the
  message says so.

X and F hold exactly their floats, and on the emulators Bt holds exactly P x C*K*K. Y
starts as NaN and is followed by a guard band of sentinel floats that must survive; on a
GPU Bt has one too. On a GPU each check runs twice, with Y and Bt back to NaN in between.

The untouched starter builds and fails: its empty kernels leave Bt and Y as NaN.

## Before you benchmark, predict

The benchmark runs on real devices only (`emu` says it is skipped). Five layer shapes
(N, C, H, W, M, K):

| layer | shape | per-sample GEMM, M x C*K*K x P | expansion, Eq. 19.4 |
|---|---|---|---|
| early | (8, 3, 224, 224, 64, 3) | 64 x 27 x 49284 | 8.84 |
| middle | (8, 64, 56, 56, 64, 3) | 64 x 576 x 2916 | 8.37 |
| late | (8, 256, 14, 14, 256, 3) | 256 x 2304 x 144 | 6.61 |
| pointwise | (8, 256, 14, 14, 256, 1) | 256 x 256 x 196 | 1 |
| small | (64, 6, 14, 14, 16, 5), LeNet C3 at batch 64 | 16 x 150 x 100 | 12.76 |

The expansion ratios are computed from Eq. 19.4, K^2 H_out W_out / (H W), not measured.
For each layer three paths take turns and the best of several rounds counts: your direct
kernel; your unroll + GEMM, timed as the whole layer, then its N unrolls alone, then its N
GEMMs alone on a filled scratch; and the provided implicit GEMM.

Write down three numbers first:

1. On the late layer, how many times faster or slower unroll + GEMM is than direct.
2. The same ratio on the early layer.
3. On the early layer, what share of unroll + GEMM's time goes to the unroll itself.

Things to weigh:

- The direct kernel requests 0.25 FLOP/B: per output, C*K*K multiply-adds and two loads
  for each (notes on section 19.2). But neighbouring threads read overlapping patches,
  and caches may serve much of that. Chapter 5's naive kernel ran well above its model
  ceiling on the machine this course was built on (section 5.1 of the lesson).
- The unroll writes B^T and the GEMM reads it back. B^T is the expansion ratio times the
  size of one sample's X.
- Tile counts with 16 x 16 tiles. Late: 16 tile rows, 9 tile columns, 144 phases. Early:
  4 tile rows and 3081 tile columns, but only 2 phases, the second 11 of 16 wide. How much
  reuse does a tile get in each?
- The small layer makes 64 unroll launches and 64 GEMM launches, each GEMM one tile row
  by 7 tile columns.
- The unroll does an integer divide and remainder (several of each) for every element it
  writes. NVIDIA documents integer division as costly ("up to 20 instructions"), AMD as
  "many more hardware cycles than multiplication", and Apple's only statement is a 2016
  talk (`notes/today/ch19-facts.md` A8, B5, C6). The direct kernel does one divide and
  one remainder per thread, then C*K*K multiply-adds.
- The solution's unroll gives each thread one element of B^T in B^T's own order, so
  neighbouring threads write neighbouring floats and read X along a patch row. Numbering
  the threads along v instead would read X contiguously and write B^T with a stride of
  C*K*K. You cannot have both without staging through shared memory. Which did you pick?
- llama.cpp's CUDA and HIP backend documents a cut-off: a layer with `IC * KW * KH < 64`
  and `OW * OH < 512` goes to its direct kernel instead of its implicit GEMM
  (`notes/today/ch19-facts.md` D1). Which of the five layers would that rule send there?
  It is a production choice for llama.cpp's kernels and hardware, not a prediction for
  yours.

The benchmark prints, per layer, the three paths in ms and GFLOP/s (2 N M C K^2 P / time),
the unrolls and the GEMMs alone with the unroll's share of the layer, and the ratio of
unroll + GEMM to direct. It also prints which GEMM ran and, on Metal, the scratch size
beside `maxBufferLength`. Its figures are measurements of your machine, not expectations.
The lab's page compares your three predictions with them.

## Beyond the book (ours)

- **Per-sample GEMMs.** Unrolling the whole batch into one C*K*K x N*P matrix would give
  one big GEMM, but its result comes out M x N x P (CNHW) and needs a permute to become
  NCHW. llama.cpp makes that copy; PyTorch's slow CUDA path loops over samples as this lab
  does (`notes/today/ch19-facts.md` D2, D4, E2).
- **B^T, not B.** Storing the unrolled matrix with one row per output pixel is what
  `launchLinear`'s (out, in) weight layout asks for. It is also the orientation of
  cuDNN's deprecated `cudnnIm2Col`, CUTLASS and llama.cpp's `ggml_im2col` (A4, A7, D2).
- **K = 1.** B is X itself, viewed as C x H*W, so the unroll only transposes. llama.cpp,
  MLX and PyTorch all skip it and call a GEMM on X directly (D1, D3, D4).
- If you compare with `torch.nn.functional.conv2d` on an NVIDIA GPU, a float32 call may
  run in TF32 unless you turn that off; compare with a tolerance and print the setting
  (A6).
