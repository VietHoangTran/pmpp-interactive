# Lab X2.2: a 4-bit GEMV on llama.cpp's Q4_0 blocks

Not in the book. Module X2 is ours, anchored on appendix A, section A.5 (p. 593) and on
chapter 15's intensity analysis. The vendor and project facts come from
`notes/today/x2-facts.md`, checked on 2026-11-20.

Lab X2.1 cut the weights to about 1 byte each, in a layout chosen for the kernel. This lab
cuts them to 4.5 bits, in the layout you will meet in a GGUF file: llama.cpp's Q4_0, exactly
as `ggml-common.h` defines it (x2-facts D1).

```cpp
struct BlockQ4_0 { uint16_t d; uint8_t qs[16]; };   // 18 bytes per 32 weights
// d:     the FP16 bits of the block's scale
// qs[j]: code j in the low nibble, code j + 16 in the high nibble (not j and j + 1)
// w = (code - 8) * d,  codes 0..15
```

```
Y[b][j] = sum over k of X[b][k] * w[j][k]      1 <= batch <= 8, in % 32 == 0
```

W is `[out x in/32]` blocks, row j holding output j's weights. X and Y are float32,
row-major, batch first.

- **The rule is llama.cpp's `_ref` rule** (`quantize_row_q4_0_ref`).
  - Take the value of largest magnitude, with its sign; the first one wins a tie.
  - `d = max / -8`. That signed extreme lands exactly on code 0, and the opposite extreme
    clips to code 15. So d is negative when the extreme is positive.
  - Each code is `min(15, (int8_t)(w / d + 8.5))`, computed from the float d; FP16(d) is
    stored.
  - An all-zero block stores d = 0 / -8 = -0, FP16 bits `0x8000`, and codes 8.

  llama.cpp's importance-matrix path picks d differently, with the same layout (D1). This
  lab does not model it.
- **Bytes per weight**: 18 / 32 = 0.5625, against 4 for FP32 and 1.0625 for Q8_0 (E1).

## Your part

1. **`work/kernel.cu`: `gemvQ4_0Kernel` and `launchGemvQ4_0`.** One warp per output j, in
   blocks of 4 warps, as in labs 22.1 and X2.1.
   - **How lanes share a block is your choice**: one lane per block, or several (the X2.4
     stepper shows the trade). It must work for every warp width, not only 32.
   - **Read `qs` as bytes or as 16-bit words, never through `int` or `char4`.** An 18-byte
     block has alignment 2, so in every other block `qs` starts at an address that is
     2 mod 4 (A5). llama.cpp reads it with two 16-bit loads (`get_int_b2`, A5).
   - **Unpack with masks and shifts.** The low nibble of byte t is weight t; the high
     nibble is weight t + 16.
   - **Folding the -8 is recommended, not required.** For one block,
     sum of (q - 8) d x = d (sum of q x - 8 sum of x). The inner loop then touches only q
     and x (E4). Every real kernel in x2-facts does this: llama.cpp through Q8_1's `s` field
     on CUDA (D3) and through `sumy` on Metal (D4), MLX through `sum` (D5).
2. **`work/kernel.metal`: `gemv_q4_0`**, with the block declared as llama.cpp declares it on
   Metal, `struct block_q4_0 { half d; uchar qs[16]; };` (C5). **`work/launch.mm`**: the
   dispatch in `encodeGemvQ4_0`.
3. Benchmark, after writing down your prediction (below).

## Files

- `work/kernel.cu` (CUDA, HIP and the emulators). Yours: `gemvQ4_0Kernel`,
  `launchGemvQ4_0`. Provided: `warpSum` (static, because two other solutions are linked
  beside it), `deviceWarpSize`, and from `labs/x2/shared/qformat.h` `BlockQ4_0` and
  `halfBitsToFloat`.
- `work/kernel.metal`: `gemv_q4_0` is yours. `work/launch.mm`: the dispatch in
  `encodeGemvQ4_0` is yours.
- `provided/unfused.cu`, `provided/unfused.metal`, `provided/unfused_launch.mm`
  (provided, correct, for the benchmark only). `launchGemvQ4_0Unfused` and
  `encodeGemvQ4_0Unfused` dequantise W into a float32 scratch matrix that the caller owns,
  then run lab 22.1's gather on it.
- `labs/x2/shared/qformat.h` and `qfixture.h`: read-only (lab X2.1's README lists them).
- `main.cpp`: the driver, the CPU quantiser and the dequantising reference. You do not need
  to edit it.
- `CMakeLists.txt` links lab 22.1's solution (FP32 gather), lab X2.1's solution (Q8_0, if
  present) and the provided unfused path.

The signatures are fixed. Lab X2.4 links them:

```cpp
void launchGemvQ4_0(const float* X, const BlockQ4_0* W, float* Y, int batch, int in, int out);
// Metal: kernel gemv_q4_0, struct block_q4_0 { half d; uchar qs[16]; };
//        buffers X 0 (float4), W 1, Y 2, batch 3, in 4, out 5.
void encodeGemvQ4_0(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                    id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
                    uint32_t batch, uint32_t in, uint32_t out);
// Provided:
void launchGemvQ4_0Unfused(const float* X, const BlockQ4_0* W, float* Wf, float* Y,
                           int batch, int in, int out);      // Wf: out x in floats of scratch
void encodeGemvQ4_0Unfused(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState> dequant,
                           id<MTLComputePipelineState> gather, id<MTLBuffer> X, id<MTLBuffer> W,
                           id<MTLBuffer> Wf, id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
```

## Commands

- Test: `python3 labs/pmpp.py run X2.2 --backend emu` first, then
  `python3 labs/pmpp.py run X2.2` for your GPU.
- Measure: `python3 labs/pmpp.py run X2.2 --bench` (real devices only).
- The reference solution: add `--solution`. Start again: `python3 labs/pmpp.py reset X2.2 --yes`.

## What the tests do

- **The CPU quantiser comes first**, on two hand-derived blocks (`qfixture.h`):
  - one whose extreme is negative, d = +1/8 (`0x3000`);
  - one whose extreme is positive, d = -1/4 (`0xB400`);
  - an all-zero block (`0x8000` and bytes `0x88`).

  The test checks the scale bits, every packed byte and the dequantised values. The
  lessons' simulator test is meant to use the same blocks.
- **Shapes (batch, in, out):** (1, 32, 1), (1, 64, 5), (2, 96, 7), (3, 256, 64),
  (8, 512, 130), (5, 4096, 3), and (1, 32 x 33, 3). The last has 33 blocks per row, so
  every other row starts at an odd block, at an address that is 2 mod 4. On a GPU also
  (1, 4096, 4096), (8, 4096, 4096), (4, 14336, 4096), (1, 4096, 14336) and (8, 1024, 4099).
- **The weights:**
  - Each block has its own power-of-two magnitude, from 2^-12 to 2^12.
  - Each block has a +peak and a -peak. Which comes first alternates, so both signs of d
    occur and the other peak clips to code 15.
  - One block per row has its first 16 weights positive and its last 16 negative, so
    reading a byte as two neighbours fails.
  - With out >= 3, row out / 2 is all zero, and with 2 or more blocks per row so is the
    last block of row 0.
- **Exact sizes, NaN and two launches, as lab X2.1.** X and W hold exactly their bytes
  (18 per block). Y starts as NaN, is followed by sentinels, and every case launches twice.
- **The comparison.** Results are compared with `dequantize_row_q4_0` followed by a double
  dot, within 1e-4 x (1 + sum of |x w|), never bitwise.
- **The baseline is tested too.** The provided unfused path is tested on three shapes, so
  the benchmark only compares correct kernels.

## What each backend can catch

- **`emu`** catches wrong indices, a nibble paired with the wrong weight, a scale from the
  wrong block, a missing -8, reads past X or W, and a lane that skips the reduction. There
  are no shared updates, so there is no `race` build.
- **No emulator catches a misaligned 4-byte load.** The host CPU allows one, and
  AddressSanitizer does not check alignment. An `int` read of `qs` passes on `emu` and is
  undefined on a GPU for every other block: block g's `qs` starts at byte 18g + 2 of W,
  which is 2 mod 4 for even g (A5). The 33-block shape puts those blocks at even positions
  in some rows and odd positions in others, but a device may still give the right answer.
  **Check by reading**: every load from a block is 1 or 2 bytes wide.
- **Metal** checks the results on your GPU. The same alignment rule holds there (C5).
- **Nothing here checks a 64-lane warp.** The emulators model 32-lane warps. Check by
  reading that your lane-to-block mapping uses `warpSize`.

## Where the backends differ (not in the book)

- **Reading the block.** CUDA and HIP: two 16-bit loads, as llama.cpp's `get_int_b2` (A5).
  Metal: llama.cpp reads 16-bit words through a pointer cast (C5). The MSL specification
  describes no pointer casts (infra-facts C7), so the reference solution reads `uchar`s.
- **The scale.** CUDA and HIP keep FP16 as `uint16_t` bits, decoded by `halfBitsToFloat`.
  The emulators do not model `__half`, and the two vendors' headers differ (x2-facts table).
  Metal reads `half` directly.
- **4-bit types.**
  - CUDA and HIP have no 4-bit integer type in C++ (A4, the table).
  - MSL now has `int4b_format` and `uint4b_format` with `unpack` (Metal 4.0 with SDK 26.4,
    and 4.1). The spec does not say which nibble comes first, nor which GPU families
    support them (C2). Neither llama.cpp nor MLX uses them (a code search).
  - Masks and shifts are the portable way.
- **No integer dot here either.** llama.cpp's CUDA and HIP path makes the activations int8
  and feeds unsigned nibbles and signed bytes to `dp4a`. On HIP it needs a different
  builtin on each target (B1, B2, D3). Metal has no integer dot (C1).
- **HIP** has not been built or run here: unverified.

## Before you benchmark, predict

Write these down first:

1. The speed-up of your fused Q4_0 kernel at batch 1 on 4096 -> 4096:
   - over F32: the bytes ceiling is 4 / 0.5625 = 7.11;
   - over Q8_0: 1.0625 / 0.5625 = 1.89.

   Those are byte ratios. They are not promises.
2. The speed-up of the unfused path over F32. It reads 0.5625 bytes per weight, writes 4
   and reads 4 again, against F32's 4. The bytes model says about 0.47, that is, slower than
   not quantising at all (A1 quotes PyTorch saying exactly this about unfused int8). Does
   your device agree?

Then think about batch 8: activation bytes are the same for every format, and the unpack
work per weight is shared by more rows.

## Reading the output

- **Layers and batches** as lab X2.1: 4096 -> 4096, 4096 -> 14336 and 14336 -> 4096, at
  batches 1, 4 and 8. The last two shapes are ours.
- **Kernels:**
  - `gather`: F32, lab 22.1's solution.
  - `q8_0`: lab X2.1's solution, when present.
  - `q4_0_unfused`: provided.
  - `q4_0`: yours.

  They take turns: one untimed round, then five, and each keeps its best.
- **Copies of W.** One timed submission goes through copies of each format's weights that
  together hold at least 256 MB, so no launch finds its weights in a cache (as lab 22.1).
  The unfused path writes and reads a single scratch matrix. That matrix is not copied, so a
  cache may help it. Its time is still per launch, dequantise and gather included.
- **`W GB/s`** is the bytes each format stores (Q4_0: 18-byte blocks) divided by the time.
  For the unfused path that leaves out the scratch traffic it also moves. The labs page
  compares `W GB/s` with lab 5.1's measured bandwidth.
- **`vs F32` and `vs Q8_0`** are time ratios.
- **Report rows:** `qgemv`: `{kernel, format, batch, in, out, seconds, weight_bytes,
  weight_gb_per_s, speedup_vs_f32}`.

These are measurements of your device at these shapes, not properties of the kernels.
Where the fused speed-up falls short of 7.11, ask what else the kernel pays per weight
that F32 does not: unpacking, the scale, the -8. Ask too what it pays per row of X.

## Go further

- **Prefill** (ours, after llama.cpp's `mul_mm`, x2-facts D4). With many tokens, a GEMV is
  the wrong kernel. Dequantise one tile of W into shared memory, then run lab 15.6's tiled
  GEMM on it. That is "dequantise, then multiply", done at tile granularity rather than for
  the whole matrix. Compare it with the unfused path at a batch of 64.
- **Pre-scaled activations, the Metal idiom** (D4, D5, D6). llama.cpp, MLX and PyTorch MPS
  never shift the nibbles. They mask them in place, leaving each multiplied by 1, 16, 256 or
  4096, and divide the matching activations by the same power of two, which is exact. Try
  it in `gemv_q4_0` and measure.
- **Q4_1 or MLX's affine format.** Add an offset per block, as Q4_1's `d q + m` (D1) or
  MLX's `s q + bias` (D5). The fold of E4 carries the offset through the sum of x the same
  way.
