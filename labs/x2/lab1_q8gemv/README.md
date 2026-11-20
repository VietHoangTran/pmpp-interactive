# Lab X2.1: an int8-weight GEMV for batch-1 decode

Not in the book. Module X2 is ours. It is anchored on appendix A, section A.5 (p. 593): a
smaller number format fetches fewer bytes per operation, and that raises compute intensity.
It also builds on chapter 15's intensity analysis and lab 22.1's gather. The vendor and
project facts come from `notes/today/x2-facts.md`, checked on 2026-11-20.

At decode time, every linear layer is a matrix-vector product, and its cost is reading W
once (lab 22.1). Storing W in fewer bytes is the one lever left. This lab stores it as
llama.cpp's Q8_0: blocks of 32 weights, each block an int8 code per weight plus one FP16
scale (x2-facts D1).

```
Y[b][j] = sum over k of X[b][k] * d[j][k / 32] * q[j][k]      1 <= batch <= 8, in % 32 == 0
```

- **The rule is llama.cpp's** (`quantize_row_q8_0_ref`). Per block, `d = amax / 127`, each
  code is `roundf(w / d)` (halves away from zero), and d is stored as FP16. The codes come
  from the float d before it is rounded to FP16. `labs/x2/shared/qformat.h` has the rule,
  and the driver calls it.
- **The layout is ours.** Codes live in one array `Wq [out x in]` (int8). Scales live in a
  second array `Wd [out x in/32]`, as FP16 bits in `uint16_t`. That is how MLX keeps scales
  apart from its packed weights (D5). llama.cpp's `block_q8_0` interleaves a 2-byte scale
  with 32 codes, 34 bytes, so its codes are only 2-byte aligned (A5). Split apart, every row
  of codes starts at a multiple of 32 bytes, and four codes can be read as one 32-bit word.
- **Bytes per weight**: 1 + 2/32 = 1.0625, against 4 for FP32 (E1).

## Your part

1. `work/kernel.cu`: write `codeOf`, `gemvQ8_0Kernel` and `launchGemvQ8_0`. The mapping
   is lab 22.1's gather:
   - one warp per output j, in blocks of `WARPS_PER_BLOCK` = 4 warps;
   - the lanes stride along row j together, each loading 4 codes as one 32-bit word;
   - one register sum per batch row;
   - each word's 4 products multiplied by their block's scale once;
   - `warpSum` across the warp, and lane 0 writes `Y[b][j]`.

   Never a literal 32 for the warp width. The 32 in this lab is `QK8_0`, the block size,
   and it has a name.
2. `codeOf` turns byte i of a word back into a signed code. Do it with integer operations
   that do not depend on how a compiler converts an out-of-range value to `int8_t`.
3. `work/kernel.metal`: write `gemv_q8_0`, the same kernel with codes as `char4` and scales
   as `half`. In `work/launch.mm`, write the dispatch in `encodeGemvQ8_0`.
4. Benchmark, after writing down your prediction (below).

## Files

- `work/kernel.cu` (CUDA, HIP and the emulators). Yours: `codeOf`, `gemvQ8_0Kernel`,
  `launchGemvQ8_0`. Provided: `warpSum` with its own lane-mask type, `deviceWarpSize`, and
  `halfBitsToFloat` from `labs/x2/shared/qformat.h`. `warpSum` is `static` because lab
  22.1's solution is linked beside it and has its own.
- `work/kernel.metal`: `gemv_q8_0` is yours.
- `work/launch.mm`: the dispatch in `encodeGemvQ8_0` is yours. The buffer binding is
  provided.
- `main.cpp`: the driver and the CPU reference. You do not need to edit it.
- `labs/x2/shared/qformat.h` (provided, shared with lab X2.2): `BlockQ4_0`,
  `floatToHalfBits` (round to nearest, ties to even), `halfBitsToFloat` (integer operations
  and a power of two, so that no `__half` type is needed), `quantizeRowQ8_0Ref`,
  `quantizeRowQ4_0Ref`, `dequantizeRowQ8_0`, `dequantizeRowQ4_0`, all plain loops.
- `labs/x2/shared/qfixture.h` (provided): the hand-derived fixture blocks that the driver
  checks the quantiser on first.
- `CMakeLists.txt` links lab 22.1's reference solution (`launchGemvGather`,
  `encodeGemvGather`) as the FP32 baseline.

The signatures are fixed. Lab X2.4 links them:

```cpp
void launchGemvQ8_0(const float* X, const int8_t* Wq, const uint16_t* Wd, float* Y,
                    int batch, int in, int out);
// Metal: kernel gemv_q8_0. Buffers: X 0 (float4), Wq 1 (char4), Wd 2 (half), Y 3,
//        batch 4, in 5, out 6.
void encodeGemvQ8_0(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                    id<MTLBuffer> X, id<MTLBuffer> Wq, id<MTLBuffer> Wd, id<MTLBuffer> Y,
                    uint32_t batch, uint32_t in, uint32_t out);
```

`encodeGemvQ8_0` encodes into the encoder it is given and commits nothing.

## Commands

- Test: `python3 labs/pmpp.py run X2.1 --backend emu` first, then
  `python3 labs/pmpp.py run X2.1` for your GPU.
- Measure: `python3 labs/pmpp.py run X2.1 --bench` (real devices only).
- The reference solution: add `--solution`. Start again: `python3 labs/pmpp.py reset X2.1 --yes`.

## What the tests do

- **The driver's own parts come first.**
  - `floatToHalfBits` is checked on hand-derived values: ties to even, the largest finite
    FP16, overflow to infinity, subnormals, and 1/127. Then every one of the 65536 FP16
    patterns goes through `halfBitsToFloat` and back.
  - `quantizeRowQ8_0Ref` is checked on a hand-derived block (`qfixture.h`). Its largest
    magnitude is 127/32, so d = 1/32 exactly (FP16 `0x2800`), and its codes include the
    ties 2.5 and -2.5, which `roundf` sends to 3 and -3. An all-zero block must give d = 0
    and all codes 0.

  The lessons' simulator test is meant to use the same fixture blocks.
- **Shapes (batch, in, out):** (1, 32, 1), (1, 64, 5), (2, 96, 7), (3, 256, 64),
  (8, 512, 130), (5, 4096, 3). On a GPU also (1, 4096, 4096), (8, 4096, 4096),
  (4, 14336, 4096), (1, 4096, 14336) and (8, 1024, 4099).
- **The weights:**
  - Block i of row j is scaled by 2^((3j + 5i) mod 25 - 12), between 2^-12 and 2^12.
    Neighbouring blocks then differ 32 times in magnitude, so a scale taken from the wrong
    block gives a wrong answer.
  - Every block holds both +peak and -peak, so codes 127 and -127 occur in every block.
  - With out >= 3, one row is all zero (d = 0).
  - The smallest scales are FP16 subnormals, which `halfBitsToFloat` must decode.
- **Exact sizes.** X, Wq and Wd hold exactly their bytes, so on `emu` a read past any of
  them is an AddressSanitizer report naming the line.
- **Y is checked twice.** Y starts as NaN, so an element nobody writes fails. It is followed
  by a guard band of sentinels that must survive. Every case launches twice without
  resetting Y.
- **The comparison.** Results are compared with `y = x dequant(W)^T` computed in double on
  the CPU, from the same FP16 scales the kernel reads. The tolerance is
  1e-4 x (1 + sum of |x w|) for that output, never bitwise: the kernel adds in another
  order, and Metal does not default to strict IEEE arithmetic.

## What each backend can catch

- **`emu`** runs one thread at a time under AddressSanitizer. It catches wrong indices, a
  scale from the wrong block, a byte that is not sign-extended, reads past Wq or Wd, and a
  lane that skips the reduction. There are no shared updates here, so there is no `race`
  build.
- **Metal** checks the same results on your GPU. It reads codes as `char4` and scales as
  `half`, so a mistake in `codeOf` or `halfBitsToFloat` cannot show up there. Those belong
  to the CUDA-family file, and `emu` checks them.
- **Nothing here checks a 64-lane warp.** The emulators model 32-lane warps, and Metal
  reports its own width. Check by reading: the stride is `warpSize`, and the only 32 is
  `QK8_0`.

## Where the backends differ (not in the book)

- **No `__dp4a` here.** The activations are float, so the inner product is float
  multiply-adds on the converted codes.
  - CUDA's `__dp4a` (sm_61 and later, A2) does not exist under HIP. HIP has
    `amd_mixed_dot` and target-specific builtins instead (B1, B2).
  - Metal has no per-thread integer dot product at all (C1).
  - llama.cpp's CUDA and HIP path quantises the activations to int8 every token to use an
    integer dot product (D3).
  - llama.cpp's Metal path stays in float, like this lab (D4).
- **FP16 scales.** CUDA and HIP would use `__half2float`, from two different headers (the
  table in x2-facts). The emulators and the front-end stub do not model `__half`. So the
  `.cu` keeps the bits in `uint16_t` and decodes them with `halfBitsToFloat`. Metal has
  `half` built in.
- **Alignment.** `char4` has size 4 and alignment 4 on CUDA (A5) and on Metal (infra-facts
  C9). That is why the codes live in their own array. Lab X2.2 reads llama.cpp's real
  18-byte blocks, where that is not possible.
- **`dot(half4, half4)` returns `half`** in MSL (C1), the narrow accumulator appendix A
  warns about. The Metal kernel converts to `float4` first.
- **HIP** has not been built or run here: unverified.

## Before you benchmark, predict

Write these down first:

1. The speed-up of Q8_0 over F32 at batch 1 on the 4096 -> 4096 layer. The bytes model's
   ceiling is 4 / 1.0625 = 3.76. That is the ratio of bytes stored, if bandwidth were the
   only cost. It is not a promise.
2. Whether the speed-up at batch 8 is larger or smaller than at batch 1, and why. Weigh
   two things:
   - The activation bytes grow with the batch, and they are the same for both formats.
   - The work of turning a code into a float and applying its scale is done once per
     weight and shared by the 8 rows.

## Reading the output

- **Layers:** 4096 -> 4096, 4096 -> 14336 (an MLP up-projection) and 14336 -> 4096 (a
  down-projection). The last two shapes are ours. Batches 1, 4 and 8.
- **Kernels:** `gather` (F32, lab 22.1's solution) and `q8_0` (yours). They take turns: one
  untimed round, then five rounds, and each kernel keeps its best round.
- **Copies of W.** One timed submission launches the kernel once per copy of its weights,
  going through copies that together hold at least 256 MB per format (as lab 22.1). A
  launch therefore does not find its weights where the previous launch left them. The
  header of each layer says how many copies. `time (us)` is per launch. CUDA and HIP time
  on the host clock; Metal uses the command buffer's GPU start and end times.
- **`W GB/s`** is the bytes each format stores (Q8_0: codes and scales) divided by the
  time. It is not everything the memory system moved. The labs page compares it with lab
  5.1's measured bandwidth, the speed limit of a matrix-vector product.
- **`vs F32`** is the F32 time over this kernel's time.
- **Report rows:** `qgemv`: `{kernel, format, batch, in, out, seconds, weight_bytes,
  weight_gb_per_s, speedup_vs_f32}`.

Then compare your numbers with your prediction. Where does the measured speed-up fall
short of the byte ratio, or pass it? Remember the two kernels differ in more than the
weights: this kernel reads X four floats at a time, and lab 22.1's reads one. The numbers
are measurements of your device at these shapes, not properties of the kernels.

## Go further

- **Integer activations, as llama.cpp's CUDA path does** (x2-facts D3). Quantise each block
  of 32 activations to int8 with its own scale, once per token. Then the inner product
  becomes an integer dot product with one float rescale per block. On CUDA, try `__dp4a`
  behind `#if` (A2). The emulators and the stub do not declare it, so keep a four-product
  fallback. Measure both the extra pass and what it buys.
- **llama.cpp's own layout.** Read llama.cpp's interleaved 34-byte `block_q8_0`. Your
  loads become 2-byte aligned (A5), so read the codes as 16-bit words, as lab X2.2 does.
  Measure what the split layout was worth.
