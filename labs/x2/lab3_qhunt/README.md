# Lab X2.3: bug hunt on the integer path

Not in the book: module X2 is ours, and so are the defects. The kernel's parts are the book's
(chapter 10's warp shuffle reduction, chapter 5's shared-memory hand-off behind a barrier);
what it computes is llama.cpp's CUDA decode path, as read in the facts pass on 2026-11-20. A
bare item such as (D3) below is in `notes/today/x2-facts.md`.

llama.cpp's CUDA and HIP decode does not multiply Q4_0 weights by float activations. It first
quantises the activation vector to Q8_1 (int8 codes in blocks of 32, a scale `d` and the
precomputed `s = d * sum of the codes`), then takes each 32-weight block as an integer dot
product and rescales it to float once (D3):

```
Y[b][o] = sum over blocks k of row o of   d_w * (sumi * d_x - 8 * s_x)
sumi    = sum over the 32 positions j of code_w(j) * code_x(j)          (an exact integer)
```

The weight codes are unsigned (0 to 15, value `(code - 8) * d_w`), and the `- 8` is taken out
once per block through `s` (E4). Byte j of a Q4_0 block holds weight j in its low nibble and
weight j + 16 in its high nibble (D1), which is why one `0x0F0F0F0F` mask on a 32-bit word
gives four consecutive weights and `>> 4` gives the four that are 16 further on.

`work/kernel.cu` and `work/kernel.metal` hold that GEMV in llama.cpp's CUDA shape: one block of
4 warps per output row, the warps splitting the row's blocks, a shuffle reduction per warp,
per-warp partials through shared memory, warp 0 finishing. It compiles. It has **four
independent defects**. Find and fix all four, then measure the integer path against lab
X2.2's float path.

- Edit `work/kernel.cu` (`gemvQ4_0Q8_1Kernel`) and `work/kernel.metal` (`gemv_q4_0_q8_1`). The
  same four defects are in both.
- Provided and correct, not defects: the FP16 conversions, `dot4_i8`, the quantiser
  (`launchQuantizeQ8_1`, `quantize_q8_1`), `launchGemvQ4_0Q8_1`, all of `work/launch.mm`; the
  16-bit loads of the weights (an 18-byte block is only 2-byte aligned, A5) and the 32-bit
  loads of the activations (a 36-byte block keeps them 4-byte aligned, as llama.cpp's
  `get_int_b4` assumes); the unsigned nibbles fed to a signed byte product (0 to 15 is the same
  in both); the typed lane mask and `warpSize`.
- Test: `python3 labs/pmpp.py run X2.3 --backend emu`, then `--backend race`, then
  `python3 labs/pmpp.py run X2.3` for your GPU.
- Measure: `python3 labs/pmpp.py run X2.3 --bench`.

## Your prediction

Before you run anything, read `work/kernel.cu` and write down the defects you can find by
reading. For each one, write down which of `emu`, `race` and your GPU you expect to show it,
and on which test cases:

| Defect | What it is | `emu` | `race` | your GPU |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |

Things to weigh: `emu` runs one thread at a time between synchronisation points and fills
shared memory with `0xCD` bytes; `race` runs threads at once under ThreadSanitizer and names
the two lines that collide, but only for collisions that happened; a GPU is fast enough that a
missing barrier can pass by timing. A defect in integer arithmetic shows only on inputs that
reach it.

## Method

1. Emulator first. Its three thread orders are deterministic, so a synchronisation defect
   fails on every run, and AddressSanitizer names the line of any out-of-bounds access.
2. Fix one thing, run again. The failures change as you go. Use what the failing cases have in
   common: which shapes, which kind of data, which thread order.
3. For the synchronisation defect, `--backend race` names the write and the read that collide.
4. Then your GPU. Note which defects it showed you and which it hid.

## What the tests check

- Shapes `(batch, in, out)`: `(1, 32, 1)` (one block), `(1, 256, 3)`, `(2, 512, 5)`,
  `(8, 1024, 9)` with real weights through llama.cpp's `_ref` Q4_0 quantiser
  (`../shared-g12/qformats.h`) and activations uniform in [-1, 1].
- **The worst-case block**: every weight code 15 and an activation that is a positive
  constant, so every Q8_1 code is 127 and every block's integer sum is 15 x 127 x 32 = 60,960
  (E3). It exists to catch exactly one thing.
- **j against j + 16**: in every block the low nibbles are 15 and the high nibbles 0, or the
  other way round, so a kernel that pairs a nibble with the wrong activation cannot pass.
- **Power-of-two block scales**: `d_w = 2^(k mod 8 - 9)` for block k, exact in FP16, with
  random codes.
- The reference is the same integer path on the CPU: exact integer block sums, then the float
  rescale in double, from the very Q8_1 blocks the provided quantiser wrote. Those blocks are
  first checked against the CPU's own Q8_1 (a code may differ by one and a scale by one FP16
  step, because a GPU may divide one ulp away; Metal's default math is not IEEE-strict,
  ch05-facts C7). Tolerance: `1e-5 * (1 + mag)`, where `mag` is the sum over blocks of
  `|d_w| (|sumi d_x| + 8 |s_x|)`.
- Outputs start as NaN, `Xq` as `0xEE` bytes, and 256 guard bytes past each must be untouched;
  the weights must be unchanged.
- Each case also prints, as information and not a test, how far the integer path is from the
  float path (dequantised weights times the float activations): that is the cost of rounding
  the activations to int8 and `s` to FP16 (D3).
- The float Q4_0 GEMV the benchmark compares against (lab X2.2's solution, or a stand-in if
  that is absent; the first line of the run says which) is tested too, so the comparison is
  between two correct kernels.

On a GPU, a synchronisation defect may pass by timing. On the Metal device this lab was
built on, with that defect planted alone, every case passed (2026-11-20; yours may differ).
Passing on a GPU does not show that a barrier is not needed.

## Before you benchmark, predict

Which path is faster on your device for one token of decode, **with the quantise pass
included**: (a) quantise the activations to Q8_1, then the integer GEMV; or (b) lab X2.2's
fused float GEMV, which dequantises inside the dot product and reads float activations? And
at batch 8?

Things to weigh:

- CUDA has a four-way byte dot instruction, `__dp4a`, from compute capability 6.1 (A2); this
  lab's `dot4_i8` uses it only there. The HIP build takes the plain four-product form, because
  HIP has no `__dp4a` and its integer dot builtins depend on the target (B1, B2). Metal has no
  integer dot instruction at all (C1), and llama.cpp's own Metal decode stays in float (D4).
- The quantise pass reads the activations once and writes a quarter of their bytes plus the
  scales; next to the weights it is small, but it is one more launch.
- Both paths read the same Q4_0 weight bytes, which dominate for one token (E1).

## What the benchmark measures

Real devices only. Shapes `in -> out` of 4096 -> 4096 and 14336 -> 4096 (our choice of
decode-sized projections), batch 1 and 8, random Q4_0 blocks (the kernels' work does not
depend on the values). Successive timed runs read different copies of the weights, so that a
run does not find the previous run's weights in a cache: about 512 MB of copies in all, or a
quarter of the memory the device reports if that is less. The total is printed; Apple
documents no cache sizes (infra-facts C13), so it is a choice, not a fact about your GPU. Each
figure is the best of at least 10 timed runs after an untimed one: one command buffer
(`GPUEndTime - GPUStartTime`) on Metal, one pair of events on CUDA and HIP.

Rows `qhunt`: `path` (`int8_q8_1` or `float_q4_0`), `batch`, `in`, `out`, `seconds_quantise`,
`seconds_gemv`, `seconds_total` (quantise and GEMV in one pass; for the float path, the GEMV
alone), `weight_bytes`, `copies`, `source`. The printed table adds the weight bytes over the
GEMV's time. These are measurements of your machine, not expectations.

## Portability

- HIP: the shared code is threads, blocks, shared memory, a barrier and `_sync` shuffles with a
  typed 64-bit-capable mask, which build through `pmpp/vendors/hip.h`; `dot4_i8` avoids
  `__dp4a`, which HIP does not have (B1). The quantiser's shuffles use width 32, so on a
  64-lane wavefront each half is its own group. Nothing here has been compiled or run on AMD
  hardware: unverified.
- CUDA: unverified until a `reports/verify-*.md` from a CUDA machine shows this lab passing.
- Metal: the quantiser needs a SIMD-group width that is a multiple of 32 (its 32-value groups
  must not straddle two SIMD-groups); `encodeQuantizeQ8_1` refuses to dispatch otherwise. Apple
  documents no fixed width (infra-facts C2, C5).
