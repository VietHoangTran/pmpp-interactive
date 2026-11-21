# Lab X3.1: the Llama pieces the course has not built, and a port review (pieces for milestone M10)

Not in the book: module X3 is ours. The reduction is lab 10.3's (the book's two-stage
reduction, section 10.8, Fig. 10.16, p. 243, with that lab's corrections). What the kernels
compute, and the conventions the tests check, come from the projects read for this module on
2026-11-20 at the commits pinned in `notes/today/x3-facts.md`. A bare item such as (B2) below
is in that file.

Milestones M0 to M8 give you the elementwise kernels, the GEMMs, softmax and argmax, attention
with a KV cache, the GEMV and the embedding. A Llama decoder step needs three more (project
notes in x3-facts): RMSNorm (B2), RoPE (B6) and the SiLU-gated feed-forward, SwiGLU (B1).
Part A is writing them in their smallest form. Part B is reading code the way you will read
llama.cpp: an RMSNorm in the structure of its kernels, with three defects to find.

- Edit `work/kernel.cu` (CUDA; also built for HIP and the emulators) and `work/kernel.metal` +
  `work/launch.mm` (Metal). The names are fixed: the driver calls `launchRmsNorm`,
  `launchRope`, `launchSwiGLU`, `launchRmsNormPort`, the Metal kernels `rms_norm_rows`,
  `rope_rows`, `swiglu`, `rms_norm_port`, and the encoders `encodeRmsNorm`, `encodeRope`,
  `encodeSwiGLU`, `encodeRmsNormPort`, which encode into the encoder they are given.
- Test: `python3 labs/pmpp.py run X3.1`. Start with `--backend emu`.
- Races: `python3 labs/pmpp.py run X3.1 --backend race`.
- Measure: `python3 labs/pmpp.py run X3.1 --bench` (part A only; make the prediction below first).

## Part A: the contract

Row-major float32, tokens (rows) first, raw pointers and sizes as arguments.

- `launchRmsNorm(x, w, y, rows, d, eps)`: `y[r][c] = x[r][c] * (1 / sqrtf(mean(x[r]^2) + eps)) * w[c]`,
  with `x`, `y` of `[rows x d]` and `w` of `[d]`. `y` may be `x`. One block of 256 threads per
  row, with lab 10.3's block sum: typed lane mask, `warpSize`, stage 2 guarded. This is what
  llama.cpp, MLX and PyTorch compute per row (B2); Hugging Face casts back to the input type
  before the weight multiply, which in float32 changes nothing.
- `launchRope(x, pos, cosT, sinT, rows, heads, headDim, mode)`: in place on
  `x[rows x heads x headDim]`, `headDim` even. Row `r` is at position `pos[r]`. The tables are
  `[maxPos x headDim/2]`, built by the driver in double with angle `p * 10000^(-2i/headDim)`
  (the angle llama.cpp, Hugging Face and llama2.c use, B6, D1) and rounded to float once, so
  the kernel does no trigonometry. Pair `i` of a head uses table entry `i` and maps
  `(a, b)` to `(a cos - b sin, a sin + b cos)`:
  - mode 0 rotates elements `(2i, 2i+1)`: llama.cpp's NORM (`rope_norm`), llama2.c;
  - mode 1 rotates elements `(i, i + headDim/2)`: NeoX, Hugging Face's `rotate_half`, mlx-lm's
    default (`rope_traditional = False`).
- `launchSwiGLU(gate, up, out, n)`: `out[i] = gate[i] / (1 + expf(-gate[i])) * up[i]`
  (llama.cpp's `op_silu`, B1). `out` may be `gate`.

Metal: `rms_norm_rows` (one threadgroup of 256 per row, `simd_sum`, the SIMD-group width read
from the pipeline), `rope_rows` and `swiglu`, with the same contracts. `encodeRmsNorm`
returns false, without dispatching, when the pipeline cannot run 256 threads in whole
SIMD-groups with no more SIMD-groups than lanes.

### The two RoPE pairings

They are not two ways of writing the same thing: on the same input they give different
outputs. A checkpoint trained with one runs with the other only after its Q and K weight rows
are permuted. llama.cpp runs Llama with NORM and permutes Hugging Face's weights on
conversion (B6):

```python
def permute(weights, n_head, n_head_kv):
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2).reshape(weights.shape))
```

applied to `q_proj.weight` with `(n_head, n_head)` and to `k_proj.weight` with
`(n_head, n_kv_head)`. Within each head it sends row `k * headDim/2 + j` to row `2j + k`. The
driver checks the identity this buys, on your kernel: mode 1 on `W x` equals mode 0 on
`(P W) x` with the permutation undone, for Q (4 heads) and for K (2 KV heads under 4 query
heads). It also checks that permuting K with the query head count breaks the identity
(ours; x3-facts D6 reads the same mistake in llama2.c's `export.py`). Ask yourself before
running it why the same table entry `i` serves pair `(2i, 2i+1)` in one mode and
`(i, i + headDim/2)` in the other.

### SwiGLU and infinity

The test feeds `gate = -100`. In float `expf(100)` overflows to infinity. With CUDA's default
IEEE division, `-100 / infinity` is `-0`, so llama.cpp's formula gives a finite result. Under
Metal's fast math, which may be this runtime's default (ch05-facts C7), the behaviour of
infinity is undefined (ch10-facts C3): write the Metal kernel so that nothing overflows. The
same function has a form in which the exponential's argument is never positive.

## Part B: the port

`work/kernel.cu` (`rmsNormPortKernel`, `launchRmsNormPort`) and `work/kernel.metal`
(`rms_norm_port`, with `encodeRmsNormPort` in `work/launch.mm`) hold an RMSNorm a colleague
ported in the structure of llama.cpp's kernels (B2):

- the launch picks 256 threads for rows under 1024 elements, else 1024 (llama.cpp's scalar
  path), and a float4 path when it judges it safe;
- a fixed grid of 4 blocks walks the rows (block `b` does rows `b`, `b + 4`, ...), so one
  block reuses its shared buffer row after row; llama.cpp itself runs one block per row;
- the reduction is two stages through one shared buffer `s_sum` of 64 slots (one per lane of
  the widest warp here); the scale is handed to the block through `s_sum[0]`.

It compiles, and it passes some tests. It has **three independent defects**, the same three
in both files; on Metal one of them is in `launch.mm`, because there, as in llama.cpp, the
host picks the float4 path. Find and fix all three. The function `launchRmsNormPort` takes the
same arguments as `launchRmsNorm`, and must produce the same result for any `rows`, `d >= 1`
(out of place).

Not defects: the typed lane mask and `warpSize`; the `float*` to `float4*` cast in
`kernel.cu`, which is the book's technique and is documented by no vendor (infra-facts A6);
binding each Metal buffer a second time as `device float4*`, Metal's documented way to read
four floats at once (infra-facts C7); the 64 slots; the grid of 4; the block-size rule; `1 /
sqrtf`; reading the scale from `s_sum[0]`.

Method:

1. `python3 labs/pmpp.py run X3.1 --backend emu` first. Its three thread orders are
   deterministic; shared memory starts filled with the byte `0xCD` in every block, so a value
   nobody wrote shows as garbage; and AddressSanitizer stops the run at the first read past an
   allocation and names the line. The port's test cases are ordered so the cases that read
   past an allocation come last.
2. Fix one thing, run again. The failures change as you go.
3. `python3 labs/pmpp.py run X3.1 --backend race` runs the threads at once under
   ThreadSanitizer and names the two source lines of a write and a read that nothing orders.
   It matters here: the emulator runs each thread to its next synchronisation point (a
   barrier or a warp shuffle) before the next one runs, so it shows a missing barrier only
   when a thread writes the shared buffer before its own next synchronisation point.
   ThreadSanitizer sees it either way, but only in schedules that occurred.
4. Then run on your GPU: `python3 labs/pmpp.py run X3.1`. Note which defects the GPU showed
   you and which it hid. Shared or threadgroup memory nobody wrote holds whatever it holds;
   what a GPU puts there is not something any facts file we checked documents.

The cases, and what each shape is for: rows 1 and 3 (one row per block) and rows above 4
(several rows per block; 6 on `race`, 9 on `emu`, 17 on a GPU); `d` 64 and 4096 (multiples of
4, with 256 and 1024 threads); `d` 577, 1001 and 4097 (not multiples of 4). On real devices
and on `race`, `x` and `w` are followed by a guard band of 1000.0, so a read past their ends
changes the result instead of reading zeros; on `emu` they are allocated to their exact size
so that AddressSanitizer catches the read.

When you have found all three, compare with llama.cpp: each is something its kernels guard
against (B2; ch10-facts D1, where its softmax carries the comment "sync is needed as we
reuse buf_iw across block_reduce invocations, see #26385").

## What the tests check

- RMSNorm: rows 1, 3 and 17 by `d` 1, 7, 64, 128, 576 (SmolLM2-135M's width, D4), 1000 and
  4097, in place and out of place; rows of different scales in one batch, so a kernel that
  uses another row's sum is caught; an all-zero row, whose output must be 0; `eps = 1e-5`
  (llama2.c and SmolLM2, D1, D4). Outputs start as NaN, 64 guard values after `y` must be
  untouched, and `w` (and `x`, out of place) must come back unchanged. The race backend runs
  rows 1 and 3 by `d` 1, 7, 64 and 1000.
- RoPE: rows 1 and 3 at distinct positions that include 0 and `maxPos - 1` (`maxPos` 4096 on
  a GPU, 512 on `emu`, 64 on `race`); 1, 2 and 4 heads; `headDim` 8, 64 and 128 (64 and 8 on
  `race`); both modes against the reference; position 0 must return the input exactly in
  both modes; the two modes must give different outputs; the permutation identity above.
- SwiGLU: `n` 1, 31, 1000 and 4099 (not 4099 on `race`), out of place and `out == gate`;
  gates uniform in [-8, 8] with -100 and +100 planted at the ends; every result finite.
- The port: the cases of part B.

## Tolerances, and why

Every result is compared with a reference computed in double on the CPU from the same float
inputs, as `|got - ref| <= k * FLT_EPSILON * (1 + |ref|)` (`FLT_EPSILON` is 2^-23; one
rounding is at most half of it, relative). The `1 +` keeps the bound meaningful where `ref`
is 0. Error bounds below are in units of `FLT_EPSILON`, relative to the value.

- RMSNorm, `k = 32`. The sum of squares: a thread adds at most `ceil(d / 256)` squares in
  turn (17 at `d = 4097`), then 5 levels of the warp tree and 5 more in stage 2 (6 and 6 with
  64-lane warps). Every term is positive, so the sum is off by at most one half per addition
  on the longest path, plus the squaring: about (17 + 12 + 1) / 2 = 15. The scale takes half
  of that through the square root, plus the division by `d`, the `+ eps`, the square root and
  the reciprocal: on CUDA `sqrtf` and `/` are correctly rounded by default (ch21-facts A10);
  under Metal's fast math `x / y` is within 2.5 ulp, `1.0 / x` within 1 and `sqrt` is
  `x * rsqrt(x)` with `rsqrt` within 2 (ch21-facts C3), at most about 7 together. Then two
  products. Total about 7.5 + 7 + 1, under 16; `k = 32` leaves twice that. The order in which
  `simd_sum` adds is not documented (ch10-facts A4, C1): even if it added its 32 lanes one
  after another the bound would be about 25.
- RoPE, `k = 8`. Each output is two products and a sum of values at most 1 in size: under 3,
  with or without a fused multiply-add. The reference reads the same float tables, so their
  rounding is not counted; the position-0 test is exact.
- SwiGLU, `k = 16`. CUDA: `expf` within 2 ulp (ch10-facts A11), then an add, a correctly
  rounded division and the product: under 4. The Metal form with `e = exp(-|g|)`: fast `exp`
  is within `3 + floor(|2x|)` ulp (ch10-facts C4), large where `|g|` is large, but there `e` is
  small: for `g >= 0` it moves `1 / (1 + e)` by at most about 1.5; for `g < 0` the result
  itself is small, and the absolute error stays under about 4 over [-8, 8]. HIP publishes
  measured differences against the C++ library, not bounds (ch10-facts B4, ch21-facts B2):
  unverified there.

A wrong row, a wrong pairing, a missing weight or a reduction that read garbage miss by
orders of magnitude.

## Before you benchmark

Write down your prediction (the lesson asks for it as `x3/labs/p-lab1-rmsnorm-bw`): what
fraction of your lab 5.1 bandwidth will `launchRmsNorm` reach at rows = 1, d = 4096, and at
rows = 256? At rows = 1 one call moves about 48 KiB: `x` and `y`, 16 KiB each, and the 16 KiB
weight.

The forces to weigh, not the answer:

- One row is one block, and one block runs on one of the GPU's cores; the rest wait.
- Every launch costs a fixed time, whatever it moves.
- The benchmark calls the kernel many times on the same buffers, so data that fits in the
  caches need not come from memory after the first call. Lab 5.1 streamed a working set far
  larger than any cache.

## The benchmark

Real devices only. `launchRmsNorm`, `launchRope` (mode 0) and `launchSwiGLU` at rows (tokens)
1, 16 and 256 for `d` = 576 and 4096. RoPE runs over heads of 128 where `d` is a multiple of
128 (4096 = 32 x 128) and heads of 64 otherwise (576 = 9 x 64, SmolLM2-135M's shape, D4);
SwiGLU over `n = rows x 4d`. GB/s counts the unique bytes each call must read and write, once:
RMSNorm `x`, `w` and `y`; RoPE `x` twice, `pos` and the cos and sin rows of the rows'
positions; SwiGLU `gate`, `up` and `out`. Time per call is the best of 5 timed runs after an
untimed one, each run repeating the call back to back until it lasts at least 2 ms (lab
10.3's method); the time is the device's (CUDA and HIP events, Metal's command buffer
`GPUStartTime` to `GPUEndTime`). Each timed result is checked before it is reported. Where
`reports/runs/<backend>.json` holds a lab 5.1 bandwidth (your work's if present, else the
solution's), each row also shows its fraction of it. The numbers are measurements of your
machine at these sizes, not targets.

## Portability

- HIP: `kernel.cu` builds through `pmpp/vendors/hip.h` with the typed mask and `warpSize`.
  With 64-lane warps a block of 256 has 4 warps; the port's 64 slots cover a 64-lane stage 2.
  Not compiled or run on AMD hardware here: unverified.
- Metal: the SIMD-group width comes from the pipeline (`threadExecutionWidth`), never a
  constant; threadgroup memory lengths are multiples of 16 bytes (ch05-facts C8). Whether a
  buffer bound at offset 0 starts on the 16-byte boundary a `float4` needs is not stated in
  the facts files we checked: unverified, though every test here passes through it.
- `expf` and `sqrtf` need no declaration of their own: the course's CUDA front-end check
  declares the single-precision math functions in its stub header (an earlier version of
  this file declared them under `PMPP_CUDA_FRONTEND_CHECK`, from before the stub had them).
