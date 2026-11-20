# Lab A.1: summation error: order, accumulator width and compensation

Book: section A.6 (pp. 593-594). The tree is chapter 10's shared-memory reduction
(Fig. 10.10, p. 237) behind the coarsening loop of Fig. 10.20 (p. 247). Appendix A shows no
code: the kernels, the 16-bit accumulators, the compensated loop, the device probe, the
error bounds and every measurement are ours.

Section A.6 makes two claims you can measure. The order of a float sum changes its result,
so a reduction tree and a loop legitimately disagree (p. 593). And for something sturdier
than reordering there is Kahan's compensated summation, which the book names and does not
describe (p. 594). This lab sums one float32 array three ways, in three accumulator types,
and measures the error of each against a reference and what each costs. It is the same
question a framework answers when it picks an accumulator: PyTorch adds half and bfloat16
sums in float on every device, while MLX's `sum` of a float16 array on Metal, by our reading
of its kernel, stays in float16 (appA-facts D1, D2).

## Your part

Three sums of `n` float32 values, each a template on the accumulator type `AccT`. Every
addition is rounded to `AccT`; the result goes to `out[0]` as a float.

- **(a) `sumSerial`**: one thread adds `x[0]`, `x[1]`, ... in index order. This is the CPU
  loop's order, the "sequential" order of p. 593.
- **(b) `sumTree`**, deterministic, no atomics. Thread t of block b first adds its `coarse`
  elements `x[b*B*coarse + t + j*B]`, j = 0 to coarse - 1, in that order (B is the block
  size; the stride keeps a warp's loads coalesced). The block then combines its B sums with
  Fig. 10.10's tree (stride B/2, ..., 2, 1, one barrier per level), and thread 0 writes the
  block's sum to `partials[b]`. A second launch of the same kernel, one block, reduces
  `partials` into `out[0]`.
- **(c) `sumKahan`**: as (b), but each thread's loop is a compensated sum. The tree stays
  plain. The CUDA starter gives the four lines of the loop.

Launchers, one per sum, return false without launching when the shape is outside the
contract (n < 1, coarse < 1, block not a power of two from 1 to 1024) or when this build has
no kernel for the accumulator:

```cpp
bool launchSumSerial(const float* x, float* out, int n, Acc acc);
bool launchSumTree (const float* x, float* partials, float* out, int n, int coarse, int block, Acc acc);
bool launchSumKahan(const float* x, float* partials, float* out, int n, int coarse, int block, Acc acc);
```

`Acc` (`AccFloat`, `AccHalf`, `AccBfloat`) is in `sumerror.h`. Which accumulators exist:

| | float | half | bfloat16 |
|---|---|---|---|
| Metal | `float` | `half` | `bfloat` |
| CUDA | `float` | `__half` | `__nv_bfloat16` (compute capability 8.0 or higher; the driver checks) |
| HIP | `float` | `__half` | `__hip_bfloat16`, named `__nv_bfloat16` by a typedef in `sumerror.h` |
| emu, race | `float` | none | none |

On CUDA and HIP the 16-bit instantiations go inside the `#if PMPP_NARROW_ACC` blocks the
starter provides. The emulators do not model `__half` or `__nv_bfloat16`, and the CUDA
front-end check's stub declares neither (appA-facts E2), so those blocks are compiled by
nvcc and hipcc only.

## Files

- Edit `work/kernel.cu`: `sumSerial`, `sumTree`, `sumKahan` and the three launchers. The
  helpers `addRn` and `subRn` are provided for the compensated loop.
- Edit `work/kernel.metal`: `sum_serial`, `sum_tree`, `sum_kahan`. The nine
  `[[host_name("sum_tree_half")]]` instantiations at the bottom are provided.
- Edit `work/launch.mm`: `encodeSumSerial`, `encodeSumTree`, `encodeSumKahan`. The two passes
  of the tree are two dispatches in the one encoder you are given, as in lab 10.1.
- Provided, not edited:
  - `main.cpp`: the data sets, the CPU reference, the CPU model of your orders, the bounds,
    the tests and the benchmark;
  - `sumerror.h`: `Acc`, the launcher prototypes, and the 16-bit type headers;
  - `fpformat.h`: FP32, FP16 and BF16 rounding in software, the same rules as the explorer
    of sections A.1 to A.4 (`app/src/sim/minifloat.ts`);
  - `../shared/fpprobe.cu`, `../shared/fpprobe.metal`, `../shared/fpprobe_launch.mm`: the
    device probe.

## Commands

`pmpp.py run A.1` does not work (the runner resolves `5.2`-style numbers to chapters only and
answers "matches nothing"): use `appA/lab1`, as every command below does.

- Test: `python3 labs/pmpp.py run appA/lab1 --backend emu` first, then
  `python3 labs/pmpp.py run appA/lab1`. The emulator makes over a hundred launches per test,
  each under three thread orders, so give it a few minutes.
- The tree's barriers under ThreadSanitizer: `python3 labs/pmpp.py run appA/lab1 --backend race`.
- Measure: `python3 labs/pmpp.py run appA/lab1 --bench` (real devices only).
- The reference solution: add `--solution`. Start again: `python3 labs/pmpp.py reset appA/lab1 --yes`.

## The device probe

Before any sum, the driver asks the device three questions with one thread
(`../shared/fpprobe.*`). The inputs come from a buffer, so the compiler cannot work the
answers out at build time (appA-facts E1).

- `1 + 0.75 ULP`: the next float above 1 when the device rounds to nearest, 1 when it
  rounds toward zero. Metal allows either (C2); CUDA's default is to nearest (A9).
- `2^-126 * 0.5`: the subnormal 2^-127 if FP32 denormals are kept, 0 if flushed. nvcc keeps
  them by default (A6); Metal may flush (C2); HIP's default is not documented (B3).
- `(1 + 2^-12)^2 - (1 + 2^-11)`: exactly 2^-24 with one rounding (a fused multiply-add), 0
  with the product rounded first (A9).

The probe is compiled with each backend's default settings. On the emulators it describes
the host CPU and the host compiler, not a GPU.

## How the tests judge a result

**The reference** is a Neumaier-compensated sum in double on the CPU, far more accurate than
any sum here (appA-facts E1). Errors are measured against it, scaled by `S = sum of |x|`.

**The data**, generated from fixed seeds and rounded to the accumulator's format, with every
value below the format's smallest normal number set to 0 (no subnormal inputs; Metal may
flush them, C2):

- `ones`; `ints`, whole numbers from -3 to 3;
- `uniform` on [0, 1);
- `mixed`: magnitudes from 2^-14 to 2^-2, each binade equally likely; `mixed-asc` and
  `mixed-desc` are the same values sorted; `cancel` is `mixed` with random signs;
- `softmax`: `exp(l - max)` for logits l uniform on [-9, 0], with one 1 at a random place;
- `big-small`: 1, then n - 1 copies of half an ULP of 1 in the format (2^-24 for float).

**Sizes.** emu: n = 1, 2, 33, 1000, 4097; coarse 1, 3, 8; blocks 64 and 256. race: n = 33 and
1000; coarse 1 and 3; block 64. Real devices: n = 1, 2, 31, 32, 33, 255, 256, 257, 1000,
4097, 100003 and 2^20; coarse 1, 2, 3, 8, 64; blocks 64 and 256. On Metal a block above the
pipeline's `maxTotalThreadsPerThreadgroup` is skipped, and the skip is printed. Outputs and
partials start as NaN, and every case is launched twice on the same buffers.

**Exact cases (float).** `ones` and `ints` must come out exact in every order: every partial
sum is a whole number of magnitude at most 3n, below 2^24, so no addition rounds.

**Error bounds (float, and the narrow formats on Metal, CUDA, HIP).** The derivation is ours,
the usual forward error analysis of a sum. Every element x reaches the result through some
number d of rounded additions, and each rounding multiplies what passes through it by
(1 + e) with |e| at most u = 2^-p (p = 24 for float, 11 for half, 8 for bfloat16) when
rounding to nearest. Metal may round toward zero instead (C2), where |e| is below 2u, so
the tests use v = 2u throughout. The product of d such factors is within
`g(d) = d*v / (1 - d*v)` of 1, so

`|result - reference| <= g(D) * S + a`, where D is the largest d over all elements.

Adding to the initial 0 is exact, which is where the "- 1"s come from. With
L = log2(block), blocks = ceil(n / (block*coarse)) and c2 = ceil(blocks / block), the
number each pass-2 thread adds:

- serial: D = n - 1;
- tree: D = (coarse - 1) + L + (c2 - 1) + L: the thread's loop, the tree, and the same
  again in pass 2. Storing a partial as float is exact for all three formats;
- Kahan: a compensated loop over c elements is accurate to about 2u plus a term in c*u^2,
  whatever c is. The test allows `K(c) = 2v + 4*c*v^2` for c > 1 (0 for c = 1) and combines
  the four stages: `(1 + K(coarse)) * (1 + g(L)) * (1 + K(c2)) * (1 + g(L)) - 1`. This is a
  first-order estimate with generous margins, not a proof.

`a` allows for a subnormal partial sum flushed to zero: 2n (serial, tree) or 4n (Kahan)
times the format's smallest normal number. Only FP16 data can produce one here.

**Kahan's own test (float).** `big-small` at coarse 64: thread 0 of block 0 adds 1 and then
63 halves of an ULP of 1. A plain loop drops every one of them, about 32 ULPs in all; the
compensated loop must land within 2 ULPs of the reference. This is the case that shows
whether the compiler kept your compensation.

**Narrow accumulators (Metal; CUDA and HIP where built).** The data sets are `uniform`,
`mixed` and `cancel`, and a case whose S reaches 65,504 (the largest half) is skipped, so
every partial sum stays finite in FP16. Then 4,097 ones through serial and tree must have
the same bits as the CPU model of the same order, rounded to nearest or toward zero. The
model is `main.cpp`'s "CPU model": the same additions in the same order, each rounded to the
format in software by `fpformat.h`. A kernel that quietly accumulates in float returns 4097
there.

The untouched starter builds and fails every sum test: its launchers return false without
launching. There are no planted defects; this is a write-and-measure lab.

## What the tests can and cannot see

- **emu** runs the threads one at a time in three orders, under AddressSanitizer: wrong
  indices, an out-of-range read past n, a missing barrier. It has no half or bfloat16, and
  its compiler is the host's, which does not reassociate float additions by default
  (appA-facts B5). So a compensated loop that passes on emu tells you nothing about what a
  GPU compiler does with it.
- **race** runs the threads of a block concurrently under ThreadSanitizer. The tree is a
  shared-memory update between barriers: a barrier in the wrong place is reported with its
  source line.
- **Metal** is the only backend here that tests the narrow formats and the only one with a
  GPU compiler. If `kahan, float, big-small` fails on Metal and passes on emu, the hint below
  says why. Try to find it yourself first: the question is what Metal's compiler is allowed
  to do to `(t - sum) - y` by default.
- **CUDA and HIP** run the same tests on a real device. Whether nvcc reassociates float
  additions is not documented (appA-facts A13); this lab's `big-small` test is the
  measurement.
- The bit-exact check uses ones, so it catches a float accumulator but not every order
  mistake. The benchmark's `model` column compares every data set with the CPU model of your
  order.

<details>
<summary>Hint: Kahan fails on Metal only</summary>

The correction `c = (t - sum) - y` is 0 in real arithmetic. Metal's default math mode, which
the lab runtime uses (default `MTLCompileOptions`), is documented as fast or as relaxed, and
both allow the compiler to reassociate floating-point operations; only the `safe` mode
forbids it (appA-facts C3, ch05-facts C7). The MSL specification warns that reassociation
breaks "code that relies on rounding behavior such as (x + 2^52) - 2^52" (C2), and a
compensated loop is that code. On the machine this lab was built on, the compensated kernel
without a fix was 6.5 ULPs off at n = 1000, coarse 64, where it should be within 2. That is
bit for bit what a plain loop returns on the emulator: the compensation was gone.

The fix is one line at file scope in `kernel.metal`:

```metal
#pragma METAL fp math_mode(safe)
```

The specification also allows the pragma at the start of a compound statement (C3). At file
scope it covers every kernel in the file, so the benchmark compares tree and Kahan under the
same mode.
</details>

## Before you benchmark, predict

Write these down first. The labs page asks for the first three under the ids in brackets.

1. What the half-precision serial kernel returns on your device for 65,536 uniform values
   (`appA/labs/p-lab1-half-loop`). Section A.6's model, which rounds to nearest, stops at
   2048; Metal may round toward zero instead (C2). Work out what that would give, then compare
   with the probe's rounding line. Section A.2's explorer shows the spacing of half at any
   value.
2. How many times longer the Kahan kernel takes than the plain tree kernel on 2^24 floats,
   at the same coarsening factor (`appA/labs/p-lab1-kahan-cost`). The compensated loop does
   four additions per element where the plain one does one; the sum reads 4 bytes per
   element either way.
3. How many ULPs off the plain float tree kernel will be at coarsening factor 4096 on 2^22
   uniform floats, and whether Kahan will be at the same factor (`appA/labs/p-lab1-coarse`).
4. Which order, `mixed-asc` or `mixed-desc`, the serial kernel handles better. The book's
   pre-sorting advice (p. 594) is one answer; say why.
5. Whether your device keeps FP32 denormals and rounds to nearest. The first line of the
   benchmark answers it.

## Reading the output

The benchmark prints four tables. The same rows go to `reports/runs` under
`appA/lab1_sumerror`, where sections A.4 and A.6 and the labs page read them.

- **probe**: `rounding` (`nearest` or `toward-zero`), `keeps_denormals`, `fuses_mul_add`,
  for this device under default compiler settings.
- **accuracy**: n = 65,536, coarse 16, block 256 (or the largest power of two every tree and
  Kahan pipeline allows, on Metal). One row per method, accumulator and data set:
  `result`, `reference`, `abs_err`, `rel_err` (relative to S), `ulps` (in the accumulator's
  format, at the reference) and `model`: `same` if your result has the same bits as the CPU
  model of that order under the probed rounding mode. `differs` can come from a different
  order, from a device that flushed a subnormal partial sum (the model keeps them), or from
  a compiler that reordered your additions. A result that overflows prints as `inf` and is
  stored as `null`: the true sum of 65,536 ones is above the largest half, 65,504.
- **coarse**: tree and Kahan in float on 2^22 uniform floats at coarse 1, 4, 16, 64, 256,
  1024 and 4096: `time ms` (best of the runs), `GB/s` (4 bytes per input float divided by
  that time: the bytes a sum must read, not what the memory system moved), `rel err` and
  `ulps`. At large coarse factors there are few blocks, so expect to trade parallelism for
  accuracy, and see where each starts to cost.
- **time**: serial, tree and Kahan in each accumulator, uniform data, coarse 16. Tree and
  Kahan run on 2^24 floats, or fewer if that would use more than an eighth of the memory the
  device reports (the line says which n ran). Serial runs on 2^20 floats only: one thread
  over 2^24 floats can run into a GPU's command time limit.

Before each of the two timed tables, a quarter of a second of untimed float tree sums keeps
the device busy, so the first kernel timed does not also pay for the device waking up; in
the `coarse` table tree and Kahan alternate for the same reason. Each timed run is one call
of your launcher or encoder, both passes included, and the best of several runs after one
untimed run is reported. The numbers
are measurements of this machine at these sizes, not properties of the kernels. No expected
values are printed or stored.

## Where the backends differ (not in the book)

- **CUDA**: `__fadd_rn` and `__fsub_rn` are the additions NVIDIA documents as never fused
  into a multiply-add (appA-facts A9); whether they also stop reassociation is not stated
  (A13). `__half` and `__nv_bfloat16` use the ordinary operators, each `+` one addition
  rounded to nearest even (A16). BF16 needs compute capability 8.0 or higher (A2).
- **HIP**: the same source through `pmpp/vendors/hip.h`. HIP-Clang defaults to
  `-fno-associative-math`, so the compensation survives a normal build; `-ffast-math` would
  allow the compiler to delete it (B5). `__fadd_rn` exists under the same name (B4). The
  BF16 type is `__hip_bfloat16` from `<hip/hip_bf16.h>` (B1), renamed in `sumerror.h`.
- **Metal**: `half` and `bfloat` are built-in types (`bfloat` from Metal 3.1, C1). Rounding
  may be toward zero and denormals may be flushed (C2): the probe tells you which. There is
  no `__fadd_rn`; the hint above covers what replaces it.
- A double reference on the GPU is possible on CUDA and HIP but not on Metal, which has no
  `double` (C1). The reference runs on the CPU for all three.

## Go further

- **A compensated tree.** Keep the pair (sum, c) through the tree as well: at each level,
  combine two pairs with a TwoSum of the sums and add both corrections. Does the `coarse`
  table's Kahan column change at coarse 1, where the tree does all the work?
- **The book's truncating adder.** Exercise 3 (p. 599) asks for the error of an adder that
  rounds toward zero. On CUDA, `__fadd_rz` is that adder (appA-facts A9): write a serial
  sum with it and compare its error with `__fadd_rn`'s. On HIP the `_rz` forms need
  `OCML_BASIC_ROUNDED_OPERATIONS` (B4); Metal has no per-operation rounding modes (C2).
- **Pairwise in registers.** NumPy's `sum` uses partial pairwise summation (appA-facts D4)
  and PyTorch's CPU sum a cascade of chunks of 16 or more (D1): grouping by position, so
  that similar partial sums meet. Make each thread's loop pairwise over its `coarse`
  elements and compare it with Kahan in the `coarse` table.

## Status of the reference solution

Verified on Metal (tests and benchmark), emulator-checked (`emu` and `race`),
front-end-checked for `kernel.cu` and `../shared/fpprobe.cu`. Unverified on CUDA and on
HIP: no machine with either has run it, and the FP16 and BF16 branches of `kernel.cu` and
`sumerror.h` have not been compiled by any compiler here.
