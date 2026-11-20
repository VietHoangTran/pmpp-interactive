# Lab X2.4: the quantised decode step (project milestone M9)

Not in the book: module X2 is ours. Its anchor is appendix A section A.5 (p. 593, recorded in
`notes/appA.md`): a smaller number format moves fewer bytes per operation, which raises the
intensity of a kernel that is limited by bandwidth. One decode step is such a kernel: at batch
1 every projection is a matrix-vector product that reads each weight once (`ch22-facts` A5). The
formats are llama.cpp's; the step is lab 22.2's. A bare item such as (D1) below is in
`notes/today/x2-facts.md`.

This lab takes lab 22.2's greedy decode step (M8) and stores its weights in Q8_0 or Q4_0, then
measures, on your device, how much of the byte ratio turns into speed and what it costs in
output error against the float32 step.

## Your part

(a) `launchEmbedQ(E, tokens, H, batch)`: a dequantising gather. One thread per element of H, as
in lab 22.2; the thread that owns `H[r][j]` finds element j of row `tokens[r]` of E, and for a
quantised E reads its code and its block's scale. Q8_0: `code * scale`. Q4_0: `(code - 8) *
scale`, where position k of a block is the low nibble of byte k for k < 16 and the high nibble
of byte k - 16 otherwise (D1: llama.cpp packs weight j with weight j + 16, not two neighbours).
The result must be bit for bit what the CPU's dequantiser gives: one float multiply.

(b) `launchQLinear(X, W, Y, batch)`: `Y = X W^T`, dispatched by `W.fmt` to lab 22.1's
`launchGemvGather` (F32), lab X2.1's `launchGemvQ8_0` or lab X2.2's `launchGemvQ4_0`. Batch at
most 8: these are matrix-vector kernels, as llama.cpp's are up to 8 tokens (`ch22-facts` D3).

(c) `launchQDecodeStep(m, s)`: lab 22.2's step with (a) and (b) in place of its embedding and
its linear layer. One launch per operation for the whole batch, nothing waiting:

| Operation | Piece | Shapes |
|---|---|---|
| `H = dequantised E[tokens]` | yours, `launchEmbedQ` | `[batch x d]` |
| for each layer: `T = H W_l^T` | yours, `launchQLinear` | `[batch x d]` |
| `T += b_l` | M0 `launchAddBias`, in place | |
| `H += T` | M0 `launchAddResidual` | |
| `logits = H E^T` | `launchQLinear` again: the LM head's weight is E | `[batch x vocab]` |
| `probs = softmax(logits / temperature)` | M4 `launchSoftmaxRows` | `[batch x vocab]` |
| `tokens = argmax(logits)` | M4 `launchArgmaxRows`, over the tokens the step read | `[batch]` |

The weights stay quantised in device memory. Nothing in the step writes a dequantised copy of a
weight matrix: that would move more bytes than the float32 step does (A1, where PyTorch's own
measurement of an unfused int8 path came out slower than FP16). The GEMVs dequantise inside
their dot products.

On Metal the same step is `embed_rows_q8_0` and `embed_rows_q4_0` in `kernel.metal`
(`embed_rows_f32` is provided), and `encodeEmbedQ`, `encodeQLinear` and `encodeQDecodeStep` in
`launch.mm`, which put every dispatch into the caller's compute encoder: one command buffer per
step.

## Ours, not the book's

- **The formats.** Q8_0 and Q4_0 are llama.cpp's (D1): blocks of 32 weights along a row, one
  FP16 scale each; Q8_0 codes in [-127, 127], Q4_0 codes 0 to 15 meaning `code - 8`. Q8_0's codes
  and scales are kept in two arrays (lab X2.1's layout; llama.cpp interleaves them in a 34-byte
  block); Q4_0 is llama.cpp's 18-byte block exactly. `../shared-g12/qformats.h` has the structs
  and the CPU rules.
- **Quantised at load.** The driver quantises lab 22.2's float32 toy model with llama.cpp's
  `_ref` quantisers (`quantize_row_q8_0_ref`, `quantize_row_q4_0_ref`, D1), the rule a model
  file is made with when no importance matrix is used. Nothing is downloaded.
- **Format pairs.** Every layer in one format, the tied embedding and LM head in another:
  `(F32, F32)`, `(Q8_0, Q8_0)`, `(Q4_0, Q4_0)` and `(Q4_0, F32)`. The last pair keeps the
  32,000 x 4,096 head in float32, and measures what that costs in time and buys in error.
- **The model** is lab 22.2's: embedding, two blocks of linear, bias and residual, the tied LM
  head, the sampler; no attention (chapter 20 builds it).

## Files

- `work/kernel.cu` (CUDA, HIP, emulators): `embedRowsQ8_0Kernel`, `embedRowsQ4_0Kernel`,
  `launchEmbedQ`, `launchQLinear`, `launchQDecodeStep`. `embedRowsF32Kernel` and `halfToFloat`
  are provided.
- `work/kernel.metal`: `embed_rows_q8_0`, `embed_rows_q4_0` (`embed_rows_f32` provided).
  `work/launch.mm`: `encodeEmbedQ`, `encodeQLinear`, `encodeQDecodeStep`.
- Provided, not edited: `qdecode.h` (`QFormat`, `QMatrix`, `QDecodeModel`, `QPipelines`, the
  pieces' declarations; `DecodeState` comes from lab 22.2's `decode.h`);
  `provided/pieces.cmake`, which copies the reference solutions of lab 2.3 (M0), lab 10.3 (M4),
  lab 22.1 (the gather), lab X2.1 and lab X2.2 into the build directory, modelled on lab 22.2's;
  `provided/step_timer.cu`, the event pair that times a step on CUDA and HIP. If lab X2.1's or
  X2.2's solution is missing or has another signature, `../shared-g12/qpieces.cmake` builds a
  stand-in GEMV of the same shape instead; the first lines of every run say which was used.
- `main.cpp`: the tests, the benchmark, the CPU reference step in double on the dequantised
  weights, and the deterministic toy-model generator.

## Commands

- Test: `python3 labs/pmpp.py run X2.4 --backend emu` first, then `python3 labs/pmpp.py run
  X2.4` on your GPU. The emulator run takes minutes: it runs every thread of every launch of
  every step under three thread orders, for four format pairs.
- Measure: `python3 labs/pmpp.py run X2.4 --bench`.

## What the tests do

- Toy model: d = 64, vocab = 96, 2 layers (both multiples of 32). Batches 1, 3 and 8 on the
  emulators, 1 to 8 on a GPU; 3 steps on the emulators, 4 on a GPU; every format pair.
- The embedding alone, for E in each format, on a batch of 7 that holds token 0, token
  `vocab - 1` and a token used twice: every row of H must be bit for bit the CPU's dequantised
  row of E.
- After every step, for every sequence: H and the logits against a CPU step computed in double
  on the dequantised weights (the floats the CPU's dequantisers give), within lab 22.2's
  tolerance `1e-4 * (1 + mag)`; the probabilities within `1e-20 + 1e-4 * p` of the softmax, in
  double, of the logits your step computed. The next token must equal the reference's argmax
  wherever the reference's two best logits are further apart than their two tolerances
  together; there, any correct step must choose it. The run prints how many tokens that
  compared.
- Batch invariance, after the batch has passed: every sequence of the largest batch decodes, step
  by step, exactly as it does alone.
- The weights (every format's copy) and the biases are unchanged after the run. H, T, logits,
  probs and tokens are each followed by a guard band that must be intact, and the outputs start
  as NaN. The weights are allocated to their exact size, so on `emu` a read past one is an
  AddressSanitizer error that names the line.
- On a GPU, also lab 22.2's default shape (d = 4096, 2 layers, vocab = 32,000), one batch of 8,
  two steps, every pair.
- Not on the `race` backend: your code has no shared updates (every thread writes its own
  element), and the pieces were race-tested in their own labs.

## Before you benchmark, predict

1. The speed-up of the `(Q4_0, Q4_0)` step over the `(F32, F32)` step at batch 1. Start from the
   byte ratio of the whole step (section X2.7's explorer computes it): every weight is read once,
   so the model predicts the ratio of weight bytes, about 7.1 for Q4_0 against float32 (E1).
   Then the same for `(Q4_0, F32)`, where the 32,000 x 4,096 head stays in float32: the bytes
   model says 1.21. Write down how much of each ratio you expect to survive.
2. Of 32 greedy steps from fixed start tokens, how many agree with the float32 step at Q8_0 and
   at Q4_0?

Things to weigh:

- A quantised GEMV does more work per byte than lab 22.1's float gather: it unpacks codes, reads
  scales and multiplies by them. Whether that work hides under the memory traffic is what the
  measurement tells you (A1: a published measurement kept well short of the byte ratio).
- The activations, the outputs, the biases, the softmax and the argmax do not shrink with the
  weights. At batch 1 they are small next to the weights; they grow with the batch.
- The prediction of 1.21 is a statement about bytes. If the head dominates the step's bytes, it
  dominates its time too, unless something else limits the step.
- The toy model is random and has no attention: its next token depends only on its current one.
  Its agreement says nothing about a trained model's, and its logit error says only how far
  rounding moves these particular weights.

## What the benchmark measures

Real devices only. Lab 22.2's default shape and seeds: d = 4096, 2 layers, vocab = 32,000, the
embedding tied to the LM head. All three formats of every matrix are on the device at once; if
they would take more than a quarter of the memory the device reports, d halves until they fit,
and the shape is printed. Batches 1, 2, 4 and 8. One step is one command buffer on Metal
(`GPUEndTime - GPUStartTime`) and one pair of events on CUDA and HIP; each figure is the best of
several timed runs after an untimed one.

Rows `qstep`: `format_layers`, `format_head`, `batch`, `seconds`, `tokens_per_s`
(`batch / seconds`), `weight_bytes` (E once for the head, every W_l, the biases: what one step
must read), `weight_gb_per_s`, `speedup_vs_f32` (the `(F32, F32)` step's time at the same batch
over this one's), `logits_rel_rms` (after the first step, `sqrt(sum (l - l_f32)^2 / sum
l_f32^2)` over the batch), `agree_steps` (the number of greedy steps, from the first, in which
every sequence chose the same token as the float32 step; 32 means none differed), and the shape.
The labs page adds bandwidth utilisation against your lab 5.1 measurement. These are
measurements of your machine, not expectations.

## Portability

- HIP: your `kernel.cu` uses threads, blocks and runtime launches only, which build through
  `pmpp/vendors/hip.h` unchanged; the FP16 scales are converted from their bits by
  `halfToFloat`, so no `__half` header is needed (the `cuda_fp16.h` and `hip_fp16.h` headers
  differ, infra B12). The GEMVs you call use a typed lane mask and the device's warp width.
  Nothing here has been compiled or run on AMD hardware: unverified.
- CUDA: unverified until a `reports/verify-*.md` from a CUDA machine shows this lab passing.
- Metal: `half` is a core MSL type, so the kernels read a scale with `as_type<half>` (C5). A
  Q4_0 block is 18 bytes with alignment 2 on Metal too (C5).
