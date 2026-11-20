# Lab 22.2: the batching study (project milestone M8)

Book: section 22.5 (p. 539) for batching, section 22.2 (pp. 532-533) for the embedding's
decomposition, section 22.3 (pp. 537-538) for why the loop stays on the device. The chapter
prints no code: every kernel and host function here is ours. A bare item such as (E1) below
is in `notes/today/ch22-facts.md`.

Section 22.5 says that batching the queries of many users turns each projection's
vector-matrix products into one matrix product that reuses the weights: each query waits
longer, the group finishes sooner. This lab builds one greedy decode step from the
project's own pieces, for `batch` sequences at once, and measures how much sooner on your
device, and where it stops paying.

## Your part

(a) `launchEmbed(E, tokens, H, batch, d)`: `H[r][j] = E[tokens[r]][j]`, one thread per
element of H. Output-centric (section 22.2): the thread that owns `H[i]` gathers the one
float it needs. Two sequences on the same token read the same row, and reads never
conflict. The token ids stay in device memory, where the previous step's argmax wrote
them. Before you write it, ask what the input-centric version would be (one thread per row
of E) and which of the book's criteria (p. 533) rules it out.

(b) `launchDecodeStep(m, s)`: one decode step for `s.batch` sequences, one launch per
operation for the whole batch, nothing waiting inside the step:

| Operation | Piece | Shapes |
|---|---|---|
| `H = E[tokens]` | yours, `launchEmbed` | `[batch x d]` |
| for each layer: `T = H W_l^T` | `m.linear`: M2, M3 or lab 22.1's gather, chosen by the driver | `[batch x d]` |
| `T += b_l` | M0 `launchAddBias`, in place | |
| `H += T` | M0 `launchAddResidual` | |
| `logits = H E^T` | `m.linear` again: the LM head's weight is E | `[batch x vocab]` |
| `probs = softmax(logits / temperature)` | M4 `launchSoftmaxRows`, scale `1 / s.temperature` | `[batch x vocab]` |
| `tokens = argmax(logits)` | M4 `launchArgmaxRows`, over the tokens the step read | `[batch]` |

`decode.h` declares every piece with its argument order. On Metal the same step is
`embed_rows` in `kernel.metal`, and `encodeEmbed` and `encodeDecodeStep` in `launch.mm`,
which put every dispatch into the caller's compute encoder; the whole step is one command
buffer.

## Ours, not the book's

- **The model.** A decoder cut down to what batching changes: embedding, `layers` blocks of
  linear, bias and residual, the LM head, the sampler. The shapes are ours. The embedding is
  tied to the LM head (`Wout = E`), which saves a `vocab x d` matrix; E is already stored
  `(out, in)` for the LM head, so no transpose is needed.
- **No attention.** Chapter 20 builds it (M6, M7). It is left out because it does not batch:
  each sequence has its own keys and values, so there is no shared weight for the batch to
  reuse (A6, which also explains the looser wording on p. 539). In a real step it belongs
  with the parts whose cost grows with the batch from the start, beside the elementwise and
  softmax pieces.
- **Probabilities.** Greedy decoding reads only the argmax. The step computes `probs`
  anyway, because the sampler of milestone M5 reads them, so their cost belongs in the
  measurement.
- **Tokens in place.** The argmax writes the next tokens over the ones the step read, so the
  next step starts on the device with no copy to the host and back. Section 22.3's point is
  that whatever stays on the host bounds the speed-up; a host round trip per token is such a
  remainder (the mapping to decoding is ours).
- **Serial order.** Each launch sees what the one before it wrote: CUDA and HIP launches
  without a stream run in issue order (ch02-facts A5, B3). On Metal the plain compute encoder
  "dispatches its compute commands serially" and buffers made by the device are
  hazard-tracked (ch02-facts C9). In a concurrent encoder the step would need a memory
  barrier after each dispatch the next one reads (C1).

## Files

- `work/kernel.cu` (CUDA, HIP, emulators): `embedGatherKernel`, `launchEmbed`,
  `launchDecodeStep`.
- `work/kernel.metal`: `embed_rows`. `work/launch.mm`: `encodeEmbed`, `encodeDecodeStep`.
- Provided, not edited: `decode.h` (`DecodeModel`, `DecodeState`, `LinearFn` and its Metal
  counterpart `EncodeLinearFn`, the pieces' declarations); `provided/pieces.cmake`, which
  copies the reference solutions of lab 2.3 (M0), lab 6.4 (M2, renamed `launchLinearM2`,
  `encodeLinearM2`), lab 10.3 (M4), lab 15.6 (M3, renamed `launchLinearM3`,
  `encodeLinearM3`, if present) and lab 22.1 (`launchGemvGather`, `encodeGemvGather`, if
  present) into the build directory; `provided/step_timer.cu`, the event pair that times a
  step on CUDA and HIP. The driver compiles each piece's Metal kernel from that lab's own
  `kernel.metal`, with the constants that lab's driver uses (TILE_WIDTH 16 for lab 6.4;
  SIMD_WIDTH from the pipeline's `threadExecutionWidth` for lab 15.6; lab 10.3's checks for
  256 threads in whole SIMD-groups).
- `main.cpp`: the tests, the benchmark, and the CPU reference `cpuDecodeStep`.

## Commands

- Test: `python3 labs/pmpp.py run 22.2 --backend emu` first, then `python3 labs/pmpp.py
  run 22.2` on your GPU. The emulator run takes several minutes: it runs every thread of
  every launch of every step, under three thread orders, and lab 22.1's gather does a
  warp shuffle per sequence per output.
- Measure: `python3 labs/pmpp.py run 22.2 --bench`.

## What the tests do

- Model: on the emulators d = 48, vocab = 301, 2 layers, batches 1, 2, 3 and 5, four
  steps; on a GPU d = 256, vocab = 4099, 2 layers, batches 1, 2, 3, 8, 17 and 64, six
  steps. Each batch runs with every GEMM present: m2 always, m3 when lab 15.6's solution
  exists, the gather (batch 8 and below) when lab 22.1's exists.
- `launchEmbed` alone, on batches that hold token 0, token `vocab - 1` and a token used
  twice: every row of H must be bit for bit the row of E, the tokens untouched.
- After every step, for every sequence: the token equals the argmax of a CPU reference
  computed in double from the same float weights; H and the logits agree with it within
  `1e-4 * (1 + mag)`, where `mag` is the sum of the absolute values of every product and
  bias that went into the element; the probabilities agree within lab 10.3's tolerance
  (`1e-20 + 1e-4 * p`) with the softmax, in double, of the logits your step computed.
- The test model is generated so that at every step it is tested on, the reference's top
  logit beats the runner-up by more than `1e-5 * (2 + mag)` of the two, far more than
  float32 rounding moves a logit; a seed that fails this is replaced by the next one. Ties
  would go to the lowest index (M4's rule), but the margin rules them out.
- Batch invariance, once per GEMM, after that batch has passed: every sequence's tokens in
  the batch equal its tokens when it is decoded alone. On a GPU the batch is the largest
  tested; on the emulators it is batch 3, to keep the run short.
- The weights are unchanged after the run. H, T, logits, probs and tokens are each followed
  by a guard band that must be intact, and the outputs start as NaN, so a value nobody
  wrote cannot pass. The weights are allocated to their exact size, so on `emu` a read past
  one is an AddressSanitizer error that names the line.
- Not on the `race` backend: your code has no shared updates (every thread writes its own
  element), and the pieces were race-tested in their own labs.

## Before you benchmark, predict

Take your lab 5.1 threshold T (section 22.5's explorer shows it) and write down:

1. The batch at which tokens per second, with M3, stops rising by at least 1.5x per
   doubling of the batch.
2. Tokens per second at batch 64 divided by tokens per second at batch 1, with M3.

Things to weigh:

- The model puts the knee of a float32 projection near 2T: the intensity of
  `[B x K] x [K x N]` is about B/2 FLOP/B while B is small next to K and N (E1). The
  benchmark prints the step's model FLOP/B at each batch; compare it with T.
- M3's block tile is 128 rows of the batch. Every batch from 1 to 128 launches the same
  grid and does the same arithmetic, on zero-filled rows, so its time per step stays flat
  over that range whether or not the step is bandwidth-bound (E3). Compare with M2's rows
  (tiles of 16) and, at small batches, with the gather's: llama.cpp switches from
  matrix-vector to matrix-matrix kernels at about 8 tokens for a reason (D6).
- The elementwise pieces and the softmax grow with the batch from the start: their
  intensity does not rise with B (E2). The 'piece' rows show when they start to matter.
- Caches can hold part of the weights if your device's are large; Apple documents no cache
  sizes (infra-facts C13). That would put the weight bandwidth at batch 1 above what lab 5.1
  measured.
- On Metal, lab 5.1 is the only threshold you have: Apple publishes no peak FLOPS or
  bandwidth (C13).

## What the benchmark measures

Real devices only. The model is d = 4096, 2 layers, vocab = 32000, float32, the embedding
tied to the LM head (our shape; about 650 MB of weights). If the weights would take more
than a quarter of the memory the device reports, or the allocations fail, d halves until
they fit; the shape and the weight bytes are printed. One step is one command buffer on
Metal (`GPUEndTime - GPUStartTime`) and one pair of events on CUDA and HIP. Each figure is
the best of several timed runs after an untimed one.

- `step` rows, for each GEMM present, at batch 1, 2, 4, ... up to 1024 (the gather up to
  8), stopping when a step takes more than 2 s or an allocation fails: seconds, tokens per
  second (`batch / seconds`), weight bytes and weight GB/s (weight bytes over seconds), and
  the model FLOP/B: the step's ideal FLOPs over its ideal bytes, every array read or written
  once, the GEMMs' `2 B K N` FLOPs plus one per bias and residual element (our count, E1,
  E2). The table also prints each row's tokens per second over the row before it.
- `one_by_one` rows, 1 to 32 sequences: that many separate batch-1 steps, one after
  another in one command buffer or one event pair. This is the book's "sum of all the
  vector-matrix multiplications" (p. 539), beside the batched step for the same number of
  sequences.
- `piece` rows at batch 1, 16, 256 and the largest batch measured: each piece of the step
  (embed, one layer's linear, bias, residual, the LM head, softmax, argmax) timed alone. A
  step runs the linear, bias and residual pieces once per layer.

The rows are measurements of your machine, not expectations. The app finds the knee from
the `step` rows: the first doubling of the batch that raises tokens per second by less
than 1.5x (a definition that is ours). `llama-batched-bench` measures the same quantity on
a real model (D3).

## Portability

- HIP: your `kernel.cu` uses threads, blocks and runtime launches only, which build through
  `pmpp/vendors/hip.h` unchanged. The pieces with warp intrinsics (M3, M4, lab 22.1) use
  typed lane masks and read the warp width from the device. Nothing here has been compiled
  or run on AMD hardware: unverified.
- CUDA: unverified until a `reports/verify-*.md` from a CUDA machine shows this lab passing.
- Metal: M3's SIMD-group width comes from its pipeline's `threadExecutionWidth`, as in lab
  15.6; Apple documents no fixed width (ch15-facts C1).
