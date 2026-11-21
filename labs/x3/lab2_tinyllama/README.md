# Lab X3.2: a Llama-style model end to end (project milestone M10)

Not in the book: module X3 is ours. The model is the decoder of llama2.c's `run.c`, which is
Meta's Llama architecture in about 130 lines of C; the file format is llama2.c's; the error
report is llama.cpp's. A bare item such as (D1) below is in `notes/today/x3-facts.md`.

Every kernel of a Llama decode step now exists in this course: the embedding gather (M8), the
linear layer (M2, M3, lab 22.1's gather), RMSNorm, RoPE and SwiGLU (lab X3.1), the KV cache and
decode attention (M7), the residual add (M0), the argmax (M4) and the sampler (M5). This lab
composes them into one greedy decode step, runs a model that a script in the repository writes
from a fixed seed, and checks every logit and every token against a CPU reference. You write
no arithmetic kernel. You decide the order of the pieces, their arguments, and where each
result lives, which is most of what a framework's model file does.

## Your part

(a) `advancePositionsKernel` and `launchAdvancePositions` (`advance_positions` and
`encodeAdvancePositions` on Metal): one thread per sequence, run at the end of every step.

```
out[pos[b] * batch + b] = tokens[b]                       record this step's argmax
if (pos[b] + 1 < promptLen) tokens[b] = prompt[(pos[b] + 1) * batch + b]   still in the prompt
pos[b] += 1;  len[b] += 1;  rngStep[b] += 1
```

The prompt is fed one token per step through the same step, as llama2.c does (D1): while a
sequence is inside its prompt, the prompt's next token replaces the argmax. This kernel is
what lets the whole generation loop stay on the device. Before you write it, say what the
host would have to do between two steps without it.

(b) `launchLlamaStep(m, s)` (`encodeLlamaStep` on Metal): one step for `s.batch` sequences,
one launch per operation for the whole batch, nothing that waits for the GPU or copies to the
host. The order is the reference's, and the app's step model (`sim/llamastep.ts`) shows it:

| Operation | Piece | Shapes |
|---|---|---|
| `x = E[tokens]` | lab 22.2 (M8) `launchEmbed` | `[batch x d]` |
| for each layer: `xb = rmsnorm(x) * rmsAtt` | lab X3.1 `launchRmsNorm` | `[batch x d]` |
| `q, k, v = xb wq^T, xb wk^T, xb wv^T` | `m.linear`, three calls | `[batch x dq]`, `[batch x dkv]` twice |
| rotate `q` (h_q heads) and `k` (h_kv heads) at `pos` | lab X3.1 `launchRope`, pairing `m.ropeMode` | in place |
| append `k`, `v` to layer l's caches at `pos` | lab 20.4 (M7) `launchKvAppend` | `[batch x seqLen x h_kv x 128]` |
| `att = softmax(q K^T / sqrt(128)) V` over `len` rows | `m.attention`: lab 20.4's kernel, or lab 20.5's grouped one | `[batch x dq]` |
| `x += att wo^T` | `m.linear`, then lab 2.3 (M0) `launchAddResidual` | |
| `xb = rmsnorm(x) * rmsFfn` | lab X3.1 `launchRmsNorm` | |
| `x += (silu(xb w1^T) * (xb w3^T)) w2^T` | `m.linear` twice, lab X3.1 `launchSwiGLU`, `m.linear`, M0 | `[batch x hidden]` |
| `logits = (rmsnorm(x) * rmsFinal) E^T` | lab X3.1, then `m.linear` with E: the LM head is tied | `[batch x vocab]` |
| `tokens = argmax(logits)` | lab 10.3 (M4) `launchArgmaxRows`, or `s.sample` (M5) when the driver sets it | `[batch]` |
| record, feed the prompt, advance | yours, `launchAdvancePositions` | |

Here `dq = h_q * 128 = d` and `dkv = h_kv * 128`. `llama.h` declares every piece with its
argument order and gives every buffer's shape. Two questions to settle before you run it:
which `len` must the attention see so that it reads the row you just appended and nothing
after it, and where must the next tokens go so that the next step starts on the device?

## Ours, not the book's

- **The model.** llama2.c's `forward` (D1): RMSNorm with epsilon `1e-5`, grouped-query
  attention with RoPE base 10000, a SiLU-gated feed-forward, the classifier tied to the
  embedding. The head size is 128 because that is what M7's decode kernels are compiled for
  (`HEAD_DIM 128`); a model with another head size needs a decode kernel with the head size as
  a parameter, and the loader refuses it (E2).
- **The file.** `provided/make_model.py` writes llama2.c's version 0 `.bin` file (D1): a 28-byte
  header of seven `int32`, then float32 weights in llama2.c's order, the two RoPE tables
  llama2.c skips written as zeros, and a positive `vocab_size`, which means the classifier is
  the embedding. It uses the Python standard library only, with its own splitmix64 generator
  (not the `random` module) and `struct.pack('<f')`, so every machine writes the same bytes.
  CMake runs it at configure time into the build directory. Nothing is downloaded by CMake, the
  script, the driver or the tests.
- **The weights.** Uniform values times a scale per tensor (`--block-scale`, `--qk-scale`; the
  defaults are ours), chosen so that the generated tokens vary and the reference's greedy
  steps have large top-two margins (see the tests). They are not a trained model: the tokens
  have no meaning.
- **One token per step.** The prompt goes through the decode step token by token, as llama2.c
  processes it. M6 (lab 20.2's prefill flash attention) is not composed: it takes the whole
  prompt as one `[T x d]` block in its own layout, which is not the cache layout M7 reads
  (`[batch x seqLen x h_kv x 128]`), and the reference processes the prompt one token at a
  time. A prefill path would need M6 to write the cache, which is a change to M6.
- **No host round trip in the loop.** The argmax writes the next tokens over the ones the step
  read; your advance records them in `out` (`[seqLen x batch]`) and feeds the prompt. The
  driver copies `out` back once, after the last step. On CUDA and HIP the steps are launched
  back to back on the default stream, which runs them in issue order (ch02-facts A5, B3). On
  Metal each step is one command buffer, and every step's command buffer is committed to one
  queue before the driver waits: a queue "maintains an ordered list of command buffers"
  (infra-facts C12), and inside a step the plain compute encoder runs dispatches serially on
  hazard-tracked buffers (ch02-facts C9).
- **RoPE pairing (B6).** `m.ropeMode` 0 rotates consecutive pairs `(2i, 2i+1)` (llama2.c,
  llama.cpp's `LLAMA_ROPE_TYPE_NORM`); 1 rotates `(i, i + 64)` (Hugging Face's `rotate_half`).
  `make_model.py --permute-qk` writes the same weights with llama.cpp's row permutation of `wq`
  and `wk`, the one its converter applies to Hugging Face checkpoints: the permuted model in
  mode 0 computes what the original computes in mode 1.
- **M9.** The `LinearFn` indirection is where a quantised linear layer would plug in. Lab X2.4
  (M9) exists, but its `launchQLinear(X, const QMatrix& W, Y, batch)` takes a quantised matrix
  descriptor, not `LinearFn`'s `(X, const float* W, Y, batch, in, out)`, so
  `provided/pieces.cmake` adds no M9 row: M10 runs float32 without M9.

## Files

- `work/kernel.cu` (CUDA, HIP, emulators): `advancePositionsKernel`, `launchAdvancePositions`,
  `launchLlamaStep`.
- `work/kernel.metal`: `advance_positions`. `work/launch.mm`: `encodeAdvancePositions`,
  `encodeLlamaStep`.
- Provided, not edited: `llama.h` (`LlamaConfig` in llama2.c's header order, `LlamaModel` with
  device pointers per layer, `LlamaState`, `LinearFn`, `AttentionFn`, `SampleFn` and their
  Metal counterparts, every piece's declaration); `provided/pieces.cmake`, which copies the
  reference solutions of labs 2.3, 6.4, 10.3, 20.4, 22.2 and X3.1 (required) and 15.6, 20.5,
  22.1 and X1.3 (if present) into the build directory, renaming what would clash, as lab
  22.2's does (a missing required piece, or one whose signature differs from `llama.h`'s,
  skips the lab with a note that names it); `provided/make_model.py`; `provided/step_timer.cu`,
  the event pair that times a step on CUDA and HIP (lab 22.2's).
- `main.cpp`: the loader (with the E1 size check), the CPU reference (`RefSeq`: llama2.c's
  forward in double, greedy with the lowest index on a tie, D9), the tests and the benchmark.
  The driver compiles each piece's Metal kernel from that lab's own `kernel.metal`, with the
  constants that lab's driver uses.

## Commands

- Test: `python3 labs/pmpp.py run x3.2 --backend emu` first, then `python3 labs/pmpp.py run
  x3.2` on your GPU. The emulator runs every thread of every launch of every step under three
  thread orders: expect minutes, not seconds.
- Measure: `python3 labs/pmpp.py run x3.2 --bench`, after the predictions below.
- Another seed: `PMPP_X3_SEED=5 python3 labs/pmpp.py run x3.2` rewrites the test models at
  configure time with seed 5 (and the next run without it writes the provided ones again).

## What the tests do

Shapes, printed by the driver: on a GPU d = 512 (h_q 4 over h_kv 2), hidden 1376, 4 layers,
vocab 4099, seqLen 128, a 4-token prompt and 60 generated tokens (63 steps), batches 1 and 3;
on the emulators d = 256 (h_q 2 over h_kv 1), hidden 352, 2 layers, vocab 300, seqLen 16, a
3-token prompt and 5 generated tokens (7 steps), batches 1 and 2. The emulator's sizes all
differ from one another (d, dkv = 128, hidden, vocab), so a size passed in the wrong argument
fails there too; d cannot shrink without losing grouped-query attention at head size 128, so
the emulator run takes minutes. Every
prompt starts with token 1, llama2.c's BOS (D1). The linear layer is lab 22.1's gather on a
GPU and lab 15.6's M3 on the emulators, whose 128-wide tiles launch the fewest threads and so
cost the emulator least (lab 22.2 makes the same choice); M2 if neither is present.

- **Model files.** The file's size equals the E1 formula for its header; the checksum of its
  first 256 and last 256 weights (FNV-1a 64 over the embedding's first floats and
  `rms_final_weight`'s last) equals what the script printed when CMake ran it and the value
  recorded in `provided/pieces.cmake` for the provided seed (a generator that wrote different
  bytes on another machine fails here); the `--permute-qk` file holds exactly llama.cpp's
  permutation of `wq` and `wk` and nothing else that differs.
- **Pieces alone**, before your code runs: each provided piece once against a CPU reference
  (the embedding bit for bit, RMSNorm, every linear layer present, RoPE in both pairings at
  positions 3 and seqLen - 1, the KV append, each attention kernel over a cache whose unread
  rows are NaN, the residual add bit for bit, SwiGLU, the argmax with a planted tie). A failure
  here is the course's, not yours. Then your advance alone, at positions 0, 2 and 5 of a
  4-token prompt.
- **Teacher-forced logits.** Every step is fed the reference's token (the whole reference
  trace is the prompt), so the logits of every step can be compared. Per step the driver
  prints the NMSE, `sum (got - ref)^2 / sum ref^2` (llama.cpp's metric, D7), the largest
  absolute error, and the largest error over its tolerance. A logit passes if
  `|got - ref| <= 4e-5 * (1 + mag)`, where `mag = sum_j |xn_j E_vj|` is the size of the terms
  of the LM head's dot product (`xn` the reference's final normalised hidden state). The same
  tolerance holds on every backend; it was chosen so that Metal's default math mode passes
  with room: on the machine this lab was written on, the worst error was 0.06 of it with lab
  6.4's or lab 15.6's GEMM and 0.02 of it with the gather (Metal), and about 0.013 of it on the
  emulator. These are measurements of one machine; CUDA and HIP have not been run. The step must also record its
  own argmax in `out`, feed the next prompt token, and leave `pos` and `len` at step + 1 and
  step + 2.
- **Free-running greedy.** The real prompt, then all the steps with no host work in between,
  and one copy of `out` at the end. The tokens must equal the reference's exactly at every
  step whose reference top-two margin exceeds twice the logit tolerance (the two logits'
  tolerances added). With the provided seeds every tested step qualifies, by a factor the
  driver prints. With another seed, a low-margin step inside the prompt is skipped alone (its
  successor is forced), and the first one in the generation makes the rest "inconclusive from
  step k", printed, never silently passed: a flip there changes every later token.
- **Batch invariance.** Each sequence's tokens in batch 2 (3 on a GPU) equal its tokens when it
  is decoded alone.
- **Other kernels** (GPU only): every other linear layer and attention kernel present,
  teacher-forced at batch 1. The emulator would take minutes per kernel, and the pieces-alone
  checks ran each one.
- **RoPE pairing.** The CPU reference of the permuted model in mode 0 equals the original's in
  mode 1 to 1e-9; then the original model in mode 1 and the permuted one in mode 0, each
  teacher-forced (which also checks every step's token) and, on a GPU, free-running, against
  the mode-1 reference.
- **The sampler (M5)**, GPU only, if lab X1.3's solution is present: `s.sample` at temperature 0
  gives the greedy tokens exactly; at temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, with
  a fixed seed and one stream per sequence, two runs give identical tokens. Sampled tokens are
  not compared across backends here (lab X1.3 is where that is tested).
- **Poison and guards.** The activations, the logits and every cache row start as NaN, so a
  value nobody wrote, or a cache row the attention should not read, turns the logits to NaN.
  Every output is followed by a guard band that must be intact; after each run the cache rows
  below the step count are written and the rest still NaN; the weights, the prompt and the
  cos/sin tables are unchanged. On `emu` the weights and inputs are allocated to their exact
  size, so a read past one is an AddressSanitizer error that names the line.
- Not on the `race` backend: your code launches pieces and writes no shared memory, and the
  pieces were race-tested in their own labs.

## Before you benchmark, predict

Write down two predictions (the lesson asks for them as `x3/labs/p-lab2-bwfraction` and
`x3/labs/p-lab2-attn-share`):

1. Tokens per second at batch 1, position 16, as a fraction of the ceiling: lab 5.1's
   bandwidth divided by the weight bytes per token (the benchmark prints both).
2. The attention share of the step time at position 512, given the share you measure at 16.

Things to weigh:

- **The GEMV choice.** At batch 1 every projection is a matrix-vector product. Lab 22.1's
  gather reads each weight once per step; lab 6.4's tiles were built for matrices and launch a
  16-row tile for one row of work. llama.cpp switches from matrix-vector to matrix-matrix
  kernels at about 8 tokens (ch22-facts D6). The benchmark runs both.
- **Launches per token.** A step is sixteen launches per layer plus five, each short at
  batch 1. Each costs a launch on CUDA and HIP and a dispatch on Metal, and none of it moves
  weights. The per-piece split adds up the pieces timed alone; compare its sum with the step.
- **The KV cache grows; the weights do not.** The weights read per token are fixed. The
  attention reads `2 x layers x batch x len x dkv x 4` bytes, which grows with the position:
  at position 512 (len 513) it is about 30 times what it is at 16 (len 17). Whether that matters depends on how it compares with
  the weight bytes, which the benchmark prints side by side.
- **Caches.** A small model's weights may fit in part in the device's caches, which would put
  the weight bandwidth above lab 5.1's; Apple documents no cache sizes (infra-facts C13).

## What the benchmark measures

Real devices only. The model is written by the same script at bench time (`--fill tile`: every
tensor repeats one block of random values, so a large file is written in seconds; its numbers
mean nothing): d = 2048 with 16 query heads over 4 KV heads, hidden 5632, 8 layers, vocab
32000, seqLen 1024 (our shape). If the weights would take more than a quarter of the memory
the device reports, d, the heads and the hidden width halve until they fit; the shape and the
weight bytes are printed. One step is one command buffer on Metal (`GPUEndTime -
GPUStartTime`) and one pair of events on CUDA and HIP (`provided/step_timer.cu`). Each figure
is the best of several timed runs after an untimed one.

- `step` rows, for each linear layer present, at batch 1 and 4 and positions 16 and 512 (the
  cache rows before the position hold finite values, as after a prompt; their values do not
  change the time, and each timed step is put back at its position): ms per step, tokens per
  second (`batch / seconds`), weight bytes per token (`weight bytes / batch`), weight GB/s, and
  the KV cache bytes the attention reads.
- `piece` rows, with the gather (or M2 without lab 22.1): every piece timed alone at the same
  batches and positions, times how often a step runs it, summed into four groups: matmuls,
  attention (append and attention), norms + RoPE + SwiGLU + residual, embed + LM head + argmax
  + advance; with the attention's share of their sum.
- A `ceiling` row: lab 5.1's bandwidth, read from `reports/runs/<backend>.json` when you have
  run lab 5.1's benchmark, divided by the weight bytes per token, and the measured tokens per
  second at batch 1, position 16 as a fraction of it.

The rows are measurements of your machine, not expectations.

## A real checkpoint you download yourself

The pipeline reads llama2.c's format, so the first real checkpoint to try is one of Karpathy's
tinyllamas (MIT, D2): `stories260K.bin` (1,056,540 bytes) or `stories15M.bin`. The loader's E1
check reproduces both sizes from their headers. They will not run as they are: their head
sizes are 8 and 48, and M7's decode kernels are compiled for 128 (E2), so the attention must
first take the head size as a parameter (and the `LLAMA_HEAD_DIM` constant in `llama.h`
with it). With that change, `run.c` with `-t 0.0` is the reference to compare tokens with.

A Hugging Face Llama checkpoint such as SmolLM2-135M (Apache-2.0) needs four conversions on top
(D4): bfloat16 to float32 (exact: the 16 stored bits become the top half of a float), the
half-split RoPE pairing (mode 1, or permute `wq` and `wk` as `--permute-qk` does), 9 query heads
over 3 KV heads with head size 64, and RoPE base 100000 and epsilon 1e-5 from its
`config.json`. Do not use llama2.c's `export.py` for it: by our reading its permutation of
`k_proj` assumes as many KV heads as query heads (D6). GGUF is the format llama.cpp loads, but
it adds typed metadata, alignment padding and usually quantised tensors (D5).

None of this was run while the lab was written: it is a description of the work, unverified.

## Portability

- HIP: your `kernel.cu` uses threads, blocks and runtime launches only, which build through
  `pmpp/vendors/hip.h` unchanged. The pieces with warp shuffles (lab 10.3, lab 20.4 and 20.5,
  lab 22.1, lab X3.1) use a typed lane mask and either `warpSize` or a logical 32-lane warp.
  Nothing here has been compiled or run on AMD hardware: unverified.
- CUDA: unverified until a `reports/verify-*.md` from a CUDA machine shows this lab passing.
- Metal: the decode kernels of labs 20.4 and 20.5 need 32-lane SIMD-groups; the driver asks the
  pipeline (`threadExecutionWidth`) and fails the pieces-alone check, with the width it found,
  if it reports another. Apple documents no fixed width (ch20-facts C3).
