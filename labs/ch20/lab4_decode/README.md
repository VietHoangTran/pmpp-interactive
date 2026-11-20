# Lab 20.4: KV cache append and single-token decode (milestone M7)

Book: section 20.4 (pp. 488-492, Figs. 20.5 to 20.7) for what a decode step computes;
section 20.5 (pp. 493-504, Eqs. 20.3 to 20.6, Figs. 20.10 to 20.16) for the online softmax
you build it from; section 20.6 (pp. 506-507, Eq. 20.9) for why it is bandwidth-bound.

With a KV cache, one decode step of one attention layer takes one new token per sequence:
its q, k and v (Fig. 20.6). The new k and v are appended to the cache, then q attends
over everything cached. Write the two host functions that do this for a batch:

```
launchKvAppend(k_new, v_new, Kc, Vc, pos, batch, max_seq, h_kv)
launchDecodeAttention(q, Kc, Vc, len, out, batch, h_q, h_kv, max_seq, scale)
```

| Tensor | Shape |
|---|---|
| `k_new`, `v_new` | `[batch, h_kv, 128]` |
| `Kc`, `Vc` (the cache) | `[batch, max_seq, h_kv, 128]` |
| `q`, `out` | `[batch, h_q, 128]` |
| `pos`, `len` | `[batch]`, int32 |

Row-major `float`, batch first, head dimension 128 (`HEAD_DIM`, the book's d). The cache
layout is Meta's reference Llama code's, preallocated to `max_seq` rows per sequence
(`notes/today/ch20-facts.md` D3).

- **Append.** Write row `pos[b]` of sequence b in both caches and nothing else: a scatter
  by index, as llama.cpp's `ggml_set_rows` writes its cache (D3).
- **Decode.** Query head h of sequence b attends over the first `len[b]` rows of K/V head
  `h / (h_q / h_kv)`, so one kernel covers MHA (`h_kv = h_q`), GQA and MQA (`h_kv = 1`).
  `out = softmax(scale * q K^T) V`, `scale = 1/sqrt(128)`. **No mask**: a single new query
  may see every cached token. Fig. 20.6 has none, and Meta's code builds a mask only when
  more than one token is processed (D3).

Contract: `0 <= pos[b] < max_seq`, `1 <= len[b] <= max_seq`, `h_q` a multiple of `h_kv`,
every value finite. Sequences have different lengths in one call.

## The design (ours)

The book shows no code for the cache, the append or a decode kernel (p. 492). This design
is ours. MLX's Metal decode kernel, `sdpa_vector`, has the same shape (D4).

- **One block of 256 threads per (head, sequence)**: a grid of `(h_q, batch)`.
- **Warps split the keys.** Logical warp w (8 per block) takes keys w, w + 8, w + 16, ...
- **Lanes split the head dimension**, Fig. 20.10's way: lane L holds elements L, L + 32,
  L + 64, L + 96 of q, so each read of a key row by the 32 lanes is one contiguous run.
- **The score** of key t is a butterfly sum over the warp of the four products per lane
  (`__shfl_xor_sync`), which leaves the sum in every lane.
- **Each warp keeps its own running (m, D, O~)**: max, denominator, and the output not
  yet divided by D. It updates them one key at a time, as Figs. 20.13 to 20.15 do one tile
  at a time: new max, rescale D and O~ by `e^(old max - new max)`, add `e^(s - new max)` to
  D and that times the V row to O~.
- **At the end the 8 warps merge through shared memory** with Eqs. 20.4 and 20.6 (the
  un-normalised form): weights `e^(m_w - m_all)`, one division (Fig. 20.16). This is the
  "split-K" scheme that FlashAttention-1 used across the warps of a block (footnote 7,
  p. 498; ch20-facts A7) and that Flash-Decoding uses across thread blocks (A8).

**Metal**: the same structure. `simd_sum` gives each score to every lane, the merge goes
through threadgroup memory, and the running max starts at a finite value: under Metal's
default fast math, infinities are undefined (C5).

## Where this differs from the page

- The running max starts at `M_INIT = -FLT_MAX/2`, not the book's `NEG_INFINITY`. That
  name is defined by no CUDA header, and IEEE gives NaN for `(-inf) - (-inf)` (A5). Here a
  warp that saw no key (any `len[b] < 8`) still takes part in the merge, so its m must be
  finite. llama.cpp starts at `-FLT_MAX/2`, MLX at `finite_min`, for the same reason (A5,
  C5, D1, D4).
- The book gets each S element from `cub::WarpReduce` plus a broadcast from lane 0
  (Fig. 20.12). The butterfly gives every lane the sum in one pass (ch10-facts D1), and the
  `cub::Max()` and `cub::Sum()` the page uses no longer exist (A2).
- The book's `WARP_SIZE` is the hardware warp. Here it is a logical warp of 32: every
  shuffle passes width `WARP_SIZE`, so on a 64-wide AMD wavefront two logical warps share
  a wavefront without mixing (B2). The mask is `lane_mask_t`, 64-bit on HIP (B1).
- Page 502 says Fig. 20.14's `D *= exp(last_m - m_i[ii])` computes the `D_B` term of
  Eq. 20.4. It rescales the old denominator: it is the `D_A` term (notes, "Unclear in the
  PDF").

## Files

- `work/kernel.cu` (CUDA, HIP, emulators): `warpSumAll`, `kvAppendKernel`,
  `launchKvAppend`, `decodeAttentionKernel`, `launchDecodeAttention`. The TODOs are in
  order.
- `work/kernel.metal`: `kv_append`, `decode_attention`. `work/launch.mm`: `encodeKvAppend`
  and `encodeDecodeAttention` bind their arguments already; you write the dispatches.
- `main.cpp`: the driver and the CPU references (`cpuDecodeAttention`, and
  `cpuCausalAttention`, Eq. 20.1 with its mask). The tests are in
  `../shared-g12/decode_test.hpp`, which lab 20.5 shares.

## Commands

- Test: `python3 labs/pmpp.py run 20.4 --backend emu` first. Then `--backend race`: the
  warps merge through shared memory, and ThreadSanitizer names the two lines of a missing
  barrier. Then `python3 labs/pmpp.py run 20.4` on your GPU.
- Measure: `python3 labs/pmpp.py run 20.4 --bench`.

## What the tests do

- **Append**: four shapes, two calls each into caches full of canary values. After each
  call, every element of both caches is compared bit for bit: the rows at `pos[b]` hold the
  new rows, everything else its canary. The message names the sequence, row, head and
  element.
- **Decode**: `(h_q, h_kv)` = (8, 8) MHA, (8, 2) GQA, (8, 1) MQA and (6, 2), whose g_q of 3
  is not a power of two. One to four sequences per call, each with its own length: 1, 2,
  31, 33, 100, 1000 and 4097 on a GPU; up to 33, plus one case of 200, on `emu`; up to 40
  on `race`.
- Two kinds of data. Random keys. And **late max**: the last key of every sequence scores
  about 6 for every query that reads it, while the others score near 0. The running max
  jumps at the very end, in one warp only, so the other warps' states must be scaled by
  about e^-6 in the merge. The emulators run late max only.
- Cache rows at and past `len[b]` hold NaN: reading one poisons the output. `out` starts
  as NaN and is followed by a band of guard values that must survive.
- The result is compared with a CPU reference in double. The tolerance is
  `3e-5 * (1 + |reference|)`; `main.cpp` says why that is tight enough to catch one key of
  4097 left out.
- **The milestone loop**: 16 autoregressive steps on (8, 2) for sequences with different
  prompt lengths (one starts empty). Each step appends and attends in one submission, at
  `pos[b] = prompt[b] + step`, `len[b] = pos[b] + 1`, and its q, k and v are a fixed
  function of the previous step's output. At every step each output must equal **the last
  row of a full causal attention recomputed from scratch** over all tokens so far (Eq. 20.1,
  mask and all). That is section 20.4's claim, tested: with the cache, the step needs only
  the new row. At the end both caches must hold exactly the history, and the rows past it
  must still be NaN.

## Before you benchmark, predict

The benchmark times the decode kernel alone at `h_q = h_kv = 32`, head dimension 128
(a 7B-class MHA layer, our choice), for `len` = 512, 2048, 8192 and 32768 and batch 1 and
8. It reports milliseconds per call and GB/s counted on the cache bytes one call must
read, `2 x len x h_kv x 128 x 4` per sequence. Sizes whose caches do not fit in half of
the memory your device reports are skipped and listed. It also times the append alone.

**Prediction `ch20/labs/p-lab4-ms`.** Write down, before you run it:

1. Using the bandwidth you measured in lab 5.1, the time of one decode call at
   `len` = 8192, batch 1, 32 heads of 128 in fp32.
2. Whether batch 8 at the same length takes 8 times as long.

The intensity argument for (1): every 4-byte element of K is used in one multiply-add
with q (2 FLOPs), and every element of V in one multiply-add with a probability. That is
about **0.5 FLOP/B** in fp32. (Eq. 20.9 gets "about 1" by counting one operation per
element at 1 byte per element; reference [13] gets 1 FLOP/B for 2-byte BF16, ch20-facts E4.
Same argument, different units.) That is far below the threshold your lab 5.1 measured, so
the time should be close to bytes / bandwidth, and at `len` 8192 the cache is 256 MiB.

For (2), weigh: each sequence has its own cache, so batch 8 reads 8 times the bytes
(Fig. 20.17: attention does not benefit from batching); but batch 1 launches only 32
blocks and batch 8 launches 256, and a kernel that does not keep enough requests in flight
cannot reach the bandwidth it was promised (Flash-Decoding's authors make this point about
small batches, A8).

The printed numbers are measurements of your machine, not expectations. The lesson page
sets the GB/s beside your lab 5.1 bandwidth.

## Beyond the book (ours)

- **Split-KV.** With few sequences and heads, Flash-Decoding (FlashAttention 2.2) also
  splits the keys across thread blocks and merges the partial results in a second kernel
  with the same Eq. 20.4 / 20.6 merge; `flash_attn_with_kvcache` exposes it as `num_splits`
  (A8), and llama.cpp's `flash_attn_combine_results` is that merge (D1).
- **Checking against PyTorch.** `scaled_dot_product_attention(..., is_causal=True)` aligns
  the mask to the upper left when the query is shorter than the keys, so a one-token query
  would see only key 0. Pass no mask for a single query, as this lab does, or use the
  lower-right variant (D5).
