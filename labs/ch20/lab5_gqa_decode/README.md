# Lab 20.5: grouped-query decode, loading each K/V row once

**Finish lab 20.4 first.** This lab's starter (`kernel.cu` and `kernel.metal`) begins
from lab 20.4's reference per-head kernel, so opening it gives away that lab's answer.

Book: section 20.7 (pp. 508-509, Eqs. 20.12 and 20.13) for MQA and GQA and their
arithmetic intensity; section 20.5 (Eqs. 20.4 and 20.6) for the merge you already wrote in
lab 20.4.

In a GQA model, `g_q = h_q / h_kv` query heads share each K/V head. Lab 20.4's kernel gives
every query head its own block, so the g_q blocks of one group each read the same K and V
rows: the kernel asks memory for g_q times the bytes the cache holds. Write
`launchDecodeAttentionGrouped`, with exactly the arguments and contract of lab 20.4's
`launchDecodeAttention`, so that each K and V row is read once and used for all g_q
queries. That is what llama.cpp's `gqa_ratio` path does: several Q heads against one
loaded K/V head (`notes/today/ch20-facts.md` D1).

`g_q` here is the number of query heads **per K/V head**, the meaning Eqs. 20.12 and 20.13
need (reference [13]'s convention). The GQA paper's G, "the number of groups", is
`h_q / g_q`, the number of K/V heads; the book's prose mixes the two (E3, and the notes'
"Unclear in the PDF").

Contract: lab 20.4's (`q`, `out` `[batch, h_q, 128]`; `Kc`, `Vc`
`[batch, max_seq, h_kv, 128]`; `len[b]` from 1 to `max_seq`; no mask), plus `g_q <= 8`.
For a larger group, print a message and launch nothing.

## The design (ours)

The book prints no GQA kernel. This regrouping of lab 20.4's design is ours.

- **One block of 256 threads per (K/V head, sequence)**: a grid of `(h_kv, batch)`. Block
  `kvh` serves query heads `kvh * g_q` to `kvh * g_q + g_q - 1`.
- **Warps still split the keys** (warp w takes keys w, w + 8, ...) and **lanes still split
  the head dimension** (4 values per lane). For each key a warp loads its K row and its V
  row into registers once, then computes g_q scores (g_q butterflies) and g_q online
  updates.
- **Each warp keeps g_q running states** (m, D, O~), one per query head.
- **The merge runs once per query head**, through the same shared buffer: publish the
  warps' O~ for head g, barrier, combine with Eqs. 20.4 and 20.6, write head g's output,
  barrier again before head g + 1 overwrites the buffer. Holding all eight heads' O~ at
  once would take 32 KiB, which is a Metal threadgroup's whole limit on the GPU families
  Apple's tables list (C1).
- **The group size is a compile-time constant, `GQ`.** In `kernel.cu` the kernel is a
  template, `decodeAttentionGroupedKernel<GQ>`, and the launcher picks the instance with a
  `switch` on `h_q / h_kv`. On Metal the driver compiles `kernel.metal` once per group size
  with `GQ` defined as a preprocessor macro (one of the two documented routes to
  compile-time sizes, C6). Arrays sized `GQ` and loops with a known trip count are what let
  the compiler keep the states in registers. Sizing every array for 8 heads and switching
  the unused ones off at run time compiles and gives right answers, but the reference's
  first version did that and measured slower than the per-head kernel even at `g_q = 1`.

**Metal**: the same structure. `simd_sum` gives each score to every lane, so call it once
per head of the group, from every lane. The running max starts at a finite value (C5), and
the driver checks for 32-lane SIMD-groups (C3).

## Where this differs from the page

Everything in lab 20.4's README: a finite `M_INIT` instead of `NEG_INFINITY`, a butterfly
instead of `cub::WarpReduce`, a logical 32-lane warp with width 32 on every shuffle, a
typed lane mask. And Eq. 20.13's "≈ g_q" counts 1-byte elements; in fp32 each K and V
element still costs 4 bytes, so your kernel's intensity is about `g_q / 2` FLOP/B
(lab 20.4's 0.5, times g_q).

## Files

- `work/kernel.cu`: `decodeAttentionGroupedKernel<GQ>` and `launchDecodeAttentionGrouped`.
  It starts as lab 20.4's reference per-head kernel, already a template on `GQ`; the TODOs
  mark what to regroup: shared arrays (1), block-to-head mapping (2), the per-warp state
  (3), the key loop (4), the merge (5), the launch (6).
- `work/kernel.metal`: `decode_attention_grouped`, the same starting point. `GQ` is
  defined by the driver. `work/launch.mm`: `encodeDecodeAttentionGrouped` binds its
  arguments already; you write the size check and the dispatch.
- Provided: lab 20.4's reference solution is copied into the build directory with its
  names changed (`provided/perhead.cmake`). Its per-head kernel is the benchmark's
  baseline, and its KV append feeds the 16-step loop. The first test runs it alone: if
  that test fails, the bug is the course's, not yours.
- `main.cpp`: the driver and the CPU references. The tests are lab 20.4's, in
  `../shared-g12/decode_test.hpp`.

## Commands

- Test: `python3 labs/pmpp.py run 20.5 --backend emu` first, then `--backend race` (the
  warps merge through shared memory once per head; ThreadSanitizer names the two lines of
  a missing barrier), then `python3 labs/pmpp.py run 20.5` on your GPU.
- Measure: `python3 labs/pmpp.py run 20.5 --bench`.

## What the tests do

Lab 20.4's decode tests against your kernel: `(h_q, h_kv)` = (8, 8), (8, 4), (8, 2),
(8, 1) and (6, 2), so g_q = 1, 2, 4, 8 and 3; one to four sequences per call with their
own lengths (1, 2, 31, 33, 100, 1000 and 4097 on a GPU; up to 33, plus one case of 200, on
`emu`; up to 40 on `race`); random and late-max keys; NaN in every cache row a kernel must
not read; a guard band after `out`; a CPU reference in double with the tolerance of lab
20.4. Then the 16-step loop of lab 20.4, with lab 20.4's append and your kernel in one
submission.

The tests check the answers, not the grouping. A kernel that still computes one query head
per block passes them if it is launched over `(h_q, batch)`. Only the benchmark shows
whether each row is now read once.

## Before you benchmark, predict

The benchmark runs at `h_q = 32`, head dimension 128, `len` = 8192, batch 1 and 4, and
`h_kv` = 32, 8 and 4 (g_q = 1, 4 and 8). It times lab 20.4's per-head kernel and yours in
alternating rounds and prints milliseconds and GB/s for both. GB/s counts the **unique**
cache bytes, `2 x len x h_kv x 128 x 4` per sequence, for both kernels, so a higher figure
means the same cache was consumed faster. The `blocks` column is what each design launches
by its contract: `batch x h_q` for the per-head kernel, `batch x h_kv` for the grouped one.

**Prediction `ch20/labs/p-lab5-speedup`.** Write down, before you run it: at `h_kv = 8`
(g_q = 4), batch 1, how many times faster your grouped kernel will be than the per-head
kernel.

Forces to weigh (this README gives no verdict):

- **Bytes requested.** The per-head kernel asks for g_q times the unique bytes, and your
  kernel asks for them once. Eq. 20.13 says intensity rises by about g_q.
- **Caches.** The g_q per-head blocks of one group read the same rows. If they run at about
  the same time, a cache between the cores and DRAM may serve the repeats, and DRAM sees
  far fewer than g_q times the bytes. CUDA says blocks "may be executed in any order"
  (`notes/today/ch04-facts.md` A5), and Apple's specification says nothing about the order
  of threadgroups (`ch12-facts.md` C4), so whether they overlap is something to measure.
- **Blocks in flight.** At batch 1 and `h_kv = 8` your kernel launches 8 blocks where the
  per-head kernel launches 32. Reaching full bandwidth takes many requests in flight; with
  one query per head and a small batch, few blocks may not issue enough of them. That is
  the reason Flash-Decoding splits the keys of one head across several blocks (A8).

The printed numbers are measurements of your machine, not expectations. The lesson page sets
the measured ratios beside Eq. 20.13's intensity of about g_q.

## Beyond the book (ours)

- **Split the keys across blocks as well.** Give each (K/V head, sequence) several blocks,
  each over a slice of the keys, and merge their (m, D, O~) in a second kernel with the same
  Eqs. 20.4 and 20.6. That restores the block count lost to grouping. Flash-Decoding does
  this (`num_splits` in `flash_attn_with_kvcache`, A8), llama.cpp merges the parts with
  `flash_attn_combine_results` (D1), and MLX switches to its two-pass kernel for GQA from
  4096 keys on (and from 1024 on its larger devices, D4).
- **The libraries take K/V with fewer heads than Q.** FlashAttention maps query heads to
  K/V heads exactly as this lab does (A8), and MLX asks that K and V not be "pre-tiled to
  match q" (D4). Repeating K/V heads to make an MHA-shaped tensor, as Meta's reference
  `repeat_kv` does (D3), costs the extra bytes this lab removes.
