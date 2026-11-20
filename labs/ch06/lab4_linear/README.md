# Lab 6.4: the linear layer in the framework layout, corner-turned (milestone M2)

Book: corner turning, section 6.1 and Fig. 6.4 (pp. 129-130); padding the turned tile,
section 6.4 (pp. 139-141); Exercise 2 (p. 155).

Lab 5.3 computed C = A B with B stored (k, l). Frameworks do not store a linear layer
that way: PyTorch's `nn.Linear` keeps its weight as (out_features, in_features) and
computes y = x W^T, and llama.cpp's `ggml_mul_mat` uses the same layout. Read the lab 5.3
way, that W^T is the book's column-major N (p. 128): neighbouring threads fetch floats
`in` apart.

Write `linearKernel(X, W, Y, batch, in, out)` and `launchLinear`:
`Y[batch x out] = X[batch x in] * W^T`, with W stored (out, in) row-major. Start from
your lab 5.3 kernel (the starter is that kernel reshaped to these arguments: any sizes,
every load and the store bounds-checked, zero fill). Two things change, both in the W
tile:

1. **Corner turning.** Load the W tile so that neighbouring `threadIdx.x` read
   neighbouring floats of one row of W, then put each value where the product loop
   expects it. This is the book's Exercise 2 in the layout the project needs.
2. **Padding.** The turned store writes down a column of the shared tile. Declare the
   tile `[TILE_WIDTH][TILE_WIDTH + 1]` so that column does not fall in the same banks.

**Layout decision, from here on.** The project's weights are stored (out, in) row-major,
as `nn.Linear` and ggml store them; tensors are row-major `float` with the batch first;
kernels take raw device pointers and explicit sizes. `launchLinear(X, W, Y, batch, in,
out)` is the signature chapter 15 builds on.

- Edit `work/kernel.cu` and `work/kernel.metal` (`work/launch.mm` is complete).
- `TILE_WIDTH` is 16 by default; a 32 build is also made for the benchmark, and Metal
  compiles 16 and 32 at run time. Do not hard-code 16.
- Test: `python3 labs/pmpp.py run 6.4`   Start with `--backend emu`. The tests check
  results only: the provided baseline's uncoalesced load passes them too. Whether your
  load coalesces shows in the benchmark (on NVIDIA, Nsight Compute counts it as sectors
  per request).
- Measure: `python3 labs/pmpp.py run 6.4 --bench` times the provided baseline
  (`provided/linear_baseline.*`: the same tiled kernel reading `W[Col * in + k]`, correct
  but uncoalesced) and your kernel on lab 5.3's 1024 -> 4096 layer at batch 1 to 1024.
  The two take turns, and the best of several runs of each is kept.

Before you benchmark, predict the speed-up of your kernel over the baseline at batch 256,
T = 16. Things to weigh:

- NVIDIA counts memory traffic in 32-byte sectors. At T = 32, one warp's W load touches
  32 sectors in the baseline and 4 when corner-turned: 8 times fewer. At T = 16 a
  32-thread warp covers two rows of the block: work out the figure there.
- The W load is one part of each phase. The X tile load, the barriers and the product
  loop are the same in both kernels.
- Compare what tiling bought you in lab 5.2 with the T times less traffic its model
  promised. Caches can serve some of the traffic a model charges for. Here, in the
  baseline, which other warps of the same block need the rest of each sector a warp
  fetches?
- On Metal nothing predicts the answer: Apple documents no coalescing rule and no
  threadgroup-memory banks, and leaves the split of a threadgroup into SIMD-groups to the
  implementation. Your measurement is the answer.

Afterwards, remove the padding column and run `--bench` again. What did the padding buy
on your device, at T = 16 and at T = 32? Section 6.4's bank arithmetic predicts a
difference between the two on NVIDIA; work out which one before you look.
