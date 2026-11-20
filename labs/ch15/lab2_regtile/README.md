# Lab 15.2: register tiling, the loops interchanged

Book: section 15.4 (pp. 357-360), Figs. 15.8 and 15.9. Milestone M3, stage 2 of 5.

`work/` holds lab 15.1's solution, renamed (`gemmRegTileKernel`, `launchGemmRegTile`,
`gemm_regtile`, `encodeGemmRegTile`), with the body of `mm` removed. Rewrite `mm` the way
Fig. 15.9 does: make the inner-dimension loop `i` the outermost one; for each `i`, copy
the thread's 8 values of column `i` of its rows of `A_s` into `a_r[8]` and its 8 values of
row `i` of its columns of `B_s` into `b_r[8]`; then add the outer product
`a_r[row] * b_r[col]` to all 64 elements of the tile. Unroll all five loops completely, or
`a_r`, `b_r` and the tile leave the registers.

## Files

- Edit `mm` in `work/kernel.cu` and in `work/kernel.metal`. `work/launch.mm` is complete:
  register tiling changes the kernel, not the launch.

## Commands

    python3 labs/pmpp.py run 15.2 --backend emu
    python3 labs/pmpp.py run 15.2
    python3 labs/pmpp.py run 15.2 --bench

The tests are lab 15.1's (see its README). The untouched starter builds and fails: its
`mm` adds nothing, so every element of C comes out 0.

## Before you benchmark

Count first. Per thread and per tile pair, Fig. 15.6 reads shared memory
8 x 8 x 8 x 2 = 1024 times; Fig. 15.9 reads it 8 x (8 + 8) = 128 times, 8 times fewer.
Then write down **the speed-up you expect over lab 15.1 at 2048 x 2048 x 2048**, and weigh:

- On a device with 32 four-byte banks the strip reads still conflict in this thread
  layout: the A strip 2-way, the B strip 4-way (16 lanes read words 8 apart). That is our
  arithmetic with NVIDIA's documented bank rule (`notes/today/ch15-facts.md` E1).
- The stores are unchanged: each thread still writes its 8 x 8 patch with scalar stores
  that land 8 elements away from its neighbour's.
- A compiler that fully unrolls Fig. 15.6 sees every index as a constant and may already
  keep repeated reads in registers on its own (our reasoning, not a vendor statement).
  The interchange makes the reuse explicit; whether it was missing is what the
  measurement tells you.
- On Metal, Apple documents neither banks nor registers per thread (C2, C3): nothing
  documented predicts the result.

`--bench` adds lab 15.1's solution as the column `15.1` (copied into the build when you
configure), next to lab 5.3's solution, yours and the vendor's.
