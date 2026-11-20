# Lab 5.3: a general GEMM (project milestone M2)

Book: section 5.5 (pp. 112-115), Fig. 5.13, and the rectangular case the book leaves as
an exercise.

Extend your tiled kernel to `C[j x l] = A[j x k] * B[k x l]` for any sizes. This is the
kernel behind every linear layer: j is the batch, k the inputs, l the outputs.

Three things change from lab 5.2: `Width` meant three different things; sizes are no
longer multiples of the tile, so every memory access needs its own bounds check and the
phase count rounds up; and the grid is not square.

- Edit `work/kernel.cu` and `work/kernel.metal` + `work/launch.mm`.
- Test: `python3 labs/pmpp.py run 5.3`   The emulator runs under AddressSanitizer, so a
  missing bounds check is reported with the source line.
- Measure: `python3 labs/pmpp.py run 5.3 --bench` times a 1024 -> 4096 layer at batch
  sizes 1 to 1024.

Before you benchmark, predict the ratio of GFLOP/s at batch 256 to batch 1.
