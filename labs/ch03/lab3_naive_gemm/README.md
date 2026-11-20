# Lab 3.3: naive GEMM for a linear layer (project milestone M1, first half)

Book: section 3.4 (pp. 59-64), Fig. 3.11 (p. 61), and the rectangular definition on p. 60
that the figure narrows to square matrices.

Write the kernel of Fig. 3.11 for `C[j x l] = A[j x k] * B[k x l]`: row-major `float`,
any sizes, one thread per element of C, 16 x 16 blocks. The figure has one `Width`. You
have three sizes, and every `Width` in the figure becomes exactly one of them.

As a linear layer, j is the batch (the first dimension), k the number of inputs and l the
number of outputs. B holds the weight as (in, out). PyTorch stores the transpose,
(out, in), and computes `y = x W^T`, so with `B = W^T` built on the host this kernel
computes the same `y`. One of the tests does exactly that. PyTorch and ggml (llama.cpp)
keep the weight as (out, in) and read both operands along rows. This lab keeps the book's
convention, where the second operand is read down a column, and lab 5.3 keeps it too:
same argument order, same storage, so the tiled kernel replaces this one later.

- Edit `work/kernel.cu`: `naiveGemmKernel` and `launchNaiveGemm`. The grid's x covers l,
  its y covers j.
- Edit `work/kernel.metal` (kernel `naive_gemm`) and `work/launch.mm` (`encodeNaiveGemm`).
  The kernel takes `[[thread_position_in_grid]]`, the launch is `dispatchThreadgroups`
  with whole 16 x 16 threadgroups, and the kernel keeps the guard. Rounding up to whole
  threadgroups and guarding is what PyTorch's MPS matmul and llama.cpp's Metal backend
  do. MLX calls `dispatchThreads` instead, which launches exactly the grid you ask for:
  no thread lies outside the data, so no guard is needed. Either passes the tests.
- Test: `python3 labs/pmpp.py run 3.3`   Start with `--backend emu`: it runs under
  AddressSanitizer, which stops at the first access outside a buffer and names the line.
  `--backend race` runs the emulated threads concurrently under ThreadSanitizer and
  reports two threads that store to the same element.
- Measure: `python3 labs/pmpp.py run 3.3 --bench` times square widths 256, 512 and 1024,
  then the 1024 -> 4096 layer of lab 5.3 at batch 1, 16 and 256.

The tests use square and rectangular shapes, one row (the decode shape), one column
(matrix times vector, book exercise 2) and an inner dimension of 1. C is pre-filled, so
an element you never write is reported. Where AddressSanitizer is not watching (the
`race` backend and real devices), a value stored past the end of C is reported too.

Before you benchmark: doubling the width from 512 to 1024 multiplies the arithmetic by 8.
Predict how many times longer the kernel takes on your GPU. The number you get is the
baseline chapter 5 starts from.

This is the first half of M1. Chapter 4 completes it by choosing the launch configuration
from device queries instead of a fixed 16 x 16.
