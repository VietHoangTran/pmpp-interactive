# Lab 5.1: where is your ceiling?

Book: section 5.1 (pp. 93-97) and the kernel of Fig. 3.11 (p. 61).

Two probe kernels are provided. One (vector addition) is limited by memory bandwidth, the
other (a chain of multiply-adds in registers) by arithmetic. Their measurements give your
device's threshold in FLOP/B.

**Your part:** write the naive matrix multiplication (one thread per output element) and
its launch, for any width.

- Edit `work/kernel.cu` (CUDA / HIP / emulator) and `work/kernel.metal` + `work/launch.mm` (Metal).
- Test: `python3 labs/pmpp.py run 5.1`
- Measure: `python3 labs/pmpp.py run 5.1 --bench`

Before you benchmark, predict: the model says a 0.25 FLOP/B kernel can reach at most a
quarter of your bandwidth figure in GFLOP/s. Will your kernel land below that, at it, or
above it? Then explain what you see (section 5.1 of the lesson discusses it).

The numbers are measurements of two simple kernels on your machine, not vendor peaks.
