# Lab 5.5: tile width chosen at run time

Book: section 5.6 (pp. 116-118), Fig. 5.14. The figure has several errors as printed;
the lesson lists them.

Write the tiled, boundary-checked kernel with the tile width as an argument instead of a
`#define`, using dynamically sized shared memory:

- CUDA: one unsized `extern __shared__` array, carved into the two tiles by hand, with
  its size in bytes passed as the third launch parameter.
- Metal: a `threadgroup float*` kernel parameter bound with `[[threadgroup(0)]]`, sized
  by the host with `setThreadgroupMemoryLength:atIndex:`.

The driver asks the device for its limits, derives the largest tile that fits, and tests
tile widths 4, 8, 16 and that largest one with one binary.

- Edit `work/kernel.cu` and `work/kernel.metal` + `work/launch.mm`.
- Test: `python3 labs/pmpp.py run 5.5`
- Measure: `python3 labs/pmpp.py run 5.5 --bench`. It times widths 512, 1024, 2048 and
  4096 at every power-of-two tile width that fits. Those are the widths lab 5.2's benchmark
  can settle on (it doubles from 512 until its naive kernel is slow enough to time, up to
  4096), so the labs page compares the two kernels at the width lab 5.2 used. Run lab 5.2's
  benchmark too, on the same backend.

Before you benchmark, predict: at the same tile width (16) and the same matrix width, is
this kernel faster, slower or the same as lab 5.2's, where the tile width was a
compile-time constant? Why?
