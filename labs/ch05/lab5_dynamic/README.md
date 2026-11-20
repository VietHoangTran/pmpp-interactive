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
- Measure: `python3 labs/pmpp.py run 5.5 --bench`

Before you benchmark, predict: at the same tile width, is this kernel faster, slower or
the same as lab 5.2's, where the tile width was a compile-time constant? Why?
