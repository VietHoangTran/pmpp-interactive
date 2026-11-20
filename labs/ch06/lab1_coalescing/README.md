# Lab 6.1: coalescing on your device

Book: section 6.1 (pp. 124-132), Figs. 6.2 and 6.3, and the remark on stores (p. 130).

Write three small kernels that differ only in which address each thread touches, then
measure what each pattern costs here. None of them is a listing in the book: they are
probes of the rule section 6.1 states.

1. `stridedReadKernel`: thread `i` copies `in[i * stride]` to `out[i]`, for `i < n`.
   The write is contiguous at every stride; the read is contiguous only at stride 1.
2. `transposeReadRowsKernel`: `in` is rows x cols, `out` is cols x rows. Thread
   (x = column, y = row) reads `in[row * cols + col]` and writes `out[col * rows + row]`.
3. `transposeWriteRowsKernel`: the opposite assignment, so the write to `out` is the
   contiguous side and the read from `in` is the strided one.

Each kernel has its launch function beside it (`launchStridedRead`,
`launchTransposeReadRows`, `launchTransposeWriteRows`); the transposes use 16 x 16
blocks and a grid rounded up to cover the matrix. A copy kernel is provided
(`provided/copy.*`): it reads and writes contiguously, and its bandwidth is the figure
the other kernels are compared with.

- Edit `work/kernel.cu` and `work/kernel.metal` + `work/launch.mm`. On Metal, index
  with `int`, not `uint`: that is Apple's advice for array indexing in MSL.
- Test: `python3 labs/pmpp.py run 6.1`   Start with `--backend emu`. The input holds
  exactly `n * stride` floats, so AddressSanitizer names the line of any read past it.
  Outputs are filled with NaN before each run, so an element you never write fails.
- Measure: `python3 labs/pmpp.py run 6.1 --bench` (real devices only). It times the copy,
  the strided read at strides 1 to 64 over one input of 2^25 floats, both transposes on a
  4096 x 4096 matrix (smaller if your device reports too little memory), and prints the
  SIMD width your device reports.

Before you benchmark, write down two predictions.

1. **Stride 2 as a percentage of stride 1.** NVIDIA's guide counts memory traffic in
   32-byte sectors and says a stride-2 load uses half of every sector it fetches. The
   figure this lab prints counts the bytes your kernel uses, read and written, and the
   write is contiguous at every stride. If time were proportional to the bytes moved,
   what percentage would the sector model give for that figure? Then decide whether
   your device's caches could change it. On AMD the unit is a 64- or 128-byte cache
   line. Apple documents no coalescing rule, line size or sector size, so on a Mac this
   measurement is the only answer.
2. **Which naive transpose is faster**, rows read (strided writes) or rows written
   (strided reads), and by how much? The book says uncoalesced stores can hurt more than
   uncoalesced loads (p. 130); NVIDIA ties the extra cost of scattered writes to ECC on
   GDDR memory. A 16 x 16 block reads and writes the same 256 elements either way: does
   anything keep the strided side's lines on chip while the block's other threads use
   them?

The numbers you get are measurements of your machine, not expectations.
