# Lab 3.4: one thread per element, per row or per column

Book: exercises 1a and 1b (p. 64), on the kernel of Fig. 3.11 (p. 61).

Fig. 3.11 gives every element of P its own thread. The exercise asks for two coarser
designs: one thread computes a whole output row, and one thread computes a whole output
column. Square `Width x Width` matrices, as in the exercise.

The per-element kernel and its launch are provided in the file. **Your part:**
`rowKernel` with `launchRow`, and `colKernel` with `launchCol`, each on a 1D grid of
256-thread blocks with a guard.

- Edit `work/kernel.cu` (CUDA / HIP / emulator).
- Edit `work/kernel.metal` (`row_matmul`, `col_matmul`) and `work/launch.mm` (`encodeRow`,
  `encodeCol`). The two kernels take a scalar `[[thread_position_in_grid]]` and the
  dispatch is 1D.
- Test: `python3 labs/pmpp.py run 3.4`   Start with `--backend emu`: AddressSanitizer
  stops at the first access outside a buffer and names the line. `--backend race` runs
  the emulated threads concurrently and reports two threads that store to the same element.
- Measure: `python3 labs/pmpp.py run 3.4 --bench`

Each kernel is tested and reported on its own, so the untouched starter passes on the
provided kernel and fails on your two. The widths include ones that leave most of a
256-thread block without work.

The benchmark times all three kernels at one width. The driver picks it: it doubles the
width until the per-element kernel runs long enough to time, and stops earlier if one
more doubling would push the slowest kernel past half a second per run. It prints the
width it chose and why.

Before you benchmark: rank the three kernels from fastest to slowest, and predict how
many times slower one thread per row is than one thread per element at the benchmark
width. All three do the same arithmetic. They differ in how many threads share it and in
which direction each thread writes.

Exercise 1c asks for the pros and cons of each design. Write yours once you have the
numbers.
