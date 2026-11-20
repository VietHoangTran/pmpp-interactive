# Lab 17.1: CSR and ELL SpMV on skewed rows

Book: sections 17.2 to 17.5 (pp. 404-415), Figs. 17.5, 17.9 and 17.12.

SpMV is sparse matrix-vector multiply **and accumulate**: `y = A·x + y` (p. 403). You write
it twice, one thread per row both times. Once in CSR, where a row is a run of entries
between two pointers. Once in ELL, where every row is padded to the same width and the
padded rectangle is stored column-major, so the threads of a warp read neighbouring
addresses in every iteration. The driver also runs the hybrid of section 17.5: rows longer
than K spill their tail into a COO list, which a provided kernel adds with atomics before
your ELL kernel runs.

The benchmark then asks the chapter's question on your device: what do padding,
column-major layout, sorting rows by length and spilling the long rows buy, on matrices
whose row lengths are uniform, spread out, sorted, and skewed by a few very long rows?

Where this sits under what you know (ours, not the book's): a ragged batch is the same
problem. Padding every sequence to the longest is ELL; bucketing sequences by length is
the JDS sort; serving the few long ones down a separate path is the hybrid.

## Your part

- **`spmvCsrKernel`** and **`launchSpmvCsr`**: one thread per row. Loop over the row's
  entries `rowPtrs[row] .. rowPtrs[row+1] - 1`, keep the sum in a register, then add it to
  `y[row]`. Fig. 17.9.
- **`spmvEllKernel`** and **`launchSpmvEll`**: one thread per row, over column-major arrays.
  Slot `t` of row `r` is at `t * stride + r`, with `stride >= numRows`. Pad slots hold
  column -1 and value 0. Stop at `nnzPerRow[row]` (the book's way), or visit all `width`
  slots and skip those whose column is negative (the libraries' way). Then add the sum to
  `y[row]`. Fig. 17.12, with the changes listed below.
- Both launchers: blocks of `SPMV_BLOCK` threads (128, from `spmv.h`), enough blocks to
  cover every row, and no launch at all when `numRows` is 0.
- Metal: the kernels `spmv_csr` and `spmv_ell`, which get their row from
  `[[thread_position_in_grid]]`, and `encodeSpmvCsr` and `encodeSpmvEll`, which encode
  nothing when `numRows` is 0 and otherwise dispatch one thread per row with
  `dispatchThreads`, in threadgroups of `SPMV_BLOCK` threads or the pipeline's
  `maxTotalThreadsPerThreadgroup`, whichever is smaller.

## Where this lab differs from the page

- **Raw pointers, not structs.** The book passes a `CSRMatrix`, `ELLMatrix` or `COOMatrix`
  struct to each kernel and shows neither the struct nor any host code. Here each array is
  its own argument, with explicit sizes, which maps one-to-one onto Metal buffer bindings
  (`notes/today/ch17-facts.md` A11, C2). The launchers are ours.
- **The ELL kernel accumulates.** Fig. 17.12 ends with `y[row] = sum;`, Fig. 17.9 with
  `y[row] += sum;`, and the text calls both an accumulation (pp. 409, 412). SpMV is
  multiply and accumulate, and every library computes an accumulate (ch17-facts A10). Write
  `+=` in both. It is also what lets the hybrid run the COO part first: with `=`, your ELL
  kernel would wipe out what the COO kernel added.
- **A stride, not numRows.** Fig. 17.12 steps from one slot of a row to the next by
  `numRows`. Here by `stride`, which can be larger. That is the book's own remark (p. 413):
  append a few rows before transposing so that every iteration starts on an aligned
  address. The tests use `stride = numRows`, `numRows` rounded up to 8 and `numRows + 37`.
- **Pads hold column -1.** The book says pad values are zero but not what a pad's column
  holds. Here -1, the convention of cuSPARSE's Sliced ELL and rocSPARSE's ELL (ch17-facts
  A6, B3). So a loop over the full width must test the column before reading `x`.
- The provided COO kernel is Fig. 17.5 as the book writes it (`cuda::atomic_ref`,
  `fetch_add`, `memory_order_relaxed`), with `#include <cuda/atomic>`, which the listing
  does not show (ch17-facts A1).

## Files

- Edit `work/kernel.cu` (CUDA, HIP and both emulators):
  - `__global__ void spmvCsrKernel(const int* rowPtrs, const int* colIdx, const float* values, const float* x, float* y, int numRows)`
  - `void launchSpmvCsr(` the same arguments `)`
  - `__global__ void spmvEllKernel(const int* colIdx, const float* values, const int* nnzPerRow, const float* x, float* y, int numRows, int width, int stride)`
  - `void launchSpmvEll(` the same arguments `)`
- Edit `work/kernel.metal` (`spmv_csr`, `spmv_ell`) and `work/launch.mm` (`encodeSpmvCsr`,
  `encodeSpmvEll`). The kernel signatures and the buffer bindings are given.
- Provided, not edited: `spmv.h` (`SPMV_BLOCK`); `provided/coo_spmv.cu`,
  `provided/coo_spmv.metal` and `provided/coo_spmv.mm` (`launchSpmvCoo` and
  `encodeSpmvCoo`, the hybrid's COO part); `main.cpp` (the matrix generator, the format
  builders, the CPU reference, the tests and the benchmark).

## Commands

- Test: `python3 labs/pmpp.py run 17.1 --backend emu` first, then `python3 labs/pmpp.py run 17.1`.
- The hybrid's atomics, under ThreadSanitizer: `python3 labs/pmpp.py run 17.1 --backend race`.
- Measure: `python3 labs/pmpp.py run 17.1 --bench` (real devices only).
- The reference solution: add `--solution`.

## What the tests do

Every format is built on the host from one generated matrix, and the CPU reference
accumulates in double. A row passes when it is within `1e-4 × (1 + |y0| + Σ|a·x|)` of the
reference: a tolerance scaled by the size of the row's terms, never relative-only near zero,
because Metal does not default to strict IEEE arithmetic (ch05-facts C7) and a float atomic
add on global memory flushes subnormals to zero (ch10-facts A6).

- **Matrices**, each tested in CSR and in ELL: 0 rows; 1 row of 12 in 20 columns; 7, 33, 65
  and 129 rows of random length (none a multiple of the block); 300 rows by 305 columns;
  every third row empty; every row empty; every row 5 long; rows of 1 to 3 with one row of
  40; 200 rows of random length 0 to 12. Real devices add the benchmark's four matrices
  (below) at 65,541 rows.
- **ELL** at width = the longest row and stride = `numRows`, `numRows` rounded up to 8, and
  `numRows + 37` (also rounded up to 32 on the large matrices), and once with a width 3
  more than any row, so that whole columns are pads.
- **Hybrid** on four of the matrices at K = 0 (everything in COO), 1, the median row length
  and the longest row (no COO part): the provided COO kernel first, then your ELL kernel on
  the same `y`. The provided COO kernel is also tested alone, so a broken copy shows.
- `y` starts as random non-zero values `y0` and must end as `y0 + A·x`. A failure message
  says when every wrong row holds `A·x` alone (you assigned instead of adding) or still
  holds `y0` (nothing ran).
- `y` has 128 elements after its last row, set to a sentinel that must survive. Every
  array your kernel reads also has a few elements after its end, so a thread past the last
  row stays inside the buffers, and what those elements hold makes it change `y`'s guard.
- On CUDA, HIP, `emu` and `race`, `x` is handed to your kernel as buffer + 1, and the
  element before `x[0]` is NaN: reading `x` at a pad's column -1 turns the row into NaN, and
  the message says so.
- The ELL slots of the alignment rows `numRows .. stride-1` hold value NaN and column 0, so
  a kernel that steps by `numRows` instead of `stride` fails.

What each backend can catch:

- `emu` runs one thread at a time, in three fixed orders, under AddressSanitizer: wrong
  indices, a missing row check, reads past the end of an array. It also checks the
  `numRows = 0` case: a launch of zero blocks is an error there, as on HIP (ch02-facts B4).
  CUDA's documentation does not state that case, so on CUDA a launch of zero blocks may
  pass unnoticed.
- `race` runs the 33- and 65-row matrices with the threads concurrent under
  ThreadSanitizer. Your kernels need no atomics (one thread owns each row); the COO kernel
  of the hybrid does, and a plain `+=` there is reported with its source line.
- **Metal** gets no NaN guard before `x`: its default fast math makes NaN propagation
  unreliable, and the encode functions take whole buffers, so there is no checked way to
  hand the kernel `x + 1`. A kernel that reads `x[-1]` reads whatever lies before the
  buffer, which may or may not show as a failure: do not count on it, run `emu`. With `dispatchThreads` there are no threads past the last row,
  so the row check is only exercised if you dispatch whole threadgroups. In the
  `numRows = 0` tests the driver puts a counter in place of the encoder: any dispatch is
  counted, not run, and fails the test.

## Where the backends differ (not in the book)

- **CUDA**: `cuda::atomic_ref<float, cuda::thread_scope_device>` from `<cuda/atomic>`; pass
  the order explicitly, as the book does, because the default is `seq_cst` (ch17-facts A1).
  Coalescing is counted in 32-byte sectors on every device current CUDA supports, so an ELL
  iteration starts aligned when the stride is a multiple of 8 floats (ch17-facts A3, our
  arithmetic). One ELL iteration of a 32-thread warp reads 128 contiguous bytes of each
  array.
- **HIP**: the same source. `<cuda/atomic>` comes from libhipcxx, ROCm 10.1 and later
  (infra-facts B2). AMD documents 64- and 128-byte lines, so alignment wants a stride that
  is a multiple of 16 or 32 floats; a 64-lane wavefront reads 256 contiguous bytes per ELL
  iteration (ch17-facts B1). Not run on AMD hardware here: unverified.
- **Metal**: `[[thread_position_in_grid]]` is the row, and `dispatchThreads` makes exactly
  `numRows` threads (ch05-facts C4, C5). The COO kernel declares `y` as
  `device atomic_float*` and adds with `atomic_fetch_add_explicit(..., memory_order_relaxed)`:
  Metal 3 and Apple7, device memory (ch17-facts C1). Your ELL kernel binds the same buffer as
  `device float*`. The COO and ELL dispatches share one compute pass, which runs them in
  order (ch02-facts C9). Apple documents no coalescing unit, alignment rule or divergence
  model (ch17-facts C3): on Metal the benchmark is a measurement with nothing to check it
  against.

## Before you benchmark, predict

The benchmark builds four matrices with the same number of rows, about 16 entries per row
on average, columns within ±2048 of the diagonal:

- `uniform`: every row 16 long.
- `spread`: lengths uniform on 1 to 31, in random order.
- `spread_sorted`: the same rows, longest first (the JDS sort of section 17.6). Each row
  keeps its columns, so neighbouring rows no longer read neighbouring parts of `x`.
- `skewed`: rows of 16, and one row in 512 of length 256.

It times `csr`, `ell` (stride = `numRows`), `ell_aligned` (stride rounded up to a multiple
of 32 floats) and, on `spread` and `skewed`, `hybrid` (K = 16, the typical length, COO and
ELL timed together). Write these down first. The lesson asks for the first two under the
ids in brackets.

1. On `uniform`, CSR time divided by ELL time (`ch17/labs/p-lab1-uniform`).
2. On `skewed`, ELL time divided by CSR time, and whether the hybrid gets back to CSR's
   time or better (`ch17/labs/p-lab1-skewed`). The benchmark prints CSR / ELL; on `skewed`
   it also prints ELL / CSR.
3. Not recorded: will CSR run faster on `spread_sorted` than on `spread`? Will
   `ell_aligned` beat `ell`?

The lesson's models count memory requests and divergent iterations; they ignore caches.
Apple documents no coalescing or divergence model at all (ch17-facts C3). Neither answer is
promised by anything you have read: that is why you measure.

## Reading the output

- The size is chosen from your device: 2^16 + 5 rows, doubled while CSR on `uniform` takes
  under 2 ms, up to 2^21 + 5, and only while the `skewed` matrix in ELL still fits in a
  quarter of the memory the device reports. The `+ 5` makes `stride = numRows` deliberately
  not a multiple of 8. A format that would not fit is printed as skipped, with the bytes it
  needs.
- Every format's result is checked against the CPU reference before it is timed; a wrong
  result is reported as wrong, without a time. A time is the best of `--reps` runs (5 by
  default) after one warm-up. On Metal it is the GPU's start-to-end time of one command
  buffer; on CUDA and HIP, host wall-clock time around the launches and a synchronise.
- `width`, `stride`: the ELL part's shape. `pad`: the share of the stored ELL slots that
  hold no entry (alignment rows included). `COO`: the hybrid's spilled entries.
- `matrix MB`: what the format stores (`matrix_bytes`). A kernel that stops at
  `nnzPerRow[row]` does not read the pads, so this is what a full-width loop would read, not
  what yours did.
- `GB/s`: useful bytes divided by time. Useful bytes are 8 per entry, plus 16 per row (one
  pointer or length, `x` once, `y` read and written): the least traffic any format could
  make, the same numerator for every format. It is not what the memory system moved.
- `pmpp.py` also saves each row (`name: spmv`, with `matrix`, `format`, `rows`, `nnz`,
  `width`, `stride`, `k` for the hybrid, `coo_entries`, `seconds`, `matrix_bytes`,
  `pad_fraction`, `gbps`) in `reports/runs/`, where the app reads them.

The numbers are measurements of your machine at that size. They are not properties of the
formats.

## Go further

- Exercise 5 (p. 422): write the JDS kernel. The `spread_sorted` matrix is already in JDS
  row order; what is left is the column-major layout without padding and the `iterPtr`
  array.
- Exercise 4 (p. 422) asks for the hybrid's COO part on the host. Time that against the
  provided kernel at K = 16 on `skewed`, copies included.
- Replace your ELL loop by the other correct one (full width with a column test, or
  `nnzPerRow`) and measure both on `spread` and `skewed`. Which one reads the pads?
