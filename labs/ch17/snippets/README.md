# Snippets

Kernels that chapter 17's page 17.2 and its summary page show for the sake of the explanation, and that are
not labs. The page loads them at run time with `useLabFile('ch17/snippets/<name>')`, so
there is no copy to drift; it shows each file without its header comment, which the
page's caption replaces. They are not built by the lab runner.

| File | Shown in | Checked by |
|---|---|---|
| `spmv_coo.cu` | page 17.2, "COO: one thread per non-zero" (Fig. 17.5 with the `#include <cuda/atomic>` the page leaves out and a `COOMatrix` struct, which the book never defines) | CUDA front-end check; `app/src/chapters/ch17/snippets-s172.test.ts` pins the atomic lines and runs it through `check/spmv_check.cu` |
| `spmv_csr.cu` | page 17.2, "CSR: one thread per row" (Fig. 17.9 with a `CSRMatrix` struct) | as above: the loop bounds and the accumulate |
| `spmv_ell.cu` | page 17.2, "ELL: pad, then turn" (Fig. 17.12 corrected: `y[row] += sum` for the page's `y[row] = sum`; pads hold column -1 and value 0, so `colIdx` and `col` are signed; `stride` for `numRows`; plus the same kernel looping the full width and testing `col >= 0`) | as above: the index line, the accumulate and the pad test, in both kernels |
| `spmv_csc.cu` | page 17.2, "CSC: the same idea, by column" (Fig. 17.18 with the singular field names `rowIdx` and `value` for the listing's `rowIdxs` and `values`, the include and a `CSCMatrix` struct) | as above: `x[col]` loaded once before the loop, the atomic add |
| `spmv_jds.cu` | the summary page, "Exercise 5, answered" (ours: the book shows no JDS kernel; one thread per sorted row, rows longest first, `y[row[r]] += sum`) | CUDA front-end check; `app/src/chapters/ch17/Wrap.snippets.test.ts` pins the highlighted lines and runs it through `check/spmv_jds_check.cu` |
| `check/spmv_jds_check.cu` | nothing (a driver) | CUDA front-end check. Built and run by `Wrap.snippets.test.ts` on the `emu` flavour (AddressSanitizer): the host-built arrays of Fig. 17.14 against p. 416, the kernel on Fig. 17.1, Fig. 17.14, exercise 1's matrix, the skewed 40 x 33 matrix and a matrix with empty rows in the three thread orders, then a copy that stores to `y[r]` instead of `y[row[r]]` (5 of 6 rows wrong on Fig. 17.14, the first question on the summary page). |
| `check/spmv_check.cu` | nothing (a driver) | CUDA front-end check. Built and run by `snippets-s172.test.ts` as the lab runner builds the `emu` backend (AddressSanitizer on): every kernel, both ELL loops and an ELL stride rounded up to a multiple of 8, on Fig. 17.1 and on a skewed 40 x 33 matrix, in the emulator's three thread orders, compared exactly with a CPU reference (integer values); then the hybrid of p. 414 run COO part first, with the page's `y[row] = sum` (4 of 4 rows wrong) and the snippet's `+=` (none); then Fig. 17.1 with pads holding column 0 and `x[0] = +Inf` (3 rows NaN) or -1 (none). Built again for `race` (ThreadSanitizer): the snippet's COO atomic reports nothing, and a plain `y[row] +=` copy is reported on its line. Skipped when no C++ compiler or python3 is found. |

`spmv_coo.cu` and `spmv_csc.cu` include `<cuda/atomic>` without a HIP fallback. They are
for the CUDA front-end check and the emulators; on HIP, `<cuda/atomic>` comes from
libhipcxx (`notes/today/infra-facts.md` B2). Nothing here has been run on a GPU:
status on page 17.2 is "emulator-checked and front-end-checked; unverified on CUDA and HIP
hardware".

The CUDA front-end check is the one `labs/pmpp.py` runs: `clang++ -x cuda --cuda-host-only
-nocudainc -nocudalib -fsyntax-only -std=c++17 -I labs/common/include -I
labs/common/include/pmpp/emu/shim -include pmpp/check/cuda_stub.h <file>`.
