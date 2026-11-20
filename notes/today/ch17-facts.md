# Chapter 17 fact check: sparse matrix computation (CUDA, HIP, Metal, real backends)

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims the verdict is for the item as a whole and the text says which sub-claim failed or could not be found. A verdict judges the book's statement, or the assumption a course author would most likely bring (stated in the item title), not the vendor.
- **Quote** is copied from the source (HTML flattened to text, GitHub raw files at a tag or on the default branch, Apple's documentation JSON, PDF text, local SDK headers). Line breaks inside quotes are removed.
- Versions the sources reported about themselves on the check date: cuSPARSE "13.5 documentation" (last updated Nov 04, 2026); CUDA C++ Best Practices Guide 13.5 (Nov 02, 2026); CUDA Programming Guide v13.5; CCCL docs at `nvidia.github.io/cccl/unstable/`; ROCm Core SDK 10.2.0 (rocSPARSE docs "5.1.0", hipSPARSE docs "4.8.0", HIPIFY docs), sources at tag `therock-10.2` of `ROCm/rocm-libraries`; Metal Shading Language Specification "Version 4.1" (2026-06-04, downloaded again today); macOS SDK 27.0 headers on this machine; PyTorch docs "2.15" (sparse page last updated May 12, 2026); MLX docs "0.32.4"; BLAS Technical Forum standard pages on netlib. Repositories (llama.cpp `master`, PyTorch `main`, MLX `main`, mlx-lm `main`) read today; each file's last commit date is given where it is cited.
- Not checked: anything by running code on a GPU. Nothing here was compiled for CUDA, ROCm or Metal. Group E is our own arithmetic, checked with a short script, not a vendor statement.

Already settled elsewhere, referred to by item and not repeated:

- `ch05-facts.md` B4 (`__global__`, index built-ins on HIP), B5 (wavefront 64 on gfx9, 32 on gfx10 and later), B9 (llama.cpp builds `ggml-cuda/*.cu` as HIP), C3 (SIMD-group width only from the pipeline), C4 and C5 (`thread_position_in_grid`, `dispatchThreads`).
- `infra-facts.md` A1 and B1 (legacy `atomicAdd(float*, float)`; on HIP safe, possibly a CAS loop, unless `-munsafe-fp-atomics`), A2 and B2 (`<cuda/atomic>`, `cuda::atomic_ref`, scopes, both memory-order spellings; libhipcxx keeps the spelling on ROCm 10.1 and later), C1 (`atomic_float` add and sub, device memory, Metal 3 and later; threadgroup memory from Metal 4.1), C13 (Apple documents no caches, banks or bandwidth figures).
- `ch09-facts.md` A1 (the `atomic_ref` constructor takes a reference; `fetch_add` defaults to `seq_cst`), A3 (atomics on one address serialise "in an undefined order"), C1 (MSL has no `atomic_ref`: the buffer parameter is declared with an atomic type).
- `ch10-facts.md` A6 (float atomic add on global memory is flush-to-zero), A9 (floating-point addition is not associative; NVIDIA's own worked example), B3 (`fetch_add` in libhipcxx is `__hip_atomic_fetch_add(..., __HIP_MEMORY_SCOPE_AGENT)`), C2 (Metal float atomic add yes, float atomic max no).
- `ch06-facts.md` A2 and A3 (32-byte sectors; a misaligned warp run costs five sectors instead of four; `cudaMalloc` aligns to at least 256 bytes), B1 and B2 (AMD coalescing: 64- and 128-byte lines, 64-lane wavefronts on CDNA), C1 (Apple documents no coalescing rule).
- `ch04-facts.md` A9 (warp divergence on NVIDIA), B7 (EXEC-mask divergence on AMD), C4 (Apple documents no divergence model).
- `ch12-facts.md` B1 (ballot and popcount compaction on HIP: 64-bit masks), used by D4 and D5 below.
- `ch09-facts.md` D3 (CUB and rocPRIM device histograms) and `ch11-facts.md` A8 and B2 (CUB, hipCUB and rocPRIM scans): the primitives of exercise 3.
- `ch15-facts.md` A5 (CUDA 13 targets compute capability 7.5 and later; cuSPARSE "Dropped support for pre-Turing architectures").

Totals: 22 CONFIRMED, 1 WRONG, 5 PARTLY, 2 COULD NOT VERIFY (30 items: A1 to A13, B1 to B5, C1 to C4, D1 to D7, E1). Every WRONG, PARTLY and COULD NOT VERIFY item is listed under "Summary of corrections".

---

## Group A: CUDA, and the book's dated or device-specific statements

### A1. `cuda::atomic_ref<float, cuda::thread_scope_device>` with `fetch_add(..., cuda::memory_order_relaxed)` in Figs. 17.5 and 17.18 (pp. 405, 419)

**Verdict: CONFIRMED.** Everything the two listings need is settled in `infra-facts.md` A2, `ch09-facts.md` A1 and `ch10-facts.md` A6. For this chapter's code:

- The listings show no `#include`; the header is `<cuda/atomic>` (infra A2).
- `y_ref(y[row])` is the right form: the constructor is `explicit constexpr atomic_ref(_Tp& __ref)` (ch09 A1).
- The book passes `cuda::memory_order_relaxed` explicitly, which matters: the default is `seq_cst` (ch09 A1). Relaxed is the right order for a commutative accumulation.
- Float atomic adds on global memory are flush-to-zero (ch10 A6), so a COO or CSC result can differ from a CPU reference in the subnormal range; the lab tolerance must not be relative-only near zero.
- The legacy spelling `atomicAdd(&y[row], x[col]*value)` is the same operation with relaxed order and device scope (infra A1). NVIDIA calls it "Legacy".

### A2. Atomics have "high latency and low throughput" and a non-deterministic order of floating-point adds (pp. 406 to 407)

**Verdict: CONFIRMED.** The order half is ch09 A3 ("serialized, but the order in which they occur is undefined") with ch10 A9 (non-associativity). New today: both vendor sparse libraries offer exactly the book's trade, a faster atomic or unordered algorithm and a slower deterministic one.

cuSPARSE 13.5, `cusparseSpMV()` algorithm table:

- `CUSPARSE_SPMV_COO_ALG1`: "Default algorithm for COO sparse matrix format. May produce slightly different results during different runs with the same input parameters."
- `CUSPARSE_SPMV_COO_ALG2`: "Provides deterministic (bit-wise) results for each run. If opA != CUSPARSE_OPERATION_NON_TRANSPOSE, it is identical to CUSPARSE_SPMV_COO_ALG1."
- `CUSPARSE_SPMV_CSR_ALG1` and `CUSPARSE_SPMV_CSR_ALG2`: the same pair of sentences for CSR/CSC.
- `CUSPARSE_SPMV_SELL_ALG1` and `CUSPARSE_SPMV_BSR_ALG1`: "Provides deterministic (bit-wise) results for each run."
- Performance notes: "CUSPARSE_SPMV_COO_ALG1 and CUSPARSE_SPMV_CSR_ALG1 provide higher performance than CUSPARSE_SPMV_COO_ALG2 and CUSPARSE_SPMV_CSR_ALG2."

rocSPARSE (B4) says the same for COO: `rocsparse_spmv_alg_coo_atomic` is not deterministic, "Generally the fastest COO algorithm"; `rocsparse_spmv_alg_coo` is deterministic, "Generally not as fast as the atomic algorithm but is deterministic."

An inconsistency inside the cuSPARSE documentation, for the record: section 2.4 "Result Reproducibility" says "performing the same API call twice with the exact same arguments, on the same machine, with the same executable will produce bit-wise identical results" for every non-transposed operation, while the `cusparseSpMV()` table above says ALG1 "May produce slightly different results during different runs with the same input parameters." Teach the per-algorithm table; it is the more specific statement.

- https://docs.nvidia.com/cuda/cusparse/index.html (sections 2.4 and 6.6.7)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsparse/library/include/internal/generic/rocsparse_spmv.h

### A3. "Some GPU architectures, especially in the older generations, have more strict address alignment rules", alignment unit "such as 64-bytes" (p. 413)

**Verdict: PARTLY.** Alignment still matters, but at a different unit, and the 64-byte figure appears in no NVIDIA document read today.

- Current rule (ch06 A2, A3, re-read today in the Best Practices Guide 13.5, section 10.2.1): "For devices of compute capability 6.0 or higher, the requirements can be summarized quite easily: the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal to the number of 32-byte transactions necessary to service all of the threads of the warp." And: "If sequential threads in a warp access memory that is sequential but not aligned with a 32-byte segment, five 32-byte segments will be requested".
- The only older rule the same section still prints: "For certain devices of compute capability 5.2, L1-caching of accesses to global memory can be optionally enabled. If L1-caching is enabled on these devices, the number of required transactions is equal to the number of required 128-byte aligned segments."
- Searched for a 64-byte coalescing segment: not in the 13.5 Best Practices Guide, and not in the archived CUDA 8.0 programming guide (https://docs.nvidia.com/cuda/archive/8.0/cuda-c-programming-guide/index.html). The book names no GPU, so the figure cannot be tied to a generation.
- Every device the current toolkit and cuSPARSE support is compute capability 7.5 or later (ch15 A5), so the 32-byte sector rule is the one that applies.

What the alignment trick becomes today (our arithmetic from the quotes above, not a vendor statement): in ELL, iteration `t` of the column-major array starts at element `t*numRows`. With 4-byte floats and a `cudaMalloc` base aligned to at least 256 bytes, every iteration starts on a 32-byte boundary when `numRows` is a multiple of 8. On AMD the documented lines are 64 and 128 bytes (ch06 B1, B2), which needs a multiple of 16 or 32 rows. Apple documents no unit (ch06 C1). JDS cannot do this cheaply, which is the book's own point (p. 417).

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 10.2.1)

### A4. cuSPARSE as the GPU sparse library; reference [4] `http://docs.nvidia.com/cuda/cusparse` (pp. 422 to 423)

**Verdict: CONFIRMED.** The URL resolves (`http://docs.nvidia.com/cuda/cusparse` redirects 301 to `https://docs.nvidia.com/cuda/cusparse/`, HTTP 200), titled "cuSPARSE 13.5 documentation". What a lesson should know about it today:

- Two API families: "The Legacy APIs, inspired by the Sparse BLAS standard, provide a limited set of functionalities and will not be improved in future releases" and "The Generic APIs provide the standard interface layer of cuSPARSE. They allow computing the most common sparse linear algebra operations, such as sparse matrix-vector (SpMV) and sparse matrix-matrix multiplication (SpMM)".
- Formats documented in section 3.3: Dense, Coordinate (COO), Compressed Sparse Row (CSR), Compressed Sparse Column (CSC), Sliced Ellpack (SELL), Block Sparse Row (BSR), Blocked Ellpack (BLOCKED-ELL), and Extended BSR (BSRX) "[DEPRECATED]". No plain ELL, no hybrid, no JDS (A5).
- SpMV: `cusparseStatus_t cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, void* externalBuffer)`, computing "Y = α op(A) · X + β Y". Formats accepted: "CUSPARSE_FORMAT_COO CUSPARSE_FORMAT_CSR CUSPARSE_FORMAT_CSC CUSPARSE_FORMAT_BSR CUSPARSE_FORMAT_SLICED_ELL". Blocked-ELL is an SpMM format ("cusparseSpMM supports the following data types for CUSPARSE_FORMAT_BLOCKED_ELL format and the following GPU architectures for exploiting NVIDIA Tensor Cores").
- Mixed precision: `CUDA_R_16F` or `CUDA_R_16BF` matrix with `CUDA_R_32F` vectors and compute type is supported "only for CUSPARSE_FORMAT_CSR and CUSPARSE_FORMAT_CSC".
- An experimental variant: "cusparseSpMVOp() [EXPERIMENTAL], a variant of SpMV that can fuse an optional user-provided epilogue function, supplied as LTO-IR and linked at run time ... It produces deterministic results and is faster than cusparseSpMV() with the CSR format in the majority of cases."
- NaN behaviour that differs from dense code, worth one sentence in the padding lesson (A6): "The dot product between vectors [0, 1, 0] and [1, 1, NaN] is NaN when using typical dense numerical algorithms, but will be 1.0 with typical sparse numerical algorithms."

- https://docs.nvidia.com/cuda/cusparse/index.html (sections 1.1, 2.5, 3.3, 6.6.7)

### A5. Assumed: cuSPARSE offers the book's ELL, hybrid ELL-COO and JDS formats

**Verdict: WRONG.** None of the three is in cuSPARSE 13.5. The hybrid format was removed in CUDA 11.0; ELL survives only in the sliced and blocked forms; JDS appears nowhere.

- The string "hyb" occurs zero times in today's cuSPARSE documentation (searched the full page).
- CUDA 11.0 release notes, "CUDA Libraries -- cuSPARSE", "The following functions have been removed": "Hybrid format enums and helper functions: cusparseHybPartition_t, cusparseHybPartition_t, cusparseCreateHybMat, cusparseDestroyHybMat", "Sparse hybrid format solver: cusparseXhybsv_analysis, cusparseShybsv_solve", "Reordering: cusparseXcsc2hyb, cusparseXcsr2hyb, cusparseXdense2hyb, cusparseXhyb2csc, cusparseXhyb2csr, cusparseXhyb2dense". The list does not name `cusparse<t>hybmv` itself.
- AMD's HIPIFY table fills that gap: `cusparseShybmv` and `cusparseHybMat_t` are marked deprecated in CUDA 10.2 and removed in CUDA 11.0 (columns "D 10.2", "R 11.0"; legend "A - Added; D - Deprecated; C - Changed; R - Removed").
- AMD kept both: rocSPARSE still has ELL and HYB (B3).
- The nearest cuSPARSE format to the book's ELL family is Sliced Ellpack (A6).

For the lesson: present ELL, hybrid and JDS as the book's teaching formats; say that NVIDIA's library dropped HYB in CUDA 11.0 and ships Sliced ELL instead, while AMD's library still ships ELL and HYB.

- https://docs.nvidia.com/cuda/archive/11.0/cuda-toolkit-release-notes/index.html ("CUDA Libraries -- cuSPARSE")
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUSPARSE_API_supported_by_HIP.html
- https://docs.nvidia.com/cuda/cusparse/index.html

### A6. The sectioned JDS / ELL variant (p. 417) exists today as Sliced ELL; how vendors pad

**Verdict: CONFIRMED** (the mapping to the book's variant is ours; the format description is NVIDIA's).

cuSPARSE section 3.3.5:

- "The Sliced Ellpack format is standardized and well-known as the state of the art. This format allows to significantly improve the performance of all problems that involve low variability in the number of nonzero elements per row."
- "A matrix in the Sliced Ellpack format is divided into slices of an exact number of rows (sliceSize), defined by the user. The maximum row length (i.e., the maximum non-zeros per row) is found for each slice, and every row in the slice is padded to the maximum row length. The value -1 is used for padding."
- "To improve memory coalescing and memory utilization, each slice is stored in column-major order."
- Arrays: slice offsets of length nslices + 1, column indices and values of length `sellValuesSize` ("including non-zero values and padded elements"), "Value -1 refers to padding."
- Constructor: `cusparseCreateSlicedEll(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, int64_t sellValuesSize, int64_t sliceSize, void* sellSliceOffsets, void* sellColInd, void* sellValues, cusparseIndexType_t sellSliceOffsetsType, cusparseIndexType_t sellColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType)`.

How this relates to the book (our reading): it is the book's sectioned variant (p. 417) without the sort. Slices are runs of consecutive rows; nothing in the format reorders rows, so grouping rows of similar length is left to whoever builds the matrix. With `sliceSize` equal to the row count it is the book's ELL.

The pad convention settles an item in the notes' "Unclear in the PDF" list (the book never says what `colIdx` holds in a pad slot): both NVIDIA (SELL) and AMD (ELL, B3) store column index -1 in pad slots, and AMD's reference loop tests it before reading `x`. A kernel that loops over the full padded width must test the index, as the libraries do. Padding with column 0 and value 0 instead reads a valid `x[0]`, but gives NaN whenever `x[0]` is infinite or NaN, because 0 times infinity is NaN in IEEE arithmetic; that is the dense-versus-sparse difference cuSPARSE describes in A4 (our reasoning).

- https://docs.nvidia.com/cuda/cusparse/index.html (sections 3.3.5 and 6.5.5.1)

### A7. COO "sorting is not required" (p. 404) and CSR within-row order "not required for correctness" (pp. 407 to 408)

**Verdict: PARTLY.** True of the book's kernels. The libraries are stricter about COO.

- cuSPARSE COO: "The COO format is assumed to be sorted by row." and, for COO and CSR, "cuSPARSE supports both sorted and unsorted column indices within a given row." and "If the column indices within a given row are not unique, the correctness of the computation is not always ensured."
- `cusparseSpMV()`: "The routine allows the indices of matA to be unsorted."
- The COO-to-CSR helper presupposes row order: `cusparseXcoo2csr(cusparseHandle_t handle, const int* cooRowInd, int nnz, int m, int* csrRowPtr, cusparseIndexBase_t idxBase)` "converts the array containing the uncompressed row indices (corresponding to COO format) into an array of compressed row pointers (corresponding to CSR format)" and "requires no extra storage".
- PyTorch makes the same choice explicit (D1): a coalesced COO tensor has unique indices "sorted in lexicographical order".

For the lesson: the book's per-non-zero kernel works on any order; library COO is row-sorted, which is what makes it convertible to CSR without a sort.

- https://docs.nvidia.com/cuda/cusparse/index.html (sections 3.3.2, 3.3.3, 5.9.6, 6.6.7)

### A8. Preprocessing is worth it when the same matrix is reused across solver iterations (pp. 414 to 415, 417)

**Verdict: CONFIRMED.** Both vendor libraries build this argument into their API.

- cuSPARSE: "Calling cusparseSpMV_preprocess() is optional. It may accelerate subsequent calls to cusparseSpMV(). It is useful when cusparseSpMV() is called multiple times with the same sparsity pattern (matA)." Afterwards "alpha, beta, vecX, vecY, and the values (but not indices) of matA may be different."
- rocSPARSE `rocsparse_spmv`: "The buffer size, buffer allocation, and preprocess stages only need to be called once for a given sparse matrix op(A), while the computation stage can be repeatedly used with different x and y vectors." Its adaptive CSR algorithm "Requires lengthy preprocessing that needs to be amortized over many subsequent sparse vector products"; nnzsplit is "Good when the number of sparse vector products that will be performed is less than one hundred."

The LLM-side analogue (one-time weight repacking before many forward passes) is our bridge, not a vendor statement.

- https://docs.nvidia.com/cuda/cusparse/index.html (section 6.6.7)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsparse/library/include/internal/generic/rocsparse_spmv.h

### A9. COO-to-CSR conversion with histogram and prefix sum (p. 410, exercise 3); is there a CUB SpMV?

**Verdict: CONFIRMED.** The library routes exist; CUB's SpMV does not any more.

- cuSPARSE has the conversion as one call, `cusparseXcoo2csr` (A7), plus `cusparseXcsr2coo` and `cusparseCsr2cscEx2()` for CSR to CSC (the book's "second, column-oriented copy", p. 409). rocSPARSE has `rocsparse_coo2csr`, `rocsparse_csr2coo`, `rocsparse_csr2csc`, `rocsparse_csr2ell`, `rocsparse_csr2hyb` (B3).
- The book's own route uses device-wide histogram and scan, which CUB and rocPRIM provide (ch09 D3, ch11 A8).
- CUB once had an SpMV. CCCL 3.0 migration guide, "Removed functions and classes": "cub::DeviceSpmv: Use cuSPARSE instead". `cub/cub/device/device_spmv.cuh` returns 404 on CCCL `main` today.

- https://docs.nvidia.com/cuda/cusparse/index.html (sections 5.9.6 to 5.9.8)
- https://nvidia.github.io/cccl/unstable/cccl/3.0_migration_guide.html

### A10. "SpMV" as the name of a "standardized library interface" for multiplication **and accumulation** (p. 403)

**Verdict: PARTLY.** The accumulate is right; the name is the libraries', not the standard's.

- The standard is the BLAS Technical Forum's Sparse BLAS (Chapter 3 of its standard). Its routine is `usmv`; the C prototypes on netlib: `int BLAS_susmv( enum blas_trans_type transa, float alpha, blas_sparse_matrix A, const float *x, int incx, float *y, int incy );`. `y` is in-out and there is no beta, which is the book's "multiply and accumulate". I could not extract the standard's formula text from its PDF (the text layer is font-encoded), so that reading rests on the prototype.
- Apple's Accelerate keeps the same semantics and says so (C4): `sparse_matrix_vector_product_dense_float` "Multiplies the dense vector x by the sparse matrix A and adds the result to the dense vector y (y = alpha * op(A) * x + y, where op(A) is either A or the transpose of A)."
- "SpMV" is the generic-API name: `cusparseSpMV`, `rocsparse_spmv`, `hipsparseSpMV`. All three compute y = α op(A) x + β y, so the book's operation is β = 1 and a plain product is β = 0. cuSPARSE describes its legacy API as "inspired by the Sparse BLAS standard" (A4).

Consequence for the corrected code (notes, "Unclear in the PDF" item 8, `y[row] = sum` in Fig. 17.12 against `y[row] += sum` in Fig. 17.9): every library computes an accumulate, and AMD's ELL reference loop is written as `y[i] = beta * y[i];` followed by `y[i] = y[i] + alpha * ell_val[idx] * x[ell_col_ind[idx]];` (B3). The lab's ELL kernel should accumulate like the CSR one, and the hybrid COO pass then composes in either order. Saying so is ours; the book does not.

- https://www.netlib.org/blas/blast-forum/ (Chapter 3: Sparse BLAS)
- https://www.netlib.org/blas/blast-forum/blas_sparse_proto.h

### A11. The listings pass a struct of device pointers (`COOMatrix`, `CSRMatrix`, ...) by value

**Verdict: CONFIRMED** that CUDA allows it. Programming Guide 13.5, section 5.3.10.5.4 "__global__ Function Parameters": "Function parameters are passed to the device via constant memory and their total size is limited to 32,764 bytes." A struct of a few pointers and integers is far below that. The course's lab convention (raw device pointers and explicit sizes as arguments) avoids the question and maps one-to-one onto Metal buffer bindings (C2). On HIP the same source compiles (ch05 B4); I found no stated kernel-argument size limit in the HIP pages read (language extensions, C++ kernel language).

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html (section 5.3.10.5.4)

### A12. Dated generalisations: real matrices have "thousands to millions of rows" with "tens to hundreds of non-zeros" per row (p. 409); sparse FLOPS are "much lower" than dense on CPUs and GPUs (p. 422)

**Verdict: COULD NOT VERIFY** as vendor statements. Neither NVIDIA's cuSPARSE pages nor AMD's rocSPARSE headers read today state that SpMV is memory-bound or give a typical row length. Two documented facts are consistent with the book:

- Apple's Sparse Solvers overview gives one benchmark matrix: "the sparse benchmark matrix ldoor, which arises from structural modeling, has 952,200 x 952,200 entries with an average of 25 nonzero [Double] entries per column."
- rocSPARSE motivates mixed precision with bandwidth, not compute: "The advantage of using different data types is to save on memory bandwidth and storage when a user application allows, while performing the actual computation in a higher precision."

For the lesson: keep both as the book's claims with their pages; the lab measures the reader's own SpMV against the bytes it moves.

- https://developer.apple.com/documentation/accelerate/sparse-solvers-library
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsparse/library/include/internal/generic/rocsparse_spmv.h

### A13. Reference [3]: Bell and Garland, SpMV on throughput-oriented processors (pp. 421, 423)

**Verdict: CONFIRMED.** NVIDIA Research still hosts it (HTTP 200): "Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors", Supercomputing '09, DOI 10.1145/1654059.1654078 (the DOI resolver returned 403 to our automated client; not checked further). The code it describes lives on as the CUSP library, `cusplibrary/cusplibrary` on GitHub ("CUSP : A C++ Templated Sparse Matrix Library", not archived, last push 2026-07-08). The paper's device figures are from 2009 hardware and are not for lessons.

- https://research.nvidia.com/publication/2009-11_implementing-sparse-matrix-vector-multiplication-throughput-oriented-processors
- https://github.com/cusplibrary/cusplibrary

---

## Group B: HIP / ROCm

### B1. The chapter's four kernels under HIP

**Verdict: CONFIRMED.** Nothing in Figs. 17.5, 17.9, 17.12 or 17.18 needs a HIP-specific spelling.

- Kernel and index built-ins: same spelling (ch05 B4).
- The per-row kernels (CSR, ELL, and a JDS kernel for exercise 5) use no atomics.
- COO and CSC need a float atomic add. Two documented routes: `cuda::atomic_ref<float, cuda::thread_scope_device>` with `fetch_add`, through libhipcxx on ROCm 10.1 and later with the `libhipcxx::libhipcxx` target (infra B2; at `therock-10.2` it lowers to `__hip_atomic_fetch_add(..., __HIP_MEMORY_SCOPE_AGENT)`, ch10 B3), or legacy `atomicAdd(float*, float)`, which is implemented safely, possibly as a compare-and-swap loop, unless built with `-munsafe-fp-atomics` (infra B1).
- Width: one ELL iteration of a 64-lane wavefront reads 256 contiguous bytes of `value` and of `colIdx` (our arithmetic from ch05 B5 and ch06 B1). Divergence in CSR is per wavefront, masked through EXEC (ch04 B7).

### B2. Coalescing and divergence for CSR versus ELL on AMD

**Verdict: CONFIRMED.** The general rules are ch06 B1 and ch04 B7. AMD's sparse library describes the book's CSR problem in its own words, for its one-row-per-group CSR algorithm:

- `rocsparse_spmv_alg_csr_rowsplit`: "Is best suited for matrices with all rows having a similar number of non-zeros. Can outperform adaptive and LRB algorithms in certain sparsity patterns. Will perform very poorly if some rows have few non-zeros and some rows have many non-zeros."
- `rocsparse_spmv_alg_csr_adaptive`: "Generally the fastest algorithm across all matrix sparsity patterns. This includes matrices that have some rows with many non-zeros and some rows with few non-zeros."
- The CHANGELOG confirms the width matters to the library: "Optimized architecture-aware launch configurations for RDNA (wave32) and CDNA (wave64) GPUs ... Affected routines include `rocsparse_spmv` for the CSR adaptive, nnz-split, and LRB algorithms, the COO (SoA and AoS) formats, and the ELL format (`rocsparse_Xellmv`)".

- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsparse/library/include/internal/generic/rocsparse_spmv.h
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsparse/CHANGELOG.md

### B3. rocSPARSE keeps the book's ELL and hybrid formats

**Verdict: CONFIRMED.** rocSPARSE "is a library that provides basic linear algebra subroutines for sparse matrices and vectors. It's created using the HIP programming language, implemented on top of the ROCm runtime and toolchains, and optimized for AMD discrete GPUs." (docs "rocSPARSE 5.1.0").

- Formats (`rocsparse-types.h` at `therock-10.2`): `rocsparse_format_coo`, `rocsparse_format_coo_aos`, `rocsparse_format_csr`, `rocsparse_format_csc`, `rocsparse_format_ell`, `rocsparse_format_bell` ("Blocked ELL"), `rocsparse_format_bsr`, `rocsparse_format_sell` ("Sliced ELL"), plus a separate HYB matrix object.
- ELL is the book's layout exactly. `rocsparse_ellmv` reference loop as printed in the header: `for(p = 0; p < ell_width; ++p) { idx = p * m + i; if((ell_col_ind[idx] >= 0) && (ell_col_ind[idx] < n)) { y[i] = y[i] + alpha * ell_val[idx] * x[ell_col_ind[idx]]; } }`, preceded by `y[i] = beta * y[i];`. Parameter note: "Padded column indices should be -1." `idx = p * m + i` is the book's `t*numRows + row`.
- Width helper: "`rocsparse_csr2ell_width` computes the maximum of the per row non-zero elements over all rows, the `ell_width`, for a given CSR matrix."
- HYB partitioning, the book's §17.5 choice of width: `rocsparse_hyb_partition_auto` ("Automatically decide on ELL nnz per row."), `rocsparse_hyb_partition_user` ("User-provided ELL nnz per row."), `rocsparse_hyb_partition_max` ("Max ELL nnz per row, no COO part.").
- `rocsparse_hybmv` "multiplies the scalar α with a sparse m × n matrix, defined in HYB storage format, and the dense vector x and adds the result to the dense vector y that is multiplied by the scalar β". `rocsparse_csr2hyb`: "This function requires a significant amount of storage for the HYB matrix, depending on the matrix structure." and "This function is blocking with respect to the host." (the book's preprocessing cost, p. 414).
- Neither `rocsparse_hybmv.h` nor `rocsparse_ellmv.h` contains a deprecation mark at the tag.
- Sliced ELL in the generic SpMV: the CHANGELOG says "Added sliced ELL format support to the `rocsparse_spmv` routine", and the v2 header lists `rocsparse_spmv_alg_sell` as deterministic. The v1 `rocsparse_spmv.h` note on supported formats lists BSR, COO, COO (AoS), CSR, CSC and ELL only, an omission inside AMD's own headers.

- https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/index.html
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsparse/library/include/rocsparse-types.h (lines 484 to 492, 646 to 655, 850 to 864)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsparse/library/include/internal/level2/rocsparse_ellmv.h
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsparse/library/include/internal/level2/rocsparse_hybmv.h
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsparse/library/include/internal/conversion/rocsparse_csr2hyb.h
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsparse/library/include/internal/conversion/rocsparse_csr2ell.h
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsparse/library/include/internal/generic/rocsparse_v2_spmv.h

### B4. rocSPARSE's SpMV algorithms, and row binning as the JDS idea in a library

**Verdict: CONFIRMED.** The `rocsparse_spmv` header carries a table with columns "Deterministic" and "Preprocessing":

| Algorithm | Deterministic | Preprocessing | AMD's note (abridged quote) |
|---|---|---|---|
| `rocsparse_spmv_alg_csr_rowsplit` | Yes | No | "best suited for matrices with all rows having a similar number of non-zeros" |
| `rocsparse_spmv_alg_csr_adaptive` | No | Yes | "Generally the fastest algorithm across all matrix sparsity patterns" |
| `rocsparse_spmv_alg_csr_lrb` | No | Yes | "uses a much faster pre-processing step. Good for when only a small number of sparse vector products will be performed." |
| `rocsparse_spmv_alg_csr_nnzsplit` | No | Yes | "faster than the LRB algorithm ... Good when the number of sparse vector products that will be performed is less than one hundred." |
| `rocsparse_spmv_alg_coo` | Yes | Yes | "not as fast as the atomic algorithm but is deterministic" |
| `rocsparse_spmv_alg_coo_atomic` | No | No | "Generally the fastest COO algorithm." |
| `rocsparse_spmv_alg_ell` | Yes | No | |
| `rocsparse_spmv_alg_bsr` | Yes | No | |

Also: "None of the algorithms above are deterministic when A is transposed." `rocsparse_spmv_alg_csr_stream` is the deprecated old name of rowsplit.

LRB groups rows by length, which is the book's sort-by-length idea (§17.6) done as bucketing. The header only says "(LRB)" and does not expand the name. The source at the tag bins rows into 32 bins (`n_rows_bins{}; // size of 32`) whose limits are powers of two (`uint32_t bin_max_row_len = (1 << j);` in `rocsparse_csrmv_template_lrb.cpp`), with the comment "Optionally, sort bins (adds preprocessing cost, but often substantially reduces SpMV consumer-kernel time)". The power-of-two reading is from the code, not from documentation.

- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsparse/library/include/internal/generic/rocsparse_spmv.h
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsparse/library/src/include/rocsparse_lrb_info.hpp
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsparse/library/src/level2/rocsparse_csrmv_template_lrb.cpp

### B5. Assumed: hipSPARSE gives the cuSPARSE SpMV calls the same meaning on AMD

**Verdict: PARTLY.** The names carry over; the documented guarantees do not.

- hipSPARSE (docs "hipSPARSE 4.8.0"): "hipSPARSE presents a common interface for supporting both rocSPARSE and NVIDIA CUDA cuSPARSE as backends. It's a SPARSE marshalling library".
- HIPIFY maps `cusparseSpMV` to `hipsparseSpMV`, `cusparseCreateSlicedEll` to `hipsparseCreateSlicedEll`, `cusparseXcoo2csr` to `hipsparseXcoo2csr`, `CUSPARSE_SPMV_CSR_ALG2` to `HIPSPARSE_SPMV_CSR_ALG2`.
- The enums are defined per backend: `hipsparseFormat_t` on AMD (`#if(!defined(CUDART_VERSION))`) has CSR, CSC, COO, COO_AOS, BLOCKED_ELL, SLICED_ELL, BSR; on CUDA 12.1 and later it has CSR, CSC, COO, BLOCKED_ELL, SLICED_ELL. `hipsparseSpMVAlg_t` on AMD adds `HIPSPARSE_SPMV_CSR_ALG3`, which has no cuSPARSE counterpart.
- How the names land on AMD (`amd_detail/utility.h`, `hipSpMVAlgToHCCSpMVAlg`): `HIPSPARSE_SPMV_CSR_ALG1` to `rocsparse_spmv_alg_csr_adaptive`, `HIPSPARSE_SPMV_CSR_ALG2` to `rocsparse_spmv_alg_csr_rowsplit`, `HIPSPARSE_SPMV_CSR_ALG3` to `rocsparse_spmv_alg_csr_nnzsplit`, both `HIPSPARSE_SPMV_COO_ALG1` and `HIPSPARSE_SPMV_COO_ALG2` to `rocsparse_spmv_alg_coo` (the deterministic segmented one, not the atomic one), `HIPSPARSE_SPMV_SELL_ALG1` to `rocsparse_spmv_alg_sell`.
- Determinism: hipSPARSE's own `hipsparseSpMV` table lists the algorithm names with no "Deterministic" column. So "CSR_ALG2 is deterministic on AMD" follows from the mapping in the source plus rocSPARSE's table (B4), not from a hipSPARSE statement. PyTorch's code reaches the same conclusion from the other side (D1).
- HYB through hipSPARSE: `hipsparseXhybmv` exists; its header: "This function is deprecated when using the CUDA backend (CUDA 10.0+) and will be removed in CUDA 11.0. This deprecation does not apply to the ROCm backend."

- https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/index.html
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipsparse/library/include/hipsparse-generic-types.h (lines 200 to 260, 310 to 360)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipsparse/library/src/amd_detail/utility.h (lines 490 to 516)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipsparse/library/include/internal/generic/hipsparse_spmv.h
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipsparse/library/include/internal/level2/hipsparse_hybmv.h
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUSPARSE_API_supported_by_HIP.html

---

## Group C: Metal

### C1. The chapter's four kernels in MSL

**Verdict: CONFIRMED.** The per-row kernels need nothing beyond the indexing of ch05 C4 and C5 (`[[thread_position_in_grid]]` gives the row directly; with `dispatchThreads` there is no out-of-range row). The two atomic kernels (COO, CSC) need `atomic_fetch_add_explicit(&y[row], x[col]*value, memory_order_relaxed)` on a buffer declared `device atomic_float*`: float add exists in device memory from Metal 3 and Apple7 (every Apple silicon Mac), relaxed is the only order the spec allows for fetch-and-modify, and there is no `atomic_ref` (infra C1, ch09 C1, ch10 C2). Because the buffer type differs, the COO kernel's `y` cannot be the same `device float*` the CSR kernel writes; the host allocates plain floats either way (ch09 C1).

### C2. The book's struct of device pointers in MSL

**Verdict: CONFIRMED** for the language feature. MSL spec 4.1, section 2.13 "Argument Buffers": "All OS: Metal 2 and later support argument buffers." "Argument buffers extend the basic buffer types to include pointers (buffers), textures, texture buffers, and samplers." The spec's example struct contains a `device float4* e;` member and is taken as `kernel void my_kernel(constant Foo & f [[buffer(0)]])`. So a `CSRMatrix { device const uint* rowPtrs; ... }` is expressible. Not checked: the host-side encoding of such a buffer. The course convention (one buffer binding per array, sizes in a small constant struct) needs no argument buffer and is what the labs use.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 2.13, page 54)

### C3. Coalescing, alignment and divergence for ELL versus CSR on Apple GPUs

**Verdict: COULD NOT VERIFY; none is documented.** ch06 C1 (no coalescing rule, no line or sector size), ch04 C4 (no divergence model), infra C13 (no cache figures) all still stand. The SIMD-group width comes from `threadExecutionWidth` or `[[threads_per_simdgroup]]` (ch05 C3). So the explorer's coalescing and divergence counts are labelled as NVIDIA's (and AMD's) documented model, and the Metal half of the lab is a measurement with a prediction whose README says no Apple document predicts the result. One shipped Metal kernel does act on a divergence cost, as a code comment rather than documentation (D6).

### C4. Apple's sparse libraries: is there a GPU SpMV to compare against?

**Verdict: PARTLY.** Apple documents a way to build sparse tensors on the GPU graph, and a CPU sparse library; I found no documented GPU sparse matrix-vector routine.

- MPSGraph, `MPSGraphSparseOps.h` (macOS 27.0 SDK): `typedef NS_ENUM(uint64_t, MPSGraphSparseStorageType) { MPSGraphSparseStorageCOO = 0, MPSGraphSparseStorageCSC = 1L, MPSGraphSparseStorageCSR = 2L }`, each enumerator `macos(12.0)`. `- (MPSGraphTensor *)sparseTensorWithType:(MPSGraphSparseStorageType)sparseStorageType tensors:(NSArray<MPSGraphTensor *> *)inputTensorArray shape:(MPSShape *)shape dataType:(MPSDataType)dataType name:(NSString *_Nullable)name`. Header text: "For COO, indexTensor0 is x index and indexTensor1 is y index. For CSC, indexTensor0 and indexTensor1 correspond to rowIndex and colStarts respectively. For CSR, indexTensor0 and indexTensor1 correspond to colIndex and rowStarts respectively." Neither the header nor Apple's page for the method says which graph operations accept the result; "x index" and "y index" are not tied to row and column. Apple's page for the enum gives macOS 11.0 while the header marks each enumerator macOS 12.0.
- Metal Performance Shaders: no public header in the SDK mentions "sparse". The framework's `.tbd` export list contains `MPSNDArrayMatrixMultiplicationSparse` and `MPSNDArrayMaterializeSparseTensor`, with no header that declares them; treat them as private.
- Accelerate (CPU; Apple's pages make no GPU statement, infra C11): "In the Accelerate framework, the Sparse Solvers library stores sparse matrices using the compressed sparse column (CSC) format." The header adds: "CSR format data can be simulated by using a blockSize of `1` and setting the transpose attribute". Sparse BLAS has `sparse_matrix_vector_product_dense_float(enum CBLAS_TRANSPOSE transa, float alpha, sparse_matrix_float A, const float * __restrict x, sparse_stride incx, float * __restrict y, sparse_stride incy)`, `API_AVAILABLE(macos(10.11), ...)`, with the multiply-and-accumulate semantics quoted in A10.

For the course: the Metal lab has no vendor baseline; Accelerate's routine is a usable CPU reference for correctness. PyTorch on MPS writes its own sparse Metal kernels (D2).

- Local headers: `MetalPerformanceShadersGraph.framework/Headers/MPSGraphSparseOps.h`, `Accelerate.framework/Frameworks/vecLib.framework/Headers/Sparse/Solve.h` (lines 446 to 470), `.../Sparse/BLAS/BLAS.h` (lines 430 to 465), macOS 27.0 SDK
- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/sparsetensor(sparsetensorwithtype:tensors:shape:datatype:name:)
- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphsparsestoragetype
- https://developer.apple.com/documentation/accelerate/creating-sparse-matrices
- https://developer.apple.com/documentation/accelerate/sparse-solvers-library

---

## Group D: How PyTorch, llama.cpp and MLX use this chapter's patterns (the LLM bridge)

The bridge in the lesson ("sparse attention masks and mixture-of-experts routing are where sparse patterns meet LLM inference") is ours. The items below are what today's source shows; mapping a piece of code onto a book format is our reading and is marked so.

### D1. PyTorch: sparse layouts, and SpMV on CUDA and ROCm

**Verdict: CONFIRMED.**

- Docs (2.15, sparse page): "The PyTorch API of sparse tensors is in beta and may change in the near future." "PyTorch currently supports COO, CSR, CSC, BSR, and BSC. We also have a prototype implementation to support semi-structured sparsity." And, worth quoting next to the book's warning (p. 422): "Like many other performance optimization sparse storage formats are not always advantageous. When trying sparse formats for your use case you might find your execution time to increase rather than decrease."
- COO is append-friendly, the book's flexibility point (p. 406): "addition of sparse COO tensors is implemented by simply concatenating the indices and values tensors". `coalesce()` returns a tensor whose indices "are unique" and "sorted in lexicographical order" (A7).
- CUDA SpMV goes through cuSPARSE. `aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp` (last commit 2026-11-20), `addmv_out_sparse_csr` (line 1031): calls `cusparseSpMV_bufferSize` then `cusparseSpMV` with a CSR descriptor and comments "// TODO: update this to support COO sparse layout". Algorithm choice (lines 1051 to 1063): "cuSPARSE documents bit-wise reproducible SpMV results only for CUSPARSE_SPMV_CSR_ALG2 (and COO_ALG2, BSR_ALG1) with opA == NON_TRANSPOSE." so `deterministicAlgorithms()` selects `CUSPARSE_SPMV_CSR_ALG2`, otherwise `CUSPARSE_SPMV_ALG_DEFAULT`. On ROCm it always passes `CUSPARSE_MV_ALG_DEFAULT` with the comment "hipSPARSE does not document a deterministic SpMV algorithm.", and when deterministic algorithms are requested it calls `alertNotDeterministic("addmv_out_sparse_compressed_cuda")` (compare B5).
- Pruned LLM weights: the semi-structured (2:4) layout "was first introduced in NVIDIA's Ampere architecture" and "is implemented via a Tensor subclass"; the CUDA directory has `cuSPARSELtOps.cpp` and `SparseSemiStructured*.cu` (by name). Prototype status, as quoted above.

- https://docs.pytorch.org/docs/2.15/sparse.html
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp (lines 1031 to 1110)
- https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/sparse/cuda

### D2. PyTorch on MPS: its own sparse Metal kernels, row ranges by binary search instead of atomics

**Verdict: CONFIRMED.** `aten/src/ATen/native/sparse/mps/` has `SparseMPSTensor.mm`, `SparseMPSTensorMath.mm` (last commit 2026-08-17) and `kernels/SparseTensorMath.metal` (last commit 2025-12-16).

- `s_addmm_out_sparse_dense_mps` coalesces the COO input first (`auto sparse = sparse_.coalesce();`), then dispatches `"spmm_addmm_coo_" + type` with `dispatchThreads`, one thread per output element, threadgroup width `pso.threadExecutionWidth`.
- The kernel (`SparseTensorMath.metal`, lines 244 to 282) finds its row's slice of the sorted COO arrays by binary search: `const uint start = lower_bound_i64(rows, 0u, nnz, (long)i); const uint end = upper_bound_i64(rows, 0u, nnz, (long)i);` then loops `for (uint p = start; p < end; ++p)` and writes `out[off]` once, with no atomics.

Our reading: this is the book's accessibility argument (pp. 406 to 409) solved a third way. Sorted COO plus a search recovers CSR's row access (row to its non-zeros) without building `rowPtrs`, at the cost of a logarithmic search per thread.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/sparse/mps/SparseMPSTensorMath.mm (lines 56 to 148)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/sparse/mps/kernels/SparseTensorMath.metal (lines 244 to 282)

### D3. PyTorch FlexAttention: the sparse attention mask is an ELL-shaped block format

**Verdict: CONFIRMED.** `torch/nn/attention/flex_attention.py` (last commit 2026-11-20), `class BlockMask` (line 911), docstring:

- "BlockMask is our format for representing a block-sparse attention mask. It is somewhat of a cross in-between BCSR and a non-sparse format."
- "num_blocks_in_row: Tensor[ROWS]: Describes the number of blocks present in each row. col_indices: Tensor[ROWS, MAX_BLOCKS_IN_COL]: `col_indices[i]` is the sequence of block positions for row i. The values of this row after `col_indices[i][num_blocks_in_row[i]]` are undefined."
- "This format is primarily optimized for 1. simplicity, and 2. kernel efficiency. Notably, it is *not* optimized for size".
- "(kv_num_blocks, kv_indices): Used for the forwards pass of attention, as we reduce along the KV dimension." "(q_num_blocks, q_indices): Required for the backwards pass, as computing dKV requires iterating along the mask along the Q dimension. These are autogenerated from 1."
- "(full_kv_num_blocks, full_kv_indices): This is optional and purely an optimization. As it turns out, applying masking to every block is quite expensive! If we specifically know which blocks are "full" and don't require masking at all, then we can skip applying mask_mod to these blocks. ... For causal masks, this is about a 15% speedup."

Our mapping onto the chapter: `num_blocks_in_row` plus a padded `col_indices[ROWS, MAX]` is Fig. 17.12's `nnzPerRow` plus padded ELL columns, stored row-major because one program walks one row; the generated `q_indices` copy is the book's "keep a column-oriented copy" for column access (p. 409); the full/partial split is a two-format scheme like §17.5's hybrid. The 15% figure is PyTorch's, for its kernels, and is not for lessons as a device number.

- https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py (lines 911 to 985)

### D4. llama.cpp, CUDA and HIP: sparse flash attention builds -1-padded index lists with counts

**Verdict: CONFIRMED.** `ggml/src/ggml-cuda/fattn.cu` (last commit 2026-11-19):

- `flash_attn_mask_to_sparse_indices` (line 13), comment on line 10: "one list per group of ncols1 queries: a column is selected if any query of the group can see it". It reads the dense mask, keeps the columns whose mask value `isfinite`, and compacts them into `indices` with `ggml_cuda_ballot` and `ggml_cuda_popc` (the ch12 pattern; `ggml_cuda_thread_mask_t` is 64-bit on HIP, ch05 B9). Each list has fixed width `n_kv_max`; the tail is filled with `indices[i] = -1;` (line 96) and the true length goes to `counts_ptr[...] = count;`.
- Used by the MMA flash-attention kernel when `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse` allows it, on NVIDIA with `turing_mma_available(cc)` or on AMD with `amd_mfma_available(cc)`; "On CDNA only the DKQ=576 kernel variant with ncols2=16 is compiled with sparse support."
- The dense path has a lighter version of the same idea: `flash_attn_mask_to_KV_max` in `fattn-common.cuh` (last commit 2026-11-18, line 722) finds, per query tile, the last KV tile that is not entirely `-inf`, so later tiles are skipped ("Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.").

Our mapping: fixed-width lists padded with -1 plus a per-list count is ELL with `nnzPerRow` and the library pad convention of A6 and B3; `KV_max` is a per-row length, CSR's information without the indices.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu (lines 1 to 160)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-common.cuh (lines 722 to 775, 1150 to 1190)

### D5. llama.cpp, CUDA and HIP: mixture-of-experts routing is a COO-to-CSR conversion, with padding and an inverse permutation

**Verdict: CONFIRMED.** The operation is `GGML_OP_MUL_MAT_ID` (each token row multiplied by the weight matrices of the experts its router chose).

- `ggml/src/ggml-cuda/mmid.cu` (last commit 2026-11-19), comment at line 68: "Helper function for mul_mat_id, converts ids to a more convenient format. ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert." The kernel `mm_ids_helper` runs one block per expert (`const int expert = blockIdx.x; // The expert this CUDA block handles.`), counts the entries of lower-numbered experts (`nex_prev`, "Number of columns for experts with a lower index"), and compacts this expert's (token, slot) pairs with ballot and popcount.
- Caller, `mmq.cu` (last commit 2026-11-18), line 323: `ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows + J_best-1); // Needs to be padded for unconditional memory access.`
- Fallback in `ggml-cuda.cu` (last commit 2026-11-20), `ggml_cuda_mul_mat_id` (line 2638): copies the router ids to the host, fills `std::vector<int32_t> tokens_per_expert(ne02);`, builds `ids_to_sorted_host` (rows grouped by expert) and `ids_from_sorted_host` (the inverse), gathers rows with `get_rows_cuda`, runs one matmul per expert with `tokens_per_expert[i02] != 0`, and scatters back through the inverse. Its own note: "this path should not be reached when recording CUDA graphs, because it requires stream synchronization".

Our mapping: the router output is a COO list of (token, expert) pairs; grouping it by expert with counts and offsets is exercise 3 (histogram, then prefix sum, then scatter); `ids_from_sorted` plays the role of JDS's `row` array that puts results back in the original order (p. 417); padding `ids_dst` so every thread can store unconditionally is the chapter's regularity-for-bytes trade.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmid.cu (lines 1 to 130)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cu (lines 314 to 340)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu (lines 2638 to 2760)

### D6. llama.cpp, Metal: the same routing as count, prefix sum, padded slices, and a stated divergence cost

**Verdict: CONFIRMED.** `ggml/src/ggml-metal/kernels/mul_mm.metal` (last commit 2026-11-05), `kernel_mul_mm_id_map0` (line 368), one thread per expert (`const short ide = tpitg; // expert id`):

- "pass 1: count the entries of this expert"
- Line 421 onward: "exclusive prefix sum over the counts: the entries of this expert start at bas[ide] - every slice is padded by one entry for the uniform dummy store of pass 2", then tiles of 32 entries are emitted per expert.
- Line 486 onward: "one store per token for every thread, regardless of the match - conditional stores of the divergent lanes are very slow - the dummy value of the tokens without a match lands on the padding slot of this slice and is never read".

That comment is llama.cpp's observation about Apple GPUs, not Apple documentation (C3); it is the only statement about a divergence cost on Metal found today, and it is a measured engineering choice in shipping code. Our mapping: count plus exclusive scan is `rowPtrs` construction; one padding slot per slice buys a branch-free store, the chapter's lesson at the scale of one element.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mm.metal (lines 366 to 510)

### D7. MLX and mlx-lm: no sparse arrays; mixture-of-experts sorts tokens by expert and unsorts afterwards

**Verdict: CONFIRMED.**

- The MLX 0.32.4 operations list has no sparse array type or sparse layout (searched for "sparse", "csr", "coo": only `mlx.nn.init.sparse`, an initializer, and the `sparse` flag of `meshgrid`). What MLX offers instead are gathered and masked matmuls: `gather_mm(a, b, /, lhs_indices=None, rhs_indices=None, *, sorted_indices=False, stream=None)`, "Matrix multiplication with matrix-level gather", where "If only one index is passed and it is sorted, the sorted_indices flag can be passed for a possible faster implementation."; `gather_qmm` for quantized weights; `block_masked_mm(a, b, /, block_size=64, mask_out=None, mask_lhs=None, mask_rhs=None, ...)`, "Note: Only block_size=64 and block_size=32 are currently supported"; `segmented_mm(a, b, /, segments, ...)`, whose `segments` are "The offsets into the inner dimension for each segment."
- mlx-lm `mlx_lm/layers/moe.py` (last commit 2026-11-19; the older `mlx_lm/models/switch_layers.py` path is gone after "New models layout (#2082)"): `_gather_sort` does `order = mx.argsort(indices)` and `inv_order = mx.argsort(order)`; `SwitchGLU.__call__` comments "When we have many tokens, then sort them to make sure that the access of different experts is in order.", sets `do_sort = indices.size >= 64`, passes `sorted_indices=do_sort` to `gather_qmm`/`gather_mm`, and restores order with `_scatter_unsort(x, inv_order, indices.shape)`.
- MLX Metal backend, `mlx/backend/metal/matmul.cpp` (last commit 2026-10-09), line 2961: "We are walking a in order and b is also in order so we can batch up the matmuls and reuse reading a and b." That branch (`M == 1 && right_sorted_ == true`) dispatches `steel_gather_mm_rhs_n`.

Our mapping: sorting the routed tokens by expert and keeping the inverse permutation is JDS's sort plus `row` array (pp. 416 to 417); the threshold of 64 entries is mlx-lm's own amortization choice (A8).

- https://ml-explore.github.io/mlx/build/html/python/ops.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.block_masked_mm.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.segmented_mm.html
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/layers/moe.py (lines 18 to 31, 197 to 220)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/matmul.cpp (lines 2955 to 2970)

---

## Group E: our arithmetic

### E1. The notes' example arrays and worked numbers

**Verdict: CONFIRMED** (a script rebuilt every array from the matrices alone; not a vendor statement).

- Fig. 17.1 matrix: COO `rowIdx = [0,0,1,1,1,2,2,3]`, `colIdx = [0,1,0,2,3,1,2,3]`, `value = [1,7,5,3,9,2,8,6]` (Fig. 17.3); CSR `rowPtrs = [0,2,5,7,8]` (Fig. 17.7); ELL column-major `colIdx = [0,0,1,3, 1,2,2,*, *,3,*,*]`, `value = [1,5,2,6, 7,3,8,*, *,9,*,*]`, `nnzPerRow = [2,3,2,1]` (Fig. 17.10); CSC `colPtrs = [0,2,4,6,8]`, `rowIdx = [0,1,0,2,1,2,1,3]`, `value = [1,5,7,2,3,8,9,6]` (Fig. 17.16). All match the notes.
- Fig. 17.14 letter matrix, stable sort longest first: `row = [1,3,5,2,4,0]`, `iterPtr = [0,6,11,14,15]`, `colIdx = [0,0,0,2,1,0, 1,3,2,4,4, 3,5,5, 4]`, `value = [b,h,m,f,k,a, c,i,n,g,l, d,j,o, e]`. Matches.
- Exercise 1 sketches in the notes: all four formats match (JDS with a stable longest-first sort: `row = [0,2,3,1]`, `iterPtr = [0,4,7]`).
- Fig. 17.13: row lengths 1,5,2,2,1,2,4,1 (18 non-zeros). Width 5: 40 slots, 22 pads. Width 2: 16 slots, 3 pads, 5 entries in COO (3 from row 1, 2 from row 6). Matches the notes and confirms "row 1", not "row 2" (notes, "Unclear" item 2).
- pp. 412 to 413, counting stored numbers (values plus indices): CSR 2 × 10,000 + 1,001 = 21,001, which is 2.1% of 1,000,000; ELL padded to 200 is 2 × 1,000 × 200 = 400,000, 40.0%; ratio 19.05, which the book rounds to "20×". Adding `nnzPerRow` (1,000 numbers) to ELL changes nothing visible.
- Alignment (A3): an ELL iteration starts at element `t * numRows`; 32-byte alignment of every start needs `numRows` to be a multiple of 8 for 4-byte values, 16 or 32 for AMD's 64- or 128-byte lines.

---

## Summary of corrections

Every WRONG, PARTLY and COULD NOT VERIFY item, one line each:

- A3 (PARTLY): the coalescing unit on every device current CUDA supports (compute capability 7.5 and later) is a 32-byte sector, so ELL's alignment trick is "pad `numRows` to a multiple of 8 floats"; the book's "64-bytes" for older generations appears in no NVIDIA document read today.
- A5 (WRONG): cuSPARSE has no ELL, no hybrid and no JDS. Its hybrid format (`cusparseHybMat_t`, `cusparse<t>hybmv`, the `*2hyb` conversions) was deprecated in CUDA 10.2 and removed in CUDA 11.0; the ELL family survives as Sliced ELL (SpMV) and Blocked-ELL (SpMM). AMD's rocSPARSE still has ELL and HYB.
- A7 (PARTLY): the book's COO kernel works unsorted, but cuSPARSE assumes COO "sorted by row", and PyTorch's coalesced COO is sorted; within-row order is free in both.
- A10 (PARTLY): the standard's routine is Sparse BLAS `usmv` (C prototype: `y` in-out, no beta); "SpMV" is the library name, and the libraries compute y = α op(A) x + β y, so the book's accumulate is β = 1. The ELL listing's `y[row] = sum` should accumulate like the CSR one.
- A12 (COULD NOT VERIFY): "tens to hundreds of non-zeros per row" and "sparse FLOPS much lower than dense" are the book's generalisations; no vendor page read today states them (Apple's one example matrix averages 25 non-zeros per column). The lab measures.
- B5 (PARTLY): hipSPARSE keeps cuSPARSE's names, but documents no determinism; on AMD `CSR_ALG1` is rocSPARSE's adaptive (not deterministic), `CSR_ALG2` is rowsplit (deterministic per rocSPARSE), and both `COO_ALG1` and `COO_ALG2` are the deterministic segmented COO, not the atomic one. PyTorch flags SpMV on ROCm as non-deterministic for that reason.
- C3 (COULD NOT VERIFY): Apple documents no coalescing unit, alignment rule or divergence model; the Metal half of the lab is a measurement.
- C4 (PARTLY): Apple has no documented GPU SpMV. MPSGraph can create COO, CSC and CSR tensors (macOS 12) but documents no operation that consumes them; MPS exports sparse class names with no public header; Accelerate's sparse routines (CSC solvers, Sparse BLAS SpMV with y = α op(A) x + y) are CPU.

Changes and facts worth knowing even where the verdict is CONFIRMED:

- A2: both vendor libraries ship the book's atomics trade as a choice of algorithm (fast and non-deterministic, or deterministic and slower). cuSPARSE's general reproducibility section and its SpMV table disagree; the table is the specific statement.
- A6: NVIDIA's Sliced ELL is the book's sectioned variant without the sort, column-major per slice. NVIDIA and AMD both pad column indices with -1 and test them, which answers the notes' question about what a pad slot holds.
- A8: cuSPARSE (`cusparseSpMV_preprocess`) and rocSPARSE (a preprocess stage; adaptive CSR "needs to be amortized over many subsequent sparse vector products") build the book's amortization argument into their APIs.
- A9: `cub::DeviceSpmv` was removed in CCCL 3.0 ("Use cuSPARSE instead"); COO to CSR is one library call (`cusparseXcoo2csr`, `rocsparse_coo2csr`) or the book's histogram plus scan with CUB/rocPRIM.
- B4: rocSPARSE's LRB algorithm bins rows into 32 power-of-two length classes, the book's sort-by-length idea as bucketing (read from source; the acronym is not expanded in what I read).
- D3 to D7: the LLM bridge has concrete code behind it. PyTorch's FlexAttention `BlockMask` is per-row counts plus padded indices with a generated transposed copy; llama.cpp's sparse flash attention builds -1-padded index lists plus counts; MoE routing in llama.cpp (CUDA, HIP, Metal) is count, prefix sum, padded scatter and an inverse permutation; mlx-lm sorts tokens by expert (at 64 or more entries) and unsorts afterwards.
- D6: the only statement found today about a divergence cost on Apple GPUs is a llama.cpp code comment ("conditional stores of the divergent lanes are very slow"); it is a project's observation, not Apple documentation.

## Where the backends really differ

One line per cell. Nothing here has been run on CUDA or AMD hardware by us; the Metal column is from the specification and SDK headers.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1, macOS 27.0 SDK) |
|---|---|---|---|
| One thread per row (CSR, ELL, JDS kernels) | `blockIdx.x*blockDim.x + threadIdx.x`, bounds check | Same source | `[[thread_position_in_grid]]`; no bounds check with `dispatchThreads` |
| Float atomic add for COO and CSC | `cuda::atomic_ref<float, cuda::thread_scope_device>::fetch_add(v, relaxed)` or legacy `atomicAdd` | Same spelling through libhipcxx (10.1 and later), or `atomicAdd`, safe CAS unless `-munsafe-fp-atomics` | `atomic_fetch_add_explicit` on `device atomic_float*`, relaxed only, Metal 3 and Apple7, device memory |
| Order of atomic float adds | Undefined, so not bit-reproducible | Undefined | Not documented |
| Coalescing unit | 32-byte sectors (compute capability 6.0 and later) | 64- or 128-byte cache lines | Not documented |
| Width that one ELL iteration covers | 32 lanes, 128 bytes of floats | 64 lanes on CDNA (256 bytes), 32 on RDNA | `threadExecutionWidth`, no documented value |
| ELL column alignment (our arithmetic) | `numRows` multiple of 8 floats | Multiple of 16 or 32 floats | Unknown; measure |
| Divergence model for CSR | Documented: masked passes per warp | Documented: EXEC mask per wavefront | Not documented |
| Struct of device pointers as a kernel argument | By value, all parameters at most 32,764 bytes | Same syntax; no size limit found | Argument buffer (`constant Foo&` with `device T*` members), Metal 2 and later |
| Vendor sparse library | cuSPARSE 13.5 (GPU) | rocSPARSE 5.1.0, hipSPARSE 4.8.0 as the portable layer | No GPU SpMV documented; Accelerate sparse on the CPU |
| Formats in the library | COO, CSR, CSC, Sliced ELL, BSR, Blocked-ELL | COO (also AoS), CSR, CSC, ELL, HYB, Sliced ELL, BSR, Blocked ELL | MPSGraph sparse tensors: COO, CSC, CSR (creation only); Accelerate: CSC |
| Hybrid ELL-COO | Removed in CUDA 11.0 | `rocsparse_hybmv`, partition auto, user or max | None |
| SpMV call and formula | `cusparseSpMV`: y = α op(A) x + β y | `rocsparse_spmv` (staged) or `hipsparseSpMV`: same formula | `sparse_matrix_vector_product_dense_float` (CPU): y = α op(A) x + y |
| Deterministic SpMV | `CSR_ALG2`, `COO_ALG2`, `SELL_ALG1`, `BSR_ALG1` | rocSPARSE rowsplit, coo, ell, bsr, sell; hipSPARSE docs silent | Not applicable |
| Pad marker in ELL-family arrays | -1 (Sliced ELL) | -1 (ELL) | Not applicable |
| Grouping rows by length inside the library | None documented | LRB: 32 power-of-two bins (from source) | Not applicable |
| Reusing analysis across many SpMVs | `cusparseSpMV_preprocess` | `rocsparse_spmv_stage_preprocess` | Not applicable |
| CUB-level SpMV | `cub::DeviceSpmv` removed in CCCL 3.0 | Not checked for hipCUB | None |
