# Chapter 15 fact check: CUDA, HIP, Metal

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims the verdict is for the item as a whole and the text says which sub-claim failed or could not be found. A verdict judges the book's statement, or the assumption a course author would most likely bring (stated in the item title), not the vendor.
- **Quote** is copied from the source (HTML flattened to text, GitHub raw files, Apple's documentation in Markdown form, PDF text, local SDK headers). Line breaks inside quotes are removed.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide "v13.5" (async-copies page last updated Oct 31, 2026), CUDA Toolkit 13.5 release notes (Nov 05, 2026), NVIDIA CUDA Compiler Driver 13.5 (Nov 02, 2026), cuBLAS 13.5 (Nov 02, 2026), PTX ISA 9.5, Nsight Compute v2026.4.0 (Compute Triage Guide last updated Sep 22, 2026), CCCL docs at `nvidia.github.io/cccl/unstable/`, CUTLASS docs `latest`, cuDNN docs "Latest", CMake 4.5.1 docs; ROCm Core SDK 10.2.0 (HIP docs "HIP 7.17.0", Composable Kernel docs "1.3.0", hipBLAS docs "3.8.0"), sources at tag `therock-10.2` in `ROCm/rocm-systems`, `ROCm/rocm-libraries` and `ROCm/llvm-project`; upstream `llvm/llvm-project` on `main`; Metal Shading Language Specification "Version 4.1" (2026-06-04, re-downloaded today); macOS SDK 27.0 headers and the Metal toolchain 27.1 (`metal/32023`) installed on this machine. Repositories read on their default branches today; the last commit date of a file is given where it matters.
- Not checked: anything by running code on a GPU. Nothing here was compiled for CUDA, ROCm or Metal. Group E is our own arithmetic, checked with a short script, not a vendor statement.

Already settled elsewhere, referred to by item and not repeated:

- `ch05-facts.md` A5 (32-thread warps, 1024 threads per block), A8 (H100 SXM: 67 TFLOPS FP32, 3.35 TB/s, 228 KB shared memory per SM; "66.9" printed by no NVIDIA source), A9 (cuBLAS and CUTLASS names, URLs, versions), B4 (`__shared__`, `__syncthreads()`, index built-ins on HIP), B5 (wavefront 64 on gfx9, 32 on gfx10 and later), C1 (threadgroup memory, 32 KB on Apple4 and later), C2 (`threadgroup_barrier`), C3 (SIMD-group width only from the pipeline).
- `ch03-facts.md` A7 and B6 (cuBLAS and rocBLAS are column-major), C7 (`MPSMatrix` is row-major), C9 (Apple's "Creating threads and threadgroups" page calls SIMD groups "warps or wavefronts"), D2 (how llama.cpp hands row-major data to `cublasSgemm`).
- `ch04-facts.md` A12 (per-SM limits by compute capability: 255 registers per thread and 64 K registers per SM on every listed capability; 2048 resident threads only on 8.0, 9.0, 10.0, 10.3), A13 and A14 (occupancy definition, `--resource-usage`, `cudaFuncGetAttributes(...).numRegs`, `__launch_bounds__`, `--maxrregcount`, occupancy API), B10 (AMD occupancy: wave slots, VGPR and SGPR pools, granules, "MFMA-based GEMMs on CDNA often operate at low occupancy", `__launch_bounds__` second argument means waves per execution unit), C5 (Metal documents no register pool and no occupancy API).
- `ch06-facts.md` A2 (32-byte sectors), A11 (`float4` cast idiom and alignment), A12 (same-32-bit-word accesses do not conflict; bank-conflict metrics), A13 (local memory), A16 (double buffering; LDGSTS on CC 8.0+, per-thread wait rule), B1 and B4 (AMD coalescing; AMD's own `[32][33]` padding example), B8 (no `cuda::pipeline` on HIP; this file corrects its intrinsic name in B5), C2 to C5 and C7 (Metal: no banks, `float4` buffers, no documented unroll control, occupancy in Xcode only, no async copies), D1 to D3 (llama.cpp padding and `cp.async`, MLX padding and single buffering).
- `ch08-facts.md` A4 and B4 (register ceilings, `__launch_bounds__` on CUDA and HIP; HIP compiles for a 1024-thread block unless told otherwise), B3 (AMD LDS sizes, no 48 KB opt-in step), C1 and C5 (Metal may lower a pipeline's thread cap; `[[max_total_threads_per_threadgroup]]`).
- `infra-facts.md` A6, A7, A9, A10, A12, B6, B7, B9, B10, B12, C7, C10, C11, C13, D1 to D4 (vector types, unroll and inline spellings, occupancy API, banks, WMMA and CUDA Tile, rocWMMA and hipBLAS, `simdgroup_matrix` and MPP, MPS, Apple's missing bank and cache facts, backend layouts).
- `ch24-facts.md` A19, B1, C1, D4 (how kernels reach tensor cores; AMD MFMA; Metal's matrix interfaces; PyTorch's `allow_tf32` default).

Totals: 18 CONFIRMED, 0 WRONG, 10 PARTLY, 2 COULD NOT VERIFY (30 items: A1 to A12, B1 to B7, C1 to C6, D1 to D4, E1). Every PARTLY and COULD NOT VERIFY item is listed under "Summary of corrections".

---

## Group A: CUDA, and the book's dated or device-specific statements

### A1. H100 as the reference device: 8 FLOP/B memory-bound, 32 FLOP/B compute-bound (p. 351)

**Verdict: CONFIRMED** under the book's model, with the figures of `ch05-facts.md` A8. NVIDIA's H100 SXM product figures give 67 TFLOPS / 3.35 TB/s = 20.0 FLOP/B, so 8 is below and 32 above the threshold. The chapter itself does not restate the threshold (notes, §15.2); it comes from Chapter 5.

For the lesson: present 20 FLOP/B as "the book's example device", and draw the reader's own threshold from `useReports()`. The model counts only the bytes of tile loads; caches are not in it (Group E).

- https://www.nvidia.com/en-us/data-center/h100/ (via `ch05-facts.md` A8)

### A2. 32 banks of 4 bytes, 32 threads per warp, bank = word index mod 32 (pp. 360 to 362)

**Verdict: CONFIRMED.** `infra-facts.md` A10 (32 banks, each 32 bits, every listed compute capability), `ch05-facts.md` A5 (warp size 32), `ch06-facts.md` A12 (threads reading the same 32-bit word do not conflict, which is why the four threads of one thread-row in Fig. 15.11 cost nothing extra).

Searched today and not found: any rule in the current Programming Guide for how 64-bit or 128-bit shared-memory accesses map to banks. The guide's bank section speaks only of 32-bit requests: "Any 32-bit memory read or write request made of 32 addresses that fall in 32 distinct memory banks can therefore be serviced simultaneously". A lab that reads strips from shared memory as `float4` must measure its conflicts (`l1tex__data_bank_conflicts_pipe_lsu.sum`, `ch06-facts.md` A12) rather than compute them from the 32-bit rule.

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.4.2)

### A3. "255 registers per thread", "64k registers" per SM, 256 threads per SM, 12.5 % of 2048 (p. 363)

**Verdict: PARTLY.** 255 and 64 K hold on every compute capability NVIDIA lists (`ch04-facts.md` A12). 2048 resident threads per SM holds only on 8.0, 9.0, 10.0 and 10.3; on 8.6, 8.9 and 12.x (RTX 30, 40, 50) it is 1536, and on 7.5 and 10.7 it is 1024. And whether 256 threads at 255 registers fit is decided by an allocation granularity NVIDIA does not publish.

- Nsight Compute Profiling Guide: "The most important resource under the compiler's control is the number of registers used by a kernel. Each sub partition has a set of 32-bit registers, which are allocated by the HW in fixed-size chunks." The chunk size is not given.
- Metrics to read instead of computing: `launch__registers_per_thread` ("Number of registers allocated per thread."), `launch__occupancy_limit_registers` ("Occupancy limit due to register usage."), `launch__occupancy_per_register_count` ("Number of active warps for given register count.").
- NVIDIA agrees that low occupancy can be right for this kind of kernel. Compute Triage Guide: "Low occupancy + high SM pipeline utilization: Launching extra warps is counter-productive because the busy pipe cannot absorb more work." CUTLASS docs: "The accumulator elements typically occupy at least half a thread's total register budget. Consequently, occupancy – the number of concurrent threads, warps, and threadblocks – is relatively low compared to other classes of GPU workloads." (dashes as printed)

Our arithmetic (Group E): one 256-thread block per SM is 12.5 % of 2048, 16.7 % of 1536, 25 % of 1024. The register budget calculator must take resident threads and registers per SM from the device report, offer 2048 / 65,536 / 255 as "the book's example", and say that the exact block count comes from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (`ch04-facts.md` A14), not from the per-thread product.

- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html (section "Register File" and the launch metrics table)
- https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html (section "Occupancy-vs-Utilization")
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html (section "Pipelining")

### A4. Keeping `C_r`, `a_r`, `b_r` in registers: full unrolling, `__forceinline__`, constant indices (pp. 353 to 359)

**Verdict: CONFIRMED.** Spellings and semantics: `infra-facts.md` A7 (`#pragma unroll` with no argument unrolls completely when the trip count is constant; `__forceinline__` "Forces nvcc to inline the function within a single translation unit"). Where a non-constant-indexed array goes: `ch06-facts.md` A13 ("local memory", physically device memory). How to see registers and spills: `--resource-usage`, `--ptxas-options=-v`, `cudaFuncGetAttributes(...).numRegs` (`ch04-facts.md` A13, `ch06-facts.md` A13). Nothing new was needed.

### A5. Tensor cores "beginning with the Volta architecture" (p. 368)

**Verdict: PARTLY.** True as history, but the current toolkit no longer targets Volta.

- cuBLAS 13.5: "Tensor cores were first introduced with Volta GPUs (compute capability 7.0 and above) and significantly accelerate matrix multiplications."
- CUDA 13.0 library release notes, still printed in the 13.5 notes: cuFFT "Removed support for Maxwell, Pascal, and Volta GPUs, corresponding to compute capabilities earlier than Turing." and cuSPARSE "Dropped support for pre-Turing architectures (Maxwell, Volta, and Pascal)."
- nvcc 13.5 lists real architectures from `sm_75` up (`sm_75`, `sm_80`, `sm_86`, `sm_87`, `sm_88`, `sm_89`, `sm_90`, `sm_90a`, `sm_100` ... `sm_121f`); `sm_70` does not appear, and "sm_75 is used as the default value". The guide's capability tables also start at 7.5 (`ch04-facts.md` A12).

For the lesson: "tensor cores date from Volta; today's CUDA 13 toolkit starts at Turing (compute capability 7.5)". I did not find a single toolkit-wide sentence in the 13.5 notes that says the compiler dropped `sm_70`; the evidence is the nvcc architecture list and the library notes.

- https://docs.nvidia.com/cuda/cublas/index.html (section 2.1.11 "Tensor Core Usage")
- https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html (sections 3.2.8 and 3.4.10)
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html (sections 4.2.7 and 4.2.8)

### A6. WMMA "before Hopper", WGMMA on Hopper (operands from shared memory, asynchronous), TMEM on Blackwell (p. 369)

**Verdict: PARTLY.** WMMA and WGMMA are confirmed as described. Two things the book does not say: WGMMA is Hopper-only (code built for it does not run on Blackwell), and TMEM belongs to the data-centre Blackwell families, not to the GeForce Blackwell family (compute capability 12.x).

WMMA: still documented for compute capability 7.0 and up, warp-wide, `nvcuda::wmma` in `<mma.h>` (`infra-facts.md` A12). It is not "pre-Hopper only"; it remains available. What llama.cpp actually uses is the PTX warp-level `mma` instruction, not WMMA (D1). PTX ISA 9.5, `mma`: "Requires sm_70 or higher." ... ".f16 floating point type mma operation with .m16n8k16 shape requires sm_80 or higher." ... ".tf32 alternate floating point type mma operation with .m16n8k4 and .m16n8k8 shapes requires sm_80 or higher."

WGMMA, PTX ISA section 9.7.17:

- "A warpgroup is a set of four contiguous warps such that the warp-rank of the first warp is a multiple of 4."
- "The input matrix A of the warpgroup wide MMA operations can be either in registers or in the shared memory. The input matrix B of the warpgroup wide MMA operations must be in the shared memory."
- Issue, commit and wait are separate instructions (`wgmma.mma_async`, `wgmma.commit_group`, `wgmma.wait_group`): "Issue the asynchronous matrix multiply and accumulate operations using the wgmma.mma_async operation".
- Target: "Requires sm_90a." And what an `a` target means: "Target architectures with suffix "a", such as sm_90a, include architecture-specific features that are supported on the specified architecture only, hence such targets do not follow the onion layer model. Therefore, PTX code generated for such targets cannot be run on later generation devices."

TMEM, PTX ISA section 9.7.18:

- "The 5th generation TensorCore has dedicated on-chip memory that is specialized for use by TensorCore operations. This Tensor Memory is organized as a two-dimensional matrix where the horizontal rows are called lanes and the vertical columns are called columns." (The spec gives 512 columns by 128 rows of 32-bit cells per CTA on `sm_100a`/`sm_100f`; a device figure, for this file only.)
- `tcgen05.mma` target notes: "Supported on following architectures: sm_100a sm_101a (Renamed to sm_110a from PTX ISA version 9.0) And is supported on following family-specific architectures from PTX ISA version 8.8 except .kind::i8 : sm_100f or higher in the same family ... sm_110f or higher in the same family". `sm_120` is not in the list. Compute capability 12.x is RTX 50 and RTX PRO Blackwell (`ch04-facts.md` A12).

NVIDIA's own one-paragraph summary of the three generations (Compute Triage Guide, "MMA fundamentals"): "Warp-level MMA (Volta+): runs on a single warp scheduler using all 32 threads of one warp. All inputs and outputs reside in the register file." "Warp-group MMA / WGMMA / GMMA (Hopper+): coordinates all four warp schedulers in the SM. Requires a block of at least 4 warps so one runs on each scheduler. Operand tiles are loaded directly from shared memory (effectively quadrupling the available register file) ... Executes asynchronously: issue, do other work, then wait." "5th-generation MMA (Blackwell): adds a tensor-core controller pipeline shared by the four warp schedulers and a separate on-chip Tensor Memory (TMEM) that holds operands/outputs, removing register-file pressure. A can come from shared memory or TMEM, B from shared memory, D is written to TMEM. Issued by a single thread of a single warp".

For the lesson: the progression is "operands in registers (warp) → operands from shared memory (warp group) → accumulator out of the register file entirely (TMEM)", each step attacking the register pressure of §15.7. Say which compute capabilities have each, from the PTX target notes above.

- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (sections 9.7.16.5.14, 9.7.17, 9.7.17.1, 9.7.18.1, 9.7.18.10.10.1, and the `.target` directive)
- https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html (section "MMA fundamentals")

### A7. LDGSTS on "recent NVIDIA GPUs"; nvcc fuses a vector load plus vector store into LDGSTS; `cuda::memcpy_async` from libcu++ uses it (p. 369)

**Verdict: PARTLY.** LDGSTS is confirmed for compute capability 8.0 and later. `cuda::memcpy_async` can produce it but NVIDIA says it is not guaranteed to. The automatic fusion by nvcc COULD NOT be verified: no NVIDIA page I read says nvcc does it, and NVIDIA's own advice is to convert the code yourself.

- Programming Guide 4.13.1: "The primary goal of LDGSTS (CC 8.0+, see PTX documentation) is to provide an efficient asynchronous data transfer mechanism from global memory to shared memory for smaller, element-wise data transfers". "LDGSTS supports copying 4, 8, or 16 bytes. Copying 4 or 8 bytes always happens in the so called L1 ACCESS mode ... while copying 16-bytes enables the L1 BYPASS mode". "Whether the data transfer occurs asynchronously in practice is up to the hardware implementation and may change in the future."
- On which API produces it: "we use cuda::aligned_size_t<4>() to inform the compiler that the data is aligned to 4 bytes and the size of the data to copy is a multiple of 4 to enable use of LDGSTS. Note that for interoperability with cuda::barrier, cuda::memcpy_async from the cuda/barrier header is used here." And of the `__pipeline_memcpy_async()` primitives: "It also ensures LDGSTS will be used under the hood, which is not guaranteed with the higher-level APIs."
- On what the compiler does with ordinary code, the same page says the opposite of a guaranteed fusion: "With synchronous copies, due to the conditional nature of the code, the compiler may choose to generate a sequence of load-from-global (LDG) store-to-shared (STS) instructions instead of 3 LDGs followed by 3 STSs".
- Compute Triage Guide, "Symptom: Shared-memory fills go through the register file": "The traditional path to shared memory is Device → L2 → L1 → register file → shared memory, which costs two instructions per element and consumes register-file bandwidth. On Ampere and newer, cp.async / LDGSTS moves data directly from global memory through L1 into shared memory in a single instruction, avoiding register-file pressure." Action: "Convert the global-load-then-shared-store sequence to cp.async (or use libraries that emit it, e.g. CUTLASS)." And "If the kernel has a producer-loop that loads then stores into shared memory, but the memory chart still shows traffic on the register-file path, the async-copy path is not actually being exercised."
- PTX `cp.async`: "Requires sm_80 or higher." "cp-size can only be 4, 8 and 16."
- libcu++ is the current name: "libcu++ (libcudacxx) provides fundamental, idiomatic C++ abstractions that aim to make the lives of CUDA C++ developers easier." `cuda::memcpy_async` lives in `<cuda/barrier>` and `<cuda/pipeline>`. The per-thread wait rule (`__syncthreads()` still needed after the wait) is `ch06-facts.md` A16.

For the lesson: drop "nvcc will do it for you"; say the book claims it, that NVIDIA documents the opposite tendency, and show how to check (Nsight Compute memory chart, or look for LDGSTS in the SASS).

- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html (section 4.13.1)
- https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html (section 3.6.3)
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (section 9.7.10.28.3.1)
- https://nvidia.github.io/cccl/unstable/libcudacxx/index.html

### A8. TMA "beginning with the Hopper architecture", usable through `cuda::memcpy_async` (p. 369)

**Verdict: PARTLY.** Hopper (compute capability 9.0) is confirmed. `cuda::memcpy_async` reaches TMA only for one-dimensional bulk copies into the block's own shared memory; multi-dimensional tiles, which is what a GEMM loads, need a tensor map built on the host and a different call.

- Programming Guide 4.13.2: "Asynchronous Copy (TMA, CC 9.0+)". "TMA supports copying both one-dimensional and multi-dimensional arrays (up to 5-dimensional). The programming model for bulk-asynchronous copies of one-dimensional contiguous arrays is different from the programming model for bulk-tensor asynchronous copies of multi-dimensional arrays. To perform a bulk-tensor asynchronous copy of a multi-dimensional array, the hardware requires a tensor map ... A tensor map is typically created on the host using the cuTensorMapEncode API and then transferred from host to device as a const kernel parameter annotated with __grid_constant__".
- Table 23 (one-dimensional): global to `shared::cta` with a shared-memory barrier is exposed by "cuda::memcpy_async, cuda::device::memcpy_async_tx, cuda::ptx::cp_async_bulk"; every other direction only by `cuda::ptx::cp_async_bulk`. Section 4.13.2.2 (multi-dimensional): `cuda::ptx::cp_async_bulk_tensor`, tensor map from `cuTensorMapEncodeTiled`.
- PTX `cp.async.bulk.tensor`: "Requires sm_90 or higher."
- Alignment the triage guide lists: "Strict alignment requirements (16-byte global, 16-byte stride multiple, 128-byte aligned shared, 16-byte transfer size, block dim ≤ 256 elements) can make TMA an obstacle for irregular-pattern kernels (e.g. SpMV) but a natural fit for dense tiled kernels." And on pipelining with it: "A synchronous pattern (load tile → wait → compute → wait → load next tile) leaves async tools idle and underperforms. Issue several stages of TMA loads up front, then in steady state overlap stage-i compute with stage-(i+1) load." That is Fig. 15.14 with more than two buffers.

- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html (sections 4.13.2, 4.13.2.1, 4.13.2.2)
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (section 9.7.10.28.5.3)
- https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html (section 3.6.3)

### A9. cuBLAS, cuDNN, CUTLASS, cuTile; references [1] to [3] (pp. 368, 370)

**Verdict: CONFIRMED.** cuBLAS and CUTLASS: `ch05-facts.md` A9. The three references resolve today: `http://docs.nvidia.com/cuda/cublas` (to `https://docs.nvidia.com/cuda/cublas/`, HTTP 200), `https://docs.nvidia.com/cutlass` (HTTP 200), arXiv 1410.0759 ("cuDNN: Efficient Primitives for Deep Learning").

- cuDNN today: "The NVIDIA CUDA Deep Neural Network library (cuDNN) provides optimized operations for training and inference on NVIDIA GPUs. Use attention, matrix multiplication, convolution, and multi-operation fusion through your framework or the cuDNN Python and C++ APIs." The old path `docs.nvidia.com/deeplearning/cudnn/latest/` redirects to `docs.nvidia.com/cudnn/overview`.
- cuTile: NVIDIA's product name is now "CUDA Tile": "NVIDIA CUDA Tile is a tile-based GPU programming model that targets portability for NVIDIA Tensor Cores." "CUDA Tile is based on the Tile IR specification and tools, including cuTile, which is the user-facing language support for CUDA Tile IR (Intermediate Representation) in Python (and, in the future, C++). The NVIDIA Python implementation of this tile-based programming model is cuTile Python." The same page also lists "CUDA Tile C++", and the Programming Guide says "CUDA Tile C++ is available in the CUDA Toolkit from version 13.3 onward" (`infra-facts.md` A12). The developer page's "in the future, C++" is stale against its own C++ section and the guide.

- https://docs.nvidia.com/cudnn/overview
- https://developer.nvidia.com/cuda/tile
- https://arxiv.org/abs/1410.0759

### A10. cuBLAS as the M3 baseline: header, `cublasSgemm`, layout, and whether FP32 stays FP32

**Verdict: CONFIRMED.** A plain `cublasSgemm` call computes in FP32 by default; TF32 and the BF16x9 emulation are opt-in.

- Header: "The new cuBLAS library API can be used by including the header file cublas_v2.h". CMake target from `FindCUDAToolkit` (CMake 4.5.1): `CUDA::cublas` (also `CUDA::cublas_static`, `CUDA::cublasLt`).
- Declaration as printed: `cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)`. Column-major: `ch03-facts.md` A7.
- Math modes (`cublasMath_t`): `CUBLAS_DEFAULT_MATH` "This is the default and highest-performance mode that uses compute and intermediate storage precisions with at least the same number of mantissa and exponent bits as requested. Tensor Cores will be used whenever possible." `CUBLAS_TF32_TENSOR_OP_MATH` "Enable acceleration of single-precision routines using TF32 tensor cores." `CUBLAS_FP32_EMULATED_BF16X9_MATH` "Enable acceleration of single-precision routines using the BF16x9 algorithm." `CUBLAS_PEDANTIC_MATH` "uses the prescribed precision and standardized arithmetic for all phases of calculations".
- "Setting the environment variable NVIDIA_TF32_OVERRIDE = 0 will override any defaults or programmatic configuration of NVIDIA libraries, and consequently, cuBLAS will not accelerate single-precision computations with TF32 tensor cores."
- BF16x9 (CUDA 12.9 and later) is supported on compute capability "10.0, 10.3" only, and is switched on through the math mode or "To enable floating point emulation without any code changes, the following environment variables can be used." (`CUBLAS_EMULATE_SINGLE_PRECISION`). "it only provides a performance advantage when peak BF16 throughput is more than nine times greater than peak FP32 throughput."

Row-major use (our derivation, the standard trick, same idea as llama.cpp in `ch03-facts.md` D2): a row-major M x N matrix C read column-major is Cᵀ, so `C = A·B` row-major is computed as `Cᵀ = Bᵀ·Aᵀ` with `cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N)`. The lab's CPU reference catches a wrong argument order.

For the lab: leave the handle in the default math mode so the comparison is FP32 against FP32, and print the mode. On compute capability 10.x a reader who sets the emulation variables gets a different algorithm; the README should say so.

- https://docs.nvidia.com/cuda/cublas/index.html (sections 1.5, 2.1, 2.2.10, and `cublas<t>gemm()`)
- https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html

### A11. Names as printed: "NVCC compiler", "libcu++ library", "multidimensional tenors" (p. 369; notes, "Unclear in the PDF" item 11)

**Verdict: CONFIRMED** for the two names; the third is a typo for "tensors". The compiler documentation is titled "NVIDIA CUDA Compiler Driver" and the tool is `nvcc`. libcu++ is the current name, also written libcudacxx, now part of CCCL (A7). Write "nvcc" in lessons, the spelling the reader types.

- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
- https://nvidia.github.io/cccl/unstable/libcudacxx/index.html

### A12. The chapter's techniques in NVIDIA's own GEMM library: hierarchy, outer products, double buffering, warp specialization, and how CUTLASS stores the output tile

**Verdict: CONFIRMED.** CUTLASS's "Efficient GEMM in CUDA" page describes §15.3 to §15.8 nearly point for point, and solves §15.5 a different way.

- Hierarchy: "CUTLASS implements the hierarchically blocked structure described in" the page; threadblock tile `ThreadblockShape::{kM, kN, kK}`, warp-level tile, thread-level tile.
- Register tiling (§15.4): "Threads cannot access each other's registers, so we choose an organization that enables reuse of values held in registers for multiple math instructions. This results in a 2D tiled structure within a thread, in which each thread issues a sequence of independent math instructions to the CUDA cores and computes an accumulated outer product." "SGEMM, IGEMM, HGEMM, and DGEMM are computed by SIMT math instructions issued by thread-level matrix multiply procedures."
- Bank conflicts (§15.6): "For maximum performance, access to shared memory should be bank conflict free."
- Software pipelining (§15.8), at two levels: "CUTLASS uses software pipelining to overlap memory accesses with other computation within a thread. CUTLASS accomplishes this by double buffering at the following scopes. Threadblock-scoped shared memory tiles: two tiles are allocated in shared memory. ... Warp-scoped matrix fragments: two fragments are allocated within registers." The second level (double-buffered register strips) goes beyond the book's Fig. 15.14.
- Warp specialization (p. 368): "Starting with Hopper, CUTLASS 3.0 incorporates the concept of Warp Specialization as part of the kernel design. A thread block is partitioned into two sets of warps, producer warp group and consumer warp group. The producer warp group loads data from global memory into shared memory buffers using the new Tensor Memory Accelerator (TMA)." The page notes "Blackwell SM100 kernels have a substantially different warp-specialization structure".
- Coalesced stores (§15.5), CUTLASS's way: "The mapping of logical elements in the output tile to each thread is chosen to maximize performance of the matrix multiply computation but does not result in efficient, coalesced loads and stores to global memory. The epilogue is a separate phase in which threads exchange data through shared memory then cooperatively access global memory using efficient striped access patterns." The book instead remaps threads so no shared-memory round trip is needed. Both are valid; the lesson can show both (llama.cpp's Metal kernel does the shared-memory version for edge tiles, D2).

- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html (sections "Hierarchical Structure", "Thread-level GEMM", "Epilogue", "Pipelining", "Hopper Warp Specialization")

---

## Group B: HIP / ROCm

### B1. The §15.5 warp layout (8 warps as 2 x 4, 32 threads as 8 x 4) on 64-wide wavefronts; a compile-time warp width

**Verdict: CONFIRMED** (the width facts are `ch05-facts.md` B5; what is new is AMD's documented way to get a compile-time width, which the remapped kernel needs).

- HIP language extensions: "Since warpSize can differ between devices, it can not be assumed to be a compile-time constant on the host." "Prior to ROCm 7.0, the warpSize parameter was a compile-time constant. Starting with ROCm 7.0, it is early folded by the compiler, allowing it to be used in loop bounds and enabling loop unrolling in a manner similar to a compile-time constant warp size."
- "If compile time warp size is required, for example to select the correct mask type or code path at compile time, the recommended approach is to determine the warp size of the GPU on host side and setup the kernel accordingly" (a `template <int WarpSize>` kernel), or: "For users who still require a compile-time constant warp size as a macro on the device side, it can be defined manually based on the target device architecture": `#if defined(__GFX8__) || defined(__GFX9__)` / `#define WarpSize 64` / `#else` / `#define WarpSize 32`. "mwavefrontsize64 compiler option is not supported by HIP runtime".
- llama.cpp uses exactly the macro form (D1).

Consequence for the lab (our derivation, not AMD's): with 256 threads a block is 4 wavefronts of 64 on CDNA, so the book's 2 x 4 grid of 64 x 32 warp tiles becomes, for example, 2 x 2 wavefront tiles of 64 x 64, each wavefront arranged 8 x 8 with four 32 x 32 quadrants and the same 4 x 4 per-thread sub-tiles. A thread-row of 8 lanes then stores 8 x 4 = 32 consecutive floats (128 bytes) per instruction. On RDNA (32-wide) the book's layout applies unchanged. The `.cu` must derive the layout from a compile-time width chosen by the `__GFX9__` test, never from a literal 32.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "warpSize")

### B2. LDS banks and the "one padding column" rule on AMD

**Verdict: CONFIRMED** that AMD documents banks and the same padding fix (`infra-facts.md` B10, `ch06-facts.md` B4): 32 banks on CDNA 1 to 3, 64 banks on CDNA 4 and RDNA 2 to 4, 4 bytes each. Re-read today: "A single warp can achieve up to 64 bytes per cycle throughput (16 lanes per cycle)." and "Conflicts are resolved by serializing accesses across multiple cycles. ... Broadcasting from the same address to multiple threads is handled efficiently without conflicts."

What AMD does not document is which lanes of a 64-wide wavefront are checked against each other for a conflict, given the 16-lanes-per-cycle path. So a conflict count for AMD is our model, not AMD's (Group E gives it for both bank counts). The measured number is the ROCm Compute Profiler's LDS "Bank Conflict Rate" (`ch06-facts.md` B4).

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (section "Local data share (LDS)")

### B3. Registers, occupancy and the 256-thread block on AMD (§15.7)

**Verdict: PARTLY.** The trade the book describes holds on AMD, and AMD says GEMMs run at low occupancy (`ch04-facts.md` B10). But the numbers differ in kind: up to 256 vector registers per thread (plus 256 matrix-accumulation registers on CDNA), and the register file is documented only as a range.

- HIP hardware features table (columns RDNA1, CDNA1, RDNA2, CDNA2, RDNA3, CDNA3): "Maximum number of 32-bit vector registers per thread | 256 | 256 (vector) + 256 (matrix) | 256 | 256 (vector) + 256 (matrix) | 256 | 256 (vector) + 256 (matrix)"; "Maximum number of resident threads per compute unit | 1280 | 2048 | 1024 | 2048 | 1024 | 2048"; "Maximum number of resident wavefronts per compute unit | 40 | 32 | 32 | 32 | 32 | 32".
- Register file: "Vector register files: 256-512 KiB of VGPR storage split across the four SIMDs." and accumulation registers "providing up to 512 KiB of combined register storage per CU". No per-device figure; the calculator must use `regsPerMultiprocessor` from `hipDeviceProp_t` (`ch05-facts.md` B2) and say that AMD allocates per wavefront in granules (`ch04-facts.md` B10).
- "Register usage directly impacts CU occupancy. Each warp requires a portion of the finite VGPR and SGPR pools. Higher register usage per thread reduces the maximum number of concurrent warps".
- Launch bounds matter more on HIP than on CUDA for this kernel: without `__launch_bounds__`, HIP compiles for the device's largest block and "Reducing MAX_THREADS_PER_BLOCK allows the compiler to use more resources per thread than an unconstrained compilation" (`ch08-facts.md` B4). For an 80-plus-register kernel, `__launch_bounds__(256)` in the shared `.cu` is the documented way to let the compiler keep the tile in registers (our application of that statement; not measured).

- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (sections "Vector arithmetic logic unit", "Register pressure and occupancy")
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html (section "Register pressure theory")

### B4. Matrix instructions on AMD: intrinsic spellings, and whether only CDNA has them (counterpart of A6)

**Verdict: PARTLY.** The intrinsic spellings left unverified in `ch24-facts.md` B1 are confirmed in the ROCm 10.2 compiler. But "matrix hardware is CDNA-only", which the HIP features table suggests ("Matrix Cores" ❌ for RDNA1 to 3), is not what the compiler says: RDNA3 and RDNA4 have WMMA instructions with their own builtins. This also settles the RDNA3 disagreement noted in `infra-facts.md` B12.

From `clang/include/clang/Basic/BuiltinsAMDGPU.td` at `ROCm/llvm-project` tag `therock-10.2`:

- `def __builtin_amdgcn_mfma_f32_16x16x4f32 : AMDGPUBuiltin<"_Vector<4, float>(float, float, _Vector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;` (line 649; also `__builtin_amdgcn_mfma_f32_32x32x2f32`, line 648). This is an FP32-in, FP32-out matrix instruction, so CDNA can run an FP32 GEMM on matrix units.
- `def __builtin_amdgcn_mfma_f32_16x16x8_xf32 : ... "mai-insts">;` (line 676): the reduced-precision "xf32" form.
- `def __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 : AMDGPUBuiltin<"_Vector<8, float>(_Vector<16, _Float16>, _Vector<16, _Float16>, _Vector<8, float>)", [Const], "wmma-256b-insts,wavefrontsize32">` (line 503) and `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` (line 815, "wmma-128b-insts").
- In `llvm/lib/Target/AMDGPU/AMDGPU.td` at the same tag, `FeatureWMMA256bInsts` appears in the `FeatureISAVersion11_Generic`, `11_0_Common` and `11_5_Common` sets and `FeatureWMMA128bInsts` in `11_7_Common` and `FeatureISAVersion12` (gfx11 is RDNA3, gfx12 is RDNA4); `FeatureMAIInsts` in `9_0_MI_Common` and `9_4_Common` (CDNA).

Project evidence: llama.cpp's `mma.cuh` (D1) calls `__builtin_amdgcn_mfma_f32_16x16x4f32`, `__builtin_amdgcn_mfma_f32_16x16x8_xf32` and the `__builtin_amdgcn_wmma_*_w32` and `_w32_gfx12` forms, with the comment "CDNA4 (gfx950) does not support xf32 MFMA, use f32 path like CDNA2/CDNA1".

The C++ route is rocWMMA (`infra-facts.md` B12); its umbrella header exists at the tag as `projects/rocwmma/library/include/rocwmma/rocwmma.hpp`, so the include is `#include <rocwmma/rocwmma.hpp>`. None of these share source text with `nvcuda::wmma`; a portable lab keeps tensor-core code out of the shared `.cu` (CLAUDE.md portability traps).

- https://github.com/ROCm/llvm-project/blob/therock-10.2/clang/include/clang/Basic/BuiltinsAMDGPU.td
- https://github.com/ROCm/llvm-project/blob/therock-10.2/llvm/lib/Target/AMDGPU/AMDGPU.td
- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocwmma/library/include/rocwmma/rocwmma.hpp

### B5. An LDGSTS or TMA counterpart on AMD: asynchronous global-to-LDS copies (corrects `ch06-facts.md` B8)

**Verdict: PARTLY.** AMD documents the hardware (Data Movement Engines on CDNA 3 and 4) and names an intrinsic for it, but that name does not exist in the ROCm 10.2 compiler. Global-to-LDS loads do exist in the compiler under other names, undocumented in the HIP guide.

- HIP "Hardware implementation", still today: "Programmers can access this functionality through asynchronous copy intrinsics in ROCm, such as __builtin_amdgcn_async_work_group_copy, which map directly to hardware-level DME operations."
- `async_work_group_copy` occurs zero times in `BuiltinsAMDGPU.td` at `ROCm/llvm-project` `therock-10.2`, on its `amd-staging` branch, and on upstream `llvm/llvm-project` `main`, and in the four HIP device headers I read at `therock-10.2` (`amd_device_functions.h`, `amd_hip_runtime.h`, `amd_warp_functions.h`, `amd_hip_cooperative_groups.h`). Every `__builtin_amdgcn_*` the compiler accepts is declared in that `.td` file, so the documented name is not a usable builtin in this release. (OpenCL C has a built-in called `async_work_group_copy`; the HIP page's name may come from there. That is a guess, not checked.)
- What the compiler has, same file: `def __builtin_amdgcn_global_load_lds : AMDGPUBuiltin<"void(void address_space<1> *, void address_space<3> *, _Constant unsigned int, _Constant int, _Constant unsigned int)", [], "vmem-to-lds-load-insts">;` (line 444) and `__builtin_amdgcn_load_to_lds` (line 442). The feature: `defm VMemToLDSLoad : AMDGPUSubtargetFeature<"vmem-to-lds-load-insts", "The platform has memory to lds instructions (global_load w/lds bit set, buffer_load w/lds bit set or global_load_lds. This does not include scratch_load_lds.">`, included in the GFX9 and GFX10 generation feature sets (`AMDGPU.td` lines 1767 and 1801) and not in GFX11 or GFX12.
- The same file (and upstream `main`) also declares `__builtin_amdgcn_global_load_async_lds`, `__builtin_amdgcn_global_load_async_to_lds_b128`, `__builtin_amdgcn_s_wait_asynccnt` and `__builtin_amdgcn_tensor_load_to_lds` (feature "tdm-insts", which `AMDGPU.td` puts in `FeatureISAVersion12_50_Common`, the gfx1250 target). These are compiler source for an AMD target not described in the HIP docs I read; do not teach them.
- The C++ library route is still absent: no `<cuda/pipeline>` in libhipcxx and a synchronous cooperative-groups `memcpy_async` (`ch06-facts.md` B8).

For the course: on HIP, the pipelined loop of Fig. 15.14 is the book's software form (two LDS buffers, one `__syncthreads()`); there is no documented asynchronous copy to put in it.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (section "Data movement engine (CDNA 3 / CDNA 4)")
- https://github.com/ROCm/llvm-project/blob/therock-10.2/clang/include/clang/Basic/BuiltinsAMDGPU.td (lines 442 to 444)
- https://github.com/ROCm/llvm-project/blob/therock-10.2/llvm/lib/Target/AMDGPU/AMDGPU.td (lines 1607 to 1610, 1751 to 1810)
- https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.td
- https://github.com/ROCm/rocm-systems/tree/therock-10.2/projects/clr/hipamd/include/hip/amd_detail

### B6. hipBLAS as the M3 baseline on AMD (counterpart of A10)

**Verdict: CONFIRMED.** Same call shape as cuBLAS, column-major like rocBLAS (`ch03-facts.md` B6), FP32 by default.

- Declaration at `therock-10.2` (`projects/hipblas/library/include/hipblas.h`, line 14726): `HIPBLAS_EXPORT hipblasStatus_t hipblasSgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const float* alpha, const float* AP, int lda, const float* BP, int ldb, const float* beta, float* CP, int ldc);`
- Math modes in the same header: `HIPBLAS_DEFAULT_MATH`, `HIPBLAS_XF32_XDL_MATH /* equivalent to rocblas_xf32_xdl_math_op, not supported in cuBLAS */`, `HIPBLAS_PEDANTIC_MATH /* equivalent to CUBLAS_PEDANTIC_MATH, not yet supported in rocBLAS */`, `HIPBLAS_TF32_TENSOR_OP_MATH /* use TF32 tensor cores with cuBLAS backend, not supported in rocBLAS */`. rocBLAS: `rocblas_default_math = 0x0` "Default precision" and `rocblas_xf32_xdl_math_op = 0x1` "Enable acceleration of single precision routines using XF32 xDL."
- CMake: `add_library( roc::hipblas ALIAS hipblas )` and `rocm_export_targets( TARGETS roc::hipblas ...)` in `projects/hipblas/library/src/CMakeLists.txt`; llama.cpp links `roc::hipblas` (`ch05-facts.md` B9). I found no current hipBLAS docs page on CMake use (the obvious URL returns 404 under "hipBLAS 3.8.0 Documentation").
- Name mapping as a real project does it (llama.cpp `vendors/hip.h`): `#define cublasCreate hipblasCreate`, `#define cublasSgemm hipblasSgemm`, `#define CUBLAS_OP_N HIPBLAS_OP_N`, `#define CUBLAS_OP_T HIPBLAS_OP_T`, `#define cublasHandle_t hipblasHandle_t`, `#define cublasSetStream hipblasSetStream`, and `#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS` (math mode ignored on HIP).

- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipblas/library/include/hipblas.h
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocblas/library/include/internal/rocblas-types.h (lines 327 to 336)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipblas/library/src/CMakeLists.txt (lines 59 and 183 to 184)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/hip.h

### B7. AMD's counterpart of CUTLASS

**Verdict: CONFIRMED.** Composable Kernel (CK): "The Composable Kernel library provides a programming model for writing performance critical kernels for machine learning workloads across multiple architectures including GPUs and CPUs, through general purpose kernel languages such as HIP C++." (docs "Composable Kernel 1.3.0", with a "CK Tile" section). PyTorch's ROCm build can route float GEMM to it (D4). rocWMMA and hipBLASLt: `infra-facts.md` B12.

- https://rocm.docs.amd.com/projects/composable_kernel/en/latest/index.html

---

### B8. `hipMalloc` base alignment, which the 16-byte `float4` loads rely on (added 2026-11-20 by the main session)

**Verdict: COULD NOT VERIFY.** Lab 15.3 onward loads tiles as `float4` from `hipMalloc` memory and so needs 16-byte alignment. The chapter 15 fix stage read AMD's HIP memory-management reference on 2026-11-20 (group___memory.html) and found no alignment statement for `hipMalloc`, beyond the sentence on size 0; `hipMallocPitch` says "Currently the alignment is set to 128 bytes". CUDA documents at least 256-byte alignment for `cudaMalloc` (infra-facts A6, ch06-facts A3). Until an AMD page states it, the labs' HIP `float4` paths stay labelled unverified: they assume a 16-byte-aligned base and fall back to scalar accesses only when a row length is not a multiple of 4 (labs 15.3 and 15.6, header comments).

- https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___memory.html

## Group C: Metal

### C1. The §15.5 warp layout on Metal: SIMD-group width

**Verdict: CONFIRMED** (from `ch05-facts.md` C3 and `ch03-facts.md` C9): Apple calls SIMD groups "also known as warps or wavefronts", and the width is "returned by the threadExecutionWidth of your compute pipeline state object"; no fixed value is documented. MLX (`tgp_size = WM * WN * 32`) and llama.cpp (`#define N_SIMDWIDTH 32 // assuming SIMD group size is 32`) both assume 32 (`ch06-facts.md` D2, D3).

Consequence for the lab (ours): the remapped kernel's layout is a compile-time constant in MSL too, so `launch.mm` should check `pipeline.threadExecutionWidth` against the constant the kernel was built with and refuse to run on a mismatch, rather than produce wrong stores silently.

### C2. Threadgroup-memory banks and padding on Metal (§15.6)

**Verdict: COULD NOT VERIFY; none is documented.** `infra-facts.md` C13 and `ch06-facts.md` C2 still stand (re-searched the MSL 4.1 PDF today: no "bank"). Apple's only access-order advice is "neighboring threads in a quad group" to "neighboring elements in threadgroup memory", and it advises removing many-threads-one-address reads, the opposite of NVIDIA's broadcast. MLX pads every threadgroup tile by 16 bytes (`tgp_padding_a = 16 / sizeof(T)`, D3), which is practice, not documentation. The padding stage of the lab is a measurement on Metal, with a prediction and no model.

Size check (our arithmetic, Group E): the padded tiles, double-buffered, need 17,408 bytes, under the 32 KB threadgroup limit (`ch05-facts.md` C1).

### C3. Registers, occupancy and a 256-thread threadgroup on Metal (§15.7)

**Verdict: COULD NOT VERIFY** any register count or register limit; none is documented (`ch04-facts.md` C5, `ch08-facts.md` C4). New today: what Xcode's Pipeline Statistics viewer does document, which is useful for the pipelining stage.

- "Analyzing draw command and compute dispatch performance with pipeline statistics": GPU activities include "Synchronization (barrier): The amount of time that the GPU spends when a thread reaches a barrier and the GPU waits for remaining threads in the same group to arrive at the barrier before proceeding." and "Control flow: ... Use a constant iteration count to minimize control flow time for loops because the Metal compiler can generate optimized code in those cases."
- The page lists no register or spill statistic. (A 2017 WWDC slide deck mentions register and spill remarks; I did not use it, as the current page does not.)
- Two documented facts that bear on a 256-thread register-heavy kernel: Metal "may lower the thread limit when creating the compute pipeline state" (`ch04-facts.md` C5), so `launch.mm` must check `maxTotalThreadsPerThreadgroup >= 256`; and the occupancy counter exists in Xcode with no API (`ch06-facts.md` C5).

- https://developer.apple.com/documentation/xcode/analyzing-draw-command-and-compute-dispatch-performance-with-pipeline-statistics

### C4. Full unrolling and forced inlining in MSL (the `#pragma unroll` and `__forceinline__` of Figs. 15.4 to 15.9)

**Verdict: PARTLY.** Unrolling: no documented control (`ch06-facts.md` C4; the Clang loop pragma is accepted by the front end, compile-only). Forced inlining: the MSL 4.1 specification has no inline attribute (searched today for "always_inline", "noinline", "forceinline": none). But Apple's own shipped headers define one.

- Local Metal toolchain 27.1, `usr/metal/32023/lib/clang/32023/include/metal/metal_types`, line 20: `#define METAL_FUNC inline __attribute__((__always_inline__))`. It is available in any MSL source that includes `<metal_stdlib>`. MLX writes its device helpers as `METAL_FUNC` (for example `METAL_FUNC IdxT elem_to_loc(` in `mlx/backend/metal/kernels/utils.h`) without defining the macro itself.

For the lesson: the `.metal` files may use `METAL_FUNC` (or `inline __attribute__((always_inline))`) for the helpers, labelled "Apple's header macro, not in the specification", next to the Clang unroll pragma with the same label.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- Local header: `/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v27.1.266.1.6daX5Z/Metal.xctoolchain/usr/metal/32023/lib/clang/32023/include/metal/metal_types`
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/utils.h

### C5. Matrix hardware from Metal kernels (counterpart of A6)

**Verdict: CONFIRMED** for the APIs (`infra-facts.md` C10, `ch24-facts.md` C1): `simdgroup_float8x8` with `simdgroup_multiply_accumulate` (8 x 8 only, Apple7 and later), and Metal Performance Primitives `matmul2d` (Metal 4), which Apple recommends instead. Apple documents no matrix unit that either uses; do not call them tensor cores.

Added from MSL 4.1 Table 7.3 ("MatMul2D data type supported"), read today: `float x float -> float` is supported from "Metal 4", so an FP32 GEMM can be written with `matmul2d`; and from "Metal 4 and OS 26.4" there are `half x int4b_format -> half` and `half x uint4b_format -> half` rows, which matter for the quantised-kernel chapter (X2).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 7.2.1, Table 7.3, pages 351 to 352)

### C6. `MPSMatrixMultiplication` as the M3 baseline on Metal (counterpart of A10)

**Verdict: CONFIRMED.** Not deprecated in the macOS 27.0 SDK, row-major (`ch03-facts.md` C7), and it does not lower precision unless told to.

- Header (`MetalPerformanceShaders.framework/.../MPSMatrix.framework/Headers/MPSMatrixMultiplication.h`, macOS 27.0 SDK): `MPS_CLASS_AVAILABLE_STARTING( macos(10.13), ios(10.0), macCatalyst(13.0), tvos(10.0))` `@interface MPSMatrixMultiplication : MPSKernel`, no deprecation macro. "A MPSMatrixMultiplication object computes: C = alpha * op(A) * op(B) + beta * C".
- Initialiser: `-(nonnull instancetype) initWithDevice: (nonnull id<MTLDevice>) device transposeLeft: (BOOL) transposeLeft transposeRight: (BOOL) transposeRight resultRows: (NSUInteger) resultRows resultColumns: (NSUInteger) resultColumns interiorColumns: (NSUInteger) interiorColumns alpha: (double) alpha beta: (double) beta;` Encode: `-(void) encodeToCommandBuffer: (nonnull id <MTLCommandBuffer>) commandBuffer leftMatrix: (MPSMatrix * __nonnull) leftMatrix rightMatrix: (MPSMatrix * __nonnull) rightMatrix resultMatrix: (MPSMatrix * __nonnull) resultMatrix`.
- Layout (`MetalPerformanceShaders.h`): `+[MPSMatrixDescriptor matrixDescriptorWithRows:columns:rowBytes:dataType]`; "'rowBytes' must be a multiple of the size, in bytes, of a single data element." "The value of 'rowBytes' can also have an impact on the performance of kernels which use the resulting MPSMatrix object; the convenience method, +[MPSMatrixDescriptor rowBytesForColumns:dataType], can be used to query a performant value".
- Precision: `MPSKernelOptionsAllowReducedPrecision` "When possible, MPSKernels use a higher precision data representation internally than the destination storage format to avoid excessive accumulation of computational rounding error in the result. MPSKernelOptionsAllowReducedPrecision advises the MPSKernel that the destination storage format already has too much precision ... When enabled, the precision of the result may vary by hardware and operating system." It is an option you set, so the default does not reduce precision.
- Link with `-framework MetalPerformanceShaders` (the lab runtime currently links only Metal and Foundation, `labs/cmake/Backends.cmake` line 129).

Other routes (`infra-facts.md` C11): MPSGraph matmul; Accelerate `cblas_sgemm` as the CPU reference. MLX's `mx.matmul` (D3) is a useful second Metal comparison if the reader has MLX installed; it is not an Apple system library.

- Local headers: `MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/Frameworks/MPSMatrix.framework/Versions/A/Headers/MPSMatrixMultiplication.h` (lines 39 to 192) and `.../MetalPerformanceShaders.framework/Versions/A/Headers/MetalPerformanceShaders.h` (lines 430 to 440, 795 to 830), macOS SDK 27.0
- https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication

---

## Group D: How llama.cpp, MLX and PyTorch do this chapter's GEMM

Read from raw files on the default branches on 2026-11-20. Line references are to today's files and will drift.

### D1. llama.cpp, CUDA and HIP backend

**Verdict: CONFIRMED.**

- `ggml/src/ggml-cuda/mma.cuh` (last commit 2026-11-15): "This file contains primitives that expose the tensor core PTX instructions for CUDA code. The primitives can be used in a similar way as the nvcuda::wmma interface but with a well-defined memory layout." It issues PTX directly, for example `asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 ...")`, `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`, and `ldmatrix.sync.aligned.m8n8.x2.b16` ("all pointers for load_ldmatrix must be to shared memory and aligned to 16 bytes"). On AMD the same file calls `__builtin_amdgcn_mfma_f32_16x16x4f32`, `__builtin_amdgcn_mfma_f32_16x16x8_xf32` (CDNA3), `__builtin_amdgcn_mfma_f32_16x16x16f16` and the RDNA `__builtin_amdgcn_wmma_..._w32` / `_w32_gfx12` builtins (B4).
- Compile-time warp width, `common.cuh`: `static constexpr __device__ int ggml_cuda_get_physical_warp_size() {` `#if defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))` `return 64;` `#else` `return 32;`. AMD's documented macro test (B1).
- The batch-1 decode path (the m = 1 case of §15.2): `mmvq.cuh`: `#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.` and `mmvf.cuh`: `#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.` Up to 8 tokens, llama.cpp uses matrix-vector kernels, not a tiled GEMM.
- Padding against bank conflicts in its tiled quantised GEMM: `ch06-facts.md` D1.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mma.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvf.cuh

### D2. llama.cpp, Metal `kernel_mul_mm`

**Verdict: CONFIRMED.** `ggml/src/ggml-metal/kernels/mul_mm.metal` (last commit 2026-11-05), the non-tensor path:

- Block tile: `constexpr int NR0 = 64;` `constexpr int NR1 = 32;` `constexpr int NK = 32;`, so a 64 x 32 output tile per threadgroup with a 32-deep K step (the book: 128 x 128 x 8).
- Accumulators: `simdgroup_float8x8 mc[8];` per SIMD-group, zeroed with `make_filled_simdgroup_matrix<float, 8>(0.f)`; the store indexing `32*(sgitg & 1)` and `16*(sgitg >> 1)` places four SIMD-groups on 32 x 16 sub-tiles (our reading of the indexing).
- Threadgroup memory as two staging areas: `threadgroup S0 * sa = (threadgroup S0 *)(shmem);` `threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);`, one buffer each, with `threadgroup_barrier(mem_flags::mem_threadgroup)` around the loads (no double buffering).
- Output: full tiles go straight to device memory with `simdgroup_store`; edge tiles are staged in threadgroup memory and copied out with `device float4 * D4 = (device float4 *) D;` by one SIMD-group, the CUTLASS-style epilogue of A12. Comment in the code: "if no bounds checks on the output are needed, we can directly write to device memory".
- Host side, `ggml-metal-ops.cpp`: `ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + nr1 - 1) / nr1), ((ne01 + nr0 - 1) / nr0), ne12 * ne13, 32, nsg, 1);`, threadgroups of 32 x `nsg` threads.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mm.metal (lines 155 to 365)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-ops.cpp (lines 2896 to 2910)

### D3. MLX, Metal "steel" GEMM: tile sizes chosen per device

**Verdict: CONFIRMED.**

- Instantiated shapes, `steel/gemm/kernels/steel_gemm_fused.metal`: `(bm, bn, bk, wm, wn)` = `(64, 64, 16, 2, 2)`, `(64, 64, 16, 1, 2)`, `(64, 32, 32, 2, 2)`, `(32, 64, 16, 1, 2)`, `(32, 32, 16, 2, 2)`, `(64, 32, 8, 4, 1)`, for `float16`, `bfloat16`, `float32` and `complex64`. Threads per threadgroup: `STEEL_CONST short tgp_size = WM * WN * 32;` (`steel/gemm/gemm.h`), so 64 to 128 threads, against the book's 256.
- Per-device choice, `mlx/backend/metal/matmul.cpp` (last commit 2026-10-09): `char devc = d.get_architecture().back();` then `GEMM_TPARAM_MACRO(devc)` with branches `/* Small device */` (`'g'` or `'p'`), `/* Large device */` (`'d'`) and `/* Medium device */`, for example `bm = 64; bn = 64; bk = 16; wm = 2; wn = 2;` for the medium case. Which Apple chips map to which letter is not documented in the file; I did not resolve it.
- Padding and single buffering: `ch06-facts.md` D3 (`tgp_padding_a = 16 / sizeof(T)`, two barriers per K step). Unroll macro: `#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")`.

For the reader: Apple's own GEMM uses smaller tiles and fewer threads than the book's NVIDIA-sized 128 x 128 x 8 with 256 threads. That is a design fact to show, not a reason; Apple publishes no register or threadgroup-memory bandwidth figures that would explain it (C2, C3).

- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/gemm.h
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/matmul.cpp (lines 91 to 170)

### D4. PyTorch: where a float matmul goes

**Verdict: CONFIRMED.** `aten/src/ATen/cuda/CUDABlas.cpp` (last commit 2026-11-10):

- `gemm_internal<float>` picks cuBLASLt, cuBLAS or, on ROCm, Composable Kernel: `if (preferred == BlasBackend::Ck) { // gfx11 and gfx12 have no CK fp32 GEMM` ... `if (preferred == BlasBackend::Cublaslt) { gemm_internal_cublaslt<float>(...); } else { gemm_internal_cublas<float>(...); }`.
- `gemm_internal_cublas<float>` calls `cublasGemmEx` with `auto compute_type = CUBLAS_COMPUTE_32F;`, switched to `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` `if (useBF16x9())` (CUDA 12.9 and later; I did not trace what makes `useBF16x9()` true). TF32 is PyTorch's `allow_tf32` flag (`ch24-facts.md` D4).
- The m = 1 bridge in PyTorch's own words, above `gemv`: "// gemv is bw bound, and does not benefit from TF32. But the precision // loss still happens on TF32. So we disable it here." followed by `NoTF32Guard disable_tf32;`.
- MPS: PyTorch's own Metal matmul kernels and MPSGraph (`infra-facts.md` D4).

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp (lines 1017 to 1045, 1301 to 1318, 2699 to 2703)

---

## Group E: The chapter's arithmetic, rechecked

### E1. Worked numbers, and the bank-conflict claims for every layout the labs will build

**Verdict: PARTLY.** Our arithmetic, not a vendor statement; bank counts are computed with NVIDIA's documented rule (bank = 32-bit word index mod 32, distinct words in one bank serialise, same word broadcasts, A2) and, for AMD, the same rule with 64 banks (a model, B2). Every printed number checks. One claim needs a qualifier: "B needs no change" (p. 363) holds only for the §15.5 thread layout, not for the §15.3 / §15.4 layout the labs build first.

| Book page | As printed | Recomputed | Status |
|---|---|---|---|
| 351 | 0.5·m·n/(m+n): 32 x 32 gives 8, 128 x 128 gives 32 FLOP/B | 0.5·1024/64 = 8; 0.5·16384/256 = 32 | correct |
| 351, 363 | intensity independent of k | 2mnk / (4k(m+n)): k cancels | correct |
| (notes) | m = 1 (batch-1 decode) | 0.5·n/(n+1) < 0.5 FLOP/B for any n (4-byte elements) | our derivation |
| 355 | 128 x 8 tile, 256 threads, 4 loads each | 1024 / 256 = 4 | correct |
| 363 | A_s, B_s 4 KB each | 128·8·4 = 4096 B | correct |
| 363 (notes) | padded A_s | 128·9·4 = 4608 B; with B_s 8704 B; double-buffered 17,408 B | under 48 KB static per block on CUDA (PTX: "Maximum capacity for statically allocated shared memory is 48 KB per CTA"), AMD LDS (`ch08-facts.md` B3) and Metal's 32 KB |
| 363 | 64 + 8 + 8 = 80 registers | 80; the §15.5 layout also needs 64 + 16 = 80 | correct |
| 363 | 255 registers, 64 K per SM, 256 threads, 12.5 % | 65,536 / 255 = 257, so one 256-thread block; 256/2048 = 12.5 %, 256/1536 = 16.7 %, 256/1024 = 25 % | correct for 2048-thread SMs; exact fit depends on undocumented allocation chunks (A3) |
| (notes) | shared reads per thread per tile pair | Fig. 15.6: 8·8·8·2 = 1024; Fig. 15.9: 8·(8 + 8) = 128 | 8x fewer |
| 362 | unpadded A strip: elements 0, 32, ..., 224 all in bank 0 | (4g)·8 = 32g, mod 32 = 0 | correct |
| 362 | padded: 0, 36, ..., 252 in banks 0, 4, ..., 28 | (4g)·9 = 36g, mod 32 = 4g | correct |

Worst-case conflict degree per load instruction, computed over every r, i and quadrant (script, 2026-11-20):

| Layout and access | Leading dim of A_s | 32 banks | 64 banks (model) |
|---|---|---|---|
| §15.5 layout (warp 8 x 4 lanes), A strip | 8 | 8-way | 4-way |
| same | 9 | none | none |
| same | 10 / 12 / 16 | 2 / 4 / 8-way | none / 2 / 8-way |
| same | any odd value (9, 11, 13, 15, 17) | none | none |
| §15.5 layout, B strip (row of 128) | n/a | none (4 distinct words in 4 banks, each broadcast to 8 lanes) | none |
| §15.3 / §15.4 layout (16 threads per row, 8 x 8 each), A strip | 8 | 2-way | 2-way |
| same | 9 | none | none |
| §15.3 / §15.4 layout, B strip (row of 128) | n/a | **4-way** (16 distinct words 8 apart) | 2-way |
| 64-wide wavefront, 8 x 8 lanes (B1's layout), A strip | 8 / 9 | 8-way / none | 4-way / none |

Rule for the explorer (ours): with groups of lanes 4 rows apart, padding to any odd leading dimension removes the A-strip conflict on 32 banks; on 64 banks any leading dimension not divisible by 4 does.

Consequences for the labs: (1) If the M3 stages are built in the order "coarsened, register-tiled, padded" without the §15.5 remap, padding A removes only a 2-way conflict and leaves a 4-way conflict on B, so the padding stage should come after the remap, or its prediction must say this. (2) The 64-bank figures and the wave64 row are a model for AMD; AMD does not document how a 64-lane LDS access is split for conflicts (B2). (3) Nothing here applies to Metal (C2).

Alignment for the vector loads and stores of Exercises 1 and 2 (ours, from `infra-facts.md` A6 and `ch06-facts.md` A11): with a 256-byte-aligned base, a `float4` load of an A-tile row segment starts at `(bRow + row)·K + tile·8` floats, aligned to 16 bytes only when K is a multiple of 4; B-tile and C-tile accesses need N a multiple of 4. A padded `A_s` (leading dimension 9) cannot take `float4` stores, as the notes say. The lab's sizes must include a K and N that are not multiples of 4 to exercise the scalar fallback.

The model in this chapter counts only tile loads from DRAM. On a GPU with large caches the measured gain of bigger tiles can be smaller than the 8 to 32 FLOP/B jump suggests; PROGRESS.md records that chapter 5's tiling gained far less than its model on the pilot machine. Predictions in the labs should be phrased as ranges and ratios, not as the model's factor.

---

## Summary of corrections

Every PARTLY or COULD NOT VERIFY item, one line each:

- A3: 255 registers per thread and 64 K per SM hold everywhere, but 2048 resident threads per SM only on compute capabilities 8.0, 9.0, 10.0, 10.3 (1536 on RTX 30/40/50, so one 256-thread block is 16.7 %); registers are allocated "in fixed-size chunks" of undocumented size, so the exact fit comes from the occupancy API or `launch__occupancy_limit_registers`.
- A5: tensor cores date from Volta, but CUDA 13 targets `sm_75` and later; NVIDIA's libraries "Removed support for Maxwell, Pascal, and Volta".
- A6: WMMA is still supported (not "pre-Hopper only"); WGMMA "Requires sm_90a" and `a`-target code "cannot be run on later generation devices"; TMEM (`tcgen05`) is on the `sm_100`/`sm_110` families only, not on compute capability 12.x GeForce Blackwell.
- A7: LDGSTS is CC 8.0+; `cuda::memcpy_async` does not guarantee LDGSTS (only the `__pipeline_memcpy_async` primitives do); the book's "nvcc recognises a vector load plus store and emits LDGSTS" could not be verified, and NVIDIA's triage guide tells you to convert the code yourself or use CUTLASS.
- A8: TMA is CC 9.0+, but `cuda::memcpy_async` reaches it only for 1D bulk copies into `shared::cta`; multi-dimensional tiles need a host-built tensor map (`cuTensorMapEncodeTiled`) and `cuda::ptx::cp_async_bulk_tensor`.
- B3: AMD allows up to 256 vector registers per thread (plus 256 accumulation registers on CDNA) and documents its register file only as "256-512 KiB" per CU; without `__launch_bounds__(256)` HIP compiles for 1024-thread blocks and limits registers per thread.
- B4: the MFMA and WMMA builtin spellings are confirmed in the ROCm 10.2 compiler; contrary to the HIP features table's "Matrix Cores ❌" for RDNA, RDNA3 (gfx11) and RDNA4 (gfx12) have WMMA instructions; gfx950 has no xf32 MFMA.
- B5 (corrects `ch06-facts.md` B8): the HIP docs' `__builtin_amdgcn_async_work_group_copy` does not exist in the ROCm 10.2 compiler; global-to-LDS loads exist as `__builtin_amdgcn_global_load_lds` (GFX9 and GFX10 generations), undocumented in the HIP guide.
- C2 (could not verify): Metal still documents no threadgroup-memory banks; MLX pads by 16 bytes in practice.
- C3 (could not verify): Metal documents no register count; Xcode's Pipeline Statistics viewer documents barrier wait time and advises constant loop trip counts.
- C4: MSL has no documented unroll control and no inline attribute in its specification, but Apple's `<metal_types>` header defines `METAL_FUNC` as `inline __attribute__((__always_inline__))`.
- E1: "B needs no change" holds for the §15.5 layout only; in the §15.3/§15.4 layout the B strip read is a 4-way conflict on 32 banks, so padding A alone does little before the remap.

Status changes worth knowing even where the verdict is CONFIRMED:

- A9: cuTile is now "CUDA Tile" (cuTile Python; CUDA Tile C++ from toolkit 13.3); cuDNN docs moved to `docs.nvidia.com/cudnn/`; all three of the chapter's references still resolve.
- A10 / B6 / C6: all three vendor baselines compute FP32 in FP32 by default; TF32 (cuBLAS), BF16x9 emulation (cuBLAS, CC 10.0/10.3), XF32 (rocBLAS/hipBLAS) and `MPSKernelOptionsAllowReducedPrecision` are opt-in.
- A12: CUTLASS documents the chapter's whole recipe, double-buffers at the register level too, and makes stores coalesced through a shared-memory epilogue rather than the book's thread remapping.
- D1 / D4: llama.cpp switches to matrix-vector kernels up to 8 tokens per batch, and PyTorch's code says gemv "is bw bound": both are the m = 1 case of §15.2.

## Where the backends really differ

One line per cell, documented facts only unless marked "ours". Nothing here was run on a device.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Width for the §15.5 warp layout | 32 | 64 on gfx9 (CDNA), 32 on gfx10 and later (RDNA) | `threadExecutionWidth` of the pipeline; no documented value |
| Compile-time width in kernel code | 32 (constant) | Host-chosen template parameter, or `#if defined(__GFX8__) \|\| defined(__GFX9__)` | Not documented; a constant plus a host check (ours) |
| Shared-memory banks | 32 x 4 bytes | 32 (CDNA 1 to 3) or 64 (CDNA 4, RDNA 2 to 4) x 4 bytes | None documented |
| Padding that removes the A-strip conflict (ours, Group E) | Any odd leading dimension | Odd on 32 banks; not divisible by 4 on 64 banks (model) | Measure only |
| Max registers per thread | 255 | 256 vector (+ 256 accumulation on CDNA) | Not documented |
| Register file per SM / CU | 64 K 32-bit registers, chunked allocation | "256-512 KiB" VGPR per CU; per-wave granules | Not documented ("assigned dynamically from L1" on Apple family 9, `ch06-facts.md` C2) |
| Holding the 256-thread block size | `__launch_bounds__(256)` | `__launch_bounds__(256)` (matters more: default assumes 1024) | `[[max_total_threads_per_threadgroup(256)]]`; check pipeline cap |
| Full unroll / forced inline | `#pragma unroll`, `__forceinline__` | Same spellings | Not in the spec; Clang loop pragma and Apple's `METAL_FUNC` macro work in practice |
| Static shared memory per block | 48 KB static (more is dynamic, opt-in) | Whole CU LDS, no opt-in step | 32 KB total threadgroup memory |
| Async global to shared copy | LDGSTS / `cp.async`, CC 8.0+ | `__builtin_amdgcn_global_load_lds` in the compiler (GFX9, GFX10), not in HIP docs; documented name does not exist | None documented |
| Bulk multi-dimensional copy | TMA, CC 9.0+, tensor map + `cuda::ptx::cp_async_bulk_tensor` | DME on CDNA 3/4 described, no usable documented API | None documented |
| Warp-level matrix multiply | `nvcuda::wmma` (CC 7.0+), PTX `mma.sync` | MFMA builtins (CDNA), WMMA builtins (RDNA3/4); rocWMMA | `simdgroup_float8x8` + `simdgroup_multiply_accumulate` (Apple7+) |
| Larger asynchronous matrix multiply | WGMMA (`sm_90a` only); `tcgen05` with TMEM (`sm_100`/`sm_110` families) | None documented beyond MFMA | MPP `matmul2d` over SIMD-groups (Metal 4), including `float x float -> float` |
| Template GEMM library | CUTLASS, CUDA Tile | Composable Kernel, rocWMMA | None; MPP is the nearest |
| Vendor GEMM to benchmark against | `cublasSgemm` (`cublas_v2.h`, `CUDA::cublas`), column-major | `hipblasSgemm` (`roc::hipblas`), column-major | `MPSMatrixMultiplication` (`-framework MetalPerformanceShaders`), row-major |
| FP32 baseline precision by default | FP32; TF32 and BF16x9 opt-in | FP32; XF32 opt-in | Higher-precision internal by default; reduced precision opt-in |
| Register and spill readout | `--resource-usage`, `numRegs`, `launch__registers_per_thread` | `--resource-usage`, `numRegs` (VGPRs) | None documented |
| Bank-conflict and barrier readout | `l1tex__data_bank_conflicts_pipe_lsu.sum` | LDS "Bank Conflict Rate" | Pipeline Statistics "Synchronization (barrier)"; no bank metric |
