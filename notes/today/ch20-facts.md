# Chapter 20 fact check: CUDA, HIP, Metal, and the sources the chapter cites

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims the verdict is for the item as a whole and the text says which sub-claim failed or could not be found. A verdict judges the claim in the item title: the book's statement, or the assumption a course author would most likely bring.
- **Quote** is copied from the source (HTML flattened to text, GitHub raw files at a tag or on the default branch, Apple's documentation JSON, PDF text of papers and specifications). Line breaks inside quotes are removed.
- "infra A3", "ch10 C3", "ch05 C1" point at items in `notes/today/infra-facts.md`, `ch10-facts.md`, `ch05-facts.md`. Anything settled there is cited, not re-checked. Page numbers ("p. 487") are book pages as recorded in `notes/ch20.md`.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide and CUDA Math API "v13.5"; CCCL docs at `nvidia.github.io/cccl/unstable/`; cuDNN "Attention" page (support matrix "based on cuDNN backend version 9.18.1", version history up to 9.21.0); ROCm Core SDK 10.2.0 sources at tag `therock-10.2` of `ROCm/rocm-systems` and `ROCm/rocm-libraries`; Metal Shading Language Specification "Version 4.1" (PDF dated 2026-06-04); Apple documentation JSON for MPSGraph; FlashAttention `main` (latest release tag `fa4-v4.0.0.beta40`, 2026-11-18); llama.cpp `master` (`fattn.cu` last commit 2026-11-19, `fattn-common.cuh` 2026-11-18); MLX `main` (docs "MLX 0.32.4"); mlx-lm `main`; PyTorch `main` and docs 2.15; the `meta-llama/llama` reference repository (not archived, last push 2025-01-26). Papers read as arXiv PDFs.
- Not checked: anything by running code on a CUDA or AMD GPU. Two Metal checks were run on the Mac this course is built on (Apple `metal` front end 32023.921, macOS SDK 27.0) and are labelled "our compile test" or "our run"; they show behaviour, not documentation. Group F is our arithmetic, checked with a short script.

Totals: 15 CONFIRMED, 6 WRONG, 13 PARTLY, 0 COULD NOT VERIFY (34 items: A1 to A8, B1 to B5, C1 to C7, D1 to D5, E1 to E8, F1). Sub-claims that failed or could not be found, including two COULD NOT VERIFY sub-claims (B3, A5), are collected under "Summary of corrections".

---

## Already settled elsewhere (cite, do not re-check)

| Chapter 20 needs | Settled in |
|---|---|
| 1024 threads per block, warp size 32, grid x up to 2^31 - 1 (the book's "N cannot exceed the maximum number of blocks in x", p. 487) | ch05 A5 |
| Static shared memory up to 48 KB per block without opt-in; larger needs dynamic shared memory and `cudaFuncSetAttribute` | ch05 A3 |
| 32 banks of 4 bytes on NVIDIA (the book's `LOG_NUM_BANKS` = 5); padding by one column | infra A10, ch15 A2 |
| `__shfl_sync` signature, `srcLane % width` rule, mask constraints | infra A3 |
| `cub::BlockReduce`, `cub::WarpReduce`; `cub::Max()` and `cub::Sum()` removed in CCCL 3.0 (write `cuda::maximum<>{}`, `cuda::std::plus<>{}`) | infra A11 |
| Grid-wide barrier through cooperative groups (p. 493) | infra A4, ch10 A8 |
| `expf` (2 ULP), `__expf`, `--use_fast_math`; `expf(-inf)` returns +0 | ch10 A11 |
| Tensor cores, TMA, WGMMA on Hopper (p. 504) | ch15 A5, A6, A8; ch24 A19 |
| cuBLAS `cublasSgemm` with α (p. 486), hipBLAS on AMD | ch15 A10, B6 |
| H100 figures are the book's example only (none are printed in this chapter) | ch05 A8 |
| Wavefront 64 on gfx9 (Instinct), 32 on gfx10 and later; `warpSize` not a compile-time constant | ch05 B5, ch10 B2 |
| HIP `_sync` intrinsics need a 64-bit mask; `0xffffffff` fails a `static_assert` | infra B3, ch10 B1 |
| hipCUB `BlockReduce` / `WarpReduce` and its functor spellings | infra B11 |
| AMD LDS: a block may use the whole LDS of a CU (64, 128 or 160 KiB in AMD's table); 32 or 64 banks | ch08 B3, infra B10 |
| HIP compiles for a 1024-thread block unless told otherwise; `__launch_bounds__` | ch08 B4 |
| AMD documents lockstep wavefronts; `__syncwarp` on HIP | ch04 B4, B5 |
| `expf` and `fmaxf` on HIP | ch10 B4 |
| Metal threadgroup memory: 32 KB on Apple4 to Apple11; `maxThreadgroupMemoryLength` | ch05 C1, C3 |
| Metal SIMD-group width undocumented; grouping implementation-defined; ask for `[[simdgroup_index_in_threadgroup]]` and `[[thread_index_in_simdgroup]]` | ch04 C2, infra C5 |
| `simd_shuffle`, `simd_broadcast`, `simd_sum`, `simd_max` (result in every active lane) | infra C2, ch10 C1 |
| `simdgroup_barrier` | infra C4 |
| Metal fast math makes INF and NaN undefined; default math mode documented three ways | ch10 C3, ch05 C7 |
| Metal `exp` and `fast::exp` accuracy | ch10 C4 |
| `simdgroup_matrix` and Metal Performance Primitives (no "tensor cores" in Apple's docs) | infra C10, ch24 C1 |
| Compile-time constants through `MTLCompileOptions.preprocessorMacros` | ch05 C7 |
| llama.cpp CUDA attention kernel choice (`fattn.cu`: VEC, TILE, MMA_F16); Metal `fa_*.metal` and `fa_tensor.metal` (MPP); MLX `scaled_dot_product_attention.cpp`, `sdpa_vector.h`, `steel/attn/`; PyTorch `Attention.mm` | infra D1 to D4 |
| llama.cpp assumes `N_SIMDWIDTH 32` on Metal | ch04 D2 |
| llama.cpp, MLX and PyTorch softmax kernels (M4) | ch10 D1, D3, D4 |

---

## Group A: CUDA, and the book's dated statements about NVIDIA software

### A1. "Both `gridDim.y` and `gridDim.z` are 0" for a 1D grid (p. 487)

**Verdict: WRONG.** Unused grid dimensions are 1.

- CUDA Programming Guide, section 5.4.2.2 "Built-in Variables": "Both dim3 and uint3 are trivial structures consisting of three unsigned values named x, y, and z. In C++11 and later, the default value of all components of dim3 is 1."
- HIP has the same default in its header (`hip_runtime_api.h` at `therock-10.2`, line 1448): `constexpr __host__ __device__ dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1)`.
- Metal has no default to rely on: the host passes all three components of an `MTLSize` (ch05 C5). The lesson can say "1, not 0" for all three.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.2.2)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 1443 to 1451)

### A2. CUB as used in Figs. 20.4, 20.12 and 20.15: spellings, where the result lands, and one `TempStorage` shared by every warp

**Verdict: PARTLY.** The classes exist and the result-in-lane-0 rule the book states (pp. 488, 502) is confirmed. The functors are gone (infra A11). The way Figs. 20.12 and 20.15 declare one `__shared__ WarpReduce::TempStorage temp_store` inside the device function, so that all 16 warps of the block use the same storage, is not the documented usage.

- Result location, `WarpReduce`: "Computes a warp-wide sum in the calling warp. The output is valid in warp lane0." The doc example: "The corresponding output aggregate in threads 0, 32, 64, and 96 will be 496, 1520, 2544, and 3568, respectively (and is undefined in other threads)." So the book's `__shfl_sync(..., curr_max_warp, 0)` broadcast after `Reduce` is required, as it says.
- One storage per warp: the documented example for several warps is "four concurrent warp sum reductions within a block of 128 threads (one per each of the 32-thread warps)" with `__shared__ typename WarpReduce::TempStorage temp_storage[4];` and `WarpReduce(temp_storage[warp_id]).Sum(thread_data);`. The single-storage example has only one warp reducing ("Only the first warp performs a reduction").
- Reuse: after every `WarpReduce` method, "A subsequent __syncwarp() warp-wide barrier should be invoked after calling this method if the collective's temporary storage (e.g., temp_storage) is to be reused or repurposed." After every `BlockReduce` method: "A subsequent __syncthreads() threadblock barrier should be invoked after calling this method if the collective's temporary storage (e.g., temp_storage) is to be reused or repurposed."
- Fig. 20.4 meets the `BlockReduce` rule: between the max `Reduce` (line 11) and the sum `Reduce` (line 20) there are barriers at lines 13 and 19 (our reading of the listing in `notes/ch20.md`).
- Why the shared `WarpReduce` storage may still work today: the class overview now says "The implementation uses warp-level shuffle, warp redux, vote instructions, and specialized paths, without relying on shared memory." That is a statement about the current implementation, not a licence in the API contract; the documented contract is one storage per warp. Lessons should present the per-warp array (or no CUB at all, see B1 and the labs' hand-written reductions).
- Template default: `template<typename T, int LogicalWarpThreads = detail::warp_threads> class WarpReduce`, "Default is the warp size of the targeted CUDA compute-capability". On HIP the default differs (B2).
- Also in the overview: "For floating-point types, cuda::minimum and cuda::maximum ignore NaN inputs."

- https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpReduce.html
- https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockReduce.html

### A3. Lanes of one warp hand the P segment to each other through shared memory with no warp barrier (Figs. 20.13 and 20.15)

**Verdict: WRONG** as a guarantee. In `compute_P_and_update_D` lane L writes element `jj = L` of the warp's row of `S_i`; in `compute_O` every lane reads all `B_c` elements of that row. Nothing between the write and the read orders memory across lanes.

- Shuffles do not order memory: "These intrinsics do not imply a memory barrier. They do not guarantee any memory ordering." (section 5.4.6.5, Warp Shuffle Functions). The `__shfl_sync` and `WarpReduce` calls between the write and the read therefore do not help.
- What does: "The intrinsic function __syncwarp() coordinates communication between the threads within the same warp. When some threads within a warp access the same addresses in shared or global memory, potential read-after-write, write-after-read, or write-after-write hazards may occur. These data hazards can be avoided by synchronizing the threads between these accesses." and "Calling __syncwarp(mask) provides memory ordering among the participating threads within a warp named in mask" (section 5.4.4.2). The guide's own example writes a shared array, calls `__syncwarp()`, then reads another lane's element.
- Where the barrier goes (our analysis of the listings): one `__syncwarp()` after the P write in `compute_P_and_update_D` and before `compute_O` reads the row. The S write in `compute_S_and_max` is read back only by the lane that wrote it, the next `ii` uses a different row, and the next tile is separated by the block barrier at line 39.
- Production code does exactly this. llama.cpp's vector attention kernel writes `KQ[j*nthreads + tid] = KQ_reg[j];` and then calls `ggml_cuda_syncwarp(); // Needed for KQ synchronization.` before the V accumulation reads it (D1).
- Older warp-synchronous code that relied on lockstep is invalid on NVIDIA since independent thread scheduling (ch10 A5, ch04 A10).

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.4.2, 5.4.6.5, 5.4.6.6)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-vec.cuh (lines 321 to 343)

### A4. A `__shared__` declaration inside a `__device__` function (Figs. 20.12, 20.15), and the kernel's 37,376 bytes of static shared memory

**Verdict: CONFIRMED** on CUDA.

- The guide's shared-memory example declares a static array with the comment that the enclosing function may be a kernel or a device function: `__global__ void kernel() { // or a __device__ function` / `__shared__ int smem_var1[4]; // static size` (section 5.4.1.2.1). Section 5.3.10.4.1 forbids `__device__`, `__tile__`, `__constant__` and `__managed__` on non-static locals of a device function and does not list `__shared__`.
- 37,376 bytes (Group F) plus the small `TempStorage` objects is below the 48 KB static limit (ch05 A3), so the book's constants need no opt-in on CUDA. The same footprint does not fit on Metal (C1).
- Static shared variables in a device function are one instance per block, not per warp. That is why A2's shared `TempStorage` is shared by all 16 warps.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.1.2.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html (section 5.3.10.4.1)

### A5. `exp` on a `float` stays single precision; `NEG_INFINITY`; and what minus infinity does in the online softmax

**Verdict: PARTLY.** `exp(float)` is documented as a float overload. `NEG_INFINITY` is not a CUDA name (COULD NOT VERIFY any definition; the chapter does not define it either). And IEEE arithmetic on two minus infinities gives NaN, which the book's kernel avoids only because of its tile order.

- Overloads, CUDA Math API front page: "Note that many floating-point and integer function names are overloaded for different argument types. For example, the log() function has the following prototypes: double log(double x); float log(float x); float logf(float x);". The `cuda::std::exp` table (guide Table 54, section 5.6.7.2) maps `float` to `expf(x)`, "2 ULP".
- Infinity constant: the guide documents `cuda::std::numeric_limits` in `<cuda/std/limits>` (section 5.6.2) and `expf(-inf) = +0` (ch10 A11). I searched the guide's language-extension and math pages for `NEG_INFINITY` and found nothing. Lessons should write `-INFINITY` or `-cuda::std::numeric_limits<float>::infinity()` and say the book's name is a placeholder.
- NaN hazard: "Indeterminate forms such as Inf * 0.0, Inf - Inf, Inf / Inf, and 0.0 / 0.0 result in NaN." (section 5.6.1.3). `compute_O` multiplies by `exp(last_m - m_i[ii])` unconditionally. If a row's running max were still minus infinity after a tile (every column so far masked), that factor is `exp(NaN)`. In the book's kernel tiles run left to right from column 0, so tile 0 always contains a visible column for every row and the running max is finite from then on (our analysis; the emulation in `notes/ch20.md` agrees). A split-K or right-to-left order, a padding mask, or a sliding window breaks that.
- What production code does instead of minus infinity: llama.cpp initialises the running max to `-FLT_MAX/2.0f` and documents why: `#define HALF_MAX_HALF __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.`; MLX uses a finite minimum (`Limits<U>::finite_min`) for the same purpose on Metal (D1, D4).

- https://docs.nvidia.com/cuda/cuda-math-api/index.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (sections 5.6.1.3, 5.6.2, 5.6.7.2)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-common.cuh (lines 9 to 19)

### A6. FlashAttention versions and "state-of-the-art implementations use 16-bit or 8-bit floating point and tensor cores" (pp. 495, 498, 504)

**Verdict: CONFIRMED.**

- FlashAttention (v1): Dao, Fu, Ermon, Rudra, Ré; the repository's citation block gives "Advances in Neural Information Processing Systems (NeurIPS)", 2022 (arXiv 2205.14135).
- FlashAttention-2: arXiv 2307.08691 (July 2023), the version the book's listing follows; the repository now cites it as "International Conference on Learning Representations (ICLR)", 2024.
- FlashAttention-3 (arXiv 2407.08608), abstract: "We develop three main techniques to speed up attention on Hopper GPUs: exploiting asynchrony of the Tensor Cores and TMA to (1) overlap overall computation and data movement via warp-specialization and (2) interleave block-wise matmul and softmax operations, and (3) block quantization and incoherent processing that leverages hardware support for FP8 low-precision." That is the book's list on p. 504.
- Repository README today: "FlashAttention-2 with CUDA currently supports: 1. Ampere, Ada, or Hopper GPUs ... 2. Datatype fp16 and bf16 ... 3. All head dimensions up to 256." FA3: "FlashAttention-3 is optimized for Hopper GPUs (e.g. H100). ... This is a beta release ... Currently released: FP16 / BF16 forward and backward, FP8 forward ... Requirements: H100 / H800 GPU, CUDA >= 12.3." New since the book: "FlashAttention-4 is written in CuTeDSL and optimized for Hopper and Blackwell GPUs (e.g. H100, B200)", installed as `pip install flash-attn-4`; releases are tagged `fa4-v4.0.0.betaNN`, the latest `fa4-v4.0.0.beta40` on 2026-11-18.
- NVIDIA's library route, cuDNN: "This operation computes the scaled dot product attention (SDPA) ... using the FlashAttention-2 algorithm as described in the paper FlashAttention-2 ... It is applicable for both training and inference phases". "cudnn SDPA operation requires SM80 (Ampere) or newer architectures and cuda toolkit 12.x or newer." The support matrix lists fp16 and bf16 on Ampere/Ada, and fp8, fp16, bf16 on Hopper, for prefill and decode, with a "Paged" column. No fp32 row: the book's FP32 kernel has no cuDNN counterpart at the same precision (our reading of the matrix).
- No FlashAttention version, and no library, uses the book's FP32 single-head layout; the book says so itself (p. 495).

- https://github.com/Dao-AILab/flash-attention/blob/main/README.md
- https://github.com/Dao-AILab/flash-attention/releases
- https://arxiv.org/abs/2205.14135, https://arxiv.org/abs/2307.08691, https://arxiv.org/abs/2407.08608
- https://docs.nvidia.com/cudnn/operations/Attention

### A7. What the book's kernel shares with FlashAttention-2, and what it does differently (pp. 497 to 504; Exercise 3)

**Verdict: PARTLY.** The work split, the deferred division and the initial values are FlashAttention-2's. Two things are not: the book writes the row denominator `D` "for reuse during training" where FlashAttention-2 stores the log-sum-exp, and the book computes every tile, including tiles wholly above the diagonal.

- Q across warps (footnote 7, p. 498): FA2 section 3.3: "For each block, FlashAttention splits K and V across 4 warps while keeping Q accessible by all warps. ... This is referred to as the "split-K" scheme. However, this is inefficient since all warps need to write their intermediate results out to shared memory, synchronize, then add up the intermediate results. ... In FlashAttention-2, we instead split Q across 4 warps while keeping K and V accessible by all warps." CONFIRMED.
- Deferred normalisation (the notes' `Õ` form, Fig. 20.16): FA2 section 3.1.1: "We can instead maintain an "un-scaled" version of O(2) and keep around the statistics ℓ(2) ... Only at the every end of the loop do we scale the final Õ(last) by diag(ℓ(last))^-1 to get the right output." CONFIRMED.
- Initial values (Exercise 3): FA2 Algorithm 1, line 4: "On chip, initialize O_i(0) = (0) ∈ R^{B_r×d}, ℓ_i(0) = (0) ∈ R^{B_r}, m_i(0) = (−∞) ∈ R^{B_r}." This settles the notes' open question on `m_i`: minus infinity (or a finite stand-in, A5).
- What is stored for training: FA2: "We do not have to save both the max m(j) and the sum of exponentials ℓ(j) for the backward pass. We only need to store the logsumexp L(j) = m(j) + log(ℓ(j))." Algorithm 1 returns "the output O and the logsumexp L". The book writes only `out_D`, the denominator relative to a final row max that it never writes, so the stored value alone cannot rebuild P (our reading; the notes record that `store_O` does not write `m`). Correct statement for the lesson: store `m + log(D)`.
- Causal masking: FA2 section 3.1.1: "for any blocks where all the column indices are more than the row indices (approximately half of the blocks for large sequence length), we can skip the computation of that block." and "for each row, we only need apply causal mask to 1 block (assuming square block)". (The paper's second point says the mask is unnecessary for blocks whose row indices are "strictly less than the column indices"; from the first point, "greater" is meant.) The book's loop runs all `T_c` tiles for every panel. llama.cpp's Metal kernel classifies blocks the same way FA2 does (D2).
- Backward pass (p. 493): FA2: "there are 2 matmuls in the forward pass and 5 matmuls in the backward pass, due to recomputation". CONFIRMED.

- https://arxiv.org/abs/2307.08691 (sections 3.1.1, 3.3, Algorithm 1, section 4)

### A8. Beyond the book: split-KV decode ("Flash-Decoding") and paged KV caches in the FlashAttention API

**Verdict: CONFIRMED.** Both are in FlashAttention-2's public API and are what "production kernels add" for the M7 lesson.

- Split-KV for one query: FA changelog "2.2: Optimize for inference": "Optimize for inference (iterative decoding) when query has very small sequence length (e.g., query sequence length = 1). The bottleneck here is to load KV cache as fast as possible, and we split the loading across different thread blocks, with a separate kernel to combine results."
- The authors' write-up: "Our new approach Flash-Decoding is based on FlashAttention, and adds a new parallelization dimension: the keys/values sequence length." "First, we split the keys/values in smaller chunks. We compute the attention of the query with each of these splits in parallel using FlashAttention. We also write 1 extra scalar per row and per split: the log-sum-exp of the attention values. Finally, we compute the actual output by reducing over all the splits, using the log-sum-exp to scale the contribution of each split." The same post explains why: "During inference, the query length is typically 1: this means that if the batch size is smaller than the number of streaming multiprocessors (SMs) on the GPU (108 for an A100), the operation will only use a small part of the GPU!" (A100 figure is theirs, for the lesson only as their example.)
- API: `flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, ..., cache_seqlens=None, ..., block_table=None, ..., num_splits=0, ...)`. Docstring: "If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from k and v. This is useful for incremental decoding"; "num_splits: int. If > 1, split the key/value into this many chunks along the sequence. If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic to automatically determine the number of splits."
- Paged cache: changelog "2.5: Paged KV cache. Support paged KV cache (i.e., PagedAttention)." `k_cache` is "(num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)", `block_table` is "(batch_size, max_num_blocks_per_seq), dtype torch.int32", and "page_block_size must be a multiple of 256".
- GQA in the same call: "Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads than Q. ... if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head 0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V."
- Causal alignment when the query is shorter than the keys (needed for the decode step): changelog "2.1: Change behavior of causal flag. If seqlen_q != seqlen_k and causal=True, the causal mask is aligned to the bottom right corner of the attention matrix, instead of the top-left corner." PyTorch's `is_causal` does the opposite (D5).

- https://github.com/Dao-AILab/flash-attention/blob/main/README.md (sections "How to use FlashAttention", "Changelog" 2.1, 2.2, 2.5)
- https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py (`flash_attn_with_kvcache` docstring, lines 1540 to 1590)
- https://crfm.stanford.edu/2023/10/12/flashdecoding.html (also https://pytorch.org/blog/flash-decoding/)

---

## Group B: HIP / ROCm

### B1. The book's attention kernels compile under HIP unchanged

**Verdict: WRONG.** Three textual changes are needed, all settled elsewhere:

- `__shfl_sync(0xFFFFFFFF, ...)` (Figs. 20.12, 20.15) fails HIP's `static_assert` that the mask be a 64-bit integer (infra B3, ch10 B1). Use a typed mask (`lane_mask_t`, llama.cpp's `ggml_cuda_thread_mask_t`, ch05 B9).
- `cub::BlockReduce` and `cub::WarpReduce` are `hipcub::` in `<hipcub/hipcub.hpp>`, and the functor spelling differs by toolkit on both sides (infra A11, B11). The labs write the reductions by hand, as M4 does.
- The launch configuration and `dim3` behave as on CUDA (A1).

And one change that compiles but gives wrong answers on 64-wide wavefronts: B2.

### B2. The book's 32-lane layout (`WARP_SIZE 32`, `laneIdx() = threadIdx.x % 32`) on a 64-wide wavefront

**Verdict: WRONG** (our reading of AMD's source; nothing was run on AMD hardware). With `WARP_SIZE` 32 hard-coded, a gfx9 wavefront holds two of the book's "warps", and both the Q broadcast and the CUB reduction then cross between them.

- HIP's shuffle at `therock-10.2` (`amd_warp_functions.h`, line 151): `__device__ inline int __shfl(MAYBE_UNDEF int var, int src_lane, int width = warpSize) { int self = __lane_id(); int index = (src_lane & (width - 1)) + (self & ~(width - 1)); ...`. With the default `width = warpSize` = 64, hardware lane 40 (the book's lane 8 of logical warp 1) asking for source lane 5 reads hardware lane 5, which belongs to logical warp 0. Passing `width = 32` gives `5 + 32 = 37`, the intended lane. `__shfl_sync` forwards the same `width` parameter (`amd_warp_sync_functions.h`, line 276).
- hipCUB's default logical warp: `template<typename T, int LOGICAL_WARP_THREADS = HIPCUB_DEVICE_WARP_THREADS, ...> class WarpReduce` (`backend/rocprim/warp/warp_reduce.hpp`, line 44), with `#define HIPCUB_DEVICE_WARP_THREADS ::rocprim::arch::wavefront::max_size()` on the rocPRIM backend (`config.hpp`, line 107; the comment above it reads "HIPCUB (and CUB) don't have a method to express min and max warp size."). In rocPRIM's `arch.hpp`, `max_size()` returns `min_size()` in a device compile, and `min_size()` returns 32 under `ROCPRIM_NAVI` and 64 otherwise. So `hipcub::WarpReduce<float>` on CDNA reduces 64 lanes, two rows of the book's S tile at once.
- Two consistent fixes (ours): make the width a compile-time parameter equal to the wavefront (64 gives `N_WARPS` 8, `B_r_warp` 4, `d_size` 2 for the book's 512-thread block and d = 128), or keep 32-lane logical warps and pass `width = 32` to every shuffle and `LOGICAL_WARP_THREADS = 32` to every CUB-style reduction. ch10 B2 has AMD's recommended way to choose a width at compile time.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_warp_functions.h (lines 151 to 161)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h (line 276)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipcub/hipcub/include/hipcub/backend/rocprim/warp/warp_reduce.hpp (lines 42 to 49)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipcub/hipcub/include/hipcub/config.hpp (lines 103 to 109)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocprim/rocprim/include/rocprim/intrinsics/arch.hpp (lines 52 to 111)

### B3. The book's shared memory, block size and padding on AMD

**Verdict: PARTLY.** Size and block shape are fine; the padding constant is tuned for 32 banks; whether a `__shared__` array may be declared inside a `__device__` function is not stated by AMD (COULD NOT VERIFY from documentation).

- 37,376 bytes is far below every LDS size in AMD's table, and a block may use the whole LDS of a CU (ch08 B3). 512 threads is within HIP's default 1024-thread compile target (ch08 B4).
- `addr(x) = x + (x >> 5)` assumes 32 banks; AMD has 32 banks on CDNA 1 to 3 and 64 on CDNA 4 and RDNA 2 to 4, and services a wavefront in several cycles (infra B10). The padding has to be re-measured, not carried over.
- `__shared__` in a device function: AMD's language-extensions page says only "Shared memory can be allocated statically within the kernel, but the size of it has to be known at compile time." Nothing for or against device-function scope. Declaring the tiles in the kernel and passing pointers works on all three backends and is required on Metal (C2).

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/how-to/hip_cpp_language_extensions.rst (section `__shared__`, lines 249 to 264)

### B4. The warp barrier A3 asks for, on HIP

**Verdict: CONFIRMED** from earlier files. `__syncwarp()` and a masked `__syncwarp(MaskT)` (64-bit mask) are declared in `amd_warp_sync_functions.h` (infra B3, ch04 B4); AMD documents lockstep wavefronts (ch04 B5). llama.cpp's `ggml_cuda_syncwarp()` calls `__syncwarp()` only on non-HIP builds and is empty on HIP (ch10 D1), and its attention kernel calls it at exactly the point A3 identifies (D1). Write the call in the `.cu` file; it compiles on both.

### B5. Attention libraries on AMD (the counterpart of A6)

**Verdict: CONFIRMED.**

- FlashAttention README, "AMD ROCm Support": "ROCm version has two backends. There is composable_kernel (ck) which is the default backend and a Triton backend. They provide an implementation of FlashAttention-2." Requirements "ROCm 6.0 and above". CK backend: "MI200x, MI250x, MI300x, MI355x, and RDNA 3/4 GPUs", "Datatype fp16 and bf16", head dimensions up to 256. Triton backend: "supports AMD's CDNA (MI200, MI300) and RDNA GPUs using fp16, bf16, and fp32 datatypes. It provides forward and backward passes with causal masking, variable sequence lengths, arbitrary Q/KV sequence lengths and head sizes, MQA/GQA, dropout, rotary embeddings, ALiBi, paged attention, and FP8"; its kernels come from the `aiter` package.
- AMD's own ROCm AI page: "Flash Attention supports two backend implementations on AMD GPUs. Composable Kernel (CK) - the default backend; OpenAI Triton - an alternative backend", switched with `FLASH_ATTENTION_TRITON_AMD_ENABLE`.
- PyTorch on ROCm: `aten/src/ATen/native/transformers/hip/` holds `aotriton_adapter.h` and `flash_attn/{aot,ck}`; `cmake/External/aotriton.cmake` pins `set(__AOTRITON_VER "0.15b")`, and its package list includes an entry commented "rocm7.17 ROCM 10.2".

- https://github.com/Dao-AILab/flash-attention/blob/main/README.md (section "AMD ROCm Support")
- https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/model-acceleration-libs.html (section "Flash Attention 2")
- https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/transformers/hip
- https://github.com/pytorch/pytorch/blob/main/cmake/External/aotriton.cmake

---

## Group C: Metal

### C1. The book's tile sizes (d = 128, B_r = B_c = 32: 37,376 bytes of tiles) fit in a Metal threadgroup

**Verdict: WRONG.** The documented limit is 32 KB on Apple4 to Apple11 (ch05 C1), and 37,376 > 32,768.

- Our compile test: the Metal front end accepts a kernel with the three threadgroup arrays at the book's sizes (`KT[32*128 + (32*128 >> 5)]`, `S[32][32]`, `V[32][128]`); the limit is enforced when the pipeline is built. Our run, creating the pipeline on this Mac: "Threadgroup memory size (37376) exceeds the maximum threadgroup memory allowed (32768)", with `maxThreadgroupMemoryLength` reported as 32768. This is behaviour on one machine, consistent with the feature tables; lessons read the reader's value from the reports.
- Sizes that fit, from Group F: d = 128 with B_c = 16 (18,688 bytes), or d = 64 with B_c = 32 (20,736 bytes). The lab should derive B_c from the device limit, not hard-code it.
- The 512-thread block is within the tables' 1024, but the real cap is per pipeline (`maxTotalThreadsPerThreadgroup`, ch05 C3, ch08 C1); check it after building the pipeline.

- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf (GPU family limits)
- https://developer.apple.com/documentation/metal/mtldevice/maxthreadgroupmemorylength

### C2. The book's per-function `__shared__` declarations (Figs. 20.12, 20.15) port to Metal as written

**Verdict: WRONG.** MSL allows threadgroup variables only in kernel, mesh and object functions; a helper must receive a `threadgroup` pointer.

- MSL spec section 4.4: "Use the threadgroup address space to: Allocate a threadgroup variable in a kernel, mesh, or object function. Define a kernel, fragment, or object function parameter that's a pointer to a threadgroup address." Section 4: "(The threadgroup attribute is supported here for the pointer l_data only if foo is called by a kernel function, as detailed in section 4.4.)"
- Our compile test of a helper that declares `threadgroup float buf[32];`: "error: variables in the threadgroup address space cannot be declared in a non-qualified function". The same code with the array declared in the kernel and passed as `threadgroup float*` compiles.
- So the Metal lab declares `KT_j`, `S_i`, `V_j` and any reduction scratch at kernel scope and passes pointers. Doing the same in the `.cu` file keeps the three sources parallel.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 4 and 4.4, pages 119 to 121)

### C3. Row max, row sum and the Q broadcast with SIMD-group functions

**Verdict: CONFIRMED** (infra C2, C5; ch10 C1). The mapping is ours.

- Row max and sum of a segment (`compute_S_and_max`, `compute_P_and_update_D`): `simd_max(x)` and `simd_sum(x)` return the result to every active lane, so the book's lane-0 result plus `__shfl_sync(..., 0)` broadcast collapses to one call. Float is allowed (CUDA's `__reduce_*_sync` is integer only).
- Q broadcast (`__shfl_sync(mask, Q_i[ii][dd / 32], dd % 32)`): the source lane `dd % 32` is the same for every lane, so `simd_broadcast(value, lane)` fits; the spec leaves it undefined when the lane id "isn't a valid SIMD lane ID or isn't the same for all threads in a SIMD-group". `simd_shuffle(value, lane)` also works and allows per-lane sources.
- Lane and group come from `[[thread_index_in_simdgroup]]` and `[[simdgroup_index_in_threadgroup]]`, not from `thread_index / 32` (ch04 C2). The width is `[[threads_per_simdgroup]]`, undocumented as a number; the book's interleaved register layout needs 32, so the lab must check the pipeline's width at run time and fail clearly if it differs. llama.cpp assumes 32 (`N_SIMDWIDTH`, ch04 D2).

### C4. The warp barrier A3 asks for, on Metal

**Verdict: CONFIRMED.** `simdgroup_barrier(mem_flags::mem_threadgroup)` is the counterpart of `__syncwarp()` (infra C4); whether Apple GPUs keep a SIMD-group in lockstep is undocumented (ch04 C2), so the call belongs in the Metal kernel too. llama.cpp's Metal attention kernel uses `simdgroup_barrier(mem_flags::mem_threadgroup)` between threadgroup-memory phases (`fa_common.metal`, for example lines 367 and 388).

### C5. Minus infinity for the causal mask and the running max on Metal

**Verdict: PARTLY.** The hazard is documented (ch10 C3): under fast math the behaviour of INF and NaN is undefined, and the default math mode is documented three ways (ch05 C7). The book's kernel stores minus infinity for masked scores and starts `m_i` at minus infinity.

- What the two Metal code bases do: llama.cpp starts the running max at a finite value, `float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };` (`fa_common.metal`, line 171), and keeps the mask as an additive tensor. MLX's vector kernel starts at `U max_score = Limits<U>::finite_min;` and tests mask values with `fmask[0] >= Limits<T>::finite_min` (`sdpa_vector.h`, lines 122 and 140), and MLX builds its metallib with `-fno-fast-math` (ch05 C10).
- Safe choices for the lab (ours, as in ch10 C3): skip masked columns by index (the book's loop-bound trick from Fig. 20.4) instead of writing minus infinity, start the running max at a finite sentinel, or set relaxed or safe math for the kernel and test it.

### C6. Compile-time head dimension and tile sizes on Metal

**Verdict: CONFIRMED.** Two documented routes: preprocessor macros through `MTLCompileOptions.preprocessorMacros` (ch05 C7, what the project's lab runtime uses), or function constants.

- MSL spec section 5.8: "Function constants enable the generation of multiple variants of a function. Without using function constants, you can compile one function many times with different preprocessor macro defines to enable different features (an ubershader). ... Function constants provide the same ease of use as preprocessor macros but moves the generation of the specific variants to the creation of the pipeline state". Declared as `[[function_constant(index)]]`, "index needs to be between 0 and 65535".
- llama.cpp's Metal attention uses function constants for its sizes, for example `constant int32_t FC_flash_attn_ext_vec_reduce_DV [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 0)]];` (`fa_aux.metal`, line 361).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 5.8, page 190)

### C7. A library attention on Apple GPUs (the counterpart of cuDNN SDPA and CK)

**Verdict: CONFIRMED.** MPSGraph has a fused scaled-dot-product-attention operation.

- `func scaledDotProductAttention(query queryTensor: MPSGraphTensor, key keyTensor: MPSGraphTensor, value valueTensor: MPSGraphTensor, mask maskTensor: MPSGraphTensor?, scale: Float, name: String?) -> MPSGraphTensor`, macOS 15.0 (iOS 18.0): "SDPA Op computes attention by computing softmax(scale * QK^T + M)V. queryTensor Q with shape [B, Hq, Nq, F] and keyTensor K with shape [B, Hq, Nkv, F], with Q's H dimension expandable to satisfy matmul QK^T."
- New in macOS 27.0: `scaledDotProductAttention(query:key:value:descriptor:name:)`, "A descriptor specifying scale and optional features (mask, isCausal, sinks)".
- Apple's parameter text has two slips: the mask is "applied to the scaled, matrix multiplied query and value matrices" and the scale "is applied to the result of query and value matrix multiply"; both mean query and key, as the formula in the same page shows.
- Apple does not document the algorithm (tiling, online softmax) behind it.

- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/scaleddotproductattention(query:key:value:mask:scale:name:)
- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/scaleddotproductattention(query:key:value:descriptor:name:)

---

## Group D: How llama.cpp, MLX and PyTorch do this chapter's kernels

Read on the default branches on 2026-11-20. Line numbers are a snapshot.

### D1. llama.cpp, CUDA and HIP: the online softmax, the warp barrier, split-KV and GQA

**Verdict: CONFIRMED.** The book's (m, D, O) recurrence and the Flash-Decoding merge are both in `ggml/src/ggml-cuda/`, compiled for HIP from the same files (ch05 B9).

- The per-tile update in the vector kernel (`fattn-vec.cuh`, lines 321 to 343), the book's Figs. 20.13 to 20.15 in five lines:

```cpp
const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
KQ_max[j] = KQ_max_new[j];

KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
KQ[j*nthreads + tid] = KQ_reg[j];
// ... every VKQ accumulator is multiplied by KQ_max_scale ...
ggml_cuda_syncwarp(); // Needed for KQ synchronization.
```

- Start values: `KQ_max[j] = -FLT_MAX/2.0f; KQ_sum[j] = 0.0f;` (lines 138 to 139). Numerical guards the book does not have: `#define SOFTMAX_FTZ_THRESHOLD -20.0f // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.` and `FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)`, which shifts the max by log 8 "to fix https://github.com/ggml-org/llama.cpp/issues/18606" (`fattn-common.cuh`, lines 9 to 19).
- Split-KV: the launcher computes `parallel_blocks` from occupancy and the number of KV tiles (`fattn-common.cuh`, lines 1201 to 1285) and, when it is above 1, writes partial outputs plus a `float2` (max, row sum) per split, merged by `flash_attn_combine_results` (lines 975 to 1027):

```cpp
float kqmax = meta[0].x;
for (int l = 1; l < parallel_blocks; ++l) { kqmax = max(kqmax, meta[l].x); }
float VKQ_numerator = 0.0f, VKQ_denominator = 0.0f;
for (int l = 0; l < parallel_blocks; ++l) {
    const float KQ_max_scale = expf(meta[l].x - kqmax);
    VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
    VKQ_denominator += KQ_max_scale * meta[l].y;
}
dst[tid] = VKQ_numerator / VKQ_denominator;
```

  That is Eq. 20.4 and Eq. 20.6 (in the un-normalised form) applied across thread blocks instead of across tiles.
- GQA: `const int gqa_ratio = Q->ne[2] / K->ne[2];` (line 1152); several Q heads are processed against one loaded K/V head.
- The mask is an additive half-precision tensor read per score (`sum += slope*__half2float(maskh[j*ne11 + i_KQ]);`, `fattn-vec.cuh`), built on the host (D3).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-vec.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-common.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu

### D2. llama.cpp, Metal: SIMD-group online softmax, skipped tiles, and the split reduce

**Verdict: CONFIRMED.**

- Per-tile update (`fa_common.metal`, lines 433 to 452): `M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));`, `const float ms = exp(m - M[jj]);`, `const float2 vs2 = exp(s2 - M[jj]);`, `S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);`, then "the P matrix from the paper (Q rows, C columns)" is written to threadgroup memory and every output element is multiplied by `ms`. The two matrix products use `simdgroup_multiply_accumulate` on 8 by 8 SIMD-group matrices (infra C10).
- Tiles are classified before the main loop (`fa_aux.metal`, lines 197 to 201): "scan the blocks of the mask that are not masked // 0 - masked (i.e. full of -INF, skip) // 1 - not masked (i.e. at least one element of the mask is not -INF) // 2 - all zero". Class 2 skips adding the mask (`if (blk_cur != 2)`). This is FA2's two causal-mask tweaks (A7).
- Split reduce for the vector kernel, `kernel_flash_attn_ext_vec_reduce` (`fa_aux.metal`, lines 364 to 404): each lane holds one split's (S, M); `const float m = simd_max(M); const float ms = exp(M - m); S = simd_sum(S*ms);` then the partial outputs are summed with the same factor and divided by S. The same merge as D1, written with SIMD-group reductions.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/fa_common.metal
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/fa_aux.metal

### D3. The KV cache as three code bases store it (bridge for §20.4 and M7)

**Verdict: CONFIRMED.**

- Meta's reference Llama code (`llama/model.py`): the cache is preallocated to the maximum, `self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))`, the same for V; rotary embedding is applied to Q and K before K is cached (`xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)` then `self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk`); GQA repeats K/V heads (`keys = repeat_kv(keys, self.n_rep)`); and the mask exists only when more than one token is processed (`mask = None` / `if seqlen > 1:` ... `torch.triu(mask, diagonal=1)` with zeros prepended for the cached positions). So a one-token decode step has no mask, as Fig. 20.6 shows. Defaults are the 7B shape: `dim: int = 4096`, `n_layers: int = 32`, `n_heads: int = 32`, `n_kv_heads: Optional[int] = None`.
- llama.cpp (`src/llama-kv-cache.cpp`): one K and one V tensor per layer, `ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)` (row = one token's K across the KV heads); new rows are written by slot index with `ggml_set_rows(ctx, k, k_cur, k_idxs)`; the header calls the slots "the ring buffer of KV cells" and says "token[i] -> goes to cells[idxs[i]]". There is no block table: not paged. For MLA models no V tensor is allocated (`const bool has_v = !is_mla;`), which is the absorption of E5. The causal mask is a host-built additive tensor with `mask_drop = -INFINITY`.
- llama.cpp cache precision (`common/arg.cpp`): `-ctk/--cache-type-k` accepts `f32, f16, bf16, f8_e4m3, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`, default `GGML_TYPE_F16` (`common/common.h`). `block_q8_0` is a half scale plus 32 `int8_t` (34 bytes per 32 values), `block_q4_0` a half scale plus 16 bytes of nibbles (18 bytes per 32) (`ggml-common.h`). These are the real "bytes per element" choices for the explorer: 4, 2, 1, 34/32 and 18/32.
- mlx-lm (`mlx_lm/models/cache.py`, `class KVCache`): shape `(B, n_kv_heads, n_steps * self.step, head_dim)`, grown in chunks of `step = 256` by concatenating zeros, written with `self.keys[..., prev : self.offset, :] = keys`, and read back as the first `offset` positions.

- https://github.com/meta-llama/llama/blob/main/llama/model.py (lines 20 to 31, 164 to 172, 236 to 301, 470 to 487)
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp (lines 231 to 234, 1514, 1719)
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.h (lines 33, 323)
- https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp (lines 281 to 292, 2421 to 2426)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h (lines 194 to 199, 251 to 256)
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py

### D4. MLX: separate decode and prefill kernels on Metal

**Verdict: CONFIRMED.**

- Dispatch (`mlx/backend/metal/scaled_dot_product_attention.cpp`): `else if (q_pre.shape(2) <= 8) {` selects the vector kernels (line 1820). Among them, the two-pass (split-KV) kernel is chosen when "The device is large and the sequence length long" or "The sequence length is even longer and we have gqa" (lines 1885 to 1896): `if (((devc == 'd' || devc == 's') && n_keys >= 1024) || (k.shape(1) < q.shape(1) && n_keys >= 4096)) { sdpa_vector_2pass(...); }`. Longer queries go to the tiled "steel" attention (`sdpa_full_self_attention_metal`, line 1929). So MLX has the book's prefill kernel and the chapter's missing decode kernel as separate code.
- Vector kernel (`kernels/sdpa_vector.h`): the score is a `simd_sum` over lanes (line 153), then `U new_max = max(max_score, score); U factor = fast::exp(max_score - new_max); U exp_score = fast::exp(score - new_max); max_score = new_max; sum_exp_score = sum_exp_score * factor + exp_score;` (lines 159 to 164), then a cross-SIMD-group merge through threadgroup memory with `simd_max` and `simd_sum` (lines 187 to 201). Start value `Limits<U>::finite_min` (C5).
- Documentation, `mlx.core.fast.scaled_dot_product_attention`: "Supports: Multi-Head Attention, Grouped Query Attention, Multi-Query Attention"; "The softmax operation is performed in float32 regardless of the input precision."; "For Grouped Query Attention and Multi-Query Attention, the k and v inputs should not be pre-tiled to match q."; "The "causal" mask uses lower-right alignment where the last query aligns with the last key."

- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/scaled_dot_product_attention.cpp
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html

### D5. PyTorch: `scaled_dot_product_attention`, its causal alignment, and what runs on MPS

**Verdict: PARTLY.** The API is documented and its reference code is the chapter's Eq. 20.1. Two things to teach carefully: `is_causal` aligns the mask to the upper left when the query is shorter than the keys, and the documentation's "for all other backends" sentence is behind the MPS source.

- Reference semantics in the docs (2.15): `temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)` then `attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))`; GQA by `key.repeat_interleave(query.size(-3)//key.size(-3), -3)`; `attn_weight = query @ key.transpose(-2, -1) * scale_factor`.
- `is_causal`: "If set to true, the attention masking is a lower triangular matrix when the mask is a square matrix. The attention masking has the form of the upper left causal bias due to the alignment (see torch.nn.attention.bias.CausalBias) when the mask is a non-square matrix." `CausalVariant`: "UPPER_LEFT ... with shape=(3,4) ... [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]" and "LOWER_RIGHT: Represents lower-right triangular bias, the include values are aligned to the lower right corner of the matrix." So a one-query decode step against N cached keys with `is_causal=True` lets the query see only key 0. Use no mask for a single query (as the Llama reference does, D3) or `causal_lower_right`. FlashAttention (since 2.1) and MLX align lower-right (A8, D4).
- Implementations: "There are currently three supported implementations of scaled dot product attention: FlashAttention-2 ..., Memory-Efficient Attention, A PyTorch implementation defined in C++". GQA: "an experimental feature. It works with FlashAttention, cuDNN attention, and the math kernel on CUDA tensors."
- The docs also say "The function may call optimized kernels for improved performance when using the CUDA backend. For all other backends, the PyTorch implementation will be used." The MPS source has its own Metal kernels: `aten/src/ATen/native/mps/operations/Attention.mm` routes to `sdpa_vector_2pass_mps` (long caches), `sdpa_vector_fast_mps` (short queries) or `sdpa_prefill_mps` (lines 403 to 410), and `kernels/Attention.metal` begins "Largely influeneced by https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scaled_dot_product_attention.metal" and includes `DecodeAttention.h`, `MppAttention.h`, `PrefillAttention.h`. On ROCm, see B5.

- https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://docs.pytorch.org/docs/2.15/generated/torch.nn.attention.bias.CausalVariant.html
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Attention.mm
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/Attention.metal

---

## Group E: The papers, models and serving systems the chapter cites

### E1. Timeline and references (pp. 477 to 478, 511 to 512)

**Verdict: PARTLY.** Every dated fact checked holds; the ChatGPT URL of reference [3] now redirects.

- "Attention Is All You Need", arXiv 1706.03762, first version 2017-06-12. BERT, arXiv 1810.04805, 2018-10-11. Mamba, arXiv 2312.00752 (2023-12-01). Hyena Hierarchy, arXiv 2302.10866 (2023-02-21). Online normalizer calculation for softmax (reference [7]), Milakov and Gimelshein, NVIDIA, arXiv 1805.02867 (2018). PagedAttention / vLLM: arXiv comment "SOSP 2023". Speculative decoding (Leviathan, Kalman, Matias): "ICML 2023 Oral". GQA: "Accepted at EMNLP 2023". DeepSeek-V2: arXiv 2405.04434 (2024). Mixtral: arXiv 2401.04088 (2024). Reference [13]: Zadouri, Strauss, Dao, "Hardware-Efficient Attention for Fast Decoding", arXiv 2505.21487 (2025-05-27).
- `https://chat.openai.com/` answers "HTTP/2 308" with "location: https://chatgpt.com/".

- https://arxiv.org/abs/1706.03762, https://arxiv.org/abs/1810.04805, https://arxiv.org/abs/2312.00752, https://arxiv.org/abs/2302.10866, https://arxiv.org/abs/1805.02867, https://arxiv.org/abs/2309.06180, https://arxiv.org/abs/2211.17192, https://arxiv.org/abs/2305.13245, https://arxiv.org/abs/2405.04434, https://arxiv.org/abs/2401.04088, https://arxiv.org/abs/2505.21487

### E2. The three KV-cache examples on p. 506 (PaLM 2, GPT-3, Llama 2 7B)

**Verdict: PARTLY.** The arithmetic of Eq. 20.8 is right for the numbers printed (F1). The PaLM 2 row is WRONG: Google never published PaLM 2's shape, and the numbers are those of PaLM 62B, which does not use multi-head attention. The GPT-3 row uses a sequence twice GPT-3's context window.

- PaLM 2 report, model card: "We have small, medium, and large variants that use stacked layers based on the Transformer architecture, with varying parameters depending on model size. Further details of model size and architecture are withheld from external publication."
- The first PaLM paper, Table 1: "PaLM 62B | 64 | 32 | 8192" (layers, heads, d_model), and "attention head size is always 256". Its architecture list: "Multi-Query Attention ... the key/value projections are shared for each head, i.e. "key" and "value" are projected to [1, h], but "query" is still projected to shape [k, h]. ... results in a significant cost savings at autoregressive decoding time." With one K/V head of 256, a 4096-token 16-bit cache is 256 MiB, not 8 GiB (32 times smaller; F1).
- GPT-3, Table 2.1: "GPT-3 175B or "GPT-3" | 175.0B | 96 | 12288 | 96 | 128" (layers, d_model, heads, d_head): CONFIRMED. But "All models use a context window of n_ctx = 2048 tokens." A 4096-token GPT-3 cache does not occur; at 2048 tokens it is 9 GiB.
- Llama 2 7B: the Llama 2 paper says "We adopt most of the pretraining setting and model architecture from Llama 1" and its Table 1 lists 7B with "4k" context and no GQA ("Bigger models — 34B and 70B — use Grouped-Query Attention"); Llama 1's table gives "6.7B | 4096 | 32 | 32" (dimension, heads, layers); the reference code's defaults are the same (D3). 2 GiB at 4096 tokens: CONFIRMED.
- For the explorer, two published shapes that show GQA and MLA (from the papers, not the book): Mistral 7B, Table 1: "dim 4096, n_layers 32, head_dim 128, hidden_dim 14336, n_heads 32, n_kv_heads 8, window_size 4096, context_len 8192" (and a "Rolling Buffer Cache" bounded by the window). DeepSeek-V2: "We set the number of Transformer layers to 60 and the hidden dimension to 5120. ... we set the number of attention heads n_h to 128 and the per-head dimension d_h to 128. The KV compression dimension d_c is set to 512 ... For the decoupled queries and key, we set the per-head dimension d_h^R to 64."

- https://arxiv.org/abs/2305.10403 (Appendix D.10, Table 31)
- https://arxiv.org/abs/2204.02311 (section 2, Table 1)
- https://arxiv.org/abs/2005.14165 (section 2.1, Table 2.1)
- https://arxiv.org/abs/2307.09288 (section 2.2, Table 1), https://arxiv.org/abs/2302.13971 (Table 2)
- https://arxiv.org/abs/2310.06825 (Table 1), https://arxiv.org/abs/2405.04434 (section 3.1.2)

### E3. GQA "using a few groups (g_q) of query heads" with `h_q/g_q` K/V heads (pp. 508 to 509, Eqs. 20.12 and 20.13)

**Verdict: PARTLY.** The book's formulas use one convention and its prose another.

- The GQA paper: "Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head. GQA-G refers to grouped-query with G groups. GQA-1, with a single group and therefore single key and value head, is equivalent to MQA, while GQA-H, with groups equal to number of heads, is equivalent to MHA." Here G is the number of groups, which equals the number of K/V heads. DeepSeek-V2's Table 1 uses the same meaning ("n_g denotes the number of groups in GQA", cache "2 n_g d_h l").
- The book's source for the formulas, reference [13]: "We define the group size g_q as the number of query heads per distinct KV head"; Table 1 caption: "define the group size g_q = h_q / h_kv (queries per KV head)"; GQA intensity "≈ g_q".
- So Eqs. 20.12 and 20.13 are right with [13]'s g_q (K/V heads = h_q / g_q; intensity ≈ g_q), and the sentence "a few groups (g_q)" and "increases by the number of groups" is right only with the GQA paper's G. The lesson should define g_q once as queries per K/V head, and say that the GQA paper's G is h_q / g_q.

- https://arxiv.org/abs/2305.13245 (section 2.2)
- https://arxiv.org/abs/2505.21487 (section 2, Table 1)

### E4. The decode arithmetic-intensity estimates, Eqs. 20.9, 20.11, 20.13, 20.15 (pp. 506 to 510)

**Verdict: PARTLY.** MHA, MQA and GQA match the cited source. Eq. 20.15 (MLA) has the wrong denominator, as the notes suspected.

- Reference [13], section 3.1: "Every loaded BF16 (2 bytes) element of the cached key matrix performs one MAC (2 FLOPs) with the single-token query element already in registers, yielding a 1:1 FLOP-to-byte ratio." Table 1 (L = KV length): MHA "L / (1 + L)" ≈ 1; MQA "L h_q / (h_q + L)" ≈ h_q; GQA "L h_q / (h_q + (h_q/g_q) L)" ≈ g_q; MLA "L / (1 + L/(2 h_q))" ≈ 2 h_q; and "We assume L ≫ h_q."
- L / (1 + L/(2 h_q)) = 2 L h_q / (2 h_q + L). The book prints 2·N·h_q/(h_q + N); the limit 2 h_q is unchanged for N much larger than h_q (F1 has both values).
- Why MLA gets 2 h_q: "DeepSeek absorbs its low-rank MLA projection matrices during decoding and directly uses the single latent head to compute attention, effectively doubling the arithmetic intensity relative to MQA." and "The exact latent representation loaded into on-chip memory now serves both the key and the value". The book's own reasoning on p. 510 (decompress K and V before use) would not give this figure (E5).
- [13] also supports the book's speculative-decoding remark (p. 508): Figure 3, "Right, L_q = 2: e.g., in speculative decoding setting when query length is 2, for h_q: 128 climbs beyond the roof and becomes compute bound". The H100 figures in [13] ("∼295 FLOPs per byte (989 TFLOPs / 3.35 TB/s)") are theirs, for the book's example device only (ch05 A8).

- https://arxiv.org/abs/2505.21487 (sections 1, 3.1, 3.2, Table 1, Figure 3)

### E5. MLA: "each token's K and V vectors are compressed into one latent vector, across all heads and layers"; cache `b × N × l × d × p`; K and V "decompressed by two up-projection matrices just before use" (pp. 509 to 510)

**Verdict: PARTLY.** One latent per token per layer, shared by all heads: yes. Not shared across layers. Its size in DeepSeek-V2 is 4.5 head dimensions, not one. And DeepSeek's paper says the up-projections are absorbed, so K and V are not formed at inference.

- DeepSeek-V2 section 2.1.2: "c_t^KV ∈ R^{d_c} is the compressed latent vector for keys and values; d_c (≪ d_h n_h) denotes the KV compression dimension ... During inference, MLA only needs to cache c_t^KV, so its KV cache has only d_c l elements, where l denotes the number of layers." The factor l means one latent per layer.
- Section 2.1.3: "During inference, the decoupled key should also be cached. Therefore, DeepSeek-V2 requires a total KV cache containing (d_c + d_h^R) l elements." Table 1: MHA "2 n_h d_h l", GQA "2 n_g d_h l", MQA "2 d_h l", MLA "(d_c + d_h^R) l ≈ 9/2 d_h l"; "For DeepSeek-V2, d_c is set to 4 d_h and d_h^R is set to d_h/2. So, its KV cache is equal to GQA with only 2.25 groups".
- Absorption: "In addition, during inference, since W^UK can be absorbed into W^Q, and W^UV can be absorbed into W^O, we even do not need to compute keys and values out for attention." (The decoupled RoPE key exists because "W^UK cannot be absorbed into W^Q any more during inference, since a RoPE matrix" sits between them.) llama.cpp's cache follows this: no V tensor for MLA models (D3).
- So Eq. 20.14 with a latent of size d understates DeepSeek-V2's cache by 4.5 times (F1). The lesson should give MLA its own latent width (d_c + d_rope) in the explorer.

- https://arxiv.org/abs/2405.04434 (sections 2.1.2 to 2.1.4, Table 1)

### E6. PagedAttention "splits the KV cache into equal-size blocks that are loaded into GPU memory (from the larger system memory) when needed", and in-flight batching "slots a new request into the batch with the lowest workload" (pp. 506, 509)

**Verdict: PARTLY.** Equal-size blocks: yes. The point of the paper is where the blocks live in GPU memory (anywhere, through a block table), not loading them from host memory; host memory is a preemption fallback. The batching description is not the one in the cited paper or in NVIDIA's documentation.

- vLLM paper, abstract: "existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size. To address this problem, we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems." Section 4.1: "Unlike the traditional attention algorithms, PagedAttention allows storing continuous keys and values in non-contiguous memory space. Specifically, PagedAttention partitions the KV cache of each sequence into KV blocks. Each block contains the key and value vectors for a fixed number of tokens".
- Host memory, section 4.5: "Swapping ... we copy evicted blocks to the CPU memory. ... When vLLM exhausts free physical blocks for new tokens, it selects a set of sequences to evict and transfer their KV cache to the CPU." That is eviction under memory pressure, not the normal path.
- Batching in the same paper credits Orca: "fine-grained batching mechanisms, such as cellular batching [16] and iteration-level scheduling [60], have been proposed. ... After each iteration, completed requests are removed from the batch, and new ones are added."
- "In-flight batching" is NVIDIA's name: TensorRT-LLM, "TensorRT LLM supports in-flight batching of requests (also known as continuous batching or iteration-level batching) for higher serving throughput. With this feature, sequences in the context phase can be processed together with sequences in the generation phase." Its paged cache: "The paged KV cache decomposes the KV cache into blocks that are distributed to the different requests by a cache manager during processing."
- Today's vLLM: its PagedAttention design page is now marked "This is a historical document based on the original paper for vLLM. It no longer describes the code used in vLLM today." and "Paged attention is now implemented directly inside vLLM's attention backends (for example TritonAttentionImpl ..., or the FlashAttention-based backend) using block tables built by the scheduler's KV-cache manager". The FlashAttention API's `block_table` is that interface (A8).

- https://arxiv.org/abs/2309.06180 (abstract, sections 2.3, 4.1, 4.5)
- https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html
- https://docs.vllm.ai/en/latest/design/paged_attention/

### E7. Speculative decoding [14], Mixtral [17] and multi-query attention

**Verdict: CONFIRMED.**

- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding": "we introduce speculative decoding - an algorithm to sample from autoregressive models faster without any changes to the outputs, by computing several tokens in parallel." (ICML 2023.)
- Mixtral: "Mixtral has the same architecture as Mistral 7B, with the difference that each layer is composed of 8 feedforward blocks (i.e. experts). For every token, at each layer, a router network selects two experts to process the current state and combine their outputs. ... each token has access to 47B parameters, but only uses 13B active parameters during inference."
- MQA is Shazeer 2019, "Fast Transformer Decoding: One Write-Head is All You Need" (arXiv 1911.02150), whose abstract names the problem the chapter's §20.6 is about: "incremental inference ... is often slow, due to the memory-bandwidth cost of repeatedly loading the large "keys" and "values" tensors". The book describes MQA without citing it (notes, §20.7).

- https://arxiv.org/abs/2211.17192, https://arxiv.org/abs/2401.04088, https://arxiv.org/abs/1911.02150

### E8. §20.1's block (LayerNorm, ReLU feed-forward, positional encoding added to the embeddings; "hidden dimension = h_q × d") against today's open models

**Verdict: PARTLY.** The book describes the original transformer. The open models the reader will load differ in ways that touch the attention kernel and the KV cache.

- Llama 2: "We use the standard transformer architecture (Vaswani et al., 2017), apply pre-normalization using RMSNorm (Zhang and Sennrich, 2019), use the SwiGLU activation function (Shazeer, 2020), and rotary positional embeddings (RoPE, Su et al. 2022)." PaLM also lists "RoPE Embeddings" and multi-query attention (E2).
- Consequence for the kernels (from the reference code, D3): the rotary embedding is applied to Q and K inside each attention layer before K is cached, not added once to the embeddings, so a KV cache stores rotated keys.
- "Hidden dimension = h_q × d" holds for GPT-3 (96 × 128 = 12288), PaLM 62B (32 × 256 = 8192), Llama 2 7B and Mistral 7B (32 × 128 = 4096), and fails for DeepSeek-V2 (128 heads × 128 = 16384 against a hidden dimension of 5120, E2). Our arithmetic from the papers' tables.
- Mistral's sliding window bounds the cache: "A fixed attention span means that we can limit our cache size using a rolling buffer cache." A Beyond note at most.

- https://arxiv.org/abs/2307.09288 (section 2.2)
- https://github.com/meta-llama/llama/blob/main/llama/model.py
- https://arxiv.org/abs/2310.06825 (section 2)

---

## Group F: The chapter's arithmetic, rechecked

### F1. KV-cache sizes, intensities and tile footprints

**Verdict: CONFIRMED** for the book's arithmetic as printed; the corrections come from the model facts in E2 and E5 and the formula in E4. Our script, 16-bit elements, binary units.

| Quantity | Value |
|---|---|
| Eq. 20.8, book's "PaLM 2" (l = 64, hidden 8192, 4096 tokens) | 8.0 GiB as printed |
| Same shape with PaLM 62B's real multi-query attention (one K/V head of 256) | 256 MiB (32 times smaller) |
| GPT-3 (l = 96, 96 heads of 128), 4096 tokens / its real 2048-token window | 18.0 GiB / 9.0 GiB; 18 GiB is 19.33 × 10^9 bytes |
| Llama 2 7B, 4096 tokens | 2.0 GiB (512 KiB per token) |
| Mistral 7B (GQA, 8 K/V heads of 128), per token / 4096 tokens | 128 KiB / 512 MiB |
| DeepSeek-V2 MLA, per token: (512 + 64) × 60 layers × 2 bytes | 69,120 bytes = 67.5 KiB; the MHA equivalent (128 heads of 128) is 3.75 MiB, 56.9 times more |
| Book's Eq. 20.14 for DeepSeek-V2 (l × d × p, d = 128) | 15,360 bytes per token, 4.5 times below the paper's figure |
| Linear layer 1024 to 4096, 16-bit, batch 1 / 512 (p. 505) | 0.9988 / 315.08 FLOP/B |
| GQA intensity, N = 4096, h_q = 32, g_q = 1, 4, 8, 32 (queries per K/V head) | 1.000, 3.996, 7.984, 31.752 |
| MLA intensity, N = 4096, h_q = 32: [13]'s 2Nh_q/(2h_q + N) / book's 2Nh_q/(h_q + N) | 63.02 / 63.50 |
| Fig. 20.9 tiles, d = 128, B_r = B_c = 32, 32 banks: (4096 + 128 + 1024 + 4096) floats | 37,376 bytes |
| d = 128, B_r = 32, B_c = 16 / d = 64, B_r = B_c = 32 / d = 128, B_r = B_c = 16 | 18,688 / 20,736 / 17,664 bytes |
| IEEE: (−∞) − (−∞) | NaN (A5) |

---

## Summary of corrections

Every WRONG or PARTLY item, and every sub-claim that failed or could not be found:

- A1 (WRONG): unused grid dimensions are 1, not 0 ("In C++11 and later, the default value of all components of dim3 is 1."); HIP's `dim3` constructor defaults to 1 as well.
- A2: `cub::Max()` and `cub::Sum()` no longer exist (infra A11). CUB documents one `WarpReduce::TempStorage` per warp and a `__syncwarp()` before reuse; Figs. 20.12 and 20.15 share one storage among all 16 warps of the block. Current CUB says its warp reduction does not use shared memory, so the defect may not show; teach the documented form.
- A3 (WRONG): the flash kernel needs a `__syncwarp()` between the P write in `compute_P_and_update_D` and the cross-lane reads in `compute_O`; shuffles "do not guarantee any memory ordering". llama.cpp has the barrier at the same point.
- A5: `NEG_INFINITY` is not a CUDA identifier (none found; COULD NOT VERIFY any definition); use `-INFINITY` or `cuda::std::numeric_limits<float>::infinity()`. "Inf - Inf" is NaN, so a running max that is still minus infinity poisons the rescale; the book's left-to-right causal order avoids it, other orders and masks do not. Production kernels start from a finite value (`-FLT_MAX/2`, `finite_min`).
- A7: FlashAttention-2 stores the log-sum-exp `m + log(D)` for training; the book's `out_D` alone (without the row max) cannot rebuild P. FlashAttention-2 also skips tiles above the diagonal and masks only diagonal tiles; the book's kernel computes and masks every tile.
- B1 (WRONG): the listings do not compile under HIP unchanged (64-bit mask, `hipcub::`).
- B2 (WRONG): with `WARP_SIZE` 32 on a 64-wide wavefront, HIP's `__shfl` (default `width = warpSize`) reads from the wrong half, and `hipcub::WarpReduce<float>` defaults to 64 logical lanes on non-Navi targets, mixing two rows. Make the width a compile-time parameter equal to the wavefront, or pass width 32 everywhere.
- B3: `LOG_NUM_BANKS` 5 is NVIDIA's bank count; AMD has 32 or 64. AMD's docs describe static `__shared__` arrays only "within the kernel"; device-function scope is COULD NOT VERIFY. Declare tiles in the kernel.
- C1 (WRONG): the book's 37,376 bytes of tiles exceed Metal's 32 KB; the compiler accepts the kernel and pipeline creation fails ("Threadgroup memory size (37376) exceeds the maximum threadgroup memory allowed (32768)", our run). Shrink B_c or d from the device limit.
- C2 (WRONG): MSL rejects threadgroup arrays declared in helper functions ("variables in the threadgroup address space cannot be declared in a non-qualified function", our compile test); declare them in the kernel and pass pointers.
- C5: masked scores and a running max of minus infinity are undefined under Metal fast math (ch10 C3); llama.cpp and MLX use finite sentinels.
- C7 (side note): Apple's MPSGraph SDPA page says "query and value" where it means query and key, twice.
- D5: PyTorch's `is_causal` uses an upper-left mask when the query is shorter than the keys, so a one-token decode with `is_causal=True` attends to key 0 only; FlashAttention (2.1 and later) and MLX use lower-right. PyTorch's SDPA page says non-CUDA backends use the C++ reference, but the MPS source has its own decode and prefill Metal kernels derived from MLX.
- E1: the ChatGPT URL of reference [3] redirects (308) to `https://chatgpt.com/`.
- E2 (WRONG sub-claim): PaLM 2's architecture is "withheld from external publication"; the l = 64, hidden 8192 figures are PaLM 62B, which uses multi-query attention with head size 256, so its 4096-token cache is 256 MiB, not 8 GB. GPT-3's context window is 2048 tokens, not 4096 (9 GiB at 2048).
- E3: the book's GQA prose uses the GQA paper's G (number of groups = number of K/V heads), its formulas use [13]'s g_q (query heads per K/V head). Define g_q = h_q / h_kv.
- E4: Eq. 20.15's denominator is 2·h_q + N in the cited source ([13], Table 1: L / (1 + L/(2h_q))), not h_q + N.
- E5: the MLA latent is per layer (not "across all heads and layers"), its size in DeepSeek-V2 is d_c + d_h^R = 4.5 d_h (not d), and at inference the up-projections are absorbed into W^Q and W^O, so K and V are never formed; that absorption is what gives MLA its ≈ 2 h_q intensity.
- E6: PagedAttention stores fixed-size KV blocks non-contiguously in GPU memory through a block table; copying blocks to host memory is vLLM's eviction fallback. Iteration-level scheduling is Orca's; "in-flight batching" is TensorRT-LLM's name for it ("also known as continuous batching or iteration-level batching"); neither describes slotting requests by lowest workload. vLLM's own PagedAttention design page is now marked historical.
- E8: today's open models (Llama 2, PaLM, Mistral) use RMSNorm, SwiGLU and rotary embeddings applied to Q and K inside attention, so the cache holds rotated keys; "hidden = h_q × d" fails for DeepSeek-V2.

Path and status facts worth knowing even where the verdict is CONFIRMED:

- A6: FlashAttention-2 is now cited as ICLR 2024; FlashAttention-4 (CuTeDSL, Hopper and Blackwell) ships as `flash-attn-4` betas; cuDNN's SDPA implements the FlashAttention-2 algorithm on SM80 and later, in fp16, bf16 and (Hopper) fp8, with paged caches.
- A7: FlashAttention-2's Algorithm 1 starts from O = 0, ℓ = 0, m = −∞, which answers Exercise 3's open point.
- A8: split-KV decode (`num_splits`), paged caches (`block_table`, pages a multiple of 256) and GQA are all in `flash_attn_with_kvcache`; the causal mask aligns bottom-right since 2.1.
- B5: FlashAttention-2 runs on AMD through CK (default) or Triton (`aiter`), and PyTorch on ROCm uses AOTriton and CK.
- D1, D2: llama.cpp merges split-KV partial results by carrying (max, row sum) per split, exactly Eqs. 20.4 and 20.6, on CUDA, HIP and Metal.
- D3: three cache layouts the reader will meet: preallocated `[batch, max_seq, kv_heads, head_dim]` (Llama reference), a per-layer ring of token slots written by index (llama.cpp, cache types down to 18 bytes per 32 values), and a cache grown in 256-token steps (mlx-lm).
- D4: MLX has separate Metal kernels for queries of length at most 8 (decode, one-pass or two-pass) and for prefill.

## Where the backends really differ

One line per cell, for the operations this chapter uses. "Same" means the same source text compiles through `pmpp/vendors/hip.h`, according to the documentation; nothing here was run on CUDA or AMD hardware.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Book's tile storage (37,376 bytes) | Fits the 48 KB static limit | Fits (LDS 64 KiB or more per CU) | Does not fit 32 KB; pipeline creation fails; shrink B_c or d |
| Tile arrays declared in a helper function | Allowed (`__shared__` in a `__device__` function) | Not stated in AMD docs; declare in the kernel | Compile error; declare in the kernel, pass `threadgroup` pointers |
| Warp width the book's layout assumes (32) | 32 on every listed device | 64 on gfx9, 32 on gfx10 and later; parameterise | Undocumented; read `[[threads_per_simdgroup]]`, require 32 or parameterise |
| Lane and warp index | `threadIdx.x % 32`, `threadIdx.x / 32` | Same arithmetic with the real wavefront width | `[[thread_index_in_simdgroup]]`, `[[simdgroup_index_in_threadgroup]]` |
| Broadcasting one lane's Q element | `__shfl_sync(0xffffffff, v, lane)` | `__shfl_sync` with a 64-bit mask; pass `width` 32 if the logical warp is 32 on wave64 | `simd_broadcast(v, lane)` (uniform lane) or `simd_shuffle` |
| Row max / row sum of a segment | Hand-written shuffle reduction, or `cub::WarpReduce` (result in lane 0) | Hand-written, or `hipcub::WarpReduce` (default 64 logical lanes on CDNA) | `simd_max`, `simd_sum` (result in every lane) |
| Block-wide max / sum (Fig. 20.4) | `cub::BlockReduce` with `cuda::maximum<>{}`, `cuda::std::plus<>{}`, or hand-written | `hipcub::BlockReduce`, functor spelling in transition, or hand-written | Hand-written: `simd_max`/`simd_sum` plus threadgroup memory |
| Barrier between lanes sharing the P row | `__syncwarp()` (missing in the book) | `__syncwarp()`; AMD documents lockstep | `simdgroup_barrier(mem_flags::mem_threadgroup)` |
| Block barrier between tiles | `__syncthreads()` | `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| Minus infinity for masks and the initial max | Defined (IEEE); `-INFINITY` - `-INFINITY` is NaN | Defined by default | Undefined under fast math; use index-based masking or finite sentinels |
| Compile-time d, B_r, B_c | `#define` or template parameters | Same | `preprocessorMacros` or `[[function_constant]]` |
| Padding against bank conflicts (`LOG_NUM_BANKS` 5) | 32 banks of 4 bytes | 32 or 64 banks; re-measure | No bank structure documented; measure |
| Library attention | cuDNN SDPA (FlashAttention-2 algorithm, SM80+, fp16/bf16/fp8, paged); FlashAttention 2, 3, 4 | FlashAttention-2 via CK or Triton (`aiter`); PyTorch via AOTriton and CK | MPSGraph `scaledDotProductAttention` (macOS 15; causal descriptor macOS 27); MLX `fast.scaled_dot_product_attention` |
| Split-KV decode in the projects | llama.cpp `parallel_blocks` + `flash_attn_combine_results` | Same llama.cpp source | llama.cpp `kernel_flash_attn_ext_vec_reduce`; MLX and PyTorch MPS `sdpa_vector_2pass` |
| Matrix hardware for the two products | Tensor cores (WMMA, WGMMA on Hopper; FA3, FA4) | MFMA on CDNA through CK, rocWMMA | `simdgroup_matrix` 8 by 8, MPP `matmul2d` (llama.cpp `fa_tensor.metal`) |
| 64-bit float for a high-precision reference | `double` | `double` | None; keep the reference on the CPU |
