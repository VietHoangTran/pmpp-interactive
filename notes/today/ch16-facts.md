# Chapter 16 fact check: dynamic programming and wavefronts (CUDA, HIP, Metal, real backends)

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY, and judges the claim in the item title (the book's statement from `notes/ch16.md`, or the assumption a course author would bring). Where an item has several sub-claims, the text says which one failed. **Quote** is copied from the source with line breaks removed.
- "infra A4", "ch11 A6", "ch05 C8" point at items in `notes/today/infra-facts.md`, `ch11-facts.md`, `ch05-facts.md` and so on. Anything settled there is cited, not re-checked.
- Page numbers ("p. 398") are book pages as recorded in `notes/ch16.md`.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide "v13.5" (pages "Last updated on Oct 31, 2026"); CUDA C++ Best Practices Guide 13.5 (Nov 02, 2026); NVCC docs v13.5 (Nov 02, 2026); CUDA Toolkit release notes 13.5 (Nov 05, 2026); CUDA Math API Reference Manual 13.5; PTX ISA 9.5; CCCL source on `main`; HIP documentation sources and headers at tag `therock-10.2` of `ROCm/rocm-systems` (commit `43fa692e`), the ROCm 10.2.0 release (ch05 B1); HIPIFY "CUDA DEVICE API supported by HIP" table (`/en/latest/`); AMD "Instinct MI300" CDNA3 Instruction Set Architecture Reference Guide dated 5-August-2025; Metal Shading Language Specification "Version 4.1", dated 2026-06-04 ("MSL spec"); macOS SDK 27.0 headers and the Metal compiler it ships ("Apple metal version 32023.921"); llama.cpp `master` at `18486280` (2026-11-20), PyTorch `main` at `42cde06e`, mlx-lm `main` at `2d4df8d9` (last-commit dates given per file).
- Not checked: anything by running code on hardware. Every statement is from documentation, headers or project source, except where an item says "compiled here": those MSL snippets were compiled (not run) with the local Metal compiler to settle what the spec leaves open. They are not lab code and are not in the repository. Group E is our own arithmetic, done with small scripts.

Totals: 10 CONFIRMED, 1 WRONG, 7 PARTLY, 0 COULD NOT VERIFY (18 items: A1 to A5, B1 to B3, C1 to C4, D1 to D3, E1 to E3). The WRONG and PARTLY items, and every failed or unfound sub-claim, are collected under "Summary of corrections".

---

## Already settled elsewhere (cite, do not re-check)

| Chapter 16 needs | Settled in |
|---|---|
| `__global__`, `__device__`, `<<<grid, block, bytes>>>`, `dim3`, index built-ins on CUDA, HIP and Metal | ch02 A7 to A9, B4, C1, C4; ch05 B3, B4 |
| `extern __shared__ int swTile[]` sized by the third launch argument (Figs. 16.8, 16.9, 16.15); Metal's `[[threadgroup(0)]]` pointer plus `setThreadgroupMemoryLength`, length a multiple of 16 bytes | ch05 A7, B4, C8 |
| More than 48 KB of shared memory per block needs dynamic allocation and an opt-in | ch05 A3 |
| Metal threadgroup memory limit (32 KB on Apple4 and later); read `maxThreadgroupMemoryLength` | ch05 C1, C3 |
| `blockIdx.x`, `threadIdx.x` are unsigned on CUDA (`uint3`) and HIP (`unsigned int` members); Metal position attributes are unsigned only | ch03 A2, B2, C6 |
| `__syncthreads()`: uniform-condition rule; threads that have exited do not hold the barrier (Fig. 16.4 returns before its barrier) | ch04 A6 (CUDA: "all non-exited threads"); ch04 B3 (HIP: not documented); ch04 C3 (Metal: documented for Apple silicon only) |
| `__syncthreads()` also orders shared and global memory for the block | ch10 A1 |
| `threadgroup_barrier(mem_flags)`, the `mem_flags` table | ch05 C2 |
| Back-to-back launches as the global barrier between tile anti-diagonals: in-order default stream (CUDA), same stream rule (HIP), serial compute encoder plus hazard-tracked buffers (Metal) | ch02 A5, B3, C9 |
| Blocks run in any order; no scheduler ordering guarantee | ch04 A5, B2; ch12 A10 |
| Cooperative groups, `grid.sync()`, cooperative launch, residency limit, multi-device launch removed in CUDA 13; HIP headers differ; none on Metal | infra A4, B4; ch04 A7, B11, C3 |
| Thread block clusters ("several blocks" on p. 398), compute capability 9.0 and later | ch05 A6; infra A4 |
| Flag protocol between blocks: device-scope release/acquire, plain payload made visible, flags reset per launch, forward progress and the dynamic block index, `__nanosleep` | ch11 A5, A6, A7; ch12 A10 |
| The same on HIP (libhipcxx atomics; no documented progress guarantee; no `__nanosleep`) and on Metal (`coherent(device)`, Metal 4.1 acquire/release, spinning across threadgroups unsafe) | ch11 B3, B4, C3, C4, C5; ch12 B6, C4 |
| Shared-memory banks: 32 of 4 bytes (CUDA); 32 or 64 of 4 bytes (AMD); nothing documented (Metal); the padding fix | infra A10, B10, C13; ch06 A12, B4, C2; ch15 A2, B2, C2 |
| Warp shuffles for register-held tiles (§16.8 "small DP problems") | infra A3, B3, C2; ch10 A3, B1, C1 |
| Warp / wavefront / SIMD-group width (never a literal 32) | ch05 B5; ch04 C2; infra C5 |
| L2 cache named in the locality argument (p. 390) | ch06 A4, B3; infra C13 |
| Double buffering (p. 380 points to Chapter 6) | ch06 A16, B8, C7 |
| Integer `min`/`max` on CUDA (device-only overloads, mixed signed/unsigned overloads return `unsigned`); MSL `max`, `max3`, `min3` (Metal 2.1 and later) | ch13 A5, B1, C1 |
| MSL is C++17-based from Metal 4 (C++14 before) | ch03 C6 |

---

## Group A: CUDA, and the book's dated statements

### A1. "CUDA-enabled GPUs do not efficiently support deep nesting of function calls"; device functions are inlined; recursion stays as real calls (p. 376)

**Verdict: PARTLY.** That recursion is allowed in device functions and not in kernels is confirmed, and so is a per-thread stack whose size you set. NVIDIA's current documentation says nothing about the cost of deep or recursive calls, and the v13.5 pages read no longer say that device functions are inlined by default. The vendor statements closest to "recursion is costly" are AMD's and Apple's (B1, C1).

- Guide section 5.3.10.5.1 "Recursion": "__global__ , __tile_global__ and __tile__ functions do not support recursion, while __device__ and __host__ __device__ functions do not have such restriction."
- Guide section 5.4.1, constraints for `__global__`: "Does not support recursion."
- Runtime API, `cudaDeviceSetLimit(enum cudaLimit limit, size_t value)`: "cudaLimitStackSize is the stack size in bytes of each GPU thread."
- Inlining, section 5.4.1.3 "Inlining Specifiers": "__noinline__ : Instructs nvcc not to inline the function. __forceinline__ : Forces nvcc to inline the function within a single translation unit. __inline_hint__ : Enables aggressive inlining across translation units when using Link-Time Optimization." (also infra A7).
- COULD NOT VERIFY: any v13.5 sentence that device functions are inlined by default, or any NVIDIA statement about the performance of recursion or deep call chains. Searched today: the C++ language support and language extension appendices, the Best Practices Guide, the NVCC manual ("deemed appropriate", "inlined by default", "function call overhead").

For the lesson: keep the book's argument as the book's judgement, state the two documented facts (kernels cannot recurse; each thread's stack is a set size), and use the second reason the book gives (hash-table accesses are uncoalesced, which follows from ch06 A2) as the stronger one. The bottom-up approach the chapter adopts needs neither recursion nor a hash table.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html (section 5.3.10.5.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.1, 5.4.1.3)
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html (`cudaDeviceSetLimit`)

### A2. DPX instructions "since NVIDIA Hopper architecture", and `__vimax3_s32_relu()` as the replacement for `max4()` (p. 398, exercise 6)

**Verdict: PARTLY.** The function exists with exactly the semantics exercise 6 needs. "Since Hopper" is out of date as a statement about all later GPUs: NVIDIA's feature table lists DPX as native only on compute capability 9.0 and 10.x, and as several instructions on 11.0 and 12.x. Two NVIDIA documents disagree about 12.x. A library comment in NVIDIA's own source reports a compiler bug in exactly the dependent-chain pattern dynamic programming produces, fixed in CUDA 13.4.

The function, CUDA Math API 13.5, section 14 "SIMD Intrinsics":

- `__host__ __device__ int __vimax3_s32_relu(const int a, const int b, const int c)`: "Computes max(max(max(a, b), c), 0)". "Computes the maximum of signed integers a , b , and c , then applies ReLU."
- Relatives on the same page: `__vimax3_s32` ("Computes max(max(a, b), c)"), `__vimax_s32_relu`, `__viaddmax_s32_relu` ("Computes max(max(a + b, c), 0)"; "Computes the 32-bit wrap-around sum of signed integers a and b , then takes the signed maximum of that sum and c ."), and the paired 16-bit forms such as `__vimax3_s16x2_relu`.
- Inconsistency on that page: the declarations are `__host__ __device__`, while the section introduction says "This section describes SIMD intrinsic functions that are only supported in device code."

The programming guide, section 5.4.8.6 "Dynamic Programming eXtension (DPX) Instructions":

- "The DPX set of functions enables finding minimum and maximum values, as well as fused addition and minimum/maximum for up to three 16- or 32-bit signed or unsigned integer parameters. There is an optional ReLU, namely clamping to zero, feature."
- "These instructions are hardware-accelerated or software emulated depending on compute capability."
- "The DPX is an exceptionally useful tool for implementing dynamic programming algorithms such as the Smith-Waterman and Needleman-Wunsch algorithms in genomics and the Floyd-Warshall algorithm in route optimization."
- Example as printed: `int max_value_0 = __vimax3_s32_relu(a, b, c); // max(-15, 8, 5, 0) = 8` and `int max_value_1 = __vimax3_s32_relu(a, d, e); // max(-15, -2, -4, 0) = 0`.

Which devices run it natively:

| Source | 7.x | 8.x | 9.0 | 10.x | 11.0 | 12.x |
|---|---|---|---|---|---|---|
| Guide Table 32 "Feature Support per Compute Capability", row "DPX Instructions for Accelerated Dynamic Programming" | Multiple Instr. | Multiple Instr. | Native | Native | Multiple Instr. | Multiple Instr. |
| Best Practices Guide Table 5 "Throughput of Native Arithmetic Instructions", row "DPX (32-bit and 2-way SIMD 16-bit): multiple PTX instructions" (columns 7.5, 8.0, 8.6/8.9, 9.0, 10.0, 10.3, 12.0/12.1) | Multiple instruct. (7.5) | Multiple instruct. | a throughput figure, footnote "multiple instructions for intrinsics with boolean predicate outputs" | same (10.0, 10.3) | not listed | a throughput figure (12.0/12.1), footnote "... multiple instructions for intrinsics with boolean predicate outputs" |

The two NVIDIA documents disagree for 12.x (feature table: multiple instructions; throughput table: a native rate). Neither is marked as superseding the other. The throughput figures are device numbers and do not belong in a lesson; the course statement is "native on some NVIDIA architectures, emulated with several instructions on others, per NVIDIA's own tables, which disagree for compute capability 12.x".

Lower level: the PTX ISA 9.5 has only two-operand forms, `max{.relu}.type2 d, a, b;` with `.type2 = { .s16x2, .s32, .s8x4 }`, and "max.u16x2 , max{.relu}.s16x2 and max.relu.s32 require sm_90 or higher." There is no three-operand integer `max` in PTX, consistent with the Best Practices row label "multiple PTX instructions".

Status notes found today:

- CCCL `main`, `libcudacxx/include/cuda/__simd/min_max_relu.h` (last commit 2026-11-19), which calls `::__vimax3_s32_relu` for `cuda::simd::max_relu`: "Before CUDA 13.4, ptxas can generate incorrect SM90 and SM103 code for dependent VIMNMX.RELU instructions. The workaround consists in adding an identity instruction to break the dependency. CTK 13.4 fixes the issue." The workaround is compiled only under `_CCCL_CTK_BELOW(13, 4)`. A Smith-Waterman cell whose result feeds the next cell's `max` is a dependent chain of exactly this kind (our reading). COULD NOT VERIFY in NVIDIA's release notes: the CUDA 13.5 release notes page contains no "VIMNMX" and no "ptxas" entry.
- CUDA 13.5 release notes, "CUDA Math: Release 13.5", Resolved Issues: "Fixed the host-side and older-GPU fallback implementations of the integer SIMD functions __viaddmin_s32_relu , __viaddmax_s32_relu , __viaddmin_s32 , and __viaddmax_s32 to prevent undefined behavior due to signed integer overflow." So the emulated path of the fused-add forms was itself wrong before 13.5.

For exercise 6 and any lab (ours): `__vimax3_s32_relu(nw + subs_val, w + DELETION, n + INSERTION)` equals `max4(0, nw + subs_val, w + DELETION, n + INSERTION)` for all `int` inputs, because integer `max` is associative and commutative. A DPX variant is CUDA-only (B2, C2), should require CUDA 13.4 or later (or be checked against the `max4` version on the reader's machine), and is `unverified` until a `reports/verify-*.md` from an NVIDIA machine shows it passing. Real backends do not use DPX: GitHub code search today finds `__vimax3` in none of llama.cpp, PyTorch, MLX or CUTLASS, and in CCCL only in the two `cuda/__simd/` headers.

- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.8.6)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 32)
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (Table 5, section 12.1.1)
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (section 9.7.1.14 "Integer Arithmetic Instructions: max")
- https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html (section "CUDA Math: Release 13.5")
- https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__simd/min_max_relu.h (lines 44 to 64, 136 to 142)

### A3. The C++ the listings rely on: `d - blockIdx.x` stored in an `int`, `int` compared with `unsigned`, and `INFINITY` (Figs. 16.4, 16.9, 16.16, 16.18; notes, "Unclear in the PDF" item on undefined names)

**Verdict: PARTLY.** The comparisons of a signed column with the unsigned `L` behave as the notes say, by the standard. The tile and thread index arithmetic relies on an unsigned-to-signed conversion that is implementation-defined in C++17, which is what the labs compile as. `INFINITY` from `<cmath>` is a `float`, so the Fig. 16.4 tests against an `int` table can never be true.

The rules (cppreference):

- Integral conversions: "If the destination type is signed, the value does not change if the source integer can be represented in the destination type. Otherwise the result is implementation-defined (until C++20) the unique value of the destination type equal to the source value modulo 2 n where n is the number of bits used to represent the destination type (since C++20) (note that this is different from signed integer arithmetic overflow , which is undefined)."
- Usual arithmetic conversions, mixed signedness: "If the integer conversion rank of U is greater than or equal to the integer conversion rank of S , C is U ." So `int` with `unsigned int` is computed as `unsigned int`.
- Floating operand: "If either operand is of floating-point type ... if one of the operands is of a non-floating-point type, that operand is converted to the type of the other operand."
- `INFINITY`: "If the implementation supports an infinity for the type float , the macro INFINITY expands to constant expression of type float which evaluates to positive or unsigned infinity."

Which standard applies: NVCC v13.5, `--std`: "The default C++ dialect depends on the host compiler. nvcc matches the default C++ dialect that the host compiler uses." The course's lab build sets C++17 (`labs/CMakeLists.txt`: `set(CMAKE_CXX_STANDARD 17)`; `labs/pmpp.py` passes `-std=c++17` to the emulator build). MSL is C++17-based (Metal 4) or C++14-based (ch03 C6). So in every course build the conversion is implementation-defined, not modular.

What each listing line does (ours, from the rules):

- `const int tile_col = d - blockIdx.x;` (Fig. 16.9) and `d - blockIdx.x * 2` (Fig. 16.16): `unsigned` arithmetic wraps, then the out-of-range value is converted to `int`. Every compiler we know of gives the negative number, but the standard the labs use does not promise it. The same holds for `q_tile = d_tile - threadIdx.x` if `q_tile` is an `int`. Corrected form: do the arithmetic in `int`, `const int tile_col = (int)d - 2 * (int)blockIdx.x;`. Compiled here in MSL: the book's form draws "implicit conversion changes signedness: 'unsigned int' to 'const thread int'" under `-Wsign-conversion`; the `int(d) - int(bid)` form draws nothing.
- `if (r < L && q < L)` with `int q` and `unsigned int L` (Fig. 16.18): `q` is converted to `unsigned`, so a negative `q` becomes a large value and the test fails. This is defined behaviour and is what the notes describe. Writing `q >= 1 && q < (int)L` says it directly.
- `if (dist_k_col == INFINITY)` with `int dist_k_col` (Fig. 16.4): the `int` is converted to `float` and compared with positive infinity, which no `int` value equals. The early exits never fire; and if the table's "no edge" value is `INT_MAX`, `dist_k_col + dist_k_row` overflows, which is undefined. The listing works only if `INFINITY` is the reader's own integer sentinel. Corrected form: `constexpr int INF = ...` chosen so that `INF + INF` does not overflow `int` (for example `INT_MAX / 2`), and compare against `INF`.

- https://en.cppreference.com/w/cpp/language/implicit_conversion.html (section "Integral conversions")
- https://en.cppreference.com/w/cpp/language/usual_arithmetic_conversions.html
- https://en.cppreference.com/w/cpp/numeric/math/INFINITY.html
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html (section 4.2.3, `--std`)

### A4. §16.8's ways to drop the relaunch between tile anti-diagonals, on CUDA today

**Verdict: CONFIRMED** (by citation, with our application to tiles). All three mechanisms the book names exist and their conditions are documented. The flag scheme is safe under NVIDIA's execution model only when a tile waits on tiles owned by blocks that have already started.

- Grid-wide barrier with "persistent thread blocks": `cudaLaunchCooperativeKernel`, then `this_grid().sync()`; "The total number of blocks launched cannot exceed the maximum number of blocks per multiprocessor ... times the number of multiprocessors" (infra A4, ch04 A7). That is the book's "grid size must be at most the number of blocks that can run concurrently".
- "Several blocks": thread block clusters with `cluster.sync()`, compute capability 9.0 and later, 8 blocks portable (ch05 A6). Not mentioned by the book by name.
- Multi-device cooperative launch is gone: "Multi-device launch APIs and related references for Cooperative Groups have been removed as of CUDA 13." (ch04 A7.)
- Flags: one flag per tile, written with `cuda::atomic_ref<int, cuda::thread_scope_device>` `store(1, release)` after the tile is stored, read with `load(acquire)` before the dependent tile starts; the tile's plain stores are made visible by that pair (NVIDIA's message-passing example, ch11 A5). The flags must be zeroed before every launch (ch11 A5).
- Progress (ours, from ch11 A6 and ch12 A10): "Once a device thread makes progress ... all device threads in its thread-block cluster shall eventually make progress" and other clusters are "not guaranteed to eventually make progress". So a block may wait only on a tile whose owner has already executed a step. Two documented-safe ways: (1) hand out tiles in anti-diagonal order from an atomic counter, so every tile a block waits on (above and left) was claimed earlier by a running block; (2) a cooperative launch, under which "all device threads in its grid shall eventually make progress". The book's "one persistent block per tile row" indexed by `blockIdx.x` under an ordinary launch has no such backing, even when the grid would fit on the device.
- Back-off: `__nanosleep(ns)` in the spin (ch11 A7).
- Fig. 16.21's saving (10 rows versus 9 on 2 SMs) re-checks by simulation (E2).

### A5. The book's references for hypertiling (p. 400, refs [2] to [4])

**Verdict: CONFIRMED.** All three resolve in Crossref to the papers the notes record. A lesson that names the origin of the hyperplane (supernode) transformation can cite these.

- [2] Irigoin and Triolet, "Supernode partitioning", Proceedings of the 15th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL '88), pp. 319 to 329, DOI 10.1145/73560.73588.
- [3] Di, Ye, Su, Sui, Xue, "Automatic Parallelization of Tiled Loop Nests with Enhanced Fine-Grained Parallelism on GPUs", 2012 41st International Conference on Parallel Processing, DOI 10.1109/icpp.2012.19.
- [4] Belviranli, Deng, Bhuyan, Gupta, Zhu, "PeerWave", Proceedings of the 29th ACM International Conference on Supercomputing (2015), DOI 10.1145/2751205.2751243.

- https://api.crossref.org/works/10.1145/73560.73588
- https://api.crossref.org/works/10.1109/icpp.2012.19
- https://api.crossref.org/works/10.1145/2751205.2751243

---

## Group B: HIP / ROCm

### B1. Recursion and the call stack on HIP (counterpart of A1)

**Verdict: CONFIRMED.** HIP supports recursive device functions, has a settable stack limit, and AMD's documentation warns about the memory cost. AMD's own example is a recursive Fibonacci, the book's top-down Fig. 16.1 without the memo table.

HIP docs, "Call stack" (source `projects/hip/docs/how-to/hip_runtime_api/call_stack.rst` at the tag):

- "Similar to CPU programming, recursive functions and deeply nested function calls are supported. However, developers must ensure that these functions do not exceed the available stack memory, considering the huge amount of memory needed for the call stack due to the GPUs inherent parallelism."
- "NVIDIA GPUs have the independent thread scheduling feature where each thread has its own call stack and effective program counter. On AMD GPUs threads are grouped; each warp has its own call stack and program counter."
- "If a thread or warp exceeds its stack size, a stack overflow occurs, causing kernel failure."
- Stack limit, from the page's example: `hipDeviceGetLimit(&stackSize, hipLimitStackSize)` and `hipDeviceSetLimit(hipLimitStackSize, newStackSize)`, the counterpart of `cudaLimitStackSize` (A1).
- Example `device_recursion.hip`: `__device__ unsigned long long fibonacci(unsigned long long n) { if (n == 0 || n == 1) { return n; } return fibonacci(n - 1) + fibonacci(n - 2); }`, called from a `__global__` kernel.

The page also gives a default-stack memory figure for one AMD product. It is a device-specific number; do not quote it in a lesson.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/call_stack.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/tools/example_codes/device_recursion.hip
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/tools/example_codes/call_stack_management.cpp

### B2. `__vimax3_s32_relu` (DPX) on HIP (counterpart of A2)

**Verdict: WRONG** as the assumption that the exercise-6 call compiles for HIP through a name mapping. HIP declares no DPX function in anything searched. AMD hardware documents a three-input integer max instruction without ReLU, and no HIP intrinsic exposes it.

- HIPIFY "CUDA DEVICE API supported by HIP": no row at all for `__vimax3_s32_relu` or any other DPX name (`__vimax*`, `__vimin*`, `__viaddmax*`, `__vibmax*`). The same table does list older CUDA SIMD intrinsics with an empty HIP column, for example `__vadd2` and `__vmaxs4`, so the table covers this family and simply has no DPX entries.
- HIP headers at `therock-10.2`: all 89 headers under `projects/hip/include/hip/` and `projects/clr/hipamd/include/hip/` downloaded and searched; no `vimax`, `viaddmax`, `vibmax`, `dpx` or `max3`.
- HIP math API page (`reference/math_api.html`): no `vimax`, and not the older `__vmaxs4` either.
- AMD CDNA3 ISA guide (5-August-2025), instruction `V_MAX3_I32`: "Select the maximum of three signed 32-bit integer inputs and store the selected value into a vector register." Pseudocode: `D0.i32 = v_max_i32(v_max_i32(S0.i32, S1.i32), S2.i32)`. No ReLU form; `V_MED3_I32` also exists. COULD NOT VERIFY: whether the AMD compiler turns `max(max(a, b), c)` in HIP source into this instruction; AMD documents no intrinsic for it in the pages read.

For the course (ours): a `.cu` lab keeps `max4()` as the portable path and puts any DPX call behind `#if defined(__CUDA_ARCH__)` (or the project's CUDA-only guard). `pmpp/vendors/hip.h` cannot map it, because there is nothing to map to.

- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Device_API_supported_by_HIP.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/math_api.html
- https://github.com/ROCm/rocm-systems/tree/therock-10.2/projects/hip/include/hip
- https://github.com/ROCm/rocm-systems/tree/therock-10.2/projects/clr/hipamd/include/hip
- https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf (section on VOP3 instructions, `V_MAX3_I32`)

### B3. The rest of the tiled kernels on HIP: a barrier inside a helper (Fig. 16.19), a barrier after early returns (Fig. 16.4), flags (§16.8)

**Verdict: PARTLY.** The barrier is an ordinary inline device function, so calling it from `initialize_tile` is fine. What a returned thread does to the barrier, and whether a block may wait on another block's flag, are not documented by AMD.

- Header at the tag, `amd_device_functions.h` line 720: `__device__ inline __attribute__((convergent)) void __syncthreads() {`, which issues `__barrier(__CLK_GLOBAL_MEM_FENCE | __CLK_LOCAL_MEM_FENCE)` (ch04 B3). A `__device__` helper may call it; the uniform-condition rule is CUDA's, taught as the portable rule (ch04 A6, B3). The barrier orders both global and shared memory, like CUDA's (ch10 A1).
- Fig. 16.4 returns before its barrier in the last block of a row: COULD NOT VERIFY on HIP (ch04 B3). The corrected kernel should not return early: guard the work with `if (col < n_vertices)` and let every thread reach the barrier. This also makes it portable to Metal on non-Apple-silicon GPUs (ch04 C3).
- Flags between blocks: same source through libhipcxx `cuda::atomic_ref` (ch11 B4), but no forward-progress guarantee between workgroups is documented, and AMD's own look-back needed per-architecture fixes (ch11 B3, ch12 B6). Exercises 4 and 5 are `unverified` on HIP.
- Grid barrier: `hipLaunchCooperativeKernel` with all six arguments, `<hip/hip_cooperative_groups.h>` (infra B4).
- Banks: 32 or 64 per architecture, so the `LOG_NUM_BANKS` of the book's `pad()` macro is 5 or 6 depending on the GPU (infra B10; E1 below).

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_device_functions.h (line 720)

---

## Group C: Metal

### C1. Recursion in MSL (counterpart of A1)

**Verdict: PARTLY.** Recursion in kernel code is supported from Metal 2.4 and compiles here. Apple documents a call-stack depth setting with a performance warning, but describes it as applying to indirect calls only, so whether a directly recursive helper needs it is unclear.

- MSL spec section 1.5.4 "Restrictions" (page 13): "Metal supports recursive function calls (C++ section 5.2.2, item 9) in compute (kernel) context starting with Metal 2.4."
- `MTLComputePipelineDescriptor.maxCallStackDepth` (macOS 11.0 and later). Header: "The maximum depth of the call stack in stack frames from the kernel. Defaults to 1 additional stack frame." Web docs: "The maximum call stack depth for indirect function calls in compute shaders." "The property's default value is 1. Change its value if you use recursive functions in your compute pass. The maximum call stack depth applies only to indirect function calls in your shader, and affects the upper bound of stack memory for each thread. Indirect function calls include those to visible functions, intersection functions, and to dynamic libraries. To avoid a runtime performance impact, keep this value as small as possible because the framework reserves a large call stack."
- COULD NOT VERIFY: whether a direct (non-indirect) recursive call like `fib(n - 1)` is bounded by `maxCallStackDepth`. The two sentences above point different ways.
- Compiled here: a `static int fib(int n)` that calls itself, used from a kernel, compiles at `-std=metal3.0` through `metal4.1` (the local compiler rejects `metal2.3` and `metal2.4` as values for this target). Not run.

Apple's "keep this value as small as possible because the framework reserves a large call stack" is the clearest vendor statement found that deep calls cost something, which is the book's §16.2 point.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 1.5.4, page 13)
- https://developer.apple.com/documentation/metal/mtlcomputepipelinedescriptor/maxcallstackdepth
- Local header: `Metal.framework/Headers/MTLComputePipeline.h` (macOS 27.0 SDK), lines 130 to 135

### C2. A Metal counterpart of DPX `__vimax3_s32_relu` (counterpart of A2)

**Verdict: PARTLY.** MSL has a three-input integer `max3` (ch13 C1), so `max4` becomes one call plus a clamp. Apple documents no ReLU form and nothing about how `max3` maps to hardware.

- MSL spec section 6.4, Table 6.2 (page 203): `T max3(T x, T y, T z)`: "Returns max(x, max(y, z))." "All OS: Metal 2.1 and later." Also `median3` and `min3`. "T is one of the scalar or vector integer types."
- No `relu` in the integer function table, and no DPX-like fused add-max.
- Compiled here: `static inline int max4_metal(int a, int b, int c) { return max(max3(a, b, c), 0); }` compiles at `metal3.0` to `metal4.1`. Not run.

For the course: on Metal, write `max(max3(nw + s, w + del, n + ins), 0)`; it is the same value as the book's `max4` and the CUDA DPX call. Whether it is faster than `max4` on Apple GPUs is a measurement, not a documented fact.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 6.4, Table 6.2, pages 202 to 203)

### C3. Fig. 16.19's barrier inside a helper, a threadgroup tile passed to helpers, and a barrier that also orders device memory

**Verdict: CONFIRMED** (spec plus compilation here). The book's helper structure ports, provided the tile is declared in (or passed into) the kernel.

- A helper may take a `threadgroup` pointer when called from a kernel (ch05 C1). A helper may not declare a threadgroup array itself (ch11 C2, compiler error quoted there). So `initialize_tile`, `load_n/w/nw` and `store_tile` take `threadgroup int* swTile`.
- Compiled here, `-std=metal3.0` to `metal4.1`: `static inline void initialize_tile(threadgroup int* swTile, uint tile_width, uint tid) { for (...) swTile[row * tile_width + tid] = 0; threadgroup_barrier(mem_flags::mem_threadgroup); }` called from a kernel whose tile is `threadgroup int* swTile [[threadgroup(0)]]`. No diagnostic.
- The flags are a bit field: MSL spec section 6.10.1 (page 216): "Table 6.13 describes the bit field values for the mem_flags argument to threadgroup_barrier and simdgroup_barrier." Compiled here: `threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);`.
- When the device flag matters (ours, from the `mem_flags` table in ch05 C2): CUDA's `__syncthreads()` and HIP's orders global memory for the block as well (ch10 A1, B3), Metal's `mem_threadgroup` does not. The book's square-tile kernel reads tile-edge inputs from global memory, but those were written by an earlier launch, so `mem_threadgroup` suffices. A variant that keeps the whole table in device memory and computes several wavefronts inside one threadgroup (the "small DP problem" design of §16.8) needs `mem_device` on its barrier. PyTorch's MPS CTC kernel does exactly that (D2).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 4.4 and 6.10.1, pages 120, 215 to 217)

### C4. §16.8 on Metal: what replaces relaunch, grid barrier and flags

**Verdict: CONFIRMED** (by citation). Only the book's baseline structure is documented-safe on Metal: one dispatch per tile anti-diagonal in one serial compute encoder, or the whole problem inside one threadgroup.

- One dispatch per `d`: dispatches in one compute encoder from the plain `computeCommandEncoder` call run "in dispatched order", and device-created buffers are hazard-tracked, so each dispatch sees the previous one's writes with no barrier (ch02 C9). This holds for the 2T − 1 square-tile and 3T − 1 hypertile dispatch sequences.
- Grid barrier: none, no cooperative launch, no clusters (ch04 C3; infra table).
- Flags between threadgroups: unsafe. No forward progress between threadgroups is documented and an Apple engineer called the chained-flag assumptions undefined behaviour (ch11 C5, ch12 C4); a buffer read by another threadgroup in the same dispatch must also be `coherent(device)` (ch11 C3). Exercises 4 and 5 have no Metal version.
- Short problems in one threadgroup: SIMD-group shuffles and `threadgroup_barrier` (infra C2, C4). The SIMD-group width is not documented; read `[[threads_per_simdgroup]]` (infra C5).

---

## Group D: How llama.cpp, PyTorch and MLX run recurrences on the GPU

None of these projects runs Smith-Waterman or Floyd-Warshall. What they do run, every day, is a recurrence along the sequence: the state of a recurrent or linear-attention layer, or a CTC alignment table. Each is a constant-size wavefront in the book's sense (Fig. 16.2(a), (b)): every cell of step t depends only on step t − 1, so a whole step is one wavefront, and the projects choose between a block barrier per step, a SIMD-group per independent chain, and chunking the sequence into tiles joined by launches. The mapping to the chapter is ours; the code facts are read from the files today.

### D1. llama.cpp: recurrent-state kernels (RWKV, Mamba, gated delta rule) and a triangular solve

**Verdict: CONFIRMED.**

- `ggml/src/ggml-cuda/wkv.cu` (last commit 2026-09-25), `rwkv_wkv_f32`: one block per (sequence, head) (`<<<B * H, C / H, 0, stream>>>`), one thread per state column, the column's state held in a per-thread array `float state[head_size]`, and a loop over the tokens of the sequence: `for (int t = ...; t < ...; t += C) { __syncthreads(); _k[tid] = k[t]; _r[tid] = r[t]; _td[tid] = td[t]; __syncthreads(); ... }`. The dependency from step t − 1 to t lives in each thread's own registers; the two barriers per step protect the shared staging arrays (a write-after-read and a read-after-write), which is the book's "barrier between intra-tile wavefronts" with a different reason for it. The same file casts `(float4&)(_k[j])` for vector reads of shared memory (the undocumented cast idiom of infra A6).
- `ggml/src/ggml-metal/kernels/wkv.metal`, `kernel_rwkv_wkv6_f32`: the same structure in MSL, `threadgroup float _k[head_size]` and two `threadgroup_barrier(mem_flags::mem_threadgroup)` per token; `const uint head_size = 64; // TODO: support head_size = 128`.
- `ggml/src/ggml-cuda/ssm-scan.cu` (last commit 2026-11-08) has two paths, which is the chapter's whole argument in production:
  - Sequential scan: Mamba-1 kernel loops `for (size_t i = 0; i < L; i++)` with `__syncthreads()` around the shared `smemB`/`smemC` per token; the Mamba-2 kernel gives each warp one chain and loops `for (int64_t i = 0; i < n_tok; i++)` with only `warp_reduce_sum` per step (no block barrier, because the chain is warp-private).
  - Chunked: `// Chunk size for chunked SSD. Caps matmul cost at O(chunk^2) per chunk.` `#define SSM_SSD_CHUNK_SIZE 256` and `#define SSM_SSD_MIN_TOKENS 128`, with the header comment "SSD (State Space Duality) kernels for Mamba-2 prefill (n_tok > SSM_SSD_MIN_TOKENS) ... This converts the O(T*N) sequential scan into parallel matmuls." The host runs `for (int64_t k = 0; k < n_chunks; k++)` and, per chunk, several kernels and cuBLAS calls, ending with "3e: s_cur = B_weighted @ X_dt^T + decay_total * s_cur_old (state update)". So the state is carried between chunks by kernel launches (the book's global synchronisation between tiles) and each chunk is dense parallel work (the book's tile). It is selected only `#if !defined(GGML_USE_HIP)`, on NVIDIA Turing or later, for Mamba-2 and `n_t` in (128, `SSM_SSD_MAX_TOKENS`].
  - `ggml/src/ggml-metal/kernels/ssm.metal` has the scan kernels (`kernel_ssm_scan_impl`); no chunked path appears in that file.
- `ggml/src/ggml-cuda/gated_delta_net.cu`: a per-token loop `for (int t = 0; t < n_tokens; t++)` over a state matrix, with the comment `//TODO: Add chunked kernel for even faster pre-fill`; `gated-delta-net-mma.cu` is a chunked variant ("Each block owns V independent value columns and advances by C tokens"). Metal has `kernels/gated_delta_net.metal`.
- `ggml/src/ggml-cuda/solve_tri.cu`: a triangular solve, where row i needs every earlier row (a dependency chain one row wide). Small sizes use `solve_tri_f32_fast` ("Fast Kernel (n <= 64, k <= 32) - Warp-based parallel reduction"), looping over rows with `warp_reduce_sum`; larger ones call `cublasStrsmBatched`.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/wkv.cu (lines 4 to 66, 213 to 225)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/wkv.metal (lines 3 to 75)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ssm-scan.cu (lines 13 to 25, 98 to 124, 204 to 236, 376 to 384, 663 to 677, 855 to 868)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/ssm.metal (line 167)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/gated_delta_net.cu (lines 75, 192)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/gated-delta-net-mma.cu (line 146)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/solve_tri.cu (lines 68 to 81)

### D2. PyTorch: CTC loss, a dynamic-programming table filled wavefront by wavefront, on CUDA and on MPS

**Verdict: CONFIRMED.** `torch.nn.functional.ctc_loss` computes the forward variable alpha(t, s) from alpha(t − 1, s), alpha(t − 1, s − 1) and alpha(t − 1, s − 2): a 2D DP table whose rows are wavefronts. Both backends parallelise over s, step t sequentially with a block barrier, and process s in chunks in dependency order when the row is wider than the block, which is the chapter's tiling argument in two sentences of comments.

- CUDA, `aten/src/ATen/native/cuda/LossCTC.cu` (last commit 2026-11-17): "We parallelize by batch and target sequence. Empirically, it is faster to loop over the input (log probs) sequence and do target in parallel, even if it means more frequent __syncthreads." "In contrast to the cuDNN implementation, we allow large target lengths. For this we need that all previous `s` have been computed when we start a new block_s. This is why we have our own for loop here." The loop: `for (int64_t t=1; t < max_input_length; t++) { __syncthreads(); ...`. Launch: `dim3 block(threads_target, threads_batch); dim3 grid(1, (batch_size+threads_batch-1)/threads_batch);`, so one block row handles whole sequences and no inter-block synchronisation is ever needed.
- MPS, `aten/src/ATen/native/mps/kernels/LossOps.metal` (last commit 2026-11-07): "When S_max exceeds the threadgroup size, the positions are processed in blocks of `tptg`, and each block runs the full time recursion before the next block starts. The recursion over `s` has cross-block dependencies: alpha[s] reads s-1 and s-2 (an earlier block), beta[s] reads s+1 and s+2 (a later block). Blocks therefore have to be visited in dependency order: ascending for alpha, descending for beta." Each time step begins with `threadgroup_barrier(mem_flags::mem_device);`, because the table lives in device memory (C3).
- MPS host side, `operations/LossOps.mm`: one threadgroup per batch item, `dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)` with `TG_SIZE = std::min<int64_t>([pso maxTotalThreadsPerThreadgroup], 2 * max_target_length + 1)`, the width read from the pipeline rather than hard-coded.
- Dispatch table (`native_functions.yaml`): `_ctc_loss` has `CUDA: ctc_loss_gpu` and `MPS: ctc_loss_mps`.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/LossCTC.cu (lines 64 to 72, 133, 158 to 159, 290 to 303)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/LossOps.metal (lines 242 to 283)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/LossOps.mm (lines 654 to 716)

### D3. mlx-lm: the gated delta rule and the SSM step as Metal kernels with no threadgroup barrier in the time loop

**Verdict: CONFIRMED.** mlx-lm writes its recurrent kernels with `mx.fast.metal_kernel`. Each SIMD-group owns an independent chain (one row of the state matrix), so the per-token dependency is satisfied inside the SIMD-group with `simd_sum`, and the loop over tokens needs no threadgroup barrier at all: the cheapest synchronisation that covers the dependency, which is the chapter's rule.

- `mlx_lm/kernels/gated_delta.py` (last commit 2026-11-19), `_make_gated_delta_kernel`: `constexpr int n_per_t = Dk / 32;`, a per-thread `float state[n_per_t]`, then `for (int t = 0; t < T; ++t) { ... kv_mem = simd_sum(kv_mem); ... out = simd_sum(out); ... }`, with no barrier in the loop. Launch `grid = (32, Dv, B * Hv)`, `threadgroup = (32, 4, 1)`: the x extent of 32 is one SIMD-group by assumption, as in every project read for ch11 C2.
- Used by the linear-attention models' layers, for example `mlx_lm/models/qwen3_next/layers.py` and `mlx_lm/models/kimi_linear/layers.py` (GitHub code search today), through `mlx_lm/layers/recurrent.py`.
- `mlx_lm/kernels/ssm.py`: "The SSM state update's Metal kernel." with `ssm_update_kernel`, "One step of the state update; dt is already biased and clipped."

- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/kernels/gated_delta.py (lines 24 to 127, 393 to 402)
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/kernels/ssm.py
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/layers/recurrent.py

---

## Group E: The chapter's arithmetic, rechecked (ours, not vendor statements)

### E1. Bank conflicts in the square-tile and hypertile kernels, unpadded and padded (pp. 395 to 396, Fig. 16.20; notes, "Unclear in the PDF" item on `pad()`)

**Verdict: PARTLY.** The book's claim holds: with the hypertile stored as a square, every access in an iteration walks down a column with stride W, which conflicts whenever W is a multiple or a divisor of the bank count, and one gap per row removes it. Two refinements: the square-tile kernel has no conflicts to begin with (padding it would create them), and the book's `pad()` macro is not the same as one gap per row; on NVIDIA it removes the conflicts for W = 8, 16 and 32 and leaves a 2-way conflict at W = 64.

Method: NVIDIA's rule (bank = 32-bit word index mod 32; distinct words in one bank serialise; one word to many lanes is a broadcast; infra A10), lanes = consecutive `threadIdx.x` in warps of 32, every intra-tile iteration and every warp of the block, worst case reported. Accesses from Figs. 16.9, 16.10 (square: cell `(r, d − r)`, stride W − 1 between lanes) and Figs. 16.16 to 16.19 (hypertile: stored column `q_tile = d_tile` for every lane, stride W). "Row gap" is index `r·(W + 1) + q`; "`pad()`" is the book's `x + (x >> 5)` applied to `r·W + q`. Script run 2026-11-20.

Worst-case conflict degree on NVIDIA (1 = none):

| Kernel | Accesses | Layout | W = 8 | W = 16 | W = 32 | W = 64 |
|---|---|---|---|---|---|---|
| Square (Figs. 16.9, 16.10) | write, `load_n`, `load_w`, `load_nw` | unpadded | 1 | 1 | 1 | 1 |
| Square | same | row gap | 2 | 8 | 31 to 32 | 32 |
| Square | same | `pad()` | 1 | 1 | 31 to 32 | 2 |
| Hypertile (Figs. 16.16, 16.17) | write, `load_n`, `load_w`, `load_nw` | unpadded | 2 | 8 | 31 to 32 | 32 |
| Hypertile | same | row gap | 1 | 1 | 1 | 1 |
| Hypertile | same | `pad()` | 1 | 1 | 1 | 2 |
| Both | `store_tile`, `initialize_tile` (`swTile[row·W + threadIdx.x]`) | any of the three | 1 | 1 | 1 | 1 |

(31 versus 32 is the load versus the write: an edge lane reads from global memory instead.)

- So padding is a hypertile fix only. A lab that pads both kernels makes the square one worse; the prediction for the padding stage should say so.
- `pad()` versus one gap per row: the macro inserts one gap per 32 words. For W ≤ 32 (powers of two) that is a gap every 32 / W rows, which still separates the lanes; at W = 64 it is two gaps per row, stride 66, which puts lanes 16 apart in one bank. One gap per row (stride W + 1) is conflict-free at every width. The notes' "equals one gap per row only when W equals the bank count" is right about the layouts and incomplete about the effect.
- AMD (a model, as in ch15 E1; AMD does not document how a 64-lane LDS access is split for conflicts, ch15 B2): with 64 banks and 64-lane wavefronts the unpadded hypertile gives 1, 4, 16 and 64-way at W = 8, 16, 32, 64; one gap per row, or `pad()` with `LOG_NUM_BANKS` 6, gives none. With 32 banks the book's `LOG_NUM_BANKS` is 5 as on NVIDIA. The value of `LOG_NUM_BANKS` therefore depends on the AMD architecture (B3).
- Metal: no bank structure is documented (infra C13, ch15 C2). The padded and unpadded hypertile are a measurement there, with a prediction and no model.
- Storage: one gap per row needs W·(W + 1) words; the book's host formula `threads*threads + ((threads*threads) >> LOG_NUM_BANKS)` covers the `pad()` layout (largest index `pad(W² − 1)`, for example 1,055 words used and 1,056 allocated at W = 32).

### E2. The chapter's schedule counts (pp. 386, 392, 396, 399; notes' derived lines)

**Verdict: CONFIRMED.** Recomputed by simulation from the kernels' own index formulas (`tile_col = d − blockIdx.x`, active when `0 ≤ tile_col < T`; hypertile `tile_col = d − 2·blockIdx.x`, active when `0 ≤ tile_col ≤ T`).

| T (tiles per side) | Square: launches, most tiles in one launch, tile computations | Hypertile: launches, most tiles in one launch, tile computations |
|---|---|---|
| 3 | 5, 3, 9 | 8, 2, 12 |
| 4 | 7, 4, 16 | 11, 3, 20 |
| 8 | 15, 8, 64 | 23, 5, 72 |

- Launches are 2T − 1 and 3T − 1 as the book says. The hypertile schedule for T = 3 is exactly the notes' list (d0 (0,0); d1 (0,1); d2 (0,2),(1,0); d3 (0,3),(1,1); d4 (1,2),(2,0); d5 (1,3),(2,1); d6 (2,2); d7 (2,3)). Each tile row stays two tile columns behind the row above, as p. 392 requires.
- The available parallelism per launch is lower for hypertiles (at most about T/2 + 1 tiles) even though each tile does full-width work; an explorer that shows tiles per step should show both numbers.
- Eqs. 16.3 and 16.4: n = 8 gives 64 cells in 15 iterations (4.27 per iteration) for the square tile and 64 in 8 (8 per iteration) for the hypertile; n = 32 gives 16.25 versus 32. The ratio tends to 2, the book's "twice as efficient per iteration".
- Fig. 16.21 (4 × 4 tiles, 2 SMs, one time unit per tile): barrier per tile anti-diagonal takes 10 rows, a greedy dependency-driven start takes 9, as printed.

### E3. The tile against the documented on-chip limits (for choosing lab sizes)

**Verdict: CONFIRMED** (arithmetic on cited limits). The lab must read the limits from the device; these are the documented ceilings, not values to hard-code.

- An `int` tile of W × W uses 4W² bytes: W = 32 is 4 KiB, W = 64 is 16 KiB, W = 128 is 64 KiB. With one gap per row: 4W(W + 1) bytes, for example 16,640 bytes at W = 64.
- CUDA: above 48 KB per block needs the opt-in (ch05 A3), so W = 128 would need it; W ≤ 64 does not. W is also the block size, at most 1024 (ch05 A5).
- Metal: 32 KB of threadgroup memory on Apple4 and later (ch05 C1), and `setThreadgroupMemoryLength` lengths must be multiples of 16 bytes (ch05 C8). 4W² is a multiple of 16 for every even W; 4W(W + 1) is for W = 8, 16, 32, 64; the book's `pad()` formula at W = 8 gives 66 words (264 bytes) and must be rounded up. W = 128 does not fit.
- HIP: LDS limits per workgroup are architecture-dependent (ch08 B3 via ch15 E1); read `sharedMemPerBlock` (ch05 B2).

---

## Book items with no vendor fact to check

- Genomics background: the human genome's 3.2 billion base pairs, read lengths of 50 to 300 bp and 10 K to 100 K bp, the example scores (+3, −3, gap 2) (p. 382). Not checked; the condensed lesson names §16.5's scoring and genomics detail as left out (pp. 382 to 384) and does not repeat these figures.
- Fig. 16.2's four patterns, the Floyd-Warshall example graph and its b→f misprint, the Smith-Waterman recurrence and its insertion/deletion naming, the index-convention flips, the p. 391 typos: book-internal (notes, "Unclear in the PDF").
- "Kernel-call overhead" as the remaining cost of hypertiles (p. 396): no vendor figure checked and none should be quoted; a lab measures launch counts and time on the reader's device.
- Fig. 16.21's "GPU with 2 SMs": an illustration, re-checked in E2.
- The chapter contains no timings or device figures to check.

## Answers to the notes' portability questions

| Notes' `[unverified]` item | Answer | Where |
|---|---|---|
| Kernel syntax, `<<<grid, block, bytes>>>`, built-ins on HIP; Metal entry point | Same on HIP; Metal `kernel` with position attributes and an explicit threadgroup size | ch02 B4, C1, C4; ch05 B3, B4 |
| `__shared__` scalar and `extern __shared__ int swTile[]` sized at launch; can Metal size it at run time? | Yes: `threadgroup int* swTile [[threadgroup(0)]]` plus `setThreadgroupMemoryLength`, a multiple of 16 bytes | ch05 C8; E3 |
| `__syncthreads()` in a loop, after some threads returned (Fig. 16.4), inside a helper (Fig. 16.19) | Loop: fine on all three if every thread reaches it. After return: CUDA documented, HIP not, Metal on Apple silicon only; the corrected kernel does not return early. Helper: fine on CUDA and HIP; on Metal compiled here, with the tile passed in | ch04 A6, B3, C3; B3, C3 |
| `cudaDeviceSynchronize()` per launch and in-order launches as the global barrier | Not needed for ordering: one stream (CUDA, HIP) or one serial compute encoder (Metal) already orders the launches | ch02 A5, B3, C9; C4 |
| Shared-memory banks and `pad()` | CUDA 32 banks; AMD 32 or 64, so `LOG_NUM_BANKS` 5 or 6; Metal undocumented. Padding helps the hypertile and hurts the square tile; `pad()` is 2-way at W = 64 on NVIDIA | infra A10, B10, C13; E1 |
| Cooperative groups, grid sync, persistent blocks | CUDA and HIP under a cooperative launch with a resident grid; none on Metal | infra A4, B4; ch04 A7, C3; A4 |
| Flag array between blocks; memory ordering not specified by the chapter | CUDA: device-scope release/acquire, and wait only on tiles claimed by started blocks (atomic tile counter) or use a cooperative launch. HIP: same source, `unverified`. Metal: unsafe | ch11 A5, A6, B3, B4, C3 to C5; A4, B3, C4 |
| DPX `__vimax3_s32_relu` | CUDA only; native on compute capability 9.0 and 10.x per Table 32; no HIP declaration; Metal `max(max3(...), 0)` | A2, B2, C2 |
| Shuffles for register-held tiles | HIP: same names, 64-bit mask; Metal: `simd_shuffle_*`, no mask | infra A3, B3, C2 |

---

## Summary of corrections

Every WRONG and PARTLY item, and every failed or unfound sub-claim:

- A1: recursion is allowed in `__device__` functions and not in kernels, with a per-thread stack set by `cudaLimitStackSize`; no current NVIDIA statement was found that device functions are inlined by default or that recursion is slow (could not verify). AMD (B1) and Apple (C1) do warn about stack memory.
- A2: DPX is native on compute capability 9.0 and 10.x only, per the guide's Table 32; 11.0 and 12.x run it as several instructions, so "since Hopper" does not mean "on every later GPU". The Best Practices throughput table disagrees for 12.x. The Math API page declares the functions `__host__ __device__` but calls them device-only.
- A2: CCCL's source says ptxas before CUDA 13.4 can miscompile dependent `VIMNMX.RELU` on SM90 and SM103 (not found in the release notes); CUDA 13.5 fixed undefined behaviour in the host and older-GPU fallbacks of `__viaddmax_s32_relu` and relatives. A DPX lab should require CUDA 13.4 or later and be checked against `max4`.
- A3: `d - blockIdx.x` (and `d - 2*blockIdx.x`, `d_tile - threadIdx.x`) is unsigned arithmetic whose conversion to `int` is implementation-defined under C++17, which the labs use; compute in `int`. `<cmath>`'s `INFINITY` is a `float`, so Fig. 16.4's tests against an `int` table never fire; use an integer sentinel such as `INT_MAX / 2`.
- A4 (detail): the book's "one persistent block per row of tiles" waiting on flags is not backed by NVIDIA's progress rule under an ordinary launch; hand out tiles from an atomic counter in anti-diagonal order, or use a cooperative launch.
- B2 (WRONG): HIP has no `__vimax3_s32_relu` or any DPX function (not in HIPIFY's table, the 89 headers at `therock-10.2`, or the math API page). AMD's CDNA3 ISA has `V_MAX3_I32` with no ReLU, and no documented HIP intrinsic for it.
- B3: what a returned thread does at `__syncthreads()` on HIP is not documented (ch04 B3); flags between blocks are `unverified` on HIP.
- C1: whether `maxCallStackDepth` bounds a direct recursive call is unclear from Apple's two descriptions (could not verify).
- C2: Metal's counterpart is `max(max3(a, b, c), 0)`; there is no ReLU form and nothing documented about hardware support.
- E1: the square-tile kernel has no bank conflicts unpadded, and padding it creates them (up to 32-way at W = 32); the book's `pad()` macro leaves a 2-way conflict for a 64-wide hypertile on NVIDIA, where one gap per row leaves none.

Status facts worth knowing even where the verdict is CONFIRMED:

- B1: AMD's recursion example is a recursive Fibonacci, the book's Fig. 16.1 top-down shape; AMD keeps one call stack per warp, NVIDIA one per thread.
- C3: Metal's barrier needs `mem_flags::mem_device` when threads of one threadgroup exchange values through device memory; CUDA's and HIP's `__syncthreads()` order global memory anyway.
- C4: on Metal the book's relaunch design (one dispatch per tile anti-diagonal in one serial encoder) is the documented-safe one; there is no grid barrier and spinning on flags is unsafe.
- D1 to D3: real projects run time recurrences as constant-size wavefronts with three kinds of synchronisation: a block barrier per step (llama.cpp RWKV, PyTorch CTC), a SIMD-group-private chain with no barrier (llama.cpp Mamba-2 scan, mlx-lm gated delta), and chunks joined by launches with dense work inside (llama.cpp's chunked SSD on NVIDIA). PyTorch's CTC kernels process blocks of positions in dependency order, with comments that state the chapter's rule.
- A5: the hypertiling references [2] to [4] resolve (DOIs listed).

## Where the backends really differ

One line per cell, for what this chapter uses. "Same" means the same source text compiles through `pmpp/vendors/hip.h`, according to the documentation; nothing here was run on CUDA or AMD hardware, and the Metal cells marked "compiled here" were compiled, not run.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Global barrier between tile anti-diagonals | One launch per `d` in one stream (in order) | Same | One dispatch per `d` in one serial compute encoder, hazard-tracked buffers |
| Barrier between intra-tile wavefronts | `__syncthreads()` (orders shared and global memory) | Same (fences global and local memory) | `threadgroup_barrier(mem_flags::mem_threadgroup)`; add `mem_device` if the wavefront is in device memory |
| Barrier after some threads returned (Fig. 16.4) | Allowed: "all non-exited threads" | Not documented | Allowed on Apple silicon only |
| Barrier inside a helper (Fig. 16.19) | Allowed | Allowed (`__syncthreads` is an inline device function) | Compiles here, tile passed in as `threadgroup int*` |
| Tile array declared inside a helper | Allowed (`__shared__` in a `__device__` function) | Not documented | Compile error; declare in the kernel |
| Tile memory sized at launch | `extern __shared__` + third `<<<>>>` argument; over 48 KB needs opt-in | Same | `[[threadgroup(0)]]` pointer + `setThreadgroupMemoryLength`, multiple of 16 bytes, 32 KB total |
| Grid-wide barrier (§16.8) | `grid.sync()` under `cudaLaunchCooperativeKernel`, resident grid | `hipLaunchCooperativeKernel`, all six arguments | None |
| Tile-to-tile flags (§16.8, exercises 4 and 5) | Documented-safe with device-scope release/acquire and an atomic tile counter (or cooperative launch) | Same source via libhipcxx; progress not documented, `unverified` | Unsafe: no progress guarantee; would also need `coherent(device)` |
| Back-off while waiting | `__nanosleep(ns)` | None documented | None |
| Three-way max with ReLU (exercise 6) | `__vimax3_s32_relu`, native on 9.0 and 10.x, several instructions elsewhere; use CUDA 13.4 or later | None; keep `max4` | `max(max3(a, b, c), 0)` (Metal 2.1 and later), compiled here |
| Banks for the hypertile column access | 32 banks: unpadded 32-way at W = 32; one gap per row fixes it; `pad()` 2-way at W = 64 | 32 or 64 banks by architecture; `LOG_NUM_BANKS` 5 or 6 | Not documented; measure |
| Recursion in device code (§16.2) | `__device__` yes, `__global__` no; `cudaLimitStackSize` per thread | Yes; `hipLimitStackSize`; one stack per warp | Kernel context from Metal 2.4; `maxCallStackDepth` (default 1) for indirect calls |
| `int tile_col = d - blockIdx.x` | Unsigned wrap then implementation-defined conversion (C++17); write it in `int` | Same | Same (C++14/17 base); compiler warns under `-Wsign-conversion` |
| Short DP problems in one warp (§16.8) | `__shfl_*_sync`, 32-bit mask | Same names, 64-bit mask | `simd_shuffle_*`, no mask; width from `[[threads_per_simdgroup]]` |
| Time recurrences in real code | llama.cpp `wkv.cu`, `ssm-scan.cu` (scan and chunked SSD); PyTorch `LossCTC.cu` | llama.cpp builds the same `.cu` files; the chunked SSD path is excluded on HIP | llama.cpp `wkv.metal`, `ssm.metal`; PyTorch `LossOps.metal`; mlx-lm `gated_delta.py` |
