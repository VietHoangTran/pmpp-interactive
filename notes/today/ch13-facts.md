# Chapter 13 fact check: CUDA, HIP, Metal

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims, the verdict is for the item as a whole and the text says which sub-claim failed. A verdict judges the claim in the item title (the book's statement, or the assumption a course author would most likely bring).
- **Quote** is copied from the source (HTML docs flattened to text, GitHub raw files at a tag or on the default branch, PDF text, local SDK headers). Line breaks inside quotes are removed.
- "ch05 A7", "infra B11", "appC C1" point at items in `notes/today/ch05-facts.md`, `infra-facts.md`, `appC-facts.md` and so on. Anything settled there is cited, not re-checked.
- Page numbers ("p. 316") are book pages as recorded in `notes/ch13.md`.
- Versions the sources reported about themselves on the check date: CUDA Math API Reference Manual "13.5" (page last updated Nov 02, 2026) and CUDA Programming Guide v13.5 (C++ language extensions page last updated Oct 31, 2026); CCCL source on `main` (the files read were last changed 2026-11-19; latest tagged release v3.5.0, 2026-10-07); ROCm Core SDK 10.2.0 = tag `therock-10.2` of `ROCm/rocm-systems`, `ROCm/rocm-libraries` and `ROCm/llvm-project`; LLVM `main` (`__clang_hip_math.h` last changed 2026-11-10); Metal Shading Language Specification "Version 4.1", dated 2026-06-04 ("MSL spec"); macOS 27.0 SDK headers on this machine; C++ working draft at eel.is ("Generated on 2026-08-23 ... This is not an ISO publication"); llama.cpp `master`, MLX `main`, mlx-lm `main`, PyTorch `main`, Apache Hadoop `trunk` on the check date.
- Not checked: anything by running code on a GPU. Every vendor statement is from documentation, headers or project source. The one computation in this file (E1) is our own Python check, labelled as such.

Totals: 13 CONFIRMED, 0 WRONG, 4 PARTLY, 1 COULD NOT VERIFY (18 items: 7 book and CUDA, 2 HIP, 4 Metal, 4 project, 1 our own check). Sub-claims that failed or could not be found are collected under "Summary of corrections".

The chapter names no GPU, no CUDA version and no timing (notes, "Dated or device-specific content"), so Group A is short. Most of what the lessons and the lab need from vendors is already settled in earlier files.

---

## Already settled elsewhere (cite, do not re-check)

| Chapter 13 needs | Settled in |
|---|---|
| `__global__`, `threadIdx`/`blockIdx`/`blockDim`/`gridDim`, `<<<>>>` with the same spelling on HIP | ch05 B3, B4 |
| Metal kernel entry point and index attributes (`thread_position_in_grid`, `thread_position_in_threadgroup`, `threadgroup_position_in_grid`) | ch05 C4, ch02 C1 |
| Helpers such as `co_rank` and `merge_sequential` need `__device__` (or `__host__ __device__`) on CUDA and HIP; an unqualified function is host-only. Metal has no qualifier: every function in a `.metal` file is GPU code | ch02 A7, B4, C7 |
| `extern __shared__ int shareAB[];` with the byte count as the third launch argument; "a kernel can have only one dynamically allocated shared array", so the book's split into `A_S` and `B_S` is NVIDIA's own advice | ch05 A7; HIP ch05 B4 |
| Metal dynamic threadgroup memory: `[[threadgroup(n)]]` pointer parameter, `setThreadgroupMemoryLength:atIndex:`, multiple of 16 bytes, several buffers allowed | ch05 C8 |
| Per-block on-chip limits: CUDA 48 KB without opt-in; Metal 32 KB (Apple4 and later); AMD LDS sizes | ch05 A3, C1; ch09 A8, B4 |
| `__syncthreads()` semantics and the uniform-condition rule; HIP; Metal `threadgroup_barrier(mem_flags::mem_threadgroup)` | ch04 A6, B3; ch05 C2; ch04 C3 |
| 1024 threads per block (the exercise's block size); per-pipeline cap on Metal; AMD limits | ch05 A5, C3; ch03 B3 |
| Warp 32, wavefront 64 or 32, SIMD-group width not a documented constant | ch05 B5, ch04 B6, ch04 C2 |
| Coalescing (§13.5's two problems): NVIDIA 32-byte sectors; AMD documents the same idea; Apple documents no rule | ch06 A2, B1, C1 |
| Shared memory needs no coalescing, bank conflicts aside (§13.6's strategy 3) | ch06 A6, B4, C2; infra A10, B10, C13 |
| Occupancy as blocks per SM, registers limiting it (p. 327); AMD; Apple (Xcode counter, no API) | ch04 A13, A14, B10; ch06 A10, C5 |
| One `int*` helper works on global and shared data (generic addressing) on CUDA and HIP; MSL address spaces are disjoint | appC A2, B1, C1 |
| Integer ceiling division: NVIDIA teaches `(n + b - 1) / b` and `cuda::ceil_div` from `<cuda/cmath>`, not `ceil` | ch02 A8 |
| nvcc follows ISO C++ (up to C++20/23); MSL is C++17-based in Metal 4 and C++14-based before, with no C++ standard library | ch03 A6, C6; ch02 C7 |
| CUB and hipCUB status, CCCL as the home of Thrust/CUB/libcu++, two hipCUB copies in ROCm 10.2 | infra A11, B11 |
| Vocabulary sizes against on-chip limits (for the top-k bridge) | ch09 D7 |
| llama.cpp's Metal radix-select top-k (`kernel_top_k_f32_i32`, threadgroup histogram) | ch09 D4 |
| llama.cpp samples on the host by default; a backend sampling path exists | ch01 D2 |

---

## Group A: the book's dated statements, the C++ the listings rely on, and the CUDA library route

### A1. Reference [1]: Siebert and Träff, 2012 (pp. 306, 328), and the name the libraries use

**Verdict: CONFIRMED.** The paper exists as cited. The same idea appears in today's GPU libraries under a different name, "merge path", citing a different 2012 paper.

Crossref records (queried by author and title):

- C. Siebert, J. L. Träff, "Efficient MPI Implementation of a Parallel, Stable Merge Algorithm", *Recent Advances in the Message Passing Interface* (EuroMPI 2012), Lecture Notes in Computer Science, pp. 204–213, 2012, DOI 10.1007/978-3-642-33518-1_25.
- Journal follow-up by the same authors: "Perfectly Load-Balanced, Stable, Synchronization-Free Parallel Merge", *Parallel Processing Letters*, March 2014, article 1450005, DOI 10.1142/S0129626414500054.

The name in library source:

- CUB (`cub/cub/block/block_merge_sort.cuh`): "Computes the intersection of the diagonal `diag` with the merge path in the merge matrix of two input sequences. This implements the DiagonalIntersection algorithm from Merge-Path." It cites "S. Odeh, O. Green, Z. Mwassi, O. Shmueli, Y. Birk, 'Merge Path - Parallel Merging Made Simple', ... IPDPS, 2012" and "S. Odeh, O. Green, Y. Birk, 'Merge Path - A Visually Intuitive Approach to Parallel Merging', 2014, URL: https://arxiv.org/abs/1406.2628". Return value: "The number of elements merged from the first sequence at the intersection of the diagonal with the merge path. The number of elements merged from the second sequence is `diag` minus the returned value." That is the book's co-rank definition, word for word in substance: return i, the caller derives j = k − i.
- Crossref: Odeh et al., IPDPS Workshops 2012, pp. 1611–1618, DOI 10.1109/IPDPSW.2012.202. A GPU-specific paper of the same year also exists: O. Green, R. McColl, D. A. Bader, "GPU merge path", Proceedings of the 26th ACM International Conference on Supercomputing (ICS 2012), pp. 331–340, DOI 10.1145/2304576.2304621 (Crossref; none of the code read today cites it).
- The other projects' names for the same function: rocPRIM `merge_path`, MLX `merge_partition`, PyTorch MPS `merge_partition` ("finds the split point on the merge-path diagonal"), llama.cpp "binary-search partition (i, j) such that i + j = k" (D1 to D3, B2).

For the course: "co-rank" is the book's (Siebert and Träff's) word; tell the reader that the code they will read calls it merge path or merge partition. That the two are the same function is checked in E1.

- https://doi.org/10.1007/978-3-642-33518-1_25
- https://doi.org/10.1142/S0129626414500054
- https://doi.org/10.1109/IPDPSW.2012.202
- https://doi.org/10.1145/2304576.2304621
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/block_merge_sort.cuh (lines 30 to 62)

### A2. "Modern map-reduce frameworks such as Hadoop" merge sorted results in the reduce step (p. 305)

**Verdict: CONFIRMED.** Apache Hadoop is maintained, MapReduce is still one of its modules, and its reduce-side shuffle merges sorted streams. "Modern" is the book's judgement; the facts are below.

- Releases page: "3.5.0 2026 Apr 2", "3.4.3 2026 Feb 24", "3.3.6 2023 Jun 23".
- Home page: "Hadoop MapReduce: A YARN-based system for parallel processing of large data sets."
- `mapred-default.xml` (trunk): `mapreduce.task.io.sort.factor`, default 10, "The number of streams to merge at once while sorting files." `mapreduce.reduce.merge.inmem.threshold`: "When we accumulate threshold number of files we initiate the in-memory merge and spill to disk."

The k-way merge of sorted runs is the book's point. Nothing here is GPU-related; the lesson can mention Hadoop in one line or not at all.

- https://hadoop.apache.org/releases.html
- https://hadoop.apache.org/
- https://github.com/apache/hadoop/blob/trunk/hadoop-mapreduce-project/hadoop-mapreduce-client/hadoop-mapreduce-client-core/src/main/resources/mapred-default.xml

### A3. `tile_size` 1024 uses (1024 + 1024) × 4 = 8,192 bytes of shared memory, assuming a 4-byte `int` (pp. 316–317)

**Verdict: CONFIRMED** for CUDA and Metal; HIP not separately documented.

- CUDA: the built-in vector-type table in the C++ language extensions appendix gives "int1 , uint1 4 4" (size and alignment in bytes), i.e. a 4-byte `int`. The same table warns about `long`: "long is 4 bytes on C++ LLP64 data model (Windows 64-bit), while it is 8 bytes on C++ LP64 data model (Linux 64-bit)." So the book's arithmetic holds for `int`; it would not be portable for `long`.
- Metal: MSL spec Table 2.1, "int / int32_t: A signed two's complement 32-bit integer."
- HIP: compiled by Clang as C++ `int`; I found no separate AMD statement of its size.
- The budget fits every documented per-block limit (ch05 A3: 48 KB without opt-in on CUDA; ch05 C1: 32 KB on Metal Apple4 and later; ch09 B4 for AMD LDS). On Metal each 4,096-byte tile is a multiple of 16 bytes, as `setThreadgroupMemoryLength` requires (ch05 C8).

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (vector types table)
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 2.1, Table 2.1, page 25)

### A4. The C++ rules the listings rely on: integer division before `ceil`, `%` with a negative operand, `&&` short-circuit (notes, "Unclear in the PDF" item 14; code-listing notes for Figs. 13.5 and 13.19)

**Verdict: CONFIRMED.** The notes' inferences are right by the language standard, which nvcc (ch03 A6) and HIP-Clang follow. Metal differs on `%` (C2).

Quotes (C++ working draft, [expr.mul]/4 and [expr.log.and]/1):

- "For integral operands, the / operator yields the algebraic quotient with any fractional part discarded;" footnote: "This is often called truncation towards zero."
- "if the quotient a/b is representable in the type of the result, (a/b)*b + a%b is equal to a".
- "Unlike &, && guarantees left-to-right evaluation: the second operand is not evaluated if the first operand is false."

Consequences:

- `ceil((m+n)/(blockDim.x*gridDim.x))` (Fig. 13.9), `ceil((m+n)/gridDim.x)` (Fig. 13.11) and `ceil((C_length)/tile_size)` (Fig. 13.12): the division happens first, in integers, and discards the fraction; `ceil` then receives a whole number and changes nothing. With the book's own numbers, 4000 / 1024 is 3, not the four iterations p. 317 describes (notes item 14). In the first two the divisor is built from the unsigned built-ins (ch03 A2), so the division is unsigned; for the non-negative sizes involved the result is the same. Use `(a + b - 1) / b` or `cuda::ceil_div` (ch02 A8).
- `%` takes the sign of the dividend: `(0 + 0 - 1) % 1024` is −1 in C++. That is the `co_rank_circular` case the notes flag (`A_S_start + i - 1` when i = 0). The value is defined and never used, because the `i > 0` guard fails first. On Metal it is not defined at all (C2).
- `co_rank`'s tests `i > 0 && j < n && A[i-1] > B[j]` read `A[i-1]` and `B[j]` only after the bounds tests pass. That is guaranteed by the rule quoted above; reordering the operands would read out of bounds.

- https://eel.is/c++draft/expr.mul
- https://eel.is/c++draft/expr.log.and

### A5. Integer `min` and `max` in CUDA kernel code (Figs. 13.9, 13.11 to 13.13)

**Verdict: PARTLY.** The listings' `min(int, int)` calls are documented device functions. Two things a course author would not expect: the documented overloads are device-only, and there are mixed signed/unsigned overloads that return `unsigned`.

CUDA Math API Reference Manual 13.5, section 12 "Integer Mathematical Functions":

- `__device__ int min (const int a, const int b)`: "Calculate the minimum value of the input int arguments."
- `__device__ int max (const int a, const int b)`: "Calculate the maximum value of the input int arguments."
- Mixed overloads: `__device__ unsigned int min (const int a, const unsigned int b)`: "Calculate the minimum value of the input int and unsigned int arguments." Likewise `unsigned int min (const unsigned int a, const int b)`, and the same pattern for `long`, `long long` and `max`.
- Every overload on the page is `__device__`; the page contains no `__host__` declaration.

Consequences (ours):

- A call that mixes `blockIdx.x` arithmetic (unsigned) with an `int` resolves to an overload that returns `unsigned int`, so a negative `int` argument cannot come back as itself. The chapter's sizes are non-negative, so the listings are unaffected, but `co_rank`'s `k - n` can be negative: the book wisely writes `0 > (k-n) ? 0 : k-n` there.
- A `co_rank` declared `__host__ __device__` (so the lab can unit-test it on the CPU) cannot rely on this page for a host `min`. Write ternaries as Fig. 13.5 does; they mean the same thing on CUDA, HIP, the emulator and MSL. (Repository note: `grep` today finds no global `min`/`max` in `labs/common/include/pmpp/emu/`, only `std::min` inside the runtime; not compiled.)
- Not checked: NVIDIA's toolkit headers (not online), so whether nvcc also provides host overloads in the global namespace is not established here. AMD's header comment (B1) says its mixed host overloads exist "for compatibility with CUDA", which suggests CUDA has them; that is AMD's statement, not NVIDIA's.

- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INT.html

### A6. `std::merge` as the lab's CPU reference: stable, first range wins ties, at most N − 1 comparisons

**Verdict: CONFIRMED.** The standard guarantees exactly the book's tie rule (`A[i] <= B[j]`: on equal keys the A element goes first). The current draft even defines the bounded merge through a co-rank.

Quotes (C++ working draft, [alg.merge] and [algorithm.stable]):

- Effects: "Copies the first K elements of the range [first1, last1) and the first N−K elements of the range [first2, last2) into the range [result, result + N). If an element a precedes b in an input range, a is copied into the output range before b. If e1 is an element of [first1, last1) and e2 of [first2, last2), e2 is copied into the output range before e1 if and only if E is true." where E is `bool(invoke(comp, invoke(proj2, e2), invoke(proj1, e1)))`, i.e. e2 strictly before e1.
- K is defined as "the smallest integer in [0, last1 - first1) such that for the element e1 in the position first1 + K there are at least N−K elements e2 in [first2, last2) for which E holds, and be equal to last1 - first1 if no such integer exists." For the overloads with an output bound, N is the bounded output length, so K is the co-rank of N in the first range.
- Complexity: "For the non-parallel algorithm overloads, at most N−1 comparisons and applications of each projection." "For the parallel algorithm overloads, O(N) comparisons".
- "Remarks: Stable ([algorithm.stable])." and [algorithm.stable] (1.3): "For the merge algorithms, for equivalent elements in the original two ranges, the elements from the first range (preserving their original order) precede the elements from the second range (preserving their original order)."

For the lab: compare the GPU output with `std::merge` on key–value pairs whose keys repeat often (values = source position, tagged by which array). Exact equality of the pair arrays tests stability, because the reference is guaranteed stable. A plain-`int` comparison cannot tell a stable merge from an unstable one.

Not checked: whether the K wording (which belongs to the bounded-output overloads in the draft) appears in a published ISO standard. The stability remark and the tie rule are what the lab relies on.

- https://eel.is/c++draft/alg.merge
- https://eel.is/c++draft/algorithm.stable

### A7. The CUDA library route: `thrust::merge` and `cub::DeviceMerge`

**Verdict: PARTLY.** Both exist and implement the book's tiled kernel with the Exercise 3 refinement. Their documentation disagrees about stability, and Thrust's CUDA backend calls the CUB function.

Existence and stability:

- `cub::DeviceMerge::MergeKeys` and `MergePairs` (`cub/device/device_merge.cuh`): "Merges two sorted sequences of values (called keys) into a sorted output sequence. Merging is unstable, which means any two equivalent values (neither value is ordered before the other) may be written to the output sequence in any order." ".. versionadded:: 2.7.0 First appears in CUDA Toolkit 12.8."
- `thrust::merge` (`thrust/merge.h`): "merge is stable, meaning both that the relative order of elements within each input range is preserved, and that for equivalent elements in both input ranges the element from the first range precedes the element from the second."
- Thrust's CUDA backend (`thrust/system/cuda/detail/merge.h`) implements `merge` as `cub::DeviceMerge::MergeKeys(nullptr, storage_size, keys1_begin, num_keys1, keys2_begin, num_keys2, result_begin, compare_op, stream)` (and `MergePairs` for `merge_by_key`).
- Our reading of the code: `MergePath` sends ties to the first sequence (`if (binary_pred(key2, key1)) { keys1_end = mid; } else { keys1_begin = mid + 1; }`) and `serial_merge` takes from the second sequence only when `compare_op(key2, key1)` is true. That is the book's tie rule, so the implementation reads as stable today. CUB's documentation does not promise it.
- Related: `cub::DeviceMergeSort` has `StableSortKeys` and `StableSortPairs`; its `SortPairs` "is not guaranteed to be stable".

Design (CCCL `main`, files last changed 2026-11-19):

- A separate partition kernel computes the block-level co-ranks: `// Computes the merge path intersections at equally wide intervals. The approach is outlined in the paper: Odeh et al, "Merge Path - Parallel Merging Made Simple"`, then in `device_partition_merge_path_kernel`: `const Offset diagonal_num = (::cuda::std::min) (diagonal_idx * items_per_tile, keys1_count + keys2_count); key1_beg_offsets[diagonal_idx] = _CUB::MergePath(keys1, keys2, keys1_count, keys2_count, diagonal_num, compare_op);`
- The merge agent (`cub/agent/agent_merge.cuh`, `consume_tile`) reads `keys1_beg = key1_beg_offsets[tile_idx + 0]` and `keys1_end = key1_beg_offsets[tile_idx + 1]`, derives the B range by subtraction, asserts `keys1_count_tile + keys2_count_tile == items_per_tile`, loads exactly that many keys into shared memory, then per thread `int diag0_thread = ItemsPerThread * static_cast<int>(threadIdx.x);` and `_CUB::MergePath(keys1_shared, keys2_shared, keys1_count_tile, keys2_count_tile, diag0_thread, compare_op)`, then `serial_merge`.

So, against the book: the block-level co-rank (Fig. 13.11) is computed for every tile boundary up front in its own kernel and stored in global memory, instead of by thread 0 inside the merge kernel; the tile loads exactly the A and B elements it will consume, which is the alternative of p. 316 and Exercise 3, so the "only half of what is loaded is used" problem of §13.6 does not arise and there is no circular buffer (§13.7). The per-thread co-rank in shared memory and the sequential merge are the book's Fig. 13.13.

- https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_merge.cuh
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/dispatch/dispatch_merge.cuh (lines 95 to 135)
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/agent/agent_merge.cuh (lines 150 to 260)
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/block_merge_sort.cuh
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_merge_sort.cuh
- https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/merge.h
- https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/system/cuda/detail/merge.h

---

## Group B: HIP / ROCm

Everything else the chapter's kernels use on HIP has CUDA's spelling and is settled (qualifiers ch02 B4; `extern __shared__`, `__syncthreads()`, built-ins ch05 B4; generic pointers appC B1; coalescing ch06 B1; occupancy ch04 B10).

### B1. Integer `min` and `max` in HIP kernel code (counterpart of A5)

**Verdict: PARTLY.** They exist, but HIP's own documentation does not list them; they come from the compiler's header, and the overload set differs from CUDA's documented one.

- HIP's math API reference (`projects/hip/docs/reference/math_api.rst` at `therock-10.2`) has no integer `min` or `max` entry; the only occurrences of "min" are inside the funnel-shift descriptions. HIP's runtime header `amd_hip_runtime.h` defines none.
- Clang's `__clang_hip_math.h` (LLVM `main`; the block from `template <class T> __DEVICE__ T min` to the host definitions is identical in `ROCm/llvm-project` at `therock-10.2`):

```cpp
template <class T> __DEVICE__ T min(T __arg1, T __arg2) {
  return (__arg1 < __arg2) ? __arg1 : __arg2;
}
__DEVICE__ int min(int __arg1, int __arg2) {
  return (__arg1 < __arg2) ? __arg1 : __arg2;
}
__DEVICE__
float min(float __x, float __y) { return __builtin_fminf(__x, __y); }
__DEVICE__
double min(double __x, double __y) { return __builtin_fmin(__x, __y); }
```

  (and the same for `max`). Then "// Define host min/max functions." with `DEFINE_MIN_MAX_FUNCTIONS(int, int, int)` in the global namespace by default (opt out with `__HIP_NO_HOST_MIN_MAX_IN_GLOBAL_NAMESPACE__`).
- On mixed types: "The host min/max functions below accept mixed signed/unsigned integer parameters and perform unsigned comparisons, which may produce unexpected results if a signed integer was passed unintentionally. To avoid this happening silently, these overloaded functions are not defined by default. However, for compatibility with CUDA, they will be defined if users define `__HIP_DEFINE_MIXED_HOST_MIN_MAX__`."

Consequence (our reading of overload resolution, not compiled): in device code a mixed call such as `min(blockIdx.x * x, m + n)` has no exact match on HIP (the template cannot deduce one `T` from `unsigned` and `int`, and the `int`, `float` and `double` overloads each need a conversion), so it may be ambiguous where CUDA picks its documented `unsigned` overload. Cast both arguments to one type, or write a ternary. The lab's `.cu` must compile for CUDA, HIP and the emulator, so ternaries are the simple rule.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/reference/math_api.rst
- https://github.com/llvm/llvm-project/blob/main/clang/lib/Headers/__clang_hip_math.h (lines 1286 to 1360)
- https://github.com/ROCm/llvm-project/blob/therock-10.2/clang/lib/Headers/__clang_hip_math.h

### B2. hipCUB, rocPRIM and rocThrust merge (counterpart of A7)

**Verdict: CONFIRMED.** AMD's libraries have the same merge, with the same two-kernel design. Stability: rocThrust documents it; rocPRIM is silent.

Read at `therock-10.2` of `ROCm/rocm-libraries`, in the hipCCL 3 copy (`projects/hipccl/hipccl3/...`; infra B11 explains the two copies; the older `projects/hipcub` also has `device_merge.hpp`):

- `hipcub::DeviceMerge` (`hipcub/backend/rocprim/device/device_merge.hpp`): `struct DeviceMerge` with `MergeKeys` and `MergePairs`, each `return ::rocprim::merge(d_temp_storage, ...)`.
- `rocprim::merge` (`rocprim/device/device_merge.hpp`): "Parallel merge primitive for device level." "merge function performs a device-wide merge. Function merges two ordered sets of input values based on comparison function." No statement on stability in either direction.
- rocPRIM's kernels (`rocprim/device/detail/device_merge.hpp`): a `partition_kernel_impl` that stores `indices[id] = begin;` where `begin = merge_path(keys_input1, keys_input2, input1_size, input2_size, diag, compare_function)` at `partition_id = id * spacing`; then each block `load`s exactly its two ranges into `keys_shared` followed by `::rocprim::syncthreads();`, and each thread computes `diag = ItemsPerThread * flat_id` and `merge_path(keys_shared + ..., diag, compare_function)` before a serial merge. The same design as CUB (A7).
- `rocprim::detail::merge_path` (`rocprim/detail/merge_path.hpp`): `begin = diag < input2_size ? 0u : diag - input2_size; end = min(diag, input1_size); ... if(!compare_function(input_b, input_a)) { begin = a + 1; } else { end = a; }`: first range wins ties, as in the book.
- rocThrust `thrust/merge.h` carries Thrust's sentence "merge is stable, meaning both ..."; its HIP backend (`thrust/system/hip/detail/merge.h`) calls `rocprim::merge(`.
- Top-k on the same stack (for the bridge): `rocprim/device/device_topk.hpp`, "Find the largest or smallest K elements from an input array of keys.", implemented through `#include "detail/device_topk_air.hpp"` and `rocprim::detail::device_topk_air<...>`, the same AIR radix family as CUB's `DeviceTopK` (D4).

- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipccl/hipccl3/hipcub/hipcub/include/hipcub/backend/rocprim/device/device_merge.hpp
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipccl/hipccl3/rocprim/rocprim/include/rocprim/device/device_merge.hpp
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipccl/hipccl3/rocprim/rocprim/include/rocprim/device/detail/device_merge.hpp
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipccl/hipccl3/rocprim/rocprim/include/rocprim/detail/merge_path.hpp
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipccl/hipccl3/rocprim/rocprim/include/rocprim/device/device_topk.hpp
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipccl/hipccl3/rocthrust/thrust/merge.h
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipccl/hipccl3/rocthrust/thrust/system/hip/detail/merge.h

---

## Group C: Metal

Kernel, threadgroup memory, barrier and index attributes are settled (ch05 C1 to C8, ch02 C1, C7). These items cover what the co-rank code itself needs.

### C1. `min`, `max` and a 32-bit `int` in MSL

**Verdict: CONFIRMED.**

- MSL spec section 6.4: "The header <metal_integer> defines the integer functions in Table 6.2. T is one of the scalar or vector integer types." Table 6.2: `T max(T x, T y)`, "Returns y if x < y, otherwise it returns x."; `T min(T x, T y)`, "Returns y if y < x, otherwise, it returns x."; also `T clamp(T x, T minval, T maxval)`, "Returns min(max(x, minval), maxval). Results are undefined if minval > maxval.", and `min3`, `max3` ("All OS: Metal 2.1 and later").
- Both arguments have the one type `T`, so a mixed `uint`/`int` call needs a cast (ours, from the signature). This matters on Metal more than on CUDA because the position attributes are unsigned only (ch03 C6).
- MSL spec Table 2.1: "int / int32_t: A signed two's complement 32-bit integer." (A3)

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 6.4, Table 6.2, pages 202 to 204; Table 2.1, page 25)

### C2. Do integer `/`, `%` and `&&` in MSL behave as in C++ (A4)?

**Verdict: PARTLY.** Division follows the C++ base, with an unspecified (not undefined) result out of range. `%` with a negative operand is undefined in MSL, unlike C++. The spec does not restate `&&` short-circuiting.

Quotes (MSL spec section 3.1 "Scalar and Vector Operators"):

- "Division on integer types that result in a value that lies outside of the range bounded by the maximum and minimum representable values of the integer type, such as TYPE_MIN/-1 for signed integer types or division by zero, does not cause an exception but results in an unspecified value."
- "For any component computed with a second operand that is zero, the modulus operator result is undefined. If one or both operands are negative, the results are undefined. Results for other components with nonzero operands remain defined. If both operands are nonnegative, the remainder is nonnegative."
- "The logical operators [and (&&), or (||)] act upon two operands that are Boolean expressions. The result is a scalar or vector Boolean." Nothing about evaluation order. The spec says MSL is "a C++17-based specification with extensions and restrictions" (section 1.5; ch03 C6), and I found no restriction that removes C++'s guarantee (A4); but there is no explicit Apple sentence either (could not verify).

Consequences:

- The ceiling-division bug of A4 is the same in MSL (integer division discards the fraction; MSL has no `cuda::ceil_div`, so write `(a + b - 1) / b`).
- The `co_rank_circular` wrap `(A_S_start + i - 1) % tile_size` with a −1 dividend gives a defined −1 in C++ but an undefined value in MSL. It is still never used if the guard runs first; a Metal port should form wrapped indices only inside the guarded branch, or add `tile_size` before `%`. This concerns only a reader who attempts §13.7, which the lesson leaves out.
- `co_rank`'s bounds guards in MSL rest on C++ short-circuiting. Shipping Metal code reads safely without depending on it: llama.cpp's merge kernel reads `tmp0[i]` only under `if (i < len0)`, and MLX's `merge_step` uses `(a_idx < A_sz) ? As[a_idx] : ValT(CompareOp::init)` (D1, D3). The lab can do the same at no cost.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 1.5, page 12; section 3.1, pages 111 to 113)

### C3. One co-rank helper for device memory and threadgroup memory

**Verdict: CONFIRMED** that MSL needs two versions (or a template), where CUDA and HIP use one `int*` helper (appC A2, B1).

- Disjoint spaces: "These attributes describe disjoint address spaces" (MSL spec section 4; appC C1).
- "Any variable that is a pointer or reference needs to be declared with one of the address space attributes discussed in this section. If an address space attribute is missing on a pointer or reference type declaration, a compilation error occurs."
- Section 1.5.1: "Metal supports overloading, as defined by section 13 of the C++17 and C++14 specification. Metal extends the function overloading rules to include the address space attribute of an argument."
- The spec's own helper example: `void foo(device int *g_data, threadgroup int *l_data, constant float *c_data)`, with "The threadgroup attribute is supported here for the pointer l_data only if foo is called by a kernel function".

How the projects do it (read today):

- MLX writes the function twice in `mlx/backend/metal/kernels/sort.h`: `static METAL_FUNC int merge_partition(const threadgroup ValT* As, const threadgroup ValT* Bs, short A_sz, short B_sz, short sort_md)` (line 100) and `static METAL_FUNC int merge_partition(const device ValT* As, const device ValT* Bs, int A_sz, int B_sz, int sort_md)` (line 487).
- PyTorch makes the pointer type a template parameter in `aten/src/ATen/native/mps/kernels/SortMerge.h`: `template <typename Ptr> inline int merge_partition(Ptr A, Ptr B, int a_sz, int b_sz, int diag, bool desc)`, called on `device` pointers (`vi + A0`) and on `threadgroup` arrays (`tgv`).

For the lab: the block-level co-rank (on device memory) and the thread-level co-rank (on threadgroup memory) in `kernel.metal` need two overloads or a template; the `.cu` needs one `__device__` function.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 1.5.1, page 12; section 4, pages 118 to 119)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sort.h
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/SortMerge.h

### C4. Does Apple ship a GPU merge (the counterpart of A7 and B2)?

**Verdict: COULD NOT VERIFY any; none is documented.** Apple ships sort and top-k operations, not a merge.

Searched the macOS 27.0 SDK headers of Metal Performance Shaders and MPSGraph for "merge": the only hit is `mergeLocalityThreshold`, a parameter of an image line detector (`MPSImageEDLines.h`). What exists:

- MPSGraph `sortWithTensor:axis:descending:name:` and `argSortWithTensor:axis:descending:name:` ("Sorts the elements of the input tensor along the specified axis."), `MPS_AVAILABLE_STARTING(macos(13.0), ios(16.0), tvos(16.0))` (`MPSGraphSortOps.h`).
- MPSGraph `topKWithSourceTensor:k:name:`: "Finds the k largest values along the minor dimension of the input. The source must have at least k elements along its minor dimension." (`MPSGraphTopKOps.h`).
- Neither header contains the word "stable" (0 hits each), so the order of equal keys is not documented.
- `MPSMatrixFindTopK` (`MPS_CLASS_AVAILABLE_STARTING( macos(10.13.4), ...)`): "A kernel that find top-K values and their corresponding indices withing a row of a matrix"; `numberOfTopKValues` "Must be less or equal to 16 and requesting more values results in undefined behavior."

So on Apple GPUs the merge is something you write (or take from MLX or PyTorch, D1, D2), and the library top-k is either MPSGraph or the 16-wide matrix kernel.

- Local headers: `MetalPerformanceShadersGraph.framework/Headers/MPSGraphSortOps.h`, `MPSGraphTopKOps.h`; `MetalPerformanceShaders.framework/Frameworks/MPSMatrix.framework/Headers/MPSMatrixFindTopK.h` (macOS 27.0 SDK)
- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/sort(_:axis:descending:name:)
- https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixfindtopk

---

## Group D: how llama.cpp, MLX and PyTorch do this (goal 3, and the top-k bridge)

Read from the default branches on 2026-11-20. Last-change dates are for the file named.

### D1. MLX on Metal: sort, `topk` and `argpartition` all run a merge sort whose merge passes are the book's tiled kernel

**Verdict: CONFIRMED.**

- Kernels, `mlx/backend/metal/kernels/sort.h` (2026-11-05): a multi-block path made of `mb_block_partition` and `mb_block_merge`.
  - `mb_block_partition` computes block-level co-ranks on device memory, for every tile boundary, into a buffer: `for (int i = lid.x; i <= n_blocks; i += tgp_dims.x) { ... int partition = sort_kernel::merge_partition(dev_vals + A_st, dev_vals + B_st, A_ed - A_st, B_ed - B_st, partition_at); block_partitions[i] = A_st + partition; }`.
  - `mb_block_merge` reads `int A_st = block_partitions[block_idx + 0]; int A_ed = block_partitions[block_idx + 1];`, loads exactly `A_sz + B_sz` elements, stores them to `threadgroup ValT tgp_vals[...]`, `threadgroup_barrier(mem_flags::mem_threadgroup);`, then per thread `int sort_md_local = min(A_sz + B_sz, N_PER_THREAD * int(lid.x));` and `merge_partition(tgp_vals, tgp_vals + A_sz, A_sz, B_sz, sort_md_local)` followed by `merge_step`.
  - `merge_partition`'s tie rule: `if (op(b, a)) { A_ed = md; } else { A_st = md + 1; }`: the first run wins ties, as in the book.
- Host, `mlx/backend/metal/sort.cpp` (2026-10-07): `int tn = 4;`, `bn` from 32 up to 512 (256 for element types wider than 4 bytes), `int n_per_block = bn * tn;`, `if (n_blocks > 1) { return multi_block_sort(...); }`. These are code constants, not device figures.
- `ArgPartition::eval_gpu` and `Partition::eval_gpu`: "// We direct arg partition to sort for now" then `gpu_merge_sort(...)`. `topk` in `mlx/ops.cpp` (2026-10-31) is `partition(a, -k, axis_, s)` followed by a slice. A comment in `unique` says "The sort is stable, so the first sorted position of a group holds the first occurrence in the input." (a code comment, not documentation).
- Sampling, mlx-lm `mlx_lm/sample_utils.py` (2026-09-24): `apply_top_k` does `mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]`; `apply_top_p` starts with `sorted_logprobs = mx.sort(logprobs, axis=-1)`.

So on an Apple GPU with MLX, top-k and top-p sampling over a vocabulary-sized row run the chapter's co-rank, block-level in device memory and thread-level in threadgroup memory. Versus the book: block co-ranks are a separate pass for every boundary, not thread 0 inside the merge kernel; exact loading as in Exercise 3; no circular buffer.

- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sort.h (lines 100 to 160 and 487 to 735)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/sort.cpp
- https://github.com/ml-explore/mlx/blob/main/mlx/ops.cpp
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py

### D2. PyTorch: the MPS merge sort documents its choice against the book's thread-0 broadcast; CUDA top-k is radix select

**Verdict: CONFIRMED.**

- MPS, `aten/src/ATen/native/mps/kernels/SortMerge.h` (2026-05-29). File header: "Merge-sort kernels for MPS. Two paths: Path 1 - single-block: one threadgroup per row, used when the segment fits in threadgroup memory. Path 2 - multi-block: segment split into ELEMS_PER_TG-sized blocks, each block sorted independently, then log2(n_blocks) passes of pairwise merges." And: "merge_partition finds the split point on the merge-path diagonal for two sorted runs; merge_step sequentially merges TN elements from that split."
- `mb_merge_body`, the comment on the block-level co-rank: "TG-wide partition: every thread does the same binary search over device memory in registers. Cheaper than the lid.x==0 broadcast: same result, no barriers, no narrowing through InIdxT (which would corrupt at sort_size = 65536 where Ast/Aed can equal size)." The book's Fig. 13.11 does the lid.x == 0 broadcast through `A_S[0]`, `A_S[1]` and two barriers; PyTorch rejects that design for this kernel and says why. A good discussion point for the lesson.
- Then: `tgv[x] = fromA ? vi[Ast + x] : vi[Bst + x - Asz];` for `x < Asz + Bsz` (exact loading), `threadgroup_barrier(mem_flags::mem_threadgroup);`, `int md = min(Asz + Bsz, TN * int(lid)); int ap = merge_partition(tgv, tgv + Asz, Asz, Bsz, md, desc);`, `merge_step`.
- MPS top-k, `kernels/SortTopK.h` (2026-09-29): "topk / kthvalue kernels: reuse the sort load/sort/scatter bodies (SortMerge.h, SortRadix.h) and only change the writeback to emit sorted ranks [offset, offset+count) as out[row*count + (rank-offset)]. topk = {0, k}". `operations/Sort.mm` (2026-11-11) chooses radix or merge with `should_use_radix(sort_size, elem_size, elems_per_tg, is_fp, n_rows)`.
- CUDA, `aten/src/ATen/native/cuda/TensorTopK.cu` (2026-10-15): `gatherTopK` includes `SortingRadixSelect.cuh` and calls `radixSelect<...>` under "// Find the k-th highest element in our input". No merge.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/SortMerge.h (lines 1 to 116 and 376 to 484)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/SortTopK.h
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Sort.mm
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/TensorTopK.cu

### D3. llama.cpp on Metal: the book's basic merge kernel, and "merge per-block top-k lists" shipping today

**Verdict: CONFIRMED.** The lesson's bridge (ours: "merging per-block partial top-k lists is a merge") is literally how llama.cpp's Metal backend computes top-k for short rows and small k.

- `ggml/src/ggml-metal/kernels/argsort.metal` (2026-11-08): `kernel_argsort_f32_i32` sorts one block per threadgroup with a bitonic network in threadgroup memory ("bitonic sort implementation following the CUDA kernels as reference"), writing only the first `args.top_k` indices of each block (`if (i0 + col < args.ne0 && col < args.top_k)`).
- `kernel_argsort_merge_f32_i32` merges two sorted runs straight from device memory, one output chunk per thread: `const int chunk = (total + ntg.x - 1) / ntg.x; const int k0 = tpitg.x * chunk; const int k1 = MIN(MIN(k0 + chunk, total), args.top_k); if (k0 >= args.top_k) { return; }` then `int low = k0 > len1 ? k0 - len1 : 0; int high = MIN(k0, len0); // binary-search partition (i, j) such that i + j = k`, with `take_left = (val0 <= val1);` (ascending) or `(val0 >= val1)` (descending), then a sequential merge. This is the shape of the book's `merge_basic_kernel` (Fig. 13.9): co-rank on global memory, no shared-memory staging, ties to the first run. One search per thread, since the chunk end is not searched; and threads whose chunk starts at or past `top_k` exit at once.
- Host, `ggml/src/ggml-metal/ggml-metal-ops.cpp` (2026-11-19), `ggml_metal_op_top_k_bitonic`: `/*.top_k =*/ std::min(nth, top_k), // for each block, keep just the top_k indices`, then a loop of pairwise merge passes `while (len < args.ne0) { ... /*.top_k =*/ nm == 1 ? top_k : args.ne0, // the final merge outputs top_k elements ... len <<= 1; }`, using the pipeline `kernel_argsort_merge_%s_%s_%s` (`ggml-metal-device.cpp`).
- Path choice, `ggml_metal_op_top_k`: `const bool use_radix = ncols > 2048 && (k > 64 || (nrows > 4 && ncols >= 8192));` with the comment that radix select's fixed per-row cost "is only amortized for long rows, many rows, or a large k; otherwise the bitonic path wins". The radix path is `kernel_top_k_f32_i32` (ch09 D4). The comment also quotes a timing; do not carry it into lessons.
- Full argsort (`ggml_metal_op_argsort`) uses the same bitonic blocks plus merge passes with `top_k = ne00`.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/argsort.metal (lines 11 to 240)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-ops.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.cpp

### D4. Top-k on CUDA and HIP in llama.cpp and CUB: no merge

**Verdict: CONFIRMED.** On the CUDA-family side, top-k is bitonic sort in shared memory, radix select, or CUB's radix-based `DeviceTopK`. Present the merge bridge as one of several routes, the one llama.cpp takes on Metal.

- llama.cpp `ggml/src/ggml-cuda/top-k.cu` (2026-11-19): `top_k_argsort_cuda` (bitonic; "the padded row must fit in shared memory"), `top_k_radix_cuda` (kernels `top_k_radix_histogram`, `top_k_radix_select`, `top_k_radix_gather`), and `top_k_cub` calling `DeviceTopK::MaxPairs(...)`, guarded by "// DeviceTopK has a race condition before CCCL 3.4.3." with `https://github.com/NVIDIA/cccl/pull/10627` and a version test for CCCL 3.4.3 or later.
- CUB is never used in llama.cpp's HIP build: `ggml/src/ggml-cuda/common.cuh` has `#if !defined(GGML_USE_HIP) && (defined(GGML_USE_MUSA) || CUDART_VERSION >= 11070)` / `#define GGML_CUDA_USE_CUB`. On AMD the hand-written bitonic and radix paths run.
- `ggml/src/ggml-cuda/argsort.cu` (2026-10-08): bitonic in shared memory, or CUB `DeviceSegmentedSort::SortPairs` / `DeviceSegmentedRadixSort::SortPairs` (comment: "Currently (confirmed for CCCL <= 3.2) DeviceSegmentedSort does not support stream capture, while DeviceSegmentedRadixSort does.").
- CUB `DeviceTopK` (`cub/device/device_topk.cuh`): "DeviceTopK will return K items in an unspecified order as results. It is based on an algorithm called AIR TopK" (`.. versionadded:: 3.3.0`). "Current support. This release only implements the fully opted-out configuration ... In this configuration the output is unordered and may be non-deterministic: if multiple items tie at the K-th position, the subset of tied elements returned is not uniquely defined and may vary between runs." The agent (`agent_topk.cuh`) builds histograms and merges them with `atomicAdd(global_histogram + ...)`, the chapter 9 pattern.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/top-k.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/argsort.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh (lines 117 to 119)
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_topk.cuh
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/agent/agent_topk.cuh

---

## Group E: our own check (not a vendor fact)

### E1. The book's co-rank, CUB's `MergePath`, the C++ draft's K and a stable merge agree

**Verdict: CONFIRMED** by our computation, which is not a vendor statement.

A Python transcription of Fig. 13.5 as recorded in `notes/ch13.md`, CUB's `MergePath` with `operator<` (A7), the K of [alg.merge] (A6), and "how many of the first k outputs come from A" in a sequential merge with `A[i] <= B[j]` returned the same value on all 39,032 queries: 3,000 random pairs of sorted arrays with lengths 0 to 12 and keys 0 to 6 (many duplicates), every k from 0 to m + n. On the running example A = [1, 7, 8, 9, 10], B = [7, 10, 10, 12], the co-ranks for k = 0 to 9 are [0, 1, 2, 2, 3, 4, 5, 5, 5, 5]: k = 3 gives 2 and k = 4 gives 3 (the notes' traces), k = 6 gives 5 (the text, against Fig. 13.3(b)'s label, notes item 2), k = 8 gives 5 (Exercise 1: i = 5, j = 3).

The searches in llama.cpp (`val0 <= val1` keeps the left side), MLX and PyTorch (`op(b, a)` shrinks the range) test the same predicate as CUB for keys without NaN (our reading of the code in D1 to D3). With NaN, `a <= b` and `!(b < a)` differ; PyTorch's comparator puts NaNs last explicitly.

The script was run from the session scratchpad and is not in the repository. Use it, or the lab's own unit test, as the source of the explorer's numbers; do not copy these by hand.

---

## Answers to the notes' portability questions

The notes list six portability items marked `[unverified]`. Answers:

1. Kernel entry point and 1D index: HIP has CUDA's spelling (ch05 B4); Metal uses `kernel` and position attributes, which hand you the global index directly (ch05 C4, ch02 C1).
2. Dynamically sized shared memory split in two: yes on all three, sized at launch. CUDA and HIP: one `extern __shared__` array, carved by hand (ch05 A7, B4). Metal: one or two `[[threadgroup(n)]]` buffers sized with `setThreadgroupMemoryLength`, multiples of 16 bytes (ch05 C8).
3. Block barrier: `__syncthreads()` on HIP (ch04 B3); `threadgroup_barrier(mem_flags::mem_threadgroup)` on Metal, with the same uniform-control-flow rule (ch05 C2).
4. Calling `co_rank` and `merge_sequential` from kernels: CUDA and HIP need `__device__` (ch02 A7, B4); Metal needs no qualifier, but pointer parameters need an address space and the device and threadgroup versions are separate overloads (C3).
5. Performance reasoning: AMD documents coalescing and occupancy in the same terms (ch06 B1, ch04 B10); Apple documents no coalescing rule (ch06 C1) and exposes occupancy only as an Xcode counter (ch06 C5). No numbers carry over; measure.
6. `ceil` and `min` on integers: `ceil` of an integer quotient is a no-op everywhere (A4, C2); `min`/`max` exist for `int` in device code on all three, with different overload sets for mixed signedness (A5, B1, C1). Not checked: whether `ceil` called with an integer argument even resolves in device code on each backend; the lab should not call it.

---

## Summary of corrections

Items where the book or a likely assumption is WRONG or PARTLY, one line each, plus the one COULD NOT VERIFY:

- A4 (confirms notes item 14): `ceil((m+n)/(blockDim.x*gridDim.x))`, `ceil((m+n)/gridDim.x)` and `ceil((C_length)/tile_size)` divide in integers first, so `ceil` does nothing; 4000 / 1024 gives 3 iterations, not the four on p. 317. Use `(a + b - 1) / b` or `cuda::ceil_div`.
- A5: CUDA documents integer `min`/`max` for device code only, and its mixed signed/unsigned overloads return `unsigned int`; a `__host__ __device__` helper should use ternaries. Whether nvcc also gives host overloads could not be checked (headers not online).
- A7: `thrust::merge` is documented stable, `cub::DeviceMerge` is documented unstable ("Merging is unstable"), and Thrust's CUDA backend calls `cub::DeviceMerge`. The code reads as stable (ties to the first sequence), but only Thrust promises it.
- A7, B2, D1, D2: every library and project merge read today (CUB, rocPRIM, MLX, PyTorch MPS) loads exactly the elements a tile consumes, after computing all block-level co-ranks up front; that is the book's Exercise 3 alternative, so none of them has the 2x over-load of §13.6 or the circular buffer of §13.7.
- D2: PyTorch's MPS merge deliberately replaces the book's thread-0 co-rank plus shared-memory broadcast (Fig. 13.11) with every thread computing it in registers: "same result, no barriers".
- B1: HIP's documentation lists no integer `min`/`max`; they come from Clang's `__clang_hip_math.h`, with same-type device overloads and a template, no mixed signed/unsigned device overloads, and mixed host overloads off by default.
- C2: in MSL, `%` with a negative operand is undefined (C++ defines it); out-of-range integer division gives an unspecified value; `&&` short-circuiting is not restated in the MSL spec (could not verify an explicit statement).
- C4 (could not verify any): Apple documents no GPU merge; MPSGraph has `sort`, `argSort` (macOS 13 and later) and `topK` with no stability statement, and `MPSMatrixFindTopK` is limited to k ≤ 16.
- D4: top-k on the CUDA side (llama.cpp, PyTorch, CUB `DeviceTopK`, rocPRIM) is radix select or bitonic sort, not merge; CUB's `DeviceTopK` currently returns unordered, possibly non-deterministic results, and llama.cpp requires CCCL 3.4.3 or later because of a race in earlier versions.

Facts worth knowing even where the verdict is CONFIRMED:

- A1: the libraries call the co-rank "merge path" (CUB, rocPRIM) or "merge partition" (MLX, PyTorch MPS), citing Odeh et al. 2012 rather than Siebert and Träff 2012.
- A6: `std::merge` is guaranteed stable with the first range winning ties, so the lab can test stability with key–value pairs against it.
- C3: an MSL co-rank used on both device and threadgroup memory needs two overloads (MLX) or a pointer-type template (PyTorch).
- D1: MLX's `topk` and `argpartition` on Metal run a full merge sort, so mlx-lm's top-k and top-p sampling exercise this chapter's co-rank.
- D3: llama.cpp's Metal top-k for short rows and small k keeps the top k of each block and merges the lists pairwise with a co-rank kernel shaped like Fig. 13.9; the lesson's bridge is real code.
- D4: llama.cpp's HIP build uses no CUB or hipCUB.

## Where the backends really differ

One line per cell. "Same" means the same source text, according to the documentation; nothing here has been run on CUDA or AMD hardware by us.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| `co_rank` / `merge_sequential` helper | `__device__` (or `__host__ __device__`) | Same | Plain function, no qualifier |
| One helper for global and on-chip data | Yes: generic `int*` (appC A2) | Yes: generic (appC B1) | No: `device` and `threadgroup` overloads, or a template on the pointer type (C3) |
| Two input tiles on chip | One `extern __shared__` array split in two; bytes as third launch argument | Same | One or two `[[threadgroup(n)]]` buffers; `setThreadgroupMemoryLength`, multiple of 16 bytes |
| 8,192-byte budget for `tile_size` 1024 | `int` is 4 bytes; under the 48 KB default | C++ `int` via Clang (no separate AMD statement); under the LDS limits | `int` is 32-bit; under the 32 KB cap |
| Barrier after loads and before reloading | `__syncthreads()` | `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| Integer `min` / `max` in kernels | `__device__ int min(int, int)`, plus mixed signed/unsigned overloads returning `unsigned` (device-only in the docs) | Clang header: `int`, `float`, `double` overloads and a same-type template; no mixed device overloads | `T min(T, T)` in `<metal_integer>`; both arguments one type |
| Integer division (the `ceil` bug) | Truncates toward zero | Same | Same base; out-of-range quotient unspecified |
| `%` with a negative operand | Defined: `-1 % 1024 == -1` | Same | Undefined |
| `&&` short-circuit (co-rank bounds tests) | C++ guarantee | Same | Not restated; C++17 base |
| Library merge | `thrust::merge` (documented stable); `cub::DeviceMerge` (documented unstable; CCCL 2.7.0, CUDA 12.8) | rocThrust `thrust::merge` (documented stable); `hipcub::DeviceMerge` over `rocprim::merge` (stability not stated) | None documented |
| Library sort / top-k | CUB `DeviceMergeSort` (`StableSort*`), `DeviceTopK` (AIR radix, unordered) | rocPRIM `device_topk` (AIR) | MPSGraph `sort`, `argSort`, `topK` (stability not stated); `MPSMatrixFindTopK`, k ≤ 16 |
| How the projects do top-k | llama.cpp: bitonic, radix select, CUB `DeviceTopK`; PyTorch: radix select | llama.cpp: bitonic, radix select (no CUB) | llama.cpp: per-block bitonic plus co-rank merge for small k, else radix select; MLX: full merge sort; PyTorch MPS: merge or radix sort, writing ranks [0, k) |
| Block-level co-rank in shipping code | CUB: separate partition kernel, one result per tile boundary | rocPRIM: same | MLX: separate partition kernel; PyTorch: every thread recomputes it; llama.cpp: per thread, no tiling |
