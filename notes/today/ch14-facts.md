# Chapter 14 fact check: CUDA, HIP, Metal

Checked on 2026-11-20 against the live vendor and project sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims, the verdict is for the item as a whole and the text says which sub-claim failed. A verdict judges the claim in the item title (the book's statement, or the assumption a course author would bring).
- **Quote** is copied from the source (HTML docs flattened to text, GitHub raw files at a tag or on the default branch, Apple's documentation JSON, PDF text, local SDK headers). Line breaks inside quotes are removed.
- "infra A11", "ch10 A12", "ch09 D4" point at items in `notes/today/infra-facts.md`, `ch10-facts.md`, `ch09-facts.md` and so on. Anything settled there is cited, not re-checked.
- Page numbers ("p. 336") are book pages as recorded in `notes/ch14.md`.
- Versions the sources reported about themselves on the check date: CUDA Math API "v13.5" (last updated Nov 02, 2026); CCCL docs at `nvidia.github.io/cccl/unstable/` and CCCL source on `main` (last commit to the radix-sort dispatch 2026-11-19; latest tagged release v3.5.0, 2026-10-07); ROCm Core SDK 10.2.0 = tag `therock-10.2` of `ROCm/rocm-systems` and `ROCm/rocm-libraries` (HIP docs "HIP 7.17.0"; two hipCUB copies, infra B11); Metal Shading Language Specification "Version 4.1", dated 2026-06-04 ("MSL spec"); Apple documentation JSON and the macOS 27.0 SDK headers on this machine; llama.cpp `master`, MLX `main`, mlx-lm `main`, PyTorch `main` and docs 2.15, Hugging Face transformers `main`, vLLM `main`, all read today.
- Not checked: anything by running code on a GPU. Every vendor statement is from documentation, headers or project source. Where this file says what a piece of project code does on a particular input (ties, -0.0, NaN), that is our reading of the source, labelled as such. Group E is our own arithmetic, run in Python on the host.
- `notes/today/ch11-facts.md` did not exist when this file was written. Vendor facts about scan itself (`cub::DeviceScan`, decoupled look-back and its forward-progress requirement on CUDA and HIP) are left to it; this file covers only what sorting adds.

Totals: 22 CONFIRMED, 0 WRONG, 7 PARTLY, 1 COULD NOT VERIFY (30 items: 11 CUDA, 7 HIP, 4 Metal, 7 project, 1 arithmetic). Failed or unfound sub-claims are collected under "Summary of corrections".

---

## Already settled elsewhere (cite, do not re-check)

| Chapter 14 needs | Settled in |
|---|---|
| "Each thread block may have up to 1024 threads" (p. 337) | CUDA ch05 A5; HIP ch08 B1 (correcting ch03 B3); Metal ch05 C3 (1024 in the tables, real cap per pipeline) |
| Warp, wavefront, SIMD-group width; never a literal 32 | ch05 B5, infra C5, ch04 C2 |
| Coalescing rule, 32-byte sectors, 128-byte lines (the p. 340 argument) | CUDA ch06 A2, A4; AMD ch06 B1; Metal: none documented, ch06 C1 |
| Thrust and CUB reference URLs (refs [8], [9], p. 348) return 404; live pages under `/cccl/unstable/` | ch10 A12 |
| CUB lives in CCCL; toolkit compatibility; `cub::Max()` and friends removed in CCCL 3.0 | infra A11 |
| ROCm 10.2 ships two hipCUB copies (4.7.0 and 5.0.0 under hipCCL) | infra B11 |
| A data race is undefined behaviour in the CUDA memory model (the p. 333 caveat) | ch10 A1 (guide section 5.8: "Any such data race results in undefined behavior.") |
| The atomic for the `hasChanged` flag: `atomicExch`, `cuda::atomic_ref<...>::store` | CUDA infra A1, A2; HIP infra B1, B2; Metal `atomic_store_explicit` on `device atomic_uint*`, infra C1 and ch09 C1 |
| Grid-wide barrier for a kernel-internal `gridExclusiveScan` | CUDA ch04 A7, infra A4; HIP ch04 B11; Metal: none, use separate dispatches, ch04 C3 |
| Threadgroup memory 32 KB; dynamic threadgroup memory | ch05 C1, C8 |
| `simd_prefix_exclusive_sum`, `simd_ballot`, `simd_sum` for the local scan | infra C2 |
| Threadgroup atomic arrays in MSL (local histograms) | ch09 C1 |
| Reading a device-written flag on the host needs the command buffer to complete | ch05 C6, infra C12 |
| Metal default math mode documented three ways | ch05 C7 |
| `-INFINITY` under Metal fast math | ch10 C3 |
| llama.cpp Metal `kernel_top_k_f32_i32`: 256-bin threadgroup histogram over a float-to-uint key | ch09 D4 (extended in D2 below) |
| llama.cpp samples on the host by default; experimental backend sampling exists | ch01 D2 |
| Vocabulary sizes 50,257 / 151,936 / 201,088 for the M5 lab | ch09 D7 |
| PyTorch on ROCm is the CUDA source after hipify | ch01 D4 |

---

## Group A: CUDA and the book's dated statements

### A1. OneSweep is "a state-of-the-art implementation of radix sort for GPUs" (p. 336, reference [3])

**Verdict: CONFIRMED, with an update.** The reference is real and current; the algorithm is now CUB's own default, so "use CUB" (p. 347) and "OneSweep" are no longer two different recommendations.

- arXiv 2206.01784, "[Submitted on 3 Jun 2022]", title as printed "Onesweep: A Faster Least Significant Digit Radix Sort for GPUs", authors "Andy Adinets, Duane Merrill". The book spells it "OneSweep"; the paper spells it "Onesweep".
- Abstract: "Our parallel algorithm employs a method of single-pass prefix sum that only requires ~2n global read/write operations for each digit-binning iteration. This exhibits a significant reduction in last-level memory traffic versus contemporary GPU radix sorting implementations, where each iteration of digit binning requires two passes through the dataset totaling ~3n global memory operations." The abstract then reports a throughput on one named NVIDIA GPU and a speedup "Compared to CUB" (device figures; not for lessons). The "CUB" it beat is CUB's older multi-pass path.
- CUB today, `cub/device/dispatch/tuning/tuning_radix_sort.cuh` (CCCL `main`): `enum class RadixSortAlgorithm { multi_pass, //!< Multi-pass radix sort (upsweep + scan + downsweep per digit)` / `onesweep //!< Single-pass radix sort using decoupled look-back };`
- When it is chosen (same file): `Policy500` has `ONESWEEP = false`; `Policy600` through `Policy800` have `static constexpr bool ONESWEEP = sizeof(KeyT) >= sizeof(uint32_t);`; `Policy900` and `Policy1000` inherit `OnesweepSmallKeyTunedPolicy` with `ONESWEEP = true`. So on compute capability 6.0 to 8.x OneSweep is used for keys of 4 bytes or more, and from 9.0 for every key size. The dispatch (`dispatch_radix_sort.cuh` line 1145) branches on `algorithm == RadixSortAlgorithm::onesweep`.
- The chapter's per-pass structure (scan of bits, then scatter) is the `multi_pass` path; OneSweep folds the scan into the scatter with decoupled look-back, which is Chapter 11's §11.9 technique (notes/ch11.md, pp. 279 to 280).

- https://arxiv.org/abs/2206.01784
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/dispatch/tuning/tuning_radix_sort.cuh (lines 50 to 54, 1163 to 1261, 1824 to 1835)
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/dispatch/dispatch_radix_sort.cuh (lines 560 to 590, 1140 to 1150)

### A2. "Use Thrust or CUB" (p. 347): `cub::DeviceRadixSort` today, keys with values, descending, stability

**Verdict: CONFIRMED.** Everything the lab's library comparison needs is documented: pairs, descending order, a bit range, a stated stability guarantee, and the two-phase temporary-storage call.

- Methods: `SortKeys`, `SortKeysDescending`, `SortPairs`, `SortPairsDescending` (twelve overloads each on the docs page).
- Signature as printed (current overload): `static inline cudaError_t SortPairsDescending(void * d_temp_storage, size_t & temp_storage_bytes, const KeyT * d_keys_in, KeyT * d_keys_out, const ValueT * d_values_in, ValueT * d_values_out, NumItemsT num_items, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8, const EnvT & env = {})`. "Sorts key-value pairs into descending order using ≈2N auxiliary storage." The docs mark these overloads "Added in version 2.2.0: First appears in CUDA Toolkit 12.3." I did not find when the method itself first appeared.
- Two-phase call: "If d_temp_storage is nullptr, the required size is written to temp_storage_bytes without dereferencing iterators or launching kernels." The page's example calls the function twice around a `cudaMalloc`.
- Restrictions: "Pointers to contiguous memory must be used; iterators are not currently supported." "In-place operations are not supported." "This operation requires an allocation of temporary device storage that is O(N+P) ... For sorting using only O(P) temporary storage, see the sorting interface using DoubleBuffer wrappers below."
- Stability: "DeviceRadixSort is stable. For floating-point types, -0.0 and +0.0 are considered equal and appear in the result in the same order as they appear in the input."
- Example on the page: keys `[8, 6, 7, 5, 3, 0, 9]`, values `[0, 1, 2, 3, 4, 5, 6]`, descending output keys `[9, 8, 7, 6, 5, 3, 0]` with values `[6, 0, 2, 1, 3, 4, 5]`. That is the M5 output shape: values are the token indices.
- Complexity: "The work-complexity of radix sort as a function of input size is linear".
- Supported key types include `__half` and `__nv_bfloat16`, so bf16 logits can be sorted as 16-bit keys.

- https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceRadixSort.html

### A3. Float keys in CUB: the documented bit transform, signed zeros, NaN, descending

**Verdict: CONFIRMED.** The key transform this course needs for logits is documented by NVIDIA, and CUB adds one step for signed zeros that a naive version lacks.

Docs (same page as A2):

- "Although the direct radix sorting method can only be applied to unsigned integral types, DeviceRadixSort is able to sort signed and floating-point types via simple bit-wise transformations that ensure lexicographic key ordering."
- "For unsigned integral values, the keys are used directly. For signed integral values, the sign bit is inverted. For positive floating point values, the sign bit is inverted. For negative floating point values, the full key is inverted. For floating point types, positive and negative zero are a special case and will be considered equivalent during sorting."
- "If descending sort is used, the keys are inverted after performing any type-specific transformations, and the resulting keys are sorted in ascending order."
- "These transformations must be considered when restricting the [begin_bit, end_bit) range, as the bitwise transformations will occur before the bit-range truncation."
- "No special handling is implemented for NaN values; these are sorted according to their bit representations after any transformations."

Source, `cub/detail/numeric_traits.cuh` (float traits):

```cpp
_CCCL_HOST_DEVICE_API static _CCCL_FORCEINLINE UnsignedBits TwiddleIn(UnsignedBits key)
{
  const UnsignedBits mask = (key & HIGH_BIT) ? UnsignedBits(-1) : HIGH_BIT;
  return key ^ mask;
};
```

Source, `cub/block/radix_rank_sort_operations.cuh`, the signed-zero step: "Twiddling alone maps -0.0f to 0x7fffffff and +0.0f to 0x80000000 for float, which are subsequent bit patterns and bitwise complements of each other. For onesweep, both -0.0f and +0.0f are mapped to the bit pattern of +0.0f (0x80000000) for ascending sort, and to the pattern of -0.0f (0x7fffffff) for descending sort. For all other sorting implementations in CUB, both are always mapped to +0.0f." The code: `return key == TWIDDLED_MINUS_ZERO_BITS ? TWIDDLED_ZERO_BITS : key;`, applied when the digit is extracted, so the stored key keeps its sign.

What this means for the course (ours): the transform in our lessons is the same as CUB's, so it can be presented as "ours, and also what CUB documents". Two details a reader's own kernel must copy to match CUB and `std::stable_sort`: map -0.0 to +0.0 before digit extraction, and get descending order by inverting the key, not by reversing an ascending result (which would reverse the order of ties). Group E checks the bit patterns.

- https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceRadixSort.html (sections "Floating-Point Special Cases", "Transformations", "Stability")
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/detail/numeric_traits.cuh (lines 170 to 183)
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/radix_rank_sort_operations.cuh (lines 38 to 79, 532 to 570)

### A4. The choice of radix value (§14.7) and coarsening (§14.8) in CUB

**Verdict: CONFIRMED** that CUB exposes both of the chapter's knobs and picks a digit width per architecture. The book gives no number (p. 344); CUB's numbers are a tuning, not a rule.

- Digit widths in `tuning_radix_sort.cuh`: OneSweep `ONESWEEP_RADIX_BITS = 8` in every policy; multi-pass `PRIMARY_RADIX_BITS = (sizeof(KeyT) > 1) ? 7 : 5` and `SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5` (Policy500 to Policy800). So for 32-bit keys: 4 passes at 8 bits (OneSweep), or 5 passes at 7 bits (multi-pass). The comments beside some policies record measured throughputs on named GPUs; they are not copied here.
- The docs' own tuning example sets `.histogram = {... .radix_bits = 8}`, `.onesweep = {... .items_per_thread = 21, ... .radix_bits = 8}`, `.downsweep = {... .items_per_thread = 25, ... .radix_bits = 7}`. `items_per_thread` is the book's coarsening factor under another name.
- Block-level sort, the chapter's "local sort in shared memory": `template <typename KeyT, int BlockDimX, int ItemsPerThread, typename ValueT = NullType, int RadixBits = 4, ...> class BlockRadixSort`. Docs: "BlockRadixSort is stable. For floating-point types -0.0 and +0.0 are considered equal and appear in the result in the same order as they appear in the input." "Efficiency is increased with increased granularity ITEMS_PER_THREAD. Performance is also typically increased until the additional register pressure or shared memory allocation size causes SM occupancy to fall too low." Descending differs from the device sort: "Unlike DeviceRadixSort, BlockRadixSort does not invert the input key bits when performing a descending sort. Instead, it has special logic to reverse the order of the keys while sorting."

- https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/dispatch/tuning/tuning_radix_sort.cuh
- https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceRadixSort.html (section "Tuning")
- https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockRadixSort.html

### A5. Sorting every row of a batch: `DeviceSegmentedRadixSort` and `DeviceSegmentedSort`

**Verdict: CONFIRMED** that CUB has a stable, descending, row-wise sort of (key, value) pairs, the M5 operation. One trap: the plain `DeviceSegmentedSort` methods are documented as not stable.

- `DeviceSegmentedRadixSort`: `SortPairsDescending(void* d_temp_storage, size_t& temp_storage_bytes, const KeyT* d_keys_in, KeyT* d_keys_out, const ValueT* d_values_in, ValueT* d_values_out, ::cuda::std::int64_t num_items, ::cuda::std::int64_t num_segments, BeginOffsetIteratorT d_begin_offsets, EndOffsetIteratorT d_end_offsets, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8, const EnvT& env = {})`. "When input a contiguous sequence of segments, a single sequence segment_offsets (of length num_segments + 1) can be aliased for both the d_begin_offsets and d_end_offsets parameters". "Note, the size of any segment may not exceed INT_MAX." Its page says only "DeviceSegmentedRadixSort shares its implementation with DeviceRadixSort. See that algorithm's documentation for more information." Stability is therefore stated by reference, not on its own page.
- `DeviceSegmentedSort`: "The underlying sorting algorithm is undefined. Depending on the segment size, it might be radix sort, merge sort or something else." "DeviceSegmentedRadixSort is optimized for significantly large segments (tens of thousands of items and more)." `DeviceSegmentedSort` targets "segment size imbalance or moderate segment sizes (up to thousands of items)". Methods: `SortKeys`, `SortKeysDescending`, `SortPairs`, `SortPairsDescending`, `StableSortKeys`, `StableSortKeysDescending`, `StableSortPairs`, `StableSortPairsDescending`.
- "SortKeys is not guaranteed to be stable." (same sentence for the other non-`Stable` methods). "StableSortPairsDescending is stable: it preserves the relative ordering of equivalent elements."
- For a batch of vocabulary rows (tens of thousands to hundreds of thousands of items each, ch09 D7), the documented fit is `DeviceSegmentedRadixSort::SortPairsDescending` or `DeviceSegmentedSort::StableSortPairsDescending`.

- https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceSegmentedRadixSort.html
- https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceSegmentedSort.html

### A6. Thrust: which sort `thrust::sort` and `thrust::stable_sort_by_key` actually run

**Verdict: CONFIRMED.** For arithmetic keys with a less or greater comparator, Thrust's CUDA backend calls CUB's radix sort; for anything else it calls CUB's merge sort.

- `thrust/system/cuda/detail/sort.h`: "Smart sort picks at compile-time whether to dispatch radix or merge sort". Radix dispatch maps `less` to `cub::DeviceRadixSort::SortKeys`/`SortPairs` and `greater` to `SortKeysDescending`/`SortPairsDescending`; the other branch calls `cub::DeviceMergeSort::SortKeys`/`SortPairs`.
- Eligibility, `cub/device/device_radix_sort.cuh`: `__can_use_radix_sort = (::cuda::std::is_arithmetic_v<ValueType> || ... __half ... __nv_bfloat16) && ::cuda::std::__is_one_of_v<..., ::cuda::std::less<>, ::cuda::std::less<ValueType>, ::cuda::std::greater<>, ::cuda::std::greater<ValueType>>`.
- So `thrust::sort` on `float` keys is stable in practice (it is the radix sort), although `thrust::sort` promises no stability; a custom comparator lambda silently moves to merge sort.
- On `main`, `thrust::greater` and `thrust::less` are deprecated aliases: `using greater CCCL_DEPRECATED_BECAUSE("Use cuda::std::greater instead") = ::cuda::std::greater<T>;`. Whether a given toolkit's CCCL already carries the deprecation was not checked.

- https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/system/cuda/detail/sort.h (lines 140 to 330)
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_radix_sort.cuh (lines 52 to 68)
- https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/functional.h (lines 133 to 142)

### A7. Top-k without a full sort: `cub::DeviceTopK` and `cub::DeviceBatchedTopK`

**Verdict: CONFIRMED.** CUB now has a selection primitive, which supports the lesson's statement that top-k does not need a full sort. Its current guarantees are narrow.

- "DeviceTopK provides device-wide, parallel operations for finding the largest (or smallest) K items from sequences of unordered data items". "DeviceTopK will return K items in an unspecified order as results. It is based on an algorithm called AIR TopK" (link: https://dl.acm.org/doi/10.1145/3581784.3607062). Methods `MaxKeys`, `MaxPairs`, `MinKeys`, `MinPairs`; first overloads "Added in version 3.3.0", later ones 3.4.0 (CUDA 13.4) and 3.5.0 (CUDA 13.5).
- Current support: "This release only implements the fully opted-out configuration, which must be requested explicitly: cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::tie_break::unspecified, cuda::execution::output_ordering::unsorted)." "In this configuration the output is unordered and may be non-deterministic: if multiple items tie at the K-th position, the subset of tied elements returned is not uniquely defined and may vary between runs." The documented future default is "deterministic results ... ties resolved toward the smaller (lower) source index ... and stable-sorted output".
- `DeviceBatchedTopK`: "Added in version 3.5.0: First appears in CUDA Toolkit 13.5." "Given a batch of segments, DeviceBatchedTopK finds, independently for each segment, the K largest (or smallest) items." Constraints: host-only; "On pre-Hopper GPUs (compute capability < 9.0) every segment must be processable by a single thread block"; from 9.0 "the thread-block-cluster backend also handles larger segments"; "an individual segment is currently limited to a maximum of 2^21 (about 2 million) items".
- For the sampler (ours): top-k is a selection, so the M5 sort lab should not be read as "how top-k is done"; module X1 owns selection.

- https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceTopK.html
- https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceBatchedTopK.html

### A8. Reinterpreting a float's bits in CUDA device code

**Verdict: CONFIRMED.**

- CUDA Math API, Type Casting Intrinsics: `__device__ unsigned int __float_as_uint(float x)`: "Reinterpret the bits in the single-precision floating-point value x as an unsigned integer." `__device__ float __uint_as_float(unsigned int x)`: "Reinterpret the bits in the unsigned integer value x as a single-precision floating-point value." Also `__float_as_int`, `__int_as_float`.
- libcu++: "cuda::std::bit_cast extended the standard std::bit_cast to also recognize CUDA extended floating-point scalar and vector types as trivially copyable." Header `<cuda/std/bit>`.
- These are what llama.cpp (D2) and PyTorch (D5) call in their key transforms.

- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html
- https://nvidia.github.io/cccl/unstable/libcudacxx/standard_api/numerics_library/bit.html

### A9. The benign race on `hasChanged` (p. 333)

**Verdict: CONFIRMED** (cited, not re-checked). The book's caveat matches NVIDIA's current text: the CUDA C++ memory model makes any data race undefined behaviour (ch10 A1), idempotent or not. The book does not name the atomic; the documented choices are `atomicExch(flag, 1u)` (infra A1, relaxed, device scope) or `cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(*flag).store(1u, cuda::memory_order_relaxed)` (infra A2). HIP: B7. Metal: `atomic_store_explicit` on a `device atomic_uint*` (infra C1, ch09 C1).

### A10. The scatter's coalescing: "a warp writes to two buckets, most likely two cache lines, or a little more" (p. 340)

**Verdict: PARTLY.** True in cache-line terms; in NVIDIA's documented unit (32-byte sectors, ch06 A2) the cost of a 1-bit scatter is smaller than "twice", and the cost of a wide radix is larger than the line picture suggests. The arithmetic below is ours, from ch06 A2 and A4.

- In Fig. 14.7 the zeros of one warp get consecutive destinations (each zero's `i - numOnesBefore` is one more than the previous zero's), and so do the ones. So a warp of 32 threads writing 4-byte keys writes two contiguous runs of z and 32 - z keys.
- A run of b bytes starting at offset o covers ceil(((o mod 32) + b) / 32) sectors and at most two 128-byte lines when b is 128 or less. Two runs: between 1 and 4 lines (the book's "two, or a little more"), and between 4 and 7 sectors, against 4 sectors for a fully coalesced warp store.
- With an r-bit digit a warp writes up to 2^r runs; for 8-bit digits and random keys most of a warp's 32 keys land in different buckets, which approaches one sector per key (32 sectors). That is the growth §14.7 warns about, and why the local sort of §14.6 matters more as r grows.
- AMD: a 64-wide wavefront and 64- or 128-byte lines change the counts (ch06 B1). Metal: no rule documented (ch06 C1); measure.

### A11. `std::stable_sort` as the labs' CPU baseline

**Verdict: CONFIRMED.** Not a vendor fact, but the labs measure against it and test against it.

- cppreference: "Sorts the elements in the target range [first, last). The order of equivalent elements is guaranteed to be preserved." Complexity: "O(N·log²(N)) applications of the comparator comp (or only O(N·log(N)) comparisons if enough extra memory is available)."
- The comparator must induce a strict weak ordering ("For all a, comp(a, a) == false. If comp(a, b) == true then comp(b, a) == false. if comp(a, b) == true and comp(b, c) == true then comp(a, c) == true", with equivalence transitive). Our inference: `>` on floats containing NaN breaks transitivity of equivalence (NaN is "equivalent" to every value), so a CPU reference must either keep NaN out of the input or compare the transformed integer keys. With `>` on non-NaN floats, -0.0 and +0.0 are equivalent and keep input order, which is what CUB and rocPRIM produce (A3, B2) and what a GPU transform without the -0.0 step does not.
- `std::sort` and `std::partial_sort` carry no stability guarantee; llama.cpp's host sampler uses them (D3). Apple's `vDSP_vsorti` is not stable either (C2).

- https://en.cppreference.com/w/cpp/algorithm/stable_sort.html
- https://en.cppreference.com/w/cpp/named_req/Compare.html

---

## Group B: HIP / ROCm

### B1. hipCUB `DeviceRadixSort`, `DeviceSegmentedRadixSort`, `DeviceSegmentedSort`

**Verdict: CONFIRMED.** Same class and method names as CUB in both hipCUB copies at `therock-10.2`, implemented on rocPRIM.

- `projects/hipcub` (4.7.0) and `projects/hipccl/hipccl3/hipcub` (5.0.0), `backend/rocprim/device/device_radix_sort.hpp`: `SortPairs`, `SortPairsDescending`, `SortKeys`, `SortKeysDescending`, each forwarding to `rocprim::radix_sort_pairs`, `rocprim::radix_sort_pairs_desc`, `rocprim::radix_sort_keys`, `rocprim::radix_sort_keys_desc`.
- `device_segmented_sort.hpp` in both copies has `SortPairs`, `SortPairsDescending`, `StableSortPairs` and `StableSortPairsDescending` (plus the key-only forms). In 4.7.0 the segmented sort is built on `rocprim::segmented_radix_sort_pairs(_desc)`, and `StableSortPairsDescending` simply returns `SortPairsDescending(...)`.
- Header and namespace: `<hipcub/hipcub.hpp>`, `hipcub::` (infra B11). The HIPIFY "CUB API supported by HIP" table lists only CUB data types, not these classes, so it is no guide to which sort methods exist; the headers are.
- So the CUB call of A2 and A5 ports by renaming `cub::` to `hipcub::` and the include; nothing else in the call changes.

- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipcub/hipcub/include/hipcub/backend/rocprim/device/device_radix_sort.hpp
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipcub/hipcub/include/hipcub/backend/rocprim/device/device_segmented_sort.hpp
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipccl/hipccl3/hipcub/hipcub/include/hipcub/backend/rocprim/device/device_segmented_sort.hpp
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUB_API_supported_by_HIP.html

### B2. rocPRIM radix sort: stability, float keys, signed zeros

**Verdict: PARTLY.** Stability, the float transform and the signed-zero fix match CUB. One documented difference: a custom bit range is not supported for float keys.

- `rocprim/device/device_radix_sort.hpp`, `radix_sort_keys`: "\p radix_sort_keys is \b stable: it preserves the relative ordering of equivalent keys." Key type "must be an arithmetic type (that is, an integral type or a floating-point type)".
- `begin_bit`: "Must be in range [0; 8 * sizeof(Key)). Default value: 0. Non-default value not supported for floating-point key-types." (same for `end_bit`). CUB allows a range and documents that it applies after the transform (A3).
- Float codec, `rocprim/type_traits.hpp` (the old `thread/radix_key_codec.hpp` now only emits "Functionality from rocprim/detail/radix_key_codec.hpp has been moved to rocprim/type_traits.hpp."): `bit_key ^= (sign_bit & bit_key) == 0 ? sign_bit : bit_key_type(-1);`, and descending is `return Descending ? ~bit_key : bit_key;`.
- Signed zeros: "Digit extractor takes into account that (+0.0 == -0.0) is true for floats, so both +0.0 and -0.0 are reflected into the same bit pattern for digit extraction." The comment table gives 0.0 as `0x8000'0000` ascending and `0x7FFF'FFFF` descending, -0.0 the reverse, and "For ascending sort, both should be mapped to 0x8000'0000, and for descending sort, both should be mapped to 0x7FFF'FFFF."

- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocprim/rocprim/include/rocprim/device/device_radix_sort.hpp (lines 870 to 966)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocprim/rocprim/include/rocprim/type_traits.hpp (lines 484 to 615, 664 to 735)

### B3. OneSweep and digit width on AMD; hipCUB `BlockRadixSort`

**Verdict: PARTLY.** rocPRIM implements OneSweep and tunes 8-bit digits almost everywhere. hipCUB's block sort accepts CUB's `RADIX_BITS` parameter and ignores it.

- `device_radix_sort.hpp` contains `radix_sort_onesweep_histogram_impl`, `radix_sort_onesweep_iteration` with `onesweep_lookback_state* lookback_states`: the same single-pass, look-back design as A1.
- Tuned configs, `device/detail/config/device_radix_sort_onesweep.hpp`: 384 config lines; `'radix_bits': 8` in 374, `4` in 8, `6` in 2. Targets named include gfx906, gfx908, gfx90a, gfx942, gfx950, gfx1030, gfx1100, gfx1201.
- hipCUB `BlockRadixSort` (4.7.0): `template<typename KeyT, int BLOCK_DIM_X, int ITEMS_PER_THREAD, typename ValueT = NullType, int RADIX_BITS = 4, /* ignored */ bool MEMOIZE_OUTER_SCAN = true, /* ignored */ BlockScanAlgorithm INNER_SCAN_ALGORITHM = BLOCK_SCAN_WARP_SCANS, /* ignored */ ...> class BlockRadixSort : private ::rocprim::block_radix_sort<KeyT, BLOCK_DIM_X, ITEMS_PER_THREAD, ValueT, BLOCK_DIM_Y, BLOCK_DIM_Z>`. Code that sets the digit width through hipCUB compiles and does nothing.

- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocprim/rocprim/include/rocprim/device/detail/config/device_radix_sort_onesweep.hpp
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipcub/hipcub/include/hipcub/backend/rocprim/block/block_radix_sort.hpp (lines 82 to 110)

### B4. Top-k and partial sorting on AMD

**Verdict: PARTLY.** rocPRIM has selection primitives; hipCUB at 10.2 does not wrap them, so CUB's `DeviceTopK` call does not port by renaming.

- rocPRIM device headers at the tag include `device_topk.hpp` (`topk`, `topk_pairs`: "Find the largest or smallest K elements from an input array of keys."), `device_segmented_topk.hpp` (`segmented_topk`, `segmented_topk_pairs`: "Returns the K largest or smallest elements for each segment. These may be be in any order."), `device_nth_element.hpp` ("nth_element is not stable"), `device_partial_sort.hpp` ("partial_sort is not stable").
- `segmented_topk` template notes: "Ordered output is not supported yet." "Deterministic output is not supported yet." The implementation picks a "Naive segmented top-k" (ordered, stable) or "Segmented top-k air" (not ordered, not deterministic, not stable).
- hipCUB device directories in both copies list `device_merge_sort.hpp`, `device_radix_sort.hpp`, `device_segmented_radix_sort.hpp`, `device_segmented_sort.hpp`, `device_select.hpp` and no top-k header.

- https://github.com/ROCm/rocm-libraries/tree/therock-10.2/projects/rocprim/rocprim/include/rocprim/device
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocprim/rocprim/include/rocprim/device/device_segmented_topk.hpp (lines 120 to 240)
- https://github.com/ROCm/rocm-libraries/tree/therock-10.2/projects/hipcub/hipcub/include/hipcub/backend/rocprim/device

### B5. rocThrust: what `thrust::sort` runs on AMD

**Verdict: CONFIRMED.** Same split as Thrust (A6): `less` and `greater` on arithmetic keys go to `rocprim::radix_sort_keys`, `radix_sort_keys_desc`, `radix_sort_pairs`, `radix_sort_pairs_desc`; otherwise `rocprim::merge_sort`. Identical in `projects/rocthrust` and `projects/hipccl/hipccl3/rocthrust` at the tag (`thrust/system/hip/detail/sort.h`, lines 88 to 284).

- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocthrust/thrust/system/hip/detail/sort.h

### B6. `__float_as_uint` and `__uint_as_float` on HIP

**Verdict: PARTLY.** Both exist with the CUDA spelling, so the `.cu` key transform compiles unchanged; but they are not in AMD's published math API reference.

- Header `hipamd/include/hip/amd_detail/amd_device_functions.h` at the tag: `__device__ static inline unsigned int __float_as_uint(float x) { static_assert(sizeof(unsigned int) == sizeof(float), ""); unsigned int tmp; __builtin_memcpy(&tmp, &x, sizeof(tmp)); return tmp; }` and the matching `__uint_as_float`.
- The HIP 7.17.0 "HIP math API" page and its source `docs/reference/math_api.rst` contain no "float_as", "cast" or "reinterpret" (searched). The names appear in `docs/device_md_gen.py` and in hip-tests, not in a published reference page.
- llama.cpp's HIP build uses `__float_as_uint` in its top-k (D2) through the shared `.cu` source.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_device_functions.h (lines 528 to 597)
- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/math_api.html

### B7. Does HIP document the data-race rule behind the `hasChanged` caveat?

**Verdict: COULD NOT VERIFY.** A search of the HIP documentation tree at the tag for "data race" returns three passages (asynchronous calls, the programming model's note that the C++ abstract machine assumes "the absence of data races", and the histogram tutorial's advice to restrict atomics "to code paths where data races cannot be eliminated"). None states what a racy plain write means in HIP. HIP is compiled as C++ by Clang, where the standard's rule applies; that is our inference, not AMD's statement. The atomic itself is the same `atomicExch` (infra B1).

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/understand/programming_model.rst

---

## Group C: Metal

### C1. What Apple documents for sorting on the GPU

**Verdict: PARTLY.** Apple documents graph-level sort, argsort and top-k, and one small top-k kernel. It documents no stability or tie order for any of them, no sort in MSL, and no sorting sample.

| API | Availability | What Apple says |
|---|---|---|
| `MPSGraph.sort(_:axis:descending:name:)` | macOS 13.0 | "Sorts the elements of the input tensor along the specified axis." `descending`: "If true, reverse the sort direction" |
| `MPSGraph.argSort(_:axis:descending:name:)` | macOS 13.0 | "Computes the indices that sort the elements of the input tensor along the specified axis." Returns "A valid MPSGraphTensor object with 32-bit integer data type" |
| `MPSGraph.topK(_:axis:k:name:)` | macOS 14.0 | "Finds the k largest values along the minor dimension of the input." Returns values and indices |
| `MPSMatrixFindTopK` | macOS 10.13.4 | "A kernel that find top-K values and their corresponding indices withing a row of a matrix". `numberOfTopKValues`: "Must be less or equal to 16 and requesting more values results in undefined behavior." |

- Stability and ties: the documentation pages and the SDK headers `MPSGraphSortOps.h`, `MPSGraphTopKOps.h` contain no "stable", "tie" or "equal" (searched). So `argSort` cannot be assumed stable, and the M5 lab's stable-ties requirement cannot be met by citing Apple.
- MSL spec 4.1: no occurrence of "sort", "sorted", "sorting" or "radix" (searched).
- Metal sample code library: of the 51 references the page's documentation JSON links today, none is about sorting, scan or reduction (searched titles and abstracts; the only hits for "reduc" are about reducing CPU overhead or render passes).
- Context: an Apple Developer Forums thread from July 2018 asked why MPS lacks "Reduce, Scan or even Radix sort"; the Apple engineer's accepted reply asked developers to file requests. The MPSGraph sort APIs above arrived later (macOS 13).
- For the course: the Metal lab writes its own radix sort, and the "library comparison" on Metal is MPSGraph `argSort` with its stability unstated.

- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/sort(_:axis:descending:name:)
- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/argsort(_:axis:descending:name:)
- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/topk(_:axis:k:name:)
- https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixfindtopk
- Local header: `MetalPerformanceShaders.framework/Frameworks/MPSMatrix.framework/Headers/MPSMatrixFindTopK.h` (macOS 27.0 SDK), lines 15 to 125
- https://developer.apple.com/documentation/metal/metal-sample-code-library
- https://developer.apple.com/forums/thread/105886

### C2. Accelerate's sorts are on the CPU, and the index sort is not stable

**Verdict: CONFIRMED.** Relevant because a Mac reader might reach for Accelerate as the CPU reference.

- `vDSP_vsorti(const float *__C, vDSP_Length *__I, vDSP_Length *__Temporary, vDSP_Length __N, int __Order)`: "Performs an in-place sort of the indices into a single-precision vector." `__Order`: "Pass 1 to specify ascending order, or -1 for descending order." "Note that this function doesn't provide a stable sort. So, if the function compares two elements as equal, it may place them in any order within the sorted array."
- `vDSP_vsort`: "Performs an in-place sort of a single-precision vector."
- The lab's CPU reference should be `std::stable_sort` (A11), not vDSP.

- https://developer.apple.com/documentation/accelerate/vdsp_vsorti
- https://developer.apple.com/documentation/accelerate/vdsp_vsort

### C3. `as_type<uint>(float)` for the key transform, and the float rules it must survive

**Verdict: CONFIRMED.** The reinterpretation is documented to copy bits unchanged. Metal's fast-math rules make a float-side treatment of -0.0, infinities or NaN unreliable, so the transform should be integer-only after `as_type`.

- MSL spec section 2.23 "Type Conversions and Reinterpreting Data": "Metal adds an as_type<type-id> operator to allow any scalar or vector data type (that is not a pointer) to be reinterpreted as another scalar or vector data type of the same size. The bits in the operand are returned directly without modification as the new type." Example as printed: `float f = 1.0f; // Legal. Contains: 0x3f800000` / `uint u = as_type<uint>(f);`
- Section 1.6.3, fast math includes "No Signed Zeroes: Allow optimizations to treat the sign of a zero argument or result as insignificant." Both `fast` ("no NaNs, no INFs, no signed zeros, ...") and `relaxed` ("honors INFs and NaNs. These include no signed zeros, ...") include it, so whichever is the default (ch05 C7), a float test such as `x == 0.0f ? 0.0f : x` may be rewritten.
- Section 8.1: "If fast math is enabled the behavior of handling NaN or INF (as inputs or outputs) is undefined." "Denormalized single-precision ... numbers passed as input to or produced as the output of single-precision ... arithmetic operations may be flushed to zero."
- Consequence (ours): load the float, `as_type<uint>` it, and do the sign flip, the -0.0 fix (`u == 0x80000000u ? 0u : u`) and the descending inversion in `uint`. No float operation remains for fast or relaxed math to rewrite, and subnormal probabilities keep their order. PyTorch's MPS transform uses a float test `val != val` for NaN (D6); under fast math that test has undefined behaviour by the rule above.
- 64-bit keys are available if a lab packs (key, index) into one key: `long`/`ulong` "All OS: Metal 2.2 and later".

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 1.6.3, 2.1, 2.23, 8.1; pages 15, 25, 110, 368)

### C4. Building a radix pass on Metal: what exists, and what a single-pass design would need

**Verdict: PARTLY.** Every piece of the chapter's multi-kernel design exists (table below, each row cited). Nothing documents the inter-threadgroup progress a OneSweep-style single pass relies on.

| Chapter step | Metal piece | Where settled |
|---|---|---|
| Extract a bit, count ones in a SIMD-group | `simd_ballot`, `simd_sum` | infra C2 |
| Block-local exclusive scan | `simd_prefix_exclusive_sum` plus threadgroup memory | infra C2, ch05 C1 |
| Local buckets in shared memory | `threadgroup` arrays, 32 KB total | ch05 C1, C8 |
| Per-block digit histogram | `threadgroup atomic_uint` | ch09 C1 |
| Grid-wide scan of the bucket table | separate dispatch; no grid barrier | ch04 C3 |
| Host loop over passes and the odd-even flag | one command buffer per readback | ch05 C6, infra C12 |

- Forward progress: the MSL spec's only statement is within a threadgroup: "SIMD-groups execute concurrently within a given threadgroup and make independent forward progress with respect to each other, in the absence of threadgroup barrier operations." (section 5.2.3.6). Nothing is said about one threadgroup waiting on another (searched "forward", "spin", "deadlock", "starvation"). So a decoupled look-back on Metal has no documented basis; the count, scan and scatter dispatches are the documented route, and that is what PyTorch's MPS radix sort does (D6).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 5.2.3.6)

---

## Group D: How llama.cpp, MLX, PyTorch (and two serving stacks) sort for sampling

All files read on the default branch today; dates are the last commit to the file. These are snapshots: link to files, not line numbers, in lessons.

### D1. llama.cpp `argsort` on CUDA and HIP: bitonic in shared memory, CUB for long rows, nothing for long rows on HIP

**Verdict: CONFIRMED** as source read; stability and the HIP consequence are our reading.

- `ggml/src/ggml-cuda/argsort.cu` (2026-10-08). Rows whose padded length fits shared memory and that have at most 1024 columns use a bitonic network: "Bitonic sort of one row per block. Each thread owns the columns threadIdx.x + i * blockDim.x". `const int ncols_pad = next_power_of_2(ncols);` `GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);`. That is §14.9's sorting network, used where the book says networks win: short sequences.
- Longer rows use CUB with the token index as the value: one row `DeviceRadixSort::SortPairs(Descending)`; several rows under CUDA-graph capture `DeviceSegmentedRadixSort::SortPairs(Descending)` (comment: "DeviceSegmentedSort does not support stream capture, while DeviceSegmentedRadixSort does", "confirmed for CCCL <= 3.2"); otherwise `DeviceSegmentedSort::SortPairs(Descending)`.
- `common.cuh`: `#if !defined(GGML_USE_HIP) && (defined(GGML_USE_MUSA) || CUDART_VERSION >= 11070)` / `#define GGML_CUDA_USE_CUB`. So the HIP build has no CUB or hipCUB path.
- `ggml-cuda.cu`, `supports_op`: `case GGML_OP_ARGSORT:` / `#ifndef GGML_CUDA_USE_CUB` / "bitonic path: the padded row must fit in shared memory" / `return ncols_pad * sizeof(int) <= ...smpb;`. A vocabulary row (151,936 pads to 262,144 ints, 1 MiB) is therefore reported unsupported on HIP, and ggml schedules it on another backend (our reading of `supports_op`).
- Stability (ours): the bitonic path compares with strict `>`/`<` and is not stable; `DeviceSegmentedSort::SortPairs` is documented as not stable (A5); only the single-row and graph-capture paths are stable radix sorts. So ties in `ggml_argsort` on CUDA may come out in any order depending on the path.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/argsort.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh (lines 117 to 119)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu (`supports_op`, `case GGML_OP_ARGSORT`)

### D2. llama.cpp top-k: a radix select, CUB's `DeviceTopK`, and the same float key on CUDA, HIP and Metal

**Verdict: CONFIRMED** as source read.

- `ggml/src/ggml-cuda/top-k.cu` (2026-11-19). The key transform, without a -0.0 fix:

```cpp
static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}
```

- `top_k_radix_cuda`: `constexpr int RADIX_BITS = 8;`, per-block shared-memory histograms (`atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);`), a select kernel that walks bins from the top, and a gather. Tie rule in the source: "the ties fill the last rank slots in column order, the lowest columns win." This is an MSD radix select (§14.9's MSD idea, applied to selection), not a sort.
- `DeviceTopK` is used for a single row when CCCL is new enough; the guard's comment: "DeviceTopK has a race condition before CCCL 3.4.3." with a link to NVIDIA/cccl PR 10627 (llama.cpp's statement). Thresholds differ for HIP: "not measured on HIP/MUSA, keep the old split". Without CUB (HIP) the long-row path is `top_k_radix_cuda`.
- Metal, `ggml/src/ggml-metal/kernels/argsort.metal` (2026-11-08): `kernel_argsort_f32_i32` is a threadgroup bitonic sort "following the CUDA kernels as reference"; `kernel_argsort_merge_f32_i32` merges sorted runs across threadgroups with a "binary-search partition (i, j) such that i + j = k", which is Chapter 13's co-rank, preferring the left run on equal values. The top-k key: `uint y = as_type<uint>(x); if ((y & 0x80000000u) != 0u) { y ^= 0xFFFFFFFFu; // negative floats: flip all bits } else { y |= 0x80000000u; // positive floats: set the sign bit }`. `kernel_top_k_f32_i32` (ch09 D4) ends with "deterministic compaction: rank the kept elements by column order, so the result does not depend on the thread scheduling ... the lowest tied indices are the ones kept".
- Our reading: neither transform merges -0.0 with +0.0, so +0.0 ranks above -0.0 in llama.cpp's top-k, unlike CUB (A3).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/top-k.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/argsort.metal

### D3. llama.cpp samplers: top-k is a partial sort; top-p sorts adaptively on the host and fully on the backend

**Verdict: CONFIRMED** as source read. `src/llama-sampler.cpp` (2026-11-12).

- Host top-k (`llama_sampler_top_k_impl`): `if (!cur_p->sorted) { llama_token_data_array_partial_sort_inplace(cur_p, k); }`. For k up to 128 that is `std::partial_sort`; above it, `llama_token_data_array_partial_sort` buckets logits into 128 value buckets over [-10, 10], `std::sort`s the full buckets above the cut and `std::partial_sort`s the cut bucket. A one-level bucket partition by value, then sorting only what is needed.
- Host top-p (`llama_sampler_top_p_apply`): softmax, then "if not sorted, try adaptive top-k sorting": partial sort of the top 256 when there are more than 1024 candidates, a running cumulative sum, and "we exceeded the current top-k heuristic -> increase k and continue" with a full partial sort. So the host sorts the whole vocabulary only when the top 256 do not reach p.
- Backend top-p (`llama_sampler_top_p_backend_apply`, experimental, ch01 D2): `ggml_argsort(ctx, logits, GGML_SORT_ORDER_DESC)`, gather, `ggml_soft_max`, `ggml_cumsum`, then a step mask "Make top-p inclusive (i.e. return all values such that cum_sum/cdf >= p)". Sort plus scan, as the lesson describes. Backend top-k calls `ggml_top_k` (D2).
- Stability (ours): `std::partial_sort` and `std::sort` are not stable, so the host's order among equal logits is unspecified.

- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp (lines 136 to 215, 322 to 340, 1493, 1556 to 1700)

### D4. MLX: stable merge sort whose first stage is odd-even transposition; `topk` is a full sort on Metal today

**Verdict: CONFIRMED.**

- Docs: `mlx.core.sort`: "The sort is stable, meaning equal elements preserve their relative order. NaN values are placed at the end." Same text for `mlx.core.argsort` ("The uint32 array containing indices that sort the input."). No `descending` parameter. `mlx.core.topk`: "The elements will not necessarily be in sorted order." `argpartition`: "The ordering of the elements within a partition in given by the indices is undefined."
- `mlx/backend/metal/kernels/sort.h` (2026-11-05): "Based on GPU merge sort algorithm at https://github.com/NVIDIA/cccl/tree/main/cub/cub". Thread stage `ThreadSort`: `for (short i = 0; i < N_PER_THREAD; ++i) { for (short j = i & 1; j < N_PER_THREAD - 1; j += 2) { if (op(vals[j + 1], vals[j])) { thread_swap(...); } } }`, which is §14.2's odd-even transposition sort on a thread's registers. Block stage `BlockMergeSort` with `merge_partition`, a binary search on the diagonal (Chapter 13's co-rank), and `merge_step` taking from the right run only when strictly smaller (stable). NaN ordering in `LessThan`: `if (an | bn) { return (!an) & bn; }`.
- `mlx/backend/metal/sort.cpp` (2026-10-07): `single_block_sort` and `multi_block_sort` (ping-pong buffers, `block_partitions`). `ArgPartition::eval_gpu`: "// We direct arg partition to sort for now" then `gpu_merge_sort(...)`; same for `Partition`. `mlx/ops.cpp` `topk`: `array a_partitioned = partition(a, -k, axis_, s);` then a slice. So on Metal, MLX's top-k costs a full sort today.
- `mlx/backend/cuda/sort.cu` has its own `BlockMergeSort` and `KernelMultiBlockMergeSort`; no CUB call found.
- mlx-lm `mlx_lm/sample_utils.py` (2026-09-24): `apply_top_k` uses `mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]` to find what to mask; `apply_top_p` sorts values ascending (`mx.sort(logprobs, axis=-1)`), takes `mx.cumsum(sorted_probs, axis=-1, reverse=True, inclusive=False)` (mass above each rank), finds a threshold value and masks `logprobs < threshold`. It never needs indices, and every token tied with the threshold is kept.

- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sort.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argsort.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.topk.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argpartition.html
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sort.h (lines 8 to 160)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/sort.cpp (lines 272 to 366)
- https://github.com/ml-explore/mlx/blob/main/mlx/ops.cpp (`topk`)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/cuda/sort.cu
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py (lines 138 to 160, 213 to 230)

### D5. PyTorch on CUDA (and ROCm): `torch.sort`, `torch.topk`, and the book's cascaded stable sort in production

**Verdict: CONFIRMED.**

- Docs 2.15: `torch.sort(input, dim=-1, descending=False, *, stable=False, out=None)`: "If stable is True then the sorting routine becomes stable, preserving the order of equivalent elements." `torch.argsort`: "If False, the relative order of values which compare equal is not guaranteed. True is slower." `torch.topk`: "When using torch.topk, the indices of tied elements are not guaranteed to be stable and may vary across different invocations."
- `aten/src/ATen/native/cuda/Sort.cpp` and `Sort.cu`: slices of at most 32 use `bitonicSortKVInPlace` ("For very small unstable sorts (n <= 32)"), at most 128 `warpMergeSortKVInPlace`, at most 4096 `radixSortKVInPlace`; larger ones `launch_stable_sort_kernel` ("this algorithm is always stable").
- `SortStable.cu` (2026-10-19), for a (rows, vocabulary) tensor: one segment, or segments of a million or more, call `at::cuda::cub::radix_sort_pairs` per segment; fewer than 128 segments use `segmented_sort_pairs_by_full_sort`, whose comment walks an example: "First we sort by values, globally ... Then we stable sort by segment id". That is the book's cascaded sort with a stable second pass (§14.1, p. 330). 128 segments or more call `cub::DeviceSegmentedRadixSort::SortPairs(Descending)` (`aten/src/ATen/cuda/cub.cuh`). On ROCm, `#define NO_ROCM(x)` drops the `at_cuda_detail` prefix and the hipify pass turns `cub` into `hipcub` (ch01 D4).
- Float key, `SortingRadixSelect.cuh`: `RadixType x = __float_as_int(v); RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000; return (v == v) ? (x ^ mask) : 0xffffffff;`. NaN maps above +inf; -0.0 and +0.0 stay distinct (our reading).
- `TensorTopK.cu` (2026-10-15): single-block path `radixSelect` with `constexpr int RADIX_BITS = 2;` then "we perform an exclusive prefix sum of `hasTopK`" to place results (Chapter 12's compaction on Chapter 11's scan); multi-block path `mbtopk` with `constexpr int RADIX_BITS = 8;` and a digit loop `for (int current_bit = sizeof(T) * 8 - RADIX_BITS; current_bit >= 0; current_bit -= RADIX_BITS)`, most significant digit first.

- https://docs.pytorch.org/docs/2.15/generated/torch.sort.html
- https://docs.pytorch.org/docs/2.15/generated/torch.argsort.html
- https://docs.pytorch.org/docs/2.15/generated/torch.topk.html
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Sort.cpp
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Sort.cu (lines 42 to 280)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SortStable.cu (lines 34 to 58, 155 to 312)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/cub.cuh (lines 31 to 37, 181 to 186)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SortingRadixSelect.cuh (lines 13 to 39)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/TensorTopK.cu

### D6. PyTorch on MPS: the chapter's §14.6 to §14.7 design, written in Metal

**Verdict: CONFIRMED.** `aten/src/ATen/native/mps/kernels/SortRadix.h` (2026-06-12) and `operations/Sort.mm` (2026-11-11).

- Header comment: "LSD radix sort kernels for MPS (Path 3 in Sort.mm). One full pass over an RBITS-bit digit = three kernels: radix_count: per-block digit histograms (rows x digits x n_blocks). radix_scan: exclusive scan over the histogram to get global offsets. ... radix_scatter: each block sorts its own elements by the current digit using a per-bit split, then scatters to global offsets." "Radix is naturally stable: the per-bit split preserves order within zeros and ones, and the global prefix-sum scatter preserves cross-block order."
- The table layout is the book's Fig. 14.9 and 14.12 order, all blocks' counts for digit 0, then digit 1: `histograms[row * RSIZE * n_blocks + d * n_blocks + block_idx] = count;`. The local sort is r one-bit splits (§14.7). The per-block histogram is `threadgroup atomic_uint local_hist[RSIZE];` (ch09 C1).
- Digit width, `Sort.mm`: "16/32-bit types use 8-bit radix (256 bins) - halves global memory traffic vs 4-bit radix while keeping the same total block-local work (RBITS sub-passes x n_passes = total bits either way). 8-bit types stay on 4-bit radix" / `const int radix_bits = (elem_size == 1) ? 4 : 8;`. That is §14.7's trade-off stated as a design rule.
- When radix is used at all, `should_use_radix`: never for 8-byte types; always for 2-byte floats with at least 2048 elements; never for 256 rows or more; otherwise when `radix_dispatches <= 2 * merge_dispatches + 2` with `radix_dispatches = n_radix_passes * 3`. Otherwise a merge sort (`SortMerge.h`). The comments cite PyTorch's own measurements; no numbers are copied here.
- Key, `to_radix_key`: "// val != val is NaN check; metal::isnan is buggy at large TGs on M2." (PyTorch's claim, not Apple's) / `if (val != val) return desc ? 0u : key_mask;` / `uint mask = (uint(0) - is_neg) | sign_bit; uint result = (bits ^ mask) & key_mask; if (desc) result = (~result) & key_mask;`. NaN goes last in both directions; descending is by inversion; -0.0 and +0.0 stay distinct (our reading); the NaN test is a float operation (C3).

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/SortRadix.h (lines 1 to 78, 147 to 210)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Sort.mm (lines 88 to 138, 278 to 360)

### D7. Top-p as "sort, then scan" in the code the reader knows, and serving stacks that avoid the sort

**Verdict: CONFIRMED.** Top-p over a sorted list is the common formulation in PyTorch-level code; top-k is done with a selection; a production engine now computes both without sorting by default.

- Hugging Face transformers, `src/transformers/generation/logits_process.py` (2026-11-18), `TopPLogitsWarper.__call__`: `sorted_logits, sorted_indices = torch.sort(scores, descending=False)` / `cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)` / `sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)` / then `scatter` back. Ascending, with the default (unstable) sort. `TopKLogitsWarper.__call__`: `indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]`, a selection.
- vLLM, `vllm/v1/sample/ops/topk_topp_sampler.py` (2026-11-19): the PyTorch fallback `apply_top_k_top_p_pytorch` has "# Avoid sorting vocab for top-k only case." and otherwise `logits_sort, logits_idx = logits.sort(dim=-1, descending=False)` then `torch.cumsum`. The default when Triton is present is `apply_top_k_top_p_triton` in `topk_topp_triton.py`: "Combined Top-K and Top-P Triton kernels. Based on the paper "Qrita: High-performance Top-k and Top-p Algorithm for GPUs using Pivot-based Truncation and Selection" By Park et al.", a pivot search with no sort. On CUDA it can also call `flashinfer.sampling.top_k_top_p_sampling_from_logits` (FlashInfer's algorithm not read).
- So for the lesson: "top-p over a sorted list" is the formulation in Hugging Face, mlx-lm (D4) and llama.cpp's backend path (D3); it is not the only one in production.

- https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py (lines 502 to 625)
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py (lines 198 to 290)
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_triton.py (lines 1 to 10)

---

## Group E: Our key transform, checked by arithmetic

### E1. The float-to-ordered-uint transform on negatives, subnormals, both zeros, infinities and NaN

**Verdict: CONFIRMED** (our arithmetic, run in Python with `struct` on the host today; the transform is ours as taught, and identical to CUB's documented one, A3).

Transform: `u = bits(f); key = u ^ (u & 0x80000000 ? 0xFFFFFFFF : 0x80000000)`. "Canon" adds the -0.0 step first (`u == 0x80000000 ? 0 : u`). "Desc" is `~canon`.

| Value | Bits | Key | Canon key | Desc key |
|---|---|---|---|---|
| -inf | `0xFF800000` | `0x007FFFFF` | `0x007FFFFF` | `0xFF800000` |
| -FLT_MAX | `0xFF7FFFFF` | `0x00800000` | `0x00800000` | `0xFF7FFFFF` |
| -1.0 | `0xBF800000` | `0x407FFFFF` | `0x407FFFFF` | `0xBF800000` |
| smallest negative subnormal | `0x80000001` | `0x7FFFFFFE` | `0x7FFFFFFE` | `0x80000001` |
| -0.0 | `0x80000000` | `0x7FFFFFFF` | `0x80000000` | `0x7FFFFFFF` |
| +0.0 | `0x00000000` | `0x80000000` | `0x80000000` | `0x7FFFFFFF` |
| smallest positive subnormal | `0x00000001` | `0x80000001` | `0x80000001` | `0x7FFFFFFE` |
| 1.0 | `0x3F800000` | `0xBF800000` | `0xBF800000` | `0x407FFFFF` |
| FLT_MAX | `0x7F7FFFFF` | `0xFF7FFFFF` | `0xFF7FFFFF` | `0x00800000` |
| +inf | `0x7F800000` | `0xFF800000` | `0xFF800000` | `0x007FFFFF` |

- The plain keys are strictly increasing down the table (checked), so an ascending unsigned sort of keys is an ascending numeric sort, negatives included.
- Without the canon step, -0.0 sorts strictly before +0.0; with it they are equal keys and a stable sort keeps their input order, matching `std::stable_sort` with `<` or `>`.
- NaN: +qNaN (`0x7FC00000`) maps to `0xFFC00000`, above +inf; -qNaN (`0xFFC00000`) maps to `0x003FFFFF`, below -inf. So NaNs split by sign bit to both ends, which is CUB's documented "sorted according to their bit representations". MLX and PyTorch force NaN to the end instead (D4, D5, D6). The lesson should pick one rule and test it.
- Descending by inversion keeps equal keys in input order under a stable ascending sort, so the M5 output lists tied probabilities by increasing token index. Reversing an ascending result would list them by decreasing index.
- For M5's input (softmax probabilities, all at least +0.0) the sign branch never fires and the raw bits already order correctly; the tests should still include negatives, both zeros and infinities because the same kernel sorts logits.

---

## Summary of corrections

Every PARTLY or COULD NOT VERIFY item, and every failed or unfound sub-claim:

- A1 (update, not an error): OneSweep is now inside CUB: the default for keys of 4 bytes or more from compute capability 6.0 and for all keys from 9.0, with 8-bit digits. The paper spells it "Onesweep"; its speedup "Compared to CUB" was over CUB's older multi-pass path.
- A5: `cub::DeviceSegmentedSort::SortPairs(Descending)` is documented as not stable; the stable call is `StableSortPairsDescending`. `DeviceSegmentedRadixSort`'s stability is stated only by reference to `DeviceRadixSort`.
- A7: `cub::DeviceTopK` today accepts only the unordered, non-deterministic, unspecified-tie configuration; `DeviceBatchedTopK` needs CUDA 13.5 and caps a segment at 2^21 items. llama.cpp reports a `DeviceTopK` race before CCCL 3.4.3.
- A10: "two cache lines" holds in lines; in NVIDIA's 32-byte sectors a 1-bit scatter costs 4 to 7 sectors against 4, and an 8-bit scatter approaches one sector per key.
- B2: rocPRIM does not support a custom `begin_bit`/`end_bit` for float keys; CUB does (applied after the transform).
- B3: hipCUB's `BlockRadixSort` ignores `RADIX_BITS` (and the scan-algorithm parameters).
- B4: hipCUB at ROCm 10.2 has no top-k; rocPRIM has `topk`, `segmented_topk`, `nth_element`, `partial_sort`, with ordered and deterministic segmented top-k "not supported yet".
- B6: HIP's `__float_as_uint` / `__uint_as_float` exist in the header but not in AMD's published math API reference.
- B7 (could not verify): HIP documents no rule for plain-access data races.
- C1: Apple documents MPSGraph `sort`, `argSort` (macOS 13) and `topK` (macOS 14), and `MPSMatrixFindTopK` limited to k of 16 or less; none states stability or tie order; MSL has no sort and Apple's sample library has no sorting sample.
- C2: Accelerate's `vDSP_vsorti` is documented as not stable; do not use it as the stable CPU reference.
- C3: both candidate Metal defaults (fast and relaxed) allow "no signed zeros", and fast math makes INF and NaN undefined, so the -0.0 fix and NaN handling must be done on the `uint` after `as_type`.
- C4: Metal documents no forward progress between threadgroups, so a OneSweep-style look-back has no documented basis there; use count, scan and scatter dispatches.
- D1 (our reading): llama.cpp's HIP build has no CUB path, so a vocabulary-sized `argsort` is reported unsupported on HIP; on CUDA, `ggml_argsort` is stable only on some paths.
- D2, D5, D6 (our reading): llama.cpp's top-k keys and PyTorch's CUDA and MPS keys do not merge -0.0 with +0.0, unlike CUB and rocPRIM.
- D4: MLX's `topk`, `partition` and `argpartition` run a full merge sort on Metal today ("for now").
- D5: PyTorch's default `torch.sort` is not stable (`stable=False`), small slices use an unstable bitonic sort, and `torch.topk` documents unstable tie indices.
- D7: production top-p is not always sort-based; vLLM's default Triton path is a pivot search.

Path and status facts worth knowing even where the verdict is CONFIRMED:

- A2: the current `DeviceRadixSort` overloads are marked "First appears in CUDA Toolkit 12.3"; the two-phase temporary-storage call and the no-in-place rule are unchanged.
- A4: CUB's digit widths are 8 (OneSweep), 7 (multi-pass), 6 (single tile) for multi-byte keys; rocPRIM uses 8 in 374 of 384 tuned OneSweep configs; PyTorch MPS uses 8 for 16- and 32-bit keys.
- A6/B5: Thrust and rocThrust run a radix sort (stable) for arithmetic keys with `less`/`greater` and a merge sort otherwise; `thrust::greater` is a deprecated alias of `cuda::std::greater` on CCCL `main`.
- D4: MLX's per-thread sort is §14.2's odd-even transposition sort, and its block merge uses Chapter 13's co-rank search.
- D5: PyTorch's few-row segmented sort is §14.1's cascaded stable sort (sort by value, then stable sort by row id).
- D6: PyTorch's MPS radix sort is §14.6 to §14.7 as designed in the book: per-block digit counts in a digit-major table, an exclusive scan, a local split sort and a scatter.

Not checked: the book's statements about sample sort across multiple GPUs (p. 346) and the cited algorithm papers other than OneSweep (Cole, Batcher, Satish et al., Blelloch et al.); any behaviour on real hardware.

## Where the backends really differ

One line per cell, for what this chapter's lessons and labs use. "Same" means the same source text compiles through `pmpp/vendors/hip.h`, according to the documentation; nothing here was run on CUDA or AMD hardware.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Library device radix sort of (key, index) pairs, descending | `cub::DeviceRadixSort::SortPairsDescending` | `hipcub::DeviceRadixSort::SortPairsDescending` (to `rocprim::radix_sort_pairs_desc`) | None at kernel level; MPSGraph `argSort(_:axis:descending:name:)`, macOS 13 |
| Stability documented | Yes, "DeviceRadixSort is stable" | Yes (rocPRIM "is stable") | Not stated for any MPSGraph sort |
| Stable row-wise (segmented) descending sort | `DeviceSegmentedSort::StableSortPairsDescending`; `DeviceSegmentedRadixSort::SortPairsDescending` | Same names in both hipCUB copies | MPSGraph sort along an axis; stability unstated |
| Float keys in the library | Documented transform; -0.0 equals +0.0; NaN by bits; bit range after transform | Same transform and -0.0 fix; no custom bit range for floats | No library sort over buffers; write the transform yourself |
| Reinterpret float bits | `__float_as_uint`, `__uint_as_float`, `cuda::std::bit_cast` | Same intrinsics in the header; not in the math API docs | `as_type<uint>(f)`, bits "returned directly without modification" |
| Float rules the transform must survive | Reinterpretation unaffected | Same | Fast and relaxed: no signed zeros; fast: INF and NaN undefined; denormals may flush in arithmetic. Keep it integer-only |
| Block-local radix sort | `cub::BlockRadixSort<Key, Threads, ItemsPerThread, Value, RadixBits = 4>` | `hipcub::BlockRadixSort`; `RADIX_BITS` ignored | None; build from `simd_prefix_exclusive_sum`, `simd_ballot` and 32 KB of threadgroup memory |
| Library digit width | 8 bits (OneSweep), 7 (multi-pass), 6 (single tile) | 8 bits in 374 of 384 OneSweep configs | No library; PyTorch MPS chooses 8 |
| Single-pass (look-back) digit pass | CUB OneSweep, default from compute capability 6.0 for 4-byte keys | rocPRIM OneSweep | No documented inter-threadgroup progress; separate count, scan, scatter dispatches |
| Scan of the bucket table between phases | Separate kernel, or `grid.sync()` under a cooperative launch | Separate kernel, or `grid.sync()` | Separate dispatch only |
| Top-k without a full sort | `cub::DeviceTopK` (unordered, non-deterministic only, so far); `DeviceBatchedTopK` from CUDA 13.5 | `rocprim::topk`, `segmented_topk`; no hipCUB wrapper | MPSGraph `topK` (macOS 14); `MPSMatrixFindTopK`, k of 16 or less |
| High-level sort | `thrust::sort`, `thrust::stable_sort_by_key` (CUB radix for arithmetic keys) | rocThrust, same dispatch to rocPRIM | None |
| Atomic for the odd-even flag | `atomicExch`, or `cuda::atomic_ref::store` relaxed | `atomicExch` | `atomic_store_explicit` on `device atomic_uint*` |
| Rule for a racy plain write | Undefined behaviour (guide section 5.8) | Not stated in HIP docs | Not defined for plain accesses |
| Packed 64-bit (key, index) key | `unsigned long long` | Same | `ulong`, Metal 2.2 and later |
| Stable CPU reference | `std::stable_sort` | Same | Same; Accelerate's `vDSP_vsorti` is documented as not stable |
