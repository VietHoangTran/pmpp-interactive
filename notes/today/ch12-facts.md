# Chapter 12 fact check: filtering, warp voting, privatised and stable compaction (CUDA, HIP, Metal, real backends)

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md` and `notes/today/infra-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY, and judges the claim in the item title (usually the book's statement, quoted with its page from `notes/ch12.md`, or the assumption a course author would bring). **Quote** is copied from the source with line breaks removed.
- "infra A3", "ch09 A9", "ch10 A12" point at items in `infra-facts.md`, `ch09-facts.md`, `ch10-facts.md` and so on. Anything settled there is cited, not re-checked. `ch11-facts.md` did not exist when this file was written (chapter 11 is being built at the same time); A10, B6 and C4 below cover only what chapter 12's stable and in-place filters need from the single-pass scan.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide "v13.5" (pages "Last updated on Oct 31, 2026"); CUDA Math API 13.5 (Nov 02, 2026); PTX ISA 9.5; CCCL docs at `nvidia.github.io/cccl/unstable/` and CCCL source on `main`; HIP docs for ROCm 10.2.0 ("HIP 7.17.0"), read as `.rst` sources and headers at tag `therock-10.2` of `ROCm/rocm-systems`, plus `ROCm/rocm-libraries` and `ROCm/libhipcxx` at the same tag; upstream LLVM `main`; Metal Shading Language Specification "Version 4.1" dated 2026-06-04 ("MSL spec"); local Metal toolchain `Apple metal version 32023.921` (it defines `__METAL_VERSION__ 410`) and macOS SDK 27.0; llama.cpp `master`, PyTorch `main`, MLX docs 0.32.4, mlx-lm `main`, PyTorch docs 2.15.
- Not checked: anything by running code on a GPU. One MSL kernel was compiled (not run) with the local toolchain to confirm that the names in C1 and C2 exist with the types given; that is recorded where it matters. Nothing was compiled for CUDA or ROCm.
- Where this file says what a piece of project code does, that is our reading of the source, labelled as such.

Totals: 14 CONFIRMED, 1 WRONG, 9 PARTLY, 3 COULD NOT VERIFY (27 items: A1 to A10, B1 to B6, C1 to C5, D1 to D6). The WRONG, PARTLY and COULD NOT VERIFY items are collected under "Summary of corrections".

---

## Already settled elsewhere (cite, do not re-check)

| Chapter 12 needs | Settled in |
|---|---|
| Warp width 32 on NVIDIA; 64 on gfx9 and 32 on gfx10 and later; Metal width undocumented, read `[[threads_per_simdgroup]]` | ch05 A5, ch05 B5, ch04 C2, infra C5 |
| `__all_sync`, `__any_sync`, `__ballot_sync`, `__activemask` signatures (the vote functions named on p. 295, which the book prints without parameters) | infra A3, ch10 A4; HIP infra B3, ch04 B12 |
| `__shfl_sync` signature, out-of-range rule, mask constraints | infra A3; HIP infra B3; Metal infra C2 |
| HIP masks are 64-bit on every target and a 32-bit literal fails a `static_assert`; `__ballot_sync`/`__activemask` return `unsigned long long`; `1 << lane` must become `1ull << lane` | infra B3, ch05 B5, ch10 B1 |
| `cuda::atomic_ref`, `thread_scope_block` / `thread_scope_device`, `fetch_add`, `memory_order_relaxed`, defaults (`seq_cst`, system scope) | infra A2, ch09 A1, ch09 A2 |
| `<cuda/atomic>` on HIP through libhipcxx (ROCm 10.1 and later); no `_block` legacy atomics on HIP | infra B2, infra B1, ch09 B1 |
| MSL atomics: `device atomic_uint*`, `threadgroup atomic_uint`, `atomic_fetch_add_explicit(..., memory_order_relaxed)`; no `atomic_ref`; no scope on read-modify-write | infra C1, ch09 C1 |
| Atomics on one address serialise in no fixed order (why Fig. 12.2 is unstable) | ch09 A3 (NVIDIA), ch09 B1 (AMD) |
| Global atomics execute at L2; shared-memory atomics are faster; NVIDIA's "warp level, then block level, then privatisation" recipe | ch09 A4, A5, A6; AMD ch09 B2 |
| Cooperative groups headers and namespace, HIP header path `<hip/hip_cooperative_groups.h>`, tiles no wider than a wavefront on HIP; no cooperative groups in Metal | infra A4, infra B4, ch04 B11, ch04 C3 |
| Coalescing rules behind §12.3's and §12.6's "adjacent stores coalesce" | ch06 A2 (NVIDIA), ch06 B1 and ch09 B3 (AMD), ch06 C1 and ch09 C4 (Apple: not documented) |
| Metal threadgroup memory 32 KB, dynamic threadgroup memory (budget for §12.4's private list and §12.6's staging) | ch05 C1, ch05 C8 |
| Metal SIMD-group functions and their GPU families (permute Apple6, reductions and prefix sums Apple7, shift-and-fill Apple8) | infra C2 |
| Grid-wide barrier only under cooperative launch (CUDA, HIP); none on Metal | ch04 A7, ch04 B11, ch04 C3, ch10 A8 |
| Thrust and CUB live in CCCL; the book-style URL `nvidia.github.io/cccl/cub/index.html` returns 404 (re-checked today for p. 302's reference [4]: still 404; `/cccl/unstable/cub/index.html` returns 200) | ch10 A12, infra A11 |
| GPU race detectors: `compute-sanitizer --tool racecheck` sees shared memory only, so an unsynchronised global output counter (a planted defect in Fig. 12.2) is invisible to it, while a non-atomic private counter in Fig. 12.6 is in its reach; none documented for AMD or Apple; ThreadSanitizer as used by the course's `race` backend | ch09 A10, B6, C5, E1 |
| A vocabulary row does not fit in any documented per-block on-chip memory (sizes for the M5 lab) | ch09 D7 |
| Metal default math mode documented three ways (relevant to a float threshold test only at NaN/INF) | ch05 C7, ch10 C3 |

---

## Group A: CUDA, and the book's dated statements

### A1. `__activemask()` called inside `if(cond(val))` returns "exactly the lanes that passed" (Fig. 12.3, p. 292)

**Verdict: PARTLY.** The type and bit layout are exactly as the book says (32-bit, bit N for lane N). NVIDIA's guide now carries an explicit warning against the use the book makes of it: the mask is a snapshot of the lanes that happen to be converged, not the set of lanes that took the branch.

Quotes (guide section 5.4.6.1 "Warp Active Mask"):

- `unsigned __activemask();` "The function returns a 32-bit integer mask representing all currently active threads in the calling warp. The Nth bit is set if the Nth lane in the warp is active when __activemask() is called. Inactive threads are represented by 0 bits in the returned mask. Threads that have exited the program are always marked as inactive."
- "Warning: __activemask() cannot be used to determine which warp lanes execute a given branch. This function is intended for opportunistic warp-level programming and only provides an instantaneous snapshot of the active threads within a warp." The guide's own counter-example is `if (pred) { at_least_one = __activemask() > 0; }` marked "Invalid: the value of 'at_least_one' is non-deterministic and could vary between executions."
- "Note that threads convergent at an __activemask() call are not guaranteed to remain convergent at subsequent instructions unless those instructions are warp synchronizing intrinsics (__sync)."
- Section 3.2.2.1.1: "In GPUs of compute capability 7.0 and later, independent thread scheduling allows full concurrency between threads, regardless of warp." Every compute capability in the current tables is 7.5 or later (ch04 A10).

What this means for Fig. 12.3 (our reading of the rules above and of A4):

- The kernel stays correct. Whatever subset of passing lanes is converged at the call, each lane in it reads the same mask, elects the same leader, reserves `__popc(mask)` slots, and the following `__shfl_sync(mask, ...)` is a `_sync` intrinsic that re-converges exactly those lanes. If the passing lanes split into several snapshots, each snapshot does its own reservation (the guide allows concurrent calls with disjoint masks, A4).
- What is not guaranteed is the book's count of "one atomic per warp" (p. 292). A lab that counts atomics per warp may see more than one, and a lesson should say the hardware is allowed to.
- The documented way to learn exactly which lanes passed is a vote in uniform control flow, before the branch: `unsigned m = __ballot_sync(FULL_MASK, keep);` (signature in infra A3; "returns an integer whose Nth bit is set if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active"). With every lane executing the vote, `m` is the exact pass set, the leader can be `__ffs(m) - 1`, and every lane computes its rank from `m`. This is our recommendation, built only from the quoted rules; it is also what llama.cpp and PyTorch do (D1, D4).

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.6.1, 5.4.6.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (section 3.2.2.1.1)

### A2. `__ffs()` "returns index + 1, 0 if none" and `__popc()` counts set bits (p. 293)

**Verdict: CONFIRMED.** Both are documented in the CUDA Math API "Integer Intrinsics" group, each with a 64-bit sibling that the HIP port needs (B1).

| Declaration (exact) | Documented behaviour |
|---|---|
| `__device__ int __ffs(int x)` | "Find the position of the first (least significant) bit set to 1 in x, where the least significant bit position is 1." "Returns a value between 0 and 32 inclusive representing the position of the first bit set." "__ffs(0) returns 0." |
| `__device__ int __ffsll(long long int x)` | Same, "between 0 and 64 inclusive"; "__ffsll(0) returns 0." |
| `__device__ int __popc(unsigned int x)` | "Count the number of bits that are set to 1 in x." "Returns a value between 0 and 32 inclusive". |
| `__device__ int __popcll(unsigned long long int x)` | Same, "between 0 and 64 inclusive". |

Details for the lesson:

- `__ffs` takes a signed `int`; the book passes the `unsigned int` mask, which converts implicitly. Harmless for the bit pattern.
- Also in the same group and useful for the visualiser: `__clz`, `__brev`, and `__fns(unsigned mask, unsigned base, int offset)` ("find the n-th (given by offset) set bit in mask from the base bit. If not found, return 0xFFFFFFFF"), which maps a rank back to a lane.

- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html

### A3. The lanes-below mask must be built by hand as `(1 << laneIdx()) - 1` (Fig. 12.3, lines 22 to 24)

**Verdict: PARTLY.** The book's expression is correct for a 32-lane warp. CUDA now documents two ready-made forms, and NVIDIA's own filtering blog uses a third name that no current NVIDIA reference page lists.

- PTX special register (PTX ISA 9.5, section 10.20): `%lanemask_lt`: "32-bit mask with bits set in positions less than the thread's lane number in the warp." Declared `.sreg .u32 %lanemask_lt;`. Siblings `%lanemask_le`, `%lanemask_ge`, `%lanemask_gt`, `%lanemask_eq`. It includes inactive lanes, so it must be ANDed with the vote mask, as the book does.
- libcu++ (CCCL `unstable` docs): `cuda::device::lane_mask`, "Defined in <cuda/warp> header", "a fancy wrapper around a single 32-bit unsigned integer value", with `static lane_mask all_less()`, `all_less_equal()`, `all_greater()`, `this_lane()`, `all_active()` ("equivalent to the result __activemask()"); "They are implemented using the PTX special registers." The header `libcudacxx/include/cuda/__warp/lane_mask.h` is absent at tag `v3.0.0` and present from `v3.1.0` on, so it needs CCCL 3.1 or later. Which CUDA Toolkit first bundles CCCL 3.1 was not checked. libhipcxx does not provide it (infra B2 lists `cuda::device::lane_mask` as unsupported).
- NVIDIA's 2014 filtering blog (updated November 2017, A7) writes `__popc(active & __lanemask_lt())` and describes `__lanemask_lt()` as returning "the mask of all lanes (including inactive ones) with ID less than the current lane". A search of the current programming guide's language-extensions and device-API pages finds no `__lanemask_lt`. Treat it as undocumented on CUDA; HIP does declare a function of that name, with different semantics (B2).
- C++ note: `1 << laneIdx()` with `laneIdx() == 31` shifts a signed `int` into the sign bit. cppreference, for the rules until C++20: "if a * 2b is representable in the unsigned version of the return type, then that value, converted to signed, is the value of a << b (this makes it legal to create INT_MIN as 1 << 31)"; since C++20 the result is defined modulo 2^N. In every version: "If the value of rhs is negative or is not less than the number of bits in lhs, the behavior is undefined." So lane 31 is fine, writing `1u << lane` avoids the question, and on a 64-wide wavefront any 32-bit shift is undefined for lanes 32 to 63 (ch05 B5, B1 here).

- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (section 10.20)
- https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/warp/lane_mask.html
- https://github.com/NVIDIA/cccl/blob/v3.1.0/libcudacxx/include/cuda/__warp/lane_mask.h
- https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
- https://en.cppreference.com/w/cpp/language/operator_arithmetic (section "Bitwise shift operators")

### A4. `__shfl_sync(activeThreads, j, leader)`: a shuffle whose mask is the active mask, not `0xffffffff` (Fig. 12.3, line 20)

**Verdict: CONFIRMED.** The book's use satisfies every documented constraint, and the guide explicitly allows several disjoint masks at once in divergent code, which is what happens if the passing lanes are split across snapshots (A1).

Quotes (guide section 5.4.6.6 "Warp __sync Intrinsic Constraints"):

- "Each calling thread must have its corresponding bit set in the mask." "Each non-calling thread must have its corresponding bit set to zero in the mask. Exited threads are ignored." "All non-exited threads specified in the mask must execute the intrinsic with the same mask value."
- "Warp threads may call the intrinsic concurrently with different mask values, provided the masks are disjoint. Such condition is valid even in divergent control flow."
- "The intrinsics achieve the best efficiency when all warp threads participate in the call, namely when the mask is set to 0xFFFFFFFF."
- From section 5.4.6.5: "Threads may only read data from another thread that is actively participating" (infra A3). The leader is in the mask by construction, so the read is legal.

So a planted defect of `__shfl_sync(0xffffffff, j, leader)` inside the branch violates the second rule (lanes that failed `cond` have their bit set but do not call), which the guide classes as "invalid, such as kernel hang, or undefined".

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.6.5, 5.4.6.6)

### A5. Cooperative groups: `coalesced_group`, `coalesced_threads()`, `.size()`, `.thread_rank()`, `.shfl(value, rank)` (Fig. 12.4, pp. 293 to 294)

**Verdict: PARTLY.** Every name in Fig. 12.4 exists and means what the book says. Two differences: `size()` is now listed as a legacy alias of `num_threads()`, and the listing omits the header and namespace (the notes' "Unclear" item).

- Header and namespace, as NVIDIA's own examples write them: `#include <cooperative_groups.h>` and `namespace cg = cooperative_groups;` (guide section 5.7.3). The 2017 blog writes `using namespace cooperative_groups;`.
- Class reference, section 5.7.3.1.5: "Constructing the group handle via coalesced_threads() is opportunistic. It returns the set of active threads at that point in time, and makes no guarantee about which threads are returned (as long as they are active) or that they will stay coalesced throughout execution (they will be brought back together for the execution of a collective but can diverge again afterwards)." The same caveat as A1.
- Members as listed: `void sync() const`, `unsigned long long num_threads() const` ("Total number of threads in the group"), `unsigned long long thread_rank() const` ("Rank of the calling thread within [0, num_threads)"), `T shfl(T var, unsigned int src_rank) const`, `shfl_up`, `shfl_down`, `int any(int) const`, `int all(int) const`, `unsigned int ballot(int) const`, `match_any`, `match_all`. Under "Legacy member functions (aliases)": `unsigned long long size() const: Total number of threads in the group (alias of num_threads())`.
- The `shfl` argument is a rank, not a lane, which the book points out (`activeThreads.shfl(j, 0)`). The 2017 blog says the same: "The 0 is the index of the leader thread, which works since only active threads are part of the group".
- Guide section 4.4.3.1: "For best performance it is recommended that you create a handle for the implicit group upfront (as early as possible, before any branching has occurred)". `coalesced_threads()` inside a branch is the exception the opportunistic pattern needs.

For lessons: write `num_threads()` and say the book's `size()` still compiles as an alias.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html (section 5.7.3.1.5)
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html (sections 4.4.2, 4.4.3)

### A6. Beyond the book: cooperative groups now ship the whole warp-aggregated append, and a stream-compaction example

**Verdict: CONFIRMED.** Not in the book. Current CUDA documents four things that subsume §12.3, and the guide's own example of a scan is a compaction.

- `cooperative_groups/scan.h` (section 5.7.3.5): `TyVal exclusive_scan(const TyGroup& group, TyVal&& val);` "Result for each thread is a reduction of data from threads with lower thread_rank than that thread in case of exclusive_scan." "group: Valid group types are coalesced_group and thread_block_tile." Default operator `cg::plus<TyVal>()`. So `cg::exclusive_scan(cg::coalesced_threads(), 1)` is the per-lane offset of Fig. 12.3 lines 22 to 24.
- `exclusive_scan_update(group, atomic, val)` (same section): "These variants will perform an update to the atomic according to op with value of the sum of input values of all threads in the group. Previous value of the atomic will be combined with the result of scan by each thread and returned." "atomic ... can be either of cuda::atomic or cuda::atomic_ref". "Scope of the atomic must include all the threads in the group". "Atomic update is performed with relaxed memory ordering." The pseudocode shows one selected thread doing the atomic and a `group.shfl(old, selected_thread)`. With `val = 1` this is Fig. 12.3 lines 7 to 25 in one call. The guide's example uses it to allocate shared-memory buffer space per tile.
- The guide's example titled "Example of stream compaction using exclusive_scan": `int my_idx = cg::exclusive_scan(g, my_count);` after each thread compacts its own chunk, then each thread copies its run to `output[my_idx + i]`, and returns `g.shfl(my_idx + my_count, g.num_threads() - 1)` as the count. That is §12.5 to §12.6 (coarsened, scan of per-thread counts) inside one group. (The example misspells "contiguous" as "contagious"; it also compacts its input in place, unordered within a thread.)
- `cg::invoke_one(group, fn)` / `invoke_one_broadcast(group, fn)` (section 4.4.6.3): "selects a single arbitrary thread from the calling group and uses that thread to call the supplied invocable"; "invoke_one_broadcast is the same as invoke_one except the result of the call is also broadcast to all threads in the group." "The thread selection mechanism is not guaranteed to be deterministic." That is the leader election plus broadcast of Fig. 12.4.
- `cg::binary_partition(tile, pred)` (section 5.7.3.3.3) returns a `coalesced_group` of the lanes that share `pred`, computed collectively over the whole tile ("The implementation may cause the calling thread to wait until all the members of the parent group have invoked the operation"). It gives the exact pass set from uniform control flow, the CG counterpart of A1's ballot advice.

The course's emulators do not provide `scan.h`, `invoke_one` or `binary_partition` (project fact, "Project notes" below); a lab that uses them is real-device only.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html (sections 5.7.3.3.3, 5.7.3.5)
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html (sections 4.4.6.2, 4.4.6.3)

### A7. "The compiler already implements the optimization of coalescing atomic operations", and "refrains ... if it can establish that only one thread in the warp is active" (p. 294)

**Verdict: PARTLY.** The first half is NVIDIA's statement, but only in a blog post last updated in 2017, with no conditions (already found in ch09 A9). The second half appears in no NVIDIA source found. The AMD compiler's equivalent is documented in source and does exactly Fig. 12.3 (B4).

- The blog is the book's own example. Title "CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics" (Oct 01, 2014; "This post has been updated (November 2017) for CUDA 9"). Its kernel is `filter_k`, "copy all elements of src satisfying the predicate into the destination array ... the order of elements in the dst array does not matter", and its `atomicAggInc` is Fig. 12.3 line for line: `unsigned int active = __activemask(); int leader = __ffs(active) - 1; int change = __popc(active); unsigned int rank = __popc(active & __lanemask_lt());` ... `warp_res = __shfl_sync(active, warp_res, leader); return warp_res + rank;`.
- On the compiler: "The NVCC compiler now performs warp aggregation for atomics automatically in many cases, so you can get higher performance with no extra effort. In fact, the code generated by the compiler is actually faster than the manually-written warp aggregation code." "the compiler does the optimization for post-Kepler GPUs starting with CUDA 7.5, and in CUDA 9, it also does it for Kepler GPUs." With CUDA 9 on Kepler, Pascal and Volta: "The performance of simple atomicAdd() is similar to that of warp-aggregated atomics."
- On stability, the same post: "Thrust implements a stable filter: it preserves the relative order of the input elements in the output. This is a result of using prefix sum to implement it, but it is more expensive as a result. If we don't need a stable filter, then a purely atomic approach is simpler and performs less work." That is the book's §12.2 versus §12.5 trade-off in NVIDIA's words.
- COULD NOT VERIFY: any current NVIDIA document (programming guide, best practices, Nsight Compute guides read for ch09) that restates the automatic aggregation, its conditions, or the "only one thread active" exception. The blog's figures are Kepler, Pascal and Volta measurements and stay out of lessons.

For the lab (ours): the prediction "where does contention bite" must allow that the plain-atomic kernel may already be aggregated on NVIDIA and is aggregated on AMD (B4); the honest measurement is the one on the reader's device. Reading the SASS or PTX of the lab kernel is the only way to see what the compiler did.

- https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
- ch09 A9 (PTX `red` versus `atom`)

### A8. "In practice, the Thrust and CUB libraries offer filter implementations" (p. 302)

**Verdict: PARTLY.** Both libraries do, and every entry read is stable. The book's CUB URL (reference [4], `https://nvidia.github.io/cccl/cub/index.html`) returns 404 today (ch10 A12). No unstable select was found in either library.

CUB `cub::DeviceSelect` (CCCL `unstable` docs; header `<cub/device/device_select.cuh>`, "or equivalently" `<cub/cub.cuh>`):

- "DeviceSelect provides device-wide, parallel operations for compacting selected items from sequences of data items residing within device-accessible memory. It is similar to DevicePartition, except that non-selected items are discarded, whereas DevicePartition retains them."
- `If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op, env)`: "Uses the select_op functor to selectively copy items from d_in into d_out. The total number of items selected is written to d_num_selected_out." "Copies of the selected items are compacted into d_out and maintain their original relative ordering." "The range [d_out, d_out + *d_num_selected_out) shall not overlap [d_in, d_in + num_items) nor d_num_selected_out in any way." Marked "Added in version 2.2.0: First appears in CUDA Toolkit 12.3." Environment-based overloads without explicit temporary storage: "Added in version 3.4.0: First appears in CUDA Toolkit 13.4."
- In-place overloads exist: `If(d_temp_storage, temp_storage_bytes, d_data, d_num_selected_out, num_items, select_op, env)`: "Uses the select_op functor to selectively compact items in d_data." "Copies of the selected items are compacted in d_data and maintain their original relative ordering." That is §12.7 as a library call (how it is made safe: A9).
- `Flagged` (keep flags as a separate array: §12.5's `keep` array, exercise 1), `FlaggedIf` (predicate applied to the flags), `Unique` ("Given an input sequence d_in having runs of consecutive equal-valued keys, only the first key from each run is selectively copied": §12.8's "removing duplicates from a sorted list"), `UniqueByKey`.
- The documented tuning knob is the book's coarsening: a `cub::SelectPolicy` with `.algorithm = cub::SelectAlgorithm::lookback`, `.threads_per_block = 128`, `.items_per_thread = cc > cuda::compute_capability{9, 0} ? 16 : 10` and a `lookback_delay`. That is an example of custom tuning in the docs, not the library default.

Thrust (`thrust/copy.h`, `thrust/remove.h`):

- `copy_if`: "copies elements from the range [first,last) to a range beginning at result, except that any element which causes pred to be false is not copied. copy_if is stable, meaning that the relative order of elements that are copied is unchanged." A stencil overload applies `pred` to a separate array.
- `remove_if`: "remove_if is stable, meaning that the relative order of elements that are not removed is unchanged." It is in place; the tail `[new_last, last)` holds unspecified values.

Related, for the sampler (not a filter): `cub::DeviceTopK` (added 3.3.0) documents three orthogonal requirements, `cuda::execution::determinism`, `tie_break`, `output_ordering`, with a "committed default" of `gpu_to_gpu`, `prefer_smaller_index`, `stable_sorted`. But: "Current support. This release only implements the fully opted-out configuration, which must be requested explicitly: cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::tie_break::unspecified, cuda::execution::output_ordering::unsorted)." "if multiple items tie at the K-th position, the subset of tied elements returned is not uniquely defined and may vary between runs." So today's CUB top-k is the unstable kind, and llama.cpp guards an older race in it (D1).

- https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceSelect.html
- https://nvidia.github.io/cccl/unstable/thrust/api/group__stream__compaction_1ga291bd49963dc7d3d0529fb8ed05de66e.html (`copy_if`)
- https://nvidia.github.io/cccl/unstable/thrust/api/group__stream__compaction_1gaf01d45b30fecba794afae065d625f94f.html (`remove_if`)
- https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceTopK.html

### A9. CUB's select kernel is §12.6 and §12.7 in production: staging through shared memory, and acquire/release for in place

**Verdict: CONFIRMED** (from CCCL source; `cub/cub/agent/agent_select_if.cuh` on `main`, last commit 2026-11-19).

- Shared-memory staging is chosen by selectivity, which bears directly on the lab's selectivity sweep: "Do a two-phase scatter if two-phase is enabled and the average number of selection_flags items per thread is greater than one" / `if (TWO_PHASE_SCATTER && (num_tile_selections > BLOCK_THREADS)) { ScatterSelectedTwoPhase<IsLastTile>(...); } else { ScatterSelectedDirect<IsLastTile>(...); }`. The two-phase path writes `temp_storage.raw_exchange.Alias()[local_scatter_offset] = items[ITEM];` and then copies out. So CUB stages in shared memory (Fig. 12.9) only when a tile keeps more items than it has threads, and scatters directly otherwise.
- The single pass is the decoupled look-back of chapter 11: `using ScanTileStateT = ScanTileState<OffsetT>;` and a static assertion that "the decoupled look-back needs a full warp to enter the block-scan prefix callback".
- In place, the book's §12.7 argument appears as a code comment, with the condition the book leaves implicit: "If this may be an *in-place* stream compaction, we need to ensure that all of a tile's items have been loaded before signalling a subsequent thread block's partial or inclusive state, hence we need a store release when updating a tile state. Similarly, we need to make sure that the load of previous tile states precede writing of the stream-compacted items and, hence, we need a load acquire when reading those tile states." The order is `acquire_release` when `SelectionOpt == SelectImpl::SelectPotentiallyInPlace` and loads do not go through shared memory, otherwise `relaxed`.
- So the book's claim that "the code written for the out-of-place case also works in place" (p. 300) holds for the book's own scan, whose flag write is a release and whose flag read is an acquire (Fig. 11.17, per `notes/ch11.md`), and on condition that every thread's input read is ordered before the block's release (a barrier between "all threads read" and the flag store). A look-back written with relaxed flags, which is CUB's choice for the out-of-place case, would not be safe in place. That condition is our reading of CUB's comment applied to the book's kernel.

- https://github.com/NVIDIA/cccl/blob/main/cub/cub/agent/agent_select_if.cuh (lines 205 to 226, 270 to 282, 683 to 771)

### A10. Spinning on an earlier block (the `gridExclusiveScan` of Fig. 12.8 and the in-place filter of §12.7): what NVIDIA guarantees

**Verdict: CONFIRMED** that NVIDIA now documents a forward-progress rule, and that under it the book's dynamic block index (chapter 11, p. 280) is what makes waiting on an earlier block legal. The reasoning from the rule to the kernel is ours.

- Programming model, section 1.2.2.1: "the CUDA programming model, with some exceptions, requires that there be no data dependencies between threads in different thread blocks." "Different thread blocks within the grid are scheduled among the available SMs and may be executed in any order. In short, the CUDA programming model requires that it be possible to execute thread blocks in any order, in parallel or in series."
- Execution model, section 5.9 (a section the book does not cite): "CUDA C++ aims to provide parallel forward progress [intro.progress.9]". Section 5.9.2 "Device threads": "Once a device thread makes progress: If it is part of a Cooperative Grid, all device threads in its grid shall eventually make progress. Otherwise, all device threads in its thread-block cluster shall eventually make progress. [Note: Threads in other thread-block clusters are not guaranteed to eventually make progress. - end note.] [Note: This implies that all device threads within its thread block shall eventually make progress. - end note.]"
- The section's own example of starvation: two kernels in two streams, one spinning on a flag the other sets: "Allowed outcome: eventually, no thread makes progress." "`second` may starve `first`."

Applied to the single-pass scan (ours): a block that obtained its index from the global counter has already made progress, so the rule says all its threads eventually make progress; a block waiting only on blocks with smaller dynamic indices therefore waits only on blocks that have started and will finish. With `blockIdx.x` as the index instead, block b may wait on block b - 1 that has not started, and the note says nothing guarantees it ever will. This matches the chapter 11 notes' "Fallback that needs none of this: the three-kernel scan" and is the reason the stable-filter lab should offer the three-kernel version (exercise 1) as the portable baseline.

Note the section number: the memory model the book's chapter 11 cites is section 5.8 (ch10 A1); the execution model is 5.9, at `05-appendices/cuda-cpp-execution-model.html`.

- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html (section 1.2.2.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-execution-model.html (sections 5.9.2, 5.9.3)

---

## Group B: HIP / ROCm

### B1. Does Fig. 12.3 (activemask, ffs, popc, shuffle, bit-mask rank) compile and work under HIP unchanged?

**Verdict: WRONG.** It does not compile, and the obvious one-line fix leaves a silent bug on 64-wide wavefronts. Each step is documented:

1. `unsigned int activeThreads = __activemask();` compiles but narrows: HIP declares `unsigned long long __activemask()` (infra B3). On gfx9 (64 lanes, ch05 B5) lanes 32 to 63 vanish from the mask.
2. `__shfl_sync(activeThreads, j, leader)` fails to compile: every HIP `_sync` function has `static_assert(... sizeof(MaskT) == 8, "The mask must be a 64-bit integer. Implicitly promoting a smaller integer is almost always an error.");` (infra B3).
3. `__popc` and `__ffs` are 32-bit on HIP as on CUDA. Header `amd_device_functions.h` at `therock-10.2`: `__device__ static inline unsigned int __popc(unsigned int input) { return __builtin_popcount(input); }`, `__device__ static inline unsigned int __popcll(unsigned long long int input)`, `__device__ static inline int __ffs(unsigned int input)`, `__device__ static inline int __ffsll(unsigned long long int input)` (plus `int` and `long long` overloads). Passing a 64-bit mask to `__popc` or `__ffs` converts it to 32 bits without a diagnostic required by the language. The 64-bit mask needs `__popcll` and `__ffsll`.
4. `(1 << laneIdx()) - 1` is undefined for lanes 32 and above with a 32-bit `int`; it must be `(1ull << lane) - 1` (ch05 B5).

So a port that only widens the shuffle's mask argument compiles and is wrong on Instinct (gfx9) hardware for any warp whose passing lanes include lane 32 or above: `numActive` and offsets ignore the upper half, and `__ffs` of a mask with only upper lanes set returns 0, giving `leader = -1` (our arithmetic from the declarations). It is right on RDNA (32 lanes). That makes it a good planted defect for a HIP-aware lab, and it is exactly the trap llama.cpp closes with a deleted template overload (D1).

Two documented differences that work in the book's favour on AMD:

- The header comment above `__activemask`: "Since threads in a wave do not make independent progress, __activemask() always returns the exact active mask, i.e, all active threads in the wave." So the A1 caveat does not arise on AMD; inside the branch the mask is the pass set.
- HIP docs: "Note that the __ballot and __activemask built-ins in HIP have a 64-bit return value (unlike the 32-bit value returned by the CUDA built-ins). Code ported from CUDA should be adapted to support the larger warp sizes that the HIP version requires."

Small doc and header mismatch: the HIP math API table prints `unsigned int __ffs(int x)` and `unsigned int __ffsll(long long int x)`; the header returns `int`. It also notes "The HIP-Clang implementation of __ffs() and __ffsll() contains code to add a constant +1 to produce the ffs result format."

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_device_functions.h (lines 47 to 77)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_warp_functions.h (lines 124 to 136)
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "Warp vote and ballot functions")
- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/math_api.html (integer intrinsics table)

### B2. HIP's own `__lanemask_lt()` and `__lane_id()`

**Verdict: PARTLY.** Both exist in the HIP headers at `therock-10.2`, neither appears in the HIP documentation sources at that tag (searched all 98 `.rst`/`.md` files under `projects/hip/docs`), and `__lanemask_lt()` does not mean what PTX's `%lanemask_lt` means.

```cpp
__device__ inline __hip_uint64_t __lanemask_lt() {
  __hip_uint32_t lane = __lane_id();
  __hip_int64_t ballot = __ballot64(1);
  __hip_uint64_t mask = ((__hip_uint64_t)1 << lane) - (__hip_uint64_t)1;
  return mask & ballot;
}
```

- It returns only the active lanes below the caller (it ANDs with a ballot of active lanes). PTX's `%lanemask_lt` includes inactive lanes (A3). For Fig. 12.3 the result after `& activeThreads` is the same either way; for any other use the two differ.
- `__lane_id()` reads the hardware lane through `__builtin_amdgcn_mbcnt_lo` / `_hi`, so on HIP the lane index does not have to be `threadIdx.x % warpSize`.
- Because they are undocumented, a lab should compute the mask from `__lane_id()` or `threadIdx.x % warpSize` with a 64-bit shift rather than rely on `__lanemask_lt()`. PyTorch does that on ROCm (D4).

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_device_functions.h (lines 660 to 680)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_warp_functions.h (lines 139 to 150)

### B3. Fig. 12.4's coalesced group on HIP

**Verdict: PARTLY.** `coalesced_threads()`, `num_threads()`, `size()`, `thread_rank()` and a rank-addressed `shfl()` all exist with CUDA's meaning. The header path differs (infra B4), the group's mask is 64-bit, and of A6's newer helpers only `exclusive_scan` exists.

- Docs ("Coalesced groups"): "Threads (64 threads on CDNA and 32 threads on RDNA) in a warp cannot execute different instructions simultaneously ... The active threads are referred to as coalesced, and coalesced group represents an active thread group within a warp." Warning on the same page: "AMD GPUs do not support independent thread scheduling. Some CUDA application can rely on this feature and the ported HIP version on AMD GPUs can deadlock, when they try to make use of independent thread scheduling."
- Public members as documented: "thread_rank(), size(), cg_type(), is_valid(), sync(), meta_group_rank(), meta_group_size(), shfl(), shfl_down(), shfl_up(), ballot(), any(), all(), match_any() and match_all()". The header also has `num_threads()`.
- Header `amd_hip_cooperative_groups.h`: the group is built from the hardware exec mask (`__builtin_amdgcn_read_exec()`), `coalesced_info.num_threads = __popcll(coalesced_info.member_mask);`, `thread_rank()` counts member bits below the lane, and `template <class T> T shfl(T var, int srcRank) const` maps the rank to a lane with `__fns64` or `__fns32` before `__shfl`. Its doc comment: "The type can be a 32-bit integer or single-precision floating point." So `activeThreads.shfl(j, 0)` with `unsigned int j` is within the documented types.
- Scans: `<hip/cooperative_groups/hip_scan.h>` declares `inclusive_scan` and `exclusive_scan` for "coalesced_group or a thread_block_tile". The docs add: "The value returned for the first active lane is platform-dependent and may change in future releases." ("AMD: the identity value is returned ... plus: always 0"). The docs also say "Participation of all the threads belonging to the group is expected, with each thread contributing the same per-thread value", which reads like a slip for "the same type"; we did not resolve it.
- Not present at the tag: `exclusive_scan_update`, `invoke_one`, `invoke_one_broadcast` (no match in `amd_hip_cooperative_groups.h` or `amd_hip_cooperative_groups_scan.h`). `binary_partition` is declared.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html (sections "Coalesced groups", "inclusive_scan", "exclusive_scan")
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_hip_cooperative_groups.h (lines 388 to 520, 663 to 670)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/cooperative_groups/hip_scan.h

### B4. The AMD compiler performs Fig. 12.3 on Fig. 12.2 by itself

**Verdict: CONFIRMED** (from compiler source, upstream LLVM `main`, `AMDGPUAtomicOptimizer.cpp` last commit 2026-11-17). This extends ch09 B5, which found the pass and its uniform-address condition; read today for what it does when the atomic's return value is used, which is the filter's case.

- Scope: "This pass optimizes atomic operations by using a single lane of a wavefront to perform the atomic operation, thus reducing contention on that memory location." Operations: add, sub, and, or, xor, signed and unsigned min and max, and float add, sub, min, max (32 and 64 bit). Condition: "If the pointer operand is divergent, then each lane is doing an atomic operation on a different address, and we cannot optimize that." Both global and LDS atomics are handled (`IsLDS`). A single output counter, global (Fig. 12.2) or in shared memory (Fig. 12.6), has a wavefront-uniform address, so it qualifies.
- What it emits, step by step, in the pass's own comments, which map one to one onto the book's four steps (p. 292):
  - Active set: "We need to know how many lanes are active within the wavefront, and we do this by doing a ballot of active lanes."
  - Rank: "We need to know how many lanes are active within the wavefront that are below us ... We do this by using the mbcnt intrinsic." (the masked popcount of lines 22 to 24)
  - Leader: "We only want a single lane to enter our new control flow, and we do this by checking if there are any active lanes below us. Only one lane will have 0 active lanes below us" (the lowest active lane, as `__ffs - 1`)
  - Broadcast: "We need to broadcast the value who was the lowest active lane (the first lane) to all other lanes in the wavefront." (`readfirstlane`, the shuffle of line 20)
  - Offset: "we need to get our individual lane's slice into the result. We use the lane offset we previously calculated combined with the atomic result value we got from the first lane". For a uniform increment the offset is `V * Mbcnt`; for per-lane values it is an exclusive scan.
- Enabled when optimising: the pass is added if `TM.getOptLevel() >= CodeGenOptLevel::Less` and the strategy is not `None`; the default strategy is `Iterative` (`AMDGPUOptions.td`). So at `-O1` and above, Fig. 12.2 on AMD already issues one atomic per wavefront per call.
- One limit: "The iterative scan runs once per active lane and costs more than the hardware serialization of a native LDS atomic", so an LDS atomic with per-lane values is left alone under the default strategy. The filter's increment of 1 is uniform, so this does not apply to it.
- Caveat carried from ch09 B5: this is upstream LLVM. I did not confirm that ROCm 10.2's `amdclang++` carries the same default.

- https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUAtomicOptimizer.cpp (lines 9 to 21, 195 to 260, 705 to 735, 784 to 800, 855 to 900)
- https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUOptions.td (line 203)
- https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp (lines 1422 to 1426, 2228 to 2231)

### B5. Library filters on AMD: hipCUB `DeviceSelect` over rocPRIM `select`

**Verdict: PARTLY.** The CUB names exist and forward to rocPRIM. rocPRIM exposes the book's dynamic block index as an option. Two things a lesson would want are not documented: rocPRIM's select pages never state in words that order is preserved (their example output is in order), and the in-place overload passes the same pointer as input and output to a function whose documentation says nothing about aliasing.

- hipCUB (`projects/hipcub`, the 4.7.0 copy at `therock-10.2`; infra B11 explains the two copies): `static hipError_t If(void* d_temp_storage, size_t& temp_storage_bytes, InputIteratorT d_in, OutputIteratorT d_out, NumSelectedIteratorT d_num_selected_out, int64_t num_items, SelectOp select_op, hipStream_t stream = 0)` returns `::rocprim::select(...)`. `Flagged` likewise. The in-place form `If(d_temp_storage, temp_storage_bytes, IteratorT d_data, d_num_selected_out, num_items, select_op, stream)` calls `If(..., d_data, d_data, ...)`.
- rocPRIM `device_select.hpp` (`projects/hipccl/hipccl3/rocprim`): "Performs a device-wide selection using selection operator." The example selects even values from `[1, 2, 3, 4, 5, 6, 7, 8]` and prints "output: [2, 4, 6, 8]". The template parameter `UsingOrderedBlockId`: "If true, uses an atomic counter to assign block id instead of natural blockIdx-based ordering. Can increase performance on MI3xx architectures when using streams. The default is false." So by default rocPRIM's look-back keys on `blockIdx` order (see B6).
- COULD NOT VERIFY: that in-place selection is safe on AMD. I searched rocPRIM's `device/detail/device_partition.hpp` at the tag for "alias", "in-place", "overlap": no hit, and no equivalent of CUB's acquire/release switch (A9) was found. That is absence of documentation, not a demonstrated bug.
- Look-back details worth knowing when reading rocPRIM: `lookback_scan_state.hpp` has a `UseSleep` option ("the execution of a wavefront is paused for a short duration, allowing other threads or processes to execute during idle periods", via `__builtin_amdgcn_s_sleep`) and uses "workgroup" scope fences plus cache-bypassing atomics on gfx942, gfx950 and gfx1250 because the "agent" fence there "does L2 cache flushing and invalidation".

- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipcub/hipcub/include/hipcub/backend/rocprim/device/device_select.hpp (lines 140 to 215)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipccl/hipccl3/rocprim/rocprim/include/rocprim/device/device_select.hpp (lines 169 to 262)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipccl/hipccl3/rocprim/rocprim/include/rocprim/device/detail/lookback_scan_state.hpp (lines 38 to 56, 100 to 126)

### B6. Spinning on an earlier block on AMD (the HIP side of A10)

**Verdict: COULD NOT VERIFY.** AMD documents no forward-progress guarantee for workgroups.

- The only statement found in the HIP documentation sources at the tag is in the reduction tutorial: "The API doesn't guarantee the order in which blocks are scheduled even though all GPUs schedule them in the same monotonically increasing order of block ids. Relying on this implicitly, the last block of a grid is in the optimal position to observe the side effects of all other blocks (using spinlocks or other methods)". That is an observation about current hardware, explicitly not an API guarantee.
- libhipcxx's documentation source at `therock-10.2` (`docs/libcudacxx/extended_api/execution_model.rst`) carries NVIDIA's execution-model text unchanged, including "CUDA C++ aims to provide parallel forward progress" and the thread-block-cluster rule of A10, with links to NVIDIA's pages. The published page I tried (`rocm.docs.amd.com/projects/libhipcxx/en/latest/libcudacxx/extended_api/execution_model.html`) returns 404. Text copied from the CUDA docs is not an AMD statement about AMD hardware.
- rocPRIM's look-back defaults to natural `blockIdx` order (B5), which works in practice only because of the scheduling order the tutorial describes.

For lessons: on HIP, present the dynamic block index as the defensive choice (it needs no ordering assumption) and the three-kernel stable filter as the documented-safe one.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/tutorial/reduction.rst (around line 702)
- https://github.com/ROCm/libhipcxx/blob/therock-10.2/docs/libcudacxx/extended_api/execution_model.rst

---

## Group C: Metal

### C1. The warp-aggregated append in MSL, with no cooperative groups

**Verdict: CONFIRMED.** Metal has a direct counterpart for every step of Fig. 12.3, two of them as single calls, all defined over the active threads of the SIMD-group. Names and types were confirmed by compiling (not running) a kernel with the local toolchain under `-std=metal3.0`, `-std=metal4.0` and the default; no errors.

| Fig. 12.3 / 12.4 step | MSL (MSL spec section 6.10.2) |
|---|---|
| Active mask (`__activemask()`) | `simd_vote simd_active_threads_mask()`: "This function is equivalent to simd_ballot (true) and sets the bits that represent active threads to 1, and inactive Threads to 0." |
| Vote on a predicate (`__ballot_sync`) | `simd_vote simd_ballot(bool expr)`: "a bitmask of the evaluation of the Boolean expression for all active threads in the SIMD-group for which expr is true. The function sets the bits that correspond to inactive threads to 0." |
| Leader (`__ffs(mask) - 1`, or `thread_rank() == 0`) | `bool simd_is_first()`: "Returns true if the current thread is the first active thread, the active thread with the smallest index, in the current SIMD-group; otherwise, false." Or `ctz(mask)` (C2). |
| Count (`__popc(mask)`, or `size()`) | `T simd_sum(T data)` with `data = 1`: "Returns the sum of the input values in data across all active threads in the SIMD-group and broadcasts the result to all active threads". Or `popcount(mask)` (C2). |
| Broadcast the reserved start from the leader (`__shfl_sync(mask, j, leader)`) | `T simd_broadcast_first(T data)`: "Broadcasts data from the first active thread, the active thread with the smallest index, in the SIMD-group to all active threads." |
| Rank among the passing lanes (lines 22 to 24, or `thread_rank()`) | `T simd_prefix_exclusive_sum(T data)` with `data = 1`: "For a given thread, returns the sum of the input values in data for all active threads with a lower index in the SIMD-group. The first thread in the group, returns T(0)." |
| The atomic | `atomic_fetch_add_explicit(counter, count, memory_order_relaxed)` on `device atomic_uint*` (infra C1) |

- "SIMD-group functions allow threads in a SIMD-group ... to share data without using threadgroup memory or requiring any synchronization operations, such as a barrier." "An active thread is a thread that is executing. An inactive thread is a thread that is not executing. For example, a thread may not be active due to flow control".
- Availability: `simd_ballot`, `simd_is_first`, `simd_broadcast_first` "macOS: Metal 2.1 and later"; `simd_sum` and `simd_prefix_exclusive_sum` "macOS: Metal 2.1 and later", GPU family Apple7 and later for the reductions (infra C2). Every Apple silicon Mac is Apple7 or later (infra Group C intro).
- The form that compiled, written in uniform control flow (ours; every lane calls the SIMD functions, which sidesteps C3): `bool keep = gid < n && v > thr; uint rank = simd_prefix_exclusive_sum(keep ? 1u : 0u); uint count = simd_sum(keep ? 1u : 0u); uint base = 0; if (simd_is_first()) base = atomic_fetch_add_explicit(counter, count, memory_order_relaxed); base = simd_broadcast_first(base); if (keep) out[base + rank] = v;`. Note that in uniform control flow `simd_is_first()` is lane 0 even if lane 0 does not keep, and it reserves for the whole SIMD-group, which is what one atomic per SIMD-group means. It also makes the order within the SIMD-group follow lane order. Compiled only; not run.
- There is no Metal counterpart of cooperative groups (infra table, ch04 C3); the table above is the replacement.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 6.10.2, Tables 6.14 and 6.15, pages 217 to 225)

### C2. `simd_ballot` gives a 64-bit vote with undefined top bits; `popcount` and `ctz` work on `ulong`

**Verdict: CONFIRMED.** The ballot route of Fig. 12.3 (mask, popcount, masked popcount) is also expressible in MSL, but the mask is 64 bits wide whatever the SIMD width, and its unused top bits are undefined.

- Spec, `simd_vote` example: `explicit constexpr simd_vote(vote_t v = 0); explicit constexpr operator vote_t() const;` "bit i in v represents the 'vote' for the thread in the SIMD-group at index i", with a private `uint64_t v`. "On hardware with fewer than 64 threads in a SIMD-group, the value of the top bits in simd_vote is undefined. Because you can initialize these bits, do not assume that the top bits are set to 0."
- Toolchain fact (local `metal_simdgroup` header and predefined macros): `typedef __METAL_VOTE_T__ vote_t;` with `#define __METAL_VOTE_T__ long unsigned int`, that is MSL `ulong`. Convert with `simd_vote::vote_t(simd_ballot(keep))`, as llama.cpp does (D2).
- Integer functions (MSL spec section 6.4, Table 6.2; "T is one of the scalar or vector integer types"): `T popcount(T x)` "Returns the number of nonzero bits in x." `T ctz(T x)` "Returns the count of trailing 0-bits in x. If x is 0, returns the size in bits of the type of x". `T clz(T x)` likewise. The local `metal_integer` header declares `ulong popcount(ulong)`, `ulong ctz(ulong)`, `ulong clz(ulong)`. So `ctz(mask)` is the leader lane (the MSL form of `__ffs(mask) - 1`, with "no lane" reported as 64 rather than -1).
- Consequences (ours): the rank `popcount(mask & ((1ul << lane) - 1))` never touches the undefined top bits, because `lane` is below the SIMD width. The count does: use `simd_sum`, or mask with the lanes that exist (`simd_size == 64 ? ~0ul : (1ul << simd_size) - 1`), reading `simd_size` from `[[threads_per_simdgroup]]`. Both compiled (C1).
- Types the SIMD functions accept exclude `long` and `ulong` ("Table 6.14 uses the placeholder T to represent a scalar or vector of any integer or floating-point type, except: bool, bfloat, long, ulong ..."). A 64-bit running output index cannot be broadcast with `simd_broadcast_first` directly; keep counts in `uint` (PyTorch's MPS `nonzero` says the same about shuffles, D5).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 6.4 and 6.10.2, pages 201 to 202 and 225)
- Local headers: Metal toolchain `usr/metal/32023/lib/clang/32023/include/metal/metal_simdgroup` (line 74) and `metal_integer` (lines 3049 to 3111, 4270 to 4332)

### C3. Inside a branch, are the active threads exactly the threads that took the branch?

**Verdict: COULD NOT VERIFY.** Apple defines active and inactive threads and says flow control can make a thread inactive, but documents neither NVIDIA's warning (A1) nor AMD's "exact active mask" guarantee (B1), and says nothing about independent scheduling of threads within a SIMD-group.

- What is documented: SIMD-groups (not threads) "execute concurrently within a given threadgroup and make independent forward progress with respect to each other, in the absence of threadgroup barrier operations" (MSL spec section 5.2.3.6; infra C5). The mapping of threads to SIMD-groups "is invariant for the duration of a kernel's execution" (section 4.4.1).
- For lessons: say that the CUDA, HIP and Metal documents give three different answers (no, yes, unstated), and write the Metal version in uniform control flow as in C1, where the question does not arise.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 4.4.1, 5.2.3.6, 6.10.2)

### C4. Spinning on an earlier threadgroup on Metal (the Metal side of A10)

**Verdict: COULD NOT VERIFY.** The MSL spec's only forward-progress statement is about SIMD-groups inside one threadgroup (C3). The spec contains no statement about threadgroups making progress relative to each other (searched "progress", "starv", "deadlock", "spin", "any order"); section 4.4 says only that "a GPU can execute a separate threadgroup for each of its compute units."

Consequences (ours): a single-dispatch decoupled look-back, or the cross-threadgroup ordering that §12.7's in-place filter relies on, rests on undocumented behaviour on Apple GPUs, and there is no grid barrier or cooperative launch to fall back on (ch04 C3). The documented-safe Metal designs are:

- several dispatches (count, scan of block counts, scatter: the book's exercise 1), which is what PyTorch's MPS `nonzero` does (D5);
- one threadgroup per row, with only threadgroup barriers, which is what llama.cpp's Metal top-k does (D2) and what fits the M5 lab's batch-by-vocabulary layout.

The memory orders a look-back would need exist from Metal 4.1 only, and the spec contradicts itself on them for read-modify-write functions (infra C1).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 4.4, 4.4.1, 5.2.3.6)

### C5. A library compaction on Apple GPUs

**Verdict: PARTLY.** MPSGraph has a non-zero-indices operation, the counterpart of `torch.nonzero`; its output order is shown by example, not stated. I found no MPS kernel for compacting the values of an `MTLBuffer` by a predicate, and Metal has nothing like CUB's `DeviceSelect`.

- Header `MPSGraphNonZeroOps.h` (macOS 27.0 SDK): `-(MPSGraphTensor *) nonZeroIndicesOfTensor:(MPSGraphTensor *) tensor name:(NSString * _Nullable) name` (Swift `nonZeroIndices(_:name:)`), `MPS_AVAILABLE_STARTING(macos(14.0), ios(17.0), tvos(17.0))`. "Computes the indices of the non-zero elements of the input tensor. The indices are returned as a two-dimensional tensor of size `[number_of_nonzeros, input_rank]`." "Returns: A valid MPSGraphTensor containing indices in signed int32 data type." The example lists the indices of `[[1, 0, 3], [0, 10, 0]]` as `[[0, 0], [0, 2], [1, 1]]`, that is in row-major order.
- Not stated: ordering in general, and how the graph handles the data-dependent output shape.

- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/nonzeroindices(_:name:)
- Local header: `MetalPerformanceShadersGraph.framework/Headers/MPSGraphNonZeroOps.h` (macOS 27.0 SDK)

---

## Group D: How llama.cpp, PyTorch and MLX do the same things

Read on the default branches on 2026-11-20; dates are the last commit to the file. These are snapshots: in lessons, link to files and quote short lines, not line numbers.

### D1. llama.cpp, CUDA (and HIP): a per-row top-k gather that is Fig. 12.2 for the clear winners and Fig. 12.3's ballot rank for ties

**Verdict: CONFIRMED.** `ggml/src/ggml-cuda/top-k.cu` (2026-11-19), the radix path used for multi-row top-k (`top_k_radix_cuda`), selects the k-th key per row by 8-bit radix histograms and then gathers with a filter in `top_k_radix_gather`. It is chapter 12 inside an LLM sampler.

- Keys strictly above the threshold are appended with one atomic per kept key on a per-row counter (Fig. 12.2, unstable):

```cpp
if (key > threshold) {
    const int pos = atomicAdd(&state->greater_count, 1);
    row_dst[pos] = col;
}
```

  The per-row counters sit on separate cache lines: `// one state per cache line, so the gather atomics of different rows do not compete` / `struct alignas(128) top_k_radix_state { uint32_t prefix; uint32_t prefix_mask; int rank; int greater_count; };`.
- Keys equal to the threshold are placed in column order, with a ballot and a masked popcount in uniform control flow (the A1 advice): `const ggml_cuda_thread_mask_t ties = ggml_cuda_ballot(is_tie); const int tie_pos = pos + ggml_cuda_popc(ties & (((ggml_cuda_thread_mask_t) 1 << lane) - 1));` then `pos += ggml_cuda_popc(ties);`, with the comment "the ties fill the last rank slots in column order, the lowest columns win."
- The portability helpers in `common.cuh` (2026-11-19) close the B1 trap: `ggml_cuda_ballot` wraps `__ballot_sync(mask, predicate)` with the typed full mask (ch10 D1), and `ggml_cuda_popc` calls `__popc` or `__popcll` by the size of `ggml_cuda_thread_mask_t`, followed by `template<typename T> static __device__ __forceinline__ int ggml_cuda_popc(T x) = delete; // Makes implicit type casts result in an error.`
- The same file uses CUB's `DeviceTopK::MaxPairs` for single rows when CCCL is new enough: `// DeviceTopK has a race condition before CCCL 3.4.3.` (guarded by a CCCL version check; see A8 for what `DeviceTopK` guarantees).
- So the output order of a llama.cpp CUDA top-k over many rows is: winners in atomic order (run-dependent), then ties in column order (our reading). That is a concrete case for the M5 lab's "order preserved" requirement and its test.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/top-k.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh (lines 451 to 463)

### D2. llama.cpp, Metal: one threadgroup per row, `simd_ballot` plus `ctz`, and a deterministic compaction that is not column order

**Verdict: CONFIRMED.** `ggml/src/ggml-metal/kernels/argsort.metal` (2026-11-08), `kernel_top_k_f32_i32`: one threadgroup per row (`tgpig` indexes the row), radix select with a 256-bin `threadgroup atomic_uint` histogram (ch09 D4), then a compaction written with the C1/C2 functions.

- Finding the first lane whose running sum reaches `desired` is the leader election of Fig. 12.3 in MSL: `const simd_vote::vote_t hit = simd_vote::vote_t(simd_ballot(acc + sum >= desired)); if (tid == (uint) ctz(hit)) { ... }`.
- The compaction is the book's §12.5 to §12.6 shape: each lane counts its kept elements over a strided chunk (coarsening), `simd_sum` gives per-SIMD-group totals, the first SIMD-group scans those with `simd_prefix_exclusive_sum`, and each lane writes at `base_wa + excl_a + na` with `excl_a = simd_prefix_exclusive_sum(ca)`. The comment: "deterministic compaction: rank the kept elements by column order, so the result does not depend on the thread scheduling", and further down: "the lanes write their kept elements at lane-major ranks, which does not depend on the thread scheduling. with more tied candidates than remaining slots, the ties are instead ranked in column order with a per-step cross-lane scan".
- Our reading: because each lane owns a stride-32 substream and lanes are ranked before their elements, elements above the threshold land in lane-major order, not column order, so the first comment overstates. The tie path, which scans per 32-element step, is in column order. Deterministic is not the same as stable, which is the distinction the M5 lab tests.
- Width: `const uint lane = tid & 31u; const uint wg = tid >> 5u;` hard-codes 32-lane SIMD-groups, the assumption ch04 D2 recorded for llama.cpp's Metal backend.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/argsort.metal (lines 252 to 431)

### D3. llama.cpp samplers: the host min-p is a stable filter; the experimental GPU samplers mask instead of compacting

**Verdict: CONFIRMED.** `src/llama-sampler.cpp` (2026-11-12).

- Host min-p over the candidate array (ch01 D2: the default sampler chain runs on the host): `const float min_logit = max_logit + logf(ctx->p); // min logit for p_i >= p * p_max`, then `if (cur_p->data[i].logit >= min_logit) { filtered_tokens.push_back(cur_p->data[i]); }`, copied back with `cur_p->size = filtered_tokens.size();`. That is a sequential stable filter of (token id, logit) pairs against a threshold, exactly the M5 lab's operation, with its CPU reference written for us. The threshold test on logits avoids computing a softmax.
- Experimental backend path (`llama_sampler_min_p_backend_apply`, `llama_sampler_top_p_backend_apply`): no compaction; the graph keeps a fixed shape and sets losers to minus infinity: "Create a mask where logits below the threshold are 0 (discard), and others are 1 (keep)." `ggml_step(ctx, sub)`, then "Apply -INFINITY bias for masked-out tokens / log(1) = 0 (keep), log(0) = -INF (discard)" with `ggml_log` and `ggml_add`.
- Backend top-k does shrink the row, but by a static `k`: `ggml_top_k(ctx, logits, sctx->k)` followed by `ggml_get_rows(ctx, logits_rows, top_k)`, a gather with a known output size.
- Design point for the lessons (ours): a compute graph with static shapes cannot express "output length = number of survivors", so GPU graph samplers mask; a hand-written kernel can compact and return a count, which is what chapter 12 teaches and what the M5 lab builds.

- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp (lines 1456 to 1520, 1634 to 1720, 1756 to 1883)

### D4. PyTorch, CUDA and ROCm: `nonzero` is CUB select plus a host sync; `nonzero_static` is exercise 1; `topk` gathers with ballots and refuses atomics for order

**Verdict: CONFIRMED.**

- `aten/src/ATen/native/cuda/Nonzero.cu` (2026-05-27), `nonzero_cuda_out_impl`: count with `cub::DeviceReduce::Sum` over a `NonZeroOp` transform iterator, copy the count to pinned host memory with `at::cuda::memcpy_and_sync`, size the output, then `cub::DeviceSelect::Flagged(..., counting_itr, itr, out_temp.mutable_data_ptr<int64_t>(), ...)`: a stable filter of indices (A8). The documentation states the cost: "When input is on CUDA, torch.nonzero() causes host-device synchronization." and "The result is sorted lexicographically, with the last index changing the fastest (C-style)."
- `torch.nonzero_static(input, *, size, fill_value=-1)`: "Pad the out tensor with fill_value if the size is larger than total number of non-zero elements, truncate out tensor if size is smaller." Its CUDA path is three kernels, the book's exercise 1 with coarsening: `calc_block_sums` (count per block), `compute_agg` (one block scans the block counts with `BlockScan<int64_t, ...>::InclusiveSum`), and `flag_kernel` (`BlockLoad`, `BlockScan::ExclusiveSum` of the flags, `BlockExchange::BlockedToStriped` "for a slightly better coalescing for writes", scatter), with `BLOCK_THREADS = 512`, `ITEMS_PER_THREAD = 16` and a grid sized near `multiProcessorCount`. No host sync. The CUB types go through a `ROCM_HIPCUB(...)` macro, so the same file builds against hipCUB for ROCm.
- `aten/src/ATen/native/cuda/TensorTopK.cu` (2026-10-15), `gatherTopK`, the design comment: "Here each warp counts its matches with a ballot and publishes the count to shared memory; after a single barrier every thread derives its warp's exclusive prefix and the iteration total. Results therefore land in ascending input-index order, exactly like the prefix-scan version, so the output is deterministic run to run (an atomicAdd-based reservation would make the order depend on warp scheduling)." That sentence is the book's §12.2 versus §12.5 choice made by PyTorch for `torch.topk`.
- The older helper is the book's binary prefix sum (ref. [1], p. 293), with the HIP fix: `aten/src/ATen/cuda/ScanUtils.cuh` (2023-08-07), `inclusiveBinaryPrefixScan`: `#if defined (USE_ROCM)` `unsigned long long int vote = WARP_BALLOT(in); T index = __popcll(getLaneMaskLe() & vote); T carry = __popcll(vote);` `#else` the same with `__popc`. `aten/src/ATen/cuda/AsmUtils.cuh` (2026-07-19) builds `getLaneMaskLe()` from PTX `%lanemask_le` on CUDA and as `UINT64_MAX >> (64 - (getLaneId() + 1))` on ROCm (not from HIP's undocumented `__lanemask_lt`, B2).

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Nonzero.cu (lines 74 to 296, 330 to 365)
- https://docs.pytorch.org/docs/2.15/generated/torch.nonzero.html
- https://docs.pytorch.org/docs/2.15/generated/torch.nonzero_static.html
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/TensorTopK.cu (lines 185 to 215)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/ScanUtils.cuh (lines 13 to 70)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/AsmUtils.cuh (lines 105 to 135)

### D5. PyTorch, MPS: `nonzero` as three dispatches of block-local SIMD scans

**Verdict: CONFIRMED.** `aten/src/ATen/native/mps/operations/Indexing.mm` (2026-11-19) and `kernels/Indexing.metal` (2026-10-31).

- Host comment: "Metal kernel-based nonzero using prefix-sum + scatter. Step 1: Per-block nonzero totals (block-local prefix scan). Step 2: GPU prefix sum of block totals → block offsets + total count. Host (optional): Read back total count, allocate output, unless max_element is provided. Step 3: Scatter multi-dimensional indices into the output." The dynamic path reads the total with `total_nonzero_buf.item<int64_t>()` (a sync, as on CUDA); `nonzero_static` skips it.
- Kernel side: `count_nonzero_prefix_sum` does an inclusive SIMD-group scan with `simd_shuffle_and_fill_up`, writes SIMD-group totals to `threadgroup uint simdgroup_totals[32]`, barrier, scans those in the first SIMD-group; step 2 runs "in a single threadgroup". The kernel comment explains a type limit that matches C2: "Metal 3.x cannot simd_shuffle 64-bit integers, so the i64 path scans in threadgroup memory."
- This is the multi-dispatch design that C4 recommends on Metal, in shipping code. Our note: `simd_shuffle_and_fill_up` is listed under GPU family Apple8 in the feature tables (infra C2); how PyTorch serves Apple7 (M1-series) for this kernel was not checked.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Indexing.mm (lines 309 to 445)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/Indexing.metal (lines 1008 to 1200)

### D6. MLX and mlx-lm: no data-dependent shapes, so samplers mask

**Verdict: CONFIRMED.**

- MLX docs, "Indexing Arrays" (MLX 0.32.4): "Indexing with boolean masks is something that MLX may support in the future. In general, MLX has limited support for operations for which output shapes are dependent on input data. Other examples of these types of operations which MLX does not yet support include numpy.nonzero() and the single input version of numpy.where()." "Boolean mask based indexing is supported for assignment only". So there is no MLX-level compaction to call.
- mlx-lm `mlx_lm/sample_utils.py` (2026-09-24): `apply_top_k` uses `mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]` and `mx.put_along_axis(logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1)`; `apply_min_p` ends `return mx.where(tokens_to_remove, -float("inf"), logprobs)`; `apply_top_p` ends `return mx.where(logprobs < threshold, -float("inf"), logprobs)`. Every filter is a mask over the full vocabulary row, as in llama.cpp's backend samplers (D3).

- https://ml-explore.github.io/mlx/build/html/usage/indexing.html
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py (lines 138 to 229)

---

## Project notes (not vendor facts)

Recorded so lab authors do not have to rediscover them; these describe this repository on the check date.

- The emulators model what Figs. 12.3 and 12.4 need: `__activemask`, `__ballot_sync`, `__shfl_sync`, `__popc`, `__popcll`, `__ffs`, `__ffsll` (`labs/common/include/pmpp/emu/emu_device.h`), and a `coalesced_group` with `num_threads()`, `size()`, `thread_rank()` and `shfl` (`cooperative_groups_emu.h`). Its header comment lists as not provided: "binary_partition, labeled_partition, match_any/match_all on groups, and the reduce / scan / memcpy_async / invoke_one headers". The emulated warp is 32 lanes and the emulated masks are `unsigned`.
- `labs/common/include/pmpp/vendors/hip.h` already documents, in its header comment, the 64-bit mask, the cooperative-groups header path and the libhipcxx requirement; it cannot fix `__popc` versus `__popcll` (a type issue, not a name), so a lab `.cu` needs a typed mask and a typed popcount, as llama.cpp does (D1).

## Book items with no vendor fact to check

- Fig. 12.1's two "(a)" captions; p. 300's "k0" that should be "k1"; p. 301's second "earlier" that should be "later"; Fig. 12.8's undefined `gridExclusiveScan` and missing `i < N` guard; the Fig. 12.6 typo and the reused name `j`; the arbitrary block order in Fig. 12.5. All book-internal (notes, "Unclear in the PDF").
- References [1] (Harris and Garland, 2012) and [2] (Sung et al., 2014): bibliographic, not checked. Reference [4]'s URL: 404 (A8).

## Answers to the notes' portability questions

- Device-scope atomic counter: HIP same spelling through libhipcxx or legacy `atomicAdd` (infra B1, B2); Metal `atomic_fetch_add_explicit` on `device atomic_uint*` (infra C1).
- Block-scope atomic on a shared counter: HIP libhipcxx `thread_scope_block` (no `atomicAdd_block`); Metal `threadgroup atomic_uint` (ch09 C1).
- `__activemask`, `__ffs`, `__popc`, `__shfl_sync`: HIP same names with 64-bit masks and `__popcll`/`__ffsll` (B1); Metal `simd_active_threads_mask`/`simd_ballot`, `ctz`, `popcount`, `simd_broadcast_first`, or the one-call `simd_prefix_exclusive_sum` and `simd_sum` (C1, C2). Width: read it, never assume 32 (settled table).
- `__all_sync`, `__any_sync`, `__ballot_sync`: HIP same names, 64-bit (infra B3); Metal `simd_all`, `simd_any`, `simd_ballot` (infra C2).
- `coalesced_group`: HIP yes, different header, 64-bit member mask, `exclusive_scan` but no `exclusive_scan_update` or `invoke_one` (B3); Metal none (C1).
- The compiler's automatic aggregation: NVIDIA, blog only (A7); AMD, documented in LLVM source and on by default at `-O1` and above (B4); Apple, not documented (ch09 table).
- `gridExclusiveScan` and in-place ordering: CUDA has a documented forward-progress rule that the dynamic block index satisfies (A10); AMD and Apple document none (B6, C4); the three-kernel stable filter needs none of it.

---

## Summary of corrections

Every WRONG, PARTLY and COULD NOT VERIFY item, one line each:

- A1: NVIDIA's guide now warns that `__activemask()` "cannot be used to determine which warp lanes execute a given branch"; Fig. 12.3 stays correct (each converged snapshot aggregates on its own and the `_sync` shuffle re-converges it), but "one atomic per warp" is not guaranteed. Use `__ballot_sync(FULL_MASK, keep)` in uniform control flow for the exact pass set.
- A3: the book's `(1 << laneIdx()) - 1` is right for 32 lanes; CUDA documents `%lanemask_lt` (PTX) and `cuda::device::lane_mask::all_less()` (`<cuda/warp>`, CCCL 3.1 and later); the blog's `__lanemask_lt()` is in no current NVIDIA reference page.
- A5: `coalesced_group::size()` is a "legacy" alias of `num_threads()`; the book omits `#include <cooperative_groups.h>` and the `cooperative_groups` namespace.
- A7: "the compiler already coalesces atomics" comes only from a 2017-updated NVIDIA blog, without conditions; "it refrains when only one thread is active" is in no NVIDIA source found.
- A8: Thrust and CUB filters are all stable (`copy_if is stable`, "maintain their original relative ordering"), CUB has in-place select overloads, and the book's CUB URL returns 404; `cub::DeviceTopK` currently implements only the unordered, non-deterministic configuration.
- B1 (WRONG): Fig. 12.3 does not compile under HIP (32-bit shuffle mask rejected), and widening only the shuffle mask leaves `unsigned int activeThreads`, `__popc`, `__ffs` and `1 << lane` silently wrong on 64-wide wavefronts; use 64-bit masks, `__popcll`, `__ffsll`, `1ull << lane`.
- B2: HIP's `__lanemask_lt()` and `__lane_id()` exist only in headers, and `__lanemask_lt()` returns active lanes below the caller, unlike PTX's `%lanemask_lt`.
- B3: HIP coalesced groups match CUDA's names but use `<hip/hip_cooperative_groups.h>`, a 64-bit member mask, and lack `exclusive_scan_update` and `invoke_one`; HIP's `exclusive_scan` docs say the first lane's value is "platform-dependent".
- B5: hipCUB's `DeviceSelect` forwards to `rocprim::select`, whose docs show but do not state order preservation; the in-place overload aliases input and output, and no aliasing guarantee or in-place ordering was found in rocPRIM (could not verify that it is safe).
- B6 (could not verify): AMD documents no forward progress for workgroups; its own tutorial says block order is "not guaranteed" by the API though hardware schedules in increasing order; libhipcxx's execution-model source is NVIDIA's text and its published page returns 404.
- C3 (could not verify): Apple does not say whether the active threads inside a branch are exactly the threads that took it; write Metal SIMD-group code in uniform control flow.
- C4 (could not verify): Apple documents no forward progress between threadgroups, so single-dispatch look-back and cross-threadgroup in-place ordering are undocumented on Metal; use several dispatches or one threadgroup per row.
- C5: MPSGraph `nonZeroIndices` (macOS 14 and later, `int32` indices) is the only library compaction found on Apple GPUs; its ordering is shown by example, not stated.

Path and status facts worth knowing even where the verdict is CONFIRMED:

- A6: CUDA cooperative groups now provide `exclusive_scan`, `exclusive_scan_update` (the whole warp-aggregated append in one call), `invoke_one_broadcast` and `binary_partition`, and the guide's scan example is a stream compaction.
- A9: CUB's select stages through shared memory only when a tile keeps more items than threads, and uses release/acquire on its look-back state exactly when selection may be in place; the book's §12.7 claim needs that ordering (the book's Fig. 11.17 has it).
- A10: CUDA's execution model (guide section 5.9) guarantees progress only within a thread-block cluster once a thread has started, which is what makes the dynamic block index necessary for spinning on earlier blocks.
- B4: AMD's LLVM backend rewrites a uniform-address atomic into ballot, mbcnt rank, lowest-lane atomic and `readfirstlane` broadcast at `-O1` and above: Fig. 12.3, automatically.
- D1, D2, D4: llama.cpp's CUDA top-k uses Fig. 12.2 for winners and a ballot rank for ties; its Metal top-k compacts deterministically but in lane-major, not column, order; PyTorch's `topk` gather rejects atomics explicitly to keep ascending index order.
- D3, D6: GPU graph samplers (llama.cpp backend, mlx-lm) mask losers with minus infinity rather than compact; llama.cpp's host min-p is a sequential stable filter on the threshold `max_logit + logf(p)`.
- D4, D5: `torch.nonzero` syncs with the host on CUDA and MPS; `nonzero_static` does not and is the three-kernel stable filter of exercise 1.

## Where the backends really differ

One line per cell, for the operations this chapter uses. Nothing here was run on CUDA or AMD hardware; the Metal cells marked "compiles" were compiled, not run.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Active-lane mask | `unsigned __activemask()`, 32-bit; documented as a snapshot, not the branch's lane set | `unsigned long long __activemask()`; header says it is "the exact active mask" | `simd_active_threads_mask()` returns `simd_vote` (64-bit `vote_t`); branch semantics not stated |
| Vote on a predicate | `__ballot_sync(mask, p)` returns `unsigned` | `__ballot_sync(mask, p)` with 64-bit mask, returns `unsigned long long` | `simd_ballot(p)`; top bits undefined below 64 lanes |
| Leader lane | `__ffs(mask) - 1` (1-based, 0 for none) | `__ffsll(mask) - 1`; `__ffs` truncates a 64-bit mask | `simd_is_first()`, or `ctz(mask)` (64 for none); compiles |
| Count of passing lanes | `__popc(mask)` | `__popcll(mask)`; `__popc` truncates | `simd_sum(keep ? 1u : 0u)` or `popcount(mask & lanes)`; compiles |
| Lanes-below mask | `(1u << lane) - 1`, PTX `%lanemask_lt`, `cuda::device::lane_mask::all_less()` (CCCL 3.1+) | `(1ull << lane) - 1`; header-only `__lanemask_lt()` returns active lanes below | `(1ul << lane) - 1` on the `ulong` vote |
| Rank among passing lanes in one call | `cg::exclusive_scan(coalesced_threads(), 1)` | `exclusive_scan` in `<hip/cooperative_groups/hip_scan.h>` | `simd_prefix_exclusive_sum(keep ? 1u : 0u)` (active threads); compiles |
| Broadcast the reserved start | `__shfl_sync(mask, j, leader)` or `g.shfl(j, 0)` | Same names, 64-bit mask; `g.shfl` maps rank to lane | `simd_broadcast_first(j)`; compiles |
| Whole aggregated append in one call | `cg::exclusive_scan_update(group, atomic_ref, 1)` | None (no `exclusive_scan_update`) | None |
| Coalesced group | `<cooperative_groups.h>`; `size()` legacy alias of `num_threads()` | `<hip/hip_cooperative_groups.h>`; 64-bit member mask | None |
| Compiler aggregates a uniform-address atomic | "In many cases" (2017 blog); conditions undocumented | LLVM atomic optimizer, on at `-O1`+: ballot, mbcnt, lowest lane, `readfirstlane` | Not documented |
| Library stable filter | CUB `DeviceSelect::If/Flagged/Unique` (stable, in-place overloads), Thrust `copy_if`/`remove_if` (stable) | hipCUB `DeviceSelect` over `rocprim::select`; order shown, not stated; in-place aliasing undocumented | MPSGraph `nonZeroIndices` (macOS 14+, indices only); no value select |
| Forward progress when one block waits on an earlier one | Documented: started threads' cluster progresses (guide 5.9); dynamic block index makes the wait legal | Not documented; tutorial says increasing block order is hardware behaviour, not API | Not documented between threadgroups |
| In-place stable filter | CUB in-place select uses release/acquire on look-back state | rocPRIM: no in-place handling found | Use several dispatches or one threadgroup per row |
| Data-dependent output size in frameworks | `torch.nonzero` syncs with host; `nonzero_static` does not | Same PyTorch source through hipCUB | PyTorch MPS `nonzero` reads back the count; MLX has no `nonzero` and mlx-lm masks |
