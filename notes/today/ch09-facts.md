# Chapter 9 fact check: histogram, atomics, privatisation (CUDA, HIP, Metal, real backends)

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md` and `notes/today/infra-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY, and judges the assumption stated in the item title (usually the book's sentence, quoted with its page from `notes/ch09.md`). **Quote** is copied from the source with line breaks removed.
- Already settled elsewhere, referred to by item number and not repeated: legacy atomics and their types (infra A1, B1); `<cuda/atomic>`, the five scope enumerators, both memory-order spellings, `fetch_min`/`fetch_max`, the missing toolkit version (infra A2); libhipcxx as HIP's `<cuda/atomic>` (infra B2); MSL atomic types, functions and memory orders (infra C1); Apple's silence on caches and banks (infra C13); 1024 threads per block (ch05 A5, ch03 B3 for AMD); shared memory opt-in above 48 KB (ch05 A3); clusters and distributed shared memory (ch05 A6, infra A4, B4); Metal's 32 KB threadgroup memory (ch05 C1); `threadgroup_barrier` (ch05 C2); `__syncthreads` (ch04 A6); block scheduling (ch04 A5); divergence (ch04 A9, A10).
- Versions the sources reported on the check date: CUDA Programming Guide v13.5 (pages "Last updated on Oct 31, 2026"); CCCL `main` (last commit to `libcudacxx/include/cuda/__atomic` 2026-11-19); Compute Sanitizer and Nsight Compute docs 2026.4.0; PTX ISA 9.5; HIP docs "HIP 7.17.0" (ROCm 10.2.0), headers at tag `therock-10.2`; libhipcxx docs 3.0.2; MSL Specification 4.1 (2026-06-04); local macOS SDK 27.0; Clang docs at `clang.llvm.org/docs` (in-progress branch). Project files were read on their default branches today; the last-commit date is given where it matters.
- Not checked: anything by running code. Every statement is from documentation, headers or project source. Device figures appear here only so authors can see where they come from; lessons read the reader's own numbers from `python3 labs/pmpp.py doctor`.

Totals: 19 CONFIRMED, 0 WRONG, 9 PARTLY, 2 COULD NOT VERIFY (30 items: A1 to A10, B1 to B6, C1 to C6, D1 to D7, E1). The PARTLY and COULD NOT VERIFY items are collected under "Summary of corrections".

---

## Group A: CUDA, and the book's dated statements

### A1. `cuda::atomic_ref` and `fetch_add` as the book prints them (pp. 206 to 209)

**Verdict: PARTLY.** The API the book uses is current and is what NVIDIA recommends. Three details of the printed text differ from the library: the scope parameter has a default (system scope), `fetch_add`'s order parameter has a default (`seq_cst`), and the printed declaration is a schematic, not the header.

Book: `template <typename T, cuda::thread_scope> class cuda::atomic_ref;` (p. 207); `T fetch_add(T arg, cuda::std::memory_order order)` (p. 208); construction "adds little execution cost at run time" (p. 208); methods for "addition, minimum, maximum, and, or, xor" (p. 208).

What the library says today:

- Declaration (libcu++ docs): `template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system> class cuda::atomic_ref;` "It has the same interface and semantics as cuda::std::atomic_ref, with the following additional operations": `fetch_min`, `fetch_max`. So leaving out the scope gives system scope, the widest one, which NVIDIA advises against when a narrower scope suffices (A2).
- Constructor (header `cuda/__atomic/atomic.h`, `main`): `_CCCL_HOST_DEVICE_API explicit constexpr atomic_ref(_Tp& __ref)`. It takes a reference and is `explicit` and `constexpr`. The book's `bins_ref(bins[b])` is the right form; `= bins[b]` (copy-initialisation) would not compile. The constructor only stores the address, which agrees with the book's "little execution cost"; no NVIDIA document states a cost.
- `fetch_add` (header `cuda/std/__atomic/api/common.h`, `main`): `remove_cv_t<_Tp> fetch_add(remove_cv_t<_Tp> __op, memory_order __m = memory_order_seq_cst) const noexcept`. The order is an ordinary function argument with a default of `seq_cst`; the book's "compile-time constant" is how it is normally passed, not a requirement of the signature (our reading). The literal `1` in the book's call is an `int` converted to `unsigned int` by the parameter type.
- Trap for lessons: the same header defines `operator++()` as `fetch_add(1) + 1` with the default order, so `++bins_ref` is atomic but `seq_cst`. Only the explicit `fetch_add(1, cuda::memory_order_relaxed)` matches the book.
- Member names for the book's list: `fetch_add`, `fetch_sub`, `fetch_and`, `fetch_or`, `fetch_xor` (from the standard) and `fetch_min`, `fetch_max` (CUDA extensions; standard C++ only gains them in C++26 per cppreference). Namespace spelling of the order: both `cuda::memory_order_relaxed` and `cuda::std::memory_order_relaxed` are valid (infra A2), which resolves the notes' "Unclear" item.
- Rules a lesson needs, from the libcu++ page: "No object or subobject of an object referenced by an atomic_ref shall be concurrently referenced by any other atomic_ref that has a different Scope." "The referenced object must be aligned to its size: alignof(T) == sizeof(T)." "Types of T, where sizeof(T) < 4, are not natively supported by the underlying hardware. For these types atomic operations are emulated and will be drastically slower. Contention with contiguous memory in the current 4 byte boundary will be exacerbated." And: before compute capability 6, `atomic_ref` "may not be used" (irrelevant to current toolkits, whose tables start at 7.5).
- The book's footnote 2 reason for using a reference (plain access elsewhere) is legal under the C++ rule, provided the plain access does not overlap a live reference. cppreference: "While any std::atomic_ref instance referencing an object exists, the object must be exclusively accessed through these std::atomic_ref instances." In Figs. 9.9 and 9.10 every update-phase reference has gone out of scope before the barrier, so the plain read in the commit phase is fine (our reading).
- NVIDIA's own guide example has a slip: section 2.3.5.1 writes `cuda::atomic_ref<float, cuda::thread_scope_device> result_ref(result);` with `float *result`, which passes a pointer to a constructor that takes `T&` (it needs `*result`). Do not copy it.
- The book's reference [1], cppreference, still documents `std::atomic_ref` ("since C++20").

- https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html
- https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__atomic/atomic.h (lines 85 to 128)
- https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/std/__atomic/api/common.h (lines 120 to 158)
- https://en.cppreference.com/w/cpp/atomic/atomic_ref
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.5.1)

### A2. Thread scopes: "three example scopes", and "narrower scope may be faster" (pp. 207, 212 to 213)

**Verdict: PARTLY.** Block and device scope are exactly as the book uses them. The book's third scope ("across all GPU devices in a multi-GPU system") is NVIDIA's system scope, which also includes the CPU. NVIDIA now states the speed claim plainly, but compares block with system scope, not block with device scope.

- Scope table, guide section 3.2.3 (columns: C++ scope, PTX scope, description, point of coherency): "cuda::thread_scope_block | .cta | Memory operations are visible to other threads in the same thread block. | L1"; "cuda::thread_scope_device | .gpu | Memory operations are visible to other threads in the same GPU device. | L2"; "cuda::thread_scope_system | .sys | Memory operations are visible to other threads in the same system (CPU, other GPUs). | L2 + connected caches". The `.cluster` row has no C++ name in this table; libcu++ has `cuda::thread_scope_cluster` (infra A2).
- Guide section 3.2.4.1.2 "Performance Considerations": "Use the narrowest scope possible: block-scoped atomics are much faster than system-scoped atomics." "Prefer weaker orderings: use stronger orderings only when necessary for correctness." "Consider memory location: shared memory atomics are faster than global memory atomics."
- Memory model, section 5.8.4, which makes the wrong scope a bug and not only a slowdown: "The execution of a program contains a data race if it contains two potentially concurrent conflicting actions, at least one of which is not atomic at a scope that includes the thread that performed the other operation, and neither happens before the other ... Any such data race results in undefined behavior." Its worked example marks a block-scope store read by a thread of another block "UB: data race". So a commit phase written with `thread_scope_block` on the public bins (one of the notes' planted defects) is undefined behaviour, whatever numbers it happens to produce.
- Not found: any NVIDIA figure for block scope versus device scope on global memory, which is the comparison in Fig. 9.9.

- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (sections 3.2.3, 3.2.4.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html (sections 5.8.1, 5.8.4, 5.8.5)

### A3. Atomics on one location serialise in no fixed order; a plain increment loses updates (pp. 204 to 206)

**Verdict: CONFIRMED.** NVIDIA documents both halves of the book's interleaving tables, including the case where the racing threads are in the same warp.

- Atomics, guide section 3.2.2.1.1: "If an atomic instruction executed by a warp reads, modifies, and writes to the same location in global memory for more than one of the threads of the warp, each read/modify/write to that location occurs and they are all serialized, but the order in which they occur is undefined."
- Plain writes, same section: "If a non-atomic instruction executed by a warp writes to the same location in global or shared memory from more than one of the threads of the warp, the number of serialized writes that occur to that location may vary depending on the compute capability of the device. However, for all compute capabilities, which thread performs the final write is undefined." For a stepper this is the "all lanes read the old value, one store survives" case, the extreme of Fig. 9.4(b).
- Guide section 2.3.5 describes atomics with the book's lock picture: "Atomic functions allow a thread to obtain a lock on a global memory location and perform a read-modify-write operation on that location. No other thread can access the same location while the lock is held." and "Atomic functions should be used sparingly as they enforce thread synchronization that can impact performance."
- Formally a plain `++bins[b]` from two threads is a data race and therefore undefined behaviour (A2 quote). The lost update is the typical outcome, not a guaranteed one; say so in the stepper's `<Simplification>`.

- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (section 3.2.2.1.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.5)

### A4. "Modern GPUs ... perform device-scope atomics in the last-level cache", "tens of cycles rather than hundreds" (p. 210)

**Verdict: PARTLY.** That global atomics are executed at the L2 is documented by NVIDIA's profiler guide, and the scope table gives L2 as the point of coherency for device scope. No current NVIDIA document gives L2 or DRAM latencies in cycles.

- Nsight Compute "Compute Triage Guide", section 3.7.1 "Symptom: L2 atomic throughput is a bottleneck": "Global atomics (atomicAdd, atomicCAS, etc.) are serialized at the L2 atomic stage." "When many warps contend on the same or nearby addresses, L2 atomic throughput becomes the limiter." "This often manifests as moderate L2 throughput with high long-scoreboard stalls, because the SM is waiting for the atomic to complete." The metric to read is named: "lts__t_sectors_srcunit_tex_op_atom can confirm the volume".
- Guide section 2.3.3.6: "The L2 cache is located on the device and is shared among all the SMs. The size of the L2 cache can be queried with the l2CacheSize device property element".
- Not found: "tens of cycles" or any L2 latency figure. A search of the guide pages read for cycle counts found only an unrelated "FFMA→FFMA is 4 cycles" in the triage guide. Present the book's 200-cycle and "tens of cycles" numbers as the book's example only.

- https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html (section 3.7.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.3.6)

### A5. Shared memory "is private to each SM and has very short access latency (a few cycles)"; shared-memory atomics are much faster (p. 213)

**Verdict: PARTLY.** Physically on each SM and faster: confirmed. "Private to each SM" mixes up where it lives with who can see it (each block sees its own allocation; ch05 A1 already flags the book's per-SM wording). "A few cycles" is printed nowhere in current NVIDIA docs.

- Guide section 2.3.3.2: "Shared memory is a memory space that is accessible by all threads in a thread block. It is physically located on each SM and uses the same physical resource as the L1 cache, the unified data cache." "because shared memory is located on each SM, the bandwidth is higher and the latency is lower than accessing global memory." No number (`ch06-facts.md` A12 reached the same result for the book's chapter 6 wording).
- Triage guide 3.7.1: "Shared-memory atomics are served inside the SM and are significantly faster than global atomics."
- History worth one sentence: Maxwell Tuning Guide 1.4.3.3: Kepler and Fermi "implemented shared memory atomics using a lock/update/unlock pattern that could be expensive in the case of high contention"; "Maxwell improves upon this by implementing native shared memory atomic operations for 32-bit integers and native shared memory 32-bit and 64-bit compare-and-swap (CAS)". Every GPU the current toolkit supports is newer than Maxwell.

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.3.2)
- https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html (section 3.7.1)
- https://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html (section 1.4.3.3)

### A6. Privatisation per block, with a commit phase; the per-cluster variant (pp. 211 to 214)

**Verdict: CONFIRMED.** NVIDIA's profiler guide lists warp-level combining, then block-level reduction, then per-block privatisation in shared memory; the last is the book's kernel. (The book has no warp-level step: its sequence is last-level cache, per-block privatisation, coarsening, then per-thread run merging. Corrected 2026-11-20 after review; the earlier wording said "the book's remedies in the book's order".) The programming guide's distributed-shared-memory section is now a histogram example, with code, for the per-cluster variant the book describes only in words.

- Triage guide 3.7.1, "Actions (apply hierarchically)": "1. Warp level: combine values among threads of the same warp using shuffle-based reductions ... This collapses up to 32 atomics into 1 per warp." "2. Block level: if the reduction set fits in shared memory, finish the reduction there. ... Emit at most one global atomic per block." "3. Privatization: for histogram-style patterns where target indices are scattered, build per-block (or per-warp) local histograms in shared memory and merge them in a final pass."
- Guide section 2.3.3.8 "Distributed Shared Memory": "A standard way of computing histograms is to perform the computation in the shared memory of each thread block and then perform global memory atomics. A limitation of this approach is the shared memory capacity. Once the histogram bins no longer fit in the shared memory, a user needs to directly compute histograms and hence the atomics in the global memory. With distributed shared memory, CUDA provides an intermediate step, where depending on the histogram bins size, the histogram can be computed in shared memory, distributed shared memory or global memory directly."
- The example kernel `clusterHist_kernel` zeroes `extern __shared__ int smem[]`, calls `cluster.sync()`, updates with `int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank); atomicAdd(dst_smem + dst_offset, 1);` in a grid-stride loop, calls `cluster.sync()` again "to ensure all distributed shared memory operations are completed and no thread block exits while other thread blocks are still accessing distributed shared memory", then commits with `atomicAdd(&lbins[i], smem[i])`. Differences from the book worth showing: it uses the legacy `atomicAdd`, it commits every bin (no zero test), and the launch checks `cuda::arch_traits_for(...).cluster_supported` and `max_shared_memory_per_block_optin` and launches with `cudaLaunchKernelEx` plus `cudaLaunchAttributeClusterDimension`.
- "Threads that belong to a thread block cluster can read, write or perform atomics in the distributed address space, regardless of whether the address belongs to the local thread block or a remote thread block." Clusters need compute capability 9.0 or later (ch05 A6).

- https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html (section 3.7.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.3.8)

### A7. Contiguous versus interleaved partitioning is a coalescing question; "four pixels (a 32-bit word) per iteration" (pp. 214 to 217)

**Verdict: PARTLY.** The coalescing argument for interleaved partitioning is NVIDIA's current rule. The hint that each thread should read four pixels per iteration "to fully utilize the interconnect bandwidth between the caches and the SMs" is not in current NVIDIA docs.

- Guide section 2.3.4.1: "Global memory is accessed via 32-byte memory transactions. When a CUDA thread requests a word of data from global memory, the relevant warp coalesces the memory requests from all the threads in that warp into the number of memory transactions necessary to satisfy the request". "Coalesced memory access occurs provided different threads in the warp access elements from the same 32-byte segments of memory in some linear or permuted way. Stated another way, the best way to achieve coalesced memory access is to maximize the ratio of bytes used to bytes transferred." "In the worst case, 32 global memory transactions may be required to satisfy the addresses requested by a single load instruction from a single warp."
- Applied to the book (our arithmetic): with 1-byte pixels and interleaved indexing, a warp's 32 reads fall in one 32-byte segment; with contiguous indexing and `COARSE_FACTOR` of 32 or more, each lane's byte is in a different segment, the guide's worst case.
- The book's interleaved index is the grid-stride idiom NVIDIA uses itself: the A6 histogram loops `for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)`. `ch06-facts.md` A2 (NVIDIA), B1 (AMD) and C1 (Apple) own the general coalescing rules and agree with this item; this item only records what chapter 9 needs.
- Not found: any statement about per-thread load width and "interconnect bandwidth". CUB's histogram does load several samples per thread as one vector (`VecT = CubVector<SampleT, vec_size>` in `agent_histogram.cuh`), which supports the idea in practice without documenting it.

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.4.1)
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/agent/agent_histogram.cuh

### A8. The limits that decide whether a private histogram fits on chip (pp. 212 to 213, 217)

**Verdict: CONFIRMED.** 1024 threads per block still holds (ch05 A5). Per-block shared memory is 48 KB without opt-in on every listed compute capability and between 64 KB and 327 KB with opt-in.

- Compute capabilities table, row "Maximum amount of shared memory per thread block" (with opt-in): 7.5: 64 KB; 8.0: 163 KB; 8.6: 99 KB; 8.7: 163 KB; 8.9: 99 KB; 9.0, 10.0, 10.3, 11.0: 227 KB; 10.7: 327 KB; 12.x: 99 KB. The `cuda::arch_traits` names: "Maximum amount of shared memory per thread block, without opt-in (48 KB) | max_shared_memory_per_block" and "with explicit opt-in ... | max_shared_memory_per_block_optin". KB here is 1024 bytes (ch05 A4).
- How to opt in and the 10.7 extra switch: ch05 A3.
- For lessons: these are device facts. A lab decides "does it fit" from `sharedMemPerBlock` / `sharedMemPerBlockOptin` as reported by `doctor`, exactly as PyTorch does (D1). D7 does the vocabulary arithmetic.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Tables 33 and 34)

### A9. What the compiler does with a histogram's atomics: `red` versus `atom`, and automatic warp aggregation

**Verdict: PARTLY.** The PTX ISA documents a no-return reduction instruction beside the returning atomic, both with the C++ scopes. The claim that nvcc aggregates same-address atomics across a warp by itself appears only in a 2014 blog post updated in 2017, without conditions; current docs do not mention it.

- PTX ISA 9.5, section 9.7.15.6 `red`: "Reduction operations on global and shared memory." "Performs a reduction operation with operand b and the value in location a, and stores the result of the specified operation at location a, overwriting the original value." Scopes "`.scope = { .cta, .cluster, .gpu, .sys }`"; "If the .scope qualifier is absent, .gpu scope is assumed by default." Section 9.7.15.5 `atom`: "Atomic reduction operations for thread-to-thread communication" (returns the old value). This is why the book's unused return value (p. 209) can matter, and why PyTorch calls a helper named `gpuAtomicAddNoReturn` (D1). Which instruction nvcc emits for a given call is not documented; read the PTX of the lab kernel to see.
- NVIDIA blog "CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics" (published Oct 1, 2014, updated November 2017): "The NVCC compiler now performs warp aggregation for atomics automatically in many cases"; "the compiler does the optimization for post-Kepler GPUs starting with CUDA 7.5"; "in CUDA 9, it also does it for Kepler GPUs". No conditions are stated.
- Consequence for the uniform-versus-skewed lab: on NVIDIA an all-one-value input may run faster than the book's model predicts if the compiler aggregates; whether it does is not documented. Make the reader predict, then measure; do not promise either outcome. AMD's equivalent is documented in source (B5).

- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (sections 9.7.15.5, 9.7.15.6)
- https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/

### A10. A GPU-side race detector on NVIDIA: Compute Sanitizer racecheck

**Verdict: CONFIRMED,** with a limit that matters for this chapter: racecheck examines shared memory only, so the global-memory lost update of Fig. 9.6 without atomics is invisible to it. A plain `++bins_s[b]` in Fig. 9.10 is in its reach.

- "The racecheck tool is a run time shared memory data access hazard detector." "Currently, this tool only supports detecting accesses to on-chip shared memory." Invocation: `compute-sanitizer --tool racecheck [sanitizer_options] app_name [app_options]`.
- Hazard kinds it reports: Write-After-Write, Write-After-Read, Read-After-Write. "These hazards cause data races where the behavior or the output of the application depends on the order in which all parallel threads are executed by the hardware. Race conditions manifest as intermittent application failures or as failures when attempting to run a working application on a different GPU."
- "The racecheck tool does not perform any memory access error checking. It is recommended that users first run the memcheck tool".
- So on a real NVIDIA GPU the course's race backend (E1) is still the only tool that names the source line of the global-memory race.

- https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html (section "Racecheck Tool")

---

## Group B: HIP / ROCm

### B1. The chapter's atomics on HIP: names, scopes, semantics

**Verdict: PARTLY.** The book's code ports with the same spelling through libhipcxx (infra B2), and the legacy `atomicAdd(unsigned*, unsigned)` has the same relaxed, device-scope meaning as CUDA's. Two differences: no `_block` legacy functions (infra B1), and AMD's docs describe scope as if it were a memory space.

- HIP language extensions, "Atomic functions": "Atomic functions are read-modify-write (RMW) operations, whose result is visible to all other threads on the scope of the atomic operation, once the operation completes." "If multiple instructions from different devices or threads target the same memory location, the instructions are serialized in an undefined order." (the book's "no order" point, in AMD's words).
- Same page: "Atomic operations in kernels can operate on block scope (i.e. shared memory), device scope (global memory), or system scope (system memory), depending on hardware support." This equates scope with memory space; CUDA's scope is independent of the space (A2), and the book's Fig. 9.9 uses block scope on global memory. With libhipcxx the scope is a template argument as on CUDA, so the book's code still expresses it; the wording is AMD's.
- Header `amd_hip_atomic.h` at `therock-10.2`: `__device__ inline unsigned int atomicAdd(unsigned int* address, unsigned int val) { return __scoped_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE); }`. The same header defines `__HIP_MEMORY_SCOPE_SINGLETHREAD` (1), `_WAVEFRONT` (2), `_WORKGROUP` (3), `_AGENT` (4), `_SYSTEM` (5) for the scoped built-ins; I found them documented nowhere in the HIP pages read, so a lab should not use them.
- libhipcxx's `cuda::atomic_ref` page is NVIDIA's text carried over, including the `Scope = cuda::thread_scope_system` default, the different-scope rule and "Types of T, where sizeof(T) < 4, are not natively supported by the underlying hardware. For these types atomic operations are emulated and will be drastically slower."
- Warps on AMD run in lockstep, which is the within-wavefront lost update of A3: "AMD GPUs guarantee, that all warp lanes are executed in lockstep, whereas NVIDIA GPUs that support Independent Thread Scheduling might require additional synchronization".

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (sections "Atomic functions", "Warp cross-lane functions")
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_hip_atomic.h (lines 34 to 38, 163 to 165)
- https://rocm.docs.amd.com/projects/libhipcxx/en/latest/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html

### B2. Where AMD performs atomics: L2 for global memory, LDS banks for shared memory

**Verdict: CONFIRMED.** AMD documents the book's model (device atomics at the shared last-level cache, shared-memory atomics on chip, same-address updates serialised) more explicitly than NVIDIA does.

- HIP "Hardware implementation", L2 cache: "The L2 cache serves as the coherence point for all GPU memory accesses and is shared by all compute units." "Atomic operation support: Atomics execute directly in the L2 cache for coherence." Under memory coherence: "Atomic operations (executed at L2 level)".
- Local data share (AMD's shared memory): "Each bank can be accessed independently every cycle for reads, writes, and/or atomic operations." "Conflicts are resolved by serializing accesses across multiple cycles. Address conflicts (multiple threads atomically updating the same address) are similarly serialized."
- The same page gives per-cycle LDS throughput figures ("4-byte values: 8 cycles for 64 threads (50% peak bandwidth)"). Those are AMD device figures and stay out of lessons.
- No latency in cycles for L2 or LDS atomics was found.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (sections "Local data share (LDS)", "L2 cache architecture", "Memory coherence")

### B3. Coalescing on AMD (the interleaved-partitioning argument)

**Verdict: CONFIRMED.** Same rule, different granularity words (cache lines rather than 32-byte segments).

- "The coalescing hardware in the VMEM unit analyzes addresses from all threads in a warp and groups them into the minimum number of cache line requests." "When consecutive threads access consecutive memory addresses, the hardware can combine all 64 thread requests into as few as 4-8 cache line requests (depending on data size and alignment)." "When threads access widely separated addresses, each thread can generate a separate memory transaction, reducing effective bandwidth by up to 16x or more." "Align data structures to cache line boundaries (64B and/or 128B)".
- Performance-optimization page: "Memory coalescing is relevant when accessing global memory (HBM and GDDR attached to the GPU). For efficient access to LDS and shared memory, see the discussion of bank conflicts below."

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (section "Memory coalescing")
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html (section "Memory coalescing theory")

### B4. How many bins fit in LDS on AMD

**Verdict: PARTLY.** AMD publishes LDS size per product, and it differs by a factor of more than two between current parts; I found no documented per-workgroup maximum in the HIP pages read. Use `sharedMemPerBlock` from `hipGetDeviceProperties` (field spelling: ch05 B2).

- AMD's GPU specification table has an "LDS (KiB)" column. Examples read today: MI355X (CDNA4, gfx950) 160; MI300X (CDNA3, gfx942) 64; Radeon RX 7900 XTX (RDNA3, gfx1100) 128. The table does not say whether one workgroup may use all of it.
- For course arithmetic only (D7): every one of these is far below a vocabulary-sized `uint32` histogram.

- https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html

### B5. Does the AMD compiler aggregate a histogram's atomics across a wavefront?

**Verdict: CONFIRMED** (from compiler source): the AMDGPU backend has an atomic optimizer, on by default, that applies only when the address is the same for the whole wavefront. A histogram's per-pixel atomic has a data-dependent address and is not optimised; a single global counter is.

- `AMDGPUAtomicOptimizer.cpp` header comment: "This pass optimizes atomic operations by using a single lane of a wavefront to perform the atomic operation, thus reducing contention on that memory location." In the code: "If the pointer operand is divergent, then each lane is doing an atomic operation on a different address, and we cannot optimize that."
- Default: `AMDGPUOptions.td` declares `amdgpu-atomic-optimizer-strategy` with default `AMDGPU::ScanOptions::Iterative` and choices "DPP,Iterative,None".
- Caveat: this is upstream LLVM `main`. I did not check that ROCm 10.2's compiler carries the same default.

- https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUAtomicOptimizer.cpp (lines 9 to 20, 224 to 231)
- https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUOptions.td (line 203)

### B6. A documented race detector for HIP kernels

**Verdict: COULD NOT VERIFY.** I found no race-detection tool in AMD's published ROCm documentation. Work in progress exists in source only.

- In `ROCm/rocm-systems` on `develop`, the experimental emulator project `emulation/rocjitsu` gained "ConSan for native AMDGPU LDS race detection" (PR #13860, merged 2026-10-30): it "records effective LDS addresses and participating lanes, then attributes the observations to the original instruction sites", across waves of a workgroup. A follow-up "Treat LDS atomics as read-modify-write in the race detector" (PR #15229) was still open today. It covers LDS only, not global memory, and is not in the 10.2 release docs.
- For lessons: say that AMD documents no race detector today, and use the course's race backend (E1).

- https://github.com/ROCm/rocm-systems/pull/13860
- https://github.com/ROCm/rocm-systems/pull/15229

---

## Group C: Metal

### C1. The book's atomics in MSL: no `atomic_ref`, atomic-typed pointers, no scope on the update

**Verdict: PARTLY.** Every operation the chapter needs exists for `uint` in both device and threadgroup memory (infra C1). Three things in the book's design have no Metal counterpart and change the code's shape.

- No atomic view of a plain object. MSL 2.6: "Metal atomic functions (as described in section 6.16) can only use Metal atomic data types." There is no `atomic_ref` in the spec (searched). In practice the kernel declares the buffer parameter with an atomic type while the host allocates plain 32-bit integers: the spec's own example is `device atomic_int *output [[buffer(1)]]` with `atomic_fetch_add_explicit(output, val, memory_order_relaxed);`, and PyTorch's histogram binds a `kUInt32` tensor as `device atomic_uint* counts` (D2).
- The book's plain (non-atomic) read of the private bins in the commit phase becomes `atomic_load_explicit(&bins[b], memory_order_relaxed)`; zeroing becomes `atomic_store_explicit(..., 0, memory_order_relaxed)`. That is how PyTorch (D2), llama.cpp (D4) and PyTorch's radix sort (`threadgroup atomic_uint local_hist[RSIZE];`, `SortRadix.h` line 170) write it. The spec shows no example of a `threadgroup` atomic array declaration; the form above is used by shipping code.
- No scope on a read-modify-write. MSL 6.16.2: "The enumeration thread_scope denotes a set of threads for the memory order constraint ... You use it with atomic_thread_fence." The fetch-and-modify signatures take `(object, operand, memory_order)` and, from Metal 4.1, an extra `mem_flags`; none takes a scope. So the book's block-versus-device distinction is expressed only by the address space (`threadgroup` versus `device`).
- Order: MSL 6.16.4.5 still says for fetch-and-modify "The only supported value for order is memory_order_relaxed." The book uses relaxed, so nothing is lost (infra C1 records the spec's inconsistency for other functions).
- MSL 2.6: "Objects of atomic types are free from data races." The spec has no general definition of a data race for plain accesses (searched "data race", "race condition").

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 2.6, 6.16.1, 6.16.2, 6.16.4.5; reduction example in 6.10.2)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/SortRadix.h (line 170)

### C2. Counters narrower or wider than 32 bits on Metal

**Verdict: CONFIRMED** that Metal has no native 8-bit or 16-bit atomic type and no 64-bit atomic add (types: MSL 2.6, infra C1). Both PyTorch and MLX emulate them with a compare-and-swap loop on the containing 32-bit word, which is the slow path libcu++ warns about on CUDA (A1).

- PyTorch `c10/metal/atomic.h`: sub-word adds pack into `atomic<uint>` and retry `atomic_compare_exchange_weak_explicit`; 64-bit: "There are no atomic 64-bit add in Metal yet, but templates below implements a consistent add I.e. if multiple threads are modify the same 64-bit value, results stored at the address will eventually be equal to its original value plus sum of all operands", done as two 32-bit `atomic_fetch_add_explicit` calls with a carry. The same file casts plain pointers to atomic ones: `reinterpret_cast<device ::metal::atomic<uint>*>(data) + offset` (line 22), a practice the spec does not document.
- MLX `mlx/backend/metal/kernels/atomic.h`: `template <typename T, typename = void> struct mlx_atomic { atomic<uint> val; };` for every type that is not `int`, `uint`, `ulong` or `float`, updated by `mlx_atomic_update_and_store` with a CAS loop over a `uint_or_packed<T>` union (lines 160 to 200; last commit 2026-08-04).
- For the vocabulary lab (D7): 16-bit counters to save threadgroup memory are not available natively on Metal and are emulated on CUDA and HIP.

- https://github.com/pytorch/pytorch/blob/main/c10/metal/atomic.h (lines 15 to 33, 250 to 268)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/atomic.h

### C3. Threadgroup memory as the private-histogram budget on Metal

**Verdict: CONFIRMED.** 32 KB total threadgroup memory on Apple4 and later (ch05 C1), read at run time from `MTLDevice.maxThreadgroupMemoryLength` (ch05 C3). A 256-bin `uint` histogram (1 KiB) fits easily; dynamic sizing uses `setThreadgroupMemoryLength(_:index:)` with a multiple of 16 bytes (ch05 C8).

- PyTorch's MPS histogram keeps its private copy smaller than the maximum on purpose: "Keep threadgroup memory below 8 KiB so at least four local histograms can reside within the 32 KiB available on supported Apple GPUs." (`HistogramKernel.mm` lines 28 to 30). That is an occupancy argument, PyTorch's, not Apple's.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/HistogramKernel.mm (lines 27 to 35, 186 to 212)

### C4. Where Apple GPUs perform atomics, their latency, and Apple's coalescing rule

**Verdict: COULD NOT VERIFY.** Apple documents none of the three. Infra C13 found no cache, bank or bandwidth figures, `ch06-facts.md` C1 found no coalescing rule in the MSL spec, and a search of developer.apple.com today found no statement about adjacent threads and contiguous addresses (the hits were WWDC videos on threadgroup sizing and 16-byte alignment of threadgroup memory).

- For lessons: the Metal half of §9.3 and §9.5 is "Apple does not document this; measure it". The uniform-versus-skewed lab and a contiguous-versus-interleaved timing on the reader's Mac are the evidence.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- https://developer.apple.com/videos/play/tech-talks/10580/

### C5. Metal has no documented data-race detector

**Verdict: CONFIRMED.** Xcode's Shader Validation lists the GPU errors it detects, and data races are not among them.

- Apple, "Validating your app's Metal shader usage": "Metal Shader Validation detects errors only discoverable during shader execution, such as accesses to non-resident resources, out-of-bounds memory accesses, undefined behavior, and attempts to access nil textures." The table lists: resource not resident, out-of-bounds memory access, misuse of Metal Performance Primitives, undefined interpolant behavior, null resources usage, illegal address space cast, binding of incorrect texture type, binding of incorrect acceleration structure type, usage flag mismatch. Enabling: Xcode scheme, Diagnostics tab, or environment variables such as `MTL_SHADER_VALIDATION_DEFAULT_STATE`.
- For lessons: on a Mac, the planted race is found by the course's race backend (E1), not by a Metal tool.

- https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage

### C6. Apple's library histogram: `MPSImageHistogram`

**Verdict: CONFIRMED.** Metal Performance Shaders ships an image histogram filter, the library counterpart of the book's running example.

- "A filter that computes the histogram of an image." Available macOS 10.13 and later. Header (`MPSImageHistogram.h`, macOS 27.0 SDK): `numberOfHistogramEntries`: "The value stored in each histogram bin is a 32-bit unsigned integer." "numberOfHistogramEntries must be a power of 2." Related: `MPSImageNormalizedHistogram`, `MPSImageHistogramEqualization`, `MPSImageHistogramSpecification`.
- It works on `MTLTexture` images, not on a buffer of token ids, so it does not serve the vocabulary lab.

- https://developer.apple.com/documentation/metalperformanceshaders/mpsimagehistogram
- Local header: `MetalPerformanceShaders.framework/.../MPSImage.framework/Headers/MPSImageHistogram.h` (macOS 27.0 SDK)

---

## Group D: How real backends do it (llama.cpp, MLX, PyTorch, and libraries)

Read on the default branches today; dates are the last commit to the file. These are snapshots: link to files, not line numbers, in lessons.

### D1. PyTorch CUDA: `torch.histc` and `torch.bincount` choose shared or global privatisation at run time

**Verdict: CONFIRMED.** `aten/src/ATen/native/cuda/SummaryOps.cu` (2026-09-29) is the book's Fig. 9.10 with Fig. 9.6 as the fallback, chosen by the per-block shared-memory limit.

- `enum class CUDAHistogramMemoryType { SHARED, GLOBAL };` Selection: `auto maxSharedMem = getCurrentDeviceProperties()->sharedMemPerBlock; auto sharedMem = nbins * sizeof(output_t) + 8; // 8 guard bytes` then `if (sharedMem < maxSharedMem) { ... memType = CUDAHistogramMemoryType::SHARED; }` (lines 202 to 219). Comment: "SHARED: Each block atomically adds to it's own **shared** hist copy, then atomically updates the global tensor." "GLOBAL: all threads atomically update to a single **global** hist copy."
- The shared kernel zeroes `smem`, `__syncthreads()`, updates with `gpuAtomicAddNoReturn(&smem[bin], ...)`, `__syncthreads()`, then commits every bin with `gpuAtomicAddNoReturn(&a.data[aOffset], smem[i])` (no zero test). The loop is a grid-stride loop: `#define FOR_KERNEL_LOOP(i, lim) for (IndexType i = blockIdx.x * blockDim.x + threadIdx.x; i < lim; i += gridDim.x * blockDim.x)`, the book's interleaved partitioning.
- The book's §9.5 trade-off is coded as a heuristic: `#define RATIO_OF_GMEM_ATOMIC_ADD_TO_SMEM_ATOMIC_ADD 8` and a comment solving for the grid size that balances "#(smem atomicAdd per SM)" against "#(gmem atomicAdd) = grid.x * nbins" (lines 206 to 218). Fewer blocks means fewer commits, the coarsening argument.
- `bincount` without weights counts in `kLong` (8-byte) bins, so the shared path needs `nbins * 8 + 8` bytes. A float-weighted call raises `alertNotDeterministic("_bincount_cuda")` "because of floating point atomicAdd usage". The comment block says "3 implementations" but the enum has two: stale comment.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SummaryOps.cu

### D2. PyTorch MPS: the same choice in Metal

**Verdict: CONFIRMED.** `aten/src/ATen/native/mps/kernels/HistogramKernel.metal` (2026-10-13) and `operations/HistogramKernel.mm` (2026-11-11) contain a Metal version of Fig. 9.10 next to a Fig. 9.6 fallback.

- `kernel void histc_atomic_threadgroup(...)` with `threadgroup atomic_uint* local_counts [[threadgroup(0)]]`: zero with `atomic_store_explicit(&local_counts[bin], 0, memory_order_relaxed)` in a stride-`threads_per_threadgroup` loop, `threadgroup_barrier(mem_flags::mem_threadgroup)`, grid-stride update `for (uint index = tid; index < num_elements; index += total_threads)` with `atomic_fetch_add_explicit(&local_counts[bin], 1, memory_order_relaxed)`, barrier, then commit `uint value = atomic_load_explicit(&local_counts[bin], memory_order_relaxed); if (value != 0) { atomic_fetch_add_explicit(&counts[bin], value, memory_order_relaxed); }` (lines 221 to 256). The zero test is the book's.
- `kernel void histc_atomic_global(...)`: one thread per element, `atomic_fetch_add_explicit(&counts[bin], 1, memory_order_relaxed)` on `device atomic_uint* counts`.
- Host choice: `const bool use_threadgroup = num_bins <= ... max_threadgroup_memory_length / sizeof(uint)` with `max_threadgroup_memory_length = std::min(kHistcMaxThreadgroupMemoryLength, [device maxThreadgroupMemoryLength])`, `kHistcMaxThreadgroupMemoryLength = 8 * 1024`, 256 threads per threadgroup, at most 128 threadgroups ("M1 sweeps over 8, 32, 64, 128, and 256 threadgroups showed a broad performance plateau from 32 through 256"). That cap is the book's coarsening, tuned by measurement.
- `torch.bincount` on MPS (`operations/SummaryOps.mm`) calls `self.to(kLong).histc(nbins, 0, nbins)`; with weights it calls `index_add_`.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/HistogramKernel.metal
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/HistogramKernel.mm
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/SummaryOps.mm

### D3. CUB and rocPRIM device histograms: the book's whole chapter as a library

**Verdict: CONFIRMED.** CUB (`cub::DeviceHistogram::HistogramEven`, `HistogramRange`, `MultiHistogramEven`) implements shared-memory privatisation up to 256 bins, per-block global privatisation above that, and an optional run-length accumulator that is Fig. 9.15. rocPRIM (behind hipCUB) makes a similar choice with an AMD-specific twist.

- CUB `dispatch_histogram.cuh` (CCCL `main`, 2026-11-19): `// Maximum number of bins per channel for which we will use a privatized smem strategy` / `inline constexpr int max_privatized_smem_bins = 256;`. Above it, `constexpr int PrivatizedSmemBins = 0;`, which `agent_histogram.cuh` defines as "Zero indicates privatized counters to be maintained in device-accessible memory". The global pool is sized `size_t(num_thread_blocks) * (num_privatized_levels[CHANNEL] - 1) * kernel_source.CounterSize()`: the book's `bins_pool` of Fig. 9.9. (Both dispatch branches carry the same comment "Dispatch shared-privatized approach", which is wrong for the first one.)
- `agent_histogram.cuh`: `enum BlockHistogramMemoryPreference { GMEM, SMEM, BLEND }`; shared storage `CounterT histograms[NumActiveChannels][PrivatizedSmemBins + 1];` ("with 1 word of padding"). The RLE path (`IS_RLE_COMPRESS`) counts a run in `CounterT accumulator` and flushes it when the bin changes with `atomicAdd_block(privatized_histograms[ch] + bins[item], accumulator)` on compute capability 6.0 and later, plus a final flush for the last item (lines 322 to 370): Fig. 9.15, including the trailing flush, written with the legacy block-scope atomic.
- rocPRIM `device_histogram.hpp` (`ROCm/rocm-libraries`, `develop`, 2026-06-24): `const bool use_shared_mem = total_shared_bins <= shared_impl_max_bins;` with power-of-two bin counts padded by one "Prevent LDS bank conflicts"; `const bool use_private_histogram = current_target.i == target_arch::gfx942;`. When the bins do not fit, gfx942 gets one private global histogram per compute unit (`global_histogram_grid_size = num_multi_processors`), every other target updates the output directly. The value of `shared_impl_max_bins` comes from a tuning config I did not open.

- https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/dispatch/dispatch_histogram.cuh
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/agent/agent_histogram.cuh
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_histogram.cuh
- https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocprim/rocprim/include/rocprim/device/device_histogram.hpp

### D4. llama.cpp: repetition penalties count on the CPU; GPU histograms appear in top-k and in counting

**Verdict: CONFIRMED.** llama.cpp does not histogram token ids on the GPU for its penalties. It keeps counts incrementally on the host and, when sampling runs on the backend, uploads them as (token, count) pairs.

- `src/llama-sampler.cpp` (2026-11-12): `struct llama_sampler_penalties` holds `std::unordered_map<llama_token, int> token_count;` over a window of `penalty_last_n` tokens; each accepted token does `ctx->token_count[token]++;` and the token leaving the window is decremented and erased at zero. Apply: `cur_p->data[i].logit -= float(count) * ctx->penalty_freq + float(count > 0) * ctx->penalty_present;` after the sign-aware repeat penalty.
- Backend path (`llama_sampler_penalties_backend_apply` / `_backend_set_input`): the host fills `host_token_ids` and `host_counts` from the map, sorts them ("Sorting is required because backend_apply uses ggml_set_rows (a scatter-back operation)"), and the graph scatters the counts into a vocabulary-sized row with `ggml_set_rows` and gathers with `ggml_get_rows`. Design point for the milestone: during generation the counts change by one token per step, so an incremental update avoids recounting; a full GPU histogram matters when a whole prompt or batch must be counted at once (D6).
- One-bin histogram on CUDA, `ggml/src/ggml-cuda/count-equal.cu` (2026-10-13): each thread counts in a register over a stride-`WARP_SIZE` loop, `nequal = warp_reduce_sum(nequal);`, then lane 0 does `atomicAdd((int *) dst, nequal);`. Register privatisation, warp reduction, one atomic per warp. The destination is an `int64_t` updated through an `int*` cast, which relies on little-endian layout and a zeroed counter below 2^31 (our reading).
- Same on Metal, `kernels/misc.metal` (2026-10-27) `kernel_count_equal`: `simd_sum`, threadgroup partials, then `atomic_fetch_add_explicit(dst, (int32_t) total, memory_order_relaxed)` on `device atomic_int * dst`.
- A 256-bin threadgroup histogram for top-k, `kernels/argsort.metal` `kernel_top_k_f32_i32` (2026-11-08): `threadgroup atomic_uint * histo [[threadgroup(0)]]`, zeroed with `atomic_store_explicit`, filled with `atomic_fetch_add_explicit(&histo[(key >> uint(shift)) & 0xFFu], 1u, memory_order_relaxed)` for each 8-bit digit of a float-to-uint key, then scanned to find the bucket holding the k-th value. This is the "bucketed selection for top-k" in the notes' LLM mapping, shipping today.

- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/count-equal.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/misc.metal
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/argsort.metal

### D5. MLX, mlx-lm and PyTorch: the lost update is documented API behaviour for indexed assignment

**Verdict: CONFIRMED.** Both array libraries document that `x[idx] += 1` with repeated indices applies once, and provide a separate accumulating scatter. That is the chapter's race and its cure at the Python level, and mlx-lm uses one for presence and the other for frequency.

- MLX docs, `array.at`: "Regular in-place updates map to assignment." With `a = mx.array([0, 0])` and `idx = mx.array([0, 1, 0, 1])`, `a[idx] += 1` gives `array([1, 1], dtype=int32)`, while `a.at[idx].add(1)` gives `array([2, 2], dtype=int32)`: "Using x.at[idx].add(y) will correctly apply all updates to all indices." Methods: add, subtract, multiply, divide, maximum, minimum.
- PyTorch `Tensor.index_put_` (docs 2.15): "If accumulate is True, the elements in values are added to self. If accumulate is False, the behavior is undefined if indices contain duplicate elements." `torch.bincount`: "Count the frequency of each value in an array of non-negative ints." "The number of bins (size 1) is one larger than the largest value in input unless input is empty".
- mlx-lm `mlx_lm/sample_utils.py` (2026-09-24): presence penalty `logits[:, tokens] -= penalty` (line 329: duplicates collapse, which is what "presence" wants); frequency penalty `logits = logits.at[:, tokens].subtract(penalty)` (line 358: every occurrence counts). Default window `context_size: int = 20`. No explicit histogram is built.
- MLX's Metal scatter relies on `mlx_atomic` (C2) for types Metal lacks.

- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.at.html
- https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.index_put_.html
- https://docs.pytorch.org/docs/2.15/generated/torch.bincount.html
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py

### D6. vLLM (not in the brief's list; added because it is the GPU-batch case of milestone M5)

**Verdict: CONFIRMED.** vLLM counts tokens for penalties on the GPU into vocabulary-sized bins in global memory, with plain global atomics and no privatisation.

- `vllm/v1/worker/gpu/sample/penalties.py` (2026-10-29): `self.output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32, device=self.device)` with the comment "This tensor is rarely used but can be very large, taking up GBs of GPU memory." Prompt presence is a bitmask, `cdiv(vocab_size, 32)` `int32` words per request.
- Triton `_bincount_kernel`: prompt tokens set bits with `tl.atomic_or(prompt_bin_mask_ptr + ... + idx, bit, mask=mask)` where `idx = prompt_tokens // 32`; output tokens do `tl.atomic_add(output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + output_tokens, 1, mask=mask)`.
- The older helper `vllm/model_executor/layers/utils.py` (2026-11-16) `get_token_bin_counts_and_mask`: `bin_counts = torch.zeros((num_seqs, vocab_size + 1), dtype=torch.long, ...)`; `bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))`, where the extra bin absorbs padding tokens ("vocab_size + 1 for padding").

- https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/penalties.py
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/utils.py

### D7. Vocabulary-sized bins versus on-chip memory (for the M5 lab's "which privatisation still applies")

**Verdict: CONFIRMED** (sizes read today; the arithmetic is ours).

- `vocab_size` in public model configs: `openai-community/gpt2` 50,257; `Qwen/Qwen3-8B` 151,936; `openai/gpt-oss-20b` 201,088.
- Bytes for one private copy (1 KiB = 1024 bytes): `uint32` counts: 196.3 KiB, 593.5 KiB, 785.5 KiB. `uint16` counts: 98.2 KiB, 296.75 KiB, 392.75 KiB. `int64` (PyTorch `bincount`, D1): 392.6 KiB, 1,187 KiB, 1,571 KiB.
- Documented per-block on-chip limits: CUDA 48 KB without opt-in, 64 to 327 KB with it (A8); Metal 32 KB (C3); AMD LDS 64 to 160 KiB on the parts in B4.
- Consequences to teach (ours): a GPT-2-sized `uint32` histogram fits only behind the opt-in on compute capabilities 9.0, 10.x and 11.0 (not 8.x or 12.x, whose limits are 163 KB and 99 KB); a current vocabulary does not fit in any documented per-block limit, even as `uint16` (only 10.7's 327 KB holds Qwen3's 296.75 KiB, and 16-bit atomics are emulated: A1, C2). What still applies: per-block (or per-CU, D3) private copies in global memory, which cost blocks × vocabulary × 4 bytes and a commit pass over the whole vocabulary; per-cluster distributed shared memory on CUDA 9.0 and later (three 227 KB blocks hold 681 KB) and on HIP devices that report `clusterLaunch` (infra B4), none on Metal; thread-level privatisation of the most recent token (Fig. 9.15), which needs no shared memory at all; and not recounting (D4).
- `vocab_size` can exceed the tokenizer's real token count (padding rows); count against the config value.

- https://huggingface.co/openai-community/gpt2/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
- https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json

---

## Group E: The race-hunting lab's tool

### E1. ThreadSanitizer, as used by the `race` backend

**Verdict: CONFIRMED.** Clang documents what the lab relies on: it reports data races with file and line for each access, it runs on macOS arm64, and it supports C++11 threading with libc++ (so `std::atomic` operations count as synchronisation).

- "ThreadSanitizer is a tool that detects data races. It consists of a compiler instrumentation module and a run-time library. Typical slowdown introduced by ThreadSanitizer is about 5x-15x. Typical memory overhead introduced by ThreadSanitizer is about 5x-10x."
- Supported platforms include "Darwin arm64, x86_64" and "Linux aarch64, x86_64, powerpc64, powerpc64le". "Support for 32-bit platforms is problematic and is not planned."
- "Simply compile and link your program with -fsanitize=thread. To get a reasonable performance add -O1 or higher. Use -g to get file names and line numbers in the warning messages." The sample report: "WARNING: ThreadSanitizer: data race (pid=19219) Write of size 4 at 0x7fcf47b21bc0 by thread T1: #0 Thread1 tiny_race.c:4 ... Previous write of size 4 at 0x7fcf47b21bc0 by main thread: #0 main tiny_race.c:10". So a report names both racing lines, which is what the lab asks the reader to find.
- "ThreadSanitizer is in beta stage. It is known to work on large C++ programs using pthreads ... C++11 threading is supported with llvm libc++." "ThreadSanitizer generally requires all code to be compiled with -fsanitize=thread. If some code ... is not compiled with the flag, TSan may fail to detect races or may report false positives."
- What it does not prove (in the project's own README, consistent with the above): only schedules that happened are examined, so a clean run is not a proof. It runs CPU threads; it says nothing about GPU timing (A3, B1 describe the GPU behaviour).
- GPU-side counterparts for comparison: A10 (NVIDIA, shared memory only), B6 (AMD, none documented), C5 (Apple, none).

- https://clang.llvm.org/docs/ThreadSanitizer.html

---

## Book items with no vendor fact to check

Recorded so the wrap page's errata list does not wait on this file:

- The §9.3 memory system is hypothetical. Its arithmetic holds: 8 B × 2 × 10^9 × 8 channels = 128 GB/s; 1 GHz / 400 cycles = 2.5 M atomics/s; × 256 bins = 640 M/s.
- Leftovers of the older text-histogram example (notes, "Unclear in the PDF"): "In our text histogram example" (p. 211) and "the four threads access characters 8 through 15" (p. 216) against eight threads and pixels in Fig. 9.13. Book-internal errata; no current document bears on them.
- "In Fig. 9.11 ... `COARSE_FACTOR` times `blockDim.x`" (p. 215) describes the code of Fig. 9.12; the "line 14" reference on p. 213; exercise 4's 1 ns versus 2 ns reading. Book-internal.

## Answers to the notes' portability questions

- `<cuda/atomic>` on HIP: yes, same spelling, through libhipcxx from ROCm 10.1 (infra B2; B1). On Metal: no reference type; atomic-typed pointers, `atomic_fetch_add_explicit` on `atomic_uint` with `memory_order_relaxed`, device and threadgroup (C1, infra C1).
- Scope on HIP: through libhipcxx the same template argument; legacy functions only `_system` (B1). On Metal: no scope on read-modify-write, only on fences (C1).
- Last-level-cache atomics: NVIDIA, L2 (A4); AMD, L2 (B2); Apple, not documented (C4).
- `__shared__` / `__syncthreads()`: HIP same spelling (ch05 B4); Metal `threadgroup` and `threadgroup_barrier(mem_flags::mem_threadgroup)` (ch05 C1, C2).
- Clusters and distributed shared memory: CUDA 9.0 and later (ch05 A6, A6 here); HIP partial (infra B4); Metal none.
- Limits: 1024 threads per block (ch05 A5, ch03 B3, ch05 C3); on-chip histogram budget (A8, B4, C3, D7).
- Coalescing: NVIDIA (A7), AMD (B3), Apple not documented (C4).

---

## Summary of corrections

Every PARTLY and COULD NOT VERIFY item, one line each:

- A1: `cuda::atomic_ref`'s scope parameter defaults to `cuda::thread_scope_system`, and `fetch_add`'s order defaults to `seq_cst`; the book's printed declaration and signature show neither. `++ref` is atomic but `seq_cst`. The constructor is `explicit` and takes `T&` (NVIDIA's own guide example in 2.3.5.1 passes a pointer, which does not match).
- A2: the book's "across all GPU devices" scope is NVIDIA's system scope, which includes the CPU. NVIDIA's speed statement compares block with system scope ("much faster"); no block-versus-device figure is documented. Mixing scopes on one object is a data race and undefined behaviour, so a block-scope commit to the public bins is a correctness bug.
- A4: device atomics at the L2 is documented (Nsight Compute triage guide, scope table); "tens of cycles" and any L2 or DRAM latency in cycles are not.
- A5: shared memory is physically on the SM but allocated and visible per block; "a few cycles" is in no current NVIDIA document.
- A7: the coalescing rule behind interleaved partitioning is current (32-byte transactions per warp); the "four pixels (a 32-bit word) per iteration for interconnect bandwidth" hint is not documented.
- A9: PTX has a no-return `red` beside `atom`; automatic warp aggregation by nvcc is stated only in a 2017-updated blog, "in many cases", with no conditions.
- B1: HIP keeps the book's spelling through libhipcxx, but has no `_block` legacy atomics, and its docs describe block scope as "shared memory" and device scope as "global memory", which is not CUDA's meaning.
- B4: AMD publishes LDS per product (64, 128, 160 KiB on the parts read) but no per-workgroup maximum in the pages read; query `sharedMemPerBlock`.
- B6 (could not verify): no race detector in AMD's published docs; an experimental LDS-only detector exists in `ROCm/rocm-systems` `develop` (rocjitsu ConSan).
- C1: Metal has no `atomic_ref`; buffers must be declared with atomic types, plain reads become `atomic_load_explicit`, and read-modify-write functions take no scope.
- C4 (could not verify): Apple documents neither where atomics execute, nor their latency, nor a coalescing rule.

Path and status facts worth knowing even where the verdict is CONFIRMED:

- A6: the CUDA guide's distributed-shared-memory section is now a histogram example with code (`clusterHist_kernel`), the book's per-cluster variant made concrete.
- A10: `compute-sanitizer --tool racecheck` sees shared memory only; the global lost update of Fig. 9.6 is invisible to it.
- B5: the AMDGPU backend aggregates atomics only for wavefront-uniform addresses; a histogram's atomics are not aggregated.
- C2: Metal has no 8-, 16-bit or 64-bit atomic add; PyTorch and MLX emulate them with CAS loops on 32-bit words.
- D1, D2, D3: PyTorch (CUDA and MPS), CUB and rocPRIM all choose between a shared-memory private copy and a global fallback by bin count; CUB's run-length path is Fig. 9.15.
- D4: llama.cpp counts penalty tokens incrementally on the host, not with a GPU histogram; its Metal top-k uses a 256-bin threadgroup histogram.
- D5: `x[idx] += 1` with duplicate indices is documented to apply once in MLX and to be undefined in PyTorch's `index_put_`; `.at[idx].add` / `accumulate=True` is the atomic scatter.

## Where the backends really differ

One line per cell. Nothing here was run on hardware by us.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Atomic view of a plain array (the book's design) | `cuda::atomic_ref<T, Scope> r(x[i]);` from `<cuda/atomic>` | Same, through libhipcxx (ROCm 10.1 and later) | None; declare the parameter `device atomic_uint*` or `threadgroup atomic_uint*` |
| Atomic add on a 32-bit counter | `r.fetch_add(v, cuda::memory_order_relaxed)`; legacy `atomicAdd` | Same; legacy `atomicAdd` | `atomic_fetch_add_explicit(p, v, memory_order_relaxed)` |
| Order if you omit it | `seq_cst` (`fetch_add`, `++r`); legacy functions are relaxed | Same | Must be passed; fetch-and-modify accepts relaxed only |
| Scope on the update | `thread_scope_block`, `_cluster`, `_device`, `_system`; default system | Same names without cluster; legacy `_system` only, no `_block` | None on read-modify-write; scope exists only on fences |
| Plain read of a private bin after the barrier | Plain load is fine once no `atomic_ref` is live | Same | `atomic_load_explicit(..., memory_order_relaxed)` |
| Counters narrower than 32 bits | Emulated, "drastically slower" (libcu++) | Same text in libhipcxx | No type; projects emulate with CAS on a 32-bit word |
| Where global atomics execute | L2 (profiler guide; device scope's point of coherency is L2) | L2 ("Atomics execute directly in the L2 cache") | Not documented |
| Shared-memory atomics | Native for 32-bit integers since Maxwell; "served inside the SM" | Per LDS bank; same-address updates serialised | `threadgroup` atomics since Metal 2; nothing on speed |
| Per-block on-chip budget for private bins | 48 KB default; 64 to 327 KB with opt-in by compute capability | LDS per product (64 to 160 KiB on parts read); query | 32 KB total threadgroup memory (Apple4 and later) |
| Private copy per cluster (distributed shared memory) | Compute capability 9.0 and later; guide's histogram example | `cluster_group`, `map_shared_rank` on devices reporting `clusterLaunch` | None |
| Compiler aggregation of same-address atomics | "In many cases" (2017 blog); conditions undocumented | LLVM atomic optimizer, wavefront-uniform addresses only, on by default | Not documented |
| Coalescing rule for interleaved partitioning | 32-byte transactions per warp | Consecutive lanes merged into 64 B or 128 B cache-line requests | Not documented |
| Lost update inside one warp | "which thread performs the final write is undefined" | Lanes run in lockstep | Not documented |
| GPU-side race detector | `compute-sanitizer --tool racecheck`, shared memory only | None documented (experimental rocjitsu in `develop`) | None (Shader Validation lists no race check) |
| Library histogram | CUB `DeviceHistogram`: shared up to 256 bins, else per-block global copies; optional run-length path | hipCUB / rocPRIM `histogram_even`: shared up to a tuned limit; per-CU global copies on gfx942 only | `MPSImageHistogram` on textures, power-of-two bins, `uint32` counts |
| Framework histogram today | PyTorch `SummaryOps.cu`: shared if `nbins` fits `sharedMemPerBlock`, else global | Same PyTorch source compiled for ROCm (not checked separately) | PyTorch `HistogramKernel.metal`: threadgroup if bins fit 8 KiB, else global |
