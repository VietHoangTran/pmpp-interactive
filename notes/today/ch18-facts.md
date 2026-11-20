# Chapter 18 fact check: graph traversal, frontiers, privatised queues, grid-wide barriers (CUDA, HIP, Metal, real backends)

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md` and `notes/today/infra-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY, and judges the claim in the item title (the book's statement, quoted with its page from `notes/ch18.md`, or the assumption a course author would bring). **Quote** is copied from the source with line breaks removed.
- "infra A4", "ch11 A6", "ch05 B2" point at items in `infra-facts.md`, `ch11-facts.md`, `ch05-facts.md` and so on. Anything settled there is cited, not re-checked (a few were re-read today where this chapter leans on them; those say so).
- Versions the sources reported about themselves on the check date: CUDA Programming Guide and Runtime API "v13.5" (pages "Last updated on Oct 31, 2026"); CCCL `main` (commit `82333df`, read today); ROCm Core SDK 10.2.0 = tag `therock-10.2` of `ROCm/rocm-systems` and `ROCm/libhipcxx` (HIP docs "HIP 7.17.0"); HIPIFY docs for ROCm Core SDK 10.2.0; AMD Data Science docs 26.10 (hipGRAPH and rocGRAPH pages still under 26.03); Metal Shading Language Specification "Version 4.1", dated 2026-06-04 ("MSL spec"); local Metal compiler "Apple metal version 32023.921" and macOS SDK 27.0; NVIDIA cuGraph docs "27.02" (`VERSION` on `main` is `27.02.00`); SuiteSparse:GraphBLAS README "VERSION 10.5.4, Nov 20, 2026"; GraphBLAS C API 2.1.0; llama.cpp `master`, PyTorch docs 2.15, MLX docs 0.32.4, all read today. The GitHub API was rate-limited during the session, so last-commit dates for project files are not given; every project file was read as a raw file on its default branch today.
- Not checked: anything by running code on a GPU. Two things were run on this machine and say so: one MSL kernel was compiled (not run) with the local toolchain (C4, C5), and one Python script enumerated thread interleavings (E1). Neither is in the repository.

Totals: 21 CONFIRMED, 1 WRONG, 9 PARTLY, 1 COULD NOT VERIFY (32 items: A1 to A13, B1 to B6, C1 to C6, D1 to D6, E1). The WRONG, PARTLY and COULD NOT VERIFY items, and every failed or unfound sub-claim, are collected under "Summary of corrections".

---

## Already settled elsewhere (cite, do not re-check)

| Chapter 18 needs | Settled in |
|---|---|
| `cuda::atomic_ref<T, Scope>`, `fetch_add`, `cuda::memory_order_relaxed`; defaults are system scope and `seq_cst`; the constructor takes a reference | infra A2, ch09 A1 |
| `thread_scope_block` for the shared-memory counter, `thread_scope_device` for the global one; a narrower scope on an object another block touches is a data race | ch09 A2 |
| An object referenced by a live `atomic_ref` must be accessed only through `atomic_ref`s | ch09 A1 (cppreference quote) |
| A data race is undefined behaviour in CUDA's memory model, idempotent or not (the benign race of pp. 433 to 434); HIP documents no rule of its own | ch10 A1, ch14 A9, ch09 A2; ch14 B7 |
| Atomics on one address serialise in no fixed order (why Fig. 18.16's public frontier is grouped by block but in no block order) | ch09 A3, B1 |
| Per-block privatisation with a commit phase; global atomics at L2; shared-memory atomics faster | ch09 A4 to A6; AMD ch09 B2 |
| The warp-aggregated append (ballot, popcount, one atomic, broadcast) on CUDA, HIP (64-bit masks) and Metal (`simd_*`); compilers' automatic aggregation | ch12 A1 to A7, B1 to B4, C1, C2 |
| libhipcxx as HIP's `<cuda/atomic>` (ROCm 10.1 and later; link `libhipcxx::libhipcxx`); no `_block` legacy atomics on HIP | infra B2, B1 |
| MSL atomics: atomic-typed pointers, `atomic_fetch_add_explicit(..., memory_order_relaxed)`, `threadgroup atomic_uint`; no `atomic_ref`; no scope on read-modify-write; acquire/release only in the Metal 4.1 `mem_flags` overloads | infra C1, ch09 C1, ch11 C4 |
| Cooperative-groups header and namespace; HIP header `<hip/hip_cooperative_groups.h>`; `cudaLaunchCooperativeKernel` signatures; HIP's launch has no default arguments; multi-device cooperative launch removed in CUDA 13 | infra A4, B4; ch04 A7, B11 |
| Occupancy query `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and its HIP twin | infra A9, B9; ch04 A14, B10 |
| `cudaDeviceProp` field `multiProcessorCount` (capital P); on AMD it counts CUs, or WGPs in WGP mode | ch05 A1, A2, B2 |
| Blocks run in any order and are admitted as resources free up | ch04 A5, B2 |
| Forward progress: CUDA documents it (including "Cooperative Grid"); AMD documents none for workgroups; Metal documents none between threadgroups | ch11 A6, B3, C5; ch12 A10, B6, C4 |
| No grid-wide barrier, cooperative launch or cluster on Metal | ch04 C3, ch10 A8 |
| Metal exposes no core count and no occupancy query | ch04 C1, C5 |
| Metal threadgroup memory: 32 KB total on Apple4 and later, read `maxThreadgroupMemoryLength`; `threadgroup_barrier` | ch05 C1, C3, C2 |
| Device memory is threadgroup-coherent by default within one Metal dispatch; `coherent(device)` | ch11 C3 |
| `__shared__`, `__syncthreads()` keep their spelling on HIP | ch05 B4, ch04 A6 |
| Warp width is never a literal: 32 on NVIDIA, 64 or 32 on AMD, read on Metal | ch05 B5, ch04 C2 |
| GPU race detectors (racecheck sees shared memory only; none for AMD or Apple); ThreadSanitizer as the `race` backend | ch09 A10, B6, C5, E1 |
| llama.cpp's cooperative softmax for vocabulary-sized rows | ch10 D1 (re-read today in D2) |
| `torch.nonzero` synchronises with the host; MLX has no data-dependent shapes | ch12 D4 to D6 |
| Library stream compaction (CUB `DeviceSelect`, hipCUB, MPSGraph `nonZeroIndices`) | ch12 A8, B5, C5 |

---

## Group A: CUDA, and the book's dated statements

### A1. `cudaLaunchCooperativeKernel((void*)bfs_kernel, numBlocks, numThreadsPerBlock, kernelArgs)` with four arguments (p. 446)

**Verdict: CONFIRMED.** The four-argument call resolves to the C++ overload, whose last two parameters have defaults, and NVIDIA's own example uses the same shape.

- C++ overload (runtime API, "C++ API Routines"): `template<class T> __host__ cudaError_t cudaLaunchCooperativeKernel(T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem = 0, cudaStream_t stream = 0)`. The C API form has no defaults: `__host__ cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)`.
- Guide example (section 5.7.3.6.2.1), as printed: `void *kernelArgs[] = { /* add kernel args */ };` ... `cudaLaunchCooperativeKernel((void*)my_kernel, dimGrid, dimBlock, kernelArgs);`. The book's cast and argument array are this pattern.
- "If the kernel has N parameters the args should point to array of N pointers. Each pointer, from args[0] to args[N - 1], point to the region of memory from which the actual parameter will be copied." So `(void*)&numPrevFrontier` (a by-value `unsigned int`) is right: it points at the host copy of the value.
- "For templated functions, pass the function symbol as follows: func_name<template_arg_0,…,template_arg_N>".
- "cudaLaunchCooperativeKernel ensures that the kernel launch is atomic, i.e. if the API call succeeds, then the provided number of thread blocks will launch on the specified device." (guide section 4.4.8.1)
- "The kernel cannot make use of CUDA dynamic parallelism."
- The notes' "Unclear" item about the argument list is settled: four arguments are correct on CUDA. On HIP they are not (B1).

- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html (section 5.7.3.6.2.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html (section 4.4.8.1)

### A2. Sizing the grid as `numSMs * numBlocksPerSM` from `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, bfs_kernel, numThreadsPerBlock, 0)` (pp. 445 to 446)

**Verdict: CONFIRMED.** It is NVIDIA's own recipe, and the runtime defines the cooperative limit as exactly this product.

- Guide section 5.7.3.6.2.1: "To guarantee co-residency of the thread blocks on the GPU, the number of blocks launched needs to be carefully considered." Two printed recipes: one block per SM (`cudaLaunchCooperativeKernel((void*)my_kernel, deviceProp.multiProcessorCount, numThreads, args);`), or "maximize the exposed parallelism by calculating how many blocks can fit simultaneously per-SM using the occupancy calculator": `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, my_kernel, numThreads, 0);` then `dim3 dimGrid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);`. The book's code is the second recipe almost token for token.
- Runtime reference, on the launch: "The total number of blocks launched cannot exceed the maximum number of blocks per multiprocessor as returned by cudaOccupancyMaxActiveBlocksPerMultiprocessor (or cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors as specified by the device attribute cudaDevAttrMultiProcessorCount."
- The fourth argument: "dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes". Fig. 18.17's private frontier is a static `__shared__` array, which belongs to the compiled kernel, so 0 is correct (our reading; a kernel that used `extern __shared__` would pass its byte count here).
- Signatures and the HIP twin: infra A9, B9. Field spelling: ch05 A1.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html (section 5.7.3.6.2.1)
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html

### A3. "A grid-wide barrier would deadlock" unless every block is resident, so launch no more blocks than can run at once (p. 445)

**Verdict: CONFIRMED.** NVIDIA states the same residency requirement. What the book does not say: under the cooperative API a grid that is too large is refused at launch with an error code, so the deadlock is what you would get only by bypassing that API.

- Error code (runtime types): `cudaErrorCooperativeLaunchTooLarge = 720`: "This error indicates that the number of blocks launched per grid for a kernel that was launched via either cudaLaunchCooperativeKernel exceeds the maximum number of blocks as allowed by cudaOccupancyMaxActiveBlocksPerMultiprocessor or cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as specified by the device attribute cudaDevAttrMultiProcessorCount."
- `grid_group` (guide 5.7.3.1.3): "APIs other than sync() are available at all times, but to be able to synchronize across the grid, you need to use the cooperative launch API." `bool is_valid() const: Returns whether the grid_group can synchronize`.
- Device support: `cudaDevAttrCooperativeLaunch = 95`, "Device supports launching cooperative kernels via cudaLaunchCooperativeKernel." Guide: "Only devices with compute capability of 6.0 and higher are supported. In addition, you need to be running on either of these: The Linux platform without MPS; The Linux platform with MPS and on a device with compute capability 7.0 or higher; The latest Windows platform".
- Progress: under a cooperative launch, "all device threads in its grid shall eventually make progress" once one has (ch11 A6). That is the documented guarantee the book's deadlock argument is the informal version of.
- Not documented in the pages read: what `grid.sync()` does in a kernel launched with `<<<>>>`. Teach `is_valid()` as the check, not a description of the failure.

- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html (sections 5.7.3.1.3, 5.7.3.6.2.1)

### A4. `grid_group grid = this_grid();`, `grid.sync()`, `grid.thread_rank()`, `grid.num_threads()` (Fig. 18.17, pp. 446 to 448)

**Verdict: CONFIRMED.** All four exist with these names. The return types are 64-bit.

Guide section 5.7.3.1.3, `class grid_group`, public members as printed: `bool is_valid() const`; `void sync() const: Synchronize the threads named in the group, equivalent to g.barrier_wait(g.barrier_arrive())`; `grid_group::arrival_token barrier_arrive()`; `void barrier_wait(grid_group::arrival_token&& t)`; `static unsigned long long thread_rank(): Rank of the calling thread within [0, num_threads)`; `static unsigned long long block_rank()`; `static unsigned long long num_threads(): Total number of threads in the group`; `static unsigned long long num_blocks()`; `static dim3 dim_blocks()`; `static dim3 block_index()`. Legacy aliases: `static unsigned long long size()` "(alias of num_threads())", `static dim3 group_dim()`.

- The book's `unsigned int i = grid.thread_rank()` narrows a 64-bit value; harmless for any grid sized by A2, but say so. HIP returns 32-bit (B2).
- The split barrier (`barrier_arrive` / `barrier_wait`) is documented for the grid as well; it is the tool for doing independent work between "I am done appending" and "everyone is done" (not used by the book).
- No separate-compilation (`-rdc`) requirement for `grid.sync()` appears in the current guide pages read; the only stated requirement is the cooperative launch. Which toolkit dropped an older requirement was not checked.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html (section 5.7.3.1.3)

### A5. The three-barrier frontier swap relies on `grid.sync()` making every earlier write visible; data can stay on chip across it (pp. 447 to 448)

**Verdict: CONFIRMED.**

- Guide section 4.4.5.1 "Sync", for every group type: "Like __syncthreads(), the sync() function makes the following guarantees: All memory accesses (e.g., reads and writes) made by threads in the group before the synchronization point are visible to all threads in the group after the synchronization point. All threads in the group reach the synchronization point before any thread is allowed to proceed beyond it." That covers the plain read `numPrevFrontier = *numCurrFrontier` after the first barrier and the plain reset `*numCurrFrontier = 0` before the third.
- The p. 448 benefit (state survives in shared memory and registers), in NVIDIA's words, section 5.7.3.6.2.1: "The kernel boundary carries with it an implicit invalidation of state, and with it, potential performance implications." and "the ability to provide global inter thread block synchronization would allow the application to be restructured to have persistent thread blocks, which are able to synchronize on the device when a given stage is complete."
- Caveat for the corrected code (ch09 A1 rule): if a `cuda::atomic_ref` to `*numCurrFrontier` is declared at kernel scope, it is alive during the plain read and the plain reset, which the `atomic_ref` exclusivity rule forbids. Declare each reference in the narrowest block (as Fig. 18.12 does), or do the read and the reset through the reference.

- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html (section 4.4.5.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html (section 5.7.3.6.2.1)

### A6. `compare_exchange_strong(unvisited, currLevel, relaxed, relaxed)`: the expected value must be an lvalue; the return says whether the swap happened (Fig. 18.14, pp. 441 to 442)

**Verdict: CONFIRMED.** And the notes' unverified bug-hunt idea (reusing a stale `unvisited`) is real C++ semantics.

- libcu++ header (`cuda/std/__atomic/api/common.h`, `main`): `bool compare_exchange_strong(remove_cv_t<_Tp>& __e, remove_cv_t<_Tp> __d, memory_order __s, memory_order __f)`, and a one-order overload with `memory_order __m = memory_order_seq_cst`. The expected value is a non-const lvalue reference, so `UNVISITED` (a constant) cannot be passed, which is the book's reason for the local `unvisited`.
- cppreference, `std::atomic_ref::compare_exchange_*` (libcu++ defers to the standard, infra A2): "Atomically compares the value representation of the referenced object with that of expected, and if those are bitwise-equal, replaces the former with desired (performs a read-modify-write operation). Otherwise, loads the actual value stored in the referenced object into expected (performs a load operation)." So after a failed call the variable holds the winner's level; reusing it for the next neighbour without resetting it to `UNVISITED` compares against the wrong value. The book's function declares it fresh on each call, which is correct.
- "If failure is not std::memory_order_relaxed, std::memory_order_consume, std::memory_order_acquire or std::memory_order_seq_cst, the behavior is undefined." Relaxed and relaxed is valid.
- "The weak forms (1,2) of the functions are allowed to fail spuriously, that is, act as if *this != expected even if they are equal." The book's choice of the strong form matters: a spurious failure would leave a vertex unlabelled at its correct level. HIP and Metal do not give the strong form in practice (B4, C4).

- https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/std/__atomic/api/common.h (lines 60 to 103)
- https://en.cppreference.com/w/cpp/atomic/atomic_ref/compare_exchange

### A7. Benign races "violate the C++ memory model"; the frontier kernels fix this with atomics (pp. 433 to 434, 440 to 444)

**Verdict: PARTLY.** The caveat on Figs. 18.6, 18.8 and 18.10 is exactly NVIDIA's rule (cited above: ch10 A1, ch14 A9). But the book's own Fig. 18.15 line 29 (and Fig. 18.17 line 35), `numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;`, is a plain store to an object other threads of the block are updating with `fetch_add` at the same time. Under the definition quoted in ch09 A2 ("two potentially concurrent conflicting actions, at least one of which is not atomic ... Any such data race results in undefined behavior") that is a data race, and while those threads' `atomic_ref`s are alive it also breaks the exclusivity rule (ch09 A1). The book presents the line as intended ("restores the value of the private frontier counter", p. 444) and does not discuss it.

- Fix: store through the reference (`numCurrFrontier_s_ref.store(PRIVATE_FRONTIER_CAPACITY, cuda::memory_order_relaxed)`), or drop the store and clamp after the barrier (`min(numCurrFrontier_s, PRIVATE_FRONTIER_CAPACITY)`). E1 checks that the first fix keeps the arithmetic right.
- Consequence for the lab (our inference, not run): the `race` backend instruments plain and atomic accesses separately (ch09 E1), so the book's line 29 should be reported there as a race whenever two threads of a block overflow together. That makes it a natural planted defect.

### A8. "We can assume that the coalescing of atomic operations is applied by the compiler" (p. 442, citing Chapter 12)

**Verdict: PARTLY** (cited, not re-checked). NVIDIA states automatic warp aggregation only in a blog post last updated in 2017, with no conditions (ch12 A7, ch09 A9). AMD's LLVM backend does it in source, on by default at `-O1` and above, for wavefront-uniform addresses, which includes the frontier counter (ch12 B4). Apple documents nothing. The lesson can say "may"; the lab measures. cuGraph does not rely on the compiler and aggregates by hand (D1).

### A9. `UNVISITED` "could be set to the largest possible unsigned integer `cuda::std::numeric_limits<unsigned>`" (p. 433; notes "Unclear in the PDF" item 3)

**Verdict: PARTLY.** The trait exists in libcu++, but the printed expression names only the class. The full expression is `cuda::std::numeric_limits<unsigned int>::max()`.

- libcu++ "Standard API" table lists `<limits>` as `<cuda/std/limits>` with a check mark (supported). The header on `main` includes `cuda/std/__limits/numeric_limits.h`.
- HIP: libhipcxx at `therock-10.2` ships the same header (`include/cuda/std/limits`, including `cuda/std/__limits/numeric_limits.h`). Plain `UINT_MAX` from `<climits>` or `~0u` needs no library on any backend.
- Metal: C5.

- https://nvidia.github.io/cccl/unstable/libcudacxx/standard_api.html
- https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/std/limits
- https://github.com/ROCm/libhipcxx/blob/therock-10.2/include/cuda/std/limits

### A10. Launch overhead "may dominate" for small frontiers and is negligible for large ones (p. 445); what CUDA offers today besides one cooperative kernel

**Verdict: CONFIRMED** as a qualitative statement; NVIDIA gives the same reasoning and no figure. NVIDIA now documents three other ways to cut per-level launch cost, which the book does not mention.

- CUDA Graphs, guide section 4.2: "when you place a kernel into a stream, the host driver performs a sequence of operations in preparation for the execution of the kernel on the GPU. These operations, necessary for setting up and launching the kernel, are an overhead cost which must be paid for each kernel that is issued. For a GPU kernel with a short execution time, this overhead cost can be a significant fraction of the overall end-to-end execution time. By creating a CUDA graph that encompasses a workflow that will be launched many times, these overhead costs can be paid once for the entire graph during instantiation".
- Conditional nodes, section 4.2.4, which put the BFS level loop itself on the device: "Conditional nodes allow conditional execution and looping of a graph contained within the conditional node. This allows dynamic and iterative workflows to be represented completely within a graph and frees up the host CPU to perform other work in parallel." "Conditional WHILE nodes execute their body graph if the condition value is non-zero when the node is executed and will continue to execute their body graph until the condition value is zero." The condition is set from device code: "The condition value can be set by device code using cudaGraphSetConditional()". Handles: `cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault)`, node type `cudaGraphCondTypeWhile`. Body requirements: "The graph can only contain kernel nodes, empty nodes, memcpy nodes, memset nodes, child graph nodes, and conditional nodes." "Cooperative launches are permitted so long as MPS is not in use." HIPIFY dates `cudaGraphConditionalHandleCreate` and `cudaGraphCondTypeWhile` to CUDA 12.3 (B5).
- Programmatic Dependent Launch, section 4.5: "allows for a dependent secondary kernel to launch before the primary kernel it depends on in the same CUDA stream has finished executing. Available starting with devices of compute capability 9.0". Device calls `cudaTriggerProgrammaticLaunchCompletion()` and `cudaGridDependencySynchronize()`, launch attribute `cudaLaunchAttributeProgrammaticStreamSerialization`. llama.cpp uses it (D3).
- For the lesson (ours): the frontier loop's per-level read of `numCurrFrontier` back to the host is the obstacle to plain graph capture (PyTorch says so in its own terms, D5); a WHILE node whose body kernel calls `cudaGraphSetConditional(handle, 0)` when the new frontier is empty removes it. No device numbers: the reader measures launch cost on their GPU.

- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html (sections 4.2, 4.2.4 to 4.2.4.4)
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html (section 4.5)

### A11. Degree bucketing: "One notable implementation [ref 4] uses three buckets": small-degree vertices get a thread, medium a warp, large a block (p. 449)

**Verdict: PARTLY.** The three tiers are real, but the paper's smallest tier is not one thread per vertex, the tiers are chosen inside one kernel rather than by sorting into buckets, and the paper builds its queues with prefix sums, not atomics.

Merrill, Garland, Grimshaw, "Scalable GPU Graph Traversal", PPoPP 2012 (PDF on NVIDIA Research, read today):

- "Scan+warp+CTA gathering. We can mitigate this imbalance by supplementing fine-grained scan-based expansion with coarser CTA-based and warp-based expansion. CTA-wide gathering is similar to warp-based gathering, except threads vie for control of the entire CTA for strip-mining very large adjacency lists. Then we apply warp-based gathering to acquire adjacency smaller than the CTA size, but greater than the warp width. Finally we perform scan-based gathering to efficiently acquire the remaining "loose ends"." So: block for the largest lists, warp for the middle, and for the rest a block-wide prefix sum that spreads many short lists over all threads.
- "Enlistment operates by having each thread attempt to vie for control of its warp by writing its thread-identifier into a single word shared by all threads of that warp." Dynamic, per iteration, inside the expansion kernel.
- Queues: "We present a BFS parallelization focused on fine-grained task management constructed from efficient prefix sum that achieves an asymptotically optimal O(|V|+|E|) work complexity." The authors argue against the book's atomic counter: "serialization from atomic synchronization is particularly expensive for GPUs".
- Visiting: they accept the benign race of p. 433 ("This is generally considered benign since all such contending tasks would apply the same dj") and cull duplicates heuristically: "Atomic read-modify-write updates to visitation status yield zero false-negatives. As alternatives, we introduce two localized mechanisms for reducing false-negatives: (1) warp culling and (2) history culling." That is a different trade-off from the book's CAS.
- For the lesson: "sort by degree, one kernel per bucket" is a fair general description of degree bucketing; attribute the thread/warp/block idea to [ref 4] with the qualification above. cuGraph's current split is in D1.

- https://research.nvidia.com/publication/2012-02_scalable-gpu-graph-traversal
- https://research.nvidia.com/sites/default/files/pubs/2012-02_Scalable-GPU-Graph/ppo213s-merrill.pdf

### A12. GraphBLAS is "an API specification" with SuiteSparse:GraphBLAS as reference [2] (p. 438)

**Verdict: CONFIRMED.** Current versions, for the one line the lesson needs:

- graphblas.org: "Current versions: GraphBLAS C API, version 2.0.0 (November 15, 2021); GraphBLAS C API, version 2.1.0 (December 22, 2023)". "The C reference implementation is SuiteSparse:GraphBLAS, which implements the version 2.1.0 (final) C API."
- SuiteSparse:GraphBLAS README (branch `stable`): "VERSION 10.5.4, Nov 20, 2026". "Versions 9.0.0 and later conform to the v2.1.0 C API." On GPUs: "CUDA: GPU interface, a work in progress. ... It appears in this release but the CUDA folder is a draft that isn't ready to use yet." So the linear-algebra route of p. 438 is a CPU library today; a GPU BFS comes from cuGraph (A13) or hand-written kernels.

- https://graphblas.org/
- https://github.com/DrTimothyAldenDavis/GraphBLAS/blob/stable/README.md

### A13. "Efficient implementations of many graph algorithms are available in RAPIDS cuGraph" [ref 5] (p. 449)

**Verdict: CONFIRMED,** with a changed address and name.

- `https://docs.rapids.ai/api/cugraph/stable/` now answers 301 to `https://docs.nvidia.com/cugraph/latest/`, titled "NVIDIA cuGraph Documentation" (version selector 27.02): "NVIDIA cuGraph is an open-source collection of GPU-accelerated graph analytics libraries." It lists the Python `cugraph` package, `pylibcugraph`, `libcugraph` and `nx-cugraph` ("a NetworkX backend that accelerates supported NetworkX algorithms on NVIDIA GPUs without changing application code").
- BFS: `cugraph.bfs(G, start=None, depth_limit=None, i_start=None, directed=None, return_predecessors=True)`: "Find the distances and predecessors for a breadth first traversal of a graph. Unlike SSSP, BFS supports unweighted graphs."
- NVIDIA GPUs only. AMD's port is in early access (B6); Apple has none (C6). What the library does inside: D1.

- https://docs.nvidia.com/cugraph/latest/
- https://docs.nvidia.com/cugraph/latest/api_docs/api/cugraph/cugraph.bfs/
- https://github.com/rapidsai/cugraph/blob/main/VERSION

---

## Group B: HIP / ROCm

### B1. The book's four-argument `cudaLaunchCooperativeKernel` call ports to HIP through a rename

**Verdict: WRONG.** It does not compile: HIP's launch takes six arguments with no defaults (infra B4; re-read in the header today).

- `hip_runtime_api.h` at `therock-10.2`: `hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX, void** kernelParams, unsigned int sharedMemBytes, hipStream_t stream);` and `template <class T> inline hipError_t hipLaunchCooperativeKernel(T f, dim3 gridDim, dim3 blockDim, void** kernelParams, unsigned int sharedMemBytes, hipStream_t stream)`. Documented return codes include `#hipErrorCooperativeLaunchTooLarge`.
- The portable form passes all six, `cudaLaunchCooperativeKernel((void*)bfs_kernel, numBlocks, numThreadsPerBlock, kernelArgs, 0, 0)`, which is valid CUDA too. Our `pmpp/vendors/hip.h` already says so: "hipLaunchCooperativeKernel has no default arguments: pass all six." llama.cpp passes six (D2).
- A slip in AMD's own example (`cooperative_groups.rst` at the tag, "Kernel launch", Grid tab): it declares `void* params[] = {};` and then calls `hipLaunchCooperativeKernel(vector_reduce_kernel<partition_size>, dim3(num_blocks), dim3(threads_per_block), 0, 0, hipStreamDefault)`, passing `0` for the arguments of a kernel that has three parameters. Do not copy it.
- The include also differs: `<hip/hip_cooperative_groups.h>` (infra B4), so a single `.cu` needs `#if` around it, as llama.cpp does (D2).

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 7661 to 7683, 10922 to 10927)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/how-to/hip_runtime_api/cooperative_groups.rst (around lines 439 to 453); published at https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html

### B2. `grid_group`, `this_grid()`, `sync()`, `thread_rank()`, `num_threads()` on HIP

**Verdict: PARTLY.** All exist with CUDA's names, but with 32-bit return types, and AMD's docs do not state the memory guarantee of `sync()` that A5 quotes for CUDA.

- Header `amd_hip_cooperative_groups.h` at the tag: `class grid_group : public thread_group` with `__hip_uint32_t thread_rank() const`, `__hip_uint32_t block_rank() const`, `bool is_valid() const`, `void sync() const`, `dim3 group_dim() const`, `barrier_arrive()`, `barrier_wait(arrival_token&&)`. `num_threads()` and `size()` come from the base class: `__hip_uint32_t num_threads() const { return _num_threads; }` and "Total number of threads in the group (alias of num_threads())". The docs' list for `grid_group` names `thread_rank()`, `size()`, `cg_type()`, `is_valid()`, `sync()`, `barrier_arrive()` and `barrier_wait()` but not `num_threads()`; the header has it.
- Implementation (`hip_cooperative_groups_helper.h`): `sync()` is `wait_async_copies(); __ockl_grid_sync();` and `is_valid()` is `__ockl_grid_is_valid()`. 32-bit ranks match the header note "HIP does not support kernel launch with total work items defined in dimension with size gridDim · blockDim ≥ 2^32".
- The book's `unsigned int` loop variables fit both vendors.
- Memory visibility: the HIP page says of the split barrier only that after `barrier_wait()` "it is safe to read results written by other threads"; I found no statement for `sync()` itself or for grid scope. COULD NOT VERIFY a documented AMD equivalent of A5's guarantee; the lab should treat it as the same contract and say it is CUDA's wording.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_hip_cooperative_groups.h (lines 39 to 110, 185 to 216)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/hip_cooperative_groups_helper.h (lines 164 to 216)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/how-to/hip_runtime_api/cooperative_groups.rst (lines 96 to 113, "Split barrier")

### B3. HIP enforces the same residency rule, against the same product

**Verdict: CONFIRMED** (from runtime source).

- `hip_module.cpp` at the tag, for a cooperative launch: it fails with `hipErrorLaunchFailure` if `!device->info().cooperativeGroups_`, then calls `ihipOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, &max_blocks_per_grid, ...)` and `if (((launch_params.global_.product()) / block_size) > unsigned(max_blocks_per_grid)) { return hipErrorCooperativeLaunchTooLarge; }`.
- `hip_platform.cpp`: `*numBlocksPerGrid = maxCUs * std::min(bestBlocksPerCU, lds_occupancy_wgs);`, where `maxCUs` is `device.info().maxComputeUnits_` adjusted by a factor of 2 when the kernel's WGP mode differs from the device setting, and the per-CU count returned to the user is adjusted the opposite way ("The count above is per scheduling unit of the kernel: a WGP for a WGP mode kernel, a single CU for a kernel compiled with -mcumode."). `hip_device.cpp` sets `deviceProps.multiProcessorCount = info.maxComputeUnits_;`.
- Our reading of these lines: `multiProcessorCount * hipOccupancyMaxActiveBlocksPerMultiprocessor(...)` equals the runtime's limit, so A2's recipe is the matching one on AMD. Not tested on hardware.
- Support flag: `hipDeviceGetAttribute(&supports_coop_launch, hipDeviceAttributeCooperativeLaunch, device)` (docs example).

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_module.cpp (lines 406 to 423)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_platform.cpp (lines 22 to 140)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_device.cpp (lines 665, 877)

### B4. `cuda::atomic_ref<unsigned, ...>::compare_exchange_strong` keeps its strong semantics on HIP (libhipcxx)

**Verdict: PARTLY.** The call compiles with the same spelling, but at block and device scope libhipcxx implements it with the weak built-in. The legacy `atomicCAS` on HIP is strong.

- Dispatch (`include/cuda/std/__atomic/types/base.h` at `therock-10.2`): `__atomic_compare_exchange_strong_dispatch` calls `__atomic_compare_exchange_cuda(__a->get(), __expected, __val, false, ...)`; the weak dispatch passes `true`.
- Backend (`include/cuda/std/__atomic/functions/atomic_hip_generated.h`): for `__thread_scope_block_tag` and `__thread_scope_device_tag` the flag parameter is unnamed and ignored, and both call `__hip_atomic_compare_exchange_weak(__ptr, __expected, __desired, __success_memorder, __failure_memorder, __HIP_MEMORY_SCOPE_WORKGROUP)` (block) or `..., __HIP_MEMORY_SCOPE_AGENT)` (device). Only `__thread_scope_system_tag` checks `__is_weak` and calls `__hip_atomic_compare_exchange_strong` when it is false.
- What "weak" permits, in LLVM's IR reference: "If the cmpxchg operation is marked as weak then a spurious failure is permitted: the operation may not write <new> even if the comparison matched." COULD NOT VERIFY whether AMD hardware ever fails a compare-and-swap spuriously; I found no AMD statement either way (the LLVM AMDGPU usage page does not mention cmpxchg strength).
- Legacy route (`amd_hip_atomic.h`): `unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val) { __scoped_atomic_compare_exchange_n(address, &compare, val, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE); return compare; }`, where the fourth argument 0 means not weak. It returns the old value, so `atomicCAS(&level[v], UNVISITED, currLevel) == UNVISITED` is an exactly-once visit.
- For the lab (ours): either use the legacy `atomicCAS` (strong on CUDA and HIP), or keep the book's `atomic_ref` and retry while the failure was spurious: `unsigned e = UNVISITED; while (!ref.compare_exchange_strong(e, currLevel, relaxed, relaxed) && e == UNVISITED) {}`. The loop costs nothing when no failure is spurious and is the same shape Metal needs (C4). The emulator's CAS is strong and ignores scope (project notes), so neither problem can be shown there.

- https://github.com/ROCm/libhipcxx/blob/therock-10.2/include/cuda/std/__atomic/types/base.h (lines 138 to 177)
- https://github.com/ROCm/libhipcxx/blob/therock-10.2/include/cuda/std/__atomic/functions/atomic_hip_generated.h (lines 58 to 61, 122 to 125, 190 to 196)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_hip_atomic.h (lines 76 to 81)
- https://llvm.org/docs/LangRef.html (section "cmpxchg Instruction")

### B5. HIP has CUDA's other launch-overhead tools: graphs, conditional (WHILE) nodes, programmatic dependent launch

**Verdict: PARTLY.** Graphs, yes. Conditional nodes and PDL, no.

- HIP graphs page (`hipgraph.rst` at the tag): "The standard method of launching kernels incurs a small overhead for each iteration of the operation involved. ... Especially for some AI frameworks, a GPU kernel might run faster than the time it takes for the framework to set up and launch the kernel, and so the overhead of repeatedly launching kernels can have a significant impact on performance. HIP graphs are designed to address this issue, by predefining the HIP API calls and their dependencies with a graph, and performing most of the initialization beforehand."
- HIPIFY's "CUDA Runtime API supported by HIP" table (ROCm Core SDK 10.2.0) lists `cudaGraphConditionalHandleCreate` (12.3), `cudaGraphConditionalHandleCreate_v2` (13.1), `cudaGraphCondTypeIf`, `cudaGraphCondTypeWhile` (12.3), `cudaGraphCondTypeSwitch` (12.8), `cudaGraphConditionalHandle`, and `cudaLaunchAttributeProgrammaticStreamSerialization` (11.8) with no HIP name, where supported rows carry one (for example `cudaGraphCreate` to `hipGraphCreate`, 4.3.0). `hip_runtime_api.h` at the tag has no conditional-node API (searched "conditional"). `cudaGridDependencySynchronize` and `cudaTriggerProgrammaticLaunchCompletion` do not appear in HIPIFY's device-API table.
- So on AMD the per-level loop stays on the host (or in a cooperative kernel); a graph can replay a fixed number of levels but cannot decide on the device when to stop.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/how-to/hip_runtime_api/hipgraph.rst (lines 38 to 52); published at https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Device_API_supported_by_HIP.html

### B6. AMD's counterpart of cuGraph

**Verdict: PARTLY.** It exists, is derived from cuGraph, and is in early access.

- hipGRAPH docs (page dated 2026-07-29, under "Data Science 26.03"): "hipGRAPH is in an early access state. Running production workloads is not recommended. The early access version of the hipGRAPH Python layer is untested and the Python API is unsupported." "hipGRAPH is a graph marshalling library that acts as a wrapper between your application and a worker graph library, such as rocGRAPH." "hipGRAPH is derived from the cuGraph library that forms part of the NVIDIA RAPIDS open source project. The hipGRAPH library described here should not be confused with HIP graphs in the HIP runtime API." The API list includes `pylibhipgraph.bfs` under "Traversal Functions".
- rocGRAPH docs: "rocGRAPH is in an early access state. Running production workloads is not recommended."
- The current AMD Data Science release notes (26.10, "updates the toolkit to ROCm 10.0.0") list hipDF, hipMM, hipRAFT, hipVS, dask-hip and hip-ucxx; hipGRAPH and rocGRAPH are not in the component table. Whether they are dropped or only unchanged: not stated, not determined.
- The naming trap AMD itself points out is worth one line in the lesson: "graph" in "CUDA graphs", "HIP graphs" and "MPSGraph" means a recorded sequence of GPU work, not a graph to traverse.

- https://rocm.docs.amd.com/projects/hipGRAPH/en/latest/
- https://rocm.docs.amd.com/projects/rocGRAPH/en/latest/
- https://rocm.docs.amd.com/projects/datascience/en/latest/about/release-notes.html

---

## Group C: Metal

### C1. §18.7 (one kernel, grid-wide barriers, occupancy-sized grid) has no Metal form

**Verdict: CONFIRMED** (cited). Metal documents no grid-wide execution barrier and no cooperative launch (ch04 C3), no forward progress between threadgroups (ch11 C5, with an Apple engineer's statement that the chained-scan assumptions are undefined), no occupancy query and no core count from which to size a resident grid (ch04 C5, C1). The Metal version of the chapter is one dispatch per level (C2), optionally sized on the GPU (C3).

### C2. One dispatch per BFS level on Metal: are the dispatches ordered, and does the next one see the previous one's writes?

**Verdict: CONFIRMED** for the default setup the labs use.

- `MTLDispatchType` (header `MTLCommandBuffer.h`, macOS 27.0 SDK): "MTLDispatchTypeSerial: Command encoder dispatches are executed in dispatched order." "MTLDispatchTypeConcurrent: Command encoder dispatches are executed in parallel with each other." `computeCommandEncoder` takes no dispatch type; `computeCommandEncoderWithDispatchType:` is the way to ask for concurrent ("Optionally allow this command encoder to execute dispatches concurrently"). That wording makes serial the behaviour of the plain call; the header does not state a default in so many words (our reading). Apple's page for `.concurrent`: "If you encode multiple commands that access a single resource, you're responsible for synchronizing the memory operations to that resource."
- Hazard tracking, `MTLHazardTrackingMode.tracked`: "When at least one command writes to a tracked resource, the framework takes the following actions: Delay write operations until all previous read operations finish; Prevent subsequent commands from running until write operations finish". `.default`: "The default tracking mode for a type that inherits MTLResource is tracked". Limit: "Metal applies hazard tracking to resources you create with tracked, but only when you submit commands that use those resources to an MTLCommandQueue." "Metal doesn't apply hazard tracking to commands you submit to an MTL4CommandQueue".
- With a concurrent encoder the dependency is stated with `memoryBarrier(scope:)`: "Memory barriers ensure the relevant passes finish updating resources before starting the stages of subsequent commands that depend on those resources." (llama.cpp's choice, D3.)
- Within one dispatch nothing changes from ch11 C3: another threadgroup's plain stores are not guaranteed visible. The frontier kernels do not need that (the CAS on `level` is atomic; frontier entries are read in the next dispatch).

- https://developer.apple.com/documentation/metal/mtldispatchtype/serial
- https://developer.apple.com/documentation/metal/mtldispatchtype/concurrent
- https://developer.apple.com/documentation/metal/mtlcommandbuffer/makecomputecommandencoder(dispatchtype:)
- https://developer.apple.com/documentation/metal/mtlhazardtrackingmode
- https://developer.apple.com/documentation/metal/mtlhazardtrackingmode/tracked
- https://developer.apple.com/documentation/metal/mtlhazardtrackingmode/default
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/memorybarrier(scope:)
- Local header: `Metal.framework/Headers/MTLCommandBuffer.h` (macOS 27.0 SDK), lines 222 to 238, 391 to 402

### C3. Sizing the next level's dispatch on the GPU: indirect dispatch

**Verdict: CONFIRMED.** Metal can take a dispatch's threadgroup count from a buffer the GPU wrote, which removes the per-level CPU read of the frontier size. CUDA's nearest documented counterpart is a graph WHILE node (A10); HIP has neither (B5).

- `dispatchThreadgroups(indirectBuffer:indirectBufferOffset:threadsPerThreadgroup:)` (macOS 10.11 and later): "Encodes a dispatch call for a compute pass, using an indirect buffer that defines the size of a grid that aligns to threadgroup boundaries." "Lay out the data in this buffer as described in the MTLDispatchThreadgroupsIndirectArguments structure." Offset: "This value needs to be a multiple of 4." "The GPU fetches parameters from the indirect buffer just before the thread grid starts. This process lets the compute function run based on GPU feedback, without latency from data transfer between the CPU and the GPU."
- Layout (header `MTLComputeCommandEncoder.h`): `typedef struct { uint32_t threadgroupsPerGrid[3]; } MTLDispatchThreadgroupsIndirectArguments;`
- Not documented in the pages read: whether a threadgroup count of 0 is allowed. A design that encodes a fixed maximum number of levels should write at least 1 and let the kernel return early on an empty frontier (ours).

- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/dispatchthreadgroups(indirectbuffer:indirectbufferoffset:threadsperthreadgroup:)
- https://developer.apple.com/documentation/metal/mtldispatchthreadgroupsindirectarguments
- Local header: `Metal.framework/Headers/MTLComputeCommandEncoder.h` (macOS 27.0 SDK), lines 27 to 29, 230 to 236

### C4. Compare-and-swap for "visit exactly once" in MSL

**Verdict: PARTLY.** It exists for `device` and `threadgroup` atomics, but only in a weak form, and the spec does not say whether it can fail spuriously.

- MSL spec section 6.16.4.4 "Atomic Compare and Exchange Functions" (pages 304 to 305): "These compare-and-exchange functions atomically compare the value in *object with the value in *expected. If those values are equal, the compare-and-exchange function performs a read-modify-write operation to replace *object with desired. Otherwise if those values are not equal, the compare-and-exchange function loads the actual value from *object into *expected. If the underlying atomic value in *object was successfully changed, the compare-and-exchange function returns true; otherwise it returns false. These functions support atomic types A of atomic_int, atomic_uint, atomic_bool, and atomic_float."
- Signatures: `bool atomic_compare_exchange_weak_explicit(device A* object, thread C *expected, C desired, memory_order success, memory_order failure)` (and `threadgroup`, and `volatile` forms; Metal 4.1 adds a trailing `mem_flags`). "support for memory_order_relaxed for indicating success and failure". There is no `atomic_compare_exchange_strong_explicit` in the spec (searched "compare_exchange_strong").
- The spec never uses "spurious"; the name "weak" is C11's name for the form that may fail spuriously. COULD NOT VERIFY whether Apple GPUs do. Write the loop (as in B4); it is what PyTorch and MLX do for their CAS-based atomics (ch09 C2).
- Compiled here (not run), at `-std=metal3.0`, `metal3.2` and `metal4.1`, no diagnostics: a kernel with `device atomic_uint* level`, `uint expected = UNVISITED; bool won = false; while (!(won = atomic_compare_exchange_weak_explicit(&level[v], &expected, lvl, memory_order_relaxed, memory_order_relaxed)) && expected == UNVISITED) {}`, a `threadgroup atomic_uint cnt_s;` zeroed with `atomic_store_explicit(&cnt_s, 0u, memory_order_relaxed)` before `threadgroup_barrier(mem_flags::mem_threadgroup)`, and `atomic_fetch_add_explicit(&cnt_s, 1u, memory_order_relaxed)` for the private append.
- The Metal form of the book's line 29 is automatically an atomic store (`atomic_store_explicit`), since a `threadgroup atomic_uint` cannot be assigned plainly. The race the CUDA code has (A7) cannot be written on Metal.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 6.16.4.4, pages 304 to 305)

### C5. The `UNVISITED` sentinel in MSL

**Verdict: CONFIRMED.** Both spellings exist and agree.

- Local Metal toolchain headers: `metal_types` line 273, `#define UINT_MAX (__INT_MAX__ * 2U + 1U)`; `metal_limits` defines `numeric_limits` and returns `UINT_MAX` for the unsigned `max()` (line 385). The MSL spec itself uses the class in an example (`metal::numeric_limits<float>::lowest()`) but does not list it.
- Compiled here: `constant uint UNVISITED = numeric_limits<uint>::max();` with `static_assert(numeric_limits<uint>::max() == UINT_MAX, "limits");` at `metal3.0`, `metal3.2`, `metal4.1`.

- Local toolchain: `Metal.xctoolchain/usr/metal/32023/lib/clang/32023/include/metal/metal_types` and `.../metal_limits`

### C6. A library BFS on Apple GPUs

**Verdict: COULD NOT VERIFY** (none found). A web search restricted to developer.apple.com for Metal breadth-first search or graph traversal returned no Apple page; the hits were third-party and academic. Apple's `MPSGraph` is "The optimized representation of a compute graph of operations and tensors", a computation-graph framework, not graph analytics (the naming trap of B6). For the course: on a Mac the BFS is the reader's own kernel.

- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph

---

## Group D: How real libraries do this chapter (cuGraph, llama.cpp, PyTorch, MLX)

### D1. cuGraph's BFS: direction-optimising switch, degree segments, and a warp-aggregated frontier append

**Verdict: CONFIRMED** as source read (`main`, read today). It answers two things the book leaves open: a concrete push/pull switch rule (Exercise 2) and what degree bucketing looks like in production (§18.8).

- `cpp/src/traversal/bfs_impl.cuh`: "direction optimizing BFS implementation is based on "S. Beamer, K. Asanovic, D. Patterson, Direction-Optimizing Breadth-First Search, 2012"". It requires a symmetric graph: "input graph should be symmetric for direction optimizing BFS." (the book's p. 436 point that undirected graphs need only one of CSR and CSC).
- The switch (our paraphrase of the code): go bottom-up when `m_f * alpha > m_u` and the frontier is not shrinking, where `m_f` and `m_u` are edge counts out of the frontier and out of unvisited vertices and `alpha` is the average degree times a tuning factor; go back top-down when `next_frontier_size * beta < unvisited_vertices` and the frontier is shrinking, with `constexpr vertex_t direction_optimizing_beta = 24; // tuning parameter`.
- The level loop is a host `while (true)` that ends with `if (next_aggregate_frontier_size == 0) { break; }`: the book's one-launch-sequence-per-level design, not a cooperative kernel.
- Degree segments (`cpp/include/cugraph/graph_view.hpp`): `low_degree_threshold{raft::warp_size()}` ("belongs to the low degree segment if the global degree is smaller than this value") and `mid_degree_threshold{1024}` ("belongs to the medium degree segment ... otherwise, belongs to the high degree segment"), with the comment "FIXME: threshold values require tuning".
- One kernel per segment (`cpp/include/cugraph/prims/detail/extract_transform_if_v_frontier_e.cuh`): `extract_transform_if_v_frontier_e_hypersparse_or_low_degree` (a warp takes up to a warp's worth of frontier vertices and spreads their edges over its lanes with `cub::WarpScan` over the degrees: Merrill's scan-based gathering at warp size), `..._mid_degree` (`idx = tid / raft::warp_size()`: one warp per vertex), `..._high_degree` (one thread per edge across the whole grid, finding each edge's vertex with `thrust::upper_bound` over degree offsets: edge-centric, §18.4, applied only to the heavy vertices).
- The frontier append is Chapter 12's warp aggregation, not the book's block privatisation: `auto ballot = __ballot_sync(raft::warp_full_mask(), e_op_result ? uint32_t{1} : uint32_t{0}); ... warp_buffer_start_idx = buffer_idx.fetch_add(increment, cuda::std::memory_order_relaxed);` on a `cuda::atomic_ref<size_t, cuda::thread_scope_device>`, then `__shfl_sync` to broadcast and `__popc(ballot & ~(raft::warp_full_mask() << lane_id))` for the rank. This is CUDA-only code (32-bit masks; ch12 B1).

- https://github.com/rapidsai/cugraph/blob/main/cpp/src/traversal/bfs_impl.cuh (lines 52 to 66, 150 to 160, 295 to 306, 441 to 456, 606 to 666, 858 to 870)
- https://github.com/rapidsai/cugraph/blob/main/cpp/include/cugraph/graph_view.hpp (lines 244 to 254)
- https://github.com/rapidsai/cugraph/blob/main/cpp/include/cugraph/prims/detail/extract_transform_if_v_frontier_e.cuh (lines 88 to 116, 128 to 300, 310 to 420, 423 to 470)

### D2. llama.cpp's grid-wide barrier: the cooperative softmax, re-read

**Verdict: CONFIRMED** (extends ch10 D1). The one place in llama.cpp's CUDA and HIP backend found today that uses `grid.sync()`.

- `ggml/src/ggml-cuda/softmax.cu`: includes split by vendor (`#ifdef GGML_USE_HIP` / `#include <hip/hip_cooperative_groups.h>` / `#else` / `#include <cooperative_groups.h>`), `const cg::grid_group g = cg::this_grid();` and four `g.sync()` calls.
- Sizing is the guide's first recipe (A2), one block per SM, with no occupancy query: `cudaLaunchCooperativeKernel(kernel_fn, dim3(ggml_cuda_info().devices[id].nsm, 1, 1), dim3(WARP_SIZE * 8, 1, 1), kernel_args, 0, stream)`, where `nsm` is `prop.multiProcessorCount` (`ggml-cuda.cu` line 410). All six arguments, so it compiles for HIP (B1).
- Guarded by `ggml_cuda_info().devices[id].supports_cooperative_launch`, filled from `cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id)`. The path is taken only for rows too big for one block ("Parallelize across SMs for top-p/dist-sampling"), when `ncols_x / (ne01 * ne02 * ne03) > 8192`, with no mask, no sinks, scale 1 and `max_bias` 0. Their comment on the heuristic: "done on the basis of a B6000 GPU and Can be adapted further for lower-SM-count GPUs". That is a device-specific tuning note; lessons quote the rule, not the device.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/softmax.cu (lines 1 to 10, 150 to 290, 405 to 450)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu (lines 410 to 416)

### D3. llama.cpp's answer to launch overhead on each backend: CUDA graphs, HIP graphs, PDL, and concurrent Metal encoders

**Verdict: CONFIRMED** as source read. Token generation is the small-frontier case of p. 445: many short kernels per token.

- CUDA graphs: `ggml/CMakeLists.txt` has `option(GGML_CUDA_GRAPHS "ggml: use CUDA graphs (llama.cpp only)" ${GGML_CUDA_GRAPHS_DEFAULT})` with the default OFF for ggml alone, and llama.cpp's top-level `CMakeLists.txt` sets `GGML_CUDA_GRAPHS_DEFAULT ON` when the option is not given. `option(GGML_HIP_GRAPHS "ggml: use HIP graph" ON)`. `common.cuh`: `#if (defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS)) || defined(GGML_MUSA_GRAPHS)` / `#define USE_CUDA_GRAPH`. The backend captures with `cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed)`, instantiates, updates with `cudaGraphExecUpdate`, and replays with `cudaGraphLaunch`; `vendors/hip.h` maps each name (`#define cudaGraphLaunch hipGraphLaunch`, `#define cudaGraphInstantiate hipGraphInstantiate`, ...).
- What breaks a graph is what breaks the BFS host loop: `ggml_cuda_graph_check_compability` turns graphs off for a `GGML_OP_MUL_MAT_ID` node when "the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs".
- PDL (A10) on CUDA only: `#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && (CUDART_VERSION >= 12030 || ...)` / `#define GGML_CUDA_USE_PDL`, and device helpers that call `cudaGridDependencySynchronize()` and `cudaTriggerProgrammaticLaunchCompletion()` only `#if defined(GGML_CUDA_USE_PDL) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER`. None on HIP, consistent with B5.
- Metal, which has no graph API: compute encoders are created with `computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent` when concurrency is on (off with `GGML_METAL_CONCURRENCY_DISABLE`), and a dependency inserts `[encoder->obj memoryBarrierWithScope:MTLBarrierScopeBuffers]` (C2). The CPU encoding cost is hidden by splitting the graph over several command buffers encoded on extra threads: "the first n_main nodes are encoded and submitted for processing directly by the calling thread while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes" (`dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async)`).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt (lines 117 to 118, 210, 217)
- https://github.com/ggml-org/llama.cpp/blob/master/CMakeLists.txt (lines 171 to 173)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh (lines 118 to 157, 1384 to 1386)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu (lines 3307 to 3343, 3443 to 3495, 5420 to 5428, 5510 to 5538)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/hip.h (lines 128 to 141)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-context.m (lines 55, 142, 488 to 520, 607)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m (lines 835 to 845, 893 to 895)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-ops.cpp (lines 159 to 185)

### D4. A privatised work list in LLM inference: llama.cpp's mixture-of-experts routing

**Verdict: CONFIRMED** as source read. This is the chapter's queue-building pattern applied to "which tokens go to which expert", and the best bridge for this reader.

- `ggml/src/ggml-cuda/mmid.cu`, `mm_ids_helper`: "Helper function for mul_mat_id, converts ids to a more convenient format. ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert." Launch: `const dim3 num_blocks(n_experts, 1, 1); const dim3 block_size(warp_size, 1, 1);`, one single-warp block per expert, with a dynamic shared-memory buffer `store` of `ne_store` entries, at most the per-block opt-in limit (`smpbo`).
- The append is a ballot rank, not an atomic: `iex_used = ggml_cuda_ballot(expert_found == expert); ... it_compact_offset = ggml_cuda_popc(iex_used & mask); if (expert_found == expert) { store[it_compact + it_compact_offset] = ...; } it_compact += ggml_cuda_popc(iex_used);`. So the private list is in token order (stable), unlike the book's Fig. 18.16.
- Overflow is handled by flushing, the counterpart of Fig. 18.15's overflow branch: "Step 1.5: If the store does not have enough space to hold all values, iterate over the rest of the tokens just to increment nex_prev." "Step 3: If we were not able to process all values, resume at the index up to which we were able to dump the output and re-do." The code dumps whenever `it_compact + max(warp_size, n_expert_used) >= ne_store`.
- The global allocation is computed, not claimed: `nex_prev` counts entries for lower-numbered experts, so each block knows where its slice starts without the book's `fetch_add` on a public counter. The commit (`dump_store`) is a strided loop over the private store (`for (int itc = threadIdx.x; itc < it_compact; itc += warp_size)`), the book's coalesced commit.
- Mask type `ggml_cuda_thread_mask_t` is 32 or 64 bits by vendor (ch05 B9), which is how the same file serves HIP.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmid.cu (lines 4 to 20, 27 to 66, 71 to 175, 250 to 261)

### D5. PyTorch: why a BFS-style host loop cannot be captured, and the conditional-node escape

**Verdict: CONFIRMED** (PyTorch docs 2.15, "CUDA semantics").

- "Why CUDA Graphs? Replaying a graph sacrifices the dynamic flexibility of typical eager execution in exchange for greatly reduced CPU overhead. ... Under the hood, a replay submits the entire graph's work to the GPU with a single call to cudaGraphLaunch."
- Constraints that a per-level frontier loop violates: "Ops that synchronize the CPU with the GPU (e.g., .item() calls) are prohibited." "Dynamic control flow (based on CPU or GPU data) is prohibited, unless it is based on GPU data and implemented via higher order operator torch.cond()." "Dynamic shapes are prohibited." Reading `numCurrFrontier` to decide whether to launch the next level is an `.item()`.
- "Data-dependent control flow can [work] with cuda graphs if the control flow is implemented using torch.cond(). If your function uses this function, compiling it with the "cudagraphs" backend will enable control flow in the resulting cuda graph via conditional nodes." (A10's mechanism, exposed for `if` only; no loop form is documented on this page.)
- The data-dependent output size itself (`torch.nonzero`) already costs a host sync: ch12 D4.

- https://docs.pytorch.org/docs/2.15/notes/cuda.html (sections "CUDA Graphs", "Constraints", "Data Dependent Control Flow")

### D6. MLX: small work is "overhead bound", and fusion is the cure

**Verdict: CONFIRMED** (MLX docs 0.32.4, "Compilation"). The §18.7 argument (p. 445) and its fusion benefit (p. 448), in MLX's words, on Metal.

- "If you use this function with small arrays, it will be overhead bound. If you use it with large arrays it will be memory bandwidth bound. However, all of the operations in the gelu are fusible into a single kernel with compile(). This can speedup both cases considerably."
- MLX has no grid-wide barrier to fuse with (C1); `mx.compile` fuses element-wise chains into one kernel, which removes launches and the round trips through memory between them. A data-dependent loop like BFS stays a host loop in MLX, which also has no data-dependent shapes (ch12 D6).

- https://ml-explore.github.io/mlx/build/html/usage/compile.html

---

## Group E: our own check (not a vendor fact)

### E1. The overflow branch of Fig. 18.15 is arithmetically right when the counter reset is an atomic store

**Verdict: CONFIRMED** by our computation, which is not a vendor statement.

A Python script enumerated every interleaving of N threads, each doing `i = fetch_add(cnt, 1)`, then either `private[i] = me` (if `i < CAP`) or `store(cnt, CAP)` followed by a public append, with each shared step atomic and sequentially consistent. For every N from 1 to 5 and CAP from 1 to 4 (up to 12,600 interleavings for N = 5, CAP = 1), there were no failures of: every thread lands exactly once; private slots are distinct and below CAP; after all threads finish, `cnt == min(N, CAP)`, which is what the allocate and commit steps read. The argument behind it (ours): only a thread whose `fetch_add` returned at least CAP stores, and that store follows its own increment in the counter's modification order, so the last modification is never an increment past CAP. Because only one atomic object is involved, the same holds under `memory_order_relaxed` (all modifications of one atomic object have one total order in C++).

So the corrected lab code needs only to make line 29 an atomic store through the same reference (A7), or to clamp with `min` after the barrier. The book's plain store gives the same numbers on a sequentially consistent model but is a data race (A7). The script was run from the session scratchpad and is not in the repository.

---

## Project notes (not vendor facts)

Recorded so lab authors do not rediscover them; they describe this repository on the check date.

- The emulators model the cooperative launch and grid barrier: `cudaLaunchCooperativeKernel` in `labs/common/include/pmpp/emu/emu_host.h` ("Every block of the grid is alive at once, so that grid_group::sync() can work"), a synthetic `cudaOccupancyMaxActiveBlocksPerMultiprocessor` ("it says nothing about occupancy"), and `grid_group` with `is_valid`, `sync`, `thread_rank`, `num_threads` (64-bit, as CUDA) in `cooperative_groups_emu.h`. `labs/README.md`: grid x block at most 1024 threads, else `cudaErrorCooperativeLaunchTooLarge`; kernel templates are not supported in a cooperative launch.
- The emulator's `cuda::atomic_ref` (`cuda_atomic_emu.h`) implements `compare_exchange_strong` as a strong CAS and accepts but ignores the scope ("every scope behaves the same on one CPU"). So a wrong scope (ch09 A2) and HIP's weak lowering (B4) are invisible to `emu` and `race`; only the plain store of A7 is in their reach.
- `labs/common/include/pmpp/vendors/hip.h` maps `cudaLaunchCooperativeKernel`, `cudaDeviceGetAttribute`, `cudaDevAttrCooperativeLaunch`, `cudaDevAttrMultiProcessorCount` and the occupancy functions, with the six-argument warning (B1). It cannot map the include path (infra B4).
- The Metal lab runtime (`labs/common/src/backend_metal.mm`) uses `[device newCommandQueue]` (an `MTLCommandQueue`, so hazard tracking applies, C2), `computeCommandEncoder` (no dispatch type, so not the concurrent option), and `MTLResourceStorageModeShared` buffers, and waits with `waitUntilCompleted`. It sets no language version.

## Book items with no vendor fact to check

- The example graph, its CSR, CSC and COO arrays (Fig. 18.3), the BFS levels of Fig. 18.4, the work-complexity table (pp. 438 to 439) and the exercise answer sketches in the notes: arithmetic, to be checked by the simulator's unit tests.
- The notes' errata 1, 2, 4, 5 and 6 (Fig. 18.12 cited for Fig. 18.6 on p. 441; "current" for "previous" on p. 439; the unbalanced parenthesis; "breath-first"; "not schedule") and the final-level labels on Figs. 18.7 to 18.13: book-internal. Item 3 (`UNVISITED`) is A9; the unchecked launch argument list is A1.
- Exercise 3's one-barrier-per-level design: our analysis in the notes; no vendor statement bears on it beyond A4 (the split barrier exists) and A5.
- References [1] (Kepner and Gilbert 2011) and [3] (Luo, Wong, Hwu 2010): bibliographic, not checked. [2], [4], [5]: A12, A11, A13.
- Maze routing (§18.2), graphs larger than GPU memory and reordering (§18.8): no vendor fact.

## Answers to the notes' portability questions

| Notes' `[unverified]` row | Answer | Where |
|---|---|---|
| `__global__`, `__device__`, index built-ins | HIP same spelling; Metal kernel functions with `[[thread_position_in_grid]]` and friends | ch05 B4, C4 |
| `cuda::atomic_ref` + `fetch_add` on global memory | HIP same source through libhipcxx; Metal `atomic_fetch_add_explicit` on `device atomic_uint*` | infra B2, C1 |
| `compare_exchange_strong` on `level[vertex]` | HIP: compiles, but lowered to the weak built-in at block and device scope; legacy `atomicCAS` is strong. Metal: `atomic_compare_exchange_weak_explicit` only. Write the retry loop for both | A6, B4, C4 |
| Block-scope atomics on a `__shared__` counter | HIP libhipcxx `thread_scope_block` (`__HIP_MEMORY_SCOPE_WORKGROUP`); Metal `threadgroup atomic_uint` (compiled here) | ch09 C1, B4, C4 |
| `__shared__` arrays and `__syncthreads()` | HIP same; Metal `threadgroup` arrays (declared in the kernel) and `threadgroup_barrier(mem_flags::mem_threadgroup)` | ch05 B4, C1, C2; ch11 C2 |
| Grid-wide barrier (`this_grid().sync()`) | CUDA and HIP under a cooperative launch with a resident grid; HIP needs a different header and 32-bit ranks; none on Metal | A3 to A5, B1 to B3, C1 |
| `cudaLaunchCooperativeKernel` | HIP `hipLaunchCooperativeKernel`, six arguments; none on Metal | A1, B1 |
| Occupancy query, device properties, SM count | HIP same names, same limit enforced at launch; Metal none | A2, B3, ch04 C5 |
| `cuda::std::numeric_limits` | `<cuda/std/limits>` on CUDA and HIP (libhipcxx); `numeric_limits<uint>::max()` or `UINT_MAX` in MSL | A9, C5 |
| RAPIDS cuGraph counterparts | AMD hipGRAPH / rocGRAPH (early access); Apple none found | A13, B6, C6 |

---

## Summary of corrections

Every WRONG, PARTLY and COULD NOT VERIFY item, and every failed or unfound sub-claim:

- A3 (detail): under the cooperative API a too-large grid is refused with `cudaErrorCooperativeLaunchTooLarge`; it does not deadlock. What `grid.sync()` does in an ordinary `<<<>>>` launch is not documented; check `is_valid()`.
- A4 (detail): `grid_group::thread_rank()` and `num_threads()` return `unsigned long long` on CUDA; `size()` is a legacy alias.
- A7: the book's benign-race caveat matches NVIDIA, but its own Fig. 18.15 line 29 / Fig. 18.17 line 35 (plain store to a counter other threads `fetch_add`) is a data race under the same model. Store through the `atomic_ref`, or clamp with `min` after the barrier (E1).
- A8 (cited): compiler aggregation of atomics is NVIDIA blog-only, AMD in LLVM source, Apple undocumented.
- A9: the printed `cuda::std::numeric_limits<unsigned>` is incomplete; write `cuda::std::numeric_limits<unsigned int>::max()` (`<cuda/std/limits>`).
- A11: Merrill, Garland, Grimshaw use CTA, warp and block-wide scan-based gathering chosen dynamically inside one kernel, not one thread per small vertex in a separate kernel per bucket; they build queues with prefix sums and accept the benign race with culling instead of atomics.
- A13 (detail): `docs.rapids.ai/api/cugraph/stable/` now redirects to `docs.nvidia.com/cugraph/latest/` ("NVIDIA cuGraph").
- B1 (WRONG): the book's four-argument `cudaLaunchCooperativeKernel` does not compile on HIP; pass all six. AMD's own HIP example passes `0` as the argument array of a kernel with parameters.
- B2: HIP's `grid_group` ranks and sizes are 32-bit; the docs omit `num_threads()` (the header has it); no AMD statement of `sync()`'s memory guarantee was found (could not verify).
- B4: libhipcxx implements `compare_exchange_strong` with `__hip_atomic_compare_exchange_weak` at block and device scope; whether AMD hardware fails spuriously could not be verified; HIP's legacy `atomicCAS` is strong.
- B5: HIP has graphs but no conditional (WHILE) nodes and no programmatic dependent launch (no HIP names in HIPIFY's tables; none in the header).
- B6: hipGRAPH and rocGRAPH exist (derived from cuGraph) but are early access; their docs sit under the 26.03 release and the 26.10 release notes do not list them.
- C3 (detail): indirect dispatch with a threadgroup count of 0 is not documented.
- C4: MSL has only `atomic_compare_exchange_weak_explicit`, relaxed orders (before Metal 4.1), and does not say whether it fails spuriously (could not verify); use a retry loop.
- C6 (could not verify): no Apple library for graph traversal was found; `MPSGraph` is a compute-graph framework.

Status facts worth knowing even where the verdict is CONFIRMED:

- A1: four arguments are correct on CUDA (C++ overload defaults); NVIDIA's own example has the same form.
- A2: the book's grid sizing is NVIDIA's printed recipe and the exact quantity the runtime checks; B3 shows HIP checks the same product.
- A5: `sync()` on any group makes all earlier memory accesses visible to the group, which is what the three-barrier swap needs.
- A6: after a failed compare-and-swap the expected variable holds the current value, so reusing it without resetting to `UNVISITED` is a bug (the notes' bug-hunt idea 3 is valid).
- A10: CUDA now offers graphs, conditional WHILE nodes (CUDA 12.3; device code ends the loop with `cudaGraphSetConditional`; cooperative launches allowed in the body without MPS), and PDL (compute capability 9.0) as alternatives to per-level host launches.
- A12: SuiteSparse:GraphBLAS 10.5.4 implements GraphBLAS C API 2.1.0; its CUDA interface is still a draft.
- C2: per-level dispatches on an `MTLCommandQueue` with default buffers are ordered and see each other's writes (serial encoder, tracked hazards); `MTL4CommandQueue` does not track hazards.
- C3: Metal can size the next level's dispatch from a GPU-written buffer (`dispatchThreadgroups(indirectBuffer:...)`), something neither CUDA nor HIP offers as a launch call.
- D1: cuGraph is direction-optimising (Beamer 2012; symmetric graphs only; `beta = 24`), splits vertices at warp size and 1024 by degree, and appends to the frontier with warp-aggregated atomics.
- D2, D3: llama.cpp uses a cooperative one-block-per-SM softmax, CUDA and HIP graphs by default in its own build, PDL on Hopper and later (CUDA only), and concurrent encoders with buffer barriers plus multi-threaded command-buffer encoding on Metal.
- D4: llama.cpp's MoE id helper is a privatised, overflow-flushing work-list builder with ballot ranks and computed (not atomic) global offsets.

## Where the backends really differ

One line per cell, for what this chapter uses. "Same" means the same source text compiles through `pmpp/vendors/hip.h`, according to the documentation; nothing here was run on CUDA or AMD hardware, and the Metal cells marked "compiled here" were compiled, not run.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Grid-wide barrier | `this_grid().sync()` under a cooperative launch | Same names; header `<hip/hip_cooperative_groups.h>` | None documented; one dispatch per level |
| Cooperative launch call | `cudaLaunchCooperativeKernel(f, grid, block, args)`; last two default to 0 | `hipLaunchCooperativeKernel`, all six arguments required | None |
| Too-large grid | Refused: `cudaErrorCooperativeLaunchTooLarge` | Refused: `hipErrorCooperativeLaunchTooLarge`, checked against the runtime's own occupancy | Not applicable |
| Sizing a resident grid | `multiProcessorCount * cudaOccupancyMaxActiveBlocksPerMultiprocessor(...)`, or one block per SM | Same product (CUs or WGPs, adjusted in the runtime) | No core count, no occupancy query |
| `grid_group` rank and size | `unsigned long long thread_rank()`, `num_threads()`; `size()` legacy | 32-bit `thread_rank()`, `num_threads()`, `size()` | None |
| Memory visibility across the grid barrier | Documented for `sync()` on every group | Not stated for `sync()` | Dispatch boundary: ordered, tracked hazards on `MTLCommandQueue` |
| Visit-once compare-and-swap | `atomic_ref::compare_exchange_strong` (strong); legacy `atomicCAS` | libhipcxx strong lowered to the weak built-in at block/device scope; legacy `atomicCAS` strong | `atomic_compare_exchange_weak_explicit` only; spurious failure unstated; loop compiled here |
| Expected value after a failed CAS | Overwritten with the current value | Same | Same ("loads the actual value from *object into *expected") |
| Global frontier counter | `atomic_ref<unsigned, thread_scope_device>::fetch_add(1, relaxed)` | Same through libhipcxx | `atomic_fetch_add_explicit` on `device atomic_uint*` |
| Block-private counter | `atomic_ref<unsigned, thread_scope_block>` on a `__shared__` variable | Same (`__HIP_MEMORY_SCOPE_WORKGROUP`) | `threadgroup atomic_uint`, reset with `atomic_store_explicit`; compiled here |
| Resetting the private counter on overflow (Fig. 18.15 line 29) | Must be an atomic store or a clamp; the plain store is a data race | Same | Only an atomic store can be written |
| Max-unsigned sentinel | `cuda::std::numeric_limits<unsigned int>::max()` (`<cuda/std/limits>`) | Same through libhipcxx; or `UINT_MAX` | `numeric_limits<uint>::max()` or `UINT_MAX`; compiled here |
| Next level's size without a CPU read | Graph conditional WHILE node (CUDA 12.3 and later) | None (no conditional nodes) | `dispatchThreadgroups(indirectBuffer:...)` reads the threadgroup count from a GPU buffer |
| Recording many launches once | CUDA graphs | HIP graphs | No graph API; concurrent encoders with `memoryBarrier`, multi-threaded encoding (llama.cpp) |
| Overlapping dependent launches | PDL, compute capability 9.0 and later | None | None found |
| Forward progress for a persistent kernel | Documented for a cooperative grid | Co-residency enforced at launch; no progress text of AMD's own | Not documented between threadgroups |
| Graph-analytics library | cuGraph (`cugraph.bfs`), direction-optimising | hipGRAPH / rocGRAPH, early access | None found |
| What llama.cpp does with a grid barrier | Cooperative softmax for very long rows, one block per SM | Same source | Not available; multi-dispatch |
