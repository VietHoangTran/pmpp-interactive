# Chapter 6 fact check: CUDA, HIP, Metal

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims the verdict is for the item as a whole and the text says which sub-claim failed or could not be found. A verdict judges the book's statement, or the assumption a course author would most likely bring (stated in the item title), not the vendor.
- **Quote** is copied from the source (HTML flattened to text, GitHub raw files, Apple documentation JSON, PDF text). Line breaks inside quotes are removed.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide "v13.5", CUDA C++ Best Practices Guide "13.5", PTX ISA "9.5", Nsight Compute "v2026.4.0"; HIP docs "HIP 7.17.0" (ROCm Core SDK 10.2.0, source tag `therock-10.2`), ROCm Compute Profiler "3.10.0"; Metal Shading Language Specification "Version 4.1" (PDF created 2026-06-04, re-downloaded today); Apple's Xcode documentation pages (no version printed); local Metal front end "Apple metal version 32023.921 (metalfe-32023.921.6)", target `air64-apple-darwin27.0.0`. Repositories read on their default branches today; the last commit date of each file is given where it is cited.
- Not checked: anything by running code on a GPU. Two Metal observations (C3, C4) come from compiling a test kernel with the local front end and reading the AIR it emitted; that is compile-only evidence, not documentation and not a run. Nothing here was compiled for CUDA or ROCm.

Already settled elsewhere, referred to by item and not repeated:

- `ch05-facts.md` A1 to A9 (device properties, shared-memory size, H100 figures), B5 (wavefront width 32 or 64), C1 to C4 (threadgroup memory, `threadgroup_barrier`, SIMD-group width only from the pipeline, index attributes), C6 (unified memory, shared storage).
- `ch04-facts.md` A4 (H100 SXM5 has 5 HBM3 stacks and 10 memory controllers), A6 (`__syncthreads()` semantics), A11 (zero-cost warp switching; "SMs ... do not perform branch prediction"), A13 and A14 (occupancy, `--resource-usage`), A19 (the 264-block wave example), B3 (`__syncthreads()` on HIP), B10 (AMD occupancy, `rocprofv3 --occupancy`), C2 (SIMD-group mapping is implementation-defined).
- `infra-facts.md` A6, B6, C7 (the `float*` to `float4*` cast; refined below in A11 and C3), A7, B7 (`#pragma unroll`), A10, B10, C13 (shared-memory banks: 32 x 4 B on NVIDIA, 32 or 64 x 4 B on AMD, nothing documented by Apple; refined below in C2), B2 and B4 (no `<cuda/pipeline>` in libhipcxx; HIP `memcpy_async` is synchronous), D1 to D4 (backend directory layouts).

Totals: 19 CONFIRMED, 0 WRONG, 15 PARTLY, 4 COULD NOT VERIFY (38 items: A1 to A17, B1 to B9, C1 to C7, D1 to D4, E1). Every PARTLY and COULD NOT VERIFY item is listed under "Summary of corrections".

---

## Group A: CUDA, and the book's dated or device-specific statements

### A1. DRAM latency "tens of nanoseconds", clocks "sub-nanosecond", latency "has not decreased over time" (pp. 124 to 125)

**Verdict: COULD NOT VERIFY.** No vendor page read today gives DRAM latency in nanoseconds or a trend over time. What vendors document is the same point in cycles or in words:

- NVIDIA Best Practices Guide, section 12.2: "When accessing uncached local or global memory, there are hundreds of clock cycles of memory latency."
- AMD, "Understanding GPU performance": "These Dynamic Random-Access Memory (DRAM) technologies provide very high bandwidth but have relatively long access latency."
- Apple tech talk "Metal Compute on MacBook Pro" (2021): "When reads miss the cache, the calling threads will stall while the read is fetched from RAM and placed into the cache."

For the lesson: keep "tens of nanoseconds" as the book's figure with its page, and let the reader's own numbers come from lab benchmarks. The "has not decreased" claim is the book's.

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 12.2, "Memory Instructions")
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html (section "Memory coalescing theory")
- https://developer.apple.com/videos/play/tech-talks/10580/

### A2. The coalescing rule on "current CUDA devices" (p. 126), and its granularity (notes, "Unclear in the PDF" item 16)

**Verdict: PARTLY.** The rule is confirmed in substance. Two things in the book's picture are simplifications that NVIDIA's current text corrects: the unit is a 32-byte sector, not one burst per warp, and consecutive addresses are the usual way to coalesce, not a requirement.

Quotes (Programming Guide section 2.3.4.1 "Coalesced Global Memory Access"):

- "Global memory is accessed via 32-byte memory transactions. When a CUDA thread requests a word of data from global memory, the relevant warp coalesces the memory requests from all the threads in that warp into the number of memory transactions necessary to satisfy the request".
- "if consecutive threads in the warp request consecutive 4-byte words in memory, then the warp will request 128 bytes of memory total, and this 128 bytes required will be fetched in four 32-byte memory transactions."
- Worst case: "when consecutive threads access data elements that are 32 bytes or more apart from each other in memory ... the warp will be forced to issue a 32-byte memory transaction for each thread ... the memory utilization will only be 128 / 1024 = 12.5%."
- "There is no requirement that consecutive threads access consecutive elements of memory to achieve coalesced memory access, it is merely the typical way coalescing is achieved. Coalesced memory access occurs provided different threads in the warp access elements from the same 32-byte segments of memory in some linear or permuted way. Stated another way, the best way to achieve coalesced memory access is to maximize the ratio of bytes used to bytes transferred."

Quotes (Best Practices Guide section 10.2.1):

- "Global memory loads and stores by threads of a warp are coalesced by the device into as few as possible transactions."
- "For devices of compute capability 6.0 or higher, the requirements can be summarized quite easily: the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal to the number of 32-byte transactions necessary to service all of the threads of the warp."
- "if accesses by the threads of the warp had been permuted within or across the four segments, still only four 32-byte transactions would have been performed".
- Strided access, the case the notes say the book never treats: "A stride of 2 results in a 50% of load/store efficiency since half the elements in the transaction are not used and represent wasted bandwidth. As the stride increases, the effective bandwidth decreases until the point where 32 32-byte segments are loaded for the 32 threads in a warp".
- The guide's own transposed-operand example is the book's Fig. 6.3: for `a[col*TILE_DIM+i]`, "col*TILE_DIM represents a strided access of global memory with a stride of w, resulting in plenty of wasted bandwidth."

Consequences for the coalescing interactive (our arithmetic from the quotes above):

- A warp of 32 lanes loading 4-byte floats touches between 4 sectors (`N[k*Width+col]`, consecutive) and 32 sectors (`N[col*Width+k]` once `Width` is 8 floats or more). The interactive can count sectors per warp load exactly with NVIDIA's documented 32-byte unit, and should label that unit as NVIDIA's.
- Notes, Exercise 3: under the sector rule, a tile row of 16 floats that starts on a 64-byte boundary uses its two sectors fully, so `BLOCK_SIZE` = 16 wastes no bandwidth even though one warp covers two separate address runs. The notes' sketch ("only 32") holds only for the stricter reading "one warp, one run". Present both readings.

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.4.1)
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (sections 10.2.1, 10.2.1.4, 10.2.3.3)

### A3. Alignment of the first address (p. 126)

**Verdict: CONFIRMED.**

- "If sequential threads in a warp access memory that is sequential but not aligned with a 32-byte segment, five 32-byte segments will be requested".
- "Memory allocated through the CUDA Runtime API, such as via cudaMalloc(), is guaranteed to be aligned to at least 256 bytes. Therefore, choosing sensible thread block sizes, such as multiples of the warp size (i.e., 32 on current GPUs), facilitates memory accesses by warps that are properly aligned."
- Row padding for 2D arrays is a documented API: `__host__ cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)`. "The function may pad the allocation to ensure that corresponding pointers in any given row will continue to meet the alignment requirements for coalescing as the address is updated from row to row." Address of element (row, column): `T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;`

Note for the lessons: this row-pitch padding (global memory, keeps each row aligned) is a different thing from the book's shared-memory padding in §6.4 (one extra column, breaks the bank stride). Both are called padding.

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (sections 10.2.1.2 and 10.2.1.3)
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html (`cudaMallocPitch`)

### A4. L1 and L2 caches, cache lines, and "GPU caches are mainly designed to combine" repeated accesses (pp. 125 to 126, 136 to 137)

**Verdict: PARTLY.** Cache lines and their size are documented. The book's statement of what GPU caches are "mainly designed" for is not a vendor statement anywhere I read.

- Nsight Compute Profiling Guide, glossary: sector: "Aligned 32 byte-chunk of memory in a cache line or device memory. An L1 or L2 cache line is four sectors, i.e. 128 bytes." Tag: "The L1 and L2 both have 128 byte cache lines."
- Best Practices Guide: "On devices of compute capability 6.0 or higher, L1-caching is the default, however the data access unit is 32-byte regardless of whether global loads are cached in L1 or not."
- Best Practices Guide on reuse across iterations, which is the book's point that blocks loading the same data may or may not merge: "We can reuse this cache line in subsequent iterations of the loop ... however, when many warps execute on the same multiprocessor simultaneously, as is generally the case, the cache line may easily be evicted from the cache between iterations i and i+1."
- The nearest documented mechanism for merging duplicate requests is AMD's (B3): "Hit-on-miss behavior: If a request arrives for a pending cache line fill, it counts as a hit, improving the effective hit rate."

So: say the caches have 128-byte lines of 32-byte sectors on NVIDIA, that they may merge accesses from blocks running close together, and that how often they do is a measurement. Do not repeat "mainly designed to combine" as fact.

- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html (glossary entries "Sector" and "Tag")
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (sections 10.2.1 and 10.2.3.2)

### A5. Uncoalesced stores, partial writes and ECC (p. 130)

**Verdict: PARTLY.** NVIDIA confirms that scattered writes cost more with ECC, but ties it to GDDR memory and says HBM2 has dedicated ECC with much smaller overhead. The read-modify-write mechanism the book names is not described in the pages read.

- "On devices with GDDR memory, accessing memory in a coalesced way is even more important when ECC is turned on. Scattered accesses increase ECC memory transfer overhead, especially when writing data to global memory."
- "On GPUs with GDDR memory with ECC enabled the available DRAM is reduced by 6.25% to allow for the storage of ECC bits. Fetching ECC bits for each memory transaction also reduced the effective bandwidth by approximately 20% compared to the same GPU with ECC disabled ... HBM2 memories, on the other hand, provide dedicated ECC resources, allowing overhead-free ECC protection."
- Footnote: "As an exception, scattered writes to HBM2 see some overhead from ECC but much less than the overhead with similar access patterns on ECC-protected GDDR5 memory."
- Nsight Compute triage, for stores: "Significantly exceeding expected sector count for the store pattern." Fix: "Widen store granularity and align writes to full sectors".
- AMD documents a related mechanism at L2: "Write combining: To handle partial cache line updates from different CUs, the GPU uses write masks indicating which bytes to update."

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (sections 9.2.1 and 10.2.1)
- https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html

### A6. Corner turning and the padded tile are NVIDIA's own worked examples (pp. 129 to 130, 139 to 140)

**Verdict: CONFIRMED.** Both NVIDIA guides teach exactly the book's two steps: load in the coalesced direction, turn the tile in shared memory, then pad to remove the 32-way conflict that the turn creates.

- Best Practices Guide, `C = AA^T` example: "a warp reads a row of A into a column of a shared memory tile". The kernel writes `transposedTile[threadIdx.x][threadIdx.y] = a[(blockIdx.x*blockDim.x + threadIdx.y)*TILE_DIM + threadIdx.x];` Then: "bank conflicts occur when copying the tile from global memory into shared memory ... every thread of the warp hits the same bank (Recall that w is selected as 32). These many-way bank conflicts are very expensive. The simple remedy is to pad the shared memory array so that it has an extra column": `__shared__ float transposedTile[TILE_DIM][TILE_DIM+1];` "now the stride between threads is w+1 banks (i.e., 33 for current devices), which, due to modulo arithmetic used to compute bank indices, is equivalent to a unit stride."
- Programming Guide sections 2.3.4.2.1 and 2.3.4.2.2 (matrix transpose): "threadIdx.x and threadIdx.y are swapped when they read from smemArray" and "This results in a 32-way bank conflict"; then "A common fix to avoid bank conflicts is to pad the shared memory by adding one to the column dimension of the array" (infra A10).
- The book's claim that shared memory "does not require coalescing" (p. 130): "Aside from memory bank conflicts, there is no penalty for non-sequential or unaligned accesses by a warp in shared memory." (Best Practices Guide section 10.2.3.2.) AMD says the same in B4.

The lesson can cite these as "the same fix, in NVIDIA's own guide" without borrowing NVIDIA's text. The V100 bandwidths printed beside them are NVIDIA's example device and must not appear in a lesson.

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 10.2.3.3, "Shared Memory in Matrix Multiplication (C=AA^T)")
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (sections 2.3.4.2.1 and 2.3.4.2.2)

### A7. Channels "one to eight", the DDR5-6400 arithmetic, and how many buses a CPU and a GPU need (p. 133)

**Verdict: PARTLY.** The bandwidth formula and the DDR5 arithmetic are right, and NVIDIA documents the same formula. "One to eight channels" does not describe current GPUs, which document more. The DDR5 bus structure could not be checked against a primary source.

- NVIDIA's formula, with its own example: "the NVIDIA Tesla V100 uses HBM2 (double data rate) RAM with a memory clock rate of 877 MHz and a 4096-bit-wide memory interface. ... the memory clock rate is converted into Hz, multiplied by the interface width (divided by 8, to convert bits to bytes) and multiplied by 2 due to the double data rate." That is the book's width x clock x 2.
- Book's DDR5-6400: 8 B x 2 x 3.2 GHz = 51.2 GB/s; 200 / 51.2 = 3.9, so four buses; 1000 / 51.2 = 19.5, so 20 buses. Arithmetic checks.
- GPU memory controllers: H100 SXM5 has "10 512-bit memory controllers" (ch04 A4). AMD: "Memory channels: Multiple independent memory controllers (typically 8-16)" (B3).
- COULD NOT VERIFY: that a DDR5 module's 64 bits are split into two independent 32-bit subchannels. JEDEC's DDR5 pages and Micron's DDR5 white paper both refused automated access today (HTTP 403 and "Request Rejected"). A search summary attributed the subchannel statement to Micron, but I did not read it, so it is not recorded as fact.

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 9.2.1 "Theoretical Bandwidth Calculation")
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (section "Memory organization")

### A8. HBM3E: "4.9 GHz/s", 16 64-bit buses, "1 TB/s" (p. 133; notes, "Unclear in the PDF" item 3)

**Verdict: PARTLY.** The structure is right: an HBM3E stack has a 1024-bit interface in 16 independent channels of 64 bits. The unit is wrong, the book's inputs give 1.25 TB/s rather than 1 TB/s, and every figure is per stack, not per GPU.

Vendor pages read today:

- Micron: "HBM3E has 1024 IO pins and the HBM3E pin speed of greater than 9.2Gbps achieves a rate higher than 1.2TB/s". "HBM3E offers 16 independent channels and 32 pseudo channels." "bandwidth of more than 1.2 TB/s per placement".
- Samsung: "Its 12-layer stack offers up to 1,180GB/s bandwidth at 9.2Gbps".
- SK hynix product table: HBM3E "9.6Gbps and higher (1.23TB/s and higher)". The same page's one-line blurb for HBM3E says "The fastest DRAM with Bandwidth reaching 1TB/s", which may be where a round "1 TB/s" comes from.

Arithmetic (ours): the book's inputs, 8 B x 2 x 4.9 GHz x 16 buses = 1254.4 GB/s, which is a per-pin rate of 9.8 Gb/s (4.9 GHz double data rate) over 1024 pins. None of the three vendor pages read lists 9.8 Gb/s; they list more than 9.2 (Micron), 9.2 (Samsung) and 9.6 and higher (SK hynix), giving 1.18 to 1.23 TB/s and up per stack.

For the lesson: "GHz/s" should read "GHz"; the result of the book's own formula is about 1.25 TB/s per stack; a GPU carries several stacks (H100 SXM5 has five HBM3 stacks, ch04 A4). Present all of these as the book's or the vendor's example figures, not as the reader's device.

- https://www.micron.com/products/memory/hbm/hbm3e
- https://semiconductor.samsung.com/dram/hbm/hbm3e/
- https://product.skhynix.com/products/dram/hbm.go

### A9. Interleaving data across channels and banks; the R + 1 rule (pp. 133 to 135)

**Verdict: PARTLY.** Interleaving across channels is documented by AMD, with its granularity. NVIDIA's address-to-channel mapping is not documented in the pages read, Apple documents nothing, and the bank count per channel of real HBM could not be read. The R + 1 rule is the book's model, not a vendor statement.

- AMD: "The L2 consists of multiple independent channels (32 on CDNA GPUs at 256-byte interleaving) that operate in parallel." "Channel organization: Each channel handles a portion of the address space, with addresses interleaved across channels for load balancing."
- Micron, for the channel level of one HBM3E stack: "16 independent channels and 32 pseudo channels" (A8).
- COULD NOT VERIFY: banks per HBM3E channel (the JEDEC standards are behind a login), and any NVIDIA statement of how addresses map to its memory controllers.

So the Fig. 6.9 dealer can show the book's toy mapping and, as a documented real-world counterpart, AMD's 256-byte interleave across 32 L2 channels; it must say that NVIDIA and Apple do not publish theirs.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (section "L2 cache architecture")
- https://www.micron.com/products/memory/hbm/hbm3e

### A10. Occupancy as a memory-bandwidth tool (pp. 135 to 136, 149)

**Verdict: CONFIRMED.** NVIDIA's current triage guide names the book's failure mode and gives the book's three remedies.

- Nsight Compute Compute Triage Guide: "Classic bytes-in-flight starvation: the kernel is not presenting enough outstanding loads to the bus, so the SM stalls on memory even though no memory tier is near its limit." Remedy: "Increase bytes in flight via ILP, vectorized loads, or async copy".
- Best Practices Guide, the counterweight the book gives in Ch. 15: "higher occupancy does not always equate to better performance ... with a high degree of exposed instruction-level parallelism (ILP) it is, in some cases, possible to fully cover latency with a low occupancy."
- Occupancy definitions and APIs: ch04 A13, A14 (NVIDIA), B10 (AMD, including "Memory-bound kernels benefit more from high occupancy"), C5 below (Apple).

- https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 11.3)

### A11. Vector loads: the `float*` to `float4*` cast, 16 bytes per instruction (pp. 138 to 139; refines `infra-facts.md` A6)

**Verdict: PARTLY.** Still true: NVIDIA's reference documents (Programming Guide, Best Practices Guide) do not describe the cast. New today: NVIDIA's developer blog does teach it, with the alignment rule the book leaves out and a cost the book does not mention. And 16 bytes is no longer the widest global load on the newest NVIDIA parts.

NVIDIA technical blog "CUDA Pro Tip: Increase Performance with Vectorized Memory Access" (by Justin Luitjens and Rajeshwari Devaramani; "A version of this blog ran Dec. 4, 2013. It has been updated to reflect behavior on current GPUs."; page metadata dateModified 2025-08-21):

- "CUDA C++ vector types including int2, int4, float2, and float4 enable the compiler to generate these wider memory operations through simple pointer casting with reinterpret_cast."
- Example: `reinterpret_cast<int4*>(d_out)[i] = reinterpret_cast<int4*>(d_in)[i];` with SASS `LDG.E.128` / `STG.E.128`: "This version of the code has reduced the instruction count by a factor of four."
- Alignment: "These instructions require aligned data. Device-allocated memory is automatically aligned to a multiple of the size of the data type, but if you offset the pointer, the offset must also be aligned. For example, reinterpret_cast<int2*>(d_in+1) is invalid because d_in+1 is not aligned to a multiple of sizeof(int2)."
- Cost: "Vectorized loads increase register pressure and reduce parallelism, so scalar loads may remain preferable for kernels that are already register limited or have very low parallelism."

Other documented facts:

- Nsight Compute, sectors per request: "For warps with 32 active threads, the optimal ratios per access size are: 32-bit: 4, 64-bit: 8, 128-bit: 16." So a warp's `float4` load is 512 bytes, 16 sectors, in one instruction.
- Wider than 16 bytes exists in PTX: `ld` with `.v8` for `.f32` on the global space, "// 256-bit load", and "Support for ... .v8.b32/.v4.b64 introduced in PTX ISA version 8.8." / "require sm_100 or higher." CUDA C++ has 32-byte-aligned vector types such as `double4_32a`. There is no `float8` type.
- Deprecation to know about if a lab uses 64-bit vector types: "Note that long4, ulong4, longlong4, ulonglong4, and double4 have been deprecated in CUDA 13, and may be removed in a future release."

For the lesson: present the cast as the book's technique, add that NVIDIA's developer blog (not its programming guide) shows the same idiom and requires the cast pointer to be aligned to the vector size, and show PyTorch's runtime alignment check (D4) as what production code does. The 16-byte figure is right for `float4` on every current GPU; 32-byte loads are an sm_100-and-later PTX feature, not needed for the labs.

- https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html (Memory Tables, "Sectors/Req")
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (section 9.7.10.8, `ld`)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (Tables 48 and 49)

### A12. Shared memory: "latency of a few clock cycles", 32 banks of 4 bytes, the k-way conflict (pp. 139 to 141)

**Verdict: PARTLY.** The bank structure is confirmed (`infra-facts.md` A10). No NVIDIA page read gives a shared-memory latency in cycles; NVIDIA says only "much higher bandwidth and lower latency".

- Best Practices Guide: "Because it is on-chip, shared memory has much higher bandwidth and lower latency than local and global memory - provided there are no bank conflicts". "The hardware splits a memory request that has bank conflicts into as many separate conflict-free requests as necessary, decreasing the effective bandwidth by a factor equal to the number of separate memory requests." Broadcast exception: "when multiple threads in a warp address the same shared memory location, resulting in a broadcast."
- Nsight Compute adds one precision the book does not state: "A shared memory request for a warp does not generate a bank conflict between two threads that access any address within the same 32-bit word (even though the two addresses fall in the same bank)."
- What to measure it with on CUDA: metric `l1tex__data_bank_conflicts_pipe_lsu.sum`, and the derived metric `derived__memory_l1_conflicts_shared_nway`, "Average N-way conflict in L1 per shared memory instruction. A 1-way access has no conflicts and resolves in a single pass."

The k-way count the interactive shows (bank = word index mod 32, conflict degree = threads per bank) matches NVIDIA's model. Label it NVIDIA's; see B4 and C2 for the other two backends.

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (sections 10.2.3 and 10.2.3.1)
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html (section "Shared Memory", metric tables)

### A13. Local arrays are "placed in global memory by default"; constant indices let them live in registers (p. 146; notes, "Unclear in the PDF" item 4)

**Verdict: PARTLY.** The behaviour is confirmed. NVIDIA's name for the place is local memory, which physically lives in device memory, and NVIDIA lists the conditions instead of a default.

- Programming Guide section 2.3.3.4: "Local memory is thread local storage similar to registers and managed by NVCC, but the physical location of local memory is in the global memory space. The 'local' label refers to its logical scope, not its physical location." Variables likely to go there: "Arrays for which it cannot determine that they are indexed with constant quantities, Large structures or arrays that would consume too much register space, Any variable if the kernel uses more registers than available, that is register spilling."
- "local memory accesses have the same latency and bandwidth as global memory accesses and are subject to the same requirements for memory coalescing ... However, local memory is organized such that consecutive 32-bit words are accessed by consecutive thread IDs."
- How to see it, Best Practices Guide section 10.2.4: in PTX a local variable is "declared using the .local mnemonic and accessed using the ld.local and st.local mnemonics", and "the compiler reports total local memory usage per kernel (lmem) when run with the --ptxas-options=-v option."

So the book's §6.6 point (unroll the coarsening loop so `x[c]` has constant indices) is right; the lesson should say "local memory, which is off-chip like global memory" and give the `--ptxas-options=-v` check as the way to see it. AMD's wording matches the book's (B6). Apple's counterpart is in C4.

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.3.4)
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 10.2.4)

### A14. Loop unrolling is automatic; `#pragma unroll N`; no branch prediction (pp. 144 to 145)

**Verdict: CONFIRMED.** Pragma forms and semantics: `infra-facts.md` A7. No branch prediction: ch04 A11 ("SMs issue instructions in order and do not perform branch prediction or speculative execution"). One addition from the Best Practices Guide that the book's §6.6 does not mention: "Sometimes, the compiler may unroll loops or it may optimize out short if or switch blocks by using branch predication instead ... In these cases, no warp can ever diverge. The programmer can also control loop unrolling using the #pragma unroll directive".

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 13.2 "Branch Predication")

### A15. Thread coarsening (pp. 141 to 143)

**Verdict: CONFIRMED** as a technique NVIDIA also recommends; there is no API. Best Practices Guide section 11.4: "kernels with 32x32 or 64x16 threads can be launched with each thread processing four elements of the shared memory array. The approach of using a single thread to process multiple elements of a shared memory array can be beneficial even if limits such as threads per block are not an issue. This is because some operations common to each element can be performed by the thread once, amortizing the cost over the number of shared memory elements processed by the thread."

The wave arithmetic with 264 resident blocks is the book's hypothetical device (ch04 A19); a calculator takes resident blocks from the reader's device.

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 11.4 "Effects of Shared Memory")

### A16. Double buffering today (pp. 146 to 147)

**Verdict: CONFIRMED.** The book's version (two shared buffers, one barrier per iteration) needs only `__syncthreads()` (ch04 A6). NVIDIA now documents double buffering mainly together with asynchronous copies, which is where the reader will meet it in real kernels.

- Programming Guide section 4.13: "To enable concurrency between the producer and the consumer threads, we use double-buffering in shared memory. While consumer warps process data in one buffer, the producer warp asynchronously fetches the next batch of data into the other buffer."
- The copy engines: "LDGSTS (CC 8.0+ ...)" for "smaller, element-wise data transfers", and the Tensor Memory Accelerator (TMA) "for bulk (one-dimensional and multi-dimensional) transfers"; multi-stage prefetch through `cuda::pipeline` with `cuda::memcpy_async`.
- A synchronisation rule that matters for any lab that later swaps in async copies: "By default, each thread only waits for its own LDGSTS copies. Thus, if you use LDGSTS to prefetch some data that will be shared with other threads, a __syncthreads() is necessary after synchronizing with the LDGSTS completion mechanism."
- LDGSTS alignment: "The pointers need to be aligned to 4, 8, or 16 bytes depending on the size of the data being copied. Best performance is achieved when the alignment of both shared memory and global memory is 128 bytes."

For the lesson: the book's toy removes the barrier that guards the write-after-read (false) dependence; the async-copy version goes further and overlaps the next tile's load with this tile's compute. The second is a Ch. 15 topic; mention it as where the idea leads. HIP and Metal counterparts: B8 and C7.

- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html (sections 4.13, 4.13.1, 4.13.1.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html

### A17. The profiler the book cites, and the roofline as stopping rule (pp. 153 to 155)

**Verdict: CONFIRMED.** Reference [1]'s URL, `https://docs.nvidia.com/nsight-compute/index.html`, still resolves (HTTP 200), to "Nsight Compute Documentation", "v2026.4.0". The tool draws rooflines, and NVIDIA's new triage guide states the book's §6.9 procedure almost step for step.

- Roofline sections in the Profiling Guide: `SpeedOfLight_RooflineChart` ("GPU Speed Of Light Roofline Chart (Overview)") and hierarchical versions for double, single, half precision and tensor cores. "The distance from the achieved value to the respective roofline boundary ... represents the opportunity for performance improvement. The closer the achieved value is to the roofline boundary, the more optimal is its performance." "Hierarchical roofline overlays can separate L1, L2, and DRAM bandwidth ceilings."
- Compute Triage Guide, "Apply this iterative optimization plan": "Focus on the single largest current limiter first." "Apply one targeted refactor at a time." "Re-profile immediately and keep or revert based on measured impact." "Validate every change against absolute kernel duration. SOL percentages can drop while the kernel gets faster (e.g. when total work is reduced), so duration is the ground truth, not SOL %." "Re-evaluate the dominant limiter on each new GPU generation". "Stop when expected gains are small relative to complexity and risk."
- Metrics for this chapter's two problems: uncoalesced loads, "Sectors per request ... exceed the ideal for a 128 B cache line", computed as `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`; bank conflicts, "Bank conflicts exceeding ~10% of shared-memory wavefronts." with the fix "Remap shared-memory indexing".

The "duration is the ground truth" sentence is a good companion to the course rule that labs report measured time. The Compute Triage Guide is new since the book's 2023 citation.

- https://docs.nvidia.com/nsight-compute/index.html
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html (section 2.9 "Roofline Charts")
- https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html

---

## Group B: HIP / ROCm

### B1. Coalescing on AMD

**Verdict: CONFIRMED.** AMD documents the same mechanism, explains it with DRAM bursts as the book does, and gives its own granularity: 64- and 128-byte cache lines, and a 64-lane wavefront on CDNA.

- "Coalescing takes advantage of how DRAM is organized internally. When a DRAM address is accessed, the hardware actually fetches or writes a burst: a run of consecutive addresses fetched together in a single transaction. If multiple threads in a warp access addresses that fall into the same burst, those logical accesses can be coalesced into that single transaction." ("Understanding GPU performance", section "Memory coalescing theory")
- "The coalescing hardware in the VMEM unit analyzes addresses from all threads in a warp and groups them into the minimum number of cache line requests." "When consecutive threads access consecutive memory addresses, the hardware can combine all 64 thread requests into as few as 4-8 cache line requests (depending on data size and alignment)." "When threads access widely separated addresses, each thread can generate a separate memory transaction, reducing effective bandwidth by up to 16x or more." ("Hardware implementation", section "Memory coalescing")
- "Align data structures to cache line boundaries (64B and/or 128B)". For RDNA: "128-byte cache lines: Aligning with Wave32 access patterns (32 threads x 4 bytes = 128 bytes)."
- Strided example in the docs: `int idx = threadIdx.x * stride; // Non-coalesced if stride > 1`.

Difference that matters for the interactive: on a 64-wide wavefront, 64 consecutive floats are 256 bytes, so the "one warp, one 128-byte line" picture becomes two lines (or four 64-byte ones). The interactive should take the warp width as a parameter (`ch05-facts.md` B5) and label its segment size per vendor.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html

### B2. Alignment and pitched allocations on HIP

**Verdict: CONFIRMED.**

- Performance guidelines: "Memory hardware loads data in aligned chunks (typically 128 bytes). Using naturally aligned data types ensures each access maps to a single memory transaction". "Padding 2D arrays to multiples of the warp size ensures each row starts at an aligned memory boundary."
- `hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height);` exists at `therock-10.2`, header comment: "Padding may occur to ensure alighnment requirements are met for the given row ... Currently the alignment is set to 128 bytes". Same name pattern as CUDA, so `cudaMallocPitch` would map by prefix.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html (section "Device memory access")
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (line 5158)

### B3. Channels, interleaving and cache merging on AMD

**Verdict: CONFIRMED.** Quoted in A7, A9 and A4: "Memory channels: Multiple independent memory controllers (typically 8-16)"; "The L2 consists of multiple independent channels (32 on CDNA GPUs at 256-byte interleaving)"; "Hit-on-miss behavior: If a request arrives for a pending cache line fill, it counts as a hit". Also: "On AMD GPUs, global memory is backed by HBM (in data-center parts) or GDDR (on many client parts)."

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html

### B4. LDS bank conflicts and padding on AMD (adds to `infra-facts.md` B10)

**Verdict: CONFIRMED.** AMD documents the same +1 padding fix, with the same 32 by 33 array, and the same "no coalescing needed" rule for shared memory.

- Performance guidelines, "Avoid bank conflicts in shared memory": `// Bad: power-of-2 stride causes conflicts` `__shared__ float data[32][32]; float value = data[threadIdx.x][threadIdx.y];` and `// Good: padding avoids conflicts` `__shared__ float data[32][33]; // Extra column`. "Padding arrays by one element shifts addresses to avoid systematic conflicts."
- Programming model: "Unlike global memory, shared memory accesses do not require coalescing, but they can suffer from bank conflicts".
- Bank counts (32 on CDNA 1 to 3, 64 on CDNA 4 and RDNA 2 to 4, 4 bytes each) and the wave64 caveat: `infra-facts.md` B10. Per AMD the LDS serves "16 lanes per cycle", so a wavefront's access takes several cycles even without conflicts; a conflict count computed for NVIDIA's 32 x 32 case is not AMD's count.
- How to measure it: ROCm Compute Profiler's LDS panel reports "Bank Conflict Rate", "the percentage of active LDS cycles that were spent servicing bank conflicts" (B9).

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/cdna/local-data-share.html

### B5. Vector loads on HIP (adds to `infra-facts.md` B6)

**Verdict: PARTLY.** `float4` exists, is 16-byte aligned, and AMD recommends naturally aligned vector types. As for CUDA's reference docs, no AMD page describes the pointer cast.

- Performance guidelines: `float4 data; // 16-byte aligned` under "Use naturally aligned types", and `struct __align__(16) MyStruct { float4 data; };`.
- What a real HIP backend does: llama.cpp's copy helper allows 16-byte copies on HIP for every target (`ggml_cuda_get_max_cpy_bytes()` returns 16 under `GGML_USE_HIP`) with the comment "Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes, especially on AMD." (D1). PyTorch's ROCm path allows 16-element vectors for 1-byte types and has a gfx942-specific nontemporal load (D4).

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html (section "Ensure proper alignment")

### B6. Local arrays and spills on AMD

**Verdict: CONFIRMED,** and AMD's wording is the book's: "Threads have private storage for local variables, spilled registers, and function call stacks, which reside in global memory when on-chip register limits are exceeded." Also: "In extreme cases, the compiler may even “spill” register values into global memory".

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html (section "How register pressure arises")

### B7. Unrolling, instruction-level parallelism and no speculation on AMD

**Verdict: CONFIRMED.** `#pragma unroll` spellings: `infra-facts.md` B7 (AMD calls the numeric form a hint). Instruction-level parallelism as a stated goal: "Exploit instruction-level parallelism within warps". No speculation: "GPUs hide memory and instruction latency through massive hardware multithreading rather than complex CPU techniques such as out-of-order execution and/or speculation."

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html (section "Multiprocessor level")
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (section "Latency hiding through multithreading")

### B8. Double buffering and asynchronous copies on AMD

**Verdict: PARTLY.** The book's two-buffer loop with one `__syncthreads()` compiles unchanged for HIP (ch04 B3). An asynchronous global-to-LDS copy is documented for CDNA 3 and 4 only, through a compiler intrinsic, and I did not confirm that intrinsic in compiler sources.

- "CDNA 3 and CDNA 4 architectures include specialized Data Movement Engine (DME) hardware units ... DMEs perform high-throughput, low-overhead copies between global memory (HBM) and the on-chip memory hierarchy, particularly the Local Data Share (LDS)". "A single warp can issue a DME copy command, immediately resume computation, and later synchronize only when the transfer has completed." "Programmers can access this functionality through asynchronous copy intrinsics in ROCm, such as __builtin_amdgcn_async_work_group_copy".
- At the C++ library level: libhipcxx has no `<cuda/pipeline>` (`infra-facts.md` B2) and HIP's cooperative-groups `memcpy_async` is synchronous (`infra-facts.md` B4). So `cuda::memcpy_async` code from A16 has no drop-in HIP equivalent.
- Not checked: whether `__builtin_amdgcn_async_work_group_copy` exists in the ROCm 10.2 compiler, and on which targets.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (section "Data movement engine (CDNA 3 / CDNA 4)")

### B9. Profilers and the roofline on AMD

**Verdict: CONFIRMED,** with a coverage limit. `rocprofv3` is ch04 B10. ROCm Compute Profiler (`rocprof-compute`, docs 3.10.0) draws rooflines by default and reports LDS bank conflicts.

- "Roofline analysis occurs on any profile mode run, provided --no-roof option is not included." "If you want to focus only on roofline-specific performance data and reduce the time it takes to profile, you can use the --roof-only option." Example: `rocprof-compute profile --name vcopy -- ./vcopy -n 1048576 -b 256`.
- LDS metric "Bank Conflict Rate" (B4).
- Coverage: the "Compatible GPUs/APUs" table ("SoCs ... tested for compatibility") lists AMD Instinct MI100, MI200, MI300 and MI350 Series and Ryzen AI Max APUs as supported, and MI50/MI60 as not supported. No discrete Radeon GPU is listed. A reader on a Radeon card should expect `rocprofv3`, not necessarily this tool.

- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html
- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/reference/compatible-accelerators.html

---

## Group C: Metal

Sources: MSL spec 4.1; Apple's Xcode documentation (JSON form of the pages); three Apple video transcripts on developer.apple.com, cited as Apple's statements with their upload dates; `MTLComputeCommandEncoder` docs; the local Metal front end for the two compile-only observations.

### C1. Does Apple document a coalescing rule?

**Verdict: COULD NOT VERIFY; none is documented.** The MSL spec has no occurrence of "coalesc", "bank" or "cache line" (searched today). What Apple does say is qualitative and framed as cache efficiency:

- WWDC22 "Scale compute workloads across Apple GPUs" (uploaded 2022-06-10): "when the first SIMD-group access the data, the requests are packed in a cache lines. Most of the cache line won't be used, however still occupying space in the cache. Re-arrange the data to fit the access pattern better". "Try to play with the threadgroup size to create groups that map better to your memory layout, like a more rectangular shape". "inefficient memory access patterns always cause high Last Level Cache or Memory Management Unit, or MMU limiters, and pretty low utilizations."
- Xcode docs, "Reducing shader bottlenecks", texture writes only: "Modify textures that cluster writes in space or time (higher spatial or temporal locality), which the GPU can coalesce into fewer write transactions to memory." For buffers: "Access memory that clusters reads in space or time (higher spatial or temporal locality), which can reduce cache misses".
- Tech talk "Metal Compute on MacBook Pro" (2021-12-07): "there's a cache for improving the performance of future buffer reads to the same local area of memory." "Apple silicon GPUs contains a second cache alongside the buffer cache that's dedicated to texture reads."

No cache-line size, no sector size, no per-SIMD-group merging rule. For the course: the coalescing interactive models NVIDIA's documented rule (and AMD's), and the Metal half of the coalescing lab is a measurement with a prediction, whose README says no Apple document predicts the result.

- https://developer.apple.com/videos/play/wwdc2022/10159/
- https://developer.apple.com/documentation/xcode/reducing-shader-bottlenecks
- https://developer.apple.com/videos/play/tech-talks/10580/
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf

### C2. Threadgroup memory: banks, padding and what Apple does advise (adds to `infra-facts.md` C13)

**Verdict: PARTLY.** Still no bank structure documented. But Apple does publish access-pattern advice for threadgroup memory, and on its newest GPUs says threadgroup memory is carved from the L1 cache.

- Xcode docs, "Reducing shader bottlenecks": "Tile memory is local to the GPU and is synonymous with threadgroup memory and imageblock memory." If its counters show a bottleneck: "Align threadgroup memory allocations to a 16-byte boundary." "Reduce a kernel's atomic reads from or writes to threadgroup memory." "Reorder your memory access patterns so that neighboring threads in a quad group write (or read) to neighboring elements in threadgroup memory." "For the threadgroup and imageblock read operations, you can also try removing accesses to the same memory location from multiple threads in the same threadgroup."
- Tech talk "Discover new Metal profiling tools for M3 and A17 Pro" (2023-11-09), Apple family 9: "Registers, threadgroup, tile stack, and other shader core memory types are assigned dynamically from L1 cache, which is then backed by GPU last level cache and device memory."

Two consequences. First, the "quad group neighbours" sentence is the only documented Metal rule about threadgroup access order, and it is about adjacent elements, not banks. Second, Apple's advice to avoid many threads reading the same location is the opposite of NVIDIA's documented broadcast (A12); do not carry NVIDIA's "same word is free" rule over to Metal. The padding lab on Metal measures; it cannot predict.

- https://developer.apple.com/documentation/xcode/reducing-shader-bottlenecks
- https://developer.apple.com/videos/play/tech-talks/111374/

### C3. Vector loads on Metal (adds to `infra-facts.md` C7)

**Verdict: PARTLY.** The documented route is unchanged: declare the buffer `device float4*` (16-byte aligned) or `device packed_float4*` (4-byte aligned). New today: Apple's tools docs recommend vector types for buffer reads, an Apple talk explains an indexing trap that stops loads being vectorised, the bind offset must be aligned, and Apple's own MLX uses a pointer cast. The pointer cast itself is still undocumented.

- Xcode docs, buffer read or write bottlenecks: "Pack scalar values with SIMD types, such as the float4 SIMD type instead of four separate float values." "Use smaller data types, such as the packed_half3 type for positional data instead of float4."
- Tech talk "Metal Compute on MacBook Pro": "When indexing into an array, use signed integer types over unsigned types." "Due to the wrapping characteristics of uint in the shader language spec, this disables vectorized loads." "as the wrapping behavior of int is undefined, the load will be vectorized and likely give a performance improvement." So a Metal vector-load lab should index with `int`, and a scalar baseline indexed with `uint` may be slower for a reason unrelated to the lesson.
- `setBuffer(_:offset:index:)`: "For buffers binding to an argument using the device address space, align the offset to the data type's size. The maximum size for an offset is 16 bytes." For `device float4*` the bind offset must therefore be a multiple of 16.
- Practice in Apple's own code: MLX's `BlockLoader` reads through `*((const device ReadVector*)(&src[i * src_ld]))`, where `ReadVector` is `struct alignas(alignment * sizeof(T))` (D3). llama.cpp's Metal matmul writes `device float4 * D4 = (device float4 *) D;` (D2).
- Compile-only observation (local front end, see header): a kernel containing `((device const float4*)x)[i]` and `((device float4*)z)[i] = ...` compiles with `-Wall` and no warnings, and the emitted AIR contains `load <4 x float>, <4 x float> addrspace(1)* %6, align 16` and `store <4 x float> ...`. That shows what the front end emits, not what the GPU executes (the driver compiles AIR further), and it is not documentation.

For the lesson: keep the documented alternative (declare the parameter as `device float4*`) as the Metal answer; mention that MLX and llama.cpp also cast pointers in MSL and that Apple's spec does not describe it.

- https://developer.apple.com/documentation/xcode/reducing-shader-bottlenecks
- https://developer.apple.com/videos/play/tech-talks/10580/
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbuffer(_:offset:index:)

### C4. Loop unrolling and register promotion on Metal

**Verdict: COULD NOT VERIFY** any documented unroll control. Register promotion has a documented Apple counterpart.

- The MSL spec 4.1 contains no "unroll" and no loop pragma. Its only `#pragma` forms are `#pragma METAL fp math_mode(...)`, `#pragma METAL fp contract(...)` and `#pragma STDC FP_CONTRACT OFF`. It states: "Metal supports the preprocessing directives, as defined by section 16 of the C++17 and C++14 Specification."
- In practice: both MLX and llama.cpp use Clang's loop pragma in MSL, `#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")` (MLX) and `#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)` (llama.cpp) (D2, D3).
- Compile-only observation: the local front end accepts `#pragma unroll 4` and `_Pragma("clang loop unroll(full)")` without warnings and records them in the AIR as `!{!"llvm.loop.unroll.count", i32 4}` and `!{!"llvm.loop.unroll.full"}`. Whether the driver's back end honours that metadata is not documented.
- Register promotion, Apple's wording (Xcode docs, buffer read/write bottlenecks): "GPU functions can increase the read-and-write activity when they use a lot of thread memory or access it with dynamic indexing. This can happen when a function needs to store more data than can fit in the GPU's registers, which forces the GPU to store data to device memory and then read it at a later time." Advice: "Avoid implementations that randomly index into thread-scoped arrays". This is the Metal half of A13.

For the lesson: the `.cu` labs use `#pragma unroll`; the `.metal` labs may use the Clang pragma as MLX and llama.cpp do, labelled "accepted by the compiler, not documented by Apple".

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 1.5.3, 1.6.3)
- https://developer.apple.com/documentation/xcode/reducing-shader-bottlenecks

### C5. Occupancy and bandwidth counters on Metal (corrects `ch04-facts.md` C5)

**Verdict: PARTLY.** ch04-facts C5 said no Apple source documents occupancy. Apple's Xcode documentation does: it defines occupancy, shows it as a counter and gives each shader a maximum theoretical occupancy. What is still missing is any API to query it and any published per-core limit.

- "Finding your Metal app's GPU occupancy": "A GPU has a maximum number of threads that it can execute at the same time. Occupancy is a measurement of how much of this capacity the GPU is using." Compute occupancy "Measures the percentage of the GPU's total thread capacity it's using to execute compute commands." "Each shader has a maximum theoretical occupancy, depending on the amount of internal resources it consumes on the device." Causes of low occupancy include: "Your shaders have exhausted some internal resources, such as thread, threadgroup, or imageblock memory". A warning that fits §6.9: "Neither high nor low occupancy is automatically an indicator of a problem." And: "In rare cases, when overall occupancy is very high, the GPU may execute its workload poorly because threads are competing for space in the GPU's memory caches (also known as cache thrashing)."
- Apple's definition in a talk (2023-11-09): "The number of SIMD groups that are concurrently running on a shader core is called its occupancy."
- Limiter counters ("Reducing shader bottlenecks"): "GPU memory read ... Measures the percentage of the GPU's peak memory read performance", the same for write, "Last level cache", "Buffer read", "Tile memory read" and others. "A limiter counter shows how much time the GPU spends doing work, including stall time."
- Apple's rule of thumb (WWDC22): "For relatively complex kernels, 1K to 2K concurrent threads per shader core is considered a very good occupancy". It is a vendor heuristic from 2022, not a device number for lessons.
- Not found: a roofline chart in Xcode or Instruments. The memory limiters as a percentage of peak are the closest thing to the §6.9 "how close to the limit" check.

- https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy
- https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-a-visual-timeline
- https://developer.apple.com/documentation/xcode/reducing-shader-bottlenecks
- https://developer.apple.com/documentation/xcode/measuring-the-gpus-use-of-memory-bandwidth
- https://developer.apple.com/videos/play/tech-talks/111374/
- https://developer.apple.com/videos/play/wwdc2022/10159/

### C6. The book's DRAM picture on Apple silicon

**Verdict: CONFIRMED** that the picture needs one adjustment. The book draws a discrete GPU with its own DRAM or HBM; Apple GPUs share system memory with the CPU (`ch05-facts.md` C6). Apple's developer documentation gives no bandwidth figures (`infra-facts.md` C13), but Apple's product specification pages do print a unified-memory bandwidth per configuration, for example on the MacBook Pro page "M5 Max with 18-core CPU and 40-core GPU (614GB/s memory bandwidth)". That figure is shared by CPU and GPU, and Apple notes the interaction: "If the GPU's memory bandwidth usage is high when you don't expect it to be, it may impede the CPU's memory accesses." Lessons must not use product figures; the reader's number comes from lab measurements.

- https://www.apple.com/macbook-pro/specs/
- https://developer.apple.com/documentation/xcode/measuring-the-gpus-use-of-memory-bandwidth

### C7. Asynchronous copies for double buffering on Metal

**Verdict: COULD NOT VERIFY any; none is documented.** The MSL spec 4.1 has no occurrence of "async". Double buffering on Metal is therefore the book's synchronous form: two `threadgroup` arrays and one `threadgroup_barrier(mem_flags::mem_threadgroup)` per iteration (`ch05-facts.md` C2), which is expressible as written and fits in the 32 KB threadgroup limit (`ch05-facts.md` C1) for two 32 by 33 `float` tiles (8,448 bytes, our arithmetic).

Real code for contrast: MLX's main GEMM loop is single-buffered with two barriers per K step, exactly the book's "before" loop (D3).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf

---

## Group D: How llama.cpp, MLX and PyTorch do this chapter's techniques

Read from raw files on the default branches on 2026-11-20. Line references are to today's files and will drift.

### D1. llama.cpp, CUDA and HIP backend

**Verdict: CONFIRMED.**

- 16-byte copies through a cast, `ggml/src/ggml-cuda/common.cuh` (last commit 2026-11-19): `ggml_cuda_memcpy_1<nbytes, alignment>` copies through `int2` for 8 bytes and `((int4 *) dst)[i] = ((const int4 *) src)[i];` for 16. Its comment: "Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes, especially on AMD." The width limit: `ggml_cuda_get_max_cpy_bytes()` returns 16 on HIP and on CUDA Volta and later, 8 before Volta, with the comment "Maximum number of bytes that can be copied in a single instruction." A static assert warns that copying more per call "the reads and writes may not be coalesced". There is also `ggml_cuda_is_aligned(tensor, alignment)`, which checks the base pointer and the strides.
- Shared-memory padding against bank conflicts, `ggml/src/ggml-cuda/mmq.cuh` (last commit 2026-11-19): "To avoid shared memory bank conflicts each block is padded with 16 bytes. This padding is also used to store block scales/partial sums." and "The final tile size in K direction is padded to avoid shared memory bank conflicts, in terms of 32 bit elements that means K % 2 == 1 for dp4a or K % 8 == 4 for mma." It is enforced: `static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0) % 8 == 4, "Wrong padding.");`. The book's "+1 column" becomes "stride mod 8 is 4" once each thread moves more than 4 bytes.
- Double buffering with asynchronous copies, `ggml/src/ggml-cuda/fattn-mma-f16.cuh` (last commit 2026-11-17): config field `nstages_target`, "Number of pipeline stages to use ideally, 1 == always load data synchronously, 2 == preload data if there is hardware support." The copies use `cp_async_cg_16` from `cp-async.cuh` (last commit 2026-11-12), whose comment repeats NVIDIA's rule from A16: "Makes each thread wait until its asynchronous data copies are done. This does NOT provide any additional synchronization. In particular, when copying data with multiple warps a call to __syncthreads will be needed."

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-mma-f16.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/cp-async.cuh

### D2. llama.cpp, Metal backend

**Verdict: CONFIRMED.**

- `ggml/src/ggml-metal/kernels/common.h` (last commit 2026-08-24): `#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)` and `#define N_SIMDWIDTH 32 // assuming SIMD group size is 32`.
- `ggml/src/ggml-metal/kernels/mul_mm.metal` (last commit 2026-11-05): vector stores through a cast, `device float4 * D4 = (device float4 *) D;` and `threadgroup float4 * C4 = (threadgroup float4 *) C;`; and an alignment guard that cites Apple's spec: "Element-wise reads when K is not aligned (nb01 not aligned for half4x4/float4x4). MSL spec Table 2.5: half4x4 requires 8-byte alignment. When K is odd, nb01 = K*2 is not 8-byte aligned, so odd-row pointers are misaligned."

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/common.h
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mm.metal

### D3. MLX, Metal GEMM ("steel")

**Verdict: CONFIRMED.**

- `mlx/backend/metal/kernels/steel/defines.h` (last commit 2025-11-19): `#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")` and `#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")`.
- `mlx/backend/metal/kernels/steel/gemm/gemm.h` (last commit 2024-05-08): threadgroup tiles are padded by 16 bytes, `STEEL_CONST short tgp_padding_a = 16 / sizeof(T);` (same for B), and the tile shape follows the operand's storage, `transpose_a ? BK * (BM + tgp_padding_a) : BM * (BK + tgp_padding_a)`. Transposed operands are handled by loading along the contiguous direction and indexing the tile accordingly, which is corner turning plus padding.
- The main K loop is single-buffered with two barriers per step: `threadgroup_barrier(mem_flags::mem_threadgroup);` then the two loads, then a second `threadgroup_barrier(mem_flags::mem_threadgroup);` then `mma_op.mma(As, Bs);`. That is the book's §6.7 "before" loop.
- `mlx/backend/metal/kernels/steel/gemm/loader.h` (last commit 2026-08-04): `struct alignas(alignment * sizeof(T)) ReadVector { uint8_t v[sizeof(T) * vec_size]; };` and the unchecked load `*((threadgroup ReadVector*)(&dst[i * dst_ld])) = *((const device ReadVector*)(&src[i * src_ld]));`. The loader computes `thread_idx(simd_group_id * 32 + simd_lane_id)`, which assumes a 32-wide SIMD-group (`ch04-facts.md` C2 says the width is not documented).

- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/defines.h
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/gemm.h
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/loader.h

### D4. PyTorch, CUDA and ROCm elementwise kernels

**Verdict: CONFIRMED.** `aten/src/ATen/native/cuda/MemoryAccess.cuh` (last commit 2026-10-31) is the production form of the book's Fig. 6.12: the same cast, but chosen at run time from the pointer's actual alignment.

- "// aligned vector generates vectorized load/store on CUDA" `template<typename scalar_t, int vec_size> struct alignas(sizeof(scalar_t) * vec_size) aligned_vector { scalar_t val[vec_size]; };` and `auto *from = reinterpret_cast<const vec_t *>(base_ptr);`.
- `can_vectorize_up_to<scalar_t>(const char *pointer)` returns 8, 4, 2 or 1 by testing `address % vec8_alignment == 0` and so on. Under `USE_ROCM` it also allows 16 for 1-byte types, and `load_vector` has a gfx942 branch using `__builtin_nontemporal_load` with the comment "This nontemporal vectorized load is tuned for gfx942 (CDNA3, Wave64). Do NOT extend to gfx1250 (GFX12.5, Wave32) without first validating".

So the bridge for the reader: PyTorch's elementwise ops already do the vector-load-with-scalar-fallback that the book leaves to Exercise 4, deciding per tensor from its pointer alignment.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MemoryAccess.cuh

---

## Group E: The chapter's arithmetic, rechecked

### E1. Worked numbers and the errata in the notes

**Verdict: PARTLY.** Our arithmetic, not a vendor statement. Two printed expressions are wrong; the rest check.

| Book page | As printed | Recomputed | Status |
|---|---|---|---|
| 133 | 8 B x 2 x 3.2 GHz = 51.2 GB/s; four buses for 200 GB/s, 20 for 1000 GB/s | 51.2; 200/51.2 = 3.9; 1000/51.2 = 19.5 | correct |
| 133 | HBM3E "4.9 GHz/s", 16 x 64-bit, "1 TB/s" | 8 B x 2 x 4.9 x 16 = 1254.4 GB/s | unit wrong; result about 1.25 TB/s per stack (A8) |
| 134 | R = 20: utilisation 1/21 = 4.8 %; 16 GB/s gives 0.76 GB/s; 21 banks | 4.76 %; 0.762 GB/s; R + 1 = 21 | correct (model, not vendor fact) |
| 137 | 8 x 8 matrices with 2 x 2 blocks "use all four channels" | With the Fig. 6.9 mapping, element 8r + c is in burst 4r + floor(c/2); in phase ph every M load is in burst 4r + ph, so channel ph | all four channels across the run, one per phase; agrees with notes item 8 |
| 139 | two vector loads replace eight scalar loads, 75 % fewer | 1 - 2/8 = 0.75 | correct |
| 140 | 32-way conflict for `a[threadIdx.x][threadIdx.y]`, gone with 33 columns | bank = 32i mod 32 = 0; 33i mod 32 = i | correct; NVIDIA's guide says the same (A6) |
| 140 to 141 | multiply by 32 is "i >> 5"; by 33 is "i >> 5 + i" | `i << 5`; `(i << 5) + i` (in C, `i >> 5 + i` parses as `i >> (5 + i)`) | wrong as printed |
| 143 | 1056 blocks / 264 = 4 waves; x8 coarsening gives 132 = 0.5 wave; 1584 = 6 waves; x4 gives 396 = 1.5 waves | same | correct |
| 147 | "eliminate the second barrier" | the barrier removed is the first one, after the read (write-after-read) | wrong word, as notes item 2 says |
| 155 (Ex. 5) | banks for strides 32, 31, 24, 16, 12, 8, 7 | gcd rule: 32/g banks, g threads each | notes' answers check |

---

## Summary of corrections

Every WRONG, PARTLY or COULD NOT VERIFY item, one line each:

- A1 (could not verify): no vendor gives DRAM latency in nanoseconds or says it has not fallen; NVIDIA says "hundreds of clock cycles". Keep the book's figure as the book's.
- A2: the coalescing unit on NVIDIA is the 32-byte sector, so a perfectly coalesced warp of 4-byte loads is four transactions, not one; NVIDIA states that consecutive addresses are "merely the typical way", and that a stride of 2 gives 50 % efficiency. Exercise 3 has a second valid reading under the sector rule (16-wide tiles waste nothing when aligned).
- A4: L1 and L2 lines are 128 bytes of four 32-byte sectors on NVIDIA; "GPU caches are mainly designed to combine" duplicate accesses is the book's claim, not a vendor's.
- A5: the ECC penalty for scattered (especially write) access is documented for GDDR; NVIDIA says HBM2's ECC is nearly free except for scattered writes. The read-modify-write mechanism is not described.
- A7: "one to eight channels" understates GPUs (H100 SXM5 has 10 memory controllers; AMD says 8 to 16); NVIDIA documents the same width x clock x 2 formula; DDR5's two 32-bit subchannels could not be read from a primary source.
- A8: "4.9 GHz/s" should be GHz; the book's own inputs give about 1.25 TB/s, not 1 TB/s; vendors list 1.18 to 1.23+ TB/s per HBM3E stack at 9.2 to 9.6 Gb/s per pin; all of it is per stack.
- A9: AMD documents 256-byte interleaving across 32 L2 channels on CDNA; NVIDIA and Apple publish no address mapping; HBM bank counts could not be read.
- A11 (refines infra A6): NVIDIA's developer blog, updated 2025, does teach the `reinterpret_cast` to `int4*`/`float4*` idiom, requires the pointer aligned to the vector size, and warns of register pressure; the reference guides still do not. PTX has 256-bit `.v8.f32` global loads on sm_100 and later; `long4`, `double4` and friends are deprecated in CUDA 13.
- A12: no NVIDIA page gives shared-memory latency as "a few cycles"; threads touching the same 32-bit word do not conflict.
- A13: NVIDIA calls the place "local memory" (physically device memory) and lists when arrays go there; check with `--ptxas-options=-v`. The book's "global memory" matches AMD's wording.
- B5: AMD recommends naturally aligned `float4` but, like NVIDIA's reference docs, does not describe the cast.
- B8: an async global-to-LDS copy is documented only for CDNA 3 and 4 (`__builtin_amdgcn_async_work_group_copy`, not checked in compiler sources); libhipcxx has no `cuda::pipeline` and HIP's `memcpy_async` is synchronous.
- C1 (could not verify): Apple documents no coalescing rule, cache-line size or sector size; only cache-locality advice (WWDC22) and a texture-write "coalesce" sentence.
- C2 (refines infra C13): still no banks, but Apple advises 16-byte-aligned threadgroup allocations and neighbouring quad-group threads on neighbouring elements, and advises against many threads reading the same location (unlike NVIDIA's free broadcast); on Apple family 9 threadgroup memory comes from L1.
- C3: Apple recommends `float4` packing for buffer reads, says `uint` indices can disable vectorised loads (use `int`), and requires a bind offset aligned to the element type for `device` buffers; MLX and llama.cpp cast pointers in MSL although the spec does not describe it.
- C4 (could not verify): no unroll control is documented in MSL; the Clang loop pragma (used by MLX and llama.cpp) and `#pragma unroll N` are accepted by the local front end, compile-only.
- C5 (corrects ch04 C5): Apple's Xcode docs define occupancy, show a Compute occupancy counter and a per-shader Max Theoretical Occupancy; there is still no API and no published per-core limit; no roofline chart was found.
- C7 (could not verify): Metal documents no asynchronous copy into threadgroup memory; double buffering is the synchronous two-array form.
- E1: the shift is printed the wrong way (`i << 5`, `(i << 5) + i`), the HBM3E result is about 1.25 TB/s, and "second barrier" means the first.

Status changes worth knowing even where the verdict is CONFIRMED:

- A17: reference [1] still resolves (Nsight Compute v2026.4.0), which now has a Compute Triage Guide that states the §6.9 loop, including "duration is the ground truth" and "Stop when expected gains are small relative to complexity and risk".
- A16: NVIDIA documents double buffering today mostly with asynchronous copies (LDGSTS on compute capability 8.0 and later, TMA), where each thread waits only for its own copies and `__syncthreads()` is still needed.
- B4: AMD's own guide uses the same `[32][33]` padding fix as NVIDIA's.
- B9: ROCm Compute Profiler draws rooflines by default but lists only Instinct GPUs and Ryzen AI Max APUs as tested.
- D3: MLX's main Metal GEMM is single-buffered with two barriers per step, a real-world example of the book's "before" loop.

## Where the backends really differ

One line per cell, documented facts only unless marked. Nothing here was run on a device.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Coalescing rule documented | Yes: a warp's addresses are served by 32-byte transactions; aim to use every byte fetched | Yes: a wavefront's addresses grouped into the fewest cache-line requests | No rule; cache-locality advice only |
| Unit to count in the interactive | 32-byte sector; 128-byte L1/L2 line | 64-byte and/or 128-byte cache line | Not documented; measure |
| Lanes issuing one load | 32 | 64 on CDNA (Instinct), 32 on RDNA (Radeon) | SIMD-group width from the pipeline; not documented |
| Allocation alignment | `cudaMalloc` at least 256 bytes | Not checked for `hipMalloc`; `hipMallocPitch` aligns rows to 128 bytes | `device` bind offset aligned to the element type |
| Pitched 2D allocation | `cudaMallocPitch` | `hipMallocPitch` | None documented |
| Address interleave across channels | Not documented | 256 bytes across 32 L2 channels (CDNA) | Not documented |
| Shared-memory banks | 32 x 4 bytes | 32 (CDNA 1 to 3) or 64 (CDNA 4, RDNA 2 to 4) x 4 bytes | Nothing documented |
| Padding fix in vendor docs | `[TILE_DIM][TILE_DIM+1]`, `[32][33]` | `[32][33]` | None; "neighbouring threads in a quad group to neighbouring elements" |
| Many threads reading one shared word | Broadcast, no conflict | Broadcast "handled efficiently" | Apple advises removing such accesses |
| 16-byte vector load | `float4`; cast shown on NVIDIA's blog, not in the guides | `float4` (16-byte aligned); cast not documented | Declare `device float4*` / `packed_float4*`; cast compiles but is undocumented |
| Wider than 16 bytes | PTX `.v8.f32` 256-bit loads, sm_100 and later | Not checked | Not documented |
| Loop unrolling control | `#pragma unroll [N]` | `#pragma unroll [N]` (a hint) | Not documented; Clang loop pragma used by MLX and llama.cpp |
| Where non-constant-indexed local arrays go | "local memory", physically device memory | "reside in global memory" | "device memory" (Xcode docs) |
| Async global to shared copy | LDGSTS (8.0+), TMA, `cuda::memcpy_async` | DME on CDNA 3/4 via an intrinsic; CG `memcpy_async` synchronous | None documented |
| Barrier for the double-buffer loop | `__syncthreads()` | `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| Occupancy | Defined; API `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | Defined; API `hipOccupancyMaxActiveBlocksPerMultiprocessor` | Defined in Xcode docs; counter and max theoretical value in tools; no API |
| Profiler | Nsight Compute | `rocprofv3`; ROCm Compute Profiler (Instinct, Ryzen AI Max) | Xcode Metal debugger, Instruments |
| Roofline in the tool | Yes (Speed Of Light roofline, hierarchical L1/L2/DRAM) | Yes (`rocprof-compute`, default) | None found; memory limiters as % of peak |
| Coalescing metric | Sectors per request (`l1tex__t_sectors.../l1tex__t_requests...`) | Not checked by name | Last level cache and MMU limiters (indirect) |
| Bank-conflict metric | `l1tex__data_bank_conflicts_pipe_lsu.sum`, N-way derived metric | LDS "Bank Conflict Rate" | Tile memory read/write limiters (indirect) |
