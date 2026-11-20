# Chapter 4 fact check: CUDA, HIP, Metal

Checked on 2026-11-20 against the live vendor sources listed under each item.

How to read this file:

- **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims, the verdict is for the item as a whole and the text says which sub-claim failed.
- **Quote** is copied from the source (HTML docs flattened to text, GitHub raw files, Apple's documentation JSON, PDF text, local SDK headers). Line breaks inside quotes are removed.
- "ch05 A5" and similar point at an item in `notes/today/ch05-facts.md`. Anything settled there is not repeated here.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide and Runtime API reference "v13.5" (pages last updated Oct 31 and Nov 02, 2026); Nsight Compute docs "v2026.4.0"; HIP docs "HIP 7.17.0 Documentation" and source at tag `therock-10.2` of `ROCm/rocm-systems`; Metal Shading Language Specification "Version 4.1", dated 2026-06-04 ("MSL spec"); Metal headers from the macOS 27.0 SDK on this machine; NVIDIA Tesla V100 whitepaper WP-08608-001_v1.1 (August 2017).
- Not checked: anything by running code on hardware. Every statement is from documentation, headers or project source. No `notes/today/infra-facts.md` existed when this was written.
- Page numbers ("p. 85") are book pages as recorded in `notes/ch04.md`.

Totals: 30 CONFIRMED, 1 WRONG, 9 PARTLY, 4 COULD NOT VERIFY (44 items: 20 CUDA, 12 HIP, 7 Metal, 4 project, 1 arithmetic). Failed or missing sub-claims are collected at the end.

---

## Group A: CUDA and the book's dated statements

### A1. Hardware vocabulary: SM, GPC, "streaming processor"

**Verdict: PARTLY.** SMs grouped into GPCs is NVIDIA's current model. Two wording differences from the book: the programming-model page expands GPC as "Graphics Processing Clusters" (the book says "GPU Processing Clusters"), and the current guide never uses the term "streaming processor".

Quotes (programming guide, section 1.2):

- "the GPU can be considered to be a collection of Streaming Multiprocessors (SMs) which are organized into groups called Graphics Processing Clusters (GPCs). Each SM contains a local register file, a unified data cache, and a number of functional units that perform computations."
- "The sizes of different types of memory and the number of functional units within an SM can vary across GPU architectures."

Notes:

- NVIDIA is not consistent about the expansion. The same guide, section 2.1.10, says "co-scheduled to execute simultaneously on a single GPU Processing Cluster (GPC)" and the Hopper blog says "GPU processing clusters (GPCs)". Either is NVIDIA's wording; do not present one as the only name.
- A search of the five guide pages read for this file (programming model, intro to CUDA C++, writing kernels, advanced kernel programming, compute capabilities) finds "streaming processor" zero times. The guide says "functional units"; the Hopper blog and the Volta whitepaper say "FP32 CUDA Cores". The book's footnote that streaming processors are "sometimes called CUDA cores" matches the blog's term.

- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (section 2.1.10)
- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

### A2. H100 counts: 132 SMs, 128 per SM, 16,896 in total, 8 GPCs

**Verdict: PARTLY.** All four numbers are NVIDIA's, but only for the H100 SXM5 variant. The PCIe H100 has 114 SMs.

Quotes (Hopper architecture blog):

- Full GH100 chip: "8 GPCs, 72 TPCs (9 TPCs/GPC), 2 SMs/TPC, 144 SMs per full GPU" and "128 FP32 CUDA Cores per SM, 18432 FP32 CUDA Cores per full GPU".
- H100 SXM5: "8 GPCs, 66 TPCs, 2 SMs/TPC, 132 SMs per GPU" and "128 FP32 CUDA Cores per SM, 16896 FP32 CUDA Cores per GPU".
- H100 PCIe: "7 or 8 GPCs, 57 TPCs, 2 SMs/TPC, 114 SMs per GPU" and "128 FP32 CUDA Cores/SM, 14592 FP32 CUDA Cores per GPU".
- "Figure 3 shows a full GH100 GPU with 144 SMs. The H100 SXM5 GPU has 132 SMs, and the PCIe version has 114 SMs."

This answers item 7 of "Unclear in the PDF" (132 is not divisible by 8) as far as NVIDIA goes: there is a level the book leaves out, the TPC (texture processing cluster) of 2 SMs, and the shipped SXM5 part uses 66 of the full chip's 72 TPCs. NVIDIA does not say how the 66 TPCs are spread over the 8 GPCs, so the GPCs cannot all hold the same number of SMs (66 / 8 is not an integer) and the exact split is COULD NOT VERIFY.

- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

### A3. "An H100 SM has four processing blocks of 32 streaming processors" (p. 76)

**Verdict: COULD NOT VERIFY.** The Hopper blog text gives 128 FP32 CUDA cores per SM and never mentions processing blocks. "Processing block" is a real NVIDIA term, but the only NVIDIA text found that uses it is the 2017 Volta whitepaper, with different numbers for those older chips.

Quote (Volta whitepaper, page 11): "the GP100 SM is partitioned into two processing blocks, each with 32 FP32 Cores, 16 FP64 Cores, an instruction buffer, one warp scheduler, two dispatch units, and a 128 KB Register File. The GV100 SM is partitioned into four processing blocks, each with 16 FP32 Cores, 8 FP64 Cores, 16 INT32 Cores, two of the new mixed-precision Tensor Cores for deep learning matrix arithmetic, a new L0 instruction cache, one warp scheduler, one dispatch unit, and a 64 KB Register File."

What can be said safely: a processing block has its own warp scheduler (the quote above), and 128 / 4 = 32 is consistent with the book. Present "four blocks of 32" as the book's statement about its example device. The H100 whitepaper PDF is behind a form on resources.nvidia.com and was not fetched.

- https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

### A4. Device memory: GDDR before, HBM from Pascal on, several memory controllers

**Verdict: CONFIRMED,** with one qualifier: on H100 only the SXM5 variant has HBM3; the PCIe variant has HBM2e.

Quotes (Hopper blog):

- "The NVIDIA P100 was the world's first GPU architecture to support the high-bandwidth HBM2 memory technology." (P100 is the Pascal data-center part.)
- "The H100 SXM5 GPU is the world's first GPU with HBM3 memory".
- H100 SXM5: "80 GB HBM3, 5 HBM3 stacks, 10 512-bit memory controllers". H100 PCIe: "80 GB HBM2e, 5 HBM2e stacks, 10 512-bit memory controllers".

Not checked: which memory type consumer GPUs use today. The book's sentence is about what a GPU "may use", which the quotes support.

- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

### A5. Blocks are assigned whole to one SM, in any order, as resources free up

**Verdict: CONFIRMED.**

Quotes (programming guide):

- "All threads of a thread block are executed by a single SM and, in most cases [1], run to completion on that SM. There is no guarantee of scheduling between thread blocks".
- "Different thread blocks within the grid are scheduled among the available SMs and may be executed in any order. In short, the CUDA programming model requires that it be possible to execute thread blocks in any order, in parallel or in series."
- "The CUDA programming model enables arbitrarily large grids to run on GPUs of any size, whether it has only one SM or thousands of SMs."
- Section 2.3.7: "As long as SMs have sufficient hardware resources unoccupied by other thread blocks, the scheduler will continue assigning thread blocks to SMs. If at some point no SM has the capacity to accept another thread block, the scheduler will wait until the SMs complete previously assigned thread blocks."
- "The details of which thread blocks are scheduled to execute on which SMs cannot be controlled or queried by the application and no ordering guarantees are made by the scheduler".

One exception the book does not mention, footnote [1]: "In certain situations when using features such as CUDA Dynamic Parallelism, a thread block may be suspended to memory. ... This is not common."

"Transparent scalability", "wave" and "tail effect" are the book's terms. None of the three appears in the NVIDIA pages read for this file. (AMD's HIP docs do use "waves" in this sense, see B2.)

- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html (section 1.2.2.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.7)

### A6. `__syncthreads()`: semantics and the rule for conditionals

**Verdict: CONFIRMED.** The book's rule (all threads of the block or none) is NVIDIA's rule, and NVIDIA calls a violation undefined behaviour, as the book does for Fig. 4.4.

Quotes (programming guide, section 5.4.4.1):

- Signatures: `void __syncthreads();` `int __syncthreads_count(int predicate);` `int __syncthreads_and(int predicate);` `int __syncthreads_or(int predicate);`
- "__syncthreads*() wait until all non-exited threads in the thread block simultaneously reach the same __syncthreads*() intrinsic call in the program or exit."
- "The __syncthreads*() intrinsics are permitted in conditional code, but only if the condition evaluates uniformly across the entire thread block. Otherwise, execution may hang or produce unintended side effects."
- Valid example in the guide: `if (blockIdx.x > 0) { // CORRECT, uniform condition across all block threads` followed by `__syncthreads();`. Invalid example: `if (threadIdx.x > 0) { // WRONG, non-uniform condition` followed by `__syncthreads(); // Undefined Behavior`.
- Section 2.1: "__syncthreads() is expected to be lightweight. __syncthreads() only synchronizes the threads within a single thread block."

Two details beyond the book:

- "the same __syncthreads*() intrinsic call in the program": two calls at two places are two barriers. This is the exact reason Fig. 4.4 is wrong.
- "non-exited threads ... or exit": a thread that has returned does not hold the others at the barrier. The book's rule "must be executed by all threads of the block" is the safe reading; NVIDIA's wording is narrower.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.4.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html

### A7. Cluster-wide and grid-wide barriers through Cooperative Groups

**Verdict: CONFIRMED.** Cluster barrier: `this_cluster()` then `cluster.sync()`, compute capability 9.0 or higher (clusters themselves: ch05 A6). Grid barrier: `this_grid()` then `grid.sync()`, and the kernel must be launched with `cudaLaunchCooperativeKernel`, not `<<<>>>`. The grid must fit on the device all at once.

Quotes:

- "All the thread blocks in the cluster are guaranteed to be co-scheduled to execute simultaneously on a single GPU Processing Cluster (GPC) and allow thread blocks in the cluster to perform hardware-supported synchronization using the cooperative groups API cluster.sync()."
- "The this_cluster() assumes a 1x1x1 cluster when a non-cluster grid is launched. Requires Compute Capability 9.0 or greater".
- "synchronizing the entire grid requires using the cudaLaunchCooperativeKernel runtime launch API."
- "To guarantee co-residency of the thread blocks on the GPU, the number of blocks launched needs to be carefully considered." The guide's example launches `deviceProp.multiProcessorCount` blocks, or `numBlocksPerSm * multiProcessorCount` with the count taken from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
- "It is good practice to first ensure the device supports cooperative launches by querying the device attribute cudaDevAttrCooperativeLaunch". "Only devices with compute capability of 6.0 and higher are supported."
- New since the book's reference [1] (guide release 12.3): "Multi-device launch APIs and related references for Cooperative Groups have been removed as of CUDA 13." The CUDA 13.0 release notes list `cudaLaunchCooperativeKernelMultiDevice`, `this_multi_grid` and `multi_grid_group` as removed.

This is what the book means by "several restrictions" on the grid-wide barrier (p. 70): a special launch call, and a grid no larger than what is resident at once.

- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html (sections 4.4.3 and 4.4.8)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html (section 5.7.3.6.2.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (section 2.1.10)
- https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html

### A8. Warps: 32 threads, cut from the block in a documented order

**Verdict: CONFIRMED.** Warp size is 32 on every compute capability the guide lists (ch05 A5, Table 33). The cutting rule the book gives (linear order with x fastest, then consecutive runs of 32, last warp padded) is documented.

Quotes:

- "Within a thread block, threads are organized into groups of 32 threads called warps."
- "Warp lanes are numbered 0 to 31 and threads from a thread block are assigned to warps in a predictable fashion".
- "The threads of a block are linearized predictably: the first index x moves the fastest, followed by y and then z. This means that in the linearization of a thread indices, consecutive values of threadIdx.x indicate consecutive threads, threadIdx.y has a stride of blockDim.x, and threadIdx.z has a stride of blockDim.x * blockDim.y. This affects how threads are assigned to warps".
- "The way a block is partitioned into warps is always the same; each warp contains threads of consecutive, increasing thread IDs with the first warp containing thread 0."
- Number of warps in a block: ceil(T / Wsize), with "Wsize is the warp size, which is equal to 32".
- "It is legal to use any number of threads, but when the total is not a multiple of 32, the last warp of the thread block will have some lanes that are unused throughout execution."
- `warpSize` in device code: "A run-time value defined as the number of threads in a warp, commonly 32."
- The guide adds: "The use of multi-dimensional thread blocks and grids is for convenience only and does not affect performance." Block shape changes which threads share a warp; it does not change the cost of launching.

So a thread's warp in a block is `(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) / warpSize`, and its lane is the same expression modulo `warpSize`. The formula is the quoted rule written out; NVIDIA does not print it in this form.

The book's "may change in future" still stands as NVIDIA's own caution: the reference says "commonly 32", not "always 32". On other vendors the width differs (B6, C2).

- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html (section 1.2.2.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (section 3.2.2.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (built-in variables)

### A9. SIMT and control divergence

**Verdict: CONFIRMED.** NVIDIA's name for the effect is "warp divergence" or "branch divergence"; the book's "control divergence" is the same thing.

Quotes:

- "All threads in the warp execute the same instruction simultaneously. If some threads within a warp follow a control flow branch in execution while others do not, the threads which do not follow the branch will be masked off while the threads which follow the branch are executed. ... When different threads in a warp follow different code paths, this is sometimes called warp divergence. It follows that utilization of the GPU is maximized when threads within a warp follow the same control flow path."
- "A warp executes one common instruction at a time, so full efficiency is realized when all 32 threads of a warp agree on their execution path. If threads of a warp diverge via a data-dependent conditional branch, the warp executes each branch path taken, disabling threads that are not on that path. Branch divergence occurs only within a warp; different warps execute independently".
- Active and inactive: "The threads of a warp that are participating in the current instruction are called the active threads, whereas threads not on the current instruction are inactive (disabled). Threads can be inactive for a variety of reasons including having exited earlier than other threads of their warp, having taken a different branch path than the branch path currently executed by the warp, or being the last threads of a block whose number of threads is not a multiple of the warp size."
- SIMT against SIMD: "SIMD vector organizations expose the SIMD width to the software, whereas SIMT instructions specify the execution and branching behavior of a single thread." And the comparison NVIDIA draws for how much to care: "In practice, this is analogous to the role of cache lines: the cache line size can be safely ignored when designing for correctness but must be considered in the code structure when designing for peak performance."
- A warning the simulator's `<Simplification>` should carry: "Exploiting knowledge of how warp execution is actually mapped to real hardware is discouraged. ... Hardware may optimize masked lanes in ways that are transparent to the program so long as the programming model is followed."

Not documented by NVIDIA: any number for what a divergent pass costs. "Each branch path taken" is the model; the price on a given device is something to measure. Also not documented: when the compiler replaces a short branch with a select so that no divergence happens at all. A lab that measures divergence has to make both paths expensive enough that this cannot occur (our caution, not a vendor statement).

- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html (section 1.2.2.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (section 3.2.2.1)

### A10. Sequential passes before Volta, interleaved passes after; `__syncwarp()`

**Verdict: CONFIRMED.** Independent thread scheduling starts at compute capability 7.0 (Volta). NVIDIA still tells you not to assume lockstep and to use `__syncwarp()`.

Quotes (Volta whitepaper):

- Pascal and earlier: "Divergent branches within a warp are serialized so that all statements in one side of the branch are executed together to completion before any statements in the other side are executed. After the else statement, the threads of the warp will typically reconverge."
- Volta: "Statements from the if and else branches in the program can now be interleaved in time ... Note that execution is still SIMT: at any given clock cycle, CUDA cores execute the same instruction for all active threads in a warp just as before".
- "Volta includes a schedule optimizer which determines how to group active threads from the same warp together into SIMT units."

Quotes (current programming guide, section 3.2.2.1.1):

- "On GPUs with compute capability lower than 7.0, warps used a single program counter shared amongst all 32 threads in the warp together with an active mask specifying the active threads of the warp."
- "In GPUs of compute capability 7.0 and later, independent thread scheduling allows full concurrency between threads, regardless of warp. With independent thread scheduling, the GPU maintains execution state per thread, including a program counter and call stack, and can yield execution at a per-thread granularity".
- "Warp-synchronous code assumes that threads in the same warp execute in lockstep at every instruction, but the ability for threads to diverge and reconverge at sub-warp granularity makes such assumptions invalid. ... Developers should explicitly synchronize such code using __syncwarp() to ensure correct behavior across all GPU generations."

`__syncwarp` (guide section 5.4.4.2): `void __syncwarp(unsigned mask = 0xFFFFFFFF);` "Calling __syncwarp(mask) provides memory ordering among the participating threads within a warp named in mask". The mask is 32 bits wide on CUDA (compare B4).

Related caution for anyone tempted to probe divergence from inside a kernel: "__activemask() cannot be used to determine which warp lanes execute a given branch. This function is intended for opportunistic warp-level programming and only provides an instantaneous snapshot of the active threads within a warp."

The oldest compute capability in the current guide's tables is 7.5, so every device the current toolkit documents has independent thread scheduling. The "Pascal-style sequential" order is history; show it as that.

- https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf (pages 26 to 28)
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (section 3.2.2.1.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.4.2 and 5.4.6.1)

### A11. Latency tolerance and zero-overhead warp switching

**Verdict: CONFIRMED.**

Quotes:

- "The execution context (program counters, registers, etc.) for each warp processed by an SM is maintained on-chip throughout the warp's lifetime. Therefore, switching between warps incurs no cost. At each instruction issue cycle, a warp scheduler selects a warp with threads ready to execute its next instruction (the active threads of the warp) and issues the instruction to those threads."
- "GPUs inherently hide memory latency through parallelism. That is, the SM switches to execute another warp while memory operations complete."
- "Unlike CPU cores, SMs issue instructions in order and do not perform branch prediction or speculative execution." This is NVIDIA's version of the book's point that GPUs spend little area on branch prediction (p. 84).

The book's names "fine-grained multithreading", "priority logic" and "zero-overhead scheduling" do not appear in these pages; NVIDIA's heading is "Hardware Multithreading". How the scheduler picks among ready warps is not documented.

- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (sections 3.2.2, 3.2.2.2 and 3.2.5)

### A12. Per-SM limits: blocks, warps, threads, registers

**Verdict: CONFIRMED** for the book's H100 numbers (compute capability 9.0): 32 resident blocks, 64 resident warps, 2048 resident threads, 64 K registers per SM, 1024 threads per block. The limits are not the same on other current GPUs, which matters for any occupancy calculator.

Table 33 and Table 34 of the guide, rows read with their column spans:

| Compute capability | Resident blocks per SM | Resident warps per SM | Resident threads per SM | Example devices (from NVIDIA's compute capability page) |
|---|---|---|---|---|
| 7.5 | 16 | 32 | 1024 | T4, RTX 20 series |
| 8.0 | 32 | 64 | 2048 | A100, A30 |
| 8.6 | 16 | 48 | 1536 | A40, A10, RTX 30 series |
| 8.7 | 16 | 48 | 1536 | Jetson AGX Orin, Orin NX, Orin Nano |
| 8.9 | 24 | 48 | 1536 | L4, L40, L40S, RTX 40 series |
| 9.0 | 32 | 64 | 2048 | GH200, H200, H100 |
| 10.0 | 32 | 64 | 2048 | GB200, B200 |
| 10.3 | 32 | 64 | 2048 | GB300, B300 |
| 10.7 | 16 | 32 | 1024 | not listed on the compute capability page |
| 11.0 | 24 | 48 | 1536 | Jetson T5000, T4000 |
| 12.x | 24 | 48 | 1536 | RTX 50 series, RTX PRO Blackwell (12.0); GB10 / DGX Spark (12.1) |

Rows that are the same for all eleven columns: "Number of 32-bit registers per SM | 64 K", "Maximum number of 32-bit registers per thread block | 64 K", "Maximum number of 32-bit registers per thread | 255", "Maximum number of threads per block | 1024", "Warp size | 32". "K" is 1024 (ch05 A4), so 64 K is the book's 65,536.

Consequences, by arithmetic on the table (ours):

- The book's "16 times more threads than streaming processors" is 2048 / 128 with the SM size from A2.
- The smallest block size that can fill the thread slots is threads per SM divided by blocks per SM: 64 on most rows, 96 on compute capabilities 8.6 and 8.7 (1536 / 16).
- On the 1536-thread rows the book's worked cases change. A 768-thread block fits twice and gives 100%, not 75%. A 1024-thread block fits once and gives about 67%. Full occupancy needs at most 65,536 / 1536 = 42 registers per thread (42.67 rounded down), not 32.
- A calculator must therefore take its limits from the device (A16, A18) and offer the book's 32 / 64 / 2048 / 65,536 only as "the book's example".

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Tables 33 and 34)
- https://developer.nvidia.com/cuda/gpus

### A13. Occupancy: definition, the book's worked cases, registers

**Verdict: CONFIRMED.** NVIDIA defines occupancy with warps, as the book does on p. 85, and works two of the book's cases with the same answers.

Quotes (programming guide, section 2.3.7):

- "The occupancy of a CUDA kernel is the ratio of the number of active warps to the maximum number of active warps supported by the SM. In general, it's a good practice to have occupancy as high as possible which hides latency and increases performance."
- Example device "such as compute capability 10.0": `maxBlocksPerMultiProcessor` 32, `regsPerMultiprocessor` 65536, `maxThreadsPerMultiProcessor` 2048, `regsPerBlock` 65536, `maxThreadsPerBlock` 1024.
- "If a kernel was launched as testKernel<<<512, 768>>>(), i.e., 768 threads per block, each SM would only be able to execute 2 thread blocks at a time. ... So the occupancy would be (768 * 2) / 2048, or 75%."
- "If a kernel was launched as testKernel<<<512, 32>>>() ... the total number of threads resident on the SM would be 32 blocks * 32 threads per block, or 1024 total threads. ... the occupancy in this case is 1024 / 2048, or 50%."
- Registers: "Each SM has a set of 32-bit registers that are partitioned among the warps, and a shared memory that is partitioned among the thread blocks." And: "If there are not enough resources available per SM to process at least one block, the kernel will fail to launch."

How to find a kernel's register count, all documented:

- At compile time: "one can use the --resource-usage option to nvcc, which will show the number of registers and shared memory required by the kernel." nvcc reference: "--resource-usage (-res-usage) Show resource usage such as registers and memory of the GPU code."
- At run time: `cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func)`; field `numRegs`: "The number of registers used by each thread of this function."

How to steer it:

- `--maxrregcount amount`: "Specify the maximum amount of registers that GPU functions can use. ... because thread registers are allocated from a global register pool on each GPU, a higher value of this option will also reduce the maximum thread block size, thereby reducing the amount of thread parallelism."
- `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)` on the kernel definition; "A kernel will fail to launch if it is executed with: more threads per block than its launch bound maxThreadsPerBlock."
- `__maxnreg__()` per kernel: "The __launch_bounds__() and __maxnreg__() qualifiers cannot be applied to the same kernel together."
- Spilling: "If the kernel needs more registers than this specified amount, the kernel is likely to spill to local memory ... In some cases even though spilling occurs, limiting registers allows more thread blocks to be scheduled which in turn increases occupancy and may result in a net increase in performance."

The book's register cliff (31 registers per thread and 512-thread blocks giving 4 blocks, 33 giving 3; p. 86) is not an NVIDIA example. It is consistent with "65536 registers per SM" by arithmetic, but NVIDIA says registers are "partitioned among the warps" and does not publish the allocation granularity. The exact register count at which a block stops fitting is what `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (A14) reports for a real kernel; a calculator built on the book's per-thread arithmetic should say that it is the book's model.

The book's reference [4] (Ryoo et al., CGO 2008, source of "performance cliff") was not looked up.

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.7)
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (section 3.2.2.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.3.2)
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html (sections 4.2.7.6 and 4.2.8.18)
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaFuncAttributes.html

### A14. Occupancy API and the Occupancy Calculator

**Verdict: CONFIRMED.** Both functions the book names exist with these signatures, and the calculator is still part of Nsight Compute.

Runtime API reference, section 6.27 "Occupancy":

- `__host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize)`: "Returns in *numBlocks the maximum number of active blocks per streaming multiprocessor for the device function."
- `__host__ cudaError_t cudaOccupancyMaxPotentialClusterSize(int *clusterSize, const void *func, const cudaLaunchConfig_t *launchConfig)`: "Given the kernel function ( func ) and launch configuration ( config ), return the maximum cluster size in *clusterSize."
- Also there: `cudaOccupancyMaxActiveClusters`, `cudaOccupancyAvailableDynamicSMemPerBlock`, `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`.
- C++ only, useful for deriving a launch configuration: `cudaOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, T func, size_t dynamicSMemSize = 0, int blockSizeLimit = 0)`: "Returns in *minGridSize and *blocksize a suggested grid / block size pair that achieves the best potential occupancy (i.e. the maximum number of active warps with the smallest number of blocks)."

The function returns a block count, not a percentage. Occupancy is `numBlocks * ceil(blockSize / warpSize) * warpSize / maxThreadsPerMultiProcessor`, with the last two read from the device (our arithmetic from the definition in A13).

Nsight Compute (docs v2026.4.0), section 4.13: "NVIDIA Nsight Compute provides an Occupancy Calculator that allows you to compute the multiprocessor occupancy of a GPU for a given CUDA kernel. It replaces the previously provided CUDA Occupancy Calculator spreadsheet." "The tables show the occupancy, as well as the number of active threads, warps, and thread blocks per multiprocessor, and the maximum number of active blocks on the GPU." "The graphs show the occupancy for your chosen block size as a blue circle, and for all other possible block sizes as a line graph." The UI executable is `ncu-ui`.

- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__OCCUPANCY.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__HIGHLEVEL.html
- https://docs.nvidia.com/nsight-compute/NsightCompute/index.html (section 4.13)

### A15. Compute capability: A100 is 8.0, H100 is 9.0

**Verdict: CONFIRMED.** NVIDIA's page lists "8.0 | NVIDIA A100, NVIDIA A30" and "9.0 | NVIDIA GH200, NVIDIA H200, NVIDIA H100". Newer values exist now: 10.0, 10.3, 11.0, 12.0, 12.1 on that page and 10.7 in the guide's tables (A12).

The book's "higher levels generally mean more resources per SM" does not hold as a rule today. In A12, compute capability 12.x has fewer resident threads per SM (1536) than 9.0 (2048), and 8.6 has fewer than 8.0. The number orders architectures, not capacity. Query the device.

The device reports it in `cudaDeviceProp.major` and `.minor` (ch05 A1).

- https://developer.nvidia.com/cuda/gpus
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html

### A16. `cudaGetDeviceCount` and the query loop

**Verdict: CONFIRMED.** `__host__ cudaError_t cudaGetDeviceCount(int *count)`: "Returns in *count the number of devices with compute capability greater or equal to 2.0 that are available for execution." `cudaGetDeviceProperties` is in ch05 A2. NVIDIA's own enumeration sample is the book's loop with `int` for the index, which removes the signed/unsigned comparison the notes point out:

```c
int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}
```

Single attributes can be read without the struct: `cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)`, for example with `cudaDevAttrWarpSize` ("Warp size in threads.") or `cudaDevAttrMaxRegistersPerBlock`.

Both calls return `cudaError_t`; the guide says "In production applications, it is best practice to always check and manage the return value of every CUDA API call."

- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html (section 3.4.1.1)
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__TYPES.html

### A17. "A modern PC often has two or more CUDA-capable devices because of integrated GPUs" (p. 88)

**Verdict: COULD NOT VERIFY.** No NVIDIA source read for this file says anything about how many CUDA devices a typical PC has. What the documentation does establish:

- `cudaGetDeviceCount` counts "devices with compute capability greater or equal to 2.0", and compute capability is defined per "NVIDIA GPU architecture" on NVIDIA's compute capability page, whose tables contain only NVIDIA products. An integrated GPU from another vendor is not in that list. The book's sentence does not make this distinction (the notes flag it as item 13 of "Unclear in the PDF").
- `cudaDeviceProp.integrated` exists: "Device is integrated as opposed to discrete." NVIDIA's own integrated devices in the current list are the Jetson modules (compute capabilities 8.7 and 11.0).

Safe course statement: enumerate and inspect every device as the book says, because multi-GPU machines exist, but do not repeat the reason the book gives. One piece of project evidence that the flag is not always usable: llama.cpp reads the property and then overrides it, `info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)` (D1).

- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaDeviceProp.html
- https://developer.nvidia.com/cuda/gpus

### A18. The `cudaDeviceProp` fields on p. 89

**Verdict: WRONG** for one field, which is why the item as a whole fails: `devProp.clockRate` no longer exists. The other seven names are confirmed, and the book's description of `regsPerBlock` needs a correction.

| Book's field | Status today | Documented meaning (v13.5) |
|---|---|---|
| `maxThreadsPerBlock` | CONFIRMED | "Maximum number of threads per block." |
| `multiProcessorCount` | CONFIRMED | "Number of multiprocessors on device." |
| `clockRate` | WRONG: removed in CUDA 13.0 | not in the struct; use `cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, dev)`, "Peak clock frequency in kilohertz." |
| `maxThreadsDim[3]` | CONFIRMED | "Maximum size of each dimension of a block." |
| `maxGridSize[3]` | CONFIRMED | "Maximum size of each dimension of a grid." |
| `regsPerBlock` | PARTLY | "32-bit registers available per block". It is the per-block limit, not the per-SM pool. |
| `warpSize` | CONFIRMED | "Warp size in threads." |

`clockRate`. The CUDA 13.0 release notes: "In CUDA 13.0, several deprecated fields have been removed from the cudaDeviceProperties structure." The table of removed fields includes "clockRate | cudaDeviceGetAttribute(cudaDevAttrClockRate)", "memoryClockRate | cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate)", "computeMode", "kernelExecTimeoutEnabled", "deviceOverlap", "singleToDoublePrecisionPerfRatio", "maxTexture1DLinear" and "cooperativeMultiDeviceLaunch". The 12.9.1 reference already said "Deprecated, Clock frequency in kilohertz". The v13.5 struct page has no `clockRate` member. Code written from the book's list does not compile against CUDA 13 headers. HIP still has the member (B9), so a single source that reads `prop.clockRate` builds on HIP and fails on CUDA.

`regsPerBlock`. The book says it gives "the number of registers available in each SM" and that the name is misleading. The current docs have a separate field for the SM, `regsPerMultiprocessor` ("32-bit registers available per multiprocessor"), and the guide lists them apart: "regsPerMultiprocessor: The number of 32-bit registers available per SM." and "regsPerBlock: The maximum number of 32-bit registers that can be allocated by a thread block." For occupancy arithmetic use `regsPerMultiprocessor`. The book's remark that on some compute capability levels the block maximum is below the SM total cannot be confirmed from today's tables: both are 64 K on every listed compute capability (A12).

Fields the book does not list but an occupancy calculation needs (spelling per ch05 A1): `maxBlocksPerMultiProcessor`, `maxThreadsPerMultiProcessor`, `regsPerMultiprocessor`, `sharedMemPerMultiprocessor`.

The guide now also offers the same limits without a device, from libcu++: `cuda::arch_traits<cuda::arch_id::sm_100>()` with members such as `max_blocks_per_multiprocessor`, `max_threads_per_multiprocessor`, `max_registers_per_multiprocessor`. Not needed for the labs; noted so nobody is surprised to meet it in new NVIDIA samples.

- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaDeviceProp.html
- https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/structcudaDeviceProp.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.7)

### A19. "Hundreds or even thousands of blocks simultaneously" and the 264-block tail example (p. 73)

**Verdict: CONFIRMED** by arithmetic on A2 and A12; neither number is printed by NVIDIA. An H100 SXM5 has 132 SMs and at most 32 resident blocks per SM, so at most 4,224 blocks are resident at once, and at most 264 when each SM holds only 2 blocks (the 1024-thread case, 132 x 2). That is the inference the notes already mark for the tail example; the book still does not name the device.

For a calculator that draws waves: blocks resident at once is `multiProcessorCount` times the per-SM block count from the occupancy function (A14). NVIDIA uses the same product in its cooperative-launch example (A7).

- Sources as in A2, A7 and A12.

### A20. Warp-level primitives (named in the chapter, used in chapters 10 to 12)

**Verdict: CONFIRMED** that they exist under these names; details belong to the later chapters' fact files.

Guide section 5.4.6 "Warp Functions": `unsigned __activemask();`, `int __all_sync(unsigned mask, int predicate);`, `int __any_sync(unsigned mask, int predicate);`, `unsigned __ballot_sync(unsigned mask, int predicate);`, `__match_any_sync`, `__match_all_sync`, `T __shfl_sync(unsigned mask, T value, int srcLane, int width=warpSize);` and the `_up`, `_down`, `_xor` variants, and `__reduce_add_sync` and relatives ("Supported by devices of compute capability 8.x or higher."). All masks are `unsigned`, 32 bits.

NVIDIA now steers people to library wrappers: "It is suggested to use the CUB Warp-Wide "Collective" Primitives to perform warp operations whenever possible for efficiency, safety, and portability reasons."

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.6)

---

## Group B: HIP / ROCm

Names that carry over unchanged are in ch05 B2 (struct fields), B3 (launch syntax), B4 (qualifiers and `__syncthreads`), B5 (warp width).

### B1. AMD's hardware vocabulary for the same picture

**Verdict: CONFIRMED.** AMD documents a model with the same shape and different names. The SM's counterpart is the compute unit (CU); a warp is a wavefront, and HIP's docs mostly keep the word "warp".

Quotes (HIP docs, "Hardware implementation" and "Programming model"):

- "Shader engines (SE): Top-level organizational units containing multiple shader arrays and shared resources"; "Shader arrays: Groups of compute units (CUs) sharing instruction and scalar caches"; "Compute units (CU): Basic execution units containing the arithmetic logic units (ALUs) and registers for thread execution".
- "The innermost grouping of threads is called a warp (HIP terminology) or wavefront (AMD ISA terminology)."
- "The size of a warp is architecture dependent and always fixed: 64 threads for CDNA architectures, 32 threads for RDNA architectures".
- Inside a CU (described for CDNA): "Four SIMD processors: Each containing 16 single-precision ALUs (or equivalent), for 64 total ALUs per CU." and "On architectures with 64-thread warps and 16-instruction wide SIMD units, executing one instruction takes four cycles (one cycle per 16 threads)."
- `multiProcessorCount` on AMD counts CUs, or half of them in workgroup-processor mode (header comment quoted in ch05 B2).

Loose mapping for the lesson, ours and not AMD's: SM to CU, processing block to SIMD, GPC to shader engine or shader array. AMD does not draw these equivalences. The second one is weaker than it looks: an NVIDIA processing block of 32 runs a 32-thread warp in one step, while a CDNA SIMD has 16 lanes and takes four cycles for a 64-thread wavefront.

AMD's per-product table (compute units, wavefront size, LDS, register file sizes) is at the last URL below. Device numbers stay out of lessons; the page is where the reader can look up their own card.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html (the documented link `.../reference/gpu-arch-specs.html` redirects here)

### B2. Block scheduling and scalability on AMD

**Verdict: CONFIRMED.** Same guarantees as CUDA: a block runs whole on one CU, order is not defined, blocks cannot use barriers with each other.

Quotes:

- "All warps of a block execute on the same CU, ensuring they can access the same LDS and synchronize efficiently."
- Workgroup manager: "Ensures all warps of a workgroup execute on the same CU for synchronization" and "Monitors resource availability and queues workgroups when resources are exhausted".
- "Workgroup-to-CU mapping is non-deterministic and based on available resources. You should not assume any specific mapping pattern".
- "Execution order is non-deterministic, work-groups may execute concurrently or in arbitrary order depending on hardware availability and resource usage."
- "This grid-level abstraction allows a single kernel launch to scale transparently with GPU size: larger GPUs simply execute more work-groups concurrently, while smaller GPUs schedule them in more waves."

The last quote is the only vendor text found that uses "waves" the way the book does.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html

### B3. `__syncthreads()` on HIP

**Verdict: PARTLY.** Spelling and meaning are confirmed (ch05 B4). The rule for conditionals is not stated in the HIP pages read.

- Docs: "Synchronization functions cause all threads in a group to wait at this synchronization point until all threads reached it. These functions implicitly include a threadfence".
- Header at `therock-10.2`: `__device__ inline __attribute__((convergent)) void __syncthreads() { __barrier(__CLK_GLOBAL_MEM_FENCE | __CLK_LOCAL_MEM_FENCE); }`, which ends in the hardware barrier instruction `__builtin_amdgcn_s_barrier()`.
- COULD NOT VERIFY: an AMD statement equivalent to NVIDIA's "only if the condition evaluates uniformly across the entire thread block" (A6), or what a thread that has already returned does to the barrier. Teach NVIDIA's rule as the portable rule and label it as taken from CUDA's documentation.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_device_functions.h (lines 697 to 722)

### B4. `__syncwarp()` on HIP

**Verdict: PARTLY.** It exists in the headers, with a different mask type from CUDA. It is not described on the HIP language-extensions page.

Header `amd_warp_sync_functions.h` at `therock-10.2`:

```cpp
__device__ inline void __syncwarp() {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence))
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_wave_barrier))
    __builtin_amdgcn_wave_barrier();
  ...
}

template <typename MaskT> __device__ inline void __syncwarp(MaskT mask) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_check_mask(mask);
  return __syncwarp();
}
```

Differences that matter:

- `__syncwarp()` with no argument is source-compatible. `__syncwarp(0xFFFFFFFF)`, which is what CUDA's own example comment spells out, does not compile on HIP: the literal is a 32-bit `unsigned` and the static assert rejects it.
- The header comment: "Warp sync builtins (with explicit mask argument) introduced in ROCm 6.2 as a preview ... These are enabled by default, and can be disabled by setting the macro "HIP_DISABLE_WARP_SYNC_BUILTINS"." The docs say the same from the other side: "the __sync variants are available in ROCm 7.0 (and enabled by default, unlike in previous versions)". The `HIP_ENABLE_WARP_SYNC_BUILTINS` define that llama.cpp still sets (ch05 B9) is for older ROCm.
- A search for "syncwarp" in the HIP language extensions, porting guide, performance guidelines and C++ support pages returns nothing. The function is in the header only, so treat its documentation status as thin.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h (lines 8 to 13 and 167 to 182)
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "Warp cross-lane functions")

### B5. Lockstep: AMD documents no independent thread scheduling

**Verdict: CONFIRMED.** This is a documented difference from NVIDIA (A10).

Quotes:

- "AMD GPUs guarantee, that all warp lanes are executed in lockstep, whereas NVIDIA GPUs that support Independent Thread Scheduling might require additional synchronization, or the use of the __sync variants."
- "Instructions always issue at warp granularity, with all threads in the warp executing the same instruction in lockstep."
- Header comment on `__activemask()`: "Since threads in a wave do not make independent progress, __activemask() always returns the exact active mask, i.e, all active threads in the wave." (Contrast NVIDIA's warning in A10 that `__activemask()` is only a snapshot.)

For the lesson: on AMD the "passes" picture of Fig. 4.9 is closer to the documented hardware than on current NVIDIA GPUs. Code should still call `__syncwarp()` where the book says to, because the same source has to run on both.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_warp_functions.h (lines 132 to 137)

### B6. Warp width and how a block is cut into warps on HIP

**Verdict: PARTLY.** Width: ch05 B5 (64 on gfx9, 32 on gfx10 and above, read `warpSize`). The linear order is documented the same way as CUDA's. The statement that warps are consecutive runs of that order is not spelled out, and AMD's product table disagrees with the HIP docs about Radeon.

Quotes:

- "When linearizing thread IDs within a block, assume the fast index is the x dimension, followed by the y and z dimensions."
- "if the number of threads can't be evenly divided by the warpSize, then the unused lanes are masked out from the corresponding SIMD execution."
- "Lane IDs aren't queried like other thread IDs, but are user-calculated."
- Docs example, lane id as `threadIdx.x % warpSize` (ch05 B5).

Gaps:

- COULD NOT VERIFY: an explicit AMD sentence like NVIDIA's "each warp contains threads of consecutive, increasing thread IDs". The pieces above and AMD's own lane-id arithmetic imply it; a warp painter can use the CUDA rule with the device's width and say that AMD documents the linear order and the lane arithmetic but not the cut in one sentence.
- AMD's GPU specification table lists "Wavefront Size | 32 or 64" for RDNA products (for example Radeon PRO W7900), while the HIP docs say "AMD devices return 64 for gfx9 and 32 for gfx10 and above. HIP doesn't support warpSize of 64 on gfx10 and above." For HIP code the HIP statement is the one that applies; the table describes the hardware.

Effect on the book's examples with a 64-wide warp (arithmetic, see Group E): a 16x16 block is 4 warps of four rows each, not 8 warps of two rows; an 8x8 block is one warp; a 64-thread 1D block is one warp.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html

### B7. Control divergence on AMD

**Verdict: CONFIRMED.** AMD describes it as masking through an execution mask (EXEC), names a metric for it, and makes the book's boundary-check argument itself.

Quotes:

- "When threads in a warp take different execution paths, the hardware serializes both branches, executing each path with only the relevant threads active. This reduces effective parallelism and wastes cycles on inactive threads."
- "For branching threads where conditional instructions lead to thread divergence, ALUs still process the full warp, but the result for divergent threads is masked out. This leads to wasted ALU cycles".
- "On AMD architectures, this spatial uniformity is tracked via the EXEC mask. Divergence forces EXEC to toggle individual bits to deactivate lanes following a different control path."
- "the branch unit handles warp-level control flow, not execution mask updates for thread divergence, which are handled through predication."
- "Branch efficiency: The ratio of non-divergent to total branches. Low efficiency indicates significant divergence overhead."
- On the boundary check `if (idx < n)`: it "usually has very high branch efficiency, since nearly all warps consist entirely of threads that either satisfy idx < n or not—except perhaps for the last partial warp, which straddles the boundary of n."
- A bridge sentence worth borrowing in our own words: "While CPUs also optimize branch behavior, they focus on temporal uniformity ... GPUs, on the other hand, care about spatial uniformity: whether all lanes in the warp take the same branch at the same time."
- Remedies AMD lists: "restructure the code to separate divergent paths into different kernel launches or use predication (branchless programming) to keep all threads active".

A trap in AMD's own page: its example marked "Good: no divergence" is `if (threadIdx.x < 32)`, with the comment "All threads in first half-warp execute". That is divergence-free only where a warp is 32 threads. On a 64-wide device the first warp of each block splits 32 / 32. It is a ready-made case for "never hard-code 32" (our reading, by the arithmetic of B6).

"SIMD efficiency", the term in the book's exercise 1, is used once in passing on AMD's page ("maximizing SIMD efficiency") and is not defined there either. Keep the definition the notes chose and label it ours.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html (section "Control flow optimization")
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html

### B8. Latency hiding and zero-overhead switching on AMD

**Verdict: CONFIRMED.**

- "The hardware can perform single-cycle context switching between warps with zero overhead, as all warp contexts remain resident on the CU. This enables efficient latency hiding, allowing the CU to switch to another warp immediately when the current warp encounters a stall condition such as a memory access."
- "When a warp issues an instruction whose operands are not yet ready, for instance, a global memory load from HBM, it becomes stalled. Rather than idling, the warp scheduler selects another ready warp to execute."
- Scheduling detail AMD publishes and NVIDIA does not: "The fetch arbiter selects one warp per cycle to fetch instructions from memory, prioritizing the oldest warps." and "The issue arbiter ... selecting warps from one pool per cycle in round-robin fashion."
- "GPUs allocate the majority of their die area to arithmetic pipelines", against CPUs that spend it on "instruction flow control, branch prediction, and complex caching hierarchies".

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html

### B9. Device queries on HIP, and what AMD puts in the fields

**Verdict: PARTLY.** The calls and field names exist (ch05 B2) and `clockRate` is still a member. Two things need care: several fields are computed differently from NVIDIA's, and AMD's reference page carries a "Bug" note that the source at the release tag contradicts.

API: `hipError_t hipGetDeviceCount(int* count);` "Returns in *count the number of devices that have ability to run compute commands. If there are no such devices, then hipGetDeviceCount will return hipErrorNoDevice." (CUDA's call is documented to return a count; this one documents an error for zero.) `hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr, int deviceId)` with `hipDeviceAttributeWarpSize`, `hipDeviceAttributeClockRate` and others.

How `ihipGetDeviceProperties` fills the struct (`hip_device.cpp` and `rocdevice.cpp` at `therock-10.2`):

| Field | Assigned from | What that means |
|---|---|---|
| `warpSize` | `info.wavefrontWidth_` | wavefront width reported by the driver |
| `multiProcessorCount` | `info.maxComputeUnits_` | CUs (halved in WGP mode) |
| `maxThreadsPerBlock` | `info.maxWorkGroupSize_` | |
| `maxThreadsPerMultiProcessor` | `info.maxThreadsPerCU_` = `wavefrontWidth_ * max_waves_per_cu` | wave slots times width |
| `maxBlocksPerMultiProcessor` | `maxThreadsPerCU_ / wavefrontWidth_` | the number of wave slots; not a separate block-slot limit |
| `regsPerBlock` and `regsPerMultiprocessor` | both `info.availableRegistersPerCU_` = `vgprsPerSimd_ * simdPerCU_ * wavefrontWidth_` | vector registers only, same value in both fields |
| `clockRate` | `info.maxEngineClockFrequency_ * 1000` | still present; header: "Max clock frequency of the multiProcessors in khz." |
| `integrated` | `info.hostUnifiedMemory_` | header: "APU vs dGPU" |
| `cooperativeLaunch` | `info.cooperativeGroups_` | |
| `clusterLaunch` | `info.clusterMaxSize_ > 1` | see B11 |

Consequences:

- `maxBlocksPerMultiProcessor` on AMD is derived, so the book's "block slots bind before thread slots" case (32-thread blocks giving 50%) has no AMD counterpart in these fields: one-wave blocks can fill every wave slot. A calculator that reads this field from a HIP report must not present it as an independent limit.
- `regsPerBlock` equals `regsPerMultiprocessor` on AMD and counts VGPRs. Scalar registers (SGPRs) are a second pool that the fields do not show (B10).

Conflict to know about: the HIP reference page for `hipGetDeviceProperties` says "Bug: HIP-Clang always returns 0 for maxThreadsPerMultiProcessor", "HIP-Clang always returns 0 for regsPerBlock", "HIP-Clang always returns 0 for l2CacheSize". The code at `therock-10.2` assigns all three (table above). Which one a given installation shows was not tested; a lab that prints these fields should accept 0 and say what it means.

Second conflict: the language-extensions page says "hipDeviceProp_t also has a field executionUnitsPerMultiprocessor." No such member is in `hip_runtime_api.h` or `hip_device.cpp` at `therock-10.2`.

- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_device.cpp (lines 626 to 826)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/rocclr/device/rocm/rocdevice.cpp (lines 1626 to 1740)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (struct from line 263; `hipDeviceAttribute_t` from line 450)
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/initialization.html

### B10. Occupancy on AMD: same idea, different limits

**Verdict: CONFIRMED.** AMD defines occupancy the same way and provides the same API names, but the binding resources are wave slots, two register pools and LDS. There is no block-slot limit in AMD's description or in the runtime's calculation.

Quotes ("Occupancy theory", and "Hardware implementation" for the slot counts):

- "Occupancy = Active warps / Max warps per CU".
- "Each CU has finite resources—Vector General-Purpose Registers (VGPRs), Scalar General-Purpose Registers (SGPRs), LDS (shared memory), and wave slots—that must be shared among all resident blocks."
- "The sequencer organizes active warps into four pools, each containing slots for up to ten warps (eight on the CDNA2 MI200 Series). ... This organization theoretically allows up to 40 concurrent warps per CU, though actual occupancy is typically limited by register and LDS usage."
- A caution the book does not give, relevant to GEMM: "once occupancy is sufficient for latency hiding, increasing it further can hurt performance by reducing the number of available registers or LDS per warp" and "High-performance kernels (for example, MFMA-based GEMMs on CDNA) often operate at low occupancy because only a few warps are needed to fully saturate the MFMA and memory pipelines."
- "Memory-bound kernels benefit more from high occupancy".

API (header at `therock-10.2`, same shapes as CUDA): `hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk)`, `hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize, const void* f, size_t dynSharedMemPerBlk, int blockSizeLimit)`, `hipOccupancyAvailableDynamicSMemPerBlock`, `hipOccupancyMaxActiveClusters`, `hipOccupancyMaxPotentialClusterSize`.

What the runtime computes (`ihipOccupancyMaxActiveBlocksPerMultiprocessor` in `hip_platform.cpp`), in its own words and identifiers:

- "Find wave occupancy per CU => simd_per_cu * GPR usage"; `MaxWavesPerSimd = (device.isa().versionMajor() <= 9) ? 8 : 16`.
- Waves per SIMD limited by vector registers: `maxVGPRs / alignUp(usedVGPRs_, VgprGranularity)`, then by scalar registers ("Mirrors LLVM AMDGPUUtils::getOccupancyWithNumSGPRs()").
- "Need to align with hardware wavefront size. If they want 65 threads, but waves are 64, then we need 128 threads per block." Then `*maxBlocksPerCU = alu_limited_threads / aligned_input_size`, capped by LDS: `lds_pool_size / total_used_lds`.

Small mismatch inside AMD's material: the docs say up to 10 slots per pool (40 per CU) on newer CDNA, the runtime uses 8 waves per SIMD for ISA major version 9 and below, with the comment "Limited by SPI 32 per CU, hence 8 per SIMD". CDNA parts are major version 9 in AMD's table. The function's answer, not the 40, is what a HIP program gets.

Registers: registers are allocated per wave in granules, not per thread, so the book's "registers per thread times threads" arithmetic is only the right shape on AMD. `hipFuncGetAttributes` fills `numRegs` with `wginfo->usedVGPRs_` (vector registers only).

Tools AMD documents: "hipcc --resource-usage kernel.hip" ("The compiler can report per-kernel register usage statistics") and "rocprofv3 --occupancy ./your_application" ("Profiling tools report the ratio of active warps to maximum possible warps per CU"). Neither was run. `hipcc` itself is labelled legacy (ch05 B6); whether `amdclang++` takes the same flag was not checked.

`__launch_bounds__` on HIP takes `(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)`. The second parameter is "the minimum number of warps that must be able to run concurrently on an execution unit", where "On AMD GPUs a Compute Unit consists of 4 Execution Units, also known as SIMDs". CUDA's second parameter is `minBlocksPerMultiprocessor` (A13). Same spelling, different unit; the macro expands to `amdgpu_flat_work_group_size(1, max)` and `amdgpu_waves_per_eu(n)`.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html (section "Occupancy theory")
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/occupancy.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_platform.cpp (lines 22 to 135)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_global.cpp (lines 134 to 141)
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "__launch_bounds__")
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h (lines 144 to 151)

### B11. Cooperative Groups, grid barrier and clusters on HIP

**Verdict: PARTLY.** Grid-wide sync is confirmed and works as in CUDA. HIP now documents a cluster group as well, which the notes assumed absent, but which AMD devices support it could not be established.

Grid:

- Header: `#include <hip/hip_cooperative_groups.h>` (the porting guide maps `cooperative_groups.h => hip/hip_cooperative_groups.h`).
- "Represents an inter-block cooperative group type where the group's participating threads span multiple blocks running the same kernel on the same device. Use the cooperative launch API to enable synchronization across the grid group." `grid_group g = this_grid();` then `g.sync()`.
- Launch: `hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX, void** kernelParams, unsigned int sharedMemBytes, hipStream_t stream);`. The header documents a dedicated error for a grid that cannot be resident at once, `hipErrorCooperativeLaunchTooLarge`.
- Support flag: `hipDeviceProp_t.cooperativeLaunch` ("HIP device supports cooperative launch").
- Still present where CUDA 13 removed it: `multi_grid_group`, `hipLaunchCooperativeKernelMultiDevice`, with the note "Multi-grid has been deprecated since ROCm 5.0."
- "Explicit warp-level thread handling is absent from the Cooperative groups API."

Clusters:

- Docs: "Cluster group: Represents a group of thread blocks that execute together on the device. A cluster can be 1D, 2D, or 3D and can contain up to 15 workgroups. Each block in the cluster runs on a separate Workgroup Processor (WGP). The cluster group provides synchronization and shared-memory mapping across those blocks." `cluster_group g = this_cluster();`
- "map_shared_rank() and query_shared_rank() require a device that reports multi-block cluster support. Check hipDeviceProp_t::clusterLaunch at runtime before relying on these functions."
- Source: `deviceProps.clusterLaunch = info.clusterMaxSize_ > 1;` with the comment "A cluster of size 1 is a regular single-block launch (legal on all GPUs); clusterLaunch advertises multi-block cluster support, which only devices reporting a max size > 1 have." The size comes from the driver query `HSA_AMD_AGENT_INFO_CLUSTER_MAX_SIZE`; "devices with no cluster support; max size is 0".
- Header note: "On AMD devices, non-portable cluster size and cluster scheduling policy are stored for readback but may not affect dispatch."
- COULD NOT VERIFY: which AMD products report a cluster size above 1. No page read names one. The limit differs from CUDA's portable 8 blocks (ch05 A6).

For chapter 4 this only changes one sentence: clusters are no longer NVIDIA-only at the API level, and a portable program must check `clusterLaunch` on both vendors.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_device.cpp (lines 801 to 804)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/rocclr/device/rocm/rocdevice.cpp (lines 1843 to 1861)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 3004 to 3024, 7675 to 7683, 7895 to 7917)

### B12. Warp-level primitives on HIP (forward reference)

**Verdict: CONFIRMED** that they exist; the mask width is the difference (ch05 B5 has the lane-mask hazard).

- "All mask values returned or accepted by these built-ins are 64-bit unsigned integer values, even when compiled for a device with 32 threads per warp."
- Signatures in the docs: `unsigned long long __ballot(int predicate)`, `unsigned long long __activemask()`, `int __all_sync(unsigned long long mask, int predicate)`, `unsigned long long __ballot_sync(unsigned long long mask, int predicate)`, `__match_any_sync`, `__match_all_sync`, and the shuffles with `int width = warpSize`.
- "The implementation includes a static assert to check that the program source uses the correct type for the mask."

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "Warp cross-lane functions")

---

## Group C: Metal

Sources: MSL spec 4.1; Apple developer documentation (JSON form of the pages); `MTLDevice.h` and `MTLComputePipeline.h` from the macOS 27.0 SDK on this machine. Property meanings already settled: ch05 C2 (barrier), C3 (pipeline and device thread properties), C4 (kernel attributes), C5 (the two dispatch calls).

### C1. Enumerating and querying a device

**Verdict: CONFIRMED.** Metal has device enumeration and a set of queryable properties. None of them describes the compute hardware the way `cudaDeviceProp` does.

- `MTLCreateSystemDefaultDevice()`: "Returns the device instance Metal selects as the default." "In macOS, in order for the system to provide a default Metal device object, you need to link to the CoreGraphics framework. You usually need to do this explicitly if you're writing apps that don't use graphics by default, such as command line tools."
- `MTLCopyAllDevices()` (macOS 10.11 and later, iOS 18.0 and later): "Returns an array of all the Metal device instances in the system." Header: "On macOS and macCatalyst, this API will not cause the system to switch devices and leaves the decision about which GPU to use up to the application".
- Device properties useful to a compute program (header names): `name`, `registryID`, `architecture` (an `MTLArchitecture` whose only property is `name`, "The device's architecture name."; macOS 14.0 and later), `maxThreadsPerThreadgroup`, `maxThreadgroupMemoryLength`, `maxBufferLength`, `recommendedMaxWorkingSetSize`, `hasUnifiedMemory`, `currentAllocatedSize`, and `supportsFamily:`.
- Deprecated in the 27.0 SDK with the message "Not applicable on Apple Silicon": `lowPower`, `headless`, `removable`, `location`, `maxTransferRate`. These were how Intel-era Macs told an integrated GPU from a discrete or external one. The closest thing Metal has to the book's "pick the capable device" loop is therefore a legacy path.

The complete list of `MTLDevice` properties in the header was read. There is no count of cores or compute units, no SIMD width, no clock rate, no register count and no resident-thread limit. A grep of `MTLDevice.h` and `MTLComputePipeline.h` for "simd", "core", "occupan" and "register" finds nothing.

- https://developer.apple.com/documentation/metal/mtlcreatesystemdefaultdevice()
- https://developer.apple.com/documentation/metal/mtlcopyalldevices()
- https://developer.apple.com/documentation/metal/mtldevice/architecture
- Local header: `Metal.framework/Headers/MTLDevice.h` (macOS 27.0 SDK), lines 126 to 137 and 440 to 547

### C2. The SIMD-group: Metal's unit under the threadgroup

**Verdict: PARTLY.** The SIMD-group is documented and is the nearest thing to a warp (the comparison is ours, as in ch05 C3). Two things CUDA documents are missing: the width has no documented value, and the way a threadgroup is cut into SIMD-groups is explicitly left to the implementation.

Quotes (MSL spec):

- Section 4.4.1: "Within a threadgroup, you can divide threads into SIMD-groups, which are collections of threads that execute concurrently. The mapping to SIMD-groups is invariant for the duration of a kernel's execution, across dispatches of a given kernel with the same launch parameters, and from one threadgroup to another within the dispatch (excluding the trailing edge threadgroups in the presence of nonuniform threadgroup sizes). In addition, all SIMD-groups within a threadgroup needs to be the same size, apart from the SIMD-group with the maximum index, which may be smaller".
- Section 5.2.3.6: "Within a threadgroup, threads are divided into SIMD-groups in an implementation-defined fashion. Any given thread in a SIMD-group can query its SIMD lane ID and which SIMD-group it is a member of."
- "For standard Metal compute functions (other than tile functions), SIMD-groups are linear and one-dimensional. (Threadgroups may be multidimensional.)" and `simdgroups_per_threadgroup = ceil(threads_per_threadgroup / threads_per_simdgroup)`.
- "SIMD-groups execute concurrently within a given threadgroup and make independent forward progress with respect to each other, in the absence of threadgroup barrier operations."
- Kernel attributes (macOS: Metal 2 and later): `[[threads_per_simdgroup]]` "The thread execution width of a SIMD-group (compute unit).", `[[simdgroup_index_in_threadgroup]]` "The scalar index of a SIMD-group within a threadgroup.", `[[thread_index_in_simdgroup]]` "The scalar index of a thread within a SIMD-group.", `[[simdgroups_per_threadgroup]]`. Types: `ushort` or `uint`, and "The types for declaring these built-in variables need to match."

What follows for the course:

- A Metal kernel must not compute its SIMD-group as `thread_index_in_threadgroup / width`. The spec gives the linear thread index (`ly * Sx + lx`, ch05 C4) and says the grouping is implementation-defined. Ask for `[[simdgroup_index_in_threadgroup]]` and `[[thread_index_in_simdgroup]]`. A "warp painter" can draw the CUDA rule for CUDA and HIP; for Metal it can only draw what a kernel reports on the reader's device.
- The width is read from `MTLComputePipelineState.threadExecutionWidth` on the host or `[[threads_per_simdgroup]]` in the kernel. It is a property of a compiled pipeline, not of the device, so a device report cannot contain it; a lab has to build a pipeline first. No Apple source gives a value for any chip. Apple's sizing article uses 16 in an example ("a thread execution width of 16"). llama.cpp and parts of MLX assume 32 (D2, D3). Lessons must not state a number.
- Whether an Apple GPU keeps the threads of a SIMD-group in lockstep, as AMD documents (B5), or lets them interleave, as NVIDIA documents (A10), is not stated. COULD NOT VERIFY.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 4.4.1, page 121; section 5.2.3.6 and Table 5.8, pages 153 to 158)
- https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth
- https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes

### C3. Barriers below and above the threadgroup

**Verdict: CONFIRMED.** `simdgroup_barrier` is the counterpart of `__syncwarp()`. Metal documents no barrier wider than a threadgroup.

- `void simdgroup_barrier(mem_flags flags)` (macOS: Metal 2 and later): "All threads in a SIMD-group executing the kernel, fragment, mesh, or object need to execute this function before any thread can continue execution beyond the simdgroup_barrier." It takes no lane mask; the group is the whole SIMD-group.
- The rule for conditionals is written for both barriers, and for loops too: "If threadgroup_barrier (or simdgroup_barrier) is inside a loop, for each iteration of the loop, if any thread in the threadgroup (or SIMD-group) executes the barrier, then all threads in the threadgroup (or SIMD-group) need to execute the barrier function before any threads continue execution beyond the barrier function." (The conditional sentence is quoted in ch05 C2.)
- Exited threads: "On Apple silicon, a thread that has ended no longer participates or blocks remaining threads at a barrier." This matches NVIDIA's "non-exited threads" (A6), and it is documented for Apple silicon only.
- Wider scopes: the Metal 4.1 overloads take `thread_scope scope`, and `enum thread_scope { thread_scope_thread, thread_scope_simdgroup, thread_scope_threadgroup, thread_scope_device }` exists ("All OS: Metal 3.2 and later support thread_scope for Apple silicon"). The spec describes the argument as a memory-ordering scope: "The scope argument ... specifies which threads can observe the memory accesses to the address space identified by flags." It does not say that a barrier with `thread_scope_device` holds every thread of the grid. So: no documented grid-wide execution barrier, no cooperative launch, and no thread block cluster in Metal. A phase that needs the whole grid to finish is a separate dispatch.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 6.10.1, pages 215 to 217; section 6.16.2, pages 300 to 301)

### C4. Control divergence in Metal

**Verdict: COULD NOT VERIFY.** Apple does not document how threads of a SIMD-group that take different paths are executed, or what it costs. The spec only acknowledges that threads can be inactive because of control flow.

- MSL spec section 6.10.2: "An active thread is a thread that is executing. An inactive thread is a thread that is not executing. For example, a thread may not be active due to flow control or when a task has insufficient work to fill the group."
- `simd_active_threads_mask()`: "Returns a simd_vote mask that represents the active threads. This function is equivalent to simd_ballot (true)".
- Cooperative operations demand uniform control flow: SIMD-group matrix operations "must be executed only under uniform control-flow within the SIMD-group or the behavior is undefined."
- A text search of the spec for "divergen" finds one sentence, about tensor operations ("different SIMD-groups can be divergent with each other"); "occupancy" does not occur.

So the multi-pass picture of Figs. 4.9 and 4.10 is NVIDIA's and AMD's documented model. On Metal it is a hypothesis the reader can test: the divergence lab is the evidence, and its README should say that no Apple document predicts the outcome. Whether the Metal compiler turns a short branch into a select is not documented either.

One thing Metal does document removes the most common source of divergence: with `dispatchThreads:threadsPerThreadgroup:` there are no out-of-range threads, so there is no boundary check to diverge on (ch05 C5: "You don't need to check a thread's position in a grid if you use the dispatchThreads(_:threadsPerThreadgroup:) technique."). The edge threadgroups are smaller instead, and the SIMD-group "with the maximum index ... may be smaller" (C2).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 2.4, page 38; section 6.10.2, pages 217 to 218)

### C5. Occupancy, per-core limits and register counts in Metal

**Verdict: COULD NOT VERIFY any equivalent; none is documented.** No Apple source read for this file (MSL spec, the `MTLDevice` and `MTLComputePipelineState` headers and pages, the sizing articles) describes resident-thread limits, block slots, a register pool, a per-kernel register count, an occupancy figure or an occupancy function. Say so plainly in the lesson.

The nearest documented behaviour is that the thread cap of a kernel depends on the kernel:

- `MTLComputePipelineState.maxTotalThreadsPerThreadgroup`: "When you create a compute pipeline state, it calculates the maximum number of threads available on the device. This value never changes, but may be different for different pipeline objects."
- `MTLDevice.maxThreadsPerThreadgroup`: "the maximum thread group size the device can support for a trivial shader. This size isn't guaranteed for all shaders." (ch05 C3)
- `MTLComputePipelineDescriptor.maxTotalThreadsPerThreadgroup`: "By default, this property's value is 0, which instructs Metal to calculate the maximum number of threads per threadgroup based on the device's capabilities and the compute shader's memory usage." and "Metal may return an error if this value exceeds the available resources for the device, or Metal may lower the thread limit when creating the compute pipeline state, which can reduce runtime performance."
- MSL attribute `[[max_total_threads_per_threadgroup(x)]]` on a kernel: "If the [[max_total_threads_per_threadgroup]] value is greater than the [MTLDevice maxThreadsPerThreadgroup] property, then compute pipeline state creation fails." Apple: "Set this property to the same value as your shader's [[max_total_threads_per_threadgroup]] attribute when setting both values; different values can create a runtime error."
- `MTLComputePipelineDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth`: "If you can guarantee that the threadgroup size used by all compute commands in this pipeline is a multiple of threadExecutionWidth, set this property to true to take advantage of additional Metal optimizations." and "When this configuration value is true and the threadgroup size isn't a multiple of thread execution width, the compute pass's execution results are undefined."

These are loosely like `__launch_bounds__` and like "a kernel that needs more resources gets a smaller block", but Apple does not explain them in terms of registers or occupancy, and the link is ours. A Metal version of the block-size lab measures time against threadgroup size and reports it; it cannot be explained with an occupancy number.

- https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/maxtotalthreadsperthreadgroup
- https://developer.apple.com/documentation/metal/mtlcomputepipelinedescriptor/maxtotalthreadsperthreadgroup
- https://developer.apple.com/documentation/metal/mtlcomputepipelinedescriptor/threadgroupsizeismultipleofthreadexecutionwidth
- https://developer.apple.com/documentation/metal/mtldevice/maxthreadsperthreadgroup
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 5.1.3, pages 126 to 127)
- Local header: `Metal.framework/Headers/MTLComputePipeline.h` (macOS 27.0 SDK), lines 45 to 55 and 220 to 237

### C6. Deriving a launch configuration from queried values on Metal

**Verdict: CONFIRMED.** Apple publishes a recipe, and it is the Metal half of milestone M1.

```objc
// Calculate the maximum threads per threadgroup based on the thread execution width.
NSUInteger w = pipelineState.threadExecutionWidth;
NSUInteger h = pipelineState.maxTotalThreadsPerThreadgroup / w;
MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);
```

Then either `[computeCommandEncoder dispatchThreads: threadsPerGrid threadsPerThreadgroup: threadsPerThreadgroup];` or, for uniform groups, `MTLSizeMake((texture.width + w - 1) / w, (texture.height + h - 1) / h, 1)` threadgroups with a bounds check in the kernel.

Apple: "You calculate the number of threads per threadgroup based on two MTLComputePipelineState properties". Both inputs are per pipeline, which is the practical difference from CUDA, where the limits come from the device and the block shape is the programmer's choice within them.

- https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes

### C7. SIMD-group functions (forward reference)

**Verdict: CONFIRMED** that they exist; details belong to the chapters that use them.

Header `<metal_simdgroup>`. "SIMD-group functions allow threads in a SIMD-group (see section 4.4.1) to share data without using threadgroup memory or requiring any synchronization operations, such as a barrier." Names in Tables 6.14 and 6.15: `simd_active_threads_mask`, `simd_all`, `simd_any`, `simd_ballot`, `simd_broadcast`, `simd_broadcast_first`, `simd_is_first`, `simd_shuffle`, `simd_shuffle_up`, `simd_shuffle_down`, `simd_shuffle_xor`, `simd_shuffle_and_fill_up`, `simd_shuffle_and_fill_down`, and reductions `simd_sum`, `simd_max`, `simd_min`, `simd_product`, `simd_and`, `simd_or`, `simd_xor`, `simd_prefix_inclusive_sum`, `simd_prefix_exclusive_sum`. The ballot result is a wrapper class `simd_vote` around an unspecified integer type `vote_t`, not a raw 32-bit or 64-bit mask.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 6.10.2, pages 217 to 226)

---

## Group D: how the projects do it

All files read from the default branch on 2026-11-20. Last-commit dates are from the GitHub commits API.

### D1. llama.cpp, CUDA and HIP: device query and warp width

**Verdict: CONFIRMED.**

- Device query in `ggml_cuda_init()` (`ggml/src/ggml-cuda/ggml-cuda.cu`, last commit 2026-11-19): `cudaGetDeviceCount(&info.physical_device_count)`, then per device `cudaGetDeviceProperties(&prop, physical_id)` and

```cpp
info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
info.devices[id].nsm        = prop.multiProcessorCount;
info.devices[id].smpb       = prop.sharedMemPerBlock;
info.devices[id].warp_size  = prop.warpSize;
```

  plus `cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id)` and `prop.major`, `prop.minor` for the compute capability. On HIP the same code runs through the renames of `vendors/hip.h` (ch05 B9) and also reads `prop.gcnArchName`.
- The stored `warp_size` is used on the host to choose kernels: `ggml_cuda_should_use_mmvf(src0, cc, warp_size, ...)`.
- Two warp-width constants in `common.cuh`: `#define WARP_SIZE 32`, the logical width most kernels are written for, and

```cpp
static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
#if defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
    return 64;
#else
    return 32;
#endif // defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
}
```

  which is the default `width` of the shuffle wrappers (`ggml_cuda_shfl_xor(T value, int lane_mask, int width = ggml_cuda_get_physical_warp_size(), ...)`).
- `argmax.cu` (last commit 2026-11-15) is written for 32 throughout: `const int n_warps = blockDim.x / WARP_SIZE;`, `const int lane_id = threadIdx.x % WARP_SIZE;`, shuffles called with `WARP_SIZE` as the width. On a 64-wide AMD device this runs as two logical 32-thread groups per wavefront, which is the case AMD describes as "it only utilizes half of the compute resources" (ch05 B5). `mmvq.cu` uses the physical width instead: `__launch_bounds__(calc_nwarps(...)*ggml_cuda_get_physical_warp_size(), 1)`.

This is a concrete, current example of both habits in one codebase, and a natural place for a first contribution.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu (lines 267 to 470)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh (lines 46, 415 to 449, 515 to 530)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/argmax.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu (lines 627 to 653)

### D2. llama.cpp, Metal: device query, pipeline limits, assumed SIMD width

**Verdict: CONFIRMED.**

- Device: `dev->mtl_device = MTLCreateSystemDefaultDevice();` then `hasUnifiedMemory`, `maxBufferLength`, `maxThreadgroupMemoryLength`, `recommendedMaxWorkingSetSize`, and feature checks by GPU family, for example `dev->props.has_simdgroup_reduction = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];` (`ggml-metal-device.m`).
- Pipeline: after compiling each kernel it logs `obj.maxTotalThreadsPerThreadgroup` and `obj.threadExecutionWidth` ("th_max = %4d | th_width = %4d") and rejects a pipeline where either is 0.
- Threadgroup sizes at dispatch are capped by the pipeline limit, for example `const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);` (`ggml-metal-ops.cpp`).
- The kernels assume a width: `#define N_SIMDWIDTH 32 // assuming SIMD group size is 32` in `kernels/common.h` (last commit 2026-08-24), used as in `softmax.metal`: `float max_val = simd_max(lmax); if (tptg.x > N_SIMDWIDTH) { ... }`, with the group and lane taken from `uint sgitg[[simdgroup_index_in_threadgroup]]` and `uint tiisg[[thread_index_in_simdgroup]]`.

So llama.cpp asks Metal which SIMD-group a thread is in (the documented way, C2) but hard-codes how wide a group is (not documented by Apple).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m (lines 800 to 812, 1118 to 1292)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-ops.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/common.h (line 23)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/softmax.metal

### D3. MLX: device info and SIMD width

**Verdict: CONFIRMED.**

- What MLX exposes about the device (`mlx/backend/metal/device_info.cpp`, last commit 2026-05-07): keys `device_name`, `architecture`, `max_buffer_length`, `max_recommended_working_set_size`, `memory_size`, `resource_limit`. `device_count()` returns 1 or 0. Nothing about threads, SIMD width or cores, consistent with C1.
- Threadgroup sizes come from the pipeline: `NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();` (`mlx/backend/metal/reduce.cpp`).
- Both habits again. `kernels/softmax.h` hard-codes the width: `constexpr int SIMD_SIZE = 32;` with `threadgroup AccT local_max[SIMD_SIZE];`. `kernels/arg_reduce.metal` (last commit 2026-08-28) asks for it: `uint simd_size [[threads_per_simdgroup]]`, then `for (uint offset = simd_size / 2; offset > 0; offset /= 2)` and `uint simd_groups = ceildiv(lsize.x, simd_size);`.

- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device_info.cpp
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/reduce.cpp
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/softmax.h (line 21)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/arg_reduce.metal (lines 95 to 165)

### D4. PyTorch: warp width on CUDA and ROCm, device properties, MPS dispatch

**Verdict: CONFIRMED.**

- Warp width (`torch/headeronly/macros/Macros.h`, last commit 2026-10-07; `c10/macros/Macros.h` now only includes it). On CUDA builds: `#define C10_WARP_SIZE 32`. On ROCm builds it is a function, 64 for `__GFX9__` device code and 32 otherwise, and the host must ask at run time. The comment is the clearest statement of the problem in any of the three projects:
  - "C10_WARP_SIZE is only allowed for device code. Host code dynamically-sized launch configs _must_ use at::cuda::warp_size(). Host or device statically-sized arrays _must_ use either C10_WARP_SIZE_UPPER_BOUND or C10_WARP_SIZE_LOWER_BOUND, as needed."
  - "In ROCm 7, warpSize is no longer constexpr, matching CUDA behavior."
- Device properties in Python: `torch.cuda.get_device_properties()` returns `_CudaDeviceProperties`, a direct binding of `cudaDeviceProp` (`torch/csrc/cuda/Module.cpp`, last commit 2026-11-18) with read-only attributes including `name`, `major`, `minor`, `is_integrated`, `multi_processor_count`, `total_memory`, `max_threads_per_multi_processor`, `max_threads_per_block`, `warp_size`, `shared_memory_per_block`, `shared_memory_per_block_optin`, `regs_per_multiprocessor`, `L2_cache_size`, `gcnArchName`. A reader can see the chapter's numbers for their own GPU from a Python prompt before writing any C++. There is no `max_blocks_per_multi_processor` in the lines read. `clock_rate` is there, and it is read the post-13.0 way (A18): `AT_CUDA_CHECK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, devicePropertiesIndex(prop)));`.
- MPS backend (`aten/src/ATen/native/mps/OperationUtils.h`, last commit 2026-11-19): launch configuration is derived from the pipeline and uses `dispatchThreads`, so kernels need no bounds check:

```objc
const auto maxThreadsPerGroup = [cplState maxTotalThreadsPerThreadgroup];
auto size = MTLSizeMake(length, 1, 1);
auto threadGroupSize = MTLSizeMake(std::clamp(length, 1UL, maxThreadsPerGroup), 1, 1);
[encoder dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
```

  with the guard "thread_position_in_grid is at most 32-bit per axis; Metal silently truncates larger grids". It does not read `threadExecutionWidth`.

- https://github.com/pytorch/pytorch/blob/main/torch/headeronly/macros/Macros.h (lines 337 to 418)
- https://github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/Module.cpp (lines 1190 to 1253; `clock_rate` at 1207 to 1214)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/OperationUtils.h (lines 617 to 630)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/block_reduce.cuh

---

## Group E: the book's divergence counts, recomputed

Not vendor facts. These are brute-force counts made for this file with the cutting rule of A8 (linear order, x fastest, consecutive runs of the warp width) and the usual guard (`i < n`, or `col < width && row < height`). A warp counts as divergent when some but not all of its threads pass the guard. They are here so the simulator's unit tests have agreed numbers, and so the 64-wide column is not guessed.

### E1. Boundary-check examples (pp. 81 to 82)

**Verdict: PARTLY.** Five of the book's counts reproduce; two do not.

| Case | Block | Warps (32-wide) | Divergent (32-wide) | Book says | Warps (64-wide) | Divergent (64-wide) |
|---|---|---|---|---|---|---|
| vector of 1003 | 64 | 32 | 1 | 1 of 32 | 16 | 1 |
| vector of 100 | 64 | 4 | 1 | 1 of 4 | 2 | 1 |
| vector of 1000 | 64 | 32 | 1 | 1 of 32 | 16 | 1 |
| vector of 10,000 | 64 | 314 | 1 | 1 of 313 | 157 | 1 |
| vector of 2000 (exercise 3) | 512 | 64 | 1 | answer sketch: 1 | 32 | 1 |
| image 76 wide, 62 high | 16x16 | 160 | 31 | 31 of 160 | 80 | 20 |
| image 200 wide, 150 high | 16x16 | 1040 | 75 | 80 of 1040 | 520 | 50 |
| image 150 wide, 200 high | 16x16 | 1040 | 100 | 80 of 1040 | 520 | 50 |

- 10,000 elements with 64-thread blocks is 157 blocks and 314 warps. The book's 313 is the number of warps that contain at least one valid element; the 314th is entirely out of range and does not diverge. The conclusion (well under 1%) is unaffected.
- The "200 x 150" picture: neither reading gives the printed 80. This agrees with item 4 of "Unclear in the PDF" in `notes/ch04.md` (75 for 200 wide, 100 for 150 wide). With 200 rows and 150 columns the fraction is 9.6%, above the book's "under 8%".
- With 64-wide warps every 2D count changes, because a 16x16 block is cut into four-row warps. The 1D counts stay at one divergent warp.

No URL: computed locally.

---

## Summary of corrections

Items where the book, the notes' assumptions, or a vendor's own page turned out WRONG, PARTLY right, or unverifiable:

- A18: `cudaDeviceProp.clockRate` was removed in CUDA 13.0. Use `cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, dev)`. The book's field list (p. 89) does not compile against current headers. HIP still has the member.
- A18: `regsPerBlock` is the per-block register limit ("32-bit registers available per block"). The per-SM pool the book describes is `regsPerMultiprocessor`. Both are 64 K on every compute capability NVIDIA currently lists, so the book's "less than the SM total on some levels" is not visible in today's tables.
- A1: NVIDIA's programming-model page expands GPC as "Graphics Processing Clusters"; other NVIDIA text says "GPU Processing Cluster". "Streaming processor" is not used in the current guide ("functional units", "FP32 CUDA Cores").
- A2: 132 SMs is the H100 SXM5. The PCIe H100 has 114. The level between GPC and SM is the TPC (2 SMs); the SXM5 has 66 TPCs in 8 GPCs, and NVIDIA does not publish the split.
- A3: "four processing blocks of 32" for H100 is not in any NVIDIA text found. The Volta whitepaper has four blocks of 16 for GV100 and two of 32 for GP100.
- A4: HBM3 is the SXM5 variant of H100 only; the PCIe variant has HBM2e.
- A5: "transparent scalability", "wave" and "tail effect" are the book's terms, not NVIDIA's. Blocks run to completion on their SM "in most cases"; dynamic parallelism can suspend one.
- A6: NVIDIA's rule is about "the same __syncthreads*() intrinsic call" and "non-exited threads": two call sites are two barriers, and a thread that has returned does not block the rest.
- A7: grid-wide `sync()` needs `cudaLaunchCooperativeKernel` and a grid that is resident all at once. Multi-device cooperative launch was removed in CUDA 13.
- A10: every compute capability in the current guide (7.5 and up) has independent thread scheduling. Strictly sequential passes are pre-Volta history.
- A12: per-SM limits differ across current GPUs (16, 24 or 32 blocks; 32, 48 or 64 warps; 1024, 1536 or 2048 threads). The book's worked occupancy cases are right for compute capability 9.0 and wrong for, say, 8.6 or 12.x. Read the limits from the device.
- A13: the book's register cliff is arithmetic on a per-thread model. NVIDIA says registers are partitioned among warps and does not publish the granularity; `cudaOccupancyMaxActiveBlocksPerMultiprocessor` is the authority for a real kernel.
- A15: a higher compute capability does not mean more resources per SM.
- A17: the book's reason for expecting several CUDA devices in a PC (integrated GPUs) has no NVIDIA source, and integrated GPUs from other vendors are not CUDA devices.
- B3: AMD's pages do not state the rule for `__syncthreads()` inside a conditional. Teach NVIDIA's rule as the portable one.
- B4: HIP's `__syncwarp` exists but its mask overload requires a 64-bit integer type; `__syncwarp(0xFFFFFFFF)` fails a static assert. It is not on the HIP language-extensions page.
- B5: AMD guarantees lockstep within a warp; NVIDIA since compute capability 7.0 does not.
- B6: AMD documents the linear order (x fastest) but not, in one sentence, that warps are consecutive runs of it. AMD's product table says "32 or 64" for Radeon wavefronts while HIP reports 32 on gfx10 and above.
- B7: AMD's "Good: no divergence" example `if (threadIdx.x < 32)` diverges on a 64-wide device.
- B9: on AMD, `maxBlocksPerMultiProcessor` is `maxThreadsPerCU / wavefrontWidth`, the wave-slot count, not an independent block limit; `regsPerBlock` equals `regsPerMultiprocessor` and counts vector registers only.
- B9: AMD's reference page says three properties "always" return 0 (`maxThreadsPerMultiProcessor`, `regsPerBlock`, `l2CacheSize`); the source at `therock-10.2` assigns them. The docs also mention a field `executionUnitsPerMultiprocessor` that is not in the header.
- B10: AMD's limits are wave slots, VGPRs, SGPRs and LDS, allocated per wave. Docs say up to 40 wave slots per CU on newer CDNA; the runtime's occupancy function uses 8 per SIMD (32 per CU) for ISA major version 9 and below.
- B10: `__launch_bounds__` has the same spelling on HIP, but its second argument is minimum warps per execution unit (SIMD), not minimum blocks per multiprocessor.
- B11: the notes assumed no cluster support outside NVIDIA. HIP now documents `cluster_group`, cluster launch attributes and `hipDeviceProp_t::clusterLaunch`; which AMD devices report support could not be found.
- C1: Metal's device object has no property for core count, SIMD width, clock rate, registers or resident threads. Its integrated-versus-discrete properties are deprecated as "Not applicable on Apple Silicon".
- C2: Metal cuts a threadgroup into SIMD-groups "in an implementation-defined fashion". Do not compute the group from the thread index; ask for `[[simdgroup_index_in_threadgroup]]`. No width is documented; llama.cpp and MLX assume 32 in places.
- C3: Metal documents no grid-wide barrier and no cluster level. `thread_scope_device` is described as a memory-ordering scope only.
- C4: Apple does not document how divergent paths in a SIMD-group execute or what they cost.
- C5: Metal has no documented occupancy concept, limit or API.
- E1: the book's "1 of 313" warps for 10,000 elements is 1 of 314 launched warps; its "80 of 1040" for the 200 x 150 picture is 75 or 100 depending on which number is the width.

## Where the backends really differ

| Feature | CUDA | HIP on AMD | Metal |
|---|---|---|---|
| Unit a block runs on | SM, grouped into GPCs (A1) | Compute unit (CU), grouped into shader arrays and shader engines (B1) | Not documented. The MSL spec uses the words "compute unit" without defining them (C2) |
| SIMD group name | warp | warp in HIP docs, wavefront in AMD ISA terms (B1) | SIMD-group (C2) |
| Width | 32 on every listed compute capability; "commonly 32" (A8, ch05 A5) | 64 on gfx9 (Instinct), 32 on gfx10 and above (ch05 B5) | No documented value (C2) |
| Where to read the width | `cudaDeviceProp.warpSize`, `warpSize` in device code | `hipDeviceProp_t.warpSize`, `warpSize` in device code | `MTLComputePipelineState.threadExecutionWidth` per pipeline; `[[threads_per_simdgroup]]` in the kernel |
| How a block is cut into groups | Documented: linear order with x fastest, consecutive runs of 32, thread 0 in warp 0 (A8) | Linear order documented, lane = index mod `warpSize` in AMD's examples; no one-sentence rule (B6) | "implementation-defined"; query `[[simdgroup_index_in_threadgroup]]` and `[[thread_index_in_simdgroup]]` (C2) |
| Threads of a group in lockstep | Not guaranteed from compute capability 7.0 (independent thread scheduling) (A10) | Guaranteed: "all warp lanes are executed in lockstep" (B5) | Not stated (C2) |
| Divergence model | Documented: each path executed with the other threads masked (A9) | Documented: both branches serialised, lanes masked through EXEC; "branch efficiency" metric (B7) | Not documented; inactive threads "due to flow control" are acknowledged (C4) |
| Block-level barrier | `__syncthreads()`; conditional rule documented (A6) | `__syncthreads()`; conditional rule not found (B3) | `threadgroup_barrier(mem_flags)`; conditional and loop rules documented (ch05 C2, C3) |
| Group-level barrier | `__syncwarp(unsigned mask = 0xFFFFFFFF)` (A10) | `__syncwarp()`; mask overload needs a 64-bit integer (B4) | `simdgroup_barrier(mem_flags)`, no mask (C3) |
| Lane masks | `unsigned`, 32 bits (A20) | `unsigned long long`, 64 bits, on all devices (B12) | `simd_vote` wrapper around an unspecified `vote_t` (C7) |
| Grid-wide barrier | `this_grid().sync()` with `cudaLaunchCooperativeKernel` (A7) | `this_grid().sync()` with `hipLaunchCooperativeKernel` (B11) | None documented; use separate dispatches (C3) |
| Clusters of blocks | Compute capability 9.0 and up; portable size 8 (ch05 A6) | API present; `clusterLaunch` only where the driver reports a size above 1; devices unknown (B11) | None (C3) |
| Limits per SM or CU | Block slots, warp or thread slots, registers, shared memory; all in `cudaDeviceProp` and Table 33 (A12) | Wave slots, VGPRs, SGPRs, LDS; no independent block-slot limit (B9, B10) | None exposed (C1, C5) |
| Occupancy function | `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyMaxPotentialBlockSize` (A14) | `hipOccupancyMaxActiveBlocksPerMultiprocessor`, `hipOccupancyMaxPotentialBlockSize`; block size rounded up to whole waves (B10) | None (C5) |
| Registers used by a kernel | `nvcc --resource-usage`; `cudaFuncAttributes.numRegs` (A13) | `hipcc --resource-usage`; `hipFuncAttributes.numRegs` = VGPRs only (B10) | Not exposed (C5) |
| Hint to the compiler | `__launch_bounds__(maxThreads, minBlocksPerMultiprocessor, maxBlocksPerCluster)`; `--maxrregcount` (A13) | `__launch_bounds__(maxThreads, minWarpsPerExecutionUnit)` (B10) | `[[max_total_threads_per_threadgroup(x)]]`; descriptor `maxTotalThreadsPerThreadgroup` (C5) |
| Threads-per-block cap | Device: 1024 (ch05 A5); lower for a kernel with launch bounds | Device: `maxThreadsPerBlock` from the driver's max work-group size (B9) | Per pipeline: `maxTotalThreadsPerThreadgroup`, "may be different for different pipeline objects" (C5) |
| Enumerating devices | `cudaGetDeviceCount`, `cudaGetDeviceProperties` (A16) | `hipGetDeviceCount` (documents `hipErrorNoDevice` for zero), `hipGetDeviceProperties` (B9) | `MTLCopyAllDevices()`, `MTLCreateSystemDefaultDevice()` (C1) |
| Clock rate | `cudaDeviceGetAttribute(..., cudaDevAttrClockRate, ...)`; struct member removed in 13.0 (A18) | `hipDeviceProp_t.clockRate` still a member (B9) | None (C1) |
| Meaning of `multiProcessorCount` | Number of SMs | Number of CUs, or half in WGP mode (ch05 B2) | No counterpart |
| Boundary check needed | Yes, grid is whole blocks | Yes | Not with `dispatchThreads` (nonuniform edge groups), yes with `dispatchThreadgroups` (ch05 C5, C4) |
| Tool that shows occupancy | Nsight Compute Occupancy Calculator (A14) | `rocprofv3 --occupancy` per AMD's guide (B10) | None documented (C5) |
