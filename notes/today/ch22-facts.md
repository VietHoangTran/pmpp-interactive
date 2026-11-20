# Chapter 22 fact check: decomposition, Amdahl, batching (CUDA, HIP, Metal, real servers)

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md` and `notes/today/infra-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY, and judges the assumption stated in the item title (usually the book's sentence, with its page from `notes/ch22.md`). **Quote** is copied from the source with line breaks removed.
- Chapter 22 has no code and names no device, API version or GPU. Group A therefore checks the few statements that rest on vendor or literature facts. Groups B and C give the HIP and Metal counterparts of the CUDA features the chapter leans on (streams, concurrent kernels, atomics) and of the two library or runtime features the lessons will need to name (graphs, batched GEMM). Group D is the bridge for the reader's third goal: how real servers and backends batch. Group E holds the arithmetic the lessons and the M8 lab will rely on, all ours.
- Already settled elsewhere, cited by item number and not repeated: see the table below.
- Versions the sources reported on the check date: CUDA Programming Guide v13.5 (pages "Last updated on Oct 31, 2026"); CUDA C++ Best Practices Guide and cuBLAS docs ("Last updated on Nov 02, 2026", cuBLAS 13.5); NVIDIA Deep Learning Performance guides (matrix multiplication and GPU background pages "Last updated on Feb 1, 2023", fully-connected page "Jul 26, 2023"); HIP docs and headers at tag `therock-10.2` of `ROCm/rocm-systems` (ROCm Core SDK 10.2.0); hipBLAS header at tag `therock-10.2` of `ROCm/rocm-libraries`; Apple documentation JSON and the macOS 27.0 SDK headers on this machine; PyTorch docs 2.15. Project files were read on their default branches today; the last-commit date of each file is given where it matters.
- Not checked: anything by running code. Every statement is from documentation, headers, papers or project source.
- **Device figures in this file are evidence for a verdict, not lesson content.** Several NVIDIA pages quoted below argue with V100 or A100 numbers; the lessons keep the project rule (the book's H100 figures only as the book's example, the reader's figures from `python3 labs/pmpp.py doctor` and lab benchmarks).

Totals: 26 CONFIRMED, 0 WRONG, 3 PARTLY, 0 COULD NOT VERIFY (29 items: A1 to A11, B1 to B3, C1 to C3, D1 to D8, E1 to E4). Sub-claims that could not be verified are called out inside A1, A5 and C2 and collected under "Summary of corrections".

---

## Already settled elsewhere (cite, do not re-check)

| Topic chapter 22 touches | Where it is settled |
|---|---|
| CPU latency-oriented, GPU throughput-oriented (p. 540) | ch01 A9 (A8 below adds one quote) |
| Amdahl's formula and the chapter 1 numbers | ch01 A16 |
| llama.cpp's default sampler runs on the host (an Amdahl remainder) | ch01 D2 |
| Streams, `cudaMemcpyAsync`, pinned memory, events and their HIP names | infra A8, B8 |
| Metal command queues and buffers as the analogue of streams; cross-queue concurrency not documented | infra C12 |
| Serial versus concurrent compute encoders, hazard tracking, MTL4 queues untracked | ch02 C8, C9 |
| Legacy atomics, `<cuda/atomic>`, HIP atomics (CAS-safe float add by default), MSL atomics (float add/sub only) | infra A1, A2, B1, B2, C1 |
| Same-address atomics serialise; device atomics run in the L2; latency in cycles not documented | infra A1, ch09 A3, A4 |
| AMD atomics in the L2 and LDS; Apple documents neither place nor latency | ch09 B2, C4 |
| Kernel timing: `cudaEventElapsedTime`, `hipEventElapsedTime`, Metal `GPUEndTime - GPUStartTime` per command buffer | infra A8, B8, ch05 C9 |
| Apple publishes no peak FLOPS, bandwidth, cache or bank figures | infra C13 |
| Registers and shared memory: per-thread and per-block on all three backends | ch05 A1, A7, B4, C1, C8 |
| HIP targets AMD GPUs only; llama.cpp builds its HIP backend from the CUDA sources | ch01 B1, ch05 B9 |
| Library GEMM on each backend (cuBLAS, hipBLAS, MPS) | infra A12, B12, C11 |
| Where llama.cpp, MLX and PyTorch keep their kernels | infra D1 to D4 |

---

## Group A: the book's statements, checked against NVIDIA documentation and the literature

### A1. Gather is "usually more desirable in CUDA devices" (private registers, shared inputs); scatter needs atomics that are "significantly slower than the register accesses" (p. 533)

**Verdict: PARTLY.** The mechanism is documented: registers cost nothing extra per instruction, and atomics on one address serialise and run in the L2. The size of the gap ("significantly slower") is not documented anywhere I read; no NVIDIA page gives an atomic latency or throughput to compare with a register.

- Registers, Best Practices Guide section 10.2.7: "Generally, accessing a register consumes zero extra clock cycles per instruction, but delays may occur due to register read-after-write dependencies and register memory bank conflicts."
- Registers, Programming Guide section 2.3.3.3: "Registers are located on the SM and have thread local scope. Register usage is managed by the compiler and registers are used for thread local storage during the execution of a kernel."
- Atomics on one address serialise (Programming Guide section 3.2.2, quoted in infra A1): "each read/modify/write to that location occurs and they are all serialized, but the order in which they occur is undefined."
- Where device-scope atomics execute (L2) and that no latency figure is published: ch09 A4.
- Sharing inputs through shared memory or caches: shared memory is per block (ch05 A1); NVIDIA's own histogram example in section 2.3.3.8 is the privatise-then-commit pattern, which is the book's way of turning a scatter into mostly on-chip work (ch09 A6).

What the lesson can say: gather keeps the running sum in a register and writes once; scatter turns every contribution into a read-modify-write on shared memory or global memory, and conflicting writes to one address are serialised. Do not attach a speed ratio; measure it (chapter 9's labs did).

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 10.2.7)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (sections 2.3.3.3, 2.3.3.8)
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (section 3.2.2)

### A2. "CUDA devices support task parallelism with streams"; several small kernels can run simultaneously (p. 538)

**Verdict: CONFIRMED.** NVIDIA documents concurrent kernel execution through non-default streams, a device property that reports it, and a per-device limit on resident grids. It also documents the conditions: no implicit synchronisation through the default stream, no event dependency, and enough free resources.

- Best Practices Guide section 11.5 "Concurrent Kernel Execution": "On devices that are capable of concurrent kernel execution, streams can also be used to execute multiple kernels simultaneously to more fully take advantage of the device's multiprocessors. Whether a device has this capability is indicated by the concurrentKernels field of the cudaDeviceProp structure". And: "Non-default streams (streams other than stream 0) are required for concurrent execution because kernel calls that use the default stream begin only after all preceding calls on the device (in any stream) have completed, and no operation on the device (in any stream) commences until they are finished."
- Programming Guide section 3.1.3 "More on Streams and Events": "Work submitted on different CUDA streams may execute concurrently under specific circumstances, e.g., if there are no event dependencies, if there is no implicit synchronization, if there are sufficient resources, etc."
- Programming Guide section 2.5.8: "Two operations from different streams cannot run concurrently if any CUDA operation on the NULL stream is submitted in-between them, unless the streams are non-blocking streams (created with the cudaStreamNonBlocking flag)."
- Runtime API, `cudaDeviceProp`: `int concurrentKernels` "Device can possibly execute multiple kernels concurrently."
- Compute capability Table 33: "Maximum number of resident grids per device (Concurrent Kernel Execution) | 128", one value spanning every listed compute capability (7.5 to 12.x).

The book's "may" is the right word: the guide promises possibility, not overlap. A lesson that shows overlap should show it from a timeline the reader produces, not assert it.

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 11.5)
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html (section 3.1.3)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html (section 2.5.8)
- https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 33)

### A3. Not in the book: CUDA Graphs as NVIDIA's documented remedy for many short kernels (the "many small activities" of p. 537)

**Verdict: CONFIRMED.** The book answers Amdahl's small activities with streams (forward reference to chapter 23; `notes/ch23.md` mentions no graph API). NVIDIA now documents graphs specifically for the per-launch CPU cost that dominates short kernels. Label this `<Beyond>` in the lesson.

- Section 4.2 "CUDA Graphs": "Separating the definition of a graph from its execution enables a number of optimizations: first, CPU launch costs are reduced compared to streams, because much of the setup is done in advance". And: "These operations, necessary for setting up and launching the kernel, are an overhead cost which must be paid for each kernel that is issued. For a GPU kernel with a short execution time, this overhead cost can be a significant fraction of the overall end-to-end execution time. By creating a CUDA graph that encompasses a workflow that will be launched many times, these overhead costs can be paid once for the entire graph during instantiation, and the graph itself can then be launched repeatedly with very little overhead."
- Section 2.5.9.2: "Capturing or creating a graph can help reduce latency and CPU overhead of repeatedly invoking the same chain of API calls from the host thread." The three steps it lists: capture (or build with the graph API), instantiate once, then launch the instantiated graph "as many times as required".
- The same guide's sample comment: "tuned such that kernel takes a few microseconds" for its "very lightweight kernel". (An NVIDIA example size, not a reader figure.)

Link trap: section 2.5.1 of the guide refers to "section 2.4.9.2 Introduction to CUDA Graphs with Stream Capture" and "section 4.1 CUDA Graphs"; the pages are numbered 2.5.9.2 and 4.2. Link by URL, not by section number.

How real code uses it: D7.

- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html (section 4.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html (sections 2.5.1, 2.5.9.2)

### A4. Batching turns the QKV projection's vector-matrix products into one matrix product that reuses the weights, raising arithmetic intensity (p. 539)

**Verdict: CONFIRMED.** NVIDIA's deep-learning performance guides make the same argument with the same formula, state that a matrix-vector product is always memory-limited, and print the very linear-layer example chapter 20 uses (1 FLOPS/B at batch 1, 315 at batch 512). These pages date from 2023 and argue with V100 and A100 figures; use the reasoning, not the device numbers.

- Matrix multiplication guide: "Arithmetic Intensity = number of FLOPS / number of byte accesses = 2·(M·N·K) / 2·(M·K + N·K + M·N) = M·N·K / (M·K + N·K + M·N)". The factor 2 in the denominator is 2 bytes per element (the page assumes "FP16 inputs").
- Same page: "matrix-vector products (general matrix-vector product or GEMV), where either M=1 or N=1, are always memory limited; their arithmetic intensity is less than 1." In FP32 the same count gives less than 0.5 FLOP/B (E1).
- Same page, the caveat the lesson should carry: "the comparison of arithmetic intensity with the ops:byte ratio is a simplified rule of thumb, and does not consider many practical aspects of implementing this computation (such as non-algorithm instructions like pointer arithmetic, or the contribution of the GPU's on-chip memory hierarchy)."
- GPU performance background guide, the roofline in words: "an algorithm is math limited on a given processor if the algorithm's arithmetic intensity is higher than the processor's ops:byte ratio. Conversely, an algorithm is memory limited if its arithmetic intensity is lower than the processor's ops:byte ratio." Its Table 1, "Limiters assume FP16 data": "Linear layer (4096 outputs, 1024 inputs, batch size 512) | 315 FLOPS/B | arithmetic"; "Linear layer (4096 outputs, 1024 inputs, batch size 1) | 1 FLOPS/B | memory"; "ReLU activation | 0.25 FLOPS/B | memory"; "Layer normalization | < 10 FLOPS/B | memory". This is the source of chapter 20's p. 505 example (`notes/ch20.md`).
- Fully-connected layer guide: "Larger values for batch size and the number of inputs and outputs improve parallelization and efficiency" and "the computation will always be bandwidth-limited for very small batch sizes, for example, 8 and below." (Its A100-specific threshold of 128 is device evidence, not lesson content.)

- https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
- https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html (Table 1)
- https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html (sections 1 and 3.2)

### A5. The LLM decode step is memory-bound, and batching requests is how servers raise throughput (p. 539)

**Verdict: CONFIRMED** for the two claims in the title. The book's third claim, that each query's own projection latency grows with the batch, is not stated by any vendor source I read; it is the book's reasoning and it is exactly what M8 measures. Present it as the prediction, then let the reader's numbers confirm or refute it.

NVIDIA technical blog "Mastering LLM Techniques: Inference Optimization" (Nov 17, 2023):

- Decode: "This is like a matrix-vector operation that underutilizes the GPU compute ability compared to the prefill phase. The speed at which the data (weights, keys, values, activations) is transferred to the GPU from memory dominates the latency, not how fast the computation actually happens. In other words, this is a memory-bound operation."
- Batching: "The simplest way to improve GPU utilization, and effectively throughput, is through batching. Since multiple requests use the same model, the memory cost of the weights is spread out."
- "Model execution is frequently memory-bandwidth bound—in particular, bandwidth-bound in the weights."
- The limit: "Batch sizes, however, can only be increased up to a certain limit, at which point they may lead to a memory overflow." (KV cache; chapter 20's point.)

- https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

### A6. Batching "increases the arithmetic intensity of the attention heads" (p. 539), against chapter 20's "attention layers do not benefit from batching" (p. 505)

**Verdict: PARTLY.** True only if "attention" includes the QKV and output projections. The paper that introduced iteration-level batching (D1) says the attention operation proper is not batched across requests because it has no weights to reuse, and it names the two definitions of "attention" explicitly. Teach chapter 20's version and flag p. 539's wording, as `notes/ch22.md` already suggests.

Orca (OSDI '22), section 1:

- "selective batching splits the batch and processes each request individually for the Attention operation while applying batching to other operations of the Transformer model. We observe that the decision not to batch the executions of the Attention operation has only a small impact on efficiency. Since the Attention operation is not associated with any model parameters, applying batching to Attention has no benefit of reducing the amount of GPU memory reads by reusing the loaded parameters across multiple requests."
- Footnote 1: "In some literature the Attention operation has an extended definition that includes linear layers (QKV Linear and Attn Out Linear; Figure 1b). On the other hand, we use a narrow definition as described in Figure 1b."

This is a paper, not a vendor document; it is the primary source for the serving technique, and NVIDIA's TensorRT-LLM docs use its name for it (D1).

- https://www.usenix.org/conference/osdi22/presentation/yu
- https://www.usenix.org/system/files/osdi22-yu.pdf

### A7. Flash attention shortens latency for one input "by eliminating redundant accesses to global memory" (p. 539)

**Verdict: CONFIRMED** by the paper's own abstract (the book cites chapter 20 for it).

- "We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention".

The abstract reports wall-clock speedups for training; the book's use of it as the example of a latency optimisation is a fair reading but the word "latency" is the book's.

- https://arxiv.org/abs/2205.14135

### A8. The latency-versus-throughput trade-off is the same as CPU versus GPU design (p. 540)

**Verdict: CONFIRMED.** Settled in ch01 A9 from the Programming Guide. The Best Practices Guide, section 3.1 "Differences between Host and Device", says it in the book's terms: "In short, CPU cores are designed to minimize latency for a small number of threads at a time each, whereas GPUs are designed to handle a large number of concurrent, lightweight threads in order to maximize throughput."

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 3.1)

### A9. Amdahl's worked example: 95 % accelerated 100 times gives "17x", or 20x with overlap (pp. 537 to 538)

**Verdict: CONFIRMED** by arithmetic and by NVIDIA's statement of the law. 1 / (0.05 + 0.95/100) = 1 / 0.0595 = 16.807, printed as 17 (rounding, as `notes/ch22.md` says). With the device time hidden behind the host's 5 %: 1 / 0.05 = 20.

- Best Practices Guide section 4.1.3.1 "Strong Scaling and Amdahl's Law": "the maximum speedup S of a program is: S = 1 / ((1 - P) + P/N). Here P is the fraction of the total serial execution time taken by the portion of code that can be parallelized and N is the number of processors over which the parallel portion of the code runs."
- Section 4.1.3.2 "Weak Scaling and Gustafson's Law": "Weak scaling is a measure of how the time to solution changes as more processors are added to a system with a fixed problem size per processor; i.e., where the overall problem size increases as the number of processors is increased." and "Another way of looking at Gustafson's Law is that it is not the problem size that remains constant as we scale up the system but rather the execution time."

A bridge that is ours, not the book's or NVIDIA's: section 22.5's two motivations are strong and weak scaling. "Finish this one problem sooner" is Amdahl's question; "finish more problems in the same time" (batching) is Gustafson's. Label it `<Beyond>` if used.

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (sections 4.1.3.1, 4.1.3.2)

### A10. The chapter's one reference: J. M. Wing, "Computational thinking", CACM 49(3), 2006 (p. 540)

**Verdict: CONFIRMED.** Crossref record: title "Computational thinking", author Jeannette M. Wing, *Communications of the ACM*, volume 49, issue 3, pages 33 to 35, published March 2006, DOI 10.1145/1118178.1118215.

- https://doi.org/10.1145/1118178.1118215
- https://api.crossref.org/works/10.1145/1118178.1118215

### A11. Not in the book: cuBLAS "batched GEMM" is a different thing from the book's batching

**Verdict: CONFIRMED** (API facts; the distinction is ours). The book batches requests by stacking their vectors into one matrix that multiplies one shared weight matrix: one GEMM with M equal to the number of requests. cuBLAS's "batched" routines run many independent GEMMs, each with its own A[i] and B[i]; they are what per-sequence attention needs (each request has its own K and V, A6), not what the shared projection needs. The reader will meet both names in llama.cpp (D6).

- `cublas<t>gemmBatched()` (section 2.7.3) and `cublasGemmBatchedEx()` (2.8.13): "This function performs the matrix-matrix multiplication of a batch of matrices. The batch is considered to be "uniform", i.e. all instances have the same dimensions (m, n, k), leading dimensions (lda, ldb, ldc) and transpositions (transa, transb) for their respective A, B and C matrices. The address of the input matrices and the output matrix of each instance of the batch are read from arrays of pointers passed to the function by the caller."
- `cublas<t>gemmStridedBatched()` (2.7.4) and `cublasGemmStridedBatchedEx()` (2.8.14): "Input matrices A, B and output matrix C for each instance of the batch are located at fixed offsets in number of elements from their locations in the previous instance." Formula as printed: C + i*strideC = alpha op(A + i*strideA) op(B + i*strideB) + beta (C + i*strideC), for i in [0, batchCount - 1].
- cuBLAS stores matrices column-major ("arrays of pointers to matrices stored in column-major format"); the project's labs are row-major, so any comparison against cuBLAS needs the usual transpose trick. Not needed for M8.

- https://docs.nvidia.com/cuda/cublas/index.html (sections 2.7.3, 2.7.4, 2.8.13, 2.8.14)

---

## Group B: HIP / ROCm

### B1. Concurrent kernels on AMD: same model, but a documented cap of 4 hardware queues per device by default (counterpart of A2)

**Verdict: CONFIRMED,** with one documented difference that matters for task-level parallelism: streams beyond the hardware-queue count share queues round-robin.

- HIP "Asynchronous concurrent execution": "Streams are FIFO buffers of commands to execute in order on a given device. ... Multiple streams tied to the same device are not guaranteed to execute their commands in order."
- Same page, "Concurrent kernel execution": "Independent kernels can only run concurrently if there are enough registers and shared memory for the kernels. To enable concurrent kernel executions, the developer may have to reduce the block size of the kernels." And: "Multiple kernels executing concurrently is only beneficial under specific conditions. It is most effective when the kernels do not fully utilize the GPU's resources." This is the book's p. 538 condition, stated by AMD.
- Same page, stream flags: "hipStreamDefault: creates a default stream suitable for most operations. The default stream is a blocking operation." and "hipStreamNonBlocking: creates a non-blocking stream, allowing concurrent execution of operations."
- HIP environment variables table: "GPU_MAX_HW_QUEUES | The maximum number of hardware queues allocated per device. | 4 | The recommended maximum value for this setting is 4. The variable controls how many independent hardware queues HIP runtime can create per process, per device. If an application allocates more HIP streams than this number, then HIP runtime reuses the same hardware queues for the new streams in a round-robin manner."
- Header at the tag: `int concurrentKernels; ///< Device can possibly execute multiple kernels concurrently.` in `hipDeviceProp_t` (same name as CUDA).

Not checked: whether a HIP null-stream launch blocks other streams in exactly CUDA's legacy way. AMD's page says only that the default stream "is a blocking operation".

Mapped in `pmpp/vendors/hip.h`: the stream calls already are (infra B8); nothing new.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html (source: `projects/hip/docs/how-to/hip_runtime_api/asynchronous.rst` at `therock-10.2`)
- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html (source: `projects/hip/docs/data/env_variables_hip.rst`, line 146)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (line 322)

### B2. HIP graphs (counterpart of A3)

**Verdict: CONFIRMED.** Same model and the same API shape with the `hip` prefix: `hipStreamBeginCapture(hipStream_t, hipStreamCaptureMode)`, `hipGraphInstantiate(hipGraphExec_t*, hipGraph_t, ...)`, `hipGraphLaunch`. AMD frames the overhead differently from NVIDIA.

- "HIP graphs are an alternative way of executing tasks on a GPU that can provide performance benefits over launching kernels using the standard method via streams."
- "The standard method of launching kernels incurs a small overhead for each iteration of the operation involved. That overhead is negligible, when the kernel is launched directly with the HIP C/C++ API, but depending on the framework used, there can be several levels of redirection, until the actual kernel is launched by the HIP runtime, leading to significant overhead. Especially for some AI frameworks, a GPU kernel might run faster than the time it takes for the framework to set up and launch the kernel".
- Its figure caption: "Qualitative presentation of the execution time of many short-running kernels when launched using HIP stream versus HIP graph. This does not include the time needed to set up the graph."

Difference worth a sentence: NVIDIA calls the per-kernel launch cost potentially "a significant fraction" for short kernels (A3); AMD calls it "negligible" for direct HIP launches and significant through framework layers. Neither gives a number.

Not mapped in `pmpp/vendors/hip.h` (the course's labs use no graphs). llama.cpp maps them in its own header (D7).

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html (source: `projects/hip/docs/how-to/hip_runtime_api/hipgraph.rst` at `therock-10.2`)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 8903, 9197)

### B3. hipBLAS batched GEMM (counterpart of A11)

**Verdict: CONFIRMED.** Same four entry points with the `hipblas` prefix in `hipblas.h` at the tag: `hipblasSgemmBatched` (line 14944), `hipblasSgemmStridedBatched` (15325), `hipblasGemmBatchedEx` (24084), `hipblasGemmStridedBatchedEx` (24479). Header comment: "The gemmStridedBatched functions perform one of the strided batched matrix-matrix operations: C_i = alpha*op( A_i )*op( B_i ) + beta*C_i, for i = 1, ..., batchCount". The header also declares grouped variants (`hipblasDgemmGroupedBatched_64`, ...) for batches whose sizes differ.

- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipblas/library/include/hipblas.h

---

## Group C: Metal

### C1. Several independent kernels at once: a concurrent compute encoder plus `memoryBarrier(scope:)` (counterpart of A2)

**Verdict: CONFIRMED.** Metal's documented way to let dispatches overlap is a compute encoder created with `MTLDispatchTypeConcurrent` (ch02 C9: "Command encoder dispatches are executed in parallel with each other"), with the programmer inserting barriers where a later dispatch reads an earlier one's output. Between queues, overlap is not documented (infra C12).

- Header, `MTLComputeCommandEncoder.h` (macOS 27.0 SDK), `memoryBarrierWithScope:`: "Encodes a barrier between currently dispatched kernels in a concurrent compute command encoder and any subsequent ones on a specified resource group". Discussion: "This API ensures that all dispatches in the encoder have completed execution and their side effects are visible to subsequent dispatches in that encoder. Calling barrier on a serial encoder is allowed, but ignored." Available macOS 10.14 and later. Scopes: `MTLBarrierScopeBuffers`, `MTLBarrierScopeTextures`, `MTLBarrierScopeRenderTargets` (macOS only).
- Web docs, `memoryBarrier(scope:)`: "Creates a memory barrier that enforces the order of write and read operations for specific resource types." "To determine whether a GPU supports memory barriers, see the Metal-Feature-Set-Tables.pdf." (I did not look up the row.)
- Metal 4 (`MTL4` types, macOS 26 and later): "In Metal 4, the framework considers all resources untracked. You need to synchronize pipeline stages that can concurrently access a resource if any of the shaders in these pipelines modify it." Within one pass the barrier is `barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:`, which "guarantees that any subsequent work you encode in the current command encoder ... doesn't begin until all prior commands in this command encoder ... completes." Apple's pages do not state in so many words that an `MTL4ComputeCommandEncoder`'s dispatches overlap by default; the untracked-resources rule means the author must assume they may.

Both llama.cpp and MLX use exactly this (D8).

- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/memorybarrier(scope:)
- https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api
- Local headers: `MTLComputeCommandEncoder.h` (lines 315 to 319), `MTLCommandEncoder.h` (lines 27 to 32, 95 to 116)

### C2. Indirect command buffers as the closest Metal counterpart of graphs (counterpart of A3)

**Verdict: PARTLY.** Apple documents indirect command buffers (ICBs) as a way to "encode commands once and reuse them" and to "Reduce CPU overhead", and they can hold compute dispatches. That much is the graph idea. The analogy is ours: Apple never compares ICBs with CUDA graphs, and an ICB is a list of commands, not a graph with dependency edges.

- `MTLIndirectCommandBuffer` (macOS 10.14+): "A command buffer containing reusable commands, encoded either on the CPU or GPU." "Use an indirect command buffer to encode commands once and reuse them, and to encode commands on multiple CPU or GPU threads."
- Sample "Encoding indirect command buffers on the CPU": "Reduce CPU overhead and simplify your command execution by reusing commands." "Because Metal discards a normal command buffer and its commands after Metal executes them, use ICBs to save expensive allocation, deallocation, and encoding time for your app's common instructions."
- Compute commands: `MTLIndirectCommandType.concurrentDispatch` ("A compute command using a grid aligned to threadgroup boundaries", macOS 11.0+) and `concurrentDispatchThreads`; header comments "Dispatch threadgroups with concurrent execution" and "Dispatch threads with concurrent execution". Executed with `executeCommandsInBuffer:withRange:` ("The same indirect command buffer may be executed any number of times within the same encoder", macOS 11.0+).
- `MTL4ComputeCommandEncoder`: "you can't use a barrier to wait for all commands in an indirect command buffer to complete."

Not verified: any Apple statement of how much CPU time an ICB saves, and whether either llama.cpp or MLX uses ICBs (I searched only `ggml-metal-context.m`, `ggml-metal-ops.cpp`, `ggml-metal-device.m`, `ggml-metal-device.cpp` and MLX's `device.cpp`; none mention them).

- https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer
- https://developer.apple.com/documentation/metal/encoding-indirect-command-buffers-on-the-cpu
- https://developer.apple.com/documentation/metal/mtlindirectcommandtype/concurrentdispatch
- https://developer.apple.com/documentation/metal/mtl4computecommandencoder
- Local headers: `MTLIndirectCommandBuffer.h` (lines 25 to 27), `MTLComputeCommandEncoder.h` (lines 297 to 310)

### C3. Batched matrix multiplication in Metal Performance Shaders (counterpart of A11)

**Verdict: CONFIRMED.** An `MPSMatrix` can hold several matrices, and `MPSMatrixMultiplication` (through `MPSMatrixBinaryKernel`) runs over a range of them.

- `MPSMatrix.matrices` (macOS 10.13+): "The number of matrices in the MPSMatrix."
- `batchStart`: "The index of the first matrix in the batch. This property is modifiable and defaults to 0 at initialization time." `batchSize` is the count (web page has only the declaration `var batchSize: Int { get set }`).
- `MPSMatrixMultiplication`: "C = alpha * op(A) * op(B) + beta * C".

- https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication
- https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixbinarykernel/batchstart
- https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix/matrices
- Local headers: `MPSMatrix.framework/.../MPSMatrixTypes.h` (lines 55 to 62), `MPSCore.framework/.../MPSMatrix.h` (lines 263 to 266)

---

## Group D: how real servers and backends batch (the bridge, and goal 3)

All repository files read on their default branches today. These projects move daily; link to directories or files, and treat line numbers and option defaults as a snapshot.

### D1. The technique and its three names: iteration-level scheduling, in-flight batching, continuous batching

**Verdict: CONFIRMED.** Chapter 20 calls it "in-flight batching" (p. 506, citing vLLM's paper); the course's bridge calls it continuous batching. They are the same thing, and NVIDIA's docs say so.

- Orca (OSDI '22), abstract: existing systems use "inflexible scheduling mechanism that cannot change the current batch of requests being processed; requests that have finished earlier than other requests in a batch cannot return to the client, while newly arrived requests have to wait until the current batch completely finishes. In this paper, we propose iteration-level scheduling, a new scheduling mechanism that schedules execution at the granularity of iteration (instead of request) where the scheduler invokes the execution engine to run only a single iteration of the model on the batch." (USENIX's page prints the headline result as "36:9× throughput improvement at the same level of latency", a typo for 36.9.)
- TensorRT-LLM docs, "Paged Attention, IFB, and Request Scheduling" (file last changed 2026-11-03): "TensorRT LLM supports in-flight batching of requests (also known as continuous batching or iteration-level batching) for higher serving throughput. With this feature, sequences in the context phase can be processed together with sequences in the generation phase. The purpose of that technique is to better interleave requests to reduce latency as well as make better use of the GPUs."
- NVIDIA blog (A5): "Traditional batching (also called static batching) is suboptimal. ... all requests in the batch must wait until the longest request is finished" and "With in-flight batching, rather than waiting for the whole batch to finish before moving on to the next set of requests, the server runtime immediately evicts finished sequences from the batch. It then begins executing new requests while other requests are still in flight."

Why it belongs in chapter 22 (ours): continuous batching is what keeps the decode batch full, so the server lives at a large B on the reader's M8 curve; the cost each user pays is the per-step time at that B.

- https://www.usenix.org/conference/osdi22/presentation/yu
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/paged-attention-ifb-scheduler.md
- https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

### D2. vLLM: the batch-size knobs and the latency-throughput trade-off in its own docs

**Verdict: CONFIRMED.**

- README (last changed 2026-11-06): "Continuous batching of incoming requests, chunked prefill, prefix caching".
- `vllm/config/scheduler.py` (2026-11-19): `max_num_seqs` "Maximum number of sequences to be processed in a single iteration."; `max_num_batched_tokens` "Maximum number of tokens that can be processed in a single iteration."
- `docs/configuration/optimization.md` (2026-11-12), chunked prefill: "This feature helps improve both throughput and latency by better balancing compute-bound (prefill) and memory-bound (decode) operations." "It batches all pending decode requests before scheduling any prefill operations." Tuning: "Smaller values (e.g., 2048) achieve better ITL because there are fewer prefills slowing down decodes." "Higher values achieve better time to first token (TTFT) as you can process more prefill tokens in a batch." And when memory runs short: "Decrease `max_num_seqs` or `max_num_batched_tokens`. This reduces the number of concurrent requests in a batch, thereby requiring less KV cache space."
- Same file, graphs: "Serve without CUDA graphs using `--enforce-eager`. Skips both compilation and CUDA-graph capture for the fastest possible startup, at the cost of steady-state decode performance." `vllm/config/compilation.py` (2026-11-20) captures graphs per size: `cudagraph_capture_sizes` "Sizes to capture cudagraph", marked deprecated with "model runner V2" in favour of new options. Do not teach option names from this file; they are changing.
- The same docs mention ROCm ("On CUDA and ROCm, the flag also turns off the extensible KV cache"), so vLLM is a CUDA and ROCm server. I did not check its Metal status.

- https://github.com/vllm-project/vllm/blob/main/README.md
- https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py
- https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md (rendered: https://docs.vllm.ai/en/latest/configuration/optimization/)
- https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py

### D3. llama.cpp: continuous batching in `llama-server`, and `llama-batched-bench`, which is M8 as a tool

**Verdict: CONFIRMED.**

`llama-server` (`tools/server/`, last commit 2026-11-20):

- README features: "Parallel decoding with multi-user support", "Continuous batching". Options: "`-np, --parallel N` | number of server slots (default: -1, -1 = auto)" and "`-cb, --cont-batching, -nocb, --no-cont-batching` | whether to enable continuous batching (a.k.a dynamic batching) (default: enabled)". Metric: "`llamacpp:n_busy_slots_per_decode` | Gauge | Average number of busy slots per llama_decode() call."
- `server-context.cpp`: each iteration starts with `// start populating the batch for this iteration` and `batch.clear();`, adds every generating slot's sampled token (`// update the batch with the sampled/drafted tokens`), then `// next, batch any pending prompts without exceeding n_batch` under `if (params_base.cont_batching || batch.size() == 0)`. So one `llama_decode` carries one decode token per active request plus prompt chunks of newcomers: Orca's iteration-level scheduling.

`llama-batched-bench` (`tools/batched-bench/`, last commit 2026-09-30):

- README: "Benchmark the batched decoding performance of `llama.cpp`". Columns: "`B` - number of batches", "`T_TG` - time to generate all batches", "`S_TG` - text generation speed (`(B*TG)/T_TG`)", "`T_PP` - prompt processing time (i.e. time to first token)". Example invocation: `-npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32`.
- `common/arg.cpp`: `-npl` "number of parallel prompts"; `-pps` "is the prompt shared across parallel sequences"; `-tgs` "is the text generation separated across the different sequences". The README does not document `-tgs`.
- `batched-bench.cpp`: with `-tgs` the loop decodes one sequence at a time (`// decode pattern: // 0 0 0 ... 1 1 1 ...`); without it, one batch per step holds one token from every sequence (`// 0123 0123 0123 ...`). That is the book's "B vector-matrix products" against "one matrix product", on a real model, timed by `ggml_time_us()` on the host.

For M8: the lab measures the same quantity on the project's own kernels; `llama-batched-bench` is the "see it in a real engine" follow-up. The README's sample table is one machine's numbers and must not be quoted in a lesson.

- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp (lines 3799 to 3948)
- https://github.com/ggml-org/llama.cpp/blob/master/tools/batched-bench/README.md
- https://github.com/ggml-org/llama.cpp/blob/master/tools/batched-bench/batched-bench.cpp (lines 177 to 225)
- https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp (lines 2545 to 2550, 3260 to 3296)

### D4. MLX: `mlx-lm` serves with continuous batching on Metal

**Verdict: CONFIRMED.** (MLX core is an array library; serving lives in `ml-explore/mlx-lm`, `mlx_lm/server.py` and `generate.py`, both last changed 2026-11-19.)

- `generate.py`: `class BatchGenerator:` docstring "A batch generator implements continuous batching. This class provides automatic management of prompt processing and generation batches, handling the transition between the two." Defaults `completion_batch_size: int = 32`, `prefill_batch_size: int = 8`.
- `server.py` options: `--decode-concurrency` (default 32) "When a request is batchable then decode that many requests in parallel"; `--prompt-concurrency` (default 8) "When a request is batchable then process that many prompts in parallel".
- Not batched with speculative decoding: `is_batchable = draft_model is None`, and every cache layer must support `merge`.

- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py

### D5. Hugging Face TGI is in maintenance mode and archived; do not use it as the current example

**Verdict: CONFIRMED.** README: "text-generation-inference is now in maintenance mode. Going forward, we will accept pull requests for minor bug fixes, documentation improvements and lightweight maintenance tasks." It recommends "vllm, SGLang, as well as local engines with inter-compatibility such as llama.cpp or MLX". The GitHub repository reports `archived: true` (last push 2026-03-21). Its feature list still says "Continuous batching of incoming requests for increased total throughput"; that is history, not a current recommendation.

- https://github.com/huggingface/text-generation-inference/blob/main/README.md

### D6. Where a real backend switches from GEMV kernels to GEMM kernels as the batch grows

**Verdict: CONFIRMED.** llama.cpp has separate small-batch matrix-vector kernels and picks them by the number of tokens in the batch (`ne11`), with thresholds tuned per architecture and weight type. It is the book's p. 539 point as a dispatch rule.

- CUDA and HIP (same sources, ch05 B9): `mmvf.cuh` `#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.`; `mmvq.cuh` `#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.` `ggml_cuda_should_use_mmvq` lowers the limit for some quantised types on specific GPUs, with comments `// tuned on RTX 4090` and `// tuned on RTX 5090` and the reason "k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner." `ggml_cuda_should_use_mmvf` has branches for AMD (`GGML_CUDA_CC_IS_RDNA(cc)`).
- Above the threshold, a plain 2-D weight times a batch of tokens goes to one `cublasSgemm` or `cublasGemmEx` (`ggml_cuda_mul_mat_cublas_impl`, `ne12 == 1 && ne13 == 1`); `cublasGemmStridedBatchedEx` and `cublasGemmBatchedEx` are used only for 3-D and 4-D operands, with broadcast factors `r2 = ne12/ne02`, `r3 = ne13/ne03` (A11). Comment: "Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix. However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower".
- Metal (`ggml-metal-ops.cpp`, 2026-11-19): `// first try to use small-batch mat-mv kernels // these should be efficient for BS [2, ~8]`, condition `(ne11 >= 2 && ne11 <= 8)` for most types and `(ne11 >= 4 && ne11 <= 8)` for k-quants, kernel `mul_mv_ext`; otherwise `mul_mm` (simdgroup matrices) when `ggml_metal_op_mul_mat_use_mm` allows.

For the lesson: "8" is a code constant, not a device figure, and the per-GPU exceptions show the crossover is measured, not derived. That is the M8 lab's message.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvf.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu (lines 321 to 345)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvf.cu (line 869 on)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu (lines 1912 to 2110)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-ops.cpp (lines 2761 to 2875)

### D7. Launch overhead in practice: graphs in llama.cpp (CUDA and HIP) and PyTorch

**Verdict: CONFIRMED.**

- llama.cpp: `common.cuh` `#if (defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS)) || defined(GGML_MUSA_GRAPHS)` then `#define USE_CUDA_GRAPH`. `ggml/CMakeLists.txt`: `option(GGML_CUDA_GRAPHS "ggml: use CUDA graphs (llama.cpp only)" ${GGML_CUDA_GRAPHS_DEFAULT})` with the default OFF in ggml, and the top-level `CMakeLists.txt` sets `GGML_CUDA_GRAPHS_DEFAULT ON` for llama.cpp builds; `option(GGML_HIP_GRAPHS "ggml: use HIP graph" ON)`. `vendors/hip.h` maps the names: `#define cudaGraphLaunch hipGraphLaunch`, `#define cudaGraphInstantiate hipGraphInstantiate`, and so on. `ggml-cuda.cu` captures with `cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed)` and replays with `cudaGraphLaunch`, and falls back when a node would synchronise: "the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs".
- PyTorch docs 2.15, "CUDA Graphs": "Replaying a graph sacrifices the dynamic flexibility of typical eager execution in exchange for greatly reduced CPU overhead. A graph's arguments and kernels are fixed, so a graph replay skips all layers of argument setup and kernel dispatch, including Python, C++, and CUDA driver overheads." "Kernels in a replay also execute slightly faster on the GPU, but eliding CPU overhead is the main benefit."
- `torch.compile`: "”reduce-overhead” is a mode that reduces the overhead of python with CUDA graphs, useful for small batches." The link to batching is in PyTorch's own words: at small batch the step is short, so fixed per-launch costs are a larger share (Amdahl).
- vLLM's `--enforce-eager` note (D2) says the same from the server side.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh (lines 1384 to 1386)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt (lines 117 to 118, 210, 217)
- https://github.com/ggml-org/llama.cpp/blob/master/CMakeLists.txt (lines 171 to 172)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/hip.h (lines 128 to 141)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu (lines 3307 to 3343, 5313 to 5619)
- https://docs.pytorch.org/docs/2.15/notes/cuda.html (section "CUDA Graphs")
- https://docs.pytorch.org/docs/2.15/generated/torch.compile.html

### D8. llama.cpp and MLX on Metal: concurrent encoders, with barriers placed by a "who writes, who reads" rule

**Verdict: CONFIRMED.** Both Metal backends open concurrent compute encoders and insert `memoryBarrierWithScope:MTLBarrierScopeBuffers` only where a dispatch depends on an earlier one. The rule they apply is the chapter's decomposition question (who writes, who conflicts) at the level of whole kernels, which makes it a good real-code example for the section 22.2 explorer and for section 22.3's task parallelism.

- llama.cpp `ggml-metal-device.m`: `res->obj = [cmd_buf computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent];` when concurrency is on, and `ggml_metal_encoder_memory_barrier` calls `[encoder->obj memoryBarrierWithScope:MTLBarrierScopeBuffers];`. `ggml-metal-context.m`: `res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;` (on unless that variable is set).
- llama.cpp `ggml-metal-ops.cpp`: "check if the current node can run concurrently with other nodes before it // the condition is that: //  - the current node cannot write to any previous src or dst ranges //  - the current node cannot read from any previous dst ranges // // if the condition is not satisfied, we put a memory barrier and clear all ranges // otherwise, we add the new ranges to the encoding context and process the node concurrently".
- MLX `mlx/backend/metal/device.cpp` (2026-11-01): `buffer_->computeCommandEncoder(MTL::DispatchTypeConcurrent)`; `register_output_array` sets `needs_barrier_` when an output buffer was an input of the previous group, and `maybeInsertBarrier()` then calls `memoryBarrier(MTL::BarrierScopeBuffers)`. `device.h` has a `ConcurrentContext` for regions known to be independent.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m (lines 836 to 842, 894 to 896)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-context.m (line 142)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-ops.cpp (lines 159 to 250)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device.cpp (lines 405 to 445, 670 to 680)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device.h (lines 32 to 40, 92 to 93)

---

## Group E: arithmetic for the lessons and the M8 lab (ours)

No vendor fact here beyond the quotes cited. These numbers are for authors and browser checks; lessons show formulas and the reader's own measurements, not these values as device facts.

### E1. Intensity of one decode-step GEMM as a function of batch size B

**Verdict: CONFIRMED** (arithmetic; reproduces NVIDIA's and chapter 20's numbers).

Project convention is row-major FP32 (`CLAUDE.md`, milestones). For `Y[B x N] = X[B x K] W^T` with W of N x K floats, counting each matrix read or written once from DRAM:

- FLOPs = 2·B·K·N; bytes = 4·(K·N + B·K + B·N); intensity I(B) = B·K·N / (2·(K·N + B·K + B·N)) FLOP/B.
- For B much smaller than K and N, I(B) is about B/2 FLOP/B in FP32 (about B in FP16, NVIDIA's form in A4). It levels off at K·N / (2·(K + N)) as B grows.
- Check against A4's table (1024 inputs, 4096 outputs): FP16 gives 0.9988 at B = 1 and 315.08 at B = 512, matching "1" and "315 FLOPS/B". In FP32 the same layer gives 0.4994 and 157.54.
- Crossing the device threshold T (FLOP/B, chapter 5's term, measured by lab 5.1): solve I(B) = T, giving B* = 2T / (1 - 2T(K + N)/(K·N)) in FP32, valid while the denominator is positive. B* is a little above 2T for model-sized K and N. With the book's example threshold (H100, 20.0 FLOP/B, `notes/ch05.md`, p. 94) and K = N = 4096, B* = 40.8. The reader's B* comes from their own T.
- This is the ideal count. The project's own kernels re-read tiles from DRAM unless caches catch them, so their effective intensity is lower and the measured knee later; NVIDIA calls the comparison "a simplified rule of thumb" (A4). The lab asks for the prediction and then reports what was measured.

### E2. The other pieces of the decode step do not get more intense with B

**Verdict: CONFIRMED** (arithmetic; NVIDIA's Table 1 lists ReLU and layer normalisation as memory-limited, A4).

- Bias add over a B x N output, FP32: B·N FLOPs; bytes 4·(2·B·N + N) (read Y, write Y, read the bias once). I = B / (4·(2B + 1)): 0.083 at B = 1, approaching 0.125. Residual add (read two, write one): 1/12 FLOP/B at every B. Scale: 1/8.
- Row softmax: a handful of operations per element for each element read and written; whatever the exact count, it is a constant per element, independent of B.
- Consequence for the M8 prediction (ours): as B grows, the GEMM's time stays nearly flat until about B*, while the elementwise and softmax time grows in proportion to B. Tokens per second therefore rises roughly in proportion to B at first, then flattens once the GEMM crosses the threshold or once the B-proportional parts dominate. The whole step never becomes compute-bound in the elementwise parts.

### E3. Tile quantization in the project's own GEMMs can mimic a bandwidth plateau

**Verdict: CONFIRMED** (NVIDIA documents the effect; tile sizes read from the repository today).

- NVIDIA matrix multiplication guide, section 3.1: "Tile quantization occurs when matrix dimensions are not divisible by the thread block tile size." "While libraries ensure that invalid memory accesses are not performed by any of the tiles, all tiles will perform the same amount of math." Section 3.2: "there is a second quantization effect where the total number of tiles is quantized to the number of multiprocessors on the GPU: Wave quantization."
- Project tiles: M2 (`labs/ch05/lab3_gemm/solution/kernel.cu`) `#define TILE_WIDTH 16`; M3 (`labs/ch15/lab6_linear/solution/kernel.cu`) `#define bM 128 // block-level output tile: bM x bN (rows of Y = batch, columns = out)`, `bN 128`.
- Consequence for M8 (ours): with M3, every B from 1 to 128 launches the same grid and does the same arithmetic on zero-filled rows, so time per step is flat over that range whether or not the kernel is bandwidth-bound. With M2 the same happens in steps of 16. The lab should either report this or let the reader discover it, and the prediction question should ask about it. At small output widths the grid may also be smaller than the device can hold (wave quantization), which no batch size fixes.

- https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html (sections 3.1, 3.2)

### E4. The section's two throughput formulas and the notes' predict-then-reveal numbers

**Verdict: CONFIRMED** (arithmetic).

- Book (p. 539): with batching, throughput = queries / (one GEMM's latency); without, queries / (sum of the GEMV latencies). For the decode step, tokens per second = B / (time per step); `llama-batched-bench` computes the same, `S_TG = B·TG / T_TG` (D3).
- Notes' example: a batch of 8 whose GEMM takes 3 times one GEMV: per-query latency x3, throughput x8/3 = 2.67 (numbers ours for the exercise).
- Amdahl example (p. 537): 16.807 and 20 (A9).

---

## Book items with no vendor fact to check

Recorded so the wrap page's errata list does not wait on this file (all from `notes/ch22.md`, "Unclear in the PDF"):

- "scan (Chapter 12)" on p. 534: scan is chapter 11.
- "warp-level shuffling register tiling" on p. 539: unclear list.
- 1/5.95 % printed as 17x on p. 537: 16.8 exactly (A9).
- p. 536: "the benefit over the gather pattern of the scatter pattern" is reversed in meaning.
- p. 537: "grid points" where the example is about atoms; "non-bonded" and "non-bonding" alternate.
- p. 539: "vector-matrix" in the prose, "matrix-vector" in the formula.
- p. 539 against p. 505: "attention heads" (A6 resolves it with Orca's two definitions).
- The overflow list (p. 531), cutoff binning, and the pattern tour (pp. 533 to 536) are book-internal; no current document bears on them.

## Answers to the notes' portability questions

- Gather versus scatter on HIP and Metal (`notes/ch22.md` marks it unverified): both backends have per-thread registers and per-block on-chip memory (ch05 B4, C1). Scatter costs differ in documented ways: on AMD, global atomics run in the L2 (ch09 B2) and float `atomicAdd` is compiled as a safe operation, possibly a CAS loop, unless built with `-munsafe-fp-atomics` (infra B1); on Metal, float atomics exist only for add and sub, there is no float atomic max at all, and Apple documents neither where atomics run nor their cost (infra C1, ch09 C4). No vendor compares gather and scatter on AMD or Apple hardware, so "gather is usually better" carries over as the same reasoning, ours on every backend (A1).
- Streams on HIP: same API (infra B8), same "may run concurrently" semantics, plus the 4-hardware-queue default (B1). On Metal: command queues and buffers (infra C12) and, for overlap inside one queue, a concurrent compute encoder with explicit barriers (C1).
- The book's portability claim (p. 540, techniques "build a foundation for parallel thinking in general") is an opinion; nothing to check.

---

## Summary of corrections

Every PARTLY item, and every sub-claim that could not be verified, one line each:

- A1: the gather-over-scatter argument rests on documented mechanisms (registers cost "zero extra clock cycles"; same-address atomics serialise in the L2), but no NVIDIA document quantifies "significantly slower"; do not attach a ratio.
- A5 (sub-claim not found): no vendor source states that batching lengthens each request's own projection latency; it is the book's reasoning and M8's measurement.
- A6: batching raises the intensity of the QKV and output projections, not of attention proper; Orca's footnote names the two definitions of "attention", which reconciles p. 539 with chapter 20's p. 505.
- C2: indirect command buffers are Metal's documented way to reuse encoded commands and cut CPU overhead, but the comparison with CUDA graphs is ours; an ICB has no dependency edges, and Apple gives no saving figure. Not verified whether llama.cpp or MLX use ICBs anywhere outside the files searched.

Status and path facts worth knowing even where the verdict is CONFIRMED:

- A3: the CUDA guide's cross-references to its graph sections are misnumbered ("2.4.9.2", "4.1"); the pages are 2.5.9.2 and 4.2.
- A4: NVIDIA's deep-learning performance guides that carry the batching argument (and chapter 20's 1 and 315 FLOPS/B example) date from 2023 and argue with V100 and A100 figures.
- B1: HIP creates at most `GPU_MAX_HW_QUEUES` (default 4) hardware queues per device per process; more streams share them round-robin, which limits task-level overlap on AMD.
- B2: AMD calls direct-launch overhead "negligible" and framework overhead significant; NVIDIA calls per-kernel launch overhead potentially "a significant fraction" for short kernels.
- D1: "continuous batching", "in-flight batching" and "iteration-level batching" are one technique (TensorRT-LLM docs); chapter 20 uses the second name.
- D5: Hugging Face TGI is in maintenance mode and its repository is archived.
- D6: llama.cpp switches from matrix-vector to matrix-matrix kernels at a batch of about 8 tokens, with per-GPU tuned exceptions on CUDA and a 2 to 8 window on Metal.
- D7: llama.cpp builds use CUDA graphs by default (standalone ggml does not) and HIP graphs by default.

## Where the backends really differ

One line per cell. Nothing here was run on hardware by us.

| Feature | CUDA | HIP (ROCm 10.2) | Metal |
|---|---|---|---|
| Gather: accumulate in a register, share inputs on chip | Registers; `__shared__` per block | Same spelling | Thread registers; `threadgroup` memory (32 KB) |
| Scatter with float add | `atomicAdd(float*, float)` | Same call; compiled safe (possibly a CAS loop) unless `-munsafe-fp-atomics` | `atomic_fetch_add_explicit` on `atomic_float`, device memory (threadgroup from Metal 4.1) |
| Scatter with float max | No float overload; CAS loop | `atomicMax(float*, float)` exists (not portable to CUDA) | None documented |
| Several independent kernels at once | Non-default streams; `concurrentKernels`; up to 128 resident grids per device | Streams; same property; at most `GPU_MAX_HW_QUEUES` (default 4) hardware queues, more streams share them round-robin | One compute encoder of type `MTLDispatchTypeConcurrent` plus `memoryBarrier(scope:)`; overlap across queues not documented |
| What blocks concurrency by default | Legacy default stream; use `cudaStreamNonBlocking` | "The default stream is a blocking operation"; `hipStreamNonBlocking` | Plain encoder is serial; MTL4 resources are untracked, so the author must barrier |
| Launch-overhead remedy | CUDA graphs (capture, instantiate, `cudaGraphLaunch`) | HIP graphs, same shape with `hip` prefix | Indirect command buffers (reusable commands, no dependency edges; analogy ours) |
| Many independent GEMMs in one call | `cublasGemmBatchedEx`, `cublasGemmStridedBatchedEx` | `hipblasGemmBatchedEx`, `hipblasGemmStridedBatchedEx` | `MPSMatrixMultiplication` with `batchStart`, `batchSize` over a multi-matrix `MPSMatrix` |
| Timing one decode step | Two events, `cudaEventElapsedTime` (ms) | `hipEventElapsedTime` (ms) | `GPUEndTime - GPUStartTime` of the command buffer holding the step (s) |
| The device threshold (FLOP/B) for the M8 prediction | Vendor peaks exist (the book's H100 example only); the reader measures with lab 5.1 | AMD publishes peaks (ch01 B3); the reader measures | Apple publishes no peak FLOPS or bandwidth (infra C13); lab 5.1 is the only source |
| llama.cpp's small-batch matmul path | `mmvf` / `mmvq` up to 8 tokens, tuned per GPU | The same sources, with AMD-specific branches | `mul_mv_ext` for 2 to 8 tokens, then `mul_mm` |
| A continuous-batching server to point at | vLLM; `llama-server` | vLLM (documents ROCm); `llama-server` | `llama-server`; `mlx-lm` server (`BatchGenerator`) |
