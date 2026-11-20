# Chapter 23 fact check: streams, events, pinned memory, overlap; MPI, NCCL, NVSHMEM (CUDA, HIP, Metal, real backends)

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md` and `notes/today/infra-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY, and judges the assumption stated in the item title (usually the book's sentence, with its page from `notes/ch23.md`; otherwise the assumption a lesson author would bring). **Quote** is copied from the source (HTML flattened to text, GitHub raw files at a tag or on the default branch, Apple's documentation JSON, local SDK headers, PDF text) with line breaks removed.
- Chapter 23 is Condensed. The lessons keep section 23.3 (streams, asynchronous copies, pinned memory, events, overlap), which applies on one GPU, and give MPI, NCCL and NVSHMEM a paragraph each. Group A therefore checks the CUDA facts of section 23.3 in depth and the MPI, NCCL and NVSHMEM statements only as far as a paragraph and the wrap page's errata need. Groups B and C give the HIP and Metal counterparts. Group D is the bridge and goal 3: how llama.cpp, PyTorch, MLX and vLLM use the same mechanisms. Group E is the stepper's arithmetic, ours.
- Versions the sources reported on the check date: CUDA Runtime API reference v13.5 ("Last updated on Nov 02, 2026"), CUDA Programming Guide v13.5 ("Last updated on Oct 31, 2026"), CUDA C++ Best Practices Guide ("Last updated on Nov 02, 2026"); ROCm Core SDK 10.2.0 = tag `therock-10.2` of `ROCm/rocm-systems` (HIP docs 7.17.0, rocSHMEM docs 3.8.0, RCCL docs 2.32.3); NCCL docs "NCCL 2.32.3 documentation", latest GitHub release v2.32.6-1 (2026-11-09); NVSHMEM docs at `docs.nvidia.com/nvshmem/api/latest/`, latest GitHub release "NVSHMEM 3.8.0-0" (2026-09-22); MPI Forum: MPI 5.0 ("approved by the MPI Forum on June 5, 2025"), no later standard listed; Open MPI docs default to v5.0.x; MPICH latest stable release v5.0.2 (2026-09-28); Apple documentation JSON and the macOS 27.0 SDK headers on this machine (macOS 27.0.1); llama.cpp `master` at `184862801` (2026-11-20); MLX `main` (`mlx/backend/metal/device.cpp` last changed 2026-11-01); mlx-lm `main` (last push 2026-11-19); PyTorch docs 2.15 and `main` on 2026-11-20; vLLM docs "latest".
- Not checked: anything by running code on a GPU. One small simulation of our own (Group E) checks arithmetic only. Nothing was run on CUDA, AMD or Apple hardware for this file.
- **Device figures in this file are evidence for a verdict, not lesson content.** The only numbers below are API constants and defaults (default priority 0, 8 work queues, 64 in-flight command buffers, ROCm's constant 8 for `asyncEngineCount`), not device measurements. The project rule stands: reader figures come from `python3 labs/pmpp.py doctor` and lab benchmarks.

Totals: 28 CONFIRMED, 2 WRONG, 8 PARTLY, 1 COULD NOT VERIFY (39 items: A1 to A16, B1 to B7, C1 to C6, D1 to D7, E1 to E3). The WRONG and PARTLY items are collected under "Summary of corrections".

---

## Already settled elsewhere (cite, do not re-check)

| Topic chapter 23 touches | Where it is settled |
|---|---|
| Kernel launches are asynchronous; the default stream is in order; `cudaMemcpy` blocks; event timing pattern | ch02 A5 |
| Signatures of `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamSynchronize`, `cudaMemcpyAsync`, `cudaMallocHost`, `cudaHostAlloc` and its flags, `cudaEventCreateWithFlags` and its flag values (`cudaEventDisableTiming` = 0x02), `cudaEventRecord`, `cudaEventElapsedTime`; "page-locking excessive amounts ... may degrade system performance" | infra A8 |
| HIP names for all of the above (`cudaMallocHost` is `hipHostMalloc`, `cudaFreeHost` is `hipHostFree`, `hipMallocHost` deprecated); `hipEventElapsedTime` resolution | infra B8 |
| Concurrent kernels need non-default streams, no event dependency and free resources; legacy default stream blocks; `cudaStreamNonBlocking`; `concurrentKernels`; 128 resident grids per device | ch22 A2 |
| HIP concurrent kernels; `GPU_MAX_HW_QUEUES` default 4, extra streams share hardware queues round-robin | ch22 B1 |
| CUDA graphs, HIP graphs, Metal indirect command buffers as launch-overhead remedies | ch22 A3, B2, C2 |
| Metal command queue, command buffer, `commit`, `waitUntilCompleted`, `addCompletedHandler`, blit copy, `MTLEvent`, `MTLSharedEvent` as the course's analogue of streams | infra C12 (refined by C1 below) |
| Serial vs concurrent compute encoders; hazard tracking; Metal 4 resources untracked | ch02 C8, C9; ch22 C1 |
| Apple silicon: one system memory; shared buffers need no copy, only ordering; no pinned/pageable distinction; private buffers filled by blit | ch05 C6, appC C2, ch02 C3, infra C12 |
| Metal GPU time per command buffer (`GPUStartTime`, `GPUEndTime`) | ch05 C9 |
| Zero-copy (mapped) pinned memory, `cudaHostRegister`, HIP coherence flags for `hipHostMalloc` | appC A3, B2 |
| Peer GPU access on CUDA, HIP; none on Apple silicon | appC A8, B4, C4 |
| PyTorch `pin_memory`, `non_blocking=True`, `cudaHostAlloc` in the caching host allocator; llama.cpp pinned host buffers (`cudaMallocHost`, `GGML_CUDA_NO_PINNED`) | appC D1, D3 |
| NCCL: what it is, "topology-aware", interconnects "PCIe, NVLINK, InfiniBand Verbs, and IP sockets", used for training, repository since 2015 | ch24 A16 |
| RCCL is AMD's NCCL with the same function names; xGMI and Infinity Fabric links | ch24 B2, B3 |
| No NVLink and no collective library in Metal; MLX distributed backends (MPI, ring, JACCL, NCCL) | ch24 C3, C4 |

---

## Group A: CUDA, and the book's dated statements (CUDA, MPI, NCCL, NVSHMEM)

### A1. Stream priorities: `cudaDeviceGetStreamPriorityRange(&low, &high)`, `cudaStreamCreateWithPriority(&s, cudaStreamDefault, prio)`; "only relative priority values matter"; streams from `cudaStreamCreate` share one priority (p. 558)

**Verdict: CONFIRMED.** This also settles the notes' "Unclear in the PDF" item on direction: lower numbers mean higher priority. The book's `low, high` order matches the parameter order (least priority first, greatest second), so its snippet is correct as printed.

- Runtime API, `cudaDeviceGetStreamPriorityRange ( int * leastPriority , int * greatestPriority )`: "Returns in *leastPriority and *greatestPriority the numerical values that correspond to the least and greatest stream priorities respectively. Stream priorities follow a convention where lower numbers imply greater priorities. The range of meaningful stream priorities is given by [ *greatestPriority , *leastPriority ]." Values outside are "automatically clamped". "This function will return ‘0’ in both *leastPriority and *greatestPriority if the current context’s device does not support stream priorities".
- `cudaStreamCreateWithPriority ( cudaStream_t * pStream , unsigned int flags , int priority )`: "priority follows a convention where lower numbers represent higher priorities. ‘0’ represents default priority." Note: "Stream priorities are supported only on GPUs with compute capability 3.5 or higher."
- Guide 2.5.9.1: "The general scheme is that lower numbers correspond to higher priorities." "The default priority of a stream is 0." Its example names the outputs `minPriority, maxPriority` and passes `minPriority` with the comment "// Lowest priority".
- Device property: `int streamPrioritiesSupported` "Device supports stream priorities."
- NVIDIA's own version of the chapter's code (the book's reference [4], `mpi_overlap/jacobi.cpp` on `master`) does exactly what the book does: `cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority)`, then the compute stream with `leastPriority` and the two halo-push streams with `greatestPriority`.

Lesson wording: say "lower number, higher priority" once and never print a range; the range is device-specific.

- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html (section 2.5.9.1)
- https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
- https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/mpi_overlap/jacobi.cpp (lines 241 to 252)

### A2. Priorities fix the ordering hazard: if the internal kernel "runs first and occupies all GPU resources", give the boundary streams higher priority (p. 558)

**Verdict: PARTLY.** A priority is a scheduling hint for pending work. NVIDIA says it does not preempt work already running and guarantees no order. So it helps exactly when both kernels are waiting to be scheduled, and does nothing once the internal kernel's blocks already fill the GPU; the boundary kernels then wait for blocks to retire. It also has no effect on the copies.

- `cudaStreamCreateWithPriority`: "Priorities provide a hint to preferentially run work with higher priority when possible, but do not preempt already-running work or provide any other functional guarantee on execution order."
- Same page, note: "In the current implementation, only compute kernels launched in priority streams are affected by the stream’s priority. Stream priorities have no effect on host-to-device and device-to-host memory operations."
- Guide 2.5.9.1: "a priority of a stream is only a hint to the runtime and generally applies primarily to kernel launches, and may not be respected for memory transfers. Stream priorities will not preempt already executing work, or guarantee any specific execution order."
- Guide 2.5.2: "Streams may be assigned a priority which acts as a hint to the runtime to influence the scheduling, but does not guarantee a specific order of execution."

What the lesson should say: the book's two remedies work together. Issue the boundary kernels first (the guide's own advice in 2.5.8: "All independent operations should be issued before dependent operations") and give their streams higher priority; neither guarantees the order, and a measurement is the only proof. The same caveat applies to the bridge the notes draw (a small sampling kernel behind a long GEMM): priority does not interrupt a GEMM already resident.

- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html (sections 2.5.2, 2.5.8, 2.5.9.1)

### A3. "`cudaMemset` is blocking"; `cudaMemsetAsync` is the non-blocking form (p. 559)

**Verdict: WRONG** as a statement about the host. NVIDIA documents `cudaMemset` on device memory as asynchronous with respect to the host. The book's conclusion (use `cudaMemsetAsync` in `internalStream`) is still right, for a different reason: a plain `cudaMemset` goes into the legacy default stream, which orders itself against every blocking stream on the device, so it would serialise with the top and bottom streams.

- Runtime API, "API synchronization behavior", Memset: "The cudaMemset functions are asynchronous with respect to the host except when the target memory is pinned host memory. The Async versions are always asynchronous with respect to the host."
- `cudaMemset ( void * devPtr , int value , size_t count )`: "Note that this function is asynchronous with respect to the host unless devPtr refers to pinned host memory."
- Same page, opening: "The synchronous forms of these APIs issue these copies through the default stream." And "Any CUDA API call may block or synchronize for various reasons such as contention for or unavailability of internal resources. Such behavior is subject to change and undocumented behavior should not be relied upon."
- "Stream synchronization behavior": "The legacy default stream is an implicit stream which synchronizes with all other streams in the same CUcontext except for non-blocking streams ... When an action is taken in the legacy stream such as a kernel launch or cudaStreamWaitEvent() , the legacy stream first waits on all blocking streams, the action is queued in the legacy stream, and then all blocking streams wait on the legacy stream."

So the accurate sentence is: "`cudaMemset` does not make the host wait, but it is queued in the default stream, which waits for all other (blocking) streams and makes them wait for it; `cudaMemsetAsync` in a named stream orders the reset with that stream only."

- https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html

### A4. `cudaMemsetAsync(ptr, 0, n, stream)` and `cudaMemcpyAsync(dst, src, n, kind, stream)` are non-blocking and ordered in their stream (pp. 559 to 560)

**Verdict: CONFIRMED.**

- `cudaMemsetAsync ( void * devPtr , int value , size_t count , cudaStream_t stream = 0 )`: "cudaMemsetAsync() is asynchronous with respect to the host, so the call may return before the memset is complete. The operation can optionally be associated to a stream by passing a non-zero stream argument. If stream is non-zero, the operation may overlap with operations in other streams."
- `cudaMemcpyAsync` signature and overlap quote: infra A8.
- Guide 2.5.2.3: "Like other asynchronous function calls, this function call returns immediately, whereas the cudaMemcpy() function blocks until the memory transfer is complete. In order to access the results of the transfer safely, the application must determine that the operation has completed using some form of synchronization."
- The fourth launch parameter (p. 556) as the guide prints it: `kernel <<< grid , block , shared_mem_size , stream >>> (...);` "The kernel launch is asynchronous and the function call returns immediately."

- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html (sections 2.5.2.2, 2.5.2.3)

### A5. Events: `cudaEventCreateWithFlags(&e, cudaEventDisableTiming)` makes synchronisation cheaper; `cudaEventRecord(e, stream)` marks a point; `cudaStreamWaitEvent(stream, e, 0)` inserts a wait, 0 meaning default (p. 560)

**Verdict: CONFIRMED.** Two semantic details the book does not state, both needed for the lesson's dependency puzzle and for the Metal contrast (C2):

- `cudaEventRecord ( cudaEvent_t event , cudaStream_t stream = 0 )`: "Captures in event the contents of stream at the time of this call." "cudaEventRecord() can be called multiple times on the same event and will overwrite the previously captured state. Other APIs such as cudaStreamWaitEvent() use the most recent captured state at the time of the API call, and are not affected by later calls to cudaEventRecord() . Before the first call to cudaEventRecord() , an event represents an empty set of work".
- `cudaStreamWaitEvent ( cudaStream_t stream , cudaEvent_t event , unsigned int flags = 0 )`: "Makes all future work submitted to stream wait for all work captured in event ." "The synchronization will be performed efficiently on the device when applicable." Flags: "cudaEventWaitDefault : Default event creation flag." and "cudaEventWaitExternal : Event is captured in the graph as an external event node when performing stream capture."
- `cudaEventCreateWithFlags`: "cudaEventDisableTiming : Specifies that the created event does not need to record timing data. Events created with this flag specified and the cudaEventBlockingSync flag not specified will provide the best performance when used with cudaStreamWaitEvent() and cudaEventQuery() ."
- Guide 2.5.7: "cudaStreamWaitEvent() takes a stream and an event as parameters ... and makes all the commands added to the given stream after the call to cudaStreamWaitEvent() delay their execution until the given event has completed."

The book calls `cudaEventDisableTiming` "a compile-time constant"; it is a flag value (0x02, infra A8). Harmless.

So a CUDA event is a re-recordable snapshot of one stream's queue, with no counter. A wait captures whatever the event held when the wait was issued; re-recording the event in the next iteration (as Fig. 23.14 does) is safe because waits already issued are unaffected.

- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html (section 2.5.7)

### A6. Pinned memory: DMA works on physical addresses, so `cudaMemcpy` from pageable memory stages through a pinned buffer and is synchronous; "`cudaMemcpyAsync()` requires the host buffer to be already allocated as pinned memory" (pp. 562 to 563)

**Verdict: PARTLY.** The mechanism is confirmed by NVIDIA's own wording (a staging buffer "for DMA transfer"). The requirement is stated too strongly: the runtime reference and the guide say an async copy with pageable memory still works but may become synchronous and then overlaps nothing. NVIDIA's Best Practices Guide does use the book's word "requires"; the reference is the precise statement.

- "API synchronization behavior", synchronous memcpy: "For transfers from pageable host memory to device memory, a stream sync is performed before the copy is initiated. The function will return once the pageable buffer has been copied to the staging memory for DMA transfer to device memory, but the DMA to final destination may not have completed." "For transfers from device to either pageable or pinned host memory, the function returns only once the copy has completed."
- Same page, asynchronous memcpy: "For transfers between device memory and pageable host memory, the function might be synchronous with respect to host." "If pageable memory must first be staged to pinned memory, the driver may synchronize with the stream and stage the copy into pinned memory. For all other transfers, the function should be fully asynchronous."
- Guide 2.5.2.3, note: "cudaMemcpyAsync() will function correctly if host memory which is not pinned and page-locked is used, but it will revert to a synchronous behavior which will not overlap with other work." (also quoted in infra A8)
- Guide 2.6.3: "Host allocations made through traditional allocation mechanisms like malloc , new , or mmap are not page-locked, which means they may be swapped to disk or physically relocated by the operating system. Page-locked host memory is required for asynchronous copies between the CPU and GPU . Page-locked host memory also improves performance of synchronous copies."
- Best Practices 10.1.2: "In contrast with cudaMemcpy() , the asynchronous transfer version requires pinned host memory".
- `cudaMallocHost`: "Since the memory can be accessed directly by the device, it can be read or written with much higher bandwidth than pageable memory obtained with functions such as ::malloc()." And, new since the book: "On systems where ::pageableMemoryAccessUsesHostPageTables is true, cudaMallocHost may not page-lock the allocated memory." (Those are HMM/ATS systems, appC A12; PyTorch's note on hardware-coherent systems says the same, appC D1.)

Lesson wording: "an async copy from pageable memory is legal but silently stops being asynchronous", which is also the failure the lab's planted defect (pageable buffer) should show as a measurement, not as a crash.

- https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html (section 2.5.2.3)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html (section 2.6.3)
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 10.1.2)
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html

### A7. Not in the book, needed by the stepper: copy engines, and when copies overlap kernels and each other

**Verdict: CONFIRMED.** NVIDIA documents the chunked copy-compute pipeline (the stepper's subject), the copy-engine count that decides whether host-to-device and device-to-host copies can overlap each other, and a rough time estimate the stepper's model must reproduce (E1 does).

Best Practices Guide 10.1.2, "Asynchronous and Overlapping Transfers with Computation":

- "On devices that are capable of concurrent copy and compute, it is possible to overlap kernel execution on the device with data transfers between the host and the device. Whether a device has this capability is indicated by the asyncEngineCount field of the cudaDeviceProp structure ... On devices that have this capability, the overlap once again requires pinned host memory, and, in addition, the data transfer and kernel must use different, non-default streams (streams with non-zero stream IDs)."
- The staged loop as printed: `size = N * sizeof ( float ) / nStreams ; for ( i = 0 ; i < nStreams ; i ++ ) { offset = i * N / nStreams ; cudaMemcpyAsync ( a_d + offset , a_h + offset , size , dir , stream [ i ]); kernel <<< N / ( nThreads * nStreams ), nThreads , 0 , stream [ i ] >>> ( a_d + offset ); }`
- "GPUs with a single copy engine can perform one asynchronous data transfer and execute kernels whereas GPUs with two copy engines can simultaneously perform one asynchronous data transfer from the host to the device, one asynchronous data transfer from the device to the host, and execute kernels. The number of copy engines on a GPU is given by the asyncEngineCount field of the cudaDeviceProp structure".
- "it is not possible to overlap a blocking transfer with an asynchronous transfer, because the blocking transfer occurs in the default stream, so it will not begin until all previous CUDA calls complete. It will not allow any other CUDA call to begin until it has completed."
- The estimate: "when the execution time ( tE ) exceeds the transfer time ( tT ), a rough estimate for the overall time is tE + tT/nStreams for the staged version versus tE + tT for the sequential version. If the transfer time exceeds the execution time, a rough estimate for the overall time is tT + tE/nStreams ."
- Runtime API, `cudaDeviceProp`: `int asyncEngineCount` "Number of asynchronous engines." (The older `deviceOverlap` field no longer appears on the v13.5 struct page.)

What the guide does not give: a per-chunk overhead, and any statement of how one copy engine orders a device-to-host copy against the next chunk's host-to-device copy. E2 and E3 show why the stepper must state its assumptions about both.

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 10.1.2)
- https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html

### A8. Not in the book, a stepper caveat: streams map onto a limited number of hardware work queues

**Verdict: CONFIRMED.** More streams than work queues can create false dependencies between unrelated streams. HIP has the same limit with a smaller default (ch22 B1). This is the documented reason "more streams" stops helping, besides per-chunk overhead.

- Guide, environment variables, `CUDA_DEVICE_MAX_CONNECTIONS`: "The environment variable controls the number of concurrent compute and copy engine connections (work queues), setting both to the specified value. If independent GPU tasks, namely kernels or copy operations launched from different CUDA streams, map to the same work queue, a false dependency is created which can lead to GPU work serialization, since the same underlying resource(s) are used. To reduce the probability of such false dependencies, it is recommended that the work queue count, controlled via this environment variable, be greater than or equal to the number of active CUDA streams per context." "Possible Values : 1 to 32 connections, default is 8 (assumes no MPS)".
- `CUDA_DEVICE_MAX_COPY_CONNECTIONS`: "controls the number of concurrent copy connections (work queues) involved in copy operations. It affects only devices of compute capability 8.0 and above." Default 8.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html (sections 5.2.3.2, 5.2.3.3)

### A9. The default stream the book mentions (p. 556), and the `cudaStreamDefault` flag in its priority snippet (p. 558)

**Verdict: CONFIRMED.** Two default-stream behaviours exist today; the book's code (streams created with `cudaStreamDefault`) gets blocking streams, which is fine as long as nothing is queued in the legacy default stream mid-loop (A3).

- Guide 2.5.9.1 code comment: "cudaStreamDefault indicates the stream should be created with default flags // in other words they will be blocking streams with respect to the legacy default stream // One could also use the option `cudaStreamNonBlocking` here to create a non-blocking streams".
- Guide 2.5.6.2: "Starting in CUDA-7, CUDA allows for each host thread to have its own independent default stream, rather than the shared legacy default stream. In order to enable this behavior one must either use the nvcc compiler option --default-stream per-thread or define the CUDA_API_PER_THREAD_DEFAULT_STREAM preprocessor macro."
- "Stream synchronization behavior": "The per-thread default stream is not a non-blocking stream and will synchronize with the legacy default stream if both are used in a program. The per-thread default stream can be used explicitly with the CUstream ( cudaStream_t ) handle CU_STREAM_PER_THREAD ( cudaStreamPerThread )."

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html (sections 2.5.6, 2.5.9.1)
- https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html

### A10. MPI as the book uses it: `mpiexec` is the standard's launcher, `mpirun` a common synonym; `MPI_Sendrecv`, `MPI_Allreduce` (pp. 546 to 553)

**Verdict: CONFIRMED.** The current standard is MPI 5.0. The C bindings the book prints are unchanged; MPI 4.0 and later add large-count `_c` variants beside them.

- MPI Forum documents page: "MPI-5.0 was approved by the MPI Forum on June 5, 2025." The previous is MPI 4.1 (November 2, 2023).
- MPI 5.0 report, section 11.5: "In order that the “standard” command not be confused with existing practice, which is not standard and not portable among implementations, instead of mpirun MPI specifies mpiexec." "MPI specifies an mpiexec startup command and recommends but does not require it".
- MPI 5.0 C binding: `int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status)`, followed by `int MPI_Sendrecv_c(const void *sendbuf, MPI_Count sendcount, ...)`. `int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, ...` likewise.

For the Condensed lesson, MPI needs one paragraph: what it is for (processes with separate memories exchanging messages, launched as ranks by `mpiexec`), the book's pages (pp. 545 to 553), and the CUDA-aware caveat in A11. "MPI is the dominating programming interface for computing clusters today" (p. 545) is an opinion; nothing to check.

- https://www.mpi-forum.org/docs/
- https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report.pdf (section 11.5, "Portable MPI Process Startup"; section 3.2.7, "Blocking Send-Receive", for `MPI_Sendrecv`)

### A11. "Most MPI implementations (e.g., MPICH, OpenMPI, MVAPICH2) are designed to be CUDA-aware and are built with this feature enabled" (p. 552)

**Verdict: PARTLY.** All three projects support passing device pointers, but "built with this feature enabled" holds by default only for MPICH (and only if CUDA is found at configure time). Open MPI needs to be configured with CUDA (and UCX built with CUDA); the MVAPICH project has renamed its stacks, and GPU support lives in a separate series.

- MPICH `README.md` (`main`), "GPU Support": "MPICH automatically detects and enables GPU support for CUDA, Intel Level Zero (ZE), and HIP runtimes during configuration." "GPU support is automatically enabled if any supported GPU runtime is detected during the configure process." Disable with `--without-cuda`, or at run time with `MPIR_CVAR_ENABLE_GPU=0`.
- Open MPI v5.0.x docs, CUDA: "Open MPI offers two flavors of CUDA support: Via UCX . This is the preferred mechanism. Since UCX will be providing the CUDA support, it is important to ensure that UCX itself is built with CUDA support." "Regardless of which flavor of CUDA support (or both) you plan to use, Open MPI should be configured using the --with-cuda=<path-to-cuda> configure option to build CUDA support into Open MPI." Check a given build with `ompi_info --parsable --all | grep mpi_built_with_cuda_support:value`.
- MVAPICH home page: "MVAPICH-Plus Advanced MPI for GPU clusters ... This new series replaces the older MVAPICH2-GDR and MVAPICH2-X stacks." The current base release line is "MVAPICH 4.1". So the book's "MVAPICH2" names an older line.
- NCCL docs add a caution the book does not: "Using NCCL to perform inter-GPU communication concurrently with CUDA-aware MPI may create deadlocks. ... Using both MPI and NCCL to perform transfers between the same sets of CUDA devices concurrently is therefore not guaranteed to be safe."

Lesson wording: "CUDA-aware MPI is a property of how your MPI was built; check, do not assume."

- https://github.com/pmodels/mpich/blob/main/README.md (section 3)
- https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/cuda.html (sections 11.2.6.1, 11.2.6.2)
- https://mvapich.cse.ohio-state.edu/
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html

### A12. The NCCL calls in Figs. 23.16 to 23.20: `ncclGetUniqueId`, `ncclCommInitRank`, `ncclCommDestroy`, `ncclSend`, `ncclRecv`, `ncclGroupStart`, `ncclGroupEnd`, `ncclFloat`; NCCL calls take a stream and return once enqueued (pp. 564 to 568)

**Verdict: CONFIRMED.** Every signature is as the book prints it. Details worth one line each:

- `ncclSend ( const void * sendbuff , size_t count , ncclDataType_t datatype , int peer , ncclComm_t comm , cudaStream_t stream )`: "Rank peer needs to call ncclRecv with the same datatype and the same count as this rank. This operation is blocking for the GPU. If multiple ncclSend() and ncclRecv() operations need to progress concurrently to complete, they must be fused within a ncclGroupStart() / ncclGroupEnd() section." (`ncclRecv` mirrors it.) "Blocking for the GPU" means the stream does not move past the send until the peer's receive matches; the host is not blocked.
- `ncclGroupStart`: "All subsequent calls to NCCL until ncclGroupEnd will not block due to inter-CPU synchronization." `ncclGroupEnd`: "Returns when all operations since ncclGroupStart have been processed. This means the communication primitives have been enqueued to the provided streams, but are not necessarily complete." This is the book's p. 567 point.
- "CUDA Stream Semantics": "The NCCL call returns when the operation has been effectively enqueued to the given stream, or returns an error. The collective operation is then executed asynchronously on the CUDA device." And a trap for multi-stream groups: using several streams in one group "will enforce a stream dependency of all streams before the NCCL kernel starts and block all streams until the NCCL kernel completes."
- Overview, on why NCCL runs "on the GPU" (p. 564): "NCCL ... implements each collective in a single kernel handling both communication and computation operations." And "It is not a full-blown parallel programming framework; rather, it is a library focused on accelerating inter-GPU communication."
- `ncclCommInitRank ( ncclComm_t * comm , int nranks , ncclUniqueId commId , int rank )`: "Each rank is associated to a CUDA device, which has to be set before calling ncclCommInitRank." The book's Fig. 23.17 shows no `cudaSetDevice`; a lesson that shows the setup should say it is required.
- NVIDIA's own MPI bootstrap example is the book's Fig. 23.17 without the `MPI_Barrier`: `if ( myRank == 0 ) ncclGetUniqueId ( & id ); MPI_Bcast ( & id , sizeof ( id ), MPI_BYTE , 0 , MPI_COMM_WORLD ); ... ncclCommInitRank ( & comm , nRanks , id , myRank );`.
- "NCCL is not a complete replacement for MPI" (p. 564): the docs say "It is therefore easy to use MPI for CPU-to-CPU communication and NCCL for GPU-to-GPU communication." NCCL also has single-process initialisation without MPI (`ncclCommInitAll`).

- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html

### A13. "NCCL recently added support for symmetric memory and device-initiated communication, and may add support for one-sided communication soon" (p. 575)

**Verdict: PARTLY.** The first two halves are confirmed. The third is overtaken: NCCL already has host-initiated one-sided operations. Version by version, from NCCL's own docs:

- Symmetric memory: "Since 2.27, NCCL supports window registration, which allows users to register local buffers into NCCL window and enables extremely low latency and high bandwidth communication in NCCL." API `ncclCommWindowRegister`, flag `NCCL_WIN_COLL_SYMMETRIC`; allocator `ncclMemAlloc`.
- Device-initiated: "Starting with version 2.28, NCCL provides a device-side communication API, making it possible to use communication primitives directly from user CUDA kernels." Modules: LSA ("Load/Store Accessible ... using CUDA P2P"), Multimem, "GIN (GPU-Initiated Networking) – for communication over the network (since NCCL 2.28.7)", CFT ("since NCCL 2.31").
- One-sided: "(Since NCCL 2.29) One-sided communication enables a rank to write data to remote memory using ncclPutSignal() without requiring the target rank to issue a matching operation. The target memory must be pre-registered using ncclCommWindowRegister() . Point-to-point synchronization can be achieved by having the target rank call ncclWaitSignal() to wait for signals." Signature: `ncclPutSignal ( const void * localbuff , size_t count , ncclDataType_t datatype , int peer , ncclWindow_t peerWin , size_t peerWinOffset , int sigIdx , int ctx , unsigned int flags , ncclComm_t comm , cudaStream_t stream )`.
- Related, and directly about the chapter's theme: "Zero-CTA optimization to avoid the use of CTA for communication and to overlap communication and computation. Zero-CTA over NVLink with Copy Engine (CE) is supported since NCCL 2.28".

So for the wrap page: NCCL and NVSHMEM have converged further than the book says; NCCL today offers two-sided host-initiated, one-sided host-initiated, and device-initiated communication.

- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html (sections "Window Registration", "Zero-CTA Optimization")
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html (section "One-sided communication")
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html

### A14. The NVSHMEM calls in Figs. 23.22 to 23.25: `nvshmem_malloc`, `nvshmem_free`, `nvshmem_float_p`, `nvshmem_float_g`, `nvshmemx_init_attr_t` with `mpi_comm`, `nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr)`, `nvshmem_finalize`, `nvshmemx_barrier_all_on_stream`, `nvshmemx_float_put_block` (pp. 570 to 574)

**Verdict: CONFIRMED.** Every name exists with the book's meaning. Three details:

- `nvshmem_TYPENAME_p(TYPE *dest, TYPE value, int pe)` (`__device__` form listed): "These routines provide a very low latency put capability for single elements of most basic types. As with nvshmem_put , these routines start the remote transfer and may return before the data is delivered to the remote PE. Use nvshmem_quiet to force completion of all remote Put transfers." This is why Fig. 23.25 needs the barrier (p. 573).
- `nvshmemx_barrier_all_on_stream ( void , cudaStream_t stream )` exists; for `nvshmem_barrier_all`: "Before synchronizing with other PEs, nvshmem_barrier_all ensures completion of all previously issued memory stores and remote memory updates". Also: "Ordering APIs ... issued on the CPU and the GPU only order communication operations that were issued from the CPU and the GPU, respectively."
- `nvshmemx_init_attr ( unsigned int flags , nvshmemx_init_attr_t * attributes )`; field "void *mpi_comm A pointer to the MPI communicator handle that will be used as the NVSHMEM world team." (the book's `attr.mpi_comm = &mpi_comm` is right). "It is recommended, but not necessary to statically initialize the attribute struct using the NVSHMEMX_INIT_ATTR_INITIALIZER macro befor calling this function. The optional nature of the initializer may change in future major releases." The book declares `nvshmemx_init_attr_t attr;` without it; a lesson showing the snippet should add the initializer. "A call to nvshmem_finalize is required before the MPI communicator is destroyed", matching the book's order.
- `nvshmem_malloc`: "collective operations that require participation by all PEs." Block-scope put: `__device__ void nvshmemx_TYPENAME_put_block(TYPE *dest, const TYPE *source, size_t nelems, int pe)`.

The book's granularity advice (fine-grained puts fine over NVLink, coarse puts over networks, p. 574) is the book's; I found no NVSHMEM page that states it with numbers, and the lesson should present it as the book's.

- https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html
- https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html
- https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html
- https://docs.nvidia.com/nvshmem/api/latest/gen/api/memory.html

### A15. "NVSHMEM is NVIDIA's implementation of OpenSHMEM for clusters of GPUs" and "also supports host-initiated communication" (pp. 569, 575)

**Verdict: CONFIRMED.** And the library is now open source.

- Docs front page: "NVSHMEM implements the OpenSHMEM parallel programming model for clusters of NVIDIA ® GPUs. The NVSHMEM Partitioned Global Address Space (PGAS) spans the memory across GPUs and includes an API for fine-grained GPU-GPU data movement from within a CUDA kernel, on CUDA streams, and from the CPU."
- `nvshmem_init` is "as defined in the OpenSHMEM 1.3 specification".
- GitHub `NVIDIA/nvshmem`: created 2025-08-27, Apache-2.0, last push 2026-11-12, latest release "NVSHMEM 3.8.0-0" (2026-09-22). Description: "NVSHMEM can significantly reduce multi-process communication and coordination overheads by allowing programmers to perform one-sided communication from within CUDA kernels and on CUDA streams."

- https://docs.nvidia.com/nvshmem/api/latest/index.html
- https://github.com/NVIDIA/nvshmem

### A16. The chapter's references (p. 576) still resolve

**Verdict: PARTLY.** Two of the nine links have moved; one project has renamed itself.

| Ref. | As printed | Today |
|---|---|---|
| [2] NCCL overview | docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html | 200, "NCCL 2.32.3 documentation" |
| [3] NVSHMEM docs | docs.nvidia.com/nvshmem/api/index.html | a meta-refresh page that sends you to `https://docs.nvidia.com/nvshmem/api/latest/index.html` |
| [4] Multi GPU programming models | github.com/NVIDIA/multi-gpu-programming-models | live, last push 2026-11-06; directories include `mpi`, `mpi_overlap`, `nccl`, `nccl_overlap`, `nccl_graphs`, `nccl_device`, `nvshmem`, `single_gpu` (the chapter's variants, plus graph and device-API ones) |
| [5] CUB | nvidia.github.io/cccl/cub/index.html | 404; current docs at `https://nvidia.github.io/cccl/unstable/cub/index.html` (infra A2 found the same move) |
| [6], [7] MPICH, Open MPI | project sites | live |
| [8] MVAPICH | project site | live; the GPU stack is now "MVAPICH-Plus" (A11) |

- https://github.com/NVIDIA/multi-gpu-programming-models
- https://nvidia.github.io/cccl/unstable/cub/index.html

---

## Group B: HIP / ROCm

### B1. "HIP streams and events map one for one" onto CUDA's

**Verdict: PARTLY.** The names do (infra B8: every stream, event, async-copy and pinned-memory call in section 23.3 has a HIP name, two of them not a plain prefix swap). The behaviour does not, in three documented places a lesson must name: priorities are ignored by default (B2), `asyncEngineCount` is a runtime constant rather than a copy-engine count (B3), and at most 4 hardware queues are used per device by default, extra streams sharing them round-robin (ch22 B1; compare CUDA's 8 work queues, A8).

- See infra B8, ch22 B1, and B2 and B3 below.

### B2. Stream priorities on AMD (counterpart of A1 and A2)

**Verdict: WRONG** for the assumption "they work as on CUDA". The API exists with CUDA's names and convention, but ROCm 10.2 ignores priorities by default, and AMD's own sources disagree about it.

- Header (`hip_runtime_api.h` at `therock-10.2`), `hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority)`: "Priority of the stream. Lower numbers represent higher priorities. Stream priority is disabled at this time." and "@warning Stream priority is currently disabled to avoid queue-priority-related issues in KFD." Same warning on `hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority)`. On `develop`, `hipStreamGetPriority` reads "Stream priority is currently disabled by default".
- The switch, in the runtime (`projects/clr/rocclr/utils/flags.hpp` at the tag): `release(bool, DEBUG_HIP_IGNORE_STREAM_PRIORITY, true, "Treat all streams as normal priority")`. In `rocdevice.cpp`: "// If flag set, force all streams to normal priority" then `queue_priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;`.
- What the API reports anyway: the priority enum is `enum Priority : int { High = -1, Normal = 0, Low = 1 };` (`hip_internal.hpp`), so `hipDeviceGetStreamPriorityRange` returns -1 and 1 (`hip_stream.cpp`), and `hipGetDeviceProperties` sets `deviceProps.streamPrioritiesSupported = 1;` (`hip_device.cpp`). A program that checks the property and the range sees "supported" while every stream runs at normal priority.
- AMD's how-to page still says `hipStreamCreateWithPriority` "Allows creating a stream with a specified priority, enabling prioritization of certain tasks." That is the inconsistency.
- Minor: HIP's header gives the range as "[*leastPriority,*greatestPriority]" where CUDA writes "[ *greatestPriority , *leastPriority ]"; same meaning.

Not checked: what happens on AMD hardware if `DEBUG_HIP_IGNORE_STREAM_PRIORITY=0` is set; not run. A lesson should say "HIP accepts the call; ROCm 10.2 ignores the priority unless a debug variable is changed", and a lab must not ask a reader to measure a priority effect on AMD.

Not mapped in `pmpp/vendors/hip.h` (see "Notes for the lab runtime").

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 3215 to 3255)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/rocclr/utils/flags.hpp (lines 266 to 267)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/rocclr/device/rocm/rocdevice.cpp (lines 3476 to 3482)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_internal.hpp (line 373)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_stream.cpp (lines 408 to 431)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_device.cpp (lines 739, 743)
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html

### B3. Copy engines on AMD: `asyncEngineCount` (counterpart of A7)

**Verdict: PARTLY.** AMD documents the property and the overlap it enables in the same words as NVIDIA. But the ROCm runtime fills it with a constant, so it does not count hardware copy engines, and CUDA's "1 means copies in one direction at a time, 2 means both" reading does not carry over.

- HIP how-to: "Some AMD HIP-enabled devices can perform asynchronous memory copy operations to or from the GPU concurrently with kernel execution. Applications can query this capability by checking the asyncEngineCount device property. Devices with an asyncEngineCount greater than zero support concurrent data transfers. Additionally, if host memory is involved in the copy, it should be page-locked to ensure optimal performance."
- Same page: "Devices that support the concurrentKernels property can perform intra-device copies concurrently with kernel execution."
- Runtime: `deviceProps.asyncEngineCount = info.numAsyncQueues_;` (`hip_device.cpp`), `info_.numAsyncQueues_ = kMaxAsyncQueues;` (`rocdevice.cpp`, line 1696), `static constexpr uint kMaxAsyncQueues = 8;` (`rocdefs.hpp`, line 17). Header: `int deviceOverlap; ///< Deprecated. Use asyncEngineCount instead`.

So on AMD the stepper's "copy engines: 1 or 2" control is a CUDA concept; the honest Metal and HIP labels are "not documented as a hardware count".

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html (source `projects/hip/docs/how-to/hip_runtime_api/asynchronous.rst` at the tag, lines 117 to 150)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_device.cpp (line 743)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/rocclr/device/rocm/rocdefs.hpp (line 17)

### B4. `hipMemsetAsync`, `hipStreamWaitEvent` flags, `hipEventDisableTiming`, and `hipMemcpyAsync` with pageable memory (counterparts of A3 to A6)

**Verdict: CONFIRMED**, with one sub-claim COULD NOT VERIFY: AMD does not document whether plain `hipMemset` returns before the fill completes (CUDA does, A3).

- `hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream __dparm(0));` "hipMemsetAsync() is asynchronous with respect to the host, so the call may return before the memset is complete. ... If stream is non-zero, the operation may overlap with operations in other streams." (CUDA's wording.) `hipMemset`'s header comment says nothing about host synchronisation; AMD's how-to lists only `hipMemcpy` as a synchronous call.
- `hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags __dparm(0));` "All future work submitted to stream will wait until event reports completion before beginning execution." Flags `hipEventWaitDefault` (0x00), `hipEventWaitExternal` (0x01). Its comment also says "This function only waits for commands in the current stream to complete. Notably, this function does not implicitly wait for commands in the default stream to complete, even if the specified stream is created with hipStreamNonBlocking = 0." (ambiguous; recorded as written).
- `hipEventDisableTiming`: "Disable recording of timing information. Events created with this flag would not record profiling data and provide best performance if used for synchronization."
- `hipMemcpyAsync`, header note: "If host or dst are not pinned, the memory copy will be performed synchronously. For best performance, use hipHostMalloc to allocate host memory that is transferred asynchronously." (Same outcome as CUDA, A6.)
- AMD's host-memory page: "using pinned memory instead of pageable memory on the host can lead to a three times improvement in bandwidth." That is AMD's claim with no device named; not lesson content (project rule).

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 909 to 926, 3310 to 3339, 3725 to 3740, 5642 to 5643, 5669, 5734 to 5749)
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/host_memory.html

### B5. NCCL on AMD: RCCL keeps NCCL's names, including the new one-sided and window APIs (counterpart of A12, A13)

**Verdict: CONFIRMED.** Settled in ch24 B3 (same function names, "RCCL 2.32.3"); what this chapter adds:

- Header name: RCCL's CMake installs the API as `include/rccl/rccl.h` ("For external linking") and also generates `include/nccl.h` from the same `src/nccl.h.in`. llama.cpp's HIP build includes `<rccl/rccl.h>` (D2).
- The streams are HIP streams: `ncclResult_t ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t* signalDescs, ncclComm_t comm, hipStream_t stream);`
- The same header at the tag declares `ncclPutSignal`, `ncclWaitSignal` and `ncclCommWindowRegister`, so the A13 additions are present in RCCL too. Not checked: whether each is functional on every AMD GPU.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/rccl/src/nccl.h.in (lines 320, 353, 671, 1145, 1167, 1181, 1215, 1317)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/rccl/CMakeLists.txt (lines 793 to 794)

### B6. NVSHMEM on AMD: rocSHMEM (counterpart of A14, A15)

**Verdict: CONFIRMED.** AMD ships an OpenSHMEM-style library in the ROCm Core SDK. The shape matches NVSHMEM; the spellings do not.

- Docs ("rocSHMEM 3.8.0 documentation", ROCm Core SDK 10.2.0): "The ROCm OpenSHMEM (rocSHMEM) is an intra-kernel networking library that provides GPU-centric networking through an OpenSHMEM-like interface. It simplifies application code complexity and enables finer communication and computation overlap than traditional host-driven networking. rocSHMEM uses a single symmetric heap allocated to GPU memories." Source moved to `ROCm/rocm-systems/projects/rocshmem` (the old `ROCm/rocSHMEM` repository says "[DEPRECATED] Moved to ROCm/rocm-systems repo").
- Names at the tag: `__device__ ... void rocshmem_float_p(float *dest, float value, int pe);`, `__host__ void *rocshmem_malloc(size_t size);`, `__host__ int rocshmem_init_attr(unsigned int flags, rocshmem_init_attr_t *attr);` with `ROCSHMEM_INIT_WITH_MPI_COMM`, `__host__ void rocshmem_barrier_all_on_stream(hipStream_t stream);`. Cooperative variants are `_wg` (workgroup) and `_wave` where NVSHMEM has `_block` and `_warp`, and there is no `nvshmemx_`-style split of extensions.
- PyTorch treats the two as interchangeable backends (D5).

- https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/
- https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/api/rma.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/rocshmem/include/rocshmem/rocshmem.hpp (lines 165, 224, 581)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/rocshmem/include/rocshmem/rocshmem_RMA.hpp (line 268)

### B7. GPU-aware MPI on AMD (counterpart of A11)

**Verdict: CONFIRMED.** Same situation as CUDA: a build property.

- Open MPI v5.0.x docs, ROCm: "ROCm-aware support means that the MPI library can send and receive data from AMD GPU device buffers directly. As of today, ROCm support is available through UCX. While other communication transports might work as well, UCX is the only transport formally supported in Open MPI v5.0.x for ROCm devices."
- MPICH auto-detects HIP at configure time (A11; `--with-hip=<path>`).
- AMD's cluster documentation, "GPU-enabled Message Passing Interface": "You can use these technologies add GPU pointers to MPI calls and enable ROCm-aware MPI libraries to deliver optimal performance for both intra-node and inter-node GPU-to-GPU communication." (Old URL `rocm.docs.amd.com/en/latest/how-to/gpu-enabled-mpi.html` now redirects to `instinct.docs.amd.com`.)

- https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/rocm.html
- https://instinct.docs.amd.com/projects/cluster-documentation/latest/how-to/gpu-enabled-mpi.html

---

## Group C: Metal

Sources: Apple documentation JSON, the macOS 27.0 SDK headers on this machine, MSL spec 4.1 where cited.

### C1. Streams map to command queues; can two queues run at once? (refines infra C12)

**Verdict: PARTLY.** The mapping is the course's and the projects' (PyTorch and MLX both give each stream its own `MTLCommandQueue`, D4, D6). infra C12 found no Apple statement that two queues execute concurrently; Apple does say it in one place, as a possibility: the device "can schedule" commands on two queues concurrently. There is no guarantee, no copy-engine concept, and no queue priority (C3).

- "Synchronizing events within a single device": "The first render pass and first compute pass are assumed to not depend on each other’s results and modify the same data. By encoding them on different queues, the device object can schedule these commands concurrently." Its code creates `_commandQueueA` and `_commandQueueB` and orders them with one `MTLEvent` (C2).
- "Resource synchronization": "By design, GPUs can run multiple commands in parallel."
- "Setting up a command structure": "Committing a command buffer doesn’t run its commands immediately. Instead, Metal schedules the buffer’s commands to run only after you commit prior command buffers that are waiting in the queue." "As much as possible, the perceived order in which Metal executes the commands is the same as the way you order them. Although Metal might reorder some of your commands before processing them, this usually only occurs when there’s a performance gain and no other perceivable impact."
- A documented limit relevant to a chunked lab: `newCommandQueue`: "Command Queues created via this method will only allow up to 64 non-completed command buffers." (`MTLDevice.h`); `newCommandQueueWithMaxCommandBufferCount:` and `MTLCommandQueueDescriptor.maxCommandBufferCount` (macOS 15) change it.

What a Metal lab may claim: whether two queues overlapped on the reader's Mac, measured from each command buffer's `GPUStartTime`/`GPUEndTime` (ch05 C9), labelled as observed. Apple's sentence is "can", not "will".

- https://developer.apple.com/documentation/metal/synchronizing-events-within-a-single-device
- https://developer.apple.com/documentation/metal/resource-synchronization
- https://developer.apple.com/documentation/metal/setting-up-a-command-structure
- Local headers: `MTLDevice.h` (lines 639 to 651), `MTLCommandQueue.h` (lines 88 to 94)

### C2. Metal events have different semantics from CUDA events (counterpart of A5)

**Verdict: CONFIRMED.** An `MTLEvent` is a monotonically increasing 64-bit counter that work signals and waits on by value, only between passes. A CUDA event is a re-recordable snapshot of one stream's queue (A5). The lesson's comparison table can rest on these quotes:

- `MTLEvent`: "Each event represents an unsigned 64-bit integer that starts with a value of 0, which can only increase over time." "The Metal device begins running any dependent workloads when the event equals or exceeds the value that the wait command is waiting for." "You can signal an event only with a new value that’s greater than its current value."
- Metal 3 calls: `encodeSignalEvent(_:value:)`, `encodeWaitForEvent(_:value:)` on `MTLCommandBuffer` (macOS 10.14). "A command buffer can instruct the GPU to wait for an event only between passes, not within a pass." Signal: "Metal updates the event after the GPU finishes the buffer’s prior commands." "The method can unblock one or more command buffers that are waiting for event, including those in other command queues". Metal 4: `signalEvent:value:` and `waitForEvent:value:` on `MTL4CommandQueue` ("Schedules an operation to wait for a GPU event of a specific value before continuing to execute any future GPU work.").
- No combining producers in one event: "Multiple producing workloads can’t combine their signals with one event to unblock any dependent workloads. This is because an event’s signal method can only increase its value to a specific number, unlike a semaphore ... Instead, signal when each producing workload finishes by updating its own separate event." So the book's two waits in `internalStream` (Fig. 23.14 lines 19 to 20, one per boundary kernel) translate to two events, or two values in a known order.
- Timeouts, which CUDA waits do not have: "Wait commands that time out in an [MTL4CommandQueue] unblock subsequent work in the queue, and wait commands that time out in an [MTLCommandBuffer] terminate the command buffer with the [timeout] error code."
- CPU side: `MTLSharedEvent` (CPU, other devices, other processes): `- (BOOL)waitUntilSignaledValue:(uint64_t)value timeoutMS:(uint64_t)milliseconds` (macOS 12.0), `@property (readwrite) uint64_t signaledValue;`, `notifyListener:atValue:block:`. Apple's tip: "Start with an [MTLEvent] instance until you need to synchronize work with a task that runs on the CPU or another Metal device, because an [MTLSharedEvent] can add overhead".
- Within one command buffer, across passes: `MTLFence` ("A synchronization mechanism that orders memory operations between GPU passes"), with Apple's tip to prefer `MTLEvent` across queues.

- https://developer.apple.com/documentation/metal/mtlevent
- https://developer.apple.com/documentation/metal/mtlsharedevent
- https://developer.apple.com/documentation/metal/mtlcommandbuffer/encodewaitforevent(_:value:)
- https://developer.apple.com/documentation/metal/mtlcommandbuffer/encodesignalevent(_:value:)
- https://developer.apple.com/documentation/metal/synchronizing-events-between-a-gpu-and-the-cpu
- https://developer.apple.com/documentation/metal/mtlfence
- Local headers: `MTLEvent.h` (lines 15, 52 to 64), `MTLCommandBuffer.h` (lines 404 to 416), `MTL4CommandQueue.h` (lines 260 to 275)

### C3. Stream priorities on Metal (counterpart of A1, A2)

**Verdict: CONFIRMED that none is documented** for compute or render work. A search of every Metal header in the macOS 27.0 SDK for "priority" finds it only on `MTLIOCommandQueue`, the queue that loads file data into Metal resources.

- `MTLCommandQueueDescriptor` has `maxCommandBufferCount` and `logState`; `MTL4CommandQueueDescriptor` has `label` and `feedbackQueue`. Neither has a priority.
- `MTLIOCommandQueue.h`: `typedef NS_ENUM(NSInteger, MTLIOPriority) { MTLIOPriorityHigh = 0, MTLIOPriorityNormal = 1, MTLIOPriorityLow = 2, } API_AVAILABLE(macos(13.0), ios(16.0));` and on the descriptor "@abstract The priority of the commands executed by this queue." That is file I/O, not kernels.

Lesson wording: "Metal has no documented way to ask for one queue's kernels to be preferred over another's."

- Local headers: `MTLCommandQueue.h` (lines 88 to 100), `MTL4CommandQueue.h` (lines 75 to 86), `MTLIOCommandQueue.h` (lines 19 to 24, 141 to 144)

### C4. Asynchronous copies, pinned memory, copy engines and memset on Apple silicon (counterpart of A4, A6, A7)

**Verdict: CONFIRMED.** Settled in ch05 C6, appC C2, ch02 C3 and infra C12: a shared buffer is the same memory for CPU and GPU, so the host-to-device and device-to-host copies of section 23.3 have no counterpart; there is no pinned/pageable distinction; what remains is ordering (do not touch a buffer from the CPU until the command buffer that uses it has completed). What this chapter adds:

- No copy engine is documented. No `MTLDevice` property counts copy hardware, and the blit encoder is described as a pass type ("Encodes commands that copy and modify resources for a single blit pass", infra C12), not as separate hardware. Whether a blit pass overlaps a compute pass: COULD NOT VERIFY beyond C1's "can schedule concurrently" for different queues.
- The counterpart of `cudaMemsetAsync(l2norm_d, 0, ...)` is a blit `- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range value:(uint8_t)value;` ("Fill a buffer with a fixed value in each byte."), ordered by its position in the command buffer; or, on a shared buffer, a CPU write before commit.
- The counterpart of the chapter's per-iteration "copy one float back" (Fig. 23.14 line 22) is reading `contents` of a shared buffer after the command buffer completes (ch02 C3). There is no copy to queue early; the cost to hide is the wait itself.

So on Apple silicon the stepper's "copy-in" lane becomes CPU work that writes the next chunk into a shared buffer, and "copy-out" becomes CPU work that reads a finished chunk. C5 is Apple's documented way to overlap that CPU work with GPU work.

- See ch05 C6, appC C2, ch02 C3, infra C12 for sources.
- Local header: `MTLBlitCommandEncoder.h` (lines 99 to 103)

### C5. The overlap Apple documents: several buffer instances in flight, a semaphore, a completion handler (the Metal lab's counterpart experiment)

**Verdict: CONFIRMED.** Apple's sample "Synchronizing CPU and GPU work" is the documented Metal version of the chapter's idea: let the CPU prepare chunk n+1 while the GPU works on chunk n, by giving each in-flight chunk its own buffer.

- "because the CPU and GPU are separate processors, you can make them work simultaneously by using multiple instances of a resource. ... the CPU can write position data to a buffer used for frame n+1, at the same time that the GPU reads position data from a buffer used for frame n."
- Buffers are shared storage: `[_device newBufferWithLength:triangleVertexBufferSize options:MTLResourceStorageModeShared]`, `MaxFramesInFlight` of them ("set to 3" in the sample, for a reason specific to drawables that a compute lab does not share).
- Rate control: "use a semaphore to wait for full frame completions in case the CPU is working much faster than the GPU." "At the end of the render loop, you register a command buffer completion handler. When the GPU completes the command buffer’s execution, it calls this completion handler and you increment the semaphore’s counting value by 1."
- The hazard: "You need to finalize all CPU writes to one buffer instance before you commit a command buffer that references it. Otherwise, the GPU may begin reading the buffer instance while the CPU is still writing to it."

What the Metal lab can therefore measure, with a prediction first: total time for N chunks with one buffer and `waitUntilCompleted` after each (no overlap) against K buffers in flight with a semaphore (CPU fill overlapping GPU compute). That is supported by Apple's documentation. A second, optional measurement (two command queues, compare `GPUStartTime`/`GPUEndTime` intervals) is an observation only (C1). Sample is listed for macOS 14.0 and later.

- https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work

### C6. Multi-GPU, collectives and one-sided communication on Metal (counterpart of MPI, NCCL, NVSHMEM)

**Verdict: CONFIRMED that none is documented** for Apple silicon. Settled in ch24 C3 (no NVLink counterpart; peer groups deprecated, "Not applicable on Apple Silicon"), ch24 C4 (no collective API in Metal; MLX supplies distributed communication across machines: MPI, ring over TCP, JACCL over Thunderbolt RDMA, NCCL on CUDA) and appC C4 (no peer dereference). The lesson's one sentence: "on a Mac, multi-device work is a framework feature across machines (MLX), not a Metal API".

- See ch24 C3, C4 and appC C4 for sources.

---

## Group D: how real backends use these mechanisms (the bridge, and goal 3)

Read on the default branches on 2026-11-20. Line numbers drift.

### D1. llama.cpp: `--split-mode layer` is pipeline parallelism, built on the scheduler's copies and events

**Verdict: CONFIRMED.** The default multi-GPU mode splits layers across devices and, when every device supports asynchronous compute and events, overlaps successive batches through the stages using up to 4 copies of each input with events between them. This is the chapter's overlap (one stage's transfer hidden behind another stage's compute) in production.

- `common/arg.cpp`: `"-sm", "--split-mode"`, `{none,layer,row,tensor}`: "- layer (default): split layers and KV across GPUs (pipelined)\n" "- row: split weight across GPUs by rows (parallelized)\n" "- tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL)".
- `src/llama-context.cpp`: "// enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary" with the condition `model.n_devices() > 1 && model.n_gpu_layers() > model.hparams.n_layer_all && model.split_mode() == LLAMA_SPLIT_MODE_LAYER && cparams.offload_kqv && ...`, then "// pipeline parallelism requires support for async compute and events in all devices" (`if (!props.caps.async || !props.caps.events)` turns it off), then the log line "pipeline parallelism enabled".
- `ggml/src/ggml-backend.cpp`: `#define GGML_SCHED_MAX_COPIES 4`; in `struct ggml_backend_sched`: "// pipeline parallelism support" `int n_copies; int cur_copy; int next_copy; ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];`; `sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;`; each split records an event after it and the next split waits on it (`ggml_backend_event_record`, `ggml_backend_event_wait`); `next_copy = (next_copy + 1) % n_copies` rotates the buffers. The CUDA backend implements those two calls as `cudaEventRecord` and `cudaStreamWaitEvent` (`ggml-cuda.cu`, lines 5624 to 5634).

- https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp (lines 2803 to 2808)
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp (lines 583 to 616)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp (lines 1171 to 1172, 1216 to 1220, 2118 to 2129, 2292 to 2293, 2326, 2455 to 2456)
- https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h (lines 201 to 206)

### D2. llama.cpp: `--split-mode tensor` is tensor parallelism with an all-reduce through NCCL (RCCL on AMD)

**Verdict: CONFIRMED.**

- `ggml/CMakeLists.txt`: `option(GGML_CUDA_NCCL "ggml: use NVIDIA Collective Comm. Library" ON)`; `ggml/src/ggml-cuda/CMakeLists.txt`: `find_package(NCCL)`, `add_compile_definitions(GGML_USE_NCCL)`, else "Warning: NCCL not found, performance for multiple CUDA GPUs will be suboptimal".
- `ggml-cuda.cu`: "// Communication context for multi-GPU AllReduce during tensor parallelism." and "// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large // tensors (bandwidth-bound), then converts back to FP32." The call is the book's group pattern: `NCCL_CHECK(ncclGroupStart()); ... ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()); ... NCCL_CHECK(ncclGroupEnd());`. Fallbacks named in the same struct: "comm_init_{nccl, internal, none}".
- HIP build: `ggml/src/ggml-cuda/vendors/hip.h` has `#ifdef GGML_USE_NCCL` `#include <rccl/rccl.h>`, so the same source calls RCCL (B5).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt (line 211)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/CMakeLists.txt (lines 187 to 193)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu (lines 1360 to 1443)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/hip.h (lines 11 to 13)

### D3. llama.cpp on one GPU: non-blocking streams, `cudaEventDisableTiming` events, and an opt-in fork/join of the Q, K and V projections

**Verdict: CONFIRMED.** This is the chapter's section 23.3 mechanism inside a single-GPU inference backend, and the best goal-3 pointer for the lesson.

- `common.cuh`: `#define GGML_CUDA_MAX_STREAMS 8`; `CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking));`; `cudaEvent_t events[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]; // events for synchronizing multiple GPUs`.
- Cross-device copy (`ggml-cuda.cu`): after `cudaMemcpyPeerAsync`, "// record event on src stream after the copy" with `cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming)` and `cudaEventRecord(...)`, then "// wait on dst stream for the copy to complete" `cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0)`. Exactly Fig. 23.14's record-and-wait, between two devices.
- Opt-in concurrency: enabled only when `GGML_CUDA_GRAPH_OPT` is `1` and CUDA graphs are in use. Comments: "// Target Q, K, V for concurrency", "// 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")", "// 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")". At run time: `cudaEventRecord(concurrent_event->fork_event, main_stream)`, each side stream `cudaStreamWaitEvent(stream, concurrent_event->fork_event)`, and join events recorded back into the main stream.
- The per-token "small result to host" (notes, relevance section): `ggml_backend_cuda_get_tensor_async` is `cudaMemcpyAsync(data, ..., cudaMemcpyDeviceToHost, cuda_ctx->stream())`; pinned host buffers come from `cudaMallocHost` (appC D3).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh (lines 201, 1380, 1397, 1435)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu (lines 3210, 3277 to 3286, 5330 to 5346, 5811 to 5818, 5870 to 5877)

### D4. PyTorch: stream priority convention, `wait_stream`, `record_stream`; MPS streams on `main` are command queues

**Verdict: CONFIRMED.**

- `torch/cuda/streams.py` (`main`): "priority(int, optional): priority of the stream, which can be positive, 0, or negative. A lower number indicates a higher priority. By default, the priority is set to 0. If the value falls outside of the allowed priority range, it will automatically be mapped to the nearest valid priority". `wait_event`: "This is a wrapper around ``cudaStreamWaitEvent()``". `wait_stream(stream)` is implemented as `self.wait_event(stream.record_event())`, the book's record-then-wait in one call.
- PyTorch 2.15 CUDA notes, "CUDA streams": "operations from different streams can execute concurrently in any relative order, unless explicit synchronization functions (such as synchronize() or wait_stream() ) are used." A hazard the book does not have, because the book never frees: "A may correspond to memory reallocated by the CUDA caching allocator, with pending operations from the old (deallocated) memory", which is why the fixed example adds `A . record_stream ( s )`.
- MPS (Apple): on `main`, `torch/mps/streams.py` defines `torch.mps.Stream` ("An MPS stream is a linear sequence of execution that is independent from other streams"), added in November 2026 commits (#203652, #203622, #204528) and absent from the 2.15 docs' `torch.mps` list. Underneath, `aten/src/ATen/mps/MPSStream.mm`: `_commandQueue = [MPSDevice::getInstance()->device() newCommandQueue];` per stream, and `constexpr int kMPSStreamsPerPool = 32;`. So PyTorch maps "stream" to "command queue", as C1 does. Not checked: whether it ships in a release yet.

- https://github.com/pytorch/pytorch/blob/main/torch/cuda/streams.py (lines 26 to 100)
- https://docs.pytorch.org/docs/2.15/notes/cuda.html (section "CUDA streams")
- https://github.com/pytorch/pytorch/blob/main/torch/mps/streams.py
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/mps/MPSStream.mm (lines 48, 439)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/mps/MPSStream.h (lines 203 to 210)

### D5. PyTorch: symmetric memory over NVSHMEM, rocSHMEM or NCCL windows; pipeline parallelism and its "bubble"

**Verdict: CONFIRMED.** Two pages that tie the chapter's last two sections to a framework the reader uses.

- 2.15, "PyTorch Symmetric Memory" (alpha): "software libraries such as NVSHMEM or rocSHMEM abstract away the programming difference between intra-node access and inter-node access with primitives that are slightly higher level than pointer access, such as put and get. PyTorch provides NVSHMEM plugins to augment Triton kernels’ cross-node capabilities." `is_nvshmem_available()`: "Check if NVSHMEM (CUDA) or rocSHMEM (ROCm) is available". With `symm_mem.set_backend("NCCL")`, "rendezvous window-registers the allocation" (the A13 windows). This is the convergence the book predicts on p. 575.
- 2.15, "Pipeline Parallelism" (alpha): "It allows the execution of a model to be partitioned such that multiple micro-batches can execute different parts of the model code concurrently. Pipeline parallelism can be an effective technique for: large-scale training bandwidth-limited clusters large model inference". The same page uses "pipeline startup bubble" for the idle time before a rank's first forward, the word the stepper uses.

- https://docs.pytorch.org/docs/2.15/symmetric_memory.html
- https://docs.pytorch.org/docs/2.15/distributed.pipelining.html

### D6. MLX on Metal: one command queue per stream, `MTLSharedEvent` values between streams, and `async_eval` overlapping the decode loop

**Verdict: CONFIRMED.**

- `mlx/backend/metal/device.cpp`: each `CommandEncoder` makes its own queue, `queue_ = NS::TransferPtr(device_.mtl_device()->newCommandQueue());`, and `get_command_encoder(Stream s)` looks it up by `s.index`. Cross-stream ordering: `buffer_->encodeSignalEvent(event.cast<EventImpl>().mtl_event(), value);` and `buffer_->encodeWait(event.cast<EventImpl>().mtl_event(), value);`. `mlx/backend/metal/event.cpp`: `newSharedEvent()`, and the CPU wait `mtl_event_->waitUntilSignaledValue(value, -1); // never times out`. That is C2's value-based events in production.
- mlx-lm `mlx_lm/generate.py`: `generation_stream = mx.new_thread_local_stream(mx.default_device())`, and in `generate_step` the next step is queued before the current token is read: `next_y, next_logprobs = _step(y)` `mx.async_eval(next_y, next_logprobs)` ... `yield y.item(), logprobs`. The CPU builds and submits step n+1 while the GPU is still computing; `y.item()` is the per-token host read the notes compare with the chapter's per-iteration norm copy.
- MLX docs: `async_eval(*args)`: "Asynchronously evaluate an array or tree of array . Note This is an experimental API and may change in future versions."
- Tensor-parallel layers in `python/mlx/nn/layers/distributed.py`: `class AllToShardedLinear`, "Each member of the group applies part of the affine transformation such that the result is sharded across the group." and `class ShardedToAllLinear`, "Each member of the group applies part of the affine transformation and then aggregates the results." (the aggregation is `mx.distributed.all_sum`).

- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device.cpp (lines 341 to 351, 557, 563, 1012 to 1017)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/event.cpp (lines 16, 29)
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py (lines 245, 476 to 483)
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html
- https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/distributed.py (lines 228, 309, 364)

### D7. vLLM: tensor parallelism inside a node, pipeline parallelism across nodes (for the bridge)

**Verdict: CONFIRMED.** The bridge the lesson draws ("tensor and pipeline parallelism in inference servers rest on the same overlap") is ours; these are the server facts it stands on.

- "Parallelism and Scaling": "Single-node multi-GPU using tensor parallel inference: if the model is too large for a single GPU but fits on a single node with multiple GPUs, use tensor parallelism ." "Multi-node multi-GPU using tensor parallel and pipeline parallel inference: if the model is too large for a single node, combine tensor parallelism with pipeline parallelism . Set tensor_parallel_size to the number of GPUs per node and pipeline_parallel_size to the number of nodes."
- "The implementation includes Megatron-LM's tensor parallel algorithm ." "Efficient tensor parallelism requires fast internode communication, preferably through high-speed network adapters such as InfiniBand." (with `NCCL_IB_HCA` in the example).

What is ours, to label `<Beyond>`: tensor parallelism puts a collective (an all-reduce, D2) on every layer's critical path, so it wants the fastest link and overlaps poorly; pipeline parallelism sends activations only between stages and hides the transfer behind other micro-batches' compute, at the cost of a fill-and-drain bubble (E1). That mapping is the chapter's halo-exchange argument applied to model layers; neither the book nor vLLM states it in those words.

- https://docs.vllm.ai/en/latest/serving/parallelism_scaling/

---

## Group E: arithmetic for the stepper (ours)

No vendor fact here beyond the quotes cited. The numbers are unitless and for authors and unit tests, never lesson content as device figures.

### E1. Chunked copy-in, compute, copy-out with enough streams and separate resources

**Verdict: CONFIRMED** (arithmetic, checked by a brute-force schedule simulation over 2,000 random cases).

Model: n chunks; per chunk host-to-device time a, kernel time b, device-to-host time c; one stream per chunk (so no false dependencies, A8); three independent resources (one H2D engine, one D2H engine, the SMs running one chunk's kernel at a time); each resource serves work in issue order.

- Total time T = a + b + c + (n − 1) · max(a, b, c). The sequential (one stream, no overlap) time is n · (a + b + c).
- With c = 0 and a = tT/n, b = tE/n this gives tE + tT/n when tE ≥ tT and tT + tE/n otherwise: exactly NVIDIA's "rough estimate" (A7). The simulation reproduced it in all 500 random cases tried.
- Idle time of the bottleneck resource, the "bubble": T − n · max(a, b, c) = (a + b + c) − max(a, b, c), the fill and drain at the two ends. It does not shrink as n grows; its share of T does.
- Example (unitless): n = 4, a = 1, b = 2, c = 1 gives T = 10 against 16 sequential.

### E2. One shared copy resource: the answer depends on issue order

**Verdict: CONFIRMED** (arithmetic; simulation). This is why the stepper must say what it assumes.

If host-to-device and device-to-host share one copy resource served in issue order (the single-copy-engine case of A7):

- Issuing depth-first (copy-in 0, kernel 0, copy-out 0, copy-in 1, ...) gives no overlap at all in the example above: T = 16, the same as sequential, because copy-out 0 sits in the copy queue ahead of copy-in 1 and must wait for kernel 0. With n = 8 and a = b = c = 1 it is 24 (sequential 24).
- Issuing breadth-first (all copy-ins, then all kernels, then all copy-outs) gives T = 10 and 16 for the same two cases.
- The bottleneck formula a + b + c + (n − 1) · max(a + c, b) failed for depth-first issue in about 1,830 of 2,000 random cases.
- NVIDIA documents neither the FIFO assumption for today's GPUs nor how `CUDA_DEVICE_MAX_COPY_CONNECTIONS` (A8) changes it. So the stepper should model two separate copy resources by default (the case A7 describes for two engines), offer one shared resource as an option, state the issue order it uses, and say both are simplifications.

### E3. Per-chunk overhead (launch and copy setup) that makes "more streams" stop helping

**Verdict: COULD NOT VERIFY** a vendor figure. NVIDIA's guide calls per-launch cost "a significant fraction" for short kernels without a number (ch22 A3), AMD calls direct-launch overhead "negligible" (ch22 B2), Apple gives none. The stepper should expose it as a parameter (time added to every copy and every kernel), default it to a visible but arbitrary unitless value, and label it as such. The lab is where the reader finds their own optimum chunk count.

---

## Book items with no vendor fact to check

Recorded so the wrap page's errata list does not wait on this file (from `notes/ch23.md`, "Unclear in the PDF"):

- "Fig. 23.5 shows five essential MPI functions" (p. 546); four are shown.
- Fig. 23.14 line 19 has an extra closing parenthesis (p. 557).
- "line 32" (p. 559) should be line 31.
- "Soon after the cudaMemset operation is complete" (pp. 561 to 562) means `cudaMemsetAsync`; A3 adds that the book's description of `cudaMemset` as blocking is itself wrong.
- Identifier slips: `l2Norm`, `cudaStreamsynchronize`, `nccl_comm` vs `ncclComm`, undefined `PE`/`numPEs`.
- "Thick dotted" vs the legend's thick dashed (p. 562).
- The periodic-boundary sentence (p. 551), "5-point stencil" vs the 4-neighbour average (p. 576), "L2 norm" vs its square, the unshown initial `l2norm`, the undefined `launch_jacobi_kernel` and `blockReduce`, Fig. 23.13(b)'s bar placement, the exercise ambiguities and grammar slips: book-internal.
- Resolved here, no longer unclear: the stream priority direction (A1: lower number, higher priority; the book's code is right).

## Answers to the notes' portability questions

- Streams on HIP: same names (infra B8), same "may run concurrently" (ch22 B1); priorities accepted and ignored by default (B2); at most 4 hardware queues by default (ch22 B1). On Metal: command queues; Apple says two queues "can" run concurrently (C1); no priority (C3).
- Events on HIP: same names and flags (infra B8, B4). On Metal: `MTLEvent` and `MTLSharedEvent` with monotonic values, waits only between passes, one event per producer, waits can time out (C2).
- Async memset and memcpy, pinned memory on HIP: `hipMemsetAsync`, `hipMemcpyAsync`, `hipHostMalloc` (infra B8, B4); pageable async copies become synchronous, as on CUDA. On Metal: no host-device copy for shared buffers, no pinned memory, `fillBuffer:range:value:` for a memset (C4).
- CUDA-aware MPI on AMD: ROCm-aware Open MPI through UCX, MPICH with HIP detected at configure time (B7).
- NCCL and NVSHMEM counterparts: RCCL with the same names and header `<rccl/rccl.h>` (B5); rocSHMEM with `rocshmem_` names and `_wg`/`_wave` variants (B6); on Metal, none (C6).
- `atomicAdd` and block reduction in Fig. 23.1: settled in infra A1, B1, C1, A11, B11 (float atomic add exists on all three; CUB, hipCUB, and on Metal `simd_sum` plus threadgroup memory).

---

## Notes for the lab runtime (not vendor facts)

Recorded so the chapter build can plan; this file changes nothing in shared code.

- `labs/common/include/pmpp/vendors/hip.h` maps the stream and event calls listed in infra B8 but not `cudaMemsetAsync`, `cudaStreamCreateWithPriority`, `cudaDeviceGetStreamPriorityRange`, `cudaStreamGetPriority`, `cudaHostRegister` or `cudaStreamPerThread`. All have HIP counterparts with the plain prefix swap (`hipMemsetAsync` and the priority calls at `hip_runtime_api.h` lines 3235, 3255, 3374, 5749). A lab that uses them needs them added to the shared header (a change the chapter build must request, not make). B2 says priorities do nothing on AMD by default.
- The emulator (`emu/emu_host.h`) has synchronous stand-ins for streams, events, `cudaMemcpyAsync` and `cudaMemsetAsync`, but no `cudaStreamCreateWithPriority`, `cudaDeviceGetStreamPriorityRange` or `cudaHostRegister`. Its `cudaDeviceProp` (`emu/emu_types.h`) and the front-end stub's (`check/cuda_stub.h`) have `concurrentKernels` but not `asyncEngineCount` or `streamPrioritiesSupported`. A lab's `main.cpp` that prints those fields needs them added or guarded. The emulator runs everything synchronously, so it checks correctness only; it cannot show overlap (consistent with `labs/README.md`).
- `python3 labs/pmpp.py doctor` reports neither `asyncEngineCount` nor `concurrentKernels` on CUDA or HIP (`labs/common/src/backend_cudalike.cpp`). The stepper's "copy engines" control cannot read the reader's value from `useReports()` today; either the lab prints it into its run JSON or the stepper offers 1 and 2 as "the two cases NVIDIA describes" (A7). On HIP the value is a constant 8 and means nothing about hardware (B3).
- Metal side of the lab: time per command buffer only (ch05 C9), so each chunk's GPU work should be its own command buffer if chunk intervals are to be drawn; queues hold at most 64 uncompleted command buffers by default (C1).

---

## Summary of corrections

Every WRONG, PARTLY and COULD NOT VERIFY item, one line each:

- A2: stream priorities are a scheduling hint for pending kernels only; they "do not preempt already-running work", guarantee no order, and have no effect on memory copies. The book's fix helps only when both kernels are still waiting to start.
- A3 (WRONG): `cudaMemset` on device memory is asynchronous with respect to the host; what makes it unsuitable in Fig. 23.14 is that it is queued in the legacy default stream, which synchronises with all blocking streams.
- A6: `cudaMemcpyAsync` with pageable host memory is legal but "will revert to a synchronous behavior which will not overlap with other work"; "requires pinned" is the Best Practices Guide's wording, not the reference's. On HMM/ATS systems `cudaMallocHost` "may not page-lock" at all.
- A11: CUDA-aware MPI is a build property: MPICH enables GPU support automatically if CUDA (or HIP) is found at configure time; Open MPI needs `--with-cuda` and a CUDA-enabled UCX; MVAPICH2's GPU line is now MVAPICH-Plus. NCCL warns that mixing NCCL and CUDA-aware MPI on the same GPUs concurrently can deadlock.
- A13: NCCL already has one-sided communication (`ncclPutSignal`, `ncclWaitSignal`, since 2.29), on top of symmetric windows (2.27) and a device API (2.28); the book's "may add soon" is out of date.
- A16: reference [5]'s CUB URL is 404 (now `nvidia.github.io/cccl/unstable/cub/`); [3] redirects to `/nvshmem/api/latest/`; [8]'s GPU stack is now "MVAPICH-Plus".
- B1: HIP's stream and event calls match CUDA's by name, not fully by behaviour (B2, B3, ch22 B1).
- B2 (WRONG): ROCm 10.2 ignores stream priorities by default (`DEBUG_HIP_IGNORE_STREAM_PRIORITY` defaults to true), while still reporting `streamPrioritiesSupported = 1` and a range of -1 to 1; AMD's how-to page still describes priorities as working.
- B3: on ROCm, `asyncEngineCount` is the runtime constant 8 for every device, not a count of copy engines.
- B4 (sub-claim, COULD NOT VERIFY): whether `hipMemset` returns before the fill completes is not documented.
- C1: Apple does document that commands on two queues "can" be scheduled concurrently (refines infra C12, which found no such statement); it is a possibility, not a guarantee, and there is no queue priority or copy-engine concept.
- C4 (sub-claim, COULD NOT VERIFY): whether a blit pass overlaps a compute pass on Apple silicon.
- E3 (COULD NOT VERIFY): no vendor gives a per-chunk launch or copy overhead; the stepper must treat it as a labelled parameter.

Status and naming facts worth knowing even where the verdict is CONFIRMED:

- A1: the book's `low, high` variable order is correct (least priority is the first output); lower numbers mean higher priority on CUDA, HIP and in PyTorch.
- A5: a CUDA event is a re-recordable snapshot of a stream's queue; `cudaStreamWaitEvent` uses the state captured when it is called.
- A7: NVIDIA's own staged copy-compute example and its time estimate are in the Best Practices Guide, section 10.1.2; copies overlap each other only with two copy engines.
- A8: CUDA maps streams onto 8 work queues by default (`CUDA_DEVICE_MAX_CONNECTIONS`, 1 to 32); more active streams than queues can serialise unrelated work.
- A10: MPI 5.0 (June 2025) is the current standard; the standard names `mpiexec`.
- A12: `ncclCommInitRank` needs the CUDA device set first; NVIDIA's own bootstrap example has no `MPI_Barrier`.
- A14: NVSHMEM recommends initialising `nvshmemx_init_attr_t` with `NVSHMEMX_INIT_ATTR_INITIALIZER`; `nvshmem_float_p` may return before delivery.
- A15: NVSHMEM is open source on GitHub (Apache-2.0) since August 2025; current release 3.8.0.
- C2: Metal events are monotonic 64-bit counters with value-based waits that can time out; one event cannot combine several producers.
- D1 to D3: llama.cpp's default multi-GPU mode is pipelined with scheduler events; its tensor mode all-reduces through NCCL or RCCL; on one GPU it uses non-blocking streams and can (opt-in) run Q, K and V on separate streams with fork and join events.
- D4: PyTorch `main` adds `torch.mps.Stream`, each a separate `MTLCommandQueue` from a pool of 32; not in the 2.15 docs.

## Where the backends really differ

One line per cell. Nothing here was run on hardware by us.

| Feature | CUDA (13.5) | HIP (ROCm 10.2) | Metal (macOS 27 SDK) |
|---|---|---|---|
| The ordered queue of work | `cudaStream_t` | `hipStream_t`, same calls | `MTLCommandQueue` of `MTLCommandBuffer`s; a serial compute encoder within one buffer |
| Implicit cross-queue blocking | Legacy default stream syncs with all blocking streams; avoid with `cudaStreamNonBlocking` or per-thread default stream | "The default stream is a blocking operation"; `hipStreamNonBlocking` | None; queues are independent |
| May two queues run at once | "may execute concurrently" given resources | Same; at most 4 hardware queues by default, extra streams share round-robin | Apple: the device "can schedule these commands concurrently"; not guaranteed |
| Streams per hardware queue | 8 work queues by default (`CUDA_DEVICE_MAX_CONNECTIONS`, 1 to 32) | `GPU_MAX_HW_QUEUES` default 4 | Not documented; 64 uncompleted command buffers per queue by default |
| Stream priority | `cudaStreamCreateWithPriority`; lower number is higher; hint only, no preemption, kernels only | Same API; ignored by default (`DEBUG_HIP_IGNORE_STREAM_PRIORITY=true`) yet reports supported | None for compute queues (only `MTLIOCommandQueue` has a priority) |
| Cross-queue dependency | `cudaEventRecord` + `cudaStreamWaitEvent` (snapshot of a queue) | `hipEventRecord` + `hipStreamWaitEvent` | `encodeSignalEvent`/`encodeWaitForEvent` with a monotonic value, between passes only; waits can time out |
| Cheap sync-only event | `cudaEventDisableTiming` | `hipEventDisableTiming` | `MTLEvent` (vs costlier `MTLSharedEvent`) |
| Host waits for a queue | `cudaStreamSynchronize` | `hipStreamSynchronize` | `waitUntilCompleted`; or `MTLSharedEvent waitUntilSignaledValue:timeoutMS:` |
| Async host-to-device copy | `cudaMemcpyAsync` from pinned memory | `hipMemcpyAsync` from `hipHostMalloc` memory | None needed: write a shared buffer's `contents` before commit |
| Async copy from pageable memory | Works, but "will revert to a synchronous behavior" | "the memory copy will be performed synchronously" | Not applicable |
| Pinned memory | `cudaMallocHost`, `cudaHostAlloc`, `cudaHostRegister` | `hipHostMalloc`, `hipHostRegister` (+ coherence flags) | No pinned/pageable distinction documented |
| Copy hardware query | `asyncEngineCount` = number of copy engines; 2 lets H2D and D2H overlap | `asyncEngineCount` is the constant 8 on every device | No copy engine documented; blit is a pass type |
| Async memset | `cudaMemsetAsync`; plain `cudaMemset` is host-async but in the default stream | `hipMemsetAsync`; `hipMemset` host behaviour not documented | Blit `fillBuffer:range:value:` |
| Documented overlap pattern | Staged copy and execute over n streams (Best Practices 10.1.2) | Same, described in HIP's async page | Several buffers in flight, semaphore, completion handler ("Synchronizing CPU and GPU work") |
| Timing the overlap | Events, `cudaEventElapsedTime` (ms) | `hipEventElapsedTime` (ms) | `GPUStartTime`/`GPUEndTime` per command buffer (s) |
| Collective library | NCCL (`<nccl.h>`), with one-sided and device APIs | RCCL (`<rccl/rccl.h>`), NCCL's names on `hipStream_t` | None in Metal; MLX distributed (MPI, ring, JACCL; NCCL on CUDA) |
| One-sided, in-kernel communication | NVSHMEM (`nvshmem_float_p`, `nvshmemx_barrier_all_on_stream`) | rocSHMEM (`rocshmem_float_p`, `rocshmem_barrier_all_on_stream`, `_wg`/`_wave`) | None |
| GPU-aware MPI | CUDA-aware builds (MPICH auto, Open MPI `--with-cuda` + UCX, MVAPICH-Plus) | ROCm-aware Open MPI via UCX; MPICH auto-detects HIP | None |
| What runtimes use | llama.cpp: non-blocking streams, `DisableTiming` events, NCCL all-reduce for tensor split; PyTorch `torch.cuda.Stream` | llama.cpp: same source with RCCL; PyTorch ROCm uses the same API | MLX: a command queue per stream, `MTLSharedEvent` values, `async_eval`; PyTorch `main`: `torch.mps.Stream` = command queue |
