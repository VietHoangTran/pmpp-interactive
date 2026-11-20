# Chapter 2 fact check: CUDA, HIP, Metal

Checked on 2026-11-20 against the live vendor sources listed under each item.

How to read this file:

- **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims, the verdict is for the item as a whole and the text says which sub-claim failed.
- **Quote** is copied from the source (HTML docs flattened to text, GitHub raw files, Apple's documentation JSON, PDF text, local SDK headers). Line breaks inside quotes are removed.
- Items already settled in `notes/today/ch05-facts.md` are not repeated. They are cited as "ch05 A5", "ch05 C6" and so on. There is no `notes/today/infra-facts.md` on the check date.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide "v13.5" (pages last updated Oct 31, 2026), CUDA Runtime API reference and nvcc manual (last updated Nov 02, 2026); ROCm Core SDK 10.2.0 (HIP docs "HIP 7.17.0 Documentation", headers at tag `therock-10.2`); Metal Shading Language Specification "Version 4.1", dated 2026-06-04; local macOS SDK 27.0 headers; llama.cpp `master` at commit `b94959a` (2026-11-19); MLX `main` at `8f0f391` (2026-11-20); PyTorch `main` at `538adf4` (2026-11-20).
- The NVIDIA runtime API reference moved: `https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html` now redirects to `https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html`. The URLs below are the ones the server finally answered with.
- Not checked: anything by running code on hardware. Every statement is from documentation, headers or project source.

Totals: 31 CONFIRMED, 0 WRONG, 7 PARTLY, 3 COULD NOT VERIFY (41 items). The PARTLY items are A1, A2, A7, A13, A14, A15, A18. The COULD NOT VERIFY items are A19, A21, B8. Sub-claims that could not be found are called out inside A2, A5, A8 and B7.

---

## Group A: CUDA, and the book's dated statements

### A1. The book's reference [3], "CUDA C++ Programming Guide, Release 13.0, 2025"

**Verdict: PARTLY.** That document still exists, but only as an archive. The current guide is v13.5, has been restructured, and now opens with the same vector addition program this chapter builds.

- Current guide header: "CUDA Programming Guide v13.5". Its section 2.1 "Intro to CUDA C++" walks through a `vecAdd` kernel, the index formula, bounds checking, the block-count arithmetic, `cudaMalloc`/`cudaMemcpy`/`cudaFree`, error checking and `__host__`/`__device__`. It is the closest thing to an official companion for this chapter.
- The 13.0 text is at the archive URL and identifies itself as "v13.0". Section numbers in it (for example "6.2.8.1. Concurrent Execution between Host and Device", "13.1" for dynamic parallelism) do not exist in v13.5.
- URL changes are in ch05 A9.

Sources:

- https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html
- https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html

### A2. `cudaMalloc`: signature, the `(void**)` cast, size in bytes

**Verdict: PARTLY.** The signature and the "size in bytes" rule are confirmed. The book's statement that the first argument must be cast to `(void**)` is no longer the whole story: the C++ API now has a typed overload, and NVIDIA's own tutorial calls `cudaMalloc(&devA, ...)` with a `float*` and no cast.

Quotes (runtime API reference, Memory Management):

- `__host__ cudaError_t cudaMalloc(void **devPtr, size_t size)`: "Allocate memory on the device. Allocates size bytes of linear memory on the device and returns in *devPtr a pointer to the allocated memory. The allocated memory is suitably aligned for any kind of variable. The memory is not cleared. cudaMalloc() returns cudaErrorMemoryAllocation in case of failure."
- Returns: "cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation, cudaErrorExternalDevice".
- "Note that this function may also return error codes from previous, asynchronous launches."

Quotes (runtime API reference, "C++ API Routines"):

- `template<class T> __host__ cudaError_t cudaMalloc(T **devPtr, size_t size)` is listed beside the C form, with matching typed overloads for `cudaMallocHost`, `cudaMallocManaged`, `cudaMallocAsync` and `cudaMallocPitch`.

Quote (programming guide 2.1.3.2): `float* devA = nullptr;` ... `cudaMalloc(&devA, vectorLength*sizeof(float));`

Things to know:

- "The memory is not cleared." A kernel that skips elements (a wrong block count) leaves whatever was there. Compare C2: Metal's plain allocation is zero-filled.
- Could not find: which toolkit version introduced the typed overload. The 13.5 reference lists it; I did not look for the first release that had it.
- For this course's labs the cast is still the portable spelling. HIP has the same typed wrapper (B1), but the course emulator declares only the `void**` form on purpose (`labs/common/include/pmpp/emu/cuda_emu.h`, comment "No typed overload on purpose: real CUDA takes void**"). That comment is out of date as a statement about CUDA; the cast compiles everywhere, so student code should keep it and the lesson can say the cast is optional in current CUDA C++.

Sources:

- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__HIGHLEVEL.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (section 2.1.3.2)

### A3. `cudaFree`

**Verdict: CONFIRMED.** `__host__ cudaError_t cudaFree(void *devPtr)`, taking the pointer value.

Quotes:

- "Frees the memory space pointed to by devPtr, which must have been returned by a previous call to one of the following memory allocation APIs - cudaMalloc(), cudaMallocPitch(), cudaMallocManaged(), cudaMallocAsync(), cudaMallocFromPoolAsync()."
- "If cudaFree(devPtr) has already been called before, an error is returned. If devPtr is 0, no operation is performed."
- "For all other pointers, this API may perform implicit synchronization." (The sentence before it exempts pointers from the stream-ordered allocator.)
- Returns: "cudaSuccess, cudaErrorInvalidValue".

So a double free is a reportable error, freeing a null pointer is harmless, and a timer around `cudaFree` may include waiting for outstanding work.

- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html

### A4. `cudaMemcpy`: argument order and copy kinds

**Verdict: CONFIRMED.** Destination, source, byte count, kind. The four kinds the book lists exist with these values, and there is a fifth.

Quotes:

- `__host__ cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)`: "Copies count bytes from the memory area pointed to by src to the memory area pointed to by dst, where kind specifies the direction of the copy, and must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or cudaMemcpyDefault."
- "Passing cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer values. However, cudaMemcpyDefault is only allowed on systems that support unified virtual addressing."
- "Calling cudaMemcpy() with dst and src pointers that do not match the direction of the copy results in an undefined behavior."
- "This function exhibits synchronous behavior for most use cases."
- Returns: "cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection".
- Enum `cudaMemcpyKind`: `cudaMemcpyHostToHost = 0`, `cudaMemcpyHostToDevice = 1`, `cudaMemcpyDeviceToHost = 2`, `cudaMemcpyDeviceToDevice = 3`, `cudaMemcpyDefault = 4` ("Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing").
- Programming guide 2.1.3.2: "The cudaMemcpy API is synchronous. That is, it does not return until the copy has completed."

Note for the argument-order drill: swapping `dst` and `src` while keeping the kind is documented as undefined behaviour, not as a guaranteed error code. NVIDIA's own tutorial now passes `cudaMemcpyDefault` in both directions.

- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__TYPES.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html

### A5. Does a kernel launch block? Synchronisation and timing

**Verdict: CONFIRMED** (the book is silent, see the notes on Fig. 2.3 and Fig. 2.8; the docs are explicit). A launch returns immediately. `cudaDeviceSynchronize()` or a later blocking call is what waits.

Quotes:

- Guide 2.1.2.2: "Kernel launches are asynchronous with respect to the host thread. That is, the kernel will be setup for execution on the GPU, but the host code will not wait for the kernel to complete (or even start) executing on the GPU before proceeding."
- Guide 2.1.4: "The simplest way to synchronize the GPU and a host thread is with the use of cudaDeviceSynchronize, which blocks the host thread until all previously issued work on the GPU has completed."
- Guide 5.4.1.1: "Calls to a __global__ and __tile_global__ function are asynchronous. They return to the host thread before the device completes execution."
- Guide 1.2.1: "The CPU and GPU can both be executing code simultaneously".

Why the book's Fig. 2.13 still copies back finished results without an explicit wait. Two documented facts combine:

- Guide 2.5.2: "CUDA has a default stream, and operations and kernel launches without a specific stream are queued into this default stream." and "The order of execution of operations in a stream is sequential and the operations are executed in the order they are enqueued into the stream."
- Guide 2.5.2.3: "the cudaMemcpy() function blocks until the memory transfer is complete."
- Guide 2.5.5: "CUDA streams are what are known as in-order streams. ... An operation in a stream cannot leap-frog other operations."

The inference (launch and blocking copy share the default stream, the stream is in-order, the copy does not return until done) is mine; I did not find one sentence in v13.5 that says "cudaMemcpy waits for earlier kernels".

Timing, for the copy-versus-kernel lab:

- A host timer around the launch statement alone measures only the launch. Put `cudaDeviceSynchronize()` before stopping the clock.
- The guide's own method for kernel time (section 2.5.3.3) is events: `cudaEventRecord(start, stream);` launch `cudaEventRecord(stop, stream);` `cudaStreamSynchronize(stream);` `cudaEventElapsedTime(&elapsedTime, start, stop);` with the result in milliseconds ("Kernel execution time: ... ms"). "When an event reaches the front of a stream it records a timestamp."
- Guide 2.1.6: runtime initialisation is lazy. "This is important when timing runtime function calls and when interpreting the error code from the first call into the runtime." The first API call pays for context creation, during which "the device code is just-in-time compiled if necessary and loaded into device memory". Warm up before timing.
- Could not find in v13.5: the older rule that small host-to-device copies are asynchronous. The archived 13.0 guide lists, among operations "asynchronous with respect to the host", "Memory copies from host to device of a memory block of 64 KB or less". I did not find that sentence in the v13.5 pages I searched (1.2, 1.3, 2.1, 2.3, 2.5, 2.6, 3.1, 3.5). Treat it as possibly still true: for small arrays, synchronise before stopping a copy timer too.
- Debug aid (guide 2.5.4.2): "set the environment variable CUDA_LAUNCH_BLOCKING=1 ... The effect of this environment variable is to synchronize after every single kernel launch."

Sources:

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (sections 2.1.2.2, 2.1.4, 2.1.6)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html (sections 2.5.2, 2.5.2.3, 2.5.3.3, 2.5.4.2, 2.5.5)
- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html (section 1.2.1)
- https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html (section 6.2.8.1)

### A6. Error checking: `cudaError_t`, `cudaSuccess`, `cudaGetErrorString`

**Verdict: CONFIRMED.** All three names are current, and the book's suggestion to wrap the check in a macro is what NVIDIA's guide does. The book's sidebar leaves out the part that matters most for a first kernel: the launch itself returns nothing.

Quotes:

- `typedef enum cudaError cudaError_t`; `cudaSuccess = 0` ("The API call returned with no errors.").
- `__host__ const char *cudaGetErrorString(cudaError_t error)`: "Returns the description string for an error code." Also `cudaGetErrorName` ("Returns the string representation of an error code enum name.").
- Guide 2.1.7: "Every CUDA API returns a value of an enumerated type, cudaError_t. ... In production applications, it is best practice to always check and manage the return value of every CUDA API call."
- The guide's macro: `#define CUDA_CHECK(expr_to_check) do { cudaError_t result = expr_to_check; if(result != cudaSuccess) { fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__, __LINE__, result, cudaGetErrorString(result)); } } while(0)`
- Guide 2.1.7.1: "Kernel launches using triple chevron notation do not return a cudaError_t. It is good practice to check the error state immediately after kernel launches". And: "A value of cudaSuccess when checking the error state immediately after a kernel launch does not mean the kernel has executed successfully or even started execution. It only verifies that the kernel launch parameters and execution configuration passed to the runtime did not trigger any errors".
- "cudaGetLastError returns current error state and then resets it to cudaSuccess. Alternatively, cudaPeekAtLastError returns error state without resetting it." The state is "for each host thread".
- Guide 2.1.7.2: "errors which occur during the execution of asynchronous operations will only be reported when the error state is examined next." and "error code from an asynchronous error, such as an invalid memory access by a kernel, will be returned by every CUDA runtime API until the error state has been cleared by calling cudaGetLastError."
- The guide's launch pattern: `vecAdd<<<blocks, threads>>>(devA, devB, devC);` then `CUDA_CHECK(cudaGetLastError());` then `CUDA_CHECK(cudaDeviceSynchronize());` with the comment "The CUDA_CHECK will report errors that occurred during execution of the kernel".

Error values a first program meets:

| Enumerator | Value | Documented meaning |
|---|---|---|
| `cudaErrorInvalidValue` | 1 | "one or more of the parameters passed to the API call is not within an acceptable range of values." |
| `cudaErrorMemoryAllocation` | 2 | "unable to allocate enough memory or other resources to perform the requested operation." (the book's out-of-memory example) |
| `cudaErrorInvalidConfiguration` | 9 | "a kernel launch is requesting resources that can never be satisfied by the current device." |
| `cudaErrorInvalidMemcpyDirection` | 21 | (returned by `cudaMemcpy`) |
| `cudaErrorIllegalAddress` | 700 | "The device encountered a load or store instruction on an invalid memory address." |
| `cudaErrorLaunchFailure` | 719 | "An exception occurred on the device while executing a kernel." |

What this means for the bug hunt: a kernel that writes past the end of an array is not guaranteed to be reported. The docs describe `cudaErrorIllegalAddress` for an invalid address; an out-of-range index that still lands inside mapped device memory is not an invalid address. I found no statement that CUDA bounds-checks device arrays, which is why the emulator with AddressSanitizer is the tool that catches the missing guard.

Also current: the `CUDA_LOG_FILE` environment variable. The guide's example launches `k<<<8192, 4096>>>()` and gets "invalid argument" from the macro, while the log says "One or more of block dimensions of (4096,1,1) exceeds corresponding maximum value of (1024,1024,64)". "Error log management and CUDA_LOG_FILE are available with NVIDIA Driver version r570 and later."

Sources:

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (section 2.1.7)
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__ERROR.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__TYPES.html

### A7. Function qualifiers `__global__`, `__device__`, `__host__` (the book's Fig. 2.11)

**Verdict: PARTLY.** The three qualifiers and their callable-from and executed-on rules are confirmed. The book's table is no longer complete: the current guide's table has five specifiers.

Guide Table 45 "Execution Space Specifier" (v13.5), restricted to the rows the book has:

| Specifier | Executed in | Callable from |
|---|---|---|
| `__host__`, no specifier | Host | Host |
| `__device__` | SIMT (device) | SIMT (device) |
| `__global__` | SIMT (device) | Host and SIMT (device) |

Quotes:

- "Constraints for __global__ and __tile_global__ functions: Must return void. Cannot be a member of a class, struct, or union. Requires an execution configuration ... Does not support recursion."
- "Functions declared with multiple execution spaces (for example, __host__ __device__) are compiled for each context. The __CUDA_ARCH__ macro can be used to differentiate the host and device code paths".
- Guide 2.1.8: "A function, including class member functions, functors, and lambdas, can be specified as both __device__ and __host__".
- "Most often, kernels are launched from the host, however it is possible to launch a kernel from within another kernel using dynamic parallelism." (see A13)

New since the book: `__tile__` and `__tile_global__`. "CUDA Tile provides a different approach to writing GPU kernel code than the single instruction multiple thread (SIMT) model", and "CUDA Tile C++ is available in the CUDA Toolkit from version 13.3 onward." The guide adds that the host side is unchanged: "allocating device memory, transferring data between host and device, and sequencing kernel launches is identical". Everything in this chapter is the SIMT model. I did not check tile kernels further; they are NVIDIA-only as far as the HIP and Metal sources I read go (neither mentions them).

Sources:

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.1.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (sections 2.1.8, 2.1.9.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html

### A8. Launch syntax `<<< >>>` and the block-count arithmetic

**Verdict: CONFIRMED.** The first two parameters are grid and block dimensions. There are two more optional ones (ch05 A7).

Quotes:

- Guide 5.4.3: "The execution configuration is specified by inserting an expression in the form <<<grid_dim, block_dim, dynamic_smem_bytes, stream>>> between the function name and the parenthesized argument list". `grid_dim` and `block_dim` are "of type dim3".
- Guide 2.1.2.2.1: "When using 1-dimensional thread blocks or grids, integers can be used to specify dimensions."
- "The arguments for the execution configuration are evaluated before the arguments for the actual function."
- "The function call fails if grid_dim or block_dim exceeds the maximum sizes allowed for the device".
- "There are two ways of launching kernels from CPU code, triple chevron notation and cudaLaunchKernelEx. Triple chevron notation, the most common way of launching kernels".

The ceiling division. The guide teaches the integer form, not the book's floating-point form:

- "The number of thread blocks which are needed can be calculated as the ceiling of the number of threads needed ... divided by the number of threads per block. ... A common way of expressing this as a single integer division is given below." `int blocks = (vectorLength + threads-1)/threads;`
- "The CUDA Core Compute Library (CCCL) provides a convenient utility, cuda::ceil_div ... available by including the header <cuda/cmath>." `int blocks = cuda::ceil_div(vectorLength, threads);` The header documents preconditions: "\p __a must be non-negative", "\p __b must be positive".
- "The choice of 256 threads per block here is arbitrary, but this is quite often a good value to start with."
- "Launching extra threads in a block that do no work does not incur a large overhead cost, however launching thread blocks in which no threads do work should be avoided."

Notes for the launch calculator:

- The book's `ceil(n/256.0)` passes a `double` where a `dim3` is expected. No NVIDIA page I read shows that form. It compiles through an implicit conversion to the unsigned component type; I did not find that conversion documented, so present the integer form `(n + 255) / 256` as the one the vendor documents and the book's form as the book's.
- The integer form matches the book's own Fig. 2.15, `(N + 128 - 1)/128`.
- Could not find: what CUDA does with a grid dimension of zero (what `ceil(n/256)` with integer `n < 256` produces). The docs describe failure when a dimension "exceeds the maximum"; they do not mention zero. HIP documents it (B4).
- Grid limits (2^31 - 1 blocks in x) are in ch05 A5.

Sources:

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.3)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (sections 2.1.2.2, 2.1.2.3.1)
- https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__cmath/ceil_div.h

### A9. Built-in variables `threadIdx`, `blockIdx`, `blockDim`

**Verdict: CONFIRMED.** Names, three components each, zero-based indices.

Quotes:

- Guide 5.4.2.2: "dim3 gridDim", "dim3 blockDim", "uint3 blockIdx", "uint3 threadIdx", "int warpSize : A run-time value defined as the number of threads in a warp, commonly 32."
- "Both dim3 and uint3 are trivial structures consisting of three unsigned values named x, y, and z. In C++11 and later, the default value of all components of dim3 is 1."
- Guide 2.1.2.3: "threadIdx.x will take on values from 0 up to and including blockDim.x-1" and "blockIdx.x will have values from 0 up to and including gridDim.x-1".
- The guide's index line: `int workIndex = threadIdx.x + blockDim.x * blockIdx.x;` and "This computation of workIndex is very common for 1-dimensional parallelizations."
- The guide's guard: `if(workIndex < vectorLength)`. "When workIndex exceeds vectorLength, threads exit and do not do any work."

The components are unsigned. Storing the index in `int` (as the book and the guide both do) is a conversion; the book's footnote 4 on `int` versus `unsigned int` is about exactly this choice.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.2.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (section 2.1.2.3)

### A10. "Up to 1024 threads per block on current systems"

**Verdict: CONFIRMED.** See ch05 A5 for the table (1024 total; 1024 x 1024 x 64 per dimension; all listed compute capabilities). The v13.5 tutorial repeats it with the reason: "There is a limit to the number of threads per block, since all threads of a block reside on the same streaming multiprocessor(SM) and must share the resources of the SM. On current GPUs, a thread block may contain up to 1024 threads."

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (section 2.1.2.2.1)

### A11. Block size "a multiple of 32 for hardware efficiency reasons"

**Verdict: CONFIRMED.** The reason the book defers is the warp, and the guide states the rule as a recommendation, not a requirement.

Quote (guide 1.2.2.2): "One implication of warp execution is that thread blocks are best specified to have a total number of threads which is a multiple of 32. It is legal to use any number of threads, but when the total is not a multiple of 32, the last warp of the thread block will have some lanes that are unused throughout execution."

Warp size 32 for every listed compute capability is in ch05 A5. The AMD and Apple counterparts are in ch05 B5 and ch05 C3: 32 is not the right number everywhere.

- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html

### A12. "NVIDIA Hopper H100 comes with 80 GB or 94 GB of global memory"

**Verdict: CONFIRMED.** These are two products, not two options of one card. NVIDIA's specification table has columns "H100 SXM" and "H100 NVL" with row "GPU Memory": "80GB" and "94GB". The H100 NVL is the PCIe card; NVIDIA describes a bridged pair as having "188GB HBM3 memory".

Under the project rule this figure may appear only as the book's example.

- https://www.nvidia.com/en-us/data-center/h100/ (Product Specifications table)

### A13. Dynamic parallelism "starting with the Kepler architecture"

**Verdict: PARTLY.** The feature exists and is current. The architecture name could not be confirmed from a current page, and the version the book's readers will meet has changed.

- Archived 13.0 guide: "Dynamic Parallelism is only supported by devices of compute capability 3.5 and higher." It does not say "Kepler" in that sentence, and I did not look up which compute capability numbers NVIDIA assigns to Kepler.
- Current guide 4.21: "This section documents the updated version of CUDA Dynamic Parallelism, sometimes called CDP2, which is the default in CUDA 12.0 and later. CDP2 is the only version of CUDA dynamic parallelism available on devices of CC 9.0 and higher." and "CDP1 is slated for removal from a future version of CUDA."
- The current guide's capability tables start at compute capability 7.5 (ch05 A5), so the "starting with" clause is of historical interest only.
- HIP: not supported (ch05 B4). Metal: none documented (C7).

Sources:

- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html
- https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html (section 13.1.1)

### A14. Compilation: NVCC, PTX, run-time compilation (the book's Fig. 2.14)

**Verdict: PARTLY.** `nvcc` and the host/device split are confirmed. Two parts of the book's picture are loose: device code is normally compiled all the way to machine code at build time, and the run-time compiler is in the driver, not in NVCC.

Quotes:

- Guide 2.7: "The NVIDIA CUDA Compiler nvcc is a toolchain from NVIDIA for compiling CUDA C/C++ as well as PTX code." and "nvcc drives offline compilation of CUDA code, in contrast to online or Just-in-Time (JIT) compilation driven by the CUDA runtime compiler nvrtc."
- Guide 2.7.2: "In the initial phase, nvcc separates the device code from the host code and dispatches their compilation to the GPU and the host compilers, respectively." Then "The GPU compiler compiles the C/C++ device code to PTX assembly code." and "Individual PTX code is then passed to the ptxas tool, which generates Cubin for the target hardware ISAs."
- Guide 2.7.3.1: "By default, nvcc generates PTX and Cubin for the earliest GPU architecture (lowest compute_XY and sm_XY version) supported by the CUDA Toolkit to maximize compatibility."
- Guide 1.3.4.3: "PTX code loaded by an application at runtime is compiled to binary code by the device driver. This is called just-in-time (JIT) compilation." and "it automatically caches a copy of the generated binary code".
- Guide 1.3.4: "A fatbin can contain both cubin GPU binary code and PTX virtual ISA code. PTX code can be JIT compiled for future targets."
- File extensions (guide 2.7.1): ".cu for files that contain device code or a mix of host and device code"; "Headers containing device code typically adopt the .cuh extension".
- Basic command: `nvcc <source_file>.cu -o <output_file>`. `-arch=native` "automatically detects and generates Cubin for the current GPU".
- Host compiler: "nvcc requires a compatible host compiler to be available"; choose it with `-ccbin <compiler>` or `NVCC_CCBIN`.

So the accurate picture today: source goes to `nvcc`; host code goes to the host C++ compiler; device code goes to PTX, then through `ptxas` to a cubin, and both PTX and cubin are packed in a fatbin inside the executable. The driver JIT-compiles the PTX only when no suitable cubin is present. "Virtual binary files" is the book's phrase; NVIDIA's is "PTX virtual ISA code" or "PTX assembly code".

Sources:

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html (section 1.3.4)
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html

### A15. Footnote 3: kernels reaching host memory "through Unified Virtual Address (UVA)"

**Verdict: PARTLY.** UVA exists, but in NVIDIA's current wording it is about addresses, not access. What lets a kernel touch host memory is "mapped memory" or unified (managed) memory. The lower-bandwidth caveat is confirmed.

Quotes:

- Guide 2.6.1: "A single virtual address space is used for all host memory and all global memory on all GPUs in the system within a single OS process." Consequences listed: "The location of any memory ... can be determined from the value of a pointer using cudaPointerGetAttributes()" and "The cudaMemcpyKind parameter of cudaMemcpy*() can be set to cudaMemcpyDefault".
- Guide 2.6 overview: "Mapped memory - A mechanism (different from unified memory) for accessing host memory directly from a kernel".
- Guide 2.6.3.1: "Host memory allocated with cudaMallocHost or cudaHostAlloc is automatically mapped. The pointers returned by these APIs can be directly used in kernel code to access the memory on the host. The host memory is accessed over the CPU-GPU interconnect." and "These operations have higher latency and lower bandwidth compared to accessing device memory."
- Guide 1.2.3.4, footnote: "mapped memory is not a performant replacement to unified memory or placing data in the appropriate memory space."

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html (section 1.2.3.4)

### A16. The "more advanced allocation functions" the book defers (p. 29)

**Verdict: CONFIRMED.** The book does not name them; the current docs do, and they match its description (faster copies, or no explicit copy).

| Function | Signature | What it is |
|---|---|---|
| `cudaMallocHost` | `cudaError_t cudaMallocHost(void **ptr, size_t size)` | "Allocates size bytes of host memory that is page-locked and accessible to the device. ... it can be read or written with much higher bandwidth than pageable memory obtained with functions such as ::malloc()." Freed with `cudaFreeHost`. |
| `cudaHostAlloc` | `cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)` | Same, with flags (`cudaHostAllocDefault`, `cudaHostAllocPortable`, `cudaHostAllocMapped`, `cudaHostAllocWriteCombined`). |
| `cudaMallocManaged` | `cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags = 0x01)` | "Allocates memory that will be automatically managed by the Unified Memory system." "The pointer is valid on the CPU and on all GPUs in the system that support managed memory." Freed with `cudaFree`. |

Quotes worth teaching:

- Guide 2.1.3.2: "In general, it is good practice to use page-locked memory for CPU buffers that will be used in data transfers to and from GPUs." and "Performance can degrade on some systems if too much host memory is page-locked."
- Guide 2.1.3.1: "Unified memory is a feature of the CUDA runtime which lets the NVIDIA Driver manage movement of data between host and device(s)." The guide presents the unified-memory version of `vecAdd` first and the explicit version second.
- Guide 1.2.3.4: "Even with unified memory, optimal performance is attained by keeping the migration of memory to a minimum".

Important for the portability lesson: CUDA unified memory is not the same thing as Apple's unified memory. In CUDA the driver "relocates the data to the correct place when needed" (pages migrate between two physical memories). On Apple silicon there is one physical memory (ch05 C6).

- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (section 2.1.3)

### A17. Host and device have separate memory; do not dereference a device pointer on the host

**Verdict: CONFIRMED** for CUDA, with the book's own hedge ("often hardware cards") matching NVIDIA's.

Quotes:

- Guide 1.2.1: "The CPU and the memory directly connected to it are called the host and host memory, respectively. A GPU and the memory directly connected to it are referred to as the device and device memory, respectively. In some system-on-chip (SoC) systems, these may be part of a single package."
- Guide 1.2.3.1: "From the perspective of device code, the DRAM attached to the GPU is called global memory, because it is accessible to all SMs in the GPU. This terminology does not mean it is necessarily accessible everywhere within the system."
- Guide 1.2.3.4: "When an application allocates memory explicitly on the GPU or CPU, that memory is only accessible to code running on that device. That is, CPU memory can only be accessed from CPU code, and GPU memory can only be accessed from kernels running on the GPU".

The docs state the rule. They do not say what happens when host code breaks it (the book says "exceptions or other run-time errors"); I found no NVIDIA sentence describing the failure. HIP documents different behaviour (B7), and on Metal the rule does not apply to shared buffers (C3).

- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html

### A18. No ordering between blocks; "one or two blocks" on a small GPU, "128 or 256" on a large one

**Verdict: PARTLY.** The ordering rule and the scalability argument are confirmed. The block counts are illustrative; no source gives them.

Quotes (guide 1.2.2.1):

- "There is no guarantee of scheduling between thread blocks, so a thread block cannot rely on results from other thread blocks".
- "the CUDA programming model requires that it be possible to execute thread blocks in any order, in parallel or in series."
- "The CUDA programming model enables arbitrarily large grids to run on GPUs of any size, whether it has only one SM or thousands of SMs."
- "A grid may consist of millions of thread blocks, while the GPU executing the grid may have only tens or hundreds of SMs."

How many blocks a device runs at once depends on the device and on the kernel. The queryable inputs are `multiProcessorCount` and `maxBlocksPerMultiProcessor` (ch05 A1). Do not print the book's 1, 2, 128, 256 as facts about any device.

- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html

### A19. "CUDA threads take very few clock cycles to generate and schedule", CPU threads "thousands"

**Verdict: COULD NOT VERIFY.** I found no cycle counts for thread creation in the v13.5 programming model, platform, or kernel chapters. The nearest statement is qualitative: "GPU threads operate similarly to threads on CPUs, though there are some differences important to both correctness and performance". Present the claim as the book's.

- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html

### A20. "A single-precision float is 4 bytes"

**Verdict: CONFIRMED** on all three backends, so `n * sizeof(float)` ports unchanged.

- CUDA guide 5.6.1 "Floating-Point Computation": "32-bit, also known as single-precision, corresponding to the float data type in C, C++, and CUDA."
- MSL spec Table 2.1: "float: A 32-bit floating-point. The float data type must conform to the IEEE 754 single precision storage format." Table 2.2 (size and alignment of scalar data types): `float` 4 and 4.
- HIP compiles the same C++ `float` with Clang; I did not look for a separate HIP statement.

Storage format is the same; arithmetic is not guaranteed identical on Metal (ch05 C7), which is why labs compare against a CPU reference with a tolerance.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 2.1)

### A21. "CUDA is currently the most mature framework for massively parallel computing"

**Verdict: COULD NOT VERIFY.** This is an opinion, not a documented fact. Attribute it to the book or leave it out.

---

## Group B: HIP / ROCm

Release and toolchain facts (ROCm 10.2.0, `amdclang++`, `.hip`, CMake, HIPIFY) are ch05 B1, B6, B7, B8. Launch syntax and qualifier spelling are ch05 B3 and B4.

### B1. `hipMalloc`, `hipFree`, `hipMemcpy` and the copy kinds

**Verdict: CONFIRMED.** Same names with `hip` for `cuda`, same parameter order, same enum values.

From `projects/hip/include/hip/hip_runtime_api.h` and `driver_types.h` at tag `therock-10.2`:

```c
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);

typedef enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,            ///< Host-to-Host Copy
  hipMemcpyHostToDevice = 1,          ///< Host-to-Device Copy
  hipMemcpyDeviceToHost = 2,          ///< Device-to-Host Copy
  hipMemcpyDeviceToDevice = 3,        ///< Device-to-Device Copy
  hipMemcpyDefault = 4,               ///< Runtime will automatically determine
                                      ///< copy-kind based on virtual addresses.
  hipMemcpyDeviceToDeviceNoCU = 1024  ///< Device-to-Device Copy without using compute units
} hipMemcpyKind;
```

Typed wrapper, same idea as CUDA's (A2): `template <class T> static inline hipError_t hipMalloc(T** devPtr, size_t size) { return hipMalloc((void**)devPtr, size); }`, documented as "Perform automatic type conversion to eliminate the need for excessive typecasting (ie void**)". AMD's docs use it: `HIP_CHECK(hipMalloc(&device_input, size_bytes));` with `int *device_input`.

Documented differences from CUDA:

- `hipMalloc` with size 0: "If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned." (CUDA's reference does not state the size-0 case for `cudaMalloc`.)
- `hipMalloc` returns: "#hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)". The CUDA-style name is kept as a deprecated alias with the same value: `hipErrorOutOfMemory = 2`, `hipErrorMemoryAllocation = 2` under "// Deprecated".
- `hipFree`: "This API performs an implicit hipDeviceSynchronize() call. If pointer is NULL, the hip runtime is initialized and hipSuccess is returned." CUDA says only "may perform implicit synchronization". An invalid pointer returns `hipErrorInvalidDevicePointer` (CUDA: `cudaErrorInvalidValue`).
- `hipMemcpy`: "The src and dst must not overlap." and "Calling hipMemcpy with dst and src pointers that do not match the hipMemcpyKind results in undefined behavior."
- Whether `hipMalloc` clears memory: not stated in the header. Assume it does not.

The explicit-copy host stub from AMD's own docs has exactly the book's shape: `hipMalloc`, `hipMemcpy(..., hipMemcpyHostToDevice)`, kernel, `hipMemcpy(..., hipMemcpyDeviceToHost)`, `hipFree`.

Sources:

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 4148, 5196, 5255, 11264)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/driver_types.h (line 357)
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/device_memory.html

### B2. Error checking in HIP

**Verdict: CONFIRMED.** `hipError_t`, `hipSuccess`, `hipGetErrorString`, `hipGetErrorName`, `hipGetLastError`, `hipPeekAtLastError` all exist with CUDA's meaning.

Quotes:

- Docs: "Every HIP runtime function, apart from launching kernels, has hipError_t as return type. hipGetLastError() and hipPeekAtLastError() can be used for catching errors from kernel launches, as kernel launches don't return an error directly."
- "hipGetLastError() returns and resets that error to hipSuccess, while hipPeekAtLastError() just returns the error without changing it."
- AMD's macro: `#define HIP_CHECK(expression) { const hipError_t status = expression; if(status != hipSuccess){ std::cerr << "HIP error " << status << ": " << hipGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; } }`
- Header: `const char* hipGetErrorString(hipError_t hipError);`, `hipError_t hipGetLastError(void);`, `hipError_t hipPeekAtLastError(void);`

Differences that matter:

- Ignoring a return value warns. Header: "Ignoring error-code return values from hip APIs is discouraged. On C++17, we can make that yield a warning" and `#define __HIP_NODISCARD [[nodiscard]]` applied to `enum hipError_t` when `__cplusplus >= 201703L`. The book's unchecked listings compile on CUDA without comment and draw warnings under HIP with C++17. One more reason for the labs to check every call.
- `hipGetLastError` changed in ROCm 7.0: "Prior to ROCm 7.0, hipGetLastError might also return hipSuccess or hipErrorNotReady from the last HIP runtime API call, which are not errors." The 7.0 behaviour "matches CUDA behavior".

Sources:

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/error_handling.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/hip-7-changes.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 90 to 113, 3091 to 3134)

### B3. Asynchronous launch and synchronisation in HIP

**Verdict: CONFIRMED.** Same model as CUDA: the launch returns, `hipDeviceSynchronize()` waits.

Quotes:

- "Computations launched on the device execute asynchronously with respect to the host, and it is the user's responsibility to synchronize their data dispatch/fetch with computations on the device."
- "HIP performs implicit synchronization on occasions, unlike some APIs where the responsibility for synchronization is left to the user."
- "In the HIP runtime, kernels run on the default stream when one is not specified". "Within the same stream, operations run in the order they were issued".
- Header, `hipDeviceSynchronize`: "Waits on all active streams on current device".
- AMD's "typical sequence of operations" for a host program is the book's host stub: initialise, "allocating the required memory on the host and device, preparing input data and transferring it from the host to the device", launch "using the triple chevron syntax or appropriate API call", synchronise, handle errors, clean up.

Not found: a HIP sentence saying `hipMemcpy` blocks until the copy is complete. The docs call explicit copies "long blocking operations" in a performance tip, which implies it; time HIP copies the same way as CUDA copies (A5).

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (line 2442)

### B4. Kernel, qualifiers, built-ins and launch in HIP

**Verdict: CONFIRMED.** Identical spelling for everything this chapter's kernel uses (ch05 B3, B4). Additional facts for chapter 2:

- `__host__ __device__` together is supported. To tell the two compilations apart HIP offers its own macro: "The __HIP_DEVICE_COMPILE__ macro is defined when the compiler is compiling for the device. This macro can be used to replace the __CUDA_ARCH__ macro when porting from CUDA." Mechanism: "amdclang++ makes multiple passes over the code: one pass for the host code, and for the device code one pass for each GPU architecture to be compiled for."
- The header is required: built-ins "are not implicitly defined by the compiler" (ch05 B4), and "HIP-Clang does not include default headers, and instead you must explicitly include all required files." Use `#include <hip/hip_runtime.h>`.
- `dim3` in HIP: `constexpr __host__ __device__ dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1)`. A plain integer block count converts, as in CUDA.
- Kernel arguments: the HIP header rejects reference parameters at compile time ("A __global__ function cannot have a reference as one of its arguments.").
- Launch checks, documented in the HIP 7.0 change list: "if any input global work size dimension is zero, returns hipErrorInvalidValue" and "If the total number of threads per block exceeds the maximum work group limit during a kernel launch, the return value is changed to hipErrorInvalidConfiguration from hipErrorInvalidValue". The first is the integer-division bug for `n` smaller than the block size (zero blocks); CUDA's docs do not state that case (A8).
- AMD's own kernel in the error-handling example: `const std::size_t index = threadIdx.x + blockDim.x * blockIdx.x; if(index < size)`.

Sources:

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/hip-7-changes.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (line 1443)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h (lines 177 to 187)

### B5. Threads-per-block limit on AMD

**Verdict: CONFIRMED** as a typical value, not a guarantee.

Quote: "The size of a block, or the block dimension, is the user-configurable number of threads per block, but is limited by the queryable capabilities of the executing hardware. Common limits include: Maximum threads per block: typically 1024. Maximum block dimensions: 1024 x 1024 x 64 (x, y, z). Limited by available resources (registers, LDS, warp slots)".

Query `maxThreadsPerBlock` and `maxThreadsDim` (ch05 B2). The "multiple of 32" advice becomes "multiple of the warp size", which is 64 on some AMD devices (ch05 B5).

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html

### B6. Compilation flow in HIP (the counterpart of the book's Fig. 2.14)

**Verdict: CONFIRMED,** and it differs from NVCC's. There is one Clang-based compiler for both halves, device code is compiled ahead of time for named GPU architectures, and no portable intermediate form like PTX is shipped by default.

Quotes:

- "Offline compilation is performed in two steps: host and device code compilation. Host-code compilation: On the host side, amdclang++ or hipcc can compile the host code in one step without other C++ compilers. Device-code compilation: The compiled device code is embedded into the host object file."
- "Instructions in the AMDGPU instruction set architecture (ISA) are compatible only with certain physical GPU architectures." Targets are named by GFX IP, for example "gfx942 - CDNA3 (MI300 series)", and chosen with `amdclang++ --offload-arch=gfx942 kernel.cpp -o kernel.out`.
- Run-time compilation exists as a separate API, not as the default path: "HIP allows you to compile kernels at runtime using the hiprtc* API. Kernels are stored as a text string".

The driver, flags and `.hip` extension are in ch05 B6; `--offload-arch=native` targets the build machine's GPU. Contrast with A14: a CUDA binary can carry PTX that the driver compiles for a GPU that did not exist at build time; a HIP binary runs on the architectures it was built for.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html

### B7. Device memory seen from the host; unified and managed memory in HIP

**Verdict: CONFIRMED** as documented. This is a real difference from CUDA, and it is hardware- and configuration-dependent.

AMD's comparison tables (unified memory page):

| Call | Allocation origin | Access outside the origin |
|---|---|---|
| `hipMalloc()` | device | "zero copy" |
| `cudaMalloc()` | device | "not accessible on host" |
| `hipHostMalloc()` / `cudaMallocHost()` | pinned host | "zero copy" |
| `hipMallocManaged()` (without HMM or with `HSA_XNACK=0`) | pinned host | "zero copy" |
| `hipMallocManaged()` (with HMM and `HSA_XNACK=1`) | first touch | "page-fault migration" |

Quotes:

- "CPUs can access memory allocated via hipMalloc(), providing bidirectional memory accessibility within the unified address space."
- Footnote: "Zero copy is a feature, where the memory is pinned to either the device or the host, and won't be transferred when accessed by another device or the host. Instead only the requested memory is transferred".
- `hipMalloc()`: "Device-local allocation; zero-copy access from host; no automatic migration; best performance for GPU-intensive workloads".
- "For maximum performance with known access patterns, explicit memory management with hipMalloc() and hipMemcpy() typically provides the best performance."
- Device memory page: "accesses from the host or other devices have to go over a special interface which is considerably slower, usually the PCIe bus or the AMD Infinity Fabric." and "On certain architectures like APUs, the GPU and CPU share the same physical memory."
- `hipMallocManaged` header comment: "If HMM is not supported, the function behaves the same as @p hipMallocHost." Requirements: "it is essential to set the environment variable HSA_XNACK=1 and use a GPU kernel mode driver that supports HMM."

For the course: keep the book's rule (pass device pointers only to API calls and kernels). It is what CUDA requires and it is the fast path on AMD too. The lesson can say truthfully that on AMD the docs describe a host dereference of `hipMalloc` memory as a slow zero-copy access, not as an error. Not checked by running code, and I did not find which GPUs or driver settings the "zero copy" row requires.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/device_memory.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 4262 to 4282)

### B8. Does HIP still target NVIDIA GPUs?

**Verdict: COULD NOT VERIFY.** The sources disagree and I found no statement that settles it.

- HIP FAQ (10.2.0 docs): "What hardware does HIP support? HIP supports AMD GPUs."
- The header at `therock-10.2` still has both branches: `#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)` followed by `#include <hip/nvidia_detail/nvidia_hip_runtime.h>`.
- llama.cpp refuses the NVIDIA path (ch05 B9: "The HIP backend supports only AMD targets").

For the course: describe HIP as the AMD backend. Do not claim one HIP source also builds for NVIDIA.

- https://rocm.docs.amd.com/projects/HIP/en/latest/faq.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime.h

---

## Group C: Metal

Sources: Apple developer documentation (JSON behind each page), Metal Shading Language Specification Version 4.1 ("MSL spec"), Metal headers in the macOS 27.0 SDK on this machine. Shared storage, dispatch, thread-index attributes, thread limits, math mode and GPU timing are ch05 C3 to C7 and C9.

### C1. The kernel: what replaces `__global__`, the pointer arguments and the index formula

**Verdict: CONFIRMED.** Apple's own introductory sample is vector addition, so every step of the book's example has an Apple-written counterpart.

Apple's kernel ("Performing calculations on a GPU"):

```cpp
kernel void add_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index] + inB[index];
}
```

Quotes:

- "MSL is a variant of C++ designed for GPU programming." "The sample project defines this function in the add.metal file."
- The `kernel` keyword "declares that the function is: A public GPU function. Public functions are the only functions that your app can see. Public functions also can't be called by other shader functions. A compute function (also known as a compute kernel), which performs a parallel calculation using a grid of threads."
- "Whenever you declare a pointer in MSL, you need to supply a keyword to declare its address space. Use the device address space to declare persistent memory that the GPU can read from and write to."
- "The MSL version removes the for-loop from the C version because the GPU calls the function with multiple threads in the compute grid." and "If you want to convert similar code from C or C++ to MSL, replace the loop logic with a grid in the same way." (the book's "where did the loop go?")
- "the function takes a new index argument, with another MSL keyword, thread_position_in_grid ... This keyword declares that Metal should calculate a unique index for each thread and pass that index in this argument."
- MSL spec 5.1.3: "Functions declared with the kernel or [[kernel]] attribute must return void." Both spellings `kernel` and `[[kernel]]` ("since Metal 2.3") are valid.
- MSL spec 4: "If an address space attribute is missing on a pointer or reference type declaration, a compilation error occurs."
- MSL spec 4.1: "The device address space name refers to buffer memory objects allocated from the device memory pool that are both readable and writeable."
- Binding: "Device buffers, constant buffers ...: [[buffer(index)]]". Apple's sample omits it: "Metal automatically assigns indices for the buffer arguments in the order that the arguments appear in the function declaration ..., starting with 0."

Differences from the CUDA kernel:

- Separate language and separate file. There is no single source holding host and device code.
- No index formula. `thread_position_in_grid` is the global index (ch05 C4 gives the spec's own formula showing it equals `blockIdx * blockDim + threadIdx`).
- Apple's sample has no `if (index < n)` guard because it dispatches exactly `n` threads (C4, ch05 C5). A kernel dispatched in whole threadgroups needs the guard, as in CUDA.
- A scalar such as `n` is not a by-value kernel argument. It arrives through the buffer argument table (C3, `setBytes`).

Sources:

- https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 4, 4.1, 5.1, 5.1.3, 5.2.1)

### C2. Allocation: what replaces `cudaMalloc` and `cudaFree`

**Verdict: CONFIRMED.** Buffers are objects created by the device. There is no free call.

| Objective-C method (macOS 27.0 SDK, `MTLDevice.h`) | Apple's description |
|---|---|
| `- (nullable id <MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options;` | "Creates a buffer the method clears with zero values." Length is "The size of the new buffer, in bytes." |
| `- (nullable id <MTLBuffer>)newBufferWithBytes:(const void *)pointer length:(NSUInteger)length options:(MTLResourceOptions)options;` | "Allocates a new buffer of a given length and initializes its contents by copying existing data into it." |
| `- (nullable id <MTLBuffer>)newBufferWithBytesNoCopy:(void *)pointer length:(NSUInteger)length options:(MTLResourceOptions)options deallocator:(...)deallocator;` | "Creates a buffer that wraps an existing contiguous memory allocation." Pointer: "A page-aligned pointer to the starting memory address." Length: "results in a page-aligned region of memory." "The existing memory allocation needs to exist within a single virtual memory (VM) region." |

Other facts:

- Failure is a nil return: "A new MTLBuffer instance if the method completed successfully; otherwise nil." No error code.
- CPU access: `- (void*)contents` ("Gets the system address of the buffer's storage allocation."). Returns "A pointer to the shared copy of the buffer data, or NULL for buffers allocated with a private resource storage mode". "Private resources aren't CPU-accessible."
- Size limit: `maxBufferLength`, "The largest amount of memory, in bytes, that a GPU device can allocate to a buffer instance." "The property's value is at least 256 MB". Read the device's value; do not print one.
- Freeing: `MTLBuffer.h` and `MTLResource.h` declare no free or release method (the nearest is `setPurgeableState:`). A buffer is an Objective-C object and goes away when the last reference does. With `newBufferWithBytesNoCopy`, the `deallocator` block is "A block the framework invokes when it deallocates the buffer so that your app can release the underlying memory".
- The device itself: `MTLCreateSystemDefaultDevice()` "Returns the device instance Metal selects as the default." In a command-line tool: "in order for the system to provide a default Metal device object, you need to link to the Core Graphics framework."

Differences worth teaching: the plain allocation is zero-filled (CUDA: "The memory is not cleared."), and the byte count is the only thing Metal knows about a buffer ("Metal manages each buffer as an opaque collection of bytes").

Sources:

- https://developer.apple.com/documentation/metal/mtldevice/makebuffer(length:options:)
- https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytes:length:options:)
- https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:)
- https://developer.apple.com/documentation/metal/mtlbuffer/contents()
- https://developer.apple.com/documentation/metal/mtldevice/maxbufferlength
- https://developer.apple.com/documentation/metal/mtlcreatesystemdefaultdevice()
- Local headers: `Metal.framework/Headers/MTLDevice.h` (lines 680 to 695), `MTLBuffer.h` (line 46)

### C3. Copies: what replaces `cudaMemcpy`

**Verdict: CONFIRMED.** For a shared-mode buffer on Apple silicon nothing replaces it: the two copy steps of the CUDA host stub have no counterpart. The underlying facts are ch05 C6. What chapter 2 adds:

- Apple's sample fills the inputs through the pointer, `float* dataPtr = buffer.contents;` followed by ordinary writes, and reads the result the same way after the wait: `float* result = _mBufferResult.contents;`.
- "The sample app uses shared memory (storageModeShared), which both the CPU and GPU can access."
- "After the command buffer completes, the GPU's calculations are stored in the output buffer and Metal performs any necessary steps to make sure the CPU can see them."
- The ordering rule still applies. Header comment on `MTLBuffer`: "Unlike in OpenGL and OpenCL, access to buffers is not synchronized. The caller may use the CPU to modify the data at any time but is also responsible for ensuring synchronization and coherency. The contents become undefined if both the CPU and GPU write to the same buffer without a synchronizing action between those writes. This is true even when the regions written do not overlap."

Where a copy does still exist in Metal:

- `newBufferWithBytes:length:options:` copies once at creation. Wrapping host arrays that the caller already owns (the book's `vecAdd(A, B, C, n)` interface) therefore costs a copy unless the arrays are page-aligned and wrapped with `newBufferWithBytesNoCopy`.
- Private storage: "because private resources aren't accessible to the CPU, you can't populate them with it. To write data from the CPU to a private resource, you need to first write the data to a shared or managed resource. You can then copy the data from that resource to the private resource." The copy is a blit command: `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:` (Swift `copy(from:sourceOffset:to:destinationOffset:size:)`), "Encodes a command that copies data from one buffer into another." On macOS offsets and size must be multiples of 4.
- Small by-value arguments: `setBytes:length:atIndex:`, "Copies data directly to the GPU to populate an entry in the buffer argument table." "This method only works for data smaller than 4 kilobytes that doesn't persist." This is how a scalar like `n` reaches a Metal kernel.

A consequence for the lab's "copy time versus kernel time" measurement: on Metal with shared buffers there may be no copy to time at all. Report that as the finding. Do not invent a copy to make the columns match.

Sources:

- https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu
- https://developer.apple.com/documentation/metal/copying-data-to-a-private-resource
- https://developer.apple.com/documentation/metal/mtlblitcommandencoder/copy(from:sourceoffset:to:destinationoffset:size:)
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbytes(_:length:index:)
- Local header: `Metal.framework/Headers/MTLBuffer.h` (lines 26 to 31)

### C4. The launch: what replaces `kernel<<<blocks, threads>>>(args)`

**Verdict: CONFIRMED.** One CUDA statement becomes a sequence of objects and calls. Apple's sample, in order:

| Step | Apple's code | CUDA counterpart |
|---|---|---|
| Compile the function for this GPU | `[_mDevice newComputePipelineStateWithFunction: addFunction error:&error]` | done by `nvcc` and the driver |
| Queue | `[_mDevice newCommandQueue]` | the default stream |
| Command buffer | `[_mCommandQueue commandBuffer]` | none |
| Encoder | `[commandBuffer computeCommandEncoder]` | none |
| Choose the kernel | `[computeEncoder setComputePipelineState:_mAddFunctionPSO]` | the function name |
| Arguments | `[computeEncoder setBuffer:_mBufferA offset:0 atIndex:0]` ... | the argument list |
| Grid | `MTLSize gridSize = MTLSizeMake(arrayLength, 1, 1);` | blocks x threads |
| Threads per group | `_mAddFunctionPSO.maxTotalThreadsPerThreadgroup`, shrunk if larger than the data | the block size |
| Launch | `[computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize]` | `<<< >>>` |
| Close and submit | `[computeEncoder endEncoding];` `[commandBuffer commit];` | implicit |
| Wait | `[commandBuffer waitUntilCompleted];` | `cudaDeviceSynchronize()` |

Quotes:

- "Metal subdivides the grid into smaller grids called threadgroups. Each threadgroup is calculated separately. Metal can dispatch threadgroups to different processing elements on the GPU".
- "The command queue created the command buffer, so committing the buffer always places it on that queue. After you commit the command buffer, Metal asynchronously prepares the commands for execution and then schedules the command buffer to execute on the GPU."
- "Your app can do other work while the GPU is processing your commands."
- `waitUntilCompleted`: "Blocks the current thread until the GPU finishes executing the command buffer and all of its completion handlers."
- `commit`: "You can commit a command buffer to its command queue only one time." "You can't encode additional commands to a command buffer after you commit it."
- Pipeline creation: "When you create a pipeline state object, the device object finishes compiling the function for this specific GPU. ... Because compiling does take a while, avoid creating pipeline state objects synchronously in performance-sensitive code." For a benchmark this belongs outside the timed region.
- "You can follow the same steps using the encoder to encode multiple compute commands into the compute pass".

No block-count arithmetic is needed with `dispatchThreads` (ch05 C5: Metal makes partial threadgroups). With `dispatchThreadgroups` the ceiling division and the kernel guard come back, exactly as in CUDA.

Sources:

- https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu
- https://developer.apple.com/documentation/metal/mtlcommandbuffer/commit()
- https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted()
- Local headers: `MTLComputeCommandEncoder.h` (lines 59, 71, 228, 243), `MTLCommandBuffer.h` (lines 306, 354)

### C5. Error checking in Metal

**Verdict: CONFIRMED.** There is no per-call error code and no last-error state. Errors arrive in three ways.

- A nil return from a creation method: `newBufferWithLength:options:`, `newCommandQueue`, `newDefaultLibrary`, `newFunctionWithName:` (Apple's sample checks each with `if (x == nil)`).
- An `NSError` out-parameter where compilation is involved: `newLibraryWithSource:(NSString *)source options:(nullable MTLCompileOptions *)options error:(__autoreleasing NSError **)error` and `newComputePipelineStateWithFunction:(id <MTLFunction>)computeFunction error:(__autoreleasing NSError **)error`. "otherwise Swift throws an error and Objective-C returns nil."
- The command buffer after it finishes: `@property (readonly) MTLCommandBufferStatus status;` and `@property (nullable, readonly) NSError *error;`. `MTLCommandBufferStatusCompleted = 4` is "A command buffer's successful, final state"; `MTLCommandBufferStatusError = 5` is "A command buffer's unsuccessful, final state, which indicates the GPU stopped running the buffer's commands because of a runtime issue." `error` "remains nil unless the GPU can't successfully run the command buffer."

Like CUDA, the failure of the kernel itself is only visible after the wait. Unlike CUDA, encoder calls such as `setBuffer:offset:atIndex:` and `dispatchThreads:` return `void`.

Out-of-bounds access: not reported by default as far as Apple documents. The tool is Shader Validation: "Metal Shader Validation detects errors only discoverable during shader execution, such as accesses to non-resident resources, out-of-bounds memory accesses". Listed issue: "Out-of-bounds memory access: Loading or storing data outside the bounds of a buffer you pass from the host to the GPU". Enable outside Xcode with `MTL_SHADER_VALIDATION=1`, and `MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1` to see messages on standard error. It "has a corresponding impact on GPU performance". I did not run it; whether it catches a specific missing-guard case in a lab is unverified.

Sources:

- https://developer.apple.com/documentation/metal/mtlcommandbuffer/status
- https://developer.apple.com/documentation/metal/mtlcommandbuffer/error
- https://developer.apple.com/documentation/metal/mtlcommandbufferstatus/error
- https://developer.apple.com/documentation/metal/mtldevice/makelibrary(source:options:)
- https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage
- Local headers: `MTLDevice.h` (lines 787, 835), `MTLCommandBuffer.h` (lines 56 to 63, 360, 366)

### C6. Compilation in Metal (the counterpart of the book's Fig. 2.14)

**Verdict: CONFIRMED.** Two documented routes, both ending with a per-GPU compile at pipeline creation.

- Build time: "The metal compiler tool converts each shader source file into an intermediate representation file." `xcrun -sdk macosx metal -o Shadow.ir -c Shadow.metal`, then `xcrun -sdk macosx metal -o LightsAndShadow.metallib Lights.metalar Shadow.ir`. "The command produces a Metal library that your app can load at runtime." In Xcode: "Xcode compiles all the Metal source files (ending in .metal) in an Xcode project into a single default library", loaded with `newDefaultLibrary`.
- Run time: `newLibraryWithSource:options:error:`, "Synchronously creates a Metal library instance by compiling the functions in a source string." Limitation: "the source may only import the Metal default library."
- MSL spec 1.6: "You can use the Metal compiler online (with the appropriate APIs to compile Metal sources) or offline."
- Final step in both routes: "The function object is a proxy for the MSL function, but it's not executable code. You convert the function into executable code by creating a pipeline." and "the device object finishes compiling the function for this specific GPU."

So the shape is the same three stages as CUDA (source, portable intermediate, device-specific code at run time), with the intermediate being a `.metallib` and the final stage always happening. Which route llama.cpp and MLX take is ch05 C10; the course's lab runtime compiles `.metal` source at run time.

- https://developer.apple.com/documentation/metal/building-a-shader-library-by-precompiling-source-files
- https://developer.apple.com/documentation/metal/mtldevice/makedefaultlibrary()
- https://developer.apple.com/documentation/metal/mtldevice/makelibrary(source:options:)
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 1.6)

### C7. CUDA features of this chapter with no documented Metal equivalent

**Verdict: CONFIRMED** in the sense that I looked and the sources document none.

- `__device__` and `__host__`: no counterpart. Every function in a `.metal` file is GPU code; only functions with an attribute such as `kernel` are visible to the app. "Kernel, vertex, and fragment functions can't call one another without triggering a compilation error" (MSL spec 5.1). An unqualified MSL function plays the role of a `__device__` helper.
- `__host__ __device__` (one function compiled for both sides): none documented. MSL "is a C++17-based specification with extensions and restrictions" (Metal 4 and later; C++14 before), and "Do not use the C++ standard library in Metal code." What projects do share between host and shader is a plain header of constants and structs (D2).
- Dynamic parallelism (a kernel launching a kernel): none documented. The MSL spec has no dispatch call, and kernels cannot call kernels.
- A last-error state, an error-string function, a copy-kind enum: none (C3, C5).
- `cudaMemcpyDefault` and pointer-based location queries: no counterpart needed; there is one address for a shared buffer.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 1.5, 1.5.4, 5.1)

### C8. Metal 4 has a second set of command types

**Verdict: CONFIRMED.** The calls in C4 are the original Metal API, which remains available. macOS 26.0 and iOS 26.0 added parallel types: "Metal 4 introduces several types with the MTL4 prefix that are completely independent from the original MTL types they replace, such as MTL4CommandQueue versus MTLCommandQueue." `MTL4ComputeCommandEncoder` "combines compute dispatch commands, blit commands, and acceleration structure commands into a single pass" and has `dispatchThreads(threadsPerGrid:threadsPerThreadgroup:)`. "You can incrementally adopt Metal 4 over time".

For the course: teach the original `MTLCommandQueue` / `MTLCommandBuffer` / `MTLComputeCommandEncoder` path (it is what Apple's sample, llama.cpp, MLX and PyTorch use in the files read for Group D) and mention that a newer family exists. I did not check whether Apple recommends one over the other for compute-only work.

- https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api
- https://developer.apple.com/documentation/metal/mtl4computecommandencoder

---

## Group D: how llama.cpp, MLX and PyTorch do the same things

All files read on 2026-11-20 at the commits named at the top. Line numbers are for those commits and will drift.

### D1. llama.cpp, CUDA and HIP backend

**Verdict: CONFIRMED** (read from source).

- Allocation (`ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml_cuda_device_malloc`, line 200): `err = cudaMalloc(ptr, size);`, or `cudaMallocManaged(ptr, size)` when unified memory is enabled, with a HIP-only fallback: "fall back to cudaMalloc if not supported (e.g. on Windows)". On failure it logs "cudaMalloc failed: %s" with `cudaGetErrorString(err)` (line 1289).
- Host to device (`ggml_backend_cuda_buffer_set_tensor`, line 1079): `CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));` then `CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));`. Device to host is the mirror image with `cudaMemcpyDeviceToHost`. So production code uses the async copy plus an explicit stream wait, not the book's blocking `cudaMemcpy`.
- Error macro (`ggml/src/ggml-cuda/common.cuh`, line 206): `#define CUDA_CHECK_GEN(err, success, error_fn) do { auto err_ = (err); if (err_ != (success)) { ggml_cuda_error(#err, __func__, __FILE__, __LINE__, error_fn(err_)); } } while (0)` and `#define CUDA_CHECK(err) CUDA_CHECK_GEN(err, cudaSuccess, cudaGetErrorString)`. This is the book's sidebar pattern.
- A kernel with this chapter's shape (`ggml/src/ggml-cuda/scale.cu`): `const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;` (line 13), `if (i >= k) { return; }` (line 16), body `dst[i_dst] = ggml_cuda_cast<T>(scale * ggml_cuda_cast<float>(x[i_src]) + bias);`. It is the scale-and-offset op of milestone M0 and of the book's Exercise 9.
- Block count (same file, line 40): `const int64_t num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;` with `#define CUDA_SCALE_BLOCK_SIZE 256` in `scale.cuh`. Integer ceiling division, block size 256.
- Launch: most kernels no longer write `<<< >>>` at the call site. They call `ggml_cuda_kernel_launch(kernel, launch_params, args...)` (`common.cuh`, line 1573), which ends in `kernel<<<launch_params.block_nums, launch_params.block_dims, launch_params.shmem, launch_params.stream>>>(std::forward<Args>(args)... );` or, when programmatic dependent launch is enabled, `CUDA_CHECK(cudaLaunchKernelEx(&pdl_cfg.cfg, kernel, ...))`.
- HIP: `ggml/src/ggml-cuda/vendors/hip.h` renames the API calls of this chapter one for one: `#define cudaMalloc hipMalloc`, `#define cudaFree hipFree`, `#define cudaMemcpy hipMemcpy`, `#define cudaMemcpyHostToDevice hipMemcpyHostToDevice`, `#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost`, `#define cudaError_t hipError_t`, `#define cudaSuccess hipSuccess`, `#define cudaGetErrorString hipGetErrorString`, `#define cudaDeviceSynchronize hipDeviceSynchronize`. Nothing is defined for `__global__`, the built-ins or `<<< >>>` (ch05 B9).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/scale.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/scale.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/hip.h

### D2. llama.cpp, Metal backend

**Verdict: CONFIRMED** (read from source). This is the clearest real example of the chapter's portability lesson.

- Shared buffers follow the device (`ggml/src/ggml-metal/ggml-metal-device.m`, line 1271): `dev->props.use_shared_buffers = dev->props.has_unified_memory;` where `has_unified_memory = dev->mtl_device.hasUnifiedMemory`. Environment variables `GGML_METAL_SHARED_BUFFERS_DISABLE` and `GGML_METAL_SHARED_BUFFERS_ENABLE` override it.
- Allocation (line 2206): host memory first (`ggml_metal_host_malloc`, which uses `vm_allocate`, so it is page-aligned), then `[res->dev->mtl_device newBufferWithBytesNoCopy:res->all_data length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];`. Without shared buffers: `[res->dev->mtl_device newBufferWithLength:size_aligned options:MTLResourceStorageModePrivate];`.
- "Copy to device" (`ggml_metal_buffer_set_tensor`, line 2389): `if (buf->is_shared) { memcpy((char *) tensor->data + offset, data, size); return; }`. Only the private path builds a blit (`blitCommandEncoder`, `copyFromBuffer:`) and waits. The same test guards `ggml_metal_buffer_get_tensor`.
- Dispatch (line 890): `[encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];`. llama.cpp uses whole threadgroups, so its kernels take `uint3 tgpig[[threadgroup_position_in_grid]]` and `ushort3 tpitg[[thread_position_in_threadgroup]]` (`kernels/binbcast.metal`), the CUDA-style pair, not Apple's `thread_position_in_grid`.
- Sharing definitions between host and shader: `kernels/common.h` begins `#include "ggml-metal-impl.h"`, and the host file `ggml-metal-ops.cpp` includes the same header. It holds `#define` constants and argument structs. This is the working substitute for `__host__ __device__` (C7).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/binbcast.metal
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/common.h
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-impl.h

### D3. MLX

**Verdict: CONFIRMED** (read from source).

- Metal allocation (`mlx/backend/metal/allocator.cpp`): `constexpr size_t resource_options = MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;` and `buf = device_->newBuffer(size, resource_options);` (small sizes come from a heap: `heap_->newBuffer(size, resource_options)`). Every array is a shared buffer.
- The CPU pointer of an array is the buffer's contents: `void* Buffer::raw_ptr() { ... assert(buf->storageMode() != MTL::StorageModePrivate); return buf->contents(); }`. There is no host-to-device copy step in the allocator.
- Elementwise kernels (`mlx/backend/metal/kernels/binary.h`): `template <typename T, typename U, typename Op> [[kernel]] void binary_ss(device const T* a, device const T* b, device U* c, uint index [[thread_position_in_grid]]) { c[index] = Op()(a[0], b[0]); }`. The vector forms process `N = WorkPerThread<T>::n` elements per thread (`index *= N;` then a loop with a tail guard `index + i < size`), which is the book's Exercise 2 in production.
- Dispatch (`mlx/backend/metal/binary.cpp`, line 120): `auto thread_group_size = kernel->maxTotalThreadsPerThreadgroup();` then `compute_encoder.dispatch_threads(grid_dims, group_dims);`.
- MLX also has a CUDA backend (`mlx/backend/cuda/allocator.cpp`). Its host-visible allocation prefers managed memory: `if (supports_managed_memory()) { CHECK_CUDA_ERROR(cudaMallocManaged(&data, size)); } else { CHECK_CUDA_ERROR(cudaMallocHost(&data, size)); }`; device-only allocations use `cudaMallocAsync` or `cudaMalloc`. So MLX keeps its "one pointer for both sides" model on NVIDIA by using CUDA unified memory (A16).

- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/binary.h
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/binary.cpp
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/cuda/allocator.cpp

### D4. PyTorch

**Verdict: CONFIRMED** (read from source).

- CUDA allocation (`c10/cuda/CUDACachingAllocator.cpp`, `allocPrimitive`, line 1452): `return C10_CUDA_ERROR_HANDLED(cudaMalloc(ptr, size));`. The caching allocator sits on top so that most tensor allocations do not call `cudaMalloc` at all; blocks are returned with `C10_CUDA_CHECK(cudaFree((void*)block->ptr));`.
- Error macros (`c10/cuda/CUDAException.h`): `C10_CUDA_CHECK(EXPR)` captures `const cudaError_t __err = EXPR;` and passes it with `__FILE__`, `__func__`, `__LINE__` to a checker. After a launch: `#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())`, commented "This should be used directly after every kernel launch to ensure the launch happened correctly and provide an early, close-to-source diagnostic if it didn't." This is A6's rule as project policy.
- `tensor.to("cuda")` and back (`aten/src/ATen/native/cuda/Copy.cu`, from line 793): `kind = cudaMemcpyHostToDevice;` or `kind = cudaMemcpyDeviceToHost;`, then `AT_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, stream));` when `non_blocking`, else `at::cuda::memcpy_and_sync(dst, src, nbytes, kind, stream);`. The book's two copy directions are literally what moves a tensor.
- Launches use the chevrons directly, for example `<<<grid, num_threads(), 0, stream>>>(N, f, data);` in `aten/src/ATen/native/cuda/CUDALoops.cuh` (line 429).
- MPS (`aten/src/ATen/mps/MPSAllocator.mm`): `TORCH_CHECK(m_device.hasUnifiedMemory, "MPS backend is only supported on devices with unified memory");` and the default allocator is `static MPSAllocator s_mps_shared_alloc(HeapAllocator::UsageFlags::SHARED);`, where `SHARED` selects `MTLResourceStorageModeShared` (`MPSAllocator.h`, line 170).
- A caution for the portability lesson: unified memory does not make `tensor.to("mps")` free. `aten/src/ATen/native/mps/operations/Copy.mm` still copies between a CPU tensor and an MPS tensor: the host pages are "wrapped in an autoreleased newBufferWithBytesNoCopy" and blitted (a pinned tensor's "own shared MTLBuffer" is used directly). The copy disappears only when the data is created in a shared buffer in the first place, as in Apple's sample and in MLX.

- https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp
- https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDAException.h
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Copy.cu
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/CUDALoops.cuh
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/mps/MPSAllocator.mm
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/mps/MPSAllocator.h
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Copy.mm

---

## Summary of corrections

Items where the book, or a commonly assumed fact, turned out PARTLY right, out of date, or unverifiable:

- A1: the book's reference [3] (guide Release 13.0) is now an archive. The current guide is v13.5, restructured, and its section 2.1 is a vector addition tutorial covering the same ground as this chapter.
- A2: the `(void**)` cast on `cudaMalloc` is no longer required in C++. The reference lists `template<class T> cudaMalloc(T **devPtr, size_t size)` and NVIDIA's tutorial omits the cast. HIP has the same wrapper. The cast remains valid and is the form the course emulator accepts.
- A2: `cudaMalloc` memory "is not cleared". Metal's `newBufferWithLength:options:` is zero-filled.
- A4: there is a fifth copy kind, `cudaMemcpyDefault` (= 4), which NVIDIA now recommends and uses in its tutorial. Mismatched pointers and kind are undefined behaviour, not a guaranteed error.
- A5: a kernel launch is asynchronous. The book does not say so in this chapter; its listings work because the blocking `cudaMemcpy` follows in the same in-order default stream. Timing a kernel needs `cudaDeviceSynchronize()` or events.
- A5: the "host-to-device copies of 64 KB or less are asynchronous" rule is in the archived 13.0 guide; I could not find it in v13.5.
- A6: the book's error-check pattern does not cover the launch, which returns no `cudaError_t`. NVIDIA's pattern is `cudaGetLastError()` right after the launch and a checked `cudaDeviceSynchronize()` after it. Asynchronous errors are sticky until `cudaGetLastError` clears them.
- A7: the qualifier table has grown to five rows with `__tile__` and `__tile_global__` (CUDA Tile C++, toolkit 13.3 onward). The book's three rows are unchanged.
- A8: NVIDIA documents the integer ceiling `(n + threads-1)/threads` and `cuda::ceil_div` from `<cuda/cmath>`, not the book's `ceil(n/256.0)`. What CUDA does with a zero-block grid is not documented in the pages read; HIP returns `hipErrorInvalidValue`.
- A13: "starting with Kepler" could not be confirmed by name. The archived guide says compute capability 3.5 and higher; the current guide documents only CDP2 and lists no capability below 7.5. HIP does not support dynamic parallelism; Metal documents nothing like it.
- A14: by default `nvcc` produces PTX and a compiled cubin at build time, packed in a fatbin. The run-time compiler is in the device driver, and it runs only when PTX has to be compiled for the device at hand. "A run-time component of NVCC" is not how NVIDIA describes it.
- A15: UVA is one address space for host and device allocations. Kernel access to host memory is "mapped memory" (page-locked allocations) or unified memory, at lower bandwidth as the book says.
- A18: "one or two blocks" and "128 or 256 blocks" are illustrations with no source. The any-order rule is confirmed.
- A19, A21: the thread-creation cycle counts and "most mature framework" are the book's claims; no documentation supports or refutes them.
- B1: `hipFree` always performs an implicit `hipDeviceSynchronize()`; `hipMalloc(…, 0)` succeeds with a null pointer; the out-of-memory code is named `hipErrorOutOfMemory` (the CUDA-style `hipErrorMemoryAllocation` is a deprecated alias).
- B2: under C++17, ignoring a HIP return value is a compiler warning (`[[nodiscard]]` on `hipError_t`).
- B6: HIP compiles device code ahead of time for named architectures (`--offload-arch`). It has no default PTX-like stage that a driver compiles for future GPUs.
- B7: AMD documents `hipMalloc` memory as host-accessible by "zero copy", where it lists `cudaMalloc` memory as "not accessible on host". Configuration-dependent and not tested here; the book's rule is still the right habit.
- B8: whether HIP still builds for NVIDIA GPUs could not be established. The FAQ says "HIP supports AMD GPUs."
- C1 to C4: Apple's introductory sample is vector addition with a shared buffer, no copies, no index formula and no bounds guard. It is the direct Metal counterpart of the book's Figs. 2.10 and 2.13.
- C3: on Metal a host stub that receives ordinary host arrays still has to get them into buffers, by one copy (`newBufferWithBytes`) or by wrapping page-aligned memory (`newBufferWithBytesNoCopy`). The copy is absent only when the data lives in a shared buffer from the start.
- C5: Metal has no per-call error code. Check for nil, read `NSError`, and after the wait read `commandBuffer.status` and `.error`. Out-of-bounds writes need Shader Validation (`MTL_SHADER_VALIDATION=1`) to be reported.
- C8: macOS 26 added a parallel `MTL4` command API. The calls taught here are the original ones and remain valid.
- D4: PyTorch's MPS backend requires unified memory and allocates shared buffers, yet `tensor.to("mps")` still copies. Unified memory removes the need for a copy; it does not remove copies a framework chooses to make.
- Note for the lab runtime, not a vendor fact: the comment in `labs/common/include/pmpp/emu/cuda_emu.h` saying real CUDA takes only `void**` is out of date (A2). Behaviour is fine; only the comment is stale. This file changes nothing there.

## Where the backends really differ

| Feature | CUDA | HIP | Metal |
|---|---|---|---|
| Where the kernel is written | Same `.cu` file as host code | Same file as host code (`.hip`, or `.cpp` with `-x hip`); `#include <hip/hip_runtime.h>` required | Separate `.metal` file in a separate language (MSL, C++17-based) |
| Marking a kernel | `__global__ void` | `__global__ void` | `kernel void` or `[[kernel]] void` |
| Device helper / host function / both | `__device__`, `__host__`, `__host__ __device__` (tell apart with `__CUDA_ARCH__`) | Same three (tell apart with `__HIP_DEVICE_COMPILE__`) | No qualifiers: every MSL function is GPU code. No single-source host-and-device function; projects share a plain header |
| Pointer arguments | Raw device pointers by value | Raw device pointers by value | `device T*` with an address-space keyword, bound to buffer index slots; scalars via `setBytes` (under 4 KB) |
| Thread's global index | Compute `blockIdx.x * blockDim.x + threadIdx.x` | Same | Given: `[[thread_position_in_grid]]` (group and local positions also available) |
| Bounds guard `if (i < n)` | Needed | Needed | Not needed with `dispatchThreads`; needed with `dispatchThreadgroups` |
| Device allocation | `cudaMalloc(void**, bytes)` (typed overload exists); not cleared | `hipMalloc(void**, bytes)` (typed wrapper exists) | `[device newBufferWithLength:options:]`; zero-filled; nil on failure |
| Free | `cudaFree(ptr)`; double free is an error | `hipFree(ptr)`; always synchronises first | None: the buffer object is released by reference counting |
| Host to device, device to host | `cudaMemcpy(dst, src, bytes, kind)`, blocking | `hipMemcpy(dst, src, bytes, kind)` | None for shared buffers: read and write through `contents`. Private buffers need a blit command |
| Host dereference of a device pointer | Not allowed ("GPU memory can only be accessed from kernels") | Documented as slow "zero copy" access; avoid | Normal for shared buffers, after waiting for GPU work |
| Launch | `k<<<blocks, threads>>>(args)` | Same syntax | Command buffer, encoder, `setComputePipelineState`, `setBuffer`, `dispatchThreads`, `endEncoding`, `commit` |
| Block-count arithmetic | `(n + B - 1) / B`, or `cuda::ceil_div` | `(n + B - 1) / B`; zero blocks returns `hipErrorInvalidValue` | Not needed with `dispatchThreads`; same arithmetic with `dispatchThreadgroups` |
| Does the launch wait? | No. `cudaDeviceSynchronize()` | No. `hipDeviceSynchronize()` | No (`commit` is asynchronous). `[commandBuffer waitUntilCompleted]` |
| Error reporting | `cudaError_t` from every call; `cudaGetLastError()` after a launch; `cudaGetErrorString` | `hipError_t` (`[[nodiscard]]` in C++17); `hipGetLastError()`; `hipGetErrorString` | nil returns, `NSError**` on compile steps, `commandBuffer.status` and `.error` after the wait |
| Out-of-bounds write in a kernel | Not reliably reported | Not checked here | Reported only with Shader Validation enabled |
| Threads per block or group | 1024 (ch05 A5) | "typically 1024"; query the device | Up to 1024; read `maxTotalThreadsPerThreadgroup` from the pipeline (ch05 C3) |
| Efficient group size | Multiple of 32 (the warp) | Multiple of the warp size, 32 or 64 (ch05 B5) | Multiple of `threadExecutionWidth` (ch05 C3) |
| Compiler | `nvcc file.cu` (host compiler plus GPU compiler) | `amdclang++` (one Clang for both; `hipcc` is legacy, ch05 B6) | `xcrun metal`, or `newLibraryWithSource:` at run time |
| Shipped device code | PTX and cubin in a fatbin; driver JIT-compiles PTX when needed | Machine code for the architectures named with `--offload-arch` | `.metallib` or source; always finished for the GPU at pipeline creation |
| Kernel launched from a kernel | Dynamic parallelism (CDP2) | Not supported (ch05 B4) | None documented |
| One pointer valid on both sides | `cudaMallocManaged` (pages migrate between two memories) | `hipMallocManaged` (needs HMM and `HSA_XNACK=1` for migration) | Shared storage mode: one physical memory, no migration (ch05 C6) |
| Timing a kernel | Synchronise, or `cudaEventElapsedTime` | Synchronise (events also exist) | `GPUStartTime` / `GPUEndTime` per command buffer (ch05 C9) |
