# Chapter 7 fact check: CUDA, HIP, Metal

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims, the verdict is for the item as a whole and the text says which sub-claim failed. A verdict judges the book's statement, or the assumption a course author would most likely bring, named in the item's title.
- **Quote** is copied from the source (HTML docs flattened to text, GitHub raw files at a tag or on the default branch, PDF text, SDK headers). Line breaks inside quotes are removed.
- "Book" means `notes/ch07.md` with its page numbers. Nothing here is taken from memory of the book.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide "v13.5" (pages last updated Oct 31, 2026), CUDA Runtime API, Best Practices Guide and NVCC docs (Nov 02, 2026), Nsight Compute Profiling Guide (Sep 22, 2026), cuDNN Backend docs (Nov 06, 2026); "HIP 7.17.0 Documentation", "AMD ROCm 10.2.0" architecture pages, "ROCm Compute Profiler (rocprofiler-compute) 3.10.0", HIP and CLR source at tag `therock-10.2` in `ROCm/rocm-systems`, MIOpen at `therock-10.2` in `ROCm/rocm-libraries`, Clang "HIP Support" page (LLVM main docs); Metal Shading Language Specification "Version 4.1" dated 2026-06-04, Metal Feature Set Tables dated October 23, 2026, macOS 27.0 SDK headers, WWDC16 session 606 and WWDC20 session 10603 transcripts; project sources at the default branch head on the check date (verified against `git ls-remote`): llama.cpp `master` at `9d6f04f` (2026-11-20), MLX `main` at `5078f27` (2026-11-20), PyTorch `main` at `d473a25` (2026-11-20); PyTorch docs 2.15, torchvision docs 0.29, MLX docs 0.32.4.
- Not checked: anything by running code on a GPU. Two kinds of local compile were done and are labelled where they appear: Metal front-end compiles with `xcrun -sdk macosx metal` ("Apple metal version 32023.921") for C1 and C3, and a host C++ program with Apple clang 21.0.0 for A8. No CUDA or ROCm toolchain was used.

Already settled elsewhere and not repeated here (cited as "ch05 A4", "infra B5" and so on):

| Topic | Item |
|---|---|
| Constant memory is 64 KB on every listed compute capability; query `totalConstMem` | ch05 A4 |
| `cudaMemcpyToSymbol` signatures (C and C++), optional `offset` and `kind`, `__constant__` at file scope | infra A5 |
| HIP keeps `__constant__`, copies with `hipMemcpyToSymbol`; constant cache only for warp-uniform access; no size in bytes documented; mapped in `pmpp/vendors/hip.h` | infra B5 |
| Metal's two forms of a read-only table: initialised program-scope `constant` array, or a buffer taken as `constant T*` | infra C6 |
| 1024 threads per block, block at most 1024 x 1024 x 64, warp size 32 on CUDA | ch05 A5 |
| HIP block and grid limits come from the driver; the current HIP docs state no number | ch03 B3 |
| Wavefront width 64 or 32 on AMD | ch05 B5 |
| L1 and shared memory are one "unified data cache" on NVIDIA; carveout attribute | ch05 A3 |
| `__shared__`, `__syncthreads()` and the index built-ins keep their spelling on HIP | ch05 B4 |
| Metal `threadgroup` memory (32 KB on Apple4 and later), `threadgroup_barrier` | ch05 C1, C2 |
| Metal threadgroup size: 1024 is a theoretical maximum, check the pipeline's `maxTotalThreadsPerThreadgroup` | ch05 C3 |
| Dynamically sized threadgroup memory on Metal (`setThreadgroupMemoryLength`) | ch05 C8 |
| Metal `preprocessorMacros` for compile-time constants; MSL math is not IEEE-strict by default, so test with a tolerance | ch05 C7 |
| `threadIdx` and friends are unsigned on CUDA; Metal position attributes are unsigned only | ch03 A2, ch03 C6 |
| Small arguments on Metal: `constant T&` filled with `setBytes` (under 4 KB) | ch03 C5 |
| H100 SXM figures behind the book's 20 FLOP/B threshold (67 TFLOPS FP32, 3.35 TB/s) | ch05 A8 |
| The book's pointer to the "CUDA C++ Programming Guide": the old URL is gone, the guide was restructured | ch05 A9 |
| Apple publishes no cache sizes, bank structure or bandwidth figures | infra C13 |
| Timing on each backend | infra A8, ch05 C9 |

Totals: 16 CONFIRMED, 0 WRONG, 8 PARTLY, 1 COULD NOT VERIFY (25 items: A1 to A9, B1 to B6, C1 to C5, D1 to D5). The PARTLY and COULD NOT VERIFY items are collected under "Summary of corrections".

---

## Group A: CUDA

### A1. `__constant__`: scope, and "the kernel file needs the external declaration" (book p. 169)

**Verdict: PARTLY.** Declaration at file scope, grid-wide visibility and read-only access from kernels are confirmed. The book's advice for host code and kernel in different files does not work in nvcc's default mode: an `extern __constant__` declaration needs separate compilation (`-rdc=true`).

Quotes (Programming Guide):

- Section 2.3.3.5: "Constant memory means that a variable: Resides in constant memory space, Has a distinct object per device, Is accessible from all the threads within the grid and from the host in C++ through the runtime library (cudaGetSymbolAddress() / cudaGetSymbolSize() / cudaMemcpyToSymbol() / cudaMemcpyFromSymbol())."
- Section 5.4.1.2: "__constant__ variables are read-only in device code and can only be modified from the host using the CUDA Runtime API."
- Section 5.3.10.4.5 "extern Variables": "When compiling in the whole program compilation mode, __device__, __tile__, __shared__, __managed__, and __constant__ variables cannot be defined with external linkage using the extern keyword." Example as printed: `extern __device__ int y; // ERROR in whole program compilation mode`.
- Section 5.3.10.5.2: "Device variables or functions with external linkage require separate compilation mode across multiple translation units."
- NVCC docs: "CUDA programs are compiled in the whole program compilation mode by default, i.e., the device code cannot reference an entity from a separate file." The switch is `--relocatable-device-code {true|false} (-rdc)` (section 4.2.7.4).

For the labs (our reading of the rules above): keep the `__constant__` definition, the kernel and the host function that calls `cudaMemcpyToSymbol` in the same `.cu` file, which in the course layout means `kernel.cu` beside `launch<Name>`, with `main.cpp` passing the filter in as a plain host array. HIP has the same default (B3).

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.3.5)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.1.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html (sections 5.3.10.4.5, 5.3.10.5.2)
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html (section 4.2.7.4)

### A2. What `cudaMemcpyToSymbol` does (book p. 169)

**Verdict: PARTLY.** The signature and the optional `offset` and `kind` arguments are as the book says (infra A5). The book's description that this copy "informs the CUDA runtime" that the data will not change during the kernel is not how NVIDIA describes it: the function copies to any device symbol, in global or constant memory, and the read-only property comes from the `__constant__` qualifier.

Quotes:

- Runtime API (infra A5): "symbol is a variable that resides in global or constant memory space. kind can be either cudaMemcpyHostToDevice or cudaMemcpyDeviceToDevice."
- The guide's own example uses it on a `__device__` variable: `__device__ float device_var = 4.0f; // Variable in device memory` ... `cudaMemcpyToSymbol(device_var, &host_var, sizeof(host_var)); // Copies from host to device.` (section 5.4.1.2).
- An asynchronous variant exists: the NVIDIA blog in A4 names "cudaMemcpyToSymbol or cudaMemcpyToSymbolAsync".

Safe lesson wording: `__constant__` places the filter in the constant space and makes it read-only in kernels; `cudaMemcpyToSymbol` is how the host writes it.

- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.1.2)

### A3. The constant cache: broadcast, serialisation, working set (book p. 171)

**Verdict: CONFIRMED.** A warp that reads one address is served at register speed; distinct addresses within a warp are serialised. The cache's working set is 8 KB per SM on every listed compute capability, which is smaller than the 64 KB constant space.

Quotes:

- Best Practices Guide section 10.2.6: "There is a total of 64 KB constant memory on a device. The constant memory space is cached. As a result, a read from constant memory costs one memory read from device memory only on a cache miss; otherwise, it just costs one read from the constant cache. Accesses to different addresses by threads within a warp are serialized, thus the cost scales linearly with the number of unique addresses read by all threads within a warp. As such, the constant cache is best when threads in the same warp accesses only a few distinct locations. If all threads of a warp access the same location, then constant memory can be as fast as a register access."
- Programming Guide Table 34: "Cache working set per SM for constant memory | 8 KB" (one value spanning compute capabilities 7.5 to 12.x).
- Programming Guide section 2.3.3.5: "Constant memory is useful for small amounts of data that each thread will use in a read-only fashion."

For the lessons: the convolution filter is the best case because its index does not depend on the thread. A table indexed by per-thread data is the worst case (compare D4). A 15 by 15 float filter is 900 bytes (our arithmetic), well inside the 8 KB working set.

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 10.2.6)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 34)

### A4. Kernel parameters live in constant memory; passing the filter by value (book p. 172)

**Verdict: CONFIRMED**, with a larger limit than older material gives and a newer qualifier.

Quotes:

- C++ language support, section 5.3.10.5.4: "Function parameters are passed to the device via constant memory and their total size is limited to 32,764 bytes."
- NVIDIA blog "CUDA 12.1 Supports Large Kernel Parameters" (Jun 05, 2023): "CUDA 12.1 increases the kernel parameter limit from 4,096 bytes to 32,764 bytes on NVIDIA Volta and later architectures." "Previously, passing kernel arguments exceeding 4,096 bytes required working around the kernel parameter limit by copying excess arguments into constant memory with cudaMemcpyToSymbol or cudaMemcpyToSymbolAsync". "When linking device objects, all objects must be recompiled with CUDA 12.1 if any kernel uses the expanded parameter limit."
- Section 5.4.1.5 `__grid_constant__`: "Annotating a __global__ function parameter with __grid_constant__ prevents the compiler from creating a per-thread copy of the parameter. Instead, all threads in the grid will access the parameter through a single address, which can improve performance." "Kernel parameters annotated with __grid_constant__ must have const-qualified non-reference types." Example as printed: `__global__ void kernel(const __grid_constant__ MyStruct s)`.
- Best Practices Guide: "--ptxas-options=-v or -Xptxas=-v lists per-kernel register, shared, and constant memory usage." A lab can use this to show where the filter went.

A radius-7 float filter in a struct is 900 bytes, far under 32,764 (our arithmetic). Every compute capability the current guide lists (7.5 and later) is Volta or newer. HIP documents neither the limit nor `__grid_constant__` (B2), so the portable `.cu` should pass a plain `const` struct.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html (section 5.3.10.5.4)
- https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.1.5)
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

### A5. The cache figures on p. 171 (L1, L2, L3)

**Verdict: PARTLY.** "L2 is shared among the SMs" is confirmed. "L1 typically up to 128 KB" is out of date for NVIDIA. NVIDIA documents no L2 latency in cycles. No NVIDIA GPU in the docs has an L3; AMD's MI300X lists one (B5).

| Book (p. 171) | Current source | Status |
|---|---|---|
| L1 "typically up to 128 KB" | L1 and shared memory are one unified data cache per SM. Table 35 "Unified Data Cache Size (KB)": 96 (7.5), 192 (8.0, 8.7), 128 (8.6, 8.9, 12.x), 256 (9.0, 10.0, 10.3, 11.0), 336 (10.7). Hopper blog: "256 KB of combined shared memory and L1 data cache". | Out of date as an upper bound |
| L2 "a few hundred KBs to a few tens of MBs" | Hopper blog: "A 50 MB L2 cache in H100 is 1.25x larger than the A100 40 MB L2"; full GH100: "60 MB L2 cache". | Holds for these products |
| L2 "shared among multiple ... SMs in a CUDA device" | Guide 2.3.3.6: "The L2 cache is located on the device and is shared among all the SMs. The size of the L2 cache can be queried with the l2CacheSize device property element". | CONFIRMED |
| L2 "can take tens to hundreds of cycles to access" | No figure found. The Best Practices Guide gives only "latency of access costs hundreds of clock cycles" for global memory. | COULD NOT VERIFY |
| L3 "hundreds of MBs" in "some high-end processors" | No L3 in NVIDIA's docs. AMD's spec table lists MI300X "L3 Cache (MiB)" 256 (B5). | Holds for one AMD GPU; not NVIDIA |

Newer than the book and relevant to section 7.6: from compute capability 8.0 a kernel can reserve part of L2 for data it reads repeatedly. Best Practices Guide section 10.2.2: "Starting with CUDA 11.0, devices of compute capability 8.0 and above have the capability to influence persistence of data in the L2 cache." Example: `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize);` with a stream `accessPolicyWindow`. The guide's default otherwise: "If these hints are not used, the compiler and runtime will do their best to utilize the caches efficiently." The labs do not need this.

Lesson rule: the reader's L2 size comes from `l2CacheSize` (`cudaDeviceProp`: "Size of L2 cache in bytes.") through the doctor, not from these product figures.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 35)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.3.6)
- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (sections 10.2.2 and the memory optimisation overview)
- https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html

### A6. Other read-only paths: `const __restrict__`, `__ldg`, textures

**Verdict: CONFIRMED.** Two NVIDIA documents disagree about textures; the Programming Guide is the newer and more specific statement.

Quotes:

- Section 5.4.1.4: "Accesses to __global__ function const pointers marked with __restrict__ are compiled as read-only cache loads, similar to the PTX ld.global.nc or __ldg() low-level load and store functions instructions." Printed pair: `void kernel1(const float* in, float* out) { *out = *in; // PTX: ld.global` and `void kernel2(const float* __restrict__ in, float* out) { *out = *in; // PTX: ld.global.nc`. Caveat in the same section: "Since register pressure is a critical issue in many CUDA codes, the use of restricted pointers can negatively impact performance by reducing occupancy."
- Section 5.4.8.3: `T __ldg(const T* address);` "The function __ldg() performs a read-only L1/Tex cache load."
- Section 2.3.3.7: "Texture and surface memory do not provide any performance advantage for non-graphics applications in CUDA on any currently supported NVIDIA GPU." And: "On all currently supported GPUs, these scenarios may be handled using direct load and store instructions".
- Against that, Best Practices Guide section 10.2.5 still says "In certain addressing situations, reading device memory through texture fetching can be an advantageous alternative to reading device memory from global or constant memory", with Table 4 listing "Addressing modes | Automatic handling of boundary cases".
- Texture address modes (Runtime API): `cudaAddressModeWrap = 0` "Wrapping address mode.", `cudaAddressModeClamp = 1` "Clamp to edge address mode.", `cudaAddressModeMirror = 2`, `cudaAddressModeBorder = 3` "Border address mode."

For the lessons: the book's three boundary policies (zero, replicate, wrap) correspond to border, clamp and wrap addressing (the mapping is ours). Mention it as a `<Beyond>` note at most; follow the Programming Guide and do not teach textures as a performance technique for this chapter. `const float* __restrict__ N` is worth one sentence as the read-only-cache route for the input.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.1.4, 5.4.8.3)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.3.7)
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 10.2.5)
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html (`cudaTextureAddressMode`)

### A7. The H100 examples on p. 167 and the shading of Fig. 7.14

**Verdict: CONFIRMED** as the book's example, on the H100 SXM figures of ch05 A8.

With NVIDIA's published 67 TFLOPS FP32 and 3.35 TB/s the threshold is 20.0 FLOP/B (19.97 with the book's 66.9; our arithmetic). A 3 by 3 filter's ideal 2.25 FLOP/B is below it and an 11 by 11 filter's 30.25 is above it under either figure. In Fig. 7.14 every green cell is above 20 (the lowest is m = 9 ideal, 20.25, then m = 13, t = 16 at 20.80) and every pink cell below it (the highest is m = 9, t = 64 at 17.88), so the colouring matches a 20 FLOP/B threshold. The FP32 peak is a non-tensor-core figure (ch05 A8). Present all of this as the book's example; the explorer takes the reader's numbers from `useReports()`.

- https://www.nvidia.com/en-us/data-center/h100/ (via ch05 A8)

### A8. Unsigned built-ins and the guard in Fig. 7.15 (notes, "Unclear in the PDF")

**Verdict: CONFIRMED** (the notes' analysis holds). `threadIdx.x - FILTER_RADIUS + fCol` is `unsigned`, so the printed `>= 0` test is always true; the `< TILE_DIM` half still rejects the wrapped values, so the guard works by accident and the real erratum is the missing `- FILTER_RADIUS` in the `N_s` index.

- The built-ins are unsigned (ch03 A2).
- C++ rule (cppreference, "Usual arithmetic conversions"): when one operand is signed and the other unsigned, "If the integer conversion rank of U is greater than or equal to the integer conversion rank of S, C is U." So `unsigned int` minus `int` is `unsigned int`.
- Compiled and run here (Apple clang 21.0.0, host C++, a struct of three `unsigned` fields standing in for `uint3`): `static_assert(std::is_same_v<decltype(threadIdx.x - FILTER_RADIUS + 0), unsigned>)` passes; with `threadIdx.x = 0`, `FILTER_RADIUS = 2`, `fCol = 0` the expression is 4294967294, `>= 0` is true and `< 32` is false. With `-Wall -Wextra -Wtautological-unsigned-zero-compare` clang reports "result of comparison of unsigned expression >= 0 is always true". Not compiled with nvcc (no CUDA toolchain on this machine).

For the labs: convert to `int` first (`int tx = threadIdx.x;`), as the book already does for `row` and `col`. The same care applies on Metal, whose position attributes accept only unsigned types (ch03 C6).

- https://en.cppreference.com/w/cpp/language/usual_arithmetic_conversions.html

### A9. Checking section 7.6's "the halos are probably in L2" on NVIDIA

**Verdict: CONFIRMED** that NVIDIA's profiler reports cache hit rates, with a documented caveat.

Quotes (Nsight Compute Profiling Guide):

- Memory Chart: "Performance data includes transfer sizes, hit rates, number of instructions or requests, etc." It shows "Hit Rate" for L1 ("Sector hit rate (percentage of requested sectors that do not miss) in the L1 cache.") and for L2.
- "A metric such as hit rate (hits / queries) can have significant error if hits and queries are collected on different passes and the kernel does not saturate the GPU to reach a steady state (generally > 20 µs)."

The labs measure time only; a hit rate is an optional check for a reader with Nsight Compute. AMD and Apple counterparts are in B5 and C4.

- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

---

## Group B: HIP / ROCm

### B1. `__constant__` on AMD: size and the cache behind it (counterpart of A1 and A3)

**Verdict: PARTLY.** The API is CUDA's (infra B5). The "64 KB with a constant cache" picture does not carry over: AMD documents no size, `totalConstMem` reports something else, and the cache that serves `__constant__` is the scalar cache, which helps only uniform reads.

Quotes:

- HIP language extensions, `__constant__`: "Variables marked with __constant__ reside in device memory. Variables in that address space are routed through the constant cache, but that address space has a limited logical size." "the access has to be uniform within a warp, otherwise the accesses are serialized." "The constant cache reduces the pressure on the other caches and may enable higher throughput and lower latency accesses."
- HIP hardware implementation, "Scalar L1 data cache (sL1D)": "The sL1D is shared between CUs and caches data that is uniform across a warp, including: Kernel arguments and pointers; Grid and block dimensions; Constants accessed uniformly across threads; Data from __constant__ memory when accessed uniformly". "Unlike the vector L1 cache, the sL1D doesn't use a "hit-on-miss" approach, meaning subsequent requests to the same pending cache line count as duplicated misses rather than hits."
- ROCm GPU specifications, MI300X row: "L1 Scalar Cache (KiB)" = "16 per 2 CUs".
- CLR source at `therock-10.2`: `deviceProps.totalConstMem = std::min(info.maxConstantBufferSize_, kInt32Max);` (`hip_device.cpp` line 662) and `info_.maxConstantBufferSize_ = info_.maxMemAllocSize_;` (`rocdevice.cpp` line 1424). So on AMD `totalConstMem` reports the largest single allocation, capped at 2^31 - 1, not a hardware constant-memory size. The same function sets `info_.preferredConstantBufferSize_ = 16 * Ki;` (line 1423); I found no HIP property or document that exposes or explains it. Not run on AMD hardware.

For the lessons: do not say "64 KB on AMD"; expect the doctor's `totalConstMem` to look very different on AMD and NVIDIA. The chapter's filter is a uniform read, which is the case both vendors' constant caches are built for.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "__constant__")
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (section "Scalar L1 data cache (sL1D)")
- https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_device.cpp (lines 652 to 666)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/rocclr/device/rocm/rocdevice.cpp (lines 1420 to 1425)

### B2. Kernel arguments on AMD: size limit and `__grid_constant__` (counterpart of A4)

**Verdict: COULD NOT VERIFY.** AMD documents that kernel arguments are read through the scalar path, but I found no documented size limit for HIP kernel arguments and no HIP documentation of `__grid_constant__`.

What is documented:

- Hardware page: the sL1D caches "Kernel arguments and pointers" (B1), and the scalar unit handles "Loading kernel arguments and constants".
- Language extensions page, kernel parameter restrictions, in full: "Kernels can't: have a parameter of type std::initializer_list or va_list; have a variable number of arguments; use references as parameters; use parameters having different sizes in host and device code, e.g. long double arguments, or structs containing long double members; use struct-type arguments which have different layouts in host and device code." No size limit appears.

Not found: a byte limit in the HIP 7.17.0 language-extensions page, the porting guide, `hip_runtime_api.h` at `therock-10.2`, or a web search of AMD and LLVM sites; `__grid_constant__` in the same places and in the CLR `host_defines.h` (which defines `__constant__` as `__attribute__((constant))`). The CLR sets an OpenCL-style `info_.maxParameterSize_ = 1024;` (`rocdevice.cpp` line 1372), which I found used only for the device-side enqueue queue in `rocvirtual.cpp`; I did not establish that it limits ordinary HIP launches.

For the labs: a filter struct of a few hundred bytes passed by value is the portable choice; do not use `__grid_constant__` in a `.cu` that must also build for HIP.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "Kernels")
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/host_defines.h (line 378)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/rocclr/device/rocm/rocvirtual.cpp (lines 4695 to 4756)

### B3. Cross-file device symbols on HIP: `-fgpu-rdc` (counterpart of A1)

**Verdict: CONFIRMED** that HIP's default is also self-contained device code per file, with relocatable device code as the opt-in.

Quotes (Clang "HIP Support"):

- "-fno-gpu-rdc Mode (default): Compiles to a self-contained, fully linked offloading device binary for each offloading device architecture. Device code within a Translation Unit (TU) cannot call functions located in another TU."
- "RDC, enabled with the -fgpu-rdc compiler option, is necessary for linking device codes across translation units."

The page speaks of functions and "device codes"; it does not mention `extern` variables by name. Applying it to an `extern __constant__` declaration is our reading, consistent with A1. The same lab layout (symbol, kernel and setter in one file) avoids the question on both vendors.

- https://clang.llvm.org/docs/HIPSupport.html (sections "Predefined Macros" and "Compilation Modes")

### B4. `__ldg`, `__restrict__` and texture address modes on HIP (counterpart of A6)

**Verdict: CONFIRMED.** All three exist with CUDA's spellings; `__ldg` is an ordinary load.

- `hip_ldg.h` at `therock-10.2`, every overload has the form `__device__ inline static int __ldg(const int* ptr) { return ptr[0]; }`, so code that uses `__ldg` compiles for HIP and gets a plain load.
- Language extensions, `__restrict__`: "The __restrict__ keyword tells the compiler that the associated memory pointer does not alias with any other pointer in the function. This can help the compiler perform better optimizations." I found no AMD statement that `const __restrict__` selects a separate read-only cache.
- `texture_types.h`: `enum hipTextureAddressMode { hipAddressModeWrap = 0, hipAddressModeClamp = 1, hipAddressModeMirror = 2, hipAddressModeBorder = 3 }`, the same four values as CUDA.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/hip_ldg.h
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "__restrict__")
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/texture_types.h (lines 59 to 64)

### B5. The caches behind section 7.6's halo reads on AMD (counterpart of A5 and A9)

**Verdict: PARTLY.** AMD GPUs do have an L2 that serves global reads, but on MI300X it is per chiplet (XCD), not per GPU, and AMD says which compute unit runs a workgroup is not predictable. The book's premise that neighbouring blocks share one L2 is an NVIDIA-shaped assumption here.

Quotes:

- Hardware implementation, vector L1: "Each CU contains a dedicated vector L1 data cache (vL1D) serving vector memory operations." "Write-through design (writes go directly to L2)" "Typical size of 16 KB per CU".
- Same page, L2: "The L2 cache serves as the coherence point for all GPU memory accesses and is shared by all compute units." "Hit-on-miss behavior: If a request arrives for a pending cache line fill, it counts as a hit, improving the effective hit rate."
- Same page, scheduling: "Workgroup-to-CU mapping is non-deterministic and based on available resources. You should not assume any specific mapping pattern, as the same kernel launched multiple times can have different workgroup distributions."
- MI300 microarchitecture (ROCm 10.2.0): "The XCD has 40 CUs: 38 active CUs at the aggregate level and 2 disabled CUs for yield management. The CUs all share a 4 MB L2 cache that serves to coalesce all memory traffic for the die." "the processor can include 6 or 8 XCDs". Figure caption: "40 Compute Units, each with 32 KB L1 cache".
- ROCm GPU specifications, MI300X row: "LDS (KiB)" 64, "L3 Cache (MiB)" 256, "L2 Cache (MiB)" "32 (4 per XCD)", "L1 Vector Cache (KiB)" 32. (Radeon rows call the same level "Infinity Cache (MiB)".)
- rocprofiler-compute 3.10.0, L2 cache: "Hit Rate | The ratio of the number of L2 cache line requests that hit in the L2 cache over the total number of incoming cache line requests to the L2 cache." Given the hit-on-miss rule above, this rate counts some waiting requests as hits.

Two inconsistencies in AMD's sources: the generic "Typical size of 16 KB per CU" for the vector L1 against 32 KiB for MI300X in the spec table; and the header comment on `hipGetDeviceProperties` at `therock-10.2` says "@bug HIP-Clang always returns 0 for l2CacheSize", while the CLR fills it (`deviceProps.l2CacheSize = info.l2CacheSize_;`, `hip_device.cpp` line 666, from the HSA cache sizes in `rocdevice.cpp`). Read the value through the doctor and treat 0 as "not reported".

For the lessons (our reasoning from the quotes): on MI300X, two neighbouring blocks may run on different XCDs, where their shared halo can only meet in the 256 MiB last-level cache; on a single-die AMD GPU the generic "shared by all compute units" applies. The section 7.6 lab should measure rather than assume.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (sections "Vector L1 cache", "L2 cache architecture", "Workgroup manager (SPI)")
- https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300.html
- https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/cdna/l2-cache.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 2624 to 2638)

### B6. A 32 by 32 block on HIP, and `__launch_bounds__` (Fig. 7.12 at `IN_TILE_DIM 32`)

**Verdict: CONFIRMED.** Without `__launch_bounds__`, HIP compiles a kernel for the device's largest block, and validates launches against it.

Quotes (HIP language extensions, `__launch_bounds__`):

- "If no __launch_bounds__ are specified, MAX_THREADS_PER_BLOCK is the maximum block size supported by the device".
- "MAX_THREADS_PER_BLOCK is particularly useful in cases, where the compiler is constrained by register usage in order to meet requirements of large block sizes that are never used at launch time."
- "When launching kernels HIP will validate the launch configuration to make sure the requested block size is not larger than MAX_THREADS_PER_BLOCK and return an error if it is exceeded."

The device maximum comes from the driver (`maxThreadsPerBlock`, ch03 B3, which could not confirm the values AMD devices report). For the lab: read `maxThreadsPerBlock` through the doctor and check it before launching a 1024-thread block. On a 64-wide wavefront a 32 by 32 block is 16 wavefronts (our arithmetic).

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "__launch_bounds__")

---

## Group C: Metal

### C1. Holding the filter on Metal: three documented ways, none of them a symbol copy

**Verdict: CONFIRMED** for the mechanisms. No current Apple document gives a size limit for constant-address-space buffers.

The book's three options (p. 172) and their Metal counterparts:

1. `__constant__` plus `cudaMemcpyToSymbol`: an `MTLBuffer` the host fills (shared storage, ch05 C6), bound as `constant float* F [[buffer(n)]]` (infra C6). Feature Set Tables, every Apple family: "Maximum number of constant buffer arguments in vertex, fragment, tile, or kernel functions | 31" and "Minimum constant buffer offset alignment | 4B".
2. Weights passed with the launch: a struct with a fixed-size array, taken as `constant Filter& F [[buffer(n)]]` and filled with `setBytes` (ch03 C5). Feature Set Tables, every family: "Maximum length of inlined buffer contents using setBytes | 4 KB". SDK header (`MTLComputeCommandEncoder.h`): `- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index API_AVAILABLE(macos(10.11), ios(8.3));` "Set the data (by copy) for a given buffer binding point." A radius-7 float filter (900 bytes) fits; a 32 by 32 float filter (4096 bytes) does not fit under "smaller than 4 kilobytes" (ch03 C5) (our arithmetic).
3. Hard-coded values: a program-scope `constant` array. MSL spec 4.2: "You must declare variables in program scope in the constant address space and initialize them during the declaration statement." "Declaring such a variable without initialization is also a compile-time error." Compiled here: `constant float F[25];` fails with "error: variable in constant address space must be initialized"; `constant float F5[25] = {1,2,3,2,1, ...};` read in a loop compiles.

So the host cannot write a program-scope `constant` array after compilation; option 3 is the counterpart of the book's hard-coding, not of `__constant__`.

Size: the current Feature Set Tables have no row for the size of a constant-address-space buffer, and the MSL spec states none. Apple's WWDC16 talk says only "There are limits on how much data you can pass through this address space" (C2). COULD NOT VERIFY a number.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 4.2, page 119)
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf (GPU implementation limits by family, page 7)
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbytes(_:length:index:)
- Local header: `Metal.framework/Headers/MTLComputeCommandEncoder.h` (macOS 27.0 SDK), line 65

### C2. Is there a constant cache on Apple GPUs?

**Verdict: PARTLY.** Apple documents no constant cache and no size for one. What it has said, in conference talks rather than reference documentation, is that the `constant` address space is the path built for heavily reused read-only data, and that bounded constant data can be preloaded into registers.

Quotes:

- MSL spec 4.2: "The constant address space name refers to buffer memory objects allocated from the device memory pool that are read-only." No cache statement (infra C6).
- WWDC16 session 606, "Advanced Metal Shader Optimization": "As the name implies, this is a read only address space, but there are a couple of additional restrictions. There are limits on how much data you can pass through this address space, and additionally the buffer offsets that you specify at the API level have more stringent alignment requirements. However, this is the address space that's optimized for cases with a lot of data reuse." "The first question is, do I know how much data I have. And if you have a potentially variable amount of data, this is usually a sign that you need to be using the device address space." "if these items can potentially be read many times, this is usually a sign that you want to put them into the constant address space."
- Same talk, "constant buffer preloading": "rather than loading through the constant address space, what we can actually do is take your data and put it into special constant registers that are even faster for the ALU to access. So we can do this as long as we know exactly what data will be read." "First, you need to make sure that this data is in the constant address space. And additionally you need to indicate that your accesses are statically bounded. The best way to do this is to pass your arguments by reference rather than pointer where possible." "you can embed that size array and pass that struct by reference rather than passing the original pointer." "In practice we see that on A8 later GPUs that this is a significant performance win."
- WWDC20 session 10603: "Metal buffers are accessed only by the Shader Core, which does have a dedicated L1 as well as support for different address spaces. Address spaces for buffer data are device for read-write data or constant for read-only data. ... use constant address space for data that is utilized by many vertices or fragments."

For the lessons: a fixed-size filter passed as `constant Filter&` is exactly the case the 2016 advice describes. Whether a given Mac preloads it is undocumented; the talk is about iOS GPUs of 2016. Neither talk mentions a requirement that SIMD-group lanes read the same address, unlike NVIDIA (A3) and AMD (B1). Put the three C1 variants in the lab and let the reader measure; do not state a speed-up as fact.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 4.2)
- https://developer.apple.com/videos/play/wwdc2016/606/
- https://developer.apple.com/videos/play/wwdc2020/10603/

### C3. A compile-time radius on Metal: function constants and their limits

**Verdict: CONFIRMED.** Function constants are Metal's documented way to fix a scalar such as the radius when the pipeline is created. They cannot be arrays, so they cannot hold the weights, and they cannot size a `threadgroup` array.

Quotes (MSL spec section 5.8):

- "Function constants enable the generation of multiple variants of a function." "Function constants provide the same ease of use as preprocessor macros but moves the generation of the specific variants to the creation of the pipeline state, so you don't have to compile the variants offline."
- "Function constants can only be a scalar or vector type. Using a user-defined type or an array of a scalar or vector type for a function constant results in a compilation error."
- "You can also use function constants to initialize variables in program scope declared in the constant address space." Example: `constant int a [[function_constant(0)]];` ... `constant int d = (a * 4);`. "The value index needs to be between 0 and 65535."
- Host side (macOS 27.0 SDK headers): `MTLFunctionConstantValues` `- (void)setConstantValue:(const void *)value type:(MTLDataType)type atIndex:(NSUInteger)index;` and `MTLLibrary` `newFunctionWithName:(NSString *)name constantValues:(MTLFunctionConstantValues *)constantValues ...`.

Compiled here (`xcrun metal`, front end only): with `constant int R [[function_constant(0)]]; constant int M = 2*R + 1;` a loop `for (int i = 0; i < M; ++i)` compiles; `threadgroup float tile[32 + 2*R][32 + 2*R];` fails with "error: array size is not a constant expression".

For the tiled lab on Metal: size the tile with a macro (`preprocessorMacros`, ch05 C7) or with dynamic threadgroup memory (ch05 C8). MLX's depthwise convolution makes the same choice: kernel size, stride and flip are function constants, and the threadgroup array is sized by `constexpr` values for the largest case it accepts (D2).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 5.8, pages 190 to 191)
- Local headers: `Metal.framework/Headers/MTLFunctionConstantValues.h` (line 19), `MTLLibrary.h` (line 492), macOS 27.0 SDK

### C4. A shared last-level cache on Apple GPUs (section 7.6 on Metal)

**Verdict: PARTLY.** Apple has described a GPU-wide last-level cache in a talk, qualitatively; there are no sizes, and the talk is about graphics on an iPhone-class GPU.

Quotes (WWDC20 session 10603, "Optimize Metal apps and games with GPU counters"):

- "Both the ALU and the TPU have dedicated L1s. All of the GPU cores share a last level cache."
- "The GPU Last Level Cache is shared across all GPU Cores. It caches both texture and buffer data, as well as storing device atomics. It is optimized for spatial and temporal locality."
- "Notice, though, that there's also System Level Cache, or SLC."
- "Typing "Last Level Cache" will, very unsurprisingly, reveal all of the GPU LLC counters, including the limiter".

Consistent with infra C13 (no numbers). So section 7.6's assumption has a documented structure on Apple GPUs to rest on, and Xcode's counters can show the cache's load; whether halo reads hit it on the reader's Mac is a measurement.

- https://developer.apple.com/videos/play/wwdc2020/10603/

### C5. The book's boundary policies as sampler address modes

**Verdict: CONFIRMED.** Metal samplers implement zero, replicate and wrap boundaries in hardware, plus two more.

Quotes (MSL spec Table 2.7 and the text after it):

- `address` valid values: "repeat, mirrored_repeat, clamp_to_edge (default), clamp_to_zero, clamp_to_border".
- "clamp_to_zero is equivalent to clamp_to_border with a border color of transparent_black (0.0, 0.0, 0.0) with the alpha component value from the texture."
- "If coord is set to pixel, the min_filter and mag_filter values must be the same, the mip_filter value must be none, and the address modes must be either clamp_to_zero, clamp_to_border, or clamp_to_edge."

Mapping (ours): zero ghost cells are `clamp_to_zero`, replicate-the-edge is `clamp_to_edge` (the default, so not zero), wrap is `repeat`. Integer pixel coordinates rule out `repeat`. The labs use buffers, as the book does; this is a `<Beyond>` note. CUDA and HIP have the corresponding texture modes (A6, B4).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 2.10, Table 2.7, pages 45 to 46)

---

## Group D: What PyTorch, MLX and llama.cpp do (for the third goal)

Read from the clones named at the top, which matched the remote default branches on the check date. "By name" means the file exists and I did not read it.

### D1. What frameworks call convolution is the un-flipped form, and the book's boundary policies are padding modes

**Verdict: CONFIRMED.** Every library checked computes cross-correlation by default, which is the book's formula (notes, section 7.1).

- PyTorch 2.15 `torch.nn.Conv2d`: "where ⋆ is the valid 2D cross-correlation operator". `padding_mode (str, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'`. Default `padding=0`.
- PyTorch 2.15 `torch.nn.functional.conv2d`: "padding='valid' is the same as no padding. padding='same' pads the input so the output has the same shape as the input. However, this mode doesn't support any stride values other than 1."
- MLX 0.32.4 `mlx.core.conv_general(..., flip: bool = False, ...)`: "flip (bool, optional) – Flip the order in which the spatial dimensions of the weights are processed. Performs the cross-correlation operator when flip is False and the convolution operator otherwise."
- cuDNN (legacy, deprecated enum): "The filter used for the convolution can be applied in two different ways, corresponding mathematically to a convolution or to a cross-correlation. (A cross-correlation is equivalent to a convolution with its filter rotated by 180 degrees.)" Values `CUDNN_CONVOLUTION`, `CUDNN_CROSS_CORRELATION`. PyTorch sets `.setMathMode(CUDNN_CROSS_CORRELATION)` (`aten/src/ATen/native/cudnn/Conv_v8.cpp` line 128).
- MIOpen (`therock-10.2`, `miopen.h` line 408): `miopenConvolution = 0, /*!< Cross-Correlation convolution */`. PyTorch's ROCm path uses `miopenConvolution` (`aten/src/ATen/native/miopen/Conv_miopen.cpp` line 860).

Bridge (ours): the book's kernel, with zero ghost cells and one output per input, is `F.conv2d(x, w, padding='same')` for an odd square filter, or `padding=r`; the book's replicate and wrap policies are `padding_mode='replicate'` and `'circular'`. PyTorch's default `padding=0` produces no ghost cells at all and a smaller output.

- https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html
- https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.conv2d.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv_general.html
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html (`cudnnConvolutionMode_t`)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/miopen/include/miopen/miopen.h (line 408)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/miopen/Conv_miopen.cpp

### D2. How the libraries compute a 2D convolution today (bridge to chapter 19)

**Verdict: CONFIRMED.** The general path is a matrix multiplication (explicit im2col or implicit GEMM, chapter 19); direct kernels of this chapter's shape remain for special cases.

llama.cpp:

- `ggml_conv_2d` is `ggml_im2col` followed by `ggml_mul_mat` (`ggml/src/ggml.c` lines 4679 to 4701). Header comment on `ggml_im2col`: "converts data into a format that effectively results in a convolution when combined with matrix multiplication" (`ggml/include/ggml.h` line 1828). There is also `ggml_conv_2d_direct` (line 2013).
- CUDA `conv2d.cu`: `conv2d_kernel` gives one thread one output, computes the valid tap range once (`calculate_kernel_bounds`) instead of an `if` per tap, takes its sizes as `const conv_params P` by value and its tensors as `__restrict__` pointers (lines 73 to 106). `conv2d_implicit_gemm_f16` stages `__shared__` tiles and uses tensor-core `tile<>` types (lines 127 onward). `conv2d_pad_f16` writes zero ghost cells explicitly with `(unsigned) x < (unsigned) iw && (unsigned) y < (unsigned) ih ? input[...] : 0.0f`, one unsigned compare per axis in place of two signed ones (lines 116 to 125).
- Metal `kernels/conv.metal`: `kernel_conv_2d` (line 164) loops over outputs, clips `ky_start`/`ky_end` the same way, and reads weights from `device const char * weights`.

MLX (`mlx/backend/metal/conv.cpp`, `dispatch_conv_2D_gpu`, lines 1408 onward): Winograd for 3 by 3, stride 1, channel counts multiples of 32 and large enough inputs; implicit GEMM; explicit GEMM through a `naive_unfold` (im2col) kernel; and a depthwise path when each group has one channel, the kernel is at most 7 by 7 and the stride at most 2. That depthwise kernel (`kernels/conv.metal`, `depthwise_conv_2d`, lines 179 to 290) is the book's halo tiling: each threadgroup loads `span_h = tgp_h * str_h + ker_h - 1` by `span_w` inputs into `threadgroup T ins[...]`, writes `T(0)` for ghost cells, calls `threadgroup_barrier(mem_flags::mem_threadgroup)`, then computes. Its threads loop to load the whole input tile (the block-equals-output-tile organisation of the book's Exercise 11). Weights come from `const device T* wt`, sizes from `const constant MLXConvParams<2>& params`.

PyTorch: `enum class ConvBackend` lists `CudaDepthwise2d`, `Cudnn`, `Miopen`, `MiopenDepthwise`, `Mps`, `Slow2d`, `Winograd3x3Depthwise` and others (`aten/src/ATen/native/ConvUtils.h` lines 112 to 136). MPS calls `[mpsGraph convolution2DWithSourceTensor:...]` (`aten/src/ATen/native/mps/operations/Convolution.mm` line 955). The native CUDA fallbacks are `ConvolutionMM2d.cu`, `Im2Col.cu`, `DepthwiseConv2d.cu` (by name).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml.c
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/conv2d.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/conv.metal
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/conv.cpp
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/conv.metal
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ConvUtils.h
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Convolution.mm
- https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda

### D3. Where halo tiling meets LLM inference: the short causal convolution

**Verdict: CONFIRMED** (from source). Several model families that llama.cpp serves run a short 1D convolution per channel, and its CUDA kernel stages an input tile with a halo in shared memory.

- `ggml_ssm_conv(ctx, sx, c)` (`ggml.h` line 2291) is called from 17 model files under `src/models/` today, including `mamba-base.cpp`, `qwen3next.cpp`, `qwen35.cpp`, `lfm-base.cpp` (`build_lfm_shortconv`), `plamo2.cpp` and `kimi-linear.cpp`.
- The halo is the past, not zeros: `// => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}` / `ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, x), 0);` (`mamba-base.cpp` lines 55 to 56); afterwards "copy last (d_conv - 1) columns back into the state cache". In `ggml_ssm_conv`: `const int64_t n_t = sx->ne[0] - d_conv + 1; // tokens per sequence` (`ggml.c` line 5609). The filter is causal, so the halo is one-sided, `d_conv - 1` wide.
- CUDA `ssm-conv.cu`: the filter width is a template parameter chosen by `switch (nc)` for 2, 3, 4, 5, 7, 9 and 15 taps, else `GGML_ABORT("Only support kernel sizes 2, 3, 4, 5, 7, 9, 15 right now.")`: a compile-time radius, as with the book's `FILTER_RADIUS`. Calls with `n_t <= 32` (token generation) use `ssm_conv_f32`, where each thread keeps a `d_conv`-long window in registers and slides it. Longer calls (prompt processing) use `ssm_conv_long_token_f32`, which loads `d_conv - 1 + split_n_t` columns per channel into `extern __shared__ float smem[]` (`smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float)`, `split_n_t = 32`), calls `__syncthreads()`, then reads each window from shared memory. Per-channel weights go to registers: "// Load weights into registers (done once, small)" (lines 5 to 121, 136 to 160).
- Metal `kernels/ssm.metal` has `kernel_ssm_conv_f32_f32`, `_f32_4`, `_batched`, `_batched_4` (by name).

Honest scope (ours): this is a depthwise 1D convolution of a few taps, not the chapter's 2D image filter, and its cost is small next to the model's matmuls. It is a real place where an input tile plus halo appears in LLM serving. Constant memory plays no part: every thread has its own channel's weights.

- https://github.com/ggml-org/llama.cpp/blob/master/src/models/mamba-base.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/models/lfm-base.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ssm-conv.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/ssm.metal

### D4. Small read-only tables and kernel arguments in real backends

**Verdict: CONFIRMED.** llama.cpp keeps its quantisation lookup tables in global memory on CUDA and HIP and in the `constant` address space on Metal; neither llama.cpp's CUDA backend nor PyTorch's native CUDA kernels use `__constant__`. Both use the "parameters by value" route for small argument structs.

- `ggml/src/ggml-common.h` lines 467 to 505: `#elif defined(GGML_COMMON_IMPL_METAL)` ... `#define GGML_TABLE_BEGIN(type, name, size) static const constant type name[size] = {` and `#elif defined(GGML_COMMON_IMPL_CUDA) || defined(GGML_COMMON_IMPL_HIP) || defined(GGML_COMMON_IMPL_MUSA)` ... `#define GGML_TABLE_BEGIN(type, name, size) static const __device__ type name[size] = {`. The tables include `kmask_iq2xs` (8), `ksigns_iq2xs` (128), `iq2xxs_grid` (256), `iq2xs_grid` (512), `iq2s_grid` (1024), `iq3xxs_grid`, `iq3s_grid`, `kvalues_iq4nl` (16), `kvalues_fp4` (16), `iq1s_grid`.
- `grep -rn __constant__` finds nothing in `ggml/src/ggml-cuda/` or in `aten/src/ATen/native/cuda/`.
- Our reading, not stated in the source: these codebooks are indexed by each thread's quantised value, the pattern NVIDIA says the constant cache serialises (A3) and AMD says it does not help (B1).
- Arguments: llama.cpp's Metal kernels take `constant ggml_metal_kargs_conv_2d & args` (`conv.metal` line 165), bound by `ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0)`, which is `[encoder->obj setBytes:data length:size atIndex:idx]` (`ggml-metal-device.m` lines 875 to 876). Its CUDA `conv2d_kernel` takes `const conv_params P` by value (D2), which NVIDIA places in constant memory (A4). MLX passes `const constant MLXConvParams<2>& params [[buffer(3)]]` (D2).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m
- https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda

### D5. "Gaussian filters ... can be used to sharpen boundaries and edges" (book p. 159)

**Verdict: PARTLY.** In the reader's tools a Gaussian filter is a blur. torchvision 0.29 `torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))`: "Blurs image with randomly chosen Gaussian blur." That matches the chapter's own use of the blur kernel as a smoothing filter. I did not look for a source on sharpening built from a Gaussian and assert nothing about it. Lessons should call the Gaussian a smoothing filter and not repeat "sharpen".

- https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.GaussianBlur.html

---

## Summary of corrections

Every PARTLY or COULD NOT VERIFY item, one line each:

- A1: an `extern __constant__` declaration in another file is an error in nvcc's default whole-program mode; it needs `-rdc=true`. Keep the symbol, the kernel and the `cudaMemcpyToSymbol` call in one `.cu`.
- A2: `cudaMemcpyToSymbol` does not tell the runtime anything about read-only data; it copies to any device symbol, `__device__` or `__constant__`. The `__constant__` qualifier makes the filter read-only in kernels.
- A5: L1 "typically up to 128 KB" is out of date: NVIDIA's unified L1/shared memory is 96 to 256 KB per SM on listed compute capabilities (336 KB on 10.7), 256 KB on H100. NVIDIA documents no L2 latency in cycles and no L3; the 256 MiB L3 of the book's description exists on AMD's MI300X. L2 persistence controls (compute capability 8.0 and later) are newer than the book.
- B1: AMD documents no size for `__constant__` memory, and HIP's `totalConstMem` reports the largest allocation (capped at 2^31 - 1) according to the CLR source, not a 64 KB-style limit. The cache that serves `__constant__` is the scalar L1, which helps only uniform reads and does not count hits on pending lines.
- B2 (could not verify): no documented kernel-argument size limit for HIP, and no HIP documentation of `__grid_constant__`.
- B5: on MI300X the L2 is per XCD (4 MB, 38 active CUs), the GPU-wide level is the 256 MiB last-level cache, and AMD says workgroup-to-CU mapping is non-deterministic, so neighbouring blocks need not share an L2. AMD's own figures disagree on vector L1 size (16 KB "typical" against 32 KiB on MI300X), and a HIP header comment says `l2CacheSize` is always 0 while the runtime fills it.
- C1: Metal has no symbol-copy counterpart of `cudaMemcpyToSymbol`; a program-scope `constant` array must be initialised in the source. No current Apple document gives a size limit for constant-address-space buffers; `setBytes` is limited to 4 KB.
- C2: Apple documents no constant cache. Its 2016 and 2020 talks describe the `constant` address space as the path for heavily reused read-only data and "constant buffer preloading" for statically bounded data passed by reference.
- C4: Apple's only description of a GPU-wide cache is a 2020 talk ("All of the GPU cores share a last level cache"), with no sizes.
- D5: torchvision documents a Gaussian filter as a blur; the book's "sharpen" should not be repeated.

Documentation inconsistencies found along the way (verdicts above are unaffected):

- A6: NVIDIA's Programming Guide says textures give "no performance advantage for non-graphics applications" on any supported GPU; the Best Practices Guide still presents texture fetching as an advantageous alternative in some cases.
- C3 (new constraint, from a local compile): a Metal function constant cannot size a `threadgroup` array ("array size is not a constant expression"), and the spec forbids array-typed function constants.

## Where the backends really differ

One line per cell. Nothing here was run on CUDA or AMD hardware; Metal entries marked "compiled here" were checked with the Metal front end only.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Read-only table set by the host | `__constant__` array at file scope + `cudaMemcpyToSymbol` | Same, with `hipMemcpyToSymbol` | `MTLBuffer` bound as `constant T*`; no symbol copy |
| Table fixed at compile time | Initialised `__constant__` array or `#define` values | Same | Program-scope `constant` array, which must be initialised |
| Documented size | 64 KB (`totalConstMem`); 8 KB cache working set per SM | No number; `totalConstMem` reports the max allocation (CLR source) | No number for constant buffers; `setBytes` under 4 KB |
| Cache behind it | Constant cache; one access per distinct address in a warp | Scalar L1 (sL1D), only for warp-uniform reads, no hit-on-miss | None documented; talks describe preloading bounded data into registers |
| Small struct of parameters | Kernel argument by value, via constant memory, up to 32,764 bytes (CUDA 12.1 and later) | Kernel argument by value, cached in sL1D; no documented limit | `constant T&` argument filled with `setBytes` (under 4 KB) |
| Single shared copy of a parameter | `const __grid_constant__` | Not documented | Not applicable (the argument is already one buffer) |
| Compile-time radius | `#define` or template parameter | Same | Macro (`preprocessorMacros`) or function constant; a function constant cannot size a `threadgroup` array (compiled here) |
| Constant symbol used from another file | Needs `-rdc=true` | Needs `-fgpu-rdc` | Not applicable |
| Read-only load path for the input | `const __restrict__` compiles to `ld.global.nc`; `__ldg` | `__ldg` is a plain load; `__restrict__` is an aliasing hint only | `const device T*`; no separate read-only path documented |
| Cache that section 7.6 relies on | L2 shared by all SMs; size in `l2CacheSize` | L2 shared by the CUs of one die; MI300X: 4 MB per XCD plus a 256 MiB last-level cache | "GPU Last Level Cache shared across all GPU cores" (WWDC20); no size |
| Per-core L1 | Unified with shared memory, 96 to 256 KB per SM | Vector L1 per CU, 16 KB "typical", 32 KiB on MI300X; LDS separate | Separate buffer and texture L1s; no size |
| Hardware boundary handling | Texture modes Border, Clamp, Wrap, Mirror; "no performance advantage" for non-graphics code | Same four `hipAddressMode` values | Sampler `clamp_to_zero`, `clamp_to_edge` (default), `repeat`, `mirrored_repeat`, `clamp_to_border` |
| 32 x 32 block for Fig. 7.12 | 1024 threads on every listed compute capability | Default launch bound is the device maximum; check `maxThreadsPerBlock` | 1024 theoretical; check the pipeline's `maxTotalThreadsPerThreadgroup` |
| Cache hit rates | Nsight Compute Memory Chart | rocprofiler-compute L2 "Hit Rate" (counts hit-on-miss) | Xcode GPU counters ("Last Level Cache") |
| Library convolution (cross-correlation) | cuDNN, `CUDNN_CROSS_CORRELATION` | MIOpen, `miopenConvolution` ("Cross-Correlation convolution") | MPSGraph `convolution2D` (used by PyTorch MPS); MLX's own kernels |
