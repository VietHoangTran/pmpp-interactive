# Chapter 5 fact check: CUDA, HIP, Metal

Checked on 2026-11-20 against the live vendor sources listed under each item.

How to read this file:

- **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims, the verdict is for the item as a whole and the text says which sub-claim failed.
- **Quote** is copied from the source (HTML docs flattened to text, GitHub raw files, Apple's documentation JSON, PDF text). Line breaks inside quotes are removed.
- Versions the sources reported about themselves on the check date: CUDA docs "v13.5" (pages last updated Oct 31 to Nov 2, 2026); ROCm Core SDK 10.2.0 (HIP docs 7.17.0, HIPCC docs 1.1.1); Metal Shading Language Specification "Version 4.1", dated 2026-06-04; Metal Feature Set Tables dated October 23, 2026; local macOS SDK 27.0 headers.
- Not checked: anything by running code on hardware. Every statement is from documentation, headers or project source.

Totals: 22 CONFIRMED, 0 WRONG, 6 PARTLY, 0 COULD NOT VERIFY (28 items). Sub-claims that failed or could not be found are called out inside A1, A8, B6, B7, C3, C7 and collected at the end.

---

## Group A: CUDA

### A1. `cudaDeviceProp` field names and meanings

**Verdict: PARTLY.** All 15 spellings are confirmed. The textbook's description of `sharedMemPerBlock` as the shared memory "in each SM" is wrong: the docs define it per block. The per-SM figure is `sharedMemPerMultiprocessor`.

Fields exactly as they appear in the CUDA Runtime API Reference (v13.5), with the documented one-line meaning:

| Field (exact spelling) | Type | Documented meaning |
|---|---|---|
| `sharedMemPerBlock` | `size_t` | "Shared memory available per block in bytes." |
| `sharedMemPerMultiprocessor` | `size_t` | "Shared memory available per multiprocessor in bytes." |
| `sharedMemPerBlockOptin` | `size_t` | "Per device maximum shared memory per block usable by special opt in." |
| `regsPerBlock` | `int` | "32-bit registers available per block" |
| `regsPerMultiprocessor` | `int` | "32-bit registers available per multiprocessor" |
| `warpSize` | `int` | "Warp size in threads." |
| `maxThreadsPerBlock` | `int` | "Maximum number of threads per block." |
| `maxThreadsPerMultiProcessor` | `int` | "Maximum resident threads per multiprocessor." |
| `maxBlocksPerMultiProcessor` | `int` | "Maximum number of resident blocks per multiprocessor." |
| `multiProcessorCount` | `int` | "Number of multiprocessors on device." |
| `totalGlobalMem` | `size_t` | "Global memory available on device in bytes." |
| `maxThreadsDim` | `int[3]` | "Maximum size of each dimension of a block." |
| `maxGridSize` | `int[3]` | "Maximum size of each dimension of a grid." |
| `major` | `int` | "Major compute capability." |
| `minor` | `int` | "Minor compute capability." |

Spelling trap: the capital P is inconsistent. It is `maxThreadsPerMultiProcessor`, `maxBlocksPerMultiProcessor`, `multiProcessorCount` (capital P) but `sharedMemPerMultiprocessor`, `regsPerMultiprocessor` (lowercase p).

Related fields on the same page: `reservedSharedMemPerBlock` ("Shared memory reserved by CUDA driver per block in bytes."), `totalConstMem` ("Constant memory available on device in bytes.").

The programming guide says the same thing about per-block vs per-SM: "The CUDA runtime API provides functions to query the shared memory size on a per SM basis and a per thread block basis, using the cudaGetDeviceProperties function and investigating the cudaDeviceProp.sharedMemPerMultiprocessor and cudaDeviceProp.sharedMemPerBlock device properties." and "sharedMemPerBlock : The maximum amount of shared memory that can be allocated by a thread block in bytes."

- https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (sections 2.3.3.2 and 2.3.7)

### A2. `cudaGetDeviceProperties` signature

**Verdict: CONFIRMED.** Current signature is `__host__ cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)`.

Quote: "__host__ cudaError_t cudaGetDeviceProperties ( struct cudaDeviceProp * prop , int device ) Returns information about the compute-device. Returns in *prop the properties of device dev."

- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html

### A3. More than 48 KB of shared memory per block; shared memory vs L1 trade-off

**Verdict: CONFIRMED.** Opt in with `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`, use dynamic (`extern __shared__`) shared memory, and pass the byte count at launch. The shared/L1 split is steered with `cudaFuncAttributePreferredSharedMemoryCarveout`; NVIDIA prefers that over `cudaFuncSetCacheConfig`.

Quotes (programming guide, section 3.2.6 "Configuring L1/Shared Memory Balance"):

- "Kernels relying on shared memory allocations over 48 KB per block are architecture-specific. As such they must use dynamic shared memory rather than statically-sized arrays and require an explicit opt-in using cudaFuncSetAttribute as follows."
- Example as printed: `int maxbytes = 98304; // 96 KB` / `cudaFuncSetAttribute(MyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);` / `MyKernel<<<gridDim, blockDim, maxbytes>>>(...);`
- "the L1 and shared memory on an SM use the same physical resource, known as the unified data cache. On most architectures, if a kernel uses little or no shared memory, the unified data cache can be configured to provide the maximal amount of L1 cache allowed by the architecture."
- `cudaFuncSetAttribute(kernel_name, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);` with carveout "as an integer percentage of the maximum supported shared memory capacity of that architecture" or one of `cudaSharedmemCarveoutDefault` (-1), `cudaSharedmemCarveoutMaxL1` (0), `cudaSharedmemCarveoutMaxShared` (100).
- "Where a chosen integer percentage carveout does not map exactly to a supported shared memory capacity, the next larger capacity is used."
- "cudaFuncSetAttribute is interpreted by the driver as a hint, and the driver may choose a different carveout size if required to execute the kernel."
- "Another CUDA API, cudaFuncSetCacheConfig, also allows an application to adjust the balance between L1 and shared memory for a kernel. However, this API set a hard requirements on shared/L1 balance for kernel launch. As a result, interleaving kernels with different shared memory configurations would needlessly serialize launches behind shared memory reconfigurations. cudaFuncSetAttribute is preferred..."

Quotes (runtime API reference):

- `cudaFuncAttributeMaxDynamicSharedMemorySize` (= 8): "The requested maximum size in bytes of dynamically-allocated shared memory. The sum of this value and the function attribute ::sharedSizeBytes cannot exceed the device attribute cudaDevAttrMaxSharedMemoryPerBlockOptin. The maximal size of requestable dynamic shared memory may differ by GPU architecture."
- `cudaFuncAttributePreferredSharedMemoryCarveout` (= 9): "On devices where the L1 cache and shared memory use the same hardware resources, this sets the shared memory carveout preference, in percent of the total shared memory. ... This is only a hint, and the driver can choose a different ratio if required to execute the function."
- `cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig)`, values `cudaFuncCachePreferNone` (default), `cudaFuncCachePreferShared`, `cudaFuncCachePreferL1`, `cudaFuncCachePreferEqual`. "This setting does nothing on devices where the size of the L1 cache and shared memory are fixed."

Two things to know:

- The two NVIDIA documents disagree in tone about `cudaFuncSetCacheConfig`: the reference calls it "only a preference", the guide says it sets "a hard requirements". Both are quoted above; the guide's recommendation is to use the carveout attribute.
- New since older textbooks: on compute capability 10.7 the 328 KB configuration needs a second opt-in. "kernels that use the 328 KB shared-memory configuration must explicitly enable cudaSharedMemoryModeAllowOversizedSharedMemory by setting either the cudaFuncAttributeSharedMemoryMode function attribute or the cudaLaunchAttributeSharedMemoryMode launch attribute."

- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (section 3.2.6)
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Tables 34 and 35)

### A4. Total size of `__constant__` memory

**Verdict: CONFIRMED.** 64 KB, which is 65,536 bytes because the guide defines KB as 1024 bytes. It is the same for every compute capability the guide currently lists (7.5 through 12.x).

Quotes: Table 34 row "Constant memory size | 64 KB" (spanning all 11 columns); "Note that the KB and K units used in the following tables correspond to 1024 bytes (i.e., a KiB) and 1024 respectively."; "Constant memory is small relative to other memories, typically 64KB per device."; "The total amount of constant memory can be queried with the totalConstMem device property element."

Note the guide's own wording is "typically", so the safe course statement is "64 KB on all currently documented compute capabilities; query `totalConstMem`".

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 34)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.3.5)

### A5. Maximum threads per block and block dimensions

**Verdict: CONFIRMED.** 1024 threads per block; block dimensions at most 1024 (x), 1024 (y), 64 (z). Same for all listed compute capabilities 7.5 through 12.x.

Table 33 rows (each value spans all 11 compute-capability columns):

| Row | Value |
|---|---|
| Maximum dimensionality of a thread block | 3 |
| Maximum x- or y-dimensionality of a thread block | 1024 |
| Maximum z-dimension of a thread block | 64 |
| Maximum number of threads per block | 1024 |
| Warp size | 32 |
| Maximum dimensionality of a grid | 3 |
| Maximum x-dimension of a grid | 2^31 - 1 |
| Maximum y- or z-dimension of a grid | 65535 |

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 33)

### A6. Thread block clusters and distributed shared memory

**Verdict: CONFIRMED.** Introduced with compute capability 9.0 (Hopper, H100). Exposed through the compile-time kernel attribute `__cluster_dims__(X,Y,Z)` or at launch through `cudaLaunchKernelEx` with `cudaLaunchAttributeClusterDimension`; inside the kernel through Cooperative Groups (`cluster_group`, `cluster.sync()`).

Quotes:

- "From compute capability 9.0 onward, the CUDA programming model includes an optional level of hierarchy called thread block clusters that are made up of thread blocks. ... thread blocks in a cluster are also guaranteed to be co-scheduled on a GPU Processing Cluster (GPC) in the GPU."
- "a maximum of 8 thread blocks in a cluster is supported as a portable cluster size in CUDA." Larger or smaller limits are "architecture-specific and can be queried using the cudaOccupancyMaxPotentialClusterSize API."
- "A thread block cluster can be enabled in a kernel either using a compile-time kernel attribute using __cluster_dims__(X,Y,Z) or using the CUDA kernel launch API cudaLaunchKernelEx."
- Example as printed: `__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float *output)`. "The grid dimension must be a multiple of cluster size."
- "Thread Block Clusters, introduced in compute capability 9.0 and facilitated by Cooperative Groups, provide the ability for threads in a thread block cluster to access shared memory of all the participating thread blocks in that cluster. This partitioned shared memory is called Distributed Shared Memory ... The size of distributed shared memory is just the number of thread blocks per cluster multiplied by the size of shared memory per thread block."
- Table 32: "Distributed Shared Memory" and "Thread Block Cluster" are "No" for 7.x and 8.x, "Yes" for 9.0, 10.x, 11.0, 12.x.
- Device query fields: `cudaDeviceProp.clusterLaunch` ("Indicates device supports cluster launch."). Function attributes `cudaFuncAttributeRequiredClusterWidth/Height/Depth`, `cudaFuncAttributeNonPortableClusterSizeAllowed`. On compute capability 10.0 and higher there is also `cudaLaunchAttributePreferredClusterDimension`.
- NVIDIA's compute capability table maps 9.0 to "NVIDIA GH200 NVIDIA H200 NVIDIA H100".

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (section 2.1.10)
- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html (section 1.2.2.1.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.3.8)
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html (section 3.1.2)
- https://developer.nvidia.com/cuda/gpus

### A7. Dynamic shared memory declaration and the third launch parameter

**Verdict: CONFIRMED.** The documented form is `extern __shared__ float sharedArray[];` and the third `<<<>>>` argument is a per-block byte count that defaults to 0.

Quotes:

- "the programmer can specify the desired amount of shared memory per thread block in bytes as the third (and optional) argument to the kernel launch in the triple chevron notation like this functionName<<<grid, block, sharedMemoryBytes>>>() . When this parameter is not specified, it defaults to 0."
- "Inside the kernel, the programmer can use the extern __shared__ specifier with empty [] in C++ to declare a variable that will be allocated dynamically at kernel launch." Example: `extern __shared__ float sharedArray[];`
- "One caveat is that a kernel can have only one dynamically allocated shared array." Several logical arrays must be carved manually out of one, respecting alignment.

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.3.2.2)

### A8. NVIDIA H100 published specs

**Verdict: PARTLY.** Three of the five textbook numbers are confirmed as printed, one is a rounding that NVIDIA does not print, and one needs a qualifier.

| Textbook figure | NVIDIA source says | Status |
|---|---|---|
| 3.35 TB/s memory bandwidth | H100 SXM "GPU Memory Bandwidth 3.35TB/s" (H100 NVL: 3.9TB/s) | CONFIRMED for the SXM variant |
| 228 KB shared memory per SM | "Maximum amount of shared memory per SM" = 228 KB for compute capability 9.0; Hopper blog: "Shared memory size / SM: Configurable up to 228 KB" | CONFIRMED |
| 2048 threads per SM | "Maximum number of resident threads per SM" = 2048 for compute capability 9.0 (64 warps of 32) | CONFIRMED |
| 66.9 TFLOPS FP32 | H100 SXM "FP32 67 teraFLOPS" (H100 NVL: 60) | PARTLY: NVIDIA prints 67. I found no NVIDIA page that prints 66.9; a web search for the exact figure found none either. |
| 989 TFLOPS tensor core | H100 SXM "TF32 Tensor Core * 989 teraFLOPS", footnote "* With sparsity" | PARTLY: the number is real, but NVIDIA attaches it to TF32 with sparsity. The same table lists "FP16 Tensor Core * 1,979 teraFLOPS" and "BFLOAT16 Tensor Core * 1,979 teraFLOPS", also with sparsity. |

Notes:

- The per-block shared memory limit for compute capability 9.0 is 227 KB (1 KB less than the per-SM figure).
- Reading 989 as "dense FP16 tensor-core peak" is an inference (half of the with-sparsity 1,979). The live NVIDIA product page does not state the dense figure. NVIDIA's earlier Hopper blog (marked preliminary) lists "Peak FP16 Tensor Core 1000 TFLOPS | 2000 TFLOPS" with the second value "Effective TFLOPS / TOPS using the Sparsity feature", which supports the inference but uses rounded numbers.
- Older NVIDIA material differs: the Hopper architecture blog still shows preliminary "Peak FP32 60 TFLOPS" and "3000 GB/sec" for H100 SXM5. Cite the product page for final numbers.
- I did not fetch the H100 datasheet PDF itself (it is behind a form on resources.nvidia.com); the product page spec table was used instead.

- https://www.nvidia.com/en-us/data-center/h100/ (Product Specifications table)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Tables 33 and 34)
- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- https://developer.nvidia.com/cuda/gpus

### A9. CUTLASS and cuBLAS: name, status, URL

**Verdict: CONFIRMED.**

- cuBLAS: documentation title "cuBLAS 13.5 documentation", at https://docs.nvidia.com/cuda/cublas/
- CUTLASS: repository https://github.com/NVIDIA/cutlass, not archived, last push 2026-11-20. GitHub description: "CUDA Templates and Python DSLs for High-Performance Linear Algebra". Latest tagged release v4.8.1 (2026-10-21); the README on `main` is already headed "CUTLASS 4.9.0 - Nov. 2026". Docs at https://docs.nvidia.com/cutlass/ (redirects to https://docs.nvidia.com/cutlass/latest/).
- Docs intro: "CUTLASS is a collection of CUDA C++ template abstractions and Python domain-specific languages (DSLs) designed to enable high-performance matrix-matrix multiplication (GEMM) and related computations across all levels within CUDA."
- Since CUTLASS 4 the project has two halves: the C++ templates (including CuTe) and Python DSLs ("CuTe DSL"). The README still says "CuTe DSL is currently in public beta and will graduate out of beta by end of summer 2026", which is stale relative to the check date; I did not find a statement of its current status.

Side finding for anyone linking the guide: the old URL `https://docs.nvidia.com/cuda/cuda-c-programming-guide/` now returns 404, and `.../cuda-c-programming-guide/index.html` 301-redirects to the restructured guide at https://docs.nvidia.com/cuda/cuda-programming-guide/. Old section anchors do not carry over. The new guide links the legacy text at https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html

---

## Group B: HIP / ROCm

### B1. Latest ROCm release

**Verdict: CONFIRMED.** ROCm Core SDK 10.2.0, released November 18, 2026.

Release history table as printed: "10.2.0 | November 18, 2026", "10.1.0 | October 5, 2026", "10.0.0 | August 26, 2026", "7.14.1 | September 2, 2026", "7.14.0 | July 15, 2026", "7.2.4 | May 29, 2026".

Notes:

- The 10.x line is the "TheRock"-based "ROCm Core SDK"; the 7.2.x line is what AMD's docs call "ROCm legacy". Sources now live in the monorepo `ROCm/rocm-systems`, tagged `therock-10.2` (published 2026-11-18).
- The GitHub releases list of `ROCm/ROCm` stops at `rocm-7.14.0` (2026-07-16), so it is not a reliable indicator of the latest version.
- The HIP documentation shipped with 10.2.0 is titled "HIP 7.17.0 Documentation".

- https://rocm.docs.amd.com/en/latest/release/versions.html
- https://rocm.docs.amd.com/en/latest/about/release-notes.html
- https://github.com/ROCm/rocm-systems/releases

### B2. `hipDeviceProp_t` field names

**Verdict: CONFIRMED.** All 15 CUDA names from A1 exist in `hipDeviceProp_t` with identical spelling, including the same inconsistent capital P. `maxSharedMemoryPerMultiProcessor` also exists, as an additional HIP-only member, not as a replacement for `sharedMemPerMultiprocessor`.

Struct excerpt from `projects/hip/include/hip/hip_runtime_api.h` at tag `therock-10.2` (members in declaration order, unrelated members omitted):

```c
typedef struct hipDeviceProp_t {
  char name[256];                   ///< Device name.
  ...
  size_t totalGlobalMem;            ///< Size of global memory region (in bytes).
  size_t sharedMemPerBlock;         ///< Size of shared memory per block (in bytes).
  int regsPerBlock;                 ///< Registers per block.
  int warpSize;                     ///< Warp size.
  ...
  int maxThreadsPerBlock;           ///< Max work items per work group or workgroup max size.
  int maxThreadsDim[3];             ///< Max number of threads in each dimension (XYZ) of a block.
  int maxGridSize[3];               ///< Max grid dimensions (XYZ).
  ...
  int major;  ///< Major compute capability version. ...
  int minor;  ///< Minor compute capability version. ...
  ...
  int multiProcessorCount;       ///< Number of multi-processors. When the GPU works in Compute
                                 ///< Unit (CU) mode, this value equals the number of CUs;
                                 ///< when in Workgroup Processor (WGP) mode, this value equels
                                 ///< half of CUs, because a single WGP contains two CUs.
  ...
  int maxThreadsPerMultiProcessor;  ///< Maximum resident threads per multi-processor.
  ...
  size_t sharedMemPerMultiprocessor;  ///< Amount of shared memory available per multiprocessor.
  int regsPerMultiprocessor;          ///< registers available per multiprocessor
  ...
  size_t sharedMemPerBlockOptin;  ///< Per device m ax shared mem per block usable by special opt in
  ...
  int maxBlocksPerMultiProcessor;      ///< Max number of blocks on CU
  ...
  /* HIP Only struct members */
  char gcnArchName[256];                    ///< AMD GCN Arch Name. HIP Only.
  size_t maxSharedMemoryPerMultiProcessor;  ///< Maximum Shared Memory Per CU. HIP Only.
  ...
} hipDeviceProp_t;
```

Notes:

- `major`/`minor` do not mean an NVIDIA compute capability on AMD. Header comment: "For example, a value of 11 would correspond to Navi III (RDNA3). See the arch feature flags for portable ways to query feature caps."
- Query function: `static inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId)` (it forwards to the versioned `hipGetDevicePropertiesR0600`).
- The struct on the `develop` branch differs from the tag only in one comment.
- This confirms the names exist. It does not tell you what values an AMD device reports in them; I did not check that.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (struct starts at line 263)

### B3. Kernel launch syntax on AMD: `<<<>>>` or `hipLaunchKernelGGL`

**Verdict: CONFIRMED** that CUDA-style `kernel<<<grid, block, shmem, stream>>>(args)` is accepted directly by HIP-Clang. `hipLaunchKernelGGL` is not required; it is a macro that expands to `<<<>>>`.

Quotes:

- Language extensions page: "Kernels are called using the triple chevron <<<>>> syntax known from CUDA, but HIP also supports the hipLaunchKernelGGL macro."
- Porting guide: "HIP-Clang supports kernel launching using either the triple chevron (<<<>>>) syntax, hipLaunchKernel, or hipLaunchKernelGGL. The last option is a macro that expands to the <<<>>> syntax by default. It can also be turned into a template by defining HIP_TEMPLATE_KERNEL_LAUNCH."
- Header (`amd_hip_runtime.h`): `kernelName<<<(numBlocks), (numThreads), (memPerBlock), (streamId)>>>(__VA_ARGS__);`
- The docs' own examples use it: `kernel<<<dim3(gridSize), dim3(blockSize), sharedMemSize, 0>>>();`
- hipify-clang option table: "--cuda-kernel-execution-syntax | Keep CUDA kernel launch syntax (default)" and "--hip-kernel-execution-syntax | Transform CUDA kernel launch syntax to a regular HIP function call".

What the docs recommend: neither is labelled preferred. The triple chevron is presented first and is what the examples and the HIPIFY default use. If you do use `hipLaunchKernelGGL`, its first five arguments are kernel name, `dim3 gridDim`, `dim3 blockDim`, `size_t dynamicShared`, `hipStream_t`, and templated kernel names need `HIP_KERNEL_NAME(...)`.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h (lines 221 to 226)
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/hipify-clang-cmd.html

### B4. Same spelling for `__shared__`, `extern __shared__`, `__syncthreads()`, `__global__`, `__device__`, index built-ins

**Verdict: CONFIRMED.** All are documented with the CUDA spelling: `__host__`, `__device__`, `__global__`, `__shared__`, `__constant__`, `__managed__`, `__syncthreads()`, `threadIdx`, `blockIdx`, `blockDim`, `gridDim`, `warpSize`.

Quotes:

- "Variables marked with __shared__ are only accessible by threads within the same block and have the lifetime of that block."
- "In order to dynamically allocate shared memory during runtime, but before the kernel is launched, the variable has to be declared extern, and the kernel launch has to specify the needed amount of extern shared memory in the launch configuration."
- Docs example (`extern_shared_memory.hip`): `extern __shared__ int shared_array[];` then `kernel<<<dim3(gridSize), dim3(blockSize), sharedMemSize, 0>>>();`
- "void __syncthreads() simply synchronizes the threads of a block." Also documented: `__syncthreads_count`, `__syncthreads_and`, `__syncthreads_or`.
- "These built-ins are of type dim3" (for `threadIdx`, `blockIdx`, `blockDim`, `gridDim`).
- Header: `#define __global__ __attribute__((global))`, `#define __shared__ __attribute__((shared))`, `#define __device__ __attribute__((device))`.

Differences that matter for a course:

- The built-ins need the header: "These built-ins are not implicitly defined by the compiler, the hip_runtime.h header has to be included instead."
- "Unlike CUDA, HIP does not support dynamic parallelism, meaning that kernels can not be called from the device."
- The docs example declares the `extern __shared__` array at file scope, where NVIDIA's guide shows it inside the kernel.
- Older HIP material used a `HIP_DYNAMIC_SHARED` macro for this. I did not find it in the headers I checked at `therock-10.2` (`hip_runtime_api.h`, `amd_hip_runtime.h`, `host_defines.h`); I did not search the whole tree, so treat its removal as unverified.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/tools/example_codes/extern_shared_memory.hip
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/host_defines.h (lines 374 to 378)

### B5. Warp (wavefront) size on AMD under HIP

**Verdict: CONFIRMED.** 64 on gfx9 architectures (GCN/CDNA, which includes AMD Instinct), 32 on gfx10 and above (RDNA, AMD Radeon). The porting guide says not to hard-code either value.

Quotes:

- Porting guide: "Code should not assume a warp size of 32 or 64, as AMD GPU architectures have different warp sizes. The warpSize built-in should be used in device code, while the host can query it during runtime via the device properties."
- Language extensions: "warpSize should not be assumed to be a specific value in portable HIP applications. NVIDIA devices return 32 for this variable; AMD devices return 64 for gfx9 and 32 for gfx10 and above. HIP doesn't support warpSize of 64 on gfx10 and above. While code that assumes a warpSize of 32 can run on devices with a warpSize of 64, it only utilizes half of the compute resources."
- "On AMD hardware, this is referred to as wavefront size".
- Performance guidelines: "AMD Instinct GPUs execute threads in warps of 64, while AMD Radeon GPUs execute threads in warps of 32."
- "Prior to ROCm 7.0, the warpSize parameter was a compile-time constant. Starting with ROCm 7.0, it is early folded by the compiler". On the host it is not a compile-time constant; query it with `hipDeviceGetAttribute(&w, hipDeviceAttributeWarpSize, deviceId)`.
- Concrete hazard the guide documents: 32-bit lane masks. "GCN/CDNA architectures have a warp size of 64, threadIdx.x % warpSize and val in the example may obtain values greater than 31", so `1 << n` must become `1ull << n` (or use `lane_mask_t`).

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html (section "warpSize")
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "warpSize")
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html

### B6. Compiler driver and source file extension

**Verdict: PARTLY.** `hipcc` still exists and is still documented, but AMD now labels it legacy and tells new projects to call `amdclang++` directly. The conventional HIP source extension is `.hip`.

Quotes:

- HIPCC docs front page: "hipcc and hipconfig are legacy tools. New projects should use amdclang++ directly and CMake-native HIP language support instead. See the hipcc and hipconfig migration guide for step-by-step migration guidance."
- "hipcc is a compiler driver utility that wraps amdclang++ and automatically injects the flags needed to compile HIP source code and link to the HIP runtime. C++ executable versions of hipcc and hipconfig compiler driver utilities are provided for backward compatibility."
- Migration guide, before and after: `hipcc -o my_kernel my_kernel.cpp --offload-arch=gfx1100` becomes `amdclang++ --hip-path=${ROCM_PATH} --hip-device-lib-path=${ROCM_PATH}/lib/llvm/amdgcn/bitcode -x hip --offload-arch=gfx1100 -o my_kernel my_kernel.cpp`. "--offload-arch=native can be used to automatically target the GPUs present on the build machine."
- HIP FAQ: "HIP-Clang is a Clang/LLVM-based compiler used to compile HIP programs for AMD platforms. The executable is named amdclang++ on Linux and clang++ on Windows."
- Extension: "Source code written in the HIP dialect of C++ typically uses the .hip extension." (ROCm 7.2.4 "Using CMake" page). The HIP tutorial compiles `main.hip`: `amdclang++ ./HIP-Basic/saxpy/main.hip -o saxpy -I ./Common -lamdhip64 -L /opt/rocm/lib -O2`.
- `.cpp` also works but then needs `-x hip`: the macros `__HIPCC__` and `__HIP__` are "Defined when compiling .hip files or specifying -x hip".

Inconsistency inside AMD's docs: the HIP porting guide still presents both without the "legacy" label ("you can use amdclang++, also called HIP-Clang, or you can use hipcc which offers a higher-level compiler interface"). The HIPCC project's own page is the more specific and more recent statement.

- https://rocm.docs.amd.com/projects/HIPCC/en/latest/index.html
- https://rocm.docs.amd.com/projects/HIPCC/en/latest/migration.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/faq.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/saxpy.html

### B7. CMake support

**Verdict: PARTLY.** `enable_language(HIP)` and `project(... LANGUAGES HIP)` are confirmed (CMake 3.21 and later), as are `find_package(hip)`, `hip::device` and `hip::host`. The `-D CMAKE_PREFIX_PATH=/opt/rocm` instruction is only documented on the legacy 7.2.4 "Using CMake" page; that page does not exist in the current 10.2.0 docs, and the 10.x install prefix has changed.

Quotes, CMake's own docs:

- CMake 3.21 release notes: "CMake learned to support HIP as a first-class language that can be enabled via the project() and enable_language() commands."
- `enable_language` page: "HIP | Added in version 3.21."
- CMake 3.21 release notes, 3.21.3 section: "hipcc may no longer be used as a HIP compiler because it interferes with flags CMake needs to pass to Clang. Use Clang directly."

Quotes, current ROCm docs (hipcc migration guide, 10.2.0):

- "CMake-native HIP language support (available since CMake 3.21) for CMake projects, using find_package(hip CONFIG) and enable_language(HIP)"
- "For projects using CMake's native HIP language support (recommended), set CMAKE_HIP_COMPILER instead: cmake -DCMAKE_HIP_COMPILER=amdclang++ ..."
- "hipconfig --cxxflags | Use hip::host and hip::device imported targets"

Quotes, ROCm 7.2.4 "Using CMake" page (legacy line, page dated 2026-02-19):

- Canonical single-source setup: `cmake_minimum_required(VERSION 3.21) # HIP language support requires 3.21` / `cmake_policy(VERSION 3.21.3...3.27)` / `project(MyProj LANGUAGES HIP)` / `add_executable(MyApp Main.hip)`
- "AMD packaged ROCm installs can typically be added to the config file search paths such as: Windows: -D CMAKE_PREFIX_PATH=${env:HIP_PATH} Linux: -D CMAKE_PREFIX_PATH=/opt/rocm"
- "The find_package(hip) provides the hip::host imported target" for C++ code that only calls the HIP API, and "The find_package(hip) provides the hip::device imported target to add all the flags necessary for device compilation" when compiling device code in CXX language mode.
- "When mixing such CXX sources with HIP sources holding device-code, link only to hip::host."
- For existing `.cu` files: `set_source_files_properties(MyLib.cu PROPERTIES LANGUAGE HIP)`. Target architectures via `CMAKE_HIP_ARCHITECTURES`.

What changed in 10.x (transition guide): "Installation directory | /opt/rocm/core-10.2 | /opt/rocm-7.2/" and "the amdrocm meta package configures update-alternatives and provides backward-compatible symlinks for /opt/rocm/bin, /opt/rocm/lib, and other /opt/rocm/ directories." So `/opt/rocm` probably still resolves for package-manager installs, but I found no 10.2.0 page that states `CMAKE_PREFIX_PATH=/opt/rocm`, and tarball or wheel installs live elsewhere (`ROCM_PATH`).

- https://cmake.org/cmake/help/latest/release/3.21.html
- https://cmake.org/cmake/help/latest/command/enable_language.html
- https://rocm.docs.amd.com/projects/HIPCC/en/latest/migration.html
- https://rocm.docs.amd.com/en/docs-7.2.4/conceptual/cmake-packages.html (the same path under `/en/latest/` returns 404)
- https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html

### B8. HIPIFY tools

**Verdict: CONFIRMED.** The tools are still named `hipify-clang` and `hipify-perl`.

Quotes:

- "HIPIFY is a ROCm tool to help developers migrate GPU programming from NVIDIA's CUDA language to AMD's HIP C++ programming language for use on AMD GPUs."
- "hipify-clang: A clang-based tool that parses CUDA code and converts it to HIP code. It handles syntax changes, API calls, and kernel launch differences."
- "hipify-perl: A simpler tool generated from hipify-clang that replaces CUDA API calls with HIP equivalents for basic code translation needs. hipify-perl is useful for simple CUDA programs, but offers less error detection when running into issues during translation."
- "HIPIFY does not automatically convert all CUDA code into HIP code seamlessly. ... there are some limitations and areas where manual intervention is often required."
- A third related tool is mentioned: "hipify_torch is a related tool that also translates CUDA source code into portable HIP C++."
- hipify-clang needs a CUDA installation: "--cuda-path=<directory> | CUDA installation path. The CUDA path is required for hipify-clang".

- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/index.html
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/hipify-clang-cmd.html

### B9. llama.cpp: how the HIP backend is built from the CUDA source

**Verdict: CONFIRMED.** The path is unchanged (`ggml/src/ggml-cuda/vendors/hip.h`, last commit 2026-11-16). The header is a list of `#define`s that rename CUDA and cuBLAS identifiers to HIP and hipBLAS ones, and the HIP backend compiles the same `ggml-cuda/*.cu` files as language HIP.

Representative lines from `hip.h`:

```c
#define HIP_ENABLE_WARP_SYNC_BUILTINS 1
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
...
#define cublasHandle_t hipblasHandle_t
#define cublasSgemm hipblasSgemm
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaFree hipFree
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
...
#if !defined(__HIP_PLATFORM_AMD__)
#error "The HIP backend supports only AMD targets"
#endif // !defined(__HIP_PLATFORM_AMD__)

#define __CUDA_ARCH__ 1300
```

Selection in `ggml/src/ggml-cuda/common.cuh`:

```c
#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#elif defined(GGML_USE_MUSA)
#include "vendors/musa.h"
#else
#include "vendors/cuda.h"
#endif // defined(GGML_USE_HIP)
```

Build side, `ggml/src/ggml-hip/CMakeLists.txt`: `cmake_minimum_required(VERSION 3.21)` and `enable_language(HIP)`; `find_package(hip REQUIRED)`, `find_package(hipblas REQUIRED)`, `find_package(rocblas REQUIRED)`; `file(GLOB GGML_SOURCES_ROCM "../ggml-cuda/*.cu")`; `set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)`; `add_compile_definitions(GGML_USE_HIP)`; link `hip::host roc::rocblas roc::hipblas`. It requires "At least ROCM/HIP V6.3" and warns "Setting hipcc as the C++ compiler is legacy behavior. Prefer setting the HIP compiler directly." It also carries the comment "# TODO: do not use CUDA definitions for HIP".

Things `hip.h` does not need to rename, consistent with B3 and B4: `__global__`, `__device__`, `__shared__`, `threadIdx`, the `<<<>>>` launch. There is no such `#define` in the file. The one visible warp-width accommodation is the lane-mask type: `hip.h` has `typedef uint64_t ggml_cuda_thread_mask_t;` where `vendors/cuda.h` has `typedef uint32_t ggml_cuda_thread_mask_t;` (compare B5).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/hip.h
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hip/CMakeLists.txt

---

## Group C: Metal

Sources: Metal Shading Language Specification Version 4.1 (PDF dated 2026-06-04), cited as "MSL spec"; Apple developer documentation pages; Metal framework headers in the macOS 27.0 SDK installed on this machine; Metal Feature Set Tables PDF dated October 23, 2026.

### C1. Address space for per-threadgroup scratch memory

**Verdict: CONFIRMED.** The keyword is `threadgroup`. It can be used for a local variable inside a kernel function body and for a kernel parameter that is a pointer.

Quotes (MSL spec section 4.4 "Threadgroup Address Space"):

- "Threads in a threadgroup can work together by sharing data in threadgroup memory, which is faster on most devices than sharing data in device memory. Use the threadgroup address space to: Allocate a threadgroup variable in a kernel, mesh, or object function. Define a kernel, fragment, or object function parameter that's a pointer to a threadgroup address."
- "Threadgroup variables in a kernel, mesh, or object function only exist for the lifetime of the threadgroup that executes the kernel."
- Example as printed:

```cpp
kernel void
my_kernel(threadgroup float *sharedParameter [[threadgroup(0)]],
          …)
{
     // Allocate a float in the threadgroup address space.
     threadgroup float sharedFloat;
     // Allocate an array of 10 floats in the threadgroup address space.
     threadgroup float sharedFloatArray[10];
     ...
}
```

Notes:

- A non-kernel helper can take a `threadgroup` pointer only when called from a kernel: "The threadgroup attribute is supported here for the pointer l_data only if foo is called by a kernel function".
- Program scope: the spec says both "The address space for a variable at program scope needs to be constant" (section 4) and "In Metal 3.2 and later, you can declare device, constant, and threadgroup buffers, texture, and sampler in the program scope (see section 5.9)" (section 5.2). The second refers to bound buffers with explicit binding indexes, not ordinary globals. For a course, stick to the two forms in the example.
- Size limit from the feature set tables: "Maximum total threadgroup memory allocation" is 32 KB for GPU families Apple4 through Apple11 (16 KB for Apple3). That is far below CUDA's 48 KB default and 227 KB opt-in on H100.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 4.4, page 120)
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf (GPU family limits table, page 7)

### C2. Barrier function

**Verdict: CONFIRMED.** `void threadgroup_barrier(mem_flags flags)`, called as `threadgroup_barrier(mem_flags::mem_threadgroup)`, declared in `<metal_compute>`.

Quotes (MSL spec section 6.10.1, Tables 6.12 and 6.13):

- "void threadgroup_barrier(mem_flags flags): All threads in a threadgroup executing the kernel, fragment, mesh, or object need to execute this function before any thread can continue execution beyond the threadgroup_barrier."
- New overload: `void threadgroup_barrier(mem_flags flags, memory_order order = memory_order_seq_cst, thread_scope scope = thread_scope_threadgroup)`, "All OS: Metal 4.1 and later."
- "A barrier function (threadgroup_barrier or simdgroup_barrier) acts as an execution and memory barrier."
- "If threadgroup_barrier (or simdgroup_barrier) is inside a conditional statement and if any thread enters the conditional statement and executes the barrier function, then all threads in the threadgroup (or SIMD-group) need to enter the conditional and execute the barrier function."
- "On Apple silicon, a thread that has ended no longer participates or blocks remaining threads at a barrier."

`mem_flags` values and what each guarantees:

| Value | Spec description |
|---|---|
| `mem_none` | "The flag sets threadgroup_barrier or simdgroup_barrier to only act as an execution barrier and doesn't apply a Memory fence." |
| `mem_device` | "The flag ensures the GPU correctly orders the memory operations to device memory for threads in the threadgroup or SIMD-group." |
| `mem_threadgroup` | "The flag ensures the GPU correctly orders the memory operations to threadgroup memory for threads in a threadgroup or SIMD-group." |
| `mem_texture` | "...to texture memory for threads in a threadgroup or SIMD-group for a texture with the read_write access qualifier." (macOS: Metal 1.2 and later) |
| `mem_threadgroup_imageblock` | "...to threadgroup imageblock memory for threads in a threadgroup or SIMD-group." |
| `mem_object_data` | "...to object_data memory for threads in the threadgroup or SIMD-group." |

There is also `simdgroup_barrier(mem_flags)` with the same shape, scoped to one SIMD-group.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 6.10.1, pages 215 to 217)

### C3. Device and pipeline thread properties

**Verdict: PARTLY.** The meanings are confirmed. The API page for `threadExecutionWidth` does not describe it as a SIMD group or warp width; only the MSL spec ties "thread execution width" to a SIMD-group. No Apple source uses the word "warp".

| Property | Documented meaning |
|---|---|
| `MTLDevice.maxThreadgroupMemoryLength` | "The maximum threadgroup memory available to a compute kernel, in bytes." (macOS 10.13+) |
| `MTLDevice.maxThreadsPerThreadgroup` (`MTLSize`) | "The maximum number of threads along each dimension of a threadgroup." Discussion: "This property reports the maximum thread group size the device can support for a trivial shader. This size isn't guaranteed for all shaders. For the actual thread group size of a specific compute shader, see the maxTotalThreadsPerThreadgroup property of your compute pipeline state." |
| `MTLComputePipelineState.threadExecutionWidth` | "The number of threads that the GPU executes simultaneously." Discussion: "For better performance, when dispatching a compute command, make the number of threads in the threadgroup a multiple of threadExecutionWidth." Header comment: "For most efficient execution, the threadgroup size should be a multiple of this when executing the kernel." |
| `MTLComputePipelineState.maxTotalThreadsPerThreadgroup` | "The maximum number of threads in a threadgroup that you can dispatch to the pipeline." Discussion: "This value never changes, but may be different for different pipeline objects." |
| `MTLComputePipelineState.staticThreadgroupMemoryLength` | "The length, in bytes, of statically allocated threadgroup memory." (macOS 10.13+) |

The SIMD-group link, from the MSL spec section 5.2.3.6:

- "Within a compute unit, a threadgroup is partitioned into multiple smaller groups for execution. The execution width of the compute unit, referred to as the threads_per_simdgroup, determines the recommended size of this smaller group. For best performance, make the total number of threads in the threadgroup a multiple of the threads_per_simdgroup."
- Attribute table: `threads_per_simdgroup`: "The thread execution width of a SIMD-group (compute unit)." The older MSL attribute `thread_execution_width` has the same description and is marked "Deprecated as of Metal 3 - use threads_per_simdgroup."

So "threadExecutionWidth is the SIMD-group width, the closest Metal analogue of a warp" is a reasonable teaching statement, but the warp comparison is yours, not Apple's. Apple's own worked example uses a width of 16 purely as an illustration ("a thread execution width of 16"); I found no documented fixed value, so read it from the pipeline state.

Feature set table limits: "Maximum threads per threadgroup" is 1024 for Apple4 through Apple11 (512 for Apple2 and Apple3), with the footnote "The values in this row are the theoretical maximum number of threads per threadgroup. Check the actual maximum by inspecting the MTLComputePipelineState.maxTotalThreadsPerThreadgroup".

- https://developer.apple.com/documentation/metal/mtldevice/maxthreadgroupmemorylength
- https://developer.apple.com/documentation/metal/mtldevice/maxthreadsperthreadgroup
- https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth
- https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/maxtotalthreadsperthreadgroup
- https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/staticthreadgroupmemorylength
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 5.2.3.6)
- Local header: `Metal.framework/Headers/MTLComputePipeline.h` (macOS 27.0 SDK)

### C4. Kernel input attributes for thread indexing

**Verdict: CONFIRMED.** All five names are exact. Each accepts `ushort`, `ushort2`, `ushort3`, `uint`, `uint2` or `uint3`.

| Attribute | Types | Spec description |
|---|---|---|
| `thread_position_in_grid` | ushort, ushort2, ushort3, uint, uint2, or uint3 | "The thread's position in an N-dimensional grid of threads." |
| `thread_position_in_threadgroup` | same | "The thread's unique position within a threadgroup" |
| `threadgroup_position_in_grid` | same | "The threadgroup's unique position within a grid." |
| `threads_per_threadgroup` | same | "The thread execution width of a threadgroup." |
| `threadgroups_per_grid` | same | "The number of threadgroups in a grid." |

Also in the same table: `threads_per_grid` ("The grid size."), `thread_index_in_threadgroup` (ushort or uint, "The scalar index of a thread within a threadgroup."), `dispatch_threads_per_threadgroup`, `threads_per_simdgroup`, `thread_index_in_simdgroup`, `simdgroup_index_in_threadgroup`.

CUDA mapping the spec itself supports: "The thread position in the grid (thread_position_in_grid) is: (gx, gy) = (wx * Sx + lx, wy * Sy + ly)", which is `blockIdx * blockDim + threadIdx`. Metal hands you the global index directly.

Type rules: the types of `thread_position_in_threadgroup`, `threads_per_threadgroup` and `dispatch_threads_per_threadgroup` "need to match"; the types of `thread_position_in_grid` and `threads_per_grid` "need to match"; if vector types are used, the component counts must match.

Edge case: "It is therefore possible that the actual threadgroup size of a specific threadgroup may be smaller than the threadgroup size specified in the dispatch. The [[threads_per_threadgroup]] attribute specifies the actual threadgroup size for a given threadgroup executing the kernel. The [[dispatch_threads_per_threadgroup]] attribute is the threadgroup size specified at dispatch."

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 5.2.3.6, Table 5.8, pages 153 to 158)

### C5. `dispatchThreadgroups` vs `dispatchThreads`

**Verdict: CONFIRMED.** `dispatchThreadgroups:threadsPerThreadgroup:` takes a count of threadgroups (grid = groups x group size, always uniform). `dispatchThreads:threadsPerThreadgroup:` takes the total thread grid and lets Metal create partial edge threadgroups. Nonuniform threadgroup sizes are supported on Apple GPU family Apple4 and later.

Quotes:

- `dispatchThreadgroups(_:threadsPerThreadgroup:)`: "Encodes a compute dispatch command using a grid aligned to threadgroup boundaries." "Metal calculates the number of threads in a grid by multiplying threadsPerThreadgroup by threadgroupsPerGrid." "If the size of your data doesn't match the size of the grid, perform boundary checks in your compute function to avoid accessing data out of bounds." Tip: "Prefer using dispatchThreads for your kernel calls on Apple4 and later Apple GPUs."
- `dispatchThreads(_:threadsPerThreadgroup:)` (macOS 10.13+): "Encodes a compute command using an arbitrarily sized grid." "Metal calculates the number of threadgroups needed, providing partial threadgroups if necessary." Warning: "Use this method only if the device your app is running on supports nonuniform threadgroup sizes. Check for device capabilities with supportsFamily(_:)".
- Feature set tables: "Nonuniform threadgroup size | Metal 3 & 4 | Apple4".
- "You don't need to check a thread's position in a grid if you use the dispatchThreads(_:threadsPerThreadgroup:) technique."

Apple's page has a slip: the `threadgroupsPerGrid` parameter of `dispatchThreadgroups` is described as "the number of threads for each grid dimension". The discussion text and the parameter name make clear it is the number of threadgroups.

Not verified: which Apple GPU family a given Mac chip belongs to. The table lists the feature under both the "Metal 3 & 4" programming-model family and hardware family Apple4, and footnotes "All Metal 3 and Metal 4 devices support at least the limits listed in the Apple7 column", which implies every Metal 3 capable Mac qualifies; I did not look up a chip-to-family table.

- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/dispatchthreadgroups(_:threadsperthreadgroup:)
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/dispatchthreads(_:threadsperthreadgroup:)
- https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf (feature availability table, page 4)

### C6. `MTLResourceStorageModeShared` and unified memory

**Verdict: CONFIRMED.** On Apple silicon the CPU and GPU share system memory, a shared-mode buffer is visible to both with no copy, and no synchronization API call exists for shared mode. You are still responsible for ordering: do not touch the buffer from one side until work scheduled on the other side has completed.

Quotes:

- "Apple GPUs have a unified memory model in which the CPU and the GPU share system memory. However, CPU and GPU access to that memory depends on the storage mode you choose for your resources. The shared mode defines system memory that both the CPU and the GPU can access."
- `storageModeShared`: "The CPU and GPU share access to the resource, allocated in system memory." "This is the default storage mode for MTLBuffer instances on integrated GPUs and both MTLBuffer and MTLTexture instances on Apple silicon GPUs."
- "When either the CPU or GPU changes the contents of the resource, you're responsible for synchronizing access to the texture from the other participant. Ensure that all changes you schedule on either the CPU or GPU for a resource that uses shared memory complete before accessing that resource on the other processor."
- `hasUnifiedMemory` (macOS 10.15+): "A Boolean value that indicates whether the GPU shares all of its memory with the CPU." "A GPU with unified memory (true) is typically an integrated GPU. A GPU with dedicated memory (false) may take additional time to synchronize managed resources or copy data into private GPU resources."
- The explicit calls belong to managed mode only: `didModifyRange(_:)`: "If you write information to a buffer created with the managed storage mode, you need to call this method". And `MTLStorageMode.managed` is now deprecated (macOS 10.11 to 27.0): "Managed storage has no effect on Apple Silicon, use Shared storage instead".
- `contents()`: "A pointer to the shared copy of the buffer data, or NULL for buffers allocated with a private resource storage mode".

In practice "ensure changes complete" means waiting for the command buffer (`waitUntilCompleted` or a completion handler) before reading results on the CPU. That phrasing is mine; the quote above is Apple's.

- https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared
- https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
- https://developer.apple.com/documentation/metal/setting-resource-storage-modes
- https://developer.apple.com/documentation/metal/mtldevice/hasunifiedmemory
- https://developer.apple.com/documentation/metal/mtlstoragemode/managed

### C7. `MTLCompileOptions`: `preprocessorMacros`, `fastMathEnabled`, `mathMode`

**Verdict: PARTLY.** `preprocessorMacros` is confirmed. `fastMathEnabled` does default to YES, but it has been deprecated since macOS 15 / iOS 18 in favour of `mathMode`. MSL does default to non-IEEE floating-point optimizations, but Apple's sources disagree on exactly which mode is the default on Apple silicon.

`preprocessorMacros`:

- "A list of preprocessor macros to apply when compiling the library source." "Define the macros as a dictionary where each key is a string, and the values can be either an NSString or NSNumber instance. The default value is nil."
- Declaration: `@property (nullable, readwrite, copy, nonatomic) NSDictionary <NSString *, NSObject *> *preprocessorMacros;`
- Used with `newLibraryWithSource:options:error:` (Swift `makeLibrary(source:options:)`): "Synchronously creates a Metal library instance by compiling the functions in a source string." Limitation: "Because there's no search path to find other functions, the source may only import the Metal default library." So runtime-compiled source cannot `#include` your own headers.

`fastMathEnabled`:

- Header: `@property (readwrite, nonatomic) BOOL fastMathEnabled API_DEPRECATED("Use mathMode instead", macos(10.11, 15.0), ios(8.0, 18.0));` and "fastMathEnabled defaults to YES."
- Web docs: "The default value is true." Deprecation: "Use mathMode instead."

`mathMode` (macOS 15.0+), enum `MTLMathMode { MTLMathModeSafe = 0, MTLMathModeRelaxed = 1, MTLMathModeFast = 2 }`, and `mathFloatingPointFunctions` (`MTLMathFloatingPointFunctionsFast = 0`, `...Precise = 1`):

- "If fastMathEnabled is true, the system sets mathMode to fast and mathFloatingPointFunctions to fast. If fastMathEnabled is false, the system sets mathMode to safe and mathFloatingPointFunctions to precise."

The default, three Apple sources, three answers:

| Source | Statement |
|---|---|
| Web docs, `MTLMathMode.relaxed` | "This is the default for Apple silicon devices." (`MTLMathMode.fast`: "This is the default for Intel and AMD devices.") |
| SDK header (macOS 27.0), `mathMode` | "Sets the floating-point arithmetic optimizations. Default depends on the language standard version." |
| MSL spec 4.1 section 1.6.3 (command-line compiler) | "-fmetal-math-mode=<fast, relaxed, safe> ... The default is fast." and "-fmetal-math-fp32-functions=<fast\|precise> ... The default is fast." |

What the modes mean (MSL spec): fast assumes "no NaNs, no INFs, no signed zeros, allow reciprocal, allow reassociation, and FP contract to be fast"; relaxed is the same "but honors INFs and NaNs"; safe "disables unsafe floating-point optimizations". Legacy flags: `-ffast-math` equals fp32-functions=fast plus math-mode=fast; `-fno-fast-math` equals fp32-functions=precise plus math-mode=safe.

Safe statement for the course: with default options MSL is not IEEE-strict (reassociation and contraction are allowed in both candidate defaults), so results can differ from a CPU reference or from CUDA; set `mathMode = .safe` (or compile with `-fno-fast-math`) when you need bit-comparable output. Do not state which of fast or relaxed is the default without testing on the target OS.

- https://developer.apple.com/documentation/metal/mtlcompileoptions/preprocessormacros
- https://developer.apple.com/documentation/metal/mtlcompileoptions/fastmathenabled
- https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode
- https://developer.apple.com/documentation/metal/mtlmathmode/relaxed
- https://developer.apple.com/documentation/metal/mtlmathmode/fast
- https://developer.apple.com/documentation/metal/mtldevice/makelibrary(source:options:)
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 1.6.3, pages 15 to 16)
- Local header: `Metal.framework/Headers/MTLLibrary.h` (macOS 27.0 SDK), lines 269 to 338

### C8. Dynamically sized threadgroup memory

**Verdict: CONFIRMED.** Metal has a direct analogue of `extern __shared__` plus a launch-time byte count: the kernel declares a `threadgroup` pointer parameter with `[[threadgroup(index)]]`, and the host sets its size with `setThreadgroupMemoryLength:atIndex:` before dispatch.

Quotes:

- `setThreadgroupMemoryLength(_:index:)`: "Configures the size of a block of threadgroup memory." Parameters: length is "The size of the threadgroup memory, in bytes, which needs to be a multiple of 16 bytes."; index is "The index in the threadgroup memory argument table using this allocation."
- "Before using any threadgroup memory, call this method to configure the threadgroup memory argument table. Kernels accessing their arguments from threadgroup memory have the [[threadgroup]] attribute."
- "The sum of all threadgroup memory allocations (whether made using this method or directly in the shader) can't exceed the device limits for threadgroup memory."
- MSL spec, argument kinds: "A buffer shared between threads in a threadgroup - a pointer to a type in the threadgroup address space that can only be used as arguments for kernel functions." Binding attribute: "Threadgroup buffers: [[threadgroup(index)]]".
- Kernel-side declaration as printed in the spec: `threadgroup int *ldata [[threadgroup(0)]],`
- The Metal 4 encoder has the same call: `MTL4ComputeCommandEncoder.setThreadgroupMemoryLength(_:index:)` (macOS 26.0+), index being "the index of the argument you annotate with attribute [[threadgroup(index)]] in the shader function."

Differences from CUDA worth teaching: several independent dynamic blocks are allowed (one per index), where CUDA allows one `extern __shared__` array; lengths must be multiples of 16 bytes; the total cap is 32 KB on Apple4 and later.

One oddity on Apple's page: it says "Check threadgroup memory limits with the staticThreadgroupMemoryLength property", but that property is the statically allocated length of a pipeline (C3). The device limit is `MTLDevice.maxThreadgroupMemoryLength`.

- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setthreadgroupmemorylength(_:index:)
- https://developer.apple.com/documentation/metal/mtl4computecommandencoder/setthreadgroupmemorylength(_:index:)
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 4.4, 5.2 and 5.2.1)

### C9. `MTLCommandBuffer.GPUStartTime` / `GPUEndTime`

**Verdict: CONFIRMED.** Both are host times in seconds (`CFTimeInterval`), available on macOS 10.15 and later (iOS 10.3 and later). They are 0 until the GPU has finished the command buffer.

Quotes:

- `gpuStartTime`: "The host time, in seconds, when the GPU starts command buffer execution."
- `gpuEndTime`: "The host time, in seconds, when the GPU finishes execution of the command buffer."
- "You can calculate how much time the GPU spends running a command buffer by subtracting this value from gpuEndTime. Both values are relative to system mach time."
- "The GPU start and end times remain 0.0 until the GPU finishes running the command buffer. Check this value after the waitUntilCompleted() method returns, or within a completion handler passed to the addCompletedHandler(_:) method."
- Header: `@property (readonly) CFTimeInterval GPUStartTime API_AVAILABLE(macos(10.15), macCatalyst(13.0), ios(10.3));` (same for `GPUEndTime`).

Granularity is one command buffer, not one kernel dispatch. Related: `kernelStartTime` / `kernelEndTime` measure CPU-side scheduling by the driver.

- https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime
- https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpuendtime
- Local header: `Metal.framework/Headers/MTLCommandBuffer.h` (macOS 27.0 SDK), lines 286 to 294

### C10. How MLX and llama.cpp hold their Metal kernels

**Verdict: CONFIRMED** (answered from current source; both projects are hybrids, with opposite defaults).

llama.cpp (`master`, Metal backend last changed 2026-11-19):

- Default build: MSL source is embedded in the binary and compiled at runtime. `ggml/CMakeLists.txt` line 243: `option(GGML_METAL_EMBED_LIBRARY "ggml: embed Metal library" ${GGML_METAL})`, so it is on whenever Metal is on. `ggml/src/ggml-metal/CMakeLists.txt` concatenates each kernel source with its headers and embeds the text with an assembler `.incbin` under symbols `_ggml_metallib_<kind>_start/_end`. `ggml/src/ggml-metal/ggml-metal-device.m` then calls `[device newLibraryWithSource:src options:options error:&error]` with `options.preprocessorMacros = prep`.
- With `GGML_METAL_EMBED_LIBRARY=OFF`: the build runs `xcrun ... metal` (flags `-O3`, or `-fno-fast-math -fno-inline` under `GGML_METAL_SHADER_DEBUG`) to produce `default.metallib`, loaded with `newLibraryWithURL:`. If that file is missing it falls back: "default.metallib not found, loading kernel sources", optionally from `GGML_METAL_PATH_RESOURCES`.
- Path change: the single file `ggml/src/ggml-metal/ggml-metal.metal` no longer exists on `master` (HTTP 404). Kernels are now split across `ggml/src/ggml-metal/kernels/` (41 files, for example `mul_mm.metal`, `mul_mv.metal`, `softmax.metal`, `fa_f32.metal`, `common.h`).

MLX (`main`, `device.cpp` last changed 2026-11-01):

- Default build: precompiled metallib. `CMakeLists.txt` line 44: `option(MLX_METAL_JIT "Use JIT compilation for Metal kernels" OFF)`. `mlx/backend/metal/kernels/CMakeLists.txt` compiles each `.metal` file with `xcrun -sdk macosx metal ${METAL_FLAGS} -c ${SRCFILE} ... -o ${TARGET}.air` and links them into `${MLX_METAL_PATH}/mlx.metallib`. The flags include `-fno-fast-math`. `mlx/backend/metal/device.cpp` searches for `mlx.metallib` next to the binary, then in `Resources/`, then a SwiftPM bundle `default.metallib`, then the compiled-in `METAL_PATH`.
- Runtime compilation also exists in every build: `Device::build_library_` in `device.cpp` calls `device_->newLibrary(ns_code, options, &error)` on a source string, with `set_compile_options` choosing `MTL::MathModeSafe/Relaxed/Fast`. It serves kernels generated at runtime (the `make_jit_source` preambles in `mlx/backend/metal/CMakeLists.txt`, for example `indexing/gather`, `indexing/scatter`, `unary_ops`, `binary_ops`; `compiled.cpp`; `custom_kernel.cpp`). With `MLX_METAL_JIT=ON` the block `if(NOT MLX_METAL_JIT)` in `kernels/CMakeLists.txt` is skipped, so most kernels (`binary`, `copy`, `unary`, `softmax`, the steel GEMM and conv kernels, and others) move from the metallib to runtime compilation, selected by `jit_kernels.cpp` vs `nojit_kernels.cpp`.

Not verified: I read the build files and the loader, not every kernel's call site, so the exact split of which MLX kernels are always JIT is taken from the CMake lists only.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/CMakeLists.txt
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m
- https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal/kernels
- https://github.com/ml-explore/mlx/blob/main/CMakeLists.txt
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/CMakeLists.txt
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/CMakeLists.txt
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device.cpp

---

## Summary of corrections

Items where a commonly assumed fact turned out WRONG or PARTLY:

- A1: `sharedMemPerBlock` is shared memory per block, not per SM ("Shared memory available per block in bytes."); the per-SM field is `sharedMemPerMultiprocessor`. The textbook sentence is wrong.
- A1: capitalisation is inconsistent and easy to get wrong: `maxThreadsPerMultiProcessor`, `maxBlocksPerMultiProcessor`, `multiProcessorCount` but `sharedMemPerMultiprocessor`, `regsPerMultiprocessor`.
- A3: NVIDIA now recommends `cudaFuncSetAttribute` with `cudaFuncAttributePreferredSharedMemoryCarveout` over `cudaFuncSetCacheConfig`; and compute capability 10.7's 328 KB configuration needs an extra `cudaSharedMemoryModeAllowOversizedSharedMemory` opt-in.
- A8: NVIDIA publishes H100 SXM FP32 as 67 teraFLOPS; "66.9" appears in no NVIDIA source I found.
- A8: 989 teraFLOPS is NVIDIA's "TF32 Tensor Core" figure with sparsity, not an unqualified tensor-core peak; FP16/BF16 tensor core is listed as 1,979 with sparsity.
- A8/A9: the old programming guide URL `docs.nvidia.com/cuda/cuda-c-programming-guide/` is gone (404, or redirect to the restructured `cuda-programming-guide/`); section anchors from older references no longer resolve.
- B2: `maxSharedMemoryPerMultiProcessor` is not HIP's renaming of `sharedMemPerMultiprocessor`; HIP has both, and all 15 CUDA field names keep their CUDA spelling.
- B3: `hipLaunchKernelGGL` is not required on AMD; `kernel<<<grid, block, shmem, stream>>>(args)` compiles directly, and hipify-clang keeps that syntax by default.
- B6: `hipcc` and `hipconfig` are now labelled legacy; AMD tells new projects to use `amdclang++` directly (with `-x hip` or a `.hip` file) and CMake's HIP language.
- B7: `-D CMAKE_PREFIX_PATH=/opt/rocm` is documented only on the legacy ROCm 7.2.4 "Using CMake" page, which is missing from the 10.2.0 docs; ROCm 10.2 installs to `/opt/rocm/core-10.2` with compatibility symlinks.
- B7: `hipcc` cannot be used as the compiler for CMake's HIP language (CMake 3.21.3 and later); use Clang/`amdclang++` as `CMAKE_HIP_COMPILER`.
- C3: Apple's API docs do not describe `threadExecutionWidth` as a SIMD-group or warp width; only the MSL spec links "thread execution width" to a SIMD-group (`threads_per_simdgroup`), and "warp" is never Apple's term.
- C7: `fastMathEnabled` is deprecated (macOS 15 / iOS 18); use `mathMode`. Its default was YES.
- C7: the default math mode on Apple silicon is documented three different ways (web docs: relaxed; MSL spec for the command-line compiler: fast; SDK header: "depends on the language standard version"). Either way MSL is not IEEE-strict by default.
- C8: Apple's `setThreadgroupMemoryLength` page points to `staticThreadgroupMemoryLength` for the limit; the device limit is `MTLDevice.maxThreadgroupMemoryLength` (32 KB on Apple4 and later per the feature set tables).
- C10: llama.cpp's Metal kernels are no longer in one `ggml-metal.metal` file; they are split across `ggml/src/ggml-metal/kernels/*.metal`, and by default they are embedded as source and compiled at runtime, while MLX by default ships a precompiled `mlx.metallib`.
