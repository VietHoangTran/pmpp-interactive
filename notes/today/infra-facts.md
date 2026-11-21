# Infrastructure fact check for the later chapters: CUDA, HIP, Metal, real backends

Checked on 2026-11-20 against the live vendor sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims the verdict is for the item as a whole and the text says which sub-claim failed. A verdict judges the assumption a course author would most likely bring to the item (stated in the item title or first line), not the vendor.
- **Quote** is copied from the source (HTML docs flattened to text, GitHub raw files at a tag or on the default branch, Apple's documentation JSON, PDF text). Line breaks inside quotes are removed.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide and Runtime API "v13.5" (pages last updated Oct 31 and Nov 02, 2026); CCCL docs at `nvidia.github.io/cccl/unstable/` (latest tagged release v3.5.0, 2026-10-07); ROCm Core SDK 10.2.0 = tag `therock-10.2` in `ROCm/rocm-systems`, `ROCm/rocm-libraries` and `ROCm/libhipcxx` (HIP docs "HIP 7.17.0", libhipcxx docs "3.0.2", rocWMMA docs "2.2.2"); Metal Shading Language Specification "Version 4.1" dated 2026-06-04; Metal Feature Set Tables dated October 23, 2026.
- Not checked: anything by running code on hardware. Every statement is from documentation, headers or project source. Nothing here was compiled against ROCm or CUDA.
- Group B answers, for each A item, "does HIP have it, how is it spelled, what differs". Names that differ and are confirmed went into `labs/common/include/pmpp/vendors/hip.h`; each B item says which.

Totals: 24 CONFIRMED, 1 WRONG, 15 PARTLY, 1 COULD NOT VERIFY (41 items: A1 to A12, B1 to B12, C1 to C13, D1 to D4). The WRONG and PARTLY items are collected under "Summary of corrections".

---

## Group A: CUDA

### A1. Legacy atomic functions: list, types, memory spaces

**Verdict: CONFIRMED.** Eleven functions. They work on global and shared memory. `atomicAdd` does support `float` and `double`; `atomicMin`/`atomicMax` are integer only; `atomicCAS` is integer only (there is no `float` overload, so a float compare-and-swap goes through an `unsigned` reinterpretation).

Types exactly as listed in the CUDA Programming Guide v13.5, section 5.4.5.1:

| Function (signature as printed) | Supported types |
|---|---|
| `T atomicAdd(T* address, T val);` | "int, unsigned, unsigned long long, float, double, __half2, __half."; "__nv_bfloat16, __nv_bfloat162 on devices of compute capability 8.x and higher."; "float2, float4 on devices of compute capability 9.x and higher, and only supported for global memory addresses." |
| `T atomicSub(T* address, T val);` | "int, unsigned" |
| `unsigned atomicInc(unsigned* address, unsigned val);` | computes "old >= val ? 0 : (old + 1)" |
| `unsigned atomicDec(unsigned* address, unsigned val);` | computes "(old == 0 \|\| old > val) ? val : (old - 1)" |
| `T atomicAnd(T* address, T val);` | "int, unsigned, unsigned long long." |
| `T atomicOr(T* address, T val);` | "int, unsigned, unsigned long long." |
| `T atomicXor(T* address, T val);` | "int, unsigned, unsigned long long." |
| `T atomicMin(T* address, T val);` | "int, unsigned, unsigned long long, long long." |
| `T atomicMax(T* address, T val);` | "int, unsigned, unsigned long long, long long." |
| `T atomicExch(T* address, T val);` | "int, unsigned, unsigned long long, float." plus a template for 128-bit types on compute capability 9.x and higher |
| `T atomicCAS(T* address, T compare, T val);` | "int, unsigned, unsigned long long, unsigned short." plus a template for 128-bit types on compute capability 9.x and higher |

Every function "returns the old value."

Quotes:

- "Legacy atomic functions perform atomic read-modify-write operations on a 32-, 64-, or 128-bit word stored in global or shared memory."
- "Atomic functions can only be used in device functions."
- "The atomic functions described in this section have a memory ordering of cuda::std::memory_order_relaxed and are only atomic at a particular thread scope: Atomic APIs without a suffix, for example atomicAdd, are atomic at scope cuda::thread_scope_device. Atomic APIs with the _block suffix, for example, atomicAdd_block, are atomic at scope cuda::thread_scope_block. Atomic APIs with the _system suffix, for example, atomicAdd_system, are atomic at scope cuda::thread_scope_system if they meet particular conditions."
- "Unlike built-in atomic functions, legacy atomic functions only ensure atomicity and do not introduce synchronization points (fences)."
- Vector types are not atomic as a whole: "The atomicity of atomicAdd() applied to vector types, for example __half2 or float4, is guaranteed separately for each of the components".
- Performance, section 3.2.2: "If an atomic instruction executed by a warp reads, modifies, and writes to the same location in global memory for more than one of the threads of the warp, each read/modify/write to that location occurs and they are all serialized, but the order in which they occur is undefined."

Things a lesson should say:

- NVIDIA now calls these "Legacy" and lists five families of atomics. Its hint: "Using the Extended CUDA C++ atomic functions provided by libcu++ is recommended for efficiency, safety, and portability." (see A2).
- A third family exists that older textbooks do not mention, compiler built-ins such as `__nv_atomic_fetch_add(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM)`: "They have been available since CUDA 12.8."
- For a float max or min there is no legacy function. The guide's own recipe is a CAS loop: "any atomic operation can be implemented based on atomicCAS() (Compare and Swap)", shown for float add by casting the pointer to `unsigned*`.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.5 and 5.4.5.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (section 3.2.2)

### A2. `<cuda/atomic>`: `cuda::atomic`, `cuda::atomic_ref`, scopes, member functions, memory orders, toolkit version

**Verdict: PARTLY.** Everything about the API is confirmed. Not found: which CUDA Toolkit version first shipped `<cuda/atomic>`. One inconsistency between two NVIDIA documents about the cluster scope.

Declarations as printed in the libcu++ docs:

```cpp
// Defined in header <cuda/atomic>:
template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
class cuda::atomic;

template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
class cuda::atomic_ref;
```

Thread scopes, exact enumerator names (libcu++ "Memory model" page):

```cpp
namespace cuda {
enum thread_scope {
  thread_scope_system,
  thread_scope_device,
  thread_scope_cluster,
  thread_scope_block,
  thread_scope_thread
};
}  // namespace cuda
```

Quotes:

- "The class template cuda::atomic is an extended form of cuda::std::atomic that takes an additional cuda::thread_scope argument ... It has the same interface and semantics as cuda::std::atomic, with the following additional operations." The additions listed: `cuda::atomic_thread_fence`, `cuda::atomic::fetch_min` ("Atomically find the minimum of the stored value and a provided value."), `cuda::atomic::fetch_max`.
- So the member functions are those of `std::atomic` (`load`, `store`, `exchange`, `compare_exchange_weak`, `compare_exchange_strong`, `fetch_add`, `fetch_sub`, `fetch_and`, `fetch_or`, `fetch_xor`, ...) plus `fetch_min` and `fetch_max`. libcu++ defers the standard part to cppreference: "libcu++ does not provide its own documentation for Standard Library features."
- Header signature on `main`: `fetch_max(const _Tp& __op, memory_order __m = memory_order_seq_cst) noexcept` (same for `fetch_min`). The default order is `seq_cst`, as in the standard; pass `relaxed` explicitly for a counter or a running maximum.
- `cuda::atomic_ref`: "This class additionally deviates from the standard by being backported to C++11." Limits: "sizeof(T) <= 16", "The referenced object must be aligned to its size: alignof(T) == sizeof(T)". "No object or subobject of an object referenced by an atomic_ref shall be concurrently referenced by any other atomic_ref that has a different Scope."
- Small types are slow: "Types of T, where sizeof(T) < 4, are not natively supported by the underlying hardware. For these types atomic operations are emulated and will be drastically slower."
- "On architectures before SM90, thread_scope_cluster operations are widened to thread_scope_device."
- Guide example (section 3.2.4.1): `__shared__ cuda::atomic<int, cuda::thread_scope_block> counter;` ... `int old_value = counter.fetch_add(1, cuda::memory_order_relaxed);`
- Guide example (section 2.3.5.1): `cuda::atomic_ref<float, cuda::thread_scope_device> result_ref(result);`

Memory-order spellings: both are valid. `cuda/std/__atomic/order.h` on `main` declares the enumerators in `cuda::std` and then re-exports them:

```cpp
_CCCL_BEGIN_NAMESPACE_CUDA
using memory_order = ::cuda::std::memory_order;
inline constexpr memory_order memory_order_relaxed = ::cuda::std::memory_order_relaxed;
...
_CCCL_END_NAMESPACE_CUDA
```

The programming guide writes `cuda::memory_order_relaxed` in its `cuda::atomic` examples and `cuda::std::memory_order_relaxed` when it describes the legacy functions. They are the same value.

Sub-claims that did not hold or could not be found:

- Toolkit version: COULD NOT VERIFY. The current docs only say "libcu++ is included in the NVIDIA HPC SDK and the CUDA Toolkit. It is on the default include path." I checked the archived release notes for CUDA 11.2, 11.3 and 11.4 for "libcu++" and found nothing. What is documented: the CCCL README's compatibility table has the row "3.x | 12.x, 13.x (only latest patch releases)" under "Supports CUDA Toolkit Version", and since CUDA 13.0 the headers moved from `${CTK_ROOT}/include/cuda/` to `${CTK_ROOT}/include/cccl/cuda/` (nvcc still finds `<cuda/atomic>`; a host-only compile needs the new include path).
- Cluster scope: the libcu++ page lists `cuda::thread_scope_cluster` and shows `cuda::atomic<int, cuda::thread_scope_cluster> d;`. The programming guide's thread-scope table (section 3.2.3) has rows for `cuda::thread_scope_thread`, `cuda::thread_scope_block`, `cuda::thread_scope_device`, `cuda::thread_scope_system` and leaves the C++ name blank on the `.cluster` row. The header on `main` defines `thread_scope_cluster = 3`, so the libcu++ page is the accurate one.
- Both libcu++ pages say the scope argument is "defaulted to cuda::std::thread_scope_system" while the declaration right above says `cuda::thread_scope_system`. Use the declaration.

- https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic.html
- https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html
- https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html
- https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__atomic/atomic.h
- https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/std/__atomic/order.h
- https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/std/__atomic/scopes.h
- https://nvidia.github.io/cccl/unstable/libcudacxx/setup/getting.html
- https://nvidia.github.io/cccl/unstable/cccl/3.0_migration_guide.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (sections 3.2.3 and 3.2.4.1)

Note for links: the programming guide still links `https://nvidia.github.io/cccl/libcudacxx/...` in places; those URLs return 404. The working prefix is `https://nvidia.github.io/cccl/unstable/libcudacxx/...`.

### A3. Warp intrinsics: signatures, types, out-of-range source lane

**Verdict: CONFIRMED.**

Signatures exactly as printed (programming guide v13.5, sections 5.4.4.2 and 5.4.6):

```cpp
T __shfl_sync     (unsigned mask, T value, int      srcLane,  int width=warpSize);
T __shfl_up_sync  (unsigned mask, T value, unsigned delta,    int width=warpSize);
T __shfl_down_sync(unsigned mask, T value, unsigned delta,    int width=warpSize);
T __shfl_xor_sync (unsigned mask, T value, int      laneMask, int width=warpSize);

int      __all_sync   (unsigned mask, int predicate);
int      __any_sync   (unsigned mask, int predicate);
unsigned __ballot_sync(unsigned mask, int predicate);

unsigned __activemask();
void __syncwarp(unsigned mask = 0xFFFFFFFF);
```

Types for the shuffles: "T can be: int, unsigned, long, unsigned long, long long, unsigned long long, float or double. __half and __half2 with the cuda_fp16.h header included. __nv_bfloat16 and __nv_bfloat162 with the cuda_bf16.h header included."

Out-of-range behaviour, one rule per function:

| Function | Documented behaviour |
|---|---|
| `__shfl_sync` | "If srcLane is outside the range [0, width - 1], the result corresponds to the value held by the srcLane % width, which is within the same subsection." |
| `__shfl_up_sync` | "The source lane index will not wrap around the value of width, so the lower delta lanes will remain unchanged." |
| `__shfl_down_sync` | "the ID number of the source lane will not wrap around the value of width and so the upper delta lanes will effectively remain unchanged." |
| `__shfl_xor_sync` | with `width < warpSize`: "each group of width consecutive threads are able to access elements from earlier groups. However, if they attempt to access elements from later groups of threads their own value of value will be returned." |

Other rules worth quoting:

- "Threads may only read data from another thread that is actively participating in the intrinsics. If the target thread is inactive, the retrieved value is undefined."
- "width must be a power of two in the range [1, warpSize], namely 1, 2, 4, 8, 16, or 32. Other values will produce undefined results."
- Mask rules ("Warp __sync Intrinsic Constraints", 5.4.6.6): "Each calling thread must have its corresponding bit set in the mask." "All non-exited threads specified in the mask must execute the intrinsic with the same mask value." Violations are "invalid, such as kernel hang, or undefined".
- The vote functions and shuffles "do not provide any memory ordering." `__syncwarp` does: "the call to __syncwarp(mask) strongly happens before ... any warp thread named in mask is unblocked from the wait or exits."
- `__activemask()` "cannot be used to determine which warp lanes execute a given branch."
- `__ballot_sync` "returns an integer whose Nth bit is set if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active."
- Also documented and useful for softmax/argmax kernels: `T __reduce_add_sync(unsigned mask, T value);`, `__reduce_min_sync`, `__reduce_max_sync` ("T can be an unsigned or signed integer"), and `unsigned __match_any_sync(unsigned mask, T value);`.
- NVIDIA's hint: "It is suggested to use the libcu++ cuda::device::warp_shuffle() functions as a generalized and safer alternative" and "the CUB Warp-Wide “Collective” Primitives ... whenever possible".

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.4.2, 5.4.6.1 to 5.4.6.6)

### A4. Cooperative groups, cooperative launch, thread block clusters

**Verdict: PARTLY.** Names and signatures are confirmed. Two traps: (1) the reference says `cg::reduce` accepts only `coalesced_group` and `thread_block_tile`, while the guide's own example passes a `thread_block`; (2) multi-device cooperative launch was removed in CUDA 13.

`cudaLaunchCooperativeKernel`, runtime API reference:

```cpp
// C API
__host__ cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                                 void **args, size_t sharedMem, cudaStream_t stream)
// C++ API
template<class T>
__host__ cudaError_t cudaLaunchCooperativeKernel(T *func, dim3 gridDim, dim3 blockDim, void **args,
                                                 size_t sharedMem = 0, cudaStream_t stream = 0)
```

Requirements as documented:

- "The device on which this kernel is invoked must have a non-zero value for the device attribute cudaDevAttrCooperativeLaunch."
- "The total number of blocks launched cannot exceed the maximum number of blocks per multiprocessor as returned by cudaOccupancyMaxActiveBlocksPerMultiprocessor (or cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors as specified by the device attribute cudaDevAttrMultiProcessorCount."
- "The kernel cannot make use of CUDA dynamic parallelism."
- "If the kernel has N parameters the args should point to array of N pointers."
- Guide: "Only devices with compute capability of 6.0 and higher are supported. In addition, you need to be running on either of these: The Linux platform without MPS; The Linux platform with MPS and on a device with compute capability 7.0 or higher; The latest Windows platform".
- "to be able to synchronize across the grid, you need to use the cooperative launch API" and "it is necessary to use, instead of the <<<...>>> execution configuration syntax, the cudaLaunchCooperativeKernel CUDA runtime launch API".
- `grid_group` has `bool is_valid() const: Returns whether the grid_group can synchronize`.

Group handles (namespace `cooperative_groups`, header `<cooperative_groups.h>`):

| Accessor | Documented meaning |
|---|---|
| `this_thread_block()` | "Returns the handle to a group containing all threads in current thread block." (`thread_block`) |
| `this_grid()` | "Returns the handle to a group containing all threads in the grid." (`grid_group`) |
| `coalesced_threads()` | "Returns the handle to a group of currently active threads in a warp." (`coalesced_group`); "makes no guarantee about which threads are returned ... or that they will stay coalesced throughout execution." |
| `this_cluster()` | "Returns the handle to a group of threads in the current cluster." (`cluster_group`); "assumes a 1x1x1 cluster when a non-cluster grid is launched. Requires Compute Capability 9.0 or greater" |

Tiles: `template <unsigned int Size, typename ParentT = void> class thread_block_tile;` built with `cg::thread_block_tile<8> my_subgroup = cg::tiled_partition<8>(my_group);`. "Size must be a power of 2 and less than or equal to 1024." `tiled_partition`: "The allowed parent groups are thread_block or thread_block_tile." Tile members include `shfl`, `shfl_up`, `shfl_down`, `shfl_xor`, `any`, `all`, `ballot`, `match_any`, `match_all` (the shuffles other than `shfl`, and `ballot`/`match_*`, are "available only for sizes lower or equal to 32").

Reduce (`<cooperative_groups/reduce.h>`):

```cpp
template <typename TyGroup, typename TyArg, typename TyOp>
auto reduce(const TyGroup& group, TyArg&& val, TyOp&& op) -> decltype(op(val, val));
```

- Operators: `cg::plus`, `cg::less` (minimum), `cg::greater` (maximum), `cg::bit_and`, `cg::bit_or`, `cg::bit_xor`. "These must be constructed, hence the TyVal template argument is required, i.e. plus<int>()."
- "group: Valid group types are coalesced_group and thread_block_tile."
- "Only 4B types are accelerated by hardware." and "cg::plus<float> fails to match with an accelerator and instead performs a standard shuffle based reduction".
- The inconsistency: section 4.4.6.1 of the same guide prints `cg::thread_block my_group = cg::this_thread_block(); ... int sum = cg::reduce(my_group, val, cg::plus<int>());` and calls it "a block-wide sum reduction". I did not compile it. Teach the tile form (`tiled_partition<32>` then `cg::reduce(tile, ...)`), which both pages agree on.

Removed: "As of CUDA 13, cooperative groups can no longer be used for multi-device synchronization." and "Multi-device launch APIs and related references for Cooperative Groups have been removed as of CUDA 13." (no `this_multi_grid()`, no `cudaLaunchCooperativeKernelMultiDevice`).

Clusters (already checked as `ch05-facts.md` A6, re-read today): "A thread block cluster can be enabled in a kernel either using a compile-time kernel attribute using __cluster_dims__(X,Y,Z) or using the CUDA kernel launch API cudaLaunchKernelEx." Printed example: `__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)`. `cluster_group` members include `sync()`, `block_rank()`, `num_blocks()`, `dim_blocks()`, `block_index()`, `map_shared_rank(T *addr, int rank)` ("Obtain the address of a shared memory variable of another block in the cluster"). "The APIs are available on all hardware with Compute Capability 9.0+."

- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html (section 5.7.3)
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EXECUTION.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__HIGHLEVEL.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (section 2.1.10)

### A5. Constant memory: `cudaMemcpyToSymbol`, size limit

**Verdict: CONFIRMED.**

```cpp
// C++ API (what cudaMemcpyToSymbol(coeffs, h_coeffs, sizeof(h_coeffs)) resolves to)
template<class T>
__host__ cudaError_t cudaMemcpyToSymbol(const T &symbol, const void *src, size_t count,
                                        size_t offset = 0,
                                        enum cudaMemcpyKind kind = cudaMemcpyHostToDevice)
// C API
__host__ cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count,
                                        size_t offset=0, enum cudaMemcpyKind kind=cudaMemcpyHostToDevice)
```

Quotes:

- "Copies count bytes from the memory area pointed to by src to the memory area offset bytes from the start of symbol symbol. The memory areas may not overlap. symbol is a variable that resides in global or constant memory space. kind can be either cudaMemcpyHostToDevice or cudaMemcpyDeviceToDevice."
- Guide example: `__constant__ float coeffs[4];` ... `cudaMemcpyToSymbol(coeffs, h_coeffs, sizeof(h_coeffs));`
- "In C++, variables or arrays are declared with the __constant__ specifier outside any kernel or function."
- Size: "Constant memory is small relative to other memories, typically 64KB per device." and "The total amount of constant memory can be queried with the totalConstMem device property element." Table 34 gives "Constant memory size | 64 KB" for every listed compute capability (see `ch05-facts.md` A4).
- Device side: "constant data cannot be altered from within a running kernel ... as references to __constant__ space are read-only."

- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__HIGHLEVEL.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.3.5)

### A6. Vector loads: is casting `float*` to `float4*` the documented way to get a 16-byte load?

**Verdict: COULD NOT VERIFY.** No NVIDIA page I read documents the cast, or promises that reading a `float4` compiles to one 16-byte load. What is documented is the type's layout and the allocator's alignment.

Documented facts:

- Table 49 "Alignment Requirements": `float4` has size 16 and alignment 16 (`float3`: size 12, alignment 4; `int4, uint4`: 16 and 16; `float2`: 8 and 8).
- "Vector types are structures. Their first, second, third, and fourth components are accessible through the x, y, z, and w fields".
- Best Practices Guide: "Memory allocated through the CUDA Runtime API, such as via cudaMalloc(), is guaranteed to be aligned to at least 256 bytes."
- The cache-hint loads accept vector types: "The function __ldg() performs a read-only L1/Tex cache load. It supports all C++ fundamental types, CUDA vector types (except x3 components), and extended floating-point types".
- `__builtin_assume_aligned(const void* ptr, size_t align)` exists to tell the compiler about alignment; NVIDIA suggests `cuda::std::assume_aligned()`.

What NVIDIA's own code does (sections 4.6.4 and 4.6.6 of the guide): the kernel takes `const float4 *__restrict__ in` as its parameter type and reads `float4 v = in[chunk];`. For the store, where a single 128-bit operation is required for correctness, it does not rely on the compiler: "The producer writes each aligned 16-byte chunk with a single 128-bit store, for example the PTX st.relaxed.gpu.global.b128 instruction that the example wraps in a small helper."

Safe statements for the course: declare the buffer as `float4*` (or cast a pointer that is known to be 16-byte aligned), and say that a single wide load is what compilers produce in practice, to be confirmed by looking at the generated code, not something the documentation guarantees. Reading through a `float4*` that is not 16-byte aligned breaks the type's alignment requirement.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (Tables 48 and 49, sections 5.4.8 and 5.4.9.2)
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 10.2.1.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/fine-grained-synchronization.html (sections 4.6.4 and 4.6.6)

### A7. `#pragma unroll` and `__forceinline__`

**Verdict: CONFIRMED.**

- "The compiler unrolls small loops with a known trip count by default. However, the #pragma unroll directive can be used to control the unrolling of any given loop. This directive must be placed immediately before the loop and only applies to that loop."
- Optional argument: "If it is absent, the loop will be completely unrolled if its trip count is constant. If it evaluates to 0 or 1, the loop will not be unrolled. If it is a non-positive integer or greater than INT_MAX, the pragma will be ignored, and a warning will be issued." Forms printed: `#pragma unroll`, `#pragma unroll 1`, `#pragma unroll (Count + 1)`.
- "__noinline__: Instructs nvcc not to inline the function. __forceinline__: Forces nvcc to inline the function within a single translation unit. __inline_hint__: Enables aggressive inlining across translation units when using Link-Time Optimization." "These specifiers are mutually exclusive."
- Scope note: the optimisation hints are "always available in SIMT device code, but are not supported in Tile code."

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.1.3 and 5.4.9.1)

### A8. Streams, `cudaMemcpyAsync`, pinned memory, events for timing

**Verdict: CONFIRMED.**

Signatures (runtime API reference v13.5):

```cpp
cudaError_t cudaStreamCreate(cudaStream_t *pStream)
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)   // cudaStreamDefault 0x00, cudaStreamNonBlocking 0x01
cudaError_t cudaStreamSynchronize(cudaStream_t stream)
cudaError_t cudaStreamDestroy(cudaStream_t stream)
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream = 0)
cudaError_t cudaMallocHost(void **ptr, size_t size)
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
cudaError_t cudaFreeHost(void *ptr)
cudaError_t cudaEventCreate(cudaEvent_t *event)
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)       // cudaEventDefault 0x00, cudaEventBlockingSync 0x01, cudaEventDisableTiming 0x02
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream=0)
cudaError_t cudaEventSynchronize(cudaEvent_t event)
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
cudaError_t cudaEventDestroy(cudaEvent_t event)
```

`cudaHostAlloc` flags, as documented: `cudaHostAllocDefault` ("defined to be 0 and causes cudaHostAlloc() to emulate cudaMallocHost()"), `cudaHostAllocPortable`, `cudaHostAllocMapped`, `cudaHostAllocWriteCombined`. "All of these flags are orthogonal to one another". The C++ API now also has `cudaMallocHost(void **ptr, size_t size, unsigned int flags)`.

Quotes:

- "CUDA has a default stream, and operations and kernel launches without a specific stream are queued into this default stream."
- "cudaMemcpyAsync() is asynchronous with respect to the host, so the call may return before the copy is complete. ... If kind is cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost and the stream is non-zero, the copy may overlap with operations in other streams."
- "In order for memory copies involving CPU memory to be carried out asynchronously, the host buffers must be pinned and page-locked. cudaMemcpyAsync() will function correctly if host memory which is not pinned and page-locked is used, but it will revert to a synchronous behavior which will not overlap with other work."
- `cudaMallocHost`: "Allocates size bytes of host memory that is page-locked and accessible to the device." "Page-locking excessive amounts of memory with cudaMallocHost() may degrade system performance".
- Timing: "When an event reaches the front of a stream it records a timestamp. By surrounding a kernel in a stream with two events we can get an accurate timing of the duration of the kernel execution". The printed pattern is `cudaEventRecord(start, stream); kernel<<<grid, block, 0, stream>>>(...); cudaEventRecord(stop, stream); cudaStreamSynchronize(stream); cudaEventElapsedTime(&elapsedTime, start, stop);`
- `cudaEventElapsedTime`: "Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds)." If an event has not completed, `cudaErrorNotReady` is returned. "If either event was last recorded in a non-NULL stream, the resulting time may be greater than expected".

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html (section 2.6.3)
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__STREAM.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__TYPES.html

### A9. Occupancy API

**Verdict: CONFIRMED.**

```cpp
// C API
__host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func,
                                                                   int blockSize, size_t dynamicSMemSize)
// C++ API
template<class T>
__host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, T func,
                                                                   int blockSize, size_t dynamicSMemSize)
```

"Returns in *numBlocks the maximum number of active blocks per streaming multiprocessor for the device function." Guide usage: `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, my_kernel, numThreads, 0);`. Related: `cudaOccupancyMaxPotentialBlockSize` (C++ only), `cudaOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, const void *func, int numBlocks, int blockSize)`, `cudaOccupancyMaxActiveClusters`, `cudaOccupancyMaxPotentialClusterSize`.

- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__OCCUPANCY.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__HIGHLEVEL.html

### A10. Shared memory banks

**Verdict: CONFIRMED.** 32 banks, each 32 bits (4 bytes) wide, for every compute capability the guide lists.

- "Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. Each bank has a bandwidth of 32 bits per clock cycle."
- "When multiple threads in the same warp attempt to access different elements in the same bank, a bank conflict occurs. In this case, the access to the data in that bank will be serialized".
- "The two exceptions to this scenario happen when multiple threads in the same warp are accessing (either reading or writing) the same shared memory location. For read accesses, the word is broadcast to the requesting threads. For write accesses, each shared memory address is written by only one of the threads (which thread performs the write is undefined)."
- Table 34: "Number of shared memory banks | 32" spanning all compute-capability columns.
- The guide's fix for a 32 by 32 array is the padded declaration (32 by 33): "A common fix to avoid bank conflicts is to pad the shared memory by adding one to the column dimension".

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (sections 2.3.4.2 and 2.3.4.2.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 34)

### A11. CUB: `cub::BlockReduce`, `cub::WarpReduce`, functor spellings, Thrust/CUB status

**Verdict: PARTLY.** The classes and `.Sum()` are unchanged. The functors `cub::Max()` and `cub::Sum()` are gone: they were removed in CCCL 3.0. Current spelling is `cuda::maximum<>{}`, `cuda::minimum<>{}`, `cuda::std::plus`.

Current declarations and examples (CCCL docs):

```cpp
#include <cub/cub.cuh>   // or equivalently <cub/block/block_reduce.cuh>
template<typename T, int BlockDimX, BlockReduceAlgorithm Algorithm = BLOCK_REDUCE_WARP_REDUCTIONS,
         int BlockDimY = 1, int BlockDimZ = 1, int WarpAggregateThreshold = -1>
class BlockReduce

using BlockReduce = cub::BlockReduce<int, 128>;
__shared__ typename BlockReduce::TempStorage temp_storage;
int aggregate = BlockReduce(temp_storage).Sum(thread_data);
int aggregate = BlockReduce(temp_storage).Reduce(thread_data, cuda::maximum<>{});

using WarpReduce = cub::WarpReduce<int>;
__shared__ typename WarpReduce::TempStorage temp_storage[4];
int aggregate = WarpReduce(temp_storage[warp_id]).Sum(thread_data);
int aggregate = WarpReduce(temp_storage[warp_id]).Reduce(thread_data, cuda::maximum<>{});
```

Quotes:

- CCCL 3.0 migration guide, "Removed functions and classes": "cub::Max: Use cuda::maximum instead", "cub::Min: Use cuda::minimum instead", "cub::Sum: Use cuda::std::plus instead", "cub::max: Use cuda::std::max instead".
- "Computes a block-wide reduction for thread₀ using addition (+) as the reduction operator." The result is valid in thread 0 only (for `WarpReduce`, in lane 0 of each warp: "Return the warp-wide sums to each lane0").
- NaN handling: "For floating-point types, cuda::minimum and cuda::maximum ignore NaN inputs. If all inputs are NaN, the result is NaN."
- A fourth algorithm now exists: `cub::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC`, "This variant uses atomic operations to reduce the warp-wide reduction results, making it non-deterministic".
- Status: "This repository unifies three essential CUDA C++ libraries into a single, convenient repository: Thrust, CUB, libcudacxx". The old `nvidia/thrust` and `nvidia/cub` repositories are "former repo". CCCL ships in the toolkit: "The easiest way to get started using CCCL is via the CUDA Toolkit which includes the CCCL headers." Compatibility table: CCCL "2.x | 11.1 - 11.8, 12.x (only latest patch releases)", "3.x | 12.x, 13.x (only latest patch releases)".

Not checked: which CCCL version each CUDA 12.x toolkit bundles, so I cannot say from documentation at which toolkit version `cub::Max()` stops compiling. A reader on a 12.x toolkit may still have CCCL 2.x, where `cub::Max()` exists (deprecated in 2.8).

- https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockReduce.html
- https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpReduce.html
- https://nvidia.github.io/cccl/unstable/cccl/3.0_migration_guide.html
- https://github.com/NVIDIA/cccl/blob/main/README.md

### A12. Tensor cores from CUDA C++; half-precision types

**Verdict: PARTLY.** The WMMA intrinsics are still documented and supported. But NVIDIA's stated recommendation is libraries (cuBLAS, cuDNN) or CUTLASS, and the guide now documents a second in-language route, CUDA Tile C++ (`ct::mma`), from toolkit 13.3.

Quotes:

- Compute capabilities appendix: "The Tensor Core feature set is available within the CUDA compilation toolchain through inline PTX. It is strongly recommended that applications use this feature set through CUDA-X libraries such as cuDNN, cuBLAS, and cuFFT, for example, or through CUTLASS".
- WMMA (section 5.4.11): "C++ warp matrix operations leverage Tensor Cores to accelerate matrix problems of the form D=A*B+C. These operations are supported on mixed-precision floating point data for devices of compute capability 7.0 or higher. This requires co-operation from all threads in a warp."
- API, namespace `nvcuda::wmma`, header `<mma.h>`: `template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;`, `void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);`, `void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);`, `void fill_fragment(fragment<...> &a, const T& v);`, `void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);`
- Supported combinations include `__half, __half, float` and `__half, __half, __half` at 16x16x16, 32x8x16, 8x32x16; `unsigned char`/`signed char` with `int` accumulator; `__nv_bfloat16` with `float` accumulator; `precision::tf32` at 16x16x8.
- "The mapping of matrix elements into fragment internal storage is unspecified and subject to change in future architectures." Loads need "a 256-bit aligned pointer".
- CUDA Tile: "CUDA Tile C++ is available in the CUDA Toolkit from version 13.3 onward." "The programmer writes them at tile granularity and the compiler maps them to hardware, including tensor cores where available." Matrix multiply-accumulate is `ct::mma(a, b, acc)` in a `__tile_global__` kernel.
- Table 36 lists tensor-core input types per compute capability. Its 9.0 row (the book's example device) marks FP64, TF32, BF16, FP16, FP8 and INT8; FP6 and FP4 first appear in the 10.0 row.

Half-precision types: `__half` and `__half2` come from `cuda_fp16.h` ("__half and __half2 with the cuda_fp16.h header included"), `__nv_bfloat16` and `__nv_bfloat162` from `cuda_bf16.h`. The guide's WMMA example writes `half` (`__global__ void wmma_ker(half *a, half *b, float *c)`), while every table writes `__half`; use `__half`.

cuBLAS and CUTLASS status: see `ch05-facts.md` A9.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.11)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 36 and the paragraph above it)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html (sections 2.4 and 2.4.9.1)

---
## Group B: HIP / ROCm

Sources: HIP documentation for ROCm 10.2.0 ("HIP 7.17.0"), read as the `.rst` sources at tag `therock-10.2` of `ROCm/rocm-systems` (`projects/hip/docs/...`) with the published page URL given; HIP headers at the same tag; the HIPIFY "CUDA Runtime API supported by HIP" table, which is AMD's own CUDA-name to HIP-name mapping; `ROCm/rocm-libraries` and `ROCm/libhipcxx` at `therock-10.2`.

"Mapped in hip.h" under an item lists the `#define`s added to `labs/common/include/pmpp/vendors/hip.h` because of it. Each was checked in two places: the HIPIFY table row and the declaration in the header at the tag.

### B1. Legacy atomics in HIP (counterpart of A1)

**Verdict: PARTLY.** Supported with the same spelling, including `atomicAdd` on `float` and `double`. Differences: HIP has more types on some functions, has no `_block` variants, has no `__half` or vector `atomicAdd`, and by default implements floating-point atomics safely (possibly as a CAS loop).

Quotes (HIP C++ language extensions, "Atomic functions"):

- "HIP supports the following atomic operations, where TYPE is one of int, unsigned int, unsigned long, unsigned long long, float or double, while INTEGER is int, unsigned int, unsigned long, unsigned long long". Listed: `TYPE atomicAdd(TYPE* address, TYPE val)`, `TYPE atomicSub(...)`, `TYPE atomicMin(...)`, `long long atomicMin(long long* address, long long val)`, `TYPE atomicMax(...)`, `long long atomicMax(...)`, `TYPE atomicExch(...)`, `TYPE atomicCAS(TYPE* address, TYPE compare, TYPE val)`, `INTEGER atomicAnd/atomicOr/atomicXor(...)`, `unsigned int atomicInc(unsigned int* address)`, `unsigned int atomicDec(unsigned int* address)`.
- "The listed functions are also available with the _system (e.g. atomicAdd_system) suffix, operating on system scope ... The functions without suffix operate on shared or global memory on the executing device".
- Floating point: "Some HIP devices support fast atomic operations on floating-point values. For example, atomicAdd on single- or double-precision floating-point values may generate a hardware instruction that is faster than emulating the atomic operation using an atomic compare-and-swap (CAS) loop." "By default the compiler does not generate unsafe instructions. This is the same behaviour as with the -mno-unsafe-fp-atomics compilation flag." AMD-only overrides: `FLOAT_TYPE unsafeAtomicAdd(FLOAT_TYPE* address, FLOAT_TYPE val)` and `FLOAT_TYPE safeAtomicAdd(...)`.
- Hardware features table (columns RDNA1, CDNA1, RDNA2, CDNA2, RDNA3, CDNA3): "atomic functions on 32-bit integer values in global and shared memory" and the 64-bit integer row are yes in all six columns; "Atomic addition on 32-bit floating point values in global and shared memory" is no for RDNA1 and CDNA1 and yes for the other four (same for the 64-bit floating point row).

Header (`amd_hip_atomic.h` at the tag) confirms and adds detail:

- `__device__ inline float atomicAdd(float* address, float val)` calls `unsafeAtomicAdd` only under `__AMDGCN_UNSAFE_FP_ATOMICS__`, otherwise `__scoped_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)`.
- So HIP also has `float`/`double` overloads of `atomicSub`, `atomicMin`, `atomicMax`, `atomicExch` and `atomicCAS` that CUDA does not. A kernel that uses `atomicMax(float*, float)` compiles for HIP and fails for CUDA.
- The string `_block` does not occur in the header: `atomicAdd_block` and friends have no HIP counterpart at this tag.
- I found no `__half`, `__half2`, `float2` or `float4` overload of `atomicAdd` in `amd_hip_atomic.h` or `amd_hip_unsafe_atomics.h`.
- Docs and header disagree on `atomicInc`/`atomicDec`: the docs print one parameter, the header declares the CUDA form `atomicInc(unsigned int* address, unsigned int val)`.

Mapped in hip.h: nothing (same names).

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "Atomic functions")
- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_hip_atomic.h

### B2. `<cuda/atomic>` / libcu++ on HIP (counterpart of A2)

**Verdict: PARTLY.** There is a HIP equivalent, and it keeps the CUDA spelling: AMD's libhipcxx, a port of libcu++, provides `#include <cuda/atomic>` and `cuda::atomic` (also reachable as `<hip/atomic>` and `hip::atomic`). It is part of the ROCm Core SDK from 10.1. It has no cluster scope, and the barrier, latch, semaphore and pipeline headers are unsupported.

Quotes (libhipcxx 3.0.2 docs, pages dated 2026-10-23):

- "libhipcxx is the C++ Standard Library for HIP. ... libhipcxx is derived from libcudacxx and aims to support the same APIs on AMD GPUs. It is a header-only library ... There is no CUDA backend for libhipcxx."
- "Starting with ROCm Core SDK 10.1, libhipcxx is distributed as part of the ROCm Core SDK."
- "To use utilities that are extensions to Standard Library features, drop the std from both the include path and namespace: #include <cuda/atomic> / cuda::atomic<int, cuda::thread_scope_device> x;"
- "The cuda:: and hip:: prefixes, and their include path equivalents, are interchangeable aliases that resolve to the same headers."
- Available: "cuda::atomic / hip::atomic", "cuda::atomic_ref / hip::atomic_ref", "cuda::maximum / hip::maximum", "cuda::minimum / hip::minimum". Its `cuda::atomic` page lists the same additions as NVIDIA's: `cuda::atomic_thread_fence`, `cuda::atomic::fetch_min`, `cuda::atomic::fetch_max`.
- Unsupported: "<cuda/latch>, <cuda/barrier>, <cuda/semaphore>, <cuda/pipeline> - scoped synchronization primitives", "cuda::memcpy_async", "<cuda/ptx>", and "cuda::device::warp_shuffle_idx, warp_shuffle_up, warp_shuffle_down, warp_shuffle_xor, cuda::device::warp_match_all, cuda::device::lane_mask - warp functions, which are only compiled for NVIDIA PTX targets".
- "Some APIs from libcudacxx may be missing or may not achieve the same performance on AMD GPUs".

Header at the tag (`include/cuda/std/__atomic/scopes.h` in `ROCm/libhipcxx`): the enum has `thread_scope_system`, `thread_scope_device`, `thread_scope_block`, `thread_scope_thread` and no `thread_scope_cluster`. `cuda::memory_order_relaxed` and the other five orders are declared in namespace `cuda` exactly as in A2. `fetch_min`/`fetch_max` are present in `include/cuda/__atomic/atomic.h`.

Not verified: whether a default ROCm 10.2 install puts the libhipcxx headers on `amdclang++`'s include path without a CMake target. The docs say to link `libhipcxx::libhipcxx` ("any target you link against libhipcxx::libhipcxx can include libhipcxx headers without specifying the include path manually"). A lab that includes `<cuda/atomic>` needs that target added to the HIP build. On ROCm older than 10.1 there is no such header in the SDK.

Mapped in hip.h: nothing (same names; an include path, not a rename).

- https://rocm.docs.amd.com/projects/libhipcxx/en/latest/
- https://rocm.docs.amd.com/projects/libhipcxx/en/latest/install/install.html
- https://rocm.docs.amd.com/projects/libhipcxx/en/latest/conceptual/libhipcxx-extensions.html
- https://rocm.docs.amd.com/projects/libhipcxx/en/latest/conceptual/libhipcxx-hip-abstractions.html
- https://rocm.docs.amd.com/projects/libhipcxx/en/latest/reference/libhipcxx-limitations.html
- https://rocm.docs.amd.com/projects/libhipcxx/en/latest/libcudacxx/extended_api/synchronization_primitives/atomic.html
- https://github.com/ROCm/libhipcxx/blob/therock-10.2/include/cuda/std/__atomic/scopes.h

### B3. Warp intrinsics in HIP; `HIP_ENABLE_WARP_SYNC_BUILTINS`; 64-bit masks (counterpart of A3)

**Verdict: WRONG** on the macro and on "64-bit masks on gfx9". The functions themselves are confirmed with the CUDA names.

- The gate is reversed. In ROCm 10.2 the `_sync` built-ins are on by default and the macro that matters is `HIP_DISABLE_WARP_SYNC_BUILTINS`. The string `HIP_ENABLE_WARP_SYNC_BUILTINS` does not occur in any HIP header I downloaded at the tag (`amd_warp_functions.h`, `amd_warp_sync_functions.h`, `amd_device_functions.h`, `amd_hip_cooperative_groups.h`, `hip_runtime_api.h`).
- Masks are 64-bit on every AMD target, not only on gfx9, and the type is enforced at compile time.

Quotes:

- Docs: "Note that the __sync variants are available in ROCm 7.0 (and enabled by default, unlike in previous versions). They can be disabled by setting the preprocessor macro HIP_DISABLE_WARP_SYNC_BUILTINS."
- Header comment: "Warp sync builtins (with explicit mask argument) introduced in ROCm 6.2 as a preview to allow end-users to adapt to the new interface involving 64-bit masks. These are enabled by default, and can be disabled by setting the macro "HIP_DISABLE_WARP_SYNC_BUILTINS". This arrangement also applies to the __activemask() builtin".
- Docs: "All mask values returned or accepted by these built-ins are 64-bit unsigned integer values, even when compiled for a device with 32 threads per warp. On such devices the higher bits are unused. CUDA code ported to HIP requires changes to ensure that the correct type is used."
- Header, on every `_sync` function: `static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8, "The mask must be a 64-bit integer. Implicitly promoting a smaller integer is almost always an error.");`
- Header: "When compiling for wave32 mode, ignore the upper half of the 64-bit mask."

Signatures as documented:

```cpp
int __all_sync(unsigned long long mask, int predicate)
int __any_sync(unsigned long long mask, int predicate)
unsigned long long __ballot_sync(unsigned long long mask, int predicate)
unsigned long long __activemask()
T __shfl_sync      (unsigned long long mask, T var, int srcLane, int width=warpSize);
T __shfl_up_sync   (unsigned long long mask, T var, unsigned int delta, int width=warpSize);
T __shfl_down_sync (unsigned long long mask, T var, unsigned int delta, int width=warpSize);
T __shfl_xor_sync  (unsigned long long mask, T var, int laneMask, int width=warpSize);
```

Consequences for a `.cu` file compiled for both vendors:

- `__shfl_xor_sync(0xffffffff, v, 16)` is a compile error on HIP (the literal is a 32-bit `unsigned`). Write the mask through a typed helper, for example a `lane_mask_t` that is `unsigned` for CUDA and `unsigned long long` for HIP. llama.cpp does exactly this (`ggml_cuda_thread_mask_t`, see `ch05-facts.md` B9). An all-ones 64-bit mask is correct in both wave32 and wave64 mode because the header masks off the upper half in wave32.
- `__ballot_sync` and `__activemask` return `unsigned long long`; storing the result in `unsigned` drops lanes 32 to 63 on a 64-wide wavefront.
- Types: "T can be a 32-bit integer type, 64-bit integer type or a single precision or double precision floating point type." I found no `__half` overload of `__shfl` in `amd_warp_functions.h` (CUDA has one, A3).
- Out of range, as documented: `__shfl_up`: "If the resulting source lane is out of range, the thread returns its own var." Same sentence for `__shfl_down`. `__shfl_xor` with a smaller width: "If it tries to read values from later subgroups, the function returns the var of the calling thread." The docs do not state what `__shfl` does for an out-of-range `srcLane` (CUDA documents `srcLane % width`).
- `__syncwarp()` and a masked `template <typename MaskT> void __syncwarp(MaskT mask)` are declared in `amd_warp_sync_functions.h`. The masked form has no default argument value of `0xFFFFFFFF`; the no-argument form exists instead.
- The non-`_sync` forms (`__shfl`, `__ballot`, `__any`, `__all`) exist in HIP and are documented: "AMD GPUs guarantee, that all warp lanes are executed in lockstep, whereas NVIDIA GPUs that support Independent Thread Scheduling might require additional synchronization, or the use of the __sync variants."
- Reductions: `T __reduce_add_sync (unsigned long long mask, T var);` and min/max/and/or/xor exist. On AMD `T` is "int or unsigned int; if the user defines the macro HIP_ENABLE_EXTRA_WARP_SYNC_TYPES, then: unsigned long long, long long, half/float/double".
- Mask semantics are stricter than CUDA's wording: "The union of all mask values supplied at a static instance must be equal to the activemask at the program point." (header comment), checked by `__hip_assert` in debug builds.

llama.cpp still writes `#define HIP_ENABLE_WARP_SYNC_BUILTINS 1` before including the HIP header (`ch05-facts.md` B9); that is for the ROCm 6.2 to 6.x preview period and does nothing on 10.2. I did not add it to our `hip.h` because no current AMD page documents it.

Mapped in hip.h: nothing (same names; the mask type cannot be fixed by a `#define`).

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "Warp cross-lane functions")
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_warp_functions.h
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html (section "warpSize", `lane_mask_t` example)

### B4. Cooperative groups in HIP (counterpart of A4)

**Verdict: PARTLY.** Supported, same namespace and class names, but the header names differ, tiles cannot be larger than a wavefront, `memcpy_async` is synchronous, and the launch function has no default arguments.

Quotes (HIP "Cooperative groups" page):

- "The API is accessible in the cooperative_groups namespace after the hip_cooperative_groups.h header is included."
- Groups documented with the CUDA spelling: `thread_block g = this_thread_block();`, `grid_group g = this_grid();`, `coalesced_group active = coalesced_threads();`, `thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g)`, `cluster_group g = this_cluster();`, and `multi_grid_group g = this_multi_grid();` ("Multi grid deprecated since ROCm 5.0"; CUDA 13 removed it).
- Tiles: "Size must be a power of 2 and not larger than warp (wavefront) size." (CUDA allows up to 1024.) "shfl() functions support integer or float type."
- Grid sync: "Use the cooperative launch API to enable synchronization across the grid group." Check with `hipDeviceGetAttribute(&supports_coop_launch, hipDeviceAttributeCooperativeLaunch, device)`.
- Reduce: `auto reduce(const TyGroup& group, T&& val, Operation&& op)`, "Defined in cooperative_groups/hip_reduce.h." "group is either a coalesced_group or a thread_block_tile". Functors `cooperative_groups::plus`, `less`, `greater`, `bit_and`, `bit_or`, `bit_xor`.
- `memcpy_async` (`#include <hip/cooperative_groups/memcpy_async.h>`): "The copy is completed before the function returns; no separate wait call is needed."
- "HIP doesn't support the following CUDA functions/operators in cooperative_groups namespace: synchronize, wait and wait_prior".
- "AMD GPUs do not support independent thread scheduling. Some CUDA application can rely on this feature and the ported HIP version on AMD GPUs can deadlock".
- Clusters exist in HIP 10.2: "A cluster can be 1D, 2D, or 3D and can contain up to 15 workgroups. Each block in the cluster runs on a separate Workgroup Processor (WGP)." "map_shared_rank() and query_shared_rank() require a device that reports multi-block cluster support. Check hipDeviceProp_t::clusterLaunch at runtime". HIPIFY lists `hipLaunchAttributeClusterDimension` and `hipOccupancyMaxPotentialClusterSize` as added in HIP 7.13.0. I found no `__cluster_dims__` attribute in `host_defines.h` or in the docs, so only the launch-attribute route (`hipLaunchKernelEx` with a `hipLaunchConfig_t`) is confirmed. Which AMD devices report `clusterLaunch` as non-zero: not checked.

Launch function at the tag:

```cpp
hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX,
                                      void** kernelParams, unsigned int sharedMemBytes,
                                      hipStream_t stream);
template <class T>
inline hipError_t hipLaunchCooperativeKernel(T f, dim3 gridDim, dim3 blockDim, void** kernelParams,
                                             unsigned int sharedMemBytes, hipStream_t stream)
```

Differences from CUDA's C++ overload: no defaults for the last two parameters (CUDA: `sharedMem = 0, stream = 0`), and the shared-memory size is `unsigned int`, not `size_t`. A call that passes all six arguments compiles for both. Error code `hipErrorCooperativeLaunchTooLarge` exists.

Header names are the practical obstacle for a single `.cu` source: CUDA's `<cooperative_groups.h>` and `<cooperative_groups/reduce.h>` are `<hip/hip_cooperative_groups.h>` and `<hip/cooperative_groups/hip_reduce.h>` in HIP. A `#define` cannot rename an include; a lab needs an `#if` around the include (or the build needs a forwarding header).

Mapped in hip.h: `cudaLaunchCooperativeKernel`, `cudaDeviceGetAttribute`, `cudaDeviceAttr`, `cudaDevAttrCooperativeLaunch`, `cudaDevAttrMultiProcessorCount`, `cudaDevAttrWarpSize`, `cudaLaunchKernelEx`, `cudaLaunchKernelExC`, `cudaLaunchConfig_t`, `cudaLaunchAttribute`, `cudaLaunchAttributeClusterDimension`. (`cudaLaunchKernelEx` is not in the HIPIFY table; the template `hipLaunchKernelEx` is in the header at the tag.)

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 7660 to 7683, 10918 to 10927, 11224 to 11232)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_cooperative_groups.h
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_hip_cooperative_groups_reduce.h
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html

### B5. Constant memory and `hipMemcpyToSymbol` (counterpart of A5)

**Verdict: CONFIRMED.** `__constant__` keeps its spelling; the copy function is `hipMemcpyToSymbol` with the same parameter list as CUDA's. AMD documents no size limit in bytes.

```cpp
hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes,
                             size_t offset __dparm(0),
                             hipMemcpyKind kind __dparm(hipMemcpyHostToDevice));
template <typename T>
hipError_t hipMemcpyToSymbol(const T& symbol, const void* src, size_t sizeBytes,
                             size_t offset __dparm(0), hipMemcpyKind kind __dparm(hipMemcpyHostToDevice))
```

Quotes:

- "Variables marked with __constant__ reside in device memory. Variables in that address space are routed through the constant cache, but that address space has a limited logical size." No number is given.
- "This memory space is read-only from within kernels and can only be set by the host before kernel execution."
- "To get the best performance benefit, these variables need a special access pattern to benefit from the constant cache - the access has to be uniform within a warp, otherwise the accesses are serialized."
- Example: `__constant__ int const_array[8];` ... `hipMemcpyToSymbol(const_array, host_data, sizeof(int) * 8);`
- "__constant__ memory space qualifiers can not be applied to member variables."

The uniform-access condition is an AMD-specific performance statement with no counterpart sentence in NVIDIA's guide: a lookup table indexed by a per-thread value does not get the constant-cache benefit on AMD.

Mapped in hip.h: `cudaMemcpyToSymbol`, `cudaMemcpyFromSymbol`, `cudaGetSymbolAddress`.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "__constant__")
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (scalar L1 data cache)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 5579 to 5581, 10639 to 10644)

### B6. Vector types and 16-byte loads in HIP (counterpart of A6)

**Verdict: PARTLY.** The vector types exist with the CUDA names and `float4` is 16-byte aligned in the header. As with CUDA, no AMD page documents casting `float*` to `float4*` as the way to get a wide load.

- "Fundamental vector types ... are defined in hip_vector_types.h, which is included by hip_runtime.h." "All vector types can be created with 1, 2, 3 or 4 elements". "All vector types support a constructor function of the form make_<type_name>()." Members `x`, `y`, `z`, `w`.
- Header: `template <typename T> struct alignas(4 * sizeof(T)) HIP_vector_base<T, 4>` and `using float4 = HIP_vector_type<float, 4>;`, so `float4` has alignment 16.
- AMD does document that wide accesses to shared memory are cheaper per byte: LDS access of "4-byte values: 8 cycles for 64 threads (50% peak bandwidth)" versus "16-byte values: 20 cycles for 64 threads (80% peak bandwidth)".

Mapped in hip.h: nothing (same names).

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "Vector types")
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_hip_vector_types.h
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (section "Local data share (LDS)")

### B7. `#pragma unroll` and `__forceinline__` in HIP (counterpart of A7)

**Verdict: CONFIRMED.** Same spelling.

- "The unroll pragma for unrolling loops with a compile-time constant is supported", with `#pragma unroll 16`, `#pragma unroll 1` ("tell compiler to never unroll the loop") and `#pragma unroll` ("hint to compiler to completely unroll next loop.").
- "HIP adds the __noinline__ and __forceinline__ function qualifiers. __noinline__ is a hint to the compiler to not inline the function, whereas __forceinline__ forces the compiler to inline the function. These qualifiers can be applied to both __host__ and __device__ functions."

AMD's wording calls the numeric form a "hint"; NVIDIA's says the loop "will be completely unrolled". I found no `__inline_hint__` in the HIP docs.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/kernel_language_cpp_support.html (section "Pragma Unroll")
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "Inline qualifiers")

### B8. Streams, async copies, pinned memory, events in HIP (counterpart of A8)

**Verdict: CONFIRMED**, with two renames that are not a plain prefix swap: `cudaMallocHost` is `hipHostMalloc` and `cudaFreeHost` is `hipHostFree`.

HIPIFY table rows (CUDA name, HIP name): `cudaStreamCreate` `hipStreamCreate`; `cudaStreamCreateWithFlags` `hipStreamCreateWithFlags`; `cudaStreamDestroy` `hipStreamDestroy`; `cudaStreamSynchronize` `hipStreamSynchronize`; `cudaStreamQuery` `hipStreamQuery`; `cudaStreamWaitEvent` `hipStreamWaitEvent`; `cudaMemcpyAsync` `hipMemcpyAsync`; `cudaMallocHost` `hipHostMalloc`; `cudaHostAlloc` `hipHostAlloc`; `cudaFreeHost` `hipHostFree`; `cudaHostRegister` `hipHostRegister`; `cudaHostUnregister` `hipHostUnregister`; `cudaEventCreate` `hipEventCreate`; `cudaEventCreateWithFlags` `hipEventCreateWithFlags`; `cudaEventRecord` `hipEventRecord`; `cudaEventSynchronize` `hipEventSynchronize`; `cudaEventQuery` `hipEventQuery`; `cudaEventElapsedTime` `hipEventElapsedTime`; `cudaEventDestroy` `hipEventDestroy`; `cudaEvent_t` `hipEvent_t`; `cudaHostAllocDefault` `hipHostMallocDefault` (and `...Mapped`, `...Portable`, `...WriteCombined` likewise); `cudaEventDefault`/`BlockingSync`/`DisableTiming` to `hipEvent...`; `cudaStreamDefault`/`NonBlocking` to `hipStream...`; `cudaMemcpyDefault` `hipMemcpyDefault`; `cudaErrorNotReady` `hipErrorNotReady`.

Declarations at the tag:

```cpp
hipError_t hipStreamCreate(hipStream_t* stream);
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                          hipStream_t stream __dparm(0));
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);      // C function: no default
template <class T>
static inline hipError_t hipHostMalloc(T** ptr, size_t size, unsigned int flags = hipHostMallocDefault)
hipError_t hipHostFree(void* ptr);
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = NULL);
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);
```

Things to know:

- A two-argument `cudaMallocHost(&p, n)` mapped to `hipHostMalloc` resolves to the C++ template, which supplies the flags default. It stops compiling if `__HIP_DISABLE_CPP_FUNCTIONS__` is defined.
- `hipMallocHost` and `hipFreeHost` also exist in the header, the first marked `HIP_DEPRECATED("use hipHostMalloc instead")`. Do not map to them.
- The header defines both `hipHostMallocDefault` and `hipHostAllocDefault` (both 0x0), and the same pairs for Portable (0x1), Mapped (0x2), WriteCombined (0x4). HIP adds flags CUDA does not have: `hipHostMallocUncached`, `hipHostMallocNumaUser`, `hipHostMallocCoherent`, `hipHostMallocNonCoherent`.
- "This API performs an implicit hipDeviceSynchronize() call." (header comment on `hipHostFree`).
- `hipEventElapsedTime`: "Time is computed in ms, with a resolution of approximately 1 us." (CUDA: "around 0.5 microseconds").
- Async: "Some AMD HIP-enabled devices can perform asynchronous memory copy operations to or from the GPU concurrently with kernel execution. Applications can query this capability by checking the asyncEngineCount device property." "if host memory is involved in the copy, it should be page-locked to ensure optimal performance."
- "Page-locked (pinned) memory cannot be swapped to disk by the operating system, allowing the GPU to access it directly via DMA without CPU involvement." Example: `hipHostMalloc(&h_pinned, size);`.

Mapped in hip.h: `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamQuery`, `cudaStreamWaitEvent`, `cudaStreamDefault`, `cudaStreamNonBlocking`, `cudaMemcpyAsync`, `cudaMemcpyDefault`, `cudaMallocHost`, `cudaHostAlloc`, `cudaFreeHost`, `cudaHostAllocDefault`, `cudaHostAllocPortable`, `cudaHostAllocMapped`, `cudaHostAllocWriteCombined`, `cudaEvent_t`, `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventQuery`, `cudaEventElapsedTime`, `cudaEventDestroy`, `cudaEventDefault`, `cudaEventBlockingSync`, `cudaEventDisableTiming`, `cudaErrorNotReady`.

- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 895 to 1016, 3188 to 3339, 3746 to 3913, 4193 to 4246, 5225, 5658, 11295)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/driver_types.h (`hipMemcpyDefault = 4`)
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html

### B9. Occupancy API in HIP (counterpart of A9)

**Verdict: CONFIRMED.**

```cpp
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f,
                                                        int blockSize, size_t dynSharedMemPerBlk);
template <class T>
inline hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, T f, int blockSize,
                                                               size_t dynSharedMemPerBlk)
```

"Returns occupancy for a device function." Same parameter order and types as CUDA. HIPIFY also lists `hipOccupancyMaxPotentialBlockSize` (1.6.0) and `hipOccupancyAvailableDynamicSMemPerBlock` (added in HIP 7.2.0). Remember what a "multiprocessor" is on AMD (`ch05-facts.md` B2): a compute unit, or a workgroup processor of two CUs, depending on the mode.

Mapped in hip.h: `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyMaxPotentialBlockSize`.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 7838 to 7848, 10692 to 10700)
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html (section 9 "Occupancy")

### B10. Shared-memory banks on AMD (counterpart of A10)

**Verdict: CONFIRMED** that AMD documents them. The number is not the same on all AMD GPUs: 32 or 64 banks, each 4 bytes wide.

- "The LDS contains 32 (CDNA, CDNA 2, and CDNA 3) or 64 (CDNA 4 and RDNA 2, RDNA 3, and RDNA 4) banks, each 4-bytes wide, providing 128 (CDNA, CDNA 2, and CDNA 3) or 256 (CDNA 4 and RDNA 2, RDNA 3, and RDNA 4) bytes per cycle total bandwidth."
- "A single warp can achieve up to 64 bytes per cycle throughput (16 lanes per cycle)."
- "The LDS includes hardware to detect and resolve bank conflicts when multiple threads access different addresses in the same bank. Conflicts are resolved by serializing accesses across multiple cycles. ... Broadcasting from the same address to multiple threads is handled efficiently without conflicts."
- "A bank conflict occurs when multiple threads in a warp simultaneously access different addresses that reside in the same LDS bank."
- AMD's name for shared memory is the local data share (LDS): "the on-chip LDS (HIP's "shared memory") inside each compute unit".

Inconsistency inside AMD's docs: the programming-model page says "Shared memory is divided into multiple memory banks (usually 32 banks on modern GPUs)", which contradicts the 64-bank figure above for CDNA 4 and RDNA 2 and later. The hardware page is the specific one.

Consequence: the "+1 column" padding trick from A10 assumes 32 banks and a 32-wide warp. With 32 banks of 4 bytes and a 64-wide wavefront, lanes i and i+32 of a stride-1 `float` access fall in the same bank (my arithmetic from the figures above, not a sentence in AMD's docs), and AMD's "16 lanes per cycle" sentence says a wavefront's LDS access takes several cycles in any case. Do not present a bank-conflict count computed for NVIDIA as valid for AMD.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (section "Local data share (LDS)")
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html (section "Bank conflict theory")
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html

### B11. hipCUB, rocPRIM, rocThrust (counterpart of A11)

**Verdict: PARTLY.** `hipcub::BlockReduce` and `hipcub::WarpReduce` exist with CUB's member names. The functor spelling is in transition at this release and I could not determine which of two copies a ROCm 10.2 install ships.

- "hipCUB is a thin, header-only wrapper library for rocPRIM." Namespace `hipcub` (HIPIFY maps `CUB_NAMESPACE_BEGIN` to `BEGIN_HIPCUB_NAMESPACE`), header `<hipcub/hipcub.hpp>`.
- Header at the tag (`backend/rocprim/block/block_reduce.hpp`): `class BlockReduce` with `T Sum(T input)`, `T Sum(T input, int valid_items)`, `T Reduce(T input, ReduceOp reduce_op)`; algorithms `BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY`, `BLOCK_REDUCE_RAKING`, `BLOCK_REDUCE_WARP_REDUCTIONS`. `class WarpReduce` has `int LOGICAL_WARP_THREADS = HIPCUB_DEVICE_WARP_THREADS` and the same `Sum`/`Reduce`.
- Two copies exist at `therock-10.2`:
  - `projects/hipcub` (the published docs are titled "hipCUB 4.7.0 Documentation"): `struct Sum` and `struct Max` are plain, with the comment "TODO: this is deprecated in cub, we should also mark this as deprecated when we have libhipcxx". Its README: "This project location is being retired. hipCUB is now developed as part of the unified hipCCL project ... This location will no longer be actively maintained after ROCm 10.2."
  - `projects/hipccl/hipccl3/hipcub` (changelog "hipCUB-5.0.0 for ROCm 10.2.0", "Feature parity with CCCL/CUB 3.0.0"): `struct HIPCUB_DEPRECATED_BECAUSE("Use hip::std::plus<T> instead.") Sum` and `struct HIPCUB_DEPRECATED_BECAUSE("Use hip::maximum<T> instead.") Max`.
- So `hipcub::Sum()` and `hipcub::Max()` compile in both copies (with a deprecation warning in the second), and `hip::maximum<>{}` / `cuda::maximum<>{}` needs libhipcxx (B2). The migration page in the source tree lists "hipcub::Sum: Use hip::std::plus instead" under removed APIs, which describes a later state than either header.
- "hipCCL is AMD's unified package for rocPRIM, hipCUB, and rocThrust - one version, one install layout, one package, and one find_package(hipccl) entry point for all three, mirroring the role NVIDIA's CCCL plays for CUB/Thrust/libcudacxx."
- Repositories: `ROCm/hipCUB`, `ROCm/rocPRIM`, `ROCm/rocThrust` are each described "[DEPRECATED] Moved to ROCm/rocm-libraries repo".

For the course: a reduction written against CUB cannot be shared by text between CUDA and HIP (namespace and header differ, and the non-deprecated functor differs by toolkit version on both sides). Treat CUB/hipCUB as a per-vendor comparison point, and write portable lab reductions by hand.

Mapped in hip.h: nothing (not runtime names).

- https://rocm.docs.amd.com/projects/hipCUB/en/latest/index.html
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipcub/hipcub/include/hipcub/backend/rocprim/thread/thread_operators.hpp
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipccl/hipccl3/hipcub/hipcub/include/hipcub/backend/rocprim/thread/thread_operators.hpp
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipcub/hipcub/include/hipcub/backend/rocprim/block/block_reduce.hpp
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipccl/README.md
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipcub/docs/about/hipCUB-cccl-updates.rst
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUB_API_supported_by_HIP.html

### B12. Matrix cores, rocBLAS / hipBLAS, half precision on AMD (counterpart of A12)

**Verdict: PARTLY.** HIP itself has no `nvcuda::wmma` and `<mma.h>` does not exist there. The equivalents are a separate header library, rocWMMA (namespace `rocwmma`, WMMA-shaped API), and the BLAS libraries. Half precision is `__half` from `<hip/hip_fp16.h>`.

Hardware (HIP "Hardware implementation" page):

- "CDNA architectures (MI100 and newer) include specialized matrix acceleration units for high-throughput matrix operations." "In AMD architectures, these units are formally known as MFMA (matrix fused multiply-add) units". "MFMA units support various data types including INT8, FP16, BF16, and FP32".
- Example instruction: `v_mfma_f32_16x16x4f16 v[0:15], v[16:31], v[32:47], v[0:15]`, "a tile size of 16 x 16, with an inner dimension of 4, operating on half-precision (FP16) inputs and accumulating into 32-bit floating-point outputs."
- "Programmers can access MFMA functionality at multiple levels: through optimized libraries, compiler intrinsics, and/or inline assembly".
- Hardware features table, row "Matrix Cores": no for RDNA1, RDNA2, RDNA3; yes for CDNA1, CDNA2, CDNA3.

rocWMMA (docs "rocWMMA 2.2.2"):

- "rocWMMA is a C++ header library for accelerating mixed-precision matrix multiply-accumulate operations. It leverages specialized GPU matrix cores on the latest AMD discrete GPUs."
- API names mirror CUDA's: `rocwmma::fragment`, `rocwmma::matrix_a`, `rocwmma::matrix_b`, `rocwmma::accumulator`, `rocwmma::row_major`, `rocwmma::col_major`, `void rocwmma::fill_fragment(FragT &frag, DataT value)`, `void rocwmma::load_matrix_sync(FragT &frag, const DataT *data, uint32_t ldm)`, `rocwmma::store_matrix_sync`, `mma_sync()`.
- README: supports "CDNA class GPU featuring matrix core support: gfx908, gfx90a, gfx942, gfx950" and "RDNA class GPU featuring AI acceleration support: gfx1100 ... gfx1200, gfx1201". That is wider than the HIP hardware table's "Matrix Cores" row, which stops at RDNA3/CDNA3 and marks RDNA3 as no; I did not resolve the difference.
- Not checked: the exact rocWMMA include line and whether llama.cpp-style hand-written MFMA code is possible through documented intrinsics.

Libraries:

- "rocBLAS is the ROCm Basic Linear Algebra Subprograms (BLAS) library. rocBLAS is implemented in the HIP programming language and optimized for AMD GPUs."
- "hipBLAS is a Basic Linear Algebra Subprograms (BLAS) marshalling library with multiple supported backends. ... Currently, hipBLAS supports rocBLAS and cuBLAS backends." HIPIFY maps `cublasSgemm` to `hipblasSgemm`, `cublasHgemm` to `hipblasHgemm`, `cublasGemmEx` to `hipblasGemmEx`, `cublasCreate_v2` to `hipblasCreate`.
- "hipBLASLt is a library that provides general matrix-matrix operations. ... The general matrix-multiply (GEMM) operation is performed by the hipblasLtMatmul API."
- All now live in `ROCm/rocm-libraries` (`projects/rocblas`, `projects/hipblas`, `projects/hipblaslt`, `projects/rocwmma`).

Half precision: the HIP low-precision page uses `#include <hip/hip_fp16.h>`, `__half`, `__half2` ("holds 2 values of float16 numbers"), `__float2half`, `__half2float`. CUDA's header name `cuda_fp16.h` does not exist in HIP, so this is another include that needs an `#if`.

Mapped in hip.h: nothing (library and include names, not runtime names). cuBLAS names are out of scope for this header; llama.cpp's `vendors/hip.h` shows the pattern if a lab ever calls cuBLAS.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (section "Matrix fused multiply-add (MFMA)")
- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html
- https://rocm.docs.amd.com/projects/rocWMMA/en/latest/index.html
- https://rocm.docs.amd.com/projects/rocWMMA/en/latest/api-reference/api-reference-guide.html
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocwmma/README.md
- https://rocm.docs.amd.com/projects/hipBLAS/en/latest/index.html
- https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html
- https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUBLAS_API_supported_by_HIP.html

---
## Group C: Metal

Sources: Metal Shading Language Specification Version 4.1 (PDF dated 2026-06-04, downloaded again today), cited as "MSL spec"; Metal Feature Set Tables PDF dated October 23, 2026, cited as "feature tables"; Apple developer documentation pages (read through Apple's documentation JSON). The feature tables map chips to GPU families: M1-series is Apple7, M2-series Apple8, M3- and M4-series Apple9, M5-series Apple10, M6-series Apple11; "Metal 3 & 4" in the tables is a programming-model family that all of those belong to.

### C1. Atomics in MSL

**Verdict: PARTLY.** The types and `atomic_fetch_add_explicit` are confirmed, both `device` and `threadgroup` memory are allowed, and float atomics exist. The sub-claim "the only supported memory order is `memory_order_relaxed`" is out of date: MSL 4.1 adds acquire, release and acq_rel (and `seq_cst` came in Metal 4), although for the fetch-and-modify functions the spec still says relaxed only.

Types (MSL spec section 2.6):

- "Metal supports atomic<T>, where T can be int, int32_t, uint, uint32_t, bool, or ulong for all OSes that support Metal 2.4 and later, or T can be float for all OSes that support Metal 3 and later."
- Aliases: `atomic_int` and `atomic_uint` ("OSes that support Metal 1 and later"), `atomic_bool` and `atomic_ulong` ("Metal 2.4 and later"), `atomic_float` ("Metal 3 and later").
- "Metal atomic functions (as described in section 6.16) can only use Metal atomic data types." So you cannot apply an atomic function to a plain `int` buffer; the kernel parameter must be declared `device atomic_int *`.

Functions (section 6.16, header `<metal_atomic>`), each in four address-space forms as printed for store:

```cpp
void atomic_store_explicit(threadgroup A* object, C desired, memory_order order)          // All OS: Since Metal 2.
void atomic_store_explicit(volatile threadgroup A* object, C desired, memory_order order) // All OS: Since Metal 1.
void atomic_store_explicit(device A* object, C desired, memory_order order)               // All OS: Since Metal 2.
void atomic_store_explicit(volatile device A* object, C desired, memory_order order)      // All OS: Since Metal 1.
C atomic_load_explicit(const device A* object, memory_order order)
C atomic_exchange_explicit(device A* object, C desired, memory_order order)
bool atomic_compare_exchange_weak_explicit(device A* object, thread C *expected, C desired,
                                           memory_order success, memory_order failure)
C atomic_fetch_key_explicit(device A* object, M operand, memory_order order)
```

- "The key in the function name is a placeholder for an operation name listed in the first column of Table 6.27": `add`, `and`, `max`, `min`, `or`, `sub`, `xor`. So `atomic_fetch_add_explicit`, `atomic_fetch_max_explicit`, `atomic_fetch_min_explicit` and so on. All return the previous value.
- "These functions are applicable to any atomic object of type atomic_int, and atomic_uint. Atomic add and sub support atomic_float only in device memory. Metal 4.1 and later add atomic_float support for atomic add and sub in threadgroup memory."
- Store, load, exchange and compare-exchange "support atomic types A of atomic_int, atomic_uint, atomic_bool, and atomic_float"; "Metal 3 supports the atomic_float for device memory only."
- 64-bit: `void atomic_key_explicit(device A* object, M operand, memory_order order)` with key `max` or `min` on `atomic_ulong`, device memory only, returns void. "All OS: Metal 2.4 and later support the following atomic modify functions for Apple silicon."
- Address spaces: only `threadgroup` and `device` appear in the signatures. There is no form for `thread` or `constant` memory.
- Spec example: `device atomic_int *output [[buffer(1)]]` ... `atomic_fetch_add_explicit(output, val, memory_order_relaxed);`

Memory order (section 6.16.1):

```cpp
enum memory_order {
     memory_order_relaxed,
     memory_order_acquire,
     memory_order_release,
     memory_order_acq_rel,
     memory_order_seq_cst
};
```

- "All OS: Metal 4.1 and later support the additional memory order values, memory_order_acquire, memory_order_release, and memory_order_acq_rel." `memory_order_seq_cst`: "All OS: Metal 4 and later."
- "In Metal 4.1 and later, you can specify memory_order values for atomic operations, atomic_thread_fence, threadgroup_barrier, and simdgroup_barrier."
- But section 6.16.4.5 still says of the fetch-and-modify functions: "The only supported value for order is memory_order_relaxed." And the same section 6.16.1 keeps the older sentence "For atomic operations other than atomic_thread_fence, memory_order_relaxed is the only enumeration value." The spec is not self-consistent here. Safe course statement: use `memory_order_relaxed` for counters and reductions (it works on every Metal version); stronger orders exist from Metal 4.1 and need a device check.
- Fences and scopes: `void atomic_thread_fence(mem_flags flags, memory_order order, thread_scope scope = thread_scope_device)` with `enum thread_scope { thread_scope_thread, thread_scope_simdgroup, thread_scope_threadgroup, thread_scope_device }`, "All OS: Metal 3.2 and later ... for Apple silicon."

Feature tables (hardware family that first supports the feature): "Floating-point atomics | Metal 3 & 4 | Apple7"; "64-bit atomics | Metal 3 & 4 | Apple9" with the footnote "GPU devices in the Apple8 family support 64-bit atomic minimum and maximum using ulong, on both buffers and textures, only on macOS."; "Acquire and release memory ordering atomics | Metal 3 & 4 | Apple7".

Differences from CUDA to teach: no `double`, no 64-bit add; atomics need an atomic-typed pointer; float atomic max/min is not listed (Table 6.27's max/min apply to `atomic_int` and `atomic_uint`; only add and sub are stated for `atomic_float`).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 2.6 and 6.16, pages 39 and 298 to 307)
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf (feature availability table, pages 5 and 6)

### C2. SIMD-group functions and the vote type

**Verdict: CONFIRMED.** All the names exist in `<metal_simdgroup>`. Permute functions need Apple6 or later, reductions and prefix scans Apple7 or later, so every Apple silicon Mac has both.

Signatures and availability as printed (MSL spec section 6.10.2, Tables 6.14 and 6.15):

| Function | Availability line | Notes from the spec |
|---|---|---|
| `T simd_shuffle(T data, ushort simd_lane_id)` | macOS: Metal 2 and later | "The simd_lane_id needs to be a valid SIMD lane ID but doesn't have to be the same for all threads" |
| `T simd_shuffle_down(T data, ushort delta)` | macOS: Metal 2 and later | "doesn't modify the upper delta lanes of data because it doesn't wrap values around the SIMD-group" |
| `T simd_shuffle_up(T data, ushort delta)` | macOS: Metal 2 and later | "doesn't modify the lower delta lanes of data" |
| `Ti simd_shuffle_xor(Ti value, ushort mask)` | macOS: Metal 2 and later | "The value of mask needs to be the same for all threads in a SIMD-group." |
| `T simd_shuffle_rotate_down(T data, ushort delta)`, `simd_shuffle_rotate_up` | macOS: Metal 2.1 and later | "This function wraps values around the SIMD-group." |
| `T simd_shuffle_and_fill_down(T data, T filling_data, ushort delta)` and `..._up`, plus forms with `ushort modulo` | All OS: Metal 2.4 and later | modulo "must be 2, 4, 8, 16, or 32" |
| `T simd_broadcast(T data, ushort broadcast_lane_id)`, `T simd_broadcast_first(T data)` | macOS: Metal 2 / 2.1 and later | |
| `simd_vote simd_ballot(bool expr)` | macOS: Metal 2.1 and later | "sets the bits that correspond to inactive threads to 0" |
| `bool simd_any(bool expr)`, `bool simd_all(bool expr)` | macOS: Metal 2.1 and later | over active threads |
| `simd_vote simd_active_threads_mask()` | macOS: Metal 2.1 and later | "equivalent to simd_ballot(true)" |
| `bool simd_is_first()` | macOS: Metal 2.1 and later | |
| `T simd_sum(T data)`, `simd_product`, `simd_min`, `simd_max` | macOS: Metal 2.1 and later | result is broadcast "to all active threads in the SIMD-group" |
| `Ti simd_and(Ti data)`, `simd_or`, `simd_xor` | macOS: Metal 2.1 and later | |
| `T simd_prefix_exclusive_sum(T data)`, `simd_prefix_inclusive_sum`, `simd_prefix_exclusive_product`, `simd_prefix_inclusive_product` | macOS: Metal 2.1 and later | exclusive sum: "The first thread in the group, returns T(0)." |

- T: "a scalar or vector of any integer or floating-point type, except: bool, bfloat, long, ulong, void, size_t, ptrdiff_t". So `float` and `half` reductions are allowed (CUDA's `__reduce_add_sync` is integer only). "For bitwise operations, Ti needs to be an integer scalar or vector." Note `simd_shuffle_xor` is declared with `Ti`.
- "SIMD-group functions allow threads in a SIMD-group ... to share data without using threadgroup memory or requiring any synchronization operations, such as a barrier."
- "A thread needs to only read data from another active thread in the SIMD-group."
- `simd_broadcast`: "The specification doesn't define the behavior when broadcast_lane_id isn't a valid SIMD lane ID or isn't the same for all threads in a SIMD-group."
- There is no mask parameter anywhere: the functions act on the active threads, where CUDA's `_sync` intrinsics take an explicit mask.

Vote type: `simd_vote`, a class wrapping `vote_t`, with `explicit constexpr simd_vote(vote_t v = 0);`, `explicit constexpr operator vote_t() const;`, `bool all() const;`, `bool any() const;` and a private `uint64_t v`. "On hardware with fewer than 64 threads in a SIMD-group, the value of the top bits in simd_vote is undefined." And: "simd_all(expr) returns true if all active threads evaluate expr to true. simd_ballot(expr).all() returns true if all threads were active and evaluated the expr to true."

GPU families (feature tables): "SIMD-scoped permute operations | Metal 3 & 4 | Apple6", "SIMD-scoped reduction operations | Metal 3 & 4 | Apple7", "SIMD shift and fill | Metal 3 & 4 | Apple8" (so `simd_shuffle_and_fill_*` is not on M1-series, which is Apple7).

The SIMD-group width is not a documented constant; read `[[threads_per_simdgroup]]` (C5). The spec's worked example draws 16 lanes, as an illustration.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 4.4.1 and 6.10.2, pages 217 to 226)
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf (GPU list on page 2, feature availability on page 4)

### C3. Quad-group functions

**Verdict: CONFIRMED.** "A quad-group is a SIMD-group with the thread execution width of 4." The functions mirror the `simd_` set with a `quad_` prefix (the spec names `quad_broadcast`, `quad_shuffle`, `quad_shuffle_up`, `quad_shuffle_down`, `quad_shuffle_xor`; Tables 6.16 to 6.18 list the function, permute and reduction sets).

- "macOS: Metal 2.1 and later support quad-group functions."
- "Kernels and fragment functions can call the quad-group functions". Their origin is fragment shading: "Helper threads only execute to compute gradients for quad-groups in a fragment shader".
- Feature tables: "Quad-scoped permute operations | Apple4", "Quad-scoped ballot operation | Apple6", "Quad-scoped reduction operations | Apple7".

Not needed for LLM kernels; mention only so the reader recognises the names.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 6.10.3, pages 226 to 234)

### C4. `simdgroup_barrier`

**Verdict: CONFIRMED.**

- `void simdgroup_barrier(mem_flags flags)`: "All threads in a SIMD-group executing the kernel, fragment, mesh, or object need to execute this function before any thread can continue execution beyond the simdgroup_barrier." "macOS: Metal 2 and later".
- New form: `void simdgroup_barrier(mem_flags flags, memory_order order = memory_order_seq_cst, thread_scope scope = thread_scope_simdgroup)`, "All OS: Metal 4.1 and later." (The PDF prints the second parameter as `memory_order_order`, a typo.)
- "A barrier function (threadgroup_barrier or simdgroup_barrier) acts as an execution and memory barrier."
- "If threadgroup_barrier (or simdgroup_barrier) is inside a conditional statement and if any thread enters the conditional statement and executes the barrier function, then all threads in the threadgroup (or SIMD-group) need to enter the conditional and execute the barrier function."
- Feature tables: "SIMD barrier | Metal 3 & 4 | Apple2".

It is the counterpart of `__syncwarp()`. The `mem_flags` values are in `ch05-facts.md` C2.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 6.10.1, page 215)

### C5. SIMD-group kernel attributes

**Verdict: CONFIRMED.** All three names are exact; each takes `ushort` or `uint`.

| Attribute | Spec description | Availability |
|---|---|---|
| `[[threads_per_simdgroup]]` | "The thread execution width of a SIMD-group (compute unit)." | macOS: Metal 2 and later |
| `[[thread_index_in_simdgroup]]` | "The scalar index of a thread within a SIMD-group." | macOS: Metal 2 and later |
| `[[simdgroup_index_in_threadgroup]]` | "The scalar index of a SIMD-group within a threadgroup." | macOS: Metal 2 and later |
| `[[simdgroups_per_threadgroup]]` | "The SIMD-group execution width of a threadgroup." | macOS: Metal 2 and later |

- "For standard Metal compute functions (other than tile functions), SIMD-groups are linear and one-dimensional. (Threadgroups may be multidimensional.)"
- "SIMD-groups execute concurrently within a given threadgroup and make independent forward progress with respect to each other, in the absence of threadgroup barrier operations. The thread index in a SIMD-group (given by [[thread_index_in_simdgroup]]) is a value between 0 and SIMD-group size -1, inclusive."
- "all SIMD-groups within a threadgroup needs to be the same size, apart from the SIMD-group with the maximum index, which may be smaller".
- Spec example parameter list: `uint simd_size [[threads_per_simdgroup]], uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]`.

CUDA has no built-in lane index (`threadIdx.x % warpSize` is the idiom the NVIDIA guide uses); Metal hands it over as an attribute.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 5.2.3.6, Table 5.8)

### C6. Constant address space and small read-only tables

**Verdict: CONFIRMED.** There are two documented ways, and neither is a host-side "copy to symbol" call.

1. A compile-time table at program scope: "You must declare variables in program scope in the constant address space and initialize them during the declaration statement. The initializer(s) expression must be a core constant expression." Example: `constant float samples[] = { 1.0f, 2.0f, 3.0f, 4.0f };`. "Declaring such a variable without initialization is also a compile-time error."
2. A buffer bound as a kernel argument and declared read-only: "The constant address space name refers to buffer memory objects allocated from the device memory pool that are read-only." "Buffers in the constant address space passed to kernel, vertex, and fragment functions have minimum alignment requirements based on the GPU. See "Minimum constant buffer offset alignment" in the Metal Feature Set Tables". The table gives 4 B for every Apple family column.

So the Metal counterpart of `__constant__` plus `cudaMemcpyToSymbol` is an ordinary `MTLBuffer` that the host fills and the kernel takes as `constant T*` (or `const device T*`), or, for values known when the shader is compiled, a `constant` array or a preprocessor macro (`ch05-facts.md` C7). The spec gives no size limit for constant data and makes no statement about a separate constant cache.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 4.2)
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf (GPU family limits table)

### C7. Packed vector types and `float4` loads

**Verdict: PARTLY.** Packed types are confirmed. As on the other two platforms, no document describes casting a `float*` to a `float4*`; the documented way to read four floats at once is to declare the buffer with a vector element type.

- `float4`: size 16, alignment 16 (Table 2.3). "You must align the vector data types described in section 2.2 to the size of the vector."
- Packed names: `packed_charn`, `packed_shortn`, `packed_intn`, `packed_ucharn`, `packed_ushortn`, `packed_uintn`, `packed_halfn`, `packed_bfloatn` (Metal 3.1 and later), `packed_floatn`, `packed_longn` (Metal 2.3 and later), n = 2, 3 or 4; also `packed_vec<T,n>`.
- `packed_float4`: size 16, alignment 4. `packed_float3`: size 12, alignment 4.
- "Packed vector data types are typically used as a data storage format. ... Metal also supports loads and stores from a packed vector data type to an aligned vector data type and vice-versa."
- Spec example: `device float4 *buffer; device packed_float4 *packed_buffer;` ... `packed_float4 f ( buffer[i] );` ... `buffer[i] = float4( packed_buffer[i] );`
- Reinterpreting values (not pointers) is documented: "Metal adds an as_type<type-id> operator to allow any scalar or vector data type ... to be reinterpreted as another data type of the same size."

For a kernel that wants 16-byte loads from a float array: declare the parameter `const device float4 *` when the data is 16-byte aligned, or `const device packed_float4 *` when only 4-byte alignment is known. I searched the spec for "reinterpret_cast" and for pointer-cast rules and found neither, so do not teach a pointer cast as documented behaviour.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 2.2, 2.2.3 and 2.23)

### C8. Half precision (`half`)

**Verdict: CONFIRMED.** `half` is a core scalar type, not an optional extension: "A 16-bit floating-point. The half data type must conform to the IEEE 754 binary16 storage format." Literal suffix "h or H" (`0.5h`). Vector forms `half2`/`half3`/`half4` (`half4`: size 8, alignment 8), `packed_halfn`, and `simdgroup_half8x8` exist.

- Also `bfloat`: "A 16-bit brain floating-point ... All OS: Metal 3.1 and later." with suffix `bf`. Mixing is restricted: "Because bfloat and half are not implicitly convertible to each other, the operators do not support mixing bfloat and half."
- Accuracy of half operations is specified separately (Table 8.3 "Accuracy of half-precision floating-point operations and functions").
- "Metal does not support the double, long long, unsigned long long, and long double data types."

What the spec does not say: whether `half` arithmetic is faster than `float` on a given Apple GPU. That is a measurement for the reader's machine, not a documented fact.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 2.1, Tables 2.1 to 2.3)

### C9. Integer and 8-bit types

**Verdict: CONFIRMED.** Table 2.1: `char`/`int8_t` ("A signed two's complement 8-bit integer."), `unsigned char`/`uchar`/`uint8_t`, `short`/`int16_t`, `ushort`/`uint16_t`, `int`/`int32_t`, `uint`/`uint32_t`, `long`/`int64_t` and `ulong`/`uint64_t` ("All OS: Metal 2.2 and later"), `bool`, `size_t` ("a 64-bit unsigned integer"). Vector forms such as `char4` (size 4, alignment 4) and `packed_char4` (size 4, alignment 1) exist.

For quantised kernels: 8-bit weights can be read directly as `char`/`uchar` buffers; `long` in MSL is always 64-bit (in CUDA `long` is 4 or 8 bytes depending on the host data model, A6's table footnote); "64-bit integer math" is listed in the feature tables from Apple3.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 2.1)

### C10. `simdgroup_matrix` and Metal Performance Primitives

**Verdict: PARTLY.** These are two different things. `simdgroup_matrix` is a core MSL type from `<metal_simdgroup_matrix>` (Metal 2.3 and later). Metal Performance Primitives (MPP) is a newer shader-side library (Metal 4 and later) built on tensors, and Apple now recommends it over `simdgroup_matrix`.

`simdgroup_matrix` (MSL spec sections 2.4 and 6.8):

- "Metal supports a matrix type simdgroup_matrix<T,Cols,Rows> defined in <metal_simdgroup_matrix>. Operations on SIMD-group matrices are executed cooperatively by threads in the SIMD-group. Therefore, all operations must be executed only under uniform control-flow within the SIMD-group or the behavior is undefined."
- Types: `simdgroup_half8x8`, `simdgroup_bfloat8x8` (Metal 3.1 and later), `simdgroup_float8x8`. Only 8 by 8.
- "The mapping of matrix elements to threads in the SIMD-group is unspecified."
- Functions: `make_filled_simdgroup_matrix(T value)`; `void simdgroup_load(thread simdgroup_matrix<T,Cols,Rows>& d, const device T *src, ulong elements_per_row = Cols, ulong2 matrix_origin = 0, bool transpose_matrix = false)` (also from `threadgroup` memory); `simdgroup_store(...)`; `void simdgroup_multiply_accumulate(d, a, b, c)`, "Returns d = a * b + c"; and `simdgroup_multiply`, "Returns d = a * b".
- Feature tables: "SIMD-scoped matrix multiply operations | Metal 3 & 4 | Apple7".
- Apple's own advice in section 6.8: "Instead of using simdgroup matrix multiplication, consider using Tensors (section 2.22) and Metal Performance Primitives (section 7). Together, they provide a performant and portable library supporting more general matrix and tensor operations across a wider variety of dimensions and extents."

MPP (MSL spec chapter 7):

- "All OS: Metal 4 and later support Metal Performance Primitives." "The header <MetalPerformancePrimitives/MetalPerformancePrimitives.h> defines these functions within the namespace mpp. The tensor_ops namespace ... contains functions that operate on tensors, including matrix multiplication and convolution."
- `matmul2d` "performs a generalized matrix multiplication of two tensors (C = A*B) or matrix multiplication accumulated into a tensor (C = A*B + C)." Scopes: `execution_thread`, `execution_simdgroup`, `execution_simdgroups<N>`.
- Table 7.3 type combinations include `half, half -> float`, `char, half -> half`, `char, char -> int` ("Metal 4"), `bfloat` combinations ("Metal 4 and OS 26.1") and `uchar` combinations ("Metal 4 and OS 26.4").
- Feature tables: "Tensors | Metal 4 | Apple7".

What they are for: `simdgroup_matrix` plays the role of CUDA's WMMA fragments (a small matrix tile spread over one SIMD-group, with a multiply-accumulate), and both MLX and llama.cpp use it for GEMM today (D2, D3); MPP `matmul2d` is the newer, more general route and both projects have started to use it. Apple's documentation does not say what hardware executes either (it never mentions a matrix unit), so do not call them "tensor cores".

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 2.4, 6.8 and 7, pages 38, 212 to 214, 349 onward)
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf (pages 4 and 5)

### C11. The library route for GEMM: MPS, MPSGraph, Accelerate

**Verdict: CONFIRMED.**

| Library | What Apple documents | GPU? |
|---|---|---|
| Metal Performance Shaders, `MPSMatrixMultiplication` (macOS 10.13+) | "A matrix multiplication kernel." "computes the following operation: C = alpha * op(A) * op(B) + beta * C Where A, B, and C are matrices represented by MPSMatrix objects" | yes, encoded into an `MTLCommandBuffer` |
| Metal Performance Shaders Graph, `MPSGraph.matrixMultiplication(primary:secondary:name:)` (macOS 11.0+) | "Computes the matrix multiplication of 2 input tensors with support for broadcasting." Framework: "Build, compile, and execute compute graphs utilizing all the different compute devices on the platform, including GPU, CPU, and Neural Engine." | yes |
| Accelerate BLAS, `cblas_sgemm` | "Multiplies two matrices (single-precision)." "Apple provides the BLAS and LAPACK libraries under the Accelerate framework to be in line with LAPACK 3.9.1. Starting with iOS 26, iPadOS 26, macOS 26 ... the libraries are in line with LAPACK 3.12.0." | no statement that it uses the GPU; Accelerate is the CPU reference |
| Metal 4 machine-learning pass, `MTL4MachineLearningCommandEncoder` (macOS 26.0+) | "Encodes machine learning model inference commands for a single pass." with `MTLTensor`, "A resource representing a multi-dimensional array that you can use with machine learning workloads." | yes |

MPS overview: "a collection of highly optimized compute and graphics shaders ... specially tuned to take advantage of the unique hardware characteristics of each GPU family".

So the cuBLAS counterpart is `MPSMatrixMultiplication` or an MPSGraph matmul; Accelerate is the CPU baseline to test against. PyTorch's MPS backend uses all of MPSGraph, `MPSNDArrayMatrixMultiplication` and its own Metal kernels (D4).

- https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication
- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/matrixmultiplication(primary:secondary:name:)
- https://developer.apple.com/documentation/metalperformanceshadersgraph
- https://developer.apple.com/documentation/accelerate/cblas_sgemm(_:_:_:_:_:_:_:_:_:_:_:_:_:_:)
- https://developer.apple.com/documentation/metal/mtl4machinelearningcommandencoder
- https://developer.apple.com/documentation/metal/mtltensor

### C12. Command buffers and blit encoders as the analogue of streams and async copies

**Verdict: PARTLY.** The API facts are confirmed. The analogy is the course's, not Apple's, and it is loose in two places: ordering is per command queue, and on Apple silicon a shared buffer needs no copy at all.

Facts:

- `MTLCommandBuffer`: "A container that stores a sequence of GPU commands that you encode into it."
- `MTLCommandQueue`: "A command queue maintains an ordered list of command buffers." `enqueue()`: "Reserves the next available place for the command buffer in its command queue."
- `commit()`: "Submits the command buffer to run on the GPU." It returns without waiting; the blocking call is `waitUntilCompleted()`: "Blocks the current thread until the GPU finishes executing the command buffer and all of its completion handlers." The non-blocking one is `addCompletedHandler(_:)`: "Registers a completion handler the GPU device calls immediately after the GPU finishes running the commands in the command buffer."
- `MTLBlitCommandEncoder`: "Encodes commands that copy and modify resources for a single blit pass.", including "Copying data between buffers". `copy(from:sourceOffset:to:destinationOffset:size:)`: "Encodes a command that copies data from one buffer into another."
- Cross-buffer dependencies: `MTLEvent`, "synchronize commands across a single Metal device by instructing it to wait before starting a workload, such as a compute pass, until another workload finishes." `MTLSharedEvent` extends that to "multiple Metal devices and the CPU".
- Timing: `GPUStartTime`/`GPUEndTime` per command buffer (`ch05-facts.md` C9), the counterpart of a pair of CUDA events.

Mapping for a lesson, with the caveats:

| CUDA | Closest Metal | Caveat |
|---|---|---|
| a stream | a command queue (ordered), or a sequence of command buffers on one queue | Apple documents ordering within a queue; I found no Apple statement that two queues, or two command buffers, execute concurrently on the GPU |
| `cudaMemcpyAsync` host to device | nothing: write into `buffer.contents` of a shared-storage `MTLBuffer` | unified memory, `ch05-facts.md` C6 |
| `cudaMemcpyAsync` device to device | blit encoder `copy(from:...)` | runs in command-buffer order |
| pinned host memory | no counterpart documented | no pageable/pinned distinction appears in the Metal buffer docs I read |
| `cudaStreamSynchronize` | `waitUntilCompleted()` | |
| `cudaStreamWaitEvent` | `MTLEvent` wait/signal | |
| event timing | `GPUEndTime - GPUStartTime` | whole command buffer, not one kernel |

- https://developer.apple.com/documentation/metal/mtlcommandbuffer
- https://developer.apple.com/documentation/metal/mtlcommandqueue
- https://developer.apple.com/documentation/metal/mtlcommandbuffer/commit()
- https://developer.apple.com/documentation/metal/mtlcommandbuffer/enqueue()
- https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted()
- https://developer.apple.com/documentation/metal/mtlcommandbuffer/addcompletedhandler(_:)
- https://developer.apple.com/documentation/metal/mtlblitcommandencoder
- https://developer.apple.com/documentation/metal/mtlblitcommandencoder/copy(from:sourceoffset:to:destinationoffset:size:)
- https://developer.apple.com/documentation/metal/mtlevent
- https://developer.apple.com/documentation/metal/mtlsharedevent

### C13. Anything Apple documents about caches, banks or bandwidth on Apple GPUs

**Verdict: PARTLY.** Apple publishes qualitative statements and measuring tools, and no numbers. There is nothing about threadgroup-memory banks at all.

- MSL spec and feature tables: I searched both for "cache", "bank" and "bandwidth". The only hits are a compiler module-cache flag and one sentence about tile shading ("utilizing high-bandwidth local memory"). No cache sizes, no line sizes, no bank count, no memory bandwidth.
- Apple tech talk "Metal Compute on MacBook Pro" (transcript on developer.apple.com): "Apple Silicon GPUs contain separate L1 caches for texture reads and buffer reads." "there's a cache for improving the performance of future buffer reads to the same local area of memory. The cache is of a limited size, though, and will fill quickly, so older data that hasn't been read for a while will be discarded to make way for newer reads." It suggests moving some data to textures to use both caches.
- Xcode documentation, "Measuring the GPU's use of memory bandwidth": "The GPU bandwidth counters measure how much, in gigabytes per second, the GPU reads and writes, or accesses memory." "If the GPU's memory bandwidth usage is high when you don't expect it to be, it may impede the CPU's memory accesses." (Instruments and the Metal debugger's Performance timeline expose the counters.)
- Not found: any Apple page with a bank structure for `threadgroup` memory, an L1 or last-level cache size, or a per-chip GPU memory bandwidth figure. A web search restricted to developer.apple.com returned only WWDC and tech-talk videos; I read one transcript (quoted above) and did not read the others.

For the course: the bank-conflict lesson has no Metal half. Say so, and let the reader measure (the padded versus unpadded lab, timed on their Mac) instead of quoting a number.

- https://developer.apple.com/videos/play/tech-talks/10580/
- https://developer.apple.com/documentation/xcode/measuring-the-gpus-use-of-memory-bandwidth
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf

---
## Group D: Real backends (for the "reading real backends" module)

Read from the repositories' default branches on 2026-11-20 through the GitHub API and raw files. "Opened" means I read part of the file; "by name" means the path exists in today's listing and the description is inferred from its name only. These projects move daily (all four directories below had commits on 2026-11-19 or 2026-11-20), so link to directories in lessons and treat file names as a snapshot.

### D1. llama.cpp: CUDA backend

**Verdict: CONFIRMED.** Directory `ggml/src/ggml-cuda/` on `master` (last commit to the directory 2026-11-19). Kernels are split one operation per `.cu`/`.cuh` pair; heavily templated kernels are instantiated from generated files in `template-instances/`.

| Path under `ggml/src/ggml-cuda/` | What it is |
|---|---|
| `ggml-cuda.cu` | Backend entry and operation dispatch (opened). The library route lives here: calls to `cublasSgemm`, `cublasGemmEx` and `cublasGemmBatchedEx`. |
| `mmq.cu`, `mmq.cuh` | Quantised matrix times matrix: `ggml_cuda_mul_mat_q`, switching on the weight type (`mul_mat_q_case<GGML_TYPE_Q1_0, ...>` and so on) (opened). Per-type instances in `template-instances/mmq-instance-q4_0.cu`, `...-q4_k.cu`, `...-q8_0.cu`, ... |
| `mmq-config-ampere.cuh`, `mmq-config-blackwell.cuh`, `mmq-config-cdna.cuh`, `mmq-config-rdna3.cuh`, ... | Per-architecture tuning tables for the quantised matmul, for NVIDIA and AMD families (by name). |
| `mmf.cu`, `mmf.cuh` | Float matrix times matrix: `ggml_cuda_mul_mat_f` (opened). |
| `mmvq.cu`, `mmvq.cuh` | Quantised matrix times vector: `static __global__ void mul_mat_vec_q(...)` (opened). The token-by-token generation path. |
| `mmvf.cu`, `mmvf.cuh` | Float matrix times vector: `static __global__ void mul_mat_vec_f(...)` (opened). |
| `mma.cuh` | Its header comment: "This file contains primitives that expose the tensor core PTX instructions for CUDA code. The primitives can be used in a similar way as the nvcuda::wmma interface but with a well-defined memory layout." Also carries AMD paths (`AMD_WMMA_256B_AVAILABLE`; callers test `amd_mfma_available(cc)`). |
| `fattn.cu`, `fattn.cuh`, `fattn-common.cuh` | Flash-attention entry and kernel choice: an enum with `BEST_FATTN_KERNEL_VEC`, `BEST_FATTN_KERNEL_TILE`, `BEST_FATTN_KERNEL_MMA_F16` (opened). |
| `fattn-vec.cuh` | The VEC attention kernel, one instance per K/V type pair (`template-instances/fattn-vec-instance-f16-q4_0.cu`, ...) (not read in detail). |
| `fattn-tile.cu`, `fattn-tile.cuh` | The TILE attention kernel, one instance per head size (`fattn-tile-instance-dkq64-dv64.cu`, ...) (not read in detail). |
| `fattn-mma-f16.cuh` | The MMA_F16 attention kernel; includes `mma.cuh` (2348 lines today, not read in detail). `fattn.cu` selects it when `turing_mma_available(cc)` on NVIDIA or `amd_mfma_available(cc)`. |
| `softmax.cu`, `softmax.cuh` | Softmax (opened only for length; 564 lines). |
| `vecdotq.cuh`, `dequantize.cuh`, `quantize.cu` | Quantised dot products and (de)quantisation helpers (by name). |
| `common.cuh`, `vendors/cuda.h`, `vendors/hip.h`, `vendors/musa.h` | Shared definitions and the vendor name-mapping headers (`ch05-facts.md` B9). |

- https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmf.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvf.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mma.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-mma-f16.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-vec.cuh
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-tile.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/softmax.cu
- https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda/template-instances

### D2. llama.cpp: Metal kernels; HIP, Vulkan and SYCL status

**Verdict: CONFIRMED.**

Metal: `ggml/src/ggml-metal/kernels/` holds 41 files today (last commit to `ggml-metal` 2026-11-19).

| Path under `ggml/src/ggml-metal/kernels/` | What it is |
|---|---|
| `mul_mm.metal` | Matrix times matrix: `kernel void kernel_mul_mm(...)`, built on `simdgroup_float8x8` / `simdgroup_multiply_accumulate` (opened). |
| `mul_mv.metal` | Matrix times vector for float and quantised types; no SIMD-group matrices (opened; 3430 lines). |
| `mul_mv_mma.metal` | Matrix times vector using SIMD-group matrices: `kernel_mul_mv_mma_blk` with `simdgroup_float8x8` (opened). |
| `fa_f32.metal`, `fa_f16.metal`, `fa_q4_0.metal`, `fa_q8_0.metal`, ... with `fa_common.metal`, `fa_aux.metal` | Flash-attention kernels, one file per K/V type (opened `fa_f32.metal`). |
| `fa_vec_f16.metal`, `fa_vec_q4_0.metal`, ... with `fa_vec_common.metal` | The "vec" attention variant, one file per type (by name). |
| `fa_tensor.metal` | Attention on Metal 4 tensors: `kernel_flash_attn_ext_tensor` calling `mpp::tensor_ops::matmul2d` (opened). The Metal Performance Primitives route of C10. |
| `softmax.metal`, `norm.metal`, `rope.metal`, `reduce.metal`, `unary.metal`, `binbcast.metal` | The remaining per-operation kernels (by name; `softmax.metal` is 223 lines). |
| `common.h`, `dequantize.h`, `quantize.h` | Shared headers concatenated into each kernel source at build time (`ch05-facts.md` C10). |

Host side: `ggml/src/ggml-metal/ggml-metal-ops.cpp`, `ggml-metal-device.m`, `ggml-metal-context.m`, `ggml-metal.cpp`.

Other GPU backends, from the directory listing and `docs/build.md`:

- HIP: `ggml/src/ggml-hip/` contains only `CMakeLists.txt`; it compiles `../ggml-cuda/*.cu` as language HIP (`ch05-facts.md` B9). `docs/build.md` has a "## HIP" section: "This provides GPU acceleration on HIP-supported AMD GPUs. Make sure to have ROCm installed.", enabled with `-DGGML_HIP=ON`.
- Vulkan: `ggml/src/ggml-vulkan/` (`ggml-vulkan.cpp` and `vulkan-shaders/` with 193 files today, compute shaders named `*.comp`), a separate implementation; "## Vulkan" section in `docs/build.md`, `-DGGML_VULKAN=ON`.
- SYCL: `ggml/src/ggml-sycl/`, a separate C++ implementation; `docs/build.md`: "llama.cpp based on SYCL is used to support Intel GPU (Data Center Max series, Flex series, Arc series, Built-in GPU and iGPU)."
- Also present as directories today: `ggml-opencl`, `ggml-webgpu`, `ggml-musa`, `ggml-cann`, `ggml-openvino`, `ggml-hexagon`, `ggml-virtgpu`, `ggml-zdnn`, `ggml-zendnn`, `ggml-rpc`, `ggml-blas`, `ggml-et`.

- https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal/kernels
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mm.metal
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mv.metal
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mv_mma.metal
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/fa_f32.metal
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/fa_tensor.metal
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/softmax.metal
- https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-hip
- https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-vulkan
- https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-sycl
- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md

### D3. MLX: Metal backend layout; matmul, softmax, quantised matmul

**Verdict: CONFIRMED.** `mlx/backend/metal/` on `main` (last commit to the directory 2026-11-20). Layout: one host-side `.cpp` per operation family, which builds a kernel name and encodes the dispatch, and `kernels/` with the MSL (`.metal` files that instantiate templates defined in the `.h` next to them). MLX now also has `mlx/backend/cuda/` beside `metal/`, `cpu/` and `common/`.

| Topic | Host side | Kernels |
|---|---|---|
| Matmul (GEMM) | `mlx/backend/metal/matmul.cpp`: `steel_matmul_regular_axpby`, `steel_gemm_splitk_axpby` and `_nax` variants, building names such as `steel_gemm_fused_...` and `steel_gemm_splitk_...` (opened) | `kernels/steel/gemm/kernels/steel_gemm_fused.metal`, `steel_gemm_splitk.metal`, `steel_gemm_masked.metal`, `steel_gemm_gather.metal`, `steel_gemm_segmented.metal`; block logic in `kernels/steel/gemm/gemm.h`, `loader.h`, and `mma.h`, which wraps `metal::simdgroup_matrix` and `simdgroup_multiply_accumulate` (opened) |
| Matmul, newer path | same file, the `_nax` functions | `kernels/steel/gemm/nax.h` includes `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` ("NAX Steel with new tiles"); `steel_gemm_fused_nax.metal`, `steel_gemm_splitk_nax.metal`, ... (opened `nax.h`) |
| Matrix times vector | (by name) | `kernels/gemv.metal`, `gemv.h`, `gemv_masked.metal` |
| Softmax | `mlx/backend/metal/softmax.cpp`: `Softmax::eval_gpu` picks `"looped_"` or `"block_"` + `"softmax_"` + optional `"precise_"` + type name (opened) | `kernels/softmax.h`, `kernels/softmax.metal` |
| Quantised matmul | `mlx/backend/metal/quantized.cpp`: kernel families `qmv`, `qmv_fast`, `qmv_quad`, `qmv_wide`, `qvm`, `qvm_splitk`, `qmm_t`, `qmm_n`, plus `qmv_mma`, `qmv_nax`, `qmm_t_nax`, `qmm_n_nax` (opened) | `kernels/quantized.h`, `quantized.metal`, `quantized_nax.h`, `quantized_nax.metal`, `quantized_utils.h`; also `fp_quantized.metal` |
| Attention | `mlx/backend/metal/scaled_dot_product_attention.cpp` (by name) | `kernels/scaled_dot_product_attention.metal`, `sdpa_vector.h`, `sdpa_blocked.h`, `kernels/steel/attn/` |
| Reductions, atomics helpers | `reduce.cpp` (by name) | `kernels/reduce.metal`, `kernels/reduction/`, `kernels/atomic.h` |
| Device, library loading, JIT | `device.cpp`, `kernels.h`, `jit_kernels.cpp`, `nojit_kernels.cpp`, `jit/` (`ch05-facts.md` C10) | |

- https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/matmul.cpp
- https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels/steel/gemm
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/mma.h
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/nax.h
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/softmax.cpp
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/softmax.h
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/quantized.cpp
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h
- https://github.com/ml-explore/mlx/tree/main/mlx/backend/cuda

### D4. PyTorch: MPS backend and CUDA native kernels

**Verdict: CONFIRMED.** On `main` (last commit under `aten/src/ATen/native/mps` 2026-11-19).

MPS backend, three places:

| Path | What it is |
|---|---|
| `aten/src/ATen/mps/` | Runtime plumbing: `MPSDevice.mm`, `MPSStream.mm`, `MPSAllocator.mm`, `MPSEvent.mm`, `MPSProfiler.mm`, `MPSFallback.mm`. |
| `aten/src/ATen/native/mps/operations/*.mm` | Operator implementations in Objective-C++. This is where both the MPSGraph calls and the Metal-kernel dispatches are. `Matmul.mm` does both: `lib.getPipelineStateForFunc("matmul_" + ...)`, `"naive_bmm_"`, `"addmm_"` for its own shaders, and `[graph matrixMultiplicationWithPrimaryTensor:selfTensor secondaryTensor:otherTensor name:nil]` for MPSGraph (opened). `Linear.mm` uses `MPSNDArrayMatrixMultiplication` (opened). `SoftMax.mm` and `Blas.mm` dispatch to the bundled Metal library (opened). `Attention.mm`, `Quantized.mm`, `ReduceOps.mm` (by name). |
| `aten/src/ATen/native/mps/kernels/*.metal` (with `.h`) | PyTorch's own Metal shaders: `Matmul.metal` (kernels `matmul_simd` and `matmul_mpp`, opened), `SoftMax.metal`, `Attention.metal`, `Quantized.metal`, `ReduceOps.metal`, `Gemv.h`, `MppAttention.h`, `BinaryKernel.metal`, `UnaryKernel.metal`, ... |
| `aten/src/ATen/native/mps/MetalShaderLibrary.h`, `OperationUtils.h`, `OperationUtils.mm` | `class MetalShaderLibrary` (`getBundledLibrary()`, `getPipelineStateForFunc`) and the shared MPSGraph helpers (first opened, the other two by name). |

So the older description "PyTorch MPS is MPSGraph" is now only half of it: matmul, softmax and attention have hand-written Metal kernels, with MPSGraph still used on some paths in the same files.

CUDA native kernels: `aten/src/ATen/native/cuda/` (317 entries today). For the topics of this course: `Blas.cpp` (matmul entry points; calls `at::cuda::blas::gemm_and_bias` and friends, opened), `SoftMax.cu` and `PersistentSoftmax.cuh`, `Loops.cuh` / `CUDALoops.cuh` (elementwise kernel templates), `Reduce.cuh`, `block_reduce.cuh`, `ReduceSumProdKernel.cu`, `KernelUtils.cuh`, `MemoryAccess.cuh`, `linalg/` (all by name). The cuBLAS wrappers themselves are one directory over, in `aten/src/ATen/cuda/CUDABlas.cpp` and `CUDABlas.h`; `aten/src/ATen/cuda/Atomic.cuh` holds PyTorch's atomic helpers (by name).

- https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/mps
- https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/mps/operations
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Matmul.mm
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Linear.mm
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/SoftMax.mm
- https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/mps/kernels
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/Matmul.metal
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/MetalShaderLibrary.h
- https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Blas.cpp
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SoftMax.cu
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp

---

## Summary of corrections

Every WRONG or PARTLY item, one line each (plus the one COULD NOT VERIFY, for completeness):

- A2: the first CUDA Toolkit version that shipped `<cuda/atomic>` is not stated in current docs (could not verify); both `cuda::memory_order_relaxed` and `cuda::std::memory_order_relaxed` are valid; `cuda::thread_scope_cluster` exists in libcu++ although the programming guide's scope table omits it; several guide links to `nvidia.github.io/cccl/libcudacxx/` are dead (use `/cccl/unstable/libcudacxx/`).
- A4: the reference says `cg::reduce` takes only `coalesced_group` or `thread_block_tile`, while the guide's own example passes a `thread_block`; multi-device cooperative launch was removed in CUDA 13.
- A6 (could not verify): no NVIDIA document says that casting `float*` to `float4*` yields one 16-byte load; documented are `float4` size 16 / alignment 16 and `cudaMalloc` alignment of at least 256 bytes, and NVIDIA's own sample uses inline PTX when a single 128-bit store is required.
- A11: `cub::Max()`, `cub::Min()` and `cub::Sum()` were removed in CCCL 3.0; write `cuda::maximum<>{}`, `cuda::minimum<>{}`, `cuda::std::plus`; `BlockReduce(...).Sum(x)` is unchanged.
- A12: WMMA (`nvcuda::wmma`, `<mma.h>`) is still supported, but NVIDIA "strongly" recommends cuBLAS/cuDNN or CUTLASS, and CUDA Tile C++ (`ct::mma`, toolkit 13.3 onward) is a second documented in-language route; the type is `__half` from `cuda_fp16.h` (the guide's example writes `half`).
- B1: HIP legacy atomics keep CUDA's names and cover `float`/`double`, but there are no `_block` variants and no `__half` or vector `atomicAdd`; HIP additionally accepts `float`/`double` for `atomicMin`/`atomicMax`/`atomicSub`/`atomicExch`/`atomicCAS`, which CUDA rejects; float atomics are safe (CAS-style) unless built with `-munsafe-fp-atomics`.
- B2: HIP does have a libcu++ counterpart, libhipcxx (in the ROCm Core SDK from 10.1), and it keeps the spelling `#include <cuda/atomic>` / `cuda::atomic`; it has no `thread_scope_cluster` and no `<cuda/barrier>`, `<cuda/latch>`, `<cuda/semaphore>`, `<cuda/pipeline>`.
- B3 (WRONG): `HIP_ENABLE_WARP_SYNC_BUILTINS` gates nothing in ROCm 10.2; the `_sync` built-ins are on by default since ROCm 7.0 and the switch is `HIP_DISABLE_WARP_SYNC_BUILTINS`.
- B3 (WRONG): lane masks are 64-bit on every AMD target, not just gfx9, enforced by a `static_assert`, so `__shfl_xor_sync(0xffffffff, ...)` does not compile under HIP; `__ballot_sync` and `__activemask` return `unsigned long long`.
- B4: HIP cooperative groups use different headers (`<hip/hip_cooperative_groups.h>`, `<hip/cooperative_groups/hip_reduce.h>`), limit `thread_block_tile` to the wavefront size, make `memcpy_async` synchronous (no `wait`), and `hipLaunchCooperativeKernel` has no default arguments and takes `unsigned int sharedMemBytes`.
- B4: HIP 10.2 does have `cluster_group` / `this_cluster()` and cluster launch attributes, but no `__cluster_dims__` attribute was found.
- B6: HIP's `float4` is 16-byte aligned in the header, but as for CUDA no AMD document describes the pointer-cast idiom.
- B11: `hipcub::BlockReduce`/`WarpReduce` exist, but ROCm 10.2 carries two hipCUB copies (4.7.0 with plain `hipcub::Sum`/`Max`, and 5.0.0 under hipCCL with them deprecated in favour of `hip::std::plus` / `hip::maximum`); which one an install provides was not determined.
- B12: HIP has no `nvcuda::wmma`; AMD's matrix cores (MFMA units, CDNA only per the HIP hardware table) are reached through the separate rocWMMA header library (`rocwmma::fragment`, `mma_sync`) or rocBLAS / hipBLAS / hipBLASLt; half precision is `__half` from `<hip/hip_fp16.h>`, not `cuda_fp16.h`.
- C1: "relaxed is the only memory order" is outdated: MSL 4.1 adds acquire, release and acq_rel (seq_cst from Metal 4), though the spec still says relaxed only for the fetch-and-modify functions; `atomic_float` exists (add and sub; device memory, and threadgroup memory from Metal 4.1); there is no 64-bit atomic add.
- C7: packed vector types are confirmed (`packed_float4`: size 16, alignment 4), but MSL documents no `float*` to `float4*` pointer cast; declare the buffer as `device float4*` or `device packed_float4*`.
- C10: `simdgroup_matrix` is not part of Metal Performance Primitives; it is a core MSL type (Metal 2.3 and later, 8 by 8 only), and Apple now recommends MPP `matmul2d` on tensors (Metal 4 and later) instead.
- C12: command buffers and blit encoders as "streams and async copies" is our analogy, not Apple's; ordering is documented per command queue, concurrency between queues is not documented, and on Apple silicon a shared-storage buffer needs no host-to-device copy.
- C13: Apple documents no cache sizes, no bank structure for threadgroup memory and no bandwidth figures; only qualitative statements (separate L1 caches for texture and buffer reads) and bandwidth counters in Instruments.

Path and status changes worth knowing even where the verdict is CONFIRMED:

- A1: NVIDIA now labels `atomicAdd` and its siblings "Legacy Atomic Functions" and recommends `cuda::atomic` / `cuda::atomic_ref`.
- A8/B8: `cudaMallocHost` maps to `hipHostMalloc` and `cudaFreeHost` to `hipHostFree` (HIP's `hipMallocHost` is deprecated).
- B10: AMD's shared memory (LDS) has 32 banks on CDNA 1 to 3 and 64 banks on CDNA 4 and RDNA 2 to 4, each 4 bytes wide.
- D4: PyTorch's MPS backend now has its own Metal kernels for matmul, softmax and attention alongside MPSGraph.
- D3: MLX has a CUDA backend directory next to the Metal one.

## Where the backends really differ

One line per cell. "Same" means the same source text compiles for both through `pmpp/vendors/hip.h`, according to the documentation; nothing here has been run on CUDA or AMD hardware by us.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Warp / wavefront / SIMD-group width | 32 on every listed compute capability | 64 on gfx9 (Instinct), 32 on gfx10 and later (Radeon); read `warpSize` | Not a documented constant; read `[[threads_per_simdgroup]]` |
| Simple atomic add on integers | `atomicAdd(int*, int)`, global or shared memory | Same call | `atomic_fetch_add_explicit(p, v, memory_order_relaxed)` on a pointer declared `device atomic_int*` |
| Atomic add on `float` | `atomicAdd(float*, float)`; also `double` | Same call; implemented safely (possibly a CAS loop) unless built with `-munsafe-fp-atomics` | `atomic_float`, add and sub only, Metal 3 and later; no `double` type at all |
| Atomic max / min | Integer types only; float needs a CAS loop | Also accepts `float` and `double` (not portable back to CUDA) | `atomic_fetch_max_explicit` / `min` on `atomic_int` and `atomic_uint`; none documented for float |
| Scoped C++ atomics | `<cuda/atomic>`: `cuda::atomic<T, cuda::thread_scope_block>`, `fetch_min`, `fetch_max` | Same spelling through libhipcxx (ROCm 10.1 and later); no cluster scope | `atomic<T>` only; scopes exist for fences (`thread_scope_simdgroup`, ...) |
| Lane mask in `_sync` intrinsics | `unsigned` (32-bit) | Must be a 64-bit integer on every target; a 32-bit literal is a compile error | No mask; functions act on the active threads |
| Shuffle | `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync` | Same names, 64-bit mask; no `__half` overload found | `simd_shuffle`, `simd_shuffle_up`, `simd_shuffle_down`, `simd_shuffle_xor` |
| Ballot / any / all | `__ballot_sync` returns `unsigned` | `__ballot_sync` returns `unsigned long long` | `simd_ballot` returns `simd_vote` (wraps a 64-bit value; top bits undefined) |
| Built-in warp reduction | `__reduce_add_sync` and friends, integers only | Same, integers; float with `HIP_ENABLE_EXTRA_WARP_SYNC_TYPES` | `simd_sum`, `simd_max`, `simd_min`, `simd_prefix_exclusive_sum` on integer and floating types |
| Warp-level barrier | `__syncwarp()` | `__syncwarp()` | `simdgroup_barrier(mem_flags::mem_threadgroup)` |
| Lane index | `threadIdx.x % warpSize` | Same | `[[thread_index_in_simdgroup]]` attribute |
| Cooperative groups | `<cooperative_groups.h>`; tiles up to 1024 threads | `<hip/hip_cooperative_groups.h>`; tiles no larger than a wavefront; `memcpy_async` is synchronous | None; use the SIMD-group functions directly |
| Whole-grid barrier | `grid.sync()` after `cudaLaunchCooperativeKernel` | `grid.sync()` after `hipLaunchCooperativeKernel` (all six arguments required) | None documented; `threadgroup_barrier` is defined over one threadgroup |
| Thread block clusters | `__cluster_dims__` or `cudaLaunchKernelEx`, compute capability 9.0 and later | `cluster_group` and launch attributes exist; no `__cluster_dims__` found; device support not checked | None |
| Small read-only table | `__constant__` array + `cudaMemcpyToSymbol`, 64 KB | `__constant__` + `hipMemcpyToSymbol`; fast only when the access is uniform across the wavefront; size "limited", no number | `constant` array initialised in the shader, or an ordinary buffer taken as `constant T*` |
| 16-byte vector load | `float4` (size 16, align 16); the pointer-cast idiom is not documented | Same type and alignment; idiom not documented | Declare the buffer `device float4*` or `packed_float4*`; no pointer cast documented |
| Loop unrolling / forced inlining | `#pragma unroll`, `__forceinline__` | Same spelling | Neither appears in the MSL spec (no "unroll", no inline attribute) |
| Shared-memory banks | 32 banks of 4 bytes | 32 banks (CDNA 1 to 3) or 64 banks (CDNA 4, RDNA 2 to 4) of 4 bytes | Nothing documented |
| Streams and async copies | Streams, `cudaMemcpyAsync`, pinned memory from `cudaMallocHost` | Same model; `hipHostMalloc`, `hipHostFree` | Command buffers on a command queue; shared buffers need no copy; no pinned-memory concept |
| Timing | Two events + `cudaEventElapsedTime` (ms) | `hipEventElapsedTime` (ms) | `GPUEndTime - GPUStartTime` of a command buffer (seconds) |
| Occupancy query | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `hipOccupancyMaxActiveBlocksPerMultiprocessor`, same parameters | No counterpart in the pages read |
| Block / warp reduction library | CUB: `cub::BlockReduce`, functor `cuda::maximum<>{}` | hipCUB: `hipcub::BlockReduce`, functor spelling in transition | None; write it with `simd_sum` and threadgroup memory |
| Matrix hardware from kernel code | `nvcuda::wmma` fragments (`__half` in, `float` accumulate); CUDA Tile `ct::mma` | rocWMMA library (`rocwmma::fragment`); matrix cores on CDNA only per HIP's table | `simdgroup_float8x8` + `simdgroup_multiply_accumulate`; MPP `matmul2d` on Metal 4 |
| GEMM library | cuBLAS (`cublasSgemm`), CUTLASS | hipBLAS (`hipblasSgemm`) over rocBLAS; hipBLASLt | `MPSMatrixMultiplication`, MPSGraph; Accelerate BLAS on the CPU |
| Half precision | `__half` from `cuda_fp16.h` | `__half` from `<hip/hip_fp16.h>` | `half` is a built-in type (`0.5h`) |
| 8-bit integers | `char`, `unsigned char` | Same | `char`, `uchar` |
| 64-bit floating point | `double` | `double` | Not supported |

---

## Group E (2026-11-21): what the emulators and the front-end stub were extended with

Checked on 2026-11-21 for the final pass on the lab runtime (`labs/common/include/pmpp/`). Sources reported "v13.5" (CUDA Math API Reference Manual, Programming Guide, Runtime API reference) and HIPIFY's `amd-develop` branch (last commit to the table 2026-11-19). Same conventions as above; **Quote** is from the page as fetched and flattened. Nothing was compiled against CUDA or ROCm. Where each fact went: `pmpp/check/cuda_stub.h` (front-end stub), `pmpp/emu/emu_types.h`, `emu_device.h`, `emu_host.h`, `emu_runtime.h`, `cuda_fp16_emu.h` (emulators), `pmpp/vendors/hip.h`; tested by `labs/selftest/{intrinsics,halfprec,hostapi,shared}.cu`.

### E1. Sizes and alignments of the built-in vector types

**Verdict: CONFIRMED** (and it settles the alignments the stub and emulator had marked unverified).

Source: CUDA Programming Guide 13.5, section 5.4.2.3 "Built-in Types", Table 49 "Alignment Requirements", https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (read 2026-11-21).

- Size and alignment in bytes: `char1`/`uchar1` 1 1; `char2`/`uchar2` 2 2; `char3`/`uchar3` 3 1; `char4`/`uchar4` 4 4; `short1`/`ushort1` 2 2; `short2`/`ushort2` 4 4; `short3`/`ushort3` 6 2; `short4`/`ushort4` 8 8; `int2`/`uint2` 8 8; `int3`/`uint3` 12 4; `int4`/`uint4` 16 16; `float2` 8 8; `float3` 12 4; `float4` 16 16; `double2` 16 16; `double3` 24 8; `double4` (deprecated) 32 16; `double4_16a` 32 16; `double4_32a` 32 32.
- "They all have a factory function of the form `make_<type_name>()`", example `make_int4(x + 1, y + 1, z + 1, w + 1)`.

### E2. `__dp4a`: the four documented overloads

**Verdict: CONFIRMED** (matches x2-facts A2).

Source: CUDA Math API 13.5, "Integer Intrinsics", https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html (read 2026-11-21).

- `__device__ int __dp4a(int srcA, int srcB, int c)`: "Four-way signed int8 dot product with int32 accumulate." "Extracts four pairs of packed byte-sized integers from srcA and srcB, then creates four pairwise products and adds them together to a signed 32-bit integer c."
- `__device__ int __dp4a(char4 srcA, char4 srcB, int c)`; `__device__ unsigned int __dp4a(unsigned int srcA, unsigned int srcB, unsigned int c)` ("Four-way unsigned int8 dot product with unsigned int32 accumulate."); `__device__ unsigned int __dp4a(uchar4 srcA, uchar4 srcB, unsigned int c)`.
- COULD NOT VERIFY on this page: sign extension of the bytes (only "signed int8" is said), wrap-around of the 32-bit accumulation, and a compute capability requirement (none is stated on the page; x2-facts A2 has 6.1 from llama.cpp and the PTX ISA, whose `dp4a` semantics add into a 32-bit `d`). The emulators sign-extend for the signed forms and wrap modulo 2^32.
- HIP: no `__dp4a` (x2-facts B1). `hip.h` maps nothing and says so.

### E3. `__byte_perm`, `__umulhi`, `__mulhi`, `__umul64hi`

**Verdict: CONFIRMED.** Same page as E2.

- `__device__ unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s)`: "Return selected bytes from two 32-bit unsigned integers." The page's own description: `uint64_t tmp64 = ((uint64_t)y << 32) | x;` `selector0 = (s >> 0) & 0x7;` `selector1 = (s >> 4) & 0x7;` `selector2 = (s >> 8) & 0x7;` `selector3 = (s >> 12) & 0x7;` `res[07:00] = tmp64[selector0];` ... `res[31:24] = tmp64[selector3];`. So bits 3, 7, 11, 15 and 16 to 31 of `s` are never read.
- `__device__ __tile__ unsigned int __umulhi(unsigned int x, unsigned int y)`: "Returns the most significant 32 bits of the product x * y." `__device__ __tile__ int __mulhi(int x, int y)`: the same for signed. `__device__ __tile__ unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y)`: "the most significant 64 bits". (The stub leaves out `__tile__`.)
- HIPIFY maps `__byte_perm` to `__byte_perm` (x2-facts B1); `__umulhi` exists in HIP (x1-facts B3).

### E4. Type-casting intrinsics

**Verdict: CONFIRMED.** Source: CUDA Math API 13.5, "Type Casting Intrinsics", https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html (read 2026-11-21).

- `__device__ unsigned int __float_as_uint(float x)` "Reinterpret bits in a float as an unsigned integer."; `__device__ float __uint_as_float(unsigned int x)`; `__device__ int __float_as_int(float x)`; `__device__ float __int_as_float(int x)`.

### E5. Single-precision math functions and fast intrinsics: device-only signatures

**Verdict: CONFIRMED** for the names and signatures; the emulators do NOT model NVIDIA's error bounds.

Sources: CUDA Math API 13.5, "Single Precision Mathematical Functions", https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html, and "Single Precision Intrinsics", https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html (both read 2026-11-21).

- Each listed as `__device__` only: `float expf(float x)`, `exp2f`, `logf`, `log2f`, `rsqrtf`, `sqrtf`, `tanhf`, `erff`, `fabsf`, `float fmaf(float x, float y, float z)`, `float fminf(float x, float y)`, `fmaxf`, `powf`, `copysignf`. The page shows no `__host__` qualifier.
- Intrinsics, `__device__` only: `__expf` ("the fast approximation to the base e exponential"), `__logf`, `__sinf`, `__cosf`, `float __powf(float x, float y)` ("a fast approximation to x^y"), `float __fdividef(float x, float y)` ("the fast approximation to the quotient"), `float __saturatef(float x)` ("Clamp the input argument to [+0.0, 1.0]"), and the `_rn` forms `__frsqrt_rn`, `__fmaf_rn`, `__fadd_rn`, `__fmul_rn`.
- For the project: clang's CUDA mode needs these declared `__device__` for a kernel to call them; the stub now declares them (labs no longer need their own `#if PMPP_CUDA_FRONTEND_CHECK` prototypes, which still compile as redeclarations). The `_rn` intrinsics were left out of both because lab A.1 defines `__fadd_rn`/`__fsub_rn` itself for the emulator. NaN behaviour of `__saturatef` was not checked.

### E6. Per-dimension limits on blocks and grids

**Verdict: CONFIRMED.**

Source: CUDA Programming Guide 13.5, section 5.1 "Compute Capabilities", Table 33 "Device and Streaming Multiprocessor (SM) Information per Compute Capability", https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (read 2026-11-21).

- "Maximum dimensionality of a grid" 3; "Maximum x-dimension of a grid" 2^31-1; "Maximum y- or z-dimension of a grid" 65535; "Maximum dimensionality of a thread block" 3; "Maximum x- or y-dimensionality of a thread block" 1024; "Maximum z-dimension of a thread block" 64; "Maximum number of threads per block" 1024; "Warp size" 32. One value per row, spanning every compute capability the table lists (7.5 to 12.x).
- The emulators now refuse a launch over any of these with `cudaErrorInvalidConfiguration` and a message citing the table, and report them as `maxThreadsDim` and `maxGridSize`.

### E7. Stream priorities and the current device

**Verdict: CONFIRMED.**

Sources: CUDA Runtime API 13.5, "Stream Management", https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html, and "Device Management", https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html (read 2026-11-21).

- `__host__ cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority)`; `__host__ cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority)`. "lower numbers represent higher priorities. '0' represents default priority." An out-of-range priority "will automatically be clamped to the lowest or the highest number in the range".
- `__host__ cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority)`: the valid range is `*greatestPriority` to `*leastPriority`; if the device does not support stream priorities both are returned as 0; either pointer may be NULL.
- `__host__ cudaError_t cudaGetDevice(int *device)`: "Returns the device on which the active host thread executes the device code."
- The emulators answer as a device without priorities (range 0 to 0, every stream priority 0) and report device 0.

### E8. Half precision: `__half`, `__half2`, conversions and arithmetic

**Verdict: CONFIRMED** for the subset the emulators and the stub provide; see "not modelled" in `pmpp/emu/cuda_fp16_emu.h`.

Sources: CUDA Math API 13.5 (read 2026-11-21): "Half Precision Conversion and Data Movement", https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html; "Half Arithmetic Functions", https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__ARITHMETIC.html; "Half2 Arithmetic Functions", https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__ARITHMETIC.html; struct pages https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html and https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html.

- Conversions, `__host__ __device__` unless noted: `__half __float2half(const float a)` and `__float2half_rn` (round-to-nearest-even), `float __half2float(const __half a)`, `__half2 __float2half2_rn(const float a)` (both halves), `__half2 __floats2half2_rn(const float a, const float b)`, `__half2 __float22half2_rn(const float2 a)`, `float2 __half22float2(const __half2 a)`, `__halves2half2`, `make_half2`, `__half2half2`, `float __low2float(const __half2 a)` and `__high2float` ("the low 16 bits" / "the high 16 bits"), `__low2half`, `__high2half`, `__half_as_ushort`, `__ushort_as_half`, `__half_as_short`, `__short_as_half`, `__int2half_rn`; `__device__ int __half2int_rn(const __half h)`.
- Arithmetic: `__hadd`, `__hsub`, `__hmul`, `__hdiv` ("round-to-nearest-even"), `__device__ __half __hfma(...)` (one rounding), `__hneg`, `__habs`; `__hadd2`, `__hsub2`, `__hmul2`, `__h2div`, `__device__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c)` ("fused multiply-add ... with one rounding", round-to-nearest-even), `__hneg2`, `__habs2`. Operators on `__half`: binary `+ - * /`, `+=`, unary `-` (each "see" the matching function); on `__half2`: unary `+ -`, binary `+ - * /`, `+= -= *= /=`, `++`, `--`.
- `struct __half`: default constructor (uninitialised), `constexpr __half(const __half_raw&)`, implicit constructors from `double`, `float`, `int`, `long long`, `long`, `short` and the unsigned types, conversion operators to `float`, `int`, `bool` and the other integer types. `struct __half2`: members `__half x` (lower) and `__half y` (upper), `constexpr __half2(const __half&, const __half&)`, copy and move, from `__half2_raw`.
- COULD NOT VERIFY: the alignment of `__half2` (no statement on the pages; the stub and emulator use 4, from memory of `cuda_fp16.h`); the bit pattern of a canonical half NaN; the compute capability needed for half arithmetic in device code. HIP's header is `<hip/hip_fp16.h>` (B12).

### E9. `cudaFuncGetAttributes`, `struct cudaFuncAttributes`, and their HIP names

**Verdict: CONFIRMED.**

Sources: CUDA Runtime API 13.5, "Execution Control", https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html, "C++ API Routines", https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html, and https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaFuncAttributes.html; HIPIFY "CUDA Runtime API supported by HIP", https://github.com/ROCm/HIPIFY/blob/amd-develop/docs/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.md (all read 2026-11-21).

- `__host__ cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func)` ("Find out attributes for a given function."), and `template<class T> __host__ cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, T *entry)`.
- Fields used by the course: `int binaryVersion`, `size_t constSizeBytes`, `size_t localSizeBytes`, `int maxDynamicSharedSizeBytes`, `int maxThreadsPerBlock`, `int numRegs`, `int preferredShmemCarveout`, `int ptxVersion`, `size_t sharedSizeBytes` (the struct has more: cluster fields, `cacheModeCA`, `sharedMemoryMode`, `reserved`).
- HIPIFY rows: `cudaGetDevice` -> `hipGetDevice` (HIP 1.6.0); `cudaFuncGetAttributes` -> `hipFuncGetAttributes` (1.9.0); `cudaFuncAttributes` -> `hipFuncAttributes` (1.9.0; the CUDA column marks it changed in 13.3); `cudaStreamCreateWithPriority` -> `hipStreamCreateWithPriority` and `cudaDeviceGetStreamPriorityRange` -> `hipDeviceGetStreamPriorityRange` (2.0.0; already in `hip.h`). The first three were added to `hip.h`.
- In the emulators every field is synthetic: 0 except `maxThreadsPerBlock` (1024) and `maxDynamicSharedSizeBytes` (the kernel's `cudaFuncSetAttribute` value, else 48 KiB).

## Group F (2026-11-21): what the lab runner's device report reads

### F1. `asyncEngineCount` and `concurrentKernels` in `cudaDeviceProp` and `hipDeviceProp_t`

**Verdict: CONFIRMED** (names and types); the meaning on ROCm differs (ch23-facts B3).

Sources (read 2026-11-21): CUDA Runtime API 13.5, `cudaDeviceProp` struct reference, https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html; HIP 7.2.53211 documentation (ROCm docs-7.2.4), `hipDeviceProp_t` struct reference, https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.4/doxygen/html/structhip_device_prop__t.html; HIP how-to "Asynchronous concurrent execution", https://rocm.docs.amd.com/projects/HIP/en/develop/_sources/how-to/hip_runtime_api/asynchronous.rst.

- CUDA: `int asyncEngineCount` "Number of asynchronous engines."; `int concurrentKernels` "Device can possibly execute multiple kernels concurrently."
- HIP: `int hipDeviceProp_t::asyncEngineCount` "Number of async engines."; `int hipDeviceProp_t::concurrentKernels` "Device can possibly execute multiple kernels concurrently."; `int deviceOverlap` "Deprecated. Use asyncEngineCount instead."
- HIP how-to: "Applications can query this capability by checking the `asyncEngineCount` device property." "Devices with an `asyncEngineCount` greater than zero support concurrent data transfers." "Devices that support the `concurrentKernels` property can perform intra-device copies concurrently with kernel execution."
- What one and two copy engines allow on NVIDIA is ch23-facts A7 (Best Practices Guide 10.1.2); that on ROCm `asyncEngineCount` is the runtime constant 8 for every device is ch23-facts B3 (from the ROCm sources). `python3 labs/pmpp.py doctor` prints both fields on CUDA and HIP (`labs/common/src/backend_cudalike.cpp`) with those readings beside them. The emulator's `cudaDeviceProp` has `concurrentKernels` (0) but no `asyncEngineCount`; its device report gives 0 for it, since the emulator runs streams synchronously.
- Not compiled on CUDA or HIP here (no such device): the field accesses are unverified on both.
