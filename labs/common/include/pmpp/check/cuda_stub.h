// Stub declarations for a CUDA *front-end* check on machines with no CUDA SDK.
//
// Used only as:  clang++ -x cuda --cuda-host-only -nocudainc -nocudalib
//                        -fsyntax-only -I common/include -I common/include/pmpp/emu/shim
//                        -include pmpp/check/cuda_stub.h file.cu
//
// clang's CUDA mode then parses kernel launches (<<< >>>) and enforces the
// host/device call rules for real. What this does NOT establish: that these
// hand-written prototypes match NVIDIA's headers, that nvcc accepts the file,
// or anything about code generation. Nothing here is ever linked or run.
//
// Where the prototypes come from (read on 2026-11-20 unless dated otherwise):
//   [G]  CUDA Programming Guide 13.5, "C/C++ Language Extensions" and
//        "Device-Callable APIs" (cooperative groups); Table 49 (vector type alignments,
//        2026-11-21)
//   [R]  CUDA Runtime API reference 13.5 (the group__CUDART__* pages)
//   [M]  CUDA Math API Reference Manual 13.5 (read 2026-11-21; notes/today/infra-facts.md,
//        group E)
//   [L]  libcu++ sources in NVIDIA/cccl, main branch
//   [?]  NOT found in those documents; written from memory of the toolkit
//        headers and marked UNVERIFIED where it appears.
//
// The file has four sections. The first is always active. The other three are
// switched on by the stand-in headers pmpp/emu/shim/cuda/atomic,
// pmpp/emu/shim/cooperative_groups.h and pmpp/emu/shim/cuda_fp16.h, which re-include
// this file, so that code must #include <cuda/atomic>, <cooperative_groups.h> or
// <cuda_fp16.h> to use them, as with nvcc.

#ifndef PMPP_CUDA_STUB_H
#define PMPP_CUDA_STUB_H
#define PMPP_CUDA_FRONTEND_CHECK 1
#include <stddef.h>

#define __global__   __attribute__((global))
#define __device__   __attribute__((device))
#define __host__     __attribute__((host))
#define __shared__   __attribute__((shared))
#define __constant__ __attribute__((constant))
// [G 5.4.1.3]. clang treats __noinline__ as a keyword in CUDA mode, so it needs
// no definition here. __restrict__ is a keyword of the compiler.
#define __forceinline__ __inline__ __attribute__((always_inline))
// Kernel launch bounds (ch08-facts A4/B4): clang's CUDA mode has the attribute itself.
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __inline_hint__

// ---- vector types [G Table 49] (sizes and alignments; make_<type> as section 5.4.2.3) -----
struct char1 { signed char x; };
struct __attribute__((aligned(2))) char2 { signed char x, y; };
struct char3 { signed char x, y, z; };
struct __attribute__((aligned(4))) char4 { signed char x, y, z, w; };
struct uchar1 { unsigned char x; };
struct __attribute__((aligned(2))) uchar2 { unsigned char x, y; };
struct uchar3 { unsigned char x, y, z; };
struct __attribute__((aligned(4))) uchar4 { unsigned char x, y, z, w; };
struct short1 { short x; };
struct __attribute__((aligned(4))) short2 { short x, y; };
struct short3 { short x, y, z; };
struct __attribute__((aligned(8))) short4 { short x, y, z, w; };
struct ushort1 { unsigned short x; };
struct __attribute__((aligned(4))) ushort2 { unsigned short x, y; };
struct ushort3 { unsigned short x, y, z; };
struct __attribute__((aligned(8))) ushort4 { unsigned short x, y, z, w; };
__host__ __device__ char1 make_char1(signed char x);
__host__ __device__ char2 make_char2(signed char x, signed char y);
__host__ __device__ char3 make_char3(signed char x, signed char y, signed char z);
__host__ __device__ char4 make_char4(signed char x, signed char y, signed char z, signed char w);
__host__ __device__ uchar1 make_uchar1(unsigned char x);
__host__ __device__ uchar2 make_uchar2(unsigned char x, unsigned char y);
__host__ __device__ uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z);
__host__ __device__ uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w);
__host__ __device__ short1 make_short1(short x);
__host__ __device__ short2 make_short2(short x, short y);
__host__ __device__ short3 make_short3(short x, short y, short z);
__host__ __device__ short4 make_short4(short x, short y, short z, short w);
__host__ __device__ ushort1 make_ushort1(unsigned short x);
__host__ __device__ ushort2 make_ushort2(unsigned short x, unsigned short y);
__host__ __device__ ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z);
__host__ __device__ ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w);
struct uint3 { unsigned x, y, z; };
struct __attribute__((aligned(8)))  uint2 { unsigned x, y; };
struct __attribute__((aligned(16))) uint4 { unsigned x, y, z, w; };
struct __attribute__((aligned(8)))  int2 { int x, y; };
struct int3 { int x, y, z; };
struct __attribute__((aligned(16))) int4 { int x, y, z, w; };
struct __attribute__((aligned(8)))  float2 { float x, y; };
struct float3 { float x, y, z; };
struct __attribute__((aligned(16))) float4 { float x, y, z, w; };
struct __attribute__((aligned(16))) double2 { double x, y; };
struct double3 { double x, y, z; };
struct __attribute__((aligned(16))) double4 { double x, y, z, w; };
__host__ __device__ uint2 make_uint2(unsigned x, unsigned y);
__host__ __device__ uint3 make_uint3(unsigned x, unsigned y, unsigned z);
__host__ __device__ uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w);
__host__ __device__ int2 make_int2(int x, int y);
__host__ __device__ int3 make_int3(int x, int y, int z);
__host__ __device__ int4 make_int4(int x, int y, int z, int w);
__host__ __device__ float2 make_float2(float x, float y);
__host__ __device__ float3 make_float3(float x, float y, float z);
__host__ __device__ float4 make_float4(float x, float y, float z, float w);
__host__ __device__ double2 make_double2(double x, double y);
__host__ __device__ double3 make_double3(double x, double y, double z);
__host__ __device__ double4 make_double4(double x, double y, double z, double w);

struct dim3 {
  unsigned x, y, z;
  __host__ __device__ constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
  __host__ __device__ constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
  __host__ __device__ constexpr operator uint3() const { return uint3{x, y, z}; }
};

// ---- built-in variables [G 5.4.2] ---------------------------------------------------------
extern __device__ const uint3 threadIdx;
extern __device__ const uint3 blockIdx;
extern __device__ const dim3 blockDim;
extern __device__ const dim3 gridDim;
extern __device__ const int warpSize;   // "a run-time value ... commonly 32"

// ---- synchronisation [G 5.4.4] ---------------------------------------------------------------
__device__ void __syncthreads();
__device__ int __syncthreads_count(int predicate);
__device__ int __syncthreads_and(int predicate);
__device__ int __syncthreads_or(int predicate);
__device__ void __syncwarp(unsigned mask = 0xFFFFFFFF);

// ---- warp functions [G 5.4.6] ------------------------------------------------------------------
__device__ unsigned __activemask();
__device__ int __all_sync(unsigned mask, int predicate);
__device__ int __any_sync(unsigned mask, int predicate);
__device__ unsigned __ballot_sync(unsigned mask, int predicate);
// The default `width` is warpSize in the guide; it is written as 32 here.
#define PMPP_STUB_WARP_TYPE(T)                                                        \
  __device__ T __shfl_sync(unsigned mask, T value, int srcLane, int width = 32);      \
  __device__ T __shfl_up_sync(unsigned mask, T value, unsigned delta, int width = 32); \
  __device__ T __shfl_down_sync(unsigned mask, T value, unsigned delta, int width = 32); \
  __device__ T __shfl_xor_sync(unsigned mask, T value, int laneMask, int width = 32); \
  __device__ unsigned __match_any_sync(unsigned mask, T value);                       \
  __device__ unsigned __match_all_sync(unsigned mask, T value, int* pred);
PMPP_STUB_WARP_TYPE(int)
PMPP_STUB_WARP_TYPE(unsigned)
PMPP_STUB_WARP_TYPE(long)
PMPP_STUB_WARP_TYPE(unsigned long)
PMPP_STUB_WARP_TYPE(long long)
PMPP_STUB_WARP_TYPE(unsigned long long)
PMPP_STUB_WARP_TYPE(float)
PMPP_STUB_WARP_TYPE(double)
#undef PMPP_STUB_WARP_TYPE

// ---- integer intrinsics [?] UNVERIFIED: not on the guide pages that were read -------------------
__device__ int __popc(unsigned x);
__device__ int __popcll(unsigned long long x);
__device__ int __clz(int x);
__device__ int __clzll(long long x);
__device__ int __ffs(int x);
__device__ int __ffsll(long long x);
__device__ unsigned __brev(unsigned x);
__device__ unsigned long long __brevll(unsigned long long x);

// ---- more integer intrinsics [M group__CUDA__MATH__INTRINSIC__INT] -----------------------------
// (The manual adds a __tile__ qualifier to __umulhi, __mulhi and __umul64hi; it is left out.)
// HIP has no __dp4a (x2-facts B1).
__device__ int __dp4a(int srcA, int srcB, int c);
__device__ int __dp4a(char4 srcA, char4 srcB, int c);
__device__ unsigned int __dp4a(unsigned int srcA, unsigned int srcB, unsigned int c);
__device__ unsigned int __dp4a(uchar4 srcA, uchar4 srcB, unsigned int c);
__device__ unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s);
__device__ unsigned int __umulhi(unsigned int x, unsigned int y);
__device__ int __mulhi(int x, int y);
__device__ unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y);

// ---- type casting intrinsics [M group__CUDA__MATH__INTRINSIC__CAST] ------------------------------
__device__ unsigned int __float_as_uint(float x);
__device__ float __uint_as_float(unsigned int x);
__device__ int __float_as_int(float x);
__device__ float __int_as_float(int x);

// ---- single-precision math [M group__CUDA__MATH__SINGLE, group__CUDA__MATH__INTRINSIC__SINGLE] ----
// The manual marks all of these __device__ only. Host code calls the host library's
// functions of the same names (<cmath>); clang keeps the two apart by their attributes.
__device__ float expf(float x);
__device__ float exp2f(float x);
__device__ float logf(float x);
__device__ float log2f(float x);
__device__ float sqrtf(float x);
__device__ float rsqrtf(float x);
__device__ float powf(float x, float y);
__device__ float fmaf(float x, float y, float z);
__device__ float tanhf(float x);
__device__ float erff(float x);
__device__ float fminf(float x, float y);
__device__ float fmaxf(float x, float y);
__device__ float fabsf(float x);
__device__ float copysignf(float x, float y);
__device__ float __expf(float x);
__device__ float __logf(float x);
__device__ float __powf(float x, float y);
__device__ float __sinf(float x);
__device__ float __cosf(float x);
__device__ float __fdividef(float x, float y);
__device__ float __saturatef(float x);

// ---- legacy atomic functions [G 5.4.5.1] ---------------------------------------------------------
// The overloads the guide lists, minus __half/__half2/__nv_bfloat16*, float2/float4
// (compute capability 9.x) and the 128-bit templates.
__device__ int atomicAdd(int* address, int val);
__device__ unsigned atomicAdd(unsigned* address, unsigned val);
__device__ unsigned long long atomicAdd(unsigned long long* address, unsigned long long val);
__device__ float atomicAdd(float* address, float val);
__device__ double atomicAdd(double* address, double val);
__device__ int atomicSub(int* address, int val);
__device__ unsigned atomicSub(unsigned* address, unsigned val);
__device__ int atomicExch(int* address, int val);
__device__ unsigned atomicExch(unsigned* address, unsigned val);
__device__ unsigned long long atomicExch(unsigned long long* address, unsigned long long val);
__device__ float atomicExch(float* address, float val);
__device__ int atomicMin(int* address, int val);
__device__ unsigned atomicMin(unsigned* address, unsigned val);
__device__ unsigned long long atomicMin(unsigned long long* address, unsigned long long val);
__device__ long long atomicMin(long long* address, long long val);
__device__ int atomicMax(int* address, int val);
__device__ unsigned atomicMax(unsigned* address, unsigned val);
__device__ unsigned long long atomicMax(unsigned long long* address, unsigned long long val);
__device__ long long atomicMax(long long* address, long long val);
__device__ unsigned atomicInc(unsigned* address, unsigned val);
__device__ unsigned atomicDec(unsigned* address, unsigned val);
__device__ int atomicCAS(int* address, int compare, int val);
__device__ unsigned atomicCAS(unsigned* address, unsigned compare, unsigned val);
__device__ unsigned long long atomicCAS(unsigned long long* address, unsigned long long compare, unsigned long long val);
__device__ unsigned short atomicCAS(unsigned short* address, unsigned short compare, unsigned short val);
__device__ int atomicAnd(int* address, int val);
__device__ unsigned atomicAnd(unsigned* address, unsigned val);
__device__ unsigned long long atomicAnd(unsigned long long* address, unsigned long long val);
__device__ int atomicOr(int* address, int val);
__device__ unsigned atomicOr(unsigned* address, unsigned val);
__device__ unsigned long long atomicOr(unsigned long long* address, unsigned long long val);
__device__ int atomicXor(int* address, int val);
__device__ unsigned atomicXor(unsigned* address, unsigned val);
__device__ unsigned long long atomicXor(unsigned long long* address, unsigned long long val);

// ---- runtime types [R group__CUDART__TYPES] ---------------------------------------------------------
enum cudaError {
  cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2, cudaErrorInitializationError = 3,
  cudaErrorInvalidConfiguration = 9, cudaErrorInvalidDeviceFunction = 98, cudaErrorInvalidResourceHandle = 400,
  cudaErrorNotReady = 600, cudaErrorLaunchFailure = 719, cudaErrorCooperativeLaunchTooLarge = 720,
  cudaErrorNotSupported = 801, cudaErrorUnknown = 999
};
typedef enum cudaError cudaError_t;
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1,
                      cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 };
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
                         cudaFuncAttributePreferredSharedMemoryCarveout = 9 };
enum cudaDeviceAttr { cudaDevAttrCooperativeLaunch = 95 };
#define cudaHostAllocDefault        0x00
#define cudaHostAllocPortable       0x01
#define cudaHostAllocMapped         0x02
#define cudaHostAllocWriteCombined  0x04
#define cudaMemAttachGlobal         0x01
#define cudaMemAttachHost           0x02
#define cudaEventDefault            0x00
#define cudaEventBlockingSync       0x01
#define cudaEventDisableTiming      0x02
#define cudaStreamDefault           0x00
#define cudaStreamNonBlocking       0x01
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
// [R structcudaFuncAttributes] A subset of the fields, not in the real order.
struct cudaFuncAttributes {
  int binaryVersion;
  size_t constSizeBytes;
  size_t localSizeBytes;
  int maxDynamicSharedSizeBytes;
  int maxThreadsPerBlock;
  int numRegs;
  int preferredShmemCarveout;
  int ptxVersion;
  size_t sharedSizeBytes;
};
// A subset of the real struct's fields (the ones the course reads), not in the real order.
struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem, totalConstMem, sharedMemPerBlock, sharedMemPerBlockOptin, sharedMemPerMultiprocessor;
  int regsPerBlock, regsPerMultiprocessor, warpSize, maxThreadsPerBlock,
      maxThreadsPerMultiProcessor, maxBlocksPerMultiProcessor, multiProcessorCount;
  int maxThreadsDim[3], maxGridSize[3], major, minor;
  int cooperativeLaunch, concurrentKernels, unifiedAddressing, managedMemory, integrated;
  int l2CacheSize;
};

// ---- runtime functions, C forms [R] -------------------------------------------------------------------
extern "C" {
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags = 0x01);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream = 0);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorString(cudaError_t error);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device);
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority);
cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority);
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, const void* func);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0);
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
// clang lowers `k<<<grid, block, shmem, stream>>>(...)` to a call to one of these.
int cudaConfigureCall(dim3 gridSize, dim3 blockSize, size_t sharedSize = 0, cudaStream_t stream = 0);
int __cudaPushCallConfiguration(dim3 gridSize, dim3 blockSize, size_t sharedSize = 0, void* stream = 0);
}

// ---- runtime functions, C++ forms [R group__CUDART__HIGHLEVEL] ------------------------------------------
// (The C forms of the next six take `const void* func` / `const void* symbol`; the
// C++ templates below are what a .cu file normally reaches.)
// Typed allocation: the runtime reference lists template<class T> cudaMalloc(T **devPtr,
// size_t size), so cudaMalloc(&floatPtr, bytes) compiles without a cast (ch02-facts A2).
template <class T> __host__ cudaError_t cudaMalloc(T** devPtr, size_t size);
template <class T> __host__ cudaError_t cudaMallocHost(T** ptr, size_t size, unsigned int flags = 0);
template <class T> __host__ cudaError_t cudaMallocManaged(T** devPtr, size_t size, unsigned int flags = 0x01);
template <class T> __host__ cudaError_t cudaMemcpyToSymbol(const T& symbol, const void* src, size_t count, size_t offset = 0,
                                                           enum cudaMemcpyKind kind = cudaMemcpyHostToDevice);
template <class T> __host__ cudaError_t cudaMemcpyFromSymbol(void* dst, const T& symbol, size_t count, size_t offset = 0,
                                                             enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost);
template <class T> __host__ cudaError_t cudaGetSymbolAddress(void** devPtr, const T& symbol);
template <class T> __host__ cudaError_t cudaFuncSetAttribute(T* func, enum cudaFuncAttribute attr, int value);
template <class T> __host__ cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, T* entry);
template <class T> __host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, T func, int blockSize,
                                                                                      size_t dynamicSMemSize);
template <class T> __host__ cudaError_t cudaLaunchCooperativeKernel(T* func, dim3 gridDim, dim3 blockDim, void** args,
                                                                    size_t sharedMem = 0, cudaStream_t stream = 0);
#endif  // PMPP_CUDA_STUB_H

// =============================================================================================
// <cuda/atomic> [L]: cuda/atomic, cuda/__atomic/atomic.h, cuda/std/__atomic/{scopes,order}.h,
// cuda/std/__atomic/api/common.h. The member list of atomic_ref was assembled from the
// macros in api/common.h; that the arithmetic and bitwise groups apply to atomic_ref, and
// the exact cv-qualifiers, were NOT confirmed at the expansion site (UNVERIFIED).
#if defined(PMPP_STUB_WANT_CUDA_ATOMIC) && !defined(PMPP_STUB_HAVE_CUDA_ATOMIC)
#define PMPP_STUB_HAVE_CUDA_ATOMIC
namespace cuda {
namespace std {
enum thread_scope {
  thread_scope_system = 0, thread_scope_device = 1, thread_scope_cluster = 3, thread_scope_block = 2,
  thread_scope_thread = 10
};
// The C++17 spelling in libcu++: an unscoped enum.
enum memory_order {
  memory_order_relaxed, memory_order_consume, memory_order_acquire, memory_order_release,
  memory_order_acq_rel, memory_order_seq_cst
};
}  // namespace std
using std::thread_scope;
using std::thread_scope_block;
using std::thread_scope_cluster;
using std::thread_scope_device;
using std::thread_scope_system;
using std::thread_scope_thread;
using memory_order = std::memory_order;
inline constexpr memory_order memory_order_relaxed = std::memory_order_relaxed;
inline constexpr memory_order memory_order_consume = std::memory_order_consume;
inline constexpr memory_order memory_order_acquire = std::memory_order_acquire;
inline constexpr memory_order memory_order_release = std::memory_order_release;
inline constexpr memory_order memory_order_acq_rel = std::memory_order_acq_rel;
inline constexpr memory_order memory_order_seq_cst = std::memory_order_seq_cst;

#define PMPP_STUB_ATOMIC_MEMBERS(T, CONST)                                                                       \
  __host__ __device__ bool is_lock_free() const noexcept;                                                       \
  __host__ __device__ T load(memory_order m = memory_order_seq_cst) const noexcept;                             \
  __host__ __device__ operator T() const noexcept;                                                              \
  __host__ __device__ void store(T d, memory_order m = memory_order_seq_cst) CONST noexcept;                    \
  __host__ __device__ T exchange(T d, memory_order m = memory_order_seq_cst) CONST noexcept;                    \
  __host__ __device__ bool compare_exchange_weak(T& e, T d, memory_order s, memory_order f) CONST noexcept;     \
  __host__ __device__ bool compare_exchange_strong(T& e, T d, memory_order s, memory_order f) CONST noexcept;   \
  __host__ __device__ bool compare_exchange_weak(T& e, T d, memory_order m = memory_order_seq_cst) CONST noexcept;   \
  __host__ __device__ bool compare_exchange_strong(T& e, T d, memory_order m = memory_order_seq_cst) CONST noexcept; \
  __host__ __device__ T fetch_add(T op, memory_order m = memory_order_seq_cst) CONST noexcept;                  \
  __host__ __device__ T fetch_sub(T op, memory_order m = memory_order_seq_cst) CONST noexcept;                  \
  __host__ __device__ T fetch_and(T op, memory_order m = memory_order_seq_cst) CONST noexcept;                  \
  __host__ __device__ T fetch_or(T op, memory_order m = memory_order_seq_cst) CONST noexcept;                   \
  __host__ __device__ T fetch_xor(T op, memory_order m = memory_order_seq_cst) CONST noexcept;                  \
  __host__ __device__ T fetch_max(const T& op, memory_order m = memory_order_seq_cst) CONST noexcept;           \
  __host__ __device__ T fetch_min(const T& op, memory_order m = memory_order_seq_cst) CONST noexcept;           \
  __host__ __device__ T operator++(int) CONST noexcept;                                                         \
  __host__ __device__ T operator--(int) CONST noexcept;                                                         \
  __host__ __device__ T operator++() CONST noexcept;                                                            \
  __host__ __device__ T operator--() CONST noexcept;                                                            \
  __host__ __device__ T operator+=(T op) CONST noexcept;                                                        \
  __host__ __device__ T operator-=(T op) CONST noexcept;                                                        \
  __host__ __device__ T operator&=(T op) CONST noexcept;                                                        \
  __host__ __device__ T operator|=(T op) CONST noexcept;                                                        \
  __host__ __device__ T operator^=(T op) CONST noexcept;

template <class T, thread_scope Sco = thread_scope::thread_scope_system>
struct atomic_ref {
  using value_type = T;
  __host__ __device__ explicit constexpr atomic_ref(T& ref) : pmpp_p(&ref) {}
  __host__ __device__ T operator=(T v) const noexcept;
  PMPP_STUB_ATOMIC_MEMBERS(T, const)
  T* pmpp_p;
};
template <class T, thread_scope Sco = thread_scope::thread_scope_system>
struct atomic {
  using value_type = T;
  constexpr atomic() noexcept = default;
  __host__ __device__ constexpr atomic(T d) noexcept : pmpp_v(d) {}
  atomic(const atomic&) = delete;
  __host__ __device__ T operator=(T d) noexcept;
  PMPP_STUB_ATOMIC_MEMBERS(T, )
  T pmpp_v;
};
#undef PMPP_STUB_ATOMIC_MEMBERS
}  // namespace cuda
#endif

// =============================================================================================
// <cooperative_groups.h> [G 5.7.3 "Cooperative Groups API"]. The same subset the emulator
// provides (pmpp/emu/cooperative_groups_emu.h lists what is left out).
#if defined(PMPP_STUB_WANT_COOPERATIVE_GROUPS) && !defined(PMPP_STUB_HAVE_COOPERATIVE_GROUPS)
#define PMPP_STUB_HAVE_COOPERATIVE_GROUPS
namespace cooperative_groups {
class thread_block {
 public:
  __device__ static void sync();
  __device__ static unsigned int thread_rank();
  __device__ static dim3 group_index();
  __device__ static dim3 thread_index();
  __device__ static dim3 dim_threads();
  __device__ static unsigned int num_threads();
  __device__ static unsigned int size();        // legacy alias of num_threads()
  __device__ static dim3 group_dim();           // legacy alias of dim_threads()
};
__device__ thread_block this_thread_block();

class grid_group {
 public:
  __device__ bool is_valid() const;
  __device__ void sync() const;
  __device__ static unsigned long long thread_rank();
  __device__ static unsigned long long block_rank();
  __device__ static unsigned long long num_threads();
  __device__ static unsigned long long num_blocks();
  __device__ static dim3 dim_blocks();
  __device__ static dim3 block_index();
  __device__ static unsigned long long size();  // legacy alias of num_threads()
  __device__ static dim3 group_dim();           // legacy alias of dim_blocks()
};
__device__ grid_group this_grid();

#define PMPP_STUB_GROUP_MEMBERS                                             \
  __device__ void sync() const;                                             \
  __device__ unsigned long long num_threads() const;                        \
  __device__ unsigned long long size() const;                               \
  __device__ unsigned long long thread_rank() const;                        \
  __device__ unsigned long long meta_group_size() const;                    \
  __device__ unsigned long long meta_group_rank() const;                    \
  template <class T> __device__ T shfl(T var, unsigned int src_rank) const; \
  template <class T> __device__ T shfl_up(T var, int delta) const;          \
  template <class T> __device__ T shfl_down(T var, int delta) const;        \
  __device__ int any(int predicate) const;                                  \
  __device__ int all(int predicate) const;                                  \
  __device__ unsigned int ballot(int predicate) const;

class coalesced_group {
 public:
  PMPP_STUB_GROUP_MEMBERS
};
__device__ coalesced_group coalesced_threads();

template <unsigned int Size, class ParentT = void>
class thread_block_tile {
 public:
  template <class P> __device__ thread_block_tile(const thread_block_tile<Size, P>&);
  PMPP_STUB_GROUP_MEMBERS
  template <class T> __device__ T shfl_xor(T var, int laneMask) const;
};
#undef PMPP_STUB_GROUP_MEMBERS
template <unsigned int Size, class ParentT>
__device__ thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g);

template <class T> __device__ void sync(T& group);
}  // namespace cooperative_groups
#endif

// =============================================================================================
// <cuda_fp16.h> [M] "Half Precision Conversion and Data Movement", "Half Arithmetic Functions",
// "Half2 Arithmetic Functions", struct __half, struct __half2. The same subset the emulator
// provides (pmpp/emu/cuda_fp16_emu.h lists what is left out). The __tile__ qualifier the manual
// shows on some of them is left out.
#if defined(PMPP_STUB_WANT_FP16) && !defined(PMPP_STUB_HAVE_FP16)
#define PMPP_STUB_HAVE_FP16
struct __half_raw { unsigned short x; };
struct __half2_raw { unsigned short x, y; };
struct __half {
  __half() = default;
  __host__ __device__ constexpr __half(const __half_raw& hr) : pmpp_bits(hr.x) {}
  __host__ __device__ __half(float f);
  __host__ __device__ __half(double d);
  __host__ __device__ __half(int i);
  __host__ __device__ __half(unsigned int i);
  __host__ __device__ __half(short i);
  __host__ __device__ __half(unsigned short i);
  __host__ __device__ __half(long i);
  __host__ __device__ __half(unsigned long i);
  __host__ __device__ __half(long long i);
  __host__ __device__ __half(unsigned long long i);
  __host__ __device__ __half& operator=(const __half_raw& hr);
  __host__ __device__ __half& operator=(float f);
  __host__ __device__ operator float() const;
  __host__ __device__ operator __half_raw() const;
  unsigned short pmpp_bits;
};
struct __attribute__((aligned(4))) __half2 {
  __half x, y;
  __half2() = default;
  __host__ __device__ constexpr __half2(const __half& a, const __half& b) : x(a), y(b) {}
  __host__ __device__ __half2(const __half2_raw& h2r);
  __host__ __device__ operator __half2_raw() const;
};
__host__ __device__ __half __float2half(const float a);
__host__ __device__ __half __float2half_rn(const float a);
__host__ __device__ float __half2float(const __half a);
__host__ __device__ __half2 __float2half2_rn(const float a);
__host__ __device__ __half2 __floats2half2_rn(const float a, const float b);
__host__ __device__ __half2 __float22half2_rn(const float2 a);
__host__ __device__ float2 __half22float2(const __half2 a);
__host__ __device__ __half2 __halves2half2(const __half a, const __half b);
__host__ __device__ __half2 make_half2(const __half x, const __half y);
__host__ __device__ __half2 __half2half2(const __half a);
__host__ __device__ float __low2float(const __half2 a);
__host__ __device__ float __high2float(const __half2 a);
__host__ __device__ __half __low2half(const __half2 a);
__host__ __device__ __half __high2half(const __half2 a);
__host__ __device__ unsigned short int __half_as_ushort(const __half h);
__host__ __device__ __half __ushort_as_half(const unsigned short int i);
__host__ __device__ short int __half_as_short(const __half h);
__host__ __device__ __half __short_as_half(const short int i);
__host__ __device__ __half __int2half_rn(const int i);
__device__ int __half2int_rn(const __half h);
__host__ __device__ __half __hadd(const __half a, const __half b);
__host__ __device__ __half __hsub(const __half a, const __half b);
__host__ __device__ __half __hmul(const __half a, const __half b);
__host__ __device__ __half __hdiv(const __half a, const __half b);
__device__ __half __hfma(const __half a, const __half b, const __half c);
__host__ __device__ __half __hneg(const __half a);
__host__ __device__ __half __habs(const __half a);
__host__ __device__ __half2 __hadd2(const __half2 a, const __half2 b);
__host__ __device__ __half2 __hsub2(const __half2 a, const __half2 b);
__host__ __device__ __half2 __hmul2(const __half2 a, const __half2 b);
__host__ __device__ __half2 __h2div(const __half2 a, const __half2 b);
__device__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c);
__host__ __device__ __half2 __hneg2(const __half2 a);
__host__ __device__ __half2 __habs2(const __half2 a);
__host__ __device__ __half operator+(const __half& lh, const __half& rh);
__host__ __device__ __half operator-(const __half& lh, const __half& rh);
__host__ __device__ __half operator*(const __half& lh, const __half& rh);
__host__ __device__ __half operator/(const __half& lh, const __half& rh);
__host__ __device__ __half& operator+=(__half& lh, const __half& rh);
__host__ __device__ __half operator-(const __half& h);
__host__ __device__ __half2 operator+(const __half2& h);
__host__ __device__ __half2 operator-(const __half2& h);
__host__ __device__ __half2 operator+(const __half2& lh, const __half2& rh);
__host__ __device__ __half2 operator-(const __half2& lh, const __half2& rh);
__host__ __device__ __half2 operator*(const __half2& lh, const __half2& rh);
__host__ __device__ __half2 operator/(const __half2& lh, const __half2& rh);
__host__ __device__ __half2& operator+=(__half2& lh, const __half2& rh);
__host__ __device__ __half2& operator-=(__half2& lh, const __half2& rh);
__host__ __device__ __half2& operator*=(__half2& lh, const __half2& rh);
__host__ __device__ __half2& operator/=(__half2& lh, const __half2& rh);
#endif
