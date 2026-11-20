// pmpp CPU emulator, part 1 of 4: types, qualifiers and constants.
// Included by pmpp/emu/cuda_emu.h; do not include directly. See that file for
// what the emulator models and what it does not.
#pragma once
#include <cstddef>
#include <cstdint>

// ---- function and variable qualifiers ---------------------------------------
// The emulator compiles a .cu file as ordinary host C++, so the qualifiers that
// tell nvcc where code runs expand to nothing. That host code may not call
// __device__ functions (and the reverse) is checked by the front-end check, not here.
#define __global__
#define __device__
#define __host__
#define __constant__          /* a plain global here: kernel writes are NOT rejected */
#define __forceinline__ inline __attribute__((always_inline))
// A register-budget hint for the GPU compiler; nothing to do on the CPU.
#define __launch_bounds__(...)
#define __inline_hint__
// libc++ and libstdc++ spell an attribute __attribute__((__noinline__)) in their
// own headers, which this macro would break. cuda_emu.h includes the standard
// headers a kernel file normally needs before this point; a standard header
// first included by a .cu file after it may fail to compile in the emulator.
#define __noinline__ __attribute__((noinline))
// __restrict__ is a keyword of GCC and clang already.

// A __shared__ declaration that reaches the compiler was not rewritten by
// labs/tools/cu2emu.py (see there for the forms it understands).
#define __shared__ _Pragma("GCC error \"pmpp-emu: this __shared__ declaration was not understood by labs/tools/cu2emu.py\"") static

// ---- vector types (vector_types.h / vector_functions.h in the CUDA toolkit) --
// Field names and order follow CUDA. The alignments (8 for the 2-vectors, 16
// for the 4-vectors) are from memory of vector_types.h and were not re-checked;
// they do not change sizeof.
struct uint3 { unsigned x, y, z; };
struct alignas(8)  uint2 { unsigned x, y; };
struct alignas(16) uint4 { unsigned x, y, z, w; };
struct alignas(8)  int2 { int x, y; };
struct int3 { int x, y, z; };
struct alignas(16) int4 { int x, y, z, w; };
struct alignas(8)  float2 { float x, y; };
struct float3 { float x, y, z; };
struct alignas(16) float4 { float x, y, z, w; };
struct alignas(16) double2 { double x, y; };
struct double3 { double x, y, z; };
struct alignas(16) double4 { double x, y, z, w; };

inline uint2 make_uint2(unsigned x, unsigned y) { return uint2{x, y}; }
inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) { return uint3{x, y, z}; }
inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) { return uint4{x, y, z, w}; }
inline int2 make_int2(int x, int y) { return int2{x, y}; }
inline int3 make_int3(int x, int y, int z) { return int3{x, y, z}; }
inline int4 make_int4(int x, int y, int z, int w) { return int4{x, y, z, w}; }
inline float2 make_float2(float x, float y) { return float2{x, y}; }
inline float3 make_float3(float x, float y, float z) { return float3{x, y, z}; }
inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }
inline double2 make_double2(double x, double y) { return double2{x, y}; }
inline double3 make_double3(double x, double y, double z) { return double3{x, y, z}; }
inline double4 make_double4(double x, double y, double z, double w) { return double4{x, y, z, w}; }

struct dim3 {
  unsigned x, y, z;
  constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
  constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
  constexpr operator uint3() const { return uint3{x, y, z}; }
};

// ---- runtime types ----------------------------------------------------------
// Numeric values of the enumerators and flags below were checked against the
// CUDA Runtime API reference 13.5 (group__CUDART__TYPES), 2026-11-20.
enum cudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorInvalidConfiguration = 9,
  cudaErrorInvalidDeviceFunction = 98,
  cudaErrorInvalidResourceHandle = 400,
  cudaErrorNotReady = 600,
  cudaErrorLaunchFailure = 719,
  cudaErrorCooperativeLaunchTooLarge = 720,
  cudaErrorNotSupported = 801,
  cudaErrorUnknown = 999
};
typedef enum cudaError cudaError_t;

enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1,
                      cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3,
                      cudaMemcpyDefault = 4 };

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

typedef struct pmpp_emu_stream* cudaStream_t;
typedef struct pmpp_emu_event* cudaEvent_t;

// Only the fields this course reads; the real struct has many more. Values are
// synthetic: they describe the emulator, not any real GPU.
struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
  size_t totalConstMem;
  int l2CacheSize;
  size_t sharedMemPerBlock;
  size_t sharedMemPerBlockOptin;
  size_t sharedMemPerMultiprocessor;
  int regsPerBlock;
  int regsPerMultiprocessor;
  int warpSize;
  int maxThreadsPerBlock;
  int maxThreadsPerMultiProcessor;
  int maxBlocksPerMultiProcessor;
  int multiProcessorCount;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int major, minor;
  int cooperativeLaunch;
  int concurrentKernels;
  int unifiedAddressing;
  int managedMemory;
  int integrated;
};

// The real `warpSize` is a run-time value ("commonly 32"); here it is fixed.
inline constexpr int warpSize = 32;

// Real CUDA declares these four `const`. They are per-thread variables here, set
// by the emulator before a kernel body runs; assigning to one is an error that
// only the front-end check and nvcc report.
inline thread_local uint3 threadIdx{0, 0, 0};
inline thread_local uint3 blockIdx{0, 0, 0};
inline thread_local dim3 blockDim{1, 1, 1};
inline thread_local dim3 gridDim{1, 1, 1};
