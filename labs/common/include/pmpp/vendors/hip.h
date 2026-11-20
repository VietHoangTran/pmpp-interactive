// Build the course's CUDA sources for AMD GPUs by mapping CUDA runtime names to
// their HIP counterparts, the same approach llama.cpp uses for its HIP backend.
// Kernel-side syntax (__global__, __shared__, threadIdx, __syncthreads, <<< >>>)
// is accepted by HIP as written.
//
// STATUS: UNVERIFIED on real hardware. Written on a Mac with no ROCm install.
// The names were checked against AMD's documentation and the ROCm 10.2 headers
// (notes/today/ch05-facts.md, group B), but this file has not been compiled
// against them. If a name is wrong, `./verify.sh` on a ROCm machine will show
// it in the build log.
//
// The names below the first block come from notes/today/infra-facts.md, group B
// (items B4, B5, B8, B9). Each was found in two places: AMD's HIPIFY table
// "CUDA Runtime API supported by HIP" and hip_runtime_api.h at tag therock-10.2.
// What a #define cannot fix, and a lab must handle itself (same file, items
// B2, B3, B4, B11, B12):
//   - warp intrinsics keep their names, but the mask must be a 64-bit integer
//     under HIP and __ballot_sync/__activemask return unsigned long long;
//   - <cooperative_groups.h> is <hip/hip_cooperative_groups.h>, <cuda_fp16.h>
//     is <hip/hip_fp16.h>, <cub/cub.cuh> is <hipcub/hipcub.hpp> (namespace
//     hipcub), <mma.h> has no HIP counterpart;
//   - <cuda/atomic> keeps its name but needs libhipcxx (ROCm 10.1 or later);
//   - atomicAdd_block and the other _block atomics do not exist in HIP.
#pragma once
#include <hip/hip_runtime.h>

#define cudaError_t             hipError_t
#define cudaSuccess             hipSuccess
#define cudaStream_t            hipStream_t
#define cudaDeviceProp          hipDeviceProp_t
#define cudaMemcpyHostToHost    hipMemcpyHostToHost
#define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMalloc              hipMalloc
#define cudaFree                hipFree
#define cudaMemcpy              hipMemcpy
#define cudaMemset              hipMemset
#define cudaDeviceSynchronize   hipDeviceSynchronize
#define cudaGetLastError        hipGetLastError
#define cudaPeekAtLastError     hipPeekAtLastError
#define cudaGetErrorString      hipGetErrorString
#define cudaGetDeviceCount      hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties

// infra-facts.md B5 (constant memory and device symbols).
#define cudaMemcpyToSymbol      hipMemcpyToSymbol
#define cudaMemcpyFromSymbol    hipMemcpyFromSymbol
#define cudaGetSymbolAddress    hipGetSymbolAddress

// infra-facts.md B8 (streams and asynchronous copies).
#define cudaStreamCreate          hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy         hipStreamDestroy
#define cudaStreamSynchronize     hipStreamSynchronize
#define cudaStreamQuery           hipStreamQuery
#define cudaStreamWaitEvent       hipStreamWaitEvent
#define cudaStreamDefault         hipStreamDefault
#define cudaStreamNonBlocking     hipStreamNonBlocking
#define cudaMemcpyAsync           hipMemcpyAsync
#define cudaMemcpyDefault         hipMemcpyDefault

// infra-facts.md B8 (pinned host memory). Not a plain prefix swap: HIP's own
// hipMallocHost is deprecated. A two-argument cudaMallocHost(&p, n) relies on
// HIP's C++ template overload of hipHostMalloc, which defaults the flags.
#define cudaMallocHost             hipHostMalloc
#define cudaHostAlloc              hipHostAlloc
#define cudaFreeHost               hipHostFree
#define cudaHostAllocDefault       hipHostMallocDefault
#define cudaHostAllocPortable      hipHostMallocPortable
#define cudaHostAllocMapped        hipHostMallocMapped
#define cudaHostAllocWriteCombined hipHostMallocWriteCombined

// infra-facts.md B8 (events, used for timing).
#define cudaEvent_t              hipEvent_t
#define cudaEventCreate          hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord          hipEventRecord
#define cudaEventSynchronize     hipEventSynchronize
#define cudaEventQuery           hipEventQuery
#define cudaEventElapsedTime     hipEventElapsedTime
#define cudaEventDestroy         hipEventDestroy
#define cudaEventDefault         hipEventDefault
#define cudaEventBlockingSync    hipEventBlockingSync
#define cudaEventDisableTiming   hipEventDisableTiming
#define cudaErrorNotReady        hipErrorNotReady

// infra-facts.md B9 (occupancy).
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxPotentialBlockSize            hipOccupancyMaxPotentialBlockSize

// infra-facts.md B4 (cooperative launch and device attributes). HIP's
// hipLaunchCooperativeKernel has no default arguments: pass all six.
#define cudaLaunchCooperativeKernel    hipLaunchCooperativeKernel
#define cudaDeviceGetAttribute         hipDeviceGetAttribute
#define cudaDeviceAttr                 hipDeviceAttribute_t
#define cudaDevAttrCooperativeLaunch   hipDeviceAttributeCooperativeLaunch
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrWarpSize            hipDeviceAttributeWarpSize

// infra-facts.md B4 (launch with attributes, the route to thread block
// clusters). hipLaunchKernelEx is in the header but not in the HIPIFY table.
// Which AMD devices support cluster launch was not checked.
#define cudaLaunchKernelEx                  hipLaunchKernelEx
#define cudaLaunchKernelExC                 hipLaunchKernelExC
#define cudaLaunchConfig_t                  hipLaunchConfig_t
#define cudaLaunchAttribute                 hipLaunchAttribute
#define cudaLaunchAttributeClusterDimension hipLaunchAttributeClusterDimension
