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
