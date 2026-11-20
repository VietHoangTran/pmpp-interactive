// Stub declarations for a CUDA *front-end* check on machines with no CUDA SDK.
//
// Used only as:  clang++ -x cuda --cuda-host-only -nocudainc -nocudalib
//                        -fsyntax-only -include pmpp/check/cuda_stub.h file.cu
//
// clang's CUDA mode then parses kernel launches (<<< >>>) and enforces the
// host/device call rules for real. What this does NOT establish: that these
// hand-written prototypes match NVIDIA's headers, that nvcc accepts the file,
// or anything about code generation. Nothing here is ever linked or run.
#pragma once
#include <stddef.h>

#define __global__   __attribute__((global))
#define __device__   __attribute__((device))
#define __host__     __attribute__((host))
#define __shared__   __attribute__((shared))
#define __constant__ __attribute__((constant))

struct uint3 { unsigned x, y, z; };
struct dim3 {
  unsigned x, y, z;
  __host__ __device__ dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
extern __device__ const uint3 threadIdx;
extern __device__ const uint3 blockIdx;
extern __device__ const dim3 blockDim;
extern __device__ const dim3 gridDim;
__device__ void __syncthreads();

typedef int cudaError_t;
enum { cudaSuccess = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1,
                      cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };
typedef struct CUstream_st* cudaStream_t;
struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem, sharedMemPerBlock, sharedMemPerMultiprocessor;
  int regsPerBlock, regsPerMultiprocessor, warpSize, maxThreadsPerBlock,
      maxThreadsPerMultiProcessor, maxBlocksPerMultiProcessor, multiProcessorCount;
  int maxThreadsDim[3], maxGridSize[3], major, minor;
};
extern "C" {
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorString(cudaError_t error);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
// clang lowers `k<<<grid, block, shmem, stream>>>(...)` to a call to one of these.
int cudaConfigureCall(dim3 gridSize, dim3 blockSize, size_t sharedSize = 0, cudaStream_t stream = 0);
int __cudaPushCallConfiguration(dim3 gridSize, dim3 blockSize, size_t sharedSize = 0, void* stream = 0);
}
