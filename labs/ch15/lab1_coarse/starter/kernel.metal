// Lab 15.1 starter (Metal Shading Language).
// The same kernel as kernel.cu, in Metal's vocabulary: a threadgroup of 256 threads
// computes a 128 x 128 tile of C, each thread an 8 x 8 patch of it held in registers.
//   C[M x N] = A[M x K] * B[K x N], row-major, any sizes.
// Book: section 15.3 (pp. 352-357), Figs. 15.3 to 15.7.
//
// CUDA                            Metal
// __device__ __forceinline__      METAL_FUNC: Apple's macro for inline
//                                 __attribute__((__always_inline__)), from <metal_stdlib>;
//                                 not in the MSL specification (notes/today/ch15-facts.md C4)
// #pragma unroll                  #pragma clang loop unroll(full): accepted by Apple's
//                                 compiler, used by MLX; Apple documents no unroll control (C4)
// __launch_bounds__(256)          [[max_total_threads_per_threadgroup(256)]] (given below)
// __shared__ float a[..]          threadgroup float a[..]   (declared in the kernel body)
// __syncthreads()                 threadgroup_barrier(mem_flags::mem_threadgroup)
// threadIdx.x (one-dimensional)   [[thread_index_in_threadgroup]]; pass it to the helpers
// blockIdx                        [[threadgroup_position_in_grid]]
// pointer into shared memory      threadgroup float*;  into a buffer: device float*;
//                                 into a thread's own array: thread float*
//
// Use int, not uint, for indices (as in lab 6.2).
#include <metal_stdlib>
using namespace metal;

#define bM 128          // block-level output tile: bM x bN
#define bN 128
#define bK 8            // depth of the input tiles in threadgroup memory
#define tM 8            // thread-level output tile: tM x tN
#define tN 8
#define NUM_THREADS 256 // the book's NUM_THREADS_PER_BLOCK

struct GemmSizes { int M; int K; int N; };

// TODO 1 to 4: the helpers of kernel.cu, with METAL_FUNC in place of
// static __device__ __forceinline__, address-space qualifiers on every pointer, and the
// thread's index t as an extra argument to loadTile.
METAL_FUNC void clear(thread float C_r[][tN], int m, int n) {
}

METAL_FUNC void loadTile(device const float* T, int lda, int maxRow, int maxCol,
                         threadgroup float* T_s, int ldas, int height, int width, int t) {
}

METAL_FUNC void mm(int m, int n, int k, threadgroup const float* a, int lda,
                   threadgroup const float* b, int ldb, thread float c[][tN]) {
}

METAL_FUNC void writeTile(device float* c, int ldc, int maxRow, int maxCol,
                          thread float C_r[][tN], int m, int n) {
}

[[max_total_threads_per_threadgroup(NUM_THREADS)]]
kernel void gemm_coarse(device const float* A [[buffer(0)]],
                        device const float* B [[buffer(1)]],
                        device float*       C [[buffer(2)]],
                        constant GemmSizes& s [[buffer(3)]],
                        uint  tid [[thread_index_in_threadgroup]],
                        uint2 bid [[threadgroup_position_in_grid]]) {
    // TODO 5 to 9: the kernel body of kernel.cu.
}
