// Lab 20.4 starter (Metal Shading Language). The same two kernels as kernel.cu, with the
// same layouts and contract (README.md).
//
// CUDA                               Metal
// __shared__ (in the kernel)         threadgroup (in the kernel: helpers cannot declare it)
// threadIdx.x % 32, threadIdx.x / 32 [[thread_index_in_simdgroup]], [[simdgroup_index_in_threadgroup]]
// a butterfly of __shfl_xor_sync     simd_sum(x): the sum over the SIMD-group, in every lane
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
//
// The driver checks that the pipeline's SIMD-groups have 32 lanes (Apple does not
// document the width as a number) and dispatches 256 threads per threadgroup. Under
// Metal's default fast math infinities are undefined: keep the running max finite (M_INIT).
#include <metal_stdlib>
using namespace metal;

#define HEAD_DIM 128
#define SIMD_WIDTH 32
#define BLOCK_SIZE 256
#define N_WARPS (BLOCK_SIZE / SIMD_WIDTH)
#define D_PER_LANE (HEAD_DIM / SIMD_WIDTH)
#define M_INIT (-FLT_MAX / 2.0f)

// One thread per element of k_new (and v_new).
kernel void kv_append(device const float* k_new [[buffer(0)]],
                      device const float* v_new [[buffer(1)]],
                      device float*       Kc    [[buffer(2)]],
                      device float*       Vc    [[buffer(3)]],
                      device const int*   pos   [[buffer(4)]],
                      constant uint&      batch [[buffer(5)]],
                      constant uint&      maxSeq [[buffer(6)]],
                      constant uint&      hKv   [[buffer(7)]],
                      uint i [[thread_position_in_grid]]) {
    // TODO: kernel.cu's TODO 2.
}

// Threadgroup (h, b) = (query head, sequence).
kernel void decode_attention(device const float* q     [[buffer(0)]],
                             device const float* Kc    [[buffer(1)]],
                             device const float* Vc    [[buffer(2)]],
                             device const int*   len   [[buffer(3)]],
                             device float*       out   [[buffer(4)]],
                             constant uint&      hQ    [[buffer(5)]],
                             constant uint&      hKv   [[buffer(6)]],
                             constant uint&      maxSeq [[buffer(7)]],
                             constant float&     scale [[buffer(8)]],
                             uint2 tg   [[threadgroup_position_in_grid]],
                             uint  tid  [[thread_index_in_threadgroup]],
                             uint  lane [[thread_index_in_simdgroup]],
                             uint  w    [[simdgroup_index_in_threadgroup]]) {
    // TODO: kernel.cu's TODOs 4 to 8, in Metal's vocabulary. simd_sum replaces warpSumAll.
}
