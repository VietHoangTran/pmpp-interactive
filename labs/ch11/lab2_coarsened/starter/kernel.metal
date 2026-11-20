// Lab 11.2 starter (Metal Shading Language). The kernels of kernel.cu:
// (a) coarsened_scan, (b) register_tiled_scan, and the block scan of lab 11.1.
//
// CUDA                                  Metal
// template <unsigned COARSE_FACTOR>     the macro COARSE_FACTOR, set by the driver when it
//                                       builds the pipeline (one pipeline per factor)
// extern __shared__ float buffer_s[];   threadgroup float* buffer_s [[threadgroup(0)]],
//   + bytes in <<<grid, block, bytes>>>   sized by setThreadgroupMemoryLength in launch.mm
// #pragma unroll                        none: MSL documents no unroll control; the loops
//                                       have constant trip counts and the compiler decides
//
// launch.mm dispatches only blocks that are whole SIMD-groups, at most `width` of them,
// and only when the threadgroup memory fits the device.
#include <metal_stdlib>
using namespace metal;

#ifndef COARSE_FACTOR
#error "COARSE_FACTOR is set by the lab driver when it builds the pipeline"
#endif

#define MAX_BLOCK_DIM 1024

// ---------------------------------------------------------------- from lab 11.1
// TODO 0: replace this stub with warp_scan and block_scan_shuffle (renamed block_scan)
//         from your lab 11.1 kernel.metal, unchanged, and define the size of the
//         SIMD-group totals array as you did there.
inline float block_scan(float val, threadgroup float* warpSums_s,
                        uint lane, uint sg, uint width, uint groups) {
    return val;
}

// ---------------------------------------------------------------- (a)
kernel void coarsened_scan(device const float* input    [[buffer(0)]],
                           device float*       output   [[buffer(1)]],
                           constant uint&      N        [[buffer(2)]],
                           threadgroup float*  buffer_s [[threadgroup(0)]],
                           uint t      [[thread_position_in_threadgroup]],
                           uint b      [[threadgroup_position_in_grid]],
                           uint bdim   [[threads_per_threadgroup]],
                           uint width  [[threads_per_simdgroup]],
                           uint lane   [[thread_index_in_simdgroup]],
                           uint sg     [[simdgroup_index_in_threadgroup]],
                           uint groups [[simdgroups_per_threadgroup]]) {
    // TODO 1: the steps of CoarsenedScanKernel in kernel.cu. buffer_s is already here,
    //         sized by launch.mm; declare threadSums and the SIMD-group totals in the
    //         kernel body.
}

// ---------------------------------------------------------------- (b)
kernel void register_tiled_scan(device const float* input    [[buffer(0)]],
                                device float*       output   [[buffer(1)]],
                                constant uint&      N        [[buffer(2)]],
                                threadgroup float*  buffer_s [[threadgroup(0)]],
                                uint t      [[thread_position_in_threadgroup]],
                                uint b      [[threadgroup_position_in_grid]],
                                uint bdim   [[threads_per_threadgroup]],
                                uint width  [[threads_per_simdgroup]],
                                uint lane   [[thread_index_in_simdgroup]],
                                uint sg     [[simdgroup_index_in_threadgroup]],
                                uint groups [[simdgroups_per_threadgroup]]) {
    // TODO 3: the steps of RegisterTiledScanKernel in kernel.cu.
}
