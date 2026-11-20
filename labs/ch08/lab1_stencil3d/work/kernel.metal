// Lab 8.1 starter (Metal Shading Language).
// The harness defines TILE_WIDTH when it compiles this file (it tests 8, 16 and,
// where the pipeline allows 1024 threads, 32), so do not hard-code 32 anywhere.
//
// CUDA                         Metal
// __shared__ float a[..]       threadgroup float a[..]   (declared in the kernel body)
// __syncthreads()              threadgroup_barrier(mem_flags::mem_threadgroup)
// threadIdx                    [[thread_position_in_threadgroup]]
// blockIdx                     [[threadgroup_position_in_grid]]
// blockIdx * blockDim + threadIdx   [[thread_position_in_grid]]
//
// Every position attribute is a uint3: MSL requires all of a kernel's position
// attributes to have the same number of components. The positions are unsigned;
// convert to int before you subtract 1 from one of them.
#include <metal_stdlib>
using namespace metal;

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif
#define IN_TILE_DIM TILE_WIDTH
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

// Same layout as ../stencil.h (a file compiled at run time cannot include it).
struct StencilCoeffs { float c0, c1, c2, c3, c4, c5, c6; };

// Fig. 8.6: one thread per grid point.
kernel void stencil_basic(device const float*    in [[buffer(0)]],
                          device float*         out [[buffer(1)]],
                          constant uint&          N [[buffer(2)]],
                          constant StencilCoeffs& c [[buffer(3)]],
                          uint3 gid [[thread_position_in_grid]]) {
    // TODO: TODO 1 and 2 of kernel.cu, in Metal's vocabulary.
}

// Fig. 8.12: a 2D threadgroup marches through `depth` planes in z.
kernel void stencil_regtile(device const float*    in [[buffer(0)]],
                            device float*         out [[buffer(1)]],
                            constant uint&          N [[buffer(2)]],
                            constant uint&      depth [[buffer(3)]],
                            constant StencilCoeffs& c [[buffer(4)]],
                            uint3 tid [[thread_position_in_threadgroup]],
                            uint3 bid [[threadgroup_position_in_grid]]) {
    // TODO: TODO 4 to 7 of kernel.cu, in Metal's vocabulary. The same rule holds:
    //       every thread reaches both barriers on every iteration of the loop.
}
