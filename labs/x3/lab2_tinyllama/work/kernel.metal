// Lab X3.2 starter (Metal Shading Language). Not in the book.
// The advance of kernel.cu. Every other kernel of the step is an earlier lab's own
// kernel.metal, which the driver compiles.
//
// CUDA                                   Metal
// blockIdx.x * blockDim.x + threadIdx.x  [[thread_position_in_grid]]
// if (b < batch)                         not needed if launch.mm dispatches exactly batch threads
// unsigned long long                     ulong (64 bits in MSL)
#include <metal_stdlib>
using namespace metal;

// One thread per sequence. launch.mm binds the buffers and scalars at these indices.
kernel void advance_positions(device int*        tokens    [[buffer(0)]],
                              device int*        pos       [[buffer(1)]],
                              device int*        len       [[buffer(2)]],
                              device const int*  prompt    [[buffer(3)]],
                              device int*        out       [[buffer(4)]],
                              device ulong*      rngStep   [[buffer(5)]],
                              constant uint&     promptLen [[buffer(6)]],
                              constant uint&     batch     [[buffer(7)]],
                              uint b [[thread_position_in_grid]]) {
    // TODO 1: as advancePositionsKernel in kernel.cu.
}
