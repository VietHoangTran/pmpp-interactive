// Lab X3.2 reference solution (Metal Shading Language). Not in the book.
// The advance of kernel.cu. Every other kernel of the step is an earlier lab's own
// kernel.metal, which the driver compiles.
//
// CUDA                                   Metal
// blockIdx.x * blockDim.x + threadIdx.x  [[thread_position_in_grid]]
// if (b < batch)                         not needed: launch.mm dispatches exactly batch threads
// unsigned long long                     ulong (64 bits in MSL)
#include <metal_stdlib>
using namespace metal;

// One thread per sequence: record this step's argmax at its position, feed the next prompt
// token if the sequence is still inside its prompt, move the position on.
kernel void advance_positions(device int*        tokens    [[buffer(0)]],
                              device int*        pos       [[buffer(1)]],
                              device int*        len       [[buffer(2)]],
                              device const int*  prompt    [[buffer(3)]],
                              device int*        out       [[buffer(4)]],
                              device ulong*      rngStep   [[buffer(5)]],
                              constant uint&     promptLen [[buffer(6)]],
                              constant uint&     batch     [[buffer(7)]],
                              uint b [[thread_position_in_grid]]) {
    const int p = pos[b];
    out[uint(p) * batch + b] = tokens[b];
    if (uint(p + 1) < promptLen) {
        tokens[b] = prompt[uint(p + 1) * batch + b];
    }
    pos[b] = p + 1;
    len[b] = p + 2;
    rngStep[b] += 1;
}
