// Lab 3.5 reference solution (Metal Shading Language): the three kernel defects in
// the starter, fixed. The fourth is in launch.mm. Not in the book.
//
// CUDA                                      Metal
// blockIdx.x*blockDim.x + threadIdx.x       gid.x   ([[thread_position_in_grid]])
// blockIdx.y*blockDim.y + threadIdx.y       gid.y
// blockIdx.z*blockDim.z + threadIdx.z       gid.z
// const float* / float*                     device const float* / device float*  [[buffer(n)]]
// int seq (by value)                        constant int& seq, bound with setBytes
//
// The position attribute is unsigned. The window needs `t + d` to go negative
// and be tested against 0, so the coordinates are converted to int first.
#include <metal_stdlib>
using namespace metal;

kernel void seq_mean(device const float* in     [[buffer(0)]],
                     device float*       out    [[buffer(1)]],
                     constant int&       batch  [[buffer(2)]],
                     constant int&       seq    [[buffer(3)]],
                     constant int&       hidden [[buffer(4)]],
                     constant int&       r      [[buffer(5)]],
                     uint3 gid [[thread_position_in_grid]]) {
    int h = int(gid.x);   // column: position along hidden
    int t = int(gid.y);   // row:    position along seq
    int b = int(gid.z);   // plane:  which sequence of the batch

    // FIX B: launch.mm dispatches whole threadgroups on all three axes, so all
    // three indices need the test. The starter left out `b < batch`.
    if (h < hidden && t < seq && b < batch) {
        float sum = 0.0f;
        int count = 0;
        for (int d = -r; d < r+1; ++d) {
            int cur = t + d;
            // FIX C: clip the window at both ends. The starter tested only `cur < seq`.
            if (cur >= 0 && cur < seq) {
                sum += in[b*seq*hidden + cur*hidden + h];
                ++count;   // how many positions went into the sum
            }
        }
        // FIX A: the row stride is hidden. The starter wrote `t*seq`, so with
        // seq < hidden two threads wrote one element, and with seq > hidden rows
        // spilled past their plane.
        out[b*seq*hidden + t*hidden + h] = sum/float(count);
    }
}
