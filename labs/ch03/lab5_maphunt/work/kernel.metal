// Lab 3.5 starter (Metal Shading Language). Not in the book.
// The same kernel as kernel.cu. Together with launch.mm it has the same FOUR
// defects, in the same places.
//
// CUDA                                      Metal
// blockIdx.x*blockDim.x + threadIdx.x       gid.x   ([[thread_position_in_grid]])
// blockIdx.y*blockDim.y + threadIdx.y       gid.y
// blockIdx.z*blockDim.z + threadIdx.z       gid.z
// const float* / float*                     device const float* / device float*  [[buffer(n)]]
// int seq (by value)                        constant int& seq, bound with setBytes
//
// The position attribute is unsigned. The window needs `t + d` to go negative,
// so the coordinates are converted to int first.
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

    if (h < hidden && t < seq) {
        float sum = 0.0f;
        int count = 0;
        for (int d = -r; d < r+1; ++d) {
            int cur = t + d;
            if (cur < seq) {
                sum += in[b*seq*hidden + cur*hidden + h];
                ++count;   // how many positions went into the sum
            }
        }
        out[b*seq*hidden + t*seq + h] = sum/float(count);
    }
}
