// Lab 3.2 starter (Metal Shading Language). Not in the book.
//
// CUDA                                      Metal
// blockIdx.x*blockDim.x + threadIdx.x       gid.x   ([[thread_position_in_grid]])
// blockIdx.y*blockDim.y + threadIdx.y       gid.y
// unsigned char*                            device uchar*  [[buffer(n)]]
// int w (by value)                          constant int& w, bound with setBytes
#include <metal_stdlib>
using namespace metal;

// in, out: h x w pixels, row-major, 1 byte per pixel. r is the patch radius.
kernel void blur(device const uchar* in  [[buffer(0)]],
                 device uchar*       out [[buffer(1)]],
                 constant int&       w   [[buffer(2)]],
                 constant int&       h   [[buffer(3)]],
                 constant int&       r   [[buffer(4)]],
                 uint2 gid [[thread_position_in_grid]]) {
    // TODO: the same kernel as in kernel.cu. Two things to settle first:
    //   - gid is unsigned. What is gid.y + blurRow when gid.y is 0 and blurRow
    //     is -1, and what does that do to a `>= 0` test?
    //   - launch.mm rounds the number of threadgroups up. Does this kernel need
    //     the outer range test?
}
