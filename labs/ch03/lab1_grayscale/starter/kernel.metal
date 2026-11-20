// Lab 3.1 starter (Metal Shading Language). Not in the book.
//
// CUDA                                      Metal
// blockIdx.x*blockDim.x + threadIdx.x       gid.x   ([[thread_position_in_grid]])
// blockIdx.y*blockDim.y + threadIdx.y       gid.y
// unsigned char*                            device uchar*  [[buffer(n)]]
// int width (by value)                      constant uint& width, bound with setBytes
#include <metal_stdlib>
using namespace metal;

#define CHANNELS 3

// Pin:  height x width pixels, row-major, 3 consecutive bytes per pixel.
// Pout: height x width pixels, row-major, 1 byte per pixel.
kernel void grayscale(device const uchar* Pin    [[buffer(0)]],
                      device uchar*       Pout   [[buffer(1)]],
                      constant uint&      width  [[buffer(2)]],
                      constant uint&      height [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    // TODO: the same kernel as in kernel.cu. `gid` is already this thread's
    //       position in the whole grid. Whether you need a range test depends on
    //       how launch.mm sizes the grid: decide that first.
    //       Float literals need the f suffix here.
}
