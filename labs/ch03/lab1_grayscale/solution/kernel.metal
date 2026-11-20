// Lab 3.1 reference solution (Metal Shading Language). Not in the book.
//
// CUDA                                      Metal
// blockIdx.x*blockDim.x + threadIdx.x       gid.x   ([[thread_position_in_grid]])
// blockIdx.y*blockDim.y + threadIdx.y       gid.y
// unsigned char*                            device uchar*  [[buffer(n)]]
// int width (by value)                      constant uint& width, bound with setBytes
//
// The picture is three bytes per pixel, so it is read through a uchar pointer.
// A uchar3 is four bytes in this language: a `device const uchar3*` would step
// four bytes per pixel and read the wrong data.
#include <metal_stdlib>
using namespace metal;

#define CHANNELS 3

kernel void grayscale(device const uchar* Pin    [[buffer(0)]],
                      device uchar*       Pout   [[buffer(1)]],
                      constant uint&      width  [[buffer(2)]],
                      constant uint&      height [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x;
    uint row = gid.y;
    // No range test: launch.mm uses dispatchThreads with a grid of exactly
    // width x height threads, so every thread has a pixel. (`height` is bound so
    // that both backends take the same arguments; this kernel does not need it.)
    uint grayOffset = row*width + col;
    uint rgbOffset = grayOffset*CHANNELS;
    uchar r = Pin[rgbOffset    ];
    uchar g = Pin[rgbOffset + 1];
    uchar b = Pin[rgbOffset + 2];
    // Same grammar as C++: without the outer parentheses the cast would apply
    // to 0.299f alone. Write the f suffix; this language has no double.
    Pout[grayOffset] = (uchar)(0.299f*r + 0.587f*g + 0.114f*b);
}
