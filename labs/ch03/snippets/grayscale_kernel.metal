#include <metal_stdlib>
using namespace metal;

#define CHANNELS 3   // bytes per colour pixel: r, g, b

// The same kernel in Metal Shading Language. gid is the thread's position in the
// whole grid: .x is the column, .y the row. Nothing has to be multiplied out.
kernel void color_to_grayscale(device const uchar* Pin  [[buffer(0)]],
                               device uchar*       Pout [[buffer(1)]],
                               constant uint2&     size [[buffer(2)]],   // (width, height)
                               uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x;
    uint row = gid.y;
    // Needed when the launch is dispatchThreadgroups. With dispatchThreads over
    // (width, height) it is never false.
    if (col < size.x && row < size.y) {
        uint grayOffset = row * size.x + col;
        uint rgbOffset = grayOffset * CHANNELS;
        uchar r = Pin[rgbOffset];
        uchar g = Pin[rgbOffset + 1];
        uchar b = Pin[rgbOffset + 2];
        Pout[grayOffset] = (uchar)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}
