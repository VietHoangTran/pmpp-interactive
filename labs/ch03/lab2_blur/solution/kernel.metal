// Lab 3.2 reference solution (Metal Shading Language). Not in the book.
//
// CUDA                                      Metal
// blockIdx.x*blockDim.x + threadIdx.x       gid.x   ([[thread_position_in_grid]])
// blockIdx.y*blockDim.y + threadIdx.y       gid.y
// unsigned char*                            device uchar*  [[buffer(n)]]
// int w (by value)                          constant int& w, bound with setBytes
#include <metal_stdlib>
using namespace metal;

kernel void blur(device const uchar* in  [[buffer(0)]],
                 device uchar*       out [[buffer(1)]],
                 constant int&       w   [[buffer(2)]],
                 constant int&       h   [[buffer(3)]],
                 constant int&       r   [[buffer(4)]],
                 uint2 gid [[thread_position_in_grid]]) {
    // gid is unsigned. Convert before adding a signed offset. In unsigned
    // arithmetic 0 + (-1) is 4294967295: `curRow >= 0` would always be true, and
    // the kernel would be right only because that huge value also fails
    // `curRow < h`. With signed values the four clauses mean what they say.
    int col = int(gid.x);
    int row = int(gid.y);

    // launch.mm uses dispatchThreadgroups with the count rounded up, so the grid
    // can be larger than the picture, as in CUDA. Surplus threads do nothing.
    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow = -r; blurRow < r + 1; ++blurRow) {
            for (int blurCol = -r; blurCol < r + 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow*w + curCol];
                    ++pixels;
                }
            }
        }

        out[row*w + col] = (uchar)((float)pixVal/pixels);
    }
}
