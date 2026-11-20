// Lab 3.1 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Colour to grayscale, one thread per pixel. Book: section 3.2, Fig. 3.4 (p. 53);
// the launch is on p. 50.
//
// Four places this file differs from the page, all on purpose:
//   cast       Line 19 of Fig. 3.4 reads `(unsigned char) 0.21f*r+0.71f*g+0.07f*b`.
//              A cast binds tighter than `*`, so only 0.21f is cast, to 0, and the
//              red term disappears. Here the whole sum is in parentheses.
//   constants  The listing prints 0.21, 0.71, 0.07. The text on p. 52 gives
//              L = 0.299 r + 0.587 g + 0.114 b, which is what is used here.
//   arguments  The listing takes (Pout, Pin, ...); the host call on p. 50 passes
//              (Pin_d, Pout_d, ...). Here the input comes first in both.
//   name       The listing prints colortoGrayscaleConvertion, the text
//              colorToGrayscaleConversion. Here: colorToGrayscaleKernel.
// Also: Pin is const, and the grid is sized with integer round-up where p. 50
// prints ceil(m/16.0).

#define CHANNELS 3   // bytes per input pixel: r, g, b (footnote 2, p. 53)

// Pin:  height x width pixels, row-major, 3 consecutive unsigned chars per pixel.
// Pout: height x width pixels, row-major, 1 unsigned char per pixel.
__global__ void colorToGrayscaleKernel(const unsigned char* Pin, unsigned char* Pout,
                                       int width, int height) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;   // x fields give the column
    int row = blockIdx.y*blockDim.y + threadIdx.y;   // y fields give the row

    // The grid is whole blocks, so it is usually larger than the picture.
    // Threads that landed outside it do nothing.
    if (col < width && row < height) {
        int grayOffset = row*width + col;            // skip `row` full rows, then `col` bytes
        int rgbOffset = grayOffset*CHANNELS;         // same pixel, three bytes each
        unsigned char r = Pin[rgbOffset    ];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        // Bytes in, float arithmetic, one byte out. The cast truncates.
        Pout[grayOffset] = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b);
    }
}

void launchGrayscale(const unsigned char* Pin, unsigned char* Pout, int width, int height,
                     int blockW, int blockH) {
    dim3 dimBlock(blockW, blockH, 1);
    // Whole blocks only, so round up. x is sized from the width, y from the height.
    dim3 dimGrid((width + blockW - 1) / blockW, (height + blockH - 1) / blockH, 1);
    colorToGrayscaleKernel<<<dimGrid, dimBlock>>>(Pin, Pout, width, height);
}
