// Lab 3.1 starter (CUDA; also built for HIP and for the CPU emulator).
// Colour to grayscale, one thread per pixel. Book: section 3.2, Fig. 3.4 (p. 53);
// the launch is on p. 50.
//
// Do not copy Fig. 3.4 as printed. Its name, its argument order, its constants
// and its cast all differ from what this lab asks for; README.md lists the four.

#define CHANNELS 3   // bytes per input pixel: r, g, b (footnote 2, p. 53)

// Pin:  height x width pixels, row-major, 3 consecutive unsigned chars per pixel.
// Pout: height x width pixels, row-major, 1 unsigned char per pixel.
__global__ void colorToGrayscaleKernel(const unsigned char* Pin, unsigned char* Pout,
                                       int width, int height) {
    // TODO 1: which column and which row of the picture does this thread own?
    //         Blocks are blockDim.x threads wide and blockDim.y threads high.

    // TODO 2: the grid is made of whole blocks, so some threads have no pixel.
    //         Make sure they touch nothing.

    // TODO 3: the flat offset of this pixel in Pout (one byte per pixel), and
    //         from it the offset of its first byte in Pin (CHANNELS bytes per pixel).

    // TODO 4: L = 0.299 r + 0.587 g + 0.114 b, stored as one byte.
    //         Before you write the cast, work out what the cast on line 19 of
    //         Fig. 3.4 applies to.
}

void launchGrayscale(const unsigned char* Pin, unsigned char* Pout, int width, int height,
                     int blockW, int blockH) {
    // TODO 5: blocks of blockW x blockH threads, and enough of them to cover a
    //         picture `width` pixels wide and `height` pixels high. Round up with
    //         integer arithmetic. Decide which of width and height sizes which
    //         field of the grid, then launch the kernel.
}
