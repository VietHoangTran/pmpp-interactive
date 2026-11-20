// Lab 3.2 starter (CUDA; also built for HIP and for the CPU emulator).
// Image blur: each output pixel is the average of the input pixels in a square
// patch around it. Book: section 3.3, Fig. 3.8 (p. 57).
//
// One difference from the page: the book's patch radius is a constant BLUR_SIZE
// defined outside the kernel. Here it is the argument `r`, so the benchmark can
// sweep it. The patch is (2r+1) pixels on a side.

// in, out: h x w pixels, row-major, 1 unsigned char per pixel.
__global__ void blurKernel(const unsigned char* in, unsigned char* out, int w, int h, int r) {
    // TODO 1: the column and row of the output pixel this thread owns, and the
    //         test that keeps surplus threads from touching anything.
    //         Choose the type of these two variables with TODO 2 in mind.

    // TODO 2: visit every position of the patch around (row, col). Positions
    //         outside the picture do not count: skip them. A patch can hang
    //         over any of the four edges.

    // TODO 3: keep a running sum of the values you did visit, and whatever
    //         else you need to turn that sum into their average.

    // TODO 4: store the average as one byte.
}

void launchBlur(const unsigned char* in, unsigned char* out, int w, int h, int r) {
    // TODO 5: 16 by 16 blocks, enough of them to cover w by h, then the launch.
}
