// Lab 3.2 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Image blur: each output pixel is the average of the input pixels in a square
// patch around it. Book: section 3.3, Fig. 3.8 (p. 57).
//
// Where this file differs from the page:
//   radius   The book uses a constant BLUR_SIZE defined outside the kernel. Here
//            the radius `r` is a kernel argument, so the benchmark can sweep it.
//            The patch is (2r+1) pixels on a side. (The comment on line 09 of
//            Fig. 3.8 calls it a "BLUR_SIZE x BLUR_SIZE box"; the text on p. 58
//            has it right.)
//   const    `in` is const here.
//   launch   The book prints none for this kernel. The one below is the
//            grayscale launch of p. 50 with integer round-up.

// in, out: h x w pixels, row-major, 1 unsigned char per pixel.
__global__ void blurKernel(const unsigned char* in, unsigned char* out, int w, int h, int r) {
    // int, not unsigned: row + blurRow has to be able to go below zero.
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    // Outer test: threads the rounded-up grid placed outside the picture do nothing.
    if (col < w && row < h) {
        int pixVal = 0;   // running sum of the patch
        int pixels = 0;   // how many values went into it

        for (int blurRow = -r; blurRow < r + 1; ++blurRow) {
            for (int blurCol = -r; blurCol < r + 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                // Inner test: skip the part of the patch that hangs over an edge.
                // All four clauses are needed. A column past the right edge is
                // still a valid address (the start of the next row), so nothing
                // but a wrong value would tell you that clause was missing.
                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow*w + curCol];
                    ++pixels;
                }
            }
        }

        // Divide by the count, not by the patch area: a corner pixel averages
        // fewer values than an interior one.
        out[row*w + col] = (unsigned char)((float)pixVal/pixels);
    }
}

void launchBlur(const unsigned char* in, unsigned char* out, int w, int h, int r) {
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((w + 15) / 16, (h + 15) / 16, 1);   // x from the width, y from the height
    blurKernel<<<dimGrid, dimBlock>>>(in, out, w, h, r);
}
