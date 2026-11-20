// Section 3.3, "Signed arithmetic": the kernel of Fig. 3.8 (blur_kernel.cu) with every
// index kept unsigned. Do not copy it. curRow >= 0 and curCol >= 0 are always true here,
// and the four other comparisons mix unsigned and signed values. labs/ch03/snippets/check.py
// runs it on the course's emulator against blur_kernel.cu and checks that a build with
// -Wall -Wextra warns about those comparisons.
#define BLUR_SIZE 1
__global__
void blurKernelUnsigned(unsigned char *in, unsigned char *out, int w, int h) {
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                unsigned int curRow = row + blurRow;   // 0 + (-1) is 4294967295
                unsigned int curCol = col + blurCol;
                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow*w + curCol];
                    ++pixels;
                }
            }
        }
        out[row*w + col] = (unsigned char)((float)pixVal/pixels);
    }
}
