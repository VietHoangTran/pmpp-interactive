#define BLUR_SIZE 1   // patch radius: the patch is 2*BLUR_SIZE+1 pixels on a side
__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;

        // Average the (2*BLUR_SIZE+1) x (2*BLUR_SIZE+1) patch around (row, col)
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                // Keep only the positions that are inside the picture
                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow*w + curCol];
                    ++pixels;   // how many values went into the sum
                }
            }
        }
        // Store the average of the values that were kept
        out[row*w + col] = (unsigned char)((float)pixVal/pixels);
    }
}
