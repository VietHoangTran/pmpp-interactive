#define CHANNELS 3   // bytes per colour pixel: r, g, b

// One thread per pixel. Pin holds three bytes for each pixel, Pout one.
__global__ void colorToGrayscaleConversion(unsigned char* Pout,
                                           unsigned char* Pin,
                                           int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        // Offset in the flat gray picture: skip the rows above.
        int grayOffset = row * width + col;
        // The colour picture: the same walk, CHANNELS bytes a pixel.
        int rgbOffset = grayOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        // The cast covers the whole sum. It drops the fraction.
        Pout[grayOffset] =
            (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// n rows (y), m columns (x): x is sized from the width, y from the height.
void launchColorToGrayscale(unsigned char* Pin_d, unsigned char* Pout_d,
                            int m, int n) {
    dim3 dimGrid((m + 15) / 16, (n + 15) / 16, 1);
    dim3 dimBlock(16, 16, 1);
    colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, m, n);
}
