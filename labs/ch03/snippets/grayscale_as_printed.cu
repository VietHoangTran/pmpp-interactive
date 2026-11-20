// The code lines of Fig. 3.4 (book p. 53) as printed, with their line numbers kept,
// for the erratum hunt in lesson 3.2. Comments are reworded. Line 19 is wrong in two
// ways and the name is misspelt; see notes/ch03.md, "Unclear in the PDF", items 1 to 4.
// This file exists to be read, not launched: it only has to parse.
#define CHANNELS 3
// ---- as printed ----
// Input bytes are unsigned chars, 0 to 255
// A pixel is 3 chars in a row, one per channel (RGB)
__global__
void colortoGrayscaleConvertion(unsigned char * Pout,
                unsigned char * Pin, int width, int height) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        // 1D offset into the gray picture
        int grayOffset = row*width + col;
        // The colour picture is CHANNELS times
        // as wide, counted in bytes
        int rgbOffset = grayOffset*CHANNELS;
        unsigned char r = Pin[rgbOffset    ]; // red
        unsigned char g = Pin[rgbOffset + 1]; // green
        unsigned char b = Pin[rgbOffset + 2]; // blue
        // Weight the channels, add them up
        // and store one byte
        Pout[grayOffset] = (unsigned char) 0.21f*r+0.71f*g+0.07f*b;
    }
}
