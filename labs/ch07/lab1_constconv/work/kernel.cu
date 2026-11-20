// Lab 7.1 starter (CUDA; also built for HIP and for the CPU emulator).
// Three ways to hold a convolution filter. Book: sections 7.2 and 7.4,
// Fig. 7.7 (p. 166), Fig. 7.9 (p. 170) and the by-value alternative on p. 172.
//
// Contract for all three kernels: N and P are height x width floats, row-major.
// The filter is (2r+1) x (2r+1) floats, row-major, 0 <= r <= MAX_FILTER_RADIUS,
// applied without flipping (the book's formula, p. 162). Inputs outside the image
// are ghost cells with value 0. One thread per output element, in
// TILE_WIDTH x TILE_WIDTH blocks. The build defines TILE_WIDTH; do not hard-code it.
//
// The three kernels compute the same thing. Get the first one right, then the other
// two differ only in where the weights come from.
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

// The book: "the radius of most convolution filters is 7 or smaller" (p. 168).
#define MAX_FILTER_RADIUS 7
#define MAX_FILTER_DIM (2 * MAX_FILTER_RADIUS + 1)

// @part basic
// Fig. 7.7, with the filter as a pointer to global memory. The printed listing does
// not compile as it stands: F and P are plain float pointers. Write it so it does.
__global__ void convBasicKernel(const float* N, const float* F, float* P, int r, int width, int height) {
    // TODO 1: which output element (row, column) does this thread own?

    // TODO 2: the grid is rounded up to whole blocks, so some threads own no output.
    //         Make sure they neither read nor write.

    // TODO 3: add up the (2r+1) x (2r+1) products of filter weights and input
    //         elements, in a register. The patch's upper-left corner is (row - r,
    //         column - r). Skip any tap whose input element is outside the image: that
    //         is a ghost cell, worth 0. F is flat: weight (fRow, fCol) is one element.

    // TODO 4: write the result to P, once.
}

void launchConvBasic(const float* N, const float* F, float* P, int r, int width, int height) {
    // TODO 5: TILE_WIDTH x TILE_WIDTH blocks, and enough of them to cover width x height.
}

// @part const
// Fig. 7.9: the same kernel with the filter in constant memory.
//
// TODO 6: declare F_c at file scope, in constant memory: a flat float array with
//         room for the largest filter, MAX_FILTER_DIM * MAX_FILTER_DIM weights. For a
//         radius r, the filter uses its first (2r+1) * (2r+1) entries, laid out as in F.

__global__ void convConstKernel(const float* N, float* P, int r, int width, int height) {
    // TODO 7: as convBasicKernel, reading the weights from F_c. No filter argument:
    //         a kernel reads a constant-memory variable by its name.
}

void launchConvConst(const float* N, const float* F_h, float* P, int r, int width, int height) {
    // TODO 8: F_h is the filter in HOST memory. Copy it into F_c (count the size in
    //         bytes, not elements), then launch convConstKernel as in TODO 5.
}

// @part byvalue
// The book's alternative on p. 172: the weights travel with the launch, in a struct
// passed by value. Provided: the struct. A plain const parameter keeps this file
// buildable for HIP.
struct Filter {
    int r;
    float w[MAX_FILTER_DIM * MAX_FILTER_DIM];
};

__global__ void convByValueKernel(const float* N, float* P, const Filter f, int width, int height) {
    // TODO 9: as convBasicKernel, with the radius and the weights taken from f.
}

void launchConvByValue(const float* N, const float* F_h, float* P, int r, int width, int height) {
    // TODO 10: fill a Filter on the host from r and F_h, and pass it to the kernel as an
    //          ordinary argument. No copy call is needed.
}
