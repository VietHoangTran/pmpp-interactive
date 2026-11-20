// Lab 5.5 starter (CUDA; also built for HIP and for the CPU emulator).
// Tiled matrix multiplication whose tile width is chosen at RUN time, using
// dynamically sized shared memory. Book: section 5.6, pp. 116-118.
//
// Square Width x Width matrices of ANY width (keep your boundary checks from
// lab 5.3). The tile width arrives as an argument instead of a #define.

__global__ void tiledMatmulDynamicKernel(const float* M, const float* N, float* P,
                                         int Width, int tileWidth) {
    // TODO 1: declare ONE unsized shared array (extern __shared__ ...).
    // TODO 2: point Mds at its first half and Nds at its second half.
    //         Careful: is the offset a number of bytes or a number of elements?
    // TODO 3: the tiled algorithm, indexing the tiles as flat arrays.
}

void launchTiledMatmulDynamic(float* M, float* N, float* P, int Width, int tileWidth) {
    // TODO 4: compute the shared-memory size in BYTES for two tileWidth x tileWidth
    //         tiles of float, and pass it as the third launch parameter.
}
