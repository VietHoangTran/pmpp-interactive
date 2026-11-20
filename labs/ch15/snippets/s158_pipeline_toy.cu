// Section 15.8, Fig. 15.14 (p. 367): the double-buffered, software-pipelined tile loop,
// reduced to a toy so that the section's stepper can run it thread by thread. Each of
// the block's THREADS threads copies one slot of a tile into shared memory (loadTile)
// and adds up every slot of the current buffer (mm). The loop shape is the book's; the
// toy computation, the names and the K = 0 guard are ours. The stepper's model is
// app/src/sim/gemmpipe.ts, and gemmpipe.test.ts checks that its loop is this file's
// loop line for line. Not built or run; ./verify.sh front-end-checks it.
// Launch one block of THREADS threads.
#define THREADS 4

// Thread t copies slot t of tile `tile`.
__device__ __forceinline__ void loadTile(float* buf, const float* tiles, unsigned int tile) {
    buf[threadIdx.x] = tiles[tile * THREADS + threadIdx.x];
}

// Every thread adds up the whole buffer: it reads the slots the other threads wrote.
__device__ __forceinline__ void mm(const float* buf, float& sum) {
    for (unsigned int j = 0; j < THREADS; ++j) sum += buf[j];
}

__global__ void pipelinedToy(const float* tiles, float* sums, unsigned int numTiles) {
    __shared__ float buf[2][THREADS];
    float sum = 0.0f;
    if (numTiles == 0) {            // numTiles - 1 would wrap to 4,294,967,295
        sums[threadIdx.x] = sum;
        return;
    }
    loadTile(buf[0], tiles, 0);
    __syncthreads();
    for (unsigned int tile = 0; tile < numTiles - 1; ++tile) {
        mm(buf[tile & 1], sum);
        loadTile(buf[(tile + 1) & 1], tiles, tile + 1);
        __syncthreads();
    }
    mm(buf[(numTiles - 1) & 1], sum);
    sums[threadIdx.x] = sum;
}
