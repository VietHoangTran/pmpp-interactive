// Section 15.8's pipelined toy (s158_pipeline_toy.cu) with one planted defect, for the
// section's stepper: one buffer instead of two and still one barrier per iteration. The false
// dependence the second buffer removed is back, unguarded.
// Deliberately wrong. gemmpipe.test.ts checks that the stepper's listing for this variant
// is this file's loop, line for line. Not built or run; the CUDA front-end check parses it.
#define THREADS 4

__device__ __forceinline__ void loadTile(float* buf, const float* tiles, unsigned int tile) {
    buf[threadIdx.x] = tiles[tile * THREADS + threadIdx.x];
}

__device__ __forceinline__ void mm(const float* buf, float& sum) {
    for (unsigned int j = 0; j < THREADS; ++j) sum += buf[j];
}

__global__ void singleBufferToy(const float* tiles, float* sums, unsigned int numTiles) {
    __shared__ float buf[1][THREADS];
    float sum = 0.0f;
    if (numTiles == 0) {
        sums[threadIdx.x] = sum;
        return;
    }
    loadTile(buf[0], tiles, 0);
    __syncthreads();
    for (unsigned int tile = 0; tile < numTiles - 1; ++tile) {
        mm(buf[0], sum);
        loadTile(buf[0], tiles, tile + 1);
        __syncthreads();
    }
    mm(buf[0], sum);
    sums[threadIdx.x] = sum;
}
