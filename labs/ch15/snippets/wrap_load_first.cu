// Section 15.8's pipelined toy (s158_pipeline_toy.cu) with the two calls in the loop
// swapped: the loads of tile + 1 come before mm on tile. For a question on the
// chapter's summary page; not built or run.
#define THREADS 4

__device__ __forceinline__ void loadTile(float* buf, const float* tiles, unsigned int tile) {
    buf[threadIdx.x] = tiles[tile * THREADS + threadIdx.x];
}

__device__ __forceinline__ void mm(const float* buf, float& sum) {
    for (unsigned int j = 0; j < THREADS; ++j) sum += buf[j];
}

__global__ void loadFirstToy(const float* tiles, float* sums, unsigned int numTiles) {
    __shared__ float buf[2][THREADS];
    float sum = 0.0f;
    if (numTiles == 0) {
        sums[threadIdx.x] = sum;
        return;
    }
    loadTile(buf[0], tiles, 0);
    __syncthreads();
    for (unsigned int tile = 0; tile < numTiles - 1; ++tile) {
        loadTile(buf[(tile + 1) & 1], tiles, tile + 1);
        mm(buf[tile & 1], sum);
        __syncthreads();
    }
    mm(buf[(numTiles - 1) & 1], sum);
    sums[threadIdx.x] = sum;
}
