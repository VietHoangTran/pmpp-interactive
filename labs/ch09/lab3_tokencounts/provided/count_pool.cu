// Lab 9.3, provided (do not edit): the comparator for the benchmark.
// Fig. 9.9's design (section 9.4, p. 212) applied to token counts: every block gets a
// private copy of a whole vocabulary in a global-memory pool, counts its tokens there
// with block-scope atomics, waits for the block, then commits every non-zero bin of its
// copy to the request's row with a device-scope atomic. One thread per token, as in
// Fig. 9.9. The tests run it too, so a broken copy of this file shows.
//
// Where this differs from the page:
//   - NUM_BINS is the run-time vocab. The pool holds one copy per block of every row:
//     batch * countPoolBlocksPerRow(len) * vocab counters, allocated by the caller.
//   - Fig. 9.9 does not say who zeroes bins_pool (notes/ch09.md, "Unclear in the PDF").
//     Here the launcher does, with cudaMemset, and that time is part of what the
//     benchmark measures: it is the "initialise each private copy" half of the overhead
//     that section 9.5 describes. The commit loop is the other half.
//   - Each block finds its row in blockIdx.y, as countTokens does.
#include <cuda/atomic>

#define POOL_BLOCK_SIZE 256

int countPoolBlocksPerRow(int len) { return (len + POOL_BLOCK_SIZE - 1) / POOL_BLOCK_SIZE; }

__global__ void countTokensPool(const int* tokens, unsigned int* counts, unsigned int* bins_pool,
                                int len, int vocab) {
    const int row = blockIdx.y;
    // This block's private copy of the vocabulary (Fig. 9.9, line 04).
    unsigned int* bins_priv = &bins_pool[((size_t)row * gridDim.x + blockIdx.x) * vocab];

    // Update phase: contention only among the threads of this block.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        const int t = tokens[(size_t)row * len + i];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_priv_ref(bins_priv[t]);
        bins_priv_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();

    // Commit phase: every bin of the copy is read, whatever this block counted.
    unsigned int* rowCounts = counts + (size_t)row * vocab;
    for (int bin = threadIdx.x; bin < vocab; bin += blockDim.x) {
        const unsigned int binValue = bins_priv[bin];
        if (binValue > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(rowCounts[bin]);
            bins_ref.fetch_add(binValue, cuda::memory_order_relaxed);
        }
    }
}

void launchCountPool(const int* tokens, unsigned int* counts, unsigned int* pool,
                     int batch, int len, int vocab) {
    const int blocksPerRow = countPoolBlocksPerRow(len);
    cudaMemset(pool, 0, (size_t)batch * blocksPerRow * vocab * sizeof(unsigned int));
    dim3 grid(blocksPerRow, batch);
    countTokensPool<<<grid, POOL_BLOCK_SIZE>>>(tokens, counts, pool, len, vocab);
}
