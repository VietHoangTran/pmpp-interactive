// PROVIDED, do not edit. A copy of labs/ch09/lab3_tokencounts/solution/kernel.cu (lab 9.3: token counts and penalties),
// linked into lab X1.3 beside other copies. The only change: helper functions are static, so
// that no two linked files define the same name. app/src/chapters/x1/copies-g11.test.ts checks
// that nothing else differs.
// Lab 9.3 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Token counts for repetition penalties: the counting piece of milestone M5.
// Book: sections 9.2 to 9.5, pp. 203-217, Figs. 9.6 and 9.14. Counting token ids
// instead of pixels, and the penalty formula, are not in the book (see below).
//
// Contract (row-major, batch first, raw device pointers):
//   tokens     [batch][len]    int32 ids, each in [0, vocab)
//   counts     [batch][vocab]  uint32; the caller zeroes it before countTokens
//   newTokens  [batch]         one id per row, in [0, vocab)
//   logits     [batch][vocab]  float, updated in place
//   batch <= 65535 (it is gridDim.y); batch * vocab < 2^32.
//
// Where this differs from the book, and why:
//   - The bins are a whole vocabulary per request (50,257 to 201,088 ids in current
//     models): 196.3 to 785.5 KiB of uint32 per private copy. That is more than any
//     documented per-block limit for Qwen3 and gpt-oss; GPT-2's 196.3 KiB fits only
//     behind CUDA's opt-in on a few compute capabilities (notes/today/ch09-facts.md D7,
//     A8). So there is no shared-memory copy (Fig. 9.10). A per-block copy in global memory (Fig. 9.9) would
//     cost a whole vocabulary to zero and to commit per block; provided/count_pool.cu is
//     that design, for the benchmark.
//   - What does still apply: the batch dimension. Rows of different requests never share
//     a bin, so each request's row is already a private histogram that nobody merges.
//     Contention is confined to the blocks of one row. countTokens is Fig. 9.6's shape,
//     atomics straight into that row in global memory.
//   - Fig. 9.6 gives each thread one pixel. Here the blocks of a row walk it with the
//     interleaved index of Fig. 9.14 (p. 217) stretched over all blocks of the row, so
//     any len works with any number of blocks.
//   - appendTokens is ours: one decode step adds one token per row. One thread owns a
//     row, so the owner-computes rule (p. 201) holds again and no atomic is needed.
//   - applyPenalties is llama.cpp's frequency and presence penalty
//     (logit -= count * freq + (count > 0) * presence; ch09-facts D4).
#include <cuda/atomic>

#define BLOCK_SIZE 256           // threads per block, all three kernels
#define MAX_BLOCKS_PER_ROW 1024  // past this, each thread takes several tokens

__global__ void countTokens(const int* tokens, unsigned int* counts, int len, int vocab) {
    const int row = blockIdx.y;                              // one request per grid row
    const int* rowTokens = tokens + (size_t)row * len;
    unsigned int* rowCounts = counts + (size_t)row * vocab;  // this request's histogram

    // Interleaved: in each step the blocks of this row cover the next
    // gridDim.x * blockDim.x tokens together, so a warp reads consecutive ids.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) {
        const int t = rowTokens[i];
        // Several blocks update this row, so the scope is the device (p. 208).
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> count_ref(rowCounts[t]);
        count_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
}

void launchCountTokens(const int* tokens, unsigned int* counts, int batch, int len, int vocab) {
    int blocksPerRow = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;  // one token per thread...
    if (blocksPerRow > MAX_BLOCKS_PER_ROW) blocksPerRow = MAX_BLOCKS_PER_ROW;  // ...up to here
    dim3 grid(blocksPerRow, batch);
    countTokens<<<grid, BLOCK_SIZE>>>(tokens, counts, len, vocab);
}

__global__ void appendTokens(const int* newTokens, unsigned int* counts, int batch, int vocab) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch) {
        // This thread is the only one touching row `row`: a plain increment is enough.
        counts[(size_t)row * vocab + newTokens[row]] += 1;
    }
}

void launchAppendTokens(const int* newTokens, unsigned int* counts, int batch, int vocab) {
    appendTokens<<<(batch + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(newTokens, counts, batch, vocab);
}

__global__ void applyPenalties(float* logits, const unsigned int* counts, size_t n,
                               float frequencyPenalty, float presencePenalty) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;  // one element each
    if (i < n) {
        const unsigned int c = counts[i];
        logits[i] -= (float)c * frequencyPenalty + (float)(c > 0) * presencePenalty;
    }
}

void launchApplyPenalties(float* logits, const unsigned int* counts, int batch, int vocab,
                          float frequencyPenalty, float presencePenalty) {
    const size_t n = (size_t)batch * vocab;  // the operation is elementwise: flatten
    const unsigned int blocks = (unsigned int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    applyPenalties<<<blocks, BLOCK_SIZE>>>(logits, counts, n, frequencyPenalty, presencePenalty);
}
