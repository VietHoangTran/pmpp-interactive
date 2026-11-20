// Lab 13.1 starter (CUDA; also built for HIP and for the CPU emulators).
// Ordered merge of two sorted arrays with co-rank partitioning.
// Book: sections 13.2 to 13.6 (pp. 305-320), Figs. 13.2, 13.5 and 13.9, and the idea
// of Exercise 3 (p. 328).
//
// Contract (../merge.h): A has m elements and B has n, both sorted by .key; either may
// be empty. Write the first `total` outputs of their stable merge to C, where
// 0 <= total <= m + n, and nothing at or past C[total]. Compare .key only. On equal keys
// A's element goes first, and elements from the same array keep their order.
// Every launch uses MERGE_BLOCK (128) threads per block.
//
// Where this lab differs from the page: README, "Where this lab differs from the page".
#include "../merge.h"

// ---------------------------------------------------------------- Part (a)
// Fig. 13.5. Return i: of the first k outputs of the stable merge of A and B, i come
// from A (and k - i from B). Any k from 0 to m + n is a legal query.
__device__ int co_rank(int k, const KV* A, int m, const KV* B, int n) {
    // TODO 1: start i at its largest possible value and j at k - i; keep i + j == k.
    //         Also the smallest values i and j could still take (how many of the k
    //         outputs must come from A if B has only n elements?).
    //         Write min and max as ternaries: the device min/max overloads differ
    //         between CUDA and HIP (notes/today/ch13-facts.md A5, B1).
    // TODO 2: loop until neither of these holds, moving i and j by half the remaining
    //         range each time:
    //           i is too high: A[i-1] would come after B[j] in the merge
    //           j is too high: B[j-1] would come after A[i] in the merge
    //         Think about which of the two tests allows equal keys, given that ties go
    //         to A. Never read A[-1], A[m], B[-1] or B[n].
    return 0;
}

// Fig. 13.2. Merge A (m elements) and B (n elements) into C, stably.
__device__ void merge_sequential(const KV* A, int m, const KV* B, int n, KV* C) {
    // TODO 3: the two-phase sequential merge. Either m or n may be 0.
}

// Fig. 13.9. Thread tid produces outputs tid*perThread up to (tid+1)*perThread, but
// never one at or past `total`.
__global__ void mergeBasicKernel(const KV* A, int m, const KV* B, int n, KV* C, int total, int perThread) {
    // TODO 4: this thread's first output k_curr and the next thread's first output
    //         k_next, both limited to total. A thread with nothing to produce returns.
    // TODO 5: two co-ranks give this thread's slices of A and B; merge them into C.
}

void launchMergeBasic(const KV* A, int m, const KV* B, int n, KV* C, int total, int perThread) {
    // TODO 6: MERGE_BLOCK threads per block and enough blocks for `total` outputs at
    //         perThread each. Integer division truncates: a ceil() around it rounds
    //         nothing up. When total is 0, launch nothing: HIP rejects a grid of 0
    //         blocks (notes/today/ch02-facts.md B4), and so does the emulator.
}

// ---------------------------------------------------------------- Part (b)
// One thread per tile boundary b = 0 .. numTiles, where a tile is the tileSize outputs
// of one block: bounds[b] = co_rank of output b * tileSize, limited to total.
__global__ void partitionKernel(const KV* A, int m, const KV* B, int n, int total, int tileSize, int* bounds) {
    // TODO 7
}

// One tile per block. bounds[blockIdx.x] and bounds[blockIdx.x + 1] give this block's
// range of A; its range of B follows from the outputs it produces.
__global__ void mergeTiledKernel(const KV* A, const KV* B, KV* C, int total, int perThread, const int* bounds) {
    extern __shared__ KV tile[];   // MERGE_BLOCK * perThread elements, sized by the launch

    // TODO 8: this block's output range C_curr .. C_next (the last block may be short),
    //         its A range and its B range.
    // TODO 9: copy the block's A elements, then its B elements, into `tile`, with
    //         consecutive threads on consecutive elements.
    // TODO 10: wait until the tile is complete. Every thread of the block must reach
    //          this barrier: do not return before it.
    // TODO 11: as in Part (a), but inside the tile: this thread's perThread outputs
    //          (limited to the block's length), two co-ranks on the shared copies, and
    //          merge_sequential from the tile into the right place of C.
}

void launchMergeTiled(const KV* A, int m, const KV* B, int n, KV* C, int total, int perThread, int* bounds) {
    // TODO 12: nothing when total is 0. Otherwise two launches on the same stream: the
    //          partition (one thread per tile boundary), then the tiled merge with one
    //          block per tile and the tile's size in bytes as the third launch parameter.
}

// ---------------------------------------------------------------- Provided
// Lets the tests call your co_rank directly: out[k] = co_rank(k, A, m, B, n) for
// k = 0 .. m + n, one thread per k. Do not edit.
__global__ void coRankKernel(const KV* A, int m, const KV* B, int n, int* out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k <= m + n) out[k] = co_rank(k, A, m, B, n);
}

void launchCoRank(const KV* A, int m, const KV* B, int n, int* out) {
    const int queries = m + n + 1;
    coRankKernel<<<(queries + MERGE_BLOCK - 1) / MERGE_BLOCK, MERGE_BLOCK>>>(A, m, B, n, out);
}
