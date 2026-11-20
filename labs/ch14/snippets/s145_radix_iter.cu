// Section 14.5: one pass of the 1-bit radix sort of Fig. 14.7 (p. 339), split at the scan.
//
// On the page, every thread calls gridExclusiveScan(bits, N) in the middle of the kernel
// and waits for the whole grid's result. That is pseudocode (notes/ch14.md, "Unclear in the
// PDF" item 6): an ordinary launch promises neither that all blocks are resident at once nor
// any order between them, so a block cannot wait there for the others. Here the pass is
// three launches, in order on one stream: extract the bits, scan them, scatter the keys.
// Two more differences from the page, both stated in the lesson:
//   - the scan covers N + 1 entries, the N bits and a zero after them, so that bits[N] holds
//     the total number of ones; the page reads bits[N] but never says where it comes from
//     (item 4). The bits array therefore has N + 1 entries;
//   - the scatter recomputes each key's bit from the key, because the scan has overwritten
//     bits[i] with the number of ones before it.
// The page's lines become the two kernels below, in the same order and with the same names
// (numOnesBefore, numOnesTotal, dst). The same source compiles for HIP through
// pmpp/vendors/hip.h. Shown in section 14.5; its lines are pinned, and it is run on the CPU
// emulator against app/src/sim/radixsort.ts, by app/src/chapters/ch14/S145.snippets.test.ts
// (driver: check/s145_check.cu).

#define SCAN_THREADS 256

// region: scan
// Not the book's code: an exclusive scan of bits[0..N-1] followed by one zero, in one block,
// written in place. Each thread sums a contiguous chunk, the chunk sums are scanned in
// shared memory (Kogge-Stone, section 11.2), and each thread then rewrites its chunk as a
// running sum: chapter 11's coarsened scan (section 11.6) in the reduce-scan-scan order of
// section 11.9, with the chunks left in global memory. bits[N] receives the total.
__global__ void exclusive_scan_kernel(unsigned int* bits, unsigned int N) {
    __shared__ unsigned int sums[SCAN_THREADS];
    unsigned int t = threadIdx.x;
    unsigned int per = (N + blockDim.x - 1) / blockDim.x;
    unsigned int begin = (t * per < N) ? t * per : N;
    unsigned int end = (begin + per < N) ? begin + per : N;
    unsigned int mine = 0;
    for(unsigned int k = begin; k < end; ++k) mine += bits[k];
    sums[t] = mine;
    __syncthreads();
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        unsigned int left = (t >= stride) ? sums[t - stride] : 0;
        __syncthreads();
        sums[t] += left;
        __syncthreads();
    }
    unsigned int run = sums[t] - mine;          // ones in the chunks before this one
    for(unsigned int k = begin; k < end; ++k) {
        unsigned int b = bits[k];
        bits[k] = run;
        run += b;
    }
    if(t == blockDim.x - 1) bits[N] = run;      // after the last chunk: the total
}
// endregion

// region: pass
__global__ void radix_bits_kernel(unsigned int* input, unsigned int* bits,
        unsigned int N, unsigned int iter) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        bits[i] = bit;
    }
}

__global__ void radix_scatter_kernel(unsigned int* input,
        unsigned int* output, unsigned int* bits, unsigned int N,
        unsigned int iter) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        unsigned int numOnesBefore = bits[i];
        unsigned int numOnesTotal = bits[N];
        unsigned int dst = (bit == 0)
            ? (i - numOnesBefore)
            : (N - numOnesTotal + numOnesBefore);
        output[dst] = key;
    }
}

// bits has N + 1 entries.
void radix_sort_iter(unsigned int* input, unsigned int* output,
        unsigned int* bits, unsigned int N, unsigned int iter,
        unsigned int blockSize) {
    unsigned int numBlocks = (N + blockSize - 1) / blockSize;
    radix_bits_kernel<<<numBlocks, blockSize>>>(input, bits, N, iter);
    exclusive_scan_kernel<<<1, SCAN_THREADS>>>(bits, N);
    radix_scatter_kernel<<<numBlocks, blockSize>>>(input, output, bits,
                                                   N, iter);
}
// endregion
