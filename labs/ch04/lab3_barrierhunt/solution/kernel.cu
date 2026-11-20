// Lab 4.3 reference solution: the three defects in the starter, fixed.
// Reverses each block's slice of a[0..n) in place: a 1D grid, block b owns the
// elements b*blockDim.x up to (b+1)*blockDim.x, and the last slice may be short.
// The kernel is ours, not a listing from the book. It applies the barrier rules
// of section 4.3 (p. 72) and section 4.4's warning (p. 74) that nothing but a
// barrier keeps the threads of a block in step.

__global__ void reverseBlocksKernel(float* a, int n) {
    int t = threadIdx.x;
    int base = blockIdx.x * blockDim.x;     // where this block's slice starts

    // FIX 2: a slice is one block long, and the block size is whatever the launch
    // chose. The starter mirrored around a literal 32, which is right for one
    // block size only.
    int size = blockDim.x;

    // FIX 3: the last slice ends where the array ends. The starter mirrored it
    // around the full block size, which sends the first valid elements of a short
    // slice past the end of the array.
    int valid = n - base;                   // at least 1 for every block launched
    if (valid > size) valid = size;

    // Phase 1: every thread reads its own element into a register.
    float mine = 0.0f;
    if (t < valid) mine = a[base + t];

    // FIX 1: no thread may write before every thread of the block has read.
    // Thread t writes the element that thread valid-1-t reads. The starter had no
    // barrier here.
    //
    // Threads past the end of the array stay and reach the barrier with nothing
    // to do. Returning early instead (`if (t >= valid) return;` before this line)
    // is covered by NVIDIA's wording, "all non-exited threads", and by Apple's
    // for Apple silicon. AMD's HIP pages do not say what an exited thread does to
    // a barrier (notes/today/ch04-facts.md, A6, B3, C3). Guarding the read and
    // the write is the form all three document.
    __syncthreads();

    // Phase 2: every thread writes its register to the mirrored position.
    if (t < valid) a[base + valid - 1 - t] = mine;
}

void launchReverseBlocks(float* a, int n, int threadsPerBlock) {
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    reverseBlocksKernel<<<blocks, threadsPerBlock>>>(a, n);
}
