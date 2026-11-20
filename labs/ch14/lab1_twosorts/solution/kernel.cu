// Lab 14.1 reference solution (CUDA; also built for HIP and for both CPU emulators).
// Book: section 14.2 (pp. 331-333, Fig. 14.2) and section 14.5 (pp. 336-339, Figs. 14.6, 14.7).
//
// Part A: parallel odd-even sort, one launch per step, plus the host loop the book describes
// but does not show. Part B: a one-bit LSD radix sort of (key, value) pairs, 32 passes.
//
// Where this file differs from the pages:
//   oddEvenStep (Fig. 14.2)
//     - `i + 1 >= n` returns instead of the page's `if(i < N - 1)`: N - 1 is unsigned, so
//       with N == 0 it wraps to the largest value and every thread passes the check.
//     - The flag is set with atomicExch, not the page's plain `*hasChanged = 1;`. Several
//       threads may set it in one step; that is a data race, and the CUDA memory model makes a
//       data race undefined behaviour even when every thread writes the same 1 (p. 333;
//       notes/today/ch14-facts.md A9). cuda::atomic_ref<unsigned int,
//       cuda::thread_scope_device>(*hasChanged).store(1u, cuda::memory_order_relaxed) is the
//       other documented choice.
//   radixExtractBits + launchExclusiveScan + radixScatter (Fig. 14.7)
//     - The page is one kernel that calls gridExclusiveScan between extracting the bits and
//       scattering the keys. The book never defines that function, and a scan across blocks
//       inside one kernel needs blocks to wait for one another (a cooperative launch on CUDA
//       and HIP; nothing documented on Metal: ch14-facts C4). So the kernel is cut at the scan
//       into three launches per bit, with the provided scan in the middle.
//     - The scan covers n + 1 entries, the last one 0, so that bits[n] ends up holding the
//       total number of ones. The page reads bits[N] but never says the array has N + 1
//       entries (notes/ch14.md, "Unclear in the PDF", item 4).
//     - Values travel with their keys; the page moves bare keys.

// Provided in ../../shared-g11/scan.cu.
int scanScratchCount(int count);
void launchExclusiveScan(unsigned int* data, int count, unsigned int* scratch);

// ---------------------------------------------------------------- part A: odd-even sort

// One step: thread t owns the pair (i, i + 1) with i = 2t on an even step and 2t + 1 on an odd
// step. Pairs in one step never overlap, so the swaps cannot race; only the flag is shared.
__global__ void oddEvenStep(unsigned int* data, unsigned int* hasChanged, unsigned int n,
                            unsigned int isOddStep) {
    if (n < 2) {
        return;
    }
    unsigned int i = 2*(blockIdx.x*blockDim.x + threadIdx.x) + (isOddStep ? 1 : 0);
    if (i + 1 >= n) {
        return;                       // this thread's pair runs past the end
    }
    if (data[i] > data[i + 1]) {
        unsigned int tmp = data[i];
        data[i] = data[i + 1];
        data[i + 1] = tmp;
        atomicExch(hasChanged, 1u);   // many threads may set it: atomically, never a plain store
    }
}

// Not in the book. One launch per step, floor(n / 2) threads (every pair of either parity).
// The flag is cleared, an even step and the odd step after it run, and only then is the flag
// read: the copy back waits for both launches. A quiet pair (no swap on the even step and none
// on the odd step that follows) means every adjacent pair is in order, so the list is sorted.
// One quiet step alone proves nothing: [1, 3, 2, 4] makes no swap on its first, even, step
// (notes/ch14.md, "Unclear in the PDF", item 7). Returns the number of steps launched.
int launchOddEvenSort(unsigned int* data, unsigned int* flag, int n) {
    if (n < 2) {
        return 0;                     // nothing to compare; also avoids a launch of 0 blocks
    }
    const unsigned int block = 256;
    const unsigned int pairs = (unsigned int)n / 2;
    const unsigned int grid = (pairs + block - 1) / block;
    const int maxSteps = 2 * n + 4;   // a correct sort never gets here; a broken one must not hang
    int steps = 0;
    while (steps < maxSteps) {
        unsigned int changed = 0;
        cudaMemcpy(flag, &changed, sizeof(changed), cudaMemcpyHostToDevice);
        oddEvenStep<<<grid, block>>>(data, flag, (unsigned int)n, 0);
        oddEvenStep<<<grid, block>>>(data, flag, (unsigned int)n, 1);
        steps += 2;
        cudaMemcpy(&changed, flag, sizeof(changed), cudaMemcpyDeviceToHost);
        if (changed == 0) {
            break;
        }
    }
    return steps;
}

// ---------------------------------------------------------------- part B: one-bit radix sort

// Fig. 14.7, lines 03-09: thread i extracts bit `bit` of key i. The thread just past the last
// key writes the extra entry bits[n] = 0, which the scan of n + 1 entries turns into the total.
__global__ void radixExtractBits(const unsigned int* keys, unsigned int* bits, unsigned int n,
                                 unsigned int bit) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int key = keys[i];
        bits[i] = (key >> bit) & 1;
    } else if (i == n) {
        bits[n] = 0;
    }
}

// Fig. 14.7, lines 11-17, after the scan: onesBefore[i] is the number of ones before key i and
// onesBefore[n] the number of ones in total. A zero goes after the zeros before it; a one goes
// after every zero and after the ones before it (Fig. 14.6). The page introduces the second
// formula as "for keys mapping to the zero bucket"; it is the one bucket's (item 3 of "Unclear
// in the PDF"). The value moves with its key.
__global__ void radixScatter(const unsigned int* keysIn, const unsigned int* valsIn,
                             unsigned int* keysOut, unsigned int* valsOut,
                             const unsigned int* onesBefore, unsigned int n, unsigned int bit) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int key = keysIn[i];
        unsigned int b = (key >> bit) & 1;
        unsigned int numOnesBefore = onesBefore[i];
        unsigned int numOnesTotal = onesBefore[n];
        unsigned int dst = (b == 0) ? (i - numOnesBefore) : (n - numOnesTotal + numOnesBefore);
        keysOut[dst] = key;
        valsOut[dst] = valsIn[i];
    }
}

// Not in the book: the host loop of p. 336, one pass per bit, least significant first.
// Each pass reads one pair of buffers and writes the other; 32 passes swap them an even number
// of times, so the result ends in keys/vals.
void launchRadixSortOneBit(unsigned int* keys, unsigned int* vals, unsigned int* tmpKeys,
                           unsigned int* tmpVals, unsigned int* bits, unsigned int* scanScratch, int n) {
    if (n < 1) {
        return;
    }
    const unsigned int count = (unsigned int)n;
    const unsigned int block = 256;
    unsigned int* keysIn = keys;
    unsigned int* valsIn = vals;
    unsigned int* keysOut = tmpKeys;
    unsigned int* valsOut = tmpVals;
    for (unsigned int bit = 0; bit < 32; ++bit) {
        radixExtractBits<<<(count + 1 + block - 1) / block, block>>>(keysIn, bits, count, bit);
        launchExclusiveScan(bits, n + 1, scanScratch);   // n + 1 entries: bits[n] becomes the total
        radixScatter<<<(count + block - 1) / block, block>>>(keysIn, valsIn, keysOut, valsOut, bits,
                                                           count, bit);
        unsigned int* swap = keysIn; keysIn = keysOut; keysOut = swap;
        swap = valsIn; valsIn = valsOut; valsOut = swap;
    }
}
