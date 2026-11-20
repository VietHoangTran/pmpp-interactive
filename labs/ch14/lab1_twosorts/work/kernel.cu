// Lab 14.1 starter (CUDA; also built for HIP and for both CPU emulators).
// Book: section 14.2 (pp. 331-333, Fig. 14.2) and section 14.5 (pp. 336-339, Figs. 14.6, 14.7).
//
// Part A: parallel odd-even sort, one launch per step, and the host loop that alternates the
// steps (the book describes it but does not show it).
// Part B: a one-bit LSD radix sort of (key, value) pairs in 32 passes. Fig. 14.7 is one kernel
// that calls gridExclusiveScan in the middle; the book never defines that function, so here a
// pass is three launches: radixExtractBits, the provided launchExclusiveScan, radixScatter.
//
// Read the README's list of traps before you start: five places where the pages, read
// literally, give a wrong or fragile program.

// Provided in ../../shared-g11/scan.cu (do not edit it):
//   scanScratchCount(count): unsigned ints of scratch the scan needs for `count` elements.
//   launchExclusiveScan(data, count, scratch): in place, data[0] = 0 and data[i] = the sum of
//   the old data[0..i-1]. The old value of the last element reaches no output.
int scanScratchCount(int count);
void launchExclusiveScan(unsigned int* data, int count, unsigned int* scratch);

// ---------------------------------------------------------------- part A: odd-even sort

// One step of odd-even sort (Fig. 14.2), in place.
__global__ void oddEvenStep(unsigned int* data, unsigned int* hasChanged, unsigned int n,
                            unsigned int isOddStep) {
    // TODO A1: which pair (i, i + 1) does this thread own on an even step, and on an odd step?
    // TODO A2: return early when there is no such pair (mind the unsigned arithmetic).
    // TODO A3: put the pair in order; if you swapped, record it in *hasChanged. Many threads may
    //          record a swap in the same step: use an atomic operation, never a plain store.
}

// The host loop. `flag` is one unsigned int of device memory. Alternate even and odd steps,
// starting with an even step; read the flag after each even-odd pair; stop only when an even
// step and the odd step after it both made no swap. Return the number of steps launched.
int launchOddEvenSort(unsigned int* data, unsigned int* flag, int n) {
    // TODO A4: how many threads does one step need? (It must cover every pair of either parity.)
    // TODO A5: the loop. Before each pair of steps clear the flag on the device; after it copy
    //          the flag back (cudaMemcpy waits for the kernels before it to finish).
    // TODO A6: lists of 0 or 1 element need no step at all. Also stop after 2n + 4 steps
    //          whatever the flag says, so a broken kernel fails the test instead of hanging it.
    return 0;
}

// ---------------------------------------------------------------- part B: one-bit radix sort

// bits[i] = bit `bit` of keys[i] for every key, and one extra entry bits[n] = 0.
__global__ void radixExtractBits(const unsigned int* keys, unsigned int* bits, unsigned int n,
                                 unsigned int bit) {
    // TODO B1
}

// After the scan, onesBefore[i] is the number of ones before key i. Move key i and its value
// to the key's destination (Fig. 14.6). Where is the total number of ones?
__global__ void radixScatter(const unsigned int* keysIn, const unsigned int* valsIn,
                             unsigned int* keysOut, unsigned int* valsOut,
                             const unsigned int* onesBefore, unsigned int n, unsigned int bit) {
    // TODO B2
}

// 32 passes, least significant bit first. bits has room for n + 1 entries; scanScratch holds
// scanScratchCount(n + 1). tmpKeys/tmpVals are yours to use. When this returns, the sorted
// pairs must be in keys/vals.
void launchRadixSortOneBit(unsigned int* keys, unsigned int* vals, unsigned int* tmpKeys,
                           unsigned int* tmpVals, unsigned int* bits, unsigned int* scanScratch, int n) {
    // TODO B3: for each bit: extract, scan, scatter. Which buffers does each pass read and
    //          write, and where is the result after 32 passes?
}
