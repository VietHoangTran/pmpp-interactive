// Lab 4.4 starter (CUDA; also built for HIP and for the CPU emulator).
// A kernel that spends its time waiting on memory, launched with a block size
// chosen at run time, and the book's occupancy arithmetic to set beside the
// measurement. Book: section 4.6 (pp. 82-85) for why waiting is tolerable,
// section 4.7 (pp. 85-87) for the arithmetic. The kernel is ours, not a listing
// from the book.
#include "sm_limits.h"

// An embedding bag: each output is the sum of k table entries chosen by idx.
//   out[i] = sum over j < k of table[idx[i*k + j]]      for i < n
// The indices are read in order. The table entries are wherever the indices
// point, so nearly every step of the loop is a load from global memory that
// nothing predicts.
__global__ void gatherSumKernel(const float* table, const int* idx, float* out, int n, int k) {
    // TODO 1: one thread per output, on a 1D grid. The grid is whole blocks.
}

void launchGatherSum(const float* table, const int* idx, float* out, int n, int k, int threadsPerBlock) {
    // TODO 2: 1D blocks of threadsPerBlock threads, enough of them for n outputs.
    //         The tests use block sizes that do not divide n.
}

// ---------- the book's occupancy model (pp. 85-86) ----------
// How many blocks of threadsPerBlock threads, each thread holding
// regsPerThread registers, fit on one SM at the same time.
//   0   a block of this size cannot run at all
//  -1   unknown: none of the limits that would bound it is reported
//
// Three resources bound the answer: thread slots, block slots and registers.
// Work each of the book's cases by hand first (p. 85 for the slots, p. 86 for
// the registers), then write the arithmetic that gives the same answers.
//   - A block takes thread slots in whole warps. 48 threads take two warps'
//     worth where a warp is 32 (p. 75). The width is L.warpSize, not 32.
//   - Registers: threads times registers per thread, as on p. 86.
//   - A limit of 0 in SmLimits, or regsPerThread of 0, means "not known":
//     skip the rule that needs it.
//   - A block larger than maxThreadsPerBlock (when that is reported) cannot run.
int modelBlocksPerSm(const SmLimits& L, int threadsPerBlock, int regsPerThread) {
    // TODO 3
    return 0;
}

// Occupancy as the book defines it on p. 85, as a fraction between 0 and 1.
// -1 if the thread slots are not reported or the block count is unknown.
double modelOccupancy(const SmLimits& L, int threadsPerBlock, int regsPerThread) {
    // TODO 4
    return 0.0;
}
