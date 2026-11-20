// Lab 4.4 reference solution (CUDA; also built for HIP and for the CPU emulator).
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
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0;
        for (int j = 0; j < k; ++j) {
            sum += table[idx[i*k + j]];
        }
        out[i] = sum;
    }
}

void launchGatherSum(const float* table, const int* idx, float* out, int n, int k, int threadsPerBlock) {
    dim3 dimBlock(threadsPerBlock, 1, 1);
    dim3 dimGrid((n + threadsPerBlock - 1)/threadsPerBlock, 1, 1);
    gatherSumKernel<<<dimGrid, dimBlock>>>(table, idx, out, n, k);
}

// ---------- the book's occupancy model (pp. 85-86) ----------
// How many blocks of threadsPerBlock threads, each thread holding
// regsPerThread registers, fit on one SM at the same time.
//   0   a block of this size cannot run at all
//  -1   unknown: none of the limits that would bound it is reported
//
// Where this goes beyond the page:
//   - The book's block sizes are all whole warps. A block that is not takes
//     whole warps of thread slots all the same, the last one padded (p. 75),
//     so the block size is rounded up before the thread slots are divided.
//   - Registers are threads times registers per thread, the book's product
//     (p. 86). The padding lanes run no thread and are charged none. NVIDIA's
//     guide says registers are partitioned among the warps and does not give
//     the granularity, so for a real kernel the vendor's occupancy function is
//     the authority and this is a model.
//   - A limit of 0, or regsPerThread of 0, means "not known" and its rule is skipped.
int modelBlocksPerSm(const SmLimits& L, int threadsPerBlock, int regsPerThread) {
    if (threadsPerBlock < 1) return 0;
    if (L.maxThreadsPerBlock > 0 && threadsPerBlock > L.maxThreadsPerBlock) return 0;

    int width = L.warpSize > 0 ? L.warpSize : 1;
    int warps = (threadsPerBlock + width - 1)/width;
    int slots = warps*width;                   // thread slots one block takes

    int blocks = -1;
    if (L.maxThreadsPerMultiProcessor > 0) {   // thread slots: whole blocks only, a remainder stays empty
        blocks = L.maxThreadsPerMultiProcessor/slots;
    }
    if (L.maxBlocksPerMultiProcessor > 0) {    // block slots
        if (blocks < 0 || L.maxBlocksPerMultiProcessor < blocks) blocks = L.maxBlocksPerMultiProcessor;
    }
    if (L.regsPerMultiprocessor > 0 && regsPerThread > 0) {   // registers
        int byRegisters = L.regsPerMultiprocessor/(threadsPerBlock*regsPerThread);
        if (blocks < 0 || byRegisters < blocks) blocks = byRegisters;
    }
    return blocks;
}

// Occupancy as the book defines it on p. 85: warps on the SM over the warps it
// can hold, as a fraction between 0 and 1. -1 if the thread slots are not
// reported or the block count is unknown.
// (The summary on p. 90 says threads instead of warps. The two differ only
// when a block is not whole warps.)
double modelOccupancy(const SmLimits& L, int threadsPerBlock, int regsPerThread) {
    int blocks = modelBlocksPerSm(L, threadsPerBlock, regsPerThread);
    if (blocks < 0 || L.maxThreadsPerMultiProcessor <= 0) return -1.0;
    int width = L.warpSize > 0 ? L.warpSize : 1;
    int warps = (threadsPerBlock + width - 1)/width;
    int warpSlots = L.maxThreadsPerMultiProcessor/width;
    return (double)(blocks*warps)/warpSlots;
}
