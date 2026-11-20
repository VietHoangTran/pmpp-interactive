// Lab 4.4 reference solution (CUDA; also built for HIP and for the CPU emulator).
// A kernel that spends its time waiting on memory, launched with a block size
// chosen at run time, and the book's occupancy arithmetic to set beside the
// measurement. Book: section 4.6 (pp. 82-85) for why waiting is tolerable,
// section 4.7 (pp. 85-87) for the arithmetic.
#include "limits.h"

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
//     so the block size is rounded up before anything is divided.
//   - Registers are counted for the padded warps too. The book's cases do not
//     decide this. NVIDIA's guide says registers are partitioned among the
//     warps and does not give the granularity, so this is the model's choice.
//   - A limit of 0, or regsPerThread of 0, means "not known" and its rule is skipped.
int modelBlocksPerSm(const SmLimits& L, int threadsPerBlock, int regsPerThread) {
    if (L.maxThreadsPerBlock > 0 && threadsPerBlock > L.maxThreadsPerBlock) return 0;

    int warps = (threadsPerBlock + L.warpSize - 1)/L.warpSize;
    int slots = warps*L.warpSize;              // thread slots one block takes

    int blocks = -1;
    if (L.maxThreadsPerMultiProcessor > 0) {   // thread slots: whole blocks only, a remainder stays empty
        blocks = L.maxThreadsPerMultiProcessor/slots;
    }
    if (L.maxBlocksPerMultiProcessor > 0) {    // block slots
        if (blocks < 0 || L.maxBlocksPerMultiProcessor < blocks) blocks = L.maxBlocksPerMultiProcessor;
    }
    if (L.regsPerMultiprocessor > 0 && regsPerThread > 0) {   // registers
        int byRegisters = L.regsPerMultiprocessor/(slots*regsPerThread);
        if (blocks < 0 || byRegisters < blocks) blocks = byRegisters;
    }
    return blocks;
}

// Occupancy as the book defines it on p. 85: warps on the SM over the warps it
// can hold, as a fraction between 0 and 1. -1 if the thread slots are not
// reported or the block count is unknown.
double modelOccupancy(const SmLimits& L, int threadsPerBlock, int regsPerThread) {
    int blocks = modelBlocksPerSm(L, threadsPerBlock, regsPerThread);
    if (blocks < 0 || L.maxThreadsPerMultiProcessor <= 0) return -1.0;
    int warps = (threadsPerBlock + L.warpSize - 1)/L.warpSize;
    int warpSlots = L.maxThreadsPerMultiProcessor/L.warpSize;
    return (double)(blocks*warps)/warpSlots;
}
