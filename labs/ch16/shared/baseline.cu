// Provided baseline for lab 16.1 (CUDA; also built for HIP and for the CPU emulator).
// One kernel launch per anti-diagonal of the table, one thread per cell, every input
// read from global memory: the "basic wavefront parallelization" of book section 16.6,
// p. 384, Fig. 16.7(a). The book leaves its code as an exercise; this version is ours.
//
// Anti-diagonal s holds the cells with r + q == s, for s = 2 .. 2 * L_seq. Its cells
// depend only on anti-diagonals s - 1 and s - 2, which earlier launches finished.
#include <algorithm>

#include "../lab1_wavefront/sw.h"

// Threads per block. The CPU emulators start one OS thread per thread of a block,
// cell or no cell, so they get small blocks; at the small tables they run, that also
// puts several blocks on one anti-diagonal. Their timings mean nothing either way.
#if defined(PMPP_BACKEND_EMU) || defined(PMPP_BACKEND_RACE)
#define PMPP_SW_BASELINE_BLOCK 16
#else
#define PMPP_SW_BASELINE_BLOCK 128
#endif

// Thread t of the launch computes row firstRow + t of anti-diagonal s.
__global__ void swBaselineKernel(int* H, const int* a, const int* b, int L, int s, int firstRow, int count) {
    const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (t < count) {
        const int r = firstRow + t;
        const int q = s - r;
        const int n  = H[(r - 1) * L + q];
        const int w  = H[r * L + (q - 1)];
        const int nw = H[(r - 1) * L + (q - 1)];
        const int score = (a[r - 1] == b[q - 1]) ? SW_MATCH : SW_MISMATCH;
        int h = 0;
        if (h < nw + score) h = nw + score;
        if (h < w + SW_GAP) h = w + SW_GAP;
        if (h < n + SW_GAP) h = n + SW_GAP;
        H[r * L + q] = h;
    }
}

// H: (Lseq + 1) x (Lseq + 1) ints with row 0 and column 0 already zero.
// 2 * Lseq - 1 launches, in one stream, so each sees the previous one's writes.
void launchSwBaseline(int* H, const int* a, const int* b, int Lseq) {
    const int L = Lseq + 1;
    for (int s = 2; s <= 2 * Lseq; s++) {
        const int firstRow = std::max(1, s - Lseq);
        const int lastRow = std::min(Lseq, s - 1);
        const int count = lastRow - firstRow + 1;
        const int blocks = (count + PMPP_SW_BASELINE_BLOCK - 1) / PMPP_SW_BASELINE_BLOCK;
        swBaselineKernel<<<blocks, PMPP_SW_BASELINE_BLOCK>>>(H, a, b, L, s, firstRow, count);
    }
}
