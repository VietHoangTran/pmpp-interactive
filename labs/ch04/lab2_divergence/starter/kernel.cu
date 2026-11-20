// Lab 4.2 starter (CUDA; also built for HIP and for the CPU emulator).
// Three kernels for measuring what control divergence costs.
// Book: section 4.5, pp. 79-82, Figs. 4.9 and 4.10.
//
// paths.h (beside this file, provided) gives you
//     float pathA(float x, int iters);   float stepA(float x);
//     float pathB(float x, int iters);   float stepB(float x);
// pathA and pathB cost the same: `iters` steps each. Read its comment for why
// the two paths are loops.
//
// Every array has n elements and thread i of the grid owns element i. The grid
// is whole blocks, so it can hold more than n threads.
#include "paths.h"

// Fig. 4.9's shape: out[i] is pathA(in[i], iters) where which[i] == 0, and
// pathB(in[i], iters) everywhere else. Each thread runs ONE of the two.
__global__ void branchKernel(const float* in, const int* which, float* out, int n, int iters) {
    // TODO 1
}

// The same out[i] as branchKernel, written so that no thread's control flow
// depends on which[i].
__global__ void selectKernel(const float* in, const int* which, float* out, int n, int iters) {
    // TODO 2
}

// Fig. 4.10's shape: out[i] is in[i] after trips[i] steps of path A (stepA).
// trips[i] can be 0.
__global__ void loopKernel(const float* in, const int* trips, float* out, int n) {
    // TODO 3
}

// TODO 4: the three launches. One thread per element, 1D blocks of
// threadsPerBlock threads. The driver passes multiples of the device's warp
// width and builds its patterns on the assumption that warp w of the grid then
// holds elements w*width up to (w+1)*width.
void launchBranch(const float* in, const int* which, float* out, int n, int iters, int threadsPerBlock) {
}

void launchSelect(const float* in, const int* which, float* out, int n, int iters, int threadsPerBlock) {
}

void launchLoop(const float* in, const int* trips, float* out, int n, int threadsPerBlock) {
}
