// Lab 4.2 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Three kernels for measuring what control divergence costs.
// Book: section 4.5, pp. 79-82. The book shows the two shapes as fragments
// inside Figs. 4.9 and 4.10, not as kernels; these are ours.
#include "paths.h"

// Fig. 4.9's shape (p. 80): an if-else. The figure branches on threadIdx.x < 24;
// here the data decides, so the driver can lay out any pattern it likes.
// which[i] == 0 sends thread i down path A, anything else down path B.
__global__ void branchKernel(const float* in, const int* which, float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (which[i] == 0) {
            out[i] = pathA(in[i], iters);
        } else {
            out[i] = pathB(in[i], iters);
        }
    }
}

// The same result with nothing to diverge on: every thread runs both paths and
// keeps one value. Twice the arithmetic, one control-flow path.
__global__ void selectKernel(const float* in, const int* which, float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = pathA(in[i], iters);
        float b = pathB(in[i], iters);
        out[i] = (which[i] == 0) ? a : b;
    }
}

// Fig. 4.10's shape (p. 81): a loop whose trip count is per-thread data. The
// figure reads it from a[threadIdx.x]; here it is trips[i] with i the global
// index, so every block has its own counts. The loop body is one step of path A.
__global__ void loopKernel(const float* in, const int* trips, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        int N = trips[i];
        for (int k = 0; k < N; ++k) {
            x = stepA(x);
        }
        out[i] = x;
    }
}

// One thread per element, in 1D blocks of the size the driver asks for. The
// driver passes multiples of the device's warp width, so that warp w of the
// grid holds elements w*width up to (w+1)*width.
void launchBranch(const float* in, const int* which, float* out, int n, int iters, int threadsPerBlock) {
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    branchKernel<<<blocks, threadsPerBlock>>>(in, which, out, n, iters);
}

void launchSelect(const float* in, const int* which, float* out, int n, int iters, int threadsPerBlock) {
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    selectKernel<<<blocks, threadsPerBlock>>>(in, which, out, n, iters);
}

void launchLoop(const float* in, const int* trips, float* out, int n, int threadsPerBlock) {
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    loopKernel<<<blocks, threadsPerBlock>>>(in, trips, out, n);
}
