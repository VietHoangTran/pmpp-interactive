// Lab 2.2 starter (CUDA; also built for HIP and for the CPU emulator).
// The vecAdd of lab 2.1 with a clock read between its parts. It measures, on
// your machine, the book's remark on p. 28 that outsourcing a computation this
// way can be very inefficient because of the copying back and forth.
//
// Not in the book: the book gives no timings, and this chapter does not say
// whether a kernel call waits for the grid. The README of this lab says what
// the vendors document.
#include <chrono>
#include <cstdio>

// The sidebar on p. 32, corrected (the page tests `error`, a name it never
// declared), as a macro. It reports and makes the enclosing function return 1.
#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            printf("%s in %s at line %d\n", cudaGetErrorString(err),     \
                   __FILE__, __LINE__);                                  \
            return 1;                                                    \
        }                                                                \
    } while (0)

// The book's block size (p. 38). Both functions below use it.
static const int threadsPerBlock = 256;

// Provided: a clock that never runs backwards, and the seconds between two reads.
typedef std::chrono::steady_clock::time_point Instant;
static Instant now() { return std::chrono::steady_clock::now(); }
static double seconds(Instant from, Instant to) {
    return std::chrono::duration<double>(to - from).count();
}

// Compute vector sum C = A + B.
// Each thread performs one pair-wise addition.
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Provided. Launches the kernel on arrays that are already on the device, and
// returns as soon as the launch is queued. The driver calls it between its own
// clock reads, with its own synchronise: that is the "resident" measurement.
void launchVecAdd(const float* A_d, const float* B_d, float* C_d, int n) {
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vecAddKernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, n);
}

// Below is the reference vecAdd of lab 2.1 under a new name (paste your own if
// you prefer it). It computes the right sums and measures nothing.
//
// Contract:
//   - phase_s[0..4] receive the seconds spent in: allocation, copies in,
//     kernel, copy back, free. Use now() and seconds() above.
//   - syncAfterLaunch != 0: the kernel phase ends after a checked
//     cudaDeviceSynchronize().
//     syncAfterLaunch == 0: the clock is read right after the launch has been
//     issued and checked, with no synchronise. This is the experiment: the
//     sums must still be right.
//   - returns 0, or non-zero when a call failed, as in lab 2.1.
int vecAddTimed(const float* A_h, const float* B_h, float* C_h, int n,
                int syncAfterLaunch, double* phase_s) {
    float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    size_t size = (size_t)n * sizeof(float);

    // TODO 1: decide where each clock read goes. Five phases need six reads.

    CHECK(cudaMalloc((void**)&A_d, size));
    CHECK(cudaMalloc((void**)&B_d, size));
    CHECK(cudaMalloc((void**)&C_d, size));

    CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vecAddKernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, n);
    CHECK(cudaGetLastError());
    // TODO 2: this synchronise becomes conditional on syncAfterLaunch.
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));

    // TODO 3: a clock read measures the host. For each phase, ask whether the
    //         device can still be busy with that phase's work, or with an
    //         earlier phase's, when the clock is read. The README names the
    //         two cases the vendors document. Add a synchronise where your
    //         answer is "yes" and you want the phase to hold its own work.

    // TODO 4: fill phase_s[0..4], in seconds.
    return 0;
}
